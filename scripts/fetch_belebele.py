#!/usr/bin/env python3
"""Fetch the Belebele language files this project evaluates on.

Belebele ships 122 language variants; we use English and Nepali (Devanagari).
The files are small (~4 MB together) and are committed, so this script exists to
make the provenance reproducible rather than because it has to be re-run.

Usage:
    python -m scripts.fetch_belebele
    python -m scripts.fetch_belebele --check   # verify what is on disk
"""

import argparse
import json
import sys
import urllib.request
from collections import Counter

from edge_slm_ace.data.belebele import (
    BELEBELE_LANGUAGES,
    belebele_path,
    item_id,
    load_belebele,
)

SOURCE = "https://huggingface.co/datasets/facebook/belebele/resolve/main/data/{flores}.jsonl"
EXPECTED_ROWS = 900


def fetch(language: str) -> int:
    """Download one language file. Returns the row count."""
    flores = BELEBELE_LANGUAGES[language]
    dest = belebele_path(language)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SOURCE.format(flores=flores), dest)
    with open(dest, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def check() -> int:
    """
    Verify the files on disk are the parallel corpus the protocol assumes.

    Returns:
        Process exit code.
    """
    problems = []
    loaded = {}
    for language in BELEBELE_LANGUAGES:
        path = belebele_path(language)
        if not path.exists():
            problems.append(f"{language}: missing ({path})")
            continue
        rows = load_belebele(language)
        loaded[language] = rows
        if len(rows) != EXPECTED_ROWS:
            problems.append(f"{language}: {len(rows)} rows, expected {EXPECTED_ROWS}")

        with open(path, encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
        gold = Counter(r["correct_answer_num"] for r in raw)
        print(f"  {language:3} {len(rows):4} items   gold position {dict(sorted(gold.items()))}")

    if len(loaded) == len(BELEBELE_LANGUAGES):
        ids = [[e["id"] for e in rows] for rows in loaded.values()]
        if ids[0] != ids[1]:
            problems.append("languages are not parallel: id sets differ after sorting")

        # Raw gold (pre-shuffle) must agree; MD5(passage+question) may place the
        # gold letter in different slots after the per-language shuffle.
        raw_gold = {}
        raw_orders = []
        for language in loaded:
            with open(belebele_path(language), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            raw_gold[language] = {item_id(r): r["correct_answer_num"] for r in rows}
            raw_orders.append([item_id(r) for r in rows])
        if raw_gold["en"] != raw_gold["ne"]:
            problems.append("languages disagree on the raw correct_answer_num")
        if raw_orders[0] == raw_orders[1]:
            print(
                "  note: the files are currently in the same row order. Keep using "
                "harness_indices() anyway -- that is not guaranteed upstream."
            )

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if problems:
        return 1

    print("  parallel across languages, gold agrees on every item.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify, do not download")
    args = parser.parse_args()

    if not args.check:
        for language in BELEBELE_LANGUAGES:
            rows = fetch(language)
            print(f"  {language}: {rows} rows -> {belebele_path(language)}")

    print("Checking:")
    return check()


if __name__ == "__main__":
    sys.exit(main())
