#!/usr/bin/env python3
"""Fetch / verify Global-MMLU (Nepali & English) for Week 1 screening.

Usage:
    python -m scripts.fetch_globalmmlu
    python -m scripts.fetch_globalmmlu --check
"""

import argparse
import sys

from edge_slm_ace.data.globalmmlu import (
    EXPECTED_SIZES,
    GLOBAL_MMLU_LANGUAGES,
    HF_DATASET,
    globalmmlu_path,
    load_globalmmlu,
    verify_parallelism,
    verify_schema_and_size,
)


def check() -> int:
    problems = []
    print(f"Dataset: {HF_DATASET}")
    for language in GLOBAL_MMLU_LANGUAGES:
        for split, expected in EXPECTED_SIZES.items():
            try:
                summary = verify_schema_and_size(language, split)
                print(
                    f"  {language}/{split}: {summary['n']} rows "
                    f"(expected {expected}), schema ok"
                )
            except Exception as exc:
                problems.append(f"{language}/{split}: {exc}")
                print(f"  FAIL {language}/{split}: {exc}", file=sys.stderr)

    try:
        parallel = verify_parallelism("test")
        print(f"  parallel en/ne test: {parallel['n']} shared sample_ids, gold agrees")
    except Exception as exc:
        problems.append(str(exc))
        print(f"  FAIL parallelism: {exc}", file=sys.stderr)

    # Normalised load must stay parallel by id; gold *indices* may differ after
    # per-language MD5(passage+question) option shuffle.
    try:
        en = load_globalmmlu("en", split="test")
        ne = load_globalmmlu("ne", split="test")
        if [e["id"] for e in en] != [e["id"] for e in ne]:
            problems.append("normalised en/ne id order diverged")
        if any(e["answer"] != e["options"][e["gold_option_idx"]] for e in en + ne):
            problems.append("normalised gold_option_idx inconsistent with options")
        print(f"  normalised load: {len(en)} items, ids parallel across languages")
    except Exception as exc:
        problems.append(str(exc))
        print(f"  FAIL normalised load: {exc}", file=sys.stderr)

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


def cache_jsonl() -> None:
    """Optional: write Hub rows to data/tasks/global_mmlu/ for offline reuse."""
    import json as _json

    from edge_slm_ace.data.globalmmlu import _load_split_from_hub

    for language in GLOBAL_MMLU_LANGUAGES:
        for split in EXPECTED_SIZES:
            dest = globalmmlu_path(language, split)
            dest.parent.mkdir(parents=True, exist_ok=True)
            rows = _load_split_from_hub(language, split)
            with open(dest, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(_json.dumps(dict(row), ensure_ascii=False) + "\n")
            print(f"  cached {language}/{split} -> {dest} ({len(rows)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify only; do not cache")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Also write jsonl under data/tasks/global_mmlu/",
    )
    args = parser.parse_args()

    if args.cache and not args.check:
        print("Caching:")
        cache_jsonl()

    print("Checking:")
    return check()


if __name__ == "__main__":
    sys.exit(main())
