#!/usr/bin/env python3
"""Check committed Belebele rows against the pinned Hub revision.

Adaptation reads ``data/tasks/belebele/*.jsonl``. Evaluation loads
``facebook/belebele`` from the Hub at the revision in the task YAML.
``per_item_correctness`` only checks that the ids agree, so a text change
at a new Hub commit would score a different passage than the one the
playbook was adapted on, and the accuracy would still look clean.

This compares the passage, the question, the four options and the gold
numeral. It does not compare ``dialect`` or ``ds``: those are file metadata,
not text the model is shown.

Usage:
    python -m scripts.verify_task_data
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Sequence, Tuple

from edge_slm_ace.data.belebele import (
    BELEBELE_LANGUAGES,
    BELEBELE_REVISION,
    HF_DATASET,
    belebele_path,
)

# The fields that become the prompt or the gold. correct_answer_num is a
# numeral; the jsonl stores it as a string and datasets may return an int.
_TEXT_FIELDS = (
    "flores_passage",
    "question",
    "mc_answer1",
    "mc_answer2",
    "mc_answer3",
    "mc_answer4",
    "correct_answer_num",
)


def _key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row["link"]), str(row["question_number"]))


def _text(row: Dict[str, Any], field: str) -> str:
    """
    The comparable text of one field.

    ``correct_answer_num`` is compared as a numeral, not a Python type: the
    committed jsonl quotes it and the datasets loader may not.
    """
    value = row[field]
    if field == "correct_answer_num":
        return str(value).strip()
    return value if isinstance(value, str) else str(value)


def content_mismatches(
    local_rows: Sequence[Dict[str, Any]],
    hub_rows: Sequence[Dict[str, Any]],
) -> List[str]:
    """
    Differences between a committed jsonl and the Hub rows it is supposed to be.

    Args:
        local_rows: Rows from ``data/tasks/belebele/*.jsonl``.
        hub_rows: The same config's ``test`` split at ``BELEBELE_REVISION``.

    Returns:
        One string per problem. Empty when every local row's text matches
        the Hub row with the same ``(link, question_number)``, and the Hub
        has no extra rows.
    """
    problems: List[str] = []
    hub_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in hub_rows:
        key = _key(row)
        if key in hub_by_key:
            problems.append(f"hub duplicate link={key[0]!r} question_number={key[1]}")
        hub_by_key[key] = row

    seen = set()
    for row in local_rows:
        key = _key(row)
        if key in seen:
            problems.append(f"committed duplicate link={key[0]!r} question_number={key[1]}")
            continue
        seen.add(key)
        hub = hub_by_key.get(key)
        if hub is None:
            problems.append(f"missing on hub: link={key[0]!r} question_number={key[1]}")
            continue
        for field in _TEXT_FIELDS:
            if field not in row or field not in hub:
                problems.append(f"link={key[0]!r} question_number={key[1]} missing field {field}")
                continue
            if _text(row, field) != _text(hub, field):
                problems.append(f"link={key[0]!r} question_number={key[1]} {field} differs")

    for key in sorted(set(hub_by_key) - seen):
        problems.append(
            f"hub row not in the committed file: link={key[0]!r} question_number={key[1]}"
        )
    return problems


def _load_local(language: str) -> List[Dict[str, Any]]:
    path = belebele_path(language)
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_hub(config: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(HF_DATASET, config, split="test", revision=BELEBELE_REVISION)
    return [dict(row) for row in dataset]


def check() -> List[str]:
    """Compare every committed Belebele language file to the pinned Hub split."""
    problems: List[str] = []
    for language, config in BELEBELE_LANGUAGES.items():
        path = belebele_path(language)
        if not path.exists():
            problems.append(f"{language}: missing {path}")
            continue
        found = content_mismatches(_load_local(language), _load_hub(config))
        problems.extend(f"{language}: {problem}" for problem in found)
        if not found:
            print(f"  {language}: {path.name} matches {HF_DATASET}@{BELEBELE_REVISION[:12]}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(f"Checking data/tasks/belebele against {HF_DATASET}@{BELEBELE_REVISION}")
    problems = check()
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if problems:
        return 1
    print("  committed rows match the pinned Hub text.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
