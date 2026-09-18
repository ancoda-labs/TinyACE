#!/usr/bin/env python3
"""Regenerate the frozen split manifests in data/splits/.

This is not part of any normal workflow. The manifests are committed and every
run loads them from disk; rerunning this is how you would *change* the study's
splits, which invalidates the pre-registration and every result already
collected against them.

It is wired to `make splits-REGENERATE` rather than `make splits` for that
reason, and refuses to overwrite an existing manifest without --force.

Usage:
    python -m scripts.freeze_splits --check    # verify what is committed
    python -m scripts.freeze_splits --force    # rewrite the manifests
"""

import argparse
import sys
from typing import List

from edge_slm_ace.data.belebele import (
    assert_zero_passage_overlap,
    load_belebele,
    passage_disjoint_split,
)
from edge_slm_ace.data.globalmmlu import load_globalmmlu
from edge_slm_ace.data.splits import (
    ADAPT_SUBSAMPLE_SIZE,
    BELEBELE_SPLIT,
    EVAL_SUBSAMPLE_SIZE,
    GLOBAL_MMLU_ADAPT,
    GLOBAL_MMLU_EVAL,
    MANIFESTS,
    SUBSAMPLE_SEED,
    ids_checksum,
    load_manifest,
    manifest_path,
    stratified_sample,
    write_manifest,
)
from edge_slm_ace.utils.config import ADAPTATION_SIZE
from edge_slm_ace.utils.repro import DEFAULT_SEED


def build_global_mmlu() -> List[str]:
    """
    Freeze the Global-MMLU evaluation subsample and the adaptation split.

    The eval subsample is drawn first and the adaptation split from what it
    leaves, so disjointness is a property of the construction rather than a
    check bolted on afterwards.
    """
    examples = load_globalmmlu("ne", "test")
    subject_of = {e["id"]: str(e["subject"]) for e in examples}

    eval_ids = stratified_sample(examples, EVAL_SUBSAMPLE_SIZE, SUBSAMPLE_SEED)
    adapt_ids = stratified_sample(examples, ADAPT_SUBSAMPLE_SIZE, SUBSAMPLE_SEED, exclude=eval_ids)

    overlap = set(eval_ids) & set(adapt_ids)
    if overlap:  # pragma: no cover - construction makes this unreachable
        raise AssertionError(f"{len(overlap)} ids are in both splits")

    common = {
        "task": "global_mmlu",
        "split": "test",
        "seed": SUBSAMPLE_SEED,
        "stratified_by": "subject",
        # sample_ids are identical across language configs, so English
        # evaluates the same ids. That is what keeps McNemar paired.
        "languages": ["en", "ne"],
        "corpus_size": len(examples),
    }

    # The ids are blake2s digests, so a subject cannot be read back off one.
    # Recording the histogram here keeps the stratification auditable without
    # reloading 14,042 rows, and makes a dropped subject visible in the diff.
    def histogram(ids):
        counts = {}
        for item in ids:
            subject = subject_of[item]
            counts[subject] = counts.get(subject, 0) + 1
        return dict(sorted(counts.items()))

    write_manifest(
        GLOBAL_MMLU_EVAL,
        {
            **common,
            "role": "evaluation",
            "size": len(eval_ids),
            "subjects": histogram(eval_ids),
            "item_ids": eval_ids,
        },
    )
    write_manifest(
        GLOBAL_MMLU_ADAPT,
        {
            **common,
            "role": "adaptation",
            "size": len(adapt_ids),
            "disjoint_from": GLOBAL_MMLU_EVAL,
            "subjects": histogram(adapt_ids),
            "item_ids": adapt_ids,
        },
    )
    return [
        f"  {GLOBAL_MMLU_EVAL}: {len(eval_ids)} of {len(examples)} items",
        f"  {GLOBAL_MMLU_ADAPT}: {len(adapt_ids)} items, disjoint from the eval subsample",
    ]


def build_belebele() -> List[str]:
    """Freeze the Belebele passage-disjoint split exactly as the loader builds it."""
    examples = load_belebele("ne")
    adapt_ids, eval_ids = passage_disjoint_split(examples, ADAPTATION_SIZE, DEFAULT_SEED)
    assert_zero_passage_overlap(adapt_ids, eval_ids, examples)

    by_id = {e["id"]: e for e in examples}
    write_manifest(
        BELEBELE_SPLIT,
        {
            "task": "belebele",
            "split": "test",
            "seed": DEFAULT_SEED,
            "unit": "passage",
            "adaptation_size": ADAPTATION_SIZE,
            "languages": ["en", "ne"],
            "role": "evaluation",
            "size": len(eval_ids),
            "adaptation_item_ids": sorted(adapt_ids),
            "adaptation_ids_sha256": ids_checksum(adapt_ids),
            "item_ids": eval_ids,
        },
    )
    return [
        f"  {BELEBELE_SPLIT}: {len(adapt_ids)} adaptation "
        f"({len({by_id[i]['passage_id'] for i in adapt_ids})} passages) / "
        f"{len(eval_ids)} evaluation "
        f"({len({by_id[i]['passage_id'] for i in eval_ids})} passages)"
    ]


def check() -> int:
    """
    Verify every committed manifest loads, verifies and stays disjoint.

    Returns:
        Process exit code.
    """
    problems = []
    for name in MANIFESTS:
        try:
            body = load_manifest(name)
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"{name}: {exc}")
            continue
        print(f"  {name}: {len(body['item_ids'])} items, ids_sha256 {body['ids_sha256'][:16]}...")

    if not problems:
        shared = set(manifest_or_empty(GLOBAL_MMLU_EVAL)) & set(
            manifest_or_empty(GLOBAL_MMLU_ADAPT)
        )
        if shared:
            problems.append(f"{len(shared)} ids are in both Global-MMLU splits")

        belebele = load_manifest(BELEBELE_SPLIT)
        both = set(belebele["adaptation_item_ids"]) & set(belebele["item_ids"])
        if both:
            problems.append(f"{len(both)} ids are on both sides of the Belebele split")

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


def manifest_or_empty(name: str) -> List[str]:
    try:
        return load_manifest(name)["item_ids"]
    except (FileNotFoundError, ValueError):
        return []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="Verify, do not write")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite manifests that already exist. Invalidates the pre-registration.",
    )
    args = parser.parse_args(argv)

    if args.check:
        print("Checking frozen splits:")
        return check()

    existing = [n for n in MANIFESTS if manifest_path(n).exists()]
    if existing and not args.force:
        print(
            f"Refusing to overwrite committed split manifests: {', '.join(existing)}.\n"
            "Every result already collected is keyed to these ids. Pass --force "
            "if you really mean to change the study's splits.",
            file=sys.stderr,
        )
        return 1

    print("Rewriting frozen splits. This invalidates the pre-registration.\n")
    lines = build_global_mmlu() + build_belebele()
    print("\n".join(lines))
    print("\nChecking what was written:")
    return check()


if __name__ == "__main__":
    sys.exit(main())
