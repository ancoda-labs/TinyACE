#!/usr/bin/env python3
"""Compare two experiment arms with confidence intervals and a paired test.

Point estimates alone are not decidable at these sample sizes: at n=50 the
95% CI on an accuracy near 0.74 is roughly +/-12 percentage points, so a
"+4%" improvement is two questions and is indistinguishable from noise. Run
this before putting any delta in a table.

Each arm is compared against the reference `reporting.reference_for()` names
for it -- ACE arms against `cot_control`, ablations against `tinyace_wm_256` --
because comparing an ablation to `baseline` measures ACE *plus* the ablation
rather than the ablated component. All comparisons printed together form one
family and are Holm-corrected; the adjusted p-value is the one that decides.

Usage:
    # Two prediction files
    python -m scripts.compare_arms \
        results/phi_3_mini/sciq_test/baseline/cuda/predictions.jsonl \
        results/phi_3_mini/sciq_test/tinyace_wm_256/cuda/predictions.jsonl

    # Every arm against its registered reference, under one results root
    python -m scripts.compare_arms --results-root results

    # Override the registry to ask one specific question
    python -m scripts.compare_arms --results-root results --reference baseline
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from edge_slm_ace.eval.stats import compare_arms, format_comparison, holm_bonferroni
from edge_slm_ace.reporting import arm_label, reference_for


def load_predictions(path: Path) -> List[Dict]:
    """Load a predictions.jsonl file into a list of per-item result dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_arms(results_root: Path) -> Dict[str, Path]:
    """
    Find every predictions.jsonl under a results root.

    Returns:
        Mapping of arm label -> path. The label is the directory path
        relative to the root, which encodes model/task/mode/device.
    """
    arms = {}
    for path in sorted(results_root.rglob("predictions.jsonl")):
        arms[str(path.parent.relative_to(results_root))] = path
    return arms


def arm_key_of(label: str) -> str:
    """
    The arm segment of a results path, which is {model}/{task}/{arm}/{device}.

    Args:
        label: Directory path relative to the results root.

    Returns:
        The arm key, as registered in `reporting.schema`.
    """
    parts = Path(label).parts
    return parts[-2] if len(parts) >= 2 else label


def cell_of(label: str) -> str:
    """The {model}/{task} prefix a comparison must not cross."""
    return str(Path(label).parent.parent)


def pair_with_registered_references(
    arms: Dict[str, Path],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Pair each arm with the reference `reporting.reference_for()` names for it.

    Ablations belong against full TinyACE and ACE arms against the CoT control;
    comparing either to `baseline` measures the wrong thing. That mapping was
    already defined and tested in the schema, and this script imported it
    without ever calling it -- every arm was compared against a label matching
    the literal string "baseline", which is the defect the registry exists to
    prevent.

    Args:
        arms: Mapping of results-relative label -> predictions path.

    Returns:
        Tuple of (pairs, missing):
          pairs   (reference_label, arm_label), reference first.
          missing (arm_label, expected_reference_key) for arms whose reference
                  was not run, which are skipped rather than silently
                  re-pointed at another arm.
    """
    by_cell: Dict[str, Dict[str, str]] = {}
    for label in arms:
        by_cell.setdefault(cell_of(label), {})[arm_key_of(label)] = label

    pairs: List[Tuple[str, str]] = []
    missing: List[Tuple[str, str]] = []
    for label in sorted(arms):
        key = arm_key_of(label)
        reference_key = reference_for(key)
        if reference_key == key:
            continue  # the arm is its own reference; nothing to compare
        reference_label = by_cell[cell_of(label)].get(reference_key)
        if reference_label is None:
            missing.append((label, reference_key))
            continue
        pairs.append((reference_label, label))
    return pairs, missing


def pair_with_explicit_reference(
    arms: Dict[str, Path],
    reference: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Pair every arm against a caller-named reference, within the same cell.

    An explicit override of the registry. Use it to ask a specific question,
    not as the default -- see `pair_with_registered_references`.
    """
    references = [label for label in arms if reference in label]
    pairs = [
        (reference_label, label)
        for reference_label in references
        for label in sorted(arms)
        if label != reference_label and cell_of(label) == cell_of(reference_label)
    ]
    return pairs, []


def _pick_metric(rows: List[Dict], requested: Optional[str]) -> str:
    """Choose the correctness field to compare on."""
    if requested:
        return requested
    # Prefer OMA on MCQ tasks; exact match is ~0 for any verbose model.
    if any(r.get("oma_correct") is not None for r in rows):
        return "oma_correct"
    return "is_correct"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare experiment arms with CIs and an exact McNemar test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "arms", nargs="*", type=str, help="Two predictions.jsonl paths to compare directly"
    )
    parser.add_argument(
        "--results-root", type=str, default=None, help="Compare every arm found under this root"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help=(
            "Substring identifying one reference arm, overriding the registry. "
            "By default each arm is compared against the reference "
            "reporting.reference_for() names for it: ACE arms against "
            "cot_control, ablations against tinyace_wm_256."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=["oma_correct", "is_correct"],
        help="Correctness field (default: oma_correct when present)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        choices=[0.90, 0.95, 0.99],
        help="Confidence level",
    )
    parser.add_argument(
        "--json-out", type=str, default=None, help="Write the comparisons to a JSON file"
    )

    args = parser.parse_args()

    comparisons = []

    if len(args.arms) == 2:
        rows_a = load_predictions(Path(args.arms[0]))
        rows_b = load_predictions(Path(args.arms[1]))
        comparisons.append(
            compare_arms(
                rows_a,
                rows_b,
                name_a=Path(args.arms[0]).parent.name,
                name_b=Path(args.arms[1]).parent.name,
                metric=_pick_metric(rows_a + rows_b, args.metric),
                confidence=args.confidence,
            )
        )

    elif args.results_root:
        root = Path(args.results_root)
        if not root.exists():
            print(f"Error: results root not found: {root}", file=sys.stderr)
            return 1

        arms = discover_arms(root)
        if not arms:
            print(f"Error: no predictions.jsonl found under {root}", file=sys.stderr)
            return 1

        if args.reference:
            pairs, missing = pair_with_explicit_reference(arms, args.reference)
            if not pairs:
                print(
                    f"Error: no arm matching '{args.reference}'. Found:\n  "
                    + "\n  ".join(sorted(arms)),
                    file=sys.stderr,
                )
                return 1
        else:
            pairs, missing = pair_with_registered_references(arms)

        for reference_label, arm_label_path in missing:
            print(
                f"Note: skipping {arm_label_path} -- its reference arm "
                f"'{reference_label}' was not run in this cell.",
                file=sys.stderr,
            )

        for reference_label, label in pairs:
            reference_rows = load_predictions(arms[reference_label])
            rows = load_predictions(arms[label])
            comparisons.append(
                compare_arms(
                    reference_rows,
                    rows,
                    name_a=arm_label(arm_key_of(reference_label)),
                    name_b=arm_label(arm_key_of(label)),
                    metric=_pick_metric(reference_rows + rows, args.metric),
                    confidence=args.confidence,
                )
            )
    else:
        parser.error("Provide two predictions.jsonl paths, or --results-root")

    # Every comparison printed together is one family of tests. Reporting each
    # p<0.05 on its own across a 10-arm sweep gives roughly a 40% chance of at
    # least one false positive, so the adjusted value is what decides.
    adjusted = holm_bonferroni([c["mcnemar"]["p_value"] for c in comparisons])
    for comparison, p_adjusted in zip(comparisons, adjusted):
        comparison["mcnemar"]["p_adjusted"] = p_adjusted
        comparison["mcnemar"]["significant_05_adjusted"] = p_adjusted < 0.05

    for comparison in comparisons:
        print()
        print("=" * 72)
        print(format_comparison(comparison))
        test = comparison["mcnemar"]
        if len(comparisons) > 1:
            verdict = "survives" if test["significant_05_adjusted"] else "does not survive"
            print(
                f"  -> p={test['p_value']:.4f} {verdict} Holm correction "
                f"across {len(comparisons)} tests (p_adj={test['p_adjusted']:.4f})"
            )

    raw = [c for c in comparisons if c["mcnemar"]["significant_05"]]
    survivors = [c for c in comparisons if c["mcnemar"]["significant_05_adjusted"]]
    print()
    print("=" * 72)
    print(
        f"{len(survivors)} of {len(comparisons)} comparisons survive Holm "
        f"correction at p<0.05 ({len(raw)} before correction, "
        f"family size {len(comparisons)})."
    )
    if comparisons and not survivors:
        print("Report these as 'no detectable difference', not as an ordering.")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(comparisons, f, indent=2, default=str)
        print(f"Wrote comparisons to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
