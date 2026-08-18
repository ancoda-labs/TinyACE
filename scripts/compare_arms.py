#!/usr/bin/env python3
"""Compare two experiment arms with confidence intervals and a paired test.

Point estimates alone are not decidable at these sample sizes: at n=50 the
95% CI on an accuracy near 0.74 is roughly +/-12 percentage points, so a
"+4%" improvement is two questions and is indistinguishable from noise. Run
this before putting any delta in a table.

Usage:
    # Two prediction files
    python -m scripts.compare_arms \
        results/phi_3_mini/sciq_test/baseline/cuda/predictions.jsonl \
        results/phi_3_mini/sciq_test/tinyace_wm_256/cuda/predictions.jsonl

    # Every arm against a chosen reference, under one results root
    python -m scripts.compare_arms --results-root results --reference baseline

    # Ablations belong against full TinyACE, not against baseline
    python -m scripts.compare_arms --results-root results --reference tinyace_wm_256
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from edge_slm_ace.utils.stats import compare_arms, format_comparison


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
    parser.add_argument("arms", nargs="*", type=str,
                        help="Two predictions.jsonl paths to compare directly")
    parser.add_argument("--results-root", type=str, default=None,
                        help="Compare every arm found under this root")
    parser.add_argument("--reference", type=str, default="baseline",
                        help="Substring identifying the reference arm (default: baseline)")
    parser.add_argument("--metric", type=str, default=None,
                        choices=["oma_correct", "is_correct"],
                        help="Correctness field (default: oma_correct when present)")
    parser.add_argument("--confidence", type=float, default=0.95,
                        choices=[0.90, 0.95, 0.99], help="Confidence level")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Write the comparisons to a JSON file")

    args = parser.parse_args()

    comparisons = []

    if len(args.arms) == 2:
        rows_a = load_predictions(Path(args.arms[0]))
        rows_b = load_predictions(Path(args.arms[1]))
        comparisons.append(compare_arms(
            rows_a, rows_b,
            name_a=Path(args.arms[0]).parent.name,
            name_b=Path(args.arms[1]).parent.name,
            metric=_pick_metric(rows_a + rows_b, args.metric),
            confidence=args.confidence,
        ))

    elif args.results_root:
        root = Path(args.results_root)
        if not root.exists():
            print(f"Error: results root not found: {root}", file=sys.stderr)
            return 1

        arms = discover_arms(root)
        if not arms:
            print(f"Error: no predictions.jsonl found under {root}", file=sys.stderr)
            return 1

        references = [label for label in arms if args.reference in label]
        if not references:
            print(
                f"Error: no arm matching '{args.reference}'. Found:\n  "
                + "\n  ".join(sorted(arms)),
                file=sys.stderr,
            )
            return 1

        for ref_label in references:
            ref_rows = load_predictions(arms[ref_label])
            # Only compare arms that differ in mode, not in model or task.
            ref_prefix = str(Path(ref_label).parent.parent)
            for label, path in arms.items():
                if label == ref_label:
                    continue
                if str(Path(label).parent.parent) != ref_prefix:
                    continue
                rows = load_predictions(path)
                comparisons.append(compare_arms(
                    ref_rows, rows,
                    name_a=ref_label, name_b=label,
                    metric=_pick_metric(ref_rows + rows, args.metric),
                    confidence=args.confidence,
                ))
    else:
        parser.error("Provide two predictions.jsonl paths, or --results-root")

    for comparison in comparisons:
        print()
        print("=" * 72)
        print(format_comparison(comparison))

    significant = [c for c in comparisons if c["mcnemar"]["significant_05"]]
    print()
    print("=" * 72)
    print(f"{len(significant)} of {len(comparisons)} comparisons "
          f"are distinguishable from noise at p<0.05.")
    if comparisons and not significant:
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
