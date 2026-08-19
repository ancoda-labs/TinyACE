#!/usr/bin/env python3
"""Compare two experiment arms with confidence intervals and a paired test.

Point estimates alone are not decidable at these sample sizes: at n=50 the
95% CI on an accuracy near 0.74 is roughly +/-12 percentage points, so a
"+4%" improvement is two questions and is indistinguishable from noise. Run
this before putting any delta in a table.

Each arm is compared against the reference `reporting.reference_for()` names
for it -- ACE arms against `scaffold_control`, ablations against `tinyace` --
because comparing an ablation to `baseline` measures ACE *plus* the ablation
rather than the ablated component. All comparisons printed together form one
family and are Holm-corrected; the adjusted p-value is the one that decides.

Usage:
    # Two prediction files
    python -m scripts.compare_arms \
        results/qwen3-1.7b/ne/baseline/predictions.jsonl \
        results/qwen3-1.7b/ne/tinyace/predictions.jsonl

    # Every arm against its registered reference, under one results root
    python -m scripts.compare_arms --results-root results

    # Override the registry to ask one specific question
    python -m scripts.compare_arms --results-root results --reference baseline
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from edge_slm_ace.eval.stats import compare_arms, format_comparison, holm_bonferroni
from edge_slm_ace.reporting import (
    PRIMARY_METRIC,
    arm_label,
    get_arm,
    invalidating_issues,
    load_metrics,
    parse_cell,
    reference_for,
)


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
        Mapping of cell label -> path. The label is the directory path
        relative to the root, which encodes model/language/arm.
    """
    arms = {}
    for path in sorted(results_root.rglob("predictions.jsonl")):
        arms[str(path.parent.relative_to(results_root))] = path
    return arms


def arm_key_of(label: str) -> str:
    """
    The arm a results path belongs to, via `reporting.layout`.

    This used to be `parts[-2]`, against a four-segment layout that carried a
    device. On the three-segment layout the runners write it returned the
    *language*, so every arm resolved to "ne" or "en", none of them registered,
    and every comparison was skipped for want of a reference.

    Args:
        label: Cell directory, relative to the results root.

    Returns:
        The arm key, as registered in `reporting.schema`.
    """
    cell = parse_cell(label)
    return cell.arm if cell else label


def cell_of(label: str) -> str:
    """The {model}/{language} prefix a comparison must not cross."""
    cell = parse_cell(label)
    return cell.group if cell else str(Path(label).parent)


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
        if get_arm(key) is None:
            # reference_for() falls back to "baseline" for a key it does not
            # know, which is the wrong reference for anything ACE-shaped and
            # exactly the mistake this function exists to prevent. Say so
            # rather than emitting a comparison that looks authoritative.
            print(
                f"Note: '{key}' is not a registered arm, so its reference "
                f"defaults to baseline. Register it in reporting/schema.py if "
                f"it should be compared against something else.",
                file=sys.stderr,
            )
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


def health_of(path: Path) -> List:
    """
    Invalidating issues for the run whose predictions live at `path`.

    This script used to read `predictions.jsonl` and nothing else, so the one
    tool the project points at as the gate on whether a delta is a result could
    not see the condition that makes a delta uninterpretable. Truncation is not
    noise: lm-eval truncates from the left, which for Belebele eats the passage,
    and a longer prefix truncates more -- so an ACE arm loses passages its
    control keeps, on the same items. That is a difference in how much each arm
    got to read, reported as a difference in what the playbook did.
    """
    return invalidating_issues(load_metrics(path.parent))


def _pick_metric(rows: List[Dict], requested: Optional[str]) -> str:
    """
    Choose the correctness field to compare on.

    `PRIMARY_METRIC` unless the caller names something else. This used to
    prefer `oma_correct` whenever any row carried it, which after the move to
    loglikelihood scoring could only ever mean one thing: a withdrawn SciQ
    result quietly deciding a live comparison.
    """
    return requested or PRIMARY_METRIC


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
            "scaffold_control, ablations against tinyace."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help=f"Correctness column to compare on (default: {PRIMARY_METRIC}, "
        f"which is what every runner writes). Naming another one is how a "
        f"pre-harness results tree is read, deliberately and on the record.",
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
        path_a, path_b = Path(args.arms[0]), Path(args.arms[1])
        rows_a = load_predictions(path_a)
        rows_b = load_predictions(path_b)
        comparison = compare_arms(
            rows_a,
            rows_b,
            name_a=path_a.parent.name,
            name_b=path_b.parent.name,
            metric=_pick_metric(rows_a + rows_b, args.metric),
            confidence=args.confidence,
        )
        comparison["health"] = {
            comparison["name_a"]: health_of(path_a),
            comparison["name_b"]: health_of(path_b),
        }
        comparisons.append(comparison)

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

        for skipped_label, reference_key in missing:
            print(
                f"Note: skipping {skipped_label} -- its reference arm "
                f"'{reference_key}' was not run in this cell.",
                file=sys.stderr,
            )

        for reference_label, label in pairs:
            reference_rows = load_predictions(arms[reference_label])
            rows = load_predictions(arms[label])
            comparison = compare_arms(
                reference_rows,
                rows,
                name_a=arm_label(arm_key_of(reference_label)),
                name_b=arm_label(arm_key_of(label)),
                metric=_pick_metric(reference_rows + rows, args.metric),
                confidence=args.confidence,
            )
            comparison["health"] = {
                comparison["name_a"]: health_of(arms[reference_label]),
                comparison["name_b"]: health_of(arms[label]),
            }
            comparisons.append(comparison)
    else:
        parser.error("Provide two predictions.jsonl paths, or --results-root")

    # Every comparison printed together is one family of tests. Reporting each
    # p<0.05 on its own across a 10-arm sweep gives roughly a 40% chance of at
    # least one false positive, so the adjusted value is what decides.
    #
    # Two kinds of pair sit outside the family:
    #
    # - No comparable items. A forced p=1.0 carrying no evidence; counting it
    #   would inflate the family size and cost the real comparisons power for
    #   nothing.
    # - An arm with an invalidating health issue. Truncation is arm-asymmetric
    #   -- the longer prefix truncates more -- so the pair carries *biased*
    #   evidence, which is worse than none. It must not consume family power and
    #   must not be counted as a survivor.
    for comparison in comparisons:
        unhealthy = sorted(name for name, issues in comparison["health"].items() if issues)
        comparison["not_reportable"] = unhealthy
        comparison["mcnemar"]["in_test_family"] = comparison["mcnemar"]["n"] > 0 and not unhealthy
        comparison["mcnemar"]["p_adjusted"] = 1.0
        comparison["mcnemar"]["significant_05_adjusted"] = False
        if unhealthy:
            # Replace the verdict rather than printing a banner under it. The
            # verdict is the line a reader takes away, and "TinyACE is better
            # than Scaffold Control (p=0.0151)" is the wrong thing to leave at
            # the top of a block whose numbers are confounded.
            comparison["verdict"] = (
                f"NOT REPORTABLE -- {', '.join(unhealthy)} failed a health check; "
                f"this delta is confounded, not a measurement of the playbook"
            )

    testable = [c for c in comparisons if c["mcnemar"]["in_test_family"]]
    no_items = [c for c in comparisons if c["mcnemar"]["n"] == 0]
    invalid = [c for c in comparisons if c["not_reportable"]]

    adjusted = holm_bonferroni([c["mcnemar"]["p_value"] for c in testable])
    for comparison, p_adjusted in zip(testable, adjusted):
        comparison["mcnemar"]["p_adjusted"] = p_adjusted
        comparison["mcnemar"]["significant_05_adjusted"] = p_adjusted < 0.05

    for comparison in comparisons:
        print()
        print("=" * 72)
        print(format_comparison(comparison))

        for name in comparison["not_reportable"]:
            for issue in comparison["health"][name]:
                print(f"     {name}: {issue.message}.")
            print("     Excluded from the test family.")

        test = comparison["mcnemar"]
        if test["in_test_family"] and len(testable) > 1:
            verdict = "survives" if test["significant_05_adjusted"] else "does not survive"
            print(
                f"  -> p={test['p_value']:.4f} {verdict} Holm correction "
                f"across {len(testable)} tests (p_adj={test['p_adjusted']:.4f})"
            )

    raw = [c for c in comparisons if c["mcnemar"]["significant_05"]]
    survivors = [c for c in comparisons if c["mcnemar"]["significant_05_adjusted"]]
    print()
    print("=" * 72)
    print(
        f"{len(survivors)} of {len(testable)} comparisons survive Holm "
        f"correction at p<0.05 ({len(raw)} before correction, "
        f"family size {len(testable)})."
    )
    if no_items:
        print(f"{len(no_items)} comparison(s) had no items in common and were not tested.")
    if invalid:
        print(
            f"{len(invalid)} comparison(s) involve a run that failed a health "
            f"check and are NOT REPORTABLE. They are excluded from the family "
            f"above; re-run those cells before reporting anything about them."
        )
    if testable and not survivors:
        print("Report these as 'no detectable difference', not as an ordering.")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        serialisable = []
        for comparison in comparisons:
            record = dict(comparison)
            # `default=str` would stringify the dataclass into an unparseable
            # repr; the health record is the reason a reader trusts or discards
            # the delta next to it, so it has to survive as structured data.
            record["health"] = {
                name: [asdict(issue) for issue in issues]
                for name, issues in comparison["health"].items()
            }
            serialisable.append(record)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, default=str)
        print(f"Wrote comparisons to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
