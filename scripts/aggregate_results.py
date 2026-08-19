#!/usr/bin/env python3
"""Aggregate a results directory into summary tables.

Walks `{results_root}/{model}/{language}/{arm}/` -- the layout
`reporting.layout` defines -- reads every `metrics.json` and
`predictions.jsonl`, and writes:

    summary_runs.csv          one row per run (run-level metrics)
    summary_accuracy.csv      one row per arm, with Wilson confidence intervals

Accuracies are reported with intervals because a point estimate alone is not
decidable at these sample sizes. Use `scripts/compare_arms.py` for the paired
significance test between two arms.

Usage:
    python -m scripts.aggregate_results
    python -m scripts.aggregate_results --results-root results --out-dir results/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from edge_slm_ace.reporting import (
    health_issues,
    load_predictions,
    load_run_metrics,
    summarize_predictions,
)

# Run-level columns worth surfacing, in display order. Anything absent is
# skipped rather than filled with a placeholder.
_RUN_COLUMNS = [
    "model_label",
    "language",
    "arm_label",
    "seed",
    "n_eval",
    "accuracy",
    "truncation_rate",
    "tokens_dropped",
    "relevance_active",
    "token_counts_exact",
    "playbook_size",
]


def print_health_warnings(runs: pd.DataFrame) -> int:
    """
    Surface conditions that invalidate a run, rather than burying them.

    Invalidating and limiting issues are distinguished by
    `reporting.health`, not re-derived here: `compare_arms` acts on the same
    classification when it decides what enters the test family, and two
    definitions of "invalid" is how a confounded delta reaches a table.

    Args:
        runs: Frame from `load_run_metrics`.

    Returns:
        How many runs carry an invalidating issue.
    """
    if runs.empty:
        return 0

    invalid = 0
    for _, row in runs.iterrows():
        issues = health_issues(row.to_dict())
        if not issues:
            continue
        label = f"{row.get('arm_label', '?')} / {row.get('model_label', '?')}"
        if any(issue.invalidating for issue in issues):
            invalid += 1
        for issue in issues:
            severity = "INVALID " if issue.invalidating else "limitation"
            print(f"  {severity} {label}: {issue.message}", file=sys.stderr)

    return invalid


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results into summary tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory to walk (default: results)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to write the CSVs (default: --results-root)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        choices=[0.90, 0.95, 0.99],
        help="Confidence level",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the printed tables")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any run carries an invalidating health issue. "
        "Off by default so `make report` still reaches compare_arms on a tree "
        "with one bad cell; turn it on for CI or a pre-publication check.",
    )

    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        print(f"Error: results root not found: {root}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir or root)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_run_metrics(root)
    predictions = load_predictions(root)

    if runs.empty and predictions.empty:
        print(f"Error: no metrics.json or predictions.jsonl under {root}", file=sys.stderr)
        return 1

    if not runs.empty:
        columns = [c for c in _RUN_COLUMNS if c in runs.columns]
        runs_out = runs[columns]
        runs_path = out_dir / "summary_runs.csv"
        runs_out.to_csv(runs_path, index=False)
        if not args.quiet:
            print(f"\n{len(runs)} run(s) -> {runs_path}\n")
            print(runs_out.to_string(index=False))

    if not predictions.empty:
        accuracy = summarize_predictions(predictions, confidence=args.confidence)
        accuracy_path = out_dir / "summary_accuracy.csv"
        accuracy.to_csv(accuracy_path, index=False)
        if not args.quiet:
            print(
                f"\nAccuracy by arm ({int(args.confidence * 100)}% Wilson CI) "
                f"-> {accuracy_path}\n"
            )
            display = accuracy.copy()
            for column in ("accuracy", "ci_low", "ci_high", "ci_halfwidth"):
                if column in display.columns:
                    display[column] = display[column].map(lambda v: f"{v:.1%}")
            print(display.to_string(index=False))

    print("\nRun health:", file=sys.stderr)
    invalid = print_health_warnings(runs)

    print(
        "\nA difference between two arms is only a result if it survives "
        "scripts/compare_arms.py, which excludes an invalidated run from the "
        "test family rather than testing it.",
    )

    if invalid and args.strict:
        print(
            f"\nError: {invalid} run(s) carry an invalidating health issue and "
            f"--strict was given. Do not report these numbers; re-run those "
            f"cells.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
