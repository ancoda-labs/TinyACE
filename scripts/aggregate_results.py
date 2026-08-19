#!/usr/bin/env python3
"""Aggregate a results directory into summary tables.

Walks `{results_root}/{model}/{task}/{arm}/{device}/`, reads every
`metrics.json` and `predictions.jsonl`, and writes:

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
    "playbook_size",
]


def print_health_warnings(runs: pd.DataFrame) -> None:
    """
    Surface conditions that invalidate a run, rather than burying them.

    Args:
        runs: Frame from `load_run_metrics`.
    """
    if runs.empty:
        return

    if "truncation_rate" in runs.columns:
        bad = runs[runs["truncation_rate"].fillna(0) > 0]
        for _, row in bad.iterrows():
            print(
                f"  WARNING {row.get('arm_label', '?')} / {row.get('model_label', '?')}: "
                f"{row['truncation_rate']:.1%} of prompts truncated -- results invalid",
                file=sys.stderr,
            )

    if "tokens_dropped" in runs.columns:
        bad = runs[runs["tokens_dropped"].fillna(0) > 0]
        for _, row in bad.iterrows():
            print(
                f"  WARNING {row.get('arm_label', '?')} / {row.get('model_label', '?')}: "
                f"{row['tokens_dropped']:.0f} tokens dropped from the passage. A longer "
                f"prefix truncates more, so this arm is not comparable with a shorter one",
                file=sys.stderr,
            )

    if "relevance_active" in runs.columns:
        inactive = runs[runs["relevance_active"] == False]  # noqa: E712
        if not inactive.empty:
            print(
                f"  WARNING {len(inactive)} run(s) had no embedding backend, so "
                f"retrieval was not query-conditioned",
                file=sys.stderr,
            )


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
    print_health_warnings(runs)

    print(
        "\nA difference between two arms is only a result if it survives "
        "scripts/compare_arms.py.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
