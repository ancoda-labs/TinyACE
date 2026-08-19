"""Loading and aggregating experiment results.

A runner writes two artefacts per cell into the directory `reporting.layout`
defines, `{results_root}/{model}/{language}/{arm}/`:

    metrics.json        run-level summary
    predictions.jsonl   one row per example

This module is the single reader for both, so that every consumer -- tables,
figures, significance tests -- sees the same columns under the same names.

Coordinates come from `layout.parse_cell` rather than from segment arithmetic
done here, and a field the file already carries always wins over the path. The
previous version did the opposite on both counts and overwrote each row's
correct `arm` with the language segment.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from edge_slm_ace.reporting.layout import parse_cell
from edge_slm_ace.reporting.schema import (
    CANONICAL_COLUMNS,
    arm_label,
    arm_order,
    model_label,
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename legacy columns to their canonical names.

    Args:
        df: Raw results frame.

    Returns:
        A copy using canonical names. Where both a legacy and a canonical
        column exist, the canonical one wins and the legacy one is dropped.
    """
    df = df.copy()
    for legacy, canonical in CANONICAL_COLUMNS.items():
        if legacy not in df.columns:
            continue
        if canonical not in df.columns:
            df[canonical] = df[legacy]
        df = df.drop(columns=[legacy])
    return df


def _arm_of(record: Dict) -> str:
    """
    Determine which arm a run belongs to.

    `mode` is "ace" for every ACE variant, so prefer the more specific
    `run_name`/`ace_mode` when present.
    """
    for key in ("arm", "mode_name", "ace_mode"):
        value = record.get(key)
        if value and str(value) != "nan":
            return str(value)
    return str(record.get("mode", "unknown"))


def load_run_metrics(results_root: Path) -> pd.DataFrame:
    """
    Load every metrics.json under a results root.

    Args:
        results_root: Directory to walk.

    Returns:
        One row per run, with `arm`, `arm_label`, `model_label` and
        `arm_order` added. Empty frame when nothing is found.
    """
    rows: List[Dict] = []
    for path in sorted(Path(results_root).rglob("metrics.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: skipping unreadable {path}: {e}")
            continue

        record["_path"] = str(path.parent)
        cell = parse_cell(path.parent.relative_to(results_root))
        if cell is not None:
            # setdefault, not assignment: metrics.json is written by the run
            # itself and is the authority on what it was. The path only fills
            # in what the file does not say.
            record.setdefault("arm", cell.arm)
            record.setdefault("language", cell.language)
            record.setdefault("model_key", cell.model)
        rows.append(record)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["arm"] = df.apply(_arm_of, axis=1)
    df["arm_label"] = df["arm"].map(arm_label)
    df["arm_order"] = df["arm"].map(arm_order)
    if "model_id" in df.columns:
        df["model_label"] = df["model_id"].map(model_label)
    sort_columns = [c for c in ("model_label", "language", "arm_order") if c in df.columns]
    return df.sort_values(sort_columns) if sort_columns else df


def load_predictions(results_root: Path) -> pd.DataFrame:
    """
    Load every predictions.jsonl under a results root.

    Args:
        results_root: Directory to walk.

    Returns:
        One row per example across all runs, with canonical column names and
        `arm`/`language` filled in from the cell path where the rows omit them.
    """
    frames = []
    for path in sorted(Path(results_root).rglob("predictions.jsonl")):
        try:
            rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: skipping unreadable {path}: {e}")
            continue
        if not rows:
            continue

        frame = normalize_columns(pd.DataFrame(rows))
        cell = parse_cell(path.parent.relative_to(results_root))
        # The rows carry `arm` and `language` already; the path is the
        # fallback, not the override. Overriding them merged every arm in a
        # cell under the language segment and pooled their accuracies.
        for column, value in (
            ("arm", cell.arm if cell else "unknown"),
            ("language", cell.language if cell else "unknown"),
        ):
            if column not in frame.columns:
                frame[column] = value
            else:
                frame[column] = frame[column].fillna(value)
        frame["_path"] = str(path.parent)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["arm_label"] = df["arm"].map(arm_label)
    df["arm_order"] = df["arm"].map(arm_order)
    if "model" in df.columns:
        df["model_label"] = df["model"].map(model_label)
    return df


# The correctness column every current runner writes.
#
# There used to be a preference for `oma_correct` here, and in
# `compare_arms._pick_metric`, on the reasoning that exact match is near zero
# for a verbose model. Nothing is generated any more -- correctness is the
# loglikelihood of A-D, decided by the harness -- so no runner writes that
# column, and the only trees that carry it are the withdrawn SciQ results. The
# preference could therefore do exactly one thing: silently report a withdrawn
# metric in place of the live one whenever both were present.
PRIMARY_METRIC = "is_correct"


def summarize_predictions(
    df: pd.DataFrame,
    group_by: Optional[List[str]] = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Aggregate per-example rows into per-arm accuracies with intervals.

    Args:
        df: Frame from `load_predictions`.
        group_by: Grouping columns; defaults to model/language/arm.
        confidence: Confidence level for the Wilson interval.

    Returns:
        One row per group with n, accuracy, ci_low, ci_high and ci_halfwidth.
        `ci_halfwidth` is what a claimed improvement must be compared against.

    Raises:
        KeyError: If the rows carry no `is_correct` column, which means they
            were not written by this pipeline.
    """
    from edge_slm_ace.eval.stats import summarize_accuracy

    if df.empty:
        return pd.DataFrame()

    if PRIMARY_METRIC not in df.columns:
        raise KeyError(
            f"No '{PRIMARY_METRIC}' column in these predictions. Every current "
            f"runner writes it; a tree without it is from the retired pipeline, "
            f"whose results are withdrawn and must not be aggregated."
        )

    group_by = group_by or [c for c in ("model_label", "language", "arm") if c in df.columns]

    rows = []
    for keys, group in df.groupby(group_by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        metric = PRIMARY_METRIC
        stats = summarize_accuracy(group[metric].dropna(), confidence=confidence)
        row = dict(zip(group_by, keys))
        row.update(
            {
                "metric": metric,
                "n": stats["n"],
                "accuracy": stats["accuracy"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "ci_halfwidth": stats["ci_halfwidth"],
            }
        )
        for optional in (
            "latency_ms",
            "prompt_tokens",
            "output_tokens",
            "context_tokens",
            "playbook_tokens",
        ):
            if optional in group.columns:
                row[f"mean_{optional}"] = group[optional].mean()
        rows.append(row)

    summary = pd.DataFrame(rows)
    if "arm" in summary.columns:
        summary["arm_order"] = summary["arm"].map(arm_order)
        summary = summary.sort_values(
            [c for c in ("model_label", "language", "arm_order") if c in summary.columns]
        ).drop(columns=["arm_order"])
    return summary
