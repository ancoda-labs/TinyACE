"""Loading and aggregating experiment results.

The runners write three artefacts per run into
`{results_root}/{model}/{task}/{arm}/{device}/`:

    metrics.json        run-level summary
    predictions.jsonl   one row per example
    results.csv         the same rows as CSV

This module is the single reader for all of them, so that every consumer --
tables, figures, significance tests -- sees the same columns under the same
names.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

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
        # Directory layout is {root}/{model}/{task}/{arm}/{device}/
        parts = path.parent.relative_to(results_root).parts
        if len(parts) >= 4:
            record.setdefault("arm", parts[-2])
            record.setdefault("device_used", parts[-1])
        rows.append(record)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["arm"] = df.apply(_arm_of, axis=1)
    df["arm_label"] = df["arm"].map(arm_label)
    df["arm_order"] = df["arm"].map(arm_order)
    if "model_id" in df.columns:
        df["model_label"] = df["model_id"].map(model_label)
    sort_columns = [c for c in ("model_label", "task_name", "arm_order") if c in df.columns]
    return df.sort_values(sort_columns) if sort_columns else df


def load_predictions(results_root: Path) -> pd.DataFrame:
    """
    Load every predictions.jsonl under a results root.

    Args:
        results_root: Directory to walk.

    Returns:
        One row per example across all runs, with canonical column names and
        an `arm` column derived from the directory layout.
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
        parts = path.parent.relative_to(results_root).parts
        frame["arm"] = parts[-2] if len(parts) >= 2 else "unknown"
        frame["device"] = parts[-1] if len(parts) >= 1 else "unknown"
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


def _primary_metric(df: pd.DataFrame) -> str:
    """
    Pick the correctness column to aggregate.

    OMA where available; exact match otherwise. Exact match is near zero for
    any verbose model, so it must not be chosen when OMA exists.
    """
    if "oma_correct" in df.columns and df["oma_correct"].notna().any():
        return "oma_correct"
    return "is_correct"


def summarize_predictions(
    df: pd.DataFrame,
    group_by: Optional[List[str]] = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Aggregate per-example rows into per-arm accuracies with intervals.

    Args:
        df: Frame from `load_predictions`.
        group_by: Grouping columns; defaults to model/task/arm.
        confidence: Confidence level for the Wilson interval.

    Returns:
        One row per group with n, accuracy, ci_low, ci_high and ci_halfwidth.
        `ci_halfwidth` is what a claimed improvement must be compared against.
    """
    from edge_slm_ace.eval.stats import summarize_accuracy

    if df.empty:
        return pd.DataFrame()

    group_by = group_by or [c for c in ("model_label", "task", "arm") if c in df.columns]

    rows = []
    for keys, group in df.groupby(group_by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        # Chosen per group, not per frame. Choosing once over the whole frame
        # picked oma_correct as soon as any MCQ run was present, so a non-MCQ
        # task in the same results root reported 0.0% accuracy with n=0.
        metric = _primary_metric(group)
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
            [c for c in ("model_label", "task", "arm_order") if c in summary.columns]
        ).drop(columns=["arm_order"])
    return summary
