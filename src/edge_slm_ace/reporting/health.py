"""What makes a run unreportable, defined once.

Two different things get called a "warning" about a run, and they are not the
same kind of thing:

- **Invalidating.** The measurement itself is corrupted, and asymmetrically
  between arms. Truncation is the case: lm-eval truncates from the LEFT, which
  for Belebele eats the passage, and a longer prefix truncates more -- so an ACE
  arm loses passages that its control keeps, on the same items. A delta across
  that pair is a difference in how much of the passage each arm got to read.
- **Limiting.** The measurement is sound; what it supports is narrower. A run
  with no embedding backend really did score what it scored, it just was not
  query-conditioned while doing it.

The distinction has to live in one place because two consumers act on it:
`compare_arms` drops an invalidated pair out of the test family, and
`aggregate_results --strict` exits non-zero on one.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass(frozen=True)
class HealthIssue:
    """One thing wrong with a run."""

    field: str
    invalidating: bool
    message: str


def load_metrics(cell_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Read the `metrics.json` beside a cell's predictions.

    Args:
        cell_dir: A results cell directory.

    Returns:
        The parsed metrics, or None when absent or unreadable. None means
        "unknown health", which callers must not read as "healthy".
    """
    path = Path(cell_dir) / "metrics.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def health_issues(metrics: Optional[Dict[str, Any]]) -> List[HealthIssue]:
    """
    Everything wrong with one run, worst first.

    Args:
        metrics: A parsed `metrics.json`, or None if it could not be read.

    Returns:
        Issues, invalidating ones first. Empty for a clean run *and* for
        `metrics=None` -- absent metrics are unknown health, not bad health,
        and `is_reportable` is where that distinction is drawn.
    """
    if not metrics:
        return []

    issues: List[HealthIssue] = []

    truncated = metrics.get("truncated_prompts") or 0
    if truncated:
        rate = metrics.get("truncation_rate") or 0.0
        dropped = metrics.get("tokens_dropped") or 0
        issues.append(
            HealthIssue(
                field="truncated_prompts",
                invalidating=True,
                message=(
                    f"{truncated} prompt(s) truncated ({rate:.1%}), {dropped} tokens "
                    f"dropped from the LEFT -- that is the passage. A longer prefix "
                    f"truncates more, so this arm is not comparable with a "
                    f"shorter-prefix arm on these items"
                ),
            )
        )

    if metrics.get("relevance_active") is False:
        issues.append(
            HealthIssue(
                field="relevance_active",
                invalidating=False,
                message=(
                    "no embedding backend, so playbook retrieval was not "
                    "query-conditioned -- state it as a limitation"
                ),
            )
        )

    if metrics.get("token_counts_exact") is False:
        issues.append(
            HealthIssue(
                field="token_counts_exact",
                invalidating=False,
                message=(
                    "playbook tokens counted with the words * 1.3 estimate, which is "
                    "fertility-blind -- do not compare playbook_tokens across languages"
                ),
            )
        )

    return issues


def invalidating_issues(metrics: Optional[Dict[str, Any]]) -> List[HealthIssue]:
    """The issues that corrupt the measurement rather than narrowing it."""
    return [issue for issue in health_issues(metrics) if issue.invalidating]


def is_reportable(metrics: Optional[Dict[str, Any]]) -> bool:
    """
    Whether a delta involving this run can be reported.

    Args:
        metrics: A parsed `metrics.json`, or None.

    Returns:
        False only for a run with an invalidating issue. Unknown health
        (`metrics=None`) is reportable: a results tree written by an older
        version carries no health fields, and refusing to compare it would be a
        stronger claim than the data supports.
    """
    return not invalidating_issues(metrics)
