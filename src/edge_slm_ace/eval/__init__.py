"""Uncertainty, significance, and deterministic option shuffling.

Answer scoring lives in `harness/` now: lm-evaluation-harness decides
correctness by the loglikelihood of the option letters. What remains here is
deciding whether a difference between two arms is distinguishable from noise,
plus ``process_docs`` / custom task YAMLs under ``eval/tasks/``.
"""

from edge_slm_ace.eval.stats import (
    align_on_key,
    compare_arms,
    format_comparison,
    holm_bonferroni,
    mcnemar_exact,
    summarize_accuracy,
    wilson_interval,
)
from edge_slm_ace.eval.tasks import TASKS_DIR
from edge_slm_ace.eval.utils import process_docs, shuffle_mcq_doc

__all__ = [
    "TASKS_DIR",
    "align_on_key",
    "compare_arms",
    "format_comparison",
    "holm_bonferroni",
    "mcnemar_exact",
    "process_docs",
    "shuffle_mcq_doc",
    "summarize_accuracy",
    "wilson_interval",
]
