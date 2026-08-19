"""Uncertainty and significance.

Answer scoring lives in `harness/` now: lm-evaluation-harness decides
correctness by the loglikelihood of the option letters, which removed the
bespoke MCQ mapping and text-similarity metrics this package used to carry.
What remains here is the part the harness does not do -- deciding whether a
difference between two arms is distinguishable from noise.
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

__all__ = [
    "align_on_key",
    "compare_arms",
    "format_comparison",
    "holm_bonferroni",
    "mcnemar_exact",
    "summarize_accuracy",
    "wilson_interval",
]
