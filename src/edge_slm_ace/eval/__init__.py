"""Evaluation: metrics, MCQ scoring, and statistical inference.

Separated from `utils` because this is the measurement layer, not a grab bag
of helpers. Everything a reported number depends on lives here, so that it is
obvious what needs auditing when a number looks wrong.

- `metrics`: answer-level scoring (exact match, lexical overlap, embeddings)
- `mcq`:     multiple-choice option handling, mapping and OMA/GOM/ACR
- `stats`:   confidence intervals and paired significance testing
"""

from edge_slm_ace.eval.metrics import (
    compute_accuracy,
    compute_average_latency,
    compute_bleu_score,
    compute_semantic_accuracy,
    semantic_answer_score,
    PeakMemoryTracker,
    SemanticEvaluator,
)
from edge_slm_ace.eval.mcq import (
    MCQEvaluator,
    build_prompt_with_choices,
    compute_mcq_aggregate_metrics,
    detect_choice_marker,
    evaluate_mcq_with_indices,
    extract_mcq_options,
    extract_mcq_options_with_indices,
    format_choices_block,
    has_mcq_options,
    is_sciq_task,
    map_prediction_to_option,
    permutation_for,
)
from edge_slm_ace.eval.stats import (
    compare_arms,
    format_comparison,
    mcnemar_exact,
    summarize_accuracy,
    wilson_interval,
)

__all__ = [
    # metrics
    "compute_accuracy",
    "compute_average_latency",
    "compute_bleu_score",
    "compute_semantic_accuracy",
    "semantic_answer_score",
    "PeakMemoryTracker",
    "SemanticEvaluator",
    # mcq
    "MCQEvaluator",
    "build_prompt_with_choices",
    "compute_mcq_aggregate_metrics",
    "detect_choice_marker",
    "evaluate_mcq_with_indices",
    "extract_mcq_options",
    "extract_mcq_options_with_indices",
    "format_choices_block",
    "has_mcq_options",
    "is_sciq_task",
    "map_prediction_to_option",
    "permutation_for",
    # stats
    "compare_arms",
    "format_comparison",
    "mcnemar_exact",
    "summarize_accuracy",
    "wilson_interval",
]
