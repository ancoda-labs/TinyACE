"""lm-evaluation-harness integration.

Evaluation is a stock harness run; only adaptation is ours.
"""

from edge_slm_ace.harness.evaluate import (
    accuracy_of,
    per_item_correctness,
    run_frozen_eval,
)
from edge_slm_ace.harness.prompts import (
    assert_matches_harness,
    build_system_instruction,
    choice_continuations,
    render_question,
)
from edge_slm_ace.harness.scorer import OptionScorer, option_margin

__all__ = [
    "OptionScorer",
    "accuracy_of",
    "assert_matches_harness",
    "build_system_instruction",
    "choice_continuations",
    "option_margin",
    "per_item_correctness",
    "render_question",
    "run_frozen_eval",
]
