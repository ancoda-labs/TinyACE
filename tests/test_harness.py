"""Tests for the lm-evaluation-harness integration.

The value of using the harness is that scoring is not ours. That only holds if
our prompts are the harness's prompts, so the parity checks here are the ones
that matter: if the template drifts, adaptation tunes a playbook against one
rendering and evaluation scores it against another.
"""

import pytest

from edge_slm_ace.data import load_belebele
from edge_slm_ace.data.belebele import HARNESS_TASKS, belebele_path
from edge_slm_ace.harness.prompts import (
    BELEBELE_DOC_TO_TEXT,
    CHOICE_LETTERS,
    TARGET_DELIMITER,
    assert_matches_harness,
    build_system_instruction,
    choice_continuations,
    render_question,
)
from edge_slm_ace.harness.scorer import option_margin

lm_eval = pytest.importorskip("lm_eval", reason="lm-eval not installed")

needs_data = pytest.mark.skipif(not belebele_path("en").exists(), reason="Belebele not fetched")


class TestHarnessParity:
    def test_template_still_matches_the_installed_harness(self):
        """Fails loudly if lm-eval changes the Belebele task under us."""
        assert_matches_harness()

    def test_continuations_use_the_harness_delimiter(self):
        assert choice_continuations() == [f"{TARGET_DELIMITER}{c}" for c in CHOICE_LETTERS]
        assert choice_continuations() == [" A", " B", " C", " D"]

    @needs_data
    def test_rendered_question_matches_the_template_shape(self):
        example = load_belebele("en")[0]
        rendered = render_question(example)
        assert rendered.startswith("P: ")
        assert "\nQ: " in rendered
        assert rendered.endswith("Answer:")
        for letter, option in zip(CHOICE_LETTERS, example["options"]):
            assert f"\n{letter}: {option}" in rendered

    def test_template_constant_is_the_jinja_source(self):
        """The constant is documentation; keep it honest."""
        assert "{{flores_passage}}" in BELEBELE_DOC_TO_TEXT
        assert BELEBELE_DOC_TO_TEXT.endswith("Answer:")

    def test_task_names_exist_in_the_harness(self):
        from lm_eval.tasks import TaskManager

        available = set(TaskManager().all_tasks)
        for task in HARNESS_TASKS.values():
            assert task in available, f"{task} is not a task lm-eval knows"


class TestSystemInstruction:
    def test_baseline_gets_no_prefix(self):
        assert build_system_instruction(include_scaffold=False) == ""

    def test_control_and_ace_share_the_scaffold(self):
        """The only difference between them must be playbook content."""
        control = build_system_instruction(None)
        ace = build_system_instruction(["Check what the passage actually states."])
        assert ace.startswith(control)

    def test_lessons_are_numbered_in_order(self):
        text = build_system_instruction(["first lesson", "second lesson"])
        assert text.index("1. first lesson") < text.index("2. second lesson")

    def test_empty_lesson_list_is_the_control_prefix(self):
        assert build_system_instruction([]) == build_system_instruction(None)


class TestOptionMargin:
    def test_positive_when_gold_is_preferred(self):
        assert option_margin([-1.0, -3.0, -4.0, -5.0], gold_idx=0) == pytest.approx(2.0)

    def test_negative_when_a_distractor_wins(self):
        assert option_margin([-3.0, -1.0, -4.0, -5.0], gold_idx=0) == pytest.approx(-2.0)

    def test_compares_against_the_best_distractor_not_the_mean(self):
        margin = option_margin([-1.0, -1.5, -9.0, -9.0], gold_idx=0)
        assert margin == pytest.approx(0.5)


class TestTruncationCounter:
    """
    The harness truncates a too-long prompt from the LEFT and logs a line. For
    Belebele that eats the passage, and a longer prefix eats more of it -- so
    the ACE arms lose more context than baseline on the same items. The count
    has to travel with the results or that confound is invisible.
    """

    def test_counts_events_and_tokens(self):
        import logging

        from edge_slm_ace.harness.evaluate import TruncationCounter

        with TruncationCounter() as counter:
            log = logging.getLogger("lm_eval.test")
            log.warning(
                "Combined length of context (1135) and continuation (1) exceeds "
                "model's maximum length (1024). Truncating 113 tokens from the left."
            )
            log.warning(
                "Combined length of context (1200) and continuation (1) exceeds "
                "model's maximum length (1024). Truncating 177 tokens from the left."
            )
        assert counter.events == 2
        assert counter.tokens_dropped == 290

    def test_ignores_unrelated_log_lines(self):
        import logging

        from edge_slm_ace.harness.evaluate import TruncationCounter

        with TruncationCounter() as counter:
            logging.getLogger("lm_eval.test").warning("Loading weights: 100%")
        assert counter.events == 0

    def test_detaches_on_exit(self):
        import logging

        from edge_slm_ace.harness.evaluate import TruncationCounter

        with TruncationCounter() as counter:
            pass
        logging.getLogger("lm_eval.test").warning(
            "exceeds model's maximum length (1024). Truncating 5 tokens"
        )
        assert counter.events == 0
