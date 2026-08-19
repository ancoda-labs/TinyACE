"""Tests for the lm-evaluation-harness integration.

The value of using the harness is that scoring is not ours. That only holds if
our prompts are the harness's prompts, so the parity checks here are the ones
that matter: if the template drifts, adaptation tunes a playbook against one
rendering and evaluation scores it against another.
"""

import pytest

from edge_slm_ace.data import load_belebele
from edge_slm_ace.data.belebele import HARNESS_TASKS, belebele_path, item_id
from edge_slm_ace.harness.prompts import (
    BELEBELE_DOC_TO_TEXT,
    CHOICE_LETTERS,
    TARGET_DELIMITER,
    assert_matches_harness,
    build_context,
    build_system_instruction,
    choice_continuations,
    render_question,
)
from edge_slm_ace.harness.evaluate import per_item_correctness
from edge_slm_ace.harness.scorer import OptionScorer, option_margin

lm_eval = pytest.importorskip("lm_eval", reason="lm-eval not installed")

needs_data = pytest.mark.skipif(not belebele_path("en").exists(), reason="Belebele not fetched")


def fake_chat_template(messages, add_generation_prompt=True):
    """A chat template with visible markers, so a missing wrap is obvious."""
    body = "|".join(f"{m['role']}:{m['content']}" for m in messages)
    return f"<CHAT>{body}" + ("<GEN>" if add_generation_prompt else "")


@pytest.fixture(scope="module")
def local_belebele_task():
    """
    A real `ConfigurableTask` over the committed English file.

    The point is to compare against the harness's *own* prompt assembly rather
    than against our description of it, without needing the Hub. The config
    mirrors `belebele_eng_Latn`; `assert_matches_harness` is what keeps that
    mirror honest.
    """
    from lm_eval.api.task import ConfigurableTask

    return ConfigurableTask(
        config={
            "task": "belebele_local_parity_probe",
            "dataset_path": "json",
            "dataset_kwargs": {"data_files": {"test": str(belebele_path("en"))}},
            "test_split": "test",
            "output_type": "multiple_choice",
            "doc_to_text": BELEBELE_DOC_TO_TEXT,
            "doc_to_choice": CHOICE_LETTERS,
            "doc_to_target": "{{['1', '2', '3', '4'].index(correct_answer_num)}}",
        }
    )


class TestHarnessParity:
    def test_template_still_matches_the_installed_harness(self):
        """Fails loudly if lm-eval changes the Belebele task under us."""
        assert_matches_harness()

    def test_continuations_use_the_harness_delimiter(self):
        assert choice_continuations() == [f"{TARGET_DELIMITER}{c}" for c in CHOICE_LETTERS]
        assert choice_continuations() == [" A", " B", " C", " D"]

    def test_a_chat_template_drops_the_target_delimiter(self):
        """construct_requests zeroes it when the task has no gen_prefix."""
        assert choice_continuations(apply_chat_template=True) == ["A", "B", "C", "D"]

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

    @needs_data
    @pytest.mark.parametrize("apply_chat_template", [False, True])
    @pytest.mark.parametrize("instruction", [None, "Read the passage."])
    def test_context_is_byte_identical_to_the_harness(
        self, local_belebele_task, apply_chat_template, instruction
    ):
        """
        The check that matters, and the one that was missing.

        Adaptation scored a hand-assembled string while evaluation ran through
        `fewshot_context`. In completion mode the two differed by the "\n\n"
        this module inserted and the harness does not; with a chat template they
        differed by the whole template. Both shift the loglikelihoods, and more
        for the arm carrying the longer prefix.
        """
        doc = list(local_belebele_task.test_docs())[0]
        expected = local_belebele_task.fewshot_context(
            doc,
            0,
            system_instruction=instruction,
            apply_chat_template=apply_chat_template,
            chat_template=fake_chat_template if apply_chat_template else None,
        )
        # load_belebele sorts by id, the raw file does not; join on the passage.
        example = next(e for e in load_belebele("en") if e["context"] == doc["flores_passage"])

        assert (
            build_context(
                render_question(example),
                instruction,
                apply_chat_template=apply_chat_template,
                chat_template=fake_chat_template,
            )
            == expected
        )

    @needs_data
    @pytest.mark.parametrize("apply_chat_template", [False, True])
    def test_continuations_are_the_harness_continuations(
        self, local_belebele_task, apply_chat_template
    ):
        doc = list(local_belebele_task.test_docs())[0]
        requests = local_belebele_task.construct_requests(
            doc, "ctx", apply_chat_template=apply_chat_template
        )
        assert [r.arguments[1] for r in requests] == choice_continuations(apply_chat_template)

    @needs_data
    def test_the_scorer_sends_the_harness_prompt(self):
        """End to end: what OptionScorer puts on the wire, not just a helper."""
        seen = []

        class RecordingLM:
            def apply_chat_template(self, messages, add_generation_prompt=True):
                return fake_chat_template(messages, add_generation_prompt)

            def loglikelihood(self, requests):
                seen.extend(r.arguments for r in requests)
                return [(0.0, False)] * len(requests)

        example = load_belebele("en")[0]
        scorer = OptionScorer("stub", lm=RecordingLM(), apply_chat_template=True)
        scorer.score_one(example, lessons=["Lesson one."])

        contexts = {context for context, _ in seen}
        assert len(contexts) == 1
        assert contexts.pop().startswith("<CHAT>system:")
        assert [continuation for _, continuation in seen] == ["A", "B", "C", "D"]

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


class TestPerItemCorrectness:
    """
    The `samples` selection was never verified against what came back.

    `harness_indices` maps positions in the committed jsonl; the harness loads
    the same split from the Hub. If those orders ever diverge the run scores a
    different question set and still reports a clean accuracy, which is the one
    failure that would invalidate every paired comparison at once.
    """

    def _results(self, docs, accs):
        return {
            "samples": {
                HARNESS_TASKS["en"]: [{"doc": doc, "acc": acc} for doc, acc in zip(docs, accs)]
            }
        }

    @needs_data
    def _docs(self, n):
        import json

        with open(belebele_path("en"), encoding="utf-8") as f:
            return [json.loads(line) for _, line in zip(range(n), f)]

    @needs_data
    def test_reads_correctness_keyed_by_item_id(self):
        docs = self._docs(3)
        per_item = per_item_correctness(self._results(docs, [1, 0, 1]), "en")

        assert set(per_item) == {item_id(d) for d in docs}
        assert sum(per_item.values()) == 2

    @needs_data
    def test_accepts_exactly_the_requested_items(self):
        docs = self._docs(3)
        expected = [item_id(d) for d in docs]
        assert per_item_correctness(self._results(docs, [1, 1, 1]), "en", expected)

    @needs_data
    def test_rejects_a_different_item_set(self):
        docs = self._docs(3)
        wanted = [item_id(d) for d in self._docs(4)]

        with pytest.raises(ValueError, match="different item set"):
            per_item_correctness(self._results(docs, [1, 1, 1]), "en", wanted)

    @needs_data
    def test_a_missing_metric_is_not_scored_as_a_wrong_answer(self):
        """`.get("acc", 0)` turned an upstream metric rename into lost accuracy."""
        results = {"samples": {HARNESS_TASKS["en"]: [{"doc": self._docs(1)[0]}]}}

        with pytest.raises(ValueError, match="no 'acc' metric"):
            per_item_correctness(results, "en")

    def test_unlogged_samples_raise_rather_than_return_nothing(self):
        with pytest.raises(ValueError, match="log_samples"):
            per_item_correctness({"results": {}}, "en")
