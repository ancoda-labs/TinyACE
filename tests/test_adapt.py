"""Tests for the adaptation loop.

The loop is the only code in this project that can influence a reported number,
so it is tested without a model: the scorer and the generator are both stubbed,
which makes every branch reachable and every counter checkable.
"""

import csv

import pytest

from edge_slm_ace.adapt import (
    ADAPT_LOG_FIELDS,
    adapt_playbook,
    build_reflector_prompt,
    frozen_lessons,
    save_adaptation_log,
)
from edge_slm_ace.memory.playbook import Playbook

LESSON = "- Prefer the option whose wording is supported by a sentence in the passage."
CURATOR_KEEP = "Lesson 1: is_generic=False"
CURATOR_REJECT = "Lesson 1: is_generic=True"


def make_examples(n=4):
    return [
        {
            "id": f"item-{i}",
            "question": f"Question {i}?",
            "context": f"Passage {i}. It states a fact.",
            "options": [f"opt{i}a", f"opt{i}b", f"opt{i}c", f"opt{i}d"],
            "gold_option_idx": i % 4,
            "answer": f"opt{i}{'abcd'[i % 4]}",
            "domain": "belebele_en",
        }
        for i in range(n)
    ]


class StubScorer:
    """Returns a fixed choice per example, and records the lessons it was shown."""

    def __init__(self, picks):
        self.picks = list(picks)
        self.seen_lessons = []

    def score_one(self, example, lessons=None, include_scaffold=True):
        self.seen_lessons.append(list(lessons or []))
        choice = self.picks.pop(0)
        logprobs = [-5.0] * 4
        logprobs[choice] = -1.0
        return choice, logprobs


def stub_generate(reflection=LESSON, curation=CURATOR_KEEP):
    def generate(prompt, max_new_tokens):
        return curation if "Curator" in prompt else reflection

    return generate


class RecordingGenerate:
    """A stub that remembers which role it was asked to play.

    Asserting on the *outcome* of a disabled Curator cannot tell it from a
    Curator that ran and kept everything -- both leave zero rejections and the
    lesson stored. Only the call record separates them.
    """

    def __init__(self, reflection=LESSON, curation=CURATOR_KEEP):
        self.reflection = reflection
        self.curation = curation
        self.prompts = []

    def __call__(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        if "Curator" in prompt:
            return self.curation
        return self.reflection

    @property
    def curator_calls(self):
        return [p for p in self.prompts if "Curator" in p]

    @property
    def reflector_calls(self):
        return [p for p in self.prompts if "Curator" not in p]


def run(picks, playbook=None, **kwargs):
    examples = make_examples(len(picks))
    scorer = StubScorer(picks)
    playbook = playbook or Playbook(token_budget=256)
    kwargs.setdefault("generate", stub_generate())
    kwargs.setdefault("prune_every_n", 0)
    kwargs.setdefault("progress", False)
    out = adapt_playbook(scorer, examples, playbook, domain="belebele_en", **kwargs)
    return out, playbook, scorer


class TestCorrectness:
    def test_accuracy_counts_loglik_choices(self):
        """Gold indices for the four items are 0, 1, 2, 3."""
        out, _, _ = run([0, 1, 0, 0])  # two right, two wrong
        assert out["accuracy"] == pytest.approx(0.5)

    def test_all_correct(self):
        out, _, _ = run([0, 1, 2, 3])
        assert out["accuracy"] == 1.0

    def test_all_wrong(self):
        out, _, _ = run([1, 0, 0, 0])
        assert out["accuracy"] == 0.0


class TestReflection:
    def test_reflects_only_on_errors_by_default(self):
        out, _, _ = run([0, 1, 2, 3])  # all correct
        assert sum(r["reflected"] for r in out["log"]) == 0

    def test_reflects_on_every_error(self):
        out, _, _ = run([1, 0, 0, 0])  # all wrong
        assert sum(r["reflected"] for r in out["log"]) == 4

    def test_can_also_reflect_on_correct_answers(self):
        out, _, _ = run([0, 1, 2, 3], reflect_on_correct_every_n=2)
        assert sum(r["reflected"] for r in out["log"]) == 2


class TestCurator:
    def test_rejected_lessons_are_counted_and_not_stored(self):
        out, playbook, _ = run([1, 1, 1, 1], generate=stub_generate(curation=CURATOR_REJECT))
        assert out["lessons_rejected_by_curator"] > 0
        assert len(playbook.entries) == 0

    def test_accepted_lessons_are_stored_once(self):
        """The same lesson every step must deduplicate, not accumulate."""
        out, playbook, _ = run([1, 0, 0, 0])
        assert len(playbook.entries) == 1
        assert out["log"][0]["entries_added"] == 1
        assert out["log"][1]["entries_added"] == 0

    def test_curator_can_be_disabled(self):
        out, playbook, _ = run([1, 0, 0, 0], use_curator=False)
        assert out["lessons_rejected_by_curator"] == 0
        assert len(playbook.entries) == 1

    def test_disabling_the_curator_stops_it_being_called(self):
        """`tinyace_ablate_no_curator` is a registered arm; this is what it is.

        Asserting zero rejections does not establish it. `use_curator and
        proposed` changed to `or` runs the Curator with use_curator=False, and
        the keep-everything stub then leaves the same zero rejections and the
        same stored lesson -- the outcome assertions cannot see the difference.
        Only the call record can.
        """
        generate = RecordingGenerate()
        run([1, 0, 0, 0], use_curator=False, generate=generate)

        assert generate.reflector_calls, "the Reflector must still run"
        assert generate.curator_calls == [], "the Curator ran with use_curator=False"

    def test_enabling_the_curator_does_call_it(self):
        generate = RecordingGenerate()
        run([1, 0, 0, 0], use_curator=True, generate=generate)
        assert generate.curator_calls, "the Curator never ran with use_curator=True"

    def test_the_curator_is_not_called_when_nothing_was_proposed(self):
        """No candidates means no verdict to ask for -- and no generation to pay for."""
        generate = RecordingGenerate(reflection="too short")
        run([1, 0, 0, 0], use_curator=True, generate=generate)
        assert generate.curator_calls == []

    def test_the_rejected_count_is_proposed_minus_kept(self):
        """`len(proposed) - len(curated)`, and a partial verdict is what shows it.

        When the Curator rejects everything, `curated` is empty and a sum reads
        the same as a difference -- both give len(proposed). Only a verdict that
        keeps some and rejects others separates them.
        """
        two_lessons = (
            "- Reject an option that is true in general but is not stated in the passage\n"
            "- Eliminate an option that reverses the direction of the stated cause"
        )
        partial = "Lesson 1: is_generic=True\nLesson 2: is_generic=False"

        out, playbook, _ = run(
            [1, 0, 0, 0], generate=stub_generate(reflection=two_lessons, curation=partial)
        )

        first = out["log"][0]
        assert first["lessons_proposed"] == 2
        assert first["lessons_rejected_by_curator"] == 1, "one rejected, one kept"
        assert len(playbook.entries) == 1


class TestReflectOnCorrectSchedule:
    """`reflect_on_correct_every_n` costs a generation per firing.

    The guard is `n and step % n == 0`. Nothing pinned either half, so the
    feature could have fired on every step, or never, without a test objecting.
    """

    def test_zero_means_errors_only(self):
        generate = RecordingGenerate()
        run([0, 1, 2, 3], reflect_on_correct_every_n=0, generate=generate)
        assert generate.reflector_calls == [], "all four were correct"

    def test_it_fires_on_the_nth_correct_step_only(self):
        out, _, _ = run([0, 1, 2, 3], reflect_on_correct_every_n=2)
        reflected = [row["step"] for row in out["log"] if row["reflected"]]
        assert reflected == [2, 4]

    def test_every_step_when_n_is_one(self):
        out, _, _ = run([0, 1, 2, 3], reflect_on_correct_every_n=1)
        assert [row["step"] for row in out["log"] if row["reflected"]] == [1, 2, 3, 4]

    def test_errors_still_reflect_whatever_the_schedule(self):
        """Gold is `i % 4`, so [1, 0, 2, 3] is wrong at steps 1 and 2."""
        out, _, _ = run([1, 0, 2, 3], reflect_on_correct_every_n=3)
        reflected = [row["step"] for row in out["log"] if row["reflected"]]
        assert {1, 2} <= set(reflected), "wrong answers always reflect"
        assert 3 in reflected, "step 3 was correct and is on the schedule"
        assert 4 not in reflected, "step 4 was correct and is off the schedule"


class TestChatTemplateFollowsTheScorer:
    """The Reflector prompt must be wrapped exactly when scoring wraps.

    `scorer.apply_chat_template and chat_template is not None` changed to `or`
    applies the template to a run that asked for raw completion -- which is what
    `--no-chat-template` exists to turn off for a base checkpoint.
    """

    def scorer_with(self, apply_chat_template):
        class LM:
            def __init__(self):
                self.wrapped = []

            def apply_chat_template(self, messages, **kwargs):
                self.wrapped.append(messages)
                return "<|template|>"

            def generate_until(self, requests):
                return ["- a strategy about the passage and its options here"]

        class Scorer:
            pass

        scorer = Scorer()
        scorer.lm = LM()
        scorer.apply_chat_template = apply_chat_template
        return scorer

    def test_the_template_is_applied_when_scoring_applies_it(self):
        from edge_slm_ace.adapt import _default_generate

        scorer = self.scorer_with(True)
        _default_generate(scorer)("a prompt", 32)
        assert scorer.lm.wrapped, "an instruct checkpoint was handed a raw completion"

    def test_the_template_is_not_applied_when_scoring_does_not(self):
        from edge_slm_ace.adapt import _default_generate

        scorer = self.scorer_with(False)
        _default_generate(scorer)("a prompt", 32)
        assert scorer.lm.wrapped == [], "--no-chat-template was ignored for reflection"


class TestRetrieval:
    def test_lessons_are_shown_once_the_playbook_is_non_empty(self):
        _, _, scorer = run([1, 0, 0, 0])
        assert scorer.seen_lessons[0] == [], "nothing to show on the first step"
        assert scorer.seen_lessons[-1], "later steps must see the learned lesson"

    def test_feedback_follows_the_outcome(self):
        """All four wrong, so a retrieved lesson only ever accrues failures."""
        _, playbook, _ = run([1, 0, 0, 0])
        entry = playbook.entries[0]
        assert entry.failure_count > 0
        assert entry.success_count == 0


class TestLogSchema:
    def test_log_rows_match_the_declared_schema(self):
        out, _, _ = run([1, 0, 2, 3])
        for row in out["log"]:
            assert list(row) == ADAPT_LOG_FIELDS

    def test_log_round_trips_to_csv(self, tmp_path):
        out, _, _ = run([1, 0, 2, 3])
        path = tmp_path / "adapt_log.csv"
        save_adaptation_log(out["log"], path)
        with open(path, encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == 4

    def test_gold_margin_is_signed(self):
        out, _, _ = run([0, 0, 0, 0])  # item 0 correct, rest wrong
        assert out["log"][0]["gold_margin"] > 0
        assert out["log"][1]["gold_margin"] < 0


class TestReflectorPrompt:
    def test_prompt_shows_both_options_and_forbids_restating(self):
        example = make_examples(1)[0]
        prompt = build_reflector_prompt(example, chosen_idx=1)
        assert example["context"] in prompt
        assert "Chosen: B." in prompt
        assert "Correct: A." in prompt
        assert "Do NOT restate the answer" in prompt


class TestFrozenLessons:
    def test_returns_the_ranked_lessons(self):
        _, playbook, _ = run([1, 0, 0, 0])
        assert frozen_lessons(playbook, "belebele_en", top_k=5) == [
            e.text for e in playbook.entries
        ]

    def test_empty_playbook_gives_the_control_prefix(self):
        assert frozen_lessons(Playbook(), "belebele_en") == []


class TestFrozenLessonsAgeTheirCandidates:
    """
    frozen_lessons() passed current_step=0, so every entry's age was 0 and the
    recency bonus gamma was a constant across candidates. A constant cannot
    reorder anything, so recency had no effect on the prefix the model is
    actually shown -- and tinyace_ablate_no_recency was ablating a term that was
    already inert there. Playbook.prune carried the identical defect and was
    fixed; this one survived in the function that decides what ships.
    """

    def _playbook(self):
        playbook = Playbook()
        stale = playbook.add_entry("d", "Compare the option against the passage wording", 1)
        fresh = playbook.add_entry("d", "Eliminate an option that reverses the cause", 2)
        # Same everything except when they were last used.
        stale.last_used_at = 1
        fresh.last_used_at = 200
        return playbook, stale, fresh

    def test_the_recently_used_lesson_ranks_first(self):
        playbook, _, fresh = self._playbook()
        assert frozen_lessons(playbook, "d", top_k=1) == [fresh.text]

    def test_an_explicit_step_is_honoured(self):
        playbook, _, fresh = self._playbook()
        assert frozen_lessons(playbook, "d", top_k=1, current_step=500) == [fresh.text]

    def test_recency_actually_changes_the_ranking(self):
        """The regression: with step 0 both orderings were identical."""
        playbook, stale, fresh = self._playbook()
        aged = frozen_lessons(playbook, "d", top_k=2)

        stale.last_used_at, fresh.last_used_at = 200, 1
        reversed_ = frozen_lessons(playbook, "d", top_k=2)

        assert aged != reversed_


class TestEntryCapHoldsOnAnySplitLength:
    """The cap must not depend on the split dividing by the prune interval.

    Pruning ran only on the interval, so a run that ended mid-cycle saved a
    playbook over `max_entries_per_domain` by up to one interval's worth of
    lessons. 400 items at every 25 is exact and hid it; `--limit`, a different
    ADAPTATION_SIZE or any odd split did not. That oversized playbook is the
    artifact the cross-lingual arm borrows.
    """

    def adapt(self, n_items, prune_every_n=5, cap=4):
        counter = {"i": 0}

        def generate(prompt, max_new_tokens):
            if "Curator" in prompt:
                return CURATOR_KEEP
            counter["i"] += 1
            return (
                f"- Eliminate the option that reverses the stated cause, "
                f"case {counter['i']} of the passage"
            )

        examples = make_examples(n_items)
        # Always wrong, so every step reflects and proposes a fresh lesson.
        scorer = StubScorer([(e["gold_option_idx"] + 1) % 4 for e in examples])
        playbook = Playbook()
        summary = adapt_playbook(
            scorer,
            examples,
            playbook,
            domain="belebele_en",
            generate=generate,
            prune_every_n=prune_every_n,
            max_entries_per_domain=cap,
            progress=False,
        )
        return playbook, summary

    @pytest.mark.parametrize("n_items", [5, 6, 7, 8, 9, 11, 13])
    def test_the_saved_playbook_is_within_the_cap(self, n_items):
        playbook, summary = self.adapt(n_items)
        assert len(playbook.entries) <= 4
        assert summary["playbook_size"] <= 4

    def test_a_split_that_divides_evenly_still_works(self):
        playbook, _ = self.adapt(10)
        assert len(playbook.entries) <= 4

    def test_the_final_prune_is_counted_in_the_log(self):
        _, summary = self.adapt(7)
        assert sum(row["num_evictions"] for row in summary["log"]) > 0
