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
