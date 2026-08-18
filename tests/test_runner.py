"""Tests for the evaluation loop.

`core/runner.py` is the largest module in the package and decides what
"correct" means, what every result row contains, and what the only difference
between two arms is. It had no tests, which is why three separate defects --
a playbook-log header that fell behind its writer, a Curator counter summed
over the wrong collection, and a citation block parsed into the scored answer
-- all survived a full audit.

The loop is tested without a model: `generate` and `count_tokens` are the only
things in it that need one, and both are patched here with deterministic
stand-ins that dispatch on the prompt they are handed.
"""

import csv

import pytest

from edge_slm_ace.core import runner
from edge_slm_ace.core.runner import (
    PLAYBOOK_LOG_FIELDS,
    run_dataset_ace,
    run_dataset_baseline,
)
from edge_slm_ace.memory.playbook import Playbook
from edge_slm_ace.utils.config import ModelConfig

CONFIG = ModelConfig(model_id="stub", max_new_tokens=64, temperature=0.0, top_p=1.0)

EXAMPLES = [
    {"id": "q1", "question": "What organelle makes ATP?", "answer": "mitochondria"},
    {"id": "q2", "question": "What gas do plants absorb?", "answer": "carbon dioxide"},
]


class StubTokenizer:
    """Whitespace tokenizer; enough for token counting and budget arithmetic."""

    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return text.split()


def make_generate(generator_reply, reflector_reply="", curator_reply=""):
    """
    Build a stand-in for `model_manager.generate`.

    Dispatches on the prompt so the stub does not depend on call ordering.
    """

    def _generate(model, tokenizer, prompt, return_meta=False, **kwargs):
        if "You are a Curator" in prompt:
            text = curator_reply
        elif "Your task: Extract very specific" in prompt:
            text = reflector_reply
        else:
            text = generator_reply
        if return_meta:
            return text, {"used_chat_template": True, "prompt_truncated": False}
        return text

    return _generate


@pytest.fixture
def stub_generation(monkeypatch):
    """Patch generation and token counting; return a setter for the replies."""

    def install(generator_reply, reflector_reply="", curator_reply=""):
        monkeypatch.setattr(
            runner, "generate", make_generate(generator_reply, reflector_reply, curator_reply)
        )
        monkeypatch.setattr(runner, "count_tokens", lambda tok, text: len(str(text).split()))

    return install


def run_ace(stub_generation, tmp_path, generator_reply, reflector_reply="", curator_reply=""):
    """Run one ACE pass over EXAMPLES with the given canned generations."""
    stub_generation(generator_reply, reflector_reply, curator_reply)
    playbook = Playbook(token_budget=256)
    return run_dataset_ace(
        model=None,
        tokenizer=StubTokenizer(),
        dataset=EXAMPLES,
        domain="science",
        config=CONFIG,
        playbook=playbook,
        playbook_path=tmp_path / "playbook.jsonl",
        model_id="stub",
        task_name="iot_tiny",  # non-SciQ: keeps the embedding backend out of it
    )


class TestPlaybookLogSchema:
    """The log header and the log writer must not be able to drift apart."""

    def test_schema_matches_what_the_loop_writes(self, stub_generation, tmp_path):
        _, summary = run_ace(stub_generation, tmp_path, "Answer:\nmitochondria")
        log = summary["playbook_log"]
        assert log, "expected one log row per step"
        for row in log:
            assert list(row.keys()) == PLAYBOOK_LOG_FIELDS

    def test_log_round_trips_through_dictwriter(self, stub_generation, tmp_path):
        """The exact write scripts/run_experiment.py performs."""
        _, summary = run_ace(stub_generation, tmp_path, "Answer:\nmitochondria")
        path = tmp_path / "playbook_log.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PLAYBOOK_LOG_FIELDS)
            writer.writeheader()
            writer.writerows(summary["playbook_log"])  # raised ValueError before the fix

        with open(path, encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == len(EXAMPLES)


class TestCuratorAccounting:
    """A rejected lesson has to be visible, or the ablation measures nothing."""

    REFLECTION = (
        "- For ATP questions, name the organelle that performs oxidative phosphorylation.\n"
    )

    def test_rejections_are_counted(self, stub_generation, tmp_path):
        _, summary = run_ace(
            stub_generation,
            tmp_path,
            generator_reply="Answer:\nwrong",
            reflector_reply=self.REFLECTION,
            curator_reply="Lesson 1: is_generic=True",
        )
        assert summary["lessons_rejected_by_curator"] > 0
        assert summary["playbook_size"] == 0, "a rejected lesson must not be stored"

    def test_accepted_lessons_are_not_counted_as_rejections(self, stub_generation, tmp_path):
        _, summary = run_ace(
            stub_generation,
            tmp_path,
            generator_reply="Answer:\nwrong",
            reflector_reply=self.REFLECTION,
            curator_reply="Lesson 1: is_generic=False",
        )
        assert summary["lessons_rejected_by_curator"] == 0
        assert summary["playbook_size"] > 0

    def test_curator_cost_is_reported(self, stub_generation, tmp_path):
        _, summary = run_ace(
            stub_generation,
            tmp_path,
            generator_reply="Answer:\nwrong",
            reflector_reply=self.REFLECTION,
            curator_reply="Lesson 1: is_generic=False",
        )
        assert "total_curator_latency_ms" in summary


class TestResultSchema:
    """Both arms must emit the columns the reporting layer joins on."""

    JOIN_COLUMNS = {"qid", "task", "model", "mode", "is_correct", "latency_ms"}

    def test_baseline_rows_carry_the_join_columns(self, stub_generation, tmp_path):
        stub_generation("Answer:\nmitochondria")
        results, _ = run_dataset_baseline(
            model=None,
            tokenizer=StubTokenizer(),
            dataset=EXAMPLES,
            domain="science",
            config=CONFIG,
            model_id="stub",
            task_name="iot_tiny",
        )
        assert self.JOIN_COLUMNS <= set(results[0])

    def test_ace_rows_carry_the_same_join_columns(self, stub_generation, tmp_path):
        results, _ = run_ace(stub_generation, tmp_path, "Answer:\nmitochondria")
        assert self.JOIN_COLUMNS <= set(results[0])


class TestArmIdentity:
    """
    The control arm must be distinguishable from the arm it controls for.

    run_dataset_ace wrote `mode = ace_mode` into every row, and cot_control
    goes through the same code path with ace_mode still set -- so the control
    labelled itself "ace_full". Anything grouping on that column pooled the two,
    and `ace - cot_control` is the entire claim the protocol rests on.
    """

    def run(self, stub_generation, tmp_path, mode, ace_mode="ace_full", learning=True):
        stub_generation("Answer:\nmitochondria")
        results, _ = run_dataset_ace(
            model=None,
            tokenizer=StubTokenizer(),
            dataset=EXAMPLES,
            domain="science",
            config=CONFIG,
            playbook=Playbook(token_budget=256),
            playbook_path=tmp_path / "pb.jsonl",
            model_id="stub",
            task_name="iot_tiny",
            mode=mode,
            ace_mode=ace_mode,
            enable_learning=learning,
        )
        return results

    def test_control_rows_are_labelled_cot_control(self, stub_generation, tmp_path):
        rows = self.run(stub_generation, tmp_path, mode="cot_control", learning=False)
        assert {r["mode"] for r in rows} == {"cot_control"}

    def test_ace_rows_keep_the_ace_mode_variant(self, stub_generation, tmp_path):
        """ace_full and ace_working_memory still have to be told apart."""
        rows = self.run(stub_generation, tmp_path, mode="ace", ace_mode="ace_working_memory")
        assert {r["mode"] for r in rows} == {"ace_working_memory"}


class TestEvictionAccounting:
    """A deduplicated lesson is not an addition, so it is not an eviction."""

    LESSON = "- For ATP questions, name the organelle performing oxidative phosphorylation.\n"

    def test_repeated_lesson_is_not_counted_as_an_eviction(self, stub_generation, tmp_path):
        _, summary = run_ace(
            stub_generation,
            tmp_path,
            generator_reply="Answer:\nwrong",
            reflector_reply=self.LESSON,
            curator_reply="Lesson 1: is_generic=False",
        )
        log = summary["playbook_log"]
        assert log[0]["entries_added"] == 1, "first step stores the lesson"
        assert log[1]["entries_added"] == 0, "second step deduplicates into it"
        assert log[1]["num_evictions"] == 0, "deduplication is not an eviction"


class TestStreaming:
    """Rows must reach the caller as they are produced, not only at the end."""

    def test_on_result_fires_per_example(self, stub_generation, tmp_path):
        stub_generation("Answer:\nmitochondria")
        seen = []
        results, _ = run_dataset_baseline(
            model=None,
            tokenizer=StubTokenizer(),
            dataset=EXAMPLES,
            domain="science",
            config=CONFIG,
            model_id="stub",
            task_name="iot_tiny",
            on_result=seen.append,
        )
        assert [r["qid"] for r in seen] == [r["qid"] for r in results]
