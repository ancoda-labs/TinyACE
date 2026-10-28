"""Pins and thinking-mode must reach every loader, and the jsonl must match them.

ISSUES.md #11 and the dataset-revision half of #14. A run that loads `main`
or leaves Qwen3/SmolLM3 thinking on is a different measurement than the one
the task YAML and the metrics file name.
"""

import re
from pathlib import Path

import pytest

from edge_slm_ace.utils import MODELS, resolve_model

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "src" / "edge_slm_ace" / "eval" / "tasks"

_BELEBELE_YAMLS = ("tinyace_belebele_eng.yaml", "tinyace_belebele_npi.yaml")
_MMLU_YAMLS = ("tinyace_global_mmlu_en.yaml", "tinyace_global_mmlu_ne.yaml")


def _yaml_revision(name: str) -> str:
    text = (TASKS / name).read_text(encoding="utf-8")
    match = re.search(r"(?m)^dataset_kwargs:\n  revision:\s*([0-9a-f]{40})\s*$", text)
    assert match, f"{name} has no dataset_kwargs.revision pin"
    return match.group(1)


def _belebele_row(**overrides):
    row = {
        "link": "https://example.test/passage",
        "question_number": 1,
        "flores_passage": "The passage.",
        "question": "What happened?",
        "mc_answer1": "one",
        "mc_answer2": "two",
        "mc_answer3": "three",
        "mc_answer4": "four",
        "correct_answer_num": "2",
    }
    row.update(overrides)
    return row


class TestDatasetRevisions:
    def test_task_yamls_pin_the_same_commit_as_the_loaders(self):
        from edge_slm_ace.data.belebele import BELEBELE_REVISION
        from edge_slm_ace.data.globalmmlu import GLOBAL_MMLU_REVISION

        for name in _BELEBELE_YAMLS:
            assert _yaml_revision(name) == BELEBELE_REVISION
        for name in _MMLU_YAMLS:
            assert _yaml_revision(name) == GLOBAL_MMLU_REVISION

    def test_globalmmlu_hub_load_passes_the_pin(self, monkeypatch):
        from edge_slm_ace.data import globalmmlu

        captured = {}

        def fake_load(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return []

        import datasets

        monkeypatch.setattr(datasets, "load_dataset", fake_load)
        assert globalmmlu._load_split_from_hub("en", "test") == []
        assert captured["kwargs"]["revision"] == globalmmlu.GLOBAL_MMLU_REVISION
        assert captured["args"][0] == globalmmlu.HF_DATASET

    def test_fetch_belebele_downloads_the_pinned_commit(self):
        from edge_slm_ace.data.belebele import BELEBELE_REVISION
        from scripts.fetch_belebele import SOURCE

        assert BELEBELE_REVISION in SOURCE.format(revision=BELEBELE_REVISION, flores="eng_Latn")
        assert "/resolve/main/" not in SOURCE


class TestThinkingModeReachesEveryLoader:
    """enable_thinking is a ModelSpec field, not a per-call guess.

    Qwen3 and SmolLM3 default to thinking, which changes the prompt from day
    to day and from call to call. Models without the switch stay None, and
    HFLM only forwards the flag when it is not None.
    """

    def test_build_model_args_carries_revision_and_thinking(self):
        from lm_eval.utils import simple_parse_args_string

        from edge_slm_ace.harness.evaluate import build_model_args

        for spec in MODELS.values():
            parsed = simple_parse_args_string(build_model_args(spec.hf_id))
            assert parsed["pretrained"] == spec.hf_id
            assert parsed["revision"] == spec.revision
            assert parsed["enable_thinking"] == spec.enable_thinking

        qwen = resolve_model("qwen3-1.7b")
        assert qwen.enable_thinking is False
        assert resolve_model("qwen3-4b").enable_thinking is False
        # SmolLM3's template stamps "Today Date:" even with thinking off, so
        # the prompt would change every day. It is not in the lineup.
        assert "smollm3-3b" not in MODELS
        for key in ("gemma-3-1b", "gemma-3-4b", "phi-4-mini", "phi-3.5-mini", "tiny-gpt2"):
            assert resolve_model(key).enable_thinking is None

    def test_option_scorer_constructs_hflm_from_the_spec(self, monkeypatch):
        captured = {}

        class FakeHFLM:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import lm_eval.models.huggingface as hf

        monkeypatch.setattr(hf, "HFLM", FakeHFLM)
        from edge_slm_ace.harness.scorer import OptionScorer

        spec = resolve_model("qwen3-1.7b")
        OptionScorer(spec.hf_id, device="cpu", batch_size=2)
        assert captured["pretrained"] == spec.hf_id
        assert captured["revision"] == spec.revision
        assert captured["enable_thinking"] is False

        captured.clear()
        plain = resolve_model("tiny-gpt2")
        OptionScorer(plain.key, device="cpu")
        assert captured["revision"] == plain.revision
        assert captured["enable_thinking"] is None

    def test_run_grid_constructs_hflm_from_the_spec(self, monkeypatch, tmp_path):
        captured = {}

        class FakeHFLM:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import lm_eval.models.huggingface as hf
        import scripts.run_arm as run_arm

        monkeypatch.setattr(hf, "HFLM", FakeHFLM)
        monkeypatch.setattr(run_arm, "main", lambda *args, **kwargs: 0)

        from scripts import run_grid
        from scripts.run_grid import ArmSpec, main

        # One arm. A stubbed run_arm writes no playbook, so a borrowing arm
        # would fail the process for a reason this test is not about.
        monkeypatch.setattr(run_grid, "GRID", [ArmSpec("baseline")])

        spec = resolve_model("qwen3-4b")
        assert (
            main(
                [
                    "--models",
                    spec.key,
                    "--languages",
                    "en",
                    "--results-root",
                    str(tmp_path),
                    "--limit",
                    "1",
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                ]
            )
            == 0
        )
        assert captured["pretrained"] == spec.hf_id
        assert captured["revision"] == spec.revision
        assert captured["enable_thinking"] is False
        assert captured["device"] == "cpu"
        assert captured["dtype"] == "float32"


class TestVerifyTaskData:
    def test_identical_rows_match(self):
        from scripts.verify_task_data import content_mismatches

        rows = [_belebele_row(), _belebele_row(question_number=2, question="Other?")]
        assert content_mismatches(rows, list(rows)) == []

    def test_a_changed_passage_is_reported(self):
        from scripts.verify_task_data import content_mismatches

        local = [_belebele_row()]
        hub = [_belebele_row(flores_passage="A different passage.")]
        problems = content_mismatches(local, hub)
        assert problems
        assert any("flores_passage" in problem for problem in problems)

    def test_a_string_gold_matches_an_int_gold(self):
        """The jsonl stores the numeral as a string; datasets may return an int."""
        from scripts.verify_task_data import content_mismatches

        local = [_belebele_row(correct_answer_num="2")]
        hub = [_belebele_row(correct_answer_num=2)]
        assert content_mismatches(local, hub) == []

    def test_a_missing_hub_row_is_reported(self):
        from scripts.verify_task_data import content_mismatches

        problems = content_mismatches([_belebele_row()], [])
        assert problems

    @pytest.mark.network
    def test_committed_belebele_matches_the_pinned_hub(self):
        from scripts.verify_task_data import main

        assert main([]) == 0


class TestEnvironmentSnapshot:
    def test_capture_environment_names_the_harness_and_the_jetson(self):
        import lm_eval

        from edge_slm_ace.utils.repro import capture_environment

        env = capture_environment()
        assert env["lm_eval_version"] == lm_eval.__version__
        # This checkout is not a Jetson. The keys stay, so the schema does not
        # depend on the machine.
        assert env["nv_tegra_release"] is None
        assert env["nvpmodel"] is None

    def test_a_tegra_release_file_is_recorded(self, tmp_path, monkeypatch):
        from edge_slm_ace.utils import repro

        release = tmp_path / "nv_tegra_release"
        release.write_text("# R36 (release), REVISION: 4.3\n", encoding="utf-8")
        monkeypatch.setattr(repro, "_TEGRA_RELEASE", release)

        def fake_run(cmd, **kwargs):
            assert cmd == ["nvpmodel", "-q"]

            class Out:
                returncode = 0
                stdout = "NV Power Mode: MODE_15W\n"
                stderr = ""

            return Out()

        monkeypatch.setattr(repro.subprocess, "run", fake_run)
        snap = repro._jetson_snapshot()
        assert "R36" in snap["nv_tegra_release"]
        assert snap["nvpmodel"] == "NV Power Mode: MODE_15W"


def test_model_revisions_are_full_commit_shas():
    for spec in MODELS.values():
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision), spec.key
        assert spec.revision != "main"
