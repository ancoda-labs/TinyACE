"""Tests for the grid plan and the screening gate.

The plan decides two things that cost real money or real validity: which cells
reuse a loaded model, and which arms borrow a playbook rather than re-adapting.
A borrowing arm that cannot see the lessons it borrowed is the dangerous case --
it produces a complete, plausible result that is silently the control arm.
"""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from edge_slm_ace.memory.playbook import Playbook
from edge_slm_ace.adapt import frozen_lessons
from edge_slm_ace.reporting import get_arm
from edge_slm_ace.utils import CHANCE_FLOOR, SCREENING_FLOOR, SCREENING_N, screening_verdict
from scripts.run_grid import (
    GRID,
    cell_dir,
    is_complete,
    order_languages,
    plan,
    source_languages,
    survivors,
)


class TestGridPlan:
    def test_every_arm_is_registered(self):
        """An unregistered arm has no label and no reference arm."""
        for arm in GRID:
            assert get_arm(arm.key) is not None, f"{arm.key} is not in reporting/schema.py"

    def test_adapting_arms_run_before_borrowing_arms(self):
        jobs = plan(["m"], ["en"], GRID)
        keys = [j["arm"].key for j in jobs]
        for job in jobs:
            if job["arm"].playbook_from:
                source = job["arm"].playbook_from[0]
                assert keys.index(source) < keys.index(job["arm"].key)

    def test_cells_are_grouped_by_model(self):
        """A model is loaded once; interleaving models would reload it."""
        jobs = plan(["a", "b"], ["en", "ne"], GRID)
        models = [j["model"] for j in jobs]
        assert models == sorted(models, key=["a", "b"].index)

    def test_a_language_restricted_arm_is_not_planned_everywhere(self):
        """The printed cell count used to be models x languages x arms.

        `only_languages` means an arm is planned in some cells and not others,
        so that product overstates the grid and disagrees with the count the
        same run prints when it finishes.
        """
        jobs = plan(["m1"], ["en", "ne"], GRID)
        assert len(jobs) < 1 * 2 * len(GRID)

    def test_cross_lingual_arm_is_scoped_to_nepali(self):
        """On English it would borrow the English playbook to score English."""
        jobs = plan(["m"], ["en", "ne"], GRID)
        langs = {j["language"] for j in jobs if j["arm"].key == "tinyace_playbook_en"}
        assert langs == {"ne"}

    def test_borrowed_playbook_arms_name_their_source_domain(self):
        """
        The regression: the cross-lingual arm loaded the English playbook and
        retrieved under the Nepali domain, so no lesson matched and the arm
        silently became scaffold_control while carrying an ACE label.
        """
        arm = next(a for a in GRID if a.key == "tinyace_playbook_en")
        assert "--playbook-domain" in arm.flags
        domain = arm.flags[arm.flags.index("--playbook-domain") + 1]
        assert domain == "belebele_en", "must match the language it was adapted on"


class TestDomainMismatchIsVisible:
    def test_lessons_are_invisible_across_domains(self):
        """The mechanism behind the regression, asserted directly."""
        playbook = Playbook()
        playbook.add_entry("belebele_en", "Prefer what the passage states.", step=1)
        assert frozen_lessons(playbook, "belebele_en") != []
        assert frozen_lessons(playbook, "belebele_ne") == []


class TestCellCompletion:
    def write(self, directory, seed=42, commit="abc123"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "predictions.jsonl").write_text('{"qid":"q1"}\n', encoding="utf-8")
        (directory / "metrics.json").write_text(
            json.dumps({"seed": seed, "environment": {"git_commit": commit}}), encoding="utf-8"
        )
        return directory

    def test_complete_cell_is_skipped(self, tmp_path):
        assert is_complete(self.write(tmp_path), 42, "abc123") is True

    def test_a_different_seed_is_not_a_result(self, tmp_path):
        """Multi-seed studies share a layout; skipping on commit alone loses them."""
        assert is_complete(self.write(tmp_path, seed=42), 43, "abc123") is False

    def test_a_different_commit_is_not_a_result(self, tmp_path):
        assert is_complete(self.write(tmp_path), 42, "deadbeef") is False

    def test_truncated_metrics_are_not_a_result(self, tmp_path):
        cell = self.write(tmp_path)
        (cell / "metrics.json").write_text('{"trunc', encoding="utf-8")
        assert is_complete(cell, 42, "abc123") is False

    def test_missing_artefact_is_not_a_result(self, tmp_path):
        cell = self.write(tmp_path)
        (cell / "predictions.jsonl").unlink()
        assert is_complete(cell, 42, "abc123") is False

    def test_layout_is_model_language_arm(self, tmp_path):
        assert cell_dir(tmp_path, "qwen3-4b", "ne", "tinyace").parts[-3:] == (
            "qwen3-4b",
            "ne",
            "tinyace",
        )


class TestScreeningRule:
    def test_floor_is_above_chance(self):
        assert SCREENING_FLOOR > CHANCE_FLOOR

    def test_rule_is_on_the_interval_not_the_point_estimate(self):
        """An observed 35% at n=200 is consistent with a true 28%."""
        assert screening_verdict(0.31) is True
        assert screening_verdict(0.30) is False
        assert screening_verdict(0.28) is False

    def test_screening_n_is_large_enough_to_resolve_the_floor(self):
        """The interval at the floor must not straddle chance."""
        from edge_slm_ace.eval.stats import wilson_interval

        low, _ = wilson_interval(int(0.35 * SCREENING_N), SCREENING_N)
        assert low > CHANCE_FLOOR, "a 35% model must be distinguishable from chance"

    def test_week1_lineup_is_qwen3_and_gemma3_on_both_tasks(self):
        from edge_slm_ace.utils import SCREENING_MODELS, SCREENING_TASKS

        assert SCREENING_MODELS == (
            "qwen3-1.7b",
            "qwen3-4b",
            "gemma-3-1b",
            "gemma-3-4b",
        )
        assert SCREENING_TASKS == ("belebele", "global_mmlu")


class TestScreeningSurvivors:
    """
    The grid admits models from the Nepali Global-MMLU screen only.

    `screening.json` used to hold every language. An English screen written
    afterwards replaced the Nepali result, and a Belebele pass was enough to
    enter a grid whose primary task is Global-MMLU.
    """

    def write(
        self,
        root,
        benchmark,
        language,
        rows,
        *,
        payload_language=None,
        payload_benchmark=None,
    ):
        directory = root / "screening"
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": benchmark if payload_benchmark is None else payload_benchmark,
            "language": language if payload_language is None else payload_language,
            "models": rows,
        }
        (directory / f"screening_{benchmark}_{language}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return root

    def test_an_english_screen_cannot_change_nepali_survivors(self, tmp_path):
        """Written second, the English file used to be the only file."""
        self.write(
            tmp_path,
            "global_mmlu",
            "ne",
            [
                {"model": "kept", "task": "global_mmlu", "passes": True},
                {"model": "out", "task": "global_mmlu", "passes": False},
            ],
        )
        before = survivors(tmp_path)
        self.write(
            tmp_path,
            "global_mmlu",
            "en",
            [
                {"model": "kept", "task": "global_mmlu", "passes": False},
                {"model": "english-only", "task": "global_mmlu", "passes": True},
            ],
        )
        assert survivors(tmp_path) == before == ["kept"]

    def test_a_payload_with_the_wrong_language_raises(self, tmp_path):
        """The filename is not enough: the body has to say Nepali too."""
        self.write(
            tmp_path,
            "global_mmlu",
            "ne",
            [{"model": "a", "task": "global_mmlu", "passes": True}],
            payload_language="en",
        )
        with pytest.raises(ValueError, match="language"):
            survivors(tmp_path)

    def test_a_payload_missing_its_language_raises(self, tmp_path):
        directory = tmp_path / "screening"
        directory.mkdir()
        (directory / "screening_global_mmlu_ne.json").write_text(
            json.dumps(
                {
                    "benchmark": "global_mmlu",
                    "models": [{"model": "a", "task": "global_mmlu", "passes": True}],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="language"):
            survivors(tmp_path)

    def test_a_payload_with_the_wrong_benchmark_raises(self, tmp_path):
        self.write(
            tmp_path,
            "global_mmlu",
            "ne",
            [{"model": "a", "task": "global_mmlu", "passes": True}],
            payload_benchmark="belebele",
        )
        with pytest.raises(ValueError, match="benchmark"):
            survivors(tmp_path)

    def test_belebele_only_passes_do_not_admit_a_model(self, tmp_path):
        """A Belebele pass is a different task. There is no fallback to it."""
        self.write(
            tmp_path,
            "belebele",
            "ne",
            [{"model": "belebele-only", "task": "belebele", "passes": True}],
        )
        with pytest.raises(FileNotFoundError):
            survivors(tmp_path)

        self.write(
            tmp_path,
            "global_mmlu",
            "ne",
            [
                {"model": "belebele-only", "task": "global_mmlu", "passes": False},
                {"model": "belebele-row", "task": "belebele", "passes": True},
                {"model": "admitted", "task": "global_mmlu", "passes": True},
            ],
        )
        assert survivors(tmp_path) == ["admitted"]

    def test_a_model_is_listed_once(self, tmp_path):
        self.write(
            tmp_path,
            "global_mmlu",
            "ne",
            [
                {"model": "a", "task": "global_mmlu", "passes": True},
                {"model": "a", "task": "global_mmlu", "passes": True},
            ],
        )
        assert survivors(tmp_path) == ["a"]

    def test_a_screen_that_passes_nobody_is_empty(self, tmp_path):
        """An empty list means the screen ran. A missing file is a raise."""
        self.write(
            tmp_path,
            "global_mmlu",
            "ne",
            [{"model": "a", "task": "global_mmlu", "passes": False}],
        )
        assert survivors(tmp_path) == []

    def test_a_missing_screen_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            survivors(tmp_path)

    def test_the_old_screening_json_is_ignored(self, tmp_path):
        """It has no language in the name, so it cannot be the Nepali gate."""
        directory = tmp_path / "screening"
        directory.mkdir()
        (directory / "screening.json").write_text(
            json.dumps(
                {
                    "language": "en",
                    "models": [{"model": "legacy", "task": "belebele", "passes": True}],
                }
            ),
            encoding="utf-8",
        )
        self.write(
            tmp_path,
            "global_mmlu",
            "ne",
            [{"model": "kept", "task": "global_mmlu", "passes": True}],
        )
        with pytest.warns(UserWarning, match=r"screening\.json"):
            assert survivors(tmp_path) == ["kept"]

    def test_the_old_file_alone_does_not_admit_anyone(self, tmp_path):
        directory = tmp_path / "screening"
        directory.mkdir()
        (directory / "screening.json").write_text(
            json.dumps(
                {"language": "ne", "models": [{"model": "a", "task": "belebele", "passes": True}]}
            ),
            encoding="utf-8",
        )
        with pytest.warns(UserWarning, match=r"screening\.json"):
            with pytest.raises(FileNotFoundError):
                survivors(tmp_path)


class TestScreenWritesOneFilePerTaskAndLanguage:
    """Each task and language keeps the manifest and the git state it ran under."""

    def run(self, tmp_path, monkeypatch, language, tasks):
        from scripts import screen

        monkeypatch.setattr(
            screen,
            "run_frozen_eval",
            lambda *a, **k: {
                "tinyace": {
                    "truncation_rate": 0.0,
                    "tokens_dropped": 0,
                    "truncated_prompts": 0,
                }
            },
        )

        def fake_items(results, language, expected_ids=None, **kwargs):
            return {item_id: 1 for item_id in expected_ids}

        monkeypatch.setattr(screen, "per_item_correctness", fake_items)
        assert (
            screen.main(
                [
                    "--models",
                    "tiny-gpt2",
                    "--tasks",
                    *tasks,
                    "--language",
                    language,
                    "--results-root",
                    str(tmp_path),
                    "--n",
                    "2",
                ]
            )
            == 0
        )

    def test_each_task_and_language_gets_its_own_file(self, tmp_path, monkeypatch):
        from edge_slm_ace.data.splits import BELEBELE_SPLIT, GLOBAL_MMLU_ADAPT, load_manifest
        from edge_slm_ace.utils import capture_environment

        self.run(tmp_path, monkeypatch, "ne", ["belebele", "global_mmlu"])
        self.run(tmp_path, monkeypatch, "en", ["global_mmlu"])

        screening = tmp_path / "screening"
        assert not (screening / "screening.json").exists()
        ne_mmlu = json.loads(
            (screening / "screening_global_mmlu_ne.json").read_text(encoding="utf-8")
        )
        en_mmlu = json.loads(
            (screening / "screening_global_mmlu_en.json").read_text(encoding="utf-8")
        )
        ne_bel = json.loads((screening / "screening_belebele_ne.json").read_text(encoding="utf-8"))

        adapt = load_manifest(GLOBAL_MMLU_ADAPT)
        belebele = load_manifest(BELEBELE_SPLIT)
        env = capture_environment()

        assert ne_mmlu["benchmark"] == "global_mmlu"
        assert ne_mmlu["language"] == "ne"
        assert ne_mmlu["tasks"] == ["global_mmlu"]
        assert ne_mmlu["manifest_name"] == adapt["name"]
        assert ne_mmlu["manifest_checksum"] == adapt["ids_sha256"]
        assert {row["task"] for row in ne_mmlu["models"]} == {"global_mmlu"}
        assert en_mmlu["language"] == "en"
        assert en_mmlu["benchmark"] == "global_mmlu"
        # Both splits share belebele_split.json. Screening reads the adaptation
        # ids, so the checksum that covers them is adaptation_ids_sha256.
        assert ne_bel["benchmark"] == "belebele"
        assert ne_bel["manifest_name"] == belebele["name"]
        assert ne_bel["manifest_checksum"] == belebele["adaptation_ids_sha256"]
        assert ne_bel["manifest_checksum"] != belebele["ids_sha256"]
        assert {row["task"] for row in ne_bel["models"]} == {"belebele"}
        for key in ("n", "seed", "rule", "models", "environment"):
            assert key in ne_mmlu
        assert ne_mmlu["environment"]["git_commit"] == env["git_commit"]
        assert ne_mmlu["environment"]["git_dirty"] == env["git_dirty"]


class TestLanguageOrdering:
    """
    Ordering the arms was not enough; the languages need ordering too.

    `tinyace_playbook_en` borrows English specifically. On `--languages ne en`
    Nepali ran first and the arm failed on a missing path, and on
    `--languages ne` the source was never going to exist at all -- both silently,
    one cell at a time, after a model had been loaded.
    """

    def test_a_borrowed_from_language_runs_first(self):
        ordered = order_languages(["ne", "en"], GRID)
        assert ordered.index("en") < ordered.index("ne")

    def test_the_requested_order_is_otherwise_kept(self):
        assert order_languages(["en", "ne"], GRID) == ["en", "ne"]

    def test_the_source_cell_is_planned_before_the_borrower(self):
        jobs = plan(["m"], ["ne", "en"], GRID)
        order = [(j["language"], j["arm"].key) for j in jobs]

        assert order.index(("en", "tinyace")) < order.index(("ne", "tinyace_playbook_en"))

    def test_the_grid_names_the_language_it_borrows_from(self):
        assert source_languages(GRID) == ["en"]


class TestUnscorableRunWritesNothing:
    """A run the harness reported no accuracy for must not leave a cell behind.

    `metrics.json` and `predictions.jsonl` were written before the accuracy was
    formatted for the console, so a run whose task no longer reports `acc` died
    on the format string with both artefacts already on disk. `is_complete`
    reads that as a finished result for this commit and seed, so the rerun
    skipped the cell and it stayed empty for the rest of the study.
    """

    def run(self, tmp_path, monkeypatch):
        from scripts import run_arm

        # The harness ran and logged samples; it just no longer reports `acc`,
        # which is what a renamed metric upstream looks like from here.
        results = {
            "results": {"belebele_eng_Latn": {"acc_norm,none": 0.5}},
            "tinyace": {"truncated_prompts": 0, "truncation_rate": 0.0, "tokens_dropped": 0},
        }
        monkeypatch.setattr(run_arm, "run_frozen_eval", lambda *a, **k: results)
        monkeypatch.setattr(run_arm, "per_item_correctness", lambda *a, **k: {"q1": 1, "q2": 0})
        return run_arm.main(
            [
                "--model",
                "tiny-gpt2",
                "--language",
                "en",
                "--arm",
                "scaffold_control",
                "--output-dir",
                str(tmp_path),
                "--limit",
                "2",
            ]
        )

    def test_a_missing_accuracy_fails_the_run(self, tmp_path, monkeypatch):
        assert self.run(tmp_path, monkeypatch) == 1

    def test_a_missing_accuracy_leaves_no_cell(self, tmp_path, monkeypatch):
        self.run(tmp_path, monkeypatch)
        assert not (tmp_path / "metrics.json").exists()
        assert not (tmp_path / "predictions.jsonl").exists()
        assert is_complete(tmp_path, 42, None) is False


class TestPlaybookGetsARealTokenizer:
    """run_arm must hand the playbook the tokenizer of the model it ran.

    Nothing passed one, so entry token counts were always `words * 1.3` --
    fertility-blind, and therefore blind to the effect the study is about.
    """

    def picker(self):
        from scripts.run_arm import playbook_tokenizer

        return playbook_tokenizer

    def test_it_takes_the_tokenizer_off_a_reused_model(self):
        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [0]

        class LM:
            tokenizer = Tok()

        lm = LM()
        assert self.picker()(lm, None) is lm.tokenizer

    def test_it_falls_back_to_the_scorers_model(self):
        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [0]

        class LM:
            tokenizer = Tok()

        class Scorer:
            lm = LM()

        scorer = Scorer()
        assert self.picker()(None, scorer) is scorer.lm.tokenizer

    def test_no_model_means_no_tokenizer_rather_than_a_broken_one(self):
        assert self.picker()(None, None) is None

    def test_an_object_without_encode_is_not_accepted(self):
        class LM:
            tokenizer = object()

        assert self.picker()(LM(), None) is None


def _ordered_id_sha256(item_ids) -> str:
    """Order-sensitive. ids_checksum sorts, which would hide a seed change."""
    return hashlib.sha256("\n".join(item_ids).encode("utf-8")).hexdigest()


class TestRunArmReadsTheFrozenSplit:
    """The adaptation and evaluation sets come from the manifest.

    The seed orders the adaptation walk and nothing else. Two seeds that
    selected different items would make the multi-seed grid a comparison of
    different questions.
    """

    def run(self, tmp_path, monkeypatch, seed, limit=None, extra=None):
        from scripts import run_arm

        captured = {}

        class Scorer:
            def __init__(self, *args, **kwargs):
                pass

        def fake_adapt(scorer, examples, playbook, **kwargs):
            captured["adapt"] = [example["id"] for example in examples]
            captured["adapt_kwargs"] = kwargs
            return {"accuracy": 0.0, "playbook_size": 0, "log": []}

        def fake_eval(model_id, language, item_ids, **kwargs):
            captured["eval"] = list(item_ids)
            return {
                "results": {"tinyace_belebele_eng": {"acc,none": 0.25}},
                "tinyace": {
                    "truncated_prompts": 0,
                    "truncation_rate": 0.0,
                    "tokens_dropped": 0,
                },
            }

        def fake_items(results, language, expected_ids=None, **kwargs):
            return {item_id: 1 for item_id in expected_ids}

        class Rel:
            available = False
            encoder_name = None

        monkeypatch.setattr(run_arm, "OptionScorer", Scorer)
        monkeypatch.setattr(run_arm, "adapt_playbook", fake_adapt)
        monkeypatch.setattr(run_arm, "run_frozen_eval", fake_eval)
        monkeypatch.setattr(run_arm, "per_item_correctness", fake_items)
        monkeypatch.setattr(run_arm.LessonRelevance, "get_instance", classmethod(lambda cls: Rel()))

        argv = [
            "--model",
            "tiny-gpt2",
            "--language",
            "en",
            "--arm",
            "tinyace",
            "--output-dir",
            str(tmp_path),
            "--seed",
            str(seed),
            "--no-chat-template",
        ]
        if limit is not None:
            argv += ["--limit", str(limit)]
        if extra:
            argv += extra
        assert run_arm.main(argv) == 0
        captured["metrics"] = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        return captured

    def test_two_seeds_share_the_manifest_sets(self, tmp_path, monkeypatch):
        from edge_slm_ace.data.splits import frozen_split

        adapt, evaluate = frozen_split("belebele")
        first = self.run(tmp_path / "s42", monkeypatch, seed=42)
        second = self.run(tmp_path / "s43", monkeypatch, seed=43)

        assert set(first["adapt"]) == set(adapt)
        assert set(second["adapt"]) == set(adapt)
        assert first["eval"] == list(evaluate)
        assert second["eval"] == list(evaluate)
        assert first["adapt"] != second["adapt"]
        assert first["metrics"]["adaptation_order_seed"] == 42
        assert second["metrics"]["adaptation_order_seed"] == 43
        assert first["metrics"]["adaptation_order_sha256"] == _ordered_id_sha256(first["adapt"])
        assert second["metrics"]["adaptation_order_sha256"] == _ordered_id_sha256(second["adapt"])
        assert (
            first["metrics"]["adaptation_order_sha256"]
            != second["metrics"]["adaptation_order_sha256"]
        )

    def test_the_same_seed_walks_adaptation_in_the_same_order(self, tmp_path, monkeypatch):
        first = self.run(tmp_path / "a", monkeypatch, seed=42)
        second = self.run(tmp_path / "b", monkeypatch, seed=42)
        assert first["adapt"] == second["adapt"]
        assert first["eval"] == second["eval"]

    def test_adaptation_is_called_with_a_real_manifest_name(self, tmp_path, monkeypatch):
        """The leak guard used to be skipped because this call omitted it."""
        from edge_slm_ace.data.splits import MANIFESTS, eval_manifest_for

        captured = self.run(tmp_path, monkeypatch, seed=42)
        name = captured["adapt_kwargs"].get("eval_manifest")
        assert name == eval_manifest_for("belebele")
        assert name in MANIFESTS

    def test_limit_keeps_the_same_prefix_in_different_orders(self, tmp_path, monkeypatch):
        from edge_slm_ace.data.splits import frozen_split

        adapt, evaluate = frozen_split("belebele")
        first = self.run(tmp_path / "s42", monkeypatch, seed=42, limit=4)
        second = self.run(tmp_path / "s43", monkeypatch, seed=43, limit=4)
        assert set(first["adapt"]) == set(adapt[:4])
        assert set(second["adapt"]) == set(adapt[:4])
        assert first["adapt"] != second["adapt"]
        assert first["eval"] == list(evaluate[:4])
        assert second["eval"] == first["eval"]

    def test_metrics_record_the_run_settings(self, tmp_path, monkeypatch):
        """A cell must say which split, pin and flags produced it.

        top_k, the curator switch and --limit used to be absent, so a debug
        prefix and a full run were the same file apart from n_eval.
        """
        from edge_slm_ace.data.belebele import BELEBELE_REVISION
        from edge_slm_ace.data.globalmmlu import GLOBAL_MMLU_REVISION
        from edge_slm_ace.data.splits import (
            BELEBELE_SPLIT,
            frozen_split,
            ids_checksum,
            load_manifest,
        )
        from edge_slm_ace.utils import resolve_model

        captured = self.run(tmp_path, monkeypatch, seed=42)
        metrics = captured["metrics"]
        adapt, _evaluate = frozen_split("belebele")
        manifest = load_manifest(BELEBELE_SPLIT)

        for field in (
            "benchmark",
            "budget",
            "top_k",
            "no_curator",
            "adaptation_ids_checksum",
            "eval_manifest",
            "eval_manifest_checksum",
            "init_playbook",
            "init_playbook_sha256",
            "model_revision",
            "dataset_revisions",
            "limit",
            "debug",
            "enable_thinking",
        ):
            assert field in metrics, field

        assert metrics["benchmark"] == "belebele"
        assert metrics["budget"] is None
        assert metrics["top_k"] == 5
        assert metrics["no_curator"] is False
        assert metrics["adaptation_ids_checksum"] == ids_checksum(adapt)
        assert metrics["eval_manifest"] == BELEBELE_SPLIT
        assert metrics["eval_manifest_checksum"] == manifest["ids_sha256"]
        assert metrics["init_playbook"] is None
        assert metrics["init_playbook_sha256"] is None
        assert metrics["model_revision"] == resolve_model("tiny-gpt2").revision
        assert metrics["dataset_revisions"] == {
            "facebook/belebele": BELEBELE_REVISION,
            "CohereForAI/Global-MMLU": GLOBAL_MMLU_REVISION,
        }
        assert metrics["limit"] is None
        assert metrics["debug"] is False
        assert metrics["enable_thinking"] is None
        for field in ("lm_eval_version", "nv_tegra_release", "nvpmodel"):
            assert field in metrics["environment"], field
        assert metrics["environment"]["lm_eval_version"]

    def test_an_init_playbook_is_recorded_with_its_digest(self, tmp_path, monkeypatch):
        playbook = tmp_path / "borrowed.jsonl"
        playbook.write_text("", encoding="utf-8")
        captured = self.run(
            tmp_path / "cell",
            monkeypatch,
            seed=42,
            extra=["--init-playbook", str(playbook)],
        )
        metrics = captured["metrics"]
        assert metrics["init_playbook"] == str(playbook)
        assert metrics["init_playbook_sha256"] == hashlib.sha256(b"").hexdigest()

    def test_a_limit_run_is_flagged_debug(self, tmp_path, monkeypatch):
        from edge_slm_ace.data.splits import frozen_split, ids_checksum
        from edge_slm_ace.reporting import health_issues, is_reportable

        adapt, _evaluate = frozen_split("belebele")
        captured = self.run(tmp_path, monkeypatch, seed=42, limit=4)
        metrics = captured["metrics"]
        assert metrics["limit"] == 4
        assert metrics["debug"] is True
        assert metrics["adaptation_ids_checksum"] == ids_checksum(adapt[:4])
        issues = [issue for issue in health_issues(metrics) if issue.field == "debug"]
        assert len(issues) == 1
        assert issues[0].invalidating is True
        assert is_reportable(metrics) is False


class TestRunnersDoNotRebuildTheSplit:
    """A runner that calls study_split again selects items with --seed."""

    def test_entry_points_do_not_name_the_split_builders(self):
        forbidden = {"study_split", "parallel_split", "passage_disjoint_split"}
        root = Path(__file__).resolve().parents[1] / "scripts"
        for name in ("run_arm.py", "screen.py", "run_grid.py"):
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            found = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in forbidden:
                    found.add(node.id)
                elif isinstance(node, ast.Attribute) and node.attr in forbidden:
                    found.add(node.attr)
                elif isinstance(node, ast.alias) and (
                    node.name in forbidden or node.asname in forbidden
                ):
                    found.add(node.name)
            assert not found, f"{name} references {sorted(found)}"
