"""Tests for the grid plan and the screening gate.

The plan decides two things that cost real money or real validity: which cells
reuse a loaded model, and which arms borrow a playbook rather than re-adapting.
A borrowing arm that cannot see the lessons it borrowed is the dangerous case --
it produces a complete, plausible result that is silently the control arm.
"""

import json


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
