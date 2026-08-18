"""Tests for resuming an interrupted run and skipping completed grid cells.

Predictions were buffered in memory and written once, at the end, so a run that
died at example 900 of 1000 lost all 900 -- and the grid re-ran every cell on
every invocation, which combines badly with a run that dies late.
"""

import json

import pytest

from scripts.run_eval_grid import completed_cell
from scripts.run_experiment import completed_ids, recompute_correctness


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


class TestCompletedIds:
    def test_reads_back_finished_ids(self, tmp_path):
        path = write_jsonl(tmp_path / "p.jsonl", [{"qid": "q1"}, {"qid": "q2"}])
        assert completed_ids(path) == {"q1", "q2"}

    def test_missing_file_is_empty(self, tmp_path):
        assert completed_ids(tmp_path / "absent.jsonl") == set()

    def test_no_path_is_empty(self):
        assert completed_ids(None) == set()

    def test_stops_at_a_torn_row(self, tmp_path):
        """An interrupted write leaves a partial line; everything after is suspect."""
        path = tmp_path / "p.jsonl"
        path.write_text('{"qid": "q1"}\n{"qid": "q2"}\n{"qid": "q3"', encoding="utf-8")
        assert completed_ids(path) == {"q1", "q2"}


class TestRecomputeCorrectness:
    """Merged rows must not leave metrics.json describing only the tail."""

    ROWS = [
        {"qid": "q1", "is_correct": 1, "oma_correct": 1, "gom": 0.4},
        {"qid": "q2", "is_correct": 0, "oma_correct": 0, "gom": -0.2},
        {"qid": "q3", "is_correct": 1, "oma_correct": 1, "gom": 0.6},
        {"qid": "q4", "is_correct": 0, "oma_correct": 0, "gom": 0.0},
    ]

    def test_accuracy_covers_every_row(self):
        tail_only = {"accuracy": 0.0, "num_examples": 2}
        merged = recompute_correctness(tail_only, self.ROWS, resumed=2)
        assert merged["accuracy"] == 0.5
        assert merged["num_examples"] == 4

    def test_oma_and_gom_are_recomputed(self):
        merged = recompute_correctness({}, self.ROWS, resumed=2)
        assert merged["oma_accuracy"] == 0.5
        assert merged["avg_gom"] == pytest.approx(0.2)
        assert merged["oma_ci"]["n"] == 4

    def test_timing_coverage_is_reported_not_assumed(self):
        """Latency measures this process, so say how many examples it covers."""
        merged = recompute_correctness({"avg_latency_ms": 12.0}, self.ROWS, resumed=3)
        assert merged["resumed_examples"] == 3
        assert merged["latency_covers_examples"] == 1
        assert merged["avg_latency_ms"] == 12.0, "timing must not be silently rescaled"


class TestCompletedCell:
    COMMIT = "abc123"

    SEED = 42

    def complete(self, tmp_path, commit=COMMIT, seed=SEED):
        (tmp_path / "results.csv").write_text("qid\nq1\n", encoding="utf-8")
        write_jsonl(tmp_path / "predictions.jsonl", [{"qid": "q1"}])
        (tmp_path / "metrics.json").write_text(
            json.dumps({"seed": seed, "environment": {"git_commit": commit}}), encoding="utf-8"
        )
        return tmp_path

    def test_empty_directory_is_not_complete(self, tmp_path):
        assert completed_cell(tmp_path, self.COMMIT, self.SEED) is False

    def test_all_three_artefacts_from_this_commit(self, tmp_path):
        assert completed_cell(self.complete(tmp_path), self.COMMIT, self.SEED) is True

    def test_a_different_commit_is_not_a_result_for_this_one(self, tmp_path):
        assert completed_cell(self.complete(tmp_path, "deadbeef"), self.COMMIT, self.SEED) is False

    def test_truncated_metrics_are_not_a_result(self, tmp_path):
        cell = self.complete(tmp_path)
        (cell / "metrics.json").write_text('{"trunc', encoding="utf-8")
        assert completed_cell(cell, self.COMMIT, self.SEED) is False

    def test_a_missing_artefact_is_not_a_result(self, tmp_path):
        cell = self.complete(tmp_path)
        (cell / "predictions.jsonl").unlink()
        assert completed_cell(cell, self.COMMIT, self.SEED) is False

    def test_empty_artefact_is_not_a_result(self, tmp_path):
        cell = self.complete(tmp_path)
        (cell / "results.csv").write_text("", encoding="utf-8")
        assert completed_cell(cell, self.COMMIT, self.SEED) is False

    def test_a_different_seed_is_not_a_result_for_this_one(self, tmp_path):
        """docs/evaluation.md asks for >=3 seeds, and the layout has no seed
        segment -- skipping on commit alone silently no-op'd every seed after
        the first and left the earlier seed's numbers in place."""
        cell = self.complete(tmp_path, seed=42)
        assert completed_cell(cell, self.COMMIT, 43) is False
        assert completed_cell(cell, self.COMMIT, 42) is True
