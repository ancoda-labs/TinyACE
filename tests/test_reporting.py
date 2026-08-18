"""Tests for the results-reporting layer: schema, loading and aggregation."""

import json

import pandas as pd
import pytest

from edge_slm_ace.reporting import (
    ABLATION_REFERENCE,
    ARMS,
    arm_label,
    arm_order,
    is_ablation,
    load_predictions,
    load_run_metrics,
    model_label,
    normalize_columns,
    reference_for,
    summarize_predictions,
)


@pytest.fixture
def results_tree(tmp_path):
    """A minimal results directory in the layout the runners produce."""

    def write_run(model, task, arm, device, n_correct, n=50):
        run_dir = tmp_path / model / task / arm / device
        run_dir.mkdir(parents=True)

        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "model_id": "microsoft/Phi-3-mini-4k-instruct",
                    "task_name": task,
                    "mode": "ace" if arm.startswith("tinyace") else arm,
                    "num_examples": n,
                    "oma_accuracy": n_correct / n,
                    "seed": 42,
                    "truncation_rate": 0.0,
                    "chat_template_rate": 1.0,
                }
            )
        )

        with open(run_dir / "predictions.jsonl", "w") as f:
            for i in range(n):
                f.write(
                    json.dumps(
                        {
                            "qid": f"q{i}",
                            "model": "microsoft/Phi-3-mini-4k-instruct",
                            "task": task,
                            "oma_correct": 1 if i < n_correct else 0,
                            "latency_ms": 100.0,
                        }
                    )
                    + "\n"
                )

    write_run("phi_3_mini", "sciq_test", "baseline", "cuda", 37)
    write_run("phi_3_mini", "sciq_test", "tinyace_wm_256", "cuda", 36)
    return tmp_path


class TestArmRegistry:
    """One vocabulary, so the same run cannot appear under two names."""

    def test_every_arm_has_a_label_and_family(self):
        for arm in ARMS:
            assert arm.label and arm.family

    def test_arm_keys_are_unique(self):
        keys = [arm.key for arm in ARMS]
        assert len(keys) == len(set(keys))

    def test_unknown_arms_get_a_readable_fallback(self):
        assert arm_label("some_new_arm") == "Some New Arm"

    def test_unknown_arms_sort_last(self):
        assert arm_order("some_new_arm") > arm_order("baseline")

    def test_ace_arms_reference_the_control_not_the_baseline(self):
        """The playbook claim is ace - cot_control."""
        assert reference_for("ace_full") == "cot_control"
        assert reference_for("tinyace_wm_512") == "cot_control"

    def test_ablations_reference_full_tinyace(self):
        """Comparing an ablation to baseline measures ACE plus the ablation."""
        for arm in (
            "tinyace_ablate_no_failure",
            "tinyace_ablate_no_recency",
            "tinyace_ablate_no_vagueness",
            "tinyace_fifo",
        ):
            assert (
                reference_for(arm) == ABLATION_REFERENCE
            ), f"{arm} must be compared against full TinyACE"

    def test_ablations_are_identified(self):
        assert is_ablation("tinyace_ablate_no_curator")
        assert is_ablation("tinyace_fifo")
        assert not is_ablation("baseline")
        assert not is_ablation("cot_control")


class TestModelLabels:
    def test_known_models_get_short_names(self):
        assert model_label("microsoft/Phi-3-mini-4k-instruct") == "Phi-3-mini"
        assert model_label("Qwen/Qwen2.5-3B-Instruct") == "Qwen2.5-3B"
        assert model_label("TinyLlama/TinyLlama-1.1B-Chat-v1.0") == "TinyLlama-1.1B"

    def test_specific_patterns_win_over_general_ones(self):
        """'phi-3-mini' must not be shadowed by a broader 'phi-3' match."""
        assert model_label("microsoft/Phi-3-mini-4k-instruct") == "Phi-3-mini"

    def test_unknown_models_fall_back_to_the_last_path_segment(self):
        assert model_label("someorg/some-new-model") == "some-new-model"


class TestNormalizeColumns:
    def test_legacy_names_are_mapped_forward(self):
        df = pd.DataFrame([{"sample_id": "q1", "task_name": "t", "model_id": "m", "correct": 1}])
        out = normalize_columns(df)
        assert set(out.columns) == {"qid", "task", "model", "is_correct"}

    def test_canonical_names_win_when_both_exist(self):
        df = pd.DataFrame([{"sample_id": "legacy", "qid": "canonical"}])
        out = normalize_columns(df)
        assert out["qid"].iloc[0] == "canonical"
        assert "sample_id" not in out.columns

    def test_frames_without_legacy_columns_are_unchanged(self):
        df = pd.DataFrame([{"qid": "q1", "is_correct": 1}])
        assert list(normalize_columns(df).columns) == ["qid", "is_correct"]


class TestLoading:
    def test_loads_every_run(self, results_tree):
        runs = load_run_metrics(results_tree)
        assert len(runs) == 2
        assert set(runs["arm"]) == {"baseline", "tinyace_wm_256"}

    def test_arm_is_recovered_from_the_directory_layout(self, results_tree):
        runs = load_run_metrics(results_tree)
        assert "TinyACE-256" in set(runs["arm_label"])

    def test_loads_every_prediction_row(self, results_tree):
        predictions = load_predictions(results_tree)
        assert len(predictions) == 100
        assert set(predictions["arm"]) == {"baseline", "tinyace_wm_256"}

    def test_missing_root_yields_empty_frames(self, tmp_path):
        assert load_run_metrics(tmp_path / "nope").empty
        assert load_predictions(tmp_path / "nope").empty

    def test_malformed_files_are_skipped_not_fatal(self, results_tree):
        bad = results_tree / "broken" / "sciq_test" / "baseline" / "cuda"
        bad.mkdir(parents=True)
        (bad / "metrics.json").write_text("{not valid json")

        runs = load_run_metrics(results_tree)
        assert len(runs) == 2, "The two good runs should still load"


class TestSummarize:
    def test_reports_accuracy_with_an_interval(self, results_tree):
        summary = summarize_predictions(load_predictions(results_tree))
        row = summary[summary["arm"] == "baseline"].iloc[0]

        assert row["n"] == 50
        assert row["accuracy"] == pytest.approx(0.74)
        assert row["ci_low"] < row["accuracy"] < row["ci_high"]

    def test_interval_halfwidth_dwarfs_the_arm_difference(self, results_tree):
        """The point of shipping the interval: 1 question is not a result."""
        summary = summarize_predictions(load_predictions(results_tree))
        accuracies = summary.set_index("arm")["accuracy"]
        difference = abs(accuracies["baseline"] - accuracies["tinyace_wm_256"])

        assert difference < summary["ci_halfwidth"].min()

    def test_prefers_oma_over_exact_match(self, results_tree):
        summary = summarize_predictions(load_predictions(results_tree))
        assert set(summary["metric"]) == {"oma_correct"}

    def test_falls_back_to_exact_match_without_oma(self):
        df = pd.DataFrame([{"qid": f"q{i}", "arm": "baseline", "is_correct": 1} for i in range(4)])
        summary = summarize_predictions(df, group_by=["arm"])
        assert summary["metric"].iloc[0] == "is_correct"

    def test_empty_input_yields_empty_output(self):
        assert summarize_predictions(pd.DataFrame()).empty
