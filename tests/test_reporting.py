"""Tests for the results-reporting layer: schema, loading and aggregation."""

import json
from pathlib import Path

import pandas as pd
import pytest

from edge_slm_ace.reporting import (
    ABLATION_REFERENCE,
    ARMS,
    Cell,
    arm_label,
    arm_order,
    cell_dir,
    get_arm,
    is_ablation,
    load_predictions,
    load_run_metrics,
    model_label,
    normalize_columns,
    parse_cell,
    reference_for,
    summarize_predictions,
)

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


def write_cell(root, model, language, arm, n_correct, n=50):
    """Write one cell exactly where `cell_dir` puts it, in the runner's schema."""
    run_dir = cell_dir(root, model, language, arm)
    run_dir.mkdir(parents=True)

    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "model_id": MODEL_ID,
                "language": language,
                "accuracy": n_correct / n,
                "n_eval": n,
                "seed": 42,
                "truncation_rate": 0.0,
            }
        )
    )

    with open(run_dir / "predictions.jsonl", "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "qid": f"q{i}",
                        "is_correct": 1 if i < n_correct else 0,
                        "arm": arm,
                        "model": MODEL_ID,
                        "language": language,
                    }
                )
                + "\n"
            )
    return run_dir


@pytest.fixture
def results_tree(tmp_path):
    """A minimal results directory in the layout `run_grid` produces."""
    write_cell(tmp_path, "phi-3-mini", "ne", "baseline", 37)
    write_cell(tmp_path, "phi-3-mini", "ne", "tinyace", 36)
    return tmp_path


class TestLayout:
    """The layout is defined once; readers must not re-derive it."""

    def test_round_trips_through_cell_dir(self, tmp_path):
        directory = cell_dir(tmp_path, "qwen3-1.7b", "ne", "tinyace")
        assert parse_cell(directory.relative_to(tmp_path)) == Cell("qwen3-1.7b", "ne", "tinyace")

    def test_arm_is_the_last_segment_not_the_second_to_last(self):
        """The old readers took parts[-2], which is the language."""
        assert parse_cell("qwen3-1.7b/ne/tinyace").arm == "tinyace"
        assert parse_cell("qwen3-1.7b/ne/tinyace").language == "ne"

    def test_a_comparison_group_is_model_and_language(self):
        assert parse_cell("qwen3-1.7b/ne/tinyace").group == "qwen3-1.7b/ne"

    def test_too_shallow_a_path_is_rejected_rather_than_guessed(self):
        assert parse_cell("ne/tinyace") is None


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
        """The playbook claim is ace - scaffold_control."""
        assert reference_for("tinyace") == "scaffold_control"
        assert reference_for("tinyace_retrieval") == "scaffold_control"

    def test_ablations_reference_full_tinyace(self):
        """Comparing an ablation to baseline measures ACE plus the ablation."""
        for arm in (
            "tinyace_ablate_no_failure",
            "tinyace_ablate_no_recency",
            "tinyace_ablate_no_vagueness",
            "tinyace_fifo",
            "tinyace_playbook_en",
            "tinyace_equal_lessons",
        ):
            assert (
                reference_for(arm) == ABLATION_REFERENCE
            ), f"{arm} must be compared against full TinyACE"

    def test_ablations_are_identified(self):
        assert is_ablation("tinyace_ablate_no_curator")
        assert is_ablation("tinyace_fifo")
        assert not is_ablation("baseline")
        assert not is_ablation("scaffold_control")


class TestArmNotesDescribeTheArm:
    def test_equal_lessons_does_not_claim_a_token_budget(self):
        """The note is what appears in generated documentation.

        It read "budget by lesson count instead of tokens", which implies
        TinyACE budgets by tokens. Nothing in the pipeline does -- both arms
        pass a `top_k`, and the grid gives this one 10 against TinyACE's 5.
        What the arm varies is prefix length.
        """
        from scripts.run_grid import GRID

        spec = next(a for a in GRID if a.key == "tinyace_equal_lessons")
        assert spec.flags == ["--top-k", "10"]

        note = get_arm("tinyace_equal_lessons").note
        assert "instead of tokens" not in note
        assert "top-k 10" in note


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

    def test_no_pattern_shadows_a_later_one(self):
        """Patterns match as substrings in list order, so containment is ordered.

        The list was commented as sorted longest-first and is not. Nothing is
        currently shadowed; this holds that, so a new entry that introduces a
        containment fails here rather than silently mislabelling a model in
        every figure.
        """
        from edge_slm_ace.reporting.schema import _MODEL_PATTERNS

        patterns = [p for p, _ in _MODEL_PATTERNS]
        shadowed = [
            (early, late)
            for i, early in enumerate(patterns)
            for late in patterns[i + 1 :]
            if early in late
        ]
        assert not shadowed, f"unreachable pattern(s): {shadowed}"


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
        assert set(runs["arm"]) == {"baseline", "tinyace"}

    def test_arm_is_recovered_from_the_directory_layout(self, results_tree):
        runs = load_run_metrics(results_tree)
        assert "TinyACE" in set(runs["arm_label"])

    def test_loads_every_prediction_row(self, results_tree):
        predictions = load_predictions(results_tree)
        assert len(predictions) == 100
        assert set(predictions["arm"]) == {"baseline", "tinyace"}

    def test_the_language_segment_is_never_mistaken_for_the_arm(self, results_tree):
        """The defect this layout module exists to prevent."""
        predictions = load_predictions(results_tree)
        assert "ne" not in set(predictions["arm"])
        assert set(predictions["language"]) == {"ne"}

    def test_rows_keep_the_arm_the_runner_wrote(self, tmp_path):
        """The path fills gaps; it must not overwrite what the file says."""
        write_cell(tmp_path, "phi-3-mini", "ne", "tinyace", 30)
        predictions = load_predictions(tmp_path)
        assert set(predictions["arm"]) == {"tinyace"}

    def test_missing_root_yields_empty_frames(self, tmp_path):
        assert load_run_metrics(tmp_path / "nope").empty
        assert load_predictions(tmp_path / "nope").empty

    def test_malformed_files_are_skipped_not_fatal(self, results_tree):
        bad = results_tree / "broken" / "ne" / "baseline"
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
        difference = abs(accuracies["baseline"] - accuracies["tinyace"])

        assert difference < summary["ci_halfwidth"].min()

    def test_each_arm_gets_its_own_row(self, results_tree):
        """Pooling the arms into one row is what the layout defect produced."""
        summary = summarize_predictions(load_predictions(results_tree))
        assert len(summary) == 2
        assert set(summary["arm"]) == {"baseline", "tinyace"}
        assert set(summary["n"]) == {50}

    def test_a_withdrawn_metric_does_not_displace_the_live_one(self):
        """`oma_correct` used to win here, which is the whole defect.

        No runner has written that column since scoring moved to the harness,
        so the only rows carrying it come from the withdrawn SciQ results. A
        preference for it could therefore only ever report a withdrawn number
        in place of the live one.
        """
        df = pd.DataFrame(
            [
                {"qid": f"q{i}", "arm": "baseline", "is_correct": 1, "oma_correct": 0}
                for i in range(4)
            ]
        )
        summary = summarize_predictions(df, group_by=["arm"])
        assert summary["metric"].iloc[0] == "is_correct"
        assert summary["accuracy"].iloc[0] == 1.0

    def test_the_metric_is_the_one_every_runner_writes(self):
        df = pd.DataFrame([{"qid": f"q{i}", "arm": "baseline", "is_correct": 1} for i in range(4)])
        summary = summarize_predictions(df, group_by=["arm"])
        assert summary["metric"].iloc[0] == "is_correct"

    def test_predictions_without_the_metric_are_refused(self):
        df = pd.DataFrame([{"qid": f"q{i}", "arm": "baseline", "oma_correct": 1} for i in range(4)])
        with pytest.raises(KeyError, match="retired pipeline"):
            summarize_predictions(df, group_by=["arm"])

    def test_empty_input_yields_empty_output(self):
        assert summarize_predictions(pd.DataFrame()).empty


class TestPairingAgainstReferences:
    """`compare_arms` must find a reference for every arm the grid writes.

    It previously found none: the arm key it derived from the path was the
    language segment, so every comparison was skipped and the script printed
    "0 of 0" while exiting 0.
    """

    def _arms(self, root):
        from scripts.compare_arms import discover_arms

        return discover_arms(root)

    def test_every_ace_arm_is_paired_with_its_registered_reference(self, tmp_path):
        from scripts.compare_arms import pair_with_registered_references

        for arm in ("baseline", "scaffold_control", "tinyace"):
            write_cell(tmp_path, "qwen3-1.7b", "ne", arm, 30)

        pairs, missing = pair_with_registered_references(self._arms(tmp_path))

        assert missing == []
        paired = {(Path(a).name, Path(b).name) for a, b in pairs}
        assert ("baseline", "scaffold_control") in paired
        assert ("scaffold_control", "tinyace") in paired

    def test_a_comparison_never_crosses_languages(self, tmp_path):
        from scripts.compare_arms import pair_with_registered_references

        for language in ("en", "ne"):
            for arm in ("baseline", "scaffold_control"):
                write_cell(tmp_path, "qwen3-1.7b", language, arm, 30)

        pairs, _ = pair_with_registered_references(self._arms(tmp_path))

        for reference, arm in pairs:
            assert parse_cell(reference).language == parse_cell(arm).language


class TestEveryRegisteredArmCanBeProduced:
    """
    Six arms were registered with labels and reference arms and no way to run
    them. Four needed only a CLI flag for a ScoringParams field that already
    existed; two need evaluation paths this project does not have. `run_arm`
    accepted any of the keys and wrote a result identical to another arm under
    that arm's label, which is worse than the arm being absent.
    """

    def test_the_grid_only_contains_implementable_arms(self):
        from scripts.run_grid import GRID

        for spec in GRID:
            arm = get_arm(spec.key)
            assert arm is not None, f"{spec.key} is not registered"
            assert arm.implemented, f"{spec.key} has no runner"

    def test_run_arm_has_a_flag_for_every_scoring_ablation(self):
        """The registry's delta/gamma/beta/FIFO arms must be reachable."""
        from scripts.run_arm import parse_args, scoring_params

        for flag, field in (
            ("--disable-vagueness-penalty", "disable_vagueness_penalty"),
            ("--disable-recency-decay", "disable_recency_decay"),
            ("--disable-failure-penalty", "disable_failure_penalty"),
            ("--fifo-memory", "fifo_memory"),
        ):
            args = parse_args(
                ["--model", "m", "--language", "ne", "--arm", "tinyace", "--output-dir", ".", flag]
            )
            assert getattr(scoring_params(args), field) is True

    def test_the_grid_covers_every_implemented_ace_arm(self):
        from scripts.run_grid import GRID

        planned = {spec.key for spec in GRID}
        for arm in ARMS:
            if arm.implemented and arm.family in ("ace", "control", "reference"):
                assert arm.key in planned, f"{arm.key} is runnable but not in the grid"

    def test_an_unimplemented_arm_is_marked_as_such(self):
        assert not get_arm("tinyace_retrieval").implemented
        assert not get_arm("generative_cot").implemented
