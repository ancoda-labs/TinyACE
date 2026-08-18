"""Tests for accuracy intervals and paired significance testing."""

import pytest

from edge_slm_ace.utils.stats import (
    align_on_key,
    compare_arms,
    mcnemar_exact,
    summarize_accuracy,
    wilson_interval,
)


class TestWilsonInterval:
    def test_contains_point_estimate(self):
        low, high = wilson_interval(37, 50)
        assert low < 0.74 < high

    def test_small_sample_is_wide(self):
        """The whole point: n=50 cannot resolve a few-point difference."""
        low, high = wilson_interval(37, 50)
        assert (high - low) / 2 > 0.10, "n=50 halfwidth should exceed 10pp"

    def test_larger_sample_is_tighter(self):
        small = wilson_interval(37, 50)
        large = wilson_interval(370, 500)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_bounds_stay_in_range_at_ceiling(self):
        """Normal approximation would exceed 1.0 here; Wilson must not."""
        low, high = wilson_interval(50, 50)
        assert 0.0 <= low <= high <= 1.0
        assert high == 1.0
        assert low < 1.0, "A perfect score is still uncertain"

    def test_bounds_stay_in_range_at_floor(self):
        low, high = wilson_interval(0, 50)
        assert 0.0 <= low <= high <= 1.0
        assert low == 0.0

    def test_zero_samples_is_uninformative(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_rejects_unsupported_confidence(self):
        with pytest.raises(ValueError):
            wilson_interval(37, 50, confidence=0.42)


class TestMcNemar:
    def test_two_question_difference_is_not_significant(self):
        """The paper's headline '+4% at n=50' is two questions."""
        arm_a = [1] * 37 + [0] * 13
        arm_b = [1] * 39 + [0] * 11
        result = mcnemar_exact(arm_a, arm_b)
        assert result["n_discordant"] == 2
        assert result["p_value"] > 0.05
        assert result["significant_05"] is False

    def test_large_consistent_difference_is_significant(self):
        arm_a = [0] * 40 + [1] * 10
        arm_b = [1] * 40 + [1] * 10
        result = mcnemar_exact(arm_a, arm_b)
        assert result["p_value"] < 0.05
        assert result["significant_05"] is True

    def test_identical_arms_have_no_discordant_pairs(self):
        arm = [1, 0, 1, 1, 0]
        result = mcnemar_exact(arm, arm)
        assert result["n_discordant"] == 0
        assert result["p_value"] == 1.0
        assert result["delta"] == 0.0

    def test_counts_discordant_pairs_in_both_directions(self):
        arm_a = [1, 1, 0, 0]
        arm_b = [1, 0, 1, 0]
        result = mcnemar_exact(arm_a, arm_b)
        assert result["b"] == 1  # A right, B wrong
        assert result["c"] == 1  # A wrong, B right

    def test_misaligned_arms_are_rejected(self):
        with pytest.raises(ValueError):
            mcnemar_exact([1, 0], [1, 0, 1])

    def test_empty_arms(self):
        result = mcnemar_exact([], [])
        assert result["n"] == 0 and result["p_value"] == 1.0


class TestAlignment:
    def test_aligns_on_shared_ids_regardless_of_order(self):
        a = [{"qid": "q2", "oma_correct": 1}, {"qid": "q1", "oma_correct": 0}]
        b = [{"qid": "q1", "oma_correct": 1}, {"qid": "q2", "oma_correct": 1}]
        a_vals, b_vals, keys = align_on_key(a, b)
        assert keys == ["q1", "q2"]
        assert a_vals == [0, 1]
        assert b_vals == [1, 1]

    def test_drops_items_missing_from_either_arm(self):
        a = [{"qid": "q1", "oma_correct": 1}, {"qid": "q2", "oma_correct": 1}]
        b = [{"qid": "q1", "oma_correct": 0}]
        a_vals, b_vals, keys = align_on_key(a, b)
        assert keys == ["q1"]
        assert len(a_vals) == len(b_vals) == 1

    def test_drops_items_with_a_missing_metric(self):
        a = [{"qid": "q1", "oma_correct": None}, {"qid": "q2", "oma_correct": 1}]
        b = [{"qid": "q1", "oma_correct": 1}, {"qid": "q2", "oma_correct": 1}]
        _, _, keys = align_on_key(a, b)
        assert keys == ["q2"]


class TestCompareArms:
    def _rows(self, n_correct, n=50):
        return [
            {"qid": f"q{i}", "oma_correct": 1 if i < n_correct else 0}
            for i in range(n)
        ]

    def test_verdict_refuses_to_rank_noise(self):
        comparison = compare_arms(
            self._rows(37), self._rows(39), name_a="baseline", name_b="tinyace_fifo"
        )
        assert comparison["mcnemar"]["significant_05"] is False
        assert "no detectable difference" in comparison["verdict"]

    def test_intervals_are_attached_to_both_arms(self):
        comparison = compare_arms(self._rows(37), self._rows(39))
        for arm in ("arm_a", "arm_b"):
            assert comparison[arm]["ci_low"] < comparison[arm]["accuracy"]
            assert comparison[arm]["accuracy"] < comparison[arm]["ci_high"]


class TestSummarizeAccuracy:
    def test_reports_halfwidth_to_compare_deltas_against(self):
        summary = summarize_accuracy([1] * 37 + [0] * 13)
        assert summary["n"] == 50
        assert summary["accuracy"] == pytest.approx(0.74)
        assert summary["ci_halfwidth"] > 0.10

    def test_ignores_missing_values(self):
        summary = summarize_accuracy([1, 0, None, 1])
        assert summary["n"] == 3
