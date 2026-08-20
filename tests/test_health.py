"""Tests for what makes a run unreportable.

The distinction this module draws decides two things: whether `compare_arms`
lets a delta into the test family, and whether `aggregate_results --strict`
fails. Both are gates on what reaches a paper, so the classification has to be
exactly right in both directions -- a limitation wrongly called invalidating
throws away good data, and the reverse publishes a confound.
"""

import json
import pathlib

import pytest

from edge_slm_ace.reporting import (
    health_issues,
    invalidating_issues,
    is_reportable,
    load_metrics,
)

CLEAN = {
    "truncated_prompts": 0,
    "truncation_rate": 0.0,
    "tokens_dropped": 0,
    "relevance_active": True,
    "token_counts_exact": True,
}


def with_(**overrides):
    return {**CLEAN, **overrides}


class TestTruncationInvalidates:
    """The measurement itself is corrupted, and asymmetrically between arms.

    lm-eval truncates from the LEFT, which for Belebele eats the passage, and a
    longer prefix truncates more -- so an ACE arm loses passages its control
    keeps, on the same items.
    """

    TRUNCATED = with_(truncated_prompts=60, truncation_rate=0.125, tokens_dropped=3480)

    def test_it_is_invalidating(self):
        issues = invalidating_issues(self.TRUNCATED)
        assert [i.field for i in issues] == ["truncated_prompts"]

    def test_the_run_is_not_reportable(self):
        assert is_reportable(self.TRUNCATED) is False

    def test_the_message_names_the_passage_and_the_asymmetry(self):
        message = invalidating_issues(self.TRUNCATED)[0].message
        assert "passage" in message
        assert "longer prefix truncates more" in message

    def test_a_clean_run_is_reportable(self):
        assert is_reportable(CLEAN) is True
        assert health_issues(CLEAN) == []


class TestLimitationsDoNotInvalidate:
    """These narrow what the number supports; they do not corrupt it.

    A run with no embedding backend really did score what it scored. Calling it
    invalid would discard a sound measurement.
    """

    @pytest.mark.parametrize("field", ["relevance_active", "token_counts_exact"])
    def test_it_is_reported_but_not_invalidating(self, field):
        metrics = with_(**{field: False})
        assert [i.field for i in health_issues(metrics)] == [field]
        assert invalidating_issues(metrics) == []
        assert is_reportable(metrics) is True

    def test_both_at_once_still_do_not_invalidate(self):
        metrics = with_(relevance_active=False, token_counts_exact=False)
        assert len(health_issues(metrics)) == 2
        assert is_reportable(metrics) is True

    def test_invalidating_issues_come_first(self):
        metrics = with_(truncated_prompts=1, relevance_active=False, token_counts_exact=False)
        assert [i.invalidating for i in health_issues(metrics)] == [True, False, False]


class TestUnknownHealthIsNotBadHealth:
    def test_absent_metrics_are_reportable(self):
        """An older results tree carries no health fields.

        Refusing to compare it would be a stronger claim than the data
        supports: nothing says those runs truncated, only that they did not
        record whether they did.
        """
        assert is_reportable(None) is True
        assert health_issues(None) == []

    def test_metrics_without_the_fields_are_reportable(self):
        assert is_reportable({"arm": "tinyace", "accuracy": 0.4}) is True


class TestLoadMetrics:
    def test_it_reads_the_file_beside_the_predictions(self, tmp_path):
        (tmp_path / "metrics.json").write_text(json.dumps(CLEAN), encoding="utf-8")
        assert load_metrics(tmp_path) == CLEAN

    def test_a_missing_file_is_none_not_an_error(self, tmp_path):
        assert load_metrics(tmp_path) is None

    def test_a_truncated_file_is_none_not_an_error(self, tmp_path):
        (tmp_path / "metrics.json").write_text('{"trunc', encoding="utf-8")
        assert load_metrics(tmp_path) is None


class TestCompareArmsExcludesAnInvalidatedPair:
    """The gate on whether a delta is a result must see what invalidates one.

    `compare_arms` read `predictions.jsonl` and nothing else, so it could print
    "TinyACE is better than Scaffold Control (p=0.015)" from a pair where the
    ACE arm had lost its passages to left-truncation and the control had not.
    Truncation is arm-asymmetric -- the longer prefix truncates more -- so that
    pair carries biased evidence, which is worse for a family of tests than no
    evidence at all.
    """

    def tree(self, tmp_path, truncated_arm=None):
        import json as _json

        ids = [f"q{i}" for i in range(40)]
        for arm, correct in (("scaffold_control", 10), ("tinyace", 30)):
            cell = tmp_path / "m" / "ne" / arm
            cell.mkdir(parents=True)
            rows = [
                {"qid": q, "is_correct": int(i < correct), "arm": arm} for i, q in enumerate(ids)
            ]
            cell.joinpath("predictions.jsonl").write_text(
                "\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )
            truncated = 20 if arm == truncated_arm else 0
            cell.joinpath("metrics.json").write_text(
                _json.dumps(
                    {
                        "arm": arm,
                        "truncated_prompts": truncated,
                        "truncation_rate": truncated / (len(ids) * 4),
                        "tokens_dropped": truncated * 50,
                        "relevance_active": True,
                        "token_counts_exact": True,
                    }
                ),
                encoding="utf-8",
            )
        return tmp_path

    def run(self, root):
        """Run compare_arms as the CLI it is, in a child process.

        The child needs `src/` on its path explicitly: pytest's `pythonpath`
        ini setting applies to the pytest process, not to anything it spawns.
        Without this the test passed only when PYTHONPATH happened to be
        exported in the shell, and failed on a bare `pytest tests/`.
        """
        import os
        import subprocess
        import sys as _sys

        repo_root = pathlib.Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(repo_root / "src"), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)

        return subprocess.run(
            [_sys.executable, "-m", "scripts.compare_arms", "--results-root", str(root)],
            capture_output=True,
            text=True,
            cwd=repo_root,
            env=environment,
        )

    def test_a_clean_pair_is_tested_and_can_be_significant(self, tmp_path):
        out = self.run(self.tree(tmp_path)).stdout
        assert "family size 1" in out
        assert "NOT REPORTABLE" not in out

    def test_a_truncated_arm_leaves_an_empty_family(self, tmp_path):
        out = self.run(self.tree(tmp_path, truncated_arm="tinyace")).stdout
        assert "family size 0" in out
        assert "1 comparison(s) involve a run that failed a health check" in out

    def test_the_verdict_does_not_say_one_arm_is_better(self, tmp_path):
        """The verdict is the line a reader takes away."""
        out = self.run(self.tree(tmp_path, truncated_arm="tinyace")).stdout
        assert "NOT REPORTABLE" in out
        assert "is better than" not in out

    def test_a_truncated_reference_also_disqualifies_the_pair(self, tmp_path):
        out = self.run(self.tree(tmp_path, truncated_arm="scaffold_control")).stdout
        assert "family size 0" in out
        assert "NOT REPORTABLE" in out

    def test_the_reason_names_the_passage(self, tmp_path):
        out = self.run(self.tree(tmp_path, truncated_arm="tinyace")).stdout
        assert "that is the passage" in out
