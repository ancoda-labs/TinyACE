"""The network marker and its skip gate — ISSUES.md #12.

These tests load Belebele from the Hub (or through HF_HOME) and used to fail the
suite when the cache path was unmounted. The marker and the skip in conftest are
what turn that into a skip rather than a red build.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Prompt 0 step 4: every test that needs the Hub or the HF datasets cache.
NETWORK_NODE_IDS = (
    "tests/test_data.py::TestNoTaskDrawsFewShotFromTheEvaluationSplit"
    "::test_the_loaded_tasks_have_no_fewshot_split",
    "tests/test_harness.py::TestHarnessParity"
    "::test_context_is_byte_identical_to_the_harness[None-False]",
    "tests/test_harness.py::TestHarnessParity"
    "::test_context_is_byte_identical_to_the_harness[None-True]",
    "tests/test_harness.py::TestHarnessParity"
    "::test_context_is_byte_identical_to_the_harness[Read the passage.-False]",
    "tests/test_harness.py::TestHarnessParity"
    "::test_context_is_byte_identical_to_the_harness[Read the passage.-True]",
    "tests/test_harness.py::TestHarnessParity"
    "::test_continuations_are_the_harness_continuations[False]",
    "tests/test_harness.py::TestHarnessParity"
    "::test_continuations_are_the_harness_continuations[True]",
)


class TestNetworkMarkerRegistration:
    def test_network_marker_is_registered(self, pytestconfig):
        markers = pytestconfig.getini("markers")
        assert any(
            m.startswith("network:") or m.split(":", 1)[0].strip() == "network" for m in markers
        ), (
            "Register `network` under [tool.pytest.ini_options] markers so an "
            "unknown-mark warning cannot hide a mistyped skip."
        )


class TestHubDependentTestsAreMarked:
    def test_every_hub_dependent_test_carries_the_network_marker(self):
        """
        Collect with `-m network` and require every Prompt 0 step-4 node id.

        A missing mark reintroduces the #12 failure mode: those tests run under
        `pytest -m "not network"` and blow up when HF_HOME is unmounted.
        """
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-m",
                "network",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        collected = {
            line.strip()
            for line in proc.stdout.splitlines()
            if line.startswith("tests/") and "::" in line
        }
        missing = [nodeid for nodeid in NETWORK_NODE_IDS if nodeid not in collected]
        assert not missing, (
            "Hub/cache-dependent tests must carry @pytest.mark.network:\n"
            + "\n".join(missing)
            + f"\n\ncollect stderr:\n{proc.stderr}"
        )


class TestNetworkSkipReason:
    def test_names_a_missing_hf_home_path(self, monkeypatch):
        from tests.conftest import network_skip_reason

        missing = "/Volumes/swarr-data-tinyace-missing"
        monkeypatch.setenv("HF_HOME", missing)
        reason = network_skip_reason(hub_reachable=True)
        assert reason is not None
        assert missing in reason

    def test_skips_when_the_hub_is_unreachable(self, monkeypatch):
        from tests.conftest import network_skip_reason

        monkeypatch.delenv("HF_HOME", raising=False)
        reason = network_skip_reason(hub_reachable=False)
        assert reason is not None
        assert "unreachable" in reason.lower() or "hub" in reason.lower()

    def test_none_when_hf_home_exists_and_hub_is_up(self, monkeypatch, tmp_path):
        from tests.conftest import network_skip_reason

        monkeypatch.setenv("HF_HOME", str(tmp_path))
        assert network_skip_reason(hub_reachable=True) is None
