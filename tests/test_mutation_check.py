"""Tests for the mutation checker.

A tool whose whole job is deciding whether tests catch things must not itself
be wrong about it. Its first version was: it copied `__pycache__` into the
mutated tree, so Python reused bytecode from the previous mutant and reported
the current one as survived. The same module scored 61% and 68% on consecutive
runs.
"""

import ast
import textwrap

import pytest

from scripts.mutation_check import TARGETS, Mutation, mutations


def write(tmp_path, source: str):
    path = tmp_path / "subject.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


class TestMutationsAreSingleValidEdits:
    SOURCE = """
        def decide(a, b, flag):
            if a > b and flag:
                return a - b
            return 0
    """

    def test_every_mutant_parses(self, tmp_path):
        for mutation in mutations(write(tmp_path, self.SOURCE)):
            ast.parse(mutation.source)

    def test_no_mutant_equals_the_original(self, tmp_path):
        path = write(tmp_path, self.SOURCE)
        original = ast.unparse(ast.parse(path.read_text()))
        for mutation in mutations(path):
            assert mutation.source != original

    def test_each_mutant_differs_from_the_others(self, tmp_path):
        produced = [m.source for m in mutations(write(tmp_path, self.SOURCE))]
        assert len(produced) == len(set(produced)), "the same edit was emitted twice"

    def test_each_mutant_changes_exactly_one_line(self, tmp_path):
        path = write(tmp_path, self.SOURCE)
        original = ast.unparse(ast.parse(path.read_text())).splitlines()
        for mutation in mutations(path):
            changed = [
                i
                for i, (before, after) in enumerate(zip(original, mutation.source.splitlines()))
                if before != after
            ]
            assert len(changed) == 1, f"{mutation.label()} changed {len(changed)} lines"

    def test_it_reports_a_line_number(self, tmp_path):
        for mutation in mutations(write(tmp_path, self.SOURCE)):
            assert mutation.line > 0


class TestOperatorsCoverTheDefectShapesSeenHere:
    """Each operator matches a defect class in this repository's history."""

    def descriptions(self, tmp_path, source):
        return {m.description for m in mutations(write(tmp_path, source))}

    def test_comparisons_flip(self, tmp_path):
        # `is_generic` on 0.5 against an admission gate of 0.6.
        found = self.descriptions(tmp_path, "def f(x):\n    return x > 0.5\n")
        assert "Gt -> GtE" in found

    def test_constants_move(self, tmp_path):
        found = self.descriptions(tmp_path, "def f(x):\n    return x > 0.5\n")
        assert "0.5 -> 1.5" in found

    def test_booleans_invert(self, tmp_path):
        # An ablation flag that silently defaults the wrong way.
        found = self.descriptions(tmp_path, "def f(flag=True):\n    return flag\n")
        assert "True -> False" in found

    def test_negation_is_dropped(self, tmp_path):
        found = self.descriptions(tmp_path, "def f(x):\n    return not x\n")
        assert "drop `not`" in found

    def test_boolean_operators_swap(self, tmp_path):
        found = self.descriptions(tmp_path, "def f(a, b):\n    return a and b\n")
        assert "And -> Or" in found

    def test_arithmetic_swaps(self, tmp_path):
        # The Holm multiplier: `(m - rank) * p`.
        found = self.descriptions(tmp_path, "def f(m, rank, p):\n    return (m - rank) * p\n")
        assert "Sub -> Add" in found
        assert "Mult -> Div" in found

    def test_strings_are_left_alone(self, tmp_path):
        """Mutating every string would drown the signal in message assertions."""
        found = self.descriptions(tmp_path, "def f():\n    return 'hello'\n")
        assert found == set()


class TestABrokenPairingIsRefusedNotScored:
    """A run where the tests never executed reads exactly like a survivor.

    `_find_repo_root` searches upward for `data/tasks` and finds nothing above a
    temp directory, so every data-backed test hit its skipif and pytest still
    exited 0. `belebele` scored 3/28 with 25 phantom survivors, and the number
    looked like a finding about test quality.
    """

    def test_all_skipped_is_a_problem(self, monkeypatch, tmp_path):
        import scripts.mutation_check as mc

        monkeypatch.setattr(
            mc, "run_tests", lambda t, s, timeout=60: (True, "22 skipped in 0.02s\n")
        )
        assert "no tests actually ran" in mc.baseline_problem(["tests/x.py"], tmp_path)

    def test_already_failing_tests_are_a_problem(self, monkeypatch, tmp_path):
        import scripts.mutation_check as mc

        monkeypatch.setattr(mc, "run_tests", lambda t, s, timeout=60: (False, "1 failed\n"))
        assert "already fail" in mc.baseline_problem(["tests/x.py"], tmp_path)

    def test_a_sound_pairing_reports_no_problem(self, monkeypatch, tmp_path):
        import scripts.mutation_check as mc

        monkeypatch.setattr(
            mc, "run_tests", lambda t, s, timeout=60: (True, "27 passed in 0.01s\n")
        )
        assert mc.baseline_problem(["tests/x.py"], tmp_path) is None

    def test_check_refuses_rather_than_scoring(self, monkeypatch):
        import scripts.mutation_check as mc

        monkeypatch.setattr(mc, "run_tests", lambda t, s, timeout=60: (True, "22 skipped\n"))
        with pytest.raises(RuntimeError, match="cannot score"):
            mc.check("health")


class TestANonTerminatingMutantIsCaught:
    """A mutant can make the code loop forever, and then nothing ever returns.

    `while str(self._next_id) in existing_ids` mutated to `not in` never exits.
    The tool sat on that one mutant with no output, indistinguishable from slow
    progress. A run that has to be abandoned is not a survivor -- it is the most
    emphatic kill available.
    """

    def test_a_timeout_counts_as_killed(self, monkeypatch, tmp_path):
        import subprocess

        import scripts.mutation_check as mc

        def hang(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="pytest", timeout=kwargs.get("timeout", 60))

        monkeypatch.setattr(mc.subprocess, "run", hang)
        passed, output = mc.run_tests(["tests/x.py"], tmp_path, timeout=1)
        assert passed is False
        assert output == ""

    def test_the_timeout_is_passed_to_the_subprocess(self, monkeypatch, tmp_path):
        import scripts.mutation_check as mc

        seen = {}

        def record(*args, **kwargs):
            seen["timeout"] = kwargs.get("timeout")

            class R:
                returncode = 0
                stdout = "1 passed"

            return R()

        monkeypatch.setattr(mc.subprocess, "run", record)
        mc.run_tests(["tests/x.py"], tmp_path, timeout=7)
        assert seen["timeout"] == 7

    def test_there_is_a_default_so_a_hang_cannot_be_unbounded(self):
        import scripts.mutation_check as mc

        assert mc.DEFAULT_TIMEOUT > 0


class TestTargets:
    @pytest.mark.parametrize("name", sorted(TARGETS))
    def test_each_target_names_files_that_exist(self, name):
        from scripts.mutation_check import REPO_ROOT

        source, tests = TARGETS[name]
        assert (REPO_ROOT / source).is_file(), source
        for test in tests:
            assert (REPO_ROOT / test).is_file(), test

    def test_a_mutation_carries_enough_to_report_it(self):
        mutation = Mutation(line=7, description="Gt -> GtE", source="x = 1")
        assert "line 7" in mutation.label()
        assert "Gt -> GtE" in mutation.label()
