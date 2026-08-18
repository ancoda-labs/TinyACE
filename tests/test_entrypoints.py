"""Static guards on the scripts that produce reportable numbers.

Every entrypoint has to seed its RNGs and permute MCQ option order. Three of
the four did; `run_ace_epoch.py` did neither, so `permutation_for` returned the
identity permutation and -- because every SciQ row stores the gold answer first
-- it presented the correct answer as option (A) for 100% of examples. That is
the single largest artefact the previous audit found, reintroduced in a script
nobody re-checked.

These are AST checks rather than runs: the point is to catch a *missing* call,
which no amount of running the happy path will surface.
"""

import ast
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

# Runners that render MCQ options and therefore must be told the permutation.
OPTION_ORDER_RUNNERS = {"run_dataset_baseline", "run_dataset_ace"}


def script_paths():
    """Every runnable script, excluding the package marker."""
    return sorted(p for p in SCRIPTS_DIR.glob("*.py") if p.name != "__init__.py")


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def calls_named(tree: ast.Module, names: set) -> list:
    """Every Call node invoking one of `names`, by bare name or attribute."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if called in names:
            found.append(node)
    return found


@pytest.mark.parametrize("path", script_paths(), ids=lambda p: p.name)
def test_scripts_running_an_evaluation_pass_option_shuffle_seed(path):
    """A runner invoked without a shuffle seed pins the gold answer to (A)."""
    tree = parse(path)
    for call in calls_named(tree, OPTION_ORDER_RUNNERS):
        keywords = {kw.arg for kw in call.keywords}
        assert "option_shuffle_seed" in keywords, (
            f"{path.name}:{call.lineno} calls an evaluation runner without "
            f"option_shuffle_seed, which puts the gold answer at (A) for every example"
        )


@pytest.mark.parametrize("path", script_paths(), ids=lambda p: p.name)
def test_scripts_running_an_evaluation_seed_their_rngs(path):
    """Decoding, option order and model init are all seed-dependent."""
    tree = parse(path)
    if not calls_named(tree, OPTION_ORDER_RUNNERS):
        pytest.skip("does not run an evaluation")
    assert calls_named(tree, {"set_seed"}), (
        f"{path.name} runs an evaluation without calling set_seed(); its numbers "
        f"cannot be regenerated"
    )
