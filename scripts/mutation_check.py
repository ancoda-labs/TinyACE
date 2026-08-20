#!/usr/bin/env python3
"""Check that the tests would actually catch a change, not just describe one.

CONTRIBUTING asks for "a test that fails without the change". That is the right
bar and it is unverifiable by reading: a test that asserts what the code
currently does passes whether or not the code is right, and looks identical to
one that pins behaviour down. Every defect under Fixed in CHANGELOG.md sat in a
suite that was green.

So this makes the small edits a regression would make -- flip a comparison,
move a constant, swap a boolean -- and runs the tests against each one. A
mutation the tests *fail* on is caught. A mutation they *pass* on is a line
nothing is holding: the behaviour on that line can change without any test
objecting.

This is a development tool, not part of the study pipeline. Nothing it does can
touch a result: the mutations are applied to a throwaway copy of `src/`, and the
working tree is never written to.

Usage:
    python -m scripts.mutation_check                    # the modules below
    python -m scripts.mutation_check --module eval      # one of them
    python -m scripts.mutation_check --list             # what would be checked
"""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# Module -> the tests that must catch a change to it.
#
# Paired deliberately rather than running the whole suite per mutant. The suite
# takes 13s and most single files take 0.2s, so pairing is what makes this
# runnable at all -- and a mutation in `stats.py` that only `test_harness.py`
# notices is not the coverage anyone means.
TARGETS = {
    "stats": ("src/edge_slm_ace/eval/stats.py", ["tests/test_stats.py"]),
    "health": ("src/edge_slm_ace/reporting/health.py", ["tests/test_health.py"]),
    "playbook": ("src/edge_slm_ace/memory/playbook.py", ["tests/test_playbook.py"]),
    "belebele": ("src/edge_slm_ace/data/belebele.py", ["tests/test_belebele.py"]),
    "schema": ("src/edge_slm_ace/reporting/schema.py", ["tests/test_reporting.py"]),
    "relevance": ("src/edge_slm_ace/memory/relevance.py", ["tests/test_relevance.py"]),
    "ace_roles": ("src/edge_slm_ace/core/ace_roles.py", ["tests/test_ace_roles.py"]),
    "adapt": ("src/edge_slm_ace/adapt.py", ["tests/test_adapt.py"]),
}

# Comparison flips. These are the shape of several real defects here: a
# screening gate on `>` where `>=` was meant, `is_generic` on 0.5 where the
# admission gate used 0.6, an eviction key sorted the wrong way round.
_COMPARISONS = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}

_ARITHMETIC = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}

_BOOLOPS = {ast.And: ast.Or, ast.Or: ast.And}


class Mutation(NamedTuple):
    """One edit, and where it was made."""

    line: int
    description: str
    source: str

    def label(self) -> str:
        return f"line {self.line}: {self.description}"


class _Mutator(ast.NodeTransformer):
    """Applies exactly the `target`-th mutation it finds, and no other."""

    def __init__(self, target: int):
        self.target = target
        self.seen = 0
        self.applied: Optional[str] = None
        self.line = 0

    def _take(self, node, description: str) -> bool:
        """True when this is the occurrence we were asked to change."""
        hit = self.seen == self.target
        self.seen += 1
        if hit:
            self.applied = description
            self.line = getattr(node, "lineno", 0)
        return hit

    def visit_Compare(self, node):
        self.generic_visit(node)
        for index, op in enumerate(node.ops):
            replacement = _COMPARISONS.get(type(op))
            if replacement is None:
                continue
            if self._take(node, f"{type(op).__name__} -> {replacement.__name__}"):
                node.ops[index] = replacement()
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        replacement = _ARITHMETIC.get(type(node.op))
        if replacement is not None and self._take(
            node, f"{type(node.op).__name__} -> {replacement.__name__}"
        ):
            node.op = replacement()
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        replacement = _BOOLOPS.get(type(node.op))
        if replacement is not None and self._take(
            node, f"{type(node.op).__name__} -> {replacement.__name__}"
        ):
            node.op = replacement()
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and self._take(node, "drop `not`"):
            return node.operand
        return node

    def visit_Constant(self, node):
        # Only values that decide something. Mutating every string would produce
        # thousands of mutants whose death proves only that a message is
        # asserted somewhere.
        if isinstance(node.value, bool):
            if self._take(node, f"{node.value} -> {not node.value}"):
                return ast.copy_location(ast.Constant(value=not node.value), node)
        elif isinstance(node.value, (int, float)):
            new = node.value + 1
            if self._take(node, f"{node.value!r} -> {new!r}"):
                return ast.copy_location(ast.Constant(value=new), node)
        return node


def mutations(source_path: Path) -> Iterator[Mutation]:
    """
    Every single-edit variant of a module, one at a time.

    Args:
        source_path: The module to mutate.

    Returns:
        One `Mutation` per edit, each carrying the full mutated source.
    """
    original = source_path.read_text(encoding="utf-8")
    tree = ast.parse(original)

    total = _Mutator(target=-1)
    total.visit(ast.parse(original))

    for index in range(total.seen):
        mutator = _Mutator(target=index)
        mutated = mutator.visit(ast.parse(original))
        if mutator.applied is None:
            continue
        ast.fix_missing_locations(mutated)
        try:
            rendered = ast.unparse(mutated)
        except Exception:
            continue
        # An edit that renders back to the original changed nothing real.
        if rendered == ast.unparse(tree):
            continue
        yield Mutation(line=mutator.line, description=mutator.applied, source=rendered)


# A mutant can make the code non-terminating, and then the test run never ends.
# `while str(self._next_id) in existing_ids` becomes `not in` and loops forever;
# the tool sat on that one mutant with no output and no way to tell it from slow
# progress. Paired test files run in under six seconds, so this is generous.
DEFAULT_TIMEOUT = 60


def run_tests(
    test_paths: List[str], source_root: Path, timeout: int = DEFAULT_TIMEOUT
) -> Tuple[bool, str]:
    """
    Run the paired tests against a mutated copy of `src/`.

    Bytecode caching has to be off, and this is not a detail. Python decides a
    `.pyc` is stale by the source's mtime and size, both at one-second
    granularity on some filesystems. Mutants are written far faster than that
    and `ast.unparse` often renders two of them at the same length, so a run
    would import the *previous* mutant's bytecode and report the current one as
    survived. That made the whole tool nondeterministic: the same module scored
    61% and 68% on consecutive runs. A tool for checking whether tests catch
    things has to be the last thing in the repo that lies about it.

    Args:
        test_paths: Repo-relative test files.
        source_root: The mutated source tree to import from.
        timeout: Seconds before the run is abandoned. A mutant that never
            terminates has not been survived -- it has been caught, in the most
            emphatic way available -- so a timeout counts as a kill.

    Returns:
        `(passed, stdout)`. `passed` is True when the tests pass, which for a
        mutant means it survived. A timeout returns `(False, "")`.
    """
    environment = dict(os.environ)
    # An inherited PYTHONPATH pointing at the real `src/` would shadow the
    # mutated copy, and every mutant would survive.
    environment["PYTHONPATH"] = str(source_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # `_find_repo_root` searches upward from the package for `data/tasks`, and
    # finds nothing above a temp directory -- so every data-backed test hit its
    # skipif and pytest still exited 0. All 22 tests in test_belebele.py skipped
    # and all 28 of its mutations were scored as survivors.
    environment["TINYACE_DATA_ROOT"] = str(REPO_ROOT)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                *test_paths,
                "-x",
                "-q",
                "--no-header",
                "-p",
                "no:cacheprovider",
                "-o",
                f"pythonpath={source_root}",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=environment,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, ""
    return result.returncode == 0, result.stdout


def baseline_problem(
    test_paths: List[str], source_root: Path, timeout: int = DEFAULT_TIMEOUT
) -> Optional[str]:
    """
    Why this module/test pairing cannot be scored, if it cannot.

    A mutant is judged by whether the tests fail, so a run where the tests never
    executed reads identically to a mutant nothing caught. That is not
    hypothetical: every data-backed test skipped against the mutated tree,
    pytest exited 0, and the module scored 3/28 with 25 phantom survivors.

    Args:
        test_paths: Repo-relative test files.
        source_root: An *unmutated* copy of the source tree.

    Returns:
        A description of the problem, or None when the pairing is sound.
    """
    passed, output = run_tests(test_paths, source_root, timeout)
    if not passed:
        return "the paired tests already fail before any mutation"
    if "passed" not in output:
        return (
            "no tests actually ran -- they were all skipped or none were "
            "collected, so every mutation would be scored as a survivor"
        )
    return None


def check(
    name: str, verbose: bool = False, timeout: int = DEFAULT_TIMEOUT
) -> Tuple[int, int, List[Mutation]]:
    """
    Mutate one module and report which edits the tests did not notice.

    Args:
        name: A key of TARGETS.
        verbose: Print each mutant as it is decided.
        timeout: Per-mutant seconds before the run is abandoned.

    Returns:
        (killed, total, survivors).

    Raises:
        RuntimeError: If the paired tests do not run cleanly before any
            mutation, which would make every score meaningless.
    """
    relative, test_paths = TARGETS[name]
    source_path = REPO_ROOT / relative

    survivors: List[Mutation] = []
    killed = 0
    all_mutations = list(mutations(source_path))

    with tempfile.TemporaryDirectory() as tmp:
        source_root = Path(tmp) / "src"
        # Never copy __pycache__: a stale .pyc beside a mutated .py is exactly
        # the staleness `run_tests` disables bytecode to avoid.
        shutil.copytree(
            REPO_ROOT / "src", source_root, ignore=shutil.ignore_patterns("__pycache__")
        )
        mutated_path = source_root / Path(relative).relative_to("src")

        problem = baseline_problem(test_paths, source_root, timeout)
        if problem is not None:
            raise RuntimeError(f"cannot score '{name}': {problem}")

        for index, mutation in enumerate(all_mutations, 1):
            mutated_path.write_text(mutation.source, encoding="utf-8")
            survived, _ = run_tests(test_paths, source_root, timeout)
            if survived:
                survivors.append(mutation)
            else:
                killed += 1
            if verbose:
                mark = "SURVIVED" if survived else "killed  "
                print(f"    [{index}/{len(all_mutations)}] {mark} {mutation.label()}")
            else:
                print(
                    f"    {index}/{len(all_mutations)} "
                    f"({killed} killed, {len(survivors)} survived)",
                    end="\r",
                    flush=True,
                )

    if not verbose:
        print(" " * 70, end="\r")
    return killed, len(all_mutations), survivors


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--module", action="append", choices=sorted(TARGETS), help="Default: all")
    p.add_argument("--list", action="store_true", help="Print the pairings and stop")
    p.add_argument("--verbose", action="store_true", help="Print every mutant")
    p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Seconds per mutant before giving up (default: {DEFAULT_TIMEOUT}). "
        "A mutant that never terminates counts as caught.",
    )
    p.add_argument(
        "--max-survivors",
        type=int,
        default=None,
        help="Exit non-zero if more than this many mutations survive. Omit to "
        "report without failing.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list:
        for name, (source, tests) in sorted(TARGETS.items()):
            print(f"  {name:12} {source}  <-  {', '.join(tests)}")
        return 0

    names = args.module or sorted(TARGETS)
    total_survivors: List[Tuple[str, Mutation]] = []
    total_killed = total_count = 0

    for name in names:
        source, _ = TARGETS[name]
        print(f"\n{name} ({source})")
        killed, count, survivors = check(name, verbose=args.verbose, timeout=args.timeout)
        total_killed += killed
        total_count += count
        total_survivors.extend((name, m) for m in survivors)
        score = killed / count if count else 1.0
        print(f"  {killed}/{count} mutations caught ({score:.0%})")
        for mutation in survivors:
            print(f"    SURVIVED  {mutation.label()}")

    print()
    print("=" * 72)
    score = total_killed / total_count if total_count else 1.0
    print(
        f"{total_killed}/{total_count} mutations caught ({score:.0%}) across {len(names)} module(s)"
    )
    if total_survivors:
        print(
            f"{len(total_survivors)} survived. Each is a line whose behaviour can "
            f"change with no test objecting -- not necessarily a bug, but not "
            f"covered either."
        )

    if args.max_survivors is not None and len(total_survivors) > args.max_survivors:
        print(
            f"\nError: {len(total_survivors)} mutations survived, over the "
            f"--max-survivors budget of {args.max_survivors}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
