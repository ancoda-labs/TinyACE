#!/usr/bin/env python3
"""Run every arm for every model and language, loading each model once.

The grid is organised around what actually costs GPU time. Evaluation is four
forward passes per question; adaptation is a generation per error plus one per
curation, so adaptation dominates. Two consequences shape this runner:

1. **A model is loaded once** and reused for every arm and both languages.
   Reloading per cell was the single largest avoidable cost.
2. **Arms that differ only in how a frozen playbook is *presented* reuse the
   playbook** rather than re-adapting. Arms that change adaptation itself --
   any scoring ablation, the Curator ablation, retrieval weight -- get their
   own adaptation run, because that is the thing being ablated.

Ordering matters: an arm with `playbook_from` runs after the arm it borrows
from, which is why the plan is built before anything executes.

This runner does not choose items. Each cell calls run_arm, which reads
the frozen manifest. --seed is forwarded for model and sampling RNGs and
for the order that cell walks its adaptation ids.

Usage:
    python -m scripts.run_grid --dry-run
    python -m scripts.run_grid --models qwen3-1.7b qwen3-4b --device cuda
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from edge_slm_ace.harness.evaluate import hflm_kwargs
from edge_slm_ace.reporting import cell_dir, get_arm, legacy_screening_path, screening_path
from edge_slm_ace.utils import (
    DEFAULT_SEED,
    REPO_ROOT,
    capture_environment,
    resolve_model,
    set_seed,
)

# Written before screening. The grid reads this and does not restate the rule.
MODEL_SELECTION = REPO_ROOT / "configs" / "model_selection.yaml"
# screen.py names the statistic wilson_ci_low and stores the number as ci_low.
_CI_LOW_FIELD = "ci_low"


@dataclass(frozen=True)
class ArmSpec:
    """One arm, and what it needs before it can run."""

    key: str
    adapts: bool = False
    flags: List[str] = field(default_factory=list)
    # Borrow a frozen playbook instead of adapting. "(arm, language)"; the
    # language is None to mean "this cell's language".
    playbook_from: Optional[tuple] = None
    # Restrict the arm to certain languages. The cross-lingual arm is only
    # meaningful on Nepali: on English it would borrow the English playbook to
    # evaluate English, which is just `tinyace` under another name.
    only_languages: Optional[tuple] = None


# The default grid. Every key must be registered in reporting/schema.py, which
# is where its label and its reference arm live.
GRID: List[ArmSpec] = [
    ArmSpec("baseline"),
    ArmSpec("scaffold_control"),
    ArmSpec("tinyace", adapts=True),
    # Presentation-only: same playbook, more lessons in the prefix. Tests
    # whether Devanagari fertility, not lesson quality, is the binding
    # constraint on how much guidance fits.
    ArmSpec("tinyace_equal_lessons", flags=["--top-k", "10"], playbook_from=("tinyace", None)),
    # Cross-lingual: Nepali questions, English lessons.
    ArmSpec(
        "tinyace_playbook_en",
        playbook_from=("tinyace", "en"),
        only_languages=("ne",),
        # Retrieve from the domain the borrowed playbook was adapted under.
        # Without this the English lessons are invisible to a Nepali-domain
        # lookup and the arm degrades to scaffold_control with an ACE label.
        flags=["--playbook-domain", "belebele_en"],
    ),
    # These change adaptation, so they adapt.
    ArmSpec("tinyace_ablate_no_curator", adapts=True, flags=["--no-curator"]),
    ArmSpec("tinyace_ablate_no_relevance", adapts=True, flags=["--relevance-weight", "0"]),
    # Retention-score ablations. Each zeroes one term of the equation, so each
    # has to build its own playbook: the term is used during adaptation, not
    # only when the prefix is frozen.
    ArmSpec("tinyace_ablate_no_vagueness", adapts=True, flags=["--disable-vagueness-penalty"]),
    ArmSpec("tinyace_ablate_no_recency", adapts=True, flags=["--disable-recency-decay"]),
    ArmSpec("tinyace_ablate_no_failure", adapts=True, flags=["--disable-failure-penalty"]),
    ArmSpec("tinyace_fifo", adapts=True, flags=["--fifo-memory"]),
]


def is_complete(directory: Path, seed: int, commit: Optional[str]) -> bool:
    """
    True when this cell already holds a complete result from this commit and seed.

    A truncated metrics.json is what an interrupted write leaves behind, so it
    must not count as a result.
    """
    metrics, predictions = directory / "metrics.json", directory / "predictions.jsonl"
    if not all(p.exists() and p.stat().st_size > 0 for p in (metrics, predictions)):
        return False
    try:
        recorded = json.loads(metrics.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if recorded.get("seed") != seed:
        return False
    return commit is None or recorded.get("environment", {}).get("git_commit") == commit


def source_languages(arms: List[ArmSpec]) -> List[str]:
    """Languages some arm borrows a playbook from, by name."""
    return sorted({a.playbook_from[1] for a in arms if a.playbook_from and a.playbook_from[1]})


def order_languages(languages: List[str], arms: List[ArmSpec]) -> List[str]:
    """
    Put every borrowed-from language ahead of the rest.

    Ordering arms was not enough. `tinyace_playbook_en` borrows English
    specifically, so on `--languages ne en` -- or on `--languages ne` alone --
    Nepali ran first, found no English playbook, and the arm was recorded as a
    failure with no explanation beyond a missing path.

    Args:
        languages: The requested languages, in the requested order.
        arms: The grid.

    Returns:
        The same languages, with any that another language's arm borrows from
        moved to the front. Relative order is otherwise preserved.
    """
    sources = set(source_languages(arms))
    return [l for l in languages if l in sources] + [l for l in languages if l not in sources]


def plan(models: List[str], languages: List[str], arms: List[ArmSpec]) -> List[Dict]:
    """
    Build the ordered job list.

    Returns:
        One dict per cell, model-outer so a model is loaded once, and with any
        borrowing arm placed after the arm it borrows from -- in both
        dimensions, arm and language.
    """
    jobs = []
    languages = order_languages(languages, arms)
    for model_key in models:
        # Arms that adapt run first, so a borrowing arm always finds its source.
        ordered = [a for a in arms if a.playbook_from is None] + [
            a for a in arms if a.playbook_from is not None
        ]
        for language in languages:
            for arm in ordered:
                if arm.only_languages and language not in arm.only_languages:
                    continue
                jobs.append({"model": model_key, "language": language, "arm": arm})
    return jobs


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model keys (default: one passing model per family, from "
        "configs/model_selection.yaml)",
    )
    p.add_argument("--languages", nargs="*", default=["en", "ne"])
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Forwarded to each cell. Seeds model and sampling RNGs and the "
        "order that cell walks the frozen adaptation ids. It does not choose "
        "which items are used.",
    )
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps", "xpu"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default=None)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Forwarded to each cell: first N ids of each frozen split "
        "(debug). Not a study run.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the plan and stop")
    p.add_argument("--force", action="store_true", help="Re-run cells already complete")
    p.add_argument("--no-chat-template", action="store_true")
    return p.parse_args(argv)


def _screening_payload(results_root: Path, benchmark: str, language: str) -> dict:
    """
    The one screening file the grid may read for this task and language.

    The pre-v3 ``screening.json`` has no language in its name. When it is
    still on disk it is ignored. A payload whose body names a different
    language or benchmark is refused: the filename is not the check.
    """
    path = screening_path(results_root, benchmark, language)
    legacy = legacy_screening_path(results_root)
    if legacy.exists():
        # The name cannot say which language this is. Using it would let an
        # English overwrite of the old file decide the Nepali gate.
        warnings.warn(
            f"ignoring {legacy}: it is the pre-v3 single screening.json, which "
            f"has no language in its name. This grid reads {path.name}.",
            # survivors() warned at stacklevel=2, naming its caller. This
            # helper is one frame deeper, so 3 keeps that attribution.
            stacklevel=3,
        )

    if not path.exists():
        raise FileNotFoundError(
            f"no screening result at {path}. The grid admits models from a "
            f"{language} {benchmark} screen."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    got_language = payload.get("language")
    got_benchmark = payload.get("benchmark")
    if got_language != language or got_benchmark != benchmark:
        raise ValueError(
            f"{path.name}: payload is language={got_language!r} "
            f"benchmark={got_benchmark!r}, expected language={language!r} "
            f"benchmark={benchmark!r}."
        )
    return payload


def _model_selection() -> dict:
    """The pre-registered family rule. Absent or rewritten fields are a raise."""
    if not MODEL_SELECTION.exists():
        raise FileNotFoundError(f"no model-selection rule at {MODEL_SELECTION}")
    loaded = yaml.safe_load(MODEL_SELECTION.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{MODEL_SELECTION.name} did not contain a mapping")
    statistic = loaded.get("statistic")
    if statistic != "wilson_ci_low":
        raise ValueError(
            f"{MODEL_SELECTION.name}: statistic is {statistic!r}, expected 'wilson_ci_low'."
        )
    benchmark, language = loaded.get("benchmark"), loaded.get("language")
    if not benchmark or not language:
        raise ValueError(f"{MODEL_SELECTION.name}: benchmark and language are required")
    families = loaded.get("families")
    if not isinstance(families, dict) or not families:
        raise ValueError(f"{MODEL_SELECTION.name}: families must name at least one family")
    seen: List[str] = []
    for name, members in families.items():
        if not isinstance(members, list) or not members:
            raise ValueError(f"{MODEL_SELECTION.name}: family {name} has no models")
        for key in members:
            if key in seen:
                raise ValueError(f"{MODEL_SELECTION.name}: {key} is listed in more than one family")
            seen.append(key)
    excluded = loaded.get("excluded") or []
    if not isinstance(excluded, list):
        raise ValueError(f"{MODEL_SELECTION.name}: excluded must be a list")
    return loaded


def _family_winner(
    members: List[str],
    by_model: Dict[str, dict],
    excluded: set,
    benchmark: str,
) -> Optional[str]:
    """
    The passing member with the highest Wilson lower bound.

    ``members`` is smaller-first. A tie keeps the earlier name, which is the
    smaller model, instead of parsing a parameter-count string.
    """
    candidates = []
    for index, key in enumerate(members):
        if key in excluded:
            continue
        row = by_model.get(key)
        if row is None or not row.get("passes"):
            continue
        # A row that names another task does not count, even inside this file.
        if row.get("task", benchmark) != benchmark:
            continue
        score = row.get(_CI_LOW_FIELD)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{key} passes but has no wilson_ci_low ({_CI_LOW_FIELD}) to rank on")
        # Highest score first, then the earlier (smaller) list entry.
        candidates.append((-float(score), index, key))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def select_grid_models(results_root: Path) -> List[str]:
    """
    One model per family from the pre-registered rule.

    ``survivors()`` returns every model that cleared the floor, so both sizes
    of one family would enter and the size the paper reports would be chosen
    after the scores existed. This reads ``configs/model_selection.yaml`` and
    the screening file that yaml names. A family with no passing model is
    absent from the result and is not replaced. If nobody passes, this raises
    rather than switching the study to Belebele: that fallback is written in
    the config, and taking it here would hide it inside the grid.

    Args:
        results_root: Directory that contains ``screening/``.

    Returns:
        Chosen model keys, in family order, one per family that had a pass.

    Raises:
        FileNotFoundError: The config or the screening file is absent.
        ValueError: The payload does not match the config, a passing candidate
            has no Wilson lower bound, or no family contributes a model.
    """
    selection = _model_selection()
    benchmark = selection["benchmark"]
    language = selection["language"]
    payload = _screening_payload(results_root, benchmark, language)
    excluded = set(selection["excluded"])

    by_model: Dict[str, dict] = {}
    for row in payload["models"]:
        key = row.get("model")
        if key in by_model:
            raise ValueError(f"{key} appears more than once in the {benchmark}/{language} screen")
        by_model[key] = row

    chosen = []
    for members in selection["families"].values():
        winner = _family_winner(members, by_model, excluded, benchmark)
        if winner is not None:
            chosen.append(winner)
    if not chosen:
        rule = " ".join(str(selection.get("rule") or "").split())
        raise ValueError(
            "no model passes the pre-registered grid selection rule "
            f"({benchmark}/{language}). {rule}"
        )
    return chosen


def survivors(
    results_root: Path,
    benchmark: str = "global_mmlu",
    language: str = "ne",
) -> List[str]:
    """
    Model keys that cleared the pre-registered floor on this task and language.

    The primary task is Global-MMLU and the pre-registered rule is a Nepali
    result, so those are the defaults. This reads only
    ``screening_{benchmark}_{language}.json`` and checks that the payload names
    the same task and language. An English file written later cannot change
    who is admitted, and a Belebele pass does not.

    The pre-v3 ``screening.json`` has no language in its name. When it is
    still on disk it is ignored.

    Args:
        results_root: Directory that contains ``screening/``.
        benchmark: Task the grid is admitting models for.
        language: Language the rule requires. There is no other language to try.

    Returns:
        Passing model keys, in file order, each once. Empty when the screen
        ran and nobody cleared the floor.

    Raises:
        FileNotFoundError: ``screening_{benchmark}_{language}.json`` is absent.
        ValueError: The payload's language or benchmark is missing or differs
            from the file that was requested.
    """
    payload = _screening_payload(results_root, benchmark, language)
    passed = []
    for row in payload["models"]:
        if not row.get("passes"):
            continue
        # A row that names another task does not count, even inside this file.
        # A missing task means the payload benchmark, which was just checked.
        if row.get("task", benchmark) != benchmark:
            continue
        passed.append(row["model"])
    return list(dict.fromkeys(passed))


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    commit = capture_environment().get("git_commit")

    if args.models:
        models = list(args.models)
    else:
        try:
            models = select_grid_models(args.results_root)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    if not models:
        print(
            "Error: no models. Run `make screen` first, or pass --models "
            "explicitly. Entering the grid is gated on the pre-registered "
            "Nepali Global-MMLU screening rule.",
            file=sys.stderr,
        )
        return 1

    jobs = plan(models, args.languages, GRID)
    unregistered = sorted({j["arm"].key for j in jobs if get_arm(j["arm"].key) is None})
    if unregistered:
        print(f"Error: unregistered arms: {', '.join(unregistered)}", file=sys.stderr)
        return 1
    unimplemented = sorted({j["arm"].key for j in jobs if not get_arm(j["arm"].key).implemented})
    if unimplemented:
        print(
            f"Error: no runner for: {', '.join(unimplemented)}. Remove them "
            f"from GRID or implement them; running one writes another arm's "
            f"result under its label.",
            file=sys.stderr,
        )
        return 1

    absent = [l for l in source_languages(GRID) if l not in args.languages]
    if absent:
        # Every borrowing arm would fail one at a time on a missing path; say it
        # once, before a model is loaded.
        print(
            f"Error: arms borrow a playbook from {', '.join(absent)}, which is "
            f"not in --languages. Add it, or drop the arms that need it.",
            file=sys.stderr,
        )
        return 1

    # Not models x languages x arms: an arm with `only_languages` is planned in
    # some cells and not others, so that product overstates the grid and does
    # not match the line printed at the end.
    restricted = sorted({a.key for a in GRID if a.only_languages})
    print(
        f"{len(jobs)} cells over {len(models)} model(s) and "
        f"{len(args.languages)} language(s), from {len(GRID)} arms"
        + (
            f" ({len(restricted)} language-restricted: {', '.join(restricted)})"
            if restricted
            else ""
        )
    )
    if args.dry_run:
        for job in jobs:
            directory = cell_dir(args.results_root, job["model"], job["language"], job["arm"].key)
            state = "done" if is_complete(directory, args.seed, commit) else "run"
            borrow = f"  <- {job['arm'].playbook_from}" if job["arm"].playbook_from else ""
            adapt = " [adapts]" if job["arm"].adapts else ""
            print(f"  [{state}] {directory}{adapt}{borrow}")
        return 0

    from scripts.run_arm import main as run_arm

    failures = []
    lm = None
    current_model = None

    for job in jobs:
        arm, model_key, language = job["arm"], job["model"], job["language"]
        directory = cell_dir(args.results_root, model_key, language, arm.key)
        label = f"{model_key}/{language}/{arm.key}"

        if not args.force and is_complete(directory, args.seed, commit):
            print(f"= {label}: already complete for this commit and seed")
            continue

        # One load per model, reused across every arm and language.
        if model_key != current_model:
            from lm_eval.models.huggingface import HFLM

            spec = resolve_model(model_key)
            print(f"\nloading {spec.hf_id}")
            # The same revision and enable_thinking build_model_args passes
            # when a cell loads the model itself. The grid loads once and
            # hands the object over, so this is the only place those pins
            # can reach a grid cell.
            kwargs = hflm_kwargs(
                spec.hf_id,
                batch_size=args.batch_size,
                device=args.device,
                dtype=args.dtype,
            )
            lm, current_model = HFLM(**kwargs), model_key

        argv_cell = [
            "--model",
            model_key,
            "--language",
            language,
            "--arm",
            arm.key,
            "--output-dir",
            str(directory),
            "--seed",
            str(args.seed),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.device:
            argv_cell += ["--device", args.device]
        # The grid builds the model, so run_arm never sees this on the load
        # path -- it is forwarded purely so the cell records what it ran under.
        if args.dtype:
            argv_cell += ["--dtype", args.dtype]
        if args.limit:
            argv_cell += ["--limit", str(args.limit)]
        if args.no_chat_template:
            argv_cell += ["--no-chat-template"]
        argv_cell += arm.flags

        if arm.playbook_from:
            source_arm, source_language = arm.playbook_from
            source = (
                cell_dir(args.results_root, model_key, source_language or language, source_arm)
                / "playbook.jsonl"
            )
            if not source.exists():
                print(f"! {label}: skipped, no playbook at {source}", file=sys.stderr)
                failures.append(label)
                continue
            argv_cell += ["--init-playbook", str(source)]

        try:
            if run_arm(argv_cell, lm=lm) != 0:
                failures.append(label)
        except Exception as e:
            print(f"! {label}: {type(e).__name__}: {e}", file=sys.stderr)
            failures.append(label)

    print(f"\n{len(jobs) - len(failures)} of {len(jobs)} cells succeeded")
    if failures:
        print("failed: " + ", ".join(failures), file=sys.stderr)
    print("Aggregate with `make report`. A delta is a result only after compare_arms.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
