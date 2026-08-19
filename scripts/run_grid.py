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

Usage:
    python -m scripts.run_grid --dry-run
    python -m scripts.run_grid --models qwen3-1.7b qwen3-4b --device cuda
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from edge_slm_ace.reporting import cell_dir, get_arm
from edge_slm_ace.utils import DEFAULT_SEED, capture_environment, resolve_model, set_seed


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
        "--models", nargs="*", default=None, help="Model keys (default: screening survivors)"
    )
    p.add_argument("--languages", nargs="*", default=["en", "ne"])
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default=None)
    p.add_argument("--limit", type=int, default=None, help="Truncate both splits (debug)")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and stop")
    p.add_argument("--force", action="store_true", help="Re-run cells already complete")
    p.add_argument("--no-chat-template", action="store_true")
    return p.parse_args(argv)


def survivors(results_root: Path) -> Optional[List[str]]:
    """Model keys that cleared the pre-registered screening floor, if screened."""
    path = results_root / "screening" / "screening.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [row["model"] for row in payload["models"] if row.get("passes")]


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    commit = capture_environment().get("git_commit")

    models = args.models or survivors(args.results_root)
    if not models:
        print(
            "Error: no models. Run `make screen` first, or pass --models "
            "explicitly. Entering the grid is gated on the pre-registered "
            "Nepali screening rule.",
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

    print(
        f"{len(jobs)} cells: {len(models)} model(s) x {len(args.languages)} language(s) "
        f"x {len(GRID)} arms"
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
            kwargs = {"pretrained": spec.hf_id, "batch_size": args.batch_size}
            if args.device:
                kwargs["device"] = args.device
            if args.dtype:
                kwargs["dtype"] = args.dtype
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
