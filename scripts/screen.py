#!/usr/bin/env python3
"""Day-1 screening gate: Nepali competence before models enter the main grid.

Evaluates Qwen3-1.7B, Qwen3-4B, Gemma-3-1B-it, and Gemma-3-4B-it on Global-MMLU
Nepali and Belebele. A model/task cell passes only if the Wilson score lower
bound (95% CI) on accuracy exceeds 30%. Chance on these 4-option tasks is 25%.

Screening draws from the adaptation split only, so model selection never
touches items the study reports on.

Usage:
    python -m scripts.screen --device cuda
    python -m scripts.screen --models qwen3-1.7b --tasks belebele global_mmlu
    make screen DEVICE=cuda
"""

import argparse
import json
import sys
from pathlib import Path

from edge_slm_ace.data import load_belebele, load_globalmmlu
from edge_slm_ace.data.belebele import study_split as belebele_study_split
from edge_slm_ace.data.globalmmlu import study_split as globalmmlu_study_split
from edge_slm_ace.eval.stats import summarize_accuracy, wilson_interval
from edge_slm_ace.harness import per_item_correctness, run_frozen_eval
from edge_slm_ace.utils import (
    ADAPTATION_SIZE,
    CHANCE_FLOOR,
    DEFAULT_SEED,
    SCREENING_FLOOR,
    SCREENING_MODELS,
    SCREENING_N,
    SCREENING_TASKS,
    capture_environment,
    resolve_model,
    screening_verdict,
    set_seed,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Model keys (default: "
            f"{', '.join(SCREENING_MODELS)} — Qwen3-1.7B/4B, Gemma-3-1B/4B-it)"
        ),
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        choices=list(SCREENING_TASKS),
        help="Benchmarks (default: belebele + global_mmlu, Nepali via --language)",
    )
    p.add_argument(
        "--language",
        default="ne",
        choices=["en", "ne"],
        help="Language for both tasks (default: ne)",
    )
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default=None)
    p.add_argument("--n", type=int, default=SCREENING_N, help=f"Items (default: {SCREENING_N})")
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Score as raw completion. Only correct for a base (non-instruct) model.",
    )
    return p.parse_args(argv)


def _load_and_split(task: str, language: str, seed: int):
    if task == "belebele":
        examples = load_belebele(language)
        adapt_ids, eval_ids = belebele_study_split(examples, seed)
        return examples, adapt_ids, eval_ids
    if task == "global_mmlu":
        examples = load_globalmmlu(language, split="test")
        adapt_ids, eval_ids = globalmmlu_study_split([e["id"] for e in examples], seed)
        return examples, adapt_ids, eval_ids
    raise KeyError(f"Unknown screening task '{task}'")


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    keys = args.models or list(SCREENING_MODELS)
    tasks = args.tasks or list(SCREENING_TASKS)

    out_dir = args.results_root / "screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print(
        f"Screening language={args.language}, n≤{args.n}, "
        f"tasks={', '.join(tasks)}, models={', '.join(keys)}"
    )
    print(
        f"Rule: Wilson 95% CI lower bound > {SCREENING_FLOOR:.0%} "
        f"(chance = {CHANCE_FLOOR:.0%})\n"
    )

    for task in tasks:
        _, adapt_ids, eval_ids = _load_and_split(task, args.language, args.seed)
        if args.n > len(adapt_ids):
            print(
                f"Error: --n {args.n} exceeds the {task} adaptation split "
                f"({len(adapt_ids)} items).",
                file=sys.stderr,
            )
            return 1
        screen_ids = adapt_ids[: args.n]

        leaked = set(screen_ids) & set(eval_ids)
        if leaked:
            print(
                f"Error: {len(leaked)} {task} screening item(s) overlap the " f"evaluation split.",
                file=sys.stderr,
            )
            return 1

        print(f"=== {task} ({args.language}), n={len(screen_ids)} ===")

        for key in keys:
            spec = resolve_model(key)
            print(f"  {spec.hf_id} ...", flush=True)
            try:
                results = run_frozen_eval(
                    spec.hf_id,
                    args.language,
                    screen_ids,
                    lessons=None,
                    include_scaffold=False,
                    seed=args.seed,
                    device=args.device,
                    batch_size=args.batch_size,
                    dtype=args.dtype,
                    apply_chat_template=not args.no_chat_template,
                    benchmark=task,
                )
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}", file=sys.stderr)
                rows.append(
                    {
                        "task": task,
                        "model": key,
                        "hf_id": spec.hf_id,
                        "error": str(e),
                        "passes": False,
                    }
                )
                continue

            per_item = per_item_correctness(
                results, args.language, expected_ids=screen_ids, benchmark=task
            )
            stats = summarize_accuracy(per_item.values())
            # Explicit Wilson 95% lower bound — same as summarize_accuracy["ci_low"].
            ci_low, ci_high = wilson_interval(stats["successes"], stats["n"], confidence=0.95)
            passes = screening_verdict(ci_low)
            health = results["tinyace"]

            rows.append(
                {
                    "task": task,
                    "model": key,
                    "hf_id": spec.hf_id,
                    "params": spec.params,
                    "generation": spec.generation,
                    "n": stats["n"],
                    "accuracy": stats["accuracy"],
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "passes": passes,
                    "truncation_rate": health["truncation_rate"],
                    "tokens_dropped": health["tokens_dropped"],
                }
            )
            flag = "PASS" if passes else "screened out"
            print(
                f"    {stats['accuracy']:6.1%}  "
                f"[{ci_low:.1%}, {ci_high:.1%}]  {flag}"
                + (
                    f"   (truncated {health['truncation_rate']:.0%})"
                    if health["truncated_prompts"]
                    else ""
                )
            )
        print()

    payload = {
        "language": args.language,
        "n": args.n,
        "seed": args.seed,
        "tasks": tasks,
        "rule": {
            "chance": CHANCE_FLOOR,
            "floor": SCREENING_FLOOR,
            "statistic": "wilson_ci_low",
            "confidence": 0.95,
            "pre_registered": True,
            "adaptation_size": ADAPTATION_SIZE,
            "drawn_from": "adaptation_split",
        },
        "models": rows,
        "environment": capture_environment(),
    }
    (out_dir / "screening.json").write_text(json.dumps(payload, indent=2, default=str), "utf-8")

    survivors = [f"{r['model']}@{r.get('task', '?')}" for r in rows if r.get("passes")]
    print(
        f"{len(survivors)} of {len(rows)} model×task cells cleared the floor: "
        f"{', '.join(survivors) or '(none)'}"
    )
    if not survivors:
        print(
            "No model clears the floor. Per the pre-committed kill criteria the "
            "study becomes a screening + fertility + audit paper.",
            file=sys.stderr,
        )
    print(f"-> {out_dir / 'screening.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
