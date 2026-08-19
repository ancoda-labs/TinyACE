#!/usr/bin/env python3
"""Screen models for Nepali competence before they enter the main grid.

A model that cannot read Nepali at all cannot show a context-adaptation effect
on Nepali, and including it would dilute every aggregate with noise centred on
chance. The rule is pre-registered and stated on the interval, not the point
estimate: Belebele is four-option, so chance is 25%, and at n=200 the Wilson
halfwidth near 35% is about 6.6 points -- an observed 35% is consistent with a
true 28%. Screening runs at n=400 and requires the lower bound to clear 30%.

Screened-out models go in the appendix, not the bin: "current small models
cannot read Nepali" is itself a finding, and it is the one the kill criteria
fall back on.

Usage:
    python -m scripts.screen_models --device cuda
    python -m scripts.screen_models --models qwen3-1.7b gemma-3-1b --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

from edge_slm_ace.data import load_belebele, parallel_split
from edge_slm_ace.eval.stats import summarize_accuracy
from edge_slm_ace.harness import per_item_correctness, run_frozen_eval
from edge_slm_ace.utils import (
    CHANCE_FLOOR,
    MODELS,
    SCREENING_FLOOR,
    SCREENING_N,
    DEFAULT_SEED,
    capture_environment,
    resolve_model,
    screening_verdict,
    set_seed,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", nargs="*", default=None, help="Model keys (default: all)")
    p.add_argument("--language", default="ne", choices=["en", "ne"])
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default=None)
    p.add_argument("--n", type=int, default=SCREENING_N, help=f"Items (default: {SCREENING_N})")
    p.add_argument(
        "--adaptation-size",
        type=int,
        default=SCREENING_N,
        help="Size of the adaptation split screening draws from. Screening must "
        "never touch the evaluation split: it is a decision made before the "
        "study, and deciding it on the scored items would be selection on the "
        "outcome.",
    )
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Score as raw completion. Only correct for a base (non-instruct) model.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    keys = args.models or [k for k in MODELS if MODELS[k].generation != "debug"]
    examples = load_belebele(args.language)
    # Screening draws from the adaptation split, so the evaluation split stays
    # untouched by a decision made before the study starts.
    adapt_ids, _ = (
        parallel_split([e["id"] for e in examples], args.adaptation_size, args.seed)
        if hasattr(args, "adaptation_size")
        else parallel_split([e["id"] for e in examples], max(args.n, 200), args.seed)
    )
    screen_ids = adapt_ids[: args.n]

    out_dir = args.results_root / "screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"Screening on Belebele {args.language}, n={len(screen_ids)}")
    print(f"Rule: Wilson lower bound > {SCREENING_FLOOR:.0%} (chance = {CHANCE_FLOOR:.0%})\n")

    for key in keys:
        spec = resolve_model(key)
        print(f"  {spec.hf_id} ...", flush=True)
        try:
            results = run_frozen_eval(
                spec.hf_id,
                args.language,
                screen_ids,
                lessons=None,
                include_scaffold=False,  # bare baseline: raw ability, no scaffold
                seed=args.seed,
                device=args.device,
                batch_size=args.batch_size,
                dtype=args.dtype,
                apply_chat_template=not args.no_chat_template,
            )
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            rows.append({"model": key, "hf_id": spec.hf_id, "error": str(e), "passes": False})
            continue

        per_item = per_item_correctness(results, args.language)
        stats = summarize_accuracy(per_item.values())
        passes = screening_verdict(stats["ci_low"])
        health = results["tinyace"]

        rows.append(
            {
                "model": key,
                "hf_id": spec.hf_id,
                "params": spec.params,
                "generation": spec.generation,
                "n": stats["n"],
                "accuracy": stats["accuracy"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "passes": passes,
                "truncation_rate": health["truncation_rate"],
                "tokens_dropped": health["tokens_dropped"],
            }
        )
        flag = "PASS" if passes else "screened out"
        print(
            f"    {stats['accuracy']:6.1%}  [{stats['ci_low']:.1%}, {stats['ci_high']:.1%}]  {flag}"
            + (
                f"   (truncated {health['truncation_rate']:.0%})"
                if health["truncated_prompts"]
                else ""
            )
        )

    payload = {
        "language": args.language,
        "n": len(screen_ids),
        "seed": args.seed,
        "rule": {
            "chance": CHANCE_FLOOR,
            "floor": SCREENING_FLOOR,
            "statistic": "wilson_ci_low",
            "pre_registered": True,
        },
        "models": rows,
        "environment": capture_environment(),
    }
    (out_dir / "screening.json").write_text(json.dumps(payload, indent=2, default=str), "utf-8")

    survivors = [r["model"] for r in rows if r.get("passes")]
    print(
        f"\n{len(survivors)} of {len(rows)} models cleared the floor: {', '.join(survivors) or '(none)'}"
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
