#!/usr/bin/env python3
"""Run one arm: adapt a playbook if the arm needs one, then evaluate frozen.

The two phases are deliberately separate. Adaptation is the only loop this
project owns and it sees the adaptation split only; evaluation is a stock
lm-evaluation-harness run over a frozen artifact. Nothing computed during
evaluation can feed back into the playbook, because the playbook is a file by
then.

Usage:
    # Control arm: the scaffold, no playbook
    python -m scripts.run_arm --model qwen3-1.7b --language ne --arm scaffold_control

    # Full arm: adapt on the adaptation split, freeze, score the eval split
    python -m scripts.run_arm --model qwen3-1.7b --language ne --arm tinyace

    # Reuse a playbook adapted in English against Nepali questions
    python -m scripts.run_arm --model qwen3-1.7b --language ne \\
        --arm tinyace_playbook_en --init-playbook results/.../playbook.jsonl
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from edge_slm_ace.adapt import adapt_playbook, frozen_lessons, save_adaptation_log
from edge_slm_ace.data import load_belebele, parallel_split
from edge_slm_ace.harness import OptionScorer, accuracy_of, per_item_correctness, run_frozen_eval
from edge_slm_ace.memory.playbook import Playbook, ScoringParams
from edge_slm_ace.memory.relevance import LessonRelevance
from edge_slm_ace.reporting import get_arm
from edge_slm_ace.utils import DEFAULT_SEED, capture_environment, resolve_model, set_seed

# Arms that need no playbook at all.
NO_PLAYBOOK_ARMS = {"baseline", "scaffold_control"}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="Model key or HuggingFace id")
    p.add_argument("--language", required=True, choices=["en", "ne"])
    p.add_argument("--arm", required=True, help="Arm key from reporting/schema.py")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--adaptation-size", type=int, default=200)
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default=None, help="Torch dtype, recorded in metadata")
    p.add_argument("--top-k", type=int, default=5, help="Lessons in the frozen prefix")
    p.add_argument("--limit", type=int, default=None, help="Truncate both splits (debug)")
    p.add_argument(
        "--init-playbook",
        type=Path,
        default=None,
        help="Skip adaptation and freeze this playbook instead. Used by the "
        "cross-lingual arm, which evaluates one language with another's lessons.",
    )
    p.add_argument("--no-curator", action="store_true")
    p.add_argument("--relevance-weight", type=float, default=0.5)
    p.add_argument("--playbook-domain", default=None, help="Override the lesson domain")
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Score as raw completion. Only correct for a base (non-instruct) model.",
    )
    return p.parse_args(argv)


def main(argv=None, lm=None) -> int:
    """Run one arm. `lm` reuses an already-loaded model across arms."""
    args = parse_args(argv)
    set_seed(args.seed)
    environment = capture_environment()

    if get_arm(args.arm) is None:
        print(
            f"Error: '{args.arm}' is not a registered arm. Register it in "
            f"reporting/schema.py so it gets a label and a reference arm.",
            file=sys.stderr,
        )
        return 1

    spec = resolve_model(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    domain = args.playbook_domain or f"belebele_{args.language}"

    examples = load_belebele(args.language, domain=domain)
    by_id = {e["id"]: e for e in examples}
    adapt_ids, eval_ids = parallel_split(
        [e["id"] for e in examples], args.adaptation_size, seed=args.seed
    )
    if args.limit:
        adapt_ids, eval_ids = adapt_ids[: args.limit], eval_ids[: args.limit]

    print(f"{spec.hf_id} | {args.language} | {args.arm}")
    print(f"  {len(adapt_ids)} adaptation / {len(eval_ids)} evaluation items")

    needs_playbook = args.arm not in NO_PLAYBOOK_ARMS
    scorer = None
    playbook = Playbook(scoring_params=ScoringParams(relevance_weight=args.relevance_weight))
    adapt_summary = None

    if needs_playbook and args.init_playbook:
        playbook = Playbook.load(args.init_playbook)
        playbook.scoring_params = ScoringParams(relevance_weight=args.relevance_weight)
        print(f"  frozen playbook from {args.init_playbook}: {len(playbook.entries)} entries")

    elif needs_playbook:
        scorer = OptionScorer(spec.hf_id, device=args.device, batch_size=args.batch_size, lm=lm)
        print("  adapting...")
        adapt_summary = adapt_playbook(
            scorer,
            [by_id[i] for i in adapt_ids],
            playbook,
            domain=domain,
            use_curator=not args.no_curator,
        )
        playbook.save(args.output_dir / "playbook.jsonl")
        save_adaptation_log(adapt_summary["log"], args.output_dir / "adaptation_log.csv")
        print(
            f"  adaptation accuracy {adapt_summary['accuracy']:.1%}, "
            f"playbook {adapt_summary['playbook_size']} entries"
        )

    lessons = frozen_lessons(playbook, domain, top_k=args.top_k) if needs_playbook else None
    if needs_playbook and not lessons and playbook.entries:
        # The playbook has content but none of it under the domain being asked
        # for -- which happens when a playbook adapted on one language is
        # evaluated under another's domain. The arm would run and record a
        # result identical to scaffold_control while carrying an ACE label, so
        # refuse rather than write it.
        present = sorted({e.domain for e in playbook.entries})
        print(
            f"Error: playbook has {len(playbook.entries)} entries but none in "
            f"domain '{domain}' (present: {', '.join(present)}). Pass "
            f"--playbook-domain to retrieve from the domain the playbook was "
            f"adapted under; otherwise this arm silently becomes scaffold_control.",
            file=sys.stderr,
        )
        return 1
    if needs_playbook and not lessons:
        print(
            "Warning: the playbook is empty, so this arm is identical to "
            "scaffold_control. Do not report it as an ACE arm.",
            file=sys.stderr,
        )

    print("  evaluating (frozen)...")
    results = run_frozen_eval(
        spec.hf_id,
        args.language,
        eval_ids,
        lessons=lessons,
        include_scaffold=args.arm != "baseline",
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        dtype=args.dtype,
        apply_chat_template=not args.no_chat_template,
        lm=lm,
    )

    accuracy = accuracy_of(results, args.language)
    per_item = per_item_correctness(results, args.language)

    health = {
        k: results["tinyace"][k] for k in ("truncated_prompts", "truncation_rate", "tokens_dropped")
    }
    if health["truncated_prompts"]:
        print(
            f"WARNING: {health['truncated_prompts']} prompt(s) truncated "
            f"({health['truncation_rate']:.1%}), {health['tokens_dropped']} tokens dropped "
            f"from the LEFT -- that is the passage. A longer prefix truncates more, so "
            f"this arm is NOT comparable with a shorter-prefix arm on these items.",
            file=sys.stderr,
        )

    metrics = {
        "arm": args.arm,
        "model_id": spec.hf_id,
        "model_key": spec.key,
        "model_params": spec.params,
        "model_generation": spec.generation,
        "language": args.language,
        "seed": args.seed,
        "accuracy": accuracy,
        "n_eval": len(eval_ids),
        "n_adaptation": len(adapt_ids) if needs_playbook and not args.init_playbook else 0,
        "playbook_domain": domain,
        "playbook_size": len(playbook.entries),
        "frozen_lessons": lessons or [],
        "relevance_active": bool(
            args.relevance_weight > 0 and LessonRelevance.get_instance().available
        ),
        "relevance_encoder": LessonRelevance.get_instance().encoder_name,
        "apply_chat_template": not args.no_chat_template,
        **health,
        "scoring": asdict(playbook.scoring_params),
        "adaptation": {k: v for k, v in (adapt_summary or {}).items() if k != "log"},
        "environment": environment,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    with open(args.output_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for item_id, correct in sorted(per_item.items()):
            f.write(
                json.dumps(
                    {
                        "qid": item_id,
                        "is_correct": correct,
                        "arm": args.arm,
                        "model": spec.hf_id,
                        "language": args.language,
                    }
                )
                + "\n"
            )

    print(f"  accuracy {accuracy:.1%} on n={len(per_item)} -> {args.output_dir}")
    print("  A difference is only a result if it survives scripts/compare_arms.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
