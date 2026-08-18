#!/usr/bin/env python3
"""End-to-end smoke test: load a model, run a few examples, check the plumbing.

Verifies that a model loads, generates, and produces scoreable output on the
target device, without waiting for a full evaluation. Checks the conditions
that silently invalidate a real run:

  - the chat template was applied (instruct models prompted raw ramble)
  - no prompt was truncated (the tail of a prompt is the question)
  - predictions map to options above the embedding tier (an OMA resting on
    embedding argmax over free text is weak evidence)

Usage:
    # CPU, tiny model, no network-heavy download
    python -m scripts.smoke_test

    # GPU with a real model
    python -m scripts.smoke_test --model phi3-mini --device cuda

    # Any registered model or HuggingFace id
    python -m scripts.smoke_test --model Qwen/Qwen2.5-1.5B-Instruct --limit 5
"""

import argparse
import json
import sys
from pathlib import Path

from edge_slm_ace.core.runner import run_dataset_baseline
from edge_slm_ace.models.model_manager import load_model_and_tokenizer
from edge_slm_ace.utils.config import get_model_config, get_task_config, resolve_task_path
from edge_slm_ace.utils.repro import DEFAULT_SEED, capture_environment, set_seed


def load_dataset(task_name: str, limit: int):
    """Load the first `limit` examples of a registered task."""
    path = resolve_task_path(task_name)
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = json.load(f)
    return rows[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test the evaluation pipeline on a few examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiny-gpt2",
        help="Model key or HuggingFace id (default: tiny-gpt2)",
    )
    parser.add_argument(
        "--task", type=str, default="sciq_tiny", help="Registered task name (default: sciq_tiny)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device override (default: auto-detect)",
    )
    parser.add_argument("--limit", type=int, default=3, help="Examples to run (default: 3)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)

    args = parser.parse_args()

    set_seed(args.seed)
    environment = capture_environment()

    print("=" * 64)
    print("TinyACE smoke test")
    print("=" * 64)
    print(f"  torch        {environment['torch_version']}")
    print(f"  transformers {environment['transformers_version']}")
    print(f"  commit       {environment['git_commit']}")
    print(f"  model        {args.model}")
    print(f"  task         {args.task} (first {args.limit} examples)")
    print(f"  device       {args.device or 'auto'}")
    print("=" * 64)

    config = get_model_config(args.model)
    config.max_new_tokens = args.max_new_tokens

    try:
        model, tokenizer = load_model_and_tokenizer(config.model_id, device_override=args.device)
    except Exception as e:
        print(f"\nFAIL: could not load {config.model_id}: {e}", file=sys.stderr)
        return 1

    try:
        dataset = load_dataset(args.task, args.limit)
    except Exception as e:
        print(f"\nFAIL: could not load task {args.task}: {e}", file=sys.stderr)
        return 1

    domain = get_task_config(args.task)["domain"]

    results, summary = run_dataset_baseline(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        domain=domain,
        config=config,
        model_id=config.model_id,
        task_name=args.task,
        option_shuffle_seed=args.seed,
    )

    print("\n" + "=" * 64)
    print("Checks")
    print("=" * 64)

    failures = []

    if not results:
        failures.append("no results produced")
    else:
        if summary.get("truncation_rate", 0) > 0:
            failures.append(f"{summary['truncation_rate']:.0%} of prompts truncated")

        template_rate = summary.get("chat_template_rate", 0)
        if template_rate < 1 and getattr(tokenizer, "chat_template", None):
            failures.append(f"chat template applied to only {template_rate:.0%} of prompts")

        if not any(r.get("pred", "").strip() for r in results):
            failures.append("every prediction was empty")

    tiers = summary.get("mapping_tier_distribution") or {}
    if tiers:
        print("  option mapping:", ", ".join(f"{k}={v:.0%}" for k, v in sorted(tiers.items())))
        if tiers.get("embedding", 0) > 0.5:
            print("    note: mostly embedding fallback -- OMA here is weak evidence")

    print(f"  examples      {summary.get('num_examples', 0)}")
    print(f"  chat template {summary.get('chat_template_rate', 0):.0%} of prompts")
    print(f"  truncated     {summary.get('truncation_rate', 0):.0%} of prompts")
    if summary.get("oma_accuracy") is not None:
        print(
            f"  OMA           {summary['oma_accuracy']:.0%} "
            f"(n={summary.get('num_examples')}, not a result at this n)"
        )
    print(f"  latency       {summary.get('avg_latency_ms', 0):.0f} ms/example")

    print("\n  sample prediction:")
    if results:
        sample = results[0]
        print(f"    Q:    {str(sample.get('question', ''))[:80]}")
        print(f"    pred: {str(sample.get('pred', ''))[:80]}")
        print(f"    gold: {str(sample.get('gold', ''))[:80]}")

    print("=" * 64)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print("PASS: pipeline is functional end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
