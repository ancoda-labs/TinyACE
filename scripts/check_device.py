#!/usr/bin/env python3
"""Score a fixed Global-MMLU Nepali adaptation prefix on one device.

A device comparison, not a study run and not a screening run. It loads
Qwen3-1.7B at the revision pinned in MODELS and scores the first ``--n``
ids of the frozen Global-MMLU adaptation manifest, in manifest order,
through ``run_frozen_eval``. The harness accuracy, the per-item
correctness, and the device, dtype and torch version are written to
``results/device_check/<device>.json``.

The language and the benchmark are fixed: two devices have to score the
same items. The seed does not choose the ids. ``--n`` only shortens that
prefix. Arguments this script does not pass stay at ``run_frozen_eval``'s
defaults, including the scaffold and the batch size. ``--dtype`` is
recorded even when it is unset, so a null dtype is the harness default
rather than a missing field.

Usage:
    python -m scripts.check_device --device cpu
    python -m scripts.check_device --device xpu --n 20
"""

import argparse
import json
import sys
from pathlib import Path

from edge_slm_ace.data.splits import frozen_split
from edge_slm_ace.harness import accuracy_of, per_item_correctness, run_frozen_eval
from edge_slm_ace.reporting import device_check_path
from edge_slm_ace.utils import DEFAULT_SEED, capture_environment, resolve_model, set_seed

# Passed as the registry key's HuggingFace id so run_frozen_eval's model_args
# carries MODELS[key].revision. A second from_pretrained would be a different
# loader, and could miss the pin.
MODEL_KEY = "qwen3-1.7b"
# The screening gate's task and language. Not a flag: two devices that
# scored different benchmarks would not be a comparison.
BENCHMARK = "global_mmlu"
LANGUAGE = "ne"
# Default length of the manifest-order prefix. Not SCREENING_N. --n may
# shorten it; it does not resample.
DEFAULT_N = 20


def adaptation_ids(n: int = DEFAULT_N) -> list:
    """
    The first ``n`` frozen adaptation ids, in manifest order.

    A shuffle before the slice would make the prefix depend on the seed,
    which is what ``run_arm --limit`` is careful not to do.
    """
    adapt_ids, _eval_ids = frozen_split(BENCHMARK)
    if n > len(adapt_ids):
        raise ValueError(
            f"device check asks for {n} {BENCHMARK} adaptation items but the "
            f"frozen adaptation split has {len(adapt_ids)}."
        )
    return list(adapt_ids[:n])


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--device",
        required=True,
        choices=["cpu", "cuda", "mps", "xpu"],
        help="Device the checkpoint is loaded on. The result file is named after it.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"Adaptation items, manifest order (default: {DEFAULT_N}).",
    )
    p.add_argument(
        "--dtype",
        default=None,
        help="Torch dtype passed to the harness. Recorded in the result file. "
        "Unset means the harness default.",
    )
    p.add_argument("--results-root", type=Path, default=Path("results"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.n < 1:
        print(f"Error: --n must be at least 1, got {args.n}.", file=sys.stderr)
        return 1
    set_seed(DEFAULT_SEED)
    spec = resolve_model(MODEL_KEY)
    try:
        item_ids = adaptation_ids(args.n)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        f"{spec.hf_id} @ {spec.revision} | {BENCHMARK}/{LANGUAGE} | "
        f"n={len(item_ids)} | device={args.device}",
        flush=True,
    )
    eval_kwargs = {
        "seed": DEFAULT_SEED,
        "device": args.device,
        "benchmark": BENCHMARK,
    }
    # None is the harness default. Passing it would be a second spelling of
    # "unset" and is not what the other runners do.
    if args.dtype is not None:
        eval_kwargs["dtype"] = args.dtype
    results = run_frozen_eval(
        spec.hf_id,
        LANGUAGE,
        item_ids,
        **eval_kwargs,
    )
    accuracy = accuracy_of(results, LANGUAGE, BENCHMARK)
    if accuracy is None:
        # Same failure as run_arm: writing first would leave a file whose
        # accuracy a later device diff would treat as measured.
        print(
            "Error: the harness reported no 'acc' for this run. Nothing was written.",
            file=sys.stderr,
        )
        return 1
    per_item = per_item_correctness(results, LANGUAGE, expected_ids=item_ids, benchmark=BENCHMARK)
    # Manifest order. per_item_correctness follows sample order, which is
    # not the manifest order, and a sorted id list would not be either.
    ordered = {item_id: per_item[item_id] for item_id in item_ids}
    environment = capture_environment()
    payload = {
        "model": spec.key,
        "hf_id": spec.hf_id,
        "revision": spec.revision,
        "device": args.device,
        "dtype": args.dtype,
        "torch_version": environment.get("torch_version"),
        "benchmark": BENCHMARK,
        "language": LANGUAGE,
        "n": len(item_ids),
        "seed": DEFAULT_SEED,
        "item_ids": item_ids,
        "accuracy": accuracy,
        "per_item": ordered,
        "environment": environment,
    }
    path = device_check_path(args.results_root, args.device)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"accuracy {accuracy:.1%} on n={len(item_ids)} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
