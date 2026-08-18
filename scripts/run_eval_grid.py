#!/usr/bin/env python3
"""Grid experiment runner for systematic model × task × mode × device evaluation.

This script reads a YAML configuration file and runs experiments for all
combinations of models, tasks, modes, and devices.

Usage:
    # Dry run (print commands without executing)
    python -m scripts.run_eval_grid --config configs/experiment_grid.yaml --dry-run

    # Run all experiments
    python -m scripts.run_eval_grid --config configs/experiment_grid.yaml

    # Run with limit per experiment
    python -m scripts.run_eval_grid --config configs/experiment_grid.yaml --limit 10
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


# Keys a mode entry may carry. Anything else is a typo or a knob that no
# consumer reads, and both should be loud: configs/experiment_grid.yaml carried
# a documented `scoring:` block for months that nothing forwarded, so editing it
# changed no behaviour and reported no error.
KNOWN_MODE_KEYS = {
    "name",
    "mode",
    "description",
    "only_tasks",
    "ace_mode",
    "playbook_mode",
    "init_playbook",
    "token_budget",
    "working_memory_token_budget",
    "store_token_capacity",
    "top_k",
    "prune_every_n",
    "max_entries_per_domain",
    "reflect_on_correct_every_n",
    "relevance_weight",
    "disable_vagueness_penalty",
    "disable_recency_decay",
    "disable_failure_penalty",
    "fifo_memory",
    "no_curator",
}

# Retention-scoring hyperparameters, forwarded from the config's `scoring:`
# block and overridable per mode.
SCORING_KEYS = ("alpha", "beta", "gamma", "delta", "lambda_decay", "epsilon")


def validate_mode_configs(modes: List[Dict[str, Any]]) -> List[str]:
    """
    Report mode keys that no consumer reads.

    Args:
        modes: The `modes:` list from the grid config.

    Returns:
        One message per unrecognised key, empty when the config is clean.
    """
    problems = []
    allowed = KNOWN_MODE_KEYS | set(SCORING_KEYS)
    for mode_config in modes:
        name = mode_config.get("name", mode_config.get("mode", "unnamed"))
        for key in sorted(set(mode_config) - allowed):
            problems.append(f"mode '{name}': unrecognised key '{key}' (nothing reads it)")
    return problems


def load_grid_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment grid configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_device_availability(device: str) -> Tuple[bool, str]:
    """
    Check if a device is available.

    Args:
        device: Device name ("cpu", "cuda", "mps").

    Returns:
        Tuple of (is_available, fallback_device).
    """
    if device == "cpu":
        return True, "cpu"

    if device == "cuda":
        if torch.cuda.is_available():
            return True, "cuda"
        else:
            return False, "cpu"

    if device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, "mps"
        else:
            return False, "cpu"

    return False, "cpu"


def sanitize_for_path(s: str) -> str:
    """Sanitize a string for use in file paths."""
    return s.replace("/", "_").replace("-", "_").replace(" ", "_")


def build_output_dir(
    results_root: str,
    model_name: str,
    task_name: str,
    mode_name: str,
    device: str,
) -> Path:
    """Build the output directory path for an experiment."""
    return Path(results_root) / sanitize_for_path(model_name) / task_name / mode_name / device


def build_experiment_command(
    model_config: Dict[str, Any],
    task_config: Dict[str, Any],
    mode_config: Dict[str, Any],
    device: str,
    output_dir: Path,
    defaults: Dict[str, Any],
    limit: Optional[int],
    seed: int,
    scoring: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Build the command to run a single experiment."""
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_experiment",
        "--model-id",
        model_config["hf_id"],
        "--task-name",
        task_config["task_name"],
        "--mode",
        mode_config["mode"],
        "--output-path",
        str(output_dir / "results.csv"),
        "--metrics-path",
        str(output_dir / "metrics.json"),
        "--predictions-path",
        str(output_dir / "predictions.jsonl"),
        "--device",
        device,
    ]

    # Add ACE-specific parameters. cot_control shares the ACE prompt path
    # (that is the point of it), so it takes the same scaffold arguments.
    if mode_config["mode"] in ("ace", "cot_control"):
        ace_mode = mode_config.get("ace_mode", "ace_full")
        cmd.extend(["--ace-mode", ace_mode])

        # Playbook path
        playbook_path = output_dir / "playbook.jsonl"
        cmd.extend(["--playbook-path", str(playbook_path)])

        # Frozen evaluation: adapt on one split, freeze, score read-only on
        # another. `init_playbook` is templated on the model so one entry in
        # the config covers every model in the grid.
        playbook_mode = mode_config.get("playbook_mode", "learn")
        if playbook_mode == "frozen":
            init_playbook = str(mode_config["init_playbook"]).format(
                model=sanitize_for_path(model_config["name"]),
                task=task_config["task_name"],
                device=device,
                results_root=defaults.get("results_root", "results"),
            )
            cmd.extend(["--playbook-mode", "frozen", "--init-playbook", init_playbook])

        # Token budget (for working memory mode)
        # Support both token_budget and working_memory_token_budget
        token_budget = mode_config.get("working_memory_token_budget") or mode_config.get(
            "token_budget"
        )
        if token_budget:
            cmd.extend(["--token-budget", str(token_budget)])

        # Top-k
        if "top_k" in mode_config:
            cmd.extend(["--top-k", str(mode_config["top_k"])])

        # Pruning settings
        if "prune_every_n" in mode_config:
            cmd.extend(["--prune-every-n", str(mode_config["prune_every_n"])])
        if "max_entries_per_domain" in mode_config:
            cmd.extend(["--max-entries-per-domain", str(mode_config["max_entries_per_domain"])])

        # Ablation flags (learning-only; a control has nothing to ablate)
        if mode_config.get("disable_vagueness_penalty"):
            cmd.extend(["--disable-vagueness-penalty"])
        if mode_config.get("disable_recency_decay"):
            cmd.extend(["--disable-recency-decay"])
        if mode_config.get("disable_failure_penalty"):
            cmd.extend(["--disable-failure-penalty"])
        if mode_config.get("fifo_memory"):
            cmd.extend(["--fifo-memory"])
        if mode_config.get("no_curator"):
            cmd.extend(["--no-curator"])
        if "relevance_weight" in mode_config:
            cmd.extend(["--relevance-weight", str(mode_config["relevance_weight"])])
        if "store_token_capacity" in mode_config:
            cmd.extend(["--store-token-capacity", str(mode_config["store_token_capacity"])])
        if "reflect_on_correct_every_n" in mode_config:
            cmd.extend(
                ["--reflect-on-correct-every-n", str(mode_config["reflect_on_correct_every_n"])]
            )

        # Retention-scoring weights: the config-level `scoring:` block, with
        # per-mode overrides on top.
        resolved_scoring = dict(scoring or {})
        resolved_scoring.update({k: mode_config[k] for k in SCORING_KEYS if k in mode_config})
        for key in SCORING_KEYS:
            if key in resolved_scoring:
                cmd.extend([f"--{key.replace('_', '-')}", str(resolved_scoring[key])])

    # Add limit if specified
    effective_limit = limit or defaults.get("limit")
    if effective_limit:
        cmd.extend(["--limit", str(effective_limit)])

    # Add generation parameters from defaults
    if "max_new_tokens" in defaults:
        cmd.extend(["--max-new-tokens", str(defaults["max_new_tokens"])])
    if "temperature" in defaults:
        cmd.extend(["--temperature", str(defaults["temperature"])])
    if "top_p" in defaults:
        cmd.extend(["--top-p", str(defaults["top_p"])])

    # Seed (forwarded so every cell of the grid is reproducible)
    cmd.extend(["--seed", str(seed)])

    # Run name
    run_name = f"{model_config['name']}_{task_config['name']}_{mode_config['name']}_{device}"
    cmd.extend(["--run-name", run_name])

    return cmd


def run_experiment(
    cmd: List[str],
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
    show_progress: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Run a single experiment.

    Args:
        cmd: Command to run.
        output_dir: Output directory for the experiment.
        dry_run: If True, just print the command without running.
        verbose: If True, print progress information.
        show_progress: If True, show live output from the experiment.

    Returns:
        Tuple of (success, error_message).
    """
    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return True, None

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create log files
    stdout_log = output_dir / "stdout.log"
    stderr_log = output_dir / "stderr.log"
    combined_log = output_dir / "combined.log"

    # Run the experiment
    try:
        if show_progress:
            # Stream output to terminal AND capture to logs
            # Open log files for writing
            with (
                open(stdout_log, "w", encoding="utf-8") as stdout_f,
                open(stderr_log, "w", encoding="utf-8") as stderr_f,
                open(combined_log, "w", encoding="utf-8") as combined_f,
            ):

                # Run with streaming output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line buffered
                )

                import threading

                # Thread to read stdout
                def read_stdout():
                    for line in process.stdout:
                        print(line, end="")  # Print to terminal
                        stdout_f.write(line)  # Write to stdout log
                        combined_f.write(line)  # Write to combined log
                        stdout_f.flush()
                        combined_f.flush()

                # Thread to read stderr
                def read_stderr():
                    for line in process.stderr:
                        print(line, end="", file=sys.stderr)  # Print to terminal
                        stderr_f.write(line)  # Write to stderr log
                        combined_f.write(line)  # Write to combined log
                        stderr_f.flush()
                        combined_f.flush()

                # Start threads
                stdout_thread = threading.Thread(target=read_stdout)
                stderr_thread = threading.Thread(target=read_stderr)
                stdout_thread.start()
                stderr_thread.start()

                # Wait for process to complete
                returncode = process.wait()

                # Wait for threads to finish
                stdout_thread.join()
                stderr_thread.join()

                combined_f.write(f"\n\n=== EXIT CODE: {returncode} ===\n")

                if returncode == 0:
                    return True, None
                else:
                    return False, f"Exit code: {returncode} (logs: {combined_log})"
        else:
            # Original behavior: capture all output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

        # Save logs to files
        with open(stdout_log, "w", encoding="utf-8") as f:
            f.write(result.stdout)

        with open(stderr_log, "w", encoding="utf-8") as f:
            f.write(result.stderr)

        with open(combined_log, "w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr)
            f.write(f"\n\n=== EXIT CODE: {result.returncode} ===\n")

        if result.returncode == 0:
            return True, None
        else:
            error_msg = result.stderr or result.stdout or f"Exit code: {result.returncode}"
            # Truncate error message but mention log file
            error_preview = error_msg[:500] + "..." if len(error_msg) > 500 else error_msg
            return False, f"{error_preview}\n(Full logs saved to {combined_log})"

    except Exception as e:
        # Save exception to logs
        with open(combined_log, "w", encoding="utf-8") as f:
            f.write(f"=== EXCEPTION ===\n{str(e)}\n")
        return False, f"{str(e)} (logs: {combined_log})"


def main() -> int:
    """Main entry point for the grid runner."""
    parser = argparse.ArgumentParser(
        description="Run experiments for all combinations in a grid configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment grid YAML configuration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed forwarded to every experiment in the grid (default: 42)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override limit per experiment (for quick testing)",
    )
    parser.add_argument(
        "--only-task",
        type=str,
        action="append",
        default=None,
        help=(
            "Restrict the grid to these tasks (repeatable). The frozen arm "
            "reads a playbook the adaptation split produced, so the two stages "
            "have to run in order: --only-task sciq_val, then "
            "--only-task sciq_test. `make adapt && make evaluate` does this."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        default=True,
        help="Show live progress from experiments (default: True)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live progress output (only save to logs)",
    )
    parser.add_argument(
        "--skip-unavailable-devices",
        action="store_true",
        default=True,
        help="Skip experiments for unavailable devices (default: True)",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    try:
        config = load_grid_config(config_path)
    except Exception as e:
        print(f"Error: Failed to load config: {e}")
        return 1

    # Extract configuration sections
    models = config.get("models", [])
    tasks = config.get("tasks", [])
    modes = config.get("modes", [])
    devices = config.get("devices", ["cpu"])
    defaults = config.get("defaults", {})
    scoring = config.get("scoring", {})

    problems = validate_mode_configs(modes)
    if problems:
        print("Error: the config sets keys that nothing reads:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    if not models:
        print("Error: No models specified in config")
        return 1
    if not tasks:
        print("Error: No tasks specified in config")
        return 1
    if not modes:
        print("Error: No modes specified in config")
        return 1

    # Calculate total experiments
    total_experiments = len(models) * len(tasks) * len(modes) * len(devices)

    print("=" * 60)
    print("Experiment Grid Runner")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Models: {len(models)}")
    print(f"Tasks: {len(tasks)}")
    print(f"Modes: {len(modes)}")
    print(f"Devices: {len(devices)}")
    print(f"Total combinations: {total_experiments}")
    if args.dry_run:
        print("Mode: DRY RUN")
    print("=" * 60)

    # Track results
    results = {
        "success": [],
        "failed": [],
        "skipped": [],
    }

    start_time = time.time()
    experiment_num = 0

    # Iterate over all combinations
    for model_config in models:
        model_name = model_config.get("name", model_config.get("hf_id", "unknown"))

        for task_config in tasks:
            task_name = task_config.get("name", task_config.get("task_name", "unknown"))

            if args.only_task and task_config.get("task_name") not in args.only_task:
                continue

            for mode_config in modes:
                mode_name = mode_config.get("name", mode_config.get("mode", "unknown"))

                # A mode may restrict itself to particular tasks. The frozen
                # arm needs this: pairing it with sciq_val would have it read
                # the playbook adapted on sciq_val and then score on sciq_val,
                # which is the leak the arm exists to avoid.
                only_tasks = mode_config.get("only_tasks")
                if only_tasks and task_config.get("task_name") not in only_tasks:
                    continue

                for device in devices:
                    experiment_num += 1
                    experiment_id = f"{model_name}/{task_name}/{mode_name}/{device}"

                    print(f"\n[{experiment_num}/{total_experiments}] {experiment_id}")

                    # Check device availability
                    device_available, fallback_device = check_device_availability(device)

                    if not device_available:
                        if args.skip_unavailable_devices:
                            print(f"  ⚠ Device '{device}' not available, skipping")
                            results["skipped"].append(
                                {
                                    "experiment": experiment_id,
                                    "reason": f"Device '{device}' not available",
                                }
                            )
                            continue
                        else:
                            print(f"  ⚠ Device '{device}' not available, using '{fallback_device}'")
                            device = fallback_device

                    # Build output directory
                    results_root = defaults.get("results_root", "results")
                    output_dir = build_output_dir(
                        results_root, model_name, task_name, mode_name, device
                    )

                    # Build command
                    cmd = build_experiment_command(
                        model_config=model_config,
                        task_config=task_config,
                        mode_config=mode_config,
                        device=device,
                        output_dir=output_dir,
                        defaults=defaults,
                        limit=args.limit,
                        seed=args.seed,
                        scoring=scoring,
                    )

                    if args.verbose or args.dry_run:
                        print(f"  Output: {output_dir}")

                    # Run experiment
                    show_progress = args.show_progress and not args.no_progress
                    success, error_msg = run_experiment(
                        cmd=cmd,
                        output_dir=output_dir,
                        dry_run=args.dry_run,
                        verbose=args.verbose,
                        show_progress=show_progress,
                    )

                    if success:
                        if not args.dry_run:
                            print("  ✓ Success")
                        results["success"].append(
                            {
                                "experiment": experiment_id,
                                "output_dir": str(output_dir),
                            }
                        )
                    else:
                        print(f"  ✗ Failed: {error_msg[:200] if error_msg else 'Unknown error'}")
                        results["failed"].append(
                            {
                                "experiment": experiment_id,
                                "error": error_msg,
                            }
                        )

    # Print summary
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Grid Run Complete")
    print("=" * 60)
    print(f"Total time: {elapsed_time:.1f}s")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Skipped: {len(results['skipped'])}")

    if results["failed"]:
        print("\nFailed experiments:")
        for failure in results["failed"]:
            print(f"  - {failure['experiment']}")

    if results["skipped"]:
        print("\nSkipped experiments:")
        for skipped in results["skipped"]:
            print(f"  - {skipped['experiment']}: {skipped['reason']}")

    print("=" * 60)

    # Return non-zero if any experiments failed
    return 1 if results["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
