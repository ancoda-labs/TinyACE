"""Custom lm-evaluation-harness tasks for TinyACE.

Load with ``TaskManager(include_path=str(TASKS_DIR))`` or
``lm_eval --include_path <TASKS_DIR>``. Each task runs
``process_docs: !function utils.process_docs`` so option order is a
deterministic per-item permutation rather than the release order.
"""

from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent

# Day-1 task names from the build plan.
BELEBELE_HARNESS_TASKS = {
    "en": "tinyace_belebele_eng",
    "ne": "tinyace_belebele_npi",
}

GLOBAL_MMLU_HARNESS_TASKS = {
    "en": "tinyace_global_mmlu_en",
    "ne": "tinyace_global_mmlu_ne",
}

HARNESS_TASKS_BY_BENCHMARK = {
    "belebele": BELEBELE_HARNESS_TASKS,
    "global_mmlu": GLOBAL_MMLU_HARNESS_TASKS,
}

__all__ = [
    "TASKS_DIR",
    "BELEBELE_HARNESS_TASKS",
    "GLOBAL_MMLU_HARNESS_TASKS",
    "HARNESS_TASKS_BY_BENCHMARK",
]
