"""Frozen evaluation through lm-evaluation-harness.

Evaluation is a harness run over the custom TinyACE task YAMLs under
`eval/tasks/` (loaded via ``TaskManager(include_path=...)``). The playbook
enters as `system_instruction`, a static prefix, and the split enters as
`samples`, a list of document positions. Nothing about scoring is ours: no
answer parsing, no option mapping, no bespoke accuracy.

That is the point of freezing the playbook first. A playbook that updated during
evaluation would need our own loop, and our own loop is what the previous
version of this project had to withdraw its results over.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from edge_slm_ace.data.belebele import HARNESS_TASKS as BELEBELE_HARNESS_TASKS
from edge_slm_ace.data.belebele import harness_indices as belebele_harness_indices
from edge_slm_ace.data.globalmmlu import HARNESS_TASKS as GLOBAL_MMLU_HARNESS_TASKS
from edge_slm_ace.data.globalmmlu import harness_indices as globalmmlu_harness_indices
from edge_slm_ace.eval.tasks import TASKS_DIR
from edge_slm_ace.harness.prompts import CHOICE_LETTERS, build_system_instruction
from edge_slm_ace.utils.config import resolve_model

# Back-compat alias: Belebele remains the primary study task.
HARNESS_TASKS = BELEBELE_HARNESS_TASKS


def hflm_kwargs(
    model_id: str,
    *,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Keyword arguments for every ``HFLM(...)`` this study constructs.

    ``revision`` and ``enable_thinking`` come from the ModelSpec. A loader
    that omitted them would score ``main`` with thinking left at the
    template default, which for Qwen3 is on. ``enable_thinking``
    is always present, including None: that is how a model without the
    switch is distinguished from one where it was turned off.

    Args:
        model_id: Model key or HuggingFace id. Resolved to a ModelSpec.
        batch_size: Requests per forward batch, when this process builds the model.
        device: "cuda", "mps", "cpu", or None to let the harness decide.
        dtype: Torch dtype name.

    Returns:
        Arguments for ``HFLM`` or, via ``build_model_args``, for ``model_args``.
    """
    spec = resolve_model(model_id)
    kwargs: Dict[str, Any] = {
        "pretrained": spec.hf_id,
        "revision": spec.revision,
        "enable_thinking": spec.enable_thinking,
    }
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


def build_model_args(model_id: str, dtype: Optional[str] = None) -> str:
    """
    Render the `model_args` string the harness expects.

    ``batch_size`` and ``device`` are arguments of ``simple_evaluate`` itself,
    not of this string. The revision and the thinking switch are.

    Args:
        model_id: HuggingFace model id or a registry key.
        dtype: Optional torch dtype name, recorded so a run is reproducible.

    Returns:
        A comma-separated key=value string. ``enable_thinking=None`` is
        included on purpose: the harness parses it back to None.
    """
    kwargs = hflm_kwargs(model_id, dtype=dtype)
    parts = []
    for key in ("pretrained", "revision", "dtype", "enable_thinking"):
        if key not in kwargs:
            continue
        value = kwargs[key]
        if isinstance(value, bool):
            parts.append(f"{key}={value}")
        elif value is None:
            parts.append(f"{key}=None")
        else:
            parts.append(f"{key}={value}")
    return ",".join(parts)


def task_manager():
    """TaskManager that registers TinyACE YAMLs via ``--include_path``."""
    from lm_eval.tasks import TaskManager

    return TaskManager(include_path=str(TASKS_DIR))


def resolve_task(language: str, benchmark: str = "belebele") -> str:
    """Harness task name for a language on one of the Week 1 benchmarks."""
    if benchmark == "belebele":
        return BELEBELE_HARNESS_TASKS[language]
    if benchmark == "global_mmlu":
        return GLOBAL_MMLU_HARNESS_TASKS[language]
    raise KeyError(f"Unknown benchmark '{benchmark}'. Known: belebele, global_mmlu")


def resolve_indices(
    language: str, item_ids: Sequence[str], benchmark: str = "belebele"
) -> List[int]:
    """Document positions for `samples=` on the chosen benchmark."""
    if benchmark == "belebele":
        return belebele_harness_indices(language, item_ids)
    if benchmark == "global_mmlu":
        return globalmmlu_harness_indices(language, item_ids)
    raise KeyError(f"Unknown benchmark '{benchmark}'")


class TruncationCounter(logging.Handler):
    """
    Counts prompts the harness had to truncate.

    lm-eval truncates a too-long prompt **from the left** and logs a line. For
    Belebele the prompt is `P: <passage>` first, so left-truncation eats the
    passage -- the thing the question is about -- and the run degrades into
    guessing without failing.

    It is worse than a uniform loss. The instruction prefix and the playbook
    make the prompt longer, so the ACE arms truncate *more* than baseline on the
    same items, which is an arm-asymmetric confound rather than noise. Devanagari
    fertility pushes the Nepali side further again.

    So the count travels with the results. A run with a non-zero truncation rate
    is not comparable across arms and must not be reported as one.
    """

    MARKER = "exceeds model's maximum length"

    def __init__(self):
        super().__init__()
        self.events = 0
        self.tokens_dropped = 0
        self._attached = []

    def emit(self, record):
        message = record.getMessage()
        if self.MARKER not in message:
            return
        self.events += 1
        match = re.search(r"Truncating (\d+) tokens", message)
        if match:
            self.tokens_dropped += int(match.group(1))

    def __enter__(self):
        # Attach to the root logger only, and rely on propagation. Attaching to
        # the lm_eval loggers as well counts every propagating record twice --
        # which does not show up when those loggers do not exist yet, and does
        # as soon as anything has already logged through them.
        self._attached = [logging.getLogger()]
        lm_eval_logger = logging.getLogger("lm_eval")
        if not lm_eval_logger.propagate:
            self._attached.append(lm_eval_logger)
        for logger in self._attached:
            logger.addHandler(self)
        return self

    def __exit__(self, *exc):
        for logger in self._attached:
            logger.removeHandler(self)
        self._attached = []
        return False


def run_frozen_eval(
    model_id: str,
    language: str,
    item_ids: Sequence[str],
    lessons: Optional[Sequence[str]] = None,
    include_scaffold: bool = True,
    seed: int = 42,
    device: Optional[str] = None,
    batch_size: int = 8,
    dtype: Optional[str] = None,
    apply_chat_template: bool = True,
    log_samples: bool = True,
    lm=None,
    benchmark: str = "belebele",
) -> Dict[str, Any]:
    """
    Evaluate one arm on one language, over a fixed item set.

    Args:
        model_id: HuggingFace model id.
        language: "en" or "ne".
        item_ids: The evaluation split, from `study_split`. Converted to this
            language's document positions -- the two language files are in
            different row orders, so a shared index list would evaluate a
            different question set per language.
        lessons: Frozen playbook lessons, or None for a control arm.
        include_scaffold: False evaluates the bare baseline prompt.
        seed: Forwarded to every RNG the harness seeds.
        device: "cuda", "mps", "cpu", or None.
        batch_size: Requests per forward batch.
        dtype: Torch dtype name.
        apply_chat_template: Instruction-tuned checkpoints must be prompted in
            their own turn markers; fed a raw completion they continue the text
            instead of answering it.
        log_samples: Keep per-document records. Required for the paired McNemar
            test, and for verifying that `samples` selected the intended items.
        lm: A live `lm_eval.api.model.LM` to reuse instead of loading the
            checkpoint again. A grid runs a dozen arms per model, and reloading
            for each is the largest avoidable cost in the whole study.
        benchmark: ``belebele`` (default) or ``global_mmlu``.

    Returns:
        The harness results dict, with the arm's configuration recorded under
        "tinyace" so a number can be traced back to what produced it.
    """
    import lm_eval

    task = resolve_task(language, benchmark)
    indices = resolve_indices(language, item_ids, benchmark)
    instruction = build_system_instruction(lessons, include_scaffold=include_scaffold)

    with TruncationCounter() as truncation:
        results = lm_eval.simple_evaluate(
            model=lm if lm is not None else "hf",
            model_args=None if lm is not None else build_model_args(model_id, dtype),
            tasks=[task],
            task_manager=task_manager(),
            samples={task: indices},
            system_instruction=instruction or None,
            apply_chat_template=apply_chat_template,
            batch_size=batch_size,
            device=None if lm is not None else device,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            log_samples=log_samples,
        )

    n_items = len(indices)
    results["tinyace"] = {
        # Non-zero means the passage was clipped. Longer prefixes clip more, so
        # this is not comparable across arms when it fires.
        "truncated_prompts": truncation.events,
        "truncation_rate": truncation.events / (n_items * len(CHOICE_LETTERS)) if n_items else 0.0,
        "tokens_dropped": truncation.tokens_dropped,
        "model_id": model_id,
        "language": language,
        "benchmark": benchmark,
        "task": task,
        "include_path": str(TASKS_DIR),
        "n_items": n_items,
        "seed": seed,
        "num_lessons": len(lessons or []),
        "include_scaffold": include_scaffold,
        "system_instruction": instruction,
        "apply_chat_template": apply_chat_template,
    }
    return results


def accuracy_of(
    results: Dict[str, Any],
    language: str,
    benchmark: str = "belebele",
) -> Optional[float]:
    """
    Pull the headline accuracy out of a harness results dict.

    Reports `acc`, never `acc_norm`. The choices are the single tokens A-D, so
    length normalisation divides every option by the same length and `acc_norm`
    carries no information beyond `acc`.

    Args:
        results: A `run_frozen_eval` return value.
        language: "en" or "ne".
        benchmark: ``belebele`` or ``global_mmlu``.

    Returns:
        Accuracy, or None when the task is absent.
    """
    task = resolve_task(language, benchmark)
    return results.get("results", {}).get(task, {}).get("acc,none")


def per_item_correctness(
    results: Dict[str, Any],
    language: str,
    expected_ids: Optional[Sequence[str]] = None,
    benchmark: str = "belebele",
) -> Dict[str, int]:
    """
    Per-item correctness, keyed by our cross-language item id.

    Needed for the paired McNemar test: a per-arm accuracy cannot be paired, and
    en/ne comparisons only mean anything item by item.

    This is also the only place the `samples` selection can be checked, and it
    is the check that guards the study's largest untested assumption.
    `harness_indices` derives document positions from the committed jsonl, while
    the harness loads the same split from the Hub. Those two orders agreeing is
    an assumption, not a fact; if it ever fails, the run scores a different
    question set per arm and still reports a clean accuracy. Recovering the ids
    from the logged docs and comparing them to what was asked for costs nothing
    and closes it.

    Requires the run to have been made with `log_samples=True`.

    Args:
        results: A `run_frozen_eval` return value.
        language: "en" or "ne".
        expected_ids: The ids the run was asked to score. When given, a mismatch
            raises rather than returning a plausible-looking subset.
        benchmark: ``belebele`` or ``global_mmlu``.

    Returns:
        Mapping of item id -> 1/0.

    Raises:
        ValueError: If samples were not logged, if a sample is missing its doc
            or its `acc` metric, or if the ids scored are not the ids requested.
    """
    if benchmark == "global_mmlu":
        from edge_slm_ace.data.globalmmlu import item_id
    else:
        from edge_slm_ace.data.belebele import item_id

    task = resolve_task(language, benchmark)
    samples: List[Dict] = results.get("samples", {}).get(task, [])
    if not samples:
        raise ValueError(
            f"No logged samples for '{task}'. Per-item correctness needs "
            f"log_samples=True; without it the paired test has nothing to pair."
        )

    correctness: Dict[str, int] = {}
    for sample in samples:
        if "doc" not in sample:
            raise ValueError(f"A logged sample for '{task}' carries no doc to identify it.")
        if "acc" not in sample:
            # `.get("acc", 0)` recorded a missing metric as a wrong answer,
            # which is a silent accuracy loss rather than a failure.
            raise ValueError(
                f"A logged sample for '{task}' has no 'acc' metric. The task's "
                f"metric list changed upstream; do not score this run."
            )
        correctness[item_id(sample["doc"])] = int(sample["acc"])

    if expected_ids is not None:
        expected = set(expected_ids)
        scored = set(correctness)
        if scored != expected:
            raise ValueError(
                f"The harness scored a different item set than was requested for "
                f"'{task}': {len(scored - expected)} unexpected, "
                f"{len(expected - scored)} missing, of {len(expected)} requested. "
                f"harness_indices() maps positions in the committed jsonl; the "
                f"harness loads the split from the Hub. Those orders have "
                f"diverged, so this run compares different questions per arm."
            )

    return correctness
