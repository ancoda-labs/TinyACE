"""Canonical vocabulary for experiment results.

One definition of what an arm is called, what a model is called, and what the
result columns are named. Three scripts previously carried their own copies of
this and they disagreed: `plot_results.MODE_LABELS` mapped "baseline" to
"Baseline" while `tinyace_plots.normalize_mode` mapped the same value to
"zero_shot", so the same run appeared under two different names depending on
which script rendered it.

Anything that reads results should import from here rather than re-deriving
labels.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

# Canonical per-example result columns. Runners emit these; older runs used the
# names on the right, which `normalize_columns` maps forward.
CANONICAL_COLUMNS: Dict[str, str] = {
    "sample_id": "qid",
    "task_name": "task",
    "model_id": "model",
    "correct": "is_correct",
}


@dataclass(frozen=True)
class Arm:
    """One evaluation arm.

    Attributes:
        key: Canonical identifier, as written into results.
        label: Human-readable name for figures and tables.
        family: Grouping for plots -- "reference", "control", "ace" or
            "refine".
        note: Why the arm exists, shown in generated documentation.
    """

    key: str
    label: str
    family: str
    note: str = ""


# The registry. Order is the display order in figures and tables.
ARMS: List[Arm] = [
    Arm("baseline", "Baseline", "reference", "Terse prompt, no playbook."),
    Arm(
        "cot_control",
        "CoT Control",
        "control",
        "ACE prompt scaffold with an empty playbook. The playbook claim is "
        "ace - cot_control, not ace - baseline.",
    ),
    Arm("ace_full", "ACE Full", "ace", "Top-k retrieval from an unbounded playbook."),
    Arm("ace_working_memory", "TinyACE WM", "ace", "Token-budgeted working memory."),
    Arm("tinyace_wm_256", "TinyACE-256", "ace", "256-token prompt budget."),
    Arm("tinyace_wm_512", "TinyACE-512", "ace", "512-token prompt budget."),
    Arm(
        "tinyace_fifo",
        "Ablate: FIFO Eviction",
        "ace",
        "Oldest-first eviction instead of lowest-score.",
    ),
    Arm("tinyace_ablate_no_vagueness", "Ablate: No Vagueness", "ace", "delta = 0."),
    Arm("tinyace_ablate_no_recency", "Ablate: No Recency", "ace", "gamma = 0."),
    Arm("tinyace_ablate_no_failure", "Ablate: No Failure", "ace", "beta = 0."),
    Arm(
        "tinyace_ablate_no_curator", "Ablate: No Curator", "ace", "Skip the Curator screening pass."
    ),
    Arm(
        "tinyace_ablate_no_relevance",
        "Ablate: No Relevance",
        "ace",
        "Domain-only retrieval; every question gets the same lessons.",
    ),
    Arm(
        "self_refine",
        "Self-Refine",
        "refine",
        "Critique and rewrite using only the model's own output.",
    ),
    Arm(
        "self_refine_oracle",
        "Self-Refine (Oracle)",
        "refine",
        "Shown the gold answer during refinement. An upper bound, not a " "baseline.",
    ),
]

_ARMS_BY_KEY: Dict[str, Arm] = {arm.key: arm for arm in ARMS}

# The reference arm each family should be compared against. Ablations belong
# against full TinyACE, not against baseline -- comparing them to baseline
# measures ACE plus the ablation, not the ablated component.
DEFAULT_REFERENCE: Dict[str, str] = {
    "reference": "baseline",
    "control": "baseline",
    "ace": "cot_control",
    "refine": "baseline",
}

ABLATION_REFERENCE = "tinyace_wm_256"


def get_arm(key: str) -> Optional[Arm]:
    """Look up an arm by its canonical key, or None if unregistered."""
    return _ARMS_BY_KEY.get(key)


def arm_label(key: str) -> str:
    """
    Display label for an arm key.

    Args:
        key: Canonical arm key.

    Returns:
        The registered label, or a title-cased fallback for unknown keys.
    """
    arm = get_arm(key)
    if arm is not None:
        return arm.label
    return str(key).replace("_", " ").title()


def arm_order(key: str) -> int:
    """Sort position for an arm; unregistered arms sort last."""
    for index, arm in enumerate(ARMS):
        if arm.key == key:
            return index
    return len(ARMS)


def is_ablation(key: str) -> bool:
    """True when an arm is an ablation and belongs against ABLATION_REFERENCE."""
    return "ablate" in key or key == "tinyace_fifo"


def reference_for(key: str) -> str:
    """
    The arm a given arm should be compared against.

    Args:
        key: Canonical arm key.

    Returns:
        The key of the appropriate reference arm.
    """
    if is_ablation(key):
        return ABLATION_REFERENCE
    arm = get_arm(key)
    return DEFAULT_REFERENCE.get(arm.family if arm else "reference", "baseline")


# Short display names for models, longest pattern first so that "phi-3-mini"
# is not shadowed by "phi-3".
_MODEL_PATTERNS = [
    ("qwen2.5-1.5b", "Qwen2.5-1.5B"),
    ("qwen2.5-3b", "Qwen2.5-3B"),
    ("qwen2.5-7b", "Qwen2.5-7B"),
    ("phi-3-mini", "Phi-3-mini"),
    ("phi3-mini", "Phi-3-mini"),
    ("tinyllama", "TinyLlama-1.1B"),
    ("mistral-7b", "Mistral-7B"),
    ("llama-3.2-1b", "Llama-3.2-1B"),
    ("llama-3-8b", "Llama-3-8B"),
    ("tiny-gpt2", "tiny-gpt2"),
]


def model_label(model_id: str) -> str:
    """
    Short display name for a HuggingFace model id.

    Args:
        model_id: Full model id, e.g. "microsoft/Phi-3-mini-4k-instruct".

    Returns:
        A short label, e.g. "Phi-3-mini". Falls back to the final path segment.
    """
    lowered = str(model_id).lower()
    for pattern, label in _MODEL_PATTERNS:
        if pattern in lowered:
            return label
    return str(model_id).split("/")[-1]


def task_label(task_name: str) -> str:
    """Short display name for a task."""
    return str(task_name).replace("_tiny", "").replace("_", " ").strip()
