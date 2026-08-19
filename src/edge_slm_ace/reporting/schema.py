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
        implemented: False for an arm that is designed but has no runner. A
            registered arm with no way to produce it is worse than an absent
            one: `run_arm` accepted the key and wrote a result identical to
            another arm under this arm's label. It now refuses instead.
    """

    key: str
    label: str
    family: str
    note: str = ""
    implemented: bool = True


# The registry. Order is the display order in figures and tables.
ARMS: List[Arm] = [
    Arm("baseline", "Baseline", "reference", "Bare task prompt, no instruction prefix."),
    Arm(
        "scaffold_control",
        "Scaffold Control",
        "control",
        "The identical instruction prefix over an empty playbook. The playbook "
        "claim is ace - scaffold_control, not ace - baseline. Named for what it "
        "is: under loglikelihood option scoring the model emits no text, so a "
        "chain-of-thought control is not expressible in this track.",
    ),
    Arm(
        "tinyace",
        "TinyACE",
        "ace",
        "Scaffold plus a playbook adapted on the adaptation split and frozen " "before evaluation.",
    ),
    Arm(
        "tinyace_retrieval",
        "TinyACE (retrieved)",
        "ace",
        "Per-question retrieval from the frozen playbook. Needs per-item "
        "scoring rather than a static prefix, so it cannot go through "
        "simple_evaluate and has no runner yet.",
        implemented=False,
    ),
    Arm(
        "tinyace_playbook_en",
        "Ablate: English playbook",
        "ace",
        "Nepali questions with an English-language playbook. Tests whether the "
        "lessons have to be in the question's language.",
    ),
    Arm(
        "tinyace_equal_lessons",
        "Ablate: equal lessons",
        "ace",
        "Budget by lesson count instead of tokens, so Devanagari fertility does "
        "not silently shrink the Nepali playbook.",
    ),
    Arm(
        "tinyace_ablate_no_relevance",
        "Ablate: No Relevance",
        "ace",
        "Retention-only ranking; every question receives the same lessons.",
    ),
    Arm("tinyace_ablate_no_vagueness", "Ablate: No Vagueness", "ace", "delta = 0."),
    Arm("tinyace_ablate_no_recency", "Ablate: No Recency", "ace", "gamma = 0."),
    Arm("tinyace_ablate_no_failure", "Ablate: No Failure", "ace", "beta = 0."),
    Arm(
        "tinyace_ablate_no_curator",
        "Ablate: No Curator",
        "ace",
        "Skip the Curator screening pass.",
    ),
    Arm(
        "tinyace_fifo",
        "Ablate: FIFO Eviction",
        "ace",
        "Oldest-first eviction instead of lowest-score.",
    ),
    Arm(
        "generative_cot",
        "Generative CoT",
        "generative",
        "Secondary track: the model generates reasoning and an answer, scored by "
        "a letter/option-text cascade. The only track where chain-of-thought is "
        "possible, and where the generative-vs-loglik gap is measured. No runner "
        "yet -- it needs a scoring path this project does not own.",
        implemented=False,
    ),
]

_ARMS_BY_KEY: Dict[str, Arm] = {arm.key: arm for arm in ARMS}

# The reference arm each family should be compared against. Ablations belong
# against full TinyACE, not against baseline -- comparing them to baseline
# measures ACE plus the ablation, not the ablated component.
DEFAULT_REFERENCE: Dict[str, str] = {
    "reference": "baseline",
    "control": "baseline",
    "ace": "scaffold_control",
    "generative": "baseline",
}

ABLATION_REFERENCE = "tinyace"


def implemented_arms() -> List[Arm]:
    """Arms a runner can actually produce."""
    return [arm for arm in ARMS if arm.implemented]


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
    return "ablate" in key or key in {
        "tinyace_fifo",
        "tinyace_playbook_en",
        "tinyace_equal_lessons",
    }


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
    ("qwen3-1.7b", "Qwen3-1.7B"),
    ("qwen3-4b", "Qwen3-4B"),
    ("gemma-3-1b", "Gemma3-1B"),
    ("gemma-3-4b", "Gemma3-4B"),
    ("gemma-3n-e4b", "Gemma3n-E4B"),
    ("phi-4-mini", "Phi-4-mini"),
    ("phi-3.5-mini", "Phi-3.5-mini"),
    ("smollm3-3b", "SmolLM3-3B"),
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
