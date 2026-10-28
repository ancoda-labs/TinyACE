"""Model registry, screening rule, and data-root resolution."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Environment override for the directory holding `data/tasks/`, for the case
# where the package is installed away from its datasets.
DATA_ROOT_ENV = "TINYACE_DATA_ROOT"


def _find_repo_root() -> Path:
    """
    Locate the directory that holds `data/tasks/`.

    This was `Path(__file__).resolve().parents[3]`, which is the repo root only
    under an editable install. A real `pip install` puts the package in
    site-packages, where parents[3] is an unrelated directory -- and the
    datasets are not in the wheel either, since packages.find only takes
    `src/`. Searching upward makes the checkout case robust to layout changes,
    and the environment variable covers an installed package whose data lives
    elsewhere.

    Returns:
        The directory containing `data/tasks/`, or the editable-install guess
        when there is none, so error messages name a plausible path.
    """
    override = os.environ.get(DATA_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "data" / "tasks").is_dir():
            return candidate
    return here.parents[3]


REPO_ROOT = _find_repo_root()


# Models screened for the Nepali track.
#
# `nepali_capable` is not a claim -- it is filled in by the screening run
# (scripts/screen.py) and gates entry to the main grid. The rule is
# pre-registered: a model enters only if the lower bound of its Wilson interval
# on Nepali Belebele clears CHANCE_FLOOR by a margin, not if its point estimate
# happens to.
@dataclass(frozen=True)
class ModelSpec:
    """One checkpoint in the study."""

    key: str
    hf_id: str
    params: str
    generation: str  # "2026" or "2024", for the generational contrast
    # Hub commit from HfApi().model_info(hf_id).sha. A floating "main" would
    # move the weights under a reported number.
    revision: str
    # False turns the chat template's thinking switch off. None means the
    # checkpoint has no such switch; passing False to those templates is a
    # different prompt, and passing True makes Qwen3 think.
    enable_thinking: Optional[bool] = None


# Revisions looked up 2026-10-28. enable_thinking is False for Qwen3, whose
# template defaults to thinking. SmolLM3-3B was in this list and was removed
# the same day: with enable_thinking=False the rendered prompt still contains
# "Today Date: ..." from the template's strftime_now, so the string changes
# every day. It is not a study model.
MODELS: Dict[str, ModelSpec] = {
    m.key: m
    for m in [
        ModelSpec(
            "qwen3-1.7b",
            "Qwen/Qwen3-1.7B",
            "1.7B",
            "2026",
            "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            False,
        ),
        ModelSpec(
            "qwen3-4b",
            "Qwen/Qwen3-4B",
            "4B",
            "2026",
            "1cfa9a7208912126459214e8b04321603b3df60c",
            False,
        ),
        ModelSpec(
            "gemma-3-1b",
            "google/gemma-3-1b-it",
            "1B",
            "2026",
            "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
            None,
        ),
        ModelSpec(
            "gemma-3-4b",
            "google/gemma-3-4b-it",
            "4B",
            "2026",
            "093f9f388b31de276ce2de164bdc2081324b9767",
            None,
        ),
        ModelSpec(
            "phi-4-mini",
            "microsoft/Phi-4-mini-instruct",
            "3.8B",
            "2026",
            "cfbefacb99257ffa30c83adab238a50856ac3083",
            None,
        ),
        # Generational contrast: a 2024-era model on the same protocol.
        ModelSpec(
            "phi-3.5-mini",
            "microsoft/Phi-3.5-mini-instruct",
            "3.8B",
            "2024",
            "2fe192450127e6a83f7441aef6e3ca586c338b77",
            None,
        ),
        # Debug only. Random weights; never report a number from it.
        ModelSpec(
            "tiny-gpt2",
            "sshleifer/tiny-gpt2",
            "0.008B",
            "debug",
            "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be",
            None,
        ),
    ]
}

# How many of the 900 Belebele items the playbook is built on. The rest are the
# frozen evaluation split.
#
# One constant, because two of them is what let screening leak into the
# evaluation split: `screen_models` split at 400 while `run_arm` defaulted to
# 200, and `parallel_split` shuffles once and slices, so `shuffled[200:400]` was
# both screened on and scored on. Models were selected on 200 of the ~500 items
# their results are reported over.
#
# Screening must fit inside it, which is what the assertion below enforces.
ADAPTATION_SIZE = 400

# Belebele is 4-option, so chance is 0.25. The screening rule is stated in terms
# of the interval rather than the point estimate: at n=400 the Wilson halfwidth
# near 0.35 is about 4.7 points, so an observed 35% is consistent with a true
# 30%. Screen at SCREENING_N and require the lower bound to clear
# SCREENING_FLOOR.
CHANCE_FLOOR = 0.25
SCREENING_N = 400
SCREENING_FLOOR = 0.30

# Week 1 screening lineup: Qwen3 (1.7B, 4B) and Gemma-3 (1B-it, 4B-it) on both
# Belebele and Global-MMLU. Wilson lower bound > SCREENING_FLOOR locks the
# primary task/model set before Week 2 adaptation work.
SCREENING_MODELS = (
    "qwen3-1.7b",
    "qwen3-4b",
    "gemma-3-1b",
    "gemma-3-4b",
)
SCREENING_TASKS = (
    "belebele",
    "global_mmlu",
)

assert SCREENING_N <= ADAPTATION_SIZE, (
    f"Screening at n={SCREENING_N} cannot fit inside an adaptation split of "
    f"{ADAPTATION_SIZE}, so it would score items the study later reports on."
)


def screening_verdict(ci_low: float) -> bool:
    """
    Whether a model qualifies for the main grid.

    Args:
        ci_low: Lower bound of the Wilson interval on Nepali Belebele accuracy.

    Returns:
        True if the model clears the pre-registered floor.
    """
    return ci_low > SCREENING_FLOOR


def resolve_model(key_or_id: str) -> ModelSpec:
    """
    Look up a model by short key or HuggingFace id.

    Args:
        key_or_id: A key from MODELS, or a full HuggingFace id.

    Returns:
        The spec. Unregistered ids get a synthesised spec so an ad-hoc model
        still runs.
    """
    if key_or_id in MODELS:
        return MODELS[key_or_id]
    for spec in MODELS.values():
        if spec.hf_id == key_or_id:
            return spec
    # Not in the registry, so there is no pinned commit. "main" is the
    # revision HFLM already defaults to; it is not a study pin, and thinking
    # stays off the template because we do not know whether the model has
    # the switch.
    return ModelSpec(key_or_id, key_or_id, "unknown", "unknown", "main", None)


def model_keys(generation: str = None) -> List[str]:
    """Model keys, optionally filtered to one generation."""
    return [k for k, m in MODELS.items() if generation is None or m.generation == generation]
