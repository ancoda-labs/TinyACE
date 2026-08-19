"""Model registry, screening rule, and data-root resolution."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
# (scripts/screen_models.py) and gates entry to the main grid. The rule is
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


MODELS: Dict[str, ModelSpec] = {
    m.key: m
    for m in [
        ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B", "1.7B", "2026"),
        ModelSpec("qwen3-4b", "Qwen/Qwen3-4B", "4B", "2026"),
        ModelSpec("gemma-3-1b", "google/gemma-3-1b-it", "1B", "2026"),
        ModelSpec("gemma-3-4b", "google/gemma-3-4b-it", "4B", "2026"),
        ModelSpec("phi-4-mini", "microsoft/Phi-4-mini-instruct", "3.8B", "2026"),
        ModelSpec("smollm3-3b", "HuggingFaceTB/SmolLM3-3B", "3B", "2026"),
        # Generational contrast: a 2024-era model on the same protocol.
        ModelSpec("phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct", "3.8B", "2024"),
        # Debug only. Random weights; never report a number from it.
        ModelSpec("tiny-gpt2", "sshleifer/tiny-gpt2", "0.008B", "debug"),
    ]
}

# Belebele is 4-option, so chance is 0.25. The screening rule is stated in terms
# of the interval rather than the point estimate: at n=200 the Wilson halfwidth
# near 0.35 is about 6.6 points, so an observed 35% is consistent with a true
# 28% -- barely above chance. Screen at SCREENING_N and require the lower bound
# to clear SCREENING_FLOOR.
CHANCE_FLOOR = 0.25
SCREENING_N = 400
SCREENING_FLOOR = 0.30


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
    return ModelSpec(key_or_id, key_or_id, "unknown", "unknown")


def model_keys(generation: str = None) -> List[str]:
    """Model keys, optionally filtered to one generation."""
    return [k for k, m in MODELS.items() if generation is None or m.generation == generation]
