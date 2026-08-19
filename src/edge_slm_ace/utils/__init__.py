"""Run infrastructure: model registry, split sizes, reproducibility.

Device selection is not here. `device_utils` offered `get_device`,
`resolve_device_override` and `time_function`; nothing called any of them --
every entrypoint passes `--device` straight through to the harness, which does
its own resolution -- and importing the module pulled torch in as a hard
dependency of anything that wanted the model registry.
"""

from edge_slm_ace.utils.config import (
    ADAPTATION_SIZE,
    CHANCE_FLOOR,
    DATA_ROOT_ENV,
    MODELS,
    REPO_ROOT,
    SCREENING_FLOOR,
    SCREENING_N,
    ModelSpec,
    model_keys,
    resolve_model,
    screening_verdict,
)
from edge_slm_ace.utils.repro import DEFAULT_SEED, capture_environment, set_seed

__all__ = [
    "ADAPTATION_SIZE",
    "CHANCE_FLOOR",
    "DATA_ROOT_ENV",
    "DEFAULT_SEED",
    "MODELS",
    "REPO_ROOT",
    "SCREENING_FLOOR",
    "SCREENING_N",
    "ModelSpec",
    "capture_environment",
    "model_keys",
    "resolve_model",
    "screening_verdict",
    "set_seed",
]
