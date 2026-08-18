"""Run infrastructure: configuration, device selection, reproducibility.

Deliberately narrow. Metrics, MCQ scoring and statistics used to live here
too, which meant "utils" contained the entire measurement layer; they are in
`edge_slm_ace.eval` now.
"""

from edge_slm_ace.utils.config import (
    ACE_MODE_FULL,
    ACE_MODE_WORKING,
    MODEL_CONFIGS,
    TASK_CONFIGS,
    ModelConfig,
    get_model_config,
    get_task_config,
    resolve_task_path,
    validate_task_registry,
)
from edge_slm_ace.utils.device_utils import get_device, resolve_device_override
from edge_slm_ace.utils.repro import DEFAULT_SEED, capture_environment, set_seed

__all__ = [
    # config
    "ACE_MODE_FULL",
    "ACE_MODE_WORKING",
    "MODEL_CONFIGS",
    "TASK_CONFIGS",
    "ModelConfig",
    "get_model_config",
    "get_task_config",
    "resolve_task_path",
    "validate_task_registry",
    # device
    "get_device",
    "resolve_device_override",
    # reproducibility
    "DEFAULT_SEED",
    "capture_environment",
    "set_seed",
]
