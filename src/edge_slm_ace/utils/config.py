"""Configuration utilities for models and device settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch


# ACE mode constants
ACE_MODE_FULL = "ace_full"
ACE_MODE_WORKING = "ace_working_memory"


@dataclass
class ModelConfig:
    """Configuration for a language model."""
    model_id: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    
    def __post_init__(self):
        """Validate configuration values."""
        assert 0.0 <= self.temperature <= 2.0, "Temperature must be in [0, 2]"
        assert 0.0 <= self.top_p <= 1.0, "top_p must be in [0, 1]"
        assert self.max_new_tokens > 0, "max_new_tokens must be positive"


# Default model configurations mapped to HuggingFace model IDs.
#
# Decoding is identical across every entry: greedy (temperature=0.0,
# top_p=1.0) with the same max_new_tokens. Previously mistral-7b and
# llama-3.2-1b defaulted to temperature=0.7 while everything else was greedy,
# and mistral-7b generated twice as many tokens, so any cross-model table
# mixed sampled and deterministic decoding at different output lengths.
# Override per run with --temperature/--top-p/--max-new-tokens if a study
# genuinely needs sampling, and record it (run_experiment writes the resolved
# values into metrics.json).
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "phi3-mini": ModelConfig(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens=256,
        temperature=0.0,  # Greedy decoding for reproducible evaluation
        top_p=1.0,
    ),
    "llama-3.2-1b": ModelConfig(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        max_new_tokens=256,
        temperature=0.0,
        top_p=1.0,
    ),
    "mistral-7b": ModelConfig(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens=256,
        temperature=0.0,
        top_p=1.0,
    ),
    "llama-3-8b": ModelConfig(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        max_new_tokens=256,
        temperature=0.0,
        top_p=1.0,
    ),
    # Tiny model for testing
    "tiny-gpt2": ModelConfig(
        model_id="sshleifer/tiny-gpt2",
        max_new_tokens=50,
        temperature=0.0,
        top_p=1.0,
    ),
    # Qwen models - use greedy decoding to avoid CUDA numerical instability
    # These are parameter-matched rivals to TinyLlama (1.1B), Phi-3 (3.8B), and Mistral (7B)
    "qwen-2.5-1.5b": ModelConfig(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=256,
        temperature=0.0,  # Greedy decoding for reproducibility
        top_p=1.0,
    ),
    "qwen-2.5-3b": ModelConfig(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens=256,
        temperature=0.0,  # Greedy decoding for reproducibility
        top_p=1.0,
    ),
    "qwen-2.5-7b": ModelConfig(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=256,
        temperature=0.0,  # Greedy decoding for reproducibility
        top_p=1.0,
    ),
    # TinyLlama - match settings for fair comparison
    "tinyllama-1.1b": ModelConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=256,
        temperature=0.0,  # Greedy decoding for reproducibility
        top_p=1.0,
    ),
}


def get_model_config(model_id_or_key: str) -> ModelConfig:
    """
    Get a model configuration by key or model ID.

    Args:
        model_id_or_key: Either a key from MODEL_CONFIGS or a HuggingFace model ID.

    Returns:
        ModelConfig: The configuration for the model.
    """
    if model_id_or_key in MODEL_CONFIGS:
        # Return a copy so modifications don't affect the default
        config = MODEL_CONFIGS[model_id_or_key]
        return ModelConfig(
            model_id=config.model_id,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
    else:
        # Check if it matches any model_id in the configs
        for key, config in MODEL_CONFIGS.items():
            if config.model_id == model_id_or_key:
                # Found a match, return a copy
                return ModelConfig(
                    model_id=config.model_id,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )

        # Not found anywhere - create default config with the provided model ID
        # Use temperature=0.0 for greedy decoding to avoid CUDA numerical issues
        return ModelConfig(model_id=model_id_or_key, temperature=0.0, top_p=1.0)


# Repository root, so task paths resolve from anywhere rather than from the
# caller's working directory.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Task registry: maps task names to dataset paths and domains.
#
# Every entry here must correspond to a file that ships with the repository.
# The registry previously advertised tatqa_tiny, medqa_train, math_train and
# sciq_train, none of which exist in data/tasks -- so the README's own example
# (--task-name tatqa_tiny) failed, and two of the three tasks in
# experiment_grid.yaml could not run. Use `validate_task_registry()` (and the
# accompanying test) to keep that from recurring.
TASK_CONFIGS: Dict[str, Dict[str, str]] = {
    # Tiny datasets for smoke tests
    "medqa_tiny": {
        "path": "data/tasks/medqa_tiny.json",
        "domain": "medical",
    },
    "iot_tiny": {
        "path": "data/tasks/iot_tiny.json",
        "domain": "iot",
    },
    "sciq_tiny": {
        "path": "data/tasks/sciq_tiny.json",
        "domain": "science",
    },
    # Full SciQ splits.
    #
    # sciq_val is the adaptation split: build and freeze a playbook here, then
    # evaluate read-only on sciq_test. The two share no questions (verified),
    # so this keeps the playbook off the scored set.
    "sciq_val": {
        "path": "data/tasks/sciq_val.json",
        "domain": "science",
    },
    "sciq_test": {
        "path": "data/tasks/sciq_test.json",
        "domain": "science",
    },
    "sciq_mcq_test": {
        "path": "data/tasks/sciq_mcq_test.jsonl",
        "domain": "science",
    },
}


def resolve_task_path(task_name: str) -> Path:
    """
    Absolute path to a task's dataset file.

    Args:
        task_name: Task name from TASK_CONFIGS.

    Returns:
        Absolute path, resolved against the repo root rather than the caller's
        working directory.
    """
    path = Path(get_task_config(task_name)["path"])
    return path if path.is_absolute() else REPO_ROOT / path


def validate_task_registry() -> Dict[str, str]:
    """
    Check that every registered task points at a file that exists.

    Returns:
        Mapping of task name -> missing path, empty when the registry is
        consistent.
    """
    return {
        name: str(resolve_task_path(name))
        for name in TASK_CONFIGS
        if not resolve_task_path(name).exists()
    }


def get_task_config(task_name: str) -> Dict[str, str]:
    """
    Get task configuration by name.
    
    Args:
        task_name: Task name from TASK_CONFIGS.
        
    Returns:
        Dict with 'path' and 'domain' keys.
        
    Raises:
        KeyError: If task_name is not found.
    """
    if task_name not in TASK_CONFIGS:
        raise KeyError(
            f"Task '{task_name}' not found. Available tasks: {list(TASK_CONFIGS.keys())}"
        )
    return TASK_CONFIGS[task_name]

