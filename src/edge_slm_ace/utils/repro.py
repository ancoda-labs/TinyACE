"""Seeding and environment capture for reproducible runs.

Every entrypoint must call `set_seed()` before loading a model, and record
`capture_environment()` in its run metadata. Without both, a reported number
cannot be regenerated: decoding is stochastic whenever temperature > 0, option
order is permuted per example, and generation behaviour changes across
transformers releases.
"""

import os
import platform
import random
import subprocess
from typing import Any, Dict, Optional

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """
    Seed every RNG that can affect a run.

    Args:
        seed: The seed to apply.
        deterministic: If True, also request deterministic cuDNN kernels.
            This costs throughput but removes a source of run-to-run drift.

    Returns:
        The seed that was applied (for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import transformers

        transformers.set_seed(seed)
    except ImportError:
        pass

    return seed


def _git_revision() -> Optional[str]:
    """Return the current git SHA, or None outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _git_is_dirty() -> Optional[bool]:
    """Return True if the working tree has uncommitted changes."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def capture_environment() -> Dict[str, Any]:
    """
    Snapshot everything needed to explain why a number came out the way it did.

    Returns:
        Dict with library versions, hardware and git state. Missing values are
        None rather than absent, so the schema is stable across machines.
    """
    env: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_revision(),
        "git_dirty": _git_is_dirty(),
        "torch_version": None,
        "transformers_version": None,
        "sentence_transformers_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_count": 0,
    }

    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    try:
        import transformers

        env["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    try:
        import sentence_transformers

        env["sentence_transformers_version"] = sentence_transformers.__version__
    except ImportError:
        pass

    return env
