"""Seeding and environment capture for reproducible runs.

Every entrypoint must call `set_seed()` before loading a model, and record
`capture_environment()` in its run metadata. Without both, a reported number
cannot be regenerated: the adaptation/evaluation split is a seeded shuffle, the
Reflector and Curator generations are stochastic whenever temperature > 0, and
generation behaviour changes across transformers releases.

The seed does not permute option order. Nothing does -- see
`data/belebele.py`.
"""

import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

# Present on a Jetson, absent everywhere else. The contents plus `nvpmodel -q`
# are what a power-mode difference has to be blamed on; a Mac recording
# None keeps the schema the same on both machines.
_TEGRA_RELEASE = Path("/etc/nv_tegra_release")

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
        # Same as CUDA: an unseeded XPU is a second RNG. is_available()
        # raises when the Level Zero stack is missing. CPU and CUDA were
        # seeded above; that raise must not undo them.
        xpu = getattr(torch, "xpu", None)
        try:
            if xpu is not None and xpu.is_available():
                xpu.manual_seed_all(seed)
        except Exception:
            pass
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


def _jetson_snapshot() -> Dict[str, Optional[str]]:
    """
    Jetson power-mode identity, or nulls when this is not a Jetson.

    Returns:
        ``nv_tegra_release`` is the file text. ``nvpmodel`` is what
        ``nvpmodel -q`` printed. Both are None when ``/etc/nv_tegra_release``
        is absent, and ``nvpmodel`` is None when the file is present but the
        command cannot be run.
    """
    if not _TEGRA_RELEASE.is_file():
        return {"nv_tegra_release": None, "nvpmodel": None}

    try:
        release = _TEGRA_RELEASE.read_text(encoding="utf-8", errors="replace").strip() or None
    except OSError:
        release = None

    nvpmodel = None
    try:
        out = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        rendered = ((out.stdout or "") + (out.stderr or "")).strip()
        nvpmodel = rendered or None
    except (OSError, subprocess.SubprocessError):
        nvpmodel = None

    return {"nv_tegra_release": release, "nvpmodel": nvpmodel}


def _xpu_snapshot(torch_module) -> Dict[str, Any]:
    """
    Intel XPU identity, or unavailable when this torch cannot use one.

    ``xpu_available`` is ``torch.xpu.is_available()``. ``xpu_device_name`` is
    device 0's name when that is true, else None. A torch build with no
    ``xpu`` attribute, and one whose runtime raises (a missing Level Zero
    stack), both keep the keys at the unavailable values. This runs from
    every entrypoint, so that raise must not stop a CPU run.
    """
    unavailable = {"xpu_available": False, "xpu_device_name": None}
    xpu = getattr(torch_module, "xpu", None)
    if xpu is None:
        return unavailable
    try:
        available = bool(xpu.is_available())
    except Exception:
        return unavailable
    if not available:
        return unavailable
    try:
        name = xpu.get_device_name(0)
    except Exception:
        name = None
    return {"xpu_available": True, "xpu_device_name": name}


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
        "xpu_available": False,
        "xpu_device_name": None,
        "lm_eval_version": None,
        "nv_tegra_release": None,
        "nvpmodel": None,
    }

    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_name"] = torch.cuda.get_device_name(0)
        # Beside CUDA, not in place of it. An XPU-only machine used to record
        # gpu_name=None, which is what a CPU machine records.
        env.update(_xpu_snapshot(torch))
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

    try:
        import lm_eval

        env["lm_eval_version"] = lm_eval.__version__
    except ImportError:
        pass

    env.update(_jetson_snapshot())
    return env
