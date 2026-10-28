"""Shared fixtures.

Deliberately almost empty. It used to hold `tiny_model` and `tiny_tokenizer`,
which imported `edge_slm_ace.models.model_manager` -- a module deleted with the
old pipeline. Nothing used the fixtures, so nothing failed; they would have
raised ImportError the moment a test asked for one.

The checkpoint they loaded is exercised where it belongs now: the CI smoke test
runs a real arm end to end on `sshleifer/tiny-gpt2`.

Network-marked tests touch the Hugging Face Hub or the HF datasets cache
(ISSUES.md #12). They are skipped here when `HF_HOME` points at a path that
does not exist, or when the Hub is unreachable, so an offline laptop or an
unmounted cache volume does not fail the suite. `make test` / CI exclude them
with `-m "not network"`; `make test-all` runs them when the environment can.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

import pytest

_HUB_PROBE_URL = "https://huggingface.co"
_HUB_PROBE_TIMEOUT_S = 2.0


def _hub_is_reachable() -> bool:
    """True when a short HEAD/GET to the Hub succeeds."""
    try:
        with urlopen(_HUB_PROBE_URL, timeout=_HUB_PROBE_TIMEOUT_S) as response:
            response.read(1)
        return True
    except (URLError, TimeoutError, socket.timeout, OSError):
        return False


def network_skip_reason(*, hub_reachable: Optional[bool] = None) -> Optional[str]:
    """
    Why a `network` test should be skipped right now, or None to run it.

    Args:
        hub_reachable: Override the Hub probe. Tests inject False/True so the
            gate itself can be checked without depending on the live network.
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home is not None:
        path = Path(hf_home)
        if not path.exists():
            # Name the path: the #12 failure was an unmounted HF_HOME, and a
            # generic "cache unavailable" message would not have pointed at it.
            return f"HF_HOME points to a path that does not exist: {hf_home}"

    reachable = _hub_is_reachable() if hub_reachable is None else hub_reachable
    if not reachable:
        return "Hugging Face Hub is unreachable"
    return None


def pytest_collection_modifyitems(config, items):
    reason = network_skip_reason()
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
