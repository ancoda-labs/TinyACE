"""Frozen split manifests: the splits are decided once, on disk, before any GPU runs.

G2 and G3. Everything the study reports rides on three id lists, and all three
are files in ``data/splits/`` rather than the output of a function called at run
time.

**Why files.** A split recomputed per run is a split that can change per run --
when a seed default moves, when a loader's sort changes, when someone passes
``--limit`` on a Tuesday. It changes silently, because every downstream number
still computes. Freezing the ids and verifying a checksum on load turns that
class of drift into a loud failure, and makes the pre-registration something a
reviewer can diff rather than a claim in a paper.

**Why a subsample at all.** ``study_split`` leaves 13,642 Global-MMLU items on
the evaluation side. At four forward passes each that is roughly 91 GPU-h for a
single task, before any arm or budget sweep. 2,000 items stratified by subject
give a Wilson halfwidth near 2.2 points -- comfortably inside the effects we are
trying to resolve -- and English evaluates the *same* sample_ids, so the
comparison stays paired.

The three manifests:

    global_mmlu_eval_2000    2,000 of 14,042 test items, stratified by subject
    global_mmlu_adapt_400    400 items, drawn from what the eval subsample left
    belebele_split           the passage-disjoint split, frozen as it stands

**What the checksums do and do not do.** ``ids_sha256`` is computed over the id
list and re-checked on load, so an accidental edit, a truncated file or a
half-written regeneration fails immediately. It is a tripwire, not a signature:
anyone can regenerate both the ids and the checksum. Git history is what makes
the freeze real, which is why the regeneration entry point is named to be
awkward and prints what it invalidates.
"""

from __future__ import annotations

import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from edge_slm_ace.utils.config import REPO_ROOT

SPLITS_DIR = Path("data/splits")

# Fixed at the point of pre-registration, and deliberately not DEFAULT_SEED:
# the run seed varies across the study's three seeds, and the *split* must not
# move when it does, or the three seeds would be scoring different items.
SUBSAMPLE_SEED = 20260915

EVAL_SUBSAMPLE_SIZE = 2000
ADAPT_SUBSAMPLE_SIZE = 400

# Manifest names, so a typo is an ImportError rather than a missing file at
# hour three of a grid run.
GLOBAL_MMLU_EVAL = "global_mmlu_eval_2000"
GLOBAL_MMLU_ADAPT = "global_mmlu_adapt_400"
BELEBELE_SPLIT = "belebele_split"

MANIFESTS = (GLOBAL_MMLU_EVAL, GLOBAL_MMLU_ADAPT, BELEBELE_SPLIT)


def manifest_path(name: str) -> Path:
    """Path to a frozen split manifest."""
    return REPO_ROOT / SPLITS_DIR / f"{name}.json"


def ids_checksum(item_ids: Iterable[str]) -> str:
    """
    Checksum over an id list, order-independent.

    Sorted before hashing, so a manifest rewritten in a different order still
    verifies. A split is a *set* of items, and treating a reordering as
    corruption would train people to regenerate manifests -- the one habit this
    module exists to prevent.
    """
    return hashlib.sha256("\n".join(sorted(item_ids)).encode("utf-8")).hexdigest()


def stratified_sample(
    examples: Sequence[Dict[str, Any]],
    size: int,
    seed: int,
    stratum_key: str = "subject",
    exclude: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Draw a stratified sample of item ids, deterministically.

    Proportional allocation with largest-remainder rounding, then a seeded draw
    within each stratum. Simple random sampling would leave the small subjects
    -- 24 of Global-MMLU's 57 hold about 100 items -- swinging enough between
    seeds to move a subject-level breakdown on sampling noise alone.

    Determinism does not depend on the caller: strata are visited in sorted
    order and each pool is sorted before shuffling, so the result is a function
    of the arguments and nothing else.

    Args:
        examples: Loaded examples, each with "id" and `stratum_key`.
        size: How many ids to draw.
        seed: Fixed at SUBSAMPLE_SEED for the study's own splits.
        stratum_key: Field to stratify on.
        exclude: Ids to remove before drawing, for carving a disjoint split out
            of what an earlier draw left.

    Returns:
        Sorted item ids.

    Raises:
        ValueError: If the requested size does not fit the eligible pool.
    """
    blocked = set(exclude or ())
    by_stratum: Dict[str, List[str]] = {}
    for example in examples:
        if example["id"] in blocked:
            continue
        by_stratum.setdefault(str(example[stratum_key]), []).append(example["id"])

    total = sum(len(pool) for pool in by_stratum.values())
    if not 0 < size <= total:
        raise ValueError(
            f"cannot draw {size} from an eligible pool of {total} ({len(blocked)} excluded)"
        )

    exact = {k: len(pool) * size / total for k, pool in by_stratum.items()}
    allocation = {k: int(v) for k, v in exact.items()}

    # Largest remainder, ties broken by stratum name so the result does not
    # depend on dict ordering. Only strata with room left can take a remainder
    # seat; without that guard a stratum whose exact share landed on a whole
    # number gets allocated one item more than it holds.
    shortfall = size - sum(allocation.values())
    eligible = sorted(
        (k for k in by_stratum if allocation[k] < len(by_stratum[k])),
        key=lambda k: (-(exact[k] - int(exact[k])), k),
    )
    for stratum in eligible[:shortfall]:
        allocation[stratum] += 1

    rng = random.Random(seed)
    chosen: List[str] = []
    for stratum in sorted(by_stratum):
        pool = sorted(by_stratum[stratum])
        rng.shuffle(pool)
        chosen.extend(pool[: allocation[stratum]])

    return sorted(chosen)


def write_manifest(name: str, payload: Dict[str, Any]) -> Path:
    """
    Write a frozen split manifest, stamping it with its own id checksum.

    Args:
        name: Manifest name, e.g. GLOBAL_MMLU_EVAL.
        payload: Everything except ``ids_sha256``, which is computed here.

    Returns:
        The path written.

    Raises:
        ValueError: If the payload carries no ``item_ids``.
    """
    if "item_ids" not in payload:
        raise ValueError("a split manifest must carry 'item_ids'")

    body = dict(payload)
    body["name"] = name
    body["ids_sha256"] = ids_checksum(body["item_ids"])
    body["item_ids"] = sorted(body["item_ids"])

    path = manifest_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@lru_cache(maxsize=None)
def load_manifest(name: str) -> Dict[str, Any]:
    """
    Load a frozen split manifest and verify its checksum.

    Cached, because ``assert_not_in_eval`` is called once per adaptation item
    and re-reading a 2,000-id file each time would make the guard expensive
    enough that somebody would eventually move it out of the loop.

    Args:
        name: Manifest name, e.g. GLOBAL_MMLU_EVAL.

    Returns:
        The manifest, with ``item_ids`` as a sorted list.

    Raises:
        FileNotFoundError: If the manifest is absent, naming how to rebuild it.
        ValueError: If the stored checksum does not match the ids.
    """
    path = manifest_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Frozen splits are committed; on a fresh "
            f"checkout it should be here. To rebuild it (and invalidate the "
            f"pre-registration) run `make splits-REGENERATE`."
        )

    body = json.loads(path.read_text(encoding="utf-8"))
    recorded, actual = body.get("ids_sha256"), ids_checksum(body["item_ids"])
    if recorded != actual:
        raise ValueError(
            f"{path.name}: ids_sha256 is {recorded}, but the ids hash to "
            f"{actual}. The manifest has been edited or truncated; this is no "
            f"longer the split the study pre-registered."
        )
    return body


def manifest_ids(name: str) -> List[str]:
    """The item ids of a frozen split."""
    return list(load_manifest(name)["item_ids"])


@lru_cache(maxsize=None)
def _eval_id_set(name: str) -> frozenset:
    return frozenset(load_manifest(name)["item_ids"])


def clear_manifest_cache() -> None:
    """
    Drop both manifest caches.

    There are two -- the parsed manifest and the id frozenset -- and clearing
    only the first leaves a stale id set behind, which in a test reads as the
    guard firing on the wrong fixture.
    """
    load_manifest.cache_clear()
    _eval_id_set.cache_clear()


def assert_not_in_eval(item_id: str, eval_manifest: str) -> None:
    """
    Refuse to adapt on an item the study evaluates on.

    G3. Called from inside the adaptation loop, per item, and not only where
    the split is constructed. A construction-time check verifies the split that
    was built; this verifies the item actually about to be reasoned over, which
    is what survives a ``--limit``, a resumed run, a hand-assembled id list, or
    a future caller that builds its own batch.

    Args:
        item_id: The id about to enter adaptation.
        eval_manifest: Name of the frozen evaluation manifest for this task.

    Raises:
        ValueError: If the item is in the evaluation split.
    """
    if item_id in _eval_id_set(eval_manifest):
        raise ValueError(
            f"{item_id} is in the frozen evaluation split ({eval_manifest}). "
            f"Adapting on it would tune the playbook on an item the study "
            f"reports over."
        )
