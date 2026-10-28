"""Global-MMLU (Nepali & English) loader and schema verification.

Pulls `CohereForAI/Global-MMLU` (Hub id resolves to CohereLabs/Global-MMLU) for
the `en` and `ne` configs. The test split is the screening/eval pool; the tiny
dev split is available for few-shot probes but is not the study split.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from edge_slm_ace.eval.utils import permute_options, stable_item_seed
from edge_slm_ace.utils.config import ADAPTATION_SIZE, REPO_ROOT

# User-facing Hub id from the Week 1 checklist. HF redirects CohereForAI → CohereLabs.
HF_DATASET = "CohereForAI/Global-MMLU"

GLOBAL_MMLU_LANGUAGES: Dict[str, str] = {
    "en": "en",
    "ne": "ne",
}

GLOBAL_MMLU_DIR = Path("data/tasks/global_mmlu")

EXPECTED_SIZES: Dict[str, int] = {
    "test": 14042,
    "dev": 285,
}

_REQUIRED_FIELDS = (
    "sample_id",
    "subject",
    "subject_category",
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "answer",
)

_LETTERS = ("A", "B", "C", "D")
_N_OPTIONS = 4

# lm-eval task names registered under eval/tasks/.
HARNESS_TASKS: Dict[str, str] = {
    "en": "tinyace_global_mmlu_en",
    "ne": "tinyace_global_mmlu_ne",
}


def globalmmlu_path(language: str, split: str = "test") -> Path:
    """Local jsonl cache path for one language/split (optional offline store)."""
    if language in GLOBAL_MMLU_LANGUAGES:
        config = GLOBAL_MMLU_LANGUAGES[language]
    elif language in GLOBAL_MMLU_LANGUAGES.values():
        config = language
    else:
        raise KeyError(
            f"Unknown Global-MMLU language '{language}'. " f"Known: {sorted(GLOBAL_MMLU_LANGUAGES)}"
        )
    return REPO_ROOT / GLOBAL_MMLU_DIR / f"{config}_{split}.jsonl"


def item_id(record: Dict[str, Any]) -> str:
    """
    Stable cross-language id for a Global-MMLU row.

    `sample_id` is already shared across language configs (e.g.
    ``abstract_algebra/test/0``), so it is the join key. Hashed only so the
    package id shape stays short and uniform with Belebele.
    """
    sample = str(record["sample_id"])
    digest = hashlib.blake2s(sample.encode("utf-8"), digest_size=5).hexdigest()
    return f"gmmlu-{digest}"


def _normalize(record: Dict[str, Any], language: str, domain: str) -> Dict[str, Any]:
    missing = [f for f in _REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"Global-MMLU record missing fields: {', '.join(missing)}")

    answer = str(record["answer"]).strip().upper()
    if answer not in _LETTERS:
        raise ValueError(f"answer must be A–D, got {record['answer']!r}")

    options = [record[f"option_{letter.lower()}"] for letter in _LETTERS]
    gold = _LETTERS.index(answer)
    seed = stable_item_seed(record)
    options, gold = permute_options(options, gold, seed)

    return {
        "id": item_id(record),
        "sample_id": str(record["sample_id"]),
        "language": language,
        "domain": domain,
        "subject": record["subject"],
        "subject_category": record.get("subject_category"),
        "context": "",
        "question": record["question"],
        "options": options,
        "gold_option_idx": gold,
        "answer": options[gold],
    }


def _load_split_from_hub(language: str, split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading Global-MMLU requires the `datasets` package "
            "(pulled in by lm-eval[hf]). Install with: pip install -e '.[dev]'"
        ) from exc

    config = GLOBAL_MMLU_LANGUAGES[language] if language in GLOBAL_MMLU_LANGUAGES else language
    ds = load_dataset(HF_DATASET, config, split=split)
    return [dict(row) for row in ds]


def load_globalmmlu(
    language: str,
    split: str = "test",
    path: Optional[Path] = None,
    domain: Optional[str] = None,
    shuffle_options: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load one Global-MMLU language split, normalised and sorted by id.

    Args:
        language: ``en`` / ``ne`` (or the Hub config name).
        split: ``test`` (default) or ``dev``.
        path: Optional local jsonl; otherwise load from the Hub.
        domain: Playbook domain. Defaults to ``global_mmlu_<language>``.
        shuffle_options: When False, skip the deterministic option permutation
            (verification / raw-schema checks only).

    Returns:
        Examples in the package schema, sorted by id.
    """
    short = (
        language
        if language in GLOBAL_MMLU_LANGUAGES
        else next((k for k, v in GLOBAL_MMLU_LANGUAGES.items() if v == language), language)
    )
    domain = domain or f"global_mmlu_{short}"

    if path is not None:
        import json

        with open(path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        records = _load_split_from_hub(short, split)

    if shuffle_options:
        examples = [_normalize(r, short, domain) for r in records]
    else:
        examples = []
        for r in records:
            missing = [f for f in _REQUIRED_FIELDS if f not in r]
            if missing:
                raise ValueError(f"Global-MMLU record missing fields: {', '.join(missing)}")
            answer = str(r["answer"]).strip().upper()
            options = [r[f"option_{letter.lower()}"] for letter in _LETTERS]
            gold = _LETTERS.index(answer)
            examples.append(
                {
                    "id": item_id(r),
                    "sample_id": str(r["sample_id"]),
                    "language": short,
                    "domain": domain,
                    "subject": r["subject"],
                    "subject_category": r.get("subject_category"),
                    "context": "",
                    "question": r["question"],
                    "options": options,
                    "gold_option_idx": gold,
                    "answer": options[gold],
                }
            )

    ids = [e["id"] for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Global-MMLU {short}/{split} contains duplicate item ids")

    return sorted(examples, key=lambda e: e["id"])


def verify_schema_and_size(
    language: str,
    split: str = "test",
    records: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Assert Hub (or provided) rows match the expected schema and cardinality.

    Returns:
        A summary dict suitable for printing or embedding in a check script.
    """
    if records is None:
        records = _load_split_from_hub(language, split)

    expected = EXPECTED_SIZES[split]
    if len(records) != expected:
        raise AssertionError(
            f"Global-MMLU {language}/{split}: expected {expected} rows, got {len(records)}"
        )

    for i, row in enumerate(records):
        missing = [f for f in _REQUIRED_FIELDS if f not in row]
        if missing:
            raise AssertionError(
                f"Global-MMLU {language}/{split} row {i} missing fields: {missing}"
            )
        answer = str(row["answer"]).strip().upper()
        if answer not in _LETTERS:
            raise AssertionError(
                f"Global-MMLU {language}/{split} row {i} bad answer {row['answer']!r}"
            )

    sample_ids = [str(r["sample_id"]) for r in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise AssertionError(f"Global-MMLU {language}/{split}: duplicate sample_id values")

    return {
        "dataset": HF_DATASET,
        "language": language,
        "split": split,
        "n": len(records),
        "expected_n": expected,
        "fields": list(_REQUIRED_FIELDS),
        "ok": True,
    }


def verify_parallelism(split: str = "test") -> Dict[str, Any]:
    """English and Nepali must share the same sample_id set (parallel corpus)."""
    en = _load_split_from_hub("en", split)
    ne = _load_split_from_hub("ne", split)
    en_ids = {str(r["sample_id"]) for r in en}
    ne_ids = {str(r["sample_id"]) for r in ne}
    if en_ids != ne_ids:
        raise AssertionError(
            f"Global-MMLU en/ne {split} sample_id sets differ: "
            f"{len(en_ids - ne_ids)} only-en, {len(ne_ids - en_ids)} only-ne"
        )
    en_gold = {str(r["sample_id"]): str(r["answer"]).strip().upper() for r in en}
    ne_gold = {str(r["sample_id"]): str(r["answer"]).strip().upper() for r in ne}
    disagree = [sid for sid in en_gold if en_gold[sid] != ne_gold[sid]]
    if disagree:
        raise AssertionError(
            f"Global-MMLU en/ne disagree on gold for {len(disagree)} items, " f"e.g. {disagree[:3]}"
        )
    return {"split": split, "n": len(en_ids), "parallel": True, "gold_agrees": True}


def study_split(
    item_ids: Sequence[str],
    seed: int,
    adaptation_size: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """
    Item-level adaptation / evaluation split for Global-MMLU.

    No passages, so this is ``parallel_split``. Runners must not call it.
    The study's Global-MMLU ids are the frozen subsample in ``data/splits/``,
    loaded through ``frozen_split``. Rebuilding from a seed draws a different
    400 and a different 2,000 from the full test set, which is not the split
    that was pre-registered.
    """
    from edge_slm_ace.data.belebele import parallel_split

    return parallel_split(item_ids, adaptation_size or ADAPTATION_SIZE, seed)


def harness_indices(
    language: str,
    item_ids: Sequence[str],
    split: str = "test",
) -> List[int]:
    """
    Positional indices in Hub load order for ``simple_evaluate(samples=...)``.

    Global-MMLU language configs share sample_ids but are not guaranteed to
    share row order, so indices are always computed per language.
    """
    records = _load_split_from_hub(language, split)
    positions = {item_id(r): i for i, r in enumerate(records)}
    wanted = set(item_ids)
    missing = wanted - positions.keys()
    if missing:
        raise KeyError(
            f"{len(missing)} id(s) not present in Global-MMLU {language}/{split}, "
            f"e.g. {sorted(missing)[:3]}"
        )
    return sorted(positions[i] for i in wanted)
