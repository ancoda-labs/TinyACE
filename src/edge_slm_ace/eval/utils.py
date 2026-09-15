"""Deterministic option shuffling for TinyACE lm-eval tasks.

``process_docs`` permutes multiple-choice options with an RNG seeded on the
MD5 of passage + question text, then records ``gold_option_idx`` so scoring
and adaptation share one gold slot after the shuffle.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Union

_N_OPTIONS = 4
_LETTERS = ("A", "B", "C", "D")


def stable_item_hash(doc: Mapping[str, Any]) -> str:
    """
    MD5 hex digest of passage + question text.

    Belebele uses ``flores_passage``; Global-MMLU has no passage, so the
    passage half is empty and the hash is driven by the question alone.
    """
    passage = str(doc.get("flores_passage") or doc.get("context") or doc.get("passage") or "")
    question = str(doc.get("question") or "").strip()
    return hashlib.md5(f"{passage}\n{question}".encode("utf-8")).hexdigest()


def stable_item_seed(doc: Mapping[str, Any]) -> int:
    """Integer RNG seed derived from ``stable_item_hash``."""
    return int(stable_item_hash(doc)[:16], 16)


def permute_options(options: Sequence[str], gold_idx: int, seed: int) -> tuple[List[str], int]:
    """
    Shuffle options with a seed, tracking the gold index.

    Returns:
        (shuffled_options, new_gold_idx).
    """
    if len(options) != _N_OPTIONS:
        raise ValueError(f"Expected {_N_OPTIONS} options, got {len(options)}")
    if not 0 <= gold_idx < _N_OPTIONS:
        raise ValueError(f"gold_idx out of range: {gold_idx}")

    indexed = list(enumerate(options))
    random.Random(seed).shuffle(indexed)
    shuffled = [text for _, text in indexed]
    new_gold = next(i for i, (old, _) in enumerate(indexed) if old == gold_idx)
    return shuffled, new_gold


def shuffle_mcq_doc(doc: Union[MutableMapping[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Return a copy of ``doc`` with options permuted and ``gold_option_idx`` set.

    Supports Belebele (``mc_answer*``, ``correct_answer_num``) and Global-MMLU
    (``option_*``, ``answer`` as A–D).
    """
    out = dict(doc)
    seed = stable_item_seed(out)

    if "mc_answer1" in out:
        options = [out[f"mc_answer{i}"] for i in range(1, _N_OPTIONS + 1)]
        gold = int(out["correct_answer_num"]) - 1
        shuffled, new_gold = permute_options(options, gold, seed)
        for i, text in enumerate(shuffled, start=1):
            out[f"mc_answer{i}"] = text
        out["correct_answer_num"] = str(new_gold + 1)
        out["gold_option_idx"] = new_gold
        out["options"] = shuffled
        return out

    if "option_a" in out:
        options = [out[f"option_{letter.lower()}"] for letter in _LETTERS]
        answer = str(out["answer"]).strip().upper()
        if answer not in _LETTERS:
            raise ValueError(f"Global-MMLU answer must be A–D, got {out['answer']!r}")
        gold = _LETTERS.index(answer)
        shuffled, new_gold = permute_options(options, gold, seed)
        for letter, text in zip(_LETTERS, shuffled):
            out[f"option_{letter.lower()}"] = text
        out["answer"] = _LETTERS[new_gold]
        out["gold_option_idx"] = new_gold
        out["options"] = shuffled
        return out

    if "options" in out and "gold_option_idx" in out:
        options = list(out["options"])
        gold = int(out["gold_option_idx"])
        shuffled, new_gold = permute_options(options, gold, seed)
        out["options"] = shuffled
        out["gold_option_idx"] = new_gold
        out["answer"] = shuffled[new_gold]
        return out

    raise ValueError(
        "Document is neither Belebele nor Global-MMLU MCQ shape; "
        "need mc_answer*, option_*, or options+gold_option_idx"
    )


def process_docs(dataset):
    """lm-eval ``process_docs`` entrypoint: shuffle every row deterministically."""
    return dataset.map(shuffle_mcq_doc)
