"""Deterministic option shuffling for TinyACE lm-eval tasks.

``process_docs`` permutes multiple-choice options per item and records
``gold_option_idx``, so scoring and adaptation share one gold slot after the
shuffle. Both corpora put gold in a non-uniform slot, and a letter-biased small
model turns that into accuracy it has not earned.

**The seed key is the whole correctness story here.** It used to be
``MD5(passage + question)`` -- that is, the item's *text*. A translation is
different text, so English and Nepali hashed differently and the same item was
scored with its options in a different order in each language. Measured on the
900 Belebele items, 857 of them (95.2%) disagreed.

Nothing about that failure is visible downstream. Accuracies still compute,
McNemar still runs, a paired delta still gets reported -- it is simply no longer
paired, because item *i* in one language is not the same rendering as item *i*
in the other. That is the guard G1 names, and it is why the key is now an id the
*dataset* guarantees is identical across languages, never anything derived from
content.

Per corpus, the invariant key is:

    Global-MMLU   ``sample_id``       ("abstract_algebra/test/0")
    Belebele      ``link`` + ``question_number``
    normalised    ``id``              (already derived from the above)

``invariant_item_key`` raises when it can find none of those rather than falling
back to text, because a silent fallback is exactly the bug this module just had.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Union

_N_OPTIONS = 4
_LETTERS = ("A", "B", "C", "D")


def invariant_item_key(doc: Mapping[str, Any]) -> str:
    """
    The language-invariant key for one MCQ document.

    Args:
        doc: A raw Belebele or Global-MMLU row, or a normalised example.

    Returns:
        A key identical across every language variant of the same item.

    Raises:
        ValueError: If the document carries none of the known invariant keys.
            It does not fall back to hashing question text: that is precisely
            what broke the pairing, and a fallback would make the failure
            silent again the next time a caller passes an unexpected shape.
    """
    sample_id = doc.get("sample_id")
    if sample_id is not None and str(sample_id).strip():
        return f"gmmlu:{sample_id}"

    link, number = doc.get("link"), doc.get("question_number")
    if link is not None and str(link).strip() and number is not None:
        return f"belebele:{link}#{number}"

    item = doc.get("id")
    if item is not None and str(item).strip():
        return f"id:{item}"

    raise ValueError(
        "Cannot derive a language-invariant key for this document: it has no "
        "'sample_id', no ('link', 'question_number') and no 'id'. Refusing to "
        "fall back to the question text -- a translated item hashes "
        "differently in every language, which silently unpairs the study."
    )


def stable_item_hash(doc: Mapping[str, Any]) -> str:
    """Hex digest of the document's language-invariant key."""
    return hashlib.blake2s(invariant_item_key(doc).encode("utf-8"), digest_size=8).hexdigest()


def stable_item_seed(doc: Mapping[str, Any]) -> int:
    """
    Integer RNG seed derived from ``stable_item_hash``.

    An int rather than the string itself: ``random.Random(some_str)`` seeds from
    the string's hash, which ``PYTHONHASHSEED`` randomises per process, so the
    adaptation run and the evaluation run would permute the same item
    differently.
    """
    return int(stable_item_hash(doc), 16)


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
