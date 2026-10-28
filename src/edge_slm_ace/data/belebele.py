"""Belebele reading comprehension, in the shape the rest of the package expects.

Belebele is 900 multiple-choice questions over FLORES passages, released in 122
language variants. The variants are *parallel*: the same 900 questions, the same
correct option, translated. That is what makes an en/ne comparison paired rather
than merely matched, so a difference between the two can be tested with McNemar
on the same items.

Three properties of the raw files drive the design here, all verified against
`facebook/belebele` rather than assumed:

1. **The language files are not in the same row order.** They contain the same
   `(link, question_number)` keys, but positionally row *i* of `eng_Latn` is not
   row *i* of `npi_Deva`. Splitting or joining by index silently pairs unrelated
   questions -- so every id here is derived from the content key, never from
   position.
2. **`correct_answer_num` is a 1-based string**, not a 0-based int.
3. **Multiple questions share one FLORES passage** (488 passages for 900 items).
   The study split is therefore *passage-disjoint*: no `passage_id` may appear
   in both adaptation and evaluation, or the Reflector would have seen the
   passage text of items it must not score. That invariant is asserted, not
   merely documented.

Option order is permuted deterministically per item (see ``eval.utils``) so
residual position bias cannot favour a fixed gold slot. The same MD5(passage +
question) seed is used by the custom lm-eval tasks under ``eval/tasks/``, so
adaptation and harness evaluation see identical permutations.
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from edge_slm_ace.eval.tasks import BELEBELE_HARNESS_TASKS
from edge_slm_ace.eval.utils import permute_options, stable_item_seed
from edge_slm_ace.utils.config import ADAPTATION_SIZE, REPO_ROOT

# Short language code -> the FLORES-200 code Belebele files are named by.
BELEBELE_LANGUAGES: Dict[str, str] = {
    "en": "eng_Latn",
    "ne": "npi_Deva",
}

BELEBELE_DIR = Path("data/tasks/belebele")

# Fields every Belebele record must carry. Checked on load, because a silently
# renamed upstream field would mismap the gold answer rather than fail.
_REQUIRED_FIELDS = (
    "link",
    "question_number",
    "flores_passage",
    "question",
    "mc_answer1",
    "mc_answer2",
    "mc_answer3",
    "mc_answer4",
    "correct_answer_num",
)

_N_OPTIONS = 4


def belebele_path(language: str) -> Path:
    """
    Path to a Belebele language file.

    Args:
        language: Short code from BELEBELE_LANGUAGES, or a FLORES code.

    Returns:
        Absolute path, resolved against the data root.

    Raises:
        KeyError: If the language is not one this project uses.
    """
    if language in BELEBELE_LANGUAGES:
        flores = BELEBELE_LANGUAGES[language]
    elif language in BELEBELE_LANGUAGES.values():
        flores = language
    else:
        raise KeyError(
            f"Unknown Belebele language '{language}'. "
            f"Known: {sorted(BELEBELE_LANGUAGES)} or {sorted(BELEBELE_LANGUAGES.values())}"
        )
    return REPO_ROOT / BELEBELE_DIR / f"{flores}.jsonl"


def passage_id(record: Dict) -> str:
    """
    Stable identifier for the FLORES passage a question is about.

    Several questions share one `link`; that shared key is what the
    adaptation/evaluation split must keep disjoint.
    """
    digest = hashlib.blake2s(str(record["link"]).encode("utf-8"), digest_size=5).hexdigest()
    return f"bel-pass-{digest}"


def item_id(record: Dict) -> str:
    """
    Stable identifier for a question, shared across every language variant.

    `(link, question_number)` is unique across all 900 rows and identical
    between language files, which makes it the only safe join key. The link is
    a long URL, so it is hashed rather than embedded.

    Do not shorten the link to its last path segment: six different passages end
    in ".../Introduction", and collapsing them collides. The uniqueness check in
    `load_belebele` exists because that mistake mismaps gold answers rather than
    failing.

    Args:
        record: A raw Belebele record.

    Returns:
        An id of the form "bel-<10 hex chars>-q<question_number>".
    """
    digest = hashlib.blake2s(str(record["link"]).encode("utf-8"), digest_size=5).hexdigest()
    return f"bel-{digest}-q{record['question_number']}"


def _normalize(record: Dict, language: str, domain: str) -> Dict:
    """Convert one raw Belebele record into the package's example schema."""
    missing = [f for f in _REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"Belebele record missing fields: {', '.join(missing)}")

    gold = int(record["correct_answer_num"]) - 1  # 1-based in the file
    if not 0 <= gold < _N_OPTIONS:
        raise ValueError(f"correct_answer_num out of range: {record['correct_answer_num']!r}")

    options = [record[f"mc_answer{i}"] for i in range(1, _N_OPTIONS + 1)]
    options, gold = permute_options(options, gold, stable_item_seed(record))

    return {
        "id": item_id(record),
        "passage_id": passage_id(record),
        "language": language,
        "domain": domain,
        # The passage is the context. It is long -- 79 words on average in
        # English, up to 217 -- which is why prompt-length bounds and the
        # truncation health check matter more here than they did on SciQ.
        "context": record["flores_passage"],
        "question": record["question"],
        "options": options,
        "gold_option_idx": gold,
        "answer": options[gold],
    }


def load_belebele(
    language: str,
    path: Optional[Path] = None,
    domain: Optional[str] = None,
) -> List[Dict]:
    """
    Load one Belebele language, normalised and in a language-independent order.

    Rows are returned sorted by `id`, so `load_belebele("en")[i]` and
    `load_belebele("ne")[i]` are the same question. The files themselves are not
    in the same order, so this is the only safe way to line them up.

    Args:
        language: Short code ("en", "ne") or FLORES code.
        path: Override the file location.
        domain: Playbook domain to file lessons under. Defaults to
            "belebele_<language>", which keeps an English playbook and a Nepali
            playbook separate. Pass an explicit shared value to study
            cross-lingual playbook transfer.

    Returns:
        Examples in the package's schema, sorted by id.
    """
    source = path or belebele_path(language)
    short = next((k for k, v in BELEBELE_LANGUAGES.items() if v == source.stem), language)
    domain = domain or f"belebele_{short}"

    with open(source, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    examples = [_normalize(r, short, domain) for r in records]

    ids = [e["id"] for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{source} contains duplicate item ids")

    return sorted(examples, key=lambda e: e["id"])


def parallel_split(
    item_ids: Sequence[str],
    adaptation_size: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Split ids into an adaptation set and a frozen evaluation set.

    Deterministic in `(seed, ids)` and independent of the order ids arrive in,
    so every language and every arm gets the identical split. If the split were
    derived from row order, English and Nepali would adapt on different
    questions and the paired comparison would be meaningless.

    This is the item-level primitive. Belebele's study split goes through
    `passage_disjoint_split` instead, so shared FLORES passages cannot leak.

    Args:
        item_ids: All ids in the language.
        adaptation_size: How many to reserve for building the playbook.
        seed: Run seed.

    Returns:
        (adaptation_ids, eval_ids), both sorted.

    Raises:
        ValueError: If the adaptation set would take everything.
    """
    unique = sorted(set(item_ids))
    if not 0 < adaptation_size < len(unique):
        raise ValueError(f"adaptation_size must be in (0, {len(unique)}); got {adaptation_size}")

    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:adaptation_size]), sorted(shuffled[adaptation_size:])


def assert_zero_passage_overlap(
    adaptation_ids: Iterable[str],
    eval_ids: Iterable[str],
    examples: Sequence[Dict],
) -> None:
    """
    Fail hard if any FLORES passage appears on both sides of the split.

    Week 1 gate: ``assert passage_id not in eval_passages`` for every
    adaptation item. Prose is not enforcement.
    """
    by_id = {e["id"]: e for e in examples}
    adapt_passages: Set[str] = set()
    eval_passages: Set[str] = set()

    for iid in adaptation_ids:
        example = by_id[iid]
        pid = example["passage_id"]
        assert pid not in eval_passages, (
            f"passage_id {pid!r} already reserved for evaluation while building "
            f"adaptation (item {iid})"
        )
        adapt_passages.add(pid)

    for iid in eval_ids:
        example = by_id[iid]
        pid = example["passage_id"]
        assert pid not in adapt_passages, (
            f"passage_id {pid!r} overlaps adaptation and evaluation " f"(eval item {iid})"
        )
        eval_passages.add(pid)


def passage_disjoint_split(
    examples: Sequence[Dict],
    adaptation_size: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Split examples so no `passage_id` appears in both adaptation and evaluation.

    Passages are shuffled by `seed`, then packed greedily into adaptation until
    `adaptation_size` questions are filled (exact), topping up with leftover
    single- and double-question passages when the greedy pass undershoots.

    Returns:
        (adaptation_ids, eval_ids), both sorted and passage-disjoint.

    Raises:
        ValueError: If an exact passage-disjoint packing of `adaptation_size`
            cannot be formed.
    """
    by_passage: Dict[str, List[str]] = defaultdict(list)
    for example in examples:
        by_passage[example["passage_id"]].append(example["id"])

    for pid, ids in by_passage.items():
        by_passage[pid] = sorted(ids)

    passages = sorted(by_passage)
    rng = random.Random(seed)
    rng.shuffle(passages)

    adapt_ids: List[str] = []
    adapt_passages: Set[str] = set()
    eval_passages: Set[str] = set()

    for pid in passages:
        ids = by_passage[pid]
        if len(adapt_ids) + len(ids) <= adaptation_size:
            adapt_ids.extend(ids)
            adapt_passages.add(pid)
        else:
            eval_passages.add(pid)

    need = adaptation_size - len(adapt_ids)
    if need:
        # Prefer singles so we can hit the exact target without overshooting.
        for size in (1, 2):
            if need <= 0:
                break
            candidates = [p for p in passages if p in eval_passages and len(by_passage[p]) == size]
            for pid in candidates:
                if len(by_passage[pid]) > need:
                    continue
                adapt_ids.extend(by_passage[pid])
                adapt_passages.add(pid)
                eval_passages.discard(pid)
                need = adaptation_size - len(adapt_ids)
                if need <= 0:
                    break

    if len(adapt_ids) != adaptation_size:
        raise ValueError(
            f"passage-disjoint split could not hit adaptation_size={adaptation_size}; "
            f"got {len(adapt_ids)}"
        )

    eval_ids = [iid for pid in passages if pid in eval_passages for iid in by_passage[pid]]
    adapt_ids_sorted, eval_ids_sorted = sorted(adapt_ids), sorted(eval_ids)

    # Explicit Week 1 assertion: zero passage_id overlap.
    assert_zero_passage_overlap(adapt_ids_sorted, eval_ids_sorted, examples)
    for pid in adapt_passages:
        assert pid not in eval_passages

    return adapt_ids_sorted, eval_ids_sorted


def study_split(
    examples: Sequence[Dict],
    seed: int,
    adaptation_size: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """
    Passage-disjoint split of Belebele into adaptation and evaluation.

    Runners must not call this. They load ``data/splits/belebele_split.json``
    through ``frozen_split``, so a run seed cannot move the study onto a
    different set of items. ``scripts/freeze_splits.py`` calls
    ``passage_disjoint_split`` directly to write that file; this wrapper
    remains for the packer tests.

    Screening and the grid once each chose their own size -- 400 and 200 --
    and since the split is one shuffle sliced at that index, the 200 items
    between them were screened on *and* scored on. Model selection was
    therefore made on items the study reports over. Freezing the ids is what
    closed that.

    The split is passage-disjoint: sharing a FLORES passage across the boundary
    would let adaptation see evaluation context.

    Args:
        examples: Normalised Belebele examples (must carry `id` and
            `passage_id`).
        seed: Seed for the passage shuffle. The committed manifest was built
            at ``DEFAULT_SEED``. A run seed does not belong here.
        adaptation_size: Override, for tests of the packer. A manifest written
            with an override is not the study's split.

    Returns:
        (adaptation_ids, eval_ids), both sorted, disjoint, and passage-disjoint.
    """
    if not examples:
        raise ValueError("study_split requires at least one example")
    if "passage_id" not in examples[0]:
        raise ValueError(
            "study_split expects normalised Belebele examples with passage_id; "
            "pass load_belebele(...) output, not bare id strings"
        )
    return passage_disjoint_split(examples, adaptation_size or ADAPTATION_SIZE, seed)


# Custom lm-eval tasks (eval/tasks/) with deterministic option shuffling.
HARNESS_TASKS: Dict[str, str] = dict(BELEBELE_HARNESS_TASKS)


def harness_indices(
    language: str, item_ids: Sequence[str], path: Optional[Path] = None
) -> List[int]:
    """
    Positional indices the harness needs in order to evaluate `item_ids`.

    `lm_eval.simple_evaluate(samples={task: [...]})` selects documents by their
    position in the loaded dataset, which is raw file order. The English and
    Nepali files hold the same 900 questions in *different* orders, so passing
    one index list to both tasks would evaluate two disjoint question sets while
    looking exactly like a matched comparison.

    This maps a language-independent id set onto the positions that language
    actually stores them at.

    Args:
        language: Short code ("en", "ne") or FLORES code.
        item_ids: Ids to evaluate, from `parallel_split`.
        path: Override the file location.

    Returns:
        Sorted positional indices, one per id.

    Raises:
        KeyError: If an id is absent from this language file.
    """
    source = path or belebele_path(language)
    # Number the records, not the lines. `enumerate(f)` counted blank lines that
    # the filter then dropped, so one blank line anywhere in the file shifted
    # every position after it -- and the harness, which skips blanks when it
    # loads the split, would have scored the wrong document while reporting a
    # clean accuracy. That is the failure this whole function exists to prevent.
    positions = {}
    with open(source, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                positions[item_id(json.loads(line))] = len(positions)

    wanted = set(item_ids)
    missing = wanted - positions.keys()
    if missing:
        raise KeyError(
            f"{len(missing)} id(s) not present in {source.name}, " f"e.g. {sorted(missing)[:3]}"
        )
    return sorted(positions[i] for i in wanted)
