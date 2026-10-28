"""Nepali versus English token cost on one tokenizer.

The ratio has to be measured on the model that will spend the budget.
OpenAI's cl100k count is a different tokenizer, and a chat template adds
control tokens (and, for some models, today's date) that are not part of
the passage. ``token_ratio`` therefore calls ``encode`` with
``add_special_tokens=False`` and never applies a chat template.

``f`` is the ratio of totals, not the mean of per-passage ratios. Those
two disagree as soon as passages have different lengths, and the budget
is spent on the totals.
"""

import random
from typing import Any, Dict, List, Sequence, Tuple


def unique_parallel_passages(
    en_examples: Sequence[Dict],
    ne_examples: Sequence[Dict],
) -> Tuple[List[str], List[str]]:
    """
    One English/Nepali passage pair per ``passage_id``.

    Belebele asks several questions about one FLORES passage, so counting
    rows would weigh a passage by how many questions it has. The two
    language files are also not in the same row order, so the pair is the
    shared passage id, not the row index.

    Args:
        en_examples: ``load_belebele("en")`` records. ``context`` is the
            FLORES passage.
        ne_examples: The Nepali records, same schema.

    Returns:
        ``(en_texts, ne_texts)`` in passage-id order.

    Raises:
        ValueError: A passage id is missing on one side, or the same id
            carries two different passage strings.
    """
    en = _passages_by_id(en_examples, "en")
    ne = _passages_by_id(ne_examples, "ne")
    if set(en) != set(ne):
        missing_ne = sorted(set(en) - set(ne))
        missing_en = sorted(set(ne) - set(en))
        raise ValueError(
            "English and Nepali passages are not the same set: "
            f"missing in ne={missing_ne[:5]}, missing in en={missing_en[:5]}"
        )
    ids = sorted(en)
    return [en[i] for i in ids], [ne[i] for i in ids]


def _passages_by_id(examples: Sequence[Dict], language: str) -> Dict[str, str]:
    passages: Dict[str, str] = {}
    for example in examples:
        passage_id = example["passage_id"]
        text = example["context"]
        previous = passages.get(passage_id)
        if previous is not None and previous != text:
            raise ValueError(
                f"{language} passage {passage_id} has two different texts; "
                "a fertility count has to pick one and would do so silently"
            )
        passages[passage_id] = text
    return passages


def token_ratio(
    tokenizer,
    en_texts: Sequence[str],
    ne_texts: Sequence[str],
    n_boot: int = 10_000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Token cost of paired English and Nepali texts.

    Args:
        tokenizer: Anything with ``encode(text, add_special_tokens=False)``.
        en_texts: English strings, aligned with ``ne_texts``.
        ne_texts: Nepali strings, one per English string.
        n_boot: Bootstrap resamples. Each resample draws pairs with
            replacement, keeping the two languages together.
        seed: Seed for that resample. The token counts themselves do not
            use it.

    Returns:
        Totals, tokens per whitespace-delimited word, tokens per Unicode
        code point, ``f = ne_tokens / en_tokens``, and the 2.5/97.5
        percentiles of the paired bootstrap distribution of ``f``.

    Raises:
        ValueError: The two sides differ in length, a text is empty, or a
            total that the ratios divide by is zero.
    """
    if len(en_texts) != len(ne_texts):
        raise ValueError(
            f"English and Nepali have different lengths ({len(en_texts)} vs {len(ne_texts)}); "
            "f is only defined on pairs"
        )
    if not en_texts:
        raise ValueError("no text pairs")
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")

    en_counts = [_token_count(tokenizer, text) for text in en_texts]
    ne_counts = [_token_count(tokenizer, text) for text in ne_texts]
    en_tokens = sum(en_counts)
    ne_tokens = sum(ne_counts)
    en_words = sum(_word_count(text) for text in en_texts)
    ne_words = sum(_word_count(text) for text in ne_texts)
    en_characters = sum(_character_count(text) for text in en_texts)
    ne_characters = sum(_character_count(text) for text in ne_texts)
    _require_positive("en_tokens", en_tokens)
    _require_positive("ne_tokens", ne_tokens)
    _require_positive("en_words", en_words)
    _require_positive("ne_words", ne_words)
    _require_positive("en_characters", en_characters)
    _require_positive("ne_characters", ne_characters)

    # Draw the pairs after the token counts. The seed then depends only on
    # the resampling, not on how many times encode() was called.
    samples = _bootstrap_f(en_counts, ne_counts, n_boot, seed)
    return {
        "n_pairs": len(en_texts),
        "en_tokens": en_tokens,
        "ne_tokens": ne_tokens,
        "en_words": en_words,
        "ne_words": ne_words,
        "en_characters": en_characters,
        "ne_characters": ne_characters,
        "en_tokens_per_word": en_tokens / en_words,
        "ne_tokens_per_word": ne_tokens / ne_words,
        "en_tokens_per_character": en_tokens / en_characters,
        "ne_tokens_per_character": ne_tokens / ne_characters,
        "f": ne_tokens / en_tokens,
        "f_ci_low": _percentile(samples, 0.025),
        "f_ci_high": _percentile(samples, 0.975),
        "n_boot": n_boot,
        "seed": seed,
    }


def _token_count(tokenizer, text: str) -> int:
    if not isinstance(text, str) or text == "":
        raise ValueError("a fertility text is empty")
    # Special tokens and a chat template are not part of the passage. Counting
    # them would mix template overhead into the language ratio.
    ids = tokenizer.encode(text, add_special_tokens=False)
    count = len(ids)
    if count == 0:
        raise ValueError("tokenizer returned no tokens for a non-empty text")
    return count


def _word_count(text: str) -> int:
    """Whitespace-delimited words, the same split the playbook heuristic uses."""
    return len(text.split())


def _character_count(text: str) -> int:
    """Unicode code points, spaces included. A space is text the tokenizer sees."""
    return len(text)


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} is {value}; the ratio would be undefined")


def _bootstrap_f(
    en_counts: Sequence[int],
    ne_counts: Sequence[int],
    n_boot: int,
    seed: int,
) -> List[float]:
    """``f`` on ``n_boot`` resamples of the pairs, drawn with replacement."""
    rng = random.Random(seed)
    n = len(en_counts)
    samples: List[float] = []
    for _ in range(n_boot):
        en_total = 0
        ne_total = 0
        for _ in range(n):
            index = rng.randrange(n)
            en_total += en_counts[index]
            ne_total += ne_counts[index]
        if en_total <= 0:
            raise ValueError("a bootstrap resample has no English tokens")
        samples.append(ne_total / en_total)
    samples.sort()
    return samples


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    """
    Linear percentile of an already-sorted sample.

    Position ``p * (n - 1)`` with interpolation between the surrounding
    ranks. That is the usual definition, so a rerun with the same seed
    reproduces the same endpoints.
    """
    n = len(sorted_values)
    if n == 0:
        raise ValueError("no bootstrap samples")
    if n == 1:
        return float(sorted_values[0])
    pos = p * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    weight = pos - lo
    return float(sorted_values[lo]) * (1.0 - weight) + float(sorted_values[hi]) * weight
