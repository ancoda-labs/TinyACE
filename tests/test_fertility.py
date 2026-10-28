"""Nepali/English token ratio on the study tokenizers.

The README used to claim a ~4.5x cl100k inflation with no script behind it.
``token_ratio`` is that measurement: each model's own tokenizer, no chat
template, and a paired bootstrap so the interval is about the same passages
in both languages. A mean of per-passage ratios is a different number.
"""

import json
import random
from pathlib import Path

import pytest

from edge_slm_ace.eval.fertility import token_ratio, unique_parallel_passages
from edge_slm_ace.reporting.layout import fertility_csv, fertility_json
from edge_slm_ace.utils.config import MODELS
from scripts.measure_fertility import measure_all, study_models, write_fertility


class _CharTokenizer:
    """One token per Unicode code point. Deterministic, and not a Hub model."""

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append(add_special_tokens)
        return list(text)


def _ratio(en, ne, **kwargs):
    kwargs.setdefault("n_boot", 32)
    kwargs.setdefault("seed", 0)
    return token_ratio(_CharTokenizer(), en, ne, **kwargs)


class TestTokenRatio:
    def test_f_is_the_ratio_of_totals(self):
        """Mean of the pair ratios is 1.5 here; the total ratio is 5/3."""
        result = _ratio(["aa", "a"], ["aaaa", "a"])
        assert result["en_tokens"] == 3
        assert result["ne_tokens"] == 5
        assert result["f"] == pytest.approx(5 / 3)
        assert result["f"] != pytest.approx(1.5)

    def test_rates_are_ratios_of_totals(self):
        result = _ratio(["a b", "cd"], ["ab", "c"])
        # "a b" is two words and three characters; the space is a character.
        assert result["en_words"] == 3
        assert result["ne_words"] == 2
        assert result["en_characters"] == 5
        assert result["ne_characters"] == 3
        assert result["en_tokens_per_word"] == pytest.approx(5 / 3)
        assert result["ne_tokens_per_word"] == pytest.approx(3 / 2)
        assert result["en_tokens_per_character"] == pytest.approx(1.0)
        assert result["ne_tokens_per_character"] == pytest.approx(1.0)

    def test_encode_does_not_add_special_tokens(self):
        tokenizer = _CharTokenizer()
        token_ratio(tokenizer, ["ab"], ["cd"], n_boot=4, seed=0)
        assert tokenizer.calls
        assert set(tokenizer.calls) == {False}

    def test_a_single_pair_has_a_degenerate_interval(self):
        result = _ratio(["ab"], ["abcd"], n_boot=8)
        assert result["f"] == pytest.approx(2.0)
        assert result["f_ci_low"] == pytest.approx(result["f"])
        assert result["f_ci_high"] == pytest.approx(result["f"])

    def test_the_interval_is_deterministic_in_the_seed(self):
        first = _ratio(["a", "aa", "aaa"], ["aaaa", "a", "aa"], n_boot=40, seed=0)
        second = _ratio(["a", "aa", "aaa"], ["aaaa", "a", "aa"], n_boot=40, seed=0)
        other = _ratio(["a", "aa", "aaa"], ["aaaa", "a", "aa"], n_boot=40, seed=1)
        assert first["f_ci_low"] == second["f_ci_low"]
        assert first["f_ci_high"] == second["f_ci_high"]
        assert (first["f_ci_low"], first["f_ci_high"]) != (
            other["f_ci_low"],
            other["f_ci_high"],
        )

    def test_the_interval_matches_a_paired_resample(self):
        """
        The interval is the 2.5 and 97.5 percentiles of f on pairs drawn
        together. Resampling the two languages independently is a different
        draw sequence and a different interval.
        """
        en_counts = [1, 2, 3]
        ne_counts = [4, 1, 2]
        n_boot = 40
        seed = 0
        result = _ratio(["a", "aa", "aaa"], ["aaaa", "a", "aa"], n_boot=n_boot, seed=seed)

        rng = random.Random(seed)
        samples = []
        n = len(en_counts)
        for _ in range(n_boot):
            en_total = ne_total = 0
            for _ in range(n):
                index = rng.randrange(n)
                en_total += en_counts[index]
                ne_total += ne_counts[index]
            samples.append(ne_total / en_total)
        samples.sort()

        def percentile(values, p):
            pos = p * (len(values) - 1)
            lo = int(pos)
            hi = min(lo + 1, len(values) - 1)
            weight = pos - lo
            return values[lo] * (1 - weight) + values[hi] * weight

        assert result["f_ci_low"] == pytest.approx(percentile(samples, 0.025))
        assert result["f_ci_high"] == pytest.approx(percentile(samples, 0.975))
        assert result["n_boot"] == n_boot
        assert result["seed"] == seed
        assert result["n_pairs"] == 3

    def test_unequal_pair_lists_are_rejected(self):
        with pytest.raises(ValueError):
            _ratio(["a"], ["b", "c"])


class TestUniqueParallelPassages:
    def test_questions_that_share_a_passage_count_once(self):
        en = [
            {"passage_id": "p", "context": "the river"},
            {"passage_id": "p", "context": "the river"},
            {"passage_id": "q", "context": "the hill"},
        ]
        ne = [
            {"passage_id": "q", "context": "पहाड"},
            {"passage_id": "p", "context": "नदी"},
            {"passage_id": "p", "context": "नदी"},
        ]
        en_texts, ne_texts = unique_parallel_passages(en, ne)
        # Sorted by passage id, not by row order: the files are not aligned.
        assert en_texts == ["the river", "the hill"]
        assert ne_texts == ["नदी", "पहाड"]

    def test_a_passage_with_two_texts_is_rejected(self):
        rows = [
            {"passage_id": "p", "context": "one"},
            {"passage_id": "p", "context": "two"},
        ]
        with pytest.raises(ValueError):
            unique_parallel_passages(rows, [{"passage_id": "p", "context": "एक"}])

    def test_an_unpaired_passage_is_rejected(self):
        en = [{"passage_id": "p", "context": "one"}]
        ne = [{"passage_id": "q", "context": "दुई"}]
        with pytest.raises(ValueError):
            unique_parallel_passages(en, ne)


class TestStudyModels:
    def test_the_debug_model_is_excluded(self):
        keys = [spec.key for spec in study_models()]
        assert "tiny-gpt2" not in keys
        assert keys == [key for key, spec in MODELS.items() if spec.generation != "debug"]
        assert "gemma-3-1b" in keys
        assert "gemma-3-4b" in keys

    def test_each_study_model_is_measured_at_its_pinned_revision(self, tmp_path: Path):
        seen = []

        def tokenize(spec):
            seen.append((spec.hf_id, spec.revision))
            return _CharTokenizer()

        en = [{"passage_id": "p", "context": "ab"}]
        ne = [{"passage_id": "p", "context": "abcd"}]
        rows = measure_all(study_models(), (en, ne), tokenize, n_boot=4, seed=0)
        assert seen == [(spec.hf_id, spec.revision) for spec in study_models()]
        assert {row["model"] for row in rows} == {spec.key for spec in study_models()}
        assert all(row["revision"] == MODELS[row["model"]].revision for row in rows)
        assert all(row["f"] == pytest.approx(2.0) for row in rows)

        provenance = {"belebele_revision": "abc", "n_boot": 4, "seed": 0}
        write_fertility(tmp_path, rows, provenance)
        csv_path = fertility_csv(tmp_path)
        json_path = fertility_json(tmp_path)
        assert csv_path.is_file()
        assert json_path.is_file()
        header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        for column in ("model", "revision", "f", "f_ci_low", "f_ci_high", "en_tokens", "ne_tokens"):
            assert column in header
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["provenance"]["belebele_revision"] == "abc"
        assert len(payload["rows"]) == len(rows)


class TestReadmeHasNoHandTypedRatio:
    def test_the_token_economics_question_points_at_the_table(self):
        readme = (
            Path(__file__).resolve().parents[1].joinpath("README.md").read_text(encoding="utf-8")
        )
        question = next(line for line in readme.splitlines() if "Token Economics" in line)
        assert "4.5" not in question
        assert "cl100k" not in question.lower()
        assert "results/fertility/fertility.csv" in question
