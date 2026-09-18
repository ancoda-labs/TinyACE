"""Tests for deterministic MCQ option shuffling (eval.utils).

The headline property is `TestLanguageParity`: the same item must get the same
option order in every language. The seed used to be `MD5(passage + question)`,
which made that false for 857 of Belebele's 900 items while every downstream
number still computed.
"""

import pytest

from edge_slm_ace.eval.utils import (
    invariant_item_key,
    permute_options,
    process_docs,
    shuffle_mcq_doc,
    stable_item_hash,
    stable_item_seed,
)


class TestInvariantKey:
    def test_global_mmlu_uses_sample_id(self):
        assert invariant_item_key({"sample_id": "subj/test/3"}) == "gmmlu:subj/test/3"

    def test_belebele_uses_link_and_question_number(self):
        key = invariant_item_key({"link": "http://example/p", "question_number": 2})
        assert key == "belebele:http://example/p#2"

    def test_a_normalised_example_uses_its_id(self):
        assert invariant_item_key({"id": "bel-abc-q1"}) == "id:bel-abc-q1"

    def test_sample_id_wins_over_id(self):
        """Both are invariant; preferring the dataset-native key is stable."""
        assert invariant_item_key({"sample_id": "s/0", "id": "x"}) == "gmmlu:s/0"

    @pytest.mark.parametrize(
        "doc",
        [
            {"flores_passage": "P", "question": "Q?"},
            {"question": "Q?"},
            {"sample_id": "  ", "question": "Q?"},
            {"link": "", "question_number": 1},
            {},
        ],
    )
    def test_a_document_with_no_invariant_key_raises(self, doc):
        """
        It must not fall back to text. A translated item hashes differently in
        every language, so a fallback silently unpairs the study -- which is
        the bug this function was introduced to end.
        """
        with pytest.raises(ValueError, match="language-invariant key"):
            invariant_item_key(doc)


class TestLanguageParity:
    """G1. The same item, in any language, gets the same option order."""

    def test_belebele_translation_keeps_its_order(self):
        en = {
            "link": "http://example/p",
            "question_number": 1,
            "flores_passage": "An English passage.",
            "question": "What is it?",
            "mc_answer1": "one",
            "mc_answer2": "two",
            "mc_answer3": "three",
            "mc_answer4": "four",
            "correct_answer_num": "2",
        }
        ne = {
            **en,
            "flores_passage": "नेपाली अनुच्छेद।",
            "question": "यो के हो?",
            "mc_answer1": "एक",
            "mc_answer2": "दुई",
            "mc_answer3": "तीन",
            "mc_answer4": "चार",
        }
        assert stable_item_seed(en) == stable_item_seed(ne)
        assert shuffle_mcq_doc(en)["gold_option_idx"] == shuffle_mcq_doc(ne)["gold_option_idx"]

    def test_global_mmlu_translation_keeps_its_order(self):
        en = {
            "sample_id": "abstract_algebra/test/0",
            "question": "Find the degree.",
            "option_a": "0",
            "option_b": "4",
            "option_c": "2",
            "option_d": "6",
            "answer": "B",
        }
        ne = {
            **en,
            "question": "डिग्री फेला पार्नुहोस्।",
            "option_a": "०",
            "option_b": "४",
            "option_c": "२",
            "option_d": "६",
        }
        assert stable_item_seed(en) == stable_item_seed(ne)
        assert shuffle_mcq_doc(en)["gold_option_idx"] == shuffle_mcq_doc(ne)["gold_option_idx"]

    def test_differing_text_alone_does_not_change_the_seed(self):
        """The regression guard, stated directly."""
        base = {"sample_id": "s/0", "question": "Q", "option_a": "1"}
        translated = {"sample_id": "s/0", "question": "प्रश्न", "option_a": "१"}
        assert stable_item_hash(base) == stable_item_hash(translated)

    def test_different_items_still_differ(self):
        a = {"sample_id": "s/0"}
        b = {"sample_id": "s/1"}
        assert stable_item_seed(a) != stable_item_seed(b)


class TestSeedStability:
    def test_the_seed_is_an_int_so_it_survives_a_new_process(self):
        """
        `random.Random(str)` seeds from the string's hash, which
        PYTHONHASHSEED randomises per process -- adaptation and evaluation
        would then permute the same item differently.
        """
        assert isinstance(stable_item_seed({"sample_id": "s/0"}), int)

    def test_pinned_values(self):
        """
        Produced by this implementation. If these change, the option order of
        every item changed with them.
        """
        assert stable_item_hash({"sample_id": "abstract_algebra/test/0"}) == "6bcf7498bca9b980"
        assert stable_item_hash({"link": "http://example/p", "question_number": 1}) == (
            "cf8d7f6961be0fec"
        )


class TestPermuteOptions:
    def test_gold_tracks_the_correct_string(self):
        options = ["w", "x", "y", "z"]
        shuffled, new_gold = permute_options(options, gold_idx=2, seed=123)
        assert shuffled[new_gold] == "y"
        assert sorted(shuffled) == sorted(options)

    def test_same_seed_is_deterministic(self):
        opts = ["a", "b", "c", "d"]
        assert permute_options(opts, 0, 99) == permute_options(opts, 0, 99)


class TestShuffleMcqDoc:
    def test_belebele_shape(self):
        doc = {
            # Every real Belebele row carries these; they are the invariant key.
            "link": "http://example/p",
            "question_number": 1,
            "flores_passage": "A passage.",
            "question": "1",
            "mc_answer1": "A1",
            "mc_answer2": "A2",
            "mc_answer3": "A3",
            "mc_answer4": "A4",
            "correct_answer_num": "3",
        }
        out = shuffle_mcq_doc(doc)
        gold = out["gold_option_idx"]
        assert out[f"mc_answer{gold + 1}"] == "A3"
        assert {out[f"mc_answer{i}"] for i in range(1, 5)} == {"A1", "A2", "A3", "A4"}

    def test_global_mmlu_shape(self):
        doc = {
            "sample_id": "subj/test/1",
            "question": "Which?",
            "option_a": "A1",
            "option_b": "A2",
            "option_c": "A3",
            "option_d": "A4",
            "answer": "B",
        }
        out = shuffle_mcq_doc(doc)
        assert out["options"][out["gold_option_idx"]] == "A2"

    def test_process_docs_maps_a_hf_dataset(self):
        datasets = pytest.importorskip("datasets")
        ds = datasets.Dataset.from_list(
            [
                {
                    "sample_id": "s/0",
                    "question": "q",
                    "option_a": "a",
                    "option_b": "b",
                    "option_c": "c",
                    "option_d": "d",
                    "answer": "A",
                }
            ]
        )
        out = process_docs(ds)
        assert out[0]["answer"] in "ABCD"
        assert "gold_option_idx" in out[0]
        assert {out[0][f"option_{x}"] for x in "abcd"} == {"a", "b", "c", "d"}


class TestParityOnTheRealCorpus:
    """
    G1 on the actual 900 Belebele items rather than on two hand-written dicts.

    This is the measurement that exposed the defect: under the old
    text-derived seed, 857 of these 900 items (95.2%) were scored with their
    options in a different order in English than in Nepali. Synthetic fixtures
    did not catch it because they used the same text for both languages.
    """

    @staticmethod
    def _rows(path):
        import json

        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @pytest.fixture(scope="class")
    def paired(self):
        from edge_slm_ace.data.belebele import belebele_path

        for language in ("en", "ne"):
            if not belebele_path(language).exists():
                pytest.skip("Belebele not fetched; run scripts/fetch_belebele.py")

        english = {(r["link"], r["question_number"]): r for r in self._rows(belebele_path("en"))}
        nepali = {(r["link"], r["question_number"]): r for r in self._rows(belebele_path("ne"))}
        shared = sorted(set(english) & set(nepali))
        assert len(shared) == 900
        return [(english[k], nepali[k]) for k in shared]

    def test_every_item_gets_the_same_seed_in_both_languages(self, paired):
        differing = [
            en["link"] for en, ne in paired if stable_item_seed(en) != stable_item_seed(ne)
        ]
        assert not differing, f"{len(differing)} of {len(paired)} items disagree"

    def test_every_item_lands_gold_in_the_same_slot(self, paired):
        differing = [
            en["link"]
            for en, ne in paired
            if shuffle_mcq_doc(en)["gold_option_idx"] != shuffle_mcq_doc(ne)["gold_option_idx"]
        ]
        assert not differing, f"{len(differing)} of {len(paired)} items disagree"

    def test_the_shuffle_still_moves_gold_around(self, paired):
        """
        Parity is trivially satisfiable by not shuffling at all. Gold must
        still be spread across the four slots, or the debiasing is gone.
        """
        slots = {0: 0, 1: 0, 2: 0, 3: 0}
        for en, _ in paired:
            slots[shuffle_mcq_doc(en)["gold_option_idx"]] += 1
        assert min(slots.values()) > 150, slots

    def test_gold_text_survives_the_permutation(self, paired):
        for en, _ in paired:
            out = shuffle_mcq_doc(en)
            original = en[f"mc_answer{int(en['correct_answer_num'])}"]
            assert out[f"mc_answer{out['gold_option_idx'] + 1}"] == original
