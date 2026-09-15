"""Tests for deterministic MCQ option shuffling (eval.utils)."""

import hashlib

import pytest

from edge_slm_ace.eval.utils import (
    permute_options,
    process_docs,
    shuffle_mcq_doc,
    stable_item_hash,
    stable_item_seed,
)


class TestStableSeed:
    def test_md5_of_passage_and_question(self):
        doc = {"flores_passage": "P", "question": "Q?"}
        assert stable_item_hash(doc) == hashlib.md5(b"P\nQ?").hexdigest()

    def test_same_text_same_seed(self):
        a = {"flores_passage": "x", "question": "y", "mc_answer1": "a"}
        b = {"flores_passage": "x", "question": "y", "mc_answer1": "अ"}
        assert stable_item_seed(a) == stable_item_seed(b)

    def test_global_mmlu_uses_question(self):
        a = {"sample_id": "s/0", "question": "Q", "option_a": "1"}
        b = {"sample_id": "s/0", "question": "Q", "option_a": "१"}
        assert stable_item_hash(a) == stable_item_hash(b)


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
            "flores_passage": "http://example/p",
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
