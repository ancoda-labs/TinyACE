"""Day-1 data-layer tests: Global-MMLU, Belebele passage split, option shuffle."""

import json

import pytest

from edge_slm_ace.data.belebele import (
    assert_zero_passage_overlap,
    belebele_path,
    load_belebele,
    study_split,
)
from edge_slm_ace.data.globalmmlu import (
    EXPECTED_SIZES,
    HF_DATASET,
    HARNESS_TASKS as GMMLU_TASKS,
    item_id,
    load_globalmmlu,
    verify_schema_and_size,
)
from edge_slm_ace.eval.tasks import BELEBELE_HARNESS_TASKS, GLOBAL_MMLU_HARNESS_TASKS
from edge_slm_ace.eval.utils import (
    process_docs,
    shuffle_mcq_doc,
    stable_item_hash,
    stable_item_seed,
)
from edge_slm_ace.utils import SCREENING_FLOOR, screening_verdict
from edge_slm_ace.eval.stats import wilson_interval

needs_belebele = pytest.mark.skipif(
    not belebele_path("en").exists() or not belebele_path("ne").exists(),
    reason="Belebele not fetched; run scripts/fetch_belebele.py",
)


def _gmmlu_row(sample_id: str, answer: str = "B", **kwargs):
    row = {
        "sample_id": sample_id,
        "subject": "abstract_algebra",
        "subject_category": "STEM",
        "question": "What is Q?",
        "option_a": "a",
        "option_b": "b",
        "option_c": "c",
        "option_d": "d",
        "answer": answer,
    }
    row.update(kwargs)
    return row


class TestGlobalMMLULoader:
    def test_hub_id_and_expected_counts(self):
        assert HF_DATASET == "CohereForAI/Global-MMLU"
        assert EXPECTED_SIZES == {"test": 14042, "dev": 285}
        assert GMMLU_TASKS == {"en": "global_mmlu_en", "ne": "global_mmlu_ne"}
        assert GLOBAL_MMLU_HARNESS_TASKS == GMMLU_TASKS

    def test_verify_schema_records_item_counts(self):
        rows = [
            _gmmlu_row(f"subj/test/{i}", answer="ABCD"[i % 4]) for i in range(EXPECTED_SIZES["dev"])
        ]
        summary = verify_schema_and_size("en", "dev", records=rows)
        assert summary["ok"] is True
        assert summary["n"] == EXPECTED_SIZES["dev"]
        assert summary["expected_n"] == EXPECTED_SIZES["dev"]
        assert "sample_id" in summary["fields"]

    def test_verify_rejects_wrong_cardinality(self):
        with pytest.raises(AssertionError, match="expected 14042"):
            verify_schema_and_size("ne", "test", records=[_gmmlu_row("s/0")])

    def test_load_from_jsonl_normalises(self, tmp_path):
        path = tmp_path / "ne_test.jsonl"
        rows = [_gmmlu_row(f"subj/test/{i}", answer="C") for i in range(4)]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        examples = load_globalmmlu("ne", path=path)
        assert len(examples) == 4
        assert all("gold_option_idx" in e for e in examples)
        assert [e["id"] for e in examples] == sorted(e["id"] for e in examples)
        by_sample = {e["sample_id"]: e for e in examples}
        hit = by_sample["subj/test/0"]
        assert hit["answer"] == hit["options"][hit["gold_option_idx"]]
        assert hit["id"] == item_id(rows[0])


@needs_belebele
class TestBelebelePassageSplit:
    def test_zero_passage_intersection(self):
        examples = load_belebele("ne")
        adapt_ids, eval_ids = study_split(examples, seed=42)

        adapt_passages = {e["passage_id"] for e in examples if e["id"] in set(adapt_ids)}
        eval_passages = {e["passage_id"] for e in examples if e["id"] in set(eval_ids)}

        for passage_id in adapt_passages:
            assert passage_id not in eval_passages

        assert_zero_passage_overlap(adapt_ids, eval_ids, examples)
        assert adapt_passages.isdisjoint(eval_passages)

    def test_split_is_by_link_not_question_id(self):
        """Sibling questions that share a FLORES link stay on the same side."""
        examples = load_belebele("en")
        adapt_ids, eval_ids = study_split(examples, seed=42)
        side = {i: "adapt" for i in adapt_ids}
        side.update({i: "eval" for i in eval_ids})

        by_passage = {}
        for e in examples:
            by_passage.setdefault(e["passage_id"], []).append(e["id"])

        for ids in by_passage.values():
            sides = {side[i] for i in ids}
            assert len(sides) == 1, "questions from one link must not straddle the split"

    def test_harness_task_name_is_belebele_npi(self):
        assert BELEBELE_HARNESS_TASKS["ne"] == "belebele_npi"
        assert BELEBELE_HARNESS_TASKS["en"] == "belebele_eng"


class TestDeterministicOptionShuffling:
    def test_saves_gold_option_idx(self):
        doc = {
            "link": "http://example/once",
            "question_number": 1,
            "flores_passage": "Once upon a time.",
            "question": "Who?",
            "mc_answer1": "A1",
            "mc_answer2": "A2",
            "mc_answer3": "A3",
            "mc_answer4": "A4",
            "correct_answer_num": "2",
        }
        out = shuffle_mcq_doc(doc)
        assert "gold_option_idx" in out
        assert 0 <= out["gold_option_idx"] < 4
        assert out[f"mc_answer{out['gold_option_idx'] + 1}"] == "A2"

    def test_seeded_on_the_invariant_key_not_the_text(self):
        """
        Was `test_seeded_on_md5_of_passage_and_question`, which pinned the
        defect: seeding on passage + question text gave a translated item a
        different option order, and 857 of Belebele's 900 items disagreed
        between English and Nepali while every downstream number still
        computed. The seed now comes from (link, question_number).
        """
        english = {
            "link": "http://example/p",
            "question_number": 1,
            "flores_passage": "P",
            "question": "Q",
            "mc_answer1": "a",
            "mc_answer2": "b",
            "mc_answer3": "c",
            "mc_answer4": "d",
            "correct_answer_num": "1",
        }
        nepali = {
            **english,
            "flores_passage": "अनुच्छेद",
            "question": "प्रश्न",
            "mc_answer1": "क",
            "mc_answer2": "ख",
            "mc_answer3": "ग",
            "mc_answer4": "घ",
        }
        assert stable_item_hash(english) == stable_item_hash(nepali)
        assert isinstance(stable_item_seed(english), int)

    def test_idempotent_on_raw_docs(self):
        """Same raw input → identical shuffle every time (deterministic)."""
        doc = {
            "link": "http://example/river",
            "question_number": 2,
            "flores_passage": "The river flooded the plains.",
            "question": "What flooded?",
            "mc_answer1": "fire",
            "mc_answer2": "river",
            "mc_answer3": "wind",
            "mc_answer4": "snow",
            "correct_answer_num": "2",
        }
        a = shuffle_mcq_doc(doc)
        b = shuffle_mcq_doc(doc)
        assert a == b
        assert a["gold_option_idx"] == b["gold_option_idx"]
        assert [a[f"mc_answer{i}"] for i in range(1, 5)] == [
            b[f"mc_answer{i}"] for i in range(1, 5)
        ]

    def test_global_mmlu_shape_sets_gold_option_idx(self):
        doc = _gmmlu_row("subj/test/9", answer="D", question="Pick D")
        out = shuffle_mcq_doc(doc)
        assert out["options"][out["gold_option_idx"]] == "d"
        assert out["answer"] == "ABCD"[out["gold_option_idx"]]

    def test_process_docs_maps_dataset(self):
        datasets = pytest.importorskip("datasets")
        ds = datasets.Dataset.from_list(
            [
                {
                    "link": "http://example/ctx",
                    "question_number": 1,
                    "flores_passage": "ctx",
                    "question": "q",
                    "mc_answer1": "a",
                    "mc_answer2": "b",
                    "mc_answer3": "c",
                    "mc_answer4": "d",
                    "correct_answer_num": "1",
                }
            ]
        )
        out = process_docs(ds)
        assert "gold_option_idx" in out[0]
        assert out[0]["gold_option_idx"] == int(out[0]["correct_answer_num"]) - 1


class TestWilsonScreeningRule:
    def test_wilson_95_and_thirty_percent_floor(self):
        low, high = wilson_interval(140, 400, confidence=0.95)
        assert 0.0 <= low <= high <= 1.0
        assert screening_verdict(0.31) is True
        assert screening_verdict(0.30) is False
        assert SCREENING_FLOOR == 0.30
