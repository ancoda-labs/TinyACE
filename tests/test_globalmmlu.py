"""Tests for Global-MMLU loader (schema fixtures + optional Hub check)."""

import json

import pytest

from edge_slm_ace.data.globalmmlu import (
    EXPECTED_SIZES,
    HF_DATASET,
    HARNESS_TASKS,
    item_id,
    load_globalmmlu,
    verify_schema_and_size,
)


def _row(sample_id: str, answer: str = "B", **kwargs):
    base = {
        "sample_id": sample_id,
        "subject": "abstract_algebra",
        "subject_category": "STEM",
        "question": "Q?",
        "option_a": "a",
        "option_b": "b",
        "option_c": "c",
        "option_d": "d",
        "answer": answer,
    }
    base.update(kwargs)
    return base


class TestItemId:
    def test_stable_across_language_payloads(self):
        en = _row("abstract_algebra/test/0", option_a="zero")
        ne = _row("abstract_algebra/test/0", option_a="शून्य")
        assert item_id(en) == item_id(ne)


class TestVerifySchemaFixture:
    def test_expected_sizes_are_the_hub_card_numbers(self):
        assert EXPECTED_SIZES == {"test": 14042, "dev": 285}
        assert HF_DATASET == "CohereForAI/Global-MMLU"
        assert set(HARNESS_TASKS) == {"en", "ne"}

    def test_size_mismatch_raises(self):
        rows = [_row("s/0")]
        with pytest.raises(AssertionError, match="expected 14042"):
            verify_schema_and_size("en", "test", records=rows)

    def test_missing_field_raises(self):
        rows = [_row(f"s/{i}") for i in range(EXPECTED_SIZES["dev"])]
        rows[0].pop("question")
        with pytest.raises(AssertionError, match="missing fields"):
            verify_schema_and_size("en", "dev", records=rows)

    def test_dev_size_ok(self):
        rows = [_row(f"s/{i}", answer="ABCD"[i % 4]) for i in range(EXPECTED_SIZES["dev"])]
        summary = verify_schema_and_size("ne", "dev", records=rows)
        assert summary["ok"] and summary["n"] == EXPECTED_SIZES["dev"]


class TestLoadFromJsonl:
    def test_normalises_and_shuffles(self, tmp_path):
        path = tmp_path / "en_test.jsonl"
        rows = [_row(f"subj/test/{i}", answer="C") for i in range(5)]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        examples = load_globalmmlu("en", path=path)
        assert len(examples) == 5
        assert examples == sorted(examples, key=lambda e: e["id"])
        for e in examples:
            assert e["answer"] == e["options"][e["gold_option_idx"]]
            assert len(e["options"]) == 4
