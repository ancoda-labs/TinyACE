"""Tests for the Belebele data layer.

The whole en/ne claim rests on the two language files being the *same* 900
questions. They are -- but not in the same row order, which makes every
position-based shortcut a silent methodology bug rather than an error.
"""

import json

import pytest

from edge_slm_ace.utils import ADAPTATION_SIZE, SCREENING_N
from edge_slm_ace.data.belebele import (
    BELEBELE_LANGUAGES,
    HARNESS_TASKS,
    belebele_path,
    harness_indices,
    item_id,
    load_belebele,
    parallel_split,
    study_split,
)

pytestmark = pytest.mark.skipif(
    not belebele_path("en").exists() or not belebele_path("ne").exists(),
    reason="Belebele not fetched; run scripts/fetch_belebele.py",
)


@pytest.fixture(scope="module")
def languages():
    return load_belebele("en"), load_belebele("ne")


class TestParallelism:
    def test_both_languages_have_the_same_items(self, languages):
        en, ne = languages
        assert [e["id"] for e in en] == [e["id"] for e in ne]

    def test_gold_agrees_across_languages(self, languages):
        """A translated question keeps its answer; if not, the pairing is wrong."""
        en, ne = languages
        assert all(a["gold_option_idx"] == b["gold_option_idx"] for a, b in zip(en, ne))

    def test_files_are_not_in_the_same_order(self):
        """The premise for every id-based join in this module."""
        raw = {}
        for lang in ("en", "ne"):
            with open(belebele_path(lang), encoding="utf-8") as f:
                raw[lang] = [item_id(json.loads(line)) for line in f if line.strip()]
        assert set(raw["en"]) == set(raw["ne"])
        assert raw["en"] != raw["ne"], "if these ever align, the trap is dormant, not gone"


class TestItemIds:
    def test_ids_are_unique(self, languages):
        en, _ = languages
        ids = [e["id"] for e in en]
        assert len(set(ids)) == len(ids) == 900

    def test_truncating_the_link_would_collide(self):
        """Why the id hashes the whole link: six passages end in '/Introduction'."""
        with open(belebele_path("en"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        naive = {(str(r["link"]).rsplit("/", 1)[-1], r["question_number"]) for r in rows}
        assert len(naive) < len(rows), "the naive key is expected to collide"
        assert len({item_id(r) for r in rows}) == len(rows)


class TestSplit:
    def test_split_is_identical_across_languages(self, languages):
        en, ne = languages
        a_en, e_en = parallel_split([e["id"] for e in en], 200, seed=42)
        a_ne, e_ne = parallel_split([e["id"] for e in ne], 200, seed=42)
        assert (a_en, e_en) == (a_ne, e_ne)

    def test_split_is_disjoint_and_complete(self, languages):
        en, _ = languages
        ids = [e["id"] for e in en]
        adapt, evaluate = parallel_split(ids, 200, seed=42)
        assert not set(adapt) & set(evaluate)
        assert set(adapt) | set(evaluate) == set(ids)
        assert len(adapt) == 200

    def test_split_is_independent_of_input_order(self, languages):
        """Order-dependence would make the split differ per language."""
        en, _ = languages
        ids = [e["id"] for e in en]
        assert parallel_split(ids, 200, 42) == parallel_split(list(reversed(ids)), 200, 42)

    def test_different_seeds_give_different_splits(self, languages):
        en, _ = languages
        ids = [e["id"] for e in en]
        assert parallel_split(ids, 200, 42) != parallel_split(ids, 200, 43)

    def test_rejects_a_degenerate_size(self, languages):
        en, _ = languages
        with pytest.raises(ValueError):
            parallel_split([e["id"] for e in en], 900, seed=42)


class TestHarnessIndices:
    """
    `simple_evaluate(samples=...)` selects documents by position. Reusing one
    language's index list for the other silently evaluates a different question
    set -- and because the split is 200/700, part of that set is the adaptation
    split the playbook was built on.
    """

    def test_indices_select_exactly_the_requested_items(self, languages):
        en, _ = languages
        _, evaluate = parallel_split([e["id"] for e in en], 200, seed=42)
        for lang in ("en", "ne"):
            with open(belebele_path(lang), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            selected = {item_id(rows[i]) for i in harness_indices(lang, evaluate)}
            assert selected == set(evaluate)

    def test_reusing_one_languages_indices_leaks_adaptation_items(self, languages):
        """The bug this function exists to prevent, asserted rather than assumed."""
        en, _ = languages
        adapt, evaluate = parallel_split([e["id"] for e in en], 200, seed=42)
        with open(belebele_path("ne"), encoding="utf-8") as f:
            ne_rows = [json.loads(line) for line in f if line.strip()]

        wrong = {item_id(ne_rows[i]) for i in harness_indices("en", evaluate)}
        assert wrong != set(evaluate)
        assert wrong & set(adapt), "expected adaptation items to leak into the eval set"

    def test_unknown_id_is_an_error_not_a_silent_drop(self):
        with pytest.raises(KeyError):
            harness_indices("en", ["bel-deadbeefff-q1"])

    def test_a_blank_line_does_not_shift_the_positions(self, tmp_path):
        """Positions must number records, the way the harness loads them.

        `enumerate(f)` counted blank lines the filter then dropped, so one blank
        line shifted every position after it and the run scored a different
        document while reporting a clean accuracy.
        """
        rows = [
            {
                "link": f"http://example.invalid/passage-{i}",
                "question_number": 1,
                "flores_passage": "p",
                "question": "q",
                "mc_answer1": "a",
                "mc_answer2": "b",
                "mc_answer3": "c",
                "mc_answer4": "d",
                "correct_answer_num": "1",
            }
            for i in range(3)
        ]
        path = tmp_path / "gappy.jsonl"
        path.write_text(
            "\n".join([json.dumps(rows[0]), "", json.dumps(rows[1]), json.dumps(rows[2])]) + "\n",
            encoding="utf-8",
        )
        ids = [item_id(r) for r in rows]
        assert harness_indices("en", ids, path=path) == [0, 1, 2]


class TestSchema:
    def test_examples_carry_what_the_runner_needs(self, languages):
        en, _ = languages
        example = en[0]
        assert {"id", "question", "context", "options", "gold_option_idx", "answer"} <= set(example)
        assert len(example["options"]) == 4
        assert example["answer"] == example["options"][example["gold_option_idx"]]

    def test_domain_separates_the_languages_by_default(self, languages):
        en, ne = languages
        assert en[0]["domain"] == "belebele_en"
        assert ne[0]["domain"] == "belebele_ne"

    def test_domain_can_be_shared_for_cross_lingual_transfer(self):
        shared = load_belebele("ne", domain="belebele")
        assert shared[0]["domain"] == "belebele"

    def test_harness_task_names_cover_both_languages(self):
        assert set(HARNESS_TASKS) == set(BELEBELE_LANGUAGES)


class TestStudySplit:
    """One split for the whole study, so screening cannot leak into evaluation.

    `screen_models` split at 400 and `run_arm` at 200. The split is one shuffle
    sliced at that index, so the 200 items between them were screened on *and*
    scored on: models were selected on items their own results are reported
    over.
    """

    def test_screening_and_evaluation_never_overlap(self):
        ids = [f"bel-{i:04d}" for i in range(900)]
        adapt, evaluate = study_split(ids, seed=42)

        screening = adapt[:SCREENING_N]
        assert len(screening) == SCREENING_N
        assert not set(screening) & set(evaluate)

    def test_screening_fits_inside_the_adaptation_split(self):
        """The constraint the two entrypoints have to satisfy jointly."""
        assert SCREENING_N <= ADAPTATION_SIZE

    def test_every_entrypoint_gets_the_same_split(self):
        ids = [f"bel-{i:04d}" for i in range(900)]
        assert study_split(ids, seed=42) == study_split(list(reversed(ids)), seed=42)

    def test_an_override_is_still_available_for_debugging(self):
        ids = [f"bel-{i:04d}" for i in range(900)]
        adapt, _ = study_split(ids, seed=42, adaptation_size=10)
        assert len(adapt) == 10
