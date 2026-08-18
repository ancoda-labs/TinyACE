"""Tests for query-conditioned playbook retrieval."""

import numpy as np
import pytest

from edge_slm_ace.memory.playbook import Playbook, ScoringParams
from edge_slm_ace.memory.relevance import LessonRelevance, blend


class _FakeEncoder:
    """Deterministic stand-in for MiniLM: one dimension per keyword."""

    KEYWORDS = ["wind", "force", "acid"]

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        vectors = []
        for text in texts:
            vec = np.array(
                [1.0 if kw in text.lower() else 0.0 for kw in self.KEYWORDS],
                dtype=float,
            )
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm else vec)
        return np.vstack(vectors)


@pytest.fixture
def fake_relevance():
    LessonRelevance.reset()
    instance = LessonRelevance.get_instance()
    LessonRelevance._model = _FakeEncoder()
    instance._cache = {}
    yield instance
    LessonRelevance.reset()


class TestBlend:
    def test_no_relevance_returns_retention_unchanged(self):
        assert blend([0.9, 0.1], None, 0.5) == [0.9, 0.1]

    def test_zero_weight_ignores_relevance(self):
        assert blend([0.9, 0.1], [0.0, 1.0], 0.0) == [0.9, 0.1]

    def test_full_weight_ignores_retention(self):
        assert blend([0.9, 0.1], [0.0, 1.0], 1.0) == [0.0, 1.0]

    def test_retention_is_normalised_before_blending(self):
        """Raw retention spans a different range than cosine; scale first."""
        keys = blend([100.0, 0.0], [0.0, 1.0], 0.5)
        assert keys == [0.5, 0.5], "Unnormalised retention would dominate"

    def test_equal_retention_lets_relevance_decide(self):
        keys = blend([0.5, 0.5], [0.2, 0.8], 0.5)
        assert keys[1] > keys[0]


class TestQueryConditionedRetrieval:
    LESSONS = [
        "For wind direction questions, the Coriolis effect deflects flows.",
        "For force problems, apply F equals m times a with mass in kilograms.",
        "For pH questions, an acid measures below 7 on the scale.",
    ]

    def _playbook(self, relevance_weight):
        playbook = Playbook(
            token_budget=500,
            store_token_capacity=5000,
            scoring_params=ScoringParams(relevance_weight=relevance_weight),
        )
        for i, lesson in enumerate(self.LESSONS, start=1):
            playbook.add_entry("science", lesson, step=i)
        return playbook

    def test_retrieval_follows_the_question(self, fake_relevance):
        playbook = self._playbook(relevance_weight=0.9)
        top = playbook.get_top_k(
            "science", k=1, current_step=10,
            query="What makes global wind curve?",
        )
        assert "Coriolis" in top[0].text

    def test_a_different_question_retrieves_a_different_lesson(self, fake_relevance):
        playbook = self._playbook(relevance_weight=0.9)
        top = playbook.get_top_k(
            "science", k=1, current_step=10,
            query="How do I compute force from mass?",
        )
        assert "force" in top[0].text.lower()

    def test_without_relevance_every_question_gets_the_same_list(self, fake_relevance):
        """The original behaviour, kept reachable for ablation."""
        playbook = self._playbook(relevance_weight=0.0)
        first = playbook.get_top_k("science", k=2, current_step=10, query="wind?")
        second = playbook.get_top_k("science", k=2, current_step=10, query="force?")
        assert [e.id for e in first] == [e.id for e in second]

    def test_budgeted_retrieval_is_also_query_conditioned(self, fake_relevance):
        playbook = self._playbook(relevance_weight=0.9)
        top = playbook.get_top_entries_for_budget(
            "science", token_budget=15, current_step=10,
            query="What makes global wind curve?",
        )
        assert top and "Coriolis" in top[0].text

    def test_missing_encoder_falls_back_instead_of_failing(self):
        LessonRelevance.reset()
        LessonRelevance._instance = LessonRelevance.__new__(LessonRelevance)
        LessonRelevance._instance._cache = {}
        LessonRelevance._model = None

        playbook = self._playbook(relevance_weight=0.9)
        top = playbook.get_top_k("science", k=2, current_step=10, query="wind?")

        assert len(top) == 2, "Retrieval must still work without an encoder"
        LessonRelevance.reset()
