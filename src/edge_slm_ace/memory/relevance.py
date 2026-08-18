"""Query-conditioned relevance for playbook retrieval.

Retrieval originally filtered on `entry.domain == domain` and ranked by a
score that does not depend on the question at all. Every task in this repo
maps to a single domain (all sciq_* tasks are "science"), so every question in
a run received the identical lesson list. That is not retrieval -- it is a
slowly drifting fixed prompt prefix, which is pure distraction for a small
model and is a plausible cause of the negative results reported for
TinyLlama.

This module scores lessons against the question so that retrieval selects.
The embedding model is the same MiniLM already used for OMA and semantic
similarity, so it adds no new dependency and no second model load.
"""

from typing import List, Optional, Sequence

import numpy as np


class LessonRelevance:
    """
    Cosine relevance between a question and candidate lesson texts.

    Singleton, sharing the sentence-transformers model with SemanticEvaluator.
    Lesson embeddings are cached by text: the playbook is small and mostly
    stable between steps, so re-encoding it every question would dominate
    retrieval cost.
    """

    _instance: Optional["LessonRelevance"] = None
    _model = None

    def __init__(self):
        self._cache = {}
        self._load_model()

    @classmethod
    def get_instance(cls) -> "LessonRelevance":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton and its cache (used by tests)."""
        cls._instance = None
        cls._model = None

    def _load_model(self) -> None:
        """Load the shared MiniLM encoder, if one is available."""
        if LessonRelevance._model is not None:
            return
        try:
            from edge_slm_ace.eval.metrics import SemanticEvaluator

            SemanticEvaluator.get_instance()
            if SemanticEvaluator._model is not None:
                LessonRelevance._model = SemanticEvaluator._model
                return
        except Exception:
            pass

        try:
            from sentence_transformers import SentenceTransformer

            LessonRelevance._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            # No encoder available. `available` reports False and callers fall
            # back to retention-only ranking rather than failing the run.
            LessonRelevance._model = None

        if LessonRelevance._model is None:
            # Say so once. Silently degrading to retention-only ranking would
            # turn every question's lesson list back into a fixed prefix while
            # the config still claimed relevance_weight > 0.
            print(
                "[relevance] sentence-transformers unavailable; playbook "
                "retrieval falls back to retention-only ranking and is NOT "
                "query-conditioned. Install the 'metrics' extra to enable it."
            )

    @property
    def available(self) -> bool:
        """True when an encoder is loaded and relevance can be computed."""
        return LessonRelevance._model is not None

    def _encode(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        """Encode texts, reusing cached lesson embeddings where possible."""
        if not self.available or not texts:
            return None

        missing = [t for t in texts if t not in self._cache]
        if missing:
            try:
                vectors = LessonRelevance._model.encode(
                    missing, convert_to_numpy=True, normalize_embeddings=True
                )
            except Exception:
                return None
            for text, vector in zip(missing, vectors):
                self._cache[text] = vector

        return np.vstack([self._cache[t] for t in texts])

    def score(self, query: str, lesson_texts: Sequence[str]) -> Optional[List[float]]:
        """
        Cosine relevance of each lesson to the question.

        Args:
            query: The question being answered.
            lesson_texts: Candidate lesson texts.

        Returns:
            One score in [0, 1] per lesson, or None when no encoder is
            available or the query is empty -- in which case the caller
            should rank on retention score alone.
        """
        if not self.available or not query or not query.strip() or not lesson_texts:
            return None

        query_vec = self._encode([query])
        lesson_vecs = self._encode(list(lesson_texts))
        if query_vec is None or lesson_vecs is None:
            return None

        sims = lesson_vecs @ query_vec[0]
        return [float(np.clip(s, 0.0, 1.0)) for s in sims]


def blend(
    retention_scores: Sequence[float],
    relevance_scores: Optional[Sequence[float]],
    relevance_weight: float,
) -> List[float]:
    """
    Combine retention and relevance into one ranking key.

    Retention scores are unbounded and typically span roughly [-0.9, 1.3],
    while relevance is a cosine in [0, 1]. Adding them directly would let
    whichever happens to have the larger spread dominate, so retention is
    min-max normalised across the candidate set first.

    Args:
        retention_scores: Retention score per candidate.
        relevance_scores: Relevance per candidate, or None to ignore relevance.
        relevance_weight: Weight on relevance in [0, 1]. 0 reproduces
            retention-only ranking.

    Returns:
        One combined key per candidate; higher ranks first.
    """
    retention = list(retention_scores)
    if not relevance_scores or relevance_weight <= 0.0 or not retention:
        return retention

    low, high = min(retention), max(retention)
    span = high - low
    # All candidates equally retained: relevance decides outright.
    normalized = [0.5] * len(retention) if span <= 0 else [(s - low) / span for s in retention]

    w = max(0.0, min(1.0, relevance_weight))
    return [w * rel + (1.0 - w) * ret for ret, rel in zip(normalized, relevance_scores)]
