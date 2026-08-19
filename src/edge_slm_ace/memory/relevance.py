"""Query-conditioned relevance for playbook retrieval.

Retrieval that filters on domain and ranks by a question-independent score is
not retrieval -- it is a slowly drifting fixed prefix, identical for every
question in a run. This module scores lessons against the question so that
retrieval selects.

**The encoder must be multilingual.** The previous default,
`all-MiniLM-L6-v2`, is English-only: it maps Devanagari to near-noise, so a
Nepali question would have been matched against Nepali lessons by an encoder
that cannot read either. That is worse than no retrieval, because it looks like
retrieval. The default is now a multilingual model, and `available` reports
False rather than degrading silently when none can be loaded.
"""

import os
from typing import List, Optional, Sequence

import numpy as np

# Multilingual sentence encoders, best first. Both cover Nepali (Devanagari).
# BGE-m3 is stronger and much larger; the MiniLM is the default because
# retrieval runs once per question and the playbook is small.
DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENCODER_ENV = "TINYACE_RELEVANCE_ENCODER"


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
    _model_name: Optional[str] = None

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
        cls._model_name = None

    def _load_model(self) -> None:
        """Load the multilingual encoder, if one is available."""
        if LessonRelevance._model is not None:
            return

        name = os.environ.get(ENCODER_ENV, DEFAULT_ENCODER)
        try:
            from sentence_transformers import SentenceTransformer

            LessonRelevance._model = SentenceTransformer(name)
            LessonRelevance._model_name = name
            return
        except Exception as e:
            # No encoder available. `available` reports False and callers fall
            # back to retention-only ranking rather than failing the run.
            LessonRelevance._model = None
            reason = f"{type(e).__name__}: {e}"

        # Say so once. Silently degrading would turn every question's lesson
        # list back into a fixed prefix while the config still claimed
        # relevance_weight > 0.
        print(
            f"[relevance] could not load '{name}' ({reason}); playbook "
            f"retrieval falls back to retention-only ranking and is NOT "
            f"query-conditioned. Install the 'metrics' extra to enable it."
        )

    @property
    def available(self) -> bool:
        """True when an encoder is loaded and relevance can be computed."""
        return LessonRelevance._model is not None

    @property
    def encoder_name(self) -> Optional[str]:
        """Which encoder is in use, for the run metadata."""
        return LessonRelevance._model_name

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
