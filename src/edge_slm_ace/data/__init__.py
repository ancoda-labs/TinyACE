"""Dataset loaders."""

from edge_slm_ace.data.belebele import (
    BELEBELE_LANGUAGES,
    belebele_path,
    load_belebele,
    parallel_split,
    study_split,
)

__all__ = [
    "BELEBELE_LANGUAGES",
    "belebele_path",
    "load_belebele",
    "parallel_split",
    "study_split",
]
