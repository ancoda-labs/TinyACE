"""Dataset loaders."""

from edge_slm_ace.data.belebele import (
    BELEBELE_LANGUAGES,
    belebele_path,
    load_belebele,
    parallel_split,
    passage_id,
    study_split,
)
from edge_slm_ace.data.globalmmlu import (
    GLOBAL_MMLU_LANGUAGES,
    load_globalmmlu,
    verify_parallelism,
    verify_schema_and_size,
)

__all__ = [
    "BELEBELE_LANGUAGES",
    "GLOBAL_MMLU_LANGUAGES",
    "belebele_path",
    "load_belebele",
    "load_globalmmlu",
    "parallel_split",
    "passage_id",
    "study_split",
    "verify_parallelism",
    "verify_schema_and_size",
]
