"""Shim so lm-eval ``!function utils.process_docs`` resolves under include_path.

The implementation lives in ``edge_slm_ace.eval.utils``; this module re-exports
it because TaskManager loads ``utils`` from the tasks directory.
"""

from edge_slm_ace.eval.utils import (  # noqa: F401
    permute_options,
    process_docs,
    shuffle_mcq_doc,
    stable_item_hash,
    stable_item_seed,
)

__all__ = [
    "permute_options",
    "process_docs",
    "shuffle_mcq_doc",
    "stable_item_hash",
    "stable_item_seed",
]
