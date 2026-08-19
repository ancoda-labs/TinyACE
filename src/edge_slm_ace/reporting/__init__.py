"""Reading, aggregating and presenting experiment results.

Everything that turns a results directory into a table or a figure lives here.
Scripts in `scripts/` are thin CLIs over this package, so that the vocabulary
(arm names, model labels, canonical columns) is defined once rather than
re-derived by each consumer.
"""

from edge_slm_ace.reporting.layout import CELL_DEPTH, Cell, cell_dir, parse_cell
from edge_slm_ace.reporting.load import (
    load_predictions,
    load_run_metrics,
    normalize_columns,
    summarize_predictions,
)
from edge_slm_ace.reporting.schema import (
    ABLATION_REFERENCE,
    ARMS,
    Arm,
    arm_label,
    arm_order,
    get_arm,
    is_ablation,
    model_label,
    reference_for,
    task_label,
)

__all__ = [
    "ABLATION_REFERENCE",
    "ARMS",
    "CELL_DEPTH",
    "Arm",
    "Cell",
    "arm_label",
    "arm_order",
    "cell_dir",
    "get_arm",
    "is_ablation",
    "load_predictions",
    "load_run_metrics",
    "model_label",
    "normalize_columns",
    "parse_cell",
    "reference_for",
    "summarize_predictions",
    "task_label",
]
