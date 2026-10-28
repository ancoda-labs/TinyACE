"""The results directory layout, defined once.

A runner writes one cell to `{root}/{model}/{language}/{arm}/`. The reporting
layer used to re-derive the arm by counting path segments from the right,
against an older four-segment layout that also carried a device. On the current
layout that arithmetic returns the *language*: every arm in a cell collapsed
onto one key, `summary_accuracy.csv` pooled all of them into a single row, and
`compare_arms` found no reference for anything and printed zero comparisons
while still exiting 0.

The writer and every reader now go through this module, so the layout can only
change in one place.
"""

from pathlib import Path
from typing import NamedTuple, Optional, Union

# model / language / arm
CELL_DEPTH = 3


class Cell(NamedTuple):
    """The coordinates of one result cell."""

    model: str
    language: str
    arm: str

    @property
    def group(self) -> str:
        """The `{model}/{language}` prefix a comparison must not cross."""
        return f"{self.model}/{self.language}"


def cell_dir(root: Union[str, Path], model: str, language: str, arm: str) -> Path:
    """The directory one cell's artefacts are written to."""
    return Path(root) / model / language / arm


def screening_path(root: Union[str, Path], benchmark: str, language: str) -> Path:
    """
    Where one task and language's screening result is written.

    The task and the language are part of the filename. A shared
    ``screening.json`` let a later English run replace the Nepali result, and
    the grid then admitted models on the wrong language.
    """
    return Path(root) / "screening" / f"screening_{benchmark}_{language}.json"


def legacy_screening_path(root: Union[str, Path]) -> Path:
    """
    The single screening file written before results were split by task and language.

    It has no language in its name, so a reader cannot tell a Nepali result
    from an English one that overwrote it. Callers detect it to warn, and do
    not admit models from it.
    """
    return Path(root) / "screening" / "screening.json"


def parse_cell(path: Union[str, Path]) -> Optional[Cell]:
    """
    Recover a cell's coordinates from its path relative to the results root.

    Args:
        path: The cell directory, relative to the results root. Longer paths
            are read from the right, so a nested root still parses.

    Returns:
        The `Cell`, or None when the path is too shallow to be one -- callers
        must handle that rather than guess, because guessing is what produced
        the pooled-arm defect this module exists to prevent.
    """
    parts = Path(path).parts
    if len(parts) < CELL_DEPTH:
        return None
    return Cell(*parts[-CELL_DEPTH:])
