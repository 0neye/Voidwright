"""Corpus rule: reject ships that exceed a maximum size threshold."""

from __future__ import annotations

from corpus.context import CorpusContext
from corpus.rules.base import CorpusRule, RuleResult

__all__ = ["MaxSizeRule"]


class MaxSizeRule(CorpusRule):
    """Reject ships whose part count or occupied-cell count exceeds a threshold.

    Either or both thresholds may be set. Metrics with no threshold are skipped.
    """

    name = "max_size"
    version = 1

    def __init__(
        self,
        max_parts: int | None = None,
        max_occupied_cells: int | None = None,
    ) -> None:
        self.max_parts = max_parts
        self.max_occupied_cells = max_occupied_cells

    def evaluate(self, context: CorpusContext) -> RuleResult:
        if self.max_parts is not None and context.part_count > self.max_parts:
            return RuleResult(
                passed=False,
                message=(
                    f"ship exceeds max_parts threshold "
                    f"({context.part_count} > {self.max_parts})"
                ),
            )
        if (
            self.max_occupied_cells is not None
            and context.occupied_cells > self.max_occupied_cells
        ):
            return RuleResult(
                passed=False,
                message=(
                    f"ship exceeds max_occupied_cells threshold "
                    f"({context.occupied_cells} > {self.max_occupied_cells})"
                ),
            )
        return RuleResult(passed=True)
