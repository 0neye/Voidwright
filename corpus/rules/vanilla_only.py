"""Corpus rule: reject ships that contain any non-vanilla (modded) parts."""

from __future__ import annotations

from common.geometry import is_vanilla_part_id
from corpus.context import CorpusContext
from corpus.rules.base import CorpusRule, RuleResult

__all__ = ["VanillaOnlyRule"]


class VanillaOnlyRule(CorpusRule):
    """Reject ships that contain one or more non-vanilla part IDs.

    A part is considered vanilla when its ``part_id`` starts with the
    ``cosmoteer.`` namespace prefix (as defined by
    :func:`common.geometry.is_vanilla_part_id`).
    """

    name = "vanilla_only"
    version = 1

    def evaluate(self, context: CorpusContext) -> RuleResult:
        for node in context.part_nodes:
            pid = str(node.get("part_id", ""))
            if pid and not is_vanilla_part_id(pid):
                return RuleResult(
                    passed=False,
                    message=f"ship contains non-vanilla part: {pid!r}",
                )
        return RuleResult(passed=True)
