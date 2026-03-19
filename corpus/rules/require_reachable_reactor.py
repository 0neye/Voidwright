"""Corpus rule: reject ships with crew rooms but no reachable reactor."""

from __future__ import annotations

from corpus.context import CorpusContext
from corpus.rules.base import CorpusRule, RuleResult

__all__ = ["RequireReachableReactorRule"]


class RequireReachableReactorRule(CorpusRule):
    """Reject ships whose crew rooms cannot reach any reactor.

    Requires expansion graph data (``X_expansion_structural``). If the expansion
    graph is absent this rule raises ``RuntimeError`` at filter startup — call
    ``validate_corpus_has_expansion`` before processing begins.
    """

    name = "require_reachable_reactor"
    version = 1

    def evaluate(self, context: CorpusContext) -> RuleResult:
        if not context.has_expansion_graph:
            # Should have been caught at startup, but fail defensively.
            raise RuntimeError(
                f"require_reachable_reactor: expansion graph missing for "
                f"{context.source_path.name}"
            )
        if context.crew_room_count > 0 and context.crew_access_reactor_edges == 0:
            return RuleResult(
                passed=False,
                message="ship has crew rooms but no reachable reactor",
            )
        return RuleResult(passed=True)
