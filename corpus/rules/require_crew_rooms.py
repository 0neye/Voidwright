"""Corpus rule: reject ships with no crew rooms."""

from __future__ import annotations

from corpus.context import CorpusContext
from corpus.rules.base import CorpusRule, RuleResult

__all__ = ["RequireCrewRoomsRule"]


class RequireCrewRoomsRule(CorpusRule):
    """Reject ships that contain no crew-room parts."""

    name = "require_crew_rooms"
    version = 1

    def evaluate(self, context: CorpusContext) -> RuleResult:
        if context.crew_room_count == 0:
            return RuleResult(passed=False, message="ship has no crew rooms")
        return RuleResult(passed=True)
