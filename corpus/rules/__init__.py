"""Built-in corpus filter rules."""

from __future__ import annotations

from corpus.rules.base import CorpusRule, RuleResult
from corpus.rules.max_size import MaxSizeRule
from corpus.rules.require_crew_rooms import RequireCrewRoomsRule
from corpus.rules.require_reachable_reactor import RequireReachableReactorRule
from corpus.rules.vanilla_only import VanillaOnlyRule

__all__ = [
    "CorpusRule",
    "MaxSizeRule",
    "RequireCrewRoomsRule",
    "RequireReachableReactorRule",
    "RuleResult",
    "VanillaOnlyRule",
]
