"""Base types for corpus filter rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corpus.context import CorpusContext

__all__ = ["RuleResult", "CorpusRule"]


@dataclass(slots=True)
class RuleResult:
    """Result returned by a corpus rule evaluation."""

    passed: bool
    message: str | None = None


class CorpusRule(ABC):
    """Abstract base class for corpus filter rules."""

    name: str
    version: int = 1

    @abstractmethod
    def evaluate(self, context: "CorpusContext") -> RuleResult:
        raise NotImplementedError
