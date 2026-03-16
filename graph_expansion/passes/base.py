"""Base interfaces for graph expansion passes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from graph_expansion.context import ExpansionContext

__all__ = ["ExpansionPass"]


class ExpansionPass(ABC):
    """Abstract base class for expansion passes.

    Each pass declares:

    - ``name``: short stable identifier
    - ``version``: integer version for provenance
    - ``requires``: optional tuple of pass names that should precede it
    - ``provides``: optional tuple naming caches or annotations
    """

    #: Stable pass name used in metadata and dependency declarations.
    name: str
    #: Integer version for this pass implementation.
    version: int
    #: Optional dependency hints for future orchestration logic.
    requires: tuple[str, ...] = ()
    #: Optional declaration of caches/annotations produced by this pass.
    provides: tuple[str, ...] = ()

    @abstractmethod
    def run(self, context: ExpansionContext) -> Mapping[str, Any] | None:
        """Execute the pass and mutate the provided context.

        Implementations may return a compact summary mapping that will be
        stored in the context's pass reports. Returning ``None`` is
        allowed for passes that do not have anything interesting to
        report.
        """

