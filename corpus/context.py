"""Per-ship context object for corpus filter rule evaluation."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

from graph_expansion.context import EXPANSION_GRAPH_NAME, STRUCTURAL_GRAPH_NAME
from graph_expansion.passes.travel_support import detect_part_role

__all__ = ["CorpusContext"]


class CorpusContext:
    """Wraps a parsed graph JSON payload and exposes cached helpers for rules.

    Rules should read from this context rather than digging through the raw
    payload directly.
    """

    def __init__(self, source_path: Path, payload: dict[str, Any]) -> None:
        self._source_path = source_path
        self._payload = payload

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def source_path(self) -> Path:
        return self._source_path

    @cached_property
    def ship_name(self) -> str:
        return str(self._payload.get("ship_name", self._source_path.stem))

    @cached_property
    def author(self) -> str:
        return str(self._payload.get("author", ""))

    # ------------------------------------------------------------------
    # Structural graph helpers
    # ------------------------------------------------------------------

    @cached_property
    def _graphs(self) -> dict[str, Any]:
        return self._payload.get("graphs", {})

    @cached_property
    def _structural_graph(self) -> dict[str, Any]:
        return self._graphs.get(STRUCTURAL_GRAPH_NAME, {})

    @cached_property
    def _structural_summary(self) -> dict[str, Any]:
        return self._structural_graph.get("summary", {})

    @cached_property
    def _structural_nodes(self) -> list[dict[str, Any]]:
        return self._structural_graph.get("nodes", [])

    @cached_property
    def part_nodes(self) -> list[dict[str, Any]]:
        """Structural part nodes from the structural graph."""
        return list(self._structural_nodes)

    @cached_property
    def part_count(self) -> int:
        """Number of parts according to the structural graph summary."""
        return int(self._structural_summary.get("parts", len(self.part_nodes)))

    @cached_property
    def occupied_cells(self) -> int:
        """Total occupied 2x-cell count from the structural graph summary."""
        return int(self._structural_summary.get("occupied_cells", 0))

    @cached_property
    def traversable_cells(self) -> int:
        """Total traversable 2x-cell count from the structural graph summary."""
        return int(self._structural_summary.get("traversable_cells", 0))

    @cached_property
    def crew_room_count(self) -> int:
        """Number of structural part nodes identified as crew rooms."""
        return sum(
            1
            for node in self.part_nodes
            if detect_part_role(str(node.get("part_id", ""))) == "crew_room"
        )

    # ------------------------------------------------------------------
    # Expansion graph helpers
    # ------------------------------------------------------------------

    @cached_property
    def has_expansion_graph(self) -> bool:
        """True when the expansion graph is present in the payload."""
        return EXPANSION_GRAPH_NAME in self._graphs

    @cached_property
    def _expansion_summary(self) -> dict[str, Any]:
        return self._graphs.get(EXPANSION_GRAPH_NAME, {}).get("summary", {})

    @cached_property
    def crew_access_reactor_edges(self) -> int:
        """Count of crew_access_reactor cross-edges from the expansion summary."""
        return int(self._expansion_summary.get("crew_access_reactor_edges", 0))
