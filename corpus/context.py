"""Per-ship context object for corpus filter rule evaluation."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

__all__ = ["CorpusContext"]

# Graph name constants (mirrors graph_expansion.context)
_STRUCTURAL_GRAPH_NAME = "A_structural_part_graph"
_EXPANSION_GRAPH_NAME = "X_expansion_structural"

# Crew-room part-ID substrings (must stay aligned with
# graph_expansion/passes/travel_support.py _CREW_ROOM_SUBSTRINGS)
_CREW_ROOM_SUBSTRINGS: tuple[str, ...] = ("crew_quarters", "quarters")


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
    def _structural_graph(self) -> dict[str, Any]:
        return self._payload.get("graphs", {}).get(_STRUCTURAL_GRAPH_NAME, {})

    @cached_property
    def _structural_summary(self) -> dict[str, Any]:
        return self._structural_graph.get("summary", {})

    @cached_property
    def _structural_nodes(self) -> list[dict[str, Any]]:
        return self._structural_graph.get("nodes", [])

    @cached_property
    def part_nodes(self) -> list[dict[str, Any]]:
        """Structural part nodes (excludes virtual/other node types)."""
        return [n for n in self._structural_nodes if n.get("kind") == "part"]

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
        """Number of structural nodes whose part_id matches crew-room substrings."""
        count = 0
        for node in self.part_nodes:
            part_id = str(node.get("part_id", "")).lower()
            if any(token in part_id for token in _CREW_ROOM_SUBSTRINGS):
                count += 1
        return count

    # ------------------------------------------------------------------
    # Expansion graph helpers
    # ------------------------------------------------------------------

    @cached_property
    def has_expansion_graph(self) -> bool:
        """True when the expansion graph is present in the payload."""
        return _EXPANSION_GRAPH_NAME in self._payload.get("graphs", {})

    @cached_property
    def _expansion_summary(self) -> dict[str, Any]:
        return (
            self._payload.get("graphs", {})
            .get(_EXPANSION_GRAPH_NAME, {})
            .get("summary", {})
        )

    @cached_property
    def crew_access_reactor_edges(self) -> int:
        """Count of crew_access_reactor cross-edges from the expansion summary."""
        return int(self._expansion_summary.get("crew_access_reactor_edges", 0))
