"""ExpansionContext for graph expansion passes.

This module defines the stateful per-run context object used by the
graph expansion framework. An ``ExpansionContext`` instance wraps a
single source graph JSON payload and is mutated by a sequence of
ordered expansion passes before being serialized back to JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, MutableMapping

__all__ = [
    "EXPANSION_GRAPH_NAME",
    "ExpansionContext",
    "PassReport",
    "STRUCTURAL_GRAPH_NAME",
]

#: Canonical name of the structural source graph in the preprocessing payload.
STRUCTURAL_GRAPH_NAME = "A_structural_part_graph"
#: Canonical name of the emitted structural expansion graph.
EXPANSION_GRAPH_NAME = "X_expansion_structural"


@dataclass(slots=True)
class PassReport:
    """Compact record of one expansion pass execution.

    Attributes:
        name: Human-readable pass name, usually matching ``ExpansionPass.name``.
        version: Integer pass version.
        summary: Optional pass-specific summary payload. This is kept
            lightweight and intended primarily for debugging and tests.
    """

    name: str
    version: int
    summary: Mapping[str, Any] | None = None


class ExpansionContext:
    """Mutable per-artifact context shared across expansion passes.

    The context owns:

    - the original source payload
    - expansion pipeline identity and version
    - lazily-built caches and annotations
    - emitted graphs and per-pass reports

    Expansion passes are expected to mutate the context in-place and to
    record a ``PassReport`` via :meth:`add_pass_report`. Once all passes
    have run, :meth:`finalize` materializes a new graph JSON payload
    that:

    - preserves all existing top-level keys from the source
    - merges any emitted graphs into the ``graphs`` mapping
    - attaches an ``expansion`` block with pipeline metadata and the
      ordered list of executed passes
    """

    def __init__(
        self,
        graph_data: Mapping[str, Any],
        *,
        expansion_name: str,
        expansion_version: int,
    ) -> None:
        """Initialize a new context for one graph payload.

        Args:
            graph_data: Parsed graph JSON payload from preprocessing.
            expansion_name: Name of the expansion pipeline (for example
                ``"structural"``).
            expansion_version: Integer pipeline version used for the
                top-level ``expansion.version`` field.
        """

        self.source: Mapping[str, Any] = graph_data
        self.expansion_name: str = expansion_name
        self.expansion_version: int = expansion_version

        # Caches and annotations are shared across passes but never
        # serialized directly into the final JSON artifact.
        self.caches: Dict[str, Any] = {}
        self.annotations: Dict[str, Any] = {}

        # Emitted graphs are stored under their graph-name keys. Each
        # graph dict may contain ``nodes``, ``edges``, ``cross_edges``,
        # and ``summary`` entries.
        self.emitted_graphs: Dict[str, Dict[str, Any]] = {}

        # Ordered log of executed passes for debugging and metadata.
        self.pass_reports: List[PassReport] = []

    # ------------------------------------------------------------------
    # Source accessors
    # ------------------------------------------------------------------

    def get_source_graph(self, name: str) -> Mapping[str, Any]:
        """Return a source graph by name.

        Args:
            name: Key in the source ``graphs`` mapping.

        Raises:
            KeyError: If the source payload lacks a ``graphs`` section
                or the named graph.
        """

        graphs = self.source.get("graphs")
        if not isinstance(graphs, Mapping):
            raise KeyError("source payload does not contain a 'graphs' mapping")
        graph = graphs.get(name)
        if graph is None:
            raise KeyError(f"source payload does not contain graph {name!r}")
        if not isinstance(graph, Mapping):
            raise KeyError(f"graph {name!r} is not a mapping")
        return graph

    # ------------------------------------------------------------------
    # Caches and annotations
    # ------------------------------------------------------------------

    def get_or_build_cache(self, key: str, builder: Callable[[], Any]) -> Any:
        """Return a cached value, building it once if necessary.

        The builder is only invoked when the cache entry is missing.
        """

        if key in self.caches:
            return self.caches[key]
        value = builder()
        self.caches[key] = value
        return value

    def get_annotation(self, key: str, default: Any | None = None) -> Any:
        """Return a transient annotation value, or *default* if absent."""

        return self.annotations.get(key, default)

    def set_annotation(self, key: str, value: Any) -> None:
        """Store a transient annotation value for later passes."""

        self.annotations[key] = value

    # ------------------------------------------------------------------
    # Emitted graphs
    # ------------------------------------------------------------------

    def ensure_emitted_graph(self, name: str) -> MutableMapping[str, Any]:
        """Return an emitted-graph dict, creating it if needed.

        The created graph is initialized with empty ``nodes``,
        ``edges``, and ``cross_edges`` lists and an empty ``summary``
        mapping so passes can rely on those keys being present.
        """

        graph = self.emitted_graphs.get(name)
        if graph is None:
            graph = {
                "nodes": [],
                "edges": [],
                "cross_edges": [],
                "summary": {},
            }
            self.emitted_graphs[name] = graph
        else:
            # Ensure required keys exist even if a pass created a
            # partial structure.
            graph.setdefault("nodes", [])
            graph.setdefault("edges", [])
            graph.setdefault("cross_edges", [])
            graph.setdefault("summary", {})
        return graph

    def increment_summary(self, graph_name: str, **counts: int) -> None:
        """Increment named counter(s) in an emitted graph's summary block.

        Creates missing keys at zero before incrementing.  The graph is
        created via :meth:`ensure_emitted_graph` if it does not yet exist.
        """

        summary = self.ensure_emitted_graph(graph_name)["summary"]
        for key, value in counts.items():
            summary[key] = summary.get(key, 0) + value

    # ------------------------------------------------------------------
    # Pass reports
    # ------------------------------------------------------------------

    def add_pass_report(
        self,
        name: str,
        version: int,
        summary: Mapping[str, Any] | None,
    ) -> None:
        """Record the execution of one expansion pass."""

        self.pass_reports.append(PassReport(name=name, version=version, summary=summary))

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _build_expansion_metadata(self) -> Dict[str, Any]:
        """Construct the final ``expansion`` metadata block."""

        graphs_added: List[str] = sorted(self.emitted_graphs)
        passes: List[Dict[str, Any]] = [
            {"name": report.name, "version": report.version} for report in self.pass_reports
        ]
        return {
            "backend": self.expansion_name,
            "version": self.expansion_version,
            "graphs_added": graphs_added,
            "passes": passes,
        }

    def finalize(self) -> Dict[str, Any]:
        """Materialize the final enriched graph payload.

        The returned dict is a shallow copy of the source payload with:

        - the ``graphs`` mapping updated to include any emitted graphs
        - an ``expansion`` block describing the pipeline and executed
          passes

        The original ``source`` mapping is never mutated.
        """

        # Start from a shallow copy of the original mapping to preserve
        # any non-graph top-level keys.
        enriched: Dict[str, Any] = dict(self.source)

        # Merge graphs deterministically: all original graphs are kept
        # and emitted graphs are added or overwritten by name if a pass
        # explicitly targets an existing key.
        source_graphs = self.source.get("graphs") or {}
        if not isinstance(source_graphs, Mapping):
            raise KeyError("source payload 'graphs' field is not a mapping")

        merged_graphs: Dict[str, Any] = dict(source_graphs)
        # Insert emitted graphs in sorted-name order for reproducibility.
        for graph_name in sorted(self.emitted_graphs):
            merged_graphs[graph_name] = self.emitted_graphs[graph_name]

        enriched["graphs"] = merged_graphs
        enriched["expansion"] = self._build_expansion_metadata()
        return enriched

