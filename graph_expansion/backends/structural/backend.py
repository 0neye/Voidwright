"""Structural expansion backend for the graph expansion module."""

from __future__ import annotations

import argparse
import orjson
from concurrent.futures import as_completed
from pathlib import Path
from typing import Sequence

from common.files import inputs_needing_regeneration, prune_stale_json_outputs, write_output_version
from graph_expansion.base import ExpansionBackend
from preprocessing.concurrency import (
    add_concurrency_arguments,
    resolve_worker_count,
    run_auto_parallel_work,
)

__all__ = ["StructuralExpansionBackend"]

_BACKEND_NAME = "structural"
_EXPANSION_VERSION = 1
_EXPANSION_GRAPH_NAME = "X_expansion_structural"

# Part-ID substrings that identify corridor and moving-walkway parts.  These
# part types do not require a door to be crew-traversable to each other when
# their walkable cells are adjacent.
_CORRIDOR_LIKE_SUBSTRINGS: tuple[str, ...] = ("corridor", "walkway")


def _is_corridor_like(part_id: str) -> bool:
    """Return True when *part_id* identifies a corridor or moving-walkway part."""
    lower_id = part_id.lower()
    return any(token in lower_id for token in _CORRIDOR_LIKE_SUBSTRINGS)


def _build_traversable_clusters(nodes: list[dict], edges: list[dict]) -> list[list[int]]:
    """Group node IDs into traversable clusters.

    Two parts join the same cluster when either condition holds:

    1. **Door edge**: they are connected by an edge with ``kind == "door"`` in
       the structural part graph (applies to any two walkable parts).
    2. **Corridor-like adjacency**: both parts are corridor or moving-walkway
       parts (matched via :data:`_CORRIDOR_LIKE_SUBSTRINGS`) *and* at least one
       walkable cell of each part is adjacent in the 2x coordinate frame
       (differs by 2 in exactly one axis).

    Args:
        nodes: Structural part graph nodes.  Each node must have an ``"id"``
            field and may have ``"part_id"`` and ``"walkable_cells_2x"``.
        edges: Structural part graph edges.  Only ``kind == "door"`` edges
            participate in traversal connectivity.

    Returns:
        Sorted list of sorted member-ID lists, one per cluster.
    """

    parts_with_walkable: set[int] = {node["id"] for node in nodes if node.get("walkable_cells_2x")}
    if not parts_with_walkable:
        return []

    parent: dict[int, int] = {node_id: node_id for node_id in parts_with_walkable}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    # Rule 1: door edges connect any two walkable parts regardless of type.
    for edge in edges:
        if edge.get("kind") != "door":
            continue
        src = edge.get("source")
        tgt = edge.get("target")
        if src in parts_with_walkable and tgt in parts_with_walkable:
            union(src, tgt)

    # Rule 2: corridor-like parts merge when their walkable cells are adjacent.
    corridor_nodes = [
        node for node in nodes
        if node["id"] in parts_with_walkable and _is_corridor_like(node.get("part_id", ""))
    ]
    cell_to_corridor_parts: dict[tuple[int, int], set[int]] = {}
    for node in corridor_nodes:
        for cell in node.get("walkable_cells_2x", []):
            key = (cell[0], cell[1])
            cell_to_corridor_parts.setdefault(key, set()).add(node["id"])

    for (cx, cy), part_ids in cell_to_corridor_parts.items():
        # Merge corridor parts that share the same walkable cell.
        part_ids_list = sorted(part_ids)
        for i in range(1, len(part_ids_list)):
            union(part_ids_list[0], part_ids_list[i])
        # Merge corridor parts whose walkable cells are adjacent.
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            neighbor_parts = cell_to_corridor_parts.get((cx + dx, cy + dy))
            if neighbor_parts:
                for pid_a in part_ids:
                    for pid_b in neighbor_parts:
                        if pid_a != pid_b:
                            union(pid_a, pid_b)

    clusters: dict[int, list[int]] = {}
    for node_id in parts_with_walkable:
        root = find(node_id)
        clusters.setdefault(root, []).append(node_id)

    sorted_clusters = [sorted(member_ids) for member_ids in clusters.values()]
    return sorted(sorted_clusters)


def _enrich_graph(graph_data: dict) -> dict:
    """Add virtual nodes and cross-edges to one graph JSON payload.

    Adds a new ``X_expansion_structural`` graph containing:
    - One global ship-info node connected to every structural part node.
    - One traversable-cluster super-node per crew-reachable part cluster,
      connected to its member part nodes.

    Existing keys in ``graph_data`` are never removed or overwritten.
    """

    structural_graph = graph_data["graphs"]["A_structural_part_graph"]
    structural_nodes = structural_graph["nodes"]
    structural_edges = structural_graph.get("edges", [])
    ship_info = graph_data.get("ship", {})

    # Global ship-info node — one per file, carries top-level ship metadata
    global_node: dict = {
        "id": "global_ship",
        "kind": "global_ship_info",
        "ship": ship_info,
    }
    global_cross_edges = [
        {
            "source": "global_ship",
            "source_graph": _EXPANSION_GRAPH_NAME,
            "target": node["id"],
            "target_graph": "A_structural_part_graph",
            "kind": "global_member",
        }
        for node in structural_nodes
    ]

    # Traversable cluster super-nodes — connectivity determined by door edges
    # and corridor-like adjacency (see _build_traversable_clusters).
    clusters = _build_traversable_clusters(structural_nodes, structural_edges)
    cluster_nodes: list[dict] = []
    cluster_cross_edges: list[dict] = []
    for cluster_index, member_ids in enumerate(clusters):
        cluster_id = f"traversable_cluster_{cluster_index}"
        cluster_nodes.append({
            "id": cluster_id,
            "kind": "traversable_cluster",
            "member_count": len(member_ids),
        })
        for member_id in member_ids:
            cluster_cross_edges.append({
                "source": cluster_id,
                "source_graph": _EXPANSION_GRAPH_NAME,
                "target": member_id,
                "target_graph": "A_structural_part_graph",
                "kind": "super_member",
            })

    expansion_graph: dict = {
        "nodes": [global_node] + cluster_nodes,
        "cross_edges": global_cross_edges + cluster_cross_edges,
        "summary": {
            "global_ship_nodes": 1,
            "traversable_clusters": len(clusters),
            "global_member_edges": len(global_cross_edges),
            "super_member_edges": len(cluster_cross_edges),
        },
    }

    enriched = dict(graph_data)
    enriched["expansion"] = {
        "backend": _BACKEND_NAME,
        "version": _EXPANSION_VERSION,
        "graphs_added": [_EXPANSION_GRAPH_NAME],
    }
    enriched["graphs"] = {**graph_data["graphs"], _EXPANSION_GRAPH_NAME: expansion_graph}
    return enriched


def _read_existing_expansion_summary(output_path: Path) -> dict | None:
    """Read a compact summary from an already-expanded graph JSON file.

    Returns the same shape as the dict returned by _expand_single_graph, or
    None if the file cannot be read or is structurally incomplete.
    """
    try:
        graph_data = orjson.loads(output_path.read_bytes())
        summary = graph_data["graphs"][_EXPANSION_GRAPH_NAME]["summary"]
        return {
            "output_name": output_path.name,
            "traversable_clusters": summary["traversable_clusters"],
            "global_member_edges": summary["global_member_edges"],
            "super_member_edges": summary["super_member_edges"],
        }
    except Exception:
        return None


def _expand_single_graph(source_path_str: str, output_dir_str: str) -> dict:
    """Enrich one graph JSON file and write it to the output directory."""

    source_path = Path(source_path_str)
    output_dir = Path(output_dir_str)
    graph_data = orjson.loads(source_path.read_bytes())
    enriched = _enrich_graph(graph_data)
    output_path = output_dir / source_path.name
    output_path.write_bytes(orjson.dumps(enriched) + b"\n")
    summary = enriched["graphs"][_EXPANSION_GRAPH_NAME]["summary"]
    return {
        "output_name": output_path.name,
        "traversable_clusters": summary["traversable_clusters"],
        "global_member_edges": summary["global_member_edges"],
        "super_member_edges": summary["super_member_edges"],
    }


class StructuralExpansionBackend(ExpansionBackend):
    """Expansion backend that enriches structural graphs with virtual nodes and cross-edges."""

    name = _BACKEND_NAME

    def register_expand_parser(self, backend_subparsers: argparse._SubParsersAction) -> None:
        """Register the structural expand parser."""

        parser = backend_subparsers.add_parser(
            self.name,
            help="Enrich structural graph JSON with global and traversable-cluster virtual nodes",
        )
        parser.add_argument(
            "--input-dir",
            type=Path,
            default=Path("generated_ship_graphs_canonical"),
            help="Directory containing preprocessing graph JSON artifacts",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("expanded_ship_graphs"),
            help="Directory to write enriched graph JSON",
        )
        add_concurrency_arguments(
            parser,
            worker_flag="--workers",
            executor_flag="--executor",
            help_prefix="graph expansion",
        )

    def expand_dir(
        self,
        input_dir: Path,
        output_dir: Path,
        workers: int | None = None,
        executor: str = "auto",
    ) -> dict:
        """Expand all graph JSON files in *input_dir* and write results to *output_dir*.

        Args:
            input_dir: Directory containing preprocessing graph JSON artifacts
            output_dir: Destination directory for enriched graph JSON files
            workers: Optional worker-count override
            executor: Executor mode: ``auto``, ``thread``, or ``process``

        Returns:
            A summary payload describing the expansion results
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in input_dir.glob("*.json") if p.name != "manifest.json" and not p.name.startswith("."))

        if not files:
            print(f"[graph-expansion:structural] No graph JSON files found in {input_dir}")
            return {"files_expanded": 0, "files_skipped": 0, "traversable_clusters_total": 0}

        files_to_expand, skipped_files = inputs_needing_regeneration(
            files,
            output_dir,
            current_version=_EXPANSION_VERSION,
            version_key="expansion_version",
        )
        files_skipped = len(skipped_files)
        if files_skipped:
            print(
                f"[graph-expansion:structural] Skipping {files_skipped} up-to-date file(s) in {output_dir}",
                flush=True,
            )

        results: list[dict] = []
        if files_to_expand:
            worker_count = resolve_worker_count(
                task_count=len(files_to_expand),
                stage_name="graph_expansion",
                requested_workers=workers,
                requested_mode=executor,
            )

            def submit_expand_work(executor_factory: type) -> list[dict]:
                inner_results: list[dict] = []
                with executor_factory(max_workers=worker_count) as pool:
                    future_to_path = {
                        pool.submit(_expand_single_graph, str(f), str(output_dir)): f
                        for f in files_to_expand
                    }
                    for index, future in enumerate(as_completed(future_to_path), start=1):
                        try:
                            inner_results.append(future.result())
                        except Exception as exc:
                            failed_path = future_to_path[future]
                            print(
                                f"Warning: skipping {failed_path.name} — expansion failed: {exc}",
                                flush=True,
                            )
                        if index % 1000 == 0:
                            print(
                                f"Expanded {index}/{len(files_to_expand)} graph files with {worker_count} worker(s)...",
                                flush=True,
                            )
                return inner_results

            results, _ = run_auto_parallel_work(
                stage_name="graph_expansion",
                requested_mode=executor,
                worker_count=worker_count,
                submit_work=submit_expand_work,
            )

        # Prune stale outputs and record the current expansion version.
        # Exclude manifest.json so in-place expansion (output_dir == input_dir)
        # does not delete the preprocessing manifest written by the graphs stage.
        pruned_count = prune_stale_json_outputs(
            output_dir, (f.name for f in files), exclude=["manifest.json"]
        )
        if pruned_count:
            print(
                f"[graph-expansion:structural] Pruned {pruned_count} stale file(s) from {output_dir}",
                flush=True,
            )

        # Always persist the version sentinel. Failed expansions produce no
        # output file, so there is no stale artifact to hide — the failed
        # source will simply be retried on the next run.
        write_output_version(output_dir, "expansion_version", _EXPANSION_VERSION)

        files_expanded = len(results)

        # Collect summaries from skipped (up-to-date) output files so the
        # printed totals reflect the full corpus, not just the incremental delta.
        for skipped_path in skipped_files:
            summary = _read_existing_expansion_summary(output_dir / skipped_path.name)
            if summary is not None:
                results.append(summary)

        total_clusters = sum(r["traversable_clusters"] for r in results)
        print(
            f"[graph-expansion:structural] expanded {files_expanded} files, "
            f"skipped {files_skipped}, "
            f"{total_clusters} traversable clusters total -> {output_dir}"
        )
        return {
            "files_expanded": files_expanded,
            "files_skipped": files_skipped,
            "traversable_clusters_total": total_clusters,
            "output_dir": str(output_dir),
        }

    def run_expand(self, args: argparse.Namespace) -> int:
        """Run structural graph expansion from CLI args."""

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        self.expand_dir(
            input_dir=input_dir,
            output_dir=output_dir,
            workers=args.workers,
            executor=args.executor,
        )
        return 0
