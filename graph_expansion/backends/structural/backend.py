"""Structural expansion backend for the graph expansion module."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import as_completed
from pathlib import Path
from typing import Sequence

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


def _build_traversable_clusters(nodes: list[dict]) -> list[list[int]]:
    """Group node IDs into traversable clusters via adjacent walkable-cell connectivity.

    Two parts are placed in the same cluster when any of their walkable cells
    are adjacent (differ by 2 in exactly one axis in the 2x coordinate frame,
    which corresponds to touching grid cells in game space).
    """

    cell_to_parts: dict[tuple[int, int], set[int]] = {}
    for node in nodes:
        for cell in node.get("walkable_cells_2x", []):
            key = (cell[0], cell[1])
            cell_to_parts.setdefault(key, set()).add(node["id"])

    parent: dict[int, int] = {node["id"]: node["id"] for node in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for (cx, cy), part_ids in cell_to_parts.items():
        # Merge parts that share the same walkable cell
        part_ids_list = sorted(part_ids)
        for i in range(1, len(part_ids_list)):
            union(part_ids_list[0], part_ids_list[i])
        # Merge parts whose walkable cells are adjacent
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            neighbor_parts = cell_to_parts.get((cx + dx, cy + dy))
            if neighbor_parts:
                for pid_a in part_ids:
                    for pid_b in neighbor_parts:
                        if pid_a != pid_b:
                            union(pid_a, pid_b)

    parts_with_walkable = {node["id"] for node in nodes if node.get("walkable_cells_2x")}
    clusters: dict[int, list[int]] = {}
    for node_id in parts_with_walkable:
        root = find(node_id)
        clusters.setdefault(root, []).append(node_id)

    return [sorted(member_ids) for member_ids in sorted(clusters.values())]


def _enrich_graph(graph_data: dict) -> dict:
    """Add virtual nodes and cross-edges to one graph JSON payload.

    Adds a new ``X_expansion_structural`` graph containing:
    - One global ship-info node connected to every structural part node.
    - One traversable-cluster super-node per crew-reachable part cluster,
      connected to its member part nodes.

    Existing keys in ``graph_data`` are never removed or overwritten.
    """

    structural_nodes = graph_data["graphs"]["A_structural_part_graph"]["nodes"]
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

    # Traversable cluster super-nodes — derived from walkable_cells_2x on structural nodes
    clusters = _build_traversable_clusters(structural_nodes)
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


def _expand_single_graph(source_path_str: str, output_dir_str: str) -> dict:
    """Enrich one graph JSON file and write it to the output directory."""

    source_path = Path(source_path_str)
    output_dir = Path(output_dir_str)
    graph_data = json.loads(source_path.read_text(encoding="utf-8"))
    enriched = _enrich_graph(graph_data)
    output_path = output_dir / source_path.name
    output_path.write_text(json.dumps(enriched, separators=(",", ":")) + "\n", encoding="utf-8")
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
            required=True,
            help="Directory containing preprocessing graph JSON artifacts",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help=(
                "Directory to write enriched graph JSON. "
                "Defaults to --input-dir for in-place enrichment."
            ),
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
        files = sorted(p for p in input_dir.glob("*.json") if p.name != "manifest.json")

        if not files:
            print(f"[graph-expansion:structural] No graph JSON files found in {input_dir}")
            return {"files_expanded": 0, "traversable_clusters_total": 0}

        worker_count = resolve_worker_count(
            task_count=len(files),
            stage_name="graph_expansion",
            requested_workers=workers,
            requested_mode=executor,
        )

        def submit_expand_work(executor_factory: type) -> list[dict]:
            results: list[dict] = []
            with executor_factory(max_workers=worker_count) as pool:
                future_to_path = {
                    pool.submit(_expand_single_graph, str(f), str(output_dir)): f
                    for f in files
                }
                for index, future in enumerate(as_completed(future_to_path), start=1):
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        failed_path = future_to_path[future]
                        print(
                            f"Warning: skipping {failed_path.name} — expansion failed: {exc}",
                            flush=True,
                        )
                    if index % 1000 == 0:
                        print(
                            f"Expanded {index}/{len(files)} graph files with {worker_count} worker(s)...",
                            flush=True,
                        )
            return results

        results, _ = run_auto_parallel_work(
            stage_name="graph_expansion",
            requested_mode=executor,
            worker_count=worker_count,
            submit_work=submit_expand_work,
        )

        total_clusters = sum(r["traversable_clusters"] for r in results)
        print(
            f"[graph-expansion:structural] expanded {len(results)} files, "
            f"{total_clusters} traversable clusters total -> {output_dir}"
        )
        return {
            "files_expanded": len(results),
            "traversable_clusters_total": total_clusters,
            "output_dir": str(output_dir),
        }

    def run_expand(self, args: argparse.Namespace) -> int:
        """Run structural graph expansion from CLI args."""

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir
        self.expand_dir(
            input_dir=input_dir,
            output_dir=output_dir,
            workers=args.workers,
            executor=args.executor,
        )
        return 0
