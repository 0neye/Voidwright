"""Structural graph expansion pipeline.

This module is the canonical graph-expansion implementation for Voidwright.
It orchestrates an ordered list of expansion passes over an ExpansionContext
and provides both single-payload and directory-oriented entrypoints.
"""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed
from pathlib import Path
from typing import Sequence

import orjson

from common.files import inputs_needing_regeneration, prune_stale_json_outputs, write_output_version
from graph_expansion.context import EXPANSION_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base_indexes import BaseIndexesPass
from graph_expansion.passes.base import ExpansionPass
from graph_expansion.passes.global_ship_info import GlobalShipInfoPass
from graph_expansion.passes.traversable_clusters import (
    TraversableClustersPass,
    build_traversable_clusters,
    is_corridor_like,
)
from graph_expansion.passes.crew_access_layer1 import Layer1CrewAccessPass
from graph_expansion.passes.core_support_layer2 import Layer2CoreSupportPass
from graph_expansion.passes.hull_perimeter import HullPerimeterPass
from graph_expansion.passes.spatial_zones import SpatialZonesPass, SpatialZonesRotatedPass
from graph_expansion.passes.weapon_groups import WeaponGroupsPass
from graph_expansion.passes.global_virtual_linker import GlobalVirtualLinkerPass
from preprocessing.concurrency import (
    add_concurrency_arguments,
    resolve_worker_count,
    run_auto_parallel_work,
)

__all__ = [
    "DEFAULT_PASSES",
    "EXPANSION_GRAPH_NAME",
    "EXPANSION_NAME",
    "EXPANSION_VERSION",
    "build_parser",
    "build_traversable_clusters",
    "enrich_graph",
    "expand_dir",
    "expand_graph_file",
    "is_corridor_like",
    "read_existing_expansion_summary",
    "run_from_args",
]

EXPANSION_NAME = "structural"
EXPANSION_VERSION = 9
DEFAULT_PASSES: tuple[type[ExpansionPass], ...] = (
    BaseIndexesPass,
    GlobalShipInfoPass,
    TraversableClustersPass,
    Layer1CrewAccessPass,
    Layer2CoreSupportPass,
    HullPerimeterPass,
    SpatialZonesPass,
    SpatialZonesRotatedPass,
    WeaponGroupsPass,
    GlobalVirtualLinkerPass,
)


def iter_default_passes() -> list[ExpansionPass]:
    """Return the canonical ordered structural expansion pass list."""

    return [pass_type() for pass_type in DEFAULT_PASSES]



def enrich_graph(graph_data: dict, passes: Sequence[ExpansionPass] | None = None) -> dict:
    """Enrich one graph payload by running the ordered pass pipeline."""

    context = ExpansionContext(
        graph_data,
        expansion_name=EXPANSION_NAME,
        expansion_version=EXPANSION_VERSION,
    )

    for expansion_pass in (list(passes) if passes is not None else iter_default_passes()):
        summary = expansion_pass.run(context)
        context.add_pass_report(expansion_pass.name, expansion_pass.version, summary)

    return context.finalize()



def read_existing_expansion_summary(output_path: Path) -> dict | None:
    """Read a compact summary from an already-expanded graph JSON file."""

    try:
        graph_data = orjson.loads(output_path.read_bytes())
        summary = graph_data["graphs"][EXPANSION_GRAPH_NAME]["summary"]
        return {
            "output_name": output_path.name,
            "traversable_clusters": summary["traversable_clusters"],
            "global_member_edges": summary["global_member_edges"],
            "super_member_edges": summary["super_member_edges"],
        }
    except Exception:
        return None



def expand_graph_file(source_path_str: str, output_dir_str: str) -> dict:
    """Enrich one graph JSON file and write it to the output directory."""

    source_path = Path(source_path_str)
    output_dir = Path(output_dir_str)
    graph_data = orjson.loads(source_path.read_bytes())
    enriched = enrich_graph(graph_data)
    output_path = output_dir / source_path.name
    output_path.write_bytes(orjson.dumps(enriched) + b"\n")
    summary = enriched["graphs"][EXPANSION_GRAPH_NAME]["summary"]
    return {
        "output_name": output_path.name,
        "traversable_clusters": summary["traversable_clusters"],
        "global_member_edges": summary["global_member_edges"],
        "super_member_edges": summary["super_member_edges"],
    }



def expand_dir(
    input_dir: Path,
    output_dir: Path,
    workers: int | None = None,
    executor: str = "auto",
) -> dict:
    """Expand all graph JSON files in *input_dir* and write results to *output_dir*."""

    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in input_dir.glob("*.json") if p.name != "manifest.json" and not p.name.startswith("."))

    if not files:
        print(f"[graph-expansion:{EXPANSION_NAME}] No graph JSON files found in {input_dir}")
        return {"files_expanded": 0, "files_skipped": 0, "traversable_clusters_total": 0}

    files_to_expand, skipped_files = inputs_needing_regeneration(
        files,
        output_dir,
        current_version=EXPANSION_VERSION,
        version_key="expansion_version",
    )
    files_skipped = len(skipped_files)
    if files_skipped:
        print(
            f"[graph-expansion:{EXPANSION_NAME}] Skipping {files_skipped} up-to-date file(s) in {output_dir}",
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
                    pool.submit(expand_graph_file, str(file_path), str(output_dir)): file_path
                    for file_path in files_to_expand
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

    pruned_count = prune_stale_json_outputs(
        output_dir, (file_path.name for file_path in files), exclude=["manifest.json"]
    )
    if pruned_count:
        print(
            f"[graph-expansion:{EXPANSION_NAME}] Pruned {pruned_count} stale file(s) from {output_dir}",
            flush=True,
        )

    write_output_version(output_dir, "expansion_version", EXPANSION_VERSION)

    files_expanded = len(results)
    for skipped_path in skipped_files:
        summary = read_existing_expansion_summary(output_dir / skipped_path.name)
        if summary is not None:
            results.append(summary)

    total_clusters = sum(result["traversable_clusters"] for result in results)
    print(
        f"[graph-expansion:{EXPANSION_NAME}] expanded {files_expanded} files, "
        f"skipped {files_skipped}, "
        f"{total_clusters} traversable clusters total -> {output_dir}"
    )
    return {
        "files_expanded": files_expanded,
        "files_skipped": files_skipped,
        "traversable_clusters_total": total_clusters,
        "output_dir": str(output_dir),
    }



def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Build the structural graph-expansion parser."""

    parser = parser or argparse.ArgumentParser(
        description="Enrich preprocessing graph JSON with virtual nodes and cross-edges."
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
    return parser



def run_from_args(args: argparse.Namespace) -> int:
    """Run graph expansion from parsed CLI args."""

    expand_dir(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        workers=args.workers,
        executor=args.executor,
    )
    return 0
