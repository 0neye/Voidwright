"""Model-building helpers for canonical and graph training corpora."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from common.geometry import is_vanilla_part_id, iter_ship_files, load_vanilla_part_geometry

from .corpus import iter_vanilla_parts_from_graph, iter_vanilla_parts_from_ship
from .order import order_ship_parts, order_ship_parts_from_graph, parts_touch
from .state import state_key
from .types import (
    END_TOKEN,
    ROOT_ANCHOR,
    RelativePlacementToken,
    ShipPart,
    TrainingConfig,
    TrainingStats,
    _config_as_dict,
)

__all__ = ["build_payload_from_corpus", "build_payload_from_graph_corpus"]


def _accumulate_ordered_ship_tokens(
    *,
    ordered_parts: Sequence[Tuple[ShipPart, Optional[ShipPart]]],
    config: TrainingConfig,
    geometry_cache: Dict[str, object],
    stats: TrainingStats,
    start_counts: Counter,
    transition_counts: Dict[str, Counter],
    part_frequency: Counter,
) -> None:
    """Accumulate token counts from one ordered ship sequence

    Args:
        ordered_parts: Ordered `(part, anchor)` tuples for a single ship
        config: Training configuration used for Markov order
        geometry_cache: Shared vanilla geometry cache
        stats: Mutable aggregate training stats
        start_counts: Mutable root-token frequency counter
        transition_counts: Mutable state -> token frequency map
        part_frequency: Mutable emitted part ID frequency counter
    """

    history: List[str] = []
    for idx, (part, anchor) in enumerate(ordered_parts):
        if idx == 0:
            root_token = RelativePlacementToken(
                part_id=part.part_id,
                rotation=part.rotation,
                anchor_part_id=ROOT_ANCHOR,
                anchor_rotation=0,
                dx=0,
                dy=0,
            )
            root_key = root_token.as_key()
            start_counts[root_key] += 1
            history.append(root_key)
            part_frequency[part.part_id] += 1
            stats.root_tokens += 1
            continue

        assert anchor is not None
        transition_token = RelativePlacementToken(
            part_id=part.part_id,
            rotation=part.rotation,
            anchor_part_id=anchor.part_id,
            anchor_rotation=anchor.rotation,
            dx=part.x - anchor.x,
            dy=part.y - anchor.y,
        )
        transition_key = transition_token.as_key()
        transition_counts[state_key(history, config.markov_order)][transition_key] += 1
        history.append(transition_key)
        part_frequency[part.part_id] += 1
        stats.transition_tokens += 1
        if parts_touch(anchor, part, geometry_cache):
            stats.touching_transitions += 1
        else:
            stats.non_touching_transitions += 1

    transition_counts[state_key(history, config.markov_order)][END_TOKEN] += 1
    stats.end_tokens += 1


def build_payload_from_corpus(input_dir: Path, config: TrainingConfig) -> dict:
    """Build a serialized model payload from canonical extracted ship JSON files"""

    geometry_cache = load_vanilla_part_geometry()
    stats = TrainingStats()
    start_counts: Counter = Counter()
    transition_counts: Dict[str, Counter] = defaultdict(Counter)
    part_frequency: Counter = Counter()

    for ship_path in iter_ship_files(input_dir):
        stats.ships_seen += 1
        with ship_path.open(encoding="utf-8") as file_handle:
            ship_data = json.load(file_handle)

        all_parts = [part for part in ship_data.get("Parts", []) if isinstance(part, dict)]
        for part in all_parts:
            part_id = part.get("ID") or part.get("IDString")
            if not part_id:
                continue
            if not is_vanilla_part_id(part_id):
                stats.non_vanilla_parts_excluded += 1
            elif part_id not in geometry_cache:
                stats.unknown_vanilla_geometry_excluded += 1

        vanilla_parts = iter_vanilla_parts_from_ship(ship_data, geometry_cache=geometry_cache)
        if config.part_allowlist is not None:
            vanilla_parts = [part for part in vanilla_parts if part.part_id in config.part_allowlist]

        stats.vanilla_parts_seen += len(vanilla_parts)
        if len(vanilla_parts) < config.min_parts_per_ship:
            stats.ships_skipped_too_small += 1
            continue
        if len(vanilla_parts) > config.max_parts_per_ship:
            stats.ships_skipped_too_large += 1
            continue

        stats.ships_used += 1
        stats.vanilla_parts_used += len(vanilla_parts)
        ordered_parts = order_ship_parts(vanilla_parts, anchor_window=config.anchor_window)
        _accumulate_ordered_ship_tokens(
            ordered_parts=ordered_parts,
            config=config,
            geometry_cache=geometry_cache,
            stats=stats,
            start_counts=start_counts,
            transition_counts=transition_counts,
            part_frequency=part_frequency,
        )

    return {
        "schema_version": 2,
        "model_type": "relative_markov_first_pass",
        "config": _config_as_dict(config),
        "corpus": {"input_dir": str(input_dir)},
        "stats": asdict(stats),
        "start_counts": dict(start_counts),
        "transition_counts": {key: dict(counter) for key, counter in transition_counts.items()},
        "part_frequency": dict(part_frequency),
        "notes": [
            "Vanilla-only corpus model built from canonical deduped extracted ship JSON.",
            "Relative placement uses origin-to-origin deltas between an anchor part and the next emitted part.",
            "Sampling rejects overlaps using full vanilla footprint geometry from game-file exports.",
            "This first pass intentionally defers door synthesis, accessibility cleanup, and gameplay-grade legality checks.",
        ],
    }


def build_payload_from_graph_corpus(graph_dir: Path, config: TrainingConfig) -> dict:
    """Build a serialized model payload from pre-generated graph JSON artifacts"""

    geometry_cache = load_vanilla_part_geometry()
    stats = TrainingStats()
    start_counts: Counter = Counter()
    transition_counts: Dict[str, Counter] = defaultdict(Counter)
    part_frequency: Counter = Counter()

    graph_files = sorted(file_path for file_path in graph_dir.glob("*.json") if file_path.name != "manifest.json" and not file_path.name.startswith("."))

    for graph_path in graph_files:
        stats.ships_seen += 1
        try:
            with graph_path.open(encoding="utf-8") as file_handle:
                graph_data = json.load(file_handle)
        except (json.JSONDecodeError, OSError):
            continue

        all_nodes = graph_data.get("graphs", {}).get("A_structural_part_graph", {}).get("nodes", [])
        for node in all_nodes:
            part_id = node.get("part_id", "")
            if not is_vanilla_part_id(part_id):
                stats.non_vanilla_parts_excluded += 1
            elif part_id not in geometry_cache:
                stats.unknown_vanilla_geometry_excluded += 1

        parts, node_id_to_idx = iter_vanilla_parts_from_graph(graph_data, geometry_cache=geometry_cache)

        if config.part_allowlist is not None:
            filtered_parts: List[ShipPart] = []
            filtered_node_id_map: Dict[int, int] = {}
            for node_id, old_idx in sorted(node_id_to_idx.items(), key=lambda item: item[1]):
                if parts[old_idx].part_id in config.part_allowlist:
                    filtered_node_id_map[node_id] = len(filtered_parts)
                    filtered_parts.append(parts[old_idx])
            parts = filtered_parts
            node_id_to_idx = filtered_node_id_map

        stats.vanilla_parts_seen += len(parts)
        if len(parts) < config.min_parts_per_ship:
            stats.ships_skipped_too_small += 1
            continue
        if len(parts) > config.max_parts_per_ship:
            stats.ships_skipped_too_large += 1
            continue

        stats.ships_used += 1
        stats.vanilla_parts_used += len(parts)

        edges = graph_data.get("graphs", {}).get("A_structural_part_graph", {}).get("edges", [])
        ordered_parts = order_ship_parts_from_graph(
            parts,
            node_id_to_idx,
            edges,
            anchor_window=config.anchor_window,
        )
        _accumulate_ordered_ship_tokens(
            ordered_parts=ordered_parts,
            config=config,
            geometry_cache=geometry_cache,
            stats=stats,
            start_counts=start_counts,
            transition_counts=transition_counts,
            part_frequency=part_frequency,
        )

    return {
        "schema_version": 2,
        "model_type": "relative_markov_first_pass",
        "config": _config_as_dict(config),
        "corpus": {"graph_dir": str(graph_dir), "source": "pre-generated ship graphs"},
        "stats": asdict(stats),
        "start_counts": dict(start_counts),
        "transition_counts": {key: dict(counter) for key, counter in transition_counts.items()},
        "part_frequency": dict(part_frequency),
        "notes": [
            "Vanilla-only corpus model built from pre-generated ship graph JSONs.",
            "Ordering uses BFS traversal over A_structural_part_graph touching edges.",
            "Anchors are touching graph neighbours (BFS parent), improving structural connectivity.",
            "Disconnected components fall back to geometric nearest-neighbour anchoring.",
            "Relative placement uses origin-to-origin deltas between anchor and next part.",
            "Sampling rejects overlaps using full vanilla footprint geometry from game-file exports.",
        ],
    }
