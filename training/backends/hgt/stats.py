"""Corpus statistics and calibration helpers for the HGT backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import torch

from training.backends.hgt.vocab import _SKIP_FILENAMES

__all__ = [
    "collect_corpus_stats",
    "rotation_class_weights_from_stats",
]

_VIRTUAL_MEMBER_EDGE_KINDS: tuple[str, ...] = (
    "super_member",
    "thermal_member",
    "zone_member",
    "zone_member_rotated",
    "weapon_member",
)


def _quantile(sorted_values: list[int], q: float) -> int:
    """Return nearest-rank quantile for a pre-sorted integer list."""
    if not sorted_values:
        return 0
    if q <= 0.0:
        return sorted_values[0]
    if q >= 1.0:
        return sorted_values[-1]
    idx = int(round((len(sorted_values) - 1) * q))
    return sorted_values[idx]


def collect_corpus_stats(input_dir: Path) -> dict[str, Any]:
    """Compute corpus statistics from expanded graph JSON files."""
    ship_count = 0
    total_parts = 0
    part_counts: list[int] = []
    ships_with_overclock = 0
    overclocked_parts = 0
    oc_fraction_per_ship: list[float] = []
    rotation_counts = [0, 0, 0, 0]

    door_edges = 0
    touching_edges = 0
    ships_with_zero_door = 0

    virtual_member_edges: dict[str, int] = {kind: 0 for kind in _VIRTUAL_MEMBER_EDGE_KINDS}
    ships_with_virtual_member_edge: dict[str, int] = {kind: 0 for kind in _VIRTUAL_MEMBER_EDGE_KINDS}

    for json_path in sorted(input_dir.iterdir()):
        if json_path.suffix != ".json" or json_path.name in _SKIP_FILENAMES:
            continue

        payload = orjson.loads(json_path.read_bytes())
        sg = payload.get("graphs", {}).get("A_structural_part_graph", {})
        xg = payload.get("graphs", {}).get("X_expansion_structural", {})
        nodes = sg.get("nodes", [])
        edges = sg.get("edges", [])
        cross_edges = xg.get("cross_edges", [])

        n_parts = len(nodes)
        if n_parts == 0:
            continue
        ship_count += 1
        total_parts += n_parts
        part_counts.append(n_parts)

        oc_parts_in_ship = 0
        for node in nodes:
            rotation = int(node.get("rotation", 0))
            if 0 <= rotation <= 3:
                rotation_counts[rotation] += 1
            if bool(node.get("overclocked", False)):
                oc_parts_in_ship += 1
        overclocked_parts += oc_parts_in_ship
        if oc_parts_in_ship > 0:
            ships_with_overclock += 1
        oc_fraction_per_ship.append(oc_parts_in_ship / n_parts)

        door_in_ship = 0
        touching_in_ship = 0
        for edge in edges:
            kind = edge.get("kind")
            if kind == "door":
                door_in_ship += 1
            elif kind == "touching":
                touching_in_ship += 1
        door_edges += door_in_ship
        touching_edges += touching_in_ship
        if door_in_ship == 0:
            ships_with_zero_door += 1

        per_ship_member_counts: dict[str, int] = {kind: 0 for kind in _VIRTUAL_MEMBER_EDGE_KINDS}
        for edge in cross_edges:
            kind = edge.get("kind")
            if kind in per_ship_member_counts:
                per_ship_member_counts[kind] += 1
        for kind, count in per_ship_member_counts.items():
            virtual_member_edges[kind] += count
            if count > 0:
                ships_with_virtual_member_edge[kind] += 1

    if ship_count == 0:
        raise ValueError(f"No graph JSON files found in {input_dir}")

    part_counts.sort()
    oc_fraction_per_ship.sort()

    mean_parts = total_parts / ship_count
    median_parts = _quantile(part_counts, 0.5)
    p5_parts = _quantile(part_counts, 0.05)
    p95_parts = _quantile(part_counts, 0.95)
    median_oc_fraction = oc_fraction_per_ship[int(round((len(oc_fraction_per_ship) - 1) * 0.5))]
    global_oc_rate = overclocked_parts / max(1, total_parts)
    ships_oc_rate = ships_with_overclock / ship_count

    rotation_total = sum(rotation_counts)
    rotation_frequencies = [count / max(1, rotation_total) for count in rotation_counts]
    inv = [1.0 / max(freq, 1e-12) for freq in rotation_frequencies]
    inv_mean = sum(inv) / len(inv)
    rotation_weights = [val / inv_mean for val in inv]

    virtual_density = {
        kind: {
            "edges_per_part": virtual_member_edges[kind] / max(1, total_parts),
            "ships_with_edges_fraction": ships_with_virtual_member_edge[kind] / ship_count,
        }
        for kind in _VIRTUAL_MEMBER_EDGE_KINDS
    }

    return {
        "ship_count": ship_count,
        "total_parts": total_parts,
        "ship_size": {
            "mean_parts_per_ship": mean_parts,
            "median_parts_per_ship": median_parts,
            "p5_parts_per_ship": p5_parts,
            "p95_parts_per_ship": p95_parts,
        },
        "overclock": {
            "global_part_rate": global_oc_rate,
            "ships_with_overclock_fraction": ships_oc_rate,
            "median_per_ship_fraction": median_oc_fraction,
            "parts_masked_per_ship_at_rate_0_03": median_parts * 0.03,
            "expected_positive_masks_per_median_ship_at_rate_0_03": median_parts * 0.03 * global_oc_rate,
            "recommended_pos_weight_ratio": (1.0 - global_oc_rate) / max(1e-12, global_oc_rate),
        },
        "rotation": {
            "counts": {
                "r0": rotation_counts[0],
                "r1": rotation_counts[1],
                "r2": rotation_counts[2],
                "r3": rotation_counts[3],
            },
            "frequencies": {
                "r0": rotation_frequencies[0],
                "r1": rotation_frequencies[1],
                "r2": rotation_frequencies[2],
                "r3": rotation_frequencies[3],
            },
            "normalized_inverse_frequency_weights": {
                "r0": rotation_weights[0],
                "r1": rotation_weights[1],
                "r2": rotation_weights[2],
                "r3": rotation_weights[3],
            },
            "majority_class_baseline_acc": max(rotation_frequencies),
        },
        "edge_density": {
            "door_edges_per_part": door_edges / max(1, total_parts),
            "touching_edges_per_part": touching_edges / max(1, total_parts),
            "ships_with_zero_door_fraction": ships_with_zero_door / ship_count,
        },
        "virtual_membership_density": virtual_density,
    }


def rotation_class_weights_from_stats(stats_payload: dict[str, Any]) -> torch.Tensor:
    """Return rotation class weights tensor [4] from a stats payload."""
    weights = stats_payload["rotation"]["normalized_inverse_frequency_weights"]
    return torch.tensor(
        [
            float(weights["r0"]),
            float(weights["r1"]),
            float(weights["r2"]),
            float(weights["r3"]),
        ],
        dtype=torch.float,
    )
