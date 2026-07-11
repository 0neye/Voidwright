from __future__ import annotations

import orjson

from training.backends.hgt.stats import collect_corpus_stats, rotation_class_weights_from_stats


def _write_graph(path, *, nodes, edges, cross_edges) -> None:
    payload = {
        "graphs": {
            "A_structural_part_graph": {
                "nodes": nodes,
                "edges": edges,
            },
            "X_expansion_structural": {
                "nodes": [],
                "cross_edges": cross_edges,
            },
        }
    }
    path.write_bytes(orjson.dumps(payload))


def test_collect_corpus_stats_and_rotation_weights(tmp_path) -> None:
    _write_graph(
        tmp_path / "ship_a.json",
        nodes=[
            {"rotation": 0, "overclocked": False},
            {"rotation": 0, "overclocked": True},
        ],
        edges=[
            {"kind": "door"},
            {"kind": "touching"},
            {"kind": "touching"},
        ],
        cross_edges=[
            {"kind": "zone_member"},
            {"kind": "zone_member_rotated"},
            {"kind": "super_member"},
        ],
    )
    _write_graph(
        tmp_path / "ship_b.json",
        nodes=[
            {"rotation": 1, "overclocked": False},
            {"rotation": 2, "overclocked": False},
            {"rotation": 3, "overclocked": False},
            {"rotation": 0, "overclocked": False},
        ],
        edges=[
            {"kind": "touching"},
        ],
        cross_edges=[
            {"kind": "weapon_member"},
            {"kind": "thermal_member"},
        ],
    )

    stats = collect_corpus_stats(tmp_path)
    assert stats["ship_count"] == 2
    assert stats["total_parts"] == 6
    assert stats["rotation"]["counts"] == {"r0": 3, "r1": 1, "r2": 1, "r3": 1}
    assert stats["overclock"]["global_part_rate"] == 1 / 6
    assert stats["edge_density"]["door_edges_per_part"] == 1 / 6
    assert stats["edge_density"]["touching_edges_per_part"] == 3 / 6
    assert stats["virtual_membership_density"]["zone_member"]["ships_with_edges_fraction"] == 0.5
    assert stats["virtual_membership_density"]["thermal_member"]["ships_with_edges_fraction"] == 0.5

    weights = rotation_class_weights_from_stats(stats).tolist()
    assert len(weights) == 4
    assert weights[0] < weights[1]
    assert weights[0] < weights[2]
    assert weights[0] < weights[3]
