"""Regression tests for preprocessing structural-edge and door-edge rules."""

from collections import defaultdict

from common.geometry import infer_meta, load_vanilla_part_geometry
from preprocessing.graphs import part_cells, structural_door_edges, structural_edges


def _build_cell_to_parts(part_records: list[dict]) -> dict:
    """Derive cell_to_parts from part_records using the real geometry, mirroring
    what process_ship() does so tests stay consistent with footprint changes."""

    cell_to_parts: dict = defaultdict(set)
    for record in part_records:
        meta, _ = infer_meta(record["part_id"], record["rotation"])
        part_dict = {"Location": record["location"], "Rotation": record["rotation"]}
        for cell in part_cells(part_dict, meta):
            cell_to_parts[cell].add(record["index"])
    return dict(cell_to_parts)


def test_door_edge_connects_two_distinct_parts() -> None:
    """A door whose cells are owned by two different parts becomes a door edge
    with source_cell_2x and target_cell_2x matching each part's occupied cell."""

    # Orientation 0 joins (x, y-1) <-> (x, y). Cell=[0,1] → cells (0,0) and (0,1).
    # Cell2x=[-2, 0] is the local 2x of the stored cell (0,1) given center_2x=[2,2].
    # prev cell (0,0) in 2x = [-2, 0] - (0,2) = [-2, -2].
    # Part 0 owns (0,0) → source_cell_2x = [-2,-2]; part 1 owns (0,1) → target_cell_2x = [-2,0].
    cell_to_parts = {
        (0, 0): {0},
        (0, 1): {1},
    }
    doors = [{"Cell": [0, 1], "Cell2x": [-2, 0], "Orientation": 0}]

    edges, stats = structural_door_edges(
        [
            {
                "index": 0,
                "part_id": "cosmoteer.corridor",
                "location": [0, 0],
                "rotation": 0,
            },
            {
                "index": 1,
                "part_id": "cosmoteer.corridor",
                "location": [0, 1],
                "rotation": 0,
            },
        ],
        doors,
        cell_to_parts,
        load_vanilla_part_geometry(),
    )

    assert len(edges) == 1
    edge = edges[0]
    assert edge["source"] == 0
    assert edge["target"] == 1
    assert edge["kind"] == "door"
    assert edge["door_index"] == 0
    assert edge["orientation"] == 0
    assert edge["source_cell_2x"] == [-2, -2]
    assert edge["target_cell_2x"] == [-2, 0]
    assert stats["door_edges"] == 1
    assert stats["dangling_door_records"] == 0
    assert stats["internal_door_records"] == 0
    assert stats["non_structural_door_records"] == 0


def test_door_edge_is_skipped_when_cell_is_not_in_ship() -> None:
    """Doors whose cells cannot be resolved to occupied ship cells are dangling."""

    # Only one of the two door-adjacent cells is occupied.
    cell_to_parts = {
        (0, 1): {1},
    }
    doors = [{"Cell": [0, 1], "Cell2x": [-2, 0], "Orientation": 0}]

    edges, stats = structural_door_edges([], doors, cell_to_parts, load_vanilla_part_geometry())

    assert edges == []
    assert stats["door_edges"] == 0
    assert stats["dangling_door_records"] == 1
    assert stats["internal_door_records"] == 0
    assert stats["non_structural_door_records"] == 0


def test_door_edge_is_skipped_when_both_cells_belong_to_same_part() -> None:
    """Doors connecting two cells of the same part are internal and produce no edge."""

    cell_to_parts = {
        (0, 0): {0},
        (0, 1): {0},
    }
    doors = [{"Cell": [0, 1], "Cell2x": [-2, 0], "Orientation": 0}]

    edges, stats = structural_door_edges([], doors, cell_to_parts, load_vanilla_part_geometry())

    assert edges == []
    assert stats["door_edges"] == 0
    assert stats["dangling_door_records"] == 0
    assert stats["internal_door_records"] == 1
    assert stats["non_structural_door_records"] == 0


def test_door_edge_is_skipped_without_shared_structural_wall() -> None:
    """Door records must not create structural edges across non-attachable contacts."""

    part_records = [
        {
            "index": 0,
            "part_id": "cosmoteer.armor_wedge",
            "location": [0, 0],
            "rotation": 0,
        },
        {
            "index": 1,
            "part_id": "cosmoteer.corridor",
            "location": [0, -1],
            "rotation": 0,
        },
    ]
    cell_to_parts = {
        (0, -1): {1},
        (0, 0): {0},
    }
    doors = [{"Cell": [0, 0], "Cell2x": [0, 0], "Orientation": 0}]

    edges, stats = structural_door_edges(
        part_records,
        doors,
        cell_to_parts,
        load_vanilla_part_geometry(),
    )

    assert edges == []
    assert stats["door_edges"] == 0
    assert stats["dangling_door_records"] == 0
    assert stats["internal_door_records"] == 0
    assert stats["non_structural_door_records"] == 1


def test_structural_edges_filter_out_wedge_air_contacts() -> None:
    """Structural edge generation should ignore wedge empty-space contacts."""

    geometry_cache = load_vanilla_part_geometry()
    part_records = [
        {
            "index": 0,
            "part_id": "cosmoteer.armor_wedge",
            "location": [0, 0],
            "rotation": 0,
        },
        {
            "index": 1,
            "part_id": "cosmoteer.armor",
            "location": [0, -1],
            "rotation": 0,
        },
    ]
    # Derive cell_to_parts from real geometry so the map stays consistent with
    # footprint changes.  The armor directly above the wedge's air side should
    # produce no structural edge.
    cell_to_parts = _build_cell_to_parts(part_records)

    assert structural_edges(part_records, cell_to_parts, geometry_cache) == []


def test_structural_edges_accept_r_wedge_alias_ids() -> None:
    """Structural edge generation should support mirrored `_R` wedge aliases."""

    geometry_cache = load_vanilla_part_geometry()
    part_records = [
        {
            "index": 0,
            "part_id": "cosmoteer.armor_1x2_wedge_R",
            "location": [0, 0],
            "rotation": 0,
        },
        {
            "index": 1,
            "part_id": "cosmoteer.armor",
            "location": [1, 0],
            "rotation": 0,
        },
    ]
    # Derive cell_to_parts from real geometry so the test stays consistent with
    # footprint changes and does not rely on a manually guessed cell layout.
    cell_to_parts = _build_cell_to_parts(part_records)

    edges = structural_edges(part_records, cell_to_parts, geometry_cache)
    assert len(edges) == 1
    assert edges[0]["source"] == 0
    assert edges[0]["target"] == 1
