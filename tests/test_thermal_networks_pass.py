"""Focused tests for ThermalNetworksPass graph expansion pass.

These tests use unittest.mock.patch to inject synthetic thermal port geometry
into the vanilla part geometry cache, since the ``thermal_ports`` attribute on
``RotationGeometry`` is being added by a parallel workstream and may not yet
be present in the live geometry database.

Engine-room special-case tests use real part IDs (``cosmoteer.engine_room`` /
``cosmoteer.thruster_*``) and include a ``footprint`` field on each node so that
``_build_engine_room_thruster_edges`` can compute 2x-space adjacency.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from common.heat_exchanger import (
    HEAT_EXCHANGER_ABSORPTION_RADIUS_TILES,
    footprint_tile_origins_2x,
    heat_exchanger_radius_region_tile_origins_2x,
)
from graph_expansion.context import EXPANSION_GRAPH_NAME, STRUCTURAL_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base_indexes import BaseIndexesPass
from graph_expansion.passes.thermal_networks import ThermalNetworksPass

__all__: list[str] = []

_STRUCTURAL_GRAPH_NAME = STRUCTURAL_GRAPH_NAME
_EXPANSION_GRAPH_NAME = EXPANSION_GRAPH_NAME

# Stable part IDs used across tests.
_HEAT_PIPE_ID = "cosmoteer.heat_pipe"
_RADIATOR_ID = "cosmoteer.radiator"
_HEAT_EXCHANGER_ID = "cosmoteer.heat_exchanger"
_OVERCLOCK_PART_ID = "cosmoteer.ion_beam_emitter"
_ENGINE_ROOM_ID = "cosmoteer.engine_room"
_THRUSTER_SMALL_ID = "cosmoteer.thruster_small"
_ARMOR_ID = "cosmoteer.armor_1x1"
_RAILGUN_LAUNCHER_ID = "cosmoteer.railgun_launcher"
_RAILGUN_ACCELERATOR_ID = "cosmoteer.railgun_accelerator"
_RAILGUN_LOADER_ID = "cosmoteer.railgun_loader"
_RESONANCE_BEAM_ID = "cosmoteer.resonance_beam_turret"
_POWER_STORAGE_ID = "cosmoteer.power_storage"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_thermal_port(
    location: tuple[int, int],
    direction: str,
    *,
    overclock_conditional: bool = False,
) -> SimpleNamespace:
    """Build a minimal thermal port descriptor matching the expected API."""

    return SimpleNamespace(
        location=location,
        direction=direction,
        overclock_conditional=overclock_conditional,
    )


def _make_rotation_geometry(thermal_ports: tuple[SimpleNamespace, ...]) -> SimpleNamespace:
    """Build a minimal RotationGeometry-like object with thermal_ports."""

    return SimpleNamespace(thermal_ports=thermal_ports)


def _make_vanilla_geo(thermal_ports: tuple[SimpleNamespace, ...]) -> SimpleNamespace:
    """Build a minimal VanillaPartGeometry-like object with one rotation."""

    rot_geo = _make_rotation_geometry(thermal_ports)

    def rotation_geometry(rotation: int) -> SimpleNamespace:
        return rot_geo

    return SimpleNamespace(rotation_geometry=rotation_geometry)


def make_node(
    node_id: int,
    location_2x: list[int],
    part_id: str,
    *,
    rotation: int = 0,
    overclocked: bool = False,
    footprint: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a minimal structural node dict with location_2x and overclocked fields.

    Args:
        node_id: Integer node ID.
        location_2x: ``[x, y]`` position in the centered-2x frame.
        part_id: Part identifier string.
        rotation: Part rotation (0–3, default 0).
        overclocked: Whether the part is overclocked (default False).
        footprint: Optional ``{"width": w, "height": h}`` in tile units.  Required
            for engine-room proximity edge detection; omit for port-matching-only tests.
    """

    node: dict[str, Any] = {
        "id": node_id,
        "part_id": part_id,
        "location_2x": location_2x,
        "rotation": rotation,
        "overclocked": overclocked,
    }
    if footprint is not None:
        node["footprint"] = footprint
    return node


def make_graph_payload(nodes: list[dict[str, Any]], edges: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build a minimal graph JSON payload for use with ExpansionContext."""

    return {
        "graphs": {
            _STRUCTURAL_GRAPH_NAME: {
                "nodes": nodes,
                "edges": edges if edges is not None else [],
            }
        }
    }


def _run_thermal_pass(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]] | None = None,
    *,
    fake_geometry: dict[str, SimpleNamespace] | None = None,
) -> tuple[ExpansionContext, dict[str, Any]]:
    """Run BaseIndexesPass then ThermalNetworksPass and return the context and summary.

    Args:
        nodes: Structural nodes for the test graph.
        edges: Optional structural edges (defaults to empty list).
        fake_geometry: Optional mapping of part_id → fake VanillaPartGeometry.
            When provided, ``load_vanilla_part_geometry`` is patched to return it.

    Returns:
        A tuple of ``(context, summary)`` after both passes have run.
    """

    payload = make_graph_payload(nodes, edges)
    context = ExpansionContext(payload, expansion_name="structural", expansion_version=1)
    BaseIndexesPass().run(context)

    if fake_geometry is not None:
        with patch(
            "graph_expansion.passes.thermal_networks.load_vanilla_part_geometry",
            return_value=fake_geometry,
        ):
            summary = ThermalNetworksPass().run(context)
    else:
        summary = ThermalNetworksPass().run(context)

    return context, summary


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_thermal_networks_two_adjacent_heat_pipes_form_one_network() -> None:
    """Two heat pipes with complementary adjacent ports should form one thermal network of 2."""

    # Part 0 at location_2x=[0, 0]: port facing Right at tile (0, 0) → 2x pos (0, 0, Right)
    # Part 1 at location_2x=[2, 0]: port facing Left  at tile (0, 0) → 2x pos (2, 0, Left)
    # Right port at (0,0) + delta(2,0) = (2, 0) matches Left port at (2, 0) ✓
    # Each part has only its own facing port, forming a complementary pair.
    nodes = [
        make_node(0, [0, 0], _HEAT_PIPE_ID),
        make_node(1, [2, 0], _HEAT_PIPE_ID + "_receiver"),
    ]
    fake_geo = {
        _HEAT_PIPE_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Right"),)
        ),
        _HEAT_PIPE_ID + "_receiver": _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Left"),)
        ),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["networks"] == 1
    assert summary["thermal_edges"] == 1
    assert summary["network_sizes"] == [2]
    assert summary["parts_with_ports"] == 2

    clusters = context.get_annotation("thermal_networks")
    assert clusters == [[0, 1]]

    network_by_part_id = context.get_annotation("thermal_network_by_part_id")
    assert network_by_part_id[0] == "thermal_network_0"
    assert network_by_part_id[1] == "thermal_network_0"

    expansion_graph = context.emitted_graphs[_EXPANSION_GRAPH_NAME]
    node_ids = [n["id"] for n in expansion_graph["nodes"]]
    assert "thermal_network_0" in node_ids

    thermal_node = next(n for n in expansion_graph["nodes"] if n["id"] == "thermal_network_0")
    assert thermal_node["kind"] == "thermal_network"
    assert thermal_node["member_count"] == 2

    cross_edge_targets = {
        (e["kind"], e["target"])
        for e in expansion_graph["cross_edges"]
        if e.get("source") == "thermal_network_0"
    }
    assert cross_edge_targets == {("thermal_member", 0), ("thermal_member", 1)}


def test_thermal_networks_radiator_adjacent_to_heat_pipe_connects() -> None:
    """A radiator adjacent to a heat pipe on a matching side should connect."""

    # Radiator at [0, 0] has a Down port at tile offset (0, 0) → 2x pos (0, 0)
    # Heat pipe at [0, 2] has an Up port at tile offset (0, 0) → 2x pos (0, 2)
    # Down port at (0, 0) + delta(0, 2) = (0, 2) matches Up port at (0, 2) ✓
    nodes = [
        make_node(0, [0, 0], _RADIATOR_ID),
        make_node(1, [0, 2], _HEAT_PIPE_ID),
    ]
    fake_geo = {
        _RADIATOR_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Down"),)),
        _HEAT_PIPE_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Up"),)),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["networks"] == 1
    assert summary["thermal_edges"] == 1
    assert summary["network_sizes"] == [2]

    clusters = context.get_annotation("thermal_networks")
    assert clusters == [[0, 1]]


def test_thermal_networks_overclock_conditional_port_inactive_when_not_overclocked() -> None:
    """An overclock-conditional port on a non-overclocked part must not form a thermal edge."""

    # Part 0 has a normal port facing Right → active regardless of overclocked
    # Part 1 has an overclock_conditional port facing Left, but overclocked=False → inactive
    nodes = [
        make_node(0, [0, 0], _HEAT_PIPE_ID, overclocked=False),
        make_node(1, [2, 0], _OVERCLOCK_PART_ID, overclocked=False),
    ]
    # Part 0: normal Right port at (0, 0)
    # Part 1: overclock-conditional Left port at (0, 0) in local tile → 2x (2, 0)
    fake_geo = {
        _HEAT_PIPE_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Right", overclock_conditional=False),)
        ),
        _OVERCLOCK_PART_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Left", overclock_conditional=True),)
        ),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    # The overclock-conditional port on part 1 is inactive → no thermal edge
    assert summary["thermal_edges"] == 0
    assert summary["networks"] == 0
    assert summary["network_sizes"] == []

    clusters = context.get_annotation("thermal_networks")
    assert clusters == []

    # Part 0 still has an active port, but part 1's port is inactive
    assert summary["parts_with_ports"] == 1


def test_thermal_networks_overclock_conditional_port_active_when_overclocked() -> None:
    """An overclock-conditional port on an overclocked part SHOULD form a thermal edge."""

    # Part 0 at [0, 0]: normal Right port → active
    # Part 1 at [2, 0]: overclock-conditional Left port, overclocked=True → active
    nodes = [
        make_node(0, [0, 0], _HEAT_PIPE_ID, overclocked=False),
        make_node(1, [2, 0], _OVERCLOCK_PART_ID, overclocked=True),
    ]
    fake_geo = {
        _HEAT_PIPE_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Right", overclock_conditional=False),)
        ),
        _OVERCLOCK_PART_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Left", overclock_conditional=True),)
        ),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 1
    assert summary["networks"] == 1
    assert summary["network_sizes"] == [2]

    clusters = context.get_annotation("thermal_networks")
    assert clusters == [[0, 1]]

    expansion_graph = context.emitted_graphs[_EXPANSION_GRAPH_NAME]
    assert any(n["id"] == "thermal_network_0" for n in expansion_graph["nodes"])
    cross_edge_targets = {
        (e["kind"], e["target"])
        for e in expansion_graph["cross_edges"]
        if e.get("source") == "thermal_network_0"
    }
    assert cross_edge_targets == {("thermal_member", 0), ("thermal_member", 1)}


def test_thermal_networks_no_ports_yields_empty_result() -> None:
    """Parts with no thermal ports in the geometry database produce no networks."""

    nodes = [
        make_node(0, [0, 0], "cosmoteer.armor_1x1"),
        make_node(1, [2, 0], "cosmoteer.armor_1x1"),
    ]
    # Empty geometry → no thermal ports
    fake_geo: dict[str, Any] = {}

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 0
    assert summary["networks"] == 0
    assert summary["parts_with_ports"] == 0

    assert context.get_annotation("thermal_networks") == []
    assert context.get_annotation("thermal_network_by_part_id") == {}

    assert _EXPANSION_GRAPH_NAME not in context.emitted_graphs


def test_thermal_networks_non_adjacent_ports_do_not_connect() -> None:
    """Ports that face the right direction but are not adjacent must not connect."""

    # Part 0 at [0, 0] facing Right → 2x port at (0, 0)
    # Part 1 at [4, 0] facing Left  → 2x port at (4, 0)
    # Gap of 4 units (not the expected 2 units for one tile) → no connection
    nodes = [
        make_node(0, [0, 0], _HEAT_PIPE_ID),
        make_node(1, [4, 0], _HEAT_PIPE_ID),
    ]
    fake_geo = {
        _HEAT_PIPE_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Right"),)
        )
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 0
    assert summary["networks"] == 0


def test_thermal_networks_three_parts_chain_forms_one_network() -> None:
    """A chain of three thermally linked parts should form a single network."""

    # Part 0 ←Right→ Part 1 ←Right→ Part 2
    # 0 at [0,0], 1 at [2,0], 2 at [4,0]
    nodes = [
        make_node(0, [0, 0], _HEAT_PIPE_ID),
        make_node(1, [2, 0], _HEAT_PIPE_ID),
        make_node(2, [4, 0], _HEAT_PIPE_ID),
    ]
    # Each heat pipe has a Right port at (0,0) and a Left port at (0,0) in local tile,
    # mapping to 2x positions matching their location_2x.
    fake_geo = {
        _HEAT_PIPE_ID: _make_vanilla_geo(
            (
                _make_thermal_port((0, 0), "Right"),
                _make_thermal_port((0, 0), "Left"),
            )
        )
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["networks"] == 1
    assert summary["thermal_edges"] == 2
    assert summary["network_sizes"] == [3]

    clusters = context.get_annotation("thermal_networks")
    assert clusters == [[0, 1, 2]]


def test_thermal_networks_two_isolated_pairs_form_two_networks() -> None:
    """Two disconnected thermal pairs should each form their own network."""

    # Pair A: part 0 (Right) ↔ part 1 (Left) at [0,0] and [2,0]
    # Pair B: part 2 (Right) ↔ part 3 (Left) at [100,0] and [102,0] (far from pair A)
    _HEAT_PIPE_R = _HEAT_PIPE_ID + "_R"
    _HEAT_PIPE_L = _HEAT_PIPE_ID + "_L"
    nodes = [
        make_node(0, [0, 0], _HEAT_PIPE_R),
        make_node(1, [2, 0], _HEAT_PIPE_L),
        make_node(2, [100, 0], _RADIATOR_ID),
        make_node(3, [102, 0], _HEAT_PIPE_ID),
    ]
    fake_geo = {
        _HEAT_PIPE_R: _make_vanilla_geo((_make_thermal_port((0, 0), "Right"),)),
        _HEAT_PIPE_L: _make_vanilla_geo((_make_thermal_port((0, 0), "Left"),)),
        _RADIATOR_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Right"),)),
        _HEAT_PIPE_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Left"),)),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["networks"] == 2
    assert summary["thermal_edges"] == 2
    assert sorted(summary["network_sizes"]) == [2, 2]

    clusters = context.get_annotation("thermal_networks")
    assert len(clusters) == 2
    assert [0, 1] in clusters
    assert [2, 3] in clusters


def test_thermal_networks_summary_increments_in_expansion_graph() -> None:
    """The expansion graph summary should record thermal_network_nodes and thermal_member_edges."""

    nodes = [
        make_node(0, [0, 0], _HEAT_PIPE_ID),
        make_node(1, [2, 0], _RADIATOR_ID),
    ]
    fake_geo = {
        _HEAT_PIPE_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Right"),)),
        _RADIATOR_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Left"),)),
    }

    context, _ = _run_thermal_pass(nodes, fake_geometry=fake_geo)
    expansion_graph = context.emitted_graphs[_EXPANSION_GRAPH_NAME]

    assert expansion_graph["summary"]["thermal_network_nodes"] == 1
    assert expansion_graph["summary"]["thermal_member_edges"] == 2


# ---------------------------------------------------------------------------
# Heat-exchanger radius special-case tests
# ---------------------------------------------------------------------------


def test_connected_heat_exchanger_pulls_in_overclocked_part_within_radius() -> None:
    """A connected heat exchanger should include nearby overclocked parts."""

    nodes = [
        make_node(0, [0, 0], _HEAT_EXCHANGER_ID, overclocked=False, footprint={"width": 1, "height": 1}),
        make_node(1, [0, 2], _HEAT_PIPE_ID,      overclocked=False, footprint={"width": 1, "height": 1}),
        make_node(2, [8, 0], _ARMOR_ID,          overclocked=True,  footprint={"width": 1, "height": 1}),
    ]
    fake_geo = {
        _HEAT_EXCHANGER_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Down"),)),
        _HEAT_PIPE_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Up"),)),
        _ARMOR_ID: _make_vanilla_geo(()),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 1
    assert summary["heat_exchanger_radius_edges"] == 1
    assert summary["networks"] == 1
    assert summary["network_sizes"] == [3]
    assert context.get_annotation("thermal_networks") == [[0, 1, 2]]


def test_heat_exchanger_radius_mask_matches_fixed_101_cell_stencil() -> None:
    """The stencil should be exact and anchored on local 2x tile origins."""

    exchanger_tiles = footprint_tile_origins_2x(
        make_node(0, [0, 0], _HEAT_EXCHANGER_ID, footprint={"width": 1, "height": 1})
    )
    assert exchanger_tiles == {(0, 0)}

    region = heat_exchanger_radius_region_tile_origins_2x(
        (0, 0),
        HEAT_EXCHANGER_ABSORPTION_RADIUS_TILES,
    )
    assert len(region) == 101
    row_spans = {
        tile_y: (min(xs), max(xs), len(xs))
        for tile_y in sorted({tile_y for _, tile_y in region})
        for xs in [[tile_x for tile_x, y in region if y == tile_y]]
    }

    assert row_spans == {
        -10: (-4, 4, 5),
        -8: (-8, 8, 9),
        -6: (-8, 8, 9),
        -4: (-10, 10, 11),
        -2: (-10, 10, 11),
        0: (-10, 10, 11),
        2: (-10, 10, 11),
        4: (-10, 10, 11),
        6: (-8, 8, 9),
        8: (-8, 8, 9),
        10: (-4, 4, 5),
    }


def test_heat_exchanger_radius_mask_preserves_odd_local_anchor() -> None:
    """Odd local 2x anchors must keep the same odd parity throughout the stencil."""

    region = heat_exchanger_radius_region_tile_origins_2x(
        (-45, -49),
        HEAT_EXCHANGER_ABSORPTION_RADIUS_TILES,
    )

    assert len(region) == 101
    assert all((x % 2) == 1 for x, _ in region)
    assert all((y % 2) == 1 for _, y in region)
    assert (-45, -49) in region


def test_connected_heat_exchanger_includes_part_near_center_radius_boundary() -> None:
    """A part near the center-based radius boundary should be included."""

    nodes = [
        make_node(0, [0, 0], _HEAT_EXCHANGER_ID, overclocked=False, footprint={"width": 1, "height": 1}),
        make_node(1, [0, 2], _HEAT_PIPE_ID,      overclocked=False, footprint={"width": 1, "height": 1}),
        # 5 tiles right from exchanger origin in 2x-space.
        make_node(2, [10, 0], _ARMOR_ID,         overclocked=True,  footprint={"width": 1, "height": 1}),
    ]
    fake_geo = {
        _HEAT_EXCHANGER_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Down"),)),
        _HEAT_PIPE_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Up"),)),
        _ARMOR_ID: _make_vanilla_geo(()),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 1
    assert summary["heat_exchanger_radius_edges"] == 1
    assert summary["networks"] == 1
    assert summary["network_sizes"] == [3]
    assert context.get_annotation("thermal_networks") == [[0, 1, 2]]


def test_connected_heat_exchanger_does_not_pull_in_overclocked_part_outside_radius() -> None:
    """Overclocked parts outside the heat exchanger radius must stay excluded."""

    nodes = [
        make_node(0, [0, 0], _HEAT_EXCHANGER_ID, overclocked=False, footprint={"width": 1, "height": 1}),
        make_node(1, [0, 2], _HEAT_PIPE_ID,      overclocked=False, footprint={"width": 1, "height": 1}),
        # 6 tiles right from exchanger origin in 2x-space.
        make_node(2, [12, 0], _ARMOR_ID,         overclocked=True,  footprint={"width": 1, "height": 1}),
    ]
    fake_geo = {
        _HEAT_EXCHANGER_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Down"),)),
        _HEAT_PIPE_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Up"),)),
        _ARMOR_ID: _make_vanilla_geo(()),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 1
    assert summary["heat_exchanger_radius_edges"] == 0
    assert summary["networks"] == 1
    assert summary["network_sizes"] == [2]
    assert context.get_annotation("thermal_networks") == [[0, 1]]


def test_connected_heat_exchanger_excludes_far_diagonal_outside_radius() -> None:
    """Diagonal offset (5,5) should remain outside center-based radius 5."""

    nodes = [
        make_node(0, [0, 0], _HEAT_EXCHANGER_ID, overclocked=False, footprint={"width": 1, "height": 1}),
        make_node(1, [0, 2], _HEAT_PIPE_ID,      overclocked=False, footprint={"width": 1, "height": 1}),
        # 5 tiles right + 5 tiles down from exchanger origin in 2x-space.
        # This remains outside center-based radius 5.
        make_node(2, [10, 10], _ARMOR_ID,        overclocked=True,  footprint={"width": 1, "height": 1}),
    ]
    fake_geo = {
        _HEAT_EXCHANGER_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Down"),)),
        _HEAT_PIPE_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Up"),)),
        _ARMOR_ID: _make_vanilla_geo(()),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 1
    assert summary["heat_exchanger_radius_edges"] == 0
    assert summary["networks"] == 1
    assert summary["network_sizes"] == [2]
    assert context.get_annotation("thermal_networks") == [[0, 1]]


def test_connected_heat_exchanger_includes_3_4_offset_within_radius() -> None:
    """Offset (3,4) should be included by center-based radius 5."""

    nodes = [
        make_node(0, [0, 0], _HEAT_EXCHANGER_ID, overclocked=False, footprint={"width": 1, "height": 1}),
        make_node(1, [0, 2], _HEAT_PIPE_ID,      overclocked=False, footprint={"width": 1, "height": 1}),
        # 3 tiles right + 4 tiles down in 2x-space.
        make_node(2, [6, 8], _ARMOR_ID,          overclocked=True,  footprint={"width": 1, "height": 1}),
    ]
    fake_geo = {
        _HEAT_EXCHANGER_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Down"),)),
        _HEAT_PIPE_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Up"),)),
        _ARMOR_ID: _make_vanilla_geo(()),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 1
    assert summary["heat_exchanger_radius_edges"] == 1
    assert summary["networks"] == 1
    assert summary["network_sizes"] == [3]
    assert context.get_annotation("thermal_networks") == [[0, 1, 2]]


def test_connected_heat_exchanger_ignores_non_overclocked_parts_within_radius() -> None:
    """Radius inclusion applies only to overclocked parts."""

    nodes = [
        make_node(0, [0, 0], _HEAT_EXCHANGER_ID, overclocked=False, footprint={"width": 1, "height": 1}),
        make_node(1, [0, 2], _HEAT_PIPE_ID,      overclocked=False, footprint={"width": 1, "height": 1}),
        make_node(2, [8, 0], _ARMOR_ID,          overclocked=False, footprint={"width": 1, "height": 1}),
    ]
    fake_geo = {
        _HEAT_EXCHANGER_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Down"),)),
        _HEAT_PIPE_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Up"),)),
        _ARMOR_ID: _make_vanilla_geo(()),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 1
    assert summary["heat_exchanger_radius_edges"] == 0
    assert summary["networks"] == 1
    assert summary["network_sizes"] == [2]
    assert context.get_annotation("thermal_networks") == [[0, 1]]


def test_disconnected_heat_exchanger_does_not_create_radius_only_network() -> None:
    """A heat exchanger must already be connected to a thermal network to expand by radius."""

    nodes = [
        make_node(0, [0, 0], _HEAT_EXCHANGER_ID, overclocked=False, footprint={"width": 1, "height": 1}),
        make_node(1, [8, 0], _ARMOR_ID,          overclocked=True,  footprint={"width": 1, "height": 1}),
    ]
    fake_geo = {
        _HEAT_EXCHANGER_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Down"),)),
        _ARMOR_ID: _make_vanilla_geo(()),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 0
    assert summary["heat_exchanger_radius_edges"] == 0
    assert summary["networks"] == 0
    assert context.get_annotation("thermal_networks") == []


# ---------------------------------------------------------------------------
# Engine-room special-case tests
# ---------------------------------------------------------------------------
# These tests use real part IDs so that _build_engine_room_thruster_edges picks
# them up by the _ENGINE_ROOM_PART_ID / _THRUSTER_PART_ID_SUBSTRING constants.
# Nodes carry a ``footprint`` field for 2x-cell adjacency computation.
# The geometry patch is still applied so that no real thermal ports are returned
# for these parts (keeping the tests focused on the proximity-edge path).


def test_overclocked_engine_room_adjacent_thruster_forms_network() -> None:
    """An overclocked engine room touching a thruster creates an implicit thermal network."""

    # Engine room (3x3) at location_2x=[0, 0]: occupies 2x cells (0,0)..(4,4).
    # Thruster (1x1) at location_2x=[6, 0]: cell (6, 0) is tile-adjacent to (4, 0) ✓.
    nodes = [
        make_node(0, [0, 0], _ENGINE_ROOM_ID, overclocked=True,  footprint={"width": 3, "height": 3}),
        make_node(1, [6, 0], _THRUSTER_SMALL_ID, overclocked=True, footprint={"width": 1, "height": 1}),
    ]
    # No thermal ports → proximity edge is the only mechanism.
    fake_geo: dict[str, Any] = {}

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["engine_room_thruster_edges"] == 1
    assert summary["networks"] == 1
    assert summary["network_sizes"] == [2]

    clusters = context.get_annotation("thermal_networks")
    assert clusters == [[0, 1]]

    network_by_part_id = context.get_annotation("thermal_network_by_part_id")
    assert network_by_part_id[0] == "thermal_network_0"
    assert network_by_part_id[1] == "thermal_network_0"


def test_non_overclocked_engine_room_does_not_create_proximity_edge() -> None:
    """A non-overclocked engine room must not create implicit edges to adjacent thrusters."""

    nodes = [
        make_node(0, [0, 0], _ENGINE_ROOM_ID, overclocked=False, footprint={"width": 3, "height": 3}),
        make_node(1, [6, 0], _THRUSTER_SMALL_ID, overclocked=False, footprint={"width": 1, "height": 1}),
    ]
    fake_geo: dict[str, Any] = {}

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["engine_room_thruster_edges"] == 0
    assert summary["networks"] == 0
    assert context.get_annotation("thermal_networks") == []


def test_overclocked_engine_room_non_thruster_neighbour_no_proximity_edge() -> None:
    """An overclocked engine room must not create implicit edges to non-thruster neighbours."""

    nodes = [
        make_node(0, [0, 0], _ENGINE_ROOM_ID, overclocked=True, footprint={"width": 3, "height": 3}),
        make_node(1, [6, 0], _ARMOR_ID,        overclocked=False, footprint={"width": 1, "height": 1}),
    ]
    fake_geo: dict[str, Any] = {}

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["engine_room_thruster_edges"] == 0
    assert summary["networks"] == 0


def test_overclocked_engine_room_pulls_thruster_into_heat_pipe_network() -> None:
    """A thruster touching an overclocked engine room joins the ER's heat-pipe network."""

    # Heat pipe at [0, 0] ↔ engine room at [2, 0] via port match.
    # Thruster at [8, 0]: tile-adjacent to the engine room's right edge (cell x=6).
    # Expected: all three in one network.
    #
    # Engine room (3x3) at [2, 0]: 2x cells (2,0)..(6,4).  Right edge cells: (6,0),(6,2),(6,4).
    # Thruster (1x1) at [8, 0]: cell (8, 0). Delta from (6,0) → (8,0) is (+2,0) ✓.
    nodes = [
        make_node(0, [0, 0], _HEAT_PIPE_ID,      overclocked=False, footprint={"width": 1, "height": 1}),
        make_node(1, [2, 0], _ENGINE_ROOM_ID,     overclocked=True,  footprint={"width": 3, "height": 3}),
        make_node(2, [8, 0], _THRUSTER_SMALL_ID,  overclocked=True,  footprint={"width": 1, "height": 1}),
    ]
    # Heat pipe has a Right port at tile (0,0) → 2x (0,0).
    # Engine room has a Left port at tile (0,0) → 2x (2,0).  Right port at (2,0) → 2x (6,0).
    # But thruster has no port facing Left at (8,0), so no port match for ER↔thruster.
    # Only the heat pipe ↔ engine room port match fires; thruster joins via proximity edge.
    fake_geo = {
        _HEAT_PIPE_ID:    _make_vanilla_geo((_make_thermal_port((0, 0), "Right"),)),
        _ENGINE_ROOM_ID:  _make_vanilla_geo((_make_thermal_port((0, 0), "Left"),)),
        _THRUSTER_SMALL_ID: _make_vanilla_geo(()),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 1           # heat pipe ↔ engine room
    assert summary["engine_room_thruster_edges"] == 1  # engine room → thruster
    assert summary["networks"] == 1
    assert summary["network_sizes"] == [3]

    clusters = context.get_annotation("thermal_networks")
    assert clusters == [[0, 1, 2]]


def test_overclocked_engine_room_not_adjacent_no_proximity_edge() -> None:
    """A thruster not tile-adjacent to an overclocked engine room must not get a proximity edge."""

    # Engine room (3x3) at [0, 0]: right edge at x=4 (2x-space).
    # Thruster at [8, 0]: leftmost cell at x=8.  Gap of 4 units, not 2 → not adjacent.
    nodes = [
        make_node(0, [0, 0], _ENGINE_ROOM_ID,    overclocked=True, footprint={"width": 3, "height": 3}),
        make_node(1, [8, 0], _THRUSTER_SMALL_ID, overclocked=True, footprint={"width": 1, "height": 1}),
    ]
    fake_geo: dict[str, Any] = {}

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["engine_room_thruster_edges"] == 0
    assert summary["networks"] == 0


# ---------------------------------------------------------------------------
# Overclocked heat-pipe restriction tests
# ---------------------------------------------------------------------------


def test_two_adjacent_overclocked_parts_do_not_connect_via_ports() -> None:
    """Two overclocked non-exempt parts with matching ports must NOT form an edge."""

    # Part 0 (overclocked ion_beam_emitter) at [0, 0]: Right port.
    # Part 1 (overclocked ion_beam_emitter) at [2, 0]: Left port.
    # Both overclocked, neither is engine room or railgun → edge suppressed.
    nodes = [
        make_node(0, [0, 0], _OVERCLOCK_PART_ID, overclocked=True),
        make_node(1, [2, 0], _OVERCLOCK_PART_ID, overclocked=True),
    ]
    fake_geo = {
        _OVERCLOCK_PART_ID: _make_vanilla_geo(
            (
                _make_thermal_port((0, 0), "Right"),
                _make_thermal_port((0, 0), "Left"),
            )
        )
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 0
    assert summary["networks"] == 0
    assert context.get_annotation("thermal_networks") == []


def test_overclocked_part_connects_to_non_overclocked_heat_pipe() -> None:
    """An overclocked part next to a non-overclocked heat pipe must still connect."""

    # Heat pipe (not overclocked) at [0, 0]: Right port.
    # Overclocked part at [2, 0]: Left port (overclock_conditional).
    # Only one side is overclocked → edge is allowed.
    nodes = [
        make_node(0, [0, 0], _HEAT_PIPE_ID, overclocked=False),
        make_node(1, [2, 0], _OVERCLOCK_PART_ID, overclocked=True),
    ]
    fake_geo = {
        _HEAT_PIPE_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Right"),)),
        _OVERCLOCK_PART_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Left", overclock_conditional=True),)
        ),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 1
    assert summary["networks"] == 1
    assert context.get_annotation("thermal_networks") == [[0, 1]]


def test_overclocked_parts_do_not_chain_through_each_other() -> None:
    """Overclocked part A must not join B's heat-pipe network via overclocked intermediary B."""

    # Heat pipe at [0, 0] — Right port.
    # Overclocked part 1 at [2, 0] — Left and Right ports.
    # Overclocked part 2 at [4, 0] — Left port.
    # heat_pipe ↔ part1 is allowed (one side not overclocked).
    # part1 ↔ part2 is suppressed (both overclocked, not exempt).
    # Result: two separate networks — {0,1} and nothing for part 2.
    nodes = [
        make_node(0, [0, 0], _HEAT_PIPE_ID, overclocked=False),
        make_node(1, [2, 0], _OVERCLOCK_PART_ID, overclocked=True),
        make_node(2, [4, 0], _OVERCLOCK_PART_ID, overclocked=True),
    ]
    fake_geo = {
        _HEAT_PIPE_ID: _make_vanilla_geo((_make_thermal_port((0, 0), "Right"),)),
        _OVERCLOCK_PART_ID: _make_vanilla_geo(
            (
                _make_thermal_port((0, 0), "Left", overclock_conditional=True),
                _make_thermal_port((0, 0), "Right", overclock_conditional=True),
            )
        ),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    # heat_pipe ↔ part1 produces one thermal edge; part1 ↔ part2 is blocked.
    assert summary["thermal_edges"] == 1
    assert summary["networks"] == 1
    assert summary["network_sizes"] == [2]
    clusters = context.get_annotation("thermal_networks")
    assert clusters == [[0, 1]]


def test_overclocked_railgun_lateral_port_connection_is_blocked() -> None:
    """Two overclocked railgun parts with matching side ports must NOT connect via ports."""

    # Launcher (2×4) at [0, 0] — Right port at tile (1, 0) → 2x pos (2, 0) facing Right.
    # Accelerator (2×3) at [4, 0] — Left port at tile (0, 0) → 2x pos (4, 0) facing Left.
    # Right port at (2,0) + delta(2,0) = (4,0) facing Left → port match exists.
    # Both overclocked → port edge is suppressed.
    # Parts are side-by-side (X-axis gap), not barrel-stacked (Y-axis for rotation 0)
    # → no barrel-axis virtual edge fires either.
    nodes = [
        make_node(0, [0, 0], _RAILGUN_LAUNCHER_ID,    overclocked=True, footprint={"width": 2, "height": 4}),
        make_node(1, [4, 0], _RAILGUN_ACCELERATOR_ID, overclocked=True, footprint={"width": 2, "height": 3}),
    ]
    fake_geo = {
        _RAILGUN_LAUNCHER_ID: _make_vanilla_geo(
            (_make_thermal_port((1, 0), "Right", overclock_conditional=True),)
        ),
        _RAILGUN_ACCELERATOR_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Left"),)
        ),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 0
    assert summary["railgun_assembly_edges"] == 0
    assert summary["networks"] == 0
    assert context.get_annotation("thermal_networks") == []


def test_barrel_stacked_railgun_parts_form_virtual_thermal_edge() -> None:
    """Railgun parts stacked end-to-end along the barrel axis form a virtual thermal edge."""

    # Launcher (2×4) at [0, 0], rotation 0 → occupies 2x cells y=0..6.
    # Accelerator (2×3) at [0, 8], rotation 0 → top cells at y=8, barrel-adjacent to y=6.
    # No ports needed — the virtual edge comes from physical barrel adjacency.
    nodes = [
        make_node(0, [0, 0], _RAILGUN_LAUNCHER_ID, overclocked=True, footprint={"width": 2, "height": 4}),
        make_node(1, [0, 8], _RAILGUN_ACCELERATOR_ID, overclocked=True, footprint={"width": 2, "height": 3}),
    ]
    fake_geo: dict[str, Any] = {}

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["railgun_assembly_edges"] == 1
    assert summary["thermal_edges"] == 0
    assert summary["networks"] == 1
    assert summary["network_sizes"] == [2]
    assert context.get_annotation("thermal_networks") == [[0, 1]]


def test_barrel_stacked_railgun_non_overclocked_also_connects() -> None:
    """Barrel-axis virtual edges apply regardless of overclocked status."""

    nodes = [
        make_node(0, [0, 0], _RAILGUN_LOADER_ID, overclocked=False, footprint={"width": 2, "height": 3}),
        make_node(1, [0, 6], _RAILGUN_LAUNCHER_ID, overclocked=False, footprint={"width": 2, "height": 4}),
    ]
    fake_geo: dict[str, Any] = {}

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["railgun_assembly_edges"] == 1
    assert summary["networks"] == 1
    assert context.get_annotation("thermal_networks") == [[0, 1]]


def test_overclocked_railgun_does_not_extend_to_non_railgun_overclocked_part() -> None:
    """An overclocked railgun barrel-adjacent to an overclocked non-railgun part must not connect."""

    # Railgun launcher (2×4) at [0, 0], rotation 0.
    # Overclocked non-railgun part at [0, 8] — barrel-adjacent but not a railgun component.
    # No port connections (both overclocked, blocked).
    # No railgun virtual edge (other part is not railgun).
    nodes = [
        make_node(0, [0, 0], _RAILGUN_LAUNCHER_ID, overclocked=True, footprint={"width": 2, "height": 4}),
        make_node(1, [0, 8], _OVERCLOCK_PART_ID, overclocked=True, footprint={"width": 1, "height": 1}),
    ]
    fake_geo: dict[str, Any] = {}

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["railgun_assembly_edges"] == 0
    assert summary["thermal_edges"] == 0
    assert summary["networks"] == 0
    assert context.get_annotation("thermal_networks") == []


def test_overclocked_engine_room_port_connects_to_overclocked_part() -> None:
    """An overclocked engine room's port-based connection to another overclocked part is allowed."""

    # Engine room at [0, 0] — Right port (overclock_conditional).
    # Overclocked part at [2, 0] — Left port (overclock_conditional).
    # Engine room is exempt → edge allowed.
    nodes = [
        make_node(0, [0, 0], _ENGINE_ROOM_ID, overclocked=True),
        make_node(1, [2, 0], _OVERCLOCK_PART_ID, overclocked=True),
    ]
    fake_geo = {
        _ENGINE_ROOM_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Right", overclock_conditional=True),)
        ),
        _OVERCLOCK_PART_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Left", overclock_conditional=True),)
        ),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 1
    assert summary["networks"] == 1
    assert context.get_annotation("thermal_networks") == [[0, 1]]

# ---------------------------------------------------------------------------
# Thermal-conduit restriction tests (OC part <-> non-OC non-conduit blocked)
# ---------------------------------------------------------------------------


def test_overclocked_part_does_not_connect_to_non_thermal_conduit() -> None:
    """An overclocked part must NOT form a thermal edge with a non-conduit non-OC part.

    A railgun accelerator has thermal ports for weapon-assembly purposes; those
    ports must not bridge into the ship's overclocked thermal network.
    """

    nodes = [
        make_node(0, [0, 0], _POWER_STORAGE_ID, overclocked=True),
        make_node(1, [2, 0], _RAILGUN_ACCELERATOR_ID, overclocked=False),
    ]
    fake_geo = {
        _POWER_STORAGE_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Right", overclock_conditional=True),)
        ),
        _RAILGUN_ACCELERATOR_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Left", overclock_conditional=False),)
        ),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 0
    assert summary["networks"] == 0
    assert context.get_annotation("thermal_networks") == []


def test_overclocked_part_connects_to_resonance_beam_turret() -> None:
    """An overclocked part MUST connect to a resonance beam turret (thermal lance).

    The resonance beam turret is a first-class thermal conduit; it must be able
    to form port-based edges with overclocked parts despite the non-OC side
    being a weapon.
    """

    nodes = [
        make_node(0, [0, 0], _POWER_STORAGE_ID, overclocked=True),
        make_node(1, [2, 0], _RESONANCE_BEAM_ID, overclocked=False),
    ]
    fake_geo = {
        _POWER_STORAGE_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Right", overclock_conditional=True),)
        ),
        _RESONANCE_BEAM_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Left", overclock_conditional=False),)
        ),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    assert summary["thermal_edges"] == 1
    assert summary["networks"] == 1
    assert context.get_annotation("thermal_networks") == [[0, 1]]


def test_railgun_isolation_from_overclocked_capacitor_network() -> None:
    """Railguns must not merge into the thermal network of an overclocked capacitor.

    The capacitor's ports physically face both the resonance beam and the railgun
    accelerator, but only the resonance beam is a thermal conduit.
    """

    nodes = [
        # Overclocked capacitor at centre
        make_node(0, [0, 0], _POWER_STORAGE_ID, overclocked=True),
        # Resonance beam turret on the right (thermal conduit — should connect)
        make_node(1, [2, 0], _RESONANCE_BEAM_ID, overclocked=False),
        # Railgun accelerator on the left (not a thermal conduit — must NOT connect)
        make_node(2, [-2, 0], _RAILGUN_ACCELERATOR_ID, overclocked=False),
    ]
    fake_geo = {
        # Capacitor: Left port faces railgun, Right port faces resonance beam
        _POWER_STORAGE_ID: _make_vanilla_geo(
            (
                _make_thermal_port((0, 0), "Right", overclock_conditional=True),
                _make_thermal_port((0, 0), "Left", overclock_conditional=True),
            )
        ),
        _RESONANCE_BEAM_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Left", overclock_conditional=False),)
        ),
        _RAILGUN_ACCELERATOR_ID: _make_vanilla_geo(
            (_make_thermal_port((0, 0), "Right", overclock_conditional=False),)
        ),
    }

    context, summary = _run_thermal_pass(nodes, fake_geometry=fake_geo)

    # Only capacitor <-> resonance_beam is allowed; capacitor <-> railgun is blocked.
    assert summary["thermal_edges"] == 1
    assert summary["networks"] == 1
    assert summary["network_sizes"] == [2]
    clusters = context.get_annotation("thermal_networks")
    assert clusters == [[0, 1]]  # railgun (2) is excluded
