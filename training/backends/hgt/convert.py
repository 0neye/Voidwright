"""Convert expanded graph JSON payloads to PyG HeteroData objects."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import orjson

from common.geometry import FLIPPABLE_PART_IDS, FLIPPED_PART_ID_SUFFIX
from training.backends.hgt.vocab import (
    VocabRegistry,
    WEAPON_TYPES,
    ZONE_LABELS,
    ZONE_ROT_LABELS,
    _SKIP_FILENAMES,
)

__all__ = [
    "NODE_TYPES",
    "EDGE_TYPES",
    "DOOR_EDGE_KEY",
    "METADATA",
    "convert_graph",
    "convert_corpus",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

# Typed constant for the door edge key, used by the training loop.
DOOR_EDGE_KEY: tuple[str, str, str] = ("part", "door", "part")

# Short node type names used in PyG HeteroData.
NODE_TYPES: list[str] = [
    "part",        # structural parts
    "cluster",     # traversable_cluster virtual nodes
    "thermal",     # thermal_network virtual nodes
    "zone",        # spatial_zone virtual nodes (cardinal)
    "zone_rot",    # spatial_zone_rotated virtual nodes (22.5°-rotated)
    "weapon_grp",  # weapon_group virtual nodes
    "ship_info",   # global_ship_info virtual node
]

# All (src_type, relation, tgt_type) triples that can appear in the data.
# Base edge types always present.
_BASE_EDGE_TYPES: list[tuple[str, str, str]] = [
    # Structural part ↔ part
    ("part", "door", "part"),
    ("part", "touching", "part"),
    # Crew-access cross-edges (part → part)
    ("part", "crew_access_reactor", "part"),
    ("part", "crew_access_factory", "part"),
    # Reactor support cross-edges (part → part)
    ("part", "reactor_supports_power_storage", "part"),
    ("part", "reactor_supports_shield", "part"),
    ("part", "reactor_supports_engine_room", "part"),
    ("part", "reactor_supports_thruster", "part"),
    ("part", "reactor_supports_energy_weapon", "part"),
    # Factory support cross-edges (part → part)
    ("part", "factory_supports_storage", "part"),
    ("part", "factory_supports_ammo_weapon", "part"),
    ("part", "factory_supports_missile_weapon", "part"),
    # Thermal / zone / weapon / cluster membership (virtual → part)
    ("thermal", "thermal_member", "part"),
    ("zone", "zone_member", "part"),
    ("zone_rot", "zone_member_rotated", "part"),
    ("weapon_grp", "weapon_member", "part"),
    ("cluster", "super_member", "part"),
    # Global ship node → every other virtual node type
    ("ship_info", "links_cluster", "cluster"),
    ("ship_info", "links_thermal", "thermal"),
    ("ship_info", "links_zone", "zone"),
    ("ship_info", "links_zone_rot", "zone_rot"),
    ("ship_info", "links_weapon_grp", "weapon_grp"),
]

# Reverse membership edges (part → virtual) — only included when reverse_edges is enabled.
_REVERSE_EDGE_TYPES: list[tuple[str, str, str]] = [
    ("part", "rev_thermal_member", "thermal"),
    ("part", "rev_zone_member", "zone"),
    ("part", "rev_zone_member_rotated", "zone_rot"),
    ("part", "rev_weapon_member", "weapon_grp"),
    ("part", "rev_super_member", "cluster"),
]

# Full list including reverse edges (used as default for backward compat).
EDGE_TYPES: list[tuple[str, str, str]] = _BASE_EDGE_TYPES + _REVERSE_EDGE_TYPES

# Default METADATA includes all edge types.  Use ``build_metadata()`` to get a
# schema that matches the actual training configuration.
METADATA: tuple[list[str], list[tuple[str, str, str]]] = (NODE_TYPES, EDGE_TYPES)


def build_metadata(*, reverse_edges: bool = True) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Return a ``(node_types, edge_types)`` metadata tuple for HGTConv.

    When *reverse_edges* is False the reverse membership edge types are excluded
    so HGTConv does not allocate per-head transforms for unused relations.
    """
    if reverse_edges:
        return (NODE_TYPES, EDGE_TYPES)
    return (NODE_TYPES, list(_BASE_EDGE_TYPES))

# ---------------------------------------------------------------------------
# Internal lookup tables
# ---------------------------------------------------------------------------

_KIND_TO_NODE_TYPE: dict[str, str] = {
    "traversable_cluster": "cluster",
    "thermal_network": "thermal",
    "spatial_zone": "zone",
    "spatial_zone_rotated": "zone_rot",
    "weapon_group": "weapon_grp",
    "global_ship_info": "ship_info",
}

# cross_edge.kind → (src_pyg_type, rel_name, tgt_pyg_type)
_CROSS_EDGE_SCHEMA: dict[str, tuple[str, str, str]] = {
    "super_member": ("cluster", "super_member", "part"),
    "crew_access_reactor": ("part", "crew_access_reactor", "part"),
    "crew_access_factory": ("part", "crew_access_factory", "part"),
    "reactor_supports_power_storage": ("part", "reactor_supports_power_storage", "part"),
    "reactor_supports_shield": ("part", "reactor_supports_shield", "part"),
    "reactor_supports_engine_room": ("part", "reactor_supports_engine_room", "part"),
    "reactor_supports_thruster": ("part", "reactor_supports_thruster", "part"),
    "reactor_supports_energy_weapon": ("part", "reactor_supports_energy_weapon", "part"),
    "factory_supports_storage": ("part", "factory_supports_storage", "part"),
    "factory_supports_ammo_weapon": ("part", "factory_supports_ammo_weapon", "part"),
    "factory_supports_missile_weapon": ("part", "factory_supports_missile_weapon", "part"),
    "thermal_member": ("thermal", "thermal_member", "part"),
    "zone_member": ("zone", "zone_member", "part"),
    "zone_member_rotated": ("zone_rot", "zone_member_rotated", "part"),
    "weapon_member": ("weapon_grp", "weapon_member", "part"),
}

# Edge kinds that carry a travel_distance float feature.
_TRAVEL_EDGE_KINDS: frozenset[str] = frozenset({
    "crew_access_reactor", "crew_access_factory",
    "reactor_supports_power_storage", "reactor_supports_shield",
    "reactor_supports_engine_room", "reactor_supports_thruster",
    "reactor_supports_energy_weapon",
    "factory_supports_storage", "factory_supports_ammo_weapon",
    "factory_supports_missile_weapon",
})

# global_virtual_member target pyg_type → relation name
_GLOBAL_LINK_RELS: dict[str, str] = {
    "cluster":    "links_cluster",
    "thermal":    "links_thermal",
    "zone":       "links_zone",
    "zone_rot":   "links_zone_rot",
    "weapon_grp": "links_weapon_grp",
}

# Forward membership edge key → reverse edge key (part → virtual).
# Inverted directly from _REVERSE_EDGE_TYPES: (part, rev_rel, virtual) → (virtual, rel, part).
_REVERSE_MEMBERSHIP: dict[tuple[str, str, str], tuple[str, str, str]] = {
    (tgt, rel[4:], src): (src, rel, tgt)
    for src, rel, tgt in _REVERSE_EDGE_TYPES
}

_ZONE_LABEL_IDX: dict[str, int] = {z: i for i, z in enumerate(ZONE_LABELS)}
_ZONE_ROT_LABEL_IDX: dict[str, int] = {z: i for i, z in enumerate(ZONE_ROT_LABELS)}
_WEAPON_TYPE_IDX: dict[str, int] = {w: i for i, w in enumerate(WEAPON_TYPES)}



# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_graph(payload: dict[str, Any], vocab: VocabRegistry, *, reverse_edges: bool = True) -> "HeteroData":  # type: ignore[name-defined]
    """Convert one expanded graph JSON payload to a PyG :class:`HeteroData` object.

    Node feature layout
    -------------------
    ``part``:
      - ``.part_id`` LongTensor [N] — vocab index
      - ``.rotation`` LongTensor [N] — 0..3
      - ``.x`` FloatTensor [N, 7] — [loc_x, loc_y, fp_cells, fp_w, fp_h, traversable, overclocked]

    Virtual node types (``cluster``, ``thermal``):
      - ``.x`` FloatTensor [K, 1] — log1p(member_count)

    ``zone`` / ``zone_rot``:
      - ``.x`` FloatTensor [Z, 1]
      - ``.zone_label`` LongTensor [Z]

    ``weapon_grp``:
      - ``.x`` FloatTensor [W, 1]
      - ``.weapon_type`` LongTensor [W]

    ``global``:
      - ``.x`` FloatTensor [1, 1] — constant 1.0

    Edge feature layout
    -------------------
    ``(part, touching, part)``:
      - ``.edge_attr`` FloatTensor [E, 1] — shared_sides (float)

    Travel-distance edges (crew_access / *_supports_*):
      - ``.edge_attr`` FloatTensor [E, 1] — log1p(travel_distance)
    """
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    sg: dict[str, Any] = payload["graphs"]["A_structural_part_graph"]
    xg: dict[str, Any] = payload["graphs"].get("X_expansion_structural", {})

    # ------------------------------------------------------------------
    # Structural part nodes
    # ------------------------------------------------------------------
    nodes: list[dict[str, Any]] = sg["nodes"]
    n_parts = len(nodes)

    # Map part node id (arbitrary int) → local index 0..N-1
    part_id_to_idx: dict[int, int] = {nd["id"]: i for i, nd in enumerate(nodes)}

    # Flipped wedges are encoded as virtual part IDs (e.g.
    # "cosmoteer.armor_1x2_wedge__flipped") so the model learns flipped geometry
    # as a distinct part type rather than via a separate binary head.
    encoded_part_ids: list[int] = []
    for nd in nodes:
        pid = nd["part_id"]
        pid_lower = pid.lower()
        if nd.get("flip_x") and pid_lower in FLIPPABLE_PART_IDS:
            encoded_part_ids.append(vocab.encode(pid_lower + FLIPPED_PART_ID_SUFFIX))
        else:
            encoded_part_ids.append(vocab.encode(pid))

    data["part"].part_id = torch.tensor(encoded_part_ids, dtype=torch.long)
    data["part"].rotation = torch.tensor(
        [nd["rotation"] for nd in nodes], dtype=torch.long
    )
    locs = torch.tensor(
        [[nd["location_2x"][0], nd["location_2x"][1]] for nd in nodes],
        dtype=torch.float,
    )
    fp = torch.tensor(
        [
            [
                nd["footprint"]["cell_count"],
                nd["footprint"]["width"],
                nd["footprint"]["height"],
            ]
            for nd in nodes
        ],
        dtype=torch.float,
    )
    flags = torch.tensor(
        [[float(nd.get("traversable", False)), float(nd.get("overclocked", False))] for nd in nodes],
        dtype=torch.float,
    )
    data["part"].x = torch.cat([locs, fp, flags], dim=1)  # [N, 7]
    data["part"].num_nodes = n_parts

    # ------------------------------------------------------------------
    # Structural edges (door / touching)
    # ------------------------------------------------------------------
    door_srcs, door_tgts = [], []
    touch_srcs, touch_tgts, touch_ss = [], [], []

    for e in sg["edges"]:
        s = part_id_to_idx.get(e["source"])
        t = part_id_to_idx.get(e["target"])
        if s is None or t is None:
            continue
        if e["kind"] == "door":
            door_srcs.append(s)
            door_tgts.append(t)
        elif e["kind"] == "touching":
            touch_srcs.append(s)
            touch_tgts.append(t)
            touch_ss.append(float(e.get("shared_sides", 1)))

    if door_srcs:
        data["part", "door", "part"].edge_index = torch.tensor(
            [door_srcs, door_tgts], dtype=torch.long
        )
    if touch_srcs:
        data["part", "touching", "part"].edge_index = torch.tensor(
            [touch_srcs, touch_tgts], dtype=torch.long
        )
        data["part", "touching", "part"].edge_attr = torch.tensor(
            [[s] for s in touch_ss], dtype=torch.float
        )

    # ------------------------------------------------------------------
    # Virtual nodes
    # ------------------------------------------------------------------
    virt_by_kind: dict[str, list[dict[str, Any]]] = {}
    for vn in xg.get("nodes", []):
        virt_by_kind.setdefault(vn["kind"], []).append(vn)
    for vns in virt_by_kind.values():
        vns.sort(key=lambda n: str(n["id"]))

    # Map virtual node string id → (pyg_node_type, local_index)
    virt_id_to_local: dict[str, tuple[str, int]] = {}
    for kind, vns in virt_by_kind.items():
        pyg_type = _KIND_TO_NODE_TYPE.get(kind)
        if pyg_type is None:
            continue
        for i, vn in enumerate(vns):
            virt_id_to_local[str(vn["id"])] = (pyg_type, i)

    for kind, vns in virt_by_kind.items():
        pyg_type = _KIND_TO_NODE_TYPE.get(kind)
        if pyg_type is None:
            continue

        if pyg_type == "ship_info":
            vn = vns[0]
            data["ship_info"].x = torch.tensor([[
                math.log1p(float(vn.get("total_parts", 0))),
                math.log1p(float(vn.get("occupied_cells", 0))),
                float(vn.get("footprint_w_2x", 0.0)),
                float(vn.get("footprint_h_2x", 0.0)),
                math.log1p(float(vn.get("cluster_count", 0))),
                math.log1p(float(vn.get("thermal_count", 0))),
                math.log1p(float(vn.get("weapon_grp_count", 0))),
                math.log1p(float(vn.get("zone_count", 0))),
            ]], dtype=torch.float)
            data["ship_info"].num_nodes = 1
            continue

        if kind == "traversable_cluster":
            rows = []
            for vn in vns:
                mc = float(vn.get("member_count", 0))
                rows.append([
                    math.log1p(mc),
                    math.log1p(float(vn.get("door_count", 0))),
                    math.log1p(float(vn.get("walkable_cells_2x", 0))),
                    float(vn.get("centroid_x", 0.0)),
                    float(vn.get("centroid_y", 0.0)),
                ])
            data[pyg_type].x = torch.tensor(rows, dtype=torch.float)
            data[pyg_type].num_nodes = len(vns)
            continue

        if kind == "thermal_network":
            rows = []
            for vn in vns:
                mc = float(vn.get("member_count", 1))
                bc = float(vn.get("backbone_count", 0))
                oc = float(vn.get("overclocked_count", 0))
                leaf_frac = (mc - bc) / mc if mc > 0 else 0.0
                rows.append([math.log1p(mc), math.log1p(bc), math.log1p(oc), leaf_frac])
            data[pyg_type].x = torch.tensor(rows, dtype=torch.float)
            data[pyg_type].num_nodes = len(vns)
            continue

        if kind in ("spatial_zone", "spatial_zone_rotated"):
            rows = []
            for vn in vns:
                rows.append([
                    math.log1p(float(vn.get("member_count", 0))),
                    math.log1p(float(vn.get("occupied_cells", 0))),
                    float(vn.get("avg_radius_2x", 0.0)),
                ])
            data[pyg_type].x = torch.tensor(rows, dtype=torch.float)
            data[pyg_type].num_nodes = len(vns)
            if kind == "spatial_zone":
                data[pyg_type].zone_label = torch.tensor(
                    [_ZONE_LABEL_IDX.get(vn.get("zone_label", ""), 0) for vn in vns],
                    dtype=torch.long,
                )
            else:
                data[pyg_type].zone_label = torch.tensor(
                    [_ZONE_ROT_LABEL_IDX.get(vn.get("zone_label", ""), 0) for vn in vns],
                    dtype=torch.long,
                )
            continue

        if kind == "weapon_group":
            rows = []
            for vn in vns:
                rows.append([
                    math.log1p(float(vn.get("member_count", 0))),
                    float(vn.get("centroid_x", 0.0)),
                    float(vn.get("centroid_y", 0.0)),
                    math.log1p(float(vn.get("spatial_spread", 0.0))),
                ])
            data[pyg_type].x = torch.tensor(rows, dtype=torch.float)
            data[pyg_type].num_nodes = len(vns)
            data[pyg_type].weapon_type = torch.tensor(
                [_WEAPON_TYPE_IDX.get(vn.get("weapon_type", ""), 0) for vn in vns],
                dtype=torch.long,
            )
            continue

    # ------------------------------------------------------------------
    # Cross edges
    # ------------------------------------------------------------------
    # Accumulate per-(src_type, rel, tgt_type) lists before building tensors.
    edge_srcs: dict[tuple[str, str, str], list[int]] = {}
    edge_tgts: dict[tuple[str, str, str], list[int]] = {}
    edge_feats: dict[tuple[str, str, str], list[float]] = {}

    def _add(key: tuple[str, str, str], s: int, t: int, feat: float | None = None) -> None:
        edge_srcs.setdefault(key, []).append(s)
        edge_tgts.setdefault(key, []).append(t)
        if feat is not None:
            edge_feats.setdefault(key, []).append(feat)

    for ce in xg.get("cross_edges", []):
        ck: str = ce["kind"]

        if ck == "global_virtual_member":
            tgt_id = str(ce["target"])
            entry = virt_id_to_local.get(tgt_id)
            if entry is None:
                continue
            tgt_type, tgt_local = entry
            rel = _GLOBAL_LINK_RELS.get(tgt_type)
            if rel is None:
                continue
            _add(("ship_info", rel, tgt_type), 0, tgt_local)
            continue

        schema = _CROSS_EDGE_SCHEMA.get(ck)
        if schema is None:
            continue

        src_pyg, rel, tgt_pyg = schema
        src_raw = ce["source"]
        tgt_raw = ce["target"]

        # Resolve source
        if src_pyg == "part":
            src_local = part_id_to_idx.get(src_raw)
            if src_local is None:
                continue
        else:
            entry = virt_id_to_local.get(str(src_raw))
            if entry is None:
                continue
            _, src_local = entry

        # Resolve target (always a structural part for CROSS_EDGE_SCHEMA entries)
        tgt_local = part_id_to_idx.get(tgt_raw)
        if tgt_local is None:
            continue

        feat = None
        if ck in _TRAVEL_EDGE_KINDS:
            feat = math.log1p(float(ce.get("travel_distance", 0.0)))

        _add((src_pyg, rel, tgt_pyg), src_local, tgt_local, feat)

    for key in edge_srcs:
        data[key].edge_index = torch.tensor(
            [edge_srcs[key], edge_tgts[key]], dtype=torch.long
        )
        if key in edge_feats:
            data[key].edge_attr = torch.tensor(
                [[f] for f in edge_feats[key]], dtype=torch.float
            )

    # Mirror membership edges so virtual nodes can aggregate from their members.
    if reverse_edges:
        for fwd_key, rev_key in _REVERSE_MEMBERSHIP.items():
            if fwd_key in edge_srcs:
                data[rev_key].edge_index = torch.tensor(
                    [edge_tgts[fwd_key], edge_srcs[fwd_key]], dtype=torch.long
                )

    return data


def convert_corpus(
    input_dir: Path,
    cache_dir: Path,
    vocab: VocabRegistry,
    *,
    force: bool = False,
    reverse_edges: bool = True,
) -> list[Path]:
    """Convert every graph JSON in *input_dir* to a ``.pt`` file in *cache_dir*.

    Skips files that already have a matching ``.pt`` unless *force* is True.
    Returns the list of ``.pt`` paths in sorted order.
    """
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    pt_paths: list[Path] = []
    skipped = converted = errors = 0

    for json_path in sorted(input_dir.iterdir()):
        if json_path.suffix != ".json" or json_path.name in _SKIP_FILENAMES:
            continue
        pt_path = cache_dir / (json_path.stem + ".pt")
        if pt_path.exists() and not force:
            pt_paths.append(pt_path)
            skipped += 1
            continue
        try:
            payload = orjson.loads(json_path.read_bytes())
            data = convert_graph(payload, vocab, reverse_edges=reverse_edges)
            torch.save(data, pt_path)
            pt_paths.append(pt_path)
            converted += 1
        except Exception as exc:
            log.warning("Failed to convert %s: %s", json_path.name, exc)
            errors += 1

    log.info(
        "Corpus conversion: %d converted, %d skipped (cached), %d errors",
        converted, skipped, errors,
    )
    return sorted(pt_paths)
