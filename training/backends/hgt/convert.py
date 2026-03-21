"""Convert expanded graph JSON payloads to PyG HeteroData objects."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import orjson

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
    "hull_peri",   # hull_perimeter virtual node
    "hull_int",    # hull_interior virtual node
    "zone",        # spatial_zone virtual nodes (cardinal)
    "zone_rot",    # spatial_zone_rotated virtual nodes (22.5°-rotated)
    "weapon_grp",  # weapon_group virtual nodes
    "ship_info",      # global_ship_info virtual node
]

# All (src_type, relation, tgt_type) triples that can appear in the data.
EDGE_TYPES: list[tuple[str, str, str]] = [
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
    # Thermal / hull / zone / weapon membership (virtual → part)
    ("thermal", "thermal_member", "part"),
    ("hull_peri", "hull_member", "part"),
    ("hull_int", "interior_member", "part"),
    ("zone", "zone_member", "part"),
    ("zone_rot", "zone_member_rotated", "part"),
    ("weapon_grp", "weapon_member", "part"),
    ("cluster", "super_member", "part"),
    # Reverse membership edges (part → virtual) so virtual nodes aggregate from members
    ("part", "rev_thermal_member", "thermal"),
    ("part", "rev_hull_member", "hull_peri"),
    ("part", "rev_interior_member", "hull_int"),
    ("part", "rev_zone_member", "zone"),
    ("part", "rev_zone_member_rotated", "zone_rot"),
    ("part", "rev_weapon_member", "weapon_grp"),
    ("part", "rev_super_member", "cluster"),
    # Global ship node → every other virtual node type
    ("ship_info", "links_cluster", "cluster"),
    ("ship_info", "links_thermal", "thermal"),
    ("ship_info", "links_hull_peri", "hull_peri"),
    ("ship_info", "links_hull_int", "hull_int"),
    ("ship_info", "links_zone", "zone"),
    ("ship_info", "links_zone_rot", "zone_rot"),
    ("ship_info", "links_weapon_grp", "weapon_grp"),
]

METADATA: tuple[list[str], list[tuple[str, str, str]]] = (NODE_TYPES, EDGE_TYPES)

# ---------------------------------------------------------------------------
# Internal lookup tables
# ---------------------------------------------------------------------------

_KIND_TO_NODE_TYPE: dict[str, str] = {
    "traversable_cluster": "cluster",
    "thermal_network": "thermal",
    "hull_perimeter": "hull_peri",
    "hull_interior": "hull_int",
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
    "hull_member": ("hull_peri", "hull_member", "part"),
    "interior_member": ("hull_int", "interior_member", "part"),
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
    "hull_peri":  "links_hull_peri",
    "hull_int":   "links_hull_int",
    "zone":       "links_zone",
    "zone_rot":   "links_zone_rot",
    "weapon_grp": "links_weapon_grp",
}

# Forward membership edge key → reverse edge key (part → virtual).
# Used to mirror membership edges so virtual nodes aggregate from their members.
_REVERSE_MEMBERSHIP: dict[tuple[str, str, str], tuple[str, str, str]] = {
    ("thermal",    "thermal_member",      "part"): ("part", "rev_thermal_member",      "thermal"),
    ("hull_peri",  "hull_member",         "part"): ("part", "rev_hull_member",          "hull_peri"),
    ("hull_int",   "interior_member",     "part"): ("part", "rev_interior_member",      "hull_int"),
    ("zone",       "zone_member",         "part"): ("part", "rev_zone_member",          "zone"),
    ("zone_rot",   "zone_member_rotated", "part"): ("part", "rev_zone_member_rotated",  "zone_rot"),
    ("weapon_grp", "weapon_member",       "part"): ("part", "rev_weapon_member",        "weapon_grp"),
    ("cluster",    "super_member",        "part"): ("part", "rev_super_member",         "cluster"),
}

_ZONE_LABEL_IDX: dict[str, int] = {z: i for i, z in enumerate(ZONE_LABELS)}
_ZONE_ROT_LABEL_IDX: dict[str, int] = {z: i for i, z in enumerate(ZONE_ROT_LABELS)}
_WEAPON_TYPE_IDX: dict[str, int] = {w: i for i, w in enumerate(WEAPON_TYPES)}



# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_graph(payload: dict[str, Any], vocab: VocabRegistry) -> "HeteroData":  # type: ignore[name-defined]
    """Convert one expanded graph JSON payload to a PyG :class:`HeteroData` object.

    Node feature layout
    -------------------
    ``part``:
      - ``.part_id`` LongTensor [N] — vocab index
      - ``.rotation`` LongTensor [N] — 0..3
      - ``.x`` FloatTensor [N, 7] — [loc_x, loc_y, fp_cells, fp_w, fp_h, traversable, overclocked]

    Virtual node types (``cluster``, ``thermal``, ``hull_peri``, ``hull_int``):
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

    data["part"].part_id = torch.tensor(
        [vocab.encode(nd["part_id"]) for nd in nodes], dtype=torch.long
    )
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
            data["ship_info"].x = torch.ones(1, 1, dtype=torch.float)
            data["ship_info"].num_nodes = 1
            continue

        mc = torch.tensor(
            [[math.log1p(vn["member_count"])] for vn in vns], dtype=torch.float
        )
        data[pyg_type].x = mc
        data[pyg_type].num_nodes = len(vns)

        if kind == "spatial_zone":
            data[pyg_type].zone_label = torch.tensor(
                [_ZONE_LABEL_IDX.get(vn.get("zone_label", ""), 0) for vn in vns],
                dtype=torch.long,
            )
        elif kind == "spatial_zone_rotated":
            data[pyg_type].zone_label = torch.tensor(
                [_ZONE_ROT_LABEL_IDX.get(vn.get("zone_label", ""), 0) for vn in vns],
                dtype=torch.long,
            )
        elif kind == "weapon_group":
            data[pyg_type].weapon_type = torch.tensor(
                [_WEAPON_TYPE_IDX.get(vn.get("weapon_type", ""), 0) for vn in vns],
                dtype=torch.long,
            )

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
            data = convert_graph(payload, vocab)
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
