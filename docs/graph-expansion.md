# Graph Expansion

This document describes the `graph_expansion/` module: what it consumes, what it emits, how the pass pipeline is structured, and where to extend it.

## Purpose

Graph expansion enriches preprocessing graph JSON artifacts with additional virtual nodes, cross-edges, and compact expansion metadata. It is intentionally separate from preprocessing graph construction and from training/generation.

Today the canonical implementation is a single structural expansion pipeline built from ordered passes.

## Where it fits

The broader data flow is:

```text
.ship.png
  -> extract
  -> canonicalize
  -> preprocessing graphs
  -> graph expansion (optional)
  -> training / downstream analysis
```

Graph expansion operates on preprocessing graph JSON files, typically from `generated_ship_graphs_canonical/`, and writes enriched JSON files to a separate output directory such as `expanded_ship_graphs/`.

## Entry points

### CLI

Canonical CLI usage:

```bash
python main.py graph-expansion expand \
  --input-dir generated_ship_graphs_canonical \
  --output-dir expanded_ship_graphs
```

A legacy positional pipeline name is still accepted for compatibility:

```bash
python main.py graph-expansion expand structural \
  --input-dir generated_ship_graphs_canonical \
  --output-dir expanded_ship_graphs
```

### Pipeline integration

The preprocessing pipeline can invoke graph expansion automatically:

```bash
python main.py preprocessing pipeline downloaded_ships \
  --output-dir generated_ship_graphs_canonical \
  --extracted-dir extracted_ship_data \
  --canonical-dir extracted_ship_data_canonical \
  --expansion-output-dir expanded_ship_graphs
```

## Module layout

The graph expansion package is now organized around passes rather than swappable backends.

```text
graph_expansion/
├── __init__.py
├── cli.py
├── context.py
├── structural.py
└── passes/
    ├── __init__.py
    ├── base.py
    ├── base_indexes.py
    ├── core_support_layer2.py
    ├── crew_access_layer1.py
    ├── global_ship_info.py
    ├── global_virtual_linker.py
    ├── hull_perimeter.py
    ├── spatial_zones.py
    ├── thermal_networks.py
    ├── travel_support.py
    ├── traversable_clusters.py
    └── weapon_groups.py
```

### `graph_expansion/structural.py`

This is the canonical orchestration module. It owns:

- pipeline constants such as `EXPANSION_NAME`, `EXPANSION_VERSION`, and `EXPANSION_GRAPH_NAME`
- the canonical ordered pass list in `DEFAULT_PASSES`
- single-payload expansion via `enrich_graph(...)`
- directory expansion via `expand_dir(...)`
- CLI parser wiring via `build_parser(...)` and `run_from_args(...)`

### `graph_expansion/context.py`

`ExpansionContext` is the shared mutable workspace for one expansion run. It owns:

- `source`: the original parsed graph payload
- `caches`: reusable derived structures built during this run
- `annotations`: transient derived facts for later passes
- `emitted_graphs`: virtual nodes / edges / summaries being accumulated
- `pass_reports`: ordered records of which passes ran

It also provides deterministic finalization of the enriched payload.

### `graph_expansion/passes/`

Each pass is a small unit of enrichment logic implementing `ExpansionPass` from `passes/base.py`.

Current structural passes:

- `BaseIndexesPass`
  - builds common structural graph lookups once
  - populates caches like `node_by_id`, `walkable_part_ids`, `door_edges`, and `touching_edges`

- `GlobalShipInfoPass`
  - emits a single global ship-info node
  - emits `global_member` cross-edges from that node to every structural node

- `TraversableClustersPass`
  - computes crew-traversable clusters conservatively from doors plus corridor-like adjacency
  - stores transient annotations like `traversable_clusters` and `cluster_by_part_id`
  - emits traversable-cluster super-nodes and `super_member` cross-edges

- `travel_support.py`
  - shared weighted-travel helpers for Layer 1 / Layer 2 passes
  - centralizes movement semantics, role classification, cached traversable-cluster cell graphs, and reverse-Dijkstra distance queries

- `Layer1CrewAccessPass`
  - emits direct structural-to-structural `crew_access_reactor` and `crew_access_factory` edges
  - computes weighted travel distance with Dijkstra over exact walkable `2x` cells plus explicit door portals
  - currently restricts crew-access edges to the crew room's own traversable cluster; if no in-cluster core target exists, the room is skipped
  - proxy recovery via structural `touching` edges remains available in shared helper code but is currently disabled
  - loads richer per-rotation travel metadata on demand from `common.geometry` via the shared travel-support module rather than inlining it into preprocessing graph JSON

- `Layer2CoreSupportPass`
  - emits downstream structural support edges from reactors/factories to infrastructure and weapon consumers in the same traversable cluster
  - reactor edges currently include `reactor_supports_power_storage`, `reactor_supports_shield`, `reactor_supports_engine_room`, `reactor_supports_thruster`, and `reactor_supports_energy_weapon`
  - factory edges currently include `factory_supports_storage`, `factory_supports_ammo_weapon`, and `factory_supports_missile_weapon`
  - reuses the same shared weighted Dijkstra travel helpers as Layer 1, but does not use classic-ship proxy fallback

- `ThermalNetworksPass`
  - builds thermal edges by matching facing thermal ports between adjacent structural nodes (ports from `common.geometry`)
  - overclock-conditional ports are only active when the owning node has `overclocked=True`
  - overclocked engine rooms add implicit thermal edges to every tile-adjacent thruster regardless of port alignment
  - railgun components (loaders, launchers, accelerators) receive virtual barrel-axis thermal edges so the whole assembly always forms one thermal unit
  - OC conduit restriction: port-matched edges between an overclocked part and a non-overclocked non-conduit part are suppressed; only dedicated thermal conduits (heat pipes, radiators, exchangers, resonance beam turrets) may bridge into an overclocked network via ports
  - two-phase clustering: Phase 1 unions non-OC thermal conduits into backbone spines; Phase 2 attaches non-backbone sub-groups as leaves without merging separate backbones
  - multi-network leaf membership: non-backbone sub-groups that touch more than one backbone cluster are added as leaves to all of them; the same part may hold `thermal_member` edges to multiple `thermal_network_N` nodes
  - connected heat exchangers expand their network by pulling in nearby overclocked non-conduit parts using a fixed 101-tile corner-cutout stencil (11×11 square with 5 cells removed from each corner), centered on each exchanger tile; stencil helpers live in `common.heat_exchanger` and are memoized
  - connected components form `thermal_network_N` virtual nodes with `thermal_member` cross-edges; isolated parts receive no node
  - the `thermal_network_by_part_id` annotation maps each node ID to a `List[str]` of network IDs; most nodes have a single-element list, multi-network leaf members have longer lists

- `HullPerimeterPass`
  - classifies each part as `perimeter` (has at least one unoccupied 2x neighbor cell) or `interior`
  - stores `hull_role_by_part_id` annotation
  - emits `hull_perimeter` and `interior` virtual nodes with `hull_member` / `interior_member` cross-edges

- `SpatialZonesPass`
  - computes each part's centroid from `location_2x` and rotation-adjusted footprint dimensions
  - assigns parts to one of eight compass-direction zones (`zone_e`, `zone_ne`, … `zone_se`) by angle from the 2x origin
  - stores `zone_by_part_id` annotation
  - emits one virtual zone node per populated zone with `zone_member` cross-edges
  - mirrored parts naturally land in opposing zone pairs, enabling downstream models to learn left-right symmetry as a co-occurrence pattern

- `SpatialZonesRotatedPass`
  - same semantics as `SpatialZonesPass` but sector boundaries rotated 22.5°, so they fall on cardinal and semi-cardinal directions
  - zone IDs use the 16-point naming convention (`zone_ene`, `zone_nne`, …)
  - cross-edges carry `kind="zone_member_rotated"`

- `WeaponGroupsPass`
  - detects weapon parts by matching `part_id` against an ordered substring vocabulary (`cannon`, `railgun`, `missile_launcher`, …)
  - groups them by weapon type (first match wins)
  - stores `weapon_group_by_part_id` annotation
  - emits `weapon_group_<type>` virtual nodes with `weapon_member` cross-edges

- `GlobalVirtualLinkerPass`
  - emits `global_virtual_member` cross-edges from the `global_ship` node to every other virtual node in the expansion graph
  - links the global anchor to all zone, cluster, hull, thermal-network, and weapon-group nodes

## Expansion flow

At a high level:

```text
source graph JSON
  -> ExpansionContext
  -> BaseIndexesPass
  -> GlobalShipInfoPass
  -> TraversableClustersPass
  -> Layer1CrewAccessPass
  -> Layer2CoreSupportPass
  -> ThermalNetworksPass
  -> HullPerimeterPass
  -> SpatialZonesPass
  -> SpatialZonesRotatedPass
  -> WeaponGroupsPass
  -> GlobalVirtualLinkerPass
  -> finalize enriched JSON
```

The pass order is explicit and deterministic. The current orchestration simply runs the passes in the order listed in `DEFAULT_PASSES`.

## Output shape

The structural pipeline currently adds an expansion graph named:

```text
X_expansion_structural
```

That graph contains:

- `nodes`
  - one `global_ship_info` node
  - zero or more `traversable_cluster` nodes
  - zero or more `thermal_network` nodes (one per connected thermal component)
  - one `hull_perimeter` node and one `interior` node
  - zero or more `spatial_zone` nodes (one per populated compass-direction zone)
  - zero or more `spatial_zone` nodes using the 22.5°-rotated sector layout (one per populated rotated zone, IDs use `zone_ene` / `zone_nne` / … naming)
  - zero or more `weapon_group` nodes (one per detected weapon type)
- `cross_edges`
  - `global_member` edges from the global node to every structural node
  - `super_member` edges from cluster nodes to their member structural nodes
  - direct structural-to-structural `crew_access_reactor` / `crew_access_factory` edges with weighted `travel_distance`
  - downstream structural support edges such as `reactor_supports_power_storage`, `reactor_supports_shield`, `reactor_supports_engine_room`, `reactor_supports_thruster`, `reactor_supports_energy_weapon`, `factory_supports_storage`, `factory_supports_ammo_weapon`, and `factory_supports_missile_weapon`
  - `thermal_member` edges from each thermal-network node to its member structural nodes
  - `hull_member` edges from the `hull_perimeter` node to perimeter parts
  - `interior_member` edges from the `interior` node to interior parts
  - `zone_member` edges from each zone node to its member structural nodes
  - `zone_member_rotated` edges from each rotated-zone node to its member structural nodes
  - `weapon_member` edges from each weapon-group node to its member structural nodes
  - `global_virtual_member` edges from the `global_ship` node to every virtual node
- `summary`
  - compact counts for all of the above: cluster count, crew-access edge counts, Layer 2 core-support edge counts, hull-perimeter/interior counts, spatial-zone and weapon-group node/edge counts

The top-level payload also gets an `expansion` metadata block like:

```json
{
  "expansion": {
    "backend": "structural",
    "version": 7,
    "graphs_added": ["X_expansion_structural"],
    "passes": [
      {"name": "base_indexes", "version": 1},
      {"name": "global_ship_info", "version": 1},
      {"name": "traversable_clusters", "version": 2},
      {"name": "crew_access_layer1", "version": 2},
      {"name": "core_support_layer2", "version": 1},
      {"name": "thermal_networks", "version": 9},
      {"name": "hull_perimeter", "version": 1},
      {"name": "spatial_zones", "version": 1},
      {"name": "weapon_groups", "version": 1}
    ]
  }
}
```

The external key is still named `backend` for artifact compatibility, even though the internal architecture is now pass-oriented.

## Determinism rules

Deterministic output is a deliberate contract.

Important rules:

- pass order is fixed
- traversable cluster memberships are sorted
- cluster list ordering is stable
- emitted graphs are finalized in deterministic key order
- pass metadata preserves execution order
- caches and transient annotations do not leak arbitrary container ordering into the persisted artifact

This matters for corpus comparisons, regression testing, and any future training use of expanded graphs.

## Regeneration behavior

Directory expansion is incremental.

`graph_expansion/structural.py` uses the standard version-sentinel helpers from `common.files`:

- skips outputs that are already current for `expansion_version`
- prunes stale output JSON files
- writes the current expansion version sentinel after a successful run

If expansion logic changes in a way that should force regeneration, bump `EXPANSION_VERSION`.

## Extending graph expansion

The intended extension point is a new pass, not a new backend layer.

### Add a new pass

1. Create a new file in `graph_expansion/passes/`
2. Implement a class derived from `ExpansionPass`
3. Use `ExpansionContext` caches / annotations / emitted graphs to share state
4. Add the pass to `DEFAULT_PASSES` in `graph_expansion/structural.py` at the correct position
5. Add focused tests

### Prefer passes over orchestration growth

Try to keep `graph_expansion/structural.py` small. It should mostly:

- define constants
- define the canonical pass order
- run the passes
- handle file / directory orchestration

New enrichment logic should live in passes, not in the orchestration layer.

## Testing

Focused test coverage currently lives in:

- `tests/test_graph_expansion.py`
  - end-to-end behavior
  - cluster rules
  - artifact shape
  - CLI and directory expansion behavior

- `tests/test_graph_expansion_context_and_passes.py`
  - `ExpansionContext` behavior
  - direct pass behavior
  - targeted cache / annotation / emitted-output checks

- `tests/test_thermal_networks_pass.py`
  - port matching, overclock-conditional behavior, connected components
  - engine room heat-conduit edges
  - heat-exchanger radius stencil shape, parity preservation, inclusion/exclusion boundaries

When changing graph expansion, prefer exact behavior assertions over loose smoke tests.

## Current limitations

A few deliberate constraints remain:

- there is no pass plugin registry or dependency solver yet
- `requires` / `provides` on passes are descriptive metadata, not enforced scheduling rules
- the persisted expansion metadata still uses the key `backend` for compatibility
- richer travel geometry is loaded on demand from `common.geometry`; preprocessing graph JSON still does not inline directional speed maps or blocked-travel direction metadata
- no specialized graph library such as `rustworkx` is used yet; plain Python structures remain the default

Those are intentional for now. The current goal is a clean, contributor-friendly, deterministic pass pipeline with efficient, pass-local travel analysis where needed.
