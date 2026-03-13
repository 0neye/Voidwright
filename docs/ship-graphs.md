# Ship Graphs

## Purpose

`preprocessing/graphs.py` builds graph-oriented analysis artifacts from extracted or canonical ship JSON files.
It is meant for structural and traversability analysis, not exact gameplay simulation.

The script produces:

- a structural part graph (parts as nodes, touching and door edges)
- a manifest with aggregate counts and unknown-part reporting

## Inputs and outputs

Preferred canonical-corpus run:

```bash
python -m preprocessing.cli graphs \
  --input-dir extracted_ship_data_canonical \
  --output-dir generated_ship_graphs_canonical
```

The input corpus is expected to be the extractor/canonicalizer output that already
stores centered `2x` coordinates plus `coord_transform.center_2x` replay metadata.

Legacy raw extracted payloads without that transform metadata are no longer the
documented graph-input contract.

Alternate output directory example:

```bash
python -m preprocessing.cli graphs \
  --input-dir extracted_ship_data_canonical \
  --output-dir generated_ship_graphs
```

The script writes one JSON output per ship plus `manifest.json`.

## Graph products

### Structural part graph

Stored at `graphs.A_structural_part_graph`.

Nodes represent individual part instances and include:

- normalized part ID
- centered local 2x location (`location_2x`)
- rotation
- footprint dimensions and cell count
- traversability flag
- exact centered local 2x walkable cells for that part (`walkable_cells_2x`)
- metadata note about how geometry was derived

Edges are of two kinds:

**Touching edges** (`kind = "touching"`) represent physical hull contact between distinct parts and include:

- `source`
- `target`
- `kind = "touching"`
- `shared_sides`

**Door edges** (`kind = "door"`) represent explicit door connections between distinct parts and include:

- `source`
- `target`
- `kind = "door"`
- `door_index` (index into the top-level `doors` array)
- `orientation`

Doors whose cells cannot be resolved to known ship cells are counted as `dangling_door_records`
and omitted. Doors where both cells belong to the same part are counted as `internal_door_records`
and omitted.

The summary includes `parts`, `touching_edges`, `door_edges`, `door_records`,
`dangling_door_records`, `internal_door_records`, and `non_structural_door_records`.

### Top-level doors array

Normalized door records are preserved at the top level alongside the graph so callers can
replay or synthesize doors without reverse-engineering them from graph edges alone. Each record
includes `Cell2x` and `Orientation`.

## Assumptions

### Location anchoring

- `Location2x` is the source-of-truth part-origin coordinate in the ship-local centered 2x frame
- `coord_transform.center_2x` provides the ship-level replay anchor needed to recover global grid coordinates when needed
- World-grid coordinates are reconstructed internally for geometry helpers and final replay/export paths
- odd rotations still swap width and height for rectangular fallback footprints

### Door orientation mapping

- `Door.Cell2x` stores the right or bottom occupied doorway cell in the same centered 2x frame
- `Orientation = 0` joins `(x, y-1)` and `(x, y)`
- `Orientation = 1` joins `(x-1, y)` and `(x, y)`

### Traversability

- Vanilla parts use game-file geometry when available
- Unknown or non-vanilla parts fall back to regex and name-hint inference
- Structural nodes also preserve exact per-part walkable cells in `walkable_cells_2x`, so the merged graph can drive traversability analysis without a separate cell graph
- The boolean `traversable` flag is only a coarse summary; exact walkability should come from `walkable_cells_2x`

## Historical geometry note

Earlier graph generation used a compact local metadata table plus regex or `1x1` fallbacks.
The current code path imports shared geometry from `common/geometry.py`, so
vanilla parts can use the same authoritative geometry source as the door-rule
tooling.

Unknown or non-vanilla parts still rely on fallback inference.

## Canonical corpus notes

Useful corpus snapshot from the canonical graph run:

- input corpus: `extracted_ship_data_canonical`
- canonical ship count: `12913`
- raw extracted ship count before dedupe: `15610`
- duplicates removed before canonical graph generation: `2697`

This is useful when comparing aggregate graph counts between raw-corpus and canonical-corpus runs.

## Limitations

- Wedge and triangle behavior is only as good as the available geometry data or fallback approximation
- Unknown and modded parts are less trustworthy than vanilla parts backed by authoritative geometry
- Some door records remain dangling because they do not resolve cleanly to two occupied cells
- The graph output is strong for connectivity analysis, but not a substitute for full in-game legality or pathfinding
