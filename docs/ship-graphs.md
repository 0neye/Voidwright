# Ship Graphs

## Purpose

`preprocessing/graphs.py` builds graph-oriented analysis artifacts from extracted or canonical ship JSON files.
It is meant for structural and traversability analysis, not exact gameplay simulation.

The script produces:

- a structural part graph
- a cell-level connectivity graph
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
- metadata note about how geometry was derived

Edges represent physical touching between distinct parts and include:

- `source`
- `target`
- `kind = "touching"`
- `shared_sides`

### Cell graph

Stored at `graphs.C_cell_graph`.

Nodes represent occupied cells and include:

- `id = "x,y"`
- coordinates
- centered local 2x companion coordinate (`center_2x`)
- `occupied = true`
- traversability flag
- owning part indices

Edges are intentionally conservative and include:

- `kind = "intra_part"` for orthogonal traversal within the same traversable part
- `kind = "door"` for explicit connections derived from `Doors[].Cell2x` and `Orientation`

The script does not invent free traversal between neighboring separate parts unless a door record exists.

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
- Traversability is conservative and meant for analysis, not exact crew routing

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
