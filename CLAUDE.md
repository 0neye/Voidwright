# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Any changes to this file should be mirrored in AGENTS.md.

## What this repo does

Generates Cosmoteer `.ship.png` files from a learned model. The main workflow is:

1. (Optional) Download `.ship.png` files from Discord
2. Preprocess local images: extract embedded JSON -> canonicalize -> build ship graphs
3. Train a Markov model from graph outputs
4. Generate new encoded `.ship.png` files from the trained model

## Commands

**Python environment:**

Always activate the project venv before running any Python commands:
```bash
source .venv/bin/activate
```

**Install dependencies:**
```bash
source .venv/bin/activate
pip install -e .[dev]
# For the Discord download script only:
pip install -e .[scripts]
```

**Root CLI discovery:**
```bash
python main.py commands
python main.py help
python main.py repl
```

**Full preprocessing pipeline** (images -> graph JSON):
```bash
python main.py preprocessing pipeline downloaded_ships \
  --output-dir generated_ship_graphs_canonical \
  --extracted-dir extracted_ship_data \
  --canonical-dir extracted_ship_data_canonical \
  --verbose
```

Pipeline concurrency defaults are hardware-agnostic. Stage-specific overrides are available with:

- `--extract-workers` / `--extract-executor`
- `--canonicalize-workers` / `--canonicalize-executor`
- `--graph-workers` / `--graph-executor`
- `--expansion-workers` / `--expansion-executor`
- `auto` falls back to threads when process pools are unavailable in the current runner

Pass `--expansion-output-dir` to run graph expansion immediately after the graphs stage:

```bash
python main.py preprocessing pipeline downloaded_ships \
  --output-dir generated_ship_graphs_canonical \
  --extracted-dir extracted_ship_data \
  --canonical-dir extracted_ship_data_canonical \
  --expansion-output-dir expanded_ship_graphs \
  --verbose
```

**Preprocessing stages individually:**
```bash
python main.py preprocessing extract downloaded_ships --output-dir extracted_ship_data
python main.py preprocessing canonicalize --input-dir extracted_ship_data --output-dir extracted_ship_data_canonical
python main.py preprocessing graphs --input-dir extracted_ship_data_canonical --output-dir generated_ship_graphs_canonical
python main.py preprocessing door-rules --input-dir extracted_ship_data_canonical
```

**Expand graph JSON with virtual nodes and cross-edges:**
```bash
python main.py graph-expansion expand \
  --input-dir generated_ship_graphs_canonical \
  --output-dir expanded_ship_graphs
```

**Filter graph corpus** (optional rule-based accept/reject pass):
```bash
python main.py corpus \
  --input-dir generated_ship_graphs_canonical \
  --output-dir filtered_ship_graphs_canonical \
  --max-parts 300 \
  --require-crew-rooms \
  --require-reachable-reactor
```

Available filter flags: `--max-parts N`, `--max-occupied-cells N`, `--require-crew-rooms`, `--require-reachable-reactor` (needs expanded graphs), `--no-rejections-log`. Writes `manifest.json` and (when ships are rejected) `rejections.jsonl` to the output directory.

The package-level CLIs remain supported:

```bash
python -m preprocessing.cli ...
python -m training.cli ...
python -m generator.cli ...
python -m graph_expansion.cli ...
python -m corpus.cli ...
```

The `extract`, `canonicalize`, and `graphs` stages also accept:

- `--workers <n>`
- `--executor {auto,thread,process}`
- `--limit <n>` (non-destructive subset runs; skips pruning and version-sentinel writes)

**Train a Markov model** (preferred - from graph corpus):
```bash
python main.py training build markov \
  --graph-input-dir generated_ship_graphs_canonical \
  --output models/markov/markov-model.v2.json
```

**Validate coordinate assumptions:**
```bash
python main.py training validate markov \
  --input-dir extracted_ship_data_canonical \
  --output models/markov/coordinate-validation.v2.json
```

**Compute HGT corpus statistics (mask/loss calibration):**

> **Important:** All HGT training runs (including stats) must use `filtered_hgt_corpus` as the input directory, not `expanded_ship_graphs`. `filtered_hgt_corpus` is the corpus-filtered subset of `expanded_ship_graphs` that has been validated for HGT training quality.

```bash
python main.py training stats hgt \
  --input-dir filtered_hgt_corpus \
  --output models/hgt/corpus-stats.json
```

**Generate ships:**
```bash
python main.py generator generate markov \
  --model models/markov/markov-model.v2.json \
  --output-dir out/generated-ships \
  --count 5 \
  --seed 1337
```

Useful generation options:

- `--json-output-dir` writes generated JSON payloads
- `--seed-json` / `--seed-png` seeds generation from an existing layout
- `--mirror-symmetry` enables runtime left-right symmetry
- `--allowlist`, `--allowlist-file`, `--require`, and `--requirements-file` constrain output
- `--visualize` and `--visualization-fps` render MP4 growth videos alongside samples

**Render static ship visualizations** (expanded graph -> tinted PNG):
```bash
python main.py visualizer render spatial-zones --input <ship.png> [--input <ship2.png> ...]
python main.py visualizer render cardinal-zones --input <ship.png> ...
python main.py visualizer render traversable-clusters --input <ship.png> ...
```

Icons are auto-discovered (Steam on Windows; Linux Steam paths and WSL2 `/mnt/*/Program Files*/Steam` on POSIX; local cache at `assets/local/cosmoteer-icons/terran/`). Override with `--icons-root` or `--game-root`. Outputs go to `out/visualizations/<backend>/`.

**Discord acquisition** (requires `DISCORD_BOT_TOKEN` in `.env`):
```bash
python scripts/download_ship_images.py --output-dir downloaded_ships --verbose
```

**Patch blank Author fields in donated ships:**

Used when a donor forgot to set their author name in the ship file. The author
name to patch in should be looked up from the opt-in form CSV.

```bash
python scripts/patch_ship_author.py <input.ship.png> <output.ship.png> <author>
```

## Module architecture

The codebase is split into purpose-specific packages:

- **`preprocessing/`** - four-stage pipeline (extract -> canonicalize -> graphs -> door-rules). Each stage is its own submodule with a `main(argv)` and `build_parser()`. `pipeline.py` orchestrates all stages.
- **`graph_expansion/`** - structural graph enrichment implemented as a pass-oriented pipeline. `structural.py` orchestrates an ordered list of passes under `graph_expansion/passes/` that add virtual nodes and cross-edges to preprocessing graph JSON: a global ship-info node, traversable-cluster super-nodes, crew-access and core-support cross-edges, thermal-network virtual nodes, hull-perimeter/interior classification nodes, 8-sector spatial zone nodes, a 22.5°-rotated 8-sector zone variant, weapon-group nodes, and a global virtual linker node.
- **`training/`** - backend-agnostic router. `router.py` resolves backend names; each backend under `training/backends/<name>/` registers its own CLI parser via `register_build_parser` / `register_validate_parser` / `register_stats_parser`.
- **`generator/`** - backend-agnostic generation router. `generator/backends/markov/backend.py` wires CLI options; `generator/backends/markov/export.py` handles `.ship.png` encoding and roundtrip validation.
- **`markov/`** - shared Markov internals used by both training and generation: `model.py`, `generation.py`, `inputs.py`, and related helpers. `symmetry.py` is a backward-compat shim; mirror computation lives in `ship_layout/symmetry.py`.
- **`ship_layout/`** - shared structural geometry, connectivity, mirror symmetry (`symmetry.py`), and the `PlacementValidator` API (`validator.py`) used by generation and analysis.
- **`visualizer/`** - generation event capture, icon loading, frame rendering, and MP4 export; also hosts a static visualization system (`cli.py`, `router.py`, `static_render.py`, `backends/`) that renders expanded graph data as static PNGs tinted by zone, cluster, hull membership, or thermal-network membership; the thermal-networks backend also draws heat-exchanger absorption-radius overlays as stroke-only stencil outlines.
- **`corpus/`** - rule-based corpus filtering. `filter.py` scans a graph JSON directory, applies an ordered list of `CorpusRule` objects, copies accepted files to an output directory, and writes `manifest.json` / `rejections.jsonl`. Rules live under `corpus/rules/` (`MaxSizeRule`, `RequireCrewRoomsRule`, `RequireReachableReactorRule`). `context.py` provides `CorpusContext` (lazy accessors over the parsed graph payload). The flat CLI (`corpus/cli.py`) is registered in `main.py` as the `corpus` domain.
- **`common/`** - geometry metadata (`geometry.py`), heat-exchanger radius helpers (`heat_exchanger.py`), file helpers, logging, and `common/cosmoteer/` (parser and encoder for `.ship.png` LSB payloads). `common/data/vanilla_parts_full_geometry.json` is the authoritative part geometry source.

### Key data flow

```text
.ship.png -> parser -> raw JSON -> centered 2x extracted JSON -> canonical JSON -> graph JSON -> (optional) expanded graph JSON -> (optional) corpus filter -> Markov model -> generated JSON -> encoder -> .ship.png
```

### Adding a new backend

Register it in `training/router.py` and `generator/router.py` alongside the Markov backend. Implement `register_build_parser` / `run_build` (training) and `register_generate_parser` / `run_generate` (generator) following `MarkovTrainingBackend` / `MarkovGeneratorBackend` as templates.

Graph expansion no longer has a backend registry. Extend it by adding or reordering passes in `graph_expansion/structural.py` and implementing new passes under `graph_expansion/passes/`.

## Commit messages

Include the AI model name in every commit message footer. The noreply email for
the model's lab must be wrapped in `<>` angle brackets. Format:

```text
<subject line>

<body if needed>

Co-Authored-By: Model Name <noreply@lab.com>
```

## Major change workflow

After making a major change or refactor and running appropriate tests, please update the AGENTS.md and CLAUDE.md as well as any applicable docs, when you deem it appropriate. If unsure, prompt the user.

This includes files under `docs/`. For example: update `docs/graph-expansion.md` whenever passes, their behavior, output schema, or the expansion flow changes; update `docs/pipeline-and-artifacts.md` when preprocessing stages or artifact schemas change.

## Temp files

If you need to write a temporary script or generate a temp output of some kind, the file name should start with either "TEMP" or ".tmp".

## Important conventions

- **Geometry source of truth:** `common/data/vanilla_parts_full_geometry.json` via `common/geometry.py`. All vanilla part footprints, dimensions, traversability, stored-location rect metadata, mirrored-footprint behavior, and richer per-rotation travel metadata (directional crew speeds, blocked travel directions, Manhattan-path flags) come from here. Preprocessing graph JSON intentionally stays compact and does not inline that richer travel metadata; graph expansion should load it on demand from `common.geometry`. Non-vanilla parts fall back to regex inference.
- **Model artifacts** live under `models/markov/`. Preferred artifact: `markov-model.v2.json` (built from graph corpus). Legacy `v1` artifacts used the raw canonical corpus.
- **Graph training is preferred** over the legacy `--input-dir` raw-corpus path. Use `--graph-input-dir` when building models.
- **Canonicalization is content-based.** Files may get `__dedup-<12 hex>` suffixes - this is normal, not a failure.
- **Preprocessing concurrency is deterministic.** Parallel scan and graph workers are allowed, but manifests and output naming are reduced in sorted order so results stay stable across runs. Bad files during parallel graph generation are skipped with a warning rather than aborting the batch.
- **Adding a preprocessing stage** requires registering it in `_AUTO_STAGE_EXECUTORS` in `preprocessing/concurrency.py`; omitting it raises a `ValueError` when the stage runs with `executor=auto`.
- **The Markov generator does not synthesize doors.** Door-rule logic in `preprocessing/door_rules.py` and `preprocessing/door_rules_engine.py` is for analysis and future passes only.
- **Placement validation lives in `ship_layout`.** `ship_layout/validator.py` owns `PlacementValidator` and `ValidationResult`. New generator backends should use this API for all geometry, allowlist, connectivity, overlap, bounds, and mirror checks rather than implementing their own inline chains.
- **Mirror symmetry axis** is at `x = -0.5`. Left half: all footprint cells `x <= -1`; right half: `x >= 0`. Centerline-straddling parts are allowed only when their occupied footprint is mirror-balanced, and such parts count as valid primary anchors.
- **Mirror generation roots and companions can collapse to one part.** If a root or mirrored placement reflects onto the same occupied cells, generation keeps only the centered self-mirroring part instead of emitting an overlapping duplicate.
- **Seeded mirror mode validates occupied cells, not just part lists.** Asymmetric part compositions are accepted when the combined occupied footprint is mirror-balanced; asymmetric occupied footprints are rejected.
- **Seeded startup uses a synthetic virtual root.** Seeded generation picks a start token compatible with available seed anchors, preferring roots that have a viable next transition before roots that only match a seed signature.
- **`--seed-png` input is normalized through preprocessing coordinates.** Parsed PNG payloads with world `Parts[*].Location` values are rewritten through `preprocessing.relative_coords.apply_relative_coords_transform` so seed loading sees the same centered `Location2x` / `coord_transform.center_2x` frame as extracted corpus files.
- **Token format:** `(part_id, rotation, anchor_part_id, anchor_rotation, dx, dy)`. Root tokens use `anchor_part_id = "__ROOT__"`. END token is `"__END__"`.
- **Pipeline extract failures are partially tolerated.** `preprocessing/pipeline.py` treats extract exit code `2` as partial success and still runs canonicalize/graphs for successfully extracted files.
- **Pipeline stage outputs are persistent by default.** `preprocessing/pipeline.py` writes extraction/canonicalization/graph artifacts directly to persistent stage directories (no temp-dir sync step).
- **Each preprocessing stage has its own schema-version sentinel key.** Stage outputs use `.pipeline-version.json` with stage-specific keys (`extract_schema_version`, `canonical_schema_version`, `graph_schema_version`) to decide incremental vs full regeneration.
- **`--limit` is non-destructive at every preprocessing stage.** Limited runs do not prune stale outputs and do not update schema-version sentinels.
- **Canonical collision naming is hash-ordered.** In `preprocessing/canonicalize.py`, if multiple contents want the same canonical filename, the lexicographically smallest SHA-256 keeps the base name and the rest get `__dedup-<12hex>`.
- **Parser/encoder location math is save-rect aware.** `common/cosmoteer/parser.py` normalizes `Part.Location` to footprint-origin coordinates and `common/cosmoteer/encoder.py` denormalizes back through `common/save_rect.py` for roundtrip fidelity.
- **No-arg root CLI is interactive.** `main.py` defaults to REPL when no entrypoint is passed; no-argument runs are not non-interactive help output.
- **Windows REPL parsing uses native semantics.** `main.py` uses `CommandLineToArgvW` on Windows, so REPL quoting/path behavior follows Win32 command-line parsing rather than POSIX `shlex`.
- **Generation failures are soft by sample.** `generator/backends/markov/backend.py` skips per-sample `RuntimeError`s with warnings and still exits `0` for the overall command.
- **Graph training silently skips bad corpus files.** `markov/model.py` ignores malformed/unreadable graph JSON files and also ignores `manifest.json` when scanning `--graph-input-dir`.
- **Build-time validation requires canonical corpus input.** In `training/backends/markov/backend.py`, `--validation-output` only executes when `--input-dir` is provided; graph-only builds skip validation.
- **Part requirements merge by max, not sum.** `markov/inputs.py` merges duplicate requirement entries by per-part maximum required count.
- **Python module exports should be declared near the top.** New Python modules should define `__all__` near the top of the file (after imports) instead of at the bottom.
- **Graph expansion is optional and separate from training.** `graph_expansion/` enriches graph JSON with virtual nodes and cross-edges but is not consumed by training or generation by default. Travel-aware passes may load richer movement metadata from `common.geometry` at expansion time rather than embedding it into preprocessing graph JSON. Run graph expansion via `graph-expansion expand` or by passing `--expansion-output-dir` to the pipeline command.
- **Graph expansion cluster ordering is deterministic.** `graph_expansion/structural.py` normalizes each cluster's member list with `sorted()` before sorting the cluster list, so `traversable_cluster_N` indices are stable across runs.
- **Visualizer static backends** implement `StaticVisualizationBackend` from `visualizer/backends/base.py` and are registered in `visualizer/router.py`. Shared rendering utilities (bounds, grid, icon paste, zone legend) live in `visualizer/static_render.py`. Add new backends there; do not grow the CLI or router with backend-specific rendering logic.
- **Icon auto-discovery order:** `--icons-root` > `--game-root` > Steam auto-discovery (Windows registry on `nt`; Linux Steam paths plus WSL2 `/mnt/*/Program Files*/Steam` on `posix`) > local cache at `assets/local/cosmoteer-icons/terran/`. `blueprints.png` is preferred over `icon.png` when loading icons; `_validate_icons_root` accepts directories containing either.

## Graph expansion framework

The `graph_expansion/` package uses a pass-oriented framework built around a
stateful per-run `ExpansionContext`:

- `graph_expansion/context.py` defines `ExpansionContext`, which wraps a single
  source graph JSON payload and owns backend metadata, shared caches,
  transient annotations, emitted graphs, and ordered pass reports
- `graph_expansion/passes/` contains small, focused passes implementing
  `ExpansionPass` from `graph_expansion/passes/base.py`
- the structural expansion pipeline (`graph_expansion/structural.py`)
  now orchestrates an ordered list of passes instead of performing all
  enrichment inline

Structural passes (in pipeline order):

- `BaseIndexesPass` builds common structural graph indexes and stores them in
  `ExpansionContext.caches`
- `TraversableClustersPass` computes traversable clusters, stores cluster
  annotations, and emits traversable-cluster super-nodes with `super_member`
  cross-edges; single-part clusters and small clusters (combined walkable-cell
  footprint ≤ 16 2x-cells with no door edges) are filtered out and receive no
  super-node or cross-edges; each cluster node carries `member_count`,
  `door_count`, `walkable_cells_2x`, and `centroid_x`/`centroid_y`
- `Layer1CrewAccessPass` emits direct structural-to-structural `crew_access_reactor`
  and `crew_access_factory` cross-edges. It uses weighted Dijkstra over exact
  walkable 2x cells plus door portals, and may repair legacy isolated crew rooms
  via touching-edge proxy discovery. It intentionally loads richer travel
  metadata from `common.geometry` on demand instead of expanding preprocessing
  graph JSON.
- `Layer2CoreSupportPass` emits downstream structural support edges from
  reactors/factories to in-cluster infrastructure and weapon consumers such as
  `reactor_supports_power_storage`, `reactor_supports_shield`,
  `reactor_supports_engine_room`, `reactor_supports_thruster`,
  `reactor_supports_energy_weapon`, `factory_supports_storage`,
  `factory_supports_ammo_weapon`, and `factory_supports_missile_weapon`.
  It reuses the same weighted cluster-local Dijkstra machinery as Layer 1.
- `ThermalNetworksPass` identifies thermal connections between structural parts
  by matching thermal port geometry in ship space (ports loaded from
  `common.geometry`); overclock-conditional ports are only active when the
  owning part has `overclocked=True`; overclocked engine rooms force all
  directly connected thrusters to be overclocked too (already reflected in graph
  data) and act as heat conduits — any thruster tile-adjacent to an overclocked
  engine room gets an implicit thermal edge regardless of explicit port
  alignment; railgun components (loaders, launchers, accelerators) receive
  virtual barrel-axis thermal edges so the whole assembly always forms one
  thermal unit; an OC conduit restriction suppresses port-matched edges between
  an overclocked part and a non-overclocked non-conduit part (only dedicated
  thermal conduits — heat pipes, radiators, exchangers, resonance beam turrets —
  may bridge into an overclocked network via ports); connected components are
  built via two-phase clustering: Phase 1 unions non-OC thermal conduits into
  backbone spines, Phase 2 attaches non-backbone sub-groups as leaves without
  merging separate backbones; non-backbone sub-groups that touch more than one
  backbone cluster are added as leaves to all of them (multi-network leaf
  membership — the same part may hold `thermal_member` edges to multiple
  `thermal_network_N` virtual nodes); connected heat exchangers expand their
  network by pulling in nearby overclocked non-conduit parts using a fixed
  101-tile corner-cutout stencil (11×11 square with 5 cells removed from each
  corner) centered on each exchanger tile — helpers live in
  `common.heat_exchanger` and are memoized; isolated parts (no matching opposite
  port) receive no node; the `thermal_network_by_part_id` annotation maps each
  node ID to a list of network IDs (`Dict[int, List[str]]`; most nodes have a
  single-element list, but multi-network leaf members have longer lists); each
  thermal network node carries `member_count`, `backbone_count` (non-OC conduit
  members), and `overclocked_count`
- `HullPerimeterPass` classifies each part as perimeter or interior using 2x
  footprint cell neighbor checks; emits `hull_perimeter` / `interior` virtual
  nodes with `hull_member` / `interior_member` cross-edges
- `SpatialZonesPass` assigns each part to one or more of eight compass-direction
  zones by 2x footprint cell angles from the origin; parts straddling a zone
  boundary receive `zone_member` edges in every touched zone; mirrored parts
  land in opposing zone pairs; the `zone_by_part_id` annotation is
  `Dict[int, List[str]]`; each zone node carries `member_count`,
  `occupied_cells`, and `avg_radius_2x`
- `SpatialZonesRotatedPass` same semantics as `SpatialZonesPass` but sector
  boundaries rotated 22.5° so they fall on cardinal and semi-cardinal directions;
  zone IDs use the `zone_ene` / `zone_nne` / … 16-point naming convention;
  cross-edges carry `kind="zone_member_rotated"`; same additional node fields
  as `SpatialZonesPass`
- `WeaponGroupsPass` detects weapon parts by `part_id` substring matching,
  groups them by type, and emits `weapon_group_<type>` virtual nodes with
  `weapon_member` cross-edges; each weapon group node carries `member_count`,
  `centroid_x`/`centroid_y`, and `spatial_spread`
- `GlobalVirtualLinkerPass` emits the `global_ship_info` node with the top-level
  `ship` metadata and computed structural summary (`total_parts`, `occupied_cells`,
  `footprint_w_2x`/`footprint_h_2x`, and virtual node kind counts), and
  `global_virtual_member` cross-edges from it to every other virtual node in the
  expansion graph; runs last so all zone, cluster, hull, thermal-network, and
  weapon-group nodes are present

Contributor guidelines for graph expansion:

- add new enrichment logic as a pass under `graph_expansion/passes/`, not by
  growing the orchestration logic in `graph_expansion/structural.py`
- reuse cached structure via `ExpansionContext.caches` and keep heavy
  intermediate results in `ExpansionContext.annotations` instead of
  serializing everything
- keep output deterministic: sort cluster memberships, avoid leaking set
  iteration order into serialized lists, and rely on the context to merge
  graphs in a stable way
- keep persisted JSON changes minimal and focused; prefer adding compact,
  well-documented summaries and virtual nodes over dumping internal analysis
  structures

The final enriched payload includes an `expansion` block:

- `backend`: backend name (for example `"structural"`)
- `version`: backend version (bumped when behavior or schema evolves)
- `graphs_added`: list of added graph names (for example
  `"X_expansion_structural"`)
- `passes`: ordered list of `{"name", "version"}` records describing the
  passes that ran for that artifact

When extending graph expansion, add or update focused tests:

- context behavior: `tests/test_graph_expansion_context_and_passes.py`
- per-pass behavior and summaries
- end-to-end structural expansion behavior in `tests/test_graph_expansion.py`
