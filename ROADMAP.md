
## Project:

1. ~~Decide on a name~~ → **Voidwright**
2. ~~Post ship training opt-in form on Excelsior~~
3. ~~Open-source repo~~
3. Recruit contributors


## Preprocessing:

Status: Completed

All five stages are implemented with hardware-agnostic parallelism (`--workers`, `--executor {auto,thread,process}`):

1. ~~Extract~~ — `.ship.png` → raw JSON
2. ~~Canonicalize~~ — SHA-256 deduplication, hash-ordered collision naming
3. ~~Graphs~~ — structural part graphs with connectivity and door-rule analysis
4. ~~Door-rules~~ — analysis stage (generator does not synthesize doors)
5. ~~Pipeline~~ — end-to-end orchestration with manifest-based sync and partial-failure tolerance

### Relative coords transform:
Goal:
Normalize ship positions into a canonical local coordinate frame so model inputs are translation-invariant and consistent across the dataset, while remaining reversible and backward-compatible.

Reasoning:
Current preprocessing preserves absolute placement on the global grid, which can introduce unnecessary variance from arbitrary ship offsets. We want to remove this failure mode now in a way that also supports future architectures (e.g., heterogeneous graph transformers). Because many ships have even dimensions and no single center cell, use an integer-only 2x scaled coordinate frame centered on the occupied-cell bbox center (allowing half-cell centers without floats). Keep transform metadata so coordinates can be mapped back exactly when needed.

Status: Completed


## Graphs:

Ship connectivity and crew traversal graphs should be processed into hierarchical graphs that encode more meaningful information such as corridor pathways, weapon modules, exterior facing parts, connected ion cores, etc.

Additional information should be added to the nodes of the graph as well, such as part distance from the ship's center anchor part, the distance from the surface, from suppliers, and the minimum distance to the nearest crew, etc.

Potentially we should add more edge types as well to connect related parts together.

### Graph expansion — structural backend:

Status: Completed

Adds a global `ship_info` virtual node and traversable-cluster super-nodes (with cross-edges) to preprocessing graph JSON. Supports parallelism and deterministic output. See `graph_expansion/backends/structural/`.

Remaining enrichment (weapon modules, exterior-facing parts, ion-core groupings, per-node distance metrics) is not yet implemented.

### Graph expansion performance:

The current `graph_expansion` structural backend processes ships sequentially and may be slow on large corpora. Profile and investigate parallelism, algorithmic improvements, or batching to bring throughput in line with the preprocessing pipeline stages.


## Markov Generator:

Status: Completed

Fully functional Markov backend with seeded generation (JSON and PNG), mirror-symmetry mode, allowlist/requirements filtering, per-sample error tolerance, and MP4 visualization. See `generator/backends/markov/`.


## Heterogeneous Graph Transformer:

This is the model that will use the more heavily processed graph data. We should perform pre-training masked prediction on the ship corpus. And then run tests to see if the resulting embeddings are meaningful.

The `graph_expansion` structural backend provides the virtual-node / super-node structure that this model will consume — but no training infrastructure exists yet.

Architecture: GraphGPS-ish hybrid, sparse/local attention

Layers: 4 to 6

Hidden size: 128

Heads: 4

Local neighborhood / sparse degree: modest, not huge

Batch size: 1, then use gradient accumulation

Precision: AMP on

Graph design: be selective with shortcut edges, don't create a fake dense graph


## Verification Layer:

### PlacementValidator:

Status: Completed

`ship_layout/validator.py` owns `PlacementValidator` and covers geometry, connectivity, overlap, bounds, allowlist, mirror mode, companion collapse, and seed validation. All generator backends should use this API.

### Remaining checks (not yet implemented):

1.
Expand verification to check actual non-rectangular part geometry for wedge/triangle parts and parts with a physical rect field — dead-zone and exclusion-zone overlap is not yet enforced during generation.

2.
Crew traversal access validation beyond cluster membership (fine-grained pathfinding to verify a placed part is actually reachable).

3.
Weapon firing arc restrictions and shield bubble interference checks.


## Generator:

Beyond the simple Markov generator, we should build a new generator to mesh well with the encoder from the heterogeneous graph transformer. This generator may include some form of auto regressive generation with verification checks. Maybe also a hierarchical generation step where supernodes are predicted first, and then an auto-regressive generator fills in the supernode area.
