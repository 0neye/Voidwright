
## Project:

1. Decide on a name
2. Post ship training opt-out form on Excelsior
3. Open-source repo
3. Recruit contributors


## Graphs:

Ship connectivity and crew traversal graphs should be processed into hierarchical graphs that encode more meaningful information such as corridor pathways, weapon modules, exterior facing parts, connected ion cores, etc.

Additional information should be added to the nodes of the graph as well, such as part distance from the ship's center anchor part, the distance from the surface, from suppliers, and the minimum distance to the nearest crew, etc.

Potentially we should add more edge types as well to connect related parts together.


## Heterogeneous Graph Transformer:

This is the model that will use the more heavily processed graph data. We should perform pre-training masked prediction on the ship corpus. And then run tests to see if the resulting embeddings are meaningful.

Architecture: GraphGPS-ish hybrid, sparse/local attention

Layers: 4 to 6

Hidden size: 128

Heads: 4

Local neighborhood / sparse degree: modest, not huge

Batch size: 1, then use gradient accumulation

Precision: AMP on

Graph design: be selective with shortcut edges, don’t create a fake dense graph


## Verification Layer:

Before implementing a generator, we should implement a robust verification system for part candidates. We should expand this beyond the simple connection/collision checks to include things like crew traversal access, dead zone/exclusion zone overlap, weapon firing arc restrictions, and shield bubble interference.


## Generator:

Beyond the simple Markov generator, we should build a new generator to mesh well with the encoder from the heterogeneous graph transformer. This generator may include some form of auto regressive generation with verification checks. Maybe also a hierarchical generation step where supernodes are predited first, and then an auto-regressive generator fills in the supernode area.