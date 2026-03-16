# Project Docs

This folder keeps durable repository knowledge that should outlive one-off run
reports and transient implementation notes.

## Guides

- `docs/pipeline-and-artifacts.md` - end-to-end data pipeline, entrypoints, and artifact conventions
- `docs/ship-graphs.md` - graph outputs, schema, geometry assumptions, and door-edge semantics
- `docs/markov-generator.md` - Markov architecture, validation assumptions, generation behavior, and export limits
- `docs/generation-modes.md` - allowlists, mirror symmetry, part requirements, and seeded generation
- `docs/door-rules.md` - door-rule inference, validation behavior, curated overrides, and caveats
- `docs/graph-expansion.md` - graph expansion entrypoints, pass pipeline, artifact shape, and extension guidance

## Scope

These docs focus on stable implementation behavior and repository contracts.
They intentionally avoid preserving dated sample runs, temporary metrics tables,
or file-by-file change logs.
