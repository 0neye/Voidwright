# Project Docs

This folder collects durable repository knowledge that was previously scattered across one-off reports.
It is meant to help future agents understand the current workflow, generator behavior, and known constraints
without preserving transient run logs or demo-specific output tables.

## Guides

- `docs/pipeline-and-artifacts.md` - end-to-end data pipeline, major scripts, and artifact conventions
- `docs/ship-graphs.md` - graph outputs, schema, assumptions, and corpus-level graph notes
- `docs/markov-generator.md` - Markov model architecture, validation assumptions, export behavior, and limits
- `docs/generation-modes.md` - allowlists, mirror symmetry, part requirements, and seeded generation
- `docs/door-rules.md` - door-rule inference, validation behavior, curated overrides, and current caveats

## Scope

The docs here keep implementation-relevant constraints and usage details.
They intentionally omit report-only material such as dated sample runs, temporary metrics tables,
and file-by-file "what changed" summaries.
