# Markov module layout

This package is intentionally split into leaf modules to keep runtime and training
concerns separate while preserving `markov.model` as the compatibility facade.

## Module responsibilities

- `types.py`: shared dataclasses, constants, token serialization, and config JSON helpers
- `state.py`: compact Markov history/state key encoding
- `corpus.py`: adapters from extracted JSON/graph payloads into `ShipPart` records
- `order.py`: root selection, part ordering, and touch checks
- `training.py`: payload construction from canonical and graph corpora
- `generation.py`: weighted sampling and runtime ship layout generation
- `validation.py`: coordinate/footprint consistency validation over canonical corpus
- `model.py`: stable public API surface (`RelativeMarkovModel`, build helpers, re-exports)

## Dependency direction

Prefer this import direction to avoid circular imports:

- leaf/shared: `types -> state/corpus/order/training/generation/validation`
- runtime facade: `model -> generation/training/validation + leaf modules`
- package facade: `__init__ -> model`

## Compatibility rules

- External callers should continue importing from `markov.model` or `markov`
- Internal modules should import siblings directly, not through `markov.__init__`
- Keep `RelativeMarkovModel.load`, `.save`, and `.generate` stable
- Keep `ShipPart`, `TrainingConfig`, and `GenerationConfig` field names stable
