# Markov module layout

This package is intentionally split into leaf modules to keep runtime and training
concerns separate while preserving `markov.model` as the compatibility facade.

## Module responsibilities

- `types.py`: shared dataclasses, constants, token serialization, and config JSON helpers
- `state.py`: compact Markov history/state key encoding
- `corpus.py`: adapters from extracted JSON/graph payloads into `ShipPart` records
- `order.py`: root selection, part ordering, and touch checks
- `training.py`: payload construction from canonical and graph corpora
- `generation.py`: weighted sampling and runtime ship layout generation; delegates all placement validation to `ship_layout.validator.PlacementValidator`
- `symmetry.py`: backward-compat re-export shim; real mirror computation lives in `ship_layout/symmetry.py`
- `validation.py`: coordinate/footprint consistency validation over canonical corpus
- `model.py`: stable public API surface (`RelativeMarkovModel`, build helpers, re-exports)

## Dependency direction

Prefer this import direction to avoid circular imports:

- leaf/shared: `types -> state/corpus/order/training/generation/validation`
- runtime facade: `model -> generation/training/validation + leaf modules`
- package facade: `__init__ -> model`
- placement logic: `generation -> ship_layout.validator` (never the reverse)

## Compatibility rules

- External callers should continue importing from `markov.model` or `markov`
- Internal modules should import siblings directly, not through `markov.__init__`
- Keep `RelativeMarkovModel.load`, `.save`, and `.generate` stable
- Keep `ShipPart`, `TrainingConfig`, and `GenerationConfig` field names stable
