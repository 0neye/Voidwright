# Markov Backend Implementation

This directory contains the generator-facing Markov adapter and exporter used by
the repository's backend-agnostic `generator/` module.

## Responsibility split

- `generator/backends/markov/backend.py`
  - generator CLI adapter that wires Markov runtime options into the shared model
- `generator/backends/markov/export.py`
  - `.ship.png` export and roundtrip validation
- `markov/`
  - shared model and backend-specific input parsing used by both training and generation
- `ship_layout/`
  - mirror symmetry computation (`ship_layout/symmetry.py`) and the `PlacementValidator` API (`ship_layout/validator.py`) used during generation
- `training/backends/markov/`
  - build and validation adapters
- `preprocessing/door_rules.py` and `preprocessing/door_rules_engine.py`
  - door-rule inference and validation for later passes
- `common/cosmoteer/`
  - shared `.ship.png` parser and encoder

## Canonical user-facing docs

Use `docs/markov-generator.md` for the full architecture guide, CLI usage,
mirror symmetry behavior, visualization flags, icon discovery behavior, and
artifact expectations. This README is intentionally kept short so it stays
aligned with the current package layout.
