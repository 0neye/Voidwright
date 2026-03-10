# Markov Generator

## Scope

The generator in `generators/markov/` is a conservative first-pass, vanilla-only relative-placement model.
It is designed to produce structurally plausible layouts, not fully game-ready ships.

Implemented behavior:

- Train only from the canonical corpus in `extracted_ship_data_canonical`
- Restrict training and placement to vanilla `cosmoteer.*` parts with known geometry
- Model ships as a sequence of root and anchor-relative placement tokens
- Support an explicit END token
- Reject overlap using game-file footprint geometry
- Enforce configurable caps for parts, attempts, resamples, and world bounds
- Export generated layouts back to `.ship.png`
- Roundtrip-validate exported PNGs with the existing extraction pipeline

Deferred behavior:

- Door synthesis during generation
- Pathfinding and accessibility cleanup
- Connectivity cleanup beyond overlap and bounds rejection
- Broader gameplay legality checks

## Training representation

Each training ship is reduced to vanilla parts with known geometry and then ordered conservatively.

Ordering strategy:

1. Choose a root part near the ship centroid
2. Repeatedly attach the next part using a nearest-anchor heuristic
3. Record each placement relative to the anchor part origin

Token forms:

- Root token: `(part_id, rotation, __ROOT__, 0, 0, 0)`
- Placement token: `(part_id, rotation, anchor_part_id, anchor_rotation, dx, dy)`
- END token

The Markov state is the recent token history, with order controlled by `--markov-order` and defaulting to `2`.

## Generation behavior

Generation samples a root, then repeatedly samples the next token from the current Markov state.

For each sampled placement token:

- The runtime finds an already placed anchor with matching part ID and rotation
- The new part origin is reconstructed as `anchor_origin + (dx, dy)`
- The placement is rejected if:
  - no matching anchor is available
  - the footprint overlaps occupied cells
  - the footprint exceeds configured bounds

Generation stops on:

- `end_token`
- `max_parts`
- `max_attempts`
- `no_transition_for_state`
- `placement_rejected_by_caps_or_anchor_missing`

When part requirements are active, `end_token` can also be suppressed until the requirements are met or attempts are exhausted.

## Coordinate assumptions and validation

The current model assumes extracted `Location` values are stable part-origin coordinates.
That assumption was validated against real canonical ships by reconstructing placements from anchor-relative offsets
and checking both exact part origins and exact world footprint cells.

Important implication:

- The model stores origin-relative `(dx, dy)` values
- Runtime safety still comes from full footprint geometry, not from the token sequence alone

The validation pass exists so future work can distinguish model limitations from coordinate-system mistakes.

## Build, validate, and generate

Build a fresh artifact from the canonical corpus:

```bash
python scripts/build_markov_generator.py build \
  --input-dir extracted_ship_data_canonical \
  --output out/markov/markov-model.v2.json \
  --validation-output out/markov/coordinate-validation.v2.json
```

Generate JSON samples:

```bash
python scripts/build_markov_generator.py generate \
  --model out/markov/markov-model.v2.json \
  --output out/markov/samples-v2 \
  --count 5 \
  --seed 1337 \
  --max-parts 250 \
  --max-attempts 3000
```

Validate coordinate assumptions directly against the canonical corpus:

```bash
python scripts/build_markov_generator.py validate \
  --input-dir extracted_ship_data_canonical \
  --output out/markov/coordinate-validation.v2.json
```

## Export behavior

The export pipeline in `generators/markov/export.py` converts generated JSON back into loadable `.ship.png` files.

The exporter:

1. Converts generated part records into Cosmoteer ship-part format
2. Builds a minimal ship dictionary with sensible defaults
3. Serializes the ship into the Cosmoteer binary object stream
4. Compresses the payload and embeds it in a carrier PNG
5. Re-extracts the PNG and compares the resulting parts to the source data

Current export limitations:

- Doors are not synthesized
- Colors and appearance are mostly defaults
- Advanced optional structures such as complex control groups are not the focus
- Successful export does not imply gameplay legality

## Practical limitations

- The generator is conservative and can terminate early when the sampled transition cannot be realized in the current layout
- Models built before stricter geometry-backed filtering may contain candidates that the current runtime must now skip
- Better structure, accessibility, and door passes are intended as later stages rather than part of this first-pass sampler
