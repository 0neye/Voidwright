# First-pass vanilla-only Markov ship generator

This directory now contains two complementary pieces of generator infrastructure:

1. `door_rules.py` and `data/door-placement-rules.*.json`
   - reusable vanilla-only door-rule inference/validation for later passes
2. `model.py` + `cli.py` + `export.py`
   - first-pass vanilla-only **relative-placement Markov** builder, sampler, and `.ship.png` exporter

## Scope

This is intentionally a **first pass**.

Implemented here:
- train only from the canonical deduped corpus in `extracted_ship_data_canonical/`
- exclude all non-vanilla parts and skip vanilla IDs missing game-file geometry
- model ship growth as a sequence of relative placements
- include part ID, rotation, anchor-part identity, and origin-to-origin relative offsets
- support an explicit END token
- enforce configurable generation hard caps (parts, attempts, resamples, bounds)
- reject generated overlaps using full vanilla footprint geometry from game-file exports
- validate coordinate assumptions against the real extracted corpus
- export generated ships to `.ship.png` files loadable in-game
- roundtrip validation: encode → extract → compare parts list
- **allowlist support**: restrict both training and generation to a user-specified set of part IDs

Explicitly deferred:
- door synthesis during generation
- pathfinding / accessibility cleanup
- disconnected-subgraph cleanup beyond simple overlap/bounds rejection
- richer gameplay legality checks beyond conservative footprint overlap and world bounds

## Model shape

Each training ship is reduced to its vanilla parts and ordered conservatively:
- choose a root part near the ship centroid
- repeatedly attach the next part to the placed set using a nearest-anchor heuristic
- record the placement as a relative transform from the anchor origin to the new part origin

The learned sequence is:
- **root token**: `(part_id, rotation)` anchored to `__ROOT__`
- **placement token**: `(part_id, rotation, anchor_part_id, anchor_rotation, dx, dy)`
- **END token**

The Markov state is the last `N` emitted tokens (`--markov-order`, default `2`), compacted to part+rotation history.

## Coordinate and geometry assumptions

The current generator assumes the extracted ship `Location` values are stable part-origin coordinates.
That assumption is validated against **real canonical ships**, not toy examples, by:
- reconstructing each ordered placement from its stored anchor-relative `(dx, dy)`
- checking exact origin recovery
- checking exact world-footprint recovery using `vanilla-parts-from-game-files.json`
- measuring how often the chosen training anchor is footprint-touching vs merely near

This is deliberately conservative: the model stores **relative part origins**, while sampling uses full vanilla footprint geometry to reject overlapping placements.

## CLI

Use the thin wrapper script:

```bash
python scripts/build_markov_generator.py build \
  --input-dir extracted_ship_data_canonical \
  --output out/markov/markov-model.v2.json \
  --validation-output out/markov/coordinate-validation.v2.json
```

Generate samples:

```bash
python scripts/build_markov_generator.py generate \
  --model out/markov/markov-model.v2.json \
  --output out/markov/samples-v2 \
  --count 5 \
  --seed 1337 \
  --max-parts 250 \
  --max-attempts 3000 \
  --bounds-min-x -64 --bounds-max-x 64 \
  --bounds-min-y -64 --bounds-max-y 64
```

Generate samples **and** immediately export as `.ship.png`:

```bash
python scripts/build_markov_generator.py generate \
  --model out/markov/markov-model.v2.json \
  --output out/markov/samples-v2 \
  --export-png-dir out/markov/exported-ships \
  --count 5 \
  --seed 1337
```

Export existing JSON samples to `.ship.png` (without re-generating):

```bash
python scripts/build_markov_generator.py export \
  --input-dir  out/markov/samples-v2 \
  --output-dir out/markov/exported-ships \
  --report     out/markov/export-report.json
```

Validate the origin-to-origin relative-coordinate assumption directly against real corpus data:

```bash
python scripts/build_markov_generator.py validate \
  --input-dir extracted_ship_data_canonical \
  --output out/markov/coordinate-validation.v2.json
```

## Allowlist support

### Training-time allowlist

Pass `--allowlist` or `--allowlist-file` to `build` to restrict the model to a set of part IDs.
Parts not in the allowlist are treated as if they were non-vanilla and excluded from training
sequences.  This produces a focused model that only generates allowlisted parts.

```bash
python scripts/build_markov_generator.py build \
  --input-dir extracted_ship_data_canonical \
  --output out/markov/markov-model-armor-only.json \
  --allowlist \
    cosmoteer.armor_1x1 \
    cosmoteer.armor_2x2 \
    cosmoteer.armor_1x2 \
    cosmoteer.armor_1x4
```

Using a file (one part ID per line, `#` comments allowed):

```bash
python scripts/build_markov_generator.py build \
  --input-dir extracted_ship_data_canonical \
  --output out/markov/markov-model-custom.json \
  --allowlist-file my-allowed-parts.txt
```

### Generation-time allowlist

Pass `--allowlist` / `--allowlist-file` to `generate` to filter sampled tokens at generation time
without rebuilding the model.  This is faster for experimentation but may produce sparser or shorter
ships if the allowlist is very restrictive.

```bash
python scripts/build_markov_generator.py generate \
  --model out/markov/markov-model.v2.json \
  --output out/markov/samples-armor \
  --allowlist cosmoteer.armor_1x1 cosmoteer.armor_2x2 cosmoteer.armor_1x2 \
  --count 3
```

### Overconstrained allowlists

- If no root token matches the allowlist after many attempts, the command prints an error message
  naming the problem and suggesting you check which part IDs are in the training corpus.
- If the allowlist is very restrictive, generated ships may be small (few parts) but will not crash.
- Check `stop_reason` in the output JSON for `placement_rejected_by_caps_or_anchor_missing`; this
  indicates the allowlist exhausted reachable transitions.
- Use `python scripts/build_markov_generator.py build ... --allowlist ...` for the most reliable
  results: the model will learn transitions only within the allowed set.

## `.ship.png` export

The `export` module (`generators/markov/export.py`) converts generated JSON to a valid Cosmoteer
`.ship.png` file using the inverse of the extraction pipeline in `ship_parser/cosmoteer_ship_encoder.py`.

### How it works

1. Convert `part_id / rotation / x / y` → `ID / Location / Rotation` (Cosmoteer part format).
2. Build a minimal ship dict with sensible defaults (name, FlightDirection, ShipRulesID, colors,
   empty Doors list).
3. Serialize the dict to the Cosmoteer binary object stream (type-tagged nodes with varint lengths).
4. Gzip-compress the stream and prepend the `COSMOSHIP` header + 4-byte length prefix.
5. Generate a solid-gray carrier PNG sized to fit the payload.
6. Embed the payload bytes into the LSBs of the PNG's RGB channels (1 bit per channel).
7. Write the resulting `.ship.png` file.

### Roundtrip validation

Each export is roundtrip-validated by default:
- Re-extract the PNG with the existing parser.
- Compare parts count and per-part (ID, Location, Rotation) identity.
- Compare Version and ShipRulesID fields.
- Report any mismatches or warnings.

### Export limitations

- **No doors synthesised**.  Ships load in-game but crews lack door access to most rooms.
  Door synthesis is a second-pass concern (see `door_rules.py`).
- Colors use default values; ships appear with a stock roof texture.
- Decals, PartControlGroups, PartUIToggleStates, WeaponSelfTargets, and Decals are omitted.
- The carrier PNG thumbnail is a solid-gray placeholder (not a screenshot).
- The encoder only handles value types that actually appear in minimal vanilla ships.  Complex
  nested PartControlGroups / PartUIToggleStates structures are not tested.
- Ships are not tested for in-game gameplay legality (disconnected subgraphs, power, propulsion).

## Artifact expectations

Recommended outputs:
- user-facing summary/report: `for-0neye/`
- generic artifacts/logs/samples: `out/markov/`

Typical files:
- `out/markov/markov-model.v2.json`
- `out/markov/coordinate-validation.v2.json`
- `out/markov/samples-v2/sample-000.json`
- `out/markov/exported-ships/sample-000.ship.png`
- `out/markov/export-report.json`
