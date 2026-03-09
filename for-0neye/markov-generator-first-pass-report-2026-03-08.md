# First-pass vanilla-only Markov generator report (2026-03-08)

## What was built

I implemented a conservative first-pass **vanilla-only relative-placement Markov generator** under `generators/markov/`.

Main pieces:
- `generators/markov/model.py`
  - training/build logic from the canonical corpus only
  - relative-placement tokenization with rotation and anchor-relative `(dx, dy)`
  - END-token support
  - generation hard caps
  - footprint-aware overlap rejection using vanilla game-file geometry
  - real-corpus coordinate validation helpers
- `generators/markov/cli.py`
  - `build`, `generate`, and `validate` subcommands
- `scripts/build_markov_generator.py`
  - thin runnable entrypoint
- `generators/markov/README.md`
  - model details, CLI, limitations, and artifact conventions
- `README.md`
  - repo-level mention of the Markov generator pipeline

## Generator architecture

Training representation:
- filter ship parts to **vanilla only** (`cosmoteer.*`) and require known geometry from `generators/markov/data/vanilla-parts-from-game-files.json`
- choose a root part near the ship centroid
- order remaining parts with a conservative nearest-anchor heuristic
- emit tokens of the form:
  - root: `(part_id, rotation, __ROOT__, 0, 0, 0)`
  - placement: `(part_id, rotation, anchor_part_id, anchor_rotation, dx, dy)`
  - END token
- Markov state = recent token history (order configurable; default 2)

Generation behavior:
- sample a root
- sample next placement tokens from the current Markov state
- resolve to an already placed anchor with matching part ID + rotation
- place the new part at `anchor_origin + (dx, dy)`
- reject placements when:
  - the anchor is unavailable
  - the new part would overlap existing occupied footprint cells
  - the new part would exceed configured world bounds
- stop on END or hard caps (`max_parts`, `max_attempts`, `max_resample_per_step`, bounds)

## Validation performed

Real-corpus validation artifact produced:
- `out/markov/coordinate-validation.sample500.v2.json`

Sample-500 validation results:
- ships checked: **500**
- placements checked: **1,000,392**
- largest checked ship vanilla part count: **55,754**
- origin reconstruction failures: **0**
- footprint reconstruction failures: **0**
- touching placements: **811,041**
- non-touching placements: **189,351**
- touching fraction: **0.8107**
- max abs dx: **598**
- max abs dy: **8092**

Interpretation:
- the extracted `Location` coordinates behave consistently as stable part origins for this relative-placement scheme
- reconstructing from anchor-relative `(dx, dy)` exactly reproduces both part origins and full world footprint cells on real corpus ships
- the nearest-anchor ordering is often, but not always, physically touching (~81% touching in this 500-ship validation), which is acceptable for a conservative first pass

## Artifact paths

Code/docs:
- `generators/markov/model.py`
- `generators/markov/cli.py`
- `generators/markov/README.md`
- `scripts/build_markov_generator.py`

Validation:
- `out/markov/coordinate-validation.sample500.v2.json`

Existing full-corpus Markov artifact already present in repo state:
- `out/markov/markov-model.v1.json`

Sample outputs generated with the upgraded footprint-aware sampler:
- `out/markov/samples-v1-footprint/sample-000.json`
- `out/markov/samples-v1-footprint/sample-001.json`
- `out/markov/samples-v1-footprint/sample-002.json`

## Sample generation notes

Smoke-generation from the existing full-corpus artifact produced these first-pass layout samples:
- `sample-000.json`: 17 parts, stop=`end_token`
- `sample-001.json`: 36 parts, stop=`placement_rejected_by_caps_or_anchor_missing`
- `sample-002.json`: 10 parts, stop=`placement_rejected_by_caps_or_anchor_missing`

These are generator-oriented layout artifacts, not ready-to-play `.ship` exports.

## Known limitations / deferred second pass

Still intentionally deferred:
- door placement during generation
- pathfinding / accessibility cleanup
- connectivity cleanup
- stronger gameplay legality rules
- direct `.ship` export

Practical current limitation:
- the previously built `markov-model.v1.json` predates the stricter geometry-backed vanilla filtering, so the runtime now defensively skips candidates that do not have known vanilla geometry
- a stricter rebuilt artifact should be generated with the current code path and used as the new default once that build is allowed to finish

## Recommended next step

Run a full strict rebuild with the new code and then sample from that artifact:

```bash
python scripts/build_markov_generator.py build \
  --input-dir extracted_ship_data_canonical \
  --output out/markov/markov-model.v2.json \
  --validation-output out/markov/coordinate-validation.v2.json

python scripts/build_markov_generator.py generate \
  --model out/markov/markov-model.v2.json \
  --output out/markov/samples-v2 \
  --count 10 \
  --seed 1337
```

After that, the exact next engineering step should be:
- add a **second-pass structural/accessibility filter** that consumes generated layouts and prunes/rejects obviously disconnected or inaccessible results before any door synthesis work
