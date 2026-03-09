# Requirements-Constrained and Seeded Generation

**Date:** 2026-03-09
**Build:** `out/markov/markov-model.v2.json`

---

## What Was Built

Two new generation features layered on top of the existing first-pass Markov generator:

1. **Part requirements** — generalized `(part_id → min_count)` constraints that keep the generator running until specific parts appear (or the attempt budget runs out).
2. **Seeded generation** — start from an existing ship layout (generated JSON or `.ship.png`) and let the Markov chain grow additional structure from there.

Both features integrate with mirror symmetry and are exposed through the CLI.

---

## Feature 1: Part Requirements

### What it does

Requirements let you say: *"this ship must have at least N of part X somewhere on it."*

When the Markov sampler would naturally stop (by emitting `END_TOKEN`), it checks whether all requirements are satisfied first. If they aren't, `END_TOKEN` is suppressed and generation continues. This makes the ship grow until requirements are met or the attempt budget (`--max-attempts`) is exhausted.

**Count semantics: total ship count** (primary + mirror halves both count). So in mirror mode, one primary `crew_quarters_med` placement gives you 2 total, satisfying a requirement of 2.

### CLI usage

```bash
# Require at least 1 control room (small), 2 missile launchers, 3 crew quarters
python scripts/build_markov_generator.py generate \
  --model out/markov/markov-model.v2.json \
  --output out/markov/samples-requirements \
  --count 8 \
  --max-parts 400 \
  --max-attempts 15000 \
  --seed 2000 \
  --require cosmoteer.crew_quarters_med 3 \
  --require cosmoteer.missile_launcher 2 \
  --export-png-dir out/markov/exported-ships-requirements
```

`--require` may be repeated for any number of parts.

**Requirements file** (JSON format):
```json
{
  "cosmoteer.crew_quarters_med": 3,
  "cosmoteer.missile_launcher": 2,
  "cosmoteer.shield_gen_small": 1
}
```

```bash
python scripts/build_markov_generator.py generate \
  --model out/markov/markov-model.v2.json \
  --output out/markov/samples-requirements \
  --requirements-file my-requirements.json \
  ...
```

Plain-text format also supported (`PART_ID COUNT` per line, `#` comments OK):
```
# minimum combat loadout
cosmoteer.crew_quarters_med 3
cosmoteer.missile_launcher 2
```

### Demo results

Run: 8 samples with `crew_quarters_med ≥ 3` and `missile_launcher ≥ 2`, seeds 2000–2007:

| Sample | Parts | Stop reason | Requirements |
|--------|-------|-------------|--------------|
| 000 | 400 | max_parts | UNMET (missile_launcher=1) |
| 001 | 400 | max_parts | UNMET (missile_launcher=1) |
| 002 | 11 | placement_rejected | UNMET |
| **003** | **270** | **end_token** | **OK** |
| **004** | **140** | **end_token** | **OK** |
| 005 | 16 | placement_rejected | UNMET |
| **006** | **400** | **max_parts** | **OK** |
| **007** | **400** | **max_parts** | **OK** |

4/8 satisfied. The generator is **working correctly**: it suppresses END_TOKEN until satisfied. Whether it satisfies depends on whether the Markov chain naturally produces those parts in that run.

Output files: `out/markov/samples-requirements/`, `out/markov/exported-ships-requirements/`

### Mirror mode + requirements

```bash
python scripts/build_markov_generator.py generate \
  --model out/markov/markov-model.v2.json \
  --output out/markov/samples-mirror-requirements \
  --count 5 \
  --max-parts 400 \
  --max-attempts 15000 \
  --seed 5000 \
  --mirror-symmetry \
  --require cosmoteer.crew_quarters_med 4 \
  --require cosmoteer.shield_gen_small 2 \
  --export-png-dir out/markov/exported-ships-mirror-requirements
```

In mirror mode: 1 primary `crew_quarters_med` → 2 total (counts toward the 4 requirement). So the requirement of 4 means you need 2 primary placements. Sample-002 satisfied both requirements (400 parts, max_parts, reqs=OK).

Output files: `out/markov/samples-mirror-requirements/`, `out/markov/exported-ships-mirror-requirements/`

### Honest limitations

Requirements work by **suppressing END_TOKEN** — they cannot force the Markov sampler to generate specific parts it doesn't naturally sample. If the model never samples `cosmoteer.reactor_small` in a particular run, no amount of requirement-pressure will add one. Parts that appear rarely in the corpus (control rooms, reactors) are hard to guarantee.

**Best results:** use requirements for parts that appear with moderate frequency in generated ships — crew quarters, shield generators, missile launchers, thrusters. Check `model.part_frequency` to get a sense of what the corpus provides.

Stats in the output JSON include `requirements.progress` showing exactly which parts were satisfied and by how much:

```json
"requirements": {
  "satisfied": true,
  "progress": {
    "cosmoteer.crew_quarters_med": {"required": 3, "actual": 5, "satisfied": true},
    "cosmoteer.missile_launcher": {"required": 2, "actual": 3, "satisfied": true}
  }
}
```

---

## Feature 2: Seeded Generation

### What it does

Instead of starting from a freshly-sampled root, you provide an existing ship and the generator places all its vanilla parts first (establishing the collision map), then continues growing the ship using the Markov chain.

**Semantics:** the seed is the initial occupied state. The Markov chain picks up from a "virtual root" whose part_id matches a part in the seed, so transitions are anchored to real seed parts. The seed parts serve as the anchor pool for all subsequent placements.

### CLI usage

**From a generated ship JSON:**
```bash
python scripts/build_markov_generator.py generate \
  --model out/markov/markov-model.v2.json \
  --output out/markov/samples-seeded \
  --count 4 \
  --max-parts 250 \
  --max-attempts 8000 \
  --seed 3000 \
  --seed-json out/markov/samples-v3-export/sample-002.json \
  --export-png-dir out/markov/exported-ships-seeded
```

**From a Cosmoteer extracted ship JSON** (canonical corpus format):
```bash
python scripts/build_markov_generator.py generate \
  --model out/markov/markov-model.v2.json \
  --output out/markov/samples-seeded \
  --count 4 \
  --max-parts 250 \
  --seed 4000 \
  --seed-json "extracted_ship_data_canonical/Recce_Mk._II.ship.json"
```

Both `--seed-json` formats are auto-detected: if the file has a `parts` key (our generated format), it reads directly. If it has a `Parts` key (Cosmoteer extracted format), it converts automatically.

**From a `.ship.png` directly:**
```bash
python scripts/build_markov_generator.py generate \
  --model out/markov/markov-model.v2.json \
  --output out/markov/samples-seeded \
  --count 4 \
  --seed-png path/to/ship.ship.png
```

Non-vanilla parts and parts with unknown geometry are silently skipped. Overlapping parts (can occur when some seed parts were excluded) are also skipped.

### Demo results

**Seed: generated ship (sample-002.json, 35 parts, mostly armor_wedge + armor_2x1)**

Seeds 3000–3003:

| Sample | Seed placed | Total parts | Stop reason |
|--------|-------------|-------------|-------------|
| 000 | 35 | 35 | placement_rejected |
| **001** | **35** | **166** | **end_token** |
| **002** | **35** | **155** | **end_token** |
| **003** | **35** | **105** | **end_token** |

3/4 samples successfully grew from the seed. Ships of 105–166 total parts built on the 35-part foundation.

**Seed: canonical corpus ship (Recce_Mk._II.ship.json, 26 parts, 18 vanilla)**

Seeds 4000–4003: 1/4 succeeded (250 parts), 3/4 failed due to exotic anchor types in the seed.

Output files: `out/markov/samples-seeded/`, `out/markov/exported-ships-seeded/`

The `stats.seed` field in the output JSON shows what happened:
```json
"seed": {
  "seed_parts_input": 26,
  "seed_parts_placed": 18,
  "seed_skipped_geometry": 8,
  "seed_skipped_overlap": 0,
  "seed_skipped_allowlist": 0
}
```

### Combined seeded + requirements

```bash
python scripts/build_markov_generator.py generate \
  --model out/markov/markov-model.v2.json \
  --output out/markov/samples-seeded \
  --count 4 \
  --max-parts 300 \
  --max-attempts 10000 \
  --seed 6000 \
  --seed-json out/markov/samples-v3-export/sample-001.json \
  --require cosmoteer.crew_quarters_med 5 \
  --export-png-dir out/markov/exported-ships-seeded
```

Works: seed establishes initial layout, requirements suppress early stopping.

### Mirror mode + seed

Seeds are placed before mirror mode takes effect. In mirror mode:
- Seed parts with all footprint cells at `x ≤ -1` become **primary anchors** (the Markov chain can build off them)
- Seed parts at `x ≥ 0` are pre-placed but NOT anchors (treated as pre-existing mirror placements)

**Practical implication:** for mirror mode seeding, your seed should contain parts on the primary (left) half only (`x ≤ -1`). Seeds from non-symmetric ships in mirror mode will have some parts ignored as anchors, reducing continuability.

### Honest limitations

**The anchor-matching problem:** The Markov model generates new parts relative to SPECIFIC anchor types (e.g., "place `corridor` next to `armor_2x1|0`"). If the seed contains unusual parts (exotic thrusters, rare weapons) that rarely appear as anchors in the training data, the generator can't build off them — all resamples fail with `missing_anchor` and the ship stops immediately.

**Best seed types for maximum continuation:**
- Ships with common parts: `armor_2x1`, `armor_wedge`, `corridor`, `structure`, `crew_quarters_med`
- Our own generated ships (which by construction use the most-common corpus parts)
- Compact seeds (fewer parts = more free space for new placements)

**Worst seed types:**
- Ships with exotic/rare parts as their majority
- Very large seeds that fill most of the bounds (no room to grow)
- Seeds with parts not in the vanilla geometry cache

---

## Stats reference

Generated JSON now includes additional stats fields:

```json
{
  "stats": {
    "parts_generated": 270,
    "stop_reason": "end_token",
    "rejections": {
      "missing_anchor": 45,
      "overlap": 120,
      "bounds": 8,
      "allowlist": 0,
      "requirements": 12
    },
    "requirements": {
      "satisfied": true,
      "progress": {
        "cosmoteer.crew_quarters_med": {"required": 3, "actual": 5, "satisfied": true},
        "cosmoteer.missile_launcher": {"required": 2, "actual": 2, "satisfied": true}
      }
    },
    "seed": {
      "seed_parts_input": 35,
      "seed_parts_placed": 35,
      "seed_skipped_geometry": 0,
      "seed_skipped_overlap": 0,
      "seed_skipped_allowlist": 0
    }
  }
}
```

New `stop_reason` values:
- `max_attempts_requirements_unsatisfied` — exhausted attempts without meeting requirements
- `requirements_unsatisfied` — fallback safety (shouldn't occur in practice)

---

## Summary of Changes

### Files modified
- `generators/markov/model.py` — `GenerationConfig` new field `part_requirements`, `generate()` signature `generate(config, *, seed_parts=None)`, seed handling logic, requirements suppression of END_TOKEN, updated stats output
- `generators/markov/cli.py` — `_load_requirements()` helper, `_load_seed_parts_from_json()`, `_load_seed_parts_from_png()`, new `generate` args (`--require`, `--requirements-file`, `--seed-json`, `--seed-png`), updated `cmd_generate()`

### Files added
- `for-0neye/requirements-and-seeds.md` (this file)

### Outputs generated
- `out/markov/samples-requirements/` (8 samples, 4 reqs-satisfied)
- `out/markov/exported-ships-requirements/` (PNG exports)
- `out/markov/samples-seeded/` (mixed seeds, 3-4/8 successful growths)
- `out/markov/exported-ships-seeded/` (PNG exports)
- `out/markov/samples-mirror-requirements/` (5 samples, 1 satisfied)
- `out/markov/exported-ships-mirror-requirements/` (PNG exports)

---

## Next recommended step

**Requirements satisfaction rate is low for rare parts** (control rooms, reactors, specific thruster types). The current implementation can only suppress END — it cannot bias sampling toward required parts.

A useful next step would be a **requirements-aware resampling bias**: when requirements are unsatisfied, upweight transitions that produce required parts. This could be a weighted blend of the current Markov distribution and a "required parts" prior.

Alternatively, a **post-hoc insertion pass** could find free cells adjacent to existing parts and try to insert required parts there, ignoring the Markov sequence order. This would have much higher satisfaction rates for isolated parts (control rooms, reactors).

**Seeded generation quality** depends heavily on seed part composition. A preprocessing step that scores seeds by "continuability" (fraction of parts with known common transition states) would help users pick good seeds.
