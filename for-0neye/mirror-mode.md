# Mirror mode — ship generator update

**Date:** 2026-03-08
**Priority item:** #1 — symmetry handling / mirror mode

---

## What was built

Left-right mirror symmetry is now fully implemented in the first-pass Markov generator.
Ships are enforced to be **exactly symmetric** at the cell level across a vertical center axis.

The implementation is **generation-time only** — the existing trained model is reused as-is.
No retraining is needed.

---

## How it works

### Axis convention

The mirror axis sits at **x = −0.5**, between grid columns −1 and 0.

- **Primary (left) half:** all footprint cells at x ≤ −1
- **Mirror (right) half:** all footprint cells at x ≥ 0
- No part ever straddles the axis; the two halves are always disjoint

The root part is placed flush against the axis so its rightmost cell sits at x = −1.

### Mirror transform

For a part at origin `(ox, oy)` with rotation `r` and width `W` (for that rotation):

```
mirror_x        = -ox - W
mirror_y        = oy              (y unchanged)
mirror_rotation = (4 - r) % 4    → 0↔0, 1↔3, 2↔2, 3↔1
```

This reversal of CW handedness (`1↔3`) is what correctly flips the visual orientation of
directional parts like wedge armor, thrusters, and weapons.

### What "exact symmetry" means

After generation, every occupied grid cell `(cx, cy)` has a corresponding cell `(-cx - 1, cy)`
also occupied.  This was verified across all generated samples — **zero asymmetric cells**.

### Generation algorithm

1. Sample root from Markov model → place at `x = -W` (flush left of axis) + place mirror at `x = 0`.
2. At each step: sample a token, find a valid primary anchor, compute candidate placement.
3. Check candidate is on left half; check its mirror is valid too (no overlap, in bounds).
4. If both are valid: commit primary **and** mirror atomically.  If either fails: reject both.
5. Only primary parts are added to the Markov anchor pool — the model builds on the left side,
   and mirrors follow automatically.

---

## CLI usage

```bash
# Generate 20 mirrored ships, export to .ship.png
python scripts/build_markov_generator.py generate \
    --model out/markov/markov-model.v2.json \
    --output out/markov/samples-mirror \
    --export-png-dir out/markov/exported-ships-mirror \
    --count 20 \
    --max-parts 200 \
    --max-attempts 6000 \
    --mirror-symmetry \
    --seed 2025
```

Key flag: `--mirror-symmetry`
`--max-parts 200` gives ~100 unique shapes on each side.

---

## Sample output

**JSON samples:** `out/markov/samples-mirror/sample-000.json` … `sample-019.json`
**PNG ships:**   `out/markov/exported-ships-mirror/sample-000.ship.png` … `sample-019.ship.png`
**Selection for review:** `for-0neye/mirror-samples/*.ship.png` (12 ships)

All 20 ships:
- Passed roundtrip encode → decode validation (OK)
- Have exactly equal primary and mirror part counts
- Zero asymmetric cells verified by automated check

Example stats from the run:

| sample | total parts | primary | mirror | stop reason |
|--------|------------|---------|--------|-------------|
| 001    | 56         | 28      | 28     | end_token |
| 002    | 200        | 100     | 100    | max_parts |
| 006    | 200        | 100     | 100    | max_parts |
| 016    | 116        | 58      | 58     | end_token |
| 018    | 182        | 91      | 91     | end_token |

---

## Files changed / created

| file | change |
|------|--------|
| `generators/markov/symmetry.py` | **NEW** — mirror transform helpers (`mirror_part`, `primary_root_x`, `verify_mirror_footprint`) |
| `generators/markov/model.py` | Added `mirror_symmetry: bool` to `GenerationConfig`; rewrote `generate()` with mirror-mode branch |
| `generators/markov/cli.py` | Added `--mirror-symmetry` flag; mirror stats in per-sample print |
| `generators/markov/README.md` | Added "Mirror symmetry mode" section with axis convention, rotation table, limitations |
| `out/markov/samples-mirror/` | 20 generated JSON samples |
| `out/markov/exported-ships-mirror/` | 20 `.ship.png` files |
| `for-0neye/mirror-samples/` | 12 curated `.ship.png` files for review |

---

## Known limitations / caveats

1. **Early termination is more common** than in non-mirror mode.  The Markov model was
   not trained to know which left-side placements will have a valid mirrored right side;
   rejected mirrors cause the primary placement to be rejected too, which can exhaust the
   attempt budget faster.  Ships therefore tend to be smaller than non-mirror runs.
   Workaround: increase `--max-attempts` (e.g. `--max-attempts 6000`).

2. **Axis is fixed at x = −0.5.**  The ship is always centered on this column gap.
   There is no option to shift the axis (e.g. to center on a single column).

3. **Only left-right symmetry.**  Top-bottom and radial symmetry are not implemented.

4. **No doors.**  Like the baseline generator, mirror mode produces no door placements.
   Crew will not be able to traverse most rooms until the door-synthesis pass is added
   (priority #3 on the roadmap).

---

## Recommended next steps

**Priority #2 — part-requirement solving**
After seeing the mirrored output, the most immediate quality issue is that ships often
lack functional parts (control room, reactor, thrusters, etc.).  Implementing a
"must-have parts" constraint layer on top of the Markov sampler would make ships
game-legal.  Suggested approach: post-generation requirement checking with targeted
forced insertions, or a two-phase generation (skeleton of required parts first, then
Markov fill-in).

**Priority #3 — door placement and pathing**
The door-rules infrastructure already exists in `door_rules.py`.  The next step is
to wire it into a post-generation pass that inserts doors between adjacent accessible
tiles and verifies crew connectivity.

**Mirror + requirements interaction**
When requirement solving is added, each required part placed on the left half should
automatically get its mirror on the right half, ensuring the ship remains symmetric
after the requirement pass.  This is a natural fit for the current architecture since
mirror placement is a simple transform applied after any primary placement.
