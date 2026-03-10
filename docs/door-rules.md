# Door Rules

## Purpose

`generators/markov/door_rules.py` contains the reusable door-placement inference and validation logic
for later generation passes. The current system is intentionally conservative and keeps a distinction
between confident decisions and unresolved cases.

## Validation outputs

Whole-ship and single-door validation use three decision levels:

- `allow` - enough evidence exists to accept the door location
- `reject` - enough evidence exists to reject the door location
- `unresolved` - current evidence or geometry resolution is intentionally insufficient

This distinction matters because unresolved is not the same as invalid.
Future agents should treat it as "not yet modeled confidently."

## Rule sources

Validation prefers curated overrides first and then falls back to inferred corpus rules.

Override layer:

- Runs before inferred side and pair rules
- Can immediately return `allow`, `reject`, or `unresolved`
- Is intentionally scoped to vanilla `cosmoteer.*` part IDs for this phase

Inferred layer:

- Uses rules learned from the canonical deduped corpus
- Includes side rules and pair rules
- Produces the main machine-readable artifact used by later steps

Default artifact path:

- `generators/markov/data/door-placement-rules.v2.json`

Geometry source of truth for vanilla parts:

- `generators/markov/data/vanilla-parts-from-game-files.json`

This geometry is used as the authoritative vanilla footprint and walkability source for the current refinement path.

## Curated override behavior

Hard-reject classes:

- Armor
- Structure
- Wedge-like parts

Vanilla crew-room families with explicit curated door sites:

- `cosmoteer.crew_quarters_small`
- `cosmoteer.crew_quarters_med`
- `cosmoteer.crew_quarters_large`

The override layer also enforces per-part door-count caps for these curated crew rules.

Important caution:

- The curated crew-room interpretation is stricter than broad historical corpus evidence
- It is useful as an audit and semantics layer
- It may still be too strict to treat as a production reject filter without further refinement

## Vanilla-only safety scope

The generator-safe layer is intentionally vanilla-only.

- Only `cosmoteer.*` part IDs are treated as vanilla for this phase
- Non-vanilla parts are excluded from the generator-safe semantic layer
- Doors touching non-vanilla context should be treated as `unresolved`, not force-fit
- Curated semantic overrides apply only to vanilla parts

This is a deliberate safety tradeoff: better conservative coverage for vanilla generation, not broad mod compatibility

## Resolver fallback for shared-cell edge cases

The code includes a narrow fallback for a previously large `resolver_none` bucket.

This fallback only runs when the normal occupied-cell boundary resolver cannot produce a clean answer.
It handles cases where:

- one vanilla part appears to cover both cells of the doorway span
- another vanilla part only shares one side cell
- the strict boundary model cannot emit two distinct adjacent boundaries
- the touched vanilla part still matches exported `allowed_door_locations`

Fallback behavior:

1. Detect shared-cell-overlap cases
2. Match the doorway back to vanilla `allowed_door_locations`
3. Allow a very small anchor-drift search
4. Return `allow` only if there is still a distinct neighboring vanilla part on the opposite side

This fallback is intentionally narrow because it is meant to remove false unresolveds without broadening the accept policy recklessly.

## Corpus behavior and unresolved caveat

Inference and validation stream one canonical ship JSON at a time instead of loading the full corpus into memory.

Important current caveat:

- Authoritative vanilla geometry improved raw resolution substantially
- But a large unresolved bucket still remains on vanilla-only ships
- The dominant remaining blocker is not simply "modded noise"

The unresolved mass now points at a deeper coordinate or semantics question around one or more of:

- how ship-file `Part.Location` anchors map onto game-file local geometry
- how ship-file `Door.Cell` maps onto game-file `allowed_door_locations`
- whether some observed doors behave like legal interior access sites on a single multi-tile part

Practical implication:

- The current door tooling is strong enough for conservative vanilla-first validation
- It is not yet a complete or fully solved general door model
- Future changes in this area should be careful not to guess at transforms without corpus-backed verification

## What is safe to assume

Future agents can safely assume:

- Door validation is vanilla-first and conservative
- Non-vanilla parts are outside the curated semantic model and often remain unresolved
- Inferred rules are backed by canonical-corpus evidence
- Curated overrides intentionally bias toward safe interpretation rather than coverage

Future agents should not assume:

- Every unresolved door is a real problem
- Crew-room overrides are final gameplay truth
- The current validator is ready to act as a complete post-generation legality pass

## Rebuilding rules

Regenerate the rule artifact from the canonical corpus with:

```bash
python scripts/infer_door_rules.py \
  --input-dir extracted_ship_data_canonical \
  --output generators/markov/data/door-placement-rules.v2.json
```

Threshold knobs:

- `--min-side-observations`
- `--min-side-ratio`
- `--min-pair-observations`
- `--min-pair-ratio`

Use those only when intentionally changing inference sensitivity; they affect the learned rule surface.
