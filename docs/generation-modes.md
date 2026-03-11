# Generation Modes

## Allowlists

The generator supports allowlists at both build time and generation time.

Build-time allowlist:

- Use `--allowlist` or `--allowlist-file` with `build`
- Excluded parts are removed from the training sequences
- This produces the cleanest focused model for a restricted part set

Generation-time allowlist:

- Use `--allowlist` or `--allowlist-file` with `generate`
- Sampled tokens are filtered at runtime without rebuilding the model
- This is faster for experiments, but restrictive allowlists can make generation sparse or short

Practical guidance:

- Prefer build-time allowlists for stable experiments
- Use generation-time allowlists for quick iteration
- If generation repeatedly ends with `placement_rejected_by_caps_or_anchor_missing`, the allowlist may be too restrictive for the learned transitions

## Mirror symmetry

`--mirror-symmetry` enables strict left-right generation-time symmetry without retraining the model.

Axis convention:

- The mirror axis is fixed at `x = -0.5`
- Primary placements live entirely on the left half where all footprint cells satisfy `x <= -1`
- Mirrored placements live on the right half where all footprint cells satisfy `x >= 0`
- No part is allowed to straddle the axis

Mirror transform:

- `mirror_x = -origin_x - width`
- `mirror_y = origin_y`
- Rotation usually mirrors as `mirror_rotation = (4 - rotation) % 4`
- Wedges (`armor_wedge`, `structure_wedge`, `armor_structure_hybrid_1x1`) use a handedness swap mapping `{0:1, 1:0, 2:3, 3:2}`
- Triangle half-cells (`armor_tri`, `structure_tri`, `armor_structure_hybrid_tri`) keep the same rotation and mirror through `FlipX`
- Mirroring toggles `FlipX` and preserves `FlipY`

Generation rules:

1. The root is placed flush against the axis on the left half
2. Each accepted primary placement must also have a valid mirror placement
3. Primary and mirror placements are committed atomically
4. Only primary-side parts are used as future Markov anchors

Important semantics:

- `--max-parts` counts total parts, not unique left-side parts
- Mirror mode tends to terminate earlier because a placement is rejected if either side fails
- Increasing `--max-attempts` helps compensate for the stricter acceptance rule

## Part requirements

Part requirements let generation continue until specific part counts are reached or the attempt budget is exhausted.

CLI forms:

- Repeat `--require PART_ID COUNT`
- Or use `--requirements-file`

Supported file formats:

- JSON object: `{"part_id": count, ...}`
- Plain text lines: `PART_ID COUNT`
- `#` comments are allowed in the plain text format

Runtime semantics:

- When the sampler wants to emit END, the runtime checks whether all requirements are satisfied
- If not, END is suppressed and generation continues
- This does not force the model to sample a missing part; it only prevents early stopping

Counting rules:

- Requirement counts use total-ship semantics
- In mirror mode, both the primary part and its mirror count toward the requirement total

Useful stats:

- Output JSON includes `stats.requirements`
- Requirement-driven END suppression increments `stats.rejections.requirements`
- If attempts run out first, stop reason can become `max_attempts_requirements_unsatisfied`

Practical guidance:

- Requirements work best for parts that already appear with moderate frequency in the model
- Rare parts are still hard to guarantee because the transition distribution itself is unchanged

## Seeded generation

Seeded generation starts from an existing layout and lets the Markov chain grow from that occupied state.

Input forms:

- `--seed-json <generated-sample.json>`
- `--seed-json <canonical-corpus ship.json>`
- `--seed-png <ship.ship.png>`

Seed loading behavior:

- Generated sample JSON is detected by a `parts` key
- Extracted Cosmoteer JSON is detected by a `Parts` key
- `.ship.png` input is parsed through the existing extraction path

Seed placement rules:

- Vanilla parts with known geometry are placed first
- Parts with unknown geometry are skipped
- Parts excluded by an allowlist are skipped
- Overlapping seed parts are skipped

Runtime behavior:

- The seed creates the initial occupied map
- Generation then samples a virtual root and continues using the placed seed parts as the anchor pool
- Output JSON includes `stats.seed` with counts for placed and skipped seed parts

Best seeds:

- Compact layouts with common vanilla parts
- Layouts dominated by common anchors such as armor, corridors, structure, and common rooms
- Previously generated ships from this model

Poor seeds:

- Large seeds that leave little room to grow
- Seeds dominated by rare or exotic anchor types
- Seeds containing many unknown-geometry or non-vanilla parts

## Mirror symmetry plus seeds

Seed placement happens before mirror-mode generation logic takes over.

Important rule in mirror mode:

- Only seed parts whose entire footprint lies on the primary left half can become active anchors for continued generation
- Seed parts on the right half can exist as occupied cells but are not useful for further growth

Practical implication:

- Mirror-mode seeds should be authored on the left half only
- Non-symmetric full-ship seeds are usually poor starting points for mirror-mode continuation
