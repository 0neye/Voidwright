# Part anchor / resolver_none validation update (2026-03-08, fallback pass)

## Result

Full streaming validation over `extracted_ship_data_canonical` now gives:

- ships: `12913`
- doors total: `2058692`
- allow: `1445798`
- reject: `539235`
- unresolved: `73659`
- ships with rejects: `12573`
- ships with unresolved: `3040`

Delta from the previous validated state:

- allow: `1280455 -> 1445798` (`+165343`)
- reject: `539235 -> 539235` (`no change`)
- unresolved: `239002 -> 73659` (`-165343`)

## What `resolver_none` actually was

The big `resolver_none` bucket was mostly **not** a remaining corpus-rule ambiguity.

It was dominated by a **resolver/model edge case** where:

- one vanilla part appears to cover **both** cells of the doorway span,
- a second vanilla part shares only one side cell,
- so the strict boundary resolver cannot emit two distinct adjacent boundaries,
- but the door still matches the touched vanilla part's exported `allowed_door_locations`.

Typical shape:

- one side cell: `shield_gen_small`
- other side cell: `shield_gen_small + corridor` (or conveyor / storage / power)

This same pattern also showed up heavily for:

- `disruptor`
- `ion_beam_emitter`
- `point_defense`
- `cannon_large`
- `laser_blaster_large`
- `chaingun`

## Patch applied

`generators/markov/door_rules.py` now includes a narrow fallback that only runs when the normal occupied-cell boundary resolver fails.

The fallback:

1. looks for shared-cell-overlap cases,
2. matches the door back to vanilla `allowed_door_locations`,
3. allows a tiny local anchor-drift search,
4. only returns `allow` when there is still a distinct neighboring vanilla part on the opposite side.

This removed the huge false-unresolved bucket without changing reject behavior.

## Exact unresolved breakdown

- bunk/quarters historical-drift: `0`
- other vanilla: `1945`
- modded: `5701`
- resolver_none residual: `66013`

Residual `resolver_none` reason split:

- `a_empty`: `45980`
- `b_empty`: `19728`
- `both_occupied_no_boundary_match`: `5968`

Residual top touched vanilla parts inside `resolver_none`:

- `cosmoteer.cannon_large`: `6758`
- `cosmoteer.flak_cannon_large`: `2542`
- `cosmoteer.cannon_med`: `1028`
- `cosmoteer.chaingun`: `1010`
- `cosmoteer.shield_gen_small`: `438`
- `cosmoteer.storage_2x2`: `339`
- `cosmoteer.storage_3x2`: `121`

## Readiness

At this point I would say **yes: we are finally ready for the first-pass Markov generator**, with one caveat:

- the remaining unresolved mass is no longer dominated by the previously mysterious vanilla shared-cell cases,
- but there is still a cleanup opportunity in the residual cannon / chaingun / storage `resolver_none` cluster and the small `other_vanilla` tail.

That should no longer block a first vanilla training pass.
