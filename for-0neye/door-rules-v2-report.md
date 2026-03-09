# Refined door-rule validator report (v2)

## What changed

I added a curated override layer on top of the previously inferred rules in `generators/markov/door_rules.py`.

Override behavior:

- **Hard reject** if a resolved door touches any part whose ID semantically matches **armor**, **structure**, or **wedge**.
- **Crew-room semantic families** are modeled explicitly from the canonical base-game parts:
  - `cosmoteer.crew_quarters_small` → **bunk**
  - `cosmoteer.crew_quarters_med` → **quarters**
  - `cosmoteer.crew_quarters_large` → **barracks**
- The override runs **before** inferred side/pair rules.
  - If an override can decide, it returns `allow` or `reject`.
  - Otherwise validation falls back to the inferred corpus rules.
- Whole-ship validation now exposes **tiered output**:
  - `allow`
  - `reject`
  - `unresolved`

## Explicit corpus part IDs treated as crew-family parts

Curated semantic mapping used for overrides:

- **bunk / small**
  - `cosmoteer.crew_quarters_small`
- **quarters / med**
  - `cosmoteer.crew_quarters_med`
- **barracks / large**
  - `cosmoteer.crew_quarters_large`

Other crew-like IDs observed in the canonical corpus but **not** given curated geometry yet:

- `quarters`
- `ultranova.crew_quarters_walkthrough`
- `janiTNT.1x1quarters`
- `bunk`
- `janiTNT.crew_quarters_4x1`
- `kurim.3x2crewquarters`
- `Kroom.CrewQuarters_1x1`
- `juanTNT.crew_quarters_long`
- `test.crew_quarters_small_c`
- `test.crew_quarters_small_b`
- `janiTNT.crew_quarters_med_long`
- `kurim.4x3crewquarters`
- `ben.CrewQuartersSmall_5`

Observed canonical counts for crew-like IDs:

- `cosmoteer.crew_quarters_med`: 310,856
- `cosmoteer.crew_quarters_small`: 156,077
- `cosmoteer.crew_quarters_large`: 20,229
- `quarters`: 383
- `ultranova.crew_quarters_walkthrough`: 160
- `janiTNT.1x1quarters`: 124
- `bunk`: 94
- `janiTNT.crew_quarters_4x1`: 58
- `kurim.3x2crewquarters`: 30
- `Kroom.CrewQuarters_1x1`: 22
- `juanTNT.crew_quarters_long`: 15
- `test.crew_quarters_small_c`: 8
- `test.crew_quarters_small_b`: 8
- `janiTNT.crew_quarters_med_long`: 4
- `kurim.4x3crewquarters`: 3
- `ben.CrewQuartersSmall_5`: 1

## Orientation / side inference from the real canonical corpus

Using the inferred side signatures already saved in `door-placement-rules.v1.json`, the dominant concrete side family in stored coordinates is:

- **Bunk (`crew_quarters_small`)**
  - Rotation `0/2` (`2x1`): side family = `W@0` plus `N@{0,1}`
  - Rotation `1/3` (`1x2`): side family = `W@{0,1}` plus `N@0`
  - This encodes the requested **three candidate sites** with **max one door total**.
- **Quarters (`crew_quarters_med`)**
  - Rotation `0/2` (`3x2`): curated side = `W@{0,1}`
  - Rotation `1/3` (`2x3`): curated side = `N@{0,1}`
  - This encodes **two positions on one side only**, with **both allowed together**.
- **Barracks (`crew_quarters_large`)**
  - Rotation `0/2` (`4x3`): curated side = `W@{0,2}`
  - Rotation `1/3` (`3x4`): curated side = `N@{0,2}`
  - This encodes the requested **middle-top / middle-bottom on the short sides**, with **both allowed**.

## Validation counts

### Baseline (existing inferred-only v1 artifact)

From `generators/markov/data/door-placement-rules.v1.json`:

- ships: `12,913`
- doors total: `2,058,692`
- pass: `863,167`
- fail: `1,907`
- unresolved: `1,193,618`
- ships with failed observed doors: `1,131`

### Refined v2 revalidation

From `generators/markov/data/door-placement-rules.v2.json` after rerunning validation against the canonical corpus:

- ships: `12,913`
- doors total: `2,058,692`
- allow: `723,986`
- reject: `139,497`
- unresolved: `1,195,209`
- ships with rejects: `11,746`
- ships with unresolved: `12,629`

## Interpretation

The override layer is working mechanically, but the current curated crew-part interpretation is **too strict to use as a production reject filter yet**.

Why the reject count jumped so much:

1. The semantic crew-room geometry is much narrower than the broad inferred corpus evidence.
2. The current boundary resolver is still approximate for many ambiguous canonical door situations.
3. When the resolver assigns one side of a door to a curated crew-room signature that falls outside the curated site list, the override now produces a hard `reject`.
4. That means the override is currently acting more like an **audit lens** than a safe runtime filter.

## Remaining ambiguities

- The stored coordinate frame for crew rooms is now mapped to concrete sides **empirically**, but the corpus still contains many observed doors that disagree with those strict semantic positions.
- Some disagreements are likely real corpus noise/modded content.
- Some are likely due to the current occupied-rectangle boundary resolver, especially around ambiguous multi-part boundaries and non-rectangular parts.
- Crew-like modded parts have not been given curated geometry yet and remain outside the explicit semantic mapping.

## Recommendation

**Do one more refinement first before generator implementation.**

Specifically:

1. Add an **override audit mode** that records when curated crew rules disagree with inferred evidence without immediately hard-rejecting all such cases.
2. Sample and inspect the highest-volume crew-room rejects to separate:
   - true semantic violations
   - resolver mistakes
   - modded/corpus exceptions
3. Only then promote the crew-room overrides into a strict runtime reject filter.

## Artifacts

- Original inferred rules: `generators/markov/data/door-placement-rules.v1.json`
- Refined rules with overrides: `generators/markov/data/door-placement-rules.v2.json`
