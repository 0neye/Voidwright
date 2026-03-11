# Test Guidelines

This directory contains automated tests for the ship generator pipeline.

## Principles

- Add a regression test for every major issue or bug that is found
- Add round-trip tests for major features whenever possible
- Do not chase high coverage numbers for their own sake
- Prioritize unit tests for important underlying infrastructure
- Validate tests against real game data to avoid false assumptions

## Practical Expectations

- **Bug fixes:** Include a test that reproduces the old behavior and proves the fix
- **Feature work:** Prefer end-to-end or round-trip assertions where data is transformed across stages
- **Core modules:** Keep focused unit tests around parser, encoder, geometry, graph, and model infrastructure
- **Data realism:** Use representative real `.ship.png`-derived data (or fixtures based on it) when validating behavior

## Scope Note

Coverage is a signal, not the goal. The goal is confidence that critical behavior is correct and stays correct.
