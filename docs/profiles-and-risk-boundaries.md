# Profiles and Risk Boundaries

## Why Profiles Exist

Users do not always optimize for the same thing.

Sometimes the right answer is:
- preserve validation strength
- refuse hidden fallbacks
- ship only when the structure is still tight

Sometimes the right answer is:
- cross a boundary on purpose
- get faster or cheaper
- measure whether the boundary crossing was actually worth it

`omegacal` supports both with one engine.

## Guarded

Default for beginners.

Priorities:
- structural integrity
- strong validation
- visible safety boundaries

Typical behavior:
- blocks non-ship-grade LLM judges
- blocks unsupported strict-schema requests instead of silently degrading
- uses stricter default walk-forward thresholds
- recommends `hold` or `block` quickly when risk is visible

## Expedition

Used when efficiency-adjusted uplift may justify higher structural risk.

Typical behavior:
- allows controlled schema degradation
- allows experimental adapters
- records each relaxed safeguard
- records the extra uplift gained by crossing guarded boundaries
- downgrades ship advice when the result is still exploration-grade

## Beginner-Friendly Risk Language

Warnings are grouped into:
- `safety boundary`: a requested control or guarantee was weakened
- `validation strength`: the judge path is not strong enough for ship-grade trust
- `experimental risk`: the adapter or pathway is exploratory
- `deployment readiness`: held-out validation is missing or failed

## Structural Fatigue Signals

The tool is designed to surface these explicitly:
- weak judges being used like ship-grade judges
- strict schema degrading to JSON validation
- missing or failed walk-forward validation
- placeholder providers being used as if they were real
- local capability fallbacks being hidden

## How to Read Boundary-Crossing Uplift

`additional_uplift_from_boundary_crossing` answers one question:

"How much extra fitness did I gain by stepping outside guarded boundaries?"

Interpretation:
- near zero: the boundary crossing was not worth much
- clearly positive: expedition bought measurable uplift
- positive but accompanied by critical warnings: keep it experimental, not ship-grade
