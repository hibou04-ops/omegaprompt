# Profiles and Risk Boundaries

Profiles define how much structural risk the run is allowed to take. They do
not certify production success. For the full trust model, see
[trust-model.md](trust-model.md).

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

`omegaprompt` supports both with one engine. `omegacal` remains a compatibility CLI alias.

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
- treats `FAIL_KC4_GATE`, hard-gate failures, and hidden provider degradation
  as CI-relevant failure signals

## Expedition

Used when efficiency-adjusted uplift may justify higher structural risk.

Typical behavior:
- allows controlled schema degradation
- allows experimental adapters
- records each relaxed safeguard
- records the extra uplift gained by crossing guarded boundaries
- downgrades ship advice when the result is still exploration-grade

Expedition is not ship-grade by default. It is a way to measure boundary
crossing honestly, not a way to suppress the boundary.

## Validation Modes

Walk-forward validation records a `validation_mode`:

- `paired`: train/test slices are expected to share item ids. KC4 correlation
  is part of the gate and fails closed when the shared set is insufficient.
- `disjoint`: train/test slices intentionally do not share item ids. KC4 is not
  applicable; the gate is gap-only.
- `auto`: compatibility mode. KC4 is computed when enough shared ids exist, and
  otherwise the artifact records why it was not computed.

The profile does not make an unrepresentative holdout representative. A pass is
evidence that the declared local gate passed, not a production guarantee.

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

## Diff Regression Use

`omegaprompt diff` should compare a baseline artifact against a candidate
artifact in Prompt CI. A candidate that newly fails status, changes
`ship_recommendation` to `hold` or `block`, regresses walk-forward fitness, or
crosses a guarded boundary should block the CI gate even if one raw fitness
number improved.
