# Migration From omegaprompt

## What Stayed the Same

The calibration discipline is unchanged:
- sensitivity ranking and top-K unlock
- `hard_gate × soft_score`
- walk-forward held-out gate
- JSON artifact output

If you trusted `omegaprompt` because it was hard to overfit silently, that property is still the point.

## What Changed

The public engine name is now `omegacal`.

Compatibility retained:
- `omegaprompt` CLI still works
- `omegaprompt` imports still work
- `omegacal` is added as the new CLI/import entry point

## Axis Migration

Old axis style:
- `thinking_enabled`
- `effort_idx`
- raw token knobs

New axis style:
- `system_prompt_variant`
- `few_shot_count`
- `reasoning_profile`
- `output_budget_bucket`
- `response_schema_mode`
- `tool_policy_variant`

The new names describe intent, not one provider's API.

## Artifact Migration

Old artifact emphasis:
- best params
- best fitness
- walk-forward block

New artifact adds:
- selected profile
- neutral baseline params and fitness
- calibrated params and fitness
- uplift absolute and percent
- quality-per-cost and quality-per-latency
- boundary warnings
- degraded capabilities
- ship recommendation
- guarded-boundary status
- boundary-crossing uplift

`best_params` and `best_fitness` are still present for compatibility.

## Provider Migration

Old state:
- provider-neutral core started, but cloud assumptions still dominated

New state:
- Anthropic first-class
- OpenAI first-class
- Gemini explicit placeholder
- local adapters explicit and honest about limits

## Recommended Upgrade Path

1. Switch CLI usage from `omegaprompt` to `omegacal`.
2. Update any code or dashboards that read artifact fields to consume `calibrated_*`, `neutral_*`, and risk sections.
3. Rename custom search-space references to the new meta-axis names.
4. If you were using local or OpenAI-compatible endpoints, check degraded capability warnings before treating runs as ship-grade.

## Non-Goals of This Migration

- pretending all providers are equal
- claiming local judge parity with frontier cloud judges
- weakening walk-forward rules to simplify migration
