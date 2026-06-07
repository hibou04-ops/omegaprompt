"""PromptTarget - the provider-neutral calibrable target adapter."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.enums import (
    OUTPUT_BUDGET_ORDINALS,
    OutputBudgetBucket,
    REASONING_ORDINALS,
    ReasoningProfile,
)
from omegaprompt.domain.judge import JudgeRubric
from omegaprompt.domain.params import (
    MetaAxisSpace,
    PromptVariants,
    ResolvedPromptParams,
    validate_space_against_variants,
)
from omegaprompt.domain.profiles import (
    BoundaryWarning,
    ExecutionProfile,
    RiskCategory,
    ShipRecommendation,
)
from omegaprompt.domain.result import EvalItemResult, EvalResult
from omegaprompt.judges.base import Judge
from omegaprompt.providers.base import (
    LLMProvider,
    ProviderRequest,
    empty_usage,
    estimate_cost_units,
)


@dataclass
class _ParamSpecMini:
    name: str
    dtype: str
    low: Any = None
    high: Any = None
    neutral: Any = None


class PromptTarget:
    """``CalibrableTarget`` adapter for prompt calibration."""

    def __init__(
        self,
        *,
        target_provider: LLMProvider,
        judge: Judge,
        dataset: Dataset,
        rubric: JudgeRubric,
        variants: PromptVariants,
        space: MetaAxisSpace | None = None,
        execution_profile: ExecutionProfile = ExecutionProfile.GUARDED,
        max_workers: int = 1,
    ) -> None:
        if not variants.system_prompts:
            raise ValueError("PromptTarget requires at least one system prompt variant.")
        if len(dataset) == 0:
            raise ValueError("PromptTarget requires a non-empty dataset.")

        from omegaprompt.core.fitness import CompositeFitness

        self.target_provider = target_provider
        self.judge = judge
        self.dataset = dataset
        self.rubric = rubric
        self.variants = variants
        self.execution_profile = execution_profile

        if space is None:
            max_few_shot = min(3, len(variants.few_shot_examples))
            space = MetaAxisSpace(
                system_prompt_idx_max=len(variants.system_prompts) - 1,
                few_shot_min=0,
                few_shot_max=max_few_shot,
            )
        else:
            # Reviewer P1 #10: when the user supplies a space, its bounds
            # must match the variant pool sizes. Auto-derived spaces are
            # already consistent by construction.
            validate_space_against_variants(space, variants)
        self.space = space

        # C2 (opt-in, default off): bounds how many dataset items are
        # evaluated concurrently within a single evaluate() pass. Each item
        # still runs its target call then its judge call sequentially, so the
        # number of concurrent calls to any one provider never exceeds
        # max_workers. Default 1 = serial = byte-identical to prior versions.
        self.max_workers = max(1, int(max_workers))
        self._fitness = CompositeFitness(rubric)
        self.last_usage: dict[str, int] = empty_usage()
        self.total_api_calls = 0
        self.evaluation_history: list[EvalResult] = []
        # H1: memoize evaluate() results keyed on RESOLVED params so a repeat
        # eval of an already-seen configuration (e.g. the best-candidate
        # re-eval at the end of calibrate()) reuses the prior result instead
        # of re-calling the providers. Per-instance, GC'd with the target.
        self._eval_cache: dict[str, EvalResult] = {}

    def param_space(self) -> list:
        try:
            from omega_lock import ParamSpec  # type: ignore
        except Exception:
            ParamSpec = None  # type: ignore

        specs = self._param_specs_mini()
        if ParamSpec is None:
            return specs
        return [
            ParamSpec(name=s.name, dtype=s.dtype, low=s.low, high=s.high, neutral=s.neutral)
            for s in specs
        ]

    def neutral_params(self) -> dict[str, Any]:
        return {
            "system_prompt_variant": 0,
            "few_shot_count": self.space.few_shot_min,
            "reasoning_profile": REASONING_ORDINALS[ReasoningProfile.STANDARD],
            "output_budget_bucket": OUTPUT_BUDGET_ORDINALS[OutputBudgetBucket.MEDIUM],
            "response_schema_mode": 0,
            "tool_policy_variant": 0,
        }

    def _param_specs_mini(self) -> list[_ParamSpecMini]:
        return [
            _ParamSpecMini(
                name="system_prompt_variant",
                dtype="int",
                low=0,
                high=self.space.system_prompt_idx_max,
                neutral=0,
            ),
            _ParamSpecMini(
                name="few_shot_count",
                dtype="int",
                low=self.space.few_shot_min,
                high=self.space.few_shot_max,
                neutral=self.space.few_shot_min,
            ),
            _ParamSpecMini(
                name="reasoning_profile",
                dtype="int",
                low=0,
                high=len(self.space.reasoning_profiles) - 1,
                neutral=min(1, len(self.space.reasoning_profiles) - 1),
            ),
            _ParamSpecMini(
                name="output_budget_bucket",
                dtype="int",
                low=0,
                high=len(self.space.output_budgets) - 1,
                neutral=min(1, len(self.space.output_budgets) - 1),
            ),
            _ParamSpecMini(
                name="response_schema_mode",
                dtype="int",
                low=0,
                high=len(self.space.response_schema_modes) - 1,
                neutral=0,
            ),
            _ParamSpecMini(
                name="tool_policy_variant",
                dtype="int",
                low=0,
                high=len(self.space.tool_policy_variants) - 1,
                neutral=0,
            ),
        ]

    def evaluate(self, params: dict | None) -> EvalResult:
        resolved, param_clamp_warnings = self._resolve_params(params)

        # H1: return the memoized result for an already-evaluated resolved
        # configuration. The key is the resolved params (typed + clamped), so
        # two different raw param dicts that resolve to the same configuration
        # share a cache entry. A hit performs zero provider calls and does not
        # increment total_api_calls / usage (it returns the prior EvalResult
        # verbatim). Downstream consumers (runtime.py) read only fitness,
        # within_guarded_boundaries, degraded_capabilities, and resolved_params
        # off the returned result, never the raw .params, so reusing the prior
        # result is safe. If the grid never re-evaluates an identical resolved
        # configuration this is a harmless no-op.
        _cache_key = json.dumps(
            resolved.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        cached = self._eval_cache.get(_cache_key)
        if cached is not None:
            return cached

        system_prompt = self.variants.system_prompts[resolved.system_prompt_variant]
        few_shots = self.variants.few_shot_examples[: resolved.few_shot_count]

        target_call_count = 0
        judge_call_count = 0
        run_usage = empty_usage()
        total_latency_ms = 0.0
        judge_results: list[tuple[str, Any]] = []
        item_results: list[EvalItemResult] = []
        degraded_capabilities = []

        def _eval_one(item):
            """Evaluate one dataset item (target call then judge call).

            Mutates nothing shared: providers and the judge are stateless
            request->response Protocols, so this is safe to run concurrently
            across items. All accumulation into shared run state happens in
            the single-threaded fold below, in dataset order.
            """
            target_request = ProviderRequest(
                system_prompt=system_prompt,
                user_message=item.input,
                few_shots=list(few_shots),
                reasoning_profile=resolved.reasoning_profile,
                output_budget_bucket=resolved.output_budget_bucket,
                response_schema_mode=resolved.response_schema_mode,
                tool_policy_variant=resolved.tool_policy_variant,
                execution_profile=self.execution_profile,
            )

            target_started = perf_counter()
            target_response = self.target_provider.call(target_request)
            target_wall_ms = (perf_counter() - target_started) * 1000.0

            judge_started = perf_counter()
            judge_outcome = self.judge.score(
                rubric=self.rubric,
                item=item,
                target_response=target_response.text,
            )
            judge_result, judge_usage = judge_outcome
            judge_wall_ms = (perf_counter() - judge_started) * 1000.0

            # Reviewer P0: judge provider degradation must surface in
            # the artifact. Pre-fix only target_response.degraded_*
            # was accumulated; a judge that fell back from STRICT_SCHEMA
            # to JSON_OBJECT was invisible downstream — even though
            # judge-side fallbacks affect fitness reliability MORE than
            # target-side fallbacks.
            judge_degraded = list(
                getattr(judge_outcome, "degraded_capabilities", []) or []
            )

            item_latency_ms = max(target_response.latency_ms, target_wall_ms) + judge_wall_ms
            return (
                item,
                target_response,
                judge_result,
                judge_usage,
                judge_degraded,
                item_latency_ms,
            )

        if self.max_workers > 1:
            # ThreadPoolExecutor.map yields results in INPUT order regardless
            # of completion order, so the fold below is identical to the serial
            # path — the artifact is byte-stable across worker counts.
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                computed = list(executor.map(_eval_one, self.dataset.items))
        else:
            # Serial path: byte-identical to prior versions (default).
            computed = [_eval_one(item) for item in self.dataset.items]

        for (
            item,
            target_response,
            judge_result,
            judge_usage,
            judge_degraded,
            item_latency_ms,
        ) in computed:
            target_call_count += 1
            _accumulate_usage(run_usage, target_response.usage)
            judge_call_count += 1
            _accumulate_usage(run_usage, judge_usage)
            total_latency_ms += item_latency_ms
            degraded_capabilities.extend(target_response.degraded_capabilities)
            degraded_capabilities.extend(judge_degraded)
            judge_results.append((item.id, judge_result))
            item_results.append(
                EvalItemResult(
                    item_id=item.id,
                    params=(params or {}),
                    raw_output=target_response.text,
                    judge=judge_result,
                    token_usage=_merge_usage(target_response.usage, judge_usage),
                    latency_ms=item_latency_ms,
                    degraded_capabilities=(
                        list(target_response.degraded_capabilities) + judge_degraded
                    ),
                )
            )

        fitness = self._fitness.evaluate(judge_results)
        pass_rate = self._fitness.pass_rate()

        self.total_api_calls += target_call_count + judge_call_count
        _accumulate_usage(self.last_usage, run_usage)

        resolved_json = resolved.model_dump(mode="json")
        compat_resolved = {
            **resolved_json,
            "system_prompt_idx": resolved.system_prompt_variant,
            "output_budget": resolved.output_budget_bucket.value,
            "tool_policy": resolved.tool_policy_variant.value,
        }
        estimated_cost = estimate_cost_units(run_usage)
        # Reviewer P1 #11: param-clamp warnings under expedition profile
        # are also "off the guarded path" — they signal the optimizer
        # emitted out-of-bounds values that we silently clamped.
        within_guarded_boundaries = (
            not degraded_capabilities and not param_clamp_warnings
        )
        ship_recommendation = (
            ShipRecommendation.SHIP if within_guarded_boundaries else ShipRecommendation.EXPERIMENT
        )

        result = EvalResult(
            params=params or {},
            resolved_params=resolved_json,
            item_results=item_results,
            fitness=fitness,
            n_trials=len(self.dataset),
            hard_gate_pass_rate=pass_rate,
            usage_summary=dict(run_usage),
            latency_ms=total_latency_ms,
            estimated_cost_units=estimated_cost,
            degraded_capabilities=list(degraded_capabilities),
            boundary_warnings=list(param_clamp_warnings),
            within_guarded_boundaries=within_guarded_boundaries,
            ship_recommendation=ship_recommendation,
            metadata={
                "hard_gate_pass_rate": pass_rate,
                "per_item_scores": [
                    {
                        "item_id": p.item_id,
                        "soft_score": p.soft_score,
                        "gates_passed": p.gates_passed,
                        "final_score": p.final_score,
                    }
                    for p in self._fitness.last_per_item
                ],
                "resolved_params": compat_resolved,
                "target_calls": target_call_count,
                "judge_calls": judge_call_count,
                "usage": dict(run_usage),
                "latency_ms": total_latency_ms,
                "estimated_cost_units": estimated_cost,
                "degraded_capabilities": [e.model_dump(mode="json") for e in degraded_capabilities],
                "within_guarded_boundaries": within_guarded_boundaries,
                "target_provider": self.target_provider.name,
                "target_model": self.target_provider.model,
                "judge_name": self.judge.name,
            },
        )
        self.evaluation_history.append(result)
        self._eval_cache[_cache_key] = result
        return result

    def unique_param_count(self) -> int:
        seen = {
            str(sorted((res.resolved_params or {}).items()))
            for res in self.evaluation_history
            if res.resolved_params
        }
        return len(seen)

    def best_guarded_eval(self) -> EvalResult | None:
        guarded = [res for res in self.evaluation_history if res.within_guarded_boundaries]
        if not guarded:
            return None
        return max(guarded, key=lambda res: res.fitness)

    def _resolve_params(
        self, params: dict | None
    ) -> tuple[ResolvedPromptParams, list[BoundaryWarning]]:
        """Resolve raw params to typed values, surfacing out-of-range drift.

        Reviewer P1 #11: silent clamping is dangerous in an audit-first
        tool. An optimizer that emits ``system_prompt_variant=99`` against
        a 3-prompt pool gets clamped to 2; the artifact only shows the
        clamped value, so a reviewer cannot tell the optimizer was
        misconfigured.

        Profile policy:

        - ``GUARDED`` (default): out-of-range values raise. The optimizer
          contract says "respect axis bounds"; a violation is a setup
          bug we want surfaced before the eval runs.
        - ``EXPEDITION``: clamp + emit a ``BoundaryWarning`` per drift.
          Useful when iterating on the optimizer and a few drifts are
          expected; the warnings end up on ``EvalResult.boundary_warnings``
          and ``CalibrationArtifact.boundary_warnings`` so they remain
          visible in audits.

        Returns ``(resolved, warnings)``. Warnings is empty when no value
        was clamped or under guarded profile (which raises instead).
        """
        params = params or {}
        neutral = self.neutral_params()
        warnings: list[BoundaryWarning] = []

        def _get(key: str, default: Any) -> Any:
            return params.get(key, default)

        def _clamp_or_raise(name: str, raw: int, low: int, high: int) -> int:
            clamped = max(low, min(high, raw))
            if clamped == raw:
                return clamped
            if self.execution_profile == ExecutionProfile.GUARDED:
                raise ValueError(
                    f"Parameter {name}={raw} is out of axis bounds "
                    f"[{low}, {high}] under guarded profile. Either fix "
                    f"the optimizer to respect axis bounds, or run under "
                    f"expedition profile (which clamps with a "
                    f"BoundaryWarning instead of raising)."
                )
            warnings.append(
                BoundaryWarning(
                    code="param_clamped",
                    category=RiskCategory.SAFETY_BOUNDARY,
                    severity="warning",
                    summary=f"Parameter {name} was clamped to axis bounds.",
                    detail=(
                        f"{name}: {raw} -> {clamped} "
                        f"(bounds: [{low}, {high}])"
                    ),
                )
            )
            return clamped

        sys_idx_raw = int(
            _get(
                "system_prompt_variant",
                _get("system_prompt_idx", neutral["system_prompt_variant"]),
            )
        )
        sys_idx = _clamp_or_raise(
            "system_prompt_variant", sys_idx_raw, 0, self.space.system_prompt_idx_max
        )

        fs_count_raw = int(_get("few_shot_count", neutral["few_shot_count"]))
        fs_count = _clamp_or_raise(
            "few_shot_count",
            fs_count_raw,
            self.space.few_shot_min,
            self.space.few_shot_max,
        )

        reasoning_idx_raw = int(
            _get(
                "reasoning_profile",
                _get("reasoning_profile_idx", neutral["reasoning_profile"]),
            )
        )
        reasoning_idx = _clamp_or_raise(
            "reasoning_profile",
            reasoning_idx_raw,
            0,
            len(self.space.reasoning_profiles) - 1,
        )
        reasoning = self.space.reasoning_profiles[reasoning_idx]

        budget_idx_raw = int(
            _get(
                "output_budget_bucket",
                _get("output_budget_idx", neutral["output_budget_bucket"]),
            )
        )
        budget_idx = _clamp_or_raise(
            "output_budget_bucket",
            budget_idx_raw,
            0,
            len(self.space.output_budgets) - 1,
        )
        budget = self.space.output_budgets[budget_idx]

        schema_idx_raw = int(
            _get(
                "response_schema_mode",
                _get("response_schema_mode_idx", neutral["response_schema_mode"]),
            )
        )
        schema_idx = _clamp_or_raise(
            "response_schema_mode",
            schema_idx_raw,
            0,
            len(self.space.response_schema_modes) - 1,
        )
        schema = self.space.response_schema_modes[schema_idx]

        tool_idx_raw = int(
            _get(
                "tool_policy_variant",
                _get("tool_policy_idx", neutral["tool_policy_variant"]),
            )
        )
        tool_idx = _clamp_or_raise(
            "tool_policy_variant",
            tool_idx_raw,
            0,
            len(self.space.tool_policy_variants) - 1,
        )
        tool = self.space.tool_policy_variants[tool_idx]

        resolved = ResolvedPromptParams(
            system_prompt_variant=sys_idx,
            few_shot_count=fs_count,
            reasoning_profile=reasoning,
            output_budget_bucket=budget,
            response_schema_mode=schema,
            tool_policy_variant=tool,
        )
        return resolved, warnings


def _accumulate_usage(acc: dict[str, int], delta: dict[str, int] | None) -> None:
    if not delta:
        return
    for k in acc:
        acc[k] = acc.get(k, 0) + int(delta.get(k, 0) or 0)
    for k, v in delta.items():
        if k not in acc:
            acc[k] = int(v or 0)


def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    out = dict(left or {})
    _accumulate_usage(out, right or {})
    return out
