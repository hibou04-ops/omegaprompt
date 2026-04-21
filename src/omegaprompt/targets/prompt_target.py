"""PromptTarget - the CalibrableTarget adapter for LLM prompts.

Implements the structural interface omega-lock's search layer expects
without hard-importing omega-lock at module load (keeps it a soft dep for
users who only want the judge pieces).

v1.0 change: consumes a :class:`Judge` (rule / LLM / ensemble) rather
than a hard-coded LLM judge-call boundary, and translates the searcher's
parameter dict into a :class:`ResolvedPromptParams` / :class:`ProviderRequest`
using provider-neutral meta-axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaprompt.core.fitness import CompositeFitness
from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.enums import (
    OUTPUT_BUDGET_ORDINALS,
    OutputBudgetBucket,
    REASONING_ORDINALS,
    ReasoningProfile,
)
from omegaprompt.domain.judge import JudgeRubric
from omegaprompt.domain.params import MetaAxisSpace, PromptVariants, ResolvedPromptParams
from omegaprompt.judges.base import Judge
from omegaprompt.providers.base import LLMProvider, ProviderRequest, empty_usage


@dataclass
class _ParamSpecMini:
    name: str
    dtype: str
    low: Any = None
    high: Any = None
    neutral: Any = None


@dataclass
class _EvalResultMini:
    fitness: float
    n_trials: int
    metadata: dict = field(default_factory=dict)


class PromptTarget:
    """``CalibrableTarget`` adapter for prompt calibration.

    Parameters
    ----------
    target_provider:
        :class:`LLMProvider` the prompt-under-calibration runs on.
    judge:
        Any :class:`Judge` implementation. The provider used by the judge
        (if any) is the judge's own concern; the target does not need to
        know it.
    dataset:
        Items the target is scored on each ``evaluate()`` call.
    rubric:
        Scoring rubric passed through to ``judge.score()``.
    variants:
        Pool of system prompts and few-shot examples the searcher samples
        indices into.
    space:
        Bounds for each meta-axis. Derived from ``variants`` if omitted.
    """

    def __init__(
        self,
        *,
        target_provider: LLMProvider,
        judge: Judge,
        dataset: Dataset,
        rubric: JudgeRubric,
        variants: PromptVariants,
        space: MetaAxisSpace | None = None,
    ) -> None:
        if not variants.system_prompts:
            raise ValueError("PromptTarget requires at least one system prompt variant.")
        if len(dataset) == 0:
            raise ValueError("PromptTarget requires a non-empty dataset.")

        self.target_provider = target_provider
        self.judge = judge
        self.dataset = dataset
        self.rubric = rubric
        self.variants = variants

        if space is None:
            max_few_shot = min(3, len(variants.few_shot_examples))
            space = MetaAxisSpace(
                system_prompt_idx_max=len(variants.system_prompts) - 1,
                few_shot_min=0,
                few_shot_max=max_few_shot,
            )
        self.space = space

        self._fitness = CompositeFitness(rubric)
        self.last_usage: dict[str, int] = empty_usage()
        self.total_api_calls = 0

    # ---- CalibrableTarget protocol ----

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
            "system_prompt_idx": 0,
            "few_shot_count": self.space.few_shot_min,
            "reasoning_profile_idx": REASONING_ORDINALS[ReasoningProfile.STANDARD],
            "output_budget_idx": OUTPUT_BUDGET_ORDINALS[OutputBudgetBucket.MEDIUM],
            "response_schema_mode_idx": 0,
            "tool_policy_idx": 0,
        }

    def _param_specs_mini(self) -> list[_ParamSpecMini]:
        return [
            _ParamSpecMini(
                name="system_prompt_idx",
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
                name="reasoning_profile_idx",
                dtype="int",
                low=0,
                high=len(self.space.reasoning_profiles) - 1,
                neutral=min(1, len(self.space.reasoning_profiles) - 1),
            ),
            _ParamSpecMini(
                name="output_budget_idx",
                dtype="int",
                low=0,
                high=len(self.space.output_budgets) - 1,
                neutral=min(1, len(self.space.output_budgets) - 1),
            ),
            _ParamSpecMini(
                name="response_schema_mode_idx",
                dtype="int",
                low=0,
                high=len(self.space.response_schema_modes) - 1,
                neutral=0,
            ),
            _ParamSpecMini(
                name="tool_policy_idx",
                dtype="int",
                low=0,
                high=len(self.space.tool_policy_variants) - 1,
                neutral=0,
            ),
        ]

    def evaluate(self, params: dict):
        resolved = self._resolve_params(params)
        system_prompt = self.variants.system_prompts[resolved.system_prompt_idx]
        few_shots = self.variants.few_shot_examples[: resolved.few_shot_count]

        target_call_count = 0
        judge_call_count = 0
        run_usage = empty_usage()
        judge_results: list = []

        for item in self.dataset.items:
            target_request = ProviderRequest(
                system_prompt=system_prompt,
                user_message=item.input,
                few_shots=list(few_shots),
                reasoning_profile=resolved.reasoning_profile,
                output_budget=resolved.output_budget,
                response_schema_mode=resolved.response_schema_mode,
                tool_policy=resolved.tool_policy,
            )
            target_response = self.target_provider.call(target_request)
            target_call_count += 1
            _accumulate_usage(run_usage, target_response.usage)

            judge_result, judge_usage = self.judge.score(
                rubric=self.rubric,
                item=item,
                target_response=target_response.text,
            )
            judge_call_count += 1
            _accumulate_usage(run_usage, judge_usage)

            judge_results.append((item.id, judge_result))

        fitness = self._fitness.evaluate(judge_results)
        pass_rate = self._fitness.pass_rate()

        self.total_api_calls += target_call_count + judge_call_count
        _accumulate_usage(self.last_usage, run_usage)

        try:
            from omega_lock import EvalResult  # type: ignore
        except Exception:
            EvalResult = None  # type: ignore

        metadata = {
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
            "resolved_params": resolved.model_dump(),
            "target_calls": target_call_count,
            "judge_calls": judge_call_count,
            "usage": run_usage,
            "target_provider": self.target_provider.name,
            "target_model": self.target_provider.model,
            "judge_name": self.judge.name,
        }
        if EvalResult is not None:
            return EvalResult(
                fitness=fitness,
                n_trials=len(self.dataset),
                metadata=metadata,
            )
        return _EvalResultMini(
            fitness=fitness,
            n_trials=len(self.dataset),
            metadata=metadata,
        )

    # ---- internals ----

    def _resolve_params(self, params: dict | None) -> ResolvedPromptParams:
        params = params or {}
        neutral = self.neutral_params()

        def _get(key: str, default: Any) -> Any:
            return params.get(key, default)

        sys_idx = int(_get("system_prompt_idx", neutral["system_prompt_idx"]))
        sys_idx = max(0, min(self.space.system_prompt_idx_max, sys_idx))

        fs_count = int(_get("few_shot_count", neutral["few_shot_count"]))
        fs_count = max(self.space.few_shot_min, min(self.space.few_shot_max, fs_count))

        reasoning_idx = int(_get("reasoning_profile_idx", neutral["reasoning_profile_idx"]))
        reasoning_idx = max(0, min(len(self.space.reasoning_profiles) - 1, reasoning_idx))
        reasoning = self.space.reasoning_profiles[reasoning_idx]

        budget_idx = int(_get("output_budget_idx", neutral["output_budget_idx"]))
        budget_idx = max(0, min(len(self.space.output_budgets) - 1, budget_idx))
        budget = self.space.output_budgets[budget_idx]

        schema_idx = int(_get("response_schema_mode_idx", neutral["response_schema_mode_idx"]))
        schema_idx = max(0, min(len(self.space.response_schema_modes) - 1, schema_idx))
        schema = self.space.response_schema_modes[schema_idx]

        tool_idx = int(_get("tool_policy_idx", neutral["tool_policy_idx"]))
        tool_idx = max(0, min(len(self.space.tool_policy_variants) - 1, tool_idx))
        tool = self.space.tool_policy_variants[tool_idx]

        return ResolvedPromptParams(
            system_prompt_idx=sys_idx,
            few_shot_count=fs_count,
            reasoning_profile=reasoning,
            output_budget=budget,
            response_schema_mode=schema,
            tool_policy=tool,
        )


def _accumulate_usage(acc: dict[str, int], delta: dict[str, int] | None) -> None:
    if not delta:
        return
    for k in acc:
        acc[k] = acc.get(k, 0) + int(delta.get(k, 0) or 0)
    for k, v in delta.items():
        if k not in acc:
            acc[k] = int(v or 0)
