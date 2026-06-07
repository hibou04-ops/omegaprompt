"""C2 fail-fast: a mid-dataset provider raise propagates out of evaluate().

ThreadPoolExecutor.map re-raises the first input-order failure on iteration, so
a target call that raises on a mid-dataset item surfaces as an exception out of
evaluate() — matching serial fail-first semantics — and no partial artifact is
returned.
"""

from __future__ import annotations

import pytest

from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.judge import Dimension, HardGate, JudgeResult, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.providers.base import (
    CapabilityTier,
    ProviderCapabilities,
    ProviderResponse,
)
from omegaprompt.targets.prompt_target import PromptTarget


class _BoomError(RuntimeError):
    pass


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[Dimension(name="q", description="q", weight=1.0, scale=(0, 1))],
        hard_gates=[HardGate(name="g", description="g")],
    )


def _dataset(n: int = 6) -> Dataset:
    return Dataset.from_items([{"id": f"t{i}", "input": f"item-{i}"} for i in range(n)])


def _variants() -> PromptVariants:
    return PromptVariants(
        system_prompts=["sp-A"],
        few_shot_examples=[{"input": "q1", "output": "a1"}],
    )


class _RaisingOnItemProvider:
    name = "anthropic"
    model = "target-model"

    def __init__(self, raise_on: str) -> None:
        self.raise_on = raise_on

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(provider=self.name, tier=CapabilityTier.CLOUD)

    def call(self, request) -> ProviderResponse:
        if request.user_message == self.raise_on:
            raise _BoomError(f"provider failed on {request.user_message}")
        return ProviderResponse(
            text="ok",
            usage={
                "input_tokens": 5,
                "output_tokens": 3,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )


def _judge():
    class _J:
        name = "mock-judge"

        def score(self, *, rubric, item, target_response):
            return (
                JudgeResult(scores={"q": 1}, gate_results={"g": True}),
                {},
            )

    return _J()


@pytest.mark.parametrize("max_workers", [1, 4])
def test_mid_dataset_raise_propagates_no_partial_artifact(max_workers):
    t = PromptTarget(
        target_provider=_RaisingOnItemProvider(raise_on="item-3"),
        judge=_judge(),
        dataset=_dataset(n=6),
        rubric=_rubric(),
        variants=_variants(),
        max_workers=max_workers,
    )
    with pytest.raises(_BoomError):
        t.evaluate({"system_prompt_variant": 0})
    # No partial result is recorded in history or cache.
    assert t.evaluation_history == []
    assert t._eval_cache == {}
