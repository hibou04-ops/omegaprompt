"""C2 concurrency bound: observed max concurrent items <= max_workers.

Each item runs its target call then its judge call sequentially, so the number
of items in flight at once is bounded by max_workers — which also bounds the
concurrent calls to any one provider. A counting provider with an atomic
in-flight counter records the observed peak; we assert it never exceeds the
configured ceiling.
"""

from __future__ import annotations

import threading
import time

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


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[Dimension(name="q", description="q", weight=1.0, scale=(0, 1))],
        hard_gates=[HardGate(name="g", description="g")],
    )


def _dataset(n: int = 8) -> Dataset:
    return Dataset.from_items([{"id": f"t{i}", "input": f"in{i}"} for i in range(n)])


def _variants() -> PromptVariants:
    return PromptVariants(
        system_prompts=["sp-A"],
        few_shot_examples=[{"input": "q1", "output": "a1"}],
    )


class _InFlightCountingProvider:
    name = "anthropic"
    model = "target-model"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_observed = 0

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(provider=self.name, tier=CapabilityTier.CLOUD)

    def call(self, request) -> ProviderResponse:
        with self._lock:
            self._in_flight += 1
            if self._in_flight > self.max_observed:
                self.max_observed = self._in_flight
        try:
            time.sleep(0.01)  # hold the slot so overlap is observable
        finally:
            with self._lock:
                self._in_flight -= 1
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
                {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            )

    return _J()


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_observed_concurrency_never_exceeds_ceiling(max_workers):
    provider = _InFlightCountingProvider()
    t = PromptTarget(
        target_provider=provider,
        judge=_judge(),
        dataset=_dataset(n=8),
        rubric=_rubric(),
        variants=_variants(),
        max_workers=max_workers,
    )
    t.evaluate({"system_prompt_variant": 0})
    assert provider.max_observed <= max_workers, (
        f"observed {provider.max_observed} concurrent calls > ceiling {max_workers}"
    )
    # Serial must be exactly 1; parallel should actually overlap (>1) so the
    # bound is a meaningful upper limit, not a vacuous one.
    if max_workers == 1:
        assert provider.max_observed == 1
    else:
        assert provider.max_observed >= 2, (
            "parallel path did not overlap — the bound assertion would be vacuous"
        )
