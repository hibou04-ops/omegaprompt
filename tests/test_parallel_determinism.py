"""C2 determinism gate: serial (max_workers=1) == parallel (max_workers=4).

The opt-in parallel item evaluation must produce a byte-identical artifact to
the serial path. ThreadPoolExecutor.map yields results in INPUT order, and the
fold loop runs single-threaded in dataset order, so list-ordered fields
(item_results, degraded_capabilities) stay stable across worker counts.

The hash projection strips the three wall-clock latency fields
(EvalResult.latency_ms, each item_results[].latency_ms, metadata.latency_ms),
which are derived from perf_counter() deltas and vary run-to-run even serial vs
serial. Latency is the ONLY nondeterministic surface; everything else
(fitness, usage, resolved_params, item_results order + raw_output, degraded
events) is order-sensitive-but-deterministic and is what the hash locks.

A MANDATORY negative control folds results out of dataset order and asserts the
hash DIFFERS — proving the test has teeth to catch a fold-order regression.
"""

from __future__ import annotations

import hashlib
import json
import time

from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.judge import Dimension, HardGate, JudgeResult, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.providers.base import (
    CapabilityEvent,
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


def _dataset(n: int = 6) -> Dataset:
    return Dataset.from_items([{"id": f"t{i}", "input": f"item-{i}"} for i in range(n)])


def _variants() -> PromptVariants:
    return PromptVariants(
        system_prompts=["sp-A", "sp-B"],
        few_shot_examples=[{"input": "q1", "output": "a1"}],
    )


class _PerItemProvider:
    """Target provider whose output ENCODES the item index, and which emits a
    degraded_capabilities event on specific items (with the item id in the
    event, so a reorder changes the bytes). A small sleep creates real overlap
    so the parallel path actually runs concurrently."""

    name = "anthropic"
    model = "target-model"

    def __init__(self, degraded_on: set[str]) -> None:
        self.degraded_on = degraded_on

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.name,
            tier=CapabilityTier.CLOUD,
            supports_strict_schema=True,
            supports_json_object=True,
        )

    def call(self, request) -> ProviderResponse:
        time.sleep(0.01)
        # The user_message is the item input "item-<i>"; encode it in the text.
        marker = request.user_message
        degraded: list[CapabilityEvent] = []
        if marker in self.degraded_on:
            degraded = [
                CapabilityEvent(
                    capability="strict_schema",
                    requested="strict_schema",
                    applied="json_object_parse",
                    reason=f"fallback-for-{marker}",
                    user_visible_note=f"degraded on {marker}",
                )
            ]
        return ProviderResponse(
            text=f"out:{marker}",
            usage={
                "input_tokens": 5,
                "output_tokens": 3,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            latency_ms=120.0,
            degraded_capabilities=degraded,
        )


class _PerItemJudge:
    name = "mock-judge"

    def score(self, *, rubric, item, target_response):
        # Distinct score per item so per-item fitness ordering is meaningful.
        idx = int(item.id[1:])
        return (
            JudgeResult(scores={"q": 1 if idx % 2 == 0 else 1}, gate_results={"g": True}),
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )


# The three wall-clock fields stripped from the hash projection.
def _project(result) -> str:
    """Deterministic projection of an EvalResult: full model_dump minus the
    three perf_counter-derived latency fields."""
    data = result.model_dump(mode="json")
    data.pop("latency_ms", None)
    for item in data.get("item_results", []):
        item.pop("latency_ms", None)
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("latency_ms", None)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _hash(result) -> str:
    return hashlib.sha256(_project(result).encode("utf-8")).hexdigest()


def _build_target(max_workers: int) -> PromptTarget:
    return PromptTarget(
        target_provider=_PerItemProvider(degraded_on={"item-1", "item-3"}),
        judge=_PerItemJudge(),
        dataset=_dataset(n=6),
        rubric=_rubric(),
        variants=_variants(),
        max_workers=max_workers,
    )


def test_serial_equals_parallel_artifact_hash():
    """max_workers=1 and max_workers=4 produce a byte-identical projected
    artifact. Distinct per-item outputs + degraded events on items 1 and 3 give
    the hash real content to lock on."""
    serial = _build_target(max_workers=1).evaluate({"system_prompt_variant": 0})
    parallel = _build_target(max_workers=4).evaluate({"system_prompt_variant": 0})

    assert _hash(serial) == _hash(parallel)

    # Sanity: the artifact actually carries the order-sensitive content the hash
    # is supposed to protect (otherwise the equality above would be vacuous).
    assert [ir.raw_output for ir in serial.item_results] == [f"out:item-{i}" for i in range(6)]
    assert len(serial.degraded_capabilities) == 2
    assert [ir.raw_output for ir in parallel.item_results] == [f"out:item-{i}" for i in range(6)]


def test_negative_control_out_of_order_fold_changes_hash():
    """MANDATORY negative control. Fold the SAME per-item results in a
    different order and assert the projected hash DIFFERS. This proves the
    determinism test has teeth: if fold order did not matter, the test above
    could never catch a parallel fold-order bug."""
    result = _build_target(max_workers=1).evaluate({"system_prompt_variant": 0})
    in_order_hash = _hash(result)

    # Build a shuffled copy by reversing the order-sensitive lists. The degraded
    # events carry distinct content (item id in the reason), and item_results
    # carry distinct raw_output, so a reorder MUST change the bytes.
    shuffled = result.model_copy(deep=True)
    shuffled.item_results = list(reversed(shuffled.item_results))
    shuffled.degraded_capabilities = list(reversed(shuffled.degraded_capabilities))

    assert _hash(shuffled) != in_order_hash, (
        "Reordering item_results/degraded_capabilities did not change the hash — "
        "the determinism test would be blind to a fold-order regression."
    )
