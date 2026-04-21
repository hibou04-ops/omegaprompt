"""Reproducible reference run - produces the artifact cited in the README.

This script drives omega_lock.run_p1 with a *deterministic* in-memory
target + judge. No LLM API calls. The fitness function is a closed-form
function of the meta-axis parameters, so the same run produces byte-
identical results on every machine.

Run::

    python examples/reference/reproduce_reference_artifact.py

Output: examples/reference/reference_artifact.json
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from omega_lock import P1Config, run_p1  # type: ignore

from omegaprompt.commands import calibrate as calibrate_mod
from omegaprompt.core.artifact import save_artifact
from omegaprompt.core.walkforward import evaluate_walk_forward
from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.judge import Dimension, HardGate, JudgeResult, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.domain.profiles import ExecutionProfile
from omegaprompt.domain.result import CalibrationArtifact
from omegaprompt.judges.llm_judge import LLMJudge
from omegaprompt.providers.base import ProviderResponse
from omegaprompt.targets.prompt_target import PromptTarget


# ---------- deterministic stubs ----------

class DeterministicProvider:
    """Provider whose response text encodes the prompt configuration.

    ``resp:{len(system_prompt)}:{len(few_shots)}:{reasoning_profile}`` —
    later decoded by the deterministic judge into a fitness contribution.
    """

    name = "anthropic"
    model = "deterministic-reference"

    def call(self, request):
        encoded = f"resp:{len(request.system_prompt)}:{len(request.few_shots)}:{request.reasoning_profile.value}"
        return ProviderResponse(
            text=encoded,
            usage={
                "input_tokens": 40 + len(request.system_prompt) // 4,
                "output_tokens": 20,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            latency_ms=120.0,
        )

    def capabilities(self):
        from omegaprompt.providers.base import CapabilityTier, ProviderCapabilities

        return ProviderCapabilities(
            provider=self.name,
            tier=CapabilityTier.CLOUD,
            supports_strict_schema=True,
            supports_json_object=True,
            supports_reasoning_profiles=True,
            supports_usage_accounting=True,
            supports_llm_judge=True,
            ship_grade_judge=True,
            notes=["Deterministic reference stub; not a real API call."],
        )


_REASONING_BONUS = {"off": 0, "light": 1, "standard": 2, "deep": 3}


def deterministic_score(self, *, rubric, item, target_response):
    """Score encodes ``sp_len // 20 + fs_count + reasoning_bonus``, clamped to [1, 5]."""
    parts = target_response.split(":")
    sp_len = int(parts[1]) if len(parts) >= 2 else 0
    fs_count = int(parts[2]) if len(parts) >= 3 else 0
    reasoning = parts[3] if len(parts) >= 4 else "standard"
    base = sp_len // 20 + fs_count + _REASONING_BONUS.get(reasoning, 1)
    score = max(1, min(5, base))
    return (
        JudgeResult(
            scores={"accuracy": score, "clarity": max(1, min(5, score - 1))},
            gate_results={"no_refusal": True},
            notes=f"deterministic stub; base={base}, clamped={score}",
        ),
        {
            "input_tokens": 60,
            "output_tokens": 15,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 120,
        },
    )


# ---------- dataset / rubric / variants ----------


def build_inputs():
    train_items = [
        {"id": f"train_{i}", "input": f"compute {i} + {i + 1}"}
        for i in range(6)
    ]
    test_items = [
        {"id": f"test_{i}", "input": f"compute {i * 2} + {i}"}
        for i in range(4)
    ]
    train_ds = Dataset.from_items(train_items)
    test_ds = Dataset.from_items(test_items)

    rubric = JudgeRubric(
        dimensions=[
            Dimension(name="accuracy", description="is the answer correct", weight=0.7),
            Dimension(name="clarity", description="is the reasoning readable", weight=0.3),
        ],
        hard_gates=[
            HardGate(name="no_refusal", description="model attempts the task", evaluator="judge"),
        ],
    )

    variants = PromptVariants(
        system_prompts=[
            "You are an assistant.",
            "You are a terse senior engineer. Show your reasoning compactly.",
            "You are a thorough tutor. Explain each step, then answer.",
        ],
        few_shot_examples=[
            {"input": "1 + 1 =", "output": "2"},
            {"input": "3 + 4 =", "output": "7"},
        ],
    )

    return train_ds, test_ds, rubric, variants


# ---------- run ----------


def main():
    train_ds, test_ds, rubric, variants = build_inputs()
    target = DeterministicProvider()
    judge_provider = DeterministicProvider()
    judge = LLMJudge(provider=judge_provider)

    with patch.object(LLMJudge, "score", deterministic_score):
        train_target = PromptTarget(
            target_provider=target,
            judge=judge,
            dataset=train_ds,
            rubric=rubric,
            variants=variants,
            execution_profile=ExecutionProfile.GUARDED,
        )
        test_target = PromptTarget(
            target_provider=target,
            judge=judge,
            dataset=test_ds,
            rubric=rubric,
            variants=variants,
            execution_profile=ExecutionProfile.GUARDED,
        )

        config = P1Config(unlock_k=2)
        result = run_p1(train_target=train_target, test_target=test_target, config=config)

    # Extract artifact fields from the real P1Result.
    grid_best = getattr(result, "grid_best", None) or {}
    best_unlocked = grid_best.get("unlocked", {}) if isinstance(grid_best, dict) else {}
    best_fitness = float(grid_best.get("fitness", 0.0)) if isinstance(grid_best, dict) else 0.0

    wf_block = getattr(result, "walk_forward", None) or {}
    test_fits = wf_block.get("test_fitnesses") or []
    test_fitness = float(test_fits[0]) if test_fits else None

    walk_forward = None
    if test_fitness is not None:
        walk_forward = evaluate_walk_forward(
            best_fitness, test_fitness, max_gap=0.30, min_kc4=0.30
        )

    sensitivity_rows = []
    for rank, entry in enumerate(getattr(result, "stress_results", []) or []):
        if isinstance(entry, dict):
            sensitivity_rows.append(
                {
                    "axis": entry.get("name"),
                    "gini_delta": entry.get("normalized_stress"),
                    "raw_stress": entry.get("raw_stress"),
                    "rank": rank,
                }
            )
    sensitivity_rows.sort(key=lambda r: (r["gini_delta"] or 0.0), reverse=True)
    for i, row in enumerate(sensitivity_rows):
        row["rank"] = i

    baseline = getattr(result, "baseline_result", None) or {}
    neutral_fitness = float(baseline.get("fitness", 0.0)) if isinstance(baseline, dict) else 0.0
    neutral_params = train_target.neutral_params()

    uplift_absolute = best_fitness - neutral_fitness
    uplift_percent = (uplift_absolute / neutral_fitness * 100.0) if neutral_fitness > 0 else 0.0

    artifact = CalibrationArtifact(
        method="p1",
        unlock_k=2,
        selected_profile=ExecutionProfile.GUARDED,
        neutral_baseline_params=neutral_params,
        neutral_fitness=neutral_fitness,
        calibrated_params=best_unlocked,
        calibrated_fitness=best_fitness,
        uplift_absolute=uplift_absolute,
        uplift_percent=uplift_percent,
        best_params=best_unlocked,
        best_fitness=best_fitness,
        walk_forward=walk_forward,
        hard_gate_pass_rate=train_target._fitness.pass_rate(),
        sensitivity_ranking=sensitivity_rows,
        n_candidates_evaluated=train_target.total_api_calls
        // max(len(train_ds) * 2, 1),
        total_api_calls=train_target.total_api_calls + test_target.total_api_calls,
        usage_summary=dict(train_target.last_usage),
        target_provider=target.name,
        target_model=target.model,
        judge_provider=judge_provider.name,
        judge_model=judge_provider.model,
        status="OK" if (walk_forward is None or walk_forward.passed) else "FAIL_KC4_GATE",
        rationale=(
            "deterministic stubs; reproducible reference artifact"
            if walk_forward is None or walk_forward.passed
            else f"KC-4 failed: gap={walk_forward.generalization_gap:.3f}"
        ),
    )

    out_path = Path(__file__).with_name("reference_artifact.json")
    save_artifact(artifact, out_path)
    print(f"Wrote {out_path}")
    print(f"status: {artifact.status}")
    print(f"neutral_fitness:    {neutral_fitness:.4f}")
    print(f"calibrated_fitness: {best_fitness:.4f}")
    print(f"uplift_absolute:    {uplift_absolute:+.4f}")
    print(f"uplift_percent:     {uplift_percent:+.2f}%")
    if walk_forward is not None:
        print(
            f"walk_forward:       train={walk_forward.train_best_fitness:.4f} "
            f"test={walk_forward.test_fitness:.4f} "
            f"gap={walk_forward.generalization_gap:.4f} "
            f"kc4={walk_forward.kc4_correlation} "
            f"passed={walk_forward.passed}"
        )
    print("sensitivity ranking:")
    for row in sensitivity_rows:
        print(f"  rank {row['rank']}: {row['axis']:<30} gini_delta={row['gini_delta']}")


if __name__ == "__main__":
    main()
