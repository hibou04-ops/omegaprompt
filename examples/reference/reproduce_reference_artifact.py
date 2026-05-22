"""Reproducible offline golden artifacts for the README and CI.

This script drives ``omega_lock.run_p1`` with a deterministic in-memory
target + judge for the clean reference artifact, then derives additional
semantic golden cases from that validated baseline. No LLM API calls. No
network. No random values.

Run::

    python examples/reference/reproduce_reference_artifact.py

Outputs:

- ``examples/reference/reference_artifact.json``
- ``examples/reference/reference_fail_kc4_gate.json``
- ``examples/reference/reference_fail_hard_gates.json``
- ``examples/reference/reference_provider_degradation.json``
- ``examples/reference/reference_diff_regression.json``
- ``examples/reference/golden_manifest.json``
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from omega_lock import P1Config, run_p1  # type: ignore

from omegaprompt.core.artifact import save_artifact
from omegaprompt.core.artifact_integrity import (
    check_artifact_integrity,
    normalized_artifact_hash,
)
from omegaprompt.core.walkforward import evaluate_walk_forward
from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.judge import Dimension, HardGate, JudgeResult, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.domain.profiles import (
    BoundaryWarning,
    ExecutionProfile,
    RelaxedSafeguard,
    RiskCategory,
    ShipRecommendation,
)
from omegaprompt.domain.result import CalibrationArtifact
from omegaprompt.judges.llm_judge import LLMJudge
from omegaprompt.providers.base import (
    CapabilityEvent,
    CapabilityTier,
    ProviderCapabilities,
    ProviderResponse,
)
from omegaprompt.runtime import diff as artifact_diff
from omegaprompt.targets.prompt_target import PromptTarget


REFERENCE_DIR = Path(__file__).resolve().parent
REFERENCE_COMMAND = "python examples/reference/reproduce_reference_artifact.py"

CASE_FILES = {
    "clean_ok_ship": "reference_artifact.json",
    "fail_kc4_gate": "reference_fail_kc4_gate.json",
    "fail_hard_gates": "reference_fail_hard_gates.json",
    "provider_degradation": "reference_provider_degradation.json",
    "diff_regression_candidate": "reference_diff_regression.json",
}


class DeterministicProvider:
    """Provider whose response text encodes the prompt configuration."""

    name = "anthropic"
    model = "deterministic-reference"

    def call(self, request):
        encoded = (
            f"resp:{len(request.system_prompt)}:"
            f"{len(request.few_shots)}:{request.reasoning_profile.value}"
        )
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
        return deterministic_capabilities(self.name)


def deterministic_capabilities(provider: str) -> ProviderCapabilities:
    return ProviderCapabilities(
        provider=provider,
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
    """Decode the deterministic provider response into a judge score."""

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
            HardGate(
                name="no_refusal",
                description="model attempts the task",
                evaluator="judge",
            ),
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


def build_clean_reference_artifact() -> CalibrationArtifact:
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

        result = run_p1(
            train_target=train_target,
            test_target=test_target,
            config=P1Config(unlock_k=2),
        )

    grid_best = getattr(result, "grid_best", None) or {}
    best_unlocked = grid_best.get("unlocked", {}) if isinstance(grid_best, dict) else {}
    best_fitness = float(grid_best.get("fitness", 0.0)) if isinstance(grid_best, dict) else 0.0

    wf_block = getattr(result, "walk_forward", None) or {}
    test_fits = wf_block.get("test_fitnesses") or []
    test_fitness = float(test_fits[0]) if test_fits else None
    walk_forward = (
        evaluate_walk_forward(best_fitness, test_fitness, max_gap=0.30, min_kc4=0.30)
        if test_fitness is not None
        else None
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
    for index, row in enumerate(sensitivity_rows):
        row["rank"] = index

    baseline = getattr(result, "baseline_result", None) or {}
    neutral_fitness = float(baseline.get("fitness", 0.0)) if isinstance(baseline, dict) else 0.0
    uplift_absolute = best_fitness - neutral_fitness
    uplift_percent = (uplift_absolute / neutral_fitness * 100.0) if neutral_fitness > 0 else 0.0

    return CalibrationArtifact(
        method="p1",
        unlock_k=2,
        selected_profile=ExecutionProfile.GUARDED,
        neutral_baseline_params=train_target.neutral_params(),
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
        n_candidates_evaluated=train_target.total_api_calls // max(len(train_ds) * 2, 1),
        total_api_calls=train_target.total_api_calls + test_target.total_api_calls,
        usage_summary=dict(train_target.last_usage),
        target_provider=target.name,
        target_model=target.model,
        judge_provider=judge_provider.name,
        judge_model=judge_provider.model,
        target_capabilities=target.capabilities(),
        judge_capabilities=judge_provider.capabilities(),
        ship_recommendation=(
            ShipRecommendation.SHIP
            if walk_forward is None or walk_forward.passed
            else ShipRecommendation.HOLD
        ),
        status="OK" if (walk_forward is None or walk_forward.passed) else "FAIL_KC4_GATE",
        rationale=(
            "deterministic stubs; reproducible reference artifact"
            if walk_forward is None or walk_forward.passed
            else f"KC-4 failed: gap={walk_forward.generalization_gap:.3f}"
        ),
    )


def build_golden_artifacts() -> dict[str, CalibrationArtifact]:
    clean = build_clean_reference_artifact()
    return {
        "clean_ok_ship": clean,
        "fail_kc4_gate": _fail_kc4_gate(clean),
        "fail_hard_gates": _fail_hard_gates(clean),
        "provider_degradation": _provider_degradation(clean),
        "diff_regression_candidate": _diff_regression_candidate(clean),
    }


def build_golden_manifest(
    artifacts: dict[str, CalibrationArtifact],
) -> dict[str, object]:
    cases = []
    for case_id, artifact in artifacts.items():
        path = REFERENCE_DIR / CASE_FILES[case_id]
        report = check_artifact_integrity(path) if path.exists() else None
        diff_baseline_case_id = "clean_ok_ship" if case_id == "diff_regression_candidate" else None
        diff_regressed = None
        if diff_baseline_case_id is not None:
            diff_regressed = artifact_diff(
                artifacts[diff_baseline_case_id],
                artifact,
            ).regressed
        integrity_classification = _integrity_classification_from_artifact(
            artifact, report
        )
        cases.append(
            {
                "case_id": case_id,
                "artifact": f"examples/reference/{CASE_FILES[case_id]}",
                "reproducible_command": REFERENCE_COMMAND,
                "expected_status": _enum_value(artifact.status),
                "expected_ship_recommendation": _enum_value(artifact.ship_recommendation),
                "expected_validation_mode": (
                    artifact.walk_forward.validation_mode
                    if artifact.walk_forward is not None
                    else None
                ),
                "expected_integrity_classification": integrity_classification,
                "normalized_artifact_hash": normalized_artifact_hash(artifact),
                "exact_metrics_may_be_displayed": True,
                "diff_baseline_case_id": diff_baseline_case_id,
                "expected_diff_regression": diff_regressed,
            }
        )

    return {
        "schema_version": "1.0",
        "description": "Deterministic no-network golden CalibrationArtifact harness.",
        "network": "none",
        "provider_mode": "deterministic in-memory stubs only",
        "cases": cases,
    }


def write_golden_artifacts() -> dict[str, object]:
    artifacts = build_golden_artifacts()
    for case_id, artifact in artifacts.items():
        save_artifact(artifact, REFERENCE_DIR / CASE_FILES[case_id])

    manifest = build_golden_manifest(artifacts)
    (REFERENCE_DIR / "golden_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _clone(base: CalibrationArtifact, **updates: object) -> CalibrationArtifact:
    data = base.model_dump(mode="json")
    data.update(updates)
    return CalibrationArtifact.model_validate(data)


def _fail_kc4_gate(base: CalibrationArtifact) -> CalibrationArtifact:
    walk_forward = evaluate_walk_forward(
        0.90,
        0.86,
        per_item_train={"a": 0.2, "b": 0.4, "c": 0.6, "d": 0.8},
        per_item_test={"a": 0.8, "b": 0.6, "c": 0.4, "d": 0.2},
        max_gap=0.30,
        min_kc4=0.30,
        validation_mode="paired",
    )
    return _clone(
        base,
        calibrated_fitness=0.90,
        best_fitness=0.90,
        uplift_absolute=0.90 - base.neutral_fitness,
        uplift_percent=((0.90 - base.neutral_fitness) / base.neutral_fitness * 100.0),
        walk_forward=walk_forward.model_dump(mode="json"),
        status="FAIL_KC4_GATE",
        ship_recommendation=ShipRecommendation.HOLD.value,
        rationale=(
            "KC-4 failed in paired validation: correlation fell below the "
            "pre-declared threshold."
        ),
    )


def _fail_hard_gates(base: CalibrationArtifact) -> CalibrationArtifact:
    warning = BoundaryWarning(
        code="hard_gate_failed",
        category=RiskCategory.DEPLOYMENT_READINESS,
        severity="critical",
        summary="A required hard gate failed in the deterministic fixture.",
        detail="This artifact is intentionally blocked for golden harness coverage.",
    )
    return _clone(
        base,
        hard_gate_pass_rate=0.0,
        boundary_warnings=[warning.model_dump(mode="json")],
        status="FAIL_HARD_GATES",
        ship_recommendation=ShipRecommendation.BLOCK.value,
        rationale="Required hard gate failed; artifact is blocked.",
    )


def _provider_degradation(base: CalibrationArtifact) -> CalibrationArtifact:
    degraded = CapabilityEvent(
        capability="strict_schema",
        requested="strict_schema",
        applied="json_object_parse",
        reason="deterministic degradation fixture",
        user_visible_note=(
            "Strict schema was unavailable in this fixture; local validation "
            "remained visible but the guarded boundary was crossed."
        ),
        affects_guarded_boundary=True,
    )
    relaxed = RelaxedSafeguard(
        name="strict_schema_native_path",
        reason="expedition fixture records provider capability degradation",
        increased_risk="Structured output guarantee is weaker than guarded mode.",
    )
    return _clone(
        base,
        selected_profile=ExecutionProfile.EXPEDITION.value,
        degraded_capabilities=[degraded.model_dump(mode="json")],
        relaxed_safeguards=[relaxed.model_dump(mode="json")],
        stayed_within_guarded_boundaries=False,
        guarded_boundary_crossed=True,
        additional_uplift_from_boundary_crossing=0.0,
        status="OK",
        ship_recommendation=ShipRecommendation.EXPERIMENT.value,
        rationale="Provider capability degradation is explicit; expedition-only artifact.",
    )


def _diff_regression_candidate(base: CalibrationArtifact) -> CalibrationArtifact:
    walk_forward = evaluate_walk_forward(0.70, 0.68, max_gap=0.30, min_kc4=0.30)
    return _clone(
        base,
        calibrated_fitness=0.70,
        best_fitness=0.70,
        uplift_absolute=0.70 - base.neutral_fitness,
        uplift_percent=((0.70 - base.neutral_fitness) / base.neutral_fitness * 100.0),
        quality_per_cost_best=0.0,
        quality_per_latency_best=0.0,
        walk_forward=walk_forward.model_dump(mode="json"),
        status="OK",
        ship_recommendation=ShipRecommendation.SHIP.value,
        rationale="Individually shippable but regresses against the clean golden baseline.",
    )


def _integrity_classification_from_artifact(
    artifact: CalibrationArtifact,
    report,
) -> str:
    if report is not None:
        if not report.schema_valid:
            return "schema_error"
        if not report.valid:
            return "integrity_error"
        if report.release_approved:
            return "release_approved"
        return "valid_non_release"
    if (
        artifact.status == "OK"
        and artifact.ship_recommendation == ShipRecommendation.SHIP
        and artifact.stayed_within_guarded_boundaries
        and not artifact.guarded_boundary_crossed
        and not artifact.relaxed_safeguards
    ):
        return "release_approved"
    return "valid_non_release"


def _enum_value(value) -> str:
    return str(getattr(value, "value", value))


def main() -> None:
    manifest = write_golden_artifacts()
    print("Wrote deterministic golden artifacts:")
    for case in manifest["cases"]:
        print(
            "  {case_id}: {artifact} status={status} ship={ship} hash={hash}".format(
                case_id=case["case_id"],
                artifact=case["artifact"],
                status=case["expected_status"],
                ship=case["expected_ship_recommendation"],
                hash=case["normalized_artifact_hash"],
            )
        )
    print("Wrote examples/reference/golden_manifest.json")


if __name__ == "__main__":
    main()
