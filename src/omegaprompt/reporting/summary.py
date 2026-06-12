"""Stable, CI-consumable JSON summary for a calibration artifact.

``report --format json`` emits this compact, schema-versioned summary
instead of dumping the full artifact. It is a deterministic projection of
the artifact (no timestamps, no latency, sorted keys when serialized) so
CI can diff it byte-for-byte and golden tests stay stable.
"""

from __future__ import annotations

import json
from typing import Any

from omegaprompt.core.overfit import extract_overfit_metrics

# Versioned independently of the artifact schema so the summary surface can
# evolve without touching the frozen artifact contract.
REPORT_SUMMARY_SCHEMA_VERSION = "1.0"


def build_report_summary(a) -> dict[str, Any]:
    """Return a deterministic, machine-readable summary dict for ``a``."""

    overfit = extract_overfit_metrics(a)
    return {
        "summary_schema_version": REPORT_SUMMARY_SCHEMA_VERSION,
        "artifact_schema_version": a.schema_version,
        "engine_name": a.engine_name,
        "status": str(getattr(a.status, "value", a.status)),
        "ship_recommendation": a.ship_recommendation.value,
        "selected_profile": a.selected_profile.value,
        "method": a.method,
        "unlock_k": a.unlock_k,
        "neutral_fitness": a.neutral_fitness,
        "calibrated_fitness": a.calibrated_fitness,
        "uplift_absolute": a.uplift_absolute,
        "uplift_percent": a.uplift_percent,
        "hard_gate_pass_rate": a.hard_gate_pass_rate,
        "quality_per_cost_best": a.quality_per_cost_best,
        "quality_per_latency_best": a.quality_per_latency_best,
        "stayed_within_guarded_boundaries": a.stayed_within_guarded_boundaries,
        "guarded_boundary_crossed": a.guarded_boundary_crossed,
        "degraded_capability_count": len(a.degraded_capabilities),
        "relaxed_safeguard_count": len(a.relaxed_safeguards),
        "boundary_warning_count": len(a.boundary_warnings),
        "target_provider": a.target_provider,
        "target_model": a.target_model,
        "judge_provider": a.judge_provider,
        "judge_model": a.judge_model,
        # The prominent "is my prompt overfit" block.
        "overfit": overfit.model_dump(mode="json"),
    }


def render_summary_json(a) -> str:
    """Deterministic JSON string (sorted keys, stable separators)."""

    return json.dumps(
        build_report_summary(a),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
