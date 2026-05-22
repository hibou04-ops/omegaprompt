"""Deterministic no-network demo for the examples gallery.

The demo reads the checked-in golden reference artifact and verifies it with
the same integrity checker used by CI-oriented commands. It does not call LLM
providers, does not need API keys, and does not depend on wall-clock state.

Run::

    PYTHONIOENCODING=utf-8 python examples/demo_calibration.py

Refresh the replay snapshot after intentional output changes::

    PYTHONIOENCODING=utf-8 python examples/demo_calibration.py > examples/_demo_output.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaprompt.core.artifact import load_artifact
from omegaprompt.core.artifact_integrity import check_artifact_integrity

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / "reference"

DATASET = HERE / "sample_dataset.jsonl"
RUBRIC = HERE / "rubric_example.json"
VARIANTS = HERE / "variants_example.json"
REFERENCE_ARTIFACT = REFERENCE_DIR / "reference_artifact.json"
MANIFEST = REFERENCE_DIR / "golden_manifest.json"


def _section(label: str) -> None:
    print(f"\n---- {label} ----")


def _metric(value: float) -> str:
    return f"{value:.4f}"


def _percent(value: float) -> str:
    return f"{value:.2f}%"


def _require_files(paths: list[Path]) -> bool:
    missing = [path for path in paths if not path.exists()]
    for path in missing:
        print(f"ERROR: missing fixture {path}", file=sys.stderr)
    return not missing


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_case(manifest: dict, case_id: str) -> dict:
    for case in manifest.get("cases", []):
        if case.get("case_id") == case_id:
            return case
    raise KeyError(f"missing manifest case: {case_id}")


def main() -> int:
    if not _require_files([DATASET, RUBRIC, VARIANTS, REFERENCE_ARTIFACT, MANIFEST]):
        return 1

    report = check_artifact_integrity(REFERENCE_ARTIFACT)
    if not report.valid:
        print("ERROR: reference artifact failed integrity checks", file=sys.stderr)
        for finding in report.findings:
            print(f"{finding.severity} {finding.id}: {finding.message}", file=sys.stderr)
        return 1

    artifact = load_artifact(REFERENCE_ARTIFACT)
    manifest = _load_json(MANIFEST)
    clean_case = _manifest_case(manifest, "clean_ok_ship")
    rubric = _load_json(RUBRIC)
    variants = _load_json(VARIANTS)
    items = sum(1 for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip())
    top_axes = artifact.sensitivity_ranking[:3]

    print("=== omegaprompt deterministic offline demo ===")
    print("mode: no API keys, no network, deterministic in-memory reference providers")
    print("reproduce: python examples/reference/reproduce_reference_artifact.py")
    print("report:    omegaprompt report examples/reference/reference_artifact.json")

    _section("Inputs")
    print(f"dataset:  examples/sample_dataset.jsonl     ({items} items)")
    print(f"rubric:   examples/rubric_example.json      ({len(rubric.get('dimensions', []))} dimensions)")
    print(
        "variants: examples/variants_example.json    "
        f"({len(variants.get('system_prompts', []))} system prompts, "
        f"{len(variants.get('few_shot_examples', []))} few-shot examples)"
    )

    _section("Reference artifact integrity")
    print(f"schema_version: {artifact.schema_version}")
    print(f"status: {artifact.status.value}")
    print(f"ship_recommendation: {artifact.ship_recommendation.value}")
    print(f"release_approved: {report.release_approved}")
    print(f"strict_blocking_findings: {report.strict_blocking_findings}")
    print(f"normalized_hash: {clean_case['normalized_artifact_hash']}")

    _section("Deterministic metrics from the artifact")
    print(f"neutral_fitness:    {_metric(artifact.neutral_fitness)}")
    print(f"calibrated_fitness: {_metric(artifact.calibrated_fitness)}")
    print(f"uplift_absolute:    {_metric(artifact.uplift_absolute)}")
    print(f"uplift_percent:     {_percent(artifact.uplift_percent)}")
    if artifact.walk_forward is not None:
        print(f"walk_forward_mode:  {artifact.walk_forward.validation_mode}")
        print(f"test_fitness:       {_metric(artifact.walk_forward.test_fitness)}")
        print(f"generalization_gap: {_percent(artifact.walk_forward.generalization_gap * 100)}")
        print(f"kc4_status:         {artifact.walk_forward.kc4_status}")
        print(f"walk_forward_passed: {artifact.walk_forward.passed}")

    _section("Sensitivity ranking")
    for row in top_axes:
        axis = str(row.get("axis", "unknown"))
        stress = float(row.get("gini_delta") or 0.0)
        raw = float(row.get("raw_stress") or 0.0)
        print(f"{int(row.get('rank', 0)) + 1}. {axis}: stress={stress:.4f}, raw={raw:.4f}")

    _section("Live provider path")
    print("offline demo: examples/demo_replay.py and examples/reference/*")
    print("live examples: task directories under examples/; opt-in only, API keys required")
    print("default CI: no live provider calls")
    print("gallery: examples/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
