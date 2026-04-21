"""Calibration artifact read/write.

Artifacts are the system of record for a completed calibration. They are
JSON-on-disk, schema-versioned, and intended to be diffed across runs in
CI to detect regression.
"""

from __future__ import annotations

from pathlib import Path

from omegaprompt.domain.result import CalibrationArtifact


def save_artifact(artifact: CalibrationArtifact, path: str | Path) -> None:
    """Write the artifact as pretty-printed JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_artifact(path: str | Path) -> CalibrationArtifact:
    """Read and validate an artifact from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {p}")
    return CalibrationArtifact.model_validate_json(p.read_text(encoding="utf-8"))
