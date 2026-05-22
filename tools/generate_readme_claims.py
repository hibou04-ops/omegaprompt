#!/usr/bin/env python
"""Generate and validate the public README claim ledger.

This script is offline by design. Evidence is limited to local source files,
deterministic reference artifacts, and reproducible commands recorded in the
ledger. It never calls provider APIs, GitHub, or PyPI.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, TextIO


LEDGER_PATH = Path("docs/claims/public_claim_ledger.json")
GENERATED_PATH = Path("docs/claims/README_CLAIMS.generated.md")

EVIDENCE_TYPES = {
    "source_of_truth",
    "generated_doc",
    "reproducible_command",
    "deterministic_artifact",
    "qualitative_marker",
}
OWNER_SURFACES = {
    "repo",
    "PyPI",
    "CLI",
    "MCP",
    "provider",
    "artifact",
    "CI",
    "release",
}
DRIFT_RISKS = {"low", "medium", "high"}
REQUIRED_CLAIM_FIELDS = {
    "claim_id",
    "locations",
    "evidence_type",
    "source",
    "drift_risk",
    "exact_numeric_values_allowed",
    "owner_surface",
}
PUBLIC_DOCS = [
    "README.md",
    "README_KR.md",
    "EASY_README.md",
    "EASY_README_KR.md",
]
GENERATED_MARKER = "<!-- generated from docs/claims/public_claim_ledger.json; do not edit by hand -->"


def load_ledger(root: Path) -> dict[str, Any]:
    path = root / LEDGER_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing ledger: {LEDGER_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {LEDGER_PATH}: {exc}") from exc


def validate_ledger(ledger: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    if ledger.get("schema_version") != "1.0":
        errors.append("ledger schema_version must be '1.0'")
    if ledger.get("generated_doc") != GENERATED_PATH.as_posix():
        errors.append(f"ledger generated_doc must be {GENERATED_PATH.as_posix()!r}")

    claims = ledger.get("claims")
    if not isinstance(claims, list) or not claims:
        errors.append("ledger claims must be a non-empty list")
        return errors

    seen: set[str] = set()
    for index, claim in enumerate(claims):
        prefix = f"claims[{index}]"
        if not isinstance(claim, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = sorted(REQUIRED_CLAIM_FIELDS - set(claim))
        if missing:
            errors.append(f"{prefix} missing required fields: {missing}")
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id:
            errors.append(f"{prefix}.claim_id must be a non-empty string")
            claim_id = f"<invalid-{index}>"
        elif claim_id in seen:
            errors.append(f"duplicate claim_id: {claim_id}")
        seen.add(str(claim_id))

        if not (claim.get("claim") or claim.get("claim_pattern")):
            errors.append(f"{claim_id} must define claim or claim_pattern")
        if claim.get("evidence_type") not in EVIDENCE_TYPES:
            errors.append(f"{claim_id} has invalid evidence_type: {claim.get('evidence_type')!r}")
        if claim.get("owner_surface") not in OWNER_SURFACES:
            errors.append(f"{claim_id} has invalid owner_surface: {claim.get('owner_surface')!r}")
        if claim.get("drift_risk") not in DRIFT_RISKS:
            errors.append(f"{claim_id} has invalid drift_risk: {claim.get('drift_risk')!r}")
        if not isinstance(claim.get("exact_numeric_values_allowed"), bool):
            errors.append(f"{claim_id}.exact_numeric_values_allowed must be boolean")

        locations = claim.get("locations")
        if not isinstance(locations, list) or not locations:
            errors.append(f"{claim_id}.locations must be a non-empty list")
        else:
            for location in locations:
                if not isinstance(location, str):
                    errors.append(f"{claim_id}.locations entries must be strings")
                    continue
                if location == GENERATED_PATH.as_posix():
                    continue
                if _is_file_location(location) and not (root / _location_path(location)).exists():
                    errors.append(f"{claim_id} location does not exist: {location}")

        source = claim.get("source")
        if isinstance(source, str):
            _validate_source(claim_id=str(claim_id), source=source, root=root, errors=errors)
        else:
            errors.append(f"{claim_id}.source must be a string")

        if claim.get("claim_pattern"):
            try:
                re.compile(str(claim["claim_pattern"]))
            except re.error as exc:
                errors.append(f"{claim_id}.claim_pattern is invalid regex: {exc}")

        if claim.get("unsupported") and claim.get("exact_numeric_values_allowed"):
            errors.append(f"{claim_id} is unsupported but allows exact numeric values")

    required_ids = {
        "identity.names",
        "install.core_and_mcp_extra",
        "artifact.schema_version",
        "mcp.tool_contract",
        "provider.gemini_status",
        "ci.no_live_provider_calls",
        "reference.metrics",
        "tests.no_exact_readme_prose_counts",
        "unsupported.downloads_adoption",
    }
    missing_ids = sorted(required_ids - seen)
    if missing_ids:
        errors.append(f"ledger missing required claim coverage ids: {missing_ids}")
    return errors


def _is_file_location(location: str) -> bool:
    if location.startswith("PyPI description"):
        return False
    return bool(re.match(r"^[A-Za-z0-9_.\\/\-]+(?:[:#].*)?$", location))


def _location_path(location: str) -> Path:
    clean = location.split("#", 1)[0]
    if re.search(r":\d+$", clean):
        clean = clean.rsplit(":", 1)[0]
    return Path(clean)


def _validate_source(*, claim_id: str, source: str, root: Path, errors: list[str]) -> None:
    if source.startswith("command:"):
        command = source.removeprefix("command:").strip()
        if not command:
            errors.append(f"{claim_id} has empty reproducible command")
        lowered = command.lower()
        if "api_key" in lowered or "live_provider" in lowered or "curl " in lowered:
            errors.append(f"{claim_id} command source looks live/networked: {command}")
        return
    for candidate in re.split(r"[;,]", source):
        candidate = candidate.strip()
        if not candidate or candidate.startswith("see "):
            continue
        path_text = candidate.split("#", 1)[0].strip()
        if re.search(r"\s", path_text):
            continue
        if any(path_text.endswith(suffix) for suffix in (".py", ".md", ".json", ".toml", ".yml", ".yaml")):
            if not (root / path_text).exists():
                errors.append(f"{claim_id} source path does not exist: {path_text}")


def render_claims_doc(ledger: dict[str, Any], root: Path) -> str:
    claims = sorted(ledger["claims"], key=lambda c: c["claim_id"])
    lines = [
        "# README Public Claim Ledger",
        "",
        GENERATED_MARKER,
        "",
        "This file is generated by `python tools/generate_readme_claims.py`.",
        "It records public README/PyPI-facing claims and the local evidence allowed to support them.",
        "",
        "## Claims",
        "",
        "| Claim ID | Owner | Evidence | Exact Numbers | Drift Risk | Locations | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    for claim in claims:
        lines.append(
            "| {claim_id} | {owner} | {evidence} | {exact} | {risk} | {locations} | {source} |".format(
                claim_id=_md(claim["claim_id"]),
                owner=_md(claim["owner_surface"]),
                evidence=_md(claim["evidence_type"]),
                exact="yes" if claim["exact_numeric_values_allowed"] else "no",
                risk=_md(claim["drift_risk"]),
                locations=_md(", ".join(claim["locations"])),
                source=_md(claim["source"]),
            )
        )

    lines.extend(
        [
            "",
            "## Generated Reference Metrics",
            "",
            "These exact values are allowed only because they come from a deterministic artifact or recorded reproducible command.",
            "",
        ]
    )
    artifact = _load_json_if_exists(root / "examples/reference/reference_artifact.json")
    if isinstance(artifact, dict):
        walk = artifact.get("walk_forward") or {}
        lines.extend(
            [
                f"- `schema_version`: `{artifact.get('schema_version')}`",
                f"- `status`: `{artifact.get('status')}`",
                f"- `ship_recommendation`: `{artifact.get('ship_recommendation')}`",
                f"- `neutral_fitness`: `{_fmt_float(artifact.get('neutral_fitness'))}`",
                f"- `calibrated_fitness`: `{_fmt_float(artifact.get('calibrated_fitness'))}`",
                f"- `uplift_absolute`: `{_fmt_float(artifact.get('uplift_absolute'))}`",
                f"- `uplift_percent`: `{_fmt_float(artifact.get('uplift_percent'))}`",
                f"- `walk_forward.test_fitness`: `{_fmt_float(walk.get('test_fitness'))}`",
                f"- `walk_forward.kc4_status`: `{walk.get('kc4_status')}`",
            ]
        )
    else:
        lines.append("- Reference artifact unavailable.")

    lines.extend(
        [
            "",
            "## Unsupported Claim Classes",
            "",
            "- Exact README prose test counts are unsupported. The static top badge is preserved separately.",
            "- Download, user, adoption, and popularity counts are unsupported unless a source-of-truth claim is added to the ledger.",
            "- Provider superiority, model superiority, prompt superiority, and benchmark aggregates require deterministic artifacts or reproducible commands.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_json_if_exists(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _fmt_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return "null"


def _md(value: str) -> str:
    return str(value).replace("|", "\\|")


def scan_unsupported_public_claims(ledger: dict[str, Any], root: Path) -> list[str]:
    exact_allowed = [
        re.compile(str(claim["claim_pattern"]))
        for claim in ledger.get("claims", [])
        if claim.get("exact_numeric_values_allowed") and claim.get("claim_pattern")
    ]
    errors: list[str] = []
    for rel in PUBLIC_DOCS:
        path = root / rel
        if not path.exists():
            continue
        for line_no, line in _iter_public_prose_lines(path):
            if _is_allowed_exact_line(line, exact_allowed):
                continue
            lowered = line.lower()
            external_project_line = "omega-lock" in lowered or "antemortem" in lowered
            current_repo_line = (
                "current head" in lowered
                or "test suite" in lowered
                or "ci" in lowered
                or "live api call" in lowered
            )
            if re.search(r"(?<![\d.])\d{2,4}\s+tests?\b", line, flags=re.IGNORECASE):
                if external_project_line and not current_repo_line:
                    continue
                errors.append(f"{rel}:{line_no}: exact README prose test count is unsupported: {line.strip()}")
            if re.search(r"\b\d[\d,]*(?:\+)?\s+(downloads?|users?|adopters?|installs?)\b", line, flags=re.IGNORECASE):
                errors.append(f"{rel}:{line_no}: exact adoption/download claim is unsupported: {line.strip()}")
            if _looks_like_uncovered_benchmark_metric(line):
                errors.append(f"{rel}:{line_no}: exact benchmark metric needs ledger coverage: {line.strip()}")
    return errors


def _iter_public_prose_lines(path: Path) -> Iterable[tuple[int, str]]:
    in_code = False
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if stripped.startswith("[!["):
            continue
        if stripped.startswith("<!--"):
            continue
        yield line_no, line


def _is_allowed_exact_line(line: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(line) for pattern in patterns)


def _looks_like_uncovered_benchmark_metric(line: str) -> bool:
    if not re.search(r"(?<![A-Za-z])\d+\.\d+%?|\d+%", line):
        return False
    lowered = line.lower()
    if "illustrative" in lowered or "schema" in lowered:
        return False
    if lowered.lstrip().startswith("candidate "):
        return False
    if "threshold" in lowered or "default" in lowered or "min_kc4" in lowered or "max_gap" in lowered:
        return False
    if "unit test" in lowered or "hard-gate fitness collapse" in lowered or "hard gate" in lowered:
        return False
    if "v0." in lowered or "~" in lowered:
        return False
    metric_words = (
        "baseline",
        "calibrated",
        "uplift",
        "fitness",
        "hard-gate pass rate",
        "walk_forward.test_fitness",
    )
    if not any(word in lowered for word in metric_words):
        return False
    numeric_metric = re.search(r"(`?\d+(?:\.\d+)?%?`?)", line)
    return bool(numeric_metric)


def generate(root: Path, *, check: bool = False) -> tuple[int, list[str]]:
    messages: list[str] = []
    try:
        ledger = load_ledger(root)
    except ValueError as exc:
        return 1, [str(exc)]

    errors = validate_ledger(ledger, root)
    errors.extend(scan_unsupported_public_claims(ledger, root))
    if errors:
        return 1, errors

    rendered = render_claims_doc(ledger, root) + "\n"
    output_path = root / GENERATED_PATH
    if check:
        try:
            current = output_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return 1, [f"generated claim doc missing: {GENERATED_PATH}"]
        if current != rendered:
            return 1, [f"generated claim doc is stale: {GENERATED_PATH}"]
        return 0, [f"{GENERATED_PATH} is fresh"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return 0, [f"wrote {GENERATED_PATH}"]


def main(argv: list[str] | None = None, *, root: Path | None = None, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate README claim evidence docs.")
    parser.add_argument("--check", action="store_true", help="Fail if generated claim docs are stale.")
    args = parser.parse_args(argv)

    repo_root = (root or Path.cwd()).resolve()
    code, messages = generate(repo_root, check=args.check)
    stream = stdout or sys.stdout
    for message in messages:
        _write_line(stream, message)
    return code


def _write_line(stream: TextIO, text: str) -> None:
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        stream.write(safe + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
