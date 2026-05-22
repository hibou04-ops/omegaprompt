from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "tools" / "generate_readme_claims.py"
SPEC = importlib.util.spec_from_file_location("generate_readme_claims", SCRIPT_PATH)
assert SPEC and SPEC.loader
claims_tool = importlib.util.module_from_spec(SPEC)
sys.modules["generate_readme_claims"] = claims_tool
SPEC.loader.exec_module(claims_tool)


def test_public_claim_ledger_schema_and_sources_are_valid() -> None:
    ledger = claims_tool.load_ledger(ROOT)
    errors = claims_tool.validate_ledger(ledger, ROOT)
    assert errors == []


def test_generated_claim_doc_is_fresh_and_byte_stable() -> None:
    ledger = claims_tool.load_ledger(ROOT)
    expected = claims_tool.render_claims_doc(ledger, ROOT) + "\n"
    actual = (ROOT / "docs/claims/README_CLAIMS.generated.md").read_text(encoding="utf-8")
    assert actual == expected

    code, messages = claims_tool.generate(ROOT, check=True)
    assert code == 0
    assert any("is fresh" in message for message in messages)


def test_check_mode_fails_when_generated_doc_is_stale(monkeypatch) -> None:
    monkeypatch.setattr(claims_tool, "render_claims_doc", lambda ledger, root: "stale replacement")
    code, messages = claims_tool.generate(ROOT, check=True)
    assert code == 1
    assert any("stale" in message for message in messages)


def test_major_public_claims_have_ledger_coverage() -> None:
    ledger = claims_tool.load_ledger(ROOT)
    by_id = {claim["claim_id"]: claim for claim in ledger["claims"]}

    required = {
        "identity.names": "repo",
        "install.core_and_mcp_extra": "PyPI",
        "artifact.schema_version": "artifact",
        "mcp.tool_contract": "MCP",
        "provider.gemini_status": "provider",
        "ci.no_live_provider_calls": "CI",
        "reference.metrics": "artifact",
        "tests.no_exact_readme_prose_counts": "CI",
        "unsupported.downloads_adoption": "release",
    }
    for claim_id, owner in required.items():
        assert claim_id in by_id
        assert by_id[claim_id]["owner_surface"] == owner
        assert by_id[claim_id]["evidence_type"] in claims_tool.EVIDENCE_TYPES

    assert by_id["reference.metrics"]["exact_numeric_values_allowed"] is True
    assert by_id["tests.no_exact_readme_prose_counts"]["exact_numeric_values_allowed"] is False
    assert by_id["unsupported.downloads_adoption"]["unsupported"] is True


def test_generated_doc_reports_reference_artifact_metrics() -> None:
    artifact = json.loads((ROOT / "examples/reference/reference_artifact.json").read_text(encoding="utf-8"))
    generated = (ROOT / "docs/claims/README_CLAIMS.generated.md").read_text(encoding="utf-8")
    assert f"`schema_version`: `{artifact['schema_version']}`" in generated
    assert f"`status`: `{artifact['status']}`" in generated
    assert f"`ship_recommendation`: `{artifact['ship_recommendation']}`" in generated
    assert "`neutral_fitness`: `0.425`" in generated
    assert "`calibrated_fitness`: `0.925`" in generated


def test_scan_rejects_unsupported_exact_test_counts_and_adoption_claims(tmp_path: Path) -> None:
    ledger = claims_tool.load_ledger(ROOT)
    (tmp_path / "README.md").write_text(
        "The current head passes 999 tests.\n"
        "The package has 10,000 downloads.\n",
        encoding="utf-8",
    )
    for name in ("README_KR.md", "EASY_README.md", "EASY_README_KR.md"):
        (tmp_path / name).write_text("# empty\n", encoding="utf-8")

    errors = claims_tool.scan_unsupported_public_claims(ledger, tmp_path)
    assert any("exact README prose test count" in err for err in errors)
    assert any("adoption/download" in err for err in errors)


def test_scan_allows_deterministic_reference_metric_patterns(tmp_path: Path) -> None:
    ledger = claims_tool.load_ledger(ROOT)
    (tmp_path / "README.md").write_text(
        "Calibration earned +117.6% over the neutral baseline.\n",
        encoding="utf-8",
    )
    for name in ("README_KR.md", "EASY_README.md", "EASY_README_KR.md"):
        (tmp_path / name).write_text("# empty\n", encoding="utf-8")

    assert claims_tool.scan_unsupported_public_claims(ledger, tmp_path) == []
