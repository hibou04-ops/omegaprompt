"""Structural validity tests for the composite GitHub Action (2.1.0).

PyYAML is not a project dependency, so these checks are deliberately
text-structural: they assert the required composite-action keys, the
declared inputs/outputs, and that the action wraps ``omegaprompt gate``.
A minimal indentation-based parser confirms the document is at least
block-structured (no tabs, top-level keys present).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "action.yml"
EXAMPLE = ROOT / "examples" / "ci" / "ship-gate.yml"


def _action_text() -> str:
    return ACTION.read_text(encoding="utf-8")


def test_action_file_exists_at_repo_root() -> None:
    assert ACTION.exists(), "action.yml must live at the repo root for `uses:`"


def test_action_has_required_top_level_keys() -> None:
    text = _action_text()
    for key in ("name:", "description:", "inputs:", "outputs:", "runs:"):
        assert key in text


def test_action_is_a_composite_action() -> None:
    text = _action_text()
    assert 'using: "composite"' in text or "using: composite" in text


def test_action_declares_artifact_input() -> None:
    text = _action_text()
    assert "artifact:" in text
    assert "required: true" in text


def test_action_declares_format_and_generalization_inputs() -> None:
    text = _action_text()
    assert "format:" in text
    assert "require-generalization:" in text


def test_action_declares_passed_and_exit_code_outputs() -> None:
    text = _action_text()
    assert "passed:" in text
    assert "exit-code:" in text


def test_action_wraps_omegaprompt_gate() -> None:
    text = _action_text()
    assert "omegaprompt gate" in text


def test_action_uses_no_tabs() -> None:
    # YAML forbids tabs for indentation.
    assert "\t" not in _action_text()


def test_action_indentation_is_block_structured() -> None:
    # Lightweight sanity parse: every indented line is a multiple-of-2 indent
    # and there is at least one nested block under runs:.
    lines = _action_text().splitlines()
    nonblank = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    for ln in nonblank:
        indent = len(ln) - len(ln.lstrip(" "))
        assert indent % 2 == 0, f"odd indent: {ln!r}"
    assert any(ln.startswith("runs:") for ln in nonblank)


def test_example_workflow_exists_and_uses_pinned_action() -> None:
    assert EXAMPLE.exists()
    text = EXAMPLE.read_text(encoding="utf-8")
    assert "uses: hibou04-ops/omegaprompt@v2.1.0" in text
    assert "artifact:" in text
