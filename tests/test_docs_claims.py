from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


TRUST_DOCS = [
    "docs/trust-model.md",
    "docs/toolkit-positioning.md",
    "docs/provider-capabilities.md",
    "docs/profiles-and-risk-boundaries.md",
]


def test_readmes_link_trust_docs_without_touching_badge_claims() -> None:
    readme = _read("README.md")
    readme_kr = _read("README_KR.md")

    for rel in TRUST_DOCS:
        assert f"]({rel})" in readme
        assert f"]({rel})" in readme_kr

    badge_lines = [
        line
        for line in readme.splitlines()[:20]
        if line.strip().startswith("[![")
    ]
    assert [line.split("]", 1)[0] for line in badge_lines] == [
        "[![CI",
        "[![License: Apache 2.0",
        "[![Python",
        "[![PyPI",
        "[![Tests",
        "[![Artifact schema",
        "[![MCP",
        "[![Parent framework",
    ]


def test_trust_model_covers_required_trust_boundaries() -> None:
    text = _read("docs/trust-model.md").lower()
    required_phrases = [
        "what the artifact proves",
        "what the artifact does not prove",
        "train/test split",
        "walk-forward",
        "kc4",
        "paired",
        "disjoint",
        "pre-declared",
        "holdout correlation",
        "cross-vendor",
        "self-agreement",
        "guarded",
        "expedition",
        "capabilityevent",
        "offline",
        "live provider",
        "default ci",
        "omegaprompt[mcp]",
        "diff",
        "superiority claims",
    ]

    for phrase in required_phrases:
        assert phrase in text


def test_toolkit_positioning_covers_name_and_scope_boundaries() -> None:
    text = _read("docs/toolkit-positioning.md").lower()
    required_phrases = [
        "omegaprompt vs omega-lock",
        "omegaprompt vs antemortem-cli",
        "mini-omega-lock",
        "mini-antemortem-cli",
        "omegacal",
        "compatibility alias",
        "no dashboard",
        "web application",
        "omegaprompt[mcp]",
    ]

    for phrase in required_phrases:
        assert phrase in text


def test_provider_docs_match_contract_claims() -> None:
    text = _read("docs/provider-capabilities.md")

    assert "| Anthropic | Tier 2 cloud-grade | First-class | Ship-grade |" in text
    assert "| OpenAI | Tier 2 cloud-grade | First-class | Ship-grade |" in text
    assert "| Gemini | Tier 2 cloud-grade | Implemented target adapter | Not ship-grade judge |" in text
    assert "Experimental target path | Exploration-grade judge only" in text
    assert "`placeholder=False`" in text
    assert "`ship_grade_judge=False`" in text
    assert "cloud-equivalent judge semantics" in text
    assert "`CapabilityEvent`" in text
    assert "Default" in text and "CI uses mocks and deterministic reference artifacts" in text


def test_profiles_doc_states_expedition_and_diff_limits() -> None:
    text = _read("docs/profiles-and-risk-boundaries.md").lower()

    assert "expedition is not ship-grade by default" in text
    assert "`paired`" in text
    assert "`disjoint`" in text
    assert "`auto`" in text
    assert "diff regression use" in text
    assert "block the ci gate" in text


def test_trust_docs_do_not_make_superiority_or_guarantee_claims() -> None:
    combined = "\n".join(_read(rel).lower() for rel in TRUST_DOCS)
    forbidden_patterns = [
        "is the best provider",
        "is the best model",
        "therefore the provider is superior",
        "therefore the model is superior",
        "guarantees production success",
        "proves that production will succeed",
    ]

    for pattern in forbidden_patterns:
        assert pattern not in combined
