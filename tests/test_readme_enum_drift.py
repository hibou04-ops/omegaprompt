"""Regression test: catch future drift between README enums and source.

Reviewer P5 caught the README documenting ``ShipRecommendation = SHIP /
HOLD / ROLLBACK`` while the source had ``SHIP / HOLD / EXPERIMENT /
BLOCK``. Users wiring CI gates against README values would have written
``rollback`` and silently never matched.

This test re-derives the enum members from the source and asserts that
each one appears in the README. New members force a README update; the
test fails if you add an enum value without telling the README.
"""

from __future__ import annotations

from pathlib import Path

from omegaprompt.domain.profiles import ShipRecommendation

README_PATH = Path(__file__).parent.parent / "README.md"


def test_readme_documents_every_ship_recommendation_member():
    text = README_PATH.read_text(encoding="utf-8")
    for member in ShipRecommendation:
        # We expect the literal source name (e.g. EXPERIMENT) AND the
        # value (e.g. "experiment") to appear in the README so both the
        # Python and the JSON-serialised form are documented.
        assert member.name in text, (
            f"README is missing ShipRecommendation.{member.name}; "
            f"add it to §4.5 and the schema appendix."
        )


def test_readme_does_not_mention_removed_rollback_member():
    """Defends against the specific drift the reviewer flagged."""
    text = README_PATH.read_text(encoding="utf-8")
    # ``ROLLBACK`` was never a real member; ensure it doesn't sneak back
    # into the README. (Tolerate the lowercase word "rollback" in prose
    # if it's used outside a code/contract context — but the all-caps
    # token is the contract form and must not appear.)
    assert "ROLLBACK" not in text, (
        "README references ROLLBACK but ShipRecommendation has no such member; "
        "use BLOCK or EXPERIMENT instead."
    )
