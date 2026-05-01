"""Smoke tests for the omegaprompt MCP server.

Verifies that all eight runtime entrypoints are registered as MCP tools
and that their generated JSON schemas reflect the documented argument
shapes. Tool execution is covered by the runtime tests; this layer is
exercised purely for the wiring contract.
"""

from __future__ import annotations

import asyncio

import pytest

mcp = pytest.importorskip("mcp")


@pytest.fixture(scope="module")
def mcp_app():
    from omegaprompt.mcp import mcp_app as app

    return app


@pytest.fixture(scope="module")
def tools(mcp_app):
    return asyncio.run(mcp_app.list_tools())


EXPECTED_TOOLS = {
    "calibrate",
    "evaluate",
    "report",
    "diff",
    "measure_sensitivity",
    "grade",
    "preflight",
    "classify_traps",
}


def test_all_eight_runtime_entrypoints_registered(tools):
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOLS


def test_each_tool_has_description(tools):
    for tool in tools:
        assert tool.description, f"tool {tool.name!r} has no description"
        assert len(tool.description) > 50, (
            f"tool {tool.name!r} description is too short for agents to use"
        )


def test_each_tool_has_input_schema(tools):
    for tool in tools:
        assert tool.inputSchema is not None
        assert "properties" in tool.inputSchema
        assert tool.inputSchema["properties"], (
            f"tool {tool.name!r} declares no input properties"
        )


def test_calibrate_required_args_match_runtime(tools):
    calibrate = next(t for t in tools if t.name == "calibrate")
    required = set(calibrate.inputSchema.get("required", []))
    assert {"train", "rubric", "variants", "target"}.issubset(required)


def test_mcp_calibrate_accepts_adaptation_plan(tools):
    """Reviewer P0 #4: MCP must reach CLI/Python parity for tuning AND
    adaptation_plan. Pre-fix the tool signature took ``tuning`` but not
    ``adaptation_plan``, so an agent had no way to thread a serialized
    plan through MCP."""
    calibrate = next(t for t in tools if t.name == "calibrate")
    properties = calibrate.inputSchema["properties"]
    assert "adaptation_plan" in properties
    assert "tuning" in properties


def test_grade_required_args_match_runtime(tools):
    grade = next(t for t in tools if t.name == "grade")
    required = set(grade.inputSchema.get("required", []))
    assert {"rubric", "item", "response", "provider"}.issubset(required)


def test_preflight_required_args_match_runtime(tools):
    preflight = next(t for t in tools if t.name == "preflight")
    required = set(preflight.inputSchema.get("required", []))
    assert {"target", "judge"}.issubset(required)
