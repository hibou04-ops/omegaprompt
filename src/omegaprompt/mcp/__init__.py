"""MCP server exposing the eight runtime entrypoints as agent-callable tools.

Run with:

    python -m omegaprompt.mcp           # stdio transport (Claude Code, Cursor)
    python -m omegaprompt.mcp --http    # streamable-http transport

Or import the FastMCP app directly:

    from omegaprompt.mcp import mcp_app

The eight tools mirror :mod:`omegaprompt.runtime`:

* ``calibrate``           — full calibration pipeline → CalibrationArtifact
* ``evaluate``            — single-config evaluation → EvalResult
* ``measure_sensitivity`` — axis-stress probe → SensitivityResult
* ``grade``               — score one response → JudgeResult
* ``report``              — artifact → markdown
* ``diff``                — compare two artifacts → ArtifactDiff
* ``preflight``           — environment sanity check → PreflightReport
* ``classify_traps``      — antemortem trap classification → list of findings
"""

from __future__ import annotations

MCP_MISSING_MESSAGE = (
    "TOOLING_MISSING: omegaprompt MCP support requires the optional MCP "
    "extra. Install with `pip install omegaprompt[mcp]`."
)


def _is_missing_mcp_dependency(exc: ModuleNotFoundError) -> bool:
    missing_name = getattr(exc, "name", "") or ""
    return missing_name == "mcp" or missing_name.startswith("mcp.")


def __getattr__(name: str):
    if name != "mcp_app":
        raise AttributeError(name)
    try:
        from omegaprompt.mcp.server import mcp_app
    except ModuleNotFoundError as exc:
        if _is_missing_mcp_dependency(exc):
            raise ModuleNotFoundError(MCP_MISSING_MESSAGE, name="mcp") from exc
        raise
    return mcp_app


__all__ = ["mcp_app", "MCP_MISSING_MESSAGE"]
