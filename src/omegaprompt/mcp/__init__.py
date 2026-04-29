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

from omegaprompt.mcp.server import mcp_app

__all__ = ["mcp_app"]
