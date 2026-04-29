"""Entry point for ``python -m omegaprompt.mcp``.

Default transport is stdio (the standard for Claude Code, Cursor, and
other MCP clients that spawn the server as a subprocess). Pass ``--http``
to run streamable-http on the host/port configured on the FastMCP app
(127.0.0.1:8000 by default).

Examples
--------

Stdio (recommended, used by Claude Code's `mcpServers` config)::

    python -m omegaprompt.mcp

Streamable HTTP::

    python -m omegaprompt.mcp --http
"""

from __future__ import annotations

import argparse
import sys

from omegaprompt.mcp.server import mcp_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omegaprompt-mcp",
        description="MCP server exposing omegaprompt's eight runtime entrypoints.",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run streamable-http transport instead of the default stdio.",
    )
    args = parser.parse_args(argv)

    if args.http:
        mcp_app.run(transport="streamable-http")
    else:
        mcp_app.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
