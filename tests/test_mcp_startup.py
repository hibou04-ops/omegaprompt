"""The official entrypoint initializes the engine before tool dispatch starts."""
import importlib

import pytest


@pytest.mark.parametrize("arguments,transport", [([], "stdio"), (["--http"], "streamable-http")])
def test_official_entrypoint_initializes_engine_before_transport(monkeypatch, arguments, transport):
    pytest.importorskip("mcp")
    from omegaprompt.mcp import __main__ as entrypoint
    from omegaprompt.mcp.server import mcp_app

    events = []
    original = importlib.import_module

    def observe_import(name, *args, **kwargs):
        result = original(name, *args, **kwargs)
        if name == "omega_lock":
            events.append("engine_ready")
        return result

    def start_transport(*, transport):
        events.append(transport)

    monkeypatch.setattr(importlib, "import_module", observe_import)
    monkeypatch.setattr(mcp_app, "run", start_transport)
    assert entrypoint.main(arguments) == 0
    assert events == ["engine_ready", transport]
