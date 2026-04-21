"""Provider-neutral target protocol for the calibration engine."""

from __future__ import annotations

from typing import Any, Protocol

from omegaprompt.domain.result import EvalResult


class CalibrableTarget(Protocol):
    """Minimal target surface the search engine depends on."""

    def param_space(self) -> list[Any]: ...

    def neutral_params(self) -> dict[str, Any]: ...

    def evaluate(self, params: dict[str, Any] | None) -> EvalResult: ...
