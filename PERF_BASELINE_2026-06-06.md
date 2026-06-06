# omegaprompt performance baseline — measured 2026-06-06

Environment: Windows 11, Python 3.11.15, pydantic 2.13.3, omegaprompt 2.0.0.
Method: `.venv/Scripts/python.exe`, warm caches, `Measure-Command`, 5x median unless noted.

## End-to-end command latency (the denominator that matters)
| Command (real artifact) | median ms |
|---|---|
| report           | 231.7 |
| check-artifact   | 228.0 |
| diff             | 232.6 |
| --version        | 232.0 |
| --help           | 306.6 |

All four non-help commands cluster at ~228-233ms; compute is negligible, so the
floor is import/startup. --help adds ~75ms for rich/markdown rendering (only the
help path loads rich).

## Import decomposition (warm, 5x median)
| Import | median ms |
|---|---|
| bare python (`pass`)                                   |  32.0 |
| pydantic alone                                         | 101.7 |
| typer alone                                            |  87.3 |
| pydantic + typer (unavoidable per-command floor)       | 128.8 |
| `from omegaprompt.cli import app` (full CLI)           | 226.1 |
| `import omegaprompt` (full __init__)                   | 192.9 |
| `omegaprompt.commands.report` + `check_artifact` only  | 218.9 |
| `omegaprompt.commands.diff` (pulls runtime)            | 225.6 |
| `importlib.metadata.version('omegaprompt')` path       |  80.7 |

importtime cumulative attribution: omegaprompt.domain.dataset 110ms, domain 129ms,
core 144ms, omegaprompt total 156ms; pydantic_core ~60ms. rich NOT loaded on
`import app` (only on --help render).

## Decisive defer_build experiment (PROTOTYPED + reverted via git)
Applied `defer_build=True` to 16/19 domain models (3 had no model_config:
CapabilityEvent, ProviderCapabilities, Dataset), then re-measured end-to-end:

| Command | baseline ms | defer_build ms | delta |
|---|---|---|---|
| report         | 231.7 | 221.8 | -10 |
| check-artifact | 228.0 | 225.2 |  -3 |
| --version      | 232.0 | 221.9 | -10 |
| --help         | 306.6 | 293.5 | -13 |

In-process: `import omegaprompt.domain` 156.6 -> 146.9ms (only -10ms);
first validate 0.12 -> 2.02ms (deferred build now runs at validate).

CORRECTED MECHANISM: schema BUILD is only ~10ms (the deferrable part). The
~150ms floor is pydantic_core + typer IMPORT itself, which is intrinsic and
NOT removable by any config flag. Cross-check: pydantic-alone 101.7ms +
typer-alone 87.3ms == the floor. defer_build is a ~10ms lever, only on
non-validating paths; validating commands build the schema anyway (net ~0).

Validate compute is trivial: first 0.12ms, steady-state 0.015ms (avg 50).

## Theoretical CLI floor (bare typer app, NO omegaprompt imports)
- bare typer `--help`/error render: ~183.5 ms (typer pulls rich on render)
- bare typer run command (no help render): ~190 ms
=> Even a perfectly lazy CLI importing zero omegaprompt code cannot get below
~183ms; the floor is python + typer + rich. Lazy command registration's max
headroom is for --version/--help only; validating commands gain nothing.

## Structural facts (verified in source)
- `cli.py` line 5 `from omegaprompt import __version__` triggers the FULL __init__,
  AND lines 6-9 eagerly import all 4 command modules (which pull domain/runtime).
  So --version pays the full pydantic_core+typer import (~150ms intrinsic) for
  nothing. Making __version__ cheap alone saves only ~7ms because the command
  modules still import domain regardless.
- `commands/__init__.py` imports all 4 command modules eagerly; `diff.py` imports
  `omegaprompt.runtime`; `report.py`/`check_artifact.py` import core/domain only.
- `report+check_artifact alone = 218.9ms` vs `full cli = 226.1ms` -> making
  __version__ cheap + lazy __init__ saves only ~7ms for validating commands (but
  helps --version/--help much more, and is the prerequisite for defer_build to
  help the help/version paths).
- `artifact_integrity.py` is NOT in core/__init__; pulled only via check_artifact
  command. Lazy-loading it saves ~0.2-2ms only. Too small to lead with.
- `omegacal` = pure `from omegaprompt import *` re-export; ~197ms = same as
  omegaprompt, ZERO independent weight. Dual-package is maintenance-only, NOT a
  perf lever. Shim modules (fitness/judge/dataset) likewise negligible.
