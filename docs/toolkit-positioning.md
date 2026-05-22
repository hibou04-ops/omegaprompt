# Toolkit Positioning

This repository is `hibou04-ops/omegaprompt`. The PyPI distribution, primary
import package, and primary CLI are all `omegaprompt`. `omegacal` is a
secondary compatibility import package and CLI alias.

## omegaprompt

`omegaprompt` is the prompt-calibration adapter and audit surface. It provides:

- prompt-specific data contracts (`Dataset`, `JudgeRubric`, `PromptVariants`)
- provider adapters and capability records
- rule, LLM, and ensemble judge paths
- guarded/expedition profile policy
- `CalibrationArtifact` schema v2.0
- CLI, Python runtime, and optional MCP tool wrappers
- deterministic reference artifacts and Prompt CI diff semantics

It does not publish packages, create releases, or run live provider calls in
default CI. It also does not include a dashboard or web app.

## omegaprompt vs omega-lock

`omega-lock` is the parent calibration framework. It owns the generic
calibration machinery: stress measurement, top-K unlock search, walk-forward
evaluation, and kill-criteria discipline.

`omegaprompt` adapts prompt work to that framework. It turns prompt variants,
judge rubrics, provider calls, and prompt-specific risk boundaries into a
`CalibrableTarget` that `omega-lock` can evaluate. The separation matters:
`omegaprompt` should not reimplement the calibration engine when the
`omega-lock` search layer already owns that job.

## omegaprompt vs antemortem-cli

`antemortem-cli` is a pre-implementation reconnaissance tool. It is used before
building or changing a system to identify likely traps, assumptions, and review
questions.

`omegaprompt` runs after prompt candidates exist. It checks whether a prompt
configuration generalizes under declared gates and whether the produced
artifact should pass Prompt CI.

The tools can be used together, but they answer different questions:

- `antemortem-cli`: what could go wrong before implementation?
- `omegaprompt`: did this prompt artifact satisfy its declared validation
  contract?

## Optional Preflight Plugins

`omegaprompt.preflight` defines the contract and adaptation logic. The heavier
preflight implementations live outside the core package:

- `mini-omega-lock`: empirical preflight plugin for live environment probes
  such as judge consistency, endpoint reliability, context margin, latency, and
  noise floor.
- `mini-antemortem-cli`: analytical preflight plugin for deterministic
  calibration-trap classification.

These packages are optional. A standalone `omegaprompt` install does not need
them, and their live or environment-specific evidence should be recorded as
preflight data rather than implied as a public benchmark claim.

## omegacal Compatibility Alias

`omegacal` exists for historical compatibility. It is not the PyPI distribution
name and should not be presented as the primary product identity. New docs,
commands, and install instructions should use `omegaprompt` unless describing
backward compatibility.

The public CLI executables are distinct:

- `omegaprompt`: primary CLI
- `omegacal`: compatibility alias
- `omegaprompt-mcp`: optional MCP server entrypoint, available when the MCP
  extra is installed

## MCP Boundary

MCP is installed as `omegaprompt[mcp]`. The core package import and core CLI
must remain clean without the MCP SDK. The MCP server is a wrapper over the
runtime entrypoints; it does not introduce new gates, hidden provider calls, or
dashboard behavior.

## No Dashboard or Web-App Scope

`omegaprompt` is a library, CLI, runtime surface, and optional MCP server. It is
not a hosted evaluation dashboard, SaaS interface, or web application. Artifact
review is intentionally file-based: JSON artifacts, markdown reports, CI logs,
and PR diffs.

## Claim Discipline

The toolkit docs should avoid provider superiority, model superiority, prompt
superiority, download/adoption metrics, and exact benchmark aggregates unless a
source of truth is named. Exact deterministic reference metrics belong in the
reference artifact, generated claim document, or a reproducible command output.
