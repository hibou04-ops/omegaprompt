# Pre-existing Intellectual Property Declaration

> **Purpose**: This document is a tamper-evident timestamped declaration that the
> work in this repository constitutes pre-existing personal intellectual property
> of the Primary Author, authored prior to and independent of any current or
> future employment relationship.

## Repository Identification

- **Repository**: [hibou04-ops/omegaprompt](https://github.com/hibou04-ops/omegaprompt)
- **License**: Apache License 2.0
- **Primary Author**: **Kyunghoon Gwak (곽경훈)** — operating as [@hibou04-ops](https://github.com/hibou04-ops)
  - Primary contact email: `hibouaile04@gmail.com`
  - Git author email on commits prior to 2026-04-28: `hibou04@gmail.com`
  - Both email addresses are verified personal accounts of the Primary Author

## Authorship Timeline (Tamper-Evident)

The following git artifacts establish the authorship timeline. The git commit graph
and the public GitHub remote (`github.com/hibou04-ops/omegaprompt`) provide
independent timestamp witnesses.

| Anchor | Commit Hash | Date (KST) | Description |
|---|---|---|---|
| Apache 2.0 relicense | `106c003` | 2026-04-22 | MIT → Apache 2.0 for patent grant + trademark preservation |
| 1.2.0 release | `50b9cc8` | 2026-04-27 | Calibration suite, 60s demo, Apache 2.0 metadata fix |
| 1.3.0 release | `1a0312f` | 2026-04-29 | Eight runtime entrypoints + MCP server |
| Pre-employment snapshot | (tagged on commit) | 2026-04-29 | This declaration committed; tagged `pre-employment-ip-snapshot-2026-04-29` |

## Scope of Pre-existing IP

The following work product is declared as pre-existing personal intellectual property:

1. **Provider-neutral calibration engine**: The translation of omega-lock's
   sensitivity-driven coordinate descent calibration discipline to the prompt
   engineering setting:
   - Provider-neutral meta-axes (`reasoning_profile`, `output_budget_bucket`,
     `response_schema_mode`, `tool_policy_variant`, `system_prompt_variant`,
     `few_shot_count`)
   - Capability-tier reporting and `CapabilityEvent` recording for silent
     provider degradation
   - Two execution profiles (`guarded` / `expedition`) and the explicit
     trade between validation strength and exploratory reach
2. **Calibration kernel**: Fitness aggregation (hard-gate × soft-score),
   walk-forward gate (KC-4 Pearson + generalisation gap), sensitivity ranking
   and top-K axis unlock policy, run-risk assessment, schema v2.0
   `CalibrationArtifact` shape.
3. **Provider adapter layer**: Anthropic adapter, OpenAI / OpenAI-compatible
   adapter (Azure OpenAI, Groq, Together.ai, OpenRouter, Ollama drop-ins),
   local placeholder adapter, factory + `supported_providers` helper.
4. **Judge protocol**: `Judge` Protocol with `LLMJudge`, `RuleJudge`, and
   `EnsembleJudge` implementations including the rule-helper library
   (`default_no_refusal`, `default_non_empty`, `json_object_check`,
   `regex_check`).
5. **Eight runtime entrypoints** (`omegaprompt/runtime.py`): `calibrate`,
   `evaluate`, `report`, `diff`, `measure_sensitivity`, `grade`, `preflight`,
   `classify_traps`, plus the supporting Pydantic types (`ProviderSpec`,
   `CalibrateTuning`, `SensitivityTuning`, `SensitivityResult`, `ArtifactDiff`).
6. **MCP server** (`omegaprompt.mcp`): FastMCP-based server wrapping the eight
   runtime entrypoints as agent-callable tools, including the stdio /
   streamable-http transport entry point and the `omegaprompt-mcp` console
   script.
7. **CLI surface** (`omegaprompt calibrate`, `omegaprompt report`,
   `omegaprompt diff`).
8. **Reporting layer**: Markdown rendering of `CalibrationArtifact`, artifact
   loading / saving primitives.
9. **Test suite**: All materials under `tests/`.
10. **Examples and demonstrations**: All materials under `examples/`,
    including the nine-task validated calibration suite (code review, code
    writing, debugging, refactoring, explanation, summarization, translation,
    test writing, commit message), demo replay scripts, and the 60-second
    walkthrough.
11. **Documentation**: README, README_KR, EASY_README, EASY_README_KR,
    CHANGELOG, NOTICE.
12. **Specific terminology and application**: The compound term "omegaprompt"
    as used to label this specific provider-neutral prompt calibration engine,
    along with the meta-axis names and the runtime / MCP entrypoint names as
    defined within `src/omegaprompt/`. *No claim is made to the generic words
    "omega" or "prompt" in isolation; the claim is to the specific compound
    and its application within this corpus.*

## Companion Repositories

This repository is the parent of a multi-repo IP defense package authored by the
same Primary Author:

- [omega-lock](https://github.com/hibou04-ops/omega-lock) — methodology framework
- [Antemortem](https://github.com/hibou04-ops/Antemortem) — pre-implementation reconnaissance methodology
- [antemortem-cli](https://github.com/hibou04-ops/antemortem-cli) — CLI for Antemortem
- [mini-omega-lock](https://github.com/hibou04-ops/mini-omega-lock) — empirical preflight sub-tool
- [mini-antemortem-cli](https://github.com/hibou04-ops/mini-antemortem-cli) — analytical preflight sub-tool

See each repository's `PRE_EXISTING_IP.md` for its own authorship binding.

## Development Conditions

This work was developed:

- Using **personal time** (outside of any third-party working hours)
- Using **personal equipment** (no employer-issued hardware)
- Using **personal accounts** (no employer-issued cloud, LLM, or API credentials)
- **Without reference** to any third party's confidential or proprietary information

## Use in Future Employment Agreements

This declaration is intended to be attached as a Schedule / Exhibit (commonly
"Schedule A: Pre-existing IP") to any future employment, contractor, or
consulting agreement, to clarify that:

- The work in this repository remains the personal property of the Primary Author.
- Future development on this codebase, conducted on personal time and outside the
  scope of any employment, continues to be the Primary Author's personal IP.
- Any contributions from a future employer's domain, made on employer time using
  employer resources, would be governed by the relevant employment agreement —
  the boundary is preserved by maintaining a separate repository, fork, or
  branch for any such employer-domain contributions.

## Verification

To independently verify this declaration:

1. Inspect git log:
   ```
   git log --format="%H | %ai | %an <%ae>" | grep "Hibou04-ops"
   ```
2. Confirm tag (when committed):
   ```
   git tag -l "pre-employment-ip-snapshot-*"
   git show pre-employment-ip-snapshot-2026-04-29
   ```
3. Cross-reference with public GitHub timestamps:
   - https://github.com/hibou04-ops/omegaprompt/commit/106c003
   - https://github.com/hibou04-ops/omegaprompt/releases

---

**Declaration date**: 2026-04-29
**License**: Apache License 2.0
**Document version**: 1.0
