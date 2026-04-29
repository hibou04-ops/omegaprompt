# Authors

This project was conceived, designed, and authored by:

**Primary Author**: Kyunghoon Gwak (곽경훈) *(see [PRE_EXISTING_IP.md](PRE_EXISTING_IP.md) for full IP binding)*
**GitHub Handle**: [@hibou04-ops](https://github.com/hibou04-ops)
**Primary Contact Email**: hibouaile04@gmail.com
**Git Author Email** (historical, on commits prior to 2026-04-28): hibou04@gmail.com

Both email addresses are verified personal accounts of the Primary Author.
The GitHub handle [@hibou04-ops](https://github.com/hibou04-ops) is operated by the Primary Author.

## Authorship Scope

The following work in this repository is the personal intellectual property of
the Primary Author, developed independently using personal time, equipment, and
resources:

- The `omegaprompt` provider-neutral prompt calibration engine
- All source code under `src/omegaprompt/`, including:
  - Domain types (meta-axes, dataset, rubric, params, profiles, result)
  - Calibration kernel (fitness, artifact I/O, walk-forward, sensitivity ranking, profile policy, run risk)
  - Provider adapters (Anthropic, OpenAI / OpenAI-compatible, local / Ollama / vllm / llama.cpp)
  - Judge protocol + LLMJudge / RuleJudge / EnsembleJudge implementations
  - Target adapters (PromptTarget) and the CalibrableTarget composition point
  - Reporting and CLI command surfaces
  - Eight high-level runtime entrypoints (`runtime.py`)
  - The MCP server (`omegaprompt.mcp`)
- Test suite under `tests/`
- Examples and demonstrations under `examples/` (the 9-task validated calibration suite, demo replay scripts)
- Documentation (README, README_KR, EASY_READMEs, CHANGELOG)

This project is the parent calibration framework for the multi-repo toolkit:

- [omega-lock](https://github.com/hibou04-ops/omega-lock) — sensitivity-driven coordinate descent calibration framework (the methodology that omegaprompt ports to the prompt setting)
- [Antemortem](https://github.com/hibou04-ops/Antemortem) — pre-implementation reconnaissance methodology applied during omegaprompt development
- [antemortem-cli](https://github.com/hibou04-ops/antemortem-cli) — CLI tool for the Antemortem methodology
- [mini-omega-lock](https://github.com/hibou04-ops/mini-omega-lock) — empirical preflight probes (sub-tool of omegaprompt)
- [mini-antemortem-cli](https://github.com/hibou04-ops/mini-antemortem-cli) — analytical preflight classifier (sub-tool of omegaprompt)

All authored by the same Primary Author. See each repository's `PRE_EXISTING_IP.md`
for its own authorship binding.

## Contributors

External contributions are accepted under the terms of the Apache License 2.0.
Contributors retain copyright in their own contributions and grant the project
a license under Apache 2.0 §5.

---

See [LICENSE](LICENSE), [NOTICE](NOTICE), and [PRE_EXISTING_IP.md](PRE_EXISTING_IP.md).
