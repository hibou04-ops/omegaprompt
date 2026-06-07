# omegaprompt (한국어)

> **요약이 먼저 필요하면**: [EASY_README_KR.md](EASY_README_KR.md). 본 문서는 architecture·schema·검증까지 다루는 reference입니다.
> English: [README.md](README.md) · [EASY_README.md](EASY_README.md)

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![PyPI](https://img.shields.io/badge/pypi-2.0.2-blue.svg)](https://pypi.org/project/omegaprompt/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Artifact schema](https://img.shields.io/badge/artifact-schema%20v2.0-blueviolet.svg)](#10-calibrationartifact-schema-v20)
[![MCP](https://img.shields.io/badge/MCP-server-blueviolet.svg)](#18-mcp-서버)
[![Parent framework](https://img.shields.io/badge/framework-omega--lock-blueviolet.svg)](https://github.com/hibou04-ops/omega-lock)

문서: [English](README.md) · [한국어](README_KR.md) · [Easy English](EASY_README.md) · [쉬운 한국어](EASY_README_KR.md)

> **omegaprompt toolkit의 한 부분** — [omegaprompt](https://github.com/hibou04-ops/omegaprompt) (calibration engine, 본 repo) · [omega-lock](https://github.com/hibou04-ops/omega-lock) (audit framework) · [antemortem-cli](https://github.com/hibou04-ops/antemortem-cli) (pre-implementation recon CLI) · [mini-omega-lock](https://github.com/hibou04-ops/mini-omega-lock) (empirical preflight) · [mini-antemortem-cli](https://github.com/hibou04-ops/mini-antemortem-cli) (analytical preflight) · [Antemortem](https://github.com/hibou04-ops/Antemortem) (방법론). agent 시나리오 9개를 정리한 cross-toolkit cookbook: [AGENT_TRIGGERS.md](AGENT_TRIGGERS.md).

---

## 한 줄 정의

**omegaprompt는 prompt engineering을 위한 CI입니다.** train/test split, 사전 선언된 gate, holdout 상관 검증, regression 시 빌드를 깨뜨리는 audit artifact — *calibration set에 overfit한 prompt는 ship되지 않습니다.*

```bash
pip install omegaprompt              # core
pip install "omegaprompt[mcp]"       # + MCP server
```

> **v2.0.2 (2026-06-08)** — **새 내용:** **opt-in 병렬 item 평가** — `--concurrency N`(CLI) / `CalibrateTuning(max_workers=N)`이 각 candidate 안에서 dataset item을 local thread pool로 동시에 평가합니다. 각 item은 여전히 target 호출 후 judge 호출을 순차로 실행하므로, 어느 한 provider에 대한 동시 호출 수는 N을 넘지 않습니다. 기본값(`1`)은 serial이며 이전 버전과 byte-identical한 artifact를 만듭니다. wall-clock 속도 향상은 **전적으로 사용 중인 provider 계정의 동시성 한도(RPM/TPM)에 따라 달라집니다**: 계정이 허용하면 N=2에서 대략 50%, 계정이 사실상 호출을 직렬화하면 **0%**입니다. 또한 `PromptTarget.evaluate()`가 이제 **resolved params 기준으로 memoize**되어, 마지막 best-candidate 평가가 provider를 다시 호출하는 대신 grid-search 결과를 재사용합니다 — `calibrate()` 동안 live API 호출이 줄며 public surface나 artifact schema 변경은 없습니다. omega-lock pin은 그대로(`>=0.3.0,<0.4.0`)이고 `CalibrationArtifact.schema_version`은 `2.0`을 유지하며 golden reference artifact는 변경되지 않습니다. 프로젝트의 PyPI Development Status는 이제 **4 - Beta**입니다. public README/PyPI claim과 deterministic reference metric은 생성된 [claim ledger](docs/claims/README_CLAIMS.generated.md)에서 추적합니다. release gate는 repository consistency, generated claims, golden reference artifacts, artifact integrity, wheel smoke, local/post-release verification을 확인합니다.

이름 경계: GitHub repo는 `hibou04-ops/omegaprompt`; PyPI distribution, primary import package, primary CLI는 `omegaprompt`; compatibility package / CLI alias는 `omegacal`; 별도 parent calibration framework는 `omega-lock`입니다.

신뢰 문서: [trust model](docs/trust-model.md) · [toolkit positioning](docs/toolkit-positioning.md) · [provider capabilities](docs/provider-capabilities.md) · [profiles and risk boundaries](docs/profiles-and-risk-boundaries.md).

<!-- public-claim-ledger:start -->
> Claim evidence source: [docs/claims/public_claim_ledger.json](docs/claims/public_claim_ledger.json), rendered by `python tools/generate_readme_claims.py`.
<!-- public-claim-ledger:end -->

---

## 목차

- [1. 해결 대상 — 왜 prompt에 CI가 필요한가](#1-해결-대상--왜-prompt에-ci가-필요한가)
- [2. KC4 — holdout 상관 gate](#2-kc4--holdout-상관-gate)
- [3. Quick start](#3-quick-start)
- [4. 인접 도구와의 차이](#4-인접-도구와의-차이)
- [5. Calibratable한 6 meta-axes](#5-calibratable한-6-meta-axes)
- [6. Provider 지원 — 두 boundary](#6-provider-지원--두-boundary)
- [7. Cross-vendor 검증](#7-cross-vendor-검증)
- [8. Judge 3종 — Rule / LLM / Ensemble](#8-judge-3종--rule--llm--ensemble)
- [9. 2개 profile — guarded vs expedition](#9-2개-profile--guarded-vs-expedition)
- [10. CalibrationArtifact schema v2.0](#10-calibrationartifact-schema-v20)
- [11. 데이터 계약](#11-데이터-계약)
- [12. 아키텍처](#12-아키텍처)
- [13. 비-trivial한 설계 결정](#13-비-trivial한-설계-결정)
- [14. 이 도구가 *아닌* 것](#14-이-도구가-아닌-것)
- [15. 비용 및 성능](#15-비용-및-성능)
- [16. 검증](#16-검증)
- [17. 3-layer stack](#17-3-layer-stack)
- [18. MCP 서버](#18-mcp-서버)
- [19. 회의론자를 위한 FAQ](#19-회의론자를-위한-faq)
- [20. 선행 연구 및 credit](#20-선행-연구-및-credit)
- [21. 상태 & 로드맵](#21-상태--로드맵)
- [신뢰 및 toolkit 문서](#신뢰-및-toolkit-문서)
- [22. 라이선스](#22-라이선스)

---

## 1. 해결 대상 — 왜 prompt에 CI가 필요한가

직접 고른 소수 eval set에 prompt를 튜닝, 높은 점수, ship. 이후 production 지표 악화. 이건 prompt 작성 실력 문제가 아니라 **검증되지 않은 generalization**의 문제입니다. ML 커리큘럼은 1990년대부터 똑같은 failure mode에 train/test split을 답으로 갖고 있었습니다 — 학습된 configuration은 학습한 데이터에서 검증되지 않습니다. 단지 prompt 워크플로 다수가 그 단계 없이 ship할 뿐.

omegaprompt가 강제하는 5개 layer:

- **train/test split + 사전 선언된 상관 gate.** score를 본 뒤 threshold를 낮출 수 없습니다.
- **walk-forward 검증.** `test_fitness`가 `train_fitness`를 따라가지 못하면 run이 flagged.
- **cross-vendor judge.** Anthropic target ↔ OpenAI judge로 self-agreement bias를 구조적으로 제거.
- **provider-neutral meta-axes.** 벤더 고유 knob이 아닌 semantic 축에서 탐색 — 마이그레이션에서 깨지지 않음.
- **`CalibrationArtifact` schema v2.0.** PR diff에 그대로 들어가는 JSON. regression이면 CI fail.

대부분의 LLM eval 도구가 답하는 질문은 *"어느 prompt가 best score인가?"* (eval set을 ground truth로 가정). omegaprompt가 답하는 질문은 그 *다음*에 옵니다 — *"그 score가 일반화되는가?"*

---

## 2. KC4 — holdout 상관 gate

KC4는 train에서는 좋아 보였는데 test에서는 declared target과 더 이상 상관 없는 경우 — 즉 calibration set에 overfit한 경우 — fail시키는 gate입니다.

```
train_ranks  ─┐
              ├─▶ Pearson(train_ranks, test_ranks) ≥ --min-kc4 ?
test_ranks   ─┘                                   ─▶ PASS
                                                  ─▶ FAIL_KC4_GATE  (ship-blocker)

|train_fitness − test_fitness| ≤ --max-gap ?
   ─▶ PASS / FAIL
```

핵심은 **사전 선언**입니다. `--min-kc4`, `--max-gap`은 score를 보기 *전*에 config에 적습니다. 결과를 본 뒤 완화하면 KC4는 더 이상 gate가 아니라 사후 정당화. quant finance의 Winchester defense에서 빌려온 규율입니다 — *kill criterion은 declared 이후 조정 불가.*

`status = FAIL_KC4_GATE`는 설계상 ship-blocker. score가 아무리 높아도 status가 fail이면 artifact는 SHIP recommendation을 내지 않습니다.

---

## 3. Quick start

### 0. 스모크 테스트 — API 키 없이, 네트워크 없이

```bash
git clone https://github.com/hibou04-ops/omegaprompt.git
cd omegaprompt && pip install -e .

# Reference artifact 재현 (in-memory target+judge, 결정론)
python examples/reference/reproduce_reference_artifact.py

# 포함된 reference artifact를 audit report로 렌더
omegaprompt report examples/reference/reference_artifact.json
```

매 실행이 같은 `CalibrationArtifact`를 만듭니다. API 키를 붙이기 전에 CI를 wiring할 때 권장.

### 1. 실제 calibration — cloud target + cross-vendor judge

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...   # 또는 GOOGLE_API_KEY — free tier: https://aistudio.google.com/apikey

omegaprompt calibrate examples/sample_dataset.jsonl \
  --rubric examples/rubric_example.json \
  --variants examples/variants_example.json \
  --target-provider anthropic \
  --judge-provider openai \
  --output artifact.json
```

### 2. Audit report 렌더 (PR description, CI 로그, 사람 리뷰용)

```bash
omegaprompt report artifact.json > report.md
```

### 3. CI regression gate

`omegaprompt diff`는 `calibrated_fitness`, `walk_forward.test_fitness`, `hard_gate_pass_rate`, cost/latency frontier, guarded-mode 위반 중 어느 하나라도 regression이면 non-zero로 종료합니다.

```bash
omegaprompt diff previous.json artifact.json
```

```yaml
# .github/workflows/prompt-audit.yml
- run: omegaprompt diff previous.json artifact.json
```

---

## 4. 인접 도구와의 차이

| Capability | omegaprompt | 일반 eval runner | optimizer | 벤더 dashboard |
|---|:-:|:-:|:-:|:-:|
| eval 실행 | ✓ | ✓ | ✓ | ✓ |
| **walk-forward ship gate** | ✓ | 보통 수동 | 보통 수동 | ✗ |
| **사전 선언 kill criteria** | ✓ | 보통 수동 | 부분 | ✗ |
| **cross-vendor judge 규율** | ✓ | 설정 가능 | 수동 | ✗ |
| provider-neutral 축 | ✓ | 다양 | 부분 | ✗ |
| JSON-diff audit artifact | ✓ | 로그 | 다양 | ✗ |
| MCP 도구 surface (Claude Code / Cursor) | ✓ | 다양 | ✗ | ✗ |

> **포지셔닝**: omegaprompt는 **audit-first**입니다 — search-first가 아닙니다. 후보가 이미 있다고 가정하고 (eval runner나 optimizer 출력에서 흔히 옴) *"실제로 일반화됐는가?"* 라는 search 다음 질문에 답합니다. 기존 도구의 출력은 *입력으로* 꽂힙니다 — assertion 스타일 체크는 `RuleJudge`로, optimizer 출력은 `PromptVariants`로.

---

## 5. Calibratable한 6 meta-axes

벤더별 parameter 이름이 아니라 semantic 카테고리로 탐색합니다. provider adapter가 내부에서 번역. 결과로 동일 artifact가 벤더 간 재생됩니다.

| Axis | 값 | 통제 대상 |
|---|---|---|
| `system_prompt_variant` | prompt 리스트 인덱스 | 어느 system prompt |
| `few_shot_count` | 0, 1, 2, ... | 포함할 예제 개수 |
| `reasoning_profile` | OFF / LIGHT / STANDARD / DEEP | extended thinking 깊이 |
| `output_budget_bucket` | SMALL(1024) / MEDIUM(4096) / LARGE(16000) | max_tokens 버킷 |
| `response_schema_mode` | FREEFORM / JSON_OBJECT / STRICT_SCHEMA | 출력 구조 강제 |
| `tool_policy_variant` | NO_TOOLS / TOOL_OPTIONAL / TOOL_REQUIRED | tool-use 정책 (declared; wiring 일부) |

`PromptSpace`로 축을 잠글 수 있습니다 (`reasoning_profile_min == reasoning_profile_max`) — 일부 dimension이 pre-decided일 때. 전형적인 calibration은 3개 축을 열어두고, sensitivity 측정 결과에 따라 top-K를 unlock합니다. 모든 축을 단일 값으로 잠그면 사실상 *judge를 통한 고정 prompt 벤치마크*가 됩니다 — prompt 개정에 대한 regression test 모드로 정당한 사용.

v0.1 시점의 축 셋이 의도적으로 보수적인 이유: `temperature` / `top_p`처럼 frontier 모델이 점진적으로 deprecate하는 knob을 축으로 두면 릴리스마다 의미가 바뀝니다. 위 6개는 어느 chat-style frontier 모델에서도 transfer합니다.

---

## 6. Provider 지원 — 두 boundary

omegaprompt가 LLM을 호출하는 경계는 정확히 둘입니다 — **target** (calibration 대상 prompt, free-form 출력) 과 **judge** (rubric 기반 채점, schema-enforced 출력). 둘 다 `LLMProvider` Protocol을 통과. 각 adapter는 벤더의 가장 강한 native 경로를 사용합니다 — 파이프라인 어디에도 클라이언트 측 JSON regex 파싱은 없습니다.

| Provider | target flag | judge flag | 기본 모델 | env var | native 경로 |
|---|---|---|---|---|---|
| Anthropic | `--target-provider anthropic` | `--judge-provider anthropic` | `claude-opus-4-7` | `ANTHROPIC_API_KEY` | 명시적 `cache_control` 동반 `messages.create` / `messages.parse` |
| OpenAI | `--target-provider openai` | `--judge-provider openai` | `gpt-4o` | `OPENAI_API_KEY` | `chat.completions.create` / `beta.chat.completions.parse` |
| OpenAI-호환 | 위 + `--target-base-url` | 위 + `--judge-base-url` | `--*-model`로 지정 | `OPENAI_API_KEY` (로컬 미인증 endpoint는 임의 문자열) | OpenAI와 동일 |

`--base-url`로 커버되는 호환 endpoint: **Azure OpenAI**, **Groq**, **Together.ai**, **OpenRouter**, **로컬 Ollama** (`http://localhost:11434/v1`), **vLLM**. 새 provider를 추가하려면 `LLMProvider` Protocol 두 메서드(`complete`, `structured_complete`)만 만족하면 됩니다 — 모듈 하나 + factory 한 줄.

```python
class LLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self, *, system_prompt, user_message, few_shots,
        max_tokens, enable_thinking=True, effort="high",
    ) -> tuple[str, dict[str, int]]: ...

    def structured_complete(
        self, *, system_prompt, user_content, output_schema,
        max_tokens=2048,
    ) -> tuple[T, dict[str, int]]: ...
```

SDK 누출 없음. 동일 prompt가 모든 벤더에서 동일하게 동작.

---

## 7. Cross-vendor 검증

이 도구가 활용 가능하게 만드는 가장 강력한 단일 구성: **target은 한 provider, judge는 다른 provider.**

이유는 단순합니다. target과 judge가 동일 모델 (또는 동일 벤더) 이면 self-agreement bias가 점수를 오염시킵니다. OpenAI judge가 채점한 약한 OpenAI 출력은 — judge가 target의 bias를 공유하므로 — 여전히 acceptable해 보일 수 있습니다. 같은 출력을 Anthropic judge로 (또는 반대로) 채점하면 그 bias가 구조적으로 깨집니다. judge가 더 이상 peer가 아니라 *disinterested second opinion*.

```bash
# Target: production prompt를 gpt-4o로 실행.
# Judge: 다른 벤더의 higher-tier disinterested 모델.
omegaprompt calibrate train.jsonl \
  --rubric rubric.json \
  --variants variants.json \
  --test test.jsonl \
  --target-provider openai \
  --target-model gpt-4o \
  --judge-provider anthropic \
  --judge-model claude-opus-4-7 \
  --output artifact.json
```

`CalibrationArtifact`는 `target_provider`, `target_model`, `judge_provider`, `judge_model`을 기록합니다 — 재현성이 모든 run의 machine-readable 속성.

또 하나 유용한 조합: 로컬 Ollama target + cloud judge로 채점.

```bash
omegaprompt calibrate train.jsonl \
  --rubric rubric.json --variants variants.json --test test.jsonl \
  --target-provider openai \
  --target-base-url http://localhost:11434/v1 \
  --target-model llama3.1:70b \
  --judge-provider anthropic \
  --judge-model claude-opus-4-7
```

---

## 8. Judge 3종 — Rule / LLM / Ensemble

- **`RuleJudge`** — deterministic regex/predicate gate. `no_refusal`, JSON 유효성, 정규식 일치 등. **API 비용 0.** 이게 답할 수 있는 질문은 이것으로.
- **`LLMJudge`** — capable LLM이 STRICT_SCHEMA로 dimension별 정수 점수 + binary gate를 산출. self-congratulation에 저항하는 calibration-hygiene 지시가 system prompt에 명시 ("midpoint anchor, full-range 사용, 자신과 agree"). Pydantic + 벤더 native structured-output로 schema enforce.
- **`EnsembleJudge`** — `RuleJudge`를 먼저 돌리고, gate 하나라도 실패하면 **short-circuit** (LLM 호출 안 함). 통과한 응답만 LLM judge로 escalate. 깨진 응답에 들어가는 비용을 잘라냅니다 — 전형적인 calibration에서 LLM judge call의 30–60%가 절감.

---

## 9. 2개 profile — guarded vs expedition

| | `guarded` (기본) | `expedition` |
|---|---|---|
| silent schema fallback | ❌ raise | ✅ 허용, `CapabilityEvent` 기록 |
| placeholder provider | ❌ raise | ✅ 허용 |
| non-ship-grade judge | ❌ raise | ✅ 허용 |
| 기본 `--max-gap` | 0.25 | 0.35 |
| 기본 `--min-kc4` | 0.5 | 0.3 |

`expedition`에서 풀린 모든 가드는 artifact의 `relaxed_safeguards` 배열에 `RelaxedSafeguard` 항목으로 기록됩니다. *strict냐 reach냐*의 trade가 명시적이고 audit 가능하게 — 슬그머니 풀려서 잊히는 일이 구조적으로 막힙니다.

production용 audit은 `guarded`로. 탐색·실험은 `expedition`으로. 두 모드 사이 차이가 artifact diff에 그대로 드러납니다.

---

## 10. CalibrationArtifact schema v2.0

모든 run이 산출하는 단일 JSON. CI에서 diff 가능, PR review에 그대로, 시간에 따른 calibration 상태의 audit log.

```json
{
  "schema_version": "2.0",
  "status": "OK",
  "ship_recommendation": "SHIP",
  "neutral_baseline": {
    "fitness": "<float>",
    "hard_gate_pass_rate": "<float>"
  },
  "calibrated": {
    "best_params": {
      "system_prompt_variant": 2,
      "few_shot_count": 1,
      "reasoning_profile": "STANDARD",
      "output_budget_bucket": "MEDIUM",
      "response_schema_mode": "STRICT_SCHEMA",
      "tool_policy_variant": "NO_TOOLS"
    },
    "fitness": "<float>",
    "uplift": "<float>"
  },
  "walk_forward": {
    "train_fitness": "<float>",
    "test_fitness": "<float>",
    "generalization_gap": "<float>",
    "validation_mode": "auto | paired | disjoint",
    "shared_item_count": "<integer>",
    "kc4_status": "COMPUTED | MISSING_PER_ITEM_SCORES | ...",
    "kc4_pearson": "<float-or-null>",
    "min_kc4_threshold": "<float-or-null>",
    "max_gap_threshold": "<float>",
    "passed": true
  },
  "providers": {
    "target": {"name": "anthropic", "model": "claude-opus-4-7"},
    "judge":  {"name": "openai", "model": "gpt-4o"}
  },
  "capability_events": [],
  "relaxed_safeguards": [],
  "usage_summary": {
    "input_tokens": "<integer>",
    "cache_read_input_tokens": "<integer>",
    "output_tokens": "<integer>",
    "total_api_calls": "<integer>"
  },
  "profile": "guarded"
}
```

`CalibrationArtifact`는 Pydantic v2 모델입니다. 핵심 nested contract는 엄격하게 검증되고, top-level unknown extra는 `omegaprompt check-artifact`가 review-visible warning으로 분류합니다. malformed run이 silent하게 통과하지 못하도록 schema validation과 integrity check를 함께 사용합니다. 모든 schema bump는 CHANGELOG의 *Schema revisions*에 추적됩니다.

---

## 11. 데이터 계약

```python
# Input — 사용자 작성 파일 셋
Dataset(items=[DatasetItem(id="t1", input="...", reference="..."), ...])
JudgeRubric(
    dimensions=[
        Dimension(name="correctness", description="...", weight=0.5, scale=(1, 5)),
        Dimension(name="clarity",     description="...", weight=0.3),
        Dimension(name="conciseness", description="...", weight=0.2),
    ],
    hard_gates=[
        HardGate(name="no_refusal",          description="...", evaluator="judge"),
        HardGate(name="no_safety_violation", description="...", evaluator="judge"),
    ],
)
PromptVariants(
    system_prompts=["You are a...", "You are a senior...", "You are a teacher..."],
    few_shot_examples=[{"input": "...", "output": "..."}, ...],
)

# ↓ omega-lock.run_p1이 PromptTarget.evaluate()를 (params × slice) 조합으로 drive.
# 매 evaluate()는 주어진 params dict에 대해 N target call + N judge call.

# ↓ 벤더 native structured-output가 schema-enforce
JudgeResult(
    scores={"correctness": 5, "clarity": 4, "conciseness": 3},
    gate_results={"no_refusal": True, "no_safety_violation": True},
    notes="응답이 task를 정확히 풀었음. wrap-up 문단에 약간의 padding.",
)

# ↓ CompositeFitness가 (hard_gate × soft_score)로 집계
# fitness = sum(soft_score_i * all_gates_passed_i) / n_items

# ↓ omega-lock run_p1이 walk-forward 결과 산출
CalibrationArtifact(...)   # 위 §10 참조
```

malformed judge 응답은 SDK 경계에서 `ValidationError` raise — fitness 오염 없음. `JudgeResult.gate_results`에서 missing한 hard-gate 결과는 *"답하지 않음"* 으로 간주, 통과와 별개. 모든 model의 모든 필드는 계약입니다 — 위반은 loud.

---

## 12. 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│  Dataset (.jsonl)   PromptVariants (.json)   JudgeRubric (.json)│
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  PromptTarget                                                 │
│    omega-lock CalibrableTarget 구현                           │
│    param_space() ───▶ 6 meta-axes                            │
│    evaluate(params):                                          │
│       for each dataset item:                                  │
│         1. params에서 prompt build (variant, few-shot, ...)   │
│         2. call_target(target_provider, ...) ─▶ response text │
│         3. judge.score(rubric, ...) ─▶ JudgeResult            │
│       CompositeFitness(judge_results) ─▶ fitness              │
│    returns EvalResult(fitness, n_trials, metadata)            │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  omega-lock run_p1                                            │
│    measure_stress + select_unlock_top_k                       │
│    unlocked subspace에서 GridSearch                           │
│    --test slice에서 WalkForward (KC4)                         │
│    grid_best, test_fitness, status 산출                       │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  CalibrationArtifact (JSON, schema v2.0)                      │
│    neutral_baseline, calibrated, walk_forward,                │
│    providers, capability_events, relaxed_safeguards,          │
│    usage_summary, status, ship_recommendation                 │
└──────────────────────────────────────────────────────────────┘
```

omega-lock 라인 위는 새로 구현. 그 라인 또는 아래는 변경 없이 재사용. 합성 경계는 `CalibrableTarget` 프로토콜 — 함수 하나(`param_space()`) + 메서드 하나(`evaluate(params)`).

`PromptTarget`이 ~200 라인인 이유가 여기 있습니다. 비싼 부분 — sensitivity 분석, search, walk-forward, kill criteria — 은 omega-lock 자체 test suite와 release record가 source of truth입니다. omegaprompt가 기여하는 것: prompt-specific adapter, LLM-as-judge machinery, fitness shape. 나머지는 모두 composition.

---

## 13. 비-trivial한 설계 결정

**두 boundary 모두 provider-agnostic.** vendor SDK를 만지는 곳은 target call과 judge call 둘뿐. 둘 다 `LLMProvider`를 통과. 파이프라인(stress, grid, walk-forward, KC4)은 구성상 vendor-neutral. target과 judge는 *독립적으로* provider를 받습니다 — CLI 한 줄로 cross-vendor 검증으로 flip.

**재구현 아닌 단일 책임 adapter.** omega-lock이 stress 측정, top-K unlock, grid search, walk-forward, kill criteria, benchmark scorecard, iterative lock-in, 연속-공간용 `run_p2_tpe`까지 처리합니다. 여기서 재구현하는 것은 잘못. `PromptTarget`은 얇은 adapter — 가치는 omega-lock 파이프라인 전체가 변경 없이 작동한다는 점에 있습니다.

**Per-vendor native + schema-enforced judge.** Anthropic은 `messages.parse(output_format=JudgeResult)`, OpenAI는 `beta.chat.completions.parse(response_format=JudgeResult)`. malformed judge 응답 — missing 필드, 선언된 scale 밖 점수, non-boolean gate 결과 — 은 SDK 경계에서 `ValidationError` raise. 이게 없으면 잘못된 judge call 한 번이 calibration 전체를 오염시킬 수 있고 최종 report까지도 알아채지 못합니다. judge prose를 regex parse하는 것은 antemortem-cli에서 모델의 self-report line 번호를 신뢰하는 것과 같은 종류의 실수.

**Hard gate가 fitness를 0으로 collapse, gradient 없음.** refusal에 soft penalty (예: "20% 감점")를 주면 *almost* refuse하는 prompt에 reward — soft score 대부분을 capture하면서 safety boundary를 테스트하는 — 효과를 줍니다. hard-zero는 boundary 근처를 reward signal에서 invisible하게 만듭니다. 실제 deployment가 prompt를 평가하는 방식과 일치합니다 — 10번에 1번 refuse하는 prompt는 *"90%만큼 좋은"* 게 아니라 unshippable.

**Adapter-native judge prompt caching.** judge의 ~5k 토큰 system prompt는 각 벤더의 cacheable-prefix 최소값을 넘도록 sized. Anthropic은 명시적 `cache_control={"type": "ephemeral"}`; OpenAI의 자동 prompt caching은 system prompt가 provider threshold를 넘으면 발동. 전형적 run의 수백 judge call에서 cache hit이 비용을 dominate합니다. CLI는 provider 간 정규화된 `cache_read_input_tokens`를 surface — silent invalidator가 어디서 실행하든 loud fail. Windows path 누출 (`src\foo.py` vs `src/foo.py`) 은 prompt 진입 전에 forward slash로 정규화 — 100 run 당 ~$15짜리 캐시 무효화 버그.

**`reasoning_profile`과 `output_budget_bucket`은 first-class axis, global이 아님.** 어떤 prompt는 LIGHT만 필요. 전역 DEEP을 강제하면 fitness 개선 없이 토큰만 소진. calibration이 이를 surface하게 하는 것이 핵심 — `reasoning_profile`이 high stress면 task에 중요, low stress면 OFF로 lock + 토큰 절약. *"측정하기 전에 최적화하지 말라"* 의 parameter-calibration 버전.

**Target과 judge provider는 독립적.** 같은 provider를 양쪽에 넘기는 게 일반 케이스. 분리하면 (a) cross-vendor 검증으로 self-agreement bias 구조적 제거, (b) 비대칭 quality/cost — 로컬 cheap target + cloud judge, (c) 양쪽 독립 mock으로 테스트, (d) judge call billing 분리. CLI는 서로 미러링되는 `--target-*` / `--judge-*` flag 노출.

**UTF-8 + `errors='replace'`로 파일 읽기.** non-UTF-8 파일 (BOM 동반 Windows 파일, legacy cp949 데이터셋, Latin-1 prompt) 이 도구를 crash시키지 않습니다. 바이트 수준 교체와 CLI warning으로 진행. *"BOM 때문에 calibration 실패"* 와 *"calibration 돌았고 인코딩 이슈를 노트"* 의 차이.

---

## 14. 이 도구가 *아닌* 것

잘못된 용도로 쓰면 규율이 실패합니다. 명시적 non-goal:

| 이 도구는 | 이유 |
|---|---|
| DSPy의 prompt optimizer가 아님 | DSPy는 program 추상화 + bootstrapped few-shot으로 prompt를 *합성*. omegaprompt는 *이미 쓴* prompt를 *calibrate*. DSPy가 후보 3개를 만들면 omegaprompt가 어느 것을 ship할지 결정. 둘은 compose. |
| promptfoo류 prompt 테스트 프레임워크가 아님 | promptfoo는 assertion 기반 grading으로 test case에 prompt를 실행. omegaprompt는 그 위에 *사전 선언된 walk-forward gate*를 추가. promptfoo 출력은 omegaprompt 내부에서 fitness dimension의 하나로 들어옴. |
| 실제 트래픽 A/B의 대체가 아님 | curated 데이터셋의 오프라인 calibration은 cheap한 screening step. business 지표의 real-traffic A/B가 ground truth. omegaprompt는 오프라인 step을 규율할 뿐 온라인 step을 대체하지 않음. |
| 안전성 evaluator가 아님 | `no_safety_violation`을 hard gate로 선언할 수 있지만 judge는 훈련된 safety classifier가 아님. 진지한 safety eval은 omegaprompt + 전용 safety suite (AILuminate, HELM 등) 함께 사용. |
| Frontier LLM 능력 벤치마크가 아님 | 숫자는 judge rubric과 데이터셋만큼만 좋음. omegaprompt는 *당신의 prompt를 당신의 데이터에 당신의 rubric으로* 측정. 다른 task로의 일반화는 unsupported. |
| Free-money 도구가 아님 | calibration은 API 크레딧을 소비. 전형적 run $10–20. iteration 중에는 cheaper judge, shipped run에만 strong judge로 승격. |

위 중 하나의 용도로 이 도구를 쓰고 있다면 잘못 쓰고 있는 것. 비용은 단지 낭비된 API call이 아니라 *여전히 실제로 평가되지 않은 prompt에 대한 false confidence*.

---

## 15. 비용 및 성능

`evaluate()` 호출당 API call: `2 × dataset_size`. dataset 10개면 후보 파라미터 셋당 20 call. 비용은 boundary별 provider tier에 따라 변동.

| 구성 (target / judge) | 10-item 후보 | 125-candidate grid | walk-forward 포함 |
|---|---|---|---|
| Anthropic frontier / Anthropic frontier | ~$0.05–0.10 | ~$6–12 | ~$12–24 |
| OpenAI `gpt-4o` / OpenAI `gpt-4o` | ~$0.03–0.06 | ~$4–8 | ~$8–16 |
| OpenAI `gpt-4o-mini` / Anthropic frontier | ~$0.02–0.04 | ~$2.5–5 | ~$5–10 |
| 로컬 Ollama / OpenAI `gpt-4o` | ~$0.015 (judge only) | ~$2 (judge only) | ~$4 (judge only) |
| 로컬 Ollama / 로컬 Ollama | 무료 (compute만) | 무료 | 무료 |

cost-efficient한 iteration 패턴: 작은 로컬/mini-tier target + frontier-tier judge로 채점 품질만 frontier급. 또는 탐색 중에는 frontier target + cheaper judge, 최종 shipped calibration에만 frontier-both로 승격.

매 CLI invocation이 마지막에 집계 토큰 사용량을 출력. `cache_read_input_tokens`는 provider 간 정규화 — Anthropic의 native 필드와 OpenAI의 `prompt_tokens_details.cached_tokens`가 같은 슬롯에 들어옴. 연속 run에서 0이면 judge prompt에서 뭔가 drift — CLI가 명시적으로 surface하고 비용을 silent하게 흡수하지 않음. 로컬 endpoint에선 0이 정상.

---

## 16. 검증

기본 test suite는 live provider/API call 없이 mock provider로 실행됩니다. 정확한 test count는 README prose에 반복하지 않고, 보존된 top badge와 claim ledger에서만 관리합니다. provider API 테스트는 `SimpleNamespace` / `MagicMock`으로 mock하고 request payload shape (model, `response_format`, thinking config, `cache_control` 배치, few-shot 순서) 를 assert합니다.

| 모듈 | 커버리지 |
|---|---|
| `domain/` | `PromptVariants` / `PromptSpace` / `CalibrationArtifact` / 6 enum — 필수 필드, 범위, JSON roundtrip, provider metadata, schema v2.0 호환. |
| `dataset.py` | JSONL loader — 스키마 검증, 중복 id, 빈 줄, missing file. |
| `judge/` | `RuleJudge` / `LLMJudge` / `EnsembleJudge` — short-circuit 동작, scale 검증, 가중치 정규화, out-of-scale 클램프, gate 집계. |
| `core/fitness.py` | `CompositeFitness` — 빈 batch, all-pass, partial-fail, all-fail, per-item 보존. |
| `core/walkforward.py` | KC4 Pearson 계산, gap 계산, threshold 비교, status 산출. |
| `providers/` | factory가 unknown 이름 거부, `base_url` 존중. Anthropic adapter가 `messages.create` + `messages.parse` payload 정확히 build, refusal raise. OpenAI adapter가 `chat.completions.create` + `beta.chat.completions.parse` build, `prompt_tokens_details.cached_tokens` → `cache_read_input_tokens` 정규화, `content_filter` raise. |
| `api.py` | `call_target` / `judge.score` — provider delegation, payload shape, thinking on/off, reference-block 포함, rubric serialisation. |
| `target.py` | `PromptTarget` — mocked provider로 end-to-end, cross-vendor metadata 기록, default 해결, parameter clamping, usage 누적. |
| `cli.py` | help / version / subcommand wiring, exit code, profile flag, `check-artifact` 동작. |

`uv run pytest -q`로 실행. 전형 wall time: 2초 미만.

---

## 17. 3-layer stack

```
       ┌─────────────────────────────────────────────┐
LAYER  │  omegaprompt  (본 repo)                      │  "prompt에 규율 적용 — Prompt CI"
APPLY  │  v2.0.0 — schema v2.0, 6 meta-axes          │
       └────────────────────┬────────────────────────┘
                            │ depends on
                            ▼
       ┌─────────────────────────────────────────────┐
LAYER  │  omega-lock                                 │  "calibration framework"
CORE   │  v0.3.0 — stress + grid + walk-forward + KC │
       └────────────────────┬────────────────────────┘
                            │ built under
                            ▼
       ┌─────────────────────────────────────────────┐
LAYER  │  Antemortem + antemortem-cli                │  "build 주변 규율"
META   │  pre-impl recon 방법론 + tooling             │
       └─────────────────────────────────────────────┘
```

- **[omega-lock](https://github.com/hibou04-ops/omega-lock)** — calibration engine 공급: stress 측정, top-K unlock, grid search, walk-forward, kill criteria, benchmark scorecard. 자체 test suite와 release record는 해당 repo가 source of truth입니다.
- **[Antemortem](https://github.com/hibou04-ops/Antemortem)** + **[antemortem-cli](https://github.com/hibou04-ops/antemortem-cli)** — omega-lock과 omegaprompt 모두 이 pre-implementation reconnaissance 규율 아래 빌드. Antemortem은 코드 작성 *전* ghost trap을 잡고, omega-lock은 ship 전 overfit parameter를, omegaprompt는 deploy 전 overfit prompt를 잡습니다. 같은 패턴이 spec → parameters → prompts 세 스케일에서 반복.

omegaprompt가 ~200 라인 adapter인 이유는 layering이 신뢰성을 만들기 때문입니다 — calibration engine이 본 adapter 작성 *전*에 ship되고 검증되었습니다.

---

## 18. MCP 서버

`omegaprompt[mcp]` extras가 FastMCP 기반 서버를 포함합니다. Claude Code / Cursor가 다음 도구를 직접 호출할 수 있습니다:

- `calibrate` — 데이터셋·rubric·variants를 받아 full calibration 실행, `CalibrationArtifact` 반환.
- `evaluate` — 단일 (params, dataset) 조합 evaluate. cost estimate에 유용.
- `report` — artifact JSON을 markdown audit report로 렌더.
- `diff` — 두 artifact 비교, regression 분류.
- `measure_sensitivity` — 6 meta-axes 별 fitness sensitivity 측정.
- `grade` — judge만 단독 실행.
- `preflight` — mini-omega-lock / mini-antemortem-cli 플러그인을 통해 환경 적합성 검사.
- `classify_traps` — config를 7개 calibration trap 패턴으로 분류.

agent 시나리오 9개 cookbook: [AGENT_TRIGGERS.md](AGENT_TRIGGERS.md).

---

## 19. 회의론자를 위한 FAQ

**이건 promptfoo + walk-forward step 아닌가요?**
어느 정도 그렇습니다 — 그게 요점입니다. walk-forward step *자체*가 전체 규율입니다. promptfoo는 예제를 실행, omegaprompt는 그 랭킹을 mechanical하게 falsifiable로 만듭니다. KC4 없이는 promptfoo가 의미가 있을지 없을지 모르는 숫자를 줍니다. KC4 있으면 failing run이 ship 전에 iterate가 필요하다고 알려줍니다.

**왜 `temperature`를 축으로 안 두나요?**
모던 frontier 모델들이 sampling knob을 점진적으로 제거/deprecate합니다. pinned 계약은 *"우리가 sampling을 고른다; 당신은 effort와 prompt shape를 고른다."* 제거된 knob을 calibrate하는 것은 의미가 없습니다. 여전히 `temperature`를 받는 모델용이라면 `PromptSpace`를 20줄짜리 axis adapter로 확장 — framework 변경이 아닙니다.

**target과 judge가 같은 모델이면 judge가 그냥 동의하지 않나요?**
잘 쓴 rubric에서는 그렇지 않습니다. judge는 *특정 scale의 정수* 와 *binary gate*로 *특정 rubric에 대해* 채점하도록 prompt 받습니다. 두 호출은 독립 — 다른 system prompt, 다른 task framing, target call에 대한 기억 없음. 그래도 *다른* (이상적으로 더 강한) 모델을 judge로 쓰는 것이 권장 production 셋업입니다. v0.2의 multi-judge 패턴이 이를 formalize.

**Judge가 miscalibrated면?**
그러면 calibration도 miscalibrated. judge quality는 first-class concern. 세 방어선: (1) judge의 ~5k 토큰 system prompt가 명시적 calibration-hygiene 지시 포함 ("midpoint anchor, full-range, self-agree"), (2) rubric 스키마가 dimension별 정수 scale을 요구하고 out-of-range를 clamp, (3) v0.2 multi-judge 패턴 — `judge_v1` vs `judge_v2` 랭킹 비교, disagreement = trust signal이지 결과가 아님.

**Anthropic의 built-in evals와 어떻게 다른가요?**
Anthropic evals는 native API surface입니다. omegaprompt는 그 위에 *사전 선언된* gating을 추가합니다 — rubric은 run 전에 고정, threshold는 config에 있고, score를 본 뒤에 둘 다 조정 불가. quant finance에서 빌려온 Winchester defense — 사전 선언된 kill criteria는 완화될 수 없음. native eval은 이를 *수동으로 할 수 있게*; omegaprompt는 *구조적으로 enforce*.

**non-Anthropic 모델에서 작동하나요?**
네. 두 boundary 모두 `LLMProvider` Protocol을 통과합니다. Anthropic, OpenAI, OpenAI-호환 endpoint (Azure, Groq, Together, OpenRouter, vLLM, Ollama) 가 ship됩니다. cross-vendor target/judge가 first-class 구성.

**private/proprietary 데이터셋에서 실행 가능한가요?**
네. dataset, rubric, variants 모두 로컬 JSON/JSONL. 도구가 이들을 읽고 calibration 호출을 위해 API로 보내고 artifact를 로컬에 씁니다. 데이터가 외부로 나가는 경로는 LLM provider API 하나뿐 — 일반 application call과 동일.

**비용은?**
10-item dataset, 125-candidate grid, walk-forward 기준 frontier 가격으로 $10–20 / run. 두 mitigation: (1) iteration 중 cheaper judge (4–5× 감소), (2) cache-aware prompt가 judge 비용을 dominate하므로 5분 window 내 *두 번째* run은 첫 번째의 ~50%.

**calibration이 실제로 generalize됐는지 어떻게 아나요?**
artifact의 `walk_forward.generalization_gap`을 읽으세요. 10% 미만 = 강한 generalization. 10–25% = non-critical path에 수용 가능. 25% 초과 = training slice가 production을 대표하지 못함, dataset 확장. `status`가 `OK`가 아니거나 `ship_recommendation`이 `SHIP`이 아니면 score와 무관하게 ship 금지.

---

## 20. 선행 연구 및 credit

세 아이디어 위에 서 있습니다:

- **사전 선언된 gate를 동반한 train/test split** — 모든 학부 ML 커리큘럼의 기초 ML 방어. 여기 사용된 형태 (KC4, Pearson rank correlation gate) 는 [omega-lock](https://github.com/hibou04-ops/omega-lock)의 kill-criteria framework에서 옴.
- **LLM-as-judge** — *[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)* (Zheng et al., 2023) 에서 popularize. omegaprompt는 이 패턴을 schema enforcement (Pydantic + 벤더 native structured output) 로 구현해, free-form judge 응답이 scoring 파이프라인을 오염시키는 알려진 failure mode를 차단.
- **Winchester defense** — 원래 quant-finance 규율: *run 전에 선언된 kill criteria는 이후 완화 불가*. KC4가 score를 본 후 사후 조정이 아니라 config에서 enforce되어야 한다는 주장의 근거.

네이밍: *omega-lock* (parameter calibration) → *omegaprompt* (prompt calibration) 은 의도적 family branding. omega-lock 자체가 `KC-4 FAIL`로 끝난 quant 거래전략 calibration 실험에서 추출된 도구입니다 — overfitting 방어가 설계대로 작동한 결과. omegaprompt는 같은 통찰을 한 layer 위에 적용.

---

## 21. 상태 & 로드맵

**v2.0.1** — package release 2.0.1입니다. omega-lock 0.3.0 호환(`EvalResult.sample_count` alias)과 consumer docking hardlock(contract test + canary)을 추가했습니다. CalibrationArtifact schema는 v2.0으로 유지되며, 6 meta-axes, RuleJudge / LLMJudge / EnsembleJudge, guarded/expedition profile, MCP server, generated claim ledger를 포함합니다.

데이터 계약 (`Dataset`, `JudgeRubric`, `PromptVariants`, `PromptSpace`, `CalibrationArtifact`), CLI 계약 (`omegaprompt calibrate / report / diff / check-artifact`), Python/MCP runtime 계약 (`preflight`, `measure_sensitivity`, `grade`, `classify_traps`) 은 stable. CLI exit code는 `0` clean/advisory success, `1` CI gate failure 또는 regression, `2` environment/config/tooling/input 문제로 고정합니다. judge prompt는 실 run의 scoring-quality 데이터가 축적되면서 iterate — minor bump가 judge prompt 개정용, CHANGELOG의 *Judge prompt revisions* 섹션에 추적.

**다음 트랙**

- **judge 품질 dogfood** — code generation, reasoning, extraction, classification 다양 task에서 scoring drift 측정. reference scoring-quality benchmark.
- **multi-judge 검증 패턴** — top-K에 `judge_v1` + `judge_v2`, disagreement = trust signal.
- **`--dry-run`** — calibration 시작 전 비용 추정.
- **rubric 측 second `cache_control` breakpoint** — 같은 rubric 반복 run의 캐시 최적화.
- **GitHub Action** — PR에서 calibration 실행, KC4 fail 시 merge 차단.

**명시적 out of scope**: 웹 대시보드, proprietary hosting, 멀티 테넌트. omegaprompt는 로컬 개발자 도구 — 로컬에 유지합니다.

전체 changelog: [CHANGELOG.md](CHANGELOG.md).

---

## Troubleshooting

### `omegaprompt calibrate`가 401 / Incorrect API key를 리턴할 때

provider SDK가 키는 받았지만 invalid한 경우. 두 가지 흔한 원인:

- **만료/회수된 키**: 발급한 dashboard (Anthropic / OpenAI / Google AI Studio)에서 확인하고 재발급 후 export.
- **잘못된 환경변수**: provider별로 자기 변수만 읽으며, **vendor 간 fallback은 없습니다**:

  | Provider | 허용 환경변수 |
  |---|---|
  | `anthropic` | `ANTHROPIC_API_KEY` |
  | `openai` | `OPENAI_API_KEY` |
  | `gemini` | `GEMINI_API_KEY` **또는** `GOOGLE_API_KEY` (먼저 set된 것이 우선) |
  | `local` / `ollama` / `vllm` / `llama_cpp` | 불필요 (`--base-url` 사용) |

  `OPENAI_API_KEY`를 설정해도 `--target-provider gemini`는 인증되지 않습니다. 반대도 마찬가지.

### `ProviderError: Gemini API key is required for provider='gemini'`

`GEMINI_API_KEY`도 `GOOGLE_API_KEY`도 set되지 않은 경우. <https://aistudio.google.com/apikey> 에서 free tier key 발급 후:

```bash
export GEMINI_API_KEY=AIza...
```

### API 비용 쓰기 전에 sanity check하려면

deterministic smoke test — 키도 네트워크도 필요 없습니다:

```bash
python examples/reference/reproduce_reference_artifact.py
omegaprompt report examples/reference/reference_artifact.json
```

calibration kernel + judge + artifact 직렬화를 provider 호출 없이 end-to-end로 돌립니다. 여기서 fail이면 install 자체가 깨진 것이고, pass면 provider 호출이 다음 검증 대상.

provider별 single live call (최소 비용 spend)을 직접 만들려면:

```python
from omegaprompt.providers.factory import make_provider
from omegaprompt.providers.base import ProviderRequest

req = ProviderRequest(system_prompt="Be brief.", user_message="Say OK.")
for name in ("anthropic", "openai", "gemini"):
    try:
        resp = make_provider(name).call(req)
        print(name, "OK", resp.usage)
    except Exception as e:
        print(name, "FAIL", e)
```

여기서 401은 키 문제, `ImportError`는 vendor SDK 누락 (`pip install "omegaprompt[anthropic]"` 등), 정상 응답은 provider 경로가 healthy하다는 뜻 — 그러면 eval 진행 가능.

### Gemini 호출은 되는데 guarded mode에서 `LLMJudge`가 거부될 때

설계상 의도된 동작입니다. Gemini는 `ship_grade_judge=False` — guarded profile에서 judge tier 체크가 fail-fast 합니다 (검증되지 않은 judge 위에 ship recommendation을 올리지 않으려는 것). 두 가지 해결책:

- Gemini를 **target**으로 쓰고, Anthropic / OpenAI를 **judge**로 (cross-vendor도 guarded 만족).
- `--profile expedition` 사용 — fail 대신 `RelaxedSafeguard`를 기록하고 진행. artifact에 relaxed boundary가 남아 downstream `diff`에서 surface됩니다.

forked adapter에서 `ship_grade_judge=True`로 바꾸려면 별도 검증을 먼저 거치세요.

---

## 신뢰 및 toolkit 문서

claim-heavy한 설명은 별도 문서에 둡니다. README는 경로를 안내하고, 정확한 public claim은 claim ledger와 deterministic artifact가 source of truth입니다.

- [Trust model](docs/trust-model.md): `CalibrationArtifact`가 증명하는 것과 증명하지 않는 것, train/test discipline, walk-forward/KC4 한계, offline/live evidence, default CI no-live 원칙, MCP optional boundary, diff regression 사용.
- [Toolkit positioning](docs/toolkit-positioning.md): `omegaprompt`와 `omega-lock`, `antemortem-cli`, optional `mini-*` preflight plugin, `omegacal` compatibility alias, dashboard/web-app 비범위.
- [Provider capabilities](docs/provider-capabilities.md): provider capability code와 contract test에 묶인 adapter claim.
- [Profiles and risk boundaries](docs/profiles-and-risk-boundaries.md): guarded vs expedition, validation mode 해석.

---

## 22. 라이선스

Apache 2.0. [LICENSE](LICENSE) 참조. © 2026 Kyunghoon Gwak (hibou04-ops).

## Colophon

solo 설계·구현·ship. omega-lock 위의 adapter layer — calibration engine 재구현 zero. 기본 CI/test 경로는 live provider/API call 없이 실행되며, public claim은 claim ledger로 추적합니다. 본 도구는 caller에게 제공하는 pre-implementation reconnaissance 규율 아래 빌드되었습니다.
