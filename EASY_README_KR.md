# omegaprompt — 빠른 안내 (한국어)

> 본 문서는 [README_KR.md](README_KR.md)의 압축판입니다. 전체 reference가 필요하면 본 문서를 먼저 읽고 README_KR로 넘어가세요.
> English: [EASY_README.md](EASY_README.md) · [README.md](README.md)

Public claim과 deterministic reference metric은 생성된 [claim ledger](docs/claims/README_CLAIMS.generated.md)에서 추적합니다.

---

## 한 줄 정의

**omegaprompt는 prompt engineering을 위한 CI입니다.** train/test split, 사전 선언된 gate, holdout 상관 검증, CI를 fail시키는 audit artifact — ML이 1990년대부터 써 온 방어선을 prompt에 그대로 옮긴 것. **calibration set에 overfit한 prompt는 빌드를 깨뜨립니다.**

## 해결 대상

직접 고른 소수 예제에 prompt를 맞춰 높은 점수 → ship → production 지표 악화. 이건 prompt 실력이 아니라 검증되지 않은 *generalization* 문제입니다. 대부분의 LLM eval 도구는 *어느* prompt가 best score인지만 알려주고, 그 score가 일반화되는지는 묻지 않습니다.

omegaprompt가 강제하는 것:

- **train/test split + 사전 선언 KC4 gate** — score를 본 뒤 threshold를 낮출 수 없습니다.
- **walk-forward 검증** — `test_fitness`가 `train_fitness`를 따라가지 못하면 run이 flagged.
- **cross-vendor judge** — Anthropic target을 OpenAI judge로 (또는 반대) 평가해 self-agreement bias 위험을 구조적으로 줄임.
- **provider-neutral meta-axes** — 벤더 고유 knob이 아니라 semantic 축(reasoning profile, output budget, schema mode, tool policy)에서 탐색. 마이그레이션에서 깨지지 않음.
- **`CalibrationArtifact` schema v2.0** — PR diff에 그대로 들어가는 JSON. regression이면 CI fail.

## KC4 한 줄 요약

KC4 = **holdout 상관 gate**. train에서는 좋아 보였는데 test에서는 declared target과 더 이상 상관 없는 경우 — 즉 calibration set에 overfit한 경우 — fail. threshold(`--min-kc4`, `--max-gap`)는 *score를 보기 전*에 config로 선언합니다. `status = FAIL_KC4_GATE`는 설계상 ship-blocker.

## 60초 멘탈 모델

```
dataset (train) ──┐
rubric            ├─▶ meta-axes 탐색 ─▶ walk-forward (test) ─▶ KC4 gate ─▶ artifact
prompt 후보       │   (reasoning,         (Pearson + gap)        PASS / FAIL    (JSON, diffable)
provider/judge ───┘    budget, schema, ...)
```

입력: `dataset + rubric + prompt 후보 + provider`.
출력: `CalibrationArtifact` JSON — neutral baseline vs calibrated fitness, uplift, walk-forward 수치, capability degradation event, ship recommendation.

## 설치

```bash
pip install omegaprompt              # core
pip install "omegaprompt[mcp]"       # + MCP server (Claude Code, Cursor)
```

`omegaprompt`는 PyPI distribution, primary import package, primary CLI입니다. `omegacal`은 compatibility alias이고, `omega-lock`은 별도 calibration engine dependency입니다.

## Offline vs live

- Offline deterministic path, key/network 불필요: `python examples/reference/reproduce_reference_artifact.py`.
- Live provider path: provider API key를 설정한 뒤 `omegaprompt calibrate ...`를 실행합니다. 기본 test와 generated claim은 live API call을 하지 않습니다.

## 가장 짧은 경로 — CLI

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

omegaprompt calibrate train.jsonl \
  --test test.jsonl \
  --rubric rubric.json \
  --variants variants.json \
  --output artifact.json \
  --target-provider anthropic \
  --judge-provider openai \
  --profile guarded
```

확인할 필드: `.status` (`OK` / `FAIL_KC4_GATE` / `FAIL_HARD_GATES`), `.ship_recommendation` (`SHIP` / `HOLD` / `EXPERIMENT` / `BLOCK`).

## CI에 꽂기

```bash
omegaprompt diff previous.json artifact.json   # 회귀 시 non-zero exit
```

```yaml
# .github/workflows/prompt-audit.yml
- run: omegaprompt diff previous.json artifact.json
```

`calibrated_fitness`, `walk_forward.test_fitness`, `hard_gate_pass_rate`, cost/latency frontier, guarded-mode 위반 — 어느 하나라도 regression이면 PR을 막습니다.

## Python 경로

```python
from omegaprompt import (
    Dataset, Dimension, HardGate, JudgeRubric,
    PromptVariants, PromptTarget, LLMJudge, make_provider,
)
from omega_lock import run_p1

train = Dataset.from_jsonl("train.jsonl")
test  = Dataset.from_jsonl("test.jsonl")

rubric = JudgeRubric(
    dimensions=[Dimension(name="accuracy", description="정확한가?", weight=1.0)],
    hard_gates=[HardGate(name="no_refusal", description="응답 자체는 시도해야 함", evaluator="judge")],
)

variants = PromptVariants(
    system_prompts=["You are a helpful assistant.", "Be terse. Be accurate."],
    few_shot_examples=[],
)

provider = make_provider("anthropic")           # "openai" | "local" | "ollama" | "vllm"
judge    = LLMJudge(provider=provider)

train_t = PromptTarget(target_provider=provider, judge=judge,
                       dataset=train, rubric=rubric, variants=variants)
test_t  = PromptTarget(target_provider=provider, judge=judge,
                       dataset=test,  rubric=rubric, variants=variants)

result = run_p1(train_target=train_t, test_target=test_t)
# 최적 파라미터: result.grid_best["unlocked"]
```

## 6개 meta-axes — 실제 탐색 공간

벤더별 parameter 이름이 아니라 semantic 카테고리로 탐색합니다. 각 provider adapter가 내부에서 번역. 동일 artifact가 벤더 간 재생됩니다.

| Axis | 값 | 통제 대상 |
|---|---|---|
| `system_prompt_variant` | prompt 리스트 인덱스 | 어느 system prompt |
| `few_shot_count` | 0, 1, 2, ... | 포함할 예제 개수 |
| `reasoning_profile` | OFF / LIGHT / STANDARD / DEEP | extended thinking 깊이 |
| `output_budget_bucket` | SMALL(1024) / MEDIUM(4096) / LARGE(16000) | max_tokens 버킷 |
| `response_schema_mode` | FREEFORM / JSON_OBJECT / STRICT_SCHEMA | 출력 구조 강제 |
| `tool_policy_variant` | NO_TOOLS / TOOL_OPTIONAL / TOOL_REQUIRED | tool-use 정책 (declared; wiring 일부) |

## 3개 judge

- **`RuleJudge`** — deterministic regex/predicate gate (no_refusal, JSON 유효성, regex 일치). API 비용 0.
- **`LLMJudge`** — capable LLM이 STRICT_SCHEMA로 dimension 별 점수 산출. self-congratulation에 저항하도록 calibration-hygiene 지시가 포함된 system prompt 내장.
- **`EnsembleJudge`** — `RuleJudge` 먼저, 한 gate라도 실패하면 **short-circuit** (LLM 호출 안 함). 통과하면 LLM judge로 escalate. 깨진 응답에 들이는 비용을 잘라냅니다.

## 2개 profile

| | `guarded` (기본) | `expedition` |
|---|---|---|
| silent schema fallback | ❌ raise | ✅ 허용, `CapabilityEvent` 기록 |
| placeholder provider | ❌ raise | ✅ 허용 |
| non-ship-grade judge | ❌ raise | ✅ 허용 |
| 기본 `--max-gap` | 0.25 | 0.35 |
| 기본 `--min-kc4` | 0.5 | 0.3 |

`expedition`에서 풀린 모든 가드는 `RelaxedSafeguard` 항목으로 artifact에 기록됩니다. *strict냐 reach냐*의 trade가 명시적이고 audit 가능.

## 쓸 만한 경우

- 진짜 **train/test 분리**가 있다 (또는 만들 수 있다).
- prompt를 **누군가가 신뢰해야 한다** — ops, compliance, 3개월 뒤의 나.
- **동일 calibration을 다른 벤더에서 재생**하고 싶다 (탐색 축을 다시 짜지 않고).
- **provider의 조용한 capability 저하를 잡아야 한다** (예: 로컬 endpoint가 어느 날 strict schema를 silently drop).

## 과한 경우

- 데모용 1회성 prompt.
- test set이 없고, 결과를 아무도 리뷰하지 않는다.
- 출력 10개 눈으로 훑어보고 ship해도 만족한다.

이 경우는 그냥 playground에서 반복하는 편이 낫습니다.

## 선택적 preflight 플러그인

`omegaprompt`는 **plug-in 인터페이스**(`omegaprompt.preflight.contracts` + `.adaptation`)만 포함합니다. probe 코드는 분리:

- **[mini-omega-lock](https://pypi.org/project/mini-omega-lock/)** — *실 환경* 측정. judge consistency, endpoint 신뢰도, context margin, latency. 실제 API 호출.
- **[mini-antemortem-cli](https://pypi.org/project/mini-antemortem-cli/)** — *config*를 7개 calibration trap 패턴으로 분류 (self-agreement, small-sample power, variant homogeneity 등). 순수 deterministic rule, API 호출 0.

환경 맞춤 adaptive threshold가 필요할 때만 설치하세요.

## 더 깊이

- 전체 reference: [README_KR.md](README_KR.md)
- meta-axes 정의: `src/omegaprompt/domain/enums.py`
- fitness 식 (hard_gate × soft_score): `src/omegaprompt/core/fitness.py`
- walk-forward gate: `src/omegaprompt/core/walkforward.py`
- provider adapter: `src/omegaprompt/providers/`

License: Apache 2.0. © 2026 Kyunghoon Gwak (hibou04-ops).
