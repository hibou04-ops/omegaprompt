# omegaprompt — 쉬운 설명

> 16-섹션 학술 README가 어렵게 느껴지는 분들을 위한 압축 버전.
> 원본: [README_KR.md](README_KR.md) · English easy: [EASY_README.md](EASY_README.md)

## 이게 고치는 문제

Prompt 20개 손수 뽑은 예제로 반복 튜닝. 최고점을 배포. **2일 차 production에서 20개 예제에 없던 입력에 실패.**

그게 overfitting. ML은 1990년대부터 이미 방어법 알고 있음: **held-out test slice**, **사전 선언 상관 임계값**, **수치 미달 시 배포 차단 gate**. 모든 ML 교재에 있는 내용. 대부분의 prompt 튜닝 도구가 이걸 건너뜀.

omegaprompt는 이 3개 방어를 내장한 prompt calibration. **Provider-중립** (동일 artifact를 Anthropic / OpenAI / local / OpenAI-호환 간 재생 가능) + **모든 degradation 기록** (provider가 조용히 capability 떨어뜨리면 CI가 봄).

## 60초 멘탈 모델

```
당신의 데이터셋 → meta-axes 탐색 → held-out walk-forward → PASS 또는 FAIL_KC4_GATE
                  (reasoning, budget,   (Pearson + gap gate)     ship-ready artifact
                   schema, variants)
```

당신이 정의: **dataset + rubric + 후보 prompt + provider**.
리턴값: **`CalibrationArtifact` JSON** — neutral-baseline vs calibrated fitness, uplift, walk-forward 수치, capability degradation 이벤트, ship recommendation.

## 설치

```bash
pip install omegaprompt
```

## 가장 쉬운 길: CLI

```bash
export ANTHROPIC_API_KEY=sk-ant-...

omegaprompt calibrate train.jsonl \
  --test test.jsonl \
  --rubric rubric.json \
  --variants variants.json \
  --output result.json \
  --target-provider anthropic \
  --judge-provider anthropic \
  --profile guarded
```

`result.json` (= `CalibrationArtifact`) 에서 `.status` (`OK` / `FAIL_KC4_GATE` / `FAIL_HARD_GATES`) 와 `.ship_recommendation` (`SHIP` / `HOLD` / `EXPERIMENT` / `BLOCK`) 확인.

## Python 경로

```python
from omegaprompt import (
    Dataset, Dimension, HardGate, JudgeRubric,
    PromptVariants, PromptTarget, LLMJudge, make_provider,
)
from omega_lock import run_p1

# 1. 데이터셋 + rubric
train = Dataset.from_jsonl("train.jsonl")
test  = Dataset.from_jsonl("test.jsonl")

rubric = JudgeRubric(
    dimensions=[Dimension(name="accuracy", description="맞나?", weight=1.0)],
    hard_gates=[HardGate(name="no_refusal", description="시도는 해야 함", evaluator="judge")],
)

# 2. 후보 prompt (sensitivity 신호 내려면 최소 2개)
variants = PromptVariants(
    system_prompts=["You are a helpful assistant.", "Be terse. Be accurate."],
    few_shot_examples=[],
)

# 3. Provider + judge
provider = make_provider("anthropic")           # or "openai", "local", "ollama", "vllm"
judge    = LLMJudge(provider=provider)

# 4. Calibrable target 으로 wrap
train_t = PromptTarget(target_provider=provider, judge=judge, dataset=train, rubric=rubric, variants=variants)
test_t  = PromptTarget(target_provider=provider, judge=judge, dataset=test,  rubric=rubric, variants=variants)

# 5. 실행 — 내부적으로 omega-lock의 run_p1 사용
result = run_p1(train_target=train_t, test_target=test_t)

# 최적 prompt 는 result.grid_best["unlocked"] 안에
```

## 6개 meta-axes (실제 탐색 공간)

벤더별 파라미터 이름 대신, semantic 카테고리로 탐색. 각 provider adapter가 내부적으로 번역:

| Axis | 값 | 제어 대상 |
|---|---|---|
| `system_prompt_variant` | prompt 리스트 인덱스 | 어느 system prompt |
| `few_shot_count` | 0, 1, 2, ... | 포함할 예제 개수 |
| `reasoning_profile` | OFF / LIGHT / STANDARD / DEEP | Extended thinking 깊이 |
| `output_budget_bucket` | SMALL (1024) / MEDIUM (4096) / LARGE (16000) | max_tokens |
| `response_schema_mode` | FREEFORM / JSON_OBJECT / STRICT_SCHEMA | 출력 구조 강제 |
| `tool_policy_variant` | NO_TOOLS / TOOL_OPTIONAL / TOOL_REQUIRED | Tool-use 정책 (declared; provider wiring 미완) |

Axes가 semantic이라 동일 artifact 가 벤더 간 재생 가능.

## 3개 judge

- **`RuleJudge`** — deterministic regex/predicate gate (no_refusal, JSON 유효, regex 매치). API 비용 0.
- **`LLMJudge`** — capable LLM이 STRICT_SCHEMA로 dimension 점수 매김. Self-congratulation에 저항하는 judge system prompt 내장.
- **`EnsembleJudge`** — RuleJudge 먼저. Rule gate 하나라도 실패하면 **short-circuit** (LLM 호출 X). 통과하면 escalate. 깨진 응답에 비용 절약.

## 2개 profile

| | `guarded` (기본) | `expedition` |
|---|---|---|
| Silent schema fallback | ❌ raise | ✅ 허용, CapabilityEvent로 기록 |
| Placeholder provider | ❌ raise | ✅ 허용 |
| Non-ship-grade judge | ❌ raise | ✅ 허용 |
| 기본 `--max-gap` | 0.25 | 0.35 |
| 기본 `--min-kc4` | 0.5 | 0.3 |

`expedition`의 모든 완화는 artifact에 `RelaxedSafeguard` 항목으로 기록. strictness vs reach 의 거래가 explicit + auditable.

## 쓸 가치 있는 경우

- 실제 **train/test 분리** 보유 (또는 만들 수 있음).
- 누군가가 prompt를 **신뢰**해야 함 — ops, compliance, 3개월 후의 나.
- **동일 calibration을 다른 벤더에서 재생**하고 싶음 (검색 axes 재작성 없이).
- **Provider degradation 조용한 것 포착** 필요 (예: local 엔드포인트가 strict schema drop).

## 과한 경우

- 데모용 일회성 prompt.
- Test set / holdout 없음. 결과 아무도 리뷰 안 함.
- Output 10개 눈으로 훑어보고 배포해도 만족.

위 경우엔 playground에서 그냥 반복.

## 선택적 preflight 플러그인

`omegaprompt`는 **플러그인 인터페이스** (`omegaprompt.preflight.contracts` + `.adaptation`) 만 ship. Probe 코드 없음. 별도 패키지 2개가 slot 채움:

- **[mini-omega-lock](https://pypi.org/project/mini-omega-lock/)** — *실제 환경* 측정 (judge consistency, 엔드포인트 신뢰도, context margin, latency). 실제 API 호출.
- **[mini-antemortem-cli](https://pypi.org/project/mini-antemortem-cli/)** — *config*를 7개 calibration trap 패턴으로 분류 (self-agreement, small-sample power, variant homogeneity, ...). Pure deterministic rule, API 호출 0.

실제 셋업에 맞춘 adaptive threshold 원할 때만 설치.

## 더 깊이

- 전체 학술 README: [README_KR.md](README_KR.md) (시스템 구조, 데이터 contract, 모든 appendix)
- Meta-axes 정의: `src/omegaprompt/domain/enums.py`
- Fitness 규칙 (hard_gate × soft_score): `src/omegaprompt/core/fitness.py`
- Walk-forward gate: `src/omegaprompt/core/walkforward.py`
- Provider adapters: `src/omegaprompt/providers/`

License: Apache 2.0. Copyright (c) 2026 hibou.
