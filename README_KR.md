# omegaprompt (한국어)

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![PyPI](https://img.shields.io/badge/pypi-0.2.0-blue.svg)](https://pypi.org/project/omegaprompt/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#상태--로드맵)
[![Providers](https://img.shields.io/badge/providers-anthropic%20%7C%20openai%20%7C%20openai--compatible-informational.svg)](#provider-지원)
[![Parent framework](https://img.shields.io/badge/framework-omega--lock-blueviolet.svg)](https://github.com/hibou04-ops/omega-lock)

> **당신의 prompt 가 당신이 고른 예시들에서 4.8/5 를 받습니다. 프로덕션 2일차에 무너집니다.**
>
> 이 실패는 ML 에서 이름이 있습니다 — *overfitting* — 그리고 LLM 보다도 오래된 해결책도 있습니다: 사전 선언된 gate 가 있는 train/test split. omegaprompt 는 prompt engineering 을 위한 그 gate, **설계상 model-agnostic**. Target 과 judge 각각이 pluggable `LLMProvider` 와 통신 (Anthropic, OpenAI, 또는 로컬 Ollama 포함 OpenAI-호환 endpoint). Training-best prompt 가 held-out 데이터에서 랭크되지 않으면 KC-4 에 실패하고 어떤 것도 ship 되지 않음.

```bash
pip install omegaprompt
```

English README: [README.md](README.md)

---

## 목차

- [이 도구가 해결하는 failure mode](#이-도구가-해결하는-failure-mode)
- [Worked example: overfit 탐지](#worked-example-overfit-탐지)
- [Calibratable 한 축들](#calibratable-한-축들)
- [Provider 지원](#provider-지원)
- [Cross-vendor 검증](#cross-vendor-검증)
- [데이터 계약](#데이터-계약)
- [아키텍처](#아키텍처)
- [Non-trivial 한 설계 결정](#non-trivial-한-설계-결정)
- [이것은 아닙니다](#이것은-아닙니다)
- [비용 및 성능](#비용-및-성능)
- [검증](#검증)
- [3-layer stack](#3-layer-stack)
- [인접 도구와의 관계](#인접-도구와의-관계)
- [회의론자를 위한 FAQ](#회의론자를-위한-faq)
- [선행 연구 및 credit](#선행-연구-및-credit)
- [상태 & 로드맵](#상태--로드맵)
- [기여](#기여)
- [인용](#인용)
- [라이선스](#라이선스)

---

## 이 도구가 해결하는 failure mode

Prompt engineering 에는 예측 가능한 failure mode 가 있습니다. 직접 고른 몇 예시에 대해 iterate. 점수가 올라감. Prompt 가 좋아 보임. Ship 함. 이틀 뒤, 상상도 못 한 input 에서 무너짐.

이건 prompt-engineering 실력 문제가 아닙니다. 1990년대에 전체 train/test/validate split 을 주류 ML 에 밀어넣은 그 실패와 똑같습니다: **학습된 configuration 은 학습한 데이터에서 검증될 수 없습니다.** 모든 ML 강의의 모든 튜토리얼이 여기서 시작. 모든 prompt-engineering 가이드가 이걸 건너뜀.

omegaprompt 는 sibling 프로젝트 [omega-lock](https://github.com/hibou04-ops/omega-lock) 을 통해 ML 방어를 prompt 설정으로 포팅. 세 아이디어가 그대로 transfer:

1. **Sensitivity 측정.** 어떤 prompt 축이 실제로 중요한가? 각각을 neutral baseline 주변에서 perturb, fitness delta 의 Gini coefficient 로 랭크. 점수를 안 움직이는 축에 search budget 쓰지 않음.
2. **Top-K unlock, 나머지 lock.** Fitness 를 움직이는 subspace 에서만 탐색. 나머지는 neutral 고정.
3. **사전 선언된 gate 와 함께 하는 walk-forward.** Training slice 에서 탐색 완료 후, searcher 가 본 적 없는 held-out test slice 에서 재평가. 사전 선언된 threshold (`KC-4`) 이상의 Pearson correlation 을 요구. 사후에 threshold 건드리기 없음.

두 가드레일이 omegaprompt 를 *"judge 에게 물어보고 최고 점수 고르기"* 와 구분짓습니다:

1. **Hard gate 가 fitness 를 0 으로 collapse.** `no_refusal`, `format_valid`, `no_safety_violation` — 한 item 에서 한 gate 라도 실패하면, 그 item 은 0 기여. 10개 task 에서 5.0 받고 11번째에서 refuse 하는 prompt 가 refusal 없이 꾸준히 4.2 받는 prompt 보다 랭크 위 *안* 올라옴. Soft 감점은 *almost* refuse 하는 prompt 에 reward. Hard zero 는 안 함.
2. **Walk-forward 가 ship gate.** omega-lock 의 KC-4 (train 과 test 랭킹 간 Pearson correlation) 는 config 에 사전 선언되고 mechanical 로 enforce. Borderline 후보를 구하려고 사후에 threshold 낮추기 불가.

---

## Worked example: overfit 탐지

전형적 실패: 10-example training set 에 대해 prompt 를 iterate 하고 승자를 고름.

```
Candidate prompt A:  train_fitness = 0.923  (평균 4.6/5)
Candidate prompt B:  train_fitness = 0.876  (평균 4.4/5)
```

Prompt A 가 training 에서 승리. Ship A? 10-example test slice 에서 walk-forward:

```
Candidate prompt A:  train = 0.923  test = 0.612  gen_gap = 33.7%
Candidate prompt B:  train = 0.876  test = 0.841  gen_gap =  4.0%

omega-lock run_p1 status: FAIL:KC-4
이유: spearman(train_ranks, test_ranks) = 0.18 < 0.30 threshold
Candidate A 의 train-랭킹이 test-랭킹과 uncorrelated. Ship 하지 말 것.
```

Prompt A 가 training slice 에 overfit. 당신이 우연히 고른 10개 예시에 대해 judge 의 surface feature 에 flatter 한 style 을 찾음 — 정확히 ML failure mode 가 예측하는 것. omegaprompt 의 calibration 이 이를 mechanical 로 surface; prompt B 가 올바른 ship 결정.

KC-4 없이는 A 를 ship 하고 프로덕션에서 gap 발견. KC-4 있으면 calibration 이 loud fail, deploy 전에 gap 발견.

Generalization gap 은 output artifact 에 기록:

```json
{
  "best_params": {"system_prompt_idx": 2, "few_shot_count": 1, "effort_idx": 2},
  "best_fitness": 0.876,
  "test_fitness": 0.841,
  "generalization_gap": 0.040,
  "hard_gate_pass_rate": 1.00,
  "method": "p1",
  "status": "PASS"
}
```

이 도구가 생성하는 모든 `CalibrationOutcome` 의 shape. Machine-readable, prompt 개정 간 diffable, CI 에서 gate 하기 trivial.

---

## Calibratable 한 축들

`PromptTarget` 은 searcher 에게 5개 축을 노출:

| 축 | 타입 | 의미 |
|---|---|---|
| `system_prompt_idx` | int | 후보 system prompt 풀에 대한 인덱스 (`ParamVariants.system_prompts`). |
| `few_shot_count` | int | `ParamVariants.few_shot_examples` 중 몇 개를 포함할지. 0 = zero-shot. |
| `effort_idx` | int (0-2) | `effort: low / medium / high` 에 매핑. Thinking 이 켜졌을 때만 유의미. |
| `thinking_enabled` | bool | Target call 에 adaptive thinking 을 켤지 여부. |
| `max_tokens_bucket` | int (0-2) | `max_tokens: 1024 / 4096 / 16000` 에 매핑. Length-bound 편향 surface. |

`PromptSpace` dataclass 는 축을 잠글 수 있게 해줌 (`effort_min == effort_max`) — 일부 dimension 이 pre-decided 일 때. 전형적 calibration 은 3개 축을 열어두고 sensitivity 로 top-K 를 unlock. 모든 축을 단일 값으로 두면 사실상 judge 를 통한 고정 prompt benchmark 실행; prompt 개정 regression 테스트용 legitimate 모드.

v0.1 의 축 목록은 의도적으로 보수적. `temperature`, `top_p`, reasoning budget 같은 knob 을 추가하면 릴리스 간 변하는 model-specific 동작에 coupling. 위 5개 축은 어떤 chat-style frontier 모델에서도 transfer.

---

## Provider 지원

omegaprompt 는 두 개의 LLM-호출 경계를 가짐 — **target** (calibration 대상 prompt, free-form 출력) 과 **judge** (rubric 기반 스코어링, schema-enforced 출력). 둘 다 `LLMProvider` Protocol 을 통과. 각 adapter 는 vendor 의 가장 강한 네이티브 경로 사용 — 파이프라인 어디에도 클라이언트 측 JSON regex 파싱 없음.

| Provider | Flag (target) | Flag (judge) | 기본 모델 | Env var | Native 경로 |
|---|---|---|---|---|---|
| Anthropic | `--target-provider anthropic` | `--judge-provider anthropic` | `claude-opus-4-7` | `ANTHROPIC_API_KEY` | 명시적 `cache_control` 을 동반한 `messages.create` / `messages.parse` |
| OpenAI | `--target-provider openai` | `--judge-provider openai` | `gpt-4o` | `OPENAI_API_KEY` | `chat.completions.create` / `beta.chat.completions.parse` |
| OpenAI-호환 | 위 + `--target-base-url <url>` | 위 + `--judge-base-url <url>` | `--*-model` 로 사용자 지정 | `OPENAI_API_KEY` (로컬 미인증 endpoint 는 임의 문자열) | OpenAI 와 동일 |

`--base-url` 로 커버되는 호환 endpoint: **Azure OpenAI**, **Groq**, **Together.ai**, **OpenRouter**, 그리고 **로컬 Ollama** (`http://localhost:11434/v1`). 새 provider 추가는 `LLMProvider` Protocol (두 메서드: `complete` + `structured_complete`) 을 만족하는 한 모듈 + factory registry 한 줄.

**Protocol** (`src/omegaprompt/providers/base.py`):

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

두 메서드. SDK 누출 없음. 같은 prompt 가 모든 vendor 에서 작동.

---

## Cross-vendor 검증

이 도구가 enable 하는 가장 강력한 단일 구성: **target 은 한 provider, judge 는 다른 provider.**

이유: target 과 judge 가 같은 모델 (또는 같은 vendor) 이면 self-agreement bias 가 점수를 오염. OpenAI judge 로 평가된 약한 OpenAI 출력은 judge 가 target 의 bias 를 공유하므로 여전히 acceptable 해 보일 수 있음. 같은 출력을 Anthropic judge — 혹은 반대로 — 로 평가하면 구조적으로 bias 가 깨짐. Judge 가 더 이상 peer 가 아니라 disinterested second opinion.

```bash
# Target: gpt-4o 에서 프로덕션 prompt.
# Judge: 다른 vendor 의 higher-tier disinterested 모델.
omegaprompt calibrate train.jsonl \
  --rubric rubric.json \
  --variants variants.json \
  --test test.jsonl \
  --target-provider openai \
  --target-model gpt-4o \
  --judge-provider anthropic \
  --judge-model claude-opus-4-7 \
  --output outcome.json
```

`CalibrationOutcome` artifact 는 `target_provider`, `target_model`, `judge_provider`, `judge_model` 기록 — 재현성이 모든 run 의 machine-readable 속성.

또 하나 유용한 조합: Ollama 로 로컬 target + 클라우드 judge 로 채점.

```bash
# Target: 로컬 Llama (무료 추론). Judge: 채점 품질을 위해 cloud-hosted.
omegaprompt calibrate train.jsonl \
  --rubric rubric.json --variants variants.json --test test.jsonl \
  --target-provider openai \
  --target-base-url http://localhost:11434/v1 \
  --target-model llama3.1:70b \
  --judge-provider anthropic \
  --judge-model claude-opus-4-7
```

---

## 데이터 계약

이 도구가 생성하는 모든 artifact 는 Pydantic-validated. 데이터가 end-to-end 로 흐름:

```python
# Input: 세 개의 사용자 작성 파일
Dataset(items=[DatasetItem(id="t1", input="...", reference="..."), ...])
JudgeRubric(
    dimensions=[
        Dimension(name="correctness", description="...", weight=0.5, scale=(1, 5)),
        Dimension(name="clarity",     description="...", weight=0.3),
        Dimension(name="conciseness", description="...", weight=0.2),
    ],
    hard_gates=[
        HardGate(name="no_refusal",        description="...", evaluator="judge"),
        HardGate(name="no_safety_violation", description="...", evaluator="judge"),
    ],
)
ParamVariants(
    system_prompts=["You are a...", "You are a senior...", "You are a teacher..."],
    few_shot_examples=[{"input": "...", "output": "..."}, ...],
)

# ↓ omega-lock.run_p1 이 PromptTarget.evaluate() 를 여러 (params, slice) 조합으로 drive
# 각 evaluate() 호출이 주어진 params dict 에 대해 N target calls + N judge calls 발급.

# ↓ messages.parse(output_format=JudgeResult)
JudgeResult(
    scores={"correctness": 5, "clarity": 4, "conciseness": 3},  # 선언된 scale 내 정수
    gate_results={"no_refusal": True, "no_safety_violation": True},
    notes="Response 가 task 를 올바르게 해결. Wrap-up 문단에 약간의 padding.",
)

# ↓ CompositeFitness 가 (hard_gate × soft_score) 로 집계
# fitness = sum(soft_score_i * all_gates_passed_i) / n_items

# ↓ omega-lock run_p1 이 walk-forward 결과 발급
CalibrationOutcome(
    best_params={"system_prompt_idx": 2, "few_shot_count": 1, "effort_idx": 2, ...},
    best_fitness=0.876,      # training slice
    test_fitness=0.841,      # held-out slice
    generalization_gap=0.040,
    hard_gate_pass_rate=1.00,
    method="p1",
    total_api_calls=1250,
    usage_summary={"input_tokens": ..., "cache_read_input_tokens": ...},
)
```

Malformed judge 응답은 SDK 경계에서 `ValidationError` raise — fitness 를 오염시키지 않음. `JudgeResult.gate_results` 의 missing hard-gate 결과는 "답 안 함" 으로 취급, "통과" 와 별개. 모든 model 의 모든 필드는 계약; 위반은 loud.

---

## 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│  Dataset (.jsonl)   ParamVariants (.json)   JudgeRubric (.json) │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  PromptTarget                                                 │
│    omega-lock CalibrableTarget 구현                           │
│    param_space() ───▶ 5 축                                    │
│    evaluate(params):                                          │
│       for each dataset item:                                  │
│         1. params 에서 prompt build (variant, few-shot, ...)  │
│         2. call_target(target_client, ...) ─▶ response text   │
│         3. call_judge(judge_client, rubric, ...) ─▶ JudgeResult │
│       CompositeFitness(judge_results) ─▶ fitness              │
│    returns EvalResult(fitness, n_trials, metadata)            │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  omega-lock run_p1                                            │
│    measure_stress + select_unlock_top_k                       │
│    Unlocked subspace 에서 GridSearch                          │
│    --test slice 에서 WalkForward (KC-4)                       │
│    grid_best, test_fitness, status 출력                       │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  CalibrationOutcome (JSON artifact)                           │
│    best_params, best_fitness, test_fitness,                   │
│    generalization_gap, hard_gate_pass_rate,                   │
│    n_candidates_evaluated, total_api_calls, usage_summary     │
└──────────────────────────────────────────────────────────────┘
```

Omega-lock 라인 위의 모든 조각은 새로 구현; 그 라인에 또는 아래의 모든 조각은 변경 없이 재사용. Composition 경계는 `CalibrableTarget` 프로토콜 — 하나의 함수 (`param_space()`) + 하나의 메서드 (`evaluate(params)`).

`PromptTarget` 이 ~200 라인인 이유. 비싼 부분 (sensitivity 분석, search, walk-forward, kill criteria) 은 이미 존재하고 이미 omega-lock 의 176 unit tests 로 테스트됨. omegaprompt 는 prompt-specific adapter, LLM-as-judge machinery, fitness shape 를 기여. 나머지는 모두 composition.

---

## Non-trivial 한 설계 결정

**두 boundary 에서 provider-agnostic.** omegaprompt 가 vendor SDK 를 건드리는 곳은 정확히 둘: target 호출과 judge 호출. 둘 다 `LLMProvider` Protocol 을 통과. Calibration 파이프라인 (stress, grid, walk-forward, KC-4) 은 구성상 vendor-neutral. Target 과 judge 는 *독립적* provider 를 받음 — CLI 한 줄이 cross-vendor 검증으로 flip.

**재구현이 아닌 단일 책임 adapter.** omega-lock 이 이미 stress 측정, top-K unlock, grid search, walk-forward, KC gate, benchmark scorecard, iterative lock-in, 연속-공간 탐색용 `run_p2_tpe` 처리. 여기서 이들을 재구현하는 것은 틀림. `PromptTarget` 은 얇은 adapter; 가치는 모든 omega-lock 파이프라인이 변경 없이 작동한다는 것.

**Per-vendor 네이티브로 schema-enforce 된 judge 응답.** 각 adapter 는 vendor 의 가장 강한 structured-output 경로 사용 — Anthropic 은 `messages.parse(output_format=JudgeResult)`, OpenAI 는 `beta.chat.completions.parse(response_format=JudgeResult)`. Malformed judge 응답 — missing 필드, 선언된 scale 바깥 scores, non-boolean gate 결과 — 은 SDK 경계에서 `ValidationError` raise. 이 없으면, 한 번의 잘못된 judge call 이 전체 calibration 을 tank 시킬 수 있고, 최종 report 까지 눈치채지 못함. Judge 의 prose output 을 regex-parse 하는 것은 antemortem-cli 에서 모델의 self-report line number 를 신뢰하는 것과 같은 실수.

**Hard gate 가 fitness 를 0 으로 collapse, gradient 없음.** Refusal 에 soft penalty (예: "refusal 은 20% 감점") 를 주면 *almost* refuse 하는 prompt 를 reward — soft score 대부분을 capture 하면서 safety boundary 테스트. Hard-zero 는 refusal 을 절대적으로 처벌. Searcher 는 refusal region 안쪽에서 reward signal 을 못 봐서 boundary 근접을 안 함. 실제 deployment 가 prompt 를 평가하는 방식과 일치: 10번 중 1번 refuse 하는 prompt 는 "90% 만큼 좋은" 게 아니라 unshippable.

**Adapter-native 로 judge system prompt 에 prompt caching.** Judge 의 ~5k 토큰 system prompt 는 각 vendor 의 cacheable-prefix 최소값을 넘도록 sized. Anthropic 은 명시적 `cache_control={"type": "ephemeral"}` 사용; OpenAI 의 자동 prompt caching 은 system prompt 가 provider threshold 를 넘으면 발동. 전형적 calibration run 이 수백 judge call — cache hit 이 지원되는 모든 vendor 에서 비용을 dominate. CLI 는 provider 간 정규화된 `cache_read_input_tokens` 를 surface (OpenAI 의 `prompt_tokens_details.cached_tokens` 가 같은 필드로 매핑), silent invalidator 가 어디서 실행하든 loud fail. [전체 judge prompt](src/omegaprompt/prompts.py) 는 vendor-neutral prompt-cache-aware 설계 사례.

**`effort` 와 `thinking_enabled` 는 first-class 축, global 아님.** 어떤 prompt 는 `low` effort 만 필요; 전역으로 `high` 를 강제하면 fitness 개선 없이 토큰 낭비. Calibration 이 이것을 surface 하게 하는 게 핵심 — `effort_idx` 가 high stress 면 task 에 중요; low stress 면 neutral 로 lock + 토큰 절약. "측정하기 전 최적화하지 말라" 의 parameter-calibration analog.

**Target 과 judge provider 는 독립적, first-class.** 같은 provider 를 양쪽에 넘기는 게 일반 케이스. 나누면 (a) cross-vendor 검증 — Anthropic judge 가 OpenAI 출력 평가 (또는 반대) 가 self-agreement bias 를 구조적으로 깸; (b) 비대칭 quality/cost — cheap 로컬 target (Ollama) + 클라우드 judge 채점; (c) 테스트 — 양쪽을 독립적으로 mock; (d) billing 격리 — judge call 을 다른 workspace 로 라우팅. CLI 가 서로 미러링되는 `--target-*` 와 `--judge-*` flag 노출.

**Prompt 에 들어가기 전 모든 path 는 forward slash 로 정규화.** `src\foo.py` 와 `src/foo.py` 는 API payload 에서 다른 바이트. Prompt caching 은 바이트가 일치해야 cache-invariant. Windows 사용자는 조용히 cache hit 을 잃을 것 — 100 run 당 ~\$15 비용 버그.

**`temperature` / `top_p` 축 없음.** 모던 frontier-tier chat 모델들이 점점 이들을 제거/deprecate (Anthropic 최신 Claude pin 은 drop; OpenAI 의 o1/o3 reasoning 모델은 accept 안 함). Support 하는 척하고 일부 서버에서 실패하기보다, omegaprompt 는 기본 공간에서 배제. 더 오래된 모델에 필요하면 `PromptSpace` 를 axis 추가로 extend — 20줄짜리 adapter 이지 프레임워크 변경 아님.

**UTF-8 + error='replace' 파일 읽기.** 비-UTF-8 파일 (BOM 붙은 Windows 파일, legacy cp949 데이터셋, Latin-1 extraction prompt) 이 도구를 crash 시키지 않음. 바이트 수준 교체와 CLI warning 으로 읽힘. "BOM 때문에 내 calibration 이 실패했습니다" 와 "내 calibration 이 돌았고 인코딩 이슈를 note 했습니다" 의 차이.

---

## 이것은 아닙니다

잘못된 용도로 쓰면 discipline 이 실패. 명시적 non-goals:

| 이 도구는 | 이유 |
|---|---|
| DSPy 의 prompt optimizer 가 아님 | DSPy 는 program 추상화 + bootstrapped few-shot 을 통해 prompt 를 synthesize. omegaprompt 는 *이미 쓴* prompt 를 *calibrate*. DSPy 가 세 개의 후보 prompt 생성하면, omegaprompt 가 어느 것을 ship 할지 결정. 그들은 compose. |
| promptfoo 같은 prompt 테스트 프레임워크가 아님 | promptfoo 는 assertion 기반 grading 으로 test case 에 대해 prompt 실행. omegaprompt 는 그 위에 *사전 선언된 walk-forward gate* 추가. fitness dimension 중 하나로 omegaprompt 안에서 promptfoo 실행 가능. |
| 실제 트래픽의 real eval 대체가 아님 | Curated 데이터셋의 오프라인 calibration 은 cheap 한 screening step. Business 메트릭의 real-traffic A/B 가 ground truth. omegaprompt 는 오프라인 step 을 discipline; 온라인 step 을 대체하지 않음. |
| 안전성 evaluator 가 아님 | `no_safety_violation` 은 선언할 수 있는 hard gate 지만 judge 는 훈련된 safety classifier 가 아님. 진지한 safety eval 은 omegaprompt 의 calibration 을 전용 safety eval suite (예: AILuminate, HELM) 와 함께 사용. |
| Frontier LLM 능력 벤치마크가 아님 | 숫자는 judge rubric 과 데이터셋만큼만 좋음. omegaprompt 는 *당신의 prompt 를 당신의 데이터에 당신의 rubric 으로* 측정. 이 숫자들을 다른 task 로 일반화하는 건 unsupported. |
| Free-money 도구가 아님 | Calibration 은 API 크레딧 소비. 전형적 run 은 \$10–20. 예산 세우고; iteration 중에는 cheaper model 을 judge 로, shipped run 에만 강한 judge 로 승격. |

위 중 하나로 이 도구를 쓰고 있다면 잘못 쓰고 있는 것. 잘못된 사용의 비용은 낭비된 API call, 그리고 더 나쁘게는 중요한 것에 대해 여전히 평가되지 않은 prompt 에 대한 false confidence.

---

## 비용 및 성능

`evaluate()` 당: 2 × (dataset_size) API 호출 — item 당 target 1회, judge 1회. 전형적 10-item dataset = 후보 파라미터 셋 당 20 API 호출. 비용은 각 boundary 에 어떤 provider + tier 를 두느냐에 따라 변동.

| 구성 (target / judge) | 10-item 후보 | 125-candidate grid | Walk-forward 포함 |
|---|---|---|---|
| Anthropic frontier / Anthropic frontier | ~\$0.05–0.10 | ~\$6–12 | ~\$12–24 |
| OpenAI `gpt-4o` / OpenAI `gpt-4o` | ~\$0.03–0.06 | ~\$4–8 | ~\$8–16 |
| OpenAI `gpt-4o-mini` / Anthropic frontier | ~\$0.02–0.04 | ~\$2.5–5 | ~\$5–10 |
| Ollama 로컬 / OpenAI `gpt-4o` | ~\$0.015 (judge only) | ~\$2 (judge only) | ~\$4 (judge only) |
| Ollama 로컬 / Ollama 로컬 | 무료 (컴퓨팅만) | 무료 | 무료 |

가장 cost-efficient iteration 패턴: prompt iteration 중에는 작은 로컬/mini-tier target + frontier-tier judge 로 품질 채점. 또는 탐색 중 frontier target + 더 저렴한 judge; 최종 shipped calibration 에만 frontier-both 로 승격.

매 CLI invocation 이 집계 토큰 사용량을 마지막에 출력. `cache_read_input_tokens` 가 provider 간 정규화 — CLI 가 Anthropic 의 네이티브 필드와 OpenAI 의 `prompt_tokens_details.cached_tokens` 를 같은 slot 에 읽음. 연속 run 에서 0 이면 judge prompt 에서 뭔가 drift — CLI 가 이를 명시적으로 surface 하고 조용히 비용 흡수 안 함. 로컬 endpoint (Ollama) 에서는 0 cache token 이 예상되므로 경고 무시.

---

## 검증

**73 tests, CI 에서 네트워크 호출 0.** 모든 provider (현재 + 미래) 는 `LLMProvider` Protocol 로 받음 — 모든 API 테스트는 `SimpleNamespace` 나 `MagicMock` 으로 mock. 테스트 surface 는 실제 서버와 협상 없이 각 request payload 의 *정확한* shape (model, `response_format`, thinking config, cache_control 배치, few-shot 순서) 을 assert.

| 모듈 | 커버리지 |
|---|---|
| `schema.py` | `ParamVariants` / `PromptSpace` / `CalibrationOutcome` — 필수 필드, 범위 검증, JSON roundtrip, provider metadata 필드. |
| `dataset.py` | JSONL loader — 스키마 검증, 중복 id 감지, 빈 줄 내성, 파일 없음 에러. |
| `judge.py` | `Dimension` / `HardGate` / `JudgeRubric` / `JudgeResult` — scale 검증, 가중치 정규화, out-of-scale 클램핑, gate 집계, unknown-dimension 내성. |
| `fitness.py` | `CompositeFitness` — 빈 batch, all-pass, partial-fail, all-fail, per-item 보존. |
| `providers/` | Factory 가 알 수 없는 이름 거부, `base_url` 존중, 호환 안 되는 kwargs 제거. Anthropic adapter 가 `messages.create` + `messages.parse` kwargs 를 올바르게 build, refusal raise. OpenAI adapter 가 `chat.completions.create` + `beta.chat.completions.parse` kwargs build, `prompt_tokens_details.cached_tokens` → `cache_read_input_tokens` 정규화, `content_filter` raise. |
| `api.py` | `call_target` / `call_judge` — provider delegation, payload shape, thinking on/off, reference-block 포함, rubric serialisation. |
| `target.py` | `PromptTarget` — 모킹된 provider 로 end-to-end, cross-vendor metadata 기록, 기본값 해결, 파라미터 clamping, usage 누적. |
| `cli.py` | Help / version / subcommand wiring. |

`uv run pytest -q` 로 실행. 전형적 wall time: 1초 미만.

---

## 3-layer stack

omegaprompt 는 세 프로젝트 시스템의 applied layer:

```
       ┌─────────────────────────────────────────────┐
LAYER  │  omegaprompt  (이 repo)                     │  "Discipline 을 prompt 에 적용"
APPLY  │  v0.1.0 — Claude API prompt calibration     │
       └────────────────────┬────────────────────────┘
                            │ 의존
                            ▼
       ┌─────────────────────────────────────────────┐
LAYER  │  omega-lock                                 │  "Calibration 프레임워크"
CORE   │  v0.1.4 — stress + grid + walk-forward + KC │
       └────────────────────┬────────────────────────┘
                            │ 검증됨
                            ▼
       ┌─────────────────────────────────────────────┐
LAYER  │  Antemortem + antemortem-cli                │  "빌드 주변 discipline"
META   │  pre-impl recon methodology + tooling       │
       └─────────────────────────────────────────────┘
```

- **[omega-lock](https://github.com/hibou04-ops/omega-lock)** — calibration engine 공급: stress 측정, top-K unlock, grid search, walk-forward, kill criteria, benchmark scorecard. 176 tests. 2026-04-18 ship.
- **[Antemortem](https://github.com/hibou04-ops/Antemortem)** + **[antemortem-cli](https://github.com/hibou04-ops/antemortem-cli)** — omega-lock 과 omegaprompt 둘 다 이 pre-implementation reconnaissance discipline 아래 빌드. Antemortem 은 코드가 쓰이기 전 ghost trap 을 잡고; omega-lock 은 ship 전 overfit parameter 를 잡고; omegaprompt 는 deploy 전 overfit prompt 를 잡음. 패턴이 세 스케일에서 반복: spec, parameters, prompts.

Layering 은 신뢰성에 중요. Calibration engine 이 이 prompt adapter 가 쓰이기 전에 ship 되고 검증됨. omegaprompt 가 ~200 라인 adapter code 인 이유는 필요한 모든 것이 이미 존재하기 때문.

---

## 인접 도구와의 관계

| 도구 | 하는 일 | omegaprompt 가 추가하는 것 |
|---|---|---|
| **[promptfoo](https://www.promptfoo.dev/)** | Test case 에 대해 prompt 실행; assertion 기반 grading. | 사전 선언된 walk-forward gate (KC-4) — training ≠ ship criterion. Soft 감점 아닌, fitness 를 collapse 시키는 hard gate. Stress 기반 축 선택. Composable — promptfoo output 이 여러 judge 중 하나 가능. |
| **[DSPy](https://dspy.ai/)** | Program 추상화 + bootstrapped few-shot 을 통한 prompt synthesis. | Domain-agnostic adapter (어떤 `CalibrableTarget` 이든 작동). Program synthesis 아닌 calibration-first framing. DSPy output 은 search space 의 또 하나의 `system_prompt_variant`. |
| **Optuna / Ray Tune** (prompt 에 적용) | Prompt knob 에 대한 일반 HPO. | Walk-forward 검증 + 사전 선언된 kill criteria 기본 탑재. 스키마 enforce 된 LLM-as-judge response. Project 마다 재발명 아닌 composite `hard_gate × soft_score` fitness. |
| **커스텀 "eval suite"** | Model 호출, 점수 매기기, 랭크하는 project-specific 스크립트. | 구조화된 data contract (`Dataset`, `Rubric`, `Outcome`), machine-readable artifact, 재현성, 30-run reference benchmark 가 있는 calibration engine 에 plug-and-play. |
| **Anthropic 의 built-in [evals](https://docs.anthropic.com/claude/docs/evaluate-and-improve-performance-of-claude-models)** | Provider-native eval 워크플로, `messages.create` + rubric. | 같은 인프라 + discipline: 결과를 본 후에는 threshold 선언 불가. Antemortem/omega-lock 계보가 pre-declaration 을 강제. |

USP 는 *discipline, search 아님*. Search 부분은 omega-lock 이 처리 (어떤 `CalibrableTarget` 에든 처리). omegaprompt 는 prompt-specific adapter 와 hard-gates-first fitness shape 를 기여.

---

## 회의론자를 위한 FAQ

**이건 promptfoo + walk-forward step 아닌가요?**

어느 정도 맞음 — 그게 요점. Walk-forward step *자체*가 전체 discipline. promptfoo 는 예시를 실행; omegaprompt 는 랭킹을 mechanical 로 falsifiable 하게 만듦. KC-4 없이는 promptfoo 가 의미가 있을 수도 없을 수도 있는 숫자를 줌. KC-4 있으면, failing run 이 ship 전에 iterate 해야 함을 알려줌.

**왜 `temperature` 를 축으로 쓰지 않나요?**

모던 Claude pinned 모델은 sampling knob 을 제거. Pinned 계약은 *"우리가 sampling 을 고름; 당신은 effort level 과 prompt shape 를 고름."* 제거된 knob 을 calibrate 하는 건 난센스. 여전히 `temperature` 를 accept 하는 오래된 모델용으로는 `PromptSpace` 에 `temperature_idx` 축 확장 — 20줄짜리 adapter 이지 프레임워크 변경 아님.

**Target 과 judge 가 같은 모델이면 judge 가 그냥 target 에 동의하지 않나요?**

잘 쓴 rubric 에는 해당 안 됨. Judge 는 *특정 scale 의 정수* 와 *binary gate* 로 *특정 rubric 에 대해 평가* 하도록 prompt 받음. 두 call 은 독립 — 다른 system prompt, 다른 task framing. Judge 는 target call 에 대한 기억이 없음. 그래도 *다른* 모델을 judge 로 사용 (이상적으로 더 강한) 하는 게 권장 프로덕션 셋업, v0.2 `multi-judge` 패턴이 이를 formalize.

**Judge 가 miscalibrated 면?**

그럼 당신의 calibration 도 miscalibrated. Judge quality 는 first-class 관심사. 세 방어: (1) judge 의 ~5k 토큰 system prompt 가 명시적 calibration-hygiene 지시 포함 ("midpoint 에 anchor, full range 사용, 자기 자신과 agree"); (2) rubric 스키마가 per-dimension 정수 scale 을 요구하고 out-of-range score 를 clamp; (3) v0.2 `multi-judge` 패턴이 `judge_v1` vs `judge_v2` 랭킹 비교 — disagreement 는 trust signal 이지 결과 아님.

**Anthropic 의 built-in evals 와 어떻게 다른가요?**

Anthropic 의 evals 는 네이티브 API surface. omegaprompt 는 그 위에 *사전 선언된* gating 추가: rubric 은 run 전에 고정, threshold 는 config 에 있고, score 를 본 뒤에는 둘 다 조정 불가. Quant-finance 에서 빌려온 Winchester defense: 사전 선언된 kill criteria 는 완화될 수 없음. Native evals 는 이를 수동으로 할 수 있게; omegaprompt 는 구조적으로 enforce.

**비-Anthropic 모델에서 작동하나요?**

v0.1 에서는 아님. System-prompt 계약, effort 파라미터, `messages.parse` structured-output flow, caching 의미 — 모두 Anthropic SDK-specific. Vendor 추상화 layer 는 v0.3. 당분간, Claude prompt calibrate 할 때 omegaprompt 사용; cross-vendor 필요하면 일반 HPO 라이브러리 (Optuna).

**Private/proprietary 데이터셋에서 실행 가능한가요?**

네. Dataset + rubric + variants 모두 로컬 JSON/JSONL. 도구가 이들을 읽고 calibration 호출을 위해 Anthropic API 로 보내고 artifact 를 로컬에 씀. 당신의 데이터는 Anthropic API 로만 노트북 바깥으로 나감 — 어떤 애플리케이션 호출과 같음.

**비용은?**

10-item dataset, 125-candidate grid, walk-forward 기준 현재 frontier 가격으로 calibration run 은 \$10–20. 두 mitigation: iteration 중 cheaper model 을 judge 로 사용 (4–5× 감소), cache-aware prompt 가 judge 쪽 비용을 dominate 하므로 5분 window 내 *두 번째* run 이 첫 번째의 ~50%. 예산 세우기.

**Calibration 이 실제로 generalize 됐는지 어떻게 아나요?**

Outcome 의 `generalization_gap` 읽기. 10% 미만 = 강한 generalization. 10–25% = non-critical path 에 수용 가능. 25% 초과 = training slice 가 production 을 대표하지 못함; 데이터셋 확장. `status` 필드가 `PASS` 또는 `FAIL:KC-4` — FAIL 이면 score 와 관계없이 ship 하지 말 것.

---

## 선행 연구 및 credit

이 도구가 서 있는 세 아이디어:

- **사전 선언된 gate 와 함께 하는 train/test split** — 모든 학부 ML 커리큘럼에 문서화된 기초 ML 방어. 여기서 사용된 특정 형태 (gate 로서 Pearson rank correlation, KC-4) 는 [omega-lock](https://github.com/hibou04-ops/omega-lock) 의 kill-criteria 프레임워크에서 옴.
- **LLM-as-judge** — *[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)* (Zheng et al., 2023) 에서 popularize 된 패턴. omegaprompt 는 free-form judge 응답이 scoring 파이프라인을 오염시키는 알려진 failure mode 에 대한 방어로 스키마 enforcement (Pydantic + `messages.parse`) 로 이 패턴을 구현.
- **Winchester defense** — 원래 quant-finance discipline: *run 전에 선언된 kill criteria 는 이후에 완화될 수 없음*. 여기선 `KC-4` 가 score 검사 후 사후 조정이 아니라 config 에서 enforce 되어야 함을 주장하는 데 사용.

네이밍: *omega-lock* (parameter calibration) → *omegaprompt* (prompt calibration) 은 의도적 family branding. omega-lock 은 `KC-4 FAIL` 로 끝난 거래전략 calibration 실험에서 추출 — overfitting 방어가 설계대로 작동. omegaprompt 는 같은 인사이트를 한 layer 위에 적용.

---

## 상태 & 로드맵

v0.1.0 은 **alpha**. 데이터 계약 (`Dataset`, `JudgeRubric`, `ParamVariants`, `PromptSpace`, `CalibrationOutcome`) 은 stable. CLI 계약 (`omegaprompt calibrate`, 그 flag 들, exit code) 는 stable. Judge prompt 는 실제 run 에서 scoring-quality 데이터가 축적되면서 iterate — v0.1.x bump 는 judge prompt 개정 용도, CHANGELOG 의 *"Judge prompt revisions"* 섹션에 추적.

Semver 는 v1.0 부터 엄격 적용.

**v0.1.x (judge prompt iteration 트랙)**
- 다양한 task 타입 (code generation, reasoning, extraction, classification) 에 dogfood. Scoring drift 기록.
- Reference scoring-quality benchmark — judge prompt 개정을 추측이 아니라 측정으로.
- Judge round-trip 없이 호출 가능한 추가 hard gate evaluator (format predicate, safety classifier).

**v0.2 (도구 깊이)**
- `omegaprompt report <outcome.json>` — 사람이 읽을 수 있는 debrief renderer.
- Multi-judge validation 패턴: top-K 에 대한 `judge_v1` + `judge_v2`, disagreement = trust signal.
- Calibration run 시작 전 비용 추정을 포함한 `--dry-run`.
- 반복 same-rubric run 을 위한 rubric 에 두 번째 `cache_control` breakpoint.

**v0.3 (생태계)**
- Benchmark harness: multiple (task × rubric × seed) 조합, omega-lock 과 같은 RAGAS-style scorecard.
- CI gating 용 GitHub Action — PR 에서 calibration 실행, KC-4 fail 시 merge 차단.
- 비-Anthropic 모델 (OpenAI / Gemini) 에 대해 작동하는 Vendor 추상화.

**명시적 out of scope:** 웹 대시보드, proprietary hosting, 멀티유저 tenancy. omegaprompt 는 로컬 개발자 도구; 로컬에 유지.

전체 changelog: [CHANGELOG.md](CHANGELOG.md).

---

## 기여

가장 가치있는 기여는 공개된 calibration outcome — dataset, rubric, 그리고 여러 method 에 걸친 `CalibrationOutcome.json`. Judge prompt 를 evidence-based 로 만들어줌.

Issue 와 PR 환영. Non-trivial 한 변경은 먼저 [`antemortem-cli`](https://github.com/hibou04-ops/antemortem-cli) 로 antemortem 실행 — 이 프레임워크를 빌드한 discipline 을 dogfood.

---

## 인용

```
omegaprompt v0.1.0 — calibration discipline for Claude API prompts.
https://github.com/hibou04-ops/omegaprompt, 2026.
```

Parent framework:
```
omega-lock v0.1.4 — sensitivity-driven coordinate descent calibration framework.
https://github.com/hibou04-ops/omega-lock, 2026.
```

Methodology (이것과 sibling 들이 어떻게 빌드됐나):
```
Antemortem v0.1.1 — AI-assisted pre-implementation reconnaissance for software changes.
https://github.com/hibou04-ops/Antemortem, 2026.
```

---

## 라이선스

MIT. [LICENSE](LICENSE) 참조.

## Colophon

Solo 로 설계, 구현, ship. omega-lock 위의 adapter layer; calibration engine 재구현 zero. 59 tests, CI 에서 live API 호출 0. 이 도구는 caller 에 제공하는 pre-implementation reconnaissance discipline 으로 빌드됨.
