# omegaprompt

**당신의 prompt를 위한 overfit gate.** prompt가 당신의 eval set에서 만점을 받았다 — 바로 그것이 아직 그 prompt를 믿어선 안 되는 이유입니다. `omegaprompt`는 이긴 prompt를 그 prompt가 한 번도 튜닝하지 않은 예시들로 다시 테스트하고, 그 결과가 유지되지 않으면 **당신의 CI 빌드를 실패**시킵니다.

[![CI](https://github.com/hibou04-ops/omegaprompt/actions/workflows/ci.yml/badge.svg)](https://github.com/hibou04-ops/omegaprompt/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/pypi/l/omegaprompt?color=blue&label=license&cacheSeconds=3600)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/omegaprompt?color=blue&cacheSeconds=3600)](https://pypi.org/project/omegaprompt/)
[![PyPI](https://img.shields.io/pypi/v/omegaprompt?color=blue&label=pypi&cacheSeconds=3600)](https://pypi.org/project/omegaprompt/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Artifact schema](https://img.shields.io/badge/artifact-schema%20v2.0-blueviolet.svg)](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#8-the-calibrationartifact-schema-v20)
[![MCP](https://img.shields.io/badge/MCP-server-blueviolet.svg)](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#103-mcp-server-claude-code-cursor)
[![Parent framework](https://img.shields.io/badge/framework-omega--lock-blueviolet.svg)](https://github.com/hibou04-ops/omega-lock)

문서: **[Easy start](https://github.com/hibou04-ops/omegaprompt/blob/main/EASY_README.md)** · [Full reference](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md) · [한국어](https://github.com/hibou04-ops/omegaprompt/blob/main/README_KR.md) · [쉬운 한국어](https://github.com/hibou04-ops/omegaprompt/blob/main/EASY_README_KR.md) · [Examples gallery](examples/) · [Claim ledger (신뢰 근거)](docs/claims/README_CLAIMS.generated.md)

키워드: **prompt overfitting · prompt regression testing · LLM eval CI · prompt evaluation · prompt A/B test in CI · held-out validation for prompts · CI ship gate for prompts**

```bash
pip install omegaprompt              # core
pip install "omegaprompt[mcp]"       # + MCP server (Claude Code / Cursor)
```

> **v2.1.1 (2026-06-12)** — 메타데이터 전용 릴리스: composite **Action**의 `action.yml` `description`을 GitHub Marketplace의 125자 제한 미만으로 줄여 Marketplace에 게시하고 `uses: hibou04-ops/omegaprompt@v2.1.1`로 사용할 수 있게 했습니다. 라이브러리/API 변경은 없습니다 — v2.1.0에서 추가된 `omegaprompt gate` CI ship-gate 명령, `--format json`/`--format html` 출력, key가 필요 없는 **`ollama`** provider, 기계가 읽을 수 있는 **overfit-metrics** 블록(`extract_overfit_metrics`)은 그대로입니다. MCP 도구 세트는 8개로 고정(`gate` MCP 도구는 없음); artifact schema는 `2.0`을 유지하며 backward compatible합니다. 정확한 deterministic reference metric은 생성된 [claim ledger](docs/claims/README_CLAIMS.generated.md)에서 추적합니다.

<!-- public-claim-ledger:start -->
> Claim evidence source: [docs/claims/public_claim_ledger.json](docs/claims/public_claim_ledger.json), rendered by `python tools/generate_readme_claims.py`.
<!-- public-claim-ledger:end -->

신뢰 문서: [trust model](docs/trust-model.md) · [toolkit positioning](docs/toolkit-positioning.md) · [provider capabilities](docs/provider-capabilities.md) · [profiles and risk boundaries](docs/profiles-and-risk-boundaries.md) · [release checklist](docs/release/release-checklist.md) · [post-release verification](docs/release/release-checklist.md#post-release-network-verification).

---

## 당신의 prompt는 eval set에 overfit되어 있다 — 그리고 당신은 그 사실을 모른다

당신은 30개짜리 eval set을 두고 몇 가지 prompt 변형을 튜닝합니다. 변형 #5가 이깁니다 — 4.8/5. 당신은 그것을 ship합니다.

일주일 뒤, production 품질이 이전보다 *더 나빠집니다*. 왜일까요?

4.8은 **당신이 튜닝한 바로 그 예시들** 위에서 나온 점수였기 때문입니다. 당신은 prompt를 측정한 것이 아니라 정답지를 외운 것입니다. 그것이 overfitting이고, 당신의 eval 도구가 PASS를 보고한 것은 그게 그 도구가 하라고 시킨 전부였기 때문입니다. ML은 1990년대에 train/test split으로 이 문제를 풀었습니다. 그런데 대부분의 prompt 워크플로는 여전히 그 split 없이 ship합니다.

`omegaprompt`는 ship하기 **전에** 이를 잡아내는 60초짜리 검사입니다:

1. 당신의 prompt를 **train** 슬라이스 위에서 튜닝합니다(system prompt, few-shot, reasoning effort, output budget, response-schema mode, tool policy 전반에 걸쳐).
2. 이긴 prompt를 **튜닝에 한 번도 쓰지 않은 held-out 슬라이스 위에서 다시 테스트합니다**.
3. **held-out 성능이 train 성능을 따라갈 때에만 ship합니다** — 점수를 매기기 *전에* 당신이 선언하는 두 개의 임계값이라서, 아무도 prompt를 통과시키려고 슬그머니 기준을 낮출 수 없습니다.

그런 다음 CI의 한 줄이, "작은 prompt 수정"이 품질을 조용히 떨어뜨리면 빌드를 실패시킵니다.

> **당신의 eval은 PASS라고 한다. omegaprompt는 그것이 일반화하지 못할 거라고 한다.** 이 문장이 곧 이 제품의 전부입니다.

---

## 이것은 당신의 eval 위에 얹히는 것이지, 그것을 대체하지 않는다

omegaprompt는 promptfoo, DSPy, deepeval, Braintrust, 또는 당신이 직접 만든 harness와 **경쟁하지 않습니다**. 그 도구들은 최선의 prompt를 *찾고 점수를 매깁니다*. omegaprompt는 그들이 빠뜨린 한 가지를 합니다: 그 최선의 prompt가 튜닝되지 않은 데이터 위에서도 살아남는지 알려주는 **train/held-out split + transfer gate** — 더해서, 다음 PR의 "사소한 표현 수정"이 production을 조용히 무너뜨릴 수 없도록 하는 **ship/block CI 판정**입니다.

| | promptfoo / DSPy / 당신의 harness | eval을 눈으로만 본다 | **omegaprompt** |
|---|---|---|---|
| 테스트 케이스에 prompt 실행 | ✓ | 수동 | 당신 것을 입력으로 재사용 |
| 최선의 prompt 찾기 / 최적화 | ✓ (그게 그들의 일) | 손으로 | 그건 이 도구의 일이 아님 |
| Train / held-out split | ✗ (한 세트, 한 번 채점) | ✗ | ✓ 사전 선언; tuner는 held-out을 절대 보지 않음 |
| Held-out transfer gate (train 점수가 held-out 점수를 예측하는가?) | ✗ | ✗ | ✓ per-item 상관 gate |
| Train↔held-out 최대 격차 gate | ✗ | ✗ | ✓ 사전 선언된 임계값 |
| 단일 명령 CI ship/block 판정 | 부분적 | ✗ | ✓ `gate` / `diff`가 non-zero로 종료 |
| 기계가 읽는 "overfit인가?" 숫자 | ✗ | ✗ | ✓ `extract_overfit_metrics` |
| ship **전에** overfit을 잡음 | ✗ | ✗ | ✓ 그게 전부의 핵심 |

> **당신의 tech lead을 위한 한 줄:** promptfoo/DSPy는 *어떤* prompt가 가장 높은 점수를 받았는지 알려줍니다. omegaprompt는 *그 prompt가 held-out 데이터 위에서 유지되는지*를 알려주고, CI에 ship할지 block할지를 결정할 단일 exit code를 줍니다.

당신의 기존 eval 출력은 그대로 꽂힙니다 — assertion은 rule-based gate가 되고, 당신의 dataset은 train/held-out 소스가 됩니다. omegaprompt는 **search-first가 아니라 audit-first**입니다: 당신이 이미 후보를 골랐다고 가정하고, search의 하류에 있는 질문에 답합니다 — *당신은 정말로 일반화했는가?*

---

## 30초 데모 — API key 불필요, network 불필요

https://github.com/user-attachments/assets/d4308cc3-b8c1-4bb7-b67d-f763e6c26f11

gate가 도는 모습을 가장 빨리 보는 방법은 deterministic offline replay입니다. 내장된 in-memory model + judge 대역을 쓰므로 **provider key가 필요 없고 network 호출도 하지 않습니다** — 매 실행이 byte-identical합니다:

```bash
git clone https://github.com/hibou04-ops/omegaprompt.git
cd omegaprompt && pip install -e .

# Replay the deterministic offline calibration (no keys, no network)
PYTHONIOENCODING=utf-8 python examples/demo_replay.py
```

실제 gate 출력을 보게 됩니다:

```
status: OK
ship_recommendation: ship

neutral_fitness:    0.4250     # baseline prompt, no tuning
calibrated_fitness: 0.9250     # winner on the TRAIN slice
uplift_percent:     117.65%    # how much tuning helped on TRAIN
test_fitness:       0.9250     # SAME winner on the HELD-OUT slice
generalization_gap: 0.00%      # train vs held-out — small gap = it transferred
kc4_status:         MISSING_PER_ITEM_SCORES
```

**이 데모의 정직한 해석:** 번들된 데모 dataset은 train/held-out item이 *서로 겹치지 않습니다*(공유 item id 없음). 그래서 per-item transfer gate가 발동할 수 없습니다 — `MISSING_PER_ITEM_SCORES`를 보고하며 gate는 격차 검사만으로 축소됩니다(여기서는 0.00%). per-item transfer gate가 *실제로* 발동하는 것은 진짜 **paired** dataset(train과 held-out이 item id를 공유)에서입니다. 그러니 이 데모의 깔끔한 숫자를 "transfer gate가 통과했다"로 읽지 마세요 — "격차 검사는 통과했고 transfer gate는 채점할 것이 없었다"로 읽으세요. *어떻게 동작하는가*를 참고하세요.

같은 artifact를 한 줄짜리 CI 판정으로 바꿔보세요:

```bash
omegaprompt gate examples/reference/reference_artifact.json
# exit 0 = clear to ship · 1 = ship-blocked (overfit / unmeasured) · 2 = environment/load error

omegaprompt gate examples/reference/reference_artifact.json --format json   # machine summary for CI
```

ship된 어떤 artifact든, 언제든, 오프라인으로 무결성을 검사하세요:

```bash
omegaprompt check-artifact examples/reference/reference_artifact.json --strict
```

---

## 진짜로 돌려보기 — 당신의 dataset, 당신의 provider

```bash
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY / GEMINI_API_KEY — or run keyless against Ollama

omegaprompt calibrate train.jsonl \
  --test test.jsonl \                    # held-out slice the winner is re-tested on
  --rubric rubric.json \                 # your judging rubric
  --variants variants.json \             # candidate system prompts + few-shot
  --target-provider anthropic \
  --judge-provider openai \              # cross-vendor judge avoids self-grading bias
  --min-kc4 0.5 \                        # held-out transfer gate, declared up front
  --max-gap 0.25 \                       # max allowed train↔held-out gap
  --output artifact.json
```

`--min-kc4`는 **held-out transfer gate**이고 `--max-gap`은 **train↔held-out 최대 격차**입니다 — 둘 다 점수를 매기기 *전에* 당신이 정하는 임계값입니다. 출력은 판정이 담긴 단일 JSON `CalibrationArtifact`입니다: `.status`는 `OK` / `FAIL_KC4_GATE` / `FAIL_HARD_GATES`이고, `.ship_recommendation`은 `ship` / `hold` / `experiment` / `block`입니다. 판정이 ship-clean이 아니면 `calibrate`는 **non-zero로 종료**하므로 커맨드라인에서 곧바로 gate가 됩니다. PR이나 리뷰용으로 렌더링하세요:

```bash
omegaprompt report artifact.json > report.md            # Markdown (default)
omegaprompt report artifact.json --format html > report.html   # self-contained scorecard, no JS
omegaprompt report artifact.json --format json          # stable, schema-versioned CI summary
```

---

## CI에 끼워넣기 — 빌드를 실패시키는 prompt A/B 테스트

prompt 변경은 코드 변경입니다. 코드처럼 gate하세요. 전용 **`omegaprompt gate`** 명령이 CI의 주인공입니다: 오프라인 무결성 audit과 held-out transfer/격차(overfit) 판정을 합쳐서 **`0`(ship) / `1`(block) / `2`(environment)로 종료**합니다. 이 repo는 GitHub composite Action을 제공하므로 한 줄짜리 `uses:`면 됩니다:

```yaml
# .github/workflows/prompt-audit.yml
- uses: hibou04-ops/omegaprompt@v2.1.1
  with:
    artifact: artifact.json          # a CalibrationArtifact you produced in a prior step
    format: json                     # machine-readable gate summary
    require-generalization: "true"   # an absent/unverifiable transfer verdict blocks the build
```

raw step이 더 좋으세요? 같은 명령입니다:

```yaml
- run: pip install omegaprompt
- run: |
    omegaprompt calibrate train.jsonl --test test.jsonl \
      --rubric rubric.json --variants variants.json \
      --target-provider anthropic --output new.json
- run: omegaprompt gate new.json --format json    # exit 1 on overfit/unverified — fails the build
- run: omegaprompt diff baseline.json new.json    # exit 1 on regression vs a known-good baseline
```

`gate`는 "*이* artifact가 ship해도 되는가?"에 답하고, `diff`는 "이 PR이 지난번 대비 regression했는가?"에 답합니다. 이제 "방금 system prompt를 손봤어"는 리뷰 가능하고 gate된 변경입니다 — 주사위 던지기가 아닙니다. 완전한 예시 워크플로는 [`examples/ci/ship-gate.yml`](examples/ci/ship-gate.yml)에 있습니다.

---

## 다섯 가지 명령

| 명령 | 하는 일 |
|---|---|
| `omegaprompt calibrate` | train에서 튜닝하고, 이긴 것을 held-out에서 다시 테스트하며, artifact + ship 판정을 씁니다. gate 실패 시 non-zero 종료. |
| `omegaprompt report` | artifact를 Markdown, 단일 파일 HTML scorecard(`--format html`), 또는 안정적인 CI JSON 요약(`--format json`)으로 렌더링합니다. |
| `omegaprompt diff` | 두 artifact 사이의 CI regression gate. **regression 시 non-zero 종료.** deterministic한 machine diff용 `--format json`. |
| `omegaprompt check-artifact` | artifact를 ship 근거로 믿기 전, network 없는 무결성 검사(CI용 `--strict`). |
| `omegaprompt gate` | **CI ship gate**(2.1.0 신규): 무결성 + held-out transfer/격차 판정을 한 명령으로, `0/1/2`로 종료, `--format json`. 실제로 CI에 연결하는 바로 그것. |

함께 설치되는 것: `omegaprompt-mcp`(agent 서버 런처)와 `omegacal`(같은 CLI의 호환 alias).

---

## "내 prompt는 overfit인가?" — 기계가 읽는 단일 숫자

이 질문에 답하는 두 숫자 — **transfer 상관**(per-item train↔held-out 일치도)과 **train↔held-out 격차** — 는 코드나 CI에서 읽을 수 있는 하나의 두드러진 블록으로 노출됩니다:

```python
from omegaprompt import extract_overfit_metrics
import json

artifact = json.load(open("artifact.json"))
m = extract_overfit_metrics(artifact)
print(m.overfit_verdict)        # GENERALIZES / OVERFIT / UNVERIFIABLE / UNKNOWN
print(m.transfer_correlation)   # per-item r, or None when the split is disjoint
print(m.generalization_gap)     # train fitness minus held-out fitness
```

같은 블록이 `omegaprompt report --format json`과 `omegaprompt gate --format json`에 박혀 있어서, CI step(또는 coding agent)이 산문을 파싱하지 않고도 판정을 읽을 수 있습니다. 이것은 기존 artifact에 대한 **순수 읽기**입니다 — 어떤 필드도 추가하지 않으므로 artifact schema는 `2.0`을 유지하고 모든 golden hash는 byte-stable합니다.

---

## 어떻게 동작하는가

당신은 **dataset + judging rubric + 후보 prompt**를 줍니다. 도구는 구조화되고 provider-neutral한 변형 축을 탐색하며 train 슬라이스에서 후보들에 점수를 매깁니다:

| 축 | 무엇을 변화시키는가 |
|---|---|
| System-prompt variant | 당신의 system prompt 중 어느 것 |
| Few-shot count | 예시를 몇 개 포함할지 |
| Reasoning effort | off / light / standard / deep |
| Output budget | small / medium / large 토큰 상한 |
| Response-schema mode | freeform / JSON object / strict schema |
| Tool policy | no tools / optional / required *(선언만 됨; provider에는 아직 연결되지 않음)* |

그런 다음 이긴 구성을 **held-out** 슬라이스에서 다시 테스트하고 사전 선언된 두 개의 gate를 적용합니다:

- **Held-out transfer gate** (`--min-kc4`): per-item held-out 점수가 여전히 target을 따라가야 합니다. train은 좋아 보이는데 held-out item이 더 이상 상관하지 않으면, prompt가 overfit한 것입니다 — 그 실행은 `FAIL_KC4_GATE`로 표시됩니다. *(이건 train과 held-out이 item id를 공유해야 합니다 — "paired" replay. 보통의 서로 겹치지 않는(disjoint) split에서는 아래의 격차 검사로 축소되며, 그래도 여전히 overfit을 잡습니다 — 위의 offline 데모가 정확히 보여주는 것입니다.)*
- **Max gap gate** (`--max-gap`): train↔held-out fitness 격차가 당신이 선언한 상한 아래로 유지되어야 합니다.

두 임계값 모두 점수를 매기기 **전에** 정해집니다. 결과를 본 뒤에 기준을 낮추는 일은 없습니다.

> **용어에 대하여:** 이것은 **held-out validation**입니다 — held-out 테스트 슬라이스이지, 시계열 예측이 아닙니다. prompt는 시계열이 아니므로 걸어 들어갈 진짜 "미래"가 없습니다; holdout split입니다. (밑단의 calibration 엔진은 내부적으로 그 루틴을 여전히 `walk-forward`라고 부릅니다; 쉽게 말해 *holdout*으로 읽으세요.)

### 두 가지 모드

- **Strict mode**(기본값) — 조용한 schema fallback, placeholder provider, ship-grade가 아닌 judge는 조용히 통과하는 대신 **예외를 던지며**, held-out gate는 빡빡한 기본값을 씁니다. ship할 모든 것에는 이것을 쓰세요.
- **Fast mode** — 빠른 로컬 탐색을 위해 그런 완화를 허용하되, 느슨해진 실행이 audit 가능하도록 그 **모든 완화를 artifact에 기록합니다**. 기본 gate가 더 느슨합니다.

### Provider와 agent

- **Provider adapter:** Anthropic, OpenAI, Gemini, 일반 `local` OpenAI-compatible adapter, 그리고 key가 필요 없는 전용 **`ollama`** adapter(로컬 `http://localhost:11434/v1`). 다른 OpenAI-compatible 백엔드(vLLM, llama.cpp, 그리고 모든 `base_url`: Azure, Groq, Together, OpenRouter)는 `local` adapter를 통해 도달합니다 — 그래서 전체 provider 세트는 `anthropic / openai / gemini / local / ollama / vllm / llama_cpp`입니다. 같은 artifact가 vendor 간에 replay되는 것은 축이 vendor-specific knob이 아니라 semantic하기 때문입니다. 채점자가 채점 대상의 동료가 되지 않도록 **cross-vendor judge**(예: Anthropic target, OpenAI judge)를 쓰세요.
- **MCP 서버(8개 도구):** `calibrate`, `evaluate`, `report`, `diff`, `measure_sensitivity`, `grade`, `preflight`, `classify_traps` — coding agent가 PR을 열기 전에 gate를 돌리고 판정을 읽을 수 있게 합니다. *(MCP 도구 세트는 8개로 고정되어 있습니다; 새 `gate` 명령은 CLI·Python 전용이며 MCP 도구가 아닙니다.)*

```bash
pip install "omegaprompt[mcp]"
python -m omegaprompt.mcp           # stdio; Claude Code / Cursor spawn it as a subprocess
```

```json
{ "mcpServers": { "omegaprompt": { "command": "python", "args": ["-m", "omegaprompt.mcp"] } } }
```

---

## 언제 쓰는가

**쓸 가치가 있을 때:** 진짜 train/held-out split이 있다(또는 만들 수 있다); 하류의 누군가가 prompt를 신뢰해야 한다 — ops, compliance, 미래의 당신; 같은 calibration을 다른 vendor에서 replay하고 싶다; prompt PR이 regression 시 CI를 실패시키길 원한다.

**과한 경우:** 일회성 데모 prompt; held-out set이 없고 결과를 검토할 사람도 없다; 출력 10개를 눈으로 봐도 충분하다. 그럴 땐 그냥 playground에서 반복하세요 — 이 도구가 당신에게 주는 것이 없습니다.

**정직한 범위:** offline held-out은 값싼 *스크리닝*이지 production ground truth가 아닙니다. 숫자들은 *당신의 model 위에서 당신의 dataset에 적용한 당신의 rubric*을 반영합니다 — vendor 벤치마크가 아닙니다. safety classifier가 아니며 production A/B 텔레메트리의 대체물도 아닙니다. [제한 사항](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#14-limitations-and-scope-boundaries)을 참고하세요.

---

## 내부 구조

```python
# calibrate / report / diff / check-artifact / gate are the omegaprompt CLI commands.
# run_p1 / P1Config come from the underlying omega-lock calibration engine:
from omega_lock import run_p1, P1Config   # run_p1 = the engine's core calibration run; the CLI wraps it
```

calibration 엔진은 [`omega-lock`](https://github.com/hibou04-ops/omega-lock)으로, 이 prompt adapter가 존재하기 전부터 quant-trading calibration에서 단련된 parameter-calibration 커널입니다. `omegaprompt`는 그것을 held-out gate, 세 judge(`RuleJudge` / `LLMJudge` / `EnsembleJudge`), provider-neutral 축, `CalibrationArtifact` schema(v2.0), CI `diff`, 그리고 새 `gate` ship 판정으로 감싼 prompt-calibration CLI이자 PyPI 배포판입니다.

*omega*라는 이름은 최종 ship 검사 — prompt가 나가기 전에 통과하는 마지막 gate — 입니다. **omegaprompt**라는 이름은 첫 화면에서 당연한 듯 주어지는 게 아니라, 이 페이지의 이 지점에서 비로소 얻어집니다.

> **omegaprompt toolkit의 한 부분** — [omegaprompt](https://github.com/hibou04-ops/omegaprompt) (calibration engine, 본 repo) · [omega-lock](https://github.com/hibou04-ops/omega-lock) (audit framework) · [antemortem-cli](https://github.com/hibou04-ops/antemortem-cli) (pre-implementation recon CLI) · [mini-omega-lock](https://github.com/hibou04-ops/mini-omega-lock) (empirical preflight) · [mini-antemortem-cli](https://github.com/hibou04-ops/mini-antemortem-cli) (analytical preflight) · [Antemortem](https://github.com/hibou04-ops/Antemortem) (방법론). Cross-toolkit cookbook(언제 어느 도구를 부를지, agent 시나리오 9개): [AGENT_TRIGGERS.md](AGENT_TRIGGERS.md).

---

## 더 깊이 들어가기

이 README의 나머지는 전체 reference입니다: architecture, `CalibrationArtifact` schema(v2.0), 데이터 계약, 여섯 개의 검색 축 전부, 세 judge, provider adapter 세부, 검증, 그리고 제한 사항. 처음이세요? **[EASY_README.md](https://github.com/hibou04-ops/omegaprompt/blob/main/EASY_README.md)**부터 시작하세요. 실제로 풀어본 작업들(코드 리뷰, 요약, 번역, 디버깅)은 **[`examples/`](examples/)**에서 둘러보세요.

License: Apache 2.0 · Copyright (c) 2026 hibou · PyPI: [omegaprompt](https://pypi.org/project/omegaprompt/) · CLI: `omegaprompt` (alias `omegacal`) · MCP: `omegaprompt-mcp`

---

## 목차

- [30초 데모 (offline, no keys)](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#30-second-demo--no-api-keys-no-network)
- [1. 문제 정의](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#1-problem-statement)
- [2. 기여](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#2-contributions)
- [3. 시스템 architecture](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#3-system-architecture)
- [4. 핵심 추상화](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#4-key-abstractions)
- [5. calibration 파이프라인](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#5-the-calibration-pipeline)
- [6. 세 judge](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#6-the-three-judges)
- [7. provider adapter](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#7-provider-adapters)
- [8. CalibrationArtifact (schema v2.0)](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#8-the-calibrationartifact-schema-v20)
- [9. CLI 표면](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#9-cli-surface)
- [10. 빠른 시작](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#10-quick-start)
  - [10.1 Python (high-level API)](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#101-python-high-level-api)
  - [10.2 CLI](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#102-cli)
  - [10.3 MCP 서버 (Claude Code, Cursor)](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#103-mcp-server-claude-code-cursor)
- [11. 풀어본 예시](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#11-worked-examples)
- [12. 검증](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#12-validation)
- [13. 비교 포지셔닝](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#13-comparative-positioning)
- [14. 제한 사항과 범위 경계](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#14-limitations-and-scope-boundaries)
- [15. 로드맵](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#15-roadmap)
- [16. 선행 연구와 크레딧](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#16-prior-art-and-credits)
- [신뢰 및 toolkit 문서](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#trust-and-toolkit-docs)
- [부록 A: 데이터 계약](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#appendix-a-data-contracts)
- [부록 B: meta-axis → vendor-parameter 매핑](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#appendix-b-meta-axis-to-vendor-parameter-mapping)
- [부록 C: invariant](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#appendix-c-invariants)
- [부록 D: AdaptationPlan 계약](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#appendix-d-adaptationplan-contract)
- [인용](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#citing)
- [License](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md#license)

---

## 1. 문제 정의

production 환경에서의 prompt engineering에는 아무리 수동으로 반복해도 해결되지 않는 네 가지 실패 모드가 있습니다. 각각이 기술의 문제가 아니라 구조적 문제이기 때문입니다.

### 1.1 training set에 대한 overfitting

실무자는 작은 예시 입력 묶음을 큐레이션하고, 그것에 맞춰 prompt 변형을 반복하며, 최고 득점자를 골라 ship합니다. production 둘째 날, prompt는 training set이 대표하지 못한 입력을 만나고, *"내가 고른 예시에서 4.8을 받았다"*와 *"내가 받게 될 입력에서 4.8을 받는다"* 사이의 품질 격차가 벌어집니다. 이것이 overfitting의 교과서적 정의입니다. 방어법은 1990년대부터 알려져 있었습니다: optimiser가 결코 보지 않고, 점수를 계산하기 전에 선언된 상관 임계값 아래서, ship 시점에만 평가되는 held-out 테스트 슬라이스. 모든 ML 커리큘럼이 이것을 가르칩니다. 그런데 거의 모든 prompt-engineering 워크플로가 이것을 건너뜁니다.

### 1.2 LLM-as-judge의 self-agreement bias

target model과 채점 model이 같은 vendor에서 — 더 나쁘게는 같은 model에서 — 올 때, judge의 편향은 target의 편향과 겹칩니다. vendor의 training 분포에 아첨하는 응답은, 무관심한 제2의 vendor라면 떨어뜨렸을 채점을 통과할 수 있습니다. 그러면 judge는 독립적인 평가자가 아니라 동료입니다. 표준 방어책(더 강한 judge, 다른 model, 다른 vendor)은 파이프라인이 그 두 호출 지점을 "API key 하나 골라서 재사용"이 아니라 API 경계에서 *독립적*으로 다룰 것을 요구합니다.

### 1.3 calibration 축의 vendor-coupling

대부분의 prompt-optimisation 도구는 가장 편리한 vendor의 API 표면에서 검색 축을 물려받습니다: `temperature`, `top_p`, `max_tokens`, `effort`, `thinking_enabled`. 이 이름들은 특정 API 계약의 산물입니다. 그것들로 calibration하면 calibration *원칙* 자체가 특정 vendor의 ergonomics에 묶이고, target model이 마이그레이션되는 순간 artifact는 읽을 수 없게 됩니다. *"effort = high가 이 작업을 개선한다"*를 발견한 검색은, effort parameter가 없는 model에서는 replay할 수 없는 검색이기도 합니다.

### 1.4 숨은 fallback과 조용한 degradation

실제 LLM SDK는 조용히 degrade합니다. reasoning-effort parameter가 로컬 endpoint에서 거부되는데도 요청은 그것 없이 진행됩니다. 구조화 출력 경로가 어떤 model에서 안 되어서 JSON이 regex로 파싱됩니다. cache-control 헤더가 일부 provider에서 무시되어 토큰 비용이 부풀려집니다. 각 경우에 calibration은 여전히 숫자를 만들어냅니다. 그 숫자들은 더 이상 provider 간에 비교할 수 없고, 운영자는 무엇이 바뀌었는지 기록이 없습니다. 이 degradation들을 명명하지 않는 calibration 프레임워크는, 하류의 CI 파이프라인에서 그것들을 보이지 않게 만듭니다.

`omegaprompt`는 이 넷 모두에 대한 응답입니다. 아래의 각 기여는 이 실패 모드 중 하나 이상을 겨냥합니다.

---

## 2. 기여

1. **Provider-neutral meta-axes.** 공개 검색 공간은 vendor-specific parameter 이름이 아니라 semantic 범주(reasoning profile, output budget bucket, response schema mode, tool policy variant)로 표현됩니다. 각 provider adapter는 meta-axis를 내부적으로 그 vendor의 native 표면에 매핑합니다. calibration artifact는 번역된 parameter(예: `effort: high`)가 아니라 meta-axis 값(예: `reasoning_profile: deep`)을 기록하므로, 같은 artifact가 vendor 간에 읽히고 replay됩니다.

2. **Execution profile.** strict mode(`guarded` profile, 기본값)는 검증을 조용히 완화하길 거부합니다 — ship-grade가 아닌 judge는 예외를 던지고, structured-schema에서 산문으로의 fallback도 예외를 던지며, 숨은 capability 손실도 예외를 던집니다. fast mode(`expedition` profile)는 통제된 경계 넘기를 허용하되, 모든 완화는 artifact에 `RelaxedSafeguard` 항목으로 기록됩니다. 두 profile은 엄격함과 도달 범위 사이의 거래를 명시적이고 audit 가능하게 만듭니다.

3. **Capability tier와 명시적 degradation 이벤트.** 각 provider는 `ProviderCapabilities` 레코드를 선언합니다(strict schema, json object, reasoning profile, usage accounting, LLM judging, tool 지원 여부; tier CORE / CLOUD / LOCAL; experimental / placeholder 플래그). adapter가 런타임에 degrade하면 — 예를 들어 거부된 `reasoning_effort` parameter 없이 재시도하면 — capability, 요청된 값, 적용된 fallback, 사용자에게 보이는 메모를 담은 `CapabilityEvent`를 방출합니다. 이 이벤트는 `EvalItemResult` → `EvalResult` → `CalibrationArtifact`를 거쳐 위로 흐르므로 하류의 diff가 capability regression을 탐지할 수 있습니다.

4. **Neutral-baseline 대 calibrated 비교.** `CalibrationArtifact`(schema v2.0)는 neutral-parameter baseline의 fitness와 calibrated best의 fitness를 나란히 기록하며, 절대·퍼센트 uplift, 그리고 두 지점 모두에서의 quality-per-cost·quality-per-latency 비율을 함께 담습니다. 리뷰어는 *최고 점수*만이 아니라 *검색이 아무것도 안 한 것 대비 무엇을 벌었는지*를 봅니다.

5. **사전 선언 임계값을 가진 held-out ship gate.** held-out 테스트 평가는 Pearson-correlation 임계값(`--min-kc4`)과 generalisation-gap 임계값(`--max-gap`)을 쓰며, 이 둘은 execution profile에서 기본값이 정해지고 artifact에 기록됩니다. 점수를 본 뒤에 임계값을 낮출 수 없습니다; 이것이 quant finance에서 빌려온 Winchester defence입니다. `status = FAIL_KC4_GATE`는 구조적으로 ship-blocker입니다.

6. **세 가지 구현이 제공되는 judge 프로토콜.** `LLMJudge`는 provider의 strict-schema 파싱 경로를 씁니다; `RuleJudge`는 format / refusal / regex gate를 위한 deterministic Python predicate를 API 비용 0으로 돌립니다; `EnsembleJudge`는 rule gate가 실패하면 LLM 채점을 단락시킵니다. 셋은 단일 `Judge` 프로토콜 아래 조합되며, `PromptTarget`은 어떤 전략이 연결되어 있는지 모른 채로 이를 소비합니다.

이 기여들이 합쳐져, prompt calibration을 ergonomic한 연습에서, 출력이 점수 스프레드시트가 아니라 CI-gate-ready한 artifact인 audit 가능한 엔지니어링 파이프라인으로 바꿉니다.

---

## 3. 시스템 architecture

### 3.1 계층화된 패키지 구조

```
omegaprompt/
├── domain/        Provider-neutral contracts (enums, dataset, rubric,
│                  params, result, profiles). Depends on nothing.
├── core/          Calibration kernel (fitness, artifact I/O, walk-forward,
│                  sensitivity ranking, profile policy, run risk). Depends
│                  only on domain.
├── providers/     LLMProvider Protocol + adapter implementations
│                  (Anthropic, OpenAI / OpenAI-compatible, Gemini,
│                  local/ollama/vllm/llama_cpp). Translates meta-axes to
│                  vendor parameters, reports capabilities and degradation.
├── judges/        Judge protocol + LLM / Rule / Ensemble implementations.
│                  Depends on domain and providers.
├── targets/       CalibrableTarget protocol + PromptTarget adapter. The
│                  composition point where omega-lock's search layer plugs in.
├── reporting/     Artifact → Markdown renderer.
├── commands/      Typer subcommands: calibrate, report, diff, check-artifact.
└── cli.py         Top-level Typer application.
```

### 3.2 의존성 방향

```
       ┌──────────────────────────────────────────────────────────┐
       │                     domain/                              │
       │ (enums, dataset, params, rubric, result, profiles)       │
       └──────────────────┬───────────────────────────────────────┘
                          │ imported by
                          ▼
       ┌──────────────────────────────────────────────────────────┐
       │                     core/                                │
       │ (fitness, artifact, walkforward, sensitivity, policy)    │
       └──────────────────┬───────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────────────────┐
       │                     providers/                           │
       │ (LLMProvider, ProviderRequest/Response, capabilities,    │
       │  factory, Anthropic/OpenAI/Gemini/local adapters)        │
       └──────────────────┬───────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────────────────┐
       │                     judges/                              │
       │ (Judge protocol; LLMJudge, RuleJudge, EnsembleJudge)     │
       └──────────────────┬───────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────────────────┐
       │                     targets/                             │
       │ (CalibrableTarget, PromptTarget)                         │
       └──────────────────┬───────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────────────────┐
       │              omega-lock search engine                    │
       │   (stress, top-K unlock, grid search, walk-forward)      │
       └──────────────────┬───────────────────────────────────────┘
                          │ produces
                          ▼
       ┌──────────────────────────────────────────────────────────┐
       │   CalibrationArtifact (schema_version=2.0, JSON on disk) │
       └──────────────────────────────────────────────────────────┘
```

의존성 그래프에는 back-edge가 없습니다. `domain`은 `omegaprompt` 내부 어디에서도 import하지 않습니다. `core`는 `domain`만 압니다. `providers`와 `judges`는 search나 target 코드를 결코 import하지 않습니다. `targets`는 adapter 계층이 `CalibrableTarget` 프로토콜을 통해 `omega-lock`의 search engine에 꽂히는 단일 composition 지점입니다.

### 3.3 원칙과 adapter 사이의 경계

calibration *원칙* — sensitivity 측정, top-K unlock, grid search, 사전 선언 gate를 가진 held-out validation, hard-gate × soft-score fitness, artifact schema — 은 vendor-agnostic하며 `core/`와 `domain/`에 삽니다. *adapter 계층* — `reasoning_profile: deep`이 어떻게 vendor-native API 호출이 되는지, vendor의 usage 레코드가 어떻게 `input_tokens / output_tokens / cache_creation_input_tokens / cache_read_input_tokens`로 정규화되는지 — 은 `providers/`에 삽니다. calibration의 무결성을 평가하는 독자는 어느 vendor가 연결되어 있는지 신경 쓰지 않고 `core/`와 `domain/`을 검토할 수 있습니다. 새 provider를 온보딩하는 독자는 search 계층을 읽지 않고 `LLMProvider`를 구현할 수 있습니다.

---

## 4. 핵심 추상화

### 4.1 Meta-axes

공개 검색 공간을 이루는 여섯 개의 축:

| 축 | 타입 | 의미 | vendor-native 번역 예 |
|---|---|---|---|
| `system_prompt_variant` | `int` | `PromptVariants.system_prompts`로의 인덱스. | 메시지 수준의 system-prompt 치환. |
| `few_shot_count` | `int` | `PromptVariants.few_shot_examples`에서 가져올 예시 수. | 메시지 리스트 prefix 길이. |
| `reasoning_profile` | enum `OFF / LIGHT / STANDARD / DEEP` | target이 reasoning effort를 얼마나 쓸지. | Anthropic `thinking={"type":"adaptive"}` + `{low,medium,high}` 중의 `effort`; OpenAI `reasoning_effort`; Gemini는 native 매핑이 없어 LIGHT/DEEP에 대해 현재 `CapabilityEvent`를 기록; local: system-prompt suffix. |
| `output_budget_bucket` | enum `SMALL / MEDIUM / LARGE` | 이산화된 `max_tokens`. | `1024 / 4096 / 16000`으로 해석. |
| `response_schema_mode` | enum `FREEFORM / JSON_OBJECT / STRICT_SCHEMA` | 응답을 얼마나 엄격하게 형태 제약할지. | Anthropic `messages.create` 대 `messages.parse(output_format=...)`; OpenAI `chat.completions.create` 대 `beta.chat.completions.parse(response_format=...)`; Gemini `generate_content`에 `response_mime_type=application/json`, strict mode에서는 `response_schema`. |
| `tool_policy_variant` | enum `NO_TOOLS / TOOL_OPTIONAL / TOOL_REQUIRED` | tool-use 정책. | 평범한 chat target에서는 no-op; tool 가능한 target에서는 `tool_choice` 파생. |

`MetaAxisSpace` 레코드는 특정 실행에서 각 enum의 어떤 값이 범위 안에 있는지 선언합니다; 멤버가 하나인 리스트는 그 축을 고정 값으로 잠급니다. `ResolvedPromptParams` 레코드는 searcher가 고른 뒤의 구체적 선택을 담습니다. 두 레코드 모두 `extra="forbid"`인 Pydantic model입니다 — 알 수 없는 key는 파싱 시점에 예외를 던집니다.

### 4.2 Execution profile

두 profile이 엄격함 대 탐색의 trade-off 위에서 실무자의 위치를 포착합니다.

```python
class ExecutionProfile(str, Enum):
    GUARDED   = "guarded"     # default; blocks hidden fallbacks
    EXPEDITION = "expedition" # permits recorded boundary crossing
```

strict mode(`guarded` profile):
- `supports_llm_judge` capability가 false인 provider를 judge로 쓰길 거부합니다.
- `STRICT_SCHEMA` 요청이 그것을 들어줄 수 없는 provider에 닿으면 예외를 던집니다.
- `experimental` 또는 `placeholder` adapter를 ship-grade 자리에 부적격으로 다룹니다.
- held-out gate의 `max_gap`과 `min_kc4`에 엄격한 기본값을 씁니다.

fast mode(`expedition` profile):
- 위를 허용하되, 모든 완화가 artifact에 `RelaxedSafeguard`로 기록되고 `stayed_within_guarded_boundaries`가 `False`로 설정됩니다.
- `additional_uplift_from_boundary_crossing`은 calibrated fitness 중 얼마가 strict mode라면 막았을 작업에서 왔는지를 기록하므로, 리뷰어는 경계 넘기가 실제로 값했는지 볼 수 있습니다.

profile 선택은 단일 CLI 플래그(`--profile guarded|expedition`)이며 artifact에 `selected_profile`로 나타납니다.

### 4.3 Provider capability model

모든 `LLMProvider`는 `capabilities() -> ProviderCapabilities` 메서드를 노출합니다:

```python
class ProviderCapabilities(BaseModel):
    provider: str
    tier: CapabilityTier                    # CORE / CLOUD / LOCAL
    supports_strict_schema: bool = False
    supports_json_object: bool = False
    supports_reasoning_profiles: bool = False
    supports_usage_accounting: bool = True
    supports_llm_judge: bool = False
    ship_grade_judge: bool = False
    supports_tools: bool = False
    experimental: bool = False
    placeholder: bool = False
    notes: list[str]
```

Capability *tier*는 거친 분류입니다:

| Tier | 목적 | 예 |
|---|---|---|
| `tier_1_core_parity` | Neutral contract와 calibration 커널. 필수. | In-memory 테스트 stub, 레거시 provider shim. |
| `tier_2_cloud_grade` | 일급 cloud provider; judge ship-grade는 여전히 별도로 선언됨. | Anthropic, OpenAI, Gemini. |
| `tier_3_local` | 로컬 OpenAI-compatible 백엔드. target-eligible; 기본적으로 ship-grade judge는 아님. | Ollama, vLLM, llama.cpp, 로컬 OpenAI-compatible 서버. |

tier는 정책 입력입니다: strict mode는 judge 자리에서 tier-3 provider를 거부합니다. fast mode는 허용하며 `RelaxedSafeguard`를 기록합니다.

### 4.4 Capability event

adapter가 런타임에 degrade하면 구조화된 레코드를 방출합니다:

```python
class CapabilityEvent(BaseModel):
    capability: str          # e.g. "reasoning_profile"
    requested: str           # "deep"
    applied: str             # "off"
    reason: str              # "endpoint rejected reasoning_effort"
    user_visible_note: str   # actionable English explanation
    affects_guarded_boundary: bool = True
```

이벤트는 `ProviderResponse`에서 `EvalItemResult`를 거쳐 `EvalResult.degraded_capabilities`로, 그리고 마지막으로 `CalibrationArtifact.degraded_capabilities`로 전파됩니다. artifact를 읽는 사람은 capability 이름을 grep해서 실행 중 어떤 기능이 들어지지 않았는지 볼 수 있습니다. strict mode에서는 `affects_guarded_boundary=True`인 이벤트가 실행을 막고; fast mode에서는 그저 기록만 합니다.

### 4.5 Ship recommendation

artifact의 `ship_recommendation` 필드는 다음 중 하나를 가집니다:

```python
class ShipRecommendation(str, Enum):
    SHIP = "ship"            # OK to deploy; all gates pass
    HOLD = "hold"             # do not deploy; at least one gate fails
    EXPERIMENT = "experiment" # opt-in expedition path; not a ship verdict
    BLOCK = "block"           # structural risk exceeds the current profile
```

계산은 `status`, held-out validation 결과, hard-gate 통과율, `stayed_within_guarded_boundaries`, 그리고 막는 `CapabilityEvent`의 존재 여부로부터 deterministic합니다. 같은 artifact가 들어가면 같은 recommendation이 나옵니다 — CI 파이프라인은 산문을 해석하지 않고 `SHIP`을 화이트리스트에 넣습니다. `omegaprompt diff`는 raw metric이 개선되더라도 후보의 `BLOCK`과 `HOLD`를 regression으로 다룹니다; `EXPERIMENT`는 설계상 non-blocking으로 남습니다.

---

## 5. calibration 파이프라인

### 5.1 입력

세 파일, 모두 사용자 작성, 모두 Pydantic으로 검증됩니다:

- **`dataset.jsonl`** — 한 줄당 하나의 `DatasetItem`: `id`, `input`, 선택적 `reference`, 선택적 `metadata`.
- **`rubric.json`** — 차원별 weight와 정수 scale을 가진 `JudgeRubric`, 그리고 각각 `evaluator`(`rule` / `judge` / `post`)로 라벨된 hard gate.
- **`variants.json`** — `system_prompts` 풀과 선택적 `few_shot_examples`를 가진 `PromptVariants`.

선택적으로: `space.json`(커스텀 `MetaAxisSpace`), `test.jsonl`(이긴 것을 다시 테스트하는 held-out 테스트 슬라이스).

### 5.2 sensitivity 측정

neutral-parameter baseline 주위에서, searcher는 각 meta-axis를 그 선언된 값들에 걸쳐 perturb하고 fitness 델타를 기록합니다. 축들은 fitness-델타 분포의 Gini 계수로 순위가 매겨집니다 — 높은 Gini = 집중됨, 고-leverage; 낮은 Gini = 흩어짐, 저-신호. sensitivity는 어떤 축에 검색 예산을 쓸지에 대한 *a priori* 근거입니다.

### 5.3 Top-K unlock

Gini 델타로 상위 `--unlock-k`개 축이 grid-search 부분공간에 들어갑니다. 나머지는 neutral 값에 잠긴 채 남습니다. 이것은 검색 비용을 모든 축에 대한 `Π(|axis|)`에서 top-K에 대한 `Π(|axis|)`로 줄입니다 — `k=3`에서 보통 5–20배 감소입니다.

### 5.4 grid search

unlock된 부분공간의 모든 조합이 평가됩니다. 각 평가는 dataset item당 하나의 provider 호출(target)에 더해, item당 하나의 judge 호출(judge가 `LLMJudge`이거나 LLM fallback이 발동된 `EnsembleJudge`인 경우)을 냅니다. 반환된 `EvalResult`는 fitness, per-item 점수, 집계 토큰 usage, latency, 그리고 모든 capability 이벤트를 기록합니다.

### 5.5 held-out replay

training-best parameter가 held-out 테스트 슬라이스에서 replay됩니다. replay는 *같은* `PromptTarget` adapter를 다른 dataset과 함께 씁니다; 테스트 슬라이스는 searcher가 결코 보지 않았으므로 leakage가 가능하지 않습니다.

### 5.6 held-out transfer gate (per-item 상관)

train per-item 점수와 test per-item 점수 사이의(공유 dataset id 위에서의) Pearson 상관이 `--min-kc4`와 비교됩니다. generalisation 격차 `|train - test| / |train|`이 `--max-gap`과 비교됩니다. 둘 중 하나라도 실패하면 `status = FAIL_KC4_GATE`, `ship_recommendation = HOLD`로 설정됩니다. 두 임계값 모두 artifact에 기록됩니다; 사후에 낮출 수 없습니다.

**validation_mode별 transfer-gate 의미(v1.5+).** transfer gate는 per-item 상관이므로, train과 test 슬라이스가 *같은* item id를 공유할 때 — "paired replay" — 에만 의미가 있습니다. 보통의 disjoint train/test split에서는 슬라이스에 공유 id가 없어 per-item 상관이 구조적으로 측정 불가능합니다; gate는 gap-only 검사로 축소됩니다. 이를 명시하기 위해 `CalibrateTuning.validation_mode`는 다음을 받습니다:

- `"auto"`(기본값, backward-compat): 슬라이스가 ≥3개 id를 공유할 때만 per-item 상관을 계산하고; 그렇지 않으면 조용히 건너뜁니다.
- `"paired"`: 호출자가 설계상 공유 id를 단언합니다. overlap이 3 미만이면 `ValueError`를 던집니다 — overlap 없는 paired 실행은 무료 통과가 아니라 setup 버그입니다.
- `"disjoint"`: 호출자가 설계상 공유 id 없음을 단언합니다. per-item 상관은 결코 계산되지 않고; gate는 gap-only입니다.

보통의 held-out split(disjoint id)을 돌린다면, `validation_mode="disjoint"`로 설정해 artifact에 `kc4_correlation`이 없는 것이 스스로 문서화되게 하세요. 두 prompt를 *같은* item에 점수 매기고 per-item 상관을 안정성 신호로 원할 때 `"paired"`를 쓰세요.

### 5.7 artifact 방출

`CalibrationArtifact`(§8 참고)는 `--output` 경로에 pretty-print된 JSON으로 작성됩니다. 그것은 (a) Markdown 리포트를 렌더링하고, (b) 이전 실행과 diff하고, (c) 산문을 파싱하지 않고 기계가 읽는 필드로 CI를 gate하기에 충분한 정보를 담습니다.

### 5.8 preflight와 adaptation (선택적 sub-tool 생태계)

메인 파이프라인은 자신의 기본 임계값(`min_kc4 = 0.5`, `max_gap = 0.25`, `unlock_k = 3`)이 보편적으로 옳다고 가정하지 않습니다. `omegaprompt.preflight`는, 실제 환경을 측정하고 메인 파이프라인이 소비하는 공유 :class:`AdaptationPlan`을 방출하는 두 개의 *선택적* 외부 sub-tool을 위한 안정적인 plugin 계약을 정의합니다. 원칙의 *방어책* — hard-gate fitness 붕괴, held-out ship gate, sensitivity 기반 축 unlock — 은 그대로 유지되고; 오직 수치 parameter만 환경에 맞춰 튜닝됩니다.

**단독 `omegaprompt`는 preflight probe 코드를 제공하지 않습니다.** 대부분의 사용자는 그것이 결코 필요 없습니다. preflight 모듈은 다음만 노출합니다:

- **Contract** — 외부 sub-tool이 방출하는 Pydantic 타입(`PreflightReport`, `AnalyticalFinding`, `JudgeQualityMeasurement`, `EndpointMeasurement`, `PerformanceMeasurement`).
- **Adaptation 로직** — `derive_adaptation_plan(report)`는 리포트를 `AdaptationPlan`으로 매핑하고; `apply_adaptation_plan(plan, ...)`은 그 plan을 호출자의 기본값에 대해 clip해서 adaptation이 원칙을 *강화만* 할 수 있게 합니다.

두 외부 sub-tool이 꽂힙니다:

| Sub-tool | Repository / PyPI | 역할 |
|---|---|---|
| **`mini-omega-lock`** | `pip install mini-omega-lock` (separate) | **Empirical preflight.** 라이브 judge + endpoint를 probe해서 consistency, schema reliability, context margin, latency, noise floor를 측정합니다. `JudgeQualityMeasurement`, `EndpointMeasurement`, `PerformanceMeasurement`를 방출합니다. |
| **`mini-antemortem-cli`** | `pip install mini-antemortem-cli` (separate) | **Analytical preflight.** 실행 구성을 읽고 calibration trap 패턴(self-agreement bias, 소표본 transfer-gate power, rubric 집중, variant 동질성, …)을 `REAL` / `GHOST` / `NEW` / `UNRESOLVED`로 분류합니다. `AnalyticalFinding` 레코드를 방출합니다. |

둘 중 하나만 단독으로 쓸 수도 있고; 둘 다 같은 `PreflightReport`로 조합됩니다.

sub-tool이 돌고 그 결과를 `derive_adaptation_plan`에 먹이면, 도출 규칙은 원칙을 *강화만 합니다*:

```
noise_floor >= 0.05  → min_kc4: max(default, 0.50)
noise_floor >= 0.15  → min_kc4: max(default, 0.60)
noise_floor >= 0.25  → min_kc4: max(default, 0.70)
noise_floor >= 0.35  → min_kc4: max(default, 0.80)

judge_consistency < 0.60  → rescore_count = 3 (median)
judge_consistency < 0.80  → rescore_count = 2
judge_consistency < 0.70  → judge_ensemble_shift = 0.40 (RuleJudge weight up)

schema_reliability < 0.90  → schema_mode_fallback = JSON_OBJECT

projected_wall_time > 4h and unlock_k > 1  → unlock_k -= 1

small_sample_kc4_power finding (HIGH) → max_gap: min(0.40, default * 1.6)
variants_homogeneous (REAL/NEW)       → skip_axes += ["system_prompt_variant"]
```

`apply_adaptation_plan(plan, min_kc4=..., max_gap=..., unlock_k=...)`는 `min_kc4`에 `max`를, `max_gap`에 `min`을, `unlock_k`에 `min`을 쓰므로, tolerance를 넓히려는 plan은 호출자의 구성으로 clip됩니다(부록 C invariant 10과 동반 테스트 `test_apply_plan_never_weakens_kc4` 참고).

단독 `omegaprompt`는 이 서브시스템 전체를 무시하고 선언된 기본값으로 돕니다. `mini-omega-lock` + `mini-antemortem-cli`로 보강된 calibration은 override가 artifact에 완전히 audit 가능한 plan을 만들고, 파이프라인은 약한 인프라에서 시끄럽게 실패하는 대신 원칙 안에서 adapt합니다.

---

## 6. 세 judge

```python
class Judge(Protocol):
    name: str

    def score(
        self,
        *,
        rubric: JudgeRubric,
        item: DatasetItem,
        target_response: str,
    ) -> tuple[JudgeResult, dict[str, int]]: ...
```

### 6.1 LLMJudge

`ResponseSchemaMode.STRICT_SCHEMA`를 통해 `LLMProvider`에 위임합니다. Anthropic은 `messages.parse`를, OpenAI는 `beta.chat.completions.parse`를 쓰고, Gemini는 로컬 Pydantic 검증 전에 Google GenAI SDK를 통해 `response_schema`를 요청합니다. regex fallback도, 산문에서 구조로의 추론도 없습니다. strict mode에서 `LLMJudge`는 `supports_llm_judge` capability가 false인 provider에서 돌기를 거부합니다; 별도의 run-risk 정책은 여전히 ship-grade가 아닌 judge를 strict-mode 경계 문제로 다룹니다.

### 6.2 RuleJudge

rubric에서 `evaluator="rule"`을 선언한 hard gate만 평가합니다. 각 rule은 deterministic Python 호출 가능 객체(`default_no_refusal()`, `default_non_empty()`, `json_object_check()`, `regex_check(name, pattern)`, 또는 사용자 제공 lambda)입니다. LLM 호출 없음, API 비용 0, 실행 간 재현 가능. 차원은 채점되지 않습니다; `RuleJudge`는 보통 단독으로 쓰기보다 `LLMJudge`와 조합됩니다.

### 6.3 EnsembleJudge

먼저 `RuleJudge`를 돕니다. rule gate 중 하나라도 실패하면, 결과는 rule gate 결과로 단락되고 LLM 호출이 없습니다. 모든 rule gate가 통과하면, 차원 채점과 judge-gate 평가를 위해 fallback judge(보통 `LLMJudge`)로 에스컬레이트합니다. 두 judge의 gate 결과는 반환 시 병합됩니다. 실제로 `EnsembleJudge`는 응답이 구조 gate에 얼마나 자주 실패하느냐에 따라 LLM-judge 비용의 약 0.5–0.9배를 회수합니다.

```python
from omegaprompt import EnsembleJudge, LLMJudge, RuleJudge, make_provider
from omegaprompt.judges.rule_judge import default_no_refusal, json_object_check

judge_provider = make_provider("anthropic")
rule = RuleJudge(checks=[default_no_refusal(), json_object_check("format_valid")])
llm = LLMJudge(provider=judge_provider)
judge = EnsembleJudge(rule_judge=rule, fallback=llm)
```

---

## 7. provider adapter

### 7.1 capability 선언

모든 adapter는 `ProviderCapabilities` 레코드를 선언합니다. 내장 adapter:

| Provider | Tier | Strict schema | JSON object | Reasoning | Ship-grade judge | 메모 |
|---|---|---|---|---|---|---|
| `anthropic` | cloud | yes | yes | yes | yes | `messages.parse` + 명시적 `cache_control`. |
| `openai` | cloud | yes | yes | yes | yes | `beta.chat.completions.parse`; 미지원 endpoint에서 `reasoning_effort`를 떨구고 이벤트를 기록. |
| `gemini` | cloud | yes | yes | no | no | Google GenAI `generate_content`; target/freeform/json-object와 `response_schema` + 로컬 Pydantic 검증을 통한 strict schema 지원. strict-mode judge로서 ship-grade로 표시되지 않음. |
| `ollama` / `local` / `vllm` / `llama_cpp` | local | best-effort | yes | no | no | target-eligible; strict mode에서 LLM-judge 자리 거부. |

### 7.2 Anthropic

freeform과 JSON-object 모드에는 `messages.create`(Anthropic이 native `response_format={"type":"json_object"}`를 노출하지 않으므로 JSON 출력을 지시하는 system-prompt suffix와 함께); `STRICT_SCHEMA`에는 `messages.parse(output_format=T)`. system 블록은 항상 `cache_control={"type":"ephemeral"}`로 감싸여서, calibration 실행 중 반복되는 judge 호출이 prompt cache를 칩니다. reasoning profile은 `thinking={"type":"adaptive"}`에 더해 `output_config.effort`로 매핑됩니다.

### 7.3 OpenAI 및 OpenAI-compatible

freeform에는 `chat.completions.create`, JSON 모드에는 `response_format={"type":"json_object"}`; `STRICT_SCHEMA`에는 `beta.chat.completions.parse(response_format=T)`. non-OFF reasoning profile에는 `reasoning_effort`가 시도되며; endpoint가 그 parameter를 거부하면(일부 compatible endpoint가 그렇습니다) adapter는 그것 없이 재시도하고 fallback을 명명하는 `CapabilityEvent`를 방출합니다. `base_url`을 받으므로 모든 OpenAI-compatible endpoint(Azure OpenAI, Groq, Together.ai, OpenRouter, 로컬 vLLM / Ollama)가 drop-in target이나 judge가 됩니다.

### 7.4 Gemini

공식 Google GenAI SDK(`google-genai`)를 씁니다. freeform과 JSON-object target 호출이 지원됩니다. `STRICT_SCHEMA`는 활성화되면 Gemini `response_schema`를 쓰고, 반환 전에 응답을 요청된 Pydantic model에 대해 여전히 검증합니다. native strict schema가 불가하면, strict mode는 degrade하는 대신 예외를 던지고; fast mode는 JSON-object 출력에 더해 로컬 Pydantic 검증으로 fallback하고 `CapabilityEvent`를 기록할 수 있습니다.

Gemini는 freeform과 JSON-object 실행에 target-eligible합니다, 특히 `reasoning_profile`을 `OFF`나 `STANDARD`로 잠근 경우에. `LIGHT`와 `DEEP` reasoning profile은 이 adapter가 그것들을 native Gemini 컨트롤로 매핑하지 않으므로 `CapabilityEvent`를 방출합니다. Gemini는 judge로 쓸 수 있지만 `ship_grade_judge=False`라서, strict-mode artifact를 그것만으로 ship-ready로 다뤄선 안 됩니다. Gemini judge 경로는 fast mode에서 쓰거나, capability 정책을 바꾸기 전에 judge 신뢰성을 독립적으로 검증하세요.

### 7.5 로컬 endpoint

로컬 OpenAI-compatible 백엔드는 일급 target provider이지만 기본적으로 ship-grade judge로 간주되지 않습니다. strict mode는 judge 자리에서의 사용을 막고; fast mode는 그 완화를 기록합니다. 이것은 라이브러리의 한계가 아니라 정책 입장입니다 — 당신의 도메인에서 ship-grade judge 품질을 입증하는 로컬 model은 capability override를 명시적으로 설정할 수 있습니다.

### 7.6 확장

```python
class LLMProvider(Protocol):
    name: str
    model: str
    def call(self, request: ProviderRequest) -> ProviderResponse: ...
    def capabilities(self) -> ProviderCapabilities: ...
```

두 메서드를 구현하세요. `providers/factory.py`에 등록하세요. 다른 건 아무것도 바뀌지 않습니다 — search 계층, judge, target, artifact schema 전부 그대로입니다.

---

## 8. CalibrationArtifact (schema v2.0)

schema는 일부러 풍부합니다. artifact는 그 실행의 기록 시스템이고, 리뷰어는 무엇도 다시 도출할 필요가 없어야 합니다.

```json
{
  "schema_version": "2.0",
  "engine_name": "omegaprompt",
  "method": "p1",
  "unlock_k": 3,
  "selected_profile": "guarded",

  "neutral_baseline_params": {
    "system_prompt_variant": 0,
    "few_shot_count": 0,
    "reasoning_profile": "standard",
    "output_budget_bucket": "medium"
  },
  "neutral_fitness": "<float>",

  "calibrated_params": {
    "system_prompt_variant": 2,
    "few_shot_count": 1,
    "reasoning_profile": "deep"
  },
  "calibrated_fitness": "<float>",

  "best_params": { "...": "mirror of calibrated_params for backward-compat" },
  "best_fitness": "<float>",

  "uplift_absolute": "<float>",
  "uplift_percent": "<float>",
  "quality_per_cost_neutral": "<float>",
  "quality_per_cost_best": "<float>",
  "quality_per_latency_neutral": "<float>",
  "quality_per_latency_best": "<float>",

  "walk_forward": {
    "train_best_fitness": "<float>",
    "test_fitness": "<float>",
    "generalization_gap": "<float>",
    "validation_mode": "auto | paired | disjoint",
    "shared_item_count": "<integer>",
    "kc4_status": "COMPUTED | MISSING_PER_ITEM_SCORES | ...",
    "kc4_correlation": "<float-or-null>",
    "max_gap_threshold": "<float>",
    "min_kc4_threshold": "<float-or-null>",
    "passed": true
  },

  "hard_gate_pass_rate": "<float>",
  "sensitivity_ranking": [
    { "axis": "system_prompt_variant", "gini_delta": "<float>", "rank": 0 },
    { "axis": "reasoning_profile",     "gini_delta": "<float>", "rank": 1 },
    { "axis": "few_shot_count",        "gini_delta": "<float>", "rank": 2 }
  ],

  "boundary_warnings": [],
  "degraded_capabilities": [],
  "relaxed_safeguards": [],
  "stayed_within_guarded_boundaries": true,
  "additional_uplift_from_boundary_crossing": "<float>",
  "guarded_boundary_crossed": false,

  "ship_recommendation": "ship",
  "status": "OK",
  "rationale": "passed",

  "target_provider": "openai",
  "target_model":    "gpt-4o",
  "target_capabilities": { "tier": "tier_2_cloud_grade", "supports_strict_schema": true, "ship_grade_judge": true, "...": "..." },
  "judge_provider": "anthropic",
  "judge_model":    "claude-opus-4-7",
  "judge_capabilities": { "tier": "tier_2_cloud_grade", "supports_llm_judge": true, "ship_grade_judge": true, "...": "..." },

  "usage_summary": { "input_tokens": "<integer>", "output_tokens": "<integer>", "cache_read_input_tokens": "<integer>" },
  "latency_summary_ms": { "target_p50": "<float>", "judge_p50": "<float>" },
  "cost_basis": "normalized_token_units",
  "n_candidates_evaluated": "<integer>",
  "total_api_calls": "<integer>"
}
```

핵심 구조적 선택: `neutral_baseline`과 `calibrated`가 명시적인 `uplift_absolute` / `uplift_percent` 필드와 함께 나란히 기록됩니다. 정확한 값은 생성된 artifact나 claim ledger에 속합니다; 위 스니펫은 벤치마크 성능이 아니라 형태를 문서화합니다.

---

## 9. CLI 표면

### 9.0 Exit-code 계약

모든 CLI 명령은 같은 계약을 씁니다:

- `0` — 명령이 완료되었고 요청된 gate/실패 조건이 발동하지 않음.
- `1` — CI gate 실패: calibration status가 non-OK, transfer-gate/hard-gate 실패, gate 모드에서 `ship_recommendation`이 `hold` 또는 `block`, artifact regression, 또는 `check-artifact --strict`가 무결성 오류 발견.
- `2` — environment/config/tooling/input 문제: provider env var 누락, 알 수 없는 provider, 의존성 누락, 잘못된 CLI 인자, 읽을 수 없는 파일, 또는 non-checker 명령에 전달된 잘못된 artifact. 해당하는 경우 stderr는 `TOOLING_MISSING`, `ENVIRONMENT_BLOCKED`, `INVALID_ARTIFACT` 같은 명시적 prefix를 씁니다.

### 9.1 `omegaprompt calibrate`

End-to-end 실행: 입력 파싱, target + judge 빌드, `omega_lock.run_p1` 호출, `CalibrationArtifact` 방출.

```bash
omegaprompt calibrate train.jsonl \
  --rubric rubric.json \
  --variants variants.json \
  --test test.jsonl \
  --profile guarded \
  --target-provider openai   --target-model gpt-4o \
  --judge-provider anthropic --judge-model claude-opus-4-7 \
  --method p1 --unlock-k 3 \
  --output artifact.json
```

기본 동작은 CI-gate 지향입니다. Exit code: artifact가 CI-clean일 때(`status == OK`이고 `ship_recommendation`이 `hold`/`block`이 아닐 때) `0`, status가 non-OK이거나 `ship_recommendation`이 `hold`/`block`일 때 `1`, 그리고 provider 자격 증명 누락·알 수 없는 provider·의존성 누락 같은 CLI 인자·environment·tooling 문제에 `2`. 여전히 artifact를 쓰고 출력하는 advisory 로컬 실행을 원할 때만 `--no-fail-on-gate`를 쓰세요.

### 9.2 `omegaprompt report <artifact.json>`

유효한 artifact를 Markdown으로 렌더링합니다(PR 설명, CI step 출력, 사람 리뷰용). ship/no-ship 결정을 내리지 않습니다: 유효한 artifact는 `0`으로 종료; 잘못된 artifact 입력은 `2`로 종료.

```bash
omegaprompt report artifact.json > report.md
```

### 9.3 `omegaprompt diff <old.json> <new.json>`

두 artifact를 비교합니다. 새 실행이 다음 중 어느 하나라도 regression하면 `1`로 종료: `calibrated_fitness`, `walk_forward.test_fitness`, `hard_gate_pass_rate`, `quality_per_cost_best`, `quality_per_latency_best`, `stayed_within_guarded_boundaries`(true→false가 regression), non-OK status, 또는 `{hold, block}` 안의 `ship_recommendation`. 잘못된 artifact 입력은 `2`로 종료. CI용으로 의도됨; `--no-fail-on-regression`은 regression을 출력하지만 `0`으로 종료.

```bash
omegaprompt diff previous.json artifact.json   # exit 1 on regression
```

### 9.4 `omegaprompt check-artifact <artifact.json>`

network 없는 artifact 무결성 검사기를 돕니다. `--strict` 없이는 findings를 보고하고 파일이 접근 불가가 아닌 한 `0`으로 종료. `--strict`로는 무결성 오류가 `1`로 종료; 접근 불가 파일 / environment-blocked 읽기는 `2`로 종료. `--json`은 기계가 읽는 CI 결과를 방출.

`omegacal` CLI 바이너리는 마이그레이션 동안 호환 alias로 남습니다. `omegaprompt`가 primary CLI이자 PyPI 배포 이름입니다; `omegacal`은 PyPI 배포 정체성이 아닙니다.

---

## 10. 빠른 시작

```bash
pip install omegaprompt              # core (CLI + runtime entrypoints)
pip install "omegaprompt[mcp]"       # adds the MCP server for Claude Code / Cursor
```

omegaprompt는 같은 calibration 커널 위에 세 개의 호출 표면을 노출합니다: **Python high-level API**(연산당 한 호출, agent-callable), **CLI**(사람 주도, 어떤 shell에서든 스크립팅 가능), 그리고 **MCP 서버**(Claude Code, Cursor, 그리고 다른 MCP 클라이언트가 같은 연산을 도구로 호출).

### 10.1 Python (high-level API)

`omegaprompt.runtime`의 여덟 개 한-호출 entrypoint, 패키지 수준에서 re-export됨. 각각은 경로나 inline 객체를 받고 Pydantic-modeled 결과를 반환합니다.

```python
from omegaprompt import calibrate, evaluate, diff, report

artifact = calibrate(
    train="train.jsonl",
    test="test.jsonl",
    rubric="rubric.json",
    variants="variants.json",
    target="anthropic",                  # or {"name": "openai", "model": "gpt-4o"}
    judge="openai",                      # cross-vendor breaks self-agreement
    output="artifact.json",              # opt-in disk write
)
print(artifact.status, artifact.calibrated_fitness)

# Re-score the same config on a fresh dataset (regression check).
result = evaluate(
    dataset="canary.jsonl",
    rubric="rubric.json", variants="variants.json",
    params=artifact,                     # extracts artifact.calibrated_params automatically
    target="anthropic", judge="openai",
)

# Compare two artifacts (CI regression detection).
delta = diff("baseline.json", "candidate.json")
if delta.regressed:
    raise SystemExit("\n".join(delta.regression_reasons))

# Render for a PR description.
print(report("artifact.json"))
```

네 개의 Tier 2 entrypoint — `measure_sensitivity`, `grade`, `preflight`, `classify_traps` — 는 진단 및 응답별 사용 사례를 다룹니다. 전체 표면은 `omegaprompt/runtime.py` docstring을 참고하세요.

짚어둘 만한 세 가지 입력-강제 변환 편의:

- `target`과 `judge`는 문자열(`"anthropic"`), `ProviderSpec` dict(`{"name": ..., "model": ..., "base_url": ...}`), 또는 미리 빌드된 `LLMProvider` 인스턴스를 받습니다. 기본값에는 문자열 형태를; non-default model이나 로컬 endpoint에는 dict 형태를; 이미 caching, retry 등을 구성해 둔 경우엔 인스턴스 형태를 쓰세요.
- Dataset, rubric, variant는 파일시스템 경로나 in-memory Pydantic / dict 인스턴스 중 하나를 받습니다. agent는 보통 dict를, 사람은 보통 경로를 넘깁니다.
- `evaluate()`의 `params=`는 `CalibrationArtifact`를 직접 받습니다. 흔한 "이전 best를 새 dataset에 평가" 흐름은 한 호출입니다.

저빈도 knob(검색 method, unlock-K, held-out gate 임계값, 축 공간)은 flat parameter가 아니라 `tuning=CalibrateTuning(...)` 아래 묶여 있습니다; agent 표면은 최소로 유지되고 power user는 완전한 제어를 유지합니다.

### 10.2 CLI

```bash
# Anthropic target + Anthropic judge, guarded profile.
omegaprompt calibrate examples/sample_dataset.jsonl \
  --rubric examples/rubric_example.json \
  --variants examples/variants_example.json \
  --target-provider anthropic \
  --judge-provider anthropic \
  --profile guarded \
  --output artifact.json

# Cross-vendor (OpenAI target, Anthropic judge) to break self-agreement.
omegaprompt calibrate train.jsonl \
  --rubric rubric.json --variants variants.json --test test.jsonl \
  --target-provider openai   --target-model gpt-4o \
  --judge-provider anthropic --judge-model claude-opus-4-7 \
  --output artifact.json

# Local target (Ollama) + cloud judge.
omegaprompt calibrate train.jsonl \
  --rubric rubric.json --variants variants.json --test test.jsonl \
  --target-provider ollama \
    --target-base-url http://localhost:11434/v1 \
    --target-model llama3.1:8b \
  --judge-provider openai --judge-model gpt-4o \
  --profile guarded \
  --output artifact.json

# Gemini target + OpenAI judge.
export GEMINI_API_KEY=...
omegaprompt calibrate examples/sample_dataset.jsonl \
  --rubric examples/rubric_example.json \
  --variants examples/variants_example.json \
  --target-provider gemini \
  --target-model gemini-2.5-flash \
  --judge-provider openai \
  --output artifact.json

# Gemini judge path: explicit expedition profile.
omegaprompt calibrate examples/sample_dataset.jsonl \
  --rubric examples/rubric_example.json \
  --variants examples/variants_example.json \
  --target-provider anthropic \
  --judge-provider gemini \
  --judge-model gemini-2.5-flash \
  --profile expedition \
  --output artifact.json

# Render and diff.
omegaprompt report artifact.json
omegaprompt diff previous.json artifact.json
```

### 10.3 MCP 서버 (Claude Code, Cursor)

MCP 서버는 여덟 개의 runtime entrypoint 전부를 agent-callable 도구로 노출합니다. 입력은 JSON-친화적(경로, dict, primitive)이고; 출력은 dict로 직렬화된 Pydantic model입니다. schema는 타입 힌트에서 자동 도출됩니다.

서버를 돌리세요(stdio가 기본; Claude Code가 서브프로세스로 spawn합니다):

```bash
pip install "omegaprompt[mcp]"
python -m omegaprompt.mcp           # stdio transport (recommended)
python -m omegaprompt.mcp --http    # streamable-http transport
```

Claude Code의 `mcpServers` 구성에 연결하세요:

```json
{
  "mcpServers": {
    "omegaprompt": {
      "command": "python",
      "args": ["-m", "omegaprompt.mcp"]
    }
  }
}
```

서버가 연결되면, agent는 여덟 개 도구 중 어느 것이든 이름으로(`calibrate`, `evaluate`, `report`, `diff`, `measure_sensitivity`, `grade`, `preflight`, `classify_traps`) 문서화된 인자 형태로 호출할 수 있습니다. prompt를 ship하기 전의 전형적인 agent 흐름:

```
classify_traps(...)   →  catch self-agreement bias, small-sample power, etc.
preflight(...)         →  verify provider tiers, surface vendor-vendor warnings
calibrate(...)         →  run the full pipeline; produce a CalibrationArtifact
report(...)            →  render the artifact as Markdown for the user
diff(prev, new)        →  CI regression check on the next iteration
```

MCP 서버는 이 README의 나머지가 문서화하는 기반입니다 — 모든 도구는 runtime entrypoint를 감싸는 얇은 wrapper이고, 그것은 다시 calibration 커널을 감싸는 얇은 wrapper입니다.

표준적인 *언제* — toolkit 안 네 개의 MCP 서버(omegaprompt, antemortem, mini-omega-lock, mini-antemortem-cli, 총 18개 도구)에 걸쳐 어떤 트리거가 어떤 도구 흐름을 발동하는지 — 는 [AGENT_TRIGGERS.md](AGENT_TRIGGERS.md)를 참고하세요. 그 문서는 실제 agent 상황(non-trivial한 코드 변경, prompt PR, production regression 의심, agent가 자기 출력을 자가 검증, 등)을 권장되는 비용-최소화 도구 시퀀스로 매핑합니다.

---

## 11. 풀어본 예시

### 11.1 도구가 잡는 실패 모드 (예시)

실무자가 10-item training set에 prompt를 반복하며 두 후보를 비교합니다.

```
Candidate A:  train_fitness = 0.923   (4.6 / 5 average)
Candidate B:  train_fitness = 0.876   (4.4 / 5 average)
```

Candidate A가 training에서 이깁니다. 실무자는 A를 ship합니다.

searcher가 결코 보지 않은 테스트 슬라이스에서의 held-out validation:

```
Candidate A:  train = 0.923   test = 0.612   gap = 33.7%
Candidate B:  train = 0.876   test = 0.841   gap =  4.0%

run_p1 status: FAIL_KC4_GATE
Reason: pearson(train_per_item, test_per_item) < --min-kc4
Ship recommendation: HOLD.
Rationale: candidate A's per-item train ranking does not predict its per-item test ranking;
           the calibration signal did not generalise.
```

Candidate A는 training 슬라이스에 overfit했습니다. `omegaprompt`는 실무자가 production 동작을 보기 전에 ship 결정을 기계적으로 막습니다. Candidate B — 더 낮은 training 점수, 극적으로 더 나은 일반화 — 가 올바른 결정입니다.

artifact는 두 후보 모두를 grid 히스토리에, 실패한 held-out transfer gate를, 그리고 `ship_recommendation: "hold"`를 기록합니다. `stayed_within_guarded_boundaries == true`와 `ship_recommendation == "ship"`에 대한 CI gate는 실무자가 산문을 파싱할 필요 없이 merge를 막습니다.

> 이 소절의 숫자는 실패 모드를 보여주기 위한 예시입니다. 재현 가능한, 기계가 생성한 예시는 §11.2에 있습니다.

### 11.2 재현 가능한 reference 실행

repository는 deterministic reference 실행(`examples/reference/reproduce_reference_artifact.py`)을 제공합니다. 그것은 fitness가 meta-axis parameter의 closed-form 함수인 in-memory target + judge를 상대로 **진짜** `omega_lock.run_p1`을 구동합니다. LLM API 호출 없음, network 없음, randomness 없음. 같은 실행이 모든 머신에서 byte-identical 출력을 만듭니다.

offline golden harness를 재현하고 검사하려면:

```bash
python examples/reference/reproduce_reference_artifact.py
python tools/reproduce_golden_reference.py --check
omegaprompt check-artifact examples/reference/reference_artifact.json --strict
```

harness는 `examples/reference/golden_manifest.json`을 쓰는데, 그것은 각 case id, 재현 명령, 기대 status, 기대 ship recommendation, 기대 validation mode, 무결성 분류, 정규화된 artifact hash, 그리고 정확한 metric을 표시해도 되는지를 기록합니다. 다루는 것:

- `clean_ok_ship` — deterministic target과 judge stub을 가진 진짜 `omega_lock.run_p1` 경로.
- `fail_kc4_gate` — per-item transfer 상관이 사전 선언 임계값 아래인 paired held-out validation.
- `fail_hard_gates` — BLOCK recommendation을 동반한 hard-gate 실패.
- `provider_degradation` — 명시적 capability degradation과 relaxed safeguard 가시성.
- `diff_regression_candidate` — 개별적으로는 유효하지만 `omegaprompt diff` 아래에서 clean baseline 대비 regression하는 artifact.

clean case에서 진짜 calibration 엔진은 mock되지 않습니다. `omega_lock.run_p1`은 production 검색 로직으로 돌고; 출력이 API 접근 없이 재현 가능하도록 오직 *target*과 *judge* 계층만 deterministic stub입니다. 정확한 값은 손으로 관리하는 README 산문이 아니라 생성된 artifact, golden manifest, 또는 claim ledger에 속합니다.

### 11.3 약한 인프라 구성에서의 preflight + adaptation

두 번째 재현성 스크립트는 일부러 약한 구성 — 작은 dataset, 같은 vendor의 target과 judge, 동질적 variant, 집중된 rubric, 잡음 많은 judge, 신뢰할 수 없는 schema 파싱, 긴 예상 wall time — 에서 preflight와 adaptation 계층을 돌립니다. Deterministic; API 호출 없음.

```bash
python examples/reference/reproduce_preflight_demo.py
# writes examples/reference/reference_preflight_report.json
# writes examples/reference/reference_adaptation_plan.json
```

Analytical findings (trap 일곱 개):

```
[REAL       high   ] self_agreement_bias
    Target and judge are identical: openai/gpt-4o-mini. Judge will share the target's distributional biases.
[REAL       high   ] small_sample_kc4_power
    Test slice has 5 items. Pearson correlation at n=5 has weak statistical power; KC-4 pass/fail may be random.
[NEW        medium ] variants_homogeneous
    All 2 system prompts have near-identical length (21-29 chars); they may be too similar to produce meaningful sensitivity.
[REAL       medium ] rubric_weight_concentration
    Dimension 'accuracy' carries 85% of the rubric weight; judge noise on that single dimension will dominate fitness.
[GHOST      low    ] judge_budget_too_small
[NEW        low    ] empty_reference_with_strict_rubric
[GHOST      low    ] no_held_out_slice
```

Empirical 측정값(adaptation 로직에 먹이는 시뮬레이션 값):

```
judge.consistency           = 0.55
judge.anchoring_usage       = 0.40
endpoint.schema_reliability = 0.67
endpoint.context_margin     = 0.35
perf.projected_wall_time    = 5.2h
perf.noise_floor            = 0.180
```

그 결과 `AdaptationPlan`은 여섯 개의 override에 더해 한-축 sensitivity skip을 담습니다:

| parameter | default | applied | reason |
|---|---|---|---|
| `min_kc4` | 0.5 | 0.6 | empirical noise floor 0.180 requires stronger Pearson |
| `max_gap` | 0.25 | 0.4 | small-sample test slice widens acceptable gap |
| `rescore_count` | 1 | 3 | judge consistency 0.55 < 0.60 — take median of 3 |
| `schema_mode` | strict_schema | json_object | STRICT_SCHEMA reliability 67% below 90% — fallback with post-parse validation |
| `judge_ensemble_shift` | 0.0 | 0.40 | judge consistency 0.55 — raise RuleJudge weight to 40% |
| `unlock_k` | 3 | 2 | projected wall-time 5.2h exceeds 4h — reduce unlock_k |
| `skip_axes` | `[]` | `["system_prompt_variant"]` | variants homogeneous finding |
| `preserves_discipline` | `True` | `True` | invariant never toggled off |

모든 override는 *더 엄격한 쪽으로 monotonic*합니다: `min_kc4`는 오르기만 하고, `max_gap`은 작은 표본이 실제로 가진 분산을 수용할 만큼만 오르며, `unlock_k`는 내려가기만 하고, `rescore_count`는 오르기만 합니다. 호출자의 기본값보다 약한 임계값을 시도한 plan은 `apply_adaptation_plan` 시점에 clip됩니다(부록 C invariant 10). 그 결과 실행은 여전히 유효한 calibration입니다 — 다만 frontier-tier judge가 큰 dataset에 있다는 기본 가정이 아니라 인프라의 실제 noise floor에 맞춰 튜닝된 것일 뿐입니다.

---

## 12. 검증

기본 test suite는 `pytest -q`로 돌며 라이브 provider/API 호출 대신 mock된 provider 클라이언트를 씁니다. 정확한 test 개수는 의도적으로 README 산문에서 반복하지 않습니다; static badge row는 별도로 보존되며 consistency checker로 보호됩니다. adapter 테스트는 SDK 클라이언트 대신 `SimpleNamespace`나 `MagicMock`을 쓰고 나가는 요청 payload 형태(model, messages, cache 헤더, `response_format`, reasoning 지시, few-shot 순서)를 단언합니다. sub-tool repository `mini-omega-lock`과 `mini-antemortem-cli`는 각각 probe 실행과 analytical trap 분류를 다루는 자체 test suite를 갖습니다.

| 모듈 | 커버리지 요약 |
|---|---|
| `domain/` | `PromptVariants` / `MetaAxisSpace` / `CalibrationArtifact` / `Dataset` / enum / profile — 필수 필드, 범위 검증, JSON roundtrip, ordinal clamping, compat-key 매핑, `best_params`와 `calibrated_params` 사이의 `model_post_init` 동기화. |
| `core/fitness` | empty / all-pass / partial-fail / all-fail 배치에 걸친 `CompositeFitness`; 리포팅을 위해 per-item 분해가 보존됨. |
| `core/walkforward` | 공유 id 위의 Pearson 상관; zero-variance skip (kc4=None); 격차 산술; 두 임계값 모두에 대한 gate 통과/실패 로직. |
| `core/sensitivity` | Gini-계수 순위; top-K unlock; 엣지 케이스(zero-delta 축, single-point probe). |
| `core/artifact` | 디스크 JSON을 통한 round-trip; load 시 `model_post_init` invariant. |
| `core/profiles` | `policy_for(GUARDED/EXPEDITION)`이 구별되는 기본값을 반환; `relaxed_safeguards_for(...)`가 경계 넘기를 보고. |
| `core/risk` | OK / transfer-gate 실패 / hard-gate 실패 / capability-event 시나리오에 걸친 `assess_run_risk(...)`. |
| `providers/` | factory가 알 수 없는 이름을 거부; `base_url`을 존중; `anthropic` / `openai` / `gemini` / `ollama`를 나열. Anthropic adapter: reasoning 활성 시 freeform이 thinking config와 함께 `messages.create`를 쓰고 OFF면 생략; strict schema는 `messages.parse`를 씀; refusal은 예외; JSON-object 모드는 system-prompt suffix를 추가. OpenAI adapter: `chat.completions.create` / `beta.chat.completions.parse`에서 같은 경로; `reasoning_effort` 거부-재시도가 `CapabilityEvent`를 기록; `prompt_tokens_details.cached_tokens`가 `cache_read_input_tokens`로 정규화; content-filter finish reason은 예외; `parsed` 누락은 예외. Gemini adapter: freeform/json-object/strict-schema 요청 형태, strict-mode no-degrade 규칙, fast-mode JSON fallback과 `CapabilityEvent`, malformed/schema-invalid JSON 실패, 그리고 usage 매핑. `ollama` alias는 tier `tier_3_local`, `supports_llm_judge=False`, `experimental=True`를 보고. |
| `judges/` | `RuleJudge` (no_refusal / non_empty / json_object / regex / 중복-검사 거부 / 누락-검사 예외); `LLMJudge` (strict-schema dispatch, payload 구성, non-`JudgeResult` 응답 예외, strict-mode ship-grade judge 거부); `EnsembleJudge` (rule-first 단락, rule-pass 시 LLM 에스컬레이션, 병합된 gate_results, non-`RuleJudge` 거부). |
| `targets/` | mock된 provider + judge로 end-to-end `PromptTarget`; 범위 밖 입력에 대한 meta-axis 해석과 clamping; 평가 간 usage 누적; `evaluation_history` 보존; latency 측정; degraded-capability 전파. |
| `commands/` | CLI help가 `calibrate` / `report` / `diff` / `check-artifact`를 나열; `--version`이 `omegaprompt`를 표시; `report`가 schema-v2.0 artifact를 렌더링; `diff`가 fitness, cost 비율, latency 비율, 경계 넘기 플립에서 regression을 탐지; `check-artifact`가 network 없는 무결성 검사를 수행. |
| `preflight/` | **Plugin 인터페이스만** — `omegaprompt` 안에는 probe나 classifier 코드가 없음. `contracts`: severity 순서, status enum, `PreflightReport.worst_severity` / `any_real_or_new`, Pydantic `extra="forbid"` 강제; `JudgeQualityMeasurement.consistency`(0..1), `EndpointMeasurement.schema_reliability`(0..1) 경계. `adaptation`: 네 임계값에 걸친 noise-adaptive `min_kc4`; consistency 기반 `rescore_count`; schema-fallback 트리거; wall-time 기반 `unlock_k` 감소; small-sample 격차 확대; variant-skip 축 표시; `apply_adaptation_plan` invariant(`min_kc4`를 결코 약화시키지 않음, `max_gap`을 결코 넓히지 않음, `unlock_k`를 결코 올리지 않음). Sub-tool probe + classifier 구현(각자의 test suite를 가짐)은 `mini-omega-lock`과 `mini-antemortem-cli` repository에 삽니다. |
| `test_calibrate_integration.py` | deterministic in-memory `CalibrableTarget`로 **진짜 `omega_lock.run_p1`을 구동**(search engine에는 mock 없음). artifact의 `calibrated_params`, `neutral_baseline_params`, `walk_forward`, `sensitivity_ranking`이 `P1Result`의 실제 형태와 일치하는지 단언 — 이 테스트가 잡는 regression은 per-module 단위 테스트가 닿을 수 없는, adapter 계층과 search engine 사이의 drift입니다. |

기본 no-network suite는 `uv run pytest -q -m "not live"`(또는 `python -m pytest -q -m "not live"`)로 돌리세요. wall-clock 시간은 첫 import 시의 Pydantic model 컴파일이 지배합니다.

---

## 13. 비교 포지셔닝

| 접근 | 잘하는 것 | `omegaprompt`가 더하는 것 |
|---|---|---|
| **promptfoo** | assertion 기반 채점으로 테스트 케이스에 prompt를 실행. | 사전 선언 held-out gate, sensitivity-순위 축 unlock, `hard_gate × soft_score` fitness, 기계가 읽고 diff 가능한 artifact. 조합 가능 — promptfoo 스타일 assertion이 `RuleJudge` 검사로 꽂힘. |
| **DSPy** | 프로그램 추상화 + bootstrapped few-shot을 통한 prompt 합성. | 직교하는 관심사. DSPy는 후보 prompt를 *생산*하고; `omegaprompt`는 held-out validation 뒤에 *어느 것을 ship할지 결정*. DSPy 출력은 `PromptVariants`의 `system_prompts` 항목으로 꽂힘. |
| **prompt에 적용한 Optuna / Ray Tune** | 범용 hyperparameter 최적화. | 즉시 쓸 수 있는 held-out ship gate와 사전 선언 kill criteria; 각 vendor의 native parse 경로를 통한 schema-강제 LLM-as-judge; provider-neutral meta-axes; CI가 diff할 수 있는 명시적 `CalibrationArtifact` schema. |
| **Provider-native 평가 대시보드** | 한 vendor의 콘솔 안에서의 rubric 기반 채점. | cross-vendor 채점(self-agreement bias 깨기); vendor 로그인이 필요 없는 로컬 artifact; regression 탐지를 위한 deterministic `diff`; 통제된 경계 넘기를 위한 fast mode. |
| **직접 만든 eval 스크립트** | 단일 워크로드에 빠르게 작성 가능. | 구조화된 데이터 계약(`Dataset` / `JudgeRubric` / `PromptVariants` / `CalibrationArtifact`); capability-tier 정책; 사후에 낮출 수 없는 사전 선언 gate; 맞춤 glue 없는 CI 통합. |

고유한 셀링 포인트는 *검색보다 원칙*입니다. search engine은 [`omega-lock`](https://github.com/hibou04-ops/omega-lock)으로, 이 prompt adapter가 작성되기 전에 다른 도메인(quantitative trading의 parameter calibration)에 대해 ship되고 검증되었습니다. `omegaprompt`는 prompt-specific adapter, 세 provider-neutral judge, hard-gates-first fitness 형태, 그리고 capability / profile / artifact architecture를 기여합니다.

---

## 14. 제한 사항과 범위 경계

### safety evaluator가 아님

`no_safety_violation`을 hard gate로 선언할 수 있지만, judge는 훈련된 safety classifier가 아니라 rubric으로 채점되는 LLM입니다. 규제되는 safety 평가에는 `omegaprompt`를 전용 safety eval suite(AILuminate, HELM, vendor-specific red-team harness)와 짝지으세요.

### production 텔레메트리의 대체물이 아님

큐레이션된 dataset에서의 offline calibration은 값싼 스크리닝 단계입니다. 비즈니스 metric을 가진 실-트래픽 A/B가 여전히 ground truth입니다. `omegaprompt`는 offline 단계를 원칙 있게 만들 뿐, online 평가를 대체하지 않습니다.

### vendor capability의 벤치마크가 아님

fitness 숫자는 *당신이 구성한 model 위에서 당신의 dataset에 적용한 당신의 rubric*을 반영합니다. 그것은 절대적 model capability의 벤치마크가 아니며 재검증 없이 다른 도메인으로 이식되지 않습니다.

### judge drift는 실재하는 우려

LLM judge의 채점 분포는 model 릴리스 간에 drift할 수 있습니다. 계획된 multi-judge validation 패턴(top-K에서의 `judge_v1` 대 `judge_v2`)은 불일치를 조용한 실패가 아니라 신뢰 신호로 다룹니다.

### 비용이 사소하지 않음

전형적인 실행(10-item dataset, 125-candidate grid, held-out validation)을 frontier-tier cloud provider에서 돌리면 수십 달러가 듭니다. 완화책: 반복 중에는 더 싼 judge, 5분 윈도 안의 prompt-cache-aware 스케줄링, 품질이 허용하면 Ollama를 통한 로컬 target.

### 모든 provider가 ship-grade judge는 아님

strict mode는 정책상 judge 자리에서 로컬 provider를 막습니다. Gemini는 구현되어 있고 `JudgeResult`를 검증할 수 있지만, 이 adapter에서 strict-mode judge로 ship-grade로 표시되지 않습니다. target으로는 자유롭게 쓰고; judge로는 fast mode에서 쓰거나, 독립적 도메인 검증과 의도적인 capability 정책 변경 뒤에 쓰세요.

---

## 15. 로드맵

**Shipped (v1.0)**
- Provider-neutral meta-axes (`reasoning_profile`, `output_budget_bucket`, `response_schema_mode`, `tool_policy_variant`).
- 통합 `LLMProvider.call(ProviderRequest) -> ProviderResponse` + `capabilities()` 계약.
- `ExecutionProfile` (strict mode / fast mode) + 구조적 위험 보고.
- `CalibrationArtifact` schema v2.0 (neutral baseline 대 calibrated, capability event, boundary warning, ship recommendation).
- `RuleJudge` / `LLMJudge` / `EnsembleJudge`.
- Native `gemini` adapter + `ollama` / `local` / `vllm` / `llama_cpp` 로컬 adapter 패밀리.
- CLI: `calibrate` / `report` / `diff` / `check-artifact`. Backward-compat `omegacal` alias.
- 진짜 `omega_lock.run_p1`에 대한 integration 테스트.

**Planned: judge trust + tooling depth**
- Multi-judge validation 패턴: top-K에 걸친 `judge_v1` + `judge_v2`; 불일치를 일급 신뢰 신호로.
- 실행 전 비용 추정을 동반한 `--dry-run`.
- 추가 rule-gate predicate(언어 탐지, 길이 경계, 제공된 JSON Schema에 대한 schema 검증).

**Planned: ecosystem**
- 벤치마크 harness: 다중 (task × rubric × seed) scorecard.
- `omegaprompt diff`를 통한 CI regression gating용 GitHub Action.
- HTML 리포트 렌더링 (`omegaprompt report --format html`).
- Native HuggingFace Inference adapter.

**Explicitly out of scope**
- 호스팅된 대시보드, DB 기반 히스토리, 멀티테넌트 서비스. `omegaprompt`는 로컬 개발자 도구입니다. 로컬로 유지하세요.

전체 changelog: [CHANGELOG.md](CHANGELOG.md).

---

## 16. 선행 연구와 크레딧

- **사전 선언 gate를 가진 train / test split.** overfitting에 대한 토대가 되는 ML 방어책으로, 모든 학부 커리큘럼에 문서화되어 있습니다. 여기서의 구체적 구현(Pearson 순위 상관 임계값, 사전 선언되고 수정 불가)이 held-out transfer gate입니다 — `omega-lock` 엔진이 내부적으로 `KC-4`라 부르는 kill criterion.
- **LLM-as-judge.** *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*(Zheng et al., 2023)에서 형식화된 패턴. `omegaprompt`는 이 패턴을 SDK 경계에서의 schema 강제(각 vendor의 native parse 경로를 통한 Pydantic)와 함께 구현해서, malformed judge 응답이 fitness를 오염시키기 전에 예외를 던지게 합니다.
- **Winchester defence.** quant-finance 원칙: *실행 전에 선언된 kill criteria는 실행 후에 완화될 수 없다.* 여기서는 `--max-gap`과 `--min-kc4`가 점수를 들여다본 뒤 소급적으로 튜닝되는 게 아니라 구성에서 강제되어야 한다는 주장을 위해 쓰입니다.
- **Sensitivity 기반 coordinate descent.** stress 측정과 top-K unlock은 `omega-lock`(v0.3.0)이 도입한 parameter-calibration 원시 요소로, 원래 trading-strategy calibration을 위한 것이었으며 여기서는 prompt 구성으로 이식되었습니다.
- **Antemortem 원칙.** 이 프로젝트가 설계되고 빌드된 pre-implementation reconnaissance 방법론. 모든 non-trivial한 변경은 첫 키 입력 전에 [`antemortem-cli`](https://github.com/hibou04-ops/antemortem-cli)를 거칩니다. [방법론 repository](https://github.com/hibou04-ops/Antemortem)의 사례 연구가 이 코드베이스에 대한 recon을 기록합니다.

이름: *omega-lock*(parameter calibration) → *omegaprompt*(prompt calibration). 패밀리 브랜딩은 의도적입니다. `omega-lock`은 `KC-4 FAIL`로 끝난 trading-strategy calibration에서 추출되었습니다 — overfitting 방어책이 설계된 그대로 정확히 발동한 것. `omegaprompt`는 같은 방어책을 한 계층 위에 적용한 것이고, sub-tool `mini-omega-lock` / `mini-antemortem-cli`는 그 패턴을 preflight 측정으로 확장합니다.

---

## Troubleshooting

### `omegaprompt calibrate`가 "Incorrect API key" / 401을 반환

provider SDK가 key를 받았지만, 그 key가 유효하지 않았습니다. 흔한 두 경우:

- **만료되었거나 폐기된 key.** 발급 대시보드(Anthropic / OpenAI / Google AI Studio)를 확인하세요. 회전하고 다시 export하세요.
- **잘못된 env var.** 각 provider는 자기만의 변수를 읽습니다. CLI는 vendor 간에 *폴백하지 않습니다*:

  | Provider | 허용되는 env var |
  |---|---|
  | `anthropic` | `ANTHROPIC_API_KEY` |
  | `openai` | `OPENAI_API_KEY` |
  | `gemini` | `GEMINI_API_KEY` **또는** `GOOGLE_API_KEY` (먼저 비어 있지 않은 것이 이김) |
  | `local` / `ollama` / `vllm` / `llama_cpp` | 필요 없음 (`--base-url` 사용) |

  `OPENAI_API_KEY`를 설정해도 `--target-provider gemini`를 인증하지 않습니다. `GEMINI_API_KEY`를 설정해도 `--target-provider openai`를 인증하지 않습니다.

### "ProviderError: Gemini API key is required for provider='gemini'"

`GEMINI_API_KEY`도 `GOOGLE_API_KEY`도 설정되지 않았습니다. <https://aistudio.google.com/apikey>에서 무료 tier key를 받은 다음:

```bash
export GEMINI_API_KEY=AIza...
```

### API 예산을 쓰기 전에 어떻게 sanity-check 하나요?

deterministic smoke test를 돌리세요 — key 없음, network 없음:

```bash
python examples/reference/reproduce_reference_artifact.py
omegaprompt report examples/reference/reference_artifact.json
```

이것은 provider를 결코 건드리지 않고 calibration 커널 + judge + artifact 직렬화를 end-to-end로 돌립니다. 이게 실패하면 설치가 깨진 것이고; 통과하면 다음으로 테스트할 것은 provider입니다.

provider당 단일 라이브 호출(가능한 가장 적은 지출)을 위해서는 `ProviderRequest`를 직접 구성하세요:

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

여기서의 401은 key 문제이고; `ImportError`는 누락된 선택적 vendor SDK(`pip install "omegaprompt[anthropic]"` 등)이며; 성공적 호출은 provider 경로가 건강하고 eval을 진행해도 됨을 확인합니다.

### Gemini 호출은 되는데 strict mode에서 `LLMJudge`가 거부함

설계상 그렇습니다. Gemini는 `ship_grade_judge=False`입니다. strict mode에서 judge tier 검사는, ship recommendation이 검증되지 않은 judge에 기대는 artifact를 만드는 대신 빠르게 실패합니다. 두 가지 진행 방법:

- Gemini를 **target**으로, Anthropic / OpenAI를 **judge**로 쓰세요(cross-vendor는 여전히 strict mode를 만족합니다).
- `--profile expedition` 아래 돌리세요. 실패하는 대신 `RelaxedSafeguard`를 기록합니다 — artifact는 완화된 경계를 반영하고 하류의 `diff`가 그것을 드러냅니다.

forking된 adapter에서 `ship_grade_judge=True`로 뒤집기 전에 독립적으로 검증하세요.

---

## 신뢰 및 toolkit 문서

신뢰 관련 세부는 README 산문이 source-backed로 유지되도록 집중된 문서들에 삽니다:

- [Trust model](docs/trust-model.md) — `CalibrationArtifact`가 무엇을 증명하고 무엇을 증명하지 않는지, train/test 원칙, held-out transfer-gate 한계, offline 대 live 근거, no-live-provider 기본 CI 규칙, MCP 선택 경계, 그리고 diff regression 사용.
- [Toolkit positioning](docs/toolkit-positioning.md) — `omegaprompt` 대 `omega-lock`, `antemortem-cli`, 선택적 `mini-*` preflight plugin, `omegacal` 호환 alias, 그리고 no-dashboard/no-web-app 범위.
- [Provider capabilities](docs/provider-capabilities.md) — 코드와 계약 테스트에 묶인 adapter capability 주장.
- [Profiles and risk boundaries](docs/profiles-and-risk-boundaries.md) — strict mode 대 fast mode 동작과 validation-mode 해석.

정확한 public claim은 여전히 claim ledger와 deterministic artifact가 관장합니다.

---

## 부록 A: 데이터 계약

Pydantic v2 model, 명시되지 않은 한 `extra="forbid"`.

```python
# domain/dataset.py

class DatasetItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    input: str
    reference: str | None = None
    metadata: dict = {}

class Dataset(BaseModel):
    items: list[DatasetItem]
    @classmethod
    def from_jsonl(cls, path) -> Dataset: ...


# domain/params.py

class PromptVariants(BaseModel):
    system_prompts: list[str]                        # non-empty
    few_shot_examples: list[dict[str, str]] = []     # {input, output}

class MetaAxisSpace(BaseModel):
    system_prompt_idx_max: int                        # ge=0
    few_shot_min: int = 0
    few_shot_max: int = 3                             # >= few_shot_min
    reasoning_profiles:   list[ReasoningProfile]     # non-empty
    output_budgets:       list[OutputBudgetBucket]   # non-empty
    response_schema_modes: list[ResponseSchemaMode]  # non-empty
    tool_policy_variants: list[ToolPolicyVariant]    # non-empty

class ResolvedPromptParams(BaseModel):
    system_prompt_variant: int
    few_shot_count: int
    reasoning_profile: ReasoningProfile = STANDARD
    output_budget_bucket: OutputBudgetBucket = MEDIUM
    response_schema_mode: ResponseSchemaMode = FREEFORM
    tool_policy_variant: ToolPolicyVariant = NO_TOOLS


# domain/judge.py

class Dimension(BaseModel):
    name: str
    description: str
    weight: float                                    # ge=0
    scale: tuple[int, int] = (1, 5)                  # hi > lo

class HardGate(BaseModel):
    name: str
    description: str
    evaluator: Literal["judge", "rule", "post"] = "judge"

class JudgeRubric(BaseModel):
    dimensions: list[Dimension]                      # min_length=1, unique names, sum(weight) > 0
    hard_gates: list[HardGate] = []                  # unique names

class JudgeResult(BaseModel):
    scores: dict[str, int]                           # keyed by dimension name
    gate_results: dict[str, bool] = {}               # keyed by gate name
    notes: str = ""


# domain/profiles.py

class ExecutionProfile(str, Enum):
    GUARDED = "guarded"
    EXPEDITION = "expedition"

class ShipRecommendation(str, Enum):
    SHIP = "ship"
    HOLD = "hold"
    EXPERIMENT = "experiment"
    BLOCK = "block"

class RiskCategory(str, Enum): ...
class BoundaryWarning(BaseModel): ...
class RelaxedSafeguard(BaseModel): ...


# providers/base.py

class CapabilityTier(str, Enum):
    CORE = "tier_1_core_parity"
    CLOUD = "tier_2_cloud_grade"
    LOCAL = "tier_3_local"

class CapabilityEvent(BaseModel):
    capability: str
    requested: str
    applied: str
    reason: str
    user_visible_note: str
    affects_guarded_boundary: bool = True

class ProviderCapabilities(BaseModel):
    provider: str
    tier: CapabilityTier
    supports_strict_schema: bool = False
    supports_json_object: bool = False
    supports_reasoning_profiles: bool = False
    supports_usage_accounting: bool = True
    supports_llm_judge: bool = False
    ship_grade_judge: bool = False
    supports_tools: bool = False
    experimental: bool = False
    placeholder: bool = False
    notes: list[str] = []

class ProviderRequest(BaseModel):
    system_prompt: str
    user_message: str
    few_shots: list[dict[str, str]] = []
    reasoning_profile: ReasoningProfile = STANDARD
    output_budget_bucket: OutputBudgetBucket = MEDIUM
    response_schema_mode: ResponseSchemaMode = FREEFORM
    tool_policy_variant: ToolPolicyVariant = NO_TOOLS
    execution_profile: ExecutionProfile = GUARDED
    output_schema: type[BaseModel] | None = None     # required when STRICT_SCHEMA

class ProviderResponse(BaseModel):
    text: str = ""
    parsed: BaseModel | None = None
    usage: dict[str, int] = {}
    finish_reason: str | None = None
    latency_ms: float = 0.0
    degraded_capabilities: list[CapabilityEvent] = []
    capability_notes: list[str] = []


# domain/result.py

class EvalItemResult(BaseModel):
    item_id: str
    params: dict
    raw_output: str
    judge: JudgeResult
    token_usage: dict[str, int] = {}
    latency_ms: float = 0.0
    degraded_capabilities: list[CapabilityEvent] = []
    boundary_warnings: list[BoundaryWarning] = []

class EvalResult(BaseModel):
    params: dict
    resolved_params: dict = {}
    item_results: list[EvalItemResult]
    fitness: float
    n_trials: int
    hard_gate_pass_rate: float = 0.0
    usage_summary: dict[str, int] = {}
    latency_ms: float = 0.0
    estimated_cost_units: float = 0.0
    degraded_capabilities: list[CapabilityEvent] = []
    boundary_warnings: list[BoundaryWarning] = []
    within_guarded_boundaries: bool = True
    ship_recommendation: ShipRecommendation = HOLD
    metadata: dict = {}

class WalkForwardResult(BaseModel):
    train_best_fitness: float
    test_fitness: float
    generalization_gap: float
    kc4_correlation: float | None
    passed: bool

class CalibrationArtifact(BaseModel):
    schema_version: str = "2.0"
    engine_name: str = "omegaprompt"
    method: str
    unlock_k: int                                    # ge=0
    selected_profile: ExecutionProfile = GUARDED
    neutral_baseline_params: dict = {}
    calibrated_params: dict = {}
    neutral_fitness: float = 0.0
    calibrated_fitness: float = 0.0
    uplift_absolute: float = 0.0
    uplift_percent: float = 0.0
    quality_per_cost_neutral: float = 0.0
    quality_per_cost_best: float = 0.0
    quality_per_latency_neutral: float = 0.0
    quality_per_latency_best: float = 0.0
    boundary_warnings: list[BoundaryWarning] = []
    degraded_capabilities: list[CapabilityEvent] = []
    ship_recommendation: ShipRecommendation = HOLD
    stayed_within_guarded_boundaries: bool = True
    additional_uplift_from_boundary_crossing: float = 0.0
    relaxed_safeguards: list[RelaxedSafeguard] = []
    guarded_boundary_crossed: bool = False
    cost_basis: str = "normalized_token_units"
    best_params: dict                                # kept for v1.x compat
    best_fitness: float
    walk_forward: WalkForwardResult | None = None
    hard_gate_pass_rate: float                       # 0..1
    sensitivity_ranking: list[dict] = []
    n_candidates_evaluated: int
    total_api_calls: int
    usage_summary: dict[str, int] = {}
    latency_summary_ms: dict[str, float] = {}
    target_provider: str | None = None
    target_model: str | None = None
    judge_provider: str | None = None
    judge_model: str | None = None
    target_capabilities: ProviderCapabilities | None = None
    judge_capabilities: ProviderCapabilities | None = None
    status: str = "OK"                               # OK / FAIL_KC4_GATE / FAIL_HARD_GATES / FAIL_NO_CANDIDATES
    rationale: str = ""
```

`ResolvedPromptParams`와 `ProviderRequest`는 v1.0 축 이름(`system_prompt_idx`, `output_budget`, `tool_policy`)을 받아 v1.1+ canonical 이름(`system_prompt_variant`, `output_budget_bucket`, `tool_policy_variant`)으로 다시 쓰는 `@model_validator(mode="before")` compat 매핑을 가집니다. 읽기 측의 `@property` accessor가 어느 이름이든 쓸 수 있게 합니다.

---

## 부록 B: meta-axis → vendor-parameter 매핑

adapter 구현에서 발췌한 canonical 번역 표.

| Meta-axis value | Anthropic | OpenAI / compatible | Gemini | Local (Ollama / vLLM / llama.cpp) |
|---|---|---|---|---|
| `reasoning_profile = OFF` | no `thinking` block | no `reasoning_effort` | model default | system prompt unchanged |
| `reasoning_profile = LIGHT` | `thinking={type:adaptive}` + `effort: low` | `reasoning_effort: low` (if supported) | model default + `CapabilityEvent` | system-prompt suffix: "think briefly" |
| `reasoning_profile = STANDARD` | `thinking={type:adaptive}` + `effort: medium` | `reasoning_effort: medium` | model default | system-prompt suffix: "think step by step" |
| `reasoning_profile = DEEP` | `thinking={type:adaptive}` + `effort: high` | `reasoning_effort: high` | model default + `CapabilityEvent` | system-prompt suffix: "think carefully step by step" |
| `output_budget_bucket = SMALL` | `max_tokens=1024` | `max_tokens=1024` | `max_output_tokens=1024` | `max_tokens=1024` |
| `output_budget_bucket = MEDIUM` | `max_tokens=4096` | `max_tokens=4096` | `max_output_tokens=4096` | `max_tokens=4096` |
| `output_budget_bucket = LARGE` | `max_tokens=16000` | `max_tokens=16000` | `max_output_tokens=16000` | `max_tokens=16000` |
| `response_schema_mode = FREEFORM` | `messages.create` | `chat.completions.create` | `generate_content` | `chat.completions.create` |
| `response_schema_mode = JSON_OBJECT` | `messages.create` + system-prompt JSON suffix | `response_format={type:json_object}` | `response_mime_type=application/json` + JSON suffix | best-effort system-prompt instruction |
| `response_schema_mode = STRICT_SCHEMA` | `messages.parse(output_format=T)` | `beta.chat.completions.parse(response_format=T)` | `response_schema=T` + local Pydantic validation; if unavailable, fast-mode-only JSON fallback | not supported; strict mode raises |
| `tool_policy_variant = NO_TOOLS` | no `tools` argument | no `tools` argument | no `tools` argument | no `tools` argument |
| `tool_policy_variant = TOOL_OPTIONAL` | `tools=[...]`, no `tool_choice` | `tools=[...], tool_choice="auto"` | not mapped | adapter-specific |
| `tool_policy_variant = TOOL_REQUIRED` | `tools=[...], tool_choice={type:"any"}` | `tools=[...], tool_choice="required"` | not mapped | adapter-specific |

"not supported"나 "best-effort"라고 읽히는 셀은 런타임에 `CapabilityEvent`를 방출하고, strict mode에서는 execution profile 정책에 따라 실행을 막을 수 있습니다.

---

## 부록 C: invariant

다음 속성들은 구조적으로 성립하며 Pydantic schema 계층이나 전용 테스트에서 강제됩니다. 그것들은 리뷰어가 구현을 읽지 않고도 의지할 수 있는 "정리(theorem)"입니다.

1. **클라이언트 측 schema regex 없음.** `STRICT_SCHEMA` 모드는 provider의 가장 강한 구조화 경로(Anthropic의 `messages.parse`, OpenAI의 `beta.chat.completions.parse`, 가능하면 Gemini의 `response_schema`)로 dispatch합니다. JSON-object에 로컬 Pydantic 검증을 더한 fallback은 명시적이며 fast-mode 전용입니다. malformed 구조화 응답은 calibration 루프가 보기 전에 예외를 던집니다.
2. **Hard-gate fitness collapse (하드 게이트 fitness 붕괴).** 어떤 `(item, params)` 쌍이든, 어떤 `hard_gate`라도 `False`를 반환하면 그 item의 `CompositeFitness` 기여는 `0.0`입니다. soft penalty도, 부분 점수도 없습니다.
3. **Held-out 임계값 불변성.** `--max-gap`과 `--min-kc4`는 실행 시작 시 한 번 해석되어 artifact에 기록되는 CLI 인자입니다. 실행 도중 그것들을 수정할 API 표면이 없습니다.
4. **Capability-event 전파.** `ProviderResponse.degraded_capabilities`에서 방출된 모든 `CapabilityEvent`는 `EvalItemResult` → `EvalResult` → `CalibrationArtifact.degraded_capabilities`로 변경 없이 위로 흐릅니다. adapter는 조용히 degrade할 수 없습니다.
5. **strict mode의 ship-grade judge 검사.** strict mode(`ExecutionProfile.GUARDED`) 아래서, `provider.capabilities().supports_llm_judge`가 `False`이면 `LLMJudge.score`는 `JudgeError`를 던집니다. 암묵적 면제 없음.
6. **Deterministic한 결정 도출.** `ship_recommendation`, `status`, `stayed_within_guarded_boundaries`, `guarded_boundary_crossed`는 artifact 필드와 profile 정책의 순수 함수로 계산됩니다. 같은 입력, 같은 출력.
7. **Backward-compat key 재작성은 무손실.** `ProviderRequest`와 `ResolvedPromptParams`는 레거시 key(`system_prompt_idx`, `output_budget`, `tool_policy`)를 canonical 이름으로 다시 쓰는 `@model_validator(mode="before")`를 통해 받습니다. 읽기 측의 `@property` accessor가 두 이름을 모두 보존합니다.
8. **`neutral_baseline_params`와 `calibrated_params`는 `best_params`와 결코 모순되지 않음.** `CalibrationArtifact.model_post_init`은 한쪽이 없을 때 `best_params ↔ calibrated_params`와 `best_fitness ↔ calibrated_fitness`를 동기화하므로, 하류 소비자는 존재 검사 없이 어느 쌍이든 읽을 수 있습니다.
9. **동일 입력에서의 실행 간 artifact JSON 안정.** §11.2의 재현성 스크립트가 이것을 경험적으로 입증하고; `tests/test_calibrate_integration.py`의 integration 테스트가 CI에서 강제합니다.
10. **Adaptation은 원칙을 강화만 함.** `apply_adaptation_plan`은 `max(default_min_kc4, plan_override)`, `min(default_max_gap, plan_override)`, `min(default_unlock_k, plan_override)`를 씁니다. tolerance를 넓히려는 plan은 호출자의 구성으로 clip됩니다. §5.8과 부록 D 참고.

---

## 부록 D: AdaptationPlan 계약

```python
class ParameterOverride(BaseModel):
    parameter: str
    default: Any
    applied: Any
    reason: str

class AdaptationPlan(BaseModel):
    # Walk-forward gate
    min_kc4_override: float | None = None          # never lower than caller default
    max_gap_override: float | None = None          # never higher than caller default

    # Search
    unlock_k_override: int | None = None           # never larger than caller default
    skip_axes: list[str] = []                      # axes pre-excluded from sensitivity

    # Evaluation
    rescore_count: int = 1                         # N-judge median when consistency low
    rubric_weight_overrides: dict[str, float] = {} # zero out unreliable dims only
    schema_mode_fallback: ResponseSchemaMode | None = None   # STRICT -> JSON_OBJECT only
    judge_ensemble_shift: float | None = None      # extra weight toward RuleJudge

    # Scheduling
    candidate_budget_cap: int | None = None
    dataset_reorder_for_cache: bool = False

    # Audit
    overrides: list[ParameterOverride] = []        # itemised reason trail
    rationale: list[str] = []                      # human-readable summary
    preserves_discipline: bool = True              # must be True by construction
```

### 도출 규칙 (deterministic)

같은 `PreflightReport`가 들어가면 → 같은 `AdaptationPlan`이 나옵니다. 규칙은 `omegaprompt.preflight.adaptation.derive_adaptation_plan`에 담겨 있고 noise 수준, consistency 수준, schema-reliability 수준, wall-time 예측, 그리고 analytical-finding 조합에 걸쳐 단위 테스트됩니다.

### 적용 규칙 (invariant 보존)

```python
def apply_adaptation_plan(
    plan: AdaptationPlan,
    *,
    min_kc4: float,
    max_gap: float,
    unlock_k: int,
) -> tuple[float, float, int]:
    applied_kc4   = max(min_kc4,  plan.min_kc4_override  or min_kc4)
    applied_gap   = min(max_gap,  plan.max_gap_override  or max_gap)
    applied_unlock = min(unlock_k, plan.unlock_k_override or unlock_k)
    return applied_kc4, applied_gap, applied_unlock
```

세 clipping 규칙은 하나의 원리를 인코딩합니다: **AdaptationPlan은 조일 수는 있어도 결코 풀 수 없다**. `min_kc4`를 0.6에서 0.4로 낮추려는 악의적이거나 버그 있는 plan은 조용히 0.6으로 clip됩니다 — 호출자의 엄격함이 이깁니다.

### Audit trail

모든 override는 자신의 `parameter`, `default`, `applied`, `reason`을 담습니다. plan의 `rationale`은 사람이 읽는 한 줄짜리들의 free-form 리스트입니다. 둘 다 `CalibrationArtifact`로 직렬화되므로, 실행 6개월 뒤에 artifact를 읽는 리뷰어가 어떤 parameter가 쓰였는지뿐 아니라 *왜* 기본값에서 벗어났는지도 볼 수 있습니다.

### Sub-unit 경계

`omegaprompt.preflight`는 **계약 + adaptation 로직만** 제공합니다. probe 실행(`mini-omega-lock`)과 analytical 분류(`mini-antemortem-cli`)는 이 계약에 의존하는 별도의 repository / PyPI 패키지에 삽니다. 외부 sub-tool은 `derive_adaptation_plan`이 그것을 메인 파이프라인이 소비하는 `AdaptationPlan`으로 바꾸게 하려면, :mod:`omegaprompt.preflight.contracts`를 따르는 `PreflightReport`를 방출하기만 하면 됩니다. 단독 사용자는 아무것도 추가로 설치하지 않습니다; 그들에게 preflight 인터페이스는 no-op 표면입니다.

---

## 인용

짧은 형식:

```
omegaprompt v2.0.2 — provider-neutral prompt calibration engine.
https://github.com/hibou04-ops/omegaprompt, 2026.
```

BibTeX:

```bibtex
@software{omegaprompt_2026,
  author  = {hibou04-ops},
  title   = {{omegaprompt}: Provider-neutral prompt calibration engine
             with sensitivity-ranked meta-axes, walk-forward ship gates,
             and structural capability reporting},
  version = {2.0.2},
  year    = {2026},
  url     = {https://github.com/hibou04-ops/omegaprompt}
}

@software{omegalock_2026,
  author  = {hibou04-ops},
  title   = {{omega-lock}: Sensitivity-driven coordinate-descent
             calibration framework with walk-forward validation and
             pre-declared kill criteria},
  version = {0.3.0},
  year    = {2026},
  url     = {https://github.com/hibou04-ops/omega-lock}
}

@software{antemortem_2026,
  author  = {hibou04-ops},
  title   = {{Antemortem}: AI-assisted pre-implementation reconnaissance
             for software changes with disk-verified citations},
  version = {0.1.1},
  year    = {2026},
  url     = {https://github.com/hibou04-ops/Antemortem}
}
```

---

## License

Apache 2.0. [LICENSE](LICENSE)를 참고하세요.

**License 히스토리.** 버전 1.1.0 이하의 PyPI 배포판은 MIT `LICENSE` 파일과 함께 ship되었습니다. repository는 1.1.0과 1.2.0 PyPI 업로드 사이에 Apache 2.0으로 relicense되었습니다; 1.2.0(2026-04-27)과 그 이후 모든 버전은 Apache 2.0으로 ship됩니다. 1.1.0 이하를 설치한 누구든 그 사본에 대해 MIT license를 보유합니다 — license 변경은 소급 적용되지 않습니다.

## Colophon

혼자 설계하고, 구현하고, ship했습니다. `omega-lock` 위의 adapter 계층; calibration-engine 재구현 0. 모든 non-trivial한 변경은 `antemortem-cli`의 recon 원칙을 통해 사전 작성됩니다. 테스트는 offline으로 돌고; CI에 라이브 API 호출이 없습니다.
