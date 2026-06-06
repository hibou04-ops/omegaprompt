# omegaprompt 개선 계획 — 멀티에이전트 회의(council) 결과

- 생성: 2026-06-06
- 소스: `/office-hours` + 15-에이전트 워크숍 (Workflow Task `wb6a54dld`, run `wf_698e9801-b1f`)
- 모델: Claude Opus 4.8 (council). 초안 → red-team(5/10) → 결함 4건 수정한 **최종본**
- 저장소: `hibou04-ops/omegaprompt`, branch `main`, v2.0.0
- 상태: DRAFT (다음 단계: `/plan-eng-review`)

> ⚠️ **작업 규칙**: 이 작업은 라이브 저장소 `C:\Users\hibou\omegaprompt`에서 직접 진행한다.
> 안전망 = git. 결과가 나쁘면 **마지막 커밋으로 롤백** (`git reset --hard 4c04920b954f89cff2f23b1ee3ef1c9fd2f5cd9d`).
> **작업 중 `git commit` / `git push` 절대 금지.** 커밋 시점은 사용자가 정한다.
> 이 문서는 untracked 파일이라 `git reset --hard`로 코드 롤백해도 사라지지 않는다.

---

## 0. 목표 (확정)

- **"범용성(universality)" 폐기** — 넓히기 아님.
- 목표 = **성능 격상 + "독보적 구조 복합연산의 극치" + 현재 성능 한계 극복 + 완전무결성(믿고 쓰는 툴)**.
- 포지셔닝: **(2) 돈 내는 제품 + (3) 개인 최강/장인 + (4) 오픈소스·연구**. (1) 표준-되기는 제외.
- 원칙: **깊이 > 폭.** 모든 항목은 정합성(integrity)/신뢰(trust)/CI-경제성 중 하나 이상을 강화한다.

---

## 1. 핵심 진단 (워크숍이 `src/` + `.venv/omega_lock`을 직접 읽고 확인)

### 성능 (현재 한계)
- 파이프라인 전체가 **직렬·동기**. `PromptTarget.evaluate`가 데이터셋 항목마다 target 1콜 + judge 1콜을 블로킹 호출 (`targets/prompt_target.py:178-234`). **동시성 0, 메모이제이션 0.**
- grid는 축 크기의 곱(완전 탐색). 기본 `unlock_k=3`, 5점/축 = **125 조합 × 전체 데이터셋 패스** (`omega_lock/grid.py:112`). 서러게이트/조기중단/베이지안 없음 (`omega_lock/p2_tpe.py`는 있지만 `run_p1`이 호출 안 함).
- 기본 `calibrate` 1회 ≈ `evaluate()` 150~155회. 20항목 데이터셋이면 **≈ 6,200 직렬 HTTP 왕복**. README 기준 run당 수십 달러.
- 중복 작업: 이중 walk-forward (omega_lock이 KC-4/test eval 계산해놓고 `runtime.py:325/338`이 버림), 이중 neutral 평가, 이미 점수 낸 grid 승자 재평가 (`runtime.py:330`).

### 완전무결성 (현재 구멍)
- 간판 게이트 **KC-4가 흔한 경우(disjoint/zero-variance)에 조용히 통과로 기본값 처리** — `kc4_ok`가 `True`로 default (`core/walkforward.py:188`). "KC-4 passed" 초록불이 실제로는 **한 번도 발화 안 했을 수 있음.**
- omega_lock이 계산한 **더 강한 신호(교차 후보 train/test fitness 벡터, KC-3 floor)를 runtime이 폐기** (`orchestrator.py:222-223` 계산 → `runtime.py:325` 무시).
- **출하됐지만 죽은 judge-신뢰성 기계**: `runtime.py:918,920`이 `judge_quality/performance`를 `None`으로 하드코딩 → rescore/noise-floor 적응이 절대 발화 못 함. `rescore_count`는 `APPLIED_PARAMETERS`에서 제외 (`preflight/adaptation.py:117-122`).
- **temperature/seed가 providers 어디에도 없음.** `ProviderRequest`는 `extra='forbid'`. 핀고정은 계약 변경이 필요하고 Anthropic엔 seed API 자체가 없음.
- 고정 `min_kc4`(0.5/0.3)가 **n-무관** → n=3에서 유의한 r은 ~0.997 필요 = 사실상 무의미 (`core/profiles.py:31-32,40-41`).

---

## 2. 결정적 통찰 — "거절하지 않고 등급을 매긴다"

**가장 큰 긴장**: 통계를 정직하게 하면 10~20항목 현실 데이터셋에선 거의 매번 HOLD → "믿을 verdict 없음 = 제품 아님".

**해소**: `status`와 `ship_recommendation`은 **독립 필드** (확인: `core/result.py:20-34` vs `core/profiles.py:17-22`). 검정력 부족 = `status=OK` + `ship_recommendation=EXPERIMENT`.
→ **툴은 항상 답한다. 답은 증거 등급을 달고 나온다. SHIP은 검정력을 요구하고, 부족하면 EXPERIMENT로 내려가되 여전히 쓸 수 있고 팔 수 있다.**

**honesty floor vs moat 구분 (라벨을 정직하게):**
- **정직성 바닥(floor)** = 폐기 신호 소비 + 검정력-인지 등급 + 측정형 재현성. 옳고 필요하지만 **일주일이면 복제 가능**. "독보적"이라 부르지 말 것.
- **진짜 해자(moat)** = `canonical-JSON + 암호학적 사전등록(scoring 전) + 검정력-인지 채점 + 서명된 오프라인 재현`. 이 **통합 체인**이 복제 어려움. (순서상 마지막에 출하하지만 헤드라인에선 이것을 moat로 라벨.)

---

## 3. Top 3 베팅

**Bet 1 — omega-lock이 버리는 교차 후보 train/test fitness *벡터*를 가져와, omega-lock의 verdict가 아니라 omegaprompt 자체(trade_ratio-free) 게이트로 통과시킨다.**
- 레버리지/노력 최고이며 초안의 #1 버그를 고침. `run_p1`은 이미 top-N grid 후보에 대해 WalkForward를 계산하고 `result.walk_forward['train_fitnesses']/['test_fitnesses']`로 노출(`orchestrator.py:222-223`, `walk_forward.py:48-49`)하지만 `runtime.py:325`가 `grid_best`만 읽고 버림. 이 **교차-후보** 벡터는 공통 disjoint split(n=top_n≈10)에서 잘 정의되며, 바로 거기서 per-item KC-4가 fail-open(`walkforward.py:188`).
- **핵심 수정**: omega-lock의 `check_kc4`(`kill_criteria.py:149`)는 `trade_ratio = len(test)/len(train)` 항을 품어 불균등 split(예 70/30=0.43<0.5)에서 **거짓 FAIL**, add-only 규칙으로는 거부 불가. 벡터를 가져와 trade_ratio 없는 omegaprompt 게이트로 채점 → 버그 제거 + monotonic-strengthen 충돌 해소.
- **추가 API 콜 0.** 타깃: `runtime.py:325`, `omega_lock/orchestrator.py:222-223,363`, `core/walkforward.py:171-188`, `omega_lock/kill_criteria.py:149`(가져오면 안 되는 부분).

**Bet 2 — 검정력-인지 *채점*(거절 아님): n-인지 min_kc4, held-out uplift + item-bootstrap CI(0 제외), UNDERPOWERED→EXPERIMENT.**
- red-team이 미해결이라 한 "척추", 그리고 Bet 1의 backstop. 고정 임계값을 **one-sided Pearson 유의성 기준(n 줄면 강해짐)**으로 교체. uplift는 검증 안 된 correlated-grid shrinkage 추정치 대신 **in-sample argmax uplift + held-out test uplift(이미 selection-unbiased) 둘 다 보고**(그 격차가 winner's-curse 공개), **항목 재표집 bootstrap CI가 0을 배제**할 때만 uplift-positive 게이트.
- 모두 저장된 `per_item_scores`에 대한 사후 계산 = **추가 API 콜 0.** 타깃: `core/profiles.py:31-32,40-41`, `core/walkforward.py:116-117,188`, `runtime.py:387-392,338`, `core/risk.py:99`.

**Bet 3 — 성능을 *wedge*로: 캐시-인지 async fan-out + canonical-params eval memo + cardinality-gated TPE, "per-PR CI 유료 GitHub Action"으로 판매.**
- 누락된 "돈 내는 사용자 wedge"를 명명, 그리고 Bet 2 반복의 비용 인에이블러. (a) 항목·조합 두 차원에 대한 asyncio/thread-pool gather, Anthropic ephemeral-cache 창(`anthropic_provider.py:106`) 안에서 동일 system-prompt 콜 스케줄링; (b) `(canonical resolved-params, dataset fingerprint, model id, rubric hash)` 키 eval memo로 이중-neutral·승자 재평가(`runtime.py:330`)·grid/stress/walk-forward 중복 제거(단 **repeat-draw 축은 캐시 경계에서 제외** → 반복셋을 무너뜨리지 않음); (c) **cardinality-gated 서처**: 열거 가능하면 완전 grid(예: 5 system_prompts × 2 few_shots는 grid가 더 싸고 완전), 예산 초과 시에만 기출하 TPE(`run_p2_tpe`, seed-pinned study).
- 타깃: `targets/prompt_target.py:178-234,301`, `omega_lock/grid.py:43-54,112,119`, `omega_lock/p2_tpe.py`, `runtime.py:319,330`.

---

## 4. 등급별 실행 계획

### Critical
1. **TV3 — 비활성 auto-mode KC-4를 1급 `GATE_NOT_FIRED` facet으로 만들어 SHIP을 EXPERIMENT로 내린다 (조용한 초록불 금지).**
   - 노력 **S**, 의존성 0, 의회 최고 효율. disjoint/zero-variance에서 `kc4_ok`가 조용히 True default(`walkforward.py:188`) → 리포트의 초록 "KC-4 passed"가 실제 발화 안 함. 8-state `KC4Status` 분류(`walkforward.py:55-64`)는 이미 사유를 명명; verdict 매핑만 pass-default를 멈추면 됨. `GATE_NOT_FIRED`(PASS/FAIL과 구분) 도입 + 의도적 `validation_mode='disjoint'`가 아니면 `ship_recommendation→EXPERIMENT`(절대 거절 아님). 비발화 사유별 golden 케이스 추가.
   - **모든 provenance 작업보다 먼저** — fail-open verdict에 서명하면 빈 주장을 암호로 인증하는 꼴. 타깃: `core/walkforward.py:55-64,171-188`, `core/risk.py:99`, `examples/reference/reproduce_reference_artifact.py`, `core/artifact_integrity.py:657-665`.
2. **Bet 1 — 폐기된 교차-후보 벡터를 omegaprompt 자체 게이트로 (trade_ratio 버그를 근원에서 제거).** 노력 **M**, 추가 API 콜 0. (위 Bet 1 참조.) Bet 2와 한 묶음으로 출하.

### High
3. **Bet 2 — 검정력-인지 채점 (한 stats-kernel 워크스트림).** 노력 **M**. shrinkage 추정치는 **CUT**(연기 아님). 싼 것부터: n-인지 임계값, held-out uplift, 0-배제 item-bootstrap CI. 타깃: `core/profiles.py:31-32,40-41`, `core/walkforward.py:116-117`, `runtime.py:387-392,338`, `core/fitness.py:22-34`.
4. **IC4 — 출하됐지만 죽은 judge-신뢰성 기계 되살리기 (rescore-count 활성 + `judge_quality/performance` 하드코딩 해제).** 노력 **M**. `runtime.py:918,920` 하드코딩 해제, `rescore_count`를 `APPLIED_PARAMETERS`에 추가, 측정 judge consistency<0.80일 때 median-of-N 연결. 이게 Bet 2의 n-인지 임계값이 소비할 noise_floor를 공급(현재 항상 0이라 비활성). 반복-N은 비용↑이라 **opt-in/임계 게이트**이며 **Bet 3 fan-out에 의존**. 타깃: `runtime.py:918,920`, `preflight/adaptation.py:117-122,221-246`, `contracts.py:63-84`.
5. **IC5 — 측정-전용 재현성: provenance manifest + 잔여 비결정성 probe + `verdict_reproducibility` 게이트 (seed/temp 핀고정 NO).** 노력 **M**. 초안의 핀고정은 **DROP**(providers에 temp/seed 없음, `extra='forbid'`, Anthropic seed 無, provider-중립 정체성과 충돌, golden 레퍼런스 churn). 대신 model fingerprint + dataset/rubric/split 해시 + dep 버전 manifest, **작은 probe 셋**에서 잔여 비결정성 측정, `verdict_reproducibility{fitness_std, winner_stable}`가 불안정 시 EXPERIMENT로 하향. 계약 변경·해시 churn 없음. 타깃: `targets/prompt_target.py:178-201`, `core/artifact_integrity.py:627-654`, `preflight/adaptation.py:125-139`.

### Medium
6. **Bet 3 — 캐시-인지 async fan-out + canonical-params memo + cardinality-gated TPE (CI-경제성 wedge).** 노력 **L**. 유료 per-PR GitHub Action의 매출 경로이자 IC4 반복-N의 비용 인에이블러. memo 경계는 repeat-draw 축 제외, 서처 스왑은 cardinality-gated(무조건 grid→TPE는 열거 가능 공간에서 trust 후퇴). "bytes-identical verdict" 주장 안 함(결정성은 IC5에서 측정만). 타깃: `targets/prompt_target.py:178-234,301`, `omega_lock/grid.py:112,119`, `omega_lock/p2_tpe.py`, `providers/base.py:126`(call_async thread-pool shim).
7. **CR1 — 오프라인 재현 가능한 winner's-curse + verdict-flip 연구 (장인 증명).** 노력 **M**. goals 3/4용 최강 오프라인 산출물. (1) in-sample argmax uplift가 held-out uplift를 체계적으로 과대; (2) reseeding 하 verdict-flip rate — "우리 등급 verdict는 안정, 단일-draw verdict는 뒤집힘". 모든 수치를 public claim ledger에 `deterministic_artifact`로 바인딩(네트워크 없이 재실행). Bet 2(held-out uplift)·IC4/IC5(reseed spread)에 의존. 타깃: `examples/reference/reproduce_reference_artifact.py`, `golden_manifest.json`, `docs/claims/public_claim_ledger.json`, `tools/reproduce_golden_reference.py`.
8. **PROD-VAL — 실행 가능한 생산검증 *프로토콜* 복원 (초안이 버린 반증가능성 backstop).** 노력 **M**. 완전무결성 thesis가 "생산 결과로 한 번도 검증 안 된 holdout 게이트"에 얹혀 있던 문제(`trust-model.md`도 "KC4는 예언이 아니라 가드레일"이라 인정). 보편 벤치마크 아님(=표준-positioning 회귀) — **사용자 A/B 결과를 받아 KC-4-predicts-production 상관을 사용자 데이터에서 계산하는 템플릿+커맨드**. "당신 트래픽에서 게이트를 검증하라." 타깃: `docs/trust-model.md`, **신규** `tools/validate_gate_on_ab_data.py`, `docs/claims/public_claim_ledger.json`.

### Later
9. **IC6 + TV5 — 암호학적 사전등록(scoring 전) + 서명된 오프라인 재현 attestation(후): 진짜 독보적 해자.**
   - 노력 **L**. gate-consumption이나 textbook CI가 아니라 **이 통합 provenance 체인**이 유일하게 복제-어려운 산출물. IC6은 `{dataset, split, rubric+weights, declared thresholds}` 약정을 scoring 전에 서명 → `check-artifact`가 사후 임계값 drift에 FAIL = 사전선언 게이트 규율을 관습에서 **암호 구조**로(Winchester 방어의 구조화). TV5는 사후 detached, 오프라인 검증 attestation(정규화 해시 재계산·기록 커맨드 재실행·서명 확인). **순서상 LAST**(fail-open/underpowered verdict에 서명하면 빈 주장 인증)이지만 헤드라인에선 **moat로 라벨**. 프레이밍 규율: "임상시험 사전등록을 커뮤니티가 채택" 류 표현 금지(폐기한 표준-breadth positioning에 기댐). 타깃: `core/artifact_integrity.py:627-654`, `examples/reference/golden_manifest.json`, `docs/claims/public_claim_ledger.json`.

---

## 5. 순서 근거

`TV3 → Bet1+Bet2 → IC4/IC5 → Bet3 → CR1/PROD-VAL → provenance(IC6/TV5)`

1. **TV3 먼저·단독** (S, 의존성 0): 가장 과장되기 쉬운 주장(발화 안 한 초록 "KC-4 passed")을 제거. 모든 provenance보다 선행.
2. **Bet 1 + Bet 2는 한 메커니즘** (기능+가드 아님): 수렴 탐색은 top-N train fitness를 군집화 → 상관 정의 불가 → Bet 1은 Bet 2의 UNDERPOWERED→EXPERIMENT 등급이 있어야 혼란스러운 non-fire를 피함.
3. **척추 긴장 해소 = 거절이 아니라 등급화**: `status`/`ship_recommendation` 독립(확인됨) → 검정력 부족 = OK + EXPERIMENT. 항상 답하고, 등급을 달고, SHIP은 검정력 요구.
4. **IC4 + IC5는 한 dead-code 활성 표면 공유** (`runtime.py:918,920` 하드코딩 해제 등) — 한 번에 되살림. CR1보다 먼저(단일-draw verdict의 flip 연구는 noise 측정).
5. **Bet 3(성능)은 CI-경제성 wedge이자 IC4 반복-N 의존성**: fail-closed verdict를 PR마다 게이트할 만큼 싸게 만듦(=유료 Action). 게이트가 **정확해진 다음** 빨라져야 하므로 Medium.
6. **CR1·PROD-VAL이 verdict를 증명** (CR1 오프라인, PROD-VAL 사용자 A/B).
7. **IC6+TV5(provenance) 순서상 LAST이지만 moat로 라벨**: 앞 단계가 신뢰성을 *만든* 숫자에 서명/인증.

---

## 6. Red-team이 잡고 최종본이 고친 것 (정직성 기록)

초안 점수 **5/10**. 진단·코드 인용은 전부 검증됨. 4대 결함 → 해소:

1. **#1 베팅의 trade_ratio 버그** (omega-lock verdict 소비 시 `len(test)/len(train)` 분모가 불균등 split에서 거짓 FAIL, add-only로 거부 불가) → **verdict 아닌 *벡터*를 omegaprompt 자체 게이트로.**
2. **거절 vs 사용성 긴장** (정직 통계가 현실 10~20항목 run을 거의 매번 HOLD) → **EXPERIMENT 등급화** (status≠ship 독립).
3. **생산검증 항목 삭제로 핵심 주장 반증불가** → **PROD-VAL 프로토콜(사용자 A/B 데이터) 복원.**
4. **돈 내는 사용자 wedge 부재** → **per-PR CI 유료 GitHub Action 명명** (Bet 3가 매출 경로).
- 추가: **"moat" 라벨을 복제 쉬운 항목에서 provenance 체인(IC6/TV5)으로 재귀속.**

---

## 7. 범위 밖 / 컷

- **shrinkage estimator CUT** (correlated-grid 이웃에 대한 검증된 추정량 없음 → held-out uplift로 대체).
- **seed/temperature 핀고정 DROP** (providers에 없음 + Anthropic seed API 無 + provider-중립 정체성 충돌 → 측정-전용 재현성으로).
- **judge-verdict-memo CUT** (확률적 단일 draw 캐싱은 반복과 모순).
- **무조건 grid→TPE 스왑 거부** → cardinality-gated.
- **CM1 open conformance spec CUT** (폐기한 universality와 충돌).
- **보편 production benchmark → 사용자-데이터 PROD-VAL 프로토콜로 교체.**

---

## 8. 다음 한 걸음 (The Assignment)

**TV3부터 출하한다.** 비활성 auto-mode KC-4를 `GATE_NOT_FIRED` 1급 facet으로 만들어 SHIP을 EXPERIMENT로 내리고, 비발화 사유별 golden 테스트를 추가. 노력 S, 의존성 0.
"KC-4 passed" 초록불이 **발화 안 한 게이트를 더는 숨기지 못하게** 만드는 것 — 이게 완전무결성의 첫 벽돌이고, 이후 provenance 서명이 의미를 갖는 전제다.

---

## 다음 단계

`/plan-eng-review` — 이 계획의 아키텍처/엣지케이스/테스트 리스크 검토. (gstack 스킬; Windows에선 Unix 프리앰블이 일부 안 돌 수 있어 "정신만 따라" 실행하는 방식으로 진행 가능.)
