# omegaprompt 개선 계획 v2 — omega-lock 0.3.0 기준 (0.1.4 플랜 supersede)

- 생성: 2026-06-06 / 소스: v2 멀티에이전트 council (Task `wtjrph1yg`, 15 에이전트, 1.41M 토큰)
- 검증: red-team **7/10** (TRUSTWORTHY-WITH-FIXES, 적용 완료). 모든 핵심 citation 코드 검증됨.
- 저장소: `hibou04-ops/omegaprompt`, branch `0606_2132`, omega-lock **0.3.0** 설치됨
- 상태: DRAFT

> ⚠️ **작업 규칙**: 라이브 저장소에서 직접 작업. 안전망 = git. 롤백 = `git reset --hard 56ad555e9800afea00d2ce3ce5b061c7a8c83cb7`(baseline) 또는 `git checkout <file>`.
> **작업 중 `git commit`/`git push` 절대 금지.** public `origin` 금지, 백업은 private `omegaprompt_0606`.

---

## 0. 새 현실 (왜 v1을 버렸나)

- omega-lock **0.1.4 → 0.3.0** 업그레이드가 omegaprompt를 **깼다**. 0.1.4 위에 짠 v1 플랜은 근간이 바뀜.
- 0.3.0은 v1이 "짓겠다"던 것 일부를 **이미 제공**: `PearsonStatus`/`pearson_computable`(undefined-correlation 처리), `test_best_fitness`/`test_best_params`(held-out 무료), `min_nonzero_stress_count`, `pure_objective()` 프리셋, 미사용 `run_p2_tpe`/`ZoomingGridSearch`/SC-2/`holdout_target`.
- → 새 원칙: **"재구현 말고 소비(consume)/배선(wire)."** v1보다 **작고 날카롭게**.

---

## 1. 호환 수정 (DONE · 검증됨 · uncommitted)

- **유일한 0.3.0 파손**: omega-lock 0.3.0이 target 결과의 정식 필드를 `n_trials` → `sample_count`로 바꿈. `stress.py:139/187`, `grid.py`, `walk_forward.py:157/161/181`이 `.sample_count`를 읽음. omegaprompt `EvalResult`는 `n_trials`만 노출 → `AttributeError` 크래시.
- **수정** (`src/omegaprompt/domain/result.py:88-101`): `sample_count` **read-only @property** → `n_trials` 반환. `n_trials`는 저장 필드 유지 → **hash-neutral**(Pydantic v2가 property 직렬화 안 함, golden drift 0).
- **검증**: reference gate `EXIT=0`, 전체 non-MCP 스위트(test_calibrate_integration/test_runtime/test_walkforward/test_target/test_artifact_integrity + ~40) green. 남은 pytest 에러 1건은 `test_mcp_execution.py`의 optional `mcp` 의존성 누락(기존 위생 문제, 0.3.0 무관).
- omega-lock은 0.3.0에서 이 속성을 **읽기만** 함 → property로 충분. 미래 버전이 *쓰면* 안 되므로 **계약 테스트(아래)가 필수**.

---

## 2. Top 3 베팅

1. **fail-loud omega-lock 계약 가드 트리오** (zero-network 계약 테스트 + 핀 상한 + 버전 스탬프). 완전무결성의 키스톤이자 0.1.4→0.3.0 파손의 직접 교훈. **`.venv`가 gitignore라 ripgrep이 omega_lock을 못 봄 → 계약 테스트만이 소비 seam을 볼 수 있음.** 다음 파손을 "조용한 런타임 크래시"에서 "이름 붙은 pre-merge CI 실패"로 전환. 전부 S.
2. **KC-4 fail-open 정직 발견** → EXPERIMENT/수동검토. 초록 SHIP 아티팩트가 **측정 안 한 KC-4를 통과로 주장**(`walkforward.py:188` `kc4_ok = kc4 is None or ...` + `artifact_integrity.py:657-665`도 통과로 동의). **detection-only**(게이트 플립 아님, fail-closed 아님) → 통과 run 안 깨고 6/4 reference도 살림. 유료 wedge와 직결(구매자는 "KC-4 통과" vs "KC-4 미계산"을 구분할 수 있어야 함).
3. **per-item target+judge 루프 결정적 동시화** (헤드라인 성능 베팅). per-PR CI 게이팅(유료 wedge)은 calibrate가 CI 창 안에 끝나야 팔림. XL, **determinism-gated**(동시 아티팩트 해시 == 직렬 해시). 중간 마일스톤 = PC1 메모이제이션이 먼저.

---

## 3. 등급별 실행 계획

### Critical
1. **[DONE — 유지검증만] `sample_count` read-only @property** (`result.py:88-101`). 저장 필드로 만들지 말 것(model_dump 키 뒤집혀 golden 재작성). S.
2. **fail-loud 계약 테스트** `tests/test_omega_lock_contract.py` (현재 부재, `result.py:98` docstring이 이미 참조=shipped-false). zero-network로 omegaprompt가 읽는 모든 seam 단언: omega_lock `EvalResult`가 `sample_count` 수용/노출(+ 0.3.0이 target 결과에서 `.fitness`/`.sample_count`만 읽음), `WalkForwardResult` 키 `{train_fitnesses,test_fitnesses,pearson,pearson_status,pearson_computable,test_best_fitness,test_best_params,trade_ratio_scaled}`, `P1Result` 필드 `{grid_best,walk_forward,holdout_result,kc_reports,omega_lock_version}`, `check_kc4` 시그니처, `KCThresholds.min_nonzero_stress_count` + `trade_ratio_min=0.5`/`pearson_min=0.3` + `pure_objective()`. **load-bearing**: omegaprompt 자기 결과의 round-trip `sample_count==n_trials`(omega_lock 필드만 단언하면 이 파손을 못 잡았음). 각 단언에 소비 file:line 명시. CI 최우선 실행. S.
3. **핀 상한** `pyproject.toml:39` `omega-lock>=0.1.4` → `>=0.3.0,<0.4.0`. 다음 breaking bump을 reviewed install로. S.
4. **`omega_lock_version` 스탬프 + drift 발견**. `P1Result.omega_lock_version`(orchestrator.py:200, runtime.py:325에서 in-scope) 또는 importlib.metadata로 모든 CalibrationArtifact에 스탬프. canonical-JSON 결정성 스위트(`artifact_integrity.py:627-654`) 확장해 stamped≠running일 때 발견. 0.1.4 vs 0.3.0 아티팩트가 현재 구분 불가 = 물렸던 그 실패 모드. S.
5. **stale golden 카운터 수정** `reproduce_reference_artifact.py:282-283` (`n_candidates_evaluated=22`/`total_api_calls=344` 하드코딩, 실측 어느 경로와도 불일치: run_p1-only=336/12-unique, full calibrate=592/29-unique). golden이 자기 숫자를 거짓말 = 금지된 "rigorous한 척" 패턴. 실측/파생값으로 교체 후 해시 안정 확인. S.

### High
6. **KC-4 fail-open 발견 + UNDERPOWERED→EXPERIMENT 라우팅**. `status=OK`+`ship=SHIP`인데 `kc4_status != COMPUTED`면 발견 추가, status를 EXPERIMENT/REQUIRES_MANUAL_REVIEW로(ship과 독립). `walkforward.py:188` 플립 금지, blanket fail-closed 금지(6/4 reference 생존). shared-id n이 검정력 임계 이하면 같은 라우팅으로 UNDERPOWERED 강등. 8-state `KC4Status`(`walkforward.py:55-64`) 재사용. **중요 범위**: 이건 omegaprompt **per-ITEM** 게이트(`runtime.py:339-347`) — omega_lock **per-CANDIDATE** pearson과 "reconcile" 시도 금지(다른 통계). M.
7. **Adoption spine: `P1Result.walk_forward/kc_reports/version`을 추가 provenance로 기록 + `pure_objective()` 채택 + 중복 test 재평가 제거**. omegaprompt가 omega_lock에 full walk-forward+KC를 계산시키고(`orchestrator.py:188-200`) 버림(`runtime.py:325`는 grid_best만, `:338`은 test 재평가, `:339`는 자기 per-item walk-forward). **소비**: `result.walk_forward`(+pearson_status/computable/test_best_*)+`kc_reports`+`omega_lock_version`을 아티팩트에 **추가 provenance로 RECORD**. **red-team 교정**: omega_lock pearson = per-CANDIDATE(top-N grid 벡터), omegaprompt kc4_status = per-ITEM(승자 × shared ids). **다른 통계 — 둘 다 기록, 절대 동일시/치환 금지**(치환하면 per-item 검사가 조용히 사라짐). `runtime.py:338` 재평가는 candidate-level verdict/provenance 한정 제거(PC1과 조율 — PC1은 per-item 점수 유지, `:343`이 필요). `pure_objective(pearson_min=resolved_min_kc4)` 묶기(`kill_criteria.py:39-51`, src 미사용 확인)로 KC-3/KC-4b가 6/4 작은 데이터셋을 죽이지 않고 SKIP. `core/walkforward.py` 삭제 금지. M.
8. **[KEEP — 재정당화] `core/walkforward.py` 유지: 0.3.0이 안 주는 통계를 계산**. per-ITEM 상관(`runtime.py:339-347`) ≠ omega_lock per-CANDIDATE(`kill_criteria.py:183-202`). **dedup 불가, parity 증명 불가**(이전 "delete-after-parity"는 종료 불가 경로 — 삭제). 미래 council이 잘못 "reconcile"/삭제하지 않도록 기록. 코드 변경 없음. S.

### Medium
9. **within-run 메모이제이션 (`resolved_params` 키, 중간 latency 마일스톤)**. `runtime.py:319`(neutral)/`:330`(best_train 재평가)/`:338`(test 재평가)가 evaluation_history에 이미 있는 params 재계산(~31%, 기본 calibrate 42 train eval 중 ~13). `PromptTarget.evaluation_history`(`prompt_target.py:298`)를 resolved_params로 색인해 캐시된 EvalResult 재사용. **hash-safe**(target은 계약상 결정적, `omega_lock/target.py:142-148`). 정합성 배당: 현재 `:330`/`:338`이 같은 params에 다른 fitness 기록 가능 → 메모로 run 내 자기일관. **유료 wedge의 중간 납품물**. within-run 한정(cross-run 캐시는 declared-deterministic 플래그 뒤로). M.
10. **죽은 judge-신뢰성 기계 RETRACT**. `runtime.py:918-920`이 judge_quality/endpoint/performance=None 하드코딩 → adaptation.py의 모든 적응 분기(rescore_count, judge_ensemble_shift, _adaptive_min_kc4) 도달 불가인데 문서는 empirical judge gating 광고. **싸고 올바른 절반 = 죽은 분기 삭제 + 문서에서 주장 제거**(활성화는 미배선 probe 필요 = 범위 밖). M.
11. **per-item 루프 결정적 동시화 (XL)**. `prompt_target.py:178-234`에 bounded thread pool, `provider.call`은 sync 유지. **필수**: fan-out 후 `item.id` 정렬로 gather, shared 카운터(`:239-240` total_api_calls/last_usage/run_usage) 누적을 canonical id 순서 post-gather로 — 안 그러면 카운터 레이스로 아티팩트 비용 비결정적. omegaprompt 표면 한정(omega_lock 직렬 루프 건드리지 말 것 — 그 결합이 툴을 깼음). **GATE**: 동시 해시==직렬 해시 테스트. PC1+가드 이후. XL.

### Later
12. **항목별 bootstrap CI (uplift·일반화 gap)** — `runtime.py:389-391`/`fitness.py:29-34`/`walkforward.py:145`의 bare 점추정에 "uplift CI가 0 배제 못함" 플래그. 연기: L + 4-6 item에선 bootstrap 자체가 underpowered(High의 UNDERPOWERED→EXPERIMENT가 대부분 가치 포착). 빌드 시 `test_best_fitness` 소비. L.
13. **[CONDITIONAL] `holdout_target` 배선 + `P1Result.holdout_result` 소비 — n이 3-way split 허용할 때만**. **red-team 교정**: 무료 아님 — `holdout_result`는 holdout_target을 run_p1에 넘겨야 생김(`orchestrator.py:191-196`)인데 `runtime.py:321-323`은 train+test만 넘김. 3-way split = 4-6 item에선 심하게 underpowered, test가 이미 held-out 역할. 데이터셋 규모가 의미있어질 때까지 문서화된 non-goal. M.

---

## 4. 0.3.0이 무효화 (consume, don't build)

- **TPE/`run_p2_tpe` 배선 CUT**: 0.3.0이 제공하지만 omegaprompt의 작은 **이산** 축(variant-pool 카디널리티로 cap, `grid.py:49-54`)에선 절대 발화 안 함 — exhaustive grid가 더 싸고 결정적. 실제 워크로드서 안 도는 메서드 = breadth.
- **undefined-correlation 분류**: `PearsonStatus`/`pearson_computable`(`walk_forward.py:22-69,84-92`)로 upstream 제공 → 빌드 말고 provenance로 기록.
- **held-out 승자 정보**: `test_best_fitness`/`test_best_params`(`walk_forward.py:95-96`)로 무료 → 추출 작업 moot.
- **single-spike stress 퇴화 탐지**: `KCThresholds.min_nonzero_stress_count` + stress_suppressed → 소비.
- **PC3 batch-API 스케줄러 CUT**(동시화와 중복), **PC4 cheap-probe stress pre-pass CUT**(서브셋 랭킹이 grid+해시 바꿈 = 정합성 구멍), **ST5 reproducible-study 래퍼 CUT**(폐기한 universality 인접 marketing; "자기 dep-break 탐지" 가치는 계약 테스트+버전 스탬프가 이미 납품).

---

## 5. 순서 근거 + 다음 한 걸음

순서: **compat(DONE) → 가드 트리오 + golden 카운터(Critical, 전부 S) → KC-4 fail-open + adoption spine(High) → 메모이제이션 → 동시화(XL) → bootstrap CI / holdout(Later)**.

- compat 슬랩은 이미 안착(검증됨). 그래서 **실행 가능한 Critical = 가드 트리오** — `.venv`가 gitignore라 계약 테스트만이 seam을 보고, `result.py:98`의 dangling 참조를 실재화하며, 실현된 파손을 일회성 사건으로 전환.
- 정합성 정직 버그(fail-open KC-4, stale 카운터)가 "자기 자신에 대해 무결한 Prompt CI" 속성을 직접 납품 → 마케팅 전에 신뢰 증명.
- perf는 owner goal-1이지만 싼 정합성 승리 뒤. 단 유료 wedge가 XL에 all-or-nothing 안 되게 PC1 메모(M, hash-safe, ~31% 제거)가 중간 납품물.

**The Assignment**: **가드 트리오 착수** — `tests/test_omega_lock_contract.py`(round-trip `sample_count==n_trials` + 소비 seam 단언) + 핀 상한(`>=0.3.0,<0.4.0`) + `omega_lock_version` 스탬프. 전부 S, 의존성 0. 이게 0.1.4→0.3.0 파손을 재발 불가로 만드는 키스톤이고, 그 다음 모든 작업의 안전 전제.

---

## 다음 단계
`/plan-eng-review` (이 v2 플랜 대상) 또는 가드 트리오 구현 착수.
