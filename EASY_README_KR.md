# omegaprompt — 빠른 시작

> 짧은 버전입니다. 전체 README가 부담스러웠던 분들을 위한 문서예요.

[![PyPI](https://img.shields.io/pypi/v/omegaprompt?color=blue&label=pypi&cacheSeconds=3600)](https://pypi.org/project/omegaprompt/)

전체 문서: [README.md](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md) · English Easy: [EASY_README.md](https://github.com/hibou04-ops/omegaprompt/blob/main/EASY_README.md) · 한국어 Full: [README_KR.md](https://github.com/hibou04-ops/omegaprompt/blob/main/README_KR.md)

---

## 이야기 (60초)

prompt를 튜닝하고 있다고 해봅시다. 예제 입력 20개를 만들고, prompt를 몇 가지 버전으로 그 예제들에 돌려본 뒤, 가장 점수가 좋은 걸 고릅니다. 좋아 보이네요. 그래서 출시합니다.

**이틀째, 그게 방금 교체한 이전 prompt보다 더 못합니다.**

무슨 일이 벌어진 걸까요? 당신의 예제 20개는 아주 작은 표본이었습니다. "이긴" prompt가 이긴 건 더 나아서가 아니라, *바로 그 20개 예제에* 우연히 잘 맞아떨어졌기 때문입니다. 그 20개가 다루지 못한 입력을 가진 실제 트래픽이 들어오는 순간, 당신이 믿었던 점수는 증발해 버립니다.

여기엔 이름이 있습니다. 바로 **overfit**입니다. 그리고 1990년대부터 머신러닝에서 표준이었던 해결책도 있죠 — 예제 일부를 따로 떼어 두고, 나머지로 튜닝한 다음, 떼어 둔 예제에서도 여전히 통할 때만 그 결과를 믿는 겁니다. 대부분의 prompt 워크플로는 이 단계를 건너뜁니다. omegaprompt가 바로 그 단계입니다.

---

## omegaprompt가 실제로 하는 일

세 가지를 넘겨줍니다:

1. **dataset** — 당신의 예제 입력들 (그리고 가능하면, 좋은 답이 어떤 모습인지).
2. **rubric** — 답을 어떻게 채점할지 (규칙, 또는 "모델한테 채점을 시킨다").
3. **후보 prompt 몇 개** — 비교하고 싶은 버전들.

omegaprompt는 dataset을 두 더미로 나눕니다: 튜닝에 써도 되는 **train** 더미와, 절대 *엿보면 안 되는* **held-out** 더미. train 더미만 써서 가장 좋은 prompt를 찾은 다음, 그 우승작을 held-out 더미에서 채점합니다. 우승작이 한 번도 본 적 없는 예제에서도 여전히 잘하면, 좋습니다 — 일반화된 거예요. 무너지면, overfit된 prompt를 production에 닿기 **전에** 잡아낸 겁니다.

이게 전부입니다. 이 페이지의 나머지를 읽어도 되고, 그냥 이것만 기억해도 됩니다: **튜닝에 쓰지 않은 예제에서 당신의 우승 prompt를 다시 시험한다.**

---

## 설치

```bash
pip install omegaprompt
```

---

## 지금 바로 해보기 — API key도, 인터넷도 필요 없음

가짜 in-memory 모델로 완전히 오프라인에서 돌아가는 예제가 내장돼 있어서, 한 푼도 쓰지 않고 key 설정도 없이 결과가 어떤 모양인지 볼 수 있습니다:

```bash
# 1. Produce the example result (deterministic — same numbers every time)
python examples/reference/reproduce_reference_artifact.py

# 2. Read it as a human-friendly report
omegaprompt report examples/reference/reference_artifact.json
```

리포트는 **baseline** 점수(튜닝 안 한 당신의 prompt), **calibrated** 점수(train 더미에서의 우승작), 그리고 **held-out** 점수(우승작이 한 번도 본 적 없는 예제에서의 점수)를 보여줍니다. 핵심은 마지막 두 점수 사이의 간격입니다: 간격이 작으면 일반화된 것이고, 간격이 크면 overfit된 겁니다.

> **이 예제에 관한 솔직한 한마디:** 번들로 들어 있는 dataset의 두 더미는 같은 항목을 공유하지 않습니다. 그래서 "held-out 항목 하나하나가 따라가는가?"라는 가장 세밀한 검사는 맞춰 볼 대상이 없어 건너뛰었다고 보고합니다. 그래도 이 예제는 *간격* 검사가 동작하는 모습은 보여줍니다. 당신의 데이터에서 두 더미가 항목 id를 공유하면, 항목별 검사도 함께 작동합니다. 그러니 이 예제는 "모든 검사가 발동했다"가 아니라 "간격 검사가 통과했다"로 읽으세요.

---

## 결과 읽기

결과는 JSON 파일 하나입니다. 주로 신경 쓸 필드는 두 개예요:

- **`.status`** — `OK`면 prompt가 held-out 검사를 통과했다는 뜻입니다. 그 외엔 통과 못 한 거고요.
- **`.ship_recommendation`** — 쉬운 말로 된 판정: `ship`, `hold`, `experiment`, `block`.

이게 다입니다. 예쁜 버전이 필요하면, `omegaprompt report`가 그 JSON을 pull request에 붙여 넣을 수 있는 읽기 좋은 점수표로 바꿔 줍니다.

---

## 내 prompt에 직접 돌려보기

오프라인 예제가 이해되고 나면, 실제 데이터와 실제 모델로 겨눠 보세요. flag는 멘탈 모델이 자리 잡은 뒤에만 필요합니다:

```bash
export ANTHROPIC_API_KEY=sk-ant-...

omegaprompt calibrate train.jsonl \
  --test test.jsonl \          # the held-out pile it's NOT allowed to tune on
  --rubric rubric.json \       # how to score answers
  --variants variants.json \   # your candidate prompts
  --target-provider anthropic \
  --output artifact.json
```

그런 다음 결과를 똑같은 방식으로 읽으세요: `omegaprompt report artifact.json`.

---

## 두 가지 실행 방식

- **careful (기본값):** 무언가 조용히 대충 넘어가려 하면 — 사실은 일을 못 하는 모델, 충분히 좋지 않은 채점기, 말없는 fallback 같은 것 — 멈춰 서서 알려 줍니다. 모든 게 괜찮은 척하지 않아요. 진짜 일에는 이걸 쓰세요.
- **quick:** 그런 지름길을 통과시켜서 로컬에서 빠르게 이것저것 찔러볼 수 있게 해 줍니다. 하지만 어떤 지름길을 탔는지 전부 적어 두기 때문에, 실행 결과는 여전히 스스로에게 솔직합니다.

---

## 당신이 이미 쓰는 eval 위에서 돌아갑니다

지금 쓰는 설정을 버릴 필요 없습니다. 오늘 좋은 prompt를 *찾는 데* 무엇을 쓰든, omegaprompt는 그 뒤에 자리 잡고 그 도구가 답하지 않는 하나의 질문에 답합니다: *이 우승작이 튜닝하지 않은 데이터에서도 버티는가?* 당신의 기존 예제는 train/held-out의 출처가 되고, 당신의 기존 채점 방식은 rubric이 됩니다.

---

## 쓸 만할 때 (그리고 아닐 때)

**쓸 만할 때**: 당신 말고 누군가가 그 prompt를 신뢰해야 할 때 — 팀원, ops, compliance, 아니면 6개월 뒤의 나 자신 — 또는 prompt 변경을 분위기가 아니라 리뷰 가능한 것으로 만들고 싶을 때.

**과할 때**: 아무도 리뷰하지 않는 1회용 즉석 prompt. playground에서 출력 열 개를 눈으로 훑어보는 게 만족스럽다면, 그렇게 하세요 — 이 도구는 거기서 당신에게 아무것도 더해 주지 않고, 그래도 괜찮습니다.

---

공개 주장과 정확한 deterministic reference 지표는 생성된 [claim ledger](docs/claims/README_CLAIMS.generated.md)에서 추적합니다.

License: Apache 2.0. Copyright (c) 2026 hibou.
