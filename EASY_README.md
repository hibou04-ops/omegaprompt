# omegaprompt — Easy Start

> The short version, for people who found the full README intimidating.

[![PyPI](https://img.shields.io/pypi/v/omegaprompt?color=blue&label=pypi&cacheSeconds=3600)](https://pypi.org/project/omegaprompt/)

Full doc: [README.md](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md) · 한국어 Easy: [EASY_README_KR.md](https://github.com/hibou04-ops/omegaprompt/blob/main/EASY_README_KR.md) · 한국어 Full: [README_KR.md](https://github.com/hibou04-ops/omegaprompt/blob/main/README_KR.md)

---

## The story (60 seconds)

You're tuning a prompt. You write 20 example inputs, try a few versions of your prompt against them, and pick the one that scores best. It looks great. You ship it.

**On day two, it's doing worse than the prompt you replaced.**

What happened? Your 20 examples were a tiny sample. The prompt that "won" didn't win because it's better — it won because it happened to fit *those exact 20 examples*. The moment real traffic showed up with inputs your 20 didn't cover, the score you trusted evaporated.

This has a name: **overfitting**. And it has a fix that's been standard in machine learning since the 1990s — you hold some examples back, tune on the rest, and only trust a result if it still holds up on the examples you held back. Most prompt workflows skip this step. omegaprompt is that step.

---

## What omegaprompt actually does

You hand it three things:

1. **A dataset** — your example inputs (and, ideally, what a good answer looks like).
2. **A rubric** — how to score an answer (rules, or "ask a model to grade it").
3. **A few candidate prompts** — the versions you want to compare.

It splits your dataset into two piles: a **train** pile it's allowed to tune on, and a **held-out** pile it is *not* allowed to peek at. It finds the best prompt using only the train pile, then scores that winner on the held-out pile. If the winner still does well on examples it never saw, great — it generalized. If it falls apart, you just caught an overfit prompt **before** it hit production.

That's the whole idea. You can read the rest of this page, or you can just remember: **it re-tests your winning prompt on examples it never tuned on.**

---

## Install

```bash
pip install omegaprompt
```

---

## Try it right now — no API key, no internet

There's a built-in example that runs entirely offline using fake in-memory models, so you can see the shape of the result without spending a cent or setting up a key:

```bash
# 1. Produce the example result (deterministic — same numbers every time)
python examples/reference/reproduce_reference_artifact.py

# 2. Read it as a human-friendly report
omegaprompt report examples/reference/reference_artifact.json
```

The report shows you a **baseline** score (your prompt with no tuning), a **calibrated** score (the winner on the train pile), and the **held-out** score (the winner on examples it never saw). The gap between those last two is the whole point: a small gap means it generalized; a big gap means it overfit.

> **One honest note about this example:** the bundled dataset's two piles don't share the same items, so the finest-grained "does each held-out item track?" check has nothing to line up and reports that it was skipped. The example still shows the *gap* check working. On your own data, if the two piles share item ids, the per-item check kicks in too. So read the example as "the gap check passed," not "every check fired."

---

## Reading the result

The result is one JSON file. You mostly care about two fields:

- **`.status`** — `OK` means the prompt cleared the held-out checks. Anything else means it didn't.
- **`.ship_recommendation`** — a plain-English call: `ship`, `hold`, `experiment`, or `block`.

That's it. If you want the pretty version, `omegaprompt report` turns that JSON into a readable scorecard you can paste into a pull request.

---

## Running it on your own prompt

Once the offline example makes sense, point it at real data with a real model. You only need the flags after the mental model has clicked:

```bash
export ANTHROPIC_API_KEY=sk-ant-...

omegaprompt calibrate train.jsonl \
  --test test.jsonl \          # the held-out pile it's NOT allowed to tune on
  --rubric rubric.json \       # how to score answers
  --variants variants.json \   # your candidate prompts
  --target-provider anthropic \
  --output artifact.json
```

Then read the result the same way: `omegaprompt report artifact.json`.

---

## Two ways to run

- **Careful (the default):** if anything tries to quietly cut a corner — a model that can't really do the job, a grader that isn't good enough, a silent fallback — it stops and tells you instead of pretending everything's fine. Use this for anything real.
- **Quick:** lets those shortcuts through so you can poke around fast locally, but it writes down every shortcut it took so the run is still honest about itself.

---

## It runs on top of whatever eval you already use

You don't have to throw away your current setup. Whatever you use to *find* a good prompt today, omegaprompt sits after it and answers the one question that tool doesn't: *does this winner hold up on data it wasn't tuned on?* Your existing examples become the train/held-out source; your existing scoring becomes the rubric.

---

## When it's worth it (and when it isn't)

**Worth it** when someone other than you has to trust the prompt — a teammate, ops, compliance, or future-you six months from now — or when you want a prompt change to be reviewable instead of a vibe.

**Overkill** for a one-off throwaway prompt with nobody reviewing it. If you're happy eyeballing ten outputs in a playground, do that — this tool buys you nothing there, and that's fine.

---

Public claims and exact deterministic reference metrics are tracked in the generated [claim ledger](docs/claims/README_CLAIMS.generated.md).

License: Apache 2.0. Copyright (c) 2026 hibou.
