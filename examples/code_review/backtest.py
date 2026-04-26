"""Backtest the N=5 variants pool — measures (not predicts) token efficiency.

Cost: $0
  - Input tokens: tiktoken (cl100k_base) — close approximation for Claude.
  - Output generation: local Ollama model (no API spend).
  - Output tokens (Ollama-tokenized): from Ollama's eval_count.
  - Output tokens (Claude-approx): re-tokenize Ollama's text with tiktoken.

Caveat: local LLM output behavior is a *proxy* for Claude. The relative
ordering across variants is what we trust; absolute counts will differ
between Ollama and Claude. For the user's leverage question
("which variant produces the shortest answer?"), the *delta* between
variants on the same backend is the meaningful signal.

Run: PYTHONIOENCODING=utf-8 python examples/code_review/backtest.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx
import tiktoken


ROOT = Path(__file__).parent
TEST_PATH = ROOT / "test.jsonl"
TRAIN_PATH = ROOT / "train.jsonl"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# tiktoken cl100k_base: same encoding GPT-4 / GPT-3.5-turbo use; commonly used as
# Claude-token approximation in literature. Off by ~5-15% but rank-preserving.
TIKTOKEN_ENCODING = "cl100k_base"


def load_variants(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["system_prompts"]


def load_items(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def count_tiktoken(enc: tiktoken.Encoding, text: str) -> int:
    return len(enc.encode(text))


def call_ollama(model: str, system: str, user: str, max_tokens: int = 1024) -> dict:
    payload = {
        "model": model,
        "system": system,
        "prompt": user,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.0,
        },
    }
    t0 = time.time()
    r = httpx.post(OLLAMA_URL, json=payload, timeout=180.0)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    return {
        "text": data.get("response", ""),
        "input_tokens_local": data.get("prompt_eval_count", 0),
        "output_tokens_local": data.get("eval_count", 0),
        "elapsed_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=str(ROOT / "variants.json"))
    parser.add_argument("--model", default="exaone3.5:7.8b")
    parser.add_argument("--output", default=str(ROOT / "backtest_results.json"))
    args = parser.parse_args()

    variants_path = Path(args.variants)
    output_path = Path(args.output)
    model = args.model

    variants = load_variants(variants_path)
    test_items = load_items(TEST_PATH)
    train_items = load_items(TRAIN_PATH)
    all_items = test_items + train_items

    print(f"Variants file: {variants_path}")
    print(f"Model: {model}")
    print(f"Output: {output_path}")
    print(f"Variants: N={len(variants)}")
    print(f"Items: {len(all_items)} ({len(test_items)} test + {len(train_items)} train)")
    print(f"Total Ollama calls: {len(variants) * len(all_items)}")
    print()

    enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)

    print(f"=== Step 1: Input token counts (tiktoken {TIKTOKEN_ENCODING}, $0) ===")
    sample_input = all_items[0]["input"]
    input_counts = []
    for i, sp in enumerate(variants):
        sys_only = count_tiktoken(enc, sp)
        sys_plus_sample = count_tiktoken(enc, sp + "\n\n" + sample_input)
        input_counts.append(sys_only)
        print(f"V{i}: system_only={sys_only:3d} tok | system+sample={sys_plus_sample:4d} tok")
    print()

    print(f"=== Step 2: Ollama generation backtest ({model}) ===")
    results = []
    for vi, sp in enumerate(variants):
        per_outs_tt = []
        per_elapsed = []
        for item in all_items:
            try:
                r = call_ollama(model, sp, item["input"])
            except Exception as exc:
                print(f"  V{vi} | {item['id']:25s} | ERROR: {exc}")
                continue
            text = r["text"]
            tt_out = count_tiktoken(enc, text)
            per_outs_tt.append(tt_out)
            per_elapsed.append(r["elapsed_sec"])
            results.append(
                {
                    "variant_idx": vi,
                    "item_id": item["id"],
                    "output_tokens_ollama": r["output_tokens_local"],
                    "output_tokens_tiktoken": tt_out,
                    "input_tokens_ollama": r["input_tokens_local"],
                    "elapsed_sec": r["elapsed_sec"],
                    "text": text,
                }
            )
            print(
                f"  V{vi} | {item['id']:25s} | "
                f"out_tt={tt_out:4d} | out_oll={r['output_tokens_local']:4d} | "
                f"{r['elapsed_sec']:5.1f}s"
            )
        if per_outs_tt:
            print(
                f"V{vi} subtotal (tiktoken): mean={statistics.mean(per_outs_tt):.0f} "
                f"sd={statistics.stdev(per_outs_tt) if len(per_outs_tt) > 1 else 0:.0f} "
                f"min={min(per_outs_tt)} max={max(per_outs_tt)}\n"
            )

    print("=" * 78)
    print("=== Output token summary (tiktoken-counted, Claude-approx) ===")
    print(f"{'V':<3} {'mean':>7} {'sd':>6} {'min':>5} {'max':>5} {'sum':>7} {'vs V0':>10}")
    by_variant: dict[int, list[int]] = {vi: [] for vi in range(len(variants))}
    for r in results:
        by_variant[r["variant_idx"]].append(r["output_tokens_tiktoken"])

    v0_outs = by_variant[0]
    v0_mean = statistics.mean(v0_outs) if v0_outs else 0.0
    summary_rows = []
    for vi in range(len(variants)):
        outs = by_variant[vi]
        if not outs:
            continue
        mean = statistics.mean(outs)
        sd = statistics.stdev(outs) if len(outs) > 1 else 0
        delta_pct = ((mean - v0_mean) / v0_mean * 100) if v0_mean else 0.0
        print(
            f"V{vi:<2} {mean:7.0f} {sd:6.0f} {min(outs):5d} {max(outs):5d} "
            f"{sum(outs):7d} {delta_pct:+9.1f}%"
        )
        summary_rows.append(
            {
                "variant_idx": vi,
                "input_tokens_tiktoken": input_counts[vi],
                "output_mean_tiktoken": mean,
                "output_sd_tiktoken": sd,
                "output_min_tiktoken": min(outs),
                "output_max_tiktoken": max(outs),
                "output_sum_tiktoken": sum(outs),
                "delta_vs_v0_pct": delta_pct,
                "n_items": len(outs),
            }
        )

    print()
    print("=== Cost projection (per 1000 review-style calls, Opus 4.7 rate) ===")
    OPUS_INPUT_RATE = 15.0
    OPUS_OUTPUT_RATE = 75.0
    print(f"  Output rate assumed: ${OPUS_OUTPUT_RATE}/M tokens (Opus output)")
    print(f"  Input rate assumed:  ${OPUS_INPUT_RATE}/M tokens (Opus input, ignored if cached)")
    print()
    print(f"{'V':<3} {'in_tok':>7} {'out_mean':>9} {'cost_1k_calls':>15}")
    cost_rows = []
    for vi, row in enumerate(summary_rows):
        in_tok = row["input_tokens_tiktoken"]
        out_mean = row["output_mean_tiktoken"]
        cost_per_call_first = (in_tok / 1_000_000) * OPUS_INPUT_RATE + (out_mean / 1_000_000) * OPUS_OUTPUT_RATE
        cost_1k_calls = cost_per_call_first * 1000
        print(f"V{vi:<2} {in_tok:7d} {out_mean:9.0f} ${cost_1k_calls:14.2f}")
        cost_rows.append({"variant_idx": vi, "cost_per_1k_calls_usd": cost_1k_calls})

    output_path.write_text(
        json.dumps(
            {
                "config": {
                    "ollama_model": model,
                    "tiktoken_encoding": TIKTOKEN_ENCODING,
                    "n_items": len(all_items),
                    "n_variants": len(variants),
                    "variants_file": str(variants_path),
                    "opus_input_rate_per_M": OPUS_INPUT_RATE,
                    "opus_output_rate_per_M": OPUS_OUTPUT_RATE,
                },
                "summary": summary_rows,
                "cost_projection_per_1k": cost_rows,
                "raw": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nFull results saved: {output_path}")


if __name__ == "__main__":
    main()
