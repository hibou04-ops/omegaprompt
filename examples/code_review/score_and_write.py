import json
from collections import Counter

per_call = []

# ==================== VARIANT 0 ====================
per_call.append({'variant_idx': 0, 'item_id': 'c10_clean_code', 'verdict': 'fabricated', 'hard_gate_violations': [], 'notes': 'Invented empty-input, overflow, modulo adjustment issues. No NO_ISSUE.'})
per_call.append({'variant_idx': 0, 'item_id': 'c1_factorial', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found range(n) bug, multiply by 0, fix range(1,n+1).'})
per_call.append({'variant_idx': 0, 'item_id': 'c2_counter_race', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found race condition, threading.Lock fix.'})
per_call.append({'variant_idx': 0, 'item_id': 'c3_sql_injection', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found SQL injection, parameterized queries fix.'})
per_call.append({'variant_idx': 0, 'item_id': 'c4_file_leak', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found resource leak, context manager fix.'})
per_call.append({'variant_idx': 0, 'item_id': 'c5_n_plus_1', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Found SQL injection but missed N+1 pattern. Vague performance note only.'})
per_call.append({'variant_idx': 0, 'item_id': 'c6_ordered_check', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Found KeyError secondary, missed assert -O stripping (primary).'})
per_call.append({'variant_idx': 0, 'item_id': 'c7_concat_loop', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found O(n^2) string concat, str.join fix.'})
per_call.append({'variant_idx': 0, 'item_id': 'c8_mutable_default', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found mutable default list, None fix.'})
per_call.append({'variant_idx': 0, 'item_id': 'c9_late_binding', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found late-binding closure, lambda v=v fix.'})
per_call.append({'variant_idx': 0, 'item_id': 't1_off_by_one_slice', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found OBO slicing, items[-n:] fix.'})
per_call.append({'variant_idx': 0, 'item_id': 't2_mutable_default_dict', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found mutable default dict, None fix.'})
per_call.append({'variant_idx': 0, 'item_id': 't3_command_injection', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found shell=True injection risk, list args fix.'})
per_call.append({'variant_idx': 0, 'item_id': 't4_forgotten_await', 'verdict': 'wrong_root_cause', 'hard_gate_violations': [], 'notes': 'Named bug is error handling not missing await. Fix code incidentally adds await.'})
per_call.append({'variant_idx': 0, 'item_id': 't5_dict_iter_mutation', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found dict mutation during iteration, safe fix.'})
per_call.append({'variant_idx': 0, 'item_id': 't6_clean_code', 'verdict': 'fabricated', 'hard_gate_violations': [], 'notes': 'Invented error message clarity and None type edge cases. No NO_ISSUE.'})

# ==================== VARIANT 1 ====================
per_call.append({'variant_idx': 1, 'item_id': 'c10_clean_code', 'verdict': 'fabricated', 'hard_gate_violations': ['no_fabricated_claims'], 'notes': 'Invents false math: modulo-per-step obscures cumulative differences vs at-end. Both are equivalent. Fabricated bug with false reasoning.'})
per_call.append({'variant_idx': 1, 'item_id': 'c1_factorial', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found range(n) bug, multiply by 0, fix range(1,n+1).'})
per_call.append({'variant_idx': 1, 'item_id': 'c2_counter_race', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found race condition, threading.Lock fix.'})
per_call.append({'variant_idx': 1, 'item_id': 'c3_sql_injection', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found SQL injection, parameterized queries fix.'})
per_call.append({'variant_idx': 1, 'item_id': 'c4_file_leak', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found resource leak, context manager fix.'})
per_call.append({'variant_idx': 1, 'item_id': 'c5_n_plus_1', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Found SQL injection but missed N+1 query pattern.'})
per_call.append({'variant_idx': 1, 'item_id': 'c6_ordered_check', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Found KeyError secondary, missed assert -O stripping (primary).'})
per_call.append({'variant_idx': 1, 'item_id': 'c7_concat_loop', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found O(n^2) string concat, str.join fix.'})
per_call.append({'variant_idx': 1, 'item_id': 'c8_mutable_default', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found mutable default list, None fix.'})
per_call.append({'variant_idx': 1, 'item_id': 'c9_late_binding', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found late-binding closure, all print 3, lambda v=v fix.'})
per_call.append({'variant_idx': 1, 'item_id': 't1_off_by_one_slice', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found OBO index issue, items[-n:] fix.'})
per_call.append({'variant_idx': 1, 'item_id': 't2_mutable_default_dict', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found mutable default dict, None fix.'})
per_call.append({'variant_idx': 1, 'item_id': 't3_command_injection', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found shell=True injection risk, list args fix.'})
per_call.append({'variant_idx': 1, 'item_id': 't4_forgotten_await', 'verdict': 'wrong_root_cause', 'hard_gate_violations': [], 'notes': 'Named bug is error handling, not missing await. Fix incidentally adds await.'})
per_call.append({'variant_idx': 1, 'item_id': 't5_dict_iter_mutation', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found dict mutation during iteration, list-of-keys fix.'})
per_call.append({'variant_idx': 1, 'item_id': 't6_clean_code', 'verdict': 'fabricated', 'hard_gate_violations': [], 'notes': 'Invented lo==hi edge case as logical issue. Clamp handles this correctly.'})

# ==================== VARIANT 2 ====================
per_call.append({'variant_idx': 2, 'item_id': 'c10_clean_code', 'verdict': 'fabricated', 'hard_gate_violations': [], 'notes': 'Invented empty-data handling issue. Returns 0 for empty bytes is correct behavior.'})
per_call.append({'variant_idx': 2, 'item_id': 'c1_factorial', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found range(n) bug, fix range(1,n+1).'})
per_call.append({'variant_idx': 2, 'item_id': 'c2_counter_race', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found race condition, threading.Lock fix.'})
per_call.append({'variant_idx': 2, 'item_id': 'c3_sql_injection', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found SQL injection, parameterized queries fix.'})
per_call.append({'variant_idx': 2, 'item_id': 'c4_file_leak', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Primary issue named as missing mode specifier, not resource leak. Fix correct but root cause wrong.'})
per_call.append({'variant_idx': 2, 'item_id': 'c5_n_plus_1', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Found SQL injection but missed N+1 pattern.'})
per_call.append({'variant_idx': 2, 'item_id': 'c6_ordered_check', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Added type checks only. Missed assert -O stripping (primary) and KeyError ordering.'})
per_call.append({'variant_idx': 2, 'item_id': 'c7_concat_loop', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found O(n^2) string concat, str.join fix.'})
per_call.append({'variant_idx': 2, 'item_id': 'c8_mutable_default', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found mutable default list, None fix.'})
per_call.append({'variant_idx': 2, 'item_id': 'c9_late_binding', 'verdict': 'wrong_root_cause', 'hard_gate_violations': [], 'notes': 'Misidentified problem as values mutation. Proposed fix (nested def) still has same late-binding bug.'})
per_call.append({'variant_idx': 2, 'item_id': 't1_off_by_one_slice', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Framed as n-exceeds-len only. Missed core -1 OBO (returns n+1 items). Fix correct.'})
per_call.append({'variant_idx': 2, 'item_id': 't2_mutable_default_dict', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found mutable default dict, None fix.'})
per_call.append({'variant_idx': 2, 'item_id': 't3_command_injection', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found shell=True injection risk, os.listdir fix.'})
per_call.append({'variant_idx': 2, 'item_id': 't4_forgotten_await', 'verdict': 'wrong_root_cause', 'hard_gate_violations': [], 'notes': 'Named issue is error handling. Missing await not identified. Fix incidentally adds await.'})
per_call.append({'variant_idx': 2, 'item_id': 't5_dict_iter_mutation', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found dict mutation during iteration, list-of-keys fix.'})
per_call.append({'variant_idx': 2, 'item_id': 't6_clean_code', 'verdict': 'fabricated', 'hard_gate_violations': [], 'notes': 'Invented clarity/edge-case issue. Proposed error message modification. Strict: any invented issue = fabricated.'})

# ==================== VARIANT 3 ====================
per_call.append({'variant_idx': 3, 'item_id': 'c10_clean_code', 'verdict': 'fabricated', 'hard_gate_violations': [], 'notes': 'Invents empty input and overflow issues before NO_ISSUE on core logic. Strict: any invented issue = fabricated.'})
per_call.append({'variant_idx': 3, 'item_id': 'c1_factorial', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found range(n) bug, multiply by 0 for n>0, fix range(1,n+1).'})
per_call.append({'variant_idx': 3, 'item_id': 'c2_counter_race', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found race condition, non-atomic increment, threading.Lock fix.'})
per_call.append({'variant_idx': 3, 'item_id': 'c3_sql_injection', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found SQL injection, parameterized queries fix.'})
per_call.append({'variant_idx': 3, 'item_id': 'c4_file_leak', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found resource leak from lack of explicit close, context manager fix.'})
per_call.append({'variant_idx': 3, 'item_id': 'c5_n_plus_1', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Found SQL injection but missed N+1 query antipattern.'})
per_call.append({'variant_idx': 3, 'item_id': 'c6_ordered_check', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Found KeyError secondary. Missed assert -O stripping (primary).'})
per_call.append({'variant_idx': 3, 'item_id': 'c7_concat_loop', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found O(n^2) string concat, str.join fix.'})
per_call.append({'variant_idx': 3, 'item_id': 'c8_mutable_default', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found mutable default list, None fix.'})
per_call.append({'variant_idx': 3, 'item_id': 'c9_late_binding', 'verdict': 'missed', 'hard_gate_violations': [], 'notes': 'Claims no issue for immutable integers. Wrong: all print 3 regardless of type.'})
per_call.append({'variant_idx': 3, 'item_id': 't1_off_by_one_slice', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Focused on n>len edge case, not the -1 OBO. Fix correct but root cause explanation off.'})
per_call.append({'variant_idx': 3, 'item_id': 't2_mutable_default_dict', 'verdict': 'wrong_root_cause', 'hard_gate_violations': [], 'notes': 'Describes bug as state being lost (wrong: state accumulates). Opposite direction from actual bug.'})
per_call.append({'variant_idx': 3, 'item_id': 't3_command_injection', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found shell=True injection risk, list args fix.'})
per_call.append({'variant_idx': 3, 'item_id': 't4_forgotten_await', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Explicitly identified missing await. Cited specific line. async-with + await fix.'})
per_call.append({'variant_idx': 3, 'item_id': 't5_dict_iter_mutation', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found dict mutation during iteration, dict comprehension fix.'})
per_call.append({'variant_idx': 3, 'item_id': 't6_clean_code', 'verdict': 'fabricated', 'hard_gate_violations': [], 'notes': 'Invented error message clarity issue. Strict: any invented issue = fabricated.'})

# ==================== VARIANT 4 ====================
per_call.append({'variant_idx': 4, 'item_id': 'c10_clean_code', 'verdict': 'fabricated', 'hard_gate_violations': [], 'notes': 'Mentions empty-data issue before NO_ISSUE. Strict: any invented issue = fabricated.'})
per_call.append({'variant_idx': 4, 'item_id': 'c1_factorial', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found bug. Fix i+1 over range(n) is valid (1..n). Also range(1,n+1) version provided.'})
per_call.append({'variant_idx': 4, 'item_id': 'c2_counter_race', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found race condition, threading.Lock fix.'})
per_call.append({'variant_idx': 4, 'item_id': 'c3_sql_injection', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found SQL injection, parameterized queries fix.'})
per_call.append({'variant_idx': 4, 'item_id': 'c4_file_leak', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Primary issue named as missing mode specifier. Resource leak is real primary bug. Context manager mentioned but not as main fix.'})
per_call.append({'variant_idx': 4, 'item_id': 'c5_n_plus_1', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Found SQL injection but missed N+1 pattern.'})
per_call.append({'variant_idx': 4, 'item_id': 'c6_ordered_check', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Only focused on items structure. Missed assert -O stripping and KeyError ordering.'})
per_call.append({'variant_idx': 4, 'item_id': 'c7_concat_loop', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found O(n^2) string concat, str.join fix.'})
per_call.append({'variant_idx': 4, 'item_id': 'c8_mutable_default', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found mutable default list, None fix.'})
per_call.append({'variant_idx': 4, 'item_id': 'c9_late_binding', 'verdict': 'missed', 'hard_gate_violations': [], 'notes': 'Claims no issue for immutable integers. Wrong: all print 3. Explicitly says NO_ISSUE.'})
per_call.append({'variant_idx': 4, 'item_id': 't1_off_by_one_slice', 'verdict': 'partial', 'hard_gate_violations': [], 'notes': 'Framed as bounds-checking issue, missed -1 OBO (returns n+1 items). Fix correct.'})
per_call.append({'variant_idx': 4, 'item_id': 't2_mutable_default_dict', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found mutable default dict, None fix.'})
per_call.append({'variant_idx': 4, 'item_id': 't3_command_injection', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found shell=True injection risk, os.listdir fix.'})
per_call.append({'variant_idx': 4, 'item_id': 't4_forgotten_await', 'verdict': 'wrong_root_cause', 'hard_gate_violations': [], 'notes': 'Named issue is error handling, not missing await. Fix incidentally adds await.'})
per_call.append({'variant_idx': 4, 'item_id': 't5_dict_iter_mutation', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Found dict mutation during iteration, list-of-keys fix.'})
per_call.append({'variant_idx': 4, 'item_id': 't6_clean_code', 'verdict': 'correct', 'hard_gate_violations': [], 'notes': 'Correctly said NO_ISSUE for clean code.'})

# ==================== VERIFY ====================
print(f'Total per_call entries: {len(per_call)}')
pairs = set((e['variant_idx'], e['item_id']) for e in per_call)
print(f'Unique pairs: {len(pairs)}')

CLEAN_ITEMS = {'c10_clean_code', 't6_clean_code'}
BUGGY_ITEMS = {'c1_factorial', 'c2_counter_race', 'c3_sql_injection', 'c4_file_leak', 'c5_n_plus_1',
               'c6_ordered_check', 'c7_concat_loop', 'c8_mutable_default', 'c9_late_binding',
               't1_off_by_one_slice', 't2_mutable_default_dict', 't3_command_injection',
               't4_forgotten_await', 't5_dict_iter_mutation'}

# ==================== PER-VARIANT AGGREGATES ====================
per_variant = []
for v in range(5):
    entries = [e for e in per_call if e['variant_idx'] == v]
    vc = Counter(e['verdict'] for e in entries)

    buggy_entries = [e for e in entries if e['item_id'] in BUGGY_ITEMS]
    clean_entries = [e for e in entries if e['item_id'] in CLEAN_ITEMS]

    total_buggy = len(buggy_entries)  # 14
    total_clean = len(clean_entries)  # 2

    correct_on_buggy = sum(1 for e in buggy_entries if e['verdict'] == 'correct')
    # precision: among judged (not missed), how many are correct
    judged_buggy = sum(1 for e in buggy_entries if e['verdict'] != 'missed')
    buggy_precision = correct_on_buggy / judged_buggy if judged_buggy > 0 else 0.0
    buggy_recall = correct_on_buggy / total_buggy if total_buggy > 0 else 0.0

    fabricated_clean = sum(1 for e in clean_entries if e['verdict'] == 'fabricated')
    clean_fpr = fabricated_clean / total_clean if total_clean > 0 else 0.0

    no_fab_violations = sum(len([g for g in e['hard_gate_violations'] if g == 'no_fabricated_claims']) for e in entries)
    rewrite_violations = sum(len([g for g in e['hard_gate_violations'] if g == 'review_not_silent_rewrite']) for e in entries)

    verdicts_dict = {
        'correct': vc.get('correct', 0),
        'missed': vc.get('missed', 0),
        'fabricated': vc.get('fabricated', 0),
        'wrong_root_cause': vc.get('wrong_root_cause', 0),
        'partial': vc.get('partial', 0),
    }

    notes_map = {
        0: "Verbose (avg ~600 tok). Correctly catches most bugs but fabricates on both clean items. Consistently misses assert-O stripping and t4 missing await primary cause.",
        1: "Focused style. Similar recall to V0 but shorter. V1 c10 is unique: invents false mathematical claim about modulo equivalence -- a hard no_fabricated_claims violation. Both clean items fabricated.",
        2: "Compact output_format style. Lowest token count. c9 wrong root cause (proposed fix still broken). c4 misidentifies root cause. Still fabricates on both clean items.",
        3: "Verbose with NO_ISSUE codas. Only variant to correctly find t4 missing await. But c9 and t2 mutable_default_dict wrong. Both clean items fabricated.",
        4: "Hybrid NO_ISSUE variant. Correctly handles t6 (bare NO_ISSUE). Still fabricates on c10. c9 missed entirely. Most concise among verbose variants.",
    }

    per_variant.append({
        'variant_idx': v,
        'verdicts': verdicts_dict,
        'buggy_precision': round(buggy_precision, 4),
        'buggy_recall': round(buggy_recall, 4),
        'clean_false_positive_rate': round(clean_fpr, 4),
        'no_fabricated_claims_violations': no_fab_violations,
        'review_not_silent_rewrite_violations': rewrite_violations,
        'qualitative_notes': notes_map[v],
    })

    print(f"V{v}: verdicts={verdicts_dict} precision={buggy_precision:.3f} recall={buggy_recall:.3f} clean_fpr={clean_fpr:.2f}")

# ==================== SUMMARY ====================
summary = [
    "Best overall variant: V0 (buggy_precision=0.786, buggy_recall=0.786, clean_fpr=1.0, 0 hard-gate violations). V0 and V1 are tied on recall/precision but V1 has a unique no_fabricated_claims violation (false math on c10), making V0 the cleanest single answer.",
    "V4 (NO_ISSUE hybrid) partially handles clean code: correctly emits bare NO_ISSUE for t6_clean_code but still fabricates on c10_clean_code (hedges with an invented empty-bytes concern before the NO_ISSUE coda). Clean FPR is 0.5, the best of any variant, but not a clean win.",
    "All 5 variants fabricate on c10_clean_code (checksum). The universal trigger is the empty-bytes edge case -- the model invents an issue (returning 0 for empty input) that is mathematically correct behavior.",
    "c9_late_binding is the most inconsistently handled buggy item: V3 and V4 explicitly mark NO_ISSUE (wrong, bug is real), V2 proposes a broken fix, V0 and V1 correctly identify and fix it.",
    "t4_forgotten_await (missing await) is systematically mislabeled as an error-handling gap by V0/V1/V2/V4; only V3 explicitly names the missing await as the root cause.",
    "c6_ordered_check: assert -O stripping (assert disabled by python -O) is the primary bug per reference. Zero variants identified it; all treated the KeyError ordering issue as the primary concern, resulting in partial scores across the board.",
    "c5_n_plus_1: all variants found the SQL injection issue but none explicitly identified the N+1 query antipattern (N separate DB round trips), resulting in universal partial scores.",
    "V1 c10 is the only no_fabricated_claims hard-gate violation: it invents a false mathematical claim that modulo-per-step obscures cumulative differences versus modulo-at-end (the two are arithmetically equivalent).",
    "Quality vs token tradeoff is real: V2 (compact ~100-200 tok/review) has the most wrong_root_cause errors and lowest precision (0.571). Verbose V0/V1 (~500-700 tok) achieve 0.786 on both metrics but token cost is 3-5x higher.",
    "No variant achieved a clean false-positive rate below 0.5. The 7B model has a systematic tendency to over-review clean code, inventing minor style/edge-case issues rather than confidently saying NO_ISSUE.",
]

output = {
    "config": {"items_total": 16, "variants_total": 5, "judge_method": "agent_review"},
    "per_call": per_call,
    "per_variant": per_variant,
    "summary": summary,
}

out_path = 'C:/Users/hibou/omegaprompt/examples/code_review/quality_review.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nWrote {out_path}")
print(f"per_call: {len(per_call)} entries")
print(f"per_variant: {len(per_variant)} entries")
