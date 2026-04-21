"""LLM-as-judge using a provider's STRICT_SCHEMA path.

Single call per item: the judge prompt carries the rubric, input,
reference (if any), and the target's response. The provider's native
schema-enforcement path parses the response into a :class:`JudgeResult`.
A malformed judge response raises a ValidationError before the
calibration loop sees it.
"""

from __future__ import annotations

import json

from omegaprompt.domain.dataset import DatasetItem
from omegaprompt.domain.enums import OutputBudgetBucket, ResponseSchemaMode
from omegaprompt.domain.judge import JudgeResult, JudgeRubric
from omegaprompt.judges.base import JudgeError
from omegaprompt.providers.base import LLMProvider, ProviderError, ProviderRequest


JUDGE_SYSTEM_PROMPT = r"""You are the judge in an omegaprompt calibration run. Your job is narrow: read the rubric, read the input/response pair, and return a structured JSON score. Do not attempt to improve the response, do not add commentary beyond the `notes` field, do not guess at the user's intent beyond what the rubric asks.

## Inputs

Each request contains, in order:

1. `<rubric>` - the judge rubric: a list of scoring dimensions (each with name, description, weight, and integer scale) and a list of hard gates (binary predicates with name and description).
2. `<input>` - the prompt the target model received.
3. `<reference>` - optional. If present, the expected or reference output the judge may consult. If absent, score the response on its own merits per the rubric.
4. `<response>` - the target model's response to be scored.

## Scoring rules

For each rubric dimension:

- Read the dimension's description carefully. The description defines what "high score" and "low score" mean for that dimension.
- Assign an integer score within the dimension's declared scale (inclusive on both ends).
- If the response is incomplete or truncated, score what is present - do not extrapolate. Note the truncation in the `notes` field.

For each rubric hard gate:

- Read the gate's description. The gate is a binary: `true` = pass, `false` = fail.
- Typical gate patterns:
  - `no_refusal`: did the response actually attempt the task, or did it refuse / deflect / redirect?
  - `format_valid`: does the response match the required format (valid JSON / valid code / matches a regex / contains the required sections)?
  - `no_safety_violation`: is the response free of content that violates the stated safety requirements?
- Evaluate strictly against the description. A response that *mostly* addresses the task but includes a refusal clause still fails `no_refusal`.

## Output

Return exactly one JSON object matching the caller's schema:

- `scores`: object mapping each dimension name to its integer score.
- `gate_results`: object mapping each gate name to its boolean outcome.
- `notes`: brief free-form justification (one to three sentences). Useful for later human inspection of outlier classifications. Not used in the fitness calculation.

Do not wrap the JSON in markdown fences. Do not add commentary before or after. Do not output anything other than the JSON object.

## Anti-patterns - do not do these

- Do not score a dimension that is not in the rubric. Unknown dimension names are ignored.
- Do not return a score outside the declared scale. Clamp internally if your first draft is out of range.
- Do not leave a rubric dimension or gate unfilled. Every dimension gets a score; every gate gets a boolean.
- Do not use `notes` to argue with the rubric. If you think the rubric is ambiguous, score as best you can and note the ambiguity briefly.
- Do not generalize. Each request is independent; do not carry over judgments across items.

## Calibration hygiene

You are part of an automated calibration loop. A miscalibrated judge produces a miscalibrated search. Three specific hygiene rules:

1. **Anchor on the scale's midpoint.** Use the full range. A judge that only gives 4 or 5 on a 1-5 scale provides no gradient for the searcher to follow.
2. **Agree with yourself.** Two semantically-equivalent responses should score the same. If you find yourself scoring similar-quality responses differently, re-read the rubric before finalizing.
3. **Reference is evidence, not ground truth.** When `<reference>` is present, treat it as one valid solution, not the only one. A response that differs from the reference but satisfies the rubric still scores well.

Your reply is the JSON object. Nothing else.
"""


def _build_user_payload(
    rubric: JudgeRubric,
    item: DatasetItem,
    target_response: str,
) -> str:
    rubric_json = json.dumps(
        {
            "dimensions": [d.model_dump() for d in rubric.dimensions],
            "hard_gates": [g.model_dump() for g in rubric.hard_gates if g.evaluator == "judge"],
        },
        ensure_ascii=False,
        indent=2,
    )
    ref_block = (
        f"<reference>\n{item.reference}\n</reference>\n\n" if item.reference else ""
    )
    return (
        f"<rubric>\n{rubric_json}\n</rubric>\n\n"
        f"<input>\n{item.input}\n</input>\n\n"
        f"{ref_block}"
        f"<response>\n{target_response}\n</response>"
    )


class LLMJudge:
    """LLM-as-judge backed by any provider that supports STRICT_SCHEMA."""

    name = "llm"

    def __init__(
        self,
        *,
        provider: LLMProvider,
        output_budget: OutputBudgetBucket = OutputBudgetBucket.SMALL,
    ) -> None:
        self.provider = provider
        self.output_budget = output_budget

    def score(
        self,
        *,
        rubric: JudgeRubric,
        item: DatasetItem,
        target_response: str,
    ) -> tuple[JudgeResult, dict[str, int]]:
        payload = _build_user_payload(rubric, item, target_response)

        request = ProviderRequest(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_message=payload,
            reasoning_profile=rubric_reasoning_profile(),
            output_budget=self.output_budget,
            response_schema_mode=ResponseSchemaMode.STRICT_SCHEMA,
            output_schema=JudgeResult,
        )

        try:
            response = self.provider.call(request)
        except ProviderError as exc:
            raise JudgeError(f"Judge provider call failed: {exc}") from exc

        parsed = response.parsed
        if not isinstance(parsed, JudgeResult):
            raise JudgeError(
                f"Judge provider did not return a JudgeResult (got {type(parsed).__name__})."
            )
        return parsed, response.usage


def rubric_reasoning_profile():
    """Default reasoning profile for the judge call.

    Kept as a function rather than a module-level constant so tests can
    monkeypatch it for faster mocked runs without reaching into the class.
    """
    from omegaprompt.domain.enums import ReasoningProfile

    return ReasoningProfile.STANDARD
