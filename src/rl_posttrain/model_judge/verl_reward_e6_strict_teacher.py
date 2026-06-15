from __future__ import annotations

import os
from typing import Any

from rl_posttrain.model_judge import verl_reward_e3_strict_code as base_reward


PROMPT_VERSION = "critpt-e6-strict-teacher-final-answer-v1"

E6_DEFAULTS = {
    "JUDGE_WEIGHT_ALGORITHMIC": "0.44",
    "JUDGE_WEIGHT_SCIENTIFIC": "0.16",
    "JUDGE_WEIGHT_EXACTNESS": "0.30",
    "JUDGE_WEIGHT_CONTRACT": "0.05",
    "JUDGE_WEIGHT_ANTI_GUESS": "0.05",
    "JUDGE_FATAL_CAP": "0.02",
    "JUDGE_GUESS_CAP": "0.06",
    "JUDGE_MISSING_ALGORITHM_CAP": "0.18",
    "JUDGE_REFERENCE_MISMATCH_CAP": "0.18",
    "JUDGE_RULE_FAILURE_CAP": "0.18",
    "JUDGE_CONTRACT_FAILURE_CAP": "0.40",
    "JUDGE_SYNTAX_RUNTIME_CAP": "0.12",
    "JUDGE_LOW_ALGO_CAP": "0.12",
    "JUDGE_LOW_ANTI_GUESS_CAP": "0.10",
    "JUDGE_LOW_SCIENTIFIC_CAP": "0.14",
    "JUDGE_MEDIUM_ALGO_CAP": "0.28",
    "JUDGE_LOW_EXACTNESS_CAP": "0.25",
    "JUDGE_ACC_THRESHOLD": "0.82",
}

E6_RUBRIC = """
E6 strict mode was introduced after official-70 evaluations showed that template-shaped
answer() functions with guessed constants can receive too much reward. Grade only the
returned scientific object against the trusted reference output.

Hard constraints for this run:
- A correct-looking docstring, variable names, imports, or answer() shell is worth little
  unless the returned values match the trusted final output.
- If the returned object differs from the trusted reference in any decisive value, label,
  ordering, sign, set membership, tie-break, or rounded number, keep algorithmic_equivalence
  and numeric_symbolic_exactness low and include reference_mismatch or wrong_numeric_or_symbolic_result.
- All-zero lists, repeated constants, common defaults such as 0, 1, 0.5, 0.6931, 1.23,
  generic small sets, and placeholder comments must be tagged as all_zero_or_placeholder,
  constant_guess, or shape_only_answer unless they exactly equal the trusted final output.
- A compact literal return is allowed only when it exactly matches the trusted reference.
- Do not reward plausible physics prose or a plausible mini-algorithm if its final return
  value is wrong. The official benchmark scores the returned answer, not the vibe.
""".strip()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: object,
) -> dict[str, object]:
    for name, value in E6_DEFAULTS.items():
        os.environ.setdefault(name, value)

    info: dict[str, Any] = dict(extra_info or {})
    existing_rubric = str(info.get("rubric", "") or "").strip()
    info["rubric"] = E6_RUBRIC if not existing_rubric else f"{E6_RUBRIC}\n\n{existing_rubric}"

    old_prompt_version = base_reward.PROMPT_VERSION
    base_reward.PROMPT_VERSION = PROMPT_VERSION
    try:
        return base_reward.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=info,
            **kwargs,
        )
    finally:
        base_reward.PROMPT_VERSION = old_prompt_version
