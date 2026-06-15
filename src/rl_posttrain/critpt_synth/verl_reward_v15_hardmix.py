from __future__ import annotations

from rl_posttrain.critpt_synth.verl_reward_v14_compact import compute_score as compute_v14_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: object,
) -> dict[str, object]:
    """V15 reward: keep V14 verifier reward, but make submission hygiene stricter."""

    result = compute_v14_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    score = float(result["score"])

    no_think = float(result.get("no_think_tags", 0.0)) >= 1.0
    too_long = float(result.get("too_long", 0.0)) >= 1.0
    very_long = float(result.get("very_long", 0.0)) >= 1.0
    unclosed = float(result.get("unclosed_code_block", 0.0)) >= 1.0
    repeat_ratio = float(result.get("repeat_ratio", 0.0))
    max_ngram = float(result.get("max_ngram_count", 0.0))
    acc = float(result.get("acc", 0.0)) >= 1.0

    if acc and not no_think:
        score = min(score, 0.92)
    if too_long:
        score = min(score, 0.84 if acc else 0.10)
    if very_long or unclosed:
        score = min(score, 0.12)
    if repeat_ratio > 0.20 or max_ngram >= 10:
        score = min(score, 0.82 if acc else 0.12)

    result["score"] = max(0.0, min(1.0, score))
    result["v15_no_think_strict"] = 1.0
    result["v15_hygiene_reward"] = result["score"]
    return result
