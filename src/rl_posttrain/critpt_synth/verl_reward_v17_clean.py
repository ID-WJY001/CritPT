from __future__ import annotations

from rl_posttrain.critpt_synth.verl_reward_v14_compact import compute_score as compute_v14_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: object,
) -> dict[str, object]:
    """V17 reward: keep verifier correctness dominant, but avoid flat caps.

    V15 capped every correct answer with Qwen-style ``<think>`` tags to the same
    score. That makes GRPO groups go flat when all samples are correct. V17 keeps
    a strong no-think preference, while preserving small compactness differences
    among otherwise correct code answers.
    """

    result = compute_v14_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )

    score = float(result["score"])
    acc = float(result.get("acc", 0.0)) >= 1.0
    no_think = float(result.get("no_think_tags", 0.0)) >= 1.0
    too_long = float(result.get("too_long", 0.0)) >= 1.0
    very_long = float(result.get("very_long", 0.0)) >= 1.0
    unclosed = float(result.get("unclosed_code_block", 0.0)) >= 1.0
    repeat_ratio = float(result.get("repeat_ratio", 0.0))
    max_ngram = float(result.get("max_ngram_count", 0.0))
    single_code_block = float(result.get("single_code_block", 0.0)) >= 1.0
    single_answer_func = float(result.get("single_answer_func", 0.0)) >= 1.0
    extracted_chars = max(float(result.get("extracted_code_chars", 1.0)), 1.0)
    target_chars = max(float(result.get("target_code_chars", 1.0)), 1.0)
    compact_ratio = min(1.0, target_chars / extracted_chars)

    if acc:
        score = 0.86
        score += 0.05 * compact_ratio
        score += 0.02 if single_code_block else 0.0
        score += 0.02 if single_answer_func else 0.0
        score += 0.07 if no_think else 0.0
    else:
        score = max(score, float(result.get("format_reward", 0.0)) + float(result.get("compact_reward", 0.0)))
        if no_think and single_code_block:
            score += 0.02

    if too_long:
        score = min(score, 0.84 if acc else 0.12)
    if very_long or unclosed:
        score = min(score, 0.12)
    if repeat_ratio > 0.20 or max_ngram >= 10:
        score = min(score, 0.82 if acc else 0.12)

    result["score"] = max(0.0, min(1.0, score))
    result["v17_clean_reward"] = result["score"]
    result["v17_compact_ratio"] = compact_ratio
    result["v17_no_flat_think_cap"] = 1.0
    return result
