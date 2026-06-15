from __future__ import annotations

from rl_posttrain.critpt_synth.verl_reward_v14_compact import compute_score as compute_v14_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: object,
) -> dict[str, object]:
    """V19 reward for failure-mined GRPO.

    V18 got the rough code format mostly right, but many rollouts still carried
    empty Qwen thinking tags and official eval stayed at 0. This reward keeps
    executable verifier correctness dominant, while making no-think, single
    answer function, and compact closed code blocks visibly better inside a
    GRPO group.
    """

    result = compute_v14_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )

    acc = float(result.get("acc", 0.0)) >= 1.0
    no_think = float(result.get("no_think_tags", 0.0)) >= 1.0
    single_code_block = float(result.get("single_code_block", 0.0)) >= 1.0
    single_answer_func = float(result.get("single_answer_func", 0.0)) >= 1.0
    compile_ok = float(result.get("compile_ok", 0.0)) >= 1.0
    exec_ok = float(result.get("exec_ok", 0.0)) >= 1.0
    too_long = float(result.get("too_long", 0.0)) >= 1.0
    very_long = float(result.get("very_long", 0.0)) >= 1.0
    unclosed = float(result.get("unclosed_code_block", 0.0)) >= 1.0
    repeat_ratio = float(result.get("repeat_ratio", 0.0))
    max_ngram = float(result.get("max_ngram_count", 0.0))
    extracted_chars = max(float(result.get("extracted_code_chars", 1.0)), 1.0)
    target_chars = max(float(result.get("target_code_chars", 1.0)), 1.0)
    compact_ratio = min(1.0, target_chars / extracted_chars)

    if acc:
        score = 0.76
        score += 0.12 if no_think else 0.0
        score += 0.05 if single_code_block else 0.0
        score += 0.04 if single_answer_func else 0.0
        score += 0.03 * compact_ratio
    else:
        format_reward = float(result.get("format_reward", 0.0))
        compact_reward = float(result.get("compact_reward", 0.0))
        score = max(float(result.get("score", 0.0)), format_reward + compact_reward)
        score += 0.03 if no_think and single_code_block else 0.0
        score += 0.02 if compile_ok else 0.0
        score += 0.02 if exec_ok else 0.0
        score = min(score, 0.58)

    if too_long:
        score = min(score, 0.78 if acc else 0.18)
    if very_long or unclosed:
        score = min(score, 0.16)
    if repeat_ratio > 0.20 or max_ngram >= 10:
        score = min(score, 0.74 if acc else 0.16)
    if not no_think:
        score = min(score, 0.89 if acc else 0.30)

    result["score"] = max(0.0, min(1.0, score))
    result["v19_failure_mined_reward"] = result["score"]
    result["v19_no_think_weighted"] = 1.0
    result["v19_compact_ratio"] = compact_ratio
    return result
