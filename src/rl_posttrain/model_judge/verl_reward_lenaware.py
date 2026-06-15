from __future__ import annotations

import os
import re
from typing import Any

from rl_posttrain.model_judge.verl_reward import compute_score as judge_compute_score


FINAL_ANSWER_RE = re.compile(
    r"(最终答案\s*[:：]|final\s+answer\s*[:：]|answer\s*[:：]|答案\s*[:：])",
    re.IGNORECASE,
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _find_final_marker(text: str) -> re.Match[str] | None:
    last_match: re.Match[str] | None = None
    for match in FINAL_ANSWER_RE.finditer(text):
        last_match = match
    return last_match


def _length_adjustment(solution_str: str, score: float) -> tuple[float, dict[str, object]]:
    text = solution_str.strip()
    output_chars = len(text)
    match = _find_final_marker(text)
    has_final = 1.0 if match else 0.0
    post_final_chars = len(text[match.end() :].strip()) if match else 0

    soft_chars = _env_int("JUDGE_LEN_SOFT_CHARS", 2600)
    penalty_per_1k = _env_float("JUDGE_LEN_PENALTY_PER_1K", 0.035)
    max_len_penalty = _env_float("JUDGE_LEN_MAX_PENALTY", 0.10)
    no_final_cap = _env_float("JUDGE_NO_FINAL_SCORE_CAP", 0.72)
    no_final_overlong_cap = _env_float("JUDGE_NO_FINAL_OVERLONG_SCORE_CAP", 0.55)
    no_final_overlong_chars = _env_int("JUDGE_NO_FINAL_OVERLONG_CHARS", 2200)
    post_final_chars_cap = _env_int("JUDGE_POST_FINAL_MAX_CHARS", 700)
    post_final_penalty = _env_float("JUDGE_POST_FINAL_PENALTY", 0.08)

    adjusted = score
    length_penalty = 0.0
    no_final_cap_applied = 0.0
    post_final_penalty_applied = 0.0

    if output_chars > soft_chars:
        over = (output_chars - soft_chars) / 1000.0
        length_penalty = min(max_len_penalty, max(0.0, over * penalty_per_1k))
        adjusted -= length_penalty

    if not match:
        cap = no_final_overlong_cap if output_chars > no_final_overlong_chars else no_final_cap
        if adjusted > cap:
            adjusted = cap
            no_final_cap_applied = 1.0
    elif post_final_chars > post_final_chars_cap:
        adjusted -= post_final_penalty
        post_final_penalty_applied = 1.0

    adjusted = max(0.0, min(1.0, adjusted))
    return adjusted, {
        "answer_marker_present": has_final,
        "output_chars": float(output_chars),
        "post_final_chars": float(post_final_chars),
        "length_penalty": float(length_penalty),
        "no_final_cap_applied": no_final_cap_applied,
        "post_final_penalty_applied": post_final_penalty_applied,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict[str, object]:
    payload = judge_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    try:
        original_score = float(payload.get("score", 0.0))
    except (TypeError, ValueError):
        original_score = 0.0

    adjusted_score, metrics = _length_adjustment(solution_str, original_score)
    payload["raw_judge_score"] = original_score
    payload["score"] = adjusted_score
    payload["acc"] = 1.0 if adjusted_score >= _env_float("JUDGE_ACC_THRESHOLD", 0.7) else 0.0
    payload.update(metrics)
    if adjusted_score < original_score:
        reason = str(payload.get("reason", ""))
        payload["reason"] = (reason + " [length-aware shaping applied]").strip()[:500]
    return payload
