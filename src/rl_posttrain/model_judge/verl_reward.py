from __future__ import annotations

import json
import os
from typing import Any

from rl_posttrain.model_judge.openai_compatible import (
    PROMPT_VERSION,
    JudgeSettings,
    JsonSqliteCache,
    build_judge_messages,
    cache_key,
    normalize_judge_payload,
    openai_chat_json,
)


DEFAULT_EXTRA_METRICS = {
    "correctness": 0.0,
    "instruction_following": 0.0,
    "reasoning_quality": 0.0,
    "final_answer_consistency": 0.0,
    "fatal_error": 0.0,
}


def _with_default_metrics(payload: dict[str, object], data_source: str) -> dict[str, object]:
    complete: dict[str, object] = {
        **DEFAULT_EXTRA_METRICS,
        "score": 0.0,
        "acc": 0.0,
        "judge_error": 0.0,
        "quick_reject": 0.0,
        "reason": "",
        "data_source": data_source,
    }
    complete.update(payload)
    return complete


def _coerce_extra_info(extra_info: dict | None) -> dict[str, Any]:
    if not extra_info:
        return {}
    return dict(extra_info)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _quick_reject(solution_str: str) -> dict[str, object] | None:
    text = solution_str.strip()
    if not text:
        return {
            "score": 0.0,
            "acc": 0.0,
            "judge_error": 0.0,
            "quick_reject": 1.0,
            "reason": "empty answer",
        }
    if len(text) < int(os.environ.get("JUDGE_MIN_CHARS", "12")):
        return {
            "score": 0.02,
            "acc": 0.0,
            "judge_error": 0.0,
            "quick_reject": 1.0,
            "reason": "too short",
        }
    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **_: object,
) -> dict[str, object]:
    quick = _quick_reject(solution_str)
    if quick is not None:
        return _with_default_metrics(quick, data_source)

    info = _coerce_extra_info(extra_info)
    problem = _as_text(
        info.get("prompt_text")
        or info.get("problem")
        or info.get("question")
        or info.get("prompt")
        or ""
    )
    reference_answer = _as_text(
        info.get("reference_answer")
        or info.get("target_answer")
        or info.get("target_code")
        or ground_truth
        or ""
    )
    reference_trace = _as_text(info.get("reference_trace") or info.get("solution_trace") or "")
    rubric = _as_text(info.get("rubric") or "")

    if not problem.strip():
        return _with_default_metrics({
            "score": 0.0,
            "acc": 0.0,
            "judge_error": 1.0,
            "quick_reject": 0.0,
            "reason": "missing prompt_text in extra_info",
        }, data_source)

    settings = JudgeSettings.from_env()
    messages = build_judge_messages(
        problem=problem,
        candidate_response=solution_str,
        reference_answer=reference_answer,
        reference_trace=reference_trace,
        rubric=rubric,
    )
    key = cache_key(
        PROMPT_VERSION,
        settings.model,
        problem,
        solution_str,
        reference_answer,
        reference_trace,
        rubric,
    )

    try:
        cached = None
        if settings.cache_path:
            cached = JsonSqliteCache(settings.cache_path).get(key)
        if cached is None:
            raw = openai_chat_json(settings=settings, messages=messages)
            judged = normalize_judge_payload(raw)
            if settings.cache_path:
                JsonSqliteCache(settings.cache_path).set(key, judged)
        else:
            judged = normalize_judge_payload(cached)
    except Exception as exc:
        fail_open = os.environ.get("JUDGE_FAIL_OPEN", "0").lower() in {"1", "true", "yes"}
        return _with_default_metrics({
            "score": 0.5 if fail_open else 0.0,
            "acc": 0.0,
            "judge_error": 1.0,
            "quick_reject": 0.0,
            "reason": str(exc)[:500],
        }, data_source)

    score = float(judged["reward"])
    return _with_default_metrics({
        "score": score,
        "acc": 1.0 if score >= float(os.environ.get("JUDGE_ACC_THRESHOLD", "0.7")) else 0.0,
        "judge_error": 0.0,
        "quick_reject": 0.0,
        "correctness": judged["correctness"],
        "instruction_following": judged["instruction_following"],
        "reasoning_quality": judged["reasoning_quality"],
        "final_answer_consistency": judged["final_answer_consistency"],
        "fatal_error": 1.0 if judged["fatal_error"] else 0.0,
        "reason": judged["reason"],
    }, data_source)
