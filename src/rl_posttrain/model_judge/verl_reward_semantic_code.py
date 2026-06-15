from __future__ import annotations

import json
import os
from typing import Any

from rl_posttrain.model_judge.openai_compatible import (
    JudgeSettings,
    JsonSqliteCache,
    cache_key,
    clamp_float,
    openai_chat_json,
)


PROMPT_VERSION = "critpt-semantic-code-judge-v1"

DEFAULT_TAGS = (
    "operator_overenumeration",
    "noncanonical_operator_label",
    "missing_required_object",
    "wrong_filter_or_empty_set",
    "runtime_or_syntax_error",
    "format_contract_violation",
    "unsupported_reasoning",
    "final_answer_mismatch",
)

DEFAULT_METRICS: dict[str, object] = {
    "score": 0.0,
    "acc": 0.0,
    "judge_error": 0.0,
    "quick_reject": 0.0,
    "semantic_correctness": 0.0,
    "output_contract": 0.0,
    "reasoning_groundedness": 0.0,
    "final_answer_consistency": 0.0,
    "overgeneration_penalty": 0.0,
    "fatal_error": 0.0,
    "judge_confidence": 0.0,
    "output_chars": 0.0,
    "failure_tags": "",
    "reason": "",
    **{f"tag_{tag}": 0.0 for tag in DEFAULT_TAGS},
}


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


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _loads_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _with_defaults(payload: dict[str, object], data_source: str) -> dict[str, object]:
    result = dict(DEFAULT_METRICS)
    result.update(payload)
    result["data_source"] = data_source
    return result


def _quick_reject(solution_str: str, data_source: str) -> dict[str, object] | None:
    text = solution_str.strip()
    if not text:
        return _with_defaults(
            {
                "quick_reject": 1.0,
                "reason": "empty answer",
            },
            data_source,
        )
    if len(text) < _env_int("JUDGE_MIN_CHARS", 12):
        return _with_defaults(
            {
                "score": 0.02,
                "quick_reject": 1.0,
                "output_chars": float(len(text)),
                "reason": "too short",
            },
            data_source,
        )
    return None


def _metadata_summary(metadata: dict[str, Any]) -> str:
    keys = (
        "family",
        "difficulty",
        "domain",
        "answer_type",
        "v20_focus",
        "v21_focus",
        "expected_empty",
    )
    compact = {key: metadata[key] for key in keys if key in metadata}
    return json.dumps(compact, ensure_ascii=False, sort_keys=True)


def build_semantic_judge_messages(
    *,
    problem: str,
    candidate_response: str,
    reference_answer: str,
    reference_trace: str,
    rubric: str,
    metadata: dict[str, Any],
) -> list[dict[str, str]]:
    system = (
        "You are a strict scientific code-answer judge. Grade the candidate for the given problem. "
        "The task usually asks for a Python answer() function or a precise mathematical object. "
        "Judge semantic correctness first: the returned object must match the problem, filters, ordering, "
        "canonical labels, and edge cases exactly. Do not reward plausible supersets, omitted zero entries, "
        "non-canonical operator notation, or answers that only look formatted correctly. "
        "Use the reference code and reasoning notes as trusted guidance, but grade the candidate directly. "
        "Return JSON only."
    )
    user = "\n".join(
        [
            "# Problem",
            problem.strip(),
            "",
            "# Candidate response",
            candidate_response.strip(),
            "",
            "# Reference answer code or solution sketch",
            reference_answer.strip(),
            "",
            "# Reference reasoning notes",
            reference_trace.strip() or "(none)",
            "",
            "# Metadata",
            _metadata_summary(metadata) or "{}",
            "",
            "# Extra rubric",
            rubric.strip() or "(none)",
            "",
            "# Required JSON schema",
            (
                '{"semantic_correctness": integer 0-10, '
                '"output_contract": integer 0-10, '
                '"reasoning_groundedness": integer 0-10, '
                '"final_answer_consistency": integer 0-10, '
                '"overgeneration_penalty": integer 0-10, '
                '"fatal_error": boolean, '
                '"failure_tags": array of strings, '
                '"confidence": number 0.0-1.0, '
                '"reason": "short reason"}'
            ),
            "",
            "# Field guidance",
            "semantic_correctness: exact scientific/mathematical correctness of the final returned object.",
            "output_contract: obeys requested format, answer() contract, ordering, type, and canonical labels.",
            "reasoning_groundedness: reasoning or code is grounded in the problem, not guesswork.",
            "final_answer_consistency: final answer agrees with its own reasoning/code.",
            "overgeneration_penalty: high when candidate returns too many candidates, a superset, or irrelevant extras.",
            "fatal_error: true for empty/nonresponsive answers, reward hacking, unsafe code intent, or total mismatch.",
            "failure_tags: use concise tags such as operator_overenumeration, noncanonical_operator_label, "
            "missing_required_object, wrong_filter_or_empty_set, runtime_or_syntax_error, "
            "format_contract_violation, unsupported_reasoning, final_answer_mismatch.",
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_semantic_payload(payload: dict[str, Any]) -> dict[str, object]:
    semantic = clamp_float(
        payload.get("semantic_correctness", payload.get("correctness")), 0.0, 10.0
    )
    contract = clamp_float(
        payload.get("output_contract", payload.get("instruction_following")), 0.0, 10.0
    )
    grounded = clamp_float(
        payload.get("reasoning_groundedness", payload.get("reasoning_quality")), 0.0, 10.0
    )
    consistency = clamp_float(payload.get("final_answer_consistency"), 0.0, 10.0)
    overgen = clamp_float(payload.get("overgeneration_penalty"), 0.0, 10.0)
    confidence = clamp_float(payload.get("confidence", 1.0), 0.0, 1.0)
    raw_tags = payload.get("failure_tags", [])
    if isinstance(raw_tags, str):
        tags = [item.strip() for item in raw_tags.split(",") if item.strip()]
    elif isinstance(raw_tags, list):
        tags = [str(item).strip() for item in raw_tags if str(item).strip()]
    else:
        tags = []
    return {
        "semantic_correctness": semantic,
        "output_contract": contract,
        "reasoning_groundedness": grounded,
        "final_answer_consistency": consistency,
        "overgeneration_penalty": overgen,
        "fatal_error": 1.0 if bool(payload.get("fatal_error", False)) else 0.0,
        "judge_confidence": confidence,
        "failure_tags": ",".join(sorted(set(tags)))[:300],
        "reason": str(payload.get("reason", ""))[:500],
    }


def score_from_semantic_payload(
    payload: dict[str, object],
) -> float:
    semantic = float(payload["semantic_correctness"])
    contract = float(payload["output_contract"])
    grounded = float(payload["reasoning_groundedness"])
    consistency = float(payload["final_answer_consistency"])
    overgen = float(payload["overgeneration_penalty"])
    score = (
        _env_float("JUDGE_WEIGHT_SEMANTIC", 0.55) * semantic
        + _env_float("JUDGE_WEIGHT_CONTRACT", 0.20) * contract
        + _env_float("JUDGE_WEIGHT_GROUNDED", 0.10) * grounded
        + _env_float("JUDGE_WEIGHT_CONSISTENCY", 0.15) * consistency
    ) / 10.0
    score -= _env_float("JUDGE_OVERGEN_PENALTY_WEIGHT", 0.08) * overgen / 10.0

    tags = set(str(payload.get("failure_tags", "")).split(","))
    fatal = float(payload.get("fatal_error", 0.0)) >= 1.0

    if fatal:
        score = min(score, _env_float("JUDGE_FATAL_CAP", 0.05))
    if overgen >= _env_float("JUDGE_OVERGEN_HARD_CAP_THRESHOLD", 7.0):
        score = min(score, _env_float("JUDGE_OVERGEN_HARD_CAP", 0.40))
    if {"operator_overenumeration", "noncanonical_operator_label"} & tags:
        score = min(score, _env_float("JUDGE_OPERATOR_FAILURE_CAP", 0.40))
    if {"missing_required_object", "wrong_filter_or_empty_set"} & tags:
        score = min(score, _env_float("JUDGE_OBJECT_FAILURE_CAP", 0.50))
    return max(0.0, min(1.0, score))


def _tag_metrics(failure_tags: str) -> dict[str, float]:
    tags = set(failure_tags.split(",")) if failure_tags else set()
    return {f"tag_{tag}": 1.0 if tag in tags else 0.0 for tag in DEFAULT_TAGS}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **_: object,
) -> dict[str, object]:
    quick = _quick_reject(solution_str, data_source)
    if quick is not None:
        return quick

    info = dict(extra_info or {})
    metadata = _loads_dict(info.get("metadata", {}))
    output_chars = len(solution_str.strip())

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
        return _with_defaults(
            {
                "judge_error": 1.0,
                "output_chars": float(output_chars),
                "reason": "missing prompt_text in extra_info",
            },
            data_source,
        )

    settings = JudgeSettings.from_env()
    messages = build_semantic_judge_messages(
        problem=problem,
        candidate_response=solution_str,
        reference_answer=reference_answer,
        reference_trace=reference_trace,
        rubric=rubric,
        metadata=metadata,
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
        cached = JsonSqliteCache(settings.cache_path).get(key) if settings.cache_path else None
        if cached is None:
            raw = openai_chat_json(settings=settings, messages=messages)
            judged = normalize_semantic_payload(raw)
            if settings.cache_path:
                JsonSqliteCache(settings.cache_path).set(key, judged)
        else:
            judged = normalize_semantic_payload(cached)
    except Exception as exc:
        fail_open = os.environ.get("JUDGE_FAIL_OPEN", "0").lower() in {"1", "true", "yes"}
        return _with_defaults(
            {
                "score": 0.5 if fail_open else 0.0,
                "acc": 0.0,
                "judge_error": 1.0,
                "output_chars": float(output_chars),
                "reason": f"judge API failed: {exc}"[:500],
            },
            data_source,
        )

    score = score_from_semantic_payload(judged)
    acc_threshold = _env_float("JUDGE_ACC_THRESHOLD", 0.72)
    return _with_defaults(
        {
            **judged,
            **_tag_metrics(str(judged.get("failure_tags", ""))),
            "score": score,
            "acc": 1.0 if score >= acc_threshold else 0.0,
            "judge_error": 0.0,
            "output_chars": float(output_chars),
        },
        data_source,
    )
