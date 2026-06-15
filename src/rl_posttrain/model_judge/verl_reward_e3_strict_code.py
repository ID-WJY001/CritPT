from __future__ import annotations

import json
import os
import re
from typing import Any

from rl_posttrain.model_judge.openai_compatible import (
    JudgeSettings,
    JsonSqliteCache,
    cache_key,
    clamp_float,
    openai_chat_json,
)


PROMPT_VERSION = "critpt-e3-final-answer-code-judge-v2"

DEFAULT_TAGS = (
    "missing_answer_function",
    "markdown_or_think_output",
    "syntax_or_runtime_risk",
    "constant_guess",
    "all_zero_or_placeholder",
    "shape_only_answer",
    "missing_algorithm",
    "reference_mismatch",
    "wrong_numeric_or_symbolic_result",
    "wrong_ordering_or_labels",
    "wrong_filter_or_tie_break",
    "overgenerated_superset",
    "format_contract_violation",
)

DEFAULT_METRICS: dict[str, object] = {
    "score": 0.0,
    "acc": 0.0,
    "judge_error": 0.0,
    "quick_reject": 0.0,
    "algorithmic_equivalence": 0.0,
    "scientific_correctness": 0.0,
    "numeric_symbolic_exactness": 0.0,
    "output_contract": 0.0,
    "anti_guess_evidence": 0.0,
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


def _candidate_code_for_judge(solution_str: str) -> tuple[str, list[str]]:
    text = solution_str.strip()
    tags: list[str] = []
    if not text:
        return "", tags

    lowered = text.lower()
    if "<think" in lowered or "</think" in lowered:
        tags.append("markdown_or_think_output")
        text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    if "```" not in text:
        return text, tags

    blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in blocks:
        if "def answer" in block.lower():
            return block.strip(), tags
    if blocks:
        return blocks[0].strip(), tags
    return text.replace("```python", "").replace("```py", "").replace("```", "").strip(), tags


def _quick_reject(candidate_code: str, original_solution: str, data_source: str) -> dict[str, object] | None:
    text = candidate_code.strip()
    if not text:
        return _with_defaults(
            {
                "quick_reject": 1.0,
                "output_chars": float(len(original_solution.strip())),
                "reason": "empty answer",
            },
            data_source,
        )
    lowered = text.lower()
    if "def answer" not in lowered:
        return _with_defaults(
            {
                "score": _env_float("JUDGE_MISSING_ANSWER_CAP", 0.04),
                "quick_reject": 1.0,
                "output_chars": float(len(original_solution.strip())),
                "failure_tags": "missing_answer_function",
                "tag_missing_answer_function": 1.0,
                "reason": "candidate does not define answer()",
            },
            data_source,
        )
    if len(text) < _env_int("JUDGE_MIN_CHARS", 24):
        return _with_defaults(
            {
                "score": 0.01,
                "quick_reject": 1.0,
                "output_chars": float(len(original_solution.strip())),
                "reason": "too short to be a meaningful solution",
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
        "e2_family",
        "complexity_notes",
        "expected_empty",
    )
    compact = {key: metadata[key] for key in keys if key in metadata}
    return json.dumps(compact, ensure_ascii=False, sort_keys=True)


def build_strict_code_judge_messages(
    *,
    problem: str,
    candidate_original_response: str,
    candidate_response: str,
    reference_answer: str,
    reference_final_answer: str,
    reference_trace: str,
    rubric: str,
    metadata: dict[str, Any],
) -> list[dict[str, str]]:
    system = (
        "You are a strict final-answer judge for scientific Python answer() functions. "
        "Use the reference final answer, reference reasoning, and executable reference code as trusted ground truth. "
        "The training prompt asks for exactly one Python code block containing answer(); an outer Python fence is valid. "
        "The prompt also permits compact final literal returns, so do not require the candidate to reimplement the full algorithm. "
        "A high score requires the candidate answer() to return the same scientific/mathematical object with the same numeric "
        "rounding, labels, ordering, filters, tie-breaks, empty-case behavior, and return type. "
        "Be skeptical of wrong constants, all-zero placeholders, malformed code, prose, hidden reasoning, and plausible-looking "
        "answers that differ from the trusted reference result. "
        "Return JSON only."
    )
    user = "\n".join(
        [
            "# Problem",
            problem.strip(),
            "",
            "# Candidate original response",
            candidate_original_response.strip(),
            "",
            "# Candidate code to judge",
            candidate_response.strip(),
            "",
            "# Trusted reference final answer",
            reference_final_answer.strip() or "(not provided; infer from reference code and reasoning)",
            "",
            "# Trusted reference answer code / solution",
            reference_answer.strip(),
            "",
            "# Trusted reference reasoning notes",
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
                '{"algorithmic_equivalence": integer 0-10, '
                '"scientific_correctness": integer 0-10, '
                '"numeric_symbolic_exactness": integer 0-10, '
                '"output_contract": integer 0-10, '
                '"anti_guess_evidence": integer 0-10, '
                '"fatal_error": boolean, '
                '"failure_tags": array of strings, '
                '"confidence": number 0.0-1.0, '
                '"reason": "short reason"}'
            ),
            "",
            "# Strict scoring scale",
            "0-1: empty, no answer(), unsafe/reward-hacking, or total mismatch.",
            "1-2: placeholder, all-zero/default list, malformed answer, or unrelated constants.",
            "2-4: answer has the right broad type but wrong key values, labels, ordering, or filters.",
            "4-6: partially correct final object with important numeric, symbolic, ordering, or tie-break errors.",
            "6-8: mostly correct final object with minor format, rounding, or edge-case issues.",
            "8-10: final answer is equivalent to the reference for the stated problem and answer() contract.",
            "",
            "# Field guidance",
            "algorithmic_equivalence: final-answer equivalence to the reference; full algorithmic reimplementation is not required.",
            "scientific_correctness: whether the returned scientific/mathematical object is right.",
            "numeric_symbolic_exactness: exact constants, formulas, rounding, ordering, labels, filters, and tie-breaks.",
            "output_contract: defines answer(), uses the requested code-block/function contract, correct return type, no prose or hidden reasoning.",
            "anti_guess_evidence: high when the returned values/labels are problem-specific and match the trusted reference; low for wrong constants, zeros, stubs, or unsupported shortcuts.",
            "fatal_error: true for no answer(), nonresponsive output, unsafe intent, or complete reference mismatch.",
            "failure_tags: choose any relevant tags from "
            + ", ".join(DEFAULT_TAGS)
            + ".",
            "",
            "# Important caps you must respect",
            "If the candidate returns compact final literal values that match the trusted final answer, score it highly; this is allowed.",
            "If the candidate returns constants or labels that do not match the trusted final answer, set exactness low and include "
            "constant_guess or reference_mismatch.",
            "If the candidate has the right function shape but wrong final content, set algorithmic_equivalence <= 3 and include "
            "shape_only_answer or reference_mismatch.",
            "If a finite scan/filter/tie-break/order rule is required and missing or wrong, include wrong_filter_or_tie_break "
            "or wrong_ordering_or_labels.",
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_strict_payload(payload: dict[str, Any]) -> dict[str, object]:
    algorithmic = clamp_float(payload.get("algorithmic_equivalence"), 0.0, 10.0)
    scientific = clamp_float(
        payload.get("scientific_correctness", payload.get("semantic_correctness")), 0.0, 10.0
    )
    exactness = clamp_float(payload.get("numeric_symbolic_exactness"), 0.0, 10.0)
    contract = clamp_float(
        payload.get("output_contract", payload.get("instruction_following")), 0.0, 10.0
    )
    anti_guess = clamp_float(payload.get("anti_guess_evidence"), 0.0, 10.0)
    confidence = clamp_float(payload.get("confidence", 1.0), 0.0, 1.0)
    raw_tags = payload.get("failure_tags", [])
    if isinstance(raw_tags, str):
        tags = [item.strip() for item in raw_tags.split(",") if item.strip()]
    elif isinstance(raw_tags, list):
        tags = [str(item).strip() for item in raw_tags if str(item).strip()]
    else:
        tags = []
    return {
        "algorithmic_equivalence": algorithmic,
        "scientific_correctness": scientific,
        "numeric_symbolic_exactness": exactness,
        "output_contract": contract,
        "anti_guess_evidence": anti_guess,
        "fatal_error": 1.0 if bool(payload.get("fatal_error", False)) else 0.0,
        "judge_confidence": confidence,
        "failure_tags": ",".join(sorted(set(tags)))[:300],
        "reason": str(payload.get("reason", ""))[:500],
    }


def score_from_strict_payload(payload: dict[str, object]) -> float:
    algorithmic = float(payload["algorithmic_equivalence"])
    scientific = float(payload["scientific_correctness"])
    exactness = float(payload["numeric_symbolic_exactness"])
    contract = float(payload["output_contract"])
    anti_guess = float(payload["anti_guess_evidence"])

    score = (
        _env_float("JUDGE_WEIGHT_ALGORITHMIC", 0.35) * algorithmic
        + _env_float("JUDGE_WEIGHT_SCIENTIFIC", 0.30) * scientific
        + _env_float("JUDGE_WEIGHT_EXACTNESS", 0.15) * exactness
        + _env_float("JUDGE_WEIGHT_CONTRACT", 0.10) * contract
        + _env_float("JUDGE_WEIGHT_ANTI_GUESS", 0.10) * anti_guess
    ) / 10.0

    tags = set(str(payload.get("failure_tags", "")).split(","))
    fatal = float(payload.get("fatal_error", 0.0)) >= 1.0

    if fatal:
        score = min(score, _env_float("JUDGE_FATAL_CAP", 0.03))
    if {"all_zero_or_placeholder", "shape_only_answer"} & tags:
        score = min(score, _env_float("JUDGE_GUESS_CAP", 0.15))
    if "constant_guess" in tags and algorithmic <= _env_float("JUDGE_CONSTANT_LOW_EQUIV_THRESHOLD", 4.0):
        score = min(score, _env_float("JUDGE_GUESS_CAP", 0.15))
    if "missing_algorithm" in tags:
        score = min(score, _env_float("JUDGE_MISSING_ALGORITHM_CAP", 0.25))
    if {"reference_mismatch", "wrong_numeric_or_symbolic_result"} & tags:
        score = min(score, _env_float("JUDGE_REFERENCE_MISMATCH_CAP", 0.35))
    if {"wrong_ordering_or_labels", "wrong_filter_or_tie_break", "overgenerated_superset"} & tags:
        score = min(score, _env_float("JUDGE_RULE_FAILURE_CAP", 0.45))
    if "format_contract_violation" in tags:
        score = min(score, _env_float("JUDGE_CONTRACT_FAILURE_CAP", 0.45))
    if "syntax_or_runtime_risk" in tags:
        score = min(score, _env_float("JUDGE_SYNTAX_RUNTIME_CAP", 0.20))

    if algorithmic <= _env_float("JUDGE_LOW_ALGO_THRESHOLD", 2.0):
        score = min(score, _env_float("JUDGE_LOW_ALGO_CAP", 0.20))
    if anti_guess <= _env_float("JUDGE_LOW_ANTI_GUESS_THRESHOLD", 2.0):
        score = min(score, _env_float("JUDGE_LOW_ANTI_GUESS_CAP", 0.20))
    if scientific <= _env_float("JUDGE_LOW_SCIENTIFIC_THRESHOLD", 2.0):
        score = min(score, _env_float("JUDGE_LOW_SCIENTIFIC_CAP", 0.25))
    if algorithmic <= _env_float("JUDGE_MEDIUM_ALGO_THRESHOLD", 4.0):
        score = min(score, _env_float("JUDGE_MEDIUM_ALGO_CAP", 0.45))
    if exactness <= _env_float("JUDGE_LOW_EXACTNESS_THRESHOLD", 4.0):
        score = min(score, _env_float("JUDGE_LOW_EXACTNESS_CAP", 0.65))

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
    candidate_code, pre_tags = _candidate_code_for_judge(solution_str)
    quick = _quick_reject(candidate_code, solution_str, data_source)
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
    reference_final_answer = _as_text(
        info.get("reference_output")
        or info.get("reference_final_answer")
        or info.get("expected_output")
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
    if not reference_answer.strip():
        return _with_defaults(
            {
                "judge_error": 1.0,
                "output_chars": float(output_chars),
                "reason": "missing reference answer for strict judge",
            },
            data_source,
        )

    settings = JudgeSettings.from_env()
    messages = build_strict_code_judge_messages(
        problem=problem,
        candidate_original_response=solution_str,
        candidate_response=candidate_code,
        reference_answer=reference_answer,
        reference_final_answer=reference_final_answer,
        reference_trace=reference_trace,
        rubric=rubric,
        metadata=metadata,
    )
    key = cache_key(
        PROMPT_VERSION,
        settings.model,
        problem,
        candidate_code,
        reference_answer,
        reference_final_answer,
        reference_trace,
        rubric,
    )

    try:
        cache = JsonSqliteCache(settings.cache_path) if settings.cache_path else None
        cached = cache.get(key) if cache else None
        if cached is None:
            raw = openai_chat_json(settings=settings, messages=messages)
            judged = normalize_strict_payload(raw)
            if cache:
                cache.set(key, judged)
        else:
            judged = normalize_strict_payload(cached)
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

    if pre_tags:
        merged_tags = set(str(judged.get("failure_tags", "")).split(",")) if judged.get("failure_tags") else set()
        merged_tags.update(pre_tags)
        judged["failure_tags"] = ",".join(sorted(tag for tag in merged_tags if tag))
        judged["output_contract"] = min(
            float(judged.get("output_contract", 0.0)),
            _env_float("JUDGE_THINK_CONTRACT_CAP", 7.0),
        )

    score = score_from_strict_payload(judged)
    acc_threshold = _env_float("JUDGE_ACC_THRESHOLD", 0.78)
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
