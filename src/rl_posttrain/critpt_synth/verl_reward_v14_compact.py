from __future__ import annotations

import ast
import json
import re
from collections import Counter

from rl_posttrain.critpt_synth.verifier import extract_python_code, verify_code_completion


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
TOKEN_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)


def _has_single_answer_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return sum(isinstance(node, ast.FunctionDef) and node.name == "answer" for node in tree.body) == 1


def _repeat_stats(code: str, n: int = 8) -> dict[str, float]:
    tokens = TOKEN_RE.findall(code)
    if len(tokens) < n:
        return {"repeat_ratio": 0.0, "max_ngram_count": 1.0}
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "repeat_ratio": repeated / max(len(grams), 1),
        "max_ngram_count": float(max(counts.values(), default=1)),
    }


def _max_repeated_line_count(code: str) -> int:
    lines = [line.strip() for line in code.splitlines() if len(line.strip()) >= 12]
    if not lines:
        return 1
    return max(Counter(lines).values())


def _format_and_compact_signals(
    solution_str: str,
    *,
    target_code: str,
    format_ok: bool,
    compile_ok: bool,
    exec_ok: bool,
) -> dict[str, object]:
    raw = solution_str.strip()
    code_blocks = CODE_BLOCK_RE.findall(raw)
    extracted = extract_python_code(raw)
    lower = raw.lower()
    has_code_block = bool(code_blocks)
    single_code_block = len(code_blocks) == 1
    unclosed_code_block = raw.count("```") % 2 == 1
    has_answer_def = "def answer" in extracted or "def answer" in raw
    no_think_tags = "<think>" not in lower and "</think>" not in lower
    single_answer_func = _has_single_answer_function(extracted)
    repeat_stats = _repeat_stats(extracted)
    max_repeated_line_count = _max_repeated_line_count(extracted)
    target_len = max(len(target_code.strip()), 1)
    length_limit = max(1800, 4 * target_len)
    too_long = len(extracted) > length_limit
    very_long = len(extracted) > max(3600, 8 * target_len)

    format_reward = 0.0
    if has_code_block:
        format_reward += 0.04
    if single_code_block:
        format_reward += 0.03
    if no_think_tags:
        format_reward += 0.04
    if has_answer_def:
        format_reward += 0.04
    if single_answer_func or format_ok:
        format_reward += 0.05
    if compile_ok:
        format_reward += 0.04
    if exec_ok:
        format_reward += 0.03

    compact_reward = 0.0
    if not unclosed_code_block:
        compact_reward += 0.03
    if not too_long:
        compact_reward += 0.05
    if float(repeat_stats["repeat_ratio"]) < 0.08:
        compact_reward += 0.04
    if float(repeat_stats["max_ngram_count"]) <= 3 and max_repeated_line_count <= 3:
        compact_reward += 0.03

    repetition_penalty = min(0.30, float(repeat_stats["repeat_ratio"]) * 1.4)
    if float(repeat_stats["max_ngram_count"]) >= 8:
        repetition_penalty += 0.08
    if max_repeated_line_count >= 5:
        repetition_penalty += 0.08
    repetition_penalty = min(repetition_penalty, 0.40)

    length_penalty = 0.0
    if too_long:
        length_penalty += 0.08
    if very_long:
        length_penalty += 0.12
    if unclosed_code_block:
        length_penalty += 0.10

    return {
        "format_reward": min(format_reward, 0.27),
        "compact_reward": min(compact_reward, 0.15),
        "repetition_penalty": repetition_penalty,
        "length_penalty": min(length_penalty, 0.25),
        "has_code_block": 1.0 if has_code_block else 0.0,
        "single_code_block": 1.0 if single_code_block else 0.0,
        "unclosed_code_block": 1.0 if unclosed_code_block else 0.0,
        "has_answer_def": 1.0 if has_answer_def else 0.0,
        "single_answer_func": 1.0 if single_answer_func else 0.0,
        "no_think_tags": 1.0 if no_think_tags else 0.0,
        "extracted_code_chars": len(extracted),
        "target_code_chars": target_len,
        "too_long": 1.0 if too_long else 0.0,
        "very_long": 1.0 if very_long else 0.0,
        "repeat_ratio": repeat_stats["repeat_ratio"],
        "max_ngram_count": repeat_stats["max_ngram_count"],
        "max_repeated_line_count": float(max_repeated_line_count),
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **_: object,
) -> dict[str, object]:
    extra_info = extra_info or {}
    verifier = extra_info.get("code_verifier") or extra_info.get("verifier") or {"checks": []}
    if isinstance(verifier, str):
        verifier = json.loads(verifier)
    result = verify_code_completion(solution_str, verifier)
    signals = _format_and_compact_signals(
        solution_str,
        target_code=ground_truth,
        format_ok=result.format_ok,
        compile_ok=result.compile_ok,
        exec_ok=result.exec_ok,
    )
    format_reward = float(signals["format_reward"])
    compact_reward = float(signals["compact_reward"])
    repetition_penalty = float(signals["repetition_penalty"])
    length_penalty = float(signals["length_penalty"])

    if result.ok:
        score = 1.0 if signals["no_think_tags"] else 0.97
        score -= min(0.08, repetition_penalty + length_penalty)
    else:
        shaped = max(float(result.score), format_reward + compact_reward)
        score = shaped - repetition_penalty - length_penalty
        if not result.compile_ok:
            score = min(score, 0.08)
        elif not result.exec_ok:
            score = min(score, 0.22)
        if signals["unclosed_code_block"] or signals["very_long"]:
            score = min(score, 0.18)
        if float(signals["repeat_ratio"]) > 0.18 or float(signals["max_ngram_count"]) >= 10:
            score = min(score, 0.20)
        if not signals["has_answer_def"]:
            score = min(score, 0.05)

    score = max(0.0, min(1.0, score))
    return {
        "score": score,
        "acc": 1.0 if result.ok else 0.0,
        "format_ok": 1.0 if result.format_ok else 0.0,
        "compile_ok": 1.0 if result.compile_ok else 0.0,
        "exec_ok": 1.0 if result.exec_ok else 0.0,
        "passed_checks": result.passed_checks,
        "total_checks": result.total_checks,
        "reason": result.reason,
        "data_source": data_source,
        **signals,
    }
