from __future__ import annotations

import ast
import json
import re

from rl_posttrain.critpt_synth.verifier import extract_python_code, verify_code_completion


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _has_single_answer_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return sum(isinstance(node, ast.FunctionDef) and node.name == "answer" for node in tree.body) == 1


def _format_signals(solution_str: str, *, format_ok: bool, compile_ok: bool, exec_ok: bool) -> dict[str, object]:
    raw = solution_str.strip()
    code_blocks = CODE_BLOCK_RE.findall(raw)
    extracted = extract_python_code(raw)
    lower = raw.lower()
    has_code_block = bool(code_blocks)
    has_answer_def = "def answer" in extracted or "def answer" in raw
    no_think_tags = "<think>" not in lower and "</think>" not in lower
    single_answer_func = _has_single_answer_function(extracted)

    # This is deliberately a partial shaping reward, capped below correctness.
    # It gives GRPO a gradient before exact executable answers start appearing.
    reward = 0.0
    if has_code_block:
        reward += 0.08
    if len(code_blocks) == 1:
        reward += 0.03
    if no_think_tags:
        reward += 0.05
    if has_answer_def:
        reward += 0.08
    if single_answer_func or format_ok:
        reward += 0.10
    if compile_ok:
        reward += 0.04
    if exec_ok:
        reward += 0.02

    return {
        "format_reward": min(reward, 0.40),
        "has_code_block": 1.0 if has_code_block else 0.0,
        "single_code_block": 1.0 if len(code_blocks) == 1 else 0.0,
        "has_answer_def": 1.0 if has_answer_def else 0.0,
        "single_answer_func": 1.0 if single_answer_func else 0.0,
        "no_think_tags": 1.0 if no_think_tags else 0.0,
        "extracted_code_chars": len(extracted),
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **_: object,
) -> dict[str, object]:
    del ground_truth
    extra_info = extra_info or {}
    verifier = extra_info.get("code_verifier") or extra_info.get("verifier") or {"checks": []}
    if isinstance(verifier, str):
        verifier = json.loads(verifier)
    result = verify_code_completion(solution_str, verifier)
    signals = _format_signals(
        solution_str,
        format_ok=result.format_ok,
        compile_ok=result.compile_ok,
        exec_ok=result.exec_ok,
    )
    format_reward = float(signals["format_reward"])
    if result.ok:
        score = 1.0 if signals["no_think_tags"] else 0.97
    else:
        score = max(float(result.score), format_reward)
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
