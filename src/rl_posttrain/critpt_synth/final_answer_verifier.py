from __future__ import annotations

import ast
import json
import math
import operator
import os
import re
from dataclasses import dataclass
from typing import Any

from rl_posttrain.critpt_synth.verifier import _compare_value, _decode_value


FINAL_ANSWER_RE = re.compile(
    r"(最终答案\s*[:：]|final\s+answer\s*[:：]|answer\s*[:：]|答案\s*[:：])",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
SKIP_PHRASE_RE = re.compile(
    r"(继续递推|递推至|展开可得|可得|using known expansions|by expansion|similarly|省略|略)",
    re.IGNORECASE,
)
SUPERSCRIPT_DIGITS = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
    }
)


@dataclass(frozen=True)
class FinalAnswerVerificationResult:
    ok: bool
    score: float
    reason: str
    answer_marker_present: bool
    parse_ok: bool
    passed_checks: int
    total_checks: int
    extracted_answer: str
    skip_phrase_present: bool


def extract_final_answer(text: str) -> tuple[str, bool, int]:
    stripped = text.strip()
    last_match: re.Match[str] | None = None
    for match in FINAL_ANSWER_RE.finditer(stripped):
        last_match = match
    if last_match is None:
        return stripped, False, 0
    answer = stripped[last_match.end() :].strip()
    return answer, True, len(answer)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    if stripped.startswith("`") and stripped.endswith("`") and stripped.count("`") == 2:
        return stripped[1:-1].strip()
    return stripped


def _unwrap_latex(text: str) -> str:
    out = text.strip()
    out = out.replace("，", ",")
    out = out.replace("（", "(").replace("）", ")")
    out = out.replace("−", "-").replace("–", "-")
    out = out.replace("×", "*").replace("·", "*").replace("⋅", "*")
    out = _replace_unicode_superscripts(out)
    out = out.replace("₀", "0").replace("₁", "1").replace("₂", "2").replace("₃", "3")
    out = out.replace("₄", "4").replace("₅", "5").replace("₆", "6").replace("₇", "7")
    out = out.replace("₈", "8").replace("₉", "9")
    latex_names = {
        "\\Omega": "drive",
        "\\omega": "omega",
        "\\alpha": "alpha",
        "\\delta": "delta",
        "\\gamma": "gamma",
        "\\theta": "theta",
        "\\lambda": "lambda",
    }
    for latex_name, plain_name in latex_names.items():
        out = out.replace(latex_name, plain_name)
    unicode_names = {
        "Ω": "drive",
        "ω": "omega",
        "δ": "delta",
        "Δ": "delta",
        "α": "alpha",
        "γ": "gamma",
        "θ": "theta",
        "λ": "lambda",
        "π": "pi",
    }
    for unicode_name, plain_name in unicode_names.items():
        out = out.replace(unicode_name, plain_name)
    out = _replace_unicode_superscripts(out)
    out = out.replace("\\cdot", "*").replace("\\times", "*")
    out = out.replace("^", "**")
    out = _unwrap_latex_single_arg_command(out, "boxed")
    out = out.replace("\\langle", "").replace("\\rangle", "")
    out = re.sub(r"\\left|\\right", "", out)
    out = re.sub(r"\\text\s*\{[^{}]*\}", "", out)
    out = out.replace("$", "").replace("\\(", "").replace("\\)", "")
    out = _replace_latex_fracs(out)
    out = _replace_latex_sqrts(out)
    out = re.sub(r"√\s*\(([^()]*)\)", r"sqrt(\1)", out)
    out = re.sub(r"√\s*\{([^{}]*)\}", r"sqrt(\1)", out)
    out = re.sub(r"√\s*([A-Za-z_][A-Za-z0-9_]*(?:\*\*\d+)?)", r"sqrt(\1)", out)
    out = re.sub(r"\\([A-Za-z]+)", r"\1", out)
    return out.strip()


def _replace_unicode_superscripts(text: str) -> str:
    return re.sub(
        r"([A-Za-z0-9_)\]])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        lambda match: f"{match.group(1)}**{match.group(2).translate(SUPERSCRIPT_DIGITS)}",
        text,
    )


def _replace_latex_fracs(text: str) -> str:
    pattern = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    out = text
    while previous != out:
        previous = out
        out = pattern.sub(r"((\1)/(\2))", out)
    return out


def _replace_latex_sqrts(text: str) -> str:
    out = text
    needle = "\\sqrt"
    while True:
        start = out.find(needle)
        if start < 0:
            return out
        brace_start = out.find("{", start + len(needle))
        if brace_start < 0:
            return out
        depth = 0
        brace_end = -1
        for idx in range(brace_start, len(out)):
            if out[idx] == "{":
                depth += 1
            elif out[idx] == "}":
                depth -= 1
                if depth == 0:
                    brace_end = idx
                    break
        if brace_end < 0:
            return out
        inner = out[brace_start + 1 : brace_end]
        out = f"{out[:start]}sqrt({inner}){out[brace_end + 1:]}"


def _unwrap_latex_single_arg_command(text: str, command: str) -> str:
    out = text
    needle = f"\\{command}"
    while True:
        start = out.find(needle)
        if start < 0:
            return out
        brace_start = out.find("{", start + len(needle))
        if brace_start < 0:
            return out
        depth = 0
        brace_end = -1
        for idx in range(brace_start, len(out)):
            if out[idx] == "{":
                depth += 1
            elif out[idx] == "}":
                depth -= 1
                if depth == 0:
                    brace_end = idx
                    break
        if brace_end < 0:
            return out
        inner = out[brace_start + 1 : brace_end]
        out = f"{out[:start]}{inner}{out[brace_end + 1:]}"


def _remove_assignment_prefix(text: str) -> str:
    candidate = text.strip()
    first_line = next((line.strip() for line in candidate.splitlines() if line.strip()), candidate)
    candidate = first_line
    if "=" in candidate:
        rhs = candidate.rsplit("=", 1)[-1].strip()
        if rhs:
            candidate = rhs
    candidate = candidate.rstrip(".。；;")
    return candidate.strip()


def _parse_literal(text: str) -> Any:
    candidate = _remove_assignment_prefix(_unwrap_latex(_strip_code_fence(text)))
    candidate = candidate.replace("...", "")
    try:
        return ast.literal_eval(candidate)
    except Exception:
        sequence = _parse_numeric_expression_sequence(candidate)
        if sequence is not None:
            return sequence
        return candidate


def _parse_numbers(text: str) -> list[float]:
    candidate = _unwrap_latex(text).replace("...", "")
    return [float(match.group(0)) for match in NUMBER_RE.finditer(candidate)]


def _parse_numeric_expression_sequence(text: str) -> list[float | int] | None:
    candidate = text.strip()
    if not (
        (candidate.startswith("[") and candidate.endswith("]"))
        or (candidate.startswith("(") and candidate.endswith(")"))
    ):
        return None
    inner = candidate[1:-1].strip()
    if not inner:
        return []
    parts = _split_top_level_commas(inner)
    parsed: list[float | int] = []
    for part in parts:
        value = _safe_eval_numeric_expression(part)
        if value is None:
            return None
        parsed.append(value)
    return parsed


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for idx, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _safe_eval_numeric_expression(text: str) -> float | int | None:
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def eval_node(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            left = eval_node(node.left)
            right = eval_node(node.right)
            return operators[type(node.op)](left, right)
        raise ValueError("unsupported_numeric_expression")

    try:
        value = eval_node(ast.parse(text.strip(), mode="eval"))
    except Exception:
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _sequence_close(got: list[Any], expected: list[Any], tolerance: float, numeric: bool) -> tuple[bool, int, str]:
    if len(got) != len(expected):
        return False, 0, f"length_mismatch: got={len(got)}, expected={len(expected)}"
    passed = 0
    for idx, (got_item, expected_item) in enumerate(zip(got, expected)):
        if numeric:
            try:
                ok = math.isclose(float(got_item), float(expected_item), rel_tol=tolerance, abs_tol=tolerance)
            except Exception:
                ok = False
        else:
            ok = got_item == expected_item
        if not ok:
            return False, passed, f"item_{idx}: got={got_item!r}, expected={expected_item!r}"
        passed += 1
    return True, passed, "sequence_match"


def _candidate_for_check(answer: str, check: dict[str, Any], *, completion: str = "") -> Any:
    mode = str(check.get("mode", "exact"))
    expected = _decode_value(check.get("expected"))
    if mode == "text_numeric":
        source = completion if str(check.get("scope", "completion")) == "completion" else answer
        return _tagged_numeric_value(source, str(check.get("tag", "")))
    if mode in {"numeric_sequence", "exact_sequence", "set_exact", "sequence_length", "numeric_sequence_item"}:
        return _sequence_candidate(answer, exact=mode in {"exact_sequence", "set_exact", "sequence_length"})
    if mode == "numeric":
        numbers = _parse_numbers(answer)
        if numbers:
            return numbers[-1]
    if mode == "exact":
        literal = _parse_literal(answer)
        if isinstance(expected, int) and isinstance(literal, float) and literal.is_integer():
            return int(literal)
        return literal
    if mode == "symbolic":
        return _remove_assignment_prefix(_unwrap_latex(answer))
    return _parse_literal(answer)


def _tagged_numeric_value(text: str, tag: str) -> float | int | None:
    if not tag:
        return None
    normalized = _unwrap_latex(text)
    tag_pattern = re.escape(tag)
    patterns = [
        rf"{tag_pattern}\s*(?:=|:|：)\s*([^\n,;；，]+)",
        rf"{tag_pattern}\s+([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1).strip().rstrip(".。")
        value = _safe_eval_numeric_expression(candidate)
        if value is not None:
            return value
        numbers = _parse_numbers(candidate)
        if numbers:
            number = numbers[0]
            return int(number) if float(number).is_integer() else number
    return None


def _sequence_candidate(answer: str, *, exact: bool) -> list[Any]:
    literal = _parse_literal(answer)
    if isinstance(literal, (list, tuple, set)):
        return list(literal)
    numbers = _parse_numbers(answer)
    if numbers:
        if exact:
            return [int(value) if float(value).is_integer() else value for value in numbers]
        return numbers
    return []


def verify_final_answer(
    completion: str,
    verifier: dict[str, Any] | str,
    *,
    tolerance: float | None = None,
) -> FinalAnswerVerificationResult:
    if isinstance(verifier, str):
        verifier = json.loads(verifier)
    checks = list(verifier.get("reward_checks") or verifier.get("checks", []))
    answer, has_marker, _ = extract_final_answer(completion)
    skip_phrase = bool(SKIP_PHRASE_RE.search(completion))
    total = len(checks)
    if total == 0:
        return FinalAnswerVerificationResult(
            ok=False,
            score=0.0,
            reason="no_verifier_checks",
            answer_marker_present=has_marker,
            parse_ok=False,
            passed_checks=0,
            total_checks=0,
            extracted_answer=answer,
            skip_phrase_present=skip_phrase,
        )
    if not answer.strip():
        return FinalAnswerVerificationResult(
            ok=False,
            score=0.0,
            reason="empty_final_answer",
            answer_marker_present=has_marker,
            parse_ok=False,
            passed_checks=0,
            total_checks=total,
            extracted_answer=answer,
            skip_phrase_present=skip_phrase,
        )

    passed = 0
    parse_ok = True
    reasons: list[str] = []
    tol = float(tolerance if tolerance is not None else os.environ.get("LOCAL_FINAL_TOLERANCE", "1e-4"))
    for idx, raw_check in enumerate(checks):
        check = dict(raw_check)
        check["tolerance"] = max(float(check.get("tolerance", tol)), tol)
        mode = str(check.get("mode", "exact"))
        got = _candidate_for_check(answer, check, completion=completion)
        try:
            if mode == "sequence_length":
                expected_len = int(check.get("expected", check.get("length", 0)))
                got_len = len(list(got))
                if got_len != expected_len:
                    reasons.append(f"check_{idx}_failed: length_mismatch: got={got_len}, expected={expected_len}")
                    continue
            elif mode == "numeric_sequence_item":
                got_seq = list(got)
                item_index = int(check["index"])
                if item_index < 0:
                    item_index = len(got_seq) + item_index
                if item_index < 0 or item_index >= len(got_seq):
                    reasons.append(f"check_{idx}_failed: index_out_of_range: index={check['index']}, len={len(got_seq)}")
                    continue
                ok, reason = _compare_value(got_seq[item_index], {**check, "mode": "numeric"})
                if not ok:
                    reasons.append(f"check_{idx}_failed: item_{check['index']}: {reason}")
                    continue
            elif mode in {"numeric_sequence", "exact_sequence"}:
                expected = list(_decode_value(check.get("expected")))
                got_seq = list(got)
                ok, _partial, reason = _sequence_close(got_seq, expected, float(check["tolerance"]), mode == "numeric_sequence")
                if not ok:
                    reasons.append(f"check_{idx}_failed: {reason}")
                    continue
            elif mode == "text_numeric":
                ok, reason = _compare_value(got, {**check, "mode": "numeric"})
                if not ok:
                    reasons.append(f"check_{idx}_failed: tag_{check.get('tag', '')}: {reason}")
                    continue
            else:
                ok, reason = _compare_value(got, check)
                if not ok:
                    reasons.append(f"check_{idx}_failed: {reason}")
                    continue
        except Exception as exc:
            parse_ok = False
            reasons.append(f"check_{idx}_parse_error: {exc}")
            continue
        passed += 1

    ok = passed == total and parse_ok
    if ok:
        reasons.append("all_checks_passed")
    return _result(ok, has_marker, parse_ok, passed, total, answer, skip_phrase, reasons)


def _result(
    ok: bool,
    has_marker: bool,
    parse_ok: bool,
    passed: int,
    total: int,
    answer: str,
    skip_phrase: bool,
    reasons: list[str],
) -> FinalAnswerVerificationResult:
    if ok:
        score = 1.0
        if not has_marker:
            score = 0.82
        if skip_phrase:
            score = min(score, float(os.environ.get("LOCAL_FINAL_SKIP_PHRASE_CAP", "0.65")))
    else:
        score = 0.0
        if has_marker:
            score += 0.12
        if parse_ok:
            score += 0.12
        if total:
            score += 0.55 * (min(passed, total) / total)
        if skip_phrase:
            score = min(score, float(os.environ.get("LOCAL_FINAL_SKIP_PHRASE_CAP", "0.65")))
    score = max(0.0, min(1.0, score))
    return FinalAnswerVerificationResult(
        ok=ok,
        score=score,
        reason="; ".join(reasons)[:500],
        answer_marker_present=has_marker,
        parse_ok=parse_ok,
        passed_checks=passed,
        total_checks=total,
        extracted_answer=answer[:500],
        skip_phrase_present=skip_phrase,
    )
