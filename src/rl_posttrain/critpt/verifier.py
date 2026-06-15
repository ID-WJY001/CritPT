from __future__ import annotations

import math
import re
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from rl_posttrain.critpt.schema import VerifierSpec


ANSWER_PATTERNS = [
    re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE),
    re.compile(r"final\s+answer\s*[:：]\s*(.*)", re.DOTALL | re.IGNORECASE),
    re.compile(r"答案\s*[:：]\s*(.*)", re.DOTALL),
    re.compile(r"\\boxed\s*\{(.*)\}\s*$", re.DOTALL),
]

SYMPY_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    score: float
    reason: str
    extracted: str


def extract_answer(text: str) -> str:
    stripped = text.strip()
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(stripped)
        if match:
            return match.group(1).strip().rstrip(".")
    return stripped.rstrip(".")


def verify_completion(completion: str, spec: VerifierSpec) -> VerificationResult:
    extracted = extract_answer(completion)
    if spec.kind == "exact":
        ok = extracted.strip() == spec.expected.strip()
        return VerificationResult(ok, 1.0 if ok else 0.0, "exact", extracted)
    if spec.kind == "numeric":
        return _verify_numeric(extracted, spec)
    if spec.kind == "symbolic":
        return _verify_symbolic(extracted, spec)
    return VerificationResult(False, 0.0, f"unknown verifier kind: {spec.kind}", extracted)


def _safe_parse(expr: str, variables: list[str]) -> sp.Expr:
    symbols = {name: sp.Symbol(name) for name in variables}
    allowed = {
        **symbols,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "pi": sp.pi,
    }
    return parse_expr(
        _normalise_symbolic_expr(expr),
        local_dict=allowed,
        transformations=SYMPY_TRANSFORMATIONS,
        evaluate=True,
    )


def _normalise_symbolic_expr(expr: str) -> str:
    text = expr.strip()
    text = _strip_math_wrappers(text)
    text = _drop_left_hand_side(text)
    text = _latex_frac_to_python(text)
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\!": "",
        "\\;": "",
        "\\:": "",
        "\\cdot": "*",
        "\\times": "*",
        "\\rm": "",
        "{": "(",
        "}": ")",
        "^": "**",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\[a-zA-Z]+\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text


def _strip_math_wrappers(text: str) -> str:
    stripped = text.strip()
    for prefix, suffix in (("$$", "$$"), ("\\[", "\\]"), ("$", "$")):
        if stripped.startswith(prefix) and stripped.endswith(suffix):
            return stripped[len(prefix) : -len(suffix)].strip()
    boxed = re.fullmatch(r"\\boxed\s*\{(.+)\}", stripped, flags=re.DOTALL)
    if boxed:
        return boxed.group(1).strip()
    return stripped


def _drop_left_hand_side(text: str) -> str:
    if "=" not in text:
        return text
    left, right = text.split("=", 1)
    if re.search(r"[A-Za-z]\\|F|fidelity|logical|physical|rm", left):
        return right.strip()
    return text


def _latex_frac_to_python(text: str) -> str:
    for token in ("\\tfrac", "\\dfrac", "\\frac"):
        while token in text:
            start = text.find(token)
            numerator_start = start + len(token)
            numerator, numerator_end = _read_latex_group(text, numerator_start)
            denominator, denominator_end = _read_latex_group(text, numerator_end)
            if numerator is None or denominator is None:
                break
            converted = f"(({_latex_frac_to_python(numerator)})/({_latex_frac_to_python(denominator)}))"
            text = text[:start] + converted + text[denominator_end:]
    return text


def _read_latex_group(text: str, start: int) -> tuple[str | None, int]:
    pos = start
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text) or text[pos] != "{":
        return None, start
    depth = 0
    for idx in range(pos, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[pos + 1 : idx], idx + 1
    return None, start


def _verify_symbolic(extracted: str, spec: VerifierSpec) -> VerificationResult:
    try:
        got = _safe_parse(extracted, spec.variables)
        expected = _safe_parse(spec.expected, spec.variables)
    except Exception as exc:
        return VerificationResult(False, 0.0, f"parse_error: {exc}", extracted)

    try:
        if sp.simplify(got - expected) == 0:
            return VerificationResult(True, 1.0, "symbolic_equal", extracted)
    except Exception:
        pass

    numeric = _numeric_equivalence(got, expected, spec)
    if numeric:
        return VerificationResult(True, 1.0, "numeric_equal", extracted)
    return VerificationResult(False, 0.0, "not_equivalent", extracted)


def _verify_numeric(extracted: str, spec: VerifierSpec) -> VerificationResult:
    try:
        got = float(extracted)
        expected = float(spec.expected)
    except ValueError as exc:
        return VerificationResult(False, 0.0, f"parse_error: {exc}", extracted)
    ok = math.isclose(got, expected, rel_tol=spec.tolerance, abs_tol=spec.tolerance)
    return VerificationResult(ok, 1.0 if ok else 0.0, "numeric", extracted)


def _numeric_equivalence(got: sp.Expr, expected: sp.Expr, spec: VerifierSpec) -> bool:
    if not spec.numeric_tests:
        return False
    for test in spec.numeric_tests:
        subs = {sp.Symbol(k): v for k, v in test.items()}
        try:
            got_val = float(got.evalf(subs=subs))
            expected_val = float(expected.evalf(subs=subs))
        except Exception:
            return False
        if not math.isclose(
            got_val,
            expected_val,
            rel_tol=spec.tolerance,
            abs_tol=spec.tolerance,
        ):
            return False
    return True
