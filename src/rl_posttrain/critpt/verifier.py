from __future__ import annotations

import math
import re
from dataclasses import dataclass

import sympy as sp

from rl_posttrain.critpt.schema import VerifierSpec


ANSWER_PATTERNS = [
    re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE),
    re.compile(r"final\s+answer\s*[:：]\s*(.*)", re.DOTALL | re.IGNORECASE),
    re.compile(r"答案\s*[:：]\s*(.*)", re.DOTALL),
]


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
    return sp.sympify(expr.replace("^", "**"), locals=allowed)


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

