from __future__ import annotations

import ast
import importlib
import math
import multiprocessing as mp
import queue
import re
import traceback
from dataclasses import dataclass
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
SYMPY_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

ALLOWED_IMPORT_ROOTS = {"math", "numpy", "sympy", "fractions", "itertools", "functools"}
FORBIDDEN_CALL_NAMES = {
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "format": format,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "ord": ord,
    "pow": pow,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


@dataclass(frozen=True)
class CodeVerificationResult:
    ok: bool
    score: float
    reason: str
    format_ok: bool
    compile_ok: bool
    exec_ok: bool
    passed_checks: int
    total_checks: int
    extracted_code: str


def extract_python_code(text: str) -> str:
    match = CODE_BLOCK_RE.search(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def verify_code_completion(
    completion: str,
    verifier: dict[str, Any],
    timeout_s: float | None = None,
) -> CodeVerificationResult:
    code = extract_python_code(completion)
    timeout = float(timeout_s or verifier.get("timeout_s", 2.0))
    checks = list(verifier.get("checks", []))
    total_checks = len(checks)
    format_ok = _has_single_answer_function(code)

    compile_reason = _compile_safety_reason(code)
    if compile_reason is not None:
        return CodeVerificationResult(
            ok=False,
            score=0.15 if format_ok else 0.0,
            reason=compile_reason,
            format_ok=format_ok,
            compile_ok=False,
            exec_ok=False,
            passed_checks=0,
            total_checks=total_checks,
            extracted_code=code,
        )

    result_queue: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(target=_worker, args=(code, verifier, result_queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        return CodeVerificationResult(
            ok=False,
            score=0.25 if format_ok else 0.0,
            reason="timeout",
            format_ok=format_ok,
            compile_ok=True,
            exec_ok=False,
            passed_checks=0,
            total_checks=total_checks,
            extracted_code=code,
        )

    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        payload = {"ok": False, "reason": "worker_empty_result", "passed_checks": 0, "exec_ok": False}

    passed = int(payload.get("passed_checks", 0))
    exec_ok = bool(payload.get("exec_ok", False))
    ok = bool(payload.get("ok", False))
    if ok:
        score = 1.0
    elif exec_ok and total_checks:
        score = 0.45 + 0.35 * (passed / total_checks)
    elif format_ok:
        score = 0.25
    else:
        score = 0.0
    return CodeVerificationResult(
        ok=ok,
        score=score,
        reason=str(payload.get("reason", "unknown")),
        format_ok=format_ok,
        compile_ok=True,
        exec_ok=exec_ok,
        passed_checks=passed,
        total_checks=total_checks,
        extracted_code=code,
    )


def _has_single_answer_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return sum(isinstance(node, ast.FunctionDef) and node.name == "answer" for node in tree.body) == 1


def _compile_safety_reason(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax_error: {exc}"
    answer_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "answer"]
    if len(answer_defs) != 1:
        return "format_error: expected exactly one top-level answer() function"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in getattr(node, "names", [])]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                root = name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    return f"unsafe_import: {name}"
        elif isinstance(node, ast.Call):
            func_name = _call_name(node.func)
            if func_name in FORBIDDEN_CALL_NAMES:
                return f"unsafe_call: {func_name}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"unsafe_attribute: {node.attr}"
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _worker(code: str, verifier: dict[str, Any], result_queue: mp.Queue) -> None:
    try:
        namespace = _exec_code(code)
        answer = namespace.get("answer")
        if not callable(answer):
            result_queue.put({"ok": False, "reason": "missing_answer", "passed_checks": 0, "exec_ok": False})
            return

        checks = list(verifier.get("checks", []))
        for idx, check in enumerate(checks):
            args = [_decode_value(value) for value in check.get("args", [])]
            kwargs = {key: _decode_value(value) for key, value in check.get("kwargs", {}).items()}
            got = answer(*args, **kwargs)
            ok, reason = _compare_value(got, check)
            if not ok:
                result_queue.put(
                    {
                        "ok": False,
                        "reason": f"check_{idx}_failed: {reason}",
                        "passed_checks": idx,
                        "exec_ok": True,
                    }
                )
                return
        result_queue.put(
            {"ok": True, "reason": "all_checks_passed", "passed_checks": len(checks), "exec_ok": True}
        )
    except Exception as exc:
        result_queue.put(
            {
                "ok": False,
                "reason": f"runtime_error: {exc.__class__.__name__}: {exc}",
                "traceback": traceback.format_exc(limit=4),
                "passed_checks": 0,
                "exec_ok": False,
            }
        )


def _exec_code(code: str) -> dict[str, Any]:
    safe_globals = {
        "__builtins__": {**SAFE_BUILTINS, "__import__": _safe_import},
        "math": math,
        "sp": sp,
        "sympy": sp,
    }
    try:
        safe_globals["np"] = importlib.import_module("numpy")
        safe_globals["numpy"] = safe_globals["np"]
    except Exception:
        pass
    exec(compile(code, "<model_answer>", "exec"), safe_globals, safe_globals)
    return safe_globals


def _safe_import(name: str, globals_: object = None, locals_: object = None, fromlist: tuple = (), level: int = 0):
    del globals_, locals_, fromlist, level
    root = name.split(".", 1)[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"import blocked: {name}")
    return importlib.import_module(name)


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "$sym" in value:
            return sp.Symbol(str(value["$sym"]))
        if "$tuple" in value:
            return tuple(_decode_value(item) for item in value["$tuple"])
        if "$list" in value:
            return [_decode_value(item) for item in value["$list"]]
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _compare_value(got: Any, check: dict[str, Any]) -> tuple[bool, str]:
    mode = str(check.get("mode", "exact"))
    expected = _decode_value(check.get("expected"))
    tolerance = float(check.get("tolerance", 1e-8))
    if mode == "exact":
        return (got == expected, f"got={got!r}, expected={expected!r}")
    if mode == "exact_sequence":
        got_seq = list(got)
        exp_seq = list(expected)
        return (got_seq == exp_seq, f"got={got_seq!r}, expected={exp_seq!r}")
    if mode == "sequence_length":
        got_seq = list(got)
        expected_len = int(check.get("expected", check.get("length", 0)))
        return (len(got_seq) == expected_len, f"got={len(got_seq)}, expected={expected_len}")
    if mode == "numeric_sequence_item":
        got_seq = list(got)
        item_index = int(check["index"])
        if item_index < 0:
            item_index = len(got_seq) + item_index
        if item_index < 0 or item_index >= len(got_seq):
            return False, f"index_out_of_range: index={check['index']}, len={len(got_seq)}"
        return _numeric_close(got_seq[item_index], expected, tolerance)
    if mode == "set_exact":
        got_set = set(got)
        exp_set = set(expected)
        return (got_set == exp_set, f"got={got_set!r}, expected={exp_set!r}")
    if mode == "numeric":
        return _numeric_close(got, expected, tolerance)
    if mode == "numeric_sequence":
        got_seq = list(got)
        exp_seq = list(expected)
        if len(got_seq) != len(exp_seq):
            return False, f"length_mismatch: got={len(got_seq)}, expected={len(exp_seq)}"
        for idx, (got_item, exp_item) in enumerate(zip(got_seq, exp_seq)):
            ok, reason = _numeric_close(got_item, exp_item, tolerance)
            if not ok:
                return False, f"item_{idx}: {reason}"
        return True, "numeric_sequence_close"
    if mode == "symbolic":
        variables = [str(name) for name in check.get("variables", [])]
        return _symbolic_equal(got, str(check.get("expected", "")), variables, tolerance)
    return False, f"unknown_mode: {mode}"


def _numeric_close(got: Any, expected: Any, tolerance: float) -> tuple[bool, str]:
    try:
        got_val = float(got)
        exp_val = float(expected)
    except Exception as exc:
        return False, f"numeric_parse_error: {exc}"
    ok = math.isclose(got_val, exp_val, rel_tol=tolerance, abs_tol=tolerance)
    return ok, f"got={got_val}, expected={exp_val}, tol={tolerance}"


def _symbolic_equal(got: Any, expected: str, variables: list[str], tolerance: float) -> tuple[bool, str]:
    try:
        got_expr = _to_expr(got, variables)
        expected_expr = _to_expr(expected, variables)
    except Exception as exc:
        return False, f"symbolic_parse_error: {exc}"
    try:
        if sp.simplify(got_expr - expected_expr) == 0:
            return True, "symbolic_equal"
    except Exception:
        pass
    tests = _default_symbolic_tests(variables)
    for subs in tests:
        try:
            got_val = float(got_expr.evalf(subs=subs))
            exp_val = float(expected_expr.evalf(subs=subs))
        except Exception as exc:
            return False, f"symbolic_numeric_error: {exc}"
        if not math.isclose(got_val, exp_val, rel_tol=tolerance, abs_tol=tolerance):
            return False, f"got={got_val}, expected={exp_val}, subs={subs}"
    return True, "symbolic_numeric_equal"


def _to_expr(value: Any, variables: list[str]) -> sp.Expr:
    if isinstance(value, sp.Expr):
        return value
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
        "E": sp.E,
    }
    return parse_expr(str(value), local_dict=allowed, transformations=SYMPY_TRANSFORMATIONS)


def _default_symbolic_tests(variables: list[str]) -> list[dict[sp.Symbol, float]]:
    if not variables:
        return [{}]
    base_values = [0.13, 0.27, 0.41]
    tests: list[dict[sp.Symbol, float]] = []
    for offset in range(3):
        tests.append(
            {
                sp.Symbol(name): base_values[(idx + offset) % len(base_values)] + 0.03 * idx
                for idx, name in enumerate(variables)
            }
        )
    return tests
