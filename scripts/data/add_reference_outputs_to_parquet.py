#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


SAFE_MODULES = {
    "collections",
    "functools",
    "itertools",
    "math",
    "statistics",
}


def _safe_import(name: str, globals_: dict[str, Any] | None = None, locals_: dict[str, Any] | None = None,
                 fromlist: tuple[str, ...] = (), level: int = 0) -> Any:
    root = name.split(".", 1)[0]
    if root not in SAFE_MODULES:
        raise ImportError(f"module not allowed in reference answer: {name}")
    return __import__(name, globals_, locals_, fromlist, level)


SAFE_BUILTINS: dict[str, Any] = {
    "__import__": _safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "format": format,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
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


def code_from_reference(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in blocks:
        if "def answer" in block:
            return block.strip()
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def run_reference_answer(reference_text: str) -> tuple[bool, str]:
    code = code_from_reference(reference_text)
    namespace: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "math": math,
    }
    try:
        exec(code, namespace, namespace)
        answer = namespace.get("answer")
        if not callable(answer):
            return False, "missing callable answer"
        return True, repr(answer())
    except Exception as exc:  # noqa: BLE001 - this is an offline data annotation tool.
        return False, f"{type(exc).__name__}: {exc}"


def annotate(input_path: Path, output_path: Path) -> None:
    df = pd.read_parquet(input_path)
    ok_count = 0
    fail_count = 0
    failures: list[tuple[int, str]] = []

    new_extra = []
    for idx, extra in enumerate(df["extra_info"]):
        info = dict(extra or {})
        reference = str(info.get("reference_answer") or "")
        ok, result = run_reference_answer(reference)
        if ok:
            ok_count += 1
            info["reference_output"] = result
            info["reference_output_source"] = "executed_reference_answer"
        else:
            fail_count += 1
            info["reference_output_error"] = result
            failures.append((idx, result))
        new_extra.append(info)

    out_df = df.copy()
    out_df["extra_info"] = new_extra
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "rows": int(len(out_df)),
                "ok": ok_count,
                "failed": fail_count,
                "first_failures": failures[:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    annotate(args.input, args.output)


if __name__ == "__main__":
    main()
