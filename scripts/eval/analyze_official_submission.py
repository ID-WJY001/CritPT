#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from pathlib import Path
from typing import Any


FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def extract_code(text: str) -> tuple[str, bool, bool]:
    blocks = FENCE_RE.findall(text)
    has_fence = bool(blocks)
    unclosed = text.count("```") % 2 == 1
    if blocks:
        return blocks[0].strip(), has_fence, unclosed
    return text.strip(), has_fence, unclosed


def parse_error(code: str) -> str | None:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg}"
    except Exception as exc:  # pragma: no cover - defensive for unusual parser errors
        return f"{type(exc).__name__}: {exc}"
    return None


def has_answer_func(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return bool(re.search(r"\bdef\s+answer\s*\(", code))
    return any(isinstance(node, ast.FunctionDef) and node.name == "answer" for node in tree.body)


def analyze_submission(item: dict[str, Any]) -> dict[str, Any]:
    raw = str(item.get("generated_code", ""))
    code, has_fence, unclosed = extract_code(raw)
    err = parse_error(code)
    return {
        "problem_id": item.get("problem_id", ""),
        "chars": len(raw),
        "code_chars": len(code),
        "has_fence": has_fence,
        "unclosed_code_block": unclosed,
        "has_think": "<think>" in raw.lower() or "</think>" in raw.lower(),
        "has_def_answer_raw": bool(re.search(r"\bdef\s+answer\s*\(", raw)),
        "has_def_answer": has_answer_func(code),
        "parse_ok": err is None,
        "parse_error": err,
        "generated_head": raw[:1200],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chars = [float(row["chars"]) for row in rows]
    return {
        "n": len(rows),
        "chars_mean": statistics.mean(chars) if chars else 0.0,
        "chars_p50": percentile(chars, 0.50),
        "chars_p90": percentile(chars, 0.90),
        "chars_max": max(chars) if chars else 0.0,
        "has_think": sum(1 for row in rows if row["has_think"]),
        "has_fence": sum(1 for row in rows if row["has_fence"]),
        "unclosed_code_block": sum(1 for row in rows if row["unclosed_code_block"]),
        "has_def_answer_raw": sum(1 for row in rows if row["has_def_answer_raw"]),
        "has_def_answer": sum(1 for row in rows if row["has_def_answer"]),
        "parse_ok": sum(1 for row in rows if row["parse_ok"]),
    }


def write_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    bad = [row for row in rows if not row["parse_ok"] or not row["has_def_answer"] or row["unclosed_code_block"]]
    longest = sorted(rows, key=lambda row: row["chars"], reverse=True)[:10]
    lines = [
        "# Official Submission Static Analysis",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Bad Parse / Missing Answer", ""])
    if not bad:
        lines.append("No parse or answer-function failures found.")
    for row in bad[:20]:
        lines.extend(
            [
                f"### {row['problem_id']}",
                "",
                f"- chars: `{row['chars']}`",
                f"- parse_ok: `{row['parse_ok']}`",
                f"- has_def_answer: `{row['has_def_answer']}`",
                f"- unclosed_code_block: `{row['unclosed_code_block']}`",
                f"- parse_error: `{row['parse_error']}`",
                "",
                "```text",
                str(row["generated_head"]),
                "```",
                "",
            ]
        )
    lines.extend(["", "## Longest Outputs", ""])
    lines.append("| problem_id | chars | parse_ok | has_def_answer |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in longest:
        lines.append(
            f"| `{row['problem_id']}` | {row['chars']} | "
            f"{int(row['parse_ok'])} | {int(row['has_def_answer'])} |"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated CritPt official submission JSON.")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.batch.read_text(encoding="utf-8"))
    submissions = payload.get("submissions", [])
    if not isinstance(submissions, list) or not submissions:
        raise SystemExit(f"no submissions in {args.batch}")
    rows = [analyze_submission(item) for item in submissions]
    result = {
        "batch": str(args.batch),
        "summary": summarize(rows),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(result["summary"], rows, args.out_md)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
