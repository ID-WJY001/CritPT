#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NUMERIC_KEYS = [
    "score",
    "reward",
    "raw_judge_score",
    "acc",
    "judge_error",
    "quick_reject",
    "correctness",
    "instruction_following",
    "reasoning_quality",
    "final_answer_consistency",
    "answer_marker_present",
    "output_chars",
    "post_final_chars",
    "length_penalty",
    "no_final_cap_applied",
    "post_final_penalty_applied",
    "fatal_error",
]

REASON_KEYWORDS = [
    "truncated",
    "incomplete",
    "no final",
    "never computes",
    "never states",
    "wrong",
    "incorrect",
    "format",
    "rambling",
    "too long",
    "overlong",
    "no final answer",
    "final answer",
]

FINAL_ANSWER_RE = re.compile(
    r"(最终答案\s*[:：]|final\s+answer\s*[:：]|answer\s*[:：]|答案\s*[:：])",
    re.IGNORECASE,
)


def rollout_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.stem), path.name)
    except ValueError:
        return (10**9, path.name)


def read_rows(rollout_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(rollout_dir.glob("*.jsonl"), key=rollout_sort_key):
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row.setdefault("step", path.stem)
                row["_rollout_file"] = path.name
                row["_rollout_line"] = line_no
                row["_row_index"] = len(rows)
                rows.append(row)
    return rows


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def quantile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = q * (len(ordered) - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        frac = pos - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": ordered[0],
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "max": ordered[-1],
    }


def summarize_numeric(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for key in NUMERIC_KEYS:
        values = [number for row in rows if (number := finite_float(row.get(key))) is not None]
        if values:
            out[key] = stats(values)
    out["input_chars"] = stats([float(len(str(row.get("input", "")))) for row in rows])
    out["output_chars"] = stats([float(len(str(row.get("output", "")))) for row in rows])
    out["answer_marker_present"] = stats([float(answer_marker_stats(str(row.get("output", "")))[0]) for row in rows])
    out["post_final_chars"] = stats([float(answer_marker_stats(str(row.get("output", "")))[1]) for row in rows])
    return out


def answer_marker_stats(text: str) -> tuple[int, int]:
    last_match: re.Match[str] | None = None
    for match in FINAL_ANSWER_RE.finditer(text):
        last_match = match
    if last_match is None:
        return 0, 0
    return 1, len(text[last_match.end() :].strip())


def count_reason_keywords(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        reason = str(row.get("reason", "")).lower()
        for keyword in REASON_KEYWORDS:
            if keyword in reason:
                counter[keyword] += 1
    return dict(counter)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_step_csv(rows: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        step_number = finite_float(row.get("step"))
        step = int(step_number) if step_number is not None else -1
        grouped[step].append(row)

    fields = ["step", "count"]
    for key in NUMERIC_KEYS:
        fields.extend([f"{key}_mean", f"{key}_min", f"{key}_max"])
    fields.extend(["input_chars_mean", "output_chars_mean", "answer_marker_rate", "post_final_chars_mean"])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in sorted(grouped):
            step_rows = grouped[step]
            record: dict[str, Any] = {"step": step, "count": len(step_rows)}
            for key in NUMERIC_KEYS:
                values = [number for row in step_rows if (number := finite_float(row.get(key))) is not None]
                if values:
                    record[f"{key}_mean"] = statistics.mean(values)
                    record[f"{key}_min"] = min(values)
                    record[f"{key}_max"] = max(values)
            record["input_chars_mean"] = statistics.mean(len(str(row.get("input", ""))) for row in step_rows)
            record["output_chars_mean"] = statistics.mean(len(str(row.get("output", ""))) for row in step_rows)
            marker_stats = [answer_marker_stats(str(row.get("output", ""))) for row in step_rows]
            record["answer_marker_rate"] = statistics.mean(value[0] for value in marker_stats)
            record["post_final_chars_mean"] = statistics.mean(value[1] for value in marker_stats)
            writer.writerow(record)


def truncate_text(value: Any, limit: int = 700) -> str:
    text = str(value or "").strip()
    text = text.replace("\r\n", "\n")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n..."


def score_value(row: dict[str, Any]) -> float | None:
    return finite_float(row.get("score", row.get("reward")))


def write_review(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        step_number = finite_float(row.get("step"))
        step = int(step_number) if step_number is not None else -1
        grouped[step].append(row)

    lines = [f"# Rollout Review: {title or path.parent.name}", "", "## Step Summary", ""]
    lines.append("| step | n | score mean | acc mean | marker mean | score counts |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
    for step in sorted(grouped):
        step_rows = grouped[step]
        scores = [score for row in step_rows if (score := score_value(row)) is not None]
        acc_values = [
            float(row["acc"])
            for row in step_rows
            if finite_float(row.get("acc")) is not None
        ]
        marker_values = [
            float(answer_marker_stats(str(row.get("output", "")))[0])
            for row in step_rows
        ]
        counts = Counter(round(score, 4) for score in scores)
        lines.append(
            f"| {step} | {len(step_rows)} | "
            f"{(statistics.mean(scores) if scores else 0.0):.4f} | "
            f"{(statistics.mean(acc_values) if acc_values else 0.0):.4f} | "
            f"{(statistics.mean(marker_values) if marker_values else 0.0):.4f} | "
            f"`{dict(counts)}` |"
        )

    lines.extend(["", "## Lowest Examples By Step", ""])
    for step in sorted(grouped):
        scored_rows = [
            row for row in grouped[step] if score_value(row) is not None
        ]
        scored_rows.sort(key=lambda row: float(score_value(row) or 0.0))
        if not scored_rows:
            continue
        lines.extend([f"### Step {step}", ""])
        for row in scored_rows[:3]:
            score = float(score_value(row) or 0.0)
            acc = finite_float(row.get("acc"))
            reason = truncate_text(row.get("reason", ""), 220).replace("\n", " ")
            extracted = truncate_text(row.get("extracted_answer", ""), 300).replace("\n", " ")
            output = truncate_text(row.get("output", ""), 700)
            lines.append(
                f"- score `{score:.4f}`, acc `{(acc if acc is not None else 0.0):.1f}`, "
                f"reason `{reason}`, extracted `{extracted}`"
            )
            lines.append("")
            lines.append("```text")
            lines.append(output)
            lines.append("```")
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def top_indices(rows: list[dict[str, Any]], reverse: bool) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        score = finite_float(row.get("score", row.get("reward")))
        if score is not None:
            scored.append((score, int(row["_row_index"]), row))
    scored.sort(reverse=reverse)
    return [
        {
            "row_index": row["_row_index"],
            "step": row.get("step"),
            "score": score,
            "reason": row.get("reason", ""),
        }
        for score, _, row in scored[:12]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    rows = read_rows(args.rollout_dir)
    if not rows:
        raise SystemExit(f"no rollout jsonl files in {args.rollout_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, args.out_dir / "rollouts.full.jsonl")
    (args.out_dir / "rollouts.full.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_step_csv(rows, args.out_dir / "rollout_step_summary.csv")
    write_review(rows, args.out_dir / "rollout_review.md", args.run_name)

    summary = {
        "run_name": args.run_name,
        "rollout_dir": str(args.rollout_dir),
        "row_count": len(rows),
        "steps": sorted({row.get("step") for row in rows}, key=lambda value: int(value)),
        "numeric": summarize_numeric(rows),
        "reason_keyword_counts": count_reason_keywords(rows),
        "worst": top_indices(rows, reverse=False),
        "best": top_indices(rows, reverse=True),
    }
    (args.out_dir / "rollout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"rows: {len(rows)}")
    print(f"jsonl: {args.out_dir / 'rollouts.full.jsonl'}")
    print(f"json: {args.out_dir / 'rollouts.full.json'}")
    print(f"summary: {args.out_dir / 'rollout_summary.json'}")
    print(f"step_csv: {args.out_dir / 'rollout_step_summary.csv'}")
    print(f"review: {args.out_dir / 'rollout_review.md'}")


if __name__ == "__main__":
    main()
