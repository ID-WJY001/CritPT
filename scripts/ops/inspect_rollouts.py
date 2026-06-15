#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
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


def clip(text: Any, limit: int) -> str:
    raw = "" if text is None else str(text)
    raw = raw.replace("\r\n", "\n")
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "\n...[truncated]..."


def read_rows(rollout_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(rollout_dir.glob("*.jsonl"), key=lambda p: (len(p.stem), p.stem)):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["_file"] = str(path)
                    rows.append(row)
    return rows


def numeric_summary(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for key in NUMERIC_KEYS:
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
        if not values:
            continue
        lines.append(
            f"- `{key}`: n={len(values)}, mean={statistics.mean(values):.4f}, "
            f"min={min(values):.4f}, max={max(values):.4f}"
        )
    return lines


def render_example(row: dict[str, Any], idx: int, text_limit: int) -> str:
    score = row.get("score", row.get("reward", ""))
    reason = row.get("reason", "")
    parts = [
        f"### Example {idx}",
        "",
        f"- step: `{row.get('step', '')}`",
        f"- score: `{score}`",
        f"- correctness: `{row.get('correctness', '')}`",
        f"- judge_error: `{row.get('judge_error', '')}`",
        f"- source: `{row.get('_file', '')}`",
        "",
        "**Prompt**",
        "",
        "```text",
        clip(row.get("input", ""), text_limit),
        "```",
        "",
        "**Output**",
        "",
        "```text",
        clip(row.get("output", ""), text_limit),
        "```",
    ]
    if reason:
        parts.extend(["", "**Judge Reason**", "", "```text", clip(reason, text_limit), "```"])
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--text-limit", type=int, default=1800)
    args = parser.parse_args()

    rows = read_rows(args.rollout_dir)
    if not rows:
        raise SystemExit(f"no rollout jsonl files in {args.rollout_dir}")

    def score(row: dict[str, Any]) -> float:
        try:
            return float(row.get("score", row.get("reward", 0.0)))
        except (TypeError, ValueError):
            return 0.0

    worst = sorted(rows, key=score)[: args.limit]
    best = sorted(rows, key=score, reverse=True)[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc = [
        "# Rollout Review",
        "",
        f"- rollout_dir: `{args.rollout_dir}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Numeric Summary",
        "",
        *numeric_summary(rows),
        "",
        "## Worst Cases",
        "",
        *(render_example(row, idx + 1, args.text_limit) for idx, row in enumerate(worst)),
        "",
        "## Best Cases",
        "",
        *(render_example(row, idx + 1, args.text_limit) for idx, row in enumerate(best)),
        "",
    ]
    args.out.write_text("\n".join(doc), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
