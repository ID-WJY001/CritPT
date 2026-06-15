#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def rollout_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**9, path.name


def read_step_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prompt_key(row: dict[str, Any]) -> str:
    text = str(row.get("input", row.get("prompt", "")))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def analyze_step(path: Path) -> dict[str, Any]:
    rows = read_step_rows(path)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(prompt_key(row), []).append(row)

    useful = 0
    all_correct = 0
    all_wrong = 0
    group_records: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        scores = [number(row.get("score", row.get("reward"))) for row in group_rows]
        acc = [number(row.get("acc")) for row in group_rows]
        unique_scores = {round(score, 6) for score in scores}
        is_useful = len(unique_scores) > 1
        is_all_correct = bool(acc) and all(value >= 1.0 for value in acc)
        is_all_wrong = bool(acc) and all(value <= 0.0 for value in acc)
        useful += int(is_useful)
        all_correct += int(is_all_correct)
        all_wrong += int(is_all_wrong)
        group_records.append(
            {
                "prompt_key": key,
                "n": len(group_rows),
                "useful": is_useful,
                "all_correct": is_all_correct,
                "all_wrong": is_all_wrong,
                "score_mean": statistics.mean(scores) if scores else 0.0,
                "score_min": min(scores) if scores else 0.0,
                "score_max": max(scores) if scores else 0.0,
            }
        )

    scores = [number(row.get("score", row.get("reward"))) for row in rows]
    return {
        "step": int(path.stem) if path.stem.isdigit() else path.stem,
        "rows": len(rows),
        "groups": len(groups),
        "useful": useful,
        "all_correct": all_correct,
        "all_wrong": all_wrong,
        "score_mean": statistics.mean(scores) if scores else 0.0,
        "score_min": min(scores) if scores else 0.0,
        "score_max": max(scores) if scores else 0.0,
        "max_output_chars": max((len(str(row.get("output", ""))) for row in rows), default=0),
        "group_records": group_records,
    }


def write_csv(steps: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "step",
        "rows",
        "groups",
        "useful",
        "all_correct",
        "all_wrong",
        "score_mean",
        "score_min",
        "score_max",
        "max_output_chars",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in steps:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(summary: dict[str, Any], steps: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Rollout Group Signal Analysis",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Step Summary",
            "",
            "| step | groups | useful | all_correct | all_wrong | score_mean | score_min | score_max | max_output_chars |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in steps:
        lines.append(
            f"| {row['step']} | {row['groups']} | {row['useful']} | "
            f"{row['all_correct']} | {row['all_wrong']} | "
            f"{row['score_mean']:.4f} | {row['score_min']:.4f} | "
            f"{row['score_max']:.4f} | {row['max_output_chars']} |"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GRPO rollout score diversity within prompt groups.")
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    paths = sorted(args.rollout_dir.glob("*.jsonl"), key=rollout_sort_key)
    if not paths:
        raise SystemExit(f"no rollout jsonl files in {args.rollout_dir}")
    steps = [analyze_step(path) for path in paths]
    groups_total = sum(step["groups"] for step in steps)
    useful = sum(step["useful"] for step in steps)
    summary = {
        "steps": len(steps),
        "rows": sum(step["rows"] for step in steps),
        "groups": groups_total,
        "useful": useful,
        "useful_rate": useful / groups_total if groups_total else 0.0,
        "all_correct": sum(step["all_correct"] for step in steps),
        "all_wrong": sum(step["all_wrong"] for step in steps),
        "max_output_chars": max(step["max_output_chars"] for step in steps),
    }
    result = {"summary": summary, "steps": steps}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(steps, args.out_csv)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(summary, steps, args.out_md)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
