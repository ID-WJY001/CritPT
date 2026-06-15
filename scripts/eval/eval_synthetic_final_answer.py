#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rl_posttrain.critpt_synth.final_answer_verifier import verify_final_answer
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample


def read_examples(path: Path) -> dict[str, SyntheticCritPTExample]:
    examples: dict[str, SyntheticCritPTExample] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                example = SyntheticCritPTExample.from_dict(json.loads(line))
                examples[example.problem_id] = example
    return examples


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(dict(json.loads(line)))
    return rows


def grouped(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    for value, items in sorted(groups.items()):
        out[value] = {
            "n": len(items),
            "score_mean": statistics.mean(float(item["score"]) for item in items),
            "acc_mean": statistics.mean(float(item["acc"]) for item in items),
            "marker_mean": statistics.mean(float(item["answer_marker_present"]) for item in items),
            "parse_ok_mean": statistics.mean(float(item["parse_ok"]) for item in items),
            "skip_phrase_mean": statistics.mean(float(item["skip_phrase_present"]) for item in items),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    examples = read_examples(args.data)
    rows = read_predictions(args.predictions)
    records = []
    for row in rows:
        problem_id = str(row["problem_id"])
        example = examples[problem_id]
        completion = str(row.get("completion", ""))
        result = verify_final_answer(completion, example.verifier)
        records.append(
            {
                "problem_id": problem_id,
                "family": example.family,
                "difficulty": example.difficulty,
                "completion": completion,
                "model": row.get("model", ""),
                "output_tokens": row.get("output_tokens", 0),
                **asdict(result),
                "acc": 1.0 if result.ok else 0.0,
                "answer_marker_present": 1.0 if result.answer_marker_present else 0.0,
                "parse_ok": 1.0 if result.parse_ok else 0.0,
                "skip_phrase_present": 1.0 if result.skip_phrase_present else 0.0,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "final_answer_judged.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "data": str(args.data),
        "predictions": str(args.predictions),
        "n": len(records),
        "score_mean": statistics.mean(float(item["score"]) for item in records) if records else 0.0,
        "acc_mean": statistics.mean(float(item["acc"]) for item in records) if records else 0.0,
        "marker_mean": statistics.mean(float(item["answer_marker_present"]) for item in records) if records else 0.0,
        "parse_ok_mean": statistics.mean(float(item["parse_ok"]) for item in records) if records else 0.0,
        "skip_phrase_mean": statistics.mean(float(item["skip_phrase_present"]) for item in records) if records else 0.0,
        "by_family": grouped(records, "family"),
        "by_difficulty": grouped(records, "difficulty"),
        "worst": [
            {
                "problem_id": item["problem_id"],
                "family": item["family"],
                "difficulty": item["difficulty"],
                "score": item["score"],
                "reason": item["reason"],
                "extracted_answer": item["extracted_answer"],
            }
            for item in sorted(records, key=lambda item: float(item["score"]))[:12]
        ],
        "best": [
            {
                "problem_id": item["problem_id"],
                "family": item["family"],
                "difficulty": item["difficulty"],
                "score": item["score"],
                "reason": item["reason"],
                "extracted_answer": item["extracted_answer"],
            }
            for item in sorted(records, key=lambda item: float(item["score"]), reverse=True)[:12]
        ],
    }
    (args.out_dir / "final_answer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
