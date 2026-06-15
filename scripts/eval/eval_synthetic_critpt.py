#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.critpt_synth.verifier import verify_code_completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    examples: dict[str, SyntheticCritPTExample] = {}
    with args.data.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                example = SyntheticCritPTExample.from_dict(json.loads(line))
                examples[example.problem_id] = example

    predictions: dict[str, str] = {}
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                predictions[str(row["problem_id"])] = str(row["completion"])

    details = []
    for problem_id, example in examples.items():
        completion = predictions.get(problem_id, "")
        result = verify_code_completion(completion, example.verifier)
        details.append(
            {
                "problem_id": problem_id,
                "ok": result.ok,
                "score": result.score,
                "reason": result.reason,
                "family": example.family,
                "difficulty": example.difficulty,
            }
        )
    summary = {
        "total": len(details),
        "answered": sum(1 for item in details if predictions.get(item["problem_id"])),
        "acc": sum(1 for item in details if item["ok"]) / max(1, len(details)),
        "avg_score": sum(float(item["score"]) for item in details) / max(1, len(details)),
        "details": details,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
