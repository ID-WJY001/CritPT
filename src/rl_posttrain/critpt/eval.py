from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rl_posttrain.critpt.schema import CritPTExample
from rl_posttrain.critpt.verifier import verify_completion


def load_jsonl(path: Path) -> list[CritPTExample]:
    examples: list[CritPTExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(CritPTExample.from_dict(json.loads(line)))
    return examples


def evaluate_predictions(examples: Iterable[CritPTExample], predictions: dict[str, str]) -> dict:
    rows = []
    correct = 0
    total = 0
    for example in examples:
        completion = predictions.get(example.problem_id, "")
        result = verify_completion(completion, example.verifier)
        total += 1
        correct += int(result.ok)
        rows.append(
            {
                "problem_id": example.problem_id,
                "ok": result.ok,
                "score": result.score,
                "reason": result.reason,
                "extracted": result.extracted,
            }
        )
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "rows": rows,
    }

