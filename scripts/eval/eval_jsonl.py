#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_posttrain.critpt.eval import evaluate_predictions, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()

    examples = load_jsonl(args.data)
    if args.predictions:
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    else:
        predictions = {example.problem_id: example.answer for example in examples}
    report = evaluate_predictions(examples, predictions)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

