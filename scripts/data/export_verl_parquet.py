#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rl_posttrain.critpt.schema import CritPTExample


INSTRUCTION = (
    "请逐步推理。最终答案必须放在 <answer>...</answer> 标签中，"
    "不要在标签外再给另一个最终答案。"
)


def read_examples(path: Path) -> list[CritPTExample]:
    examples: list[CritPTExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(CritPTExample.from_dict(json.loads(line)))
    return examples


def to_verl_row(example: CritPTExample, index: int) -> dict:
    verifier = {
        "kind": example.verifier.kind,
        "expected": example.verifier.expected,
        "variables": example.verifier.variables,
        "numeric_tests": example.verifier.numeric_tests,
        "tolerance": example.verifier.tolerance,
    }
    return {
        "data_source": "critpt",
        "prompt": [{"role": "user", "content": f"{example.prompt}\n\n{INSTRUCTION}"}],
        "ability": "critpt",
        "reward_model": {
            "style": "rule",
            "ground_truth": example.answer,
        },
        "extra_info": {
            "split": example.split,
            "index": index,
            "problem_id": example.problem_id,
            "verifier": verifier,
            "metadata": example.metadata,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--val-out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1, help="repeat rows for infra smoke only")
    args = parser.parse_args()

    examples = read_examples(args.input)
    if not examples:
        raise SystemExit(f"no examples found in {args.input}")

    train_examples = [example for example in examples if example.split == "train"]
    val_examples = [example for example in examples if example.split != "train"]
    train_rows = []
    for repeat_idx in range(args.repeat):
        for index, example in enumerate(train_examples):
            row = to_verl_row(example, repeat_idx * len(train_examples) + index)
            row["extra_info"]["repeat_idx"] = repeat_idx
            train_rows.append(row)
    val_rows = [to_verl_row(example, index) for index, example in enumerate(val_examples)]
    if not val_rows:
        val_rows = train_rows[: min(8, len(train_rows))]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(args.train_out, index=False)
    pd.DataFrame(val_rows).to_parquet(args.val_out, index=False)
    print(f"wrote {len(train_rows)} train rows -> {args.train_out}")
    print(f"wrote {len(val_rows)} val rows -> {args.val_out}")


if __name__ == "__main__":
    main()
