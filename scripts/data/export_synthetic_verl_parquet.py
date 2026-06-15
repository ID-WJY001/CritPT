#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample


def read_examples(path: Path) -> list[SyntheticCritPTExample]:
    examples: list[SyntheticCritPTExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(SyntheticCritPTExample.from_dict(json.loads(line)))
    return examples


def to_verl_row(example: SyntheticCritPTExample, index: int) -> dict:
    return {
        "data_source": "critpt_synth_code",
        "prompt": [{"role": "user", "content": example.prompt}],
        "ability": "critpt_code",
        "reward_model": {
            "style": "rule",
            "ground_truth": example.target_code,
        },
        "extra_info": {
            "split": example.split,
            "index": index,
            "problem_id": example.problem_id,
            "family": example.family,
            "difficulty": example.difficulty,
            "code_verifier": json.dumps(example.verifier, ensure_ascii=False),
            "metadata": json.dumps(example.metadata, ensure_ascii=False),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--val-out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        help="Optional deterministic shuffle for train rows after repeat expansion.",
    )
    args = parser.parse_args()

    train_examples = read_examples(args.train_jsonl)
    val_examples = read_examples(args.val_jsonl)
    train_rows = []
    for repeat_idx in range(args.repeat):
        for idx, example in enumerate(train_examples):
            row = to_verl_row(example, repeat_idx * len(train_examples) + idx)
            row["extra_info"]["repeat_idx"] = repeat_idx
            train_rows.append(row)
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(train_rows)
    val_rows = [to_verl_row(example, idx) for idx, example in enumerate(val_examples)]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(args.train_out, index=False)
    pd.DataFrame(val_rows).to_parquet(args.val_out, index=False)
    print(f"wrote {len(train_rows)} train rows -> {args.train_out}")
    print(f"wrote {len(val_rows)} val rows -> {args.val_out}")


if __name__ == "__main__":
    main()
