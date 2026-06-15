#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.critpt_synth.v13_official_style import (
    generate_v13_official_style_examples,
    verify_v13_example,
)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(examples: list[SyntheticCritPTExample]) -> dict:
    return {
        "total": len(examples),
        "family": dict(Counter(example.family for example in examples)),
        "difficulty": dict(Counter(example.difficulty for example in examples)),
        "split": dict(Counter(example.split for example in examples)),
        "domain": dict(Counter(str(example.metadata.get("domain", "unknown")) for example in examples)),
        "answer_type": dict(
            Counter(str(example.metadata.get("answer_type", "unknown")) for example in examples)
        ),
    }


def verify_targets(examples: list[SyntheticCritPTExample], workers: int) -> None:
    failures: list[tuple[str, str]] = []
    if workers <= 1:
        for example in examples:
            ok, reason = verify_v13_example(example)
            if not ok:
                failures.append((example.problem_id, reason))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for example, (ok, reason) in zip(
                examples, executor.map(verify_v13_example, examples, chunksize=16), strict=True
            ):
                if not ok:
                    failures.append((example.problem_id, reason))
    if failures:
        preview = "\n".join(f"{problem_id}: {reason}" for problem_id, reason in failures[:25])
        raise SystemExit(f"{len(failures)} V13 examples failed verification:\n{preview}")


def _by_split(examples: list[SyntheticCritPTExample]) -> dict[str, list[SyntheticCritPTExample]]:
    grouped: dict[str, list[SyntheticCritPTExample]] = {}
    for example in examples:
        grouped.setdefault(example.split, []).append(example)
    return grouped


def write_sample_markdown(path: Path, examples: list[SyntheticCritPTExample], per_split: int) -> None:
    selected: list[SyntheticCritPTExample] = []
    grouped = _by_split(examples)
    for split in ["train", "val", "test"]:
        selected.extend(grouped.get(split, [])[:per_split])

    lines = [
        "# V13 Official-Style 数据抽样",
        "",
        "这批数据模仿 CritPT 官方格式：`Problem setup`、`Main problem`、`Parsing template`、完整 `answer()` 函数。",
        "它不使用官方 70 题原题训练，只使用我们根据官方分布反推的合成模板。",
        "",
    ]
    for example in selected:
        lines.extend(
            [
                f"## `{example.problem_id}`",
                "",
                f"- split: `{example.split}`",
                f"- family: `{example.family}`",
                f"- difficulty: `{example.difficulty}`",
                f"- domain: `{example.metadata.get('domain')}`",
                f"- answer_type: `{example.metadata.get('answer_type')}`",
                "",
                "### Prompt",
                "",
                "```text",
                example.prompt,
                "```",
                "",
                "### Gold Assistant Output",
                "",
                example.assistant_code_block(),
                "",
                "### Verifier",
                "",
                "```json",
                json.dumps(example.verifier, ensure_ascii=False, indent=2),
                "```",
                "",
                "### Solution Trace",
                "",
                example.solution_trace,
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_gold_rollout_preview(
    markdown_path: Path,
    jsonl_path: Path,
    examples: list[SyntheticCritPTExample],
    count: int,
) -> None:
    selected = examples[:count]
    rows = []
    lines = [
        "# V13 Gold Rollout Preview",
        "",
        "这里不是模型真实 rollout，而是给你看的“理想 rollout 形态”：输入 prompt + gold `answer()` 输出。",
        "确认这批数据方向对了以后，再把同一批 prompt 扔给 Qwen3-8B 做真实 rollout/RL。",
        "",
    ]
    for idx, example in enumerate(selected, start=1):
        completion = example.assistant_code_block()
        rows.append(
            {
                "step": 0,
                "problem_id": example.problem_id,
                "input": example.prompt,
                "output": completion,
                "gts": example.target_code,
                "score": 1.0,
                "acc": 1.0,
                "reason": "gold_target_passes_verifier",
                "family": example.family,
                "difficulty": example.difficulty,
                "domain": example.metadata.get("domain"),
                "answer_type": example.metadata.get("answer_type"),
            }
        )
        lines.extend(
            [
                f"## {idx}. `{example.problem_id}`",
                "",
                f"- family: `{example.family}`",
                f"- domain: `{example.metadata.get('domain')}`",
                f"- answer_type: `{example.metadata.get('answer_type')}`",
                "",
                "### 模型输入",
                "",
                "```text",
                example.prompt,
                "```",
                "",
                "### 理想输出",
                "",
                completion,
                "",
            ]
        )
    write_jsonl(jsonl_path, rows)
    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/data/v13_official_style"))
    parser.add_argument("--name", default="synthetic_critpt_v13_official_style")
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--sample-per-split", type=int, default=4)
    parser.add_argument("--rollout-preview-count", type=int, default=12)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    train = generate_v13_official_style_examples(args.train_size, args.seed, "train")
    val = generate_v13_official_style_examples(args.val_size, args.seed + 1, "val")
    test = generate_v13_official_style_examples(args.test_size, args.seed + 2, "test")
    all_examples = train + val + test

    if not args.skip_verify:
        verify_targets(all_examples, args.workers)

    raw_paths = {
        "train": args.out_dir / "train.jsonl",
        "val": args.out_dir / "val.jsonl",
        "test": args.out_dir / "test.jsonl",
    }
    sft_paths = {
        "train": args.out_dir / "train_sft_messages.jsonl",
        "val": args.out_dir / "val_sft_messages.jsonl",
        "test": args.out_dir / "test_sft_messages.jsonl",
    }
    write_jsonl(raw_paths["train"], (example.to_dict() for example in train))
    write_jsonl(raw_paths["val"], (example.to_dict() for example in val))
    write_jsonl(raw_paths["test"], (example.to_dict() for example in test))
    write_jsonl(sft_paths["train"], (example.to_sft_row() for example in train))
    write_jsonl(sft_paths["val"], (example.to_sft_row() for example in val))
    write_jsonl(sft_paths["test"], (example.to_sft_row() for example in test))

    sample_path = args.out_dir / "samples.zh-CN.md"
    write_sample_markdown(sample_path, all_examples, args.sample_per_split)

    rollout_md = args.out_dir / "gold_rollout_preview.zh-CN.md"
    rollout_jsonl = args.out_dir / "gold_rollouts_preview.jsonl"
    write_gold_rollout_preview(rollout_md, rollout_jsonl, all_examples, args.rollout_preview_count)

    manifest = {
        "name": args.name,
        "profile": "v13_official_style",
        "seed": args.seed,
        "summary": summarize(all_examples),
        "raw_files": {split: str(path) for split, path in raw_paths.items()},
        "sft_files": {split: str(path) for split, path in sft_paths.items()},
        "sample_markdown": str(sample_path),
        "gold_rollout_preview_markdown": str(rollout_md),
        "gold_rollout_preview_jsonl": str(rollout_jsonl),
        "sha256": {
            **{f"raw_{split}": file_sha256(path) for split, path in raw_paths.items()},
            **{f"sft_{split}": file_sha256(path) for split, path in sft_paths.items()},
            "gold_rollouts_preview": file_sha256(rollout_jsonl),
        },
        "leakage_policy": {
            "official_70_prompts_used_for_training": False,
            "official_challenge_ids_used_for_training": False,
            "note": "V13 uses synthetic generators based on the official distribution analysis, not official prompt text.",
        },
        "quality_gates": {
            "gold_target_code_passes_code_verifier": not args.skip_verify,
            "prompt_contains_parsing_template": True,
            "target_is_complete_answer_function": True,
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
