#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from rl_posttrain.critpt_synth.v12_natural_physics import (
    NaturalPhysicsExample,
    generate_v12_natural_physics_examples,
    verify_v12_example,
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


def summarize(examples: list[NaturalPhysicsExample]) -> dict:
    return {
        "total": len(examples),
        "domain": dict(Counter(example.domain for example in examples)),
        "skill": dict(Counter(example.skill for example in examples)),
        "answer_type": dict(Counter(example.answer_type for example in examples)),
        "difficulty": dict(Counter(example.difficulty for example in examples)),
        "split": dict(Counter(example.split for example in examples)),
    }


def verify_examples(examples: list[NaturalPhysicsExample]) -> None:
    failures: list[str] = []
    for example in examples:
        ok, reasons = verify_v12_example(example)
        if not ok:
            failures.append(f"{example.example_id}: {'; '.join(reasons)}")
    if failures:
        preview = "\n".join(failures[:30])
        raise SystemExit(f"{len(failures)} V12 examples failed verification:\n{preview}")


def write_sample_markdown(path: Path, examples: list[NaturalPhysicsExample], per_split: int) -> None:
    selected: list[NaturalPhysicsExample] = []
    by_split: dict[str, list[NaturalPhysicsExample]] = {}
    for example in examples:
        by_split.setdefault(example.split, []).append(example)
    for split in ["train", "val", "test"]:
        selected.extend(by_split.get(split, [])[:per_split])

    lines = [
        "# V12 Natural Physics 数据抽样",
        "",
        "这批样本前台是自然语言物理/数学题；Python 不进入题面，只在后台 verifier/oracle 使用。",
        "每条都有 reference_solution、structured final_answer、deterministic verifier 和 anti-hack wrong answers。",
        "",
    ]
    for example in selected:
        lines.extend(
            [
                f"## `{example.example_id}`",
                "",
                f"- split: `{example.split}`",
                f"- domain: `{example.domain}`",
                f"- skill: `{example.skill}`",
                f"- difficulty: `{example.difficulty}`",
                f"- answer_type: `{example.answer_type}`",
                "",
                "### Question",
                "",
                example.question,
                "",
                "### Reference Solution",
                "",
                example.reference_solution,
                "",
                "### Final Answer",
                "",
                "```text",
                example.final_answer,
                "```",
                "",
                "### Verifier",
                "",
                "```json",
                json.dumps(example.verifier, ensure_ascii=False, indent=2),
                "```",
                "",
                "### Anti-Hack Wrong Answers",
                "",
                "```json",
                json.dumps(example.anti_hack_wrong_answers, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/synthetic_critpt/v12_natural_physics"))
    parser.add_argument("--name", default="synthetic_critpt_v12_natural_physics")
    parser.add_argument("--train-size", type=int, default=350)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--sample-per-split", type=int, default=8)
    args = parser.parse_args()

    train = generate_v12_natural_physics_examples(args.train_size, args.seed, "train")
    val = generate_v12_natural_physics_examples(args.val_size, args.seed + 1, "val")
    test = generate_v12_natural_physics_examples(args.test_size, args.seed + 2, "test")
    all_examples = train + val + test
    verify_examples(all_examples)

    raw_paths = {
        "train": args.out_dir / "train.jsonl",
        "val": args.out_dir / "val.jsonl",
        "test": args.out_dir / "test.jsonl",
    }
    write_jsonl(raw_paths["train"], (example.to_dict() for example in train))
    write_jsonl(raw_paths["val"], (example.to_dict() for example in val))
    write_jsonl(raw_paths["test"], (example.to_dict() for example in test))

    sample_path = args.out_dir / "samples.zh-CN.md"
    write_sample_markdown(sample_path, all_examples, args.sample_per_split)

    manifest = {
        "name": args.name,
        "profile": "v12_natural_physics",
        "seed": args.seed,
        "summary": summarize(all_examples),
        "raw_files": {split: str(path) for split, path in raw_paths.items()},
        "sample_markdown": str(sample_path),
        "sha256": {split: file_sha256(path) for split, path in raw_paths.items()},
        "leakage_policy": {
            "official_70_prompts_used_for_training": False,
            "official_challenge_ids_used_for_training": False,
            "front_prompt_contains_python": False,
            "note": "V12 uses programmatic natural-language physics/math generators and does not read official public-test prompts.",
        },
        "quality_gates": {
            "gold_final_answers_pass_verifier": True,
            "anti_hack_wrong_answers_fail_verifier": True,
            "python_template_visible_to_model": False,
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
