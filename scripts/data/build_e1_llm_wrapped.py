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

from rl_posttrain.critpt_synth.e1_llm_wrapped import (
    e1_type_names,
    generate_e1_examples,
    verify_e1_example,
)
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.model_judge.openai_compatible import JudgeSettings


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
        "split": dict(Counter(example.split for example in examples)),
        "e1_type": dict(Counter(str(example.metadata.get("e1_type", example.family)) for example in examples)),
        "difficulty": dict(Counter(example.difficulty for example in examples)),
        "domain_shell": dict(Counter(str(example.metadata.get("domain_shell", "")) for example in examples)),
        "answer_type": dict(Counter(str(example.metadata.get("answer_type", "")) for example in examples)),
        "llm_background_wrapped": dict(
            Counter(str(example.metadata.get("llm_background_wrapped", False)) for example in examples)
        ),
    }


def verify_targets(examples: list[SyntheticCritPTExample], workers: int) -> None:
    failures: list[tuple[str, str]] = []
    if workers <= 1:
        for example in examples:
            ok, reason = verify_e1_example(example)
            if not ok:
                failures.append((example.problem_id, reason))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for example, (ok, reason) in zip(
                examples,
                executor.map(verify_e1_example, examples, chunksize=16),
                strict=True,
            ):
                if not ok:
                    failures.append((example.problem_id, reason))
    if failures:
        preview = "\n".join(f"{problem_id}: {reason}" for problem_id, reason in failures[:25])
        raise SystemExit(f"{len(failures)} E1 examples failed verification:\n{preview}")


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
        "# E1 LLM-Wrapped 数据抽样",
        "",
        "E1 从头开始：程序先造可验证题芯和标准答案，LLM 只负责包装背景。",
        "如果 `llm_background_wrapped=false`，说明这条使用的是程序模板背景。",
        "",
    ]
    for example in selected:
        lines.extend(
            [
                f"## `{example.problem_id}`",
                "",
                f"- split: `{example.split}`",
                f"- e1_type: `{example.metadata.get('e1_type')}`",
                f"- difficulty: `{example.difficulty}`",
                f"- domain_shell: `{example.metadata.get('domain_shell')}`",
                f"- answer_type: `{example.metadata.get('answer_type')}`",
                f"- llm_background_wrapped: `{example.metadata.get('llm_background_wrapped')}`",
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
                "### Reference Trace",
                "",
                example.solution_trace,
                "",
                "### Core Facts",
                "",
                "```json",
                str(example.metadata.get("core_facts", "{}")),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/data/e1_llm_wrapped"))
    parser.add_argument("--name", default="synthetic_critpt_e1_llm_wrapped")
    parser.add_argument("--train-size", type=int, default=1400)
    parser.add_argument("--val-size", type=int, default=140)
    parser.add_argument("--test-size", type=int, default=140)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--wrap-mode", choices=["template", "llm"], default="template")
    parser.add_argument(
        "--llm-limit",
        type=int,
        default=None,
        help="when --wrap-mode llm is used, wrap only the first N examples per split; default wraps all",
    )
    parser.add_argument("--llm-workers", type=int, default=1)
    parser.add_argument("--llm-cache-path", type=Path, default=None)
    parser.add_argument("--sample-per-split", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    settings = JudgeSettings.from_env() if args.wrap_mode == "llm" else None
    cache_path = str(args.llm_cache_path or (args.out_dir / "llm_background_cache.sqlite3"))

    train = generate_e1_examples(
        args.train_size,
        args.seed,
        "train",
        wrap_mode=args.wrap_mode,
        llm_limit=args.llm_limit,
        llm_settings=settings,
        llm_cache_path=cache_path,
        llm_workers=args.llm_workers,
    )
    val = generate_e1_examples(
        args.val_size,
        args.seed + 1,
        "val",
        wrap_mode=args.wrap_mode,
        llm_limit=args.llm_limit,
        llm_settings=settings,
        llm_cache_path=cache_path,
        llm_workers=args.llm_workers,
    )
    test = generate_e1_examples(
        args.test_size,
        args.seed + 2,
        "test",
        wrap_mode=args.wrap_mode,
        llm_limit=args.llm_limit,
        llm_settings=settings,
        llm_cache_path=cache_path,
        llm_workers=args.llm_workers,
    )
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

    manifest = {
        "name": args.name,
        "profile": "e1_llm_wrapped",
        "seed": args.seed,
        "wrap_mode": args.wrap_mode,
        "llm_limit": args.llm_limit,
        "llm_workers": args.llm_workers,
        "llm_model": settings.model if settings else "",
        "e1_types": e1_type_names(),
        "summary": summarize(all_examples),
        "raw_files": {split: str(path) for split, path in raw_paths.items()},
        "sft_files": {split: str(path) for split, path in sft_paths.items()},
        "sample_markdown": str(sample_path),
        "sha256": {
            **{f"raw_{split}": file_sha256(path) for split, path in raw_paths.items()},
            **{f"sft_{split}": file_sha256(path) for split, path in sft_paths.items()},
        },
        "leakage_policy": {
            "official_prompts_used_for_training": False,
            "official_challenge_ids_used_for_training": False,
            "note": "E1 generates programmatic cores and optionally asks an LLM to rewrite only background/problem prose.",
        },
        "quality_gates": {
            "gold_target_code_passes_code_verifier": not args.skip_verify,
            "prompt_contains_parsing_template": True,
            "target_is_complete_answer_function": True,
            "llm_may_wrap_background_only": True,
            "answer_generated_by_program": True,
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
