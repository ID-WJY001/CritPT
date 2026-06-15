#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from rl_posttrain.critpt_synth.e2_llm_specs import generate_e2_examples, verify_e2_example
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.model_judge.openai_compatible import JudgeSettings


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(examples: list[SyntheticCritPTExample]) -> dict:
    prompt_lengths = [len(example.prompt) for example in examples]
    return {
        "total": len(examples),
        "split": dict(Counter(example.split for example in examples)),
        "family": dict(Counter(example.family for example in examples)),
        "difficulty": dict(Counter(example.difficulty for example in examples)),
        "answer_type": dict(Counter(str(example.metadata.get("answer_type", "")) for example in examples)),
        "prompt_chars": {
            "min": min(prompt_lengths) if prompt_lengths else 0,
            "mean": round(sum(prompt_lengths) / len(prompt_lengths), 1) if prompt_lengths else 0,
            "max": max(prompt_lengths) if prompt_lengths else 0,
        },
    }


def verify_targets(examples: list[SyntheticCritPTExample]) -> None:
    failures: list[tuple[str, str]] = []
    for example in examples:
        ok, reason = verify_e2_example(example)
        if not ok:
            failures.append((example.problem_id, reason))
    if failures:
        preview = "\n".join(f"{problem_id}: {reason}" for problem_id, reason in failures[:10])
        raise SystemExit(f"{len(failures)} E2 examples failed verification:\n{preview}")


def e2_spec_settings_from_env(settings: JudgeSettings) -> JudgeSettings:
    return replace(
        settings,
        model=os.environ.get("E2_SPEC_MODEL", settings.model).strip() or settings.model,
        timeout_s=float(os.environ.get("E2_SPEC_TIMEOUT_S", str(settings.timeout_s))),
        max_tokens=int(os.environ.get("E2_SPEC_MAX_TOKENS", "4096")),
        temperature=float(os.environ.get("E2_SPEC_TEMPERATURE", "0.2")),
        max_retries=int(os.environ.get("E2_SPEC_MAX_RETRIES", str(settings.max_retries))),
    )


def write_samples(path: Path, examples: list[SyntheticCritPTExample]) -> None:
    lines = [
        "# E2 LLM-Spec 数据抽样",
        "",
        "E2 的目标是让 LLM 不只包装背景，还生成可执行 solver 规格；程序再执行 verifier，过不了就不入库。",
        "这份是 prototype，小规模检查长 prompt、复杂度和 solver 可验证性。",
        "",
    ]
    for example in examples[:12]:
        lines.extend(
            [
                f"## `{example.problem_id}`",
                "",
                f"- split: `{example.split}`",
                f"- family: `{example.family}`",
                f"- prompt_chars: `{example.metadata.get('prompt_chars')}`",
                f"- answer_type: `{example.metadata.get('answer_type')}`",
                f"- complexity: {example.metadata.get('complexity_notes')}",
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
                "### Trace",
                "",
                example.solution_trace,
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/data/e2_llm_specs_demo"))
    parser.add_argument("--train-size", type=int, default=3)
    parser.add_argument("--val-size", type=int, default=1)
    parser.add_argument("--test-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--llm-cache-path", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts-per-example", type=int, default=10)
    args = parser.parse_args()

    settings = None if args.mock else e2_spec_settings_from_env(JudgeSettings.from_env())
    cache_path = str(args.llm_cache_path or (args.out_dir / "llm_spec_cache.sqlite3"))
    if not args.mock and not settings.api_key:
        raise SystemExit("OPENAI_API_KEY is required unless --mock is used")

    print(
        json.dumps(
            {
                "event": "e2_generation_start",
                "out_dir": str(args.out_dir),
                "train_size": args.train_size,
                "val_size": args.val_size,
                "test_size": args.test_size,
                "workers": args.workers,
                "max_attempts_per_example": args.max_attempts_per_example,
                "model": settings.model if settings else "",
                "max_tokens": settings.max_tokens if settings else 0,
                "timeout_s": settings.timeout_s if settings else 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print("building train split...", flush=True)
    train = generate_e2_examples(
        size=args.train_size,
        seed=args.seed,
        split="train",
        settings=settings,
        cache_path=cache_path,
        mock=args.mock,
        workers=args.workers,
        max_attempts_per_example=args.max_attempts_per_example,
    )
    print("building val split...", flush=True)
    val = generate_e2_examples(
        size=args.val_size,
        seed=args.seed + 1,
        split="val",
        settings=settings,
        cache_path=cache_path,
        mock=args.mock,
        workers=args.workers,
        max_attempts_per_example=args.max_attempts_per_example,
    )
    print("building test split...", flush=True)
    test = generate_e2_examples(
        size=args.test_size,
        seed=args.seed + 2,
        split="test",
        settings=settings,
        cache_path=cache_path,
        mock=args.mock,
        workers=args.workers,
        max_attempts_per_example=args.max_attempts_per_example,
    )
    examples = train + val + test
    print("verifying final targets...", flush=True)
    verify_targets(examples)

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
    write_samples(sample_path, examples)
    manifest = {
        "name": "synthetic_critpt_e2_llm_specs_demo",
        "profile": "e2_llm_spec_solver",
        "mock": args.mock,
        "llm_model": settings.model if settings else "",
        "summary": summarize(examples),
        "raw_files": {split: str(path) for split, path in raw_paths.items()},
        "sft_files": {split: str(path) for split, path in sft_paths.items()},
        "sample_markdown": str(sample_path),
        "quality_gates": {
            "llm_generates_problem_and_solver": True,
            "llm_generates_validation_cases": True,
            "program_verifies_target_code": True,
            "program_verifies_llm_solver_on_sample_and_validation_params": True,
            "long_prompt_target": "2500-4500 chars before template for LLM mode",
            "official_prompts_used_for_training": False,
        },
        "generation": {
            "workers": args.workers,
            "max_attempts_per_example": args.max_attempts_per_example,
            "seed": args.seed,
        },
        "env": {
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", ""),
            "JUDGE_MODEL": os.environ.get("JUDGE_MODEL", ""),
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
