#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from rl_posttrain.critpt_synth.e2_llm_templates import (
    E2Template,
    generate_template_bank,
    materialize_examples,
    verify_e2_template_example,
)
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.model_judge.openai_compatible import JudgeSettings


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_template_bank(path: Path) -> list[E2Template]:
    templates: list[E2Template] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                templates.append(E2Template.from_dict(json.loads(line)))
    return templates


def write_template_bank(path: Path, templates: list[E2Template]) -> None:
    write_jsonl(path, (template.to_dict() for template in templates))


def summarize(examples: list[SyntheticCritPTExample], templates: list[E2Template]) -> dict:
    prompt_lengths = [len(example.prompt) for example in examples]
    return {
        "total": len(examples),
        "templates": len(templates),
        "template_families": dict(Counter(template.family for template in templates)),
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
        ok, reason = verify_e2_template_example(example)
        if not ok:
            failures.append((example.problem_id, reason))
    if failures:
        preview = "\n".join(f"{problem_id}: {reason}" for problem_id, reason in failures[:10])
        raise SystemExit(f"{len(failures)} E2 template examples failed verification:\n{preview}")


def write_samples(path: Path, examples: list[SyntheticCritPTExample]) -> None:
    lines = [
        "# E2 LLM-Template 数据抽样",
        "",
        "E2-template 的目标是让 LLM 先生成题型模板、solver 和参数采样器；程序再批量采样参数、计算标准答案、执行 verifier。",
        "",
    ]
    for example in examples[:16]:
        lines.extend(
            [
                f"## `{example.problem_id}`",
                "",
                f"- split: `{example.split}`",
                f"- family: `{example.family}`",
                f"- template_id: `{example.metadata.get('template_id')}`",
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


def e2_spec_settings_from_env(settings: JudgeSettings) -> JudgeSettings:
    return replace(
        settings,
        model=os.environ.get("E2_SPEC_MODEL", settings.model).strip() or settings.model,
        timeout_s=float(os.environ.get("E2_SPEC_TIMEOUT_S", str(settings.timeout_s))),
        max_tokens=int(os.environ.get("E2_SPEC_MAX_TOKENS", "5000")),
        temperature=float(os.environ.get("E2_SPEC_TEMPERATURE", "0.2")),
        max_retries=int(os.environ.get("E2_SPEC_MAX_RETRIES", str(settings.max_retries))),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/data/e2_llm_template_full"))
    parser.add_argument("--train-size", type=int, default=1400)
    parser.add_argument("--val-size", type=int, default=140)
    parser.add_argument("--test-size", type=int, default=140)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--templates-per-family", type=int, default=2)
    parser.add_argument("--template-workers", type=int, default=4)
    parser.add_argument("--profile-limit", type=int, default=0, help="0 means use all E2 profiles")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--force-template-rebuild", action="store_true")
    parser.add_argument("--llm-cache-path", type=Path, default=None)
    args = parser.parse_args()

    settings = None if args.mock else e2_spec_settings_from_env(JudgeSettings.from_env())
    if not args.mock and not settings.api_key:
        raise SystemExit("OPENAI_API_KEY is required unless --mock is used")
    cache_path = str(args.llm_cache_path or (args.out_dir / "llm_template_cache.sqlite3"))
    template_bank_path = args.out_dir / "template_bank.jsonl"

    print(
        json.dumps(
            {
                "event": "e2_template_generation_start",
                "out_dir": str(args.out_dir),
                "train_size": args.train_size,
                "val_size": args.val_size,
                "test_size": args.test_size,
                "templates_per_family": args.templates_per_family,
                "template_workers": args.template_workers,
                "profile_limit": args.profile_limit,
                "model": settings.model if settings else "",
                "max_tokens": settings.max_tokens if settings else 0,
                "timeout_s": settings.timeout_s if settings else 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if template_bank_path.exists() and not args.force_template_rebuild:
        print(f"loading template bank: {template_bank_path}", flush=True)
        templates = read_template_bank(template_bank_path)
    else:
        print("building template bank...", flush=True)
        templates = generate_template_bank(
            templates_per_family=args.templates_per_family,
            seed=args.seed,
            settings=settings or JudgeSettings.from_env(),
            cache_path=cache_path,
            workers=args.template_workers,
            mock=args.mock,
            profile_limit=args.profile_limit,
        )
        write_template_bank(template_bank_path, templates)

    print("materializing train split...", flush=True)
    train = materialize_examples(templates=templates, size=args.train_size, split="train", seed=args.seed)
    print("materializing val split...", flush=True)
    val = materialize_examples(templates=templates, size=args.val_size, split="val", seed=args.seed + 1)
    print("materializing test split...", flush=True)
    test = materialize_examples(templates=templates, size=args.test_size, split="test", seed=args.seed + 2)
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
        "name": "synthetic_critpt_e2_llm_template_full",
        "profile": "e2_llm_template_bank_solver_sampler",
        "mock": args.mock,
        "llm_model": settings.model if settings else "",
        "summary": summarize(examples, templates),
        "raw_files": {split: str(path) for split, path in raw_paths.items()},
        "sft_files": {split: str(path) for split, path in sft_paths.items()},
        "template_bank": str(template_bank_path),
        "sample_markdown": str(sample_path),
        "quality_gates": {
            "llm_generates_problem_template": True,
            "llm_generates_solver": True,
            "llm_generates_parameter_sampler": True,
            "program_expands_parameters": True,
            "program_computes_expected_answers_from_llm_solver": True,
            "program_verifies_every_target_code": True,
            "long_prompt_target": "2500-4500 chars before parser template",
            "official_prompts_used_for_training": False,
        },
        "generation": {
            "seed": args.seed,
            "templates_per_family": args.templates_per_family,
            "template_workers": args.template_workers,
            "profile_limit": args.profile_limit,
        },
        "env": {
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", ""),
            "JUDGE_MODEL": os.environ.get("JUDGE_MODEL", ""),
            "E2_SPEC_MODEL": os.environ.get("E2_SPEC_MODEL", ""),
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
