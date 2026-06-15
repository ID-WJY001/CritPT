#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from rl_posttrain.critpt_synth.e2_llm_specs import generate_e2_examples, verify_e2_example
from rl_posttrain.critpt_synth.e4_official_style import reference_output_for_code
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.model_judge.openai_compatible import JudgeSettings


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


def summarize(examples: list[SyntheticCritPTExample]) -> dict[str, Any]:
    lengths = [len(example.prompt) for example in examples]
    return {
        "total": len(examples),
        "split": dict(Counter(example.split for example in examples)),
        "family": dict(Counter(example.family for example in examples)),
        "difficulty": dict(Counter(example.difficulty for example in examples)),
        "domain_shell": dict(
            Counter(str(example.metadata.get("domain_shell", "unknown")) for example in examples)
        ),
        "answer_type": dict(
            Counter(str(example.metadata.get("answer_type", "unknown")) for example in examples)
        ),
        "prompt_chars": {
            "min": min(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
    }


def e6_spec_settings_from_env(settings: JudgeSettings) -> JudgeSettings:
    return replace(
        settings,
        model=os.environ.get("E6_SPEC_MODEL", settings.model).strip() or settings.model,
        timeout_s=float(os.environ.get("E6_SPEC_TIMEOUT_S", str(settings.timeout_s))),
        max_tokens=int(os.environ.get("E6_SPEC_MAX_TOKENS", "4096")),
        temperature=float(os.environ.get("E6_SPEC_TEMPERATURE", "0.25")),
        max_retries=int(os.environ.get("E6_SPEC_MAX_RETRIES", str(settings.max_retries))),
    )


def verify_targets(examples: list[SyntheticCritPTExample]) -> None:
    failures: list[tuple[str, str]] = []
    for example in examples:
        ok, reason = verify_e2_example(example)
        if not ok:
            failures.append((example.problem_id, reason))
    if failures:
        preview = "\n".join(f"{problem_id}: {reason}" for problem_id, reason in failures[:25])
        raise SystemExit(f"{len(failures)} E6 teacher examples failed verification:\n{preview}")


def stable_reference_output(example: SyntheticCritPTExample) -> str:
    return reference_output_for_code(example.target_code, [])


def adapt_e6_example(example: SyntheticCritPTExample, index: int) -> SyntheticCritPTExample:
    reference_output = stable_reference_output(example)
    digest = hashlib.sha256(
        json.dumps(
            {
                "prompt": example.prompt,
                "target": example.target_code,
                "reference_output": reference_output,
                "index": index,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    metadata = {
        **example.metadata,
        "generator_profile": "e6_teacher_spec_strict_judge",
        "e6_source": "llm_teacher_problem_and_solver",
        "e6_reference_output": reference_output,
        "reference_output": reference_output,
        "official_overlap": "none",
        "uses_official_prompt": False,
        "reward_uses_llm_judge_only": True,
        "program_verifies_teacher_solver_only": True,
        "e6_design_note": "LLM creates long problem plus solver; program validates the teacher target, but RL reward is LLM judge.",
    }
    return replace(
        example,
        problem_id=f"{example.split}_e6_teacher_{index:05d}_{digest}",
        family=f"e6_{example.family}",
        metadata=metadata,
    )


def reference_answer(example: SyntheticCritPTExample) -> str:
    return (
        "Trusted executable answer code:\n"
        "```python\n"
        f"{example.target_code.strip()}\n"
        "```"
    )


def reference_rubric(example: SyntheticCritPTExample) -> str:
    answer_type = str(example.metadata.get("answer_type", "unknown"))
    family = example.family
    return " ".join(
        [
            "Use LLM-as-a-judge only. Grade the candidate answer() against the trusted reference output and code.",
            "The official failure mode we are avoiding is a correct answer() shell with guessed constants.",
            "Do not reward docstrings, variable names, imports, or plausible physics prose unless the returned object is right.",
            "Compact literal answers are valid only when they exactly match the trusted final output.",
            "Wrong numbers, wrong labels, wrong ordering, wrong sets, wrong tie-breaks, and wrong rounding should be capped very low.",
            f"The expected answer type is {answer_type}.",
            f"The synthetic teacher family is {family}.",
        ]
    )


def to_verl_row(example: SyntheticCritPTExample, index: int) -> dict[str, Any]:
    ref_answer = reference_answer(example)
    reference_output = str(example.metadata["reference_output"])
    return {
        "data_source": "critpt_e6_teacher_specs_strict_judge",
        "prompt": [{"role": "user", "content": example.prompt}],
        "ability": "critpt_problem_solving",
        "reward_model": {
            "style": "llm_final_answer_code_judge",
            "ground_truth": ref_answer,
        },
        "extra_info": {
            "split": example.split,
            "index": index,
            "problem_id": example.problem_id,
            "family": example.family,
            "difficulty": example.difficulty,
            "prompt_text": example.prompt,
            "reference_answer": ref_answer,
            "reference_output": reference_output,
            "reference_trace": example.solution_trace,
            "reference_cot_answer": example.assistant_code_block(),
            "reference_answer_type": str(example.metadata.get("answer_type", "")),
            "reference_family": example.family,
            "cot_policy": "code_completion_teacher_solver_no_think_strict_no_guess",
            "rubric": reference_rubric(example),
            "metadata": json.dumps(example.metadata, ensure_ascii=False, sort_keys=True),
        },
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_sample_markdown(path: Path, examples: list[SyntheticCritPTExample], per_family: int) -> None:
    selected: list[SyntheticCritPTExample] = []
    counts: Counter[str] = Counter()
    for example in examples:
        if counts[example.family] >= per_family:
            continue
        selected.append(example)
        counts[example.family] += 1

    lines = [
        "# E6 Teacher-Spec Data Samples",
        "",
        "E6 uses an LLM teacher to create both a long official-style problem and a deterministic solver.",
        "The program validates the teacher solver and exports trusted reference outputs.",
        "Training reward remains LLM-as-a-judge; no local correctness verifier is used for model rollouts.",
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
                f"- domain_shell: `{example.metadata.get('domain_shell')}`",
                f"- answer_type: `{example.metadata.get('answer_type')}`",
                f"- prompt_chars: `{len(example.prompt)}`",
                f"- reference_output: `{example.metadata.get('reference_output')}`",
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


def write_gold_rollout_preview(
    markdown_path: Path,
    jsonl_path: Path,
    examples: list[SyntheticCritPTExample],
    count: int,
) -> None:
    selected = examples[:count]
    rows = []
    lines = [
        "# E6 Gold Rollout Preview",
        "",
        "This preview is the teacher target, not a model rollout. It shows the answer shape RL should learn.",
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
                "reference_output": example.metadata.get("reference_output"),
                "score": 1.0,
                "acc": 1.0,
                "family": example.family,
                "difficulty": example.difficulty,
                "domain_shell": example.metadata.get("domain_shell"),
                "answer_type": example.metadata.get("answer_type"),
            }
        )
        lines.extend(
            [
                f"## {idx}. `{example.problem_id}`",
                "",
                f"- family: `{example.family}`",
                f"- reference_output: `{example.metadata.get('reference_output')}`",
                "",
                "### Model Input",
                "",
                "```text",
                example.prompt,
                "```",
                "",
                "### Ideal Output",
                "",
                completion,
                "",
            ]
        )
    write_jsonl(jsonl_path, rows)
    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_split(
    *,
    size: int,
    seed: int,
    split: str,
    settings: JudgeSettings | None,
    cache_path: str,
    mock: bool,
    workers: int,
    max_attempts_per_example: int,
    start_index: int,
) -> list[SyntheticCritPTExample]:
    raw = generate_e2_examples(
        size=size,
        seed=seed,
        split=split,
        settings=settings,
        cache_path=cache_path,
        mock=mock,
        workers=workers,
        max_attempts_per_example=max_attempts_per_example,
    )
    return [adapt_e6_example(example, start_index + idx) for idx, example in enumerate(raw)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/data/e6_teacher_specs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/e6_teacher_specs"))
    parser.add_argument("--dataset-prefix", default="critpt_e6_teacher_specs")
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--val-size", type=int, default=120)
    parser.add_argument("--test-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--llm-cache-path", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts-per-example", type=int, default=12)
    parser.add_argument("--sample-per-family", type=int, default=1)
    parser.add_argument("--rollout-preview-count", type=int, default=12)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    settings = None if args.mock else e6_spec_settings_from_env(JudgeSettings.from_env())
    cache_path = str(args.llm_cache_path or (args.out_dir / "llm_teacher_cache.sqlite3"))
    if not args.mock and (settings is None or not settings.api_key):
        raise SystemExit("OPENAI_API_KEY is required unless --mock is used")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "e6_teacher_generation_start",
                "out_dir": str(args.out_dir),
                "data_dir": str(args.data_dir),
                "dataset_prefix": args.dataset_prefix,
                "train_size": args.train_size,
                "val_size": args.val_size,
                "test_size": args.test_size,
                "workers": args.workers,
                "max_attempts_per_example": args.max_attempts_per_example,
                "mock": args.mock,
                "model": settings.model if settings else "",
                "max_tokens": settings.max_tokens if settings else 0,
                "timeout_s": settings.timeout_s if settings else 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    train = build_split(
        size=args.train_size,
        seed=args.seed,
        split="train",
        settings=settings,
        cache_path=cache_path,
        mock=args.mock,
        workers=args.workers,
        max_attempts_per_example=args.max_attempts_per_example,
        start_index=0,
    )
    val = build_split(
        size=args.val_size,
        seed=args.seed + 1,
        split="val",
        settings=settings,
        cache_path=cache_path,
        mock=args.mock,
        workers=args.workers,
        max_attempts_per_example=args.max_attempts_per_example,
        start_index=len(train),
    )
    test = build_split(
        size=args.test_size,
        seed=args.seed + 2,
        split="test",
        settings=settings,
        cache_path=cache_path,
        mock=args.mock,
        workers=args.workers,
        max_attempts_per_example=args.max_attempts_per_example,
        start_index=len(train) + len(val),
    )
    all_examples = train + val + test

    if not args.skip_verify:
        verify_targets(all_examples)

    raw_paths = {
        "train": args.out_dir / "train.jsonl",
        "val": args.out_dir / "val.jsonl",
        "test": args.out_dir / "test.jsonl",
    }
    write_jsonl(raw_paths["train"], (example.to_dict() for example in train))
    write_jsonl(raw_paths["val"], (example.to_dict() for example in val))
    write_jsonl(raw_paths["test"], (example.to_dict() for example in test))

    train_rows = [to_verl_row(example, idx) for idx, example in enumerate(train)]
    val_rows = [to_verl_row(example, idx) for idx, example in enumerate(val)]
    test_rows = [to_verl_row(example, idx) for idx, example in enumerate(test)]
    parquet_paths = {
        "train": args.data_dir / f"{args.dataset_prefix}_train.parquet",
        "val": args.data_dir / f"{args.dataset_prefix}_val.parquet",
        "test": args.data_dir / f"{args.dataset_prefix}_test.parquet",
    }
    write_parquet(parquet_paths["train"], train_rows)
    write_parquet(parquet_paths["val"], val_rows)
    write_parquet(parquet_paths["test"], test_rows)

    sample_path = args.out_dir / "samples.md"
    rollout_md = args.out_dir / "gold_rollout_preview.md"
    rollout_jsonl = args.out_dir / "gold_rollouts_preview.jsonl"
    write_sample_markdown(sample_path, all_examples, args.sample_per_family)
    write_gold_rollout_preview(rollout_md, rollout_jsonl, all_examples, args.rollout_preview_count)

    manifest = {
        "name": "synthetic_critpt_e6_teacher_specs_strict_judge",
        "profile": "e6_teacher_spec_strict_judge",
        "seed": args.seed,
        "mock": args.mock,
        "llm_model": settings.model if settings else "",
        "summary": summarize(all_examples),
        "raw_files": {split: str(path) for split, path in raw_paths.items()},
        "parquet_files": {split: str(path) for split, path in parquet_paths.items()},
        "sample_markdown": str(sample_path),
        "gold_rollout_preview_markdown": str(rollout_md),
        "gold_rollout_preview_jsonl": str(rollout_jsonl),
        "sha256": {
            **{f"raw_{split}": file_sha256(path) for split, path in raw_paths.items()},
            **{f"parquet_{split}": file_sha256(path) for split, path in parquet_paths.items()},
            "gold_rollouts_preview": file_sha256(rollout_jsonl),
        },
        "leakage_policy": {
            "official_70_prompts_used_for_training": False,
            "official_challenge_ids_used_for_training": False,
            "official_answers_used_for_training": False,
            "note": "E6 uses teacher-generated long official-style analogues, not official prompt text.",
        },
        "quality_gates": {
            "llm_generates_problem_and_solver": not args.mock,
            "teacher_solver_validated_by_program": not args.skip_verify,
            "reference_output_present": True,
            "reward_uses_llm_judge_only": True,
            "prompt_contains_parsing_template": True,
            "target_is_complete_answer_code": True,
            "anti_placeholder_rubric_present": True,
        },
        "generation": {
            "workers": args.workers,
            "max_attempts_per_example": args.max_attempts_per_example,
            "cache_path": cache_path,
        },
        "env": {
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", ""),
            "JUDGE_MODEL": os.environ.get("JUDGE_MODEL", ""),
            "E6_SPEC_MODEL": os.environ.get("E6_SPEC_MODEL", ""),
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
