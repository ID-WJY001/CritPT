#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from rl_posttrain.critpt_synth.e4_official_style import (
    generate_e4_official_style_examples,
    verify_e4_example,
)
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample


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
        "domain": dict(Counter(str(example.metadata.get("domain", "unknown")) for example in examples)),
        "answer_type": dict(
            Counter(str(example.metadata.get("answer_type", "unknown")) for example in examples)
        ),
        "prompt_chars": {
            "min": min(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
    }


def verify_targets(examples: list[SyntheticCritPTExample]) -> None:
    failures: list[tuple[str, str]] = []
    for example in examples:
        ok, reason = verify_e4_example(example)
        if not ok:
            failures.append((example.problem_id, reason))
    if failures:
        preview = "\n".join(f"{problem_id}: {reason}" for problem_id, reason in failures[:25])
        raise SystemExit(f"{len(failures)} E4 examples failed verification:\n{preview}")


def reference_answer(example: SyntheticCritPTExample) -> str:
    return (
        "Reference executable answer code:\n"
        "```python\n"
        f"{example.target_code.strip()}\n"
        "```"
    )


def reference_rubric(example: SyntheticCritPTExample) -> str:
    answer_type = str(example.metadata.get("answer_type", ""))
    family = example.family
    parts = [
        "Use LLM-as-a-judge only. Grade the candidate answer() against the trusted reference output and code.",
        "The final returned object is primary: exact values, symbolic expression, selected labels, set membership, ordering, and return type matter.",
        "Do not reward a function merely because its docstring or shape matches the template.",
        "Compact literal answers are allowed when they match the trusted final output.",
    ]
    if "set_filter" in family or "tuple_set" in family:
        parts.append("For set/filter tasks, penalize supersets, missing accepted items, and extra rejected items.")
    if "choice" in family:
        parts.append("For multiple-choice fields, the selected letter must match exactly.")
    if "sympy" in answer_type:
        parts.append("For SymPy outputs, algebraic equivalence is acceptable but wrong variables or missing factors are not.")
    if answer_type:
        parts.append(f"The expected answer type is {answer_type}.")
    return " ".join(parts)


def to_verl_row(example: SyntheticCritPTExample, index: int) -> dict[str, Any]:
    ref_answer = reference_answer(example)
    reference_output = str(example.metadata["reference_output"])
    return {
        "data_source": "critpt_e4_official_style_final_answer_judge",
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
            "cot_policy": "code_completion_official_shell_no_think",
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
        "# E4 Official-Style Data Samples",
        "",
        "E4 is synthetic data shaped like the public CritPT notebooks: problem setup, main problem, parsing template, and a complete answer() contract.",
        "It does not train on the official 70 prompts or official answers.",
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
                f"- reference_output: `{example.metadata.get('reference_output')}`",
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
        "# E4 Gold Rollout Preview",
        "",
        "This is not a model rollout. It shows the ideal answer shape for the first generated prompts.",
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
                "domain": example.metadata.get("domain"),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/data/e4_official_style"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/e4_official_style"))
    parser.add_argument("--train-size", type=int, default=1080)
    parser.add_argument("--val-size", type=int, default=144)
    parser.add_argument("--test-size", type=int, default=144)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--sample-per-family", type=int, default=1)
    parser.add_argument("--rollout-preview-count", type=int, default=12)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    train = generate_e4_official_style_examples(args.train_size, args.seed, "train")
    val = generate_e4_official_style_examples(args.val_size, args.seed + 1, "val")
    test = generate_e4_official_style_examples(args.test_size, args.seed + 2, "test")
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
        "train": args.data_dir / "critpt_e4_official_style_train.parquet",
        "val": args.data_dir / "critpt_e4_official_style_val.parquet",
        "test": args.data_dir / "critpt_e4_official_style_test.parquet",
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
        "name": "synthetic_critpt_e4_official_style_final_answer_judge",
        "profile": "e4_official_style_final_answer_judge",
        "seed": args.seed,
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
            "note": "E4 uses synthetic parameterized tasks based on public shell/type distribution, not official prompt text.",
        },
        "quality_gates": {
            "gold_target_code_executes": not args.skip_verify,
            "reference_output_present": True,
            "reward_uses_llm_judge_only": True,
            "prompt_contains_parsing_template": True,
            "target_is_complete_answer_code": True,
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
