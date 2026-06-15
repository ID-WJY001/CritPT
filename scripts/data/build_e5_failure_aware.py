#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from rl_posttrain.critpt_synth.e5_failure_aware import (
    E5_PROFILE,
    generate_e5_failure_aware_examples,
    verify_e5_example,
)
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample


FOCUS_TARGET_WEIGHTS = {
    "operator_canonical_precision": 0.25,
    "exact_filter_or_empty_set": 0.20,
    "symbolic_expression_exactness": 0.18,
    "nonplaceholder_coefficient_list": 0.12,
    "hhg_oam_channel_scan": 0.10,
    "symbolic_choice_exactness": 0.08,
    "general_final_answer_exactness": 0.07,
}


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
        "source": dict(Counter(str(example.metadata.get("e5_source", "unknown")) for example in examples)),
        "focus": dict(Counter(str(example.metadata.get("e5_focus", "unknown")) for example in examples)),
        "family": dict(Counter(example.family for example in examples)),
        "difficulty": dict(Counter(example.difficulty for example in examples)),
        "domain": dict(Counter(str(example.metadata.get("domain", "unknown")) for example in examples)),
        "answer_type": dict(Counter(str(example.metadata.get("answer_type", "unknown")) for example in examples)),
        "expected_empty": dict(Counter(str(example.metadata.get("expected_empty", "n/a")) for example in examples)),
        "prompt_chars": {
            "min": min(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
    }


def verify_targets(examples: list[SyntheticCritPTExample], workers: int) -> None:
    failures: list[tuple[str, str]] = []
    if workers <= 1:
        for example in examples:
            ok, reason = verify_e5_example(example)
            if not ok:
                failures.append((example.problem_id, reason))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for example, (ok, reason) in zip(
                examples,
                executor.map(verify_e5_example, examples, chunksize=16),
                strict=True,
            ):
                if not ok:
                    failures.append((example.problem_id, reason))
    if failures:
        preview = "\n".join(f"{problem_id}: {reason}" for problem_id, reason in failures[:25])
        raise SystemExit(f"{len(failures)} E5 examples failed verification:\n{preview}")


def generate_balanced_focus_examples(size: int, seed: int, split: str) -> list[SyntheticCritPTExample]:
    pool_size = max(size * 4, size + 256)
    pool = generate_e5_failure_aware_examples(pool_size, seed, split)
    groups: dict[str, list[SyntheticCritPTExample]] = defaultdict(list)
    for example in pool:
        groups[str(example.metadata.get("e5_focus", "general_final_answer_exactness"))].append(example)

    targets = {focus: int(size * weight) for focus, weight in FOCUS_TARGET_WEIGHTS.items()}
    remainder = size - sum(targets.values())
    ordered_focuses = list(FOCUS_TARGET_WEIGHTS)
    for idx in range(remainder):
        targets[ordered_focuses[idx % len(ordered_focuses)]] += 1

    selected: list[SyntheticCritPTExample] = []
    selected_ids: set[str] = set()
    for focus in ordered_focuses:
        for example in groups.get(focus, [])[: targets[focus]]:
            selected.append(example)
            selected_ids.add(example.problem_id)

    if len(selected) < size:
        for example in pool:
            if example.problem_id in selected_ids:
                continue
            selected.append(example)
            selected_ids.add(example.problem_id)
            if len(selected) >= size:
                break

    if len(selected) < size:
        raise RuntimeError(f"balanced E5 pool too small for {split}: selected {len(selected)} of {size}")

    return interleave_by_focus(selected[:size])


def interleave_by_focus(examples: list[SyntheticCritPTExample]) -> list[SyntheticCritPTExample]:
    groups: dict[str, list[SyntheticCritPTExample]] = defaultdict(list)
    for example in examples:
        groups[str(example.metadata.get("e5_focus", "unknown"))].append(example)

    focuses = sorted(groups, key=lambda focus: (-len(groups[focus]), focus))
    interleaved: list[SyntheticCritPTExample] = []
    while len(interleaved) < len(examples):
        made_progress = False
        for focus in focuses:
            if groups[focus]:
                interleaved.append(groups[focus].pop(0))
                made_progress = True
        if not made_progress:
            break
    return interleaved


def reference_answer(example: SyntheticCritPTExample) -> str:
    return (
        "Trusted executable answer code:\n"
        "```python\n"
        f"{example.target_code.strip()}\n"
        "```"
    )


def reference_rubric(example: SyntheticCritPTExample) -> str:
    focus = str(example.metadata.get("e5_focus", ""))
    answer_type = str(example.metadata.get("answer_type", ""))
    parts = [
        "Use LLM-as-a-judge only. Grade the candidate answer() against the trusted final output and code.",
        "The final returned object is primary: exact values, symbolic equivalence, labels, ordering, filters, empty-case behavior, and return type matter.",
        "Reward compact literal answers only when the literal final value matches the trusted reference.",
        "Strongly penalize all-zero placeholders, whole-universe/superset answers, guessed labels, and template-shaped code with wrong contents.",
    ]
    if focus == "operator_canonical_precision":
        parts.append(
            "For operator labels, canonical string notation is exact: tr(name^k) is not tr(name)^k, and extra plausible labels are wrong."
        )
    elif focus == "exact_filter_or_empty_set":
        parts.append("For filter tasks, set() can be the exact answer; supersets or rejected near misses must be penalized.")
    elif focus == "hhg_oam_channel_scan":
        parts.append("For HHG/OAM tasks, each channel's order, OAM, helicity, and ordering must match exactly.")
    elif focus == "nonplaceholder_coefficient_list":
        parts.append("For coefficient lists, every entry must be computed in the requested basis order; all-zero stubs are wrong.")
    elif focus == "symbolic_choice_exactness":
        parts.append("For symbolic choice tasks, both the selected letter and the expression/value must match.")
    elif focus == "symbolic_expression_exactness":
        parts.append("For symbolic expressions, algebraic equivalence is fine, but wrong variables, signs, or factors are not.")
    if answer_type:
        parts.append(f"The expected answer type is {answer_type}.")
    return " ".join(parts)


def to_verl_row(example: SyntheticCritPTExample, index: int) -> dict[str, Any]:
    ref_answer = reference_answer(example)
    reference_output = str(example.metadata["reference_output"])
    return {
        "data_source": "critpt_e5_failure_aware_final_answer_judge",
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
            "e5_source": str(example.metadata.get("e5_source", "")),
            "e5_focus": str(example.metadata.get("e5_focus", "")),
            "cot_policy": "code_completion_official_shell_no_think_failure_aware",
            "rubric": reference_rubric(example),
            "metadata": json.dumps(example.metadata, ensure_ascii=False, sort_keys=True),
        },
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_sample_markdown(path: Path, examples: list[SyntheticCritPTExample], per_focus: int) -> None:
    selected: list[SyntheticCritPTExample] = []
    counts: Counter[str] = Counter()
    for example in examples:
        focus = str(example.metadata.get("e5_focus", "unknown"))
        if counts[focus] >= per_focus:
            continue
        selected.append(example)
        counts[focus] += 1

    lines = [
        "# E5 Failure-Aware Data Samples",
        "",
        "E5 mixes E4 official-shell tasks with V20/V21 failure-focused tasks. It does not use official 70 prompt text or official answers.",
        "The reward path remains LLM-as-a-judge; local execution is only used here to create trusted reference outputs.",
        "",
    ]
    for example in selected:
        lines.extend(
            [
                f"## `{example.problem_id}`",
                "",
                f"- split: `{example.split}`",
                f"- source: `{example.metadata.get('e5_source')}`",
                f"- focus: `{example.metadata.get('e5_focus')}`",
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
        "# E5 Gold Rollout Preview",
        "",
        "This is not a model rollout. It shows the target answer shape for generated prompts.",
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
                "source": example.metadata.get("e5_source"),
                "focus": example.metadata.get("e5_focus"),
                "domain": example.metadata.get("domain"),
                "answer_type": example.metadata.get("answer_type"),
            }
        )
        lines.extend(
            [
                f"## {idx}. `{example.problem_id}`",
                "",
                f"- source: `{example.metadata.get('e5_source')}`",
                f"- focus: `{example.metadata.get('e5_focus')}`",
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
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/data/e5_failure_aware"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/e5_failure_aware"))
    parser.add_argument("--train-size", type=int, default=1800)
    parser.add_argument("--val-size", type=int, default=240)
    parser.add_argument("--test-size", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--dataset-prefix", default="critpt_e5_failure_aware")
    parser.add_argument("--no-balanced-focus", action="store_true")
    parser.add_argument("--sample-per-focus", type=int, default=1)
    parser.add_argument("--rollout-preview-count", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    generator = generate_e5_failure_aware_examples if args.no_balanced_focus else generate_balanced_focus_examples
    train = generator(args.train_size, args.seed, "train")
    val = generator(args.val_size, args.seed + 1, "val")
    test = generator(args.test_size, args.seed + 2, "test")
    all_examples = train + val + test

    if not args.skip_verify:
        verify_targets(all_examples, args.workers)

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
    write_sample_markdown(sample_path, all_examples, args.sample_per_focus)
    write_gold_rollout_preview(rollout_md, rollout_jsonl, all_examples, args.rollout_preview_count)

    manifest = {
        "name": "synthetic_critpt_e5_failure_aware_final_answer_judge",
        "profile": E5_PROFILE,
        "seed": args.seed,
        "dataset_prefix": args.dataset_prefix,
        "balanced_focus": not args.no_balanced_focus,
        "focus_target_weights": FOCUS_TARGET_WEIGHTS,
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
        "borrowed_from_v_series": {
            "v19": "failure mining as a design loop, not direct prompt copying",
            "v20": "operator canonical labels, empty interval filters, HHG/OAM channel scans",
            "v21": "anti-overenumeration and precise canonical operator notation",
        },
        "leakage_policy": {
            "official_70_prompts_used_for_training": False,
            "official_challenge_ids_used_for_training": False,
            "official_answers_used_for_training": False,
            "note": "E5 uses synthetic parameterized tasks and V-series failure analogues; no official prompt text is copied.",
        },
        "quality_gates": {
            "gold_target_code_executes": not args.skip_verify,
            "reference_output_present": True,
            "reward_uses_llm_judge_only": True,
            "prompt_contains_parsing_template": True,
            "target_is_complete_answer_code": True,
            "failure_aware_guardrails": True,
            "focused_operator_precision": True,
            "focused_empty_set_and_superset_penalty": True,
            "focused_nonplaceholder_coefficients": True,
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
