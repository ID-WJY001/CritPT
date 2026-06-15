#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, TypeVar

from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.critpt_synth.v18_official_long import (
    summarize_v18_source,
    verify_v18_example,
)


def read_examples(path: Path) -> list[SyntheticCritPTExample]:
    examples: list[SyntheticCritPTExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(SyntheticCritPTExample.from_dict(json.loads(line)))
    return examples


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


def rollout_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**9, path.name


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_rollout_input(text: str) -> str:
    text = text.strip()
    if text.startswith("user\n"):
        text = text[len("user\n") :]
    if text.endswith("\nassistant"):
        text = text[: -len("\nassistant")]
    if text.endswith("\nassistant\n"):
        text = text[: -len("\nassistant\n")]
    return text.strip()


def prompt_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def read_rollout_rows(rollout_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(rollout_dir.glob("*.jsonl"), key=rollout_sort_key):
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_rollout_file"] = path.name
                row["_rollout_line"] = idx + 1
                rows.append(row)
    return rows


def summarize_rollout_groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    prompt_text: dict[str, str] = {}
    for row in rows:
        normalized = normalize_rollout_input(str(row.get("input", row.get("prompt", ""))))
        key = prompt_hash(normalized)
        grouped.setdefault(key, []).append(row)
        prompt_text[key] = normalized

    result: dict[str, dict[str, Any]] = {}
    for key, group_rows in grouped.items():
        scores = [number(row.get("score", row.get("reward"))) for row in group_rows]
        accs = [number(row.get("acc")) for row in group_rows]
        unique_scores = {round(score, 6) for score in scores}
        all_wrong = bool(accs) and all(value <= 0.0 for value in accs)
        all_correct = bool(accs) and all(value >= 1.0 for value in accs)
        no_think_rate = sum(number(row.get("no_think_tags")) >= 1.0 for row in group_rows) / max(
            len(group_rows), 1
        )
        has_unclosed = any(number(row.get("unclosed_code_block")) >= 1.0 for row in group_rows)
        max_chars = max((len(str(row.get("output", ""))) for row in group_rows), default=0)
        score_mean = sum(scores) / max(len(scores), 1)
        acc_mean = sum(accs) / max(len(accs), 1)
        severity = 0.0
        severity += 3.0 if all_wrong else 0.0
        severity += 1.5 * (1.0 - acc_mean)
        severity += max(0.0, 0.92 - score_mean)
        severity += max(0.0, 0.88 - min(scores, default=0.0))
        severity += 0.35 if len(unique_scores) > 1 else 0.0
        severity += 0.20 if has_unclosed else 0.0
        severity += 0.15 if no_think_rate < 0.5 else 0.0
        result[key] = {
            "prompt_hash": key,
            "n": len(group_rows),
            "score_mean": score_mean,
            "score_min": min(scores, default=0.0),
            "score_max": max(scores, default=0.0),
            "acc_mean": acc_mean,
            "useful": len(unique_scores) > 1,
            "all_wrong": all_wrong,
            "all_correct": all_correct,
            "no_think_rate": no_think_rate,
            "has_unclosed_code_block": has_unclosed,
            "max_output_chars": max_chars,
            "severity": severity,
            "prompt": prompt_text[key],
        }
    return result


def select_failure_examples(
    examples: list[SyntheticCritPTExample],
    group_stats: dict[str, dict[str, Any]],
) -> list[tuple[SyntheticCritPTExample, dict[str, Any]]]:
    selected: list[tuple[SyntheticCritPTExample, dict[str, Any]]] = []
    for example in examples:
        stats = group_stats.get(prompt_hash(example.prompt.strip()))
        if not stats:
            continue
        hard = (
            stats["all_wrong"]
            or stats["useful"]
            or stats["acc_mean"] < 1.0
            or stats["score_min"] < 0.88
            or stats["no_think_rate"] < 0.5
        )
        if hard:
            selected.append((example, stats))
    selected.sort(key=lambda item: (-float(item[1]["severity"]), item[0].problem_id))
    return selected


def hard_background_examples(
    examples: list[SyntheticCritPTExample],
    selected_ids: set[str],
) -> list[SyntheticCritPTExample]:
    hard_terms = (
        "operator",
        "piecewise",
        "recurrence",
        "interval",
        "holography",
        "oam",
        "bns",
        "lamet",
        "failure",
        "sparse",
        "multi",
    )
    pool = [
        example
        for example in examples
        if example.problem_id not in selected_ids
        and (
            str(example.difficulty).lower() in {"hard", "expert"}
            or any(term in f"{example.family} {example.metadata}".lower() for term in hard_terms)
        )
    ]
    return pool or [example for example in examples if example.problem_id not in selected_ids] or examples


T = TypeVar("T")


def cycled_sample(pool: list[T], size: int, rng: random.Random) -> list[T]:
    if size <= 0:
        return []
    if not pool:
        raise ValueError("cannot sample from an empty pool")
    result: list[T] = []
    while len(result) < size:
        chunk = list(pool)
        rng.shuffle(chunk)
        result.extend(chunk)
    return result[:size]


def retag_example(
    example: SyntheticCritPTExample,
    *,
    split: str,
    idx: int,
    source: str,
    stats: dict[str, Any] | None = None,
) -> SyntheticCritPTExample:
    stats = stats or {}
    suffix_payload = {
        "problem_id": example.problem_id,
        "split": split,
        "idx": idx,
        "source": source,
        "prompt_hash": prompt_hash(example.prompt.strip()),
    }
    suffix = hashlib.sha256(json.dumps(suffix_payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    metadata = {
        **example.metadata,
        "generator_profile": "v19_failure_mined_from_v18",
        "v19_source": source,
        "v19_original_problem_id": example.problem_id,
        "v19_prompt_hash": prompt_hash(example.prompt.strip()),
        "v19_failure_stats": {
            key: stats[key]
            for key in [
                "score_mean",
                "score_min",
                "score_max",
                "acc_mean",
                "useful",
                "all_wrong",
                "all_correct",
                "no_think_rate",
                "max_output_chars",
                "severity",
            ]
            if key in stats
        },
        "uses_official_prompt": False,
        "official_overlap": "none",
        "no_think_target": True,
        "failure_mined": source == "v18_rollout_failure",
    }
    return replace(
        example,
        problem_id=f"{example.problem_id}_v19{idx:05d}_{suffix}",
        split=split,
        metadata=metadata,
    )


def verify_targets(examples: list[SyntheticCritPTExample]) -> None:
    failures: list[tuple[str, str]] = []
    for example in examples:
        ok, reason = verify_v18_example(example)
        if not ok:
            failures.append((example.problem_id, reason))
    if failures:
        preview = "\n".join(f"{problem_id}: {reason}" for problem_id, reason in failures[:25])
        raise SystemExit(f"{len(failures)} V19 examples failed verification:\n{preview}")


def summarize_examples(examples: list[SyntheticCritPTExample]) -> dict[str, Any]:
    return {
        "total": len(examples),
        "split": dict(Counter(example.split for example in examples)),
        "source": dict(Counter(str(example.metadata.get("v19_source", "unknown")) for example in examples)),
        "v18_source": dict(Counter(summarize_v18_source(example) for example in examples)),
        "family": dict(Counter(example.family for example in examples)),
        "difficulty": dict(Counter(example.difficulty for example in examples)),
        "domain": dict(Counter(str(example.metadata.get("domain", "unknown")) for example in examples)),
        "answer_type": dict(Counter(str(example.metadata.get("answer_type", "unknown")) for example in examples)),
    }


def write_failure_csv(path: Path, selected: list[tuple[SyntheticCritPTExample, dict[str, Any]]]) -> None:
    fields = [
        "rank",
        "problem_id",
        "prompt_hash",
        "family",
        "difficulty",
        "v18_source",
        "severity",
        "score_mean",
        "score_min",
        "score_max",
        "acc_mean",
        "useful",
        "all_wrong",
        "all_correct",
        "no_think_rate",
        "max_output_chars",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, (example, stats) in enumerate(selected, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "problem_id": example.problem_id,
                    "prompt_hash": prompt_hash(example.prompt.strip()),
                    "family": example.family,
                    "difficulty": example.difficulty,
                    "v18_source": summarize_v18_source(example),
                    **{key: stats.get(key, "") for key in fields if key not in {"rank", "problem_id", "prompt_hash", "family", "difficulty", "v18_source"}},
                }
            )


def write_sample_markdown(
    path: Path,
    selected: list[tuple[SyntheticCritPTExample, dict[str, Any]]],
    train_examples: list[SyntheticCritPTExample],
    per_section: int,
) -> None:
    lines = [
        "# V19 Failure-Mined 数据抽样",
        "",
        "V19 的核心不是继续堆随机题，而是把 V18 训练过程中真实暴露出来的低分题、全错题、同组分数分化题重新采样。",
        "这些题仍然是 synthetic hard cases，不包含官方 70/71 题原文。",
        "",
        "## 被挖出来的高优先级失败题",
        "",
    ]
    for rank, (example, stats) in enumerate(selected[:per_section], start=1):
        lines.extend(
            [
                f"### {rank}. `{example.problem_id}`",
                "",
                f"- family: `{example.family}`",
                f"- difficulty: `{example.difficulty}`",
                f"- v18_source: `{summarize_v18_source(example)}`",
                f"- rollout score_mean/score_min/score_max: `{stats['score_mean']:.4f}` / `{stats['score_min']:.4f}` / `{stats['score_max']:.4f}`",
                f"- acc_mean: `{stats['acc_mean']:.4f}`",
                f"- useful/all_wrong/no_think_rate: `{stats['useful']}` / `{stats['all_wrong']}` / `{stats['no_think_rate']:.4f}`",
                "",
                "#### Prompt",
                "",
                "```text",
                example.prompt,
                "```",
                "",
                "#### Gold",
                "",
                example.assistant_code_block(),
                "",
            ]
        )

    lines.extend(["## V19 train 抽样结果", ""])
    for example in train_examples[:per_section]:
        stats = example.metadata.get("v19_failure_stats", {})
        lines.extend(
            [
                f"### `{example.problem_id}`",
                "",
                f"- source: `{example.metadata.get('v19_source')}`",
                f"- original: `{example.metadata.get('v19_original_problem_id')}`",
                f"- family: `{example.family}`",
                f"- failure_mined: `{example.metadata.get('failure_mined')}`",
                f"- inherited_stats: `{json.dumps(stats, ensure_ascii=False)}`",
                "",
                "```text",
                example.prompt,
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v18-train-jsonl", type=Path, required=True)
    parser.add_argument("--v18-val-jsonl", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/data/v19_failure_mined"))
    parser.add_argument("--name", default="synthetic_critpt_v19_failure_mined")
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--failure-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--sample-per-section", type=int, default=12)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_source = read_examples(args.v18_train_jsonl)
    val_source = read_examples(args.v18_val_jsonl)
    rollout_rows = read_rollout_rows(args.rollout_dir)
    group_stats = summarize_rollout_groups(rollout_rows)
    selected = select_failure_examples(train_source, group_stats)
    if not selected:
        raise SystemExit("no V18 rollout failures matched the V18 train JSONL")

    selected_ids = {example.problem_id for example, _stats in selected}
    background_pool = hard_background_examples(train_source, selected_ids)
    stats_by_id = {example.problem_id: stats for example, stats in selected}

    failure_size = min(args.train_size, max(1, round(args.train_size * args.failure_ratio)))
    background_size = args.train_size - failure_size
    failure_examples = cycled_sample([example for example, _stats in selected], failure_size, rng)
    background_examples = cycled_sample(background_pool, background_size, rng)
    train_examples = [
        retag_example(
            example,
            split="train",
            idx=idx,
            source="v18_rollout_failure",
            stats=stats_by_id.get(example.problem_id, {}),
        )
        for idx, example in enumerate(failure_examples)
    ]
    train_examples.extend(
        retag_example(example, split="train", idx=failure_size + idx, source="v18_hard_background")
        for idx, example in enumerate(background_examples)
    )
    rng.shuffle(train_examples)

    val_examples = [
        retag_example(example, split="val", idx=idx, source="v18_heldout_val")
        for idx, example in enumerate(cycled_sample(val_source, args.val_size, rng))
    ]
    test_examples = [
        retag_example(example, split="test", idx=idx, source="v18_heldout_val")
        for idx, example in enumerate(cycled_sample(val_source, args.test_size, rng))
    ]
    all_examples = train_examples + val_examples + test_examples

    if not args.skip_verify:
        verify_targets(all_examples)

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
    write_jsonl(raw_paths["train"], (example.to_dict() for example in train_examples))
    write_jsonl(raw_paths["val"], (example.to_dict() for example in val_examples))
    write_jsonl(raw_paths["test"], (example.to_dict() for example in test_examples))
    write_jsonl(sft_paths["train"], (example.to_sft_row() for example in train_examples))
    write_jsonl(sft_paths["val"], (example.to_sft_row() for example in val_examples))
    write_jsonl(sft_paths["test"], (example.to_sft_row() for example in test_examples))

    failure_csv = args.out_dir / "failure_candidates.csv"
    failure_json = args.out_dir / "failure_candidates.json"
    write_failure_csv(failure_csv, selected)
    failure_json.write_text(
        json.dumps(
            [
                {
                    "rank": rank,
                    "problem_id": example.problem_id,
                    "family": example.family,
                    "difficulty": example.difficulty,
                    "v18_source": summarize_v18_source(example),
                    "prompt_hash": prompt_hash(example.prompt.strip()),
                    "stats": stats,
                }
                for rank, (example, stats) in enumerate(selected, start=1)
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    sample_path = args.out_dir / "samples.zh-CN.md"
    write_sample_markdown(sample_path, selected, train_examples, args.sample_per_section)

    manifest = {
        "name": args.name,
        "profile": "v19_failure_mined_from_v18",
        "seed": args.seed,
        "input": {
            "v18_train_jsonl": str(args.v18_train_jsonl),
            "v18_val_jsonl": str(args.v18_val_jsonl),
            "rollout_dir": str(args.rollout_dir),
            "rollout_rows": len(rollout_rows),
            "rollout_groups": len(group_stats),
        },
        "selection": {
            "matched_failure_candidates": len(selected),
            "failure_ratio": args.failure_ratio,
            "failure_train_rows": failure_size,
            "background_train_rows": background_size,
            "top_failure_problem_ids": [example.problem_id for example, _stats in selected[:20]],
        },
        "summary": summarize_examples(all_examples),
        "raw_files": {split: str(path) for split, path in raw_paths.items()},
        "sft_files": {split: str(path) for split, path in sft_paths.items()},
        "failure_candidates_csv": str(failure_csv),
        "failure_candidates_json": str(failure_json),
        "sample_markdown": str(sample_path),
        "sha256": {
            **{f"raw_{split}": file_sha256(path) for split, path in raw_paths.items()},
            **{f"sft_{split}": file_sha256(path) for split, path in sft_paths.items()},
            "failure_candidates_csv": file_sha256(failure_csv),
            "failure_candidates_json": file_sha256(failure_json),
        },
        "leakage_policy": {
            "official_prompts_used_for_training": False,
            "official_challenge_ids_used_for_training": False,
            "note": "V19 mines failures from synthetic V18 rollout logs only; it does not copy official test prompt text.",
        },
        "quality_gates": {
            "gold_target_code_passes_code_verifier": not args.skip_verify,
            "prompt_contains_parsing_template": True,
            "target_is_complete_answer_function": True,
            "failure_mined_from_real_rollout": True,
            "heldout_val_from_v18_val": True,
            "no_think_target": True,
        },
        "host": {
            "cwd": str(Path.cwd()),
            "cpu_count": os.cpu_count(),
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
