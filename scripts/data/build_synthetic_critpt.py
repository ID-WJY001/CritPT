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

from rl_posttrain.critpt_synth.generators import (
    generate_examples,
    generate_hardcase_examples,
    generate_v7_compact_examples,
    generate_v7_intermediate_examples,
    generate_v9_trace_examples,
    generate_v10_curriculum_trace_examples,
    generate_v11_template_series_trace_examples,
    generate_v13_official_style_examples,
)
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.critpt_synth.verifier import verify_code_completion


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


def _verify_one_target(example: SyntheticCritPTExample) -> tuple[str, bool, str]:
    completion = example.assistant_code_block()
    result = verify_code_completion(completion, example.verifier)
    return example.problem_id, result.ok, result.reason


def verify_targets(examples: list[SyntheticCritPTExample], workers: int) -> None:
    failures: list[tuple[str, str]] = []
    total = len(examples)
    progress_every = max(1000, total // 20)
    if workers <= 1:
        for count, example in enumerate(examples, start=1):
            problem_id, ok, reason = _verify_one_target(example)
            if not ok:
                failures.append((problem_id, reason))
            if count == total or count % progress_every == 0:
                print(f"[verify] {count}/{total} targets checked", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for count, (problem_id, ok, reason) in enumerate(
                executor.map(_verify_one_target, examples, chunksize=16), start=1
            ):
                if not ok:
                    failures.append((problem_id, reason))
                if count == total or count % progress_every == 0:
                    print(f"[verify] {count}/{total} targets checked", flush=True)
    if failures:
        preview = "\n".join(f"{pid}: {reason}" for pid, reason in failures[:20])
        raise SystemExit(f"{len(failures)} generated targets failed verifier:\n{preview}")


def summarize(examples: list[SyntheticCritPTExample]) -> dict:
    return {
        "total": len(examples),
        "family": dict(Counter(example.family for example in examples)),
        "difficulty": dict(Counter(example.difficulty for example in examples)),
        "split": dict(Counter(example.split for example in examples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/synthetic_critpt/v0"))
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--profile",
        choices=[
            "default",
            "v6_hardcase",
            "v7_intermediate",
            "v7_compact",
            "v9_trace",
            "v10_curriculum_trace",
            "v11_template_series_trace",
            "v13_official_style",
        ],
        default="default",
    )
    parser.add_argument("--train-size", type=int, default=3500)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="parallel verifier workers; target answers are generated locally and checked before writing",
    )
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    if args.profile == "v6_hardcase":
        generate = generate_hardcase_examples
    elif args.profile == "v7_intermediate":
        generate = generate_v7_intermediate_examples
    elif args.profile == "v7_compact":
        generate = generate_v7_compact_examples
    elif args.profile == "v9_trace":
        generate = generate_v9_trace_examples
    elif args.profile == "v10_curriculum_trace":
        generate = generate_v10_curriculum_trace_examples
    elif args.profile == "v11_template_series_trace":
        generate = generate_v11_template_series_trace_examples
    elif args.profile == "v13_official_style":
        generate = generate_v13_official_style_examples
    else:
        generate = generate_examples
    train = generate(args.train_size, args.seed, "train")
    val = generate(args.val_size, args.seed + 1, "val")
    test = generate(args.test_size, args.seed + 2, "test")
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

    sft_paths = {
        "train": args.out_dir / "train_sft_messages.jsonl",
        "val": args.out_dir / "val_sft_messages.jsonl",
        "test": args.out_dir / "test_sft_messages.jsonl",
    }
    write_jsonl(sft_paths["train"], (example.to_sft_row() for example in train))
    write_jsonl(sft_paths["val"], (example.to_sft_row() for example in val))
    write_jsonl(sft_paths["test"], (example.to_sft_row() for example in test))

    manifest = {
        "name": args.name or f"synthetic_critpt_{args.out_dir.name}",
        "profile": args.profile,
        "seed": args.seed,
        "summary": summarize(all_examples),
        "raw_files": {split: str(path) for split, path in raw_paths.items()},
        "sft_files": {split: str(path) for split, path in sft_paths.items()},
        "sha256": {
            **{f"raw_{split}": file_sha256(path) for split, path in raw_paths.items()},
            **{f"sft_{split}": file_sha256(path) for split, path in sft_paths.items()},
        },
        "leakage_policy": {
            "official_70_prompts_used_for_training": False,
            "official_challenge_ids_used_for_training": False,
            "note": "Generators are synthetic and do not read the official public-test prompts.",
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
