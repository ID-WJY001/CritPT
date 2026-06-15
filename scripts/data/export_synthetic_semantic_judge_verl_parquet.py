#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample

_HELPER_PATH = ROOT / "scripts" / "data" / "export_synthetic_judge_verl_parquet.py"
_HELPER_SPEC = importlib.util.spec_from_file_location("export_synthetic_judge_verl_parquet_helpers", _HELPER_PATH)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError(f"failed to load helper module: {_HELPER_PATH}")
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER)
cot_assistant_target = _HELPER.cot_assistant_target
cot_policy = _HELPER.cot_policy
prompt_for_style = _HELPER.prompt_for_style


def read_examples(path: Path) -> list[SyntheticCritPTExample]:
    examples: list[SyntheticCritPTExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(SyntheticCritPTExample.from_dict(json.loads(line)))
    return examples


def semantic_reference(example: SyntheticCritPTExample) -> str:
    return (
        "Reference reasoning notes:\n"
        f"{example.solution_trace.strip() or '(none)'}\n\n"
        "Reference executable answer code:\n"
        "```python\n"
        f"{example.target_code.strip()}\n"
        "```"
    ).strip()


def semantic_rubric(example: SyntheticCritPTExample, prompt_style: str) -> str:
    metadata = example.metadata
    focus = str(metadata.get("v20_focus") or metadata.get("v21_focus") or "")
    answer_type = str(metadata.get("answer_type", ""))
    parts = [
        "Use LLM-as-a-judge only. Grade the candidate response directly against the problem and reference.",
        "Scientific/mathematical semantic correctness is primary; formatting alone is not enough.",
        "The final returned object must have the requested type, ordering, labels, and edge-case behavior.",
        "Do not give high reward for plausible-looking supersets, missing entries, or unsupported shortcuts.",
    ]
    if prompt_style in {"audit_short", "audit_trace"}:
        parts.append("The candidate should stay concise and should not output hidden <think> text.")
    if answer_type:
        parts.append(f"The expected answer type is {answer_type}; penalize returning a different object type.")
    if focus == "operator_canonical_label":
        parts.extend(
            [
                "For operator tasks, canonical labels and ordering matter exactly.",
                "Penalize concatenated labels such as psipsi/Fpsi and non-canonical powers like tr(B)^2.",
                "Penalize returning a candidate superset instead of applying every charge/filter rule.",
            ]
        )
    elif focus == "empty_interval_filter":
        parts.extend(
            [
                "For interval/filter tasks, an empty result must be judged as exactly empty.",
                "Penalize returning rejected candidates or a non-empty set when the reference is empty.",
            ]
        )
    elif focus == "hhg_oam_runtime_safe":
        parts.extend(
            [
                "For HHG/OAM tasks, judge the exact list of selected channels/triples.",
                "Penalize variable-name mixups or formulas that produce the wrong channel set.",
            ]
        )
    return " ".join(parts)


def to_verl_row(example: SyntheticCritPTExample, index: int, prompt_style: str) -> dict[str, Any]:
    prompt_text = prompt_for_style(example, prompt_style)
    reference = semantic_reference(example)
    return {
        "data_source": "critpt_semantic_judge",
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "critpt_problem_solving",
        "reward_model": {
            "style": "llm_semantic_judge",
            "ground_truth": reference,
        },
        "extra_info": {
            "split": example.split,
            "index": index,
            "problem_id": example.problem_id,
            "family": example.family,
            "difficulty": example.difficulty,
            "prompt_text": prompt_text,
            "reference_answer": reference,
            "reference_trace": example.solution_trace,
            "reference_cot_answer": cot_assistant_target(example),
            "reference_answer_type": str(example.metadata.get("answer_type", "")),
            "reference_family": example.family,
            "cot_policy": cot_policy(prompt_style),
            "rubric": semantic_rubric(example, prompt_style),
            "metadata": json.dumps(example.metadata, ensure_ascii=False),
        },
    }


def _write_sft_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            info = row["extra_info"]
            handle.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": info["prompt_text"]},
                            {"role": "assistant", "content": info["reference_cot_answer"]},
                        ],
                        "metadata": {
                            "split": info["split"],
                            "index": info["index"],
                            "problem_id": info["problem_id"],
                            "family": info["family"],
                            "difficulty": info["difficulty"],
                            "cot_policy": info["cot_policy"],
                            "reward_style": "llm_semantic_judge",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--val-out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--prompt-style",
        choices=["natural", "code", "cot", "compact_cot", "audit_cot", "audit_short", "audit_trace"],
        default="code",
    )
    parser.add_argument("--sft-train-out", type=Path, default=None)
    parser.add_argument("--sft-val-out", type=Path, default=None)
    args = parser.parse_args()

    train_examples = read_examples(args.train_jsonl)
    val_examples = read_examples(args.val_jsonl)

    train_rows = []
    for repeat_idx in range(args.repeat):
        for idx, example in enumerate(train_examples):
            row = to_verl_row(example, repeat_idx * len(train_examples) + idx, args.prompt_style)
            row["extra_info"]["repeat_idx"] = repeat_idx
            train_rows.append(row)
    val_rows = [to_verl_row(example, idx, args.prompt_style) for idx, example in enumerate(val_examples)]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(args.train_out, index=False)
    pd.DataFrame(val_rows).to_parquet(args.val_out, index=False)
    print(f"wrote {len(train_rows)} semantic judge train rows -> {args.train_out}")
    print(f"wrote {len(val_rows)} semantic judge val rows -> {args.val_out}")

    if args.sft_train_out:
        _write_sft_rows(args.sft_train_out, train_rows)
        print(f"wrote {len(train_rows)} SFT train rows -> {args.sft_train_out}")
    if args.sft_val_out:
        _write_sft_rows(args.sft_val_out, val_rows)
        print(f"wrote {len(val_rows)} SFT val rows -> {args.sft_val_out}")


if __name__ == "__main__":
    main()
