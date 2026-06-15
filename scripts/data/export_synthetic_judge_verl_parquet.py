#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample


def read_examples(path: Path) -> list[SyntheticCritPTExample]:
    examples: list[SyntheticCritPTExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(SyntheticCritPTExample.from_dict(json.loads(line)))
    return examples


def _strip_template(prompt: str) -> str:
    return prompt.split("### Parsing template:", 1)[0].rstrip()


def naturalize_prompt(prompt: str) -> str:
    core = _strip_template(prompt)
    return (
        f"{core}\n\n"
        "请直接解题。可以写必要推导、公式或简短代码辅助说明；"
        "最后给出清晰的最终答案。不要编造题目没有给出的条件。"
    )


def cot_prompt(prompt: str) -> str:
    core = _strip_template(prompt)
    return (
        f"{core}\n\n"
        "请解题并保留必要推理过程。推理要围绕题目中的量、公式和约束展开，"
        "不要空泛自问自答，也不要编造题目没有给出的条件。"
        "最后必须单独写一行“最终答案：...”，给出清晰可判定的答案。"
    )


def compact_cot_prompt(prompt: str) -> str:
    core = _strip_template(prompt)
    return (
        f"{core}\n\n"
        "请用紧凑推理解题。最多写 6 行关键推导，只保留必要公式、代入和计算，"
        "不要反复解释题意，不要讨论多个可能解释，不要空泛自问自答。"
        "最后必须单独写一行“最终答案：...”，只给出清晰可判定的结果。"
    )


def audit_cot_prompt(prompt: str) -> str:
    core = _strip_template(prompt)
    return (
        f"{core}\n\n"
        "请用紧凑但可审计的推理解题。最多写 6 行关键推导，只保留必要公式、代入和计算。"
        "不要反复解释题意，不要讨论多个可能解释，不要空泛自问自答。"
        "禁止用“继续递推可得”“展开可得”“using known expansions”等话跳过关键计算；"
        "递推题要列出足够中间项，级数题要列出关键系数，枚举题要列出被接受的标签/权重和计数。"
        "若题目要求列表或元组，最终答案必须按题目要求给完整列表/元组，缺失位置要补 0，"
        "不要只给非零项、总和或代码片段。"
        "若题目要求 audit list，最终答案必须完整照抄该 audit list 的顺序，"
        "不要只给最后概率、trace 或 diagnostics。"
        "矩阵幂题必须做完整行列乘法并保留交叉项；高斯偶矩题不要漏掉 2^k 因子；"
        "二能级能隙题中耦合项和 detuning 独立，不要把 g 项乘上 delta；"
        "有理式不要未经验证就约分。"
        "最后必须单独写一行“最终答案：...”，只给出清晰可判定的结果。"
    )


def audit_short_prompt(prompt: str) -> str:
    core = _strip_template(prompt)
    return (
        f"{core}\n\n"
        "/no_think\n"
        "请用最多 3 行完成必要计算，不要输出 <think> 标签，不要长篇解释题意。"
        "如果题目要求短 audit list/tuple，最终答案必须只给这个 list/tuple，顺序和长度完全按题目。"
        "不要给完整推理长文，不要给代码块，不要只给最后一个标量。"
        "最后单独写一行“最终答案：...”。"
    )


def audit_trace_prompt(prompt: str, metadata: dict) -> str:
    core = _strip_template(prompt)
    tags = [str(tag) for tag in metadata.get("audit_tags", [])]
    if tags:
        audit_fields = ", ".join(f"{tag}=..." for tag in tags)
        audit_instruction = f"第一行必须写“审计：{audit_fields}”。"
    else:
        audit_instruction = "第一行必须写“审计：”，列出题目要求的关键中间量。"
    return (
        f"{core}\n\n"
        "/no_think\n"
        "请严格用两段输出，不要输出 <think> 标签，不要写代码块，不要长篇解释题意。"
        f"{audit_instruction}"
        "第二行必须写“最终答案：...”，最终答案只给题目要求的 list/tuple/表达式。"
        "审计字段和最终答案都要用实际数字或表达式，不要写变量名占位。"
    )


def cot_assistant_target(example: SyntheticCritPTExample) -> str:
    trace = example.solution_trace.strip() or "根据题目条件直接计算目标量。"
    return (
        "推理：\n"
        f"{trace}\n\n"
        "最终答案：\n"
        "```python\n"
        f"{example.target_code.strip()}\n"
        "```"
    )


def prompt_for_style(example: SyntheticCritPTExample, prompt_style: str) -> str:
    if prompt_style == "code":
        return example.prompt
    if prompt_style == "cot":
        return cot_prompt(example.prompt)
    if prompt_style == "compact_cot":
        return compact_cot_prompt(example.prompt)
    if prompt_style == "audit_cot":
        return audit_cot_prompt(example.prompt)
    if prompt_style == "audit_short":
        return audit_short_prompt(example.prompt)
    if prompt_style == "audit_trace":
        return audit_trace_prompt(example.prompt, example.metadata)
    return naturalize_prompt(example.prompt)


def cot_policy(prompt_style: str) -> str:
    if prompt_style == "cot":
        return "reasoning_allowed_reference_cot"
    if prompt_style == "compact_cot":
        return "compact_reasoning_required_final_answer"
    if prompt_style == "audit_cot":
        return "compact_auditable_reasoning_required_final_answer"
    if prompt_style == "audit_short":
        return "short_audit_no_think_final_answer"
    if prompt_style == "audit_trace":
        return "tagged_audit_trace_and_final_answer"
    if prompt_style == "code":
        return "code_completion"
    return "reasoning_allowed_reference_trace"


def naturalized_reference(example: SyntheticCritPTExample) -> str:
    return (
        "Reference reasoning notes:\n"
        f"{example.solution_trace.strip()}\n\n"
        "Reference executable answer:\n"
        f"{example.target_code.strip()}"
    ).strip()


def naturalize_legacy_prompt(prompt: str) -> str:
    core = prompt.split("### Parsing template:", 1)[0].rstrip()
    return (
        f"{core}\n\n"
        "请直接解题。可以写必要推导、公式或简短代码辅助说明；"
        "最后给出清晰的最终答案。不要编造题目没有给出的条件。"
    )


def to_verl_row(example: SyntheticCritPTExample, index: int, prompt_style: str) -> dict:
    prompt_text = prompt_for_style(example, prompt_style)
    return {
        "data_source": "critpt_model_judge",
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "critpt_problem_solving",
        "reward_model": {
            "style": "model_judge",
            "ground_truth": example.target_code,
        },
        "extra_info": {
            "split": example.split,
            "index": index,
            "problem_id": example.problem_id,
            "family": example.family,
            "difficulty": example.difficulty,
            "prompt_text": prompt_text,
            "reference_answer": naturalized_reference(example),
            "reference_trace": example.solution_trace,
            "reference_cot_answer": cot_assistant_target(example),
            "code_verifier": json.dumps(example.verifier, ensure_ascii=False),
            "cot_policy": cot_policy(prompt_style),
            "rubric": (
                "Judge scientific/mathematical correctness first. "
                "Useful reasoning is allowed and should be rewarded when it is grounded in the problem. "
                "Require a clear final answer. "
                "For compact_cot prompts, also reward concise reasoning that reaches the final answer "
                "without repetitive interpretation or unfinished exploration. "
                "For audit_cot prompts, penalize skipped calculations, unsupported 'therefore' jumps, "
                "and final answers that are guessed after omitting recurrence, series, or enumeration steps."
            ),
            "metadata": json.dumps(example.metadata, ensure_ascii=False),
        },
    }


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
        default="natural",
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
    print(f"wrote {len(train_rows)} train rows -> {args.train_out}")
    print(f"wrote {len(val_rows)} val rows -> {args.val_out}")

    if args.sft_train_out:
        sft_train_rows = []
        for row in train_rows:
            info = row["extra_info"]
            sft_train_rows.append(
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
                    },
                }
            )
        args.sft_train_out.parent.mkdir(parents=True, exist_ok=True)
        with args.sft_train_out.open("w", encoding="utf-8") as handle:
            for item in sft_train_rows:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"wrote {len(sft_train_rows)} SFT train rows -> {args.sft_train_out}")

    if args.sft_val_out:
        sft_val_rows = []
        for row in val_rows:
            info = row["extra_info"]
            sft_val_rows.append(
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
                    },
                }
            )
        args.sft_val_out.parent.mkdir(parents=True, exist_ok=True)
        with args.sft_val_out.open("w", encoding="utf-8") as handle:
            for item in sft_val_rows:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"wrote {len(sft_val_rows)} SFT val rows -> {args.sft_val_out}")


if __name__ == "__main__":
    main()
