#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample


def _strip_template(prompt: str) -> str:
    return prompt.split("### Parsing template:", 1)[0].rstrip()


def naturalize_prompt(prompt: str) -> str:
    core = _strip_template(prompt)
    return (
        f"{core}\n\n"
        "请直接解题。可以写必要推导、公式或简短代码辅助说明；"
        "最后给出清晰的最终答案。不要编造题目没有给出的条件。"
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


def prompt_for_style(example: SyntheticCritPTExample, prompt_style: str) -> str:
    if prompt_style == "compact_cot":
        return compact_cot_prompt(example.prompt)
    if prompt_style == "audit_cot":
        return audit_cot_prompt(example.prompt)
    if prompt_style == "audit_short":
        return audit_short_prompt(example.prompt)
    if prompt_style == "audit_trace":
        return audit_trace_prompt(example.prompt, example.metadata)
    return naturalize_prompt(example.prompt)


def read_examples(path: Path, limit: int | None) -> list[SyntheticCritPTExample]:
    examples: list[SyntheticCritPTExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(SyntheticCritPTExample.from_dict(json.loads(line)))
                if limit is not None and len(examples) >= limit:
                    break
    return examples


def build_chat_prompt(tokenizer: AutoTokenizer, prompt: str, *, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CritPT predictions with vLLM only.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--prompt-style",
        choices=["natural", "compact_cot", "audit_cot", "audit_short", "audit_trace"],
        default="audit_cot",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    examples = read_examples(args.data, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_texts = [prompt_for_style(example, args.prompt_style) for example in examples]
    chat_prompts = [
        build_chat_prompt(tokenizer, prompt_text, enable_thinking=args.enable_thinking)
        for prompt_text in prompt_texts
    ]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    outputs = llm.generate(
        chat_prompts,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        ),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for example, prompt_text, output in zip(examples, prompt_texts, outputs, strict=True):
            completion = output.outputs[0].text.strip()
            token_ids = output.outputs[0].token_ids or []
            handle.write(
                json.dumps(
                    {
                        "problem_id": example.problem_id,
                        "family": example.family,
                        "difficulty": example.difficulty,
                        "prompt": prompt_text,
                        "completion": completion,
                        "model": args.model,
                        "output_tokens": len(token_ids),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps({"predictions": str(args.out), "num_predictions": len(examples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
