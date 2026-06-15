#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from rl_posttrain.critpt.eval import load_jsonl
from rl_posttrain.critpt.verifier import verify_completion


DEFAULT_SYSTEM = (
    "You are solving a technical benchmark problem. Reason carefully, then put exactly one "
    "final expression inside <answer>...</answer>."
)


def build_prompt(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt: str,
    enable_thinking: bool,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a vLLM baseline on CritPT-style JSONL.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    examples = load_jsonl(args.data)
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit(f"no examples found in {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [
        build_prompt(tokenizer, example.prompt, args.system_prompt, args.enable_thinking)
        for example in examples
    ]

    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(prompts, sampling)

    rows = []
    correct = 0
    for example, output in zip(examples, outputs, strict=True):
        completion = output.outputs[0].text
        result = verify_completion(completion, example.verifier)
        correct += int(result.ok)
        rows.append(
            {
                "problem_id": example.problem_id,
                "ok": result.ok,
                "score": result.score,
                "reason": result.reason,
                "extracted": result.extracted,
                "expected": example.verifier.expected,
                "prompt": example.prompt,
                "completion": completion,
                "metadata": example.metadata,
            }
        )

    total = len(rows)
    report = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "model": str(args.model),
        "data": str(args.data),
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_thinking": args.enable_thinking,
        },
        "rows": rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "report.json", report)
    with (args.out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            if not row["ok"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, ensure_ascii=False))
    print(f"wrote report: {args.out_dir / 'report.json'}")
    print(f"wrote failures: {args.out_dir / 'failures.jsonl'}")


if __name__ == "__main__":
    main()
