#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample


def build_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def read_examples(path: Path, limit: int | None) -> list[SyntheticCritPTExample]:
    examples: list[SyntheticCritPTExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            examples.append(SyntheticCritPTExample.from_dict(json.loads(line)))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CritPT predictions with a full model.")
    parser.add_argument("--model", required=True, help="full model path")
    parser.add_argument("--data", type=Path, required=True, help="raw synthetic jsonl, e.g. val.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="prediction jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    examples = read_examples(args.data, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    do_sample = args.temperature > 0
    with args.out.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for idx, example in enumerate(examples, start=1):
            chat_prompt = build_prompt(tokenizer, example.prompt)
            inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            row = {
                "problem_id": example.problem_id,
                "completion": completion,
                "model": args.model,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"done": idx, "total": len(examples), "problem_id": example.problem_id}))

    print(json.dumps({"predictions": str(args.out), "num_predictions": len(examples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
