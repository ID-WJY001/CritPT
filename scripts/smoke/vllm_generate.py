#!/usr/bin/env python3
from __future__ import annotations

import argparse

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    params = SamplingParams(temperature=0.2, max_tokens=64)
    outputs = llm.generate(["Return the expression 1+1 inside <answer> tags."], params)
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()

