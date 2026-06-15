#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = """You are a physics research assistant solving CritPt benchmark problems.
Return only one Python code block. The code block must define the requested answer() function.
Do not include imports unless the template explicitly imports them. Do not include explanations outside code.
"""

RAW_COMPACT_SYSTEM_PROMPT = """You are a physics research assistant solving CritPt benchmark problems.
Return raw executable Python code only, with no Markdown fences and no explanations.
The code must define the requested answer() function.
Keep the solution compact: do not enumerate long lists or sets term by term when a formula,
helper function, comprehension, range, SymPy expression, or compact construction can represent it.
Never repeat the same term pattern many times. Prefer a short, complete, syntactically valid answer.
Do not include imports unless the template explicitly imports them.
"""


@dataclass
class CritPtPrompt:
    problem_id: str
    notebook_path: str
    prompt: str
    code_template: str


@dataclass
class Submission:
    problem_id: str
    generated_code: str
    model: str
    timestamp: str
    generation_config: dict[str, Any]
    messages: list[dict[str, str]]


def natural_challenge_key(path: Path) -> int:
    match = re.search(r"Challenge_(\d+)", path.stem)
    if not match:
        raise ValueError(f"cannot parse challenge index from {path}")
    return int(match.group(1))


def read_notebook_text(path: Path, prompt_style: str) -> CritPtPrompt:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    public_parts: list[str] = []
    code_template = ""
    seen_main = False

    for cell in cells:
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        stripped = source.strip()
        if not stripped:
            continue

        lowered = stripped.lower()
        header = stripped.splitlines()[0].strip().lower()
        if re.match(r"^#+\s*(answer|solution|expert solution)\b", header):
            continue
        if "test cases" in lowered or "testcases" in lowered:
            continue

        if cell_type == "markdown":
            if re.match(r"^#+\s*sub", header, flags=re.IGNORECASE):
                break
            if re.match(r"^#+\s*main problem", header, flags=re.IGNORECASE):
                seen_main = True
            public_parts.append(stripped)
            continue

        if cell_type == "code" and seen_main and not code_template:
            code_template = stripped
            public_parts.append("```python\n" + code_template + "\n```")

    if not seen_main:
        raise ValueError(f"main problem not found in {path}")
    if not code_template:
        raise ValueError(f"answer() code template not found in {path}")

    problem_id = f"{path.stem}_main"
    task_prompt = "\n\n".join(public_parts)
    if prompt_style == "raw-compact":
        prompt = (
            f"{task_prompt}\n\n"
            "Fill the template above. Respond with raw Python code only, no Markdown fences. "
            "The output must contain one complete answer() function. If the answer is a long list, "
            "set, polynomial, operator family, or symbolic expression, use compact Python/SymPy "
            "construction instead of expanding hundreds of repeated terms."
        )
    else:
        prompt = (
            f"{task_prompt}\n\n"
            "Fill the template above. Respond with exactly one Python code block containing the complete "
            "answer() function. Keep code compact; do not expand hundreds of repeated terms."
        )
    return CritPtPrompt(
        problem_id=problem_id,
        notebook_path=str(path),
        prompt=prompt,
        code_template=code_template,
    )


def load_prompts(challenges_dir: Path, limit: int | None, prompt_style: str) -> list[CritPtPrompt]:
    paths = sorted(challenges_dir.glob("Challenge_*.ipynb"), key=natural_challenge_key)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no Challenge_*.ipynb files found in {challenges_dir}")
    return [read_notebook_text(path, prompt_style) for path in paths]


def build_chat_prompt(tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str, enable_thinking: bool) -> str:
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


def write_submission_files(
    prompts: list[CritPtPrompt],
    outputs: list[str],
    out_dir: Path,
    model_name: str,
    generation_config: dict[str, Any],
    system_prompt: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    submissions = []
    for item, output in zip(prompts, outputs, strict=True):
        submission = Submission(
            problem_id=item.problem_id,
            generated_code=output,
            model=model_name,
            timestamp=datetime.now().isoformat(),
            generation_config=generation_config,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item.prompt},
                {"role": "assistant", "content": output},
            ],
        )
        submissions.append(submission)
        (out_dir / f"{item.problem_id}.json").write_text(
            json.dumps(asdict(submission), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    batch = {
        "submissions": [asdict(submission) for submission in submissions],
        "batch_metadata": {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "generation_config": generation_config,
            "num_submissions": len(submissions),
            "problem_ids": [submission.problem_id for submission in submissions],
        },
    }
    (out_dir / "submission_batch.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate official CritPt 70-main-challenge submissions with vLLM.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--challenges-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--prompt-style",
        choices=["code-block", "raw-compact"],
        default="code-block",
        help="code-block matches the original wrapper; raw-compact asks for raw compact Python to reduce runaway outputs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="only parse notebooks and write prompt_manifest.json")
    args = parser.parse_args()

    system_prompt = RAW_COMPACT_SYSTEM_PROMPT if args.prompt_style == "raw-compact" else SYSTEM_PROMPT

    prompts = load_prompts(args.challenges_dir, args.limit, args.prompt_style)
    if args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        manifest = [
            {
                "problem_id": item.problem_id,
                "notebook_path": item.notebook_path,
                "prompt_chars": len(item.prompt),
                "code_template_chars": len(item.code_template),
            }
            for item in prompts
        ]
        (args.out_dir / "prompt_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"num_prompts": len(prompts), "out_dir": str(args.out_dir)}, indent=2))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_prompts = [
        build_chat_prompt(tokenizer, system_prompt, item.prompt, args.enable_thinking)
        for item in prompts
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
    completions = [out.outputs[0].text for out in llm.generate(chat_prompts, sampling)]

    generation_config = {
        "benchmark": "CritPt",
        "reader_paths": str(args.challenges_dir),
        "run_main": True,
        "run_sub": False,
        "use_golden_for_prev_steps": False,
        "parsing": False,
        "multiturn_with_answer": False,
        "use_python": False,
        "use_web_search": False,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_thinking": args.enable_thinking,
        "prompt_style": args.prompt_style,
    }
    write_submission_files(
        prompts=prompts,
        outputs=completions,
        out_dir=args.out_dir,
        model_name=str(args.model),
        generation_config=generation_config,
        system_prompt=system_prompt,
    )
    print(
        json.dumps(
            {
                "num_submissions": len(completions),
                "out_dir": str(args.out_dir),
                "batch": str(args.out_dir / "submission_batch.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
