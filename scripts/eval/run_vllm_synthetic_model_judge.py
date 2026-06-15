#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.model_judge.verl_reward_lenaware import compute_score


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
        "最后必须单独写一行“最终答案：...”，只给出清晰可判定的结果。"
    )


def prompt_for_style(example: SyntheticCritPTExample, prompt_style: str) -> str:
    if prompt_style == "cot":
        return cot_prompt(example.prompt)
    if prompt_style == "compact_cot":
        return compact_cot_prompt(example.prompt)
    if prompt_style == "audit_cot":
        return audit_cot_prompt(example.prompt)
    return naturalize_prompt(example.prompt)


def cot_policy(prompt_style: str) -> str:
    if prompt_style == "cot":
        return "reasoning_allowed_reference_cot"
    if prompt_style == "compact_cot":
        return "compact_reasoning_required_final_answer"
    if prompt_style == "audit_cot":
        return "compact_auditable_reasoning_required_final_answer"
    return "reasoning_allowed_reference_trace"


def naturalized_reference(example: SyntheticCritPTExample) -> str:
    return (
        "Reference reasoning notes:\n"
        f"{example.solution_trace.strip()}\n\n"
        "Reference executable answer:\n"
        f"{example.target_code.strip()}"
    ).strip()


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


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(dict(json.loads(line)))
    return rows


def build_chat_prompt(
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    enable_thinking: bool,
) -> str:
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


@dataclass
class EvalRecord:
    problem_id: str
    family: str
    difficulty: str
    prompt: str
    completion: str
    model: str
    output_tokens: int
    output_chars: int
    score: float
    raw_judge_score: float
    acc: float
    judge_error: float
    quick_reject: float
    answer_marker_present: float
    no_final_cap_applied: float
    length_penalty: float
    correctness: float
    instruction_following: float
    reasoning_quality: float
    final_answer_consistency: float
    fatal_error: float
    reason: str
    metadata: dict[str, Any]


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def judge_one(
    *,
    example: SyntheticCritPTExample,
    prompt_text: str,
    completion: str,
    model_name: str,
    output_tokens: int,
    prompt_style: str,
) -> EvalRecord:
    extra_info = {
        "split": example.split,
        "problem_id": example.problem_id,
        "family": example.family,
        "difficulty": example.difficulty,
        "prompt_text": prompt_text,
        "reference_answer": naturalized_reference(example),
        "reference_trace": example.solution_trace,
        "cot_policy": cot_policy(prompt_style),
        "rubric": (
            "Judge scientific/mathematical correctness first. "
            "Useful reasoning is allowed and should be rewarded when grounded in the problem. "
            "Require a clear final answer. "
            "For compact_cot prompts, also reward concise reasoning that reaches the final answer "
            "without repetitive interpretation or unfinished exploration. "
            "For audit_cot prompts, penalize skipped calculations, unsupported 'therefore' jumps, "
            "and final answers that are guessed after omitting recurrence, series, or enumeration steps."
        ),
        "metadata": json.dumps(example.metadata, ensure_ascii=False),
    }
    judged = compute_score(
        data_source="critpt_model_judge_eval",
        solution_str=completion,
        ground_truth=example.target_code,
        extra_info=extra_info,
    )
    return EvalRecord(
        problem_id=example.problem_id,
        family=example.family,
        difficulty=example.difficulty,
        prompt=prompt_text,
        completion=completion,
        model=model_name,
        output_tokens=output_tokens,
        output_chars=len(completion),
        score=finite_float(judged.get("score")),
        raw_judge_score=finite_float(judged.get("raw_judge_score"), finite_float(judged.get("score"))),
        acc=finite_float(judged.get("acc")),
        judge_error=finite_float(judged.get("judge_error")),
        quick_reject=finite_float(judged.get("quick_reject")),
        answer_marker_present=finite_float(judged.get("answer_marker_present")),
        no_final_cap_applied=finite_float(judged.get("no_final_cap_applied")),
        length_penalty=finite_float(judged.get("length_penalty")),
        correctness=finite_float(judged.get("correctness")),
        instruction_following=finite_float(judged.get("instruction_following")),
        reasoning_quality=finite_float(judged.get("reasoning_quality")),
        final_answer_consistency=finite_float(judged.get("final_answer_consistency")),
        fatal_error=finite_float(judged.get("fatal_error")),
        reason=str(judged.get("reason", "")),
        metadata=dict(example.metadata),
    )


def numeric_summary(records: list[EvalRecord]) -> dict[str, dict[str, float | int]]:
    keys = [
        "score",
        "raw_judge_score",
        "acc",
        "judge_error",
        "quick_reject",
        "answer_marker_present",
        "no_final_cap_applied",
        "length_penalty",
        "correctness",
        "instruction_following",
        "reasoning_quality",
        "final_answer_consistency",
        "fatal_error",
        "output_tokens",
        "output_chars",
    ]
    out: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = [float(getattr(record, key)) for record in records]
        if not values:
            continue
        ordered = sorted(values)
        out[key] = {
            "n": len(values),
            "mean": statistics.mean(values),
            "min": ordered[0],
            "p50": ordered[len(ordered) // 2],
            "max": ordered[-1],
        }
    return out


def grouped_summary(records: list[EvalRecord], attr: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[EvalRecord]] = defaultdict(list)
    for record in records:
        grouped[str(getattr(record, attr))].append(record)
    return {
        key: {
            "n": len(items),
            "score_mean": statistics.mean(item.score for item in items),
            "acc_mean": statistics.mean(item.acc for item in items),
            "marker_mean": statistics.mean(item.answer_marker_present for item in items),
            "clip_like_over_1900_chars": statistics.mean(1.0 if item.output_chars >= 1900 else 0.0 for item in items),
            "judge_error_mean": statistics.mean(item.judge_error for item in items),
        }
        for key, items in sorted(grouped.items())
    }


def reason_keywords(records: list[EvalRecord]) -> dict[str, int]:
    keywords = [
        "truncated",
        "incomplete",
        "no final",
        "never computes",
        "wrong",
        "incorrect",
        "format",
        "rambling",
        "too long",
        "overlong",
    ]
    counts: Counter[str] = Counter()
    for record in records:
        reason = record.reason.lower()
        for keyword in keywords:
            if keyword in reason:
                counts[keyword] += 1
    return dict(counts)


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def write_review(records: list[EvalRecord], path: Path, text_limit: int) -> None:
    worst = sorted(records, key=lambda item: item.score)[:8]
    best = sorted(records, key=lambda item: item.score, reverse=True)[:8]
    lines = [
        "# Synthetic Model-Judge Eval Review",
        "",
        "## Summary",
        "",
        f"- rows: `{len(records)}`",
        f"- score.mean: `{statistics.mean(r.score for r in records):.4f}`",
        f"- acc.mean: `{statistics.mean(r.acc for r in records):.4f}`",
        f"- answer_marker_present.mean: `{statistics.mean(r.answer_marker_present for r in records):.4f}`",
        f"- judge_error.mean: `{statistics.mean(r.judge_error for r in records):.4f}`",
        "",
        "## Worst Cases",
        "",
    ]
    for idx, record in enumerate(worst, start=1):
        lines += [
            f"### Worst {idx}: {record.problem_id}",
            "",
            f"- family: `{record.family}`",
            f"- difficulty: `{record.difficulty}`",
            f"- score: `{record.score:.4f}`",
            f"- raw_judge_score: `{record.raw_judge_score:.4f}`",
            f"- acc: `{record.acc:.0f}`",
            f"- marker: `{record.answer_marker_present:.0f}`",
            f"- output_chars: `{record.output_chars}`",
            "",
            "**Prompt**",
            "",
            "```text",
            clip(record.prompt, text_limit),
            "```",
            "",
            "**Completion**",
            "",
            "```text",
            clip(record.completion, text_limit),
            "```",
            "",
            "**Judge Reason**",
            "",
            "```text",
            clip(record.reason, text_limit),
            "```",
            "",
        ]
    lines += ["## Best Cases", ""]
    for idx, record in enumerate(best, start=1):
        lines += [
            f"### Best {idx}: {record.problem_id}",
            "",
            f"- family: `{record.family}`",
            f"- difficulty: `{record.difficulty}`",
            f"- score: `{record.score:.4f}`",
            f"- raw_judge_score: `{record.raw_judge_score:.4f}`",
            f"- acc: `{record.acc:.0f}`",
            f"- marker: `{record.answer_marker_present:.0f}`",
            f"- output_chars: `{record.output_chars}`",
            "",
            "**Completion**",
            "",
            "```text",
            clip(record.completion, text_limit),
            "```",
            "",
            "**Judge Reason**",
            "",
            "```text",
            clip(record.reason, text_limit),
            "```",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM synthetic CritPT model-judge eval.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        default=None,
        help="Reuse existing predictions instead of running vLLM generation.",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--prompt-style",
        choices=["natural", "cot", "compact_cot", "audit_cot"],
        default="compact_cot",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument("--review-text-limit", type=int, default=1800)
    args = parser.parse_args()

    examples = read_examples(args.data, args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.run_name or args.model

    if args.predictions_jsonl is not None:
        prediction_rows = read_prediction_rows(args.predictions_jsonl)
        by_id = {example.problem_id: example for example in examples}
        filtered_examples: list[SyntheticCritPTExample] = []
        prompt_texts: list[str] = []
        completions: list[str] = []
        output_tokens: list[int] = []
        for row in prediction_rows:
            problem_id = str(row["problem_id"])
            if problem_id not in by_id:
                raise ValueError(f"prediction problem_id not found in data: {problem_id}")
            filtered_examples.append(by_id[problem_id])
            prompt_texts.append(str(row.get("prompt") or prompt_for_style(by_id[problem_id], args.prompt_style)))
            completions.append(str(row.get("completion", "")).strip())
            output_tokens.append(int(row.get("output_tokens") or 0))
        examples = filtered_examples
        predictions_path = args.out_dir / "predictions.jsonl"
        if args.predictions_jsonl.resolve() != predictions_path.resolve():
            with predictions_path.open("w", encoding="utf-8") as handle:
                for row in prediction_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
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
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        outputs = llm.generate(chat_prompts, sampling)
        completions = [output.outputs[0].text.strip() for output in outputs]
        output_tokens = [len(output.outputs[0].token_ids or []) for output in outputs]

        predictions_path = args.out_dir / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for example, prompt_text, completion, token_count in zip(
                examples, prompt_texts, completions, output_tokens, strict=True
            ):
                handle.write(
                    json.dumps(
                        {
                            "problem_id": example.problem_id,
                            "family": example.family,
                            "difficulty": example.difficulty,
                            "prompt": prompt_text,
                            "completion": completion,
                            "model": model_name,
                            "output_tokens": token_count,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    records: list[EvalRecord] = []
    with ThreadPoolExecutor(max_workers=max(1, args.judge_workers)) as executor:
        futures = [
            executor.submit(
                judge_one,
                example=example,
                prompt_text=prompt_text,
                completion=completion,
                model_name=model_name,
                output_tokens=token_count,
                prompt_style=args.prompt_style,
            )
            for example, prompt_text, completion, token_count in zip(
                examples, prompt_texts, completions, output_tokens, strict=True
            )
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(
                json.dumps(
                    {
                        "judged": idx,
                        "total": len(futures),
                        "problem_id": record.problem_id,
                        "score": record.score,
                        "acc": record.acc,
                        "judge_error": record.judge_error,
                    },
                    ensure_ascii=False,
                )
            )

    records.sort(key=lambda item: item.problem_id)
    judged_path = args.out_dir / "judged.jsonl"
    with judged_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "run_name": model_name,
        "model": args.model,
        "data": str(args.data),
        "created_at": datetime.now().isoformat(),
        "prompt_style": args.prompt_style,
        "generation_config": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "enable_thinking": args.enable_thinking,
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
        "num_examples": len(records),
        "numeric": numeric_summary(records),
        "by_family": grouped_summary(records, "family"),
        "by_difficulty": grouped_summary(records, "difficulty"),
        "reason_keyword_counts": reason_keywords(records),
        "worst": [
            {
                "problem_id": item.problem_id,
                "family": item.family,
                "difficulty": item.difficulty,
                "score": item.score,
                "reason": item.reason,
            }
            for item in sorted(records, key=lambda record: record.score)[:12]
        ],
        "best": [
            {
                "problem_id": item.problem_id,
                "family": item.family,
                "difficulty": item.difficulty,
                "score": item.score,
                "reason": item.reason,
            }
            for item in sorted(records, key=lambda record: record.score, reverse=True)[:12]
        ],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_review(records, args.out_dir / "review.md", args.review_text_limit)
    print(json.dumps({"summary": str(args.out_dir / "summary.json"), "records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
