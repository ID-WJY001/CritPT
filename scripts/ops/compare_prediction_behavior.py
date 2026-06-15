#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


FINAL_ANSWER_RE = re.compile(
    r"(最终答案\s*[:：]|final\s+answer\s*[:：]|answer\s*[:：]|答案\s*[:：])",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(dict(json.loads(line)))
    return rows


def final_marker_stats(text: str) -> tuple[int, int]:
    last_match: re.Match[str] | None = None
    for match in FINAL_ANSWER_RE.finditer(text):
        last_match = match
    if last_match is None:
        return 0, 0
    return 1, len(text[last_match.end() :].strip())


def repetition_score(text: str) -> float:
    chunks = [chunk.strip() for chunk in re.split(r"[。；;\n]+", text) if chunk.strip()]
    if not chunks:
        return 0.0
    counts = Counter(chunks)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(chunks))


def numeric(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


def summarize(rows: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    records = []
    for row in rows:
        completion = str(row.get("completion", ""))
        marker, post_final_chars = final_marker_stats(completion)
        output_tokens = int(row.get("output_tokens") or 0)
        records.append(
            {
                "problem_id": str(row.get("problem_id", "")),
                "family": str(row.get("family", "")),
                "difficulty": str(row.get("difficulty", "")),
                "output_tokens": output_tokens,
                "output_chars": len(completion),
                "answer_marker_present": marker,
                "post_final_chars": post_final_chars,
                "hit_token_cap": 1 if output_tokens >= max_tokens else 0,
                "code_block_present": 1 if "```" in completion else 0,
                "line_count": len([line for line in completion.splitlines() if line.strip()]),
                "repetition_score": repetition_score(completion),
                "completion": completion,
            }
        )

    summary = {
        "n": len(records),
        "output_tokens": numeric([float(item["output_tokens"]) for item in records]),
        "output_chars": numeric([float(item["output_chars"]) for item in records]),
        "answer_marker_present": numeric([float(item["answer_marker_present"]) for item in records]),
        "post_final_chars": numeric([float(item["post_final_chars"]) for item in records]),
        "hit_token_cap": numeric([float(item["hit_token_cap"]) for item in records]),
        "code_block_present": numeric([float(item["code_block_present"]) for item in records]),
        "line_count": numeric([float(item["line_count"]) for item in records]),
        "repetition_score": numeric([float(item["repetition_score"]) for item in records]),
        "by_family": {},
        "worst_long": sorted(
            [
                {
                    "problem_id": item["problem_id"],
                    "family": item["family"],
                    "difficulty": item["difficulty"],
                    "output_tokens": item["output_tokens"],
                    "output_chars": item["output_chars"],
                    "marker": item["answer_marker_present"],
                    "hit_token_cap": item["hit_token_cap"],
                }
                for item in records
            ],
            key=lambda item: (item["hit_token_cap"], item["output_tokens"], item["output_chars"]),
            reverse=True,
        )[:12],
    }
    families = sorted({item["family"] for item in records})
    for family in families:
        subset = [item for item in records if item["family"] == family]
        summary["by_family"][family] = {
            "n": len(subset),
            "tokens_mean": statistics.mean(float(item["output_tokens"]) for item in subset),
            "chars_mean": statistics.mean(float(item["output_chars"]) for item in subset),
            "marker_mean": statistics.mean(float(item["answer_marker_present"]) for item in subset),
            "hit_token_cap_mean": statistics.mean(float(item["hit_token_cap"]) for item in subset),
        }
    return summary


def pairwise(base_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    base_by_id = {str(row["problem_id"]): row for row in base_rows}
    cand_by_id = {str(row["problem_id"]): row for row in candidate_rows}
    common = sorted(set(base_by_id) & set(cand_by_id))
    deltas = []
    for problem_id in common:
        base = base_by_id[problem_id]
        cand = cand_by_id[problem_id]
        base_text = str(base.get("completion", ""))
        cand_text = str(cand.get("completion", ""))
        base_marker, _ = final_marker_stats(base_text)
        cand_marker, _ = final_marker_stats(cand_text)
        base_tokens = int(base.get("output_tokens") or 0)
        cand_tokens = int(cand.get("output_tokens") or 0)
        deltas.append(
            {
                "problem_id": problem_id,
                "family": str(base.get("family", "")),
                "difficulty": str(base.get("difficulty", "")),
                "base_tokens": base_tokens,
                "candidate_tokens": cand_tokens,
                "delta_tokens": cand_tokens - base_tokens,
                "base_chars": len(base_text),
                "candidate_chars": len(cand_text),
                "delta_chars": len(cand_text) - len(base_text),
                "base_marker": base_marker,
                "candidate_marker": cand_marker,
                "base_hit_cap": 1 if base_tokens >= max_tokens else 0,
                "candidate_hit_cap": 1 if cand_tokens >= max_tokens else 0,
            }
        )
    return {
        "n_common": len(common),
        "delta_tokens": numeric([float(item["delta_tokens"]) for item in deltas]),
        "delta_chars": numeric([float(item["delta_chars"]) for item in deltas]),
        "marker_improved": sum(
            1 for item in deltas if item["candidate_marker"] > item["base_marker"]
        ),
        "marker_worsened": sum(
            1 for item in deltas if item["candidate_marker"] < item["base_marker"]
        ),
        "cap_improved": sum(
            1 for item in deltas if item["candidate_hit_cap"] < item["base_hit_cap"]
        ),
        "cap_worsened": sum(
            1 for item in deltas if item["candidate_hit_cap"] > item["base_hit_cap"]
        ),
        "largest_length_increases": sorted(deltas, key=lambda item: item["delta_tokens"], reverse=True)[:12],
        "largest_length_decreases": sorted(deltas, key=lambda item: item["delta_tokens"])[:12],
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    base = payload["base"]
    cand = payload["candidate"]
    pair = payload["pairwise"]
    lines = [
        "# Prediction Behavior Compare",
        "",
        "This file compares generation behavior only. It does not prove answer correctness.",
        "",
        "## Summary",
        "",
        "| metric | base | candidate |",
        "| --- | ---: | ---: |",
        f"| n | {base['n']} | {cand['n']} |",
        f"| output_tokens.mean | {base['output_tokens']['mean']:.1f} | {cand['output_tokens']['mean']:.1f} |",
        f"| output_chars.mean | {base['output_chars']['mean']:.1f} | {cand['output_chars']['mean']:.1f} |",
        f"| answer_marker_present.mean | {base['answer_marker_present']['mean']:.4f} | {cand['answer_marker_present']['mean']:.4f} |",
        f"| hit_token_cap.mean | {base['hit_token_cap']['mean']:.4f} | {cand['hit_token_cap']['mean']:.4f} |",
        f"| post_final_chars.mean | {base['post_final_chars']['mean']:.1f} | {cand['post_final_chars']['mean']:.1f} |",
        f"| line_count.mean | {base['line_count']['mean']:.1f} | {cand['line_count']['mean']:.1f} |",
        f"| repetition_score.mean | {base['repetition_score']['mean']:.4f} | {cand['repetition_score']['mean']:.4f} |",
        "",
        "## Pairwise",
        "",
        f"- common examples: `{pair['n_common']}`",
        f"- delta_tokens.mean: `{pair['delta_tokens']['mean']:.1f}`",
        f"- marker improved/worsened: `{pair['marker_improved']}` / `{pair['marker_worsened']}`",
        f"- cap improved/worsened: `{pair['cap_improved']}` / `{pair['cap_worsened']}`",
        "",
        "## By Family",
        "",
        "| family | base n | base marker | base cap | cand marker | cand cap | base tokens | cand tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    families = sorted(set(base["by_family"]) | set(cand["by_family"]))
    for family in families:
        b = base["by_family"].get(family, {})
        c = cand["by_family"].get(family, {})
        lines.append(
            "| {family} | {bn} | {bm:.4f} | {bc:.4f} | {cm:.4f} | {cc:.4f} | {bt:.1f} | {ct:.1f} |".format(
                family=family,
                bn=b.get("n", 0),
                bm=float(b.get("marker_mean", 0.0)),
                bc=float(b.get("hit_token_cap_mean", 0.0)),
                cm=float(c.get("marker_mean", 0.0)),
                cc=float(c.get("hit_token_cap_mean", 0.0)),
                bt=float(b.get("tokens_mean", 0.0)),
                ct=float(c.get("tokens_mean", 0.0)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base_predictions)
    candidate_rows = read_jsonl(args.candidate_predictions)
    payload = {
        "base_predictions": str(args.base_predictions),
        "candidate_predictions": str(args.candidate_predictions),
        "max_tokens": args.max_tokens,
        "base": summarize(base_rows, args.max_tokens),
        "candidate": summarize(candidate_rows, args.max_tokens),
        "pairwise": pairwise(base_rows, candidate_rows, args.max_tokens),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, args.out_md)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
