#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


KEY_METRICS = [
    ("critic/score/mean", "Score Mean", "reward score"),
    ("critic/score/max", "Score Max", "reward score"),
    ("critic/score/min", "Score Min", "reward score"),
    ("actor/grad_norm", "Actor Grad Norm", "gradient norm"),
    ("actor/loss", "Actor Loss", "loss"),
    ("critic/advantages/max", "Advantage Max", "advantage"),
    ("critic/advantages/min", "Advantage Min", "advantage"),
    ("response_length/mean", "Response Length Mean", "tokens"),
    ("response_length/clip_ratio", "Response Clip Ratio", "ratio"),
    ("actor/entropy", "Actor Entropy", "entropy"),
    ("perf/throughput", "Throughput", "tokens/sec"),
]

ROLLOUT_NUMERIC_KEYS = [
    "score",
    "acc",
    "format_reward",
    "has_code_block",
    "single_code_block",
    "has_answer_def",
    "single_answer_func",
    "no_think_tags",
    "format_ok",
    "compile_ok",
    "exec_ok",
    "passed_checks",
    "total_checks",
]


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isfinite(number):
        return number
    return default


def read_metrics(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            row: dict[str, float] = {"step": float(raw.get("step", len(rows) + 1))}
            data = raw.get("data", {})
            for key, value in data.items():
                number = finite_float(value)
                if number is not None:
                    row[str(key)] = number
            rows.append(row)
    return rows


def rollout_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**9, path.name


def read_rollouts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_path in sorted(path.glob("*.jsonl"), key=rollout_sort_key):
        with file_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row.setdefault("step", int(file_path.stem))
                row["_file"] = file_path.name
                row["_line"] = line_no
                rows.append(row)
    return rows


def grouped_by_step(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        step = int(finite_float(row.get("step"), -1) or -1)
        grouped[step].append(row)
    return grouped


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def score(row: dict[str, Any]) -> float:
    return finite_float(row.get("score"), 0.0) or 0.0


def acc(row: dict[str, Any]) -> float:
    return finite_float(row.get("acc"), 0.0) or 0.0


def prompt_key(row: dict[str, Any]) -> str:
    # The rollout dump does not include problem_id, so group by full prompt text.
    return str(row.get("input", ""))


def family_guess(prompt: str) -> str:
    text = prompt.lower()
    if "ordinary generating function" in text or "recurrence" in text:
        return "recurrence_generating_function"
    if "holographic anomaly" in text or "trace basis" in text:
        return "holographic_coefficients"
    if "lamet" in text or "piecewise kernel" in text:
        return "lamet_piecewise"
    if "amplitude damping" in text:
        return "amplitude_damping"
    if "quantum fisher" in text or "qfi" in text:
        return "qfi_symbolic"
    if "galerkin" in text or "rayleigh" in text:
        return "convection_minimum"
    if "gauge theory" in text or "operator list" in text:
        return "operator_enumeration"
    if "high-harmonic" in text or "oam" in text:
        return "hhg_oam"
    return "unknown"


def failure_bucket(row: dict[str, Any]) -> str:
    if acc(row) >= 1.0:
        if finite_float(row.get("no_think_tags"), 0.0) == 1.0:
            return "pass_clean"
        return "pass_with_empty_think_tags"
    reason = str(row.get("reason", ""))
    if "symbolic" in reason or "subs=" in reason:
        return "wrong_symbolic_expression"
    if "tol=" in reason or "got=" in reason:
        return "wrong_numeric_or_sequence_value"
    if "length" in reason:
        return "wrong_return_length"
    if "syntax_error" in reason:
        return "syntax_error"
    if "runtime_error" in reason:
        return "runtime_error"
    if "format_error" in reason:
        return "format_error"
    return "other_failed_check"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_rollout_step_csv(path: Path, rollout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step, rows in sorted(grouped_by_step(rollout_rows).items()):
        record: dict[str, Any] = {"step": step, "count": len(rows)}
        for key in ROLLOUT_NUMERIC_KEYS:
            values = [number for row in rows if (number := finite_float(row.get(key))) is not None]
            if values:
                record[f"{key}_mean"] = mean(values)
                record[f"{key}_min"] = min(values)
                record[f"{key}_max"] = max(values)
        record["score_variance_prompt_groups"] = prompt_group_variance_count(rows)
        record["mean_output_chars"] = mean([float(len(str(row.get("output", "")))) for row in rows])
        out.append(record)

    fields = sorted({key for row in out for key in row})
    fields = ["step", "count", *[field for field in fields if field not in {"step", "count"}]]
    write_csv(path, out, fields)
    return out


def prompt_group_variance_count(rows: list[dict[str, Any]]) -> int:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[prompt_key(row)].append(score(row))
    return sum(1 for scores in grouped.values() if len(set(round(value, 6) for value in scores)) > 1)


def metrics_summary(metrics_rows: list[dict[str, float]]) -> dict[str, Any]:
    nonzero_grad = [int(row["step"]) for row in metrics_rows if abs(row.get("actor/grad_norm", 0.0)) > 1e-9]
    nonzero_adv = [
        int(row["step"])
        for row in metrics_rows
        if abs(row.get("critic/advantages/max", 0.0)) > 1e-9
        or abs(row.get("critic/advantages/min", 0.0)) > 1e-9
    ]
    first5 = metrics_rows[:5]
    last5 = metrics_rows[-5:]
    return {
        "steps": len(metrics_rows),
        "first_step": int(metrics_rows[0]["step"]) if metrics_rows else None,
        "last_step": int(metrics_rows[-1]["step"]) if metrics_rows else None,
        "nonzero_grad_steps": nonzero_grad,
        "nonzero_grad_step_count": len(nonzero_grad),
        "nonzero_adv_step_count": len(nonzero_adv),
        "score_mean_first5": mean([row.get("critic/score/mean", 0.0) for row in first5]),
        "score_mean_last5": mean([row.get("critic/score/mean", 0.0) for row in last5]),
        "score_mean_all": mean([row.get("critic/score/mean", 0.0) for row in metrics_rows]),
        "response_len_mean_all": mean([row.get("response_length/mean", 0.0) for row in metrics_rows]),
        "clip_ratio_max": max([row.get("response_length/clip_ratio", 0.0) for row in metrics_rows], default=0.0),
        "last": {
            key: metrics_rows[-1].get(key)
            for key in [
                "critic/score/mean",
                "critic/score/max",
                "critic/score/min",
                "actor/grad_norm",
                "actor/loss",
                "critic/advantages/max",
                "critic/advantages/min",
                "response_length/mean",
                "response_length/clip_ratio",
            ]
        }
        if metrics_rows
        else {},
    }


def rollout_summary(rollout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = Counter(failure_bucket(row) for row in rollout_rows)
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        families[family_guess(str(row.get("input", "")))].append(row)
    by_family = {}
    for family, rows in sorted(families.items()):
        by_family[family] = {
            "n": len(rows),
            "score_mean": mean([score(row) for row in rows]),
            "acc_mean": mean([acc(row) for row in rows]),
            "failures": dict(Counter(failure_bucket(row) for row in rows)),
        }
    return {
        "rows": len(rollout_rows),
        "steps": len(set(int(finite_float(row.get("step"), -1) or -1) for row in rollout_rows)),
        "score_mean": mean([score(row) for row in rollout_rows]),
        "acc_mean": mean([acc(row) for row in rollout_rows]),
        "format_reward_mean": mean([finite_float(row.get("format_reward"), 0.0) or 0.0 for row in rollout_rows]),
        "has_code_block_mean": mean([finite_float(row.get("has_code_block"), 0.0) or 0.0 for row in rollout_rows]),
        "no_think_tags_mean": mean([finite_float(row.get("no_think_tags"), 0.0) or 0.0 for row in rollout_rows]),
        "format_ok_mean": mean([finite_float(row.get("format_ok"), 0.0) or 0.0 for row in rollout_rows]),
        "compile_ok_mean": mean([finite_float(row.get("compile_ok"), 0.0) or 0.0 for row in rollout_rows]),
        "exec_ok_mean": mean([finite_float(row.get("exec_ok"), 0.0) or 0.0 for row in rollout_rows]),
        "failure_buckets": dict(buckets),
        "by_family": by_family,
    }


def scale(values: list[tuple[float, float]], width: int, height: int, pads: tuple[int, int, int, int]) -> tuple[str, tuple[float, float, float, float]]:
    left, right, top, bottom = pads
    if not values:
        return "", (0.0, 1.0, 0.0, 1.0)
    xs = [item[0] for item in values]
    ys = [item[1] for item in values]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmax += 1
    if ymin == ymax:
        delta = 0.5 if ymin == 0 else abs(ymin) * 0.1
        ymin -= delta
        ymax += delta
    points = []
    for x, y in values:
        sx = left + (x - xmin) / (xmax - xmin) * (width - left - right)
        sy = height - bottom - (y - ymin) / (ymax - ymin) * (height - top - bottom)
        points.append(f"{sx:.1f},{sy:.1f}")
    return " ".join(points), (xmin, xmax, ymin, ymax)


def tick_values(lo: float, hi: float, count: int) -> list[float]:
    if count <= 1:
        return [lo]
    return [lo + (hi - lo) * idx / (count - 1) for idx in range(count)]


def fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def write_svg(path: Path, metrics_rows: list[dict[str, float]]) -> None:
    width = 920
    panel_h = 245
    pads = (92, 34, 50, 62)
    total_h = panel_h * len(KEY_METRICS)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_h}" font-family="Arial, sans-serif">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for idx, (key, title, ylabel) in enumerate(KEY_METRICS):
        y_offset = idx * panel_h
        values = [(row["step"], row[key]) for row in metrics_rows if key in row]
        polyline, bounds = scale(values, width, panel_h, pads)
        xmin, xmax, ymin, ymax = bounds
        left, right, top, bottom = pads
        plot_w = width - left - right
        plot_h = panel_h - top - bottom
        parts.append(f'<g transform="translate(0,{y_offset})">')
        parts.append(f'<text x="{left}" y="26" font-size="15" font-weight="700">{html.escape(title)}</text>')
        parts.append(
            f'<text x="{width - right}" y="26" text-anchor="end" font-size="12" fill="#555">'
            f'{html.escape(key)}; last={fmt(values[-1][1]) if values else "NA"}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfbfb" stroke="#ddd"/>'
        )
        x_axis_y = panel_h - bottom
        parts.append(f'<line x1="{left}" y1="{x_axis_y}" x2="{width-right}" y2="{x_axis_y}" stroke="#666"/>')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{x_axis_y}" stroke="#666"/>')
        for tick in tick_values(xmin, xmax, 6):
            x = left + (tick - xmin) / (xmax - xmin) * plot_w
            parts.append(f'<line x1="{x:.1f}" y1="{x_axis_y}" x2="{x:.1f}" y2="{x_axis_y + 5}" stroke="#666"/>')
            parts.append(
                f'<text x="{x:.1f}" y="{x_axis_y + 22}" text-anchor="middle" font-size="11" fill="#444">{html.escape(fmt(tick))}</text>'
            )
        for tick in tick_values(ymin, ymax, 5):
            y = panel_h - bottom - (tick - ymin) / (ymax - ymin) * plot_h
            parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#666"/>')
            parts.append(
                f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#444">{html.escape(fmt(tick))}</text>'
            )
        parts.append(
            f'<text x="{left + plot_w / 2:.1f}" y="{panel_h - 18}" text-anchor="middle" font-size="12" fill="#333">'
            'x-axis: training step</text>'
        )
        parts.append(
            f'<text x="18" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="12" fill="#333" '
            f'transform="rotate(-90 18 {top + plot_h / 2:.1f})">y-axis: {html.escape(ylabel)}</text>'
        )
        if values:
            parts.append(f'<polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2.2"/>')
            for x, y in values:
                if int(x) in {1, 10, 20, 30, 40, 50}:
                    sx = left + (x - xmin) / (xmax - xmin) * plot_w
                    sy = panel_h - bottom - (y - ymin) / (ymax - ymin) * plot_h
                    parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="2.5" fill="#1d4ed8"/>')
        parts.append("</g>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def clip(text: str, limit: int = 1300) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]..."


def write_examples(path: Path, rollout_rows: list[dict[str, Any]]) -> None:
    worst = sorted(rollout_rows, key=lambda row: (score(row), row.get("step", 0)))[:8]
    best = sorted(rollout_rows, key=lambda row: (score(row), row.get("step", 0)), reverse=True)[:6]
    lines = ["# V13 n4 Rollout 示例", ""]
    for title, rows in [("低分样本", worst), ("高分样本", best)]:
        lines.extend([f"## {title}", ""])
        for idx, row in enumerate(rows, start=1):
            lines.extend(
                [
                    f"### {idx}. step {row.get('step')} score={score(row):.4f} acc={acc(row):.1f}",
                    "",
                    f"- bucket: `{failure_bucket(row)}`",
                    f"- reason: `{str(row.get('reason', '')).replace('`', '')}`",
                    f"- file: `{row.get('_file')}` line `{row.get('_line')}`",
                    "",
                    "```text",
                    clip(str(row.get("output", ""))),
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_report(
    path: Path,
    run_name: str,
    metrics: dict[str, Any],
    rollout: dict[str, Any],
    out_dir: Path,
) -> None:
    nonzero = metrics["nonzero_grad_steps"]
    lines = [
        f"# {run_name} 曲线与 Rollout 分析",
        "",
        "更新时间：2026-06-08",
        "",
        "## 一句话结论",
        "",
        "这轮不是 infra smoke 了，确实跑出了 GRPO 更新：50 step 中有 17 step 的 `actor/grad_norm` 非零。",
        "但是训练质量还不能说成功，因为总体 reward/acc 没有上升趋势，后半段反而低于前 5 step；当前主要价值是定位出下一轮应做 hard-case 采样和 reward 细化。",
        "",
        "## 产物位置",
        "",
        f"- 指标图：`{out_dir / 'metrics_key.svg'}`",
        f"- 指标 CSV：`{out_dir / 'metrics_key.csv'}`",
        f"- rollout step CSV：`{out_dir / 'rollout_step_summary.v13.csv'}`",
        f"- rollout 示例：`{out_dir / 'rollout_examples.zh-CN.md'}`",
        f"- 完整 rollout JSONL：`{out_dir / 'rollouts.full.jsonl'}`",
        "",
        "## 曲线",
        "",
        "图里每个小图的横轴都明确是 `training step`，纵轴是该小图标题对应的指标值。",
        "",
        "![V13 n4 key metrics](../../artifacts/experiments/qwen3_8b_grpo_v13_official_code_format_signal_n4/metrics_key.svg)",
        "",
        "## 指标摘要",
        "",
        f"- step 数：`{metrics['steps']}`",
        f"- 有真实梯度的 step：`{metrics['nonzero_grad_step_count']}/50`",
        f"- 有真实 advantage 的 step：`{metrics['nonzero_adv_step_count']}/50`",
        f"- 有梯度的 step 列表：`{nonzero}`",
        f"- 前 5 step score mean：`{metrics['score_mean_first5']:.4f}`",
        f"- 后 5 step score mean：`{metrics['score_mean_last5']:.4f}`",
        f"- 全程 score mean：`{metrics['score_mean_all']:.4f}`",
        f"- 全程 response length mean：`{metrics['response_len_mean_all']:.2f}` tokens",
        f"- 最大 response clip ratio：`{metrics['clip_ratio_max']:.4f}`",
        "",
        "最后一步：",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in metrics["last"].items():
        lines.append(f"| `{key}` | `{float(value):.6g}` |")
    lines.extend(
        [
            "",
            "## Rollout 全局统计",
            "",
            f"- rollout 总数：`{rollout['rows']}`",
            f"- rollout step 数：`{rollout['steps']}`",
            f"- rollout score mean：`{rollout['score_mean']:.4f}`",
            f"- rollout acc mean：`{rollout['acc_mean']:.4f}`",
            f"- format reward mean：`{rollout['format_reward_mean']:.4f}`",
            f"- has code block mean：`{rollout['has_code_block_mean']:.4f}`",
            f"- no think tags mean：`{rollout['no_think_tags_mean']:.4f}`",
            f"- format/compile/exec ok mean：`{rollout['format_ok_mean']:.4f}` / `{rollout['compile_ok_mean']:.4f}` / `{rollout['exec_ok_mean']:.4f}`",
            "",
            "失败/通过桶：",
            "",
            "| bucket | count |",
            "| --- | ---: |",
        ]
    )
    for bucket, count in sorted(rollout["failure_buckets"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{bucket}` | {count} |")
    lines.extend(["", "## 按题型粗分", "", "| family | n | score mean | acc mean | main failures |", "| --- | ---: | ---: | ---: | --- |"])
    for family, info in rollout["by_family"].items():
        failures = ", ".join(f"{key}:{value}" for key, value in sorted(info["failures"].items(), key=lambda item: (-item[1], item[0]))[:4])
        lines.append(
            f"| `{family}` | {info['n']} | {info['score_mean']:.4f} | {info['acc_mean']:.4f} | `{failures}` |"
        )
    lines.extend(
        [
            "",
            "## 我怎么看这些曲线",
            "",
            "1. `score/acc` 没有上升：前 5 step score mean 约 0.885，后 5 step 约 0.803，不能宣称训练改善了任务正确率。",
            "2. `grad_norm` 间歇性非零：17/50 step 有更新，说明 n=4 确实比 n=2 好；但很多 step 仍然组内同分，GRPO 没有信号。",
            "3. `response_length/clip_ratio` 基本健康：平均输出约 160 token，最大 clip ratio 0.125，说明这轮不是 max token 卡死。",
            "4. `has_code_block/format_ok/compile_ok/exec_ok` 已经很高：格式问题大幅缓解，模型基本都会交一个可执行 `answer()`。",
            "5. `no_think_tags` 仍是 0：模型几乎总是输出空 `<think></think>`，这会污染官方交卷格式；虽然目前 verifier 能抽出 code block，但这不是最终形态。",
            "",
            "## 关键判断",
            "",
            "这轮证明了三件事：",
            "",
            "- 8 卡 A100 + verl GRPO + vLLM rollout + 本地 verifier reward + checkpoint 全链路可用。",
            "- `max_response=1536` 不是瓶颈，512 的截断问题已经解决。",
            "- 当前数据太容易或太稳定，导致很多 prompt 的 4 个 rollout 同分；这会让 GRPO 的组内 advantage 变成 0。",
            "",
            "下一轮不建议直接把 step 拉长。更合理的是先做 hard-case 数据选择：用 base/当前 ckpt 对 V13 train/val 批量 rollout，挑“同一题 4 个采样有对有错”的题做训练集。这样每一步才更可能有真实梯度。",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    metrics_rows = read_metrics(args.metrics_jsonl)
    rollout_rows = read_rollouts(args.rollout_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metric_fields = ["step", *[key for key, _title, _ylabel in KEY_METRICS]]
    write_csv(args.out_dir / "metrics_key.csv", metrics_rows, metric_fields)
    write_svg(args.out_dir / "metrics_key.svg", metrics_rows)
    write_rollout_step_csv(args.out_dir / "rollout_step_summary.v13.csv", rollout_rows)
    write_examples(args.out_dir / "rollout_examples.zh-CN.md", rollout_rows)
    write_report(
        args.report,
        args.run_name,
        metrics_summary(metrics_rows),
        rollout_summary(rollout_rows),
        args.out_dir,
    )
    print(f"metrics rows: {len(metrics_rows)}")
    print(f"rollout rows: {len(rollout_rows)}")
    print(f"report: {args.report}")
    print(f"svg: {args.out_dir / 'metrics_key.svg'}")


if __name__ == "__main__":
    main()
