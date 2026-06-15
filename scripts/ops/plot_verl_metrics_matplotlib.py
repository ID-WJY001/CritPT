#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PANELS = [
    ("reward", ["critic/score/mean", "critic/rewards/mean", "reward/mean", "score/mean"]),
    ("reward spread", ["critic/score/max", "critic/score/min", "critic/rewards/max", "critic/rewards/min"]),
    ("kl", ["actor/kl_loss", "actor/kl", "critic/kl", "kl"]),
    ("entropy", ["actor/entropy", "entropy"]),
    ("response length", ["response_length/mean", "response/length/mean", "sequence_length/mean"]),
    ("prompt length", ["prompt_length/mean", "prompt/length/mean"]),
    ("grad/lr", ["actor/grad_norm", "actor/lr"]),
    ("timing", ["time/step", "timing_s/step", "perf/total_num_tokens"]),
]


def flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}/{key}" if prefix else str(key), child, out)
        return
    if isinstance(value, bool):
        out[prefix] = 1.0 if value else 0.0
        return
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        out[prefix] = float(value)


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            row: dict[str, float] = {"step": float(raw.get("step", len(rows)))}
            flatten("", raw.get("data", raw), row)
            rows.append(row)
    return rows


def first_present(rows: list[dict[str, float]], keys: list[str]) -> list[str]:
    available = {key for row in rows for key in row}
    return [key for key in keys if key in available]


def write_summary(path: Path, rows: list[dict[str, float]]) -> None:
    summary: dict[str, Any] = {"rows": len(rows), "last_step": int(rows[-1]["step"]) if rows else None}
    for title, keys in PANELS:
        for key in first_present(rows, keys):
            values = [row[key] for row in rows if key in row]
            if not values:
                continue
            summary[key] = {
                "last": values[-1],
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "panel": title,
            }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--basename", default="training_realtime")
    parser.add_argument("--title", default="VERL GRPO training metrics")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_rows(args.metrics_jsonl)
    if not rows:
        raise SystemExit(f"no metric rows found in {args.metrics_jsonl}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(12, 2.35 * len(PANELS)), constrained_layout=True)
    fig.suptitle(args.title)

    for ax, (panel_title, candidates) in zip(axes, PANELS):
        keys = first_present(rows, candidates)
        if not keys:
            ax.text(0.5, 0.5, "not logged yet", ha="center", va="center", color="#777777")
            ax.set_title(panel_title)
            ax.set_axis_off()
            continue
        for key in keys:
            points = [(row["step"], row[key]) for row in rows if key in row]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            ax.plot(xs, ys, linewidth=1.8, marker=".", markersize=3, label=key)
        ax.set_title(panel_title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    png_path = args.out_dir / f"{args.basename}.png"
    svg_path = args.out_dir / f"{args.basename}.svg"
    summary_path = args.out_dir / f"{args.basename}.summary.json"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    write_summary(summary_path, rows)
    print(f"rows: {len(rows)}")
    print(f"png: {png_path}")
    print(f"svg: {svg_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
