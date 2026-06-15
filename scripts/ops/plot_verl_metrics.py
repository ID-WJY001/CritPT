#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_PATTERNS = [
    r"reward|score|return|advantage",
    r"acc|judge_error|quick_reject|correctness|instruction_following|reasoning_quality|fatal_error",
    r"actor/(.*loss|grad_norm|entropy|lr|clip|kl|pg|ppo)",
    r"critic/(.*loss|score|value|vpred)",
    r"old_log_prob|ref_log_prob|log_prob|kl|entropy",
    r"response(_|/)length|prompt(_|/)length|sequence(_|/)length|num_tokens",
    r"data/|batch/",
    r"timing_s/|time/",
    r"perf/|throughput|tflops|tokens_per",
    r"training/",
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


def read_metrics(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            row: dict[str, float] = {}
            row["step"] = float(raw.get("step", len(rows)))
            flatten("", raw.get("data", raw), row)
            rows.append(row)
    return rows


def choose_keys(rows: list[dict[str, float]], patterns: list[str], max_keys: int) -> list[str]:
    regexes = [re.compile(pattern) for pattern in patterns]
    keys = sorted({key for row in rows for key in row if key != "step"})
    picked = [key for key in keys if any(regex.search(key) for regex in regexes)]
    if max_keys > 0:
        return picked[:max_keys]
    return picked


def write_csv(rows: list[dict[str, float]], keys: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", *keys])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ["step", *keys]})


def scale_points(
    points: list[tuple[float, float]],
    width: int,
    height: int,
    left_pad: int,
    right_pad: int,
    top_pad: int,
    bottom_pad: int,
) -> tuple[str, tuple[float, float, float, float]]:
    if not points:
        return "", (0.0, 1.0, 0.0, 1.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmax = xmin + 1
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5
    coords = []
    for x, y in points:
        sx = left_pad + (x - xmin) / (xmax - xmin) * (width - left_pad - right_pad)
        sy = height - bottom_pad - (y - ymin) / (ymax - ymin) * (height - top_pad - bottom_pad)
        coords.append(f"{sx:.1f},{sy:.1f}")
    return " ".join(coords), (xmin, xmax, ymin, ymax)


def tick_values(lo: float, hi: float, count: int) -> list[float]:
    if count <= 1 or lo == hi:
        return [lo]
    return [lo + (hi - lo) * idx / (count - 1) for idx in range(count)]


def fmt_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def write_svg(rows: list[dict[str, float]], keys: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel_w = 860
    panel_h = 230
    left_pad = 82
    right_pad = 30
    top_pad = 46
    bottom_pad = 54
    total_h = max(panel_h * len(keys), panel_h)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_w}" height="{total_h}" '
        'font-family="Arial, sans-serif">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for idx, key in enumerate(keys):
        y0 = idx * panel_h
        points = [(row["step"], row[key]) for row in rows if key in row]
        ys = [p[1] for p in points]
        label = html.escape(key)
        parts.append(f'<g transform="translate(0,{y0})">')
        parts.append(f'<text x="{left_pad}" y="24" font-size="14" font-weight="700">{label}</text>')
        parts.append(
            f'<rect x="{left_pad}" y="{top_pad}" width="{panel_w - left_pad - right_pad}" '
            f'height="{panel_h - top_pad - bottom_pad}" fill="#fafafa" stroke="#ddd"/>'
        )
        if points:
            polyline, (xmin, xmax, ymin, ymax) = scale_points(
                points, panel_w, panel_h, left_pad, right_pad, top_pad, bottom_pad
            )
            parts.append(
                f'<text x="{panel_w - right_pad}" y="24" text-anchor="end" font-size="12" fill="#555">'
                f'min={ymin:.4g} max={ymax:.4g} last={ys[-1]:.4g}</text>'
            )
            x0 = left_pad
            x1 = panel_w - right_pad
            y0_axis = panel_h - bottom_pad
            y1_axis = top_pad
            parts.append(f'<line x1="{x1}" y1="{y0_axis}" x2="{x0}" y2="{y0_axis}" stroke="#777"/>')
            parts.append(f'<line x1="{x0}" y1="{y0_axis}" x2="{x0}" y2="{y1_axis}" stroke="#777"/>')
            for tick in tick_values(xmin, xmax, min(5, max(2, len(points)))):
                tx = left_pad + (tick - xmin) / (xmax - xmin) * (panel_w - left_pad - right_pad)
                parts.append(f'<line x1="{tx:.1f}" y1="{y0_axis}" x2="{tx:.1f}" y2="{y0_axis + 5}" stroke="#777"/>')
                parts.append(
                    f'<text x="{tx:.1f}" y="{y0_axis + 20}" text-anchor="middle" '
                    f'font-size="11" fill="#555">{html.escape(fmt_tick(tick))}</text>'
                )
            for tick in tick_values(ymin, ymax, 4):
                ty = panel_h - bottom_pad - (tick - ymin) / (ymax - ymin) * (panel_h - top_pad - bottom_pad)
                parts.append(f'<line x1="{left_pad - 5}" y1="{ty:.1f}" x2="{left_pad}" y2="{ty:.1f}" stroke="#777"/>')
                parts.append(
                    f'<text x="{left_pad - 9}" y="{ty + 4:.1f}" text-anchor="end" '
                    f'font-size="11" fill="#555">{html.escape(fmt_tick(tick))}</text>'
                )
            parts.append(
                f'<text x="{(left_pad + panel_w - right_pad) / 2:.1f}" y="{panel_h - 12}" '
                'text-anchor="middle" font-size="12" fill="#444">training step</text>'
            )
            parts.append(
                f'<text x="16" y="{(top_pad + panel_h - bottom_pad) / 2:.1f}" '
                'text-anchor="middle" font-size="12" fill="#444" '
                'transform="rotate(-90 16 '
                f'{(top_pad + panel_h - bottom_pad) / 2:.1f})">metric value</text>'
            )
            parts.append(f'<polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2"/>')
        else:
            parts.append(f'<text x="{left_pad}" y="{panel_h/2}" font-size="12" fill="#999">no data</text>')
        parts.append("</g>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--patterns", nargs="*", default=DEFAULT_PATTERNS)
    parser.add_argument("--max-keys", type=int, default=40, help="0 means keep all matched keys")
    parser.add_argument("--basename", default="metrics")
    args = parser.parse_args()

    rows = read_metrics(args.metrics_jsonl)
    if not rows:
        raise SystemExit(f"no rows in {args.metrics_jsonl}")
    keys = choose_keys(rows, args.patterns, args.max_keys)
    if not keys:
        keys = sorted({key for row in rows for key in row if key != "step"})
        if args.max_keys > 0:
            keys = keys[: args.max_keys]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, keys, args.out_dir / f"{args.basename}.csv")
    write_svg(rows, keys, args.out_dir / f"{args.basename}.svg")
    print(f"rows: {len(rows)}")
    print(f"keys: {len(keys)}")
    print(f"csv: {args.out_dir / f'{args.basename}.csv'}")
    print(f"svg: {args.out_dir / f'{args.basename}.svg'}")


if __name__ == "__main__":
    main()
