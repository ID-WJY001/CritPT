#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("RL_DATA_ROOT", "/data/sdb/rl-posttrain"))
MODELS = ROOT / "models"


def size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def human(num: int) -> str:
    value = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def main() -> None:
    rows = []
    for model_dir in sorted(MODELS.glob("qwen3-*")):
        config = model_dir / "config.json"
        tokenizer = model_dir / "tokenizer.json"
        safetensors = list(model_dir.glob("*.safetensors"))
        rows.append(
            {
                "name": model_dir.name,
                "path": str(model_dir),
                "size": human(size_bytes(model_dir)),
                "has_config": config.exists(),
                "has_tokenizer": tokenizer.exists(),
                "safetensors_files": len(safetensors),
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

