#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_posttrain.critpt.synth import seed_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for example in seed_examples():
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

