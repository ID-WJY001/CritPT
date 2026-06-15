#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


ENDPOINT = "https://artificialanalysis.ai/api/v2/critpt/evaluate"


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a CritPt batch to Artificial Analysis.")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api-key-env", default="AA_API_KEY")
    parser.add_argument("--endpoint", default=ENDPOINT)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key: export {args.api_key_env}=...")

    payload = args.batch.read_bytes()
    req = urllib.request.Request(
        args.endpoint,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=3600) as response:
            body = response.read()
            status = response.status
            headers = dict(response.headers)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = exc.code
        headers = dict(exc.headers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": status,
        "headers": {
            k: v
            for k, v in headers.items()
            if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"
        },
        "body": json.loads(body.decode("utf-8")) if body else None,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if status >= 400:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
