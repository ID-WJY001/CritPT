#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


BASES = [
    "https://yunwu.ai",
    "https://api.yunwu.cloud",
    "https://api.apiplus.org",
    "https://api3.wlai.vip",
    "https://api.zhongzhuan.chat",
]

MODELS = [
    "gpt-5-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "deepseek-v4-flash",
    "gpt-5.5",
]


def chat_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def probe(base: str, model: str, key: str) -> tuple[str, str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": 'Return JSON {"ok": true} only.'}],
        "temperature": 0,
        "max_tokens": 40,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        chat_url(base),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return "OK", str(content)[:160]
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode(errors="ignore").replace("\n", " ")[:220]
        return f"HTTP {exc.code}", msg
    except Exception as exc:  # pragma: no cover - diagnostic script
        return exc.__class__.__name__, str(exc)[:220]


def main() -> int:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("missing OPENAI_API_KEY", file=sys.stderr)
        return 2
    for base in BASES:
        for model in MODELS:
            status, message = probe(base, model, key)
            print(base, model, status, message)
            if status == "OK":
                print(f"WORKING_BASE_URL={base}")
                print(f"WORKING_JUDGE_MODEL={model}")
                return 0
    print("NO_WORKING_COMBO")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
