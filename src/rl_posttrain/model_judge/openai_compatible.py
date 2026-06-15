from __future__ import annotations

import hashlib
import http.client
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROMPT_VERSION = "critpt-model-judge-v2-cot"

_CACHE_INIT_LOCKS: dict[str, threading.Lock] = {}
_CACHE_INIT_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class JudgeSettings:
    api_key: str
    base_url: str
    model: str
    timeout_s: float = 60.0
    max_tokens: int = 512
    temperature: float = 0.0
    cache_path: str = ""
    max_retries: int = 2

    @staticmethod
    def from_env() -> "JudgeSettings":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        model = os.environ.get("JUDGE_MODEL", os.environ.get("OPENAI_MODEL", "")).strip()
        if not model:
            model = "gpt-4o-mini"
        return JudgeSettings(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_s=float(os.environ.get("JUDGE_TIMEOUT_S", "60")),
            max_tokens=int(os.environ.get("JUDGE_MAX_TOKENS", "512")),
            temperature=float(os.environ.get("JUDGE_TEMPERATURE", "0")),
            cache_path=os.environ.get("JUDGE_CACHE_PATH", "").strip(),
            max_retries=int(os.environ.get("JUDGE_MAX_RETRIES", "2")),
        )


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def clamp_float(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if number < lo:
        return lo
    if number > hi:
        return hi
    return number


def cache_key(*parts: str) -> str:
    payload = "\n\n---\n\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JsonSqliteCache:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_key = str(self.path.resolve())
        with _CACHE_INIT_LOCKS_GUARD:
            init_lock = _CACHE_INIT_LOCKS.setdefault(lock_key, threading.Lock())
        with init_lock:
            with self._connect() as conn:
                self._execute_with_retries(conn, "PRAGMA journal_mode=WAL")
                self._execute_with_retries(
                    conn,
                    "CREATE TABLE IF NOT EXISTS judge_cache "
                    "(cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at REAL NOT NULL)",
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=60)
        conn.execute("PRAGMA busy_timeout=60000")
        return conn

    @staticmethod
    def _execute_with_retries(
        conn: sqlite3.Connection,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> sqlite3.Cursor:
        last_error: sqlite3.OperationalError | None = None
        for attempt in range(8):
            try:
                return conn.execute(sql, params)
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                last_error = exc
                time.sleep(0.2 * (attempt + 1))
        raise sqlite3.OperationalError(f"sqlite cache remained locked: {last_error}")

    def get(self, key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM judge_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        try:
            return dict(json.loads(row[0]))
        except json.JSONDecodeError:
            return None

    def set(self, key: str, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._connect() as conn:
            self._execute_with_retries(
                conn,
                "INSERT OR REPLACE INTO judge_cache(cache_key, payload, created_at) VALUES (?, ?, ?)",
                (key, raw, time.time()),
            )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return dict(json.loads(stripped))
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return dict(json.loads(stripped[start : end + 1]))
        raise


def build_judge_messages(
    *,
    problem: str,
    candidate_response: str,
    reference_answer: str = "",
    reference_trace: str = "",
    rubric: str = "",
) -> list[dict[str, str]]:
    system = (
        "You are a strict but fair scientific problem-solving judge. "
        "Grade only the candidate response for the given problem. "
        "Reward useful reasoning, correct final answer, and instruction following. "
        "A candidate may include a chain of reasoning; do not penalize reasoning merely for being present. "
        "When reference reasoning notes are provided, use them as a guide to check whether the candidate's "
        "reasoning is grounded and on the right path, but do not require identical wording. "
        "Penalize empty/nonresponsive answers, unsupported guesses, fabricated facts, "
        "hard-coded reward-hacking phrases, circular rambling, and answers that ignore the problem. "
        "Also penalize responses that reason for a long time but never provide a clear final answer. "
        "Return JSON only."
    )
    user_parts = [
        "# Problem",
        problem.strip(),
        "",
        "# Candidate response",
        candidate_response.strip(),
    ]
    if reference_answer.strip():
        user_parts += ["", "# Reference answer or solution sketch", reference_answer.strip()]
    if reference_trace.strip():
        user_parts += ["", "# Reference reasoning notes", reference_trace.strip()]
    if rubric.strip():
        user_parts += ["", "# Extra rubric", rubric.strip()]
    user_parts += [
        "",
        "# Required JSON schema",
        (
            '{"correctness": integer 0-10, '
            '"instruction_following": integer 0-10, '
            '"reasoning_quality": integer 0-10, '
            '"final_answer_consistency": integer 0-10, '
            '"fatal_error": boolean, '
            '"reward": number 0.0-1.0, '
            '"reason": "short reason"}'
        ),
        "",
        "Scoring guidance: correctness should primarily track whether the final scientific/mathematical result is right; "
        "reasoning_quality should track whether the chain of reasoning is relevant, logically valid, and aligned with "
        "the reference reasoning notes when available; final_answer_consistency should track whether the final answer "
        "is explicit and consistent with the reasoning. "
        "If the reference is code, judge the mathematical/scientific result it represents, not formatting alone. "
        "If no exact reference is sufficient, use your own domain judgment.",
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def openai_chat_json(
    *,
    settings: JudgeSettings,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    if not settings.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    payload = {
        "model": settings.model,
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        chat_completions_url(settings.base_url),
        data=data,
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(settings.max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=settings.timeout_s) as response:
                raw = response.read().decode("utf-8")
            body = json.loads(raw)
            content = body["choices"][0]["message"]["content"]
            return extract_json_object(content)
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt >= settings.max_retries:
                break
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"judge API call failed: {last_error}")


def normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    correctness = clamp_float(payload.get("correctness"), 0.0, 10.0)
    instruction = clamp_float(payload.get("instruction_following"), 0.0, 10.0)
    reasoning = clamp_float(payload.get("reasoning_quality"), 0.0, 10.0)
    final_consistency = clamp_float(payload.get("final_answer_consistency"), 0.0, 10.0)
    raw_reward = payload.get("reward")
    reward = clamp_float(raw_reward, 0.0, 1.0)
    if raw_reward is None:
        reward = (
            0.40 * correctness
            + 0.15 * instruction
            + 0.25 * reasoning
            + 0.20 * final_consistency
        ) / 10.0
    fatal_error = bool(payload.get("fatal_error", False))
    if fatal_error:
        reward = min(reward, 0.05)
    return {
        "reward": clamp_float(reward, 0.0, 1.0),
        "correctness": correctness,
        "instruction_following": instruction,
        "reasoning_quality": reasoning,
        "final_answer_consistency": final_consistency,
        "fatal_error": fatal_error,
        "reason": str(payload.get("reason", ""))[:500],
    }
