from __future__ import annotations

import http.client

from rl_posttrain.model_judge.openai_compatible import JudgeSettings, openai_chat_json


def test_openai_chat_json_retries_remote_disconnect(monkeypatch) -> None:
    calls: list[float] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"{\\"ok\\": true}"}}]}'

    def fake_urlopen(_request: object, *, timeout: float) -> FakeResponse:
        calls.append(timeout)
        if len(calls) == 1:
            raise http.client.RemoteDisconnected("closed before response")
        return FakeResponse()

    monkeypatch.setattr(
        "rl_posttrain.model_judge.openai_compatible.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr("rl_posttrain.model_judge.openai_compatible.time.sleep", lambda _seconds: None)

    payload = openai_chat_json(
        settings=JudgeSettings(
            api_key="unit",
            base_url="https://unit.test/v1",
            model="unit",
            timeout_s=3,
            max_retries=1,
        ),
        messages=[{"role": "user", "content": "return json"}],
    )

    assert payload == {"ok": True}
    assert calls == [3, 3]
