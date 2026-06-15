from __future__ import annotations

from rl_posttrain.critpt_synth.verl_reward_v14_compact import compute_score


def test_v14_correct_code_gets_full_reward() -> None:
    verifier = {"checks": [{"mode": "numeric_sequence", "expected": [1, 2], "tolerance": 1e-8}]}
    result = compute_score(
        "critpt_synth_code",
        "```python\ndef answer():\n    return [1, 2]\n```",
        "def answer():\n    return [1, 2]\n",
        {"code_verifier": verifier},
    )
    assert result["score"] == 1.0
    assert result["acc"] == 1.0
    assert result["compile_ok"] == 1.0


def test_v14_syntax_broken_runaway_is_capped_low() -> None:
    verifier = {"checks": [{"mode": "numeric_sequence", "expected": [1, 2], "tolerance": 1e-8}]}
    repeated = ", ".join(["0.0"] * 900)
    result = compute_score(
        "critpt_synth_code",
        f"```python\ndef answer():\n    coeffs = [{repeated}",
        "def answer():\n    return [1, 2]\n",
        {"code_verifier": verifier},
    )
    assert result["acc"] == 0.0
    assert result["compile_ok"] == 0.0
    assert result["unclosed_code_block"] == 1.0
    assert result["score"] <= 0.08


def test_v14_repeated_wrong_code_scores_below_compact_wrong_code() -> None:
    verifier = {"checks": [{"mode": "numeric_sequence", "expected": [1, 2], "tolerance": 1e-8}]}
    compact = compute_score(
        "critpt_synth_code",
        "```python\ndef answer():\n    return [1, 3]\n```",
        "def answer():\n    return [1, 2]\n",
        {"code_verifier": verifier},
    )
    repeated_lines = "\n".join(["    x = x + 1" for _ in range(80)])
    runaway = compute_score(
        "critpt_synth_code",
        f"```python\ndef answer():\n    x = 0\n{repeated_lines}\n    return [1, 3]\n```",
        "def answer():\n    return [1, 2]\n",
        {"code_verifier": verifier},
    )
    assert compact["score"] > runaway["score"]
    assert runaway["repetition_penalty"] > 0
