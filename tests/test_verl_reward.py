from __future__ import annotations

from rl_posttrain.critpt_synth.verl_reward import compute_score


def test_format_reward_for_well_formed_wrong_code() -> None:
    verifier = {"checks": [{"mode": "numeric_sequence", "expected": [1, 2], "tolerance": 1e-8}]}
    result = compute_score(
        "critpt_synth_code",
        "```python\ndef answer():\n    return [1, 3]\n```",
        "",
        {"code_verifier": verifier},
    )
    assert result["acc"] == 0.0
    assert result["format_reward"] > 0.0
    assert result["has_code_block"] == 1.0
    assert result["has_answer_def"] == 1.0
    assert result["single_answer_func"] == 1.0
    assert result["score"] >= result["format_reward"]


def test_no_format_reward_for_think_only_output() -> None:
    verifier = {"checks": [{"mode": "numeric_sequence", "expected": [1, 2], "tolerance": 1e-8}]}
    result = compute_score(
        "critpt_synth_code",
        "<think>reasoning forever",
        "",
        {"code_verifier": verifier},
    )
    assert result["score"] == 0.0
    assert result["format_reward"] == 0.0
    assert result["has_code_block"] == 0.0
    assert result["has_answer_def"] == 0.0
    assert result["no_think_tags"] == 0.0


def test_correct_code_still_gets_full_reward() -> None:
    verifier = {"checks": [{"mode": "numeric_sequence", "expected": [1, 2], "tolerance": 1e-8}]}
    result = compute_score(
        "critpt_synth_code",
        "```python\ndef answer():\n    return [1, 2]\n```",
        "",
        {"code_verifier": verifier},
    )
    assert result["score"] == 1.0
    assert result["acc"] == 1.0


def test_correct_code_with_think_tags_is_slightly_penalized() -> None:
    verifier = {"checks": [{"mode": "numeric_sequence", "expected": [1, 2], "tolerance": 1e-8}]}
    result = compute_score(
        "critpt_synth_code",
        "<think>\n\n</think>\n\n```python\ndef answer():\n    return [1, 2]\n```",
        "",
        {"code_verifier": verifier},
    )
    assert result["score"] == 0.97
    assert result["acc"] == 1.0
    assert result["no_think_tags"] == 0.0
