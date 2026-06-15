from __future__ import annotations

from rl_posttrain.model_judge.verl_reward_lenaware import _length_adjustment


def test_no_final_overlong_answer_is_capped() -> None:
    adjusted, metrics = _length_adjustment("推理。" * 1200, 0.95)

    assert adjusted == 0.55
    assert metrics["answer_marker_present"] == 0.0
    assert metrics["no_final_cap_applied"] == 1.0


def test_final_answer_with_long_tail_is_penalized() -> None:
    adjusted, metrics = _length_adjustment(
        "推理。\n最终答案：42\n" + "继续解释。" * 300,
        0.90,
    )

    assert adjusted < 0.90
    assert metrics["answer_marker_present"] == 1.0
    assert metrics["post_final_penalty_applied"] == 1.0


def test_concise_final_answer_keeps_score() -> None:
    adjusted, metrics = _length_adjustment("推理：直接计算。\n最终答案：42", 0.90)

    assert adjusted == 0.90
    assert metrics["answer_marker_present"] == 1.0
    assert metrics["length_penalty"] == 0.0
