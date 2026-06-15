from __future__ import annotations

import inspect

from rl_posttrain.model_judge import verl_reward_semantic_code as reward


def test_semantic_reward_does_not_call_local_verifier() -> None:
    source = inspect.getsource(reward)

    assert "verify_code_completion" not in source
    assert "extract_python_code" not in source
    assert "local_exact_pass" not in source
    assert "code_verifier" not in source


def test_semantic_payload_scoring_uses_judge_subscores() -> None:
    payload = reward.normalize_semantic_payload(
        {
            "semantic_correctness": 9,
            "output_contract": 8,
            "reasoning_groundedness": 7,
            "final_answer_consistency": 9,
            "overgeneration_penalty": 0,
            "fatal_error": False,
            "failure_tags": [],
            "confidence": 0.8,
            "reason": "correct",
        }
    )

    score = reward.score_from_semantic_payload(payload)

    assert 0.82 < score < 0.91


def test_semantic_reward_caps_judge_declared_operator_overenumeration() -> None:
    payload = reward.normalize_semantic_payload(
        {
            "semantic_correctness": 8,
            "output_contract": 8,
            "reasoning_groundedness": 8,
            "final_answer_consistency": 8,
            "overgeneration_penalty": 9,
            "fatal_error": False,
            "failure_tags": ["operator_overenumeration"],
            "confidence": 0.9,
            "reason": "returns a superset",
        }
    )

    assert reward.score_from_semantic_payload(payload) <= 0.40


def test_build_messages_contain_no_local_verifier_evidence_section() -> None:
    messages = reward.build_semantic_judge_messages(
        problem="Compute x.",
        candidate_response="x=1",
        reference_answer="x=1",
        reference_trace="direct",
        rubric="strict",
        metadata={"family": "unit"},
    )
    joined = "\n".join(message["content"] for message in messages)

    assert "Local verifier evidence" not in joined
    assert "Reference answer" in joined
