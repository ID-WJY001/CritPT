from __future__ import annotations

import inspect

from rl_posttrain.model_judge import verl_reward_e3_strict_code as reward


def test_e3_strict_reward_does_not_call_local_verifier() -> None:
    source = inspect.getsource(reward)

    assert "verify_code_completion" not in source
    assert "extract_python_code" not in source
    assert "local_exact_pass" not in source
    assert "code_verifier" not in source


def test_e3_strict_score_high_for_equivalent_payload() -> None:
    payload = reward.normalize_strict_payload(
        {
            "algorithmic_equivalence": 9,
            "scientific_correctness": 9,
            "numeric_symbolic_exactness": 8,
            "output_contract": 9,
            "anti_guess_evidence": 9,
            "fatal_error": False,
            "failure_tags": [],
            "confidence": 0.9,
            "reason": "implements the reference scan",
        }
    )

    assert reward.score_from_strict_payload(payload) > 0.84


def test_e3_strict_caps_constant_guess() -> None:
    payload = reward.normalize_strict_payload(
        {
            "algorithmic_equivalence": 2,
            "scientific_correctness": 2,
            "numeric_symbolic_exactness": 2,
            "output_contract": 9,
            "anti_guess_evidence": 1,
            "fatal_error": False,
            "failure_tags": ["constant_guess"],
            "confidence": 0.9,
            "reason": "returns a plausible hard-coded value",
        }
    )

    assert reward.score_from_strict_payload(payload) <= 0.15


def test_e3_strict_caps_shape_only_answer() -> None:
    payload = reward.normalize_strict_payload(
        {
            "algorithmic_equivalence": 2,
            "scientific_correctness": 4,
            "numeric_symbolic_exactness": 4,
            "output_contract": 9,
            "anti_guess_evidence": 2,
            "fatal_error": False,
            "failure_tags": ["shape_only_answer"],
            "confidence": 0.8,
            "reason": "right answer function but no computation",
        }
    )

    assert reward.score_from_strict_payload(payload) <= 0.20


def test_e3_accepts_python_fence_but_rejects_missing_answer() -> None:
    code, tags = reward._candidate_code_for_judge(
        "```python\ndef answer():\n    return 42\n```"
    )
    missing = reward._quick_reject("result = 42", "result = 42", "unit")

    assert code == "def answer():\n    return 42"
    assert tags == []
    assert missing["quick_reject"] == 1.0
    assert missing["tag_missing_answer_function"] == 1.0


def test_e3_messages_stress_reference_equivalence() -> None:
    messages = reward.build_strict_code_judge_messages(
        problem="Compute the finite scan.",
        candidate_original_response="```python\ndef answer():\n    return 0\n```",
        candidate_response="def answer():\n    return 0",
        reference_answer="def answer():\n    return scan()",
        reference_final_answer="42",
        reference_trace="Scan all candidates and apply the tie-break.",
        rubric="strict",
        metadata={"family": "finite_scan", "answer_type": "tuple"},
    )
    joined = "\n".join(message["content"] for message in messages)

    assert "final-answer judge" in joined
    assert "final literal" in joined
    assert "constant" in joined
    assert "tie-break" in joined
