from __future__ import annotations

import json

from rl_posttrain.critpt_synth.final_answer_verifier import verify_final_answer


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **_: object,
) -> dict[str, object]:
    del ground_truth
    extra_info = extra_info or {}
    verifier = extra_info.get("code_verifier") or extra_info.get("verifier") or {"checks": []}
    if isinstance(verifier, str):
        verifier = json.loads(verifier)

    result = verify_final_answer(solution_str, verifier)
    return {
        "score": result.score,
        "acc": 1.0 if result.ok else 0.0,
        "answer_marker_present": 1.0 if result.answer_marker_present else 0.0,
        "parse_ok": 1.0 if result.parse_ok else 0.0,
        "skip_phrase_present": 1.0 if result.skip_phrase_present else 0.0,
        "passed_checks": result.passed_checks,
        "total_checks": result.total_checks,
        "extracted_answer": result.extracted_answer,
        "reason": result.reason,
        "data_source": data_source,
    }
