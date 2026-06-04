from __future__ import annotations

from rl_posttrain.critpt.schema import VerifierSpec
from rl_posttrain.critpt.verifier import verify_completion


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **_: object,
) -> dict[str, object]:
    extra_info = extra_info or {}
    verifier_raw = extra_info.get("verifier")
    if isinstance(verifier_raw, dict):
        spec = VerifierSpec.from_dict(verifier_raw)
    else:
        spec = VerifierSpec(kind="exact", expected=str(ground_truth))

    result = verify_completion(solution_str, spec)
    format_ok = "<answer>" in solution_str.lower() and "</answer>" in solution_str.lower()
    score = 1.0 if result.ok else 0.05 if format_ok else 0.0
    return {
        "score": score,
        "acc": 1.0 if result.ok else 0.0,
        "format_ok": 1.0 if format_ok else 0.0,
        "reason": result.reason,
        "extracted": result.extracted,
        "data_source": data_source,
    }
