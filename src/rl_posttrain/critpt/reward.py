from __future__ import annotations

from rl_posttrain.critpt.schema import VerifierSpec
from rl_posttrain.critpt.verifier import verify_completion


def critpt_reward(completion: str, verifier: VerifierSpec) -> float:
    result = verify_completion(completion, verifier)
    if result.ok:
        return 1.0
    if "<answer>" in completion.lower() and "</answer>" in completion.lower():
        return 0.05
    return 0.0

