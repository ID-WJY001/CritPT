#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${1:-}

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

if [ -n "${ENV_FILE}" ]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "missing OPENAI_API_KEY" >&2
  exit 2
fi

python3 - <<'PY'
from rl_posttrain.model_judge.verl_reward_semantic_code import compute_score

problem = "Return a Python answer() function for the integer 42."
reference = "def answer():\n    return 42"


def judge(candidate: str) -> dict[str, object]:
    return compute_score(
        data_source="semantic_judge_smoke",
        solution_str=candidate,
        ground_truth=reference,
        extra_info={
            "prompt_text": problem,
            "reference_answer": reference,
            "reference_trace": "The requested integer is exactly 42.",
            "rubric": "Reward exact semantic correctness and the answer() contract.",
            "metadata": '{"family":"smoke","answer_type":"integer"}',
        },
    )


correct = judge("```python\ndef answer():\n    return 42\n```")
wrong = judge("```python\ndef answer():\n    return 43\n```")

for label, result in [("correct", correct), ("wrong", wrong)]:
    safe = {
        "case": label,
        "score": result.get("score"),
        "acc": result.get("acc"),
        "judge_error": result.get("judge_error"),
        "semantic_correctness": result.get("semantic_correctness"),
        "output_contract": result.get("output_contract"),
        "reason": result.get("reason"),
    }
    print(safe)
    if result.get("judge_error"):
        raise SystemExit(f"{label}: judge API failed")

if float(correct.get("score", 0.0)) <= 0.8:
    raise SystemExit("judge returned too low a score for the correct smoke answer")
if float(wrong.get("score", 1.0)) >= 0.65:
    raise SystemExit("judge returned too high a score for the wrong smoke answer")
PY
