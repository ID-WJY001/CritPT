#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${1:-/data/sdb/rl-posttrain/.env.judge}

cd /root/rl-posttrain
source /data/sdb/rl-posttrain/venvs/rl/bin/activate
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "missing ${ENV_FILE}" >&2
  echo "create it from .env.example and keep the real copy private" >&2
  exit 2
fi

source "${ENV_FILE}"

python - <<'PY'
import pandas as pd

from rl_posttrain.model_judge.verl_reward import compute_score

path = "/data/sdb/rl-posttrain/data/critpt_judge_v1_train.parquet"
df = pd.read_parquet(path)
row = df.iloc[0].to_dict()
candidate = "The Hamiltonian eigenvalue gap is sqrt(delta**2 + 16*g**2)."
result = compute_score(
    row["data_source"],
    candidate,
    row["reward_model"]["ground_truth"],
    row["extra_info"],
)
print(result)
if result.get("judge_error"):
    raise SystemExit("judge API failed")
if float(result.get("score", 0.0)) <= 0.0:
    raise SystemExit("judge returned non-positive score for a correct smoke answer")
PY
