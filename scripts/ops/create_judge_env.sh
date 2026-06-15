#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${1:-/data/sdb/rl-posttrain/.env.judge}
DEFAULT_BASE_URL=${OPENAI_BASE_URL:-https://api.yunwu.cloud}
DEFAULT_MODEL=${JUDGE_MODEL:-gpt-5.5}
DEFAULT_CACHE=${JUDGE_CACHE_PATH:-/data/sdb/rl-posttrain/data/judge_cache/model_judge.sqlite3}

mkdir -p "$(dirname "${ENV_FILE}")" "$(dirname "${DEFAULT_CACHE}")"

read -r -p "OPENAI_BASE_URL [${DEFAULT_BASE_URL}]: " BASE_URL
BASE_URL=${BASE_URL:-${DEFAULT_BASE_URL}}

read -r -p "JUDGE_MODEL [${DEFAULT_MODEL}]: " MODEL
MODEL=${MODEL:-${DEFAULT_MODEL}}

read -r -s -p "OPENAI_API_KEY: " API_KEY
printf "\n"

if [ -z "${API_KEY}" ]; then
  echo "OPENAI_API_KEY is empty; aborting." >&2
  exit 2
fi

umask 077
cat >"${ENV_FILE}" <<EOF
export OPENAI_API_KEY='${API_KEY}'
export OPENAI_BASE_URL='${BASE_URL}'
export JUDGE_MODEL='${MODEL}'
export JUDGE_CACHE_PATH='${DEFAULT_CACHE}'
export JUDGE_TIMEOUT_S=60
export JUDGE_MAX_RETRIES=2
EOF
chmod 600 "${ENV_FILE}"

echo "wrote ${ENV_FILE}"
echo "next: source ${ENV_FILE} && bash scripts/ops/test_model_judge_api.sh"
