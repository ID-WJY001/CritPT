#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: bash scripts/train/run_verl_grpo.sh CONFIG.env" >&2
  exit 2
fi

CONFIG=$1
source "${CONFIG}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-false}
export ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-false}
export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-true}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
export RESUME_MODE=${RESUME_MODE:-disable}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
export TRAINER_LOGGERS=${TRAINER_LOGGERS:-'["console","file","tensorboard"]'}
export VERL_FILE_LOGGER_PATH=${VERL_FILE_LOGGER_PATH:-"${LOG_ROOT}/${RUN_NAME}.metrics.jsonl"}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-"${LOG_ROOT}/tensorboard/${RUN_NAME}"}
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
export ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
export ACTOR_LR=${ACTOR_LR:-1e-6}

mkdir -p "${CHECKPOINT_ROOT}/${RUN_NAME}" "${LOG_ROOT}" "$(dirname "${VERL_FILE_LOGGER_PATH}")" "${TENSORBOARD_DIR}"

REWARD_PATH="${REWARD_PATH:-$(pwd)/src/rl_posttrain/critpt/verl_reward.py}"
EXTRA_ARGS=()
if [ -n "${TOTAL_TRAINING_STEPS:-}" ]; then
  EXTRA_ARGS+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi
if [ -n "${ROLLOUT_DATA_DIR:-}" ]; then
  mkdir -p "${ROLLOUT_DATA_DIR}"
  EXTRA_ARGS+=(trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}")
fi
if [ -n "${VALIDATION_DATA_DIR:-}" ]; then
  mkdir -p "${VALIDATION_DATA_DIR}"
  EXTRA_ARGS+=(trainer.validation_data_dir="${VALIDATION_DATA_DIR}")
fi
if [ -n "${ENABLE_THINKING:-}" ]; then
  EXTRA_ARGS+=(+data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}")
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  algorithm.use_kl_in_reward=false \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=true \
  data.return_multi_modal_inputs=false \
  data.truncation=error \
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  actor_rollout_ref.model.path="${MODEL_NAME}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
  actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
  actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
  +actor_rollout_ref.rollout.enable_sleep_mode="${ROLLOUT_ENABLE_SLEEP_MODE}" \
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  reward.custom_reward_function.path="${REWARD_PATH}" \
  reward.custom_reward_function.name=compute_score \
  trainer.project_name=critpt_qwen_rl \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.logger="${TRAINER_LOGGERS}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.default_local_dir="${CHECKPOINT_ROOT}/${RUN_NAME}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_ROOT}/${RUN_NAME}.log"
