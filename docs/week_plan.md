# 4-5 Day Sprint Plan

## Day 0

- SSH and key auth.
- Confirm GPU count, topology, RAM, disk.
- Prepare `/data/sdb/rl-posttrain`.
- Install dependencies.
- Run torch all-reduce.

## Day 1

- Build CritPT seed dataset and verifier.
- Load `Qwen3-14B` or `Qwen3-8B` with vLLM.
- Run baseline eval.

## Day 2

- Run the stable GRPO/SFT smoke.
- Save checkpoints and reward traces.

## Day 3

- Try guarded `Qwen3-32B` QLoRA/GRPO smoke.
- If OOM, capture logs and switch to 14B for the main result.

## Day 4

- Mine worse cases.
- Run anti-hack checks.
- Write README/results notes for interviews.

