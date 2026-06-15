# RL Post-Training Lab

This repository contains a research-and-engineering project for post-training
Qwen-style language models on CritPT-like scientific Python-answer tasks.

The task format is code-centric: the model reads a scientific problem statement
and returns an executable Python `answer()` function. The repository includes
data generation code, reward functions, GRPO training configs, evaluation
scripts, and selected training artifacts.

## Scope

- Built a full GRPO post-training loop with `verl`, vLLM rollouts, checkpoint
  merge/eval scripts, and training metric visualization.
- Implemented multiple data-generation paths: programmatic synthetic tasks,
  official-style prompt wrappers, failure-mined hard cases, and LLM-generated
  teacher specifications.
- Implemented reward variants: local final-answer verification, semantic code
  judging, length-aware shaping, strict final-answer judging, and LLM-as-a-judge
  wrappers.
- Ran iterative experiments from early V-series format/verifier training to
  E-series LLM-judge training.
- Analyzed why attractive training curves can still fail official evaluation:
  reward hacking, synthetic-task mismatch, and placeholder-like answers.

## Result

The end-to-end training and evaluation pipeline is functional. The later
official-style V/E runs improved formatting, length control, and executable
`answer()` structure, but did not improve official70 accuracy. The main finding
is that synthetic rewards and LLM-judge rewards need much tighter semantic
alignment with the target benchmark; clean formatting alone is not sufficient.

## Repository Map

```text
src/rl_posttrain/
  critpt/              Early schema, verifier, reward, and eval utilities
  critpt_synth/        Synthetic task generators and local verifier rewards
  model_judge/         OpenAI-compatible LLM judge clients and reward wrappers

scripts/
  data/                Dataset builders for V/E experiment families
  eval/                Local and official-style evaluation scripts
  ops/                 Remote run, plotting, merge, and inspection utilities
  remote/              GPU-node bootstrap helpers
  train/               SFT/GRPO launchers

configs/experiments/  Reproducible env configs for each experiment
docs/                 Project overview, pipeline, and experiment summary
artifacts/curated/    Selected training curves and summaries
tests/                Unit tests for generators, verifiers, and judge plumbing
```

## Key Documents

- [Overview](docs/overview.zh-CN.md)
- [Pipeline](docs/pipeline.zh-CN.md)
- [Experiment summary](docs/experiments.zh-CN.md)
- [Research journey](docs/research_journey.zh-CN.md)
- [Run evidence and reproducibility notes](docs/run_evidence.zh-CN.md)
- [RL diagnostics](docs/rl_diagnostics.zh-CN.md)
- [Artifact index](artifacts/README.md)

## Reproduce Locally

Local tests do not require model weights or API keys.

```bash
uv run pytest
```

LLM-judge and remote GRPO runs require private infrastructure. Copy
`.env.example` to a private `.env` file and set keys locally.

## Security Notes

No API keys or passwords should be committed. Remote host names in public docs
are represented with placeholders such as `<GPU_HOST>`. Full model weights,
checkpoints, raw rollouts, and generated parquet data are intentionally ignored
by Git.
