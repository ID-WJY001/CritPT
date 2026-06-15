# Selected Experiment Configs

This directory keeps a small set of representative training configs. The full
local history had many more smoke runs and aborted runs; those are intentionally
kept out of the public repository.

The selected configs cover the main story:

```text
v13  official code-format training
v14  compact executable answer training
v19  failure-mined data
v20  focused hardcases
e4   official-style final-answer training
e5b  failure-aware curriculum
e6   strict teacher-spec proposal
```

File names are verbose on purpose. Example:

```text
qwen3_8b_grpo_v19_failure_mined_from_v18_gs80_n8.env
```

This means: Qwen3-8B, GRPO, V19 failure-mined data, initialized from V18 step
80, rollout_n=8.

Secrets do not belong in these files. Put API keys in a private `.env` file.
