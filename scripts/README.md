# Scripts

The scripts are grouped by workflow stage:

- `data/`: build synthetic, official-style, failure-mined, and LLM-assisted
  datasets.
- `eval/`: run local or official-style evaluation and analyze submissions.
- `ops/`: launch remote experiments, merge checkpoints, plot metrics, and
  inspect rollouts.
- `remote/`: bootstrap rented GPU machines.
- `smoke/`: distributed/runtime smoke tests.
- `train/`: SFT and GRPO launchers.

Most remote scripts assume a private GPU workspace layout and should be treated
as reproducibility notes rather than one-command public demos.

