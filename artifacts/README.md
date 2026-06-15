# Curated Artifacts

Full experiment artifacts are not tracked. They often contain large rollout
dumps, generated JSON, logs, and remote paths.

This directory keeps only small curated evidence files under
`artifacts/curated/`.

## Selected Files

| File | Meaning |
| --- | --- |
| `curated/e1_e2_three_runs_training_curves.svg` | E1/E2 training curves across three runs |
| `curated/e1_e2_reward_length_focus.svg` | E1/E2 reward and response length focus plot |
| `curated/e3_realtime_curves.svg` | E3 strict final-answer judge realtime curves |
| `curated/e3_realtime_summary.json` | E3 curve summary |
| `curated/e4_realtime_curves.png` | E4 official-style realtime curves |
| `curated/v10_metrics_key.svg` | Earlier local-verifier curriculum metrics |
| `curated/v13_metrics_key.svg` | V13 official-code-format metrics |
| `curated/v14_metrics_key.svg` | V14 compact executable-answer metrics |
| `curated/v19_metrics_key.svg` | V19 failure-mined metrics |
| `curated/v19_rl_training_dashboard.svg` | V19 dashboard view |
| `curated/v20_metrics_key.svg` | V20 focused-hardcase metrics |
| `curated/e2_base_official70_score.json` | Example official70 score artifact |

These files provide compact evidence for the main training runs without
including the full training workspace.
