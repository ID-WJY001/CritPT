from __future__ import annotations

from collections import Counter

from rl_posttrain.critpt_synth.v15_hardmix import (
    generate_v15_hardmix_examples,
    summarize_v15_source,
    verify_v15_example,
)
from rl_posttrain.critpt_synth.verl_reward_v15_hardmix import compute_score


def test_v15_generation_verifies_and_covers_hard_families() -> None:
    examples = generate_v15_hardmix_examples(size=80, seed=20260615, split="train")
    failures = [(example.problem_id, verify_v15_example(example)[1]) for example in examples if not verify_v15_example(example)[0]]
    assert failures == []

    sources = Counter(summarize_v15_source(example) for example in examples)
    assert sources["v15_hardcase"] > sources["v14_base"]

    families = {example.family for example in examples}
    assert "v15_operator_charge_filter" in families
    assert "v15_recurrence_probe_suite" in families
    assert "v15_multi_channel_oam" in families
    assert all(example.metadata.get("uses_official_prompt") is False for example in examples)


def test_v15_reward_penalizes_think_tags_and_runaway() -> None:
    examples = generate_v15_hardmix_examples(size=20, seed=20260616, split="train")
    example = next(item for item in examples if item.family == "v15_operator_charge_filter")
    clean = example.assistant_code_block()
    with_think = f"<think>scratch</think>\n{clean}"
    runaway = "```python\ndef answer():\n    return " + repr(["x"] * 500) + "\n```"

    clean_score = compute_score(
        "critpt_synth_code",
        clean,
        example.target_code,
        {"code_verifier": example.verifier},
    )
    think_score = compute_score(
        "critpt_synth_code",
        with_think,
        example.target_code,
        {"code_verifier": example.verifier},
    )
    runaway_score = compute_score(
        "critpt_synth_code",
        runaway,
        example.target_code,
        {"code_verifier": example.verifier},
    )

    assert clean_score["score"] >= 0.94
    assert clean_score["score"] > think_score["score"]
    assert think_score["score"] <= 0.92
    assert runaway_score["score"] <= 0.12
