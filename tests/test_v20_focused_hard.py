from __future__ import annotations

import json

from rl_posttrain.critpt_synth.v20_focused_hard import (
    generate_v20_focused_hard_examples,
    verify_v20_example,
)
from rl_posttrain.critpt_synth.verl_reward_v20_focused_hard import compute_score


def test_v20_generates_focused_families_and_verified_targets() -> None:
    examples = generate_v20_focused_hard_examples(90, 20260620, "train")
    families = {example.family for example in examples}
    assert "v20_operator_canonical_labels" in families
    assert "v20_empty_interval_filter" in families
    assert "v20_hhg_oam_safe_channels" in families
    assert all(not example.metadata.get("uses_official_prompt") for example in examples)
    assert all("Parsing template" in example.prompt for example in examples)

    for example in examples[:24]:
        ok, reason = verify_v20_example(example)
        assert ok, (example.problem_id, reason)


def test_v20_has_empty_interval_traps() -> None:
    examples = generate_v20_focused_hard_examples(120, 20260621, "train")
    empty_examples = [
        example
        for example in examples
        if example.family == "v20_empty_interval_filter"
        and example.metadata.get("expected_empty") is True
    ]
    assert empty_examples
    for example in empty_examples[:8]:
        ok, reason = verify_v20_example(example)
        assert ok, (example.problem_id, reason)


def test_v20_reward_caps_operator_concat_near_miss() -> None:
    examples = generate_v20_focused_hard_examples(40, 20260622, "train")
    operator = next(example for example in examples if example.family == "v20_operator_canonical_labels")
    bad_completion = """```python
def answer():
    return ["psi", "psipsi", "Fpsi"]
```"""
    result = compute_score(
        data_source="critpt_synth_code",
        solution_str=bad_completion,
        ground_truth=operator.target_code,
        extra_info={
            "code_verifier": json.dumps(operator.verifier),
            "metadata": json.dumps(operator.metadata),
        },
    )
    assert result["acc"] == 0.0
    assert result["v20_operator_concat_near_miss"] == 1.0
    assert result["score"] <= 0.42


def test_v20_reward_returns_stable_metadata_keys_across_families() -> None:
    examples = generate_v20_focused_hard_examples(90, 20260623, "train")
    results = []
    for family in {
        "v20_empty_interval_filter",
        "v20_hhg_oam_safe_channels",
        "v20_operator_canonical_labels",
    }:
        example = next(item for item in examples if item.family == family)
        result = compute_score(
            data_source="critpt_synth_code",
            solution_str="```python\ndef answer():\n    return None\n```",
            ground_truth=example.target_code,
            extra_info={
                "code_verifier": json.dumps(example.verifier),
                "metadata": json.dumps(example.metadata),
            },
        )
        results.append(result)

    v20_keys = {key for key in results[0] if key.startswith("v20_")}
    assert v20_keys
    assert all({key for key in result if key.startswith("v20_")} == v20_keys for result in results)
