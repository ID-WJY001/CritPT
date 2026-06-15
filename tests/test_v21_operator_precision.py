from __future__ import annotations

import json

from rl_posttrain.critpt_synth.v20_focused_hard import generate_v20_focused_hard_examples
from rl_posttrain.critpt_synth.v21_operator_precision import generate_v21_operator_precision_examples
from rl_posttrain.critpt_synth.verl_reward_v21_operator_precision import compute_score


def _operator_example():
    examples = generate_v20_focused_hard_examples(80, 20260624, "train")
    return next(example for example in examples if example.family == "v20_operator_canonical_labels")


def _extra_info(example) -> dict[str, str]:
    return {
        "code_verifier": json.dumps(example.verifier),
        "metadata": json.dumps(example.metadata),
    }


def test_v21_caps_operator_overenumeration() -> None:
    example = _operator_example()
    labels = [f"tr(extra{i})" for i in range(96)]
    completion = f"""```python
def answer():
    return {labels!r}
```"""
    result = compute_score(
        data_source="critpt_synth_code",
        solution_str=completion,
        ground_truth=example.target_code,
        extra_info=_extra_info(example),
    )
    assert result["acc"] == 0.0
    assert result["v21_operator_too_many_literals"] == 1.0
    assert result["v21_operator_literal_count"] >= 80.0
    assert result["score"] <= 0.22


def test_v21_caps_noncanonical_power_notation() -> None:
    example = _operator_example()
    completion = """```python
def answer():
    return ["tr(B)^2", "tr(F)^2", "tr(psi)^3"]
```"""
    result = compute_score(
        data_source="critpt_synth_code",
        solution_str=completion,
        ground_truth=example.target_code,
        extra_info=_extra_info(example),
    )
    assert result["acc"] == 0.0
    assert result["v21_operator_noncanonical_power"] == 1.0
    assert result["score"] <= 0.34


def test_v21_does_not_cap_correct_programmatic_solution() -> None:
    example = _operator_example()
    completion = f"```python\n{example.target_code}\n```"
    result = compute_score(
        data_source="critpt_synth_code",
        solution_str=completion,
        ground_truth=example.target_code,
        extra_info=_extra_info(example),
    )
    assert result["acc"] == 1.0
    assert result["score"] > 0.85
    assert result["v21_operator_precision_reward"] == result["score"]


def test_v21_generates_operator_heavy_verified_mix() -> None:
    examples = generate_v21_operator_precision_examples(120, 20260625, "train")
    operator_count = sum(1 for example in examples if example.metadata.get("v21_focus") == "operator_precision")
    assert operator_count >= 80
    assert all(not example.metadata.get("uses_official_prompt") for example in examples)
    assert all("Parsing template" in example.prompt for example in examples)


def test_v21_reward_returns_stable_metadata_keys_across_families() -> None:
    examples = generate_v21_operator_precision_examples(120, 20260626, "train")
    families = {
        "v20_empty_interval_filter",
        "v20_hhg_oam_safe_channels",
        "v20_operator_canonical_labels",
    }
    results = []
    for family in families:
        example = next(item for item in examples if item.family == family)
        result = compute_score(
            data_source="critpt_synth_code",
            solution_str="```python\ndef answer():\n    return None\n```",
            ground_truth=example.target_code,
            extra_info=_extra_info(example),
        )
        results.append(result)

    v21_keys = {key for key in results[0] if key.startswith("v21_")}
    assert v21_keys
    assert all({key for key in result if key.startswith("v21_")} == v21_keys for result in results)
