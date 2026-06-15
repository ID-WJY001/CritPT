import json

from rl_posttrain.critpt_synth.generators import (
    generate_examples,
    generate_hardcase_examples,
    generate_v7_compact_examples,
    generate_v7_intermediate_examples,
    generate_v9_trace_examples,
    generate_v10_curriculum_trace_examples,
    generate_v11_template_series_trace_examples,
)
from rl_posttrain.critpt_synth.final_answer_verifier import verify_final_answer
from scripts.data.export_synthetic_judge_verl_parquet import prompt_for_style
from rl_posttrain.critpt_synth.verifier import verify_code_completion


def test_generated_targets_pass_verifier() -> None:
    examples = generate_examples(size=24, seed=1234, split="train")
    assert {example.family for example in examples}
    for example in examples:
        result = verify_code_completion(example.assistant_code_block(), example.verifier)
        assert result.ok, (example.problem_id, result.reason)


def test_hardcase_generated_targets_pass_verifier() -> None:
    examples = generate_hardcase_examples(size=32, seed=20260607, split="train")
    assert {example.family for example in examples} >= {"symbolic", "numeric"}
    assert all(example.metadata.get("generator_profile") == "v6_hardcase" for example in examples)
    for example in examples:
        result = verify_code_completion(example.assistant_code_block(), example.verifier)
        assert result.ok, (example.problem_id, result.reason)


def test_v7_intermediate_generated_targets_pass_verifier() -> None:
    examples = generate_v7_intermediate_examples(size=40, seed=20260607, split="train")
    assert {example.family for example in examples} >= {"numeric", "template"}
    assert all(example.metadata.get("generator_profile") == "v7_intermediate" for example in examples)
    assert any(
        check.get("mode") == "numeric_sequence_item"
        for example in examples
        for check in example.verifier.get("checks", [])
    )
    for example in examples:
        result = verify_code_completion(example.assistant_code_block(), example.verifier)
        assert result.ok, (example.problem_id, result.reason)


def test_v7_compact_generated_targets_pass_verifier() -> None:
    examples = generate_v7_compact_examples(size=40, seed=20260608, split="train")
    assert {example.family for example in examples} >= {"numeric", "template"}
    assert all(example.metadata.get("generator_profile") == "v7_compact" for example in examples)
    assert max(len(example.verifier.get("checks", [])) for example in examples) <= 10
    for example in examples:
        result = verify_code_completion(example.assistant_code_block(), example.verifier)
        assert result.ok, (example.problem_id, result.reason)


def test_v9_trace_generated_targets_and_reward_checks() -> None:
    examples = generate_v9_trace_examples(size=48, seed=20260609, split="train")
    assert {example.family for example in examples} >= {"numeric", "template", "discrete"}
    assert all(example.metadata.get("generator_profile") == "v9_trace" for example in examples)
    assert all(example.verifier.get("reward_checks") for example in examples)
    assert any(
        check.get("mode") == "text_numeric"
        for example in examples
        for check in example.verifier.get("reward_checks", [])
    )
    for example in examples:
        code_result = verify_code_completion(example.assistant_code_block(), example.verifier)
        assert code_result.ok, (example.problem_id, code_result.reason)

        prompt = prompt_for_style(example, "audit_trace")
        assert "审计：" in prompt
        for tag in example.metadata.get("audit_tags", []):
            assert f"{tag}=..." in prompt

        tags = example.metadata["audit_tags"]
        expected = {
            check["tag"]: check["expected"]
            for check in example.verifier["reward_checks"]
            if check.get("mode") == "text_numeric"
        }
        audit_line = "审计：" + ", ".join(f"{tag}={expected[tag]}" for tag in tags)
        final_values = [
            check["expected"]
            for check in example.verifier["checks"]
            if check.get("mode") == "numeric_sequence_item"
        ]
        result = verify_final_answer(f"{audit_line}\n最终答案：{final_values}", example.verifier)
        assert result.ok, (example.problem_id, result.reason)


def test_v10_curriculum_trace_generated_targets_and_reward_checks() -> None:
    examples = generate_v10_curriculum_trace_examples(size=60, seed=20260610, split="train")
    assert {example.family for example in examples} >= {"numeric", "template", "discrete"}
    assert all(example.metadata.get("generator_profile") == "v10_curriculum_trace" for example in examples)
    assert all(example.difficulty in {"easy", "medium"} for example in examples)
    assert all(example.verifier.get("reward_checks") for example in examples)
    for example in examples:
        code_result = verify_code_completion(example.assistant_code_block(), example.verifier)
        assert code_result.ok, (example.problem_id, code_result.reason)

        prompt = prompt_for_style(example, "audit_trace")
        assert "审计：" in prompt
        for tag in example.metadata.get("audit_tags", []):
            assert f"{tag}=..." in prompt

        tags = example.metadata["audit_tags"]
        expected = {
            check["tag"]: check["expected"]
            for check in example.verifier["reward_checks"]
            if check.get("mode") == "text_numeric"
        }
        audit_line = "审计：" + ", ".join(f"{tag}={expected[tag]}" for tag in tags)
        final_values = [
            check["expected"]
            for check in example.verifier["checks"]
            if check.get("mode") == "numeric_sequence_item"
        ]
        result = verify_final_answer(f"{audit_line}\n最终答案：{final_values}", example.verifier)
        assert result.ok, (example.problem_id, result.reason)


def test_v11_template_series_trace_generated_targets_and_reward_checks() -> None:
    examples = generate_v11_template_series_trace_examples(size=72, seed=20260611, split="train")
    assert {example.family for example in examples} == {"template"}
    assert all(example.metadata.get("generator_profile") == "v11_template_series_trace" for example in examples)
    assert all(example.difficulty in {"easy", "medium"} for example in examples)
    assert all(example.verifier.get("reward_checks") for example in examples)
    assert any("c2" in example.metadata.get("audit_tags", []) for example in examples)
    assert any("ratio" in example.metadata.get("audit_tags", []) for example in examples)
    for example in examples:
        code_result = verify_code_completion(example.assistant_code_block(), example.verifier)
        assert code_result.ok, (example.problem_id, code_result.reason)

        prompt = prompt_for_style(example, "audit_trace")
        assert "审计：" in prompt
        for tag in example.metadata.get("audit_tags", []):
            assert f"{tag}=..." in prompt

        tags = example.metadata["audit_tags"]
        expected = {
            check["tag"]: check["expected"]
            for check in example.verifier["reward_checks"]
            if check.get("mode") == "text_numeric"
        }
        audit_line = "审计：" + ", ".join(f"{tag}={expected[tag]}" for tag in tags)
        final_values = [
            check["expected"]
            for check in example.verifier["checks"]
            if check.get("mode") == "numeric_sequence_item"
        ]
        result = verify_final_answer(f"{audit_line}\n最终答案：{final_values}", example.verifier)
        assert result.ok, (example.problem_id, result.reason)


def test_sft_row_is_json_serializable() -> None:
    example = generate_examples(size=1, seed=4321, split="train")[0]
    json.dumps(example.to_sft_row(), ensure_ascii=False)


def test_verifier_blocks_unsafe_import() -> None:
    verifier = {"checks": [{"mode": "exact", "expected": 1}]}
    result = verify_code_completion(
        "```python\ndef answer():\n    import os\n    return 1\n```",
        verifier,
    )
    assert not result.ok
    assert "unsafe_import" in result.reason
