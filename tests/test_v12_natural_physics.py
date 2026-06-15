from __future__ import annotations

from collections import Counter

from rl_posttrain.critpt_synth.final_answer_verifier import verify_final_answer
from rl_posttrain.critpt_synth.v12_natural_physics import (
    generate_v12_natural_physics_examples,
    verify_v12_example,
)


def test_v12_natural_physics_examples_verify_and_have_no_python_prompt() -> None:
    examples = generate_v12_natural_physics_examples(size=80, seed=20260612, split="train")

    assert len(examples) == 80
    assert all("```python" not in example.question for example in examples)
    assert all("def answer" not in example.question for example in examples)
    assert {example.answer_type for example in examples} >= {
        "symbolic",
        "numeric_sequence",
        "exact_sequence",
        "integer",
    }
    assert len({example.domain for example in examples}) >= 6

    for example in examples:
        ok, failures = verify_v12_example(example)
        assert ok, (example.example_id, failures)
        assert example.reference_solution
        assert example.anti_hack_wrong_answers
        assert example.judge_rubric["overall_score"]["cap_if_final_answer_wrong"] == 0.6
        assert example.metadata["front_prompt_contains_python"] is False


def test_v12_hard_split_seed_changes_parameters() -> None:
    train = generate_v12_natural_physics_examples(size=120, seed=20260612, split="train")
    test = generate_v12_natural_physics_examples(size=120, seed=20260614, split="test")

    train_hashes = {example.metadata["param_hash"] for example in train}
    test_hashes = {example.metadata["param_hash"] for example in test}

    assert len(train_hashes & test_hashes) < 40
    assert Counter(example.split for example in train) == {"train": 120}
    assert Counter(example.split for example in test) == {"test": 120}


def test_v12_anti_hack_wrong_answers_get_non_full_scores() -> None:
    examples = generate_v12_natural_physics_examples(size=40, seed=20260615, split="val")

    for example in examples:
        for wrong in example.anti_hack_wrong_answers:
            result = verify_final_answer(f"最终答案：{wrong['answer']}", example.verifier)
            assert not result.ok, (example.example_id, wrong, result.reason)
            assert result.score < 1.0
