from __future__ import annotations

from collections import Counter

from rl_posttrain.critpt_synth.e4_official_style import (
    generate_e4_official_style_examples,
    verify_e4_example,
)


def test_e4_generates_official_shells_with_reference_outputs() -> None:
    examples = generate_e4_official_style_examples(size=48, seed=20260628, split="train")

    assert len(examples) == 48
    assert all("# Problem setup:" in example.prompt for example in examples)
    assert all("# Main problem:" in example.prompt for example in examples)
    assert all("### Parsing template:" in example.prompt for example in examples)
    assert all("def answer" in example.target_code for example in examples)
    assert all(example.metadata.get("reference_output") for example in examples)
    assert all(example.metadata.get("uses_official_prompt") is False for example in examples)

    failures = []
    for example in examples:
        ok, reason = verify_e4_example(example)
        if not ok:
            failures.append((example.problem_id, reason))
    assert failures == []


def test_e4_covers_official_return_shapes() -> None:
    examples = generate_e4_official_style_examples(size=160, seed=20260629, split="train")
    families = Counter(example.family for example in examples)
    answer_types = Counter(str(example.metadata.get("answer_type", "")) for example in examples)

    assert len(families) >= 8
    assert answer_types["list_float"] > 0
    assert answer_types["sympy_expr"] > 0
    assert answer_types["tuple_sympy_str_str"] > 0
    assert answer_types["set_tuple_int_int"] > 0
    assert answer_types["set_str"] > 0
    assert answer_types["set_sympy_expr"] > 0
