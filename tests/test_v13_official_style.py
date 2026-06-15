from __future__ import annotations

from collections import Counter

from rl_posttrain.critpt_synth.v13_official_style import (
    generate_v13_official_style_examples,
    verify_v13_example,
)


def test_v13_generates_official_style_prompts_and_verified_targets() -> None:
    examples = generate_v13_official_style_examples(size=32, seed=20260613, split="train")

    assert len(examples) == 32
    assert all("# Problem setup:" in example.prompt for example in examples)
    assert all("# Main problem:" in example.prompt for example in examples)
    assert all("### Parsing template:" in example.prompt for example in examples)
    assert all("complete answer() function" in example.prompt for example in examples)
    assert all("def answer" in example.target_code for example in examples)

    failures = []
    for example in examples:
        ok, reason = verify_v13_example(example)
        if not ok:
            failures.append((example.problem_id, reason))
    assert failures == []


def test_v13_covers_multiple_domains_and_answer_types() -> None:
    examples = generate_v13_official_style_examples(size=80, seed=20260614, split="train")
    domains = Counter(str(example.metadata.get("domain")) for example in examples)
    answer_types = Counter(str(example.metadata.get("answer_type")) for example in examples)

    assert len(domains) >= 6
    assert answer_types["symbolic"] > 0
    assert answer_types["numeric_sequence"] > 0
    assert answer_types["string_list"] > 0
    assert answer_types["string_tuple"] > 0
    assert all(example.metadata.get("uses_official_prompt") is False for example in examples)
