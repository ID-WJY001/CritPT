from __future__ import annotations

from collections import Counter

from rl_posttrain.critpt_synth.v14_compact_exec import (
    generate_v14_compact_exec_examples,
    summarize_v14_source,
    verify_v14_example,
)


def test_v14_generates_verified_compact_hardcases() -> None:
    examples = generate_v14_compact_exec_examples(size=48, seed=20260614, split="train")

    assert len(examples) == 48
    assert all("# Problem setup:" in example.prompt for example in examples)
    assert all("Compactness constraints" in example.prompt for example in examples)
    assert all("def answer" in example.target_code for example in examples)

    failures = []
    for example in examples:
        ok, reason = verify_v14_example(example)
        if not ok:
            failures.append((example.problem_id, reason))
    assert failures == []


def test_v14_mixes_v13_base_and_new_hardcases() -> None:
    examples = generate_v14_compact_exec_examples(size=80, seed=20260615, split="train")
    sources = Counter(summarize_v14_source(example) for example in examples)
    families = Counter(example.family for example in examples)

    assert sources["v13_base"] > 0
    assert sources["v14_hardcase"] > 0
    assert families["official_failure_sparse_coefficients"] > 0
    assert families["official_failure_bns_compact_set"] > 0
    assert families["official_failure_operator_compact_set"] > 0
