from __future__ import annotations

from rl_posttrain.critpt_synth.e1_llm_wrapped import (
    build_llm_wrapper_messages,
    e1_type_names,
    generate_e1_examples,
    verify_e1_example,
)


def test_e1_generates_all_planned_types() -> None:
    examples = generate_e1_examples(14, 20260622, "train")
    got = [example.metadata["e1_type"] for example in examples]

    assert got == e1_type_names()
    assert len(set(got)) == 14


def test_e1_gold_targets_pass_verifier() -> None:
    examples = generate_e1_examples(28, 20260622, "train")

    failures = []
    for example in examples:
        ok, reason = verify_e1_example(example)
        if not ok:
            failures.append((example.problem_id, reason))

    assert failures == []


def test_e1_examples_record_core_facts_and_no_official_prompt_use() -> None:
    example = generate_e1_examples(1, 20260622, "train")[0]

    assert "core_facts" in example.metadata
    assert example.metadata["uses_official_prompt"] is False
    assert example.metadata["official_overlap"] == "none"
    assert "### Parsing template:" in example.prompt


def test_e1_llm_wrapper_prompt_restricts_answer_and_fact_changes() -> None:
    example_core = generate_e1_examples(1, 20260622, "train")[0]
    # The persisted example exposes enough metadata to identify the policy, but
    # the direct message builder operates on cores. Regenerate via the public
    # type order and inspect the wrapper text indirectly through a fresh core.
    from rl_posttrain.critpt_synth.e1_llm_wrapped import E1_GENERATORS
    import random

    core = E1_GENERATORS[0](random.Random(20260622), 0, "train", "medium")
    messages = build_llm_wrapper_messages(core)
    joined = "\n".join(message["content"] for message in messages)

    assert "must preserve every rule" in joined
    assert "seed_problem_setup" in joined
    assert "Preserve formulas and inequalities exactly" in joined
    assert str(example_core.metadata["e1_type"]) in e1_type_names()


def test_e1_llm_wrapper_accepts_nested_required_json_payload(monkeypatch, tmp_path) -> None:
    from rl_posttrain.critpt_synth.e1_llm_wrapped import E1_GENERATORS, build_example_from_core
    from rl_posttrain.model_judge.openai_compatible import JudgeSettings
    import random

    core = E1_GENERATORS[0](random.Random(20260622), 0, "train", "medium")

    def fake_openai_chat_json(*, settings, messages):
        return {
            "required_json": {
                "problem_setup": "Wrapped setup with unchanged facts.",
                "main_problem": "Return the requested coefficients.",
            }
        }

    monkeypatch.setattr("rl_posttrain.critpt_synth.e1_llm_wrapped.openai_chat_json", fake_openai_chat_json)
    example = build_example_from_core(
        core,
        use_llm=True,
        llm_settings=JudgeSettings(api_key="unit", base_url="http://unit", model="unit"),
        llm_cache_path=str(tmp_path / "cache.sqlite3"),
    )

    assert "Wrapped setup" in example.prompt
    assert example.metadata["llm_background_wrapped"] is True


def test_json_sqlite_cache_allows_parallel_initialization(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from rl_posttrain.model_judge.openai_compatible import JsonSqliteCache

    path = tmp_path / "judge-cache.sqlite3"

    def touch_cache(i: int) -> int:
        cache = JsonSqliteCache(str(path))
        cache.set(f"k-{i}", {"value": i})
        assert cache.get(f"k-{i}") == {"value": i}
        return i

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sorted(executor.map(touch_cache, range(24))) == list(range(24))
