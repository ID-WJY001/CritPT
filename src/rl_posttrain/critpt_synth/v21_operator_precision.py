from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace

from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.critpt_synth.v20_focused_hard import (
    _empty_interval_filter,
    _hhg_oam_safe_channels,
    _operator_canonical_labels,
    _wrap_v20_long,
    verify_v20_example,
)


def generate_v21_operator_precision_examples(
    size: int,
    seed: int,
    split: str,
) -> list[SyntheticCritPTExample]:
    """V21 data focused on precise operator filtering, not broad enumeration."""

    rng = random.Random(seed + 2100)
    specs = [
        (_operator_canonical_labels, 0.80, (["medium", "hard"], [0.10, 0.90])),
        (_empty_interval_filter, 0.12, (["medium", "hard"], [0.25, 0.75])),
        (_hhg_oam_safe_channels, 0.08, (["medium", "hard"], [0.50, 0.50])),
    ]
    weights = [item[1] for item in specs]
    examples: list[SyntheticCritPTExample] = []
    seen: set[str] = set()
    attempts = 0
    while len(examples) < size:
        attempts += 1
        if attempts > max(size * 250, 250):
            raise RuntimeError(f"too many duplicate V21 examples while building {split}")
        idx = len(examples)
        generator, _weight, difficulty_spec = specs[rng.choices(range(len(specs)), weights=weights, k=1)[0]]
        difficulties, difficulty_weights = difficulty_spec
        difficulty = rng.choices(difficulties, weights=difficulty_weights, k=1)[0]
        example = _wrap_v21_precision(_wrap_v20_long(generator(rng, idx, split, difficulty), rng, idx), rng, idx)
        if example.problem_id in seen:
            continue
        seen.add(example.problem_id)
        examples.append(example)
    rng.shuffle(examples)
    return examples


def verify_v21_example(example: SyntheticCritPTExample) -> tuple[bool, str]:
    return verify_v20_example(example)


def summarize_v21_source(example: SyntheticCritPTExample) -> str:
    return str(example.metadata.get("v21_focus") or example.metadata.get("v20_focus") or example.family)


def _wrap_v21_precision(
    example: SyntheticCritPTExample,
    rng: random.Random,
    idx: int,
) -> SyntheticCritPTExample:
    focus = str(example.metadata.get("v20_focus", "focused_hard"))
    operator_note = ""
    if focus == "operator_canonical_label":
        operator_note = """
V21 operator precision guardrail:
- Do not return the whole candidate universe.
- First construct candidates, then apply every charge/filter/order rule.
- Prefer a compact loop or comprehension over a long handwritten label list.
- Canonical powers are tr(name^k), never tr(name)^k.
- Extra plausible labels are wrong even if every individual label looks physical.
"""
    prompt = f"""{operator_note}
{example.prompt}
"""
    payload = {
        "problem_id": example.problem_id,
        "focus": focus,
        "idx": idx,
        "v21_note": bool(operator_note),
        "salt": rng.randint(0, 10**9),
    }
    wrapper_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return replace(
        example,
        problem_id=f"{example.problem_id}_v21precision_{wrapper_hash}",
        prompt=prompt,
        metadata={
            **example.metadata,
            "generator_profile": "v21_operator_precision",
            "v21_wrapper_hash": wrapper_hash,
            "v21_focus": "operator_precision" if focus == "operator_canonical_label" else focus,
            "focused_operator_precision": focus == "operator_canonical_label",
            "uses_official_prompt": False,
            "official_overlap": "none",
        },
    )
