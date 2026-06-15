from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from rl_posttrain.critpt_synth.e4_official_style import (
    generate_e4_official_style_examples,
    reference_output_for_code,
)
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.critpt_synth.v20_focused_hard import generate_v20_focused_hard_examples
from rl_posttrain.critpt_synth.v21_operator_precision import generate_v21_operator_precision_examples


ExampleGenerator = Callable[[int, int, str], list[SyntheticCritPTExample]]

E5_PROFILE = "e5_failure_aware_final_answer_judge"


def generate_e5_failure_aware_examples(
    size: int,
    seed: int,
    split: str,
) -> list[SyntheticCritPTExample]:
    """Build E5 data from official-shell examples plus failure-focused V-series cases."""

    rng = random.Random(seed + 5000)
    sources: list[tuple[str, ExampleGenerator, float]] = [
        ("e4_official_shell", generate_e4_official_style_examples, 0.42),
        ("v20_focused_hard", generate_v20_focused_hard_examples, 0.28),
        ("v21_operator_precision", generate_v21_operator_precision_examples, 0.30),
    ]
    pools = {
        name: generator(max(size, 64), seed + offset * 997, split)
        for offset, (name, generator, _weight) in enumerate(sources)
    }
    weights = [weight for _name, _generator, weight in sources]
    names = [name for name, _generator, _weight in sources]
    cursors = {name: 0 for name in names}

    examples: list[SyntheticCritPTExample] = []
    seen: set[str] = set()
    attempts = 0
    while len(examples) < size:
        attempts += 1
        if attempts > max(size * 300, 300):
            raise RuntimeError(f"too many duplicate E5 examples while building {split}")
        source = rng.choices(names, weights=weights, k=1)[0]
        pool = pools[source]
        cursor = cursors[source]
        cursors[source] += 1
        base = pool[cursor % len(pool)]
        example = adapt_example_for_e5(base, source=source, index=len(examples), seed=seed)
        if example.problem_id in seen:
            continue
        seen.add(example.problem_id)
        examples.append(example)
    rng.shuffle(examples)
    return examples


def adapt_example_for_e5(
    example: SyntheticCritPTExample,
    *,
    source: str,
    index: int,
    seed: int,
) -> SyntheticCritPTExample:
    metadata = dict(example.metadata)
    call_args = [str(item) for item in metadata.get("reference_call_args", [])]
    reference_output = str(metadata.get("reference_output") or reference_output_for_code(example.target_code, call_args))
    focus = infer_e5_focus(example)
    guardrail = guardrail_for_focus(focus, example)
    prompt = f"""# E5 failure-aware guardrail:
{guardrail}

{example.prompt.strip()}
"""
    digest = hashlib.sha256(
        json.dumps(
            {
                "source": source,
                "index": index,
                "seed": seed,
                "base_problem_id": example.problem_id,
                "reference_output": reference_output,
                "focus": focus,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    metadata.update(
        {
            "generator_profile": E5_PROFILE,
            "e5_source": source,
            "e5_focus": focus,
            "e5_wrapper_hash": digest,
            "e5_failure_aware": True,
            "borrowed_v19_method": "mine visible rollout failures, then train targeted synthetic analogues",
            "borrowed_v20_v21_cases": source in {"v20_focused_hard", "v21_operator_precision"},
            "reference_output": reference_output,
            "reference_call_args": call_args,
            "official_overlap": "none",
            "uses_official_prompt": False,
            "reward_uses_llm_judge_only": True,
        }
    )
    verifier = {
        "kind": "e5_reference_output",
        "reference_output": reference_output,
        "reference_call_args": call_args,
        "source_verifier": example.verifier,
    }
    return replace(
        example,
        problem_id=f"{example.split}_e5_{source}_{index:05d}_{digest}",
        prompt=prompt,
        verifier=verifier,
        family=f"e5_{example.family}",
        metadata=metadata,
    )


def infer_e5_focus(example: SyntheticCritPTExample) -> str:
    raw = " ".join(
        [
            example.family,
            str(example.metadata.get("v21_focus", "")),
            str(example.metadata.get("v20_focus", "")),
            str(example.metadata.get("answer_type", "")),
            str(example.metadata.get("domain", "")),
        ]
    ).lower()
    if "operator" in raw or "canonical" in raw:
        return "operator_canonical_precision"
    if "empty" in raw or "filter" in raw or "set" in raw:
        return "exact_filter_or_empty_set"
    if "hhg" in raw or "oam" in raw or "helicity" in raw:
        return "hhg_oam_channel_scan"
    if "coeff" in raw or "list_float" in raw:
        return "nonplaceholder_coefficient_list"
    if "choice" in raw:
        return "symbolic_choice_exactness"
    if "sympy" in raw or "expr" in raw or "generating" in raw:
        return "symbolic_expression_exactness"
    return "general_final_answer_exactness"


def guardrail_for_focus(focus: str, example: SyntheticCritPTExample) -> str:
    base = [
        "The value returned by answer() is what gets judged.",
        "Do the small computation from the prompt; do not fill a plausible template with default constants.",
        "Return exactly one Python code block with executable def answer().",
    ]
    if focus == "operator_canonical_precision":
        base.extend(
            [
                "For operator labels, exact strings are part of the answer.",
                "Apply every charge/filter/order rule; do not return the whole candidate universe.",
                "Canonical powers are tr(name^k), not tr(name)^k, and concatenated labels are wrong.",
            ]
        )
    elif focus == "exact_filter_or_empty_set":
        base.extend(
            [
                "For finite filters, apply every condition literally.",
                "A true empty survivor set must be returned as set(); do not pad with near misses.",
                "Supersets and rejected labels are wrong even if they look close.",
            ]
        )
    elif focus == "hhg_oam_channel_scan":
        base.extend(
            [
                "Compute each listed channel in order.",
                "Keep harmonic order, OAM, and helicity separate; sign(0) is 0.",
                "Avoid clever one-liners that hide a variable-name mistake.",
            ]
        )
    elif focus == "nonplaceholder_coefficient_list":
        base.extend(
            [
                "For coefficient lists, calculate every entry in the stated basis order.",
                "All-zero or repeated placeholder lists should be used only if the rules force them.",
                "Ordering errors are correctness errors.",
            ]
        )
    elif focus == "symbolic_choice_exactness":
        base.extend(
            [
                "For multiple-choice symbolic fields, the chosen letter and expression must both match.",
                "Do not guess a letter from the shape of the options.",
            ]
        )
    elif focus == "symbolic_expression_exactness":
        base.extend(
            [
                "For symbolic answers, algebraically equivalent expressions are fine, but missing factors or wrong variables are not.",
                "Use the symbols requested by the template.",
            ]
        )
    else:
        base.append("Exact numeric, symbolic, ordering, label, and return-type details matter.")
    answer_type = example.metadata.get("answer_type")
    if answer_type:
        base.append(f"Expected answer type: {answer_type}.")
    return "\n".join(f"- {line}" for line in base)


def verify_e5_example(example: SyntheticCritPTExample) -> tuple[bool, str]:
    try:
        call_args = [str(item) for item in example.metadata.get("reference_call_args", [])]
        got = reference_output_for_code(example.target_code, call_args)
    except Exception as exc:  # noqa: BLE001 - generated trusted target smoke.
        return False, f"reference_exec_failed: {type(exc).__name__}: {exc}"
    expected = str(example.metadata.get("reference_output", ""))
    if got != expected:
        return False, f"reference_output_mismatch: got={got}, expected={expected}"
    if "def answer" not in example.target_code:
        return False, "missing answer function"
    return True, "ok"
