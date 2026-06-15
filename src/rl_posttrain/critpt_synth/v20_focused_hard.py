from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from rl_posttrain.critpt_synth.prompting import render_prompt
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.critpt_synth.verifier import verify_code_completion


GeneratorFn = Callable[[random.Random, int, str, str], SyntheticCritPTExample]


def generate_v20_focused_hard_examples(
    size: int,
    seed: int,
    split: str,
) -> list[SyntheticCritPTExample]:
    """Focused hard cases from V19 visible rollout failures."""

    rng = random.Random(seed + 2000)
    specs: list[tuple[GeneratorFn, float, tuple[list[str], list[float]]]] = [
        (_operator_canonical_labels, 0.55, (["medium", "hard"], [0.25, 0.75])),
        (_empty_interval_filter, 0.25, (["medium", "hard"], [0.30, 0.70])),
        (_hhg_oam_safe_channels, 0.20, (["medium", "hard"], [0.55, 0.45])),
    ]
    weights = [item[1] for item in specs]
    examples: list[SyntheticCritPTExample] = []
    seen: set[str] = set()
    attempts = 0
    while len(examples) < size:
        attempts += 1
        if attempts > max(size * 200, 200):
            raise RuntimeError(f"too many duplicate V20 examples while building {split}")
        idx = len(examples)
        spec_idx = rng.choices(range(len(specs)), weights=weights, k=1)[0]
        generator, _weight, difficulty_spec = specs[spec_idx]
        difficulties, difficulty_weights = difficulty_spec
        difficulty = rng.choices(difficulties, weights=difficulty_weights, k=1)[0]
        example = _wrap_v20_long(generator(rng, idx, split, difficulty), rng, idx)
        if example.problem_id in seen:
            continue
        seen.add(example.problem_id)
        examples.append(example)
    rng.shuffle(examples)
    return examples


def verify_v20_example(example: SyntheticCritPTExample) -> tuple[bool, str]:
    result = verify_code_completion(example.assistant_code_block(), example.verifier)
    return result.ok, result.reason


def summarize_v20_source(example: SyntheticCritPTExample) -> str:
    source = example.metadata.get("v20_focus")
    return str(source or example.family)


def _example(
    *,
    problem_id: str,
    split: str,
    family: str,
    difficulty: str,
    setup: str,
    main: str,
    template: str,
    target: str,
    verifier: dict[str, Any],
    solution_trace: str,
    metadata: dict[str, Any],
) -> SyntheticCritPTExample:
    prompt = render_prompt(setup, main, template)
    payload = {
        "problem_id": problem_id,
        "family": family,
        "difficulty": difficulty,
        "metadata": metadata,
    }
    param_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return SyntheticCritPTExample(
        problem_id=problem_id,
        prompt=prompt,
        code_template=template.strip(),
        target_code=target.strip(),
        verifier=verifier,
        split=split,
        family=family,
        difficulty=difficulty,
        solution_trace=solution_trace,
        metadata={
            **metadata,
            "generator_profile": "v20_focused_hard",
            "param_hash": param_hash,
            "official_overlap": "none",
            "uses_official_prompt": False,
            "prompt_style": "problem_setup_main_problem_parsing_template_v20_focused",
            "anti_runaway": True,
            "no_think_target": True,
        },
    )


def _op_notation(name: str, power: int) -> str:
    return f"tr({name})" if power == 1 else f"tr({name}^{power})"


def _operator_canonical_labels(
    rng: random.Random,
    idx: int,
    split: str,
    difficulty: str,
) -> SyntheticCritPTExample:
    field_pool = [("psi", 1), ("dpsi", 2), ("chi", 2), ("F", 3), ("B", 4)]
    rng.shuffle(field_pool)
    fields = sorted(field_pool[: rng.choice([4, 5] if difficulty == "hard" else [3, 4])], key=lambda x: x[0])
    max_charge = rng.choice([5, 6, 7] if difficulty == "hard" else [4, 5, 6])
    power_limit = rng.choice([3, 4])
    include_mixed = True
    include_products = difficulty == "hard" and rng.random() < 0.45

    items: list[tuple[int, int, str]] = []
    for name, charge in fields:
        for power in range(1, power_limit + 1):
            total = charge * power
            if total <= max_charge:
                items.append((total, 0, _op_notation(name, power)))
    for left_idx, (left, left_charge) in enumerate(fields):
        for right, right_charge in fields[left_idx + 1 :]:
            total = left_charge + right_charge
            if total <= max_charge:
                items.append((total, 1, f"tr({left} {right})"))
    if include_products:
        singles = [(total, label) for total, kind, label in items if kind == 0]
        for left_idx, (left_charge, left_label) in enumerate(singles):
            for right_charge, right_label in singles[left_idx + 1 :]:
                total = left_charge + right_charge
                if total <= max_charge + 1:
                    items.append((total, 2, f"{left_label} {right_label}"))

    expected = [label for _total, _kind, label in sorted(items, key=lambda item: (item[0], item[1], item[2]))]
    setup = f"""
A toy gauge-theory operator notebook uses canonical string labels. The allowed
letters and charges are:

{fields!r}

Rules:
- Single-trace powers must be written exactly as tr(name) or tr(name^k).
- Mixed single traces must be written exactly as tr(A B), with A before B in
  alphabetical field order. Do not concatenate labels such as psipsi or Fpsi.
- Include a candidate only if its total charge is <= {max_charge}.
- Single-trace powers use 1 <= k <= {power_limit}.
- Mixed two-field traces are enabled.
- Products of two single traces are {"enabled" if include_products else "disabled"}.
  If enabled, write them as "tr(A) tr(B)" using the already canonical factors.

Sort by increasing total charge, then candidate kind where powers come before
mixed traces and mixed traces before products, then lexicographic label.
"""
    main = "Return the exact ordered list of canonical operator labels."
    template = """
def answer():
    r\"\"\"
    Return allowed operator labels in canonical notation.
    \"\"\"
    operators = ...
    return operators
"""
    target = f"""
def answer():
    fields = {fields!r}
    max_charge = {max_charge}
    power_limit = {power_limit}
    include_products = {include_products!r}

    def notation(name, power):
        return f"tr({{name}})" if power == 1 else f"tr({{name}}^{{power}})"

    items = []
    for name, charge in fields:
        for power in range(1, power_limit + 1):
            total = charge * power
            if total <= max_charge:
                items.append((total, 0, notation(name, power)))
    for left_idx, (left, left_charge) in enumerate(fields):
        for right, right_charge in fields[left_idx + 1:]:
            total = left_charge + right_charge
            if total <= max_charge:
                items.append((total, 1, f"tr({{left}} {{right}})"))
    if include_products:
        singles = [(total, label) for total, kind, label in items if kind == 0]
        for left_idx, (left_charge, left_label) in enumerate(singles):
            for right_charge, right_label in singles[left_idx + 1:]:
                total = left_charge + right_charge
                if total <= max_charge + 1:
                    items.append((total, 2, f"{{left_label}} {{right_label}}"))
    return [label for _total, _kind, label in sorted(items, key=lambda item: (item[0], item[1], item[2]))]
"""
    return _example(
        problem_id=f"{split}_v20_operator_canonical_{idx:05d}_{max_charge}_{power_limit}_{len(fields)}_{int(include_products)}",
        split=split,
        family="v20_operator_canonical_labels",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "exact_sequence", "expected": expected}]},
        solution_trace="Enumerate only canonical labels; never concatenate field names outside tr(...).",
        metadata={
            "domain": "gauge_theory_operator_counting",
            "answer_type": "ordered_string_list",
            "v20_focus": "operator_canonical_label",
            "expected_compact_chars": len(target),
        },
    )


def _empty_interval_filter(
    rng: random.Random,
    idx: int,
    split: str,
    difficulty: str,
) -> SyntheticCritPTExample:
    prefix = str(rng.choice([117, 153, 191, 221, 307, 419]))
    start = rng.choice([12, 18, 21, 24, 30, 32])
    stop = start + rng.choice([18, 24, 30])
    parity = rng.choice([0, 1])
    modulo = rng.choice([3, 4, 5])
    residue = rng.randrange(modulo)
    candidates = [x for x in range(start, stop + 1) if x % 2 == parity and x % modulo == residue]
    force_empty = difficulty == "hard" or rng.random() < 0.65
    if force_empty:
        forbidden = set(candidates)
    else:
        forbidden = set(rng.sample(candidates, k=max(0, len(candidates) - rng.choice([1, 2]))))
    expected = {f"{prefix}.{x}" for x in candidates if x not in forbidden}

    setup = f"""
A compact detector-window filter uses string labels of the form "{prefix}.suffix".
The suffix starts as every integer in the closed interval [{start}, {stop}].

Keep a suffix only if:
1. suffix % 2 == {parity}
2. suffix % {modulo} == {residue}
3. suffix is not in this forbidden set: {sorted(forbidden)!r}

Return a Python set of labels. If no suffix survives, return the empty set
set(); do not pad with near misses and do not return a list of rejected labels.
"""
    main = "Apply all filters literally and return the surviving label set."
    template = """
def answer():
    r\"\"\"
    Return a set like {"prefix.12", ...}; return set() if empty.
    \"\"\"
    labels = ...
    return labels
"""
    target = f"""
def answer():
    prefix = {prefix!r}
    forbidden = {sorted(forbidden)!r}
    labels = {{
        f"{{prefix}}.{{suffix}}"
        for suffix in range({start}, {stop + 1})
        if suffix % 2 == {parity}
        and suffix % {modulo} == {residue}
        and suffix not in forbidden
    }}
    return labels
"""
    return _example(
        problem_id=f"{split}_v20_empty_interval_{idx:05d}_{prefix}_{start}_{stop}_{parity}_{modulo}_{residue}",
        split=split,
        family="v20_empty_interval_filter",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "set_exact", "expected": sorted(expected)}]},
        solution_trace=f"Surviving labels are {sorted(expected)!r}. Empty={not expected}.",
        metadata={
            "domain": "binary_neutron_star_interval_filter",
            "answer_type": "compact_string_set",
            "v20_focus": "empty_interval_filter",
            "expected_empty": not expected,
            "expected_compact_chars": len(target),
        },
    )


def _hhg_oam_safe_channels(
    rng: random.Random,
    idx: int,
    split: str,
    difficulty: str,
) -> SyntheticCritPTExample:
    pulses = [
        ("A", rng.choice([-2, -1, 1, 2]), rng.choice([-1, 1])),
        ("B", rng.choice([-3, -1, 2, 3]), rng.choice([-1, 1])),
        ("C", rng.choice([-2, 1, 2, 4]), rng.choice([-1, 1])),
    ]
    channel_count = 5 if difficulty == "medium" else 7
    channels: list[tuple[int, int, int]] = []
    while len(channels) < channel_count:
        channel = tuple(rng.randint(0, 4) for _ in range(3))
        if sum(channel) == 0 or channel in channels:
            continue
        channels.append(channel)

    expected: list[list[int]] = []
    ell_sigma = [(ell, sigma) for _name, ell, sigma in pulses]
    for counts in channels:
        order = sum(counts)
        oam = sum(count * ell for count, (ell, _sigma) in zip(counts, ell_sigma))
        helicity_sum = sum(count * sigma for count, (_ell, sigma) in zip(counts, ell_sigma))
        helicity = 1 if helicity_sum > 0 else -1 if helicity_sum < 0 else 0
        expected.append([order, oam, helicity])

    setup = f"""
A structured-light HHG toy cell lists pulses as (name, orbital_angular_momentum,
helicity):

{pulses!r}

For each channel (nA, nB, nC):
- harmonic_order = nA + nB + nC
- emitted_oam = nA*ell_A + nB*ell_B + nC*ell_C
- emitted_helicity = sign(nA*sigma_A + nB*sigma_B + nC*sigma_C), using 0 only
  when the sum is exactly zero.

Channels, in order:
{channels!r}

Do not use assignment expressions or clever one-liners; clear local variables
are safer here.
"""
    main = "Return one list of [harmonic_order, emitted_oam, emitted_helicity] triples in the same order."
    template = """
def answer():
    r\"\"\"
    Return [[order, oam, helicity], ...].
    \"\"\"
    results = ...
    return results
"""
    target = f"""
def answer():
    pulses = {[(ell, sigma) for _name, ell, sigma in pulses]!r}
    channels = {channels!r}
    results = []
    for counts in channels:
        order = sum(counts)
        oam = sum(count * ell for count, (ell, _sigma) in zip(counts, pulses))
        helicity_sum = sum(count * sigma for count, (_ell, sigma) in zip(counts, pulses))
        helicity = 1 if helicity_sum > 0 else -1 if helicity_sum < 0 else 0
        results.append([order, oam, helicity])
    return results
"""
    verifier = {"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "exact_sequence", "expected": expected}]}
    return _example(
        problem_id=f"{split}_v20_hhg_oam_safe_{idx:05d}_{channel_count}",
        split=split,
        family="v20_hhg_oam_safe_channels",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier=verifier,
        solution_trace=f"pulses={pulses!r}; channels={channels!r}; expected={expected!r}.",
        metadata={
            "domain": "high_harmonic_structured_light",
            "answer_type": "list_of_integer_triples",
            "v20_focus": "hhg_oam_runtime_safe",
            "expected_compact_chars": len(target),
        },
    )


def _wrap_v20_long(
    example: SyntheticCritPTExample,
    rng: random.Random,
    idx: int,
) -> SyntheticCritPTExample:
    focus = str(example.metadata.get("v20_focus", "focused_hard"))
    preamble = rng.choice(_PREAMBLES)
    trap_note = {
        "operator_canonical_label": "The exact printed operator label is part of the answer; psipsi and tr(psi^2) are not equivalent strings.",
        "empty_interval_filter": "An empty survivor set is a valid final answer; do not fill it with rejected candidates.",
        "hhg_oam_runtime_safe": "Use clear variables for OAM and helicity; avoid clever expressions that shadow variable names.",
    }.get(focus, "Use the embedded parsing template as the source of truth.")
    prompt = f"""# Problem setup:
{preamble}

The surrounding physics prose is context. The embedded benchmark cell is the
contract for the answer. Read the filters and output representation literally.

V20 focused guardrail:
- {trap_note}
- Return exactly one closed Python code block.
- Keep `def answer()` executable.
- Do not output <think> tags or prose.

### Embedded benchmark cell:

{example.prompt}
"""
    payload = {
        "problem_id": example.problem_id,
        "focus": focus,
        "idx": idx,
        "preamble": preamble,
    }
    wrapper_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return replace(
        example,
        problem_id=f"{example.problem_id}_v20long_{wrapper_hash}",
        prompt=prompt,
        metadata={
            **example.metadata,
            "generator_profile": "v20_focused_hard_long",
            "v20_wrapper_hash": wrapper_hash,
            "long_context_training": True,
            "uses_official_prompt": False,
            "official_overlap": "none",
        },
    )


_PREAMBLES = [
    """
In a symbolic post-processing notebook, most failures come not from syntax but
from small representation mismatches: a canonical string, an empty survivor set,
or a sign convention must be returned exactly.
""",
    """
The public benchmark parser evaluates the value returned by answer(). It does
not interpret explanatory text, and it does not repair near-miss labels.
""",
    """
Physics notation in the notebook can be verbose. In this reduced task, every
constant needed for the answer appears in the embedded cell below.
""",
]
