from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from rl_posttrain.critpt_synth.prompting import render_prompt
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.critpt_synth.v14_compact_exec import (
    generate_v14_compact_exec_examples,
)
from rl_posttrain.critpt_synth.verifier import verify_code_completion


GeneratorFn = Callable[[random.Random, int, str, str], SyntheticCritPTExample]


def generate_v15_hardmix_examples(
    size: int,
    seed: int,
    split: str,
) -> list[SyntheticCritPTExample]:
    """V15 focuses on V14 step-40 weak spots: operator filters, recurrence, and OAM."""

    rng = random.Random(seed)
    base_size = int(size * 0.25)
    hard_size = size - base_size
    base = [
        _add_v15_instruction(example, source_profile="v14_base_in_v15")
        for example in generate_v14_compact_exec_examples(base_size, seed, split)
    ]
    hard = _generate_hardcases(hard_size, rng, split)
    examples = base + hard
    rng.shuffle(examples)
    return examples


def _add_v15_instruction(
    example: SyntheticCritPTExample,
    *,
    source_profile: str,
) -> SyntheticCritPTExample:
    note = (
        "\n\nV15 hard-mix constraints:\n"
        "- Output exactly one Python code block containing def answer(...).\n"
        "- Do not include <think> tags or prose outside the code block.\n"
        "- Prefer loops, comprehensions, and explicit filters over manual enumeration.\n"
        "- Return only the requested object; do not add near-miss candidates."
    )
    return replace(
        example,
        prompt=example.prompt + note,
        metadata={
            **example.metadata,
            "generator_profile": source_profile,
            "v15_source": "v14_base",
            "anti_runaway": True,
            "no_think_target": True,
        },
    )


def verify_v15_example(example: SyntheticCritPTExample) -> tuple[bool, str]:
    result = verify_code_completion(example.assistant_code_block(), example.verifier)
    return result.ok, result.reason


def summarize_v15_source(example: SyntheticCritPTExample) -> str:
    if str(example.metadata.get("generator_profile")) == "v15_hardmix":
        return "v15_hardcase"
    return "v14_base"


def _generate_hardcases(size: int, rng: random.Random, split: str) -> list[SyntheticCritPTExample]:
    specs: list[tuple[GeneratorFn, float, tuple[list[str], list[float]]]] = [
        (_operator_charge_filter, 0.30, (["medium", "hard"], [0.35, 0.65])),
        (_operator_parity_spin_filter, 0.18, (["medium", "hard"], [0.45, 0.55])),
        (_recurrence_probe_suite, 0.18, (["medium", "hard"], [0.45, 0.55])),
        (_multi_channel_oam, 0.16, (["medium", "hard"], [0.50, 0.50])),
        (_piecewise_kernel_probe, 0.10, (["medium", "hard"], [0.50, 0.50])),
        (_bns_interval_intersection, 0.08, (["medium", "hard"], [0.50, 0.50])),
    ]
    weights = [item[1] for item in specs]
    examples: list[SyntheticCritPTExample] = []
    seen: set[str] = set()
    attempts = 0
    while len(examples) < size:
        attempts += 1
        if attempts > max(size * 120, 120):
            raise RuntimeError(f"too many duplicate V15 examples while building {split}")
        idx = len(examples)
        spec_idx = rng.choices(range(len(specs)), weights=weights, k=1)[0]
        generator, _weight, difficulty_spec = specs[spec_idx]
        difficulties, difficulty_weights = difficulty_spec
        difficulty = rng.choices(difficulties, weights=difficulty_weights, k=1)[0]
        example = generator(rng, idx, split, difficulty)
        if example.problem_id in seen:
            continue
        seen.add(example.problem_id)
        examples.append(example)
    return examples


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
    note = (
        "\n\nV15 hard-mix constraints:\n"
        "- Output exactly one Python code block containing def answer(...).\n"
        "- Do not include <think> tags or prose outside the code block.\n"
        "- Prefer loops, comprehensions, and explicit filters over manual enumeration.\n"
        "- Return only the requested object; do not add near-miss candidates."
    )
    prompt = render_prompt(setup, main, template) + note
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
            "generator_profile": "v15_hardmix",
            "param_hash": param_hash,
            "official_overlap": "none",
            "uses_official_prompt": False,
            "prompt_style": "problem_setup_main_problem_parsing_template_v15_hardmix",
            "anti_runaway": True,
            "no_think_target": True,
        },
    )


def _pow_notation(name: str, power: int) -> str:
    if power == 1:
        return f"tr({name})"
    return f"tr({name}^{power})"


def _operator_charge_filter(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    field_pool = [("psi", 1), ("dpsi", 2), ("F", 3), ("chi", 2)]
    rng.shuffle(field_pool)
    field_count = 3 if difficulty == "medium" else 4
    fields = sorted(field_pool[:field_count], key=lambda item: item[0])
    max_charge = rng.choice([4, 5, 6] if difficulty == "medium" else [5, 6, 7])
    power_limit = rng.choice([3, 4])
    include_mixed = rng.choice([True, False]) if difficulty == "medium" else True

    singles: list[tuple[int, str]] = []
    for name, charge in fields:
        for power in range(1, power_limit + 1):
            total = charge * power
            if total <= max_charge:
                singles.append((total, _pow_notation(name, power)))

    mixed: list[tuple[int, str]] = []
    if include_mixed:
        for left_idx, (left, left_charge) in enumerate(fields):
            for right, right_charge in fields[left_idx + 1 :]:
                total = left_charge + right_charge
                if total <= max_charge:
                    mixed.append((total, f"tr({left} {right})"))

    expected = [name for _charge, name in sorted(singles + mixed, key=lambda item: (item[0], item[1]))]
    setup = f"""
A synthetic large-N operator counter uses these field charges:
{fields!r}

Allowed single traces are tr(field^k) for 1 <= k <= {power_limit} when
k * charge(field) <= {max_charge}. Mixed two-field traces tr(A B) are
{"enabled" if include_mixed else "disabled"}; if enabled, keep only A < B in
alphabetical field order and charge(A)+charge(B) <= {max_charge}.

Return operators sorted by increasing total charge, then lexicographic string.
Do not include products of traces, reversed duplicates, or over-charge terms.
"""
    main = "Return the ordered list of allowed operator strings."
    template = """
def answer():
    r\"\"\"
    Return allowed operators in the required order.
    \"\"\"
    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    operators = ...
    # ---------------------------------------------------------------

    return operators
"""
    target = f"""
def answer():
    r\"\"\"
    Return allowed operators in the required order.
    \"\"\"
    fields = {fields!r}
    max_charge = {max_charge}
    power_limit = {power_limit}
    include_mixed = {include_mixed!r}

    def notation(name, power):
        return f"tr({{name}})" if power == 1 else f"tr({{name}}^{{power}})"

    items = []
    for name, charge in fields:
        for power in range(1, power_limit + 1):
            total = charge * power
            if total <= max_charge:
                items.append((total, notation(name, power)))
    if include_mixed:
        for left_idx, (left, left_charge) in enumerate(fields):
            for right, right_charge in fields[left_idx + 1:]:
                total = left_charge + right_charge
                if total <= max_charge:
                    items.append((total, f"tr({{left}} {{right}})"))
    return [name for _, name in sorted(items, key=lambda item: (item[0], item[1]))]
"""
    return _example(
        problem_id=f"{split}_v15_operator_charge_filter_{idx:05d}_{max_charge}_{power_limit}_{field_count}",
        split=split,
        family="v15_operator_charge_filter",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "exact_sequence", "expected": expected}]},
        solution_trace="Build candidates with their charge, filter exactly, then sort by (charge, string).",
        metadata={
            "domain": "gauge_theory_operator_counting",
            "answer_type": "ordered_string_list",
            "expected_compact_chars": len(target),
        },
    )


def _operator_parity_spin_filter(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    n_max = rng.choice([7, 8] if difficulty == "medium" else [9, 10])
    spin_max = rng.choice([3, 4] if difficulty == "medium" else [4, 5])
    target_parity = rng.choice([0, 1])
    min_twist = rng.choice([2, 3])
    expected = {
        f"O_{n}_{spin}_{parity}"
        for n in range(1, n_max + 1)
        for spin in range(0, spin_max + 1)
        for parity in [0, 1]
        if (n + spin + parity) % 2 == target_parity and n - spin >= min_twist
    }
    setup = f"""
A toy conformal-block table labels candidate operators with the exact string
format O_n_s_p, where n is the operator index, s is the spin value, and p is the
parity bit. For example, n=6, spin=3, parity=0 must be written exactly as
"O_6_3_0". Do not include the literal words "spin" or "parity" in labels.
Scan n=1..{n_max}, spin=0..{spin_max}, parity in {{0,1}}.
Keep exactly those candidates satisfying

  (n + spin + parity) mod 2 = {target_parity}
  n - spin >= {min_twist}

Return a set of strings. Do not include near misses that fail one condition.
"""
    main = "Return the allowed operator label set."
    template = """
def answer():
    r\"\"\"
    Return allowed operator labels.
    \"\"\"
    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    labels = ...
    # ---------------------------------------------------------------

    return labels
"""
    target = f"""
def answer():
    r\"\"\"
    Return allowed operator labels.
    \"\"\"
    labels = {{
        f"O_{{n}}_{{spin}}_{{parity}}"
        for n in range(1, {n_max + 1})
        for spin in range(0, {spin_max + 1})
        for parity in [0, 1]
        if (n + spin + parity) % 2 == {target_parity} and n - spin >= {min_twist}
    }}
    return labels
"""
    return _example(
        problem_id=f"{split}_v15_operator_parity_spin_{idx:05d}_{n_max}_{spin_max}_{target_parity}_{min_twist}",
        split=split,
        family="v15_operator_parity_spin_filter",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "set_exact", "expected": sorted(expected)}]},
        solution_trace="Use a set comprehension over n, spin, and parity with both filters.",
        metadata={
            "domain": "conformal_operator_filter",
            "answer_type": "compact_string_set",
            "expected_compact_chars": len(target),
        },
    )


def _recurrence_probe_suite(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    degree = rng.choice([12, 14, 16] if difficulty == "medium" else [18, 20, 22])
    a0 = rng.randint(-2, 3)
    a1 = rng.randint(-2, 4)
    alpha = rng.choice([-1, 1, 2])
    beta = rng.choice([-2, -1, 1])
    period = rng.choice([3, 4, 5])
    coeffs = [a0, a1]
    for n in range(2, degree + 1):
        coeffs.append(alpha * coeffs[-1] + beta * coeffs[-2] + (n % period) - 1)
    setup = f"""
A toy Hilbert-series numerator has coefficients a_n defined by:

  a_0 = {a0}
  a_1 = {a1}
  a_n = {alpha} a_(n-1) + {beta} a_(n-2) + ((n mod {period}) - 1), for n >= 2

Return the coefficient list [a_0, ..., a_{degree}]. Use the recurrence; do not
guess a closed form and do not stop early.
"""
    main = "Return all coefficients through the requested degree."
    template = """
def answer():
    r\"\"\"
    Return [a_0, ..., a_degree].
    \"\"\"
    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    coeffs = ...
    # ---------------------------------------------------------------

    return coeffs
"""
    target = f"""
def answer():
    r\"\"\"
    Return [a_0, ..., a_degree].
    \"\"\"
    coeffs = [{a0}, {a1}]
    for n in range(2, {degree + 1}):
        coeffs.append({alpha} * coeffs[-1] + {beta} * coeffs[-2] + (n % {period}) - 1)
    return coeffs
"""
    return _example(
        problem_id=f"{split}_v15_recurrence_probe_{idx:05d}_{degree}_{a0}_{a1}_{alpha}_{beta}_{period}",
        split=split,
        family="v15_recurrence_probe_suite",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "exact_sequence", "expected": coeffs}]},
        solution_trace="Initialize a0,a1 and iterate the recurrence through the requested degree.",
        metadata={
            "domain": "hilbert_series_recurrence",
            "answer_type": "integer_sequence",
            "expected_compact_chars": len(target),
        },
    )


def _multi_channel_oam(rng: random.Random, idx: int, split: str, difficulty: str) -> SyntheticCritPTExample:
    pulses = [
        ("A", rng.choice([-2, -1, 1]), rng.choice([-1, 1])),
        ("B", rng.choice([-1, 2, 3]), rng.choice([-1, 1])),
        ("C", rng.choice([1, 2, 4]), rng.choice([-1, 1])),
    ]
    channel_count = rng.choice([3, 4] if difficulty == "medium" else [5, 6])
    channels: list[tuple[int, int, int]] = []
    while len(channels) < channel_count:
        channel = tuple(rng.randint(0, 3) for _ in range(3))
        if sum(channel) > 0 and channel not in channels:
            channels.append(channel)

    expected: list[list[int]] = []
    for counts in channels:
        order = sum(counts)
        oam = sum(count * pulse[1] for count, pulse in zip(counts, pulses, strict=True))
        helicity_sum = sum(count * pulse[2] for count, pulse in zip(counts, pulses, strict=True))
        helicity = 1 if helicity_sum > 0 else -1 if helicity_sum < 0 else 0
        expected.append([order, oam, helicity])

    setup = f"""
A structured-light HHG toy model has three pulses (name, OAM ell, helicity sigma):
{pulses!r}

For each absorption channel (nA, nB, nC), return
[harmonic_order, emitted_oam, emitted_helicity], where harmonic_order is the
sum of absorbed photons, emitted_oam is the count-weighted OAM sum, and emitted
helicity is sign(sum count*sigma), using 0 when the sum is exactly zero.

Channels, in order:
{channels!r}
"""
    main = "Return one list of triples, in the same channel order."
    template = """
def answer():
    r\"\"\"
    Return [[order, oam, helicity], ...] for the listed channels.
    \"\"\"
    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    results = ...
    # ---------------------------------------------------------------

    return results
"""
    target = f"""
def answer():
    r\"\"\"
    Return [[order, oam, helicity], ...] for the listed channels.
    \"\"\"
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
    return _example(
        problem_id=f"{split}_v15_multi_channel_oam_{idx:05d}_{channel_count}_{sum(sum(c) for c in channels)}",
        split=split,
        family="v15_multi_channel_oam",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "exact_sequence", "expected": expected}]},
        solution_trace="Loop over channels and compute order, OAM, and helicity sign independently.",
        metadata={
            "domain": "high_harmonic_structured_light",
            "answer_type": "list_of_integer_triples",
            "expected_compact_chars": len(target),
        },
    )


def _piecewise_kernel_probe(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    left_edge = rng.choice([-2.0, -1.5, -1.0])
    right_edge = rng.choice([1.0, 1.5, 2.0])
    slope_left = rng.choice([-2.0, -1.0, 1.5])
    offset_left = rng.choice([0.5, 1.0, 2.0])
    middle = rng.choice([-0.25, 0.0, 0.75])
    scale_right = rng.choice([1.0, 2.0, 3.0])
    shift_right = rng.choice([-0.5, 0.25, 1.0])
    probes = [left_edge - 0.5, left_edge, (left_edge + right_edge) / 2, right_edge, right_edge + 0.5]

    def expected_value(x: float) -> float:
        if x < left_edge:
            return slope_left * x + offset_left
        if x <= right_edge:
            return middle
        return scale_right / (x + 3.0) + shift_right

    checks = [
        {"mode": "numeric", "args": [x], "expected": expected_value(x), "tolerance": 1e-8}
        for x in probes
    ]
    setup = f"""
A simplified LaMET matching kernel K(x) is defined by the piecewise rule:

  x < {left_edge}: K(x) = {slope_left}*x + {offset_left}
  {left_edge} <= x <= {right_edge}: K(x) = {middle}
  x > {right_edge}: K(x) = {scale_right}/(x + 3) + {shift_right}

Return a Python function answer(x). Be careful about the inclusive middle
interval endpoints.
"""
    main = "Return K(x) for arbitrary numeric x."
    template = """
def answer(x):
    r\"\"\"
    Return the piecewise kernel value K(x).
    \"\"\"
    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    K = ...
    # ---------------------------------------------------------------

    return K
"""
    target = f"""
def answer(x):
    r\"\"\"
    Return the piecewise kernel value K(x).
    \"\"\"
    if x < {left_edge}:
        return {slope_left} * x + {offset_left}
    if x <= {right_edge}:
        return {middle}
    return {scale_right} / (x + 3.0) + {shift_right}
"""
    return _example(
        problem_id=f"{split}_v15_piecewise_kernel_{idx:05d}_{left_edge}_{right_edge}_{scale_right}",
        split=split,
        family="v15_piecewise_kernel_probe",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": checks},
        solution_trace="Implement the three branches with inclusive endpoints in the middle branch.",
        metadata={
            "domain": "lamet_piecewise_kernel",
            "answer_type": "numeric_function",
            "expected_compact_chars": len(target),
        },
    )


def _bns_interval_intersection(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    prefix = rng.choice(["182", "191", "194", "221"])
    a0 = rng.randint(10, 55)
    a1 = a0 + rng.choice([30, 40, 50] if difficulty == "medium" else [55, 65, 75])
    b0 = a0 + rng.randint(5, 18)
    b1 = a1 - rng.randint(3, 15)
    modulus = rng.choice([2, 3, 4])
    residue = rng.randint(0, modulus - 1)
    excluded = sorted(rng.sample(range(b0, b1 + 1), k=min(5 if difficulty == "medium" else 8, b1 - b0 + 1)))
    expected = {
        f"{prefix}.{n}"
        for n in range(max(a0, b0), min(a1, b1) + 1)
        if n % modulus == residue and n not in excluded
    }
    setup = f"""
A magnetic-space-group sieve starts with BNS suffix interval [{a0}, {a1}],
intersects it with [{b0}, {b1}], keeps only suffixes congruent to {residue}
modulo {modulus}, and removes forbidden suffixes {excluded!r}.

Use prefix "{prefix}" and return strings like "{prefix}.42".
"""
    main = "Return the final set of BNS number strings."
    template = """
def answer():
    r\"\"\"
    Return final BNS strings after all filters.
    \"\"\"
    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    BNS_numbers = ...
    # ---------------------------------------------------------------

    return BNS_numbers
"""
    target = f"""
def answer():
    r\"\"\"
    Return final BNS strings after all filters.
    \"\"\"
    start = max({a0}, {b0})
    end = min({a1}, {b1})
    excluded = {set(excluded)!r}
    return {{
        f"{prefix}.{{n}}"
        for n in range(start, end + 1)
        if n % {modulus} == {residue} and n not in excluded
    }}
"""
    return _example(
        problem_id=f"{split}_v15_bns_intersection_{idx:05d}_{prefix}_{a0}_{a1}_{b0}_{b1}_{modulus}_{residue}",
        split=split,
        family="v15_bns_interval_intersection",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "set_exact", "expected": sorted(expected)}]},
        solution_trace="Intersect intervals, apply modular filter, remove exclusions, then add prefix.",
        metadata={
            "domain": "magnetic_space_group",
            "answer_type": "compact_string_set",
            "expected_compact_chars": len(target),
        },
    )
