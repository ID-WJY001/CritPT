from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from rl_posttrain.critpt_synth.prompting import render_prompt
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.critpt_synth.v13_official_style import (
    generate_v13_official_style_examples,
    verify_v13_example,
)
from rl_posttrain.critpt_synth.verifier import verify_code_completion


GeneratorFn = Callable[[random.Random, int, str, str], SyntheticCritPTExample]


def generate_v14_compact_exec_examples(
    size: int,
    seed: int,
    split: str,
) -> list[SyntheticCritPTExample]:
    """V14 keeps V13 coverage but over-samples official-style runaway failures."""

    rng = random.Random(seed)
    base_size = int(size * 0.55)
    hard_size = size - base_size
    base = [
        _add_compact_instruction(example, source_profile="v13_base_in_v14")
        for example in generate_v13_official_style_examples(base_size, seed, split)
    ]
    hard = _generate_hardcases(hard_size, rng, split)
    examples = base + hard
    rng.shuffle(examples)
    return examples


def _add_compact_instruction(
    example: SyntheticCritPTExample,
    *,
    source_profile: str,
) -> SyntheticCritPTExample:
    compact_note = (
        "\n\nCompactness constraints for this problem:\n"
        "- Return complete executable Python, not prose.\n"
        "- Do not enumerate hundreds of repeated terms when a range, comprehension, "
        "helper variable, or closed-form expression is available.\n"
        "- Make sure brackets, braces, parentheses, and the code block are closed."
    )
    metadata = {
        **example.metadata,
        "generator_profile": source_profile,
        "anti_runaway": True,
    }
    return replace(example, prompt=example.prompt + compact_note, metadata=metadata)


def verify_v14_example(example: SyntheticCritPTExample) -> tuple[bool, str]:
    result = verify_code_completion(example.assistant_code_block(), example.verifier)
    return result.ok, result.reason


def _generate_hardcases(size: int, rng: random.Random, split: str) -> list[SyntheticCritPTExample]:
    specs: list[tuple[GeneratorFn, float, tuple[list[str], list[float]]]] = [
        (_sparse_long_coefficients, 0.24, (["medium", "hard"], [0.50, 0.50])),
        (_compact_bns_ranges, 0.22, (["medium", "hard"], [0.45, 0.55])),
        (_trace_operator_compact_family, 0.22, (["medium", "hard"], [0.45, 0.55])),
        (_non_nested_symbolic_integral, 0.18, (["medium", "hard"], [0.55, 0.45])),
        (_compact_generating_series, 0.14, (["medium", "hard"], [0.55, 0.45])),
    ]
    weights = [item[1] for item in specs]
    examples: list[SyntheticCritPTExample] = []
    seen: set[str] = set()
    attempts = 0
    while len(examples) < size:
        attempts += 1
        if attempts > max(size * 100, 100):
            raise RuntimeError(f"too many duplicate V14 examples while building {split}")
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
    compact_note = (
        "\n\nCompactness constraints for this problem:\n"
        "- Return complete executable Python, not prose.\n"
        "- Do not enumerate hundreds of repeated terms when a range, comprehension, "
        "helper variable, or closed-form expression is available.\n"
        "- Make sure brackets, braces, parentheses, and the code block are closed."
    )
    prompt = render_prompt(setup, main, template) + compact_note
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
            "generator_profile": "v14_compact_exec",
            "param_hash": param_hash,
            "official_overlap": "none",
            "uses_official_prompt": False,
            "prompt_style": "problem_setup_main_problem_parsing_template_compact",
            "anti_runaway": True,
        },
    )


def _numeric_sequence_checks(expected: list[float | int], tolerance: float = 1e-8) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = [{"mode": "sequence_length", "expected": len(expected)}]
    checks.extend(
        {
            "mode": "numeric_sequence_item",
            "index": idx,
            "expected": value,
            "tolerance": tolerance,
        }
        for idx, value in enumerate(expected)
    )
    return checks


def _sparse_long_coefficients(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    length = rng.choice([48, 64, 96] if difficulty == "medium" else [96, 128, 160])
    nonzero_count = rng.choice([4, 5, 6] if difficulty == "medium" else [6, 7, 8])
    positions = sorted(rng.sample(range(length), nonzero_count))
    coeffs = [0.0] * length
    for pos in positions:
        coeffs[pos] = float(rng.choice([-3, -2, -1, 1, 2, 3]) * (pos % 5 + 1))
    zero_probe_positions = [pos for pos in [0, length // 3, length // 2, length - 1] if pos not in positions]
    checks = [{"mode": "sequence_length", "expected": length}]
    checks.extend(
        {
            "mode": "numeric_sequence_item",
            "index": pos,
            "expected": coeffs[pos],
            "tolerance": 1e-8,
        }
        for pos in positions + zero_probe_positions
    )
    setup = f"""
A normalized spin-chain Hamiltonian is expanded in an ordered Pauli-string basis
with {length} basis elements. Symmetry kills almost every coefficient. The only
nonzero coefficients are listed below as index -> value:

{positions!r} -> {[coeffs[pos] for pos in positions]!r}

All other coefficients are zero. This is a compactness test: the correct answer
should build the mostly-zero list programmatically rather than typing every zero.
"""
    main = "Return the full coefficient list in basis order."
    template = """
def answer():
    r\"\"\"
    Return all Hamiltonian coefficients in basis order.

    Outputs
    ----------
    coeffs: list[float]
    \"\"\"
    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    coeffs = ...
    # ---------------------------------------------------------------

    return coeffs
"""
    target = f"""
def answer():
    r\"\"\"
    Return all Hamiltonian coefficients in basis order.
    \"\"\"
    coeffs = [0.0] * {length}
    updates = { {pos: coeffs[pos] for pos in positions}!r}
    for index, value in updates.items():
        coeffs[index] = value
    return coeffs
"""
    return _example(
        problem_id=f"{split}_v14_sparse_coeffs_{idx:05d}_{length}_{sum(positions)}",
        split=split,
        family="official_failure_sparse_coefficients",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": checks},
        solution_trace=f"Use [0.0] * {length}, then assign {positions!r}.",
        metadata={
            "domain": "quantum_spin_hamiltonian",
            "answer_type": "long_sparse_numeric_sequence",
            "expected_compact_chars": len(target),
        },
    )


def _compact_bns_ranges(rng: random.Random, idx: int, split: str, difficulty: str) -> SyntheticCritPTExample:
    prefix = rng.choice(["182", "191", "194", "221"])
    start = rng.randint(20, 80)
    span = rng.choice([18, 24, 30] if difficulty == "medium" else [45, 60, 75])
    end = start + span
    excluded = sorted(rng.sample(range(start, end + 1), k=min(4 if difficulty == "medium" else 7, span)))
    expected = {f"{prefix}.{n}" for n in range(start, end + 1) if n not in excluded}
    setup = f"""
A toy magnetic space-group filter returns all BNS numbers with prefix {prefix}
and suffix in the inclusive interval [{start}, {end}], except the forbidden
suffixes {excluded!r}. The final answer is a set of strings such as "182.22".

This is not asking for a long manual enumeration. Use compact Python construction.
"""
    main = "Return the set of allowed BNS number strings."
    template = """
def answer():
    r\"\"\"
    Return allowed BNS numbers.

    Outputs
    ----------
    BNS_numbers: set[str]
    \"\"\"
    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    BNS_numbers = ...
    # ---------------------------------------------------------------

    return BNS_numbers
"""
    target = f"""
def answer():
    r\"\"\"
    Return allowed BNS numbers.
    \"\"\"
    excluded = {set(excluded)!r}
    return {{f"{prefix}.{{n}}" for n in range({start}, {end + 1}) if n not in excluded}}
"""
    return _example(
        problem_id=f"{split}_v14_bns_ranges_{idx:05d}_{prefix}_{start}_{end}",
        split=split,
        family="official_failure_bns_compact_set",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "set_exact", "expected": sorted(expected)}]},
        solution_trace=f"Allowed set is prefix {prefix}, range({start}, {end + 1}), minus {excluded!r}.",
        metadata={
            "domain": "magnetic_space_group",
            "answer_type": "compact_string_set",
            "expected_compact_chars": len(target),
        },
    )


def _trace_operator_compact_family(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    max_charge = rng.choice([8, 10] if difficulty == "medium" else [12, 15])
    single_limit = rng.choice([3, 4])
    singles = {f"tr(psi^{k})" for k in range(1, min(single_limit, max_charge) + 1)}
    products = {
        f"tr(psi^{a}) tr(psi^{b})"
        for a in range(1, single_limit + 1)
        for b in range(a, single_limit + 1)
        if a + b <= max_charge and a + b >= 4
    }
    expected = singles | products
    setup = f"""
Consider a toy rank-2 gauge theory with one adjoint fermion psi. The synthetic
Cayley-Hamilton rule keeps single traces tr(psi^k) only for 1 <= k <= {single_limit}.
Indecomposable two-trace products tr(psi^a) tr(psi^b) are kept when
1 <= a <= b <= {single_limit}, total charge a+b <= {max_charge}, and a+b >= 4.

Return strings in exactly the notation shown above. Use a compact set
comprehension instead of spelling out every operator if useful.
"""
    main = "Return the set of indecomposable operator strings."
    template = """
def answer():
    r\"\"\"
    Return all allowed operator strings.

    Outputs
    ----------
    operators: set[str]
    \"\"\"
    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    operators = ...
    # ---------------------------------------------------------------

    return operators
"""
    target = f"""
def answer():
    r\"\"\"
    Return all allowed operator strings.
    \"\"\"
    single_limit = {single_limit}
    max_charge = {max_charge}
    singles = {{f"tr(psi^{{k}})" for k in range(1, min(single_limit, max_charge) + 1)}}
    products = {{
        f"tr(psi^{{a}}) tr(psi^{{b}})"
        for a in range(1, single_limit + 1)
        for b in range(a, single_limit + 1)
        if 4 <= a + b <= max_charge
    }}
    return singles | products
"""
    return _example(
        problem_id=f"{split}_v14_trace_operator_family_{idx:05d}_{single_limit}_{max_charge}",
        split=split,
        family="official_failure_operator_compact_set",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "set_exact", "expected": sorted(expected)}]},
        solution_trace=f"Use singles up to {single_limit} and two-trace products up to charge {max_charge}.",
        metadata={
            "domain": "gauge_theory_operator_counting",
            "answer_type": "compact_string_set",
            "expected_compact_chars": len(target),
        },
    )


def _non_nested_symbolic_integral(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    numerator = rng.randint(1, 6 if difficulty == "medium" else 9)
    offset = rng.randint(2, 7 if difficulty == "medium" else 11)
    power = rng.choice([2, 3])
    expected = f"{numerator}*sp.pi**2*T**{power}/(U + {offset})"
    setup = f"""
In a simplified finite-temperature phase-space integral, all constants except
T and U have already been integrated out. The reduced closed form is

I(T, U) = A pi^2 T^p / (U + B),

with A={numerator}, p={power}, B={offset}. Return one closed SymPy expression.
Do not build a deeply nested or repeated expression.
"""
    main = "Return the symbolic phase-space integral I(T, U)."
    template = """
import sympy as sp

T, U = sp.symbols('T U', positive=True)

def answer(T, U):
    r\"\"\"
    Return I(T, U) as a SymPy expression.
    \"\"\"
    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    I = ...
    # ---------------------------------------------------------------

    return I
"""
    target = f"""
import sympy as sp

T, U = sp.symbols('T U', positive=True)

def answer(T, U):
    r\"\"\"
    Return I(T, U) as a SymPy expression.
    \"\"\"
    I = {expected}
    return I
"""
    return _example(
        problem_id=f"{split}_v14_symbolic_integral_{idx:05d}_{numerator}_{power}_{offset}",
        split=split,
        family="official_failure_symbolic_stop",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={
            "kind": "code",
            "timeout_s": 2.0,
            "checks": [
                {
                    "mode": "symbolic",
                    "args": [{"$sym": "T"}, {"$sym": "U"}],
                    "expected": expected.replace("sp.", ""),
                    "variables": ["T", "U"],
                    "tolerance": 1e-8,
                }
            ],
        },
        solution_trace=f"Closed form is {expected}.",
        metadata={
            "domain": "finite_temperature_integral",
            "answer_type": "symbolic",
            "expected_compact_chars": len(target),
        },
    )


def _compact_generating_series(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    degree = rng.choice([10, 12] if difficulty == "medium" else [15, 18])
    period = rng.choice([3, 4, 5])
    scale = rng.randint(1, 4)
    expected = [scale * ((n % period) - 1) for n in range(degree + 1)]
    setup = f"""
A trace-relation index has coefficients c_n = {scale}*((n mod {period}) - 1)
for charges 0 <= n <= {degree}. Return all coefficients as a list ordered by n.

This is intentionally a compact series-generation task. Use a list
comprehension or short loop; do not type the sequence by hand.
"""
    main = "Return [c_0, c_1, ..., c_degree]."
    template = """
def answer():
    r\"\"\"
    Return the finite coefficient list.
    \"\"\"
    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    coeffs = ...
    # ---------------------------------------------------------------

    return coeffs
"""
    target = f"""
def answer():
    r\"\"\"
    Return the finite coefficient list.
    \"\"\"
    return [{scale} * ((n % {period}) - 1) for n in range({degree + 1})]
"""
    return _example(
        problem_id=f"{split}_v14_compact_series_{idx:05d}_{degree}_{period}_{scale}",
        split=split,
        family="official_failure_compact_series",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_checks(expected)},
        solution_trace=f"Use c_n={scale}*((n mod {period}) - 1) for n=0..{degree}.",
        metadata={
            "domain": "trace_relation_index",
            "answer_type": "compact_numeric_sequence",
            "expected_compact_chars": len(target),
        },
    )


def summarize_v14_source(example: SyntheticCritPTExample) -> str:
    if str(example.metadata.get("generator_profile")) == "v14_compact_exec":
        return "v14_hardcase"
    return "v13_base"
