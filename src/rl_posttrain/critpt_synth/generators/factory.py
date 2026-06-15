from __future__ import annotations

import random
from collections.abc import Callable
from fractions import Fraction

from rl_posttrain.critpt_synth.prompting import render_prompt
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample


GeneratorFn = Callable[[random.Random, int, str, str], SyntheticCritPTExample]


FAMILIES: list[tuple[str, float, list[GeneratorFn]]] = [
    ("template", 0.25, []),
    ("symbolic", 0.30, []),
    ("numeric", 0.25, []),
    ("discrete", 0.20, []),
]


def generate_examples(size: int, seed: int, split: str) -> list[SyntheticCritPTExample]:
    rng = random.Random(seed)
    family_map = {
        "template": [
            _conditional_fidelity,
            _series_coefficients,
            _series_coefficients_algorithmic,
            _series_diagnostics_tuple,
        ],
        "symbolic": [
            _gaussian_moment,
            _two_level_gap,
            _matrix_invariant,
            _rational_response,
            _steady_state_elimination,
            _multi_output_linear_response,
        ],
        "numeric": [_markov_chain, _transfer_matrix, _linear_recurrence, _spectral_trace_power],
        "discrete": [
            _oam_selection_rule,
            _oam_selection_rule_algorithmic,
            _lattice_residue_set,
            _stabilizer_dimension,
            _hamming_weight_enumerator,
        ],
    }
    family_weights = [0.25, 0.30, 0.25, 0.20]
    family_names = list(family_map)
    difficulties = ["easy", "medium", "hard"]
    difficulty_weights = [0.25, 0.55, 0.20]
    examples: list[SyntheticCritPTExample] = []
    seen: set[str] = set()
    while len(examples) < size:
        idx = len(examples)
        family = rng.choices(family_names, weights=family_weights, k=1)[0]
        difficulty = rng.choices(difficulties, weights=difficulty_weights, k=1)[0]
        generator = rng.choice(family_map[family])
        example = generator(rng, idx, split, difficulty)
        if example.problem_id in seen:
            continue
        seen.add(example.problem_id)
        examples.append(example)
    return examples


def generate_hardcase_examples(size: int, seed: int, split: str) -> list[SyntheticCritPTExample]:
    rng = random.Random(seed)
    generator_specs: list[tuple[GeneratorFn, float, tuple[list[str], list[float]]]] = [
        (_gaussian_moment, 0.16, (["medium", "hard"], [0.45, 0.55])),
        (_two_level_gap, 0.14, (["medium", "hard"], [0.35, 0.65])),
        (_transfer_matrix, 0.12, (["medium", "hard"], [0.45, 0.55])),
        (_spectral_trace_power, 0.12, (["easy", "medium", "hard"], [0.20, 0.45, 0.35])),
        (_linear_recurrence, 0.10, (["medium", "hard"], [0.35, 0.65])),
        (_markov_chain, 0.08, (["medium", "hard"], [0.45, 0.55])),
        (_conditional_fidelity, 0.10, (["medium", "hard"], [0.45, 0.55])),
        (_series_coefficients, 0.07, (["medium", "hard"], [0.50, 0.50])),
        (_series_diagnostics_tuple, 0.06, (["medium", "hard"], [0.50, 0.50])),
        (_hamming_weight_enumerator, 0.05, (["medium", "hard"], [0.45, 0.55])),
    ]
    generators = [spec[0] for spec in generator_specs]
    weights = [spec[1] for spec in generator_specs]
    examples: list[SyntheticCritPTExample] = []
    seen: set[str] = set()
    while len(examples) < size:
        idx = len(examples)
        spec_idx = rng.choices(range(len(generators)), weights=weights, k=1)[0]
        difficulties, difficulty_weights = generator_specs[spec_idx][2]
        difficulty = rng.choices(difficulties, weights=difficulty_weights, k=1)[0]
        example = generators[spec_idx](rng, idx, split, difficulty)
        if example.problem_id in seen:
            continue
        seen.add(example.problem_id)
        metadata = {
            **example.metadata,
            "generator_profile": "v6_hardcase",
            "hardcase_focus": "symbolic_numeric_regressions",
        }
        examples.append(
            SyntheticCritPTExample(
                problem_id=example.problem_id,
                prompt=example.prompt,
                code_template=example.code_template,
                target_code=example.target_code,
                verifier=example.verifier,
                split=example.split,
                family=example.family,
                difficulty=example.difficulty,
                solution_trace=example.solution_trace,
                metadata=metadata,
            )
        )
    return examples


def generate_v7_intermediate_examples(size: int, seed: int, split: str) -> list[SyntheticCritPTExample]:
    rng = random.Random(seed)
    generator_specs: list[tuple[GeneratorFn, float, tuple[list[str], list[float]]]] = [
        (_markov_chain_audit, 0.22, (["medium", "hard"], [0.45, 0.55])),
        (_transfer_matrix_audit, 0.20, (["medium", "hard"], [0.45, 0.55])),
        (_spectral_trace_power_audit, 0.20, (["medium", "hard"], [0.45, 0.55])),
        (_series_coefficients_audit, 0.17, (["medium", "hard"], [0.40, 0.60])),
        (_series_diagnostics_audit, 0.13, (["medium", "hard"], [0.40, 0.60])),
        (_hamming_weight_enumerator_audit, 0.05, (["medium", "hard"], [0.50, 0.50])),
        (_two_level_gap, 0.03, (["hard"], [1.0])),
    ]
    generators = [spec[0] for spec in generator_specs]
    weights = [spec[1] for spec in generator_specs]
    examples: list[SyntheticCritPTExample] = []
    seen: set[str] = set()
    while len(examples) < size:
        idx = len(examples)
        spec_idx = rng.choices(range(len(generators)), weights=weights, k=1)[0]
        difficulties, difficulty_weights = generator_specs[spec_idx][2]
        difficulty = rng.choices(difficulties, weights=difficulty_weights, k=1)[0]
        example = generators[spec_idx](rng, idx, split, difficulty)
        if example.problem_id in seen:
            continue
        seen.add(example.problem_id)
        metadata = {
            **example.metadata,
            "generator_profile": "v7_intermediate",
            "hardcase_focus": "numeric_template_intermediate_checks",
        }
        examples.append(
            SyntheticCritPTExample(
                problem_id=example.problem_id,
                prompt=example.prompt,
                code_template=example.code_template,
                target_code=example.target_code,
                verifier=example.verifier,
                split=example.split,
                family=example.family,
                difficulty=example.difficulty,
                solution_trace=example.solution_trace,
                metadata=metadata,
            )
        )
    return examples


def generate_v7_compact_examples(size: int, seed: int, split: str) -> list[SyntheticCritPTExample]:
    rng = random.Random(seed)
    generator_specs: list[tuple[GeneratorFn, float, tuple[list[str], list[float]]]] = [
        (_markov_chain_compact_audit, 0.23, (["medium", "hard"], [0.55, 0.45])),
        (_transfer_matrix_compact_audit, 0.20, (["medium", "hard"], [0.55, 0.45])),
        (_spectral_trace_power_compact_audit, 0.19, (["medium", "hard"], [0.55, 0.45])),
        (_series_coefficients_compact_audit, 0.18, (["medium", "hard"], [0.55, 0.45])),
        (_series_diagnostics_compact_audit, 0.13, (["medium", "hard"], [0.55, 0.45])),
        (_hamming_weight_enumerator_compact_audit, 0.05, (["medium", "hard"], [0.60, 0.40])),
        (_two_level_gap, 0.02, (["hard"], [1.0])),
    ]
    generators = [spec[0] for spec in generator_specs]
    weights = [spec[1] for spec in generator_specs]
    examples: list[SyntheticCritPTExample] = []
    seen: set[str] = set()
    while len(examples) < size:
        idx = len(examples)
        spec_idx = rng.choices(range(len(generators)), weights=weights, k=1)[0]
        difficulties, difficulty_weights = generator_specs[spec_idx][2]
        difficulty = rng.choices(difficulties, weights=difficulty_weights, k=1)[0]
        example = generators[spec_idx](rng, idx, split, difficulty)
        if example.problem_id in seen:
            continue
        seen.add(example.problem_id)
        metadata = {
            **example.metadata,
            "generator_profile": "v7_compact",
            "hardcase_focus": "short_numeric_template_intermediate_checks",
        }
        examples.append(
            SyntheticCritPTExample(
                problem_id=example.problem_id,
                prompt=example.prompt,
                code_template=example.code_template,
                target_code=example.target_code,
                verifier=example.verifier,
                split=example.split,
                family=example.family,
                difficulty=example.difficulty,
                solution_trace=example.solution_trace,
                metadata=metadata,
            )
        )
    return examples


def generate_v9_trace_examples(size: int, seed: int, split: str) -> list[SyntheticCritPTExample]:
    rng = random.Random(seed)
    generator_specs: list[tuple[GeneratorFn, float, tuple[list[str], list[float]], Callable[[SyntheticCritPTExample], SyntheticCritPTExample]]] = [
        (_markov_chain_compact_audit, 0.12, (["medium", "hard"], [0.60, 0.40]), _trace_markov_chain),
        (_transfer_matrix_compact_audit, 0.22, (["medium", "hard"], [0.50, 0.50]), _trace_transfer_matrix),
        (_spectral_trace_power_compact_audit, 0.22, (["medium", "hard"], [0.50, 0.50]), _trace_spectral_power),
        (_series_coefficients_compact_audit, 0.17, (["medium", "hard"], [0.55, 0.45]), _trace_series_coefficients),
        (_series_diagnostics_compact_audit, 0.17, (["medium", "hard"], [0.55, 0.45]), _trace_series_diagnostics),
        (_hamming_weight_enumerator_compact_audit, 0.10, (["medium", "hard"], [0.55, 0.45]), _trace_hamming_enumerator),
    ]
    generators = [spec[0] for spec in generator_specs]
    weights = [spec[1] for spec in generator_specs]
    examples: list[SyntheticCritPTExample] = []
    seen: set[str] = set()
    while len(examples) < size:
        idx = len(examples)
        spec_idx = rng.choices(range(len(generators)), weights=weights, k=1)[0]
        generator, _weight, difficulty_spec, wrapper = generator_specs[spec_idx]
        difficulties, difficulty_weights = difficulty_spec
        difficulty = rng.choices(difficulties, weights=difficulty_weights, k=1)[0]
        example = wrapper(generator(rng, idx, split, difficulty))
        if example.problem_id in seen:
            continue
        seen.add(example.problem_id)
        examples.append(example)
    return examples


def generate_v10_curriculum_trace_examples(size: int, seed: int, split: str) -> list[SyntheticCritPTExample]:
    rng = random.Random(seed)
    generator_specs: list[tuple[GeneratorFn, float, tuple[list[str], list[float]]]] = [
        (_v10_transfer_matrix_trace, 0.24, (["easy", "medium"], [0.75, 0.25])),
        (_v10_series_coefficients_trace, 0.22, (["easy", "medium"], [0.75, 0.25])),
        (_v10_series_diagnostics_trace, 0.16, (["easy", "medium"], [0.75, 0.25])),
        (_v10_hamming_enumerator_trace, 0.20, (["easy", "medium"], [0.80, 0.20])),
        (_v10_linear_recurrence_trace, 0.18, (["easy", "medium"], [0.80, 0.20])),
    ]
    generators = [spec[0] for spec in generator_specs]
    weights = [spec[1] for spec in generator_specs]
    examples: list[SyntheticCritPTExample] = []
    seen: set[str] = set()
    while len(examples) < size:
        idx = len(examples)
        spec_idx = rng.choices(range(len(generators)), weights=weights, k=1)[0]
        difficulties, difficulty_weights = generator_specs[spec_idx][2]
        difficulty = rng.choices(difficulties, weights=difficulty_weights, k=1)[0]
        example = generators[spec_idx](rng, idx, split, difficulty)
        if example.problem_id in seen:
            continue
        seen.add(example.problem_id)
        examples.append(example)
    return examples


def generate_v11_template_series_trace_examples(size: int, seed: int, split: str) -> list[SyntheticCritPTExample]:
    rng = random.Random(seed)
    generator_specs: list[tuple[GeneratorFn, float, tuple[list[str], list[float]]]] = [
        (_v11_series_coefficients_trace, 0.45, (["easy", "medium"], [0.85, 0.15])),
        (_v11_series_diagnostics_trace, 0.35, (["easy", "medium"], [0.85, 0.15])),
        (_v11_series_consistency_trace, 0.20, (["easy", "medium"], [0.85, 0.15])),
    ]
    generators = [spec[0] for spec in generator_specs]
    weights = [spec[1] for spec in generator_specs]
    examples: list[SyntheticCritPTExample] = []
    seen: set[str] = set()
    while len(examples) < size:
        idx = len(examples)
        spec_idx = rng.choices(range(len(generators)), weights=weights, k=1)[0]
        difficulties, difficulty_weights = generator_specs[spec_idx][2]
        difficulty = rng.choices(difficulties, weights=difficulty_weights, k=1)[0]
        example = generators[spec_idx](rng, idx, split, difficulty)
        if example.problem_id in seen:
            continue
        seen.add(example.problem_id)
        examples.append(example)
    return examples


def _expected_sequence(example: SyntheticCritPTExample) -> list[float | int]:
    items: list[tuple[int, float | int]] = []
    expected_len = 0
    for check in example.verifier.get("checks", []):
        if check.get("mode") == "sequence_length":
            expected_len = int(check.get("expected", check.get("length", 0)))
        elif check.get("mode") == "numeric_sequence_item":
            items.append((int(check["index"]), check["expected"]))
    if not items:
        raise ValueError(f"no numeric_sequence_item checks for {example.problem_id}")
    if expected_len <= 0:
        expected_len = max(index for index, _value in items if index >= 0) + 1
    out: list[float | int] = [0 for _ in range(expected_len)]
    for index, value in items:
        if index < 0:
            index = expected_len + index
        out[index] = value
    return out


def _with_trace_reward(
    example: SyntheticCritPTExample,
    tag_values: list[tuple[str, float | int]],
    *,
    focus: str,
    profile: str = "v9_trace",
) -> SyntheticCritPTExample:
    reward_checks: list[dict[str, float | int | str]] = [
        {"mode": "text_numeric", "tag": tag, "expected": value, "tolerance": 1e-4}
        for tag, value in tag_values
    ]
    reward_checks.extend(dict(check) for check in example.verifier.get("checks", []))
    verifier = {**example.verifier, "reward_checks": reward_checks}
    audit_line = ", ".join(f"{tag}={value!r}" for tag, value in tag_values)
    metadata = {
        **example.metadata,
        "generator_profile": profile,
        "hardcase_focus": focus,
        "audit_tags": [tag for tag, _value in tag_values],
        "reward_check_count": len(reward_checks),
    }
    return SyntheticCritPTExample(
        problem_id=example.problem_id.replace("compact_audit", "trace_audit"),
        prompt=example.prompt,
        code_template=example.code_template,
        target_code=example.target_code,
        verifier=verifier,
        split=example.split,
        family=example.family,
        difficulty=example.difficulty,
        solution_trace=f"{example.solution_trace.strip()} audit: {audit_line}.",
        metadata=metadata,
    )


def _v10_transfer_matrix_trace(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    max_entry = 1 if difficulty == "easy" else 2
    a = rng.randint(1, max_entry + 1)
    b = rng.randint(0, max_entry)
    c = rng.randint(0, max_entry)
    d = rng.randint(1, max_entry + 1)
    power = 2 if difficulty == "easy" else 3
    matrix = [[a, b], [c, d]]
    t2 = _mat_pow_2x2(matrix, 2)
    tn = _mat_pow_2x2(matrix, power)
    expected = [float(tn[0][0] + tn[1][1]), float(t2[0][1]), float(tn[1][0])]
    template = '''
def answer():
    r"""
    Return [trace(T^n), T2_01, Tn_10].
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    audit = ...
    # ---------------------------------------------------------------

    return audit
'''
    target = f'''
def answer():
    r"""
    Return [trace(T^n), T2_01, Tn_10].
    """
    return {expected!r}
'''
    setup = f"Let $T=\\begin{{pmatrix}} {a} & {b} \\\\ {c} & {d} \\end{{pmatrix}}$."
    main = (
        f"Compute $T^2$, $T^{power}$, and $\\mathrm{{tr}}(T^{power})$. "
        f"Return exactly `[trace(T^{power}), (T^2)_01, (T^{power})_10]`."
    )
    example = _example(
        problem_id=f"{split}_numeric_transfer_matrix_curriculum_{idx:05d}_{a}_{b}_{c}_{d}_{power}",
        split=split,
        family="numeric",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(expected, 1e-8)},
        trace=f"T^2={t2!r}; T^{power}={tn!r}; answer={expected!r}.",
        metadata={"domain": "condensed_matter_transfer", "official_overlap": "none", "variant": "v10_curriculum_transfer"},
    )
    return _with_trace_reward(
        example,
        [("trace", expected[0]), ("T2_01", expected[1]), ("Tn_10", expected[2])],
        focus="curriculum_small_transfer_matrix",
        profile="v10_curriculum_trace",
    )


def _v10_series_coefficients_trace(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    max_param = 2 if difficulty == "easy" else 3
    a = rng.randint(1, max_param)
    b = rng.randint(1, max_param)
    order = 2 if difficulty == "easy" else 3
    coeffs = _series_coeffs_exp_over_pole(a, b, order)
    template = f'''
def answer():
    r"""
    Return [c0, ..., c{order}].
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    coeffs = ...
    # ---------------------------------------------------------------

    return coeffs
'''
    target = f'''
def answer():
    r"""
    Return [c0, ..., c{order}].
    """
    return {coeffs!r}
'''
    setup = f"Let $f(x)=\\exp({a}x)/(1-{b}x)$."
    main = (
        f"Return exactly the Taylor coefficient list `[c0, ..., c{order}]`. "
        "$c_n=\\sum_{k=0}^{n} a^k/k!\\,b^{n-k}$."
    )
    example = _example(
        problem_id=f"{split}_template_series_coefficients_curriculum_{idx:05d}_{a}_{b}_{order}",
        split=split,
        family="template",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(coeffs, 1e-8)},
        trace=f"Using the convolution formula gives coeffs={coeffs!r}.",
        metadata={"domain": "perturbation_series", "official_overlap": "none", "variant": "v10_curriculum_coefficients"},
    )
    return _with_trace_reward(
        example,
        [("c0", coeffs[0]), ("c1", coeffs[1]), ("c_last", coeffs[-1])],
        focus="curriculum_low_order_series_coefficients",
        profile="v10_curriculum_trace",
    )


def _v10_series_diagnostics_trace(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    max_param = 2 if difficulty == "easy" else 3
    a = rng.randint(1, max_param)
    b = rng.randint(1, max_param)
    order = 2 if difficulty == "easy" else 3
    coeffs = _series_coeffs_exp_over_pole(a, b, order)
    expected = [float(sum(coeffs)), coeffs[1], coeffs[-1] / coeffs[-2]]
    template = '''
def answer():
    r"""
    Return [sum_of_coeffs, c1, c_last_over_previous].
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    diagnostics = ...
    # ---------------------------------------------------------------

    return diagnostics
'''
    target = f'''
def answer():
    r"""
    Return [sum_of_coeffs, c1, c_last_over_previous].
    """
    return {expected!r}
'''
    setup = f"Let $f(x)=\\exp({a}x)/(1-{b}x)$ and use coefficients through order {order}."
    main = "Return exactly `[sum_of_coeffs, c1, c_last_over_previous]`."
    example = _example(
        problem_id=f"{split}_template_series_diagnostics_curriculum_{idx:05d}_{a}_{b}_{order}",
        split=split,
        family="template",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(expected, 1e-8)},
        trace=f"coeffs={coeffs!r}; diagnostics={expected!r}.",
        metadata={"domain": "perturbation_series", "official_overlap": "none", "variant": "v10_curriculum_diagnostics"},
    )
    return _with_trace_reward(
        example,
        [("sum", expected[0]), ("c1", expected[1]), ("ratio", expected[2])],
        focus="curriculum_low_order_series_diagnostics",
        profile="v10_curriculum_trace",
    )


def _v10_hamming_enumerator_trace(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    n = rng.randint(3, 4 if difficulty == "easy" else 5)
    modulus = rng.choice([2, 3])
    residue = rng.randint(0, modulus - 1)
    weights = [0 for _ in range(n + 1)]
    for mask in range(1 << n):
        weight = mask.bit_count()
        if weight % modulus == residue:
            weights[weight] += 1
    template = '''
def answer():
    r"""
    Return the full Hamming-weight enumerator coefficient list.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    coeffs = ...
    # ---------------------------------------------------------------

    return coeffs
'''
    target = f'''
def answer():
    r"""
    Return the full Hamming-weight enumerator coefficient list.
    """
    return {weights!r}
'''
    setup = (
        f"Consider all binary strings of length {n}. A string is accepted when its "
        f"Hamming weight is congruent to {residue} modulo {modulus}."
    )
    main = f"Return exactly the full length-{n + 1} list `[coeffs[0], ..., coeffs[{n}]]`."
    example = _example(
        problem_id=f"{split}_discrete_hamming_weight_enumerator_curriculum_{idx:05d}_{n}_{modulus}_{residue}",
        split=split,
        family="discrete",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(weights, 1e-8)},
        trace=f"The full enumerator is {weights!r}.",
        metadata={"domain": "toy_qec_enumerator", "official_overlap": "none", "variant": "v10_curriculum_hamming"},
    )
    return _with_trace_reward(
        example,
        [("accepted_total", sum(weights)), ("nonzero_slots", sum(1 for value in weights if value != 0))],
        focus="curriculum_small_hamming_buckets",
        profile="v10_curriculum_trace",
    )


def _v10_linear_recurrence_trace(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    u0 = rng.randint(0, 2)
    u1 = rng.randint(1, 3)
    a = rng.randint(1, 2)
    b = rng.randint(1, 2)
    n = rng.randint(3, 4 if difficulty == "easy" else 5)
    values = [u0, u1]
    for _step in range(2, n + 1):
        values.append(a * values[-1] + b * values[-2])
    mid = max(2, n // 2)
    expected = [values[2], values[mid], values[n]]
    template = '''
def answer():
    r"""
    Return [u2, u_mid, u_n].
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    audit = ...
    # ---------------------------------------------------------------

    return audit
'''
    target = f'''
def answer():
    r"""
    Return [u2, u_mid, u_n].
    """
    return {expected!r}
'''
    setup = f"A recurrence has $u_0={u0}$, $u_1={u1}$ and $u_i={a}u_{{i-1}}+{b}u_{{i-2}}$."
    main = f"Compute values up to $u_{n}$ and return exactly `[u2, u{mid}, u{n}]`."
    example = _example(
        problem_id=f"{split}_numeric_linear_recurrence_curriculum_{idx:05d}_{u0}_{u1}_{a}_{b}_{n}",
        split=split,
        family="numeric",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(expected, 1e-8)},
        trace=f"values={values!r}; return {expected!r}.",
        metadata={"domain": "many_body_recurrence", "official_overlap": "none", "variant": "v10_curriculum_recurrence"},
    )
    return _with_trace_reward(
        example,
        [("u2", expected[0]), ("u_mid", expected[1]), ("u_n", expected[2])],
        focus="curriculum_short_linear_recurrence",
        profile="v10_curriculum_trace",
    )


def _v11_series_params(rng: random.Random, difficulty: str) -> tuple[int, int, int, int]:
    if difficulty == "easy":
        return rng.randint(1, 2), rng.randint(1, 2), rng.randint(0, 2), 2
    return rng.randint(1, 3), rng.randint(1, 3), rng.randint(0, 3), 3


def _series_coeffs_poly_exp_over_pole(a: int, b: int, s: int, order: int) -> list[float]:
    base = _series_coeffs_exp_over_pole(a, b, order)
    coeffs: list[float] = []
    for n, value in enumerate(base):
        shifted = s * base[n - 1] if n > 0 else 0.0
        coeffs.append(float(value + shifted))
    return coeffs


def _v11_series_coefficients_trace(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a, b, s, order = _v11_series_params(rng, difficulty)
    coeffs = _series_coeffs_poly_exp_over_pole(a, b, s, order)
    template = f'''
def answer():
    r"""
    Return [c0, ..., c{order}].
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    coeffs = ...
    # ---------------------------------------------------------------

    return coeffs
'''
    target = f'''
def answer():
    r"""
    Return [c0, ..., c{order}].
    """
    return {coeffs!r}
'''
    setup = f"Let $f(x)=(1+{s}x)\\exp({a}x)/(1-{b}x)$."
    main = (
        f"Return exactly the Taylor coefficient list `[c0, ..., c{order}]`. "
        "Use `base_n=sum_{k=0}^n a^k/k! * b^(n-k)` for exp(a x)/(1-b x), "
        "then `c_n=base_n+s*base_{n-1}` with `base_-1=0`."
    )
    example = _example(
        problem_id=f"{split}_template_series_coefficients_v11_{idx:05d}_{a}_{b}_{s}_{order}",
        split=split,
        family="template",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(coeffs, 1e-8)},
        trace=f"base coefficients then numerator shift give coeffs={coeffs!r}.",
        metadata={"domain": "perturbation_series", "official_overlap": "none", "variant": "v11_template_coefficients"},
    )
    tags: list[tuple[str, float | int]] = [("c0", coeffs[0]), ("c1", coeffs[1])]
    if len(coeffs) > 2:
        tags.append(("c2", coeffs[2]))
    tags.append(("c_last", coeffs[-1]))
    tags.append(("sum", float(sum(coeffs))))
    return _with_trace_reward(
        example,
        tags,
        focus="template_series_coefficients_focused_curriculum",
        profile="v11_template_series_trace",
    )


def _v11_series_diagnostics_trace(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a, b, s, order = _v11_series_params(rng, difficulty)
    coeffs = _series_coeffs_poly_exp_over_pole(a, b, s, order)
    expected = [float(sum(coeffs)), coeffs[1], coeffs[-1] / coeffs[-2]]
    template = '''
def answer():
    r"""
    Return [sum_of_coeffs, c1, c_last_over_previous].
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    diagnostics = ...
    # ---------------------------------------------------------------

    return diagnostics
'''
    target = f'''
def answer():
    r"""
    Return [sum_of_coeffs, c1, c_last_over_previous].
    """
    return {expected!r}
'''
    setup = f"Let $f(x)=(1+{s}x)\\exp({a}x)/(1-{b}x)$ and use coefficients through order {order}."
    main = (
        "First compute `[c0, ..., c_order]` with "
        "`base_n=sum_{k=0}^n a^k/k! * b^(n-k)` and `c_n=base_n+s*base_{n-1}`. "
        "Return exactly `[sum_of_coeffs, c1, c_last_over_previous]`."
    )
    example = _example(
        problem_id=f"{split}_template_series_diagnostics_v11_{idx:05d}_{a}_{b}_{s}_{order}",
        split=split,
        family="template",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(expected, 1e-8)},
        trace=f"coeffs={coeffs!r}; diagnostics={expected!r}.",
        metadata={"domain": "perturbation_series", "official_overlap": "none", "variant": "v11_template_diagnostics"},
    )
    return _with_trace_reward(
        example,
        [("sum", expected[0]), ("c1", expected[1]), ("ratio", expected[2]), ("c_last", coeffs[-1])],
        focus="template_series_diagnostics_focused_curriculum",
        profile="v11_template_series_trace",
    )


def _v11_series_consistency_trace(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a, b, s, order = _v11_series_params(rng, difficulty)
    coeffs = _series_coeffs_poly_exp_over_pole(a, b, s, order)
    c2 = coeffs[2]
    sum_first3 = float(sum(coeffs[:3]))
    expected = [coeffs[0], coeffs[1], c2, sum_first3, coeffs[-1]]
    template = '''
def answer():
    r"""
    Return [c0, c1, c2, c0_plus_c1_plus_c2, c_last].
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    audit = ...
    # ---------------------------------------------------------------

    return audit
'''
    target = f'''
def answer():
    r"""
    Return [c0, c1, c2, c0_plus_c1_plus_c2, c_last].
    """
    return {expected!r}
'''
    setup = f"Let $f(x)=(1+{s}x)\\exp({a}x)/(1-{b}x)$."
    main = (
        "Compute coefficients through the requested order using the stated convolution. "
        "Return exactly `[c0, c1, c2, c0_plus_c1_plus_c2, c_last]`."
    )
    example = _example(
        problem_id=f"{split}_template_series_consistency_v11_{idx:05d}_{a}_{b}_{s}_{order}",
        split=split,
        family="template",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(expected, 1e-8)},
        trace=f"coeffs={coeffs!r}; consistency audit={expected!r}.",
        metadata={"domain": "perturbation_series", "official_overlap": "none", "variant": "v11_template_consistency"},
    )
    return _with_trace_reward(
        example,
        [
            ("c0", expected[0]),
            ("c1", expected[1]),
            ("c2", expected[2]),
            ("sum_first3", expected[3]),
            ("c_last", expected[4]),
        ],
        focus="template_series_consistency_focused_curriculum",
        profile="v11_template_series_trace",
    )


def _trace_markov_chain(example: SyntheticCritPTExample) -> SyntheticCritPTExample:
    expected = _expected_sequence(example)
    return _with_trace_reward(
        example,
        [("s1_mid", expected[0]), ("s1_final", expected[1])],
        focus="calculator_trace_markov_probabilities",
    )


def _trace_transfer_matrix(example: SyntheticCritPTExample) -> SyntheticCritPTExample:
    expected = _expected_sequence(example)
    return _with_trace_reward(
        example,
        [("trace", expected[0]), ("T2_01", expected[1]), ("Tn_10", expected[2])],
        focus="calculator_trace_transfer_cross_terms",
    )


def _trace_spectral_power(example: SyntheticCritPTExample) -> SyntheticCritPTExample:
    expected = _expected_sequence(example)
    return _with_trace_reward(
        example,
        [("Hp_trace", expected[0]), ("H2_trace", expected[1]), ("Hp_00", expected[2])],
        focus="calculator_trace_spectral_diagonals",
    )


def _trace_series_coefficients(example: SyntheticCritPTExample) -> SyntheticCritPTExample:
    expected = _expected_sequence(example)
    return _with_trace_reward(
        example,
        [("c0", expected[0]), ("c1", expected[1]), ("c_last", expected[-1])],
        focus="calculator_trace_series_convolution_coefficients",
    )


def _trace_series_diagnostics(example: SyntheticCritPTExample) -> SyntheticCritPTExample:
    expected = _expected_sequence(example)
    return _with_trace_reward(
        example,
        [("sum", expected[0]), ("c1", expected[1]), ("ratio", expected[2])],
        focus="calculator_trace_series_diagnostics",
    )


def _trace_hamming_enumerator(example: SyntheticCritPTExample) -> SyntheticCritPTExample:
    expected = _expected_sequence(example)
    return _with_trace_reward(
        example,
        [("accepted_total", sum(expected)), ("nonzero_slots", sum(1 for value in expected if value != 0))],
        focus="calculator_trace_hamming_buckets",
    )


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
    verifier: dict,
    trace: str,
    metadata: dict,
) -> SyntheticCritPTExample:
    return SyntheticCritPTExample(
        problem_id=problem_id,
        split=split,
        family=family,
        difficulty=difficulty,
        prompt=render_prompt(setup, main, template),
        code_template=template.strip(),
        target_code=target.strip(),
        verifier=verifier,
        solution_trace=trace,
        metadata=metadata,
    )


def _sym_arg(name: str) -> dict[str, str]:
    return {"$sym": name}


def _numeric_sequence_item_checks(expected: list[float | int], tolerance: float = 1e-8) -> list[dict[str, float | int | str]]:
    checks: list[dict[str, float | int | str]] = [{"mode": "sequence_length", "expected": len(expected)}]
    for index, value in enumerate(expected):
        checks.append(
            {
                "mode": "numeric_sequence_item",
                "index": index,
                "expected": value,
                "tolerance": tolerance,
            }
        )
    return checks


def _conditional_fidelity(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(1, 4)
    b = rng.randint(1, 5)
    c = rng.randint(2, 6)
    d = rng.randint(1, 6)
    expected = f"(1 - {a}*p + {b}*p**2)/(1 - {c}*p + {d}*p**2)"
    template = f'''
def answer(p):
    r"""
    Return the conditional logical fidelity F(p).

    Inputs
    ----------
    p: symbolic or float
        Physical error probability.

    Outputs
    ----------
    F: expression or float
        Conditional fidelity after post-selection.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    F = ...
    # ---------------------------------------------------------------

    return F
'''
    target = f'''
def answer(p):
    r"""
    Return the conditional logical fidelity F(p).
    """
    numerator = 1 - {a} * p + {b} * p**2
    acceptance = 1 - {c} * p + {d} * p**2
    F = numerator / acceptance
    return F
'''
    setup = (
        "A synthetic post-selection experiment keeps a run with probability "
        f"$A(p)=1-{c}p+{d}p^2$. The accepted logical-success numerator is "
        f"$N(p)=1-{a}p+{b}p^2$."
    )
    main = (
        "Compute the conditional logical fidelity $F(p)=N(p)/A(p)$. Return the exact "
        "rational expression using the given numerator and acceptance polynomial. "
        "Only cancel a factor after expanding it and verifying that it exactly matches "
        "both polynomials; otherwise leave the ratio as N(p)/A(p)."
    )
    return _example(
        problem_id=f"{split}_template_conditional_fidelity_{idx:05d}_{a}_{b}_{c}_{d}",
        split=split,
        family="template",
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
                    "args": [_sym_arg("p")],
                    "expected": expected,
                    "variables": ["p"],
                    "tolerance": 1e-8,
                }
            ],
        },
        trace=f"Use F=N/A, so F(p)={expected}.",
        metadata={"domain": "toy_qec", "official_overlap": "none"},
    )


def _series_coefficients(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(1, 5)
    b = rng.randint(1, 5)
    order = 3 if difficulty != "hard" else 4
    coeffs: list[float] = []
    for n in range(order + 1):
        total = Fraction(0, 1)
        for k in range(n + 1):
            total += Fraction(a**k, _factorial(k)) * Fraction(b ** (n - k), 1)
        coeffs.append(float(total))
    template = f'''
def answer():
    r"""
    Return the coefficients [c0, c1, ..., c{order}] of the Taylor expansion.

    Inputs
    ----------
    None

    Outputs
    ----------
    coeffs: list[float]
        Coefficients of f(x) through order {order}.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    coeffs = ...
    # ---------------------------------------------------------------

    return coeffs
'''
    target = f'''
def answer():
    r"""
    Return the coefficients [c0, c1, ..., c{order}] of the Taylor expansion.
    """
    coeffs = {coeffs!r}
    return coeffs
'''
    setup = (
        f"Consider the response function $f(x)=\\exp({a}x)/(1-{b}x)$. "
        "We only need its low-order expansion around x=0."
    )
    main = (
        f"Return the complete Taylor coefficient list $[c_0,c_1,\\ldots,c_{order}]$ "
        f"with exactly {order + 1} entries. Use the convolution "
        "$c_n=\\sum_{k=0}^{n} a^k/k!\\,b^{n-k}$ for "
        f"$f(x)=\\exp({a}x)/(1-{b}x)$."
    )
    return _example(
        problem_id=f"{split}_template_series_coefficients_{idx:05d}_{a}_{b}_{order}",
        split=split,
        family="template",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={
            "kind": "code",
            "timeout_s": 2.0,
            "checks": [{"mode": "numeric_sequence", "expected": coeffs, "tolerance": 1e-8}],
        },
        trace=(
            f"Use exp({a}x)=sum_k ({a}x)^k/k! and 1/(1-{b}x)=sum_m ({b}x)^m. "
            f"Collect x^n terms: c_n=sum_(k=0..n) {a}^k/k! * {b}^(n-k). "
            f"For the requested order the coefficients are {coeffs!r}."
        ),
        metadata={"domain": "perturbation_series", "official_overlap": "none"},
    )


def _series_coefficients_algorithmic(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(1, 7)
    b = rng.randint(1, 6)
    order = 4 if difficulty != "hard" else 6
    coeffs = _series_coeffs_exp_over_pole(a, b, order)
    template = f'''
def answer():
    r"""
    Return the coefficients [c0, c1, ..., c{order}] of the local response series.

    Inputs
    ----------
    None

    Outputs
    ----------
    coeffs: list[float]
        Coefficients of f(x) through order {order}.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    coeffs = ...
    # ---------------------------------------------------------------

    return coeffs
'''
    target = f'''
def answer():
    r"""
    Return the coefficients [c0, c1, ..., c{order}] of the local response series.
    """
    import math

    a = {a}
    b = {b}
    order = {order}
    coeffs = []
    for n in range(order + 1):
        total = 0.0
        for k in range(n + 1):
            total += (a ** k) / math.factorial(k) * (b ** (n - k))
        coeffs.append(total)
    return coeffs
'''
    setup = (
        "A perturbative response is written as a product of two standard pieces. "
        f"The smooth part contributes $\\exp({a}x)$ and the resonant denominator "
        f"contributes $(1-{b}x)^{{-1}}$. In coefficient language, this means "
        "$c_n=\\sum_{k=0}^{n} a^k/k!\\, b^{n-k}$."
    )
    main = (
        f"Compute the full coefficient list $[c_0,\\ldots,c_{order}]$. "
        "Use the convolution formula above; do not guess only the first few terms."
    )
    return _example(
        problem_id=f"{split}_template_series_coefficients_algorithmic_{idx:05d}_{a}_{b}_{order}",
        split=split,
        family="template",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={
            "kind": "code",
            "timeout_s": 2.0,
            "checks": [{"mode": "numeric_sequence", "expected": coeffs, "tolerance": 1e-8}],
        },
        trace=(
            f"Here a={a}, b={b}. Use the exact convolution "
            f"c_n=sum_(k=0..n) {a}^k/k! * {b}^(n-k), giving {coeffs!r}."
        ),
        metadata={"domain": "perturbation_series", "official_overlap": "none", "variant": "algorithmic"},
    )


def _series_diagnostics_tuple(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(1, 5)
    b = rng.randint(1, 5)
    order = 3 if difficulty == "easy" else 4
    coeffs = _series_coeffs_exp_over_pole(a, b, order)
    total_weight = float(sum(coeffs))
    leading_nontrivial = coeffs[1]
    ratio = coeffs[-1] / coeffs[-2]
    expected = [total_weight, leading_nontrivial, ratio]
    template = '''
def answer():
    r"""
    Return three diagnostics of a truncated perturbation series.

    Inputs
    ----------
    None

    Outputs
    ----------
    diagnostics: tuple[float, float, float]
        (sum_of_coeffs, c1, c_last_over_previous)
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    diagnostics = ...
    # ---------------------------------------------------------------

    return diagnostics
'''
    target = f'''
def answer():
    r"""
    Return three diagnostics of a truncated perturbation series.
    """
    coeffs = {coeffs!r}
    diagnostics = (sum(coeffs), coeffs[1], coeffs[-1] / coeffs[-2])
    return diagnostics
'''
    setup = (
        f"Let $f(x)=\\exp({a}x)/(1-{b}x)$ and let $c_n$ denote the Taylor "
        f"coefficient through order {order}. Some CritPT-style tasks ask for "
        "several derived quantities rather than the full expression."
    )
    main = (
        f"First compute every coefficient $[c_0,\\ldots,c_{order}]$ using "
        "$c_n=\\sum_{k=0}^{n} a^k/k!\\,b^{n-k}$. Then return exactly "
        "`(sum_of_coeffs, c1, c_last_over_previous)` as floats, where "
        f"`sum_of_coeffs` is $\\sum_{{i=0}}^{{{order}}} c_i$ and "
        f"`c_last_over_previous` is $c_{order}/c_{order - 1}$."
    )
    return _example(
        problem_id=f"{split}_template_series_diagnostics_{idx:05d}_{a}_{b}_{order}",
        split=split,
        family="template",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={
            "kind": "code",
            "timeout_s": 2.0,
            "checks": [{"mode": "numeric_sequence", "expected": expected, "tolerance": 1e-8}],
        },
        trace=(
            f"First compute Taylor coefficients for exp({a}x)/(1-{b}x): {coeffs!r}. "
            f"Then return (sum(coeffs), c1, c_last/c_previous) = {expected!r}."
        ),
        metadata={"domain": "perturbation_series", "official_overlap": "none", "variant": "multi_output"},
    )


def _gaussian_moment(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    k = rng.randint(1, 2 if difficulty == "easy" else 4)
    numerator = _double_factorial(2 * k - 1)
    denominator = 2**k
    expected = f"{numerator}/({denominator}*alpha**{k})"
    template = f'''
def answer(alpha):
    r"""
    Return the normalized even Gaussian moment.

    Inputs
    ----------
    alpha: symbolic or float
        Positive Gaussian width parameter.

    Outputs
    ----------
    moment: expression or float
        Integral[x^(2k) exp(-alpha x^2)] / Integral[exp(-alpha x^2)].
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    moment = ...
    # ---------------------------------------------------------------

    return moment
'''
    target = f'''
def answer(alpha):
    r"""
    Return the normalized even Gaussian moment.
    """
    moment = {numerator} / ({denominator} * alpha**{k})
    return moment
'''
    setup = (
        "Let $X$ be a zero-mean continuous variable with unnormalized density "
        "$\\exp(-\\alpha x^2)$."
    )
    main = (
        f"Compute the normalized moment $\\langle x^{2 * k}\\rangle$ as a function of $\\alpha$. "
        "Use the normalized even-moment formula "
        "$\\langle x^{2k}\\rangle=(2k-1)!!/(2^k\\alpha^k)$; keep the full "
        "$2^k$ factor in the denominator."
    )
    return _example(
        problem_id=f"{split}_symbolic_gaussian_moment_{idx:05d}_{k}",
        split=split,
        family="symbolic",
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
                    "args": [_sym_arg("alpha")],
                    "expected": expected,
                    "variables": ["alpha"],
                    "tolerance": 1e-8,
                }
            ],
        },
        trace=(
            f"For k={k}, use the standard even-moment formula "
            f"(2k-1)!!/(2^k alpha^k) = {numerator}/({denominator}*alpha**{k})."
        ),
        metadata={"domain": "amo_math", "official_overlap": "none"},
    )


def _two_level_gap(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    scale = rng.randint(1, 4)
    expected = f"sqrt(delta**2 + 4*({scale}*g)**2)"
    template = '''
def answer(delta, g):
    r"""
    Return the energy gap of a two-level Hamiltonian.

    Inputs
    ----------
    delta: symbolic or float
        Detuning.
    g: symbolic or float
        Coupling strength.

    Outputs
    ----------
    gap: expression or float
        Difference between the two eigenvalues.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    gap = ...
    # ---------------------------------------------------------------

    return gap
'''
    target = f'''
def answer(delta, g):
    r"""
    Return the energy gap of a two-level Hamiltonian.
    """
    import sympy as sp
    gap = sp.sqrt(delta**2 + 4 * ({scale} * g)**2)
    return gap
'''
    setup = (
        "An effective two-level Hamiltonian has diagonal detuning $\\pm \\delta/2$ "
        f"and off-diagonal coupling ${scale}g$, i.e. "
        f"$H=\\begin{{pmatrix}}\\delta/2 & {scale}g \\\\ {scale}g & -\\delta/2\\end{{pmatrix}}$."
    )
    main = (
        "Return the symbolic energy gap between the upper and lower eigenvalues. "
        "The gap is the positive difference $E_+-E_-$; the coupling contribution "
        f"adds inside the square root as $({scale}g)^2$, independent of $\\delta$. "
        "Do not multiply the coupling term by detuning."
    )
    return _example(
        problem_id=f"{split}_symbolic_two_level_gap_{idx:05d}_{scale}",
        split=split,
        family="symbolic",
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
                    "args": [_sym_arg("delta"), _sym_arg("g")],
                    "expected": expected,
                    "variables": ["delta", "g"],
                    "tolerance": 1e-8,
                }
            ],
        },
        trace=(
            f"The eigenvalues are +/- sqrt((delta/2)^2 + ({scale}g)^2), so the gap is "
            f"2*sqrt((delta/2)^2 + ({scale}g)^2) = sqrt(delta**2 + 4*({scale}*g)**2)."
        ),
        metadata={"domain": "quantum_two_level", "official_overlap": "none"},
    )


def _matrix_invariant(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(1, 5)
    b = rng.randint(1, 5)
    c = rng.randint(1, 5)
    expected_trace = float(2 * a)
    expected_det = float(a * a - b * c)
    template = '''
def answer():
    r"""
    Return [trace(H), det(H)] for the specified 2 by 2 matrix.

    Inputs
    ----------
    None

    Outputs
    ----------
    invariants: list[float]
        [trace, determinant].
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    invariants = ...
    # ---------------------------------------------------------------

    return invariants
'''
    target = f'''
def answer():
    r"""
    Return [trace(H), det(H)] for the specified 2 by 2 matrix.
    """
    trace = {expected_trace!r}
    determinant = {expected_det!r}
    invariants = [trace, determinant]
    return invariants
'''
    setup = f"Consider the matrix $H=\\begin{{pmatrix}} {a} & {b} \\\\ {c} & {a} \\end{{pmatrix}}$."
    main = "Return the trace and determinant as a length-2 list."
    return _example(
        problem_id=f"{split}_symbolic_matrix_invariant_{idx:05d}_{a}_{b}_{c}",
        split=split,
        family="symbolic",
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
                    "mode": "numeric_sequence",
                    "expected": [expected_trace, expected_det],
                    "tolerance": 1e-8,
                }
            ],
        },
        trace=(
            f"For [[{a},{b}],[{c},{a}]], trace={a}+{a}={expected_trace!r} and "
            f"determinant={a}*{a}-{b}*{c}={expected_det!r}."
        ),
        metadata={"domain": "linear_algebra_physics", "official_overlap": "none"},
    )


def _rational_response(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    gamma = rng.randint(1, 5)
    omega0 = rng.randint(2, 8)
    expected = f"1/(({omega0}**2 - omega**2)**2 + ({gamma}*omega)**2)"
    template = '''
def answer(omega):
    r"""
    Return the squared response amplitude.

    Inputs
    ----------
    omega: symbolic or float
        Probe frequency.

    Outputs
    ----------
    response: expression or float
        Squared response amplitude.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    response = ...
    # ---------------------------------------------------------------

    return response
'''
    target = f'''
def answer(omega):
    r"""
    Return the squared response amplitude.
    """
    response = 1 / ((({omega0})**2 - omega**2)**2 + ({gamma} * omega)**2)
    return response
'''
    setup = (
        "A damped oscillator has response amplitude "
        "$A(\\omega)=1/(\\omega_0^2-\\omega^2+i\\gamma\\omega)$ with "
        f"$\\omega_0={omega0}$ and $\\gamma={gamma}$."
    )
    main = "Return $|A(\\omega)|^2$ as a symbolic expression."
    return _example(
        problem_id=f"{split}_symbolic_rational_response_{idx:05d}_{omega0}_{gamma}",
        split=split,
        family="symbolic",
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
                    "args": [_sym_arg("omega")],
                    "expected": expected,
                    "variables": ["omega"],
                    "tolerance": 1e-8,
                }
            ],
        },
        trace="Take the modulus squared of the complex denominator.",
        metadata={"domain": "classical_response", "official_overlap": "none"},
    )


def _steady_state_elimination(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(1, 5)
    b = rng.randint(1, 5)
    c = rng.randint(1, 5)
    d = rng.randint(1, 5)
    expected = f"{d}*drive/({a}*{d} + {b}*{c})"
    template = '''
def answer(drive):
    r"""
    Return the steady-state amplitude x after eliminating y.

    Inputs
    ----------
    drive: symbolic or float
        External drive amplitude.

    Outputs
    ----------
    x: expression or float
        Steady-state value of x.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    x = ...
    # ---------------------------------------------------------------

    return x
'''
    target = f'''
def answer(drive):
    r"""
    Return the steady-state amplitude x after eliminating y.
    """
    x = {d} * drive / ({a} * {d} + {b} * {c})
    return x
'''
    setup = (
        "A linearized two-mode steady state is described by "
        f"${a}x+{b}y=\\Omega$ and ${c}x-{d}y=0$, where $\\Omega$ is the drive."
    )
    main = "Eliminate y and return x as a function of the drive variable."
    return _example(
        problem_id=f"{split}_symbolic_steady_state_elimination_{idx:05d}_{a}_{b}_{c}_{d}",
        split=split,
        family="symbolic",
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
                    "args": [_sym_arg("drive")],
                    "expected": expected,
                    "variables": ["drive"],
                    "tolerance": 1e-8,
                }
            ],
        },
        trace="From c*x=d*y, y=c*x/d. Substitute into a*x+b*y=drive.",
        metadata={"domain": "linear_response_elimination", "official_overlap": "none"},
    )


def _multi_output_linear_response(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(2, 7)
    b = rng.randint(1, 5)
    c = rng.randint(1, 5)
    d = rng.randint(2, 7)
    drive = rng.randint(2, 9)
    denom = a * d + b * c
    x = d * drive / denom
    y = c * drive / denom
    gain = d / denom
    expected = [x, y, gain]
    template = '''
def answer(drive):
    r"""
    Return the two steady-state amplitudes and the x-channel gain.

    Inputs
    ----------
    drive: float
        External drive amplitude.

    Outputs
    ----------
    result: tuple[float, float, float]
        (x, y, x_over_drive)
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    result = ...
    # ---------------------------------------------------------------

    return result
'''
    target = f'''
def answer(drive):
    r"""
    Return the two steady-state amplitudes and the x-channel gain.
    """
    denom = {a} * {d} + {b} * {c}
    x = {d} * drive / denom
    y = {c} * drive / denom
    gain = x / drive
    return (x, y, gain)
'''
    setup = (
        "A coupled linear response model is described by "
        f"${a}x+{b}y=\\Omega$ and ${c}x-{d}y=0$. "
        "The output report should include both solved amplitudes, not just x."
    )
    main = (
        f"For drive value $\\Omega={drive}$, return `(x, y, x_over_drive)` as floats. "
        "Solve the coupled equations before forming the tuple."
    )
    return _example(
        problem_id=f"{split}_symbolic_multi_output_linear_response_{idx:05d}_{a}_{b}_{c}_{d}_{drive}",
        split=split,
        family="symbolic",
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
                    "mode": "numeric_sequence",
                    "args": [drive],
                    "expected": expected,
                    "tolerance": 1e-8,
                }
            ],
        },
        trace="Eliminate y with y=c*x/d, solve x=d*drive/(a*d+b*c), then return x, y, and x/drive.",
        metadata={"domain": "linear_response_elimination", "official_overlap": "none", "variant": "multi_output"},
    )


def _markov_chain(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    p = rng.choice([0.1, 0.2, 0.25, 0.3])
    q = rng.choice([0.05, 0.1, 0.15, 0.2])
    steps = rng.randint(4, 8 if difficulty != "hard" else 14)
    state = [1.0, 0.0]
    for _ in range(steps):
        state = [state[0] * (1 - p) + state[1] * q, state[0] * p + state[1] * (1 - q)]
    expected = state[1]
    template = '''
def answer():
    r"""
    Return the probability of being in state 1 after n steps.

    Inputs
    ----------
    None

    Outputs
    ----------
    prob: float
        Probability of state 1.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    prob = ...
    # ---------------------------------------------------------------

    return prob
'''
    target = f'''
def answer():
    r"""
    Return the probability of being in state 1 after n steps.
    """
    p = {p!r}
    q = {q!r}
    state0, state1 = 1.0, 0.0
    for _ in range({steps}):
        state0, state1 = state0 * (1 - p) + state1 * q, state0 * p + state1 * (1 - q)
    prob = state1
    return prob
'''
    setup = (
        "A two-state stochastic dynamics starts in state 0. In each step, "
        f"0 -> 1 with probability {p}, and 1 -> 0 with probability {q}."
    )
    main = (
        f"Compute the probability of being in state 1 after {steps} steps. Use the update "
        "$s_0' = s_0(1-p)+s_1q$ and $s_1' = s_0p+s_1(1-q)$ in order for every step; "
        "do not use the one-step probability as the final answer."
    )
    return _example(
        problem_id=f"{split}_numeric_markov_chain_{idx:05d}_{p}_{q}_{steps}",
        split=split,
        family="numeric",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={
            "kind": "code",
            "timeout_s": 2.0,
            "checks": [{"mode": "numeric", "expected": expected, "tolerance": 1e-10}],
        },
        trace="Iterate the two-state transition matrix for the requested number of steps.",
        metadata={"domain": "stochastic_physics", "official_overlap": "none"},
    )


def _transfer_matrix(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(1, 4)
    b = rng.randint(0, 3)
    c = rng.randint(1, 4)
    d = rng.randint(1, 4)
    power = rng.randint(3, 6 if difficulty != "hard" else 9)
    matrix = [[a, b], [c, d]]
    mat_power = _mat_pow_2x2(matrix, power)
    expected = float(mat_power[0][0] + mat_power[1][1])
    template = '''
def answer():
    r"""
    Return trace(T^n) for the given 2 by 2 transfer matrix.

    Inputs
    ----------
    None

    Outputs
    ----------
    trace: float
        Trace of the matrix power.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    trace = ...
    # ---------------------------------------------------------------

    return trace
'''
    target = f'''
def answer():
    r"""
    Return trace(T^n) for the given 2 by 2 transfer matrix.
    """
    T = [[{a}, {b}], [{c}, {d}]]
    M = [[1, 0], [0, 1]]
    for _ in range({power}):
        M = [
            [M[0][0] * T[0][0] + M[0][1] * T[1][0], M[0][0] * T[0][1] + M[0][1] * T[1][1]],
            [M[1][0] * T[0][0] + M[1][1] * T[1][0], M[1][0] * T[0][1] + M[1][1] * T[1][1]],
        ]
    trace = float(M[0][0] + M[1][1])
    return trace
'''
    setup = f"Let $T=\\begin{{pmatrix}} {a} & {b} \\\\ {c} & {d} \\end{{pmatrix}}$ be a transfer matrix."
    main = (
        f"Compute $\\mathrm{{tr}}(T^{power})$. Use full 2x2 matrix multiplication at each "
        "power: every output entry is a row-column dot product. Do not multiply entries "
        "elementwise and do not drop cross terms."
    )
    return _example(
        problem_id=f"{split}_numeric_transfer_matrix_{idx:05d}_{a}_{b}_{c}_{d}_{power}",
        split=split,
        family="numeric",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={
            "kind": "code",
            "timeout_s": 2.0,
            "checks": [{"mode": "numeric", "expected": expected, "tolerance": 1e-8}],
        },
        trace="Compute T^n by repeated 2x2 multiplication and return the trace.",
        metadata={"domain": "condensed_matter_transfer", "official_overlap": "none"},
    )


def _spectral_trace_power(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    n = 3 if difficulty != "hard" else 4
    diag = [rng.randint(-3, 4) for _ in range(n)]
    offdiag = [rng.randint(1, 3) for _ in range(n - 1)]
    power = 2 if difficulty == "easy" else rng.choice([2, 3, 4])
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for row in range(n):
        matrix[row][row] = diag[row]
    for row, value in enumerate(offdiag):
        matrix[row][row + 1] = value
        matrix[row + 1][row] = value
    mat_power = _mat_pow(matrix, power)
    expected = float(sum(mat_power[i][i] for i in range(n)))
    template = '''
def answer():
    r"""
    Return trace(H^p) for the specified small tridiagonal Hamiltonian.

    Inputs
    ----------
    None

    Outputs
    ----------
    trace_power: float
        Trace of the requested matrix power.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    trace_power = ...
    # ---------------------------------------------------------------

    return trace_power
'''
    target = f'''
def answer():
    r"""
    Return trace(H^p) for the specified small tridiagonal Hamiltonian.
    """
    H = {matrix!r}
    n = len(H)
    M = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    for _ in range({power}):
        M = [
            [sum(M[i][k] * H[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)
        ]
    trace_power = float(sum(M[i][i] for i in range(n)))
    return trace_power
'''
    setup = (
        f"Consider a small tridiagonal Hamiltonian with diagonal entries {diag} "
        f"and nearest-neighbor couplings {offdiag}."
    )
    main = (
        f"Compute $\\mathrm{{tr}}(H^{power})$. Use full matrix multiplication. For $H^2$, "
        "each diagonal entry is $\\sum_k H_{ik}H_{ki}$, so off-diagonal couplings contribute "
        "to the trace; do not keep only squared diagonal entries."
    )
    return _example(
        problem_id=f"{split}_numeric_spectral_trace_power_{idx:05d}_{n}_{power}_{'_'.join(map(str, diag + offdiag))}",
        split=split,
        family="numeric",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={
            "kind": "code",
            "timeout_s": 2.0,
            "checks": [{"mode": "numeric", "expected": expected, "tolerance": 1e-8}],
        },
        trace="Multiply the small Hamiltonian p times, then sum the diagonal.",
        metadata={"domain": "small_hamiltonian_spectrum", "official_overlap": "none"},
    )


def _linear_recurrence(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    u0 = rng.randint(0, 3)
    u1 = rng.randint(1, 4)
    a = rng.randint(1, 3)
    b = rng.randint(1, 3)
    n = rng.randint(6, 10 if difficulty != "hard" else 18)
    values = [u0, u1]
    for _ in range(2, n + 1):
        values.append(a * values[-1] + b * values[-2])
    expected = values[n]
    template = '''
def answer():
    r"""
    Return u_n for the specified linear recurrence.

    Inputs
    ----------
    None

    Outputs
    ----------
    value: int
        The requested recurrence value.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    value = ...
    # ---------------------------------------------------------------

    return value
'''
    target = f'''
def answer():
    r"""
    Return u_n for the specified linear recurrence.
    """
    values = [{u0}, {u1}]
    for _ in range(2, {n} + 1):
        values.append({a} * values[-1] + {b} * values[-2])
    value = values[{n}]
    return value
'''
    setup = f"A mode amplitude obeys $u_0={u0}$, $u_1={u1}$ and $u_n={a}u_{{n-1}}+{b}u_{{n-2}}$."
    main = (
        f"Return $u_{n}$. Use the recurrence exactly as written, compute the "
        f"intermediate values $u_2,\\ldots,u_{n}$ in order, and do not skip the "
        "last update. Each update is $u_i=a u_{i-1}+b u_{i-2}$ with the same a and b "
        "from the problem statement."
    )
    return _example(
        problem_id=f"{split}_numeric_linear_recurrence_{idx:05d}_{u0}_{u1}_{a}_{b}_{n}",
        split=split,
        family="numeric",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={
            "kind": "code",
            "timeout_s": 2.0,
            "checks": [{"mode": "exact", "expected": expected}],
        },
        trace="Iterate the recurrence up to the requested index.",
        metadata={"domain": "many_body_recurrence", "official_overlap": "none"},
    )


def _markov_chain_audit(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    p = rng.choice([0.1, 0.2, 0.25, 0.3])
    q = rng.choice([0.05, 0.1, 0.15, 0.2])
    steps = rng.randint(5, 9 if difficulty != "hard" else 13)
    state0, state1 = 1.0, 0.0
    trajectory: list[float] = []
    for _ in range(steps):
        state0, state1 = state0 * (1 - p) + state1 * q, state0 * p + state1 * (1 - q)
        trajectory.extend([state0, state1])
    template = '''
def answer():
    r"""
    Return the flattened Markov trajectory [s0_1, s1_1, ..., s0_n, s1_n].

    Inputs
    ----------
    None

    Outputs
    ----------
    trajectory: list[float]
        Flattened per-step state probabilities. The final state-1 probability is the last entry.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    trajectory = ...
    # ---------------------------------------------------------------

    return trajectory
'''
    target = f'''
def answer():
    r"""
    Return the flattened Markov trajectory [s0_1, s1_1, ..., s0_n, s1_n].
    """
    p = {p!r}
    q = {q!r}
    state0, state1 = 1.0, 0.0
    trajectory = []
    for _ in range({steps}):
        state0, state1 = state0 * (1 - p) + state1 * q, state0 * p + state1 * (1 - q)
        trajectory.extend([state0, state1])
    return trajectory
'''
    setup = (
        "A two-state stochastic dynamics starts in state 0. In each step, "
        f"0 -> 1 with probability {p}, and 1 -> 0 with probability {q}."
    )
    main = (
        f"Compute the full trajectory for {steps} steps using "
        "$s_0' = s_0(1-p)+s_1q$ and $s_1' = s_0p+s_1(1-q)$. "
        f"Return exactly a length-{2 * steps} flattened list "
        "`[s0_after_1, s1_after_1, ..., s0_after_n, s1_after_n]`. "
        "The final probability of state 1 is the last entry; do not return only the last entry."
    )
    return _example(
        problem_id=f"{split}_numeric_markov_chain_audit_{idx:05d}_{p}_{q}_{steps}",
        split=split,
        family="numeric",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(trajectory, 1e-8)},
        trace=f"Iterate from (1,0); flattened trajectory is {trajectory!r}.",
        metadata={"domain": "stochastic_physics", "official_overlap": "none", "variant": "intermediate_trajectory"},
    )


def _transfer_matrix_audit(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(1, 4)
    b = rng.randint(0, 3)
    c = rng.randint(1, 4)
    d = rng.randint(1, 4)
    power = rng.randint(3, 6 if difficulty != "hard" else 8)
    matrix = [[a, b], [c, d]]
    t2 = _mat_pow_2x2(matrix, 2)
    tn = _mat_pow_2x2(matrix, power)
    expected = [float(tn[0][0] + tn[1][1]), *[float(value) for row in t2 for value in row], *[float(value) for row in tn for value in row]]
    template = '''
def answer():
    r"""
    Return [trace(T^n), T2_00, T2_01, T2_10, T2_11, Tn_00, Tn_01, Tn_10, Tn_11].

    Inputs
    ----------
    None

    Outputs
    ----------
    audit: list[float]
        Trace plus row-major T^2 and T^n entries.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    audit = ...
    # ---------------------------------------------------------------

    return audit
'''
    target = f'''
def answer():
    r"""
    Return [trace(T^n), row-major T^2, row-major T^n].
    """
    T = [[{a}, {b}], [{c}, {d}]]
    M = [[1, 0], [0, 1]]
    T2 = None
    for step in range(1, {power} + 1):
        M = [
            [M[0][0] * T[0][0] + M[0][1] * T[1][0], M[0][0] * T[0][1] + M[0][1] * T[1][1]],
            [M[1][0] * T[0][0] + M[1][1] * T[1][0], M[1][0] * T[0][1] + M[1][1] * T[1][1]],
        ]
        if step == 2:
            T2 = [row[:] for row in M]
    audit = [float(M[0][0] + M[1][1])]
    audit.extend(float(value) for row in T2 for value in row)
    audit.extend(float(value) for row in M for value in row)
    return audit
'''
    setup = f"Let $T=\\begin{{pmatrix}} {a} & {b} \\\\ {c} & {d} \\end{{pmatrix}}$ be a transfer matrix."
    main = (
        f"Compute $T^2$, $T^{power}$, and $\\mathrm{{tr}}(T^{power})$ using full row-column "
        "matrix multiplication. Return exactly "
        "`[trace(T^n), T2_00, T2_01, T2_10, T2_11, Tn_00, Tn_01, Tn_10, Tn_11]`. "
        "This audit list is required so cross terms cannot be skipped."
    )
    return _example(
        problem_id=f"{split}_numeric_transfer_matrix_audit_{idx:05d}_{a}_{b}_{c}_{d}_{power}",
        split=split,
        family="numeric",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(expected, 1e-8)},
        trace=f"T^2={t2!r}, T^{power}={tn!r}, trace={expected[0]!r}.",
        metadata={"domain": "condensed_matter_transfer", "official_overlap": "none", "variant": "intermediate_matrix"},
    )


def _spectral_trace_power_audit(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    n = 3 if difficulty != "hard" else 4
    diag = [rng.randint(-3, 4) for _ in range(n)]
    offdiag = [rng.randint(1, 3) for _ in range(n - 1)]
    power = rng.choice([2, 3, 4])
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for row in range(n):
        matrix[row][row] = diag[row]
    for row, value in enumerate(offdiag):
        matrix[row][row + 1] = value
        matrix[row + 1][row] = value
    h2 = _mat_pow(matrix, 2)
    hp = _mat_pow(matrix, power)
    h2_diag = [float(h2[i][i]) for i in range(n)]
    hp_diag = [float(hp[i][i]) for i in range(n)]
    expected = [float(sum(hp_diag)), *h2_diag, *hp_diag]
    template = '''
def answer():
    r"""
    Return [trace(H^p), diag(H^2)..., diag(H^p)...].

    Inputs
    ----------
    None

    Outputs
    ----------
    audit: list[float]
        Trace plus diagnostic diagonal entries.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    audit = ...
    # ---------------------------------------------------------------

    return audit
'''
    target = f'''
def answer():
    r"""
    Return [trace(H^p), diag(H^2), diag(H^p)].
    """
    H = {matrix!r}
    n = len(H)
    M = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    H2 = None
    for step in range(1, {power} + 1):
        M = [
            [sum(M[i][k] * H[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)
        ]
        if step == 2:
            H2 = [row[:] for row in M]
    audit = [float(sum(M[i][i] for i in range(n)))]
    audit.extend(float(H2[i][i]) for i in range(n))
    audit.extend(float(M[i][i]) for i in range(n))
    return audit
'''
    setup = (
        f"Consider a tridiagonal Hamiltonian with diagonal entries {diag} "
        f"and nearest-neighbor couplings {offdiag}."
    )
    main = (
        f"Compute $H^2$, $H^{power}$, and $\\mathrm{{tr}}(H^{power})$ using full matrix multiplication. "
        "Return exactly `[trace(H^p), diag(H^2)..., diag(H^p)...]`. "
        "For H^2, each diagonal entry must include neighboring off-diagonal couplings."
    )
    return _example(
        problem_id=f"{split}_numeric_spectral_trace_power_audit_{idx:05d}_{n}_{power}_{'_'.join(map(str, diag + offdiag))}",
        split=split,
        family="numeric",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(expected, 1e-8)},
        trace=f"H^2 diagonal={h2_diag!r}, H^{power} diagonal={hp_diag!r}, trace={expected[0]!r}.",
        metadata={"domain": "small_hamiltonian_spectrum", "official_overlap": "none", "variant": "intermediate_diagonal"},
    )


def _series_coefficients_audit(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(1, 6)
    b = rng.randint(1, 6)
    order = 4 if difficulty != "hard" else 6
    coeffs = _series_coeffs_exp_over_pole(a, b, order)
    template = f'''
def answer():
    r"""
    Return the full coefficient list [c0, ..., c{order}].

    Inputs
    ----------
    None

    Outputs
    ----------
    coeffs: list[float]
        Taylor coefficients through order {order}.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    coeffs = ...
    # ---------------------------------------------------------------

    return coeffs
'''
    target = f'''
def answer():
    r"""
    Return the full coefficient list [c0, ..., c{order}].
    """
    coeffs = {coeffs!r}
    return coeffs
'''
    setup = f"Let $f(x)=\\exp({a}x)/(1-{b}x)$."
    main = (
        f"Return the complete coefficient list $[c_0,\\ldots,c_{order}]$ with exactly {order + 1} entries. "
        "$c_n=\\sum_{k=0}^{n} a^k/k!\\,b^{n-k}$. Do not return only nonzero terms, "
        "a total, or a prose explanation."
    )
    return _example(
        problem_id=f"{split}_template_series_coefficients_audit_{idx:05d}_{a}_{b}_{order}",
        split=split,
        family="template",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(coeffs, 1e-8)},
        trace=f"The coefficient list is {coeffs!r}.",
        metadata={"domain": "perturbation_series", "official_overlap": "none", "variant": "item_checked_coefficients"},
    )


def _series_diagnostics_audit(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(1, 6)
    b = rng.randint(1, 6)
    order = 4 if difficulty != "hard" else 6
    coeffs = _series_coeffs_exp_over_pole(a, b, order)
    diagnostics = [float(sum(coeffs)), coeffs[1], coeffs[-1] / coeffs[-2]]
    expected = [*diagnostics, *coeffs]
    template = '''
def answer():
    r"""
    Return [sum_of_coeffs, c1, c_last_over_previous, c0, c1, ...].

    Inputs
    ----------
    None

    Outputs
    ----------
    audit: list[float]
        Diagnostics followed by the full coefficient list.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    audit = ...
    # ---------------------------------------------------------------

    return audit
'''
    target = f'''
def answer():
    r"""
    Return diagnostics followed by the full coefficient list.
    """
    coeffs = {coeffs!r}
    audit = [sum(coeffs), coeffs[1], coeffs[-1] / coeffs[-2], *coeffs]
    return audit
'''
    setup = f"Let $f(x)=\\exp({a}x)/(1-{b}x)$ and compute coefficients through order {order}."
    main = (
        "First compute the full coefficient list using the convolution formula. "
        "Then return exactly `[sum_of_coeffs, c1, c_last_over_previous, c0, c1, ..., c_order]`. "
        "The diagnostics must be consistent with the coefficient list."
    )
    return _example(
        problem_id=f"{split}_template_series_diagnostics_audit_{idx:05d}_{a}_{b}_{order}",
        split=split,
        family="template",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(expected, 1e-8)},
        trace=f"coeffs={coeffs!r}; diagnostics={diagnostics!r}; audit={expected!r}.",
        metadata={"domain": "perturbation_series", "official_overlap": "none", "variant": "diagnostics_with_coefficients"},
    )


def _hamming_weight_enumerator_audit(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    n = rng.randint(6, 8 if difficulty != "hard" else 10)
    modulus = rng.choice([2, 3, 4])
    residue = rng.randint(0, modulus - 1)
    weights = [0 for _ in range(n + 1)]
    for mask in range(1 << n):
        weight = mask.bit_count()
        if weight % modulus == residue:
            weights[weight] += 1
    template = '''
def answer():
    r"""
    Return the full Hamming-weight enumerator coefficient list.

    Inputs
    ----------
    None

    Outputs
    ----------
    coeffs: list[int]
        coeffs[w] is the accepted count for exact weight w.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    coeffs = ...
    # ---------------------------------------------------------------

    return coeffs
'''
    target = f'''
def answer():
    r"""
    Return the full Hamming-weight enumerator coefficient list.
    """
    coeffs = {weights!r}
    return coeffs
'''
    setup = (
        f"Consider all binary strings of length {n}. A string is accepted when its "
        f"Hamming weight is congruent to {residue} modulo {modulus}."
    )
    main = (
        f"Return exactly the full length-{n + 1} list `[coeffs[0], ..., coeffs[{n}]]`. "
        "Weights that do not satisfy the congruence must still appear as 0. "
        f"Do not add a position for weight {n + 1}; the last valid index is {n}."
    )
    return _example(
        problem_id=f"{split}_discrete_hamming_weight_enumerator_audit_{idx:05d}_{n}_{modulus}_{residue}",
        split=split,
        family="discrete",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(weights, 1e-8)},
        trace=f"The full enumerator is {weights!r}.",
        metadata={"domain": "toy_qec_enumerator", "official_overlap": "none", "variant": "item_checked_full_list"},
    )


def _markov_chain_compact_audit(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    p = rng.choice([0.1, 0.2, 0.25, 0.3])
    q = rng.choice([0.05, 0.1, 0.15, 0.2])
    steps = rng.randint(5, 7 if difficulty != "hard" else 9)
    mid = max(1, steps // 2)
    state0, state1 = 1.0, 0.0
    s1_mid = 0.0
    for step in range(1, steps + 1):
        state0, state1 = state0 * (1 - p) + state1 * q, state0 * p + state1 * (1 - q)
        if step == mid:
            s1_mid = state1
    expected = [s1_mid, state1]
    template = '''
def answer():
    r"""
    Return [s1_mid, s1_final].

    Inputs
    ----------
    None

    Outputs
    ----------
    audit: list[float]
        State-1 probability at the midpoint and final step.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    audit = ...
    # ---------------------------------------------------------------

    return audit
'''
    target = f'''
def answer():
    r"""
    Return [s1_mid, s1_final].
    """
    p = {p!r}
    q = {q!r}
    state0, state1 = 1.0, 0.0
    s1_mid = None
    for step in range(1, {steps} + 1):
        state0, state1 = state0 * (1 - p) + state1 * q, state0 * p + state1 * (1 - q)
        if step == {mid}:
            s1_mid = state1
    return [s1_mid, state1]
'''
    setup = (
        "A two-state stochastic dynamics starts in state 0. In each step, "
        f"0 -> 1 with probability {p}, and 1 -> 0 with probability {q}."
    )
    main = (
        f"Iterate for {steps} steps with $s_0'=s_0(1-p)+s_1q$ and $s_1'=s_0p+s_1(1-q)$. "
        f"Return exactly `[s1_after_{mid}, s1_after_{steps}]`. "
        "This is a short audit tuple; do not print the full trajectory."
    )
    return _example(
        problem_id=f"{split}_numeric_markov_chain_compact_audit_{idx:05d}_{p}_{q}_{steps}",
        split=split,
        family="numeric",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(expected, 1e-8)},
        trace=f"Track s1 at step {mid} and {steps}: {expected!r}.",
        metadata={"domain": "stochastic_physics", "official_overlap": "none", "variant": "compact_intermediate"},
    )


def _transfer_matrix_compact_audit(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(1, 4)
    b = rng.randint(0, 3)
    c = rng.randint(1, 4)
    d = rng.randint(1, 4)
    power = rng.randint(3, 5 if difficulty != "hard" else 7)
    matrix = [[a, b], [c, d]]
    t2 = _mat_pow_2x2(matrix, 2)
    tn = _mat_pow_2x2(matrix, power)
    expected = [float(tn[0][0] + tn[1][1]), float(t2[0][1]), float(tn[1][0])]
    template = '''
def answer():
    r"""
    Return [trace(T^n), T2_01, Tn_10].

    Inputs
    ----------
    None

    Outputs
    ----------
    audit: list[float]
        Final trace plus two cross-term-sensitive entries.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    audit = ...
    # ---------------------------------------------------------------

    return audit
'''
    target = f'''
def answer():
    r"""
    Return [trace(T^n), T2_01, Tn_10].
    """
    T = [[{a}, {b}], [{c}, {d}]]
    M = [[1, 0], [0, 1]]
    T2 = None
    for step in range(1, {power} + 1):
        M = [
            [M[0][0] * T[0][0] + M[0][1] * T[1][0], M[0][0] * T[0][1] + M[0][1] * T[1][1]],
            [M[1][0] * T[0][0] + M[1][1] * T[1][0], M[1][0] * T[0][1] + M[1][1] * T[1][1]],
        ]
        if step == 2:
            T2 = [row[:] for row in M]
    return [float(M[0][0] + M[1][1]), float(T2[0][1]), float(M[1][0])]
'''
    setup = f"Let $T=\\begin{{pmatrix}} {a} & {b} \\\\ {c} & {d} \\end{{pmatrix}}$ be a transfer matrix."
    main = (
        f"Compute $T^2$, $T^{power}$, and $\\mathrm{{tr}}(T^{power})$ with full row-column multiplication. "
        f"Return exactly `[trace(T^{power}), (T^2)_01, (T^{power})_10]`. "
        "These two off-diagonal entries are included to catch missing cross terms."
    )
    return _example(
        problem_id=f"{split}_numeric_transfer_matrix_compact_audit_{idx:05d}_{a}_{b}_{c}_{d}_{power}",
        split=split,
        family="numeric",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(expected, 1e-8)},
        trace=f"T^2[0,1]={t2[0][1]}, T^{power}[1,0]={tn[1][0]}, trace={expected[0]}.",
        metadata={"domain": "condensed_matter_transfer", "official_overlap": "none", "variant": "compact_intermediate"},
    )


def _spectral_trace_power_compact_audit(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    n = 3 if difficulty != "hard" else 4
    diag = [rng.randint(-3, 4) for _ in range(n)]
    offdiag = [rng.randint(1, 3) for _ in range(n - 1)]
    power = rng.choice([2, 3, 4])
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for row in range(n):
        matrix[row][row] = diag[row]
    for row, value in enumerate(offdiag):
        matrix[row][row + 1] = value
        matrix[row + 1][row] = value
    h2 = _mat_pow(matrix, 2)
    hp = _mat_pow(matrix, power)
    h2_trace = float(sum(h2[i][i] for i in range(n)))
    hp_diag0 = float(hp[0][0])
    trace = float(sum(hp[i][i] for i in range(n)))
    expected = [trace, h2_trace, hp_diag0]
    template = '''
def answer():
    r"""
    Return [trace(H^p), trace(H^2), (H^p)_00].

    Inputs
    ----------
    None

    Outputs
    ----------
    audit: list[float]
        Final trace plus two intermediate diagnostics.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    audit = ...
    # ---------------------------------------------------------------

    return audit
'''
    target = f'''
def answer():
    r"""
    Return [trace(H^p), trace(H^2), (H^p)_00].
    """
    H = {matrix!r}
    n = len(H)
    M = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    H2 = None
    for step in range(1, {power} + 1):
        M = [
            [sum(M[i][k] * H[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)
        ]
        if step == 2:
            H2 = [row[:] for row in M]
    return [float(sum(M[i][i] for i in range(n))), float(sum(H2[i][i] for i in range(n))), float(M[0][0])]
'''
    setup = (
        f"Consider a tridiagonal Hamiltonian with diagonal entries {diag} "
        f"and nearest-neighbor couplings {offdiag}."
    )
    main = (
        f"Compute $H^2$, $H^{power}$, and $\\mathrm{{tr}}(H^{power})$ with full matrix multiplication. "
        f"Return exactly `[trace(H^{power}), trace(H^2), (H^{power})_00]`. "
        "The trace(H^2) diagnostic should include off-diagonal coupling contributions."
    )
    return _example(
        problem_id=f"{split}_numeric_spectral_trace_power_compact_audit_{idx:05d}_{n}_{power}_{'_'.join(map(str, diag + offdiag))}",
        split=split,
        family="numeric",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(expected, 1e-8)},
        trace=f"trace(H^2)={h2_trace}, H^{power}[0,0]={hp_diag0}, trace={trace}.",
        metadata={"domain": "small_hamiltonian_spectrum", "official_overlap": "none", "variant": "compact_intermediate"},
    )


def _series_coefficients_compact_audit(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(1, 5)
    b = rng.randint(1, 5)
    order = 3 if difficulty != "hard" else 4
    coeffs = _series_coeffs_exp_over_pole(a, b, order)
    template = f'''
def answer():
    r"""
    Return [c0, ..., c{order}].

    Inputs
    ----------
    None

    Outputs
    ----------
    coeffs: list[float]
        Taylor coefficients through order {order}.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    coeffs = ...
    # ---------------------------------------------------------------

    return coeffs
'''
    target = f'''
def answer():
    r"""
    Return [c0, ..., c{order}].
    """
    return {coeffs!r}
'''
    setup = f"Let $f(x)=\\exp({a}x)/(1-{b}x)$."
    main = (
        f"Return exactly the coefficient list `[c0, ..., c{order}]` with {order + 1} entries. "
        "$c_n=\\sum_{k=0}^{n} a^k/k!\\,b^{n-k}$. Keep the answer as a list only."
    )
    return _example(
        problem_id=f"{split}_template_series_coefficients_compact_audit_{idx:05d}_{a}_{b}_{order}",
        split=split,
        family="template",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(coeffs, 1e-8)},
        trace=f"The coefficient list is {coeffs!r}.",
        metadata={"domain": "perturbation_series", "official_overlap": "none", "variant": "compact_coefficients"},
    )


def _series_diagnostics_compact_audit(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(1, 5)
    b = rng.randint(1, 5)
    order = 3 if difficulty != "hard" else 4
    coeffs = _series_coeffs_exp_over_pole(a, b, order)
    expected = [float(sum(coeffs)), coeffs[1], coeffs[-1] / coeffs[-2]]
    template = '''
def answer():
    r"""
    Return [sum_of_coeffs, c1, c_last_over_previous].

    Inputs
    ----------
    None

    Outputs
    ----------
    diagnostics: list[float]
        Three compact diagnostics.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    diagnostics = ...
    # ---------------------------------------------------------------

    return diagnostics
'''
    target = f'''
def answer():
    r"""
    Return [sum_of_coeffs, c1, c_last_over_previous].
    """
    coeffs = {coeffs!r}
    return [sum(coeffs), coeffs[1], coeffs[-1] / coeffs[-2]]
'''
    setup = f"Let $f(x)=\\exp({a}x)/(1-{b}x)$ and compute coefficients through order {order}."
    main = (
        "Compute the coefficients internally, but return only the compact diagnostic list "
        "`[sum_of_coeffs, c1, c_last_over_previous]`."
    )
    return _example(
        problem_id=f"{split}_template_series_diagnostics_compact_audit_{idx:05d}_{a}_{b}_{order}",
        split=split,
        family="template",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(expected, 1e-8)},
        trace=f"coeffs={coeffs!r}; diagnostics={expected!r}.",
        metadata={"domain": "perturbation_series", "official_overlap": "none", "variant": "compact_diagnostics"},
    )


def _hamming_weight_enumerator_compact_audit(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    n = rng.randint(5, 7 if difficulty != "hard" else 8)
    modulus = rng.choice([2, 3, 4])
    residue = rng.randint(0, modulus - 1)
    weights = [0 for _ in range(n + 1)]
    for mask in range(1 << n):
        weight = mask.bit_count()
        if weight % modulus == residue:
            weights[weight] += 1
    template = '''
def answer():
    r"""
    Return the full Hamming-weight enumerator coefficient list.

    Inputs
    ----------
    None

    Outputs
    ----------
    coeffs: list[int]
        coeffs[w] is the accepted count for exact weight w.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    coeffs = ...
    # ---------------------------------------------------------------

    return coeffs
'''
    target = f'''
def answer():
    r"""
    Return the full Hamming-weight enumerator coefficient list.
    """
    return {weights!r}
'''
    setup = (
        f"Consider all binary strings of length {n}. A string is accepted when its "
        f"Hamming weight is congruent to {residue} modulo {modulus}."
    )
    main = (
        f"Return exactly the full length-{n + 1} list `[coeffs[0], ..., coeffs[{n}]]`. "
        "Do not add an extra trailing position."
    )
    return _example(
        problem_id=f"{split}_discrete_hamming_weight_enumerator_compact_audit_{idx:05d}_{n}_{modulus}_{residue}",
        split=split,
        family="discrete",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_item_checks(weights, 1e-8)},
        trace=f"The full enumerator is {weights!r}.",
        metadata={"domain": "toy_qec_enumerator", "official_overlap": "none", "variant": "compact_full_list"},
    )


def _hamming_weight_enumerator(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    n = rng.randint(5, 8 if difficulty != "hard" else 11)
    modulus = rng.choice([2, 3, 4])
    residue = rng.randint(0, modulus - 1)
    weights = [0 for _ in range(n + 1)]
    for mask in range(1 << n):
        weight = mask.bit_count()
        if weight % modulus == residue:
            weights[weight] += 1
    template = '''
def answer():
    r"""
    Return the Hamming-weight enumerator coefficients.

    Inputs
    ----------
    None

    Outputs
    ----------
    coeffs: list[int]
        coeffs[w] is the number of accepted strings with Hamming weight w.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    coeffs = ...
    # ---------------------------------------------------------------

    return coeffs
'''
    target = f'''
def answer():
    r"""
    Return the Hamming-weight enumerator coefficients.
    """
    n = {n}
    coeffs = [0 for _ in range(n + 1)]
    for mask in range(1 << n):
        weight = mask.bit_count()
        if weight % {modulus} == {residue}:
            coeffs[weight] += 1
    return coeffs
'''
    setup = (
        f"Consider all binary strings of length {n}. A string is accepted when its "
        f"Hamming weight is congruent to {residue} modulo {modulus}."
    )
    main = (
        f"Return the full Hamming-weight enumerator coefficient list of length {n + 1}, "
        f"indexed by weights w=0,1,...,{n}. Entry coeffs[w] must be the number of "
        "accepted strings of exactly weight w; weights that do not satisfy the "
        "congruence must still appear as 0. Do not return only the nonzero accepted "
        "counts and do not return the total count."
    )
    return _example(
        problem_id=f"{split}_discrete_hamming_weight_enumerator_{idx:05d}_{n}_{modulus}_{residue}",
        split=split,
        family="discrete",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={
            "kind": "code",
            "timeout_s": 2.0,
            "checks": [{"mode": "exact_sequence", "expected": weights}],
        },
        trace="Enumerate all bit strings, count weights satisfying the congruence rule.",
        metadata={"domain": "toy_qec_enumerator", "official_overlap": "none"},
    )


def _oam_selection_rule(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    ell_a = rng.randint(-3, 3)
    ell_b = rng.randint(-3, 3)
    harmonic = rng.choice([9, 11, 13, 15, 17, 19, 21, 23])
    n_a = (harmonic + 1) // 2
    n_b = (harmonic - 1) // 2
    helicity = 1 if n_a >= n_b else -1
    oam = n_a * ell_a + n_b * ell_b
    template = '''
def answer():
    r"""
    Return the harmonic orbital angular momentum and helicity.

    Inputs
    ----------
    None

    Outputs
    ----------
    result: tuple[int, int]
        (ell_harmonic, sigma), where sigma is +1 or -1.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    result = ...
    # ---------------------------------------------------------------

    return result
'''
    target = f'''
def answer():
    r"""
    Return the harmonic orbital angular momentum and helicity.
    """
    result = ({oam}, {helicity})
    return result
'''
    setup = (
        f"Two synthetic drivers carry OAM ell_A={ell_a} and ell_B={ell_b}. "
        "For this toy selection rule, the q-th odd harmonic absorbs "
        "(q+1)/2 photons from A and (q-1)/2 photons from B."
    )
    main = f"For harmonic order q={harmonic}, return `(ell_harmonic, sigma)` with sigma=+1 for A-dominant helicity."
    return _example(
        problem_id=f"{split}_discrete_oam_selection_{idx:05d}_{ell_a}_{ell_b}_{harmonic}",
        split=split,
        family="discrete",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={
            "kind": "code",
            "timeout_s": 2.0,
            "checks": [{"mode": "exact", "expected": {"$tuple": [oam, helicity]}}],
        },
        trace="Use ell=n_A ell_A+n_B ell_B and sigma=+1 because n_A>=n_B for odd q.",
        metadata={"domain": "amo_selection_rule", "official_overlap": "none"},
    )


def _oam_selection_rule_algorithmic(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    ell_left = rng.randint(-5, 5)
    ell_right = rng.randint(-5, 5)
    q = rng.choice([7, 9, 11, 13, 15, 17, 19, 21, 23, 25])
    n_left = (q + 1) // 2
    n_right = (q - 1) // 2
    ell_harmonic = n_left * ell_left + n_right * ell_right
    spin = 1 if n_left >= n_right else -1
    template = '''
def answer():
    r"""
    Return the emitted harmonic OAM and spin label.

    Inputs
    ----------
    None

    Outputs
    ----------
    result: tuple[int, int]
        (ell_out, spin_label), where spin_label is +1 or -1.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    result = ...
    # ---------------------------------------------------------------

    return result
'''
    target = f'''
def answer():
    r"""
    Return the emitted harmonic OAM and spin label.
    """
    q = {q}
    ell_left = {ell_left}
    ell_right = {ell_right}
    n_left = (q + 1) // 2
    n_right = (q - 1) // 2
    ell_out = n_left * ell_left + n_right * ell_right
    spin_label = 1 if n_left >= n_right else -1
    return (ell_out, spin_label)
'''
    setup = (
        "In a toy bicircular high-harmonic selection rule, an odd harmonic q "
        "uses n_left=(q+1)//2 quanta from the left driver and n_right=(q-1)//2 "
        "quanta from the right driver. The emitted OAM is the integer sum "
        f"n_left*ell_left+n_right*ell_right with ell_left={ell_left} and "
        f"ell_right={ell_right}. The spin label is +1 when the left count is "
        "not smaller than the right count, otherwise -1."
    )
    main = f"For q={q}, return `(ell_out, spin_label)`."
    return _example(
        problem_id=f"{split}_discrete_oam_selection_algorithmic_{idx:05d}_{ell_left}_{ell_right}_{q}",
        split=split,
        family="discrete",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={
            "kind": "code",
            "timeout_s": 2.0,
            "checks": [{"mode": "exact", "expected": {"$tuple": [ell_harmonic, spin]}}],
        },
        trace="Compute n_left and n_right from q, then apply OAM conservation and the spin-label rule.",
        metadata={"domain": "amo_selection_rule", "official_overlap": "none", "variant": "algorithmic"},
    )


def _lattice_residue_set(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    n = rng.randint(6, 14 if difficulty != "hard" else 24)
    a = rng.randint(1, 5)
    b = rng.randint(0, n - 1)
    residue = rng.randint(0, n - 1)
    allowed = [k for k in range(n) if (a * k + b) % n == residue]
    template = '''
def answer():
    r"""
    Return all allowed momentum labels k in ascending order.

    Inputs
    ----------
    None

    Outputs
    ----------
    labels: list[int]
        Allowed labels in ascending order.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    labels = ...
    # ---------------------------------------------------------------

    return labels
'''
    target = f'''
def answer():
    r"""
    Return all allowed momentum labels k in ascending order.
    """
    labels = [k for k in range({n}) if ({a} * k + {b}) % {n} == {residue}]
    return labels
'''
    setup = f"On a ring of {n} sites, a mode label k is allowed if $({a}k+{b}) \\bmod {n}={residue}$."
    main = "Return all allowed labels in ascending order."
    return _example(
        problem_id=f"{split}_discrete_lattice_residue_{idx:05d}_{n}_{a}_{b}_{residue}",
        split=split,
        family="discrete",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={
            "kind": "code",
            "timeout_s": 2.0,
            "checks": [{"mode": "exact_sequence", "expected": allowed}],
        },
        trace="Enumerate k=0,...,n-1 and keep labels satisfying the residue constraint.",
        metadata={"domain": "lattice_selection", "official_overlap": "none"},
    )


def _stabilizer_dimension(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    n = rng.randint(5, 14)
    r = rng.randint(1, n - 1)
    expected = 2 ** (n - r)
    template = '''
def answer():
    r"""
    Return the codespace dimension.

    Inputs
    ----------
    None

    Outputs
    ----------
    dim: int
        Dimension of the simultaneous +1 eigenspace.
    """

    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    dim = ...
    # ---------------------------------------------------------------

    return dim
'''
    target = f'''
def answer():
    r"""
    Return the codespace dimension.
    """
    dim = 2 ** ({n} - {r})
    return dim
'''
    setup = f"A stabilizer-like code has n={n} qubits and r={r} independent commuting binary checks."
    main = "Return the dimension of the simultaneous +1 eigenspace."
    return _example(
        problem_id=f"{split}_discrete_stabilizer_dimension_{idx:05d}_{n}_{r}",
        split=split,
        family="discrete",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={
            "kind": "code",
            "timeout_s": 2.0,
            "checks": [{"mode": "exact", "expected": expected}],
        },
        trace="Each independent binary check halves the Hilbert-space dimension.",
        metadata={"domain": "toy_qec", "official_overlap": "none"},
    )


def _factorial(n: int) -> int:
    out = 1
    for value in range(2, n + 1):
        out *= value
    return out


def _series_coeffs_exp_over_pole(a: int, b: int, order: int) -> list[float]:
    coeffs: list[float] = []
    for n in range(order + 1):
        total = Fraction(0, 1)
        for k in range(n + 1):
            total += Fraction(a**k, _factorial(k)) * Fraction(b ** (n - k), 1)
        coeffs.append(float(total))
    return coeffs


def _double_factorial(n: int) -> int:
    out = 1
    for value in range(n, 0, -2):
        out *= value
    return out


def _mat_pow_2x2(matrix: list[list[int]], power: int) -> list[list[int]]:
    result = [[1, 0], [0, 1]]
    for _ in range(power):
        result = [
            [
                result[0][0] * matrix[0][0] + result[0][1] * matrix[1][0],
                result[0][0] * matrix[0][1] + result[0][1] * matrix[1][1],
            ],
            [
                result[1][0] * matrix[0][0] + result[1][1] * matrix[1][0],
                result[1][0] * matrix[0][1] + result[1][1] * matrix[1][1],
            ],
        ]
    return result


def _mat_pow(matrix: list[list[int]], power: int) -> list[list[int]]:
    n = len(matrix)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    for _ in range(power):
        result = [
            [sum(result[i][k] * matrix[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)
        ]
    return result
