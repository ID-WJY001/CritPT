from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from rl_posttrain.critpt_synth.final_answer_verifier import verify_final_answer


@dataclass(frozen=True)
class NaturalPhysicsExample:
    example_id: str
    split: str
    domain: str
    skill: str
    difficulty: str
    question: str
    reference_solution: str
    final_answer: str
    answer_type: str
    verifier: dict[str, Any]
    anti_hack_wrong_answers: list[dict[str, str]]
    judge_rubric: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "domain": self.domain,
            "skill": self.skill,
            "difficulty": self.difficulty,
            "question": self.question,
            "reference_solution": self.reference_solution,
            "final_answer": self.final_answer,
            "answer_type": self.answer_type,
            "verifier": self.verifier,
            "anti_hack_wrong_answers": self.anti_hack_wrong_answers,
            "judge_rubric": self.judge_rubric,
            "metadata": self.metadata,
        }

    def final_completion(self) -> str:
        return f"最终答案：{self.final_answer}"


GeneratorFn = Callable[[random.Random, int, str, str], NaturalPhysicsExample]


def generate_v12_natural_physics_examples(
    size: int,
    seed: int,
    split: str,
) -> list[NaturalPhysicsExample]:
    rng = random.Random(seed)
    specs: list[tuple[GeneratorFn, float, tuple[list[str], list[float]]]] = [
        (_two_level_gap, 0.12, (["easy", "medium"], [0.65, 0.35])),
        (_pauli_commutator, 0.12, (["easy", "medium"], [0.60, 0.40])),
        (_spin_z_expectation, 0.08, (["easy", "medium"], [0.70, 0.30])),
        (_two_site_tight_binding_gap, 0.08, (["easy", "medium"], [0.65, 0.35])),
        (_transfer_matrix_trace_audit, 0.12, (["easy", "medium"], [0.65, 0.35])),
        (_response_series_coefficients, 0.14, (["easy", "medium"], [0.65, 0.35])),
        (_stabilizer_dimension, 0.10, (["easy", "medium"], [0.70, 0.30])),
        (_oam_selection_rule, 0.08, (["easy", "medium"], [0.65, 0.35])),
        (_two_level_partition_probability, 0.08, (["easy", "medium"], [0.70, 0.30])),
        (_decay_population, 0.08, (["easy", "medium"], [0.70, 0.30])),
    ]
    weights = [spec[1] for spec in specs]
    out: list[NaturalPhysicsExample] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < size:
        attempts += 1
        if attempts > size * 50:
            raise RuntimeError(f"too many duplicate V12 examples while building {split}")
        idx = len(out)
        spec_idx = rng.choices(range(len(specs)), weights=weights, k=1)[0]
        generator, _weight, difficulty_spec = specs[spec_idx]
        difficulties, difficulty_weights = difficulty_spec
        difficulty = rng.choices(difficulties, weights=difficulty_weights, k=1)[0]
        example = generator(rng, idx, split, difficulty)
        if example.example_id in seen:
            continue
        seen.add(example.example_id)
        out.append(example)
    return out


def verify_v12_example(example: NaturalPhysicsExample) -> tuple[bool, list[str]]:
    failures: list[str] = []
    result = verify_final_answer(example.final_completion(), example.verifier)
    if not result.ok:
        failures.append(f"gold_failed: {result.reason}")
    for wrong in example.anti_hack_wrong_answers:
        wrong_result = verify_final_answer(f"最终答案：{wrong['answer']}", example.verifier)
        if wrong_result.ok:
            failures.append(f"anti_hack_passed: {wrong['tag']} -> {wrong['answer']}")
    return not failures, failures


def _base_judge_rubric(example: NaturalPhysicsExample) -> dict[str, Any]:
    return {
        "overall_score": {
            "range": [0, 1],
            "cap_if_final_answer_wrong": 0.6,
            "cap_if_hallucinated_conditions": 0.4,
        },
        "criteria": [
            "Final answer is equivalent to the structured verifier target.",
            "Reference reasoning covers the key physical or mathematical relation.",
            "No extra assumptions, invented parameters, or irrelevant long derivation.",
            "Units, signs, factors of two, ordering, and list length are consistent.",
        ],
        "anti_hack_tags": [item["tag"] for item in example.anti_hack_wrong_answers],
    }


def _with_rubric(example: NaturalPhysicsExample) -> NaturalPhysicsExample:
    return NaturalPhysicsExample(
        **{**example.to_dict(), "judge_rubric": _base_judge_rubric(example)}
    )


def _common_metadata(generator: str, params: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps({"generator": generator, "params": params}, sort_keys=True)
    return {
        "generator": generator,
        "params": params,
        "param_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        "official_overlap": "none",
        "front_prompt_contains_python": False,
    }


def _symbolic_verifier(expected: str, variables: list[str]) -> dict[str, Any]:
    return {
        "checks": [
            {
                "mode": "symbolic",
                "expected": expected,
                "variables": variables,
                "tolerance": 1e-8,
            }
        ]
    }


def _numeric_sequence_verifier(expected: list[float | int]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = [{"mode": "sequence_length", "expected": len(expected)}]
    checks.extend(
        {
            "mode": "numeric_sequence_item",
            "index": idx,
            "expected": value,
            "tolerance": 1e-8,
        }
        for idx, value in enumerate(expected)
    )
    return {"checks": checks}


def _exact_sequence_verifier(expected: list[Any]) -> dict[str, Any]:
    return {"checks": [{"mode": "exact_sequence", "expected": expected}]}


def _exact_verifier(expected: int | str | float) -> dict[str, Any]:
    return {"checks": [{"mode": "exact", "expected": expected}]}


def _wrong(tag: str, answer: str, note: str) -> dict[str, str]:
    return {"tag": tag, "answer": answer, "note": note}


def _format_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return repr(value)


def _coef_var(coef: int, var: str) -> str:
    return var if coef == 1 else f"{coef}*{var}"


def _series_coeffs(a: int, b: int, s: int, order: int) -> list[float]:
    coeffs: list[float] = []
    for n in range(order + 1):
        base = sum((a**k / math.factorial(k)) * (b ** (n - k)) for k in range(n + 1))
        shifted = 0.0
        if n > 0:
            shifted = s * sum(
                (a**k / math.factorial(k)) * (b ** (n - 1 - k)) for k in range(n)
            )
        value = base + shifted
        coeffs.append(float(int(value)) if float(value).is_integer() else float(value))
    return coeffs


def _mat_mul_2x2(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]


def _mat_pow_2x2(matrix: list[list[int]], power: int) -> list[list[int]]:
    out = [[1, 0], [0, 1]]
    for _ in range(power):
        out = _mat_mul_2x2(out, matrix)
    return out


def _two_level_gap(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> NaturalPhysicsExample:
    coupling = rng.randint(1, 3 if difficulty == "easy" else 6)
    detuning = rng.randint(1, 2 if difficulty == "easy" else 4)
    expected = f"sqrt({detuning * detuning}*delta**2 + {4 * coupling * coupling}*g**2)"
    question = (
        "A two-level effective Hamiltonian is "
        f"H = [[{_coef_var(detuning, 'delta')}/2, {_coef_var(coupling, 'g')}], "
        f"[{_coef_var(coupling, 'g')}, -{_coef_var(detuning, 'delta')}/2]]. "
        "Return the symbolic energy gap between the upper and lower eigenvalues. "
        "Use variables delta and g."
    )
    reference = (
        f"The eigenvalues are ±sqrt(({_coef_var(detuning, 'delta')}/2)^2 + "
        f"({_coef_var(coupling, 'g')})^2). "
        f"The gap is twice that value, so the answer is {expected}."
    )
    example = NaturalPhysicsExample(
        example_id=f"{split}_v12_two_level_gap_{idx:05d}_{detuning}_{coupling}",
        split=split,
        domain="quantum_two_level",
        skill="eigenvalue_gap",
        difficulty=difficulty,
        question=question,
        reference_solution=reference,
        final_answer=expected,
        answer_type="symbolic",
        verifier=_symbolic_verifier(expected, ["delta", "g"]),
        anti_hack_wrong_answers=[
            _wrong("missing_factor_two", f"sqrt(delta**2 + {coupling * coupling}*g**2)", "misses the gap factor"),
            _wrong("half_coupling_term", f"sqrt(delta**2 + {2 * coupling * coupling}*g**2)", "uses only half the coupling contribution"),
            _wrong("linear_addition", f"delta + {2 * coupling}*g", "adds scales linearly"),
        ],
        judge_rubric={},
        metadata=_common_metadata(
            "two_level_gap", {"detuning_multiplier": detuning, "coupling": coupling}
        ),
    )
    return _with_rubric(example)


def _pauli_commutator(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> NaturalPhysicsExample:
    pairs = [
        ("X", "Y", "Z", 1),
        ("Y", "Z", "X", 1),
        ("Z", "X", "Y", 1),
        ("Y", "X", "Z", -1),
        ("Z", "Y", "X", -1),
        ("X", "Z", "Y", -1),
    ]
    left, right, out_label, sign = rng.choice(pairs)
    a = rng.randint(1, 3 if difficulty == "easy" else 5)
    b = rng.randint(1, 3 if difficulty == "easy" else 5)
    coeff = 2 * a * b * sign
    expected = [coeff, out_label]
    question = (
        f"Let sigma_{left}, sigma_{right}, sigma_{out_label} be Pauli matrices. "
        f"For A={a} sigma_{left} and B={b} sigma_{right}, write "
        "[A,B] = i * c * sigma_label. Return exactly [c, sigma_label]."
    )
    reference = (
        f"Using [sigma_{left}, sigma_{right}] = {2 * sign} i sigma_{out_label}, "
        f"the coefficient is c = 2*{a}*{b}*({sign}) = {coeff}."
    )
    example = NaturalPhysicsExample(
        example_id=f"{split}_v12_pauli_commutator_{idx:05d}_{left}_{right}_{a}_{b}",
        split=split,
        domain="quantum_information",
        skill="pauli_commutator",
        difficulty=difficulty,
        question=question,
        reference_solution=reference,
        final_answer=json.dumps(expected),
        answer_type="exact_sequence",
        verifier=_exact_sequence_verifier(expected),
        anti_hack_wrong_answers=[
            _wrong("missing_factor_two", json.dumps([a * b * sign, out_label]), "drops the factor of two"),
            _wrong("wrong_sign", json.dumps([-coeff, out_label]), "uses the reversed commutator sign"),
            _wrong("wrong_pauli_label", json.dumps([coeff, left]), "returns an input Pauli label"),
        ],
        judge_rubric={},
        metadata=_common_metadata(
            "pauli_commutator", {"left": left, "right": right, "a": a, "b": b}
        ),
    )
    return _with_rubric(example)


def _spin_z_expectation(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> NaturalPhysicsExample:
    del difficulty
    observable = rng.choice(["X", "Z"])
    angle_multiplier = rng.randint(1, 4)
    angle = "theta" if angle_multiplier == 1 else f"{angle_multiplier}*theta"
    if observable == "Z":
        expected = f"cos({angle})"
        observable_text = "Z"
        reference_tail = (
            f"The expectation is cos({angle}/2)^2 - sin({angle}/2)^2 = {expected}."
        )
        wrong_answers = [
            _wrong("half_angle", "cos(theta/2)", "returns the amplitude factor instead of expectation"),
            _wrong("sine_confusion", "sin(theta)", "confuses the Bloch z component"),
            _wrong("probability_only", "cos(theta/2)**2", "forgets the -1 eigenvalue branch"),
        ]
    else:
        expected = f"sin({angle})"
        observable_text = "X"
        reference_tail = (
            f"For a real Bloch-sphere state, <X>=2 cos({angle}/2) sin({angle}/2)={expected}."
        )
        wrong_answers = [
            _wrong("z_component", "cos(theta)", "returns the Z expectation instead"),
            _wrong("half_angle", "sin(theta/2)", "returns an amplitude factor"),
            _wrong("probability_gap", "cos(theta/2)**2 - sin(theta/2)**2", "uses Z instead of X"),
        ]
    question = (
        f"A qubit is in the state |psi> = cos({angle}/2)|0> + sin({angle}/2)|1>. "
        f"Return the symbolic expectation value <psi|{observable_text}|psi> as a function of theta."
    )
    reference = (
        "Use the Pauli matrix expectation in the {|0>,|1>} basis. "
        f"{reference_tail}"
    )
    example = NaturalPhysicsExample(
        example_id=f"{split}_v12_spin_expectation_{idx:05d}_{observable}_{angle_multiplier}",
        split=split,
        domain="quantum_states",
        skill="expectation_value",
        difficulty="easy",
        question=question,
        reference_solution=reference,
        final_answer=expected,
        answer_type="symbolic",
        verifier=_symbolic_verifier(expected, ["theta"]),
        anti_hack_wrong_answers=wrong_answers,
        judge_rubric={},
        metadata=_common_metadata(
            "spin_expectation", {"observable": observable, "angle_multiplier": angle_multiplier}
        ),
    )
    return _with_rubric(example)


def _two_site_tight_binding_gap(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> NaturalPhysicsExample:
    hopping_multiplier = rng.randint(1, 4 if difficulty == "easy" else 8)
    expected = f"{2 * hopping_multiplier}*t"
    question = (
        "A two-site tight-binding Hamiltonian has matrix "
        f"H = [[epsilon, {hopping_multiplier}*t], [{hopping_multiplier}*t, epsilon]] "
        "with t > 0. Return the symbolic gap between the two eigenvalues."
    )
    reference = (
        f"The symmetric and antisymmetric eigenvalues are epsilon+{hopping_multiplier}*t "
        f"and epsilon-{hopping_multiplier}*t. For t > 0 the gap is {expected}."
    )
    example = NaturalPhysicsExample(
        example_id=f"{split}_v12_two_site_gap_{idx:05d}_{hopping_multiplier}",
        split=split,
        domain="condensed_matter",
        skill="tight_binding_eigenvalues",
        difficulty="easy",
        question=question,
        reference_solution=reference,
        final_answer=expected,
        answer_type="symbolic",
        verifier=_symbolic_verifier(expected, ["epsilon", "t"]),
        anti_hack_wrong_answers=[
            _wrong("single_hopping", f"{hopping_multiplier}*t", "forgets the splitting is twice the hopping"),
            _wrong("onsite_included", f"epsilon + {hopping_multiplier}*t", "returns one eigenvalue instead of the gap"),
            _wrong("wrong_sign", f"-{2 * hopping_multiplier}*t", "uses the wrong ordering despite t > 0"),
        ],
        judge_rubric={},
        metadata=_common_metadata(
            "two_site_tight_binding_gap", {"hopping_multiplier": hopping_multiplier}
        ),
    )
    return _with_rubric(example)


def _transfer_matrix_trace_audit(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> NaturalPhysicsExample:
    max_entry = 2 if difficulty == "easy" else 3
    matrix = [
        [rng.randint(1, max_entry), rng.randint(1, max_entry)],
        [rng.randint(1, max_entry), rng.randint(1, max_entry)],
    ]
    power = 2 if difficulty == "easy" else 3
    squared = _mat_pow_2x2(matrix, 2)
    powered = _mat_pow_2x2(matrix, power)
    expected = [
        float(powered[0][0] + powered[1][1]),
        float(squared[0][1]),
        float(powered[1][0]),
    ]
    question = (
        f"Let T = [[{matrix[0][0]}, {matrix[0][1]}], [{matrix[1][0]}, {matrix[1][1]}]] "
        f"be a transfer matrix. Compute T^2 and T^{power} using full row-column multiplication. "
        f"Return exactly [trace(T^{power}), (T^2)_01, (T^{power})_10]."
    )
    reference = (
        f"Full multiplication gives T^2={squared} and T^{power}={powered}. "
        f"Therefore the requested audit list is {expected}."
    )
    wrong_elementwise = [
        float(matrix[0][0] ** power + matrix[1][1] ** power),
        float(matrix[0][1] ** 2),
        float(matrix[1][0] ** power),
    ]
    swapped = [expected[0], expected[2], expected[1]]
    if swapped == expected:
        swapped = [expected[0], expected[1] + 1.0, expected[2]]
    example = NaturalPhysicsExample(
        example_id=(
            f"{split}_v12_transfer_trace_{idx:05d}_{matrix[0][0]}_{matrix[0][1]}_"
            f"{matrix[1][0]}_{matrix[1][1]}_{power}"
        ),
        split=split,
        domain="condensed_matter",
        skill="matrix_power_trace",
        difficulty=difficulty,
        question=question,
        reference_solution=reference,
        final_answer=json.dumps(expected),
        answer_type="numeric_sequence",
        verifier=_numeric_sequence_verifier(expected),
        anti_hack_wrong_answers=[
            _wrong("elementwise_power", json.dumps(wrong_elementwise), "uses elementwise powers"),
            _wrong("trace_only", _format_number(expected[0]), "returns only the trace"),
            _wrong("swapped_or_perturbed_entries", json.dumps(swapped), "swaps or perturbs audit entries"),
        ],
        judge_rubric={},
        metadata=_common_metadata("transfer_matrix_trace_audit", {"matrix": matrix, "power": power}),
    )
    return _with_rubric(example)


def _response_series_coefficients(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> NaturalPhysicsExample:
    max_param = 2 if difficulty == "easy" else 3
    a = rng.randint(1, max_param)
    b = rng.randint(1, max_param)
    s = rng.randint(1, max_param)
    order = 2 if difficulty == "easy" else 3
    coeffs = _series_coeffs(a, b, s, order)
    question = (
        f"For the response function f(x)=(1+{_coef_var(s, 'x')})*"
        f"exp({_coef_var(a, 'x')})/(1-{_coef_var(b, 'x')}), "
        f"return the Taylor coefficient list [c0, ..., c{order}]. "
        "Use the coefficient of x^n, not the derivative value f^(n)(0)."
    )
    reference = (
        "For exp(a x)/(1-b x), base_n=sum_{k=0}^n a^k/k! * b^(n-k). "
        f"The numerator shift gives c_n=base_n+{s}*base_(n-1). "
        f"Through order {order}, the coefficients are {coeffs}."
    )
    wrong_derivatives = [float(value * math.factorial(n)) for n, value in enumerate(coeffs)]
    wrong_no_shift = _series_coeffs(a, b, 0, order)
    example = NaturalPhysicsExample(
        example_id=f"{split}_v12_response_series_{idx:05d}_{a}_{b}_{s}_{order}",
        split=split,
        domain="linear_response",
        skill="taylor_coefficients",
        difficulty=difficulty,
        question=question,
        reference_solution=reference,
        final_answer=json.dumps(coeffs),
        answer_type="numeric_sequence",
        verifier=_numeric_sequence_verifier(coeffs),
        anti_hack_wrong_answers=[
            _wrong("derivative_values", json.dumps(wrong_derivatives), "returns derivatives instead of coefficients"),
            _wrong("missing_numerator_shift", json.dumps(wrong_no_shift), "drops the (1+s x) numerator"),
            _wrong("last_only", _format_number(coeffs[-1]), "returns only the final coefficient"),
        ],
        judge_rubric={},
        metadata=_common_metadata(
            "response_series_coefficients", {"a": a, "b": b, "s": s, "order": order}
        ),
    )
    return _with_rubric(example)


def _stabilizer_dimension(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> NaturalPhysicsExample:
    n = rng.randint(4, 7 if difficulty == "easy" else 10)
    r = rng.randint(1, n - 1)
    while r == n - r:
        r = rng.randint(1, n - 1)
    expected = 2 ** (n - r)
    question = (
        f"A stabilizer code has n={n} physical qubits and r={r} independent commuting stabilizer generators. "
        "Return the codespace dimension."
    )
    reference = (
        f"Each independent stabilizer halves the Hilbert-space dimension, so dim=2^(n-r)=2^({n}-{r})={expected}."
    )
    example = NaturalPhysicsExample(
        example_id=f"{split}_v12_stabilizer_dimension_{idx:05d}_{n}_{r}",
        split=split,
        domain="quantum_error_correction",
        skill="stabilizer_dimension",
        difficulty=difficulty,
        question=question,
        reference_solution=reference,
        final_answer=str(expected),
        answer_type="integer",
        verifier=_exact_verifier(expected),
        anti_hack_wrong_answers=[
            _wrong("uses_n_minus_r", str(n - r), "returns the number of logical qubits"),
            _wrong("missing_constraints", str(2**n), "ignores stabilizer constraints"),
            _wrong("wrong_exponent", str(2**r), "uses r instead of n-r"),
        ],
        judge_rubric={},
        metadata=_common_metadata("stabilizer_dimension", {"n": n, "r": r}),
    )
    return _with_rubric(example)


def _oam_selection_rule(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> NaturalPhysicsExample:
    ell_a = rng.randint(-3, 3 if difficulty == "easy" else 5)
    ell_b = rng.randint(-3, 3 if difficulty == "easy" else 5)
    while ell_a == ell_b:
        ell_b = rng.randint(-3, 3 if difficulty == "easy" else 5)
    harmonic = rng.choice([7, 9, 11, 13, 15])
    n_a = (harmonic + 1) // 2
    n_b = (harmonic - 1) // 2
    ell_out = n_a * ell_a + n_b * ell_b
    spin_label = 1
    expected = [ell_out, spin_label]
    question = (
        "In a toy bicircular high-harmonic selection rule, odd harmonic q absorbs "
        "(q+1)/2 photons from driver A and (q-1)/2 photons from driver B. "
        f"Driver OAM values are ell_A={ell_a} and ell_B={ell_b}. For q={harmonic}, "
        "return exactly [ell_harmonic, spin_label], with spin_label=+1 for A-dominant helicity."
    )
    reference = (
        f"n_A={n_a}, n_B={n_b}. The OAM is {n_a}*({ell_a})+{n_b}*({ell_b})={ell_out}; "
        "A contributes one more photon, so spin_label=+1."
    )
    example = NaturalPhysicsExample(
        example_id=f"{split}_v12_oam_selection_{idx:05d}_{ell_a}_{ell_b}_{harmonic}",
        split=split,
        domain="amo_optics",
        skill="selection_rule",
        difficulty=difficulty,
        question=question,
        reference_solution=reference,
        final_answer=json.dumps(expected),
        answer_type="exact_sequence",
        verifier=_exact_sequence_verifier(expected),
        anti_hack_wrong_answers=[
            _wrong("swapped_photon_counts", json.dumps([n_b * ell_a + n_a * ell_b, 1]), "swaps A and B counts"),
            _wrong("wrong_spin", json.dumps([ell_out, -1]), "uses wrong helicity label"),
            _wrong("only_oam", str(ell_out), "omits the spin label"),
        ],
        judge_rubric={},
        metadata=_common_metadata(
            "oam_selection_rule", {"ell_a": ell_a, "ell_b": ell_b, "harmonic": harmonic}
        ),
    )
    return _with_rubric(example)


def _two_level_partition_probability(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> NaturalPhysicsExample:
    gap_multiplier = rng.randint(1, 3 if difficulty == "easy" else 6)
    degeneracy = rng.randint(1, 3 if difficulty == "easy" else 5)
    expected = f"{degeneracy}/(exp({gap_multiplier}*beta*Delta)+{degeneracy})"
    question = (
        f"A two-level system has energies E0=0 and E1={gap_multiplier}*Delta. "
        f"The excited level has degeneracy {degeneracy}, while the ground level is non-degenerate. "
        "At inverse temperature beta, return the symbolic total probability of occupying the excited level."
    )
    reference = (
        f"The partition function is Z=1+{degeneracy}*exp(-{gap_multiplier}*beta*Delta). "
        "The total excited-state weight divided by Z simplifies to "
        f"p1={expected}."
    )
    wrong_gap = "1/(exp(beta*Delta)+1)"
    if gap_multiplier == 1 and degeneracy == 1:
        wrong_gap = "1/(exp(2*beta*Delta)+1)"
    example = NaturalPhysicsExample(
        example_id=f"{split}_v12_two_level_partition_{idx:05d}_{gap_multiplier}_{degeneracy}",
        split=split,
        domain="statistical_physics",
        skill="partition_probability",
        difficulty="easy",
        question=question,
        reference_solution=reference,
        final_answer=expected,
        answer_type="symbolic",
        verifier=_symbolic_verifier(expected, ["beta", "Delta"]),
        anti_hack_wrong_answers=[
            _wrong("ground_probability", f"exp({gap_multiplier}*beta*Delta)/(exp({gap_multiplier}*beta*Delta)+{degeneracy})", "returns ground-state probability"),
            _wrong("unnormalized_weight", f"exp(-{gap_multiplier}*beta*Delta)", "forgets normalization"),
            _wrong("wrong_gap", wrong_gap, "drops or changes the energy/degeneracy factor"),
        ],
        judge_rubric={},
        metadata=_common_metadata(
            "two_level_partition_probability",
            {"gap_multiplier": gap_multiplier, "excited_degeneracy": degeneracy},
        ),
    )
    return _with_rubric(example)


def _decay_population(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> NaturalPhysicsExample:
    initial = rng.randint(1, 5 if difficulty == "easy" else 9)
    rate_multiplier = rng.randint(1, 3 if difficulty == "easy" else 6)
    expected = f"{initial}*exp(-{rate_multiplier}*gamma*t)"
    question = (
        "A single excited population p(t) obeys the rate equation "
        f"dp/dt = -{rate_multiplier}*gamma*p with p(0)={initial}. "
        "Return p(t) symbolically."
    )
    reference = (
        f"Solving dp/p=-{rate_multiplier}*gamma dt gives p(t)=C exp(-{rate_multiplier}*gamma*t); "
        f"p(0)={initial} fixes C={initial}."
    )
    missing_initial = "exp(-gamma*t)"
    if initial == 1 and rate_multiplier == 1:
        missing_initial = "2*exp(-gamma*t)"
    example = NaturalPhysicsExample(
        example_id=f"{split}_v12_decay_population_{idx:05d}_{initial}_{rate_multiplier}",
        split=split,
        domain="open_quantum_systems",
        skill="rate_equation",
        difficulty="easy",
        question=question,
        reference_solution=reference,
        final_answer=expected,
        answer_type="symbolic",
        verifier=_symbolic_verifier(expected, ["gamma", "t"]),
        anti_hack_wrong_answers=[
            _wrong("growth_sign", f"{initial}*exp({rate_multiplier}*gamma*t)", "uses growth instead of decay"),
            _wrong("linear_decay", f"{initial}*(1-{rate_multiplier}*gamma*t)", "returns only first-order approximation"),
            _wrong("missing_initial_condition", missing_initial, "drops or changes the initial population"),
        ],
        judge_rubric={},
        metadata=_common_metadata(
            "decay_population", {"initial": initial, "rate_multiplier": rate_multiplier}
        ),
    )
    return _with_rubric(example)
