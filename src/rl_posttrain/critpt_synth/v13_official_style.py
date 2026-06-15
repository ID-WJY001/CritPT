from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import sympy as sp

from rl_posttrain.critpt_synth.prompting import render_prompt
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.critpt_synth.verifier import verify_code_completion


GeneratorFn = Callable[[random.Random, int, str, str], SyntheticCritPTExample]


def generate_v13_official_style_examples(
    size: int,
    seed: int,
    split: str,
) -> list[SyntheticCritPTExample]:
    rng = random.Random(seed)
    specs: list[tuple[GeneratorFn, float, tuple[list[str], list[float]]]] = [
        (_toy_weyl_coefficients, 0.14, (["medium", "hard"], [0.65, 0.35])),
        (_lamet_piecewise_kernel, 0.12, (["medium", "hard"], [0.60, 0.40])),
        (_hhg_oam_selection, 0.10, (["easy", "medium"], [0.45, 0.55])),
        (_distributed_qfi_symbolic, 0.13, (["medium", "hard"], [0.65, 0.35])),
        (_convection_galerkin_minimum, 0.15, (["medium", "hard"], [0.55, 0.45])),
        (_u2_operator_enumeration, 0.12, (["easy", "medium"], [0.50, 0.50])),
        (_linear_recurrence_generating_function, 0.13, (["easy", "medium"], [0.50, 0.50])),
        (_amplitude_damping_outputs, 0.11, (["easy", "medium"], [0.55, 0.45])),
    ]
    weights = [item[1] for item in specs]
    examples: list[SyntheticCritPTExample] = []
    seen: set[str] = set()
    attempts = 0
    while len(examples) < size:
        attempts += 1
        if attempts > size * 100:
            raise RuntimeError(f"too many duplicate V13 examples while building {split}")
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


def verify_v13_example(example: SyntheticCritPTExample) -> tuple[bool, str]:
    result = verify_code_completion(example.assistant_code_block(), example.verifier)
    return result.ok, result.reason


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
            "generator_profile": "v13_official_style",
            "param_hash": param_hash,
            "official_overlap": "none",
            "uses_official_prompt": False,
            "prompt_style": "problem_setup_main_problem_parsing_template",
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


def _exact_sequence_checks(expected: list[Any]) -> list[dict[str, Any]]:
    return [{"mode": "exact_sequence", "expected": expected}]


def _toy_weyl_coefficients(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(1, 4 if difficulty == "medium" else 7)
    b = rng.randint(1, 4 if difficulty == "medium" else 6)
    c = rng.randint(1, 3 if difficulty == "medium" else 5)
    d = rng.randint(1, 5 if difficulty == "medium" else 8)
    coeffs = [
        float(a * a - b),
        float(2 * a * b + c),
        float(b * b - a * c),
        float(a + b + c),
        float(d * (a - c)),
        float(a * b * c - d),
    ]
    setup = f"""
Consider a toy holographic anomaly functional in eight boundary dimensions. The
full tensor definitions are not needed for this problem, but the notation follows
the standard trace basis used for curvature invariants. In this synthetic variant
the scalar density X is expanded in the ordered basis

1. tr(P^4)
2. tr(P^3) tr(P)
3. tr(B P^2)
4. tr(B^2)
5. tr(O P)
6. tr(Omega P)

The coefficients are generated from four scheme constants a={a}, b={b}, c={c},
d={d}. The renormalization prescription gives

C1 = a^2 - b,
C2 = 2ab + c,
C3 = b^2 - ac,
C4 = a + b + c,
C5 = d(a-c),
C6 = abc - d.
"""
    main = "Determine the six coefficients of X in the order listed above."
    template = """
def answer():
    r\"\"\"
    Return coefficients of the ordered anomaly basis.

    Output
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
    Return coefficients of the ordered anomaly basis.
    \"\"\"
    coeffs = {coeffs!r}
    return coeffs
"""
    return _example(
        problem_id=f"{split}_v13_weyl_coefficients_{idx:05d}_{a}_{b}_{c}_{d}",
        split=split,
        family="official_template_coefficients",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_checks(coeffs)},
        solution_trace=f"Substituting a={a}, b={b}, c={c}, d={d} gives coeffs={coeffs!r}.",
        metadata={"domain": "high_energy_holography", "answer_type": "numeric_sequence"},
    )


def _lamet_piecewise_kernel(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    alpha = rng.randint(1, 3 if difficulty == "medium" else 5)
    beta = rng.randint(1, 4 if difficulty == "medium" else 6)
    gamma = rng.randint(1, 3 if difficulty == "medium" else 5)
    expr_neg = f"{alpha}/(1 - x)"
    expr_mid = f"{alpha}*log(mu/(Pz*x)) + {beta}*(1 - x)"
    expr_pos = f"-{gamma}/x"
    expected = [expr_neg, expr_mid, expr_pos]
    setup = f"""
In a toy LaMET matching calculation, a one-loop sail contribution is represented
by a piecewise kernel K(x). The real calculation would require dimensional
regularization and plus distributions. Here the finite MS-bar remainder is
already reduced to three kinematic regions:

K_-(x) = alpha/(1-x) for x < 0,
K_0(x) = alpha log(mu/(Pz x)) + beta(1-x) for 0 < x < 1,
K_+(x) = -gamma/x for x > 1.

Use alpha={alpha}, beta={beta}, gamma={gamma}. Keep the symbolic variables
x, mu and Pz exactly as written in the template.
"""
    main = "Return the three region expressions as strings in the order x<0, 0<x<1, x>1."
    template = """
def answer():
    r\"\"\"
    Return the finite piecewise kernel expressions.

    Output
    ----------
    expr_lt0: str
    expr_mid: str
    expr_gt1: str
    \"\"\"
    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    expr_lt0 = ...
    expr_mid = ...
    expr_gt1 = ...
    # ---------------------------------------------------------------

    return expr_lt0, expr_mid, expr_gt1
"""
    target = f"""
def answer():
    r\"\"\"
    Return the finite piecewise kernel expressions.
    \"\"\"
    expr_lt0 = {expr_neg!r}
    expr_mid = {expr_mid!r}
    expr_gt1 = {expr_pos!r}
    return expr_lt0, expr_mid, expr_gt1
"""
    return _example(
        problem_id=f"{split}_v13_lamet_piecewise_{idx:05d}_{alpha}_{beta}_{gamma}",
        split=split,
        family="official_template_piecewise",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _exact_sequence_checks(expected)},
        solution_trace=f"The three strings are {expected!r}.",
        metadata={"domain": "qcd_lamet", "answer_type": "string_tuple"},
    )


def _hhg_oam_selection(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    pulses = [
        (-1, -1),
        (2, 1),
        (1 if difficulty == "easy" else 3, -1),
    ]
    counts = [rng.randint(1, 3), rng.randint(1, 3), rng.randint(0, 2 if difficulty == "easy" else 3)]
    harmonic = sum(counts)
    oam = sum(count * pulse[0] for count, pulse in zip(counts, pulses))
    helicity_sum = sum(count * pulse[1] for count, pulse in zip(counts, pulses))
    helicity = 1 if helicity_sum >= 0 else -1
    expected = [harmonic, oam, helicity]
    setup = f"""
High-harmonic generation with structured light obeys conservation of angular
momentum in this simplified selection-rule model. Three driving pulses are
available. Pulse A has OAM ell={pulses[0][0]} and helicity sigma={pulses[0][1]}.
Pulse B has OAM ell={pulses[1][0]} and helicity sigma={pulses[1][1]}. Pulse C
has OAM ell={pulses[2][0]} and helicity sigma={pulses[2][1]}. A harmonic photon
is produced by absorbing nA={counts[0]}, nB={counts[1]}, nC={counts[2]} driving
photons. In this toy rule the harmonic order is nA+nB+nC, the emitted OAM is
the sum of absorbed OAM values, and the emitted helicity is the sign of the
absorbed helicity sum.
"""
    main = "Return exactly [harmonic_order, emitted_oam, emitted_helicity]."
    template = """
def answer():
    r\"\"\"
    Return [harmonic_order, emitted_oam, emitted_helicity].
    \"\"\"
    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    harmonic_ell_sigma = ...
    # ---------------------------------------------------------------

    return harmonic_ell_sigma
"""
    target = f"""
def answer():
    r\"\"\"
    Return [harmonic_order, emitted_oam, emitted_helicity].
    \"\"\"
    return {expected!r}
"""
    return _example(
        problem_id=f"{split}_v13_hhg_oam_{idx:05d}_{counts[0]}_{counts[1]}_{counts[2]}",
        split=split,
        family="official_template_discrete",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_checks(expected)},
        solution_trace=f"counts={counts!r}; harmonic/oam/helicity={expected!r}.",
        metadata={"domain": "amo_hhg", "answer_type": "integer_sequence"},
    )


def _distributed_qfi_symbolic(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    prefactor = rng.randint(1, 4 if difficulty == "medium" else 7)
    offset = rng.randint(1, 3 if difficulty == "medium" else 5)
    expected = f"{prefactor}*d*n*q**2/(q**2 - q + {offset})"
    setup = f"""
Consider a noisy distributed quantum sensing protocol with d sensor nodes and
n probe qubits per node. The target parameter is the scaled average of the
local phases. After local encoding and independent dephasing, the reduced
Fisher block in this synthetic model is diagonal in the collective mode. The
noise parameter is q=(1+exp(-gamma t))/2. Do not include gamma or t in the
final expression.

The effective collective-mode quantum Fisher information is

QFI = A d n q^2 / (q^2 - q + B),

with A={prefactor} and B={offset}.
"""
    main = "Return the symbolic QFI expression using only variables d, n and q."
    template = """
import sympy as sp

d, n, q = sp.symbols('d n q', positive=True)

def answer(d, n, q):
    r\"\"\"
    Return the quantum Fisher information for the scaled-average parameter.
    \"\"\"
    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    QFI = ...
    # ---------------------------------------------------------------

    return QFI
"""
    target = f"""
import sympy as sp

d, n, q = sp.symbols('d n q', positive=True)

def answer(d, n, q):
    r\"\"\"
    Return the quantum Fisher information for the scaled-average parameter.
    \"\"\"
    QFI = {expected}
    return QFI
"""
    return _example(
        problem_id=f"{split}_v13_qfi_symbolic_{idx:05d}_{prefactor}_{offset}",
        split=split,
        family="official_template_symbolic",
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
                    "args": [{"$sym": "d"}, {"$sym": "n"}, {"$sym": "q"}],
                    "expected": expected,
                    "variables": ["d", "n", "q"],
                    "tolerance": 1e-8,
                }
            ],
        },
        solution_trace=f"Substitute A={prefactor}, B={offset} into the stated QFI formula.",
        metadata={"domain": "quantum_information_qfi", "answer_type": "symbolic"},
    )


def _convection_galerkin_minimum(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    ks = [1.0, 1.5, 2.0, 2.5] if difficulty == "medium" else [0.8, 1.2, 1.6, 2.0, 2.4]
    a = rng.randint(2, 5)
    b = rng.randint(1, 4)
    c = rng.randint(4, 9)
    values = [((k * k + a) ** 3) / (b * k * k) + c * k for k in ks]
    best_idx = min(range(len(ks)), key=lambda i: values[i])
    expected = [round(values[best_idx], 6), ks[best_idx]]
    setup = f"""
In a one-mode Galerkin approximation to a convection stability problem, each
horizontal wavenumber k gives a candidate critical Rayleigh number

Ra(k) = ((k^2 + a)^3)/(b k^2) + c k.

This is not the full hydrodynamic eigenvalue problem; it is a controlled toy
surrogate designed to test numerical minimization and template following. Use
a={a}, b={b}, c={c}. Only evaluate the finite candidate set
K = {ks!r}. Report the smallest candidate Ra and its k.
"""
    main = "Return [Ra_c, k_c], with Ra_c rounded to six decimal places."
    template = """
def answer():
    r\"\"\"
    Return [critical_Rayleigh_number, critical_wavenumber].
    \"\"\"
    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    result = ...
    # ---------------------------------------------------------------

    return result
"""
    target = f"""
def answer():
    r\"\"\"
    Return [critical_Rayleigh_number, critical_wavenumber].
    \"\"\"
    ks = {ks!r}
    values = [((k**2 + {a})**3)/({b}*k**2) + {c}*k for k in ks]
    best = min(range(len(values)), key=lambda i: values[i])
    return [round(values[best], 6), ks[best]]
"""
    return _example(
        problem_id=f"{split}_v13_convection_min_{idx:05d}_{a}_{b}_{c}_{len(ks)}",
        split=split,
        family="official_template_numeric",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_checks(expected, 1e-6)},
        solution_trace=f"candidate values={values!r}; best={expected!r}.",
        metadata={"domain": "fluid_stability", "answer_type": "numeric_sequence"},
    )


def _u2_operator_enumeration(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    max_charge = rng.randint(3, 4 if difficulty == "easy" else 5)
    include_derivative = difficulty == "medium" and rng.choice([True, False])
    operators = ["tr(psi)"]
    if max_charge >= 2:
        operators.append("tr(psi^2)")
    if max_charge >= 3:
        operators.append("tr(psi^3)")
    if max_charge >= 4:
        operators.append("tr(psi) tr(psi^3)")
    if max_charge >= 5:
        operators.append("tr(psi^2) tr(psi^3)")
    if include_derivative:
        operators.append("tr(dpsi)")
    setup = f"""
Consider a toy rank-2 U(N) gauge theory with one adjoint fermion psi of charge
1. We write single-trace operators using the notation tr(...). In this toy
enumerator, Cayley-Hamilton relations remove all single traces longer than
three psi fields. Products are listed in increasing total charge. If derivative
insertions are enabled, dpsi denotes a separate adjoint field of charge 2.

Use max_charge={max_charge}. derivative_insertions_enabled={include_derivative}.
"""
    main = "Return the ordered list of indecomposable gauge-invariant operators allowed by these toy rules."
    template = """
def answer():
    r\"\"\"
    Return the ordered operator list.
    \"\"\"
    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    operators = ...
    # ---------------------------------------------------------------

    return operators
"""
    target = f"""
def answer():
    r\"\"\"
    Return the ordered operator list.
    \"\"\"
    operators = {operators!r}
    return operators
"""
    return _example(
        problem_id=f"{split}_v13_u2_operator_enum_{idx:05d}_{max_charge}_{int(include_derivative)}",
        split=split,
        family="official_template_enumeration",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _exact_sequence_checks(operators)},
        solution_trace=f"The toy enumeration gives {operators!r}.",
        metadata={"domain": "gauge_theory_operator_counting", "answer_type": "string_list"},
    )


def _linear_recurrence_generating_function(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    u0 = rng.randint(0, 2)
    u1 = rng.randint(1, 4)
    a = rng.randint(1, 3 if difficulty == "easy" else 5)
    b = rng.randint(1, 3 if difficulty == "easy" else 4)
    numerator_const = u0
    numerator_x = u1 - a * u0
    expected = f"({numerator_const} + {numerator_x}*x)/(1 - {a}*x - {b}*x**2)"
    setup = f"""
A lattice growth model has a total count u_n satisfying a second-order linear
recurrence. The initial values are u_0={u0}, u_1={u1}, and for n>=2,

u_n = {a} u_(n-1) + {b} u_(n-2).

Let Omega(x)=sum_{{n>=0}} u_n x^n be the ordinary generating function. The
standard recurrence manipulation gives

Omega(x) = (u_0 + (u_1-a u_0)x)/(1-a x-b x^2).
"""
    main = "Return the symbolic generating function Omega(x)."
    template = """
import sympy as sp

x = sp.symbols('x')

def answer(x):
    r\"\"\"
    Return Omega(x), the ordinary generating function.
    \"\"\"
    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    Omega = ...
    # ---------------------------------------------------------------

    return Omega
"""
    target = f"""
import sympy as sp

x = sp.symbols('x')

def answer(x):
    r\"\"\"
    Return Omega(x), the ordinary generating function.
    \"\"\"
    Omega = {expected}
    return Omega
"""
    return _example(
        problem_id=f"{split}_v13_recurrence_gf_{idx:05d}_{u0}_{u1}_{a}_{b}",
        split=split,
        family="official_template_symbolic",
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
                    "args": [{"$sym": "x"}],
                    "expected": expected,
                    "variables": ["x"],
                    "tolerance": 1e-8,
                }
            ],
        },
        solution_trace=f"Omega(x)={expected}.",
        metadata={"domain": "combinatorics_generating_function", "answer_type": "symbolic"},
    )


def _amplitude_damping_outputs(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    gamma_num = rng.choice([1, 2, 3, 4])
    gamma_den = rng.choice([5, 6, 8, 10] if difficulty == "easy" else [7, 9, 11, 12])
    if gamma_num >= gamma_den:
        gamma_num = 1
    gamma = gamma_num / gamma_den
    rho00 = rng.choice([0.25, 0.4, 0.6])
    rho11 = round(1.0 - rho00, 10)
    coh = rng.choice([0.1, 0.2, 0.3])
    sqrt_factor = (1.0 - gamma) ** 0.5
    expected = [
        round(rho00 + gamma * rho11, 8),
        round(sqrt_factor * coh, 8),
        round(1.0 - gamma, 8),
    ]
    setup = f"""
The qubit amplitude damping channel is defined by

A_gamma([[rho00, rho01], [rho10, rho11]])
  = [[rho00 + gamma rho11, sqrt(1-gamma) rho01],
     [sqrt(1-gamma) rho10, (1-gamma) rho11]].

Use gamma={gamma_num}/{gamma_den}. For the input state take rho00={rho00},
rho11={rho11}, and rho01=rho10={coh}. Return three diagnostic quantities:
the output rho00 entry, the output rho01 entry, and the population survival
factor 1-gamma.
"""
    main = "Return [rho00_out, rho01_out, survival], rounded to eight decimal places."
    template = """
def answer():
    r\"\"\"
    Return [rho00_out, rho01_out, survival].
    \"\"\"
    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    diagnostics = ...
    # ---------------------------------------------------------------

    return diagnostics
"""
    target = f"""
def answer():
    r\"\"\"
    Return [rho00_out, rho01_out, survival].
    \"\"\"
    gamma = {gamma_num} / {gamma_den}
    rho00 = {rho00}
    rho11 = {rho11}
    coh = {coh}
    return [
        round(rho00 + gamma * rho11, 8),
        round((1 - gamma) ** 0.5 * coh, 8),
        round(1 - gamma, 8),
    ]
"""
    return _example(
        problem_id=f"{split}_v13_amplitude_damping_{idx:05d}_{gamma_num}_{gamma_den}_{int(rho00*100)}",
        split=split,
        family="official_template_numeric",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_checks(expected, 1e-8)},
        solution_trace=f"Amplitude damping diagnostics={expected!r}.",
        metadata={"domain": "quantum_channel", "answer_type": "numeric_sequence"},
    )


def add_v13_rollout_instruction(example: SyntheticCritPTExample) -> SyntheticCritPTExample:
    prompt = (
        f"{example.prompt}\n\n"
        "请先在草稿中确认返回变量、返回顺序和输出类型；最终只输出一个 Python code block。"
    )
    return replace(example, prompt=prompt)
