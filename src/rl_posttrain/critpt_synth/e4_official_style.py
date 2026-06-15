from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable
from typing import Any

import sympy as sp

from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample


GeneratorFn = Callable[[random.Random, int, str, str], SyntheticCritPTExample]


E4_FINAL_INSTRUCTION = (
    "/no_think\n"
    "Fill the template above. Do not output <think> tags, hidden reasoning, prose, or comments. "
    "Respond with exactly one Python code block containing only the imports/symbol definitions needed by the template "
    "and the complete answer() implementation. Keep the code compact and stop after the closing code block."
)


def render_e4_prompt(problem_setup: str, main_problem: str, code_template: str) -> str:
    return (
        f"# Problem setup:\n{problem_setup.strip()}\n\n"
        f"# Main problem:\n{main_problem.strip()}\n\n"
        "### Parsing template:\n\n"
        f"```python\n{code_template.strip()}\n```\n\n"
        f"{E4_FINAL_INSTRUCTION}"
    )


def generate_e4_official_style_examples(size: int, seed: int, split: str) -> list[SyntheticCritPTExample]:
    rng = random.Random(seed)
    specs: list[tuple[GeneratorFn, float, tuple[list[str], list[float]]]] = [
        (_anomaly_coefficient_list, 0.14, (["medium", "hard"], [0.55, 0.45])),
        (_rounded_observable_tuple, 0.12, (["medium", "hard"], [0.60, 0.40])),
        (_sympy_log_derivative, 0.13, (["medium", "hard"], [0.50, 0.50])),
        (_distributed_qfi_expression, 0.13, (["medium", "hard"], [0.55, 0.45])),
        (_symbolic_choice_triplet, 0.12, (["medium", "hard"], [0.55, 0.45])),
        (_finite_tuple_set_filter, 0.13, (["medium", "hard"], [0.45, 0.55])),
        (_string_label_set_filter, 0.11, (["medium", "hard"], [0.55, 0.45])),
        (_operator_sympy_set, 0.11, (["medium", "hard"], [0.50, 0.50])),
        (_recurrence_generating_function, 0.11, (["medium", "hard"], [0.50, 0.50])),
    ]
    weights = [item[1] for item in specs]
    examples: list[SyntheticCritPTExample] = []
    seen: set[str] = set()
    attempts = 0
    while len(examples) < size:
        attempts += 1
        if attempts > size * 150:
            raise RuntimeError(f"too many duplicate E4 examples while building {split}")
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


def verify_e4_example(example: SyntheticCritPTExample) -> tuple[bool, str]:
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


def reference_output_for_code(code: str, call_args: list[str]) -> str:
    namespace: dict[str, Any] = {
        "math": math,
        "sp": sp,
        "sympy": sp,
        "__builtins__": __builtins__,
    }
    exec(compile(code, "<e4_reference_answer>", "exec"), namespace, namespace)
    answer = namespace.get("answer")
    if not callable(answer):
        raise ValueError("reference code did not define callable answer")
    args = [sp.Symbol(name) for name in call_args]
    return stable_repr(answer(*args))


def stable_repr(value: Any) -> str:
    if isinstance(value, sp.Basic):
        return str(sp.simplify(value))
    if isinstance(value, float):
        return repr(round(value, 12))
    if isinstance(value, tuple):
        inner = ", ".join(stable_repr(item) for item in value)
        if len(value) == 1:
            inner += ","
        return f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(stable_repr(item) for item in value) + "]"
    if isinstance(value, set):
        return "{" + ", ".join(sorted(stable_repr(item) for item in value)) + "}"
    if isinstance(value, dict):
        items = sorted((stable_repr(key), stable_repr(item)) for key, item in value.items())
        return "{" + ", ".join(f"{key}: {item}" for key, item in items) + "}"
    return repr(value)


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
    solution_trace: str,
    metadata: dict[str, Any],
    call_args: list[str] | None = None,
) -> SyntheticCritPTExample:
    call_args = call_args or []
    reference_output = reference_output_for_code(target, call_args)
    prompt = render_e4_prompt(setup, main, template)
    digest_payload = {
        "problem_id": problem_id,
        "family": family,
        "difficulty": difficulty,
        "call_args": call_args,
        "reference_output": reference_output,
    }
    param_hash = hashlib.sha256(
        json.dumps(digest_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return SyntheticCritPTExample(
        problem_id=problem_id,
        prompt=prompt,
        code_template=template.strip(),
        target_code=target.strip(),
        verifier={
            "kind": "e4_reference_output",
            "reference_output": reference_output,
            "reference_call_args": call_args,
        },
        split=split,
        family=family,
        difficulty=difficulty,
        solution_trace=solution_trace,
        metadata={
            **metadata,
            "generator_profile": "e4_official_style_final_answer_judge",
            "param_hash": param_hash,
            "reference_output": reference_output,
            "reference_call_args": call_args,
            "official_overlap": "none",
            "uses_official_prompt": False,
            "prompt_style": "official_problem_setup_main_problem_parsing_template",
        },
    )


def _anomaly_coefficient_list(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    upper = 6 if difficulty == "medium" else 9
    a = rng.randint(1, upper)
    b = rng.randint(1, upper)
    c = rng.randint(1, upper - 1)
    d = rng.randint(2, upper + 2)
    coeffs = [
        float(a * a - b),
        float(2 * a * b + c),
        float(b * b - a * c + d),
        float(a + b + c - d),
        float(d * (a - c) + b),
        float(a * b * c - d * d),
        float((a + d) * (b - c)),
    ]
    setup = f"""
Consider a synthetic eight-derivative anomaly density written in a fixed trace basis.
The physical tensor names are only labels; all needed algebra is in the reduced
coefficient rules below. The ordered basis is

1. tr(P^4)
2. tr(P^3) tr(P)
3. tr(B P^2)
4. tr(B^2)
5. tr(Omega P)
6. tr(P)^2 tr(B)
7. tr(Omega B)

For scheme constants a={a}, b={b}, c={c}, d={d}, the reduced subtraction gives
C1=a^2-b, C2=2ab+c, C3=b^2-ac+d, C4=a+b+c-d, C5=d(a-c)+b,
C6=abc-d^2, and C7=(a+d)(b-c).
"""
    main = "Return the seven coefficients in the exact basis order above as floats."
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
    coeffs = {coeffs!r}
    return coeffs
"""
    return _example(
        problem_id=f"{split}_e4_anomaly_coefficients_{idx:05d}_{a}_{b}_{c}_{d}",
        split=split,
        family="e4_coeff_list_official_shell",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        solution_trace=f"Substitute a={a}, b={b}, c={c}, d={d} into the seven coefficient rules.",
        metadata={"domain": "high_energy_anomaly", "answer_type": "list_float"},
    )


def _rounded_observable_tuple(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    count = 5 if difficulty == "medium" else 7
    rows = []
    for row_idx in range(count):
        signal = rng.randint(12, 95)
        background = rng.randint(2, 20)
        efficiency = round(rng.uniform(0.62, 0.94), 3)
        rows.append((f"bin_{row_idx+1}", signal, background, efficiency))
    corrected = [(signal - background) * efficiency for _, signal, background, efficiency in rows]
    total = round(sum(corrected), 3)
    peak_idx = max(range(len(rows)), key=lambda item: corrected[item])
    peak_label = rows[peak_idx][0]
    contrast = round(max(corrected) / max(1e-9, min(corrected)), 3)
    setup_rows = "\n".join(
        f"- {label}: signal={signal}, background={background}, efficiency={efficiency}"
        for label, signal, background, efficiency in rows
    )
    setup = f"""
A calibration note lists observed photon counts for a synthetic interferometry
scan. For each bin, the corrected yield is (signal-background)*efficiency.
Use the listed decimal efficiencies directly and round final numerical outputs
to three decimals.

{setup_rows}
"""
    main = (
        "Return total_corrected_yield, peak_bin_label, contrast_ratio, where contrast_ratio is "
        "max(corrected_yield)/min(corrected_yield)."
    )
    template = """
def answer():
    r\"\"\"
    Return rounded summary observables for the scan.

    Output
    ----------
    total_corrected_yield: float
    peak_bin_label: str
    contrast_ratio: float
    \"\"\"
    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    total_corrected_yield = ...
    peak_bin_label = ...
    contrast_ratio = ...
    # ---------------------------------------------------------------

    return total_corrected_yield, peak_bin_label, contrast_ratio
"""
    target = f"""
def answer():
    total_corrected_yield = {total!r}
    peak_bin_label = {peak_label!r}
    contrast_ratio = {contrast!r}
    return total_corrected_yield, peak_bin_label, contrast_ratio
"""
    return _example(
        problem_id=f"{split}_e4_observable_tuple_{idx:05d}_{count}_{peak_label}",
        split=split,
        family="e4_numeric_tuple_official_shell",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        solution_trace=f"Corrected yields are {[round(item, 3) for item in corrected]!r}.",
        metadata={"domain": "quantum_optics_calibration", "answer_type": "tuple_float_str_float"},
    )


def _sympy_log_derivative(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(2, 5 if difficulty == "medium" else 8)
    b = rng.randint(1, 5 if difficulty == "medium" else 9)
    c = rng.randint(1, 4 if difficulty == "medium" else 7)
    setup = f"""
In a reduced symbolic model for a deformed partition function, the normalized
generating factor is

Z(alpha) = (1+alpha)^{a} * exp({b}*alpha) / (1-{c}*alpha).

The response function is the logarithmic derivative g(alpha)=d log(Z)/d alpha.
Keep alpha symbolic and return a simplified SymPy expression.
"""
    main = "Compute g_alpha, the symbolic logarithmic derivative with respect to alpha."
    template = """
import sympy as sp

alpha = sp.symbols('alpha')

def answer(alpha):
    r\"\"\"
    Return the simplified logarithmic derivative g_alpha.
    \"\"\"
    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    g_alpha = ...
    # ---------------------------------------------------------------

    return g_alpha
"""
    target = f"""
import sympy as sp

alpha = sp.symbols('alpha')

def answer(alpha):
    Z = (1 + alpha)**{a} * sp.exp({b} * alpha) / (1 - {c} * alpha)
    g_alpha = sp.simplify(sp.diff(sp.log(Z), alpha))
    return g_alpha
"""
    return _example(
        problem_id=f"{split}_e4_sympy_log_derivative_{idx:05d}_{a}_{b}_{c}",
        split=split,
        family="e4_sympy_single_expr_official_shell",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        solution_trace="Differentiate log(Z), not Z itself, then simplify the rational expression.",
        metadata={"domain": "statistical_field_theory", "answer_type": "sympy_expr"},
        call_args=["alpha"],
    )


def _distributed_qfi_expression(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    a = rng.randint(2, 5 if difficulty == "medium" else 8)
    b = rng.randint(1, 4 if difficulty == "medium" else 7)
    c = rng.randint(1, 4 if difficulty == "medium" else 7)
    setup = f"""
Consider a synthetic distributed quantum sensing protocol. The collective mode
has d sensor nodes, n probes per node, generator amplitude F, integer block size
k, and dephasing survival q. After eliminating nuisance modes, the toy Fisher
block reduces to

Q = A * F^2 * d * n * q^2 / (k + B*q*(1-q) + C)

with A={a}, B={b}, C={c}. Return the expression exactly in terms of F, k, n, d,
and q.
"""
    main = "Return the simplified symbolic expression Q."
    template = """
import sympy as sp

F, k, n, d, q = sp.symbols('F k n d q', positive=True)

def answer(F, k, n, d, q):
    r\"\"\"
    Return the collective-mode quantum Fisher information Q.
    \"\"\"
    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    Q = ...
    # ---------------------------------------------------------------

    return Q
"""
    target = f"""
import sympy as sp

F, k, n, d, q = sp.symbols('F k n d q', positive=True)

def answer(F, k, n, d, q):
    Q = sp.simplify({a} * F**2 * d * n * q**2 / (k + {b} * q * (1 - q) + {c}))
    return Q
"""
    return _example(
        problem_id=f"{split}_e4_distributed_qfi_{idx:05d}_{a}_{b}_{c}",
        split=split,
        family="e4_sympy_param_expr_official_shell",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        solution_trace="Insert A, B, C into the reduced Fisher expression and simplify.",
        metadata={"domain": "quantum_metrology", "answer_type": "sympy_expr"},
        call_args=["F", "k", "n", "d", "q"],
    )


def _symbolic_choice_triplet(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    scale = rng.randint(1, 3 if difficulty == "medium" else 5)
    offset = rng.randint(1, 4 if difficulty == "medium" else 7)
    beta_choices = {
        "A": "beta/sigma2",
        "B": "beta**2/sigma2",
        "C": "sigma2/beta",
    }
    sigma_choices = {
        "A": "lambda_plus - lambda_minus",
        "B": "lambda_plus + lambda_minus",
        "C": "lambda_plus*lambda_minus",
    }
    beta_answer = rng.choice(["A", "B", "C"])
    sigma_answer = rng.choice(["A", "B", "C"])
    lambda_expr = f"{scale}*(lambda_plus + lambda_minus)/2 + {offset}*beta/sigma2"
    setup = f"""
A two-branch Gaussian update has eigenvalue symbols lambda_plus and lambda_minus,
drift beta, and noise variance sigma2. In this reduced benchmark, the scalar
normalization is

Lambda = {scale}*(lambda_plus + lambda_minus)/2 + {offset}*beta/sigma2.

Two auxiliary terms are reported as multiple-choice labels:

beta_term choices:
A. beta/sigma2
B. beta**2/sigma2
C. sigma2/beta

sigma_term choices:
A. lambda_plus - lambda_minus
B. lambda_plus + lambda_minus
C. lambda_plus*lambda_minus

For this instance the derivation selects beta_term={beta_answer} and
sigma_term={sigma_answer}.
"""
    main = "Return Lambda as a SymPy expression, then the two selected choice letters."
    template = """
import sympy as sp

lambda_plus, lambda_minus, beta, sigma2 = sp.symbols('lambda_plus lambda_minus beta sigma2')

def answer(lambda_plus, lambda_minus, beta, sigma2):
    r\"\"\"
    Return Lambda, answer_beta, answer_sigma2.
    \"\"\"
    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    Lambda = ...
    answer_beta = ...
    answer_sigma2 = ...
    # ---------------------------------------------------------------

    return Lambda, answer_beta, answer_sigma2
"""
    target = f"""
import sympy as sp

lambda_plus, lambda_minus, beta, sigma2 = sp.symbols('lambda_plus lambda_minus beta sigma2')

def answer(lambda_plus, lambda_minus, beta, sigma2):
    Lambda = sp.simplify({lambda_expr})
    answer_beta = {beta_answer!r}
    answer_sigma2 = {sigma_answer!r}
    return Lambda, answer_beta, answer_sigma2
"""
    return _example(
        problem_id=f"{split}_e4_symbolic_choice_{idx:05d}_{scale}_{offset}_{beta_answer}_{sigma_answer}",
        split=split,
        family="e4_sympy_choice_tuple_official_shell",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        solution_trace=(
            f"Lambda is {lambda_expr}; the selected labels are beta={beta_answer}, sigma={sigma_answer}."
        ),
        metadata={
            "domain": "statistical_inference",
            "answer_type": "tuple_sympy_str_str",
            "choice_beta_expression": beta_choices[beta_answer],
            "choice_sigma_expression": sigma_choices[sigma_answer],
        },
        call_args=["lambda_plus", "lambda_minus", "beta", "sigma2"],
    )


def _finite_tuple_set_filter(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    ell_max = 4 if difficulty == "medium" else 6
    modulus = rng.choice([3, 4, 5])
    residue = rng.randrange(modulus)
    shift = rng.randint(1, 3)
    min_abs = rng.randint(0, 2)
    allowed = {
        (ell, sigma)
        for ell in range(-ell_max, ell_max + 1)
        for sigma in (-1, 1)
        if (ell + shift * sigma) % modulus == residue and abs(ell) >= min_abs
    }
    setup = f"""
A synthetic harmonic-selection table uses two integer labels: orbital index ell
and helicity sigma. Candidate channels have ell in [-{ell_max}, {ell_max}] and
sigma in {{-1, +1}}. A channel survives if both conditions hold:

1. (ell + {shift}*sigma) mod {modulus} equals {residue}
2. abs(ell) >= {min_abs}

Return a Python set of (ell, sigma) tuples. Do not include rejected channels.
"""
    main = "Enumerate the surviving harmonic channels exactly."
    template = """
def answer():
    r\"\"\"
    Return the accepted set of (ell, sigma) channels.

    Output
    ----------
    allowed_channels: set[tuple[int, int]]
    \"\"\"
    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    allowed_channels = ...
    # ---------------------------------------------------------------

    return allowed_channels
"""
    target = f"""
def answer():
    allowed_channels = {allowed!r}
    return allowed_channels
"""
    return _example(
        problem_id=f"{split}_e4_tuple_set_filter_{idx:05d}_{ell_max}_{modulus}_{residue}_{shift}_{min_abs}",
        split=split,
        family="e4_tuple_set_filter_official_shell",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        solution_trace=f"Apply the modular rule to all ell/sigma candidates; accepted={sorted(allowed)!r}.",
        metadata={"domain": "amo_selection_rules", "answer_type": "set_tuple_int_int"},
    )


def _string_label_set_filter(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    labels = ["A1", "B2", "C3", "D4", "E5", "F6", "G7", "H8"] if difficulty == "hard" else [
        "A1",
        "B2",
        "C3",
        "D4",
        "E5",
        "F6",
    ]
    rows = []
    for label in labels:
        charge = rng.randint(-3, 4)
        spin = rng.choice([-1, 0, 1, 2])
        parity = rng.choice(["even", "odd"])
        rows.append((label, charge, spin, parity))
    target_charge = rng.choice([-2, -1, 0, 1, 2])
    spin_mod = rng.choice([2, 3])
    spin_residue = rng.randrange(spin_mod)
    required_parity = rng.choice(["even", "odd"])
    accepted = {
        label
        for label, charge, spin, parity in rows
        if charge >= target_charge and spin % spin_mod == spin_residue and parity == required_parity
    }
    table = "\n".join(
        f"- {label}: charge={charge}, spin={spin}, parity={parity}" for label, charge, spin, parity in rows
    )
    setup = f"""
A candidate list of synthetic BNS sectors is given below. Each sector has an
integer charge, an integer spin label, and a parity tag.

{table}

Keep exactly the sectors satisfying all filters:
charge >= {target_charge}; spin mod {spin_mod} equals {spin_residue}; parity is {required_parity}.
"""
    main = "Return the accepted sector labels as a Python set of strings."
    template = """
def answer():
    r\"\"\"
    Return the accepted sector labels.

    Output
    ----------
    accepted_labels: set[str]
    \"\"\"
    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    accepted_labels = ...
    # ---------------------------------------------------------------

    return accepted_labels
"""
    target = f"""
def answer():
    accepted_labels = {accepted!r}
    return accepted_labels
"""
    return _example(
        problem_id=f"{split}_e4_string_set_filter_{idx:05d}_{target_charge}_{spin_mod}_{spin_residue}_{required_parity}",
        split=split,
        family="e4_string_set_filter_official_shell",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        solution_trace=f"Filtering the listed sectors gives {sorted(accepted)!r}.",
        metadata={"domain": "condensed_matter_sector_filter", "answer_type": "set_str"},
    )


def _operator_sympy_set(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    dim_limit = 6 if difficulty == "medium" else 7
    charge_limit = rng.choice([0, 1, 2])
    candidates = [
        ("T**2", 4, 0),
        ("B*T", 5, 1),
        ("psi**2", 3, 0),
        ("F*T", 6, -1),
        ("B**2", 4, 2),
        ("F*psi", 5, -2),
        ("T*psi", 5, 1),
        ("F**2", 4, -2),
    ]
    if difficulty == "medium":
        candidates = candidates[:6]
    accepted_exprs = [expr for expr, dim, charge in candidates if dim <= dim_limit and abs(charge) <= charge_limit]
    rows = "\n".join(f"- {expr}: dimension={dim}, charge={charge}" for expr, dim, charge in candidates)
    setup = f"""
A symbolic effective-operator catalog uses SymPy symbols T, B, F, psi. Candidate
monomials are listed with engineering dimension and charge.

{rows}

Keep operators with dimension <= {dim_limit} and abs(charge) <= {charge_limit}.
Return a Python set of SymPy expressions, not strings.
"""
    main = "Return the accepted SymPy operator set."
    template = """
import sympy as sp

T, B, F, psi = sp.symbols('T B F psi')

def answer():
    r\"\"\"
    Return the accepted symbolic operators as a set of SymPy expressions.
    \"\"\"
    # ------------------ FILL IN YOUR RESULTS BELOW ------------------
    operators = ...
    # ---------------------------------------------------------------

    return operators
"""
    set_literal = "{" + ", ".join(accepted_exprs) + "}" if accepted_exprs else "set()"
    target = f"""
import sympy as sp

T, B, F, psi = sp.symbols('T B F psi')

def answer():
    operators = {set_literal}
    return operators
"""
    return _example(
        problem_id=f"{split}_e4_operator_set_{idx:05d}_{dim_limit}_{charge_limit}",
        split=split,
        family="e4_operator_set_official_shell",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        solution_trace=f"Accepted operator expressions are {accepted_exprs!r}.",
        metadata={"domain": "effective_operator_basis", "answer_type": "set_sympy_expr"},
    )


def _recurrence_generating_function(
    rng: random.Random, idx: int, split: str, difficulty: str
) -> SyntheticCritPTExample:
    p = rng.randint(1, 4 if difficulty == "medium" else 7)
    q = rng.randint(1, 3 if difficulty == "medium" else 5)
    a0 = rng.randint(1, 5)
    a1 = rng.randint(1, 8)
    setup = f"""
Let a sequence a_m be defined by a_0={a0}, a_1={a1}, and
a_(m+2) = {p}*a_(m+1) + {q}*a_m for m>=0. Let
G(z)=sum_{{m>=0}} a_m z^m be the ordinary generating function.

Use the standard recurrence identity:
G(z) = (a_0 + (a_1 - p*a_0) z)/(1 - p z - q z^2).
"""
    main = "Return the simplified symbolic generating function G(z)."
    template = """
import sympy as sp

z = sp.symbols('z')

def answer(z):
    r\"\"\"
    Return the ordinary generating function G(z).
    \"\"\"
    # ------------------ FILL IN YOUR RESULT BELOW ------------------
    G = ...
    # ---------------------------------------------------------------

    return G
"""
    target = f"""
import sympy as sp

z = sp.symbols('z')

def answer(z):
    G = sp.simplify(({a0} + ({a1} - {p}*{a0}) * z) / (1 - {p} * z - {q} * z**2))
    return G
"""
    return _example(
        problem_id=f"{split}_e4_generating_function_{idx:05d}_{p}_{q}_{a0}_{a1}",
        split=split,
        family="e4_generating_function_official_shell",
        difficulty=difficulty,
        setup=setup,
        main=main,
        template=template,
        target=target,
        solution_trace="Apply the ordinary generating-function identity for a second-order recurrence.",
        metadata={"domain": "symbolic_combinatorics", "answer_type": "sympy_expr"},
        call_args=["z"],
    )
