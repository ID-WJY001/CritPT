from __future__ import annotations

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from rl_posttrain.critpt_synth.prompting import render_prompt
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.critpt_synth.verifier import verify_code_completion
from rl_posttrain.model_judge.openai_compatible import (
    JudgeSettings,
    JsonSqliteCache,
    cache_key,
    openai_chat_json,
)


E1_PROMPT_VERSION = "critpt-e1-llm-background-wrapper-v1"


@dataclass(frozen=True)
class E1Core:
    core_id: str
    split: str
    e1_type: str
    difficulty: str
    domain_shell: str
    facts: dict[str, Any]
    setup: str
    main: str
    template: str
    target: str
    verifier: dict[str, Any]
    solution_trace: str
    metadata: dict[str, Any]


CoreGenerator = Callable[[random.Random, int, str, str], E1Core]


def generate_e1_examples(
    size: int,
    seed: int,
    split: str,
    *,
    wrap_mode: str = "template",
    llm_limit: int | None = None,
    llm_settings: JudgeSettings | None = None,
    llm_cache_path: str = "",
    llm_workers: int = 1,
) -> list[SyntheticCritPTExample]:
    rng = random.Random(seed + 100)
    cores: list[E1Core] = []
    seen: set[str] = set()
    attempts = 0
    while len(cores) < size:
        attempts += 1
        if attempts > max(size * 200, 200):
            raise RuntimeError(f"too many duplicate E1 examples while building {split}")
        idx = len(cores)
        generator = E1_GENERATORS[idx % len(E1_GENERATORS)]
        difficulty = rng.choices(["medium", "hard"], weights=[0.65, 0.35], k=1)[0]
        core = generator(rng, idx, split, difficulty)
        if core.core_id in seen:
            continue
        seen.add(core.core_id)
        cores.append(core)

    def convert(item: tuple[int, E1Core]) -> SyntheticCritPTExample:
        idx, core = item
        use_llm = wrap_mode == "llm" and (llm_limit is None or idx < llm_limit)
        return build_example_from_core(
            core,
            use_llm=use_llm,
            llm_settings=llm_settings,
            llm_cache_path=llm_cache_path,
        )

    if wrap_mode == "llm" and llm_workers > 1:
        with ThreadPoolExecutor(max_workers=llm_workers) as executor:
            return list(executor.map(convert, enumerate(cores)))
    return [convert(item) for item in enumerate(cores)]


def verify_e1_example(example: SyntheticCritPTExample) -> tuple[bool, str]:
    result = verify_code_completion(example.assistant_code_block(), example.verifier)
    return result.ok, result.reason


def build_example_from_core(
    core: E1Core,
    *,
    use_llm: bool = False,
    llm_settings: JudgeSettings | None = None,
    llm_cache_path: str = "",
) -> SyntheticCritPTExample:
    setup = core.setup
    main = core.main
    wrapper_payload: dict[str, Any] = {"mode": "template"}
    if use_llm:
        setup, main, wrapper_payload = wrap_core_with_llm(
            core,
            settings=llm_settings or JudgeSettings.from_env(),
            cache_path=llm_cache_path,
        )
    prompt = render_prompt(setup, main, core.template)
    payload = {
        "core_id": core.core_id,
        "e1_type": core.e1_type,
        "difficulty": core.difficulty,
        "facts": core.facts,
        "wrapper": wrapper_payload,
    }
    example_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    metadata = {
        **core.metadata,
        "generator_profile": "e1_llm_wrapped",
        "e1_type": core.e1_type,
        "domain_shell": core.domain_shell,
        "uses_official_prompt": False,
        "official_overlap": "none",
        "llm_background_wrapped": use_llm,
        "llm_wrapper_hash": example_hash if use_llm else "",
        "core_facts": json.dumps(core.facts, ensure_ascii=False, sort_keys=True),
    }
    return SyntheticCritPTExample(
        problem_id=f"{core.core_id}_e1_{example_hash}",
        prompt=prompt,
        code_template=core.template.strip(),
        target_code=core.target.strip(),
        verifier=core.verifier,
        split=core.split,
        family=core.e1_type,
        difficulty=core.difficulty,
        solution_trace=core.solution_trace,
        metadata=metadata,
    )


def build_llm_wrapper_messages(core: E1Core) -> list[dict[str, str]]:
    system = (
        "You rewrite synthetic scientific benchmark cores into realistic problem statements. "
        "You are not allowed to solve the problem. You must preserve every rule, formula, "
        "condition, number, symbol, variable name, return variable, and output type from the seed text. "
        "Do not reinterpret tuples/lists/tables. Return JSON only."
    )
    public_facts = {key: value for key, value in core.facts.items() if key != "expected"}
    user_payload = {
        "task": (
            "Rewrite the seed Problem setup and seed Main problem into a realistic benchmark prompt. "
            "You may add harmless scientific context, but you must not drop or change any rule. "
            "Do not compute the final returned object. Do not mention that this is synthetic."
        ),
        "required_json": {"problem_setup": "string", "main_problem": "string"},
        "core": {
            "type": core.e1_type,
            "domain_shell": core.domain_shell,
            "facts_without_answer": public_facts,
            "seed_problem_setup": core.setup,
            "seed_main_problem": core.main,
            "python_template": core.template,
            "answer_type": core.metadata.get("answer_type", ""),
        },
        "hard_constraints": [
            "Preserve formulas and inequalities exactly.",
            "Preserve whether a listed tuple is an input channel, a candidate label, or an output object.",
            "Preserve empty-set, ordering, filtering, rounding, and threshold rules.",
            "Do not replace a computational rule with a vague reference to an ordering convention.",
            "Do not reveal the final answer unless the seed setup explicitly gives a table that the problem asks to return.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def wrap_core_with_llm(
    core: E1Core,
    *,
    settings: JudgeSettings,
    cache_path: str = "",
) -> tuple[str, str, dict[str, Any]]:
    messages = build_llm_wrapper_messages(core)
    key = cache_key(
        E1_PROMPT_VERSION,
        settings.model,
        core.core_id,
        json.dumps(core.facts, sort_keys=True),
        core.template,
    )
    cache = JsonSqliteCache(cache_path) if cache_path else None
    payload = cache.get(key) if cache else None
    if payload is None:
        payload = openai_chat_json(settings=settings, messages=messages)
        if cache:
            cache.set(key, payload)
    if "problem_setup" not in payload and isinstance(payload.get("required_json"), dict):
        nested = dict(payload["required_json"])
        if "problem_setup" in nested or "main_problem" in nested:
            payload = {**payload, **nested}
    setup = str(payload.get("problem_setup", "")).strip()
    main = str(payload.get("main_problem", "")).strip()
    if not setup or not main:
        raise ValueError(f"LLM wrapper returned incomplete payload for {core.core_id}: {payload!r}")
    return setup, main, {"mode": "llm", "model": settings.model, "cache_key": key}


def _core(
    *,
    core_id: str,
    split: str,
    e1_type: str,
    difficulty: str,
    domain_shell: str,
    facts: dict[str, Any],
    setup: str,
    main: str,
    template: str,
    target: str,
    verifier: dict[str, Any],
    solution_trace: str,
    metadata: dict[str, Any],
) -> E1Core:
    return E1Core(
        core_id=core_id,
        split=split,
        e1_type=e1_type,
        difficulty=difficulty,
        domain_shell=domain_shell,
        facts=facts,
        setup=setup.strip(),
        main=main.strip(),
        template=template.strip(),
        target=target.strip(),
        verifier=verifier,
        solution_trace=solution_trace,
        metadata=metadata,
    )


def _numeric_sequence_checks(expected: list[float | int], tolerance: float = 1e-8) -> list[dict[str, Any]]:
    return [{"mode": "numeric_sequence", "expected": expected, "tolerance": tolerance}]


def _exact_sequence_checks(expected: list[Any]) -> list[dict[str, Any]]:
    return [{"mode": "exact_sequence", "expected": expected}]


def _coeff_table(rng: random.Random, idx: int, split: str, difficulty: str) -> E1Core:
    upper = 6 if difficulty == "hard" else 4
    a, b, c = [rng.randint(1, upper) for _ in range(3)]
    coeffs = [a + b, a * b - c, b * b + c, a * c - b]
    facts = {"a": a, "b": b, "c": c, "rules": ["C1=a+b", "C2=a*b-c", "C3=b^2+c", "C4=a*c-b"]}
    setup = f"""
In a toy anomaly notebook, a scalar density X is expanded in the ordered basis
[B1, B2, B3, B4]. The scheme constants are a={a}, b={b}, c={c}. The coefficients
are defined by C1=a+b, C2=a*b-c, C3=b^2+c, and C4=a*c-b.
"""
    main = "Return the coefficient list [C1, C2, C3, C4] in this exact order."
    template = """
def answer():
    coeffs = ...
    return coeffs
"""
    target = f"""
def answer():
    coeffs = {coeffs!r}
    return coeffs
"""
    return _core(
        core_id=f"{split}_e1_coeff_table_{idx:05d}_{a}_{b}_{c}",
        split=split,
        e1_type="e1_coeff_table",
        difficulty=difficulty,
        domain_shell="holography_anomaly_coefficients",
        facts=facts,
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_checks(coeffs)},
        solution_trace=f"Substitute a={a}, b={b}, c={c}; coeffs={coeffs!r}.",
        metadata={"answer_type": "numeric_list"},
    )


def _symbolic_formula(rng: random.Random, idx: int, split: str, difficulty: str) -> E1Core:
    prefactor = rng.randint(2, 7 if difficulty == "hard" else 4)
    offset = rng.randint(1, 5 if difficulty == "hard" else 3)
    expected = f"{prefactor}*d*n*q**2/(q**2 - q + {offset})"
    facts = {"prefactor": prefactor, "offset": offset, "variables": ["d", "n", "q"], "formula": expected}
    setup = f"""
A distributed sensing calculation reduces to one collective Fisher block. In this
toy version the target expression has the form A*d*n*q^2/(q^2-q+B), with
A={prefactor} and B={offset}. The final expression must use only d, n, and q.
"""
    main = "Return the symbolic expression for QFI."
    template = """
import sympy as sp

d, n, q = sp.symbols("d n q", positive=True)

def answer(d, n, q):
    QFI = ...
    return QFI
"""
    target = f"""
import sympy as sp

d, n, q = sp.symbols("d n q", positive=True)

def answer(d, n, q):
    QFI = {expected}
    return QFI
"""
    return _core(
        core_id=f"{split}_e1_symbolic_formula_{idx:05d}_{prefactor}_{offset}",
        split=split,
        e1_type="e1_symbolic_formula",
        difficulty=difficulty,
        domain_shell="quantum_information_qfi",
        facts=facts,
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
        solution_trace=f"Insert A={prefactor}, B={offset} into the given closed form.",
        metadata={"answer_type": "sympy_expression"},
    )


def _piecewise_strings(rng: random.Random, idx: int, split: str, difficulty: str) -> E1Core:
    alpha = rng.randint(1, 5 if difficulty == "hard" else 3)
    beta = rng.randint(2, 7 if difficulty == "hard" else 5)
    gamma = rng.randint(1, 5 if difficulty == "hard" else 3)
    expected = [f"{alpha}/(1-x)", f"{alpha}*log(mu/(Pz*x)) + {beta}*(1-x)", f"-{gamma}/x"]
    setup = f"""
A reduced matching kernel K(x) is already split into three regions. For x<0 it
is alpha/(1-x), for 0<x<1 it is alpha*log(mu/(Pz*x))+beta*(1-x), and for x>1
it is -gamma/x. Use alpha={alpha}, beta={beta}, gamma={gamma}.
"""
    main = "Return the three region expressions as strings in the order x<0, 0<x<1, x>1."
    template = """
def answer():
    expr_lt0 = ...
    expr_mid = ...
    expr_gt1 = ...
    return expr_lt0, expr_mid, expr_gt1
"""
    target = f"""
def answer():
    expr_lt0 = {expected[0]!r}
    expr_mid = {expected[1]!r}
    expr_gt1 = {expected[2]!r}
    return expr_lt0, expr_mid, expr_gt1
"""
    return _core(
        core_id=f"{split}_e1_piecewise_strings_{idx:05d}_{alpha}_{beta}_{gamma}",
        split=split,
        e1_type="e1_piecewise_strings",
        difficulty=difficulty,
        domain_shell="qcd_lamet_piecewise_kernel",
        facts={"alpha": alpha, "beta": beta, "gamma": gamma, "expected": expected},
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _exact_sequence_checks(expected)},
        solution_trace=f"The three region strings are {expected!r}.",
        metadata={"answer_type": "string_tuple"},
    )


def _single_numeric(rng: random.Random, idx: int, split: str, difficulty: str) -> E1Core:
    m = rng.randint(2, 9 if difficulty == "hard" else 5)
    ks = list(range(1, rng.choice([4, 5, 6]) + 1))
    expected = round(sum(1 / (k * k + m) for k in ks), 6)
    setup = f"""
A finite spectral diagnostic is approximated by S=sum_k 1/(k^2+m). Use m={m}
and the finite grid k={ks!r}. Report the value rounded to six decimal places.
"""
    main = "Return the scalar S."
    template = """
def answer():
    S = ...
    return S
"""
    target = f"""
def answer():
    S = round(sum(1/(k*k + {m}) for k in {ks!r}), 6)
    return S
"""
    return _core(
        core_id=f"{split}_e1_single_numeric_{idx:05d}_{m}_{len(ks)}",
        split=split,
        e1_type="e1_single_numeric",
        difficulty=difficulty,
        domain_shell="finite_spectral_trace",
        facts={"m": m, "ks": ks, "expected": expected},
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "numeric", "expected": expected, "tolerance": 1e-6}]},
        solution_trace=f"Direct finite sum over {ks!r} gives S={expected}.",
        metadata={"answer_type": "float"},
    )


def _numeric_algorithm(rng: random.Random, idx: int, split: str, difficulty: str) -> E1Core:
    a = rng.randint(2, 6)
    b = rng.randint(1, 4)
    c = rng.randint(2, 7)
    ks = [0.8, 1.2, 1.6, 2.0, 2.4] if difficulty == "hard" else [0.8, 1.2, 1.6, 2.0]
    values = [((k * k + a) ** 3) / (b * k * k) + c * k for k in ks]
    best = min(range(len(values)), key=lambda i: values[i])
    expected = [round(values[best], 6), ks[best]]
    setup = f"""
A Galerkin stability scan uses R(k)=((k^2+a)^3)/(b*k^2)+c*k. Use a={a},
b={b}, c={c}, and evaluate only the candidate grid k={ks!r}.
"""
    main = "Return [minimum_R, best_k], with minimum_R rounded to six decimal places."
    template = """
def answer():
    result = ...
    return result
"""
    target = f"""
def answer():
    ks = {ks!r}
    values = [((k**2 + {a})**3)/({b}*k**2) + {c}*k for k in ks]
    best = min(range(len(values)), key=lambda i: values[i])
    return [round(values[best], 6), ks[best]]
"""
    return _core(
        core_id=f"{split}_e1_numeric_algorithm_{idx:05d}_{a}_{b}_{c}_{len(ks)}",
        split=split,
        e1_type="e1_numeric_algorithm",
        difficulty=difficulty,
        domain_shell="fluid_stability_grid_scan",
        facts={"a": a, "b": b, "c": c, "ks": ks, "expected": expected},
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_checks(expected, 1e-6)},
        solution_trace=f"Candidate values={values!r}; best result={expected!r}.",
        metadata={"answer_type": "numeric_list"},
    )


def _multi_output_tuple(rng: random.Random, idx: int, split: str, difficulty: str) -> E1Core:
    r = rng.randint(3, 8)
    sigma2 = rng.randint(1, 5)
    vbar = rng.randint(2, 5)
    lam = r - sigma2 / (vbar**2)
    expected = [lam, "decrease", "decrease"]
    setup = f"""
A two-state growth surrogate reports Lambda=r-sigma2/vbar^2. In this setting,
larger beta decreases growth and larger sigma2 also decreases growth. Use r={r},
sigma2={sigma2}, and vbar={vbar}.
"""
    main = "Return Lambda, answer_beta, answer_sigma2."
    template = """
def answer():
    Lambda = ...
    answer_beta = ...
    answer_sigma2 = ...
    return Lambda, answer_beta, answer_sigma2
"""
    target = f"""
def answer():
    Lambda = {r} - {sigma2}/({vbar}**2)
    answer_beta = "decrease"
    answer_sigma2 = "decrease"
    return Lambda, answer_beta, answer_sigma2
"""
    return _core(
        core_id=f"{split}_e1_multi_output_tuple_{idx:05d}_{r}_{sigma2}_{vbar}",
        split=split,
        e1_type="e1_multi_output_tuple",
        difficulty=difficulty,
        domain_shell="stochastic_growth_mixed_output",
        facts={"r": r, "sigma2": sigma2, "vbar": vbar, "expected": expected},
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={
            "kind": "code",
            "timeout_s": 2.0,
            "checks": [{"mode": "exact_sequence", "expected": expected, "tolerance": 1e-8}],
        },
        solution_trace=f"Lambda={lam}; both qualitative answers are decrease.",
        metadata={"answer_type": "mixed_tuple"},
    )


def _operator_list(rng: random.Random, idx: int, split: str, difficulty: str) -> E1Core:
    max_charge = rng.choice([3, 4, 5] if difficulty == "hard" else [3, 4])
    fields = [("psi", 1), ("chi", 2), ("F", 3)]
    operators: list[tuple[int, str]] = []
    for name, charge in fields:
        for power in range(1, 3):
            total = charge * power
            if total <= max_charge:
                label = f"tr({name})" if power == 1 else f"tr({name}^{power})"
                operators.append((total, label))
    expected = [label for _total, label in sorted(operators)]
    setup = f"""
A toy gauge-operator table has fields {fields!r}. Allowed labels are tr(name)
and tr(name^2). Keep only labels with total charge <= {max_charge}; sort by
increasing total charge, then by label.
"""
    main = "Return the exact ordered list of operator labels."
    template = """
def answer():
    operators = ...
    return operators
"""
    target = f"""
def answer():
    fields = {fields!r}
    max_charge = {max_charge}
    items = []
    for name, charge in fields:
        for power in range(1, 3):
            total = charge * power
            if total <= max_charge:
                label = f"tr({{name}})" if power == 1 else f"tr({{name}}^{{power}})"
                items.append((total, label))
    return [label for _total, label in sorted(items)]
"""
    return _core(
        core_id=f"{split}_e1_operator_list_{idx:05d}_{max_charge}",
        split=split,
        e1_type="e1_operator_list",
        difficulty=difficulty,
        domain_shell="gauge_operator_enumeration",
        facts={"fields": fields, "max_charge": max_charge, "expected": expected},
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _exact_sequence_checks(expected)},
        solution_trace=f"Enumerating charge-filtered canonical labels gives {expected!r}.",
        metadata={"answer_type": "ordered_string_list"},
    )


def _number_set(rng: random.Random, idx: int, split: str, difficulty: str) -> E1Core:
    prefix = rng.choice(["182", "117", "153", "221"])
    start = rng.choice([10, 12, 14, 18])
    stop = start + rng.choice([10, 12, 14])
    modulo = rng.choice([3, 4, 5])
    residue = rng.randrange(modulo)
    parity = rng.randrange(2)
    candidates = [x for x in range(start, stop + 1) if x % modulo == residue and x % 2 == parity]
    if difficulty == "hard":
        forbidden = candidates[:]
    else:
        forbidden = candidates[: max(0, len(candidates) - 1)]
    expected = sorted({f"{prefix}.{x}" for x in candidates if x not in forbidden})
    setup = f"""
Candidate labels have the form "{prefix}.x" with integer suffix x from {start}
to {stop}. Keep x only if x mod {modulo} is {residue}, x mod 2 is {parity}, and
x is not in the forbidden list {forbidden!r}.
"""
    main = "Return the surviving labels as a Python set; return set() if none survive."
    template = """
def answer():
    labels = ...
    return labels
"""
    target = f"""
def answer():
    prefix = {prefix!r}
    forbidden = {forbidden!r}
    labels = {{
        f"{{prefix}}.{{x}}"
        for x in range({start}, {stop + 1})
        if x % {modulo} == {residue} and x % 2 == {parity} and x not in forbidden
    }}
    return labels
"""
    return _core(
        core_id=f"{split}_e1_number_set_{idx:05d}_{prefix}_{start}_{stop}_{modulo}_{residue}_{parity}",
        split=split,
        e1_type="e1_number_set",
        difficulty=difficulty,
        domain_shell="magnetic_group_number_filter",
        facts={"prefix": prefix, "start": start, "stop": stop, "modulo": modulo, "residue": residue, "parity": parity, "forbidden": forbidden, "expected": expected},
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "set_exact", "expected": expected}]},
        solution_trace=f"Surviving labels are {expected!r}.",
        metadata={"answer_type": "string_set", "expected_empty": len(expected) == 0},
    )


def _boolean_choice(rng: random.Random, idx: int, split: str, difficulty: str) -> E1Core:
    m1 = rng.choice([-5, -3, -2, 2, 3, 5])
    m2 = rng.choice([-4, -1, 1, 4, 6])
    expected = m1 * m2 < 0
    setup = f"""
A reduced interface criterion says that an edge state exists exactly when two
mass parameters have opposite signs, i.e. m1*m2<0. Use m1={m1} and m2={m2}.
"""
    main = "Return the boolean has_edge_state."
    template = """
def answer():
    has_edge_state = ...
    return has_edge_state
"""
    target = f"""
def answer():
    has_edge_state = ({m1} * {m2}) < 0
    return has_edge_state
"""
    return _core(
        core_id=f"{split}_e1_boolean_choice_{idx:05d}_{m1}_{m2}",
        split=split,
        e1_type="e1_boolean_choice",
        difficulty=difficulty,
        domain_shell="topological_interface_boolean",
        facts={"m1": m1, "m2": m2, "expected": expected},
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "exact", "expected": expected}]},
        solution_trace=f"m1*m2={m1*m2}; edge state={expected}.",
        metadata={"answer_type": "bool"},
    )


def _mixed_output(rng: random.Random, idx: int, split: str, difficulty: str) -> E1Core:
    exponents = {"A": rng.choice([1.2, 1.5, 2.1]), "B": rng.choice([1.7, 2.2, 2.6]), "C": rng.choice([1.4, 1.8, 2.4])}
    a, b, c = rng.randint(-2, 3), rng.randint(-2, 3), rng.randint(1, 4)
    particles = [name for name, value in exponents.items() if value < 2.0]
    code = a + 10 * b + 100 * c
    expected = [particles, code, 2]
    setup = f"""
A toy phase diagram marks a particle as crystallized when its exponent is below
2.0. The exponents are {exponents!r}. A scaling code is defined from a={a},
b={b}, c={c} by code=a+10*b+100*c.
"""
    main = "Return crystal_particles, code, threshold."
    template = """
def answer():
    crystal_particles = ...
    code = ...
    threshold = ...
    return crystal_particles, code, threshold
"""
    target = f"""
def answer():
    exponents = {exponents!r}
    crystal_particles = [name for name, value in exponents.items() if value < 2.0]
    code = {a} + 10*({b}) + 100*{c}
    threshold = 2
    return crystal_particles, code, threshold
"""
    return _core(
        core_id=f"{split}_e1_mixed_output_{idx:05d}_{a}_{b}_{c}",
        split=split,
        e1_type="e1_mixed_output",
        difficulty=difficulty,
        domain_shell="many_body_phase_diagram",
        facts={"exponents": exponents, "a": a, "b": b, "c": c, "expected": expected},
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "exact_sequence", "expected": expected}]},
        solution_trace=f"Particles below threshold are {particles!r}; code={code}.",
        metadata={"answer_type": "mixed_tuple"},
    )


def _many_arg_symbolic(rng: random.Random, idx: int, split: str, difficulty: str) -> E1Core:
    setup = """
A response calculation uses ten symbolic inputs. The numerator is a*x+b*y+c*z
and the denominator is 1+p*x+q*y+r*z+s. No other variables should appear.
"""
    main = "Return the symbolic response expression."
    template = """
def answer(a, b, c, p, q, r, s, x, y, z):
    response = ...
    return response
"""
    target = """
def answer(a, b, c, p, q, r, s, x, y, z):
    response = (a*x + b*y + c*z) / (1 + p*x + q*y + r*z + s)
    return response
"""
    expected = "(a*x + b*y + c*z)/(1 + p*x + q*y + r*z + s)"
    args = [{"$sym": name} for name in ["a", "b", "c", "p", "q", "r", "s", "x", "y", "z"]]
    return _core(
        core_id=f"{split}_e1_many_arg_symbolic_{idx:05d}",
        split=split,
        e1_type="e1_many_arg_symbolic",
        difficulty=difficulty,
        domain_shell="high_parameter_response_function",
        facts={"variables": ["a", "b", "c", "p", "q", "r", "s", "x", "y", "z"], "expected": expected},
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "symbolic", "args": args, "expected": expected, "variables": ["a", "b", "c", "p", "q", "r", "s", "x", "y", "z"]}]},
        solution_trace="Substitute the given numerator and denominator directly.",
        metadata={"answer_type": "sympy_expression", "argument_count": 10},
    )


def _long_vector(rng: random.Random, idx: int, split: str, difficulty: str) -> E1Core:
    labels = ["X0", "Z0", "X1", "Z1", "X0 X1", "Z0 Z1", "Y0 Y1", "X0 Z1"]
    if difficulty == "medium":
        labels = labels[:6]
    coeffs = [round(rng.choice([-2, -1.5, -1, 0, 0.5, 1, 1.5, 2]), 2) for _ in labels]
    setup = f"""
A sparse Hamiltonian reconstruction cell fixes the Pauli-string order as
{labels!r}. The corresponding coefficients are listed in the same order as
{coeffs!r}. Return exactly the coefficient vector in that order.
"""
    main = "Return coeffs."
    template = """
def answer():
    coeffs = ...
    return coeffs
"""
    target = f"""
def answer():
    coeffs = {coeffs!r}
    return coeffs
"""
    return _core(
        core_id=f"{split}_e1_long_vector_{idx:05d}_{len(labels)}",
        split=split,
        e1_type="e1_long_vector",
        difficulty=difficulty,
        domain_shell="pauli_string_coefficient_vector",
        facts={"labels": labels, "coeffs": coeffs},
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": _numeric_sequence_checks(coeffs)},
        solution_trace=f"Read coefficients in the given label order: {coeffs!r}.",
        metadata={"answer_type": "numeric_list", "vector_length": len(coeffs)},
    )


def _recurrence_gf(rng: random.Random, idx: int, split: str, difficulty: str) -> E1Core:
    u0 = rng.randint(0, 3)
    u1 = rng.randint(1, 6)
    a = rng.randint(1, 5 if difficulty == "hard" else 3)
    b = rng.randint(1, 4 if difficulty == "hard" else 2)
    expected = f"({u0} + ({u1} - {a}*{u0})*x)/(1 - {a}*x - {b}*x**2)"
    setup = f"""
A counting sequence obeys u0={u0}, u1={u1}, and u_n={a}*u_(n-1)+{b}*u_(n-2)
for n>=2. Let Omega(x)=sum_n u_n*x^n.
"""
    main = "Return the ordinary generating function Omega(x)."
    template = """
import sympy as sp

x = sp.symbols("x")

def answer(x):
    Omega = ...
    return Omega
"""
    target = f"""
import sympy as sp

x = sp.symbols("x")

def answer(x):
    Omega = {expected}
    return Omega
"""
    return _core(
        core_id=f"{split}_e1_recurrence_gf_{idx:05d}_{u0}_{u1}_{a}_{b}",
        split=split,
        e1_type="e1_recurrence_gf",
        difficulty=difficulty,
        domain_shell="combinatorics_generating_function",
        facts={"u0": u0, "u1": u1, "a": a, "b": b, "expected": expected},
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "symbolic", "args": [{"$sym": "x"}], "expected": expected, "variables": ["x"]}]},
        solution_trace=f"Use Omega=(u0+(u1-a*u0)*x)/(1-a*x-b*x^2) with u0={u0}, u1={u1}.",
        metadata={"answer_type": "sympy_expression"},
    )


def _discrete_channel(rng: random.Random, idx: int, split: str, difficulty: str) -> E1Core:
    pulses = [
        (-1, rng.choice([-1, 1])),
        (2, rng.choice([-1, 1])),
        (rng.choice([1, 3, 4]), rng.choice([-1, 1])),
    ]
    channel_count = 4 if difficulty == "medium" else 6
    raw_channels: list[tuple[int, int, int]] = []
    while len(raw_channels) < channel_count:
        channel = tuple(rng.randint(0, 3) for _ in range(3))
        if sum(channel) and channel not in raw_channels:
            raw_channels.append(channel)
    expected: list[list[int]] = []
    for counts in raw_channels:
        order = sum(counts)
        oam = sum(n * ell for n, (ell, _sigma) in zip(counts, pulses))
        hsum = sum(n * sigma for n, (_ell, sigma) in zip(counts, pulses))
        helicity = 1 if hsum > 0 else -1 if hsum < 0 else 0
        expected.append([order, oam, helicity])
    setup = f"""
Three driving pulses have (ell, sigma) values {pulses!r}. For each channel
(nA,nB,nC), compute order=nA+nB+nC, oam=sum(n_i*ell_i), and helicity=sign of
sum(n_i*sigma_i). The channels, in order, are {raw_channels!r}.
"""
    main = "Return [[order, oam, helicity], ...] in the same channel order."
    template = """
def answer():
    channels = ...
    return channels
"""
    target = f"""
def answer():
    pulses = {pulses!r}
    raw_channels = {raw_channels!r}
    channels = []
    for counts in raw_channels:
        order = sum(counts)
        oam = sum(n * ell for n, (ell, _sigma) in zip(counts, pulses))
        hsum = sum(n * sigma for n, (_ell, sigma) in zip(counts, pulses))
        helicity = 1 if hsum > 0 else -1 if hsum < 0 else 0
        channels.append([order, oam, helicity])
    return channels
"""
    return _core(
        core_id=f"{split}_e1_discrete_channel_{idx:05d}_{channel_count}",
        split=split,
        e1_type="e1_discrete_channel",
        difficulty=difficulty,
        domain_shell="hhg_oam_selection_rule",
        facts={"pulses": pulses, "channels": raw_channels, "expected": expected},
        setup=setup,
        main=main,
        template=template,
        target=target,
        verifier={"kind": "code", "timeout_s": 2.0, "checks": [{"mode": "exact_sequence", "expected": expected}]},
        solution_trace=f"Apply the channel formulas in order; expected={expected!r}.",
        metadata={"answer_type": "list_of_integer_triples"},
    )


E1_GENERATORS: list[CoreGenerator] = [
    _coeff_table,
    _symbolic_formula,
    _piecewise_strings,
    _single_numeric,
    _numeric_algorithm,
    _multi_output_tuple,
    _operator_list,
    _number_set,
    _boolean_choice,
    _mixed_output,
    _many_arg_symbolic,
    _long_vector,
    _recurrence_gf,
    _discrete_channel,
]


def e1_type_names() -> list[str]:
    probe_rng = random.Random(123)
    return [generator(probe_rng, idx, "probe", "medium").e1_type for idx, generator in enumerate(E1_GENERATORS)]
