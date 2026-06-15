from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import random
from dataclasses import dataclass
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


E2_PROMPT_VERSION = "critpt-e2-llm-spec-solver-v1"


@dataclass(frozen=True)
class E2Spec:
    spec_id: str
    family: str
    difficulty: str
    domain_shell: str
    problem_setup: str
    main_problem: str
    code_template: str
    solver_code: str
    sample_params: dict[str, Any]
    validation_params: list[dict[str, Any]]
    return_keys: list[str]
    answer_type: str
    verifier_mode: str
    expected_sample: Any
    expected_validations: list[Any]
    solution_trace: str
    complexity_notes: str


def generate_e2_examples(
    *,
    size: int,
    seed: int,
    split: str,
    settings: JudgeSettings | None = None,
    cache_path: str = "",
    mock: bool = False,
    workers: int = 1,
    max_attempts_per_example: int = 10,
) -> list[SyntheticCritPTExample]:
    if size <= 0:
        return []
    if workers <= 1 or mock:
        return [
            _generate_one_e2_example(
                ordinal=ordinal,
                seed=seed,
                split=split,
                settings=settings,
                cache_path=cache_path,
                mock=mock,
                max_attempts=max_attempts_per_example,
            )
            for ordinal in range(size)
        ]

    examples: list[SyntheticCritPTExample | None] = [None] * size
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _generate_one_e2_example,
                ordinal=ordinal,
                seed=seed,
                split=split,
                settings=settings,
                cache_path=cache_path,
                mock=mock,
                max_attempts=max_attempts_per_example,
            ): ordinal
            for ordinal in range(size)
        }
        for future in as_completed(futures):
            ordinal = futures[future]
            examples[ordinal] = future.result()
    return [example for example in examples if example is not None]


def _generate_one_e2_example(
    *,
    ordinal: int,
    seed: int,
    split: str,
    settings: JudgeSettings | None,
    cache_path: str,
    mock: bool,
    max_attempts: int,
) -> SyntheticCritPTExample:
    rng = random.Random(seed + ordinal * 1009)
    last_error = ""
    for attempt in range(max_attempts):
        idx = ordinal * max_attempts + attempt
        spec = mock_e2_spec(rng, idx, split) if mock else request_e2_spec(
            rng=rng,
            idx=idx,
            split=split,
            settings=settings or JudgeSettings.from_env(),
            cache_path=cache_path,
        )
        try:
            example = build_example_from_spec(spec, split=split)
            ok, reason = verify_e2_example(example)
            if ok:
                print(f"[e2] built split={split} ordinal={ordinal} attempt={attempt + 1} family={example.family}", flush=True)
                return example
            last_error = reason
        except Exception as exc:
            last_error = str(exc)
            if mock:
                raise
    raise RuntimeError(f"failed to build E2 example split={split} ordinal={ordinal}: {last_error}")


def build_example_from_spec(spec: E2Spec, *, split: str) -> SyntheticCritPTExample:
    validate_spec_contract(spec)
    validate_solver_variants(spec)
    prompt = render_prompt(spec.problem_setup, spec.main_problem, spec.code_template)
    target_code = build_target_code(spec.solver_code, spec.sample_params, spec.return_keys)
    verifier = build_verifier(spec)
    payload = {
        "spec_id": spec.spec_id,
        "family": spec.family,
        "difficulty": spec.difficulty,
        "domain_shell": spec.domain_shell,
        "problem_setup": spec.problem_setup,
        "main_problem": spec.main_problem,
        "code_template": spec.code_template,
        "solver_code": spec.solver_code,
        "sample_params": spec.sample_params,
        "return_keys": spec.return_keys,
        "expected_sample": spec.expected_sample,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    prompt_chars = len(prompt)
    return SyntheticCritPTExample(
        problem_id=f"{split}_e2_{spec.family}_{digest}",
        prompt=prompt,
        code_template=spec.code_template.strip(),
        target_code=target_code.strip(),
        verifier=verifier,
        split=split,
        family=f"e2_{spec.family}",
        difficulty=spec.difficulty,
        solution_trace=spec.solution_trace,
        metadata={
            "generator_profile": "e2_llm_spec_solver",
            "e2_family": spec.family,
            "domain_shell": spec.domain_shell,
            "answer_type": spec.answer_type,
            "prompt_chars": prompt_chars,
            "llm_generated_solver": True,
            "llm_generated_prompt": True,
            "complexity_notes": spec.complexity_notes,
            "sample_params": json.dumps(spec.sample_params, ensure_ascii=False, sort_keys=True),
            "validation_params": json.dumps(spec.validation_params, ensure_ascii=False, sort_keys=True),
            "expected_validations": json.dumps(spec.expected_validations, ensure_ascii=False, sort_keys=True),
            "uses_official_prompt": False,
            "official_overlap": "none",
        },
    )


def build_target_code(solver_code: str, params: dict[str, Any], return_keys: list[str]) -> str:
    if len(return_keys) == 1:
        result_line = f"    return result[{return_keys[0]!r}]"
    else:
        key_list = ", ".join(repr(key) for key in return_keys)
        result_line = f"    return tuple(result[key] for key in [{key_list}])"
    return (
        f"{solver_code.strip()}\n\n"
        "def answer():\n"
        f"    params = {params!r}\n"
        "    result = solve(params)\n"
        f"{result_line}\n"
    )


def build_verifier(spec: E2Spec) -> dict[str, Any]:
    return build_verifier_for_expected(spec.verifier_mode, spec.return_keys, spec.expected_sample)


def build_verifier_for_expected(mode: str, return_keys: list[str], expected_value: Any) -> dict[str, Any]:
    expected = expected_value
    if len(return_keys) > 1 and mode == "numeric_sequence":
        expected = list(expected)
    return {
        "kind": "code",
        "timeout_s": 4.0,
        "checks": [{"mode": mode, "expected": expected, "tolerance": 1e-7}],
    }


def verify_e2_example(example: SyntheticCritPTExample) -> tuple[bool, str]:
    result = verify_code_completion(example.assistant_code_block(), example.verifier)
    return result.ok, result.reason


def validate_solver_variants(spec: E2Spec) -> None:
    all_params = [spec.sample_params, *spec.validation_params]
    all_expected = [spec.expected_sample, *spec.expected_validations]
    if len(all_params) != len(all_expected):
        raise ValueError("validation_params and expected_validations length mismatch")
    for idx, (params, expected) in enumerate(zip(all_params, all_expected)):
        code = build_target_code(spec.solver_code, params, spec.return_keys)
        verifier = build_verifier_for_expected(spec.verifier_mode, spec.return_keys, expected)
        result = verify_code_completion(f"```python\n{code}\n```", verifier)
        if not result.ok:
            raise ValueError(f"solver validation {idx} failed: {result.reason}")


def validate_spec_contract(spec: E2Spec) -> None:
    template = spec.code_template.replace(" ", "")
    if "return{" in template:
        raise ValueError("code_template must return variables directly, not a wrapper dict")
    for key in spec.return_keys:
        if key not in spec.code_template:
            raise ValueError(f"return key is not present in code_template: {key}")
    if len(spec.return_keys) == 1:
        expected_return = f"return{spec.return_keys[0]}"
        if expected_return not in template:
            raise ValueError(f"single-key code_template must end by returning {spec.return_keys[0]}")


def request_e2_spec(
    *,
    rng: random.Random,
    idx: int,
    split: str,
    settings: JudgeSettings,
    cache_path: str = "",
) -> E2Spec:
    profile = rng.choice(_PROFILES)
    messages = build_e2_messages(profile=profile, idx=idx, split=split, seed=rng.randint(1, 10**9))
    key = cache_key(
        E2_PROMPT_VERSION,
        settings.model,
        split,
        str(idx),
        json.dumps(profile, ensure_ascii=False, sort_keys=True),
        messages[-1]["content"],
    )
    cache = JsonSqliteCache(cache_path) if cache_path else None
    payload = cache.get(key) if cache else None
    if payload is None:
        payload = openai_chat_json(settings=settings, messages=messages)
        if cache:
            cache.set(key, payload)
    return spec_from_payload(payload, spec_id=f"{split}_{idx:05d}_{key[:12]}")


def spec_from_payload(payload: dict[str, Any], *, spec_id: str) -> E2Spec:
    required = [
        "family",
        "difficulty",
        "domain_shell",
        "problem_setup",
        "main_problem",
        "code_template",
        "solver_code",
        "sample_params",
        "validation_params",
        "return_keys",
        "answer_type",
        "verifier_mode",
        "expected_sample",
        "expected_validations",
        "solution_trace",
        "complexity_notes",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"E2 payload missing keys: {missing}")
    return_keys = [str(key) for key in payload["return_keys"]]
    expected_sample = _normalize_expected(payload["expected_sample"], return_keys)
    expected_validations = [_normalize_expected(item, return_keys) for item in list(payload["expected_validations"])]
    return E2Spec(
        spec_id=spec_id,
        family=str(payload["family"]),
        difficulty=str(payload["difficulty"]),
        domain_shell=str(payload["domain_shell"]),
        problem_setup=str(payload["problem_setup"]).strip(),
        main_problem=str(payload["main_problem"]).strip(),
        code_template=str(payload["code_template"]).strip(),
        solver_code=_ensure_solver_imports(str(payload["solver_code"]).strip()),
        sample_params=dict(payload["sample_params"]),
        validation_params=[dict(item) for item in payload["validation_params"]],
        return_keys=return_keys,
        answer_type=str(payload["answer_type"]),
        verifier_mode=str(payload["verifier_mode"]),
        expected_sample=expected_sample,
        expected_validations=expected_validations,
        solution_trace=str(payload["solution_trace"]).strip(),
        complexity_notes=str(payload["complexity_notes"]).strip(),
    )


def _normalize_expected(value: Any, return_keys: list[str]) -> Any:
    if isinstance(value, dict):
        if len(return_keys) == 1 and return_keys[0] in value:
            return value[return_keys[0]]
        if len(return_keys) > 1 and all(key in value for key in return_keys):
            return [value[key] for key in return_keys]
    return value


def _ensure_solver_imports(code: str) -> str:
    prefix = "import math\n"
    if "def solve(" not in code:
        raise ValueError("solver_code must define solve(params)")
    if "import math" not in code:
        code = prefix + code
    return code


def build_e2_messages(*, profile: dict[str, Any], idx: int, split: str, seed: int) -> list[dict[str, str]]:
    system = (
        "You create synthetic scientific benchmark data. Unlike a prose-only wrapper, you must generate "
        "both a long realistic problem statement and a parametric Python solver. The solver defines the truth. "
        "Return JSON only. Do not use official CritPT prompt text or challenge-specific constants."
    )
    user = {
        "task": "Generate one E2 synthetic problem spec with a long prompt and a verifiable parametric solver.",
        "split": split,
        "index": idx,
        "seed": seed,
        "profile": profile,
        "target_prompt_length_chars": "2500-4500 for problem_setup + main_problem before parsing template",
        "required_json_schema": {
            "family": "short snake_case category",
            "difficulty": "hard",
            "domain_shell": "scientific domain label",
            "problem_setup": "long realistic setup, 1900-3400 chars, with some harmless distractors",
            "main_problem": "clear task, 600-1100 chars, return variables, rounding/order/type requirements",
            "code_template": "complete answer() skeleton with variables assigned to ... and returned",
            "solver_code": "Python code defining solve(params). It may import math only. It returns a dict.",
            "sample_params": "dict used for this problem instance",
            "validation_params": "list of 2 additional dicts with same keys but different values",
            "return_keys": "ordered list of keys in solve(params) to return",
            "answer_type": "numeric_tuple, numeric_list, exact_list, boolean_tuple, or symbolic_string",
            "verifier_mode": "numeric_sequence, numeric, exact_sequence, exact, or set_exact",
            "expected_sample": "result for sample_params in the same object shape returned by answer()",
            "expected_validations": "results for validation_params, used to check solver consistency",
            "solution_trace": "brief human explanation of the solver logic",
            "complexity_notes": "what makes the prompt long/non-toy",
        },
        "hard_constraints": [
            "The problem must not reveal expected_sample.",
            "The problem must contain enough formulas and parameters that solve(sample_params) is derivable.",
            "The solver must be deterministic, pure, and must not use randomness, files, network, json, eval, exec, or open.",
            "Prefer finite grids, small matrices written as lists, recurrences, root search over listed candidates, or exact enumeration.",
            "Include 2-4 irrelevant but harmless scientific details in problem_setup.",
            "Use exactly the return variables shown in code_template.",
            "The code_template must return variables directly, for example `return labels` or `return a, b`; never return a wrapper dict.",
            "For numeric outputs, round inside solve exactly as the main problem states.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


_PROFILES: list[dict[str, Any]] = [
    {
        "family": "long_numeric_grid",
        "domain_shell": "fluid stability / reduced Galerkin calculation",
        "return_shape": "Ra_c, k_c, ratio_w2T",
        "core_skill": "scan finite candidate grid; compute secondary diagnostic at best candidate",
    },
    {
        "family": "operator_enumeration_long",
        "domain_shell": "gauge-invariant operator enumeration",
        "return_shape": "ordered operator labels",
        "core_skill": "filter and sort labels with a charge cutoff and canonical naming",
    },
    {
        "family": "recurrence_generation_long",
        "domain_shell": "discrete stochastic process / generating function surrogate",
        "return_shape": "sequence summary tuple",
        "core_skill": "iterate recurrence and return rounded summary statistics",
    },
    {
        "family": "piecewise_response_long",
        "domain_shell": "phase diagram / materials response table",
        "return_shape": "phase labels and rounded aggregate",
        "core_skill": "apply piecewise thresholds to listed samples and aggregate selected values",
    },
    {
        "family": "matrix_reduction_long",
        "domain_shell": "linearized reaction network / small matrix surrogate",
        "return_shape": "rounded row diagnostics",
        "core_skill": "multiply small matrices/lists and compute requested diagnostics without numpy",
    },
    {
        "family": "candidate_root_long",
        "domain_shell": "calibrated spectroscopy / residual minimization",
        "return_shape": "best candidate and residual summary",
        "core_skill": "evaluate residual over listed candidates and tie-break deterministically",
    },
    {
        "family": "interval_inventory_long",
        "domain_shell": "detector run log / event-window accounting",
        "return_shape": "accepted ids and numeric exposure",
        "core_skill": "filter intervals by overlapping cuts and sum weighted exposure",
    },
    {
        "family": "symbolic_canonical_long",
        "domain_shell": "effective-field-theory bookkeeping",
        "return_shape": "canonical symbolic strings",
        "core_skill": "canonicalize simple symbolic labels under a finite rule table",
    },
    {
        "family": "ranked_pairing_long",
        "domain_shell": "astronomical source matching",
        "return_shape": "ordered pair labels and score",
        "core_skill": "score bipartite candidate pairs, enforce one-to-one matching, sort output",
    },
    {
        "family": "bounded_dynamic_program_long",
        "domain_shell": "energy storage schedule / lab protocol planning",
        "return_shape": "optimal value and chosen policy labels",
        "core_skill": "small dynamic program over listed states and actions",
    },
    {
        "family": "histogram_moment_long",
        "domain_shell": "particle-bin histogram calibration",
        "return_shape": "rounded moments and selected bin ids",
        "core_skill": "transform binned counts, normalize, compute moments, return thresholded bins",
    },
]


def mock_e2_spec(rng: random.Random, idx: int, split: str) -> E2Spec:
    del rng, split
    params = {
        "a": 5,
        "b": 2,
        "c": 3,
        "d": 4,
        "u0": 1.5,
        "u1": 0.25,
        "v1": 0.5,
        "ks": [0.7, 1.1, 1.5, 1.9, 2.3],
    }
    solver = """
import math

def solve(params):
    ks = params["ks"]
    a = params["a"]
    b = params["b"]
    c = params["c"]
    d = params["d"]
    u0 = params["u0"]
    u1 = params["u1"]
    v1 = params["v1"]
    values = [((k*k + a)**3)/(b*k*k) + c*k + d/(1+k*k) for k in ks]
    best = min(range(len(values)), key=lambda i: values[i])
    k_c = ks[best]
    return {
        "Ra_c": round(values[best], 6),
        "k_c": k_c,
        "ratio_w2T": round((u0 + u1*k_c*k_c)/(1 + v1*k_c), 6),
    }
""".strip()
    expected = [90.414797, 1.5, 1.178571]
    setup = (
        "Consider a reduced neutral-stability calculation for a convection layer with mixed thermal "
        "boundary conditions. The original derivation contains a pressure normalization, a Prandtl-number "
        "convention, and a weak boundary correction from the lower plate, but in this benchmark instance "
        "the Galerkin reduction has already been performed. The neutral curve is represented by the "
        "surrogate Rayleigh functional R(k)=((k^2+a)^3)/(b*k^2)+c*k+d/(1+k^2). A secondary diagnostic, "
        "the probe-height velocity-to-temperature ratio, is ratio(k)=(u0+u1*k^2)/(1+v1*k). Use "
        "a=5, b=2, c=3, d=4, u0=1.5, u1=0.25, and v1=0.5. The reported reduced model ignores the "
        "unused pressure normalization and the quoted Prandtl-number convention. Evaluate only the "
        "candidate horizontal wavenumbers k=[0.7, 1.1, 1.5, 1.9, 2.3]."
    )
    main = (
        "Find the candidate k that minimizes R(k). Return the minimum Rayleigh value rounded to six "
        "decimal places, the best candidate k, and ratio(k) at that best k rounded to six decimals."
    )
    return E2Spec(
        spec_id=f"mock_{idx:05d}",
        family="long_numeric_grid",
        difficulty="hard",
        domain_shell="fluid_stability",
        problem_setup=setup,
        main_problem=main,
        code_template="def answer():\n    Ra_c = ...\n    k_c = ...\n    ratio_w2T = ...\n    return Ra_c, k_c, ratio_w2T",
        solver_code=solver,
        sample_params=params,
        validation_params=[],
        return_keys=["Ra_c", "k_c", "ratio_w2T"],
        answer_type="numeric_tuple",
        verifier_mode="numeric_sequence",
        expected_sample=expected,
        expected_validations=[],
        solution_trace="Scan the finite k grid, choose the lowest R(k), then evaluate the ratio at that k.",
        complexity_notes="Long prompt with irrelevant physical conventions, finite-grid optimization, and a secondary diagnostic.",
    )
