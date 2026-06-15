from __future__ import annotations

import ast
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
from typing import Any

from rl_posttrain.critpt_synth.e2_llm_specs import _PROFILES, build_target_code, build_verifier_for_expected
from rl_posttrain.critpt_synth.prompting import render_prompt
from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.critpt_synth.verifier import (
    ALLOWED_IMPORT_ROOTS,
    FORBIDDEN_CALL_NAMES,
    _call_name,
    _exec_code,
    verify_code_completion,
)
from rl_posttrain.model_judge.openai_compatible import (
    JudgeSettings,
    JsonSqliteCache,
    cache_key,
    openai_chat_json,
)


E2_TEMPLATE_PROMPT_VERSION = "critpt-e2-llm-template-bank-v1"


@dataclass(frozen=True)
class E2Template:
    template_id: str
    family: str
    difficulty: str
    domain_shell: str
    problem_setup_template: str
    main_problem_template: str
    code_template: str
    solver_code: str
    sampler_code: str
    return_keys: list[str]
    answer_type: str
    verifier_mode: str
    solution_trace_template: str
    complexity_notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "family": self.family,
            "difficulty": self.difficulty,
            "domain_shell": self.domain_shell,
            "problem_setup_template": self.problem_setup_template,
            "main_problem_template": self.main_problem_template,
            "code_template": self.code_template,
            "solver_code": self.solver_code,
            "sampler_code": self.sampler_code,
            "return_keys": self.return_keys,
            "answer_type": self.answer_type,
            "verifier_mode": self.verifier_mode,
            "solution_trace_template": self.solution_trace_template,
            "complexity_notes": self.complexity_notes,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "E2Template":
        return E2Template(
            template_id=str(raw["template_id"]),
            family=str(raw["family"]),
            difficulty=str(raw.get("difficulty", "hard")),
            domain_shell=str(raw["domain_shell"]),
            problem_setup_template=str(raw["problem_setup_template"]).strip(),
            main_problem_template=str(raw["main_problem_template"]).strip(),
            code_template=str(raw["code_template"]).strip(),
            solver_code=_ensure_math_import(str(raw["solver_code"]).strip()),
            sampler_code=str(raw["sampler_code"]).strip(),
            return_keys=[str(key) for key in raw["return_keys"]],
            answer_type=str(raw["answer_type"]),
            verifier_mode=str(raw["verifier_mode"]),
            solution_trace_template=str(raw["solution_trace_template"]).strip(),
            complexity_notes=str(raw["complexity_notes"]).strip(),
        )


def generate_template_bank(
    *,
    templates_per_family: int,
    seed: int,
    settings: JudgeSettings,
    cache_path: str,
    workers: int = 4,
    mock: bool = False,
    profile_limit: int = 0,
) -> list[E2Template]:
    jobs: list[tuple[int, dict[str, Any], int]] = []
    rng = random.Random(seed)
    profiles = _PROFILES[:profile_limit] if profile_limit > 0 else _PROFILES
    for family_idx, profile in enumerate(profiles):
        for variant_idx in range(templates_per_family):
            jobs.append((family_idx * templates_per_family + variant_idx, profile, rng.randint(1, 10**9)))
    if mock:
        return [mock_template(idx, profile) for idx, profile, _ in jobs]
    templates: list[E2Template | None] = [None] * len(jobs)
    failures: list[tuple[int, str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                request_template_with_retries,
                idx=idx,
                profile=profile,
                seed=job_seed,
                settings=settings,
                cache_path=cache_path,
            ): (idx, str(profile.get("family", "")))
            for idx, profile, job_seed in jobs
        }
        for future in as_completed(futures):
            idx, requested_family = futures[future]
            try:
                template = future.result()
            except Exception as exc:
                failures.append((idx, requested_family, str(exc)))
                print(
                    f"[e2-template] failed idx={idx} requested_family={requested_family}: {exc}",
                    flush=True,
                )
                continue
            templates[idx] = template
            print(f"[e2-template] built idx={idx} family={template.family}", flush=True)
    built = [template for template in templates if template is not None]
    if not built:
        raise RuntimeError("all E2 template generation jobs failed")
    requested_counts = Counter(str(profile.get("family", "")) for _, profile, _ in jobs)
    built_family_names = Counter(template.family for template in built)
    print(
        json.dumps(
            {
                "event": "e2_template_bank_done",
                "requested": len(jobs),
                "built": len(built),
                "failed": len(failures),
                "requested_families": dict(requested_counts),
                "built_template_families": dict(built_family_names),
                "failures": [
                    {"idx": idx, "requested_family": family, "error": error[:300]}
                    for idx, family, error in failures[:20]
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return built


def request_template_with_retries(
    *,
    idx: int,
    profile: dict[str, Any],
    seed: int,
    settings: JudgeSettings,
    cache_path: str,
    max_attempts: int = 3,
) -> E2Template:
    last_error = ""
    for attempt in range(max_attempts):
        try:
            template = request_e2_template(
                idx=idx * max_attempts + attempt,
                profile=profile,
                seed=seed + attempt,
                settings=settings,
                cache_path=cache_path,
            )
            validate_template(template)
            return template
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(f"failed to build E2 template idx={idx} family={profile.get('family')}: {last_error}")


def request_e2_template(
    *,
    idx: int,
    profile: dict[str, Any],
    seed: int,
    settings: JudgeSettings,
    cache_path: str,
) -> E2Template:
    messages = build_template_messages(profile=profile, idx=idx, seed=seed)
    key = cache_key(
        E2_TEMPLATE_PROMPT_VERSION,
        settings.model,
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
    payload["template_id"] = f"e2tpl_{idx:04d}_{key[:10]}"
    return E2Template.from_dict(payload)


def build_template_messages(*, profile: dict[str, Any], idx: int, seed: int) -> list[dict[str, str]]:
    system = (
        "You create scalable synthetic scientific benchmark templates. Return JSON only. "
        "The template must be self-contained: a long problem text with Python .format placeholders, "
        "a deterministic solver, and a deterministic parameter sampler."
    )
    user = {
        "task": "Generate one reusable E2 template for programmatic data expansion.",
        "index": idx,
        "seed": seed,
        "profile": profile,
        "required_json_schema": {
            "family": "short snake_case category; should match or refine profile.family",
            "difficulty": "hard",
            "domain_shell": "scientific domain label",
            "problem_setup_template": "1900-3400 chars; use {param_name} placeholders for all sampled data",
            "main_problem_template": "600-1100 chars; clear return variables, type/order/rounding requirements",
            "code_template": "complete answer() skeleton with variables assigned to ... and returned directly",
            "solver_code": "Python code defining solve(params). It may import math only. It returns a dict.",
            "sampler_code": "Python code defining sample_params(rng, idx). No imports. It returns a JSON-serializable dict.",
            "return_keys": "ordered list of keys from solve(params) returned by answer()",
            "answer_type": "numeric_tuple, numeric_list, exact_list, boolean_tuple, or symbolic_string",
            "verifier_mode": "numeric_sequence, numeric, exact_sequence, exact, or set_exact",
            "solution_trace_template": "brief explanation; may use the same placeholders",
            "complexity_notes": "what makes this template non-toy and long-context",
        },
        "hard_constraints": [
            "The prompt must never reveal the expected answer.",
            "Every numeric/list/string parameter needed by the solver must appear in the prompt through placeholders.",
            "Use Python str.format placeholders like {matrix}, {weights}, {cutoff}; escape literal braces as {{ and }}.",
            "sampler_code must define sample_params(rng, idx), use rng.randint/rng.uniform/rng.choice, and return serializable values.",
            "solver_code must define solve(params), be pure/deterministic, and use no files, network, eval, exec, open, or randomness.",
            "code_template must return variables directly, e.g. return labels or return score, labels; never return a wrapper dict.",
            "Keep generated parameter sizes small enough for a model to reason about, but include enough rows/terms/candidates to avoid toy tasks.",
            "For numeric outputs, solver_code must round exactly as main_problem_template states.",
            "Do not copy official CritPT wording or constants.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def materialize_examples(
    *,
    templates: list[E2Template],
    size: int,
    split: str,
    seed: int,
) -> list[SyntheticCritPTExample]:
    if not templates:
        raise ValueError("template bank is empty")
    examples: list[SyntheticCritPTExample] = []
    for ordinal in range(size):
        template = templates[ordinal % len(templates)]
        examples.append(materialize_one(template=template, ordinal=ordinal, split=split, seed=seed))
    return examples


def materialize_one(*, template: E2Template, ordinal: int, split: str, seed: int) -> SyntheticCritPTExample:
    rng = random.Random(seed + ordinal * 1009)
    params = run_sampler(template, rng, ordinal)
    expected = run_solver(template, params)
    answer_expected = expected[template.return_keys[0]] if len(template.return_keys) == 1 else [
        expected[key] for key in template.return_keys
    ]
    fmt = {key: format_prompt_value(value) for key, value in params.items()}
    setup = format_template(template.problem_setup_template, fmt)
    main = format_template(template.main_problem_template, fmt)
    trace = format_template(template.solution_trace_template, fmt)
    prompt = render_prompt(setup, main, template.code_template)
    target_code = build_target_code(template.solver_code, params, template.return_keys)
    verifier = build_verifier_for_expected(template.verifier_mode, template.return_keys, answer_expected)
    digest_payload = {
        "template_id": template.template_id,
        "split": split,
        "ordinal": ordinal,
        "params": params,
        "expected": answer_expected,
    }
    digest = hashlib.sha256(json.dumps(digest_payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    example = SyntheticCritPTExample(
        problem_id=f"{split}_e2tpl_{template.family}_{digest}",
        prompt=prompt,
        code_template=template.code_template,
        target_code=target_code,
        verifier=verifier,
        split=split,
        family=f"e2_{template.family}",
        difficulty=template.difficulty,
        solution_trace=trace,
        metadata={
            "generator_profile": "e2_llm_template_bank",
            "template_id": template.template_id,
            "e2_family": template.family,
            "domain_shell": template.domain_shell,
            "answer_type": template.answer_type,
            "prompt_chars": len(prompt),
            "llm_generated_template": True,
            "programmatic_parameter_expansion": True,
            "complexity_notes": template.complexity_notes,
            "sample_params": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "expected_sample": json.dumps(answer_expected, ensure_ascii=False, sort_keys=True),
            "uses_official_prompt": False,
            "official_overlap": "none",
        },
    )
    ok, reason = verify_e2_template_example(example)
    if not ok:
        raise ValueError(f"materialized example failed verification: {reason}")
    return example


def validate_template(template: E2Template) -> None:
    validate_code_contract(template)
    check_safe_template_code(template.solver_code, required_func="solve", allow_imports=True)
    check_safe_template_code(template.sampler_code, required_func="sample_params", allow_imports=False)
    seen: set[str] = set()
    for idx in range(6):
        params = run_sampler(template, random.Random(9000 + idx), idx)
        if not isinstance(params, dict) or not params:
            raise ValueError("sampler must return a non-empty dict")
        key = json.dumps(params, ensure_ascii=False, sort_keys=True)
        seen.add(key)
        _ = materialize_one(template=template, ordinal=idx, split="validate", seed=7000)
    if len(seen) < 3:
        raise ValueError("sampler does not create enough parameter diversity")


def validate_code_contract(template: E2Template) -> None:
    compact = template.code_template.replace(" ", "")
    if "return{" in compact:
        raise ValueError("code_template must return variables directly, not a wrapper dict")
    for key in template.return_keys:
        if key not in template.code_template:
            raise ValueError(f"return key missing from code_template: {key}")
    if len(template.return_keys) == 1 and f"return{template.return_keys[0]}" not in compact:
        raise ValueError(f"single-key template must return {template.return_keys[0]} directly")


def check_safe_template_code(code: str, *, required_func: str, allow_imports: bool) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"syntax_error in {required_func}: {exc}") from exc
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == required_func]
    if len(funcs) != 1:
        raise ValueError(f"expected exactly one {required_func} function")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not allow_imports:
                raise ValueError(f"{required_func} must not import modules")
            names = [alias.name for alias in getattr(node, "names", [])]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                root = name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS or root != "math":
                    raise ValueError(f"unsafe import in {required_func}: {name}")
        elif isinstance(node, ast.Call):
            func_name = _call_name(node.func)
            if func_name in FORBIDDEN_CALL_NAMES:
                raise ValueError(f"unsafe call in {required_func}: {func_name}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"unsafe attribute in {required_func}: {node.attr}")


def run_sampler(template: E2Template, rng: random.Random, idx: int) -> dict[str, Any]:
    namespace = _exec_code(template.sampler_code)
    sample_params = namespace["sample_params"]
    params = sample_params(rng, idx)
    return json.loads(json.dumps(params, ensure_ascii=False))


def run_solver(template: E2Template, params: dict[str, Any]) -> dict[str, Any]:
    namespace = _exec_code(template.solver_code)
    result = namespace["solve"](params)
    if not isinstance(result, dict):
        raise ValueError("solve(params) must return a dict")
    for key in template.return_keys:
        if key not in result:
            raise ValueError(f"solve(params) missing return key: {key}")
    return json.loads(json.dumps(result, ensure_ascii=False))


def verify_e2_template_example(example: SyntheticCritPTExample) -> tuple[bool, str]:
    result = verify_code_completion(example.assistant_code_block(), example.verifier)
    return result.ok, result.reason


def format_prompt_value(value: Any) -> str:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def format_template(template: str, values: dict[str, str]) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        raise ValueError(f"template references missing placeholder: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"template formatting failed; literal braces may need escaping: {exc}") from exc


def _ensure_math_import(code: str) -> str:
    if "def solve(" not in code:
        raise ValueError("solver_code must define solve(params)")
    if "import math" not in code:
        code = "import math\n" + code
    return code


def mock_template(idx: int, profile: dict[str, Any]) -> E2Template:
    del idx, profile
    return E2Template(
        template_id="mock_e2_template",
        family="long_numeric_grid_template",
        difficulty="hard",
        domain_shell="fluid_stability",
        problem_setup_template=(
            "A reduced stability notebook records a neutral curve over a finite grid. The pressure "
            "normalization, wall thermistor calibration, and two discarded viscosity notes are included "
            "in the log but are irrelevant. Use a={a}, b={b}, c={c}, d={d}, u0={u0}, u1={u1}, v1={v1}, "
            "and candidate wavenumbers ks={ks}. The surrogate curve is R(k)=((k^2+a)^3)/(b*k^2)+c*k+d/(1+k^2), "
            "and the secondary diagnostic is ratio(k)=(u0+u1*k^2)/(1+v1*k)."
        ),
        main_problem_template=(
            "Evaluate only the listed candidates. Return the minimum R(k) rounded to six decimals, "
            "the winning k, and the diagnostic ratio at the winning k rounded to six decimals."
        ),
        code_template="def answer():\n    Ra_c = ...\n    k_c = ...\n    ratio_w2T = ...\n    return Ra_c, k_c, ratio_w2T",
        solver_code="""
import math

def solve(params):
    ks = params["ks"]
    values = [((k*k + params["a"])**3)/(params["b"]*k*k) + params["c"]*k + params["d"]/(1+k*k) for k in ks]
    best = min(range(len(values)), key=lambda i: values[i])
    k = ks[best]
    return {
        "Ra_c": round(values[best], 6),
        "k_c": k,
        "ratio_w2T": round((params["u0"] + params["u1"]*k*k)/(1 + params["v1"]*k), 6),
    }
""".strip(),
        sampler_code="""
def sample_params(rng, idx):
    base = 4 + (idx % 5)
    return {
        "a": base,
        "b": rng.choice([2, 3, 4]),
        "c": rng.randint(1, 5),
        "d": rng.randint(2, 8),
        "u0": round(rng.uniform(0.8, 2.2), 2),
        "u1": round(rng.uniform(0.1, 0.6), 2),
        "v1": round(rng.uniform(0.2, 0.8), 2),
        "ks": [round(0.6 + 0.35*i + 0.02*(idx % 3), 2) for i in range(6)],
    }
""".strip(),
        return_keys=["Ra_c", "k_c", "ratio_w2T"],
        answer_type="numeric_tuple",
        verifier_mode="numeric_sequence",
        solution_trace_template="Scan ks={ks}, choose the minimum neutral curve, and evaluate the ratio at that candidate.",
        complexity_notes="Finite-grid optimization with distractor physical bookkeeping and a secondary diagnostic.",
    )
