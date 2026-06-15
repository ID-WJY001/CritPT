from __future__ import annotations

import ast
import json
import re
from typing import Any

from rl_posttrain.critpt_synth.verifier import extract_python_code
from rl_posttrain.critpt_synth.verl_reward_v20_focused_hard import compute_score as compute_v20_score


NONCANONICAL_POWER_RE = re.compile(r"tr\([^)]*\)\s*\^\s*\d+")


def _loads_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _expected_operator_count(verifier: dict[str, Any]) -> int:
    checks = verifier.get("checks", [])
    if not isinstance(checks, list) or not checks:
        return 0
    expected = checks[0].get("expected") if isinstance(checks[0], dict) else None
    if not isinstance(expected, list):
        return 0
    labels = [item for item in expected if isinstance(item, str) and "tr(" in item]
    return len(labels)


def _operator_string_literal_count(solution_str: str) -> int:
    code = extract_python_code(solution_str)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return len(re.findall(r"tr\(", solution_str))
    labels: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "tr(" in node.value:
            labels.add(node.value)
    return len(labels)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: object,
) -> dict[str, object]:
    """V21 reward: keep V20 correctness, tighten operator over-enumeration.

    V20 visible rollouts were already format-clean but often failed operator
    tasks by dumping a superset of plausible labels. This reward keeps exact
    verifier correctness dominant while capping non-correct outputs that show
    obvious over-enumeration or non-canonical power notation.
    """

    result = compute_v20_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    extra_info = extra_info or {}
    metadata = _loads_dict(extra_info.get("metadata", {}))
    verifier = _loads_dict(extra_info.get("code_verifier") or extra_info.get("verifier") or {})
    focus = str(metadata.get("v20_focus", ""))
    acc = float(result.get("acc", 0.0)) >= 1.0
    score = float(result.get("score", 0.0))

    result["v21_operator_expected_count"] = 0.0
    result["v21_operator_literal_count"] = 0.0
    result["v21_operator_overenumeration_ratio"] = 0.0
    result["v21_operator_too_many_literals"] = 0.0
    result["v21_operator_noncanonical_power"] = 0.0

    if focus == "operator_canonical_label":
        expected_count = _expected_operator_count(verifier)
        literal_count = _operator_string_literal_count(solution_str)
        over_ratio = literal_count / max(expected_count, 1)
        too_many = expected_count > 0 and literal_count > max(expected_count * 1.6, expected_count + 8)
        noncanonical_power = bool(NONCANONICAL_POWER_RE.search(solution_str))

        result["v21_operator_expected_count"] = float(expected_count)
        result["v21_operator_literal_count"] = float(literal_count)
        result["v21_operator_overenumeration_ratio"] = float(over_ratio)
        result["v21_operator_too_many_literals"] = 1.0 if too_many else 0.0
        result["v21_operator_noncanonical_power"] = 1.0 if noncanonical_power else 0.0

        if not acc and noncanonical_power:
            score = min(score, 0.34)
        if not acc and too_many:
            score = min(score, 0.32)
        if not acc and literal_count >= 80:
            score = min(score, 0.22)

    result["score"] = max(0.0, min(1.0, score))
    result["v21_operator_precision_reward"] = result["score"]
    return result
