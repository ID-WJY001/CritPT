from __future__ import annotations

import json
import re

from rl_posttrain.critpt_synth.verl_reward_v19_failure_mined import compute_score as compute_v19_score


CONCAT_OPERATOR_RE = re.compile(r"\b(?:psi|dpsi|chi|F|B){2,}\b")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: object,
) -> dict[str, object]:
    """V20 focused reward for current visible semantic failures."""

    result = compute_v19_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    extra_info = extra_info or {}
    metadata_raw = extra_info.get("metadata", {})
    if isinstance(metadata_raw, str):
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            metadata = {}
    elif isinstance(metadata_raw, dict):
        metadata = metadata_raw
    else:
        metadata = {}

    focus = str(metadata.get("v20_focus", ""))
    acc = float(result.get("acc", 0.0)) >= 1.0
    score = float(result.get("score", 0.0))
    lower_solution = solution_str.lower()

    result["v20_operator_has_tr_surface"] = 0.0
    result["v20_operator_concat_near_miss"] = 0.0
    result["v20_empty_filter_mentions_empty_set"] = 0.0
    result["v20_hhg_runtime_error"] = 0.0

    if focus == "operator_canonical_label":
        has_tr_surface = "tr(" in solution_str
        has_concat_near_miss = bool(CONCAT_OPERATOR_RE.search(solution_str))
        result["v20_operator_has_tr_surface"] = 1.0 if has_tr_surface else 0.0
        result["v20_operator_concat_near_miss"] = 1.0 if has_concat_near_miss else 0.0
        if not acc and not has_tr_surface:
            score = min(score, 0.34)
        if not acc and has_concat_near_miss:
            score = min(score, 0.42)
    elif focus == "empty_interval_filter":
        mentions_empty = "set()" in lower_solution or "return {}" in lower_solution
        result["v20_empty_filter_mentions_empty_set"] = 1.0 if mentions_empty else 0.0
        if not acc and "expected=set()" in str(result.get("reason", "")):
            score = min(score, 0.38)
    elif focus == "hhg_oam_runtime_safe":
        runtime_error = str(result.get("reason", "")).startswith("runtime_error")
        result["v20_hhg_runtime_error"] = 1.0 if runtime_error else 0.0
        if runtime_error:
            score = min(score, 0.28)

    result["score"] = max(0.0, min(1.0, score))
    result["v20_focused_hard_reward"] = result["score"]
    return result
