from __future__ import annotations

import importlib.util
from pathlib import Path

from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample


ROOT = Path(__file__).resolve().parents[1]
EXPORTER_PATH = ROOT / "scripts/data/export_synthetic_semantic_judge_verl_parquet.py"
spec = importlib.util.spec_from_file_location("semantic_exporter", EXPORTER_PATH)
assert spec and spec.loader
semantic_exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(semantic_exporter)


def _example() -> SyntheticCritPTExample:
    return SyntheticCritPTExample(
        problem_id="unit_operator",
        prompt="Problem setup\n\n### Main problem\nReturn labels.\n\n### Parsing template:\ndef answer(): ...",
        code_template="def answer(): ...",
        target_code='def answer():\n    return ["tr(psi)", "tr(psi^2)"]',
        verifier={
            "checks": [
                {
                    "mode": "exact_sequence",
                    "expected": ["tr(psi)", "tr(psi^2)"],
                }
            ]
        },
        split="train",
        family="v20_operator_canonical_labels",
        difficulty="hard",
        solution_trace="Apply the charge filter and sort canonical labels.",
        metadata={
            "answer_type": "ordered_string_list",
            "v20_focus": "operator_canonical_label",
        },
    )


def test_semantic_exporter_does_not_export_verifier() -> None:
    row = semantic_exporter.to_verl_row(_example(), 0, "code")
    extra = row["extra_info"]

    assert row["data_source"] == "critpt_semantic_judge"
    assert row["reward_model"]["style"] == "llm_semantic_judge"
    assert "code_verifier" not in extra
    assert "verifier" not in extra


def test_semantic_exporter_includes_judge_ready_reference_and_rubric() -> None:
    row = semantic_exporter.to_verl_row(_example(), 0, "code")
    extra = row["extra_info"]

    assert "tr(psi)" in extra["reference_answer"]
    assert "operator" in extra["rubric"].lower()
    assert "candidate superset" in extra["rubric"].lower()
