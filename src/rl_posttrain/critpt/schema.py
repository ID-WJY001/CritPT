from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VerifierSpec:
    kind: str
    expected: str
    variables: list[str] = field(default_factory=list)
    numeric_tests: list[dict[str, float]] = field(default_factory=list)
    tolerance: float = 1e-8

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "VerifierSpec":
        return VerifierSpec(
            kind=str(raw.get("kind", "symbolic")),
            expected=str(raw["expected"]),
            variables=[str(v) for v in raw.get("variables", [])],
            numeric_tests=[dict(t) for t in raw.get("numeric_tests", [])],
            tolerance=float(raw.get("tolerance", 1e-8)),
        )


@dataclass(frozen=True)
class CritPTExample:
    problem_id: str
    prompt: str
    answer: str
    verifier: VerifierSpec
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "CritPTExample":
        return CritPTExample(
            problem_id=str(raw["problem_id"]),
            prompt=str(raw["prompt"]),
            answer=str(raw["answer"]),
            verifier=VerifierSpec.from_dict(raw["verifier"]),
            split=str(raw.get("split", "train")),
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "prompt": self.prompt,
            "answer": self.answer,
            "split": self.split,
            "metadata": self.metadata,
            "verifier": {
                "kind": self.verifier.kind,
                "expected": self.verifier.expected,
                "variables": self.verifier.variables,
                "numeric_tests": self.verifier.numeric_tests,
                "tolerance": self.verifier.tolerance,
            },
        }

