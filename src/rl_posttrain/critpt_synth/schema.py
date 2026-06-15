from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SyntheticCritPTExample:
    problem_id: str
    prompt: str
    code_template: str
    target_code: str
    verifier: dict[str, Any]
    split: str = "train"
    family: str = "template"
    difficulty: str = "easy"
    solution_trace: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "SyntheticCritPTExample":
        return SyntheticCritPTExample(
            problem_id=str(raw["problem_id"]),
            prompt=str(raw["prompt"]),
            code_template=str(raw["code_template"]),
            target_code=str(raw["target_code"]),
            verifier=dict(raw["verifier"]),
            split=str(raw.get("split", "train")),
            family=str(raw.get("family", "template")),
            difficulty=str(raw.get("difficulty", "easy")),
            solution_trace=str(raw.get("solution_trace", "")),
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "split": self.split,
            "family": self.family,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "code_template": self.code_template,
            "target_code": self.target_code,
            "solution_trace": self.solution_trace,
            "verifier": self.verifier,
            "metadata": self.metadata,
        }

    def assistant_code_block(self) -> str:
        return f"```python\n{self.target_code.strip()}\n```"

    def to_sft_row(self) -> dict[str, Any]:
        return {
            "id": self.problem_id,
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.assistant_code_block()},
            ],
            "metadata": {
                "split": self.split,
                "family": self.family,
                "difficulty": self.difficulty,
                **self.metadata,
            },
        }
