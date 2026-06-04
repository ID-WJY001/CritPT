from __future__ import annotations

from rl_posttrain.critpt.schema import CritPTExample, VerifierSpec


def seed_examples() -> list[CritPTExample]:
    prompt = (
        "A toy post-selection code accepts an output with probability "
        "1 - 2*p + 2*p**2. Its logical success numerator is (1-p)**2. "
        "Derive the conditional logical fidelity as a function of p. "
        "Put only the final expression inside <answer>...</answer>."
    )
    verifier = VerifierSpec(
        kind="symbolic",
        expected="(1-p)**2 / (1 - 2*p + 2*p**2)",
        variables=["p"],
        numeric_tests=[{"p": 0.01}, {"p": 0.1}, {"p": 0.23}],
        tolerance=1e-8,
    )
    return [
        CritPTExample(
            problem_id="toy_postselection_fidelity_001",
            prompt=prompt,
            answer="<answer>(1-p)**2/(1-2*p+2*p**2)</answer>",
            verifier=verifier,
            split="train",
            metadata={"domain": "toy_qec", "source": "synthetic_seed"},
        )
    ]

