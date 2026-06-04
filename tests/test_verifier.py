from rl_posttrain.critpt.schema import VerifierSpec
from rl_posttrain.critpt.verifier import extract_answer, verify_completion


def test_extract_answer_tags() -> None:
    assert extract_answer("reasoning\n<answer>(1-p)^2</answer>") == "(1-p)^2"


def test_symbolic_equivalence() -> None:
    spec = VerifierSpec(
        kind="symbolic",
        expected="(1-p)**2/(1-2*p+2*p**2)",
        variables=["p"],
        numeric_tests=[{"p": 0.01}, {"p": 0.11}],
    )
    result = verify_completion("<answer>((p-1)**2)/(2*p**2 - 2*p + 1)</answer>", spec)
    assert result.ok


def test_symbolic_rejects_wrong_answer() -> None:
    spec = VerifierSpec(kind="symbolic", expected="p + 1", variables=["p"])
    result = verify_completion("<answer>p + 2</answer>", spec)
    assert not result.ok

