from rl_posttrain.critpt_synth.final_answer_verifier import verify_final_answer


def test_final_answer_accepts_numeric_sequence() -> None:
    verifier = {
        "checks": [
            {
                "mode": "numeric_sequence",
                "expected": [0.8372, 0.6977, 0.1395],
                "tolerance": 1e-4,
            }
        ]
    }

    result = verify_final_answer(
        "前两项相除后得到第三项。\n最终答案：(0.83720, 0.69770, 0.13950)",
        verifier,
    )

    assert result.ok
    assert result.score == 1.0
    assert result.answer_marker_present


def test_final_answer_rejects_wrong_recurrence_value() -> None:
    verifier = {"checks": [{"mode": "exact", "expected": 154451}]}

    result = verify_final_answer("简短递推。\n最终答案：114985", verifier)

    assert not result.ok
    assert result.score < 0.5
    assert "expected=154451" in result.reason


def test_final_answer_accepts_symbolic_latex_unit() -> None:
    verifier = {
        "checks": [
            {
                "mode": "symbolic",
                "expected": "5*drive/28",
                "variables": ["drive"],
                "tolerance": 1e-8,
            }
        ]
    }

    result = verify_final_answer(
        r"代入化简。\n最终答案：$x=\frac{5}{28}\Omega$",
        verifier,
    )

    assert result.ok


def test_final_answer_accepts_common_symbolic_latex() -> None:
    gap_verifier = {
        "checks": [
            {
                "mode": "symbolic",
                "expected": "sqrt(delta**2 + 36*g**2)",
                "variables": ["delta", "g"],
                "tolerance": 1e-8,
            }
        ]
    }
    response_verifier = {
        "checks": [
            {
                "mode": "symbolic",
                "expected": "1/((6**2 - omega**2)**2 + (4*omega)**2)",
                "variables": ["omega"],
                "tolerance": 1e-8,
            }
        ]
    }

    assert verify_final_answer(r"最终答案：$\sqrt{36g^2 + \delta^2}$", gap_verifier).ok
    assert verify_final_answer(
        r"最终答案：$ |A(\omega)|^2 = \frac{1}{(36 - \omega^2)^2 + 16\omega^2} $",
        response_verifier,
    ).ok


def test_final_answer_accepts_unicode_symbolic_math() -> None:
    verifier = {
        "checks": [
            {
                "mode": "symbolic",
                "expected": "sqrt(delta**2 + 16*g**2)",
                "variables": ["delta", "g"],
                "tolerance": 1e-8,
            }
        ]
    }

    result = verify_final_answer("最终答案：√(δ² + 16g²)", verifier)

    assert result.ok


def test_final_answer_accepts_boxed_dfrac_symbolic_math() -> None:
    verifier = {
        "checks": [
            {
                "mode": "symbolic",
                "expected": "15/(8*alpha**3)",
                "variables": ["alpha"],
                "tolerance": 1e-8,
            }
        ]
    }

    result = verify_final_answer(r"最终答案：$\boxed{\dfrac{15}{8\alpha^3}}$", verifier)

    assert result.ok


def test_final_answer_accepts_nested_latex_sqrt_gap() -> None:
    verifier = {
        "checks": [
            {
                "mode": "symbolic",
                "expected": "sqrt(delta**2 + 16*g**2)",
                "variables": ["delta", "g"],
                "tolerance": 1e-8,
            }
        ]
    }

    result = verify_final_answer(
        r"最终答案：$ 2 \sqrt{ \left( \frac{\delta}{2} \right)^2 + (2g)^2 } $",
        verifier,
    )

    assert result.ok


def test_final_answer_accepts_fraction_items_in_numeric_sequence() -> None:
    verifier = {
        "checks": [
            {
                "mode": "numeric_sequence",
                "expected": [1, 4, 8.5, 13, 16.375],
                "tolerance": 1e-8,
            }
        ]
    }

    result = verify_final_answer("最终答案：[1, 4, 17/2, 13, 131/8]", verifier)

    assert result.ok


def test_final_answer_accepts_inline_code_literal() -> None:
    verifier = {"checks": [{"mode": "exact", "expected": {"$tuple": [-17, 1]}}]}

    result = verify_final_answer("最终答案：`(-17, 1)`", verifier)

    assert result.ok
    assert result.extracted_answer == "`(-17, 1)`"


def test_final_answer_sequence_mismatch_does_not_get_full_score() -> None:
    verifier = {
        "checks": [{"mode": "numeric_sequence", "expected": [1, 11, 73, 401, 2059], "tolerance": 1e-8}]
    }

    result = verify_final_answer("最终答案：[1, 11, 73, 301, 2059]", verifier)

    assert not result.ok
    assert result.score < 0.5


def test_final_answer_sequence_item_checks_allow_partial_credit() -> None:
    verifier = {
        "checks": [
            {"mode": "sequence_length", "expected": 4},
            {"mode": "numeric_sequence_item", "index": 0, "expected": 0.2, "tolerance": 1e-8},
            {"mode": "numeric_sequence_item", "index": 1, "expected": 0.34, "tolerance": 1e-8},
            {"mode": "numeric_sequence_item", "index": -1, "expected": 0.418, "tolerance": 1e-8},
        ]
    }

    good = verify_final_answer("最终答案：[0.2, 0.34, 0.398, 0.418]", verifier)
    bad_tail = verify_final_answer("最终答案：[0.2, 0.34, 0.398, 0.400]", verifier)

    assert good.ok
    assert not bad_tail.ok
    assert bad_tail.passed_checks == 3
    assert bad_tail.score > 0.5


def test_final_answer_sequence_item_checks_score_all_items_after_early_failure() -> None:
    verifier = {
        "checks": [
            {"mode": "sequence_length", "expected": 4},
            {"mode": "numeric_sequence_item", "index": 0, "expected": 1, "tolerance": 1e-8},
            {"mode": "numeric_sequence_item", "index": 1, "expected": 2, "tolerance": 1e-8},
            {"mode": "numeric_sequence_item", "index": 2, "expected": 3, "tolerance": 1e-8},
            {"mode": "numeric_sequence_item", "index": 3, "expected": 4, "tolerance": 1e-8},
        ]
    }

    result = verify_final_answer("最终答案：[99, 2, 3, 4]", verifier)

    assert not result.ok
    assert result.passed_checks == 4
    assert result.score > 0.65
    assert "check_1_failed" in result.reason


def test_final_answer_uses_reward_checks_for_tagged_trace() -> None:
    verifier = {
        "checks": [
            {"mode": "sequence_length", "expected": 3},
            {"mode": "numeric_sequence_item", "index": 0, "expected": 10},
            {"mode": "numeric_sequence_item", "index": 1, "expected": 2},
            {"mode": "numeric_sequence_item", "index": 2, "expected": 7},
        ],
        "reward_checks": [
            {"mode": "text_numeric", "tag": "trace", "expected": 10, "tolerance": 1e-8},
            {"mode": "text_numeric", "tag": "T2_01", "expected": 2, "tolerance": 1e-8},
            {"mode": "text_numeric", "tag": "Tn_10", "expected": 7, "tolerance": 1e-8},
            {"mode": "sequence_length", "expected": 3},
            {"mode": "numeric_sequence_item", "index": 0, "expected": 10, "tolerance": 1e-8},
            {"mode": "numeric_sequence_item", "index": 1, "expected": 2, "tolerance": 1e-8},
            {"mode": "numeric_sequence_item", "index": 2, "expected": 7, "tolerance": 1e-8},
        ],
    }

    result = verify_final_answer("审计：trace=10, T2_01=2, Tn_10=7\n最终答案：[10, 2, 7]", verifier)

    assert result.ok
    assert result.passed_checks == 7
    assert result.total_checks == 7


def test_final_answer_tagged_trace_missing_audit_is_not_full_credit() -> None:
    verifier = {
        "checks": [{"mode": "numeric_sequence", "expected": [10, 2, 7], "tolerance": 1e-8}],
        "reward_checks": [
            {"mode": "text_numeric", "tag": "trace", "expected": 10, "tolerance": 1e-8},
            {"mode": "sequence_length", "expected": 3},
            {"mode": "numeric_sequence_item", "index": 0, "expected": 10, "tolerance": 1e-8},
            {"mode": "numeric_sequence_item", "index": 1, "expected": 2, "tolerance": 1e-8},
            {"mode": "numeric_sequence_item", "index": 2, "expected": 7, "tolerance": 1e-8},
        ],
    }

    result = verify_final_answer("最终答案：[10, 2, 7]", verifier)

    assert not result.ok
    assert result.score < 1.0
    assert "tag_trace" in result.reason


def test_final_answer_no_checks_is_not_rewarded() -> None:
    result = verify_final_answer("最终答案：42", {"checks": []})

    assert not result.ok
    assert result.score == 0.0
    assert result.reason == "no_verifier_checks"


def test_final_answer_caps_skip_phrase_score() -> None:
    verifier = {"checks": [{"mode": "exact", "expected": 154451}]}

    result = verify_final_answer("继续递推可得。\n最终答案：154451", verifier)

    assert result.ok
    assert result.skip_phrase_present
    assert result.score <= 0.65


def test_final_answer_without_marker_gets_lower_score() -> None:
    verifier = {"checks": [{"mode": "exact", "expected": 42}]}

    result = verify_final_answer("42", verifier)

    assert result.ok
    assert result.score == 0.82
    assert not result.answer_marker_present
