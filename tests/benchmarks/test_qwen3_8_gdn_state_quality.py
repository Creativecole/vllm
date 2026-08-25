# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.accuracy.qwen3_8_gdn_state_quality import (
    build_parity_prompts,
    compare_gsm8k,
    compare_parity,
    extract_numeric_answer,
)


def test_parity_prompt_set_is_fixed_and_varied() -> None:
    prompts = build_parity_prompts()

    assert len(prompts) == 96
    assert len({row["id"] for row in prompts}) == 96
    assert len({row["domain"] for row in prompts}) == 16
    prompt_lengths = [len(str(row["prompt"])) for row in prompts]
    assert max(prompt_lengths) > 8 * min(prompt_lengths)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("reasoning\n#### 1,234", "1234"),
        ("The final answer is -2.50.", "-2.5"),
        ("work 17\n#### no numeric answer", None),
        ("no numeric answer", None),
    ],
)
def test_extract_numeric_answer(text: str, expected: str | None) -> None:
    assert extract_numeric_answer(text) == expected


def test_compare_parity_reports_token_metrics() -> None:
    settings = {"prompt_set_sha256": "same", "seed": 0}
    fp32_task = {
        "settings": settings,
        "examples": [
            {"id": "a", "output_token_ids": [1, 2, 3], "invalid_reason": None},
            {"id": "b", "output_token_ids": [4, 5, 6], "invalid_reason": None},
        ],
        "aggregate": {
            "obviously_invalid_failure_count": 0,
            "non_finite_failure_count": 0,
        },
    }
    fp8_task = {
        "settings": settings,
        "examples": [
            {"id": "a", "output_token_ids": [1, 2, 3], "invalid_reason": None},
            {"id": "b", "output_token_ids": [4, 9, 6], "invalid_reason": None},
        ],
        "aggregate": {
            "obviously_invalid_failure_count": 0,
            "non_finite_failure_count": 0,
        },
    }

    aggregate = compare_parity(fp32_task, fp8_task)["aggregate"]

    assert aggregate["fully_token_identical_count"] == 1
    assert aggregate["fully_token_identical_percentage"] == 50.0
    assert aggregate["total_aligned_token_agreement_rate"] == pytest.approx(5 / 6)
    assert aggregate["first_divergence_position_zero_based"] == {
        "mean": 1,
        "median": 1,
        "p10": None,
    }


def test_compare_gsm8k_reports_paired_accuracy() -> None:
    settings = {"question_set_sha256": "same", "seed": 0}
    fp32_task = {
        "settings": settings,
        "examples": [
            {
                "id": "a",
                "question": "one",
                "gold_answer": "1",
                "extracted_answer": "1",
                "correct": True,
            },
            {
                "id": "b",
                "question": "two",
                "gold_answer": "2",
                "extracted_answer": "0",
                "correct": False,
            },
        ],
        "aggregate": {"exact_match_accuracy": 0.5},
    }
    fp8_task = {
        "settings": settings,
        "examples": [
            {
                "id": "a",
                "question": "one",
                "gold_answer": "1",
                "extracted_answer": "0",
                "correct": False,
            },
            {
                "id": "b",
                "question": "two",
                "gold_answer": "2",
                "extracted_answer": "2",
                "correct": True,
            },
        ],
        "aggregate": {"exact_match_accuracy": 0.5},
    }

    aggregate = compare_gsm8k(fp32_task, fp8_task)["aggregate"]

    assert aggregate == {
        "num_questions": 2,
        "fp32_exact_match_accuracy": 0.5,
        "fp8_exact_match_accuracy": 0.5,
        "accuracy_difference_percentage_points_fp8_minus_fp32": 0.0,
        "absolute_accuracy_difference_percentage_points": 0.0,
        "correctness_differs_count": 2,
    }
