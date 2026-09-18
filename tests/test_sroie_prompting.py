"""Regression tests for SROIE benchmark prompt construction."""

import json

import pytest

from pixels_to_pairs.prompting.sroie import (
    SROIE_EXAMPLE_IDS,
    SYSTEM_MESSAGE,
    USER_INSTRUCTIONS_BASE,
    USER_INSTRUCTIONS_FEWSHOT,
    build_prompt,
    load_sroie_examples,
)


EXAMPLES = [
    {
        "doc_id": "X51008142065",
        "doc_text": "EXAMPLE DOCUMENT ONE",
        "kvp": [{"key": "company", "value": "Store One"}],
    },
    {
        "doc_id": "X51008164525",
        "doc_text": "EXAMPLE DOCUMENT TWO",
        "kvp": [{"key": "total", "value": "20.00"}],
    },
    {
        "doc_id": "X51008030566",
        "doc_text": "EXAMPLE DOCUMENT THREE",
        "kvp": [{"key": "date", "value": "01/01/2025"}],
    },
]


def test_system_message_preserves_output_schema():
    assert (
        'Return ONLY a JSON object with the exact schema '
        '{"kvp": [{"key":..., "value":...}, ...]}.'
        in SYSTEM_MESSAGE
    )


def test_base_prompt_preserves_sroie_target_fields():
    assert "company" in USER_INSTRUCTIONS_BASE
    assert "address" in USER_INSTRUCTIONS_BASE
    assert "date" in USER_INSTRUCTIONS_BASE
    assert "total" in USER_INSTRUCTIONS_BASE

    assert (
        '["company", "address", "date", "total"]'
        in USER_INSTRUCTIONS_BASE
    )


def test_zero_shot_contains_no_example_blocks():
    prompt = build_prompt("TARGET RECEIPT", "0shot")

    assert USER_INSTRUCTIONS_BASE in prompt
    assert USER_INSTRUCTIONS_FEWSHOT not in prompt

    for idx in range(1, 4):
        assert f"<EXAMPLE_{idx}_DOCUMENT>" not in prompt
        assert f"<EXAMPLE_{idx}_OUTPUT>" not in prompt

    assert (
        "<NEW_DOCUMENT>\nTARGET RECEIPT\n</NEW_DOCUMENT>"
        in prompt
    )


@pytest.mark.parametrize(
    ("shot_variant", "expected_shots"),
    [
        ("1shot", 1),
        ("2shot", 2),
        ("3shot", 3),
    ],
)
def test_few_shot_prompts_use_cumulative_examples(
    shot_variant,
    expected_shots,
):
    prompt = build_prompt(
        "TARGET RECEIPT",
        shot_variant,
        examples=EXAMPLES,
    )

    assert USER_INSTRUCTIONS_FEWSHOT in prompt

    for idx in range(1, 4):
        document_tag = f"<EXAMPLE_{idx}_DOCUMENT>"
        output_tag = f"<EXAMPLE_{idx}_OUTPUT>"

        if idx <= expected_shots:
            assert document_tag in prompt
            assert output_tag in prompt
        else:
            assert document_tag not in prompt
            assert output_tag not in prompt


def test_few_shot_examples_preserve_frozen_order():
    prompt = build_prompt(
        "TARGET RECEIPT",
        "3shot",
        examples=EXAMPLES,
    )

    assert (
        prompt.index("EXAMPLE DOCUMENT ONE")
        < prompt.index("EXAMPLE DOCUMENT TWO")
        < prompt.index("EXAMPLE DOCUMENT THREE")
        < prompt.index("TARGET RECEIPT")
    )


def test_few_shot_prompt_contains_anti_anchoring_rules():
    prompt = build_prompt(
        "TARGET RECEIPT",
        "1shot",
        examples=EXAMPLES,
    )

    assert (
        "NEVER copy example values into the new output."
        in prompt
    )
    assert (
        "For the new receipt, extract fields ONLY from "
        "the text inside <NEW_DOCUMENT>."
        in prompt
    )


def test_frozen_example_ids():
    assert SROIE_EXAMPLE_IDS == (
        "X51008142065",
        "X51008164525",
        "X51008030566",
    )


def test_few_shot_requires_examples_or_directory():
    with pytest.raises(
        ValueError,
        match="example_files_dir is required",
    ):
        build_prompt("TARGET RECEIPT", "1shot")


def test_unknown_shot_variant_raises_error():
    with pytest.raises(
        ValueError,
        match="Unknown shot_variant",
    ):
        build_prompt(
            "TARGET RECEIPT",
            "4shot",
            examples=EXAMPLES,
        )


def _write_example_files(
    directory,
    doc_id,
    doc_text="RECEIPT TEXT",
    kvp=None,
):
    if kvp is None:
        kvp = [
            {
                "key": "company",
                "value": "Example Store",
            }
        ]

    (directory / f"{doc_id}.txt").write_text(
        doc_text,
        encoding="utf-8",
    )
    (directory / f"{doc_id}_kvp.json").write_text(
        json.dumps(kvp),
        encoding="utf-8",
    )


def test_load_sroie_examples_from_files(tmp_path):
    for idx, doc_id in enumerate(
        SROIE_EXAMPLE_IDS,
        start=1,
    ):
        _write_example_files(
            tmp_path,
            doc_id,
            doc_text=f"RECEIPT {idx}",
            kvp=[
                {
                    "key": "company",
                    "value": f"Store {idx}",
                }
            ],
        )

    examples = load_sroie_examples(tmp_path)

    assert [
        example["doc_id"]
        for example in examples
    ] == list(SROIE_EXAMPLE_IDS)

    assert examples[0]["doc_text"] == "RECEIPT 1"
    assert examples[1]["doc_text"] == "RECEIPT 2"
    assert examples[2]["doc_text"] == "RECEIPT 3"

    assert examples[0]["kvp"] == [
        {
            "key": "company",
            "value": "Store 1",
        }
    ]


def test_loader_accepts_wrapped_kvp_object(tmp_path):
    doc_id = SROIE_EXAMPLE_IDS[0]

    (tmp_path / f"{doc_id}.txt").write_text(
        "RECEIPT TEXT",
        encoding="utf-8",
    )
    (tmp_path / f"{doc_id}_kvp.json").write_text(
        json.dumps(
            {
                "kvp": [
                    {
                        "key": "total",
                        "value": "25.00",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # Test the individual loader indirectly by completing the
    # remaining frozen examples.
    for other_id in SROIE_EXAMPLE_IDS[1:]:
        _write_example_files(
            tmp_path,
            other_id,
        )

    examples = load_sroie_examples(tmp_path)

    assert examples[0]["kvp"] == [
        {
            "key": "total",
            "value": "25.00",
        }
    ]


def test_missing_example_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sroie_examples(tmp_path)


@pytest.mark.parametrize(
    "invalid_kvp",
    [
        {"unexpected": []},
        {"kvp": "not-a-list"},
        [{"key": "company"}],
        [{"key": "company", "value": 123}],
        [{"key": "", "value": "Store"}],
        [{"key": "company", "value": ""}],
        [],
    ],
)
def test_invalid_example_kvp_is_rejected(
    tmp_path,
    invalid_kvp,
):
    first_id = SROIE_EXAMPLE_IDS[0]

    (tmp_path / f"{first_id}.txt").write_text(
        "RECEIPT TEXT",
        encoding="utf-8",
    )
    (tmp_path / f"{first_id}_kvp.json").write_text(
        json.dumps(invalid_kvp),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_sroie_examples(tmp_path)


def test_invalid_json_is_rejected(tmp_path):
    first_id = SROIE_EXAMPLE_IDS[0]

    (tmp_path / f"{first_id}.txt").write_text(
        "RECEIPT TEXT",
        encoding="utf-8",
    )
    (tmp_path / f"{first_id}_kvp.json").write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="not valid JSON",
    ):
        load_sroie_examples(tmp_path)


def test_empty_document_text_is_rejected(tmp_path):
    first_id = SROIE_EXAMPLE_IDS[0]

    (tmp_path / f"{first_id}.txt").write_text(
        "   ",
        encoding="utf-8",
    )
    (tmp_path / f"{first_id}_kvp.json").write_text(
        json.dumps(
            [
                {
                    "key": "company",
                    "value": "Store",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="document text is empty",
    ):
        load_sroie_examples(tmp_path)
