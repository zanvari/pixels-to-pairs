"""Regression tests for CORD benchmark prompt construction."""

import json

import pytest

from pixels_to_pairs.prompting.cord import (
    CORD_EXAMPLE_FILES,
    SYSTEM_MESSAGE,
    USER_INSTRUCTIONS_BASE,
    USER_INSTRUCTIONS_FEWSHOT,
    build_prompt,
    load_cord_examples,
)


EXAMPLES = {
    "ex1": {
        "doc": "EXAMPLE DOCUMENT ONE",
        "json": '{"kvp": [{"key": "TOTAL", "value": "10,000"}]}',
    },
    "ex2": {
        "doc": "EXAMPLE DOCUMENT TWO",
        "json": '{"kvp": [{"key": "SUB TOTAL", "value": "20,000"}]}',
    },
    "ex3": {
        "doc": "EXAMPLE DOCUMENT THREE",
        "json": '{"kvp": [{"key": "CHANGE DUE", "value": "5,000"}]}',
    },
}


def test_system_message_preserves_output_schema():
    assert (
        'Return ONLY a JSON object with the exact schema '
        '{"kvp": [{"key":..., "value":...}, ...]}.'
        in SYSTEM_MESSAGE
    )


def test_base_prompt_preserves_anti_schema_drift_rules():
    assert "SUB TOTAL" in USER_INSTRUCTIONS_BASE
    assert "CHANGE DUE" in USER_INSTRUCTIONS_BASE
    assert "x1, x2, 2x" in USER_INSTRUCTIONS_BASE
    assert (
        "Do NOT use pure numbers or monetary amounts as keys."
        in USER_INSTRUCTIONS_BASE
    )
    assert 'key = "ITEM_NAME"' in USER_INSTRUCTIONS_BASE
    assert 'value = "18,000"' in USER_INSTRUCTIONS_BASE


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


def test_few_shot_examples_preserve_order():
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

    assert "NEVER copy example values." in prompt
    assert (
        "Extract pairs ONLY from <NEW_DOCUMENT>."
        in prompt
    )
    assert (
        "Ignore all text inside any <EXAMPLE_*> blocks."
        in prompt
    )


def test_frozen_example_filenames():
    assert CORD_EXAMPLE_FILES == (
        ("receipt_00730.txt", "receipt_00730_kvp.json"),
        ("receipt_00760.txt", "receipt_00760_kvp.json"),
        ("receipt_00710.txt", "receipt_00710_kvp.json"),
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


def test_load_cord_examples_from_files(tmp_path):
    example_data = [
        (
            "receipt_00730.txt",
            "receipt_00730_kvp.json",
            "CORD RECEIPT ONE",
            [{"key": "TOTAL", "value": "10,000"}],
        ),
        (
            "receipt_00760.txt",
            "receipt_00760_kvp.json",
            "CORD RECEIPT TWO",
            [{"key": "SUB TOTAL", "value": "20,000"}],
        ),
        (
            "receipt_00710.txt",
            "receipt_00710_kvp.json",
            "CORD RECEIPT THREE",
            [{"key": "CHANGE DUE", "value": "5,000"}],
        ),
    ]

    for txt_name, json_name, doc_text, kvp in example_data:
        (tmp_path / txt_name).write_text(
            doc_text,
            encoding="utf-8",
        )
        (tmp_path / json_name).write_text(
            json.dumps(kvp),
            encoding="utf-8",
        )

    examples = load_cord_examples(tmp_path)

    assert list(examples) == ["ex1", "ex2", "ex3"]

    assert examples["ex1"]["doc"] == "CORD RECEIPT ONE"
    assert examples["ex2"]["doc"] == "CORD RECEIPT TWO"
    assert examples["ex3"]["doc"] == "CORD RECEIPT THREE"

    assert json.loads(examples["ex1"]["json"]) == {
        "kvp": [
            {
                "key": "TOTAL",
                "value": "10,000",
            }
        ]
    }


def test_loader_accepts_wrapped_kvp_object(tmp_path):
    data = [
        (
            "receipt_00730.txt",
            "receipt_00730_kvp.json",
            "DOC ONE",
            {"kvp": [{"key": "TOTAL", "value": "10"}]},
        ),
        (
            "receipt_00760.txt",
            "receipt_00760_kvp.json",
            "DOC TWO",
            [{"key": "TOTAL", "value": "20"}],
        ),
        (
            "receipt_00710.txt",
            "receipt_00710_kvp.json",
            "DOC THREE",
            [{"key": "TOTAL", "value": "30"}],
        ),
    ]

    for txt_name, json_name, doc_text, kvp in data:
        (tmp_path / txt_name).write_text(
            doc_text,
            encoding="utf-8",
        )
        (tmp_path / json_name).write_text(
            json.dumps(kvp),
            encoding="utf-8",
        )

    examples = load_cord_examples(tmp_path)

    assert json.loads(examples["ex1"]["json"]) == {
        "kvp": [{"key": "TOTAL", "value": "10"}]
    }


def test_missing_example_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cord_examples(tmp_path)


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"unexpected": []},
        "not-a-list-or-object",
        123,
        None,
    ],
)
def test_invalid_example_format_is_rejected(
    tmp_path,
    invalid_data,
):
    first_txt, first_json = CORD_EXAMPLE_FILES[0]

    (tmp_path / first_txt).write_text(
        "CORD RECEIPT",
        encoding="utf-8",
    )
    (tmp_path / first_json).write_text(
        json.dumps(invalid_data),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_cord_examples(tmp_path)


def test_invalid_json_is_rejected(tmp_path):
    first_txt, first_json = CORD_EXAMPLE_FILES[0]

    (tmp_path / first_txt).write_text(
        "CORD RECEIPT",
        encoding="utf-8",
    )
    (tmp_path / first_json).write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        load_cord_examples(tmp_path)
