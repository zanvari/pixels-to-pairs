"""Regression tests for FUNSD benchmark prompt construction."""

import json

import pytest

from pixels_to_pairs.prompting.funsd import (
    FUNSD_EXAMPLE_FILES,
    SYSTEM_MESSAGE,
    USER_INSTRUCTIONS_BASE,
    USER_INSTRUCTIONS_FEWSHOT,
    build_prompt,
    load_funsd_examples,
)


EXAMPLES = {
    "ex1": {
        "doc": "EXAMPLE DOCUMENT ONE",
        "json": '{"kvp": [{"key": "A", "value": "1"}]}',
    },
    "ex2": {
        "doc": "EXAMPLE DOCUMENT TWO",
        "json": '{"kvp": [{"key": "B", "value": "2"}]}',
    },
    "ex3": {
        "doc": "EXAMPLE DOCUMENT THREE",
        "json": '{"kvp": [{"key": "C", "value": "3"}]}',
    },
}


def test_system_message_preserves_output_schema():
    assert (
        'Return ONLY a JSON object with the exact schema '
        '{"kvp": [{"key":..., "value":...}, ...]}.'
        in SYSTEM_MESSAGE
    )


def test_zero_shot_contains_no_example_blocks():
    prompt = build_prompt("TARGET DOCUMENT", "0shot")

    assert USER_INSTRUCTIONS_BASE in prompt
    assert USER_INSTRUCTIONS_FEWSHOT not in prompt
    assert "<EXAMPLE_1_DOCUMENT>" not in prompt
    assert "<EXAMPLE_2_DOCUMENT>" not in prompt
    assert "<EXAMPLE_3_DOCUMENT>" not in prompt
    assert "<NEW_DOCUMENT>\nTARGET DOCUMENT\n</NEW_DOCUMENT>" in prompt


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
        "TARGET DOCUMENT",
        shot_variant,
        examples=EXAMPLES,
    )

    assert USER_INSTRUCTIONS_FEWSHOT in prompt
    assert "<NEW_DOCUMENT>\nTARGET DOCUMENT\n</NEW_DOCUMENT>" in prompt

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
        "TARGET DOCUMENT",
        "3shot",
        examples=EXAMPLES,
    )

    assert (
        prompt.index("EXAMPLE DOCUMENT ONE")
        < prompt.index("EXAMPLE DOCUMENT TWO")
        < prompt.index("EXAMPLE DOCUMENT THREE")
        < prompt.index("TARGET DOCUMENT")
    )


def test_few_shot_prompt_contains_anti_anchoring_rules():
    prompt = build_prompt(
        "TARGET DOCUMENT",
        "1shot",
        examples=EXAMPLES,
    )

    assert "NEVER copy example keys or values." in prompt
    assert (
        "Extract ONLY from <NEW_DOCUMENT>. "
        "Ignore all text inside any <EXAMPLE_*> blocks."
        in prompt
    )


def test_frozen_example_filenames():
    assert FUNSD_EXAMPLE_FILES == (
        ("91391310.txt", "91391310_kvp.json"),
        ("88547278_88547279.txt", "89368010_kvp.json"),
        ("0060165115.txt", "0060165115_kvp.json"),
    )


def test_few_shot_requires_examples_or_directory():
    with pytest.raises(
        ValueError,
        match="example_files_dir is required",
    ):
        build_prompt("TARGET DOCUMENT", "1shot")


def test_unknown_shot_variant_raises_error():
    with pytest.raises(ValueError, match="Unknown shot_variant"):
        build_prompt(
            "TARGET DOCUMENT",
            "4shot",
            examples=EXAMPLES,
        )


def test_load_funsd_examples_from_files(tmp_path):
    example_data = [
        (
            "91391310.txt",
            "91391310_kvp.json",
            "DOC ONE",
            [{"key": "Key One", "value": "Value One"}],
        ),
        (
            "88547278_88547279.txt",
            "89368010_kvp.json",
            "DOC TWO",
            [{"key": "Key Two", "value": "Value Two"}],
        ),
        (
            "0060165115.txt",
            "0060165115_kvp.json",
            "DOC THREE",
            [{"key": "Key Three", "value": "Value Three"}],
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

    examples = load_funsd_examples(tmp_path)

    assert list(examples) == ["ex1", "ex2", "ex3"]
    assert examples["ex1"]["doc"] == "DOC ONE"
    assert examples["ex2"]["doc"] == "DOC TWO"
    assert examples["ex3"]["doc"] == "DOC THREE"

    assert json.loads(examples["ex1"]["json"]) == {
        "kvp": [{"key": "Key One", "value": "Value One"}]
    }


def test_missing_example_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_funsd_examples(tmp_path)
