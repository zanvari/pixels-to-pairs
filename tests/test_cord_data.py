"""Regression tests for CORD data loading."""

import json

import pytest

from pixels_to_pairs.data.cord import (
    _coerce_kvp_list,
    load_cord_gt_kvp,
    load_doc_texts,
)


def test_coerce_kvp_list_accepts_wrapped_format():
    data = {
        "kvp": [
            {"key": "TOTAL", "value": "18,000"},
            {"key": "CHANGE DUE", "value": "2,000"},
        ]
    }

    assert _coerce_kvp_list(data) == [
        {"key": "TOTAL", "value": "18,000"},
        {"key": "CHANGE DUE", "value": "2,000"},
    ]


def test_coerce_kvp_list_accepts_list_format():
    data = [
        {"key": "TOTAL", "value": "18,000"},
    ]

    assert _coerce_kvp_list(data) == [
        {"key": "TOTAL", "value": "18,000"},
    ]


def test_coerce_kvp_list_accepts_alternate_field_names():
    data = [
        {"Key": "TOTAL", "Value": "18,000"},
        {"k": "CHANGE", "v": "2,000"},
    ]

    assert _coerce_kvp_list(data) == [
        {"key": "TOTAL", "value": "18,000"},
        {"key": "CHANGE", "value": "2,000"},
    ]


def test_coerce_kvp_list_prefers_standard_field_names():
    data = [
        {
            "key": "TOTAL",
            "Key": "WRONG KEY",
            "k": "ALSO WRONG",
            "value": "18,000",
            "Value": "999",
            "v": "888",
        }
    ]

    assert _coerce_kvp_list(data) == [
        {"key": "TOTAL", "value": "18,000"},
    ]


def test_coerce_kvp_list_strips_key_and_value():
    data = [
        {
            "key": "  TOTAL  ",
            "value": "  18,000  ",
        }
    ]

    assert _coerce_kvp_list(data) == [
        {"key": "TOTAL", "value": "18,000"},
    ]


def test_coerce_kvp_list_converts_values_to_strings():
    data = [
        {"key": 123, "value": 456},
        {"key": "missing", "value": None},
    ]

    assert _coerce_kvp_list(data) == [
        {"key": "123", "value": "456"},
        {"key": "missing", "value": "None"},
    ]


def test_coerce_kvp_list_skips_non_dict_entries():
    data = [
        {"key": "TOTAL", "value": "18,000"},
        "invalid",
        123,
        None,
        ["TOTAL", "20,000"],
    ]

    assert _coerce_kvp_list(data) == [
        {"key": "TOTAL", "value": "18,000"},
    ]


def test_coerce_kvp_list_skips_only_both_empty():
    data = [
        {"key": "", "value": ""},
        {"key": "TOTAL", "value": ""},
        {"key": "", "value": "18,000"},
    ]

    assert _coerce_kvp_list(data) == [
        {"key": "TOTAL", "value": ""},
        {"key": "", "value": "18,000"},
    ]


@pytest.mark.parametrize(
    "data",
    [
        {"unexpected": []},
        {"kvp": "not a list"},
        "invalid",
        123,
        None,
    ],
)
def test_coerce_kvp_list_rejects_unknown_formats(data):
    assert _coerce_kvp_list(data) is None


def test_load_cord_gt_kvp_loads_valid_file(tmp_path):
    annotation = {
        "kvp": [
            {"key": "TOTAL", "value": "18,000"},
        ]
    }

    (tmp_path / "receipt_1.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    gt = load_cord_gt_kvp(
        tmp_path,
        sanity_samples=0,
    )

    assert gt == {
        "receipt_1": [
            {"key": "TOTAL", "value": "18,000"},
        ]
    }


def test_load_cord_gt_kvp_uses_sorted_filenames(tmp_path):
    (tmp_path / "b.json").write_text(
        json.dumps(
            [{"key": "B", "value": "2"}]
        ),
        encoding="utf-8",
    )

    (tmp_path / "a.json").write_text(
        json.dumps(
            [{"key": "A", "value": "1"}]
        ),
        encoding="utf-8",
    )

    gt = load_cord_gt_kvp(
        tmp_path,
        sanity_samples=0,
    )

    assert list(gt) == ["a", "b"]


def test_load_cord_gt_kvp_skips_malformed_json(tmp_path):
    (tmp_path / "bad.json").write_text(
        "{not valid json",
        encoding="utf-8",
    )

    (tmp_path / "good.json").write_text(
        json.dumps(
            [{"key": "TOTAL", "value": "18,000"}]
        ),
        encoding="utf-8",
    )

    gt = load_cord_gt_kvp(
        tmp_path,
        sanity_samples=0,
    )

    assert list(gt) == ["good"]


def test_load_cord_gt_kvp_skips_unknown_format(tmp_path):
    (tmp_path / "bad.json").write_text(
        json.dumps({"unexpected": []}),
        encoding="utf-8",
    )

    (tmp_path / "good.json").write_text(
        json.dumps(
            [{"key": "TOTAL", "value": "18,000"}]
        ),
        encoding="utf-8",
    )

    gt = load_cord_gt_kvp(
        tmp_path,
        sanity_samples=0,
    )

    assert list(gt) == ["good"]


def test_load_cord_gt_kvp_raises_when_no_gt_loads(tmp_path):
    (tmp_path / "bad.json").write_text(
        "{not valid json",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="Loaded 0 GT documents",
    ):
        load_cord_gt_kvp(
            tmp_path,
            sanity_samples=0,
        )


def test_load_cord_gt_kvp_raises_for_empty_directory(tmp_path):
    with pytest.raises(
        RuntimeError,
        match="Loaded 0 GT documents",
    ):
        load_cord_gt_kvp(
            tmp_path,
            sanity_samples=0,
        )


def test_load_cord_gt_kvp_respects_sanity_sample_limit(
    tmp_path,
    capsys,
):
    for index in range(3):
        (tmp_path / f"receipt_{index}.json").write_text(
            json.dumps(
                [{"key": "TOTAL", "value": str(index)}]
            ),
            encoding="utf-8",
        )

    load_cord_gt_kvp(
        tmp_path,
        sanity_samples=1,
    )

    captured = capsys.readouterr()

    assert captured.out.count(
        "[SANITY] GT SAMPLE FILE:"
    ) == 1


def test_load_doc_texts_preserves_raw_text(tmp_path):
    raw_text = (
        "  STORE NAME  \n"
        "ITEM 18,000\n"
        "\n"
        "TOTAL 18,000\n"
    )

    (tmp_path / "receipt.txt").write_text(
        raw_text,
        encoding="utf-8",
    )

    texts = load_doc_texts(tmp_path)

    assert texts == {
        "receipt": raw_text,
    }


def test_load_doc_texts_uses_sorted_filenames(tmp_path):
    (tmp_path / "b.txt").write_text(
        "second",
        encoding="utf-8",
    )

    (tmp_path / "a.txt").write_text(
        "first",
        encoding="utf-8",
    )

    texts = load_doc_texts(tmp_path)

    assert list(texts) == ["a", "b"]


def test_load_doc_texts_ignores_non_txt_files(tmp_path):
    (tmp_path / "receipt.txt").write_text(
        "receipt",
        encoding="utf-8",
    )

    (tmp_path / "notes.json").write_text(
        "{}",
        encoding="utf-8",
    )

    texts = load_doc_texts(tmp_path)

    assert texts == {
        "receipt": "receipt",
    }
