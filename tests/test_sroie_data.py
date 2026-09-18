"""Regression tests for SROIE data loading."""

import json

from pixels_to_pairs.data.sroie import (
    load_doc_texts,
    load_sroie_gt_kvp,
)


def test_load_sroie_gt_kvp_loads_wrapped_kvp(tmp_path):
    annotation = {
        "kvp": [
            {"key": "company", "value": "ABC Store"},
            {"key": "total", "value": "12.50"},
        ]
    }

    (tmp_path / "receipt1.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    gt = load_sroie_gt_kvp(tmp_path)

    assert gt == {
        "receipt1": [
            {"key": "company", "value": "ABC Store"},
            {"key": "total", "value": "12.50"},
        ]
    }


def test_load_sroie_gt_kvp_loads_list_format(tmp_path):
    annotation = [
        {"key": "company", "value": "ABC Store"},
        {"key": "date", "value": "01/15/2025"},
    ]

    (tmp_path / "receipt1.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    gt = load_sroie_gt_kvp(tmp_path)

    assert gt == {
        "receipt1": [
            {"key": "company", "value": "ABC Store"},
            {"key": "date", "value": "01/15/2025"},
        ]
    }


def test_load_sroie_gt_kvp_strips_key_and_value(tmp_path):
    annotation = [
        {
            "key": "  company  ",
            "value": "  ABC Store  ",
        }
    ]

    (tmp_path / "receipt1.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    gt = load_sroie_gt_kvp(tmp_path)

    assert gt == {
        "receipt1": [
            {"key": "company", "value": "ABC Store"},
        ]
    }


def test_load_sroie_gt_kvp_coerces_values_to_strings(tmp_path):
    annotation = [
        {"key": "total", "value": 12.5},
        {"key": 123, "value": "numeric key"},
        {"key": "missing", "value": None},
    ]

    (tmp_path / "receipt1.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    gt = load_sroie_gt_kvp(tmp_path)

    assert gt == {
        "receipt1": [
            {"key": "total", "value": "12.5"},
            {"key": "123", "value": "numeric key"},
            {"key": "missing", "value": "None"},
        ]
    }


def test_load_sroie_gt_kvp_skips_non_dict_entries(tmp_path):
    annotation = [
        {"key": "company", "value": "ABC Store"},
        "invalid",
        123,
        None,
        ["company", "XYZ Store"],
    ]

    (tmp_path / "receipt1.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    gt = load_sroie_gt_kvp(tmp_path)

    assert gt == {
        "receipt1": [
            {"key": "company", "value": "ABC Store"},
        ]
    }


def test_load_sroie_gt_kvp_skips_only_both_empty(tmp_path):
    annotation = [
        {"key": "", "value": ""},
        {"key": "company", "value": ""},
        {"key": "", "value": "ABC Store"},
    ]

    (tmp_path / "receipt1.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    gt = load_sroie_gt_kvp(tmp_path)

    assert gt == {
        "receipt1": [
            {"key": "company", "value": ""},
            {"key": "", "value": "ABC Store"},
        ]
    }


def test_load_sroie_gt_kvp_handles_missing_fields(tmp_path):
    annotation = [
        {"key": "company"},
        {"value": "ABC Store"},
        {},
    ]

    (tmp_path / "receipt1.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    gt = load_sroie_gt_kvp(tmp_path)

    assert gt == {
        "receipt1": [
            {"key": "company", "value": ""},
            {"key": "", "value": "ABC Store"},
        ]
    }


def test_load_sroie_gt_kvp_skips_unknown_format(
    tmp_path,
    capsys,
):
    (tmp_path / "bad.json").write_text(
        json.dumps({"unexpected": []}),
        encoding="utf-8",
    )

    gt = load_sroie_gt_kvp(tmp_path)

    assert gt == {}

    captured = capsys.readouterr()

    assert (
        "Unknown SROIE annotation format for "
        "bad.json"
    ) in captured.out


def test_load_sroie_gt_kvp_uses_sorted_filenames(tmp_path):
    (tmp_path / "b.json").write_text(
        json.dumps(
            [{"key": "company", "value": "B"}]
        ),
        encoding="utf-8",
    )

    (tmp_path / "a.json").write_text(
        json.dumps(
            [{"key": "company", "value": "A"}]
        ),
        encoding="utf-8",
    )

    gt = load_sroie_gt_kvp(tmp_path)

    assert list(gt) == ["a", "b"]


def test_load_sroie_gt_kvp_ignores_non_json_files(tmp_path):
    (tmp_path / "receipt.json").write_text(
        json.dumps(
            [{"key": "company", "value": "ABC Store"}]
        ),
        encoding="utf-8",
    )

    (tmp_path / "notes.txt").write_text(
        "not an annotation",
        encoding="utf-8",
    )

    gt = load_sroie_gt_kvp(tmp_path)

    assert list(gt) == ["receipt"]


def test_load_doc_texts_preserves_raw_text(tmp_path):
    raw_text = (
        "  ABC STORE  \n"
        "123 MAIN STREET\n"
        "\n"
        "TOTAL 12.50\n"
    )

    (tmp_path / "receipt1.txt").write_text(
        raw_text,
        encoding="utf-8",
    )

    texts = load_doc_texts(tmp_path)

    assert texts == {
        "receipt1": raw_text,
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
