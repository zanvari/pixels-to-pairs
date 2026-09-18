"""Regression tests for FUNSD data loading."""

import json

from pixels_to_pairs.data.funsd import (
    kvp_from_funsd_form,
    load_doc_texts,
    load_funsd_gt_kvp,
)


def test_kvp_from_funsd_form_extracts_linked_pairs():
    data = {
        "form": [
            {
                "id": 1,
                "label": "question",
                "text": "Company",
                "linking": [[1, 2]],
            },
            {
                "id": 2,
                "label": "answer",
                "text": "ABC Store",
                "linking": [[1, 2]],
            },
            {
                "id": 3,
                "label": "question",
                "text": "Date",
                "linking": [[3, 4]],
            },
            {
                "id": 4,
                "label": "answer",
                "text": "01/15/2025",
                "linking": [[3, 4]],
            },
        ]
    }

    assert kvp_from_funsd_form(data) == [
        {"key": "Company", "value": "ABC Store"},
        {"key": "Date", "value": "01/15/2025"},
    ]


def test_kvp_from_funsd_form_preserves_multiple_links():
    data = {
        "form": [
            {
                "id": 1,
                "label": "question",
                "text": "Name",
                "linking": [[1, 2], [1, 3]],
            },
            {
                "id": 2,
                "label": "answer",
                "text": "Alice",
            },
            {
                "id": 3,
                "label": "answer",
                "text": "Bob",
            },
        ]
    }

    assert kvp_from_funsd_form(data) == [
        {"key": "Name", "value": "Alice"},
        {"key": "Name", "value": "Bob"},
    ]


def test_kvp_from_funsd_form_ignores_reversed_link():
    data = {
        "form": [
            {
                "id": 1,
                "label": "question",
                "text": "Company",
                "linking": [[2, 1]],
            },
            {
                "id": 2,
                "label": "answer",
                "text": "ABC Store",
            },
        ]
    }

    assert kvp_from_funsd_form(data) == []


def test_kvp_from_funsd_form_ignores_invalid_links():
    data = {
        "form": [
            {
                "id": 1,
                "label": "question",
                "text": "Company",
                "linking": [
                    [1],
                    [1, 2, 3],
                    "1,2",
                ],
            },
            {
                "id": 2,
                "label": "answer",
                "text": "ABC Store",
            },
        ]
    }

    assert kvp_from_funsd_form(data) == []


def test_kvp_from_funsd_form_skips_empty_question_text():
    data = {
        "form": [
            {
                "id": 1,
                "label": "question",
                "text": "   ",
                "linking": [[1, 2]],
            },
            {
                "id": 2,
                "label": "answer",
                "text": "ABC Store",
            },
        ]
    }

    assert kvp_from_funsd_form(data) == []


def test_kvp_from_funsd_form_skips_empty_answer_text():
    data = {
        "form": [
            {
                "id": 1,
                "label": "question",
                "text": "Company",
                "linking": [[1, 2]],
            },
            {
                "id": 2,
                "label": "answer",
                "text": "   ",
            },
        ]
    }

    assert kvp_from_funsd_form(data) == []


def test_kvp_from_funsd_form_skips_missing_answer():
    data = {
        "form": [
            {
                "id": 1,
                "label": "question",
                "text": "Company",
                "linking": [[1, 999]],
            },
        ]
    }

    assert kvp_from_funsd_form(data) == []


def test_load_funsd_gt_kvp_loads_form_annotations(tmp_path):
    annotation = {
        "form": [
            {
                "id": 1,
                "label": "question",
                "text": "Company",
                "linking": [[1, 2]],
            },
            {
                "id": 2,
                "label": "answer",
                "text": "ABC Store",
            },
        ]
    }

    (tmp_path / "doc1.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    gt = load_funsd_gt_kvp(tmp_path)

    assert gt == {
        "doc1": [
            {"key": "Company", "value": "ABC Store"},
        ]
    }


def test_load_funsd_gt_kvp_accepts_preconverted_annotations(tmp_path):
    annotation = [
        {"key": "Company", "value": "ABC Store"},
        {"key": "Date", "value": "01/15/2025"},
    ]

    (tmp_path / "doc1.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    gt = load_funsd_gt_kvp(tmp_path)

    assert gt == {
        "doc1": [
            {"key": "Company", "value": "ABC Store"},
            {"key": "Date", "value": "01/15/2025"},
        ]
    }


def test_load_funsd_gt_kvp_skips_unknown_format(
    tmp_path,
    capsys,
):
    (tmp_path / "bad.json").write_text(
        json.dumps({"unexpected": []}),
        encoding="utf-8",
    )

    gt = load_funsd_gt_kvp(tmp_path)

    assert gt == {}

    captured = capsys.readouterr()
    assert "Unknown annotation format for bad.json" in captured.out


def test_load_funsd_gt_kvp_uses_sorted_filenames(tmp_path):
    (tmp_path / "b.json").write_text(
        json.dumps([{"key": "B", "value": "2"}]),
        encoding="utf-8",
    )
    (tmp_path / "a.json").write_text(
        json.dumps([{"key": "A", "value": "1"}]),
        encoding="utf-8",
    )

    gt = load_funsd_gt_kvp(tmp_path)

    assert list(gt) == ["a", "b"]


def test_load_doc_texts_preserves_raw_text(tmp_path):
    raw_text = "  Company: ABC Store  \n\nTotal: $10.00\n"

    (tmp_path / "doc1.txt").write_text(
        raw_text,
        encoding="utf-8",
    )

    texts = load_doc_texts(tmp_path)

    assert texts == {"doc1": raw_text}


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
    (tmp_path / "doc1.txt").write_text(
        "document",
        encoding="utf-8",
    )
    (tmp_path / "notes.json").write_text(
        "{}",
        encoding="utf-8",
    )

    texts = load_doc_texts(tmp_path)

    assert texts == {"doc1": "document"}
