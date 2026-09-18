"""Regression tests for benchmark normalization and evaluation metrics."""

from pixels_to_pairs.evaluation.metrics import compute_doc_metrics, text_f1
from pixels_to_pairs.evaluation.normalization import normalize_key, normalize_text


def test_normalize_text():
    assert normalize_text("  Hello   WORLD  ") == "hello world"


def test_normalize_key_equivalent_labels():
    equivalent_pairs = [
        ("DATE:", "date"),
        ("• CASE NAME :", "Case Name"),
        ("NO. OF STORES", "no of stores"),
        ("P.O.S.", "POS"),
        ("SENDER /PHONE NUMBER:", "sender/phone number"),
    ]

    for first, second in equivalent_pairs:
        assert normalize_key(first) == normalize_key(second)


def test_text_f1_exact_match():
    assert text_f1("hello world", "hello world") == 1.0


def test_text_f1_no_overlap():
    assert text_f1("hello world", "goodbye moon") == 0.0


def test_text_f1_partial_overlap():
    assert text_f1("new york city", "new york") == 0.8


def test_text_f1_preserves_token_multiplicity():
    assert text_f1("total total amount", "total amount") == 0.8


def test_text_f1_empty_values():
    assert text_f1("", "") == 1.0
    assert text_f1("hello", "") == 0.0
    assert text_f1("", "hello") == 0.0


def test_compute_doc_metrics_perfect_prediction():
    gt = [
        {"key": "Company", "value": "ABC Store"},
        {"key": "Date", "value": "01/15/2025"},
        {"key": "Total", "value": "$25.00"},
    ]

    pred = [
        {"key": "Company", "value": "ABC Store"},
        {"key": "Date:", "value": "01/15/2025"},
        {"key": "TOTAL", "value": "$25.00"},
    ]

    result = compute_doc_metrics(gt, pred)

    assert result["matched_keys"] == 3
    assert result["key_recall"] == 1.0
    assert result["exact_matches"] == 3
    assert result["exact_match_rate"] == 1.0
    assert result["value_f1"] == 1.0


def test_compute_doc_metrics_missing_prediction():
    gt = [
        {"key": "Company", "value": "ABC Store"},
        {"key": "Date", "value": "01/15/2025"},
        {"key": "Total", "value": "$25.00"},
    ]

    pred = [
        {"key": "Company", "value": "ABC Store"},
        {"key": "Date", "value": "01/15/2025"},
    ]

    result = compute_doc_metrics(gt, pred)

    assert result["matched_keys"] == 2
    assert result["key_recall"] == 2 / 3
    assert result["exact_matches"] == 2
    assert result["exact_match_rate"] == 2 / 3
    assert result["value_f1"] == 2 / 3


def test_compute_doc_metrics_duplicate_aware_key_recall():
    gt = [
        {"key": "Item", "value": "Apple"},
        {"key": "Item", "value": "Banana"},
        {"key": "Total", "value": "10"},
    ]

    pred = [
        {"key": "Item", "value": "Apple"},
        {"key": "Total", "value": "10"},
    ]

    result = compute_doc_metrics(gt, pred)

    assert result["matched_keys"] == 2
    assert result["key_recall"] == 2 / 3


def test_compute_doc_metrics_does_not_reuse_predictions():
    gt = [
        {"key": "Item", "value": "Apple"},
        {"key": "Item", "value": "Apple"},
    ]

    pred = [
        {"key": "Item", "value": "Apple"},
    ]

    result = compute_doc_metrics(gt, pred)

    assert result["matched_keys"] == 1
    assert result["exact_matches"] == 1
    assert result["key_recall"] == 0.5
    assert result["exact_match_rate"] == 0.5
    assert result["value_f1"] == 0.5


def test_compute_doc_metrics_partial_value():
    gt = [
        {"key": "Address", "value": "123 Main Street"},
    ]

    pred = [
        {"key": "Address", "value": "123 Main"},
    ]

    result = compute_doc_metrics(gt, pred)

    assert result["key_recall"] == 1.0
    assert result["exact_match_rate"] == 0.0
    assert result["value_f1"] == 0.8


def test_compute_doc_metrics_empty_ground_truth():
    result = compute_doc_metrics([], [])

    assert result == {
        "num_gt_keys": 0,
        "matched_keys": 0,
        "key_recall": 0.0,
        "exact_matches": 0,
        "exact_match_rate": 0.0,
        "value_f1": 0.0,
    }
