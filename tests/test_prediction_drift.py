import pytest

from pixels_to_pairs.analysis.drift import (
    _is_numeric_like_key,
    _is_qty_like_key,
    compute_pred_drift_stats,
)


def test_empty_predictions_have_zero_drift():
    assert compute_pred_drift_stats([]) == {
        "pred_num_pairs": 0,
        "pred_unique_keys": 0,
        "pred_dup_key_frac": 0.0,
        "pred_single_token_key_frac": 0.0,
        "pred_numeric_key_frac": 0.0,
        "pred_qty_key_frac": 0.0,
    }


def test_none_predictions_have_zero_drift():
    assert compute_pred_drift_stats(None) == {
        "pred_num_pairs": 0,
        "pred_unique_keys": 0,
        "pred_dup_key_frac": 0.0,
        "pred_single_token_key_frac": 0.0,
        "pred_numeric_key_frac": 0.0,
        "pred_qty_key_frac": 0.0,
    }


@pytest.mark.parametrize(
    "key",
    [
        "123",
        "12.50",
        "$ 10.00",
        "12/34",
        "50%",
    ],
)
def test_numeric_like_keys(key):
    assert _is_numeric_like_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "total",
        "item 2",
        "x2",
    ],
)
def test_non_numeric_like_keys(key):
    assert not _is_numeric_like_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "x2",
        "2x",
        "X 12",
        "12 X",
    ],
)
def test_quantity_like_keys(key):
    assert _is_qty_like_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "",
        "x",
        "quantity",
        "2",
        "x2a",
    ],
)
def test_non_quantity_like_keys(key):
    assert not _is_qty_like_key(key)


def test_drift_statistics_match_notebook_behavior():
    predictions = [
        {"key": "Total", "value": "10"},
        {"key": " total ", "value": "20"},
        {"key": "x2", "value": "item"},
        {"key": "123", "value": "value"},
        {"key": "Unit Price", "value": "5"},
    ]

    drift = compute_pred_drift_stats(
        predictions
    )

    assert drift["pred_num_pairs"] == 5
    assert drift["pred_unique_keys"] == 4

    # Both occurrences of the duplicated "total"
    # key count toward the notebook's duplicate fraction.
    assert drift["pred_dup_key_frac"] == pytest.approx(
        2 / 5
    )

    assert drift[
        "pred_single_token_key_frac"
    ] == pytest.approx(
        4 / 5
    )

    assert drift[
        "pred_numeric_key_frac"
    ] == pytest.approx(
        1 / 5
    )

    assert drift[
        "pred_qty_key_frac"
    ] == pytest.approx(
        1 / 5
    )


def test_blank_keys_are_excluded():
    predictions = [
        {"key": "", "value": "a"},
        {"key": "   ", "value": "b"},
        {"key": "Total", "value": "10"},
    ]

    drift = compute_pred_drift_stats(
        predictions
    )

    assert drift["pred_num_pairs"] == 1
    assert drift["pred_unique_keys"] == 1
