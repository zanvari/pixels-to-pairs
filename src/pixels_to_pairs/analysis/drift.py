"""Prediction drift diagnostics used in the CORD benchmark."""

import re
from collections import Counter
from typing import Any, Dict, List

from pixels_to_pairs.evaluation.normalization import normalize_text


_NUM_LIKE_RE = re.compile(
    r"^[\d\s.,:/\-+%$€£¥()]+$"
)


def _is_numeric_like_key(key: str) -> bool:
    """Return whether a normalized key is numeric-like."""

    key = normalize_text(key)

    if not key:
        return False

    return bool(_NUM_LIKE_RE.match(key))


def _is_qty_like_key(key: str) -> bool:
    """Return whether a key resembles a quantity marker such as x2."""

    key = normalize_text(key).replace(" ", "")

    if not key:
        return False

    if key.startswith("x") and key[1:].isdigit():
        return True

    if key.endswith("x") and key[:-1].isdigit():
        return True

    return False


def compute_pred_drift_stats(
    pred_kvp: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Compute the CORD prediction drift diagnostics."""

    keys = [
        str(pair.get("key", ""))
        for pair in (pred_kvp or [])
    ]

    keys_norm = [
        normalize_text(key)
        for key in keys
        if normalize_text(key)
    ]

    if not keys_norm:
        return {
            "pred_num_pairs": 0,
            "pred_unique_keys": 0,
            "pred_dup_key_frac": 0.0,
            "pred_single_token_key_frac": 0.0,
            "pred_numeric_key_frac": 0.0,
            "pred_qty_key_frac": 0.0,
        }

    counts = Counter(keys_norm)
    total = len(keys_norm)
    unique = len(counts)

    duplicate_count = sum(
        count
        for count in counts.values()
        if count >= 2
    )

    single_token = sum(
        1
        for key in keys_norm
        if len(key.split()) == 1
    )

    numeric_like = sum(
        1
        for key in keys_norm
        if _is_numeric_like_key(key)
    )

    qty_like = sum(
        1
        for key in keys_norm
        if _is_qty_like_key(key)
    )

    return {
        "pred_num_pairs": int(total),
        "pred_unique_keys": int(unique),
        "pred_dup_key_frac": float(
            duplicate_count / total
        ),
        "pred_single_token_key_frac": float(
            single_token / total
        ),
        "pred_numeric_key_frac": float(
            numeric_like / total
        ),
        "pred_qty_key_frac": float(
            qty_like / total
        ),
    }
