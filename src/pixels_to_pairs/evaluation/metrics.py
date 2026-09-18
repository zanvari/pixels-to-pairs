"""Metrics used by the Pixels to Pairs benchmark."""

from collections import Counter
from typing import Any, Dict, List

import numpy as np

from .normalization import normalize_key, normalize_text


def text_f1(gt: str, pred: str) -> float:
    """Token-level F1 with multiplicities (Counter overlap), not set overlap."""
    gt_tokens = normalize_text(gt).split()
    pred_tokens = normalize_text(pred).split()

    if not gt_tokens and not pred_tokens:
        return 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0

    gt_c = Counter(gt_tokens)
    pred_c = Counter(pred_tokens)
    overlap = sum((gt_c & pred_c).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_doc_metrics(
    gt_kvp: List[Dict[str, str]],
    pred_kvp: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Compute document-level Key Recall, Exact Match, and Value F1."""

    num_gt = len(gt_kvp)

    if num_gt == 0:
        return {
            "num_gt_keys": 0,
            "matched_keys": 0,
            "key_recall": 0.0,
            "exact_matches": 0,
            "exact_match_rate": 0.0,
            "value_f1": 0.0,
        }

    # Normalize GT and predicted key/value strings.
    gt_keys_norm = [
        normalize_key(kv.get("key", ""))
        for kv in gt_kvp
    ]
    gt_vals_norm = [
        normalize_text(kv.get("value", ""))
        for kv in gt_kvp
    ]

    pred_keys_norm = [
        normalize_key(kv.get("key", ""))
        for kv in pred_kvp
    ]
    pred_vals_norm = [
        normalize_text(kv.get("value", ""))
        for kv in pred_kvp
    ]

    # =================================================
    # 1. KEY RECALL
    # Duplicate-aware key matching
    # =================================================

    gt_key_counts = Counter(k for k in gt_keys_norm if k)
    pred_key_counts = Counter(k for k in pred_keys_norm if k)

    matched_keys = sum(
        min(gt_key_counts[k], pred_key_counts.get(k, 0))
        for k in gt_key_counts
    )

    key_recall = matched_keys / num_gt

    # =================================================
    # 2. EXACT MATCH RATE
    # Exact normalized key AND exact normalized value.
    # Each prediction can be used at most once.
    # =================================================

    used_pred_exact = set()
    exact_matches = 0

    for gt_key, gt_value in zip(gt_keys_norm, gt_vals_norm):

        if not gt_key or not gt_value:
            continue

        for j, (pred_key, pred_value) in enumerate(
            zip(pred_keys_norm, pred_vals_norm)
        ):
            if j in used_pred_exact:
                continue

            if gt_key == pred_key and gt_value == pred_value:
                exact_matches += 1
                used_pred_exact.add(j)
                break

    exact_match_rate = exact_matches / num_gt

    # =================================================
    # 3. VALUE TOKEN F1
    #
    # For every GT pair:
    #   - consider unused predictions with the same key
    #   - select the one with maximum token-level F1
    #   - same-key but zero-overlap value -> F1 = 0
    #   - no same-key prediction -> F1 = 0
    #
    # Final Value F1 is averaged over ALL GT pairs.
    # =================================================

    used_pred_f1 = set()
    value_f1_scores = []

    for gt_pair in gt_kvp:

        gt_key = normalize_key(gt_pair.get("key", ""))
        gt_value = str(gt_pair.get("value", ""))

        # A malformed/empty GT field receives zero credit.
        if not gt_key or not normalize_text(gt_value):
            value_f1_scores.append(0.0)
            continue

        best_j = None
        best_f1 = 0.0

        for j, pred_pair in enumerate(pred_kvp):

            if j in used_pred_f1:
                continue

            pred_key = normalize_key(pred_pair.get("key", ""))

            if pred_key != gt_key or not pred_key:
                continue

            pred_value = str(pred_pair.get("value", ""))

            f1 = text_f1(gt_value, pred_value)

            # Important:
            # best_j must also be assigned when F1 == 0.
            if best_j is None or f1 > best_f1:
                best_j = j
                best_f1 = f1

        if best_j is not None:
            used_pred_f1.add(best_j)
            value_f1_scores.append(best_f1)
        else:
            # Missing key
            value_f1_scores.append(0.0)

    value_f1 = float(np.mean(value_f1_scores))

    return {
        "num_gt_keys": num_gt,
        "matched_keys": int(matched_keys),
        "key_recall": float(key_recall),
        "exact_matches": int(exact_matches),
        "exact_match_rate": float(exact_match_rate),
        "value_f1": float(value_f1),
    }
