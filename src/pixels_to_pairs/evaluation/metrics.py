"""Metrics used by the Pixels to Pairs benchmark."""

from collections import Counter

from .normalization import normalize_text


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
