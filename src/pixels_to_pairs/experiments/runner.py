"""Shared infrastructure for benchmark experiment runners."""

import os
from pathlib import Path
from typing import Iterable, List, Set


METRICS_HEADER = (
    "doc_id,num_gt_keys,matched_keys,key_recall,"
    "exact_match_rate,value_f1,"
    "prompt_tokens_no_trunc,prompt_tokens_trunc,"
    "prompt_truncated,generated_tokens,"
    "output_hit_max_new_tokens\n"
)


def select_common_doc_ids(
    gt_dict,
    text_dict,
    *,
    exclude_empty_gt: bool = False,
    max_docs=None,
) -> List[str]:
    """Select sorted document IDs available in both GT and text inputs."""

    doc_ids = sorted(
        set(gt_dict.keys()) & set(text_dict.keys())
    )

    if exclude_empty_gt:
        doc_ids = [
            doc_id
            for doc_id in doc_ids
            if len(gt_dict.get(doc_id, [])) > 0
        ]

    if max_docs is not None:
        doc_ids = doc_ids[:max_docs]

    return doc_ids


def load_processed_ids(
    csv_path,
    *,
    eligible_doc_ids: Iterable[str] = None,
) -> Set[str]:
    """Load completed document IDs from an existing metrics CSV."""

    csv_path = Path(csv_path)

    if not csv_path.exists():
        return set()

    processed_ids = set()

    with csv_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(",")

        if parts and parts[0]:
            processed_ids.add(parts[0])

    if eligible_doc_ids is not None:
        processed_ids &= set(eligible_doc_ids)

    return processed_ids


def prepare_output_files(
    csv_path,
    pred_path,
    *,
    resume: bool,
) -> None:
    """Remove previous outputs when resume mode is disabled."""

    if resume:
        return

    for path in (csv_path, pred_path):
        if os.path.exists(path):
            os.remove(path)


def compute_macro_summary(
    csv_path,
    *,
    exclude_zero_gt: bool = False,
):
    """Compute macro document averages from a benchmark metrics CSV."""

    csv_path = Path(csv_path)

    sum_key_recall = 0.0
    sum_em = 0.0
    sum_value_f1 = 0.0
    n_docs = 0

    if not csv_path.exists():
        return {
            "n_docs": 0,
            "key_recall": 0.0,
            "exact_match_rate": 0.0,
            "value_f1": 0.0,
        }

    with csv_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(",")

        if len(parts) < 6:
            continue

        try:
            num_gt = int(parts[1])
            key_recall = float(parts[3])
            exact_match_rate = float(parts[4])
            value_f1 = float(parts[5])
        except ValueError:
            continue

        if exclude_zero_gt and num_gt == 0:
            continue

        sum_key_recall += key_recall
        sum_em += exact_match_rate
        sum_value_f1 += value_f1
        n_docs += 1

    if n_docs == 0:
        return {
            "n_docs": 0,
            "key_recall": 0.0,
            "exact_match_rate": 0.0,
            "value_f1": 0.0,
        }

    return {
        "n_docs": n_docs,
        "key_recall": sum_key_recall / n_docs,
        "exact_match_rate": sum_em / n_docs,
        "value_f1": sum_value_f1 / n_docs,
    }
