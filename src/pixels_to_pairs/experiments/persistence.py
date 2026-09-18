"""Persistence utilities for benchmark experiment outputs."""

import json
import os

from .runner import METRICS_HEADER


CORD_DRIFT_FIELDS = (
    "pred_num_pairs",
    "pred_unique_keys",
    "pred_dup_key_frac",
    "pred_single_token_key_frac",
    "pred_numeric_key_frac",
    "pred_qty_key_frac",
)


CORD_METRICS_HEADER = (
    METRICS_HEADER.rstrip("\n")
    + ","
    + ",".join(CORD_DRIFT_FIELDS)
    + "\n"
)


def format_metrics_row(result):
    """Format one standard benchmark metrics CSV row."""

    metrics = result["metrics"]

    return (
        f"{result['doc_id']},"
        f"{metrics['num_gt_keys']},"
        f"{metrics['matched_keys']},"
        f"{metrics['key_recall']:.10f},"
        f"{metrics['exact_match_rate']:.10f},"
        f"{metrics['value_f1']:.10f},"
        f"{result['prompt_tokens_no_trunc']},"
        f"{result['prompt_tokens_trunc']},"
        f"{int(result['prompt_truncated'])},"
        f"{result['generated_tokens']},"
        f"{int(result['output_hit_max_new_tokens'])}\n"
    )


def format_cord_metrics_row(
    result,
    drift,
):
    """Format one CORD metrics row including drift diagnostics."""

    standard = format_metrics_row(
        result
    ).rstrip("\n")

    return (
        f"{standard},"
        f"{drift['pred_num_pairs']},"
        f"{drift['pred_unique_keys']},"
        f"{drift['pred_dup_key_frac']:.4f},"
        f"{drift['pred_single_token_key_frac']:.4f},"
        f"{drift['pred_numeric_key_frac']:.4f},"
        f"{drift['pred_qty_key_frac']:.4f}\n"
    )


def build_prediction_record(
    result,
    *,
    engine,
    shot_variant,
    model_name,
    seed,
    dtype,
    fewshot_example_ids=None,
    drift=None,
):
    """Build one JSONL prediction record."""

    record = {
        "doc_id": result["doc_id"],
        "engine": engine,
        "shot_variant": shot_variant,
        "model": model_name,
        "gt_kvp": result["gt_kvp"],
        "pred_kvp": result["pred_kvp"],
        "raw_output": result["raw_output"],
        "metrics": result["metrics"],
        "prompt_tokens_no_trunc": (
            result["prompt_tokens_no_trunc"]
        ),
        "prompt_tokens_trunc": (
            result["prompt_tokens_trunc"]
        ),
        "prompt_truncated": bool(
            result["prompt_truncated"]
        ),
        "generated_tokens": int(
            result["generated_tokens"]
        ),
        "output_hit_max_new_tokens": bool(
            result["output_hit_max_new_tokens"]
        ),
        "seed": seed,
        "dtype": str(dtype),
    }

    if drift is not None:
        record["drift"] = drift

    if fewshot_example_ids is not None:
        record["fewshot_example_ids"] = list(
            fewshot_example_ids
        )

    return record


def write_and_sync(
    file_obj,
    text,
):
    """Write, flush, and fsync one output record."""

    file_obj.write(text)
    file_obj.flush()
    os.fsync(file_obj.fileno())


def write_jsonl_record(
    file_obj,
    record,
    *,
    ensure_ascii=True,
):
    """Persist one JSONL record and force it to disk."""

    text = json.dumps(
        record,
        ensure_ascii=ensure_ascii,
    )

    write_and_sync(
        file_obj,
        text + "\n",
    )
