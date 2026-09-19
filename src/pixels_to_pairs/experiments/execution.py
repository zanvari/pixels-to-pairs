"""Shared execution helpers for benchmark experiments."""

from pathlib import Path

from pixels_to_pairs.analysis.drift import (
    compute_pred_drift_stats,
)
from pixels_to_pairs.experiments.persistence import (
    CORD_METRICS_HEADER,
    build_prediction_record,
    format_cord_metrics_row,
    format_metrics_row,
    write_and_sync,
    write_jsonl_record,
)
from pixels_to_pairs.experiments.runner import (
    METRICS_HEADER,
    compute_macro_summary,
    evaluate_batch,
    load_processed_ids,
    prepare_output_files,
)


def build_output_paths(
    results_base,
    dataset,
    engine,
    shot_variant,
    model_name,
):
    """Build the output directory, CSV, and JSONL paths."""

    model_slug = model_name.replace("/", "_")

    out_dir_name = (
        f"{dataset}_{engine}_"
        f"{shot_variant}_{model_slug}"
    )

    output_dir = (
        Path(results_base) / out_dir_name
    )

    csv_path = output_dir / (
        f"metrics_{engine}_"
        f"{shot_variant}_{model_slug}.csv"
    )

    pred_path = output_dir / (
        f"predictions_{engine}_"
        f"{shot_variant}_{model_slug}.jsonl"
    )

    return output_dir, csv_path, pred_path


def remaining_doc_ids(
    doc_ids,
    csv_path,
    *,
    resume,
    restrict_processed_to_eligible=False,
):
    """Return document IDs that still require evaluation."""

    if not resume:
        return list(doc_ids), set()

    eligible_doc_ids = (
        doc_ids
        if restrict_processed_to_eligible
        else None
    )

    processed_ids = load_processed_ids(
        csv_path,
        eligible_doc_ids=eligible_doc_ids,
    )

    doc_ids_to_run = [
        doc_id
        for doc_id in doc_ids
        if doc_id not in processed_ids
    ]

    return doc_ids_to_run, processed_ids


def persist_result(
    *,
    csv_file,
    pred_file,
    result,
    dataset,
    engine,
    shot_variant,
    model_name,
    seed,
    dtype,
    fewshot_example_ids=None,
):
    """Persist one evaluated document using dataset-specific output fields."""

    drift = None

    if dataset == "cord":
        drift = compute_pred_drift_stats(
            result["pred_kvp"]
        )

        csv_row = format_cord_metrics_row(
            result,
            drift,
        )
    else:
        csv_row = format_metrics_row(
            result
        )

    write_and_sync(
        csv_file,
        csv_row,
    )

    record = build_prediction_record(
        result,
        engine=engine,
        shot_variant=shot_variant,
        model_name=model_name,
        seed=seed,
        dtype=dtype,
        fewshot_example_ids=(
            fewshot_example_ids
        ),
        drift=drift,
    )

    write_jsonl_record(
        pred_file,
        record,
        ensure_ascii=(dataset != "cord"),
    )


def run_document_batches(
    *,
    doc_ids,
    text_dict,
    gt_dict,
    build_prompt,
    tokenizer,
    model,
    is_encoder_decoder,
    device,
    system_message,
    batch_size,
    csv_file,
    pred_file,
    dataset,
    engine,
    shot_variant,
    model_name,
    seed,
    dtype,
    fewshot_example_ids=None,
):
    """Evaluate and persist all requested document batches."""

    for start in range(
        0,
        len(doc_ids),
        batch_size,
    ):
        batch_ids = doc_ids[
            start:start + batch_size
        ]

        batch_texts = [
            text_dict[doc_id]
            for doc_id in batch_ids
        ]

        batch_gt_kvp = [
            gt_dict[doc_id]
            for doc_id in batch_ids
        ]

        results = evaluate_batch(
            batch_ids=batch_ids,
            batch_texts=batch_texts,
            batch_gt_kvp=batch_gt_kvp,
            build_prompt=build_prompt,
            tokenizer=tokenizer,
            model=model,
            is_encoder_decoder=(
                is_encoder_decoder
            ),
            device=device,
            system_message=system_message,
        )

        for result in results:
            persist_result(
                csv_file=csv_file,
                pred_file=pred_file,
                result=result,
                dataset=dataset,
                engine=engine,
                shot_variant=shot_variant,
                model_name=model_name,
                seed=seed,
                dtype=dtype,
                fewshot_example_ids=(
                    fewshot_example_ids
                ),
            )


def prepare_run_outputs(
    *,
    results_base,
    dataset,
    engine,
    shot_variant,
    model_name,
    resume,
):
    """Prepare output paths and return the appropriate CSV header."""

    (
        output_dir,
        csv_path,
        pred_path,
    ) = build_output_paths(
        results_base,
        dataset,
        engine,
        shot_variant,
        model_name,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepare_output_files(
        csv_path,
        pred_path,
        resume=resume,
    )

    header = (
        CORD_METRICS_HEADER
        if dataset == "cord"
        else METRICS_HEADER
    )

    return (
        output_dir,
        csv_path,
        pred_path,
        header,
    )


def summarize_run(
    csv_path,
    *,
    dataset,
):
    """Compute the benchmark macro summary for one completed run."""

    return compute_macro_summary(
        csv_path,
        exclude_zero_gt=(dataset == "funsd"),
    )
