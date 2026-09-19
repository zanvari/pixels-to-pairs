"""Top-level orchestration for benchmark model runs."""


from pixels_to_pairs.experiments.config import (
    validate_experiment,
)
from pixels_to_pairs.experiments.datasets import (
    build_cord_run_prompt,
    build_funsd_run_prompt,
    build_sroie_run_prompt,
    load_cord_run_data,
    load_funsd_run_data,
    load_sroie_run_data,
)
from pixels_to_pairs.experiments.execution import (
    prepare_run_outputs,
    remaining_doc_ids,
    run_document_batches,
    summarize_run,
)
from pixels_to_pairs.experiments.paths import (
    resolve_results_dir,
    resolve_text_dir,
)
from pixels_to_pairs.experiments.persistence import (
    write_and_sync,
)
from pixels_to_pairs.inference.generation import (
    build_model_and_tokenizer,
    cleanup_cuda,
    preferred_dtype,
)


DEFAULT_SEED = 0
DEFAULT_BATCH_SIZE = 1
DEFAULT_SYSTEM_MESSAGE = (
    "You are a precise information extraction system."
)


def prepare_dataset_run(
    *,
    dataset,
    engine,
    shot_variant,
    data_root,
    gt_dir,
    example_files_dir=None,
    max_docs=None,
):
    """Load dataset inputs and bind the appropriate prompt builder."""

    dataset = dataset.lower()

    validate_experiment(
        dataset,
        engine,
        shot_variant,
    )

    text_dir = resolve_text_dir(
        data_root,
        dataset,
        engine,
        shot_variant,
    )

    if dataset == "funsd":
        data = load_funsd_run_data(
            gt_dir,
            text_dir,
            max_docs=max_docs,
        )

        prompt_builder = (
            build_funsd_run_prompt(
                shot_variant,
                example_files_dir=(
                    example_files_dir
                ),
            )
        )

        fewshot_example_ids = None

    elif dataset == "sroie":
        data = load_sroie_run_data(
            gt_dir,
            text_dir,
            max_docs=max_docs,
        )

        (
            prompt_builder,
            fewshot_example_ids,
        ) = build_sroie_run_prompt(
            shot_variant,
            example_files_dir=(
                example_files_dir
            ),
        )

    else:
        cord_max_docs = (
            100
            if max_docs is None
            else max_docs
        )

        data = load_cord_run_data(
            gt_dir,
            text_dir,
            max_docs=cord_max_docs,
        )

        prompt_builder = (
            build_cord_run_prompt(
                shot_variant,
                example_files_dir=(
                    example_files_dir
                ),
            )
        )

        fewshot_example_ids = None

    return {
        "dataset": dataset,
        "text_dir": text_dir,
        "data": data,
        "prompt_builder": prompt_builder,
        "fewshot_example_ids": (
            fewshot_example_ids
        ),
    }


def run_model_experiment(
    *,
    dataset,
    engine,
    shot_variant,
    model_name,
    family,
    data_root,
    gt_dir,
    results_root,
    example_files_dir=None,
    max_docs=None,
    resume=True,
    device="cuda",
    hf_token=None,
    seed=DEFAULT_SEED,
    batch_size=DEFAULT_BATCH_SIZE,
    system_message=DEFAULT_SYSTEM_MESSAGE,
):
    """Run one benchmark model/dataset/condition configuration."""

    prepared = prepare_dataset_run(
        dataset=dataset,
        engine=engine,
        shot_variant=shot_variant,
        data_root=data_root,
        gt_dir=gt_dir,
        example_files_dir=(
            example_files_dir
        ),
        max_docs=max_docs,
    )

    dataset = prepared["dataset"]
    data = prepared["data"]

    results_base = resolve_results_dir(
        results_root,
        dataset,
    )

    (
        output_dir,
        csv_path,
        pred_path,
        header,
    ) = prepare_run_outputs(
        results_base=results_base,
        dataset=dataset,
        engine=engine,
        shot_variant=shot_variant,
        model_name=model_name,
        resume=resume,
    )

    restrict_processed = (
        dataset == "funsd"
    )

    (
        doc_ids_to_run,
        processed_ids,
    ) = remaining_doc_ids(
        data["doc_ids"],
        csv_path,
        resume=resume,
        restrict_processed_to_eligible=(
            restrict_processed
        ),
    )

    if not doc_ids_to_run:
        return {
            "status": "complete",
            "dataset": dataset,
            "engine": engine,
            "shot_variant": shot_variant,
            "model": model_name,
            "output_dir": output_dir,
            "csv_path": csv_path,
            "pred_path": pred_path,
            "processed_ids": processed_ids,
            "summary": summarize_run(
                csv_path,
                dataset=dataset,
            ),
        }

    tokenizer = None
    model = None

    try:
        try:
            (
                tokenizer,
                model,
                is_encoder_decoder,
            ) = build_model_and_tokenizer(
                model_name,
                family,
                device,
                hf_token=hf_token,
            )

        except Exception:
            # Preserve notebook behavior: FUNSD and SROIE
            # skip a model that cannot be loaded, while CORD
            # surfaces the loading failure.
            if dataset in {
                "funsd",
                "sroie",
            }:
                return {
                    "status": "model_load_failed",
                    "dataset": dataset,
                    "engine": engine,
                    "shot_variant": shot_variant,
                    "model": model_name,
                    "output_dir": output_dir,
                    "csv_path": csv_path,
                    "pred_path": pred_path,
                }

            raise

        dtype = preferred_dtype(
            device
        )

        csv_exists = csv_path.exists()

        csv_mode = (
            "a"
            if resume and csv_exists
            else "w"
        )

        with (
            csv_path.open(
                csv_mode,
                encoding="utf-8",
            ) as csv_file,
            pred_path.open(
                "a",
                encoding="utf-8",
            ) as pred_file,
        ):
            if csv_mode == "w":
                write_and_sync(
                    csv_file,
                    header,
                )

            run_document_batches(
                doc_ids=doc_ids_to_run,
                text_dict=data["text_dict"],
                gt_dict=data["gt_dict"],
                build_prompt=(
                    prepared[
                        "prompt_builder"
                    ]
                ),
                tokenizer=tokenizer,
                model=model,
                is_encoder_decoder=(
                    is_encoder_decoder
                ),
                device=device,
                system_message=(
                    system_message
                ),
                batch_size=batch_size,
                csv_file=csv_file,
                pred_file=pred_file,
                dataset=dataset,
                engine=engine,
                shot_variant=shot_variant,
                model_name=model_name,
                seed=seed,
                dtype=dtype,
                fewshot_example_ids=(
                    prepared[
                        "fewshot_example_ids"
                    ]
                ),
            )

    finally:
        # Keep model references local to this run and mirror
        # the notebook's explicit CUDA cleanup between models.
        tokenizer = None
        model = None
        cleanup_cuda()

    return {
        "status": "completed",
        "dataset": dataset,
        "engine": engine,
        "shot_variant": shot_variant,
        "model": model_name,
        "output_dir": output_dir,
        "csv_path": csv_path,
        "pred_path": pred_path,
        "processed_ids": processed_ids,
        "summary": summarize_run(
            csv_path,
            dataset=dataset,
        ),
    }
