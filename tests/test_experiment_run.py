from pathlib import Path
from unittest.mock import patch

import pytest

from pixels_to_pairs.experiments.run import (
    prepare_dataset_run,
    run_model_experiment,
)


def _prepared(dataset):
    return {
        "dataset": dataset,
        "text_dir": Path("/data/text"),
        "data": {
            "gt_dict": {
                "doc1": [
                    {
                        "key": "k",
                        "value": "v",
                    }
                ],
            },
            "text_dict": {
                "doc1": "text",
            },
            "doc_ids": ["doc1"],
        },
        "prompt_builder": lambda text: text,
        "fewshot_example_ids": None,
    }


def test_prepare_funsd_run_uses_funsd_adapter():
    with (
        patch(
            "pixels_to_pairs.experiments.run."
            "resolve_text_dir",
            return_value=Path("/texts"),
        ),
        patch(
            "pixels_to_pairs.experiments.run."
            "load_funsd_run_data",
            return_value={"doc_ids": ["a"]},
        ) as loader,
        patch(
            "pixels_to_pairs.experiments.run."
            "build_funsd_run_prompt",
            return_value="builder",
        ),
    ):
        prepared = prepare_dataset_run(
            dataset="FUNSD",
            engine="gold_text_spatial",
            shot_variant="0shot",
            data_root="/data",
            gt_dir="/gt",
        )

    loader.assert_called_once_with(
        "/gt",
        Path("/texts"),
        max_docs=None,
    )

    assert prepared["dataset"] == "funsd"
    assert prepared["prompt_builder"] == "builder"


def test_prepare_sroie_keeps_example_ids():
    with (
        patch(
            "pixels_to_pairs.experiments.run."
            "resolve_text_dir",
            return_value=Path("/texts"),
        ),
        patch(
            "pixels_to_pairs.experiments.run."
            "load_sroie_run_data",
            return_value={"doc_ids": ["a"]},
        ),
        patch(
            "pixels_to_pairs.experiments.run."
            "build_sroie_run_prompt",
            return_value=(
                "builder",
                ["e1", "e2"],
            ),
        ),
    ):
        prepared = prepare_dataset_run(
            dataset="sroie",
            engine="gold_text",
            shot_variant="2shot",
            data_root="/data",
            gt_dir="/gt",
            example_files_dir="/examples",
        )

    assert prepared[
        "fewshot_example_ids"
    ] == ["e1", "e2"]


def test_prepare_cord_defaults_to_100_docs():
    with (
        patch(
            "pixels_to_pairs.experiments.run."
            "resolve_text_dir",
            return_value=Path("/texts"),
        ),
        patch(
            "pixels_to_pairs.experiments.run."
            "load_cord_run_data",
            return_value={"doc_ids": ["a"]},
        ) as loader,
        patch(
            "pixels_to_pairs.experiments.run."
            "build_cord_run_prompt",
            return_value="builder",
        ),
    ):
        prepare_dataset_run(
            dataset="cord",
            engine="gold_text",
            shot_variant="0shot",
            data_root="/data",
            gt_dir="/gt",
        )

    loader.assert_called_once_with(
        "/gt",
        Path("/texts"),
        max_docs=100,
    )


@pytest.mark.parametrize(
    "dataset",
    [
        "funsd",
        "sroie",
    ],
)
def test_model_load_failure_is_skipped_for_funsd_and_sroie(
    tmp_path,
    dataset,
):
    with (
        patch(
            "pixels_to_pairs.experiments.run."
            "prepare_dataset_run",
            return_value=_prepared(dataset),
        ),
        patch(
            "pixels_to_pairs.experiments.run."
            "build_model_and_tokenizer",
            side_effect=RuntimeError("load failed"),
        ),
        patch(
            "pixels_to_pairs.experiments.run."
            "cleanup_cuda",
        ),
    ):
        result = run_model_experiment(
            dataset=dataset,
            engine="gold_text",
            shot_variant="0shot",
            model_name="model/name",
            family="causal",
            data_root="/data",
            gt_dir="/gt",
            results_root=tmp_path,
            device="cpu",
        )

    assert result["status"] == (
        "model_load_failed"
    )


def test_cord_model_load_failure_propagates(
    tmp_path,
):
    with (
        patch(
            "pixels_to_pairs.experiments.run."
            "prepare_dataset_run",
            return_value=_prepared("cord"),
        ),
        patch(
            "pixels_to_pairs.experiments.run."
            "build_model_and_tokenizer",
            side_effect=RuntimeError("load failed"),
        ),
        patch(
            "pixels_to_pairs.experiments.run."
            "cleanup_cuda",
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="load failed",
        ):
            run_model_experiment(
                dataset="cord",
                engine="gold_text",
                shot_variant="0shot",
                model_name="model/name",
                family="causal",
                data_root="/data",
                gt_dir="/gt",
                results_root=tmp_path,
                device="cpu",
            )


def test_completed_resume_does_not_load_model(
    tmp_path,
):
    with (
        patch(
            "pixels_to_pairs.experiments.run."
            "prepare_dataset_run",
            return_value=_prepared("funsd"),
        ),
        patch(
            "pixels_to_pairs.experiments.run."
            "remaining_doc_ids",
            return_value=(
                [],
                {"doc1"},
            ),
        ),
        patch(
            "pixels_to_pairs.experiments.run."
            "build_model_and_tokenizer",
        ) as model_loader,
    ):
        result = run_model_experiment(
            dataset="funsd",
            engine="gold_text_spatial",
            shot_variant="0shot",
            model_name="model/name",
            family="causal",
            data_root="/data",
            gt_dir="/gt",
            results_root=tmp_path,
            device="cpu",
        )

    model_loader.assert_not_called()
    assert result["status"] == "complete"


def test_successful_run_calls_batch_execution(
    tmp_path,
):
    fake_model = object()
    fake_tokenizer = object()

    with (
        patch(
            "pixels_to_pairs.experiments.run."
            "prepare_dataset_run",
            return_value=_prepared("funsd"),
        ),
        patch(
            "pixels_to_pairs.experiments.run."
            "build_model_and_tokenizer",
            return_value=(
                fake_tokenizer,
                fake_model,
                False,
            ),
        ),
        patch(
            "pixels_to_pairs.experiments.run."
            "preferred_dtype",
            return_value="torch.float32",
        ),
        patch(
            "pixels_to_pairs.experiments.run."
            "run_document_batches",
        ) as run_batches,
        patch(
            "pixels_to_pairs.experiments.run."
            "cleanup_cuda",
        ) as cleanup,
        patch(
            "pixels_to_pairs.experiments.run."
            "write_and_sync",
            side_effect=lambda f, text: f.write(text),
        ),
    ):
        result = run_model_experiment(
            dataset="funsd",
            engine="gold_text_spatial",
            shot_variant="0shot",
            model_name="model/name",
            family="causal",
            data_root="/data",
            gt_dir="/gt",
            results_root=tmp_path,
            device="cpu",
        )

    run_batches.assert_called_once()
    cleanup.assert_called_once()
    assert result["status"] == "completed"
