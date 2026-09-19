import json
from unittest.mock import patch

from pixels_to_pairs.experiments.execution import (
    build_output_paths,
    persist_result,
    prepare_run_outputs,
    remaining_doc_ids,
    run_document_batches,
    summarize_run,
)
from pixels_to_pairs.experiments.persistence import (
    CORD_METRICS_HEADER,
)
from pixels_to_pairs.experiments.runner import (
    METRICS_HEADER,
)


def _result():
    return {
        "doc_id": "doc1",
        "doc_text": "example",
        "gt_kvp": [
            {
                "key": "total",
                "value": "10",
            }
        ],
        "pred_kvp": [
            {
                "key": "total",
                "value": "10",
            }
        ],
        "raw_output": '{"kvp":[]}',
        "metrics": {
            "num_gt_keys": 1,
            "matched_keys": 1,
            "key_recall": 1.0,
            "exact_match_rate": 1.0,
            "value_f1": 1.0,
        },
        "prompt_tokens_no_trunc": 100,
        "prompt_tokens_trunc": 100,
        "prompt_truncated": False,
        "generated_tokens": 20,
        "output_hit_max_new_tokens": False,
    }


def test_build_output_paths_matches_notebook_names(
    tmp_path,
):
    output_dir, csv_path, pred_path = (
        build_output_paths(
            tmp_path,
            "funsd",
            "paddleocr",
            "3shot",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        )
    )

    expected_dir = (
        tmp_path
        / (
            "funsd_paddleocr_3shot_"
            "meta-llama_Meta-Llama-3-8B-Instruct"
        )
    )

    assert output_dir == expected_dir

    assert csv_path.name == (
        "metrics_paddleocr_3shot_"
        "meta-llama_Meta-Llama-3-8B-Instruct.csv"
    )

    assert pred_path.name == (
        "predictions_paddleocr_3shot_"
        "meta-llama_Meta-Llama-3-8B-Instruct.jsonl"
    )


def test_remaining_doc_ids_without_resume():
    remaining, processed = remaining_doc_ids(
        ["a", "b"],
        "unused.csv",
        resume=False,
    )

    assert remaining == ["a", "b"]
    assert processed == set()


def test_remaining_doc_ids_with_resume(
    tmp_path,
):
    csv_path = tmp_path / "metrics.csv"

    csv_path.write_text(
        METRICS_HEADER
        + "a,1,1,1,1,1,1,1,0,1,0\n",
        encoding="utf-8",
    )

    remaining, processed = remaining_doc_ids(
        ["a", "b"],
        csv_path,
        resume=True,
    )

    assert processed == {"a"}
    assert remaining == ["b"]


def test_funsd_resume_restricts_processed_ids(
    tmp_path,
):
    csv_path = tmp_path / "metrics.csv"

    csv_path.write_text(
        METRICS_HEADER
        + "old_doc,1,1,1,1,1,1,1,0,1,0\n"
        + "a,1,1,1,1,1,1,1,0,1,0\n",
        encoding="utf-8",
    )

    remaining, processed = remaining_doc_ids(
        ["a", "b"],
        csv_path,
        resume=True,
        restrict_processed_to_eligible=True,
    )

    assert processed == {"a"}
    assert remaining == ["b"]


def test_prepare_standard_outputs(
    tmp_path,
):
    (
        output_dir,
        csv_path,
        pred_path,
        header,
    ) = prepare_run_outputs(
        results_base=tmp_path,
        dataset="sroie",
        engine="gold_text",
        shot_variant="0shot",
        model_name="model/name",
        resume=True,
    )

    assert output_dir.exists()
    assert not csv_path.exists()
    assert not pred_path.exists()
    assert header == METRICS_HEADER


def test_prepare_cord_outputs_uses_drift_header(
    tmp_path,
):
    *_, header = prepare_run_outputs(
        results_base=tmp_path,
        dataset="cord",
        engine="gold_text",
        shot_variant="0shot",
        model_name="model/name",
        resume=True,
    )

    assert header == CORD_METRICS_HEADER


def test_prepare_outputs_removes_old_files_when_not_resuming(
    tmp_path,
):
    (
        _,
        csv_path,
        pred_path,
        _,
    ) = prepare_run_outputs(
        results_base=tmp_path,
        dataset="funsd",
        engine="gold_text",
        shot_variant="0shot",
        model_name="model/name",
        resume=True,
    )

    csv_path.write_text(
        "old",
        encoding="utf-8",
    )
    pred_path.write_text(
        "old",
        encoding="utf-8",
    )

    prepare_run_outputs(
        results_base=tmp_path,
        dataset="funsd",
        engine="gold_text",
        shot_variant="0shot",
        model_name="model/name",
        resume=False,
    )

    assert not csv_path.exists()
    assert not pred_path.exists()


def test_standard_result_persistence(
    tmp_path,
):
    csv_path = tmp_path / "metrics.csv"
    pred_path = tmp_path / "predictions.jsonl"

    with (
        csv_path.open(
            "w",
            encoding="utf-8",
        ) as csv_file,
        pred_path.open(
            "w",
            encoding="utf-8",
        ) as pred_file,
        patch(
            "pixels_to_pairs.experiments.persistence.os.fsync"
        ),
    ):
        csv_file.write(METRICS_HEADER)

        persist_result(
            csv_file=csv_file,
            pred_file=pred_file,
            result=_result(),
            dataset="funsd",
            engine="gold_text",
            shot_variant="0shot",
            model_name="model/name",
            seed=0,
            dtype="torch.float16",
        )

    csv_lines = csv_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(csv_lines) == 2
    assert len(csv_lines[1].split(",")) == 11

    record = json.loads(
        pred_path.read_text(
            encoding="utf-8"
        )
    )

    assert "drift" not in record
    assert "fewshot_example_ids" not in record


def test_cord_result_persistence_adds_drift(
    tmp_path,
):
    csv_path = tmp_path / "metrics.csv"
    pred_path = tmp_path / "predictions.jsonl"

    with (
        csv_path.open(
            "w",
            encoding="utf-8",
        ) as csv_file,
        pred_path.open(
            "w",
            encoding="utf-8",
        ) as pred_file,
        patch(
            "pixels_to_pairs.experiments.persistence.os.fsync"
        ),
    ):
        csv_file.write(CORD_METRICS_HEADER)

        persist_result(
            csv_file=csv_file,
            pred_file=pred_file,
            result=_result(),
            dataset="cord",
            engine="gold_text",
            shot_variant="0shot",
            model_name="model/name",
            seed=0,
            dtype="torch.float16",
        )

    csv_lines = csv_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(csv_lines[1].split(",")) == 17

    record = json.loads(
        pred_path.read_text(
            encoding="utf-8"
        )
    )

    assert "drift" in record
    assert record["drift"]["pred_num_pairs"] == 1


def test_sroie_result_persistence_keeps_example_ids(
    tmp_path,
):
    csv_path = tmp_path / "metrics.csv"
    pred_path = tmp_path / "predictions.jsonl"

    with (
        csv_path.open(
            "w",
            encoding="utf-8",
        ) as csv_file,
        pred_path.open(
            "w",
            encoding="utf-8",
        ) as pred_file,
        patch(
            "pixels_to_pairs.experiments.persistence.os.fsync"
        ),
    ):
        csv_file.write(METRICS_HEADER)

        persist_result(
            csv_file=csv_file,
            pred_file=pred_file,
            result=_result(),
            dataset="sroie",
            engine="gold_text",
            shot_variant="2shot",
            model_name="model/name",
            seed=0,
            dtype="torch.float16",
            fewshot_example_ids=[
                "example1",
                "example2",
            ],
        )

    record = json.loads(
        pred_path.read_text(
            encoding="utf-8"
        )
    )

    assert record["fewshot_example_ids"] == [
        "example1",
        "example2",
    ]


def test_summarize_funsd_excludes_zero_gt(
    tmp_path,
):
    csv_path = tmp_path / "metrics.csv"

    csv_path.write_text(
        METRICS_HEADER
        + "a,1,1,1.0,1.0,1.0,1,1,0,1,0\n"
        + "b,0,0,0.0,0.0,0.0,1,1,0,1,0\n",
        encoding="utf-8",
    )

    summary = summarize_run(
        csv_path,
        dataset="funsd",
    )

    assert summary["n_docs"] == 1
    assert summary["value_f1"] == 1.0


def test_summarize_sroie_keeps_all_rows(
    tmp_path,
):
    csv_path = tmp_path / "metrics.csv"

    csv_path.write_text(
        METRICS_HEADER
        + "a,1,1,1.0,1.0,1.0,1,1,0,1,0\n"
        + "b,0,0,0.0,0.0,0.0,1,1,0,1,0\n",
        encoding="utf-8",
    )

    summary = summarize_run(
        csv_path,
        dataset="sroie",
    )

    assert summary["n_docs"] == 2
    assert summary["value_f1"] == 0.5

def test_run_document_batches_evaluates_and_persists(
    tmp_path,
):
    csv_path = tmp_path / "metrics.csv"
    pred_path = tmp_path / "predictions.jsonl"

    batch_result = _result()

    with (
        csv_path.open(
            "w",
            encoding="utf-8",
        ) as csv_file,
        pred_path.open(
            "w",
            encoding="utf-8",
        ) as pred_file,
        patch(
            "pixels_to_pairs.experiments.execution.evaluate_batch",
            return_value=[batch_result],
        ) as evaluate,
        patch(
            "pixels_to_pairs.experiments.persistence.os.fsync"
        ),
    ):
        csv_file.write(METRICS_HEADER)

        run_document_batches(
            doc_ids=["doc1"],
            text_dict={
                "doc1": "example",
            },
            gt_dict={
                "doc1": [
                    {
                        "key": "total",
                        "value": "10",
                    }
                ],
            },
            build_prompt=lambda text: (
                f"PROMPT: {text}"
            ),
            tokenizer=object(),
            model=object(),
            is_encoder_decoder=False,
            device="cpu",
            system_message="system",
            batch_size=1,
            csv_file=csv_file,
            pred_file=pred_file,
            dataset="funsd",
            engine="gold_text",
            shot_variant="0shot",
            model_name="model/name",
            seed=0,
            dtype="torch.float32",
        )

    evaluate.assert_called_once()

    call = evaluate.call_args.kwargs

    assert call["batch_ids"] == ["doc1"]
    assert call["batch_texts"] == ["example"]

    assert call["batch_gt_kvp"] == [
        [
            {
                "key": "total",
                "value": "10",
            }
        ]
    ]

    csv_lines = csv_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(csv_lines) == 2

    records = pred_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(records) == 1

    record = json.loads(records[0])

    assert record["doc_id"] == "doc1"
    assert record["engine"] == "gold_text"
