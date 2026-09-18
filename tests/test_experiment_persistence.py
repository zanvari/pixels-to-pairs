import json
from unittest.mock import patch

from pixels_to_pairs.experiments.persistence import (
    CORD_METRICS_HEADER,
    build_prediction_record,
    format_cord_metrics_row,
    format_metrics_row,
    write_jsonl_record,
)
from pixels_to_pairs.experiments.runner import (
    METRICS_HEADER,
)


def make_result():
    return {
        "doc_id": "doc1",
        "gt_kvp": [
            {
                "key": "invoice",
                "value": "123",
            }
        ],
        "pred_kvp": [
            {
                "key": "invoice",
                "value": "123",
            }
        ],
        "raw_output": '{"kvp": []}',
        "metrics": {
            "num_gt_keys": 1,
            "matched_keys": 1,
            "key_recall": 1.0,
            "exact_matches": 1,
            "exact_match_rate": 1.0,
            "value_f1": 1.0,
        },
        "prompt_tokens_no_trunc": 100,
        "prompt_tokens_trunc": 100,
        "prompt_truncated": False,
        "generated_tokens": 25,
        "output_hit_max_new_tokens": False,
    }


def test_standard_metrics_header_is_unchanged():
    assert METRICS_HEADER == (
        "doc_id,num_gt_keys,matched_keys,key_recall,"
        "exact_match_rate,value_f1,"
        "prompt_tokens_no_trunc,prompt_tokens_trunc,"
        "prompt_truncated,generated_tokens,"
        "output_hit_max_new_tokens\n"
    )


def test_cord_header_extends_standard_header():
    assert CORD_METRICS_HEADER == (
        "doc_id,num_gt_keys,matched_keys,key_recall,"
        "exact_match_rate,value_f1,"
        "prompt_tokens_no_trunc,prompt_tokens_trunc,"
        "prompt_truncated,generated_tokens,"
        "output_hit_max_new_tokens,"
        "pred_num_pairs,pred_unique_keys,"
        "pred_dup_key_frac,"
        "pred_single_token_key_frac,"
        "pred_numeric_key_frac,"
        "pred_qty_key_frac\n"
    )


def test_format_standard_metrics_row():
    assert format_metrics_row(
        make_result()
    ) == (
        "doc1,1,1,"
        "1.0000000000,"
        "1.0000000000,"
        "1.0000000000,"
        "100,100,0,25,0\n"
    )


def test_format_cord_metrics_row():
    drift = {
        "pred_num_pairs": 4,
        "pred_unique_keys": 3,
        "pred_dup_key_frac": 0.25,
        "pred_single_token_key_frac": 0.5,
        "pred_numeric_key_frac": 0.125,
        "pred_qty_key_frac": 0.75,
    }

    assert format_cord_metrics_row(
        make_result(),
        drift,
    ) == (
        "doc1,1,1,"
        "1.0000000000,"
        "1.0000000000,"
        "1.0000000000,"
        "100,100,0,25,0,"
        "4,3,"
        "0.2500,0.5000,"
        "0.1250,0.7500\n"
    )


def test_build_standard_prediction_record():
    record = build_prediction_record(
        make_result(),
        engine="paddleocr",
        shot_variant="0shot",
        model_name="model",
        seed=0,
        dtype="torch.float16",
    )

    assert record == {
        "doc_id": "doc1",
        "engine": "paddleocr",
        "shot_variant": "0shot",
        "model": "model",
        "gt_kvp": [
            {
                "key": "invoice",
                "value": "123",
            }
        ],
        "pred_kvp": [
            {
                "key": "invoice",
                "value": "123",
            }
        ],
        "raw_output": '{"kvp": []}',
        "metrics": make_result()["metrics"],
        "prompt_tokens_no_trunc": 100,
        "prompt_tokens_trunc": 100,
        "prompt_truncated": False,
        "generated_tokens": 25,
        "output_hit_max_new_tokens": False,
        "seed": 0,
        "dtype": "torch.float16",
    }


def test_sroie_record_includes_fewshot_ids():
    record = build_prediction_record(
        make_result(),
        engine="gold_text",
        shot_variant="2shot",
        model_name="model",
        seed=0,
        dtype="torch.float16",
        fewshot_example_ids=[
            "example1",
            "example2",
        ],
    )

    assert record["fewshot_example_ids"] == [
        "example1",
        "example2",
    ]


def test_cord_record_includes_drift():
    drift = {
        "pred_num_pairs": 1,
    }

    record = build_prediction_record(
        make_result(),
        engine="gold_text",
        shot_variant="0shot",
        model_name="model",
        seed=0,
        dtype="torch.float16",
        drift=drift,
    )

    assert record["drift"] == drift


def test_standard_record_does_not_add_optional_fields():
    record = build_prediction_record(
        make_result(),
        engine="gold_text",
        shot_variant="0shot",
        model_name="model",
        seed=0,
        dtype="torch.float16",
    )

    assert "drift" not in record
    assert "fewshot_example_ids" not in record


def test_write_jsonl_record_writes_one_line(
    tmp_path,
):
    output_path = tmp_path / "predictions.jsonl"

    with output_path.open(
        "w+",
        encoding="utf-8",
    ) as file_obj:
        with patch(
            "pixels_to_pairs.experiments.persistence.os.fsync"
        ) as fsync:
            write_jsonl_record(
                file_obj,
                {
                    "doc_id": "doc1",
                    "value": "café",
                },
                ensure_ascii=False,
            )

        file_obj.seek(0)
        lines = file_obj.readlines()

    assert len(lines) == 1

    assert json.loads(lines[0]) == {
        "doc_id": "doc1",
        "value": "café",
    }

    assert "café" in lines[0]

    fsync.assert_called_once()
