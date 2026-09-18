from pixels_to_pairs.experiments.runner import (
    METRICS_HEADER,
    compute_macro_summary,
    load_processed_ids,
    prepare_output_files,
    select_common_doc_ids,
)


def test_metrics_header_matches_benchmark_schema():
    assert METRICS_HEADER == (
        "doc_id,num_gt_keys,matched_keys,key_recall,"
        "exact_match_rate,value_f1,"
        "prompt_tokens_no_trunc,prompt_tokens_trunc,"
        "prompt_truncated,generated_tokens,"
        "output_hit_max_new_tokens\n"
    )


def test_select_common_doc_ids_returns_sorted_intersection():
    gt_dict = {
        "doc3": [{"key": "c", "value": "3"}],
        "doc1": [{"key": "a", "value": "1"}],
        "doc2": [{"key": "b", "value": "2"}],
    }

    text_dict = {
        "doc2": "text 2",
        "doc1": "text 1",
        "doc4": "text 4",
    }

    assert select_common_doc_ids(
        gt_dict,
        text_dict,
    ) == [
        "doc1",
        "doc2",
    ]


def test_select_common_doc_ids_can_exclude_empty_gt():
    gt_dict = {
        "doc1": [{"key": "a", "value": "1"}],
        "doc2": [],
        "doc3": [{"key": "c", "value": "3"}],
    }

    text_dict = {
        "doc1": "text 1",
        "doc2": "text 2",
        "doc3": "text 3",
    }

    assert select_common_doc_ids(
        gt_dict,
        text_dict,
        exclude_empty_gt=True,
    ) == [
        "doc1",
        "doc3",
    ]


def test_select_common_doc_ids_applies_max_docs_after_filtering():
    gt_dict = {
        "doc1": [],
        "doc2": [{"key": "b", "value": "2"}],
        "doc3": [{"key": "c", "value": "3"}],
        "doc4": [{"key": "d", "value": "4"}],
    }

    text_dict = {
        doc_id: "text"
        for doc_id in gt_dict
    }

    assert select_common_doc_ids(
        gt_dict,
        text_dict,
        exclude_empty_gt=True,
        max_docs=2,
    ) == [
        "doc2",
        "doc3",
    ]


def test_load_processed_ids_reads_existing_metrics_csv(tmp_path):
    csv_path = tmp_path / "metrics.csv"

    csv_path.write_text(
        METRICS_HEADER
        + "doc1,2,2,1.0,0.5,0.8,100,100,0,20,0\n"
        + "doc2,3,2,0.5,0.2,0.4,110,110,0,30,0\n",
        encoding="utf-8",
    )

    assert load_processed_ids(
        csv_path
    ) == {
        "doc1",
        "doc2",
    }


def test_load_processed_ids_can_restrict_to_eligible_docs(tmp_path):
    csv_path = tmp_path / "metrics.csv"

    csv_path.write_text(
        METRICS_HEADER
        + "doc1,2,2,1.0,0.5,0.8,100,100,0,20,0\n"
        + "doc2,3,2,0.5,0.2,0.4,110,110,0,30,0\n",
        encoding="utf-8",
    )

    assert load_processed_ids(
        csv_path,
        eligible_doc_ids={"doc2", "doc3"},
    ) == {
        "doc2",
    }


def test_load_processed_ids_missing_file_returns_empty_set(tmp_path):
    assert load_processed_ids(
        tmp_path / "missing.csv"
    ) == set()


def test_prepare_output_files_removes_outputs_when_not_resuming(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    pred_path = tmp_path / "predictions.jsonl"

    csv_path.write_text("csv", encoding="utf-8")
    pred_path.write_text("jsonl", encoding="utf-8")

    prepare_output_files(
        csv_path,
        pred_path,
        resume=False,
    )

    assert not csv_path.exists()
    assert not pred_path.exists()


def test_prepare_output_files_preserves_outputs_when_resuming(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    pred_path = tmp_path / "predictions.jsonl"

    csv_path.write_text("csv", encoding="utf-8")
    pred_path.write_text("jsonl", encoding="utf-8")

    prepare_output_files(
        csv_path,
        pred_path,
        resume=True,
    )

    assert csv_path.exists()
    assert pred_path.exists()


def test_compute_macro_summary_averages_document_metrics(tmp_path):
    csv_path = tmp_path / "metrics.csv"

    csv_path.write_text(
        METRICS_HEADER
        + "doc1,2,2,1.0,0.5,0.8,100,100,0,20,0\n"
        + "doc2,4,2,0.5,0.25,0.4,110,110,0,30,0\n",
        encoding="utf-8",
    )

    summary = compute_macro_summary(csv_path)

    assert summary == {
        "n_docs": 2,
        "key_recall": 0.75,
        "exact_match_rate": 0.375,
        "value_f1": 0.6000000000000001,
    }


def test_compute_macro_summary_can_exclude_zero_gt_docs(tmp_path):
    csv_path = tmp_path / "metrics.csv"

    csv_path.write_text(
        METRICS_HEADER
        + "doc1,2,2,1.0,0.5,0.8,100,100,0,20,0\n"
        + "doc2,0,0,0.0,0.0,0.0,110,110,0,30,0\n",
        encoding="utf-8",
    )

    summary = compute_macro_summary(
        csv_path,
        exclude_zero_gt=True,
    )

    assert summary == {
        "n_docs": 1,
        "key_recall": 1.0,
        "exact_match_rate": 0.5,
        "value_f1": 0.8,
    }


def test_compute_macro_summary_missing_file_returns_zeros(tmp_path):
    summary = compute_macro_summary(
        tmp_path / "missing.csv"
    )

    assert summary == {
        "n_docs": 0,
        "key_recall": 0.0,
        "exact_match_rate": 0.0,
        "value_f1": 0.0,
    }
