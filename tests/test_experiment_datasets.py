from unittest.mock import patch

import pytest

from pixels_to_pairs.experiments.datasets import (
    build_cord_run_prompt,
    build_funsd_run_prompt,
    build_sroie_run_prompt,
    load_cord_run_data,
    load_funsd_run_data,
    load_sroie_run_data,
)


def test_funsd_excludes_zero_gt_before_max_docs():
    gt_dict = {
        "a": [],
        "b": [{"key": "k", "value": "v"}],
        "c": [{"key": "k", "value": "v"}],
    }

    text_dict = {
        "a": "A",
        "b": "B",
        "c": "C",
    }

    with (
        patch(
            "pixels_to_pairs.experiments.datasets."
            "load_funsd_gt_kvp",
            return_value=gt_dict,
        ),
        patch(
            "pixels_to_pairs.experiments.datasets."
            "load_funsd_texts",
            return_value=text_dict,
        ),
    ):
        data = load_funsd_run_data(
            "annotations",
            "texts",
            max_docs=1,
        )

    assert data["all_common_doc_ids"] == [
        "a",
        "b",
        "c",
    ]
    assert data["zero_gt_doc_ids"] == ["a"]

    # Filtering happens before the document cap.
    assert data["doc_ids"] == ["b"]


def test_funsd_uses_only_common_ids():
    with (
        patch(
            "pixels_to_pairs.experiments.datasets."
            "load_funsd_gt_kvp",
            return_value={
                "a": [{"key": "k", "value": "v"}],
                "b": [{"key": "k", "value": "v"}],
            },
        ),
        patch(
            "pixels_to_pairs.experiments.datasets."
            "load_funsd_texts",
            return_value={
                "b": "B",
                "c": "C",
            },
        ),
    ):
        data = load_funsd_run_data(
            "annotations",
            "texts",
        )

    assert data["doc_ids"] == ["b"]


def test_sroie_uses_sorted_intersection_and_cap():
    with (
        patch(
            "pixels_to_pairs.experiments.datasets."
            "load_sroie_gt_kvp",
            return_value={
                "c": [],
                "a": [],
                "b": [],
            },
        ),
        patch(
            "pixels_to_pairs.experiments.datasets."
            "load_sroie_texts",
            return_value={
                "c": "C",
                "b": "B",
                "x": "X",
            },
        ),
    ):
        data = load_sroie_run_data(
            "kvp",
            "texts",
            max_docs=1,
        )

    assert data["doc_ids"] == ["b"]


def test_cord_reports_intersection_sanity_counts():
    with (
        patch(
            "pixels_to_pairs.experiments.datasets."
            "load_cord_gt_kvp",
            return_value={
                "a": [{"key": "k", "value": "v"}],
                "b": [],
                "gt_only": [],
            },
        ),
        patch(
            "pixels_to_pairs.experiments.datasets."
            "load_cord_texts",
            return_value={
                "a": "A",
                "b": "B",
                "text_only": "T",
            },
        ),
    ):
        data = load_cord_run_data(
            "kvp",
            "texts",
            max_docs=None,
        )

    assert data["doc_ids"] == ["a", "b"]
    assert data["gt_doc_count"] == 3
    assert data["text_doc_count"] == 3
    assert data["intersection_count"] == 2
    assert data["gt_only_count"] == 1
    assert data["text_only_count"] == 1
    assert data["empty_gt_count"] == 1


def test_cord_rejects_empty_intersection():
    with (
        patch(
            "pixels_to_pairs.experiments.datasets."
            "load_cord_gt_kvp",
            return_value={
                "a": [{"key": "k", "value": "v"}],
            },
        ),
        patch(
            "pixels_to_pairs.experiments.datasets."
            "load_cord_texts",
            return_value={
                "b": "B",
            },
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="No intersecting doc_ids",
        ):
            load_cord_run_data(
                "kvp",
                "texts",
            )


def test_cord_rejects_all_empty_gt():
    with (
        patch(
            "pixels_to_pairs.experiments.datasets."
            "load_cord_gt_kvp",
            return_value={
                "a": [],
                "b": [],
            },
        ),
        patch(
            "pixels_to_pairs.experiments.datasets."
            "load_cord_texts",
            return_value={
                "a": "A",
                "b": "B",
            },
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="All GT KVP lists are empty",
        ):
            load_cord_run_data(
                "kvp",
                "texts",
            )


def test_cord_applies_max_docs_after_sanity():
    with (
        patch(
            "pixels_to_pairs.experiments.datasets."
            "load_cord_gt_kvp",
            return_value={
                "a": [{"key": "k", "value": "v"}],
                "b": [{"key": "k", "value": "v"}],
            },
        ),
        patch(
            "pixels_to_pairs.experiments.datasets."
            "load_cord_texts",
            return_value={
                "a": "A",
                "b": "B",
            },
        ),
    ):
        data = load_cord_run_data(
            "kvp",
            "texts",
            max_docs=1,
        )

    assert data["intersection_count"] == 2
    assert data["doc_ids"] == ["a"]


def test_funsd_zero_shot_does_not_load_examples():
    with patch(
        "pixels_to_pairs.experiments.datasets."
        "load_funsd_examples"
    ) as loader:
        prompt_builder = build_funsd_run_prompt(
            "0shot"
        )

        prompt = prompt_builder("document")

    loader.assert_not_called()
    assert isinstance(prompt, str)


def test_cord_zero_shot_does_not_load_examples():
    with patch(
        "pixels_to_pairs.experiments.datasets."
        "load_cord_examples"
    ) as loader:
        prompt_builder = build_cord_run_prompt(
            "0shot"
        )

        prompt = prompt_builder("document")

    loader.assert_not_called()
    assert isinstance(prompt, str)


def test_sroie_zero_shot_has_no_example_ids():
    with patch(
        "pixels_to_pairs.experiments.datasets."
        "load_sroie_examples"
    ) as loader:
        prompt_builder, example_ids = (
            build_sroie_run_prompt(
                "0shot"
            )
        )

        prompt = prompt_builder("document")

    loader.assert_not_called()
    assert example_ids is None
    assert isinstance(prompt, str)


def test_sroie_fewshot_ids_follow_frozen_order():
    examples = [
        {
            "doc_id": "first",
            "doc_text": "one",
            "kvp": [],
        },
        {
            "doc_id": "second",
            "doc_text": "two",
            "kvp": [],
        },
        {
            "doc_id": "third",
            "doc_text": "three",
            "kvp": [],
        },
    ]

    with patch(
        "pixels_to_pairs.experiments.datasets."
        "load_sroie_examples",
        return_value=examples,
    ):
        _, example_ids = build_sroie_run_prompt(
            "2shot",
            example_files_dir="examples",
        )

    assert example_ids == [
        "first",
        "second",
    ]
