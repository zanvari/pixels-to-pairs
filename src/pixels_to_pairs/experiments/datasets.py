"""Dataset-specific benchmark experiment adapters."""

from functools import partial
from pathlib import Path

from pixels_to_pairs.data.cord import (
    load_cord_gt_kvp,
    load_doc_texts as load_cord_texts,
)
from pixels_to_pairs.data.funsd import (
    load_doc_texts as load_funsd_texts,
    load_funsd_gt_kvp,
)
from pixels_to_pairs.data.sroie import (
    load_doc_texts as load_sroie_texts,
    load_sroie_gt_kvp,
)
from pixels_to_pairs.experiments.runner import (
    select_common_doc_ids,
)
from pixels_to_pairs.prompting.cord import (
    build_prompt as build_cord_prompt,
    load_cord_examples,
)
from pixels_to_pairs.prompting.funsd import (
    build_prompt as build_funsd_prompt,
    load_funsd_examples,
)
from pixels_to_pairs.prompting.sroie import (
    build_prompt as build_sroie_prompt,
    load_sroie_examples,
)


def load_funsd_run_data(
    annotation_dir,
    text_dir,
    *,
    max_docs=None,
):
    """Load the evaluable FUNSD document subset."""

    gt_dict = load_funsd_gt_kvp(
        annotation_dir
    )

    text_dict = load_funsd_texts(
        text_dir
    )

    all_common_doc_ids = (
        select_common_doc_ids(
            gt_dict,
            text_dict,
        )
    )

    zero_gt_doc_ids = [
        doc_id
        for doc_id in all_common_doc_ids
        if len(gt_dict.get(doc_id, [])) == 0
    ]

    doc_ids = select_common_doc_ids(
        gt_dict,
        text_dict,
        exclude_empty_gt=True,
        max_docs=max_docs,
    )

    return {
        "gt_dict": gt_dict,
        "text_dict": text_dict,
        "doc_ids": doc_ids,
        "all_common_doc_ids": all_common_doc_ids,
        "zero_gt_doc_ids": zero_gt_doc_ids,
    }


def load_sroie_run_data(
    kvp_dir,
    text_dir,
    *,
    max_docs=None,
):
    """Load the SROIE documents used by one benchmark run."""

    gt_dict = load_sroie_gt_kvp(
        kvp_dir
    )

    text_dict = load_sroie_texts(
        text_dir
    )

    doc_ids = select_common_doc_ids(
        gt_dict,
        text_dict,
        max_docs=max_docs,
    )

    return {
        "gt_dict": gt_dict,
        "text_dict": text_dict,
        "doc_ids": doc_ids,
    }


def load_cord_run_data(
    kvp_dir,
    text_dir,
    *,
    max_docs=100,
):
    """Load CORD data and apply the notebook sanity checks."""

    gt_dict = load_cord_gt_kvp(
        kvp_dir
    )

    text_dict = load_cord_texts(
        text_dir
    )

    gt_ids = set(gt_dict.keys())
    text_ids = set(text_dict.keys())
    intersection = sorted(
        gt_ids & text_ids
    )

    if not intersection:
        raise RuntimeError(
            "No intersecting doc_ids between GT and text. "
            "Check that filenames match and that the "
            "correct text and KVP directories were selected."
        )

    empty_gt_count = sum(
        1
        for doc_id in intersection
        if len(gt_dict.get(doc_id, [])) == 0
    )

    if empty_gt_count == len(intersection):
        raise RuntimeError(
            "All GT KVP lists are empty for intersected "
            "CORD documents. Check the GT directory and "
            "schema mapping."
        )

    doc_ids = intersection

    if max_docs is not None:
        doc_ids = doc_ids[:max_docs]

    return {
        "gt_dict": gt_dict,
        "text_dict": text_dict,
        "doc_ids": doc_ids,
        "gt_doc_count": len(gt_ids),
        "text_doc_count": len(text_ids),
        "intersection_count": len(intersection),
        "gt_only_count": len(
            gt_ids - text_ids
        ),
        "text_only_count": len(
            text_ids - gt_ids
        ),
        "empty_gt_count": empty_gt_count,
    }


def build_funsd_run_prompt(
    shot_variant,
    *,
    example_files_dir=None,
):
    """Return a single-document FUNSD prompt builder."""

    examples = None

    if shot_variant != "0shot":
        examples = load_funsd_examples(
            Path(example_files_dir)
        )

    return partial(
        build_funsd_prompt,
        shot_variant=shot_variant,
        examples=examples,
    )


def build_sroie_run_prompt(
    shot_variant,
    *,
    example_files_dir=None,
):
    """Return the SROIE prompt builder and frozen example IDs."""

    examples = None
    example_ids = None

    if shot_variant != "0shot":
        examples = load_sroie_examples(
            Path(example_files_dir)
        )

        n_shots = int(
            shot_variant[0]
        )

        example_ids = [
            example["doc_id"]
            for example in examples[:n_shots]
        ]

    prompt_builder = partial(
        build_sroie_prompt,
        shot_variant=shot_variant,
        examples=examples,
    )

    return prompt_builder, example_ids


def build_cord_run_prompt(
    shot_variant,
    *,
    example_files_dir=None,
):
    """Return a single-document CORD prompt builder."""

    examples = None

    if shot_variant != "0shot":
        examples = load_cord_examples(
            Path(example_files_dir)
        )

    return partial(
        build_cord_prompt,
        shot_variant=shot_variant,
        examples=examples,
    )
