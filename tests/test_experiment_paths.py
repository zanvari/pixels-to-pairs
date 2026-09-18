from pathlib import Path

import pytest

from pixels_to_pairs.experiments.paths import (
    resolve_results_dir,
    resolve_text_dir,
)


@pytest.mark.parametrize(
    ("engine", "expected"),
    [
        (
            "gold_text_spatial",
            "funsd/funsd_gold_text_spatial",
        ),
        (
            "gold_text",
            "funsd/funsd_gold_text",
        ),
        (
            "tesseract",
            "funsd/funsd_ocr/tesseract",
        ),
        (
            "paddleocr",
            "funsd/funsd_ocr/paddleocr",
        ),
        (
            "easyocr",
            "funsd/funsd_ocr/easyocr",
        ),
    ],
)
def test_resolve_funsd_text_dir(
    engine,
    expected,
):
    assert resolve_text_dir(
        "/data",
        "funsd",
        engine,
    ) == Path("/data") / expected


@pytest.mark.parametrize(
    "dataset",
    [
        "sroie",
        "cord",
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        "gold_text",
        "tesseract",
        "paddleocr",
        "easyocr",
    ],
)
def test_resolve_standard_text_dirs(
    dataset,
    engine,
):
    if engine == "gold_text":
        expected = (
            Path("/data")
            / dataset
            / f"{dataset}_gold_text"
        )
    else:
        expected = (
            Path("/data")
            / dataset
            / f"{dataset}_ocr"
            / engine
        )

    assert resolve_text_dir(
        "/data",
        dataset,
        engine,
    ) == expected


def test_resolve_text_dir_rejects_invalid_engine():
    with pytest.raises(
        ValueError,
        match="Unsupported engine",
    ):
        resolve_text_dir(
            "/data",
            "cord",
            "gold_text_spatial",
        )


@pytest.mark.parametrize(
    "dataset",
    [
        "funsd",
        "sroie",
        "cord",
    ],
)
def test_resolve_results_dir(
    dataset,
):
    assert resolve_results_dir(
        "/results",
        dataset,
    ) == (
        Path("/results")
        / dataset
    )


def test_resolve_results_dir_rejects_unknown_dataset():
    with pytest.raises(
        ValueError,
        match="Unsupported dataset",
    ):
        resolve_results_dir(
            "/results",
            "unknown",
        )
