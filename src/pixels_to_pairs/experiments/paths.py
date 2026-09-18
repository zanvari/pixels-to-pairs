"""Portable dataset and result path resolution."""

from pathlib import Path

from .config import validate_experiment


def resolve_text_dir(
    data_root,
    dataset: str,
    engine: str,
    shot: str = "0shot",
) -> Path:
    """Resolve the document-text directory for one benchmark condition."""

    dataset = dataset.lower()

    validate_experiment(
        dataset,
        engine,
        shot,
    )

    data_root = Path(data_root)

    if dataset == "funsd":
        paths = {
            "gold_text_spatial": (
                data_root
                / "funsd"
                / "funsd_gold_text_spatial"
            ),
            "gold_text": (
                data_root
                / "funsd"
                / "funsd_gold_text"
            ),
            "tesseract": (
                data_root
                / "funsd"
                / "funsd_ocr"
                / "tesseract"
            ),
            "paddleocr": (
                data_root
                / "funsd"
                / "funsd_ocr"
                / "paddleocr"
            ),
            "easyocr": (
                data_root
                / "funsd"
                / "funsd_ocr"
                / "easyocr"
            ),
        }

    elif dataset == "sroie":
        paths = {
            "gold_text": (
                data_root
                / "sroie"
                / "sroie_gold_text"
            ),
            "tesseract": (
                data_root
                / "sroie"
                / "sroie_ocr"
                / "tesseract"
            ),
            "paddleocr": (
                data_root
                / "sroie"
                / "sroie_ocr"
                / "paddleocr"
            ),
            "easyocr": (
                data_root
                / "sroie"
                / "sroie_ocr"
                / "easyocr"
            ),
        }

    else:
        paths = {
            "gold_text": (
                data_root
                / "cord"
                / "cord_gold_text"
            ),
            "tesseract": (
                data_root
                / "cord"
                / "cord_ocr"
                / "tesseract"
            ),
            "paddleocr": (
                data_root
                / "cord"
                / "cord_ocr"
                / "paddleocr"
            ),
            "easyocr": (
                data_root
                / "cord"
                / "cord_ocr"
                / "easyocr"
            ),
        }

    return paths[engine]


def resolve_results_dir(
    results_root,
    dataset: str,
) -> Path:
    """Resolve the dataset-level results directory."""

    dataset = dataset.lower()

    if dataset not in {
        "funsd",
        "sroie",
        "cord",
    }:
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

    return Path(results_root) / dataset
