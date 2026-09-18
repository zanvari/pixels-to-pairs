"""Frozen benchmark experiment configuration."""

from copy import deepcopy


ALL_MODELS = [
    {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "family": "causal",
    },
    {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "family": "causal",
    },
    {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "family": "causal",
    },
    {
        "name": "google/gemma-2b-it",
        "family": "causal",
    },
    {
        "name": "google/gemma-7b-it",
        "family": "causal",
    },
]


FEWSHOT_MODELS = [
    {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "family": "causal",
    },
    {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "family": "causal",
    },
]


VALID_SHOTS = {
    "0shot",
    "1shot",
    "2shot",
    "3shot",
}


VALID_ENGINES = {
    "funsd": {
        "gold_text_spatial",
        "gold_text",
        "tesseract",
        "paddleocr",
        "easyocr",
    },
    "sroie": {
        "gold_text",
        "tesseract",
        "paddleocr",
        "easyocr",
    },
    "cord": {
        "gold_text",
        "tesseract",
        "paddleocr",
        "easyocr",
    },
}


def get_model_specs(shot: str):
    """Return the frozen model set for a benchmark shot condition."""

    if shot not in VALID_SHOTS:
        raise ValueError(
            f"Unsupported shot setting: {shot}"
        )

    models = (
        ALL_MODELS
        if shot == "0shot"
        else FEWSHOT_MODELS
    )

    return deepcopy(models)


def validate_experiment(
    dataset: str,
    engine: str,
    shot: str,
) -> None:
    """Validate one benchmark dataset/engine/shot configuration."""

    dataset = dataset.lower()

    if dataset not in VALID_ENGINES:
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

    if engine not in VALID_ENGINES[dataset]:
        raise ValueError(
            f"Unsupported engine for {dataset}: {engine}"
        )

    if shot not in VALID_SHOTS:
        raise ValueError(
            f"Unsupported shot setting: {shot}"
        )
