import pytest

from pixels_to_pairs.experiments.config import (
    ALL_MODELS,
    FEWSHOT_MODELS,
    get_model_specs,
    validate_experiment,
)


EXPECTED_ALL_MODEL_NAMES = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "google/gemma-2b-it",
    "google/gemma-7b-it",
]


EXPECTED_FEWSHOT_MODEL_NAMES = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]


def test_all_models_match_frozen_benchmark_set():
    assert [
        model["name"]
        for model in ALL_MODELS
    ] == EXPECTED_ALL_MODEL_NAMES

    assert all(
        model["family"] == "causal"
        for model in ALL_MODELS
    )


def test_fewshot_models_match_frozen_benchmark_set():
    assert [
        model["name"]
        for model in FEWSHOT_MODELS
    ] == EXPECTED_FEWSHOT_MODEL_NAMES


def test_zero_shot_uses_all_models():
    assert [
        model["name"]
        for model in get_model_specs("0shot")
    ] == EXPECTED_ALL_MODEL_NAMES


@pytest.mark.parametrize(
    "shot",
    [
        "1shot",
        "2shot",
        "3shot",
    ],
)
def test_fewshot_conditions_use_two_models(shot):
    assert [
        model["name"]
        for model in get_model_specs(shot)
    ] == EXPECTED_FEWSHOT_MODEL_NAMES


def test_get_model_specs_returns_independent_copy():
    models = get_model_specs("0shot")

    models[0]["name"] = "changed"

    assert (
        ALL_MODELS[0]["name"]
        == "Qwen/Qwen2.5-7B-Instruct"
    )


@pytest.mark.parametrize(
    "engine",
    [
        "gold_text_spatial",
        "gold_text",
        "tesseract",
        "paddleocr",
        "easyocr",
    ],
)
def test_funsd_accepts_expected_engines(engine):
    validate_experiment(
        "funsd",
        engine,
        "0shot",
    )


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
def test_sroie_and_cord_accept_expected_engines(
    dataset,
    engine,
):
    validate_experiment(
        dataset,
        engine,
        "0shot",
    )


def test_sroie_rejects_funsd_spatial_gold():
    with pytest.raises(
        ValueError,
        match="Unsupported engine",
    ):
        validate_experiment(
            "sroie",
            "gold_text_spatial",
            "0shot",
        )


def test_unknown_dataset_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported dataset",
    ):
        validate_experiment(
            "unknown",
            "gold_text",
            "0shot",
        )


def test_unknown_shot_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported shot",
    ):
        validate_experiment(
            "funsd",
            "gold_text_spatial",
            "4shot",
        )
