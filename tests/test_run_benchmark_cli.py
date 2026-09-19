import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_benchmark.py"
)

SPEC = importlib.util.spec_from_file_location(
    "run_benchmark_cli",
    SCRIPT_PATH,
)

CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def test_zero_shot_qwen_is_allowed():
    spec = CLI.resolve_model_spec(
        "Qwen/Qwen2.5-7B-Instruct",
        "0shot",
    )

    assert spec == {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "family": "causal",
    }


def test_fewshot_qwen_is_allowed():
    spec = CLI.resolve_model_spec(
        "Qwen/Qwen2.5-7B-Instruct",
        "3shot",
    )

    assert spec["family"] == "causal"


def test_fewshot_mistral_is_rejected():
    with pytest.raises(
        ValueError,
        match="not part of",
    ):
        CLI.resolve_model_spec(
            "mistralai/Mistral-7B-Instruct-v0.2",
            "1shot",
        )


def test_fewshot_requires_example_directory():
    parser = CLI.build_parser()

    args = parser.parse_args(
        [
            "--dataset",
            "sroie",
            "--engine",
            "gold_text",
            "--shot",
            "1shot",
            "--model",
            "Qwen/Qwen2.5-7B-Instruct",
            "--data-root",
            "/data",
            "--gt-dir",
            "/gt",
        ]
    )

    with pytest.raises(
        ValueError,
        match="example-files-dir",
    ):
        CLI.validate_args(args)


def test_zero_shot_does_not_require_examples():
    parser = CLI.build_parser()

    args = parser.parse_args(
        [
            "--dataset",
            "funsd",
            "--engine",
            "gold_text_spatial",
            "--shot",
            "0shot",
            "--model",
            "Qwen/Qwen2.5-7B-Instruct",
            "--data-root",
            "/data",
            "--gt-dir",
            "/gt",
        ]
    )

    CLI.validate_args(args)


def test_nonpositive_max_docs_is_rejected():
    parser = CLI.build_parser()

    args = parser.parse_args(
        [
            "--dataset",
            "cord",
            "--engine",
            "gold_text",
            "--shot",
            "0shot",
            "--model",
            "google/gemma-2b-it",
            "--data-root",
            "/data",
            "--gt-dir",
            "/gt",
            "--max-docs",
            "0",
        ]
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        CLI.validate_args(args)


def test_main_calls_orchestration_with_frozen_family(
    tmp_path,
):
    fake_result = {
        "status": "completed",
        "csv_path": tmp_path / "metrics.csv",
        "pred_path": tmp_path / "predictions.jsonl",
        "summary": {
            "key_recall": 1.0,
        },
    }

    with patch.object(
        CLI,
        "run_model_experiment",
        return_value=fake_result,
    ) as runner:
        result = CLI.main(
            [
                "--dataset",
                "funsd",
                "--engine",
                "gold_text_spatial",
                "--shot",
                "0shot",
                "--model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--data-root",
                "/data",
                "--gt-dir",
                "/gt",
                "--results-root",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )

    runner.assert_called_once()

    kwargs = runner.call_args.kwargs

    assert kwargs["family"] == "causal"
    assert kwargs["resume"] is True
    assert kwargs["device"] == "cpu"
    assert result == fake_result


def test_no_resume_is_forwarded(
    tmp_path,
):
    fake_result = {
        "status": "completed",
    }

    with patch.object(
        CLI,
        "run_model_experiment",
        return_value=fake_result,
    ) as runner:
        CLI.main(
            [
                "--dataset",
                "cord",
                "--engine",
                "gold_text",
                "--shot",
                "0shot",
                "--model",
                "google/gemma-2b-it",
                "--data-root",
                "/data",
                "--gt-dir",
                "/gt",
                "--results-root",
                str(tmp_path),
                "--device",
                "cpu",
                "--no-resume",
            ]
        )

    assert (
        runner.call_args.kwargs["resume"]
        is False
    )
