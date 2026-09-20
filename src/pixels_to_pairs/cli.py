#!/usr/bin/env python3
"""Run one frozen Pixels-to-Pairs benchmark configuration."""

import argparse
import os
from pathlib import Path

from pixels_to_pairs.experiments.config import (
    get_model_specs,
    validate_experiment,
)
from pixels_to_pairs.experiments.run import (
    run_model_experiment,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run one Pixels-to-Pairs benchmark "
            "configuration."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "funsd",
            "sroie",
            "cord",
        ],
    )

    parser.add_argument(
        "--engine",
        required=True,
        help=(
            "Text condition, e.g. gold_text_spatial, "
            "gold_text, paddleocr, easyocr, or tesseract."
        ),
    )

    parser.add_argument(
        "--shot",
        required=True,
        choices=[
            "0shot",
            "1shot",
            "2shot",
            "3shot",
        ],
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model name.",
    )

    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help=(
            "Root directory containing the benchmark "
            "dataset text directories."
        ),
    )

    parser.add_argument(
        "--gt-dir",
        required=True,
        type=Path,
        help=(
            "Ground-truth annotation/KVP directory "
            "for the selected dataset."
        ),
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help=(
            "Root output directory. "
            "Default: results"
        ),
    )

    parser.add_argument(
        "--example-files-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing frozen few-shot "
            "example files. Required for non-zero-shot runs."
        ),
    )

    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help=(
            "Optional document cap. CORD defaults "
            "to 100 when omitted."
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "cuda",
            "cpu",
        ],
        default="cuda",
    )

    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Optional Hugging Face token. Prefer "
            "the HF_TOKEN environment variable."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Start the selected run from scratch "
            "instead of resuming existing outputs."
        ),
    )

    return parser


def resolve_model_spec(
    model_name,
    shot,
):
    """Validate a model against the frozen benchmark model set."""

    specs = get_model_specs(shot)

    for spec in specs:
        if spec["name"] == model_name:
            return spec

    allowed = ", ".join(
        spec["name"]
        for spec in specs
    )

    raise ValueError(
        f"Model {model_name!r} is not part of "
        f"the frozen {shot} benchmark model set. "
        f"Allowed models: {allowed}"
    )


def validate_args(args):
    """Validate CLI-level benchmark constraints."""

    validate_experiment(
        args.dataset,
        args.engine,
        args.shot,
    )

    if (
        args.shot != "0shot"
        and args.example_files_dir is None
    ):
        raise ValueError(
            "--example-files-dir is required "
            "for few-shot runs."
        )

    if (
        args.max_docs is not None
        and args.max_docs <= 0
    ):
        raise ValueError(
            "--max-docs must be greater than zero."
        )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)

        model_spec = resolve_model_spec(
            args.model,
            args.shot,
        )

    except ValueError as exc:
        parser.error(str(exc))

    hf_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
    )

    result = run_model_experiment(
        dataset=args.dataset,
        engine=args.engine,
        shot_variant=args.shot,
        model_name=model_spec["name"],
        family=model_spec["family"],
        data_root=args.data_root,
        gt_dir=args.gt_dir,
        results_root=args.results_root,
        example_files_dir=(
            args.example_files_dir
        ),
        max_docs=args.max_docs,
        resume=not args.no_resume,
        device=args.device,
        hf_token=hf_token,
    )

    print()
    print("Run status:", result["status"])

    if "csv_path" in result:
        print(
            "Metrics:",
            result["csv_path"],
        )

    if "pred_path" in result:
        print(
            "Predictions:",
            result["pred_path"],
        )

    if result.get("summary") is not None:
        print(
            "Summary:",
            result["summary"],
        )

    return result


if __name__ == "__main__":
    main()
