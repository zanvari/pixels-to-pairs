# Data Setup

The benchmark does not redistribute FUNSD, SROIE, CORD, or OCR-derived
dataset files. Prepare the datasets locally and pass their locations to the
benchmark runner.

## Directory layout

`--data-root` should point to a directory with the following structure:

    <data-root>/
    ├── funsd/
    │   ├── funsd_gold_text_spatial/
    │   ├── funsd_gold_text/
    │   └── funsd_ocr/
    │       ├── tesseract/
    │       ├── paddleocr/
    │       └── easyocr/
    ├── sroie/
    │   ├── sroie_gold_text/
    │   └── sroie_ocr/
    │       ├── tesseract/
    │       ├── paddleocr/
    │       └── easyocr/
    └── cord/
        ├── cord_gold_text/
        └── cord_ocr/
            ├── tesseract/
            ├── paddleocr/
            └── easyocr/

Each text directory contains one UTF-8 `.txt` file per document. The filename
stem is treated as the document ID and must match the corresponding
ground-truth JSON filename stem.

For the reported FUNSD Gold results, use `funsd_gold_text_spatial`. This is
the corrected spatially reconstructed Gold-text condition used in the final
benchmark.

## Ground-truth KVP files

The `--gt-dir` argument points to the directory containing document-level
ground-truth JSON files.

### FUNSD

FUNSD supports either the original annotation structure:

    {
      "form": [...]
    }

or pre-extracted KVP files:

    [
      {"key": "Invoice Number", "value": "12345"},
      {"key": "Date", "value": "2026-01-01"}
    ]

When original FUNSD annotations are supplied, KVPs are constructed from
linked question-answer entities. Documents with zero extracted ground-truth
KVPs are excluded from benchmark metric computation.

### SROIE

SROIE ground-truth files must contain either:

    {
      "kvp": [
        {"key": "company", "value": "Example Store"},
        {"key": "total", "value": "12.50"}
      ]
    }

or a top-level list:

    [
      {"key": "company", "value": "Example Store"},
      {"key": "total", "value": "12.50"}
    ]

### CORD

CORD ground-truth files likewise use document-level KVP lists. The loader
accepts either:

    {"kvp": [...]}

or:

    [...]

For compatibility with prepared benchmark files, CORD also accepts
`key`/`value`, `Key`/`Value`, or `k`/`v` field names.

## Few-shot example files

For `1shot`, `2shot`, or `3shot` experiments, pass `--example-files-dir`.

The few-shot demonstrations are frozen by the benchmark implementation rather
than sampled dynamically. The example directory must therefore contain the
specific text and KVP JSON files expected by the selected dataset's prompt
builder.

The sanitized reference notebook in `notebooks/benchmark_master.ipynb`
documents the original Colab experiment setup and the audited demonstration
configuration used for the reported benchmark.

## Generated results

Use `--results-root` to choose where experiment outputs are written. Results
are organized first by dataset and then by experiment condition.

For example:

    <results-root>/
    └── funsd/
        └── funsd_paddleocr_0shot_Qwen_Qwen2.5-7B-Instruct/
            ├── metrics.csv
            └── predictions.jsonl

See `results/README.md` for the output schema and metric definitions.

## Dataset files and versioning

Dataset files and generated OCR text are intentionally excluded from Git.
Keep local datasets under a separate data directory or under the repository's
ignored `/data/` directory.

For reproducible comparisons, keep document IDs aligned across ground truth
and all text conditions and avoid mixing outputs from different benchmark
protocol revisions.
