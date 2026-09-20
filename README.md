# Pixels to Pairs

Official implementation and reproducibility code for:

**From Pixels to Pairs: A Comprehensive Benchmark of LLM-Driven Key–Value Extraction in Noisy Document Settings**

Zahra Anvari

This repository provides the benchmark pipeline used to study how text quality,
document structure, model choice, and in-context demonstrations affect
text-only key–value pair (KVP) extraction from documents.

The benchmark evaluates instruction-tuned large language models on
annotation-derived and OCR-derived text from **FUNSD**, **SROIE**, and
**CORD** under a controlled inference, parsing, and evaluation protocol.

## Benchmark overview

The benchmark evaluates five instruction-tuned decoder-only LLMs:

- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `google/gemma-2b-it`
- `google/gemma-7b-it`

Each dataset is evaluated using annotation-derived Gold text and text produced
by three OCR engines:

- PaddleOCR
- EasyOCR
- Tesseract

All five models are evaluated zero-shot. Qwen2.5-7B and LLaMA-3-8B are
additionally evaluated with one-, two-, and three-shot prompting.

The study comprises **136 experimental configurations** and **17,688 planned
document-level inference instances**. The main cross-dataset benchmark contains
96 configurations (15,808 document-level instances), while a confirmatory
FUNSD sensitivity study adds 40 configurations (1,880 instances) to evaluate
the effect of three-shot demonstration composition.

## Main findings

The experiments show that:

- OCR-derived text generally reduces exact and partial value recovery, but the
  magnitude of degradation depends on the dataset, OCR engine, model, and
  metric.
- Strong performance on annotation-derived text does not necessarily imply the
  same relative behavior under OCR-derived input.
- Few-shot prompting is model- and dataset-dependent and is not consistently
  monotonic as demonstrations are added.
- FUNSD remains challenging in a text-only setting because flattening forms
  removes direct spatial association cues.
- Demonstration composition can materially affect few-shot performance even
  when the number of demonstrations is fixed.

These results motivate evaluating document extraction systems under the actual
text conditions encountered in OCR-based pipelines rather than relying only on
clean-text performance.

## Datasets

| Dataset | Document type | Evaluated documents |
|---|---|---:|
| FUNSD | Forms | 47 |
| SROIE | Receipts | 347 |
| CORD | Receipts | 100 |

FUNSD originally contains 50 test documents. Three documents contain no
evaluable ground-truth KVPs under the benchmark preprocessing and evaluation
protocol, leaving 47 documents in the reported evaluation.

For the final FUNSD Gold condition, annotation text is reconstructed
spatially before being flattened into a text sequence. This corrected
**spatially reconstructed Gold text** is the Gold representation used for the
reported FUNSD results.

## Evaluation

The benchmark reports three reference-centered document-level metrics:

- **Key Recall (KR)** — recovery of annotated key occurrences.
- **Exact Match (EM)** — recovery of both the normalized key and its exact
  normalized value.
- **Value F1 (VF1)** — token-level overlap between predicted and reference
  values for annotated KVPs.

Missing annotated pairs remain in the metric denominator and receive zero
credit. Dataset-level results are macro-averaged over evaluable documents.

Output parsing is deterministic and conservative. The benchmark does not use
semantic repair, fuzzy key matching, ground-truth-aware recovery, or
document-text fallback.

## Installation

Clone the repository:

```bash
git clone https://github.com/zanvari/pixels-to-pairs.git
cd pixels-to-pairs
```

A virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the package:

```bash
pip install -e .
```

For development and testing:

```bash
pip install -e ".[dev]"
```

The package requires Python 3.10 or newer.

## Data preparation

Dataset files are not redistributed by this repository.

Prepare FUNSD, SROIE, and CORD locally and organize annotation-derived and OCR
text according to the expected directory structure.

See **[docs/data_setup.md](docs/data_setup.md)** for the complete directory
layout, accepted ground-truth formats, and few-shot example requirements.

## Running the benchmark

Installation provides the `pixels-to-pairs` command.

For example, a zero-shot Qwen2.5-7B FUNSD run using the corrected
spatially reconstructed Gold text can be launched with:

```bash
pixels-to-pairs \
  --dataset funsd \
  --engine gold_text_spatial \
  --shot 0shot \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data-root /path/to/data \
  --gt-dir /path/to/funsd/annotations \
  --results-root results
```

A PaddleOCR condition changes only the text condition:

```bash
pixels-to-pairs \
  --dataset funsd \
  --engine paddleocr \
  --shot 0shot \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data-root /path/to/data \
  --gt-dir /path/to/funsd/annotations \
  --results-root results
```

Few-shot runs additionally require the frozen demonstration files:

```bash
pixels-to-pairs \
  --dataset funsd \
  --engine gold_text_spatial \
  --shot 3shot \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --data-root /path/to/data \
  --gt-dir /path/to/funsd/annotations \
  --example-files-dir /path/to/funsd/examples \
  --results-root results
```

Run:

```bash
pixels-to-pairs --help
```

for the complete command-line interface.

Some Hugging Face models require authentication or acceptance of their model
license. A token can be supplied through the `HF_TOKEN` environment variable.

## Experimental protocol

The final benchmark uses:

- maximum input length: **8,192 tokens**
- maximum generation length: **1,024 tokens**
- batch size: **1**
- deterministic greedy decoding
- fixed NumPy/PyTorch seeds
- standard model chat templates when available
- no model-specific prompt optimization or semantic post-processing

The reported experiments were run with Python 3.13.15, PyTorch 2.11.0 with
CUDA 12.8, Transformers 5.16.1, and an NVIDIA A100 40 GB GPU.

The package supports Python 3.10+; the versions above document the environment
used for the final reported experiments rather than imposing those exact
versions as installation requirements.

## Outputs

Each experiment writes a dataset- and configuration-specific output directory
containing:

```text
metrics.csv
predictions.jsonl
```

The saved artifacts include document-level metrics, raw model output, parsed
predictions, prompt-token diagnostics, generation diagnostics, and run
metadata.

CORD outputs additionally include prediction-drift diagnostics.

See **[results/README.md](results/README.md)** for details.

## Repository structure

```text
pixels-to-pairs/
├── docs/
│   └── data_setup.md
├── notebooks/
│   └── benchmark_master.ipynb
├── results/
│   └── README.md
├── scripts/
│   └── run_benchmark.py
├── src/
│   └── pixels_to_pairs/
│       ├── analysis/
│       ├── data/
│       ├── evaluation/
│       ├── experiments/
│       ├── inference/
│       ├── prompting/
│       ├── cli.py
│       └── parsing.py
├── tests/
├── LICENSE
├── pyproject.toml
└── README.md
```

The implementation is organized into reusable components for data loading,
prompt construction, generation, parsing, evaluation, experiment
orchestration, persistence, and analysis.

## Reference notebook

`notebooks/benchmark_master.ipynb` is a sanitized reference copy of the
validated Colab benchmark implementation from which the reusable package was
refactored.

The notebook intentionally preserves the original Colab-oriented execution
structure and paths as a historical reproducibility artifact. For new runs,
the installable package and command-line interface are recommended.

## Testing

The repository includes regression tests covering data loading, prompting,
output parsing, evaluation metrics, experiment configuration, path
resolution, persistence, model generation utilities, execution, analysis, and
the command-line interface.

Run:

```bash
pytest
```

The current test suite contains **276 tests**.

## Reproducibility notes

The public implementation preserves the final benchmark protocol used for the
reported experiments, including deterministic generation, frozen model sets,
dataset-specific prompt construction, conservative parsing, and
reference-centered evaluation.

In particular:

- use `gold_text_spatial` for the reported FUNSD Gold condition;
- FUNSD documents with no evaluable KVPs are excluded before aggregation;
- CORD uses a 100-document evaluation cap in the frozen benchmark protocol;
- few-shot model choices and demonstrations are fixed rather than selected
  dynamically during evaluation;

## Citation

If you use this benchmark or code, please cite:

```bibtex
@article{anvari2026pixelstopairs,
  title   = {From Pixels to Pairs: A Comprehensive Benchmark of LLM-Driven
             Key--Value Extraction in Noisy Document Settings},
  author  = {Anvari, Zahra},
  year    = {2026}
}
```

The citation will be updated with the permanent preprint/publication
identifier when available.

## License

This repository is released under the [MIT License](LICENSE).
