# Results

This directory is reserved for benchmark results and derived analysis artifacts
from the *Pixels to Pairs* experiments.

## Benchmark outputs

Running an experiment creates a dataset-specific output directory with the
following naming convention:

```text
{dataset}_{engine}_{shot_variant}_{model}
```

Each run produces:

- `metrics.csv` — document-level evaluation metrics and generation diagnostics.
- `predictions.jsonl` — ground-truth and predicted key-value pairs, raw model
  output, evaluation metrics, token diagnostics, and run metadata.

CORD runs additionally include prediction-drift diagnostics in the metrics
output.

## Metrics

The primary document-level metrics are:

- **Key Recall (KR):** fraction of normalized ground-truth keys recovered by
  the model.
- **Exact Match (EM):** fraction of matched ground-truth keys whose predicted
  value exactly matches the normalized ground-truth value.
- **Value F1 (VF1):** token-level F1 between predicted and ground-truth values
  for matched keys.

Macro results are computed by averaging document-level metrics over the
evaluated documents.


## Generated files

Full experiment outputs are not tracked by default because they can become
large. The benchmark runner writes generated outputs to the configured results
location.

The manuscript reports aggregate results from the frozen experimental
protocol. Reproduction instructions and the corresponding experiment
configuration are documented in the repository's main README.
