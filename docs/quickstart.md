# Quick Start Guide

**Get started with TinyACE in 5 minutes**

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, for GPU acceleration)
- 8GB+ RAM recommended

## Installation

```bash
# Clone repository
git clone https://github.com/SirAlchemist1/edge-slm-ace.git
cd edge-slm-ace

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package with every extra the pipeline needs.
# Without the `metrics` extra there is no embedding backend, and retrieval
# silently degrades to retention-only ranking.
make install
```

## Quick Test

Run a smoke test to verify installation:

```bash
# CPU test (uses tiny-gpt2)
python -m scripts.smoke_test

# GPU test with a real model (the flag is --task, not --task-name)
python -m scripts.smoke_test --model phi3-mini --task sciq_tiny --device cuda --limit 2
```

## Run Your First Experiment

### Baseline Evaluation

```bash
python -m scripts.run_experiment \
  --model-id microsoft/Phi-3-mini-4k-instruct \
  --task-name sciq_test \
  --mode baseline \
  --device cuda \
  --limit 10 \
  --output-path results/phi3/sciq_test/baseline/cuda/results.csv \
  --metrics-path results/phi3/sciq_test/baseline/cuda/metrics.json \
  --predictions-path results/phi3/sciq_test/baseline/cuda/predictions.jsonl
```

`--output-path` is required. The directory layout matters: everything that reads
results derives the arm and device from `{model}/{task}/{arm}/{device}/`.

### ACE Working Memory Mode

```bash
python -m scripts.run_experiment \
  --model-id microsoft/Phi-3-mini-4k-instruct \
  --task-name sciq_test \
  --mode ace \
  --ace-mode ace_working_memory \
  --token-budget 256 \
  --device cuda \
  --limit 10 \
  --playbook-path results/phi3/sciq_test/tinyace_wm_256/cuda/playbook.jsonl \
  --output-path results/phi3/sciq_test/tinyace_wm_256/cuda/results.csv \
  --metrics-path results/phi3/sciq_test/tinyace_wm_256/cuda/metrics.json \
  --predictions-path results/phi3/sciq_test/tinyace_wm_256/cuda/predictions.jsonl
```

`--mode ace` requires `--playbook-path`. This arm learns online from the split
it is scored on; see [evaluation.md](evaluation.md) for the frozen-playbook
protocol, which is preferred.

## View Results

Results are saved to:
- `results/{model}/{task}/{mode}/{device}/results.csv` - Per-example results
- `results/{model}/{task}/{mode}/{device}/metrics.json` - Aggregate metrics

## Next Steps

- Read [evaluation.md](evaluation.md) **before reporting any number**
- Read [architecture.md](architecture.md) to understand the system
- Check [results.md](results.md) for the withdrawn tables and why
- See [figures.md](figures.md) for visualisation
- Review [configs/experiment_grid.yaml](../configs/experiment_grid.yaml) for configuration options
