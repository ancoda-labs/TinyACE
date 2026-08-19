# Installation Guide

## System Requirements

- **Python**: 3.10 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended (CUDA-capable GPU or Apple Silicon)
- **Disk Space**: ~10GB for models and dependencies

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/SirAlchemist1/edge-slm-ace.git
cd edge-slm-ace
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip

# Editable install with every extra. This is what `make install` runs.
pip install -e ".[dev,retrieval,report]"
```

Do not install with `[dev]` alone. Without the `retrieval` extra there is no
`sentence-transformers`, and lesson ranking degrades to question-independent
scoring — it warns, and `metrics.json` records `relevance_active: false`, but
the run still completes. Without `report`, `aggregate_results` cannot run.

`lm-eval[hf]` is a required dependency and brings torch and transformers with
it. Scoring is the harness's, so nothing here re-implements a metric it already
computes.

### 4. Verify Installation

```bash
make data     # verify the committed Belebele files are the parallel corpus
make test     # the suite; no model downloads

# One real arm end to end, on a tiny random-weight checkpoint.
python -m scripts.run_arm \
  --model tiny-gpt2 --language en --arm tinyace \
  --output-dir /tmp/smoke --limit 4 --device cpu --no-chat-template
```

## Device-Specific Setup

### CUDA (NVIDIA GPU)

1. Install CUDA toolkit (11.8 or higher)
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### MPS (Apple Silicon)

1. macOS 12.3+ required
2. PyTorch automatically uses MPS if available
3. No additional setup needed

### CPU Only

1. Install CPU-only PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

## Troubleshooting

### Common Issues

**Issue**: `torch.cuda.is_available()` returns False
- **Solution**: Verify CUDA installation and PyTorch CUDA version compatibility

**Issue**: Out of memory errors
- **Solution**: Reduce `--batch-size`, or run a smaller model. Belebele prompts
  are long (79 words of passage on average, up to 217), so batch size costs more
  memory here than on a short-prompt task.

**Issue**: `[relevance] could not load ...`
- **Solution**: Install the `retrieval` extra. The run will complete without it,
  but retrieval is not query-conditioned and the results must say so.

**Issue**: A run warns that prompts were truncated
- **Solution**: Not a configuration problem — see
  [evaluation.md](evaluation.md). lm-eval truncates from the left, which eats
  the passage, and longer-prefix arms truncate more. The run is not comparable
  across arms.

**Issue**: Model download fails
- **Solution**: Check internet connection and HuggingFace access

## Development Setup

For development, install additional dependencies:

```bash
make install    # pip install -e ".[dev,retrieval,report]"
make check      # tests + lint + format check, the same as CI
```

The `dev` extra provides `pytest`, `pytest-cov`, `black` and `flake8`.

## A checkout, not a library

This is a research repository. The datasets in `data/tasks/` and the
entrypoints in `scripts/` are deliberately outside the wheel, and the
documented workflow (`make install`, `python -m scripts.*`) assumes a git
clone. Install with `pip install -e .` from a checkout.

If the package ever ends up installed away from its datasets, point
`TINYACE_DATA_ROOT` at the directory containing `data/tasks/`.
