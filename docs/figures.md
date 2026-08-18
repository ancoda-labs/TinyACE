# TinyACE Plotting Pipeline Guide

## Overview

The `scripts/make_figures.py` module automatically generates all paper figures from standardized evaluation results. It reads CSV files from the `results/` directory and generates publication-ready PDF and PNG figures.

## Quick Start

### Generate All Plots

```bash
# Generate all plots from results/
python -m scripts.make_figures

# Custom paths (note: this script uses underscores, unlike the others)
python -m scripts.make_figures --results_dir results --output_dir figures
```

### Auto-Generate Plots After Evaluation

```bash
# Run evaluation and auto-generate plots
python -m scripts.run_experiment \
  --model-id phi3-mini \
  --task-name sciq_tiny \
  --mode baseline \
  --output-path results/sciq_phi3_baseline.csv \
  --auto-plots

# Run ACE epochs and auto-generate plots
python -m scripts.run_ace_epoch \
  --model-id phi3-mini \
  --task-name sciq_tiny \
  --epochs 3 \
  --ace-mode ace_working_memory \
  --device cuda \
  --auto-plots
```

## Expected CSV Format

Each CSV file under `results/` should have at least these columns:

### Required Columns:
- `qid` or `sample_id`: Question/sample identifier
- `task` or `task_name`: Registered task name (`sciq_test`, `sciq_val`, `sciq_tiny`,
  `medqa_tiny`, `iot_tiny`)
- `model` or `model_id`: HuggingFace id; `model_label()` shortens it for display
- `mode`: A canonical arm key from `reporting/schema.py` (`baseline`,
  `cot_control`, `ace_full`, `tinyace_wm_256`, `self_refine`, ...)
- `is_correct` or `correct`: 0/1 or boolean indicating correctness

### Optional Columns (for enhanced plots):
- `context_tokens`: Integer, total tokens fed into model
- `latency_ms`: Float, end-to-end response time in milliseconds

### Legacy Columns (backward compatible):
The plotting script automatically maps:
- `sample_id` → `qid`
- `task_name` → `task`
- `model_id` → `model` (with normalization)
- `correct` → `is_correct`

## Generated Figures

### Figure 1: Memory Cliff (`fig1_memory_cliff.pdf`)
**LaTeX Reference:** `Figure~\ref{fig:motivation}`

Shows the learning curve: accuracy vs questions processed, per arm. The arms
plotted are `FIGURE_ARMS` in `make_figures.py`.

**Requirements:**
- Multiple samples with sequential `qid` values
- At least two arms, one of which should be `cot_control` — an ACE curve against
  `baseline` alone cannot separate the playbook from the prompt scaffold

### Figure 3: Token Efficiency (`fig3_token_efficiency.pdf`)
**LaTeX Reference:** Token overhead comparison figure

Compares mean `prompt_tokens` per arm — the one token count defined identically
in every arm. `context_tokens` is the task context and `playbook_tokens` the
retrieved lessons; those two are per-arm breakdowns, not cross-arm comparisons.

**Requirements:**
- `context_tokens` column in CSV files
- Multiple modes available

### Figure 6: Ablation Study (`fig6_ablation.pdf`)
**LaTeX Reference:** Ablation results figure

Shows accuracy drops for ablation variants relative to `tinyace_full`.

**Requirements:**
- Modes like `tinyace_full`, `tinyace_no_recency`, `tinyace_no_vagueness`, etc.
- If not available, function logs a warning and skips

### Figure 7: Device Comparison (`fig7_device_comparison.pdf`)
**LaTeX Reference:** Device/latency comparison (optional)

Compares latency vs accuracy across different devices.

**Requirements:**
- `device` or `hardware` column in CSV files
- `latency_ms` column
- If not available, function logs a message and skips

## Output Structure

```
figures/
├── summary.csv              # Aggregated statistics per (task, model, mode)
├── fig1_memory_cliff.pdf    # Memory cliff plot
├── fig1_memory_cliff.png    # PNG version
├── fig3_token_efficiency.pdf
├── fig3_token_efficiency.png
├── fig6_ablation.pdf
├── fig6_ablation.png
├── fig7_device_comparison.pdf (if device data available)
└── fig7_device_comparison.png
```

## Results layout

One layout, produced by `run_eval_grid.py` and required by everything that reads
results:

```
results/{model}/{task}/{arm}/{device}/
├── metrics.json       run-level summary
├── predictions.jsonl  one row per example
├── results.csv        the same rows as CSV
└── playbook.jsonl     ACE arms only
```

`reporting/load.py` derives the arm from the second-to-last path segment and the
device from the last, so the depth is not optional. Examples:

- `results/phi_3_mini/sciq_test/baseline/cuda/`
- `results/phi_3_mini/sciq_test/cot_control/cuda/`
- `results/phi_3_mini/sciq_test/tinyace_wm_256/cuda/`

A multi-seed study needs one root per seed (`results/seed42/...`), because the
layout has no seed segment.

## Arm and model names

Both come from [`reporting/schema.py`](../src/edge_slm_ace/reporting/schema.py),
which is the single registry. The plotting scripts do not define their own
labels; they used to, and disagreed with each other, so the same run appeared
under two different names depending on which script rendered it.

- Arm keys are the canonical ones (`baseline`, `cot_control`, `ace_full`,
  `tinyace_wm_256`, ...) and `arm_label()` renders them for display. There is no
  longer a `zero_shot` or a bare `tinyace`; a figure that filters on those names
  silently drops every arm.
- `model_label()` maps a HuggingFace id to a short name:
  `microsoft/Phi-3-mini-4k-instruct` -> `Phi-3-mini`,
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0` -> `TinyLlama-1.1B`.

To add an arm to the figures, register it in `schema.py` and add its key to
`FIGURE_ARMS` in `make_figures.py`.

## Troubleshooting

### "No CSV files found"
- Check that CSV files exist in `results/` directory
- Verify the layout is `{model}/{task}/{arm}/{device}/results.csv` — a flatter
  tree loads, but the arm cannot be recovered from it

### "No data for memory cliff plot"
- Ensure you have multiple samples with sequential `qid` values
- Check that at least two arms are present, and that their keys are in
  `FIGURE_ARMS`

### "No prompt_tokens data available"
- The token figure requires the `prompt_tokens` column
- Check that evaluation scripts are writing this column
- Plot will be skipped if data unavailable

### "No ablation modes found"
- Ablation plot requires modes like `tinyace_full`, `tinyace_no_recency`, etc.
- If not running ablation studies, this is expected
- Plot will be skipped with a warning

## Integration with LaTeX

In your LaTeX paper (`tinyace_paper.tex`), include figures like:

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.95\columnwidth]{figures/fig1_memory_cliff.pdf}
    \caption{Memory Cliff: Learning curve showing accuracy improvement over questions processed.}
    \label{fig:motivation}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.95\columnwidth]{figures/fig3_token_efficiency.pdf}
    \caption{Token Efficiency: Average context tokens across different modes.}
    \label{fig:token_efficiency}
\end{figure}
```

## Dependencies

Required Python packages (already in `requirements.txt`):
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`

Install with:
```bash
pip install -r requirements.txt
```

## Advanced Usage

### Custom Task/Model Selection

The plotting functions accept optional `task` and `model` parameters. You can modify `scripts/make_figures.py` to filter specific combinations:

```python
# In plot_memory_cliff()
plot_memory_cliff(df, output_path, task="sciq_test", model="Phi-3-mini")
```

### Adding New Plots

To add a new figure:

1. Create a new plotting function (e.g., `plot_new_figure()`)
2. Add it to `main()` function
3. Update LaTeX mapping in docstring
4. Follow the same pattern as existing plots

## Best Practices

1. **Standardize CSV format**: Ensure all evaluation scripts write canonical columns
2. **Run plots after experiments**: Use `--auto-plots` flag for convenience
3. **Check summary.csv**: Review aggregated statistics before generating plots
4. **Version control plots**: Commit generated PDFs to track figure evolution
5. **Document mode names**: Keep mode naming consistent across experiments

---

## Diagnostics

`make_figures.py` produces the paper figures. `make_diagnostics.py` produces the
run-health and playbook-behaviour plots, which are what you look at *before*
trusting a figure:

```bash
# Aggregate first: this writes summary_runs.csv and summary_accuracy.csv
python -m scripts.aggregate_results --results-root results

python -m scripts.make_diagnostics \
  --summary-csv results/summary_runs.csv \
  --results-root results \
  --output-dir results/plots
```

It plots accuracy, OMA, GOM, ACR, latency and peak memory per arm, plus playbook
growth and eviction counts from the per-step `playbook_log.csv`. The playbook
plots are the ones that answer "did the memory system do anything at all".
