# Quick Start

From a clean checkout to a result you are allowed to report.

## 1. Install

```bash
git clone https://github.com/SirAlchemist1/edge-slm-ace.git
cd edge-slm-ace
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
make install
```

`make install` runs `pip install -e ".[dev,retrieval,report]"`. The three extras
are:

| Extra | What it buys | Without it |
|---|---|---|
| `retrieval` | The multilingual sentence encoder for query-conditioned lesson ranking | Retrieval degrades to question-independent ranking. It warns, and `metrics.json` records `relevance_active: false` |
| `report` | pandas and matplotlib | `aggregate_results` cannot run |
| `dev` | pytest, black, flake8 | `make check` cannot run |

`lm-eval[hf]` brings torch and transformers; scoring is the harness's, so
nothing here re-implements a metric it already computes.

## 2. Check the data

The two Belebele language files are committed, so this verifies rather than
downloads:

```bash
make data
```

It confirms 900 rows per language, that the id sets match after sorting, and
that both languages agree on the correct option for every item. That last check
matters: the files are *not* in the same row order, so anything that joins them
by position pairs unrelated questions while looking like a matched comparison.

## 3. Verify the pipeline runs

```bash
make test     # 250 tests, no model downloads

# One real arm end to end on a tiny random-weight checkpoint.
python -m scripts.run_arm \
  --model tiny-gpt2 --language en --arm tinyace \
  --output-dir /tmp/smoke --limit 4 --device cpu --no-chat-template
```

`tiny-gpt2` has random weights. It exercises every code path and its numbers
mean nothing — the registry marks it `generation="debug"` and `screen_models`
excludes it. Expect the playbook to come out empty and the run to say so.

## 4. Screen

A model that cannot read Nepali at all cannot show a context-adaptation effect
on Nepali, so entry to the grid is gated:

```bash
make screen DEVICE=cuda
```

The rule is pre-registered and stated on the *interval*: n=400 items drawn from
the adaptation split, and the lower bound of the Wilson interval must clear 30%
(chance is 25%). Screening never touches the evaluation split — `screen_models`
refuses to start if it would.

Survivors are written to `results/screening/screening.json`, and `make grid`
reads its model list from there.

## 5. Run the grid

```bash
make grid DEVICE=cuda
```

Eleven arms x two languages per model, with the model loaded once and reused.
Completed cells are skipped unless the seed or the commit changed, so an
interrupted grid resumes. Preview without running anything:

```bash
python -m scripts.run_grid --dry-run
```

## 6. Report

```bash
make report
```

Two steps. `aggregate_results` writes `summary_runs.csv` and
`summary_accuracy.csv` (accuracies with Wilson intervals) and prints any health
warning that invalidates a run. `compare_arms` pairs each arm with the reference
`reporting.reference_for()` names for it, runs an exact McNemar test on the
discordant pairs, and Holm-corrects every comparison as one family.

**The adjusted p-value is the one that decides.** Read
[evaluation.md](evaluation.md) before putting any of it in a table.

## Where things land

```
results/
  screening/screening.json
  {model}/{language}/{arm}/
    metrics.json          run-level summary, config and environment
    predictions.jsonl     one row per item: qid, is_correct
    playbook.jsonl        the frozen playbook (arms that adapt)
    adaptation_log.csv    one row per adaptation step
  summary_runs.csv
  summary_accuracy.csv
```

The layout has one owner, `reporting/layout.py`. `cell_dir` writes it and
`parse_cell` reads it; nothing derives the arm by counting path segments.

## Next

- [evaluation.md](evaluation.md) — the protocol, and why each arm exists
- [architecture.md](architecture.md) — the loop and the harness boundary
- [api.md](api.md) — package reference
