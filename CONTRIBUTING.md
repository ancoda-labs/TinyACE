# Contributing

## Setup

```bash
git clone https://github.com/SirAlchemist1/edge-slm-ace.git
cd edge-slm-ace
python -m venv .venv && source .venv/bin/activate
make install          # editable install with dev, metrics and plots extras
make check            # tests + lint + format check, same as CI
```

## Before opening a PR

```bash
make check
```

CI runs exactly this and it can fail — it is a real gate, not decoration.

## Changing anything that affects a number

This is a research repository, so the bar for the measurement path is higher
than for the rest of the code. If your change touches prompting, parsing,
metrics, retrieval or scoring:

1. **Add a test that fails without the change.** Every defect listed under
   Fixed in `CHANGELOG.md` was invisible for months because nothing asserted
   the intended behaviour — including one that *did* have a failing test, which
   nobody saw because CI could not fail.
2. **Say what it does to existing results** in the PR description. A change to
   answer extraction or option handling invalidates prior runs; say so.
3. **Do not report a delta without a significance test.** See below.

## Reporting results

Read [docs/evaluation.md](docs/evaluation.md) first. The three rules that
previously went wrong:

- **Compare against the right arm.** `ace` belongs against `cot_control`, not
  `baseline` — otherwise the delta also contains the chain-of-thought
  instruction and the domain hints. Ablations belong against `tinyace_wm_256`.
  `scripts/compare_arms.py` picks the correct reference automatically.
- **Use enough data.** At n=50 the 95% interval spans about ±12 percentage
  points, which is wider than any effect this project has reported. Use n ≥ 500
  and at least three seeds.
- **Run the test.** A difference is a result only if it survives
  `python -m scripts.compare_arms`. If it does not, write "no detectable
  difference" — that is a finding, not a failure.

Check the run-health fields (`truncation_rate`, `chat_template_rate`,
`relevance_active`) before trusting any run. `make report` surfaces them.

## Style

- `black`, line length 100. `make format`.
- Docstrings on public functions: what it does, `Args`, `Returns`.
- Comments explain *why*, especially where the obvious implementation is wrong.
  Several bugs here were introduced by code that looked correct.
- Match the surrounding file.

## Commits

`type(scope): subject` — `fix`, `feat`, `refactor`, `docs`, `test`, `chore`,
`style`. Explain in the body what was wrong and why the fix is right; if it
changes a measurement, say that prior results need regenerating.
