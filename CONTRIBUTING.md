# Contributing to TinyACE

Thanks for helping. TinyACE asks whether a small language model can get better
at a task, in English and in Nepali, by writing itself a playbook rather than
updating its weights. It is a research codebase, so a contribution is
judged first on whether it keeps the numbers the project produces honest.

The constraint behind most of what follows: **nothing this project owns computes
a reported number.** Scoring belongs to lm-evaluation-harness; the adaptation
loop is the only loop we own. The [README](README.md#what-this-is) has the rest.

## Ways to help

None of these needs a GPU. The test suite stubs the model and downloads nothing,
and the end-to-end smoke run uses a tiny random-weight checkpoint on CPU.

- **Report a bug or ask a question.**
  [Open an issue](https://github.com/ancoda-labs/TinyACE/issues). If a number
  looks wrong, include the command, the commit and the run's `metrics.json`,
  which records the environment, the seed and whether the tree was dirty.
- **Hold a line nothing tests.** `python -m scripts.mutation_check` lists edits
  the suite does not notice. A test that catches one is a self-contained first
  PR.
- **Read Nepali.** `GENERIC_PHRASES` in
  [memory/playbook.py](src/edge_slm_ace/memory/playbook.py) is English-only, so
  the vagueness score cannot recognise a vague Nepali lesson and the two
  languages are scored asymmetrically. Nepali entries, with a test, address a
  limitation the README has to state. Reading the lessons a Nepali run writes
  to `playbook.jsonl` and saying which ones are vague helps too.
- **Build a missing arm.** `tinyace_retrieval` and `generative_cot` are
  registered in [reporting/schema.py](src/edge_slm_ace/reporting/schema.py)
  with `implemented=False`: designed, with no runner.
- **Fix the docs.** A documented command that does not do what it says is a
  bug.

For anything bigger than a bug fix, open an issue before writing code: a new
arm or model, a new dependency, or any change to how something is prompted,
parsed, retrieved or scored.

## Setup

Python 3.10 or newer; CI tests 3.10 and 3.11.

```bash
git clone https://github.com/ancoda-labs/TinyACE.git
cd TinyACE
python -m venv .venv && source .venv/bin/activate
make install          # editable install with the dev, retrieval and report extras
make check            # tests + lint + format check, about 15 seconds
```

`make install` pulls in torch through lm-eval, so the first install is a large
download. Without `make` (on Windows, for example), each target in the
[Makefile](Makefile) is a command or two you can run directly.
[docs/installation.md](docs/installation.md) covers devices and troubleshooting.

```bash
make test                                        # the whole suite
python -m pytest tests/test_playbook.py -v       # one file
python -m pytest tests/ -k "<pattern>"           # tests whose names match
```

## Before opening a PR

```bash
make check
```

CI runs this and it can fail — it is a real gate, not decoration. CI also runs
more than `make check`:

```bash
python -m scripts.fetch_belebele --check
python -m scripts.run_arm --model tiny-gpt2 --language en --arm tinyace \
  --output-dir /tmp/arm --limit 4 --device cpu --no-chat-template
python -m scripts.run_arm --model tiny-gpt2 --language en --arm scaffold_control \
  --output-dir /tmp/control --limit 4 --device cpu --no-chat-template
python -m scripts.run_grid --models tiny-gpt2 --dry-run --results-root /tmp/grid
```

Run these as well if you touched `scripts/`, `harness/`, `adapt.py` or the arm
registry: they are the only checks that load a real model. The first run
downloads `sshleifer/tiny-gpt2` and takes around 20 seconds on CPU.

CI installs `[dev,report]` **without** `retrieval`, so tests must pass without
`sentence-transformers` installed.

In the PR:

- [ ] `make check` passes.
- [ ] A behaviour change comes with a test that fails without it (see below).
- [ ] `CHANGELOG.md` has an entry under the top `[Unreleased]` section saying
      what was wrong, what it affected and how it was fixed.
- [ ] Docs that name a changed command, flag or arm are updated.
- [ ] The description says whether existing results are affected.

## Changing anything that affects a number

This is a research repository, so the bar for the measurement path is higher
than for the rest of the code. If your change touches prompting, parsing,
metrics, retrieval or scoring:

1. **Add a test that fails without the change.** Every defect listed under
   Fixed in `CHANGELOG.md` was invisible for months because nothing asserted
   the intended behaviour — including one that *did* have a failing test, which
   nobody saw because CI could not fail.

   This is not checkable by reading: a test asserting what the code currently
   does passes either way, and looks identical to one that pins the behaviour
   down. The mutation check tests it mechanically — it flips a comparison,
   moves a constant, swaps a boolean, and reports any edit the tests do not
   fail on. Run it on the module you touched:

   ```bash
   python -m scripts.mutation_check --list              # which modules it covers
   python -m scripts.mutation_check --module playbook   # one module
   ```

   `make mutants` runs every module and takes minutes. `harness/` and
   `scripts/` are not covered; for those, show in the PR that your test fails
   on the code before your change.
2. **Say what it does to existing results** in the PR description. A change to
   answer extraction or option handling invalidates prior runs; say so.
3. **Do not report a delta without a significance test.** See below.

## Reporting results

Read [docs/evaluation.md](docs/evaluation.md) first, and go through its
[reporting checklist](docs/evaluation.md#reporting-checklist) before a number
appears anywhere: an issue, a PR or a paper. The three rules that previously
went wrong:

- **Compare against the right arm.** `tinyace` belongs against
  `scaffold_control`, not `baseline` — otherwise the delta also contains the
  instruction prefix. Ablations belong against `tinyace`, because an ablation
  against baseline measures ACE *plus* the ablation. `scripts/compare_arms.py`
  picks the registered reference automatically.
- **Use enough data.** At n=50 the 95% interval near 40% accuracy spans about
  ±13 percentage points, which is wider than any effect this project has
  reported. The evaluation split is 500 items; use at least three seeds.
- **Run the test.** A difference is a result only if it survives
  `python -m scripts.compare_arms`. If it does not, write "no detectable
  difference" — that is a finding, not a failure.

Check the run-health fields (`truncation_rate`, `tokens_dropped`,
`relevance_active`, `token_counts_exact`) before trusting any run. `make report`
surfaces them, `make report STRICT=1` fails on an invalidating one, and
`compare_arms` excludes an invalidated run from the test family rather than
testing it. See [docs/evaluation.md](docs/evaluation.md#health-checks-that-invalidate-a-run)
for which conditions invalidate a run and which only narrow what it supports.

## Adding a model or an arm

Open an issue first for either.

- **A model** is a `ModelSpec` in
  [utils/config.py](src/edge_slm_ace/utils/config.py). It enters the grid only
  through `make screen`, a pre-registered gate: the lower bound of its Wilson
  interval on Nepali has to clear 30%.
- **An arm** is registered in `ARMS` in
  [reporting/schema.py](src/edge_slm_ace/reporting/schema.py), which also sets
  the arm it is compared against. Keep it `implemented=False` until `run_arm`
  can produce it; `run_arm` refuses such an arm by name rather than writing
  another arm's result under its label. Add it to the arms tables in the README
  and [docs/evaluation.md](docs/evaluation.md).

## Style

- `black`, line length 100. `make format`.
- Docstrings on public functions: what it does, `Args`, `Returns`, and `Raises`
  where it raises.
- Comments explain *why*, especially where the obvious implementation is wrong.
  Several bugs here were introduced by code that looked correct.
- Match the surrounding file.

## Commits

`type(scope): subject` — `fix`, `feat`, `refactor`, `docs`, `test`, `chore`,
`style`. Explain in the body what was wrong and why the fix is right; if it
changes a measurement, say that prior results need regenerating.

## License

By contributing, you agree that your contribution is licensed under the Apache
License 2.0, the same as the project ([LICENSE](LICENSE), section 5).
