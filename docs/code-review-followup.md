# TinyACE — Follow-up Code Review

**Date:** 2026-08-18 · **Branch:** `fix/review-findings` @ `dbe3a4d` · **Scope:** `src/`, `scripts/`, `configs/`, `tests/`, `docs/`, packaging (~13.9k LOC)

This is a review of the codebase **as it stands today**, not a re-run of
[`code-review.md`](code-review.md). Most of what that audit found has genuinely been
fixed — option permutation, the OMA mapping cascade, chat templates, prompt-length
bounds, seeding, the FIFO/LIFO inversion, the store-vs-prompt budget split,
query-conditioned retrieval, and the shared `extract_answer` path are all in place and
carry comments explaining why.

What follows is what the *repairs themselves* left behind: two hard defects introduced
by the newest features, a set of protocols that the docs prescribe but no code
implements, and the structural debt that lets this class of bug survive.

**Status: all 14 findings addressed** in `f81aeae..334064f`, plus three defects
the fixes themselves surfaced — a shadowed `statistics` import that crashed
every ACE run of any length, a `context_tokens` column that meant different
things in different arms, and a `--playbook-mode frozen` path that had no
implementation. Each finding below carries the commit that closed it.

**Every claim below was reproduced against the working tree.** Reproduction commands
are in the [appendix](#appendix-reproducing-these-findings).

---

## Verdict

| # | Finding | Severity | Effect | Fixed in |
|---|---|---|---|---|
| [1](#1-every-ace-run-with---metrics-path-crashes-and-discards-its-results) | `playbook_log.csv` writer crashes every ACE run | 🔴 Blocking | All results lost, misreported as a model-load failure | `f81aeae` |
| [2](#2-the-used-strategies-line-is-parsed-into-the-answer) | `Used strategies:` line lands inside the scored answer | 🔴 Blocking | Contaminates exact match in the ACE arm only | `1795ebc` |
| [3](#3-the-frozen-playbook-protocol-is-documented-but-unimplemented) | Frozen-playbook protocol has no implementation | 🔴 Blocking | ACE still learns on the scored split | `d3cc5ac` |
| [4](#4-compare_armspy-ignores-reference_for-and-compares-everything-to-baseline) | `reference_for()` is exported, tested, and never called | 🔴 Blocking | Ablations compared against the wrong arm | `1795ebc` |
| [5](#5-the-curator-ablation-reports-a-constant-zero) | `lessons_rejected_by_curator` always sums to 0 | 🟠 High | The Curator ablation has no observable effect metric | `f81aeae` |
| [6](#6-run_ace_epochpy-reintroduces-the-gold-is-always-a-artefact) | `run_ace_epoch.py` never seeds or permutes options | 🟠 High | Gold at (A) 100% of the time; numbers not comparable | `1795ebc` |
| [7](#7-no-multiple-comparison-correction-across-a-11-arm-grid) | No multiple-comparison correction over 11 arms | 🟠 High | ~43% chance of a spurious "significant" result | `1795ebc` |
| [8](#8-one-broad-except-turns-every-runtime-failure-into-failed-to-load-model) | One `except` mislabels every runtime failure | 🟠 High | Real errors are undebuggable | `f81aeae` |
| [9](#9-pruning-runs-with-current_step0-so-recency-never-affects-it) | `prune()` called with `current_step=0` | 🟡 Medium | Recency term is constant; γ ablation is partly inert | `9d5db95` |
| [10](#10-the-vagueness-heuristic-rewards-hyphens) | Vagueness scorer rewards any hyphen as a "formula" | 🟡 Medium | Generic lessons escape the δ penalty | `9d5db95` |
| [11](#11-documented-scoring-hyperparameters-are-never-read) | `scoring:` YAML block is never read | 🟡 Medium | α/β/γ/δ/λ are decorative | `d3cc5ac` |
| [12](#12-runnerpy-has-no-tests-at-all) | Zero tests for `core/runner.py` | 🟠 High | Findings 1, 2 and 5 all live in the untested file | `f81aeae`, `4434f8b` |
| [13](#13-the-installed-package-cannot-find-its-own-data) | `pip install` produces a package that can't find `data/` | 🟡 Medium | Only editable installs work | `334064f` |
| [14](#14-the-grid-reloads-every-model-from-disk-for-every-cell) | Grid reloads each model per cell, batch size 1 | 🟡 Medium | Large avoidable GPU-hour cost | `334064f` |

---

## 1. Every ACE run with `--metrics-path` crashes and discards its results

**Where:** [`scripts/run_experiment.py:598-610`](../scripts/run_experiment.py#L598-L610)

`run_dataset_ace` appends ten keys per step to `playbook_log`
([`runner.py:1044-1058`](../src/edge_slm_ace/core/runner.py#L1045-L1058)), including
`curator_latency_ms` and `lessons_rejected_by_curator`, added with the Curator feature.
The `csv.DictWriter` that persists that log still declares only eight `fieldnames`.
`DictWriter` defaults to `extrasaction="raise"`:

```
ValueError: dict contains fields not in fieldnames:
  'curator_latency_ms', 'lessons_rejected_by_curator'
```

This fires on **every** `ace` and `cot_control` run that passes `--metrics-path` — which
is every cell `run_eval_grid.py` produces
([`run_eval_grid.py:99-110`](../scripts/run_eval_grid.py#L99-L110)).

It is worse than a crash. The write happens *after* the full evaluation, inside the
`try` block whose handler is finding 8, so a 1000-example run that took an hour on a GPU
exits `1` printing `Error: Failed to load model 'Qwen/Qwen2.5-7B-Instruct': dict contains
fields not in fieldnames`. The CSV, metrics JSON and predictions are never written. The
grid records it as a failure with a message about model loading.

**Fix.** Derive the header from the data and stop hand-maintaining it:

```python
fieldnames = list(playbook_log[0].keys())
```

Better: give the per-step log a dataclass alongside `ScoringParams` so the schema has one
definition. And narrow the `try` (finding 8) so a reporting bug can never destroy an
evaluation that already succeeded.

---

## 2. The `Used strategies:` line is parsed into the answer

**Where:** [`ace_roles.py:146-152`](../src/edge_slm_ace/core/ace_roles.py#L146-L152) vs
[`ace_roles.py:297-331`](../src/edge_slm_ace/core/ace_roles.py#L297-L331)

Credit assignment asks the Generator to name the strategies it applied, appending a
`Used strategies:` section *after* `Answer:`. `parse_generator_output` treats every
non-empty line after the `Answer:` header as answer content until a *recognised* header
appears — and `Used strategies:` is in neither `reasoning_keywords` nor `answer_keywords`.

```
>>> extract_answer("Reasoning:\nThe mitochondria makes ATP.\n\n"
...                "Answer:\nmitochondria\n\nUsed strategies:\n1, 3\n")
('mitochondria\nUsed strategies:\n1, 3', 'The mitochondria makes ATP.')
```

The scored `pred` is `"mitochondria\nUsed strategies:\n1, 3"`. Consequences, in order of
severity:

- **Exact match (`is_correct`) is destroyed in the ACE arm and intact in the baseline
  arm**, because only ACE prompts request the citation. This is precisely the
  arm-asymmetric parsing that `extract_answer` was written to eliminate — the same class
  of defect, reintroduced one layer up.
- `semantic_score` and `bleu_score` are dragged down by the same trailing text.
- OMA mostly survives (the `letter`/`option_text` tiers still fire), but the appended
  digits `1, 3` can now match `_WEAK_CHOICE_PATTERNS` in edge cases.

The `cot_control` arm is affected too, but *only when its playbook is non-empty* — and it
never is — so the citation block is emitted for ACE and not for its own control. The
`ace − cot_control` delta the protocol is built around is measured across a parser
asymmetry.

**Fix.** Add `"used strategies:"` to a terminator set in `parse_generator_output` so the
answer section closes at it, and unit-test round-tripping a full generator output
including the citation block. Alternatively, request the citation on a line *before*
`Answer:`, so the answer section stays last.

---

## 3. The frozen-playbook protocol is documented but unimplemented

**Where:** [`docs/evaluation.md:49-60`](evaluation.md), [`utils/config.py:166-168`](../src/edge_slm_ace/utils/config.py#L166-L168), [`configs/experiment_grid.yaml:83`](../configs/experiment_grid.yaml#L83)

Three places state the protocol: build a playbook on `sciq_val`, **freeze it**, evaluate
read-only on `sciq_test`. `evaluation.md` is explicit that the alternative is online
continual learning, and that it is hard to defend here because 82% of `sciq_test`
examples contain the gold answer verbatim in `support`.

No code implements it:

- There is no `--freeze-playbook` / `--no-learn` flag. `enable_learning` is bound to
  `args.mode == "ace"` ([`run_experiment.py:509`](../scripts/run_experiment.py#L509)) —
  the only way to stop writing is to become the control arm and lose the playbook
  entirely.
- `run_eval_grid.py` writes the playbook to `output_dir / "playbook.jsonl"`
  ([`run_eval_grid.py:127`](../scripts/run_eval_grid.py#L127)), and `output_dir` is
  per-`{model, task, mode, device}`. Every cell therefore starts empty; nothing carries a
  `sciq_val` playbook into a `sciq_test` cell.
- The grid config lists `sciq_val` and `sciq_test` as two independent tasks, so the
  default run learns on the scored split — the exact protocol the docs argue against.

The repo is one flag away from being able to do what it says. Until that flag exists, the
documentation describes an experiment that cannot be run with the shipped tools.

**Fix.** Two arguments to `run_experiment.py`:

```
--playbook-mode {learn,frozen}   # frozen: retrieve and score, never write or reflect
--init-playbook PATH             # seed from a previous run, distinct from --playbook-path
```

Then let `run_eval_grid.py` express a dependency between cells (`depends_on: <cell>`), or
ship a two-stage `make adapt && make evaluate`.

---

## 4. `compare_arms.py` ignores `reference_for()` and compares everything to baseline

**Where:** [`scripts/compare_arms.py:29`](../scripts/compare_arms.py#L29) and
[`:79-84`](../scripts/compare_arms.py#L79-L84)

`reporting/schema.py` defines the correct reference for each arm — ablations against
`tinyace_wm_256`, ACE arms against `cot_control` — with a comment explaining that
comparing an ablation to baseline measures *ACE plus the ablation*. `docs/evaluation.md`
states: "`scripts/compare_arms.py` uses these defaults automatically."

It does not. The import is dead:

```python
from edge_slm_ace.reporting import arm_label, is_ablation, reference_for  # none are called
```

`--reference` defaults to the literal string `"baseline"` and every discovered arm is
compared against it. So the finding that `code-review.md` logged as **S6 — Ablations are
compared against the wrong reference arm** is, in behaviour, still open. It is masked
because `reference_for` *is* tested
([`tests/test_reporting.py:86-105`](../tests/test_reporting.py#L86-L105)) — the tests
pass, the docs claim the behaviour, and the tool does something else.

This is the pattern worth naming: **the registry is right, the tests are right, the wiring
is missing.** Findings 4, 5 and 11 are all instances of it.

**Fix.** Default `--reference` to `None` and, when unset, resolve each arm's reference via
`reference_for(arm_key)` from the directory layout. Add a test that runs `main()` over a
fixture results tree and asserts an ablation was compared against `tinyace_wm_256`. Note
that CI's flake8 selector (`E9,F63,F7,F82`) does not include `F401`, which is why three
unused imports sit in a file whose correctness depends on using them.

---

## 5. The Curator ablation reports a constant zero

**Where:** [`runner.py:1201-1203`](../src/edge_slm_ace/core/runner.py#L1201-L1203)

```python
"lessons_rejected_by_curator": sum(
    r.get("lessons_rejected_by_curator", 0) for r in results
),
```

`results` holds per-example rows. `lessons_rejected_by_curator` is only ever written into
`playbook_log` ([`runner.py:1056`](../src/edge_slm_ace/core/runner.py#L1056)), never into
a result row. The sum is over a key that does not exist, so it is always `0`.

`tinyace_ablate_no_curator` is a registered arm with a dedicated grid cell. Its whole
purpose is to show what the Curator screens out — and the metric that would show it is
hard-wired to zero. `curator_latency_ms` is logged per step but never aggregated into the
summary at all, so the Curator's cost is invisible too.

**Fix.** Aggregate from `playbook_log`, and surface the cost alongside the benefit:

```python
"lessons_rejected_by_curator": sum(r["lessons_rejected_by_curator"] for r in playbook_log),
"total_curator_latency_ms": sum(r["curator_latency_ms"] for r in playbook_log),
```

---

## 6. `run_ace_epoch.py` reintroduces the "gold is always (A)" artefact

**Where:** [`scripts/run_ace_epoch.py:192`](../scripts/run_ace_epoch.py#L192) and
[`:215`](../scripts/run_ace_epoch.py#L215)

Three of the four entrypoints call `set_seed()` and pass `option_shuffle_seed`.
`run_ace_epoch.py` does neither:

```
run_experiment.py    set_seed ✓   option_shuffle_seed ✓
smoke_test.py        set_seed ✓   option_shuffle_seed ✓
run_qwen_rivals.py   set_seed ✓   option_shuffle_seed ✓
run_ace_epoch.py     set_seed ✗   option_shuffle_seed ✗
```

With `option_shuffle_seed=None`, `permutation_for` returns the identity permutation
([`mcq.py:63-64`](../src/edge_slm_ace/eval/mcq.py#L63-L64)) and every SciQ row stores gold
first — so this script presents the correct answer as option (A) for 100% of examples. It
is the flagship finding of the previous audit (S1), alive in a script that is not
otherwise deprecated. The same file also re-implements dataset-path resolution
([`:114-115`](../scripts/run_ace_epoch.py#L114-L115)) instead of calling
`resolve_task_path`, so it breaks from any directory other than the repo root.

**Fix.** Either delete the script — `run_experiment.py --mode ace` invoked twice does the
same job correctly — or route it through the same seeded path. A test that asserts every
`scripts/*.py` calling `run_dataset_*` passes `option_shuffle_seed` would prevent the
fourth recurrence.

---

## 7. No multiple-comparison correction across a 11-arm grid

**Where:** [`scripts/compare_arms.py:172-177`](../scripts/compare_arms.py#L172-L177),
[`eval/stats.py:130`](../src/edge_slm_ace/eval/stats.py#L130)

The statistics module is the strongest part of this codebase: Wilson intervals, an exact
paired McNemar test, honest verdict strings. But `compare_arms.py` runs one test per arm
against the reference and reports `significant_05` per comparison, uncorrected.

`experiment_grid.yaml` defines 11 modes. Ten tests at α=0.05 give a family-wise error rate
of 1 − 0.95¹⁰ ≈ **40%** under the null. Across 6 models and 2 tasks it is a near-certainty
that some cell shows p < 0.05 from noise alone. The summary line —
`"N of M comparisons are distinguishable from noise at p<0.05"` — reads as a count of
findings and is exactly where the correction should be applied.

**Fix.** Add Holm–Bonferroni (10 lines, stdlib, fits `stats.py`'s no-optional-deps rule)
and report both `p_value` and `p_adjusted`. Make the summary line say
`"N of M significant after Holm correction (M tests in family)"`. Pre-register the primary
comparison — `tinyace_wm_256` vs `cot_control` on `sciq_test` — and label everything else
exploratory.

---

## 8. One broad `except` turns every runtime failure into "Failed to load model"

**Where:** [`scripts/run_experiment.py:621-623`](../scripts/run_experiment.py#L621-L623)

The `try` opened at line 437 wraps model loading, dataset loading, schema validation, the
*entire* evaluation loop, playbook I/O and the log CSV write. Its handler:

```python
except Exception as e:
    print(f"Error: Failed to load model '{config.model_id}': {e}")
    return 1
```

Any failure anywhere in a multi-hour run is reported as a model-load failure, with no
traceback, and every result is discarded. Finding 1 is the concrete instance; a malformed
dataset row at example 900 or a transient CUDA OOM behaves identically.

Compounding it: there is no incremental persistence. Predictions are buffered in memory
and written only at the very end, so nothing survives an interruption.

**Fix.**
1. Wrap *only* `load_model_and_tokenizer` in the load handler.
2. Let evaluation failures propagate to the outer handler at line 754, which already
   prints a traceback.
3. Stream predictions to `predictions.jsonl` as they are produced, and add `--resume` to
   skip `qid`s already present. At 1000 examples × 11 arms × 6 models, losing a run to a
   reporting bug is expensive in a way that a few lines of append-mode I/O is not.

---

## 9. Pruning runs with `current_step=0`, so recency never affects it

**Where:** [`runner.py:1018`](../src/edge_slm_ace/core/runner.py#L1018) vs
[`playbook.py:282-283`](../src/edge_slm_ace/memory/playbook.py#L282-L283)

```python
playbook.prune(max_entries_per_domain=max_entries_per_domain)   # current_step defaults to 0
```

Inside `score()`:

```python
age = max(0, current_step - self.last_used_at) if current_step > 0 else 0
recency_term = params.gamma * math.exp(-params.lambda_decay * age)
```

With `current_step=0`, every entry gets `age=0` and therefore the identical recency bonus
`γ`. A constant added to every candidate cannot change a ranking, so **pruning ignores
recency entirely** — the eviction decision runs on success/failure ratio and vagueness
alone.

This quietly weakens the `tinyace_ablate_no_recency` arm: setting γ=0 changes retrieval
ranking but changes nothing about which entries survive pruning. An ablation that reports
"no effect" is partly reporting that half its mechanism was already inert.
`get_stats()` has the same defect ([`playbook.py:873`](../src/edge_slm_ace/memory/playbook.py#L873)),
so reported `avg_score` is not the score any decision used.

**Fix.** Pass `current_step=step` at the call site and make the parameter required. The
`if current_step > 0` guard then becomes unnecessary and should go — it silently converts
a caller mistake into a plausible-looking number.

---

## 10. The vagueness heuristic rewards hyphens

**Where:** [`playbook.py:121-123`](../src/edge_slm_ace/memory/playbook.py#L121-L123)

```python
has_formula = any(sym in text for sym in ["=", "+", "-", "*", "/", "%", "^"])
```

`-` matches any hyphen in ordinary prose, and `/` matches any slash. Both grant a −0.15
specificity credit:

```
0.25  "Think carefully about the well-known question"
0.70  "Pay attention to details"
0.00  "For density, convert g/cm3 to kg/m3 by multiplying 1000"
```

The first is textbook-generic — it contains a literal `GENERIC_PHRASES` entry — yet scores
0.25, comfortably under the `is_generic` threshold of 0.5, because "well-known" reads as a
formula and the sentence is long enough to dodge the short-text penalty. Note this is the
*exact* phrase the Reflector prompt gives as its first "BAD lesson" example.

δ is a term in the headline retention equation and has a dedicated ablation arm. A
threshold that a hyphen can flip is not measuring what the equation claims.

**Fix.** Require a digit adjacent to the operator (`re.search(r"\d\s*[-+*/^=]|[-+*/^=]\s*\d", text)`),
drop `-` and `/` as standalone signals, and make the length penalty apply to the phrase
check rather than being independent of it. Then add a table-driven test with ten
hand-labelled lessons — the current tests only exercise the extremes.

---

## 11. Documented scoring hyperparameters are never read

**Where:** [`configs/experiment_grid.yaml:238-252`](../configs/experiment_grid.yaml#L238-L252)

The config ships a `scoring:` block with α, β, γ, δ, λ and ε, annotated as matching the
formal retention equation. Nothing reads it: `build_experiment_command` never forwards it,
and `run_experiment.py` has no CLI flags for any of the six. Every run uses the
`DEFAULT_*` constants in `playbook.py`, whatever the YAML says.

`reflect_on_correct_every_n` has the same problem in reverse — `run_dataset_ace` accepts
it and only `run_qwen_rivals.py` passes it, so `run_experiment.py` and the grid are stuck
at the default of 5 with no way to vary it.

Editing the YAML changes nothing and reports no error. That is worse than the knob being
absent.

**Fix.** Add `--alpha/--beta/--gamma/--delta/--lambda-decay/--epsilon` and
`--reflect-on-correct-every-n`, forward them from the grid, and echo the resolved
`ScoringParams` into `metrics.json`. Then add a config-validation pass that fails loudly
on any YAML key no consumer reads — the same guard `validate_task_registry()` provides for
task paths, applied to hyperparameters.

---

## 12. `runner.py` has no tests at all

**Where:** [`tests/`](../tests/)

| Module | LOC | Tests |
|---|---:|---:|
| `core/runner.py` | 1273 | **0** |
| `memory/playbook.py` | 876 | 42 |
| `eval/mcq.py` | 749 | 40 |
| `core/ace_roles.py` | 756 | 30 |
| `eval/metrics.py` | 713 | 0 |
| `scripts/*` (5.6k LOC) | — | **0** |

175 tests, and none touch the largest file — the one that decides what "correct" means,
assembles every result row, and is the sole difference between the arms being compared.
Findings 1, 2 and 5 all live there or in its immediate consumers. This is not a
coincidence: it is the only part of the pipeline where a mistake cannot be caught.

The scripts are untested too, which is why finding 4 (a dead import in a 190-line script)
survived a review that explicitly checked for it.

**Fix.** `runner.py` is testable without a GPU — inject a stub model/tokenizer that
returns canned strings:

```python
def test_ace_answer_excludes_citation_block():
    model, tok = StubModel(["Reasoning:\n...\n\nAnswer:\nmitochondria\n\nUsed strategies:\n1"]), StubTokenizer()
    results, _ = run_dataset_ace(model, tok, [EXAMPLE], "science", CONFIG, Playbook(), TMP, "m", "sciq_tiny")
    assert results[0]["pred"] == "mitochondria"          # catches finding 2

def test_baseline_and_ace_rows_share_a_schema():
    assert set(baseline_row) - set(ace_row) == set()      # catches silent column drift
```

Two of the four blocking findings would have been caught by ten such tests. Also add a
`--limit 2` smoke run of `run_experiment.py --mode ace --metrics-path ...` to CI on
`tiny-gpt2`, which catches finding 1 directly.

**Related CI gaps:**
- `flake8` selects only `E9,F63,F7,F82` — no `F401`, so the dead imports of finding 4 pass.
- `test_model_manager.py` downloads `sshleifer/tiny-gpt2` from the Hub with no offline
  fallback and no `pytest.importorskip`, so CI depends on Hub availability.
- `test_generate` asserts `len(output) > 0` from a randomly-initialised model; a
  whitespace-only generation strips to `""` and fails spuriously.
- CI never runs `make smoke`, so no end-to-end path is exercised.

---

## 13. The installed package cannot find its own data

**Where:** [`utils/config.py:140`](../src/edge_slm_ace/utils/config.py#L140),
[`pyproject.toml`](../pyproject.toml)

```python
REPO_ROOT = Path(__file__).resolve().parents[3]
```

Correct for an editable install (`src/edge_slm_ace/utils/` → repo root). For a real
`pip install`, `parents[3]` is somewhere above `site-packages`, and `data/tasks/` is not
packaged at all — `[tool.setuptools.packages.find] where = ["src"]` excludes it, and there
is no `package-data` or `MANIFEST.in`. `scripts/` is likewise outside the wheel, so
`python -m scripts.run_experiment` only works from a checkout.

The `pyproject.toml` advertises an installable distribution with a homepage and issue
tracker; what it builds cannot run its own quickstart.

**Related packaging drift:**
- `requirements.txt` and `pyproject.toml` are maintained separately and disagree —
  `sacrebleu>=2.3.1` vs `>=2.3.0`; `statsmodels` is in the `metrics` extra and is not
  imported anywhere in the repo; `accelerate` is a hard dependency but appears in no
  import.
- `paper/tinyace.pdf` is committed as a binary in a repo whose own docs mark the results
  it contains as withdrawn.

**Fix.** Either declare the project checkout-only and say so in `installation.md`, or move
`data/tasks/` under `src/edge_slm_ace/data/`, add `[tool.setuptools.package-data]`, resolve
via `importlib.resources`, and expose entry points (`tinyace-run = ...`) instead of
`python -m scripts.*`. Drop `statsmodels`; pin the resolved versions in a lockfile as
`requirements.txt`'s own header already advises.

---

## 14. The grid reloads every model from disk for every cell

**Where:** [`run_eval_grid.py:96-100`](../scripts/run_eval_grid.py#L96-L100),
[`runner.py:188-229`](../src/edge_slm_ace/core/runner.py#L188-L229)

`run_eval_grid.py` shells out to `python -m scripts.run_experiment` per cell. The default
grid is 6 models × 2 tasks × 11 modes = **132 subprocesses**, each loading a checkpoint
(up to 7B) from scratch. Generation is one example at a time, batch size 1, and the ACE
arm makes up to three sequential calls per example (generate → reflect → curate).

Rough order of magnitude at 1000 examples: the ACE arms alone are ~2M forward passes, all
unbatched, with ~130 redundant model loads on top. For a project whose thesis is *edge
feasibility*, the evaluation harness is the least efficient part of the system.

There is also no resume: re-running the grid re-runs completed cells from scratch, which
combines badly with finding 1.

**Fix, in impact order.**
1. **Skip completed cells** — if `metrics.json` exists and its `git_commit` matches, skip
   unless `--force`. One afternoon of work; the single biggest saving given the current
   crash rate.
2. **Group by model** — invert the loop to `for model: load once; for task/mode: run`, and
   run the grid in-process. Saves ~125 model loads.
3. **Batch generation** — `generate()` already tokenizes a single string; accepting a list
   and padding left is a contained change, and is where the 10× lives.
4. **Cache the Reflector** — reflection is triggered on every incorrect answer; identical
   `(question, wrong answer)` pairs recur across arms.

---

## What is working well

Worth stating plainly, because the fixes below should not disturb it:

- **`eval/stats.py`** is genuinely good — Wilson intervals, exact paired McNemar, pure
  stdlib so it can never be skipped for a missing dependency, and verdict strings that say
  "no detectable difference" instead of implying an ordering. Finding 7 is the only gap.
- **The mapping-tier cascade** in `mcq.py` is the right design: letter → option text →
  weak letter → embedding, with the tier recorded per example so a weak OMA is visible
  rather than inferred.
- **Generation provenance** (`prompt_truncated`, `used_chat_template`) travels on every
  row and is surfaced as run-health warnings during aggregation. That is the correct
  instinct — make invalidating conditions data, not scrollback.
- **`reporting/schema.py`** as a single vocabulary for arm and model labels, with the
  reasoning for each arm's existence in the registry itself.
- **The comments explain *why*.** `_resolve_duplicate`, `store_token_capacity`,
  `extract_answer` and the `cot_control` docstring each explain the failure mode they
  exist to prevent. That is rare and it is what made this review tractable.
- **`docs/`** is honest: results are marked withdrawn rather than quietly deleted, and
  `evaluation.md` states the protocol's weaknesses in its own text.

---

## Recommended order

**Day 1 — unblock (findings 1, 8, 5)**
Derive `fieldnames` from the log dict. Narrow the `try` around model loading. Aggregate
the Curator counters from `playbook_log`. Without these, no ACE run completes.

**Day 2 — restore measurement integrity (findings 2, 4, 6)**
Terminate the answer section at `Used strategies:`, with a test. Wire `reference_for()`
into `compare_arms.py`, with a test. Delete or fix `run_ace_epoch.py`. These three
determine whether any number produced is comparable across arms.

**Week 1 — close the protocol gap (findings 3, 7, 11, 12)**
Add `--playbook-mode frozen` and `--init-playbook`, and a two-stage adapt→evaluate target
in the Makefile. Add Holm correction and pre-register the primary comparison. Wire the
`scoring:` block. Write the first 10-15 `runner.py` tests around a stub model, and add
`F401` to the flake8 selector.

**Week 2 — before the next expensive run (findings 14, 9, 10)**
Cell-skipping and model-grouping in the grid, streaming predictions with `--resume`. Pass
`current_step` to `prune()`. Fix the vagueness operator check. Then re-run.

**Not urgent (finding 13)**
Decide whether this is an installable package or a research checkout, and make the
packaging metadata agree with the answer.

---

## Appendix: reproducing these findings

```bash
# 1 — the DictWriter crash
python - <<'PY'
import csv, io
log = [{"step_index":1,"num_entries":2,"total_tokens":30,"num_evictions":0,
        "retention_score_std":0.0,"num_retrieved":1,"num_credited":1,
        "credit_mode":"cited","curator_latency_ms":12.0,"lessons_rejected_by_curator":1}]
w = csv.DictWriter(io.StringIO(), fieldnames=["step_index","num_entries","total_tokens",
    "num_evictions","retention_score_std","num_retrieved","num_credited","credit_mode"])
w.writeheader(); w.writerows(log)     # ValueError: dict contains fields not in fieldnames
PY

# 2 — the citation block lands in the answer
python -c "
import sys; sys.path.insert(0,'src')
from edge_slm_ace.core.ace_roles import extract_answer
print(extract_answer('Reasoning:\nX makes ATP.\n\nAnswer:\nmitochondria\n\nUsed strategies:\n1, 3'))"

# 4 — reference_for is imported and never called
grep -n 'reference_for\|is_ablation' scripts/compare_arms.py

# 5 — the key is never written to a result row
grep -n 'lessons_rejected_by_curator' src/edge_slm_ace/core/runner.py

# 6 — one entrypoint out of four is unseeded
grep -n 'option_shuffle_seed\|set_seed' scripts/*.py

# 10 — a generic lesson scores below the threshold
python -c "
import sys; sys.path.insert(0,'src')
from edge_slm_ace.memory.playbook import compute_vagueness_score as v
print(v('Think carefully about the well-known question'))   # 0.25, threshold is 0.5"

# 11 — nothing reads the scoring block
grep -rn 'alpha\|lambda_decay' scripts/run_experiment.py scripts/run_eval_grid.py

# 12 — no tests for the largest module
ls tests/ | grep -i runner
```
