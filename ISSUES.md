# Known Issues

Audit of `main` at `f55d3b4` (2026-10-15). Ranked by how much each issue can
affect a reported number. Each entry says how it was checked.

Local check results: `flake8`, `black --check` and `freeze_splits --check` pass.
`pytest` gives 439 passed, 1 failed, 6 errors; all 7 come from the environment
(see #12), not from the code.

---

## Critical: these change what the study measures

### 1. The frozen split manifests are verified but never used

`data/splits/*.json` are checksummed in CI (`make splits`), but no runner reads
them. `manifest_ids()` ([splits.py:219](src/edge_slm_ace/data/splits.py#L219))
has no callers. Every entry point builds its split again from `--seed`:

- [run_arm.py:170](scripts/run_arm.py#L170) and
  [screen.py:82-90](scripts/screen.py#L82-L90) call `study_split(..., seed)`.
- The Belebele manifest records `seed: 42`. The Makefile's multi-seed recipe
  (`make grid SEED=43`) therefore gets a **different adaptation/evaluation
  split for each seed**, and only seed 42 matches the frozen file.
- In `screen.py`, the Global-MMLU split is `parallel_split` over the full test
  set at `--seed`. It does not use the frozen `global_mmlu_eval_2000` /
  `global_mmlu_adapt_400` subsample (seed `20260915`), so screening runs on a
  different set of items from the one that was frozen.

**Fix:** load the ids with `manifest_ids()` in `run_arm`, `screen` and
`run_grid`. Keep `--seed` for model and sampling RNG only.

### 2. The adaptation-loop leak guard is never turned on

`adapt_playbook(..., eval_manifest=None)` skips `assert_not_in_eval`. Its
docstring says *"None … is for tests only; a real run passes it."* The only
production caller, [run_arm.py:206-212](scripts/run_arm.py#L206-L212), does not
pass it. The per-item G3 guard from PR #2 therefore never runs in a real run.

**Fix:** pass `eval_manifest=BELEBELE_SPLIT` from `run_arm`, and add a test
that fails when a real entry point omits it.

### 3. The headline research question is not implemented

The README asks whether adaptation survives
`T_max ∈ {256, 512, 1024}` context ceilings. No runner sets a context ceiling,
`token_budget` or `store_token_capacity`. The playbook docstring
([playbook.py:7-12](src/edge_slm_ace/memory/playbook.py#L7-L12)) and
[docs/api.md:127](docs/api.md) both say *"no runner sets either."* Prompts run
at the model's native max length, and the frozen prefix is limited by `--top-k`
lessons, not by tokens. As the code stands, the grid cannot answer RQ1.

### 4. `mcnemar_exact` crashes once there are about 1,030 discordant pairs

[stats.py:58-60](src/edge_slm_ace/eval/stats.py#L58-L60) computes
`math.comb(n, i) * 0.5**i * 0.5**(n-i)`. For large `n`, `comb` is too large to
convert to a float:

```
n_disc=1000 -> p=1.0
n_disc=1030 -> OverflowError: int too large to convert to float
```

The Global-MMLU evaluation split has 2,000 items. Two near-chance arms, or a
cross-model or cross-language comparison, can reach this. **Fix:** compute in
log space (`math.lgamma`) or with exact `Fraction`/integer arithmetic
(`sum(comb(n, i) for i ≤ k) / 2**n` is exact).

---

## High: wrong outputs in specific conditions

### 5. `truncation_rate` is under-reported by 4×

[evaluate.py:202](src/edge_slm_ace/harness/evaluate.py#L202) divides by
`n_items * 4`. lm-eval's `HFLM` groups one-token continuations by context
(`group_by="contexts"` while `logits_cache` is on, which is the default). It
runs one forward pass per item and logs the truncation warning **once per
item**, not once per choice. So a run that truncates every item reports 25%.
The invalidation gate keys on `truncated_prompts > 0` and is not affected, but
the logged and printed rate is wrong.

Also, `TruncationCounter` wraps only the evaluation. Truncation during
adaptation scoring (`OptionScorer.score`) is never counted.

### 6. Screening results for one language overwrite the other's

`screen.py` always writes `results/screening/screening.json`
([screen.py:219](scripts/screen.py#L219)). The language is not part of the
filename. `run_grid.survivors()`
([run_grid.py:172-191](scripts/run_grid.py#L172-L191)) does not check
`payload["language"]`. Run `screen --language en` after the Nepali screen, and
the grid admits models on their **English** results. The pre-registered rule
requires a Nepali result.

### 7. The Reflector parser drops valid lessons that start with "Two", "Lessons" or "Strategies"

`_PREAMBLE_RE` ([ace_roles.py:25-28](src/edge_slm_ace/core/ace_roles.py#L25-L28))
matches these words at the start of any candidate, bulleted ones included:

```python
parse_reflector_output_to_lessons(
  "- Two options may paraphrase the passage; prefer the one that keeps the causal direction.\n"
  "- Lessons from the passage title can mislead; check the sentence that answers the question.")
# -> []
```

"Two options …" is close to the curator prompt's own SPECIFIC example. **Fix:**
apply the preamble filter only to un-bulleted fallback lines, or only to lines
ending in `:`.

### 8. When the Curator's output can't be parsed, every lesson passes silently

[ace_roles.py:176](src/edge_slm_ace/core/ace_roles.py#L176) treats every lesson
as not generic unless the model writes exactly `Lesson N: is_generic=…`. Small
models often answer in a different format:

```python
parse_curator_output(2, "1. GENERIC\n2. SPECIFIC")  # -> [False, False]
```

In that case the Curator arm and `--no-curator` give the same result, and
nothing records this. **Fix:** log a per-step `curator_parsed` count in
`adaptation_log.csv` and report the parse rate in `metrics.json`.

### 9. A correct answer is reflected on with the "answered incorrectly" prompt

When `reflect_on_correct_every_n > 0`, correct items go through
`build_reflector_prompt` ([adapt.py:56-98](src/edge_slm_ace/adapt.py#L56-L98)).
That prompt says *"was answered incorrectly"* and asks which trap was fallen
into, while showing `Chosen` = `Correct`. The default is `0` and no CLI flag
sets it, so this is not active yet, but the parameter is broken as written.

---

## Medium: scope, reproducibility, documentation

### 10. Global-MMLU is screened and frozen but never studied

`run_arm` and `run_grid` handle Belebele only: `load_belebele`, `benchmark`
left at its default, and a reading-comprehension Reflector prompt.
`render_question` uses the Belebele `P:/Q:` template. Global-MMLU appears only
in `screen.py`, so its frozen manifests and task YAMLs are not used by any arm.

### 11. Arm settings missing from `metrics.json`

`top_k`, `no_curator`, `adaptation_size`, `limit` and the `init_playbook` path
are not recorded ([run_arm.py:297-327](scripts/run_arm.py#L297-L327)). A
`--limit` debug run can't be told apart from a full run in the results tree,
except by `n_eval`.

### 12. Tests depend on network and HF cache, and are not marked

`tests/test_data.py::…test_the_loaded_tasks_have_no_fewshot_split` and the six
`TestHarnessParity` tests load tasks and tokenizers from the Hub, through
`HF_HOME`. They fail (here, because `HF_HOME` points at an unmounted
`/Volumes/swarr-data`) instead of skipping. **Fix:** add a `network` marker and
skip when the Hub or cache is unavailable.

### 13. The README is out of date

- The CI badge links `workflows/ci.yml`, but the file is `ci.yaml`.
- The tree lists `configs/`, which does not exist.
- It lists "Bootstrap CIs" and "Device/OOM management". Neither exists:
  `stats.py` has Wilson, McNemar and Holm only.
- It describes `adapt.py` as a "dual-pass (Trace + Loglik)" engine. Only the
  loglik track exists, and `generative_cot` is `implemented=False`.
- It says "380+ passing" tests; the suite has 446.
- The README stops after the tree (35 lines). It has no install, quickstart or
  results-status section, even though `docs/` covers all three.
- The "~4.5× cl100k" Devanagari token-inflation figure has no source and no
  script that reproduces it.

### 14. Smaller issues

- `compare_arms --reference X` matches X as a substring
  ([compare_arms.py:160](scripts/compare_arms.py#L160)). `--reference tinyace`
  treats every `tinyace_*` ablation as a reference as well.
- The two data sources are checked for matching ids, not matching content.
  Adaptation reads `data/tasks/*.jsonl` and evaluation reads `facebook/belebele`
  from the Hub. `per_item_correctness` checks that the ids agree, but it would
  not notice text differences after an upstream revision. Pin the dataset
  `revision` in the task YAMLs.
- Feedback credit is spread evenly over all retrieved lessons
  ([adapt.py:215-221](src/edge_slm_ace/adapt.py#L215-L221)). This is
  acknowledged in a comment, but it means the success/failure terms in the
  retention score are weak, shared signals. Ablations of β should be read with
  this in mind.
- `scripts/screen_models.py` is a back-compat shim with no remaining callers.
