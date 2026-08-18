# Evaluation protocol

How a number gets produced, and what has to be true for it to mean anything.

- [Arms](#arms)
- [Splits](#splits)
- [Metrics](#metrics)
- [Statistics](#statistics)
- [Run health](#run-health)
- [Worked example](#worked-example)

---

## Arms

Defined once in [`edge_slm_ace/reporting/schema.py`](../src/edge_slm_ace/reporting/schema.py).
Everything that renders a table or a figure reads its labels and ordering from
that registry.

| Arm | What it is |
|---|---|
| `baseline` | Terse prompt, no playbook |
| `cot_control` | **The ACE prompt scaffold over an empty playbook** |
| `ace_full` | Top-k retrieval, unbounded playbook |
| `ace_working_memory` | Token-budgeted working memory (`tinyace_wm_256`, `tinyace_wm_512`) |
| `tinyace_wm_256_frozen` | **The frozen-playbook arm.** Adapted on `sciq_val`, scored read-only on `sciq_test` |
| `tinyace_ablate_*` | One scoring component disabled |
| `tinyace_fifo` | Oldest-first eviction instead of lowest-score |
| `self_refine` | Critique and rewrite from the model's own output only |
| `self_refine_oracle` | Same, but shown the gold answer. An **upper bound**, not a baseline |

### Which arm to compare against

This is the part that is easy to get wrong, so the code answers it for you via
`reporting.reference_for()`:

| Arm being evaluated | Correct reference | Why |
|---|---|---|
| `ace_*` | `cot_control` | `ace − baseline` also varies the CoT instruction, the domain hints and the answer parser. Only `ace − cot_control` isolates the playbook. |
| `tinyace_ablate_*`, `tinyace_fifo` | `tinyace_wm_256` | Comparing an ablation to `baseline` measures *ACE plus the ablation*, not the ablated component. |
| `cot_control`, `self_refine` | `baseline` | These are prompting strategies against no strategy. |

`scripts/compare_arms.py` uses these defaults automatically: with no
`--reference`, each arm is paired with the reference `reference_for()` names for
it, and an arm whose reference was not run in that cell is skipped with a note
rather than re-pointed at another arm. Pass `--reference` to override.

Every comparison printed in one invocation is one **family** of tests and is
Holm-corrected. Ten arms against a reference give roughly a 40% chance of at
least one p<0.05 under the null, so the adjusted p-value is the one that
decides. Splitting a sweep across several invocations to shrink the family does
not make the correction smaller -- it hides the count.

---

## Splits

`sciq_val` and `sciq_test` share no questions (verified), which is what makes a
clean protocol possible:

1. Build a playbook on **`sciq_val`**.
2. Freeze it.
3. Evaluate read-only on **`sciq_test`**.

This is the `tinyace_wm_256_frozen` arm, and it runs in two stages because
stage 2 reads what stage 1 leaves behind:

```bash
make adapt      # stage 1: every learning arm on sciq_val
make evaluate   # stage 2: sciq_test, including the frozen arm
```

Directly, that is `--playbook-mode frozen --init-playbook <path>`. A frozen run
retrieves and ranks from the playbook but never reflects, records feedback,
prunes or writes, so the same playbook scores every example and the run cannot
learn from the split it is being scored on.

Running ACE directly on the scored set is *online continual learning*: item *i*
only sees gold from items *< i*, which is defensible but must be argued
explicitly. Two things make it harder to defend here — there is no held-out
split, and 82% of `sciq_test` examples contain the gold answer verbatim in
their `support` field, so a lesson written after seeing gold can carry answer
content forward. Prefer the frozen-playbook protocol.

---

## Metrics

### Option-Mapped Accuracy (OMA) — primary

Maps a free-text prediction onto one of the four options, then checks it against
gold. The mapping is a cascade, most to least trustworthy, and the tier that
fired is recorded per example as `mapping_tier`:

| Tier | Trigger |
|---|---|
| `letter` | A labelled letter — `Answer: B` |
| `option_text` | An option quoted verbatim, matched on word boundaries |
| `weak_letter` | A bare `(B)` or a trailing `B` |
| `embedding` | Cosine argmax over option embeddings — last resort |

**Read `mapping_tier_distribution` before reading OMA.** A run resolved mostly
at `embedding` is one where the models did not answer in a parseable form, and
its OMA is weak evidence rather than accuracy.

### Option position

Options are permuted per example with a seed derived from `(run_seed,
example_id)`. Both datasets store gold first, so presenting them in storage
order puts the answer at (A) 100% of the time and rewards first-option bias.
`gold_option_distribution` should be roughly uniform; if
`chosen_option_distribution` is not, the model is answering by position.

### Gold Option Margin (GOM)

Similarity to gold minus mean similarity to distractors. Cosines are kept in
`[-1, 1]`; a negative GOM is meaningful.

### Semantic similarity — diagnostic only

Lexical overlap with the reference. **Not a correctness measure**, and not
comparable across arms whose outputs differ in length. Use OMA for quality.

### Exact match

Near zero for any verbose model. Reported as `is_correct` for continuity; do
not build a claim on it for generative tasks.

---

## Statistics

At n=50 the 95% Wilson interval on an accuracy near 0.74 spans roughly ±12
percentage points, which is wider than any effect this project has reported.

- Use **n ≥ 500**. `sciq_test` has 1000 examples.
- Run **≥ 3 seeds** and report mean ± std. The results layout has no seed
  segment, so give each seed its own root or the second one overwrites the
  first:

  ```bash
  make grid SEED=42 RESULTS=results/seed42
  make grid SEED=43 RESULTS=results/seed43
  ```
- Every accuracy ships with a Wilson interval (`oma_ci`).
- Compare arms with the **exact McNemar test** — the arms answer identical
  items, so the comparison is paired and only discordant pairs carry
  information.

- Correct for **multiple comparisons**. Every comparison printed in one
  invocation is one family, Holm-adjusted; ten arms against a reference give
  roughly a 40% chance of at least one p<0.05 under the null. Splitting a sweep
  across invocations to shrink the family hides the count rather than reducing
  it.

```bash
python -m scripts.compare_arms --results-root results
```

The output states `b`, `c` and `n_discordant`, which make the effective sample
size visible, and `p_adjusted` alongside the raw p. A difference is a result
only if it survives this.

---

## Run health

These fields appear in every `metrics.json` and invalidate a run when wrong:

| Field | Expected | Meaning if not |
|---|---|---|
| `truncation_rate` | `0.0` | Prompts were clipped, and the clipped tail is the question |
| `chat_template_rate` | `1.0` | An instruct model was prompted as a raw completion |
| `relevance_active` | `true` | No embedding backend, so retrieval was not query-conditioned |
| `citation_rate` | high | Credit was mostly assigned uniformly; α/β are not per-lesson evidence |
| `mean_retention_score_std` | `> 0` | The retention score does not distinguish between lessons |

`scripts/aggregate_results.py` prints warnings for the first three.

---

## Worked example

```bash
# 1. Confirm the pipeline works before spending GPU hours
python -m scripts.smoke_test --model phi3-mini --device cuda

# 2. Run the grid, seeded
python -m scripts.run_eval_grid --config configs/experiment_grid.yaml --seed 42

# 3. Repeat for seeds 43 and 44

# 4. Aggregate, with intervals and health warnings
python -m scripts.aggregate_results --results-root results

# 5. Test every delta against its correct reference arm
python -m scripts.compare_arms --results-root results

# 6. Figures
python -m scripts.make_figures --results_dir results --output_dir figures
```
