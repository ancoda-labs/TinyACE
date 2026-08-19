# The Evaluation Protocol

**Read this before reporting any number.**

Everything here exists because some version of it went wrong before. The
`CHANGELOG.md` has the history; this is the standing rule.

---

## The split

Belebele is 900 parallel questions per language. `data.study_split` divides them
once, deterministically in `(seed, ids)` and independent of the order the ids
arrive in:

| Split | Size | Used for |
|---|---|---|
| Adaptation | 400 | Screening (first 400), and building the playbook |
| Evaluation | 500 | Every reported number |

Three properties this buys:

- **Every language and every arm gets the identical split.** The split is
  computed from content-derived ids, not row order. The two language files hold
  the same questions in *different* orders, so a split derived from position
  would adapt on different questions per language and make the paired comparison
  meaningless.
- **Screening cannot leak.** `ADAPTATION_SIZE` is one constant that both
  `screen_models` and `run_arm` go through. When they each chose their own size
  — 400 and 200 — the 200 items between them were screened on *and* scored on,
  so model selection was made on 200 of the reported items. `screen_models` now
  refuses to run if any screening item is in the evaluation split.
- **The Reflector never sees a scored item.** It is shown gold answers, which is
  precisely why the split it runs on must never be scored.

---

## The arms, and what each one isolates

**The playbook claim is `tinyace − scaffold_control`.**

`scaffold_control` is the identical instruction prefix over an empty playbook.
Without it, any `tinyace − baseline` difference is "an instruction prefix helps"
plus "the playbook helps", and the two are not separable. Under loglikelihood
option scoring the model emits no text, so a chain-of-thought control is not
expressible in this track — the arm is named for what it actually is.

**Ablations are compared against `tinyace`.** An ablation against `baseline`
measures ACE *plus* the ablation, not the ablated component.
`reporting.reference_for()` owns that mapping and `compare_arms` applies it; the
script previously imported it without ever calling it, and compared everything
to baseline.

| Arm | Isolates |
|---|---|
| `baseline` | Nothing. The floor. |
| `scaffold_control` | The instruction prefix, with no lesson content |
| `tinyace` | The playbook's contribution over the prefix |
| `tinyace_equal_lessons` | Whether prefix *length* rather than lesson quality is binding — Devanagari fertility means the same lesson count costs more tokens in Nepali |
| `tinyace_playbook_en` | Whether lessons must be in the question's language |
| `tinyace_ablate_no_curator` | The Curator screening pass |
| `tinyace_ablate_no_relevance` | Query-conditioned ranking during adaptation |
| `tinyace_ablate_no_vagueness` | δ |
| `tinyace_ablate_no_recency` | γ |
| `tinyace_ablate_no_failure` | β |
| `tinyace_fifo` | Eviction order, and only eviction — retrieval still ranks by retention |

An arm that changes *adaptation* builds its own playbook; an arm that changes
only how a frozen playbook is *presented* borrows one. `run_grid` orders the
plan so a borrowing arm always follows its source, in both the arm and the
language dimension.

---

## What counts as a result

### 1. An interval, always

At n=500 the 95% Wilson halfwidth near 40% is about 4.3 points. At n=50 near 74%
it is about 12. A "+4%" improvement at n=50 is two questions.

`summarize_accuracy` returns `ci_halfwidth` alongside every accuracy for exactly
this comparison. Wilson rather than the normal approximation: the latter
produces bounds above 1.0 near the ceiling and collapses to zero width at p=0
or 1.

### 2. A paired test

The arms answer identical questions, so the comparison is paired and only the
discordant pairs carry information. `mcnemar_exact` reports `b`, `c` and
`n_discordant` — **read `n_discordant`, not `n`**. It is the effective sample
size of the test, and it is usually a small fraction of the split.

An unpaired proportion test throws the pairing away and is strictly less
powerful. It is not an option here.

### 3. A multiple-comparison correction

Ten arms against a reference is ten tests. At α=0.05 the chance of at least one
false positive under the null is `1 − 0.95¹⁰ ≈ 40%`, so an uncorrected
"significant" result from a sweep is close to expected rather than surprising.

Every comparison printed together is one Holm-corrected family. Holm is
uniformly more powerful than Bonferroni and assumes nothing about independence,
which matters because the arms are scored on the same items.

**The family is every comparison you report together.** Splitting one sweep
across several invocations does not shrink the correction; it hides the count.

Comparisons with no items in common carry a forced p=1.0 and no evidence. They
sit outside the correction rather than inflating the family size and costing the
real comparisons power.

### The default reading

If nothing survives Holm correction, the result is **"no detectable
difference"** — not an ordering of the point estimates. `compare_arms` prints
that sentence itself.

---

## Health checks that invalidate a run

### Truncation

lm-eval truncates an over-long prompt **from the left**, and the Belebele prompt
opens with `P: <passage>`. Left-truncation therefore eats the passage — the
thing the question is about — and the run degrades into guessing without
failing.

It is worse than a uniform loss. The instruction prefix and the playbook make
the prompt longer, so the ACE arms truncate **more** than baseline on the same
items. That is an arm-asymmetric confound in the direction of the hypothesis,
not noise. Devanagari fertility pushes the Nepali side further again.

`truncated_prompts`, `truncation_rate` and `tokens_dropped` travel with every
result. **A run with a non-zero truncation rate is not comparable across arms
and must not be reported as one.**

### Relevance backend

`relevance_active: false` means no embedding backend loaded and retrieval was
not query-conditioned, whatever `--relevance-weight` was set to.
`aggregate_results` warns; `metrics.json` also records which encoder was used.

### Item-set mismatch

`harness_indices` maps positions in the committed jsonl; the harness loads the
split from the Hub. `per_item_correctness` recovers the item id from each logged
doc and raises if the set scored is not the set requested — if those orders ever
diverge, the run compares different questions per arm and still reports a clean
accuracy.

---

## Reproducibility

- `set_seed()` before any model loads, and every RNG the harness seeds is passed
  the same value.
- `capture_environment()` records python, platform, torch, transformers, CUDA,
  GPU, git commit and whether the tree was dirty. It is written into every
  `metrics.json`.
- `run_grid` treats a cell as complete only if its `metrics.json` records both
  this seed and this commit, so an interrupted grid resumes and a code change
  invalidates the cells it could have affected.

The results layout carries no seed segment. A multi-seed study needs one root
per seed (`make grid SEED=43 RESULTS=results/seed43`); report mean ± sd across
seeds.

---

## Reporting checklist

- [ ] Every accuracy has an interval next to it.
- [ ] Every delta went through `scripts/compare_arms.py`.
- [ ] The reference arm is the registered one — ACE against `scaffold_control`,
      ablations against `tinyace`.
- [ ] The reported p-value is the **Holm-adjusted** one, and the family size is
      stated.
- [ ] `n_discordant` is reported alongside `n`.
- [ ] Truncation is zero on every arm in the table, or the table says so.
- [ ] `relevance_active` is true, or the limitation is stated.
- [ ] The seed and commit are stated, and multi-seed runs report spread.
- [ ] Nothing is ranked that did not survive correction.
