# TinyACE — Codebase Review: Flaws and Improvements

> **Historical document.** This audit was performed against commit `306655b`
> and describes the codebase *before* the fixes. Every blocker it identifies
> has since been addressed — see `git log --oneline` for the commit series, and
> [evaluation.md](evaluation.md) for the protocol that replaced the practices
> criticised here.
>
> File and line references below point at the audited commit. They are not
> linked, because the line numbers no longer correspond to current code. Use
> `git show 306655b:<path>` to see what is being described.
>
> It is kept because the current design is only legible as a response to these
> defects: it explains why option order is permuted, why `cot_control` exists,
> why ablations resolve to `tinyace_wm_256`, and why the published results were
> withdrawn.

**Reviewed commit:** `306655b` (branch `main`)
**Date:** 2026-08-18
**Scope:** `src/edge_slm_ace/`, `scripts/`, `configs/`, `data/tasks/`, `tests/`, `.github/`, top-level docs (~10.8k LOC of Python)

This document is a technical review of the repository as it currently stands. It is organised by
severity, because the most important problems here are **not** style issues — several of them
invalidate the headline numbers reported in `README.md` and `docs/results.md`.

Every claim below was verified against the code or by running it. Where a fix is cheap, a concrete
patch is given.

---

## 0. Executive summary

| # | Area | Finding | Severity |
|---|------|---------|----------|
| S1 | Data | Gold answer is at option **(A) / index 0 in 100% of examples** | 🔴 Blocker |
| S2 | Metrics | OMA maps predictions to options by *embedding similarity of free text*; the parsed letter is computed but never used | 🔴 Blocker |
| S3 | Method | Baseline and ACE arms use **different prompts, different answer parsing, and different output-length regimes** | 🔴 Blocker |
| S4 | Method | ACE's reflector is fed the **gold answer of the test set**; the baseline gets nothing comparable | 🔴 Blocker |
| S5 | Stats | All headline deltas are 1–3 questions out of n=50, with no CI, no seed, single run | 🔴 Blocker |
| S6 | Method | Ablations are compared against **baseline**, not against full TinyACE — the wrong reference arm | 🔴 Blocker |
| B1 | Bug | `fifo_memory` implements **LIFO**, not FIFO — the repo's own test proves it and fails | 🔴 Blocker |
| B2 | Bug | `self_refine` puts the **gold answer in the generation prompt** | 🔴 Blocker |
| B3 | Bug | No chat template applied to `*-Instruct` / `*-Chat` models | 🟠 High |
| B4 | Bug | `truncation=True` with no `max_length` → silent prompt truncation | 🟠 High |
| B5 | Design | Retention scoring's α/β terms carry **no discriminative signal** by construction | 🟠 High |
| B6 | Design | Token budget makes retrieval ranking a **no-op** in working-memory mode | 🟠 High |
| B7 | Bug | Retrieval is domain-level, never query-conditioned → every question gets the same prefix | 🟠 High |
| B8 | Bug | LaTeX table emits a `SemSim` column that is **always 0.000** | 🟠 High |
| B9 | Bug | Duplicate detection discards *specific* lessons that contain a *vague* one | 🟡 Medium |
| R1 | Repro | **No seed anywhere**; `seed: 42` in configs is decorative | 🟠 High |
| R2 | Repro | `pytest` cannot import the package; CI masks every failure with `\|\| echo` | 🟠 High |
| R3 | Repro | `pyproject.toml` ships a **broken wheel** (subpackages excluded) | 🟠 High |
| R4 | Repro | No pinned versions, missing `accelerate`, 4 unused deps | 🟡 Medium |
| R5 | Repro | Configs reference 4 datasets that **do not exist** in the repo | 🟡 Medium |
| Q1–Q9 | Quality | Dead code, 4 overlapping plotting scripts, duplicated 150-line load path, junk files | 🟡 Medium |

**Bottom line:** the engineering scaffolding (grid runner, config system, result schema, docs) is
better than average for a research repo. The *measurement* layer underneath it is not sound, and
the current results should not be published without redoing the evaluation. The good news is that
S1, S2, S3 and B1 are all a few days of work to fix, and fixing them is what turns this into a
credible paper.

---

## 1. Blocking scientific-validity issues

### S1. The correct answer is always option (A) 🔴

`extract_mcq_options()` and the legacy branch of `extract_mcq_options_with_indices()` place the
gold answer at a fixed position:

- `mcq_eval.py:106` — `"A": correct,`
- `mcq_eval.py:149-150` — `options = [correct, d1, d2, d3]; return options, 0`

Verified against the data:

```
sciq_test.json      n=1000, legacy format → gold is index 0 for 1000/1000 examples
sciq_mcq_test.jsonl n=5,    gold_option_idx distribution: {0: 5, 1: 0, 2: 0, 3: 0}
```

The comment says this is "deterministic to ensure reproducible evaluation." Determinism and
position-balance are not in conflict — a seeded permutation is both. As written, **any model with a
first-option bias is rewarded, and any model without one is penalised**, and that bias differs
systematically between a terse baseline prompt and a verbose CoT prompt. This alone can produce a
±20pp swing on SciQ-style tasks and is sufficient to explain the entire Phi-3 "+4%" and much of the
TinyLlama "−26%".

**Fix:** permute options with a per-example seeded RNG derived from the example id, store the
resulting `gold_option_idx`, and additionally report **cyclic-permutation accuracy** (evaluate each
question under all 4 rotations and average) so position bias cannot contaminate the headline number.

```python
def extract_mcq_options_with_indices(example, seed: int = 0):
    ...
    rng = random.Random(f"{seed}:{example['id']}")
    order = list(range(4))
    rng.shuffle(order)
    options = [raw_options[i] for i in order]
    gold_idx = order.index(0)
    return options, gold_idx
```

---

### S2. OMA is measured by embedding similarity over free text, and ignores the letter it already parsed 🔴

`mcq_eval.py:444`:

```python
similarities = evaluator.compute_similarities(prediction, options)
chosen_option_idx = int(np.argmax(similarities))
```

`prediction` is the model's **entire output string**. Three consequences:

1. **The prompt asks for a letter, and a letter is unscorable.** `build_prompt_with_choices()`
   instructs: *"Answer with the exact choice text or the letter (A, B, C, or D)"*. If the model
   complies with the second half and answers `"B"`, the MiniLM embedding of `"B"` has no meaningful
   relationship to any option text — `argmax` over those four similarities is essentially a coin
   flip weighted by embedding artefacts. The metric silently rewards models that happen to echo
   option text and punishes models that follow the instruction.
2. **`detect_choice_marker()` already solves this and is never called for OMA.** It is used only
   for the ACR metric. The parsed letter is thrown away.
3. **Verbosity dominates.** A 200-token chain of thought is embedded as one vector; the option
   signal is diluted by the reasoning. Since the ACE arm mandates step-by-step reasoning and the
   baseline does not (see S3), the two arms are being scored by a metric with different noise
   characteristics.

**Fix — use a scoring cascade, applied identically to every arm:**

```python
def score_mcq(prediction, options, gold_idx, evaluator):
    letter = detect_choice_marker(prediction)          # 1. explicit letter wins
    if letter is not None:
        return "ABCD".index(letter)
    for i, opt in enumerate(options):                  # 2. exact option text
        if opt.strip().lower() in prediction.strip().lower():
            return i
    return int(np.argmax(evaluator.compute_similarities(prediction, options)))  # 3. last resort
```

Log which tier fired for each example, and report the tier-3 rate — if it is high, the metric is
not trustworthy.

**Better still:** for MCQ, use **length-normalised log-likelihood scoring of each option** under the
model (the standard `lm-evaluation-harness` approach). It removes parsing, formatting, verbosity,
and instruction-following from the measurement entirely, and it is the method reviewers will expect.
Keep generative OMA as a secondary "does it answer in a usable format" metric.

---

### S3. Baseline and ACE arms are not comparable 🔴

The two arms differ in at least four ways that have nothing to do with the playbook:

| | Baseline | ACE |
|---|---|---|
| Prompt | `Context: … Question: … Answer:` (`runner.py:161`) | Role preamble + strategies + **"You MUST show your step-by-step reasoning"** + `Reasoning:/Answer:` schema (`ace_roles.py:110-125`) |
| Choices block | `build_prompt_with_choices()` | different hand-rolled block (`runner.py:677-685`) |
| Domain instructions | none | injected (`ace_roles.py:10-36`) |
| Answer extraction | raw generation used as-is (`runner.py:184`) | `parse_generator_output()` (`runner.py:715`) |

So "ACE vs baseline" actually measures *playbook + CoT instruction + domain hints + a different
choices block + a different parser*. There is no way to attribute the observed delta to the
playbook, which is the paper's entire claim.

The parser asymmetry is the most damaging. `parse_generator_output()` falls through to
"return the whole cleaned text" when the model does not emit an `Answer:` header — which is exactly
what a 1.1B model does. Combined with S2, the ACE arm for TinyLlama is being scored on a
200-token reasoning dump while the baseline is scored on a short answer. **The reported "TinyLlama
−26%" is at least partly a parsing artefact, not a finding about small models.**

**Fix:** introduce a *prompt-matched* control arm. At minimum three arms:

1. `baseline` — terse prompt (current).
2. `cot_control` — the **exact ACE generator prompt with an empty playbook** (same CoT instruction,
   same domain hints, same choices block, same parser).
3. `ace_*` — the full system.

The scientific claim is `ace − cot_control`, not `ace − baseline`. Run every arm through one shared
`extract_answer()` + `score_mcq()` path. If `ace − cot_control ≈ 0` and `cot_control − baseline` is
large, the honest finding is *"CoT prompting helps; the playbook does not"* — which is still a
publishable negative result, and a much stronger paper than the current one.

---

### S4. The ACE arm consumes test-set gold answers; the baseline does not 🔴

`ace_roles.py:175` — the reflector prompt contains
`Correct Answer: {ground_truth}`. Its free-text output becomes playbook entries
(`runner.py:786-793`) that are prepended to the prompt
for every subsequent question in the same run.

This is defensible as *online continual learning* — item *i* only sees gold from items *< i* — but
it must be argued explicitly, and two things here make it worse:

- **The playbook is built on the evaluation set itself.** There is no train/dev split. The 50
  scored examples are the same 50 examples the system learns from.
- **Gold answer text can propagate.** Lessons are unconstrained free text produced by a model that
  was just shown the correct answer. Nothing prevents an entry like *"For questions about oxidising
  compounds, the answer is oxidants."* And **82% of `sciq_test` examples contain the gold answer
  verbatim in the `support` field**, so the domain is narrow enough for this to matter.

**Fix (pick one, and state it in the paper):**

- *Preferred:* build the playbook on a **held-out adaptation split** (e.g. `sciq_val`, which is
  already in the repo and has zero question overlap with `sciq_test` — verified), **freeze it**,
  then evaluate read-only on `sciq_test`. This is a clean, defensible protocol and lets you report
  a proper `playbook_frozen` arm.
- *If you keep the online protocol:* report **prefix-conditional accuracy** (accuracy on the last
  50% of the stream) alongside overall, add a shuffled-order control, and add an automatic filter
  rejecting any lesson whose text overlaps the gold answer of the example it came from
  (n-gram containment check).

---

### S5. Every reported difference is inside the noise floor 🔴

`docs/results.md` states n=50 per configuration. At n=50 and p≈0.74, the 95% CI on a single
accuracy is roughly **±12pp**. The paper's headline deltas:

- Phi-3 "+4% with FIFO" = **2 questions**
- "No Failure is critical, −4%" = **2 questions**
- "No Recency / No Vagueness, −2%" = **1 question**

There is no seed, no repeated run, no confidence interval, and no significance test anywhere in the
repo. The Component Importance Ranking in `docs/results.md` orders five configurations that span a
total of 8pp — i.e. 4 questions — and then draws five causal conclusions from that ordering.

**Fix:**

- Raise n to **≥500** (`sciq_test` already has 1000 examples; `--limit 50` is throwing away 95% of
  your data for no reason).
- Report **Wilson 95% CIs** on every accuracy.
- For paired arms (same items, same order) use **McNemar's exact test** and report the discordant
  pair counts `b`/`c` — this is the correct test and it is three lines of code.
- Run **≥3 seeds** and report mean ± std, even under greedy decoding (order effects and playbook
  dynamics are stochastic across shuffles).
- Delete any claim that does not survive.

```python
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar

lo, hi = proportion_confint(n_correct, n, method="wilson")
b = sum(1 for x, y in zip(arm_a, arm_b) if x and not y)
c = sum(1 for x, y in zip(arm_a, arm_b) if y and not x)
p = mcnemar([[0, b], [c, 0]], exact=True).pvalue
```

---

### S6. Ablations are compared against the wrong reference arm 🔴

`docs/results.md` §Ablation Study reports each `tinyace_ablate_no_X` against **baseline** (74%):

> `tinyace_ablate_no_failure` — OMA: 70% (−4% vs baseline) — **Conclusion: Failure tracking is critical**

But an ablation of a TinyACE component must be compared to **full TinyACE at the same budget**,
which the model-comparison table gives as `tinyace_wm_256` = 72%. So:

| Config | Reported vs baseline | Correct: vs `tinyace_wm_256` (72%) |
|---|---|---|
| `tinyace_fifo` 78% | +4% | **+6pp (3 questions)** |
| `no_recency` 72% | −2% | **0pp (0 questions)** |
| `no_failure` 70% | −4% | **−2pp (1 question)** |
| `no_vagueness` 72% | −2% | **0pp (0 questions)** |

Under the correct reference arm, **removing the recency term and removing the vagueness term change
nothing at all**, and removing the failure term changes one question. The stated conclusion
"Failure tracking is most critical" is not supported.

**Fix:** re-tabulate all ablations against `tinyace_wm_256`, add CIs and McNemar, and rewrite §Key
Findings. Expect most of it to collapse — which, combined with the FIFO bug (B1), is itself the
interesting result.

---

## 2. Correctness bugs

### B1. `fifo_memory` implements LIFO — the repo's own test fails 🔴

`playbook.py:202-203`:

```python
if params.fifo_memory:
    return -self.created_at
```

`created_at` is a Unix timestamp, so an **older** entry has a **smaller** `created_at` and therefore
a **larger** (less negative) score. `_evict_lowest_score_entries()` sorts ascending and drops the
front — i.e. it drops the **largest** `created_at`, the **newest** entry. Oldest entries are
retained forever. That is LIFO. And because `get_top_k()` sorts descending, the generator is always
shown the **oldest k lessons** and never sees anything learned recently — the "FIFO" arm is really a
*frozen first-N-lessons* arm.

This is not speculation; `tests/test_playbook.py::TestAblationFlags::test_fifo_memory` asserts the
documented behaviour and fails:

```
$ PYTHONPATH=src pytest tests/ -q
FAILED tests/test_playbook.py::TestAblationFlags::test_fifo_memory -
  AssertionError: In FIFO mode, older entries should have lower scores
  assert -1787029977.9048908 < -1787029987.904896
1 failed, 78 passed
```

The failure has been invisible because CI never runs the tests (see R2).

**This invalidates the paper's single strongest positive claim** — README: *"FIFO eviction achieves
the best OMA (78%), suggesting complex scoring may be unnecessary."* Whatever produced 78%, it was
not FIFO.

**Fix:** `return self.created_at` (older → lower score → evicted first), then re-run the arm. Also
decouple the two concerns — `fifo_memory` should change **eviction order only**, not retrieval
ranking; retrieval should stay score-based (or relevance-based, see B7).

```python
# eviction key vs retrieval key should be separate methods
def eviction_key(self, current_step, params):
    return self.created_at if params.fifo_memory else self.score(current_step, params)

def retrieval_key(self, current_step, params):
    return self.score(current_step, params)   # never FIFO
```

---

### B2. `self_refine` writes the gold answer into the generation prompt 🔴

`ace_roles.py:617`, used at
`runner.py:419-435`:

```
Your Initial Answer: {initial_answer}
Critique: {critique}
Correct Answer: {ground_truth}          ← in the prompt that produces the scored prediction

Your task: Provide a corrected answer …
```

The `final_answer` generated from this prompt is what gets scored
(`runner.py:460`). The model is being asked to restate an
answer it was just handed. This arm's accuracy measures **instruction-following, not reasoning**,
and its ceiling is 100% for any model that can copy.

Self-Refine (Madaan et al.) and SEAL do **not** show the model the gold answer at refinement time —
that is the whole point of the method. As implemented, this is not a baseline, it is a leak.

**Fix:** remove `ground_truth` from both `build_self_refine_critique_prompt` and
`build_self_refine_rewrite_prompt`. The critique must be genuinely self-generated ("review your
answer for errors"), and refinement must be conditioned on the critique alone. If you want an
oracle-feedback upper bound, keep it but **label it `self_refine_oracle` and report it as a
ceiling**, never as a comparable baseline.

---

### B3. No chat template is applied to instruct/chat models 🟠

`grep -rn "apply_chat_template" --include="*.py" .` → **zero hits.**

Every model in `MODEL_CONFIGS` is an instruction-tuned checkpoint —
`Phi-3-mini-4k-instruct`, `Mistral-7B-Instruct-v0.3`, `Qwen2.5-*-Instruct`,
`TinyLlama-1.1B-Chat-v1.0` — and every one is prompted as a raw completion at
`model_manager.py:288`.

These models are trained to respond only inside their own turn markers (`<|user|>…<|assistant|>`,
`[INST]…[/INST]`, ChatML). Prompting them raw produces continuation-style behaviour: rambling,
question echoing, self-dialogue. This is the most likely explanation for why Phi-3 scores **74%** on
SciQ here when SciQ with the correct template is comfortably 90%+ for a 3.8B instruct model — you
are benchmarking a misuse of the models, not the models.

It also **interacts with everything above**: template sensitivity differs per family, so the
TinyLlama-vs-Phi-3-vs-Mistral comparison is confounded by how badly each family degrades without its
template, and the ACE arm (long structured prompt) degrades differently from the terse baseline.

**Fix:**

```python
def build_inputs(tokenizer, prompt: str):
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt")
```

Record `used_chat_template` in run metadata so the ablation is auditable.

---

### B4. `truncation=True` with no `max_length` silently truncates prompts 🟠

`model_manager.py:288`:

```python
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
```

With no `max_length`, HF falls back to `tokenizer.model_max_length`, which is wildly inconsistent
across checkpoints (`1000000000000000019884624838656` for some, `512`/`2048` for others). Where it
is small, **the tail of the prompt is dropped — and the tail is the question**. Since ACE prompts
are much longer than baseline prompts (playbook + CoT scaffold), truncation would hit the ACE arm
preferentially and would look exactly like "playbook confuses small models."

`padding=True` on a single un-batched sequence is also a no-op that just adds risk.

**Fix:**

```python
max_len = min(getattr(model.config, "max_position_embeddings", 4096), 8192) - max_new_tokens
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
if inputs["input_ids"].shape[1] >= max_len:
    logger.warning("Prompt truncated to %d tokens — result is suspect", max_len)
```

Emit a `prompt_truncated` flag per example and **fail the run** if it exceeds a small threshold.

---

### B5. The α (success) and β (failure) terms carry no discriminative signal 🟠

At `runner.py:800-803`:

```python
for entry_id in used_entry_ids:
    playbook.mark_entry_used(entry_id, step)
    playbook.record_feedback(entry_id, helpful=correct)
```

Every retrieved entry receives **identical** credit or blame for the outcome. Combined with B6
(retrieval returns essentially *all* entries), this means at each step every live entry gets the
same increment. Their success ratios therefore converge to the model's running accuracy, differing
only by birth time. The α and β terms of

$$S = \alpha\frac{N_{succ}}{N_{used}+\epsilon} - \beta\frac{N_{fail}}{N_{used}+\epsilon} + \gamma e^{-\lambda(t-t_{last})} - \delta V$$

become near-constant across entries and **contribute almost nothing to the ranking**. The formula
reduces in practice to recency + vagueness. This is a structural explanation for why the ablations
in S6 show no effect — and it means the "retention scoring" contribution, listed as contribution #2
in the README, is largely inert.

**Fix — get real credit assignment:**

- *Cheap:* only credit entries the generator **actually referenced**. Ask the generator to emit
  `Used: [2, 5]`, parse it, and credit those. Imperfect but far better than uniform.
- *Sound:* **leave-one-out counterfactual** — for a sample of steps, re-run the generator with entry
  *i* removed and credit *i* by the change in outcome. Expensive but this is the measurement that
  justifies calling it a retention score.
- *Standard:* treat each entry as an arm and use a **bandit estimator** (Thompson sampling / UCB
  over per-entry Beta posteriors) with per-entry inclusion probabilities, so credit is attributable.

Whichever you pick, **validate that the score discriminates**: log the per-step variance of the
score across entries. If it is ~0, the scoring is decorative and the paper should say so.

---

### B6. In working-memory mode, retrieval selection is a no-op 🟠

- Eviction keeps *per-domain* tokens ≤ `token_budget`
  (`playbook.py:411-424`)
- Retrieval greedily fills up to the *same* `token_budget`
  (`playbook.py:330-355`)

Since total ≤ budget by construction, **every surviving entry is always retrieved**. Ranking is
irrelevant; only eviction order matters. So `tinyace_wm_256` vs `tinyace_wm_512` is not comparing
retrieval strategies — it is comparing *how many lessons survived*, and the reported difference
(9 vs 17 entries) is a capacity effect, not a selection effect.

The two budgets should be independent knobs: a **store capacity** `C` and a **prompt budget** `B`
with `B < C`, so that selection actually selects.

```python
Playbook(store_token_capacity=1024)          # what we keep
playbook.get_top_entries_for_budget(domain, prompt_token_budget=256)   # what we show
```

Then WM-256 vs WM-512 with a fixed store is a real experiment.

---

### B7. Retrieval is domain-level, never query-conditioned 🟠

`get_top_k(domain, …)` and `get_top_entries_for_budget(domain, …)` filter on `entry.domain == domain`
and rank by a score that **does not depend on the question at all**. Every task in the repo maps to
a single domain (`sciq_* → "science"`), so *every question in a run receives the same lesson list*.

This is not retrieval — it is a slowly-drifting fixed prompt prefix. It also explains the negative
results: a fixed generic prefix is pure distraction for a 1.1B model.

**Fix:** embed each lesson at insertion (MiniLM is already a dependency and already loaded), embed
the question, and rank by `λ · cos(q, lesson) + (1−λ) · retention_score`. This is a ~30-line change
that makes the system an actual memory-augmented method and is very likely to flip the sign of the
TinyLlama result.

---

### B8. The paper-ready LaTeX table emits a column of zeros 🟠

`run_qwen_rivals.py:513`:

```python
semsim = row.get("avg_semantic_similarity", 0) or 0
```

`avg_semantic_similarity` is computed **only** in `scripts/run_experiment.py`
(`run_experiment.py:530-545`). `run_qwen_rivals.py` calls
`run_dataset_baseline`/`run_dataset_ace` directly and never computes it, so the key is never in
`summary`. Every `SemSim` cell in `paper_snippets/qwen_rivals_table.tex` is `0.000` — silently,
because of the `or 0` fallback.

**Fix:** compute it in `run_single_experiment`, and make missing metrics **loud**:

```python
semsim = row.get("avg_semantic_similarity")
if semsim is None:
    raise KeyError(f"avg_semantic_similarity missing for {row['model_name']}/{row['mode_name']}")
```

Never let a silent default reach a table that goes into a paper. Audit the other `.get(k, 0)` calls
in the LaTeX/plot generators for the same pattern.

---

### B9. Deduplication systematically discards specific lessons 🟡

`playbook.py:361-373` and the near-duplicate logic in
`ace_roles.py:452-460` both test **substring
containment in either direction**:

```python
if text_lower in entry_text_lower or entry_text_lower in text_lower:
    return entry   # treated as duplicate, new text discarded
```

So if the playbook already contains the short lesson *"check the units"*, then a new, genuinely
useful lesson *"For density problems, check the units: convert g/cm³ to kg/m³ by ×1000"* is
**rejected as a duplicate**, and the vague entry is what survives. This directly fights the vagueness
penalty the paper presents as a contribution.

**Fix:** use embedding cosine similarity with a threshold (~0.9) for dedup, and on a near-duplicate
collision **keep the entry with the lower vagueness score** rather than always keeping the incumbent.

---

### B10. Additional smaller bugs 🟡

| Location | Issue |
|---|---|
| `playbook.py:294-305` | `_next_id` reconstruction ignores non-numeric ids → reloading a playbook with any non-int id resets `_next_id` to 1 → **duplicate ids** → `record_feedback` silently updates the wrong entry (it `return`s on first match). |
| `playbook.py:246-263` | `from_dict` **mutates the caller's dict** in place while filling defaults. |
| `playbook.py:157-166` | `__post_init__` treats `vagueness_score == 0.0` and `token_count == 0` as "unset" → a legitimately non-vague entry is silently rescored on every load. Use `None` sentinels. |
| `playbook.py:449-456` | When eviction cannot free enough tokens, the entry is added anyway with a `pass` where the warning should be — **the budget is silently violated**, which is the exact invariant the paper claims to enforce. |
| `playbook.py:428` | `e.id not in entries_to_remove` over a list → O(n²). Use a set. |
| `playbook.py:126-129` | `_estimate_tokens` uses `words × 1.3` while everything else uses the real tokenizer (`count_tokens`). The "256-token budget" is therefore not 256 tokens. Pass the tokenizer in. |
| `ace_roles.py:249-320` | `parse_generator_output` comments jump from "Strategy 1" to "Strategy 3" — Strategy 2 was deleted. The Strategy-3 block has dead control flow (`for prefix … break` then `if not answer` then unconditional `break`). This function needs a rewrite and a table-driven test suite. |
| `ace_roles.py:88` | `from …config import ACE_MODE_WORKING` imported inside the function and never used. |
| `ace_roles.py:161` | The reflector computes `correct` by exact string match — inconsistent with OMA, so on MCQ tasks it reports `Status: incorrect` for answers OMA counts as correct, and reflects on the wrong examples. |
| `runner.py:184`, `:460`, `:722` | `correct = answer.strip().lower() == ground_truth.strip().lower()` — exact match on a generative model's output is ≈0 for every verbose model, yet `is_correct` is what `plot_results.py` and `aggregate_results.py` chart by default for non-SciQ tasks. **`medqa_tiny` and `iot_tiny` have no working metric at all.** |
| `runner.py:816-820` | `evictions_during_add` is inferred by differencing entry counts — comment admits "This is approximate". Have `add_entry` return the eviction count. |
| `metrics.py:196-216` | `semantic_answer_score` averages token-F1 with `SequenceMatcher.ratio()`, both **length-penalised**. A correct verbose answer scores lower than a wrong terse one. This is the metric behind the README's "semantic similarity" column, so that column largely measures brevity. Phi-3 baseline 0.385 vs Mistral 0.823 is consistent with a verbosity difference, not a quality difference. |
| `metrics.py:236` | `sentence_bleu` on 1–3 word gold answers — BLEU-4 is 0 or undefined for references shorter than 4 tokens. `bleu_score` is noise for this task; drop it or use chrF. |
| `metrics.py:602`, `mcq_eval.py:272` | `np.clip(sim, 0, 1)` destroys the sign of genuinely dissimilar pairs and compresses GOM. |
| `metrics.py:461-500` | `PeakMemoryTracker` samples RSS at **enter, exit, and manual `update()` only** — it cannot observe a peak that occurs between samples. `peak_memory_mb` for CPU is not a peak. Use `resource.getrusage(RUSAGE_SELF).ru_maxrss` or a sampling thread. |
| `run_experiment.py:373-378` | The entire eval is wrapped in a `try` whose handler prints `"Failed to load model"` — a mid-run OOM, a dataset error, or a metric crash is all reported as a model-loading failure. |
| `run_experiment.py:520-522` | `avg_latency_sec or summary.get("avg_latency_sec")` — `0.0` is falsy, so a genuine zero silently falls through. |
| `model_manager.py:304` | `temperature` is still passed when `do_sample=False`; HF warns, and it obscures which decoding actually ran. Branch explicitly. |
| `model_manager.py:5-10` | Module-import-time monkey-patch of `DynamicCache.seen_tokens`. Pin `transformers` instead. |
| `model_manager.py:60-77` | On any tokenizer/model load failure the code **automatically retries with `trust_remote_code=True`** — silently executing arbitrary code from the Hub. Make this an explicit opt-in flag. |

---

## 3. Reproducibility and infrastructure

### R1. There is no seed 🟠

```
$ grep -rn "seed" --include="*.py" .
scripts/run_qwen_rivals.py:78:  "seed": 42,
scripts/run_qwen_rivals.py:84:  "seed": 42,
```

Those two lines write `42` into a metadata dict. **No `torch.manual_seed`, no `random.seed`, no
`np.random.seed`, no `transformers.set_seed` anywhere in the codebase.** Meanwhile
`configs/qwen_rivals_contract.yaml` advertises `seed: 42` under a heading that promises "IDENTICAL
across all models."

`MODEL_CONFIGS` also uses **`temperature=0.7` for `mistral-7b` and `llama-3.2-1b`** but `0.0` for
Phi-3, Qwen and TinyLlama (`config.py:29-100`) — so the
cross-model comparison in the README mixes sampled and greedy decoding, unseeded.

**Fix:** a single `set_seed(seed)` called at the top of every entrypoint, `seed` as a required CLI
arg recorded in `run_metadata.json`, and identical decoding parameters across all models in any
comparison table. Also record `torch.__version__`, `transformers.__version__`, GPU model, and
`git rev-parse HEAD` in the metadata — none are captured today.

### R2. Tests do not run, and CI cannot fail 🟠

```
$ pytest tests/ -q
ModuleNotFoundError: No module named 'edge_slm_ace'
3 errors during collection

$ PYTHONPATH=src pytest tests/ -q
1 failed, 78 passed          ← the FIFO bug (B1)
```

There is no `conftest.py` and no `pythonpath` setting in `[tool.pytest.ini_options]`, so a bare
`pytest` fails to import the package. `.github/workflows/ci.yaml` installs `requirements.txt` but
**never runs `pip install -e .`**, so CI hits the same import error — and then:

```yaml
pytest tests/ -v --cov=slm_ace … || echo "Tests not yet implemented"
black --check . || echo "Black formatting check skipped"
flake8 … || echo "Flake8 check skipped"
```

Every step is `|| echo`, so **the workflow is green no matter what**. The coverage target
`--cov=slm_ace` is also the wrong package name (it is `edge_slm_ace`), so coverage is always 0.

This is why a failing test asserting a headline experimental behaviour sat in `main` unnoticed.

**Fix:**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

```yaml
- run: pip install -e ".[dev,metrics]"
- run: pytest tests/ -v --cov=edge_slm_ace --cov-report=term-missing   # no `|| echo`
- run: ruff check . && black --check .
```

### R3. `pyproject.toml` builds a broken wheel 🟠

```toml
[tool.setuptools]
packages = ["edge_slm_ace"]
```

This lists **only the top-level package** — `edge_slm_ace.core`, `.memory`, `.models`, `.utils` are
excluded from the distribution. `pip install .` (as opposed to `-e .`) produces a package where
`from edge_slm_ace.core.runner import …` fails. The README's install instructions use `-e`, which
masks it.

`setup.py` and `pyproject.toml` also **both** exist with divergent metadata (different author lists,
`setup.py` declares a `metrics` extra that `pyproject.toml` lacks, both use `@example.com`
addresses).

**Fix:** delete `setup.py`, and use auto-discovery:

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

### R4. Dependencies are unpinned, one is missing, four are unused 🟡

- **Missing:** `accelerate` — required by `device_map="auto"`
  (`model_manager.py:89`) but only present as a
  *commented-out* line in `requirements.txt`. Every CUDA run depends on it being installed by luck.
- **Unused:** `datasets`, `jsonlines`, `rouge-score`, `tqdm` (verified — zero imports).
- **Unpinned:** everything is `>=`. The `DynamicCache` monkey-patch is direct evidence of a
  transformers API break already having bitten this project. A paper's results are not reproducible
  against `transformers>=4.35.0` — that range spans years of behaviour changes to generation.

**Fix:** pin exact versions in `requirements.lock.txt` for the paper runs, keep loose ranges in
`pyproject.toml` for library use, add `accelerate`, drop the unused four.

### R5. Configs reference datasets that do not exist 🟡

`TASK_CONFIGS` (`config.py:132-180`) registers
`tatqa_tiny`, `medqa_train`, `math_train`, `sciq_train` — **none of these files are in
`data/tasks/`**. `configs/experiment_grid.yaml` lists `medqa_train` and `math_train` as active
tasks, so the documented `run_eval_grid` invocation fails on two of its three tasks. The README's
own example (`--task-name tatqa_tiny`) is broken, as is the docstring example in
`run_experiment.py`.

Also note `TASK_CONFIGS` paths are **relative to CWD**; `run_experiment.py` re-resolves them against
the repo root, but any other consumer of `get_task_config()` breaks outside the repo root.

**Fix:** ship the datasets or a download script, validate the registry at import
(`assert path.exists()` behind a `--strict` flag), resolve paths against a package-level
`DATA_ROOT`, and add a CI check that every registered task resolves.

### R6. The paper's results cannot be regenerated from this repo 🟡

`.gitignore` excludes `results/`, `results_models/`, `results_ablation/`, `playbook.jsonl`,
`playbook_log.csv`. So none of the raw predictions, playbooks, metrics files, or figures behind
`docs/results.md` are in the repository. There is also no `run_all.sh` reproducing the exact
commands that produced them.

**Fix:** commit the final `metrics.json`, `predictions.jsonl`, `playbook.jsonl` and
`run_metadata.json` for every reported cell (they are small), plus a single `make paper` /
`scripts/reproduce_paper.sh`. This is the difference between "here is our code" and "here are our
results."

---

## 4. Code quality and maintainability

### Q1. Four overlapping result/plot scripts, 2400 lines

`plot_results.py` (1145), `tinyace_plots.py` (645), `aggregate_results.py` (331),
`summarize_results.py` (295) all read result files and emit tables/figures with overlapping
responsibilities and independently-maintained mode-label maps. `run_qwen_rivals.py` then reimplements
plotting a *fifth* time.

**Fix:** one `edge_slm_ace/reporting/` module — `load_results()`, `summarize()`, `figures()` — and
thin CLI wrappers. Delete the rest.

### Q2. `model_manager.load_model_and_tokenizer` is ~200 lines of duplicated fallback

The CUDA branch and the CPU/MPS branch contain the *same* four-level nested
safetensors→trust_remote_code→legacy-weights cascade, copy-pasted, including two copies of an
identical 12-line `tiny-gpt2` error message.

**Fix:** one `_load_with_fallbacks(model_id, **kwargs)` helper; device handling becomes a `dtype` +
`device_map` argument. Should be ~40 lines total. Delete the `tiny-gpt2` special-casing entirely and
use a modern small model (`HuggingFaceTB/SmolLM2-135M`) for smoke tests.

### Q3. `runner.py` is three near-identical 300-line functions

`run_dataset_baseline`, `run_dataset_self_refine`, `run_dataset_ace` each re-implement: example
field extraction, MCQ option extraction, latency timing, token counting, the ~30-key result dict,
the dual new/legacy MCQ metric branches, and the aggregate summary — with subtle divergences (the
baseline records `latency_sec`, self-refine does not; only ACE parses the answer).

**Fix:** extract `_evaluate_example(example, generate_fn, …) -> dict` and a `_summarize(results)`.
Each mode then supplies only its `generate_fn` and its post-step hook. This would have made S3
(prompt/parser asymmetry) structurally impossible.

### Q4. Legacy/new dual formats everywhere

`PlaybookEntry` carries four legacy fields (`helpful_count`, `harmful_count`, `last_seen_step`,
`is_generic`) shadowing their replacements and updated in lockstep by hand. Every result dict carries
canonical *and* legacy column names (`qid`/`sample_id`, `task`/`task_name`, `model`/`model_id`,
`is_correct`/`correct`). Every MCQ code path is written twice. `mcq_eval` supports two option formats.

For an unreleased research repo at v0.1.0 with no external users, this backward compatibility is pure
cost — it doubles the surface area where the two branches can diverge (and they already have: the
new-format path never sets `acr_hit`, so ACR is silently unavailable for `sciq_test`).

**Fix:** pick one schema, write a one-off migration script for existing artefacts, delete the rest.

### Q5. Curator role is built but never invoked

`build_curator_prompt()` and `parse_curator_output()`
(`ace_roles.py:377-448`) are fully implemented and
**never called from `runner.py`**. The README and the architecture diagram both present
Generate→Reflect→**Curate**→Memorize as the core loop; what actually runs is
`choose_lessons_for_playbook()`, a hardcoded keyword filter. The paper describes a component that
does not execute.

**Fix:** either wire the Curator into `run_dataset_ace` (and measure whether it helps) or delete it
and correct the architecture description. Do not ship a diagram of code that does not run.

### Q6. YAML scoring hyperparameters are never read

`configs/experiment_grid.yaml` has a `scoring:` block (α, β, γ, δ, λ, ε) with an explanatory header.
`grep` confirms **no script ever reads it** — `ScoringParams` is always constructed from
`playbook.py` defaults, and `run_eval_grid.py` only forwards the four boolean ablation flags. Anyone
tuning α/β in the YAML would silently change nothing.

`run_qwen_rivals.py:230` is worse: it constructs `Playbook(token_budget=…)` with **no
`scoring_params` at all**, so ablation flags are unreachable in that entire script.

**Fix:** thread `scoring` through `run_eval_grid` → `run_experiment` CLI → `ScoringParams`, or delete
the block. Add a test that a non-default α in YAML changes eviction behaviour.

### Q7. Ownership TODOs in the core loop

`runner.py:546-558` contains `TODO(Sathwik)` and
`TODO(Archit)` blocks describing unimplemented work ("Improve ACE logic", "Track playbook evolution
over time") in the docstring of the paper's central function. Move to issues before publication.

### Q8. Junk artefacts committed at the repo root 🟡

`results_models` and `results_ablation` are **files, not directories**, containing saved HTML from a
browser directory listing:

```html
<script>start("/Users/jarvis/TINY ACE/results_models/");</script>
<script>addRow("mistral_7b_instruct","mistral_7b_instruct",1,128,"128 B",…)</script>
```

They leak a local absolute path and shadow the `results_models/`, `results_ablation/` entries in
`.gitignore`. Also present: 6 `*.md.old` files and a `run_grid.py.old` in `docs/`, three internal
docs (`docs/dev/TEAM_MESSAGES.md`, `TEAM_NOTES.md`, `DEV_NOTES_PERSON1.md`, 45KB of standup notes),
and `FINAL_SUMMARY.md` / `REPOSITORY_STRUCTURE.md` / `docs/FINAL_STRUCTURE.md` /
`docs/RESTRUCTURING_SUMMARY.md` / `docs/STRUCTURE.md` / `docs/NAVIGATION.md` — **six documents
describing the directory layout**, which git already describes.

**Fix:** `git rm` the two HTML files and all `.old` files; move dev notes to a wiki or a
`dev/` branch; collapse the six structure docs into one short section of `README.md`.

### Q9. Broken references and placeholder metadata in public-facing docs 🟡

| File | Issue |
|---|---|
| `README.md` | References `TinyAce-3.pdf` twice; the actual file is `paper/tinyace.pdf` (with a space). |
| `README.md` citation | `author={Shahi, Suryodaya and Collaborators}`, `year={2024}` — placeholder authorship on a paper submission. |
| `pyproject.toml` / `setup.py` | `suryodaya@example.com`, `sathwik@example.com`, `archit@example.com`, `team@example.com`. |
| `README.md` vs `docs/results.md` | Contradict each other: README says Phi-3's best ACE arm is **FIFO (78%)**, `RESULTS.md` model table says **`tinyace_wm_512` (78%)** while listing `ace_full` at 60%; README says ACE overhead is **~1.4s**, `RESULTS.md` says **~2.4–2.9s**. |
| `docs/results.md` | Reports Phi-3 (3.8B) baseline latency **6.43s** but Mistral-7B (7B) baseline latency **0.408s** — a 7B model 15× faster than a 3.8B model on the same task. This is internally impossible and indicates the two rows were produced under different hardware/decoding/output-length conditions. **The cross-model comparison table should not be published until this is explained.** |
| `PUBLICATION_CHECKLIST.md` | Declares `Status: ✅ Ready for Publication` and checks off "Verify all code examples work" adjacent to a broken README example, a failing test, and a CI job that cannot fail. |
| `CHANGELOG.md`, `docs/results.md` | Dated "December 2024"; `results_*` HTML artefacts are timestamped 12/14/25. |
| `config.py:64-70` | `medqa_finetuned_small` maps to **`microsoft/DialoGPT-small`** — a chit-chat model — while the comment calls it *"our fine-tuned small upper bound"*. If this ever reaches a results table it is a misrepresentation. Remove it or actually fine-tune something. |

---

## 5. What I'd recommend doing, in order

### Phase 1 — Stop the bleeding (½ day)

1. Fix the FIFO sign (**B1**) and make the failing test pass.
2. Add `pythonpath = ["src"]`, remove every `|| echo` from CI, fix `--cov=edge_slm_ace`, add
   `pip install -e ".[dev,metrics]"`. **CI must be able to go red.**
3. Fix `pyproject.toml` package discovery (**R3**); delete `setup.py`.
4. `git rm` the two HTML junk files, the `.old` files (**Q8**).
5. Add `set_seed()` to every entrypoint; record versions + git SHA in `run_metadata.json` (**R1**).
6. Fix the `TinyAce-3.pdf` reference and the placeholder emails/authorship (**Q9**).

### Phase 2 — Make the measurement sound (~1 week)

7. Randomise option order with a per-example seed (**S1**) + add cyclic-permutation accuracy.
8. Rewrite MCQ scoring as the letter→text→embedding cascade (**S2**); add log-likelihood option
   scoring as the primary metric.
9. Apply chat templates (**B3**) and bound truncation with an explicit `max_length` + a
   `prompt_truncated` flag that fails the run (**B4**).
10. Unify the arms: one prompt builder, one answer extractor, one scorer; add the **`cot_control`
    arm** (**S3**).
11. Remove `ground_truth` from the self-refine prompts (**B2**); rename the oracle variant.
12. Split the playbook into a held-out adaptation split and a frozen eval (**S4**).

### Phase 3 — Re-run and re-analyse (~1 week, mostly GPU time)

13. Re-run everything at **n≥500**, ≥3 seeds, identical decoding across models.
14. Report Wilson CIs + McNemar for every comparison; regenerate all tables (**S5**).
15. Re-tabulate ablations against `tinyace_wm_256`, not baseline (**S6**).
16. **Rewrite `docs/results.md` and the README results section from the new numbers.** Expect the
    ablation conclusions to disappear; that is a finding, not a failure. Explain the Phi-3/Mistral
    latency inversion or drop the latency comparison.

### Phase 4 — Make the method worth the paper (~1–2 weeks)

17. Query-conditioned retrieval over lesson embeddings (**B7**) — the single highest-upside change.
18. Real credit assignment: cited-entry or leave-one-out (**B5**), with a logged check that the
    retention score actually discriminates between entries.
19. Decouple store capacity from prompt budget so selection selects (**B6**).
20. Wire in the Curator or delete it and fix the architecture diagram (**Q5**).
21. Embedding-based dedup that prefers the more specific lesson (**B9**).
22. Refactor `runner.py` into one shared evaluation core (**Q3**) and collapse the plotting scripts
    (**Q1**).

---

## 6. What is genuinely good here

To keep this balanced — these are above the bar for a research codebase and worth preserving through
the refactor:

- **The experiment-contract idea** (`configs/qwen_rivals_contract.yaml`) is the right instinct: one
  declarative file pinning dataset slice, decoding, and reflection policy across model families. It
  just needs to actually be enforced by the code (seed, scoring params).
- **The grid runner** — per-experiment output directories, `--dry-run`, per-run stdout/stderr logs,
  device-availability fallback — is clean and genuinely useful.
- **Result schema discipline.** Emitting per-example rows with a documented schema, plus
  `metrics.json` + `predictions.jsonl` + `run_metadata.json` per run, is exactly right and made this
  review possible.
- **`docs/results.md` reports negative results honestly** ("Baseline preferred", "ACE degrades
  performance") rather than burying them. The analysis needs fixing, but the disposition is right.
- **The test suite is real** — 79 tests with meaningful assertions, including the one that caught
  the FIFO bug. It only needs to be *run*.
- **The ablation flags are properly plumbed** from CLI through `ScoringParams` to the scoring
  function — the mechanism is sound even though the YAML hyperparameters are not connected.
- **Docstrings are unusually thorough**, including documented return schemas.

The core problem is not care or effort — both are evident. It is that the evaluation layer was never
independently validated, and CI was configured in a way that guaranteed nobody would find out.

---

## Appendix: reproducing the checks in this review

```bash
# B1 — the FIFO bug (fails today)
PYTHONPATH=src pytest tests/test_playbook.py::TestAblationFlags::test_fifo_memory -v

# R2 — tests cannot be collected the way CI runs them
pytest tests/ -q

# S1 — gold answer is always option A
python3 -c "
import json
t = json.load(open('data/tasks/sciq_test.json'))
print('legacy-format examples:', len(t))         # 1000 → all gold at index 0
d = [json.loads(l) for l in open('data/tasks/sciq_mcq_test.jsonl')]
print({i: sum(1 for x in d if x['gold_option_idx'] == i) for i in range(4)})
print('answer verbatim in support:',
      sum(1 for x in t if x.get('support') and x['correct_answer'].lower() in x['support'].lower()))
"

# R1 / B3 — no seed, no chat template
grep -rn "manual_seed\|set_seed\|apply_chat_template" --include="*.py" . ; echo "exit=$?"

# R5 — registered tasks that do not exist
python3 -c "
import sys; sys.path.insert(0,'src')
from pathlib import Path
from edge_slm_ace.utils.config import TASK_CONFIGS
for k, v in TASK_CONFIGS.items():
    if not Path(v['path']).exists(): print('MISSING:', k, v['path'])
"

# R4 — unused dependencies
for m in datasets jsonlines rouge_score tqdm accelerate; do
  printf '%-12s' "$m"; grep -rl "import $m\|from $m" --include='*.py' . || echo '(unused)'
done
```
