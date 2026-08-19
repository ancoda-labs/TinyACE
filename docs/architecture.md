# Architecture

Two phases, one boundary between them. Everything about the design follows from
where that boundary sits.

```
      ┌─────────────────────────────────────┐
      │  ADAPTATION   (adaptation split)    │      code this project owns
      │  score → reflect → curate → store   │
      └──────────────────┬──────────────────┘
                         │  freeze: playbook.jsonl → top-k → one static string
      ┌──────────────────▼──────────────────┐
      │  EVALUATION   (evaluation split)    │      stock lm-evaluation-harness
      │  simple_evaluate(system_instruction)│
      └─────────────────────────────────────┘
```

**Nothing downstream of the freeze is ours.** No answer parsing, no option
mapping, no bespoke accuracy — evaluation is a stock harness run over a file.
The playbook is a file by the time evaluation starts, so nothing computed during
evaluation can feed back into it. The previous version of this project had its
own scoring loop, and had to withdraw its results.

---

## Adaptation

`src/edge_slm_ace/adapt.py`. One pass over the adaptation split; per step:

1. **Retrieve** the top-k lessons for the domain, ranked by retention score
   blended with cosine relevance to the question.
2. **Score** the four options by loglikelihood — *identically to how evaluation
   will score them*.
3. **Reflect**, on an error. The Reflector is shown the passage, the chosen
   option and the gold option, and asked for reusable reading strategies that
   name the trap that was fallen into.
4. **Curate**. One extra generation screens the candidates for genericness.
5. **Store**, with deduplication; prune periodically.
6. **Record feedback** on every lesson that was retrieved.

### The correctness definition has to match

Adaptation decides "was this right?" by the loglikelihood of `A`–`D` through the
same `HFLM` the evaluation uses. It used to generate free text and parse it,
which meant the playbook was tuned against one notion of correctness and
reported against another — a mode mismatch that looks exactly like an effect of
the playbook.

Matching the *decision rule* is not enough; the *prompt* has to match too.
`harness/prompts.py` assembles the context by calling lm-eval's own `Message`,
`maybe_delimit` and `multiturn_to_singleturn` rather than describing what they
do, because three separate divergences had crept in:

- evaluation defaults to `apply_chat_template=True`, so the instruction is a
  `system` role inside the model's template — adaptation was scoring a raw
  f-string, in completion mode;
- with the template off, lm-eval concatenates the system message with **no**
  separator, and adaptation inserted `\n\n`;
- under a chat template `construct_requests` zeroes the target delimiter, so
  evaluation scores `"A"`–`"D"` while adaptation scored `" A"`–`" D"`.

`assert_matches_harness()` checks the template constant, both delimiters and the
absence of a `gen_prefix`; a golden test builds a real `ConfigurableTask` over
the committed Belebele file and asserts byte equality of the assembled context
in both modes. `OptionScorer` carries `apply_chat_template` so an arm cannot
adapt in one mode and be scored in the other.

### Credit assignment is weak, deliberately

The model emits no text at scoring time, so there is no citation to attribute a
correct answer by. Credit is uniform over whatever was retrieved. The
success/failure counts should be read as weak per-lesson evidence — the previous
pipeline's citation machinery only worked because the model was generating, and
it corrupted the scored prediction in the arm that cited.

---

## The playbook

`src/edge_slm_ace/memory/playbook.py`.

### Retention score

```
S(l, t) = α·(N_succ/(N_used+ε))          success
        − β·(N_fail/(N_used+ε))          failure
        + γ·exp(−λ·(t − t_last))         recency
        − δ·V(l)                          vagueness
```

Defaults: α=1.0, β=0.5, γ=0.3, δ=0.4, λ=0.05, ε=1.0. β, γ and δ each have an
ablation arm.

**`current_step` is load-bearing.** With every entry aged to 0, the recency term
is the same constant for every candidate, and a constant cannot reorder
anything. `prune()` takes it keyword-only with no default for that reason, and
`frozen_lessons()` defaults it to the last step any entry was used at rather
than to zero.

### Retrieval vs. eviction

Two different keys, on purpose:

- `retrieval_key` is **always** the retention score, blended with query
  relevance. Under the FIFO ablation too — so that ablation changes eviction
  order and nothing else.
- `eviction_key` is the retention score, or `created_at` under `fifo_memory`
  (lowest key evicted first, i.e. oldest first, which is what FIFO means).

### Store capacity vs. prompt budget

Distinct numbers. `store_token_capacity` is how many tokens of lessons to
**keep**; `token_budget` is how many to **show**. They were once the same value,
which made retrieval a no-op — eviction held the store at the budget and
retrieval then filled up to that same budget, so every surviving entry was
always retrieved and ranking never selected anything.

### Vagueness

`V(l)` combines a length signal, a stock-phrase list, and credits for numbers,
applied operators and procedural terms. Two things it is not:

- It is **not arithmetic vocabulary**. The terms are reading-comprehension
  procedure, matched on word boundaries. As substrings, "if" fired on *verify*
  and *specific*; as arithmetic, the term penalised exactly the lessons the
  Reflector is asked to write.
- It is **not language-neutral**. `GENERIC_PHRASES` is English, so the phrase
  term is inert on Nepali lessons. Deduplication is Unicode-aware; this list
  is not, and that is a stated limitation.

---

## Freezing

`frozen_lessons()` takes the top-k entries by retention score and
`build_system_instruction()` renders them into one string, passed to
`simple_evaluate(system_instruction=...)`.

The prefix is **static across the evaluation split**. A frozen playbook cannot
be retrieved per question through that interface, so every question sees the
same lessons. Query-conditioned retrieval is therefore measurable during
adaptation only — a property of the loglikelihood track, and one that has to be
stated as a limitation rather than glossed.

---

## Evaluation

`src/edge_slm_ace/harness/evaluate.py` is a thin wrapper over
`lm_eval.simple_evaluate`:

- the split enters as `samples={task: indices}`, converted from
  language-independent ids to *this language's* document positions — the two
  files are in different row orders, so one shared index list would evaluate a
  different question set per language;
- `log_samples=True`, which the paired test needs and which lets
  `per_item_correctness` verify the harness scored the ids that were requested;
- accuracy is read as `acc`, never `acc_norm` — the choices are single tokens, so
  length normalisation divides every option by the same length;
- a `TruncationCounter` attaches to the root logger for the duration and counts
  the prompts lm-eval had to clip.

---

## The reporting layer

`src/edge_slm_ace/reporting/`.

- `layout.py` owns the results directory shape,
  `{root}/{model}/{language}/{arm}/`. `cell_dir` writes it, `parse_cell` reads
  it, and both the runners and the readers go through it. When the readers
  derived the arm by counting path segments against an older layout, they picked
  up the *language* instead: `summary_accuracy.csv` pooled every arm into one
  row and `compare_arms` produced zero comparisons while exiting 0.
- `schema.py` is the single vocabulary — arm keys, labels, families, reference
  arms, model display names — and carries `implemented`, so an arm with no
  runner is refused rather than silently producing another arm's result under
  its label.
- `load.py` is the only reader of `metrics.json` and `predictions.jsonl`. A
  field the artefact already carries always wins over the path.

---

## Design rules

1. **The harness computes anything reported.** If a metric is ours, it is a
   diagnostic, not a result.
2. **Adaptation and evaluation must be byte-identical in prompt and decision
   rule**, and a test must assert it rather than a comment claiming it.
3. **One owner per shared fact.** The split size, the results layout, the arm
   registry, the definition of "generic" — each lives in one place, because
   every one of them has been wrong by disagreeing with a copy of itself.
4. **A stated invariant gets an assertion.** Prose in a docstring is not
   enforcement; several of the worst defects here sat directly beneath a
   paragraph explaining why they must never happen.
