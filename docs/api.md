# API Reference

Public surface of `edge_slm_ace`, by module. Docstrings in the source carry the
detail and the rationale; this is the map.

---

## `edge_slm_ace.data`

```python
from edge_slm_ace.data import load_belebele, study_split, BELEBELE_LANGUAGES
```

| | |
|---|---|
| `load_belebele(language, path=None, domain=None)` | Normalised examples, **sorted by id** so `load_belebele("en")[i]` and `load_belebele("ne")[i]` are the same question |
| `study_split(item_ids, seed, adaptation_size=None)` | The study's one split. Every entrypoint uses this rather than sizing its own |
| `parallel_split(item_ids, adaptation_size, seed)` | The primitive underneath it |
| `item_id(record)` | `bel-<hash>-q<n>`, from `(link, question_number)` — the only safe cross-language join key |
| `harness_indices(language, item_ids)` | Ids → this language's document positions |
| `HARNESS_TASKS` | `{"en": "belebele_eng_Latn", "ne": "belebele_npi_Deva"}` |

An example:

```python
{
  "id": "bel-015eac1bdc-q2", "language": "ne", "domain": "belebele_ne",
  "context": "<the FLORES passage>", "question": "...",
  "options": ["...", "...", "...", "..."],
  "gold_option_idx": 2, "answer": "...",
}
```

Do not derive ids from row position. The language files hold the same 900
questions in different orders.

---

## `edge_slm_ace.harness`

```python
from edge_slm_ace.harness import OptionScorer, run_frozen_eval, per_item_correctness
```

### `OptionScorer`

```python
scorer = OptionScorer("Qwen/Qwen3-1.7B", device="cuda", apply_chat_template=True)
chosen_idx, logprobs = scorer.score_one(example, lessons=["..."])
```

Picks an option by loglikelihood, exactly as the evaluation task does.
`apply_chat_template` is a constructor argument so a caller cannot adapt in one
mode and evaluate in the other.

### `run_frozen_eval(...)`

```python
results = run_frozen_eval(
    model_id, language, item_ids,
    lessons=frozen, include_scaffold=True,
    seed=42, device="cuda", apply_chat_template=True, lm=already_loaded,
)
```

A stock `simple_evaluate` call. Returns the harness results dict with a
`"tinyace"` key carrying the arm's configuration and the truncation counts, so a
number can be traced back to what produced it. Pass `lm=` to reuse a loaded
checkpoint — a grid runs a dozen arms per model.

### Reading results

| | |
|---|---|
| `accuracy_of(results, language)` | `acc`, never `acc_norm` |
| `per_item_correctness(results, language, expected_ids=None)` | `{item_id: 0/1}`. With `expected_ids`, raises if the harness scored a different set |

### `edge_slm_ace.harness.prompts`

| | |
|---|---|
| `render_question(example)` | The task template, rendered |
| `build_context(question, system_instruction, apply_chat_template, chat_template)` | The full context, assembled by lm-eval's own helpers |
| `choice_continuations(apply_chat_template=False)` | `[" A", ...]`, or `["A", ...]` under a chat template |
| `build_system_instruction(lessons, include_scaffold=True)` | The arm's prefix. Empty string for `baseline` |
| `assert_matches_harness()` | Raises if the installed harness's task has drifted |

---

## `edge_slm_ace.adapt`

```python
from edge_slm_ace.adapt import adapt_playbook, frozen_lessons, save_adaptation_log

summary = adapt_playbook(
    scorer, adaptation_examples, playbook, domain="belebele_ne",
    top_k=5, use_curator=True, prune_every_n=25, max_entries_per_domain=32,
)
lessons = frozen_lessons(playbook, "belebele_ne", top_k=5)
```

`adapt_playbook` mutates the playbook in place and returns `{log, accuracy,
playbook_size, ...}`. `ADAPT_LOG_FIELDS` defines the per-step CSV columns next to
the only code that builds a row.

**`examples` must be the adaptation split only.** The Reflector is shown gold
answers.

`frozen_lessons` ages entries at the last step any of them was used, so recency
actually participates in the ranking that ships.

---

## `edge_slm_ace.memory`

```python
from edge_slm_ace.memory.playbook import Playbook, PlaybookEntry, ScoringParams
```

### `Playbook`

| | |
|---|---|
| `get_top_k(domain, k, current_step, query=None)` | Retrieval ranking; `query` enables relevance blending |
| `get_top_entries_for_budget(domain, token_budget, ...)` | The same ranking, greedily filled to a token budget |
| `add_entry(domain, text, step)` | With deduplication and, if capacity is set, eviction |
| `record_feedback(entry_id, helpful)` | Only for an entry that was actually shown |
| `mark_entry_used(entry_id, step)` | Recency |
| `prune(max_entries_per_domain, *, current_step)` | `current_step` is keyword-only with **no default** |
| `save(path)` / `Playbook.load(path)` | JSONL |

### `ScoringParams`

`alpha`, `beta`, `gamma`, `delta`, `lambda_decay`, `epsilon`,
`relevance_weight`, plus the ablation switches `disable_vagueness_penalty`,
`disable_recency_decay`, `disable_failure_penalty` and `fifo_memory`. Each
switch has a registered arm and a `run_arm` flag.

### `compute_vagueness_score(text) -> float`

0 = specific, 1 = generic. Reading-comprehension procedure earns credit;
arithmetic still does, via numbers and applied operators. English-only on the
phrase list.

### `LessonRelevance`

Singleton cosine relevance over a **multilingual** encoder
(`paraphrase-multilingual-MiniLM-L12-v2` by default; override with
`TINYACE_RELEVANCE_ENCODER`). `available` reports False rather than degrading
silently — an English-only encoder maps Devanagari to near-noise, which looks
like retrieval and is not.

---

## `edge_slm_ace.eval.stats`

Standard library only, so a missing optional dependency can never cause a
comparison to be skipped.

| | |
|---|---|
| `wilson_interval(successes, n, confidence=0.95)` | |
| `summarize_accuracy(correctness, confidence=0.95)` | Adds `ci_halfwidth` — the number to compare a claimed improvement against |
| `mcnemar_exact(arm_a, arm_b)` | Exact paired test. Read `n_discordant` |
| `holm_bonferroni(p_values)` | Adjusted p-values, index-aligned |
| `align_on_key(a, b, metric="is_correct", key="qid")` | |
| `compare_arms(a, b, ...)` | The whole comparison, with a one-line `verdict` |
| `format_comparison(comparison)` | Human-readable block |

---

## `edge_slm_ace.reporting`

| | |
|---|---|
| `cell_dir(root, model, language, arm)` / `parse_cell(path)` | The one owner of the results layout |
| `ARMS`, `get_arm(key)`, `implemented_arms()` | The arm registry |
| `arm_label`, `arm_order`, `model_label` | Display vocabulary |
| `reference_for(key)`, `is_ablation(key)`, `ABLATION_REFERENCE` | Which arm a given arm is compared against |
| `load_run_metrics(root)` / `load_predictions(root)` | pandas frames |
| `summarize_predictions(df, group_by=None)` | Per-arm accuracy with intervals |

---

## `edge_slm_ace.utils`

| | |
|---|---|
| `MODELS`, `resolve_model(key_or_id)`, `ModelSpec` | The model registry. An unregistered id still runs, with a synthesised spec |
| `ADAPTATION_SIZE`, `SCREENING_N`, `SCREENING_FLOOR`, `CHANCE_FLOOR` | The split and the pre-registered gate |
| `screening_verdict(ci_low)` | Whether a model qualifies for the grid |
| `set_seed(seed)`, `DEFAULT_SEED` | Every RNG that can affect a run |
| `capture_environment()` | Versions, hardware, git commit and dirtiness |
| `REPO_ROOT`, `DATA_ROOT_ENV` | Data-root resolution; `TINYACE_DATA_ROOT` overrides |

---

## Scripts

All are `python -m scripts.<name>`; `--help` on each.

| | |
|---|---|
| `fetch_belebele` | Download or `--check` the language files |
| `screen_models` | The pre-registered Nepali gate |
| `run_arm` | One cell: adapt if the arm needs it, then evaluate frozen |
| `run_grid` | Every arm x language x model, one load per model. `--dry-run` |
| `aggregate_results` | `summary_runs.csv`, `summary_accuracy.csv`, health warnings |
| `compare_arms` | Paired tests against registered references, Holm-corrected |
