# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — third review pass

A review of the state the harness rebuild left behind. Every finding was
reproduced before it was fixed and is covered by a test.

**Results that could be lost or silently wrong**

- A run whose task reported no `acc` wrote `metrics.json` and
  `predictions.jsonl` and *then* died formatting `None` as a percentage.
  `run_grid.is_complete()` reads exactly those two files, so the half-written
  cell counted as finished for that commit and seed: the rerun skipped it and
  it stayed empty. It now fails before anything is written.
- `harness_indices` numbered file lines while filtering blank ones, so one blank
  line in a committed jsonl shifted every position after it. The harness skips
  blanks when it loads, so the run would have scored a different document per id
  and still reported a clean accuracy.
- Playbook token counts were always the `words * 1.3` estimate, because no
  entrypoint passed a tokenizer. Words per lesson is roughly
  language-independent and Devanagari fertility is *tokens per word*, so the
  estimate reported Nepali lessons as **cheaper** than the English ones they
  were translated from — 0.58x where the real ratio is 4.5x, the effect with its
  sign reversed. That is the quantity `tinyace_equal_lessons` exists to measure.
  `run_arm` now passes the scorer's tokenizer, and records `token_counts_exact`.
- `max_entries_per_domain` held only when the number of adaptation items divided
  evenly by `prune_every_n`. 400 items pruned every 25 is exact, which hid it;
  `--limit` or any odd split ended mid-cycle and saved a playbook over the cap.
  That playbook is what `tinyace_playbook_en` borrows. The last step now prunes.
- Both metric pickers preferred `oma_correct` over `is_correct`. Nothing has
  written that column since scoring moved to the harness, so the only trees
  carrying it are the withdrawn SciQ results — the preference could only report
  a withdrawn number in place of the live one. A test asserted that behaviour.
- `is_generic` was written into `playbook.jsonl` from a hardcoded 0.5 while
  `_GENERIC_FLOOR` (0.6) decided admission. The score reaches 0.55, so a lesson
  the pipeline deliberately admitted was recorded in the artifact as generic.
- `--dtype` claimed to be recorded in metadata and was recorded nowhere; under
  the grid it was also never forwarded, and a prebuilt `lm` skips
  `build_model_args`, so it had no effect on the run either.
- `blend` combined its two score lists with `zip`, which truncates: a short
  relevance list would have removed the tail entries from retrieval altogether.
  It now raises.

**Run health**

- `compare_arms` read `predictions.jsonl` and nothing else, so the one tool the
  project points at as the gate on whether a delta is a result could not see the
  condition that makes a delta uninterpretable. It would print "TinyACE is
  better than Scaffold Control (p=0.0151)" from a pair where the ACE arm had
  lost 12.5% of its passages to left-truncation and the control had not.
  Truncation is arm-asymmetric — the longer prefix truncates more — so such a
  pair carries *biased* evidence, which is worse for a family of tests than no
  evidence. It is now dropped from the Holm family, with a NOT REPORTABLE
  verdict in place of a direction.
- Invalidating and limiting conditions were both just "warnings". They are now
  distinguished in `reporting/health.py`, on one definition both tools act on:
  truncation invalidates, while `relevance_active` and `token_counts_exact` are
  limitations — those runs really did score what they scored.
- `aggregate_results --strict` (and `make report STRICT=1`) exits non-zero on an
  invalidating issue. Opt-in, because a non-zero aggregate stops make before
  compare_arms runs.

**Documentation that contradicted the code**

- `data/belebele.py` and `utils/repro.py` claimed option order is permuted per
  example. Nothing permutes, and nothing here can — the harness loads its own
  copy of the split. The residual gold-position bias is now stated as a
  limitation in `evaluation.md`.
- `relevance.py` told the user to install the `metrics` extra; the extra is
  `retrieval`. That message is the only warning at the moment retrieval stops
  being query-conditioned, so following it has to work.
- `architecture.md` presented the token-budget machinery as part of the
  protocol. Nothing in `scripts/` sets it, so `max_entries_per_domain` is the
  only bound in force.
- The `tinyace_equal_lessons` note described a token budget that does not exist.
  Both arms budget by lesson count; the grid gives this one `--top-k 10`.
- `aggregate_results` documented a four-segment layout, `playbook.py`
  advertised `ace_full`/`ace_working_memory` modes that are not arms,
  `relevance.py` cited a deleted `SemanticEvaluator`, and `schema.py` claimed a
  pattern list was sorted longest-first.

**Removed**

- `task_label`, exported and called by nothing.
- Four test wrappers that re-ran other tests under "legacy" names.

**Infrastructure**

- flake8 flags moved to `.flake8`; the Makefile and CI carried the same list
  verbatim.
- `run_grid` printed `models x languages x arms`, which overstates the plan
  whenever an arm is language-restricted.

### Fixed — second review pass

A second review of the state the first audit's repairs left behind. All 14
findings addressed, plus three defects the fixes themselves surfaced.

**Runs that could not complete**

- Every ACE run died with `UnboundLocalError: statistics`. Both runners carried
  a redundant local `import statistics`, which made the name local to the whole
  function and broke the earlier `statistics.pstdev` call. It fired as soon as
  a domain held two playbook entries, i.e. on every real run.
- Every ACE run given `--metrics-path` crashed writing `playbook_log.csv`: the
  writer's hardcoded header had not followed the Curator's two new log fields.
  The crash came *after* the evaluation, so a completed run was discarded. The
  column list now lives beside the code that writes a row.
- That crash was reported as `Error: Failed to load model`, because one
  `try/except` wrapped loading, parsing, the evaluation loop and every write.
  The handler now covers only the model load.

**Measurement**

- The `Used strategies:` citation block the ACE prompt asks for was parsed into
  the scored answer, so predictions read `"mitochondria\nUsed strategies:\n1, 3"`.
  Only the ACE arm is asked to cite, so exact match was corrupted in one arm and
  intact in the other.
- `compare_arms.py` imported `reference_for` and never called it, comparing every
  arm against `baseline` -- including ablations, which measures ACE *plus* the
  ablation. Each arm is now paired with its registered reference.
- Comparisons were reported at an uncorrected p<0.05 each. Ten tests against a
  reference give ~40% chance of a false positive; Holm correction is now applied
  across the family.
- `run_ace_epoch.py` called neither `set_seed` nor `option_shuffle_seed`, putting
  the gold answer at (A) for 100% of examples -- the first audit's headline
  finding, alive in one unchecked script.
- `context_tokens` held the retrieved lessons in the ACE arm and the SciQ support
  passage in the baseline arm, and the "token efficiency" figure plotted that
  column across arms. It now means the task context everywhere, the playbook has
  its own column, and the figure measures `prompt_tokens`.
- `prune()` defaulted `current_step` to 0, giving every entry the identical
  recency bonus, so pruning ignored recency entirely.
- The vagueness heuristic counted any hyphen as a formula, so
  "Think carefully about the well-known question" scored 0.25 against a
  threshold of 0.5.

### Added

- **Frozen-playbook protocol.** `--playbook-mode frozen` with `--init-playbook`,
  the `tinyace_wm_256_frozen` arm, and `make adapt && make evaluate`. The
  protocol was prescribed in three places and implemented in none, so the
  default run learned online from the split it was scoring.
- **Retention-scoring hyperparameters are reachable.** `--alpha/--beta/--gamma/
  --delta/--lambda-decay/--epsilon` and `--reflect-on-correct-every-n`, forwarded
  from the config's `scoring:` block, which nothing had ever read. The grid now
  fails on any mode key no consumer reads.
- **Resumable runs.** `--resume` skips examples already in the predictions file
  and merges them back, recomputing the aggregates; the grid skips cells already
  complete for the current commit.
- `tests/test_runner.py`, `tests/test_entrypoints.py`, `tests/test_grid_config.py`
  and `tests/test_resume.py`. `core/runner.py` -- the largest module, and the one
  that decides what "correct" means -- had no tests at all.
- CI runs one real end-to-end experiment, which is what found the `statistics`
  crash, and lints for dead imports (`F401`/`F841`/`F541`), whose absence let
  finding 4 survive the first audit.

### Withdrawn

- **All results reported in 0.1.0.** They were produced by a pipeline with
  defects that change the measurements themselves, not their presentation.
  See `docs/results.md` for the annotated tables, and the Fixed sections
  below for the defects. The framework is fixed; the runs have not been redone.

### Fixed — measurement

- Gold answer was at option (A) in 1000/1000 SciQ examples. Options are now
  permuted per example with a seed derived from `(run_seed, example_id)`.
- `fifo_memory` implemented LIFO: `score()` returned `-created_at`, so
  eviction dropped the newest entry and retrieval only ever showed the oldest
  lessons. Eviction and retrieval are now separate keys.
- OMA mapped predictions by embedding argmax over the entire generation, so
  answering with a letter -- which the prompt invites -- scored near-randomly.
  Replaced with a cascade (letter, verbatim option text, bare letter,
  embeddings) that records which tier resolved each example.
- `self_refine` placed the gold answer in the prompt that produced its scored
  prediction. Removed; the oracle variant is now a separate, labelled arm.
- No chat template was applied to any instruct-tuned checkpoint.
- `truncation=True` had no `max_length`, so prompts could be silently clipped.
- `semantic_answer_score` was length-penalised, favouring terse arms.
- BLEU returned noise for references shorter than four tokens.
- Cosine similarities were clipped to [0, 1], compressing GOM.
- `PeakMemoryTracker` sampled RSS three times and called it a peak.

### Fixed — attribution

- Baseline and ACE differed in prompt, choices block, domain hints and answer
  parser. Unified, and a `cot_control` arm added: the claim about the playbook
  is `ace - cot_control`, not `ace - baseline`.
- Ablations were compared against `baseline` instead of full TinyACE.
  `reporting.reference_for()` now encodes the correct reference per arm.
- Credit was assigned uniformly to every retrieved lesson, leaving the alpha
  and beta terms with no discriminative signal. The Generator now cites the
  strategies it applied.

### Added

- `edge_slm_ace.eval.stats`: Wilson intervals and the exact McNemar test.
- `scripts/compare_arms.py`: paired significance testing between arms.
- `edge_slm_ace.reporting`: one arm registry and one results reader.
- `edge_slm_ace.memory.relevance`: query-conditioned retrieval. Retrieval
  previously ignored the question entirely, so every question in a run
  received the identical lesson list.
- Store capacity separated from prompt budget, so retrieval ranking selects.
- The Curator stage, which was implemented but never invoked.
- `set_seed()` and environment capture; there was previously no seeding.
- `Makefile`, `CITATION.cff`.

### Changed

- `utils` split into `eval` (measurement) and `utils` (run infrastructure).
- Five overlapping result/plot scripts consolidated; `tinyace_plots.py` ->
  `make_figures.py`, `plot_results.py` -> `make_diagnostics.py`.
- Decoding unified across all models; two previously ran at temperature 0.7.
- Docs consolidated under `docs/` with a single index.

### Removed

- `.github/README.md`, a second README that GitHub rendered in preference to
  the root one and that still carried the withdrawn numbers.
- Task registry entries for four datasets that do not exist in the repo.
- `medqa_finetuned_small`, which mapped to DialoGPT-small while being
  described as a fine-tuned upper bound.
- Two committed HTML directory listings, six `.md.old` files, 45KB of dev
  notes, and five of the six documents describing the directory layout.
- Unused dependencies: `datasets`, `jsonlines`, `rouge-score`, `tqdm`,
  `evaluate`. Added `accelerate`, which was required but undeclared.

### Infrastructure

- CI could not fail: every step ended in `|| echo` and the package was never
  installed, so `pytest` had never actually run a test. Fixed, which
  immediately surfaced a failing FIFO test and an F821 in `metrics.py`.
- `pyproject.toml` excluded all subpackages from the built distribution.
- `.gitignore` blanket `*.jsonl` would have silently swallowed new datasets.

## [0.1.0] - 2024-12-15

### Added
- Initial release of TinyACE framework
- Playbook memory system with retention scoring
- Three evaluation modes: baseline, ACE full, ACE working memory
- Token-budgeted memory management (256/512 token budgets)
- Ablation study support (no_failure, no_recency, no_vagueness, FIFO)
- Comprehensive evaluation metrics (OMA accuracy, semantic similarity, latency)
- Multi-device support (CPU, CUDA, MPS)
- Experimental results on Mistral 7B, Phi-3 Mini, TinyLlama 1.1B
- Complete documentation suite

### Features
- **Core System**
  - ACE loop implementation (Generate → Reflect → Curate → Memorize)
  - Retention scoring formula with four components
  - Strategic forgetting via token-budgeted eviction
  - Feedback-on-use mechanism

- **Evaluation**
  - Baseline mode (vanilla prompting)
  - ACE Full mode (top-k retrieval)
  - ACE Working Memory mode (token-budgeted retrieval)
  - MCQ-aware evaluation (OMA, GOM metrics)

- **Ablation Studies**
  - No failure penalty
  - No recency decay
  - No vagueness penalty
  - FIFO eviction

### Documentation
- Comprehensive architecture documentation
- Experimental results analysis
- Quick start guide
- Installation guide
- Plotting guide
- API documentation

### Results
- Phi-3 Mini: +4% OMA improvement with FIFO eviction
- Mistral 7B: Baseline preferred (96% OMA)
- TinyLlama 1.1B: Baseline preferred (72% OMA)
- Key finding: FIFO eviction outperforms complex scoring

---

## [Unreleased]

### Planned
- Additional model support
- More evaluation tasks
- Performance optimizations
- Extended ablation studies
