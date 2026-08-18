# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
