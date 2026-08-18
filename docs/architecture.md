# TinyACE Architecture

How the loop and the playbook work, and which failure mode each design decision
prevents.

> **This document describes the mechanism, not its effect.** No accuracy number
> appears below. The 0.1.0 results are withdrawn (see
> [results.md](results.md)), so any claim that a component "helps" is open until
> the evaluation is redone. `CHANGELOG.md` records what changed and why.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [ACE Loop Implementation](#ace-loop-implementation)
4. [Retention Scoring System](#retention-scoring-system)
5. [Modes of Operation](#modes-of-operation)
6. [Ablation Studies](#ablation-studies)

---

## System Overview

TinyACE (Agentic Context Engineering) is a framework that enables Small Language Models (SLMs) to self-improve without fine-tuning by maintaining a dynamic **Playbook Memory** of domain-specific strategies.

### Key Design Principles

1. **No Model Fine-tuning**: Model weights remain frozen; only prompt context evolves
2. **Domain-Specific Learning**: Playbook accumulates strategies specific to each domain
3. **Strategic Forgetting**: Token-budgeted memory forces prioritization of high-value lessons
4. **Feedback-Driven**: Lessons are evaluated based on actual usage, not creation context

---

## Core Components

### 1. Playbook Memory (`src/edge_slm_ace/memory/playbook.py`)

The playbook stores domain-specific lessons (strategies) that guide the model's reasoning.

**Key Features:**
- Retention scoring for entry prioritization
- Token-budgeted eviction for working memory mode
- Success/failure tracking per entry
- Vagueness detection to filter generic lessons

**Data Structure:**
```python
@dataclass
class PlaybookEntry:
    id: str
    domain: str
    text: str
    success_count: int = 0
    failure_count: int = 0
    last_used_at: int = 0
    token_count: Optional[int] = None      # None = not computed yet
    vagueness_score: Optional[float] = None
```

`token_count` and `vagueness_score` default to `None`, not `0`. With `0` as the
sentinel, a genuinely non-vague entry was rescored on every load and a one-token
entry could never cache its count.

### 2. ACE Roles (`src/edge_slm_ace/core/ace_roles.py`)

Three roles orchestrate the ACE loop:

**Generator:**
- Builds prompts with playbook context
- Generates answers with step-by-step reasoning
- Parses output to extract reasoning and answer

**Reflector:**
- Analyzes incorrect answers (or periodically correct ones)
- Generates specific, actionable lessons
- Filters out generic/vague advice

**Curator:**
- Screens candidate lessons with one extra generation, marking the generic ones
- Costs one model call per reflection step; ablate with `--no-curator`
- Deduplication is the playbook's job, not the Curator's: it compares a
  candidate against the incumbent and keeps whichever is more specific

### 3. Runner (`src/edge_slm_ace/core/runner.py`)

Orchestrates the evaluation pipeline:

- **Baseline Mode**: Vanilla prompting without playbook
- **ACE Full Mode**: Top-k retrieval from unbounded playbook
- **ACE Working Memory Mode**: Token-budgeted retrieval with eviction

---

## ACE Loop Implementation

The ACE loop processes each example through these steps:

```python
for step, example in enumerate(dataset, start=1):
    # 1. RETRIEVE: rank this domain's entries against the question. Ranking is
    #    query-conditioned; with relevance_weight=0 every question in a run
    #    would receive the identical lesson list.
    if ace_mode == "ace_working_memory":
        used = playbook.get_top_entries_for_budget(domain, token_budget, step, query=question)
    else:
        used = playbook.get_top_k(domain, k=top_k, current_step=step, query=question)

    # 2. GENERATE. extract_answer is the single parser every arm goes through;
    #    an asymmetry here shows up as an effect attributed to the playbook.
    raw = generate(model, tokenizer, build_generator_prompt(...))
    answer, reasoning = extract_answer(raw)

    # 3. EVALUATE
    correct = answer.strip().lower() == ground_truth.strip().lower()

    # 4. REFLECT, then CURATE. Skipped entirely when enable_learning is False,
    #    which is what makes cot_control a prompt-matched control and what
    #    makes a frozen playbook frozen.
    if enable_learning and (not correct or step % reflect_on_correct_every_n == 0):
        lessons = choose_lessons_for_playbook(parse_reflector_output_to_lessons(...))
        if use_curator:
            lessons = [l for l, generic in zip(lessons, curate(lessons)) if not generic]
        for lesson in lessons:
            playbook.add_entry(domain, lesson, step)   # no feedback at creation

    # 5. CREDIT. Prefer the entries the Generator says it applied; fall back to
    #    crediting everything retrieved only when it cited nothing, and record
    #    which happened (`credit_mode`). Crediting everything uniformly makes
    #    every entry's success ratio converge to the run's accuracy, so the
    #    alpha and beta terms stop distinguishing between lessons.
    cited = parse_used_strategies(raw, len(used))
    credited = used if cited is None else [used[i] for i in cited]
    for entry in used:
        playbook.mark_entry_used(entry.id, step)       # recency: what was shown
    for entry in credited:
        playbook.record_feedback(entry.id, helpful=correct)   # what was used

    # 6. PRUNE. current_step is required: defaulted to 0, every entry's age
    #    came out 0 and the recency term became a constant, so pruning ignored
    #    recency entirely.
    if enable_learning and step % prune_every_n == 0:
        playbook.prune(max_entries_per_domain, current_step=step)
```

### Critical Design Decision: Feedback on Use, Not Creation

New lessons are added **without** feedback. Feedback is only recorded when a lesson is **actually used** in a subsequent prompt. This prevents biasing lessons based on the example they were derived from.

---

## Retention Scoring System

### Formal Scoring Equation

```
S(l_i, t) = α·(N_succ/(N_used+ε)) - β·(N_fail/(N_used+ε)) 
          + γ·exp(-λ·(t-t_last)) - δ·V(l_i)
```

**Hyperparameters** (defaults; each is a CLI flag, and the resolved values are
written into `metrics.json`):
- `α = 1.0`: Success ratio weight
- `β = 0.5`: Failure ratio penalty weight
- `γ = 0.3`: Recency bonus weight
- `δ = 0.4`: Vagueness penalty weight
- `λ = 0.05`: Recency decay rate
- `ε = 1.0`: Smoothing constant

### Component Breakdown

1. **Success Term**: `α · (N_succ / (N_used + ε))`
   - Rewards entries that lead to correct answers
   - Higher success rate → higher score

2. **Failure Term**: `-β · (N_fail / (N_used + ε))`
   - Penalizes entries that lead to incorrect answers
   - Higher failure rate → lower score
   - Ablate with `--disable-failure-penalty`

3. **Recency Term**: `γ · exp(-λ · (t - t_last))`
   - Bonus for recently used entries
   - Decays exponentially with steps since last use
   - Ablate with `--disable-recency-decay`

4. **Vagueness Term**: `-δ · V(l_i)`
   - Penalizes vague/generic lessons
   - `V(l_i)` computed from:
     - Generic phrases ("think carefully", "pay attention")
     - Short text (< 5 words)
     - Lack of specificity (no numbers, operators applied to numbers, or
       procedural terms). An operator has to touch a digit to count -- testing
       for the bare characters meant any hyphenated word read as a formula
   - Ablate with `--disable-vagueness-penalty`

### Implementation

```python
def score(self, current_step: int, params: ScoringParams) -> float:
    # Success ratio term
    success_term = params.alpha * (self.success_count / (n_used + params.epsilon))
    
    # Failure ratio term (can be disabled)
    if params.disable_failure_penalty:
        failure_term = 0.0
    else:
        failure_term = params.beta * (self.failure_count / (n_used + params.epsilon))
    
    # Recency term (can be disabled)
    if params.disable_recency_decay:
        recency_term = 0.0
    else:
        age = max(0, current_step - self.last_used_at)
        recency_term = params.gamma * math.exp(-params.lambda_decay * age)
    
    # Vagueness penalty (can be disabled)
    if params.disable_vagueness_penalty:
        vagueness_term = 0.0
    else:
        vagueness_term = params.delta * self.vagueness_score
    
    return success_term - failure_term + recency_term - vagueness_term
```

---

## Modes of Operation

The arm registry is [`reporting/schema.py`](../src/edge_slm_ace/reporting/schema.py);
[evaluation.md](evaluation.md) says which arm each one should be compared
against. The distinct behaviours behind those arms are:

| Behaviour | Prompt | Playbook | Learns? |
|---|---|---|---|
| `baseline` | `Question: ...\nAnswer:` plus the shared choices block | none | no |
| `cot_control` | Full ACE scaffold — role preamble, domain hints, mandated reasoning | **empty** | no |
| `ace_full` | Full ACE scaffold | top-k by rank | yes |
| `ace_working_memory` | Full ACE scaffold | token-budgeted | yes |
| frozen (`--playbook-mode frozen`) | Full ACE scaffold | loaded, read-only | **no** |
| `self_refine` | Generate → critique own answer → rewrite | none | no |

Two of these exist to make the others interpretable:

- **`cot_control`** receives the identical scaffold over an empty playbook, so
  `ace - cot_control` isolates the playbook. `ace - baseline` also varies the
  chain-of-thought instruction, the domain hints and the answer parser, which is
  four changes at once.
- **frozen** adapts on one split and scores read-only on another. Every other
  ACE arm learns online from the split it is scored on, and 82% of `sciq_test`
  examples contain the gold answer verbatim in their `support` field, so a
  lesson written after seeing gold can carry answer content forward.

### Store capacity vs prompt budget

Two different numbers, and conflating them made retrieval a no-op:

- `token_budget` — how many tokens of lessons to **show** the Generator.
- `store_token_capacity` — how many to **keep**, defaulting to 4x the budget.

When they were equal, eviction held the store at or below the budget and
retrieval then filled up to that same budget, so every surviving entry was
always retrieved and ranking never selected anything. WM-256 vs WM-512 compared
how many lessons *survived*, not which were *chosen*.

---

## Ablation Studies

The codebase supports ablation studies through scoring parameter flags:

### Available Ablations

Each disables one term, so the comparison against full TinyACE isolates that
term. Results are not listed here: the 0.1.0 ablation numbers are withdrawn.

| Arm | Flag | Isolates |
|---|---|---|
| `tinyace_ablate_no_failure` | `--disable-failure-penalty` | beta = 0 |
| `tinyace_ablate_no_recency` | `--disable-recency-decay` | gamma = 0 |
| `tinyace_ablate_no_vagueness` | `--disable-vagueness-penalty` | delta = 0 |
| `tinyace_ablate_no_relevance` | `--relevance-weight 0` | Domain-only retrieval — every question gets the same lessons |
| `tinyace_ablate_no_curator` | `--no-curator` | The Curator screening pass |
| `tinyace_fifo` | `--fifo-memory` | Oldest-first eviction instead of lowest-score |

`fifo_memory` changes **eviction only**. Retrieval always ranks by retention
score, so the ablation varies one thing. Note that this arm previously
implemented LIFO -- `score()` returned `-created_at`, so it evicted the *newest*
entry -- which is why its old numbers are not evidence about FIFO.

An ablation belongs against `tinyace_wm_256`, not against `baseline`.
Comparing it to baseline measures ACE *plus* the ablation.
`scripts/compare_arms.py` pairs them correctly by default.

### Configuration

Ablations are configured in `configs/experiment_grid.yaml`:

```yaml
modes:
  - name: tinyace_ablate_no_failure
    disable_failure_penalty: true
  - name: tinyace_ablate_no_recency
    disable_recency_decay: true
  - name: tinyace_ablate_no_vagueness
    disable_vagueness_penalty: true
  - name: tinyace_fifo
    fifo_memory: true
```

---

## File Structure

```
src/edge_slm_ace/
├── core/
│   ├── ace_roles.py      Generator/Reflector/Curator prompts + parsers
│   └── runner.py          Baseline / control / ACE / self-refine loops
├── memory/
│   ├── playbook.py       Retention scoring, eviction, token budgets
│   └── relevance.py      Query-conditioned retrieval
├── eval/                 The measurement layer
│   ├── metrics.py        Answer scoring, peak memory
│   ├── mcq.py            Option permutation, OMA / GOM / ACR, mapping cascade
│   └── stats.py          Wilson intervals, exact McNemar, Holm correction
├── reporting/
│   ├── schema.py         Arm registry + which arm to compare against
│   └── load.py           One reader for metrics.json / predictions.jsonl
├── models/
│   └── model_manager.py  Loading, chat templates, prompt-length bounds
└── utils/
    ├── config.py         Model and task registries
    ├── device_utils.py   Device selection
    └── repro.py          Seeding and environment capture
```

`eval/` and `reporting/` were carved out of `utils/`. Anything that reads
results imports its labels from `reporting/schema.py` rather than re-deriving
them -- three scripts previously carried their own copies and disagreed, so the
same run appeared under two different names depending on which script rendered
it.

---

## Key Design Decisions

Mechanism, not measured effect:

1. **Feedback on use, not creation.** A new lesson enters with no
   success/failure record. Crediting it for the example it was derived from
   would score it on its own training case.
2. **Credit to cited lessons.** Crediting everything retrieved gives every live
   entry the same increment on every step, so all success ratios converge to the
   run's accuracy and the alpha/beta terms stop distinguishing between lessons.
   `citation_rate` in `metrics.json` says how often attribution was available.
3. **Strategic forgetting.** A token budget forces a choice, but only when the
   store is larger than the budget.
4. **One parser for every arm.** `extract_answer` is shared, because a parsing
   difference between arms is indistinguishable from an effect of the playbook.
5. **Invalidating conditions travel with the data.** `prompt_truncated` and
   `used_chat_template` are columns, not log lines, so a broken run is visible
   in `metrics.json` rather than only in scrollback.

---

## References

- [evaluation.md](evaluation.md) -- the protocol: arms, splits, metrics, statistics
- [results.md](results.md) -- the withdrawn 0.1.0 tables, and why
- `CHANGELOG.md` -- what changed and which failure mode it addressed
- `configs/experiment_grid.yaml` -- the grid
