# TinyACE Experimental Results — Archive

> **These are the withdrawn results of the retired SciQ pipeline.** They
> describe scripts, arms and metrics that no longer exist in this repository:
> `run_experiment.py`, `--mode ace`, `ace_working_memory`, `self_refine`, OMA.
> The file is kept unedited, for provenance. For what the project does now, see
> [../README.md](../README.md) and [evaluation.md](evaluation.md).

> **STATUS: WITHDRAWN PENDING RE-EVALUATION.**
>
> Every number below was produced by a pipeline with defects that change the
> measurements themselves. They are retained for provenance only and must not
> be cited, quoted or carried into the paper. `CHANGELOG.md` lists every defect;
> the short version:
>
> 1. **The gold answer sat at option (A) in 100% of examples.** Verified:
>    1000/1000 rows of `sciq_test.json`. Any model with a first-option bias
>    was rewarded, and that bias differs between the terse baseline prompt and
>    the verbose ACE prompt, so it does not cancel between arms.
> 2. **The FIFO arm ran LIFO.** `score()` returned `-created_at`, so eviction
>    dropped the *newest* entry and retrieval only ever showed the *oldest*
>    lessons. Whatever produced the 78% headline, it was not FIFO eviction.
> 3. **OMA was embedding-argmax over the whole generation.** A model that
>    answered "B" -- which the prompt explicitly invites -- was scored close to
>    randomly, and long chain-of-thought output diluted the option signal.
> 4. **The arms were not comparable.** Baseline and ACE differed in prompt,
>    choices block, domain hints *and* answer parser, so no delta could be
>    attributed to the playbook.
> 5. **The self-refine arm was shown the gold answer** in the prompt that
>    produced its scored prediction.
> 6. **No seed, anywhere**, while two of the models decoded at temperature 0.7.
> 7. **Every difference reported here is inside the noise floor.** At n=50 the
>    95% Wilson CI is about +/-12pp; the deltas below are 1-3 questions.
> 8. **The ablations were compared against the wrong arm** -- see the section
>    below.
>
> Re-run with `scripts/run_eval_grid.py` at n>=500 with >=3 seeds, then check
> every delta with `scripts/compare_arms.py` before writing anything down.

---

## Original report (superseded)

**Last Updated:** December 2024  
**Evaluation Date:** December 2024

---

## Table of Contents

1. [Overview](#overview)
2. [Model Comparison Results](#model-comparison-results)
3. [Ablation Study Results](#ablation-study-results)
4. [Key Findings](#key-findings)
5. [Component Analysis](#component-analysis)

---

## Overview

This document summarizes experimental results from evaluating TinyACE across three models (Mistral 7B, Phi-3 Mini, TinyLlama 1.1B) on the SciQ test dataset, including ablation studies of the retention scoring components.

**Test Configuration:**
- **Task**: `sciq_test` (Science domain, multiple-choice questions)
- **Test Size**: 50 examples per configuration
- **Device**: CUDA
- **Metrics**: OMA Accuracy, Semantic Similarity, Latency, Playbook Size

---

## Model Comparison Results

### Mistral 7B Instruct

| Configuration | OMA Accuracy | Semantic Similarity | Avg Latency (s) | Playbook Entries | Playbook Tokens |
|--------------|--------------|---------------------|-----------------|------------------|-----------------|
| **baseline** | **96%** | **0.823** | **0.408** | N/A | N/A |
| **ace_full** | 92% | 0.803 | 1.941 | 16 | 499 |
| **tinyace_wm_256** | 92% | 0.820 | 1.847 | 9 | 241 |
| **tinyace_wm_512** | 94% | 0.796 | 1.814 | 18 | 505 |

**Key Observations:**
- Baseline performs best (96% OMA, fastest)
- ACE modes add ~1.4s latency overhead
- Larger token budget (512) allows more entries but slightly lower semantic similarity
- **Conclusion**: Strong models benefit less from ACE; baseline simplicity preferred

### Phi-3 Mini

| Configuration | OMA Accuracy | Semantic Similarity | Avg Latency (s) | Playbook Entries | Playbook Tokens |
|--------------|--------------|---------------------|-----------------|------------------|-----------------|
| **baseline** | 74% | 0.385 | **6.430** | N/A | N/A |
| **ace_full** | 60% | 0.425 | 12.161 | 26 | 472 |
| **tinyace_wm_256** | 72% | **0.495** | 11.882 | 11 | 208 |
| **tinyace_wm_512** | **78%** | 0.480 | 12.008 | 17 | 506 |

**Key Observations:**
- ACE significantly improves semantic similarity (+0.095 to +0.110)
- Best OMA achieved with `tinyace_wm_512` (78%)
- Best semantic similarity with `tinyace_wm_256` (0.495)
- Trade-off: More entries vs. higher-quality entries
- **Conclusion**: Medium models benefit most from ACE, especially semantic quality

### TinyLlama 1.1B

| Configuration | OMA Accuracy | Semantic Similarity | Avg Latency (s) | Playbook Entries | Playbook Tokens |
|--------------|--------------|---------------------|-----------------|------------------|-----------------|
| **baseline** | **72%** | **0.659** | **0.967** | N/A | N/A |
| **ace_full** | 34% | 0.274 | 6.142 | 24 | 496 |
| **tinyace_wm_256** | 40% | 0.319 | 6.994 | 14 | 244 |
| **tinyace_wm_512** | 46% | 0.392 | 7.853 | 24 | 512 |

**Key Observations:**
- ACE degrades performance across all metrics
- Baseline significantly outperforms all ACE modes
- Larger playbook (ace_full) performs worse
- **Conclusion**: Very small models may be confused by playbook context; baseline preferred

---

## Ablation Study Results

All ablation studies conducted on **Phi-3 Mini** with **256 token budget**.

| Configuration | OMA Accuracy | Semantic Similarity | Avg Latency (s) | Avg GOM | Playbook Entries | Playbook Tokens |
|--------------|--------------|---------------------|-----------------|---------|------------------|-----------------|
| **baseline** | 74% | 0.385 | **6.430** | 0.133 | N/A | N/A |
| **tinyace_fifo** | **78%** | 0.469 | 8.992 | 0.222 | 9 | 226 |
| **tinyace_ablate_no_recency** | 72% | **0.495** | 8.833 | 0.213 | 11 | 208 |
| **tinyace_ablate_no_failure** | 70% | 0.466 | 8.882 | 0.195 | 9 | 248 |
| **tinyace_ablate_no_vagueness** | 72% | 0.421 | 9.288 | 0.191 | 17 | 248 |

> **Reference-arm error.** Each `tinyace_ablate_no_X` row below is compared
> against **baseline** (74%), which is the no-ACE arm. An ablation of a TinyACE
> component must be compared against **full TinyACE at the same budget**, which
> the model table above gives as `tinyace_wm_256` = 72%. Against the correct
> reference:
>
> | Config | As reported (vs baseline) | Correct (vs tinyace_wm_256) |
> |---|---|---|
> | `tinyace_fifo` 78% | +4% | +6pp (3 questions) |
> | `no_recency` 72% | -2% | **0pp (0 questions)** |
> | `no_failure` 70% | -4% | -2pp (1 question) |
> | `no_vagueness` 72% | -2% | **0pp (0 questions)** |
>
> Removing the recency term and removing the vagueness term changed nothing at
> all. The stated conclusion that "failure tracking is most critical" rests on
> a single question. None of these differences survives a McNemar test at
> n=50 -- the +4% headline comes out at p=0.50.

### Component Importance Ranking

**By OMA Accuracy:**
1. FIFO (78%) - **Best overall**
2. Baseline (74%)
3. No Vagueness (72%) / No Recency (72%)
4. No Failure (70%) - **Worst**

**By Semantic Similarity:**
1. No Recency (0.495) - **Best**
2. FIFO (0.469)
3. No Failure (0.466)
4. No Vagueness (0.421)
5. Baseline (0.385) - **Worst**

### Detailed Ablation Analysis

#### 1. FIFO Eviction (`tinyace_fifo`)
- **Best OMA**: 78% (+4% vs baseline)
- **Semantic Similarity**: 0.469 (+0.084 vs baseline)
- **Playbook**: 9 entries, 226 tokens (smallest)
- **Conclusion**: Simple FIFO eviction outperforms complex scoring

#### 2. No Recency Bias (`tinyace_ablate_no_recency`)
- **OMA**: 72% (-2% vs baseline)
- **Best Semantic Similarity**: 0.495 (+0.110 vs baseline)
- **Playbook**: 11 entries, 208 tokens
- **Conclusion**: Removing recency improves semantic match but reduces accuracy

#### 3. No Failure Tracking (`tinyace_ablate_no_failure`)
- **OMA**: 70% (-4% vs baseline) - **Lowest**
- **Semantic Similarity**: 0.466 (+0.081 vs baseline)
- **Playbook**: 9 entries, 248 tokens
- **Conclusion**: Failure tracking is critical for maintaining accuracy

#### 4. No Vagueness Detection (`tinyace_ablate_no_vagueness`)
- **OMA**: 72% (-2% vs baseline)
- **Semantic Similarity**: 0.421 (+0.036 vs baseline)
- **Largest Playbook**: 17 entries, 248 tokens
- **Conclusion**: Without vagueness filtering, playbook accumulates generic entries

---

## Key Findings

### 1. Model-Dependent Effectiveness

- **Strong Models (Mistral 7B)**: Baseline often best; ACE adds overhead
- **Medium Models (Phi-3 Mini)**: ACE helps, especially semantic similarity
- **Weak Models (TinyLlama)**: ACE can hurt; baseline simplicity preferred

### 2. Component Importance

1. **Failure Tracking**: Most critical (OMA drops 4% without it)
2. **FIFO Eviction**: Surprisingly effective (best OMA)
3. **Recency Bias**: Trade-off (accuracy vs. semantic quality)
4. **Vagueness Detection**: Prevents playbook bloat

### 3. Scoring Formula Insights

- Complex scoring may be over-engineered for this task
- FIFO eviction matches or exceeds complex scoring performance
- Failure penalty is essential; recency and vagueness are helpful but not critical

### 4. Token Budget Trade-offs

- **256 tokens**: Smaller playbook, higher semantic quality
- **512 tokens**: Larger playbook, better OMA but lower semantic similarity
- Choice depends on priority: accuracy vs. semantic match quality

---

## Component Analysis

### Retention Scoring Components

| Component | Impact on OMA | Impact on Semantic Similarity | Impact on Playbook Size |
|-----------|---------------|-------------------------------|-------------------------|
| **Recency Bias** | Slight decrease (-2%) | Large increase (+0.110) | Similar |
| **Failure Tracking** | Large decrease (-4%) | Moderate increase (+0.081) | Similar |
| **Vagueness Detection** | Slight decrease (-2%) | Moderate increase (+0.036) | Large increase (+8 entries) |
| **FIFO Eviction** | Best (+4%) | Large increase (+0.084) | Smallest (9 entries) |

### Latency Analysis

> **Internally inconsistent.** The tables above report Phi-3 Mini (3.8B) at a
> 6.43s baseline latency and Mistral-7B (7B) at 0.408s -- a 7B model 15x faster
> than a 3.8B model on the same task and the same reported device. That cannot
> both be true, and indicates the rows were collected under different hardware,
> decoding settings or output lengths. The cross-model latency comparison is
> not usable until that is explained. (MODEL_CONFIGS did in fact give
> mistral-7b temperature 0.7 and 512 max_new_tokens against Phi-3's greedy 256;
> that has since been unified.)


- **Baseline**: Fastest (0.4-6.4s depending on model)
- **ACE Modes**: Add ~2.4-2.9s overhead per run
- **Overhead Sources**: Playbook retrieval, reflection generation, playbook updates

### Playbook Dynamics

- **Size Range**: 9-26 entries depending on mode and model
- **Token Usage**: 208-512 tokens (all within budgets)
- **Growth Pattern**: Playbooks grow initially, then stabilize through eviction/pruning
- **Eviction Frequency**: Higher with smaller token budgets (256 vs 512)

---

## Conclusions

1. **ACE is most effective for medium-capability models** (Phi-3 Mini)
2. **FIFO eviction is surprisingly effective**, suggesting complex scoring may be unnecessary
3. **Failure tracking is critical** for maintaining accuracy
4. **Recency bias trades accuracy for semantic quality** - choose based on priority
5. **Vagueness detection prevents playbook bloat** but may filter useful entries
6. **Token budgets create trade-offs**: More entries vs. higher-quality entries

---

## Recommendations

### For Production Use

1. **Use FIFO eviction** for simplicity and best OMA
2. **Keep failure tracking** - it's critical for accuracy
3. **Consider disabling recency** if semantic quality is priority
4. **Use vagueness detection** to prevent playbook bloat
5. **Choose token budget** based on priority: 256 for quality, 512 for accuracy

### For Further Research

1. Investigate why FIFO outperforms complex scoring
2. Explore adaptive token budgets based on playbook quality
3. Study model-specific optimal configurations
4. Analyze playbook entry quality over time
5. Compare with fine-tuning baselines

---

## Data Location

- **Model Results**: `results_models/`
- **Ablation Results**: `results_ablation/`
- **Metrics Files**: `{model}/{task}/{mode}/{device}/metrics.json`
- **Detailed Results**: `{model}/{task}/{mode}/{device}/results.csv`
