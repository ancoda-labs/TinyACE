# PAPER_TRACKING.md

Experiment tracking and planning document for the workshop paper.

**Paper Title:** "Domain-Specific Benchmarking of Small Language Models for Edge Devices with Agentic Context Engineering (ACE)"

**Target:** Workshop paper submission (2-week timeline)

---

## Models (Edge-Friendly Small LMs)

### Primary Models (1-8B parameters)

1. **Phi-3 Mini** (`microsoft/Phi-3-mini-4k-instruct`)
   - Size: 3.8B parameters
   - Context: 4K tokens
   - Use case: Primary edge-friendly model
   - Config key: `phi3-mini`

2. **Llama 3.2 1B** (`meta-llama/Llama-3.2-1B-Instruct`)
   - Size: 1B parameters
   - Context: 8K tokens
   - Use case: Ultra-lightweight edge model
   - Config key: `llama-3.2-1b`

### Secondary Models (for comparison)

3. **Mistral-7B** (`mistralai/Mistral-7B-Instruct-v0.3`)
   - Size: 7B parameters
   - Context: 8K tokens
   - Use case: Upper bound comparison (supercomputer only)
   - Config key: `mistral-7b`

4. **Llama-3-8B** (`meta-llama/Meta-Llama-3-8B-Instruct`)
   - Size: 8B parameters
   - Context: 8K tokens
   - Use case: Upper bound comparison (supercomputer only)
   - Config key: `llama-3-8b`

### Test Model

- **Tiny GPT-2** (`sshleifer/tiny-gpt2`)
  - Size: ~124M parameters
  - Use case: Infrastructure smoke tests only (not for paper)
  - Config key: `tiny-gpt2`

---

## Tasks (Domain-Specific Benchmarks)

### 1. Science Domain: SciQ Tiny (`sciq_tiny`)
- **Domain**: Finance
- **Dataset**: `data/tasks/sciq_tiny.json`
- **Examples**: 3 (tiny test set)
- **Type**: Financial QA with numeric reasoning
- **Metrics**: Exact Match (EM) for numeric answers

### 2. Medical Domain: MedQA Tiny (`medqa_tiny`)
- **Domain**: Medical
- **Dataset**: `data/tasks/medqa_tiny.json`
- **Examples**: 3 (tiny test set)
- **Type**: Medical question answering
- **Metrics**: Exact Match (EM) or F1 score

### 3. IoT Domain: Anomaly Detection (`iot_tiny`)
- **Domain**: IoT
- **Dataset**: `data/tasks/iot_tiny.json`
- **Examples**: 5 (tiny test set)
- **Type**: IoT sensor data interpretation and anomaly detection
- **Metrics**: Exact Match (EM)

**Note**: These are tiny test sets for development. For paper, we may need to expand to full datasets or larger subsets.

---

## Baselines (Three Evaluation Modes)

### 1. Vanilla Baseline (`baseline`)
- **Description**: Standard zero-shot inference without any adaptation
- **Implementation**: `run_dataset_baseline()` in `slm_ace/runner.py`
- **How to run**:
  ```bash
  python -m scripts.run_experiment \
    --model-id phi3-mini \
    --task-name sciq_tiny \
    --mode baseline \
    --output-path results/sciq_phi3_baseline.csv
  ```
- **Status**: ✅ Implemented and tested

### 2. ACE-Full (`ace`)
- **Description**: Full ACE pipeline with Generator → Reflector → Curator
- **Implementation**: `run_dataset_ace()` in `slm_ace/runner.py`
- **Features**:
  - Playbook-based context evolution
  - Reflection on incorrect answers (always) and correct answers (periodically)
  - Playbook pruning and deduplication
- **How to run**:
  ```bash
  python -m scripts.run_experiment \
    --model-id phi3-mini \
    --task-name sciq_tiny \
    --mode ace \
    --playbook-path playbooks/sciq_playbook.jsonl \
    --output-path results/sciq_phi3_ace.csv
  ```
- **Status**: ✅ Pipeline implemented, prompts need refinement (Sathwik's work)

### 3. Alternative Baseline (SPICE/SEAL-lite)
- **Description**: Lightweight alternative to ACE (to be implemented)
- **Options**:
  - **SPICE-lite**: Self-QA without full reflection loop
  - **SEAL-lite**: Simplified self-consistency or lightweight adaptation
- **Status**: ⏳ Not yet implemented (future work or simplified variant)
- **Note**: For paper, we may focus on Vanilla vs ACE-Full if time is limited

---

## Experiment Checklist (Next 2 Weeks)

### Week 1: Infrastructure + ACE Development

#### Person 1 (Suryodaya) - Infrastructure
- [x] ✅ Core framework complete
- [x] ✅ Task registry with 3 tasks
- [x] ✅ Baseline pipeline working
- [x] ✅ CSV schema stable
- [ ] ⏳ Verify all models load correctly on GPU/supercomputer
- [ ] ⏳ Add any missing CLI flags if needed by team

#### Person 2 (Sathwik) - ACE Logic
- [ ] ⏳ Refine generator prompts in `slm_ace/ace_roles.py`
- [ ] ⏳ Improve reflector prompts for better lesson quality
- [ ] ⏳ Test ACE mode on tiny-gpt2 (smoke test)
- [ ] ⏳ Test ACE mode on Phi-3 Mini (GPU laptop)
- [ ] ⏳ Verify playbook quality (specific, actionable lessons)
- [ ] ⏳ Tune reflection frequency and pruning parameters

#### Person 3 (Archit) - Metrics & Evaluation
- [ ] ⏳ Extend `compute_accuracy()` for robust answer comparison
- [ ] ⏳ Add latency percentiles (p50, p95, p99)
- [ ] ⏳ Create `scripts/summarize_results.py` for comparison tables
- [ ] ⏳ Test metrics on baseline CSVs

### Week 2: Full Experiments + Analysis

#### All Team Members
- [ ] ⏳ **Baseline runs** (all models × all tasks):
  - [ ] Phi-3 Mini × sciq_tiny
  - [ ] Phi-3 Mini × medqa_tiny
  - [ ] Phi-3 Mini × iot_tiny
  - [ ] Llama 3.2 1B × sciq_tiny
  - [ ] Llama 3.2 1B × medqa_tiny
  - [ ] Llama 3.2 1B × iot_tiny
  - [ ] (Optional) Mistral-7B or Llama-3-8B on supercomputer

- [ ] ⏳ **ACE runs** (all models × all tasks):
  - [ ] Phi-3 Mini × sciq_tiny (with playbook)
  - [ ] Phi-3 Mini × medqa_tiny (with playbook)
  - [ ] Phi-3 Mini × iot_tiny (with playbook)
  - [ ] Llama 3.2 1B × sciq_tiny (with playbook)
  - [ ] Llama 3.2 1B × medqa_tiny (with playbook)
  - [ ] Llama 3.2 1B × iot_tiny (with playbook)

- [ ] ⏳ **Analysis**:
  - [ ] Compare baseline vs ACE accuracy per model/task
  - [ ] Analyze latency overhead of ACE
  - [ ] Examine playbook evolution (size, quality)
  - [ ] Generate comparison tables for paper

- [ ] ⏳ **Paper writing**:
  - [ ] Methods section (describe ACE pipeline)
  - [ ] Results section (tables + figures)
  - [ ] Discussion (when ACE helps, when it doesn't)

---

## Key Metrics to Report

### Primary Metrics
1. **Accuracy**: Exact Match (EM) or domain-specific metric
   - Baseline accuracy per model/task
   - ACE accuracy per model/task
   - Improvement delta (ACE - Baseline)

2. **Latency**: Generation time
   - Baseline latency (ms per example)
   - ACE latency (generation + reflection)
   - Overhead percentage

### Secondary Metrics
3. **Playbook Quality**:
   - Number of entries after N examples
   - Specificity score (avoid generic advice)
   - Domain relevance

4. **Resource Usage** (if available):
   - Memory footprint
   - Energy consumption (for edge devices)

---

## Experiment Naming Convention

**Format**: `{model}_{task}_{mode}_{timestamp}.csv`

**Examples**:
- `phi3-mini_sciq_tiny_baseline_20241110.csv`
- `phi3-mini_sciq_tiny_ace_20241110.csv`
- `llama-3.2-1b_medqa_tiny_baseline_20241110.csv`

**Playbooks**: `{task}_playbook.jsonl`
- `sciq_playbook.jsonl`
- `medqa_playbook.jsonl`
- `iot_playbook.jsonl`

---

## Hardware Assignment

- **Mac M3 Pro (Suryodaya)**: Smoke tests, debugging, tiny-gpt2 only
- **GPU Laptop (Sathwik)**: ACE development, Phi-3 Mini runs
- **Supercomputer (Archit)**: Full benchmarks, all models, evaluation metrics

---

## Success Criteria for Paper

✅ **Minimum viable results:**
- At least 2 models (Phi-3 Mini + one other)
- At least 2 tasks (finance + one other)
- Baseline vs ACE comparison showing improvement
- Clear latency/accuracy tradeoff analysis

🎯 **Ideal results:**
- All 3 models tested
- All 3 tasks tested
- Consistent ACE improvement across domains
- Playbook quality analysis
- Edge device deployment considerations

---

## Notes

- **No weight fine-tuning**: All adaptation is via context/playbook evolution
- **Focus on edge-friendly models**: 1-8B parameters, runnable on edge devices
- **ACE is inspired, not exact reproduction**: We adapt the concept for our use case
- **Time constraint**: 2 weeks to results → prioritize working pipeline over perfect prompts

---

## Questions / Blockers

- [ ] Do we need full datasets or are tiny sets sufficient for workshop paper?
- [ ] Should we implement SPICE/SEAL-lite or focus on Vanilla vs ACE?
- [ ] What's the minimum improvement threshold to claim success?
- [ ] Do we need to measure energy consumption or just latency?

---

**Last Updated**: 2024-11-10  
**Next Review**: After Week 1 experiments complete

