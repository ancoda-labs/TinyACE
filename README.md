# TinyACE: Test-Time Context Adaptation at Edge Scale for Low-Resource Languages

[![CI](https://github.com/ancoda-labs/TinyACE/actions/workflows/ci.yml/badge.svg)](https://github.com/ancoda-labs/TinyACE/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

`TinyACE` evaluates Agentic Context Engineering (ACE) under strict sub-1,000 token context constraints on edge-scale models (1B–4B parameters) in low-resource Devanagari settings (Nepali). It serves as the foundational research layer for edge-scale context engines in resource-constrained environments.

---

## 🔬 Research Overview

Original ACE (Zhang et al., ICLR 2026) assumes context abundance (10K–100K token budgets) and high-compute cloud infrastructure with heavy KV-cache reuse. `TinyACE` tests whether evolving playbook mechanisms survive when context space becomes extremely scarce.

### Core Research Questions
1. **Sub-1K Survival:** Does test-time context adaptation improve accuracy under $T_{\text{max}} \in \{256, 512, 1024\}$ context ceilings?
2. **Token Economics vs. Capability:** Does Devanagari script token inflation mechanically degrade context adaptation budgets? The ratio on this study's models, measured with each model's own tokenizer, is `results/fertility/fertility.csv`.

---

## 🛠️ Repository Architecture

```text
TinyACE/
├── configs/                  # Hardware & evaluation specs
├── scripts/
│   └── screen.py             # Pre-registered screening gate runner
├── src/edge_slm_ace/
│   ├── adapt.py              # Dual-pass adaptation engine (Trace + Loglik)
│   ├── data/                 # Passage-disjoint data loaders (Global-MMLU, Belebele)
│   ├── eval/                 # Statistics (Wilson, McNemar, Bootstrap CIs) & YAML tasks
│   ├── harness/              # lm-evaluation-harness bridge & prompt verification
│   ├── memory/               # Playbook Curator, Retriever, and pruning logic
│   └── utils/                # Device/OOM management & reproducibility trackers
└── tests/                    # Unit & mutation test suite (380+ passing)
