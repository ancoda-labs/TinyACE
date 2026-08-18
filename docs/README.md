# Documentation

| | |
|---|---|
| **[installation.md](installation.md)** | Environment setup |
| **[quickstart.md](quickstart.md)** | First run, in a few minutes |
| **[architecture.md](architecture.md)** | How the ACE loop and playbook work |
| **[evaluation.md](evaluation.md)** | **The protocol**: arms, splits, metrics, statistics |
| **[figures.md](figures.md)** | Generating paper figures and diagnostics |
| **[api.md](api.md)** | Package reference |
| **[results.md](results.md)** | Withdrawn results, kept for provenance |
| **[code-review.md](code-review.md)** | Full audit that prompted the current state |
| **[code-review-followup.md](code-review-followup.md)** | Follow-up review of the current state: what the repairs left behind |

---

## Start here

New to the project:
[installation](installation.md) → [quickstart](quickstart.md) → [architecture](architecture.md)

About to run experiments and report numbers:
**[evaluation.md](evaluation.md)** first. It covers which arm to compare
against, why option order is permuted, and what sample size is needed for a
difference to mean anything — the things that went wrong last time.

Wondering why something is built the way it is:
[code-review.md](code-review.md) documents the defects the current design is a
response to, and
[code-review-followup.md](code-review-followup.md) reviews what the repairs
themselves left behind.

---

## Status

The previously published numbers are **withdrawn pending re-evaluation** — see
[results.md](results.md) for the tables with per-section annotations, and the
[main README](../README.md) for the summary of what was wrong.

The framework is fixed; the runs have not been redone.
