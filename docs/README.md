# Documentation

| | |
|---|---|
| **[installation.md](installation.md)** | Environment setup |
| **[quickstart.md](quickstart.md)** | Clone to first reportable result |
| **[architecture.md](architecture.md)** | The adaptation loop, the playbook, and where the harness boundary sits |
| **[evaluation.md](evaluation.md)** | **The protocol**: splits, arms, statistics, health checks |
| **[api.md](api.md)** | Package reference |
| **[results.md](results.md)** | Withdrawn results from the retired pipeline, kept for provenance |

---

## Start here

New to the project:
[installation](installation.md) → [quickstart](quickstart.md) → [architecture](architecture.md)

About to report a number:
**[evaluation.md](evaluation.md)** first. Which arm to compare against, why a
point estimate is not decidable at this sample size, and what invalidates a run
outright.

Wondering why something is built the way it is:
the docstring above it names the failure mode it prevents. `CHANGELOG.md` has
the same history in one place.

---

## Status

No results yet. The previously published numbers are **withdrawn** — see
[results.md](results.md) — the pipeline was rebuilt around
lm-evaluation-harness, and the runs have not been redone.
