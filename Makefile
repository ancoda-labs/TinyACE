# TinyACE — common tasks.
#
#   make install    development install with all extras
#   make test       run the test suite
#   make check      test + lint + format check (what CI runs)
#   make smoke      end-to-end pipeline check on a tiny model
#   make grid       full evaluation grid, seeded
#
# A multi-seed study needs one results root per seed, because the layout has
# no seed segment:
#   make grid SEED=42 RESULTS=results/seed42
#   make grid SEED=43 RESULTS=results/seed43
#   make adapt      stage 1: build playbooks on the adaptation split
#   make evaluate   stage 2: score read-only on the test split
#   make report     aggregate results, then test every delta for significance
#   make figures    paper figures
#   make clean      remove caches and build artifacts

SEED    ?= 42
RESULTS ?= results
FIGURES ?= figures
CONFIG  ?= configs/experiment_grid.yaml
PY      ?= python

.PHONY: install test check lint format smoke grid adapt evaluate report figures clean

install:
	$(PY) -m pip install -e ".[dev,metrics,plots]"

test:
	$(PY) -m pytest tests/ -v

lint:
	$(PY) -m flake8 src/ scripts/ tests/ --count --select=E9,F63,F7,F82,F401,F841,F541 --show-source --statistics

format:
	$(PY) -m black --target-version py311 src/ scripts/ tests/

check: test lint
	$(PY) -m black --check --target-version py311 src/ scripts/ tests/

smoke:
	$(PY) -m scripts.smoke_test

grid:
	$(PY) -m scripts.run_eval_grid --config $(CONFIG) --seed $(SEED) --results-root $(RESULTS)

# The protocol in docs/evaluation.md, in the order it has to run: build a
# playbook on the adaptation split, then score read-only on the test split.
# The frozen arm reads what `adapt` leaves behind, so running `evaluate` first
# has nothing to freeze.
adapt:
	$(PY) -m scripts.run_eval_grid --config $(CONFIG) --seed $(SEED) --results-root $(RESULTS) --only-task sciq_val

evaluate:
	$(PY) -m scripts.run_eval_grid --config $(CONFIG) --seed $(SEED) --results-root $(RESULTS) --only-task sciq_test

# Aggregation prints run-health warnings; compare_arms is what decides whether
# a difference is a result. Never report a delta that has not been through it.
report:
	$(PY) -m scripts.aggregate_results --results-root $(RESULTS)
	$(PY) -m scripts.compare_arms --results-root $(RESULTS)

figures:
	$(PY) -m scripts.make_figures --results_dir $(RESULTS) --output_dir $(FIGURES)

clean:
	rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
