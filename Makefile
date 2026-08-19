# TinyACE-Nepali — common tasks.
#
#   make install    development install with all extras
#   make data       fetch and verify the Belebele language files
#   make test       run the test suite
#   make check      test + lint + format check (what CI runs)
#   make screen     Nepali screening run; gates entry to the main grid
#   make grid       every arm, both languages, one seed
#   make report     aggregate, then test every delta for significance
#   make clean      remove caches and build artifacts
#
# A multi-seed study needs one results root per seed, because the layout has no
# seed segment:
#   make grid SEED=42 RESULTS=results/seed42
#   make grid SEED=43 RESULTS=results/seed43

SEED    ?= 42
RESULTS ?= results
DEVICE  ?= cuda
PY      ?= python

.PHONY: install data test check lint format screen grid report clean

install:
	$(PY) -m pip install -e ".[dev,retrieval,report]"

data:
	$(PY) -m scripts.fetch_belebele

test:
	$(PY) -m pytest tests/ -v

# Flags live in .flake8, so this and CI cannot drift apart.
lint:
	$(PY) -m flake8 src/ scripts/ tests/ --count --show-source --statistics

format:
	$(PY) -m black --target-version py311 src/ scripts/ tests/

check: test lint
	$(PY) -m black --check --target-version py311 src/ scripts/ tests/

# Pre-registered gate: a model enters the grid only if the lower bound of its
# Wilson interval on Nepali clears the floor. Point estimates near chance are
# not evidence at this sample size.
screen:
	$(PY) -m scripts.screen_models --seed $(SEED) --device $(DEVICE) --results-root $(RESULTS)

grid:
	$(PY) -m scripts.run_grid --seed $(SEED) --device $(DEVICE) --results-root $(RESULTS)

# compare_arms is what decides whether a difference is a result. Never report a
# delta that has not been through it.
report:
	$(PY) -m scripts.aggregate_results --results-root $(RESULTS)
	$(PY) -m scripts.compare_arms --results-root $(RESULTS)

clean:
	rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
