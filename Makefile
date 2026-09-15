# TinyACE — common tasks.
#
#   make install    development install with all extras
#   make data       fetch/verify Belebele + Global-MMLU (en/ne)
#   make test       run the test suite
#   make check      test + lint + format check (what CI runs)
#   make screen     Week 1 gate: Qwen3+Gemma-3 on Belebele & Global-MMLU
#   make grid       every arm, both languages, one seed
#   make report     aggregate, then test every delta for significance
#                   (STRICT=1 also fails on an invalidating health issue)
#   make mutants    check the tests would catch a change, not just describe one
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

.PHONY: install data test check lint format screen grid report mutants clean

install:
	$(PY) -m pip install -e ".[dev,retrieval,report]"

data:
	$(PY) -m scripts.fetch_belebele
	$(PY) -m scripts.fetch_globalmmlu --check

test:
	$(PY) -m pytest tests/ -v

# Flags live in .flake8, so this and CI cannot drift apart.
lint:
	$(PY) -m flake8 src/ scripts/ tests/ --count --show-source --statistics

format:
	$(PY) -m black --target-version py311 src/ scripts/ tests/

check: test lint
	$(PY) -m black --check --target-version py311 src/ scripts/ tests/

# Pre-registered gate: Qwen3 (1.7B, 4B) and Gemma-3 (1B-it, 4B-it) on Belebele
# and Global-MMLU. A cell enters the lineup only if the Wilson lower bound on
# Nepali clears 30%. Point estimates near chance are not evidence at this n.
screen:
	$(PY) -m scripts.screen --seed $(SEED) --device $(DEVICE) --results-root $(RESULTS)

grid:
	$(PY) -m scripts.run_grid --seed $(SEED) --device $(DEVICE) --results-root $(RESULTS)

# compare_arms is what decides whether a difference is a result. Never report a
# delta that has not been through it.
#
# `make report STRICT=1` also fails the build on an invalidating health issue.
# It is off by default because a non-zero aggregate_results stops make before
# compare_arms runs, which would cost the whole tree's comparisons over one bad
# cell. Either way compare_arms excludes an invalidated run from the family.
report:
	$(PY) -m scripts.aggregate_results --results-root $(RESULTS) $(if $(STRICT),--strict)
	$(PY) -m scripts.compare_arms --results-root $(RESULTS)

# Do the tests actually catch a change, or only describe one? Flips a
# comparison, moves a constant, swaps a boolean, and checks the paired tests
# fail. A surviving mutation is a line nothing is holding. Minutes, not seconds
# -- not part of `make check`.
mutants:
	$(PY) -m scripts.mutation_check

clean:
	rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
