# =============================================================================
# DefectVision - task shortcuts.
#
#   make setup      create the venv and install everything
#   make all        run the entire pipeline end to end (M2 -> M5)
#   make demo       the sequence to run during the recorded demo
#
# Windows users without `make` can read each recipe as the command to paste,
# or use the PowerShell equivalents in scripts/.
# =============================================================================

PYTHON  ?= python
VENV    ?= .venv
ifeq ($(OS),Windows_NT)
  BIN := $(VENV)/Scripts
else
  BIN := $(VENV)/bin
endif
PY  := $(BIN)/python
DV  := $(PY) -m defectvision.cli

.DEFAULT_GOAL := help
.PHONY: help setup install data train compare package serve benchmark \
        reference drift monitor retrain all demo test lint format \
        repro dag mlflow docker docker-run clean clean-all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
setup:  ## Create the virtualenv and install all dependencies
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip setuptools wheel
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps
	@echo "Environment ready. Activate with: $(BIN)/activate"

install:  ## Reinstall the package in editable mode
	$(PY) -m pip install -e . --no-deps

# -----------------------------------------------------------------------------
# M2 - Data
# -----------------------------------------------------------------------------
data:  ## M2: acquire, validate and split the dataset
	$(DV) data

# -----------------------------------------------------------------------------
# M3 - Experimentation
# -----------------------------------------------------------------------------
train:  ## M3: train every configured model arm
	$(DV) train --all

compare:  ## M3: compare runs, apply gates, promote the winner
	$(DV) compare

mlflow:  ## Open the MLflow tracking UI on :5000
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# -----------------------------------------------------------------------------
# M4 - Packaging and serving
# -----------------------------------------------------------------------------
package:  ## M4: promote the gated winner to models/production
	$(DV) package

serve:  ## M4: run the inference API on :8000
	$(DV) serve

benchmark:  ## M4: measure API latency and throughput (needs a running service)
	$(DV) benchmark --n 200 --concurrency 4

# -----------------------------------------------------------------------------
# M5 - Monitoring
# -----------------------------------------------------------------------------
reference:  ## M5: snapshot the training distribution as the drift baseline
	$(DV) reference-stats

drift:  ## M5: run the drift simulation sweep
	$(DV) simulate-drift

monitor:  ## M5: compute drift signals and write the monitoring report
	$(DV) monitor

retrain:  ## M5: evaluate the retraining trigger policy
	-$(DV) check-retrain

# -----------------------------------------------------------------------------
# Whole pipeline
# -----------------------------------------------------------------------------
all: data train compare reference drift monitor retrain  ## Run M2 -> M5 end to end
	@echo ""
	@echo "Pipeline complete. Artifacts:"
	@echo "  reports/validation_report.json    M2 data quality gates"
	@echo "  data/processed/dataset_card.json  M2 dataset version"
	@echo "  reports/model_comparison.md       M3 model comparison"
	@echo "  models/production/model_bundle.pt M4 deployable model"
	@echo "  reports/drift_report.md           M5 drift analysis"
	@echo "  reports/retraining_decision.json  M5 retraining decision"

repro:  ## Rebuild only the stale DVC stages
	$(BIN)/dvc repro

dag:  ## Print the DVC pipeline graph
	$(BIN)/dvc dag

demo:  ## The sequence to run during the recorded demo
	@echo "1. Data pipeline + quality gates"
	$(DV) data
	@echo "2. Tracked experiments"
	$(DV) train --all
	@echo "3. Gated promotion"
	$(DV) compare
	@echo "4. Start the API in another terminal:  make serve"
	@echo "5. Then:  make drift && make monitor && make retrain"

# -----------------------------------------------------------------------------
# Quality
# -----------------------------------------------------------------------------
test:  ## Run the test suite with coverage
	$(PY) -m pytest -v --cov=defectvision --cov-report=term-missing

lint:  ## Lint (see .github/workflows/ci.yml for why format-check is not run)
	$(BIN)/ruff check src tests

format:  ## Auto-fix the lint findings that have safe fixes
	$(BIN)/ruff check --fix src tests

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------
docker:  ## Build the serving image
	docker build -t defectvision:1.0.0 .

docker-run:  ## Run the serving container with the promoted model mounted
	docker run --rm -p 8000:8000 \
	  -v "$(CURDIR)/models/production:/app/models/production:ro" \
	  -v "$(CURDIR)/monitoring:/app/monitoring" \
	  defectvision:1.0.0

# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------
clean:  ## Remove caches and generated reports
	-rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	-rm -rf reports/figures reports/*.json reports/*.jsonl
	-find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-all: clean  ## Also remove data, models, and experiment tracking stores
	-rm -rf data/raw data/interim data/processed models mlruns mlartifacts
	-rm -f mlflow.db monitoring/predictions.db monitoring/reference_stats.json
