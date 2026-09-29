# Mental Health Score Predictor — Makefile
# All commands are designed to run from the repository root.
# On Windows, use `python` (or py -3) instead of python3.

PYTHON     := python
BACKEND    := backend
ML         := ml

.PHONY: help install train run-backend test lint clean all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install all Python dependencies
	$(PYTHON) -m pip install -r $(BACKEND)/requirements-dev.txt

generate-data:  ## Generate synthetic training data
	$(PYTHON) scripts/generate_synthetic_data.py

train:  ## Run full ML training pipeline (generates data if needed, trains, evaluates, saves model)
	$(PYTHON) ml/train.py

run-backend:  ## Start the FastAPI backend server (requires trained model)
	$(PYTHON) -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

test:  ## Run all tests with coverage report
	$(PYTHON) -m pytest backend/tests/ \
		--cov=backend/app \
		--cov=ml \
		--cov-report=term-missing \
		--cov-report=html:docs/coverage \
		-v

lint:  ## Run ruff linter on ml/ and backend/
	$(PYTHON) -m ruff check ml/ backend/app/ --fix

format:  ## Format code with ruff
	$(PYTHON) -m ruff format ml/ backend/app/

all: install train test  ## Full pipeline: install → train → test

clean:  ## Remove generated artifacts and caches
	Remove-Item -Recurse -Force -ErrorAction SilentlyContinue models/pipeline.joblib, models/metrics.json, models/metadata.json, data/processed/, docs/figures/, .pytest_cache/, __pycache__
