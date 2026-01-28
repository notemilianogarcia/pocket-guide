.PHONY: env lint test data train run eval clean help

# Default target
.DEFAULT_GOAL := help

# Python interpreter (will be set after env is created)
PYTHON := python3
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

help: ## Show this help message
	@echo "PocketGuide - Makefile targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

env: ## Create/update local Python environment
	@echo "Setting up Python environment..."
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
	fi
	@echo "Installing package in editable mode with dev dependencies..."
	@$(VENV_PIP) install --upgrade pip
	@$(VENV_PIP) install -e ".[dev]"
	@echo "✓ Environment ready! Activate with: source $(VENV)/bin/activate"

lint: ## Run code linting with ruff
	@echo "Running ruff linter..."
	@$(VENV_PYTHON) -m ruff check src/ tests/
	@echo "Running ruff formatter check..."
	@$(VENV_PYTHON) -m ruff format --check src/ tests/
	@echo "✓ Lint checks passed!"

format: ## Auto-format code with ruff
	@echo "Formatting code with ruff..."
	@$(VENV_PYTHON) -m ruff format src/ tests/
	@$(VENV_PYTHON) -m ruff check --fix src/ tests/
	@echo "✓ Code formatted!"

test: ## Run pytest test suite
	@echo "Running tests with pytest..."
	@$(VENV_PYTHON) -m pytest tests/ -v
	@echo "✓ Tests passed!"

data: ## Placeholder for data preparation (Milestone 1+)
	@echo "⚠️  Data preparation not yet implemented (Milestone 1+)"
	@echo "This target will download and prepare datasets in future milestones."
	@exit 0

train: ## Placeholder for model training (Milestone 2+)
	@echo "⚠️  Model training not yet implemented (Milestone 2+)"
	@echo "This target will fine-tune the model in future milestones."
	@exit 0

run: ## Run CLI inference with default prompt
	@echo "Running PocketGuide CLI stub..."
	@echo ""
	@$(VENV_PYTHON) -m pocketguide.inference.cli --prompt "What documents do I need to travel from the US to Canada?"

eval: ## Run evaluation benchmark on v0 benchmark suites
	@echo "Running evaluation benchmark on v0 benchmark suites..."
	@echo ""
	@$(VENV_PYTHON) -m pocketguide.eval.benchmark --config configs/eval.yaml --suite_dir data/benchmarks/v0
	@echo ""
	@echo "✓ Evaluation complete! Check runs/eval/ for results."

clean: ## Remove generated files and caches
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cleanup complete!"

clean-all: clean ## Remove environment and all generated files
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)
	@echo "✓ Full cleanup complete!"
