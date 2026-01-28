.PHONY: env lint test data drafts drafts-dry-run train run eval clean help

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

data: ## Generate prompt plan for synthetic dataset
	@echo "Generating prompt plan from spec..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.plan_prompts \
		--spec data/specs/dataset_v1_spec.yaml \
		--out_dir data/interim
	@echo "✓ Prompt plan generated!"

drafts: ## Generate draft responses from teacher model
	@echo "Generating draft responses from teacher model..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_drafts \
		--plan data/interim/prompt_plan_v1.jsonl \
		--out_dir data/interim
	@echo "✓ Drafts generated! Check data/interim for outputs."

drafts-dry-run: ## Generate drafts in dry-run mode (no API calls)
	@echo "Generating drafts in dry-run mode..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_drafts \
		--plan data/interim/prompt_plan_v1.jsonl \
		--out_dir data/interim \
		--dry_run
	@echo "✓ Drafts generated (dry-run)! Check data/interim for outputs."

critiques: ## Generate critiques for drafts (quality gate)
	@echo "Generating critiques for drafts..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_critiques \
		--drafts data/interim/drafts_v1.jsonl \
		--out_dir data/interim
	@echo "✓ Critiques generated! Check data/interim/critiques_v1.jsonl"

critiques-dry-run: ## Generate critiques in dry-run mode (no API calls)
	@echo "Generating critiques in dry-run mode..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_critiques \
		--drafts data/interim/drafts_v1.jsonl \
		--out_dir data/interim \
		--dry_run
	@echo "✓ Critiques generated (dry-run)! Check data/interim for outputs."

dataset: ## Generate final training dataset from plans, drafts, and critiques
	@echo "Generating final training dataset..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_dataset_v1 \
		--plan data/interim/prompt_plan_v1.jsonl \
		--drafts data/interim/drafts_v1.jsonl \
		--critiques data/interim/critiques_v1.jsonl \
		--out_dir data/processed \
		--gating_mode lenient \
		--resume
	@echo "✓ Dataset generated! Check data/processed/dataset_v1.jsonl"

dataset-dry-run: ## Generate dataset in dry-run mode (no API calls)
	@echo "Generating dataset in dry-run mode..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_dataset_v1 \
		--plan data/interim/prompt_plan_v1.jsonl \
		--drafts data/interim/drafts_v1.jsonl \
		--critiques data/interim/critiques_v1.jsonl \
		--out_dir data/processed \
		--gating_mode lenient \
		--dry_run
	@echo "✓ Dataset generated (dry-run)! Check data/processed for outputs."

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
