.PHONY: env lint test data drafts drafts-sample drafts-dry-run drafts-batch critiques critiques-sample critiques-dry-run critiques-batch dataset dataset-sample dataset-dry-run dataset-batch pipeline-batch split prepare-sft data-full pipeline-sample debug-sample regate-overconfident clean-data train train-run train-dry-run run eval clean help

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

# Number of samples for sample/smoke pipeline (override: make pipeline-sample SAMPLE_N=5)
SAMPLE_N ?= 5
# Dedicated dirs for sample pipeline (avoids mixing with full-run data)
SAMPLE_INTERIM ?= data/interim/sample
SAMPLE_PROCESSED ?= data/processed/sample
# Pass --debug to drafts and dataset steps when DEBUG=1 (e.g. make pipeline-sample DEBUG=1)
DEBUG_ARGS := $(if $(filter 1,$(DEBUG)),--debug,)

# Batch pipeline: process next BATCH_SIZE items with --resume (e.g. make pipeline-batch BATCH_SIZE=20)
BATCH_SIZE ?= 20
# Limit = current count + BATCH_SIZE so we process the *next* BATCH_SIZE (resume skips existing)
DRAFT_LIMIT := $(shell n=0; [ -f data/interim/drafts_v1.jsonl ] && n=$$(wc -l < data/interim/drafts_v1.jsonl); echo $$((n + $(BATCH_SIZE))))
CRITIQUE_LIMIT := $(shell n=0; [ -f data/interim/critiques_v1.jsonl ] && n=$$(wc -l < data/interim/critiques_v1.jsonl); echo $$((n + $(BATCH_SIZE))))

drafts: ## Generate draft responses from teacher model
	@echo "Generating draft responses from teacher model..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_drafts \
		--plan data/interim/prompt_plan_v1.jsonl \
		--out_dir data/interim
	@echo "✓ Drafts generated! Check data/interim for outputs."

drafts-sample: ## Generate drafts for first N prompts only (real API; default N=3). Requires: make data first.
	@echo "Generating drafts for first $(SAMPLE_N) prompts (real run)..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_drafts \
		--plan data/interim/prompt_plan_v1.jsonl \
		--out_dir data/interim \
		--limit $(SAMPLE_N)
	@echo "✓ Drafts (sample): data/interim/drafts_v1.jsonl"

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

critiques-sample: ## Generate critiques for first N drafts only (default N=3)
	@echo "Generating critiques for first $(SAMPLE_N) drafts..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_critiques \
		--drafts data/interim/drafts_v1.jsonl \
		--out_dir data/interim \
		--limit $(SAMPLE_N)
	@echo "✓ Critiques (sample): data/interim/critiques_v1.jsonl"

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

dataset-sample: ## Generate dataset for first N samples only (default N=3)
	@echo "Generating dataset for first $(SAMPLE_N) samples..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_dataset_v1 \
		--plan data/interim/prompt_plan_v1.jsonl \
		--drafts data/interim/drafts_v1.jsonl \
		--critiques data/interim/critiques_v1.jsonl \
		--out_dir data/processed \
		--gating_mode lenient \
		--resume \
		--limit $(SAMPLE_N)
	@echo "✓ Dataset (sample): data/processed/dataset_v1.jsonl"

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

# Batch targets: process next BATCH_SIZE (default 20) with --resume. Run pipeline-batch repeatedly.
drafts-batch: ## Generate next BATCH_SIZE drafts (resume). Requires: make data first. Use BATCH_SIZE=20 (default).
	@echo "Generating next $(BATCH_SIZE) drafts (limit=$(DRAFT_LIMIT), resume)..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_drafts \
		--plan data/interim/prompt_plan_v1.jsonl \
		--out_dir data/interim \
		--limit $(DRAFT_LIMIT) \
		--resume
	@echo "✓ Drafts batch done. Check data/interim/drafts_v1.jsonl"

critiques-batch: ## Generate next BATCH_SIZE critiques (resume). Use BATCH_SIZE=20 (default).
	@echo "Generating next $(BATCH_SIZE) critiques (limit=$(CRITIQUE_LIMIT), resume)..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_critiques \
		--drafts data/interim/drafts_v1.jsonl \
		--out_dir data/interim \
		--limit $(CRITIQUE_LIMIT) \
		--resume
	@echo "✓ Critiques batch done. Check data/interim/critiques_v1.jsonl"

dataset-batch: ## Process next BATCH_SIZE joined samples into dataset (resume). Use BATCH_SIZE=20 (default).
	@echo "Processing next $(BATCH_SIZE) samples into dataset (resume)..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_dataset_v1 \
		--plan data/interim/prompt_plan_v1.jsonl \
		--drafts data/interim/drafts_v1.jsonl \
		--critiques data/interim/critiques_v1.jsonl \
		--out_dir data/processed \
		--gating_mode lenient \
		--limit $(BATCH_SIZE) \
		--resume
	@echo "✓ Dataset batch done. Check data/processed/dataset_v1.jsonl"

# One batch: data (if needed) -> drafts-batch -> critiques-batch -> dataset-batch. Run repeatedly; when done run split and prepare-sft.
pipeline-batch: ## Run one batch (next BATCH_SIZE items, default 20). Resume-safe. Then run split + prepare-sft when all batches done.
	@echo "[pipeline-batch] BATCH_SIZE=$(BATCH_SIZE). Ensure configs/teacher.yaml dry_run: false and OPENROUTER_API_KEY set."
	@$(MAKE) data
	@$(MAKE) drafts-batch BATCH_SIZE=$(BATCH_SIZE)
	@$(MAKE) critiques-batch BATCH_SIZE=$(BATCH_SIZE)
	@$(MAKE) dataset-batch BATCH_SIZE=$(BATCH_SIZE)
	@echo "✓ Batch complete. Run 'make pipeline-batch' again for next $(BATCH_SIZE), or 'make split && make prepare-sft' when done."

split: ## Split dataset into train/val/test (requires data/processed/dataset_v1.jsonl)
	@echo "Splitting dataset into train/val/test..."
	@$(VENV_PYTHON) -m pocketguide.data_generation.split_dataset_v1 \
		--in_path data/processed/dataset_v1.jsonl \
		--out_dir data/processed/splits/v1 \
		--seed 42 \
		--train_frac 0.8 --val_frac 0.1 --test_frac 0.1
	@echo "✓ Splits written to data/processed/splits/v1/"

prepare-sft: ## Convert splits to SFT format (train_sft.jsonl, val_sft.jsonl, fixed suite)
	@echo "Preparing SFT datasets and fixed eval suite..."
	@$(VENV_PYTHON) -m pocketguide.train.prepare_sft_data \
		--splits_dir data/processed/splits/v1 \
		--out_dir data/processed/sft/v1 \
		--seed 42 \
		--fixed_prompts_out eval/suites/fixed20_v1.jsonl
	@echo "✓ SFT data in data/processed/sft/v1/ — use for training (configs/train_lora.yaml)"

data-full: data drafts critiques dataset split prepare-sft ## Run full pipeline: plan → drafts → critiques → dataset → split → SFT (API calls)
	@echo "✓ Full data pipeline complete. Train with: make train-run (or Lightning)."

# Small real run: uses dedicated dirs (SAMPLE_INTERIM, SAMPLE_PROCESSED) so it does not mix with full-run data.
# For real API calls: set runtime.dry_run: false in configs/teacher.yaml and OPENROUTER_API_KEY in .env
pipeline-sample: ## Run pipeline on first N samples only (SAMPLE_N=3). Use DEBUG=1 for per-sample failure/rejection logs.
	@echo "[pipeline-sample] Ensure configs/teacher.yaml has runtime.dry_run: false and OPENROUTER_API_KEY is set for real API calls."
	@echo "[pipeline-sample] Running: data -> drafts ($(SAMPLE_N)) -> critiques ($(SAMPLE_N)) -> dataset ($(SAMPLE_N)) -> split -> prepare-sft"
	@$(MAKE) data
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_drafts \
		--plan data/interim/prompt_plan_v1.jsonl \
		--out_dir $(SAMPLE_INTERIM) \
		--limit $(SAMPLE_N) $(DEBUG_ARGS)
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_critiques \
		--drafts $(SAMPLE_INTERIM)/drafts_v1.jsonl \
		--out_dir $(SAMPLE_INTERIM) \
		--limit $(SAMPLE_N)
	@$(VENV_PYTHON) -m pocketguide.data_generation.generate_dataset_v1 \
		--plan data/interim/prompt_plan_v1.jsonl \
		--drafts $(SAMPLE_INTERIM)/drafts_v1.jsonl \
		--critiques $(SAMPLE_INTERIM)/critiques_v1.jsonl \
		--out_dir $(SAMPLE_PROCESSED) \
		--gating_mode lenient \
		--limit $(SAMPLE_N) $(DEBUG_ARGS)
	@$(VENV_PYTHON) -m pocketguide.data_generation.split_dataset_v1 \
		--in_path $(SAMPLE_PROCESSED)/dataset_v1.jsonl \
		--out_dir $(SAMPLE_PROCESSED)/splits/v1 \
		--seed 42 --train_frac 0.8 --val_frac 0.1 --test_frac 0.1
	@$(VENV_PYTHON) -m pocketguide.train.prepare_sft_data \
		--splits_dir $(SAMPLE_PROCESSED)/splits/v1 \
		--out_dir $(SAMPLE_PROCESSED)/sft/v1 \
		--seed 42 \
		--fixed_prompts_out eval/suites/fixed20_sample_v1.jsonl
	@echo "✓ Sample pipeline complete ($(SAMPLE_N) samples). Check $(SAMPLE_PROCESSED)/dataset_v1.jsonl and $(SAMPLE_PROCESSED)/sft/v1/"

debug-sample: ## Inspect last sample run: drafts (validation errors + output prefix), rejected refinement/gates, stats. Run after pipeline-sample.
	@$(VENV_PYTHON) scripts/inspect_pipeline.py --sample

regate-overconfident: ## Re-gate rejections that failed only on overconfident_time_sensitive and append newly accepted to dataset_v1.jsonl. Run after fixing the overconfidence gate.
	@$(VENV_PYTHON) scripts/regate_overconfident_rejections.py
	@echo "✓ Re-gate complete. Check data/processed/dataset_v1.jsonl"

train: ## Alias for train-run
	@$(MAKE) train-run

train-run: ## Run LoRA training (configs/train_lora.yaml); use --dry_run to validate only
	@echo "Starting training (configs/train_lora.yaml)..."
	@$(VENV_PYTHON) -m pocketguide.train.train --config configs/train_lora.yaml
	@echo "✓ Training run complete. Check runs/train/ for outputs."

train-dry-run: ## Validate config and datasets without loading model
	@$(VENV_PYTHON) -m pocketguide.train.train --config configs/train_lora.yaml --dry_run
	@echo "✓ Dry run OK."

run-samples: ## Run base + finetuned sample generation (Lesson 5.4). Requires RUN_DIR=runs/train/<run_id>
	@test -n "$(RUN_DIR)" || (echo "Usage: make run-samples RUN_DIR=runs/train/<run_id>" && exit 1)
	@prompts="$${PROMPTS:-eval/suites/fixed20_v1.jsonl}"; \
	$(VENV_PYTHON) -m pocketguide.train.run_samples \
		--run_dir "$(RUN_DIR)" \
		--prompts "$$prompts"
	@echo "✓ Samples and comparison_metrics written under $(RUN_DIR)/samples/"

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

clean-data: ## Remove all pipeline-generated data (interim, processed, sample, fixed20 eval suites). Keeps data/specs, data/prompts, data/benchmarks.
	@rm -rf data/interim/sample data/interim/prompt_plan_v1.jsonl data/interim/prompt_plan_v1_stats.json data/interim/prompt_plan_v1_manifest.json
	@rm -rf data/interim/drafts_v1.jsonl data/interim/drafts_v1_manifest.json data/interim/drafts_v1_stats.json
	@rm -rf data/interim/critiques_v1.jsonl data/interim/critiques_v1_manifest.json data/interim/critiques_v1_stats.json
	@rm -rf data/processed/sample data/processed/splits data/processed/sft
	@rm -f data/processed/dataset_v1.jsonl data/processed/dataset_v1_rejected.jsonl data/processed/dataset_v1_manifest.json data/processed/dataset_v1_stats.json
	@rm -f eval/suites/fixed20_sample_v1.jsonl eval/suites/fixed20_v1.jsonl
	@echo "✓ Data cleaned (interim, processed, sample, fixed20 suites)."

clean-all: clean ## Remove environment and all generated files
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)
	@echo "✓ Full cleanup complete!"
