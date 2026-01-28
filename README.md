# PocketGuide

**Offline-capable travel assistant LLM with evaluation-first design**

PocketGuide is a specialized language model assistant for travelers, providing reliable guidance on visa requirements, border regulations, local customs, budgeting, and itinerary planning—without internet connectivity. Built with rigorous evaluation, deterministic outputs, and robust error handling.

## Key Features

- **Offline-first**: No external API dependencies
- **Evaluation-driven**: Comprehensive benchmark suites for development
- **Structured responses**: Consistent output format with required sections
- **Deterministic**: Reproducible outputs for testing and reliability
- **Production-ready**: Clean architecture with stable commands

## Quick Start

### Prerequisites
- Python 3.11+
- Make

### Setup
```bash
git clone <repository-url>
cd pocket-guide
make env
source .venv/bin/activate
make test

# Set up API keys for data generation
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### Basic Usage
```bash
make run          # Run inference with default prompt
make eval         # Run evaluation on benchmark suites
make lint         # Code quality checks
make data         # Generate synthetic training data (requires API key)
```

### Teacher Model (Synthetic Data Generation)
```bash
# Test in dry-run mode (no API calls)
python -m pocketguide.teachers.smoke --dry-run

# Real API calls (uses .env for API key)
python -m pocketguide.teachers.smoke --real --prompt "Best cafes in Rome?"
```

See [docs/environment-setup.md](docs/environment-setup.md) for API key configuration.

## Usage

### CLI Inference
```bash
python -m pocketguide.inference.cli --prompt "What are visa requirements for Japan?"
python -m pocketguide.inference.cli --prompt "Budget for Thailand trip" --format json
```

### Evaluation
```bash
# Run full benchmark suite
make eval

# Run specific suite
python -m pocketguide.eval.benchmark --suite_dir data/benchmarks/v0

# Smoke test with real model
python -m pocketguide.eval.benchmark --smoke_infer --prompt "Explain Japan visa policy"
```

Results saved to `runs/eval/<run_id>/` with outputs, metrics, and provenance metadata.

### Reports
Generate human-readable evaluation reports:
```bash
python -m pocketguide.eval.report_base_v0 --run_dir runs/eval/<timestamp>
```

Includes metrics summary, failure analysis, and curated examples.

## Project Structure

```
pocket-guide/
├── src/pocketguide/          # Core package
│   ├── inference/           # Model inference and CLI
│   ├── eval/                # Evaluation framework (benchmark, metrics, parsing, reporting)
│   ├── teachers/            # Teacher models for synthetic data generation
│   ├── data_generation/     # Prompt planning and dataset specs
│   ├── data/schemas/        # JSON schemas (v0 envelope, v1 payloads)
│   └── utils/               # Utilities (rate limiting, etc.)
├── configs/                 # Configuration files (eval, teacher)
├── data/
│   ├── benchmarks/v0/       # Benchmark suites (72 examples)
│   ├── prompts/teacher/v1/  # Teacher prompt templates
│   └── specs/               # Dataset specifications
├── tests/                   # Test suite (156 tests)
├── runs/                    # Evaluation outputs
├── docs/                    # Documentation
├── .env.example             # Environment variable template
├── Makefile                 # Build commands
└── pyproject.toml           # Package config
```

## Milestones

### Milestone 0: Project Foundation ✅
Clean repository structure with stable commands, configuration files, and end-to-end pipeline placeholders. Runnable in stubbed form.

### Milestone 1: Baseline Evaluation ✅
Evaluated open-source student model on travel benchmarks pre-adaptation. Base report establishes reference point for all improvements.

### Milestone 2: Behavioral Contracts ✅
- Standard envelope schema (v0) + structured output schemas (v1)
- Validation/parsing engine (strict/lenient modes)
- Objective contract compliance measurement

### Milestone 3: Synthetic Data Engine 🔄 (In Progress)

**Goal:** Teacher-driven pipeline generating high-quality travel instruction examples with proper structure, uncertainty handling, and verification guidance.

**Completed (Lessons 3.1-3.2):**

*Data Pipeline:*
- Versioned prompt templates (4 payload types)
- Dataset spec (120 examples, 7 categories, 3 difficulty levels)
- Deterministic prompt planner CLI (`make data`)

*Teacher Provider:*
- OpenRouter backend with cost-controlled fallback chain (2 free → 1 paid)
- Rate limiting (15 RPM), exponential backoff, retry logic
- Environment-based API keys (.env support)
- Typed error handling (fail-fast vs retry)
- Full observability (tokens, latency, fallback tracking)

*Features:* Dry-run mode, `fallback_to_paid` flag, 156 tests passing

**Next:** Batch generation CLI (Lesson 3.3), validation, versioning

### Milestone 4: Data Quality & Splits (Planned)
Deduplication, balancing, rejection filters, leakage prevention. Clean held-out benchmark split.

### Milestone 5: Model Adaptation (Planned)
LoRA/QLoRA fine-tuning on cleaned synthetic dataset. Experiment tracking and training report.

### Milestone 6: Rigorous Evaluation (Planned)
Base vs adapted model comparison. Objective metrics + curated qualitative examples.

### Milestone 7: Evidence-Driven Iteration (Planned)
Targeted fixes based on failure analysis. Retrain and re-evaluate.

### Milestone 8: Deployment Realism (Planned)
Quantize model, package for local/offline inference. Document resource constraints.

### Milestone 9: Portfolio Finalization (Planned)
Polish README, demo, results summary, limitations, safety considerations.

## Development

### Testing
```bash
make test                    # Run all tests
pytest tests/ --cov=pocketguide  # With coverage
```

### Code Quality
```bash
make lint                    # Check linting
make format                  # Auto-format
```

### Adding Benchmarks
Create JSONL files in `data/benchmarks/v0/`:
```jsonl
{"id": "example_001", "prompt": "Travel question here"}
```

Run evaluation:
```bash
python -m pocketguide.eval.benchmark --suite_dir data/benchmarks/v0
```

## Contributing

- Type hints and Google-style docstrings
- Comprehensive test coverage
- Ruff for linting and formatting

## License

*(To be determined)*
