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
Established clean repository structure with stable commands, configuration files, and placeholders for all pipeline stages. The project is runnable end-to-end in stubbed form, ensuring consistent and reproducible structure for all future work.

### Milestone 1: Baseline Evaluation ✅
Selected and evaluated a single open-source student model on travel-specific benchmarks before any adaptation. Produced base model report exposing hallucinations, structure failures, and uncertainty mishandling—establishing the reference point for all improvements.

### Milestone 2: Behavioral Contracts ✅
Defined valid response format through standard envelope schema (v0) and structured output schemas (v1: itinerary, checklist, procedure, decision_tree). Implemented validation and parsing logic with strict/lenient modes, enabling objective measurement of contract compliance in training and evaluation.

### Milestone 3: Synthetic Data Engine 🔄 (In Progress)

Building a teacher-driven synthetic data pipeline to generate high-quality travel instruction examples. Focus is on consistent demonstrations of good decision-support behavior: proper structure, uncertainty handling, and verification guidance.

**Current Status (Lessons 3.1-3.2 Complete):**

**Data Pipeline Infrastructure:**
- ✅ Versioned prompt templates (v1) for 4 payload types with strict JSON output requirements
- ✅ Dataset specification defining 120 examples across 7 categories, 3 difficulty levels
- ✅ Deterministic prompt planner CLI (`make data`) with seed-based reproducibility
- ✅ Generated prompt plans with statistics and manifest tracking

**Teacher Provider System:**
- ✅ Abstract teacher interface (`TeacherClient`) with request/response dataclasses
- ✅ OpenRouter backend client with OpenAI-compatible API
- ✅ Cost-controlled model fallback chain: 2 free models → 1 paid fallback
- ✅ Rate limiting (15 RPM), exponential backoff with jitter, retry logic
- ✅ Environment-based API key management (.env with python-dotenv)
- ✅ Typed error handling: fail-fast on auth/bad requests, retry on transient errors
- ✅ Comprehensive observability: token usage, latency, fallback tracking

**Technical Features:**
- Dry-run mode for testing without API costs
- `fallback_to_paid` flag to prevent accidental spending
- Smoke test CLI for manual verification
- 156 tests passing (23 new tests for teacher system)

**Next Steps:**
- 🔄 Batch generation CLI to generate all 120 examples (Lesson 3.3)
- Response validation and quality checks
- Dataset versioning and storage

### Milestone 4: Data Quality & Splits (Planned)
Apply deduplication, balancing, rejection filters, and leakage prevention to synthetic dataset. Create clean held-out benchmark split ensuring proper separation between training and evaluation data.

### Milestone 5: Model Adaptation (Planned)
Fine-tune student model using LoRA/QLoRA on cleaned synthetic dataset. Track experiments, save artifacts, and document configuration choices in training report.

### Milestone 6: Rigorous Evaluation (Planned)
Compare base and adapted models using objective metrics (format compliance, uncertainty signaling, constraint satisfaction). Produce evaluation report with curated qualitative examples showing improvements and remaining failures.

### Milestone 7: Evidence-Driven Iteration (Planned)
Apply targeted fixes based on failure analysis (data augmentation, stricter quality controls). Retrain and re-evaluate to demonstrate evidence-based improvement methodology.

### Milestone 8: Deployment Realism (Planned)
Quantize adapted model and package for local, offline inference. Document memory usage, latency, and runtime constraints proving usability on limited hardware.

### Milestone 9: Portfolio Finalization (Planned)
Polish README, add demo, summarize results, document limitations and safety considerations. Complete the project as a hireable artifact demonstrating ML research thinking and engineering execution.

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
