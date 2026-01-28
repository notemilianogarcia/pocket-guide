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
- Repository scaffolding with clean structure
- Stub CLI with structured outputs
- Evaluation framework and smoke tests
- Configuration system and stable commands

### Milestone 1: Base Model Integration ✅
- Real model inference with Transformers
- Comprehensive benchmark suites (72 examples across 4 categories)
- Automated metrics (parsing, uncertainty detection, performance)
- Report generation with failure analysis
- Hardening for reproducibility and ergonomics

### Milestone 2: Schema Validation & Fine-tuning ✅
**Lessons 2.1-2.4 Complete:**
- ✅ Canonical envelope schema with versioned layout (v0)
- ✅ Content payload schemas v1 (itinerary, checklist, decision_tree, procedure)
- ✅ Parser + validator engine with strict/lenient JSON modes
- ✅ Structured error handling with stable codes for metrics
- ✅ Robust schema loading (importlib.resources) with caching
- ✅ Integration into benchmark pipeline with contract validation
- ✅ Comprehensive test coverage (133 tests pass)

**Next Steps:**
- Base model evaluation and selection
- LoRA/QLoRA fine-tuning pipeline
- Training infrastructure with checkpointing

### Milestone 3: Synthetic Data Generation (In Progress)
**Lessons 3.1-3.2 Complete:**
- ✅ Teacher data pipeline scaffolding
- ✅ Versioned prompt templates (v1) for 4 payload types
- ✅ Dataset specification (120 examples across 7 categories)
- ✅ Deterministic prompt planner CLI (`make data`)
- ✅ Teacher provider interface with OpenRouter backend
- ✅ Model fallback chain (2 free models → 1 paid fallback)
- ✅ Rate limiting, retry logic, and cost control
- ✅ Environment-based API key management (.env support)
- ✅ Comprehensive test coverage (156 tests pass)

**Next Steps:**
- 🔄 Batch generation CLI (Lesson 3.3)
- Response validation and quality checks
- Dataset versioning and storage

### Milestone 4: Advanced Features (Planned)
- Multi-turn conversations
- Context-aware responses
- Enhanced safety and reliability

### Milestone 5: Production Deployment (Planned)
- Optimized inference
- Mobile/offline packaging
- Performance benchmarking

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
