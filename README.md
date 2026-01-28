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

## Development Roadmap

```
M0: Foundation    M1: Baseline      M2: Contracts     M3: Data Engine   M4-9: Refinement
✅ Complete      ✅ Complete       ✅ Complete       🔄 In Progress    ⏳ Planned
        └─────────────────────────────────────────────────────────────────────────────→
```

| Milestone | Status | Goal | Key Deliverables |
|-----------|--------|------|------------------|
| **M0** | ✅ | Clean foundation | Repo structure, stable commands, end-to-end pipeline |
| **M1** | ✅ | Baseline eval | Base model report, reference benchmarks (72 examples) |
| **M2** | ✅ | Response contracts | Envelope schema (v0), payload schemas (v1), validation engine |
| **M3** | 🔄 | Synthetic data | Teacher provider, prompt templates, deterministic generation |
| **M4** | ⏳ | Data quality | Deduplication, balancing, clean splits |
| **M5** | ⏳ | Model tuning | LoRA/QLoRA fine-tuning, experiment tracking |
| **M6** | ⏳ | Rigorous eval | Base vs adapted comparison, metrics + examples |
| **M7** | ⏳ | Iteration | Evidence-driven improvements, retrain & re-eval |
| **M8** | ⏳ | Deployment | Quantization, offline packaging, resource docs |
| **M9** | ⏳ | Portfolio | Polish, demo, limitations, safety |

### Current Focus: Milestone 3 — Synthetic Data Engine

**What:** Building a teacher-driven pipeline to generate high-quality travel instruction examples with proper structure, uncertainty handling, and verification guidance.

**Completed:**
- ✅ Prompt templates (v1) for 4 payload types
- ✅ Dataset spec (120 examples, 7 categories, 3 difficulty levels)
- ✅ OpenRouter backend with cost-controlled fallback (2 free → 1 paid)
- ✅ Rate limiting, exponential backoff, retry logic, error typing
- ✅ Environment-based API key management (.env + python-dotenv)
- ✅ 156 tests passing

**In Progress:**
- 🔄 Batch generation CLI (Lesson 3.3)
- 🔄 Response validation & quality checks
- 🔄 Dataset versioning

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
