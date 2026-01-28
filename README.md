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
```

### Basic Usage
```bash
make run          # Run inference with default prompt
make eval         # Run evaluation on benchmark suites
make lint         # Code quality checks
```

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
│   ├── eval/                # Evaluation framework
│   └── utils/               # Utilities
├── configs/                 # Configuration files
├── data/benchmarks/v0/      # Benchmark suites (72 examples)
├── tests/                   # Test suite
├── runs/                    # Evaluation outputs
├── docs/                    # Documentation
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

### Milestone 2: Model Selection & Fine-tuning (Planned)
- Base model evaluation and selection
- LoRA/QLoRA fine-tuning pipeline
- Training infrastructure with checkpointing

### Milestone 3: Advanced Features (Planned)
- Multi-turn conversations
- Context-aware responses
- Enhanced safety and reliability

### Milestone 4: Production Deployment (Planned)
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

**Note**: This is Milestone 0 - the CLI currently returns stub responses. Real model inference will be implemented in Milestone 2.
