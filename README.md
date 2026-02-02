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
│   ├── benchmarks/v1/       # Held-out test benchmarks
│   ├── processed/           # Cleaned dataset + splits (v1)
│   ├── prompts/teacher/v1/  # Teacher prompt templates
│   └── specs/               # Dataset specifications
├── tests/                   # Test suite (361 tests)
├── runs/                    # Evaluation outputs
├── docs/                    # Documentation
├── .env.example             # Environment variable template
├── Makefile                 # Build commands
└── pyproject.toml           # Package config
```

## Milestones

- **0** Project foundation: repo structure, configs, stubbed pipeline. Base for reproducible eval and data pipelines.
- **1** Baseline evaluation: travel benchmark suites (JSONL), metrics, human-readable report generator. Eval runs produce outputs and provenance under `runs/eval/`.
- **2** Behavioral contracts: envelope + payload JSON schemas (itinerary, checklist, procedure, etc.).
  Strict/lenient parsing and validation wired into eval and CLI; structured error envelopes on parse failure.
- **3** Synthetic data engine: prompt planning from specs → teacher (OpenRouter) for drafts/critiques → gating → `dataset_v1.jsonl`.
  Makefile pipeline for sample, batch, and full runs; optional QA and resume.
- **4** Data quality & splits: near-duplicate detection, quality filters/gates, leakage-free train/val/test splits and SFT prep.
  Fixed eval prompts exported to suites; data layout ready for Lightning.
- **5** Model adaptation: LoRA fine-tuning (PEFT) on Lightning.ai.
  Config-driven training, run provenance, adapter export for inference and quantization.
- **6** Local runtime & quantization: HF→GGUF conversion + quantization (Q4/Q5). Unified CLI (`--runtime hf|local`), llama.cpp backend, local eval (latency + schema compliance).
  See [docs/local_runtime_guide.md](docs/local_runtime_guide.md).
- **7** Iteration cycle v2 (targeted data + training fix). **Goal:** Demonstrate evaluation-driven iteration (the key research signal). **Scope:** Pick top 1–2 failure modes from M6; apply targeted interventions (dataset augmentation for those failure classes, stricter QC rules, small hyperparam tweaks); retrain adapted model v2; re-evaluate on the same benchmark.
  **Outputs:** `data/processed/dataset_v2_*`, `runs/train/*v2*`, `docs/evaluation_report_v2.md`. **DoD:** Demonstrable improvement on targeted failure modes; explanation linking intervention → metric delta.
- **8–10** Further iteration, deployment realism, final polish (planned).

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

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.