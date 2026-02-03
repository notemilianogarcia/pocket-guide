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

Phased build: evaluation and contracts first, then synthetic data and training, then iteration and local deployment.

---

### Completed

| # | Milestone | What we built |
|---|-----------|----------------|
| **0** | **Scaffolding & conventions** | Repo layout (`src/`, `configs/`, `scripts/`, `docs/`, `tests/`, `runs/`), `pyproject.toml`, Makefile (`env`, `lint`, `test`, `data`, `train`, `eval`, `run`), minimal CLI and eval harness → outputs under `runs/eval/`. |
| **1** | **Baseline evaluation (v0)** | Travel benchmark suites (JSONL), metrics (parse, schema, latency), human-readable report. Establishes how good the base model is before adaptation. |
| **2** | **Output contracts & schemas** | Canonical JSON schemas (itinerary, checklist, procedure, decision_tree) + response envelope (summary, assumptions, next_steps, verification_steps). Strict/lenient parsing, validation in eval and CLI, structured error envelopes on failure. |
| **3** | **Teacher data generation (v1)** | Prompt planning from specs → teacher (OpenRouter) for drafts → critiques → gating → `dataset_v1.jsonl`. Makefile pipeline (sample, batch, full), optional QA and resume. |
| **4** | **Data QA & splits** | Near-duplicate detection, quality filters/gates, leakage-free train/val/test splits, SFT prep. Fixed eval prompts exported to suites; data ready for Lightning. |
| **5** | **Fine-tuning (v1)** | LoRA (PEFT) on Lightning.ai. Config-driven training, run provenance, adapter export for inference and quantization. |
| **6** | **Local runtime & quantization** | HF→GGUF conversion + Q4/Q5 quantization. Unified CLI (hf and local runtimes), llama.cpp backend, local eval (latency + schema compliance). See [docs/local_runtime_guide.md](docs/local_runtime_guide.md). |
| **7** | **Iteration cycle (v2–v4)** | Dataset v2 (hard prompts, synthetic examples, stricter QC) → v2 (LR tweak) → v3 (5 epochs, max_seq_len 2048, larger LoRA) → v4 (8 epochs, SFT prompt/payload normalization). Artifacts: `data/processed/dataset_v2_*`, `runs/train/*v2*`, `*v3*`, `*v4*`, [evaluation_report_v2.md](docs/evaluation_report_v2.md). Parse success 100% (v3/v4), required_field improved; schema_valid 0% (see report). |

---

### Next

| # | Milestone | Focus |
|---|-----------|--------|
| **8** | Quantization & local packaging | GGUF export, one-command local run, inference guide (RAM, latency, example prompts). |
| **9** | Portfolio finalization | README architecture, results summary, demo GIF, limitations & safety, final report. |
| **10** | Further iteration | Deployment realism, polish. |

### Results (adapter evaluation)

Adapter runs are evaluated on a fixed prompt suite (`eval/suites/fixed20_v1.jsonl`, n=20) with **base** (no adapter) vs **finetuned** (LoRA) comparison. Metrics: `parse_success_rate`, `schema_valid_rate`, `required_field_presence_rate`, `uncertainty_marker_presence_rate`.

| Run | parse_success_rate | schema_valid_rate | required_field_presence_rate | Notes |
|-----|--------------------|-------------------|------------------------------|--------|
| Base | 0.85 | 0.05 | 0.00 | No adapter |
| v1 FT | 0.80 | 0.00 | 0.00 | 1 epoch, max_seq_len 1024 |
| v2 FT | 0.85 | 0.00 | 0.00 | LR 1.5e-4, 1 epoch, 1024, dataset v2 |
| v3 FT | 1.00 | 0.00 | 0.10 | 5 epochs, max_seq_len 2048, larger LoRA |
| v4 FT | 1.00 | 0.00 | 0.20 | 8 epochs, SFT prompt + payload normalization |

Full schema validity (envelope + payload) stayed 0% across finetuned runs; parse success reached 100% (v3/v4) and required-field presence improved to 0.20 (v4). See [docs/evaluation_report_v2.md](docs/evaluation_report_v2.md) for the iteration story and [docs/schema_validity_analysis.md](docs/schema_validity_analysis.md) for an evidence-based analysis of why (truncation, payload keys/shape) and what would need to improve.

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