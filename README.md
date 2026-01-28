# PocketGuide

**Offline-capable travel assistant LLM with evaluation-first design**

PocketGuide is a specialized language model assistant for travelers, designed to provide reliable guidance on visa requirements, border regulations, local customs, budgeting, and itinerary planning—without requiring internet connectivity. The project emphasizes rigorous evaluation, deterministic outputs, and operation under uncertainty.

## Key Features

- **Offline-first**: No external API calls; designed for use without internet access
- **Evaluation-driven**: Comprehensive benchmark suites guide development
- **Structured responses**: Consistent output format with required sections (Summary, Assumptions, Next steps)
- **Deterministic**: Reproducible outputs for testing and reliability
- **Production-ready architecture**: Clean separation of concerns, stable commands

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Make (standard on macOS/Linux)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd pocket-guide

# Create and activate environment
make env
source .venv/bin/activate

# Run tests to verify setup
make test
```

### Basic Usage

```bash
# Run inference with default prompt
make run

# Run evaluation on smoke test suite
make eval

# Run linting
make lint
```

### Running CLI Directly

```bash
# After activating the environment
python -m pocketguide.inference.cli --prompt "What are visa requirements for Japan?"

# JSON output format
python -m pocketguide.inference.cli --prompt "Budget for Thailand trip" --format json
```

### Running Evaluation

```bash
# Run benchmark on v0 suite directory (72 examples across 4 suites) - uses stub inference
make eval

# Run on specific suite directory
python -m pocketguide.eval.benchmark --suite_dir data/benchmarks/v0

# Run with explicit run ID for reproducibility (useful for tests or CI/CD)
python -m pocketguide.eval.benchmark --suite_dir data/benchmarks/v0 --run_id 20260128_143052

# Run smoke inference with real model (single prompt end-to-end test)
python -m pocketguide.eval.benchmark --smoke_infer --prompt "What are visa requirements for Japan?"

# Legacy: run on single JSONL file
python -m pocketguide.eval.benchmark \
  --config configs/eval.yaml \
  --suite eval/suites/smoke.jsonl
```

Results are saved to timestamped directories under `runs/eval/<run_id>/` with:
- `base_model_outputs.jsonl`: Model outputs for each example with per-example checks
- `metrics.json`: Aggregate metrics (parse rates, uncertainty markers, latency stats)
- `meta.json`: Comprehensive provenance (configs, versions, git commit, environment)
- `meta.json`: Comprehensive metadata for reproducibility
- `predictions.jsonl`: Legacy format (single suite mode)

### Generating Evaluation Reports

After running evaluation, generate a human-readable markdown report:

```bash
# Generate report from latest eval run
python -m pocketguide.eval.report_base_v0 --run_dir runs/eval/<timestamp>

# With custom output path
python -m pocketguide.eval.report_base_v0 \
  --run_dir runs/eval/20260128_001943 \
  --output docs/custom_report.md

# With manual failure ID overrides
python -m pocketguide.eval.report_base_v0 \
  --run_dir runs/eval/20260128_001943 \
  --overrides configs/report.yaml
```

The report (`docs/evaluation_report_base_v0.md`) includes:
- Experimental setup (model, device, generation config)
- Metrics summary table (overall + per-suite breakdown)
- Failure taxonomy v0 (7 categories)
- 10 curated failure examples (auto-selected by severity + diversity)
- Summary and next steps

**Manual Override**: Create `configs/report.yaml` to specify failure IDs:
```yaml
curated_failure_ids:
  - "fmt_001"
  - "uncertainty_007"
  - "safety_003"
```

**Metrics Output Structure:**
```json
{
  "overall": {
    "n": 72,
    "strict_json_parse_rate": 0.0,
    "lenient_json_parse_rate": 0.85,
    "required_fields_rate": 0.67,
    "assumptions_marker_rate": 0.45,
    "verification_marker_rate": 0.30,
    "clarifying_questions_rate": 0.12,
    "avg_latency_s": 1.5,
    "p50_latency_s": 1.2,
    "p90_latency_s": 2.8
  },
  "by_suite": {
    "format": { "n": 18, ... },
    "travel_tasks": { "n": 24, ... }
  },
  "definitions": { ... }
}
```

To run evaluation or smoke inference with a real model, update the model identity in [configs/eval.yaml](configs/eval.yaml):

```yaml
model:
  id: "your-model-id"  # HuggingFace model ID (e.g., "gpt2") or local path
  revision: "main"     # Git revision/tag/branch, null = latest

# Hardware and runtime
device: "cpu"          # "cpu", "cuda", or "mps" (M1/M2 Macs)
dtype: "float32"       # "float32", "float16", or "bfloat16"

# Reproducibility
seed: 42

# Generation parameters
gen:
  max_new_tokens: 256
  do_sample: false     # true for sampling, false for greedy (deterministic)
  temperature: 0.0
  top_p: 1.0
  repetition_penalty: 1.0
```

**Smoke Inference**: Test end-to-end model inference with a single prompt. This validates that model loading, generation, and metrics tracking work correctly:

```bash
# Run with default prompt
python -m pocketguide.eval.benchmark --smoke_infer

# Custom prompt
python -m pocketguide.eval.benchmark --smoke_infer --prompt "Explain Japan visa policy"
```

Output includes:
- Generated text
- Token usage (prompt_tokens, completion_tokens, total_tokens)
- Timing metrics (latency_s, tokens_per_s)

The evaluation framework captures comprehensive metadata for reproducibility, including model configuration, generation parameters, device/dtype settings, Python version, installed packages, and git commit hash.

Example model configuration in `meta.json`:

```yaml
id: "your-model-id"  # HuggingFace model ID or local path
revision: "main"     # Git revision/tag/branch
```

## Project Structure

```
pocket-guide/
├── src/pocketguide/          # Main package source
│   ├── inference/            # Inference engine and CLI
│   │   ├── cli.py           # Command-line interface (stub)
│   │   └── base_model.py    # Real Transformers inference
│   ├── eval/                # Evaluation framework
│   │   └── benchmark.py     # Benchmark runner (stub + Transformers)
│   └── __init__.py          # Package init
├── configs/                 # Configuration files
│   └── eval.yaml           # Evaluation settings (model, seed, gen params, device/dtype)
├── data/benchmarks/v0/      # Benchmark suites v0 (72 examples)
│   ├── format.jsonl        # JSON formatting tests (18 ex.)
│   ├── travel_tasks.jsonl  # Travel planning tests (24 ex.)
│   ├── uncertainty.jsonl   # Ambiguity handling tests (15 ex.)
│   ├── safety.jsonl        # Safety/emergency guidance tests (15 ex.)
│   └── README.md           # Suite documentation
├── eval/suites/            # Legacy evaluation test suites
│   └── smoke.jsonl         # Smoke test suite (5 examples)
├── tests/                  # Test suite
│   ├── test_smoke_cli.py   # CLI smoke tests (4 tests)
│   ├── test_smoke_eval.py  # Evaluation runner tests (6 tests)
│   └── test_inference_base_model.py  # Transformers integration tests (7 tests)
├── runs/                   # Timestamped run outputs (gitignored)
├── scripts/                # Utility scripts (future use)
├── docs/                   # Documentation (future use)
├── Makefile               # Stable command interface (11 targets)
├── pyproject.toml         # Package configuration
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Make Targets

All primary workflows are accessed through stable Make targets:

- `make env` - Create/update local Python environment
- `make lint` - Run code linting (ruff)
- `make format` - Auto-format code
- `make test` - Run pytest test suite
- `make run` - Run CLI with default prompt
- `make eval` - Run evaluation on smoke suite
- `make data` - *(Placeholder for Milestone 1+)* Data preparation
- `make train` - *(Placeholder for Milestone 2+)* Model fine-tuning
- `make clean` - Remove caches and generated files
- `make help` - Show all available targets

## Milestone Roadmap

### Milestone 0: Project Scaffolding & Conventions ✅
- Clean repository structure with src-layout packaging
- Stub CLI with deterministic structured outputs
- Evaluation framework with smoke test suite
- Stable Make commands and passing tests
- Configuration system with YAML files

### Milestone 1: Base Model Integration ✅ (Current)
**Lesson 1.1**: Base model identity + reproducible eval config scaffolding
- Enhanced eval.yaml with model ID, revision, device/dtype, generation parameters
- Comprehensive metadata capture (model config, environment, package versions)
- Reproducibility tracking

**Lesson 1.2**: Real inference path (Transformers)
- `src/pocketguide/inference/base_model.py` with model loading and generation
- Deterministic generation with seed management
- Token counting and latency metrics (usage + timing)
- Smoke inference mode for single-prompt end-to-end testing
- Mock-based tests that don't download large models

**Lesson 1.3**: Comprehensive benchmark suites v0 ✅
- 4 benchmark suites under data/benchmarks/v0/:
  - `format.jsonl` (18 ex.): Strict JSON output formatting tests
  - `travel_tasks.jsonl` (24 ex.): Complex travel planning with constraints
  - `uncertainty.jsonl` (15 ex.): Ambiguous prompts requiring assumption-stating
  - `safety.jsonl` (15 ex.): Emergency and security guidance
- Total: 72 examples across 4 suites
- Suite directory support in benchmark runner (--suite_dir flag)
- Output to base_model_outputs.jsonl with id, suite, prompt, output_text, usage, timing, model, seed, gen
- Tests for multi-suite iteration

**Lesson 1.4**: Metrics v0 (automated quality checks) ✅
- `src/pocketguide/eval/metrics_v0.py` with automated metrics computation
- JSON parsing metrics:
  - Strict parse rate (entire output must be valid JSON)
  - Lenient parse rate (extracts JSON from code fences, prose wrappers)
- Required field validation (dotted path support, e.g., "user.email")
- Uncertainty heuristics:
  - Assumption markers ("I assume", "might", "depends", etc.)
  - Verification markers ("verify", "check official", "confirm with", etc.)
  - Clarifying questions (missing context detection)
- Aggregate metrics per suite and overall (rates, percentiles)
- Latency/throughput statistics (avg, p50, p90)
- Output: `metrics.json` with overall, by_suite, and definitions sections
- 32 comprehensive unit tests for all metric functions

**Lesson 1.5**: Base Model Report v0 generator ✅
- `src/pocketguide/eval/report_base_v0.py` generates human-readable evaluation reports
- Report structure:
  - Title and overview (model ID, purpose)
  - Experimental setup (device, dtype, generation config, benchmark suites)
  - Metrics summary table (overall + per-suite breakdown)
  - Failure taxonomy v0 (7 categories: JSON violations, schema failures, hallucination, etc.)
  - 10 curated failure examples (auto-selected by severity score + diversity)
  - Summary and next steps (key findings, future milestones)
- Auto-selection by severity score:
  - Strict JSON failures (+3 points)
  - Required fields failures (+2 points)
  - Missing verification markers (+1 point)
  - Latency outliers (+1 point)
- Diversity: ensures at least one example from each suite
- Manual override: `configs/report.yaml` with `curated_failure_ids` list
- CLI: `python -m pocketguide.eval.report_base_v0 --run_dir runs/eval/<timestamp>`
- Output: `docs/evaluation_report_base_v0.md`
- 17 comprehensive unit tests

**Lesson 1.6**: Hardening pass (reproducibility + ergonomics) ✅
- Run ID system for deterministic evaluation runs:
  - `src/pocketguide/utils/run_id.py` with `make_run_id()` function
  - Accepts `--run_id` CLI argument for reproducible testing
  - Format: YYYYMMDD_HHMMSS (e.g., "20260128_143052")
- Enhanced meta.json with full provenance tracking:
  - **Run metadata**: run_id, created_at, timezone
  - **Model config**: model_id, revision, seed, device, dtype, generation_config
  - **Suite manifest**: suite_dir, suite_files, suite_counts (per-suite example counts)
  - **Config snapshots**: eval_config_resolved (resolved dict), configs_raw (raw YAML text)
  - **Environment**: python_version, platform, sys_platform, machine
  - **Package versions**: pocketguide, pyyaml, pytest, ruff, torch, transformers, jsonschema
  - **Git snapshot**: git_commit (hash), git_is_dirty (uncommitted changes flag)
- Better error messages with file path + line number:
  - Invalid JSON parsing errors include suite path and line number
  - Missing required fields (id, prompt) report file path and line number
- Guardrails:
  - Run directory collision detection (raises FileExistsError if run_id already exists)
  - Model ID validation (warns if still "REPLACE_ME")
  - All JSONL suite files validated for required fields
- Default behavior: `make eval` runs all suites in data/benchmarks/v0 by default
- 68 comprehensive tests (8 eval tests including error message validation)
  - Summary and next steps (key findings, future milestones)
- Auto-selection by severity score:
  - Strict JSON failures (+3 points)
  - Required fields failures (+2 points)
  - Missing verification markers (+1 point)
  - Latency outliers (+1 point)
- Diversity: ensures at least one example from each suite
- Manual override: `configs/report.yaml` with `curated_failure_ids` list
- CLI: `python -m pocketguide.eval.report_base_v0 --run_dir runs/eval/<timestamp>`
- Output: `docs/evaluation_report_base_v0.md`
- 17 comprehensive unit tests

### Milestone 2: Model Selection & Fine-tuning (Planned)
- Base model evaluation and selection
- LoRA/QLoRA fine-tuning implementation
- Training pipeline with checkpointing
- Quantization for deployment

### Milestone 3: Advanced Features (Planned)
- Multi-turn conversation support
- Context-aware responses
- Safety and reliability enhancements
- Extended evaluation metrics

### Milestone 4: Production Deployment (Planned)
- Optimized inference engine
- Mobile/offline deployment packaging
- Performance benchmarking
- User documentation

## Development

### Running Tests

```bash
# All tests
make test

# Specific test file
source .venv/bin/activate
pytest tests/test_smoke_cli.py -v

# With coverage
pytest tests/ --cov=pocketguide
```

### Code Quality

```bash
# Check linting
make lint

# Auto-fix issues
make format
```

### Adding Evaluation Suites

Create JSONL files in `data/benchmarks/v0/` (or your custom directory) with the format:

```jsonl
{"id": "example_001", "prompt": "Your travel question here"}
{"id": "example_002", "prompt": "Another travel question", "required_fields": ["name", "age"]}
{"id": "example_003", "prompt": "Nested field test", "required_fields": ["user.email", "profile.settings"]}
```

**Required Fields** (optional):
- Add `"required_fields"` to test structured output validation
- Supports dotted paths for nested fields (e.g., `"user.email"`, `"metadata.version"`)
- Fields are checked for presence and non-empty values (not null, not "", not empty list/dict)
- Used primarily in format suite for JSON compliance testing

Run with:
```bash
python -m pocketguide.eval.benchmark --suite_dir data/benchmarks/v0
```

## Current Status

**Milestone 0 Complete** - The project has a stable foundation with:
- ✅ Working CLI with structured outputs
- ✅ Evaluation framework with smoke tests
- ✅ Passing test suite
- ✅ Clean, documented codebase
- ✅ Deterministic, reproducible operations

**Next Steps**: Implement data pipeline (Milestone 1) with real travel datasets and expanded evaluation suites.

## Contributing

This project follows standard Python development practices:
- Type hints for all public functions
- Docstrings in Google style
- Test coverage for all features
- Ruff for linting and formatting

## License

*(To be determined)*

---

**Note**: This is Milestone 0 - the CLI currently returns stub responses. Real model inference will be implemented in Milestone 2.
