# PocketGuide

**Domain-adapted LLM for structured travel guidance with evaluation-first design and offline inference**

---

## Overview

PocketGuide addresses a specific ML systems challenge: adapting a general-purpose language model to produce reliable, structured outputs for travel planning—under constraints that matter in practice.

The problem is not "build a chatbot." It is: given a 7B-parameter base model, can we fine-tune it to consistently emit JSON-structured travel guidance (itineraries, checklists, decision trees, procedures) that follows a defined schema, acknowledges uncertainty, and runs entirely offline on consumer hardware?

This requires solving multiple subproblems: defining output contracts before training, generating high-quality synthetic instruction data, implementing parameter-efficient fine-tuning with reproducible evaluation, quantizing for local inference, and iterating based on measured failure modes—not vibes.

---

## What This Project Demonstrates

- **Evaluation-first development**: Benchmarks, schemas, and metrics defined before training; all iterations measured against fixed evaluation suites
- **Synthetic instruction data pipeline**: Teacher-student generation with multi-stage quality gating (drafts → critiques → acceptance)
- **Parameter-efficient fine-tuning**: LoRA adapters on Llama-2-7B with config-driven training and checkpoint management
- **Structured output contracts**: JSON envelope schema with typed payloads (itinerary, checklist, decision_tree, procedure) validated at inference time
- **Quantization and local inference**: GGUF conversion via llama.cpp with registry-driven model selection and performance measurement
- **Iterative improvement**: Multiple training iterations (v1–v5) targeting specific failure modes identified through evaluation
- **Reproducibility infrastructure**: Deterministic seeds, run provenance, artifact hashing, and config-driven execution throughout

---

## System Architecture

The system follows a contracts-first, evaluation-driven design:

1. **Contracts and benchmarks**: Output schemas (envelope + payload) and evaluation suites are defined first. All downstream work is measured against these fixed contracts.

2. **Teacher data generation**: A three-stage pipeline produces training data:
   - Prompt planning: Spec-driven generation of diverse travel queries (categories, regions, difficulty levels)
   - Draft generation: Teacher model (via OpenRouter) produces candidate responses
   - Critique and gating: Quality filters and teacher-as-judge reject low-quality or non-compliant samples

3. **Training**: LoRA fine-tuning on Llama-2-7B with SFT formatting, gradient checkpointing, and configurable hyperparameters. Adapters are checkpointed and evaluated against held-out validation sets.

4. **Inference runtimes**: Two paths—Hugging Face + PEFT for adapter-based evaluation, and llama.cpp for quantized local inference. A registry maps logical model names to GGUF paths.

5. **Evaluation and iteration**: Fixed prompt suites measure parse success, field presence, uncertainty markers, and latency. Failure analysis drives targeted dataset and training interventions.

---

## Key Components

### Evaluation Framework

`src/pocketguide/eval/` implements benchmark loading, metrics computation (strict/lenient JSON parsing, schema validation, uncertainty detection), and report generation. Evaluation runs are timestamped and produce structured outputs (`runs/eval/<run_id>/`).

### Output Schemas and Contracts

`src/pocketguide/data/schemas/` defines:
- **Envelope schema (v0)**: Required fields for all responses—summary, assumptions, uncertainty_notes, next_steps, verification_steps, payload_type, payload
- **Payload schemas (v1)**: Type-specific schemas for itinerary, checklist, decision_tree, and procedure payloads

Validation occurs at inference time via `pocketguide.eval.parsing`.

### Teacher Data Generation

`src/pocketguide/data_generation/` implements the full synthetic data pipeline:
- `plan_prompts.py`: Spec-driven prompt generation with category/region/difficulty balancing
- `generate_drafts.py`: Teacher model inference with rate limiting and resume support
- `generate_critiques.py`: Quality scoring and issue detection
- `qa_pipeline_v1.py`, `quality_filters.py`, `quality_gates.py`: Multi-stage filtering and acceptance logic

### Training Pipeline

`src/pocketguide/train/` provides:
- `train.py`: LoRA fine-tuning with PEFT, gradient accumulation, mixed precision, and checkpoint saving
- `prepare_sft_data.py`: Conversion to SFT format with system/user/assistant turns
- `run_samples.py`: Adapter-based sample generation for evaluation

### Local Inference

`src/pocketguide/inference/` and `src/pocketguide/quant/`:
- `local_llamacpp.py`: llama.cpp integration for GGUF inference
- `quantize_gguf.py`: Adapter merging and GGUF conversion pipeline
- `src/pocketguide/artifacts/`: Model packaging, registry resolution, and performance measurement

---

## Results and Iteration Summary

Five training iterations (v1–v5) were conducted, each targeting failure modes identified through evaluation on a fixed 20-prompt suite.

**Key improvements achieved:**
- Parse success rate improved from 0.80 (v1) to 1.00 (v3–v5)—the adapted model now consistently produces valid JSON
- Uncertainty marker presence increased from 0.85 to 1.00—the model reliably signals assumptions and verification needs
- Required envelope field presence improved from 0.00 to 0.20 (v4 peak)

**Iteration approach:**
- v1: Baseline LoRA training on synthetic dataset v1
- v2: Dataset intervention (hard prompts, stricter QC) + learning rate adjustment
- v3: Increased epochs, sequence length, and LoRA capacity
- v4: Prompt and payload normalization in SFT preparation
- v5: Extended training with SFT validation improvements

The iteration process demonstrated that parse success and partial structure compliance respond to data quality and training duration, while full schema compliance requires further architectural or objective changes.

See [docs/evaluation_report_v2.md](docs/evaluation_report_v2.md) for detailed metrics, failure mode analysis, and intervention rationale.

---

## Project Structure

```
pocket-guide/
├── src/pocketguide/         # Core package
│   ├── eval/                # Evaluation framework, metrics, parsing, reports
│   ├── train/               # LoRA training, SFT preparation, sample generation
│   ├── data_generation/     # Teacher pipeline (prompts, drafts, critiques, QA)
│   ├── inference/           # CLI and runtime (HF, llama.cpp)
│   ├── quant/               # GGUF quantization
│   ├── artifacts/           # Model packaging, registry, perf measurement
│   ├── teachers/            # Teacher model abstraction (OpenRouter)
│   └── data/schemas/        # JSON schemas (envelope v0, payloads v1)
├── configs/                 # All configuration files (train, eval, quant, data)
├── data/
│   ├── benchmarks/          # Evaluation suites (v0, v1)
│   ├── processed/           # Generated datasets, splits, SFT files
│   ├── prompts/             # Teacher prompt templates
│   └── specs/               # Dataset generation specifications
├── eval/suites/             # Fixed evaluation prompt suites
├── runs/                    # All run outputs (train, eval, quant)
├── docs/                    # Guides and reports
└── tests/                   # Test suite
```

---

## Usage

Primary entry points via Makefile:

```bash
# Environment setup
make env                    # Create Python environment
make test                   # Run test suite

# Data generation (requires OPENROUTER_API_KEY)
make data                   # Generate prompt plans
make drafts                 # Generate teacher drafts
make critiques              # Generate quality critiques
make dataset                # Produce final training dataset
make split                  # Create train/val/test splits
make prepare-sft            # Convert to SFT format

# Training
make train                  # Run LoRA fine-tuning

# Inference
make run PROMPT="..."                           # HF runtime (default)
make run LOCAL=1 MODEL=adapted PROMPT="..."     # Local llama.cpp runtime

# Evaluation
make eval                   # Run evaluation on benchmark suites
make eval-local             # Local runtime evaluation

# Quantization
make quantize TRAIN_RUN=<run_id>                # Convert adapter to GGUF
make package_model MODEL_NAME=... GGUF=...      # Package for deployment
```

See [docs/inference_guide.md](docs/inference_guide.md) for local inference details and [docs/local_runtime_guide.md](docs/local_runtime_guide.md) for runtime configuration.

---

## Reproducibility and Discipline

Every pipeline stage is designed for reproducibility:

- **Config-driven execution**: All training, evaluation, and data generation parameters live in YAML configs under `configs/`
- **Deterministic seeding**: Fixed seeds for data generation, shuffling, and training
- **Run provenance**: Each run produces `meta.json` with git commit, package versions, resolved config, and file hashes
- **Artifact tracking**: Dataset manifests record counts, acceptance rates, and filtering decisions; model artifacts include training and quantization lineage
- **Fixed evaluation**: Benchmark suites are versioned and unchanged across iterations; metrics are computed with identical parsing and validation logic

---

## Future Work

Planned extensions (not yet implemented):

- **Server mode**: FastAPI wrapper for persistent inference service
- **JSON repair**: Post-processing to recover from minor format violations
- **Streaming inference**: Token-by-token output for interactive use
- **Larger model exploration**: Adapter training on 13B+ base models
- **Loss weighting**: Emphasize envelope/payload fields during training to improve structure compliance

---

## Safety and Limitations

PocketGuide is a research prototype, not an authoritative travel information source.

- **Uncertainty by design**: The output schema requires `uncertainty_notes` and `verification_steps` fields, encouraging users to verify time-sensitive or safety-critical information with official sources
- **Not real-time**: Training data has a knowledge cutoff; visa requirements, border policies, and local conditions change
- **Quantization tradeoffs**: GGUF models (Q4_K_M) trade precision for size and speed; occasional malformed outputs may occur
- **No guarantees**: Travel advice should always be cross-referenced with embassy websites, official government sources, and current conditions

---

## License

See [LICENSE](LICENSE) for terms.
