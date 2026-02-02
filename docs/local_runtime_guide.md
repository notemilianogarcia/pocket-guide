# PocketGuide — Local Runtime Guide

## 1. Overview

PocketGuide supports offline local inference via llama.cpp. Models are quantized to GGUF and are not stored in the repository. This guide explains how to obtain a model and run inference locally.

## 2. Prerequisites

- **macOS** (Apple Silicon recommended for better performance)
- **Python 3.11+** (see `pyproject.toml` for the project’s required version)
- **llama.cpp** compiled locally; see the [official repository](https://github.com/ggml-org/llama.cpp) for build instructions
- **Disk space:** roughly 4–8 GB per quantized 7B model (varies by quant level)

## 3. Obtaining Model Weights

GGUF files are produced by the quantization pipeline and are expected to live under `models/gguf/`.

**Path A — Generate locally**

- Run the quantization pipeline with a trained adapter:  
  `make quantize TRAIN_RUN=<run_id>`  
  (Requires a built llama.cpp checkout and `LLAMACPP_DIR` set; see the repo’s quantization docs.)

**Path B — Copy from another machine**

- Copy the GGUF file into `models/gguf/`.
- Set `model.gguf_path` in `configs/runtime_local.yaml` to that path (e.g. `models/gguf/your_model.Q4_K_M.gguf`).

GGUF files are intentionally gitignored (`models/*`). You must manage them locally; they are not committed to the repository.

## 4. Running PocketGuide Locally

**Using Make:**

```bash
make run_local PROMPT="plan a 2-day itinerary for Montreal"
```

**Using Python directly:**

```bash
python -m pocketguide.inference.cli --runtime local --runtime_config configs/runtime_local.yaml --prompt "plan a 2-day itinerary for Montreal"
```

`configs/runtime_local.yaml` controls the model path, context length, and generation settings (temperature, top_p, max_tokens). Set `runtime.stub` to `false` for real inference; when it is `true`, the CLI returns a stub response without calling llama.cpp.

## 5. Output Contracts and Error Handling

All outputs are expected to follow the PocketGuide response envelope (summary, assumptions, uncertainty_notes, next_steps, verification_steps, payload_type, payload). Outputs are parsed and validated against the envelope and payload schemas. If parsing or validation fails, the CLI returns a structured error envelope (JSON) instead of crashing; the error payload describes the failure and suggests next steps.

## 6. Known Limitations

- Quantized models may occasionally produce invalid or malformed JSON; the CLI returns a structured error in that case.
- Schema compliance (envelope and payload) may be lower than when using the Hugging Face runtime.
- Latency depends on quant level, context length, and hardware; longer contexts and larger models increase latency.
- This runtime is single-user and CLI-oriented; it is not a multi-tenant server.

## 7. What’s Next

Possible future extensions include: a server mode for remote clients, batched inference for multiple prompts, and support for additional quantization levels or backends.
