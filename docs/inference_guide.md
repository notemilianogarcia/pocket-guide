# PocketGuide — Local Inference Guide

## 1. Overview

PocketGuide supports local, offline inference via quantized GGUF models. Models are packaged under `artifacts/models/` with provenance metadata, and model selection is registry-driven. This guide shows how to run base vs adapted models locally and what performance to expect.

---

## 2. Obtaining GGUF Models

GGUF files are not committed to git. You must generate or copy them locally.

### Option A: Generate locally

Run the quantization and packaging pipeline:

```bash
# Quantize a trained adapter
make quantize TRAIN_RUN=<train_run_id> LLAMACPP_DIR=<path_to_llama.cpp>

# Package the GGUF into artifacts/models/
make package_model MODEL_NAME=adapted GGUF=runs/quant/<quant_run_id>/model.Q4_K_M.gguf TRAIN_RUN=<train_run_id> QUANT_RUN=<quant_run_id>
```

This creates:
- `artifacts/models/<model_name>/gguf/model.Q4_K_M.gguf`
- `artifacts/models/<model_name>/meta.json` (provenance)

### Option B: Copy prebuilt artifacts

If you have a prebuilt GGUF file:

1. Copy it into `artifacts/models/<model_name>/gguf/`
2. Ensure `configs/model_registry.yaml` points to the correct path

Explicitly:
- **GGUF files are intentionally gitignored.** They are not committed due to size.
- **`meta.json` and `perf.json` are tracked** for provenance and performance characteristics.

---

## 3. Running Inference Locally

PocketGuide uses a registry-driven model selector. Set `MODEL=base` or `MODEL=adapted` to choose the model from [configs/model_registry.yaml](configs/model_registry.yaml).

### Base model

```bash
make run LOCAL=1 MODEL=base PROMPT="plan a 2-day itinerary for Montreal"
```

### Adapted model

```bash
make run LOCAL=1 MODEL=adapted PROMPT="plan a 2-day itinerary for Montreal"
```

**What this does:**
- `LOCAL=1` switches to llama.cpp runtime (requires compiled llama.cpp and `LLAMACPP_DIR` configured)
- `MODEL=base|adapted` selects the GGUF via the registry
- Output is JSON-only and follows the PocketGuide response envelope (summary, assumptions, uncertainty_notes, next_steps, verification_steps, payload_type, payload)

---

## 4. Example Prompts

```bash
# Itinerary (multi-day planning)
make run LOCAL=1 MODEL=adapted PROMPT="Create a 3-day itinerary for Kyoto, Japan for a first-time visitor. Include cultural sites and local food experiences."

# Checklist (structured tasks)
make run LOCAL=1 MODEL=adapted PROMPT="Create a travel preparation checklist for a 2-week trip to Southeast Asia. Include visa requirements, vaccinations, and packing essentials."

# Decision tree (conditional logic)
make run LOCAL=1 MODEL=adapted PROMPT="Create a decision tree for choosing accommodation in Tokyo based on budget, location preferences, and travel style."

# Uncertainty-aware question
make run LOCAL=1 MODEL=adapted PROMPT="What are the visa requirements for US citizens traveling to Vietnam? Include verification_steps for official sources."
```

---

## 5. Performance Snapshot

Performance characteristics are measured via the `measure_perf` script and recorded in `artifacts/models/<model_name>/perf.json`.

If a `perf.json` file exists for your model, it will contain:

- **Quantization type** (e.g., Q4_K_M)
- **Approximate RAM usage** (peak RSS in bytes; single-sample measurement via psutil)
- **Load time** (warm-up prompt latency; approximates model load overhead)
- **Latency distribution** (avg / p50 / p90 in milliseconds over a fixed prompt suite)
- **Approximate tokens/sec** (based on output length heuristic: len(text) / 4.0)

**Example** (machine-dependent; your numbers will vary):

```
Quantization:   Q4_K_M
Load time:      ~2500 ms (first-prompt warm-up)
Avg latency:    ~1800 ms (fixed10 suite)
P90 latency:    ~2400 ms
Approx RAM:     ~6.2 GB (process RSS; macOS M1 Pro)
```

If `perf.json` is missing, generate it:

```bash
python -m pocketguide.artifacts.measure_perf \
  --model_name adapted \
  --registry configs/model_registry.yaml \
  --suite eval/suites/fixed10_v1.jsonl \
  --runtime_config configs/runtime_local.yaml
```

This writes `artifacts/models/adapted/perf.json`.

**Important notes:**
- All performance numbers are **machine-dependent** (CPU, RAM, OS).
- Latency scales with prompt length and output complexity.
- Quantization level (Q4_K_M vs Q5_K_M vs Q8_0) trades model size for precision and speed.
- RAM usage reflects single-process measurement; actual usage depends on model size and context length.

---

## 6. Limitations and Notes

- **CPU-only inference:** llama.cpp runtime does not batch; each prompt is processed sequentially.
- **Quantized models may produce invalid JSON:** Q4_K_M and lower quantization levels occasionally emit malformed JSON. Use lenient parsing or retry logic in production.
- **Latency depends on prompt length and quantization level:** Longer prompts and lower quantization levels increase latency.
- **CLI-oriented usage:** This is a single-shot CLI inference tool, not a persistent server. For server deployment, consider wrapping the runtime in FastAPI or similar.
- **No streaming:** Output is returned in full after generation completes.
- **Model availability:** The registry points to GGUF paths under `artifacts/models/`. If the GGUF file is missing, inference will fail.

---

## 7. Related Documentation

- [local_runtime_guide.md](local_runtime_guide.md) — Runtime internals, configuration, and error handling
- [evaluation_report_v2.md](evaluation_report_v2.md) — Evaluation results for base and adapted models

---

