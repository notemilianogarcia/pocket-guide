# models/

Local model artifacts are stored here and are **not committed** to git (see `.gitignore`).

## Layout

- **base/** — Optional; base model weights (not used by default).
- **adapters/** — Optional; LoRA or other adapter weights (not used by default).
- **gguf/** — GGUF quantized models for local inference (llama.cpp). Place your `.gguf` files here and reference them in `configs/runtime_local.yaml` via `model.gguf_path`.

## Usage

1. Download or convert a GGUF model and place it under `models/gguf/`.
2. Set `model.gguf_path` in `configs/runtime_local.yaml` to the path (e.g. `models/gguf/your_model.gguf`).
3. Set `runtime.stub: false` when ready to run real inference.
