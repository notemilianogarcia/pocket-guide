"""
Model registry utilities for PocketGuide.

The registry is a small YAML file that maps logical model names to GGUF paths,
so the rest of the codebase can refer to models by name instead of hardcoding
paths or run IDs.

Example (configs/model_registry.yaml):

models:
  base:
    description: "Base quantized model"
    gguf_path: "artifacts/models/base-llama2-7b/gguf/model.Q4_K_M.gguf"
    quant: "Q4_K_M"
  adapted:
    description: "LoRA-adapted quantized model v5"
    gguf_path: "artifacts/models/pocketguide-v5/gguf/model.Q4_K_M.gguf"
    quant: "Q4_K_M"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _resolve_path(path: str | Path, project_root: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    root = project_root or Path.cwd()
    return (root / p).resolve()


def load_model_registry(path: str | Path, project_root: Path | None = None) -> dict[str, Any]:
    """Load model registry YAML and perform basic validation."""
    registry_path = _resolve_path(path, project_root)
    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry not found: {registry_path}")

    with registry_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Model registry must be a mapping, got {type(data).__name__}")

    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Model registry must have a non-empty 'models' mapping")

    return data


def resolve_model_gguf(
    model_name: str,
    registry_path: str | Path,
    project_root: Path | None = None,
) -> Path:
    """Resolve GGUF path for a logical model name from the registry.

    Returns an absolute Path. Does NOT require the file to exist, so this
    function can be used in stub/test environments as well.
    """
    data = load_model_registry(registry_path, project_root=project_root)
    models = data.get("models") or {}
    if model_name not in models:
        available = ", ".join(sorted(models.keys()))
        raise KeyError(f"Unknown model '{model_name}' in registry. Available models: {available}")

    entry = models[model_name] or {}
    gguf_path = entry.get("gguf_path")
    if not gguf_path or not str(gguf_path).strip():
        raise ValueError(f"Model '{model_name}' in registry is missing 'gguf_path'")

    return _resolve_path(gguf_path, project_root=project_root)


def resolve_model_quant(
    model_name: str,
    registry_path: str | Path,
    project_root: Path | None = None,  # unused but keeps API symmetrical
) -> str | None:
    """Resolve quantization tag for a logical model name from the registry."""
    data = load_model_registry(registry_path, project_root=project_root)
    models = data.get("models") or {}
    if model_name not in models:
        available = ", ".join(sorted(models.keys()))
        raise KeyError(f"Unknown model '{model_name}' in registry. Available models: {available}")
    entry = models[model_name] or {}
    quant = entry.get("quant")
    return quant

