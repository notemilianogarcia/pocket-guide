from pathlib import Path

import pytest
import yaml

from pocketguide.artifacts.registry import load_model_registry, resolve_model_gguf


def test_resolve_model_gguf_basic(tmp_path: Path) -> None:
    """Registry resolution should return an absolute path for a known model."""
    artifacts_dir = tmp_path / "artifacts"
    model_dir = artifacts_dir / "models" / "base-test" / "gguf"
    model_dir.mkdir(parents=True)
    gguf_path = model_dir / "model.Q4_K_M.gguf"
    gguf_path.write_bytes(b"dummy")

    registry = {
        "models": {
            "base": {
                "description": "Base model",
                "gguf_path": str(gguf_path.relative_to(tmp_path)),
                "quant": "Q4_K_M",
            }
        }
    }
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")

    # Should load and resolve to the absolute gguf path
    data = load_model_registry(registry_path, project_root=tmp_path)
    assert "models" in data
    resolved = resolve_model_gguf("base", registry_path, project_root=tmp_path)
    assert resolved == gguf_path.resolve()


def test_resolve_model_gguf_unknown_model(tmp_path: Path) -> None:
    """Unknown model names should raise a clear KeyError."""
    registry = {"models": {"base": {"gguf_path": "artifacts/models/base/gguf/model.Q4_K_M.gguf"}}}
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")

    with pytest.raises(KeyError) as exc:
        resolve_model_gguf("adapted", registry_path, project_root=tmp_path)
    assert "Unknown model 'adapted'" in str(exc.value)

