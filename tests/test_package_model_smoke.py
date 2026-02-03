import json
import hashlib
from pathlib import Path

from pocketguide.artifacts.package_model import package_model_main


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def test_package_model_smoke(tmp_path: Path) -> None:
    """Smoke test: package a dummy GGUF file and verify layout + meta."""
    # Create a tiny dummy gguf-like binary file
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    gguf_src = src_dir / "dummy.Q4_K_M.gguf"
    content = b"dummy gguf content"
    gguf_src.write_bytes(content)

    # Artifacts root under tmp_path
    artifacts_root = tmp_path / "artifacts_root"

    model_name = "test-model"

    # Run packager in copy mode so tests don't depend on symlink support
    package_model_main(
        [
            "--model_name",
            model_name,
            "--gguf",
            str(gguf_src),
            "--link_mode",
            "copy",
            "--artifacts_root",
            str(artifacts_root),
        ]
    )

    model_dir = artifacts_root / model_name
    gguf_dir = model_dir / "gguf"
    dest = gguf_dir / "model.Q4_K_M.gguf"
    assert dest.exists(), "Packaged GGUF file should exist with stable name"

    # meta.json should exist and contain gguf_files with sha256
    meta_path = model_dir / "meta.json"
    assert meta_path.exists(), "meta.json should be written"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("model_name") == model_name
    ggufs = meta.get("gguf_files") or []
    assert len(ggufs) == 1
    gg = ggufs[0]
    assert gg.get("filename") == "model.Q4_K_M.gguf"
    assert gg.get("size_bytes") == dest.stat().st_size
    assert gg.get("sha256") == _sha256(dest)

