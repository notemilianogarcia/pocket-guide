"""
Package quantized GGUF models into a stable artifacts layout.

Layout:
    artifacts/models/<model_name>/
      gguf/
        model.Q4_K_M.gguf
        model.Q5_K_M.gguf   (optional)
      meta.json

CLI:
    python -m pocketguide.artifacts.package_model \
      --model_name <name> \
      --gguf path/to/model.Q4_K_M.gguf \
      [--extra_gguf path/to/model.Q5_K_M.gguf] \
      [--train_run runs/train/<id>] \
      [--quant_run runs/quant/<id>] \
      [--link_mode copy|symlink] \
      [--artifacts_root /custom/path]    # default: artifacts/models/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GGUFFileMeta:
    filename: str
    quant: str | None
    size_bytes: int
    sha256: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _infer_quant_from_name(name: str) -> str | None:
    """Infer quantization tag from a GGUF filename."""
    if not name.endswith(".gguf"):
        return None
    parts = name.rsplit(".", 2)
    if len(parts) < 2:
        return None
    quant = parts[-2]
    return quant or None


def _stable_dest_name_for_quant(quant: str | None) -> str:
    """Map quantization tag to a stable destination filename."""
    if not quant:
        return "model.gguf"
    q = quant.upper()
    if q.startswith("Q4"):
        return "model.Q4_K_M.gguf"
    if q.startswith("Q5"):
        return "model.Q5_K_M.gguf"
    return f"model.{quant}.gguf"


def _get_git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _load_base_model_from_run(run_dir: Path) -> str | None:
    """Try to infer base_model_id from a run's config.yaml."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    try:
        with config_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        base_model_id = cfg.get("base_model_id")
        if isinstance(base_model_id, str):
            return base_model_id
    except Exception:
        return None
    return None


def package_model_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Package quantized GGUF model into artifacts/models/")
    parser.add_argument("--model_name", required=True, help="Logical model name (e.g. base-llama2-7b)")
    parser.add_argument("--gguf", required=True, help="Path to primary GGUF file (e.g. Q4_K_M)")
    parser.add_argument("--extra_gguf", help="Optional secondary GGUF file (e.g. Q5_K_M)")
    parser.add_argument("--train_run", help="Optional training run directory (e.g. runs/train/<id>)")
    parser.add_argument("--quant_run", help="Optional quantization run directory (e.g. runs/quant/<id>)")
    parser.add_argument(
        "--link_mode",
        choices=["copy", "symlink"],
        default="symlink",
        help="Whether to copy or symlink GGUF files into artifacts (default: symlink)",
    )
    parser.add_argument(
        "--artifacts_root",
        help="Root directory for packaged models (default: artifacts/models under project root)",
    )

    args = parser.parse_args(argv)

    project_root = Path.cwd()
    if args.artifacts_root:
        artifacts_root = Path(args.artifacts_root)
    else:
        artifacts_root = project_root / "artifacts" / "models"

    model_name: str = args.model_name
    gguf_paths: list[tuple[Path, str]] = []

    primary = Path(args.gguf)
    if not primary.exists():
        raise FileNotFoundError(f"GGUF file not found: {primary}")
    gguf_paths.append((primary, "primary"))

    if args.extra_gguf:
        extra = Path(args.extra_gguf)
        if not extra.exists():
            raise FileNotFoundError(f"extra_gguf file not found: {extra}")
        gguf_paths.append((extra, "extra"))

    model_dir = artifacts_root / model_name
    gguf_dir = model_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    link_mode = args.link_mode
    gguf_meta: list[GGUFFileMeta] = []

    for src_path, _role in gguf_paths:
        quant = _infer_quant_from_name(src_path.name)
        dest_name = _stable_dest_name_for_quant(quant)
        dest_path = gguf_dir / dest_name

        if dest_path.exists():
            dest_path.unlink()

        if link_mode == "copy":
            # Copy GGUF into artifacts
            data = src_path.read_bytes()
            dest_path.write_bytes(data)
        else:
            # Symlink using a relative path for portability
            rel = os.path.relpath(src_path.resolve(), dest_path.parent.resolve())
            dest_path.symlink_to(rel)

        size_bytes = dest_path.stat().st_size
        # Hash the target content (follows symlink)
        sha = _sha256(dest_path.resolve())
        gguf_meta.append(
            GGUFFileMeta(
                filename=dest_path.name,
                quant=quant,
                size_bytes=size_bytes,
                sha256=sha,
            )
        )

    # Build source provenance
    source: dict[str, Any] = {}
    if args.train_run:
        source["train_run"] = args.train_run
        train_run_dir = project_root / args.train_run
        bm = _load_base_model_from_run(train_run_dir)
        if bm:
            source["base_model_id"] = bm
    if args.quant_run:
        source["quant_run"] = args.quant_run
        quant_run_dir = project_root / args.quant_run
        bm = _load_base_model_from_run(quant_run_dir)
        if bm and "base_model_id" not in source:
            source["base_model_id"] = bm

    meta = {
        "model_name": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(project_root),
        "source": source,
        "gguf_files": [asdict(m) for m in gguf_meta],
        "notes": "",
    }

    meta_path = model_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Packaged model '{model_name}' into {model_dir}")


def main() -> None:  # pragma: no cover - thin wrapper
    package_model_main()


if __name__ == "__main__":  # pragma: no cover
    main()

