"""
Tests for quantization pipeline dry-run (Lesson 6.2).

No model loading or real conversion. Temp config with fake paths; assert run dir,
config snapshot, and meta.json with planned commands.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _minimal_quant_config(tmp_path: Path, llamacpp_dir: Path) -> dict:
    """Minimal quant config pointing to fake model/adapter and fake llama.cpp dir."""
    return {
        "input": {
            "base_model_id": "meta-llama/Llama-2-7b-hf",
            "adapter_dir": str(tmp_path / "fake_train_run" / "adapter"),
        },
        "output": {"runs_dir": str(tmp_path / "runs" / "quant"), "name": None},
        "llamacpp": {
            "repo_dir": str(llamacpp_dir),
            "convert_script": "convert_hf_to_gguf.py",
            "quantize_bin": None,
            "llama_print_system_info": True,
        },
        "merge": {"enabled": True, "merged_output_dirname": "merged_hf"},
        "gguf": {"f16_filename": "model.f16.gguf"},
        "quant": {"formats": ["Q4_K_M", "Q5_K_M"], "output_template": "model.{quant}.gguf"},
    }


def test_quantize_dry_run_creates_run_dir_and_meta(tmp_path: Path) -> None:
    """Dry run: run dir created, config snapshot and meta.json with planned commands."""
    fake_llamacpp = tmp_path / "llamacpp_fake"
    fake_llamacpp.mkdir(parents=True)

    config = _minimal_quant_config(tmp_path, fake_llamacpp)
    config_path = tmp_path / "quantize_gguf.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pocketguide.quant.quantize_gguf",
            "--config",
            str(config_path),
            "--dry_run",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)

    runs_dir = tmp_path / "runs" / "quant"
    assert runs_dir.exists(), "runs/quant should exist"
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1, "Exactly one run directory should be created"
    run_dir = run_dirs[0]

    config_yaml = run_dir / "config.yaml"
    assert config_yaml.exists(), "config.yaml snapshot must exist"
    meta_json = run_dir / "meta.json"
    assert meta_json.exists(), "meta.json must exist"

    with open(meta_json, encoding="utf-8") as f:
        meta = json.load(f)

    assert "run_id" in meta
    assert "timestamp" in meta
    assert "commands" in meta
    commands = meta["commands"]
    assert isinstance(commands, list), "meta.commands must be a list"
    assert len(commands) >= 1, "At least merge + convert + quantize planned"
    # Merge step (conceptual), convert command, quantize commands
    merge_planned = any("[merge]" in c for c in commands)
    convert_planned = any("convert_hf_to_gguf" in c for c in commands)
    assert merge_planned or convert_planned, "Planned commands should include merge or convert"

    # Config snapshot should match effective config (with our paths)
    with open(config_yaml, encoding="utf-8") as f:
        snapshot = yaml.safe_load(f)
    assert snapshot.get("input", {}).get("base_model_id") == "meta-llama/Llama-2-7b-hf"
    assert "llamacpp" in snapshot


def test_quantize_dry_run_fails_without_llamacpp_dir(tmp_path: Path) -> None:
    """When repo_dir is null and LLAMACPP_DIR unset, script should exit non-zero."""
    config = _minimal_quant_config(tmp_path, Path("/nonexistent"))
    config["llamacpp"]["repo_dir"] = None
    config_path = tmp_path / "quantize_gguf.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    env.pop("LLAMACPP_DIR", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pocketguide.quant.quantize_gguf",
            "--config",
            str(config_path),
            "--dry_run",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "LLAMACPP_DIR" in result.stderr or "repo_dir" in result.stderr.lower()
