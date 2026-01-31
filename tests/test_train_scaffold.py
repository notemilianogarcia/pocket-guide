"""
Tests for training scaffold (Milestone 5 — Lesson 5.1).

No network access, no model downloads. Uses temp train/val JSONL and minimal config.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# Project root (parent of tests/) so PYTHONPATH can find src/pocketguide
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _minimal_train_config(tmp_path: Path) -> dict:
    """Minimal train_lora-style config pointing to temp paths."""
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    return {
        "base_model_id": "test/model",
        "data": {
            "train_path": str(train_path),
            "val_path": str(val_path),
        },
        "output": {"runs_dir": str(tmp_path / "runs" / "train")},
        "training": {
            "seed": 42,
            "batch_size": 1,
            "grad_accum_steps": 16,
            "lr": 2.0e-4,
            "num_epochs": 1,
            "eval_every_steps": 100,
            "save_every_steps": 200,
        },
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "auto"},
        "runtime": {"device": "auto", "precision": "auto", "num_workers": 2, "pin_memory": True},
        "logging": {"enable_wandb": False, "project": "pocketguide"},
    }


def test_dry_run_creates_run_dir_and_artifacts(tmp_path: Path) -> None:
    """Running with --dry_run creates run directory, config.yaml, and meta.json; no model download."""
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_jsonl(
        train_path,
        [
            {"prompt": "p1", "response": {"summary": "r1"}},
            {"prompt": "p2", "response": {"summary": "r2"}},
        ],
    )
    _write_jsonl(
        val_path,
        [
            {"prompt": "p3", "response": {"summary": "r3"}},
        ],
    )

    config = _minimal_train_config(tmp_path)
    config_path = tmp_path / "train_lora.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    # Run from tmp_path so relative paths in config resolve; PYTHONPATH so package is found
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "pocketguide.train.train", "--config", str(config_path), "--dry_run"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)

    runs_dir = tmp_path / "runs" / "train"
    assert runs_dir.exists(), "runs/train should exist"
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1, "Exactly one run directory should be created"
    run_dir = run_dirs[0]

    config_yaml = run_dir / "config.yaml"
    assert config_yaml.exists(), "config.yaml must exist"
    meta_json = run_dir / "meta.json"
    assert meta_json.exists(), "meta.json must exist"

    with open(meta_json, encoding="utf-8") as f:
        meta = json.load(f)
    assert "run_id" in meta
    assert "timestamp" in meta
    assert meta.get("resolved", {}).get("device") in ("cpu", "mps", "cuda")
    assert meta.get("resolved", {}).get("precision") in ("fp32", "fp16", "bf16")
    assert meta.get("base_model_id") == "test/model"
    assert "dataset" in meta
    assert "train_hash" in meta["dataset"]
    assert "val_hash" in meta["dataset"]


def test_dry_run_fails_on_missing_config(tmp_path: Path) -> None:
    """Missing config path exits non-zero."""
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "pocketguide.train.train", "--config", str(tmp_path / "nonexistent.yaml"), "--dry_run"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "Config" in result.stderr or "not found" in result.stderr.lower()


def test_dry_run_fails_on_missing_dataset(tmp_path: Path) -> None:
    """Missing train/val files exit non-zero."""
    config = _minimal_train_config(tmp_path)
    config_path = tmp_path / "train_lora.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    # Do not create train.jsonl / val.jsonl

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "pocketguide.train.train", "--config", str(config_path), "--dry_run"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "Dataset" in result.stderr or "not found" in result.stderr.lower()


def test_dry_run_fails_on_invalid_jsonl(tmp_path: Path) -> None:
    """Invalid JSONL or missing prompt/response exits non-zero."""
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        f.write('{"prompt": "ok", "response": {}}\n')
        f.write("not valid json\n")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write('{"prompt": "v", "response": {}}\n')

    config = _minimal_train_config(tmp_path)
    config_path = tmp_path / "train_lora.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "pocketguide.train.train", "--config", str(config_path), "--dry_run"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "JSON" in result.stderr or "Invalid" in result.stderr or "Missing" in result.stderr


def test_dry_run_fails_on_missing_prompt_or_response(tmp_path: Path) -> None:
    """Records missing prompt or response fail validation."""
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_jsonl(train_path, [{"prompt": "p", "response": {}}])
    _write_jsonl(val_path, [{"prompt": "only"}]);  # missing 'response'

    config = _minimal_train_config(tmp_path)
    config_path = tmp_path / "train_lora.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "pocketguide.train.train", "--config", str(config_path), "--dry_run"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "Missing" in result.stderr or "required" in result.stderr.lower()
