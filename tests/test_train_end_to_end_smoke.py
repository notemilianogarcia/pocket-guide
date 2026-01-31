"""
Smoke test for LoRA training end-to-end (Milestone 5 — Lesson 5.3).

Uses runtime.dummy_model=true and tiny SFT files; no network or big model download.
Skips when accelerate/peft are not installed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("accelerate")
pytest.importorskip("peft")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_sft(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _sft_record(rid: str, prompt: str, target: str) -> dict:
    return {
        "id": rid,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        "target": target,
        "metadata": {"payload_type": "checklist", "source_split": "train"},
    }


def test_train_end_to_end_smoke_dummy_model(tmp_path: Path) -> None:
    """Run training with dummy_model=true, max_steps=2; assert adapter/ and train_metrics.json."""
    sft_dir = tmp_path / "sft"
    train_sft = sft_dir / "train_sft.jsonl"
    val_sft = sft_dir / "val_sft.jsonl"
    _write_sft(
        train_sft,
        [
            _sft_record("1", "P1", '{"summary":"r1"}'),
            _sft_record("2", "P2", '{"summary":"r2"}'),
            _sft_record("3", "P3", '{"summary":"r3"}'),
        ],
    )
    _write_sft(
        val_sft,
        [
            _sft_record("v1", "V1", '{"summary":"v1"}'),
        ],
    )

    config = {
        "base_model_id": "test/model",
        "data": {
            "train_path": str(tmp_path / "train.jsonl"),
            "val_path": str(tmp_path / "val.jsonl"),
            "train_sft_path": str(train_sft),
            "val_sft_path": str(val_sft),
            "max_seq_len": 128,
        },
        "output": {"runs_dir": str(tmp_path / "runs" / "train")},
        "training": {
            "seed": 42,
            "batch_size": 1,
            "grad_accum_steps": 1,
            "lr": 2e-4,
            "num_epochs": 1,
            "max_grad_norm": 1.0,
            "warmup_steps": 0,
            "eval_every_steps": 1,
            "save_every_steps": 10,
        },
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["c_attn"]},
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
            "dummy_model": True,
            "num_workers": 0,
            "pin_memory": False,
        },
        "logging": {"log_every_steps": 1},
    }
    config_path = tmp_path / "train_lora.yaml"
    import yaml
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    # Create minimal raw train/val for dry_run validation (train.py validates these first)
    raw_train = tmp_path / "train.jsonl"
    raw_val = tmp_path / "val.jsonl"
    raw_train.write_text('{"id":"1","prompt":"P1","response":{"summary":"r1"}}\n')
    raw_val.write_text('{"id":"v1","prompt":"V1","response":{"summary":"v1"}}\n')

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pocketguide.train.train",
            "--config",
            str(config_path),
            "--max_steps",
            "2",
            "--run_id",
            "smoke_run",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)

    run_dir = tmp_path / "runs" / "train" / "smoke_run"
    assert run_dir.exists(), "Run directory should exist"
    adapter_dir = run_dir / "adapter"
    assert adapter_dir.exists(), "adapter/ should exist"
    metrics_path = run_dir / "train_metrics.json"
    assert metrics_path.exists(), "train_metrics.json should exist"

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    assert "run_id" in metrics
    assert metrics["run_id"] == "smoke_run"
    assert "train" in metrics and isinstance(metrics["train"], list)
    assert "val" in metrics and isinstance(metrics["val"], list)
    assert "final" in metrics
    assert "val_loss" in metrics["final"]
    assert len(metrics["train"]) >= 1
    assert len(metrics["val"]) >= 1
