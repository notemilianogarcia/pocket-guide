"""
Smoke test for training report v1 generator (Lesson 5.5).

Creates a minimal fake run_dir with small JSON/YAML artifacts, runs the
generator, and asserts the markdown file is created with required sections
and table structure.
"""

import json
from pathlib import Path

import pytest
import yaml

from pocketguide.train.generate_report import generate_report


@pytest.fixture
def fake_run_dir(tmp_path: Path) -> Path:
    """Create minimal run_dir with all required artifacts."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    samples_dir = run_dir / "samples"
    samples_dir.mkdir()

    config = {
        "base_model_id": "meta-llama/Llama-2-7b-hf",
        "data": {
            "train_path": "data/processed/splits/v1/train.jsonl",
            "val_path": "data/processed/splits/v1/val.jsonl",
            "max_seq_len": 1024,
        },
        "training": {
            "seed": 42,
            "batch_size": 1,
            "grad_accum_steps": 16,
            "lr": 2.0e-4,
        },
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
        "runtime": {"device": "cuda", "precision": "bf16"},
    }
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    meta = {
        "run_id": "test_run_001",
        "timestamp": "2026-01-31T12:00:00",
        "base_model_id": "meta-llama/Llama-2-7b-hf",
        "resolved": {"device": "cuda", "precision": "bf16"},
        "dataset": {
            "train_path": "data/processed/splits/v1/train.jsonl",
            "val_path": "data/processed/splits/v1/val.jsonl",
        },
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    train_metrics = {
        "run_id": "test_run_001",
        "global_steps": 100,
        "train": [{"step": 0, "loss": 2.5}, {"step": 100, "loss": 1.8}],
        "val": [{"step": 100, "loss": 1.9}],
        "final": {"val_loss": 1.9, "train_steps": 100, "tokens_seen": 50000},
    }
    with open(run_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)

    base_rec = {
        "prompt_id": "p1",
        "prompt": "Short test prompt.",
        "payload_type": "itinerary",
        "parse_success": False,
        "schema_valid": False,
        "error_code": "JSON_STRICT_PARSE_FAILED",
        "missing_fields": [],
    }
    with open(samples_dir / "base_outputs.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(base_rec, ensure_ascii=False) + "\n")

    finetuned_rec = {
        "prompt_id": "p1",
        "prompt": "Short test prompt.",
        "payload_type": "itinerary",
        "parse_success": True,
        "schema_valid": True,
        "error_code": None,
        "missing_fields": [],
    }
    with open(samples_dir / "finetuned_outputs.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(finetuned_rec, ensure_ascii=False) + "\n")

    comparison = {
        "num_prompts": 1,
        "base": {
            "parse_success_rate": 0.0,
            "schema_valid_rate": 0.0,
            "required_field_presence_rate": 0.0,
            "uncertainty_marker_presence_rate": 0.0,
            "avg_latency_ms": 100.0,
        },
        "finetuned": {
            "parse_success_rate": 1.0,
            "schema_valid_rate": 1.0,
            "required_field_presence_rate": 1.0,
            "uncertainty_marker_presence_rate": 0.5,
            "avg_latency_ms": 120.0,
        },
        "delta": {
            "parse_success_rate": 1.0,
            "schema_valid_rate": 1.0,
            "required_field_presence_rate": 1.0,
            "uncertainty_marker_presence_rate": 0.5,
            "avg_latency_ms": 20.0,
        },
    }
    with open(samples_dir / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    return run_dir


def test_generate_report_creates_markdown_with_sections(
    fake_run_dir: Path, tmp_path: Path
) -> None:
    """Run generator; assert markdown created and required section headers present."""
    out_path = tmp_path / "training_report_v1.md"
    generate_report(fake_run_dir, out_path)

    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")

    assert "# PocketGuide — Training Report v1" in content
    assert "## 1. Overview" in content
    assert "## 2. Training Configuration" in content
    assert "## 3. Quantitative Metrics" in content
    assert "### 3.1 Training & Validation" in content
    assert "### 3.2 Structured Output Metrics" in content
    assert "## 4. Qualitative Examples" in content
    assert "## 5. Failure Taxonomy (v1)" in content
    assert "## 6. Notes & Limitations" in content

    # Tables (markdown pipe syntax)
    assert "|" in content
    assert "Base" in content and "Finetuned" in content
    assert "parse_success_rate" in content or "parse success" in content.lower()


def test_generate_report_fails_without_artifacts(tmp_path: Path) -> None:
    """Generator fails with clear error when required artifacts are missing."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    out_path = tmp_path / "out.md"

    with pytest.raises(FileNotFoundError, match="Missing required artifacts"):
        generate_report(run_dir, out_path)
