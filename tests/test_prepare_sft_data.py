"""
Tests for SFT data preparation (Milestone 5 — Lesson 5.2).

No model downloads. Uses temp train/val split files.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_split(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _minimal_record(rid: str, prompt: str, payload_type: str = "checklist", difficulty: str = "easy") -> dict:
    return {
        "id": rid,
        "prompt": prompt,
        "response": {
            "summary": f"Summary for {rid}",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": payload_type,
            "payload": {},
        },
        "payload_type": payload_type,
        "category": "visa",
        "difficulty": difficulty,
        "region_tags": ["JP"],
    }


def test_prepare_sft_data_produces_outputs(tmp_path: Path) -> None:
    """Running prepare_sft_data produces train_sft.jsonl, val_sft.jsonl, and fixed suite."""
    splits_dir = tmp_path / "splits"
    out_dir = tmp_path / "sft"
    fixed_path = tmp_path / "fixed20.jsonl"

    train_records = [
        _minimal_record("t1", "Prompt 1", "checklist", "easy"),
        _minimal_record("t2", "Prompt 2", "guide", "medium"),
        _minimal_record("t3", "Prompt 3", "checklist", "hard"),
    ]
    val_records = [
        _minimal_record("v1", "Val 1", "itinerary", "easy"),
        _minimal_record("v2", "Val 2", "checklist", "medium"),
    ]
    _write_split(splits_dir / "train.jsonl", train_records)
    _write_split(splits_dir / "val.jsonl", val_records)

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pocketguide.train.prepare_sft_data",
            "--splits_dir",
            str(splits_dir),
            "--out_dir",
            str(out_dir),
            "--fixed_prompts_out",
            str(fixed_path),
            "--seed",
            "42",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)

    train_sft = out_dir / "train_sft.jsonl"
    val_sft = out_dir / "val_sft.jsonl"
    assert train_sft.exists(), "train_sft.jsonl should exist"
    assert val_sft.exists(), "val_sft.jsonl should exist"
    assert fixed_path.exists(), "fixed suite should exist"

    for path in (train_sft, val_sft):
        with open(path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            obj = json.loads(line)
            assert "id" in obj and "messages" in obj and "target" in obj and "metadata" in obj
            assert isinstance(obj["messages"], list)
            assert len(obj["messages"]) >= 2
            target = obj["target"]
            assert isinstance(target, str)
            parsed = json.loads(target)
            assert isinstance(parsed, dict)
            assert "metadata" in obj
            meta = obj["metadata"]
            assert "source_split" in meta

    with open(fixed_path, encoding="utf-8") as f:
        fixed_lines = [l.strip() for l in f if l.strip()]
    assert len(fixed_lines) <= 20
    for line in fixed_lines:
        obj = json.loads(line)
        for key in ("id", "prompt", "payload_type", "category", "difficulty", "region_tags"):
            assert key in obj


def test_prepare_sft_data_deterministic(tmp_path: Path) -> None:
    """Running twice with same seed yields same first N ids in SFT output."""
    splits_dir = tmp_path / "splits"
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    fixed1 = tmp_path / "fixed1.jsonl"
    fixed2 = tmp_path / "fixed2.jsonl"

    train_records = [_minimal_record(f"t{i}", f"P{i}", "checklist", "easy") for i in range(8)]
    val_records = [_minimal_record(f"v{i}", f"V{i}", "guide", "medium") for i in range(4)]
    _write_split(splits_dir / "train.jsonl", train_records)
    _write_split(splits_dir / "val.jsonl", val_records)

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    def run(out_dir: Path, fixed_out: Path) -> None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pocketguide.train.prepare_sft_data",
                "--splits_dir",
                str(splits_dir),
                "--out_dir",
                str(out_dir),
                "--fixed_prompts_out",
                str(fixed_out),
                "--seed",
                "42",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            check=True,
            timeout=30,
        )

    run(out1, fixed1)
    run(out2, fixed2)

    with open(out1 / "train_sft.jsonl", encoding="utf-8") as f:
        ids1 = [json.loads(l)["id"] for l in f if l.strip()]
    with open(out2 / "train_sft.jsonl", encoding="utf-8") as f:
        ids2 = [json.loads(l)["id"] for l in f if l.strip()]
    assert ids1 == ids2, "Same seed should produce same ordering"

    fixed_ids1 = [json.loads(l)["id"] for l in open(fixed1, encoding="utf-8") if l.strip()]
    fixed_ids2 = [json.loads(l)["id"] for l in open(fixed2, encoding="utf-8") if l.strip()]
    assert fixed_ids1 == fixed_ids2, "Fixed suite should be deterministic"


def test_prepare_sft_data_max_samples(tmp_path: Path) -> None:
    """max_train_samples and max_val_samples cap counts deterministically."""
    splits_dir = tmp_path / "splits"
    out_dir = tmp_path / "sft"
    fixed_path = tmp_path / "fixed20.jsonl"
    train_records = [_minimal_record(f"t{i}", f"P{i}") for i in range(10)]
    val_records = [_minimal_record(f"v{i}", f"V{i}") for i in range(5)]
    _write_split(splits_dir / "train.jsonl", train_records)
    _write_split(splits_dir / "val.jsonl", val_records)

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pocketguide.train.prepare_sft_data",
            "--splits_dir",
            str(splits_dir),
            "--out_dir",
            str(out_dir),
            "--fixed_prompts_out",
            str(fixed_path),
            "--seed",
            "42",
            "--max_train_samples",
            "3",
            "--max_val_samples",
            "2",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=True,
        timeout=30,
    )

    with open(out_dir / "train_sft.jsonl", encoding="utf-8") as f:
        train_count = sum(1 for l in f if l.strip())
    with open(out_dir / "val_sft.jsonl", encoding="utf-8") as f:
        val_count = sum(1 for l in f if l.strip())
    assert train_count == 3
    assert val_count == 2


def test_fixed_suite_small_dataset(tmp_path: Path) -> None:
    """When dataset has fewer than 20 records, fixed suite has as many as available."""
    splits_dir = tmp_path / "splits"
    out_dir = tmp_path / "sft"
    fixed_path = tmp_path / "fixed20.jsonl"
    train_records = [_minimal_record("t1", "P1")]
    val_records = [_minimal_record("v1", "V1")]
    _write_split(splits_dir / "train.jsonl", train_records)
    _write_split(splits_dir / "val.jsonl", val_records)

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pocketguide.train.prepare_sft_data",
            "--splits_dir",
            str(splits_dir),
            "--out_dir",
            str(out_dir),
            "--fixed_prompts_out",
            str(fixed_path),
            "--seed",
            "42",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    with open(fixed_path, encoding="utf-8") as f:
        fixed_lines = [l for l in f if l.strip()]
    assert len(fixed_lines) <= 20
    assert len(fixed_lines) >= 1
