"""
Tests for Lesson 7.3 v2 training config guardrails.

Verifies that the one-change guardrail detects multiple training param changes
and accepts the passing case (only data paths + single one_change_key differ).
No model download or network calls.
"""

import pytest
from pathlib import Path

from pocketguide.train.train import (
    load_config,
    validate_experiment_one_change,
    ALLOWED_DIFF_KEYS,
)


def _write_config(path: Path, **overrides: object) -> None:
    import yaml

    base = {
        "base_model_id": "meta-llama/Llama-2-7b-hf",
        "data": {
            "train_path": "data/processed/splits/v1/train.jsonl",
            "val_path": "data/processed/splits/v1/val.jsonl",
            "train_sft_path": "data/processed/sft/v1/train_sft.jsonl",
            "val_sft_path": "data/processed/sft/v1/val_sft.jsonl",
            "max_seq_len": 1024,
        },
        "output": {"runs_dir": "runs/train"},
        "training": {"seed": 42, "batch_size": 1, "grad_accum_steps": 16, "lr": 2.0e-4, "num_epochs": 1},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "auto"},
        "runtime": {"device": "auto", "precision": "auto", "gradient_checkpointing": True},
        "logging": {"log_every_steps": 10, "enable_wandb": False},
    }
    for k, v in overrides.items():
        if "." in k:
            top, rest = k.split(".", 1)
            if top not in base or not isinstance(base[top], dict):
                base[top] = {}
            base[top][rest] = v
        elif k == "data" and isinstance(v, dict):
            base["data"] = {**base["data"], **v}
        else:
            base[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(base, f, default_flow_style=False)


def test_guardrail_passes_when_only_lr_and_data_paths_differ(tmp_path: Path) -> None:
    """Only lr and data paths differ: guardrail should pass."""
    v1_yaml = tmp_path / "v1" / "config.yaml"
    v2_yaml = tmp_path / "v2" / "config.yaml"
    _write_config(v1_yaml)
    _write_config(
        v2_yaml,
        **{
            "data": {
                "train_path": "data/processed/splits/v2/train.jsonl",
                "val_path": "data/processed/splits/v2/val.jsonl",
                "train_sft_path": "data/processed/sft/v2/train_sft.jsonl",
                "val_sft_path": "data/processed/sft/v2/val_sft.jsonl",
                "max_seq_len": 1024,
            },
            "training.lr": 1.5e-4,
        },
    )
    cfg_v1 = load_config(v1_yaml)
    cfg_v2 = load_config(v2_yaml)
    validate_experiment_one_change(cfg_v2, cfg_v1, "training.lr")


def test_guardrail_fails_when_more_than_one_training_param_changed(tmp_path: Path) -> None:
    """Both lr and num_epochs changed: guardrail should raise."""
    v1_yaml = tmp_path / "v1" / "config.yaml"
    v2_yaml = tmp_path / "v2" / "config.yaml"
    _write_config(v1_yaml)
    _write_config(
        v2_yaml,
        **{
            "data": {
                "train_path": "data/processed/splits/v2/train.jsonl",
                "val_path": "data/processed/splits/v2/val.jsonl",
                "train_sft_path": "data/processed/sft/v2/train_sft.jsonl",
                "val_sft_path": "data/processed/sft/v2/val_sft.jsonl",
                "max_seq_len": 1024,
            },
            "training.lr": 1.5e-4,
            "training.num_epochs": 2,
        },
    )
    cfg_v1 = load_config(v1_yaml)
    cfg_v2 = load_config(v2_yaml)
    with pytest.raises(ValueError, match="one-change guardrail|differs from v1"):
        validate_experiment_one_change(cfg_v2, cfg_v1, "training.lr")


def test_guardrail_fails_when_lora_r_changed_with_lr_as_one_change(tmp_path: Path) -> None:
    """Only lr is declared as one_change_key but lora.r also differs: should raise."""
    v1_yaml = tmp_path / "v1" / "config.yaml"
    v2_yaml = tmp_path / "v2" / "config.yaml"
    _write_config(v1_yaml)
    _write_config(
        v2_yaml,
        **{
            "data": {
                "train_path": "data/processed/splits/v2/train.jsonl",
                "val_path": "data/processed/splits/v2/val.jsonl",
                "train_sft_path": "data/processed/sft/v2/train_sft.jsonl",
                "val_sft_path": "data/processed/sft/v2/val_sft.jsonl",
                "max_seq_len": 1024,
            },
            "training.lr": 1.5e-4,
            "lora.r": 8,
        },
    )
    cfg_v1 = load_config(v1_yaml)
    cfg_v2 = load_config(v2_yaml)
    with pytest.raises(ValueError, match="one-change guardrail|lora"):
        validate_experiment_one_change(cfg_v2, cfg_v1, "training.lr")


def test_allowed_diff_keys_include_data_paths() -> None:
    """Allowed keys include the four data path keys."""
    assert "data.train_path" in ALLOWED_DIFF_KEYS
    assert "data.val_path" in ALLOWED_DIFF_KEYS
    assert "data.train_sft_path" in ALLOWED_DIFF_KEYS
    assert "data.val_sft_path" in ALLOWED_DIFF_KEYS
