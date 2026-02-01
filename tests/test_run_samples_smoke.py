"""
Smoke test for run_samples (Lesson 5.4).

Mocks inference so no model download or real inference; verifies both base and
finetuned paths run, output files are created, and comparison_metrics has expected keys.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

pytest.importorskip("peft")

from pocketguide.train.run_samples import run_samples

# Minimal valid envelope + itinerary for mock output (passes parse_and_validate)
MOCK_VALID_OUTPUT = json.dumps({
    "summary": "Smoke test summary",
    "assumptions": ["Assumption one"],
    "uncertainty_notes": "Verify with official sources.",
    "next_steps": ["Book in advance"],
    "verification_steps": ["Check embassy website"],
    "payload_type": "itinerary",
    "payload": {
        "title": "1-Day Smoke Itinerary",
        "trip_days": [
            {"day": 1, "summary": "Day one", "items": [{"time_block": "09:00", "activity": "Activity"}]}
        ],
    },
})


def _mock_generate_one(*args, **kwargs):
    """Deterministic fake generation result."""
    return {
        "text": MOCK_VALID_OUTPUT,
        "completion_text": MOCK_VALID_OUTPUT,
        "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        "timing": {"latency_s": 0.1, "tokens_per_s": 500.0},
    }


@pytest.fixture
def run_dir_with_config_and_adapter(tmp_path):
    """Create a minimal run_dir with config.yaml and adapter/ for run_samples."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {
        "base_model_id": "meta-llama/Llama-2-7b-hf",
        "data": {"max_seq_len": 256},
        "output": {"runs_dir": "runs/train"},
        "training": {"seed": 42},
        "lora": {},
        "runtime": {"device": "cpu", "precision": "fp32"},
        "logging": {},
    }
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir()
    # Minimal PEFT adapter config so _ensure_adapter passes
    (adapter_dir / "config.json").write_text(
        json.dumps({"base_model_name_or_path": "test", "peft_type": "LORA"})
    )
    return run_dir


@pytest.fixture
def prompts_path(tmp_path):
    """Create a small fixed prompts JSONL."""
    path = tmp_path / "fixed_prompts.jsonl"
    prompts = [
        {"id": "p1", "prompt": "Generate a short itinerary for Tokyo.", "payload_type": "itinerary"},
        {"id": "p2", "prompt": "Generate a decision tree for visa.", "payload_type": "decision_tree"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path


@patch("pocketguide.train.run_samples._run_inference_one", side_effect=_mock_generate_one)
@patch(
    "pocketguide.inference.base_model.load_model_and_tokenizer",
    return_value=(MagicMock(), MagicMock()),
)
def test_run_samples_smoke_base_and_finetuned_paths(
    mock_load: MagicMock,
    mock_inference: MagicMock,
    run_dir_with_config_and_adapter: Path,
    prompts_path: Path,
) -> None:
    """Both base and finetuned inference paths run; output files and metrics created."""
    run_dir = run_dir_with_config_and_adapter
    project_root = run_dir.parent

    # PeftModel.from_pretrained must return a mock model so second loop runs
    mock_peft_model = MagicMock()
    with patch("peft.PeftModel") as MockPeft:
        MockPeft.from_pretrained.return_value = mock_peft_model
        run_samples(run_dir, prompts_path, project_root, seed=42)

    # Inference called for base (2 prompts) + finetuned (2 prompts)
    assert mock_inference.call_count == 4

    samples_dir = run_dir / "samples"
    assert samples_dir.is_dir()
    base_out = samples_dir / "base_outputs.jsonl"
    finetuned_out = samples_dir / "finetuned_outputs.jsonl"
    metrics_path = samples_dir / "comparison_metrics.json"

    assert base_out.exists()
    assert finetuned_out.exists()
    assert metrics_path.exists()

    base_lines = base_out.read_text().strip().splitlines()
    finetuned_lines = finetuned_out.read_text().strip().splitlines()
    assert len(base_lines) == 2
    assert len(finetuned_lines) == 2

    metrics = json.loads(metrics_path.read_text())
    assert "num_prompts" in metrics
    assert metrics["num_prompts"] == 2
    assert "base" in metrics
    assert "finetuned" in metrics
    assert "delta" in metrics
    for key in ("parse_success_rate", "schema_valid_rate", "required_field_presence_rate", "uncertainty_marker_presence_rate"):
        assert key in metrics["base"]
        assert key in metrics["finetuned"]
    assert "avg_latency_ms" in metrics["base"] or "n" in metrics["base"]


def test_run_samples_fails_without_adapter(
    run_dir_with_config_and_adapter: Path,
    prompts_path: Path,
) -> None:
    """run_samples fails loudly when adapter directory is missing."""
    run_dir = run_dir_with_config_and_adapter
    shutil.rmtree(run_dir / "adapter")
    project_root = run_dir.parent

    with patch("pocketguide.inference.base_model.load_model_and_tokenizer", return_value=(MagicMock(), MagicMock())):
        with patch("pocketguide.train.run_samples._run_inference_one", side_effect=_mock_generate_one):
            with pytest.raises(FileNotFoundError, match="Adapter directory not found"):
                run_samples(run_dir, prompts_path, project_root)
