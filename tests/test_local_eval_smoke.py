"""
Tests for local eval (Lesson 6.4). No llama.cpp or model required.

Mock local runtime to return deterministic valid/invalid outputs;
assert local_metrics.json exists with per_suite, overall, parse counters.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from pocketguide.eval.local_eval import run_local_eval

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def stub_runtime_config(tmp_path: Path) -> Path:
    """Config with runtime.stub: true."""
    cfg = {
        "runtime": {"backend": "llamacpp", "stub": True},
        "model": {"gguf_path": "models/gguf/PLACEHOLDER.gguf", "context_length": 4096},
        "generation": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 512},
        "execution": {"llamacpp_bin": None, "extra_args": []},
    }
    path = tmp_path / "runtime_local.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


@pytest.fixture
def mini_suite(tmp_path: Path) -> Path:
    """Minimal suite: 3 prompts (id, prompt)."""
    path = tmp_path / "mini.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"id": "a", "prompt": "Plan a day in Paris."}\n')
        f.write('{"id": "b", "prompt": "Visa for Japan?"}\n')
        f.write('{"id": "c", "prompt": "Budget for Thailand."}\n')
    return path


def test_local_eval_produces_metrics_and_outputs(
    tmp_path: Path,
    stub_runtime_config: Path,
    mini_suite: Path,
) -> None:
    """run_local_eval with stub config produces local_metrics.json and optional local_outputs.jsonl."""
    out_dir = tmp_path / "runs" / "eval"
    run_dir = run_local_eval(
        runtime_config_path=stub_runtime_config,
        suite_paths=[mini_suite],
        out_dir=out_dir,
        run_id="test_run_001",
        project_root=tmp_path,
        write_outputs_jsonl=True,
    )
    metrics_path = run_dir / "local_metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "run_id" in metrics
    assert metrics["run_id"] == "test_run_001"
    assert "per_suite" in metrics
    assert "overall" in metrics
    assert "mini" in metrics["per_suite"]
    per_suite = metrics["per_suite"]["mini"]
    assert per_suite["n"] == 3
    assert "strict_parse_rate" in per_suite
    assert "lenient_parse_rate" in per_suite
    assert "envelope_valid_rate" in per_suite
    assert "payload_valid_rate" in per_suite
    assert "avg_latency_ms" in per_suite
    overall = metrics["overall"]
    assert overall["n"] == 3
    assert overall["strict_parse_rate"] == per_suite["strict_parse_rate"]
    assert "tokens_per_sec_note" in metrics

    outputs_path = run_dir / "local_outputs.jsonl"
    assert outputs_path.exists()
    lines = outputs_path.read_text().strip().split("\n")
    assert len(lines) == 3


def test_local_eval_mock_invalid_outputs_parse_rates(
    tmp_path: Path,
    stub_runtime_config: Path,
    mini_suite: Path,
) -> None:
    """When mock returns some invalid JSON, strict/lenient parse rates reflect it."""
    valid_envelope = '{"summary":"x","assumptions":[],"uncertainty_notes":"","next_steps":[],"verification_steps":[],"payload_type":"procedure","payload":{"title":"t","steps":[{"step":1,"instruction":"i"}]}}'

    call_count = [0]

    def mock_run_llamacpp(prompt: str, runtime_cfg: dict, project_root: Path = None) -> dict:
        call_count[0] += 1
        if call_count[0] % 2 == 1:
            return {
                "raw_text_output": valid_envelope,
                "latency_ms": 100.0,
                "tokens_generated": 50,
                "command_used": "[mock]",
            }
        return {
            "raw_text_output": "not valid json at all",
            "latency_ms": 50.0,
            "tokens_generated": None,
            "command_used": "[mock]",
        }

    with patch("pocketguide.eval.local_eval.run_llamacpp", side_effect=mock_run_llamacpp):
        run_dir = run_local_eval(
            runtime_config_path=stub_runtime_config,
            suite_paths=[mini_suite],
            out_dir=tmp_path / "runs" / "eval",
            run_id="test_parse_rates",
            project_root=tmp_path,
            write_outputs_jsonl=False,
        )
    metrics = json.loads((run_dir / "local_metrics.json").read_text())
    overall = metrics["overall"]
    assert overall["n"] == 3
    # 2 valid (strict+lenient ok), 1 invalid (lenient may still fail)
    assert overall["strict_parse_rate"] < 1.0
    assert overall["strict_parse_rate"] >= 0.0
    assert "lenient_parse_rate" in overall
    assert "envelope_valid_rate" in overall
    assert "payload_valid_rate" in overall
