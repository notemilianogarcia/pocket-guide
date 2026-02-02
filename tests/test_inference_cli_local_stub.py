"""
Tests for unified CLI with --runtime local in stub mode (Lesson 6.3).

No llama.cpp required. Assert output is valid JSON and passes envelope validation.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from pocketguide.eval.parsing import parse_and_validate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_ENVELOPE_FIELDS = [
    "summary",
    "assumptions",
    "uncertainty_notes",
    "next_steps",
    "verification_steps",
    "payload_type",
    "payload",
]


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


def test_cli_runtime_local_stub_returns_valid_envelope(stub_runtime_config: Path) -> None:
    """Unified CLI --runtime local with stub=true returns valid envelope JSON on stdout."""
    env = {"PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pocketguide.inference.cli",
            "--runtime",
            "local",
            "--runtime_config",
            str(stub_runtime_config),
            "--prompt",
            "plan a 2-day itinerary for Montreal",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    data = json.loads(result.stdout)
    for key in REQUIRED_ENVELOPE_FIELDS:
        assert key in data, f"Missing required field: {key}"
    assert data["payload_type"] == "procedure"
    assert isinstance(data["payload"], dict)
    assert "title" in data["payload"] and "steps" in data["payload"]


def test_cli_runtime_local_stub_passes_parse_and_validate(stub_runtime_config: Path) -> None:
    """Output from --runtime local stub passes parse_and_validate (envelope + payload)."""
    env = {"PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pocketguide.inference.cli",
            "--runtime",
            "local",
            "--runtime_config",
            str(stub_runtime_config),
            "--prompt",
            "test prompt",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    parse_result = parse_and_validate(result.stdout, strict_json=True)
    assert parse_result.success, parse_result.error
    assert parse_result.data is not None
    assert parse_result.data["payload_type"] == "procedure"
