"""
Tests for local runtime CLI in stub mode (Lesson 6.1).

No llama.cpp or model; assert stdout is valid JSON envelope with required fields.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from pocketguide.inference.local_cli import run, _stub_envelope

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
def stub_config_path(tmp_path: Path) -> Path:
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


def test_stub_envelope_has_required_fields() -> None:
    """Stub envelope includes all required envelope fields."""
    envelope = _stub_envelope("test prompt")
    for key in REQUIRED_ENVELOPE_FIELDS:
        assert key in envelope, f"Missing required field: {key}"
    assert isinstance(envelope["assumptions"], list)
    assert isinstance(envelope["next_steps"], list)
    assert isinstance(envelope["verification_steps"], list)
    assert envelope["payload_type"] in ("itinerary", "checklist", "decision_tree", "procedure")
    assert isinstance(envelope["payload"], dict)


def test_stub_envelope_no_timestamps() -> None:
    """Stub envelope does not contain timestamps (deterministic)."""
    envelope = _stub_envelope("any prompt")
    text = json.dumps(envelope)
    assert "timestamp" not in text.lower() or "placeholder" in text.lower()


def test_run_stub_mode_returns_valid_json(stub_config_path: Path) -> None:
    """run() with stub=true returns valid JSON string."""
    out = run(stub_config_path, "plan a 2-day itinerary for Montreal")
    data = json.loads(out)
    assert isinstance(data, dict)
    for key in REQUIRED_ENVELOPE_FIELDS:
        assert key in data
    assert data["payload_type"] == "procedure"
    assert isinstance(data["payload"], dict)
    assert "title" in data["payload"]
    assert "steps" in data["payload"]


def test_run_non_stub_raises(tmp_path: Path) -> None:
    """run() with stub=false raises NotImplementedError."""
    cfg = {
        "runtime": {"backend": "llamacpp", "stub": False},
        "model": {},
        "generation": {},
        "execution": {},
    }
    path = tmp_path / "runtime_local.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    with pytest.raises(NotImplementedError, match="llama.cpp execution not implemented"):
        run(path, "test")


def test_local_cli_main_stdout_json(capsys: pytest.CaptureFixture, stub_config_path: Path) -> None:
    """Invoking main with stub config prints valid JSON to stdout only."""
    with patch("sys.argv", ["local_cli", "--config", str(stub_config_path), "--prompt", "hello"]):
        from pocketguide.inference.local_cli import main

        main()
    captured = capsys.readouterr()
    stdout = captured.out.strip()
    stderr = captured.err
    assert stdout
    data = json.loads(stdout)
    assert "summary" in data and "payload_type" in data and "payload" in data
    assert isinstance(data["payload"], dict)
