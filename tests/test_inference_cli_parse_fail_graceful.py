"""
Tests for graceful parse failure: CLI returns structured error envelope (Lesson 6.3).

Mock local backend to return invalid JSON; assert CLI prints valid error envelope, no crash.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

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
    """Config with runtime.stub: true (we mock the backend anyway)."""
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


def test_parse_fail_returns_structured_error_envelope(stub_runtime_config: Path) -> None:
    """When backend returns invalid JSON, CLI prints valid error envelope (no crash)."""
    invalid_raw = "This is not JSON at all. Just plain text."

    def fake_get_raw_text(runtime: str, prompt: str, config_path: object, project_root: object) -> tuple[str, dict]:
        return invalid_raw, {"latency_ms": 0.0, "tokens_generated": None, "command_used": "[mock]"}

    with patch("pocketguide.inference.cli.get_raw_text", side_effect=fake_get_raw_text):
        from pocketguide.inference.cli import main

        import io
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            sys.argv = [
                "cli",
                "--runtime",
                "local",
                "--runtime_config",
                str(stub_runtime_config),
                "--prompt",
                "test",
            ]
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            out = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    data = json.loads(out)
    for key in REQUIRED_ENVELOPE_FIELDS:
        assert key in data, f"Error envelope missing field: {key}"
    assert data["summary"] == "Parsing/validation failed"
    assert data["payload_type"] == "procedure"
    assert isinstance(data["payload"], dict)
    assert "title" in data["payload"]
    assert "steps" in data["payload"]
    # Error details in payload.extra
    assert data["payload"].get("extra", {}).get("error_type") is not None or "Error" in str(data["payload"])


def test_parse_fail_envelope_passes_validation(stub_runtime_config: Path) -> None:
    """Error envelope is valid JSON and passes envelope schema validation."""
    invalid_raw = "not json"

    def fake_get_raw_text(runtime: str, prompt: str, config_path: object, project_root: object) -> tuple[str, dict]:
        return invalid_raw, {"latency_ms": 0.0, "command_used": "[mock]"}

    with patch("pocketguide.inference.cli.get_raw_text", side_effect=fake_get_raw_text):
        from pocketguide.inference.cli import main

        import io
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            sys.argv = ["cli", "--runtime", "local", "--runtime_config", str(stub_runtime_config), "--prompt", "x"]
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            out = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    parse_result = parse_and_validate(out, strict_json=True)
    assert parse_result.success, f"Error envelope should validate: {parse_result.error}"
    assert parse_result.data is not None
    assert parse_result.data["summary"] == "Parsing/validation failed"
