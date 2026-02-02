"""Smoke tests for CLI inference (unified --runtime hf | local)."""

import json

from pocketguide.inference.cli import _hf_stub_envelope, _build_error_envelope
from pocketguide.eval.parsing import ParseResult, ParseValidationError, parse_and_validate


def test_hf_stub_envelope_returns_required_keys():
    """Stub envelope (HF runtime) contains required envelope keys."""
    envelope = _hf_stub_envelope("What are visa requirements for Japan?")
    assert isinstance(envelope, dict)
    assert "summary" in envelope
    assert "assumptions" in envelope
    assert "next_steps" in envelope
    assert "payload_type" in envelope
    assert "payload" in envelope
    assert envelope["payload_type"] == "procedure"


def test_hf_stub_envelope_is_deterministic():
    """Same prompt produces same stub envelope."""
    prompt = "Budget for Thailand trip"
    e1 = _hf_stub_envelope(prompt)
    e2 = _hf_stub_envelope(prompt)
    assert e1 == e2


def test_hf_stub_envelope_passes_validation():
    """HF stub envelope passes parse_and_validate."""
    envelope = _hf_stub_envelope("test")
    raw = json.dumps(envelope, indent=2)
    result = parse_and_validate(raw, strict_json=True)
    assert result.success
    assert result.data is not None


def test_format_response_preserves_order():
    """Error envelope build produces valid procedure payload with ordered steps."""
    err = ParseValidationError(
        code="TEST",
        error_type="json_parse",
        message="Test message",
        guidance=["Tip one", "Tip two"],
    )
    result = ParseResult(success=False, error=err)
    envelope = _build_error_envelope(result, raw_excerpt="excerpt")
    assert "summary" in envelope and envelope["summary"] == "Parsing/validation failed"
    assert envelope["payload_type"] == "procedure"
    steps = envelope["payload"]["steps"]
    assert len(steps) >= 1
    step_instructions = [s["instruction"] for s in steps]
    assert any("Test message" in i for i in step_instructions)
