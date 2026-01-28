"""Smoke tests for CLI inference."""


from pocketguide.inference.cli import format_response, generate_stub_response


def test_generate_stub_response_returns_required_keys():
    """Test that stub response contains all required keys."""
    prompt = "What are visa requirements for Japan?"
    response = generate_stub_response(prompt)

    assert isinstance(response, dict)
    assert "summary" in response
    assert "assumptions" in response
    assert "next_steps" in response


def test_generate_stub_response_is_deterministic():
    """Test that same prompt generates same response."""
    prompt = "Budget for Thailand trip"
    response1 = generate_stub_response(prompt)
    response2 = generate_stub_response(prompt)

    assert response1 == response2


def test_format_response_includes_required_sections():
    """Test that formatted response includes all required section headers."""
    response = {
        "summary": "Test summary",
        "assumptions": "Test assumptions",
        "next_steps": "Test next steps",
    }

    formatted = format_response(response)

    assert "Summary:" in formatted
    assert "Assumptions:" in formatted
    assert "Next steps:" in formatted
    assert "Test summary" in formatted
    assert "Test assumptions" in formatted
    assert "Test next steps" in formatted


def test_format_response_preserves_order():
    """Test that sections appear in the correct order."""
    response = {
        "summary": "SUMMARY_CONTENT",
        "assumptions": "ASSUMPTIONS_CONTENT",
        "next_steps": "NEXT_STEPS_CONTENT",
    }

    formatted = format_response(response)

    summary_idx = formatted.find("Summary:")
    assumptions_idx = formatted.find("Assumptions:")
    next_steps_idx = formatted.find("Next steps:")

    assert summary_idx < assumptions_idx < next_steps_idx
