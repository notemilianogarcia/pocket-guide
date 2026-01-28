"""Tests for parsing and validation engine."""

import json

from pocketguide.eval.parsing import (
    parse_and_validate,
)


class TestStrictJsonParsing:
    """Test strict JSON parsing."""

    def test_valid_json_parses(self):
        """Valid JSON should parse successfully."""
        response = json.dumps(
            {
                "summary": "Test",
                "assumptions": [],
                "uncertainty_notes": "",
                "next_steps": [],
                "verification_steps": [],
                "payload_type": "procedure",
                "payload": {"title": "Test", "steps": [{"step": 1, "instruction": "Do"}]},
            }
        )
        result = parse_and_validate(response, strict_json=True)
        assert result.success
        assert result.data is not None
        assert result.error is None

    def test_invalid_json_fails_strict(self):
        """Invalid JSON should fail in strict mode."""
        response = '{"summary": "Test", invalid}'
        result = parse_and_validate(response, strict_json=True)
        assert not result.success
        assert result.error is not None
        assert result.error.error_type == "json_parse"

    def test_json_with_preamble_fails_strict(self):
        """JSON with preamble should fail in strict mode."""
        response = "Here is the response:\n" + json.dumps(
            {
                "summary": "Test",
                "assumptions": [],
                "uncertainty_notes": "",
                "next_steps": [],
                "verification_steps": [],
                "payload_type": "procedure",
                "payload": {"title": "Test", "steps": [{"step": 1, "instruction": "Do"}]},
            }
        )
        result = parse_and_validate(response, strict_json=True)
        assert not result.success
        assert result.error.error_type == "json_parse"


class TestLenientJsonParsing:
    """Test lenient JSON parsing with fallback."""

    def test_json_in_code_fence_markdown(self):
        """JSON in markdown code fence should parse leniently."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "procedure",
            "payload": {"title": "Test", "steps": [{"step": 1, "instruction": "Do"}]},
        }
        response = f"Here's the result:\n```json\n{json.dumps(data)}\n```"
        result = parse_and_validate(response, strict_json=False)
        assert result.success
        assert result.data is not None

    def test_json_in_plain_code_fence(self):
        """JSON in plain code fence should parse leniently."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "procedure",
            "payload": {"title": "Test", "steps": [{"step": 1, "instruction": "Do"}]},
        }
        response = f"Result:\n```\n{json.dumps(data)}\n```"
        result = parse_and_validate(response, strict_json=False)
        assert result.success

    def test_json_with_preamble_lenient(self):
        """JSON with preamble should parse leniently."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "procedure",
            "payload": {"title": "Test", "steps": [{"step": 1, "instruction": "Do"}]},
        }
        response = f"Based on your request:\n{json.dumps(data)}\nThis is helpful."
        result = parse_and_validate(response, strict_json=False)
        assert result.success

    def test_invalid_json_fails_lenient(self):
        """Completely invalid JSON should fail even in lenient mode."""
        response = "No JSON here at all, just text with no braces or brackets"
        result = parse_and_validate(response, strict_json=False)
        assert not result.success
        assert result.error.error_type == "json_parse"


class TestEnvelopeValidation:
    """Test envelope schema validation."""

    def test_missing_required_summary(self):
        """Envelope without summary should fail."""
        data = {
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "procedure",
            "payload": {"title": "Test", "steps": [{"step": 1, "instruction": "Do"}]},
        }
        response = json.dumps(data)
        result = parse_and_validate(response, strict_json=True)
        assert not result.success
        assert result.error.error_type == "envelope_schema"

    def test_invalid_payload_type_enum(self):
        """Invalid payload_type should fail."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "invalid_type",
            "payload": {},
        }
        response = json.dumps(data)
        result = parse_and_validate(response, strict_json=True)
        assert not result.success
        assert result.error.error_type == "envelope_schema"

    def test_additional_properties_rejected(self):
        """Additional envelope properties should be rejected."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "procedure",
            "payload": {"title": "Test", "steps": [{"step": 1, "instruction": "Do"}]},
            "extra_field": "should_fail",
        }
        response = json.dumps(data)
        result = parse_and_validate(response, strict_json=True)
        assert not result.success
        assert result.error.error_type == "envelope_schema"

    def test_valid_envelope_with_metadata(self):
        """Valid envelope with metadata should pass envelope validation."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "procedure",
            "payload": {"title": "Test", "steps": [{"step": 1, "instruction": "Do"}]},
            "metadata": {"model": "test"},
        }
        response = json.dumps(data)
        result = parse_and_validate(response, strict_json=True)
        # Will pass envelope but may fail payload validation depending on content
        if result.success:
            assert result.error is None
        else:
            assert "payload" in result.error.error_type


class TestPayloadValidation:
    """Test payload schema validation."""

    def test_valid_itinerary_payload(self):
        """Valid itinerary payload should pass."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "itinerary",
            "payload": {
                "title": "Trip",
                "trip_days": [
                    {
                        "day": 1,
                        "summary": "Day 1",
                        "items": [{"time_block": "morning", "activity": "Wake up"}],
                    }
                ],
            },
        }
        response = json.dumps(data)
        result = parse_and_validate(response, strict_json=True)
        assert result.success
        assert result.data is not None

    def test_valid_checklist_payload(self):
        """Valid checklist payload should pass."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "checklist",
            "payload": {
                "title": "Checklist",
                "groups": [
                    {
                        "name": "Group",
                        "items": [{"text": "Item"}],
                    }
                ],
            },
        }
        response = json.dumps(data)
        result = parse_and_validate(response, strict_json=True)
        assert result.success

    def test_valid_decision_tree_payload(self):
        """Valid decision tree payload should pass."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "decision_tree",
            "payload": {
                "title": "Tree",
                "nodes": [{"id": "n1", "type": "outcome", "text": "Done"}],
                "edges": [{"from": "n1", "to": "n1"}],
            },
        }
        response = json.dumps(data)
        result = parse_and_validate(response, strict_json=True)
        assert result.success

    def test_valid_procedure_payload(self):
        """Valid procedure payload should pass."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "procedure",
            "payload": {
                "title": "Procedure",
                "steps": [{"step": 1, "instruction": "Do something"}],
            },
        }
        response = json.dumps(data)
        result = parse_and_validate(response, strict_json=True)
        assert result.success

    def test_invalid_itinerary_missing_title(self):
        """Itinerary without title should fail."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "itinerary",
            "payload": {
                "trip_days": [
                    {
                        "day": 1,
                        "summary": "Day 1",
                        "items": [{"time_block": "morning", "activity": "Wake"}],
                    }
                ],
            },
        }
        response = json.dumps(data)
        result = parse_and_validate(response, strict_json=True)
        assert not result.success
        assert result.error.error_type == "payload_schema"

    def test_invalid_procedure_missing_steps(self):
        """Procedure without steps should fail."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "procedure",
            "payload": {"title": "Procedure"},
        }
        response = json.dumps(data)
        result = parse_and_validate(response, strict_json=True)
        assert not result.success
        assert result.error.error_type == "payload_schema"

    def test_question_node_without_options(self):
        """Decision tree question node without options should fail."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "decision_tree",
            "payload": {
                "title": "Tree",
                "nodes": [{"id": "q1", "type": "question", "text": "Question?"}],
                "edges": [{"from": "q1", "to": "q1"}],
            },
        }
        response = json.dumps(data)
        result = parse_and_validate(response, strict_json=True)
        assert not result.success
        assert result.error.error_type == "payload_schema"


class TestStructuredErrorHandling:
    """Test structured error objects and guidance."""

    def test_json_parse_error_has_guidance(self):
        """JSON parse errors should include helpful guidance."""
        response = '{"invalid":'
        result = parse_and_validate(response, strict_json=True)
        assert not result.success
        assert result.error.guidance
        assert any("JSON" in g for g in result.error.guidance)

    def test_envelope_error_has_guidance(self):
        """Envelope validation errors should include guidance."""
        data = {
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "procedure",
            "payload": {"title": "Test", "steps": [{"step": 1, "instruction": "Do"}]},
        }
        response = json.dumps(data)
        result = parse_and_validate(response, strict_json=True)
        assert not result.success
        assert result.error.guidance
        assert any("summary" in g.lower() for g in result.error.guidance)

    def test_payload_error_has_guidance(self):
        """Payload validation errors should include guidance."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "procedure",
            "payload": {"title": "Procedure"},
        }
        response = json.dumps(data)
        result = parse_and_validate(response, strict_json=True)
        assert not result.success
        assert result.error.guidance
        assert any("payload" in g.lower() or "required" in g.lower() for g in result.error.guidance)

    def test_error_string_formatting(self):
        """Error should format nicely as string."""
        response = '{"bad":'
        result = parse_and_validate(response, strict_json=True)
        error_str = str(result.error)
        assert "[json_parse]" in error_str
        assert "Guidance:" in error_str

    def test_parse_result_success_string(self):
        """Successful parse result should have nice string representation."""
        data = {
            "summary": "Test",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "procedure",
            "payload": {"title": "Test", "steps": [{"step": 1, "instruction": "Do"}]},
        }
        response = json.dumps(data)
        result = parse_and_validate(response, strict_json=True)
        assert result.success
        assert "successful" in str(result).lower()


class TestEndToEndScenarios:
    """End-to-end parsing scenarios."""

    def test_full_valid_response_flow(self):
        """Complete valid response should parse successfully."""
        response = json.dumps(
            {
                "summary": "5-day Japan trip",
                "assumptions": ["Budget $50/day"],
                "uncertainty_notes": "Prices may vary",
                "next_steps": ["Book flights"],
                "verification_steps": ["Check passport"],
                "payload_type": "itinerary",
                "payload": {
                    "title": "Japan Trip",
                    "trip_days": [
                        {
                            "day": 1,
                            "summary": "Tokyo arrival",
                            "items": [
                                {
                                    "time_block": "afternoon",
                                    "activity": "Arrive at airport",
                                    "transport": {"mode": "flight", "details": "ANA"},
                                }
                            ],
                        }
                    ],
                    "timezone": "Asia/Tokyo",
                },
            }
        )
        result = parse_and_validate(response, strict_json=False)
        assert result.success
        assert result.data["payload_type"] == "itinerary"

    def test_response_with_markdown_wrapping_lenient(self):
        """Model response wrapped in markdown should parse leniently."""
        data = {
            "summary": "Tokyo itinerary",
            "assumptions": [],
            "uncertainty_notes": "",
            "next_steps": [],
            "verification_steps": [],
            "payload_type": "itinerary",
            "payload": {
                "title": "Trip",
                "trip_days": [
                    {
                        "day": 1,
                        "summary": "Day 1",
                        "items": [{"time_block": "morning", "activity": "Wake"}],
                    }
                ],
            },
        }
        response = (
            "Here's your itinerary:\n\n```json\n" + json.dumps(data) + "\n```\n\nEnjoy your trip!"
        )
        result = parse_and_validate(response, strict_json=False)
        assert result.success

    def test_malformed_response_provides_guidance(self):
        """Malformed response should provide actionable guidance."""
        response = (
            "I'll provide the response in JSON format:\n"
            '{"summary": "Test", "payload_type": "procedure"}'
        )
        result = parse_and_validate(response, strict_json=True)
        assert not result.success
        assert result.error.guidance
        # In strict mode, fails at JSON parse due to preamble
