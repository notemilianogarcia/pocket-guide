"""Tests for JSON schema validation."""

import json
from pathlib import Path

import jsonschema
import pytest


@pytest.fixture
def schema_path():
    """Return path to response envelope schema."""
    return (
        Path(__file__).parent.parent
        / "src"
        / "pocketguide"
        / "data"
        / "schemas"
        / "v0"
        / "response_envelope.schema.json"
    )


@pytest.fixture
def schema(schema_path):
    """Load response envelope schema."""
    return json.loads(schema_path.read_text())


@pytest.fixture
def fixtures_dir():
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


def test_schema_file_exists(schema_path):
    """Verify schema file exists."""
    assert schema_path.exists(), f"Schema file not found at {schema_path}"


def test_schema_structure(schema):
    """Verify schema has expected structure."""
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    assert schema.get("type") == "object"
    assert "required" in schema
    assert "properties" in schema


def test_valid_itinerary_fixture(schema, fixtures_dir):
    """Validate itinerary fixture against schema."""
    fixture_path = fixtures_dir / "envelope_valid_itinerary.json"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

    data = json.loads(fixture_path.read_text())
    # Should not raise
    jsonschema.validate(data, schema)


def test_valid_checklist_fixture(schema, fixtures_dir):
    """Validate checklist fixture against schema."""
    fixture_path = fixtures_dir / "envelope_valid_checklist.json"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

    data = json.loads(fixture_path.read_text())
    # Should not raise
    jsonschema.validate(data, schema)


def test_invalid_missing_summary(schema):
    """Validation fails when summary is missing."""
    invalid = {
        "assumptions": [],
        "uncertainty_notes": "",
        "next_steps": [],
        "verification_steps": [],
        "payload_type": "itinerary",
        "payload": {},
    }
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(invalid, schema)
    assert "summary" in str(exc_info.value)


def test_invalid_payload_type_not_in_enum(schema):
    """Validation fails when payload_type is not in enum."""
    invalid = {
        "summary": "Test",
        "assumptions": [],
        "uncertainty_notes": "",
        "next_steps": [],
        "verification_steps": [],
        "payload_type": "invalid_type",
        "payload": {},
    }
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(invalid, schema)
    assert "enum" in str(exc_info.value).lower()


def test_invalid_additional_properties_at_top_level(schema):
    """Validation fails when unknown properties exist at top level."""
    invalid = {
        "summary": "Test",
        "assumptions": [],
        "uncertainty_notes": "",
        "next_steps": [],
        "verification_steps": [],
        "payload_type": "itinerary",
        "payload": {},
        "unknown_field": "should_fail",
    }
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(invalid, schema)
    assert "additional" in str(exc_info.value).lower()


def test_valid_with_metadata(schema):
    """Validation passes when metadata is included."""
    valid = {
        "summary": "Test response",
        "assumptions": ["Assumption 1"],
        "uncertainty_notes": "Some uncertainty",
        "next_steps": ["Next step"],
        "verification_steps": ["Verify this"],
        "payload_type": "checklist",
        "payload": {},
        "metadata": {"model": "gpt-4", "version": "1.0"},
    }
    # Should not raise
    jsonschema.validate(valid, schema)


def test_valid_empty_arrays(schema):
    """Validation passes with empty arrays for assumptions, next_steps, verification_steps."""
    valid = {
        "summary": "Test",
        "assumptions": [],
        "uncertainty_notes": "",
        "next_steps": [],
        "verification_steps": [],
        "payload_type": "procedure",
        "payload": {},
    }
    # Should not raise
    jsonschema.validate(valid, schema)


def test_invalid_empty_string_summary(schema):
    """Validation fails when summary is empty string."""
    invalid = {
        "summary": "",
        "assumptions": [],
        "uncertainty_notes": "",
        "next_steps": [],
        "verification_steps": [],
        "payload_type": "itinerary",
        "payload": {},
    }
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(invalid, schema)
    assert "minLength" in str(exc_info.value)


def test_invalid_assumption_with_empty_string(schema):
    """Validation fails when assumption array contains empty string."""
    invalid = {
        "summary": "Test",
        "assumptions": ["Valid assumption", ""],
        "uncertainty_notes": "",
        "next_steps": [],
        "verification_steps": [],
        "payload_type": "itinerary",
        "payload": {},
    }
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(invalid, schema)
    assert "minLength" in str(exc_info.value)


def test_valid_decision_tree_payload_type(schema):
    """Validation passes for decision_tree payload_type."""
    valid = {
        "summary": "Decision tree for visa selection",
        "assumptions": [],
        "uncertainty_notes": "",
        "next_steps": [],
        "verification_steps": [],
        "payload_type": "decision_tree",
        "payload": {"tree": "structure"},
    }
    # Should not raise
    jsonschema.validate(valid, schema)


def test_payload_accepts_any_object(schema):
    """Payload can contain any object structure (schema defined separately)."""
    valid = {
        "summary": "Test",
        "assumptions": [],
        "uncertainty_notes": "",
        "next_steps": [],
        "verification_steps": [],
        "payload_type": "itinerary",
        "payload": {"complex": {"nested": {"structure": [1, 2, 3]}}},
    }
    # Should not raise
    jsonschema.validate(valid, schema)
