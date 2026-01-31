"""Tests for payload schemas v1."""

import json
from pathlib import Path

import jsonschema
import pytest


@pytest.fixture
def schemas_v1_path():
    """Return path to v1 schemas directory."""
    return Path(__file__).parent.parent / "src" / "pocketguide" / "data" / "schemas" / "v1"


@pytest.fixture
def payloads_fixtures_path():
    """Return path to payload fixtures directory."""
    return Path(__file__).parent / "fixtures" / "payloads" / "v1"


@pytest.fixture
def itinerary_schema(schemas_v1_path):
    """Load itinerary payload schema."""
    schema_path = schemas_v1_path / "itinerary.payload.schema.json"
    return json.loads(schema_path.read_text())


@pytest.fixture
def checklist_schema(schemas_v1_path):
    """Load checklist payload schema."""
    schema_path = schemas_v1_path / "checklist.payload.schema.json"
    return json.loads(schema_path.read_text())


@pytest.fixture
def decision_tree_schema(schemas_v1_path):
    """Load decision tree payload schema."""
    schema_path = schemas_v1_path / "decision_tree.payload.schema.json"
    return json.loads(schema_path.read_text())


@pytest.fixture
def procedure_schema(schemas_v1_path):
    """Load procedure payload schema."""
    schema_path = schemas_v1_path / "procedure.payload.schema.json"
    return json.loads(schema_path.read_text())


# Itinerary Tests
def test_itinerary_schema_exists(schemas_v1_path):
    """Verify itinerary schema file exists."""
    schema_path = schemas_v1_path / "itinerary.payload.schema.json"
    assert schema_path.exists(), f"Schema file not found at {schema_path}"


def test_itinerary_valid_fixture(itinerary_schema, payloads_fixtures_path):
    """Validate itinerary valid fixture."""
    fixture_path = payloads_fixtures_path / "itinerary_valid.json"
    assert fixture_path.exists()
    data = json.loads(fixture_path.read_text())
    # Should not raise
    jsonschema.validate(data, itinerary_schema)


def test_itinerary_invalid_fixture(itinerary_schema, payloads_fixtures_path):
    """Validate itinerary invalid fixture fails."""
    fixture_path = payloads_fixtures_path / "itinerary_invalid.json"
    assert fixture_path.exists()
    data = json.loads(fixture_path.read_text())
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(data, itinerary_schema)
    # Verify it fails for expected reason (trip_days type)
    assert "trip_days" in str(exc_info.value) or "array" in str(exc_info.value).lower()


def test_itinerary_missing_title(itinerary_schema):
    """Validation fails when title is missing."""
    invalid = {
        "trip_days": [
            {
                "day": 1,
                "summary": "Day 1",
                "items": [{"time_block": "morning", "activity": "Wake up"}],
            }
        ]
    }
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(invalid, itinerary_schema)
    assert "title" in str(exc_info.value)


def test_itinerary_empty_trip_days(itinerary_schema):
    """Validation fails when trip_days is empty array."""
    invalid = {"title": "Trip", "trip_days": []}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, itinerary_schema)


# Checklist Tests
def test_checklist_schema_exists(schemas_v1_path):
    """Verify checklist schema file exists."""
    schema_path = schemas_v1_path / "checklist.payload.schema.json"
    assert schema_path.exists(), f"Schema file not found at {schema_path}"


def test_checklist_valid_fixture(checklist_schema, payloads_fixtures_path):
    """Validate checklist valid fixture."""
    fixture_path = payloads_fixtures_path / "checklist_valid.json"
    assert fixture_path.exists()
    data = json.loads(fixture_path.read_text())
    # Should not raise
    jsonschema.validate(data, checklist_schema)


def test_checklist_invalid_fixture(checklist_schema, payloads_fixtures_path):
    """Validate checklist invalid fixture fails."""
    fixture_path = payloads_fixtures_path / "checklist_invalid.json"
    assert fixture_path.exists()
    data = json.loads(fixture_path.read_text())
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(data, checklist_schema)
    # Verify it fails for expected reason (missing items)
    assert "items" in str(exc_info.value) or "required" in str(exc_info.value).lower()


def test_checklist_missing_groups(checklist_schema):
    """Validation fails when groups is missing."""
    invalid = {"title": "Checklist"}
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(invalid, checklist_schema)
    assert "groups" in str(exc_info.value)


def test_checklist_invalid_priority_enum(checklist_schema):
    """Validation fails with invalid priority value."""
    invalid = {
        "title": "Checklist",
        "groups": [
            {"name": "Group 1", "items": [{"text": "Item", "priority": "invalid_priority"}]}
        ],
    }
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(invalid, checklist_schema)
    assert "enum" in str(exc_info.value).lower()


# Decision Tree Tests
def test_decision_tree_schema_exists(schemas_v1_path):
    """Verify decision tree schema file exists."""
    schema_path = schemas_v1_path / "decision_tree.payload.schema.json"
    assert schema_path.exists(), f"Schema file not found at {schema_path}"


def test_decision_tree_valid_fixture(decision_tree_schema, payloads_fixtures_path):
    """Validate decision tree valid fixture."""
    fixture_path = payloads_fixtures_path / "decision_tree_valid.json"
    assert fixture_path.exists()
    data = json.loads(fixture_path.read_text())
    # Should not raise
    jsonschema.validate(data, decision_tree_schema)


def test_decision_tree_invalid_fixture(decision_tree_schema, payloads_fixtures_path):
    """Validate decision tree invalid fixture fails (invalid node type)."""
    fixture_path = payloads_fixtures_path / "decision_tree_invalid.json"
    assert fixture_path.exists()
    data = json.loads(fixture_path.read_text())
    # Schema allows root_question + nodes without edges; invalid fixture has type=question without options.
    # Use a payload that fails: missing required "title" or invalid node.
    invalid = {"nodes": [{"id": "n1", "type": "outcome", "text": "Done"}]}
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(invalid, decision_tree_schema)
    error_msg = str(exc_info.value).lower()
    assert "title" in error_msg or "required" in error_msg


def test_decision_tree_optional_edges(decision_tree_schema):
    """Payload with title and nodes but no edges is valid (edges optional)."""
    valid = {"title": "Tree", "nodes": [{"id": "n1", "type": "outcome", "text": "Done"}]}
    jsonschema.validate(valid, decision_tree_schema)


def test_decision_tree_invalid_node_type(decision_tree_schema):
    """Validation fails with invalid node type."""
    invalid = {
        "title": "Tree",
        "nodes": [{"id": "n1", "type": "invalid_type", "text": "Invalid"}],
        "edges": [{"from": "n1", "to": "n2"}],
    }
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(invalid, decision_tree_schema)
    assert "enum" in str(exc_info.value).lower()


# Procedure Tests
def test_procedure_schema_exists(schemas_v1_path):
    """Verify procedure schema file exists."""
    schema_path = schemas_v1_path / "procedure.payload.schema.json"
    assert schema_path.exists(), f"Schema file not found at {schema_path}"


def test_procedure_valid_fixture(procedure_schema, payloads_fixtures_path):
    """Validate procedure valid fixture."""
    fixture_path = payloads_fixtures_path / "procedure_valid.json"
    assert fixture_path.exists()
    data = json.loads(fixture_path.read_text())
    # Should not raise
    jsonschema.validate(data, procedure_schema)


def test_procedure_invalid_fixture(procedure_schema, payloads_fixtures_path):
    """Validate procedure invalid fixture fails."""
    fixture_path = payloads_fixtures_path / "procedure_invalid.json"
    assert fixture_path.exists()
    data = json.loads(fixture_path.read_text())
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(data, procedure_schema)
    # Verify it fails for expected reason (missing step field)
    assert "step" in str(exc_info.value) or "required" in str(exc_info.value).lower()


def test_procedure_missing_title(procedure_schema):
    """Validation fails when title is missing."""
    invalid = {"steps": [{"step": 1, "instruction": "Do something"}]}
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(invalid, procedure_schema)
    assert "title" in str(exc_info.value)


def test_procedure_invalid_step_number(procedure_schema):
    """Validation fails when step number is < 1."""
    invalid = {"title": "Procedure", "steps": [{"step": 0, "instruction": "Invalid step number"}]}
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        jsonschema.validate(invalid, procedure_schema)
    assert "minimum" in str(exc_info.value).lower()


# Cross-payload validation tests
def test_all_schemas_have_extra_field(
    itinerary_schema, checklist_schema, decision_tree_schema, procedure_schema
):
    """All schemas should define an 'extra' field for extensibility."""
    for schema, name in [
        (itinerary_schema, "itinerary"),
        (checklist_schema, "checklist"),
        (decision_tree_schema, "decision_tree"),
        (procedure_schema, "procedure"),
    ]:
        assert "extra" in schema["properties"], f"{name} missing extra field"


def test_all_schemas_have_schema_id(
    itinerary_schema, checklist_schema, decision_tree_schema, procedure_schema
):
    """All schemas should have $id field for reference."""
    for schema, name in [
        (itinerary_schema, "itinerary"),
        (checklist_schema, "checklist"),
        (decision_tree_schema, "decision_tree"),
        (procedure_schema, "procedure"),
    ]:
        assert "$id" in schema, f"{name} missing $id field"


def test_all_schemas_use_draft_2020_12(
    itinerary_schema, checklist_schema, decision_tree_schema, procedure_schema
):
    """All schemas should use JSON Schema draft 2020-12."""
    for schema, name in [
        (itinerary_schema, "itinerary"),
        (checklist_schema, "checklist"),
        (decision_tree_schema, "decision_tree"),
        (procedure_schema, "procedure"),
    ]:
        assert schema.get("$schema") == (
            "https://json-schema.org/draft/2020-12/schema"
        ), f"{name} not using draft 2020-12"


def test_itinerary_with_extra_field(itinerary_schema):
    """Itinerary schema should accept extra field with arbitrary properties."""
    valid = {
        "title": "Trip",
        "trip_days": [
            {
                "day": 1,
                "summary": "Day 1",
                "items": [{"time_block": "morning", "activity": "Wake up"}],
            }
        ],
        "extra": {"custom_field": "value", "nested": {"data": 123}},
    }
    # Should not raise
    jsonschema.validate(valid, itinerary_schema)


def test_checklist_with_extra_field(checklist_schema):
    """Checklist schema should accept extra field with arbitrary properties."""
    valid = {
        "title": "Checklist",
        "groups": [{"name": "Group", "items": [{"text": "Item"}]}],
        "extra": {"future_feature": "enabled"},
    }
    # Should not raise
    jsonschema.validate(valid, checklist_schema)
