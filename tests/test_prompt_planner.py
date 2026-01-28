"""Tests for prompt planner."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml
from pocketguide.data_generation.plan_prompts import (
    compute_stats,
    generate_prompt_plan,
    load_spec,
    main,
)


def create_minimal_spec(total_examples: int = 10) -> dict:
    """Create a minimal spec for testing.

    Args:
        total_examples: Number of examples to generate

    Returns:
        Minimal spec dictionary
    """
    return {
        "version": "v1",
        "seed": 42,
        "total_examples": total_examples,
        "payload_type_distribution": {
            "itinerary": 0.40,
            "checklist": 0.30,
            "decision_tree": 0.20,
            "procedure": 0.10,
        },
        "category_distribution": {
            "transport": 0.30,
            "itinerary_planning": 0.30,
            "entry_requirements": 0.40,
        },
        "difficulty_levels": {
            "easy": 0.50,
            "medium": 0.30,
            "hard": 0.20,
        },
        "region_tags": {
            "NA": 0.50,
            "EU": 0.50,
        },
        "prompt_constraints": {
            "avoid_realtime_facts": True,
            "ambiguity_rate": 0.30,
            "constraint_rate": 0.50,
            "include_constraints": ["budget_range", "date_constraints", "preferences", "duration"],
        },
        "locations": {
            "NA": [
                {"country": "United States", "cities": ["New York", "San Francisco"]},
                {"country": "Canada", "cities": ["Toronto", "Vancouver"]},
            ],
            "EU": [
                {"country": "France", "cities": ["Paris", "Lyon"]},
                {"country": "Germany", "cities": ["Berlin", "Munich"]},
            ],
        },
        "user_profiles": [
            "Solo traveler, mid-30s, experienced with international travel",
            "Family of 4 (2 adults, 2 children ages 8 and 12)",
            "Couple in their 60s, retired, prefer comfort over adventure",
        ],
    }


def test_load_spec_valid():
    """Test loading a valid spec file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        spec_data = create_minimal_spec()
        yaml.dump(spec_data, f)
        spec_path = Path(f.name)

    try:
        spec = load_spec(spec_path)
        assert spec["version"] == "v1"
        assert spec["seed"] == 42
        assert spec["total_examples"] == 10
    finally:
        spec_path.unlink()


def test_load_spec_missing_file():
    """Test that loading missing spec raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Spec file not found"):
        load_spec(Path("/nonexistent/spec.yaml"))


def test_load_spec_missing_keys():
    """Test that loading spec with missing keys raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        # Missing required keys
        yaml.dump({"version": "v1", "seed": 42}, f)
        spec_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Spec missing required keys"):
            load_spec(spec_path)
    finally:
        spec_path.unlink()


def test_generate_prompt_plan_determinism():
    """Test that prompt plan generation is deterministic."""
    spec = create_minimal_spec(total_examples=10)
    templates_dir = Path("data/prompts/teacher/v1")

    # Generate plan twice
    plan1 = generate_prompt_plan(spec, templates_dir)
    plan2 = generate_prompt_plan(spec, templates_dir)

    # Should have same number of records
    assert len(plan1) == 10
    assert len(plan2) == 10

    # First 3 records should be identical
    for i in range(3):
        assert plan1[i]["id"] == plan2[i]["id"], f"IDs differ at index {i}"
        assert plan1[i]["prompt"] == plan2[i]["prompt"], f"Prompts differ at index {i}"
        assert plan1[i]["payload_type"] == plan2[i]["payload_type"]
        assert plan1[i]["category"] == plan2[i]["category"]
        assert plan1[i]["difficulty"] == plan2[i]["difficulty"]


def test_generate_prompt_plan_unique_ids():
    """Test that generated IDs are unique."""
    spec = create_minimal_spec(total_examples=20)
    templates_dir = Path("data/prompts/teacher/v1")

    plan = generate_prompt_plan(spec, templates_dir)

    # Extract IDs
    ids = [record["id"] for record in plan]

    # All IDs should be unique
    assert len(ids) == len(set(ids)), "IDs are not unique"


def test_generate_prompt_plan_fields():
    """Test that generated records have all required fields."""
    spec = create_minimal_spec(total_examples=5)
    templates_dir = Path("data/prompts/teacher/v1")

    plan = generate_prompt_plan(spec, templates_dir)

    required_fields = [
        "id",
        "payload_type",
        "category",
        "difficulty",
        "region_tags",
        "template_version",
        "template_name",
        "prompt",
        "location",
        "created_by",
        "seed",
    ]

    for record in plan:
        for field in required_fields:
            assert field in record, f"Record missing field: {field}"

        # Check nested location fields
        assert "country" in record["location"]
        assert "city" in record["location"]
        assert "region" in record["location"]

        # Check values
        assert record["template_version"] == "teacher/v1"
        assert record["created_by"] == "plan_prompts"
        assert record["seed"] == 42
        assert isinstance(record["region_tags"], list)


def test_compute_stats_totals():
    """Test that stats sum to total examples."""
    spec = create_minimal_spec(total_examples=20)
    templates_dir = Path("data/prompts/teacher/v1")

    plan = generate_prompt_plan(spec, templates_dir)
    stats = compute_stats(plan)

    # Total should match
    assert stats["total_examples"] == 20

    # Counts by dimension should sum to total
    assert sum(stats["by_payload_type"].values()) == 20
    assert sum(stats["by_category"].values()) == 20
    assert sum(stats["by_difficulty"].values()) == 20
    assert sum(stats["by_region"].values()) == 20


def test_main_generates_files():
    """Test that main() generates all required output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create minimal spec
        spec_data = create_minimal_spec(total_examples=10)
        spec_path = tmpdir_path / "test_spec.yaml"
        with open(spec_path, "w") as f:
            yaml.dump(spec_data, f)

        # Run main
        out_dir = tmpdir_path / "output"
        main(spec_path=str(spec_path), out_dir=str(out_dir))

        # Check that all files exist
        plan_path = out_dir / "prompt_plan_v1.jsonl"
        stats_path = out_dir / "prompt_plan_v1_stats.json"
        manifest_path = out_dir / "prompt_plan_v1_manifest.json"

        assert plan_path.exists(), "prompt_plan_v1.jsonl not created"
        assert stats_path.exists(), "prompt_plan_v1_stats.json not created"
        assert manifest_path.exists(), "prompt_plan_v1_manifest.json not created"

        # Check plan content
        with open(plan_path) as f:
            lines = f.readlines()
            assert len(lines) == 10, f"Expected 10 lines, got {len(lines)}"

            # Parse first line to check structure
            first_record = json.loads(lines[0])
            assert "id" in first_record
            assert "prompt" in first_record
            assert "payload_type" in first_record

        # Check stats content
        with open(stats_path) as f:
            stats = json.load(f)
            assert stats["total_examples"] == 10
            assert "by_payload_type" in stats
            assert "by_category" in stats
            assert "by_difficulty" in stats
            assert "by_region" in stats

        # Check manifest content
        with open(manifest_path) as f:
            manifest = json.load(f)
            assert "version" in manifest
            assert "generated_at" in manifest
            assert "spec_hash" in manifest
            assert "template_hashes" in manifest
            assert "counts" in manifest
            assert manifest["counts"]["total_examples"] == 10


def test_prompt_template_rendering():
    """Test that prompts are properly rendered with context."""
    spec = create_minimal_spec(total_examples=5)
    templates_dir = Path("data/prompts/teacher/v1")

    plan = generate_prompt_plan(spec, templates_dir)

    for record in plan:
        prompt = record["prompt"]

        # Check that placeholders are filled
        assert "{user_profile}" not in prompt, "user_profile not replaced"
        assert "{trip_context}" not in prompt, "trip_context not replaced"
        assert "{country}" not in prompt, "country not replaced"
        assert "{city}" not in prompt, "city not replaced"
        assert "{category}" not in prompt, "category not replaced"
        assert "{difficulty}" not in prompt, "difficulty not replaced"

        # Check that location is valid
        location = record["location"]
        assert location["country"] in prompt
        assert location["city"] in prompt
