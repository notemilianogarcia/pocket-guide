"""
Tests for Base Model Report Generator v0.
"""

import json

import pytest
from pocketguide.eval.report_base_v0 import (
    _assign_taxonomy_category,
    _build_failure_example,
    _build_metrics_table,
    _build_summary,
    _build_taxonomy,
    _format_metric_row,
    _select_curated_failures,
    _truncate,
    generate_report,
)


def test_truncate():
    """Test text truncation."""
    short = "hello"
    assert _truncate(short, 10) == "hello"

    long = "a" * 100
    result = _truncate(long, 20)
    assert len(result) == 20
    assert result.endswith("...")


def test_format_metric_row():
    """Test metrics table row formatting."""
    data = {
        "n": 10,
        "strict_json_parse_rate": 0.5,
        "lenient_json_parse_rate": 0.8,
        "required_fields_rate": None,
        "assumptions_marker_rate": 1.0,
        "verification_marker_rate": 0.3,
        "clarifying_questions_rate": 0.0,
        "avg_latency_s": 2.5,
        "p90_latency_s": 4.2,
    }
    row = _format_metric_row("test", data)
    assert "| test |" in row
    assert "| 10 |" in row
    assert "| 50.0 |" in row  # strict
    assert "| 80.0 |" in row  # lenient
    assert "| — |" in row  # required_fields_rate is None
    assert "| 100.0 |" in row  # assumptions
    assert "| 2.50 |" in row  # avg latency


def test_build_metrics_table():
    """Test full metrics table generation."""
    metrics = {
        "overall": {
            "n": 20,
            "strict_json_parse_rate": 0.4,
            "lenient_json_parse_rate": 0.7,
        },
        "by_suite": {
            "format": {"n": 10, "strict_json_parse_rate": 0.5},
            "safety": {"n": 10, "strict_json_parse_rate": 0.3},
        },
    }
    table = _build_metrics_table(metrics)
    assert "| Suite | N |" in table
    assert "| **Overall** | 20 |" in table
    assert "| format | 10 |" in table
    assert "| safety | 10 |" in table


def test_build_taxonomy():
    """Test static taxonomy section."""
    taxonomy = _build_taxonomy()
    assert "JSON Format Violations" in taxonomy
    assert "Schema / Required-Field Failures" in taxonomy
    assert "Hallucinated Specificity" in taxonomy
    assert "Missed Constraints" in taxonomy
    assert "Overconfident Answers Under Uncertainty" in taxonomy
    assert "Weak Verification Guidance" in taxonomy
    assert "Safety Overreach or Vagueness" in taxonomy


def test_assign_taxonomy_category_json_fail():
    """Test taxonomy assignment for JSON failures."""
    output = {
        "checks": {"strict_json_ok": False},
        "suite": "format",
    }
    category = _assign_taxonomy_category(output)
    assert category == "JSON Format Violations"


def test_assign_taxonomy_category_required_fields():
    """Test taxonomy assignment for required field failures."""
    output = {
        "checks": {
            "strict_json_ok": True,
            "required_fields_ok": False,
            "missing_fields": ["name"],
        },
        "suite": "format",
    }
    category = _assign_taxonomy_category(output)
    assert category == "Schema / Required-Field Failures"


def test_assign_taxonomy_category_verification():
    """Test taxonomy assignment for verification failures."""
    output = {
        "checks": {
            "strict_json_ok": True,
            "required_fields_ok": True,
            "uncertainty": {"has_verification_marker": False},
        },
        "suite": "uncertainty",
    }
    category = _assign_taxonomy_category(output)
    assert category == "Weak Verification Guidance"


def test_select_curated_failures_empty():
    """Test curated selection with no outputs."""
    result = _select_curated_failures([], [], target_count=10)
    assert result == []


def test_select_curated_failures_manual_ids():
    """Test curated selection with manual overrides."""
    outputs = [
        {"example_id": "fmt_001", "suite": "format", "checks": {}},
        {"example_id": "fmt_002", "suite": "format", "checks": {}},
        {"example_id": "fmt_003", "suite": "format", "checks": {}},
    ]
    manual_ids = ["fmt_002", "fmt_003"]
    result = _select_curated_failures(outputs, manual_ids, target_count=2)
    assert len(result) == 2
    assert result[0]["example_id"] == "fmt_002"
    assert result[1]["example_id"] == "fmt_003"


def test_select_curated_failures_auto_selection():
    """Test automatic selection by severity score."""
    outputs = [
        {
            "example_id": "fmt_001",
            "suite": "format",
            "checks": {"strict_json_ok": True},
        },
        {
            "example_id": "fmt_002",
            "suite": "format",
            "checks": {"strict_json_ok": False},  # score = 3
        },
        {
            "example_id": "fmt_003",
            "suite": "format",
            "checks": {
                "strict_json_ok": False,  # score = 3
                "required_fields_ok": False,  # score += 2 = 5
            },
        },
    ]
    result = _select_curated_failures(outputs, [], target_count=2)
    assert len(result) == 2
    # fmt_003 should be first (higher score)
    assert result[0]["example_id"] == "fmt_003"
    assert result[1]["example_id"] == "fmt_002"


def test_select_curated_failures_diversity():
    """Test diversity across suites."""
    outputs = [
        {
            "example_id": "fmt_001",
            "suite": "format",
            "checks": {"strict_json_ok": False},
        },
        {
            "example_id": "saf_001",
            "suite": "safety",
            "checks": {"strict_json_ok": False},
        },
        {
            "example_id": "fmt_002",
            "suite": "format",
            "checks": {"strict_json_ok": False},
        },
    ]
    result = _select_curated_failures(outputs, [], target_count=2)
    assert len(result) == 2
    # Should prefer diversity: one from format, one from safety
    suites = {r["suite"] for r in result}
    assert len(suites) == 2


def test_build_failure_example():
    """Test formatting of a single failure example."""
    output = {
        "example_id": "fmt_001",
        "suite": "format",
        "prompt": "Format this as JSON",
        "output": '{"name": "test"}',
        "checks": {
            "strict_json_ok": False,
            "required_fields_ok": False,
            "missing_fields": ["age"],
            "uncertainty": {},
        },
    }
    example = _build_failure_example(1, output)
    assert "### Example 1: fmt_001" in example
    assert "**Suite**: format" in example
    assert "**Prompt**:" in example
    assert "Format this as JSON" in example
    assert "**Model Output**:" in example
    assert "**Failed Checks**:" in example
    assert "❌ Strict JSON parse failed" in example
    assert "❌ Missing required fields: age" in example
    assert "**Taxonomy Category**:" in example


def test_build_summary():
    """Test summary section generation."""
    metrics = {
        "overall": {
            "n": 100,
            "strict_json_parse_rate": 0.3,
            "lenient_json_parse_rate": 0.6,
            "required_fields_rate": 0.5,
            "verification_marker_rate": 0.2,
        }
    }
    curated = [
        {"suite": "format", "example_id": "fmt_001"},
        {"suite": "safety", "example_id": "saf_001"},
        {"suite": "uncertainty", "example_id": "unc_001"},
    ]
    summary = _build_summary(metrics, curated)
    assert "Key Findings" in summary
    assert "Next Steps" in summary
    assert "30.0%" in summary  # strict JSON rate
    assert "Milestone 2" in summary


def test_generate_report_missing_files(tmp_path):
    """Test error handling for missing files."""
    run_dir = tmp_path / "nonexistent"
    output_path = tmp_path / "report.md"

    with pytest.raises(FileNotFoundError, match="Missing metrics.json"):
        generate_report(run_dir, output_path)


def test_generate_report_basic(tmp_path):
    """Test full report generation with minimal data."""
    # Create synthetic run directory
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    # Create minimal metrics.json
    metrics = {
        "overall": {
            "n": 5,
            "strict_json_parse_rate": 0.4,
            "lenient_json_parse_rate": 0.8,
            "required_fields_rate": 0.6,
            "assumptions_marker_rate": 1.0,
            "verification_marker_rate": 0.4,
            "clarifying_questions_rate": 0.0,
            "avg_latency_s": 2.5,
            "p90_latency_s": 4.0,
        },
        "by_suite": {
            "format": {
                "n": 3,
                "strict_json_parse_rate": 0.33,
                "lenient_json_parse_rate": 1.0,
            },
            "safety": {
                "n": 2,
                "strict_json_parse_rate": 0.5,
                "lenient_json_parse_rate": 0.5,
            },
        },
        "definitions": {},
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    # Create minimal meta.json
    meta = {
        "model_id": "test-model",
        "revision": "main",
        "device": "cpu",
        "dtype": "float32",
        "seed": 42,
        "generation_config": {"max_new_tokens": 512, "temperature": 0.7},
        "suite_counts": {"format": 3, "safety": 2},
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    # Create minimal base_model_outputs.jsonl
    outputs = [
        {
            "example_id": "fmt_001",
            "suite": "format",
            "prompt": "Format as JSON",
            "output": "not valid json",
            "checks": {
                "strict_json_ok": False,
                "lenient_json_ok": False,
                "required_fields_ok": False,
                "missing_fields": ["name"],
                "uncertainty": {},
            },
            "timing": {"latency_s": 2.5},
        },
        {
            "example_id": "fmt_002",
            "suite": "format",
            "prompt": "Another test",
            "output": '{"name": "test"}',
            "checks": {
                "strict_json_ok": True,
                "lenient_json_ok": True,
                "required_fields_ok": True,
                "missing_fields": [],
                "uncertainty": {},
            },
            "timing": {"latency_s": 1.5},
        },
        {
            "example_id": "saf_001",
            "suite": "safety",
            "prompt": "Safety test",
            "output": "Generic safety response",
            "checks": {
                "strict_json_ok": False,
                "lenient_json_ok": False,
                "uncertainty": {"has_verification_marker": False},
            },
            "timing": {"latency_s": 3.0},
        },
    ]
    with open(run_dir / "base_model_outputs.jsonl", "w") as f:
        for output in outputs:
            f.write(json.dumps(output) + "\n")

    # Generate report
    output_path = tmp_path / "report.md"
    generate_report(run_dir, output_path, overrides_path=None)

    # Verify report exists and has expected structure
    assert output_path.exists()
    report_content = output_path.read_text()

    # Check all required sections
    assert "# Base Model Evaluation Report v0" in report_content
    assert "## Experimental Setup" in report_content
    assert "test-model" in report_content
    assert "## Metrics Summary" in report_content
    assert "| Suite | N |" in report_content
    assert "| **Overall** | 5 |" in report_content
    assert "## Failure Taxonomy v0" in report_content
    assert "JSON Format Violations" in report_content
    assert "## Curated Failure Examples" in report_content
    assert "### Example 1:" in report_content
    assert "## Summary & Next Steps" in report_content
    assert "Key Findings" in report_content


def test_generate_report_with_overrides(tmp_path):
    """Test report generation with manual override IDs."""
    # Create synthetic run directory
    run_dir = tmp_path / "run_002"
    run_dir.mkdir()

    # Minimal files
    metrics = {"overall": {"n": 2}, "by_suite": {}, "definitions": {}}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    meta = {
        "model_id": "test",
        "generation_config": {},
        "suite_counts": {"format": 2},
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    outputs = [
        {
            "example_id": "fmt_001",
            "suite": "format",
            "prompt": "Test 1",
            "output": "Output 1",
            "checks": {"strict_json_ok": False},
        },
        {
            "example_id": "fmt_002",
            "suite": "format",
            "prompt": "Test 2",
            "output": "Output 2",
            "checks": {"strict_json_ok": False},
        },
    ]
    with open(run_dir / "base_model_outputs.jsonl", "w") as f:
        for output in outputs:
            f.write(json.dumps(output) + "\n")

    # Create overrides config
    overrides_path = tmp_path / "overrides.yaml"
    with open(overrides_path, "w") as f:
        f.write("curated_failure_ids:\n")
        f.write("  - fmt_002\n")

    # Generate report
    output_path = tmp_path / "report.md"
    generate_report(run_dir, output_path, overrides_path=overrides_path)

    # Verify fmt_002 is in the report
    report_content = output_path.read_text()
    assert "fmt_002" in report_content


def test_generate_report_fewer_than_10_examples(tmp_path):
    """Test report generation with fewer than 10 examples."""
    run_dir = tmp_path / "run_003"
    run_dir.mkdir()

    metrics = {"overall": {"n": 2}, "by_suite": {}, "definitions": {}}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    meta = {"model_id": "test", "generation_config": {}, "suite_counts": {}}
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    # Only 2 outputs
    outputs = [
        {
            "example_id": "fmt_001",
            "suite": "format",
            "prompt": "Test 1",
            "output": "Output 1",
            "checks": {"strict_json_ok": False},
        },
        {
            "example_id": "fmt_002",
            "suite": "format",
            "prompt": "Test 2",
            "output": "Output 2",
            "checks": {"strict_json_ok": False},
        },
    ]
    with open(run_dir / "base_model_outputs.jsonl", "w") as f:
        for output in outputs:
            f.write(json.dumps(output) + "\n")

    output_path = tmp_path / "report.md"
    generate_report(run_dir, output_path)

    report_content = output_path.read_text()
    # Should have 2 examples, not 10
    assert "The following 2 examples" in report_content
    assert "### Example 1:" in report_content
    assert "### Example 2:" in report_content
    assert "### Example 3:" not in report_content
