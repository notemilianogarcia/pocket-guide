"""Tests for metrics_v0 module."""


from pocketguide.eval.metrics_v0 import (
    aggregate_metrics,
    check_required_fields,
    detect_uncertainty_markers,
    lenient_json_extract_and_parse,
    strict_json_parse,
)


def test_strict_json_parse_valid():
    """Test strict JSON parsing with valid JSON."""
    text = '{"name": "Alice", "age": 30}'
    ok, parsed, error = strict_json_parse(text)
    assert ok is True
    assert parsed == {"name": "Alice", "age": 30}
    assert error is None


def test_strict_json_parse_valid_array():
    """Test strict JSON parsing with valid array."""
    text = '[1, 2, 3]'
    ok, parsed, error = strict_json_parse(text)
    assert ok is True
    assert parsed == [1, 2, 3]
    assert error is None


def test_strict_json_parse_with_whitespace():
    """Test strict JSON parsing with leading/trailing whitespace."""
    text = '  \n{"key": "value"}  \n'
    ok, parsed, error = strict_json_parse(text)
    assert ok is True
    assert parsed == {"key": "value"}


def test_strict_json_parse_invalid():
    """Test strict JSON parsing with invalid JSON."""
    text = 'This is not JSON'
    ok, parsed, error = strict_json_parse(text)
    assert ok is False
    assert parsed is None
    assert "decode error" in error.lower()


def test_strict_json_parse_with_preamble():
    """Test strict JSON parsing fails when there's preamble text."""
    text = 'Here is the result: {"key": "value"}'
    ok, parsed, error = strict_json_parse(text)
    assert ok is False  # Strict mode should fail


def test_lenient_json_extract_valid_strict():
    """Test lenient extraction with valid strict JSON."""
    text = '{"name": "Bob", "score": 85}'
    ok, parsed, error = lenient_json_extract_and_parse(text)
    assert ok is True
    assert parsed == {"name": "Bob", "score": 85}


def test_lenient_json_extract_code_fence():
    """Test lenient extraction from markdown code fence."""
    text = '''Here is the data:
```json
{"city": "Paris", "country": "France"}
```
Hope this helps!'''
    ok, parsed, error = lenient_json_extract_and_parse(text)
    assert ok is True
    assert parsed == {"city": "Paris", "country": "France"}


def test_lenient_json_extract_code_fence_no_lang():
    """Test lenient extraction from code fence without language."""
    text = '''```
{"status": "success"}
```'''
    ok, parsed, error = lenient_json_extract_and_parse(text)
    assert ok is True
    assert parsed == {"status": "success"}


def test_lenient_json_extract_with_preamble():
    """Test lenient extraction with preamble and postamble."""
    text = 'The answer is: {"result": 42} and that is correct.'
    ok, parsed, error = lenient_json_extract_and_parse(text)
    assert ok is True
    assert parsed == {"result": 42}


def test_lenient_json_extract_nested_braces():
    """Test lenient extraction with nested braces."""
    text = 'Response: {"user": {"name": "Alice", "age": 30}, "status": "active"}'
    ok, parsed, error = lenient_json_extract_and_parse(text)
    assert ok is True
    assert parsed["user"]["name"] == "Alice"
    assert parsed["status"] == "active"


def test_lenient_json_extract_array():
    """Test lenient extraction with array."""
    text = 'The items are: [1, 2, 3, 4]'
    ok, parsed, error = lenient_json_extract_and_parse(text)
    assert ok is True
    assert parsed == [1, 2, 3, 4]


def test_lenient_json_extract_invalid():
    """Test lenient extraction fails when no valid JSON found."""
    text = 'This is just plain text without any JSON structure'
    ok, parsed, error = lenient_json_extract_and_parse(text)
    assert ok is False
    assert parsed is None


def test_lenient_json_extract_empty():
    """Test lenient extraction with empty input."""
    ok, parsed, error = lenient_json_extract_and_parse("")
    assert ok is False
    assert parsed is None


def test_check_required_fields_simple():
    """Test required fields check with simple fields."""
    parsed = {"name": "Alice", "age": 30, "city": "Paris"}
    required = ["name", "age"]
    ok, missing = check_required_fields(parsed, required)
    assert ok is True
    assert missing == []


def test_check_required_fields_missing():
    """Test required fields check with missing fields."""
    parsed = {"name": "Bob"}
    required = ["name", "age", "city"]
    ok, missing = check_required_fields(parsed, required)
    assert ok is False
    assert set(missing) == {"age", "city"}


def test_check_required_fields_dotted_path():
    """Test required fields check with dotted paths."""
    parsed = {"user": {"name": "Alice", "email": "alice@example.com"}, "status": "active"}
    required = ["user.name", "user.email", "status"]
    ok, missing = check_required_fields(parsed, required)
    assert ok is True
    assert missing == []


def test_check_required_fields_dotted_path_missing():
    """Test required fields check with missing nested field."""
    parsed = {"user": {"name": "Bob"}, "status": "active"}
    required = ["user.name", "user.email", "status"]
    ok, missing = check_required_fields(parsed, required)
    assert ok is False
    assert missing == ["user.email"]


def test_check_required_fields_empty_value():
    """Test that empty strings are considered missing."""
    parsed = {"name": "", "age": 30}
    required = ["name", "age"]
    ok, missing = check_required_fields(parsed, required)
    assert ok is False
    assert missing == ["name"]


def test_check_required_fields_null_value():
    """Test that null values are considered missing."""
    parsed = {"name": None, "age": 30}
    required = ["name", "age"]
    ok, missing = check_required_fields(parsed, required)
    assert ok is False
    assert missing == ["name"]


def test_check_required_fields_empty_list():
    """Test that empty lists are considered missing."""
    parsed = {"items": [], "count": 0}
    required = ["items", "count"]
    ok, missing = check_required_fields(parsed, required)
    assert ok is False
    assert missing == ["items"]


def test_check_required_fields_non_dict():
    """Test required fields check with non-dict input."""
    parsed = [1, 2, 3]
    required = ["name", "age"]
    ok, missing = check_required_fields(parsed, required)
    assert ok is False
    assert missing == required


def test_detect_uncertainty_markers_assumptions():
    """Test detection of assumption markers."""
    text = "I assume you are traveling in summer. This might work if you have a valid passport."
    result = detect_uncertainty_markers(text)
    assert result["has_assumptions_marker"] is True
    assert len(result["matched_phrases"]["assumptions"]) > 0


def test_detect_uncertainty_markers_verification():
    """Test detection of verification markers."""
    text = "Please verify this with the official embassy website. Check the latest rules before traveling."
    result = detect_uncertainty_markers(text)
    assert result["has_verification_marker"] is True
    assert len(result["matched_phrases"]["verification"]) > 0


def test_detect_uncertainty_markers_clarifying():
    """Test detection of clarifying questions."""
    text = "Which country are you traveling from? What is your nationality?"
    result = detect_uncertainty_markers(text)
    assert result["has_clarifying_questions"] is True
    assert len(result["matched_phrases"]["clarifying"]) > 0


def test_detect_uncertainty_markers_multiple():
    """Test detection of multiple marker types."""
    text = """I assume you have a valid passport. However, you should verify this information
    with the official government website. What is your departure date?"""
    result = detect_uncertainty_markers(text)
    assert result["has_assumptions_marker"] is True
    assert result["has_verification_marker"] is True
    assert result["has_clarifying_questions"] is True


def test_detect_uncertainty_markers_none():
    """Test with text containing no uncertainty markers."""
    text = "You need a passport and visa. The fee is $50. Processing takes 5 days."
    result = detect_uncertainty_markers(text)
    assert result["has_assumptions_marker"] is False
    assert result["has_verification_marker"] is False
    assert result["has_clarifying_questions"] is False


def test_detect_uncertainty_markers_max_matches():
    """Test that matched phrases are limited to 3 per category."""
    text = "I assume A. I assume B. I assume C. I assume D. I assume E."
    result = detect_uncertainty_markers(text)
    assert result["has_assumptions_marker"] is True
    assert len(result["matched_phrases"]["assumptions"]) <= 3


def test_aggregate_metrics_empty():
    """Test aggregation with empty records."""
    metrics = aggregate_metrics([])
    assert metrics["overall"]["n"] == 0
    assert "by_suite" in metrics
    assert "definitions" in metrics


def test_aggregate_metrics_basic():
    """Test basic metric aggregation."""
    records = [
        {
            "id": "1",
            "suite": "test",
            "checks": {
                "strict_json_ok": True,
                "lenient_json_ok": True,
                "required_fields_ok": True,
                "uncertainty": {
                    "has_assumptions_marker": False,
                    "has_verification_marker": False,
                    "has_clarifying_questions": False,
                },
            },
            "timing": {"latency_s": 1.0, "tokens_per_s": 10.0},
        },
        {
            "id": "2",
            "suite": "test",
            "checks": {
                "strict_json_ok": False,
                "lenient_json_ok": True,
                "required_fields_ok": False,
                "uncertainty": {
                    "has_assumptions_marker": True,
                    "has_verification_marker": False,
                    "has_clarifying_questions": False,
                },
            },
            "timing": {"latency_s": 2.0, "tokens_per_s": 5.0},
        },
    ]

    metrics = aggregate_metrics(records)

    # Check overall metrics
    assert metrics["overall"]["n"] == 2
    assert metrics["overall"]["strict_json_parse_rate"] == 0.5
    assert metrics["overall"]["lenient_json_parse_rate"] == 1.0
    assert metrics["overall"]["required_fields_rate"] == 0.5
    assert metrics["overall"]["assumptions_marker_rate"] == 0.5
    assert metrics["overall"]["avg_latency_s"] == 1.5
    assert metrics["overall"]["p50_latency_s"] == 1.5

    # Check by-suite metrics
    assert "test" in metrics["by_suite"]
    assert metrics["by_suite"]["test"]["n"] == 2


def test_aggregate_metrics_multiple_suites():
    """Test aggregation with multiple suites."""
    records = [
        {
            "id": "1",
            "suite": "suite_a",
            "checks": {
                "strict_json_ok": True,
                "lenient_json_ok": True,
                "required_fields_ok": None,
                "uncertainty": {
                    "has_assumptions_marker": False,
                    "has_verification_marker": False,
                    "has_clarifying_questions": False,
                },
            },
            "timing": {"latency_s": 1.0, "tokens_per_s": 10.0},
        },
        {
            "id": "2",
            "suite": "suite_b",
            "checks": {
                "strict_json_ok": False,
                "lenient_json_ok": False,
                "required_fields_ok": None,
                "uncertainty": {
                    "has_assumptions_marker": True,
                    "has_verification_marker": True,
                    "has_clarifying_questions": True,
                },
            },
            "timing": {"latency_s": 2.0, "tokens_per_s": 5.0},
        },
    ]

    metrics = aggregate_metrics(records)

    assert metrics["overall"]["n"] == 2
    assert "suite_a" in metrics["by_suite"]
    assert "suite_b" in metrics["by_suite"]
    assert metrics["by_suite"]["suite_a"]["n"] == 1
    assert metrics["by_suite"]["suite_b"]["n"] == 1


def test_aggregate_metrics_missing_timing():
    """Test aggregation handles missing timing data gracefully."""
    records = [
        {
            "id": "1",
            "suite": "test",
            "checks": {
                "strict_json_ok": True,
                "lenient_json_ok": True,
                "required_fields_ok": None,
                "uncertainty": {
                    "has_assumptions_marker": False,
                    "has_verification_marker": False,
                    "has_clarifying_questions": False,
                },
            },
            "timing": {},  # Empty timing
        },
        {
            "id": "2",
            "suite": "test",
            "checks": {
                "strict_json_ok": True,
                "lenient_json_ok": True,
                "required_fields_ok": None,
                "uncertainty": {
                    "has_assumptions_marker": False,
                    "has_verification_marker": False,
                    "has_clarifying_questions": False,
                },
            },
            # Missing timing entirely
        },
    ]

    metrics = aggregate_metrics(records)

    assert metrics["overall"]["avg_latency_s"] is None
    assert metrics["overall"]["p50_latency_s"] is None


def test_aggregate_metrics_percentiles():
    """Test percentile calculations."""
    records = [
        {
            "id": str(i),
            "suite": "test",
            "checks": {
                "strict_json_ok": True,
                "lenient_json_ok": True,
                "required_fields_ok": None,
                "uncertainty": {
                    "has_assumptions_marker": False,
                    "has_verification_marker": False,
                    "has_clarifying_questions": False,
                },
            },
            "timing": {"latency_s": float(i), "tokens_per_s": float(10 - i)},
        }
        for i in range(1, 11)
    ]

    metrics = aggregate_metrics(records)

    # P50 of [1,2,3,4,5,6,7,8,9,10] should be 5.5
    assert 5.0 <= metrics["overall"]["p50_latency_s"] <= 6.0
    # P90 should be around 9
    assert 8.5 <= metrics["overall"]["p90_latency_s"] <= 10.0
