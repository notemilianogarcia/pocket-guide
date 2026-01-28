"""Metrics v0: Simple automated metrics for PocketGuide evaluation outputs.

Computes JSON parse rates, required field checks, and uncertainty heuristics.
"""

import json
import re
from typing import Any


def strict_json_parse(text: str) -> tuple[bool, dict | list | None, str | None]:
    """Parse text as strict JSON (entire stripped text must be valid JSON).

    Args:
        text: Input text to parse

    Returns:
        Tuple of (success, parsed_object, error_message)
    """
    stripped = text.strip()
    if not stripped:
        return False, None, "Empty input"

    try:
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict | list):
            return False, None, f"Parsed to {type(parsed).__name__}, expected dict or list"
        return True, parsed, None
    except json.JSONDecodeError as e:
        return False, None, f"JSON decode error: {str(e)}"


def lenient_json_extract_and_parse(text: str) -> tuple[bool, dict | list | None, str | None]:
    """Extract and parse JSON from text, handling code fences and prose.

    Attempts to find the first valid JSON object {...} or array [...] in the text.
    Handles common wrappers like markdown code fences.

    Args:
        text: Input text potentially containing JSON

    Returns:
        Tuple of (success, parsed_object, error_message)
    """
    if not text or not text.strip():
        return False, None, "Empty input"

    # First try strict parse
    ok, parsed, _ = strict_json_parse(text)
    if ok:
        return True, parsed, None

    # Remove markdown code fences
    cleaned = text
    # Match ```json ... ``` or ``` ... ```
    code_fence_pattern = r"```(?:json)?\s*(.*?)\s*```"
    matches = re.findall(code_fence_pattern, cleaned, re.DOTALL)
    if matches:
        # Try parsing the first code fence content
        for match in matches:
            ok, parsed, _ = strict_json_parse(match)
            if ok:
                return True, parsed, None

    # Try bracket matching to extract JSON
    # Look for outermost { } or [ ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = cleaned.find(start_char)
        if start_idx == -1:
            continue

        # Simple bracket matching
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(cleaned)):
            char = cleaned[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    # Found matching bracket
                    candidate = cleaned[start_idx : i + 1]
                    ok, parsed, _ = strict_json_parse(candidate)
                    if ok:
                        return True, parsed, None
                    break

    return False, None, "No valid JSON object or array found"


def check_required_fields(parsed: Any, required_fields: list[str]) -> tuple[bool, list[str]]:
    """Check for required fields in parsed JSON using dotted paths.

    Args:
        parsed: Parsed JSON object (should be dict)
        required_fields: List of field paths (e.g., ["name", "user.email"])

    Returns:
        Tuple of (all_present, missing_fields)
    """
    if not isinstance(parsed, dict):
        return False, required_fields  # All fields missing if not a dict

    missing = []

    for field_path in required_fields:
        parts = field_path.split(".")
        current = parsed

        found = True
        for part in parts:
            if not isinstance(current, dict):
                found = False
                break
            if part not in current:
                found = False
                break
            current = current[part]

        # Check if value is present (not None, not empty string, not empty list/dict)
        if found:
            if current is None:
                found = False
            elif isinstance(current, str) and not current.strip():
                found = False
            elif isinstance(current, list | dict) and len(current) == 0:
                found = False

        if not found:
            missing.append(field_path)

    return len(missing) == 0, missing


def detect_uncertainty_markers(text: str) -> dict:
    """Detect uncertainty markers in text using heuristics.

    Detects:
    - Assumption markers (e.g., "I assume", "might", "depends")
    - Verification markers (e.g., "verify", "check official", "confirm with")
    - Clarifying questions (questions about missing info)

    Args:
        text: Text to analyze

    Returns:
        Dict with boolean flags and matched phrases
    """
    text_lower = text.lower()

    # Assumption markers
    assumption_patterns = [
        r"\bassumption",
        r"\bi assume",
        r"\bmight\b",
        r"\bmay\b",
        r"\bdepends?\b",
        r"\bif you\b",
        r"\bunknown\b",
        r"\bnot sure\b",
        r"\bprobably\b",
        r"\blikely\b",
        r"\bunsure\b",
    ]

    # Verification markers
    verification_patterns = [
        r"\bverif",
        r"\bcheck.*(?:official|website|embassy|airline|government)",
        r"\bconfirm with\b",
        r"\blatest rules?\b",
        r"\bup.?to.?date\b",
        r"\bconsult\b",
        r"\bas of\b",
        r"\bcurrent\b.*\bpolicy\b",
    ]

    # Clarifying questions patterns
    clarifying_patterns = [
        r"\bwhich\s+\w+\s+(?:are you|do you|will you)",
        r"\bwhat\s+(?:is your|are your)",
        r"\bwhen\s+(?:are you|will you)",
        r"\bwhere\s+(?:are you|will you)",
        r"\bwhat\s+country\b",
        r"\bhow\s+(?:long|many)",
    ]

    # Detect and collect matches (max 3 per category)
    assumption_matches = []
    for pattern in assumption_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            if len(assumption_matches) < 3:
                # Get context around match (10 chars before and after)
                start = max(0, match.start() - 10)
                end = min(len(text), match.end() + 10)
                context = text[start:end].strip()
                assumption_matches.append(context)

    verification_matches = []
    for pattern in verification_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            if len(verification_matches) < 3:
                start = max(0, match.start() - 10)
                end = min(len(text), match.end() + 10)
                context = text[start:end].strip()
                verification_matches.append(context)

    clarifying_matches = []
    for pattern in clarifying_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            if len(clarifying_matches) < 3:
                start = max(0, match.start() - 10)
                end = min(len(text), match.end() + 10)
                context = text[start:end].strip()
                clarifying_matches.append(context)

    return {
        "has_assumptions_marker": len(assumption_matches) > 0,
        "has_verification_marker": len(verification_matches) > 0,
        "has_clarifying_questions": len(clarifying_matches) > 0,
        "matched_phrases": {
            "assumptions": assumption_matches,
            "verification": verification_matches,
            "clarifying": clarifying_matches,
        },
    }


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile from sorted values.

    Args:
        values: Sorted list of values
        p: Percentile (0-100)

    Returns:
        Percentile value
    """
    if not values:
        return 0.0

    if len(values) == 1:
        return values[0]

    # Use linear interpolation
    rank = (p / 100) * (len(values) - 1)
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, len(values) - 1)
    fraction = rank - lower_idx

    return values[lower_idx] + fraction * (values[upper_idx] - values[lower_idx])


def aggregate_metrics(records: list[dict]) -> dict:
    """Aggregate metrics across all records.

    Args:
        records: List of output records with checks, timing, etc.

    Returns:
        Dict with overall and per-suite metrics
    """
    if not records:
        return {
            "overall": {"n": 0},
            "by_suite": {},
            "definitions": _get_metric_definitions(),
        }

    # Group by suite
    by_suite: dict[str, list[dict]] = {}
    for record in records:
        suite = record.get("suite", "unknown")
        if suite not in by_suite:
            by_suite[suite] = []
        by_suite[suite].append(record)

    # Compute metrics for each suite
    suite_metrics = {}
    for suite_name, suite_records in by_suite.items():
        suite_metrics[suite_name] = _compute_suite_metrics(suite_records)

    # Compute overall metrics
    overall_metrics = _compute_suite_metrics(records)

    return {
        "overall": overall_metrics,
        "by_suite": suite_metrics,
        "definitions": _get_metric_definitions(),
    }


def _compute_suite_metrics(records: list[dict]) -> dict:
    """Compute metrics for a set of records.

    Args:
        records: List of records

    Returns:
        Dict with computed metrics
    """
    n = len(records)
    if n == 0:
        return {"n": 0}

    # Parse rates (legacy checks)
    strict_json_ok_count = sum(
        1 for r in records if r.get("checks", {}).get("strict_json_ok", False)
    )
    lenient_json_ok_count = sum(
        1 for r in records if r.get("checks", {}).get("lenient_json_ok", False)
    )

    # Contract validation rates (schema-based)
    envelope_ok_count = sum(1 for r in records if r.get("contract", {}).get("envelope_ok", False))
    payload_ok_count = sum(1 for r in records if r.get("contract", {}).get("payload_ok", False))
    overall_contract_ok_count = sum(
        1 for r in records if r.get("contract", {}).get("overall_ok", False)
    )

    # Required fields (only count records that have required_fields defined)
    records_with_required = [
        r for r in records if r.get("checks", {}).get("required_fields_ok") is not None
    ]
    required_fields_ok_count = sum(
        1 for r in records_with_required if r.get("checks", {}).get("required_fields_ok", False)
    )

    # Uncertainty markers
    assumptions_count = sum(
        1
        for r in records
        if r.get("checks", {}).get("uncertainty", {}).get("has_assumptions_marker", False)
    )
    verification_count = sum(
        1
        for r in records
        if r.get("checks", {}).get("uncertainty", {}).get("has_verification_marker", False)
    )
    clarifying_count = sum(
        1
        for r in records
        if r.get("checks", {}).get("uncertainty", {}).get("has_clarifying_questions", False)
    )

    # Latency and throughput
    latencies = []
    tokens_per_s = []

    for record in records:
        timing = record.get("timing", {})
        if timing and isinstance(timing, dict):
            lat = timing.get("latency_s")
            tps = timing.get("tokens_per_s")

            if lat is not None and isinstance(lat, int | float) and lat > 0:
                latencies.append(float(lat))
            if tps is not None and isinstance(tps, int | float) and tps > 0:
                tokens_per_s.append(float(tps))

    # Sort for percentiles
    latencies.sort()
    tokens_per_s.sort()

    metrics = {
        "n": n,
        "strict_json_parse_rate": strict_json_ok_count / n if n > 0 else 0.0,
        "lenient_json_parse_rate": lenient_json_ok_count / n if n > 0 else 0.0,
        "assumptions_marker_rate": assumptions_count / n if n > 0 else 0.0,
        "verification_marker_rate": verification_count / n if n > 0 else 0.0,
        "clarifying_questions_rate": clarifying_count / n if n > 0 else 0.0,
    }

    # Contract validation rates
    metrics["envelope_pass_rate"] = envelope_ok_count / n if n > 0 else 0.0
    metrics["payload_pass_rate"] = payload_ok_count / n if n > 0 else 0.0
    metrics["overall_contract_pass_rate"] = overall_contract_ok_count / n if n > 0 else 0.0

    # Required fields rate (only if we have records with required fields)
    if records_with_required:
        metrics["required_fields_rate"] = required_fields_ok_count / len(records_with_required)
        metrics["n_with_required_fields"] = len(records_with_required)
    else:
        metrics["required_fields_rate"] = None
        metrics["n_with_required_fields"] = 0

    # Latency metrics
    if latencies:
        metrics["avg_latency_s"] = sum(latencies) / len(latencies)
        metrics["p50_latency_s"] = _percentile(latencies, 50)
        metrics["p90_latency_s"] = _percentile(latencies, 90)
    else:
        metrics["avg_latency_s"] = None
        metrics["p50_latency_s"] = None
        metrics["p90_latency_s"] = None

    # Throughput metrics
    if tokens_per_s:
        metrics["avg_tokens_per_s"] = sum(tokens_per_s) / len(tokens_per_s)
        metrics["p50_tokens_per_s"] = _percentile(tokens_per_s, 50)
    else:
        metrics["avg_tokens_per_s"] = None
        metrics["p50_tokens_per_s"] = None

    return metrics


def _get_metric_definitions() -> dict:
    """Get metric definitions for documentation.

    Returns:
        Dict with metric definitions
    """
    return {
        "strict_json_parse_rate": "Fraction of outputs that parse as valid JSON when entire stripped text is parsed",
        "lenient_json_parse_rate": "Fraction of outputs with extractable valid JSON (handles code fences, prose wrappers)",
        "required_fields_rate": "Fraction of outputs with all required fields present (where applicable)",
        "assumptions_marker_rate": "Fraction of outputs containing assumption/uncertainty language",
        "verification_marker_rate": "Fraction of outputs mentioning verification/checking with authoritative sources",
        "clarifying_questions_rate": "Fraction of outputs asking clarifying questions about missing context",
        "envelope_pass_rate": "Fraction of outputs passing envelope schema validation (v0/response_envelope.schema.json)",
        "payload_pass_rate": "Fraction of outputs passing payload schema validation (v1/*.payload.schema.json)",
        "overall_contract_pass_rate": "Fraction of outputs passing full contract: (strict OR lenient) JSON parse AND envelope AND payload schemas",
        "latency_metrics": "Response generation latency in seconds (avg, p50, p90)",
        "throughput_metrics": "Token generation rate in tokens/second (avg, p50)",
    }
