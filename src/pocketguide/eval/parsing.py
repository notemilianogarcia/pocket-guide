"""Parser and validator engine for response envelopes and payloads."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema


@dataclass
class ValidationError:
    """Structured validation error with guidance."""

    error_type: str  # "json_parse", "envelope_schema", "payload_schema", "unknown"
    message: str
    failed_at: str | None = None  # e.g., "summary", "payload.items[0].time_block"
    guidance: list[str] = field(default_factory=list)
    raw_exception: Exception | None = None

    def __str__(self) -> str:
        """Format error as human-readable string."""
        lines = [f"[{self.error_type}] {self.message}"]
        if self.failed_at:
            lines.append(f"  Location: {self.failed_at}")
        if self.guidance:
            lines.append("  Guidance:")
            for tip in self.guidance:
                lines.append(f"    - {tip}")
        return "\n".join(lines)


@dataclass
class ParseResult:
    """Result of parsing and validating a response."""

    success: bool
    data: dict[str, Any] | None = None
    error: ValidationError | None = None

    def __str__(self) -> str:
        """Format result as string."""
        if self.success:
            return "✓ Parse and validation successful"
        return f"✗ Parse failed:\n{self.error}"


def _get_schema_dir() -> Path:
    """Get path to schemas directory."""
    return Path(__file__).parent.parent / "data" / "schemas"


def _load_schema(schema_path: str) -> dict[str, Any]:
    """Load a JSON schema from path."""
    full_path = _get_schema_dir() / schema_path
    if not full_path.exists():
        raise FileNotFoundError(f"Schema not found: {full_path}")
    return json.loads(full_path.read_text())


def _parse_json_strict(text: str) -> tuple[bool, dict[str, Any] | None, str | None]:
    """
    Strictly parse JSON.

    Returns: (success, data, error_message)
    """
    try:
        data = json.loads(text)
        return True, data, None
    except json.JSONDecodeError as e:
        return False, None, str(e)


def _parse_json_lenient(text: str) -> tuple[bool, dict[str, Any] | None, str | None]:
    """
    Leniently extract JSON from text (e.g., from code fences, with preamble).

    Handles:
    - JSON in markdown code fences (```json ... ```)
    - JSON in plain code fences (``` ... ```)
    - JSON wrapped in prose

    Returns: (success, data, error_message)
    """
    # Try to find JSON in code fences first
    json_fence_patterns = [
        r"```(?:json)?\s*\n(.*?)\n```",  # markdown code fence
        r"```(.*?)```",  # plain code fence
    ]

    for pattern in json_fence_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                data = json.loads(json_str)
                return True, data, None
            except json.JSONDecodeError:
                continue

    # Try to find JSON object/array in text (first { or [)
    for start_char in ["{", "["]:
        idx = text.find(start_char)
        if idx >= 0:
            # Find matching closing bracket
            depth = 0
            end_idx = None
            closing_char = "}" if start_char == "{" else "]"

            for i in range(idx, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == closing_char:
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break

            if end_idx:
                json_str = text[idx:end_idx]
                try:
                    data = json.loads(json_str)
                    return True, data, None
                except json.JSONDecodeError:
                    continue

    return False, None, "No valid JSON found in text"


def _get_payload_schema_path(payload_type: str) -> str:
    """Get path to payload schema for a given type."""
    return f"v1/{payload_type}.payload.schema.json"


def _validate_envelope(
    data: dict[str, Any],
) -> tuple[bool, ValidationError | None]:
    """
    Validate data against envelope schema.

    Returns: (success, error)
    """
    try:
        envelope_schema = _load_schema("v0/response_envelope.schema.json")
        jsonschema.validate(data, envelope_schema)
        return True, None
    except jsonschema.ValidationError as e:
        guidance = []
        failed_at = e.json_path if hasattr(e, "json_path") else None

        # Provide helpful guidance based on error
        if "required" in str(e).lower():
            guidance.append(
                "Ensure all required envelope fields are present: "
                "summary, assumptions, uncertainty_notes, next_steps, "
                "verification_steps, payload_type, payload"
            )

        if "payload_type" in str(e).lower():
            guidance.append(
                "payload_type must be one of: " "itinerary, checklist, decision_tree, procedure"
            )

        error = ValidationError(
            error_type="envelope_schema",
            message=e.message,
            failed_at=failed_at,
            guidance=guidance,
            raw_exception=e,
        )
        return False, error


def _validate_payload(
    payload: Any,
    payload_type: str,
) -> tuple[bool, ValidationError | None]:
    """
    Validate payload against its type-specific schema.

    Returns: (success, error)
    """
    try:
        schema_path = _get_payload_schema_path(payload_type)
        payload_schema = _load_schema(schema_path)
        jsonschema.validate(payload, payload_schema)
        return True, None
    except FileNotFoundError as e:
        guidance = [
            f"Payload type '{payload_type}' schema not found. "
            "Ensure payload_type matches one of the 4 supported types."
        ]
        error = ValidationError(
            error_type="payload_schema",
            message=str(e),
            guidance=guidance,
            raw_exception=e,
        )
        return False, error
    except jsonschema.ValidationError as e:
        guidance = []
        failed_at = e.json_path if hasattr(e, "json_path") else None

        # Provide specific guidance based on error
        if "required" in str(e).lower():
            guidance.append(
                f"The payload for type '{payload_type}' is missing required fields. "
                "Check the schema for mandatory keys."
            )

        if "enum" in str(e).lower():
            guidance.append(
                "A field has an invalid value. Check enum constraints "
                "for fields like priority, transport.mode, or node.type."
            )

        error = ValidationError(
            error_type="payload_schema",
            message=e.message,
            failed_at=failed_at,
            guidance=guidance,
            raw_exception=e,
        )
        return False, error


def parse_and_validate(
    text: str,
    strict_json: bool = True,
) -> ParseResult:
    """
    Parse response text and validate against envelope + payload schemas.

    Args:
        text: Response text to parse
        strict_json: If True, use strict parsing only.
                     If False, fall back to lenient parsing on failure.

    Returns:
        ParseResult with success status, parsed data, or error details
    """
    # Attempt JSON parsing
    success, data, parse_error = _parse_json_strict(text)

    if not success:
        if strict_json:
            # Strict mode: fail immediately
            error = ValidationError(
                error_type="json_parse",
                message=parse_error or "Failed to parse JSON",
                guidance=[
                    "Ensure the response is valid JSON.",
                    "Check for unclosed braces, quotes, or missing commas.",
                    "If wrapped in markdown, ensure code fence is properly closed.",
                ],
            )
            return ParseResult(success=False, error=error)

        # Try lenient parsing
        success, data, lenient_error = _parse_json_lenient(text)
        if not success:
            error = ValidationError(
                error_type="json_parse",
                message=lenient_error or "Failed to parse JSON (strict and lenient)",
                guidance=[
                    "Ensure valid JSON appears in the response.",
                    "Try wrapping JSON in markdown code fences: ```json ... ```",
                    "Or provide pure JSON without preamble.",
                ],
            )
            return ParseResult(success=False, error=error)

    # Validate envelope
    success, error = _validate_envelope(data)
    if not success:
        return ParseResult(success=False, error=error)

    # Extract payload_type and payload
    payload_type = data.get("payload_type")
    payload = data.get("payload", {})

    # Validate payload
    success, error = _validate_payload(payload, payload_type)
    if not success:
        return ParseResult(success=False, error=error)

    return ParseResult(success=True, data=data)
