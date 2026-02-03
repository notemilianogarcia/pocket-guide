"""Parser and validator engine for response envelopes and payloads."""

import json
import re
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Callable

import jsonschema

# Error codes for stable metrics/reporting
JSON_STRICT_PARSE_FAILED = "JSON_STRICT_PARSE_FAILED"
JSON_LENIENT_PARSE_FAILED = "JSON_LENIENT_PARSE_FAILED"
ENVELOPE_SCHEMA_FAILED = "ENVELOPE_SCHEMA_FAILED"
PAYLOAD_SCHEMA_FAILED = "PAYLOAD_SCHEMA_FAILED"

# Schema cache to avoid repeated file I/O
_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}

# Maximum candidate JSON length for lenient parsing (200KB)
_MAX_CANDIDATE_LENGTH = 200_000


@dataclass
class ParseValidationError:
    """Structured validation error with guidance.

    Note: Renamed from ValidationError to avoid collision with jsonschema.ValidationError.
    """

    code: str  # Stable error code for metrics (e.g., JSON_STRICT_PARSE_FAILED)
    error_type: str  # "json_parse", "envelope_schema", "payload_schema"
    message: str
    failed_at: str | None = None  # e.g., "summary", "payload.items[0].time_block"
    guidance: list[str] = field(default_factory=list)
    raw_exception: Exception | None = None

    def __str__(self) -> str:
        """Format error as human-readable string."""
        lines = [f"[{self.code}] {self.message}"]
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
    error: ParseValidationError | None = None

    def __str__(self) -> str:
        """Format result as string."""
        if self.success:
            return "✓ Parse and validation successful"
        return f"✗ Parse failed:\n{self.error}"


def _load_schema(schema_path: str) -> dict[str, Any]:
    """
    Load a JSON schema using importlib.resources.

    Caches schemas to avoid repeated I/O.

    Args:
        schema_path: Relative path within schemas directory (e.g., "v0/response_envelope.schema.json")

    Returns:
        Loaded schema as dict

    Raises:
        FileNotFoundError: If schema file doesn't exist
    """
    if schema_path in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[schema_path]

    try:
        # Load from package resources
        schema_text = resources.files("pocketguide.data.schemas").joinpath(schema_path).read_text()
        schema = json.loads(schema_text)
        _SCHEMA_CACHE[schema_path] = schema
        return schema
    except (FileNotFoundError, AttributeError) as e:
        raise FileNotFoundError(f"Schema not found: {schema_path}") from e


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

    Prefers a JSON object over an array when both exist (envelope is always an object).
    Models sometimes output a leading array (e.g. summary bullets) then the envelope;
    taking the first object avoids wrongly using the array and failing envelope validation.

    Returns: (success, data, error_message). data is dict when an object was found,
    otherwise may be list when only array(s) exist (envelope validation will then fail).
    """
    candidates: list[Any] = []  # collect parseable values; prefer dict

    # Try to find JSON in code fences first
    json_fence_patterns = [
        r"```(?:json)?\s*\n(.*?)\n```",  # markdown code fence
        r"```(.*?)```",  # plain code fence
    ]

    for pattern in json_fence_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            json_str = match.group(1).strip()

            if len(json_str) > _MAX_CANDIDATE_LENGTH:
                continue

            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    return True, data, None
                candidates.append(data)
            except json.JSONDecodeError:
                continue

    # If we found any parse from fences, return first (object already returned above)
    if candidates:
        return True, candidates[0], None

    # Try to find JSON object first, then array (envelope is object)
    for start_char in ["{", "["]:
        idx = text.find(start_char)
        if idx >= 0:
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

                if len(json_str) > _MAX_CANDIDATE_LENGTH:
                    continue

                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        return True, data, None
                    candidates.append(data)
                except json.JSONDecodeError:
                    continue

    if candidates:
        return True, candidates[0], None

    return False, None, "No valid JSON found in text"


def _get_payload_schema_path(payload_type: str) -> str:
    """
    Get path to payload schema for a given type.

    Makes payload schema selection explicit and deterministic.
    """
    # Explicit mapping for deterministic schema selection
    schema_map = {
        "itinerary": "v1/itinerary.payload.schema.json",
        "checklist": "v1/checklist.payload.schema.json",
        "decision_tree": "v1/decision_tree.payload.schema.json",
        "procedure": "v1/procedure.payload.schema.json",
    }

    if payload_type not in schema_map:
        raise ValueError(f"Unknown payload_type: {payload_type}")

    return schema_map[payload_type]


def _validate_envelope(
    data: dict[str, Any],
) -> tuple[bool, ParseValidationError | None]:
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
        msg = getattr(e, "message", str(e))

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

        error = ParseValidationError(
            code=ENVELOPE_SCHEMA_FAILED,
            error_type="envelope_schema",
            message=msg,
            failed_at=failed_at,
            guidance=guidance,
            raw_exception=e,
        )
        return False, error


def _validate_payload(
    payload: Any,
    payload_type: str,
) -> tuple[bool, ParseValidationError | None]:
    """
    Validate payload against its type-specific schema.

    Returns: (success, error)
    """
    try:
        schema_path = _get_payload_schema_path(payload_type)
        payload_schema = _load_schema(schema_path)
        jsonschema.validate(payload, payload_schema)
        return True, None
    except (FileNotFoundError, ValueError) as e:
        guidance = [
            f"Payload type '{payload_type}' schema not found. "
            "Ensure payload_type matches one of the 4 supported types."
        ]
        error = ParseValidationError(
            code=PAYLOAD_SCHEMA_FAILED,
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

        msg = getattr(e, "message", str(e))
        error = ParseValidationError(
            code=PAYLOAD_SCHEMA_FAILED,
            error_type="payload_schema",
            message=msg,
            failed_at=failed_at,
            guidance=guidance,
            raw_exception=e,
        )
        return False, error


def parse_and_validate(
    text: str,
    strict_json: bool = True,
    validate_payload: bool = True,
    normalizer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> ParseResult:
    """
    Parse response text and validate against envelope + payload schemas.

    Args:
        text: Response text to parse
        strict_json: If True, use strict parsing only.
                     If False, fall back to lenient parsing on failure.
        validate_payload: If False, skip payload schema validation (envelope only).
        normalizer: Optional function to normalize parsed data before validation
                    (e.g. coerce string fields to arrays for teacher output).

    Returns:
        ParseResult with success status, parsed data, or error details
    """
    # Attempt JSON parsing
    success, data, parse_error = _parse_json_strict(text)

    if not success:
        if strict_json:
            # Strict mode: fail immediately
            error = ParseValidationError(
                code=JSON_STRICT_PARSE_FAILED,
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
            error = ParseValidationError(
                code=JSON_LENIENT_PARSE_FAILED,
                error_type="json_parse",
                message=lenient_error or "Failed to parse JSON (strict and lenient)",
                guidance=[
                    "Ensure valid JSON appears in the response.",
                    "Try wrapping JSON in markdown code fences: ```json ... ```",
                    "Or provide pure JSON without preamble.",
                ],
            )
            return ParseResult(success=False, error=error)

    if normalizer is not None:
        data = normalizer(data)

    # Envelope schema requires an object; lenient parse can return an array
    if not isinstance(data, dict):
        error = ParseValidationError(
            code=ENVELOPE_SCHEMA_FAILED,
            error_type="envelope_schema",
            message=f"Top-level JSON must be an object (envelope), got {type(data).__name__}",
            guidance=[
                "Output a single JSON object with keys: summary, assumptions, "
                "uncertainty_notes, next_steps, verification_steps, payload_type, payload.",
                "Do not output a JSON array as the root; the envelope is an object.",
            ],
        )
        return ParseResult(success=False, data=data, error=error)

    # Validate envelope
    success, error = _validate_envelope(data)
    if not success:
        return ParseResult(success=False, data=data, error=error)

    if not validate_payload:
        return ParseResult(success=True, data=data)

    # Extract payload_type and payload
    payload_type = data.get("payload_type")
    payload = data.get("payload", {})

    # Validate payload
    success, error = _validate_payload(payload, payload_type)
    if not success:
        return ParseResult(success=False, data=data, error=error)

    return ParseResult(success=True, data=data)
