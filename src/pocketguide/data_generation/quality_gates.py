"""
Quality gates for final dataset acceptance.

Gates are deterministic checks that ensure each refined response meets
minimum quality standards before inclusion in the training dataset.

Design principles:
- Each gate returns a clear pass/fail result with reason codes
- Gates are composable and independently testable
- Failed samples are recorded with diagnostic details
- Gates focus on objective, verifiable criteria
"""

import re
from typing import Any, Dict, List, NamedTuple


class GateResult(NamedTuple):
    """Result of applying a quality gate."""
    
    passed: bool
    reason_code: str  # e.g., "ok", "missing_verification", "overconfident"
    details: Dict[str, Any]  # diagnostic info for debugging


# Time-sensitive keywords that require verification
TIME_SENSITIVE_KEYWORDS = {
    "visa", "entry requirement", "entry requirements", "border",
    "fees", "fee", "price", "cost", "opening hours", "opening hour",
    "regulations", "regulation", "embassy", "consulate", "airport",
    "customs", "safety advisory", "safety advisories", "travel advisory",
    "travel advisories", "as of", "current", "currently", "now",
    "requirements", "requirement", "permit", "permits", "vaccination",
    "vaccinations", "quarantine", "covid", "insurance requirement"
}

# Overconfidence markers that are inappropriate for time-sensitive topics
OVERCONFIDENCE_PHRASES = {
    "guaranteed", "definitely", "always", "never",
    "no need to verify", "no need to check", "don't need to verify",
    "don't need to check", "absolutely", "certainly will",
    "100%", "for sure", "without exception"
}


def check_contract_ok(
    envelope_ok: bool,
    payload_ok: bool,
    parse_mode: str,  # "strict" or "lenient"
    parsed_envelope: Dict[str, Any] | None
) -> GateResult:
    """
    Gate 1: Contract validation
    
    Ensures the refined response was successfully parsed and validated
    against both envelope and payload schemas.
    
    Args:
        envelope_ok: Whether envelope schema validation passed
        payload_ok: Whether payload schema validation passed
        parse_mode: Which parser succeeded ("strict" or "lenient")
        parsed_envelope: The parsed envelope (for diagnostics)
    
    Returns:
        GateResult indicating pass/fail and details
    """
    if not envelope_ok:
        return GateResult(
            passed=False,
            reason_code="envelope_invalid",
            details={"parse_mode": parse_mode, "envelope_ok": False}
        )
    
    if not payload_ok:
        return GateResult(
            passed=False,
            reason_code="payload_invalid",
            details={"parse_mode": parse_mode, "payload_ok": False}
        )
    
    return GateResult(
        passed=True,
        reason_code="ok",
        details={"parse_mode": parse_mode}
    )


def check_verification_when_needed(
    full_response_text: str,
    verification_steps: List[str],
    uncertainty_notes: str
) -> GateResult:
    """
    Gate 2: Verification requirements for time-sensitive topics
    
    When a response contains time-sensitive information (visas, fees,
    regulations, etc.), it MUST include verification steps and uncertainty
    notes to guide users to authoritative sources.
    
    Args:
        full_response_text: Complete response text (for keyword detection)
        verification_steps: List of verification steps from envelope
        uncertainty_notes: Uncertainty notes from envelope
    
    Returns:
        GateResult indicating pass/fail and requirements
    """
    # Normalize text for case-insensitive matching
    text_lower = full_response_text.lower()
    
    # Check if any time-sensitive keywords are present
    found_keywords = []
    for keyword in TIME_SENSITIVE_KEYWORDS:
        if keyword.lower() in text_lower:
            found_keywords.append(keyword)
    
    # If no time-sensitive content, no requirements
    if not found_keywords:
        return GateResult(
            passed=True,
            reason_code="ok",
            details={"time_sensitive": False}
        )
    
    # Time-sensitive content detected - check requirements
    missing_requirements = []
    
    if not verification_steps or len(verification_steps) == 0:
        missing_requirements.append("verification_steps")
    
    if not uncertainty_notes or len(uncertainty_notes.strip()) == 0:
        missing_requirements.append("uncertainty_notes")
    
    if missing_requirements:
        return GateResult(
            passed=False,
            reason_code="missing_verification_requirements",
            details={
                "time_sensitive": True,
                "found_keywords": found_keywords[:3],  # Sample for debugging
                "missing": missing_requirements,
                "verification_steps_count": len(verification_steps) if verification_steps else 0,
                "uncertainty_notes_length": len(uncertainty_notes.strip()) if uncertainty_notes else 0
            }
        )
    
    return GateResult(
        passed=True,
        reason_code="ok",
        details={
            "time_sensitive": True,
            "verification_steps_count": len(verification_steps),
            "uncertainty_notes_length": len(uncertainty_notes.strip())
        }
    )


def check_overconfidence_guard(
    full_response_text: str,
    is_time_sensitive: bool
) -> GateResult:
    """
    Gate 3: Overconfidence detection for time-sensitive topics
    
    Rejects responses that use absolute certainty language when discussing
    time-sensitive topics that may change (visas, regulations, fees, etc.).
    
    Args:
        full_response_text: Complete response text (for phrase detection)
        is_time_sensitive: Whether response contains time-sensitive keywords
    
    Returns:
        GateResult indicating pass/fail and detected phrases
    """
    # Only check if content is time-sensitive
    if not is_time_sensitive:
        return GateResult(
            passed=True,
            reason_code="ok",
            details={"time_sensitive": False}
        )
    
    # Normalize text for case-insensitive matching
    text_lower = full_response_text.lower()
    
    # Check for overconfidence phrases
    found_phrases = []
    for phrase in OVERCONFIDENCE_PHRASES:
        if phrase.lower() in text_lower:
            found_phrases.append(phrase)
    
    if found_phrases:
        return GateResult(
            passed=False,
            reason_code="overconfident_time_sensitive",
            details={
                "found_phrases": found_phrases,
                "message": "Time-sensitive content should not use absolute certainty language"
            }
        )
    
    return GateResult(
        passed=True,
        reason_code="ok",
        details={"time_sensitive": True, "no_overconfidence_detected": True}
    )


def apply_all_gates(
    envelope_ok: bool,
    payload_ok: bool,
    parse_mode: str,
    parsed_envelope: Dict[str, Any] | None,
    full_response_text: str
) -> Dict[str, GateResult]:
    """
    Apply all quality gates to a refined response.
    
    Args:
        envelope_ok: Whether envelope validation passed
        payload_ok: Whether payload validation passed
        parse_mode: Which parser succeeded ("strict" or "lenient")
        parsed_envelope: The parsed envelope object
        full_response_text: Complete response text for keyword detection
    
    Returns:
        Dict mapping gate name to GateResult
    """
    gates = {}
    
    # Gate 1: Contract validation
    gates["contract_ok"] = check_contract_ok(
        envelope_ok=envelope_ok,
        payload_ok=payload_ok,
        parse_mode=parse_mode,
        parsed_envelope=parsed_envelope
    )
    
    # If contract failed, can't run remaining gates
    if not gates["contract_ok"].passed:
        gates["verification_when_needed"] = GateResult(
            passed=False,
            reason_code="skipped_contract_failed",
            details={}
        )
        gates["overconfidence_guard"] = GateResult(
            passed=False,
            reason_code="skipped_contract_failed",
            details={}
        )
        return gates
    
    # Extract envelope fields for subsequent gates
    verification_steps = parsed_envelope.get("verification_steps", [])
    uncertainty_notes = parsed_envelope.get("uncertainty_notes", "")
    
    # Gate 2: Verification requirements
    gates["verification_when_needed"] = check_verification_when_needed(
        full_response_text=full_response_text,
        verification_steps=verification_steps,
        uncertainty_notes=uncertainty_notes
    )
    
    # Determine if content is time-sensitive (for Gate 3)
    is_time_sensitive = gates["verification_when_needed"].details.get("time_sensitive", False)
    
    # Gate 3: Overconfidence guard
    gates["overconfidence_guard"] = check_overconfidence_guard(
        full_response_text=full_response_text,
        is_time_sensitive=is_time_sensitive
    )
    
    return gates


def compute_overall_quality(gates: Dict[str, GateResult]) -> bool:
    """
    Determine if a response passes overall quality standards.
    
    Args:
        gates: Dict of gate results from apply_all_gates
    
    Returns:
        True if all gates passed, False otherwise
    """
    return all(gate.passed for gate in gates.values())


def get_rejection_reasons(gates: Dict[str, GateResult]) -> List[str]:
    """
    Get list of reason codes for failed gates.
    
    Args:
        gates: Dict of gate results from apply_all_gates
    
    Returns:
        List of reason codes for gates that failed
    """
    return [
        gate.reason_code
        for gate in gates.values()
        if not gate.passed
    ]
