"""
Quality filter functions for QA dataset.

Implements deterministic filters to remove low-signal or unsafe samples:
- Length filters (too short / too long)
- Vagueness heuristics (non-specific guidance)
- Overclaim / time-sensitive filters (unverified strong certainty + time sensitivity)

Each filter function returns a FilterResult indicating pass/fail and reason_code.
Filters are independent and composable.
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class FilterResult:
    """Result of applying a quality filter."""
    passed: bool
    reason_code: str  # "ok" or specific failure code
    details: Dict[str, Any]  # Metadata: counts, matched phrases, etc.


# Configuration constants for filters

# Length filter thresholds
MIN_WORDS_DEFAULT = 100
MAX_WORDS_DEFAULT = 800

# Vagueness heuristics
VAGUE_PHRASES = [
    "it depends",
    "generally",
    "usually",
    "may vary",
    "check online",
    "do your own research",
]

# Concrete action verbs (non-exhaustive, conservative list)
ACTION_VERBS = [
    "apply",
    "bring",
    "contact",
    "verify",
    "call",
    "visit",
    "check",
    "document",
    "submit",
    "prepare",
    "obtain",
    "book",
    "schedule",
    "require",
]

# Vagueness filter thresholds
VAGUE_PHRASE_THRESHOLD = 2  # Minimum vague phrases to consider for rejection
CONCRETE_ACTION_THRESHOLD = 3  # Minimum concrete actions to pass

# Overclaim / time-sensitive markers
TIME_SENSITIVE_MARKERS = [
    "visa",
    "entry requirement",
    "fees",
    "price",
    "customs",
    "border",
    "as of",
    "currently",
]

STRONG_CERTAINTY_PHRASES = [
    "definitely",
    "guaranteed",
    "always",
    "never",
    "no need to verify",
]

# Overclaim thresholds
OVERCLAIM_REJECTION_THRESHOLD = 2  # Both time-sensitive AND certainty must appear


def _normalize_text(text: str) -> str:
    """Normalize text for analysis: lowercase, strip."""
    if not text:
        return ""
    return text.lower().strip()


def _extract_flat_text(record: Dict[str, Any]) -> str:
    """Extract and flatten relevant text from record for analysis."""
    parts = []
    
    # Add summary
    if "response" in record and "summary" in record["response"]:
        parts.append(_normalize_text(record["response"]["summary"]))
    
    # Add next_steps
    if "response" in record and "next_steps" in record["response"]:
        next_steps = record["response"]["next_steps"]
        if isinstance(next_steps, list):
            for step in next_steps:
                if isinstance(step, dict) and "description" in step:
                    parts.append(_normalize_text(step["description"]))
                elif isinstance(step, str):
                    parts.append(_normalize_text(step))
        elif isinstance(next_steps, str):
            parts.append(_normalize_text(next_steps))
    
    # Add verification_steps
    if "response" in record and "verification_steps" in record["response"]:
        verif_steps = record["response"]["verification_steps"]
        if isinstance(verif_steps, list):
            for step in verif_steps:
                if isinstance(step, dict) and "description" in step:
                    parts.append(_normalize_text(step["description"]))
                elif isinstance(step, str):
                    parts.append(_normalize_text(step))
        elif isinstance(verif_steps, str):
            parts.append(_normalize_text(verif_steps))
    
    return " ".join(parts)


def _get_full_response_text(record: Dict[str, Any]) -> str:
    """Extract full response text including uncertainty notes for analysis."""
    parts = []
    
    if "response" not in record:
        return ""
    
    response = record["response"]
    
    # Add summary
    if "summary" in response:
        parts.append(_normalize_text(response["summary"]))
    
    # Add next_steps
    if "next_steps" in response:
        next_steps = response["next_steps"]
        if isinstance(next_steps, list):
            for step in next_steps:
                if isinstance(step, dict) and "description" in step:
                    parts.append(_normalize_text(step["description"]))
                elif isinstance(step, str):
                    parts.append(_normalize_text(step))
        elif isinstance(next_steps, str):
            parts.append(_normalize_text(next_steps))
    
    # Add verification_steps
    if "verification_steps" in response:
        verif_steps = response["verification_steps"]
        if isinstance(verif_steps, list):
            for step in verif_steps:
                if isinstance(step, dict) and "description" in step:
                    parts.append(_normalize_text(step["description"]))
                elif isinstance(step, str):
                    parts.append(_normalize_text(step))
        elif isinstance(verif_steps, str):
            parts.append(_normalize_text(verif_steps))
    
    # Add uncertainty_notes
    if "uncertainty_notes" in response:
        uncertainty = response["uncertainty_notes"]
        if isinstance(uncertainty, str):
            parts.append(_normalize_text(uncertainty))
    
    return " ".join(parts)


def check_length(
    record: Dict[str, Any],
    min_words: int = MIN_WORDS_DEFAULT,
    max_words: int = MAX_WORDS_DEFAULT,
) -> FilterResult:
    """
    Check if record text length is within acceptable range.
    
    Counts words in summary + next_steps + verification_steps.
    
    Args:
        record: QA record to check
        min_words: Minimum word count threshold
        max_words: Maximum word count threshold
    
    Returns:
        FilterResult with passed, reason_code, and word count details
    """
    flat_text = _extract_flat_text(record)
    word_count = len(flat_text.split()) if flat_text else 0
    
    details = {"word_count": word_count, "min": min_words, "max": max_words}
    
    if word_count < min_words:
        return FilterResult(
            passed=False,
            reason_code="too_short",
            details=details,
        )
    
    if word_count > max_words:
        return FilterResult(
            passed=False,
            reason_code="too_long",
            details=details,
        )
    
    return FilterResult(
        passed=True,
        reason_code="ok",
        details=details,
    )


def check_vagueness(
    record: Dict[str, Any],
    vague_phrase_threshold: int = VAGUE_PHRASE_THRESHOLD,
    concrete_action_threshold: int = CONCRETE_ACTION_THRESHOLD,
) -> FilterResult:
    """
    Check for vagueness: too many vague phrases + too few concrete actions.
    
    Args:
        record: QA record to check
        vague_phrase_threshold: Minimum vague phrases to consider rejection
        concrete_action_threshold: Minimum concrete actions for passing
    
    Returns:
        FilterResult with passed, reason_code, and phrase count details
    """
    flat_text = _extract_flat_text(record)
    
    # Count vague phrase occurrences
    vague_count = 0
    matched_vague_phrases = []
    for phrase in VAGUE_PHRASES:
        # Case-insensitive word boundary matching
        pattern = r"\b" + re.escape(phrase) + r"\b"
        matches = len(re.findall(pattern, flat_text, re.IGNORECASE))
        if matches > 0:
            vague_count += matches
            matched_vague_phrases.append((phrase, matches))
    
    # Count concrete action verbs
    concrete_count = 0
    matched_actions = []
    for verb in ACTION_VERBS:
        pattern = r"\b" + re.escape(verb) + r"\b"
        matches = len(re.findall(pattern, flat_text, re.IGNORECASE))
        if matches > 0:
            concrete_count += matches
            matched_actions.append((verb, matches))
    
    # Check for numbered/bulleted steps
    has_numbered = bool(re.search(r"^\s*\d+[\.\)]\s", flat_text, re.MULTILINE))
    has_bullets = bool(re.search(r"^\s*[-*•]\s", flat_text, re.MULTILINE))
    structural_count = (1 if has_numbered else 0) + (1 if has_bullets else 0)
    
    details = {
        "vague_phrase_count": vague_count,
        "vague_phrases": matched_vague_phrases,
        "concrete_action_count": concrete_count,
        "concrete_actions": matched_actions,
        "has_numbered_steps": has_numbered,
        "has_bulleted_steps": has_bullets,
        "structural_count": structural_count,
    }
    
    # Reject if too many vague phrases AND too few concrete indicators
    if (
        vague_count >= vague_phrase_threshold
        and concrete_count < concrete_action_threshold
        and structural_count == 0
    ):
        return FilterResult(
            passed=False,
            reason_code="vague_low_specificity",
            details=details,
        )
    
    return FilterResult(
        passed=True,
        reason_code="ok",
        details=details,
    )


def check_overclaim(
    record: Dict[str, Any],
    rejection_threshold: int = OVERCLAIM_REJECTION_THRESHOLD,
) -> FilterResult:
    """
    Check for unverified strong certainty about time-sensitive information.
    
    Rejects if:
    - Time-sensitive markers present AND
    - Strong certainty phrases present AND
    - verification_steps is empty OR uncertainty_notes is empty
    
    Args:
        record: QA record to check
        rejection_threshold: Must detect both time-sensitive and certainty markers
    
    Returns:
        FilterResult with passed, reason_code, and marker details
    """
    full_text = _get_full_response_text(record)
    
    # Count time-sensitive markers
    time_sensitive_count = 0
    matched_time_sensitive = []
    for marker in TIME_SENSITIVE_MARKERS:
        pattern = r"\b" + re.escape(marker) + r"\b"
        matches = len(re.findall(pattern, full_text, re.IGNORECASE))
        if matches > 0:
            time_sensitive_count += matches
            matched_time_sensitive.append((marker, matches))
    
    # Count strong certainty phrases
    certainty_count = 0
    matched_certainty = []
    for phrase in STRONG_CERTAINTY_PHRASES:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        matches = len(re.findall(pattern, full_text, re.IGNORECASE))
        if matches > 0:
            certainty_count += matches
            matched_certainty.append((phrase, matches))
    
    # Check verification and uncertainty fields
    has_verification = False
    has_uncertainty_notes = False
    
    if "response" in record:
        response = record["response"]
        
        verification_steps = response.get("verification_steps", [])
        if isinstance(verification_steps, list) and len(verification_steps) > 0:
            has_verification = True
        elif isinstance(verification_steps, str) and verification_steps.strip():
            has_verification = True
        
        uncertainty_notes = response.get("uncertainty_notes", "")
        if isinstance(uncertainty_notes, str) and uncertainty_notes.strip():
            has_uncertainty_notes = True
    
    details = {
        "time_sensitive_count": time_sensitive_count,
        "time_sensitive_markers": matched_time_sensitive,
        "certainty_count": certainty_count,
        "certainty_phrases": matched_certainty,
        "has_verification_steps": has_verification,
        "has_uncertainty_notes": has_uncertainty_notes,
    }
    
    # Reject if both time-sensitive AND certainty present, but missing BOTH safeguards
    if (
        time_sensitive_count > 0
        and certainty_count > 0
        and not has_verification
        and not has_uncertainty_notes
    ):
        return FilterResult(
            passed=False,
            reason_code="overconfident_time_sensitive",
            details=details,
        )
    
    return FilterResult(
        passed=True,
        reason_code="ok",
        details=details,
    )


def apply_all_filters(
    record: Dict[str, Any],
    min_words: int = MIN_WORDS_DEFAULT,
    max_words: int = MAX_WORDS_DEFAULT,
    vague_phrase_threshold: int = VAGUE_PHRASE_THRESHOLD,
    concrete_action_threshold: int = CONCRETE_ACTION_THRESHOLD,
    overclaim_threshold: int = OVERCLAIM_REJECTION_THRESHOLD,
) -> Optional[FilterResult]:
    """
    Apply all quality filters sequentially.
    
    Returns on first failure (short-circuit).
    
    Args:
        record: QA record to check
        min_words: Minimum words for length check
        max_words: Maximum words for length check
        vague_phrase_threshold: Vague phrase threshold
        concrete_action_threshold: Concrete action threshold
        overclaim_threshold: Overclaim rejection threshold
    
    Returns:
        FilterResult of first failure, or None if all pass
    """
    # Check length
    result = check_length(record, min_words, max_words)
    if not result.passed:
        return result
    
    # Check vagueness
    result = check_vagueness(
        record, vague_phrase_threshold, concrete_action_threshold
    )
    if not result.passed:
        return result
    
    # Check overclaim
    result = check_overclaim(record, overclaim_threshold)
    if not result.passed:
        return result
    
    return None
