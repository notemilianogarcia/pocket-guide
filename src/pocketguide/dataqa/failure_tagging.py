"""
Failure-mode tagging for v1 records (risk tags for local-runtime failure modes).

Used to identify v1 records that match target failure modes for oversampling
or stricter rejection. v1_clean records are schema-valid; tags indicate
risk (e.g. weak verification) rather than actual parse failure.
"""

import re
from typing import Any


# Overconfidence markers (aligned with quality_gates) — strong certainty without verification
OVERCONFIDENCE_PHRASES = {
    "guaranteed", "definitely", "always", "never",
    "no need to verify", "no need to check", "don't need to verify",
    "100%", "for sure", "without exception",
}

# Generic verification only — single step like "check online" with no specifics
GENERIC_VERIFICATION_PATTERNS = [
    r"check\s+online",
    r"verify\s+online",
    r"search\s+the\s+web",
    r"look\s+it\s+up",
]


def tag_failure_modes(record: dict[str, Any]) -> set[str]:
    """
    Tag a v1 dataset record with risk tags for target failure modes.

    Uses simple heuristics on the response envelope (if valid JSON):
    - missing_verification_steps: verification_steps missing or empty
    - missing_next_steps: next_steps missing or empty
    - overconfident_no_verification: strong certainty markers and no/weak verification_steps
    - generic_verification_only: verification_steps present but only generic phrases

    Args:
        record: v1 record with "response" (envelope) and optionally "prompt"

    Returns:
        Set of tag strings (subset of the above; only tags that apply)
    """
    tags: set[str] = set()
    response = record.get("response") or {}
    if not isinstance(response, dict):
        return tags

    verification_steps = response.get("verification_steps")
    next_steps = response.get("next_steps")
    uncertainty_notes = (response.get("uncertainty_notes") or "").strip()
    # Use summary + uncertainty_notes for overconfidence check (no raw text in v1 record)
    summary = (response.get("summary") or "").strip()
    combined_text = f"{summary} {uncertainty_notes}".lower()

    # missing_verification_steps: absent or empty list
    if not verification_steps or (isinstance(verification_steps, list) and len(verification_steps) == 0):
        tags.add("missing_verification_steps")

    # missing_next_steps: absent or empty list
    if not next_steps or (isinstance(next_steps, list) and len(next_steps) == 0):
        tags.add("missing_next_steps")

    # overconfident_no_verification: certainty markers and no substantive verification
    has_certainty = any(p in combined_text for p in OVERCONFIDENCE_PHRASES)
    has_verification = verification_steps and isinstance(verification_steps, list) and len(verification_steps) > 0
    if has_certainty and not has_verification:
        tags.add("overconfident_no_verification")

    # generic_verification_only: verification_steps exist but every step matches a generic pattern
    if verification_steps and isinstance(verification_steps, list) and len(verification_steps) > 0:
        steps_list = [s for s in verification_steps if isinstance(s, str) and s.strip()]
        if steps_list:
            all_generic = all(
                any(re.search(pat, s.lower()) for pat in GENERIC_VERIFICATION_PATTERNS)
                for s in steps_list
            )
            if all_generic:
                tags.add("generic_verification_only")

    return tags
