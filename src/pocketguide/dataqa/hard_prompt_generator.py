"""
Hard prompt generator for targeted failure modes (Lesson 7.2).

Generates prompts designed to trigger target_failure_modes (e.g. missing_verification_steps,
invalid_json_truncation) for dataset v2 augmentation. Output is a list of prompt-plan-like
objects with id, prompt, payload_type, category, difficulty, region_tags, target_failure_mode.
"""

import random
from typing import Any


# Template patterns keyed by target_failure_mode; each yields a prompt that stresses the requirement.
HARD_PROMPT_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "missing_verification_steps": [
        {
            "payload_type": "checklist",
            "category": "entry_requirements",
            "difficulty": "medium",
            "region_tags": ["EU"],
            "template": (
                "Return JSON only. Include all envelope fields: summary, assumptions, uncertainty_notes, "
                "next_steps, verification_steps, payload_type, payload. "
                "For visa and entry requirements you MUST include verification_steps (array of strings) "
                "pointing to official sources. "
                "Create a short checklist for Schengen visa preparation for a solo traveler to France. "
                "Output valid JSON only. No prose. No markdown."
            ),
        },
        {
            "payload_type": "procedure",
            "category": "entry_requirements",
            "difficulty": "hard",
            "region_tags": ["APAC"],
            "template": (
                "Output must be valid JSON with all required envelope fields including verification_steps. "
                "Describe steps to verify Japan visa requirements. Include verification_steps that recommend "
                "checking embassy and official sites. JSON only. No markdown."
            ),
        },
        {
            "payload_type": "decision_tree",
            "category": "safety",
            "difficulty": "medium",
            "region_tags": ["LATAM"],
            "template": (
                "Return JSON only. Include summary, assumptions, uncertainty_notes, next_steps, "
                "verification_steps, payload_type, payload. "
                "Safety and travel advisories require verification_steps. "
                "Create a decision tree for travel safety in Brazil. Include verification_steps. "
                "JSON only. No prose."
            ),
        },
    ],
    "invalid_json_truncation": [
        {
            "payload_type": "itinerary",
            "category": "itinerary_planning",
            "difficulty": "easy",
            "region_tags": ["NA"],
            "template": (
                "Return JSON only. No prose. No markdown. Ensure the response is complete valid JSON "
                "with all envelope fields and a closed payload. "
                "Create a 1-day itinerary for Montreal. Keep the JSON concise but complete. "
                "Output must end with valid closing braces."
            ),
        },
        {
            "payload_type": "checklist",
            "category": "budgeting",
            "difficulty": "easy",
            "region_tags": ["EU"],
            "template": (
                "Output valid JSON only. No preamble. No trailing text. Include summary, assumptions, "
                "uncertainty_notes, next_steps, verification_steps, payload_type, payload. "
                "Create a short budgeting checklist for a 3-day trip to Berlin. "
                "Ensure the JSON is complete and well-formed."
            ),
        },
    ],
}

# Fallback when target_failure_mode has no template
DEFAULT_TEMPLATE = (
    "Return JSON only with envelope fields: summary, assumptions, uncertainty_notes, "
    "next_steps, verification_steps, payload_type, payload. "
    "Create a short travel checklist. Output valid JSON only. No markdown."
)


def generate_hard_prompts(
    target_failure_modes: list[str],
    count: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate hard prompts for the given target failure modes.

    Args:
        target_failure_modes: List of failure mode strings (e.g. missing_verification_steps).
        count: Total number of prompt objects to generate.
        seed: Random seed for deterministic sampling.

    Returns:
        List of dicts with keys: id, prompt, payload_type, category, difficulty, region_tags, target_failure_mode.
    """
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # Build pool of (target_failure_mode, template_config) for sampling
    pool: list[tuple[str, dict[str, Any]]] = []
    for mode in target_failure_modes:
        templates = HARD_PROMPT_TEMPLATES.get(mode)
        if templates:
            for t in templates:
                pool.append((mode, t))
        else:
            pool.append((mode, {"payload_type": "checklist", "category": "general", "difficulty": "medium", "region_tags": ["NA"], "template": DEFAULT_TEMPLATE}))

    if not pool:
        return out

    for i in range(count):
        mode, cfg = rng.choice(pool)
        prompt = cfg["template"]
        payload_type = cfg.get("payload_type", "checklist")
        category = cfg.get("category", "general")
        difficulty = cfg.get("difficulty", "medium")
        region_tags = cfg.get("region_tags", ["NA"])

        rec_id = f"v2_hard_{mode}_{i:04d}"
        if rec_id in seen_ids:
            rec_id = f"v2_hard_{mode}_{i:04d}_{rng.randint(0, 9999)}"
        seen_ids.add(rec_id)

        out.append({
            "id": rec_id,
            "prompt": prompt,
            "payload_type": payload_type,
            "category": category,
            "difficulty": difficulty,
            "region_tags": region_tags,
            "target_failure_mode": mode,
        })

    return out
