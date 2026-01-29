"""
Tests for quality_filters module.

Tests cover:
- Length filter (too_short, too_long, within range)
- Vagueness heuristic (vague phrases vs concrete actions)
- Overclaim / time-sensitive filter (certainty + time-sensitive without verification)
- Integration of all filters
- Deterministic behavior
"""

import pytest

from pocketguide.data_generation.quality_filters import (
    check_length,
    check_vagueness,
    check_overclaim,
    apply_all_filters,
    FilterResult,
    MIN_WORDS_DEFAULT,
    MAX_WORDS_DEFAULT,
    VAGUE_PHRASE_THRESHOLD,
    CONCRETE_ACTION_THRESHOLD,
    OVERCLAIM_REJECTION_THRESHOLD,
)


class TestCheckLength:
    """Test length filter."""

    def test_within_range(self):
        """Record with word count within range passes."""
        record = {
            "id": "test1",
            "prompt": "What is the capital?",
            "response": {
                "summary": "The capital is Paris. " * 10,  # ~50 words
                "next_steps": [],
                "verification_steps": [],
            },
        }
        result = check_length(record, min_words=40, max_words=800)
        assert result.passed is True
        assert result.reason_code == "ok"
        assert result.details["word_count"] >= 40

    def test_too_short(self):
        """Record with too few words is rejected."""
        record = {
            "id": "test_short",
            "prompt": "Hi",
            "response": {
                "summary": "Short answer.",
                "next_steps": [],
                "verification_steps": [],
            },
        }
        result = check_length(record, min_words=100, max_words=800)
        assert result.passed is False
        assert result.reason_code == "too_short"
        assert result.details["word_count"] < 100

    def test_too_long(self):
        """Record with too many words is rejected."""
        record = {
            "id": "test_long",
            "prompt": "Long question?",
            "response": {
                "summary": "Word " * 1000,  # ~1000 words
                "next_steps": [],
                "verification_steps": [],
            },
        }
        result = check_length(record, min_words=100, max_words=800)
        assert result.passed is False
        assert result.reason_code == "too_long"
        assert result.details["word_count"] > 800

    def test_custom_thresholds(self):
        """Length filter respects custom thresholds."""
        record = {
            "id": "test",
            "prompt": "Question",
            "response": {
                "summary": "word " * 25,  # 25 words
                "next_steps": [],
                "verification_steps": [],
            },
        }
        # With min=50, should fail
        result = check_length(record, min_words=50, max_words=200)
        assert result.passed is False
        assert result.reason_code == "too_short"

        # With min=10, should pass
        result = check_length(record, min_words=10, max_words=200)
        assert result.passed is True
        assert result.reason_code == "ok"

    def test_word_count_aggregates_components(self):
        """Word count includes summary, next_steps, and verification_steps."""
        record = {
            "id": "test",
            "prompt": "Question",
            "response": {
                "summary": "word " * 40,  # 40 words
                "next_steps": [{"description": "action " * 30}],  # 30 words
                "verification_steps": [{"description": "verify " * 30}],  # 30 words
            },
        }
        result = check_length(record, min_words=80, max_words=200)
        assert result.passed is True
        assert result.details["word_count"] >= 80

    def test_empty_response(self):
        """Empty response is too short."""
        record = {
            "id": "test",
            "prompt": "Question",
            "response": {
                "summary": "",
                "next_steps": [],
                "verification_steps": [],
            },
        }
        result = check_length(record, min_words=100, max_words=800)
        assert result.passed is False
        assert result.reason_code == "too_short"


class TestCheckVagueness:
    """Test vagueness filter."""

    def test_vague_response_passes(self):
        """Response with concrete actions and no vague phrases passes."""
        record = {
            "id": "test1",
            "prompt": "How to apply?",
            "response": {
                "summary": "Follow these steps: " + "word " * 30,
                "next_steps": [
                    {"description": "1. Call the office"},
                    {"description": "2. Submit your documents"},
                    {"description": "3. Verify your identity"},
                ],
                "verification_steps": [
                    {"description": "Check your email"},
                ],
            },
        }
        result = check_vagueness(
            record,
            vague_phrase_threshold=2,
            concrete_action_threshold=3,
        )
        assert result.passed is True
        assert result.reason_code == "ok"
        assert result.details["concrete_action_count"] >= 3

    def test_vague_without_actions_rejected(self):
        """Response with vague phrases and no concrete actions rejected."""
        record = {
            "id": "test_vague",
            "prompt": "What should I do?",
            "response": {
                "summary": (
                    "It depends on your situation. Generally, you should do your own research. "
                    "Check online for more information. This may vary based on circumstances. "
                    "Usually it's best to get advice. " + "word " * 30
                ),
                "next_steps": [],
                "verification_steps": [],
            },
        }
        result = check_vagueness(
            record,
            vague_phrase_threshold=2,
            concrete_action_threshold=3,
        )
        assert result.passed is False
        assert result.reason_code == "vague_low_specificity"
        assert result.details["vague_phrase_count"] >= 2

    def test_vague_phrase_counting(self):
        """Vagueness filter counts vague phrase occurrences."""
        record = {
            "id": "test",
            "prompt": "Question",
            "response": {
                "summary": (
                    "It depends. Generally, this may vary. "
                    "Do your own research and check online. " + "word " * 30
                ),
                "next_steps": [],
                "verification_steps": [],
            },
        }
        result = check_vagueness(record)
        # Should count: "it depends", "generally", "may vary", "do your own research", "check online"
        assert result.details["vague_phrase_count"] >= 3

    def test_concrete_action_detection(self):
        """Vagueness filter detects concrete action verbs."""
        record = {
            "id": "test",
            "prompt": "How to get visa?",
            "response": {
                "summary": (
                    "You need to apply for a visa, bring your documents, and contact the embassy. "
                    "Verify your application status and submit additional info if needed. " + "word " * 30
                ),
                "next_steps": [],
                "verification_steps": [],
            },
        }
        result = check_vagueness(record)
        assert result.details["concrete_action_count"] >= 3
        assert result.passed is True

    def test_numbered_steps_detection(self):
        """Vagueness filter detects numbered steps."""
        record = {
            "id": "test",
            "prompt": "Steps?",
            "response": {
                "summary": (
                    "1. First step\n"
                    "2. Second step\n"
                    "3. Third step\n" + "word " * 30
                ),
                "next_steps": [],
                "verification_steps": [],
            },
        }
        result = check_vagueness(record)
        assert result.details["has_numbered_steps"] is True
        assert result.passed is True

    def test_bulleted_steps_detection(self):
        """Vagueness filter detects bulleted steps."""
        record = {
            "id": "test",
            "prompt": "Steps?",
            "response": {
                "summary": (
                    "- Action one\n"
                    "- Action two\n"
                    "- Action three\n" + "word " * 30
                ),
                "next_steps": [],
                "verification_steps": [],
            },
        }
        result = check_vagueness(record)
        assert result.details["has_bulleted_steps"] is True
        assert result.passed is True

    def test_vague_with_structure_passes(self):
        """Vague phrases OK if there's clear structure."""
        record = {
            "id": "test",
            "prompt": "Generally speaking...?",
            "response": {
                "summary": (
                    "Generally, you should follow these steps:\n"
                    "1. Do one thing\n"
                    "2. Do another thing\n" + "word " * 30
                ),
                "next_steps": [],
                "verification_steps": [],
            },
        }
        result = check_vagueness(record)
        # Has structure, should pass despite vague phrase
        assert result.passed is True


class TestCheckOverclaim:
    """Test overclaim / time-sensitive filter."""

    def test_overclaim_detected(self):
        """Strong certainty + time-sensitive without verification rejected."""
        record = {
            "id": "test_overclaim",
            "prompt": "Visa requirements?",
            "response": {
                "summary": (
                    "You will definitely need a visa. Entry requirements are guaranteed to include "
                    "a passport. You never need to verify this. " + "word " * 30
                ),
                "next_steps": [],
                "verification_steps": [],
                "uncertainty_notes": "",
            },
        }
        result = check_overclaim(record)
        assert result.passed is False
        assert result.reason_code == "overconfident_time_sensitive"
        assert result.details["time_sensitive_count"] > 0
        assert result.details["certainty_count"] > 0

    def test_time_sensitive_with_verification_passes(self):
        """Time-sensitive with certainty OK if verification present."""
        record = {
            "id": "test",
            "prompt": "Visa?",
            "response": {
                "summary": (
                    "You will definitely need a visa. Entry requirements include a passport. "
                    "Always bring your documents. " + "word " * 30
                ),
                "next_steps": [],
                "verification_steps": [
                    {"description": "Check the official website for current requirements"}
                ],
                "uncertainty_notes": "Requirements may change without notice.",
            },
        }
        result = check_overclaim(record)
        assert result.passed is True
        assert result.reason_code == "ok"

    def test_time_sensitive_with_uncertainty_notes_passes(self):
        """Time-sensitive with certainty OK if uncertainty_notes present."""
        record = {
            "id": "test",
            "prompt": "Fees?",
            "response": {
                "summary": (
                    "The visa fees are definitely $100. This is never going to change. "
                    "You always need to pay exactly this amount. " + "word " * 30
                ),
                "next_steps": [],
                "verification_steps": [],
                "uncertainty_notes": "Fees can change seasonally and may vary by country.",
            },
        }
        result = check_overclaim(record)
        assert result.passed is True
        assert result.reason_code == "ok"

    def test_non_time_sensitive_certainty_passes(self):
        """Strong certainty without time-sensitive markers OK."""
        record = {
            "id": "test",
            "prompt": "Capital?",
            "response": {
                "summary": (
                    "The capital is definitely Paris. You will always find the Eiffel Tower there. "
                    "This is guaranteed. " + "word " * 30
                ),
                "next_steps": [],
                "verification_steps": [],
                "uncertainty_notes": "",
            },
        }
        result = check_overclaim(record)
        assert result.passed is True
        # No time-sensitive markers
        assert result.details["time_sensitive_count"] == 0

    def test_time_sensitive_without_certainty_passes(self):
        """Time-sensitive without strong certainty OK."""
        record = {
            "id": "test",
            "prompt": "Border crossing?",
            "response": {
                "summary": (
                    "Border customs procedures are complex. The fees may vary. "
                    "Current requirements are listed online. " + "word " * 30
                ),
                "next_steps": [],
                "verification_steps": [],
                "uncertainty_notes": "",
            },
        }
        result = check_overclaim(record)
        assert result.passed is True
        # Has time-sensitive markers but no strong certainty
        assert result.details["certainty_count"] == 0

    def test_marker_detection(self):
        """Overclaim filter detects specific markers."""
        record = {
            "id": "test",
            "prompt": "Question?",
            "response": {
                "summary": (
                    "As of now, visa fees are $100. Border customs always check everything. "
                    "Entry requirements are definitely fixed. " + "word " * 30
                ),
                "next_steps": [],
                "verification_steps": [],
                "uncertainty_notes": "",
            },
        }
        result = check_overclaim(record)
        assert len(result.details["time_sensitive_markers"]) > 0
        assert len(result.details["certainty_phrases"]) > 0


class TestApplyAllFilters:
    """Test integration of all filters."""

    def test_no_filters_triggered_passes(self):
        """Record passing all filters."""
        record = {
            "id": "test_good",
            "prompt": "How to apply?",
            "response": {
                "summary": "You should submit an application. " * 5 + "word " * 40,
                "next_steps": [
                    {"description": "Call the office"},
                    {"description": "Submit documents"},
                    {"description": "Verify status"},
                ],
                "verification_steps": [{"description": "Check website"}],
                "uncertainty_notes": "",
            },
        }
        result = apply_all_filters(record, min_words=50, max_words=500)
        assert result is None

    def test_length_filter_triggered_first(self):
        """Length filter checked first."""
        record = {
            "id": "test",
            "prompt": "Short",
            "response": {
                "summary": "Too short.",
                "next_steps": [],
                "verification_steps": [],
            },
        }
        result = apply_all_filters(record)
        assert result is not None
        assert result.reason_code == "too_short"

    def test_vagueness_filter_triggered_second(self):
        """Vagueness filter checked second (after length passes)."""
        record = {
            "id": "test",
            "prompt": "Question?",
            "response": {
                "summary": (
                    "It depends. Generally, you should do your own research. "
                    "Check online and usually it works. " + "word " * 40
                ),
                "next_steps": [],
                "verification_steps": [],
            },
        }
        result = apply_all_filters(record, min_words=50, max_words=500)
        assert result is not None
        assert result.reason_code == "vague_low_specificity"

    def test_overclaim_filter_triggered_third(self):
        """Overclaim filter checked third (after length and vagueness pass)."""
        record = {
            "id": "test",
            "prompt": "Visa?",
            "response": {
                "summary": (
                    "You definitely need a visa. Entry requirements are guaranteed to include a passport. "
                    "You never need to verify this. " + "word " * 30
                ),
                "next_steps": [
                    {"description": "Call embassy"},
                    {"description": "Submit docs"},
                    {"description": "Wait"},
                ],
                "verification_steps": [],
                "uncertainty_notes": "",
            },
        }
        result = apply_all_filters(record, min_words=50, max_words=500)
        assert result is not None
        assert result.reason_code == "overconfident_time_sensitive"

    def test_custom_thresholds_respected(self):
        """Apply_all_filters passes custom thresholds to individual filters."""
        record = {
            "id": "test",
            "prompt": "Hi?",
            "response": {
                "summary": "short",
                "next_steps": [],
                "verification_steps": [],
            },
        }
        # With lenient min_words=1, should pass length check
        result = apply_all_filters(record, min_words=1, max_words=1000)
        # But fails vagueness (no content)
        if result:
            assert result.reason_code in ["vague_low_specificity", "too_long"]


class TestFilterDeterminism:
    """Test deterministic behavior of filters."""

    def test_length_filter_deterministic(self):
        """Length filter produces consistent results."""
        record = {
            "id": "test",
            "prompt": "Question?",
            "response": {
                "summary": "word " * 50,
                "next_steps": [{"description": "action " * 30}],
                "verification_steps": [],
            },
        }
        result1 = check_length(record)
        result2 = check_length(record)

        assert result1.passed == result2.passed
        assert result1.reason_code == result2.reason_code
        assert result1.details["word_count"] == result2.details["word_count"]

    def test_vagueness_filter_deterministic(self):
        """Vagueness filter produces consistent results."""
        record = {
            "id": "test",
            "prompt": "Question?",
            "response": {
                "summary": (
                    "It depends generally. Do your own research and check online. " + "word " * 50
                ),
                "next_steps": [],
                "verification_steps": [],
            },
        }
        result1 = check_vagueness(record)
        result2 = check_vagueness(record)

        assert result1.passed == result2.passed
        assert result1.reason_code == result2.reason_code
        assert result1.details["vague_phrase_count"] == result2.details["vague_phrase_count"]

    def test_overclaim_filter_deterministic(self):
        """Overclaim filter produces consistent results."""
        record = {
            "id": "test",
            "prompt": "Visa?",
            "response": {
                "summary": "Definitely need visa. Never verified. " + "word " * 50,
                "next_steps": [],
                "verification_steps": [],
                "uncertainty_notes": "",
            },
        }
        result1 = check_overclaim(record)
        result2 = check_overclaim(record)

        assert result1.passed == result2.passed
        assert result1.reason_code == result2.reason_code
        assert result1.details["time_sensitive_count"] == result2.details["time_sensitive_count"]

    def test_apply_all_filters_deterministic(self):
        """apply_all_filters produces consistent results."""
        record = {
            "id": "test",
            "prompt": "Question?",
            "response": {
                "summary": (
                    "You definitely should call. Entry requirements are guaranteed. " + "word " * 50
                ),
                "next_steps": [],
                "verification_steps": [],
                "uncertainty_notes": "",
            },
        }
        result1 = apply_all_filters(record)
        result2 = apply_all_filters(record)

        if result1 is None and result2 is None:
            assert True
        else:
            assert result1 is not None and result2 is not None
            assert result1.reason_code == result2.reason_code


class TestFilterResultDataclass:
    """Test FilterResult type."""

    def test_filter_result_creation(self):
        """FilterResult can be created with all fields."""
        result = FilterResult(
            passed=True,
            reason_code="ok",
            details={"count": 42},
        )
        assert result.passed is True
        assert result.reason_code == "ok"
        assert result.details["count"] == 42

    def test_filter_result_rejected(self):
        """FilterResult represents rejection."""
        result = FilterResult(
            passed=False,
            reason_code="too_short",
            details={"word_count": 10, "min": 100},
        )
        assert result.passed is False
        assert result.reason_code == "too_short"
        assert result.details["word_count"] == 10


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_missing_response_field(self):
        """Handles missing response field gracefully."""
        record = {"id": "test", "prompt": "Question"}
        result = check_length(record)
        assert result.passed is False

    def test_none_text_fields(self):
        """Handles None text fields gracefully."""
        record = {
            "id": "test",
            "prompt": None,
            "response": {
                "summary": None,
                "next_steps": None,
                "verification_steps": None,
                "uncertainty_notes": None,
            },
        }
        result = check_length(record)
        assert result.passed is False

    def test_empty_lists(self):
        """Handles empty lists in next_steps/verification_steps."""
        record = {
            "id": "test",
            "prompt": "Question?",
            "response": {
                "summary": "word " * 30,
                "next_steps": [],
                "verification_steps": [],
            },
        }
        result = check_length(record, min_words=25, max_words=200)
        assert result.passed is True

    def test_mixed_step_formats(self):
        """Handles mixed dict and string formats in steps."""
        record = {
            "id": "test",
            "prompt": "Question?",
            "response": {
                "summary": "word " * 20,
                "next_steps": [
                    {"description": "Step one"},
                    "Step two as string",
                ],
                "verification_steps": ["Verify as string"],
            },
        }
        result = check_length(record, min_words=20, max_words=200)
        assert result.passed is True

    def test_case_insensitive_marker_matching(self):
        """Filters match markers case-insensitively."""
        record = {
            "id": "test",
            "prompt": "Visa?",
            "response": {
                "summary": (
                    "DEFINITELY need VISA. Entry REQUIREMENTS guaranteed. "
                    "NEVER verify. " + "word " * 50
                ),
                "next_steps": [],
                "verification_steps": [],
                "uncertainty_notes": "",
            },
        }
        result = check_overclaim(record)
        assert result.passed is False
        assert result.reason_code == "overconfident_time_sensitive"

    def test_unicode_handling(self):
        """Filters handle unicode text."""
        record = {
            "id": "test",
            "prompt": "Vraag?",  # Dutch for question
            "response": {
                "summary": (
                    "U moet absoluut een visum hebben. " +  # Dutch text
                    "Invoervoorwaarden zijn gegarandeerd. " * 30 +  # Entry requirements
                    "word " * 50
                ),
                "next_steps": [],
                "verification_steps": [],
                "uncertainty_notes": "",
            },
        }
        result = check_length(record)
        assert result.passed is True  # Should handle unicode word counting
