#!/usr/bin/env python3
"""
Test script for Phase 2 validation with partial acceptance.

Tests various edge cases to ensure the validation properly filters
out incomplete bullets while accepting valid ones.
"""

import sys
sys.path.insert(0, '/workspaces/quantbrief-daily')

from modules.executive_summary_phase2 import validate_phase2_json


def test_all_valid():
    """Test case: All bullets valid"""
    enrichments = {
        "bullet_1": {
            "impact": "high impact",
            "sentiment": "bullish",
            "reason": "delivery outperformance",
            "relevance": "direct",
            "context": "Customer represents 28% of Q2 revenue per 10-Q"
        },
        "bullet_2": {
            "impact": "medium impact",
            "sentiment": "neutral",
            "reason": "routine churn",
            "relevance": "indirect",
            "context": "No relevant filing context found for this development"
        }
    }

    is_valid, msg, valid = validate_phase2_json(enrichments)

    print("TEST: All valid bullets")
    print(f"  is_valid: {is_valid}")
    print(f"  message: {msg}")
    print(f"  valid_count: {len(valid)}")
    assert is_valid == True, "Should accept all valid bullets"
    assert len(valid) == 2, "Should return all 2 bullets"
    print("  ✅ PASSED\n")


def test_one_missing_field():
    """Test case: One bullet missing 'reason' field (original UNH issue)"""
    enrichments = {
        "bullet_1": {
            "impact": "high impact",
            "sentiment": "bullish",
            "reason": "delivery outperformance",
            "relevance": "direct",
            "context": "Customer represents 28% of Q2 revenue per 10-Q"
        },
        "cvs_health_q3_beat_guidance_raised": {
            "impact": "high impact",
            "sentiment": "bullish",
            # "reason": "MISSING!",  # This was the UNH issue
            "relevance": "direct",
            "context": "Q3 beat expectations"
        },
        "bullet_3": {
            "impact": "medium impact",
            "sentiment": "bearish",
            "reason": "margin compression",
            "relevance": "direct",
            "context": "Margin declined to 5.8% per Q3 10-Q"
        }
    }

    is_valid, msg, valid = validate_phase2_json(enrichments)

    print("TEST: One bullet missing 'reason' field (UNH issue)")
    print(f"  is_valid: {is_valid}")
    print(f"  message: {msg}")
    print(f"  valid_count: {len(valid)}")
    assert is_valid == True, "Should accept partial results"
    assert len(valid) == 2, "Should filter out 1 bad bullet, keep 2 good ones"
    assert "cvs_health_q3_beat_guidance_raised" not in valid, "Should exclude bullet with missing field"
    assert "bullet_1" in valid, "Should keep valid bullet_1"
    assert "bullet_3" in valid, "Should keep valid bullet_3"
    print("  ✅ PASSED - Filtered out bad bullet, kept 2 good ones\n")


def test_multiple_missing_fields():
    """Test case: Multiple bullets with various missing fields"""
    enrichments = {
        "good_1": {
            "impact": "high impact",
            "sentiment": "bullish",
            "reason": "delivery outperformance",
            "relevance": "direct",
            "context": "Context here"
        },
        "bad_1": {
            "impact": "high impact",
            # Missing sentiment, reason, relevance, context
        },
        "bad_2": {
            # Missing everything
        },
        "good_2": {
            "impact": "low impact",
            "sentiment": "neutral",
            "reason": "routine update",
            "relevance": "indirect",
            "context": "No relevant filing context found for this development"
        }
    }

    is_valid, msg, valid = validate_phase2_json(enrichments)

    print("TEST: Multiple bullets with various missing fields")
    print(f"  is_valid: {is_valid}")
    print(f"  message: {msg}")
    print(f"  valid_count: {len(valid)}")
    assert is_valid == True, "Should accept partial results"
    assert len(valid) == 2, "Should keep only 2 valid bullets"
    assert "good_1" in valid and "good_2" in valid, "Should keep both good bullets"
    print("  ✅ PASSED - Kept 2 valid, filtered 2 invalid\n")


def test_invalid_enum_values():
    """Test case: Invalid enum values for impact/sentiment/relevance"""
    enrichments = {
        "good": {
            "impact": "high impact",
            "sentiment": "bullish",
            "reason": "good reason",
            "relevance": "direct",
            "context": "Context"
        },
        "bad_impact": {
            "impact": "very high",  # Invalid
            "sentiment": "bullish",
            "reason": "reason",
            "relevance": "direct",
            "context": "Context"
        },
        "bad_sentiment": {
            "impact": "high impact",
            "sentiment": "positive",  # Invalid (should be bullish/bearish/neutral)
            "reason": "reason",
            "relevance": "direct",
            "context": "Context"
        },
        "bad_relevance": {
            "impact": "medium impact",
            "sentiment": "neutral",
            "reason": "reason",
            "relevance": "tangential",  # Invalid (should be direct/indirect/none)
            "context": "Context"
        }
    }

    is_valid, msg, valid = validate_phase2_json(enrichments)

    print("TEST: Invalid enum values")
    print(f"  is_valid: {is_valid}")
    print(f"  message: {msg}")
    print(f"  valid_count: {len(valid)}")
    assert is_valid == True, "Should accept partial results"
    assert len(valid) == 1, "Should keep only 1 valid bullet"
    assert "good" in valid, "Should keep the valid bullet"
    print("  ✅ PASSED - Filtered out 3 bullets with invalid enums\n")


def test_empty_string_fields():
    """Test case: Empty string values in required vs optional fields"""
    enrichments = {
        "good": {
            "impact": "high impact",
            "sentiment": "bullish",
            "reason": "delivery outperformance",
            "relevance": "direct",
            "context": "Context here"
        },
        "empty_reason": {
            "impact": "high impact",
            "sentiment": "bullish",
            "reason": "",  # Empty string in REQUIRED field - should be filtered
            "relevance": "direct",
            "context": "Context"
        },
        "empty_context": {
            "impact": "medium impact",
            "sentiment": "neutral",
            "reason": "routine",
            "relevance": "direct",
            "context": ""  # Empty string in OPTIONAL field - should be ACCEPTED
        }
    }

    is_valid, msg, valid = validate_phase2_json(enrichments)

    print("TEST: Empty string fields (required vs optional)")
    print(f"  is_valid: {is_valid}")
    print(f"  message: {msg}")
    print(f"  valid_count: {len(valid)}")
    assert is_valid == True, "Should accept partial results"
    assert len(valid) == 2, "Should keep bullets with empty OPTIONAL fields, filter empty REQUIRED fields"
    assert "good" in valid, "Should keep bullet with all fields"
    assert "empty_context" in valid, "Should keep bullet with empty context (context is optional)"
    assert "empty_reason" not in valid, "Should filter bullet with empty reason (required field)"
    print("  ✅ PASSED - Context is optional, required fields still enforced\n")


def test_all_invalid():
    """Test case: All bullets invalid (complete failure)"""
    enrichments = {
        "bad_1": {
            "impact": "high impact",
            # Missing everything else
        },
        "bad_2": {
            "sentiment": "bullish",
            # Missing everything else
        }
    }

    is_valid, msg, valid = validate_phase2_json(enrichments)

    print("TEST: All bullets invalid (complete failure)")
    print(f"  is_valid: {is_valid}")
    print(f"  message: {msg}")
    print(f"  valid_count: {len(valid)}")
    assert is_valid == False, "Should reject when no valid bullets"
    assert len(valid) == 0, "Should return empty dict"
    print("  ✅ PASSED - Correctly rejected all invalid bullets\n")


def test_empty_dict():
    """Test case: Empty enrichments dict"""
    enrichments = {}

    is_valid, msg, valid = validate_phase2_json(enrichments)

    print("TEST: Empty enrichments dict")
    print(f"  is_valid: {is_valid}")
    print(f"  message: {msg}")
    print(f"  valid_count: {len(valid)}")
    assert is_valid == False, "Should reject empty dict"
    assert len(valid) == 0, "Should return empty dict"
    print("  ✅ PASSED - Correctly rejected empty input\n")


def test_not_a_dict():
    """Test case: Enrichments is not a dict"""
    enrichments = "not a dict"

    is_valid, msg, valid = validate_phase2_json(enrichments)

    print("TEST: Enrichments not a dict")
    print(f"  is_valid: {is_valid}")
    print(f"  message: {msg}")
    print(f"  valid_count: {len(valid)}")
    assert is_valid == False, "Should reject non-dict input"
    assert len(valid) == 0, "Should return empty dict"
    print("  ✅ PASSED - Correctly rejected non-dict input\n")


def test_bullet_not_a_dict():
    """Test case: Individual bullet value is not a dict"""
    enrichments = {
        "good": {
            "impact": "high impact",
            "sentiment": "bullish",
            "reason": "delivery outperformance",
            "relevance": "direct",
            "context": "Context"
        },
        "bad": "not a dict"
    }

    is_valid, msg, valid = validate_phase2_json(enrichments)

    print("TEST: Individual bullet not a dict")
    print(f"  is_valid: {is_valid}")
    print(f"  message: {msg}")
    print(f"  valid_count: {len(valid)}")
    assert is_valid == True, "Should accept partial results"
    assert len(valid) == 1, "Should keep only valid bullet"
    assert "good" in valid, "Should keep the valid bullet"
    print("  ✅ PASSED - Filtered out non-dict bullet\n")


def test_missing_context_field():
    """Test case: Bullets with tags but missing context field entirely (optional field)"""
    enrichments = {
        "has_context": {
            "impact": "high impact",
            "sentiment": "bullish",
            "reason": "delivery outperformance",
            "relevance": "direct",
            "context": "Customer 28% Q2 revenue per 10-Q"
        },
        "no_context": {
            "impact": "medium impact",
            "sentiment": "bearish",
            "reason": "pricing pressure",
            "relevance": "direct"
            # No context field at all - should still be accepted
        },
        "empty_context": {
            "impact": "low impact",
            "sentiment": "neutral",
            "reason": "routine update",
            "relevance": "indirect",
            "context": ""  # Empty context - should be accepted
        }
    }

    is_valid, msg, valid = validate_phase2_json(enrichments)

    print("TEST: Bullets with missing context field (80% value vs 0%)")
    print(f"  is_valid: {is_valid}")
    print(f"  message: {msg}")
    print(f"  valid_count: {len(valid)}")
    assert is_valid == True, "Should accept bullets even without context"
    assert len(valid) == 3, "Should accept all 3 bullets (context is optional)"
    assert "has_context" in valid, "Should keep bullet with context"
    assert "no_context" in valid, "Should keep bullet without context field"
    assert "empty_context" in valid, "Should keep bullet with empty context"
    # Verify context was set to empty string for bullets without it
    assert valid["no_context"]["context"] == "", "Should set missing context to empty string"
    assert valid["empty_context"]["context"] == "", "Should preserve empty context"
    print("  ✅ PASSED - Tags without context = 80% value instead of 0%!\n")


def test_realistic_scenario():
    """Test case: Realistic scenario with 19/20 valid bullets (like UNH)"""
    # Generate 19 valid bullets
    enrichments = {}
    for i in range(1, 20):
        enrichments[f"bullet_{i}"] = {
            "impact": "medium impact",
            "sentiment": "neutral",
            "reason": f"reason_{i}",
            "relevance": "direct",
            "context": f"Context for bullet {i}"
        }

    # Add 1 invalid bullet (missing reason)
    enrichments["cvs_health_q3_beat_guidance_raised"] = {
        "impact": "high impact",
        "sentiment": "bullish",
        # Missing "reason"
        "relevance": "direct",
        "context": "Q3 beat expectations"
    }

    is_valid, msg, valid = validate_phase2_json(enrichments)

    print("TEST: Realistic scenario - 19/20 valid bullets (UNH scenario)")
    print(f"  is_valid: {is_valid}")
    print(f"  message: {msg}")
    print(f"  valid_count: {len(valid)}")
    assert is_valid == True, "Should accept partial results"
    assert len(valid) == 19, "Should keep 19 valid bullets"
    assert "cvs_health_q3_beat_guidance_raised" not in valid, "Should filter out bad bullet"
    assert "Accepted 19/20" in msg, "Message should show 19/20"
    print("  ✅ PASSED - Got 95% value instead of 0%!\n")


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 2 VALIDATION TEST SUITE - Partial Acceptance Logic")
    print("=" * 70)
    print()

    try:
        test_all_valid()
        test_one_missing_field()
        test_multiple_missing_fields()
        test_invalid_enum_values()
        test_empty_string_fields()
        test_all_invalid()
        test_empty_dict()
        test_not_a_dict()
        test_bullet_not_a_dict()
        test_missing_context_field()
        test_realistic_scenario()

        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✅ Partial acceptance working correctly")
        print("  ✅ Invalid bullets filtered out")
        print("  ✅ Valid bullets preserved")
        print("  ✅ Context field is optional (tags without context accepted)")
        print("  ✅ Edge cases handled properly")
        print("  ✅ UNH scenario (19/20 valid) now returns 95% value!")
        print()

    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        sys.exit(1)
