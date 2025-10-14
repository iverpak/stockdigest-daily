#!/usr/bin/env python3
"""
Test script to verify markdown stripping fix for executive summaries.
Tests the scenario where AI adds **bold** markers despite instructions.
"""

import re

def strip_markdown_formatting(text: str) -> str:
    """
    Strip markdown formatting that AI sometimes adds despite instructions.
    Removes: **bold**, *italic*, __underline__, etc.
    """
    # Strip markdown bold (**text** or __text__)
    text = re.sub(r'\*\*([^*]+?)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+?)__', r'\1', text)
    # Strip markdown italic (*text* or _text_)
    text = re.sub(r'\*([^*]+?)\*', r'\1', text)
    text = re.sub(r'_([^_]+?)_', r'\1', text)
    return text

def bold_bullet_labels(text: str) -> str:
    """
    Bold topic labels in bullet points.
    Transforms: "Topic Label: Details" → "<strong>Topic Label:</strong> Details"
    """
    # Strip any markdown formatting first
    text = strip_markdown_formatting(text)

    # Then apply HTML bold tags to topic label (text before colon)
    pattern = r'^([^:]{2,80}?:)(\s)'
    replacement = r'<strong>\1</strong>\2'
    return re.sub(pattern, replacement, text)

# Test cases
print("=" * 80)
print("Testing Markdown Stripping Fix")
print("=" * 80)

test_cases = [
    {
        "name": "Bullet with markdown bold (JPM example)",
        "input": "**Rate cut cycle initiated**: Fed reduced benchmark rate 25 bps to 4.00-4.25% range",
        "expected": "<strong>Rate cut cycle initiated:</strong> Fed reduced benchmark rate 25 bps to 4.00-4.25% range"
    },
    {
        "name": "Bottom Line with markdown bold",
        "input": "**Mixed signals** for JPM as rate cut cycle begins while competitors advance",
        "expected": "Mixed signals for JPM as rate cut cycle begins while competitors advance"
    },
    {
        "name": "Plain text bullet (no markdown)",
        "input": "Competitor dividend increase: Citigroup declared $0.60 quarterly dividend",
        "expected": "<strong>Competitor dividend increase:</strong> Citigroup declared $0.60 quarterly dividend"
    },
    {
        "name": "Multiple markdown bolds",
        "input": "**Topic One**: Details about **important thing** happening",
        "expected": "<strong>Topic One:</strong> Details about important thing happening"
    }
]

all_passed = True

for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test['name']}")
    print(f"  Input:    {test['input']}")

    # For bullets, use bold_bullet_labels
    if ":" in test['input'] and test['name'] != "Bottom Line with markdown bold":
        result = bold_bullet_labels(test['input'])
    else:
        # For Bottom Line (paragraph format), just strip markdown
        result = strip_markdown_formatting(test['input'])

    print(f"  Output:   {result}")
    print(f"  Expected: {test['expected']}")

    if result == test['expected']:
        print("  ✅ PASS")
    else:
        print("  ❌ FAIL")
        all_passed = False

print("\n" + "=" * 80)
if all_passed:
    print("✅ All tests passed!")
else:
    print("❌ Some tests failed")
print("=" * 80)
