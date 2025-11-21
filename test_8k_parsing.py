#!/usr/bin/env python3
"""
Test script for 8-K exhibit parsing and filtering.
Tests the fix for WMT-style 5-column format and exhibit filtering.
"""

from modules.company_profiles import should_process_exhibit

def test_exhibit_filtering():
    """Test the should_process_exhibit filter function"""
    print("Testing Exhibit Filtering:")
    print("=" * 80)

    test_cases = [
        # High-value exhibits (should process)
        ("99.1", True, "Press release"),
        ("99.2", True, "Earnings presentation"),
        ("10.1", True, "Material contract"),
        ("4.1", True, "Debt instrument"),
        ("1.1", True, "M&A agreement"),
        ("3.1", True, "Charter amendment"),

        # Zero-value exhibits (should skip)
        ("16.1", False, "Auditor letter"),
        ("23.1", False, "Consent of expert"),
        ("24.1", False, "Power of attorney"),
        ("32.1", False, "Section 906 certification"),
        ("101.1", False, "XBRL file"),
        ("104", False, "Cover page data"),
    ]

    passed = 0
    failed = 0

    for exhibit_num, expected, description in test_cases:
        result = should_process_exhibit(exhibit_num)
        status = "✅" if result == expected else "❌"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"  {status} Exhibit {exhibit_num:6} → {result:5} (expected {expected:5}) - {description}")

    print()
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def test_format_detection():
    """Test format detection for real SEC filings"""
    print("\nTesting Format Detection (manual inspection needed):")
    print("=" * 80)
    print("Run this command to test WMT Nov 19 parsing:")
    print()
    print("  curl -s -A 'StockDigest/1.0' \\")
    print("  'https://www.sec.gov/Archives/edgar/data/104169/000010416924000170/0000104169-24-000170-index.htm' \\")
    print("  | python3 -c '")
    print("from bs4 import BeautifulSoup")
    print("import sys")
    print("from modules.company_profiles import get_all_8k_exhibits")
    print()
    print("exhibits = get_all_8k_exhibits(\"https://www.sec.gov/Archives/edgar/data/104169/000010416924000170/0000104169-24-000170-index.htm\")")
    print("print(f\"Found {len(exhibits)} exhibits:\")")
    print("for ex in exhibits:")
    print("    print(f\"  - Exhibit {ex['exhibit_number']}: {ex['description']}\")")
    print("'")
    print()
    print("Expected output:")
    print("  📋 Format 3 detected (description in col 1): 'PRESS RELEASE' → Exhibit type in col 3: 'EX-99.1'")
    print("  ✅ Found Exhibit 99.1: PRESS RELEASE (544213 bytes)")
    print("  📋 Format 3 detected (description in col 1): 'EARNINGS PRESENTATION' → Exhibit type in col 3: 'EX-99.2'")
    print("  ✅ Found Exhibit 99.2: EARNINGS PRESENTATION (65973 bytes)")
    print("  ✅ Found 2 HTML exhibits total")


if __name__ == "__main__":
    print("8-K Exhibit Parsing & Filtering Test")
    print("=" * 80)
    print()

    # Test filtering
    success = test_exhibit_filtering()

    # Show format detection test
    test_format_detection()

    print()
    if success:
        print("✅ All filtering tests passed!")
        print()
        print("Next steps:")
        print("  1. Test WMT parsing manually (see command above)")
        print("  2. If WMT shows 2 exhibits (99.1 and 99.2), the fix is working!")
        print("  3. Deploy and test with a real 8-K job")
    else:
        print("❌ Some tests failed - review the output above")
