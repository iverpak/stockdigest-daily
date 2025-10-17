#!/usr/bin/env python3
"""
Test script to verify executive summary parsing and rendering fixes.
Tests both quiet day and material news day scenarios.
"""

import sys
sys.path.insert(0, '/workspaces/quantbrief-daily')

from app import parse_executive_summary_sections, build_executive_summary_html

# Test 1: Quiet Day
print("=" * 80)
print("TEST 1: QUIET DAY")
print("=" * 80)

quiet_day_summary = """📌 BOTTOM LINE
QUIET DAY - NO MATERIAL DEVELOPMENTS
No company-specific news, regulatory updates, or competitive developments for AAPL reported October 17, 2025.
"""

sections_quiet = parse_executive_summary_sections(quiet_day_summary)
print("\nParsed sections (quiet day):")
for section, content in sections_quiet.items():
    if content:
        print(f"  {section}: {content}")

html_quiet_with_emojis = build_executive_summary_html(sections_quiet, strip_emojis=False)
html_quiet_no_emojis = build_executive_summary_html(sections_quiet, strip_emojis=True)

print("\n✅ Email #2 HTML (with emojis):")
print(html_quiet_with_emojis[:200] + "...")
print("\n✅ Email #3 HTML (no emojis):")
print(html_quiet_no_emojis[:200] + "...")

# Test 2: Material News Day
print("\n" + "=" * 80)
print("TEST 2: MATERIAL NEWS DAY")
print("=" * 80)

material_news_summary = """📌 BOTTOM LINE
Apple announced Q3 earnings with revenue beat. CEO highlighted AI chip progress. Stock up 3% on strong guidance.

🔴 MAJOR DEVELOPMENTS
• Earnings beat (bullish, revenue growth): Q3 revenue $95.2B (+8% YoY), beat consensus $93.1B per Bloomberg; EPS $1.47 vs $1.42 expected; full-year guidance raised to $390B from $375B (Oct 15)

📊 FINANCIAL/OPERATIONAL PERFORMANCE
• Q3 revenue: $95.2B (+8% YoY), beat consensus $93.1B; EPS $1.47 vs $1.42 expected (Oct 15)

⚠️ RISK FACTORS
• Regulatory investigation: EU opened formal antitrust probe into App Store practices; potential fines up to €10B per commission statement (Oct 14)

📈 WALL STREET SENTIMENT
• Goldman Sachs upgraded to Buy from Neutral, $200 target (from $175), citing iPhone demand per analyst report (Oct 15)

⚡ COMPETITIVE/INDUSTRY DYNAMICS
• Samsung Galaxy launch (bearish, competitive threat): Samsung announced Galaxy S25 launching Q1 2026 at $899; analyst stated undercuts iPhone pricing (Oct 14)

📅 UPCOMING CATALYSTS
• Q4 Earnings: January 28 - Holiday quarter results and 2026 guidance per company announcement (Oct 15)

📈 UPSIDE SCENARIO
Goldman Sachs stated earnings beat positions company for continued growth trajectory. Morgan Stanley highlighted iPhone 16 demand demonstrating strong product cycle, projecting sustained revenue expansion. JPMorgan noted AI chip integration creates competitive advantage. Q4 earnings will provide visibility into holiday demand.

📉 DOWNSIDE SCENARIO
Credit Suisse warned EU antitrust probe creates regulatory overhang constraining margins. Barclays noted Samsung pricing pressure threatens market share. Deutsche Bank highlighted supply chain constraints continuing through Q1, potentially pressuring production targets. App Store investigation may reveal compliance challenges.

🔍 KEY VARIABLES TO MONITOR
• Q4 iPhone unit sales: Actual sales vs Goldman's 90M estimate - Timeline: January 28 earnings
• EU antitrust decision: Fine amount and compliance requirements - Timeline: Q2 2026
• AI chip production ramp: Volume targets and yield rates - Timeline: Monthly supply chain reports
"""

sections_material = parse_executive_summary_sections(material_news_summary)
print("\nParsed sections (material news):")
for section, content in sections_material.items():
    if content:
        preview = str(content)[:80] + "..." if len(str(content)) > 80 else str(content)
        print(f"  {section}: {preview}")

html_material_with_emojis = build_executive_summary_html(sections_material, strip_emojis=False)
html_material_no_emojis = build_executive_summary_html(sections_material, strip_emojis=True)

print("\n✅ Email #2 HTML (with emojis) - checking for all sections:")
if "📌 Bottom Line" in html_material_with_emojis:
    print("  ✓ Bottom Line found")
if "🔴 Major Developments" in html_material_with_emojis:
    print("  ✓ Major Developments found")
if "📈 Upside Scenario" in html_material_with_emojis:
    print("  ✓ Upside Scenario found (NEW!)")
if "📉 Downside Scenario" in html_material_with_emojis:
    print("  ✓ Downside Scenario found (NEW!)")
if "🔍 Key Variables to Monitor" in html_material_with_emojis:
    print("  ✓ Key Variables to Monitor found (NEW!)")

print("\n✅ Email #3 HTML (no emojis) - checking for all sections:")
if "Bottom Line" in html_material_no_emojis and "📌" not in html_material_no_emojis:
    print("  ✓ Bottom Line found (no emoji)")
if "Major Developments" in html_material_no_emojis and "🔴" not in html_material_no_emojis:
    print("  ✓ Major Developments found (no emoji)")
if "Upside Scenario" in html_material_no_emojis and "📈" not in html_material_no_emojis.replace("📈 WALL STREET SENTIMENT", ""):
    print("  ✓ Upside Scenario found (no emoji, NEW!)")
if "Downside Scenario" in html_material_no_emojis and "📉" not in html_material_no_emojis:
    print("  ✓ Downside Scenario found (no emoji, NEW!)")
if "Key Variables to Monitor" in html_material_no_emojis and "🔍" not in html_material_no_emojis:
    print("  ✓ Key Variables to Monitor found (no emoji, NEW!)")

# Test 3: Verify upside/downside are paragraphs, key variables are bullets
print("\n" + "=" * 80)
print("TEST 3: FORMATTING VERIFICATION")
print("=" * 80)

print("\n✅ Upside Scenario (should be paragraph, not bullets):")
if "<ul" not in html_material_with_emojis.split("📈 Upside Scenario")[1].split("📉 Downside Scenario")[0]:
    print("  ✓ Rendered as paragraph (no <ul>)")
else:
    print("  ✗ ERROR: Rendered as bullet list!")

print("\n✅ Downside Scenario (should be paragraph, not bullets):")
if "<ul" not in html_material_with_emojis.split("📉 Downside Scenario")[1].split("🔍 Key Variables")[0]:
    print("  ✓ Rendered as paragraph (no <ul>)")
else:
    print("  ✗ ERROR: Rendered as bullet list!")

print("\n✅ Key Variables to Monitor (should be bullets):")
if "<ul" in html_material_with_emojis.split("🔍 Key Variables to Monitor")[1]:
    print("  ✓ Rendered as bullet list")
else:
    print("  ✗ ERROR: Rendered as paragraph!")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETE")
print("=" * 80)
