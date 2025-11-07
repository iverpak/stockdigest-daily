"""
Quality Review Phase 2 Module

Verifies filing context accuracy against 10-K, 10-Q, and Transcript sources.
Separate from Phase 1 (article verification) to maintain focused verification tasks.
"""

import json
import logging
import time
from typing import Dict, List, Optional
import google.generativeai as genai

LOG = logging.getLogger(__name__)


# Gemini Phase 2 verification prompt
PHASE2_QUALITY_REVIEW_PROMPT = """You are a financial data analyst reviewing filing context fields for accuracy.

Your task: Verify every context field in the executive summary is accurately sourced from the 10-K, 10-Q, or Transcript provided.

═══════════════════════════════════════════
WHAT YOU'RE VERIFYING
═══════════════════════════════════════════

You will receive a merged Phase 1 + Phase 2 executive summary JSON from production.

Context fields appear in two places:
1. BULLET CONTEXTS - Each bullet object has a "context" field added by Phase 2 enrichment
2. SCENARIO CONTEXTS - Each paragraph section object has a "context" field added by Phase 2 enrichment

INPUT STRUCTURE:

BULLET SECTIONS (major_developments, financial_performance, risk_factors, competitive_industry_dynamics, upcoming_catalysts):
- Each section contains an array of bullet objects
- Each bullet has: bullet_id, topic_label, content, context (Phase 2 enrichment), impact, sentiment, etc.
- Extract the bullet_id from each bullet object
- Verify the context field against filing sources

PARAGRAPH SECTIONS (bottom_line, upside_scenario, downside_scenario):
- Each section is an object with: content, context (Phase 2 enrichment)
- Use section_name as the identifier (e.g., "bottom_line")
- Verify the context field against filing sources

Example bullet object:
{
  "bullet_id": "market_inflection_point",
  "topic_label": "Market inflection point",
  "content": "Company announced $3.4B acquisition...",
  "context": "Amedisys operates in home health (40% of revenue) with $2.1B revenue per 10-K..."
}

Your job: Verify the context field against the filing sources.

═══════════════════════════════════════════
VERIFICATION RULES
═══════════════════════════════════════════

For each context field, classify verification status:

✅ ACCURATE - Context data verified in filing sources
   - Numbers match exactly (or within rounding: $3.5B = $3,499M)
   - Facts clearly stated in claimed filing section
   - Context relates to its main bullet/paragraph
   - Context adds new information beyond main bullet

🔴 ISSUE - Context has errors (must identify error type below)

═══════════════════════════════════════════
ERROR TYPES TO DETECT
═══════════════════════════════════════════

When status is ISSUE, identify the error type:

🔴 CRITICAL ERRORS (Target: 0%):

1. Context Fabrication
   - Numbers/facts not found in any filing
   - Example: Context says "$76B backlog" but 10-Q shows "$64B"
   - Check: Every number must appear in filings (allow rounding: $3.5B = $3,499M)

2. Context-Bullet Contradiction
   - Context contradicts its own main bullet
   - Example: Bullet "margins expanding" + Context "margins compressed YoY per 10-Q"
   - Check: Context must be consistent with bullet's direction/claim

3. Wrong Context Source
   - Context cites wrong filing section
   - Example: Context says "per REVENUE STREAMS" but data is in DEBT SCHEDULE
   - Check: Verify data in claimed section (or at least verify data exists somewhere in filing)

🟠 SERIOUS ERRORS (Target: <1%):

4. Context Irrelevance
   - Context discusses unrelated topic
   - Example: Bullet "acquisition announced" + Context "R&D capitalization policy"
   - Check: Context must thematically connect to bullet topic

5. Missing Critical Baseline
   - Context discusses change without baseline
   - Example: "Members declined" without stating current member count
   - Check: Material changes need "from X to Y" structure

🟡 MINOR ERRORS (Target: <5%):

6. Context Doesn't Add Value (Within-Bullet Redundancy)
   - Context just restates main bullet
   - Example: Bullet "Q3 revenue $23.3B, up 30%" + Context "Q3 revenue $23.3B, up 30% per 10-Q"
   - Check: Context should add NEW info (trends, scale, rationale)

7. Excessive Cross-Context Duplication
   - Same filing fact appears in 5+ different contexts
   - Example: "$76B defense backlog" mentioned in 6 different bullet contexts
   - Note: 2-3 repetitions is ACCEPTABLE (self-contained bullets), 5+ is excessive
   - Check: Flag only if same specific fact repeated 5+ times

═══════════════════════════════════════════
SPECIAL INSTRUCTIONS
═══════════════════════════════════════════

- Allow rounding: $3.5B = $3,499M = $3.499 billion (all acceptable)
- Company name variations OK: "Boeing" = "The Boeing Company"
- Date formats can vary: Q3 2025 = Third Quarter 2025 = 3Q25
- If context cites specific section (e.g., "per 10-Q REVENUE STREAMS"), verify data is in that section
- If context just says "per 10-Q", data can be anywhere in 10-Q
- Scenario contexts (Bottom Line, Upside, Downside) provide strategic synthesis - same accuracy standards apply

═══════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════

Return valid JSON with this exact structure:

{
  "contexts": [
    {
      "section_name": "bottom_line" | "major_developments" | "financial_performance" | "risk_factors" | "wall_street_sentiment" | "competitive_industry_dynamics" | "upcoming_catalysts" | "upside_scenario" | "downside_scenario" | "key_variables",
      "bullet_id": "extracted from input JSON",
      "context_text": "Full context text here",
      "status": "ACCURATE" | "ISSUE",
      "error_type": "Context Fabrication" | "Context-Bullet Contradiction" | "Wrong Context Source" | "Context Irrelevance" | "Missing Critical Baseline" | "Context Doesn't Add Value" | "Excessive Cross-Context Duplication" | null,
      "severity": "CRITICAL" | "SERIOUS" | "MINOR" | null,
      "evidence": ["Filing location where data was found or should be found"],
      "notes": "Explanation of verification result"
    }
  ]
}

CRITICAL RULES FOR bullet_id:
- For BULLET sections: Extract bullet_id from the bullet object in input JSON
  Example: input["sections"]["major_developments"][0]["bullet_id"] → "market_inflection_point"
- For PARAGRAPH sections: Use section_name as bullet_id
  Example: "bottom_line", "upside_scenario", "downside_scenario"
- Each context verification should have exactly ONE bullet_id that matches the source bullet/paragraph

Example output for bullet:
{
  "section_name": "major_developments",
  "bullet_id": "market_inflection_point",  ← Extracted from input JSON
  "context_text": "Q3 leasing 62M sqft...",
  "status": "ACCURATE",
  ...
}

Example output for paragraph:
{
  "section_name": "bottom_line",
  "bullet_id": "bottom_line",  ← Use section_name
  "context_text": "Q3 2025 rental revenue $2,054M...",
  "status": "ACCURATE",
  ...
}

Review ALL context fields in ALL 10 sections (even if some sections have no contexts).

Return ONLY the JSON object, no other text.
"""


def review_phase2_context_quality(
    ticker: str,
    executive_summary: Dict,
    filings: Dict,
    gemini_api_key: str
) -> Optional[Dict]:
    """
    Review executive summary context fields against filing sources using Gemini.

    Args:
        ticker: Stock ticker
        executive_summary: Full Phase 1 + Phase 2 merged executive summary JSON
        filings: Dict with keys '10k', '10q', 'transcript' (from _fetch_available_filings)
        gemini_api_key: Gemini API key

    Returns:
        {
            "summary": {
                "ticker": str,
                "total_contexts": int,
                "accurate": int,
                "issues": int,
                "errors_by_severity": {"CRITICAL": int, "SERIOUS": int, "MINOR": int},
                "generation_time_ms": int
            },
            "contexts": [...]  # Detailed review by context
        }
    """
    if not gemini_api_key:
        LOG.error("Gemini API key not configured")
        return None

    if not filings:
        LOG.warning(f"[{ticker}] No filings available for Phase 2 verification")
        return None

    try:
        genai.configure(api_key=gemini_api_key)

        # Build review prompt
        LOG.info(f"[{ticker}] Building Phase 2 quality review prompt")

        # Format executive summary with contexts
        summary_text = "═══════════════════════════════════════════\n"
        summary_text += "EXECUTIVE SUMMARY TO REVIEW:\n"
        summary_text += "═══════════════════════════════════════════\n\n"
        summary_text += json.dumps(executive_summary, indent=2)

        # Format filing sources
        filings_text = "\n\n═══════════════════════════════════════════\n"
        filings_text += "FILING SOURCES:\n"
        filings_text += "═══════════════════════════════════════════\n\n"

        # Add 10-K if available
        if '10k' in filings:
            k = filings['10k']
            year = k.get('fiscal_year', 'Unknown')
            company = k.get('company_name', ticker)
            filings_text += f"10-K FILING (Fiscal Year {year}):\n\n"
            filings_text += f"{k['text']}\n\n\n"

        # Add 10-Q if available
        if '10q' in filings:
            q = filings['10q']
            quarter = q.get('fiscal_quarter', 'Unknown')
            year = q.get('fiscal_year', 'Unknown')
            company = q.get('company_name', ticker)
            filings_text += f"10-Q FILING ({quarter} {year}):\n\n"
            filings_text += f"{q['text']}\n\n\n"

        # Add Transcript if available
        if 'transcript' in filings:
            t = filings['transcript']
            quarter = t.get('quarter', 'Unknown')
            year = t.get('year', 'Unknown')
            company = t.get('company_name', ticker)
            filings_text += f"EARNINGS TRANSCRIPT ({quarter} {year}):\n\n"
            filings_text += f"{t['text']}\n\n\n"

        # Combine into full prompt
        full_prompt = PHASE2_QUALITY_REVIEW_PROMPT + "\n" + summary_text + filings_text

        # Log token estimate
        char_count = len(full_prompt)
        token_estimate = char_count // 4
        LOG.info(f"[{ticker}] Phase 2 quality review prompt: {char_count:,} chars (~{token_estimate:,} tokens)")

        # Call Gemini 2.5 Flash
        LOG.info(f"[{ticker}] Calling Gemini 2.5 Flash for Phase 2 quality review")
        model = genai.GenerativeModel('gemini-2.5-flash')

        generation_config = {
            "temperature": 0.0,  # Maximum determinism
            "response_mime_type": "application/json"  # Force JSON output
        }

        start_time = time.time()
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        generation_time = int((time.time() - start_time) * 1000)  # ms

        # Parse response
        result_text = response.text
        result_json = json.loads(result_text)

        # Calculate summary statistics
        total_contexts = 0
        accurate = 0
        issues = 0
        errors_by_severity = {"CRITICAL": 0, "SERIOUS": 0, "MINOR": 0}

        for context in result_json.get("contexts", []):
            total_contexts += 1
            status = context.get("status")
            severity = context.get("severity")

            if status == "ACCURATE":
                accurate += 1
            elif status == "ISSUE":
                issues += 1

            if severity:
                errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1

        # Build final result
        final_result = {
            "summary": {
                "ticker": ticker,
                "total_contexts": total_contexts,
                "accurate": accurate,
                "issues": issues,
                "errors_by_severity": errors_by_severity,
                "generation_time_ms": generation_time
            },
            "contexts": result_json.get("contexts", [])
        }

        LOG.info(f"✅ [{ticker}] Phase 2 quality review complete: {accurate}/{total_contexts} accurate, "
                f"{issues} issues, {errors_by_severity['CRITICAL']} critical errors")

        return final_result

    except json.JSONDecodeError as e:
        LOG.error(f"[{ticker}] Failed to parse Gemini Phase 2 response as JSON: {e}")
        return None
    except Exception as e:
        LOG.error(f"[{ticker}] Phase 2 quality review failed: {e}")
        import traceback
        LOG.error(traceback.format_exc())
        return None
