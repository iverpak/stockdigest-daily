"""
Quality Review Phase 4: Metadata & Structure Validation

Validates metadata tags (impact, sentiment, reason, relevance) and section placement logic.
Internal validation only - uses only the executive summary JSON itself.
"""

import json
import logging
import time
from typing import Dict, Optional
import google.generativeai as genai

LOG = logging.getLogger(__name__)


PHASE4_QUALITY_REVIEW_PROMPT = """You are a structural analyst reviewing executive summary metadata and organization.

Your task: Validate metadata tag accuracy and section placement logic using ONLY the executive summary itself.

═══════════════════════════════════════════
INPUT
═══════════════════════════════════════════

You will receive:
- Executive summary JSON (with all metadata, bullet content, and context)

You will NOT receive:
- Articles (Phase 1 already validated against articles)
- Filings (Phase 2 already validated against filings)

Your job: Internal consistency validation only.

═══════════════════════════════════════════
CHECK 1: METADATA TAG VERIFICATION
═══════════════════════════════════════════

For each bullet, validate 4 metadata fields against the bullet's own content + context:

SENTIMENT TAG (bullish | bearish | neutral | mixed)

Validation approach:
- Read the bullet content + context text
- What sentiment does the TEXT convey?
- Does the sentiment tag match?

Internal consistency check:
✅ Bullet: "Q3 revenue beat expectations by 15%" → bullish (CONSISTENT)
✅ Bullet: "Regulatory fine $500M for violations" → bearish (CONSISTENT)
❌ Bullet: "Acquired unprofitable target with -131% margin" → bullish (INCONSISTENT - should be bearish)
❌ Bullet: "Beat earnings but lowered guidance" → bullish (INCONSISTENT - should be mixed)

Common issues:
- Acquisition of unprofitable/struggling business tagged bullish (should be bearish/mixed)
- Risk materialization tagged neutral (should be bearish)
- Positive development with major caveat tagged bullish (should be mixed)
- Operational improvement tagged bearish (should be bullish)

Output if issue found:
{
  "field": "sentiment",
  "current_value": "bullish",
  "recommended_value": "bearish",
  "reason": "Bullet and context describe acquiring unprofitable business with -131% EBITDA margin. Text conveys negative business quality despite growth narrative.",
  "text_evidence": "Content mentions acquisition; context states 'target has -131% EBITDA margin, unprofitable business model'",
  "severity": "SERIOUS"
}

IMPACT TAG (high impact | medium impact | low impact)

Validation approach:
- Read bullet content + context for quantification
- Check if quantification supports the impact tag
- If bullet+context provide %, $, or scale data → validate against thresholds
- If no quantification found → note this (cannot validate)

Quantification thresholds:
- high impact: >5% of revenue/costs/exposure OR categorical events (M&A, FDA, CEO fraud)
- medium impact: 1-5% of revenue/costs/exposure
- low impact: <1% of revenue/costs/exposure

Internal consistency check:
✅ Bullet+Context: "$50M contract, represents 8% of revenue" → high impact (CONSISTENT)
✅ Bullet+Context: "Customer renewal, 0.3% of revenue" → low impact (CONSISTENT)
❌ Bullet+Context: "$50M contract, company revenue $25B = 0.2%" → high impact (INCONSISTENT - should be low)
❌ Bullet+Context: "Acquisition $3.4B, company market cap $40B = 8.5%" → low impact (INCONSISTENT - should be high)

If no quantification found in bullet+context:
{
  "field": "impact",
  "current_value": "high impact",
  "issue": "No quantification found in bullet or context to validate impact rating",
  "recommendation": "Add quantification to context, or downgrade to medium/low if truly unquantifiable",
  "severity": "MINOR"
}

Output if quantified issue found:
{
  "field": "impact",
  "current_value": "high impact",
  "recommended_value": "low impact",
  "quantification": "Bullet states $50M contract, context states company revenue $25B. Calculation: $50M / $25B = 0.2%",
  "threshold": "<1% = low impact per framework",
  "severity": "SERIOUS"
}

REASON TAG (2-4 words describing impact type)

Validation approach:
- Reason should describe BUSINESS IMPACT, not the event itself
- Good reasons: "margin compression", "revenue dependency", "capital misallocation", "strategic contraction"
- Bad reasons: "acquisition announced", "customer contract", "membership loss", "portfolio expansion"

Internal consistency check:
✅ Bullet: "Membership declined 5%" + Reason: "revenue contraction" (IMPACT TYPE)
✅ Bullet: "Acquired unprofitable target" + Reason: "capital misallocation" (IMPACT TYPE)
❌ Bullet: "Membership declined 5%" + Reason: "membership loss" (DESCRIBES EVENT)
❌ Bullet: "Announced acquisition" + Reason: "acquisition announced" (DESCRIBES EVENT)

Output if issue found:
{
  "field": "reason",
  "current_value": "membership loss",
  "recommended_values": ["revenue contraction", "market share loss", "churn acceleration"],
  "issue": "Describes the metric/event rather than business impact type",
  "guidance": "Choose based on bullet+context: Revenue impact? Competitive position? Customer retention trend?",
  "severity": "MINOR"
}

RELEVANCE TAG (direct | indirect | none)

Validation approach:
- Read bullet content: Is company explicitly mentioned?
- Read context: Does it discuss company's exposure/impact?
- Validate relevance classification

Internal consistency check:
✅ Bullet: "Company announced partnership" → direct (CONSISTENT)
✅ Bullet: "Industry adopts new regulation" + Context: "Company 15% exposed" → direct (CONSISTENT)
❌ Bullet: "Competitor launches product" + Context: "No company mention" → direct (INCONSISTENT - should be indirect)
❌ Bullet: "Industry trend accelerates" + Context: "No company data" → direct (INCONSISTENT - should be indirect/none)

Output if issue found:
{
  "field": "relevance",
  "current_value": "direct",
  "recommended_value": "indirect",
  "reason": "Bullet discusses competitor action, context does not quantify company impact or exposure",
  "text_evidence": "Content mentions competitor product launch; context discusses competitor's strategy without company connection",
  "severity": "MINOR"
}

═══════════════════════════════════════════
CHECK 2: SECTION PLACEMENT LOGIC
═══════════════════════════════════════════

Validate bullets are in correct sections by reading bullet content.

SECTION RULES:

major_developments:
- Company is the ACTOR or TARGET
- M&A (company acquiring or being acquired)
- Partnerships (company entering partnership)
- Product launches (company launching)
- Contracts (company winning/losing)
- Leadership changes (company's executives)
- Strategy shifts (company's strategy)
- NOT: Competitor actions, industry trends, analyst opinions, risks

financial_performance:
- Financial/operational METRICS about the company
- Earnings, revenue, guidance, margins, production numbers
- Company's operational performance data
- NOT: Analyst price targets (those go in wall_street_sentiment)

risk_factors:
- Risks, threats, negative developments affecting company
- Regulatory actions against company
- Lawsuits involving company
- Operational failures at company
- NOT: Competitor advantages (those go in competitive_industry_dynamics)

wall_street_sentiment:
- Analyst ratings, price targets, research reports
- Sell-side opinions about the company
- Investor sentiment from analysts
- NOT: Company's own guidance or management commentary

competitive_industry_dynamics:
- Competitor actions (even if company not mentioned)
- Industry trends and market dynamics
- Regulatory changes affecting sector
- Competitive landscape shifts
- NOT: Company-specific actions (those go in major_developments)

upcoming_catalysts:
- Future events with specific dates
- Earnings dates, votes, regulatory deadlines, product launches
- Must be forward-looking with date
- NOT: Past events or events without dates

key_variables:
- Metrics to monitor going forward
- Forward-looking tracking points
- NOT: Historical metrics or past events

MISPLACEMENT DETECTION:

Check each bullet's content against section rules.

Output if misplaced:
{
  "bullet_id": "competitor_expands",
  "current_section": "risk_factors",
  "correct_section": "competitive_industry_dynamics",
  "rule_violated": "Risk Factors should contain company risks/threats, not competitor actions",
  "text_evidence": "Bullet content describes competitor fleet expansion, not company risk event",
  "severity": "MINOR"
}

DUPLICATE DETECTION:

Identify if the same theme appears in multiple bullets across sections.

Output if duplicate:
{
  "theme": "Membership decline",
  "appears_in_sections": ["major_developments", "risk_factors"],
  "bullet_ids": ["membership_q3", "membership_risk"],
  "recommendation": "Keep factual metric in major_developments, remove from risk_factors to avoid redundancy",
  "severity": "MINOR"
}

═══════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════

Return valid JSON:

{
  "summary": {
    "ticker": "AAPL",
    "bullets_reviewed": 25,

    "issue_counts": {
      "metadata_issues": 8,
      "section_placement_issues": 2
    }
  },

  "metadata_verification": {
    "bullets_with_issues": [
      {
        "bullet_id": "acquisition_announced",
        "section": "major_developments",
        "issues": [
          {
            "field": "sentiment",
            "current_value": "bullish",
            "recommended_value": "bearish",
            "reason": "Bullet and context describe unprofitable acquisition with negative margins",
            "text_evidence": "Content: acquired Amedisys; Context: -131% EBITDA margin",
            "severity": "SERIOUS"
          }
        ]
      }
    ],
    "bullets_no_issues": ["q3_earnings", "customer_renewal", "guidance_raise"]
  },

  "section_placement": {
    "misplaced_bullets": [
      {
        "bullet_id": "competitor_expands",
        "current_section": "risk_factors",
        "correct_section": "competitive_industry_dynamics",
        "rule_violated": "Risk Factors = company risks, not competitor actions",
        "text_evidence": "Bullet describes competitor fleet expansion",
        "severity": "MINOR"
      }
    ],
    "duplicate_themes": [
      {
        "theme": "Membership decline",
        "appears_in_sections": ["major_developments", "risk_factors"],
        "bullet_ids": ["membership_q3", "membership_risk"],
        "recommendation": "Keep in major_developments, remove from risk_factors",
        "severity": "MINOR"
      }
    ]
  }
}

IMPORTANT NOTES:
- Paragraph sections (bottom_line, upside_scenario, downside_scenario) have no metadata tags - skip them
- Only review bullet sections with metadata
- Validate using ONLY the text in bullet content + context fields
- Do NOT reference external articles or filings (Phases 1-2 already did that)
- This is an INTERNAL consistency check

Return ONLY the JSON object, no other text.
"""


def review_phase4_metadata_and_structure(
    ticker: str,
    executive_summary: Dict,
    gemini_api_key: str
) -> Optional[Dict]:
    """
    Review executive summary metadata tags and section placement using Gemini.

    INTERNAL VALIDATION ONLY - uses only the executive summary JSON itself.
    Does not require articles or filings (Phases 1-2 already validated against those).

    Args:
        ticker: Stock ticker
        executive_summary: Full Phase 1+2+3 merged executive summary JSON
        gemini_api_key: Gemini API key

    Returns:
        {
            "summary": {
                "ticker": str,
                "bullets_reviewed": int,
                "issue_counts": {...},
                "generation_time_ms": int
            },
            "metadata_verification": {...},
            "section_placement": {...}
        }
    """
    if not gemini_api_key:
        LOG.error("Gemini API key not configured")
        return None

    try:
        genai.configure(api_key=gemini_api_key)

        # Build review prompt
        LOG.info(f"[{ticker}] Building Phase 4 quality review prompt")

        # Format executive summary
        summary_text = "═══════════════════════════════════════════\n"
        summary_text += "EXECUTIVE SUMMARY TO REVIEW:\n"
        summary_text += "═══════════════════════════════════════════\n\n"
        summary_text += json.dumps(executive_summary, indent=2)

        # Combine into full prompt
        full_prompt = PHASE4_QUALITY_REVIEW_PROMPT + "\n" + summary_text

        # Log token estimate
        char_count = len(full_prompt)
        token_estimate = char_count // 4
        LOG.info(f"[{ticker}] Phase 4 quality review prompt: {char_count:,} chars (~{token_estimate:,} tokens)")

        # Call Gemini 2.5 Flash
        LOG.info(f"[{ticker}] Calling Gemini 2.5 Flash for Phase 4 quality review")
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

        # Add generation time and ticker to summary
        if "summary" not in result_json:
            result_json["summary"] = {}
        result_json["summary"]["generation_time_ms"] = generation_time
        result_json["summary"]["ticker"] = ticker

        # Calculate issue counts
        metadata_issues = len(result_json.get("metadata_verification", {}).get("bullets_with_issues", []))
        placement_issues = (
            len(result_json.get("section_placement", {}).get("misplaced_bullets", [])) +
            len(result_json.get("section_placement", {}).get("duplicate_themes", []))
        )

        result_json["summary"]["issue_counts"] = {
            "metadata_issues": metadata_issues,
            "section_placement_issues": placement_issues
        }

        LOG.info(f"✅ [{ticker}] Phase 4 quality review complete: "
                f"{metadata_issues} metadata issues, {placement_issues} placement issues")

        return result_json

    except json.JSONDecodeError as e:
        LOG.error(f"[{ticker}] Failed to parse Gemini Phase 4 response as JSON: {e}")
        return None
    except Exception as e:
        LOG.error(f"[{ticker}] Phase 4 quality review failed: {e}")
        import traceback
        LOG.error(traceback.format_exc())
        return None
