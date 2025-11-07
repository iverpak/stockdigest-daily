"""
Quality Review Module

Reviews executive summaries against article summaries to verify accuracy.
Detects fabricated numbers, claims, wrong attributions, and other errors.
"""

import json
import logging
import time
from typing import Dict, List, Optional
import google.generativeai as genai

LOG = logging.getLogger(__name__)


# Gemini verification prompt
QUALITY_REVIEW_PROMPT = """You are a quality assurance analyst reviewing an executive summary for accuracy.

Your task: Verify every sentence in the executive summary is supported by the article summaries provided.

═══════════════════════════════════════════
VERIFICATION RULES
═══════════════════════════════════════════

For each sentence, classify verification status:

✅ SUPPORTED - Sentence content directly stated in article summaries
   - Numbers match exactly
   - Events/facts clearly mentioned
   - Attribution is correct

⚠️ INFERENCE - Logical synthesis from articles, but missing attribution
   - Conclusion drawn from multiple articles
   - Interpretation not explicitly stated
   - Needs "per analyst" or "per management" attribution

🔴 UNSUPPORTED - Not found in articles OR contradicts articles
   - Must identify specific error type (see below)

═══════════════════════════════════════════
ERROR TYPES TO DETECT
═══════════════════════════════════════════

When status is UNSUPPORTED, identify the error type:

🔴 CRITICAL ERRORS (Target: 0%):

1. Fabricated Number
   - Specific number/percentage not in any article
   - Example: "Revenue $5B" but no article mentions $5B
   - Check: All numerical values must appear in sources

2. Fabricated Claim
   - Event or fact not mentioned in any article
   - Example: "Stock buyback announced" but no article discusses buyback
   - Check: All events/actions must have evidence

3. Wrong Attribution
   - Source attribution is incorrect or missing in sources
   - Example: "per CFO" but articles only cite analyst reports
   - Example: "per Goldman" but Goldman never mentioned
   - Check: Attribution source must exist in articles

4. Directional Error
   - Opposite direction from what articles state
   - Example: "Revenue beat" but articles say "revenue missed"
   - Directional pairs to check:
     • beat/miss, exceed/fall short
     • up/down, rose/fell, increased/decreased
     • higher/lower, above/below
     • improved/declined, strengthened/weakened
     • accelerated/decelerated
   - Check: Direction must match sources

🟠 SERIOUS ERRORS (Target: <1%):

5. Company Confusion
   - Fact about competitor/supplier attributed to target company
   - Example: Competitor's revenue stated as target company's revenue
   - Check: Verify the subject of each claim

🟡 MINOR ERRORS (Target: <5%):

6. Inference as Fact
   - Conclusion/interpretation stated without attribution
   - Example: "Positions company for growth" (whose opinion?)
   - Example: "Stock is undervalued" (according to whom?)
   - Should say: "per management" or "per analyst" or "per [source]"
   - Check: Opinion/interpretation needs attribution

═══════════════════════════════════════════
SPECIAL INSTRUCTIONS
═══════════════════════════════════════════

- Be strict with numbers - exact match required
- Company name variations are OK (e.g., "Prologis" = "Prologis Inc" = "PLD")
- Date formats can vary (Q3 2024 = Third Quarter 2024 = 3Q24)
- Synthesis is OK if properly attributed and logical
- Multiple articles supporting same claim = strong evidence
- If uncertain, mark as UNSUPPORTED and explain why

═══════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════

Return valid JSON with this exact structure:

{
  "sections": [
    {
      "section_name": "bottom_line",
      "sentences": [
        {
          "text": "Full sentence text here",
          "status": "SUPPORTED" | "INFERENCE" | "UNSUPPORTED",
          "error_type": "Fabricated Number" | "Fabricated Claim" | "Wrong Attribution" | "Directional Error" | "Company Confusion" | "Inference as Fact" | null,
          "severity": "CRITICAL" | "SERIOUS" | "MINOR" | null,
          "evidence": ["Article X mentions...", "Article Y states..."],
          "notes": "Explanation of verification result"
        }
      ]
    }
  ]
}

Review ALL 10 sections in this order:
1. bottom_line (paragraph)
2. major_developments (bullets)
3. financial_performance (bullets)
4. risk_factors (bullets)
5. wall_street_sentiment (bullets)
6. competitive_industry_dynamics (bullets)
7. upcoming_catalysts (bullets)
8. upside_scenario (paragraph)
9. downside_scenario (paragraph)
10. key_variables (bullets)

For paragraphs: Split into sentences and review each sentence separately.
For bullets: Review the main content only (ignore filing_hints, filing_keywords, bullet_id).

Return ONLY the JSON object, no other text.
"""


def review_executive_summary_quality(
    ticker: str,
    phase1_json: Dict,
    article_summaries: List[str],
    gemini_api_key: str
) -> Optional[Dict]:
    """
    Review executive summary quality against article summaries using Gemini.

    Args:
        ticker: Stock ticker
        phase1_json: Phase 1 executive summary JSON
        article_summaries: List of ai_summary texts from articles
        gemini_api_key: Gemini API key

    Returns:
        {
            "summary": {
                "ticker": str,
                "total_sentences": int,
                "supported": int,
                "inference": int,
                "unsupported": int,
                "errors_by_severity": {"CRITICAL": int, "SERIOUS": int, "MINOR": int}
            },
            "sections": [...]  # Detailed review by section
        }
    """
    if not gemini_api_key:
        LOG.error("Gemini API key not configured")
        return None

    try:
        genai.configure(api_key=gemini_api_key)

        # Build review prompt
        LOG.info(f"[{ticker}] Building quality review prompt")

        # Format executive summary sections
        summary_text = "═══════════════════════════════════════════\n"
        summary_text += "EXECUTIVE SUMMARY TO REVIEW:\n"
        summary_text += "═══════════════════════════════════════════\n\n"
        summary_text += json.dumps(phase1_json, indent=2)

        # Format article summaries
        articles_text = "\n\n═══════════════════════════════════════════\n"
        articles_text += f"ARTICLE SUMMARIES ({len(article_summaries)} articles):\n"
        articles_text += "═══════════════════════════════════════════\n\n"

        for i, summary in enumerate(article_summaries, 1):
            articles_text += f"Article {i}:\n{summary}\n\n"

        # Combine into full prompt
        full_prompt = QUALITY_REVIEW_PROMPT + "\n" + summary_text + articles_text

        # Log token estimate
        char_count = len(full_prompt)
        token_estimate = char_count // 4
        LOG.info(f"[{ticker}] Quality review prompt: {char_count:,} chars (~{token_estimate:,} tokens)")

        # Call Gemini 2.5 Flash
        LOG.info(f"[{ticker}] Calling Gemini 2.5 Flash for quality review")
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
        total_sentences = 0
        supported = 0
        inference = 0
        unsupported = 0
        errors_by_severity = {"CRITICAL": 0, "SERIOUS": 0, "MINOR": 0}

        for section in result_json.get("sections", []):
            for sentence in section.get("sentences", []):
                total_sentences += 1
                status = sentence.get("status")
                severity = sentence.get("severity")

                if status == "SUPPORTED":
                    supported += 1
                elif status == "INFERENCE":
                    inference += 1
                elif status == "UNSUPPORTED":
                    unsupported += 1

                if severity:
                    errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1

        # Build final result
        final_result = {
            "summary": {
                "ticker": ticker,
                "total_sentences": total_sentences,
                "supported": supported,
                "inference": inference,
                "unsupported": unsupported,
                "errors_by_severity": errors_by_severity,
                "generation_time_ms": generation_time
            },
            "sections": result_json.get("sections", [])
        }

        LOG.info(f"✅ [{ticker}] Quality review complete: {supported}/{total_sentences} supported, "
                f"{unsupported} unsupported, {errors_by_severity['CRITICAL']} critical errors")

        return final_result

    except json.JSONDecodeError as e:
        LOG.error(f"[{ticker}] Failed to parse Gemini response as JSON: {e}")
        return None
    except Exception as e:
        LOG.error(f"[{ticker}] Quality review failed: {e}")
        import traceback
        LOG.error(traceback.format_exc())
        return None


def generate_quality_review_email_html(review_result: Dict) -> str:
    """
    Generate HTML email report from quality review results.

    Args:
        review_result: Output from review_executive_summary_quality()

    Returns:
        HTML string for email
    """
    summary = review_result["summary"]
    sections = review_result["sections"]

    ticker = summary["ticker"]
    total = summary["total_sentences"]
    supported = summary["supported"]
    inference = summary["inference"]
    unsupported = summary["unsupported"]
    errors = summary["errors_by_severity"]

    # Calculate percentages
    supported_pct = (supported / total * 100) if total > 0 else 0
    inference_pct = (inference / total * 100) if total > 0 else 0
    unsupported_pct = (unsupported / total * 100) if total > 0 else 0

    # Determine verdict
    critical_count = errors.get("CRITICAL", 0)
    serious_count = errors.get("SERIOUS", 0)
    minor_count = errors.get("MINOR", 0)

    # Check against thresholds
    critical_pass = critical_count == 0
    serious_pass = (serious_count / total * 100) < 1.0 if total > 0 else True
    minor_pass = (minor_count / total * 100) < 5.0 if total > 0 else True

    verdict = "✅ PASS" if (critical_pass and serious_pass and minor_pass) else "❌ FAIL"
    verdict_color = "#10b981" if verdict == "✅ PASS" else "#ef4444"

    # Build HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background-color: #f3f4f6; margin: 0; padding: 20px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); color: white; padding: 30px; border-radius: 8px 8px 0 0; }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 28px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .summary-box {{ background: #f8f9fa; border-left: 4px solid {verdict_color}; padding: 20px; margin: 20px; border-radius: 4px; }}
        .summary-box h2 {{ margin: 0 0 15px 0; font-size: 20px; color: {verdict_color}; }}
        .stats {{ display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px; }}
        .stat {{ flex: 1; min-width: 150px; }}
        .stat-label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; margin-top: 5px; }}
        .section {{ margin: 20px; padding: 20px; border: 1px solid #e5e7eb; border-radius: 6px; }}
        .section-header {{ font-size: 18px; font-weight: bold; color: #1e40af; margin-bottom: 15px; border-bottom: 2px solid #1e40af; padding-bottom: 8px; }}
        .sentence {{ margin-bottom: 20px; padding: 15px; border-radius: 6px; background: #f9fafb; }}
        .sentence.supported {{ border-left: 4px solid #10b981; }}
        .sentence.inference {{ border-left: 4px solid #f59e0b; }}
        .sentence.unsupported {{ border-left: 4px solid #ef4444; }}
        .sentence-text {{ font-size: 15px; margin-bottom: 10px; line-height: 1.6; }}
        .sentence-meta {{ font-size: 13px; color: #6b7280; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; margin-right: 8px; }}
        .badge.supported {{ background: #d1fae5; color: #065f46; }}
        .badge.inference {{ background: #fed7aa; color: #92400e; }}
        .badge.unsupported {{ background: #fee2e2; color: #991b1b; }}
        .badge.critical {{ background: #dc2626; color: white; }}
        .badge.serious {{ background: #f97316; color: white; }}
        .badge.minor {{ background: #fbbf24; color: #78350f; }}
        .evidence {{ margin-top: 8px; padding: 10px; background: white; border-radius: 4px; font-size: 13px; }}
        .evidence-item {{ margin: 5px 0; color: #374151; }}
        .footer {{ padding: 20px; text-align: center; color: #6b7280; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Quality Review: {ticker}</h1>
            <p>Executive Summary Verification Report</p>
        </div>

        <div class="summary-box">
            <h2>VERDICT: {verdict}</h2>
            <div class="stats">
                <div class="stat">
                    <div class="stat-label">Total Sentences</div>
                    <div class="stat-value">{total}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">✅ Supported</div>
                    <div class="stat-value" style="color: #10b981;">{supported} ({supported_pct:.1f}%)</div>
                </div>
                <div class="stat">
                    <div class="stat-label">⚠️ Inference</div>
                    <div class="stat-value" style="color: #f59e0b;">{inference} ({inference_pct:.1f}%)</div>
                </div>
                <div class="stat">
                    <div class="stat-label">🔴 Unsupported</div>
                    <div class="stat-value" style="color: #ef4444;">{unsupported} ({unsupported_pct:.1f}%)</div>
                </div>
            </div>

            <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
                <div class="stat-label">Error Summary</div>
                <div style="margin-top: 10px;">
                    <span class="badge critical">🔴 CRITICAL: {critical_count} (Target: 0%) {"❌ EXCEEDS" if not critical_pass else "✅"}</span>
                    <span class="badge serious">🟠 SERIOUS: {serious_count} (Target: &lt;1%) {"❌ EXCEEDS" if not serious_pass else "✅"}</span>
                    <span class="badge minor">🟡 MINOR: {minor_count} (Target: &lt;5%) {"❌ EXCEEDS" if not minor_pass else "✅"}</span>
                </div>
            </div>
        </div>
'''

    # Section names mapping
    section_names = {
        "bottom_line": "📌 BOTTOM LINE",
        "major_developments": "🔴 MAJOR DEVELOPMENTS",
        "financial_performance": "📊 FINANCIAL/OPERATIONAL PERFORMANCE",
        "risk_factors": "⚠️ RISK FACTORS",
        "wall_street_sentiment": "📈 WALL STREET SENTIMENT",
        "competitive_industry_dynamics": "⚡ COMPETITIVE/INDUSTRY DYNAMICS",
        "upcoming_catalysts": "📅 UPCOMING CATALYSTS",
        "upside_scenario": "📈 UPSIDE SCENARIO",
        "downside_scenario": "📉 DOWNSIDE SCENARIO",
        "key_variables": "🔍 KEY VARIABLES TO MONITOR"
    }

    # Render each section
    for section in sections:
        section_name = section.get("section_name", "unknown")
        display_name = section_names.get(section_name, section_name.upper())
        sentences = section.get("sentences", [])

        if not sentences:
            continue

        html += f'<div class="section">'
        html += f'<div class="section-header">{display_name} ({len(sentences)} sentences)</div>'

        for sentence in sentences:
            text = sentence.get("text", "")
            status = sentence.get("status", "").lower()
            error_type = sentence.get("error_type")
            severity = sentence.get("severity", "").upper()
            evidence = sentence.get("evidence", [])
            notes = sentence.get("notes", "")

            # Status badge
            if status == "supported":
                status_badge = '<span class="badge supported">✅ SUPPORTED</span>'
            elif status == "inference":
                status_badge = '<span class="badge inference">⚠️ INFERENCE</span>'
            else:
                status_badge = '<span class="badge unsupported">🔴 UNSUPPORTED</span>'

            # Severity badge
            severity_badge = ""
            if severity == "CRITICAL":
                severity_badge = '<span class="badge critical">🔴 CRITICAL</span>'
            elif severity == "SERIOUS":
                severity_badge = '<span class="badge serious">🟠 SERIOUS</span>'
            elif severity == "MINOR":
                severity_badge = '<span class="badge minor">🟡 MINOR</span>'

            # Error type
            error_html = ""
            if error_type:
                error_html = f'<div style="margin-top: 8px; color: #dc2626; font-weight: 600;">Error: {error_type}</div>'

            # Evidence
            evidence_html = ""
            if evidence:
                evidence_html = '<div class="evidence">'
                evidence_html += '<strong>Evidence:</strong>'
                for item in evidence:
                    evidence_html += f'<div class="evidence-item">• {item}</div>'
                evidence_html += '</div>'

            # Notes
            notes_html = ""
            if notes:
                notes_html = f'<div style="margin-top: 8px; color: #6b7280; font-style: italic;">Note: {notes}</div>'

            html += f'''
            <div class="sentence {status}">
                <div class="sentence-meta">
                    {status_badge}
                    {severity_badge}
                </div>
                <div class="sentence-text">{text}</div>
                {error_html}
                {evidence_html}
                {notes_html}
            </div>
            '''

        html += '</div>'

    html += '''
        <div class="footer">
            <p>Generated by StockDigest Quality Review System</p>
            <p>Powered by Gemini 2.5 Flash</p>
        </div>
    </div>
</body>
</html>
'''

    return html
