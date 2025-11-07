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

3. Attribution Errors
   - WRONG: Source attribution is incorrect or contradicts sources
     • Example: "per CFO" but articles only cite analyst reports
     • Example: "per Goldman" but Goldman never mentioned
   - VAGUE: Attribution too generic to verify
     • Example: "per analyst" (which analyst? which firm?)
     • Example: "per reports" (which publication?)
     • Example: "per management" (which executive?)
   - SPLIT: Multiple claims from different sources with single attribution
     • Example: "Costs stabilized with drivers identified per Source A"
       (stabilization from Source B, drivers from Source A - only one shown)
   - Check: Attribution must be specific and match sources

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

🔴 CRITICAL ERRORS (Target: 0%):

7. Confidence Upgrade
   - AI upgraded article's uncertainty to more confident language
   - Article hedge words that must be preserved:
     • WEAK: "may", "could", "possible", "exploring", "considering"
     • SPECULATIVE: "sources say", "rumored", "reportedly"
   - Forbidden upgrades:
     • "may" → "will" / "plans to" / "expects"
     • "exploring" → "pursuing" / "negotiating" / "evaluating"
     • "sources say considering" → "announced" / "confirmed"
     • "analyst sees possible" → "analyst projects" / "forecasts"
     • "could potentially" → "will" / "expects to"
   - Examples:
     • Article: "Company may launch product" → Summary: "Company will launch" ❌
     • Article: "Exploring strategic alternatives" → Summary: "Pursuing acquisition" ❌
     • Article: "Analyst sees possible upside" → Summary: "Analyst projects upside" ❌
   - Check: Preserve exact confidence level from articles - no upgrades

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
INPUT STRUCTURE
═══════════════════════════════════════════

You will receive a merged Phase 1 + Phase 2 executive summary JSON.

BULLET SECTIONS (major_developments, financial_performance, risk_factors, wall_street_sentiment, competitive_industry_dynamics, upcoming_catalysts, key_variables):
- Each section contains an array of bullet objects
- Each bullet has: bullet_id, topic_label, content, context (if enriched), impact, sentiment, etc.
- Extract the bullet_id from each bullet object
- Verify the content field (the main bullet text) against article summaries

PARAGRAPH SECTIONS (bottom_line, upside_scenario, downside_scenario):
- Each section is an object with: content, context (if enriched)
- Use section_name as the identifier (e.g., "bottom_line")
- Verify the content field (the full paragraph) as ONE UNIT against article summaries

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
          "bullet_id": "bottom_line",
          "text": "Full paragraph text here",
          "status": "SUPPORTED" | "INFERENCE" | "UNSUPPORTED",
          "error_type": "Fabricated Number" | "Fabricated Claim" | "Attribution Errors" | "Directional Error" | "Company Confusion" | "Inference as Fact" | "Confidence Upgrade" | null,
          "severity": "CRITICAL" | "SERIOUS" | "MINOR" | null,
          "evidence": ["Article X mentions...", "Article Y states..."],
          "notes": "Explanation of verification result"
        }
      ]
    },
    {
      "section_name": "major_developments",
      "sentences": [
        {
          "bullet_id": "market_inflection_point",
          "topic_label": "Market inflection point",
          "impact": "high impact",
          "sentiment": "bullish",
          "reason": "demand inflection",
          "text": "Full bullet text here",
          "status": "SUPPORTED" | "INFERENCE" | "UNSUPPORTED",
          "error_type": null,
          "severity": null,
          "evidence": ["Article X mentions..."],
          "notes": "All facts verified"
        }
      ]
    }
  ]
}

CRITICAL RULES FOR OUTPUT FIELDS:

For BULLET sections, extract these fields from input JSON:
- bullet_id: From bullet object (e.g., "market_inflection_point")
- topic_label: From bullet object (e.g., "Market inflection point")
- impact: From bullet object (e.g., "high impact") - added by Phase 2 enrichment
- sentiment: From bullet object (e.g., "bullish") - added by Phase 2 enrichment
- reason: From bullet object (e.g., "demand inflection") - added by Phase 2 enrichment
- text: The full "content" field from bullet object
- Then verify the text against article summaries and add: status, error_type, severity, evidence, notes

For PARAGRAPH sections:
- bullet_id: Use section_name (e.g., "bottom_line")
- topic_label: NOT APPLICABLE (paragraphs don't have topic_label, leave empty)
- impact, sentiment, reason: NOT APPLICABLE (paragraphs don't have these, leave empty)
- text: The full "content" field from section object
- Then verify the text against article summaries and add: status, error_type, severity, evidence, notes

IMPORTANT:
- Each "sentences" array entry represents ONE bullet or ONE paragraph (not sentence-by-sentence breakdown)
- Extract fields directly from input JSON structure - don't generate new values

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
            status = (sentence.get("status") or "").lower()
            error_type = sentence.get("error_type")
            severity = (sentence.get("severity") or "").upper()
            evidence = sentence.get("evidence", [])
            notes = sentence.get("notes", "") or ""

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


def generate_combined_quality_review_email_html(
    phase1_result: Dict,
    phase2_result: Optional[Dict]
) -> str:
    """
    Generate combined HTML email report from Phase 1 + Phase 2 quality review results.

    Shows bullet-by-bullet verification: Phase 1 article check, then Phase 2 context check.

    Args:
        phase1_result: Output from review_executive_summary_quality()
        phase2_result: Output from review_phase2_context_quality() or None if skipped

    Returns:
        HTML string for combined email
    """
    p1_summary = phase1_result["summary"]
    p1_sections = phase1_result["sections"]

    ticker = p1_summary["ticker"]

    # Phase 1 stats
    p1_total = p1_summary["total_sentences"]
    p1_supported = p1_summary["supported"]
    p1_inference = p1_summary["inference"]
    p1_unsupported = p1_summary["unsupported"]
    p1_errors = p1_summary["errors_by_severity"]

    # Calculate Phase 1 percentages
    p1_supported_pct = (p1_supported / p1_total * 100) if p1_total > 0 else 0
    p1_inference_pct = (p1_inference / p1_total * 100) if p1_total > 0 else 0
    p1_unsupported_pct = (p1_unsupported / p1_total * 100) if p1_total > 0 else 0

    # Phase 1 pass/fail
    p1_critical = p1_errors.get("CRITICAL", 0)
    p1_serious = p1_errors.get("SERIOUS", 0)
    p1_minor = p1_errors.get("MINOR", 0)
    p1_critical_pass = p1_critical == 0
    p1_serious_pass = (p1_serious / p1_total * 100) < 1.0 if p1_total > 0 else True
    p1_minor_pass = (p1_minor / p1_total * 100) < 5.0 if p1_total > 0 else True
    phase1_pass = p1_critical_pass and p1_serious_pass and p1_minor_pass

    # Phase 2 stats (if available)
    phase2_skipped = phase2_result is None
    if not phase2_skipped:
        p2_summary = phase2_result["summary"]
        p2_contexts_list = phase2_result["contexts"]

        p2_total = p2_summary["total_contexts"]
        p2_accurate = p2_summary["accurate"]
        p2_issues = p2_summary["issues"]
        p2_errors = p2_summary["errors_by_severity"]

        # Calculate Phase 2 percentages
        p2_accurate_pct = (p2_accurate / p2_total * 100) if p2_total > 0 else 0
        p2_issues_pct = (p2_issues / p2_total * 100) if p2_total > 0 else 0

        # Phase 2 pass/fail
        p2_critical = p2_errors.get("CRITICAL", 0)
        p2_serious = p2_errors.get("SERIOUS", 0)
        p2_minor = p2_errors.get("MINOR", 0)
        p2_critical_pass = p2_critical == 0
        p2_serious_pass = (p2_serious / p2_total * 100) < 1.0 if p2_total > 0 else True
        p2_minor_pass = (p2_minor / p2_total * 100) < 5.0 if p2_total > 0 else True
        phase2_pass = p2_critical_pass and p2_serious_pass and p2_minor_pass

        # Create lookup for contexts by section + bullet_id
        p2_contexts_by_key = {}
        for ctx in p2_contexts_list:
            section = ctx.get("section_name", "")
            bullet_id = ctx.get("bullet_id", "")
            key = f"{section}|{bullet_id}"
            p2_contexts_by_key[key] = ctx
    else:
        phase2_pass = True  # No Phase 2 = auto-pass
        p2_contexts_by_key = {}

    # Overall verdict
    overall_pass = phase1_pass and phase2_pass
    overall_verdict = "✅ PASS" if overall_pass else "❌ FAIL"
    verdict_color = "#10b981" if overall_pass else "#ef4444"

    # Determine verdict reason
    if overall_pass:
        verdict_reason = "All quality checks passed"
    elif not phase1_pass and not phase2_pass:
        verdict_reason = "Both Phase 1 and Phase 2 detected critical errors"
    elif not phase1_pass:
        verdict_reason = f"Phase 1 failed ({p1_critical} critical, {p1_serious} serious, {p1_minor} minor errors)"
    else:
        verdict_reason = f"Phase 2 failed ({p2_critical} critical, {p2_serious} serious, {p2_minor} minor errors)"

    # Build HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background-color: #f3f4f6; margin: 0; padding: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); color: white; padding: 30px; border-radius: 8px 8px 0 0; }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 28px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .overall-verdict {{ background: #f8f9fa; border-left: 4px solid {verdict_color}; padding: 20px; margin: 20px; border-radius: 4px; }}
        .overall-verdict h2 {{ margin: 0 0 10px 0; font-size: 22px; color: {verdict_color}; }}
        .overall-verdict p {{ margin: 5px 0; color: #374151; font-size: 14px; }}
        .phase-summary {{ margin: 20px; padding: 20px; border: 1px solid #e5e7eb; border-radius: 6px; background: #fafbfc; }}
        .phase-summary h3 {{ margin: 0 0 15px 0; font-size: 18px; color: #1e40af; }}
        .phase-summary.skipped {{ background: #f9fafb; border-color: #d1d5db; }}
        .phase-summary.skipped h3 {{ color: #6b7280; }}
        .stats {{ display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px; }}
        .stat {{ flex: 1; min-width: 140px; }}
        .stat-label {{ font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }}
        .stat-value {{ font-size: 20px; font-weight: bold; margin-top: 5px; }}
        .error-summary {{ margin-top: 15px; padding-top: 15px; border-top: 1px solid #e5e7eb; }}
        .section {{ margin: 20px; padding: 20px; border: 1px solid #e5e7eb; border-radius: 6px; }}
        .section-header {{ font-size: 18px; font-weight: bold; color: #1e40af; margin-bottom: 15px; border-bottom: 2px solid #1e40af; padding-bottom: 8px; }}
        .bullet-group {{ margin-bottom: 30px; padding: 15px; background: #f9fafb; border-radius: 6px; border: 1px solid #e5e7eb; }}
        .bullet-title {{ font-size: 16px; font-weight: bold; color: #374151; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #d1d5db; }}
        .phase-label {{ font-size: 14px; font-weight: 600; color: #1e40af; margin: 15px 0 10px 0; text-transform: uppercase; letter-spacing: 0.5px; }}
        .review-item {{ margin-bottom: 15px; padding: 15px; border-radius: 6px; background: white; }}
        .review-item.supported {{ border-left: 4px solid #10b981; }}
        .review-item.inference {{ border-left: 4px solid #f59e0b; }}
        .review-item.unsupported {{ border-left: 4px solid #ef4444; }}
        .review-item.accurate {{ border-left: 4px solid #10b981; }}
        .review-item.issue {{ border-left: 4px solid #ef4444; }}
        .review-text {{ font-size: 14px; margin-bottom: 10px; line-height: 1.6; color: #1f2937; }}
        .review-meta {{ font-size: 12px; color: #6b7280; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; text-transform: uppercase; margin-right: 6px; }}
        .badge.supported {{ background: #d1fae5; color: #065f46; }}
        .badge.inference {{ background: #fed7aa; color: #92400e; }}
        .badge.unsupported {{ background: #fee2e2; color: #991b1b; }}
        .badge.accurate {{ background: #d1fae5; color: #065f46; }}
        .badge.issue {{ background: #fee2e2; color: #991b1b; }}
        .badge.critical {{ background: #dc2626; color: white; }}
        .badge.serious {{ background: #f97316; color: white; }}
        .badge.minor {{ background: #fbbf24; color: #78350f; }}
        .badge.pass {{ background: #10b981; color: white; }}
        .badge.fail {{ background: #ef4444; color: white; }}
        .evidence {{ margin-top: 8px; padding: 10px; background: #f3f4f6; border-radius: 4px; font-size: 12px; }}
        .evidence-item {{ margin: 5px 0; color: #374151; }}
        .footer {{ padding: 20px; text-align: center; color: #6b7280; font-size: 12px; }}
        .divider {{ height: 1px; background: #e5e7eb; margin: 15px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Quality Review: {ticker}</h1>
            <p>Comprehensive Article & Filing Verification Report</p>
        </div>

        <div class="overall-verdict">
            <h2>OVERALL VERDICT: {overall_verdict}</h2>
            <p>{verdict_reason}</p>
        </div>

        <div class="phase-summary">
            <h3>📰 PHASE 1: ARTICLE VERIFICATION <span class="badge {'pass' if phase1_pass else 'fail'}">{('✅ PASS' if phase1_pass else '❌ FAIL')}</span></h3>
            <div class="stats">
                <div class="stat">
                    <div class="stat-label">Total Sentences</div>
                    <div class="stat-value">{p1_total}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">✅ Supported</div>
                    <div class="stat-value" style="color: #10b981;">{p1_supported} ({p1_supported_pct:.1f}%)</div>
                </div>
                <div class="stat">
                    <div class="stat-label">⚠️ Inference</div>
                    <div class="stat-value" style="color: #f59e0b;">{p1_inference} ({p1_inference_pct:.1f}%)</div>
                </div>
                <div class="stat">
                    <div class="stat-label">🔴 Unsupported</div>
                    <div class="stat-value" style="color: #ef4444;">{p1_unsupported} ({p1_unsupported_pct:.1f}%)</div>
                </div>
            </div>
            <div class="error-summary">
                <div class="stat-label">Error Breakdown</div>
                <div style="margin-top: 10px;">
                    <span class="badge critical">🔴 CRITICAL: {p1_critical} (Target: 0%) {'❌ EXCEEDS' if not p1_critical_pass else '✅'}</span>
                    <span class="badge serious">🟠 SERIOUS: {p1_serious} (Target: &lt;1%) {'❌ EXCEEDS' if not p1_serious_pass else '✅'}</span>
                    <span class="badge minor">🟡 MINOR: {p1_minor} (Target: &lt;5%) {'❌ EXCEEDS' if not p1_minor_pass else '✅'}</span>
                </div>
            </div>
        </div>
'''

    # Phase 2 summary box
    if phase2_skipped:
        html += '''
        <div class="phase-summary skipped">
            <h3>📄 PHASE 2: FILING CONTEXT VERIFICATION <span class="badge" style="background: #6b7280; color: white;">SKIPPED</span></h3>
            <p style="color: #6b7280; margin: 10px 0 0 0; font-size: 14px;">No 10-K, 10-Q, or Transcript filings available for verification.</p>
        </div>
'''
    else:
        html += f'''
        <div class="phase-summary">
            <h3>📄 PHASE 2: FILING CONTEXT VERIFICATION <span class="badge {'pass' if phase2_pass else 'fail'}">{('✅ PASS' if phase2_pass else '❌ FAIL')}</span></h3>
            <div class="stats">
                <div class="stat">
                    <div class="stat-label">Total Contexts</div>
                    <div class="stat-value">{p2_total}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">✅ Accurate</div>
                    <div class="stat-value" style="color: #10b981;">{p2_accurate} ({p2_accurate_pct:.1f}%)</div>
                </div>
                <div class="stat">
                    <div class="stat-label">🔴 Issues</div>
                    <div class="stat-value" style="color: #ef4444;">{p2_issues} ({p2_issues_pct:.1f}%)</div>
                </div>
            </div>
            <div class="error-summary">
                <div class="stat-label">Error Breakdown</div>
                <div style="margin-top: 10px;">
                    <span class="badge critical">🔴 CRITICAL: {p2_critical} (Target: 0%) {'❌ EXCEEDS' if not p2_critical_pass else '✅'}</span>
                    <span class="badge serious">🟠 SERIOUS: {p2_serious} (Target: &lt;1%) {'❌ EXCEEDS' if not p2_serious_pass else '✅'}</span>
                    <span class="badge minor">🟡 MINOR: {p2_minor} (Target: &lt;5%) {'❌ EXCEEDS' if not p2_minor_pass else '✅'}</span>
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

    # Now render each section bullet-by-bullet with Phase 1 then Phase 2
    for section in p1_sections:
        section_name = section.get("section_name", "unknown")
        display_name = section_names.get(section_name, section_name.upper())
        sentences = section.get("sentences", [])

        if not sentences:
            continue

        html += f'<div class="section">'
        html += f'<div class="section-header">{display_name}</div>'

        # Check if this is a bullet section or paragraph section
        # Bottom Line, Upside, Downside are paragraphs (sentences)
        # Others are bullets
        is_paragraph = section_name in ["bottom_line", "upside_scenario", "downside_scenario"]

        if is_paragraph:
            # Paragraph sections: show all Phase 1 sentences, then Phase 2 context
            html += '<div class="bullet-group">'
            html += '<div class="phase-label">📰 Article Verification (Phase 1)</div>'

            # Render all sentences
            for sentence in sentences:
                text = sentence.get("text", "")
                status = (sentence.get("status") or "").lower()
                error_type = sentence.get("error_type")
                severity = (sentence.get("severity") or "").upper()
                evidence = sentence.get("evidence", [])
                notes = sentence.get("notes", "") or ""
                bullet_id = sentence.get("bullet_id")  # Extract bullet_id (will be section_name for paragraphs)

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
                    evidence_html = '<div class="evidence"><strong>Evidence:</strong>'
                    for item in evidence:
                        evidence_html += f'<div class="evidence-item">• {item}</div>'
                    evidence_html += '</div>'

                # Notes
                notes_html = ""
                if notes:
                    notes_html = f'<div style="margin-top: 8px; color: #6b7280; font-style: italic;">Note: {notes}</div>'

                html += f'''
                <div class="review-item {status}">
                    <div class="review-meta">{status_badge}{severity_badge}</div>
                    <div class="review-text">{text}</div>
                    {error_html}
                    {evidence_html}
                    {notes_html}
                </div>
                '''

            # Now check for Phase 2 context for this paragraph
            # Use bullet_id from Phase 1 output (which should be section_name for paragraphs)
            if not phase2_skipped and bullet_id:
                context_key = f"{section_name}|{bullet_id}"
                p2_ctx = p2_contexts_by_key.get(context_key)

                if p2_ctx:
                    html += '<div class="divider"></div>'
                    html += '<div class="phase-label">📄 Filing Context Verification (Phase 2)</div>'

                    ctx_text = p2_ctx.get("context_text", "")
                    ctx_status = (p2_ctx.get("status") or "").lower()
                    ctx_error_type = p2_ctx.get("error_type")
                    ctx_severity = (p2_ctx.get("severity") or "").upper()
                    ctx_evidence = p2_ctx.get("evidence", [])
                    ctx_notes = p2_ctx.get("notes", "") or ""

                    # Status badge
                    if ctx_status == "accurate":
                        ctx_status_badge = '<span class="badge accurate">✅ ACCURATE</span>'
                    else:
                        ctx_status_badge = '<span class="badge issue">🔴 ISSUE</span>'

                    # Severity badge
                    ctx_severity_badge = ""
                    if ctx_severity == "CRITICAL":
                        ctx_severity_badge = '<span class="badge critical">🔴 CRITICAL</span>'
                    elif ctx_severity == "SERIOUS":
                        ctx_severity_badge = '<span class="badge serious">🟠 SERIOUS</span>'
                    elif ctx_severity == "MINOR":
                        ctx_severity_badge = '<span class="badge minor">🟡 MINOR</span>'

                    # Error type
                    ctx_error_html = ""
                    if ctx_error_type:
                        ctx_error_html = f'<div style="margin-top: 8px; color: #dc2626; font-weight: 600;">Error: {ctx_error_type}</div>'

                    # Evidence
                    ctx_evidence_html = ""
                    if ctx_evidence:
                        ctx_evidence_html = '<div class="evidence"><strong>Filing Evidence:</strong>'
                        for item in ctx_evidence:
                            ctx_evidence_html += f'<div class="evidence-item">• {item}</div>'
                        ctx_evidence_html += '</div>'

                    # Notes
                    ctx_notes_html = ""
                    if ctx_notes:
                        ctx_notes_html = f'<div style="margin-top: 8px; color: #6b7280; font-style: italic;">Note: {ctx_notes}</div>'

                    html += f'''
                    <div class="review-item {ctx_status}">
                        <div class="review-meta">{ctx_status_badge}{ctx_severity_badge}</div>
                        <div class="review-text" style="font-size: 13px; font-style: italic;">{ctx_text}</div>
                        {ctx_error_html}
                        {ctx_evidence_html}
                        {ctx_notes_html}
                    </div>
                    '''

            html += '</div>'  # End bullet-group

        else:
            # Bullet sections: group by bullet, show Phase 1 then Phase 2 for each
            # Group sentences by bullet (they should already be grouped in Phase 1 output)
            # For bullet sections, each "sentence" is actually a bullet
            for sentence in sentences:
                html += '<div class="bullet-group">'

                # Get bullet info
                text = sentence.get("text", "")
                status = (sentence.get("status") or "").lower()
                error_type = sentence.get("error_type")
                severity = (sentence.get("severity") or "").upper()
                evidence = sentence.get("evidence", [])
                notes = sentence.get("notes", "") or ""
                bullet_id = sentence.get("bullet_id")  # Extract bullet_id from Phase 1 output

                # Extract enrichment fields for display
                topic_label = sentence.get("topic_label", "")
                impact = sentence.get("impact", "")
                sentiment = sentence.get("sentiment", "")
                reason = sentence.get("reason", "")

                # Build bullet title: topic_label (impact, sentiment, reason)
                if topic_label:
                    enrichment_tags = []
                    if impact:
                        enrichment_tags.append(impact)
                    if sentiment:
                        enrichment_tags.append(sentiment)
                    if reason:
                        enrichment_tags.append(reason)

                    if enrichment_tags:
                        bullet_title = f"{topic_label} ({', '.join(enrichment_tags)})"
                    else:
                        bullet_title = topic_label
                else:
                    # Fallback if topic_label not provided (shouldn't happen with updated prompt)
                    bullet_title = text[:80] + "..." if len(text) > 80 else text

                html += f'<div class="bullet-title">{bullet_title}</div>'

                # Phase 1: Article verification
                html += '<div class="phase-label">📰 Article Verification (Phase 1)</div>'

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
                    evidence_html = '<div class="evidence"><strong>Evidence:</strong>'
                    for item in evidence:
                        evidence_html += f'<div class="evidence-item">• {item}</div>'
                    evidence_html += '</div>'

                # Notes
                notes_html = ""
                if notes:
                    notes_html = f'<div style="margin-top: 8px; color: #6b7280; font-style: italic;">Note: {notes}</div>'

                html += f'''
                <div class="review-item {status}">
                    <div class="review-meta">{status_badge}{severity_badge}</div>
                    <div class="review-text">{text}</div>
                    {error_html}
                    {evidence_html}
                    {notes_html}
                </div>
                '''

                # Phase 2: Context verification (if available)
                if not phase2_skipped and bullet_id:
                    # Find Phase 2 context for THIS SPECIFIC BULLET using bullet_id
                    section_contexts = [ctx for ctx in p2_contexts_by_key.values()
                                       if ctx.get("bullet_id") == bullet_id]

                    if section_contexts:
                        html += '<div class="divider"></div>'
                        html += '<div class="phase-label">📄 Filing Context Verification (Phase 2)</div>'

                        for p2_ctx in section_contexts:
                            ctx_text = p2_ctx.get("context_text", "")
                            ctx_status = (p2_ctx.get("status") or "").lower()
                            ctx_error_type = p2_ctx.get("error_type")
                            ctx_severity = (p2_ctx.get("severity") or "").upper()
                            ctx_evidence = p2_ctx.get("evidence", [])
                            ctx_notes = p2_ctx.get("notes", "") or ""

                            # Status badge
                            if ctx_status == "accurate":
                                ctx_status_badge = '<span class="badge accurate">✅ ACCURATE</span>'
                            else:
                                ctx_status_badge = '<span class="badge issue">🔴 ISSUE</span>'

                            # Severity badge
                            ctx_severity_badge = ""
                            if ctx_severity == "CRITICAL":
                                ctx_severity_badge = '<span class="badge critical">🔴 CRITICAL</span>'
                            elif ctx_severity == "SERIOUS":
                                ctx_severity_badge = '<span class="badge serious">🟠 SERIOUS</span>'
                            elif ctx_severity == "MINOR":
                                ctx_severity_badge = '<span class="badge minor">🟡 MINOR</span>'

                            # Error type
                            ctx_error_html = ""
                            if ctx_error_type:
                                ctx_error_html = f'<div style="margin-top: 8px; color: #dc2626; font-weight: 600;">Error: {ctx_error_type}</div>'

                            # Evidence
                            ctx_evidence_html = ""
                            if ctx_evidence:
                                ctx_evidence_html = '<div class="evidence"><strong>Filing Evidence:</strong>'
                                for item in ctx_evidence:
                                    ctx_evidence_html += f'<div class="evidence-item">• {item}</div>'
                                ctx_evidence_html += '</div>'

                            # Notes
                            ctx_notes_html = ""
                            if ctx_notes:
                                ctx_notes_html = f'<div style="margin-top: 8px; color: #6b7280; font-style: italic;">Note: {ctx_notes}</div>'

                            html += f'''
                            <div class="review-item {ctx_status}">
                                <div class="review-meta">{ctx_status_badge}{ctx_severity_badge}</div>
                                <div class="review-text" style="font-size: 13px; font-style: italic;">{ctx_text}</div>
                                {ctx_error_html}
                                {ctx_evidence_html}
                                {ctx_notes_html}
                            </div>
                            '''

                html += '</div>'  # End bullet-group

        html += '</div>'  # End section

    html += '''
        <div class="footer">
            <p>Generated by StockDigest Quality Review System</p>
            <p>Phase 1 & Phase 2 powered by Gemini 2.5 Flash</p>
        </div>
    </div>
</body>
</html>
'''

    return html
