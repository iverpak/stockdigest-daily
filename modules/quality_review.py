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
     • **CRITICAL: Validate against SOURCE DOMAIN tags ONLY**
       - When checking "per Yahoo Finance", look for SOURCE DOMAIN: [Yahoo Finance]
       - IGNORE parenthetical sources in summary text like "(investorplace.com)"
       - Example: Article has [Yahoo Finance] domain but text mentions "(investorplace.com)"
         → "per Yahoo Finance" is ✅ CORRECT (matches SOURCE DOMAIN)
         → "per InvestorPlace" is 🔴 WRONG (nested citation, not actual source)
   - VAGUE: Attribution too generic to verify
     • Example: "per analyst" (which analyst? which firm?)
     • Example: "per reports" (which publication?)
     • Example: "per management" (which executive?)
   - SPLIT: Multiple claims from different sources with single attribution
     • Example: "Costs stabilized with drivers identified per Source A"
       (stabilization from Source B, drivers from Source A - only one shown)
   - Check: Attribution must match SOURCE DOMAIN tags, ignore nested parenthetical citations

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

═══════════════════════════════════════════
THEME COVERAGE ANALYSIS
═══════════════════════════════════════════

After verifying sentence accuracy, perform theme coverage completeness check.

STEP 1: EXTRACT THEMES FROM ARTICLES

Review all article summaries and identify material themes relevant to investors.

For each theme, categorize as:
- financial: Earnings, revenue, guidance, financial metrics
- operational: Production, delivery, operations, efficiency
- strategic: M&A, partnerships, product launches, business model changes
- risk: Regulatory actions, lawsuits, competitive threats, operational failures
- market: Industry trends, competitive dynamics, market conditions
- sentiment: Analyst ratings, price targets, investor sentiment
- catalyst: Upcoming events with specific dates

STEP 2: EVALUATE THEME MATERIALITY

For each theme, evaluate using three tests:

1. INVESTOR MATERIALITY TEST
   Would an institutional investor in this company want to know about this theme?
   - HIGH: Regulatory actions, earnings surprises, strategic shifts, competitive threats
   - LOW: Stock price movements without fundamental driver, minor operational details

2. EARNINGS CALL TEST
   Would management likely discuss this theme on the next earnings call?
   - YES: Major contracts, guidance changes, strategic initiatives, material risks
   - NO: Routine operations, minor customer wins, industry noise

3. VALUATION IMPACT TEST
   Does this theme materially impact company valuation or operations?
   - YES: M&A, product launches, regulatory changes, market share shifts
   - NO: Minor partnerships, executive speaking engagements, small buybacks

MATERIALITY RATING:
- Material: YES to 2+ questions (should be in summary)
- Moderate: YES to 1 question (should be in summary if space allows)
- Non-material: NO to all questions (OK to exclude)

STEP 3: MAP THEMES TO EXECUTIVE SUMMARY SECTIONS

For each MATERIAL theme, check if it appears in the executive summary:

Search these sections:
- bottom_line, major_developments, financial_performance, risk_factors
- wall_street_sentiment, competitive_industry_dynamics, upcoming_catalysts
- upside_scenario, downside_scenario, key_variables

Mark theme as:
- COVERED: Theme appears in appropriate section
- MISSING: Material theme not found in any section

STEP 4: ASSIGN SEVERITY TO MISSING THEMES

For missing material themes:
- CRITICAL: YES to all 3 materiality questions (core investment thesis impact)
- SERIOUS: YES to 2 materiality questions (meaningful operational/financial impact)
- MINOR: YES to 1 materiality question (nice-to-know but not critical)

═══════════════════════════════════════════
OUTPUT FORMAT WITH THEME ANALYSIS
═══════════════════════════════════════════

Return JSON with BOTH sentence verification AND theme analysis:

{
  "sections": [...],  // Sentence verification as specified above

  "theme_analysis": {
    "themes": [
      {
        "theme": "FDA approval delay",
        "category": "risk",
        "mentioned_in_articles": [3, 7],
        "materiality_rating": "Material",
        "investor_test": "YES - regulatory setback affects product timeline",
        "earnings_call_test": "YES - material operational development requiring disclosure",
        "valuation_test": "YES - delays revenue recognition by 6 months",
        "covered_in_section": null,
        "status": "MISSING",
        "recommended_section": "risk_factors",
        "severity": "CRITICAL"
      },
      {
        "theme": "Q3 earnings beat",
        "category": "financial",
        "mentioned_in_articles": [1, 2, 4, 8],
        "materiality_rating": "Material",
        "investor_test": "YES - beats consensus, validates growth thesis",
        "earnings_call_test": "YES - primary topic of call",
        "valuation_test": "YES - affects forward estimates",
        "covered_in_section": "financial_performance",
        "status": "COVERED",
        "recommended_section": null,
        "severity": null
      },
      {
        "theme": "Stock price uptick",
        "category": "market",
        "mentioned_in_articles": [9],
        "materiality_rating": "Non-material",
        "investor_test": "NO - price movement without fundamental catalyst",
        "earnings_call_test": "NO - management does not discuss stock price",
        "valuation_test": "NO - no underlying business change",
        "covered_in_section": null,
        "status": "CORRECTLY_EXCLUDED",
        "recommended_section": null,
        "severity": null
      }
    ],

    "missing_themes": [
      {
        "theme": "FDA approval delay",
        "category": "risk",
        "mentioned_in_articles": [3, 7],
        "investor_test": "YES - regulatory setback affects timeline",
        "earnings_call_test": "YES - material operational development",
        "valuation_test": "YES - delays revenue by 6 months",
        "recommended_section": "risk_factors",
        "severity": "CRITICAL"
      }
    ]
  }
}

Return ONLY the JSON object, no other text.
"""


def review_executive_summary_quality(
    ticker: str,
    phase1_json: Dict,
    articles: List[Dict],
    gemini_api_key: str
) -> Optional[Dict]:
    """
    Review executive summary quality against articles using Gemini.

    Args:
        ticker: Stock ticker
        phase1_json: Phase 1 executive summary JSON
        articles: List of article objects with domain metadata
                 Each article: {"domain": str, "title": str, "ai_summary": str, "article_id": int}
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

        # Format articles with explicit domain labels for attribution validation
        articles_text = "\n\n═══════════════════════════════════════════\n"
        articles_text += f"ARTICLE SOURCES ({len(articles)} articles):\n"
        articles_text += "═══════════════════════════════════════════\n\n"
        articles_text += "CRITICAL: Use SOURCE DOMAIN tags below for attribution validation.\n"
        articles_text += "Ignore any parenthetical sources in summary text (e.g., \"(investorplace.com)\").\n\n"

        for i, article in enumerate(articles, 1):
            domain = article.get('domain', 'Unknown')
            title = article.get('title', 'No title')
            summary = article.get('ai_summary', '')

            articles_text += f"Article {i}:\n"
            articles_text += f"SOURCE DOMAIN: [{domain}]  ← USE THIS FOR ATTRIBUTION VALIDATION\n"
            articles_text += f"Title: {title}\n"
            articles_text += f"Summary: {summary}\n\n"

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

        # Calculate theme coverage statistics
        theme_analysis = result_json.get("theme_analysis", {})
        theme_coverage_stats = {}

        if theme_analysis:
            themes = theme_analysis.get("themes", [])
            material_themes = [t for t in themes if t.get("materiality_rating") == "Material"]
            covered_themes = [t for t in material_themes if t.get("status") == "COVERED"]
            missing_themes = [t for t in material_themes if t.get("status") == "MISSING"]

            coverage_pct = (len(covered_themes) / len(material_themes) * 100) if len(material_themes) > 0 else 100

            theme_coverage_stats = {
                "total_themes_identified": len(themes),
                "material_themes": len(material_themes),
                "themes_covered": len(covered_themes),
                "themes_missing": len(missing_themes),
                "coverage_percentage": round(coverage_pct, 1),
                "threshold": "95%+",
                "status": "PASS" if coverage_pct >= 95 else "ISSUE"
            }

        # Build final result
        final_result = {
            "summary": {
                "ticker": ticker,
                "total_sentences": total_sentences,
                "supported": supported,
                "inference": inference,
                "unsupported": unsupported,
                "errors_by_severity": errors_by_severity,
                "generation_time_ms": generation_time,
                "theme_coverage": theme_coverage_stats if theme_coverage_stats else None
            },
            "sections": result_json.get("sections", []),
            "theme_analysis": theme_analysis if theme_analysis else None
        }

        LOG.info(f"✅ [{ticker}] Quality review complete: {supported}/{total_sentences} supported, "
                f"{unsupported} unsupported, {errors_by_severity['CRITICAL']} critical errors, "
                f"theme coverage: {theme_coverage_stats.get('coverage_percentage', 0):.1f}%")

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


def generate_bullet_centric_review_email_html(
    phase1_result: Dict,
    phase2_result: Optional[Dict],
    phase3_result: Optional[Dict],
    phase4_result: Optional[Dict]
) -> str:
    """
    Generate bullet-centric HTML email report showing all 4 phases per bullet/section.

    Args:
        phase1_result: Output from review_executive_summary_quality() (includes theme analysis)
        phase2_result: Output from review_phase2_context_quality() or None
        phase3_result: Output from review_context_relevance() or None
        phase4_result: Output from review_phase4_metadata_and_structure() or None

    Returns:
        HTML string for comprehensive bullet-centric quality review email
    """
    p1_summary = phase1_result["summary"]
    p1_sections = phase1_result["sections"]
    p1_themes = phase1_result.get("theme_analysis", {})

    ticker = p1_summary["ticker"]

    # Phase 1 stats
    p1_total = p1_summary["total_sentences"]
    p1_supported = p1_summary["supported"]
    p1_inference = p1_summary["inference"]
    p1_unsupported = p1_summary["unsupported"]
    p1_errors = p1_summary["errors_by_severity"]
    p1_theme_coverage = p1_summary.get("theme_coverage")

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
        p2_completeness = p2_summary.get("completeness", {})

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
        p2_completeness = {}

    # Phase 3 stats (if available)
    phase3_skipped = phase3_result is None
    if not phase3_skipped:
        p3_summary = phase3_result["summary"]
        p3_evaluations_list = phase3_result["evaluations"]

        p3_items = p3_summary["items_evaluated"]
        p3_sentences = p3_summary["sentences_evaluated"]
        p3_keep = p3_summary["sentences_keep"]
        p3_remove = p3_summary["sentences_remove"]
        p3_issues = p3_summary["items_with_issues"]

        # Calculate Phase 3 percentages
        p3_keep_pct = (p3_keep / p3_sentences * 100) if p3_sentences > 0 else 0
        p3_remove_pct = (p3_remove / p3_sentences * 100) if p3_sentences > 0 else 0

        # Create lookup for evaluations by section + bullet_id
        phase3_evaluations_by_key = {}
        for evaluation in p3_evaluations_list:
            section = evaluation.get("section", "")
            bullet_id = evaluation.get("bullet_id", "")
            key = f"{section}|{bullet_id}" if bullet_id else section
            phase3_evaluations_by_key[key] = evaluation
    else:
        phase3_evaluations_by_key = {}

    # Phase 4 stats (if available)
    phase4_skipped = phase4_result is None
    if not phase4_skipped:
        p4_summary = phase4_result.get("summary", {})
        p4_issue_counts = p4_summary.get("issue_counts", {})
        p4_metadata_issues_count = p4_issue_counts.get("metadata_issues", 0)
        p4_placement_issues_count = p4_issue_counts.get("section_placement_issues", 0)

        # Build lookups for Phase 4 data by bullet_id
        p4_metadata_issues_by_id = {}
        for bullet_issue in phase4_result.get("metadata_verification", {}).get("bullets_with_issues", []):
            bid = bullet_issue.get("bullet_id")
            if bid:
                p4_metadata_issues_by_id[bid] = bullet_issue

        p4_placement_issues_by_id = {}
        for placement in phase4_result.get("section_placement", {}).get("misplaced_bullets", []):
            bid = placement.get("bullet_id")
            if bid:
                p4_placement_issues_by_id[bid] = placement

        p4_duplicate_themes = phase4_result.get("section_placement", {}).get("duplicate_themes", [])
    else:
        p4_metadata_issues_by_id = {}
        p4_placement_issues_by_id = {}
        p4_duplicate_themes = []
        p4_metadata_issues_count = 0
        p4_placement_issues_count = 0

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
        .theme-section {{ margin: 20px; padding: 20px; background: #fafbfc; border: 2px solid #e5e7eb; border-radius: 6px; }}
        .theme-item {{ margin: 15px 0; padding: 15px; background: white; border-radius: 4px; border-left: 3px solid #6b7280; }}
        .theme-item.covered {{ border-left-color: #10b981; }}
        .theme-item.missing {{ border-left-color: #ef4444; }}
        .missing-opp-section {{ margin-top: 10px; padding: 12px; background: #fef3c7; border-left: 3px solid #f59e0b; border-radius: 4px; }}
        .missing-opp-item {{ margin: 8px 0; padding: 8px; background: white; border-radius: 3px; font-size: 13px; }}
        .phase4-issue {{ margin: 10px 0; padding: 12px; background: #fee2e2; border-left: 3px solid #dc2626; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Quality Review: {ticker}</h1>
            <p>Comprehensive 4-Phase Article & Filing Verification Report</p>
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
            </div>'''

    # Add theme coverage stats if available
    if p1_theme_coverage:
        theme_status = p1_theme_coverage.get("status", "PASS")
        theme_pct = p1_theme_coverage.get("coverage_percentage", 100)
        themes_covered = p1_theme_coverage.get("themes_covered", 0)
        themes_missing = p1_theme_coverage.get("themes_missing", 0)
        material_themes = p1_theme_coverage.get("material_themes", 0)

        theme_badge_color = "#10b981" if theme_status == "PASS" else "#f59e0b"
        theme_badge = f'<span class="badge" style="background: {theme_badge_color}; color: white;">{"✅ PASS" if theme_status == "PASS" else "⚠️ BELOW TARGET"}</span>'

        html += f'''
            <div class="error-summary">
                <div class="stat-label">Theme Coverage {theme_badge}</div>
                <div style="margin-top: 10px;">
                    <strong>{themes_covered}/{material_themes} material themes covered ({theme_pct:.1f}%)</strong>
                    {f' - {themes_missing} themes missing' if themes_missing > 0 else ''}
                    <br><span style="font-size: 12px; color: #6b7280;">Target: 95%+</span>
                </div>
            </div>'''

    html += '''
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
            </div>'''

        # Add completeness stats if available
        if p2_completeness:
            high_priority = p2_completeness.get("high_priority_improvements", 0)
            medium_priority = p2_completeness.get("medium_priority_improvements", 0)
            low_priority = p2_completeness.get("low_priority_improvements", 0)

            if high_priority > 0 or medium_priority > 0:
                html += f'''
            <div class="error-summary">
                <div class="stat-label">Context Completeness Opportunities</div>
                <div style="margin-top: 10px;">
                    {f'<span class="badge critical">🔴 HIGH: {high_priority} improvements</span>' if high_priority > 0 else ''}
                    {f'<span class="badge serious">🟠 MEDIUM: {medium_priority} improvements</span>' if medium_priority > 0 else ''}
                    {f'<span class="badge minor">🟡 LOW: {low_priority} improvements</span>' if low_priority > 0 else ''}
                </div>
            </div>'''

        html += '''
        </div>
'''

    # Phase 3 summary box
    if phase3_skipped:
        html += '''
        <div class="phase-summary skipped">
            <h3>🔗 PHASE 3: CONTEXT RELEVANCE VERIFICATION <span class="badge" style="background: #6b7280; color: white;">SKIPPED</span></h3>
            <p style="color: #6b7280; margin: 10px 0 0 0; font-size: 14px;">No Phase 3 evaluation performed.</p>
        </div>
'''
    else:
        html += f'''
        <div class="phase-summary">
            <h3>🔗 PHASE 3: CONTEXT RELEVANCE VERIFICATION <span class="badge" style="background: #6f42c1; color: white;">ℹ️ INFORMATIONAL</span></h3>
            <div class="stats">
                <div class="stat">
                    <div class="stat-label">Items Evaluated</div>
                    <div class="stat-value">{p3_items}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Sentences Evaluated</div>
                    <div class="stat-value">{p3_sentences}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">✅ KEEP</div>
                    <div class="stat-value" style="color: #10b981;">{p3_keep} ({p3_keep_pct:.1f}%)</div>
                </div>
                <div class="stat">
                    <div class="stat-label">🔴 REMOVE</div>
                    <div class="stat-value" style="color: #ef4444;">{p3_remove} ({p3_remove_pct:.1f}%)</div>
                </div>
            </div>
            <div class="error-summary">
                <div class="stat-label">Items with Issues</div>
                <div style="margin-top: 10px;">
                    <span style="font-size: 16px; font-weight: bold; color: #6f42c1;">{p3_issues} items require context cleanup</span>
                </div>
            </div>
        </div>
'''

    # Phase 4 summary box
    if phase4_skipped:
        html += '''
        <div class="phase-summary skipped">
            <h3>🏗️ PHASE 4: METADATA & STRUCTURE VALIDATION <span class="badge" style="background: #6b7280; color: white;">SKIPPED</span></h3>
            <p style="color: #6b7280; margin: 10px 0 0 0; font-size: 14px;">No Phase 4 evaluation performed.</p>
        </div>
'''
    else:
        html += f'''
        <div class="phase-summary">
            <h3>🏗️ PHASE 4: METADATA & STRUCTURE VALIDATION <span class="badge" style="background: #8b5cf6; color: white;">ℹ️ INFORMATIONAL</span></h3>
            <div class="stats">
                <div class="stat">
                    <div class="stat-label">Metadata Issues</div>
                    <div class="stat-value" style="color: {'#ef4444' if p4_metadata_issues_count > 0 else '#10b981'};">{p4_metadata_issues_count}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Placement Issues</div>
                    <div class="stat-value" style="color: {'#ef4444' if p4_placement_issues_count > 0 else '#10b981'};">{p4_placement_issues_count}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Duplicate Themes</div>
                    <div class="stat-value" style="color: {'#f59e0b' if len(p4_duplicate_themes) > 0 else '#10b981'};">{len(p4_duplicate_themes)}</div>
                </div>
            </div>'''

        # Show duplicate themes if any
        if len(p4_duplicate_themes) > 0:
            html += '''
            <div class="error-summary">
                <div class="stat-label">Duplicate Themes Found</div>
                <div style="margin-top: 10px; font-size: 13px;">'''
            for dup in p4_duplicate_themes:
                theme = dup.get("theme", "Unknown theme")
                sections = dup.get("appears_in_sections", [])
                html += f'''
                    <div style="margin: 5px 0;">• <strong>{theme}</strong> appears in: {', '.join(sections)}</div>'''
            html += '''
                </div>
            </div>'''

        html += '''
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

    # Now render each section bullet-by-bullet with all 4 phases
    for section in p1_sections:
        section_name = section.get("section_name", "unknown")
        display_name = section_names.get(section_name, section_name.upper())
        sentences = section.get("sentences", [])

        if not sentences:
            continue

        html += f'<div class="section">'
        html += f'<div class="section-header">{display_name}</div>'

        # Check if this is a bullet section or paragraph section
        is_paragraph = section_name in ["bottom_line", "upside_scenario", "downside_scenario"]

        if is_paragraph:
            # Paragraph sections: treat entire paragraph as one "bullet"
            html += '<div class="bullet-group">'
            html += '<div class="bullet-title">Paragraph</div>'
            html += '<div class="phase-label">📰 Article Verification (Phase 1)</div>'

            # Render all sentences for the paragraph
            for sentence in sentences:
                text = sentence.get("text", "")
                status = (sentence.get("status") or "").lower()
                error_type = sentence.get("error_type")
                severity = (sentence.get("severity") or "").upper()
                evidence = sentence.get("evidence", [])
                notes = sentence.get("notes", "") or ""
                bullet_id = sentence.get("bullet_id")  # Will be section_name for paragraphs

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

            # Phase 2: Filing Context for paragraph
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
                    ctx_completeness = p2_ctx.get("completeness", {})

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

                    # NEW: Show missing opportunities (HIGH and MEDIUM only)
                    questions_missed = ctx_completeness.get("questions_missed", [])
                    high_medium_missed = [q for q in questions_missed if q.get("priority") in ["HIGH", "MEDIUM"]]

                    if high_medium_missed:
                        html += '''
                        <div class="missing-opp-section">
                            <strong>🔍 Missing Context Opportunities (HIGH/MEDIUM):</strong>'''

                        for opp in high_medium_missed:
                            priority = opp.get("priority", "UNKNOWN")
                            question_name = opp.get("question_name", "")
                            finding = opp.get("finding", "")
                            recommendation = opp.get("recommendation", "")
                            value_add = opp.get("value_add", "")

                            priority_badge = ""
                            if priority == "HIGH":
                                priority_badge = '<span class="badge critical">🔴 HIGH</span>'
                            elif priority == "MEDIUM":
                                priority_badge = '<span class="badge serious">🟠 MEDIUM</span>'

                            html += f'''
                            <div class="missing-opp-item">
                                <div>{priority_badge} <strong>{question_name}</strong></div>
                                <div style="margin-top: 5px; font-size: 12px; color: #6b7280;">
                                    <strong>Finding:</strong> {finding}<br>
                                    <strong>Recommendation:</strong> {recommendation}<br>
                                    <strong>Value:</strong> {value_add}
                                </div>
                            </div>'''

                        html += '''
                        </div>'''

            # Phase 3: Context Relevance for paragraph
            if not phase3_skipped and bullet_id:
                phase3_key = section_name
                p3_eval = phase3_evaluations_by_key.get(phase3_key)

                if p3_eval:
                    html += '<div class="divider"></div>'
                    html += '<div class="phase-label">🔗 Context Relevance Verification (Phase 3)</div>'

                    sentences_eval = p3_eval.get("sentences", [])
                    total_sentences_eval = len(sentences_eval)
                    keep_count = sum(1 for s in sentences_eval if s.get("decision") == "KEEP")
                    remove_count = sum(1 for s in sentences_eval if s.get("decision") == "REMOVE")

                    # Summary line
                    if remove_count == 0:
                        summary_text = f'✅ ALL RELEVANT ({total_sentences_eval} sentences, 100% relevant)'
                    else:
                        summary_text = f'⚠️ {keep_count} KEEP, {remove_count} REMOVE ({total_sentences_eval} sentences, {keep_count*100//total_sentences_eval if total_sentences_eval > 0 else 0}% relevant)'

                    html += f'<div style="margin: 10px 0; font-weight: 600; color: #6f42c1;">{summary_text}</div>'

                    # Show sentences with issues only (skip ones marked KEEP with HIGH confidence)
                    for sentence_eval in sentences_eval:
                        decision = sentence_eval.get("decision", "")
                        confidence = sentence_eval.get("confidence", "")

                        # Skip KEEP sentences with HIGH confidence (no issues)
                        if decision == "KEEP" and confidence == "HIGH":
                            continue

                        sentence_num = sentence_eval.get("sentence_num", "")
                        text_eval = sentence_eval.get("text", "")
                        confidence_score = sentence_eval.get("confidence_score", "")
                        test1_result = sentence_eval.get("test1_result", "")
                        test1_reason = sentence_eval.get("test1_reason", "")
                        test2_result = sentence_eval.get("test2_result", "")
                        test2_reason = sentence_eval.get("test2_reason", "")
                        test3_result = sentence_eval.get("test3_result", "")
                        test3_reason = sentence_eval.get("test3_reason", "")

                        # Decision badge
                        if decision == "KEEP":
                            decision_badge = f'<span class="badge supported">✅ KEEP ({confidence} {confidence_score})</span>'
                            item_class = "supported"
                        else:
                            decision_badge = f'<span class="badge unsupported">🔴 REMOVE ({confidence} {confidence_score})</span>'
                            item_class = "unsupported"

                        html += f'''
                        <div class="review-item {item_class}" style="margin-left: 20px;">
                            <div class="review-meta">{decision_badge}</div>
                            <div class="review-text"><strong>Sentence {sentence_num}:</strong> {text_eval}</div>
                            <div style="margin-top: 8px; margin-left: 20px; font-size: 13px; color: #6b7280;">
                                └─ Test 1 (Connection): {test1_result} - {test1_reason}<br>
                                └─ Test 2 (Specificity): {test2_result} - {test2_reason}<br>
                                └─ Test 3 (Connector): {test3_result} - {test3_reason}
                            </div>
                        </div>
                        '''

            # Phase 4: Metadata & Structure for paragraph
            if not phase4_skipped and bullet_id:
                metadata_issues = p4_metadata_issues_by_id.get(bullet_id, {}).get("issues", [])
                placement_issue = p4_placement_issues_by_id.get(bullet_id)

                if metadata_issues or placement_issue:
                    html += '<div class="divider"></div>'
                    html += '<div class="phase-label">🏗️ Metadata & Structure Validation (Phase 4)</div>'

                    issue_count = len(metadata_issues) + (1 if placement_issue else 0)
                    html += f'<div style="margin: 10px 0; font-weight: 600; color: #8b5cf6;">⚠️ {issue_count} ISSUE{"S" if issue_count > 1 else ""} FOUND</div>'

                    # Metadata issues
                    for idx, meta_issue in enumerate(metadata_issues, 1):
                        field = meta_issue.get("field", "unknown")
                        current_value = meta_issue.get("current_value", "")
                        recommended_value = meta_issue.get("recommended_value", "")
                        reason = meta_issue.get("reason", "")
                        severity = meta_issue.get("severity", "MINOR")

                        severity_badge = ""
                        if severity == "CRITICAL":
                            severity_badge = '<span class="badge critical">🔴 CRITICAL</span>'
                        elif severity == "SERIOUS":
                            severity_badge = '<span class="badge serious">🟠 SERIOUS</span>'
                        elif severity == "MINOR":
                            severity_badge = '<span class="badge minor">🟡 MINOR</span>'

                        html += f'''
                        <div class="phase4-issue">
                            <div style="font-weight: bold; margin-bottom: 5px;">Issue {idx}: Metadata - {field.title()} {severity_badge}</div>
                            <div style="font-size: 13px;">
                                <strong>Current:</strong> {current_value}<br>
                                <strong>Recommended:</strong> {recommended_value}<br>
                                <strong>Reason:</strong> {reason}
                            </div>
                        </div>'''

                    # Placement issue
                    if placement_issue:
                        current_section = placement_issue.get("current_section", "")
                        correct_section = placement_issue.get("correct_section", "")
                        rule_violated = placement_issue.get("rule_violated", "")
                        severity = placement_issue.get("severity", "MINOR")

                        severity_badge = ""
                        if severity == "CRITICAL":
                            severity_badge = '<span class="badge critical">🔴 CRITICAL</span>'
                        elif severity == "SERIOUS":
                            severity_badge = '<span class="badge serious">🟠 SERIOUS</span>'
                        elif severity == "MINOR":
                            severity_badge = '<span class="badge minor">🟡 MINOR</span>'

                        issue_num = len(metadata_issues) + 1
                        html += f'''
                        <div class="phase4-issue">
                            <div style="font-weight: bold; margin-bottom: 5px;">Issue {issue_num}: Section Placement {severity_badge}</div>
                            <div style="font-size: 13px;">
                                <strong>Current Section:</strong> {section_names.get(current_section, current_section)}<br>
                                <strong>Correct Section:</strong> {section_names.get(correct_section, correct_section)}<br>
                                <strong>Reason:</strong> {rule_violated}
                            </div>
                        </div>'''

            html += '</div>'  # Close bullet-group

        else:
            # Bullet sections: render each bullet separately with all 4 phases
            for sentence in sentences:
                bullet_id = sentence.get("bullet_id", "")
                topic_label = sentence.get("topic_label", "Bullet")
                text = sentence.get("text", "")
                status = (sentence.get("status") or "").lower()
                error_type = sentence.get("error_type")
                severity = (sentence.get("severity") or "").upper()
                evidence = sentence.get("evidence", [])
                notes = sentence.get("notes", "") or ""

                html += '<div class="bullet-group">'
                html += f'<div class="bullet-title">{topic_label}</div>'

                # Phase 1: Article Verification
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

                # Phase 2: Filing Context Verification
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
                        ctx_completeness = p2_ctx.get("completeness", {})

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

                        # NEW: Show missing opportunities (HIGH and MEDIUM only)
                        questions_missed = ctx_completeness.get("questions_missed", [])
                        high_medium_missed = [q for q in questions_missed if q.get("priority") in ["HIGH", "MEDIUM"]]

                        if high_medium_missed:
                            html += '''
                            <div class="missing-opp-section">
                                <strong>🔍 Missing Context Opportunities (HIGH/MEDIUM):</strong>'''

                            for opp in high_medium_missed:
                                priority = opp.get("priority", "UNKNOWN")
                                question_name = opp.get("question_name", "")
                                finding = opp.get("finding", "")
                                recommendation = opp.get("recommendation", "")
                                value_add = opp.get("value_add", "")

                                priority_badge = ""
                                if priority == "HIGH":
                                    priority_badge = '<span class="badge critical">🔴 HIGH</span>'
                                elif priority == "MEDIUM":
                                    priority_badge = '<span class="badge serious">🟠 MEDIUM</span>'

                                html += f'''
                                <div class="missing-opp-item">
                                    <div>{priority_badge} <strong>{question_name}</strong></div>
                                    <div style="margin-top: 5px; font-size: 12px; color: #6b7280;">
                                        <strong>Finding:</strong> {finding}<br>
                                        <strong>Recommendation:</strong> {recommendation}<br>
                                        <strong>Value:</strong> {value_add}
                                    </div>
                                </div>'''

                            html += '''
                            </div>'''

                # Phase 3: Context Relevance Verification
                if not phase3_skipped and bullet_id:
                    context_key = f"{section_name}|{bullet_id}"
                    p3_eval = phase3_evaluations_by_key.get(context_key)

                    if p3_eval:
                        html += '<div class="divider"></div>'
                        html += '<div class="phase-label">🔗 Context Relevance Verification (Phase 3)</div>'

                        sentences_eval = p3_eval.get("sentences", [])
                        total_sentences_eval = len(sentences_eval)
                        keep_count = sum(1 for s in sentences_eval if s.get("decision") == "KEEP")
                        remove_count = sum(1 for s in sentences_eval if s.get("decision") == "REMOVE")

                        # Summary line
                        if remove_count == 0:
                            summary_text = f'✅ ALL RELEVANT ({total_sentences_eval} sentences, 100% relevant)'
                        else:
                            summary_text = f'⚠️ {keep_count} KEEP, {remove_count} REMOVE ({total_sentences_eval} sentences, {keep_count*100//total_sentences_eval if total_sentences_eval > 0 else 0}% relevant)'

                        html += f'<div style="margin: 10px 0; font-weight: 600; color: #6f42c1;">{summary_text}</div>'

                        # Show sentences with issues only
                        for sentence_eval in sentences_eval:
                            decision = sentence_eval.get("decision", "")
                            confidence = sentence_eval.get("confidence", "")

                            # Skip KEEP sentences with HIGH confidence
                            if decision == "KEEP" and confidence == "HIGH":
                                continue

                            sentence_num = sentence_eval.get("sentence_num", "")
                            text_eval = sentence_eval.get("text", "")
                            confidence_score = sentence_eval.get("confidence_score", "")
                            test1_result = sentence_eval.get("test1_result", "")
                            test1_reason = sentence_eval.get("test1_reason", "")
                            test2_result = sentence_eval.get("test2_result", "")
                            test2_reason = sentence_eval.get("test2_reason", "")
                            test3_result = sentence_eval.get("test3_result", "")
                            test3_reason = sentence_eval.get("test3_reason", "")

                            # Decision badge
                            if decision == "KEEP":
                                decision_badge = f'<span class="badge supported">✅ KEEP ({confidence} {confidence_score})</span>'
                                item_class = "supported"
                            else:
                                decision_badge = f'<span class="badge unsupported">🔴 REMOVE ({confidence} {confidence_score})</span>'
                                item_class = "unsupported"

                            html += f'''
                            <div class="review-item {item_class}" style="margin-left: 20px;">
                                <div class="review-meta">{decision_badge}</div>
                                <div class="review-text"><strong>Sentence {sentence_num}:</strong> {text_eval}</div>
                                <div style="margin-top: 8px; margin-left: 20px; font-size: 13px; color: #6b7280;">
                                    └─ Test 1 (Connection): {test1_result} - {test1_reason}<br>
                                    └─ Test 2 (Specificity): {test2_result} - {test2_reason}<br>
                                    └─ Test 3 (Connector): {test3_result} - {test3_reason}
                                </div>
                            </div>
                            '''

                # Phase 4: Metadata & Structure Validation (NEW)
                if not phase4_skipped and bullet_id:
                    metadata_issues = p4_metadata_issues_by_id.get(bullet_id, {}).get("issues", [])
                    placement_issue = p4_placement_issues_by_id.get(bullet_id)

                    if metadata_issues or placement_issue:
                        html += '<div class="divider"></div>'
                        html += '<div class="phase-label">🏗️ Metadata & Structure Validation (Phase 4)</div>'

                        issue_count = len(metadata_issues) + (1 if placement_issue else 0)
                        html += f'<div style="margin: 10px 0; font-weight: 600; color: #8b5cf6;">⚠️ {issue_count} ISSUE{"S" if issue_count > 1 else ""} FOUND</div>'

                        # Metadata issues
                        for idx, meta_issue in enumerate(metadata_issues, 1):
                            field = meta_issue.get("field", "unknown")
                            current_value = meta_issue.get("current_value", "")
                            recommended_value = meta_issue.get("recommended_value", "")
                            reason = meta_issue.get("reason", "")
                            severity = meta_issue.get("severity", "MINOR")

                            severity_badge = ""
                            if severity == "CRITICAL":
                                severity_badge = '<span class="badge critical">🔴 CRITICAL</span>'
                            elif severity == "SERIOUS":
                                severity_badge = '<span class="badge serious">🟠 SERIOUS</span>'
                            elif severity == "MINOR":
                                severity_badge = '<span class="badge minor">🟡 MINOR</span>'

                            html += f'''
                            <div class="phase4-issue">
                                <div style="font-weight: bold; margin-bottom: 5px;">Issue {idx}: Metadata - {field.title()} {severity_badge}</div>
                                <div style="font-size: 13px;">
                                    <strong>Current:</strong> {current_value}<br>
                                    <strong>Recommended:</strong> {recommended_value}<br>
                                    <strong>Reason:</strong> {reason}
                                </div>
                            </div>'''

                        # Placement issue
                        if placement_issue:
                            current_section = placement_issue.get("current_section", "")
                            correct_section = placement_issue.get("correct_section", "")
                            rule_violated = placement_issue.get("rule_violated", "")
                            severity = placement_issue.get("severity", "MINOR")

                            severity_badge = ""
                            if severity == "CRITICAL":
                                severity_badge = '<span class="badge critical">🔴 CRITICAL</span>'
                            elif severity == "SERIOUS":
                                severity_badge = '<span class="badge serious">🟠 SERIOUS</span>'
                            elif severity == "MINOR":
                                severity_badge = '<span class="badge minor">🟡 MINOR</span>'

                            issue_num = len(metadata_issues) + 1
                            html += f'''
                            <div class="phase4-issue">
                                <div style="font-weight: bold; margin-bottom: 5px;">Issue {issue_num}: Section Placement {severity_badge}</div>
                                <div style="font-size: 13px;">
                                    <strong>Current Section:</strong> {section_names.get(current_section, current_section)}<br>
                                    <strong>Correct Section:</strong> {section_names.get(correct_section, correct_section)}<br>
                                    <strong>Reason:</strong> {rule_violated}
                                </div>
                            </div>'''

                html += '</div>'  # Close bullet-group

        html += '</div>'  # Close section

    # NEW: Theme Analysis Section (at end, after all sections)
    if p1_themes and p1_themes.get("themes"):
        themes = p1_themes.get("themes", [])
        material_themes = [t for t in themes if t.get("materiality_rating") == "Material"]
        covered_themes = [t for t in material_themes if t.get("status") == "COVERED"]
        missing_themes = [t for t in material_themes if t.get("status") == "MISSING"]

        coverage_pct = (len(covered_themes) / len(material_themes) * 100) if len(material_themes) > 0 else 100
        coverage_status = "✅ PASS (≥95%)" if coverage_pct >= 95 else "⚠️ BELOW TARGET (<95%)"

        html += f'''
        <div class="theme-section">
            <h3 style="margin: 0 0 15px 0; color: #1e40af;">📊 THEME ANALYSIS (Phase 1 - Coverage Check)</h3>
            <div style="font-size: 16px; font-weight: bold; margin-bottom: 20px;">
                Material Themes Coverage: {len(covered_themes)}/{len(material_themes)} ({coverage_pct:.1f}%) - {coverage_status}
            </div>'''

        if covered_themes:
            html += '''
            <div style="margin-bottom: 20px;">
                <h4 style="color: #10b981; margin-bottom: 10px;">✅ COVERED THEMES ({len(covered_themes)}):</h4>'''

            for theme in covered_themes:
                theme_name = theme.get("theme", "Unknown")
                category = theme.get("category", "")
                covered_section = theme.get("covered_in_section", "")

                html += f'''
                <div class="theme-item covered">
                    <div style="font-weight: bold;">{theme_name}</div>
                    <div style="font-size: 13px; color: #6b7280; margin-top: 5px;">
                        Category: {category} | Covered in: {section_names.get(covered_section, covered_section)}
                    </div>
                </div>'''

            html += '''
            </div>'''

        if missing_themes:
            html += f'''
            <div>
                <h4 style="color: #ef4444; margin-bottom: 10px;">❌ MISSING THEMES ({len(missing_themes)}):</h4>'''

            for theme in missing_themes:
                theme_name = theme.get("theme", "Unknown")
                category = theme.get("category", "")
                mentioned_in = theme.get("mentioned_in_articles", [])
                investor_test = theme.get("investor_test", "")
                earnings_test = theme.get("earnings_call_test", "")
                valuation_test = theme.get("valuation_test", "")

                html += f'''
                <div class="theme-item missing">
                    <div style="font-weight: bold;">{theme_name}</div>
                    <div style="font-size: 13px; color: #6b7280; margin-top: 5px;">
                        <strong>Category:</strong> {category}<br>
                        <strong>Mentioned in Articles:</strong> {', '.join(map(str, mentioned_in))}<br>
                        <strong>Materiality Tests:</strong><br>
                        <div style="margin-left: 15px;">
                            • Investor Test: {investor_test}<br>
                            • Earnings Call Test: {earnings_test}<br>
                            • Valuation Test: {valuation_test}
                        </div>
                    </div>
                </div>'''

            html += '''
            </div>'''

        html += '''
        </div>'''

    # Footer
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
