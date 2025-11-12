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
    Generate bullet-centric HTML email report showing all phases per bullet/section.

    Args:
        phase1_result: Output from review_executive_summary_quality() (includes theme analysis)
        phase2_result: Output from review_phase2_context_quality() or None
        phase3_result: Output from review_context_relevance() or None
        phase4_result: Output from review_phase4_metadata_and_structure() or None

    Returns:
        HTML string for email with bullet-centric organization
    """
    from modules.quality_review_phase2 import review_phase2_context_quality
    from modules.quality_review_phase3 import review_context_relevance
    from modules.quality_review_phase4 import review_phase4_metadata_and_structure
    
    p1_summary = phase1_result["summary"]
    p1_sections = phase1_result["sections"]
    p1_themes = phase1_result.get("theme_analysis", {})

    ticker = p1_summary["ticker"]

    # Build lookups for Phase 2, 3, 4 data by section + bullet_id
    p2_contexts_by_key = {}
    if phase2_result:
        for ctx in phase2_result.get("contexts", []):
            section = ctx.get("section_name", "")
            bullet_id = ctx.get("bullet_id", "")
            key = f"{section}|{bullet_id}"
            p2_contexts_by_key[key] = ctx

    p3_evaluations_by_key = {}
    if phase3_result:
        for evaluation in phase3_result.get("evaluations", []):
            section = evaluation.get("section", "")
            bullet_id = evaluation.get("bullet_id", "")
            key = f"{section}|{bullet_id}" if bullet_id else section
            p3_evaluations_by_key[key] = evaluation

    p4_metadata_issues = {}
    p4_placement_issues = {}
    if phase4_result:
        for bullet_issue in phase4_result.get("metadata_verification", {}).get("bullets_with_issues", []):
            bid = bullet_issue.get("bullet_id")
            p4_metadata_issues[bid] = bullet_issue

        for placement in phase4_result.get("section_placement", {}).get("misplaced_bullets", []):
            bid = placement.get("bullet_id")
            p4_placement_issues[bid] = placement

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

    html_parts = []
    html_parts.append(f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Courier New', monospace; background-color: #f3f4f6; margin: 0; padding: 20px; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .header {{ font-size: 24px; font-weight: bold; margin-bottom: 20px; border-bottom: 3px solid #1e40af; padding-bottom: 10px; }}
        .section-header {{ font-size: 18px; font-weight: bold; margin-top: 40px; margin-bottom: 15px; border-top: 3px double #666; padding-top: 20px; }}
        .bullet-header {{ font-size: 16px; font-weight: bold; margin-top: 25px; margin-bottom: 10px; background: #f3f4f6; padding: 12px; border-left: 4px solid #1e40af; }}
        .phase {{ margin-left: 20px; margin-bottom: 20px; }}
        .phase-title {{ font-weight: bold; color: #1e40af; margin-bottom: 8px; font-size: 15px; }}
        .detail {{ margin-left: 20px; line-height: 1.8; color: #374151; }}
        .issue {{ background: #fee2e2; padding: 12px; margin: 10px 0 10px 20px; border-left: 4px solid #dc2626; }}
        .pass {{ background: #d1fae5; padding: 12px; margin: 10px 0 10px 20px; border-left: 4px solid #10b981; }}
        .warning {{ background: #fef3c7; padding: 12px; margin: 10px 0 10px 20px; border-left: 4px solid #f59e0b; }}
        .missing-opp {{ margin: 8px 0; padding: 10px; background: #fff; border: 1px solid #d1d5db; }}
        .theme-section {{ margin-top: 40px; padding: 20px; background: #fafbfc; border: 2px solid #e5e7eb; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">🔍 Quality Review: {ticker}</div>
''')

    # Import the section rendering code from temp file
    exec(open('/tmp/bullet_centric_email_function.py').read())

    html_parts.append('''
        <div style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #e5e7eb; text-align: center; color: #6b7280; font-size: 12px;">
            <p>Generated by StockDigest Quality Review System</p>
            <p>Powered by Gemini 2.5 Flash</p>
        </div>
    </div>
</body>
</html>
''')

    return ''.join(html_parts)
