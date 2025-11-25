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

⚠️ BE CRITICAL AND THOROUGH: Context fields must be 100% accurate or they undermine the entire summary. Investors will cross-check these claims against SEC filings. Treat every number and claim as if it will be scrutinized by a skeptical CFO.

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

⚠️ RULE 0 - CHECK THIS FIRST (Empty Context Handling):

IF context field is empty, null, or contains only whitespace:
  → Status: "ACCURATE"
  → Note: "No context provided - auto-pass"
  → DO NOT treat as error
  → DO NOT flag as Context Fabrication
  → SKIP to next context (do not check other rules below)

Examples of empty context (ALL should return status "ACCURATE"):
  • "context": ""
  • "context": null
  • "context": "   "
  • "context": "\n"

WHY: Empty context means enrichment was skipped (e.g., low-priority bullet,
tangential topic, neutral sentiment). This is expected behavior and NOT an error.

═══════════════════════════════════════════

IF context field HAS CONTENT (non-empty string):
Proceed with normal verification below...

For each NON-EMPTY context field, classify verification status:

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
      "context_text": "Full context text here" | "",
      "status": "ACCURATE" | "ISSUE",
      "error_type": "Context Fabrication" | "Context-Bullet Contradiction" | "Wrong Context Source" | "Context Irrelevance" | "Missing Critical Baseline" | "Context Doesn't Add Value" | "Excessive Cross-Context Duplication" | null,
      "severity": "CRITICAL" | "SERIOUS" | "MINOR" | null,
      "evidence": ["Filing location where data was found or should be found"],
      "notes": "Explanation of verification result"
    }
  ]
}

IMPORTANT NOTES:
- Empty contexts (empty string, null, whitespace) should be marked as status: "ACCURATE"
- For empty contexts, use notes: "No context provided - auto-pass"
- Empty contexts count toward the "accurate" total, not "issues"
- Only non-empty contexts should be verified for errors
- Include ALL contexts in output (both empty and non-empty)

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

═══════════════════════════════════════════
EXAMPLE: EMPTY CONTEXT HANDLING
═══════════════════════════════════════════

Input bullet with empty context:
{
  "bullet_id": "mini_truck_market",
  "topic_label": "Mini truck market growth",
  "content": "Global mini truck market projected to reach $97.71 billion by 2035...",
  "context": "",
  "impact": "low impact",
  "sentiment": "neutral"
}

✅ CORRECT output:
{
  "section_name": "competitive_industry_dynamics",
  "bullet_id": "mini_truck_market",
  "context_text": "",
  "status": "ACCURATE",
  "error_type": null,
  "severity": null,
  "evidence": [],
  "notes": "No context provided - auto-pass"
}

❌ INCORRECT output (DO NOT DO THIS):
{
  "section_name": "competitive_industry_dynamics",
  "bullet_id": "mini_truck_market",
  "context_text": "",
  "status": "ISSUE",
  "error_type": "Context Fabrication",
  "severity": "CRITICAL",
  "evidence": [],
  "notes": "Context field is empty, providing no information to verify..."
}

Review ALL context fields in ALL 10 sections (even if some sections have no contexts).

═══════════════════════════════════════════
COMPLETENESS EVALUATION (12 QUESTIONS FRAMEWORK)
═══════════════════════════════════════════

After verifying context accuracy, evaluate completeness using the 12-question framework.

For each NON-EMPTY context, assess which questions are answered and which opportunities were missed.

THE 12 CONTEXT QUESTIONS:

1. MATERIALITY - What % of revenue, profit, exposure? What $ amount? Ranking?
2. ECONOMICS - Margins, profitability, returns (ROE/ROIC)? Cash generation?
3. TRENDS - QoQ sequential? YoY comparison? Multi-year baseline?
4. SEGMENT BREAKDOWN - By division, product, geography? Growth rates?
5. DEPENDENCIES - Customer/supplier concentration? Counterparty risk?
6. HISTORICAL BASELINE - vs 3-5 year average? vs historical peak? vs peers?
7. MANAGEMENT EXPLANATION - Why did this happen? What's the outlook? (from transcript)
8. TIMELINE - Specific dates? Expected duration? Milestones?
9. BUSINESS MODEL - How this fits strategy? What entity is? Why it matters?
10. BALANCE - Offsetting factors? Trade-offs? Risks vs opportunities?
11. CONTEXT - Related developments? Contingencies? Regulatory/market context?
12. ENTITY DEFINITION - What are subsidiaries/products/technologies? How big? What do they do?

PRIORITY ASSIGNMENT:

HIGH Priority:
- Question 1 (MATERIALITY) - Essential for assessing scale
- Question 2 (ECONOMICS) - Essential for assessing quality
- Question 3 (TRENDS) - Essential for assessing trajectory
- Question 7 (MANAGEMENT EXPLANATION) - Essential for understanding "why"

MEDIUM Priority:
- Question 4 (SEGMENT BREAKDOWN), 5 (DEPENDENCIES), 9 (BUSINESS MODEL), 12 (ENTITY DEFINITION)

LOW Priority:
- Question 6 (HISTORICAL BASELINE), 8 (TIMELINE), 10 (BALANCE), 11 (CONTEXT)

EVALUATION PROCESS:

For each context with content:
1. Identify which questions are already answered
2. Search filings for data that could answer unanswered questions
3. For each unanswered question with available data:
   - Extract specific finding from filings
   - Craft recommendation for how to add it
   - Explain value-add
   - Assign priority

OUTPUT FORMAT WITH COMPLETENESS:

{
  "contexts": [
    {
      "section_name": "major_developments",
      "bullet_id": "acquisition_announced",
      "context_text": "Amedisys operates in home health with $2.1B revenue per 10-K",
      "status": "ACCURATE",
      "error_type": null,
      "severity": null,
      "evidence": ["FY2024 10-K, Business Description section"],
      "notes": "Context accurately sourced from 10-K",

      "completeness": {
        "questions_answered": [1, 9],
        "questions_applicable": [1, 2, 3, 4, 9, 12],
        "completeness_score": "2/6 applicable questions answered",
        "questions_missed": [
          {
            "question_num": 2,
            "question_name": "ECONOMICS",
            "finding": "Target EBITDA margin -15% per Q3 2025 10-Q Segment Performance",
            "recommendation": "Add: 'with -15% EBITDA margin per Q3 10-Q'",
            "value_add": "Quantifies profitability risk - acquiring unprofitable business",
            "filing_source": "Q3 2025 10-Q, Segment Performance section",
            "priority": "HIGH"
          },
          {
            "question_num": 3,
            "question_name": "TRENDS",
            "finding": "Revenue declined from $2.3B (FY2023) to $2.1B (FY2024) per 10-K",
            "recommendation": "Add: 'revenue declining 9% YoY from $2.3B peak'",
            "value_add": "Shows negative trajectory, not just snapshot",
            "filing_source": "FY2024 10-K, Revenue Trends section",
            "priority": "HIGH"
          }
        ],
        "improvement_priority": "HIGH"
      }
    }
  ]
}

For contexts with no missing opportunities:
"completeness": {
  "questions_answered": [1, 2, 3, 7],
  "questions_applicable": [1, 2, 3, 7, 9],
  "completeness_score": "4/5 applicable questions answered",
  "questions_missed": [],
  "improvement_priority": "NONE"
}

For empty contexts, completeness is not applicable (skip this field entirely).

═══════════════════════════════════════════
TASK 2: METADATA VALIDATION
═══════════════════════════════════════════

Our system tracks competitors, suppliers, customers, and industry keywords for each ticker. These are generated by AI during ticker initialization, but they may not align with what the company actually discloses in filings. Your job is to validate whether each configured entity/keyword is truly relevant based on filing evidence.

**Configured Metadata to Check:**
```
{metadata_json}
```

**Where This Information Lives:**
- **10-K filings** typically discuss: Business description, competitive landscape, major customers/suppliers (especially if >10% of revenue/costs), risk factors, market dynamics, segment operations
- **10-Q filings** typically discuss: Quarterly segment performance, material business developments, market conditions, competitive pressures
- **Transcripts** typically discuss: Management commentary on competitors, operational drivers, strategic initiatives, customer/supplier relationships, industry trends

**What "Material" Means:**
- **Competitors:** Direct competitors named in competitive landscape discussions, companies compared in analyst questions, businesses competing for same customers/markets
- **Upstream (Suppliers):** Named relationships (especially concentration disclosures), recurring discussion of specific partners, material dependencies mentioned in risks
- **Downstream (Customers):** Named relationships (especially concentration disclosures), recurring discussion of specific partners, material dependencies mentioned in risks
- **Industry Keywords:** Fundamental drivers management uses to explain performance, risks the company highlights, strategic themes that recur across filings

**Your Task:**
For each configured entity/keyword, determine:
1. **Pass:** Entity/keyword is clearly material based on filing evidence (named, discussed, disclosed)
2. **Fail:** Entity/keyword doesn't appear relevant - then suggest ONE better alternative you see in the filings

**What to Suggest When Something Fails:**
- Look for subsidiaries worth tracking separately
- Look for pending M&A targets or integration discussions
- Look for major partners/customers that ARE named
- Look for competitors that ARE discussed
- Look for keywords management ACTUALLY uses
- Don't duplicate suggestions across entities

**Output Format for Metadata Validation:**
Add a "metadata_validation" field to the root of your JSON response:

```json
{
  "contexts": [...],  // Existing TASK 1 output
  "metadata_validation": {
    "competitors": [
      {
        "entity": "MSFT",
        "status": "pass",
        "reason": "Verified in 10-K Section 1 competitive landscape"
      },
      {
        "entity": "GOOGL",
        "status": "fail",
        "suggestion": "META",
        "reason": "META discussed in 5 sections as primary social competitor, GOOGL not mentioned"
      }
    ],
    "upstream": [
      {
        "entity": "TSM",
        "status": "pass",
        "reason": "Disclosed as critical semiconductor supplier in 10-K Item 1"
      }
    ],
    "downstream": [
      {
        "entity": "Best Buy",
        "status": "fail",
        "suggestion": "None - direct to consumer model",
        "reason": "No major retail partners disclosed in filings"
      }
    ],
    "industry_keywords": [
      {
        "keyword": "smartphone sales",
        "status": "pass",
        "reason": "Management frames iPhone performance around this metric in transcripts"
      },
      {
        "keyword": "5G deployment",
        "status": "fail",
        "suggestion": "services revenue growth",
        "reason": "Management focuses on services mix, not 5G infrastructure"
      }
    ]
  }
}
```

**Important Notes:**
- If metadata field is empty (no competitors/upstream/downstream configured), return empty array for that category
- Each entity gets ONE status (pass or fail)
- If fail, provide ONE specific suggestion (or "None" if no obvious alternative)
- Keep reasons concise (1-2 sentences max)

Return ONLY the JSON object with BOTH context accuracy verification AND metadata validation, no other text.
"""


def review_phase2_context_quality(
    ticker: str,
    executive_summary: Dict,
    filings: Dict,
    gemini_api_key: str,
    ticker_metadata: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Review executive summary context fields against filing sources using Gemini.

    Args:
        ticker: Stock ticker
        executive_summary: Full Phase 1 + Phase 2 merged executive summary JSON
        filings: Dict with keys '10k', '10q', 'transcript' (from _fetch_available_filings)
        gemini_api_key: Gemini API key
        ticker_metadata: Optional ticker metadata dict with competitors, value_chain, industry_keywords

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
            "contexts": [...],  # Detailed review by context
            "metadata_validation": {...}  # Metadata validation results (if ticker_metadata provided)
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
            quarter = t.get('fiscal_quarter', 'Unknown')
            year = t.get('fiscal_year', 'Unknown')
            company = t.get('company_name', ticker)
            filings_text += f"EARNINGS TRANSCRIPT ({quarter} {year}):\n\n"
            filings_text += f"{t['text']}\n\n\n"

        # Format ticker metadata for injection into prompt
        metadata_json_str = ""
        if ticker_metadata:
            # Extract tickers from competitors
            competitors_list = []
            for comp in ticker_metadata.get("competitors", []):
                if isinstance(comp, dict):
                    competitors_list.append(comp.get("ticker", comp.get("name", "")))
                elif isinstance(comp, str):
                    competitors_list.append(comp)

            # Extract tickers from value chain
            upstream_list = []
            for vc in ticker_metadata.get("value_chain", {}).get("upstream", []):
                if isinstance(vc, dict):
                    upstream_list.append(vc.get("ticker", vc.get("name", "")))
                elif isinstance(vc, str):
                    upstream_list.append(vc)

            downstream_list = []
            for vc in ticker_metadata.get("value_chain", {}).get("downstream", []):
                if isinstance(vc, dict):
                    downstream_list.append(vc.get("ticker", vc.get("name", "")))
                elif isinstance(vc, str):
                    downstream_list.append(vc)

            # Get industry keywords
            keywords_list = ticker_metadata.get("industry_keywords", [])

            metadata_dict = {
                "competitors": competitors_list,
                "upstream": upstream_list,
                "downstream": downstream_list,
                "industry_keywords": keywords_list
            }
            metadata_json_str = json.dumps(metadata_dict, indent=2)
        else:
            metadata_json_str = json.dumps({
                "competitors": [],
                "upstream": [],
                "downstream": [],
                "industry_keywords": []
            }, indent=2)

        # Inject metadata JSON into prompt
        prompt_with_metadata = PHASE2_QUALITY_REVIEW_PROMPT.replace("{metadata_json}", metadata_json_str)

        # Combine into full prompt
        full_prompt = prompt_with_metadata + "\n" + summary_text + filings_text

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

        # Completeness statistics
        contexts_with_gaps = 0
        high_priority_improvements = 0
        medium_priority_improvements = 0
        low_priority_improvements = 0

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

            # Count completeness gaps
            completeness = context.get("completeness", {})
            if completeness:
                questions_missed = completeness.get("questions_missed", [])
                if questions_missed:
                    contexts_with_gaps += 1

                    for missed in questions_missed:
                        priority = missed.get("priority", "LOW")
                        if priority == "HIGH":
                            high_priority_improvements += 1
                        elif priority == "MEDIUM":
                            medium_priority_improvements += 1
                        else:
                            low_priority_improvements += 1

        # Extract metadata validation if present
        metadata_validation = result_json.get("metadata_validation")

        # Count metadata validation stats if available
        metadata_stats = None
        if metadata_validation:
            stats = {
                "competitors_pass": 0,
                "competitors_fail": 0,
                "upstream_pass": 0,
                "upstream_fail": 0,
                "downstream_pass": 0,
                "downstream_fail": 0,
                "keywords_pass": 0,
                "keywords_fail": 0
            }

            for comp in metadata_validation.get("competitors", []):
                if comp.get("status") == "pass":
                    stats["competitors_pass"] += 1
                else:
                    stats["competitors_fail"] += 1

            for up in metadata_validation.get("upstream", []):
                if up.get("status") == "pass":
                    stats["upstream_pass"] += 1
                else:
                    stats["upstream_fail"] += 1

            for down in metadata_validation.get("downstream", []):
                if down.get("status") == "pass":
                    stats["downstream_pass"] += 1
                else:
                    stats["downstream_fail"] += 1

            for kw in metadata_validation.get("industry_keywords", []):
                if kw.get("status") == "pass":
                    stats["keywords_pass"] += 1
                else:
                    stats["keywords_fail"] += 1

            metadata_stats = stats

        # Build final result
        final_result = {
            "summary": {
                "ticker": ticker,
                "total_contexts": total_contexts,
                "accurate": accurate,
                "issues": issues,
                "errors_by_severity": errors_by_severity,
                "generation_time_ms": generation_time,
                "completeness": {
                    "contexts_with_gaps": contexts_with_gaps,
                    "high_priority_improvements": high_priority_improvements,
                    "medium_priority_improvements": medium_priority_improvements,
                    "low_priority_improvements": low_priority_improvements
                },
                "metadata_stats": metadata_stats  # NEW: Add metadata stats to summary
            },
            "contexts": result_json.get("contexts", [])
        }

        # Add metadata_validation to result if available
        if metadata_validation:
            final_result["metadata_validation"] = metadata_validation

        LOG.info(f"✅ [{ticker}] Phase 2 quality review complete: {accurate}/{total_contexts} accurate, "
                f"{issues} issues, {errors_by_severity['CRITICAL']} critical errors, "
                f"{high_priority_improvements} high-priority improvements available")

        if metadata_stats:
            LOG.info(f"📊 [{ticker}] Metadata validation: "
                    f"Competitors {metadata_stats['competitors_pass']}/{metadata_stats['competitors_pass'] + metadata_stats['competitors_fail']}, "
                    f"Upstream {metadata_stats['upstream_pass']}/{metadata_stats['upstream_pass'] + metadata_stats['upstream_fail']}, "
                    f"Downstream {metadata_stats['downstream_pass']}/{metadata_stats['downstream_pass'] + metadata_stats['downstream_fail']}, "
                    f"Keywords {metadata_stats['keywords_pass']}/{metadata_stats['keywords_pass'] + metadata_stats['keywords_fail']}")

        return final_result

    except json.JSONDecodeError as e:
        LOG.error(f"[{ticker}] Failed to parse Gemini Phase 2 response as JSON: {e}")
        return None
    except Exception as e:
        LOG.error(f"[{ticker}] Phase 2 quality review failed: {e}")
        import traceback
        LOG.error(traceback.format_exc())
        return None
