"""
Known Information Filter (Phase 1.5)

Filters Phase 1 bullets by checking claims against filing knowledge base (10-K, 10-Q, Transcript).
Identifies which claims are KNOWN (already in filings) vs NEW (not in filings).

STATUS: TEST MODE - Runs in parallel with Phase 2, emails findings only.
        Does NOT modify the production pipeline.

Future: Once validated, Phase 2 will receive filtered output instead of original Phase 1 JSON.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

LOG = logging.getLogger(__name__)


# =============================================================================
# PROMPT
# =============================================================================

KNOWN_INFO_FILTER_PROMPT = """You are a research analyst filtering a news summary against the company's official SEC filings.

Your task: For each bullet/paragraph, decompose into atomic claims and check each against the filings provided. Output filtered content containing ONLY genuinely NEW and material information.

═══════════════════════════════════════════════════════════════════════════════
WHAT IS "KNOWN" vs "NEW"?
═══════════════════════════════════════════════════════════════════════════════

KNOWN (filter out) - Information already in the filings OR stale information:
- Specific numbers that appear in filings (revenue, margins, EPS, guidance, capex)
- Events explicitly stated in filings (announced X, reported Y, launched Z)
- Management quotes from transcripts
- Guidance figures from earnings calls
- Risk factors already disclosed in 10-K/10-Q
- Business model descriptions from filings
- Historical comparisons already discussed (YoY, QoQ changes mentioned in filings)
- Material events disclosed in 8-K filings (mergers, acquisitions, executive changes, restructuring)
- PRIOR QUARTER DATA: Financial metrics from quarters before the current filing period
  (e.g., if current filings are Q3, then Q2 data is KNOWN even if not in Q3 filings)

NEW (keep) - Information NOT in the filings AND temporally fresh:
- Market reaction (stock price movement, trading volume changes)
- Analyst actions (upgrades, downgrades, price target changes, ratings)
- Third-party commentary (analyst quotes, industry expert opinions)
- Events occurring AFTER the latest filing date
- Rumors, speculation, breaking news not yet in filings
- Competitive developments not discussed in company filings
- External market data not from the company
- Specific competitor metrics (growth rates, market share) NOT in company filings

═══════════════════════════════════════════════════════════════════════════════
CRITICAL: PAIRED CLAIMS RULE
═══════════════════════════════════════════════════════════════════════════════

Some claims only make sense as pairs - particularly COMPARISONS. When a bullet
compares the company to competitors (e.g., "AWS at 20% vs Azure at 30%"), you
MUST treat the comparison as a unit:

- If the company's metric is KNOWN but competitor's metric is NEW, keep BOTH
- If removing one side would make the comparison meaningless, keep BOTH
- The comparison itself may be newsworthy even if individual numbers are known

Example:
"AWS growth of 20% lags Azure and Google Cloud at 30%"
- "AWS at 20%" = KNOWN (in transcript)
- "Azure/Google at 30%" = NEW (not in Amazon's filings)
- CORRECT: Keep BOTH because the comparison is the insight
- WRONG: Remove 20%, keep 30% (leaves reader without context)

When rewriting comparisons, either:
1. Keep the full comparison (if competitor data is NEW and valuable)
2. Remove the entire comparison (if it's just restating known competitive position)

═══════════════════════════════════════════════════════════════════════════════
STALENESS CHECK (Independent of Filings)
═══════════════════════════════════════════════════════════════════════════════

Even if a claim is NOT in our filings, it may still be STALE - information that
has been publicly available long enough that any attentive investor already knows it.

The key question: "When was this information RELEASED to the public?"

STEP 1: IS THIS CONTINUOUSLY AVAILABLE MARKET DATA?

These are stale when referencing historical values (>2 weeks old) because
they're observable at any time - there's no "release" moment:

A. Derived Financial Metrics (from data providers like Yahoo Finance, Morningstar)
   - TTM (trailing twelve month) calculations
   - Multi-year growth rates (5-year, 3-year, 10-year)
   - Industry averages and benchmarks
   - Valuation ratios (P/E, P/B, EV/EBITDA)
   - Historical return comparisons (YTD, 1-year, 5-year returns)

B. Commodity Prices
   - Oil (WTI, Brent), natural gas, coal
   - Metals (gold, silver, copper, aluminum)
   - Agricultural (corn, wheat, soybeans)
   - Spreads (crack spreads, refining margins, frac spreads)

C. Interest Rates & Fixed Income
   - Fed funds rate, SOFR, LIBOR
   - Treasury yields (2Y, 10Y, 30Y)
   - Credit spreads, corporate bond yields
   - Mortgage rates

D. Currency/FX Rates
   - Any historical exchange rate (USD/EUR, USD/JPY, etc.)

E. Market Indices & Levels
   - Historical index values (S&P 500, NASDAQ, Dow, sector ETFs)
   - VIX levels, market breadth metrics

F. Economic Data (when recapping old releases)
   - GDP, unemployment, CPI from prior periods being summarized
   - PMI, housing data, consumer confidence from months ago

→ If historical (>2 weeks old) → Mark as KNOWN
→ Evidence: "[Type] from [period] - continuously available market data"

EXCEPTION - When market data IS new:
- Current/real-time values tied to today's analysis → NEW
- Significant moves (>5%) tied to a catalyst within 2 weeks → NEW
- Forward-looking forecasts/futures → NEW

STEP 2: FOR DISCRETE RELEASES, WHEN WAS IT RELEASED?

For company-specific information that WAS released at a specific time:
- Quarterly earnings → Earnings announcement date
- Annual metrics (from 10-K) → 10-K filing date
- Guidance → Announcement date
- M&A, partnerships, contracts → Press release date
- Analyst ratings → Publication date of the rating

IMPORTANT: The COMPLETE earnings announcement includes ALL of the following:
- Actual results (revenue, EPS, margins)
- Comparison to guidance (beat, miss, in-line)
- Comparison to estimates (beat, miss, in-line)
- Prior guidance that was being compared against
- Surprise magnitude ("beat by 5%", "missed by $0.02")

These are ALL announced simultaneously at earnings, so they share the SAME release date.
"Beat guidance" is NOT new information discovered later - it was announced WITH the results.

Compare release date to CURRENT DATE:
- Released ≤4 weeks ago → NEW (recent, investors may not have digested)
- Released >4 weeks ago → KNOWN (stale - investors have had time to see this)

→ Evidence for stale discrete releases: "Released [date], [X] weeks stale"

═══════════════════════════════════════════════════════════════════════════════
STALENESS EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

DERIVED METRICS (always stale):

✗ "ROE of 26% for TTM to August 2025" (article from Dec 2025)
  → KNOWN | Evidence: "Derived TTM metric - continuously available"

✗ "Net income growth of 22% over five years"
  → KNOWN | Evidence: "5-year derived metric - continuously available"

✗ "10% industry average ROE"
  → KNOWN | Evidence: "Industry benchmark - static reference data"

✗ "P/E ratio of 34.5x per Yahoo Finance"
  → KNOWN | Evidence: "Valuation ratio - continuously available"

MARKET DATA (stale when old):

✗ "Crack spreads averaged $15/bbl in Q2"
  → KNOWN | Evidence: "Q2 commodity spread - 6 months stale"

✗ "10Y Treasury was 4.5% in September"
  → KNOWN | Evidence: "September interest rate - 3 months stale"

✗ "Oil averaged $78 last quarter"
  → KNOWN | Evidence: "Prior quarter oil price - continuously available"

✗ "USD/EUR was 1.08 in August"
  → KNOWN | Evidence: "August FX rate - 4 months stale"

✓ "Crack spreads collapsed to $8/bbl this week, pressuring margins"
  → NEW (current market data tied to analysis)

✓ "10Y Treasury surged 30bps today on inflation data"
  → NEW (significant recent move with catalyst)

✓ "Oil futures suggest $90 by Q1 2026"
  → NEW (forward-looking)

DISCRETE RELEASES (stale after 4 weeks):

✗ "Q3 EBITDA of $10.7B" (Q3 earnings released Sep 4, now Dec 1)
  → KNOWN | Evidence: "Released Sep 4, 2025 - 12 weeks stale"

✗ "Company announced $2B buyback" (announced 2 months ago)
  → KNOWN | Evidence: "Released Oct 1, 2025 - 8 weeks stale"

✗ "Revenue significantly beat guidance" (Q3 earnings released Oct 30, now Dec 1)
  → KNOWN | Evidence: "Guidance beat announced with Q3 results Oct 30 - 5 weeks stale"

✗ "Surpassed expectations of $47.5-50.5B" (Q3 earnings released Oct 30, now Dec 1)
  → KNOWN | Evidence: "Surprise vs guidance announced with Q3 results Oct 30 - 5 weeks stale"

✓ "Q4 revenue of $15.2B" (Q4 earnings released yesterday)
  → NEW (discrete release <4 weeks old)

✓ "Beat Q4 estimates by 8%" (Q4 earnings released yesterday)
  → NEW (guidance beat is part of the earnings release, which is <4 weeks old)

✓ "Analyst upgraded to Buy with $200 target" (issued 2 weeks ago)
  → NEW (discrete release <4 weeks old)

✓ "Stock fell 8% following guidance cut" (guidance cut last week)
  → NEW (market reaction to recent event)

IMPORTANT: A claim can be KNOWN because:
1. It appears in our filings (evidence = quote from filing), OR
2. It's stale (evidence = staleness reason)

Both result in status: "KNOWN" - the evidence field explains why.

═══════════════════════════════════════════════════════════════════════════════
CLAIM EXTRACTION
═══════════════════════════════════════════════════════════════════════════════

Decompose each bullet/paragraph into ATOMIC claims:
- Each specific number is ONE claim (e.g., "revenue $51.2B")
- Each percentage/growth rate is ONE claim (e.g., "+26% YoY")
- Each specific event is ONE claim (e.g., "announced partnership with X")
- Each directional statement is ONE claim (e.g., "beat guidance")
- Each attributed quote is ONE claim (e.g., "CEO said X")

IMPORTANT: Mark claims as PAIRED when they form a comparison:
- "Company X at 20% vs Competitor Y at 30%" → mark both as paired_with each other

Example decomposition:
"Revenue grew 26% to $51.2B, beating guidance of $47.5-50.5B, while stock fell 25%"
→ Claim 1: "revenue $51.2B"
→ Claim 2: "revenue grew 26%"
→ Claim 3: "beat guidance of $47.5-50.5B"
→ Claim 4: "stock fell 25%"

═══════════════════════════════════════════════════════════════════════════════
VERIFICATION PROCESS
═══════════════════════════════════════════════════════════════════════════════

For each claim:
1. Search Transcript for exact or paraphrased match
2. Search 10-Q for exact or paraphrased match
3. Search 10-K for exact or paraphrased match
4. Search 8-K filings for exact or paraphrased match (material events since last earnings)
5. If found in ANY filing → status: "KNOWN"
6. If NOT found in any filing → status: "NEW"

Matching rules:
- Numbers: Exact match required (allow rounding: $51.2B = $51,200M = $51.2 billion)
- Percentages: Exact match required (+26% = 26% growth = grew 26%)
- Events: Same event, even if worded differently = KNOWN
- Quotes: Same substance, even if not verbatim = KNOWN

IMPORTANT - What counts as KNOWN:
- The SPECIFIC fact must be in filings, not just the general topic
- "Competition exists" in filing does NOT make "Temu has 57% market share" KNOWN
- "We face regulatory risk" does NOT make "EU investigation in November 2025" KNOWN
- Only mark KNOWN if the specific data point, number, or fact appears in filings

═══════════════════════════════════════════════════════════════════════════════
ACTION LOGIC
═══════════════════════════════════════════════════════════════════════════════

Based on claim analysis, assign ONE action per bullet/paragraph:

KEEP - All claims are NEW
→ Pass original content unchanged
→ rewritten_content = original_content

REWRITE - Mix of KNOWN and NEW claims
→ Rewrite to include ONLY the NEW claims
→ Maintain narrative coherence (not just a list)
→ Add minimal context if needed for NEW claims to make sense
→ Preserve attribution if relevant ("per analyst", "per Reuters")

REMOVE - All claims are KNOWN
→ Nothing new to report
→ rewritten_content = "" (empty string)

═══════════════════════════════════════════════════════════════════════════════
MATERIALITY TEST FOR REWRITE vs REMOVE
═══════════════════════════════════════════════════════════════════════════════

Before marking as REWRITE, apply this test:

"Would a reader gain ACTIONABLE INSIGHT from ONLY the NEW claims?"

Mark as REMOVE (not REWRITE) if the NEW claims are merely:
- Dates or timing details on otherwise KNOWN events
- Minor wording variations of KNOWN information
- Context that only supports KNOWN claims
- Less than 20% of the original content's substance

Example - should be REMOVE, not REWRITE:
Original: "Amazon faces regulatory risks including potential EU investigations in November 2025
that could classify AWS as a gatekeeper under the DMA, leading to fines."
KNOWN: regulatory risks, EU investigations, gatekeeper classification, DMA, fines (all in 10-K)
NEW: "November 2025" (just a date)
→ Action: REMOVE (the date alone provides no actionable insight)

Example - should be REWRITE:
Original: "AWS growth of 20% lags Azure at 33% and Google Cloud at 35%, per analyst reports."
KNOWN: AWS at 20% (in transcript)
NEW: Azure at 33%, Google Cloud at 35%, analyst commentary
→ Action: REWRITE (competitor metrics ARE actionable insight)
→ Rewritten: "AWS growth lags Azure at 33% and Google Cloud at 35%, per analyst reports."
   (Keep AWS context for comparison to make sense, but note the competitor data is the news)

═══════════════════════════════════════════════════════════════════════════════
REWRITE GUIDELINES
═══════════════════════════════════════════════════════════════════════════════

When action is REWRITE:

1. REMOVE all KNOWN claims completely - do not include any information from filings
2. PRESERVE all NEW claims with their full context and specific details
3. MAINTAIN narrative coherence - write flowing prose, not a choppy list
4. ADD minimal bridging context if needed for the NEW claim to make sense
5. PRESERVE attribution for NEW claims ("per analyst", "according to Reuters")
6. For PAIRED CLAIMS (comparisons): Keep both sides if removing one makes it meaningless
7. Do NOT preserve CONCLUSIONS derived from KNOWN data
   - WRONG: "AWS growth lags competitors" (if "lags" conclusion is from known data)
   - RIGHT: Remove the lagging narrative entirely, or keep with NEW competitor numbers

CRITICAL - Verify your rewrite:
- Does the rewritten text contain ANY information from the filings? If yes, remove it.
- Does the rewritten text preserve conclusions/framing from KNOWN claims? If yes, rewrite.
- Would the rewritten text make sense to someone who hasn't read the filings? If no, add context.

Example:
Original: "Meta reported Q3 revenue of $51.2B (+26% YoY), beating guidance, while stock pulled back 25% on AI spending concerns."
KNOWN: Q3 revenue $51.2B, +26% YoY, beat guidance (all in transcript/10-Q)
NEW: stock pulled back 25%, AI spending concerns

Rewritten: "META shares pulled back 25% amid investor concerns over AI spending levels."
(Note: removed "in the past month" if that's not in the article, kept only verifiable NEW claims)

═══════════════════════════════════════════════════════════════════════════════
INPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

You will receive:
1. Phase 1 JSON with bullets and paragraphs
2. Filing sources (Transcript, 10-Q, 10-K, 8-K) - check claims against these
   - 8-K filings are material events filed since the last earnings call

BULLET SECTIONS TO FILTER (apply claim analysis):
- major_developments
- financial_performance
- risk_factors
- competitive_industry_dynamics
- key_variables

BULLET SECTIONS TO EXEMPT (pass through unchanged, action=KEEP, no claim analysis):
- wall_street_sentiment (analyst opinions ARE the news, even when citing known data)
- upcoming_catalysts (forward-looking editorial value)

For EXEMPT sections: Do NOT analyze claims. Set action="KEEP", claims=[], rewritten_content=original_content.

PARAGRAPH SECTIONS TO FILTER (apply claim analysis):
- bottom_line
- upside_scenario
- downside_scenario

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

Return valid JSON with this exact structure:

{
  "summary": {
    "total_bullets": 15,
    "total_paragraphs": 3,
    "kept": 3,
    "rewritten": 8,
    "removed": 4,
    "total_claims": 45,
    "known_claims": 28,
    "new_claims": 17
  },
  "bullets": [
    {
      "bullet_id": "FIN_001",
      "section": "financial_performance",
      "original_content": "Full original bullet text here",
      "claims": [
        {
          "claim": "Q3 revenue $51.2B",
          "status": "KNOWN",
          "source": "10-Q Financial Highlights section",
          "source_type": "10-Q",
          "evidence": "Total net sales increased 11% to $158.9 billion in Q3 2024"
        },
        {
          "claim": "stock pulled back 25%",
          "status": "NEW",
          "source": null,
          "source_type": null,
          "evidence": null
        }
      ],
      "action": "REWRITE",
      "rewritten_content": "Rewritten text with only NEW claims"
    }
  ],
  "paragraphs": [
    {
      "section": "bottom_line",
      "original_content": "Full original paragraph text here",
      "claims": [
        {
          "claim": "...",
          "status": "KNOWN" or "NEW",
          "source": "filing section" or null,
          "source_type": "Transcript" or "10-Q" or "10-K" or "8-K" or null,
          "evidence": "exact quote or paraphrase from filing" or null
        }
      ],
      "action": "KEEP" or "REWRITE" or "REMOVE",
      "rewritten_content": "..."
    }
  ]
}

IMPORTANT:
- Include ALL bullets from ALL 7 bullet sections
- Include ALL 3 paragraph sections
- Every bullet/paragraph must have an action
- For KEEP: rewritten_content = original_content (copy exactly)
- For REMOVE: rewritten_content = "" (empty string)
- For REWRITE: rewritten_content = new coherent text with only NEW claims
- For EXEMPT sections (wall_street_sentiment, upcoming_catalysts):
  Always set action="KEEP", claims=[], rewritten_content=original_content
- List ALL claims individually - NEVER truncate with "and X more claims" or similar
- NEVER summarize or abbreviate the claims array

EVIDENCE FIELD (required for KNOWN claims):
- For KNOWN claims: Include the actual quote or close paraphrase from the filing that proves
  this claim is known. This should be the specific text that matches the claim.
- For NEW claims: Set evidence to null
- Keep evidence concise (1-2 sentences max) but specific enough to verify the match
- Example: claim="AWS growth ~20%" → evidence="AWS segment revenue grew 19% year-over-year"

Return ONLY the JSON object, no other text.
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _build_filter_user_content(ticker: str, phase1_json: Dict, filings: Dict, eight_k_filings: List[Dict] = None) -> str:
    """
    Build user content combining Phase 1 JSON and filing sources.

    Args:
        ticker: Stock ticker symbol
        phase1_json: Phase 1 JSON output
        filings: Dict with keys '10k', '10q', 'transcript'
        eight_k_filings: List of filtered 8-K filings (optional)

    Returns:
        Formatted user content string
    """
    content = f"TICKER: {ticker}\n"
    content += f"CURRENT DATE: {datetime.now().strftime('%B %d, %Y')}\n\n"
    content += "=" * 80 + "\n"
    content += "PHASE 1 JSON TO FILTER:\n"
    content += "=" * 80 + "\n\n"
    content += json.dumps(phase1_json, indent=2)
    content += "\n\n"

    content += "=" * 80 + "\n"
    content += "FILING SOURCES (check claims against these):\n"
    content += "=" * 80 + "\n\n"

    # Add Transcript if available
    if 'transcript' in filings:
        t = filings['transcript']
        quarter = t.get('fiscal_quarter', 'Q?')
        year = t.get('fiscal_year', '????')
        company = t.get('company_name') or ticker
        date = t.get('date')
        date_str = date.strftime('%b %d, %Y') if date else 'Unknown Date'

        content += f"LATEST EARNINGS CALL (TRANSCRIPT):\n"
        content += f"[{ticker} ({company}) {quarter} {year} Earnings Call ({date_str})]\n\n"
        content += f"{t.get('text', '')}\n\n\n"

    # Add 10-Q if available
    if '10q' in filings:
        q = filings['10q']
        quarter = q.get('fiscal_quarter', 'Q?')
        year = q.get('fiscal_year', '????')
        company = q.get('company_name') or ticker
        date = q.get('filing_date')
        date_str = date.strftime('%b %d, %Y') if date else 'Unknown Date'

        content += f"LATEST QUARTERLY REPORT (10-Q):\n"
        content += f"[{ticker} ({company}) {quarter} {year} 10-Q Filing, Filed: {date_str}]\n\n"
        content += f"{q.get('text', '')}\n\n\n"

    # Add 10-K if available
    if '10k' in filings:
        k = filings['10k']
        year = k.get('fiscal_year', '????')
        company = k.get('company_name') or ticker
        date = k.get('filing_date')
        date_str = date.strftime('%b %d, %Y') if date else 'Unknown Date'

        content += f"COMPANY 10-K PROFILE:\n"
        content += f"[{ticker} ({company}) 10-K FILING FOR FISCAL YEAR {year}, Filed: {date_str}]\n\n"
        content += f"{k.get('text', '')}\n\n\n"

    # Add 8-K filings if available (filtered material events since last earnings)
    if eight_k_filings:
        content += f"RECENT 8-K FILINGS (since last earnings call):\n"
        content += f"[{len(eight_k_filings)} material 8-K filing(s) found]\n\n"

        for filing in eight_k_filings:
            filing_date = filing.get('filing_date')
            if hasattr(filing_date, 'strftime'):
                date_str = filing_date.strftime('%b %d, %Y')
            else:
                date_str = str(filing_date) if filing_date else 'Unknown Date'

            report_title = filing.get('report_title', 'Untitled')
            item_codes = filing.get('item_codes', 'Unknown')
            summary = filing.get('summary_markdown', '')

            content += f"[{ticker} 8-K Filed: {date_str} | Items: {item_codes}]\n"
            content += f"{report_title}\n\n"
            content += f"{summary}\n\n"
            content += "---\n\n"

    if not filings and not eight_k_filings:
        content += "NO FILINGS AVAILABLE - Mark all claims as NEW.\n"

    return content


def _get_filings_info(filings: Dict, eight_k_filings: List[Dict] = None) -> Dict:
    """Extract filing metadata for email display."""
    info = {}

    if 'transcript' in filings:
        t = filings['transcript']
        info['transcript'] = {
            'quarter': t.get('fiscal_quarter', 'Q?'),
            'year': t.get('fiscal_year', '????'),
            'date': t.get('date').strftime('%b %d, %Y') if t.get('date') else 'Unknown'
        }

    if '10q' in filings:
        q = filings['10q']
        info['10q'] = {
            'quarter': q.get('fiscal_quarter', 'Q?'),
            'year': q.get('fiscal_year', '????'),
            'date': q.get('filing_date').strftime('%b %d, %Y') if q.get('filing_date') else 'Unknown'
        }

    if '10k' in filings:
        k = filings['10k']
        info['10k'] = {
            'year': k.get('fiscal_year', '????'),
            'date': k.get('filing_date').strftime('%b %d, %Y') if k.get('filing_date') else 'Unknown'
        }

    # Add 8-K summary
    if eight_k_filings:
        info['8k'] = {
            'count': len(eight_k_filings),
            'filings': []
        }
        for filing in eight_k_filings:
            filing_date = filing.get('filing_date')
            if hasattr(filing_date, 'strftime'):
                date_str = filing_date.strftime('%b %d, %Y')
            else:
                date_str = str(filing_date) if filing_date else 'Unknown'

            info['8k']['filings'].append({
                'date': date_str,
                'title': filing.get('report_title', 'Untitled'),
                'items': filing.get('item_codes', 'Unknown')
            })

    return info


def _fetch_filtered_8k_filings(ticker: str, db_func, last_transcript_date=None) -> List[Dict]:
    """
    Fetch filtered 8-K filings for inclusion in knowledge base.

    Applies 3-layer filtering:
    - Layer 1: Item code filter (material events only)
    - Layer 2: Exhibit number filter (press releases, not legal docs)
    - Layer 3: Title keyword filter (exclude boilerplate)

    Time window:
    - Start: Last transcript date (or 90-day fallback)
    - End: T-3 (3 days before today, to allow articles to cover the 8-K first)
    - Max: 90 days

    Args:
        ticker: Stock ticker
        db_func: Database connection function
        last_transcript_date: Date of last earnings call (optional)

    Returns:
        List of filtered 8-K filings with filing_date, report_title, item_codes, summary_markdown
    """
    from datetime import date, timedelta

    try:
        with db_func() as conn, conn.cursor() as cur:
            # Calculate time window
            today = date.today()
            end_date = today - timedelta(days=3)  # T-3 buffer
            max_lookback = today - timedelta(days=90)  # 90-day safety cap

            # Start date: after last transcript, or 90-day fallback
            if last_transcript_date:
                start_date = max(last_transcript_date, max_lookback)
            else:
                start_date = max_lookback

            LOG.info(f"[{ticker}] Phase 1.5: Fetching 8-Ks from {start_date} to {end_date}")

            # Query with all filters applied in SQL
            # Uses window function to limit to 3 exhibits per filing date (ordered by exhibit_number)
            # This prevents token explosion from merger filings with 15+ exhibits (e.g., HBAN)
            cur.execute("""
                WITH filtered_8k AS (
                    SELECT
                        filing_date,
                        report_title,
                        item_codes,
                        exhibit_number,
                        summary_markdown,
                        ROW_NUMBER() OVER (
                            PARTITION BY filing_date
                            ORDER BY
                                -- Prioritize MAIN and 2.1, then 99.x by number
                                CASE
                                    WHEN exhibit_number = 'MAIN' THEN 0
                                    WHEN exhibit_number = '2.1' THEN 1
                                    ELSE 2
                                END,
                                exhibit_number ASC
                        ) as rn
                    FROM company_releases
                    WHERE ticker = %s
                      AND source_type = '8k_exhibit'
                      -- Time window
                      AND filing_date > %s
                      AND filing_date <= %s
                      -- Layer 1: Item code filter (include if ANY of these material codes)
                      AND (
                          item_codes LIKE '%%1.01%%' OR
                          item_codes LIKE '%%1.02%%' OR
                          item_codes LIKE '%%2.01%%' OR
                          item_codes LIKE '%%2.02%%' OR
                          item_codes LIKE '%%2.03%%' OR
                          item_codes LIKE '%%2.05%%' OR
                          item_codes LIKE '%%2.06%%' OR
                          item_codes LIKE '%%4.01%%' OR
                          item_codes LIKE '%%4.02%%' OR
                          item_codes LIKE '%%5.01%%' OR
                          item_codes LIKE '%%5.02%%' OR
                          item_codes LIKE '%%5.07%%' OR
                          item_codes LIKE '%%7.01%%' OR
                          item_codes LIKE '%%8.01%%' OR
                          item_codes = 'Unknown'
                      )
                      -- Layer 2: Exhibit number filter (press releases + main body + merger agreements)
                      AND (
                          exhibit_number LIKE '99%%' OR
                          exhibit_number = 'MAIN' OR
                          exhibit_number = '2.1'
                      )
                      -- Layer 3: Title exclusions (legal boilerplate + routine dividends)
                      AND report_title NOT ILIKE '%%Legal Opinion%%'
                      AND report_title NOT ILIKE '%%Underwriting Agreement%%'
                      AND report_title NOT ILIKE '%%Indenture%%'
                      AND report_title NOT ILIKE '%%Officers'' Certificate%%'
                      AND report_title NOT ILIKE '%%Notes Due%%'
                      AND report_title NOT ILIKE '%%Bylaws%%'
                      AND report_title NOT ILIKE '%%Dividend%%'
                )
                SELECT filing_date, report_title, item_codes, summary_markdown
                FROM filtered_8k
                WHERE rn <= 3  -- Max 3 exhibits per filing date
                ORDER BY filing_date DESC, rn ASC
            """, (ticker, start_date, end_date))

            rows = cur.fetchall()

            filings = []
            for row in rows:
                filings.append({
                    'filing_date': row['filing_date'] if isinstance(row, dict) else row[0],
                    'report_title': row['report_title'] if isinstance(row, dict) else row[1],
                    'item_codes': row['item_codes'] if isinstance(row, dict) else row[2],
                    'summary_markdown': row['summary_markdown'] if isinstance(row, dict) else row[3]
                })

            LOG.info(f"[{ticker}] Phase 1.5: Found {len(filings)} filtered 8-K filings (max 3 per filing date)")
            return filings

    except Exception as e:
        LOG.error(f"[{ticker}] Phase 1.5: Error fetching 8-K filings: {e}")
        return []


# =============================================================================
# GEMINI IMPLEMENTATION
# =============================================================================

def _filter_known_info_gemini(
    ticker: str,
    phase1_json: Dict,
    filings: Dict,
    gemini_api_key: str,
    eight_k_filings: List[Dict] = None
) -> Optional[Dict]:
    """
    Filter known information using Gemini 2.5 Flash.

    Args:
        ticker: Stock ticker
        phase1_json: Phase 1 JSON output
        filings: Dict with filing data
        gemini_api_key: Gemini API key
        eight_k_filings: List of filtered 8-K filings (optional)

    Returns:
        Filter result dict or None if failed
    """
    import google.generativeai as genai

    try:
        genai.configure(api_key=gemini_api_key)

        # Build user content (now includes 8-K filings)
        user_content = _build_filter_user_content(ticker, phase1_json, filings, eight_k_filings)

        # Concatenate system prompt + user content (matches working pattern in article_summaries.py, triage.py)
        # NOTE: Using system_instruction parameter caused empty responses with finish_reason=1
        full_prompt = KNOWN_INFO_FILTER_PROMPT + "\n\n" + user_content

        # Log sizes
        system_tokens_est = len(KNOWN_INFO_FILTER_PROMPT) // 4
        user_tokens_est = len(user_content) // 4
        total_tokens_est = len(full_prompt) // 4
        LOG.info(f"[{ticker}] Phase 1.5 Gemini prompt: system=~{system_tokens_est} tokens, user=~{user_tokens_est} tokens, total=~{total_tokens_est} tokens")

        # Create model WITHOUT system_instruction (use concatenated prompt instead)
        model = genai.GenerativeModel('gemini-2.5-flash')

        LOG.info(f"[{ticker}] Phase 1.5: Calling Gemini 2.5 Flash for known info filter")

        # Import JSON parser
        from modules.json_utils import extract_json_from_claude_response

        # Outer loop: Retry on truncated/malformed responses (content validation)
        max_content_retries = 1

        for content_attempt in range(max_content_retries + 1):
            # Inner loop: Retry on API errors (rate limits, timeouts, etc.)
            max_api_retries = 2
            response = None
            generation_time_ms = 0

            for api_attempt in range(max_api_retries + 1):
                try:
                    start_time = time.time()
                    response = model.generate_content(
                        full_prompt,
                        generation_config={
                            'temperature': 0.0,
                            'max_output_tokens': 40000
                        }
                    )
                    generation_time_ms = int((time.time() - start_time) * 1000)
                    break

                except Exception as e:
                    error_str = str(e)
                    is_retryable = (
                        'ResourceExhausted' in error_str or
                        'quota' in error_str.lower() or
                        '429' in error_str or
                        'ServiceUnavailable' in error_str or
                        '503' in error_str or
                        'DeadlineExceeded' in error_str or
                        'timeout' in error_str.lower()
                    )

                    if is_retryable and api_attempt < max_api_retries:
                        wait_time = 2 ** api_attempt
                        LOG.warning(f"[{ticker}] Phase 1.5 Gemini API error (attempt {api_attempt + 1}): {error_str[:200]}")
                        LOG.warning(f"[{ticker}] Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        LOG.error(f"[{ticker}] Phase 1.5 Gemini failed after {api_attempt + 1} API attempts: {error_str}")
                        return None

            if response is None:
                LOG.error(f"[{ticker}] Phase 1.5: No response from Gemini")
                return None

            # Extract text
            response_text = response.text
            if not response_text or len(response_text.strip()) < 10:
                LOG.error(f"[{ticker}] Phase 1.5: Gemini returned empty response")
                return None

            # Parse JSON
            json_output = extract_json_from_claude_response(response_text, ticker)

            if json_output:
                # Success! Get token counts and return
                prompt_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
                completion_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0

                LOG.info(f"[{ticker}] Phase 1.5 Gemini success: {prompt_tokens} prompt, {completion_tokens} completion, {generation_time_ms}ms")

                return {
                    "json_output": json_output,
                    "model_used": "gemini-2.5-flash",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "generation_time_ms": generation_time_ms
                }

            # JSON parsing failed - check if we should retry
            if content_attempt < max_content_retries:
                # Detect truncation: response ends mid-sentence (no closing brace, ends with comma, etc.)
                response_ending = response_text.strip()[-50:] if len(response_text.strip()) > 50 else response_text.strip()
                is_truncated = (
                    not response_text.strip().endswith('}') or
                    response_ending.endswith(',') or
                    response_ending.endswith(':')
                )

                if is_truncated:
                    LOG.warning(f"[{ticker}] Phase 1.5: Gemini response appears truncated (ends with: ...{response_ending[-30:]})")
                else:
                    LOG.warning(f"[{ticker}] Phase 1.5: JSON parsing failed (response not obviously truncated)")

                LOG.warning(f"[{ticker}] Phase 1.5: Retrying Gemini (attempt {content_attempt + 2} of {max_content_retries + 1})...")
                time.sleep(2)  # Brief pause before retry
                continue
            else:
                LOG.error(f"[{ticker}] Phase 1.5: Failed to parse Gemini JSON after {content_attempt + 1} content attempts")
                return None

        # Should not reach here, but safety return
        LOG.error(f"[{ticker}] Phase 1.5: Unexpected exit from retry loop")
        return None

    except Exception as e:
        LOG.error(f"[{ticker}] Phase 1.5 Gemini exception: {e}", exc_info=True)
        return None


# =============================================================================
# CLAUDE FALLBACK IMPLEMENTATION
# =============================================================================

def _filter_known_info_claude(
    ticker: str,
    phase1_json: Dict,
    filings: Dict,
    anthropic_api_key: str,
    eight_k_filings: List[Dict] = None
) -> Optional[Dict]:
    """
    Filter known information using Claude Sonnet 4.5 (fallback).

    Args:
        ticker: Stock ticker
        phase1_json: Phase 1 JSON output
        filings: Dict with filing data
        anthropic_api_key: Anthropic API key
        eight_k_filings: List of filtered 8-K filings (optional)

    Returns:
        Filter result dict or None if failed
    """
    try:
        # Build user content (now includes 8-K filings)
        user_content = _build_filter_user_content(ticker, phase1_json, filings, eight_k_filings)

        # Log sizes
        system_tokens_est = len(KNOWN_INFO_FILTER_PROMPT) // 4
        user_tokens_est = len(user_content) // 4
        LOG.info(f"[{ticker}] Phase 1.5 Claude prompt: system=~{system_tokens_est} tokens, user=~{user_tokens_est} tokens")

        headers = {
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        data = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 40000,
            "temperature": 0.0,
            "system": [
                {
                    "type": "text",
                    "text": KNOWN_INFO_FILTER_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }

        LOG.info(f"[{ticker}] Phase 1.5: Calling Claude Sonnet 4.5 for known info filter (fallback)")

        # Retry logic
        max_retries = 2
        response = None
        generation_time_ms = 0

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=180
                )
                generation_time_ms = int((time.time() - start_time) * 1000)

                if response.status_code == 200:
                    break

                if response.status_code in [429, 500, 503] and attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] Phase 1.5 Claude error {response.status_code} (attempt {attempt + 1})")
                    LOG.warning(f"[{ticker}] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                break

            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] Phase 1.5 Claude timeout (attempt {attempt + 1}), retrying...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] Phase 1.5 Claude timeout after {max_retries + 1} attempts")
                    return None

            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] Phase 1.5 Claude network error (attempt {attempt + 1}): {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] Phase 1.5 Claude network error after {max_retries + 1} attempts: {e}")
                    return None

        if response is None:
            LOG.error(f"[{ticker}] Phase 1.5: No response from Claude")
            return None

        if response.status_code != 200:
            LOG.error(f"[{ticker}] Phase 1.5 Claude error {response.status_code}: {response.text[:500]}")
            return None

        # Parse response
        result = response.json()
        content = result.get("content", [{}])[0].get("text", "")

        if not content or len(content.strip()) < 10:
            LOG.error(f"[{ticker}] Phase 1.5: Claude returned empty response")
            return None

        # Parse JSON
        from modules.json_utils import extract_json_from_claude_response
        json_output = extract_json_from_claude_response(content, ticker)

        if not json_output:
            LOG.error(f"[{ticker}] Phase 1.5: Failed to parse Claude JSON response")
            return None

        # Get token counts
        usage = result.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        LOG.info(f"[{ticker}] Phase 1.5 Claude success: {prompt_tokens} prompt, {completion_tokens} completion, {generation_time_ms}ms")

        return {
            "json_output": json_output,
            "model_used": "claude-sonnet-4-5-20250929",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generation_time_ms": generation_time_ms
        }

    except Exception as e:
        LOG.error(f"[{ticker}] Phase 1.5 Claude exception: {e}", exc_info=True)
        return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def filter_known_information(
    ticker: str,
    phase1_json: Dict,
    db_func,
    gemini_api_key: str = None,
    anthropic_api_key: str = None
) -> Optional[Dict]:
    """
    Filter Phase 1 bullets to identify KNOWN vs NEW claims.

    TEST MODE: This function runs in parallel with Phase 2 and emails findings.
               It does NOT modify the production pipeline.

    Args:
        ticker: Stock ticker
        phase1_json: Phase 1 JSON output (unmodified)
        db_func: Database connection function
        gemini_api_key: Gemini API key (primary)
        anthropic_api_key: Anthropic API key (fallback)

    Returns:
        {
            "ticker": str,
            "timestamp": str,
            "filings_used": {...},
            "summary": {...},
            "bullets": [...],
            "paragraphs": [...],
            "model_used": str,
            "generation_time_ms": int
        }
        Or None if failed
    """
    LOG.info(f"[{ticker}] Phase 1.5: Starting known information filter (TEST MODE)")

    # Fetch filings using Phase 2's function
    try:
        from modules.executive_summary_phase2 import _fetch_available_filings
        filings = _fetch_available_filings(ticker, db_func)

        filing_count = len(filings)
        filing_types = list(filings.keys())
        LOG.info(f"[{ticker}] Phase 1.5: Loaded {filing_count} filings: {filing_types}")

    except Exception as e:
        LOG.error(f"[{ticker}] Phase 1.5: Failed to fetch filings: {e}")
        filings = {}

    # Get last transcript date for 8-K time window
    last_transcript_date = None
    if 'transcript' in filings and filings['transcript'].get('date'):
        last_transcript_date = filings['transcript']['date']
        LOG.info(f"[{ticker}] Phase 1.5: Last transcript date: {last_transcript_date}")

    # Fetch filtered 8-K filings (material events since last earnings)
    eight_k_filings = _fetch_filtered_8k_filings(ticker, db_func, last_transcript_date)

    # Get filings info for email (now includes 8-K)
    filings_info = _get_filings_info(filings, eight_k_filings)

    # Try Gemini first
    result = None
    if gemini_api_key:
        LOG.info(f"[{ticker}] Phase 1.5: Attempting Gemini 2.5 Flash (primary)")
        result = _filter_known_info_gemini(ticker, phase1_json, filings, gemini_api_key, eight_k_filings)

        if result and result.get("json_output"):
            LOG.info(f"[{ticker}] Phase 1.5: Gemini succeeded")
        else:
            LOG.warning(f"[{ticker}] Phase 1.5: Gemini failed, falling back to Claude")
            result = None

    # Claude fallback
    if result is None and anthropic_api_key:
        LOG.info(f"[{ticker}] Phase 1.5: Using Claude Sonnet 4.5 (fallback)")
        result = _filter_known_info_claude(ticker, phase1_json, filings, anthropic_api_key, eight_k_filings)

        if result and result.get("json_output"):
            LOG.info(f"[{ticker}] Phase 1.5: Claude succeeded (fallback)")
        else:
            LOG.error(f"[{ticker}] Phase 1.5: Claude also failed")
            result = None

    if result is None:
        LOG.error(f"[{ticker}] Phase 1.5: Both Gemini and Claude failed")
        return None

    # Build final output
    json_output = result["json_output"]

    return {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "filings_used": filings_info,
        "summary": json_output.get("summary", {}),
        "bullets": json_output.get("bullets", []),
        "paragraphs": json_output.get("paragraphs", []),
        "model_used": result.get("model_used", "unknown"),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "generation_time_ms": result.get("generation_time_ms", 0)
    }


# =============================================================================
# EMAIL HTML GENERATOR
# =============================================================================

def generate_known_info_filter_email(ticker: str, filter_result: Dict) -> str:
    """
    Generate simple HTML email showing filter results.

    Args:
        ticker: Stock ticker
        filter_result: Output from filter_known_information()

    Returns:
        HTML string for email body
    """
    if not filter_result:
        return f"<html><body><h2>Phase 1.5 Filter Failed for {ticker}</h2><p>No results available.</p></body></html>"

    summary = filter_result.get("summary", {})
    bullets = filter_result.get("bullets", [])
    paragraphs = filter_result.get("paragraphs", [])
    filings = filter_result.get("filings_used", {})
    model = filter_result.get("model_used", "unknown")
    gen_time = filter_result.get("generation_time_ms", 0)

    # Build filing list string
    filing_parts = []
    if 'transcript' in filings:
        t = filings['transcript']
        filing_parts.append(f"Transcript ({t['quarter']} {t['year']})")
    if '10q' in filings:
        q = filings['10q']
        filing_parts.append(f"10-Q ({q['quarter']} {q['year']})")
    if '10k' in filings:
        k = filings['10k']
        filing_parts.append(f"10-K (FY{k['year']})")
    if '8k' in filings:
        eight_k = filings['8k']
        filing_parts.append(f"8-K ({eight_k['count']} filing{'s' if eight_k['count'] != 1 else ''})")
    filing_str = ", ".join(filing_parts) if filing_parts else "None"

    # Start HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f9f9f9; }}
.header {{ background: #1a1a2e; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
.header h2 {{ margin: 0 0 10px 0; }}
.header p {{ margin: 0; opacity: 0.8; font-size: 14px; }}
.summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin-top: 15px; }}
.stat {{ text-align: center; padding: 10px; background: #f5f5f5; border-radius: 6px; }}
.stat-value {{ font-size: 24px; font-weight: bold; color: #1a1a2e; }}
.stat-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
.bullet {{ background: white; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #ddd; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.bullet-keep {{ border-left-color: #28a745; }}
.bullet-rewrite {{ border-left-color: #ffc107; }}
.bullet-remove {{ border-left-color: #dc3545; }}
.bullet-header {{ font-weight: bold; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }}
.action-badge {{ padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
.badge-keep {{ background: #d4edda; color: #155724; }}
.badge-rewrite {{ background: #fff3cd; color: #856404; }}
.badge-remove {{ background: #f8d7da; color: #721c24; }}
.content-box {{ background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0; font-size: 14px; }}
.claims {{ margin: 10px 0; }}
.claim {{ padding: 5px 0; font-size: 13px; }}
.claim-known {{ color: #dc3545; }}
.claim-new {{ color: #28a745; }}
.rewritten {{ background: #e8f5e9; padding: 10px; border-radius: 4px; margin-top: 10px; }}
.section-header {{ background: #e9ecef; padding: 10px 15px; margin: 20px 0 10px 0; border-radius: 6px; font-weight: bold; }}
</style>
</head>
<body>

<div class="header">
<h2>Phase 1.5: Known Info Filter - {ticker}</h2>
<p>{filter_result.get('timestamp', '')[:19]} | Model: {model} | {gen_time}ms</p>
<p>Filings: {filing_str}</p>
</div>

<div class="summary">
<strong>Summary</strong>
<div class="summary-grid">
<div class="stat">
<div class="stat-value">{summary.get('total_bullets', 0) + summary.get('total_paragraphs', 0)}</div>
<div class="stat-label">Total Items</div>
</div>
<div class="stat">
<div class="stat-value" style="color: #28a745;">{summary.get('kept', 0)}</div>
<div class="stat-label">Kept</div>
</div>
<div class="stat">
<div class="stat-value" style="color: #ffc107;">{summary.get('rewritten', 0)}</div>
<div class="stat-label">Rewritten</div>
</div>
<div class="stat">
<div class="stat-value" style="color: #dc3545;">{summary.get('removed', 0)}</div>
<div class="stat-label">Removed</div>
</div>
<div class="stat">
<div class="stat-value">{summary.get('known_claims', 0)}</div>
<div class="stat-label">Known Claims</div>
</div>
<div class="stat">
<div class="stat-value">{summary.get('new_claims', 0)}</div>
<div class="stat-label">New Claims</div>
</div>
</div>
</div>
"""

    # Add paragraphs section
    if paragraphs:
        html += '<div class="section-header">Paragraph Sections</div>\n'
        for p in paragraphs:
            action = p.get('action', 'KEEP').upper()
            action_class = f"bullet-{action.lower()}"
            badge_class = f"badge-{action.lower()}"

            html += f'<div class="bullet {action_class}">\n'
            html += f'<div class="bullet-header"><span>[{p.get("section", "?")}]</span><span class="action-badge {badge_class}">{action}</span></div>\n'

            # Original content (no truncation)
            original = p.get('original_content', '')
            html += f'<div class="content-box"><strong>Original:</strong><br>{_escape_html(original)}</div>\n'

            # Claims (no truncation)
            claims = p.get('claims', [])
            if claims:
                html += '<div class="claims"><strong>Claims:</strong><br>\n'
                for c in claims:
                    status = c.get('status', 'NEW')
                    claim_class = 'claim-known' if status == 'KNOWN' else 'claim-new'
                    icon = '❌' if status == 'KNOWN' else '✅'
                    source_type = c.get('source_type', '')
                    source_section = c.get('source', '')
                    evidence = c.get('evidence', '')

                    # Build the claim line
                    html += f'<div class="claim {claim_class}">{icon} {status}: {_escape_html(c.get("claim", ""))}'

                    # For KNOWN claims, show evidence with source
                    if status == 'KNOWN' and evidence:
                        if source_type:
                            # Filing-based KNOWN - show quote with source
                            html += f'<br><span style="margin-left: 20px; color: #666; font-size: 12px;">→ "{_escape_html(evidence)}" ({source_type}, {source_section})</span>'
                        else:
                            # Staleness-based KNOWN - show reason without quotes
                            html += f'<br><span style="margin-left: 20px; color: #856404; font-size: 12px;">→ ⏰ {_escape_html(evidence)}</span>'
                    elif status == 'KNOWN' and source_section:
                        html += f'<br><span style="margin-left: 20px; color: #666; font-size: 12px;">→ {source_type}, {source_section}</span>'

                    html += '</div>\n'
                html += '</div>\n'

            # Rewritten content (no truncation)
            if action == 'REWRITE':
                rewritten = p.get('rewritten_content', '')
                if rewritten:
                    html += f'<div class="rewritten"><strong>Rewritten:</strong><br>{_escape_html(rewritten)}</div>\n'

            html += '</div>\n'

    # Add bullets section
    if bullets:
        html += '<div class="section-header">Bullet Sections</div>\n'
        for b in bullets:
            action = b.get('action', 'KEEP').upper()
            action_class = f"bullet-{action.lower()}"
            badge_class = f"badge-{action.lower()}"

            html += f'<div class="bullet {action_class}">\n'
            html += f'<div class="bullet-header"><span>[{b.get("bullet_id", "?")}] {b.get("section", "")}</span><span class="action-badge {badge_class}">{action}</span></div>\n'

            # Original content (no truncation)
            original = b.get('original_content', '')
            html += f'<div class="content-box"><strong>Original:</strong><br>{_escape_html(original)}</div>\n'

            # Claims (no truncation)
            claims = b.get('claims', [])
            if claims:
                html += '<div class="claims"><strong>Claims:</strong><br>\n'
                for c in claims:
                    status = c.get('status', 'NEW')
                    claim_class = 'claim-known' if status == 'KNOWN' else 'claim-new'
                    icon = '❌' if status == 'KNOWN' else '✅'
                    source_type = c.get('source_type', '')
                    source_section = c.get('source', '')
                    evidence = c.get('evidence', '')

                    # Build the claim line
                    html += f'<div class="claim {claim_class}">{icon} {status}: {_escape_html(c.get("claim", ""))}'

                    # For KNOWN claims, show evidence with source
                    if status == 'KNOWN' and evidence:
                        if source_type:
                            # Filing-based KNOWN - show quote with source
                            html += f'<br><span style="margin-left: 20px; color: #666; font-size: 12px;">→ "{_escape_html(evidence)}" ({source_type}, {source_section})</span>'
                        else:
                            # Staleness-based KNOWN - show reason without quotes
                            html += f'<br><span style="margin-left: 20px; color: #856404; font-size: 12px;">→ ⏰ {_escape_html(evidence)}</span>'
                    elif status == 'KNOWN' and source_section:
                        html += f'<br><span style="margin-left: 20px; color: #666; font-size: 12px;">→ {source_type}, {source_section}</span>'

                    html += '</div>\n'
                html += '</div>\n'

            # Rewritten content (no truncation)
            if action == 'REWRITE':
                rewritten = b.get('rewritten_content', '')
                if rewritten:
                    html += f'<div class="rewritten"><strong>Rewritten:</strong><br>{_escape_html(rewritten)}</div>\n'

            html += '</div>\n'

    html += """
</body>
</html>
"""

    return html


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))
