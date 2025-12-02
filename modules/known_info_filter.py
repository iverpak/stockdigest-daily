"""
Known Information Filter (Phase 1.5)

Filters Phase 1 bullets by checking claims against filing knowledge base (Transcript, 8-K).
Identifies which claims are KNOWN (already in filings or stale) vs NEW (fresh information).

This is a STALENESS FILTER - removes content that provides zero incremental value over
what an investor learned from recent filings. NOT a comprehensive fact-checker.

Knowledge Base: Transcript + 8-Ks only
- 10-K excluded (causes false matches with generic risk categories)
- 10-Q excluded (Dec 2025) - same issue, Item 1A boilerplate matches specific news

3-Step Flow:
1. Sentence-level tagging (Gemini Flash) - Tag each sentence KNOWN or NEW
2. Mechanical triage:
   - 100% KNOWN → REMOVE (skip LLM)
   - 100% NEW → KEEP as-is (skip LLM)
   - Mixed → proceed to step 3
3. LLM Rewrite (Sonnet 4.5, only for mixed bullets)
   - Surgically remove KNOWN content, preserve NEW
   - Restructure for coherence
   - Return REWRITE or REMOVE

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

KNOWN_INFO_FILTER_PROMPT = """You are a research analyst filtering a news summary for an institutional investor who has ALREADY read the latest earnings call and recent 8-K filings.

═══════════════════════════════════════════════════════════════════════════════
GUIDING PRINCIPLE
═══════════════════════════════════════════════════════════════════════════════

Your goal is NOT "remove anything that appears in filings."
Your goal IS "remove content that provides zero incremental value over what the investor learned from recent filings (earnings call and any 8-Ks since then)."

This is a STALENESS FILTER, not a comprehensive fact-checker.

PRIMARY TARGETS (remove these):
- Obvious recaps of prior earnings (Q3 results rehashed 8 weeks later)
- Articles rehashing what management said on the call without new developments
- Stale market data anyone can look up (TTM metrics, historical prices, P/E ratios)
- Prior quarter data being presented as if it's news

PRESERVE (even if mentioned in filings):
- Connective context that makes NEW claims comprehensible
- Ongoing developments (situation still evolving, new data points)
- Causal explanations (why something matters to the company)
- Known themes when they frame genuinely NEW data

The test: "If an investor read the recent filings, would this sentence tell them something they don't already know?"

If YES → KEEP (even if some context overlaps with filings)
If NO → REMOVE (pure rehash, no incremental value)

═══════════════════════════════════════════════════════════════════════════════
ANALYSIS FLOW (Follow This Exactly)
═══════════════════════════════════════════════════════════════════════════════

For EACH bullet/paragraph, follow these steps IN ORDER:

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PARSE INTO SENTENCES                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Split the text into individual sentences.                                   │
│ Each sentence becomes an entry in the "sentences" array.                    │
│ Even single-sentence bullets get a "sentences" array with one entry.        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: EXTRACT CLAIMS FROM EACH SENTENCE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ For each sentence, decompose into atomic claims:                            │
│ - Numbers: "revenue $15.9B", "up 22%", "$150 price target"                  │
│ - Events: "announced partnership", "upgraded to Buy"                        │
│ - Quotes: "CEO said X", "per analyst reports"                               │
│ Each claim becomes an entry in that sentence's "claims" array.              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: CLASSIFY EACH CLAIM AS KNOWN OR NEW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ For each claim:                                                             │
│ - Search filings for match → KNOWN (set source_type + evidence)             │
│ - Check staleness rules → KNOWN (set evidence = staleness reason)           │
│ - Not found and not stale → NEW (set source_type=null, evidence=null)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: SENTENCE VERDICT                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Look at all claims in the sentence:                                         │
│ - ANY claim is NEW and MATERIAL? → has_material_new=true, sentence_action=KEEP│
│ - ALL claims are KNOWN/stale?    → has_material_new=false, sentence_action=REMOVE│
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: BULLET/PARAGRAPH VERDICT                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Look at all sentence verdicts:                                              │
│ - ANY sentence is KEEP? → action=KEEP                                       │
│ - ALL sentences are REMOVE? → action=REMOVE                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: BUILD filtered_content                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Concatenate all KEEP sentences (space-separated, preserve punctuation):     │
│ - If 2 KEEP sentences: "Sentence one. Sentence two."                        │
│ - If 0 KEEP sentences: "" (empty string)                                    │
│ This is mechanical - no rewriting, no editing, just concatenation.          │
└─────────────────────────────────────────────────────────────────────────────┘

WHY SENTENCE-LEVEL? This approach:
- Preserves context (known facts sharing a sentence with new facts stay together)
- Avoids surgical editing failures (no word-level rewriting)
- Maintains coherence (sentences are complete grammatical units)
- Keeps paired claims together (comparisons stay intact if any part is new)

═══════════════════════════════════════════════════════════════════════════════
WHAT IS "KNOWN" vs "NEW"?
═══════════════════════════════════════════════════════════════════════════════

KNOWN (filter out) - Information already in the filings OR stale information:
- Specific numbers that appear in filings (revenue, margins, EPS, guidance, capex)
- Events explicitly stated in filings (announced X, reported Y, launched Z)
- Management quotes from transcripts
- Guidance figures from earnings calls
- Historical comparisons already discussed (YoY, QoQ changes mentioned in filings)
- Material events disclosed in 8-K filings (mergers, acquisitions, executive changes, restructuring)
- PRIOR QUARTER DATA: Financial metrics from quarters before the current filing period
  (e.g., if current filings are Q3, then Q2 data is KNOWN even if not in Q3 filings)

NEW (keep) - Information NOT in the filings AND temporally fresh:
- Analyst actions (upgrades, downgrades, price target changes, ratings)
- Third-party commentary (analyst quotes, industry expert opinions)
- Events occurring AFTER the latest filing date
- Rumors, speculation, breaking news not yet in filings
- Competitive developments not discussed in company filings
- External market data not from the company
- Specific competitor metrics (growth rates, market share) NOT in company filings

═══════════════════════════════════════════════════════════════════════════════
THE SPECIFICITY TEST (Apply to EVERY claim)
═══════════════════════════════════════════════════════════════════════════════

Before marking ANY claim as KNOWN, ask:

"Does the filing contain THIS EXACT fact - same event, same data, same source, same timing?"

If YES → KNOWN (with evidence quote)
If NO, only topic overlap → NEW

TOPIC OVERLAP IS NOT ENOUGH.

Examples of topic overlap that is NOT a match:
- Filing discusses "governance" ≠ Article reports "bylaw amendment on Nov 25"
- Filing mentions "consumer risk" ≠ Article cites "Fed data showing traffic down 5%"
- Filing mentions "competition" ≠ Article reports "competitor acquired XYZ yesterday"
- Filing mentions "litigation risk" ≠ Article reports "court ruled against company Tuesday"
- Filing mentions "regulatory exposure" ≠ Article reports "EU fined company €2B today"
- Filing says "CEO views macro as cautious" ≠ Article says "Fed Beige Book shows 9/12 districts negative"

The filing describes the LANDSCAPE at filing time. Articles report EVENTS and DATA within that landscape.
New events, new external data, and new developments are NEW - even in known topic areas.

KEY INSIGHT: Independent external data (Fed reports, court rulings, regulatory actions, analyst research)
that validates or quantifies a known risk is STILL NEW. It carries different epistemic weight than
the company's own disclosure of the risk.

═══════════════════════════════════════════════════════════════════════════════
RISK FACTOR CLAIMS - ALWAYS NEW
═══════════════════════════════════════════════════════════════════════════════

Claims describing risks, headwinds, or concerns should NOT be marked KNOWN just because
the 10-Q or filings mention similar risk categories. SEC filings (especially Item 1A Risk
Factors) contain comprehensive boilerplate that lists every conceivable risk. Matching
article content against this boilerplate creates false positives.

Risk-related claims are ALWAYS NEW unless:
- The EXACT same risk EVENT is described (e.g., "lawsuit filed in Delaware" appears in both)
- The specific QUANTIFICATION matches (e.g., "$500M exposure" in both)

Generic risk CATEGORIES are NOT matches:
❌ Filing says "competition risk" ≠ Article says "BofA warns about Marvell competition"
❌ Filing says "integration risk" ≠ Article says "VMware integration challenges cited by analyst"
❌ Filing says "customer concentration" ≠ Article says "Morgan Stanley notes Meta deferral risk"
❌ Filing says "semiconductor cycle exposure" ≠ Article discusses "exposure to semiconductor cycles"
❌ Filing says "significant debt" ≠ Article says "analysts cite net debt load as concern"

WHY: The filing discloses that risks EXIST (legal requirement). Articles and analysts discuss
which risks are SALIENT NOW and carry editorial weight. An analyst's risk assessment is news;
a company's boilerplate disclosure is not.

═══════════════════════════════════════════════════════════════════════════════
STALENESS CHECK (Independent of Filings)
═══════════════════════════════════════════════════════════════════════════════

Even if a claim is NOT in our filings, it may still be STALE - information that
has been publicly available long enough that any attentive investor already knows it.

The key question: "When was this information RELEASED to the public?"

STEP 1: IS THIS CONTINUOUSLY AVAILABLE MARKET DATA?

These are ALWAYS stale because they're observable at any time from public data
providers - there's no "release" moment. Investors can look them up themselves.

A. Stock Prices & Trading Data (ALWAYS stale)
   - Current stock price, premarket/afterhours prices
   - Daily/weekly price movements (up X%, down Y%)
   - Trading volume
   - 52-week highs/lows
   - Intraday moves
   → Evidence: "Stock price data - continuously observable"

B. Valuation Ratios & Financial Multiples (ALWAYS stale)
   - P/E ratio (current or forward)
   - P/B, P/S, EV/EBITDA, EV/Revenue
   - PEG ratio, dividend yield
   - Enterprise value, market cap
   → Evidence: "Valuation multiple - continuously observable"

C. Derived Financial Metrics (ALWAYS stale)
   - TTM (trailing twelve month) calculations
   - Multi-year growth rates (5-year, 3-year, 10-year)
   - Industry averages and benchmarks
   - Historical return comparisons (YTD, 1-year, 5-year returns)
   - ROE, ROA, ROIC calculations
   → Evidence: "Derived metric - continuously observable"

D. Commodity Prices (stale when >2 weeks old)
   - Oil (WTI, Brent), natural gas, coal
   - Metals (gold, silver, copper, aluminum)
   - Agricultural (corn, wheat, soybeans)
   - Spreads (crack spreads, refining margins, frac spreads)

E. Interest Rates & Fixed Income (stale when >2 weeks old)
   - Fed funds rate, SOFR, LIBOR
   - Treasury yields (2Y, 10Y, 30Y)
   - Credit spreads, corporate bond yields
   - Mortgage rates

F. Currency/FX Rates (stale when >2 weeks old)
   - Any historical exchange rate (USD/EUR, USD/JPY, etc.)

G. Market Indices & Levels (stale when >2 weeks old)
   - Historical index values (S&P 500, NASDAQ, Dow, sector ETFs)
   - VIX levels, market breadth metrics

H. Economic Data (stale when recapping old releases)
   - GDP, unemployment, CPI from prior periods being summarized
   - PMI, housing data, consumer confidence from months ago

→ Categories A, B, C: ALWAYS mark as KNOWN (no timing exception)
→ Categories D-H: Mark as KNOWN if >2 weeks old
→ Evidence: "[Type] - continuously available market data"

EXCEPTION - When market data IS new (applies to D-H only, NOT A-B-C):
- Forward-looking forecasts/futures → NEW

STEP 2: FOR DISCRETE RELEASES, WHEN WAS IT RELEASED?

CRITICAL: Staleness ONLY applies to BACKWARD-LOOKING claims about PAST events.
It does NOT apply to forward-looking or present-tense information.

NEVER mark as stale (regardless of when announced):
- Future events: expected close dates, scheduled conferences, planned actions
- Forward guidance: revenue targets, margin expectations, growth projections
- Regulatory timelines: expected approvals, review periods, compliance deadlines
- Current conditions: ongoing macro environment, present-tense market analysis
- Real-time data: current probabilities, today's expectations, live market sentiment

These describe the present or future - they cannot be "stale" because they're still unfolding.

CAN be stale (after 1 week):
- Historical results: Q3 revenue, past EPS, prior quarter margins
- Past events: announcements, completed transactions, historical price moves
- Backward-looking comparisons: beat/miss vs guidance (announced with results)

For stale-eligible claims, compare release date to CURRENT DATE:
- Released ≤1 week ago → NEW
- Released >1 week ago → KNOWN (stale)

IMPORTANT: The COMPLETE earnings announcement includes actual results AND the beat/miss
comparison. "Beat guidance" is announced WITH the results, so it shares the same release date.

→ Evidence for stale discrete releases: "Released [date], [X] weeks stale"

═══════════════════════════════════════════════════════════════════════════════
STALENESS EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

ALWAYS STALE (Categories A-C):

✗ "Stock has gained 67% year-to-date" → KNOWN | "Stock price data - continuously observable"
✗ "P/E ratio of 34.5x" → KNOWN | "Valuation multiple - continuously observable"
✗ "ROE of 26% for TTM" → KNOWN | "Derived metric - continuously observable"

STALE WHEN >2 WEEKS OLD (Categories D-H):

✗ "Oil averaged $78 last quarter" → KNOWN | "Prior quarter oil price - continuously available"
✓ "Oil futures suggest $90 by Q1 2026" → NEW (forward-looking exception)

DISCRETE RELEASES (stale after 1 week - BACKWARD-LOOKING ONLY):

✗ "Q3 EBITDA of $10.7B" (released Sep 4, now Dec 1) → KNOWN | "Released Sep 4 - 12 weeks stale"
✓ "Q4 revenue of $15.2B" (released yesterday) → NEW (discrete release <1 week old)

FORWARD-LOOKING (NEVER stale, regardless of announcement date):

✓ "Deal expected to close H2 2026" (announced 6 weeks ago) → NEW (future event)
✓ "Company expects 15% revenue growth in FY2026" (guidance 8 weeks ago) → NEW (forward guidance)
✓ "FDA approval expected by Q2 2026" (announced 3 months ago) → NEW (regulatory timeline)
✓ "Macro environment is pressuring deposit yields" → NEW (present-tense analysis)

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
2. Search 8-K filings for exact or paraphrased match (material events since last earnings)
3. If found in ANY filing → status: "KNOWN"
4. If NOT found in any filing → check STALENESS rules
5. If stale → status: "KNOWN" (with staleness evidence)
6. If fresh and not in filings → status: "NEW"

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
FILING IDENTIFIERS (CRITICAL)
═══════════════════════════════════════════════════════════════════════════════

Each filing in the input is labeled with a unique identifier:
- TRANSCRIPT_1 (for transcript)
- 8K_1, 8K_2, 8K_3 (for multiple 8-K filings)

When a claim is KNOWN (from filings), you MUST use the EXACT filing identifier in source_type.

Examples:
- Claim found in transcript → source_type: "TRANSCRIPT_1"
- Claim found in 8-K #1 → source_type: "8K_1"
- Claim found in first 8-K → source_type: "8K_1"
- Claim found in second 8-K → source_type: "8K_2"
- Claim is stale (not in filings) → source_type: null, evidence: "[staleness reason]"

DO NOT use generic labels like "8-K" or "Transcript" - use the specific identifier.
This allows us to display exactly which filing (with date and title) contained the claim.

═══════════════════════════════════════════════════════════════════════════════
ACTION LOGIC (SENTENCE-LEVEL)
═══════════════════════════════════════════════════════════════════════════════

For each bullet/paragraph:

1. Parse into sentences
2. For each sentence:
   - Extract claims
   - Classify each claim as KNOWN or NEW
   - If ANY claim in the sentence is NEW and material → sentence_action = KEEP
   - If ALL claims in the sentence are KNOWN or stale → sentence_action = REMOVE

3. Concatenate all KEEP sentences to form filtered_content

4. Determine bullet/paragraph action:
   - If filtered_content is empty (all sentences removed) → action = REMOVE
   - If filtered_content has content → action = KEEP
   - filtered_content = the concatenated KEEP sentences

There is NO "REWRITE" action. We do not surgically edit within sentences.
Either a sentence is kept whole, or removed whole.

═══════════════════════════════════════════════════════════════════════════════
MATERIALITY TEST (For Sentence Verdicts)
═══════════════════════════════════════════════════════════════════════════════

A NEW claim is MATERIAL (triggers KEEP for the sentence) if it provides information
an investor would update beliefs from or act on.

MATERIAL NEW claims (keep the sentence):
- Stock price movements
- Analyst actions (upgrades, downgrades, price targets, ratings)
- Competitor developments (acquisitions, partnerships, metrics)
- Events occurring after the latest filing date
- External quantitative data not from the company
- Third-party commentary or expert opinions

NOT MATERIAL (do not save the sentence by themselves):
- Dates alone ("on Tuesday", "in November")
- Source attributions alone ("per Reuters", "according to Bloomberg")
- Rewordings of KNOWN facts with no new substance
- Generic descriptors ("the tech giant", "the leading utility")

If a sentence's ONLY new content is non-material, treat the sentence as REMOVE.

Example - sentence should be REMOVED:
"Amazon faces regulatory risks including potential EU investigations in November 2025."
KNOWN: regulatory risks, EU investigations (from filings)
NEW: "November 2025" (just a date - not material)
→ Sentence verdict: REMOVE (the date alone provides no actionable insight)

Example - sentence should be KEPT:
"AWS growth of 20% lags Azure at 33% and Google Cloud at 35%, per analyst reports."
KNOWN: AWS at 20% (in transcript)
NEW: Azure at 33%, Google Cloud at 35% (competitor metrics - MATERIAL)
→ Sentence verdict: KEEP entire sentence (competitor data is actionable)

═══════════════════════════════════════════════════════════════════════════════
INPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

You will receive:
1. Phase 1 JSON with bullets and paragraphs
2. Filing sources (Transcript, 8-K) - check claims against these
   - 8-K filings are material events filed since the last earnings call

BULLET SECTIONS TO FILTER (apply full sentence-level filtering):
- major_developments
- financial_performance
- risk_factors
- competitive_industry_dynamics

BULLET SECTIONS TO EXEMPT (analyze for transparency, but ALWAYS keep original):
- wall_street_sentiment (analyst opinions ARE the news)
- upcoming_catalysts (forward-looking editorial value)
- key_variables (monitoring recommendations, not news claims)

For EXEMPT sections:
- DO perform claim extraction and sentence analysis (for transparency/QA visibility)
- DO output the sentence-level structure with claims
- But ALWAYS set action="KEEP" regardless of findings
- Set filtered_content="" (we'll use original from input)
- Add "exempt": true to the output

This allows QA review of what WOULD have been filtered, without actually filtering.

PARAGRAPH SECTIONS TO FILTER (apply full sentence-level filtering):
- bottom_line
- upside_scenario
- downside_scenario

═══════════════════════════════════════════════════════════════════════════════
CRITICAL REQUIREMENTS (READ CAREFULLY)
═══════════════════════════════════════════════════════════════════════════════

You MUST follow these requirements - no shortcuts allowed:

1. ALWAYS ANALYZE EVERY BULLET/PARAGRAPH
   - Even if you will ultimately REMOVE a bullet, you MUST show the full analysis
   - Empty "sentences" array is NEVER acceptable
   - We need to see WHY something was removed

2. ALWAYS PARSE INTO SENTENCES
   - Every bullet/paragraph must be split into individual sentences
   - Each sentence gets its own entry in the "sentences" array
   - Single-sentence bullets still have a "sentences" array with one entry

3. ALWAYS EXTRACT CLAIMS FROM EACH SENTENCE
   - Every sentence must have a "claims" array
   - Decompose into atomic claims (numbers, events, quotes, etc.)
   - Even sentences with just one claim need the array

4. ALWAYS PROVIDE EVIDENCE FOR KNOWN CLAIMS
   - source_type: The filing identifier (TRANSCRIPT_1, 8K_1, 8K_2, etc.)
   - evidence: The quote or paraphrase from the filing
   - For staleness: source_type=null, evidence="[staleness reason]"

5. THE OUTPUT MUST SHOW THE COMPLETE CHAIN OF LOGIC
   - claims → sentence verdicts → bullet verdict → filtered_content
   - A reader should be able to follow exactly why each decision was made

═══════════════════════════════════════════════════════════════════════════════
WORKED EXAMPLES (Study These Carefully)
═══════════════════════════════════════════════════════════════════════════════

EXAMPLE 1: Bullet with MIXED content → action=KEEP

Original bullet: "Revenue grew 22% to $15.9B in Q3. The stock fell 8% on AI spending concerns."

Analysis:
├─ Sentence 1: "Revenue grew 22% to $15.9B in Q3."
│  ├─ Claim: "revenue $15.9B" → KNOWN (TRANSCRIPT_1: "Q3 revenue of $15.9 billion")
│  ├─ Claim: "grew 22%" → KNOWN (TRANSCRIPT_1: "revenue increased 22% YoY")
│  ├─ has_material_new: false (0 NEW claims)
│  └─ sentence_action: REMOVE
│
├─ Sentence 2: "The stock fell 8% on AI spending concerns."
│  ├─ Claim: "stock fell 8%" → NEW (market reaction, not in filings)
│  ├─ Claim: "AI spending concerns" → NEW (investor sentiment, not in filings)
│  ├─ has_material_new: true (2 NEW claims)
│  └─ sentence_action: KEEP
│
├─ Bullet verdict: 1 KEEP sentence → action = KEEP
└─ filtered_content: "The stock fell 8% on AI spending concerns."

JSON output for this bullet:
{
  "bullet_id": "FIN_001",
  "section": "financial_performance",
  "sentences": [
    {
      "text": "Revenue grew 22% to $15.9B in Q3.",
      "claims": [
        {"claim": "revenue $15.9B", "status": "KNOWN", "source_type": "TRANSCRIPT_1", "evidence": "Q3 revenue of $15.9 billion"},
        {"claim": "grew 22%", "status": "KNOWN", "source_type": "TRANSCRIPT_1", "evidence": "revenue increased 22% YoY"}
      ],
      "has_material_new": false,
      "sentence_action": "REMOVE"
    },
    {
      "text": "The stock fell 8% on AI spending concerns.",
      "claims": [
        {"claim": "stock fell 8%", "status": "NEW", "source_type": null, "evidence": null},
        {"claim": "AI spending concerns", "status": "NEW", "source_type": null, "evidence": null}
      ],
      "has_material_new": true,
      "sentence_action": "KEEP"
    }
  ],
  "action": "KEEP",
  "filtered_content": "The stock fell 8% on AI spending concerns."
}

EXAMPLE 2: Bullet that is FULLY REMOVED → action=REMOVE (STILL requires full analysis!)

Original bullet: "Q3 EBITDA reached $10.7B, up 15% YoY, beating guidance of $10.2B."

Analysis:
├─ Sentence 1: "Q3 EBITDA reached $10.7B, up 15% YoY, beating guidance of $10.2B."
│  ├─ Claim: "EBITDA $10.7B" → KNOWN (TRANSCRIPT_1: "EBITDA of $10.7 billion")
│  ├─ Claim: "up 15% YoY" → KNOWN (TRANSCRIPT_1: "EBITDA grew 15% year-over-year")
│  ├─ Claim: "beating guidance of $10.2B" → KNOWN (staleness: "Released Oct 30, 5 weeks stale")
│  ├─ has_material_new: false (0 NEW claims)
│  └─ sentence_action: REMOVE
│
├─ Bullet verdict: 0 KEEP sentences → action = REMOVE
└─ filtered_content: ""

JSON output for this bullet (NOTE: sentences array is REQUIRED even for REMOVE):
{
  "bullet_id": "FIN_002",
  "section": "financial_performance",
  "sentences": [
    {
      "text": "Q3 EBITDA reached $10.7B, up 15% YoY, beating guidance of $10.2B.",
      "claims": [
        {"claim": "EBITDA $10.7B", "status": "KNOWN", "source_type": "TRANSCRIPT_1", "evidence": "EBITDA of $10.7 billion"},
        {"claim": "up 15% YoY", "status": "KNOWN", "source_type": "TRANSCRIPT_1", "evidence": "EBITDA grew 15% year-over-year"},
        {"claim": "beating guidance of $10.2B", "status": "KNOWN", "source_type": null, "evidence": "Released Oct 30, 5 weeks stale"}
      ],
      "has_material_new": false,
      "sentence_action": "REMOVE"
    }
  ],
  "action": "REMOVE",
  "filtered_content": ""
}

EXAMPLE 3: EXEMPT section (wall_street_sentiment) → action=KEEP, show analysis anyway

Original bullet: "Morgan Stanley upgraded to Buy with $150 price target, citing strong Q3 results."

Analysis (for transparency only - exempt sections always KEEP):
├─ Sentence 1: "Morgan Stanley upgraded to Buy with $150 price target, citing strong Q3 results."
│  ├─ Claim: "Morgan Stanley upgraded to Buy" → NEW (analyst action)
│  ├─ Claim: "$150 price target" → NEW (analyst target)
│  ├─ Claim: "citing strong Q3 results" → KNOWN (TRANSCRIPT_1: Q3 results discussed)
│  ├─ has_material_new: true
│  └─ sentence_action: KEEP (but irrelevant - section is exempt)
│
├─ Bullet verdict: EXEMPT → action = KEEP (regardless of analysis)
└─ filtered_content: "" (we use original from input)

JSON output for exempt bullet:
{
  "bullet_id": "WSS_001",
  "section": "wall_street_sentiment",
  "sentences": [
    {
      "text": "Morgan Stanley upgraded to Buy with $150 price target, citing strong Q3 results.",
      "claims": [
        {"claim": "Morgan Stanley upgraded to Buy", "status": "NEW", "source_type": null, "evidence": null},
        {"claim": "$150 price target", "status": "NEW", "source_type": null, "evidence": null},
        {"claim": "citing strong Q3 results", "status": "KNOWN", "source_type": "TRANSCRIPT_1", "evidence": "Q3 results discussed in earnings call"}
      ],
      "has_material_new": true,
      "sentence_action": "KEEP"
    }
  ],
  "action": "KEEP",
  "filtered_content": "",
  "exempt": true
}

═══════════════════════════════════════════════════════════════════════════════
WHY FULL ANALYSIS IS ALWAYS REQUIRED
═══════════════════════════════════════════════════════════════════════════════

Even when a bullet will be REMOVED, we need the full analysis because:

1. QA VISIBILITY: Humans review these results. We need to see WHY something was removed.
   - Which claims were in the filings?
   - What evidence matched?
   - Was it staleness or filing match?

2. DEBUGGING: If the filter makes a mistake, we need the chain of logic to find it.

3. AUDIT TRAIL: The analysis proves the decision was made systematically, not arbitrarily.

4. NO SHORTCUTS: If you skip analysis for REMOVE bullets, you might be wrong about the
   verdict. The analysis IS the verification.

WRONG (never do this):
{
  "bullet_id": "FIN_002",
  "sentences": [],           ← EMPTY! NO ANALYSIS!
  "action": "REMOVE",
  "filtered_content": ""
}

RIGHT (always do this):
{
  "bullet_id": "FIN_002",
  "sentences": [
    {
      "text": "Q3 EBITDA reached $10.7B...",
      "claims": [
        {"claim": "EBITDA $10.7B", "status": "KNOWN", "source_type": "TRANSCRIPT_1", "evidence": "..."},
        ...
      ],
      "has_material_new": false,
      "sentence_action": "REMOVE"
    }
  ],
  "action": "REMOVE",
  "filtered_content": ""
}

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

Return valid JSON with this exact structure:

{
  "summary": {
    "total_bullets": 15,
    "total_paragraphs": 3,
    "kept": 8,
    "removed": 7,
    "total_sentences": 45,
    "kept_sentences": 28,
    "removed_sentences": 17,
    "total_claims": 67,
    "known_claims": 40,
    "new_claims": 27
  },
  "bullets": [
    {
      "bullet_id": "FIN_001",
      "section": "financial_performance",
      "sentences": [
        {
          "text": "Revenue grew 22% to $15.9B in Q3.",
          "claims": [
            {
              "claim": "revenue $15.9B",
              "status": "KNOWN",
              "source_type": "TRANSCRIPT_1",
              "evidence": "Q3 revenue of $15.9 billion"
            },
            {
              "claim": "grew 22%",
              "status": "KNOWN",
              "source_type": "TRANSCRIPT_1",
              "evidence": "revenue increased 22% year-over-year"
            }
          ],
          "has_material_new": false,
          "sentence_action": "REMOVE"
        },
        {
          "text": "The stock fell 8% on AI spending concerns.",
          "claims": [
            {
              "claim": "stock fell 8%",
              "status": "NEW",
              "source_type": null,
              "evidence": null
            },
            {
              "claim": "AI spending concerns",
              "status": "NEW",
              "source_type": null,
              "evidence": null
            }
          ],
          "has_material_new": true,
          "sentence_action": "KEEP"
        }
      ],
      "action": "KEEP",
      "filtered_content": "The stock fell 8% on AI spending concerns."
    }
  ],
  "paragraphs": [
    {
      "section": "bottom_line",
      "sentences": [
        {
          "text": "...",
          "claims": [...],
          "has_material_new": true or false,
          "sentence_action": "KEEP" or "REMOVE"
        }
      ],
      "action": "KEEP" or "REMOVE",
      "filtered_content": "Concatenated KEEP sentences..."
    }
  ]
}

IMPORTANT - DO NOT SKIP ANY OF THESE:

1. NEVER return an empty "sentences" array
   - Even for action="REMOVE", you MUST include the sentences array with full analysis
   - We need to see the claims and evidence for WHY it was removed
   - See EXAMPLE 2 above - REMOVE bullets still have complete sentence/claim analysis

2. Parse each bullet/paragraph into sentences FIRST
   - Include ALL sentences with their claims and verdicts
   - Every sentence needs: text, claims[], has_material_new, sentence_action

3. filtered_content rules:
   - = concatenation of all KEEP sentences (space-separated)
   - If all sentences removed → filtered_content = "", action = "REMOVE"
   - If any sentences kept → action = "KEEP"

4. There is NO "REWRITE" action - only KEEP or REMOVE

5. Only include bullets that exist in Phase 1 JSON
   - If a section (e.g., major_developments) has no bullets in Phase 1, do NOT create placeholder entries
   - Do NOT report on empty sections - simply skip them
   - The summary counts should only reflect bullets/paragraphs that actually had content

6. Only include paragraph sections that have content in Phase 1 JSON
   - If bottom_line, upside_scenario, or downside_scenario has no content, skip it

7. For EXEMPT sections (wall_street_sentiment, upcoming_catalysts, key_variables):
   - DO perform sentence/claim analysis (for QA visibility)
   - Set action="KEEP", filtered_content="" (we use original)
   - Add "exempt": true to the output

8. List ALL claims individually - NEVER truncate with "and X more claims"

EVIDENCE FIELD (required for KNOWN claims):
- For KNOWN claims from filings: Include quote/paraphrase + source_type
- For KNOWN claims from staleness: Set source_type=null, evidence="[staleness reason]"
- For NEW claims: Set source_type=null, evidence=null
- Keep evidence concise (1-2 sentences max)

Return ONLY the JSON object, no other text.
"""


# =============================================================================
# SONNET REWRITE PROMPT (Step 3 - Mixed content only)
# =============================================================================

SONNET_REWRITE_PROMPT = """You are a surgical text editor. Your ONLY job is to DELETE specific sentences and GLUE the remaining text for coherence.

## YOUR TASK

1. DELETE the sentences listed under "SENTENCES TO DELETE" - remove them entirely
2. GLUE the remaining text:
   - Add minimal transition words if needed (e.g., "Additionally", "Meanwhile")
   - Fix capitalization if a sentence now starts the paragraph
   - Smooth any awkward transitions caused by deletion

## STRICT RULES - DO NOT VIOLATE

❌ DO NOT change ANY claims, facts, or numbers in the remaining sentences
❌ DO NOT remove ANY attributions (e.g., "per reports", "according to analysts", "per company announcements")
❌ DO NOT add new information
❌ DO NOT rephrase, summarize, or "improve" the remaining sentences
❌ DO NOT remove any sentence not in the deletion list

✅ The remaining sentences must be IDENTICAL to the original, except for:
   - Minor transition words for flow
   - Capitalization fixes (if sentence now starts the paragraph)
   - Punctuation adjustments at deletion boundaries

## SPECIAL CASE: REMOVE ACTION

If after deleting the specified sentences, the remaining content is NOT substantive, return REMOVE.

NOT SUBSTANTIVE (return REMOVE):
- Only attributions remain ("per Reuters", "according to Bloomberg")
- Only dates/times remain ("on Tuesday", "in November")
- Only entity names without facts
- Only generic descriptors ("the regional bank")

SUBSTANTIVE (return REWRITE with glued content):
- Specific numbers, metrics, or data points
- Named analyst actions (upgrades, downgrades, price targets)
- New events or developments
- Third-party commentary with specific claims

## OUTPUT FORMAT (JSON only)

{
  "action": "REWRITE" or "REMOVE",
  "content": "text with deletions applied and minimal glue"
}

If action is REMOVE, content should be empty string.
Return ONLY the JSON object, no other text.
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _build_filter_user_content(ticker: str, phase1_json: Dict, filings: Dict, eight_k_filings: List[Dict] = None) -> Tuple[str, Dict]:
    """
    Build user content combining Phase 1 JSON and filing sources.

    Each filing is labeled with a unique identifier (e.g., TRANSCRIPT_1, 8K_1, 8K_2)
    that the AI should use in source_type. This allows us to map back to specific
    filing metadata (date, title) in post-processing.

    NOTE: 10-K is intentionally excluded from the knowledge base (causes false matches
    with generic risk categories).

    Args:
        ticker: Stock ticker symbol
        phase1_json: Phase 1 JSON output
        filings: Dict with keys '10q', 'transcript' (10-K excluded)
        eight_k_filings: List of filtered 8-K filings (optional)

    Returns:
        Tuple of (formatted user content string, filing_lookup dict)
        filing_lookup maps identifier -> {date, title, quarter, year, etc.}
    """
    filing_lookup = {}  # Maps identifier -> metadata for post-processing

    from datetime import date as date_type, timedelta

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

    # T-7 buffer: Only include filings older than 7 days
    # This gives articles time to react before we mark their content as "known"
    today = date_type.today()
    t7_cutoff = today - timedelta(days=7)

    # Add Transcript if available AND older than T-7
    if 'transcript' in filings:
        t = filings['transcript']
        quarter = t.get('fiscal_quarter', 'Q?')
        year = t.get('fiscal_year', '????')
        company = t.get('company_name') or ticker
        filing_date = t.get('date')
        date_str = filing_date.strftime('%b %d, %Y') if filing_date else 'Unknown Date'

        # Check T-7 buffer
        filing_date_only = filing_date.date() if hasattr(filing_date, 'date') else filing_date
        if filing_date_only and filing_date_only > t7_cutoff:
            LOG.info(f"[{ticker}] Phase 1.5: Transcript excluded (T-7 buffer - filed {date_str})")
        else:
            filing_id = "TRANSCRIPT_1"
            filing_lookup[filing_id] = {
                'type': 'Transcript',
                'date': date_str,
                'quarter': quarter,
                'year': year,
                'display': f"{quarter} {year} Earnings Call ({date_str})"
            }

            content += f"=== {filing_id}: LATEST EARNINGS CALL (TRANSCRIPT) ===\n"
            content += f"[{ticker} ({company}) {quarter} {year} Earnings Call ({date_str})]\n"
            content += f"Use source_type=\"{filing_id}\" when citing this filing.\n\n"
            content += f"{t.get('text', '')}\n\n\n"

    # NOTE: 10-Q intentionally excluded from knowledge base (Dec 2025)
    # 10-Q causes false positive matches (Item 1A risk factors = boilerplate matching specific news)
    # Same issue as 10-K - SEC risk disclosures are comprehensive but generic
    # Knowledge base: Transcript + 8-Ks only
    #
    # if '10q' in filings:
    #     q = filings['10q']
    #     quarter = q.get('fiscal_quarter', 'Q?')
    #     year = q.get('fiscal_year', '????')
    #     company = q.get('company_name') or ticker
    #     filing_date = q.get('filing_date')
    #     date_str = filing_date.strftime('%b %d, %Y') if filing_date else 'Unknown Date'
    #
    #     # Check T-7 buffer
    #     filing_date_only = filing_date.date() if hasattr(filing_date, 'date') else filing_date
    #     if filing_date_only and filing_date_only > t7_cutoff:
    #         LOG.info(f"[{ticker}] Phase 1.5: 10-Q excluded (T-7 buffer - filed {date_str})")
    #     else:
    #         filing_id = "10Q_1"
    #         filing_lookup[filing_id] = {
    #             'type': '10-Q',
    #             'date': date_str,
    #             'quarter': quarter,
    #             'year': year,
    #             'display': f"{quarter} {year} 10-Q (filed {date_str})"
    #         }
    #
    #         content += f"=== {filing_id}: LATEST QUARTERLY REPORT (10-Q) ===\n"
    #         content += f"[{ticker} ({company}) {quarter} {year} 10-Q Filing, Filed: {date_str}]\n"
    #         content += f"Use source_type=\"{filing_id}\" when citing this filing.\n\n"
    #         content += f"{q.get('text', '')}\n\n\n"

    # NOTE: 10-K also intentionally excluded from knowledge base
    # 10-K causes false positive matches (generic risk categories matching specific news events)

    # Add 8-K filings if available (filtered material events since last earnings)
    if eight_k_filings:
        content += f"=== RECENT 8-K FILINGS (since last earnings call) ===\n"
        content += f"[{len(eight_k_filings)} material 8-K filing(s) found]\n\n"

        for i, filing in enumerate(eight_k_filings, start=1):
            filing_date = filing.get('filing_date')
            if hasattr(filing_date, 'strftime'):
                date_str = filing_date.strftime('%b %d, %Y')
            else:
                date_str = str(filing_date) if filing_date else 'Unknown Date'

            report_title = filing.get('report_title', 'Untitled')
            item_codes = filing.get('item_codes', 'Unknown')
            summary = filing.get('summary_markdown', '')

            # Truncate title for display (keep first 50 chars)
            short_title = report_title[:50] + '...' if len(report_title) > 50 else report_title

            filing_id = f"8K_{i}"
            filing_lookup[filing_id] = {
                'type': '8-K',
                'date': date_str,
                'title': report_title,
                'short_title': short_title,
                'item_codes': item_codes,
                'display': f"8-K filed {date_str}: {short_title}"
            }

            content += f"--- {filing_id}: 8-K Filed {date_str} ---\n"
            content += f"Title: {report_title}\n"
            content += f"Items: {item_codes}\n"
            content += f"Use source_type=\"{filing_id}\" when citing this filing.\n\n"
            content += f"{summary}\n\n"

    if not filings and not eight_k_filings:
        content += "NO FILINGS AVAILABLE - Mark all claims as NEW.\n"

    return content, filing_lookup


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

    # NOTE: 10-Q intentionally excluded from knowledge base (Dec 2025)
    # 10-Q causes false positive matches (Item 1A risk factors = boilerplate matching specific news)
    # Knowledge base: Transcript + 8-Ks only

    # NOTE: 10-K also intentionally excluded from knowledge base (causes false matches)

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


def _merge_original_content(ai_response: Dict, phase1_json: Dict) -> Dict:
    """
    Merge original_content from Phase 1 JSON into AI response.

    The AI doesn't need to echo back original content - we already have it.
    This function restores it from the source for display in emails.

    Phase 1 JSON structure:
    {
      "sections": {
        "major_developments": [  # Bullet sections are arrays of bullet objects
          {"bullet_id": "...", "content": "...", ...},
        ],
        "bottom_line": {  # Paragraph sections are objects with content
          "content": "...",
        }
      }
    }

    AI Response structure (sentence-level):
    {
      "bullets": [
        {
          "bullet_id": "...",
          "sentences": [...],  # Sentence-level analysis
          "action": "KEEP" or "REMOVE",
          "filtered_content": "..."  # Concatenated KEEP sentences
        }
      ],
      "paragraphs": [...]
    }

    Args:
        ai_response: Parsed JSON from AI (sentence-level structure)
        phase1_json: Original Phase 1 JSON that was sent to the filter

    Returns:
        ai_response with original_content fields populated
    """
    sections = phase1_json.get('sections', {})

    # Bullet sections are arrays directly under sections
    bullet_section_names = [
        'major_developments', 'financial_performance', 'risk_factors',
        'wall_street_sentiment', 'competitive_industry_dynamics',
        'upcoming_catalysts', 'key_variables'
    ]

    # Build lookup for Phase 1 bullets by bullet_id
    phase1_bullets = {}
    for section_name in bullet_section_names:
        section_data = sections.get(section_name, [])
        if isinstance(section_data, list):
            for bullet in section_data:
                if isinstance(bullet, dict):
                    bullet_id = bullet.get('bullet_id', '')
                    if bullet_id:
                        # Store content - could be 'content' or 'content_integrated'
                        content = bullet.get('content_integrated') or bullet.get('content', '')
                        phase1_bullets[bullet_id] = content

    # Paragraph sections are objects with 'content' directly
    paragraph_section_names = ['bottom_line', 'upside_scenario', 'downside_scenario']

    # Build lookup for Phase 1 paragraphs by section name
    phase1_paragraphs = {}
    for section_name in paragraph_section_names:
        section_data = sections.get(section_name, {})
        if isinstance(section_data, dict):
            content = section_data.get('content_integrated') or section_data.get('content', '')
            phase1_paragraphs[section_name] = content

    # Merge into AI response bullets
    for bullet in ai_response.get('bullets', []):
        bullet_id = bullet.get('bullet_id', '')
        if bullet_id and bullet_id in phase1_bullets:
            bullet['original_content'] = phase1_bullets[bullet_id]
        elif not bullet.get('original_content'):
            bullet['original_content'] = ''

    # Merge into AI response paragraphs
    for paragraph in ai_response.get('paragraphs', []):
        section = paragraph.get('section', '')
        if section and section in phase1_paragraphs:
            paragraph['original_content'] = phase1_paragraphs[section]
        elif not paragraph.get('original_content'):
            paragraph['original_content'] = ''

    return ai_response


def apply_filter_to_phase1(phase1_json: Dict, filter_result: Dict) -> Dict:
    """
    Apply Phase 1.5 filter results to Phase 1 JSON.

    This modifies the Phase 1 JSON to remove KNOWN information before Phase 2 enrichment.

    Three-step filtering actions:
    - REMOVE: Delete bullet entirely OR clear paragraph content (100% KNOWN)
    - KEEP: Use filtered_content if available, otherwise original (100% NEW or exempt)
    - REWRITE: Use rewritten_content from LLM (mixed NEW+KNOWN, surgically edited)

    Exempt sections (action=KEEP, exempt=true) always use original content unchanged.

    Args:
        phase1_json: Original Phase 1 JSON with structure:
            {
                "sections": {
                    "major_developments": [...],  # Array of bullet objects
                    "bottom_line": {...},         # Paragraph object
                    ...
                }
            }
        filter_result: Output from filter_known_information() with structure:
            {
                "bullets": [{"bullet_id": "...", "action": "KEEP|REMOVE|REWRITE",
                            "filtered_content": "...", "rewritten_content": "...", "exempt": true/false}],
                "paragraphs": [{"section": "...", "action": "KEEP|REMOVE|REWRITE",
                               "filtered_content": "...", "rewritten_content": "...", "exempt": true/false}]
            }

    Returns:
        Modified Phase 1 JSON ready for Phase 2 (deep copy, original unchanged)
    """
    import copy

    # Deep copy to avoid modifying original
    result = copy.deepcopy(phase1_json)
    sections = result.get('sections', {})

    # Build lookup for filter actions by bullet_id
    bullet_actions = {}
    for bullet in filter_result.get('bullets', []):
        bullet_id = bullet.get('bullet_id', '')
        if bullet_id:
            bullet_actions[bullet_id] = {
                'action': bullet.get('action', 'KEEP').upper(),
                'filtered_content': bullet.get('filtered_content', ''),
                'rewritten_content': bullet.get('rewritten_content', ''),  # For REWRITE action
                'exempt': bullet.get('exempt', False)
            }

    # Build lookup for filter actions by section name (paragraphs)
    paragraph_actions = {}
    for para in filter_result.get('paragraphs', []):
        section = para.get('section', '')
        if section:
            paragraph_actions[section] = {
                'action': para.get('action', 'KEEP').upper(),
                'filtered_content': para.get('filtered_content', ''),
                'rewritten_content': para.get('rewritten_content', ''),  # For REWRITE action
                'exempt': para.get('exempt', False)
            }

    # Process bullet sections
    bullet_section_names = [
        'major_developments', 'financial_performance', 'risk_factors',
        'wall_street_sentiment', 'competitive_industry_dynamics',
        'upcoming_catalysts', 'key_variables'
    ]

    removed_count = 0
    filtered_count = 0
    rewrite_count = 0

    for section_name in bullet_section_names:
        section_data = sections.get(section_name, [])
        if not isinstance(section_data, list):
            continue

        # Filter bullets - keep only those not marked REMOVE
        new_bullets = []
        for bullet in section_data:
            if not isinstance(bullet, dict):
                new_bullets.append(bullet)
                continue

            bullet_id = bullet.get('bullet_id', '')
            if bullet_id not in bullet_actions:
                # No filter action for this bullet - keep unchanged
                new_bullets.append(bullet)
                continue

            action_info = bullet_actions[bullet_id]
            action = action_info['action']
            exempt = action_info['exempt']

            if action == 'REMOVE':
                # Skip this bullet entirely
                removed_count += 1
                continue
            elif action == 'REWRITE':
                # Use LLM-rewritten content (from mixed classification)
                rewritten_content = action_info['rewritten_content']
                if rewritten_content:
                    # Update the content field with rewritten content
                    if 'content_integrated' in bullet:
                        bullet['content_integrated'] = rewritten_content
                    else:
                        bullet['content'] = rewritten_content
                    rewrite_count += 1
                new_bullets.append(bullet)
            elif action == 'KEEP':
                if exempt:
                    # Exempt section - use original unchanged
                    new_bullets.append(bullet)
                else:
                    # Non-exempt KEEP - use filtered_content if available
                    filtered_content = action_info['filtered_content']
                    if filtered_content:
                        # Update the content field with filtered content
                        if 'content_integrated' in bullet:
                            bullet['content_integrated'] = filtered_content
                        else:
                            bullet['content'] = filtered_content
                        filtered_count += 1
                    new_bullets.append(bullet)
            else:
                # Unknown action - keep unchanged for safety
                new_bullets.append(bullet)

        sections[section_name] = new_bullets

    # Process paragraph sections
    paragraph_section_names = ['bottom_line', 'upside_scenario', 'downside_scenario']

    for section_name in paragraph_section_names:
        section_data = sections.get(section_name, {})
        if not isinstance(section_data, dict):
            continue

        if section_name not in paragraph_actions:
            # No filter action for this paragraph - keep unchanged
            continue

        action_info = paragraph_actions[section_name]
        action = action_info['action']
        exempt = action_info['exempt']

        if action == 'REMOVE':
            # For paragraphs, we don't remove entirely - we clear the content
            # Phase 2 and Email generation can handle empty content gracefully
            if 'content_integrated' in section_data:
                section_data['content_integrated'] = ''
            else:
                section_data['content'] = ''
            removed_count += 1
        elif action == 'REWRITE':
            # Use LLM-rewritten content (from mixed classification)
            rewritten_content = action_info['rewritten_content']
            if rewritten_content:
                if 'content_integrated' in section_data:
                    section_data['content_integrated'] = rewritten_content
                else:
                    section_data['content'] = rewritten_content
                rewrite_count += 1
        elif action == 'KEEP':
            if not exempt:
                # Non-exempt KEEP - use filtered_content if available
                filtered_content = action_info['filtered_content']
                if filtered_content:
                    if 'content_integrated' in section_data:
                        section_data['content_integrated'] = filtered_content
                    else:
                        section_data['content'] = filtered_content
                    filtered_count += 1
            # Exempt section - leave unchanged
        # Unknown action - leave unchanged

    LOG.info(f"Phase 1.5 apply_filter: {removed_count} removed, {rewrite_count} rewritten, {filtered_count} filtered")

    return result


def _fetch_filtered_8k_filings(ticker: str, db_func, last_transcript_date=None) -> List[Dict]:
    """
    Fetch filtered 8-K filings for inclusion in knowledge base.

    Applies 3-layer filtering:
    - Layer 1: Item code filter (material events only)
    - Layer 2: Exhibit number filter (press releases, not legal docs)
    - Layer 3: Title keyword filter (exclude boilerplate)

    Time window:
    - Start: Last transcript date (or 90-day fallback)
    - End: T-7 (7 days before today, to allow articles to cover the 8-K first)
    - Max: 90 days

    Note: T-7 ensures weekly reports (7-day lookback) don't filter articles covering recent 8-Ks.

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
            end_date = today - timedelta(days=7)  # T-7 buffer (matches weekly report lookback)
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

        # Build user content (now includes 8-K filings and returns filing_lookup)
        user_content, filing_lookup = _build_filter_user_content(ticker, phase1_json, filings, eight_k_filings)

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
                            'max_output_tokens': 60000
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
                    "generation_time_ms": generation_time_ms,
                    "filing_lookup": filing_lookup
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
        # Build user content (now includes 8-K filings and returns filing_lookup)
        user_content, filing_lookup = _build_filter_user_content(ticker, phase1_json, filings, eight_k_filings)

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
            "generation_time_ms": generation_time_ms,
            "filing_lookup": filing_lookup
        }

    except Exception as e:
        LOG.error(f"[{ticker}] Phase 1.5 Claude exception: {e}", exc_info=True)
        return None


# =============================================================================
# STEP 2 & 3: TRIAGE AND SONNET REWRITE
# =============================================================================

def _classify_bullet_for_rewrite(item: Dict) -> str:
    """
    Classify a bullet/paragraph for rewrite step.

    Returns:
        'all_known' - 100% KNOWN sentences → REMOVE (skip LLM)
        'all_new' - 100% NEW sentences → KEEP as-is (skip LLM)
        'mixed' - Mixed KNOWN/NEW → needs Sonnet rewrite
        'exempt' - Exempt section → skip entirely
    """
    if item.get('exempt', False):
        return 'exempt'

    sentences = item.get('sentences', [])
    if not sentences:
        # No sentence analysis available, treat as all_new (keep original)
        return 'all_new'

    known_count = 0
    new_count = 0

    for s in sentences:
        action = s.get('sentence_action', 'KEEP').upper()
        if action == 'REMOVE':
            known_count += 1
        else:
            new_count += 1

    if known_count == 0:
        return 'all_new'
    elif new_count == 0:
        return 'all_known'
    else:
        return 'mixed'


def _build_rewrite_input(item: Dict, section_type: str) -> Dict:
    """
    Build input for surgical deletion rewrite.

    Args:
        item: Bullet or paragraph dict with sentences
        section_type: 'bullet' or 'paragraph'

    Returns:
        Dict with original content and sentences to delete
    """
    original_content = item.get('original_content', '') or item.get('content', '')
    sentences = item.get('sentences', [])

    sentences_to_delete = []

    for s in sentences:
        action = s.get('sentence_action', 'KEEP').upper()
        text = s.get('text', '')

        if action == 'REMOVE':
            sentences_to_delete.append(text)

    return {
        'original_content': original_content,
        'sentences_to_delete': sentences_to_delete,
        'context': {
            'section': item.get('section', ''),
            'bullet_id': item.get('bullet_id', ''),
            'type': section_type,
            'total_sentences': len(sentences),
            'delete_count': len(sentences_to_delete),
            'keep_count': len(sentences) - len(sentences_to_delete)
        }
    }


def _rewrite_single_item_sonnet(
    ticker: str,
    idx: int,
    item: Dict,
    section_type: str,
    anthropic_api_key: str
) -> Optional[Dict]:
    """
    Rewrite a single mixed-content item using Claude Sonnet 4.5.

    Includes internal HTTP retry logic for transient errors (429, 500, 503, 529).

    Args:
        ticker: Stock ticker
        idx: Item index
        item: Bullet or paragraph dict
        section_type: 'bullet' or 'paragraph'
        anthropic_api_key: Anthropic API key

    Returns:
        Dict with {'action': 'REWRITE'/'REMOVE', 'content': str, 'model': str}
        or None if failed (caller should try fallback)
    """
    from modules.json_utils import extract_json_from_claude_response

    rewrite_input = _build_rewrite_input(item, section_type)

    # Build user message for surgical deletion
    user_content = f"""ORIGINAL CONTENT:
{rewrite_input['original_content']}

SENTENCES TO DELETE ({rewrite_input['context']['delete_count']} of {rewrite_input['context']['total_sentences']}):
"""
    for i, sentence in enumerate(rewrite_input['sentences_to_delete'], 1):
        user_content += f"{i}. \"{sentence}\"\n"

    user_content += f"""
TASK: Delete the sentences above from the original content. Glue the remaining text for coherence.
Do NOT modify any other sentence - keep them exactly as they appear in the original."""

    headers = {
        "x-api-key": anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    data = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 2000,
        "temperature": 0.0,
        "system": SONNET_REWRITE_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": user_content
            }
        ]
    }

    # Internal HTTP retry logic for transient errors
    max_retries = 2
    response = None

    for attempt in range(max_retries + 1):
        try:
            start_time = time.time()
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=60
            )
            elapsed_ms = int((time.time() - start_time) * 1000)

            # Success - break retry loop
            if response.status_code == 200:
                break

            # Transient errors - retry with exponential backoff
            if response.status_code in [429, 500, 503, 529] and attempt < max_retries:
                wait_time = 2 ** attempt  # 1s, 2s
                LOG.warning(f"[{ticker}] ⚠️ Sonnet rewrite [{idx}] HTTP {response.status_code} "
                           f"(attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            # Non-retryable error or max retries reached
            break

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                LOG.warning(f"[{ticker}] ⏱️ Sonnet rewrite [{idx}] timeout "
                           f"(attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                LOG.error(f"[{ticker}] ❌ Sonnet rewrite [{idx}] timeout after {max_retries + 1} attempts")
                return None

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                LOG.warning(f"[{ticker}] 🔌 Sonnet rewrite [{idx}] network error "
                           f"(attempt {attempt + 1}/{max_retries + 1}): {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                LOG.error(f"[{ticker}] ❌ Sonnet rewrite [{idx}] network error after {max_retries + 1} attempts: {e}")
                return None

    # Check response
    if response is None:
        LOG.error(f"[{ticker}] ❌ Sonnet rewrite [{idx}] no response after {max_retries + 1} attempts")
        return None

    if response.status_code != 200:
        LOG.warning(f"[{ticker}] ❌ Sonnet rewrite [{idx}] failed with HTTP {response.status_code}")
        return None

    # Parse response
    response_json = response.json()
    response_text = response_json.get('content', [{}])[0].get('text', '')

    parsed = extract_json_from_claude_response(response_text, ticker)

    if parsed and 'action' in parsed:
        action = parsed.get('action', 'KEEP').upper()
        content = parsed.get('content', '')

        LOG.info(f"[{ticker}] ✅ Sonnet rewrite [{idx}]: {action} ({elapsed_ms}ms)")

        return {
            'action': action,
            'content': content if action == 'REWRITE' else '',
            'rewrite_time_ms': elapsed_ms,
            'model': 'claude-sonnet-4-5'
        }
    else:
        LOG.warning(f"[{ticker}] ❌ Sonnet rewrite [{idx}] failed to parse JSON response")
        return None


def _rewrite_single_item_gemini(
    ticker: str,
    idx: int,
    item: Dict,
    section_type: str,
    gemini_api_key: str
) -> Optional[Dict]:
    """
    Rewrite a single mixed-content item using Gemini 2.5 Pro (fallback).

    Includes internal retry logic for transient errors.

    Args:
        ticker: Stock ticker
        idx: Item index
        item: Bullet or paragraph dict
        section_type: 'bullet' or 'paragraph'
        gemini_api_key: Google Gemini API key

    Returns:
        Dict with {'action': 'REWRITE'/'REMOVE', 'content': str, 'model': str}
        or None if failed
    """
    import google.generativeai as genai
    from modules.json_utils import extract_json_from_claude_response

    try:
        genai.configure(api_key=gemini_api_key)

        rewrite_input = _build_rewrite_input(item, section_type)

        # Build user message for surgical deletion (same format as Sonnet)
        user_content = f"""ORIGINAL CONTENT:
{rewrite_input['original_content']}

SENTENCES TO DELETE ({rewrite_input['context']['delete_count']} of {rewrite_input['context']['total_sentences']}):
"""
        for i, sentence in enumerate(rewrite_input['sentences_to_delete'], 1):
            user_content += f"{i}. \"{sentence}\"\n"

        user_content += f"""
TASK: Delete the sentences above from the original content. Glue the remaining text for coherence.
Do NOT modify any other sentence - keep them exactly as they appear in the original."""

        # Create Gemini model with system instruction
        model = genai.GenerativeModel(
            'gemini-2.5-pro',
            system_instruction=SONNET_REWRITE_PROMPT  # Reuse same prompt
        )

        # Internal retry logic
        max_retries = 2
        response = None

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = model.generate_content(
                    user_content,
                    generation_config={
                        'temperature': 0.0,
                        'max_output_tokens': 2000
                    }
                )
                elapsed_ms = int((time.time() - start_time) * 1000)
                break  # Success

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

                if is_retryable and attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] ⚠️ Gemini rewrite [{idx}] error "
                               f"(attempt {attempt + 1}/{max_retries + 1}): {error_str[:100]}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] ❌ Gemini rewrite [{idx}] failed after {attempt + 1} attempts: {error_str[:200]}")
                    return None

        if response is None:
            LOG.error(f"[{ticker}] ❌ Gemini rewrite [{idx}] no response after {max_retries + 1} attempts")
            return None

        response_text = response.text

        if not response_text or len(response_text.strip()) < 5:
            LOG.warning(f"[{ticker}] ❌ Gemini rewrite [{idx}] empty response")
            return None

        # Parse JSON response (reuse same parser)
        parsed = extract_json_from_claude_response(response_text, ticker)

        if parsed and 'action' in parsed:
            action = parsed.get('action', 'KEEP').upper()
            content = parsed.get('content', '')

            LOG.info(f"[{ticker}] ✅ Gemini rewrite [{idx}]: {action} ({elapsed_ms}ms)")

            return {
                'action': action,
                'content': content if action == 'REWRITE' else '',
                'rewrite_time_ms': elapsed_ms,
                'model': 'gemini-2.5-pro'
            }
        else:
            LOG.warning(f"[{ticker}] ❌ Gemini rewrite [{idx}] failed to parse JSON response")
            return None

    except Exception as e:
        LOG.error(f"[{ticker}] ❌ Gemini rewrite [{idx}] exception: {e}")
        return None


def _rewrite_mixed_content(
    ticker: str,
    items_to_rewrite: List[Tuple[int, Dict, str]],
    anthropic_api_key: str,
    gemini_api_key: str = None
) -> Dict[int, Dict]:
    """
    Rewrite mixed-content bullets/paragraphs with retry and fallback.

    Flow per item:
    1. Sonnet attempt 1 → success → done
    2. Sonnet attempt 2 (orchestrator retry) → success → done
    3. Gemini 2.5 Pro fallback → success → done
    4. All failed → mark as rewrite_failed

    Args:
        ticker: Stock ticker
        items_to_rewrite: List of (index, item_dict, section_type) tuples
        anthropic_api_key: Anthropic API key
        gemini_api_key: Google Gemini API key (optional, for fallback)

    Returns:
        Dict mapping index -> {'action': 'REWRITE'/'REMOVE'/'KEEP', 'content': str, ...}
    """
    if not items_to_rewrite:
        return {}

    results = {}

    LOG.info(f"[{ticker}] Phase 1.5: Rewriting {len(items_to_rewrite)} mixed-content items")

    for idx, item, section_type in items_to_rewrite:
        result = None

        # Attempt 1: Sonnet
        if anthropic_api_key:
            result = _rewrite_single_item_sonnet(ticker, idx, item, section_type, anthropic_api_key)

        # Attempt 2: Sonnet retry (orchestrator level)
        if result is None and anthropic_api_key:
            LOG.info(f"[{ticker}] 🔄 Retrying Sonnet for item [{idx}]...")
            result = _rewrite_single_item_sonnet(ticker, idx, item, section_type, anthropic_api_key)

        # Attempt 3: Gemini fallback
        if result is None and gemini_api_key:
            LOG.info(f"[{ticker}] 🔄 Falling back to Gemini 2.5 Pro for item [{idx}]...")
            result = _rewrite_single_item_gemini(ticker, idx, item, section_type, gemini_api_key)

        # All failed - keep original
        if result is None:
            LOG.warning(f"[{ticker}] ❌ All rewrite attempts failed for item [{idx}], keeping original")
            results[idx] = {
                'action': 'KEEP',
                'content': item.get('filtered_content', ''),
                'rewrite_failed': True
            }
        else:
            results[idx] = result

    return results


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
            LOG.warning(f"[{ticker}] Phase 1.5: Gemini failed")
            result = None

    # Claude fallback - DISABLED (Dec 2025)
    # if result is None and anthropic_api_key:
    #     LOG.info(f"[{ticker}] Phase 1.5: Using Claude Sonnet 4.5 (fallback)")
    #     result = _filter_known_info_claude(ticker, phase1_json, filings, anthropic_api_key, eight_k_filings)
    #
    #     if result and result.get("json_output"):
    #         LOG.info(f"[{ticker}] Phase 1.5: Claude succeeded (fallback)")
    #     else:
    #         LOG.error(f"[{ticker}] Phase 1.5: Claude also failed")
    #         result = None

    if result is None:
        LOG.error(f"[{ticker}] Phase 1.5: Gemini failed, no fallback available")
        return None

    # Build final output
    json_output = result["json_output"]

    # Merge original_content from Phase 1 JSON (AI doesn't need to echo it back)
    json_output = _merge_original_content(json_output, phase1_json)

    # =========================================================================
    # STEP 2 & 3: TRIAGE AND SONNET REWRITE (for mixed content)
    # =========================================================================
    bullets = json_output.get("bullets", [])
    paragraphs = json_output.get("paragraphs", [])

    # Step 2: Classify each item
    bullet_classifications = {}
    paragraph_classifications = {}

    for idx, bullet in enumerate(bullets):
        bullet_classifications[idx] = _classify_bullet_for_rewrite(bullet)

    for idx, para in enumerate(paragraphs):
        paragraph_classifications[idx] = _classify_bullet_for_rewrite(para)

    # Count classifications for logging
    mixed_bullet_count = sum(1 for c in bullet_classifications.values() if c == 'mixed')
    mixed_para_count = sum(1 for c in paragraph_classifications.values() if c == 'mixed')
    total_mixed = mixed_bullet_count + mixed_para_count

    LOG.info(f"[{ticker}] Phase 1.5 Step 2 Triage - "
             f"Bullets: {dict((k, sum(1 for v in bullet_classifications.values() if v == k)) for k in ['all_known', 'all_new', 'mixed', 'exempt'])} | "
             f"Paragraphs: {dict((k, sum(1 for v in paragraph_classifications.values() if v == k)) for k in ['all_known', 'all_new', 'mixed', 'exempt'])}")

    # Step 3: Rewrite mixed items (Sonnet primary with 1 retry, Gemini Pro fallback)
    rewrite_time_ms = 0
    rewrite_models_used = set()  # Track which models were used (initialized here for all paths)

    if total_mixed > 0 and (anthropic_api_key or gemini_api_key):
        LOG.info(f"[{ticker}] Phase 1.5 Step 3: {total_mixed} items need rewrite")

        # Build lists of mixed items with their indices
        bullet_items = [(idx, bullets[idx], 'bullet') for idx in range(len(bullets))
                        if bullet_classifications.get(idx) == 'mixed']
        para_items = [(idx, paragraphs[idx], 'paragraph') for idx in range(len(paragraphs))
                      if paragraph_classifications.get(idx) == 'mixed']

        # Rewrite bullets (with retry + fallback)
        if bullet_items:
            start_time = time.time()
            bullet_rewrite_results = _rewrite_mixed_content(
                ticker, bullet_items, anthropic_api_key, gemini_api_key
            )
            rewrite_time_ms += int((time.time() - start_time) * 1000)

            # Apply results to bullets
            for idx, result_data in bullet_rewrite_results.items():
                if result_data.get('model'):
                    rewrite_models_used.add(result_data['model'])
                if result_data['action'] == 'REWRITE':
                    bullets[idx]['action'] = 'REWRITE'
                    bullets[idx]['rewritten_content'] = result_data['content']
                    bullets[idx]['filtered_content'] = ''
                elif result_data['action'] == 'REMOVE':
                    bullets[idx]['action'] = 'REMOVE'
                    bullets[idx]['filtered_content'] = ''
                elif result_data.get('rewrite_failed'):
                    bullets[idx]['action'] = 'KEEP'
                    # Keep existing filtered_content

        # Rewrite paragraphs (with retry + fallback)
        if para_items:
            start_time = time.time()
            para_rewrite_results = _rewrite_mixed_content(
                ticker, para_items, anthropic_api_key, gemini_api_key
            )
            rewrite_time_ms += int((time.time() - start_time) * 1000)

            # Apply results to paragraphs
            for idx, result_data in para_rewrite_results.items():
                if result_data.get('model'):
                    rewrite_models_used.add(result_data['model'])
                if result_data['action'] == 'REWRITE':
                    paragraphs[idx]['action'] = 'REWRITE'
                    paragraphs[idx]['rewritten_content'] = result_data['content']
                    paragraphs[idx]['filtered_content'] = ''
                elif result_data['action'] == 'REMOVE':
                    paragraphs[idx]['action'] = 'REMOVE'
                    paragraphs[idx]['filtered_content'] = ''
                elif result_data.get('rewrite_failed'):
                    paragraphs[idx]['action'] = 'KEEP'
                    # Keep existing filtered_content

        LOG.info(f"[{ticker}] Phase 1.5 Step 3: Rewrite completed in {rewrite_time_ms}ms (models: {', '.join(rewrite_models_used) or 'none'})")

    elif total_mixed > 0:
        LOG.warning(f"[{ticker}] Phase 1.5: {total_mixed} items need rewrite but no API keys available")
        # Fall back to mechanical filtering (keep filtered_content as-is)

    # Apply classifications for non-mixed items
    for idx, bullet in enumerate(bullets):
        classification = bullet_classifications.get(idx)
        if classification == 'all_known':
            bullet['action'] = 'REMOVE'
            bullet['filtered_content'] = ''
        elif classification == 'all_new':
            bullet['action'] = 'KEEP'
            # Clear filtered_content since original is unchanged
            if 'filtered_content' in bullet:
                del bullet['filtered_content']
        elif classification == 'exempt':
            bullet['action'] = 'KEEP'
            bullet['exempt'] = True

    for idx, para in enumerate(paragraphs):
        classification = paragraph_classifications.get(idx)
        if classification == 'all_known':
            para['action'] = 'REMOVE'
            para['filtered_content'] = ''
        elif classification == 'all_new':
            para['action'] = 'KEEP'
            # Clear filtered_content since original is unchanged
            if 'filtered_content' in para:
                del para['filtered_content']
        elif classification == 'exempt':
            para['action'] = 'KEEP'
            para['exempt'] = True

    # Update summary counts
    summary = json_output.get("summary", {})
    rewrite_count = sum(1 for b in bullets if b.get('action') == 'REWRITE')
    rewrite_count += sum(1 for p in paragraphs if p.get('action') == 'REWRITE')
    summary['rewritten'] = rewrite_count

    return {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "filings_used": filings_info,
        "filing_lookup": result.get("filing_lookup", {}),  # Maps identifiers to metadata
        "summary": summary,
        "bullets": bullets,
        "paragraphs": paragraphs,
        "model_used": result.get("model_used", "unknown"),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "generation_time_ms": result.get("generation_time_ms", 0),
        "rewrite_time_ms": rewrite_time_ms,
        "rewrite_models": list(rewrite_models_used)
    }


# =============================================================================
# EMAIL HTML GENERATOR
# =============================================================================

def _resolve_source_type(source_type: str, filing_lookup: Dict) -> str:
    """
    Resolve a filing identifier (e.g., '8K_1') to a display string.

    Args:
        source_type: Filing identifier from AI (e.g., 'TRANSCRIPT_1', '8K_2')
        filing_lookup: Dict mapping identifiers to metadata

    Returns:
        Formatted display string (e.g., '8-K filed Nov 24, 2025: CDO Appointment')
    """
    if not source_type or not filing_lookup:
        return source_type or ''

    # Check if this identifier exists in the lookup
    if source_type in filing_lookup:
        metadata = filing_lookup[source_type]
        return metadata.get('display', source_type)

    # Handle legacy format (if AI returns old-style labels)
    # Map old labels to new lookup keys
    legacy_map = {
        'Transcript': 'TRANSCRIPT_1',
        '10-Q': '10Q_1',
        '10-K': '10K_1',
        '8-K': '8K_1',  # Ambiguous but better than nothing
    }

    if source_type in legacy_map:
        mapped_id = legacy_map[source_type]
        if mapped_id in filing_lookup:
            return filing_lookup[mapped_id].get('display', source_type)

    # Return original if no mapping found
    return source_type


def _build_filing_timeline_html(filing_lookup: Dict) -> str:
    """
    Build HTML section showing filing timeline with days-ago calculation.

    Args:
        filing_lookup: Dict mapping filing IDs to metadata

    Returns:
        HTML string for timeline section, or empty string if no filings
    """
    if not filing_lookup:
        return ""

    from datetime import datetime as dt, timedelta

    today = dt.now().date()
    staleness_cutoff = today - timedelta(days=7)

    # Parse dates and sort chronologically (newest first)
    filings_with_dates = []
    for filing_id, meta in filing_lookup.items():
        date_str = meta.get('date', '')
        filing_type = meta.get('type', 'Unknown')

        # Parse date string (format: "Sep 04, 2025" or "Nov 30, 2025")
        filing_date = None
        if date_str and date_str != 'Unknown Date':
            try:
                filing_date = dt.strptime(date_str, '%b %d, %Y').date()
            except ValueError:
                try:
                    filing_date = dt.strptime(date_str, '%B %d, %Y').date()
                except ValueError:
                    pass

        # Build display info
        if filing_type == 'Transcript':
            quarter = meta.get('quarter', 'Q?')
            year = meta.get('year', '????')
            display_name = f"Transcript: {quarter} {year} Earnings Call"
        elif filing_type == '8-K':
            short_title = meta.get('short_title', meta.get('title', 'Untitled'))
            display_name = f"8-K: {short_title}"
        else:
            display_name = meta.get('display', filing_id)

        filings_with_dates.append({
            'id': filing_id,
            'type': filing_type,
            'date': filing_date,
            'date_str': date_str,
            'display_name': display_name
        })

    # Sort by date descending (newest first), None dates at end
    filings_with_dates.sort(key=lambda x: x['date'] if x['date'] else dt.min.date(), reverse=True)

    # Build HTML
    html = """
<div class="summary" style="margin-top: 20px;">
<strong>Filing Timeline (Knowledge Base)</strong>
<table style="width: 100%; margin-top: 15px; border-collapse: collapse; font-size: 14px;">
<tr style="border-bottom: 1px solid #e0e0e0;">
<th style="text-align: left; padding: 8px 12px; color: #666; font-weight: 600;">Filing</th>
<th style="text-align: right; padding: 8px 12px; color: #666; font-weight: 600;">Date</th>
<th style="text-align: right; padding: 8px 12px; color: #666; font-weight: 600;">Age</th>
</tr>
"""

    current_date_str = today.strftime('%B %d, %Y')
    html += f"""<tr style="background: #f8f9fa;">
<td colspan="3" style="padding: 8px 12px; font-size: 13px; color: #495057;">
<strong>Current Date:</strong> {current_date_str}
</td>
</tr>
"""

    for f in filings_with_dates:
        if f['date']:
            days_ago = (today - f['date']).days
            age_str = f"{days_ago} day{'s' if days_ago != 1 else ''} ago"
            date_display = f['date'].strftime('%b %d, %Y')
        else:
            age_str = "Unknown"
            date_display = "Unknown"

        html += f"""<tr style="border-bottom: 1px solid #f0f0f0;">
<td style="padding: 10px 12px;">{_escape_html(f['display_name'])}</td>
<td style="text-align: right; padding: 10px 12px; color: #666;">{date_display}</td>
<td style="text-align: right; padding: 10px 12px; color: #888;">{age_str}</td>
</tr>
"""

    # Add staleness cutoff line
    cutoff_str = staleness_cutoff.strftime('%b %d, %Y')
    html += f"""</table>
<div style="margin-top: 15px; padding: 12px; background: #fff3cd; border-radius: 6px; font-size: 13px; color: #856404;">
<strong>Staleness Cutoff:</strong> {cutoff_str}<br>
Any earnings call or company release before this date is STALE.
</div>
</div>
"""

    return html


def generate_known_info_filter_email(ticker: str, filter_result: Dict) -> str:
    """
    Generate simple HTML email showing sentence-level filter results.

    Args:
        ticker: Stock ticker
        filter_result: Output from filter_known_information() with sentence-level structure

    Returns:
        HTML string for email body
    """
    if not filter_result:
        return f"<html><body><h2>Phase 1.5 Filter Failed for {ticker}</h2><p>No results available.</p></body></html>"

    summary = filter_result.get("summary", {})
    bullets = filter_result.get("bullets", [])
    paragraphs = filter_result.get("paragraphs", [])
    filings = filter_result.get("filings_used", {})
    filing_lookup = filter_result.get("filing_lookup", {})  # For resolving identifiers
    model = filter_result.get("model_used", "unknown")
    gen_time = filter_result.get("generation_time_ms", 0)
    rewrite_time = filter_result.get("rewrite_time_ms", 0)
    rewrite_models = filter_result.get("rewrite_models", [])

    # Build filing list string (10-K and 10-Q intentionally excluded)
    filing_parts = []
    if 'transcript' in filings:
        t = filings['transcript']
        filing_parts.append(f"Transcript ({t['quarter']} {t['year']})")
    # NOTE: 10-Q intentionally excluded from knowledge base (Dec 2025)
    # NOTE: 10-K also intentionally excluded from knowledge base
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
.bullet {{ background: white; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #ddd; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.bullet-keep {{ border-left-color: #28a745; }}
.bullet-remove {{ border-left-color: #dc3545; }}
.bullet-exempt {{ border-left-color: #6c757d; }}
.bullet-rewrite {{ border-left-color: #007bff; }}
.bullet-header {{ font-weight: bold; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }}
.action-badge {{ padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
.badge-keep {{ background: #d4edda; color: #155724; }}
.badge-remove {{ background: #f8d7da; color: #721c24; }}
.badge-exempt {{ background: #e9ecef; color: #495057; }}
.badge-rewrite {{ background: #cce5ff; color: #004085; }}
.content-box {{ background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0; font-size: 14px; }}
.sentence {{ padding: 8px; margin: 5px 0; border-radius: 4px; font-size: 13px; }}
.sentence-keep {{ background: #d4edda; border-left: 3px solid #28a745; }}
.sentence-remove {{ background: #f8d7da; border-left: 3px solid #dc3545; }}
.sentence-text {{ margin-bottom: 5px; }}
.claims {{ margin: 5px 0 0 15px; }}
.claim {{ padding: 2px 0; font-size: 12px; }}
.claim-known {{ color: #dc3545; }}
.claim-new {{ color: #28a745; }}
.filtered {{ background: #e8f5e9; padding: 10px; border-radius: 4px; margin-top: 10px; }}
.rewritten {{ background: #e3f2fd; padding: 10px; border-radius: 4px; margin-top: 10px; border-left: 3px solid #007bff; }}
.section-header {{ background: #e9ecef; padding: 10px 15px; margin: 20px 0 10px 0; border-radius: 6px; font-weight: bold; }}
</style>
</head>
<body>

<div class="header">
<h2>Phase 1.5: Staleness Filter - {ticker}</h2>
<p><strong>Tagging:</strong> {model} ({gen_time}ms){f' | <strong>Rewrite:</strong> {", ".join(rewrite_models)} ({rewrite_time}ms)' if rewrite_models else ''}</p>
<p><strong>Knowledge Base:</strong> {filing_str}</p>
<p style="font-size: 12px; opacity: 0.7;">{filter_result.get('timestamp', '')[:19]}</p>
</div>

<div class="summary">
<strong>Sentence-Level Filter Summary</strong>
<table style="width: 100%; margin-top: 15px; border-collapse: collapse;">
<tr>
<td style="text-align: center; padding: 12px; background: #f5f5f5; border-radius: 6px; width: 12.5%;">
<div style="font-size: 24px; font-weight: bold; color: #1a1a2e;">{summary.get('total_bullets', 0) + summary.get('total_paragraphs', 0)}</div>
<div style="font-size: 11px; color: #666; margin-top: 5px;">Bullets/Paras</div>
</td>
<td style="text-align: center; padding: 12px; background: #f5f5f5; border-radius: 6px; width: 12.5%;">
<div style="font-size: 24px; font-weight: bold; color: #28a745;">{summary.get('kept', 0)}</div>
<div style="font-size: 11px; color: #666; margin-top: 5px;">Kept</div>
</td>
<td style="text-align: center; padding: 12px; background: #f5f5f5; border-radius: 6px; width: 12.5%;">
<div style="font-size: 24px; font-weight: bold; color: #007bff;">{summary.get('rewritten', 0)}</div>
<div style="font-size: 11px; color: #666; margin-top: 5px;">Rewritten</div>
</td>
<td style="text-align: center; padding: 12px; background: #f5f5f5; border-radius: 6px; width: 12.5%;">
<div style="font-size: 24px; font-weight: bold; color: #dc3545;">{summary.get('removed', 0)}</div>
<div style="font-size: 11px; color: #666; margin-top: 5px;">Removed</div>
</td>
<td style="text-align: center; padding: 12px; background: #f5f5f5; border-radius: 6px; width: 12.5%;">
<div style="font-size: 24px; font-weight: bold; color: #1a1a2e;">{summary.get('total_sentences', 0)}</div>
<div style="font-size: 11px; color: #666; margin-top: 5px;">Sentences</div>
</td>
<td style="text-align: center; padding: 12px; background: #f5f5f5; border-radius: 6px; width: 12.5%;">
<div style="font-size: 24px; font-weight: bold; color: #dc3545;">{summary.get('removed_sentences', 0)}</div>
<div style="font-size: 11px; color: #666; margin-top: 5px;">Removed Sent.</div>
</td>
<td style="text-align: center; padding: 12px; background: #f5f5f5; border-radius: 6px; width: 12.5%;">
<div style="font-size: 24px; font-weight: bold; color: #dc3545;">{summary.get('known_claims', 0)}</div>
<div style="font-size: 11px; color: #666; margin-top: 5px;">Known Claims</div>
</td>
<td style="text-align: center; padding: 12px; background: #f5f5f5; border-radius: 6px; width: 12.5%;">
<div style="font-size: 24px; font-weight: bold; color: #28a745;">{summary.get('new_claims', 0)}</div>
<div style="font-size: 11px; color: #666; margin-top: 5px;">New Claims</div>
</td>
</tr>
</table>
</div>
"""

    # Add filing timeline section (shows what AI was checking against)
    html += _build_filing_timeline_html(filing_lookup)

    # Add paragraphs section
    if paragraphs:
        html += '<div class="section-header">Paragraph Sections</div>\n'
        for p in paragraphs:
            action = p.get('action', 'KEEP').upper()
            exempt = p.get('exempt', False)

            if exempt:
                action_class = "bullet-exempt"
                badge_class = "badge-exempt"
                badge_text = "EXEMPT"
            else:
                action_class = f"bullet-{action.lower()}"
                badge_class = f"badge-{action.lower()}"
                badge_text = action

            html += f'<div class="bullet {action_class}">\n'
            html += f'<div class="bullet-header"><span>[{p.get("section", "?")}]</span><span class="action-badge {badge_class}">{badge_text}</span></div>\n'

            # Original content
            original = p.get('original_content', '') or p.get('content', '')
            if original:
                html += f'<div class="content-box"><strong>Original:</strong><br>{_escape_html(original)}</div>\n'

            # Sentence-level breakdown
            sentences = p.get('sentences', [])
            if sentences:
                html += '<div style="margin: 10px 0;"><strong>Sentence Analysis:</strong></div>\n'
                for s in sentences:
                    sent_action = s.get('sentence_action', 'KEEP').upper()
                    sent_class = 'sentence-keep' if sent_action == 'KEEP' else 'sentence-remove'
                    sent_icon = '✅' if sent_action == 'KEEP' else '❌'

                    html += f'<div class="sentence {sent_class}">\n'
                    html += f'<div class="sentence-text">{sent_icon} {_escape_html(s.get("text", ""))}</div>\n'

                    # Claims within sentence
                    claims = s.get('claims', [])
                    if claims:
                        html += '<div class="claims">\n'
                        for c in claims:
                            status = c.get('status', 'NEW')
                            claim_class = 'claim-known' if status == 'KNOWN' else 'claim-new'
                            icon = '❌' if status == 'KNOWN' else '✅'
                            source_type_raw = c.get('source_type', '')
                            source_type_display = _resolve_source_type(source_type_raw, filing_lookup)
                            evidence = c.get('evidence', '')

                            html += f'<div class="claim {claim_class}">{icon} {_escape_html(c.get("claim", ""))}'
                            if status == 'KNOWN' and evidence:
                                if source_type_raw:
                                    html += f' <span style="color: #666; font-size: 11px;">({source_type_display})</span>'
                                else:
                                    html += f' <span style="color: #856404; font-size: 11px;">⏰ {_escape_html(evidence)}</span>'
                            html += '</div>\n'
                        html += '</div>\n'
                    html += '</div>\n'

            # Rewritten/Filtered content display logic:
            # - EXEMPT: No output shown (use original)
            # - KEEP (all_new): No output shown (use original unchanged)
            # - KEEP (mixed, fallback): Show filtered_content if present
            # - REWRITE: Show rewritten_content
            # - REMOVE: No output shown
            if not exempt and action == 'REWRITE':
                rewritten = p.get('rewritten_content', '')
                if rewritten:
                    html += f'<div class="rewritten"><strong>Rewritten Content:</strong><br>{_escape_html(rewritten)}</div>\n'
            elif not exempt and action == 'KEEP':
                filtered = p.get('filtered_content', '')
                if filtered:
                    html += f'<div class="filtered"><strong>Filtered Content:</strong><br>{_escape_html(filtered)}</div>\n'

            html += '</div>\n'

    # Add bullets section
    if bullets:
        html += '<div class="section-header">Bullet Sections</div>\n'
        for b in bullets:
            action = b.get('action', 'KEEP').upper()
            exempt = b.get('exempt', False)

            if exempt:
                action_class = "bullet-exempt"
                badge_class = "badge-exempt"
                badge_text = "EXEMPT"
            else:
                action_class = f"bullet-{action.lower()}"
                badge_class = f"badge-{action.lower()}"
                badge_text = action

            html += f'<div class="bullet {action_class}">\n'
            html += f'<div class="bullet-header"><span>[{b.get("bullet_id", "?")}] {b.get("section", "")}</span><span class="action-badge {badge_class}">{badge_text}</span></div>\n'

            # Original content
            original = b.get('original_content', '') or b.get('content', '')
            if original:
                html += f'<div class="content-box"><strong>Original:</strong><br>{_escape_html(original)}</div>\n'

            # Sentence-level breakdown
            sentences = b.get('sentences', [])
            if sentences:
                html += '<div style="margin: 10px 0;"><strong>Sentence Analysis:</strong></div>\n'
                for s in sentences:
                    sent_action = s.get('sentence_action', 'KEEP').upper()
                    sent_class = 'sentence-keep' if sent_action == 'KEEP' else 'sentence-remove'
                    sent_icon = '✅' if sent_action == 'KEEP' else '❌'

                    html += f'<div class="sentence {sent_class}">\n'
                    html += f'<div class="sentence-text">{sent_icon} {_escape_html(s.get("text", ""))}</div>\n'

                    # Claims within sentence
                    claims = s.get('claims', [])
                    if claims:
                        html += '<div class="claims">\n'
                        for c in claims:
                            status = c.get('status', 'NEW')
                            claim_class = 'claim-known' if status == 'KNOWN' else 'claim-new'
                            icon = '❌' if status == 'KNOWN' else '✅'
                            source_type_raw = c.get('source_type', '')
                            source_type_display = _resolve_source_type(source_type_raw, filing_lookup)
                            evidence = c.get('evidence', '')

                            html += f'<div class="claim {claim_class}">{icon} {_escape_html(c.get("claim", ""))}'
                            if status == 'KNOWN' and evidence:
                                if source_type_raw:
                                    html += f' <span style="color: #666; font-size: 11px;">({source_type_display})</span>'
                                else:
                                    html += f' <span style="color: #856404; font-size: 11px;">⏰ {_escape_html(evidence)}</span>'
                            html += '</div>\n'
                        html += '</div>\n'
                    html += '</div>\n'

            # Rewritten/Filtered content display logic:
            # - EXEMPT: No output shown (use original)
            # - KEEP (all_new): No output shown (use original unchanged)
            # - KEEP (mixed, fallback): Show filtered_content if present
            # - REWRITE: Show rewritten_content
            # - REMOVE: No output shown
            if not exempt and action == 'REWRITE':
                rewritten = b.get('rewritten_content', '')
                if rewritten:
                    html += f'<div class="rewritten"><strong>Rewritten Content:</strong><br>{_escape_html(rewritten)}</div>\n'
            elif not exempt and action == 'KEEP':
                filtered = b.get('filtered_content', '')
                if filtered:
                    html += f'<div class="filtered"><strong>Filtered Content:</strong><br>{_escape_html(filtered)}</div>\n'

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
