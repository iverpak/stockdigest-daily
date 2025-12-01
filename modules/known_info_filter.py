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

Your task: For each bullet/paragraph, decompose into atomic claims and check each against the filings provided. Output filtered content containing ONLY NEW information.

═══════════════════════════════════════════════════════════════════════════════
WHAT IS "KNOWN" vs "NEW"?
═══════════════════════════════════════════════════════════════════════════════

KNOWN (filter out) - Information already in the filings:
- Specific numbers that appear in filings (revenue, margins, EPS, guidance, capex)
- Events explicitly stated in filings (announced X, reported Y, launched Z)
- Management quotes from transcripts
- Guidance figures from earnings calls
- Risk factors already disclosed in 10-K/10-Q
- Business model descriptions from filings
- Historical comparisons already discussed (YoY, QoQ changes mentioned in filings)
- Material events disclosed in 8-K filings (mergers, acquisitions, executive changes, restructuring)

NEW (keep) - Information NOT in the filings:
- Market reaction (stock price movement, trading volume changes)
- Analyst actions (upgrades, downgrades, price target changes, ratings)
- Third-party commentary (analyst quotes, industry expert opinions)
- Events occurring AFTER the latest filing date
- Rumors, speculation, breaking news not yet in filings
- Competitive developments not discussed in company filings
- External market data not from the company

═══════════════════════════════════════════════════════════════════════════════
CLAIM EXTRACTION
═══════════════════════════════════════════════════════════════════════════════

Decompose each bullet/paragraph into ATOMIC claims:
- Each specific number is ONE claim (e.g., "revenue $51.2B")
- Each percentage/growth rate is ONE claim (e.g., "+26% YoY")
- Each specific event is ONE claim (e.g., "announced partnership with X")
- Each directional statement is ONE claim (e.g., "beat guidance")
- Each attributed quote is ONE claim (e.g., "CEO said X")

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

Special case: If rewrite would be <15 words, mark as REMOVE instead.

═══════════════════════════════════════════════════════════════════════════════
REWRITE GUIDELINES
═══════════════════════════════════════════════════════════════════════════════

When action is REWRITE:

1. REMOVE all KNOWN claims completely - do not include any information from filings
2. PRESERVE all NEW claims with their full context
3. MAINTAIN narrative coherence - write flowing prose, not a choppy list
4. ADD minimal bridging context if needed for the NEW claim to make sense
5. PRESERVE attribution for NEW claims ("per analyst", "according to Reuters")

Example:
Original: "Meta reported Q3 revenue of $51.2B (+26% YoY), beating guidance, while stock pulled back 25% on AI spending concerns."
KNOWN: Q3 revenue $51.2B, +26% YoY, beat guidance (all in transcript/10-Q)
NEW: stock pulled back 25%, AI spending concerns

Rewritten: "META shares pulled back 25% in the past month amid investor concerns over AI spending levels."

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
          "source_type": "10-Q"
        },
        {
          "claim": "stock pulled back 25%",
          "status": "NEW",
          "source": null,
          "source_type": null
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
          "source_type": "Transcript" or "10-Q" or "10-K" or "8-K" or null
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
            cur.execute("""
                SELECT
                    filing_date,
                    report_title,
                    item_codes,
                    summary_markdown
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
                ORDER BY filing_date DESC
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

            LOG.info(f"[{ticker}] Phase 1.5: Found {len(filings)} filtered 8-K filings")
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

        # Retry logic
        max_retries = 2
        response = None
        generation_time_ms = 0

        for attempt in range(max_retries + 1):
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

                if is_retryable and attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] Phase 1.5 Gemini error (attempt {attempt + 1}): {error_str[:200]}")
                    LOG.warning(f"[{ticker}] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] Phase 1.5 Gemini failed after {attempt + 1} attempts: {error_str}")
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
        from modules.json_utils import extract_json_from_claude_response
        json_output = extract_json_from_claude_response(response_text, ticker)

        if not json_output:
            LOG.error(f"[{ticker}] Phase 1.5: Failed to parse Gemini JSON response")
            return None

        # Get token counts
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
    anthropic_api_key: str
) -> Optional[Dict]:
    """
    Filter known information using Claude Sonnet 4.5 (fallback).

    Args:
        ticker: Stock ticker
        phase1_json: Phase 1 JSON output
        filings: Dict with filing data
        anthropic_api_key: Anthropic API key

    Returns:
        Filter result dict or None if failed
    """
    try:
        # Build user content
        user_content = _build_filter_user_content(ticker, phase1_json, filings)

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

    # Claude fallback commented out for testing (Gemini-only mode)
    # if result is None and anthropic_api_key:
    #     LOG.info(f"[{ticker}] Phase 1.5: Using Claude Sonnet 4.5 (fallback)")
    #     result = _filter_known_info_claude(ticker, phase1_json, filings, anthropic_api_key)
    #
    #     if result and result.get("json_output"):
    #         LOG.info(f"[{ticker}] Phase 1.5: Claude succeeded (fallback)")
    #     else:
    #         LOG.error(f"[{ticker}] Phase 1.5: Claude also failed")
    #         result = None

    if result is None:
        LOG.error(f"[{ticker}] Phase 1.5: Gemini failed (Claude fallback disabled)")
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
                    source = f" → {c.get('source', '')}" if c.get('source') else ""
                    html += f'<div class="claim {claim_class}">{icon} {status}: {_escape_html(c.get("claim", ""))}{source}</div>\n'
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
                    source = f" → {c.get('source', '')}" if c.get('source') else ""
                    html += f'<div class="claim {claim_class}">{icon} {status}: {_escape_html(c.get("claim", ""))}{source}</div>\n'
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
