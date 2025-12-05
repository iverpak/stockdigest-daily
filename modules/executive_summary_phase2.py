"""
Phase 2 Executive Summary Enrichment Module

This module enriches Phase 1 executive summaries with filing context from 10-K, 10-Q, and Transcripts.

Phase 2 receives:
- Phase 1 JSON output (complete structure)
- Latest 10-K, 10-Q, Transcript from database (1-3 available)

Phase 2 returns:
- Enrichments dict keyed by bullet_id with: impact, sentiment, reason, context

The enrichments are then merged into Phase 1 JSON for final output.
"""

import json
import logging
import os
import copy
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Initialize logger
LOG = logging.getLogger(__name__)

# Constants
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


def detect_garbage_pattern(text: str) -> Tuple[bool, str]:
    """
    Detect if Gemini response is garbage/safety-filtered output.

    Safety filters often output repetitive patterns with very few unique characters.

    Args:
        text: Response text from Gemini

    Returns:
        Tuple of (is_garbage: bool, reason: str)
    """
    if len(text) < 100:
        return False, "too_short"

    # Count unique characters in first 1000 chars
    sample = text[:min(1000, len(text))]
    unique_chars = len(set(sample))

    # Check for repetitive pattern (safety filter signature)
    is_repetitive = unique_chars < 10  # Less than 10 unique chars = garbage

    # Check for specific patterns common in garbage output
    dash_count = text.count('-')
    dash_ratio = dash_count / len(text) if len(text) > 0 else 0
    has_dash_pattern = dash_ratio > 0.3  # >30% dashes

    number_count = text.count('1') + text.count('2') + text.count('3')
    number_ratio = number_count / len(text) if len(text) > 0 else 0
    has_number_spam = number_ratio > 0.2  # >20% numbers

    if is_repetitive or has_dash_pattern or has_number_spam:
        reason_parts = []
        if is_repetitive:
            reason_parts.append(f"unique_chars={unique_chars}")
        if has_dash_pattern:
            reason_parts.append(f"dashes={dash_count} ({dash_ratio:.1%})")
        if has_number_spam:
            reason_parts.append(f"numbers={number_count} ({number_ratio:.1%})")

        return True, f"repetitive ({', '.join(reason_parts)})"

    return False, "ok"



def get_phase2_system_prompt() -> str:
    """
    Load Phase 2 system prompt from file.

    The prompt is static (no ticker substitution) for optimal prompt caching.
    Ticker context is provided in user_content instead.

    Returns:
        str: Phase 2 system prompt
    """
    try:
        prompt_path = os.path.join(os.path.dirname(__file__), '_build_executive_summary_prompt_phase2')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        return prompt_template
    except Exception as e:
        LOG.error(f"Failed to load Phase 2 prompt: {e}")
        raise


def format_entity_references(config: Dict) -> Dict[str, str]:
    """
    Extract and format competitor/upstream/downstream entities from ticker config.

    Formats entities as: "Company Name (TICKER)" or "Company Name" if no ticker.
    Used to populate entity reference section in Phase 2 prompt.

    Args:
        config: Ticker configuration dict from ticker_reference

    Returns:
        dict with keys: 'competitors', 'upstream', 'downstream'
        Each value is comma-separated string or 'None' if empty.

    Example output:
        {
            'competitors': 'SolarEdge Technologies (SEDG), Generac Holdings (GNRC)',
            'upstream': 'ON Semiconductor (ON), Texas Instruments (TXN)',
            'downstream': 'Sunrun Inc. (RUN), Tesla, Inc. (TSLA)'
        }
    """
    if not config:
        return {
            'competitors': 'None',
            'upstream': 'None',
            'downstream': 'None'
        }

    competitors = []
    upstream = []
    downstream = []

    # Extract competitors
    for comp in config.get("competitors", []):
        if isinstance(comp, dict):
            name = comp.get("name", "")
            ticker_sym = comp.get("ticker", "")
            if name and ticker_sym:
                competitors.append(f"{name} ({ticker_sym})")
            elif name:
                competitors.append(name)

    # Extract upstream
    value_chain = config.get("value_chain", {})
    for comp in value_chain.get("upstream", []):
        if isinstance(comp, dict):
            name = comp.get("name", "")
            ticker_sym = comp.get("ticker", "")
            if name and ticker_sym:
                upstream.append(f"{name} ({ticker_sym})")
            elif name:
                upstream.append(name)

    # Extract downstream
    for comp in value_chain.get("downstream", []):
        if isinstance(comp, dict):
            name = comp.get("name", "")
            ticker_sym = comp.get("ticker", "")
            if name and ticker_sym:
                downstream.append(f"{name} ({ticker_sym})")
            elif name:
                downstream.append(name)

    return {
        'competitors': ', '.join(competitors) if competitors else 'None',
        'upstream': ', '.join(upstream) if upstream else 'None',
        'downstream': ', '.join(downstream) if downstream else 'None'
    }


def _fetch_available_filings(ticker: str, db_func) -> Dict[str, Dict]:
    """
    Fetch latest 10-K, 10-Q, and Transcript from database.

    Args:
        ticker: Stock ticker symbol
        db_func: Database connection function

    Returns:
        dict with keys '10k', '10q', 'transcript' (only keys present if data available)
        Each value is dict with: text, fiscal_year/quarter, filing_date/report_date, company_name
    """
    filings = {}

    try:
        with db_func() as conn, conn.cursor() as cur:
            # 1. Latest Transcript (prefer Claude if multiple exist for same period)
            cur.execute("""
                SELECT summary_text, fiscal_quarter, fiscal_year, report_date, company_name, ai_provider
                FROM transcript_summaries
                WHERE ticker = %s AND report_type = 'transcript'
                ORDER BY fiscal_year DESC, fiscal_quarter DESC,
                         CASE WHEN ai_provider = 'claude' THEN 0 ELSE 1 END
                LIMIT 1
            """, (ticker,))

            row = cur.fetchone()
            if row and row['summary_text']:
                filings['transcript'] = {
                    'text': row['summary_text'],
                    'fiscal_quarter': row['fiscal_quarter'],
                    'fiscal_year': row['fiscal_year'],
                    'date': row['report_date'],
                    'company_name': row['company_name']
                }
                LOG.debug(f"[{ticker}] Found Transcript: {row['fiscal_quarter']} {row['fiscal_year']}")

            # 2. Latest 10-Q
            cur.execute("""
                SELECT profile_markdown, fiscal_year, fiscal_quarter, filing_date, company_name
                FROM sec_filings
                WHERE ticker = %s AND filing_type = '10-Q'
                ORDER BY fiscal_year DESC, fiscal_quarter DESC
                LIMIT 1
            """, (ticker,))

            row = cur.fetchone()
            if row and row['profile_markdown']:
                filings['10q'] = {
                    'text': row['profile_markdown'],
                    'fiscal_year': row['fiscal_year'],
                    'fiscal_quarter': row['fiscal_quarter'],
                    'filing_date': row['filing_date'],
                    'company_name': row['company_name']
                }
                LOG.debug(f"[{ticker}] Found 10-Q: {row['fiscal_quarter']} {row['fiscal_year']}")

            # 3. Latest 10-K
            cur.execute("""
                SELECT profile_markdown, fiscal_year, filing_date, company_name
                FROM sec_filings
                WHERE ticker = %s AND filing_type = '10-K'
                ORDER BY fiscal_year DESC
                LIMIT 1
            """, (ticker,))

            row = cur.fetchone()
            if row and row['profile_markdown']:
                filings['10k'] = {
                    'text': row['profile_markdown'],
                    'fiscal_year': row['fiscal_year'],
                    'filing_date': row['filing_date'],
                    'company_name': row['company_name']
                }
                LOG.debug(f"[{ticker}] Found 10-K: FY{row['fiscal_year']}")

    except Exception as e:
        LOG.error(f"[{ticker}] Failed to fetch filings: {e}")
        return {}

    # Fetch 8-K filings (after transcript date, no T-7 delay for Phase 2)
    transcript_date = None
    if 'transcript' in filings and filings['transcript'].get('date'):
        t_date = filings['transcript']['date']
        transcript_date = t_date.date() if hasattr(t_date, 'date') else t_date

    eight_k_list = _fetch_8k_filings_for_phase2(ticker, db_func, transcript_date)
    if eight_k_list:
        filings['8k'] = eight_k_list

    return filings


def _fetch_8k_filings_for_phase2(ticker: str, db_func, last_transcript_date=None) -> List[Dict]:
    """
    Fetch 8-K filings for Phase 2 context enrichment.

    Unlike Phase 1.5 (which uses T-7 buffer for duplicate detection), Phase 2 fetches
    8-Ks IMMEDIATELY (no delay) since the purpose is context enrichment, not filtering.

    Applies 3-layer filtering (same as Phase 1.5):
    - Layer 1: Item code filter (material events only)
    - Layer 2: Exhibit number filter (press releases, not legal docs)
    - Layer 3: Title keyword filter (exclude boilerplate)

    Time window:
    - Start: Last transcript date (or 90-day fallback)
    - End: Today (NO T-7 delay - immediate access for context)
    - Max: 90 days

    No exhibit cap (unlike Phase 1.5's max 3 per filing date).

    Args:
        ticker: Stock ticker
        db_func: Database connection function
        last_transcript_date: Date of last earnings call (optional)

    Returns:
        List of 8-K filings with filing_date, report_title, item_codes, exhibit_number, summary_markdown
    """
    from datetime import date, timedelta

    try:
        with db_func() as conn, conn.cursor() as cur:
            # Calculate time window
            today = date.today()
            end_date = today  # NO T-7 delay for Phase 2 (immediate access)
            max_lookback = today - timedelta(days=90)  # 90-day safety cap

            # Start date: after last transcript, or 90-day fallback
            if last_transcript_date:
                start_date = max(last_transcript_date, max_lookback)
            else:
                start_date = max_lookback

            LOG.info(f"[{ticker}] Phase 2: Fetching 8-Ks from {start_date} to {end_date} (no T-7 delay)")

            # Query with 3-layer filtering (no exhibit cap)
            cur.execute("""
                SELECT
                    filing_date,
                    report_title,
                    item_codes,
                    exhibit_number,
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
                  -- Layer 3: Title exclusions (legal boilerplate)
                  AND report_title NOT ILIKE '%%Legal Opinion%%'
                  AND report_title NOT ILIKE '%%Underwriting Agreement%%'
                  AND report_title NOT ILIKE '%%Indenture%%'
                  AND report_title NOT ILIKE '%%Officers'' Certificate%%'
                  AND report_title NOT ILIKE '%%Notes Due%%'
                  AND report_title NOT ILIKE '%%Bylaws%%'
                ORDER BY filing_date DESC, exhibit_number ASC
            """, (ticker, start_date, end_date))

            rows = cur.fetchall()

            filings = []
            for row in rows:
                filings.append({
                    'filing_date': row['filing_date'] if isinstance(row, dict) else row[0],
                    'report_title': row['report_title'] if isinstance(row, dict) else row[1],
                    'item_codes': row['item_codes'] if isinstance(row, dict) else row[2],
                    'exhibit_number': row['exhibit_number'] if isinstance(row, dict) else row[3],
                    'summary_markdown': row['summary_markdown'] if isinstance(row, dict) else row[4]
                })

            LOG.info(f"[{ticker}] Phase 2: Found {len(filings)} 8-K filings (no exhibit cap)")
            return filings

    except Exception as e:
        LOG.error(f"[{ticker}] Phase 2: Error fetching 8-K filings: {e}")
        return []


# Emoji converter deleted - Phase 2 accepts text with emoji or markdown headers
# Claude/Gemini understand both formats equally well


def _build_phase2_user_content(ticker: str, phase1_json: Dict, filings: Dict) -> str:
    """
    Build Phase 2 user content combining Phase 1 JSON and filing sources.

    Format matches old _build_executive_summary_prompt() structure:
    - Current date for temporal checks
    - Phase 1 JSON first
    - Then filing sources with proper headers (Transcript, 8-Ks, 10-Q, 10-K)

    Args:
        ticker: Stock ticker symbol
        phase1_json: Complete Phase 1 JSON output
        filings: Dict with keys '10k', '10q', 'transcript', '8k'

    Returns:
        str: Formatted user content for Phase 2 prompt
    """
    # Add current date for temporal checks (used by Phase 2 staleness filtering)
    current_date = datetime.now().strftime('%B %d, %Y')  # e.g., "November 18, 2025"

    content = f"CURRENT DATE: {current_date}\n\n"
    content += "---\n\n"

    # Start with Phase 1 JSON
    phase1_str = json.dumps(phase1_json, indent=2)
    content += f"PHASE 1 ANALYSIS (TO BE ENRICHED):\n\n{phase1_str}\n\n"
    content += "---\n\n"
    content += "AVAILABLE FILING SOURCES FOR CONTEXT:\n\n"

    # Add Transcript if available (matches old format)
    if 'transcript' in filings:
        t = filings['transcript']
        quarter = t['fiscal_quarter']
        year = t['fiscal_year']
        company = t['company_name'] or ticker
        date = t['date'].strftime('%b %d, %Y') if t['date'] else 'Unknown Date'

        # Use transcript text directly (Claude/Gemini understand emoji headers)
        transcript_text = t['text']

        content += f"LATEST EARNINGS CALL (TRANSCRIPT):\n\n"
        content += f"[{ticker} ({company}) {quarter} {year} Earnings Call ({date})]\n\n"
        content += f"{transcript_text}\n\n\n"

    # Add 8-K filings if available (after Transcript, before 10-Q)
    # These are material events since last earnings call
    if '8k' in filings and filings['8k']:
        eight_k_list = filings['8k']

        # Calculate date range for display
        dates = [f['filing_date'] for f in eight_k_list if f.get('filing_date')]
        if dates:
            # Dates are already sorted DESC, so last is oldest, first is newest
            newest_date = dates[0]
            oldest_date = dates[-1]
            if hasattr(newest_date, 'strftime'):
                newest_str = newest_date.strftime('%b %d, %Y')
                oldest_str = oldest_date.strftime('%b %d, %Y')
            else:
                newest_str = str(newest_date)
                oldest_str = str(oldest_date)
            date_range = f"between {oldest_str} and {newest_str}"
        else:
            date_range = "recent"

        content += f"RECENT 8-K FILINGS (SINCE LAST EARNINGS):\n\n"
        content += f"[{len(eight_k_list)} filing(s) {date_range}]\n\n"

        for i, filing in enumerate(eight_k_list, start=1):
            filing_date = filing.get('filing_date')
            if hasattr(filing_date, 'strftime'):
                date_str = filing_date.strftime('%b %d, %Y')
            else:
                date_str = str(filing_date) if filing_date else 'Unknown Date'

            report_title = filing.get('report_title', 'Untitled')
            item_codes = filing.get('item_codes', 'Unknown')
            exhibit_number = filing.get('exhibit_number', 'Unknown')
            summary = filing.get('summary_markdown', '')

            content += f"--- 8-K #{i}: {date_str} (Exhibit {exhibit_number}) ---\n"
            content += f"[{ticker} - {report_title} (Items: {item_codes})]\n\n"
            content += f"{summary}\n\n"

        content += "\n"

    # Add 10-Q if available (matches old format)
    if '10q' in filings:
        q = filings['10q']
        quarter = q['fiscal_quarter']
        year = q['fiscal_year']
        company = q['company_name'] or ticker
        date = q['filing_date'].strftime('%b %d, %Y') if q['filing_date'] else 'Unknown Date'

        content += f"LATEST QUARTERLY REPORT (10-Q):\n\n"
        content += f"[{ticker} ({company}) {quarter} {year} 10-Q Filing, Filed: {date}]\n\n"
        content += f"{q['text']}\n\n\n"

    # Add 10-K if available (matches old format)
    if '10k' in filings:
        k = filings['10k']
        year = k['fiscal_year']
        company = k['company_name'] or ticker
        date = k['filing_date'].strftime('%b %d, %Y') if k['filing_date'] else 'Unknown Date'

        content += f"COMPANY 10-K PROFILE:\n\n"
        content += f"[{ticker} ({company}) 10-K FILING FOR FISCAL YEAR {year}, Filed: {date}]\n\n"
        content += f"{k['text']}\n\n\n"

    return content


def _generate_phase2_gemini(
    ticker: str,
    phase1_json: Dict,
    filings: Dict,
    config: Dict,
    gemini_api_key: str,
    db_func
) -> Optional[Dict]:
    """
    Generate Phase 2 enrichments using Gemini 2.5 Pro.

    Args:
        ticker: Stock ticker symbol
        phase1_json: Complete Phase 1 JSON output
        filings: Dict with keys '10k', '10q', 'transcript'
        config: Ticker configuration dict
        gemini_api_key: Google Gemini API key
        db_func: Database connection function

    Returns:
        dict with:
            enrichments: dict keyed by bullet_id with impact, sentiment, reason, context
            scenario_contexts: dict with bottom_line_context, upside_scenario_context, downside_scenario_context
            ai_model: "gemini-2.5-pro"
            prompt_tokens: int
            completion_tokens: int
            generation_time_ms: int
        Or None if failed
    """
    import time
    import google.generativeai as genai

    try:
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)

        # Load system prompt (static, no entity injection for caching)
        system_prompt = get_phase2_system_prompt()

        # Format entity references for user content
        entity_refs = format_entity_references(config)

        # Build user content with entity references at the TOP
        entity_header = f"""ENTITY RELATIONSHIPS (for entity tag classification):

Competitors: {entity_refs['competitors']}
Upstream Suppliers: {entity_refs['upstream']}
Downstream Customers: {entity_refs['downstream']}

---

"""
        base_user_content = _build_phase2_user_content(ticker, phase1_json, filings)
        user_content = entity_header + base_user_content

        # Log filing sources for debugging (helps identify what triggered failures)
        filing_sources = list(filings.keys())
        LOG.info(f"[{ticker}] 📄 Phase 2 input: {len(filing_sources)} filing source(s): {filing_sources}")
        for filing_type, filing_data in filings.items():
            if filing_type == '8k':
                # 8k is a list of filings, each with 'summary_markdown'
                total_chars = sum(len(f.get('summary_markdown', '')) for f in filing_data)
                LOG.info(f"[{ticker}]   - {filing_type}: {len(filing_data)} filings, {total_chars:,} chars total")
            else:
                text_len = len(filing_data.get('text', ''))
                LOG.info(f"[{ticker}]   - {filing_type}: {text_len:,} chars")

        # Estimate token counts for logging
        system_tokens_est = len(system_prompt) // 4
        user_tokens_est = len(user_content) // 4
        total_tokens_est = system_tokens_est + user_tokens_est
        LOG.info(f"[{ticker}] Phase 2 Gemini prompt size: system={len(system_prompt)} chars (~{system_tokens_est} tokens), user={len(user_content)} chars (~{user_tokens_est} tokens), total=~{total_tokens_est} tokens")

        # Create Gemini model with system instruction
        model = genai.GenerativeModel(
            'gemini-2.5-pro',
            system_instruction=system_prompt
        )

        LOG.info(f"[{ticker}] Calling Gemini 2.5 Pro for Phase 2 enrichment")

        # Retry logic for transient errors
        max_retries = 2
        response = None
        generation_time_ms = 0

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = model.generate_content(
                    user_content,
                    generation_config={
                        'temperature': 0.0,
                        'max_output_tokens': 20000,
                        'response_mime_type': 'application/json'
                    }
                )
                generation_time_ms = int((time.time() - start_time) * 1000)

                # Success - break retry loop
                break

            except Exception as e:
                error_str = str(e)

                # Check for retryable errors (quota, rate limit, service unavailable)
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
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    LOG.warning(f"[{ticker}] ⚠️ Gemini error (attempt {attempt + 1}/{max_retries + 1}): {error_str[:200]}")
                    LOG.warning(f"[{ticker}] 🔄 Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error or max retries reached
                    LOG.error(f"[{ticker}] ❌ Gemini 2.5 Pro Phase 2 failed after {attempt + 1} attempts: {error_str}")
                    return None

        # Check if we got a response
        if response is None:
            LOG.error(f"[{ticker}] ❌ No response from Gemini after {max_retries + 1} attempts")
            return None

        # Extract text from response
        response_text = response.text

        # Log Gemini response metadata for debugging
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]

            # Log finish reason
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason:
                finish_reason_name = str(finish_reason).split('.')[-1] if hasattr(finish_reason, 'name') else str(finish_reason)
                LOG.info(f"[{ticker}] 🔍 Gemini finish_reason: {finish_reason_name}")

            # Check safety ratings
            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                # Filter for non-negligible safety concerns
                safety_concerns = []
                for rating in candidate.safety_ratings:
                    # rating.probability can be: NEGLIGIBLE, LOW, MEDIUM, HIGH
                    if hasattr(rating, 'probability'):
                        prob_name = str(rating.probability).split('.')[-1] if hasattr(rating.probability, 'name') else str(rating.probability)
                        if prob_name not in ['NEGLIGIBLE', 'HARM_PROBABILITY_UNSPECIFIED']:
                            category_name = str(rating.category).split('.')[-1] if hasattr(rating.category, 'name') else str(rating.category)
                            safety_concerns.append(f"{category_name}={prob_name}")

                if safety_concerns:
                    LOG.error(f"[{ticker}] 🚨 Gemini safety filter triggered: {safety_concerns}")

        if not response_text or len(response_text.strip()) < 10:
            LOG.error(f"[{ticker}] ❌ Gemini returned empty Phase 2 response")
            return None

        # Parse JSON from response using unified parser
        from modules.json_utils import extract_json_from_claude_response
        parsed_json = extract_json_from_claude_response(response_text, ticker)

        if not parsed_json:
            # Detect garbage pattern and provide enhanced diagnostics
            is_garbage, garbage_reason = detect_garbage_pattern(response_text)

            if is_garbage:
                LOG.error(f"[{ticker}] 🚨 Gemini returned GARBAGE output (likely safety filter)")
                LOG.error(f"[{ticker}]   Pattern detected: {garbage_reason}")
                LOG.error(f"[{ticker}]   Response length: {len(response_text):,} chars")
                LOG.error(f"[{ticker}]   Filing sources: {filing_sources}")
            else:
                LOG.error(f"[{ticker}] ❌ Failed to extract Phase 2 JSON from Gemini response")
                LOG.error(f"[{ticker}]   Response looks structurally ok, but not valid JSON")
                LOG.error(f"[{ticker}]   Response length: {len(response_text):,} chars")

            # Optional: Save response to file for deep debugging (requires env var)
            save_failures = os.getenv("SAVE_GEMINI_FAILURES", "false").lower() == "true"
            if save_failures:
                try:
                    import tempfile
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = f"/tmp/gemini_phase2_failure_{ticker}_{timestamp}.txt"

                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"Ticker: {ticker}\n")
                        f.write(f"Filing sources: {filing_sources}\n")
                        f.write(f"Response length: {len(response_text)}\n")
                        f.write(f"Garbage pattern: {is_garbage} ({garbage_reason})\n")
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write(f"\n{'='*80}\n")
                        f.write(f"FULL RESPONSE:\n")
                        f.write(f"{'='*80}\n\n")
                        f.write(response_text)

                    LOG.info(f"[{ticker}] 💾 Saved full Gemini response to: {filepath}")
                except Exception as e:
                    LOG.warning(f"[{ticker}] ⚠️ Failed to save Gemini response: {e}")

            return None

        # Extract token usage from Gemini response
        prompt_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
        completion_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0

        # Handle different possible structures from Gemini (same as Claude)
        enrichments = {}
        scenario_contexts = {}

        if "enrichments" in parsed_json and isinstance(parsed_json["enrichments"], dict):
            # Case 2: Wrapped in "enrichments" key
            enrichments = parsed_json["enrichments"]
            for key in ["bottom_line_context", "upside_scenario_context", "downside_scenario_context"]:
                if key in parsed_json:
                    scenario_contexts[key] = parsed_json[key]
        elif "sections" in parsed_json and isinstance(parsed_json["sections"], dict):
            # Case 1: Full section structure - flatten to bullet_id dict
            for section_name, bullets in parsed_json["sections"].items():
                if isinstance(bullets, list):
                    for bullet in bullets:
                        if isinstance(bullet, dict) and "bullet_id" in bullet:
                            bid = bullet["bullet_id"]
                            enrichments[bid] = {
                                "impact": bullet.get("impact"),
                                "sentiment": bullet.get("sentiment"),
                                "reason": bullet.get("reason"),
                                "relevance": bullet.get("relevance"),
                                "context": bullet.get("context"),
                                "entity": bullet.get("entity")  # For competitive_industry_dynamics bullets
                            }
            for key in ["bottom_line_context", "upside_scenario_context", "downside_scenario_context"]:
                if key in parsed_json:
                    scenario_contexts[key] = parsed_json[key]
        else:
            # Case 3: Direct enrichments dict + scenario contexts
            for key, value in parsed_json.items():
                if key in ["bottom_line_context", "upside_scenario_context", "downside_scenario_context"]:
                    scenario_contexts[key] = value
                else:
                    enrichments[key] = value

        # Debug logging
        if enrichments:
            sample_size = min(3, len(enrichments))
            sample_bullets = list(enrichments.items())[:sample_size]
            LOG.info(f"[{ticker}] 📋 Phase 2 Gemini output sample ({sample_size}/{len(enrichments)} bullets, BEFORE validation):")
            for bullet_id, data in sample_bullets:
                if isinstance(data, dict):
                    present_fields = [f for f in data.keys() if data.get(f)]
                    empty_fields = [f for f in data.keys() if not data.get(f)]
                    LOG.info(f"  • {bullet_id}:")
                    LOG.info(f"      ✓ Present: {', '.join(present_fields) if present_fields else 'NONE'}")
                    if empty_fields:
                        LOG.info(f"      ✗ Empty/None: {', '.join(empty_fields)}")

        if scenario_contexts:
            LOG.info(f"[{ticker}] 📄 Phase 2 Gemini scenario contexts: {', '.join(scenario_contexts.keys())}")

        LOG.info(f"✅ [{ticker}] Phase 2 Gemini enrichment generated ({len(response_text)} chars, {len(enrichments)} bullets enriched, {prompt_tokens} prompt tokens, {completion_tokens} completion tokens, {generation_time_ms}ms)")

        return {
            "enrichments": enrichments,
            "scenario_contexts": scenario_contexts,
            "ai_model": "gemini-2.5-pro",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generation_time_ms": generation_time_ms
        }

    except Exception as e:
        LOG.error(f"❌ [{ticker}] Exception in Phase 2 Gemini generation: {e}", exc_info=True)
        return None


def _generate_phase2_claude(
    ticker: str,
    phase1_json: Dict,
    filings: Dict,
    config: Dict,
    anthropic_api_key: str,
    db_func
) -> Optional[Dict]:
    """
    Generate Phase 2 enrichments using Claude Sonnet 4.5 (fallback).

    Takes Phase 1 JSON output and filing sources, returns enrichments dict.

    Args:
        ticker: Stock ticker symbol
        phase1_json: Complete Phase 1 JSON output
        filings: Dict with keys '10k', '10q', 'transcript' (from _fetch_available_filings)
        config: Ticker configuration dict
        anthropic_api_key: Anthropic API key
        db_func: Database connection function

    Returns:
        dict with:
            enrichments: dict keyed by bullet_id with impact, sentiment, reason, context
            scenario_contexts: dict with bottom_line_context, upside_scenario_context, downside_scenario_context
            ai_model: "claude-sonnet-4-5-20250929"
            prompt_tokens: int
            completion_tokens: int
            generation_time_ms: int
        Or None if failed
    """
    import time

    try:
        # Load system prompt (static, cacheable - no entity injection)
        system_prompt = get_phase2_system_prompt()

        # Format entity references for user content
        entity_refs = format_entity_references(config)

        # Build user content with entity references at the TOP (enables prompt caching)
        entity_header = f"""ENTITY RELATIONSHIPS (for entity tag classification):

Competitors: {entity_refs['competitors']}
Upstream Suppliers: {entity_refs['upstream']}
Downstream Customers: {entity_refs['downstream']}

---

"""
        base_user_content = _build_phase2_user_content(ticker, phase1_json, filings)
        user_content = entity_header + base_user_content

        # Estimate token counts for logging
        system_tokens_est = len(system_prompt) // 4
        user_tokens_est = len(user_content) // 4
        total_tokens_est = system_tokens_est + user_tokens_est
        LOG.info(f"[{ticker}] Phase 2 prompt size: system={len(system_prompt)} chars (~{system_tokens_est} tokens), user={len(user_content)} chars (~{user_tokens_est} tokens), total=~{total_tokens_est} tokens")

        # Prepare API call
        headers = {
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01",  # Prompt caching support
            "content-type": "application/json"
        }

        data = {
            "model": "claude-sonnet-4-5-20250929",  # Sonnet 4.5
            "max_tokens": 20000,  # Generous limit for enrichments
            "temperature": 0.0,   # Deterministic
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}  # Enable prompt caching
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }

        LOG.info(f"[{ticker}] Calling Claude API for Phase 2 enrichment")

        # Retry logic for transient errors (503, 429, 500)
        max_retries = 2
        response = None
        generation_time_ms = 0

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = requests.post(ANTHROPIC_API_URL, headers=headers, json=data, timeout=180)
                generation_time_ms = int((time.time() - start_time) * 1000)

                # Success - break retry loop
                if response.status_code == 200:
                    break

                # Transient errors - retry with exponential backoff
                if response.status_code in [429, 500, 503] and attempt < max_retries:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    error_preview = response.text[:200] if response.text else "No details"
                    LOG.warning(f"[{ticker}] ⚠️ API error {response.status_code} (attempt {attempt + 1}/{max_retries + 1}): {error_preview}")
                    LOG.warning(f"[{ticker}] 🔄 Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Non-retryable error or max retries reached - break
                break

            except requests.exceptions.Timeout as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] ⏱️ Request timeout (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] ❌ Request timeout after {max_retries + 1} attempts")
                    return None

            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] 🔌 Network error (attempt {attempt + 1}/{max_retries + 1}): {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] ❌ Network error after {max_retries + 1} attempts: {e}")
                    return None

        # Check if we got a response
        if response is None:
            LOG.error(f"[{ticker}] ❌ No response received after {max_retries + 1} attempts")
            return None

        if response.status_code == 200:
            result = response.json()

            # Extract usage stats
            usage = result.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)

            # Log cache performance
            cache_creation = usage.get("cache_creation_input_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            if cache_creation > 0:
                LOG.info(f"[{ticker}] 💾 CACHE CREATED: {cache_creation} tokens cached (Phase 2)")
            elif cache_read > 0:
                LOG.info(f"[{ticker}] ⚡ CACHE HIT: {cache_read} tokens read from cache (Phase 2) - 90% savings!")

            # Parse JSON response
            response_text = result.get("content", [{}])[0].get("text", "")

            if not response_text or len(response_text.strip()) < 10:
                LOG.error(f"❌ [{ticker}] Claude returned empty Phase 2 response")
                return None

            # Parse JSON from response using unified parser (4-tier fallback strategy)
            try:
                # Use shared JSON extraction utility (handles all response formats)
                from modules.json_utils import extract_json_from_claude_response
                parsed_json = extract_json_from_claude_response(response_text, ticker)

                if not parsed_json:
                    LOG.error(f"❌ [{ticker}] Failed to extract Phase 2 JSON from response")
                    return None

                # For logging: get JSON string length
                json_str = response_text.strip()

                # Handle different possible structures from Claude
                # Claude might return:
                # 1. {"sections": {"major_developments": [...]}} - full structure
                # 2. {"enrichments": {"FIN_001": {...}}} - wrapped enrichments
                # 3. {"FIN_001": {...}, "bottom_line_context": "..."} - direct enrichments + scenario contexts (NEW)

                enrichments = {}
                scenario_contexts = {}

                if "enrichments" in parsed_json and isinstance(parsed_json["enrichments"], dict):
                    # Case 2: Wrapped in "enrichments" key
                    enrichments = parsed_json["enrichments"]
                    # Extract scenario contexts if present at root level
                    for key in ["bottom_line_context", "upside_scenario_context", "downside_scenario_context"]:
                        if key in parsed_json:
                            scenario_contexts[key] = parsed_json[key]
                elif "sections" in parsed_json and isinstance(parsed_json["sections"], dict):
                    # Case 1: Full section structure - need to flatten to bullet_id dict
                    for section_name, bullets in parsed_json["sections"].items():
                        if isinstance(bullets, list):
                            for bullet in bullets:
                                if isinstance(bullet, dict) and "bullet_id" in bullet:
                                    bid = bullet["bullet_id"]
                                    enrichments[bid] = {
                                        "impact": bullet.get("impact"),
                                        "sentiment": bullet.get("sentiment"),
                                        "reason": bullet.get("reason"),
                                        "relevance": bullet.get("relevance"),
                                        "context": bullet.get("context"),
                                        "entity": bullet.get("entity")  # For competitive_industry_dynamics bullets
                                    }
                    # Extract scenario contexts if present at root level
                    for key in ["bottom_line_context", "upside_scenario_context", "downside_scenario_context"]:
                        if key in parsed_json:
                            scenario_contexts[key] = parsed_json[key]
                else:
                    # Case 3: Direct enrichments dict + scenario contexts
                    # Separate bullet enrichments from scenario contexts
                    for key, value in parsed_json.items():
                        if key in ["bottom_line_context", "upside_scenario_context", "downside_scenario_context"]:
                            scenario_contexts[key] = value
                        else:
                            # Assume it's a bullet enrichment (dict with impact, sentiment, etc.)
                            enrichments[key] = value

                # Debug logging: Show sample of what Claude actually returned (before validation)
                if enrichments:
                    sample_size = min(3, len(enrichments))
                    sample_bullets = list(enrichments.items())[:sample_size]
                    LOG.info(f"[{ticker}] 📋 Phase 2 raw output sample ({sample_size}/{len(enrichments)} bullets, BEFORE validation):")
                    for bullet_id, data in sample_bullets:
                        if isinstance(data, dict):
                            present_fields = [f for f in data.keys() if data.get(f)]
                            empty_fields = [f for f in data.keys() if not data.get(f)]
                            LOG.info(f"  • {bullet_id}:")
                            LOG.info(f"      ✓ Present: {', '.join(present_fields) if present_fields else 'NONE'}")
                            if empty_fields:
                                LOG.info(f"      ✗ Empty/None: {', '.join(empty_fields)}")
                            # Show first 60 chars of each field value
                            for field in ["impact", "sentiment", "reason", "relevance", "context"]:
                                if field in data:
                                    value = data[field]
                                    if value:
                                        preview = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)
                                        LOG.info(f"      → {field}: {preview}")
                                    else:
                                        LOG.info(f"      → {field}: (empty)")
                        else:
                            LOG.info(f"  • {bullet_id}: ⚠️ NOT A DICT (type={type(data).__name__})")

                # Debug logging: Show scenario contexts if present
                if scenario_contexts:
                    LOG.info(f"[{ticker}] 📄 Phase 2 scenario contexts found: {', '.join(scenario_contexts.keys())}")
                    for key, value in scenario_contexts.items():
                        if value:
                            preview = value[:80] + "..." if len(value) > 80 else value
                            LOG.info(f"  • {key}: {preview}")
                        else:
                            LOG.info(f"  • {key}: (empty)")
                else:
                    LOG.info(f"[{ticker}] 📄 No scenario contexts in Phase 2 output")

                LOG.info(f"✅ [{ticker}] Phase 2 enrichment generated ({len(json_str)} chars, {len(enrichments)} bullets enriched, {prompt_tokens} prompt tokens, {completion_tokens} completion tokens, {generation_time_ms}ms)")

                return {
                    "enrichments": enrichments,
                    "scenario_contexts": scenario_contexts,
                    "ai_model": "claude-sonnet-4-5-20250929",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "generation_time_ms": generation_time_ms
                }

            except json.JSONDecodeError as e:
                LOG.error(f"❌ [{ticker}] Failed to parse Phase 2 JSON: {e}")
                LOG.error(f"Response preview: {response_text[:500]}")
                return None

        else:
            error_text = response.text[:500] if response.text else "No error details"
            LOG.error(f"❌ [{ticker}] Claude API error {response.status_code} after {max_retries + 1} attempts: {error_text}")
            return None

    except Exception as e:
        LOG.error(f"❌ [{ticker}] Exception in Phase 2 generation: {e}")
        return None


def generate_executive_summary_phase2(
    ticker: str,
    phase1_json: Dict,
    filings: Dict,
    config: Dict,
    anthropic_api_key: str,
    db_func,
    gemini_api_key: str  # Required parameter
) -> Optional[Dict]:
    """
    Generate Phase 2 enrichments with Gemini 2.5 Pro (primary) and Claude fallback.

    This is the main entry point for Phase 2 enrichment. It attempts Gemini first
    for cost savings (60% cheaper), then falls back to Claude if Gemini fails.

    Args:
        ticker: Stock ticker symbol
        phase1_json: Complete Phase 1 JSON output
        filings: Dict with keys '10k', '10q', 'transcript'
        config: Ticker configuration dict
        anthropic_api_key: Anthropic API key for Claude fallback
        db_func: Database connection function
        gemini_api_key: Google Gemini API key (required)

    Returns:
        dict with:
            enrichments: dict keyed by bullet_id with impact, sentiment, reason, context
            scenario_contexts: dict with bottom_line_context, upside_scenario_context, downside_scenario_context
            ai_model: "gemini-2.5-pro" or "claude-sonnet-4-5-20250929"
            prompt_tokens: int
            completion_tokens: int
            generation_time_ms: int
        Or None if both providers failed
    """
    # Early exit: Check if there are any bullets to enrich
    # Phase 2 only processes these 5 sections (wall_street_sentiment, bottom_line,
    # upside_scenario, downside_scenario, key_variables are skipped per prompt)
    ENRICHABLE_SECTIONS = [
        'major_developments', 'financial_performance', 'risk_factors',
        'competitive_industry_dynamics', 'upcoming_catalysts'
    ]

    bullet_count = sum(
        len(phase1_json.get('sections', {}).get(s, []))
        for s in ENRICHABLE_SECTIONS
    )

    if bullet_count == 0:
        LOG.info(f"[{ticker}] ℹ️ Phase 2: No bullets to enrich (0 across 5 sections) - skipping API calls")
        return {
            "enrichments": {},
            "scenario_contexts": {},
            "model": "skipped",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "generation_time_ms": 0
        }

    # Try Gemini 2.5 Pro first (primary)
    if gemini_api_key:
        LOG.info(f"[{ticker}] Phase 2: Attempting Gemini 2.5 Pro (primary)")
        gemini_result = _generate_phase2_gemini(
            ticker=ticker,
            phase1_json=phase1_json,
            filings=filings,
            config=config,
            gemini_api_key=gemini_api_key,
            db_func=db_func
        )

        if gemini_result and gemini_result.get("enrichments"):
            LOG.info(f"[{ticker}] ✅ Phase 2: Gemini 2.5 Pro succeeded")
            return gemini_result
        else:
            LOG.warning(f"[{ticker}] ⚠️ Phase 2: Gemini 2.5 Pro failed, falling back to Claude Sonnet")
    else:
        LOG.warning(f"[{ticker}] ⚠️ No Gemini API key provided, using Claude Sonnet only")

    # Fall back to Claude Sonnet 4.5
    if anthropic_api_key:
        LOG.info(f"[{ticker}] Phase 2: Using Claude Sonnet 4.5 (fallback)")
        claude_result = _generate_phase2_claude(
            ticker=ticker,
            phase1_json=phase1_json,
            filings=filings,
            config=config,
            anthropic_api_key=anthropic_api_key,
            db_func=db_func
        )

        if claude_result and claude_result.get("enrichments"):
            LOG.info(f"[{ticker}] ✅ Phase 2: Claude Sonnet succeeded (fallback)")
            return claude_result
        else:
            LOG.error(f"[{ticker}] ❌ Phase 2: Claude Sonnet also failed")
    else:
        LOG.error(f"[{ticker}] ❌ No Anthropic API key provided for fallback")

    # Both failed
    LOG.error(f"[{ticker}] ❌ Phase 2: Both Gemini and Claude failed - skipping enrichment")
    return None


def validate_phase2_json(enrichments: Dict, phase1_json: Dict = None, ticker: str = "") -> Tuple[bool, str, Dict]:
    """
    Validate Phase 2 enrichments structure with partial acceptance.

    This function filters out incomplete/invalid bullets but accepts the rest,
    preventing one bad bullet from destroying all Phase 2 enrichment work.

    Expected structure:
    {
        "bullet_id_1": {
            "context": "prose paragraph combining filing excerpts",
            "impact": "high impact|medium impact|low impact",
            "sentiment": "bullish|bearish|neutral",
            "reason": "brief reason string",
            "relevance": "direct|indirect|none",
            "entity": "Competitor|Market|Upstream|Downstream" (ONLY for competitive_industry_dynamics)
        },
        "bullet_id_2": { ... }
    }

    Args:
        enrichments: Dict keyed by bullet_id
        phase1_json: Optional Phase 1 JSON to determine bullet sections
        ticker: Optional ticker for logging

    Returns:
        Tuple of (is_valid: bool, error_message: str, valid_enrichments: Dict)
        - is_valid: True if ANY bullets are valid
        - error_message: Description of what was accepted/rejected
        - valid_enrichments: Dict containing only complete, valid bullets
    """
    if not isinstance(enrichments, dict):
        return False, "Enrichments must be object/dict", {}

    if not enrichments:
        return False, "Enrichments dict is empty", {}

    # Build bullet_id to section mapping
    bullet_sections = {}
    if phase1_json and "sections" in phase1_json:
        for section_name, section_data in phase1_json["sections"].items():
            if isinstance(section_data, list):
                for bullet in section_data:
                    if isinstance(bullet, dict) and "bullet_id" in bullet:
                        bullet_sections[bullet["bullet_id"]] = section_name

    valid_enrichments = {}
    invalid_bullets = []

    for bullet_id, data in enrichments.items():
        # Check if data is a dict
        if not isinstance(data, dict):
            invalid_bullets.append(f"{bullet_id} (not a dict)")
            continue

        # Determine which section this bullet belongs to
        section_name = bullet_sections.get(bullet_id, "unknown")
        is_competitive_industry = (section_name == "competitive_industry_dynamics")

        # Set required fields based on section
        if is_competitive_industry:
            required_fields = ["context", "impact", "sentiment", "reason", "relevance", "entity"]
        else:
            required_fields = ["context", "impact", "sentiment", "reason", "relevance"]

        # Fill in missing fields with empty string (accept partial data)
        missing_fields = [f for f in required_fields if f not in data or not data.get(f)]
        for field in required_fields:
            if field not in data or not data.get(field):
                data[field] = ""  # Leave blank, don't reject

        # Validate entity values if present and not empty (for competitive_industry_dynamics only)
        if is_competitive_industry and data.get("entity"):
            valid_entities = ["Competitor", "Market", "Upstream", "Downstream"]
            if data["entity"] not in valid_entities:
                LOG.warning(f"[{ticker}] Phase 2: {bullet_id} has invalid entity '{data['entity']}', setting to empty")
                data["entity"] = ""

        # Log what was missing for debugging
        if missing_fields:
            LOG.info(f"[{ticker}] Phase 2: {bullet_id} accepted with missing fields: {', '.join(missing_fields)}")
            LOG.debug(f"[{ticker}] Phase 2: {bullet_id} full data: {json.dumps(data, indent=2)}")

        # Accept bullet with partial data (no value validation, only check if empty)
        valid_enrichments[bullet_id] = data

    # If no valid enrichments, Phase 2 completely failed
    if not valid_enrichments:
        if invalid_bullets:
            # Show ALL invalid bullets when complete failure (not just first 3)
            error_detail = '; '.join(invalid_bullets[:5])
            if len(invalid_bullets) > 5:
                error_detail += f" (+{len(invalid_bullets) - 5} more with similar issues)"
        else:
            error_detail = "No enrichments provided"
        return False, f"No valid enrichments found ({len(invalid_bullets)} bullets failed). Issues: {error_detail}", {}

    # At least some enrichments are valid - accept them
    if invalid_bullets:
        # Partial success - some bullets filtered out
        error_msg = f"Accepted {len(valid_enrichments)}/{len(enrichments)} bullets. Filtered out: {'; '.join(invalid_bullets[:3])}"
        if len(invalid_bullets) > 3:
            error_msg += f" (+{len(invalid_bullets) - 3} more)"
    else:
        # Complete success - all bullets valid
        error_msg = f"All {len(valid_enrichments)} bullets validated successfully"

    return True, error_msg, valid_enrichments


def strip_escape_hatch_context(phase2_result: Dict) -> Dict:
    """
    Replace escape hatch text with empty string for cleaner display.

    When Phase 2 finds no relevant filing context, it outputs:
    "No relevant filing context found for this development"

    This function replaces that text with "" so templates can simply check
    if context exists, without seeing "not found" messages in the UI.

    Args:
        phase2_result: Phase 2 result dict with 'enrichments' and 'scenario_contexts' keys

    Returns:
        Modified phase2_result with escape hatch text replaced
    """
    # Use startswith() to handle variations (with/without period, etc.)
    ESCAPE_HATCH_PREFIX = "No relevant filing context found"

    # Strip escape hatch from bullet enrichments
    enrichments = phase2_result.get("enrichments", {})
    for bullet_id, enrichment in enrichments.items():
        if enrichment.get("context", "").startswith(ESCAPE_HATCH_PREFIX):
            enrichment["context"] = ""

    # Strip escape hatch from scenario contexts
    scenario_contexts = phase2_result.get("scenario_contexts", {})
    for key in ["bottom_line_context", "upside_scenario_context", "downside_scenario_context"]:
        if scenario_contexts.get(key, "").startswith(ESCAPE_HATCH_PREFIX):
            scenario_contexts[key] = ""

    return phase2_result


def sort_bullets_by_impact(bullets: List[Dict]) -> List[Dict]:
    """
    Sort bullets by impact level: high → medium → low → missing.

    Uses stable sort to preserve original order within same impact level.
    Un-enriched bullets (missing impact field) sink to bottom.

    Args:
        bullets: List of bullet dicts

    Returns:
        Sorted list of bullets
    """
    impact_order = {
        'high impact': 0,
        'medium impact': 1,
        'low impact': 2
    }

    def get_sort_key(bullet):
        impact = bullet.get('impact')
        if impact is None:
            return 999  # Un-enriched bullets go to bottom
        return impact_order.get(impact, 999)  # Unknown impact values go to bottom

    # Stable sort preserves original order for ties
    return sorted(bullets, key=get_sort_key)


def merge_phase1_phase2(phase1_json: Dict, phase2_result: Dict) -> Dict:
    """
    Merge Phase 2 enrichments and scenario contexts into Phase 1 JSON.

    Takes Phase 1 JSON structure and:
    1. Adds impact, sentiment, reason, relevance, context fields to each bullet
    2. Adds context field to paragraph sections (bottom_line, upside_scenario, downside_scenario)
    3. Sorts enriched bullet sections by impact (high → medium → low → missing)

    Args:
        phase1_json: Complete Phase 1 JSON output
        phase2_result: Phase 2 result dict with 'enrichments' and 'scenario_contexts' keys

    Returns:
        Merged JSON with Phase 2 fields added to bullets and scenarios, sorted by impact
    """
    merged = copy.deepcopy(phase1_json)
    enrichments = phase2_result.get("enrichments", {})
    scenario_contexts = phase2_result.get("scenario_contexts", {})

    # Merge bullet enrichments
    if enrichments:
        # List of sections with bullets (not paragraphs)
        bullet_sections = [
            "major_developments",
            "financial_performance",
            "risk_factors",
            "wall_street_sentiment",
            "competitive_industry_dynamics",
            "upcoming_catalysts",
            "key_variables"
        ]

        # Iterate through all sections
        for section_name in bullet_sections:
            if section_name not in merged.get("sections", {}):
                continue

            section_content = merged["sections"][section_name]

            if not isinstance(section_content, list):
                continue

            # Enrich each bullet
            for bullet in section_content:
                if not isinstance(bullet, dict):
                    continue

                bullet_id = bullet.get("bullet_id")
                if not bullet_id or bullet_id not in enrichments:
                    continue

                enrichment = enrichments[bullet_id]

                # HARD FILTER: Never merge context for wall_street_sentiment
                # Analyst opinions ARE the context - they already synthesize filing data
                if section_name == "wall_street_sentiment":
                    # Keep metadata (impact, sentiment, reason, relevance are valid)
                    bullet["impact"] = enrichment.get("impact")
                    bullet["sentiment"] = enrichment.get("sentiment")
                    bullet["reason"] = enrichment.get("reason")
                    bullet["relevance"] = enrichment.get("relevance")
                    # Force context to empty string (strip any generated context)
                    bullet["context"] = ""
                    if enrichment.get("context"):
                        LOG.warning(f"Stripped filing context from Wall Street bullet {bullet_id} "
                                   f"(context length: {len(enrichment.get('context', ''))} chars)")
                    continue  # Skip standard enrichment path

                # Standard enrichment for all other sections
                bullet["impact"] = enrichment.get("impact")
                bullet["sentiment"] = enrichment.get("sentiment")
                bullet["reason"] = enrichment.get("reason")
                bullet["relevance"] = enrichment.get("relevance")
                bullet["context"] = enrichment.get("context")
                bullet["entity"] = enrichment.get("entity")

    # Merge scenario contexts into paragraph sections
    if scenario_contexts:
        # Add context to bottom_line
        if "bottom_line_context" in scenario_contexts and scenario_contexts["bottom_line_context"]:
            if "bottom_line" in merged.get("sections", {}):
                merged["sections"]["bottom_line"]["context"] = scenario_contexts["bottom_line_context"]

        # Add context to upside_scenario
        if "upside_scenario_context" in scenario_contexts and scenario_contexts["upside_scenario_context"]:
            if "upside_scenario" in merged.get("sections", {}):
                merged["sections"]["upside_scenario"]["context"] = scenario_contexts["upside_scenario_context"]

        # Add context to downside_scenario
        if "downside_scenario_context" in scenario_contexts and scenario_contexts["downside_scenario_context"]:
            if "downside_scenario" in merged.get("sections", {}):
                merged["sections"]["downside_scenario"]["context"] = scenario_contexts["downside_scenario_context"]

    # Sort enriched bullet sections by impact (high → medium → low → missing)
    # Only sort sections that receive Phase 2 enrichments with impact field
    enriched_sections = [
        "major_developments",
        "financial_performance",
        "risk_factors",
        "wall_street_sentiment",
        "competitive_industry_dynamics"
    ]

    for section_name in enriched_sections:
        if section_name in merged.get("sections", {}):
            section_content = merged["sections"][section_name]
            if isinstance(section_content, list) and len(section_content) > 0:
                merged["sections"][section_name] = sort_bullets_by_impact(section_content)

    return merged


def merge_phase3_with_phase2(phase2_json: Dict, phase3_json: Dict) -> Dict:
    """
    Merge Phase 3 integrated content back into Phase 2 JSON using bullet_id matching.

    Phase 2 has: All metadata (impact, sentiment, reason, entity, date_range, filing_hints, context (original))
    Phase 3 has: bullet_id, topic_label, content (integrated), deduplication (status, absorbs/absorbed_by, etc.)

    Result: Phase 2 metadata + Phase 3 integrated content + deduplication info

    Args:
        phase2_json: Phase 1+2 merged JSON with all metadata
        phase3_json: Phase 3 output with integrated content and deduplication

    Returns:
        Final merged JSON with Phase 2 metadata + Phase 3 integrated content + deduplication
    """
    import copy

    # Deep copy Phase 2 to preserve all metadata
    merged = copy.deepcopy(phase2_json)

    # Bullet sections to merge
    bullet_sections = [
        "major_developments",
        "financial_performance",
        "risk_factors",
        "wall_street_sentiment",
        "competitive_industry_dynamics",
        "upcoming_catalysts",
        "key_variables"
    ]

    # Overlay Phase 3 integrated content onto Phase 2 bullets using bullet_id
    for section_name in bullet_sections:
        if section_name not in merged.get("sections", {}):
            continue

        # Build lookup by bullet_id from Phase 3
        phase3_bullets = phase3_json.get("sections", {}).get(section_name, [])
        phase3_map = {b['bullet_id']: b for b in phase3_bullets if 'bullet_id' in b}

        # Overlay integrated content onto Phase 2 bullets
        phase2_bullets = merged["sections"][section_name]
        for bullet in phase2_bullets:
            bullet_id = bullet.get('bullet_id')
            if bullet_id and bullet_id in phase3_map:
                phase3_bullet = phase3_map[bullet_id]

                # Add Phase 3 edited content and context as new fields
                # (preserve Phase 1 content and Phase 2 context for Quality Review comparison)
                bullet['content_integrated'] = phase3_bullet.get('content', '')
                bullet['context_integrated'] = phase3_bullet.get('context', '')

                # Add deduplication field if present
                if 'deduplication' in phase3_bullet:
                    bullet['deduplication'] = phase3_bullet['deduplication']
                else:
                    bullet['deduplication'] = {'status': 'unique'}

    # Scenarios (bottom_line, upside_scenario, downside_scenario)
    for section_name in ["bottom_line", "upside_scenario", "downside_scenario"]:
        if section_name in merged.get("sections", {}):
            phase3_section = phase3_json.get("sections", {}).get(section_name, {})
            if phase3_section and phase3_section.get("content"):
                # Add Phase 3 edited content and context as new fields
                # (preserve Phase 1 content and Phase 2 context for Quality Review comparison)
                merged["sections"][section_name]["content_integrated"] = phase3_section["content"]
                merged["sections"][section_name]["context_integrated"] = phase3_section.get("context", "")

    return merged


def apply_deduplication(phase3_merged_json: Dict) -> Dict:
    """
    Apply deduplication decisions from Phase 3 for Email #3 output.

    This function:
    1. Removes bullets marked as 'duplicate' (absorbed elsewhere)
    2. For 'primary' bullets, uses proposed_content/proposed_context if available
    3. Merges source_articles arrays from absorbed bullets into primary
    4. Returns clean JSON ready for Email #3 rendering

    Args:
        phase3_merged_json: Phase 3 merged JSON with deduplication metadata

    Returns:
        Deduplicated JSON with duplicates removed and primaries consolidated
    """
    import copy
    import logging

    LOG = logging.getLogger(__name__)

    # Deep copy to avoid modifying original
    result = copy.deepcopy(phase3_merged_json)

    # Bullet sections to process
    bullet_sections = [
        "major_developments",
        "financial_performance",
        "risk_factors",
        "wall_street_sentiment",
        "competitive_industry_dynamics",
        "upcoming_catalysts",
        "key_variables"
    ]

    # First pass: Build lookup of all bullets by bullet_id for source_articles merging
    all_bullets = {}
    for section_name in bullet_sections:
        bullets = result.get("sections", {}).get(section_name, [])
        for bullet in bullets:
            bullet_id = bullet.get('bullet_id')
            if bullet_id:
                all_bullets[bullet_id] = bullet

    # Track stats
    duplicates_removed = 0
    primaries_consolidated = 0

    # Second pass: Apply deduplication
    for section_name in bullet_sections:
        if section_name not in result.get("sections", {}):
            continue

        bullets = result["sections"][section_name]
        consolidated = []

        for bullet in bullets:
            dedup = bullet.get('deduplication', {'status': 'unique'})
            status = dedup.get('status', 'unique')

            if status == 'duplicate':
                # Skip - this bullet is absorbed elsewhere
                duplicates_removed += 1
                LOG.debug(f"Removing duplicate bullet: {bullet.get('bullet_id')} (absorbed by {dedup.get('absorbed_by')})")
                continue

            if status == 'primary':
                primaries_consolidated += 1

                # Use proposed_content for merged facts, but KEEP original context
                # (proposed_context often has duplicated filing references when merging bullets)
                proposed_content = dedup.get('proposed_content', '')
                if proposed_content:
                    bullet['content_integrated'] = proposed_content
                    # Use original Phase 1+2 context (clean, non-duplicated)
                    bullet['context_integrated'] = bullet.get('context', '')
                    LOG.debug(f"Using proposed_content with original context for primary bullet: {bullet.get('bullet_id')}")

                # Merge source_articles from absorbed bullets
                absorbed_ids = dedup.get('absorbs', [])
                if absorbed_ids:
                    primary_sources = set(bullet.get('source_articles', []))
                    for absorbed_id in absorbed_ids:
                        absorbed_bullet = all_bullets.get(absorbed_id, {})
                        absorbed_sources = absorbed_bullet.get('source_articles', [])
                        primary_sources.update(absorbed_sources)
                    # Sort for consistent output
                    bullet['source_articles'] = sorted(list(primary_sources))
                    LOG.debug(f"Merged source_articles for {bullet.get('bullet_id')}: {bullet['source_articles']}")

            # Add bullet to consolidated list (unique or primary)
            consolidated.append(bullet)

        result["sections"][section_name] = consolidated

    LOG.info(f"Deduplication applied: {duplicates_removed} duplicates removed, {primaries_consolidated} primaries consolidated")

    return result
