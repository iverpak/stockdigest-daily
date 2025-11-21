# modules/transcript_summaries.py

"""
Transcript Summaries Module

Handles earnings call transcripts and press release summarization using FMP API and Claude AI.
Extracted from app.py for better modularity.
"""

import json
import requests
import logging
import traceback
import re
from typing import Dict, List, Optional
from datetime import datetime, timezone
import pytz
from jinja2 import Environment, FileSystemLoader

LOG = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPT PROMPT V2 - JSON OUTPUT (Added 2025-01-15)
# ═══════════════════════════════════════════════════════════════════════════════

def load_transcript_prompt_v2() -> str:
    """
    Load transcript v2 prompt from file (static for optimal prompt caching).

    Returns the v2 prompt which outputs structured JSON with 15 sections.
    Pattern copied from executive_summary_phase1.get_phase1_system_prompt().

    The prompt is ticker-agnostic for optimal prompt caching. Ticker context
    is provided in user_content instead.

    Returns:
        str: Full system prompt text

    Raises:
        Exception: If prompt file cannot be loaded
    """
    try:
        import os
        # Prompt is in same directory as this module
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            '_build_research_summary_prompt_v2'
        )

        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        LOG.info(f"Loaded transcript prompt v2 ({len(prompt_template)} chars)")
        return prompt_template

    except Exception as e:
        LOG.error(f"Failed to load transcript prompt v2 from {prompt_path}: {e}")
        LOG.error(traceback.format_exc())
        raise


def strip_emoji(text: str) -> str:
    """
    Remove all emoji characters from text.

    This ensures Phase 2 can search for section headers without emoji interference.
    Phase 2 references transcript/10-K/10-Q sections, which shouldn't have emoji.

    Args:
        text: Input text that may contain emoji

    Returns:
        Text with all emoji removed
    """
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Initialize Jinja2 template environment
template_env = Environment(loader=FileSystemLoader('templates'))
research_template = template_env.get_template('email_research_report.html')

# ==============================================================================
# FMP API INTEGRATION
# ==============================================================================

def fetch_fmp_transcript_list(ticker: str, fmp_api_key: str) -> List[Dict]:
    """
    Fetch list of available transcripts for a ticker from FMP API.
    Returns list of dicts with {quarter, year, date}.
    """
    if not fmp_api_key:
        LOG.error("FMP_API_KEY not configured")
        return []

    try:
        url = f"https://financialmodelingprep.com/api/v4/earning_call_transcript"
        params = {"symbol": ticker, "apikey": fmp_api_key}

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            LOG.error(f"FMP transcript list API error {response.status_code}: {response.text[:200]}")
            return []

        data = response.json()

        # Handle error responses from FMP
        if isinstance(data, dict):
            if 'Error Message' in data:
                LOG.error(f"FMP API error for {ticker}: {data['Error Message']}")
                return []
            # Unexpected dict response
            LOG.error(f"FMP returned dict instead of list for {ticker}: {data}")
            return []

        if not isinstance(data, list):
            LOG.error(f"FMP returned unexpected type for {ticker}: {type(data)}")
            return []

        # FMP returns: [[quarter, year, date], ...]
        # Convert to dict format
        transcripts = []
        for item in data:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    transcripts.append({
                        "quarter": item[0],
                        "year": item[1],
                        "date": item[2]
                    })
                elif isinstance(item, dict) and all(k in item for k in ['quarter', 'year', 'date']):
                    # Handle object format (some tickers return this)
                    transcripts.append({
                        "quarter": item['quarter'],
                        "year": item['year'],
                        "date": item['date']
                    })
                else:
                    LOG.warning(f"Skipping invalid transcript item for {ticker}: {item}")
            except (KeyError, IndexError, TypeError) as e:
                LOG.warning(f"Failed to parse transcript item for {ticker}: {e} - item: {item}")
                continue

        LOG.info(f"Found {len(transcripts)} transcripts for {ticker}")
        return transcripts

    except Exception as e:
        LOG.error(f"Failed to fetch FMP transcript list for {ticker}: {e}")
        return []


def fetch_fmp_transcript(ticker: str, quarter: int, year: int, fmp_api_key: str) -> Optional[Dict]:
    """
    Fetch specific earnings transcript from FMP API.
    Returns dict with {quarter, year, date, content} or None if not found.
    """
    if not fmp_api_key:
        LOG.error("FMP_API_KEY not configured")
        return None

    try:
        url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}"
        params = {"quarter": quarter, "year": year, "apikey": fmp_api_key}

        response = requests.get(url, params=params, timeout=30)

        if response.status_code != 200:
            LOG.error(f"FMP transcript API error {response.status_code}: {response.text[:200]}")
            return None

        data = response.json()

        # FMP returns array with single item
        if not data or not isinstance(data, list) or len(data) == 0:
            LOG.warning(f"No transcript found for {ticker} Q{quarter} {year}")
            return None

        transcript = data[0]
        LOG.info(f"Fetched transcript for {ticker} Q{quarter} {year} ({len(transcript.get('content', ''))} chars)")
        return transcript

    except Exception as e:
        LOG.error(f"Failed to fetch FMP transcript for {ticker} Q{quarter} {year}: {e}")
        return None


def fetch_fmp_press_releases(ticker: str, fmp_api_key: str, limit: int = 20) -> List[Dict]:
    """
    Fetch press releases for a ticker from FMP API.
    Returns list of dicts with {date, title, text}.
    """
    if not fmp_api_key:
        LOG.error("FMP_API_KEY not configured")
        return []

    try:
        url = f"https://financialmodelingprep.com/api/v3/press-releases/{ticker}"
        params = {"page": 0, "apikey": fmp_api_key}

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            LOG.error(f"FMP press release API error {response.status_code}: {response.text[:200]}")
            return []

        data = response.json()

        # Return latest N releases
        releases = data[:limit] if isinstance(data, list) else []
        LOG.info(f"Fetched {len(releases)} press releases for {ticker}")
        return releases

    except Exception as e:
        LOG.error(f"Failed to fetch FMP press releases for {ticker}: {e}")
        return []


def fetch_fmp_press_release_by_date(ticker: str, target_date: str, fmp_api_key: str) -> Optional[Dict]:
    """
    Fetch specific press release by date (flexible format).

    Args:
        ticker: Stock ticker
        target_date: Date in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
        fmp_api_key: FMP API key

    Returns:
        First press release matching the date (FMP order), or None if not found
    """
    releases = fetch_fmp_press_releases(ticker, fmp_api_key, limit=50)

    # Normalize target date to YYYY-MM-DD (strip time if present)
    target_date_normalized = target_date.split()[0] if target_date else ''

    for release in releases:
        # Normalize FMP date to YYYY-MM-DD for comparison
        release_date = release.get('date', '')
        release_date_normalized = release_date.split()[0] if release_date else ''

        if release_date_normalized == target_date_normalized:
            return release  # Returns WITH full datetime preserved

    LOG.warning(f"Press release not found for {ticker} on {target_date}")
    return None


def fetch_fmp_press_release_by_date_and_title(
    ticker: str,
    target_date: str,
    target_title: str,
    fmp_api_key: str
) -> Optional[Dict]:
    """
    Fetch specific press release by full datetime AND title.

    Handles multiple PRs per day with same title (rare but possible).
    Used by worker to fetch exact PR content based on datetime + title.

    Args:
        ticker: Stock ticker
        target_date: Full datetime 'YYYY-MM-DD HH:MM:SS' (from job config)
        target_title: Press release title (exact match)
        fmp_api_key: FMP API key

    Returns:
        Exact press release matching datetime AND title, or None if not found
    """
    releases = fetch_fmp_press_releases(ticker, fmp_api_key, limit=50)

    # Normalize target datetime (strip microseconds if present, handle both formats)
    # Handles: "2025-11-13 10:00:00" or "2025-11-13 10:00:00.123456"
    target_datetime = target_date.split('.')[0] if target_date else ''

    for release in releases:
        # Get FMP datetime (already in 'YYYY-MM-DD HH:MM:SS' format)
        release_datetime = release.get('date', '')
        release_datetime_normalized = release_datetime.split('.')[0] if release_datetime else ''
        release_title = release.get('title', '')

        # Match by BOTH full datetime AND title (exact match)
        if release_datetime_normalized == target_datetime and release_title == target_title:
            return release  # Exact match found ✅

    # Fallback: Try matching by date only (backward compatibility with old job configs)
    target_date_only = target_datetime.split()[0] if target_datetime else ''
    for release in releases:
        release_date_only = release.get('date', '').split()[0] if release.get('date') else ''
        release_title = release.get('title', '')

        if release_date_only == target_date_only and release_title == target_title:
            LOG.warning(f"Press release matched by date only (no time) for {ticker}: {target_title[:50]}...")
            return release  # Fallback match ⚠️

    LOG.warning(f"Press release not found for {ticker} on {target_date} with title: {target_title[:50]}...")
    return None


# ==============================================================================
# PROMPTS
# ==============================================================================

# ==============================================================================
# AI SUMMARIZATION
# ==============================================================================

def _build_transcript_user_content_v2(
    ticker: str,
    content: str,
    config: Dict,
    content_type: str
) -> str:
    """
    Build user content for transcript v2 prompt (ticker-specific context).

    Args:
        ticker: Company ticker symbol
        content: Transcript or press release text
        config: Ticker configuration with company metadata
        content_type: 'transcript' or 'press_release'

    Returns:
        str: Formatted user content with ticker context
    """
    company_name = config.get("name", ticker)
    industry = config.get("industry", "Unknown")

    user_content = f"""# COMPANY CONTEXT

**Ticker:** {ticker}
**Company Name:** {company_name}
**Industry:** {industry}
**Document Type:** {content_type.upper()}

# CONTENT TO ANALYZE

{content}

---

Generate comprehensive JSON summary following all instructions in the system prompt.
Output must be valid, parseable JSON matching the schema exactly.
Include all 15 sections with complete data.
"""

    return user_content


def _generate_transcript_gemini_v2(
    ticker: str,
    content: str,
    config: Dict,
    content_type: str,
    gemini_api_key: str,
    max_retries: int = 2
) -> Optional[Dict]:
    """
    Generate transcript JSON summary using Gemini 2.5 Pro with v2 prompt.

    Pattern copied from executive_summary_phase1._generate_phase1_gemini().

    Args:
        ticker: Company ticker symbol
        content: Transcript or press release text
        config: Ticker configuration dict
        content_type: 'transcript' or 'press_release'
        gemini_api_key: Gemini API key
        max_retries: Maximum retry attempts (default: 2)

    Returns:
        Dict with:
            - json_output: Parsed 15-section JSON
            - model_used: 'gemini-2.5-pro'
            - prompt_tokens: Input token count
            - completion_tokens: Output token count
            - generation_time_ms: Processing time in milliseconds
        Or None if generation fails
    """
    import time
    import google.generativeai as genai

    try:
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)

        # Load v2 prompt (static, cacheable)
        system_prompt = load_transcript_prompt_v2()

        # Build ticker-specific user content
        user_content = _build_transcript_user_content_v2(
            ticker, content, config, content_type
        )

        # Create model with system instruction
        model = genai.GenerativeModel(
            'gemini-2.5-pro',
            system_instruction=system_prompt
        )

        # Retry logic for transient errors
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()

                response = model.generate_content(
                    user_content,
                    generation_config={
                        'temperature': 0.0,  # Deterministic output
                        'max_output_tokens': 16000  # Support comprehensive transcripts
                    }
                )

                generation_time_ms = int((time.time() - start_time) * 1000)

                # Extract response text
                response_text = response.text

                # Get token counts
                try:
                    prompt_tokens = response.usage_metadata.prompt_token_count
                    completion_tokens = response.usage_metadata.candidates_token_count
                except AttributeError:
                    prompt_tokens = 0
                    completion_tokens = 0

                LOG.info(
                    f"[{ticker}] Gemini 2.5 Pro generated response "
                    f"({generation_time_ms}ms, {prompt_tokens}→{completion_tokens} tokens)"
                )

                # Parse JSON using shared utility
                from modules.json_utils import extract_json_from_claude_response
                json_output = extract_json_from_claude_response(response_text, ticker)

                if not json_output:
                    raise ValueError("Failed to extract valid JSON from Gemini response")

                # Verify required structure
                if "sections" not in json_output:
                    raise ValueError("JSON missing 'sections' key")

                LOG.info(f"[{ticker}] ✅ Gemini 2.5 Pro: Valid JSON with {len(json_output.get('sections', {}))} sections")

                return {
                    "json_output": json_output,
                    "model_used": "gemini-2.5-pro",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "generation_time_ms": generation_time_ms
                }

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                    LOG.warning(
                        f"[{ticker}] Gemini attempt {attempt + 1}/{max_retries + 1} failed: {str(e)[:200]}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    LOG.error(f"[{ticker}] ❌ Gemini failed after {max_retries + 1} attempts: {str(e)[:200]}")
                    LOG.error(traceback.format_exc())

        # All retries exhausted
        return None

    except Exception as e:
        LOG.error(f"[{ticker}] ❌ Gemini setup error: {str(e)[:200]}")
        LOG.error(traceback.format_exc())
        return None


def _generate_transcript_claude_v2(
    ticker: str,
    content: str,
    config: Dict,
    content_type: str,
    anthropic_api_key: str,
    max_retries: int = 2
) -> Optional[Dict]:
    """
    Generate transcript JSON summary using Claude Sonnet 4.5 with v2 prompt.

    Pattern copied from executive_summary_phase1._generate_phase1_claude().
    Includes prompt caching for cost optimization.

    Args:
        ticker: Company ticker symbol
        content: Transcript or press release text
        config: Ticker configuration dict
        content_type: 'transcript' or 'press_release'
        anthropic_api_key: Anthropic API key
        max_retries: Maximum retry attempts (default: 2)

    Returns:
        Dict with same structure as Gemini, or None if generation fails
    """
    import time
    import anthropic

    try:
        # Initialize Claude client
        client = anthropic.Anthropic(api_key=anthropic_api_key)

        # Load v2 prompt (static, cacheable)
        system_prompt = load_transcript_prompt_v2()

        # Build ticker-specific user content
        user_content = _build_transcript_user_content_v2(
            ticker, content, config, content_type
        )

        # Retry logic for transient errors
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()

                # Call Claude with prompt caching
                response = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=16000,
                    temperature=0.0,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"}  # Enable prompt caching
                        }
                    ],
                    messages=[
                        {
                            "role": "user",
                            "content": user_content
                        }
                    ]
                )

                generation_time_ms = int((time.time() - start_time) * 1000)

                # Extract response text
                response_text = response.content[0].text

                # Get token counts
                prompt_tokens = response.usage.input_tokens
                completion_tokens = response.usage.output_tokens

                # Log cache performance
                cache_read_tokens = getattr(response.usage, 'cache_read_input_tokens', 0)
                cache_creation_tokens = getattr(response.usage, 'cache_creation_input_tokens', 0)

                if cache_read_tokens > 0:
                    LOG.info(
                        f"[{ticker}] Claude cache HIT: {cache_read_tokens} tokens read "
                        f"(~${cache_read_tokens * 0.0003 / 1000:.4f} saved)"
                    )

                LOG.info(
                    f"[{ticker}] Claude Sonnet 4.5 generated response "
                    f"({generation_time_ms}ms, {prompt_tokens}→{completion_tokens} tokens)"
                )

                # Parse JSON using shared utility
                from modules.json_utils import extract_json_from_claude_response
                json_output = extract_json_from_claude_response(response_text, ticker)

                if not json_output:
                    raise ValueError("Failed to extract valid JSON from Claude response")

                # Verify required structure
                if "sections" not in json_output:
                    raise ValueError("JSON missing 'sections' key")

                LOG.info(f"[{ticker}] ✅ Claude Sonnet 4.5: Valid JSON with {len(json_output.get('sections', {}))} sections")

                return {
                    "json_output": json_output,
                    "model_used": "claude-sonnet-4-5-20250929",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "generation_time_ms": generation_time_ms
                }

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                    LOG.warning(
                        f"[{ticker}] Claude attempt {attempt + 1}/{max_retries + 1} failed: {str(e)[:200]}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    LOG.error(f"[{ticker}] ❌ Claude failed after {max_retries + 1} attempts: {str(e)[:200]}")
                    LOG.error(traceback.format_exc())

        # All retries exhausted
        return None

    except Exception as e:
        LOG.error(f"[{ticker}] ❌ Claude setup error: {str(e)[:200]}")
        LOG.error(traceback.format_exc())
        return None


def generate_transcript_json_with_fallback(
    ticker: str,
    content: str,
    config: Dict,
    content_type: str,
    anthropic_api_key: str,
    gemini_api_key: str = None
) -> Optional[Dict]:
    """
    Generate transcript JSON with Gemini 2.5 Pro (primary) and Claude Sonnet 4.5 (fallback).

    This is the main entry point for v2 transcript generation. Pattern copied from
    executive_summary_phase1.generate_executive_summary_phase1().

    Args:
        ticker: Company ticker symbol
        content: Transcript or press release text
        config: Ticker configuration dict
        content_type: 'transcript' or 'press_release'
        anthropic_api_key: Anthropic API key (required for fallback)
        gemini_api_key: Gemini API key (optional, but recommended)

    Returns:
        Dict with:
            - json_output: Full 15-section JSON structure
            - model_used: 'gemini-2.5-pro' or 'claude-sonnet-4-5-20250929'
            - prompt_tokens: Input token count
            - completion_tokens: Output token count
            - generation_time_ms: Processing time in milliseconds
        Or None if both providers fail
    """
    # Try Gemini first (if key provided)
    if gemini_api_key:
        LOG.info(f"[{ticker}] Transcript v2: Attempting Gemini 2.5 Pro (primary)")
        gemini_result = _generate_transcript_gemini_v2(
            ticker=ticker,
            content=content,
            config=config,
            content_type=content_type,
            gemini_api_key=gemini_api_key
        )

        if gemini_result and gemini_result.get("json_output"):
            LOG.info(f"[{ticker}] ✅ Transcript v2: Gemini 2.5 Pro succeeded")
            return gemini_result
        else:
            LOG.warning(f"[{ticker}] ⚠️ Transcript v2: Gemini 2.5 Pro failed, falling back to Claude Sonnet")
    else:
        LOG.warning(f"[{ticker}] ⚠️ No Gemini API key provided, using Claude Sonnet only")

    # Fall back to Claude (if key provided)
    if anthropic_api_key:
        LOG.info(f"[{ticker}] Transcript v2: Attempting Claude Sonnet 4.5 (fallback)")
        claude_result = _generate_transcript_claude_v2(
            ticker=ticker,
            content=content,
            config=config,
            content_type=content_type,
            anthropic_api_key=anthropic_api_key
        )

        if claude_result and claude_result.get("json_output"):
            LOG.info(f"[{ticker}] ✅ Transcript v2: Claude Sonnet 4.5 succeeded (fallback)")
            return claude_result
        else:
            LOG.error(f"[{ticker}] ❌ Transcript v2: Claude Sonnet also failed")
    else:
        LOG.error(f"[{ticker}] ❌ No Anthropic API key provided for fallback")

    # Both failed
    LOG.error(f"[{ticker}] ❌ Transcript v2: Both Gemini and Claude failed")
    return None


def generate_transcript_email_v2(
    ticker: str,
    json_output: Dict,
    config: Dict,
    content_type: str,
    quarter: int = None,
    year: int = None,
    report_date: str = None,
    stock_data: Dict = None
) -> Dict:
    """
    Generate email HTML from transcript JSON (v2 format).

    Uses same Jinja2 template as v1, but converts JSON to HTML instead of parsing markdown.

    Args:
        ticker: Company ticker symbol
        json_output: Full 15-section JSON from v2 generation
        config: Ticker configuration dict
        content_type: 'transcript' or 'press_release'
        quarter: Quarter number (1-4) for transcripts
        year: Year for transcripts
        report_date: Date string for press releases
        stock_data: Optional stock price data dict

    Returns:
        Dict with:
            - html: Full email HTML
            - subject: Email subject line
    """
    try:
        # Convert JSON to old dict format, then use proven old HTML builder
        sections_dict = convert_json_to_old_dict_format(json_output)
        summary_html = build_transcript_summary_html(sections_dict, content_type)

        # Get company metadata
        company_name = config.get("name", ticker)
        industry = config.get("industry", "")

        # Configure based on report type (matching old v1 function)
        if content_type == 'transcript':
            report_type_label = "EARNINGS CALL TRANSCRIPT"
            quarter_str = f"Q{quarter}" if quarter else ""
            fiscal_period = f"{quarter_str} {year}" if quarter and year else ""
            date_label = f"Call Date: {report_date}" if report_date else ""
            filing_date_display = f"Call Date: {report_date}" if report_date else ""
            transition_text = f"Summary generated from {company_name} {quarter_str} {year} earnings call transcript."
            fmp_link_text = "View full transcript on FMP"
            subject = f"📊 Earnings Call Summary: {company_name} ({ticker}) {quarter_str} {year}"
        else:  # press_release
            report_type_label = "PRESS RELEASE"
            fiscal_period = report_date if report_date else ""
            date_label = f"Release Date: {report_date}" if report_date else ""
            filing_date_display = f"Release Date: {report_date}" if report_date else ""
            transition_text = f"Summary generated from {company_name} press release dated {report_date}."
            fmp_link_text = "View original release on FMP"
            subject = f"📰 Press Release: {ticker} - {report_date}"

        # Build FMP link box HTML (append to content like old function)
        fmp_link_html = f'''
    <div style="margin: 32px 0 20px 0; padding: 12px 16px; background-color: #eff6ff; border-left: 4px solid #1e40af; border-radius: 4px;">
        <p style="margin: 0; font-size: 12px; color: #1e40af; font-weight: 600; line-height: 1.4;">
            {transition_text} <a href="https://financialmodelingprep.com" style="color: #1e40af; text-decoration: none;">→ {fmp_link_text}</a>
        </p>
    </div>'''

        # Combine summary HTML with FMP link
        content_html = summary_html + fmp_link_html

        # Get stock price data if available
        if not stock_data:
            # Import from app.py where the function is defined
            import sys
            import os
            # Add parent directory to path to import from app
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            try:
                from app import get_filing_stock_data
                stock_data = get_filing_stock_data(ticker)
            except ImportError:
                LOG.warning(f"[{ticker}] Could not import get_filing_stock_data, stock data will be None")
                stock_data = None

        # Prepare template variables (matching old v1 function)
        template_vars = {
            "report_title": f"{ticker} Research Summary",
            "report_type_label": report_type_label,
            "company_name": company_name,
            "ticker": ticker,
            "industry": None,  # Transcripts don't include industry
            "fiscal_period": fiscal_period,
            "date_label": date_label,
            "filing_date": None,  # Don't show duplicate date below ticker (already at top right)
            "content_html": content_html,
            # Stock price data (3-row card)
            "stock_price": stock_data.get("stock_price") if stock_data else None,
            "price_change_pct": stock_data.get("price_change_pct") if stock_data else None,  # Fixed key name
            "price_change_color": stock_data.get("price_change_color") if stock_data else "#666",
            "ytd_return_pct": stock_data.get("ytd_return_pct") if stock_data else None,
            "ytd_return_color": stock_data.get("ytd_return_color") if stock_data else "#666",
            "market_status": stock_data.get("market_status", "closed") if stock_data else "closed",
            "return_label": stock_data.get("return_label", "1D") if stock_data else "1D",
        }

        # Load and render template
        import os
        template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template('email_research_report.html')

        html = template.render(**template_vars)

        LOG.info(f"[{ticker}] ✅ Generated email v2 ({len(html)} chars)")

        return {
            "html": html,
            "subject": subject
        }

    except Exception as e:
        LOG.error(f"[{ticker}] ❌ Failed to generate email v2: {str(e)[:200]}")
        LOG.error(traceback.format_exc())
        raise


# ==============================================================================
# MARKDOWN CONVERSION (For Research Page Display)
# ==============================================================================

def convert_json_to_markdown(
    json_output: Dict,
    content_type: str
) -> str:
    """
    Convert v2 JSON to clean markdown text for research page display.

    Format matches v1 output:
    - No bold markers (plain text)
    - Clean section headers (## SECTION NAME)
    - Simple bullets (- Topic: Content)
    - Q&A format: "Q: Name, Firm\\nQuestion\\n\\nA:\\nAnswer"

    This markdown is:
    1. Displayed in research page (summary_text column)
    2. Combined with 10-K/10-Q markdown elsewhere in app

    Args:
        json_output: Full JSON with 15 sections
        content_type: 'transcript' or 'press_release'

    Returns:
        Clean markdown string
    """
    sections_json = json_output.get("sections", {})
    md_parts = []

    # 1. Bottom Line (paragraph)
    if sections_json.get("bottom_line", {}).get("content"):
        md_parts.append("## BOTTOM LINE\n")
        md_parts.append(sections_json["bottom_line"]["content"] + "\n")

    # Helper for bullet sections
    def add_bullet_section(section_key: str, title: str):
        bullets = sections_json.get(section_key, [])
        if bullets:
            md_parts.append(f"\n## {title}\n\n")
            for bullet in bullets:
                topic = bullet.get("topic_label", "")
                content = bullet.get("content", "")
                # No bold: "- Topic: Content" (not "- **Topic:** Content")
                md_parts.append(f"- {topic}: {content}\n")

    # 2. Financial Results
    add_bullet_section("financial_results", "FINANCIAL RESULTS")

    # 3. Operational Metrics
    add_bullet_section("operational_metrics", "OPERATIONAL METRICS")

    # 4. Major Developments
    add_bullet_section("major_developments", "MAJOR DEVELOPMENTS")

    # 5. Guidance & Outlook
    add_bullet_section("guidance", "GUIDANCE & OUTLOOK")

    # 6. Strategic Initiatives
    add_bullet_section("strategic_initiatives", "STRATEGIC INITIATIVES")

    # 7. Management Sentiment & Tone
    add_bullet_section("management_sentiment_tone", "MANAGEMENT SENTIMENT & TONE")

    # 8. Risk Factors & Headwinds
    add_bullet_section("risk_factors", "RISK FACTORS & HEADWINDS")

    # 9. Industry & Competitive Landscape
    add_bullet_section("industry_competitive", "INDUSTRY & COMPETITIVE LANDSCAPE")

    # 10. Related Entities (special format: entity_type - company_name)
    entities = sections_json.get("related_entities", [])
    if entities:
        md_parts.append("\n## RELATED ENTITIES\n\n")
        for entity in entities:
            entity_type = entity.get("entity_type", "")
            company_name = entity.get("company_name", "")
            content = entity.get("content", "")
            # Format: "- Customer - Tesla: Signed 10-year offtake..."
            md_parts.append(f"- {entity_type} - {company_name}: {content}\n")

    # 11. Capital Allocation & Balance Sheet
    add_bullet_section("capital_allocation", "CAPITAL ALLOCATION & BALANCE SHEET")

    # 12. Q&A Highlights (special format, transcripts only)
    if content_type == 'transcript':
        qa_exchanges = sections_json.get("qa_highlights", [])
        if qa_exchanges:
            md_parts.append("\n## Q&A HIGHLIGHTS\n")
            for exchange in qa_exchanges:
                analyst_name = exchange.get("analyst_name", "")
                analyst_firm = exchange.get("analyst_firm", "")
                question = exchange.get("question", "")
                answer = exchange.get("answer", "")

                # V1 format (no bold in markdown):
                # Q: Name, Firm
                # Question text
                #
                # A:
                # Answer text
                md_parts.append(f"\nQ: {analyst_name}, {analyst_firm}\n")
                md_parts.append(f"{question}\n")
                md_parts.append(f"\nA:\n")
                md_parts.append(f"{answer}\n")

    # 13. Upside Scenario (paragraph)
    if sections_json.get("upside_scenario", {}).get("content"):
        md_parts.append("\n## UPSIDE SCENARIO\n\n")
        md_parts.append(sections_json["upside_scenario"]["content"] + "\n")

    # 14. Downside Scenario (paragraph)
    if sections_json.get("downside_scenario", {}).get("content"):
        md_parts.append("\n## DOWNSIDE SCENARIO\n\n")
        md_parts.append(sections_json["downside_scenario"]["content"] + "\n")

    # 15. Key Variables to Monitor
    add_bullet_section("key_variables", "KEY VARIABLES TO MONITOR")

    return "".join(md_parts)


def convert_json_to_old_dict_format(json_output: Dict) -> Dict[str, List[str]]:
    """
    Convert v2 JSON structure to old dict format for build_transcript_summary_html().

    JSON format:
      {"topic_label": "Revenue", "content": "$500M, up 20% YoY per CEO"}

    Old dict format:
      "Revenue: $500M, up 20% YoY per CEO"

    The old HTML builder's bold_bullet_labels() function will automatically
    bold the "Revenue:" part when rendering.

    Args:
        json_output: Full JSON with 15 sections

    Returns:
        Dict[str, List[str]] matching old format
    """
    sections_json = json_output.get("sections", {})
    old_format = {}

    # 1. Bottom Line (paragraph)
    bottom_line_content = sections_json.get("bottom_line", {}).get("content", "")
    old_format["bottom_line"] = [bottom_line_content] if bottom_line_content else []

    # Helper for bullet sections
    def convert_bullet_section(section_key: str) -> List[str]:
        bullets = []
        for bullet in sections_json.get(section_key, []):
            topic = bullet.get("topic_label", "")
            content = bullet.get("content", "")
            # Combine: "Topic: Content"
            bullets.append(f"{topic}: {content}")
        return bullets

    # 2-9. Standard bullet sections
    old_format["financial_results"] = convert_bullet_section("financial_results")
    old_format["operational_metrics"] = convert_bullet_section("operational_metrics")
    old_format["major_developments"] = convert_bullet_section("major_developments")
    old_format["guidance"] = convert_bullet_section("guidance")
    old_format["strategic_initiatives"] = convert_bullet_section("strategic_initiatives")
    old_format["management_sentiment"] = convert_bullet_section("management_sentiment_tone")
    old_format["risk_factors"] = convert_bullet_section("risk_factors")
    old_format["industry_competitive"] = convert_bullet_section("industry_competitive")

    # 10. Related Entities (special format: entity_type - company_name: content)
    entities = []
    for entity in sections_json.get("related_entities", []):
        entity_type = entity.get("entity_type", "")
        company_name = entity.get("company_name", "")
        content = entity.get("content", "")
        # Format: "Customer - Tesla: Signed 10-year offtake..."
        entities.append(f"{entity_type} - {company_name}: {content}")
    old_format["related_entities"] = entities

    # 11. Capital Allocation
    old_format["capital_allocation"] = convert_bullet_section("capital_allocation")

    # 12. Q&A Highlights (special format: Q and A on same line as content)
    qa_strings = []
    for exchange in sections_json.get("qa_highlights", []):
        analyst_name = exchange.get("analyst_name", "")
        analyst_firm = exchange.get("analyst_firm", "")
        question = exchange.get("question", "")
        answer = exchange.get("answer", "")

        # HTML builder expects: "Q: Name, Firm\nQuestion" and "A:\nAnswer" as single strings
        # But HTML collapses newlines, so put content on same line
        qa_strings.append(f"Q: {analyst_name}, {analyst_firm}\n{question}")
        qa_strings.append(f"A:\n{answer}")
    old_format["qa_highlights"] = qa_strings

    # 13-14. Scenarios (paragraphs)
    upside_content = sections_json.get("upside_scenario", {}).get("content", "")
    old_format["upside_scenario"] = [upside_content] if upside_content else []

    downside_content = sections_json.get("downside_scenario", {}).get("content", "")
    old_format["downside_scenario"] = [downside_content] if downside_content else []

    # 15. Key Variables
    old_format["key_variables"] = convert_bullet_section("key_variables")

    return old_format


# ==============================================================================
# SECTION PARSING
# ==============================================================================

def parse_transcript_summary_sections(summary_text: str, ticker: str = None) -> Dict[str, List[str]]:
    """
    Parse transcript summary text into sections by markdown headers.
    Handles special Q&A format (Q:/A: paragraphs) and top-level Upside/Downside/Variables sections.
    Uses partial matching for header flexibility (e.g., "## MANAGEMENT SENTIMENT" matches "## MANAGEMENT SENTIMENT & TONE").

    Args:
        summary_text: Raw summary text from AI
        ticker: Optional ticker for debug logging

    Returns dict: {section_name: [line1, line2, ...]}
    """
    def normalize_header(text: str) -> str:
        """
        Normalize header for flexible matching.
        Strips: ##, emojis, leading/trailing spaces
        Returns uppercase for case-insensitive matching.

        Examples:
        - "## BOTTOM LINE" → "BOTTOM LINE"
        - "BOTTOM LINE" → "BOTTOM LINE"
        - "📌 BOTTOM LINE" → "BOTTOM LINE"
        - "## 📌 BOTTOM LINE" → "BOTTOM LINE"
        """
        # Remove leading non-alphanumeric characters (##, emojis, punctuation)
        text = re.sub(r'^[^\w\s]+\s*', '', text)
        # Remove all non-ASCII characters (emojis)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        # Strip and uppercase
        return text.strip().upper()

    sections = {
        "bottom_line": [],
        "financial_results": [],
        "major_developments": [],
        "operational_metrics": [],
        "guidance": [],
        "strategic_initiatives": [],
        "management_sentiment": [],
        "risk_factors": [],
        "industry_competitive": [],
        "related_entities": [],
        "capital_allocation": [],
        "qa_highlights": [],
        "upside_scenario": [],
        "downside_scenario": [],
        "key_variables": []
    }

    if not summary_text:
        return sections

    # Split by markdown headers (updated Nov 2025 - markdown format)
    # Uses partial matching for flexibility with AI variations
    # Order matches prompt structure: Operational before Major Developments
    section_markers = [
        ("## BOTTOM LINE", "bottom_line"),
        ("## FINANCIAL RESULTS", "financial_results"),
        ("## OPERATIONAL METRICS", "operational_metrics"),
        ("## MAJOR DEVELOPMENTS", "major_developments"),
        ("## GUIDANCE", "guidance"),
        ("## STRATEGIC INITIATIVES", "strategic_initiatives"),
        ("## MANAGEMENT SENTIMENT", "management_sentiment"),      # Matches with or without "& TONE"
        ("## RISK FACTORS", "risk_factors"),                      # Matches with or without "& HEADWINDS"
        ("## INDUSTRY", "industry_competitive"),                  # Matches "## INDUSTRY & COMPETITIVE LANDSCAPE"
        ("## RELATED ENTITIES", "related_entities"),              # Matches with or without parenthetical
        ("## CAPITAL ALLOCATION", "capital_allocation"),          # Matches with or without "& BALANCE SHEET"
        ("## Q&A", "qa_highlights"),                              # Matches both "## Q&A" and "## Q&A HIGHLIGHTS"
        ("## UPSIDE SCENARIO", "upside_scenario"),
        ("## DOWNSIDE SCENARIO", "downside_scenario"),
        ("## KEY VARIABLES", "key_variables")                     # Matches with or without "TO MONITOR"
    ]

    current_section = None
    # Normalize section marker prefixes for flexible matching
    section_marker_prefixes = tuple(normalize_header(marker) for marker, _ in section_markers)

    # Q&A state tracking (for grouping Q/A pairs)
    qa_state = None  # None, 'Q', or 'A'
    qa_current_q = []  # Lines accumulating for current question
    qa_current_a = []  # Lines accumulating for current answer

    for line in summary_text.split('\n'):
        line_stripped = line.strip()

        # Skip horizontal rule separators (Claude sometimes adds these)
        if line_stripped == '---':
            continue

        # Check if line is a section header (normalized matching)
        is_header = False
        line_normalized = normalize_header(line_stripped)

        for marker, section_key in section_markers:
            # Normalize both sides for flexible matching
            marker_normalized = normalize_header(marker)

            if line_normalized.startswith(marker_normalized):
                # Flush any pending Q&A content before switching sections
                if qa_state == 'Q' and qa_current_q:
                    sections['qa_highlights'].append('\n'.join(qa_current_q))
                    qa_current_q = []
                elif qa_state == 'A' and qa_current_a:
                    sections['qa_highlights'].append('\n'.join(qa_current_a))
                    qa_current_a = []
                # Reset Q&A state when leaving qa_highlights section
                if current_section == 'qa_highlights' and section_key != 'qa_highlights':
                    qa_state = None

                current_section = section_key
                is_header = True

                # NEW: Extract content after header if on same line (Gemini format)
                # Find where actual content starts (after header in original line)
                content_after_header = line_stripped
                # Strip the header portion (try both normalized and original marker)
                for prefix_to_remove in [marker, marker_normalized, line_stripped.split()[0]]:
                    if content_after_header.startswith(prefix_to_remove):
                        content_after_header = content_after_header[len(prefix_to_remove):].strip()
                        break

                if content_after_header:
                    # Handle different section types
                    if current_section == 'qa_highlights':
                        # Q&A section - content on same line as header is rare, but handle it
                        # Check if it's a Q: or A: line
                        if content_after_header.startswith('Q:'):
                            qa_state = 'Q'
                            qa_current_q = [content_after_header]
                        elif content_after_header.startswith('A:'):
                            qa_state = 'A'
                            qa_current_a = [content_after_header]
                        # Skip content like "HIGHLIGHTS" from "## Q&A HIGHLIGHTS" header
                        # (Only process actual Q:/A: content)
                    elif current_section in ['bottom_line', 'upside_scenario', 'downside_scenario']:
                        # Paragraph sections - capture all text
                        sections[current_section].append(content_after_header)
                    else:
                        # Bullet sections - split multi-bullet lines (Fix #2)
                        # Only split on • and * (NOT on - to preserve ranges like "$100M - $120M")
                        # If starts with -, treat entire line as one bullet
                        if content_after_header.lstrip().startswith('-'):
                            # Don't split lines starting with - (preserves internal dashes)
                            bullet_text = content_after_header.lstrip('- ').strip()
                            if bullet_text:
                                sections[current_section].append(bullet_text)
                        else:
                            # Split by • and * only: "• A • B" → ["• A", "• B"]
                            bullet_parts = re.split(r'(?=[•\*]\s)', content_after_header)
                            for bullet_part in bullet_parts:
                                if bullet_part.strip().startswith(('•', '*')):
                                    bullet_text = bullet_part.lstrip('•* ').strip()
                                    if bullet_text:
                                        sections[current_section].append(bullet_text)

                break

        if not is_header and current_section:
            # Line is content, not a header

            # Special Q&A handling (accumulate into Q/A pairs)
            if current_section == 'qa_highlights':
                # Skip lines that start with section markers (normalized check)
                if not line_normalized.startswith(section_marker_prefixes):
                    # Detect Q: or A: line starts
                    if line_stripped.startswith('Q:'):
                        # Save previous A if we were accumulating one
                        if qa_state == 'A' and qa_current_a:
                            # Join answer lines with newlines
                            sections['qa_highlights'].append('\n'.join(qa_current_a))
                            qa_current_a = []
                        # Start new question
                        qa_state = 'Q'
                        qa_current_q = [line_stripped]  # Start with "Q: Name, Firm"
                    elif line_stripped.startswith('A:'):
                        # Save question and start answer
                        if qa_state == 'Q' and qa_current_q:
                            # Join question lines with newlines
                            sections['qa_highlights'].append('\n'.join(qa_current_q))
                            qa_current_q = []
                        # Start new answer
                        qa_state = 'A'
                        qa_current_a = [line_stripped]  # Start with "A:"
                    else:
                        # Regular content line - append to current Q or A
                        if qa_state == 'Q' and line_stripped:
                            qa_current_q.append(line_stripped)
                        elif qa_state == 'A' and line_stripped:
                            qa_current_a.append(line_stripped)
                        # Skip blank lines (spacing handled by HTML builder)

            # Other paragraph sections (bottom_line, upside_scenario, downside_scenario)
            elif current_section in ['bottom_line', 'upside_scenario', 'downside_scenario']:
                # Skip lines that start with section markers (normalized check)
                if not line_normalized.startswith(section_marker_prefixes):
                    # Skip empty lines at start, but keep them once content exists
                    if line_stripped or sections[current_section]:
                        sections[current_section].append(line_stripped)

            # Standard handling for bullet sections
            else:
                # Accept multiple bullet formats: •, -, *
                if line_stripped.startswith(('•', '-', '*', '• ', '- ', '* ')):
                    # Only split on • and * (NOT on - to preserve ranges like "$100M - $120M")
                    # If starts with -, treat entire line as one bullet
                    if line_stripped.startswith(('-', '- ')):
                        # Don't split lines starting with - (preserves internal dashes)
                        bullet_text = line_stripped.lstrip('- ').strip()
                        if bullet_text:
                            sections[current_section].append(bullet_text)
                    else:
                        # Split by • and * only: "• A • B • C" → ["A", "B", "C"]
                        bullet_parts = re.split(r'(?=[•\*]\s)', line_stripped)
                        for bullet_part in bullet_parts:
                            if bullet_part.strip().startswith(('•', '*')):
                                bullet_text = bullet_part.lstrip('•* ').strip()
                                if bullet_text:
                                    sections[current_section].append(bullet_text)
                elif line.startswith('  ') and sections[current_section]:
                    # Indented continuation line (e.g., "  Context: ...")
                    continuation = line_stripped
                    if continuation:
                        # Append to the last bullet with a line break
                        sections[current_section][-1] += '\n' + continuation

    # Flush any remaining Q&A content at end of parsing
    if qa_state == 'Q' and qa_current_q:
        # Question without answer (edge case)
        sections['qa_highlights'].append('\n'.join(qa_current_q))
    elif qa_state == 'A' and qa_current_a:
        # Final answer
        sections['qa_highlights'].append('\n'.join(qa_current_a))

    # Debug logging: Check if parser captured content
    total_items = sum(len(v) for v in sections.values())
    ticker_label = f"[{ticker}]" if ticker else ""
    if total_items == 0:
        LOG.warning(f"{ticker_label} Parser captured ZERO items. First 500 chars of raw text: {summary_text[:500]}")
    else:
        LOG.info(f"{ticker_label} Parser captured {total_items} total items across all sections")

    return sections


def build_transcript_summary_html(sections: Dict[str, List[str]], content_type: str) -> str:
    """
    Build HTML for transcript summary sections.
    Strips emojis from headers and bolds topic labels + Context.
    Used by email template generation.

    Args:
        sections: Parsed sections dict
        content_type: 'transcript' or 'press_release'
    """
    import re

    def strip_emoji(text: str) -> str:
        """Remove emoji characters from text"""
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001F900-\U0001F9FF"  # supplemental symbols
            u"\U00002600-\U000026FF"  # misc symbols
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub('', text).strip()

    def strip_markdown_formatting(text: str) -> str:
        """Strip markdown formatting (bold, italic) that AI sometimes adds"""
        text = re.sub(r'\*\*([^*]+?)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+?)__', r'\1', text)
        text = re.sub(r'\*([^*]+?)\*', r'\1', text)
        text = re.sub(r'_([^_]+?)_', r'\1', text)
        return text

    def bold_bullet_labels(text: str) -> str:
        """
        Bold topic labels in format 'Topic Label: Details'
        Also bolds 'Context:' when it appears in 10-K enrichment lines.
        """
        text = strip_markdown_formatting(text)
        pattern = r'^([^:]{2,130}?:)(\s)'
        replacement = r'<strong>\1</strong>\2'
        text = re.sub(pattern, replacement, text)

        # Bold "Context:" when it appears (10-K enrichment lines)
        text = text.replace('Context:', '<strong>Context:</strong>')

        return text

    def build_section(title: str, bullets: List[str], use_bullets: bool = True, bold_labels: bool = False) -> str:
        """Helper to build a section with title and content"""
        if not bullets:
            return ""

        # Always strip emojis from title
        display_title = strip_emoji(title)

        html = f'<div style="margin-bottom: 24px;">\n'
        html += f'  <h3 style="font-size: 15px; font-weight: 700; color: #1e40af; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 0.5px;">{display_title}</h3>\n'

        if use_bullets:
            html += '  <ul style="margin: 0; padding-left: 20px; color: #374151; font-size: 13px; line-height: 1.6;">\n'
            for bullet in bullets:
                # Apply label bolding if requested
                processed_bullet = bold_bullet_labels(bullet) if bold_labels else bullet
                html += f'    <li style="margin-bottom: 6px;">{processed_bullet}</li>\n'
            html += '  </ul>\n'
        else:
            # Paragraph format (for bottom_line, qa_highlights, upside/downside)
            # Strip markdown from each line before joining
            content_filtered = [strip_markdown_formatting(line) for line in bullets if line.strip()]
            content = '<br>'.join(content_filtered)
            html += f'  <div style="color: #374151; font-size: 13px; line-height: 1.6;">{content}</div>\n'

        html += '</div>\n'
        return html

    def build_qa_section(qa_content: List[str]) -> str:
        """Build Q&A section with special Q:/A: formatting (Q bold, proper spacing)"""
        if not qa_content:
            return ""

        display_title = strip_emoji("💬 Q&A Highlights")

        html = f'<div style="margin-bottom: 24px;">\n'
        html += f'  <h3 style="font-size: 15px; font-weight: 700; color: #1e40af; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 0.5px;">{display_title}</h3>\n'

        for line in qa_content:
            line_stripped = line.strip()
            if line_stripped.startswith("Q:"):
                # Bold Q, no bottom margin (A will follow immediately)
                html += f'  <p style="margin: 0; font-size: 13px; line-height: 1.6; color: #374151;"><strong>{line_stripped}</strong></p>\n'
            elif line_stripped.startswith("A:"):
                # Regular A, with bottom margin (creates space before next Q)
                html += f'  <p style="margin: 0 0 16px 0; font-size: 13px; line-height: 1.6; color: #374151;">{line_stripped}</p>\n'
            elif not line_stripped:
                # Blank lines are ignored (spacing controlled by A margin)
                pass

        html += '</div>\n'
        return html

    html = ""

    # Always render sections in fixed order (clean headers without emojis)
    html += build_section("Bottom Line", sections.get("bottom_line", []), use_bullets=False)
    html += build_section("Financial Results", sections.get("financial_results", []), use_bullets=True, bold_labels=True)
    html += build_section("Major Developments", sections.get("major_developments", []), use_bullets=True, bold_labels=True)
    html += build_section("Operational Metrics", sections.get("operational_metrics", []), use_bullets=True, bold_labels=True)
    html += build_section("Guidance", sections.get("guidance", []), use_bullets=True, bold_labels=True)
    html += build_section("Strategic Initiatives", sections.get("strategic_initiatives", []), use_bullets=True, bold_labels=True)
    html += build_section("Management Sentiment & Tone", sections.get("management_sentiment", []), use_bullets=True, bold_labels=True)
    html += build_section("Risk Factors & Headwinds", sections.get("risk_factors", []), use_bullets=True, bold_labels=True)
    html += build_section("Industry & Competitive Dynamics", sections.get("industry_competitive", []), use_bullets=True, bold_labels=True)
    html += build_section("Related Entities", sections.get("related_entities", []), use_bullets=True, bold_labels=True)
    html += build_section("Capital Allocation", sections.get("capital_allocation", []), use_bullets=True, bold_labels=True)

    # Q&A Highlights (only for transcripts, special formatting)
    if content_type == 'transcript':
        html += build_qa_section(sections.get("qa_highlights", []))

    # Top-level Upside/Downside/Variables sections (Oct 2025 - promoted from sub-sections)
    # Upside/Downside are PARAGRAPHS, Variables are BULLETS
    html += build_section("Upside Scenario", sections.get("upside_scenario", []), use_bullets=False, bold_labels=False)
    html += build_section("Downside Scenario", sections.get("downside_scenario", []), use_bullets=False, bold_labels=False)
    html += build_section("Key Variables to Monitor", sections.get("key_variables", []), use_bullets=True, bold_labels=True)

    return html


# ==============================================================================
# EMAIL GENERATION (v2 only - see generate_transcript_email_v2 above)
# ==============================================================================

# v1 function deleted - all emails now use generate_transcript_email_v2()


# ==============================================================================
# DATABASE OPERATIONS
# ==============================================================================

def save_transcript_summary_to_database(
    ticker: str,
    company_name: str,
    report_type: str,
    quarter: str,
    year: int,
    report_date: str,
    pr_title: str,
    summary_text: str,
    source_url: str,
    ai_provider: str,
    ai_model: str,
    generation_time_seconds: int,
    token_count_input: int,
    token_count_output: int,
    db_connection
) -> None:
    """Save transcript summary to database"""
    LOG.info(f"Saving transcript summary for {ticker} ({ai_model}) to database")

    try:
        cur = db_connection.cursor()

        cur.execute("""
            INSERT INTO transcript_summaries (
                ticker, company_name, report_type, quarter, year,
                report_date, pr_title, summary_text, source_url,
                ai_provider, ai_model, processing_duration_seconds
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, report_type, quarter, year)
            DO UPDATE SET
                summary_text = EXCLUDED.summary_text,
                ai_provider = EXCLUDED.ai_provider,
                ai_model = EXCLUDED.ai_model,
                processing_duration_seconds = EXCLUDED.processing_duration_seconds,
                generated_at = NOW()
        """, (
            ticker, company_name, report_type,
            quarter, year,
            report_date, pr_title,
            summary_text, source_url, ai_provider, ai_model, generation_time_seconds
        ))

        db_connection.commit()
        cur.close()

        LOG.info(f"✅ Saved transcript summary for {ticker} to database")

    except Exception as e:
        LOG.error(f"Failed to save transcript summary for {ticker}: {e}")
        raise
