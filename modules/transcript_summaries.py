# modules/transcript_summaries.py

"""
Transcript Summaries Module

Handles earnings call transcripts and press release summarization using FMP API and Claude AI.
Extracted from app.py for better modularity.
"""

import requests
import logging
import traceback
import re
from typing import Dict, List, Optional
from datetime import datetime, timezone
import pytz
from jinja2 import Environment, FileSystemLoader

LOG = logging.getLogger(__name__)


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
    Fetch specific press release by date.
    target_date format: 'YYYY-MM-DD HH:MM:SS'
    """
    releases = fetch_fmp_press_releases(ticker, fmp_api_key, limit=50)

    for release in releases:
        if release.get('date') == target_date:
            return release

    LOG.warning(f"Press release not found for {ticker} on {target_date}")
    return None


# ==============================================================================
# PROMPTS
# ==============================================================================

GEMINI_TRANSCRIPT_PROMPT = """You are extracting key information from an earnings call transcript for {company_name} ({ticker}) - Q{quarter} {fiscal_year}.

Summarize the transcript into these sections:

## 0. BOTTOM LINE
## 1. FINANCIAL RESULTS
## 2. PERFORMANCE VS EXPECTATIONS
## 3. SEGMENT PERFORMANCE
## 4. OPERATIONAL METRICS & KPIS
## 5. GUIDANCE UPDATES
## 6. STRATEGIC ANNOUNCEMENTS
## 7. CAPITAL ALLOCATION & BALANCE SHEET
## 8. MANAGEMENT COMMENTARY ON OUTLOOK
## 9. KEY RISKS & CHALLENGES DISCUSSED
## 10. ANALYST QUESTIONS & CONCERNS
## 11. INVESTMENT IMPLICATIONS

For each section, extract the most important information discussed. Use bullet points where appropriate. If a section is not discussed on the call, write "Not discussed."

Target length: 2,000-4,000 words.

---
TRANSCRIPT:
{transcript_text}
---

Generate the earnings call summary now."""


# ==============================================================================
# AI SUMMARIZATION
# ==============================================================================

def generate_transcript_summary_with_gemini(
    ticker: str,
    content: str,
    config: Dict,
    quarter: int,
    fiscal_year: int,
    gemini_api_key: str
) -> Optional[Dict]:
    """
    Generate transcript summary using Gemini 2.5 Flash (2-4k words).
    Returns dict with {summary_text, generation_time_seconds, token_count_input, token_count_output}
    or None if failed.
    """
    if not gemini_api_key:
        LOG.error("Gemini API key not configured")
        return None

    try:
        import time
        import google.generativeai as genai

        start_time = time.time()

        # Configure Gemini
        genai.configure(api_key=gemini_api_key)

        # Build prompt
        company_name = config.get('company_name', ticker)
        prompt = GEMINI_TRANSCRIPT_PROMPT.format(
            company_name=company_name,
            ticker=ticker,
            quarter=quarter,
            fiscal_year=fiscal_year,
            transcript_text=content
        )

        LOG.info(f"Generating transcript summary for {ticker} Q{quarter} {fiscal_year} using Gemini 2.5 Flash")

        # Configure model
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            generation_config={
                'temperature': 0.0,  # Maximum determinism for completely consistent transcripts
                'max_output_tokens': 16000,  # Allow up to 4k words
            }
        )

        # Generate summary
        response = model.generate_content(prompt)

        if not response or not response.text:
            LOG.error(f"Gemini returned empty response for {ticker}")
            return None

        summary_text = response.text.strip()
        generation_time = time.time() - start_time

        # Extract token counts (if available)
        token_count_input = 0
        token_count_output = 0
        if hasattr(response, 'usage_metadata'):
            token_count_input = getattr(response.usage_metadata, 'prompt_token_count', 0)
            token_count_output = getattr(response.usage_metadata, 'candidates_token_count', 0)

        word_count = len(summary_text.split())

        LOG.info(f"✅ Generated Gemini summary for {ticker} Q{quarter} {fiscal_year}")
        LOG.info(f"   Words: {word_count}, Time: {generation_time:.1f}s")
        LOG.info(f"   Tokens: in={token_count_input}, out={token_count_output}")

        # Strip emoji from summary (Phase 2 needs clean section headers for searching)
        summary_text = strip_emoji(summary_text)

        return {
            'summary_text': summary_text,
            'generation_time_seconds': int(generation_time),
            'token_count_input': token_count_input,
            'token_count_output': token_count_output
        }

    except Exception as e:
        LOG.error(f"Failed to generate Gemini summary for {ticker}: {e}")
        LOG.error(f"Stacktrace: {traceback.format_exc()}")
        return None


def generate_transcript_summary_with_claude(
    ticker: str,
    content: str,
    config: Dict,
    content_type: str,  # 'transcript' or 'press_release'
    anthropic_api_key: str,
    anthropic_model: str,
    anthropic_api_url: str,
    build_prompt_func  # Reference to _build_research_summary_prompt from app.py
) -> Optional[str]:
    """
    Generate transcript summary using Claude API with prompt caching.
    Returns summary text or None if failed.
    """
    if not anthropic_api_key:
        LOG.error("Claude API key not configured")
        return None

    try:
        # Build prompt using app.py helper (keeps prompt logic centralized)
        system_prompt, user_content, company_name = build_prompt_func(
            ticker, content, config, content_type
        )

        if not system_prompt or not user_content:
            LOG.error(f"Failed to build transcript summary prompt for {ticker}")
            return None

        LOG.info(f"Generating {content_type} summary for {ticker} using Claude (prompt caching enabled)")

        headers = {
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01",  # Prompt caching support
            "content-type": "application/json"
        }

        data = {
            "model": anthropic_model,
            "max_tokens": 16000,  # Allow long summaries (transcripts can be 3-4k words)
            "temperature": 0.0,  # Maximum determinism for completely consistent financial analysis
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}  # Cache the system prompt (~5500 tokens)
                }
            ],
            "messages": [{"role": "user", "content": user_content}]
        }

        response = requests.post(anthropic_api_url, headers=headers, json=data, timeout=(10, 300))  # 5 min timeout

        if response.status_code != 200:
            LOG.error(f"Claude API error {response.status_code}: {response.text[:200]}")
            return None

        result = response.json()
        summary = result.get("content", [{}])[0].get("text", "")

        if not summary:
            LOG.warning(f"Claude returned empty summary for {ticker}")
            return None

        # Log token usage
        usage = result.get("usage", {})
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        if cache_read > 0:
            LOG.info(f"✅ Prompt cache hit: {cache_read} tokens read from cache (90% cost savings)")
        elif cache_creation > 0:
            LOG.info(f"📝 Prompt cache created: {cache_creation} tokens cached for future use")

        LOG.info(f"✅ Generated {content_type} summary for {ticker} ({len(summary)} chars)")
        LOG.info(f"💵 Tokens: in={input_tokens}, out={output_tokens}, cached={cache_read}")

        # Strip emoji from summary (Phase 2 needs clean section headers for searching)
        summary = strip_emoji(summary)

        return summary

    except Exception as e:
        LOG.error(f"Failed to generate transcript summary for {ticker}: {e}")
        LOG.error(f"Stacktrace: {traceback.format_exc()}")
        return None


# ==============================================================================
# SECTION PARSING
# ==============================================================================

def parse_transcript_summary_sections(summary_text: str) -> Dict[str, List[str]]:
    """
    Parse transcript summary text into sections by emoji headers.
    Handles special Q&A format (Q:/A: paragraphs) and top-level Upside/Downside/Variables sections.
    Returns dict: {section_name: [line1, line2, ...]}
    """
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
        "capital_allocation": [],
        "qa_highlights": [],
        "upside_scenario": [],
        "downside_scenario": [],
        "key_variables": []
    }

    if not summary_text:
        return sections

    # Split by emoji headers (updated Oct 2025 - new section flow)
    # Order matches new prompt structure: Operational before Major Developments
    section_markers = [
        ("📌 BOTTOM LINE", "bottom_line"),
        ("💰 FINANCIAL RESULTS", "financial_results"),
        ("📊 OPERATIONAL METRICS", "operational_metrics"),
        ("🏢 MAJOR DEVELOPMENTS", "major_developments"),
        ("📈 GUIDANCE", "guidance"),
        ("🎯 STRATEGIC INITIATIVES", "strategic_initiatives"),
        ("💼 MANAGEMENT SENTIMENT", "management_sentiment"),
        ("⚠️ RISK FACTORS", "risk_factors"),
        ("🏭 INDUSTRY", "industry_competitive"),  # Matches "INDUSTRY & COMPETITIVE LANDSCAPE"
        ("💡 CAPITAL ALLOCATION", "capital_allocation"),
        ("💬 Q&A HIGHLIGHTS", "qa_highlights"),
        ("📈 UPSIDE SCENARIO", "upside_scenario"),
        ("📉 DOWNSIDE SCENARIO", "downside_scenario"),
        ("🔍 KEY VARIABLES TO MONITOR", "key_variables")
    ]

    current_section = None
    section_marker_prefixes = tuple(marker for marker, _ in section_markers)

    for line in summary_text.split('\n'):
        line_stripped = line.strip()

        # Skip horizontal rule separators (Claude sometimes adds these)
        if line_stripped == '---':
            continue

        # Check if line is a section header
        is_header = False
        for marker, section_key in section_markers:
            if line_stripped.startswith(marker):
                current_section = section_key
                is_header = True
                break

        if not is_header and current_section:
            # Line is content, not a header

            # Special handling for sections that capture ALL text (paragraphs)
            if current_section in ['bottom_line', 'qa_highlights', 'upside_scenario', 'downside_scenario']:
                # Skip lines that start with section markers
                if not line_stripped.startswith(section_marker_prefixes):
                    # Skip empty lines at start, but keep them once content exists
                    if line_stripped or sections[current_section]:
                        sections[current_section].append(line_stripped)

            # Standard handling for bullet sections
            else:
                if line_stripped.startswith('•') or line_stripped.startswith('-'):
                    # Extract bullet text
                    bullet_text = line_stripped.lstrip('•- ').strip()
                    if bullet_text:
                        sections[current_section].append(bullet_text)
                elif line.startswith('  ') and sections[current_section]:
                    # Indented continuation line (e.g., "  Context: ...")
                    continuation = line_stripped
                    if continuation:
                        # Append to the last bullet with a line break
                        sections[current_section][-1] += '\n' + continuation

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

    # Always render sections in fixed order (emojis stripped automatically)
    html += build_section("📌 Bottom Line", sections.get("bottom_line", []), use_bullets=False)
    html += build_section("💰 Financial Results", sections.get("financial_results", []), use_bullets=True, bold_labels=True)
    html += build_section("🏢 Major Developments", sections.get("major_developments", []), use_bullets=True, bold_labels=True)
    html += build_section("📊 Operational Metrics", sections.get("operational_metrics", []), use_bullets=True, bold_labels=True)
    html += build_section("📈 Guidance", sections.get("guidance", []), use_bullets=True, bold_labels=True)
    html += build_section("🎯 Strategic Initiatives", sections.get("strategic_initiatives", []), use_bullets=True, bold_labels=True)
    html += build_section("💼 Management Sentiment & Tone", sections.get("management_sentiment", []), use_bullets=True, bold_labels=True)
    html += build_section("⚠️ Risk Factors & Headwinds", sections.get("risk_factors", []), use_bullets=True, bold_labels=True)
    html += build_section("🏭 Industry & Competitive Dynamics", sections.get("industry_competitive", []), use_bullets=True, bold_labels=True)
    html += build_section("💡 Capital Allocation", sections.get("capital_allocation", []), use_bullets=True, bold_labels=True)

    # Q&A Highlights (only for transcripts, special formatting)
    if content_type == 'transcript':
        html += build_qa_section(sections.get("qa_highlights", []))

    # Top-level Upside/Downside/Variables sections (Oct 2025 - promoted from sub-sections)
    # Upside/Downside are PARAGRAPHS, Variables are BULLETS
    html += build_section("📈 Upside Scenario", sections.get("upside_scenario", []), use_bullets=False, bold_labels=False)
    html += build_section("📉 Downside Scenario", sections.get("downside_scenario", []), use_bullets=False, bold_labels=False)
    html += build_section("🔍 Key Variables to Monitor", sections.get("key_variables", []), use_bullets=True, bold_labels=True)

    return html


# ==============================================================================
# EMAIL GENERATION
# ==============================================================================

def generate_transcript_email(
    ticker: str,
    company_name: str,
    report_type: str,  # 'transcript' or 'press_release'
    quarter: str = None,  # 'Q3' for transcripts, None for PRs
    year: int = None,
    report_date: str = None,  # 'Oct 25, 2024'
    pr_title: str = None,
    summary_text: str = None,
    fmp_url: str = None,
    stock_price: str = "$0.00",
    price_change_pct: str = None,
    price_change_color: str = "#4ade80"
) -> Dict[str, str]:
    """
    Generate transcript/press release email HTML using unified Jinja2 template.

    Returns:
        {
            "html": Full email HTML string,
            "subject": Email subject line
        }
    """
    LOG.info(f"Generating transcript email for {ticker} ({report_type}) using unified template")

    # Parse sections from summary text
    sections = parse_transcript_summary_sections(summary_text)

    # Build summary HTML
    summary_html = build_transcript_summary_html(sections, report_type)

    current_date = datetime.now(timezone.utc).astimezone(pytz.timezone('US/Eastern')).strftime("%b %d, %Y")

    # Configure based on report type
    if report_type == 'transcript':
        report_type_label = "EARNINGS CALL TRANSCRIPT"
        fiscal_period = f"{quarter} {year}"
        date_label = f"Generated: {current_date} | Call Date: {report_date}"
        filing_date_display = f"Call Date: {report_date}"
        transition_text = f"Summary generated from {company_name} {quarter} {year} earnings call transcript."
        fmp_link_text = "View full transcript on FMP"
    else:  # press_release
        report_type_label = "PRESS RELEASE"
        fiscal_period = report_date
        date_label = f"Generated: {current_date} | Release Date: {report_date}"
        filing_date_display = f"Release Date: {report_date}"
        transition_text = f"Summary generated from {company_name} press release dated {report_date}."
        fmp_link_text = "View original release on FMP"

    # Build FMP link box HTML
    fmp_link_html = f'''
    <div style="margin: 32px 0 20px 0; padding: 12px 16px; background-color: #eff6ff; border-left: 4px solid #1e40af; border-radius: 4px;">
        <p style="margin: 0; font-size: 12px; color: #1e40af; font-weight: 600; line-height: 1.4;">
            {transition_text} <a href="{fmp_url}" style="color: #1e40af; text-decoration: none;">→ {fmp_link_text}</a>
        </p>
    </div>'''

    # Combine summary HTML with FMP link
    content_html = summary_html + fmp_link_html

    # Render template with variables
    html = research_template.render(
        report_title=f"{ticker} Research Summary",
        report_type_label=report_type_label,
        company_name=company_name,
        ticker=ticker,
        industry=None,  # Transcripts don't include industry
        fiscal_period=fiscal_period,
        date_label=date_label,
        filing_date=filing_date_display,
        stock_price=stock_price,
        price_change_pct=price_change_pct,
        price_change_color=price_change_color,
        content_html=content_html
    )

    # Subject line
    if report_type == 'transcript':
        subject = f"📊 Earnings Call Summary: {company_name} ({ticker}) {quarter} {year}"
    else:
        subject = f"📰 Press Release Summary: {company_name} ({ticker}) - {report_date}"

    return {"html": html, "subject": subject}


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
