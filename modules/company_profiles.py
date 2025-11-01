# modules/company_profiles.py

"""
Company Profiles Module

Handles 10-K filing ingestion and company profile generation using Gemini 2.5 Flash AI.
"""

import google.generativeai as genai
import PyPDF2
import logging
import traceback
import os
from typing import Dict, Optional
from datetime import datetime, timezone
import pytz
import markdown
from jinja2 import Environment, FileSystemLoader

LOG = logging.getLogger(__name__)

# Initialize Jinja2 template environment
template_env = Environment(loader=FileSystemLoader('templates'))
research_template = template_env.get_template('email_research_report.html')

# ==============================================================================
# GEMINI PROMPTS (10-K and 10-Q)
# ==============================================================================

GEMINI_10K_PROMPT = """You are creating a Company Profile document for an equity analyst.

This profile will be used to provide context when analyzing news articles about the company.

I am providing you with the COMPLETE Form 10-K for {company_name} ({ticker}).

Your task is to extract and synthesize information from across the entire document to create a comprehensive profile following the structure below.

CRITICAL INSTRUCTIONS:
- Be specific, not generic (name suppliers, customers, exact figures)
- Use actual numbers with units
- Extract only facts explicitly stated in the filing
- Skip sections with no disclosed data
- Target length: 4-6 pages (~2,000-4,500 words)

---
COMPLETE 10-K DOCUMENT:

{full_10k_text}

---

Create a Company Profile in Markdown format with these sections:

# {company_name} ({ticker}) - COMPANY PROFILE

*Generated from Form 10-K filed {filing_date} for fiscal year ending {fiscal_year_end}*

## 1. INDUSTRY CLASSIFICATION
## 2. BUSINESS MODEL SUMMARY
## 3. REVENUE STREAMS
## 4. KEY OPERATIONAL METRICS (KPIs)
## 5. GEOGRAPHIC PRESENCE
## 6. KEY PRODUCTS & SERVICES
## 7. MATERIAL DEPENDENCIES
## 8. INFRASTRUCTURE & ASSETS
## 9. COST STRUCTURE
## 10. FINANCIAL SNAPSHOT (Latest Year)
## 11. SPECIFIC RISKS & CONCENTRATIONS
## 12. REGULATORY OVERSIGHT
## 13. STRATEGIC PRIORITIES & OUTLOOK
## 14. KEY THINGS TO MONITOR
## 15. MANAGEMENT & GOVERNANCE
## 16. LEGAL PROCEEDINGS & CONTINGENCIES
## 17. RECENT EVENTS & DEVELOPMENTS

OUTPUT FORMAT: Valid Markdown with proper headers, bullets, and tables.
"""

GEMINI_10Q_PROMPT = """You are creating a Quarterly Update document for an equity analyst.

I am providing you with the COMPLETE Form 10-Q for {company_name} ({ticker}).

Your task is to extract and synthesize information from across the entire document to create a comprehensive quarterly update following the structure below, focusing on quarter-over-quarter and year-over-year changes.

CRITICAL INSTRUCTIONS:
- Be specific, not generic (name exact figures, percentages, line items)
- Use actual numbers with units
- Extract only facts explicitly stated in the filing
- Show QoQ and YoY comparisons with percentage changes
- Skip sections with no disclosed data
- Target length: 4-7 pages (~2,000-5,000 words)

---
COMPLETE 10-Q DOCUMENT:

{full_10q_text}

---

Create a Quarterly Update in Markdown format with these sections:

# {company_name} ({ticker}) - QUARTERLY UPDATE

*Generated from Form 10-Q for Q{quarter} {fiscal_year} filed {filing_date}*

## 1. QUARTERLY FINANCIAL PERFORMANCE (QoQ & YoY)
## 2. SEGMENT PERFORMANCE TRENDS (QoQ & YoY)
## 3. BALANCE SHEET CHANGES (vs Prior Quarter & Year-End)
## 4. DEBT SCHEDULE UPDATE
## 5. CASH FLOW ANALYSIS (QTD & YTD)
## 6. OPERATIONAL METRICS (QoQ & YoY)
## 7. GUIDANCE & OUTLOOK UPDATES
## 8. NEW RISKS & MATERIAL DEVELOPMENTS
## 9. MANAGEMENT COMMENTARY
## 10. SUMMARY OF KEY CHANGES
## 11. RECENT EVENTS & SUBSEQUENT EVENTS
## 12. COMPETITIVE & INDUSTRY CONTEXT

For each section, show current quarter vs prior quarter vs year-ago quarter with percentage changes. Extract what changed since the last period.

OUTPUT FORMAT: Valid Markdown with proper headers, bullets, and tables.
"""

GEMINI_INVESTOR_DECK_PROMPT = """You are analyzing an Investor Presentation for {company_name} ({ticker}).

Presentation Date: {presentation_date}
Deck Type: {deck_type}

PART 1: Go through the deck page by page. For each page:
- Page number and title
- What the slide shows (text, data, charts, images)
- Key takeaway
- If data is in image format and not extractable, note: "Data in image - manual review recommended"

PART 2: After analyzing all pages, provide an Executive Summary with these sections:

## 1. DECK OVERVIEW & CONTEXT
## 2. KEY MESSAGES (Top 3-5)
## 3. FINANCIAL HIGHLIGHTS
## 4. GUIDANCE & FORWARD-LOOKING TARGETS
## 5. STRATEGIC PRIORITIES
## 6. DEAL-SPECIFIC DETAILS (if M&A deck)
## 7. OPERATIONAL METRICS & UNIT ECONOMICS
## 8. MARKET OPPORTUNITY & COMPETITIVE POSITIONING
## 9. RISKS & CHALLENGES ACKNOWLEDGED
## 10. MANAGEMENT TONE & EMPHASIS
## 11. NOTABLE OMISSIONS
## 12. INVESTMENT IMPLICATIONS
## 13. PAGES FLAGGED FOR MANUAL REVIEW (if any)

Target length: 3,000-6,000 words (scales with deck length).

---
COMPLETE INVESTOR DECK:
{full_deck_text}

---

Generate the page-by-page analysis and executive summary now.
"""

# ==============================================================================
# PDF/TEXT EXTRACTION
# ==============================================================================

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from PDF file using PyPDF2.
    Returns full text concatenated from all pages.
    """
    LOG.info(f"Extracting text from PDF: {pdf_path}")

    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)

            LOG.info(f"PDF has {total_pages} pages")

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += page_text

                if (i + 1) % 50 == 0:
                    LOG.info(f"Extracted {i + 1}/{total_pages} pages")

            LOG.info(f"✅ Extracted {len(text)} characters from {total_pages} pages")
            return text

    except Exception as e:
        LOG.error(f"Failed to extract PDF text: {e}")
        raise


def extract_text_file(txt_path: str) -> str:
    """Extract text from plain text file"""
    LOG.info(f"Reading text file: {txt_path}")

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        LOG.info(f"✅ Read {len(text)} characters from text file")
        return text

    except Exception as e:
        LOG.error(f"Failed to read text file: {e}")
        raise


def fetch_sec_html_text(url: str) -> str:
    """
    Fetch 10-K HTML from SEC.gov and extract plain text.

    Args:
        url: SEC.gov HTML URL (from FMP API)

    Returns:
        Plain text extracted from HTML

    Raises:
        Exception: If fetch or parsing fails
    """
    import requests
    from bs4 import BeautifulSoup

    LOG.info(f"Fetching 10-K HTML from SEC.gov: {url}")

    try:
        # SEC requires proper User-Agent to prevent blocking
        headers = {
            "User-Agent": "StockDigest/1.0 (stockdigest.research@gmail.com)"
        }

        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        LOG.info(f"✅ Fetched HTML ({len(response.text)} chars)")

        # Parse HTML and extract text
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text with newline separation
        text = soup.get_text(separator='\n', strip=True)

        # Clean up whitespace (collapse multiple spaces/tabs)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        LOG.info(f"✅ Extracted {len(text)} characters from HTML")

        return text

    except Exception as e:
        LOG.error(f"Failed to fetch SEC HTML: {e}")
        raise


# ==============================================================================
# GEMINI AI PROFILE GENERATION
# ==============================================================================

def generate_sec_filing_profile_with_gemini(
    ticker: str,
    content: str,
    config: Dict,
    filing_type: str,  # '10-K' or '10-Q'
    fiscal_year: int,
    fiscal_quarter: str = None,  # 'Q1', 'Q2', 'Q3', 'Q4' (required for 10-Q)
    filing_date: str = None,
    gemini_api_key: str = None
) -> Optional[Dict]:
    """
    Generate comprehensive SEC filing profile using Gemini 2.5 Flash.

    Supports both 10-K (annual) and 10-Q (quarterly) filings with specialized prompts.

    Args:
        ticker: Stock ticker (e.g., 'AAPL', 'TSLA')
        content: Full text of SEC filing
        config: Ticker configuration dict with company_name, etc.
        filing_type: '10-K' for annual or '10-Q' for quarterly
        fiscal_year: Fiscal year (e.g., 2024)
        fiscal_quarter: 'Q1', 'Q2', 'Q3', 'Q4' (required for 10-Q, None for 10-K)
        filing_date: Filing date string (optional)
        gemini_api_key: Gemini API key

    Returns:
        {
            'profile_markdown': str (2,000-4,500 words for 10-K, 2,000-5,000 for 10-Q),
            'metadata': {
                'model': str,
                'generation_time_seconds': int,
                'token_count_input': int,
                'token_count_output': int
            }
        }
    """
    # DIAGNOSTIC LOGGING - Entry point
    LOG.info(f"🔍 [DIAGNOSTIC] generate_sec_filing_profile_with_gemini() called")
    LOG.info(f"🔍 [DIAGNOSTIC] ticker={ticker}, filing_type={filing_type}, fiscal_year={fiscal_year}, fiscal_quarter={fiscal_quarter}")
    LOG.info(f"🔍 [DIAGNOSTIC] filing_date={filing_date}, content_length={len(content) if content else 0}")
    LOG.info(f"🔍 [DIAGNOSTIC] gemini_api_key={'SET' if gemini_api_key else 'NOT SET'}")
    LOG.info(f"🔍 [DIAGNOSTIC] config keys={list(config.keys()) if config else 'None'}")

    if not gemini_api_key:
        LOG.error("❌ Gemini API key not configured")
        return None

    if filing_type not in ['10-K', '10-Q']:
        LOG.error(f"Unsupported filing type: {filing_type}. Must be '10-K' or '10-Q'")
        return None

    if filing_type == '10-Q' and not fiscal_quarter:
        LOG.error("fiscal_quarter is required for 10-Q filings")
        return None

    try:
        genai.configure(api_key=gemini_api_key)

        company_name = config.get("company_name", ticker)

        # Select appropriate prompt based on filing type
        if filing_type == '10-K':
            prompt_template = GEMINI_10K_PROMPT
            filing_desc = f"10-K for FY{fiscal_year}"
            target_words = "2,000-4,500"
        else:  # 10-Q
            prompt_template = GEMINI_10Q_PROMPT
            filing_desc = f"10-Q for {fiscal_quarter} {fiscal_year}"
            target_words = "2,000-5,000"
            # Extract quarter number for prompt (Q3 -> 3)
            quarter_num = fiscal_quarter[1] if fiscal_quarter else "?"

        LOG.info(f"Generating {filing_type} profile for {ticker} ({filing_desc}) using Gemini 2.5 Flash")
        LOG.info(f"Content length: {len(content):,} chars (~{len(content)//4:,} tokens)")
        LOG.info(f"Target output: {target_words} words")

        # Gemini 2.5 Flash
        model = genai.GenerativeModel('gemini-2.5-flash')

        generation_config = {
            "temperature": 0.0,  # Maximum determinism for completely consistent extraction
            "max_output_tokens": 32768  # Increased to 32,768 for comprehensive extraction (Oct 2025)
        }

        start_time = datetime.now()

        # Build full prompt with content (different formatting for 10-K vs 10-Q)
        # Gemini 2.5 Flash has 1M token context window - use 400k tokens max
        LOG.info(f"🔍 [DIAGNOSTIC] Building prompt for {filing_type}...")
        if filing_type == '10-K':
            # Construct fiscal year end date (approximate as December 31)
            fiscal_year_end = f"{fiscal_year}-12-31"
            LOG.info(f"🔍 [DIAGNOSTIC] Template variables: company_name={company_name}, ticker={ticker}, filing_date={filing_date}, fiscal_year_end={fiscal_year_end}")
            try:
                full_prompt = prompt_template.format(
                    company_name=company_name,
                    ticker=ticker,
                    filing_date=filing_date or f"{fiscal_year}-12-31",  # Use provided filing_date or approximate
                    fiscal_year_end=fiscal_year_end,
                    full_10k_text=content[:1600000]  # ~400k tokens - full 10-K content
                )
                LOG.info(f"🔍 [DIAGNOSTIC] Prompt formatted successfully, length: {len(full_prompt)}")
            except KeyError as ke:
                LOG.error(f"❌ Template formatting error: Missing variable {ke}")
                LOG.error(f"Available variables: company_name, ticker, filing_date, fiscal_year_end, full_10k_text")
                raise
        else:  # 10-Q
            LOG.info(f"🔍 [DIAGNOSTIC] Template variables: company_name={company_name}, ticker={ticker}, quarter={quarter_num}, fiscal_year={fiscal_year}, filing_date={filing_date}")
            try:
                full_prompt = prompt_template.format(
                    company_name=company_name,
                    ticker=ticker,
                    quarter=quarter_num,
                    fiscal_year=fiscal_year,
                    filing_date=filing_date or "N/A",  # Use provided filing_date or N/A
                    full_10q_text=content[:1600000]  # ~400k tokens - full 10-Q content
                )
                LOG.info(f"🔍 [DIAGNOSTIC] Prompt formatted successfully, length: {len(full_prompt)}")
            except KeyError as ke:
                LOG.error(f"❌ Template formatting error: Missing variable {ke}")
                LOG.error(f"Available variables: company_name, ticker, quarter, fiscal_year, filing_date, full_10q_text")
                raise

        LOG.info(f"🔍 [DIAGNOSTIC] Calling Gemini API...")
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        LOG.info(f"🔍 [DIAGNOSTIC] Gemini API returned response")

        end_time = datetime.now()
        generation_time = int((end_time - start_time).total_seconds())

        LOG.info(f"🔍 [DIAGNOSTIC] Extracting response.text...")
        profile_markdown = response.text
        LOG.info(f"🔍 [DIAGNOSTIC] Response text extracted, length: {len(profile_markdown) if profile_markdown else 0}")

        if not profile_markdown or len(profile_markdown) < 1000:
            LOG.warning(f"❌ Gemini returned suspiciously short profile for {ticker} {filing_type} ({len(profile_markdown) if profile_markdown else 0} chars)")
            if profile_markdown:
                LOG.warning(f"Preview of short response: {profile_markdown[:500]}")
            return None

        # Extract metadata
        metadata = {
            'model': 'gemini-2.5-flash',
            'filing_type': filing_type,
            'generation_time_seconds': generation_time,
            'token_count_input': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
            'token_count_output': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
        }

        word_count = len(profile_markdown.split())
        LOG.info(f"✅ Generated {filing_type} profile for {ticker}")
        LOG.info(f"   Length: {len(profile_markdown):,} chars (~{word_count:,} words)")
        LOG.info(f"   Time: {generation_time}s")
        LOG.info(f"   Tokens: {metadata['token_count_input']:,} in, {metadata['token_count_output']:,} out")

        return {
            'profile_markdown': profile_markdown,
            'metadata': metadata
        }

    except Exception as e:
        LOG.error(f"Failed to generate {filing_type} profile for {ticker}: {e}")
        LOG.error(f"Stacktrace: {traceback.format_exc()}")
        return None


# Backward compatibility alias (deprecated)
def generate_company_profile_with_gemini(
    ticker: str,
    content: str,
    config: Dict,
    fiscal_year: int,
    filing_date: str,
    gemini_api_key: str,
    gemini_prompt: str = None  # Ignored (using GEMINI_10K_PROMPT instead)
) -> Optional[Dict]:
    """
    DEPRECATED: Use generate_sec_filing_profile_with_gemini() instead.

    This function maintains backward compatibility with existing code.
    """
    LOG.warning("generate_company_profile_with_gemini() is deprecated. Use generate_sec_filing_profile_with_gemini()")

    return generate_sec_filing_profile_with_gemini(
        ticker=ticker,
        content=content,
        config=config,
        filing_type='10-K',  # Default to 10-K for backward compatibility
        fiscal_year=fiscal_year,
        fiscal_quarter=None,
        filing_date=filing_date,
        gemini_api_key=gemini_api_key
    )


def generate_investor_presentation_analysis_with_gemini(
    ticker: str,
    pdf_path: str,
    config: Dict,
    presentation_date: str,
    deck_type: str,  # 'earnings', 'investor_day', 'analyst_day', 'conference'
    gemini_api_key: str
) -> Optional[Dict]:
    """
    Analyze investor presentation PDF using Gemini multimodal (vision + text).

    This function uploads a PDF to Gemini, which extracts text and visuals,
    then generates a comprehensive page-by-page analysis with executive summary.

    Args:
        ticker: Stock ticker (e.g., 'AAPL')
        pdf_path: Path to PDF file on disk
        config: Ticker configuration dict with company_name, industry
        presentation_date: Date of presentation (YYYY-MM-DD)
        deck_type: Type of presentation ('earnings', 'investor_day', 'analyst_day', 'conference')
        gemini_api_key: Gemini API key

    Returns:
        {
            'analysis_markdown': str (3,000-6,000 words depending on deck size),
            'metadata': {
                'model': str,
                'generation_time_seconds': int,
                'token_count_input': int,
                'token_count_output': int,
                'file_size_bytes': int
            }
        }
    """
    if not gemini_api_key:
        LOG.error("Gemini API key not configured")
        return None

    if not os.path.exists(pdf_path):
        LOG.error(f"PDF file not found: {pdf_path}")
        return None

    try:
        genai.configure(api_key=gemini_api_key)

        company_name = config.get("company_name", ticker)

        LOG.info(f"Analyzing investor presentation for {ticker} ({deck_type}) using Gemini multimodal")
        LOG.info(f"PDF: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

        # 1. Upload PDF to Gemini
        start_upload = datetime.now()
        uploaded_file = genai.upload_file(pdf_path)
        upload_time = (datetime.now() - start_upload).total_seconds()

        LOG.info(f"✅ Uploaded to Gemini: {uploaded_file.name} ({uploaded_file.size_bytes} bytes) in {upload_time:.1f}s")

        # 2. Wait for Gemini to process (extracts text + images internally)
        while uploaded_file.state.name == "PROCESSING":
            import time
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
            LOG.info(f"Gemini processing PDF visuals...")

        if uploaded_file.state.name == "FAILED":
            LOG.error("Gemini failed to process PDF")
            return None

        LOG.info(f"✅ Gemini processed PDF successfully")

        # 3. Build prompt
        prompt = GEMINI_INVESTOR_DECK_PROMPT.format(
            company_name=company_name,
            ticker=ticker,
            presentation_date=presentation_date,
            deck_type=deck_type,
            full_deck_text="[Gemini will extract from uploaded PDF]"
        )

        # 4. Generate analysis
        model = genai.GenerativeModel('gemini-2.5-pro')

        generation_config = {
            "temperature": 0.0,  # Maximum determinism for completely consistent extraction
            "max_output_tokens": 32768  # Increased to 32,768 for comprehensive extraction (Oct 2025)
        }

        start_time = datetime.now()

        LOG.info(f"Generating comprehensive deck analysis...")

        response = model.generate_content(
            [uploaded_file, prompt],  # Pass file object + prompt for multimodal analysis
            generation_config=generation_config
        )

        end_time = datetime.now()
        generation_time = int((end_time - start_time).total_seconds())

        analysis_markdown = response.text

        if not analysis_markdown or len(analysis_markdown) < 1000:
            LOG.warning(f"Gemini returned suspiciously short analysis for {ticker} ({len(analysis_markdown)} chars)")
            return None

        # 5. Extract metadata
        word_count = len(analysis_markdown.split())
        metadata = {
            'model': 'gemini-2.5-pro',
            'generation_time_seconds': generation_time,
            'token_count_input': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
            'token_count_output': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
            'file_size_bytes': uploaded_file.size_bytes
        }

        LOG.info(f"✅ Generated deck analysis for {ticker}")
        LOG.info(f"   Length: {len(analysis_markdown):,} chars (~{word_count:,} words)")
        LOG.info(f"   Time: {generation_time}s")
        LOG.info(f"   Tokens: {metadata['token_count_input']:,} in, {metadata['token_count_output']:,} out")

        # 6. Cleanup: Delete uploaded file from Gemini
        try:
            genai.delete_file(uploaded_file.name)
            LOG.info(f"✅ Deleted temp file from Gemini: {uploaded_file.name}")
        except Exception as e:
            LOG.warning(f"Failed to delete Gemini file (non-critical): {e}")

        return {
            'analysis_markdown': analysis_markdown,
            'metadata': metadata
        }

    except Exception as e:
        LOG.error(f"Failed to analyze presentation for {ticker}: {e}")
        LOG.error(f"Stacktrace: {traceback.format_exc()}")
        return None


# ==============================================================================
# EMAIL GENERATION
# ==============================================================================

def generate_company_profile_email(
    ticker: str,
    company_name: str,
    industry: str,
    fiscal_year: Optional[int],  # Can be None for presentations
    filing_date: str,
    profile_markdown: str,
    stock_price: str = None,
    price_change_pct: str = None,
    price_change_color: str = "#4ade80",
    ytd_return_pct: str = None,
    ytd_return_color: str = "#4ade80",
    market_status: str = "LAST CLOSE",
    return_label: str = "1D",
    filing_type: str = "10-K",  # "10-K", "10-Q", or "PRESENTATION"
    fiscal_quarter: Optional[str] = None  # e.g., "Q2" (for 10-Q only)
) -> Dict[str, str]:
    """
    Generate company profile email HTML using unified Jinja2 template.

    Returns:
        {"html": Full email HTML, "subject": Email subject}
    """
    LOG.info(f"Generating company profile email for {ticker} using unified template")

    current_date = datetime.now(timezone.utc).astimezone(pytz.timezone('US/Eastern')).strftime("%b %d, %Y")

    # Determine report type label
    if filing_type == "10-Q":
        report_type_label = "10-Q QUARTERLY REPORT"
    elif filing_type == "10-K":
        report_type_label = "10-K ANNUAL REPORT"
    else:
        report_type_label = "COMPANY PROFILE"

    # Handle fiscal period display
    if filing_type == "10-Q" and fiscal_quarter:
        fiscal_period = f"{fiscal_quarter} {fiscal_year}"
    elif fiscal_year:
        fiscal_period = f"FY {fiscal_year}"
    else:
        fiscal_period = filing_date

    # Date label for header (simplified - just show fiscal period)
    date_label = fiscal_period

    # Filing date display (optional in template)
    filing_date_display = f"Form {filing_type} Filed: {filing_date}"

    # Convert markdown to HTML with table support
    profile_html = markdown.markdown(
        profile_markdown,
        extensions=['tables', 'fenced_code', 'nl2br']
    )

    # Apply dynamic column widths based on number of columns per table
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(profile_html, 'html.parser')

        for table in soup.find_all('table'):
            # Count columns by examining first row
            first_row = table.find('tr')
            if first_row:
                num_columns = len(first_row.find_all(['th', 'td']))

                # Determine first column width based on total columns
                if num_columns <= 3:
                    first_col_width = "40%"  # 2 data cols = 30% each
                elif num_columns <= 5:
                    first_col_width = "35%"  # 4 data cols = 16.25% each (user confirmed good)
                elif num_columns <= 7:
                    first_col_width = "25%"  # 6 data cols = 12.5% each (improved)
                elif num_columns <= 9:
                    first_col_width = "20%"  # 8 data cols = 10% each (fixes wrapping)
                elif num_columns <= 11:
                    first_col_width = "15%"  # 10 data cols = 8.5% each (tight but readable)
                else:
                    first_col_width = "12%"  # 11+ data cols = 8% each (extreme case)

                # Apply width to first cell in every row
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        # Get existing style or create new
                        existing_style = cells[0].get('style', '')
                        # Add width to style (will override CSS)
                        cells[0]['style'] = f"width: {first_col_width}; padding: 8px; border: 1px solid #ddd;"

        # Convert back to HTML string
        profile_html = str(soup)
        LOG.info(f"Applied dynamic column widths to tables in {ticker} profile")
    except Exception as e:
        LOG.warning(f"Failed to apply dynamic column widths for {ticker}: {e}. Using default 40% width.")
        # Fallback: Use original HTML with CSS-based 40% width

    # Wrap content with original styling (grey wrapper, default headers, dynamic column width)
    content_html = f'''
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 13px; line-height: 1.6; color: #374151; background-color: #f9fafb; padding: 20px; border-radius: 4px; overflow-x: auto;">
    <style>
        table {{
            table-layout: fixed;
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 12px;
        }}
        th {{
            background-color: #1e40af;
            color: white;
            padding: 8px;
            text-align: left;
            border: 1px solid #ddd;
            font-weight: 600;
        }}
        td {{
            padding: 8px;
            border: 1px solid #ddd;
        }}
        tr:nth-child(even) {{
            background-color: #f3f4f6;
        }}
        /* First column width now applied inline per table based on column count */
        h1 {{
            color: #1e40af;
            font-size: 20px;
            margin-top: 24px;
            margin-bottom: 12px;
            border-bottom: 2px solid #1e40af;
            padding-bottom: 6px;
        }}
        h2 {{
            color: #1e3a8a;
            font-size: 18px;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        h3 {{
            color: #1e40af;
            font-size: 16px;
            margin-top: 16px;
            margin-bottom: 8px;
        }}
        code {{
            background-color: #e5e7eb;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 12px;
        }}
        pre {{
            background-color: #1f2937;
            color: #f3f4f6;
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
            color: #f3f4f6;
        }}
    </style>
    {profile_html}
</div>
'''

    # Render template with variables
    html = research_template.render(
        report_title=f"{ticker} Company Profile",
        report_type_label=report_type_label,
        company_name=company_name,
        ticker=ticker,
        industry=industry,
        fiscal_period=fiscal_period,
        date_label=date_label,
        filing_date=filing_date_display,
        stock_price=stock_price,
        price_change_pct=price_change_pct,
        price_change_color=price_change_color,
        ytd_return_pct=ytd_return_pct,
        ytd_return_color=ytd_return_color,
        return_label=return_label,
        content_html=content_html
    )

    # Generate subject based on filing type
    if filing_type == "10-Q":
        subject = f"10-Q Profile: {company_name} ({ticker}) - {fiscal_period}"
    elif filing_type == "10-K":
        subject = f"10-K Profile: {company_name} ({ticker}) - {fiscal_period}"
    else:
        subject = f"📋 Company Profile: {company_name} ({ticker}) {fiscal_period}"

    return {"html": html, "subject": subject}


# ==============================================================================
# DATABASE OPERATIONS
# ==============================================================================

def save_company_profile_to_database(
    ticker: str,
    profile_markdown: str,
    config: Dict,
    metadata: Dict,
    db_connection
) -> None:
    """Save company profile to unified sec_filings table"""
    LOG.info(f"Saving company profile for {ticker} to sec_filings table")

    try:
        cur = db_connection.cursor()

        # Determine source type
        source_file = config.get('source_file', '')
        source_type = 'fmp_sec' if 'SEC.gov' in source_file else 'file_upload'

        # First, delete any existing 10-K profile for this ticker/year
        # Note: fiscal_quarter is always NULL for 10-K
        cur.execute("""
            DELETE FROM sec_filings
            WHERE ticker = %s
              AND filing_type = %s
              AND fiscal_year = %s
              AND fiscal_quarter IS NULL
        """, (
            ticker,
            '10-K',
            config.get('fiscal_year')
        ))

        # Then insert the new profile
        cur.execute("""
            INSERT INTO sec_filings (
                ticker, filing_type, fiscal_year, fiscal_quarter,
                company_name, industry, filing_date, period_end_date,
                profile_markdown, source_file, source_type, sec_html_url,
                ai_provider, ai_model,
                generation_time_seconds, token_count_input, token_count_output,
                status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            ticker,
            '10-K',                              # filing_type (always 10-K for now)
            config.get('fiscal_year'),           # fiscal_year
            None,                                # fiscal_quarter (NULL for 10-K)
            config.get('company_name'),
            config.get('industry'),
            config.get('filing_date'),
            config.get('period_end_date'),       # Actual fiscal year end date
            profile_markdown,
            source_file,
            source_type,
            config.get('sec_html_url'),         # SEC.gov HTML URL (if available)
            'gemini',
            metadata.get('model'),               # ai_model (e.g., 'gemini-2.5-flash')
            metadata.get('generation_time_seconds'),
            metadata.get('token_count_input'),
            metadata.get('token_count_output'),
            'active'
        ))

        db_connection.commit()
        cur.close()

        LOG.info(f"✅ Saved 10-K profile for {ticker} (FY{config.get('fiscal_year')}) to sec_filings table")

    except Exception as e:
        LOG.error(f"Failed to save company profile for {ticker}: {e}")
        raise
