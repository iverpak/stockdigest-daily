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

LOG = logging.getLogger(__name__)

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
- Target length: 3-5 pages (~3,000-5,000 words)

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
- Target length: 3-5 pages (~3,000-5,000 words)

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
            'profile_markdown': str (3,000-5,000 words for 10-K, 3,000-5,000 for 10-Q),
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
            target_words = "3,000-5,000"
        else:  # 10-Q
            prompt_template = GEMINI_10Q_PROMPT
            filing_desc = f"10-Q for {fiscal_quarter} {fiscal_year}"
            target_words = "3,000-5,000"
            # Extract quarter number for prompt (Q3 -> 3)
            quarter_num = fiscal_quarter[1] if fiscal_quarter else "?"

        LOG.info(f"Generating {filing_type} profile for {ticker} ({filing_desc}) using Gemini 2.5 Flash")
        LOG.info(f"Content length: {len(content):,} chars (~{len(content)//4:,} tokens)")
        LOG.info(f"Target output: {target_words} words")

        # Gemini 2.5 Flash
        model = genai.GenerativeModel('gemini-2.5-flash')

        generation_config = {
            "temperature": 0.3,
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
            "temperature": 0.3,
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
    stock_price: str = "$0.00",
    price_change_pct: str = None,
    price_change_color: str = "#4ade80"
) -> Dict[str, str]:
    """
    Generate company profile email HTML.

    Returns:
        {"html": Full email HTML, "subject": Email subject}
    """
    LOG.info(f"Generating company profile email for {ticker}")

    current_date = datetime.now(timezone.utc).astimezone(pytz.timezone('US/Eastern')).strftime("%b %d, %Y")

    # Handle fiscal_year for presentations (can be None)
    fiscal_year_display = f"FY{fiscal_year}" if fiscal_year else filing_date

    # Extract first ~2000 chars for email preview
    profile_preview = profile_markdown[:2000] + "..." if len(profile_markdown) > 2000 else profile_markdown

    # Convert markdown to simple HTML (basic conversion - just wrap in pre tag for monospace)
    # TODO: Could use a proper markdown library here for better rendering
    profile_html = f'<div style="font-family: monospace; font-size: 12px; line-height: 1.5; white-space: pre-wrap; color: #374151; background-color: #f9fafb; padding: 16px; border-radius: 4px; overflow-x: auto;">{profile_preview}</div>'

    # Build HTML (same structure as transcript email)
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} Company Profile</title>
    <style>
        @media only screen and (max-width: 600px) {{
            .content-padding {{ padding: 16px !important; }}
            .header-padding {{ padding: 16px 20px 25px 20px !important; }}
            .price-box {{ padding: 8px 10px !important; }}
            .company-name {{ font-size: 20px !important; }}
        }}
    </style>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f8f9fa; color: #212529;">

    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td align="center" style="padding: 40px 20px;">

                <table role="presentation" style="max-width: 700px; width: 100%; background-color: #ffffff; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-collapse: collapse; border-radius: 8px;">

                    <!-- Header -->
                    <tr>
                        <td class="header-padding" style="padding: 18px 24px 30px 24px; background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); color: #ffffff; border-radius: 8px 8px 0 0;">
                            <table role="presentation" style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td style="width: 58%;">
                                        <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0px; opacity: 0.85; font-weight: 600; color: #ffffff;">COMPANY PROFILE</div>
                                    </td>
                                    <td align="right" style="width: 42%;">
                                        <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0px; opacity: 0.85; font-weight: 600; color: #ffffff;">Generated: {current_date} | {fiscal_year_display}</div>
                                    </td>
                                </tr>
                                <tr>
                                    <td style="width: 58%; vertical-align: top;">
                                        <h1 class="company-name" style="margin: 0; font-size: 28px; font-weight: 700; letter-spacing: -0.5px; line-height: 1; color: #ffffff;">{company_name}</h1>
                                        <div style="font-size: 13px; margin-top: 2px; opacity: 0.9; color: #ffffff;">{ticker} | {industry}</div>
                                        <div style="font-size: 11px; margin-top: 4px; opacity: 0.8; color: #ffffff;">Form 10-K Filed: {filing_date}</div>
                                    </td>
                                    <td align="right" style="width: 42%; vertical-align: top;">
                                        <div style="font-size: 28px; font-weight: 700; letter-spacing: -0.5px; line-height: 1; color: #ffffff; margin-bottom: 2px;">{stock_price}</div>
                                        {f'<div style="font-size: 13px; font-weight: 600; color: {price_change_color};">{price_change_pct}</div>' if price_change_pct else ''}
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Content -->
                    <tr>
                        <td class="content-padding" style="padding: 24px;">

                            <!-- Profile Preview -->
                            <div style="margin-bottom: 20px;">
                                <h2 style="font-size: 16px; font-weight: 700; color: #1e40af; margin: 0 0 12px 0;">Profile Preview (First 2,000 Characters)</h2>
                                {profile_html}
                            </div>

                            <!-- Full Profile Info Box -->
                            <div style="margin: 32px 0 20px 0; padding: 12px 16px; background-color: #eff6ff; border-left: 4px solid #1e40af; border-radius: 4px;">
                                <p style="margin: 0; font-size: 12px; color: #1e40af; font-weight: 600; line-height: 1.4;">
                                    Full company profile ({len(profile_markdown):,} characters) saved to database.
                                    Access via Admin Panel at https://stockdigest.app/admin
                                </p>
                            </div>

                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); padding: 16px 24px; color: rgba(255,255,255,0.9); border-radius: 0 0 8px 8px;">
                            <table role="presentation" style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td>
                                        <div style="font-size: 14px; font-weight: 600; color: #ffffff; margin-bottom: 4px;">StockDigest Research Tools</div>
                                        <div style="font-size: 12px; opacity: 0.8; margin-bottom: 8px; color: #ffffff;">Company Profile Analysis</div>

                                        <!-- Legal Disclaimer -->
                                        <div style="font-size: 10px; opacity: 0.7; line-height: 1.4; margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.2); color: #ffffff;">
                                            For informational and educational purposes only. Not investment advice. See Terms of Service for full disclaimer.
                                        </div>

                                        <!-- Links -->
                                        <div style="font-size: 11px; margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.2);">
                                            <a href="https://stockdigest.app/terms-of-service" style="color: #ffffff; text-decoration: none; opacity: 0.9; margin-right: 12px;">Terms of Service</a>
                                            <span style="color: rgba(255,255,255,0.5); margin-right: 12px;">|</span>
                                            <a href="https://stockdigest.app/privacy-policy" style="color: #ffffff; text-decoration: none; opacity: 0.9; margin-right: 12px;">Privacy Policy</a>
                                            <span style="color: rgba(255,255,255,0.5); margin-right: 12px;">|</span>
                                            <a href="mailto:stockdigest.research@gmail.com" style="color: #ffffff; text-decoration: none; opacity: 0.9;">Contact</a>
                                        </div>

                                        <!-- Copyright -->
                                        <div style="font-size: 10px; opacity: 0.6; margin-top: 12px; color: #ffffff;">
                                            © 2025 StockDigest. All rights reserved.
                                        </div>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                </table>

            </td>
        </tr>
    </table>

</body>
</html>'''

    subject = f"📋 Company Profile: {company_name} ({ticker}) {fiscal_year_display}"

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
