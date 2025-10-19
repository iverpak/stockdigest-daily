# modules/company_profiles.py

"""
Company Profiles Module

Handles 10-K filing ingestion and company profile generation using Gemini 2.5 Flash AI.
"""

import google.generativeai as genai
import PyPDF2
import logging
import traceback
from typing import Dict, Optional
from datetime import datetime, timezone
import pytz

LOG = logging.getLogger(__name__)

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

def generate_company_profile_with_gemini(
    ticker: str,
    content: str,
    config: Dict,
    fiscal_year: int,
    filing_date: str,
    gemini_api_key: str,
    gemini_prompt: str  # Full prompt passed from app.py
) -> Optional[Dict]:
    """
    Generate company profile using Gemini 2.5 Flash with Thinking mode.

    Returns:
        {
            'profile_markdown': str,
            'metadata': {
                'model': str,
                'generation_time_seconds': int,
                'token_count_input': int,
                'token_count_output': int
            }
        }
    """
    if not gemini_api_key:
        LOG.error("Gemini API key not configured")
        return None

    try:
        genai.configure(api_key=gemini_api_key)

        company_name = config.get("company_name", ticker)

        LOG.info(f"Generating company profile for {ticker} using Gemini 2.5 Flash (thinking mode)")
        LOG.info(f"Content length: {len(content)} chars (~{len(content)//4} tokens)")

        # Gemini 2.5 Flash with thinking
        model = genai.GenerativeModel('gemini-2.5-flash')

        generation_config = {
            "temperature": 0.3,
            "max_output_tokens": 8000
        }

        start_time = datetime.now()

        # Build full prompt with content
        full_prompt = gemini_prompt.format(
            company_name=company_name,
            ticker=ticker,
            filing_date=filing_date,
            fiscal_year_end=f"{fiscal_year}-12-31",  # Approximate
            full_10k_text=content
        )

        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )

        end_time = datetime.now()
        generation_time = int((end_time - start_time).total_seconds())

        profile_markdown = response.text

        if not profile_markdown or len(profile_markdown) < 1000:
            LOG.warning(f"Gemini returned suspiciously short profile for {ticker} ({len(profile_markdown)} chars)")
            return None

        # Extract metadata
        metadata = {
            'model': 'gemini-2.5-flash',
            'generation_time_seconds': generation_time,
            'token_count_input': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
            'token_count_output': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
        }

        LOG.info(f"✅ Generated company profile for {ticker}")
        LOG.info(f"   Length: {len(profile_markdown)} chars")
        LOG.info(f"   Time: {generation_time}s")
        LOG.info(f"   Tokens: {metadata['token_count_input']} in, {metadata['token_count_output']} out")

        return {
            'profile_markdown': profile_markdown,
            'metadata': metadata
        }

    except Exception as e:
        LOG.error(f"Failed to generate company profile for {ticker}: {e}")
        LOG.error(f"Stacktrace: {traceback.format_exc()}")
        return None


# ==============================================================================
# EMAIL GENERATION
# ==============================================================================

def generate_company_profile_email(
    ticker: str,
    company_name: str,
    industry: str,
    fiscal_year: int,
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
                                        <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0px; opacity: 0.85; font-weight: 600; color: #ffffff;">Generated: {current_date} | FY{fiscal_year}</div>
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

    subject = f"📋 Company Profile: {company_name} ({ticker}) FY{fiscal_year}"

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

        cur.execute("""
            INSERT INTO sec_filings (
                ticker, filing_type, fiscal_year, fiscal_quarter,
                company_name, industry, filing_date,
                profile_markdown, source_file, source_type, sec_html_url,
                ai_provider, ai_model,
                generation_time_seconds, token_count_input, token_count_output,
                status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, filing_type, fiscal_year, fiscal_quarter) DO UPDATE SET
                company_name = EXCLUDED.company_name,
                industry = EXCLUDED.industry,
                filing_date = EXCLUDED.filing_date,
                profile_markdown = EXCLUDED.profile_markdown,
                source_file = EXCLUDED.source_file,
                source_type = EXCLUDED.source_type,
                sec_html_url = EXCLUDED.sec_html_url,
                ai_provider = EXCLUDED.ai_provider,
                ai_model = EXCLUDED.ai_model,
                generation_time_seconds = EXCLUDED.generation_time_seconds,
                token_count_input = EXCLUDED.token_count_input,
                token_count_output = EXCLUDED.token_count_output,
                generated_at = NOW(),
                status = 'active',
                error_message = NULL
        """, (
            ticker,
            '10-K',                              # filing_type (always 10-K for now)
            config.get('fiscal_year'),           # fiscal_year
            None,                                # fiscal_quarter (NULL for 10-K)
            config.get('company_name'),
            config.get('industry'),
            config.get('filing_date'),
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
