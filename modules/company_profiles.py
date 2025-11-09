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
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import pytz
import markdown
from jinja2 import Environment, FileSystemLoader

# Use same logger as app.py so logs are visible in Render
LOG = logging.getLogger("quantbrief")

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
# SEC 8-K FILING PROCESSING
# ==============================================================================

# Complete mapping of SEC 8-K item codes to descriptions
ITEM_CODE_MAP = {
    '1.01': 'Entry into Material Agreement',
    '1.02': 'Termination of Material Agreement',
    '1.03': 'Bankruptcy or Receivership',
    '1.04': 'Mine Safety - Reporting of Shutdowns and Patterns of Violations',
    '2.01': 'Completion of Acquisition or Disposition',
    '2.02': 'Results of Operations and Financial Condition',
    '2.03': 'Creation of Direct Financial Obligation',
    '2.04': 'Triggering Events That Accelerate Obligations',
    '2.05': 'Costs Associated with Exit Activities',
    '2.06': 'Material Impairments',
    '3.01': 'Notice of Delisting',
    '3.02': 'Unregistered Sales of Equity Securities',
    '3.03': 'Material Modification to Rights of Security Holders',
    '4.01': 'Changes in Registrant\'s Certifying Accountant',
    '4.02': 'Non-Reliance on Previously Issued Financial Statements',
    '5.01': 'Changes in Control of Registrant',
    '5.02': 'Departure/Appointment of Directors or Officers',
    '5.03': 'Amendments to Articles or Bylaws',
    '5.04': 'Temporary Suspension of Trading',
    '5.05': 'Amendments to Code of Ethics',
    '5.06': 'Change in Shell Company Status',
    '5.07': 'Submission of Matters to Vote',
    '5.08': 'Shareholder Director Nominations',
    '7.01': 'Regulation FD Disclosure',
    '8.01': 'Other Events',
    '9.01': 'Financial Statements and Exhibits',
}


def get_cik_for_ticker(ticker: str) -> str:
    """
    Get CIK (Central Index Key) for ticker using FMP API.

    1. Check sec_8k_filings table for cached CIK
    2. If not found, lookup from FMP API (most reliable)
    3. Return CIK for use in SEC Edgar queries

    Args:
        ticker: Stock ticker (e.g., 'AAPL', 'TSLA')

    Returns:
        CIK string (e.g., '0000320193')

    Raises:
        ValueError: If ticker not found or FMP API fails
    """
    import requests

    # Check cache first
    try:
        from app import db

        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT cik FROM sec_8k_filings
                WHERE ticker = %s AND cik IS NOT NULL
                LIMIT 1
            """, (ticker,))
            row = cur.fetchone()

            if row and row['cik']:
                LOG.info(f"[{ticker}] Found cached CIK: {row['cik']}")
                return row['cik']
    except Exception as e:
        LOG.warning(f"[{ticker}] Could not check CIK cache: {e}")

    # Lookup from FMP API (most reliable - returns current, active CIK)
    LOG.info(f"[{ticker}] Looking up CIK from FMP API...")

    try:
        # Get FMP API key from environment
        fmp_api_key = os.environ.get('FMP_API_KEY')
        if not fmp_api_key:
            raise ValueError("FMP_API_KEY not configured")

        # FMP profile endpoint provides CIK
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={fmp_api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        if not data or len(data) == 0:
            raise ValueError(f"Ticker {ticker} not found in FMP database")

        cik = data[0].get('cik')

        if not cik:
            raise ValueError(f"CIK not available for ticker {ticker}")

        LOG.info(f"[{ticker}] ✅ Found CIK from FMP: {cik}")
        return cik

    except Exception as e:
        LOG.error(f"[{ticker}] CIK lookup failed: {e}")
        raise ValueError(f"Could not find CIK for ticker {ticker}. Error: {str(e)}")


def parse_sec_8k_filing_list(cik: str, count: int = 10) -> List[Dict]:
    """
    Scrape SEC Edgar for last N 8-K filings.

    Args:
        cik: SEC CIK number (e.g., '0000320193')
        count: Number of filings to retrieve (default 10)

    Returns:
        List of filing dicts with:
        - filing_date: 'Jan 30, 2025'
        - accession_number: '0001193125-25-012345'
        - documents_url: Full URL to documents page
    """
    import requests
    import re
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    LOG.info(f"Fetching last {count} 8-K filings for CIK {cik}...")

    try:
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&count={count}"
        headers = {
            'User-Agent': 'StockDigest/1.0 (stockdigest.research@gmail.com)'
        }

        LOG.info(f"[8K_DEBUG] Making request to SEC for CIK {cik}...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        LOG.info(f"[8K_DEBUG] Got response, parsing HTML...")

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='tableFile2')

        if not table:
            LOG.warning(f"No 8-K filings table found for CIK {cik}")
            return []

        LOG.info(f"[8K_DEBUG] Found table, parsing rows...")
        filings = []
        rows = table.find_all('tr')[1:]  # Skip header row
        LOG.info(f"[8K_DEBUG] Found {len(rows)} rows in table")

        for i, row in enumerate(rows):
            cols = row.find_all('td')
            if len(cols) >= 4:
                filing_date = cols[3].text.strip()  # "Jan 30, 2025"

                # Find Documents link (using id='documentsbutton' to handle &nbsp; in text)
                documents_link = cols[1].find('a', id='documentsbutton')
                if not documents_link:
                    LOG.warning(f"[8K_DEBUG] Row {i}: No Documents link found")
                    continue

                LOG.info(f"[8K_DEBUG] Row {i}: Found Documents link")
                documents_url = urljoin("https://www.sec.gov", documents_link['href'])
                LOG.info(f"[8K_DEBUG] Row {i}: Documents URL: {documents_url}")

                # Extract accession number from URL (format: /0000320193-25-000077-index.htm)
                match = re.search(r'/(\d{10}-\d{2}-\d{6})-index\.htm', documents_url)
                accession = match.group(1) if match else None
                LOG.info(f"[8K_DEBUG] Row {i}: Regex match: {match is not None}, accession: {accession}")

                if accession:
                    filings.append({
                        'filing_date': filing_date,
                        'accession_number': accession,
                        'documents_url': documents_url
                    })

        LOG.info(f"✅ Found {len(filings)} 8-K filings")
        return filings[:count]

    except Exception as e:
        LOG.error(f"Failed to parse 8-K filing list: {e}")
        raise


def get_8k_html_url(documents_url: str) -> dict:
    """
    Parse documents index page to find main 8-K HTML file and Exhibit 99.1.

    Args:
        documents_url: URL to SEC documents index page

    Returns:
        Dict with:
            'main_8k_url': Full URL to main 8-K HTML file
            'exhibit_99_1_url': Full URL to Exhibit 99.1 (or None if not found)

    Raises:
        ValueError: If main 8-K file not found
    """
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    LOG.info(f"Parsing documents page: {documents_url}")

    try:
        headers = {
            'User-Agent': 'StockDigest/1.0 (stockdigest.research@gmail.com)'
        }

        response = requests.get(documents_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='tableFile')

        if not table:
            raise ValueError("Documents table not found on index page")

        rows = table.find_all('tr')[1:]  # Skip header
        LOG.info(f"[8K_DOC_DEBUG] Found {len(rows)} rows in documents table")

        main_8k_url = None
        exhibit_99_1_url = None

        # Find both the main 8-K document AND Exhibit 99.1
        for i, row in enumerate(rows):
            cols = row.find_all('td')
            LOG.info(f"[8K_DOC_DEBUG] Row {i}: {len(cols)} columns")

            if len(cols) >= 4:
                # Show first 3 rows in detail
                if i < 3:
                    LOG.info(f"[8K_DOC_DEBUG] Row {i} cols: [0]='{cols[0].text.strip()[:50]}', [1]='{cols[1].text.strip()[:50]}', [2]='{cols[2].text.strip()[:50]}', [3]='{cols[3].text.strip()[:50]}'")

                doc_type = cols[1].text.strip()  # Column 1: Type (8-K, EX-99.1, GRAPHIC, etc.)
                filename = cols[2].text.strip()  # Column 2: Filename (aapl-20251030.htm, etc.)

                LOG.info(f"[8K_DOC_DEBUG] Row {i}: doc_type='{doc_type}', filename='{filename[:50]}', is_htm={'.htm' in filename.lower()}, is_8k={'8-k' in doc_type.lower()}")

                # Look for main 8-K HTML file (type='8-K' and filename ends with .htm)
                if not main_8k_url and '8-k' in doc_type.lower() and '.htm' in filename.lower():
                    link = cols[2].find('a')
                    if link and 'href' in link.attrs:
                        html_url = urljoin(documents_url, link['href'])
                        # Strip iXBRL viewer wrapper to get raw HTML
                        html_url = html_url.replace('/ix?doc=', '')
                        main_8k_url = html_url
                        LOG.info(f"✅ Found main 8-K HTML: {html_url}")

                # Look for Exhibit 99.1 (press release with financial data)
                if not exhibit_99_1_url and 'ex-99.1' in doc_type.lower() and '.htm' in filename.lower():
                    link = cols[2].find('a')
                    if link and 'href' in link.attrs:
                        exhibit_url = urljoin(documents_url, link['href'])
                        # Strip iXBRL viewer wrapper to get raw HTML
                        exhibit_url = exhibit_url.replace('/ix?doc=', '')
                        exhibit_99_1_url = exhibit_url
                        LOG.info(f"✅ Found Exhibit 99.1: {exhibit_url}")

        # If we found the main 8-K, return both URLs
        if main_8k_url:
            return {
                'main_8k_url': main_8k_url,
                'exhibit_99_1_url': exhibit_99_1_url
            }

        # Fallback: First .htm file (any type)
        LOG.info(f"[8K_DOC_DEBUG] No match found, trying fallback (first .htm file)...")
        for i, row in enumerate(rows):
            cols = row.find_all('td')
            if len(cols) >= 4:
                filename = cols[2].text.strip()
                LOG.info(f"[8K_DOC_DEBUG] Fallback row {i}: filename='{filename[:50]}', is_htm={'.htm' in filename.lower()}")
                if '.htm' in filename.lower():
                    link = cols[2].find('a')
                    if link and 'href' in link.attrs:
                        html_url = urljoin(documents_url, link['href'])
                        # Strip iXBRL viewer wrapper to get raw HTML
                        html_url = html_url.replace('/ix?doc=', '')
                        LOG.warning(f"Using fallback HTML file: {html_url}")
                        return {
                            'main_8k_url': html_url,
                            'exhibit_99_1_url': None
                        }

        raise ValueError("Could not find main 8-K HTML file in documents list")

    except Exception as e:
        LOG.error(f"Failed to get 8-K HTML URL: {e}")
        raise


def get_all_8k_exhibits(documents_url: str) -> List[Dict[str, Any]]:
    """
    Parse SEC documents index page and find ALL exhibit files (any number).

    Extracts ALL exhibits regardless of number (1.1, 4.1, 10.1, 99.1, etc.).
    Only processes HTML files (skips images, XML, TXT).

    Args:
        documents_url: URL to SEC documents index page

    Returns:
        [
            {
                "exhibit_number": "99.1",
                "description": "Press Release",
                "url": "https://www.sec.gov/.../ex99_1.htm",
                "size": 131002
            },
            {
                "exhibit_number": "1.1",
                "description": "Underwriting Agreement",
                "url": "https://www.sec.gov/.../ex1_1.htm",
                "size": 205000
            }
        ]

    Raises:
        ValueError: If no HTML exhibits found
    """
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    LOG.info(f"Parsing documents page for all HTML exhibits: {documents_url}")

    try:
        headers = {
            'User-Agent': 'StockDigest/1.0 (stockdigest.research@gmail.com)'
        }

        response = requests.get(documents_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='tableFile')

        if not table:
            raise ValueError("Documents table not found on index page")

        rows = table.find_all('tr')[1:]  # Skip header
        exhibits = []

        LOG.info(f"[EXHIBIT_DEBUG] Found {len(rows)} rows in documents table")

        # Find ALL exhibit files (any number: 1.1, 4.1, 10.1, 99.1, etc.)
        for i, row in enumerate(rows):
            cols = row.find_all('td')

            if len(cols) >= 4:
                doc_type = cols[1].text.strip()  # Column 1: Type (EXHIBIT 1.1, EX-99.1, etc.)
                filename = cols[2].text.strip()  # Column 2: Filename
                size_text = cols[3].text.strip()  # Column 3: Size

                LOG.info(f"[EXHIBIT_DEBUG] Row {i}: Type='{doc_type}', Filename='{filename}'")

                # Match ANY exhibit (not just 99.*) that's HTML (skip images, XML, TXT)
                doc_type_upper = doc_type.upper()
                is_exhibit = doc_type_upper.startswith('EXHIBIT') or doc_type_upper.startswith('EX-')
                is_html = '.htm' in filename.lower()

                if is_exhibit and is_html:
                    link = cols[2].find('a')
                    if link and 'href' in link.attrs:
                        # Extract exhibit number (e.g., "99.1" from "EX-99.1" or "EXHIBIT 99.1")
                        exhibit_num = (
                            doc_type.replace('EXHIBIT', '')
                                    .replace('EX-', '')
                                    .replace('ex-', '')
                                    .strip()
                        )

                        # Build full URL
                        exhibit_url = urljoin(documents_url, link['href'])
                        # Strip iXBRL viewer wrapper
                        exhibit_url = exhibit_url.replace('/ix?doc=', '')

                        # Parse size (in bytes)
                        try:
                            size_bytes = int(size_text)
                        except:
                            size_bytes = 0

                        # Get description from column 0 or filename
                        # Note: Column 0 is often just a sequence number, use filename as fallback
                        desc_from_col = cols[0].text.strip()
                        if desc_from_col and len(desc_from_col) > 5 and not desc_from_col.isdigit():
                            description = desc_from_col
                        else:
                            # Use filename without extension as description
                            description = filename.replace('.htm', '').replace('_', ' ').title()

                        exhibits.append({
                            'exhibit_number': exhibit_num,
                            'description': description,
                            'url': exhibit_url,
                            'size': size_bytes
                        })

                        LOG.info(f"✅ Found Exhibit {exhibit_num}: {description} ({size_bytes} bytes)")

        if not exhibits:
            # Provide helpful error message based on what document types ARE present
            present_docs = [cols[1].text.strip() for row in rows for cols in [row.find_all('td')] if len(cols) >= 4]
            doc_list = ', '.join(set(present_docs[:5]))  # Show first 5 unique types
            raise ValueError(
                f"No HTML exhibits found in this 8-K filing. "
                f"Found document types: {doc_list}. "
                f"This filing may only have non-HTML attachments (images, XML, PDF) or no exhibits at all."
            )

        # Sort by exhibit number (1.1, 4.1, 10.1, 99.1, 99.2, etc.)
        exhibits.sort(key=lambda x: float(x['exhibit_number']))

        LOG.info(f"✅ Found {len(exhibits)} HTML exhibits total")
        return exhibits

    except Exception as e:
        LOG.error(f"Failed to parse exhibits from documents page: {e}")
        raise


def classify_exhibit_type(exhibit_num: str, description: str, char_count: int) -> str:
    """
    Classify exhibit based on number, description, and size.

    Uses heuristics to automatically categorize exhibits for easy filtering
    and future integration with executive summary generation.

    Args:
        exhibit_num: Exhibit number (e.g., "99.1", "99.2")
        description: Exhibit description from SEC
        char_count: Character count of HTML content

    Returns:
        "earnings_release" | "investor_presentation" | "press_release" | "other"
    """
    desc_lower = description.lower()

    # Earnings Release indicators
    if any(keyword in desc_lower for keyword in [
        'earnings release',
        'financial results',
        'quarterly results',
        'quarterly earnings'
    ]):
        return 'earnings_release'

    # Investor Presentation indicators (usually larger, slides)
    if any(keyword in desc_lower for keyword in [
        'supplemental',
        'presentation',
        'investor deck',
        'slides',
        'supplemental information'
    ]):
        # Large files are likely presentation decks
        if char_count > 80000:  # > 80KB
            return 'investor_presentation'

    # Press Release indicators
    if 'press release' in desc_lower:
        return 'press_release'

    # Fallback heuristic: 99.2 and smaller size likely earnings release
    if exhibit_num == '99.2' and char_count < 100000:
        return 'earnings_release'

    # Default
    return 'other'


def quick_parse_8k_header(sec_html_url: str, rate_limit_delay: float = 0.15) -> Dict:
    """
    Quick parse 8-K header (first 3KB) to extract title and item codes.

    Only fetches first 3KB of file for speed. Use for display in UI before
    user decides to generate full summary.

    Args:
        sec_html_url: Full URL to 8-K HTML on SEC.gov
        rate_limit_delay: Delay in seconds before request (default 0.15s = 6.67 req/sec)

    Returns:
        {
            'title': "Results of Operations | Apple announces Q1 2024 results",
            'item_codes': "2.02, 9.01",
            'item_description': "Results of Operations and Financial Condition"
        }
    """
    import re
    import time
    import requests

    # Rate limit protection
    time.sleep(rate_limit_delay)

    LOG.info(f"Quick parsing 8-K header: {sec_html_url}")

    try:
        headers = {
            'User-Agent': 'StockDigest/1.0 (stockdigest.research@gmail.com)',
            'Range': 'bytes=0-3000'  # Only fetch first 3KB
        }

        response = requests.get(sec_html_url, headers=headers, timeout=10)
        text = response.text

        LOG.info(f"[8K_HEADER_DEBUG] Fetched {len(text)} bytes, status={response.status_code}")
        LOG.info(f"[8K_HEADER_DEBUG] First 500 chars: {text[:500]}")

        # Check if this is iXBRL (inline XBRL) - structured data format
        is_ixbrl = '<?xml version' in text[:200] or 'xmlns:ix=' in text[:500]
        if is_ixbrl:
            LOG.info("[8K_HEADER_DEBUG] Detected iXBRL format - using defaults (full extraction will parse properly)")
            return {
                'title': "8-K Filing",
                'item_codes': "See filing",
                'item_description': "Material Events"
            }

        # Extract item codes (format: "Item 2.02" or "Item 2.02.")
        items = re.findall(r'Item\s+(\d+\.\d+)', text, re.IGNORECASE)
        LOG.info(f"[8K_HEADER_DEBUG] Found {len(items)} item codes: {items}")
        item_codes = ', '.join(sorted(set(items[:3])))  # Dedupe and limit to first 3

        # Get item description for primary item
        item_description = "Other Events"  # Default
        if items:
            primary_item = items[0]
            item_description = ITEM_CODE_MAP.get(primary_item, "Other Events")

        # Extract title from content
        # Look for patterns like "announces", "reports", "completes", etc.
        title_match = re.search(
            r'(announces?|reports?|completes?|enters?|acquires?|appoints?)[^\.]{10,200}',
            text,
            re.IGNORECASE
        )
        parsed_title = title_match.group(0).strip() if title_match else ""
        LOG.info(f"[8K_HEADER_DEBUG] Title match: {bool(title_match)}, parsed_title: '{parsed_title[:100] if parsed_title else 'NONE'}'...")

        # Option C: Both item description AND parsed title
        if parsed_title:
            full_title = f"{item_description} | {parsed_title}"
        else:
            full_title = item_description

        # Truncate to fit VARCHAR(200)
        full_title = full_title[:200]

        LOG.info(f"✅ Parsed: Items={item_codes}, Title={full_title[:50]}...")

        return {
            'title': full_title,
            'item_codes': item_codes if item_codes else "Unknown",
            'item_description': item_description
        }

    except Exception as e:
        LOG.error(f"Failed to quick parse 8-K header: {e}")
        # Return defaults on error
        return {
            'title': "8-K Filing",
            'item_codes': "Unknown",
            'item_description': "Other Events"
        }


def extract_8k_content(sec_html_url: str, exhibit_99_1_url: str = None) -> str:
    """
    Extract 8-K content for AI summarization.

    Uses ONLY Exhibit 99.1 (the actual press release/announcement). Main 8-K body
    contains mostly legal disclaimers and boilerplate, so it's only used as fallback.

    Args:
        sec_html_url: Full URL to main 8-K HTML on SEC.gov
        exhibit_99_1_url: Full URL to Exhibit 99.1 (or None if not found)

    Returns:
        Clean text content for Gemini processing (Exhibit 99.1 or main body fallback)

    Raises:
        ValueError: If content too short or empty
    """
    LOG.info(f"Extracting 8-K content from: {sec_html_url}")
    if exhibit_99_1_url:
        LOG.info(f"Exhibit 99.1 URL provided: {exhibit_99_1_url}")

    try:
        # Fetch Exhibit 99.1 if URL was provided
        if exhibit_99_1_url:
            try:
                exhibit_99_1 = fetch_sec_html_text(exhibit_99_1_url)
                LOG.info(f"✅ Fetched Exhibit 99.1 ({len(exhibit_99_1)} chars)")

                # Validate sufficient content
                if len(exhibit_99_1) > 500:
                    LOG.info(f"✅ Using Exhibit 99.1 only ({len(exhibit_99_1)} chars)")
                    return exhibit_99_1
                else:
                    LOG.warning(f"Exhibit 99.1 too short ({len(exhibit_99_1)} chars), falling back to main body")
            except Exception as e:
                LOG.warning(f"Failed to fetch Exhibit 99.1: {e}, falling back to main body")

        # Fallback: Fetch main 8-K HTML (rare - only if no Exhibit 99.1)
        # Example: Item 5.02 (officer departure) might be in main body only
        main_8k_text = fetch_sec_html_text(sec_html_url)
        LOG.info(f"Using main 8-K body as fallback ({len(main_8k_text)} chars)")

        # Validate minimum length
        if len(main_8k_text) < 500:
            raise ValueError(
                f"8-K content too short ({len(main_8k_text)} chars) - may be empty or malformed"
            )

        return main_8k_text

    except Exception as e:
        LOG.error(f"Failed to extract 8-K content: {e}")
        raise


def extract_8k_html_content(exhibit_url: str) -> str:
    """
    Extract single 8-K exhibit content AS HTML (preserving tables and formatting).

    Simplified for exhibit-level processing - fetches one exhibit at a time.
    Returns raw HTML with tables intact for display in email.

    Args:
        exhibit_url: Full URL to exhibit HTML on SEC.gov

    Returns:
        Clean HTML content with tables intact

    Raises:
        ValueError: If content too short or empty
    """
    import requests
    from bs4 import BeautifulSoup

    LOG.info(f"Extracting 8-K exhibit HTML from: {exhibit_url}")

    # SEC requires proper User-Agent
    headers = {
        "User-Agent": "StockDigest/1.0 (stockdigest.research@gmail.com)"
    }

    try:
        response = requests.get(exhibit_url, headers=headers, timeout=60)
        response.raise_for_status()

        LOG.info(f"✅ Fetched HTML ({len(response.text)} chars)")

        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements (but keep tables!)
        for tag in soup(['script', 'style', 'head', 'meta', 'link', 'noscript']):
            tag.decompose()

        # Extract body content (or full soup if no body tag)
        body = soup.find('body')
        if body:
            html_content = str(body)
        else:
            # No body tag - use everything
            html_content = str(soup)

        # Validate minimum length
        if len(html_content) < 500:
            raise ValueError(
                f"8-K HTML content too short ({len(html_content)} chars) - may be empty or malformed"
            )

        LOG.info(f"✅ Extracted clean HTML ({len(html_content)} chars, tables preserved)")
        return html_content

    except Exception as e:
        LOG.error(f"Failed to extract 8-K HTML content: {e}")
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

                # Apply inline styles to ALL cells (fixes Gmail gridlines)
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    for cell_idx, cell in enumerate(cells):
                        if cell_idx == 0:
                            # First column: width + border + padding
                            cell['style'] = f"width: {first_col_width}; padding: 8px; border: 1px solid #ddd;"
                        else:
                            # Other columns: border + padding (width handled by table-layout: fixed)
                            cell['style'] = "padding: 8px; border: 1px solid #ddd;"

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
