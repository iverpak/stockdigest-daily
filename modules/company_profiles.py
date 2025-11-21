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
- Target length: 3,000-6,000 words
- Note: Complex filings may extend to 6,500 words. Avoid exceeding 7,000 words to prevent truncation.

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
## 18. RELATED ENTITIES (COMPETITORS, SUPPLIERS, CUSTOMERS)

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
- Target length: 2,500-5,000 words
- Note: Complex filings may extend to 5,500 words. Avoid exceeding 6,500 words to prevent truncation.

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
## 13. RELATED ENTITIES (COMPETITORS, SUPPLIERS, CUSTOMERS)

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

        # Find ALL exhibit files (any number: 1.1, 4.1, 10.1, 99.1, etc.)
        for i, row in enumerate(rows):
            cols = row.find_all('td')

            if len(cols) >= 4:
                # Handle 3 SEC table formats:
                # Format 1 (PLD/TSLA): Col 1 = "EXHIBIT 10.1" | Col 2 = filename | Col 3 = "EX-10.1" | Col 4 = size
                # Format 2 (LIN/TLN):  Col 1 = "EX-4.1"       | Col 2 = filename | Col 3 = "EX-4.1"  | Col 4 = size
                # Format 3 (WMT):      Col 1 = "PRESS RELEASE"| Col 2 = filename | Col 3 = "EX-99.1" | Col 4 = size

                doc_type = cols[1].text.strip()  # Try column 1 first (handles Formats 1 & 2)
                filename = cols[2].text.strip()  # Filename is always column 2

                # Detect 5-column format (most common) vs 4-column (rare/legacy)
                if len(cols) >= 5:
                    size_text = cols[4].text.strip()  # Size in column 4 for 5-column format
                    description = cols[1].text.strip()  # Description from column 1 (for Format 3)
                else:
                    size_text = cols[3].text.strip()  # Size in column 3 for 4-column format
                    description = cols[0].text.strip()  # Description from column 0 (sequence number)

                # Check if column 1 has exhibit type (Formats 1 & 2)
                doc_type_upper = doc_type.upper()
                is_exhibit = doc_type_upper.startswith('EXHIBIT') or doc_type_upper.startswith('EX-')

                # Fallback: Check column 3 if column 1 doesn't have exhibit type (Format 3 - WMT case)
                if not is_exhibit and len(cols) >= 4:
                    doc_type_col3 = cols[3].text.strip()
                    doc_type_col3_upper = doc_type_col3.upper()
                    if doc_type_col3_upper.startswith('EXHIBIT') or doc_type_col3_upper.startswith('EX-'):
                        doc_type = doc_type_col3  # Use column 3 as doc_type
                        doc_type_upper = doc_type_col3_upper
                        is_exhibit = True
                        LOG.info(f"📋 Format 3 detected (description in col 1): '{description}' → Exhibit type in col 3: '{doc_type}'")

                is_html = '.htm' in filename.lower()

                if is_exhibit and is_html:
                    link = cols[2].find('a')
                    if link and 'href' in link.attrs:
                        # Extract exhibit number (e.g., "99.1" from "EX-99.1" or "EXHIBIT 99.1")
                        # Use regex to extract just the numeric part (handles "99.1 PRESS RELEASE..." format)
                        import re
                        match = re.search(r'(\d+\.\d+)', doc_type)
                        if match:
                            exhibit_num = match.group(1)
                        else:
                            # Fallback to old logic if no decimal number found
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

                        # Clean up description (already set above based on format detection)
                        # If description is generic or a sequence number, use filename instead
                        if not description or description.isdigit() or len(description) <= 2:
                            # Use filename without extension as description
                            description = filename.replace('.htm', '').replace('_', ' ').title()
                        elif description.upper().startswith('EXHIBIT') or description.upper().startswith('EX-'):
                            # If description is just the exhibit type (Format 1/2), use filename
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
        # Use safe float conversion - non-numeric exhibits (like "MAIN") sort to end
        def safe_float_key(exhibit):
            try:
                return (0, float(exhibit['exhibit_number']))  # Numeric exhibits first
            except ValueError:
                return (1, exhibit['exhibit_number'])  # Non-numeric exhibits last, sorted alphabetically

        exhibits.sort(key=safe_float_key)

        LOG.info(f"✅ Found {len(exhibits)} HTML exhibits total")
        return exhibits

    except ValueError as e:
        # Distinguish between expected fallback signal vs real parsing errors
        if "No HTML exhibits found" in str(e):
            # Expected condition - some 8-Ks genuinely have no exhibits
            # Don't log as error - caller will handle fallback and log at INFO level
            raise
        else:
            # Real ValueError (e.g., "Documents table not found on index page")
            LOG.error(f"Failed to parse exhibits from documents page: {e}")
            raise
    except Exception as e:
        # Network errors, HTTP errors, BeautifulSoup parsing errors, etc.
        LOG.error(f"Failed to parse exhibits from documents page: {e}")
        raise


def get_main_8k_url(documents_url: str) -> Optional[str]:
    """
    Get URL of main 8-K HTML document from SEC documents page.

    Used as fallback when no exhibits are found. Some 8-Ks put all content
    in the main form body (Items 2.02, 7.01, 8.01, etc.) without separate exhibits.

    Args:
        documents_url: URL to SEC documents index page

    Returns:
        URL to main 8-K HTML document, or None if not found
    """
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    LOG.info(f"Looking for main 8-K body in documents page: {documents_url}")

    try:
        headers = {
            'User-Agent': 'StockDigest/1.0 (stockdigest.research@gmail.com)'
        }

        response = requests.get(documents_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='tableFile')

        if not table:
            return None

        rows = table.find_all('tr')[1:]  # Skip header

        # Find the main 8-K document (type "8-K" and HTML format)
        for row in rows:
            cols = row.find_all('td')

            if len(cols) >= 4:
                doc_type = cols[1].text.strip().upper()
                filename = cols[2].text.strip()

                # Match main 8-K form (not exhibits)
                # Use substring match to handle variations: "8-K", "FORM 8-K", "8-K/A" (amended)
                is_main_8k = '8-K' in doc_type
                is_html = '.htm' in filename.lower()

                if is_main_8k and is_html:
                    link = cols[2].find('a')
                    if link and 'href' in link.attrs:
                        main_url = urljoin(documents_url, link['href'])
                        # Strip iXBRL viewer wrapper
                        main_url = main_url.replace('/ix?doc=', '')
                        LOG.info(f"✅ Found main 8-K body: {main_url}")
                        return main_url

        LOG.warning("No main 8-K HTML document found")
        return None

    except Exception as e:
        LOG.error(f"Failed to get main 8-K URL: {e}")
        return None


def classify_exhibit_type(exhibit_num: str, description: str, char_count: int, item_codes: str = None) -> str:
    """
    Classify exhibit based on item code and exhibit number.

    Uses SEC filing structure for deterministic classification:
    - Heavy-duty items (1.01, 2.01, 2.02) + Exhibit 99.1/99.2 = comprehensive analysis
    - Everything else = lightweight press release

    Args:
        exhibit_num: Exhibit number (e.g., "99.1", "99.2")
        description: Exhibit description from SEC
        char_count: Character count of HTML content
        item_codes: Item codes from 8-K (e.g., "2.02, 9.01")

    Returns:
        "earnings_release" | "press_release"
    """
    # Heavy-duty items that need comprehensive analysis:
    # - 1.01: Material Agreement (major deals, debt, partnerships)
    # - 2.01: Acquisition/Disposition (M&A transactions)
    # - 2.02: Results of Operations (earnings releases)
    heavy_duty_items = ['1.01', '2.01', '2.02']

    if item_codes and exhibit_num in ['99.1', '99.2']:
        if any(item in item_codes for item in heavy_duty_items):
            return 'earnings_release'

    # All other 8-Ks = general press release
    return 'press_release'


def should_process_exhibit(exhibit_num: str) -> bool:
    """
    Determine if exhibit has information value and should be processed.

    Uses allowlist pattern: Only process exhibits in known high-value series.
    This filters out zero-value boilerplate (auditor letters, consents, certifications, XBRL).

    High-value exhibit series:
    - 1.x through 11.x: Contracts, agreements, instruments (debt, equity, M&A)
    - 99.x: Press releases, presentations, supplemental materials

    Zero-value exhibit series (excluded):
    - 16.x: Auditor acknowledgment letters (boilerplate)
    - 23.x: Consents of experts (legal boilerplate)
    - 24.x: Powers of attorney (legal documents)
    - 32.x: Section 906 certifications (mandatory boilerplate)
    - 101.x: XBRL files (machine-readable, not for AI)
    - 104.x: Cover page data (metadata only)

    Args:
        exhibit_num: Exhibit number in format "99.1", "10.2", etc. (NOT "EX-99.1")

    Returns:
        True if exhibit should be processed, False if it should be skipped
    """
    try:
        # Parse exhibit number - handles both "99.1" and "104" formats
        if '.' in exhibit_num:
            # Standard format: "99.1" → major=99, minor=1
            parts = exhibit_num.split('.')
            if len(parts) != 2:
                # Invalid format, include by default (safe approach)
                LOG.warning(f"⚠️  Unusual exhibit number format: '{exhibit_num}' - including by default")
                return True
            major_num = int(parts[0])
        else:
            # No decimal format: "104" → major=104
            major_num = int(exhibit_num)

        # Include: 1.x through 11.x (contracts, agreements, instruments)
        if 1 <= major_num <= 11:
            return True

        # Include: 99.x (press releases, presentations, supplemental)
        if major_num == 99:
            return True

        # Exclude: Everything else (16.x+, 23.x, 24.x, 32.x, 101.x, 104.x)
        LOG.info(f"⏭️  Skipping Exhibit {exhibit_num} (zero info value - series {major_num}.x)")
        return False

    except (ValueError, IndexError) as e:
        # Can't parse exhibit number, include by default (safe approach)
        LOG.warning(f"⚠️  Failed to parse exhibit number '{exhibit_num}': {e} - including by default")
        return True


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
            'Range': 'bytes=0-15000'  # Fetch first 15KB to get past iXBRL overhead
        }

        response = requests.get(sec_html_url, headers=headers, timeout=10)
        text = response.text

        # Extract item codes (format: "Item 2.02" or "Item 2.02.")
        # Works for both regular HTML and iXBRL formats
        items = re.findall(r'Item\s+(\d+\.\d+)', text, re.IGNORECASE)
        item_codes = ', '.join(sorted(set(items[:3])))  # Dedupe and limit to first 3

        # Get item description for primary item
        item_description = "Other Events"  # Default
        if items:
            primary_item = items[0]
            item_description = ITEM_CODE_MAP.get(primary_item, "Other Events")

        # Strip HTML tags before extracting title to avoid capturing HTML fragments
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text(separator=' ')

        # Extract title from clean text content
        # Look for patterns like "announces", "reports", "completes", etc.
        title_match = re.search(
            r'(announces?|reports?|completes?|enters?|acquires?|appoints?)[^\.]{10,200}',
            clean_text,
            re.IGNORECASE
        )
        parsed_title = title_match.group(0).strip() if title_match else ""

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
            content_element = body
        else:
            # No body tag - use everything
            content_element = soup

        # Convert relative image URLs to absolute SEC.gov URLs
        # Example: <img src="pld-ex99_1s1.jpg"> → <img src="https://www.sec.gov/.../pld-ex99_1s1.jpg">
        from urllib.parse import urljoin
        for img in content_element.find_all('img'):
            if 'src' in img.attrs and not img['src'].startswith('http'):
                # Convert relative path to absolute URL
                img['src'] = urljoin(exhibit_url, img['src'])

        html_content = str(content_element)

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
            "max_output_tokens": 65536  # Official Gemini 2.5 Flash limit (increased from 32K to avoid artificial ceiling)
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

        # ====================================================================
        # DIAGNOSTIC LOGGING - Detect truncation, safety filters, recitation
        # ====================================================================

        # Store finish_reason_name for later use in truncation check
        finish_reason_name = None

        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]

            # 1. Log finish reason (critical for diagnosing truncation)
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason:
                finish_reason_name = str(finish_reason).split('.')[-1] if hasattr(finish_reason, 'name') else str(finish_reason)
                LOG.info(f"[{ticker}] 🔍 Gemini finish_reason: {finish_reason_name}")

                # Warn on abnormal finish reasons
                if 'MAX_TOKENS' in finish_reason_name:
                    LOG.error(f"[{ticker}] ❌ OUTPUT TRUNCATED - Hit max_output_tokens limit!")
                    LOG.error(f"[{ticker}]   Output length: {len(profile_markdown):,} chars")
                    LOG.error(f"[{ticker}]   Configured max_output_tokens: {generation_config.get('max_output_tokens', 'unknown')}")
                elif 'SAFETY' in finish_reason_name:
                    LOG.error(f"[{ticker}] 🚨 SAFETY FILTER triggered - Output may be incomplete or garbage")
                elif 'RECITATION' in finish_reason_name:
                    LOG.error(f"[{ticker}] 📋 RECITATION detected - Gemini aborted due to copyright concerns")
                    LOG.error(f"[{ticker}]   This means Gemini detected verbatim copying from the {filing_type}")
                    LOG.error(f"[{ticker}]   Solution: Update prompt to emphasize synthesis/paraphrasing")
                elif 'OTHER' in finish_reason_name:
                    LOG.error(f"[{ticker}] ⚠️ Abnormal finish_reason: {finish_reason_name}")
            else:
                LOG.warning(f"[{ticker}] ⚠️ No finish_reason available from Gemini response")

            # 2. Check safety ratings
            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                safety_concerns = []
                for rating in candidate.safety_ratings:
                    if hasattr(rating, 'probability'):
                        prob_name = str(rating.probability).split('.')[-1] if hasattr(rating.probability, 'name') else str(rating.probability)
                        if prob_name not in ['NEGLIGIBLE', 'HARM_PROBABILITY_UNSPECIFIED']:
                            category_name = str(rating.category).split('.')[-1] if hasattr(rating.category, 'name') else str(rating.category)
                            safety_concerns.append(f"{category_name}={prob_name}")

                if safety_concerns:
                    LOG.error(f"[{ticker}] 🚨 Gemini safety filter concerns: {safety_concerns}")

        # 3. Detect garbage pattern (like Phase 2 safety filter output)
        if len(profile_markdown) > 100:
            sample = profile_markdown[:min(1000, len(profile_markdown))]
            unique_chars = len(set(sample))

            # Check for repetitive pattern (safety filter signature)
            is_repetitive = unique_chars < 10  # Less than 10 unique chars = garbage

            # Check for specific garbage patterns
            dash_count = profile_markdown.count('-')
            dash_ratio = dash_count / len(profile_markdown) if len(profile_markdown) > 0 else 0
            has_dash_pattern = dash_ratio > 0.3  # >30% dashes

            number_count = profile_markdown.count('1') + profile_markdown.count('2') + profile_markdown.count('3')
            number_ratio = number_count / len(profile_markdown) if len(profile_markdown) > 0 else 0
            has_number_spam = number_ratio > 0.2  # >20% numbers

            if is_repetitive or has_dash_pattern or has_number_spam:
                reason_parts = []
                if is_repetitive:
                    reason_parts.append(f"unique_chars={unique_chars}")
                if has_dash_pattern:
                    reason_parts.append(f"dashes={dash_count} ({dash_ratio:.1%})")
                if has_number_spam:
                    reason_parts.append(f"numbers={number_count} ({number_ratio:.1%})")

                LOG.error(f"[{ticker}] 🚨 GARBAGE OUTPUT detected - likely safety filter")
                LOG.error(f"[{ticker}]   Pattern: {', '.join(reason_parts)}")
                LOG.error(f"[{ticker}]   First 200 chars: {profile_markdown[:200]}")
                LOG.error(f"[{ticker}]   Last 200 chars: {profile_markdown[-200:]}")

        # 4. Validate section completeness
        expected_sections = 18 if filing_type == '10-K' else 13
        section_count = profile_markdown.count('## ')

        if section_count < expected_sections:
            LOG.warning(f"[{ticker}] ⚠️ INCOMPLETE: Only {section_count}/{expected_sections} sections generated")
            LOG.warning(f"[{ticker}]   Expected {expected_sections} sections for {filing_type}, got {section_count}")

            # List which sections are present (first 50 chars of each ## header)
            sections_found = []
            for line in profile_markdown.split('\n'):
                if line.startswith('## '):
                    sections_found.append(line[:50])

            if sections_found:
                LOG.info(f"[{ticker}]   Sections found: {len(sections_found)}")
                for i, section in enumerate(sections_found[:5], 1):  # Log first 5
                    LOG.info(f"[{ticker}]     {i}. {section}")
                if len(sections_found) > 5:
                    LOG.info(f"[{ticker}]     ... and {len(sections_found) - 5} more")

        # 5. Check for mid-sentence truncation
        last_chars = profile_markdown.rstrip()[-100:] if len(profile_markdown) > 100 else profile_markdown
        clean_endings = ('.', ')', ']', '*', '`', '>', '|', ':', ';')  # Common markdown endings

        # Strip trailing quotes/backticks before checking punctuation
        text_to_check = profile_markdown.rstrip().rstrip('"\'`')

        if not text_to_check.endswith(clean_endings):
            # Determine severity based on finish_reason
            if finish_reason_name and 'STOP' in finish_reason_name:
                # Model finished naturally - likely false positive (e.g., ends with quote)
                LOG.warning(f"[{ticker}] ⚠️ Suspicious ending detected (finish_reason=STOP, likely false positive)")
                LOG.warning(f"[{ticker}]   Last 100 chars: {last_chars}")
                LOG.warning(f"[{ticker}]   Expected endings: {clean_endings}")
            elif finish_reason_name and 'MAX_TOKENS' in finish_reason_name:
                # Hit token limit with bad punctuation - real truncation
                LOG.error(f"[{ticker}] ✂️ MID-SENTENCE TRUNCATION detected!")
                LOG.error(f"[{ticker}]   Last 100 chars: {last_chars}")
                LOG.error(f"[{ticker}]   Does not end with: {clean_endings}")
            elif finish_reason_name and ('SAFETY' in finish_reason_name or 'RECITATION' in finish_reason_name):
                # Safety/recitation issue - different problem
                LOG.error(f"[{ticker}] ✂️ Incomplete output due to {finish_reason_name}")
                LOG.error(f"[{ticker}]   Last 100 chars: {last_chars}")
            else:
                # No finish_reason available or OTHER - uncertain, use warning
                LOG.warning(f"[{ticker}] ⚠️ Suspicious ending detected (finish_reason={finish_reason_name or 'unknown'})")
                LOG.warning(f"[{ticker}]   Last 100 chars: {last_chars}")
                LOG.warning(f"[{ticker}]   Expected endings: {clean_endings}")

        # ====================================================================
        # END DIAGNOSTIC LOGGING
        # ====================================================================

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
            "max_output_tokens": 65536  # Official Gemini 2.5 Flash limit (increased from 32K to avoid artificial ceiling)
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


# ==============================================================================
# PARSED PRESS RELEASES - Unified summaries for FMP PRs and 8-K exhibits
# ==============================================================================

def load_press_release_prompt() -> str:
    """Load the press release summary prompt from file."""
    try:
        prompt_path = os.path.join(os.path.dirname(__file__), '_press_release_summary_prompt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        LOG.error(f"Failed to load press release prompt: {e}")
        raise


def generate_parsed_press_release_with_gemini(
    ticker: str,
    company_name: str,
    content: str,
    document_title: str,
    source_type: str,  # 'fmp' or '8k'
    item_codes: str = None,  # 8-K item codes (e.g., "2.02, 9.01")
    gemini_api_key: str = None
) -> Optional[Dict]:
    """
    Generate structured summary of press release or 8-K exhibit using Gemini 2.5 Flash.

    This creates a unified summary format that can be fed into Phase 1 executive summaries
    as a pseudo-article that auto-passes triage.

    Args:
        ticker: Stock ticker (e.g., 'AAPL')
        company_name: Company name for context
        content: Full text of press release or 8-K exhibit
        document_title: Title of the document
        source_type: 'fmp' for FMP press releases, '8k' for SEC 8-K exhibits
        item_codes: 8-K item codes for context (optional)
        gemini_api_key: Gemini API key

    Returns:
        {
            'parsed_summary': str (structured extraction),
            'metadata': {
                'model': str,
                'generation_time_seconds': int,
                'token_count_input': int,
                'token_count_output': int
            }
        }
    """
    if not gemini_api_key:
        LOG.error("❌ Gemini API key not configured for parsed press release generation")
        return None

    if not content or len(content.strip()) < 100:
        LOG.error(f"❌ Content too short for {ticker} ({len(content) if content else 0} chars)")
        return None

    try:
        genai.configure(api_key=gemini_api_key)

        # Load prompt
        prompt_template = load_press_release_prompt()

        # Build context header based on source type
        if source_type == '8k':
            source_context = f"SEC 8-K Filing"
            if item_codes:
                source_context += f" (Items: {item_codes})"
        else:
            source_context = "Company Press Release"

        LOG.info(f"Generating parsed summary for {ticker} - {source_context}")
        LOG.info(f"Document: {document_title[:80]}...")
        LOG.info(f"Content length: {len(content):,} chars")

        # Gemini 2.5 Flash
        model = genai.GenerativeModel('gemini-2.5-flash')

        generation_config = {
            "temperature": 0.0,  # Maximum determinism for consistent extraction
            "max_output_tokens": 8192  # Sufficient for 2-6 paragraph summary
        }

        start_time = datetime.now(timezone.utc)

        # Build user content with document context
        user_content = f"""Company: {company_name} ({ticker})
Document Type: {source_context}
Title: {document_title}

---
DOCUMENT CONTENT:

{content[:400000]}
"""

        # Combine prompt with content
        full_prompt = f"{prompt_template}\n\n---\n\n{user_content}"

        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )

        end_time = datetime.now(timezone.utc)
        generation_time = int((end_time - start_time).total_seconds())

        parsed_summary = response.text.strip()

        # Get token counts from usage metadata
        token_count_input = 0
        token_count_output = 0
        if hasattr(response, 'usage_metadata'):
            token_count_input = getattr(response.usage_metadata, 'prompt_token_count', 0)
            token_count_output = getattr(response.usage_metadata, 'candidates_token_count', 0)

        # Log success
        word_count = len(parsed_summary.split())
        LOG.info(f"✅ Generated parsed summary for {ticker}: {word_count} words in {generation_time}s")
        LOG.info(f"   Tokens: {token_count_input:,} input, {token_count_output:,} output")

        return {
            'parsed_summary': parsed_summary,
            'metadata': {
                'model': 'gemini-2.5-flash',
                'generation_time_seconds': generation_time,
                'token_count_input': token_count_input,
                'token_count_output': token_count_output
            }
        }

    except Exception as e:
        LOG.error(f"❌ Gemini generation failed for {ticker} parsed PR: {e}")
        LOG.error(traceback.format_exc())
        return None


def load_8k_filing_prompt() -> str:
    """Load the 8-K filing prompt from file."""
    try:
        prompt_path = os.path.join(os.path.dirname(__file__), '_8k_filing_prompt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        LOG.error(f"Failed to load 8-K filing prompt: {e}")
        raise


def convert_earnings_json_to_markdown(json_data: dict, ticker: str, company_name: str) -> str:
    """
    Convert earnings release JSON output to markdown format for storage and display.

    Args:
        json_data: Parsed JSON from Gemini
        ticker: Stock ticker
        company_name: Company name

    Returns:
        Markdown formatted string
    """
    metadata = json_data.get('metadata', {})
    sections = json_data.get('sections', {})
    lines = []

    # Header - use report_title if available
    report_title = metadata.get('report_title', 'Earnings Release Summary')
    lines.append(f"# {company_name} ({ticker}) - {report_title}\n")

    # Bottom Line
    if 'bottom_line' in sections and sections['bottom_line'].get('content'):
        lines.append("## Bottom Line\n")
        lines.append(sections['bottom_line']['content'])
        lines.append("")

    # Financial Results
    if 'financial_results' in sections and sections['financial_results']:
        lines.append("## Financial Results\n")
        for bullet in sections['financial_results']:
            topic = bullet.get('topic_label', '')
            content = bullet.get('content', '')
            lines.append(f"**{topic}**")
            lines.append(content)
            lines.append("")

    # Operational Metrics
    if 'operational_metrics' in sections and sections['operational_metrics']:
        lines.append("## Operational Metrics\n")
        for bullet in sections['operational_metrics']:
            topic = bullet.get('topic_label', '')
            content = bullet.get('content', '')
            lines.append(f"**{topic}**")
            lines.append(content)
            lines.append("")

    # Major Developments
    if 'major_developments' in sections and sections['major_developments']:
        lines.append("## Major Developments\n")
        for bullet in sections['major_developments']:
            topic = bullet.get('topic_label', '')
            content = bullet.get('content', '')
            lines.append(f"**{topic}**")
            lines.append(content)
            lines.append("")

    # Guidance
    if 'guidance' in sections and sections['guidance']:
        lines.append("## Guidance\n")
        for bullet in sections['guidance']:
            topic = bullet.get('topic_label', '')
            content = bullet.get('content', '')
            lines.append(f"**{topic}**")
            lines.append(content)
            lines.append("")

    # Strategic Initiatives
    if 'strategic_initiatives' in sections and sections['strategic_initiatives']:
        lines.append("## Strategic Initiatives\n")
        for bullet in sections['strategic_initiatives']:
            topic = bullet.get('topic_label', '')
            content = bullet.get('content', '')
            lines.append(f"**{topic}**")
            lines.append(content)
            lines.append("")

    # Risk Factors
    if 'risk_factors' in sections and sections['risk_factors']:
        lines.append("## Risk Factors\n")
        for bullet in sections['risk_factors']:
            topic = bullet.get('topic_label', '')
            content = bullet.get('content', '')
            lines.append(f"**{topic}**")
            lines.append(content)
            lines.append("")

    # Industry & Competitive Landscape
    if 'industry_competitive' in sections and sections['industry_competitive']:
        lines.append("## Industry & Competitive Landscape\n")
        for bullet in sections['industry_competitive']:
            topic = bullet.get('topic_label', '')
            content = bullet.get('content', '')
            lines.append(f"**{topic}**")
            lines.append(content)
            lines.append("")

    # Capital Allocation
    if 'capital_allocation' in sections and sections['capital_allocation']:
        lines.append("## Capital Allocation\n")
        for bullet in sections['capital_allocation']:
            topic = bullet.get('topic_label', '')
            content = bullet.get('content', '')
            lines.append(f"**{topic}**")
            lines.append(content)
            lines.append("")

    # Upside Scenario
    if 'upside_scenario' in sections and sections['upside_scenario'].get('content'):
        lines.append("## Upside Scenario\n")
        lines.append(sections['upside_scenario']['content'])
        lines.append("")

    # Downside Scenario
    if 'downside_scenario' in sections and sections['downside_scenario'].get('content'):
        lines.append("## Downside Scenario\n")
        lines.append(sections['downside_scenario']['content'])
        lines.append("")

    # Key Variables
    if 'key_variables' in sections and sections['key_variables']:
        lines.append("## Key Variables to Monitor\n")
        for bullet in sections['key_variables']:
            topic = bullet.get('topic_label', '')
            content = bullet.get('content', '')
            lines.append(f"**{topic}**")
            lines.append(content)
            lines.append("")

    return "\n".join(lines)


def generate_earnings_release_with_gemini(
    ticker: str,
    company_name: str,
    content: str,
    document_title: str,
    item_codes: str = None,
    gemini_api_key: str = None,
    is_pdf: bool = False,
    pdf_bytes: bytes = None
) -> Optional[Dict]:
    """
    Generate structured summary of earnings release using Gemini 2.5 Flash.

    This creates a comprehensive JSON output with 11 sections that can be
    converted to markdown for storage in parsed_press_releases.

    Args:
        ticker: Stock ticker (e.g., 'AAPL')
        company_name: Company name for context
        content: Full text of earnings release (HTML or plain text)
        document_title: Title of the document
        item_codes: 8-K item codes for context (e.g., "2.02")
        gemini_api_key: Gemini API key
        is_pdf: Whether content is PDF (use pdf_bytes instead)
        pdf_bytes: Raw PDF bytes for multimodal processing

    Returns:
        {
            'parsed_summary': str (markdown formatted),
            'json_data': dict (raw JSON output),
            'metadata': {
                'model': str,
                'generation_time_seconds': int,
                'token_count_input': int,
                'token_count_output': int
            }
        }
    """
    if not gemini_api_key:
        LOG.error("❌ Gemini API key not configured for earnings release generation")
        return None

    if not is_pdf and (not content or len(content.strip()) < 100):
        LOG.error(f"❌ Content too short for {ticker} ({len(content) if content else 0} chars)")
        return None

    if is_pdf and not pdf_bytes:
        LOG.error(f"❌ PDF bytes not provided for {ticker}")
        return None

    try:
        import google.generativeai as genai
        import base64
        import json

        genai.configure(api_key=gemini_api_key)

        # Load prompt
        prompt_template = load_8k_filing_prompt()

        # Build context header
        source_context = "Earnings Release (8-K)"
        if item_codes:
            source_context += f" - Items: {item_codes}"

        LOG.info(f"Generating earnings release summary for {ticker} - {source_context}")
        LOG.info(f"Document: {document_title[:80]}...")

        # Gemini 2.5 Flash
        model = genai.GenerativeModel('gemini-2.5-flash')

        # CHANGE #1: Increased token limit from 16K to 32K (2x buffer for complex filings)
        generation_config = {
            "temperature": 0.0,  # Maximum determinism
            "max_output_tokens": 32000,  # Increased from 16000 to handle Transaction Details section
            "response_mime_type": "application/json"  # Request JSON output
        }

        # CHANGE #3: Add retry logic for transient Gemini errors (copy from Phase 1 pattern)
        max_retries = 1
        response = None
        generation_time = 0

        for attempt in range(max_retries + 1):
            try:
                start_time = datetime.now(timezone.utc)

                if is_pdf and pdf_bytes:
                    # Multimodal PDF processing
                    LOG.info(f"[{ticker}] Processing PDF ({len(pdf_bytes):,} bytes) with multimodal...")

                    pdf_part = {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": base64.b64encode(pdf_bytes).decode('utf-8')
                        }
                    }

                    user_content = f"""Company: {company_name} ({ticker})
Document Type: {source_context}
Title: {document_title}

Analyze this earnings release PDF and extract all material information."""

                    response = model.generate_content(
                        [prompt_template, user_content, pdf_part],
                        generation_config=generation_config
                    )
                else:
                    # Text-based processing with multimodal image support
                    content_length = len(content) if content else 0
                    LOG.info(f"[{ticker}] Content length: {content_length:,} chars")

                    # Extract images from HTML content
                    from bs4 import BeautifulSoup
                    import requests

                    soup = BeautifulSoup(content, 'html.parser')
                    img_tags = soup.find_all('img')

                    # Download images and build multimodal parts
                    image_parts = []
                    if img_tags:
                        LOG.info(f"[{ticker}] Found {len(img_tags)} images in HTML content")

                        for idx, img in enumerate(img_tags, 1):
                            img_url = img.get('src', '')
                            if not img_url or not img_url.startswith('http'):
                                continue

                            try:
                                # Download image
                                img_response = requests.get(
                                    img_url,
                                    headers={'User-Agent': 'StockDigest/1.0 (stockdigest.research@gmail.com)'},
                                    timeout=30
                                )
                                img_response.raise_for_status()

                                img_bytes = img_response.content
                                img_size = len(img_bytes)

                                # Detect MIME type from URL or content-type header
                                content_type = img_response.headers.get('content-type', 'image/jpeg')
                                if 'png' in img_url.lower() or 'png' in content_type:
                                    mime_type = 'image/png'
                                elif 'gif' in img_url.lower() or 'gif' in content_type:
                                    mime_type = 'image/gif'
                                else:
                                    mime_type = 'image/jpeg'

                                # Skip GIF images (Gemini doesn't support GIF format)
                                if mime_type == 'image/gif':
                                    LOG.info(f"[{ticker}]    ⏭️  Skipping image {idx}/{len(img_tags)}: {img_size:,} bytes (GIF not supported by Gemini)")
                                    continue

                                # Create inline_data part for Gemini
                                image_parts.append({
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": base64.b64encode(img_bytes).decode('utf-8')
                                    }
                                })

                                LOG.info(f"[{ticker}]    ✅ Downloaded image {idx}/{len(img_tags)}: {img_size:,} bytes ({mime_type})")

                            except Exception as img_error:
                                LOG.warning(f"[{ticker}]    ⚠️ Failed to download image {idx}: {img_error}")
                                continue

                        LOG.info(f"[{ticker}] Successfully downloaded {len(image_parts)}/{len(img_tags)} images for multimodal processing")

                    # Build user content (text part)
                    user_content = f"""Company: {company_name} ({ticker})
Document Type: {source_context}
Title: {document_title}

---
DOCUMENT CONTENT:

{content[:400000]}
"""

                    # Build multimodal request
                    if image_parts:
                        # Multimodal: prompt + text + images
                        LOG.info(f"[{ticker}] Sending multimodal request to Gemini ({len(image_parts)} images)")
                        parts = [prompt_template, user_content] + image_parts
                        response = model.generate_content(
                            parts,
                            generation_config=generation_config
                        )
                    else:
                        # Text only (no images found)
                        full_prompt = f"{prompt_template}\n\n---\n\n{user_content}"
                        response = model.generate_content(
                            full_prompt,
                            generation_config=generation_config
                        )

                end_time = datetime.now(timezone.utc)
                generation_time = int((end_time - start_time).total_seconds())

                # Success - break retry loop
                break

            except Exception as e:
                error_str = str(e)

                # Check for retryable errors (same as Phase 1 executive summary)
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
                    wait_time = 2 ** attempt  # 1s
                    LOG.warning(f"[{ticker}] ⚠️ Gemini error (attempt {attempt + 1}/{max_retries + 1}): {error_str[:200]}")
                    LOG.warning(f"[{ticker}] 🔄 Retrying in {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error or max retries reached
                    LOG.error(f"[{ticker}] ❌ Gemini generation failed after {attempt + 1} attempts: {error_str}")
                    return None

        # Check if we got a response
        if response is None:
            LOG.error(f"[{ticker}] ❌ No response from Gemini after {max_retries + 1} attempts")
            return None

        # CHANGE #2: Add finish_reason logging (copy from 10-K generation pattern)
        finish_reason_name = None
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]

            # Log finish reason (critical for diagnosing truncation)
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason:
                finish_reason_name = str(finish_reason).split('.')[-1] if hasattr(finish_reason, 'name') else str(finish_reason)
                LOG.info(f"[{ticker}] 🔍 Gemini finish_reason: {finish_reason_name}")

                # Warn on abnormal finish reasons
                if 'MAX_TOKENS' in finish_reason_name:
                    LOG.error(f"[{ticker}] ❌ OUTPUT TRUNCATED - Hit max_output_tokens limit!")
                    LOG.error(f"[{ticker}]   Configured max_output_tokens: {generation_config.get('max_output_tokens', 'unknown')}")
                elif 'SAFETY' in finish_reason_name:
                    LOG.error(f"[{ticker}] 🚨 SAFETY FILTER triggered - Output may be incomplete")
                elif 'RECITATION' in finish_reason_name:
                    LOG.error(f"[{ticker}] 📋 RECITATION detected - Gemini detected verbatim copying")
                elif 'OTHER' in finish_reason_name:
                    LOG.error(f"[{ticker}] ⚠️ Abnormal finish_reason: {finish_reason_name}")
            else:
                LOG.warning(f"[{ticker}] ⚠️ No finish_reason available from Gemini response")

        # CHANGE #4: Replace custom JSON extraction with robust json_utils.py (4-tier fallback)
        response_text = response.text.strip()
        from modules.json_utils import extract_json_from_claude_response

        json_data = extract_json_from_claude_response(response_text, ticker)
        if not json_data:
            LOG.error(f"[{ticker}] ❌ JSON extraction failed - see detailed error above")
            return None

        # Convert JSON to markdown
        markdown_summary = convert_earnings_json_to_markdown(json_data, ticker, company_name)

        # Get token counts
        token_count_input = 0
        token_count_output = 0
        if hasattr(response, 'usage_metadata'):
            token_count_input = getattr(response.usage_metadata, 'prompt_token_count', 0)
            token_count_output = getattr(response.usage_metadata, 'candidates_token_count', 0)

        # Log success
        word_count = len(markdown_summary.split())
        LOG.info(f"✅ Generated earnings release summary for {ticker}: {word_count} words in {generation_time}s")
        LOG.info(f"   Tokens: {token_count_input:,} input, {token_count_output:,} output")

        # Extract report metadata from AI output
        ai_metadata = json_data.get('metadata', {})
        report_title = ai_metadata.get('report_title', 'Earnings Release Summary')
        fiscal_quarter = ai_metadata.get('fiscal_quarter', '')
        fiscal_year = ai_metadata.get('fiscal_year', '')

        LOG.info(f"   Report: {report_title} ({fiscal_quarter} {fiscal_year})")

        return {
            'parsed_summary': markdown_summary,
            'json_data': json_data,
            'metadata': {
                'model': 'gemini-2.5-flash',
                'generation_time_seconds': generation_time,
                'token_count_input': token_count_input,
                'token_count_output': token_count_output,
                'report_title': report_title,
                'fiscal_quarter': fiscal_quarter,
                'fiscal_year': fiscal_year
            }
        }

    except Exception as e:
        LOG.error(f"❌ Gemini generation failed for {ticker} earnings release: {e}")
        LOG.error(traceback.format_exc())
        return None






def get_all_parsed_press_releases(
    db_connection,
    limit: int = 100
) -> List[Dict]:
    """
    Fetch all parsed press releases for Research viewer, sorted by date descending.

    Args:
        db_connection: Database connection
        limit: Maximum number of results

    Returns:
        List of parsed press release dicts
    """
    try:
        cur = db_connection.cursor()

        cur.execute("""
            SELECT
                cr.id, cr.ticker, cr.company_name, cr.source_type, cr.source_id,
                cr.filing_date as document_date, cr.report_title as document_title,
                NULL as source_url,
                cr.exhibit_number,
                cr.item_codes,
                LENGTH(cr.summary_markdown) as char_count,
                cr.ai_model,
                cr.processing_duration_seconds,
                FALSE as fed_to_phase1,
                cr.fiscal_year,
                cr.fiscal_quarter,
                cr.generated_at
            FROM company_releases cr
            ORDER BY cr.filing_date DESC
            LIMIT %s
        """, (limit,))

        results = []
        for row in cur.fetchall():
            # Handle both dict and tuple cursor types
            if isinstance(row, dict):
                results.append(dict(row))
            else:
                columns = [desc[0] for desc in cur.description]
                results.append(dict(zip(columns, row)))

        cur.close()
        return results

    except Exception as e:
        LOG.error(f"❌ Failed to fetch all parsed PRs: {e}")
        return []




def parse_company_release_sections(json_output: dict) -> Dict[str, List[str]]:
    """
    Parse Gemini JSON output from _8k_filing_prompt into sections dict.

    Input: Raw JSON from Gemini with structure:
    {
      "metadata": {
        "report_title": "Q3 2025 Earnings Release",
        "fiscal_quarter": "Q3",
        "fiscal_year": "2025"
      },
      "sections": {
        "bottom_line": {"content": "...", "word_count": 150},
        "financial_results": [
          {"bullet_id": "...", "topic_label": "...", "content": "..."}
        ],
        ...
      }
    }

    Output: Dict compatible with email builder
    {
      "bottom_line": ["paragraph text"],
      "financial_results": ["Topic Label: content", "Topic Label: content"],
      "upside_scenario": ["paragraph text"],
      ...
    }
    """
    sections = {
        "bottom_line": [],
        "financial_results": [],
        "operational_metrics": [],
        "major_developments": [],
        "guidance": [],
        "strategic_initiatives": [],
        "risk_factors": [],
        "industry_competitive": [],
        "capital_allocation": [],
        "upside_scenario": [],
        "downside_scenario": [],
        "key_variables": []
    }

    if not json_output or 'sections' not in json_output:
        return sections

    json_sections = json_output['sections']

    # Handle paragraph sections (bottom_line, upside/downside scenarios)
    for section_key in ['bottom_line', 'upside_scenario', 'downside_scenario']:
        if section_key in json_sections and json_sections[section_key]:
            section_data = json_sections[section_key]
            if isinstance(section_data, dict) and 'content' in section_data:
                content = section_data['content']
                # Only add if content is not empty/whitespace
                if content and content.strip():
                    sections[section_key] = [content]
            elif isinstance(section_data, str):
                # Only add if string is not empty/whitespace
                if section_data and section_data.strip():
                    sections[section_key] = [section_data]

    # Handle bullet sections (all others)
    bullet_sections = [
        'financial_results', 'operational_metrics', 'major_developments',
        'guidance', 'strategic_initiatives', 'risk_factors',
        'industry_competitive', 'capital_allocation', 'key_variables'
    ]

    for section_key in bullet_sections:
        if section_key in json_sections and json_sections[section_key]:
            section_data = json_sections[section_key]
            if isinstance(section_data, list):
                for bullet in section_data:
                    if isinstance(bullet, dict) and 'topic_label' in bullet and 'content' in bullet:
                        # Format: "Topic Label: content"
                        formatted_bullet = f"{bullet['topic_label']}: {bullet['content']}"
                        sections[section_key].append(formatted_bullet)

    return sections


def build_company_release_html(sections: Dict[str, List[str]]) -> str:
    """
    Build HTML for company release sections.

    - Uses regex to bold everything before `:` (like transcripts)
    - Strips any markdown formatting
    - Uses ## headers (no emojis)
    - Hides empty sections
    - Bullet vs paragraph logic

    Similar structure to build_transcript_summary_html() but separate function.
    """
    import re

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
        Regex pattern: Everything before first colon gets bolded
        """
        text = strip_markdown_formatting(text)
        pattern = r'^([^:]{2,130}?:)(\s)'
        replacement = r'<strong>\1</strong>\2'
        return re.sub(pattern, replacement, text)

    def build_section(title: str, bullets: List[str], use_bullets: bool = True) -> str:
        """Helper to build a section with title and content"""
        if not bullets:
            return ""

        html = f'<div style="margin-bottom: 24px;">\n'
        html += f'  <h3 style="font-size: 15px; font-weight: 700; color: #1e40af; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 0.5px;">{title}</h3>\n'

        if use_bullets:
            html += '  <ul style="margin: 0; padding-left: 20px; color: #374151; font-size: 13px; line-height: 1.6;">\n'
            for bullet in bullets:
                # Apply label bolding
                processed_bullet = bold_bullet_labels(bullet)
                html += f'    <li style="margin-bottom: 6px;">{processed_bullet}</li>\n'
            html += '  </ul>\n'
        else:
            # Paragraph format (for bottom_line, upside/downside)
            content_filtered = [strip_markdown_formatting(line) for line in bullets if line.strip()]
            content = '<br>'.join(content_filtered)
            html += f'  <div style="color: #374151; font-size: 13px; line-height: 1.6;">{content}</div>\n'

        html += '</div>\n'
        return html

    html = ""

    # Render sections in fixed order (clean headers without emojis)
    html += build_section("Bottom Line", sections.get("bottom_line", []), use_bullets=False)
    html += build_section("Financial Results", sections.get("financial_results", []), use_bullets=True)
    html += build_section("Operational Metrics", sections.get("operational_metrics", []), use_bullets=True)
    html += build_section("Major Developments", sections.get("major_developments", []), use_bullets=True)
    html += build_section("Guidance", sections.get("guidance", []), use_bullets=True)
    html += build_section("Strategic Initiatives", sections.get("strategic_initiatives", []), use_bullets=True)
    html += build_section("Risk Factors & Headwinds", sections.get("risk_factors", []), use_bullets=True)
    html += build_section("Industry & Competitive Dynamics", sections.get("industry_competitive", []), use_bullets=True)
    html += build_section("Capital Allocation", sections.get("capital_allocation", []), use_bullets=True)
    html += build_section("Upside Scenario", sections.get("upside_scenario", []), use_bullets=False)
    html += build_section("Downside Scenario", sections.get("downside_scenario", []), use_bullets=False)
    html += build_section("Key Variables to Monitor", sections.get("key_variables", []), use_bullets=True)

    return html


def generate_company_release_email(
    ticker: str,
    company_name: str,
    release_type: str,  # '8k' or 'fmp_press_release'
    filing_date: str,   # 'Nov 19, 2024'
    json_output: dict,  # Raw Gemini JSON
    stock_price: str = None,
    price_change_pct: str = None,
    price_change_color: str = "#4ade80",
    ytd_return_pct: str = None,
    ytd_return_color: str = "#4ade80",
    market_status: str = "LAST CLOSE",
    return_label: str = "1D"
) -> Dict[str, str]:
    """
    Generate company release email HTML.

    Flow:
    1. Parse JSON → sections dict
    2. Build HTML from sections
    3. Render template with report_type_label="COMPANY RELEASE"
    4. Return {html, subject}

    Returns:
        {
            "html": Full email HTML,
            "subject": Email subject line
        }
    """
    from jinja2 import Template
    import os

    LOG.info(f"Generating company release email for {ticker} ({release_type})")

    # Parse sections from JSON
    sections = parse_company_release_sections(json_output)

    # Build summary HTML
    summary_html = build_company_release_html(sections)

    # Get report title from metadata
    report_title = json_output.get('metadata', {}).get('report_title', 'Company Release')

    # Load template
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates', 'email_research_report.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()

    research_template = Template(template_content)

    # Configure labels
    report_type_label = "COMPANY RELEASE"
    date_label = f"Filing Date: {filing_date}"
    filing_date_display = f"Filing Date: {filing_date}"

    # Render template with variables
    html = research_template.render(
        report_title=f"{ticker} Research Summary",
        report_type_label=report_type_label,
        company_name=company_name,
        ticker=ticker,
        industry=None,  # Company releases don't include industry
        fiscal_period=filing_date,
        date_label=date_label,
        filing_date=filing_date_display,
        stock_price=stock_price,
        price_change_pct=price_change_pct,
        price_change_color=price_change_color,
        ytd_return_pct=ytd_return_pct,
        ytd_return_color=ytd_return_color,
        market_status=market_status,
        return_label=return_label,
        content_html=summary_html
    )

    # Subject line with report title
    subject = f"📄 {ticker} - {report_title} - {filing_date}"

    return {"html": html, "subject": subject}
