import os
import sys
import time
import logging
import hashlib
import re
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple, Set
from contextlib import contextmanager
from urllib.parse import urlparse, parse_qs, unquote

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, Request, HTTPException, Query, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

import feedparser
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from bs4 import BeautifulSoup

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
LOG = logging.getLogger("quantbrief")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
if not LOG.handlers:
    LOG.addHandler(handler)

# ------------------------------------------------------------------------------
# Config / Environment
# ------------------------------------------------------------------------------
APP = FastAPI(title="Quantbrief Stock News Aggregator")
app = APP  # for uvicorn

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    LOG.warning("DATABASE_URL not set - database features disabled")

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme-admin-token")

# OpenAI Configuration - FIXED: Remove temperature parameter for gpt-4o-mini
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Email configuration
def _first(*vals) -> Optional[str]:
    for v in vals:
        if v:
            return v
    return None

SMTP_HOST = _first(os.getenv("MAILGUN_SMTP_SERVER"), os.getenv("SMTP_HOST"))
SMTP_PORT = int(_first(os.getenv("MAILGUN_SMTP_PORT"), os.getenv("SMTP_PORT"), "587"))
SMTP_USERNAME = _first(os.getenv("MAILGUN_SMTP_LOGIN"), os.getenv("SMTP_USERNAME"))
SMTP_PASSWORD = _first(os.getenv("MAILGUN_SMTP_PASSWORD"), os.getenv("SMTP_PASSWORD"))
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "1") not in ("0", "false", "False", "")

EMAIL_FROM = _first(os.getenv("MAILGUN_FROM"), os.getenv("EMAIL_FROM"), SMTP_USERNAME)
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
DIGEST_TO = _first(os.getenv("DIGEST_TO"), ADMIN_EMAIL)

DEFAULT_RETAIN_DAYS = int(os.getenv("DEFAULT_RETAIN_DAYS", "90"))

# FIXED: Enhanced spam filtering with more comprehensive domain list
SPAM_DOMAINS = {
    "marketbeat.com", "www.marketbeat.com", "marketbeat",
    "newser.com", "www.newser.com", "newser", 
    "khodrobank.com", "www.khodrobank.com"
}

QUALITY_DOMAINS = {
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "barrons.com", "cnbc.com", "marketwatch.com",
    "yahoo.com/finance", "finance.yahoo.com",
    "businesswire.com", "prnewswire.com", "globenewswire.com",
    "tipranks.com", "www.tipranks.com", "tipranks",
    "simplywall.st", "www.simplywall.st", "simplywall",
    "dailyitem.com", "www.dailyitem.com",
    "marketscreener.com", "www.marketscreener.com", "marketscreener",
    "insidermoneky.com", "seekingalpha.com/pro", "fool.com"
}

# ------------------------------------------------------------------------------
# OpenAI Integration for Dynamic Keyword Generation
# ------------------------------------------------------------------------------
def generate_ticker_metadata_with_ai(ticker: str) -> Dict[str, List[str]]:
    """Use OpenAI to generate industry keywords and competitors for a ticker - FIXED JSON parsing"""
    if not OPENAI_API_KEY:
        LOG.error("OpenAI API key not configured")
        # Return default fallback for TLN
        if ticker == "TLN":
            return {
                "industry_keywords": ["renewable energy", "data center power", "nuclear energy", "grid stability", "energy storage"],
                "competitors": ["Vistra Corp", "VST", "NRG Energy", "NRG", "Constellation Energy"]
            }
        return {"industry_keywords": [], "competitors": []}
    
    prompt = f"""
    For the stock ticker {ticker}, provide:
    1. Five industry keywords or trends that are most relevant to this company's business and sector
    2. Five main publicly-traded competitors (include their tickers if they are well-known)
    
    Format your response as a JSON object with this exact structure:
    {{
        "industry_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
        "competitors": ["Competitor1 Name", "TICK1", "Competitor2 Name", "TICK2", "Competitor3"]
    }}
    
    Focus on:
    - Industry keywords should be specific terms that would appear in relevant news
    - Include both company names and tickers for competitors when applicable
    - Ensure all keywords are relevant for news searching
    - Be specific rather than generic (e.g. "renewable energy storage" instead of just "technology")
    
    Return ONLY the JSON object, no additional text.
    """
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Add response_format to ensure JSON output
        data = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a financial analyst expert who provides accurate information about companies and their industries. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 500,
            "response_format": {"type": "json_object"}  # ADDED: Force JSON response
        }
        
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=30)
        
        # Enhanced error handling with detailed logging
        LOG.info(f"OpenAI API response status: {response.status_code}")
        
        if response.status_code != 200:
            LOG.error(f"OpenAI API {response.status_code} error: {response.text}")
            return {"industry_keywords": [], "competitors": []}
            
        # Check if response is empty
        if not response.text.strip():
            LOG.error(f"OpenAI API returned empty response for ticker {ticker}")
            return {"industry_keywords": [], "competitors": []}
        
        # Log the raw response for debugging
        LOG.info(f"OpenAI raw response length: {len(response.text)}")
        
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            LOG.error(f"OpenAI API returned invalid JSON for ticker {ticker}: {e}")
            LOG.error(f"Raw response: {response.text[:500]}")
            return {"industry_keywords": [], "competitors": []}
        
        if 'choices' not in result or not result['choices']:
            LOG.error(f"OpenAI API response missing choices for ticker {ticker}: {result}")
            return {"industry_keywords": [], "competitors": []}
            
        content = result['choices'][0]['message']['content']
        LOG.info(f"OpenAI content preview: {content[:200]}")  # ADDED: Log content for debugging
        
        # FIXED: Parse the content directly as JSON (no quotes needed with json_object format)
        try:
            metadata = json.loads(content)
        except json.JSONDecodeError as e:
            LOG.error(f"Failed to parse OpenAI content as JSON for ticker {ticker}: {e}")
            LOG.error(f"Content: {content[:500]}")
            return {"industry_keywords": [], "competitors": []}
        
        LOG.info(f"Successfully generated AI metadata for {ticker}")
        return {
            "industry_keywords": metadata.get("industry_keywords", [])[:5],
            "competitors": metadata.get("competitors", [])[:5]
        }
        
    except Exception as e:
        LOG.error(f"OpenAI API error for ticker {ticker}: {e}")
        # Return fallback for TLN
        if ticker == "TLN":
            return {
                "industry_keywords": ["renewable energy", "data center power", "nuclear energy", "grid stability", "energy storage"],
                "competitors": ["Vistra Corp", "VST", "NRG Energy", "NRG", "Constellation Energy"]
            }
        return {"industry_keywords": [], "competitors": []}

# ------------------------------------------------------------------------------
# Database helpers
# ------------------------------------------------------------------------------
@contextmanager
def db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def ensure_schema():
    """Initialize database schema with enhanced tables"""
    with db() as conn:
        with conn.cursor() as cur:
            # Create ticker_config table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ticker_config (
                    ticker VARCHAR(10) PRIMARY KEY,
                    name VARCHAR(255),
                    industry_keywords TEXT[],
                    competitors TEXT[],
                    active BOOLEAN DEFAULT TRUE,
                    ai_generated BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS domain_names (
                    domain VARCHAR(255) PRIMARY KEY,
                    formal_name VARCHAR(255) NOT NULL,
                    ai_generated BOOLEAN DEFAULT FALSE,
                    verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Add category column to found_url if not exists
            cur.execute("""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name='found_url' AND column_name='category') 
                    THEN 
                        ALTER TABLE found_url ADD COLUMN category VARCHAR(50) DEFAULT 'company';
                    END IF;
                END $$;
            """)
            
            # Add related_ticker column to found_url if not exists
            cur.execute("""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name='found_url' AND column_name='related_ticker') 
                    THEN 
                        ALTER TABLE found_url ADD COLUMN related_ticker VARCHAR(10);
                    END IF;
                END $$;
            """)
            
            # Add original_source_url column for Yahoo Finance original sources
            cur.execute("""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name='found_url' AND column_name='original_source_url') 
                    THEN 
                        ALTER TABLE found_url ADD COLUMN original_source_url TEXT;
                    END IF;
                END $$;
            """)
            
            # Ensure indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_found_url_hash ON found_url(url_hash);
                CREATE INDEX IF NOT EXISTS idx_found_url_ticker_quality ON found_url(ticker, quality_score DESC);
                CREATE INDEX IF NOT EXISTS idx_found_url_published ON found_url(published_at DESC);
                CREATE INDEX IF NOT EXISTS idx_found_url_digest ON found_url(sent_in_digest, found_at DESC);
                CREATE INDEX IF NOT EXISTS idx_found_url_feed_foundat ON found_url(feed_id, found_at DESC);
                CREATE INDEX IF NOT EXISTS idx_found_url_category ON found_url(ticker, category);
                CREATE INDEX IF NOT EXISTS idx_ticker_config_active ON ticker_config(active);
                CREATE INDEX IF NOT EXISTS idx_domain_names_domain ON domain_names(domain);
            """)

def upsert_ticker_config(ticker: str, metadata: Dict, ai_generated: bool = False):
    """Insert or update ticker configuration"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ticker_config (ticker, name, industry_keywords, competitors, ai_generated)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE
            SET name = EXCLUDED.name,
                industry_keywords = EXCLUDED.industry_keywords,
                competitors = EXCLUDED.competitors,
                ai_generated = EXCLUDED.ai_generated,
                updated_at = NOW()
        """, (
            ticker,
            metadata.get("name", ticker),
            metadata.get("industry_keywords", []),
            metadata.get("competitors", []),
            ai_generated
        ))

def get_ticker_config(ticker: str) -> Optional[Dict]:
    """Get ticker configuration from database"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT ticker, name, industry_keywords, competitors, ai_generated
            FROM ticker_config
            WHERE ticker = %s AND active = TRUE
        """, (ticker,))
        return cur.fetchone()

def get_or_create_ticker_metadata(ticker: str, force_refresh: bool = False) -> Dict[str, List[str]]:
    """Get ticker metadata from DB or generate with AI if not exists"""
    # Check database first
    if not force_refresh:
        config = get_ticker_config(ticker)
        if config:
            LOG.info(f"Using existing metadata for {ticker} (AI generated: {config.get('ai_generated', False)})")
            return {
                "company": [ticker],
                "industry": config.get("industry_keywords", []),
                "competitors": config.get("competitors", [])
            }
    
    # Generate with AI
    LOG.info(f"Generating metadata for {ticker} using OpenAI...")
    ai_metadata = generate_ticker_metadata_with_ai(ticker)
    
    # Save to database
    metadata = {
        "name": ticker,
        "industry_keywords": ai_metadata.get("industry_keywords", []),
        "competitors": ai_metadata.get("competitors", [])
    }
    upsert_ticker_config(ticker, metadata, ai_generated=True)
    
    return {
        "company": [ticker],
        "industry": metadata["industry_keywords"],
        "competitors": metadata["competitors"]
    }

def build_feed_urls(ticker: str, keywords: Dict[str, List[str]]) -> List[Dict]:
    """Build feed URLs for different categories - FIXED to only use working feeds"""
    feeds = []
    
    # Company-specific feeds (only reliable ones)
    feeds.extend([
        {
            "url": f"https://news.google.com/rss/search?q={ticker}+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
            "name": f"Google News: {ticker} Company",
            "category": "company"
        },
        {
            "url": f"https://finance.yahoo.com/rss/headline?s={ticker}",
            "name": f"Yahoo Finance: {ticker}",
            "category": "company"
        }
    ])
    
    # Industry feeds
    industry_keywords = keywords.get("industry", [])
    LOG.info(f"Building industry feeds for {ticker} with keywords: {industry_keywords}")
    for keyword in industry_keywords[:3]:
        keyword_encoded = requests.utils.quote(keyword)
        feeds.append({
            "url": f"https://news.google.com/rss/search?q=\"{keyword_encoded}\"+when:7d&hl=en-US&gl=US&ceid=US:en",
            "name": f"Industry: {keyword}",
            "category": "industry"
        })
    
    # Competitor feeds
    competitors = keywords.get("competitors", [])
    LOG.info(f"Building competitor feeds for {ticker} with competitors: {competitors}")
    for competitor in competitors[:3]:
        comp_clean = competitor.replace("(", "").replace(")", "").strip()
        comp_encoded = requests.utils.quote(comp_clean)
        feeds.append({
            "url": f"https://news.google.com/rss/search?q=\"{comp_encoded}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
            "name": f"Competitor: {competitor}",
            "category": "competitor"
        })
    
    LOG.info(f"Built {len(feeds)} total feeds for {ticker}: {len([f for f in feeds if f['category'] == 'company'])} company, {len([f for f in feeds if f['category'] == 'industry'])} industry, {len([f for f in feeds if f['category'] == 'competitor'])} competitor")
    return feeds

def upsert_feed(url: str, name: str, ticker: str, category: str = "company", retain_days: int = 90) -> int:
    """Insert or update a feed source with category"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO source_feed (url, name, ticker, retain_days, active)
            VALUES (%s, %s, %s, %s, TRUE)
            ON CONFLICT (url) DO UPDATE
            SET name = EXCLUDED.name,
                ticker = EXCLUDED.ticker,
                retain_days = EXCLUDED.retain_days,
                active = TRUE
            RETURNING id;
        """, (url, name, ticker, retain_days))
        return cur.fetchone()["id"]

def list_active_feeds(tickers: List[str] = None) -> List[Dict]:
    """Get all active feeds, optionally filtered by tickers"""
    with db() as conn, conn.cursor() as cur:
        if tickers:
            cur.execute("""
                SELECT id, url, name, ticker, retain_days
                FROM source_feed
                WHERE active = TRUE AND ticker = ANY(%s)
                ORDER BY ticker, id
            """, (tickers,))
        else:
            cur.execute("""
                SELECT id, url, name, ticker, retain_days
                FROM source_feed
                WHERE active = TRUE
                ORDER BY ticker, id
            """)
        return list(cur.fetchall())

# ------------------------------------------------------------------------------
# URL Resolution and Quality Scoring
# ------------------------------------------------------------------------------
def extract_yahoo_finance_source(url: str) -> Optional[str]:
    """Extract original source URL from Yahoo Finance article - FIXED for real JSON patterns"""
    try:
        # Only process Yahoo Finance URLs
        if "finance.yahoo.com" not in url:
            return None
            
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        if response.status_code != 200:
            LOG.warning(f"HTTP {response.status_code} when fetching Yahoo URL: {url}")
            return None
        
        html_content = response.text
        LOG.debug(f"Yahoo page content length: {len(html_content)}")
        
        # Try multiple field names Yahoo might use
        field_patterns = [
            'providerContentUrl',
            'sourceUrl', 
            'originalUrl',
            'contentUrl',
            'canonicalUrl'
        ]
        
        for field_name in field_patterns:
            # Multiple regex patterns to handle different JSON formatting
            patterns = [
                # Pattern 1: Handle heavily escaped JSON (the actual format)
                rf'\\\\\\"{field_name}\\\\\\":\\\\\\\"([^\\\\]*?)\\\\\\\"',
                # Pattern 2: Handle escaped JSON  
                rf'\\"{field_name}\\":\\\"([^\\"]*)\\\"',
                # Pattern 3: Standard JSON
                rf'"{field_name}"\s*:\s*"([^"]*)"',
                # Pattern 4: Specifically target the stockstory URL we see in debug
                rf'{field_name}[^:]*:[^"]*"([^"]*stockstory\.org[^"]*)"'
            ]
            
            for i, pattern in enumerate(patterns):
                matches = re.finditer(pattern, html_content)
                
                for match in matches:
                    raw_url = match.group(1)
                    LOG.debug(f"Found {field_name} candidate with pattern {i+1}: {raw_url[:100]}")
                    
                    # Try to clean up the URL
                    cleaned_urls = []
                    
                    # Method 1: JSON unescape
                    try:
                        unescaped_url = json.loads(f'"{raw_url}"')
                        cleaned_urls.append(unescaped_url)
                    except json.JSONDecodeError:
                        pass
                    
                    # Method 2: Simple unescape
                    simple_unescaped = raw_url.replace('\\/', '/').replace('\\"', '"')
                    cleaned_urls.append(simple_unescaped)
                    
                    # Method 3: Use raw URL as-is
                    cleaned_urls.append(raw_url)
                    
                    # Test each cleaned URL
                    for cleaned_url in cleaned_urls:
                        try:
                            parsed = urlparse(cleaned_url)
                            if (parsed.scheme in ['http', 'https'] and 
                                parsed.netloc and 
                                len(cleaned_url) > 20 and
                                'finance.yahoo.com' not in cleaned_url):  # Don't return Yahoo URLs
                                
                                LOG.info(f"Found Yahoo Finance original source via {field_name}: {cleaned_url}")
                                return cleaned_url
                        except Exception:
                            continue
        
        # If no structured extraction worked, try a more aggressive search
        # Look for any URL patterns that might be the source
        url_patterns = [
            r'https://stockstory\.org/[^"\s]*',
            r'https://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/[^"\s]*(?:stock|news|article|finance)[^"\s]*',
        ]
        
        for pattern in url_patterns:
            matches = re.finditer(pattern, html_content)
            for match in matches:
                candidate_url = match.group(0).rstrip('",')
                try:
                    parsed = urlparse(candidate_url)
                    if parsed.scheme and parsed.netloc and 'finance.yahoo.com' not in candidate_url:
                        LOG.info(f"Found Yahoo source via URL pattern search: {candidate_url}")
                        return candidate_url
                except Exception:
                    continue
        
        LOG.debug(f"No original source found for Yahoo URL: {url}")
        return None
        
    except Exception as e:
        LOG.warning(f"Failed to extract Yahoo Finance source from {url}: {e}")
        return None

@APP.post("/admin/test-yahoo-extraction-detailed")
def test_yahoo_extraction_detailed(request: Request):
    """Enhanced test for Yahoo Finance source extraction with detailed debugging"""
    require_admin(request)
    
    # Use your specific example URL
    test_url = request.headers.get("test-url", "https://finance.yahoo.com/news/why-bloom-energy-shares-soaring-155542226.html")
    
    LOG.info(f"Testing detailed Yahoo extraction on: {test_url}")
    
    result = {
        "test_url": test_url,
        "extraction_attempted": True
    }
    
    try:
        # Fetch the page content
        response = requests.get(test_url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code != 200:
            result["error"] = f"HTTP {response.status_code}"
            return result
        
        html_content = response.text
        result["page_length"] = len(html_content)
        
        # Check for your specific pattern
        if "providerContentUrl" in html_content:
            result["providerContentUrl_found"] = True
            
            # Find all occurrences
            pattern = r'"providerContentUrl"\s*:\s*"([^"]*)"'
            matches = re.findall(pattern, html_content)
            result["providerContentUrl_matches"] = matches
            
            if matches:
                raw_url = matches[0]
                result["raw_extracted"] = raw_url
                
                # Try different unescaping methods
                try:
                    json_unescaped = json.loads(f'"{raw_url}"')
                    result["json_unescaped"] = json_unescaped
                except:
                    pass
                
                simple_unescaped = raw_url.replace('\\/', '/')
                result["simple_unescaped"] = simple_unescaped
                
                # Check the specific stockstory.org pattern
                if "stockstory.org" in raw_url:
                    result["stockstory_detected"] = True
        else:
            result["providerContentUrl_found"] = False
            
            # Search for any stockstory.org URLs
            stockstory_pattern = r'https://stockstory\.org/[^"\s]*'
            stockstory_matches = re.findall(stockstory_pattern, html_content)
            result["stockstory_urls_found"] = stockstory_matches
        
        # Test the actual function
        extracted_url = extract_yahoo_finance_source(test_url)
        result["function_result"] = extracted_url
        result["extraction_successful"] = extracted_url is not None
        
        if extracted_url:
            result["final_domain"] = urlparse(extracted_url).netloc
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

def resolve_google_news_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve Google News redirect URL - ENHANCED with multiple strategies"""
    original_source_url = None
    
    try:
        # Handle Google News article URLs
        if "news.google.com" in url and ("/articles/" in url or "/rss/" in url):
            LOG.debug(f"Processing Google News URL: {url[:100]}...")
            
            # Strategy 1: Try multiple redirect approaches
            final_url = None
            domain = None
            
            # Try different request methods
            strategies = [
                # Strategy 1: Standard redirect with different headers
                {
                    'headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1'
                    },
                    'timeout': 20
                },
                # Strategy 2: Simpler headers
                {
                    'headers': {
                        'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
                    },
                    'timeout': 15
                },
                # Strategy 3: Minimal approach
                {
                    'headers': {
                        'User-Agent': 'curl/7.68.0'
                    },
                    'timeout': 10
                }
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    LOG.debug(f"Trying Google News redirect strategy {i+1}")
                    response = requests.get(url, allow_redirects=True, **strategy)
                    
                    if response.url != url and "news.google.com" not in response.url:
                        final_url = response.url
                        domain = urlparse(final_url).netloc.lower()
                        LOG.info(f"Google News redirect SUCCESS with strategy {i+1}: {url[:80]}... → {final_url}")
                        break
                    elif response.status_code == 200:
                        # Check if the response contains a meta redirect or JavaScript redirect
                        content = response.text
                        
                        # Look for meta refresh
                        meta_redirect = re.search(r'<meta[^>]*http-equiv=["\']refresh["\'][^>]*content=["\'][^"\']*url=([^"\']*)["\']', content, re.IGNORECASE)
                        if meta_redirect:
                            redirect_url = meta_redirect.group(1)
                            if redirect_url and "news.google.com" not in redirect_url:
                                final_url = redirect_url
                                domain = urlparse(final_url).netloc.lower()
                                LOG.info(f"Google News meta redirect found: {redirect_url}")
                                break
                        
                        # Look for JavaScript redirect
                        js_redirect = re.search(r'window\.location\.href\s*=\s*["\']([^"\']*)["\']', content)
                        if js_redirect:
                            redirect_url = js_redirect.group(1)
                            if redirect_url and "news.google.com" not in redirect_url:
                                final_url = redirect_url
                                domain = urlparse(final_url).netloc.lower()
                                LOG.info(f"Google News JS redirect found: {redirect_url}")
                                break
                                
                except Exception as e:
                    LOG.debug(f"Strategy {i+1} failed: {e}")
                    continue
            
            # If no redirect worked, fall back to manual URL decoding
            if not final_url:
                LOG.warning("All Google News redirect strategies failed, trying manual decode")
                try:
                    # Try to extract from the URL structure
                    if "/articles/" in url:
                        # Extract the base64-like encoded part
                        match = re.search(r'/articles/([^?]+)', url)
                        if match:
                            encoded_part = match.group(1)
                            # This is complex - Google uses proprietary encoding
                            # For now, we'll have to accept that some Google News URLs won't redirect
                            LOG.debug(f"Could not decode Google News URL: {encoded_part[:50]}...")
                except Exception as e:
                    LOG.debug(f"Manual decode failed: {e}")
            
            # If we still don't have a final URL, return the original with Google domain
            if not final_url:
                LOG.warning(f"Google News redirect completely failed for: {url[:100]}...")
                return url, "news.google.com", None
            
            # Check if final destination is spam
            for spam_domain in SPAM_DOMAINS:
                spam_clean = spam_domain.replace("www.", "").lower()
                if spam_clean in domain:
                    LOG.info(f"BLOCKED spam destination after Google redirect: {domain} (matched: {spam_clean})")
                    return None, None, None
            
            # Check if it's Yahoo Finance and extract original source
            if "finance.yahoo.com" in final_url:
                original_source_url = extract_yahoo_finance_source(final_url)
                if original_source_url:
                    original_domain = urlparse(original_source_url).netloc.lower()
                    for spam_domain in SPAM_DOMAINS:
                        spam_clean = spam_domain.replace("www.", "").lower()
                        if spam_clean in original_domain:
                            LOG.info(f"BLOCKED original source as spam: {original_domain}")
                            return None, None, None
                    
                    LOG.info(f"Using original source instead of Yahoo: {original_source_url}")
                    return original_source_url, original_domain, final_url
            
            return final_url, domain, original_source_url
        
        # Handle other URL types (unchanged from original)
        if "google.com/url" in url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            if 'url' in params:
                actual_url = params['url'][0]
                domain = urlparse(actual_url).netloc.lower()
                
                for spam_domain in SPAM_DOMAINS:
                    spam_clean = spam_domain.replace("www.", "").lower()
                    if spam_clean in domain:
                        LOG.info(f"BLOCKED spam destination in redirect: {domain}")
                        return None, None, None
                
                if "finance.yahoo.com" in actual_url:
                    original_source_url = extract_yahoo_finance_source(actual_url)
                    if original_source_url:
                        original_domain = urlparse(original_source_url).netloc.lower()
                        for spam_domain in SPAM_DOMAINS:
                            spam_clean = spam_domain.replace("www.", "").lower()
                            if spam_clean in original_domain:
                                LOG.info(f"BLOCKED original source as spam: {original_domain}")
                                return None, None, None
                        return original_source_url, original_domain, actual_url
                        
                return actual_url, domain, original_source_url
            elif 'q' in params:
                actual_url = params['q'][0]
                domain = urlparse(actual_url).netloc.lower()
                
                for spam_domain in SPAM_DOMAINS:
                    spam_clean = spam_domain.replace("www.", "").lower()
                    if spam_clean in domain:
                        LOG.info(f"BLOCKED spam destination in q param: {domain}")
                        return None, None, None
                
                if "finance.yahoo.com" in actual_url:
                    original_source_url = extract_yahoo_finance_source(actual_url)
                    if original_source_url:
                        original_domain = urlparse(original_source_url).netloc.lower()
                        for spam_domain in SPAM_DOMAINS:
                            spam_clean = spam_domain.replace("www.", "").lower()
                            if spam_clean in original_domain:
                                return None, None, None
                        return original_source_url, original_domain, actual_url
                        
                return actual_url, domain, original_source_url
        
        # Direct URL handling
        domain = urlparse(url).netloc.lower()
        
        for spam_domain in SPAM_DOMAINS:
            spam_clean = spam_domain.replace("www.", "").lower()
            if spam_clean in domain:
                LOG.info(f"BLOCKED spam direct URL: {domain}")
                return None, None, None
        
        if "finance.yahoo.com" in url:
            original_source_url = extract_yahoo_finance_source(url)
            if original_source_url:
                original_domain = urlparse(original_source_url).netloc.lower()
                for spam_domain in SPAM_DOMAINS:
                    spam_clean = spam_domain.replace("www.", "").lower()
                    if spam_clean in original_domain:
                        return None, None, None
                return original_source_url, original_domain, url
                
        return url, domain, original_source_url
        
    except Exception as e:
        LOG.warning(f"Failed to resolve URL {url}: {e}")
        return url, urlparse(url).netloc.lower() if url else None, None

def extract_source_from_title(title: str) -> Tuple[str, str]:
    """Extract source information from Google News article titles"""
    if not title:
        return title, None
    
    original_title = title
    extracted_source = None
    
    # Pattern 1: " - Source Name" at the end
    title_source_patterns = [
        r'\s*-\s*([^-]+\.(com|org|net|co\.uk|in))$',  # Domain patterns
        r'\s*-\s*(Markets?\s+Mojo|StockTradersDaily|Market\s+Watch|Business\s+Wire|PR\s+Newswire|Globe\s+Newswire)$',  # Known sources
        r'\s*-\s*([A-Za-z\s&]+(?:News|Times|Post|Journal|Tribune|Herald|Gazette|Wire|Report|Today|Daily|Weekly))$',  # News outlet patterns
        r'\s*-\s*([A-Za-z\s&]+(?:Financial|Finance|Business|Economic|Market|Investment))$',  # Financial publication patterns
        r'\s*-\s*(Yahoo\s+Finance|Google\s+News|Reuters|Bloomberg|CNBC|MarketWatch|Seeking\s+Alpha|Motley\s+Fool)$'  # Major sources
    ]
    
    for pattern in title_source_patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            extracted_source = match.group(1).strip()
            # Remove the source from title
            title = re.sub(pattern, '', title, flags=re.IGNORECASE).strip()
            break
    
    # Pattern 2: Domain at the end without " - "
    if not extracted_source:
        domain_pattern = r'\s*([a-zA-Z0-9.-]+\.(com|org|net|co\.uk|in))$'
        match = re.search(domain_pattern, title)
        if match:
            extracted_source = match.group(1).strip()
            title = re.sub(domain_pattern, '', title).strip()
    
    return title, extracted_source

def enhance_source_with_ai(raw_source: str) -> str:
    """Use OpenAI to enhance raw source names extracted from titles"""
    if not raw_source or not OPENAI_API_KEY:
        return raw_source or "Unknown"
    
    # Check if it looks like a domain
    if '.' in raw_source and any(tld in raw_source.lower() for tld in ['.com', '.org', '.net', '.co.uk']):
        # It's a domain, use existing domain resolution
        return get_or_create_formal_domain_name(raw_source)
    
    # It's a publication name, enhance it with AI
    prompt = f"""The following text was extracted from a news article title as the source publication: "{raw_source}"

Please provide the proper, formal name of this publication as it would appear in citations.

Examples:
- "Markets Mojo" → "Markets Mojo"
- "StockTradersDaily" → "Stock Traders Daily"  
- "Business Wire" → "Business Wire"
- "Market Watch" → "MarketWatch"

For "{raw_source}", the formal name is:"""
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a media expert. Provide only the proper, formal name of publications. Be concise and accurate."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 30
        }
        
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and result['choices']:
                formal_name = result['choices'][0]['message']['content'].strip()
                formal_name = re.sub(r'^["\']|["\']$', '', formal_name)  # Remove quotes
                formal_name = formal_name.strip()
                
                if len(formal_name) > 2 and len(formal_name) < 100:
                    LOG.info(f"AI enhanced source: '{raw_source}' → '{formal_name}'")
                    return formal_name
                    
    except Exception as e:
        LOG.debug(f"AI source enhancement failed for '{raw_source}': {e}")
    
    # Fallback: Clean up the raw source
    cleaned = raw_source.replace('.com', '').replace('.org', '').replace('.net', '')
    cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)  # Add spaces between camelCase
    return cleaned.title()

def calculate_quality_score(
    title: str, 
    domain: str, 
    ticker: str,
    description: str = "",
    category: str = "company",
    keywords: List[str] = None
) -> float:
    """Calculate article quality score - ENHANCED spam filtering"""
    score = 50.0
    
    content_to_check = f"{title} {domain} {description}".lower()
    domain_clean = domain.lower().replace("www.", "") if domain else ""
    
    # ENHANCED: Check for spam domains more thoroughly
    for spam_domain in SPAM_DOMAINS:
        spam_clean = spam_domain.replace("www.", "").lower()
        if spam_clean in domain_clean or spam_clean in content_to_check:
            LOG.info(f"BLOCKED spam in quality check: {domain} (matched: {spam_clean})")
            return 0.0
    
    # Boost quality domains
    quality_domains = ["reuters", "bloomberg", "wsj", "ft", "cnbc", "finance.yahoo", "businesswire"]
    if any(q in domain_clean for q in quality_domains):
        score += 25
    
    # Category scoring
    if category == "company":
        if ticker in (title or "").upper():
            score += 15
    elif category == "industry":
        score += 8  # Give industry news a better chance
    elif category == "competitor":
        score += 5
    
    # Basic quality checks
    if len(title or "") > 20:
        score += 5
    if len(description or "") > 50:
        score += 5
        
    # Don't be too harsh - lower the minimum threshold
    return max(0, min(100, score))

def get_url_hash(url: str) -> str:
    """Generate hash for URL deduplication"""
    url_lower = url.lower()
    url_clean = re.sub(r'[?&](utm_|ref=|source=).*', '', url_lower)
    return hashlib.md5(url_clean.encode()).hexdigest()

def get_formal_domain_name_from_ai(domain: str) -> Optional[str]:
    """Use OpenAI to get the formal name of a domain"""
    if not OPENAI_API_KEY:
        LOG.warning("OpenAI API key not configured for domain name resolution")
        return None
    
    clean_domain = domain.replace("www.", "").lower()
    
    prompt = f"""What is the formal, proper name of the website/company "{clean_domain}"?

Please provide ONLY the official name as it would appear in a citation or news article.
Examples:
- "investors.com" → "Investor's Business Daily"
- "barrons.com" → "Barron's" 
- "zacks.com" → "Zacks Investment Research"

For "{clean_domain}", the formal name is:"""
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a media expert. Provide only the official, formal name of websites/publications. Be concise."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 50
        }
        
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=15)
        
        if response.status_code != 200:
            return None
        
        result = response.json()
        if 'choices' not in result or not result['choices']:
            return None
            
        formal_name = result['choices'][0]['message']['content'].strip()
        formal_name = re.sub(r'^["\']|["\']$', '', formal_name)
        formal_name = formal_name.strip()
        
        if len(formal_name) > 100 or len(formal_name) < 2:
            return None
            
        LOG.info(f"AI resolved domain {domain} to: {formal_name}")
        return formal_name
        
    except Exception as e:
        LOG.error(f"Error getting formal name for domain {domain}: {e}")
        return None

def get_or_create_formal_domain_name(domain: str) -> str:
    """Get formal domain name from cache or generate with AI"""
    if not domain:
        return "Unknown"
    
    clean_domain = domain.replace("www.", "").lower()
    
    # Check database cache first
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT formal_name FROM domain_names WHERE domain = %s", (clean_domain,))
        result = cur.fetchone()
        if result:
            return result["formal_name"]
    
    # Generate with AI
    formal_name = get_formal_domain_name_from_ai(clean_domain)
    
    # Fallback if AI fails
    if not formal_name:
        formal_name = clean_domain.replace(".com", "").replace(".org", "").replace(".net", "").title()
    
    # Store in database
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO domain_names (domain, formal_name, ai_generated)
            VALUES (%s, %s, %s)
            ON CONFLICT (domain) DO UPDATE
            SET formal_name = EXCLUDED.formal_name, updated_at = NOW()
        """, (clean_domain, formal_name, formal_name != clean_domain.title()))
    
    return formal_name

# ------------------------------------------------------------------------------
# Feed Processing
# ------------------------------------------------------------------------------
def parse_datetime(candidate) -> Optional[datetime]:
    """Parse various datetime formats"""
    if not candidate:
        return None
    if isinstance(candidate, datetime):
        return candidate if candidate.tzinfo else candidate.replace(tzinfo=timezone.utc)
    
    if hasattr(candidate, "tm_year"):
        try:
            return datetime.fromtimestamp(time.mktime(candidate), tz=timezone.utc)
        except:
            pass
    
    try:
        return datetime.fromisoformat(str(candidate))
    except:
        return None

def ingest_feed(feed: Dict, category: str = "company", keywords: List[str] = None) -> Dict[str, int]:
    """Process a single feed and store articles with category - FIXED variable scope issues"""
    stats = {"processed": 0, "inserted": 0, "duplicates": 0, "low_quality": 0, "blocked_spam": 0, "yahoo_sources_found": 0}
    
    try:
        LOG.info(f"Processing feed [{category}]: {feed['name']}")
        parsed = feedparser.parse(feed["url"])
        LOG.info(f"Feed parsing result: {len(parsed.entries)} entries found")
        
        for entry in parsed.entries:
            stats["processed"] += 1
            
            original_url = getattr(entry, "link", None)
            if not original_url:
                continue
            
            # FIXED: URL resolution with proper variable names
            resolved_result = resolve_google_news_url(original_url)
            if resolved_result[0] is None:  # Spam detected
                stats["blocked_spam"] += 1
                continue
                
            resolved_url, domain, yahoo_source_url = resolved_result  # FIXED: Proper unpacking
            if not resolved_url or not domain:
                continue
            
            # Track Yahoo source extractions
            if yahoo_source_url:
                stats["yahoo_sources_found"] += 1
                LOG.info(f"Yahoo Finance article resolved to original: {resolved_url}")
                
            url_hash = get_url_hash(resolved_url)
            
            title = getattr(entry, "title", "") or "No Title"
            description = getattr(entry, "summary", "")[:500] if hasattr(entry, "summary") else ""
            
            # Calculate quality score with category context
            quality_score = calculate_quality_score(
                title, domain, feed["ticker"], description, category, keywords
            )
            
            if quality_score < 15:  # Lowered threshold
                stats["low_quality"] += 1
                LOG.debug(f"Skipping low quality [{category}]: {title[:50]} (score: {quality_score})")
                continue
            
            published_at = None
            if hasattr(entry, "published_parsed"):
                published_at = parse_datetime(entry.published_parsed)
            elif hasattr(entry, "updated_parsed"):
                published_at = parse_datetime(entry.updated_parsed)
            
            try:
                with db() as conn, conn.cursor() as cur:
                    cur.execute("SELECT id FROM found_url WHERE url_hash = %s", (url_hash,))
                    if cur.fetchone():
                        stats["duplicates"] += 1
                        continue
                    
                    # Determine related ticker for competitor articles
                    related_ticker = None
                    if category == "competitor" and "Competitor:" in feed["name"]:
                        comp_name = feed["name"].replace("Competitor:", "").strip()
                        # Extract ticker if it looks like one (2-5 uppercase letters)
                        words = comp_name.split()
                        for word in words:
                            if re.match(r'^[A-Z]{2,5}$', word):
                                related_ticker = word
                                break
                    
                    # FIXED: Use proper variable name yahoo_source_url instead of original_source_url
                    cur.execute("""
                        INSERT INTO found_url (
                            url, resolved_url, url_hash, title, description,
                            feed_id, ticker, domain, quality_score, published_at,
                            category, related_ticker, original_source_url
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        original_url, resolved_url, url_hash, title, description,
                        feed["id"], feed["ticker"], domain, quality_score, published_at,
                        category, related_ticker, yahoo_source_url  # FIXED: Use yahoo_source_url
                    ))
                    
                    if cur.fetchone():
                        stats["inserted"] += 1
                        source_note = f" (source: {urlparse(yahoo_source_url).netloc})" if yahoo_source_url else ""
                        LOG.info(f"Inserted [{category}] (score: {quality_score:.0f}){source_note}: {title[:60]}...")
                        
            except Exception as e:
                LOG.error(f"Database insert error for '{title[:50]}': {e}")
                continue
                
    except Exception as e:
        LOG.error(f"Feed processing error for {feed['name']}: {e}")
    
    LOG.info(f"Feed {feed['name']} [{category}] stats: {stats}")
    return stats

# ------------------------------------------------------------------------------
# Email Digest
# ------------------------------------------------------------------------------
def build_digest_html(articles_by_ticker: Dict[str, Dict[str, List[Dict]]], period_days: int) -> str:
    """Build HTML email digest with enhanced styling"""
    # Get ticker metadata for display
    ticker_metadata = {}
    for ticker in articles_by_ticker.keys():
        config = get_ticker_config(ticker)
        if config:
            ticker_metadata[ticker] = {
                "industry_keywords": config.get("industry_keywords", []),
                "competitors": config.get("competitors", [])
            }
    
    html = [
        "<html><head><style>",
        "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 13px; line-height: 1.6; color: #333; }",
        "h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }",
        "h2 { color: #34495e; margin-top: 25px; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; }",
        "h3 { color: #7f8c8d; margin-top: 15px; margin-bottom: 8px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }",
        ".article { margin: 5px 0; padding: 5px; border-left: 3px solid transparent; transition: all 0.3s; }",
        ".article:hover { background-color: #f8f9fa; border-left-color: #3498db; }",
        ".company { border-left-color: #27ae60; }",
        ".industry { border-left-color: #f39c12; }",
        ".competitor { border-left-color: #e74c3c; }",
        ".meta { color: #95a5a6; font-size: 11px; }",
        ".score { display: inline-block; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 10px; margin-left: 5px; }",
        ".high-score { background-color: #d4edda; color: #155724; }",
        ".med-score { background-color: #fff3cd; color: #856404; }",
        ".low-score { background-color: #f8d7da; color: #721c24; }",
        ".source-badge { display: inline-block; padding: 2px 6px; margin-left: 8px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e9ecef; color: #495057; }",
        ".keywords { background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 11px; }",
        "a { color: #2980b9; text-decoration: none; }",
        "a:hover { text-decoration: underline; }",
        ".summary { margin-top: 20px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }",
        ".ticker-section { margin-bottom: 40px; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "</style></head><body>",
        f"<h1>Stock Intelligence Report</h1>",
        f"<div class='summary'>",
        f"<strong>Report Period:</strong> Last {period_days} days<br>",
        f"<strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC<br>",
        f"<strong>Tickers Covered:</strong> {', '.join(articles_by_ticker.keys())}",
        "</div>"
    ]
    
    for ticker, categories in articles_by_ticker.items():
        total_articles = sum(len(articles) for articles in categories.values())
        
        html.append(f"<div class='ticker-section'>")
        html.append(f"<h2>{ticker} - {total_articles} Total Articles</h2>")
        
        # Add keyword information
        if ticker in ticker_metadata:
            metadata = ticker_metadata[ticker]
            html.append("<div class='keywords'>")
            html.append(f"<strong>AI-Powered Monitoring Keywords:</strong><br>")
            if metadata.get("industry_keywords"):
                html.append(f"<strong>Industry:</strong> {', '.join(metadata['industry_keywords'])}<br>")
            if metadata.get("competitors"):
                html.append(f"<strong>Competitors:</strong> {', '.join(metadata['competitors'])}")
            html.append("</div>")
        
        # Company News Section
        if "company" in categories and categories["company"]:
            html.append(f"<h3>Company News ({len(categories['company'])} articles)</h3>")
            for article in categories["company"][:30]:
                html.append(_format_article_html(article, "company"))
        
        # Industry News Section
        if "industry" in categories and categories["industry"]:
            html.append(f"<h3>Industry & Market News ({len(categories['industry'])} articles)</h3>")
            for article in categories["industry"][:20]:
                html.append(_format_article_html(article, "industry"))
        
        # Competitor News Section
        if "competitor" in categories and categories["competitor"]:
            html.append(f"<h3>Competitor Intelligence ({len(categories['competitor'])} articles)</h3>")
            for article in categories["competitor"][:20]:
                html.append(_format_article_html(article, "competitor"))
        
        html.append("</div>")
    
    html.append("""
        <div style='margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; font-size: 11px; color: #6c757d;'>
            <strong>About This Report:</strong><br>
            • Company News: Direct mentions and updates about the ticker<br>
            • Industry News: Sector trends and market dynamics<br>
            • Competitor Intelligence: Updates on key competitors<br>
            • Quality scores: Higher scores indicate more reputable sources and relevant content (15-100 scale)<br>
            • Keywords generated by AI analysis for comprehensive coverage<br>
            • Spam domains automatically filtered out for quality<br>
            • Source badges show the original publisher of each article
        </div>
        </body></html>
    """)
    
    return "".join(html)

def _format_article_html(article: Dict, category: str) -> str:
    """Format a single article for HTML display with Google News title processing"""
    pub_date = article["published_at"].strftime("%m/%d %H:%M") if article["published_at"] else "N/A"
    
    original_title = article["title"] or "No Title"
    resolved_domain = article["domain"] or "unknown"
    
    # Check if this is a Google News article that needs title processing
    is_google_news = "news.google.com" in resolved_domain
    display_source = None
    title = original_title
    
    if is_google_news:
        # Extract source from title for Google News articles
        title, extracted_source = extract_source_from_title(original_title)
        
        if extracted_source:
            # Use AI to enhance the extracted source
            display_source = enhance_source_with_ai(extracted_source)
        else:
            # Fallback to domain resolution
            display_source = get_or_create_formal_domain_name(resolved_domain)
    else:
        # Use normal domain resolution for non-Google sources
        display_source = get_or_create_formal_domain_name(resolved_domain)
    
    # Clean up remaining title suffixes
    suffixes_to_remove = [
        " - MarketBeat", " - Newser", " - TipRanks", " - MSN", 
        " - The Daily Item", " - MarketScreener", " - Seeking Alpha",
        " - simplywall.st", " - Investopedia", " - Google News", " - Yahoo Finance"
    ]
    
    for suffix in suffixes_to_remove:
        if title.endswith(suffix):
            title = title[:-len(suffix)].strip()
    
    # Remove ticker symbols and extra spaces
    title = re.sub(r'\s*\$[A-Z]+\s*-?\s*', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    
    # Determine which URL to use for the main link
    link_url = article["resolved_url"] or article.get("original_source_url") or article["url"]
    
    score = article["quality_score"]
    score_class = "high-score" if score >= 70 else "med-score" if score >= 40 else "low-score"
    
    related = f" | Related: {article.get('related_ticker', '')}" if article.get('related_ticker') else ""
    
    # Format: Source | Score | Title | Date
    return f"""
    <div class='article {category}'>
        <span class='source-badge'>{display_source}</span>
        <span class='score {score_class}'>Score: {score:.0f}</span>
        <a href='{link_url}' target='_blank'>{title}</a>
        <span class='meta'> | {pub_date}</span>
        {related}
    </div>
    """

def fetch_digest_articles(hours: int = 24, tickers: List[str] = None) -> Dict[str, Dict[str, List[Dict]]]:
    """Fetch categorized articles for digest with original source URLs"""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    with db() as conn, conn.cursor() as cur:
        # Build query based on tickers
        if tickers:
            cur.execute("""
                SELECT 
                    f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.quality_score, f.published_at,
                    f.found_at, f.category, f.related_ticker, f.original_source_url
                FROM found_url f
                WHERE f.found_at >= %s
                    AND f.quality_score >= 15
                    AND NOT f.sent_in_digest
                    AND f.ticker = ANY(%s)
                ORDER BY f.ticker, f.category, f.quality_score DESC, f.published_at DESC
            """, (cutoff, tickers))
        else:
            cur.execute("""
                SELECT 
                    f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.quality_score, f.published_at,
                    f.found_at, f.category, f.related_ticker, f.original_source_url
                FROM found_url f
                WHERE f.found_at >= %s
                    AND f.quality_score >= 15
                    AND NOT f.sent_in_digest
                ORDER BY f.ticker, f.category, f.quality_score DESC, f.published_at DESC
            """, (cutoff,))
        
        articles_by_ticker = {}
        for row in cur.fetchall():
            ticker = row["ticker"] or "UNKNOWN"
            category = row["category"] or "company"
            
            if ticker not in articles_by_ticker:
                articles_by_ticker[ticker] = {}
            if category not in articles_by_ticker[ticker]:
                articles_by_ticker[ticker][category] = []
            
            articles_by_ticker[ticker][category].append(dict(row))
        
        # Mark articles as sent
        if tickers:
            cur.execute("""
                UPDATE found_url
                SET sent_in_digest = TRUE
                WHERE found_at >= %s AND quality_score >= 15 AND ticker = ANY(%s)
            """, (cutoff, tickers))
        else:
            cur.execute("""
                UPDATE found_url
                SET sent_in_digest = TRUE
                WHERE found_at >= %s AND quality_score >= 15
            """, (cutoff,))
    
    return articles_by_ticker

def send_email(subject: str, html_body: str, to: str = None):
    """Send email via SMTP"""
    if not all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, EMAIL_FROM]):
        LOG.error("SMTP not fully configured")
        return False
    
    try:
        recipient = to or DIGEST_TO
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = recipient
        
        # Add plain text version
        text_body = "Please view this email in HTML format."
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            if SMTP_STARTTLS:
                server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, [recipient], msg.as_string())
        
        LOG.info(f"Email sent to {recipient}")
        return True
        
    except Exception as e:
        LOG.error(f"Email send failed: {e}")
        return False

# ------------------------------------------------------------------------------
# Auth Middleware
# ------------------------------------------------------------------------------
def require_admin(request: Request):
    """Verify admin token"""
    token = request.headers.get("x-admin-token") or \
            request.headers.get("authorization", "").replace("Bearer ", "")
    
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ------------------------------------------------------------------------------
# Pydantic Models for Request Bodies - FIXED: Add models for endpoints
# ------------------------------------------------------------------------------
class CleanFeedsRequest(BaseModel):
    tickers: Optional[List[str]] = None

class ResetDigestRequest(BaseModel):
    tickers: Optional[List[str]] = None

class ForceDigestRequest(BaseModel):
    tickers: Optional[List[str]] = None

class RegenerateMetadataRequest(BaseModel):
    ticker: str

class InitRequest(BaseModel):
    tickers: List[str]
    force_refresh: bool = False

class CLIRequest(BaseModel):
    action: str
    tickers: List[str]
    minutes: int = 1440

# ------------------------------------------------------------------------------
# API Routes
# ------------------------------------------------------------------------------
@APP.get("/")
def root():
    return {"status": "ok", "service": "Quantbrief Stock News Aggregator"}

@APP.post("/admin/init")
def admin_init(request: Request, body: InitRequest):
    """Initialize database and generate AI-powered feeds for specified tickers"""
    require_admin(request)
    ensure_schema()
    
    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OpenAI API key not configured"}
    
    results = []
    for ticker in body.tickers:
        LOG.info(f"Initializing ticker: {ticker}")
        
        # Get or generate metadata with AI
        keywords = get_or_create_ticker_metadata(ticker, force_refresh=body.force_refresh)
        
        # Build feed URLs for all categories
        feeds = build_feed_urls(ticker, keywords)
        
        for feed_config in feeds:
            feed_id = upsert_feed(
                url=feed_config["url"],
                name=feed_config["name"],
                ticker=ticker,
                category=feed_config.get("category", "company"),
                retain_days=DEFAULT_RETAIN_DAYS
            )
            results.append({
                "ticker": ticker,
                "feed": feed_config["name"],
                "category": feed_config.get("category", "company"),
                "id": feed_id
            })
        
        LOG.info(f"Created {len(feeds)} feeds for {ticker}")
    
    return {
        "status": "initialized",
        "tickers": body.tickers,
        "feeds": results,
        "message": f"Generated {len(results)} feeds using AI-powered keyword analysis"
    }

@APP.post("/cron/ingest")
def cron_ingest(
    request: Request,
    minutes: int = Query(default=1440, description="Time window in minutes"),
    tickers: List[str] = Query(default=None, description="Specific tickers to ingest")
):
    """Ingest articles from feeds for specified tickers"""
    require_admin(request)
    ensure_schema()
    
    # Get feeds for specified tickers
    feeds = list_active_feeds(tickers)
    
    if not feeds:
        return {"status": "no_feeds", "message": "No active feeds found for specified tickers"}
    
    total_stats = {
        "feeds_processed": 0,
        "total_inserted": 0,
        "total_duplicates": 0,
        "total_blocked_spam": 0,
        "total_yahoo_sources": 0,
        "by_ticker": {},
        "by_category": {"company": 0, "industry": 0, "competitor": 0}
    }
    
    # Group feeds by ticker for better processing
    feeds_by_ticker = {}
    for feed in feeds:
        ticker = feed["ticker"]
        if ticker not in feeds_by_ticker:
            feeds_by_ticker[ticker] = []
        feeds_by_ticker[ticker].append(feed)
    
    # Process each ticker's feeds
    for ticker, ticker_feeds in feeds_by_ticker.items():
        # Get metadata for this ticker
        metadata = get_or_create_ticker_metadata(ticker)
        ticker_stats = {"inserted": 0, "duplicates": 0, "blocked_spam": 0, "yahoo_sources": 0}
        
        for feed in ticker_feeds:
            # Determine category from feed name
            category = "company"
            if "Industry:" in feed["name"]:
                category = "industry"
            elif "Competitor:" in feed["name"]:
                category = "competitor"
            
            # Get relevant keywords for this category
            category_keywords = metadata.get(category, [])
            
            stats = ingest_feed(feed, category, category_keywords)
            total_stats["feeds_processed"] += 1
            total_stats["total_inserted"] += stats["inserted"]
            total_stats["total_duplicates"] += stats["duplicates"]
            total_stats["total_blocked_spam"] += stats.get("blocked_spam", 0)
            total_stats["total_yahoo_sources"] += stats.get("yahoo_sources_found", 0)
            total_stats["by_category"][category] += stats["inserted"]
            ticker_stats["inserted"] += stats["inserted"]
            ticker_stats["duplicates"] += stats["duplicates"]
            ticker_stats["blocked_spam"] += stats.get("blocked_spam", 0)
            ticker_stats["yahoo_sources"] += stats.get("yahoo_sources_found", 0)
            
            LOG.info(f"Feed {feed['name']} [{category}]: {stats}")
        
        total_stats["by_ticker"][ticker] = ticker_stats
    
    # Clean old articles
    cutoff = datetime.now(timezone.utc) - timedelta(days=DEFAULT_RETAIN_DAYS)
    with db() as conn, conn.cursor() as cur:
        if tickers:
            cur.execute("DELETE FROM found_url WHERE found_at < %s AND ticker = ANY(%s)", (cutoff, tickers))
        else:
            cur.execute("DELETE FROM found_url WHERE found_at < %s", (cutoff,))
        deleted = cur.rowcount
    
    total_stats["old_articles_deleted"] = deleted
    return total_stats

@APP.post("/cron/digest")
def cron_digest(
    request: Request,
    minutes: int = Query(default=1440, description="Time window in minutes"),
    tickers: List[str] = Query(default=None, description="Specific tickers for digest")
):
    """Generate and send email digest - FIXED to lower quality threshold"""
    require_admin(request)
    ensure_schema()
    
    hours = minutes / 60
    days = int(hours / 24) if hours >= 24 else 0
    period_label = f"{days} days" if days > 0 else f"{hours:.0f} hours"
    
    # Lower the quality threshold to get more articles
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    with db() as conn, conn.cursor() as cur:
        if tickers:
            cur.execute("""
                SELECT 
                    f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.quality_score, f.published_at,
                    f.found_at, f.category, f.related_ticker, f.original_source_url
                FROM found_url f
                WHERE f.found_at >= %s
                    AND f.quality_score >= 15
                    AND NOT f.sent_in_digest
                    AND f.ticker = ANY(%s)
                ORDER BY f.ticker, f.category, f.quality_score DESC, f.published_at DESC
            """, (cutoff, tickers))
        else:
            cur.execute("""
                SELECT 
                    f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.quality_score, f.published_at,
                    f.found_at, f.category, f.related_ticker, f.original_source_url
                FROM found_url f
                WHERE f.found_at >= %s
                    AND f.quality_score >= 15
                    AND NOT f.sent_in_digest
                ORDER BY f.ticker, f.category, f.quality_score DESC, f.published_at DESC
            """, (cutoff,))
        
        articles_by_ticker = {}
        for row in cur.fetchall():
            ticker = row["ticker"] or "UNKNOWN"
            category = row["category"] or "company"
            
            if ticker not in articles_by_ticker:
                articles_by_ticker[ticker] = {}
            if category not in articles_by_ticker[ticker]:
                articles_by_ticker[ticker][category] = []
            
            articles_by_ticker[ticker][category].append(dict(row))
        
        # Mark articles as sent
        if tickers:
            cur.execute("""
                UPDATE found_url
                SET sent_in_digest = TRUE
                WHERE found_at >= %s AND quality_score >= 15 AND ticker = ANY(%s)
            """, (cutoff, tickers))
        else:
            cur.execute("""
                UPDATE found_url
                SET sent_in_digest = TRUE
                WHERE found_at >= %s AND quality_score >= 15
            """, (cutoff,))
    
    total_articles = sum(
        sum(len(arts) for arts in categories.values())
        for categories in articles_by_ticker.values()
    )
    
    if total_articles == 0:
        return {
            "status": "no_articles",
            "message": f"No new quality articles found in the last {period_label}",
            "tickers": tickers or "all"
        }
    
    html = build_digest_html(articles_by_ticker, days if days > 0 else 1)
    
    tickers_str = ', '.join(articles_by_ticker.keys())
    subject = f"Stock Intelligence: {tickers_str} - {total_articles} articles"
    success = send_email(subject, html)
    
    # Count by category
    category_counts = {"company": 0, "industry": 0, "competitor": 0}
    for ticker_cats in articles_by_ticker.values():
        for cat, arts in ticker_cats.items():
            category_counts[cat] = category_counts.get(cat, 0) + len(arts)
    
    return {
        "status": "sent" if success else "failed",
        "articles": total_articles,
        "tickers": list(articles_by_ticker.keys()),
        "by_category": category_counts,
        "recipient": DIGEST_TO
    }

# FIXED: Updated endpoint with proper request body handling
@APP.post("/admin/clean-feeds")
def clean_old_feeds(request: Request, body: CleanFeedsRequest):
    """Clean old Reddit/Twitter feeds from database"""
    require_admin(request)
    
    with db() as conn, conn.cursor() as cur:
        # Delete feeds that contain Reddit, Twitter, SEC, StockTwits
        cleanup_patterns = [
            "Reddit", "Twitter", "SEC EDGAR", "StockTwits", 
            "r/investing", "r/stocks", "r/SecurityAnalysis", 
            "r/ValueInvesting", "r/energy", "@TalenEnergy"
        ]
        
        total_deleted = 0
        if body.tickers:
            for pattern in cleanup_patterns:
                cur.execute("""
                    DELETE FROM source_feed 
                    WHERE name LIKE %s AND ticker = ANY(%s)
                """, (f"%{pattern}%", body.tickers))
                total_deleted += cur.rowcount
        else:
            for pattern in cleanup_patterns:
                cur.execute("""
                    DELETE FROM source_feed 
                    WHERE name LIKE %s
                """, (f"%{pattern}%",))
                total_deleted += cur.rowcount
    
    return {"status": "cleaned", "feeds_deleted": total_deleted, "tickers": body.tickers or "all"}

@APP.get("/admin/ticker-metadata/{ticker}")
def get_ticker_metadata(request: Request, ticker: str):
    """Get AI-generated metadata for a ticker"""
    require_admin(request)
    
    config = get_ticker_config(ticker)
    if config:
        return {
            "ticker": ticker,
            "ai_generated": config.get("ai_generated", False),
            "industry_keywords": config.get("industry_keywords", []),
            "competitors": config.get("competitors", [])
        }
    
    return {"ticker": ticker, "message": "No metadata found. Use /admin/init to generate."}

@APP.post("/admin/regenerate-metadata")
def regenerate_metadata(request: Request, body: RegenerateMetadataRequest):
    """Force regeneration of AI metadata for a ticker"""
    require_admin(request)
    
    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OpenAI API key not configured"}
    
    LOG.info(f"Regenerating metadata for {body.ticker}")
    metadata = get_or_create_ticker_metadata(body.ticker, force_refresh=True)
    
    # Rebuild feeds
    feeds = build_feed_urls(body.ticker, metadata)
    for feed_config in feeds:
        upsert_feed(
            url=feed_config["url"],
            name=feed_config["name"],
            ticker=body.ticker,
            category=feed_config.get("category", "company"),
            retain_days=DEFAULT_RETAIN_DAYS
        )
    
    return {
        "status": "regenerated",
        "ticker": body.ticker,
        "metadata": metadata,
        "feeds_created": len(feeds)
    }

@APP.get("/admin/ticker-configs")
def list_ticker_configs(request: Request):
    """List all ticker configurations"""
    require_admin(request)
    
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT ticker, name, industry_keywords, competitors, ai_generated, 
                   created_at, updated_at
            FROM ticker_config
            WHERE active = TRUE
            ORDER BY ticker
        """)
        configs = list(cur.fetchall())
    
    return {"configs": configs, "total": len(configs)}

@APP.post("/admin/force-digest")
def force_digest(request: Request, body: ForceDigestRequest):
    """Force digest with existing articles (for testing)"""
    require_admin(request)
    
    with db() as conn, conn.cursor() as cur:
        # Build query based on whether tickers are specified
        if body.tickers:
            cur.execute("""
                SELECT 
                    f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.quality_score, f.published_at,
                    f.found_at, f.category, f.related_ticker, f.original_source_url
                FROM found_url f
                WHERE f.found_at >= %s
                    AND f.quality_score >= 15
                    AND f.ticker = ANY(%s)
                ORDER BY f.ticker, f.category, f.quality_score DESC, f.published_at DESC
            """, (datetime.now(timezone.utc) - timedelta(days=7), body.tickers))
        else:
            cur.execute("""
                SELECT 
                    f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.quality_score, f.published_at,
                    f.found_at, f.category, f.related_ticker, f.original_source_url
                FROM found_url f
                WHERE f.found_at >= %s
                    AND f.quality_score >= 15
                ORDER BY f.ticker, f.category, f.quality_score DESC, f.published_at DESC
            """, (datetime.now(timezone.utc) - timedelta(days=7),))
        
        articles_by_ticker = {}
        for row in cur.fetchall():
            ticker = row["ticker"] or "UNKNOWN"
            category = row["category"] or "company"
            
            if ticker not in articles_by_ticker:
                articles_by_ticker[ticker] = {}
            if category not in articles_by_ticker[ticker]:
                articles_by_ticker[ticker][category] = []
            
            articles_by_ticker[ticker][category].append(dict(row))
    
    total_articles = sum(
        sum(len(arts) for arts in categories.values())
        for categories in articles_by_ticker.values()
    )
    
    if total_articles == 0:
        return {"status": "no_articles", "message": "No articles found in database"}
    
    html = build_digest_html(articles_by_ticker, 7)
    tickers_str = ', '.join(articles_by_ticker.keys())
    subject = f"FULL Stock Intelligence: {tickers_str} - {total_articles} articles"
    success = send_email(subject, html)
    
    return {
        "status": "sent" if success else "failed",
        "articles": total_articles,
        "tickers": list(articles_by_ticker.keys()),
        "recipient": DIGEST_TO
    }

@APP.post("/admin/wipe-database")
def wipe_database(request: Request):
    """DANGER: Completely wipe all feeds, ticker configs, and articles from database"""
    require_admin(request)
    
    # Extra safety check - require a confirmation header
    confirm = request.headers.get("x-confirm-wipe", "")
    if confirm != "YES-WIPE-EVERYTHING":
        raise HTTPException(
            status_code=400, 
            detail="Safety check failed. Add header 'X-Confirm-Wipe: YES-WIPE-EVERYTHING' to confirm"
        )
    
    with db() as conn, conn.cursor() as cur:
        # Delete everything in order of dependencies
        deleted_stats = {}
        
        # Delete all articles
        cur.execute("DELETE FROM found_url")
        deleted_stats["articles"] = cur.rowcount
        
        # Delete all feeds
        cur.execute("DELETE FROM source_feed")
        deleted_stats["feeds"] = cur.rowcount
        
        # Delete all ticker configurations (keywords, competitors, etc.)
        cur.execute("DELETE FROM ticker_config")
        deleted_stats["ticker_configs"] = cur.rowcount
        
        LOG.warning(f"DATABASE WIPED: {deleted_stats}")
    
    return {
        "status": "database_wiped",
        "deleted": deleted_stats,
        "warning": "All feeds, ticker configurations, and articles have been deleted"
    }

@APP.post("/admin/wipe-ticker")
def wipe_ticker(request: Request, ticker: str = Body(..., embed=True)):
    """Wipe all data for a specific ticker"""
    require_admin(request)
    
    with db() as conn, conn.cursor() as cur:
        deleted_stats = {}
        
        # Delete articles for this ticker
        cur.execute("DELETE FROM found_url WHERE ticker = %s", (ticker,))
        deleted_stats["articles"] = cur.rowcount
        
        # Delete feeds for this ticker
        cur.execute("DELETE FROM source_feed WHERE ticker = %s", (ticker,))
        deleted_stats["feeds"] = cur.rowcount
        
        # Delete ticker configuration
        cur.execute("DELETE FROM ticker_config WHERE ticker = %s", (ticker,))
        deleted_stats["ticker_config"] = cur.rowcount
        
        LOG.info(f"Wiped all data for ticker {ticker}: {deleted_stats}")
    
    return {
        "status": "ticker_wiped",
        "ticker": ticker,
        "deleted": deleted_stats
    }

@APP.post("/admin/extract-yahoo-sources")
def extract_yahoo_sources(request: Request, body: ForceDigestRequest):
    """Retroactively extract original sources from Yahoo Finance articles in database"""
    require_admin(request)
    
    with db() as conn, conn.cursor() as cur:
        # Find Yahoo Finance articles without original_source_url
        if body.tickers:
            cur.execute("""
                SELECT id, resolved_url
                FROM found_url
                WHERE (domain LIKE '%yahoo%' OR resolved_url LIKE '%finance.yahoo.com%')
                    AND original_source_url IS NULL
                    AND ticker = ANY(%s)
                ORDER BY found_at DESC
                LIMIT 100
            """, (body.tickers,))
        else:
            cur.execute("""
                SELECT id, resolved_url
                FROM found_url
                WHERE (domain LIKE '%yahoo%' OR resolved_url LIKE '%finance.yahoo.com%')
                    AND original_source_url IS NULL
                ORDER BY found_at DESC
                LIMIT 100
            """)
        
        articles = cur.fetchall()
        
        updated_count = 0
        failed_count = 0
        
        for article in articles:
            if article['resolved_url'] and 'finance.yahoo.com' in article['resolved_url']:
                LOG.info(f"Processing article ID {article['id']}: {article['resolved_url'][:80]}...")
                
                original_source = extract_yahoo_finance_source(article['resolved_url'])
                
                if original_source:
                    cur.execute("""
                        UPDATE found_url
                        SET original_source_url = %s
                        WHERE id = %s
                    """, (original_source, article['id']))
                    updated_count += 1
                    LOG.info(f"Updated article {article['id']} with source: {original_source}")
                else:
                    failed_count += 1
                    LOG.warning(f"Could not extract source for article {article['id']}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
    
    return {
        "status": "completed",
        "articles_processed": len(articles),
        "sources_extracted": updated_count,
        "extraction_failed": failed_count,
        "tickers": body.tickers or "all"
    }

@APP.get("/admin/yahoo-stats")
def get_yahoo_stats(request: Request):
    """Get statistics about Yahoo Finance articles and source extraction"""
    require_admin(request)
    
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT 
                COUNT(*) as total_yahoo_articles,
                COUNT(original_source_url) as sources_extracted,
                COUNT(*) - COUNT(original_source_url) as sources_missing
            FROM found_url
            WHERE domain LIKE '%yahoo%' OR resolved_url LIKE '%finance.yahoo.com%'
        """)
        stats = dict(cur.fetchone())
        
        # Get recent examples with sources
        cur.execute("""
            SELECT ticker, title, resolved_url, original_source_url, found_at
            FROM found_url
            WHERE (domain LIKE '%yahoo%' OR resolved_url LIKE '%finance.yahoo.com%')
                AND original_source_url IS NOT NULL
            ORDER BY found_at DESC
            LIMIT 5
        """)
        stats["recent_extractions"] = list(cur.fetchall())
        
        # Get articles missing sources
        cur.execute("""
            SELECT ticker, title, resolved_url, found_at
            FROM found_url
            WHERE (domain LIKE '%yahoo%' OR resolved_url LIKE '%finance.yahoo.com%')
                AND original_source_url IS NULL
            ORDER BY found_at DESC
            LIMIT 5
        """)
        stats["missing_sources"] = list(cur.fetchall())
    
    return stats

@APP.post("/admin/test-yahoo-extraction")
def test_yahoo_extraction(request: Request):
    """Test Yahoo Finance source extraction with a sample URL"""
    require_admin(request)
    
    # Use the URL you provided as a test case
    test_url = request.headers.get("test-url", "https://finance.yahoo.com/news/why-bloom-energy-stock-trading-162542112.html")
    
    LOG.info(f"Testing Yahoo extraction on: {test_url}")
    
    result = {
        "test_url": test_url,
        "extraction_attempted": True
    }
    
    # Try extraction
    original_source = extract_yahoo_finance_source(test_url)
    
    if original_source:
        result["extraction_successful"] = True
        result["original_source_url"] = original_source
        result["original_domain"] = urlparse(original_source).netloc
    else:
        result["extraction_successful"] = False
        result["error"] = "Could not find providerContentUrl in page content"
        
        # Try to fetch the page for debugging
        try:
            response = requests.get(test_url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            # Check if providerContentUrl exists anywhere in the page
            if "providerContentUrl" in response.text:
                result["pattern_found"] = True
                # Extract a snippet around the pattern for debugging
                idx = response.text.find("providerContentUrl")
                snippet = response.text[max(0, idx-100):min(len(response.text), idx+200)]
                result["snippet"] = snippet[:300]
            else:
                result["pattern_found"] = False
                result["note"] = "providerContentUrl not found in page - may be Yahoo original content"
        except Exception as e:
            result["fetch_error"] = str(e)
    
    return result

@APP.post("/admin/test-email")
def test_email(request: Request):
    """Send test email"""
    require_admin(request)
    
    test_html = """
    <html><body>
        <h2>Quantbrief Test Email</h2>
        <p>Your email configuration is working correctly!</p>
        <p>Time: {}</p>
        <p>AI Integration: {}</p>
        <p>Yahoo Source Extraction: Enabled</p>
    </body></html>
    """.format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Enabled" if OPENAI_API_KEY else "Not configured"
    )
    
    success = send_email("Quantbrief Test Email", test_html)
    return {"status": "sent" if success else "failed", "recipient": DIGEST_TO}

@APP.get("/admin/stats")
def get_stats(
    request: Request,
    tickers: List[str] = Query(default=None, description="Filter stats by tickers")
):
    """Get system statistics"""
    require_admin(request)
    ensure_schema()
    
    with db() as conn, conn.cursor() as cur:
        # Build queries based on tickers
        if tickers:
            # Article stats
            cur.execute("""
                SELECT 
                    COUNT(*) as total_articles,
                    COUNT(DISTINCT ticker) as tickers,
                    COUNT(DISTINCT domain) as domains,
                    AVG(quality_score) as avg_quality,
                    MAX(published_at) as latest_article,
                    COUNT(CASE WHEN original_source_url IS NOT NULL THEN 1 END) as yahoo_sources_extracted
                FROM found_url
                WHERE found_at > NOW() - INTERVAL '7 days'
                    AND ticker = ANY(%s)
            """, (tickers,))
            stats = dict(cur.fetchone())
            
            # Stats by category
            cur.execute("""
                SELECT category, COUNT(*) as count, AVG(quality_score) as avg_score
                FROM found_url
                WHERE found_at > NOW() - INTERVAL '7 days'
                    AND ticker = ANY(%s)
                GROUP BY category
                ORDER BY category
            """, (tickers,))
            stats["by_category"] = list(cur.fetchall())
            
            # Top domains
            cur.execute("""
                SELECT domain, COUNT(*) as count, AVG(quality_score) as avg_score
                FROM found_url
                WHERE found_at > NOW() - INTERVAL '7 days'
                    AND ticker = ANY(%s)
                GROUP BY domain
                ORDER BY count DESC
                LIMIT 10
            """, (tickers,))
            stats["top_domains"] = list(cur.fetchall())
            
            # Articles by ticker and category
            cur.execute("""
                SELECT ticker, category, COUNT(*) as count, AVG(quality_score) as avg_score
                FROM found_url
                WHERE found_at > NOW() - INTERVAL '7 days'
                    AND ticker = ANY(%s)
                GROUP BY ticker, category
                ORDER BY ticker, category
            """, (tickers,))
            stats["by_ticker_category"] = list(cur.fetchall())
        else:
            # Article stats
            cur.execute("""
                SELECT 
                    COUNT(*) as total_articles,
                    COUNT(DISTINCT ticker) as tickers,
                    COUNT(DISTINCT domain) as domains,
                    AVG(quality_score) as avg_quality,
                    MAX(published_at) as latest_article,
                    COUNT(CASE WHEN original_source_url IS NOT NULL THEN 1 END) as yahoo_sources_extracted
                FROM found_url
                WHERE found_at > NOW() - INTERVAL '7 days'
            """)
            stats = dict(cur.fetchone())
            
            # Stats by category
            cur.execute("""
                SELECT category, COUNT(*) as count, AVG(quality_score) as avg_score
                FROM found_url
                WHERE found_at > NOW() - INTERVAL '7 days'
                GROUP BY category
                ORDER BY category
            """)
            stats["by_category"] = list(cur.fetchall())
            
            # Top domains
            cur.execute("""
                SELECT domain, COUNT(*) as count, AVG(quality_score) as avg_score
                FROM found_url
                WHERE found_at > NOW() - INTERVAL '7 days'
                GROUP BY domain
                ORDER BY count DESC
                LIMIT 10
            """)
            stats["top_domains"] = list(cur.fetchall())
            
            # Articles by ticker and category
            cur.execute("""
                SELECT ticker, category, COUNT(*) as count, AVG(quality_score) as avg_score
                FROM found_url
                WHERE found_at > NOW() - INTERVAL '7 days'
                GROUP BY ticker, category
                ORDER BY ticker, category
            """)
            stats["by_ticker_category"] = list(cur.fetchall())
        
        # Check AI metadata status
        cur.execute("""
            SELECT COUNT(*) as total, 
                   COUNT(CASE WHEN ai_generated THEN 1 END) as ai_generated
            FROM ticker_config
            WHERE active = TRUE
        """)
        ai_stats = cur.fetchone()
        stats["ai_metadata"] = ai_stats
    
    return stats

@APP.post("/admin/debug-yahoo-content")
def debug_yahoo_content(request: Request):
    """Debug what's actually in the Yahoo Finance page"""
    require_admin(request)
    
    test_url = request.headers.get("test-url", "https://finance.yahoo.com/news/why-bloom-energy-shares-soaring-155542226.html")
    
    try:
        response = requests.get(test_url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        html_content = response.text
        
        # Find the specific section you mentioned
        # Look for the pattern around "providerContentUrl"
        pattern_index = html_content.find('providerContentUrl')
        
        result = {
            "url": test_url,
            "content_length": len(html_content),
            "provider_url_found": pattern_index != -1
        }
        
        if pattern_index != -1:
            # Extract 500 characters around the match for analysis
            start = max(0, pattern_index - 250)
            end = min(len(html_content), pattern_index + 250)
            context = html_content[start:end]
            
            result["context_snippet"] = context
            result["provider_url_position"] = pattern_index
            
            # Try different regex patterns to see what matches
            patterns_to_test = [
                r'"providerContentUrl"\s*:\s*"([^"]*)"',
                r'"providerContentUrl":"([^"]*)"',
                r'providerContentUrl["\s]*:["\s]*([^",\s]*)',
                r'"providerContentUrl"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
            ]
            
            matches_found = {}
            for i, pattern in enumerate(patterns_to_test):
                matches = re.findall(pattern, html_content)
                matches_found[f"pattern_{i+1}"] = {
                    "pattern": pattern,
                    "matches": matches[:3]  # First 3 matches only
                }
            
            result["regex_test_results"] = matches_found
            
            # Also look for any stockstory.org URLs anywhere in the page
            stockstory_pattern = r'https?://stockstory\.org[^"\s,]*'
            stockstory_matches = re.findall(stockstory_pattern, html_content)
            result["all_stockstory_urls"] = stockstory_matches[:5]  # First 5 matches
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

@APP.get("/admin/domain-names")
def list_domain_names(request: Request):
    """List all cached domain names"""
    require_admin(request)
    
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT domain, formal_name, ai_generated, created_at
            FROM domain_names
            ORDER BY created_at DESC
        """)
        domains = list(cur.fetchall())
    
    return {"domains": domains, "total": len(domains)}

# FIXED: Updated endpoint with proper request body handling
@APP.post("/admin/reset-digest-flags")
def reset_digest_flags(request: Request, body: ResetDigestRequest):
    """Reset sent_in_digest flags for testing"""
    require_admin(request)
    
    with db() as conn, conn.cursor() as cur:
        if body.tickers:
            cur.execute("UPDATE found_url SET sent_in_digest = FALSE WHERE ticker = ANY(%s)", (body.tickers,))
        else:
            cur.execute("UPDATE found_url SET sent_in_digest = FALSE")
        count = cur.rowcount
    
    return {"status": "reset", "articles_reset": count, "tickers": body.tickers or "all"}

@APP.post("/admin/debug-google-resolution")
def debug_google_resolution(request: Request):
    """Debug Google News URL resolution process"""
    require_admin(request)
    
    # Test URL from your example
    test_url = request.headers.get("test-url", "https://news.google.com/rss/articles/CBMiyAFBVV95cUxPRlpMdS1fSVJfMWo3VTNyVndlQVpEUlF4aENOS2p0QXoxRVB6cGJ0VEwzNVhaa3JYSmRVNnR5T244OGM0VVZjSVNKMXRsRVRmWVVqWl9BaEVQYXdJakFmTVRDNm1NUE5sSTBwYlhmQkxwU3g3TUVZaUFpdkxsME5PbHF2RUt5cHNneGZCamhhMUxhaGFiSEZEU1pQYlBXa1M3bkRTNXNxb2toZ1p3MFV0bGNiR2pPSkY4Y291clZjQ1dEX01OdFFoag?oc=5&hl=en-US&gl=US&ceid=US:en")
    
    LOG.info(f"Testing Google News resolution on: {test_url}")
    
    result = {
        "test_url": test_url,
        "resolution_attempted": True
    }
    
    try:
        # Test the resolve_google_news_url function
        resolved_result = resolve_google_news_url(test_url)
        
        if resolved_result[0] is None:
            result["resolution_status"] = "blocked_or_failed"
            result["resolved_url"] = None
            result["domain"] = None
            result["yahoo_source_url"] = None
        else:
            resolved_url, domain, yahoo_source_url = resolved_result
            result["resolution_status"] = "success"
            result["resolved_url"] = resolved_url
            result["domain"] = domain
            result["yahoo_source_url"] = yahoo_source_url
            
            # Test domain name resolution
            if domain:
                formal_name = get_or_create_formal_domain_name(domain)
                result["formal_domain_name"] = formal_name
                result["ai_generated"] = True  # Check if it was generated or cached
                
                # Check what's in the database for this domain
                with db() as conn, conn.cursor() as cur:
                    cur.execute("SELECT * FROM domain_names WHERE domain = %s", (domain.replace("www.", "").lower(),))
                    domain_record = cur.fetchone()
                    result["domain_record"] = dict(domain_record) if domain_record else None
    
    except Exception as e:
        result["error"] = str(e)
        LOG.error(f"Error testing Google News resolution: {e}")
    
    return result

@APP.post("/admin/test-title-extraction")
def test_title_extraction(request: Request):
    """Test Google News title source extraction"""
    require_admin(request)
    
    test_title = request.headers.get("test-title", "CCL Products Reaches All-Time High Amidst Market Volatility and Sector Underperformance - Markets Mojo")
    
    result = {
        "original_title": test_title,
        "extraction_attempted": True
    }
    
    try:
        # Test the extraction
        cleaned_title, extracted_source = extract_source_from_title(test_title)
        result["cleaned_title"] = cleaned_title
        result["extracted_source"] = extracted_source
        
        if extracted_source:
            enhanced_source = enhance_source_with_ai(extracted_source)
            result["enhanced_source"] = enhanced_source
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# ------------------------------------------------------------------------------
# CLI Support for PowerShell Commands
# ------------------------------------------------------------------------------
@APP.post("/cli/run")
def cli_run(request: Request, body: CLIRequest):
    """CLI endpoint for PowerShell commands"""
    require_admin(request)
    
    results = {}
    
    if body.action in ["ingest", "both"]:
        # Initialize feeds if needed
        ensure_schema()
        for ticker in body.tickers:
            metadata = get_or_create_ticker_metadata(ticker)
            feeds = build_feed_urls(ticker, metadata)
            for feed_config in feeds:
                upsert_feed(
                    url=feed_config["url"],
                    name=feed_config["name"],
                    ticker=ticker,
                    category=feed_config.get("category", "company"),
                    retain_days=DEFAULT_RETAIN_DAYS
                )
        
        # Run ingestion
        ingest_result = cron_ingest(request, body.minutes, body.tickers)
        results["ingest"] = ingest_result
    
    if body.action in ["digest", "both"]:
        # Run digest
        digest_result = cron_digest(request, body.minutes, body.tickers)
        results["digest"] = digest_result
    
    return results

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(APP, host="0.0.0.0", port=port)
