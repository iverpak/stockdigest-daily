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
    """Use OpenAI to generate industry keywords and competitors for a ticker"""
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
        
        # FIXED: Remove temperature parameter for gpt-4o-mini and use max_completion_tokens
        data = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a financial analyst expert who provides accurate information about companies and their industries."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 500
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
        LOG.info(f"OpenAI raw response preview: {response.text[:200]}")
        
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
        
        # With JSON mode, content should be valid JSON without fences
        metadata = json.loads(content)
        
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
    """Extract original source URL from Yahoo Finance article"""
    try:
        # Only process Yahoo Finance URLs
        if "finance.yahoo.com" not in url:
            return None
            
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code != 200:
            return None
        
        # Look for the providerContentUrl pattern in the HTML
        # Using regex to find the pattern in the raw HTML
        pattern = r'"providerContentUrl"\s*:\s*"([^"]+)"'
        match = re.search(pattern, response.text)
        
        if match:
            original_url = match.group(1)
            # Unescape any escaped characters
            original_url = original_url.replace('\/', '/')
            
            # Validate it's a proper URL
            parsed = urlparse(original_url)
            if parsed.scheme and parsed.netloc:
                LOG.info(f"Found Yahoo Finance original source: {original_url}")
                return original_url
        
        return None
        
    except Exception as e:
        LOG.debug(f"Failed to extract Yahoo Finance source from {url}: {e}")
        return None

def resolve_google_news_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve Google News redirect URL and check for spam domains - ENHANCED with Yahoo source extraction"""
    original_source_url = None
    
    try:
        # Extract actual URL from Google News redirect
        if "news.google.com" in url and "/articles/" in url:
            response = requests.get(url, timeout=10, allow_redirects=True)
            final_url = response.url
            domain = urlparse(final_url).netloc.lower()
            
            # FIXED: Check if final destination is spam before returning
            for spam_domain in SPAM_DOMAINS:
                spam_clean = spam_domain.replace("www.", "").lower()
                if spam_clean in domain:
                    LOG.info(f"BLOCKED spam destination after redirect: {domain} (matched: {spam_clean})")
                    return None, None, None  # Return None to skip this article
            
            # Check if it's Yahoo Finance and extract original source
            if "finance.yahoo.com" in final_url:
                original_source_url = extract_yahoo_finance_source(final_url)
            
            return final_url, domain, original_source_url
        
        # For direct Google redirect URLs
        if "google.com/url" in url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            if 'url' in params:
                actual_url = params['url'][0]
                domain = urlparse(actual_url).netloc.lower()
                
                # Check for spam in direct URLs too
                for spam_domain in SPAM_DOMAINS:
                    spam_clean = spam_domain.replace("www.", "").lower()
                    if spam_clean in domain:
                        LOG.info(f"BLOCKED spam destination in redirect: {domain}")
                        return None, None, None
                
                # Check if it's Yahoo Finance and extract original source
                if "finance.yahoo.com" in actual_url:
                    original_source_url = extract_yahoo_finance_source(actual_url)
                        
                return actual_url, domain, original_source_url
            elif 'q' in params:
                actual_url = params['q'][0]
                domain = urlparse(actual_url).netloc.lower()
                
                # Check for spam
                for spam_domain in SPAM_DOMAINS:
                    spam_clean = spam_domain.replace("www.", "").lower()
                    if spam_clean in domain:
                        LOG.info(f"BLOCKED spam destination in q param: {domain}")
                        return None, None, None
                
                # Check if it's Yahoo Finance and extract original source
                if "finance.yahoo.com" in actual_url:
                    original_source_url = extract_yahoo_finance_source(actual_url)
                        
                return actual_url, domain, original_source_url
        
        # Already a direct URL
        domain = urlparse(url).netloc.lower()
        
        # Check direct URLs for spam too
        for spam_domain in SPAM_DOMAINS:
            spam_clean = spam_domain.replace("www.", "").lower()
            if spam_clean in domain:
                LOG.info(f"BLOCKED spam direct URL: {domain}")
                return None, None, None
        
        # Check if it's Yahoo Finance and extract original source
        if "finance.yahoo.com" in url:
            original_source_url = extract_yahoo_finance_source(url)
                
        return url, domain, original_source_url
        
    except Exception as e:
        LOG.warning(f"Failed to resolve URL {url}: {e}")
        return url, urlparse(url).netloc.lower() if url else None, None

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
    """Process a single feed and store articles with category - ENHANCED with Yahoo source extraction"""
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
            
            # ENHANCED: URL resolution with spam checking and Yahoo source extraction
            resolved_result = resolve_google_news_url(original_url)
            if resolved_result[0] is None:  # Spam detected
                stats["blocked_spam"] += 1
                continue
                
            resolved_url, domain, original_source_url = resolved_result
            if not resolved_url or not domain:
                continue
            
            # Track Yahoo source extractions
            if original_source_url:
                stats["yahoo_sources_found"] += 1
                LOG.info(f"Yahoo Finance article resolved to: {original_source_url}")
                
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
                        category, related_ticker, original_source_url
                    ))
                    
                    if cur.fetchone():
                        stats["inserted"] += 1
                        source_note = f" (source: {urlparse(original_source_url).netloc})" if original_source_url else ""
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
    """Build HTML email digest with keyword information"""
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
        ".score { display: inline-block; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 10px; }",
        ".high-score { background-color: #d4edda; color: #155724; }",
        ".med-score { background-color: #fff3cd; color: #856404; }",
        ".low-score { background-color: #f8d7da; color: #721c24; }",
        ".keywords { background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 11px; }",
        ".source-indicator { color: #0066cc; font-size: 10px; font-weight: bold; margin-left: 5px; }",
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
            • [SOURCE] indicator shows when original source was extracted from aggregator sites
        </div>
        </body></html>
    """)
    
    return "".join(html)

def _format_article_html(article: Dict, category: str) -> str:
    """Format a single article for HTML display with original source support"""
    pub_date = article["published_at"].strftime("%m/%d %H:%M") if article["published_at"] else "N/A"
    
    title = article["title"] or "No Title"
    # Clean up title
    suffixes_to_remove = [
        " - MarketBeat", " - Newser", " - TipRanks", " - MSN", 
        " - The Daily Item", " - MarketScreener", " - Seeking Alpha",
        " - simplywall.st", " - Investopedia"
    ]
    
    for suffix in suffixes_to_remove:
        if title.endswith(suffix):
            title = title[:-len(suffix)].strip()
    
    title = re.sub(r'\s*\$[A-Z]+\s*-?\s*', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    
    # Determine which URL to use for the main link
    link_url = article.get("original_source_url") or article["resolved_url"] or article["url"]
    
    # Get domain for display
    domain = article["domain"] or "unknown"
    domain = domain.replace("www.", "")
    
    # Add source indicator if we found an original source
    source_indicator = ""
    if article.get("original_source_url"):
        original_domain = urlparse(article["original_source_url"]).netloc.replace("www.", "")
        source_indicator = f"<span class='source-indicator'>[SOURCE: {original_domain}]</span>"
        # Update domain display if we have original source
        domain = f"Yahoo→{original_domain}"
    
    score = article["quality_score"]
    score_class = "high-score" if score >= 70 else "med-score" if score >= 40 else "low-score"
    
    related = f" | Related: {article.get('related_ticker', '')}" if article.get('related_ticker') else ""
    
    return f"""
    <div class='article {category}'>
        <a href='{link_url}' target='_blank'>{title}</a>
        {source_indicator}
        <span class='meta'> | {domain} | {pub_date}</span>
        <span class='score {score_class}'>Score: {score:.0f}</span>
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
