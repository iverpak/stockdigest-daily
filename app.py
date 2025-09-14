import os
import sys
import time
import logging
import hashlib
import re
import pytz
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
    "khodrobank.com", "www.khodrobank.com", "khodrobank"  # ADD this line
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
    """Use OpenAI to generate industry keywords and competitors for a ticker - UPDATED with company name and competitor tickers"""
    if not OPENAI_API_KEY:
        LOG.error("OpenAI API key not configured")
        return {"industry_keywords": [], "competitors": [], "company_name": ticker}

    # NEW: Get the full company name first
    company_name_prompt = f"""For the stock ticker "{ticker}", what is the full, official company name?

Provide only the official company name as it appears in SEC filings, without "Inc.", "Corp.", "Ltd." unless critical for identification.

Examples:
- AAPL → Apple
- TSLA → Tesla
- JPM → JPMorgan Chase
- BRK.B → Berkshire Hathaway

Company name for {ticker}:"""

    # Hedge fund research assistant prompt for industry keywords
    industry_prompt = f"""You are a hedge-fund research assistant. Given a public company (name or ticker), output exactly five short tracking keywords an analyst would monitor for THIS specific business. Focus on concrete revenue drivers, products/sub-industry, supply chain/market structure, and—if material—regulatory bodies/markets (e.g., FAA, FCC, PJM, FDA). Avoid generic terms ("technology," "finance") and people names unless central to the thesis. Each keyword ≤ 3 words. No explanations. Return strict JSON:

Company ticker: {ticker}

{{"industry_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]}}"""

    # ENHANCED: Updated competitor prompt to exclude target ticker
    competitor_prompt = f"""For the publicly traded stock ticker {ticker}, identify exactly three main direct competitors that are also publicly traded companies. Focus on companies in the same specific business segment with similar revenue models.

IMPORTANT: Do NOT include {ticker} itself in the list of competitors.

For each competitor, provide both the full company name AND the stock ticker symbol.

Return in strict JSON format:

{{"competitors": [
    {{"name": "Full Company Name 1", "ticker": "TICK1"}},
    {{"name": "Full Company Name 2", "ticker": "TICK2"}},
    {{"name": "Full Company Name 3", "ticker": "TICK3"}}
]}}"""

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        # Base parameters for all requests
        base_data = {
            "model": OPENAI_MODEL,
            "temperature": 0.15,
            "max_tokens": 80,  # Increased for more complex responses
        }

        # Get company name
        company_name_data = {
            **base_data,
            "max_tokens": 30,
            "messages": [
                {"role": "system", "content": "You are a financial data expert. Provide only the official company name, nothing else."},
                {"role": "user", "content": company_name_prompt},
            ],
        }

        LOG.info(f"Requesting company name for {ticker} from {OPENAI_MODEL}")
        company_name_response = requests.post(OPENAI_API_URL, headers=headers, json=company_name_data, timeout=30)
        
        company_name = ticker  # Fallback
        if company_name_response.status_code == 200:
            try:
                company_result = company_name_response.json()
                company_name = company_result["choices"][0]["message"]["content"].strip()
                # Clean up any quotes or extra formatting
                company_name = re.sub(r'^["\']|["\']$', '', company_name)
                LOG.info(f"Retrieved company name for {ticker}: {company_name}")
            except Exception as e:
                LOG.warning(f"Failed to parse company name for {ticker}: {e}")

        # Get industry keywords
        industry_data = {
            **base_data,
            "max_tokens": 60,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a hedge-fund research assistant who provides precise, actionable tracking keywords for equity research. Always respond with valid JSON only."},
                {"role": "user", "content": industry_prompt},
            ],
        }

        LOG.info(f"Requesting industry keywords for {ticker} from {OPENAI_MODEL}")
        industry_response = requests.post(OPENAI_API_URL, headers=headers, json=industry_data, timeout=30)
        if industry_response.status_code != 200:
            LOG.error(f"OpenAI API {industry_response.status_code} error for industry keywords: {industry_response.text}")
            return {"industry_keywords": [], "competitors": [], "company_name": company_name}

        # Get competitors with tickers
        competitor_data = {
            **base_data,
            "max_tokens": 120,  # More tokens for structured competitor data
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a financial analyst expert who identifies direct competitors with their ticker symbols. Always respond with valid JSON only."},
                {"role": "user", "content": competitor_prompt},
            ],
        }

        LOG.info(f"Requesting competitors with tickers for {ticker} from {OPENAI_MODEL}")
        competitor_response = requests.post(OPENAI_API_URL, headers=headers, json=competitor_data, timeout=30)
        if competitor_response.status_code != 200:
            LOG.error(f"OpenAI API {competitor_response.status_code} error for competitors: {competitor_response.text}")
            return {"industry_keywords": [], "competitors": [], "company_name": company_name}

        # Parse industry keywords
        try:
            industry_result = industry_response.json()
            industry_content = industry_result["choices"][0]["message"]["content"]
            industry_metadata = json.loads(industry_content)
            raw_keywords = industry_metadata.get("industry_keywords", [])[:5]
            
            # Capitalize industry keywords properly
            industry_keywords = []
            for keyword in raw_keywords:
                capitalized = ' '.join(word.capitalize() for word in keyword.split())
                industry_keywords.append(capitalized)
            
        except (json.JSONDecodeError, KeyError) as e:
            LOG.error(f"Failed to parse industry keywords for {ticker}: {e}")
            industry_keywords = []

        # Parse competitors with enhanced structure
        try:
            competitor_result = competitor_response.json()
            competitor_content = competitor_result["choices"][0]["message"]["content"]
            competitor_metadata = json.loads(competitor_content)
            raw_competitors = competitor_metadata.get("competitors", [])[:3]
            
            # Process competitors to handle both old and new format
            competitors = []
            for comp in raw_competitors:
                if isinstance(comp, dict) and 'name' in comp and 'ticker' in comp:
                    # New format with name and ticker
                    competitors.append({
                        "name": comp['name'],
                        "ticker": comp['ticker']
                    })
                elif isinstance(comp, str):
                    # Old format - just company name
                    competitors.append({
                        "name": comp,
                        "ticker": None
                    })
            
        except (json.JSONDecodeError, KeyError) as e:
            LOG.error(f"Failed to parse competitors for {ticker}: {e}")
            competitors = []

        LOG.info(f"Successfully generated AI metadata for {ticker}: company='{company_name}', {len(industry_keywords)} keywords, {len(competitors)} competitors with tickers")
        return {
            "industry_keywords": industry_keywords, 
            "competitors": competitors,
            "company_name": company_name
        }

    except Exception as e:
        LOG.error(f"OpenAI API error for ticker {ticker}: {e}")
        return {"industry_keywords": [], "competitors": [], "company_name": ticker}
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
    """Initialize database schema with enhanced tables - UPDATED for search metadata"""
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

            # ADD THIS: Create domain statistics table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS domain_stats (
                    domain VARCHAR(255) PRIMARY KEY,
                    total_articles INT DEFAULT 0,
                    avg_quality_score FLOAT DEFAULT 50.0,
                    first_seen TIMESTAMP DEFAULT NOW(),
                    last_seen TIMESTAMP DEFAULT NOW(),
                    category_counts JSONB DEFAULT '{}',
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # ENHANCED: Add search metadata columns to found_url if not exists
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
            
            # ENHANCED: Add search keyword column to track what was searched
            cur.execute("""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name='found_url' AND column_name='search_keyword') 
                    THEN 
                        ALTER TABLE found_url ADD COLUMN search_keyword VARCHAR(255);
                    END IF;
                END $$;
            """)
            
            # ENHANCED: Add search metadata columns to source_feed if not exists
            cur.execute("""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name='source_feed' AND column_name='search_keyword') 
                    THEN 
                        ALTER TABLE source_feed ADD COLUMN search_keyword VARCHAR(255);
                    END IF;
                END $$;
            """)
            
            cur.execute("""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name='source_feed' AND column_name='competitor_ticker') 
                    THEN 
                        ALTER TABLE source_feed ADD COLUMN competitor_ticker VARCHAR(10);
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
                CREATE INDEX IF NOT EXISTS idx_found_url_ticker_published ON found_url(ticker, published_at DESC);
                CREATE INDEX IF NOT EXISTS idx_found_url_search_keyword ON found_url(search_keyword);
                CREATE INDEX IF NOT EXISTS idx_source_feed_search_keyword ON source_feed(search_keyword);
                CREATE INDEX IF NOT EXISTS idx_source_feed_competitor_ticker ON source_feed(competitor_ticker);
                CREATE INDEX IF NOT EXISTS idx_domain_stats_total_articles ON domain_stats(total_articles DESC);
                CREATE INDEX IF NOT EXISTS idx_domain_stats_avg_quality ON domain_stats(avg_quality_score DESC);
                CREATE INDEX IF NOT EXISTS idx_domain_stats_last_seen ON domain_stats(last_seen DESC);
            """)

def upsert_ticker_config(ticker: str, metadata: Dict, ai_generated: bool = False):
    """Insert or update ticker configuration with enhanced competitor structure"""
    with db() as conn, conn.cursor() as cur:
        # Convert new competitor format to store both name and ticker info
        competitors_for_db = []
        raw_competitors = metadata.get("competitors", [])
        
        for comp in raw_competitors:
            if isinstance(comp, dict):
                # New format - store as "Name (TICKER)" or just "Name" if no ticker
                if comp.get('ticker'):
                    competitors_for_db.append(f"{comp['name']} ({comp['ticker']})")
                else:
                    competitors_for_db.append(comp['name'])
            else:
                # Old format - just the string
                competitors_for_db.append(str(comp))
        
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
            metadata.get("company_name", ticker),
            metadata.get("industry_keywords", []),
            competitors_for_db,
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
    """Get ticker metadata from DB or generate with AI if not exists - UPDATED for new competitor structure"""
    # Check database first
    if not force_refresh:
        config = get_ticker_config(ticker)
        if config:
            LOG.info(f"Using existing metadata for {ticker} (AI generated: {config.get('ai_generated', False)})")
            
            # Process stored competitors back to structured format
            stored_competitors = config.get("competitors", [])
            processed_competitors = []
            
            for comp_str in stored_competitors:
                # Check if it's in "Name (TICKER)" format
                ticker_match = re.search(r'^(.+?)\s*\(([A-Z]{1,5})\)$', comp_str)
                if ticker_match:
                    processed_competitors.append({
                        "name": ticker_match.group(1).strip(),
                        "ticker": ticker_match.group(2)
                    })
                else:
                    # Just company name
                    processed_competitors.append({
                        "name": comp_str,
                        "ticker": None
                    })
            
            return {
                "company_name": config.get("name", ticker),
                "company": [ticker],
                "industry": config.get("industry_keywords", []),
                "competitors": processed_competitors
            }
    
    # Generate with AI
    LOG.info(f"Generating metadata for {ticker} using OpenAI...")
    ai_metadata = generate_ticker_metadata_with_ai(ticker)
    
    # Save to database
    metadata = {
        "company_name": ai_metadata.get("company_name", ticker),
        "name": ai_metadata.get("company_name", ticker),
        "industry_keywords": ai_metadata.get("industry_keywords", []),
        "competitors": ai_metadata.get("competitors", [])
    }
    upsert_ticker_config(ticker, metadata, ai_generated=True)
    
    return {
        "company_name": metadata["company_name"],
        "company": [ticker],
        "industry": metadata["industry_keywords"],
        "competitors": metadata["competitors"]
    }

# ADD THE NEW DOMAIN TRACKING FUNCTIONS HERE:

def normalize_domain(domain: str) -> str:
    """Normalize domain for consistent storage"""
    if not domain:
        return None
    
    # Convert to lowercase and remove common prefixes
    normalized = domain.lower().strip()
    
    # Remove www. prefix for consistency (except for specific cases)
    if normalized.startswith('www.') and normalized != 'www.':
        normalized = normalized[4:]
    
    # Remove trailing slashes
    normalized = normalized.rstrip('/')
    
    return normalized

def upsert_domain_stats(domain: str, quality_score: float, category: str = "company"):
    """Update domain statistics when a new article is added"""
    if not domain:
        return
        
    normalized_domain = normalize_domain(domain)
    
    with db() as conn, conn.cursor() as cur:
        # Check if domain exists
        cur.execute("SELECT total_articles, category_counts FROM domain_stats WHERE domain = %s", (normalized_domain,))
        existing = cur.fetchone()
        
        if existing:
            # Update existing domain stats
            new_total = existing['total_articles'] + 1
            category_counts = existing.get('category_counts') or {}
            category_counts[category] = category_counts.get(category, 0) + 1
            
            cur.execute("""
                UPDATE domain_stats 
                SET total_articles = %s,
                    avg_quality_score = (
                        (avg_quality_score * %s + %s) / %s
                    ),
                    last_seen = NOW(),
                    category_counts = %s,
                    updated_at = NOW()
                WHERE domain = %s
            """, (new_total, existing['total_articles'], quality_score, new_total, 
                  json.dumps(category_counts), normalized_domain))
        else:
            # Insert new domain
            category_counts = {category: 1}
            cur.execute("""
                INSERT INTO domain_stats (domain, total_articles, avg_quality_score, 
                                        first_seen, last_seen, category_counts)
                VALUES (%s, 1, %s, NOW(), NOW(), %s)
            """, (normalized_domain, quality_score, json.dumps(category_counts)))

def get_domain_stats(domain: str = None) -> List[Dict]:
    """Get domain statistics for scoring improvements"""
    with db() as conn, conn.cursor() as cur:
        if domain:
            cur.execute("""
                SELECT domain, total_articles, avg_quality_score, 
                       first_seen, last_seen, category_counts
                FROM domain_stats 
                WHERE domain = %s
            """, (normalize_domain(domain),))
            result = cur.fetchone()
            return dict(result) if result else None
        else:
            cur.execute("""
                SELECT domain, total_articles, avg_quality_score,
                       first_seen, last_seen, category_counts
                FROM domain_stats 
                ORDER BY total_articles DESC
                LIMIT 100
            """)
            return list(cur.fetchall())

def cache_missing_domain_names(limit: int = 50) -> Dict[str, int]:
    """Bulk cache formal names for domains that don't have them yet - OPTIMIZED batch processing"""
    
    with db() as conn, conn.cursor() as cur:
        # Find domains without formal names, ordered by frequency
        cur.execute("""
            SELECT f.domain, COUNT(*) as usage_count
            FROM found_url f
            LEFT JOIN domain_names dn ON dn.domain = f.domain
            WHERE f.domain IS NOT NULL 
              AND f.domain != 'news.google.com'
              AND f.domain != 'google-news-unresolved'
              AND dn.domain IS NULL
            GROUP BY f.domain
            ORDER BY usage_count DESC
            LIMIT %s
        """, (limit,))
        
        missing_domains = list(cur.fetchall())
    
    if not missing_domains:
        LOG.info("No missing domain names found")
        return {"processed": 0, "cached": 0, "failed": 0}
    
    LOG.info(f"Found {len(missing_domains)} domains needing formal names")
    
    stats = {"processed": 0, "cached": 0, "failed": 0}
    
    for domain_row in missing_domains:
        domain = domain_row['domain']
        usage_count = domain_row['usage_count']
        
        stats["processed"] += 1
        LOG.info(f"Processing domain {domain} (used in {usage_count} articles)...")
        
        # Get formal name (this will cache it automatically)
        try:
            formal_name = get_or_create_formal_domain_name(domain)
            if formal_name and formal_name != domain:
                stats["cached"] += 1
                LOG.info(f"Successfully cached: {domain} -> {formal_name}")
            else:
                stats["failed"] += 1
                LOG.warning(f"Failed to get good formal name for {domain}")
        except Exception as e:
            stats["failed"] += 1
            LOG.error(f"Error processing domain {domain}: {e}")
        
        # Small delay to avoid rate limiting
        time.sleep(0.2)
    
    LOG.info(f"Domain caching completed: {stats}")
    return stats

def build_feed_urls(ticker: str, keywords: Dict[str, List[str]]) -> List[Dict]:
    """Build feed URLs for different categories - FIXED Yahoo Finance ticker feeds"""
    feeds = []
    
    company_name = keywords.get("company_name", ticker)
    LOG.info(f"Building feeds for {ticker} (company: {company_name})")
    
    # FIXED: Company-specific feeds - use company name for Google, ticker for Yahoo
    company_name_encoded = requests.utils.quote(company_name)
    feeds.extend([
        {
            "url": f"https://news.google.com/rss/search?q=\"{company_name_encoded}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
            "name": f"Google News: {company_name}",
            "category": "company",
            "search_keyword": company_name
        },
        {
            "url": f"https://finance.yahoo.com/rss/headline?s={ticker}",
            "name": f"Yahoo Finance: {ticker}",
            "category": "company", 
            "search_keyword": ticker  # FIXED: Use ticker for Yahoo feeds
        }
    ])
    
    # Industry feeds (Google only - Yahoo doesn't work with keywords)
    industry_keywords = keywords.get("industry", [])
    LOG.info(f"Building industry feeds for {ticker} with keywords: {industry_keywords}")
    for keyword in industry_keywords[:3]:
        keyword_encoded = requests.utils.quote(keyword)
        feeds.append({
            "url": f"https://news.google.com/rss/search?q=\"{keyword_encoded}\"+when:7d&hl=en-US&gl=US&ceid=US:en",
            "name": f"Industry: {keyword}",
            "category": "industry",
            "search_keyword": keyword
        })
    
    # FIXED: Competitor feeds with better validation to prevent cross-contamination
    competitors = keywords.get("competitors", [])
    LOG.info(f"Building competitor feeds for {ticker} with competitors: {competitors}")
    
    for comp in competitors[:3]:
        if isinstance(comp, dict):
            # New structured format
            comp_name = comp.get('name', '')
            comp_ticker = comp.get('ticker')
            
            # FIXED: Validate competitor ticker is different from main ticker
            if comp_ticker and comp_ticker.upper() == ticker.upper():
                LOG.warning(f"Skipping competitor {comp_name} ({comp_ticker}) - same as main ticker {ticker}")
                continue
            
            if comp_name:
                # Use company name for Google search
                comp_name_encoded = requests.utils.quote(comp_name)
                feeds.append({
                    "url": f"https://news.google.com/rss/search?q=\"{comp_name_encoded}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
                    "name": f"Competitor: {comp_name}",
                    "category": "competitor",
                    "search_keyword": comp_name,
                    "competitor_ticker": comp_ticker
                })
                
                # Use ticker for Yahoo search if available and different from main ticker
                if comp_ticker and comp_ticker.upper() != ticker.upper():
                    feeds.append({
                        "url": f"https://finance.yahoo.com/rss/headline?s={comp_ticker}",
                        "name": f"Yahoo Competitor: {comp_name} ({comp_ticker})",
                        "category": "competitor",
                        "search_keyword": comp_ticker,  # FIXED: Use competitor ticker
                        "competitor_ticker": comp_ticker
                    })
        else:
            # Old format - just company name string
            comp_clean = str(comp).replace("(", "").replace(")", "").strip()
            
            # Try to extract ticker if present in format "Name (TICKER)"
            ticker_match = re.search(r'\(([A-Z]{1,5})\)', str(comp))
            extracted_ticker = ticker_match.group(1) if ticker_match else None
            comp_name_only = re.sub(r'\s*\([^)]*\)', '', comp_clean).strip()
            
            # FIXED: Validate extracted ticker is different from main ticker
            if extracted_ticker and extracted_ticker.upper() == ticker.upper():
                LOG.warning(f"Skipping competitor {comp_name_only} ({extracted_ticker}) - same as main ticker {ticker}")
                continue
            
            comp_name_encoded = requests.utils.quote(comp_name_only)
            feeds.append({
                "url": f"https://news.google.com/rss/search?q=\"{comp_name_encoded}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
                "name": f"Competitor: {comp_name_only}",
                "category": "competitor",
                "search_keyword": comp_name_only,
                "competitor_ticker": extracted_ticker
            })
            
            if extracted_ticker and extracted_ticker.upper() != ticker.upper():
                feeds.append({
                    "url": f"https://finance.yahoo.com/rss/headline?s={extracted_ticker}",
                    "name": f"Yahoo Competitor: {comp_name_only} ({extracted_ticker})",
                    "category": "competitor", 
                    "search_keyword": extracted_ticker,  # FIXED: Use competitor ticker
                    "competitor_ticker": extracted_ticker
                })
    
    LOG.info(f"Built {len(feeds)} total feeds for {ticker}: {len([f for f in feeds if f['category'] == 'company'])} company, {len([f for f in feeds if f['category'] == 'industry'])} industry, {len([f for f in feeds if f['category'] == 'competitor'])} competitor")
    return feeds
    
def upsert_feed(url: str, name: str, ticker: str, category: str = "company", retain_days: int = 90, search_keyword: str = None, competitor_ticker: str = None) -> int:
    """Insert or update a feed source with category and search metadata - ENHANCED"""
    with db() as conn, conn.cursor() as cur:
        # First, check if we need to add the new columns
        cur.execute("""
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                             WHERE table_name='source_feed' AND column_name='search_keyword') 
                THEN 
                    ALTER TABLE source_feed ADD COLUMN search_keyword VARCHAR(255);
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                             WHERE table_name='source_feed' AND column_name='competitor_ticker') 
                THEN 
                    ALTER TABLE source_feed ADD COLUMN competitor_ticker VARCHAR(10);
                END IF;
            END $$;
        """)
        
        cur.execute("""
            INSERT INTO source_feed (url, name, ticker, retain_days, active, search_keyword, competitor_ticker)
            VALUES (%s, %s, %s, %s, TRUE, %s, %s)
            ON CONFLICT (url) DO UPDATE
            SET name = EXCLUDED.name,
                ticker = EXCLUDED.ticker,
                retain_days = EXCLUDED.retain_days,
                active = TRUE,
                search_keyword = EXCLUDED.search_keyword,
                competitor_ticker = EXCLUDED.competitor_ticker
            RETURNING id;
        """, (url, name, ticker, retain_days, search_keyword, competitor_ticker))
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
# FIX #6: Add function to detect non-Latin script
def contains_non_latin_script(text: str) -> bool:
    """Detect if text contains non-Latin script characters (Arabic, Chinese, etc.)"""
    if not text:
        return False
    
    # Unicode ranges for non-Latin scripts that commonly appear in spam
    non_latin_ranges = [
        # Arabic and Arabic Supplement
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
        
        # Chinese, Japanese, Korean
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0x3400, 0x4DBF),  # CJK Extension A
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0xAC00, 0xD7AF),  # Hangul Syllables
        
        # Cyrillic
        (0x0400, 0x04FF),  # Cyrillic
        (0x0500, 0x052F),  # Cyrillic Supplement
        
        # Hebrew
        (0x0590, 0x05FF),  # Hebrew
        
        # Thai
        (0x0E00, 0x0E7F),  # Thai
        
        # Devanagari (Hindi, Sanskrit)
        (0x0900, 0x097F),  # Devanagari
    ]
    
    for char in text:
        char_code = ord(char)
        for start, end in non_latin_ranges:
            if start <= char_code <= end:
                return True
    
    return False

def extract_yahoo_finance_source_optimized(url: str) -> Optional[str]:
    """
    Optimized Yahoo Finance source extraction with better error handling
    """
    try:
        if "finance.yahoo.com" not in url:
            return None
            
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code != 200:
            LOG.warning(f"HTTP {response.status_code} when fetching Yahoo URL: {url}")
            return None
        
        html_content = response.text
        
        # Try most reliable patterns first
        patterns = [
            r'"providerContentUrl"\s*:\s*"([^"]*)"',
            r'"sourceUrl"\s*:\s*"([^"]*)"',
            r'"originalUrl"\s*:\s*"([^"]*)"'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html_content)
            for match in matches:
                try:
                    # Clean up escaped JSON
                    cleaned_url = json.loads(f'"{match}"')
                    parsed = urlparse(cleaned_url)
                    
                    if (parsed.scheme in ['http', 'https'] and 
                        parsed.netloc and 
                        len(cleaned_url) > 20 and
                        'finance.yahoo.com' not in cleaned_url):
                        
                        LOG.info(f"Yahoo Finance source extraction successful: {cleaned_url}")
                        return cleaned_url
                except Exception:
                    continue
        
        # Fallback pattern search
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
                        LOG.info(f"Yahoo source via pattern search: {candidate_url}")
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
    """
    Enhanced Google News URL resolution - returns URL for title processing
    """
    try:
        if "news.google.com" in url:
            # Try direct resolution first
            try:
                response = requests.get(url, timeout=10, allow_redirects=True, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                final_url = response.url
                
                if final_url != url and "news.google.com" not in final_url:
                    domain = urlparse(final_url).netloc.lower()
                    normalized_domain = normalize_domain(domain)
                    
                    # Spam check
                    for spam_domain in SPAM_DOMAINS:
                        spam_clean = normalize_domain(spam_domain)
                        if spam_clean and spam_clean in normalized_domain:
                            return None, None, None
                    
                    LOG.info(f"Successfully resolved Google News URL: {url} -> {final_url}")
                    return final_url, normalized_domain, None
                    
            except Exception as e:
                LOG.debug(f"Direct Google News resolution failed for {url}: {e}")
            
            # Return URL with special marker for title-based processing
            return url, "google-news-needs-title-processing", None
        
        # Handle direct Google redirect URLs
        if "google.com/url" in url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            if 'url' in params:
                actual_url = params['url'][0]
                domain = urlparse(actual_url).netloc.lower()
                normalized_domain = normalize_domain(domain)
                
                # Spam check
                for spam_domain in SPAM_DOMAINS:
                    spam_clean = normalize_domain(spam_domain)
                    if spam_clean and spam_clean in normalized_domain:
                        LOG.info(f"BLOCKED spam domain in redirect: {normalized_domain}")
                        return None, None, None
                
                # Yahoo Finance source extraction
                if "finance.yahoo.com" in actual_url:
                    original_source_url = extract_yahoo_finance_source_optimized(actual_url)
                    if original_source_url:
                        original_domain = urlparse(original_source_url).netloc.lower()
                        original_normalized_domain = normalize_domain(original_domain)
                        
                        # Spam check on original source
                        for spam_domain in SPAM_DOMAINS:
                            spam_clean = normalize_domain(spam_domain)
                            if spam_clean and spam_clean in original_normalized_domain:
                                LOG.info(f"BLOCKED spam domain in original source: {original_normalized_domain}")
                                return None, None, None
                        
                        return original_source_url, original_normalized_domain, actual_url
                        
                return actual_url, normalized_domain, None
        
        # Direct URL handling
        domain = urlparse(url).netloc.lower()
        normalized_domain = normalize_domain(domain)
        
        # Spam check
        for spam_domain in SPAM_DOMAINS:
            spam_clean = normalize_domain(spam_domain)
            if spam_clean and spam_clean in normalized_domain:
                LOG.info(f"BLOCKED spam domain: {normalized_domain}")
                return None, None, None
        
        # Yahoo Finance handling
        if "finance.yahoo.com" in url:
            original_source_url = extract_yahoo_finance_source_optimized(url)
            if original_source_url:
                original_domain = urlparse(original_source_url).netloc.lower()
                original_normalized_domain = normalize_domain(original_domain)
                
                # Spam check on original source
                for spam_domain in SPAM_DOMAINS:
                    spam_clean = normalize_domain(spam_domain)
                    if spam_clean and spam_clean in original_normalized_domain:
                        LOG.info(f"BLOCKED spam domain in Yahoo original source: {original_normalized_domain}")
                        return None, None, None
                
                return original_source_url, original_normalized_domain, url
                
        return url, normalized_domain, None
        
    except Exception as e:
        LOG.warning(f"Failed to resolve URL {url}: {e}")
        if url and "news.google.com" in url:
            return url, "google-news-unresolved", None
        else:
            fallback_domain = urlparse(url).netloc.lower() if url else None
            normalized_fallback = normalize_domain(fallback_domain) if fallback_domain else None
            return url, normalized_fallback, None

def process_google_news_with_title(url: str, title: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Process Google News URL using title to extract source domain
    """
    if not title:
        LOG.warning(f"No title provided for Google News URL: {url}")
        return url, "google-news-no-title"
    
    # Check for spam/non-Latin content first
    if contains_non_latin_script(title):
        LOG.info(f"BLOCKED non-Latin script in Google News title: {title[:50]}")
        return None, None
    
    # Extract source from title
    clean_title, extracted_source = extract_source_from_title_smart(title)
    
    if extracted_source:
        # Try to get cached domain from database first
        domain = get_cached_domain_for_source(extracted_source)
        if domain:
            LOG.info(f"Using cached domain for '{extracted_source}': {domain}")
            return url, domain
        
        # Try to resolve domain from source name
        domain = resolve_source_name_to_domain(extracted_source)
        if domain:
            # Cache this mapping for future use
            cache_source_domain_mapping(extracted_source, domain)
            LOG.info(f"Resolved Google News source: '{extracted_source}' -> '{domain}'")
            return url, domain
    
    # Fallback for unresolvable Google News
    LOG.warning(f"Could not extract source from Google News title: {title[:60]}")
    return url, "google-news-unresolved"

def extract_source_from_title_smart(title: str) -> Tuple[str, Optional[str]]:
    """
    Smart source extraction from title with regex-first approach
    """
    if not title:
        return title, None
    
    # Try simple regex patterns first (faster than AI)
    source_patterns = [
        r'\s*-\s*([^-]+)$',  # " - Source" at the end
        r'\s*\|\s*([^|]+)$',  # " | Source" at the end  
        r'\s+([a-zA-Z0-9.-]*\.(?:com|org|net|co\.uk))$'  # domain at the end
    ]
    
    for pattern in source_patterns:
        match = re.search(pattern, title)
        if match:
            potential_source = match.group(1).strip()
            clean_title = re.sub(pattern, '', title).strip()
            
            # Validate this looks like a real source
            if 2 < len(potential_source) < 50:
                # Check for spam
                source_lower = potential_source.lower()
                spam_sources = ["marketbeat", "newser", "khodrobank"]
                if any(spam in source_lower for spam in spam_sources):
                    LOG.info(f"BLOCKED spam source in title: {potential_source}")
                    return None, None
                
                LOG.info(f"Regex extracted source: '{potential_source}' from title: '{title[:60]}'")
                return clean_title, potential_source
    
    # Fallback to AI if regex fails
    if OPENAI_API_KEY:
        return extract_source_with_ai_optimized(title)
    
    return title, None
    
def extract_domain_hints_from_title(title: str) -> List[str]:
    """Extract potential domain hints from article titles"""
    hints = []
    
    # Look for common domain patterns in titles
    domain_patterns = [
        r'([a-zA-Z0-9-]+\.(?:com|org|net|co\.uk|co|io|ai))',  # Explicit domains
        r'([A-Z][a-zA-Z]+(?:News|Times|Post|Daily|Journal|Tribune|Herald))',  # News outlets
        r'((?:The\s+)?[A-Z][a-zA-Z\s]+(?:Magazine|Report|Review|Wire|Brief))',  # Publications
    ]
    
    for pattern in domain_patterns:
        matches = re.findall(pattern, title, re.IGNORECASE)
        for match in matches:
            clean_match = match.strip().lower()
            if len(clean_match) > 3 and clean_match not in ['the', 'and', 'for', 'with']:
                hints.append(normalize_domain(clean_match))
    
    return hints[:3]  # Return top 3 hints

def calculate_quality_score(
    title: str, 
    domain: str, 
    ticker: str,
    description: str = "",
    category: str = "company",
    keywords: List[str] = None
) -> float:
    """DISABLED: Always return neutral score - quality scoring needs refinement"""
    # Quality scoring temporarily disabled - need proper content analysis
    return 50.0

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
    """Get formal domain name from cache or generate with AI - OPTIMIZED to avoid redundant API calls"""
    if not domain:
        return "Unknown"
    
    clean_domain = normalize_domain(domain)
    if not clean_domain:
        return "Unknown"
    
    # Check database cache first
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT formal_name FROM domain_names WHERE domain = %s", (clean_domain,))
        result = cur.fetchone()
        if result:
            LOG.debug(f"Using cached formal name for {clean_domain}: {result['formal_name']}")
            return result["formal_name"]
    
    LOG.info(f"No cached formal name found for {clean_domain}, generating with AI...")
    
    # Generate with AI
    formal_name = get_formal_domain_name_from_ai(clean_domain)
    
    # Fallback if AI fails
    if not formal_name:
        formal_name = clean_domain.replace(".com", "").replace(".org", "").replace(".net", "").title()
        LOG.info(f"AI failed for {clean_domain}, using fallback: {formal_name}")
    
    # Store in database for future use
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO domain_names (domain, formal_name, ai_generated)
            VALUES (%s, %s, %s)
            ON CONFLICT (domain) DO UPDATE
            SET formal_name = EXCLUDED.formal_name, 
                updated_at = NOW(),
                ai_generated = EXCLUDED.ai_generated
        """, (clean_domain, formal_name, formal_name != clean_domain.title()))
    
    LOG.info(f"Cached new formal name: {clean_domain} -> {formal_name}")
    return formal_name

# Replace the title extraction functions with this AI-powered approach

# Add this new function to convert formal names to domains
def get_or_create_formal_domain_name(domain: str) -> str:
    """
    Optimized version that checks cache first, uses common mappings, then AI
    """
    if not domain:
        return "Unknown"
    
    clean_domain = normalize_domain(domain)
    if not clean_domain:
        return "Unknown"
    
    # Check database cache first
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT formal_name FROM domain_names WHERE domain = %s", (clean_domain,))
        result = cur.fetchone()
        if result:
            return result["formal_name"]
    
    # Check common mappings
    common_formal_names = {
        'reuters.com': 'Reuters',
        'bloomberg.com': 'Bloomberg',
        'wsj.com': 'Wall Street Journal',
        'ft.com': 'Financial Times',
        'barrons.com': "Barron's",
        'cnbc.com': 'CNBC',
        'marketwatch.com': 'MarketWatch',
        'finance.yahoo.com': 'Yahoo Finance',
        'businesswire.com': 'Business Wire',
        'zacks.com': 'Zacks Investment Research',
        'seekingalpha.com': 'Seeking Alpha',
        'tipranks.com': 'TipRanks',
        'theglobeandmail.com': 'The Globe and Mail',
        'msn.com': 'MSN',
        'defensenews.com': 'Defense News',
        'stocktitan.com': 'Stock Titan'
    }
    
    if clean_domain in common_formal_names:
        formal_name = common_formal_names[clean_domain]
        # Cache it
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO domain_names (domain, formal_name, ai_generated)
                VALUES (%s, %s, FALSE)
                ON CONFLICT (domain) DO NOTHING
            """, (clean_domain, formal_name))
        LOG.info(f"Used common formal name: {clean_domain} -> {formal_name}")
        return formal_name
    
    # Use AI as last resort
    if OPENAI_API_KEY:
        formal_name = get_formal_domain_name_from_ai(clean_domain)
        if formal_name:
            return formal_name
    
    # Fallback
    fallback_name = clean_domain.replace(".com", "").replace(".org", "").replace(".net", "").title()
    
    # Cache the fallback
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO domain_names (domain, formal_name, ai_generated)
            VALUES (%s, %s, FALSE)
            ON CONFLICT (domain) DO NOTHING
        """, (clean_domain, fallback_name))
    
    return fallback_name

def extract_source_with_ai_optimized(title: str) -> Tuple[str, Optional[str]]:
    """
    Optimized AI source extraction with spam filtering
    """
    if not title or not OPENAI_API_KEY:
        return title, None
    
    # Check for non-Latin script first
    if contains_non_latin_script(title):
        LOG.info(f"BLOCKED non-Latin script in title: {title[:50]}")
        return None, None
    
    prompt = f"""Extract the source publication from this news title:

Title: "{title}"

Rules:
- Look for source at the end after dash or space
- Return just the publication name (like "Reuters", "Bloomberg", "Barron's")
- If no clear source found, return null

JSON format:
{{"source": "publication name or null", "clean_title": "title without source"}}"""
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "Extract news publication sources from titles. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 100,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and result['choices']:
                content = result['choices'][0]['message']['content']
                
                try:
                    parsed = json.loads(content)
                    source = parsed.get("source")
                    clean_title = parsed.get("clean_title", title)
                    
                    if source == "null" or source == "None":
                        source = None
                    
                    # Spam filtering for extracted source
                    if source:
                        source_lower = source.lower()
                        spam_sources = ["marketbeat", "newser", "khodrobank"]
                        for spam in spam_sources:
                            if spam in source_lower:
                                LOG.info(f"BLOCKED spam source from AI: {source}")
                                return None, None
                    
                    LOG.info(f"AI extracted source: '{source}' from title: '{title[:60]}'")
                    return clean_title, source
                    
                except json.JSONDecodeError as e:
                    LOG.warning(f"AI response JSON error: {e}")
                    
    except Exception as e:
        LOG.warning(f"AI source extraction failed: {e}")
    
    return title, None

def get_cached_domain_for_source(source_name: str) -> Optional[str]:
    """
    Check database for existing domain mapping for a source name
    """
    with db() as conn, conn.cursor() as cur:
        # Check if we have this source name mapped to a domain
        cur.execute("""
            SELECT domain FROM domain_names 
            WHERE LOWER(formal_name) = LOWER(%s)
            LIMIT 1
        """, (source_name,))
        
        result = cur.fetchone()
        if result:
            return result['domain']
        
        # Also check if the source name itself is a domain we know
        normalized_source = normalize_domain(source_name)
        if normalized_source:
            cur.execute("""
                SELECT formal_name FROM domain_names 
                WHERE domain = %s
                LIMIT 1
            """, (normalized_source,))
            
            result = cur.fetchone()
            if result:
                return normalized_source
    
    return None

def resolve_source_name_to_domain(source_name: str) -> Optional[str]:
    """
    Convert source name to domain with minimal AI usage
    """
    # First check if it already looks like a domain
    if '.' in source_name and any(tld in source_name.lower() for tld in ['.com', '.org', '.net', '.co.uk']):
        return normalize_domain(source_name)
    
    # Check common mappings without AI
    common_mappings = {
        'reuters': 'reuters.com',
        'bloomberg': 'bloomberg.com', 
        'wall street journal': 'wsj.com',
        'wsj': 'wsj.com',
        'financial times': 'ft.com',
        'barrons': 'barrons.com',
        "barron's": 'barrons.com',
        'cnbc': 'cnbc.com',
        'marketwatch': 'marketwatch.com',
        'yahoo finance': 'finance.yahoo.com',
        'business wire': 'businesswire.com',
        'zacks': 'zacks.com',
        'zacks investment research': 'zacks.com',
        'seeking alpha': 'seekingalpha.com',
        'tipranks': 'tipranks.com',
        'the globe and mail': 'theglobeandmail.com',
        'globe and mail': 'theglobeandmail.com',
        'msn': 'msn.com',
        'microsoft news': 'msn.com',
        'msn news': 'msn.com',
        'defense news': 'defensenews.com',
        'stock titan': 'stocktitan.com'
    }
    
    source_lower = source_name.lower().strip()
    if source_lower in common_mappings:
        domain = common_mappings[source_lower]
        LOG.info(f"Used common mapping: '{source_name}' -> '{domain}'")
        return domain
    
    # Use AI as last resort
    if OPENAI_API_KEY:
        return get_domain_from_formal_name_ai_optimized(source_name)
    
    # Final fallback - create a reasonable domain-like string
    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', source_name).lower().replace(' ', '')
    if len(clean_name) > 3:
        return f"{clean_name[:20]}.com"
    
    return None

def get_domain_from_formal_name_ai_optimized(formal_name: str) -> Optional[str]:
    """
    Use AI to convert formal publication name to domain (optimized)
    """
    if not formal_name or not OPENAI_API_KEY:
        return None
    
    prompt = f"""Convert this publication name to its website domain:

Publication: "{formal_name}"

Provide ONLY the domain (like "example.com") without http/https or www.

Examples:
- "Wall Street Journal" → "wsj.com"
- "Barron's" → "barrons.com"
- "TipRanks" → "tipranks.com"

Domain for "{formal_name}":|"""
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "Convert publication names to domains. Provide only the domain name."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 30
        }
        
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and result['choices']:
                domain = result['choices'][0]['message']['content'].strip()
                domain = re.sub(r'^["\']|["\']$', '', domain)
                domain = domain.lower().strip()
                
                # Basic validation
                if '.' in domain and 4 < len(domain) < 50:
                    LOG.info(f"AI converted publication to domain: '{formal_name}' -> '{domain}'")
                    return domain
                    
    except Exception as e:
        LOG.error(f"Failed to convert formal name '{formal_name}' to domain: {e}")
    
    return None

def cache_source_domain_mapping(source_name: str, domain: str):
    """
    Cache the source name to domain mapping in database
    """
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO domain_names (domain, formal_name, ai_generated)
            VALUES (%s, %s, %s)
            ON CONFLICT (domain) DO UPDATE
            SET formal_name = EXCLUDED.formal_name,
                updated_at = NOW()
        """, (normalize_domain(domain), source_name, False))

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
        
def format_timestamp_est(dt: datetime) -> str:
    """Format datetime to EST with format like 'Sep 12, 2:08pm EST'"""
    if not dt:
        return "N/A"
    
    # Ensure we have a timezone-aware datetime
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    # Convert to Eastern Time
    eastern = pytz.timezone('US/Eastern')
    est_time = dt.astimezone(eastern)
    
    # Format as requested: "Sep 12, 2:08pm EST"
    # Handle AM/PM formatting
    time_part = est_time.strftime("%I:%M%p").lower().lstrip('0')
    date_part = est_time.strftime("%b %d")  # Changed from %B to %b for abbreviated month
    
    # Always use "EST" instead of dynamic timezone abbreviation
    return f"{date_part}, {time_part} EST"

# ENHANCED: Update the ingest_feed function
def ingest_feed(feed: Dict, category: str = "company", keywords: List[str] = None) -> Dict[str, int]:
    """
    Enhanced feed processing with optimized Google News and Yahoo Finance handling
    """
    stats = {"processed": 0, "inserted": 0, "duplicates": 0, "low_quality": 0, 
             "blocked_spam": 0, "blocked_non_latin": 0, "yahoo_sources_found": 0, 
             "google_domains_resolved": 0, "google_unresolved": 0}
    
    try:
        LOG.info(f"Processing feed [{category}]: {feed['name']}")
        parsed = feedparser.parse(feed["url"])
        LOG.info(f"Feed parsing result: {len(parsed.entries)} entries found")
        
        for entry in parsed.entries:
            stats["processed"] += 1
            
            original_url = getattr(entry, "link", None)
            if not original_url:
                continue
            
            title = getattr(entry, "title", "") or "No Title"
            description = getattr(entry, "summary", "")[:500] if hasattr(entry, "summary") else ""
            
            # Check for non-Latin script in title
            if contains_non_latin_script(title):
                stats["blocked_non_latin"] += 1
                LOG.info(f"BLOCKED non-Latin script in title: {title[:50]}")
                continue
            
            # Spam keyword filtering
            spam_keywords = ["marketbeat", "newser", "khodrobank"]
            if any(spam in title.lower() for spam in spam_keywords):
                stats["blocked_spam"] += 1
                LOG.info(f"BLOCKED spam in title: {title[:50]}")
                continue
            
            # URL resolution
            resolved_result = resolve_google_news_url(original_url)
            if resolved_result[0] is None:  # Spam detected at URL level
                stats["blocked_spam"] += 1
                continue
                
            resolved_url, domain, yahoo_source_url = resolved_result
            if not resolved_url or not domain:
                continue
            
            # Handle Google News special processing
            if domain == "google-news-needs-title-processing":
                title_result = process_google_news_with_title(resolved_url, title)
                if title_result[0] is None:  # Spam detected in title processing
                    stats["blocked_spam"] += 1
                    continue
                
                resolved_url, domain = title_result
                
                # Track resolution success
                if domain and "google-news" not in domain:
                    stats["google_domains_resolved"] += 1
                    # Extract and clean the title
                    clean_title, _ = extract_source_from_title_smart(title)
                    if clean_title:
                        title = clean_title
                else:
                    stats["google_unresolved"] += 1
                    if domain == "google-news-unresolved":
                        LOG.warning(f"Skipping unresolved Google News: {title[:60]}")
                        continue
            
            # Track Yahoo source extractions
            if yahoo_source_url:
                stats["yahoo_sources_found"] += 1
                LOG.info(f"Yahoo Finance article resolved to original: {resolved_url}")
            
            url_hash = get_url_hash(resolved_url)
            
            # Calculate quality score (currently disabled, returns 50.0)
            quality_score = calculate_quality_score(
                title, domain, feed["ticker"], description, category, keywords
            )
            
            if quality_score < 1:
                stats["low_quality"] += 1
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
                    
                    # Use the resolved domain
                    normalized_domain = normalize_domain(domain)
                    
                    # Determine related ticker and search keyword for better tracking
                    related_ticker = None
                    search_keyword = feed.get("search_keyword", "")
                    
                    if category == "competitor":
                        related_ticker = feed.get("competitor_ticker")
                    
                    cur.execute("""
                        INSERT INTO found_url (
                            url, resolved_url, url_hash, title, description,
                            feed_id, ticker, domain, quality_score, published_at,
                            category, related_ticker, original_source_url, search_keyword
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        original_url, resolved_url, url_hash, title, description,
                        feed["id"], feed["ticker"], normalized_domain, quality_score, published_at,
                        category, related_ticker, yahoo_source_url, search_keyword
                    ))
                    
                    if cur.fetchone():
                        stats["inserted"] += 1
                        
                        # Update domain statistics
                        upsert_domain_stats(normalized_domain, quality_score, category)
                        
                        # Also track original source domain if from Yahoo
                        if yahoo_source_url:
                            original_domain = normalize_domain(urlparse(yahoo_source_url).netloc.lower())
                            if original_domain:
                                upsert_domain_stats(original_domain, quality_score, f"{category}_original")
                        
                        # Get display name for logging
                        display_name = get_or_create_formal_domain_name(normalized_domain)
                        
                        LOG.info(f"Inserted [{category}] from {display_name}: {title[:60]}...")
                        
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
# Enhanced CSS styles to be added to the build_digest_html function
def build_digest_html(articles_by_ticker: Dict[str, Dict[str, List[Dict]]], period_days: int) -> str:
    """Build HTML email digest with enhanced styling and improved timestamp formatting"""
    # Get ticker metadata for display
    ticker_metadata = {}
    for ticker in articles_by_ticker.keys():
        config = get_ticker_config(ticker)
        if config:
            ticker_metadata[ticker] = {
                "industry_keywords": config.get("industry_keywords", []),
                "competitors": config.get("competitors", [])
            }
    
    # FIX #5: Use the new timestamp formatting
    current_time_est = format_timestamp_est(datetime.now(timezone.utc))
    
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
        ".competitor-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fdeaea; color: #c53030; border: 1px solid #feb2b2; max-width: 200px; white-space: nowrap; overflow: visible; }",
        ".industry-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fef5e7; color: #b7791f; border: 1px solid #f6e05e; max-width: 200px; white-space: nowrap; overflow: visible; }",
        ".keywords { background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 11px; }",
        "a { color: #2980b9; text-decoration: none; }",
        "a:hover { text-decoration: underline; }",
        ".summary { margin-top: 20px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }",
        ".ticker-section { margin-bottom: 40px; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "</style></head><body>",
        f"<h1>Stock Intelligence Report</h1>",
        f"<div class='summary'>",
        f"<strong>Report Period:</strong> Last {period_days} days<br>",
        f"<strong>Generated:</strong> {current_time_est}<br>",  # FIX #5: Better timestamp
        f"<strong>Tickers Covered:</strong> {', '.join(articles_by_ticker.keys())}",
        "</div>"
    ]
    
    for ticker, categories in articles_by_ticker.items():
        total_articles = sum(len(articles) for articles in categories.values())
        
        html.append(f"<div class='ticker-section'>")
        html.append(f"<h2>{ticker} - {total_articles} Total Articles</h2>")
        
        # Add keyword information with improved styling
        if ticker in ticker_metadata:
            metadata = ticker_metadata[ticker]
            html.append("<div class='keywords'>")
            html.append(f"<strong>🤖 AI-Powered Monitoring Keywords:</strong><br>")
            if metadata.get("industry_keywords"):
                # Display industry keywords as styled badges
                industry_badges = [f'<span class="industry-badge">🏭 {kw}</span>' for kw in metadata['industry_keywords']]
                html.append(f"<strong>Industry:</strong> {' '.join(industry_badges)}<br>")
            if metadata.get("competitors"):
                # Display competitors as styled badges
                competitor_badges = [f'<span class="competitor-badge">🏢 {comp}</span>' for comp in metadata['competitors']]
                html.append(f"<strong>Competitors:</strong> {' '.join(competitor_badges)}")
            html.append("</div>")
        
        # Company News Section
        if "company" in categories and categories["company"]:
            html.append(f"<h3>Company News ({len(categories['company'])} articles)</h3>")
            for article in categories["company"][:50]:
                html.append(_format_article_html(article, "company"))
        
        # Industry News Section
        if "industry" in categories and categories["industry"]:
            html.append(f"<h3>Industry & Market News ({len(categories['industry'])} articles)</h3>")
            for article in categories["industry"][:50]:
                html.append(_format_article_html(article, "industry"))
        
        # Competitor News Section
        if "competitor" in categories and categories["competitor"]:
            html.append(f"<h3>Competitor Intelligence ({len(categories['competitor'])} articles)</h3>")
            for article in categories["competitor"][:50]:
                html.append(_format_article_html(article, "competitor"))
        
        html.append("</div>")
    
    html.append("""
        <div style='margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; font-size: 11px; color: #6c757d;'>
            <strong>About This Report:</strong><br>
            • Company News: Direct mentions and updates about the ticker<br>
            • Industry News: Sector trends and market dynamics (🏭 industry keywords)<br>
            • Competitor Intelligence: Updates on key competitors (🏢 competitor names)<br>
            • Quality scores: Higher scores indicate more reputable sources and relevant content (15-100 scale)<br>
            • Keywords generated by AI analysis for comprehensive coverage<br>
            • Spam domains automatically filtered out for quality<br>
            • Source badges show the original publisher of each article
        </div>
        </body></html>
    """)
    
    return "".join(html)

# FIX #2: Updated _format_article_html function with better badge handling
def _format_article_html(article: Dict, category: str) -> str:
    """Format article HTML with better timestamp formatting and Google News title trimming"""
    # Format timestamp for individual articles
    if article["published_at"]:
        # For individual articles, use a shorter format like "Sep 13, 2:23pm EST"
        eastern = pytz.timezone('US/Eastern')
        pub_dt = article["published_at"]
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
        
        est_time = pub_dt.astimezone(eastern)
        pub_date = est_time.strftime("%b %d, %I:%M%p").lower().replace(' 0', ' ').replace('0:', ':')
        tz_abbrev = est_time.strftime("%Z")
        pub_date = f"{pub_date} {tz_abbrev}"
    else:
        pub_date = "N/A"
    
    original_title = article["title"] or "No Title"
    resolved_domain = article["domain"] or "unknown"
    
    # Determine source and clean title based on domain type
    if "news.google.com" in resolved_domain or resolved_domain == "google-news-unresolved":
        # Use AI to analyze Google News titles and extract source from title
        title_result = extract_source_with_ai(original_title)
        
        # Check if spam was detected during title analysis
        if title_result[0] is None:  # Spam detected
            return ""  # Return empty string to skip this article
        
        title, extracted_source = title_result
        
        if extracted_source:
            # Get formal name for the extracted source
            display_source = get_or_create_formal_domain_name(extracted_source)
        else:
            display_source = "Google News"  # Fallback if no source extracted
    else:
        # Non-Google articles use the title as-is and resolve domain normally
        title = original_title
        display_source = get_or_create_formal_domain_name(resolved_domain)
    
    # Additional title cleanup - remove any remaining ticker artifacts
    title = re.sub(r'\s*\$[A-Z]+\s*-?\s*', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    
    # Determine the actual link URL (prefer resolved/original over raw feed URL)
    link_url = article["resolved_url"] or article.get("original_source_url") or article["url"]
    
    # Quality score styling
    score = article["quality_score"]
    score_class = "high-score" if score >= 70 else "med-score" if score >= 40 else "low-score"
    
    # Build metadata badges for category-specific information
    metadata_badges = []
    
    # Add category-specific badges with full text display
    if category == "competitor" and article.get('search_keyword'):
        competitor_name = article['search_keyword']
        metadata_badges.append(f'<span class="competitor-badge">🏢 {competitor_name}</span>')
    elif category == "industry" and article.get('search_keyword'):
        industry_keyword = article['search_keyword']
        metadata_badges.append(f'<span class="industry-badge">🏭 {industry_keyword}</span>')
    
    # Join all metadata badges
    enhanced_metadata = "".join(metadata_badges)
    
    return f"""
    <div class='article {category}'>
        <span class='source-badge'>{display_source}</span>
        {enhanced_metadata}
        <span class='score {score_class}'>Score: {score:.0f}</span>
        <a href='{link_url}' target='_blank'>{title}</a>
        <span class='meta'> | {pub_date}</span>
    </div>
    """

def fetch_digest_articles(hours: int = 24, tickers: List[str] = None) -> Dict[str, Dict[str, List[Dict]]]:
    """Fetch categorized articles for digest with enhanced search metadata"""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    with db() as conn, conn.cursor() as cur:
        # Build query based on tickers - ENHANCED to include search metadata
        if tickers:
            cur.execute("""
                SELECT 
                    f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.quality_score, f.published_at,
                    f.found_at, f.category, f.related_ticker, f.original_source_url,
                    f.search_keyword
                FROM found_url f
                WHERE f.found_at >= %s
                    AND f.quality_score >= 15
                    AND NOT f.sent_in_digest
                    AND f.ticker = ANY(%s)
                ORDER BY f.ticker, f.category, COALESCE(f.published_at, f.found_at) DESC, f.found_at DESC
            """, (cutoff, tickers))
        else:
            cur.execute("""
                SELECT 
                    f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.quality_score, f.published_at,
                    f.found_at, f.category, f.related_ticker, f.original_source_url,
                    f.search_keyword
                FROM found_url f
                WHERE f.found_at >= %s
                    AND f.quality_score >= 15
                    AND NOT f.sent_in_digest
                ORDER BY f.ticker, f.category, COALESCE(f.published_at, f.found_at) DESC, f.found_at DESC
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
    """Initialize database and generate AI-powered feeds for specified tickers - ENHANCED"""
    require_admin(request)
    ensure_schema()
    
    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OpenAI API key not configured"}
    
    results = []
    for ticker in body.tickers:
        LOG.info(f"Initializing ticker: {ticker}")
        
        # Get or generate metadata with AI
        keywords = get_or_create_ticker_metadata(ticker, force_refresh=body.force_refresh)
        
        # Build feed URLs for all categories (now includes Yahoo feeds)
        feeds = build_feed_urls(ticker, keywords)
        
        for feed_config in feeds:
            feed_id = upsert_feed(
                url=feed_config["url"],
                name=feed_config["name"],
                ticker=ticker,
                category=feed_config.get("category", "company"),
                retain_days=DEFAULT_RETAIN_DAYS,
                search_keyword=feed_config.get("search_keyword"),
                competitor_ticker=feed_config.get("competitor_ticker")
            )
            results.append({
                "ticker": ticker,
                "feed": feed_config["name"],
                "category": feed_config.get("category", "company"),
                "search_keyword": feed_config.get("search_keyword"),
                "competitor_ticker": feed_config.get("competitor_ticker"),
                "id": feed_id
            })
        
        LOG.info(f"Created {len(feeds)} feeds for {ticker} (including Yahoo feeds)")
    
    return {
        "status": "initialized",
        "tickers": body.tickers,
        "feeds": results,
        "message": f"Generated {len(results)} feeds using AI-powered keyword analysis (Google + Yahoo sources)"
    }

@APP.post("/cron/ingest")
def cron_ingest(
    request: Request,
    minutes: int = Query(default=1440, description="Time window in minutes"),
    tickers: List[str] = Query(default=None, description="Specific tickers to ingest")
):
    """Ingest articles from feeds for specified tickers - ENHANCED with metadata"""
    require_admin(request)
    ensure_schema()
    
    # Get feeds for specified tickers with search metadata
    with db() as conn, conn.cursor() as cur:
        if tickers:
            cur.execute("""
                SELECT id, url, name, ticker, retain_days, search_keyword, competitor_ticker
                FROM source_feed
                WHERE active = TRUE AND ticker = ANY(%s)
                ORDER BY ticker, id
            """, (tickers,))
        else:
            cur.execute("""
                SELECT id, url, name, ticker, retain_days, search_keyword, competitor_ticker
                FROM source_feed
                WHERE active = TRUE
                ORDER BY ticker, id
            """)
        feeds = list(cur.fetchall())
    
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
            if "Industry:" in feed["name"] or "Yahoo Industry:" in feed["name"]:
                category = "industry"
            elif "Competitor:" in feed["name"] or "Yahoo Competitor:" in feed["name"]:
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
    """Generate and send email digest - ENHANCED with search metadata"""
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
                    f.found_at, f.category, f.related_ticker, f.original_source_url,
                    f.search_keyword
                FROM found_url f
                WHERE f.found_at >= %s
                    AND f.quality_score >= 15
                    AND NOT f.sent_in_digest
                    AND f.ticker = ANY(%s)
                ORDER BY f.ticker, f.category, COALESCE(f.published_at, f.found_at) DESC, f.found_at DESC
            """, (cutoff, tickers))
        else:
            cur.execute("""
                SELECT 
                    f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.quality_score, f.published_at,
                    f.found_at, f.category, f.related_ticker, f.original_source_url,
                    f.search_keyword
                FROM found_url f
                WHERE f.found_at >= %s
                    AND f.quality_score >= 15
                    AND NOT f.sent_in_digest
                ORDER BY f.ticker, f.category, COALESCE(f.published_at, f.found_at) DESC, f.found_at DESC
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
                ORDER BY f.ticker, f.category, COALESCE(f.published_at, f.found_at) DESC, f.found_at DESC
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
                ORDER BY f.ticker, f.category, COALESCE(f.published_at, f.found_at) DESC, f.found_at DESC
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

@APP.get("/admin/feed-metadata")
def get_feed_metadata(request: Request, ticker: str = Query(None)):
    """Get feed metadata including search keywords and competitor tickers"""
    require_admin(request)
    
    with db() as conn, conn.cursor() as cur:
        if ticker:
            cur.execute("""
                SELECT id, url, name, ticker, search_keyword, competitor_ticker,
                       active, created_at
                FROM source_feed
                WHERE ticker = %s
                ORDER BY name
            """, (ticker,))
        else:
            cur.execute("""
                SELECT id, url, name, ticker, search_keyword, competitor_ticker,
                       active, created_at
                FROM source_feed
                ORDER BY ticker, name
            """)
        feeds = list(cur.fetchall())
    
    return {
        "feeds": feeds,
        "total": len(feeds),
        "ticker_filter": ticker
    }

@APP.get("/admin/search-analytics")
def get_search_analytics(request: Request, days: int = Query(default=7)):
    """Get analytics on search keywords and their performance"""
    require_admin(request)
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    
    with db() as conn, conn.cursor() as cur:
        # Analytics by search keyword
        cur.execute("""
            SELECT 
                search_keyword,
                category,
                COUNT(*) as article_count,
                AVG(quality_score) as avg_quality,
                COUNT(DISTINCT domain) as unique_sources
            FROM found_url
            WHERE found_at >= %s 
                AND search_keyword IS NOT NULL
            GROUP BY search_keyword, category
            ORDER BY article_count DESC
        """, (cutoff,))
        keyword_stats = list(cur.fetchall())
        
        # Analytics by related ticker (competitors)
        cur.execute("""
            SELECT 
                related_ticker,
                ticker as main_ticker,
                COUNT(*) as article_count,
                AVG(quality_score) as avg_quality
            FROM found_url
            WHERE found_at >= %s 
                AND related_ticker IS NOT NULL
            GROUP BY related_ticker, ticker
            ORDER BY article_count DESC
        """, (cutoff,))
        competitor_stats = list(cur.fetchall())
        
        # Source distribution
        cur.execute("""
            SELECT 
                CASE 
                    WHEN domain LIKE '%yahoo%' THEN 'Yahoo Finance'
                    WHEN domain LIKE '%google%' THEN 'Google News'
                    ELSE 'Other'
                END as source_type,
                category,
                COUNT(*) as article_count
            FROM found_url
            WHERE found_at >= %s
            GROUP BY source_type, category
            ORDER BY article_count DESC
        """, (cutoff,))
        source_distribution = list(cur.fetchall())
    
    return {
        "period_days": days,
        "keyword_performance": keyword_stats,
        "competitor_tracking": competitor_stats,
        "source_distribution": source_distribution
    }  # Make sure this ends cleanly
            
@APP.get("/admin/domain-stats")
def get_domain_statistics(request: Request, limit: int = Query(default=50)):
    """Get domain statistics for quality scoring improvements"""
    require_admin(request)
    
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT domain, total_articles, avg_quality_score,
                   first_seen, last_seen, category_counts,
                   CASE WHEN domain = ANY(%s) THEN 'quality'
                        WHEN domain = ANY(%s) THEN 'spam'
                        ELSE 'neutral' END as classification
            FROM domain_stats 
            ORDER BY total_articles DESC
            LIMIT %s
        """, (list(QUALITY_DOMAINS), list(SPAM_DOMAINS), limit))
        
        stats = list(cur.fetchall())
        
        # Add some aggregate statistics
        cur.execute("""
            SELECT COUNT(*) as total_domains,
                   AVG(total_articles) as avg_articles_per_domain,
                   AVG(avg_quality_score) as overall_avg_quality
            FROM domain_stats
        """)
        
        summary = dict(cur.fetchone())
    
    return {
        "summary": summary,
        "domains": stats,
        "quality_domains_tracked": len([d for d in stats if d['classification'] == 'quality']),
        "spam_domains_tracked": len([d for d in stats if d['classification'] == 'spam'])
    }

@APP.get("/admin/resolved-domains")
def get_resolved_domains(request: Request, ticker: str = Query(None)):
    """Get resolved domains with formal names for scoring"""
    require_admin(request)
    
    with db() as conn, conn.cursor() as cur:
        base_query = """
            SELECT 
                f.domain as raw_domain,
                dn.formal_name,
                f.category,
                COUNT(*) as article_count,
                AVG(f.quality_score) as avg_quality,
                MAX(f.found_at) as last_seen
            FROM found_url f
            LEFT JOIN domain_names dn ON dn.domain = f.domain
            WHERE f.domain IS NOT NULL 
              AND f.domain != 'google-news-unresolved'
        """
        
        params = []
        if ticker:
            base_query += " AND f.ticker = %s"
            params.append(ticker)
        
        base_query += """
            GROUP BY f.domain, dn.formal_name, f.category
            ORDER BY article_count DESC
        """
        
        cur.execute(base_query, params)
        results = list(cur.fetchall())
        
        # Process results to ensure all domains have formal names
        enhanced_results = []
        for row in results:
            if not row['formal_name']:
                # Generate formal name if missing
                formal_name = get_or_create_formal_domain_name(row['raw_domain'])
                row = dict(row)
                row['formal_name'] = formal_name
            
            enhanced_results.append(row)
    
    return {
        "domains": enhanced_results,
        "ticker_filter": ticker,
        "total_domains": len(set(r['raw_domain'] for r in enhanced_results))
    }

@APP.post("/admin/cache-domain-names")
def cache_domain_names_endpoint(request: Request, limit: int = Query(default=50)):
    """Bulk cache formal names for domains that don't have them yet"""
    require_admin(request)
    
    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OpenAI API key not configured"}
    
    stats = cache_missing_domain_names(limit)
    
    return {
        "status": "completed",
        "domains_processed": stats["processed"],
        "domains_cached": stats["cached"],
        "domains_failed": stats["failed"],
        "message": f"Cached formal names for {stats['cached']} out of {stats['processed']} domains"
    }

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
