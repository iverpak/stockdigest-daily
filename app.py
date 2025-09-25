import os
import sys
import time
import logging
import hashlib
import re
import pytz
import json
import openai
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple, Set
from contextlib import contextmanager
from urllib.parse import urlparse, parse_qs, unquote, quote

import newspaper
from newspaper import Article
import random
from urllib.robotparser import RobotFileParser

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

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from collections import defaultdict

from playwright.sync_api import sync_playwright
import asyncio

# Global session for OpenAI API calls with retries
_openai_session = None

def get_openai_session():
    """Get a requests session with retry logic for OpenAI API calls"""
    global _openai_session
    if _openai_session is None:
        _openai_session = requests.Session()
        retry_strategy = Retry(
            total=3,                    # up to 3 retries
            backoff_factor=0.8,         # 0.8s, 1.6s, 3.2s delays
            status_forcelist=(429, 500, 502, 503, 504),  # retry on these HTTP codes
            allowed_methods=frozenset(["POST"])
        )
        _openai_session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    return _openai_session

def extract_text_from_responses(result: dict) -> str:
    """
    Robustly extract assistant text from Responses API.
    Covers:
      - top-level `output_text`
      - `output[*].content[*].text` when type in {"output_text", "text"}
    Returns "" if none.
    """
    try:
        # 1) Convenience field (some models return this directly)
        if isinstance(result.get("output_text"), str) and result["output_text"].strip():
            return result["output_text"].strip()

        # 2) Structured content (standard Responses format)
        for block in result.get("output", []) or []:
            for item in block.get("content", []) or []:
                t = item.get("type")
                if t in ("output_text", "text"):
                    txt = item.get("text")
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()
        return ""
    except Exception:
        return ""

def parse_json_with_fallback(text: str, ticker: str = "") -> dict:
    """Parse JSON with fallback extraction for malformed responses"""
    if not text:
        return {}
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract first JSON object from text
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        
        # Return minimum defaults
        LOG.warning(f"Failed to parse JSON for {ticker}, using defaults")
        return {
            "ticker": ticker,
            "company_name": ticker,
            "industry_keywords": [],
            "competitors": [],
            "sector": "",
            "industry": "",
            "sub_industry": "",
            "sector_profile": {"core_inputs": [], "core_channels": [], "core_geos": [], "benchmarks": []},
            "aliases_brands_assets": {"aliases": [], "brands": [], "assets": []}
        }

def clean_null_bytes(text: str) -> str:
    """Remove NULL bytes that cause PostgreSQL errors"""
    if not text:
        return text
    return text.replace('\x00', '').replace('\0', '')

def normalize_priority_to_int(priority):
    """Normalize priority to consistent integer format (1=High, 2=Medium, 3=Low)"""
    if isinstance(priority, int):
        return max(1, min(3, priority))
    elif isinstance(priority, str):
        priority_upper = priority.upper()
        if priority_upper in ["HIGH", "H", "1"]:
            return 1
        elif priority_upper in ["MEDIUM", "MED", "M", "2"]:
            return 2
        elif priority_upper in ["LOW", "L", "3"]:
            return 3
    return 2  # Default to Medium

def normalize_priority_to_display(priority):
    """Convert integer priority to display format"""
    if isinstance(priority, int):
        if priority == 1:
            return "High"
        elif priority == 2:
            return "Medium"
        elif priority == 3:
            return "Low"
    elif isinstance(priority, str):
        priority_upper = priority.upper()
        if priority_upper in ["HIGH", "H"]:
            return "High"
        elif priority_upper in ["MEDIUM", "MED", "M"]:
            return "Medium"
        elif priority_upper in ["LOW", "L"]:
            return "Low"
    return "Medium"

def _normalize_host(url_or_domain: str) -> str:
    """
    Returns a lowercase hostname (no scheme/port/path). 
    Accepts either a full URL or a bare domain.
    """
    s = (url_or_domain or "").strip().lower()
    if not s:
        return ""
    if "://" not in s:
        # treat as bare domain
        host = s.split("/")[0]
    else:
        host = urlparse(s).hostname or ""
    # strip common prefixes so matching is stable
    for prefix in ("www.", "m."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host

def _domain_matches(host: str, needle: str) -> bool:
    """
    True if host == needle OR host ends with '.' + needle
    This safely matches subdomains (e.g., foo.bar.com vs bar.com),
    and also supports specific subdomains like 'video.media.yql.yahoo.com'.
    """
    if not host or not needle:
        return False
    return host == needle or host.endswith("." + needle)

def needs_antibot(url_or_domain: str) -> bool:
    """
    Should we enable Scrapfly anti-bot + JS rendering for this target?
    """
    host = _normalize_host(url_or_domain)
    if not host:
        return False
    for dom in PROBLEMATIC_SCRAPE_DOMAINS:
        if _domain_matches(host, dom):
            return True
    return False

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

# OpenAI Configuration -
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/responses")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
SCRAPINGBEE_API_KEY = os.getenv("SCRAPINGBEE_API_KEY")
SCRAPFLY_API_KEY = os.getenv("SCRAPFLY_API_KEY")

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
    "khodrobank.com", "www.khodrobank.com", "khodrobank",
    "defenseworld.net", "www.defenseworld.net", "defenseworld",
    "defenseworld.com", "www.defenseworld.com", "defenseworld",
    "defense-world.net", "www.defense-world.net", "defense-world",
    "defensenews.com", "www.defensenews.com", "defensenews",
    "facebook.com", "www.facebook.com", "facebook",
}

QUALITY_DOMAINS = {
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "barrons.com", "cnbc.com", "marketwatch.com",
    "theglobeandmail.com",
    "apnews.com",
}

# Domains known to have consistent scraping failures or poor content quality
PROBLEMATIC_SCRAPE_DOMAINS = {
    # finance & article sites w/ bot protection or heavy JS
    "defenseworld.net", "defense-world.net", "defensenews.com",
    # Sites to avoid scraping (not spam, just problematic)
    "zacks.com", "insidermonkey.com", "fool.com",
}

# Known paywall domains to skip during content scraping
PAYWALL_DOMAINS = {
    # Hard paywalls only
    "wsj.com", "www.wsj.com",
    "ft.com", "www.ft.com",
    "economist.com", "www.economist.com",
    "nytimes.com", "www.nytimes.com",
    "seekingalpha.com", "www.seekingalpha.com",
    "washingtonpost.com", "www.washingtonpost.com",
    "barrons.com", "www.barrons.com",
    
    # Academic/research paywalls
    "sciencedirect.com", "www.sciencedirect.com",
    
    # Specific financial paywalls that consistently block
    "thefly.com", "www.thefly.com",
    "accessnewswire.com", "www.accessnewswire.com",
}

DOMAIN_TIERS = {
    "reuters.com": 1.0, 
    "wsj.com": 1.0, 
    "ft.com": 1.0,
    "bloomberg.com": 1.0,
    "sec.gov": 1.0,
    "apnews.com": 0.9,
    "nasdaq.com": 0.8,
    "techcrunch.com": 0.7, 
    "cnbc.com": 0.7, 
    "marketwatch.com": 0.7,
    "barrons.com": 0.7,
    "investors.com": 0.6,
    "finance.yahoo.com": 0.4, 
    "yahoo.com": 0.4, 
    "news.yahoo.com": 0.4,
    "msn.com": 0.4,
    "seekingalpha.com": 0.4, 
    "fool.com": 0.4, 
    "zacks.com": 0.4,
    "benzinga.com": 0.4,
    "tipranks.com": 0.4, 
    "simplywall.st": 0.4, 
    "investing.com": 0.4,
    "insidermonkey.com": 0.4,
    "businesswire.com": 0.3,
    "streetinsider.com": 0.3,
    "thefly.com": 0.3,
    "globenewswire.com": 0.2, 
    "prnewswire.com": 0.2, 
    "openpr.com": 0.2, 
    "financialcontent.com": 0.2,
    "accesswire.com": 0.2,
    "defensenews.com": 0.2,
    "defense-world.net": 0.1,
    "defenseworld.net": 0.1,
}

# Source hints for upgrading aggregator content
SOURCE_TIER_HINTS = [
    (r"\b(reuters)\b", 1.0),
    (r"\b(bloomberg)\b", 1.0),
    (r"\b(wall street journal|wsj)\b", 1.0),
    (r"\b(financial times|ft)\b", 1.0),
    (r"\b(associated press|ap)\b", 0.9),
    (r"\b(cnbc)\b", 0.7),
    (r"\b(marketwatch)\b", 0.7),
    (r"\b(barron's|barrons)\b", 0.7),
    (r"\b(motley fool|the motley fool)\b", 0.4),
    (r"\b(seeking alpha)\b", 0.4),
    (r"\b(zacks)\b", 0.4),
    (r"\b(benzinga)\b", 0.4),
    (r"\b(globenewswire|pr newswire|business wire)\b", 0.2),
]

# Enhanced scraping configuration
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

DOMAIN_STRATEGIES = {
    'simplywall.st': {
        'delay_range': (5, 8),
        'headers': {'Accept': 'text/html,application/xhtml+xml'},
    },
    'seekingalpha.com': {
        'delay_range': (3, 6),
        'referrer': 'https://www.google.com/search?q=stock+analysis',
    },
    'zacks.com': {
        'delay_range': (4, 7),
        'headers': {'Cache-Control': 'no-cache'},
    },
    'benzinga.com': {
        'delay_range': (4, 7),
        'headers': {'Accept': 'text/html,application/xhtml+xml'},
    },
    'tipranks.com': {
        'delay_range': (5, 8),
        'referrer': 'https://www.google.com/',
    }
}

# Ingestion limits (50/25/25) - for URL collection phase
ingestion_stats = {
    "company_ingested": 0,
    "industry_ingested_by_keyword": {},
    "competitor_ingested_by_keyword": {},
    "limits": {
        "company": 50,
        "industry_per_keyword": 25,
        "competitor_per_keyword": 25
    }
}

# Scraping limits (20 + 5×keywords + 5×competitors) - for content extraction phase
scraping_stats = {
    "company_scraped": 0,
    "industry_scraped_by_keyword": {},
    "competitor_scraped_by_keyword": {},
    "successful_scrapes": 0,
    "failed_scrapes": 0,
    "limits": {
        "company": 20,
        "industry_per_keyword": 5,
        "competitor_per_keyword": 5
    }
}

# Global ScrapingBee statistics tracking
scrapingbee_stats = {
    "requests_made": 0,
    "successful": 0,
    "failed": 0,
    "cost_estimate": 0.0,
    "by_domain": defaultdict(lambda: {"attempts": 0, "successes": 0})
}

# Global Scrapfly statistics tracking  
scrapfly_stats = {
    "requests_made": 0,
    "successful": 0,
    "failed": 0,
    "cost_estimate": 0.0,
    "by_domain": defaultdict(lambda: {"attempts": 0, "successes": 0})
}

# Domains known to be problematic with ScrapingBee
SCRAPINGBEE_PROBLEMATIC_DOMAINS = {
}

# Enhanced scraping statistics to track all methods
enhanced_scraping_stats = {
    "total_attempts": 0,
    "requests_success": 0,
    "playwright_success": 0,
    "scrapfly_success": 0,  # ADD THIS
    "scrapingbee_success": 0,
    "total_failures": 0,
    "by_method": {
        "requests": {"attempts": 0, "successes": 0},
        "playwright": {"attempts": 0, "successes": 0},
        "scrapfly": {"attempts": 0, "successes": 0},  # ADD THIS
        "scrapingbee": {"attempts": 0, "successes": 0}
    }
}

def reset_ingestion_stats():
    """Reset ingestion stats for new run"""
    global ingestion_stats
    ingestion_stats = {
        "company_ingested": 0,
        "industry_ingested_by_keyword": {},
        "competitor_ingested_by_keyword": {},
        "limits": {
            "company": 50,
            "industry_per_keyword": 25,
            "competitor_per_keyword": 25
        }
    }

def reset_scraping_stats():
    """Reset scraping stats for new run"""
    global scraping_stats
    scraping_stats = {
        "company_scraped": 0,
        "industry_scraped_by_keyword": {},
        "competitor_scraped_by_keyword": {},
        "successful_scrapes": 0,
        "failed_scrapes": 0,
        "limits": {
            "company": 20,
            "industry_per_keyword": 5,
            "competitor_per_keyword": 5
        }
    }

def _update_ingestion_stats(category: str, keyword: str):
    """Helper to update ingestion statistics - FIXED for competitor consolidation"""
    global ingestion_stats
    
    if category == "company":
        ingestion_stats["company_ingested"] += 1
        LOG.info(f"INGESTION: Company {ingestion_stats['company_ingested']}/{ingestion_stats['limits']['company']}")
    
    elif category == "industry":
        if keyword not in ingestion_stats["industry_ingested_by_keyword"]:
            ingestion_stats["industry_ingested_by_keyword"][keyword] = 0
        ingestion_stats["industry_ingested_by_keyword"][keyword] += 1
        keyword_count = ingestion_stats["industry_ingested_by_keyword"][keyword]
        LOG.info(f"INGESTION: Industry '{keyword}' {keyword_count}/{ingestion_stats['limits']['industry_per_keyword']}")
    
    elif category == "competitor":
        # FIXED: Use competitor_ticker as the consolidation key, not search_keyword
        if keyword not in ingestion_stats["competitor_ingested_by_keyword"]:
            ingestion_stats["competitor_ingested_by_keyword"][keyword] = 0
        ingestion_stats["competitor_ingested_by_keyword"][keyword] += 1
        keyword_count = ingestion_stats["competitor_ingested_by_keyword"][keyword]
        LOG.info(f"INGESTION: Competitor '{keyword}' {keyword_count}/{ingestion_stats['limits']['competitor_per_keyword']}")

def _check_ingestion_limit(category: str, keyword: str) -> bool:
    """Check if we can ingest more articles for this category/keyword - FIXED for competitor consolidation"""
    global ingestion_stats
    
    if category == "company":
        return ingestion_stats["company_ingested"] < ingestion_stats["limits"]["company"]
    
    elif category == "industry":
        keyword_count = ingestion_stats["industry_ingested_by_keyword"].get(keyword, 0)
        return keyword_count < ingestion_stats["limits"]["industry_per_keyword"]
    
    elif category == "competitor":
        # FIXED: Use competitor_ticker as the consolidation key, not search_keyword
        keyword_count = ingestion_stats["competitor_ingested_by_keyword"].get(keyword, 0)
        return keyword_count < ingestion_stats["limits"]["competitor_per_keyword"]
    
    return False

# Global tracking for domain access timing
domain_last_accessed = {}
last_scraped_domain = None

# Global Playwright statistics tracking
playwright_stats = {
    "attempted": 0,
    "successful": 0,
    "failed": 0,
    "by_domain": defaultdict(lambda: {"attempts": 0, "successes": 0})
}

def log_playwright_stats():
    """Log current Playwright performance statistics"""
    if playwright_stats["attempted"] == 0:
        return
    
    success_rate = (playwright_stats["successful"] / playwright_stats["attempted"]) * 100
    LOG.info(f"PLAYWRIGHT STATS: {success_rate:.1f}% success rate ({playwright_stats['successful']}/{playwright_stats['attempted']})")
    
    # Log top performing and failing domains
    domain_stats = playwright_stats["by_domain"]
    if domain_stats:
        successful_domains = [(domain, stats) for domain, stats in domain_stats.items() if stats["successes"] > 0]
        failed_domains = [(domain, stats) for domain, stats in domain_stats.items() if stats["successes"] == 0 and stats["attempts"] > 0]
        
        if successful_domains:
            LOG.info(f"PLAYWRIGHT SUCCESS DOMAINS: {len(successful_domains)} domains working")
        if failed_domains:
            LOG.info(f"PLAYWRIGHT FAILED DOMAINS: {len(failed_domains)} domains still blocked")

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
    """Complete database schema initialization with ticker-specific AI analysis"""
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS found_url (
                    id SERIAL PRIMARY KEY,
                    url TEXT NOT NULL,
                    resolved_url TEXT,
                    url_hash VARCHAR(32) NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    feed_id INTEGER,
                    ticker VARCHAR(10) NOT NULL,
                    domain VARCHAR(255),
                    published_at TIMESTAMP,
                    found_at TIMESTAMP DEFAULT NOW(),
                    sent_in_digest BOOLEAN DEFAULT FALSE,
                    category VARCHAR(20) DEFAULT 'company',
                    search_keyword TEXT,
                    original_source_url TEXT,
                    scraped_content TEXT,
                    content_scraped_at TIMESTAMP,
                    scraping_failed BOOLEAN DEFAULT FALSE,
                    scraping_error TEXT,
                    ai_summary TEXT,
                    ai_triage_selected BOOLEAN DEFAULT FALSE,
                    triage_priority INTEGER,
                    triage_reasoning TEXT,
                    qb_score INTEGER,
                    qb_level VARCHAR(20),
                    qb_reasoning TEXT,
                    competitor_ticker VARCHAR(10),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    ai_analysis_ticker VARCHAR(10)
                );
                
                -- Add missing columns if they don't exist
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS competitor_ticker VARCHAR(10);
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS qb_score INTEGER;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS qb_level VARCHAR(20);
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS qb_reasoning TEXT;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS ai_triage_selected BOOLEAN DEFAULT FALSE;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS triage_priority INTEGER;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS triage_reasoning TEXT;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS ai_summary TEXT;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS scraped_content TEXT;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS content_scraped_at TIMESTAMP;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS scraping_failed BOOLEAN DEFAULT FALSE;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS scraping_error TEXT;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS ai_analysis_ticker VARCHAR(10);
                
                -- Update NULL values
                UPDATE found_url SET updated_at = found_at WHERE updated_at IS NULL;
                UPDATE found_url SET created_at = found_at WHERE created_at IS NULL;
                
                -- Create regular indexes first
                CREATE INDEX IF NOT EXISTS idx_found_url_hash ON found_url(url_hash);
                CREATE INDEX IF NOT EXISTS idx_found_url_ticker_published ON found_url(ticker, published_at DESC);
                CREATE INDEX IF NOT EXISTS idx_found_url_digest ON found_url(sent_in_digest, found_at DESC);
                
                -- Drop any existing problematic constraints/indexes first
                DROP INDEX IF EXISTS idx_found_url_unique_analysis;
                ALTER TABLE found_url DROP CONSTRAINT IF EXISTS unique_url_ticker_analysis;
                
                -- Create simple unique constraint on the three columns
                CREATE UNIQUE INDEX IF NOT EXISTS idx_found_url_unique_analysis 
                ON found_url(url_hash, ticker, COALESCE(ai_analysis_ticker, ''));
                
                -- Rest of schema...
                CREATE TABLE IF NOT EXISTS ticker_config (
                    ticker VARCHAR(10) PRIMARY KEY,
                    name VARCHAR(255),
                    industry_keywords TEXT[],
                    competitors TEXT[],
                    active BOOLEAN DEFAULT TRUE,
                    ai_generated BOOLEAN DEFAULT FALSE,
                    sector VARCHAR(255),
                    industry VARCHAR(255),
                    sub_industry VARCHAR(255),
                    sector_profile JSONB,
                    aliases_brands_assets JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS domain_names (
                    domain VARCHAR(255) PRIMARY KEY,
                    formal_name VARCHAR(255) NOT NULL,
                    ai_generated BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS source_feed (
                    id SERIAL PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    ticker VARCHAR(10) NOT NULL,
                    category VARCHAR(20) DEFAULT 'company',
                    retain_days INTEGER DEFAULT 90,
                    active BOOLEAN DEFAULT TRUE,
                    search_keyword TEXT,
                    competitor_ticker VARCHAR(10),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS competitor_metadata (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    company_name VARCHAR(255) NOT NULL,
                    parent_ticker VARCHAR(10) NOT NULL,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(ticker, parent_ticker)
                );
            """)

    update_schema_for_enhanced_metadata()
    update_schema_for_triage()

# Add these fields to your database schema
def update_schema_for_content():
    """Add content scraping fields to found_url table"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE found_url ADD COLUMN IF NOT EXISTS scraped_content TEXT;
            ALTER TABLE found_url ADD COLUMN IF NOT EXISTS content_scraped_at TIMESTAMP;
            ALTER TABLE found_url ADD COLUMN IF NOT EXISTS scraping_failed BOOLEAN DEFAULT FALSE;
            ALTER TABLE found_url ADD COLUMN IF NOT EXISTS scraping_error TEXT;
        """)

def update_schema_for_enhanced_metadata():
    """Add enhanced metadata fields to ticker_config table"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE ticker_config ADD COLUMN IF NOT EXISTS sector VARCHAR(255);
            ALTER TABLE ticker_config ADD COLUMN IF NOT EXISTS industry VARCHAR(255);
            ALTER TABLE ticker_config ADD COLUMN IF NOT EXISTS sub_industry VARCHAR(255);
            ALTER TABLE ticker_config ADD COLUMN IF NOT EXISTS sector_profile JSONB;
            ALTER TABLE ticker_config ADD COLUMN IF NOT EXISTS aliases_brands_assets JSONB;
        """)

def update_schema_for_qb_scores():
    """Add QB scoring fields to found_url table"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE found_url ADD COLUMN IF NOT EXISTS qb_score INTEGER;
            ALTER TABLE found_url ADD COLUMN IF NOT EXISTS qb_level VARCHAR(20);
            ALTER TABLE found_url ADD COLUMN IF NOT EXISTS qb_reasoning TEXT;
        """)

def update_schema_for_triage():
    """Add triage fields to found_url table - FIXED for HIGH/MEDIUM/LOW"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE found_url ADD COLUMN IF NOT EXISTS ai_triage_selected BOOLEAN DEFAULT FALSE;
            ALTER TABLE found_url ADD COLUMN IF NOT EXISTS triage_priority VARCHAR(10);
            ALTER TABLE found_url ADD COLUMN IF NOT EXISTS triage_reasoning TEXT;
        """)

def create_ticker_reference_table():
    """Create simplified ticker reference table"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ticker_reference (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20) UNIQUE NOT NULL,  -- This IS the Yahoo format
                country VARCHAR(5) NOT NULL,
                company_name VARCHAR(255) NOT NULL,
                industry VARCHAR(255),
                sector VARCHAR(255),
                exchange VARCHAR(20),
                active BOOLEAN DEFAULT TRUE,
                industry_keywords TEXT[],
                competitors TEXT[],
                ai_generated BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_ticker_reference_ticker ON ticker_reference(ticker);
        """)

def store_competitor_metadata(ticker: str, competitors: List[Dict]) -> None:
    """Store competitor metadata in a dedicated table for better normalization"""
    with db() as conn, conn.cursor() as cur:
        # Create competitor metadata table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS competitor_metadata (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                company_name VARCHAR(255) NOT NULL,
                parent_ticker VARCHAR(10) NOT NULL,  -- The ticker this competitor relates to
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, parent_ticker)
            );
        """)
        
        # Store each competitor
        for comp in competitors:
            if isinstance(comp, dict) and comp.get('ticker') and comp.get('name'):
                cur.execute("""
                    INSERT INTO competitor_metadata (ticker, company_name, parent_ticker)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (ticker, parent_ticker) 
                    DO UPDATE SET 
                        company_name = EXCLUDED.company_name,
                        updated_at = NOW()
                """, (comp['ticker'], comp['name'], ticker))

def extract_article_content(url: str, domain: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Enhanced article extraction with intelligent delay management and content cleaning
    """
    try:
        # Check for known paywall domains first (reduced list)
        if normalize_domain(domain) in PAYWALL_DOMAINS:
            return None, f"Paywall domain: {domain}"
        
        # Get domain-specific strategy
        strategy = get_domain_strategy(domain)
        
        # Add domain-specific headers
        headers = strategy.get('headers', {})
        headers['Referer'] = get_referrer_for_domain(url, domain)
        headers['User-Agent'] = get_random_user_agent()
        
        # Apply headers to session
        scraping_session.headers.update(headers)
        
        # INTELLIGENT DELAY - only delay when necessary
        delay, reason = calculate_intelligent_delay(domain)
        if delay > 0.5:  # Only log significant delays
            LOG.info(f"Waiting {delay:.1f}s before scraping {domain} ({reason})")
            time.sleep(delay)
        else:
            LOG.debug(f"Quick scrape of {domain} ({reason})")
        
        # Try enhanced scraping with backoff
        response = scrape_with_backoff(url)
        if not response:
            return None, "Failed to fetch URL after retries"
        
        # Handle cookies and JavaScript redirects
        if 'Set-Cookie' in response.headers:
            scraping_session.cookies.update(response.cookies)
        
        # Check for JavaScript redirects
        if 'window.location' in response.text or 'document.location' in response.text:
            return None, "JavaScript redirect detected"
        
        # Use newspaper3k to parse the HTML
        config = newspaper.Config()
        config.browser_user_agent = headers.get('User-Agent')
        config.request_timeout = 15
        config.fetch_images = False
        config.memoize_articles = False
        
        article = newspaper.Article(url, config=config)
        article.set_html(response.text)
        article.parse()
        
        # Get the main text content
        raw_content = article.text.strip()
        
        if not raw_content:
            return None, "No content extracted"
        
        # ENHANCED CONTENT CLEANING
        cleaned_content = clean_scraped_content(raw_content, url, domain)
        
        if not cleaned_content or len(cleaned_content.strip()) < 100:
            return None, "Content too short after cleaning"
        
        # Enhanced cookie banner detection on cleaned content
        content_lower = cleaned_content.lower()
        cookie_indicators = [
            "we use cookies", "accept all cookies", "cookie policy",
            "privacy policy and terms of service", "consent to the use"
        ]
        
        cookie_count = sum(1 for indicator in cookie_indicators if indicator in content_lower)
        
        # If multiple cookie indicators and content is short, likely a cookie page
        if cookie_count >= 2 and len(cleaned_content) < 500:
            return None, "Cookie consent page detected"
        
        # Enhanced paywall indicators (check cleaned content)
        paywall_indicators = [
            "subscribe to continue", "unlock this story", "premium content",
            "sign up to read", "become a member", "subscription required"
        ]
        
        if any(indicator in content_lower for indicator in paywall_indicators):
            return None, "Paywall content detected"
        
        # Validate content quality on cleaned content
        is_valid, validation_msg = validate_scraped_content(cleaned_content, url, domain)
        if not is_valid:
            return None, validation_msg
        
        LOG.info(f"Successfully extracted and cleaned {len(cleaned_content)} chars from {domain}")
        return cleaned_content, None
        
    except newspaper.article.ArticleException as e:
        error_msg = f"Newspaper article error: {str(e)}"
        LOG.warning(f"Failed to extract content from {url}: {error_msg}")
        return None, error_msg
    except Exception as e:
        error_msg = f"Content extraction failed: {str(e)}"
        LOG.warning(f"Failed to extract content from {url}: {error_msg}")
        return None, error_msg

def create_scraping_session():
    session = requests.Session()
    
    # Retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Comprehensive headers
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    })
    
    return session

# Add this function after your existing extract_article_content function
def extract_article_content_with_playwright(url: str, domain: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Memory-optimized article extraction using Playwright with enhanced content cleaning
    """
    try:
        # Check for known paywall domains first
        if normalize_domain(domain) in PAYWALL_DOMAINS:
            return None, f"Paywall domain: {domain}"
        
        LOG.info(f"PLAYWRIGHT: Starting browser for {domain}")
        
        with sync_playwright() as p:
            # Launch browser with memory optimization
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--memory-pressure-off',
                    '--disable-background-timer-throttling',
                    '--disable-renderer-backgrounding',
                    '--disable-extensions',
                    '--disable-plugins',
                    '--disable-images',
                    '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                ]
            )
            
            LOG.info(f"PLAYWRIGHT: Browser launched, navigating to {url}")
            
            # Create new page with smaller viewport to save memory
            page = browser.new_page(
                viewport={'width': 1280, 'height': 720},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            # Set additional headers to look more human
            page.set_extra_http_headers({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            })
            
            try:
                # REDUCED TIMEOUT - 15 seconds instead of 30
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                LOG.info(f"PLAYWRIGHT: Page loaded for {domain}, extracting content...")
                
                # Shorter wait for dynamic content
                page.wait_for_timeout(1000)
                
                # Try multiple content extraction methods
                raw_content = None
                extraction_method = None
                
                # Method 1: Try article tag first
                try:
                    article_element = page.query_selector('article')
                    if article_element:
                        raw_content = article_element.inner_text()
                        extraction_method = "article tag"
                except Exception:
                    pass
                
                # Method 2: Try main content selectors
                if not raw_content or len(raw_content.strip()) < 200:
                    selectors = [
                        '[role="main"]',
                        'main',
                        '.article-content',
                        '.story-content',
                        '.entry-content',
                        '.post-content',
                        '.content',
                        '[data-module="ArticleBody"]'
                    ]
                    for selector in selectors:
                        try:
                            element = page.query_selector(selector)
                            if element:
                                temp_content = element.inner_text()
                                if temp_content and len(temp_content.strip()) > 200:
                                    raw_content = temp_content
                                    extraction_method = f"selector: {selector}"
                                    break
                        except Exception:
                            continue
                
                # Method 3: Smart body text extraction (removes navigation/ads)
                if not raw_content or len(raw_content.strip()) < 200:
                    try:
                        raw_content = page.evaluate("""
                            () => {
                                // Remove unwanted elements
                                const unwanted = document.querySelectorAll(`
                                    script, style, nav, header, footer, aside,
                                    .advertisement, .ads, .ad, .sidebar,
                                    .navigation, .nav, .menu, .social,
                                    [class*="ad"], [class*="sidebar"], [class*="nav"]
                                `);
                                unwanted.forEach(el => el.remove());
                                
                                // Try to find main content area
                                const candidates = [
                                    document.querySelector('main'),
                                    document.querySelector('[role="main"]'),
                                    document.querySelector('.main-content'),
                                    document.querySelector('.content'),
                                    document.body
                                ];
                                
                                for (let candidate of candidates) {
                                    if (candidate && candidate.innerText && candidate.innerText.length > 200) {
                                        return candidate.innerText;
                                    }
                                }
                                
                                return document.body ? document.body.innerText : '';
                            }
                        """)
                        extraction_method = "smart body extraction"
                    except Exception:
                        try:
                            raw_content = page.evaluate("() => document.body ? document.body.innerText : ''")
                            extraction_method = "fallback body text"
                        except Exception:
                            raw_content = None
                
            except Exception as e:
                LOG.warning(f"PLAYWRIGHT: Navigation/extraction failed for {domain}: {str(e)}")
                raw_content = None
            finally:
                # Always close browser to free memory
                try:
                    browser.close()
                except Exception:
                    pass
            
            if not raw_content or len(raw_content.strip()) < 100:
                LOG.warning(f"PLAYWRIGHT FAILED: {domain} -> Insufficient raw content extracted")
                return None, "Insufficient content extracted"
            
            # ENHANCED CONTENT CLEANING
            cleaned_content = clean_scraped_content(raw_content, url, domain)
            
            if not cleaned_content or len(cleaned_content.strip()) < 100:
                LOG.warning(f"PLAYWRIGHT FAILED: {domain} -> Content too short after cleaning")
                return None, "Content too short after cleaning"
            
            # Enhanced content validation on cleaned content
            is_valid, validation_msg = validate_scraped_content(cleaned_content, url, domain)
            if not is_valid:
                LOG.warning(f"PLAYWRIGHT FAILED: {domain} -> {validation_msg}")
                return None, validation_msg
            
            # Check for common error pages or blocking messages on cleaned content
            content_lower = cleaned_content.lower()
            error_indicators = [
                "403 forbidden", "access denied", "captcha", "robot", "bot detection",
                "please verify you are human", "cloudflare", "rate limit", "blocked",
                "security check", "unusual traffic"
            ]
            
            if any(indicator in content_lower for indicator in error_indicators):
                LOG.warning(f"PLAYWRIGHT FAILED: {domain} -> Error page detected")
                return None, "Error page or blocking detected"
            
            LOG.info(f"PLAYWRIGHT SUCCESS: {domain} -> {len(cleaned_content)} chars extracted and cleaned via {extraction_method}")
            return cleaned_content.strip(), None
            
    except Exception as e:
        error_msg = f"Playwright extraction failed: {str(e)}"
        LOG.error(f"PLAYWRIGHT ERROR: {domain} -> {error_msg}")
        return None, error_msg
    finally:
        # Add this cleanup
        import gc
        gc.collect()

def scrape_with_scrapingbee(url: str, domain: str, max_retries: int = 2) -> Tuple[Optional[str], Optional[str]]:
    """
    Enhanced ScrapingBee with improved text extraction and content cleaning
    """
    global scrapingbee_stats, enhanced_scraping_stats
    
    for attempt in range(max_retries + 1):
        try:
            if not SCRAPINGBEE_API_KEY:
                return None, "ScrapingBee API key not configured"
            
            if normalize_domain(domain) in PAYWALL_DOMAINS:
                return None, f"Paywall domain: {domain}"
            
            if attempt > 0:
                delay = 2 ** attempt
                LOG.info(f"SCRAPINGBEE RETRY {attempt}/{max_retries} for {domain} after {delay}s delay")
                time.sleep(delay)
            
            LOG.info(f"SCRAPINGBEE: Starting scrape for {domain} (attempt {attempt + 1})")
            
            # Update usage stats
            scrapingbee_stats["requests_made"] += 1
            scrapingbee_stats["cost_estimate"] += 0.001
            scrapingbee_stats["by_domain"][domain]["attempts"] += 1
            enhanced_scraping_stats["by_method"]["scrapingbee"]["attempts"] += 1
            
            # Improved parameters with better text extraction targeting
            params = {
                'api_key': SCRAPINGBEE_API_KEY,
                'url': url,
                'render_js': 'false',
                'premium_proxy': 'true',
                'country_code': 'us',
                'timeout': 15000,
                'extract_rules': '{"article_text": "article, main, .article-content, .story-content, .entry-content, .post-content, .content, [role=main], .post-body, .article-body"}',
                'block_ads': 'true',
                'block_resources': 'true'
            }
            
            response = requests.get('https://app.scrapingbee.com/api/v1/', 
                                   params=params, 
                                   timeout=30)
            
            if response.status_code == 200:
                try:
                    # ScrapingBee returns JSON when using extract_rules
                    result = response.json()
                    raw_content = result.get('article_text', '') or response.text
                except:
                    # Fallback to raw text if JSON parsing fails
                    raw_content = response.text
                
                if not raw_content or len(raw_content.strip()) < 100:
                    LOG.warning(f"SCRAPINGBEE: Insufficient raw content for {domain} (attempt {attempt + 1})")
                    continue
                
                # Apply content cleaning to the extracted text
                cleaned_content = clean_scraped_content(raw_content, url, domain)
                
                if not cleaned_content or len(cleaned_content.strip()) < 100:
                    LOG.warning(f"SCRAPINGBEE: Content too short after cleaning for {domain}")
                    continue
                
                # Validate cleaned content quality
                is_valid, validation_msg = validate_scraped_content(cleaned_content, url, domain)
                if is_valid:
                    scrapingbee_stats["successful"] += 1
                    scrapingbee_stats["by_domain"][domain]["successes"] += 1
                    enhanced_scraping_stats["by_method"]["scrapingbee"]["successes"] += 1
                    
                    # Enhanced logging to show cleaning effectiveness
                    raw_len = len(str(raw_content))
                    clean_len = len(cleaned_content)
                    reduction = ((raw_len - clean_len) / raw_len * 100) if raw_len > 0 else 0
                    
                    LOG.info(f"SCRAPINGBEE SUCCESS: {domain} -> {clean_len} chars (cleaned from {raw_len}, {reduction:.1f}% reduction)")
                    return cleaned_content, None
                else:
                    LOG.warning(f"SCRAPINGBEE: Content validation failed for {domain}: {validation_msg}")
                    break
            
            elif response.status_code == 500:
                LOG.warning(f"SCRAPINGBEE: Server error 500 for {domain} (attempt {attempt + 1})")
                if attempt < max_retries:
                    continue
                else:
                    LOG.warning(f"SCRAPINGBEE: Max retries reached for {domain} after repeated 500 errors")
            
            elif response.status_code == 422:
                LOG.warning(f"SCRAPINGBEE: Invalid parameters for {domain}")
                break
            
            else:
                LOG.warning(f"SCRAPINGBEE: HTTP {response.status_code} for {domain}")
                if attempt < max_retries:
                    continue
                    
        except requests.RequestException as e:
            LOG.warning(f"SCRAPINGBEE: Request error for {domain} (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                continue
        except Exception as e:
            LOG.error(f"SCRAPINGBEE: Unexpected error for {domain}: {e}")
            break
    
    scrapingbee_stats["failed"] += 1
    return None, f"ScrapingBee failed after {max_retries + 1} attempts"

def scrape_with_scrapfly(
    url: str,
    domain: str,
    max_retries: int = 2,
    base_params: dict | None = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Scrapfly scraping with improved text extraction and content cleaning.
    - Uses client-side retries/backoff here (do not rely on Scrapfly retry).
    - Does NOT send 'timeout' in Scrapfly params (prevents 400 when retry is enabled server-side).
    - Cranks up anti-bot only for domains in LOCAL_SCRAPFLY_ANTIBOT.
    Returns: (cleaned_content, error_message)
    """
    global scrapfly_stats, enhanced_scraping_stats

    def _host(u: str) -> str:
        h = (urlparse(u).hostname or "").lower()
        if h.startswith("www."):
            h = h[4:]
        return h

    def _matches(host: str, dom: str) -> bool:
        return host == dom or host.endswith("." + dom)

    # Local list affects ONLY this function (you said you’ll empty the global list upstream)
    LOCAL_SCRAPFLY_ANTIBOT = {
        "simplywall.st", "seekingalpha.com", "zacks.com", "benzinga.com",
        "cnbc.com", "investing.com", "gurufocus.com", "fool.com",
        "insidermonkey.com", "nasdaq.com", "markets.financialcontent.com",
        "thefly.com", "streetinsider.com", "accesswire.com",
        "247wallst.com", "barchart.com", "telecompaper.com",
        "news.stocktradersdaily.com", "sharewise.com",
        "video.media.yql.yahoo.com", "templates.cds.yahoo.com"
    }

    for attempt in range(max_retries + 1):
        try:
            if not SCRAPFLY_API_KEY:
                return None, "Scrapfly API key not configured"

            if normalize_domain(domain) in PAYWALL_DOMAINS:
                return None, f"Paywall domain: {domain}"

            if attempt > 0:
                delay = 2 ** attempt
                LOG.info(f"SCRAPFLY RETRY {attempt}/{max_retries} for {domain} after {delay}s delay")
                time.sleep(delay)

            LOG.info(f"SCRAPFLY: Starting scrape for {domain} (attempt {attempt + 1})")

            # --- usage stats
            scrapfly_stats["requests_made"] += 1
            scrapfly_stats["cost_estimate"] += 0.002  # rough est.
            scrapfly_stats["by_domain"][domain]["attempts"] += 1
            enhanced_scraping_stats["by_method"]["scrapfly"]["attempts"] += 1

            host = _host(url)

            # --- Build Scrapfly params (NO 'timeout' here; keep client timeout in requests.get)
            params = {
                "key": SCRAPFLY_API_KEY,
                "url": url,
                "render_js": False,
                "country": "us",
                "cache": False,
                **(base_params or {}),
            }

            # Toggle anti-bot/JS only for local list
            if any(_matches(host, d) for d in LOCAL_SCRAPFLY_ANTIBOT):
                params["asp"] = True
                params["render_js"] = True

            response = requests.get("https://api.scrapfly.io/scrape", params=params, timeout=30)

            # ---- Status handling
            if response.status_code == 200:
                try:
                    result = response.json()
                    html_content = result.get("result", {}).get("content", "") or ""
                except Exception as json_error:
                    LOG.warning(f"SCRAPFLY: JSON parsing failed for {domain}: {json_error}")
                    html_content = response.text or ""

                # Extract text from HTML (don't rely on unsupported API-side text formats)
                raw_content = ""
                if html_content:
                    try:
                        article = newspaper.Article(url)
                        article.set_html(html_content)
                        article.parse()
                        raw_content = article.text or ""
                    except Exception as e:
                        LOG.warning(f"SCRAPFLY: Newspaper parse failed for {domain}: {e}")
                        raw_content = html_content

                if not raw_content or len(raw_content.strip()) < 100:
                    LOG.warning(f"SCRAPFLY: Insufficient raw content for {domain} (attempt {attempt + 1})")
                    continue

                cleaned_content = clean_scraped_content(raw_content, url, domain)
                if not cleaned_content or len(cleaned_content.strip()) < 100:
                    LOG.warning(f"SCRAPFLY: Content too short after cleaning for {domain}")
                    continue

                is_valid, validation_msg = validate_scraped_content(cleaned_content, url, domain)
                if not is_valid:
                    LOG.warning(f"SCRAPFLY: Content validation failed for {domain}: {validation_msg}")
                    break

                # success stats
                scrapfly_stats["successful"] += 1
                scrapfly_stats["by_domain"][domain]["successes"] += 1
                enhanced_scraping_stats["by_method"]["scrapfly"]["successes"] += 1

                raw_len = len(str(raw_content))
                clean_len = len(cleaned_content)
                reduction = ((raw_len - clean_len) / raw_len * 100) if raw_len > 0 else 0.0
                LOG.info(f"SCRAPFLY SUCCESS: {domain} -> {clean_len} chars (cleaned from {raw_len}, {reduction:.1f}% reduction)")
                return cleaned_content, None

            elif response.status_code == 422:
                LOG.warning(f"SCRAPFLY: 422 invalid parameters for {domain} body: {response.text[:500]}")
                break

            elif response.status_code == 429:
                LOG.warning(f"SCRAPFLY: 429 rate limited for {domain} (attempt {attempt + 1})")
                if attempt < max_retries:
                    time.sleep(5)
                    continue

            else:
                req_id = response.headers.get("x-request-id") or response.headers.get("cf-ray")
                LOG.warning(
                    f"SCRAPFLY: HTTP {response.status_code} for {domain} "
                    f"(attempt {attempt + 1}) id={req_id} body: {response.text[:500]}"
                )
                if attempt < max_retries:
                    continue

        except requests.RequestException as e:
            LOG.warning(f"SCRAPFLY: Request error for {domain} (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                continue
        except Exception as e:
            LOG.error(f"SCRAPFLY: Unexpected error for {domain}: {e}")
            break

    scrapfly_stats["failed"] += 1
    return None, f"Scrapfly failed after {max_retries + 1} attempts"

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def get_referrer_for_domain(url, domain):
    """Add realistic referrer headers"""
    referrers = {
        'default': [
            'https://www.google.com/',
            'https://duckduckgo.com/',
            'https://www.bing.com/',
        ],
        'news_sites': [
            'https://news.google.com/',
            'https://finance.yahoo.com/',
            'https://www.reddit.com/r/investing/',
        ]
    }
    
    if any(news in domain for news in ['news', 'finance', 'investing']):
        return random.choice(referrers['news_sites'])
    return random.choice(referrers['default'])

def clean_scraped_content(content: str, url: str = "", domain: str = "") -> str:
    """
    Conservative content cleaning that removes obvious junk while preserving article content
    """
    if not content:
        return ""
    
    original_length = len(content)
    
    # Stage 1: Remove obvious binary/encoded data
    # Remove sequences that look like binary data or encoding artifacts
    content = re.sub(r'[¿½]{3,}.*?[¿½]{3,}', '', content)  # Remove sequences with encoding markers
    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]+', '', content)  # Remove control characters
    content = re.sub(r'[A-Za-z0-9+/]{50,}={0,2}', '', content)  # Remove base64-like sequences
    
    # Stage 2: Remove HTML/CSS/JavaScript remnants
    # Remove HTML tags that newspaper3k might have missed
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'&[a-zA-Z0-9#]+;', ' ', content)  # HTML entities
    
    # Remove CSS-like patterns
    content = re.sub(r'\{[^}]*\}', '', content)  # CSS rules
    content = re.sub(r'@media[^{]*\{[^}]*\}', '', content)  # Media queries
    content = re.sub(r'font-[a-z-]+:[^;]+;', '', content)  # CSS font properties
    content = re.sub(r'text-[a-z-]+:[^;]+;', '', content)  # CSS text properties
    content = re.sub(r'color:[^;]+;', '', content)  # CSS colors
    content = re.sub(r'margin[^;]*:[^;]+;', '', content)  # CSS margins
    content = re.sub(r'padding[^;]*:[^;]+;', '', content)  # CSS padding
    
    # Remove class and data attributes
    content = re.sub(r'class="[^"]*"', '', content)
    content = re.sub(r'data-[a-z-]+="[^"]*"', '', content)
    
    # Stage 3: Remove technical metadata
    # Remove image metadata
    content = re.sub(r'EXIF[^.]*\.', '', content)
    content = re.sub(r'sRGB\.IEC[^.]*\.', '', content)
    content = re.sub(r'Adobe RGB[^.]*\.', '', content)
    content = re.sub(r'Photoshop[^.]*\.', '', content)
    
    # Remove file paths and technical specs
    content = re.sub(r'/[A-Z]+/[A-Z0-9]+', '', content)  # File paths like /MARCH/MARCH30
    content = re.sub(r'[A-Z]{2,}\d+[A-Z]*\d*', '', content)  # Technical codes like RTAO, RGB
    content = re.sub(r'\d{8,}', '', content)  # Long number sequences
    
    # Stage 4: Remove navigation and UI elements
    navigation_patterns = [
        r'Home\s*>\s*[^.]*',  # Breadcrumbs
        r'Share on [A-Za-z]+',  # Social sharing
        r'Tweet this',
        r'Follow us on',
        r'Subscribe to',
        r'Sign up for',
        r'Newsletter',
        r'Related Articles?',
        r'More from',
        r'Continue reading',
        r'Read more',
        r'Click here',
        r'Advertisement',
        r'Sponsored Content',
        r'Exit Content Preview',
    ]
    
    for pattern in navigation_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
    # Stage 5: Remove cookie and consent text
    cookie_patterns = [
        r'We use cookies[^.]*\.',
        r'Accept all cookies[^.]*\.',
        r'Cookie policy[^.]*\.',
        r'Privacy policy[^.]*\.',
        r'By continuing to use[^.]*\.',
        r'This site uses cookies[^.]*\.',
    ]
    
    for pattern in cookie_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
    # Stage 6: Clean up whitespace but preserve paragraph structure
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Reduce excessive line breaks
    content = re.sub(r'[ \t]+', ' ', content)  # Normalize spaces
    content = re.sub(r'\n +', '\n', content)  # Remove leading spaces on lines
    content = re.sub(r' +\n', '\n', content)  # Remove trailing spaces on lines
    
    # Stage 7: Remove obviously broken sentences/paragraphs
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip lines that are mostly special characters or numbers
        if len(re.sub(r'[a-zA-Z\s]', '', line)) > len(line) * 0.5:
            continue
            
        # Skip very short fragments that don't end properly
        if len(line) < 20 and not line.endswith(('.', '!', '?', ':')):
            continue
            
        # Skip lines that look like technical data
        if re.search(r'^[A-Z]{3,}[0-9]', line):  # Technical codes
            continue
            
        if re.search(r'^\d+px|em|rem|%', line):  # CSS measurements
            continue
            
        # Keep lines that look like real sentences
        cleaned_lines.append(line)
    
    # Reconstruct content with proper paragraph breaks
    content = '\n\n'.join(cleaned_lines)
    
    # Final cleanup
    content = content.strip()
    
    # Log cleaning effectiveness
    final_length = len(content)
    reduction = ((original_length - final_length) / original_length * 100) if original_length > 0 else 0
    
    LOG.debug(f"Content cleaning: {original_length} → {final_length} chars ({reduction:.1f}% reduction)")
    
    return content

def get_domain_strategy(domain):
    return DOMAIN_STRATEGIES.get(domain, {
        'delay_range': (4, 8),
        'headers': {},
    })

def calculate_intelligent_delay(domain):
    """
    Calculate delay based on domain access patterns:
    - No delay for first-time domains
    - Short delay (1-2s) if different from last domain
    - Full delay (4-8s) if same as last domain or recent access
    """
    global last_scraped_domain, domain_last_accessed
    
    current_time = time.time()
    strategy = get_domain_strategy(domain)
    
    # Check if we've accessed this domain recently (within last 10 seconds)
    last_access_time = domain_last_accessed.get(domain, 0)
    time_since_last_access = current_time - last_access_time
    
    # Determine delay strategy
    if domain == last_scraped_domain:
        # Same domain as last scrape - use full delay
        delay_min, delay_max = strategy.get('delay_range', (4, 8))
        delay = random.uniform(delay_min, delay_max)
        reason = "same domain as previous"
    elif time_since_last_access < 10:
        # Recently accessed this domain - use medium delay
        delay = random.uniform(2, 4)
        reason = f"accessed {time_since_last_access:.1f}s ago"
    elif domain not in domain_last_accessed:
        # First time accessing this domain - minimal delay
        delay = random.uniform(0.5, 1.5)
        reason = "first access"
    else:
        # Different domain, not recently accessed - short delay
        delay = random.uniform(1, 2)
        reason = "different domain"
    
    # Update tracking
    domain_last_accessed[domain] = current_time
    last_scraped_domain = domain
    
    return delay, reason

def scrape_with_backoff(url, max_retries=3):
    """Enhanced scraping with exponential backoff for 500, 429, 503 errors"""
    for attempt in range(max_retries):
        try:
            response = scraping_session.get(url, timeout=15)
            if response.status_code == 200:
                return response
            elif response.status_code in [429, 500, 503]:  # Added 500
                # Server error or rate limited - wait with exponential backoff
                delay = (2 ** attempt) + random.uniform(0, 1)
                LOG.info(f"HTTP {response.status_code} error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                if attempt == max_retries - 1:
                    LOG.warning(f"Max retries reached for {url} after {response.status_code} errors")
                    return None
            else:
                # For other errors (404, 403, etc.), don't retry
                LOG.warning(f"HTTP {response.status_code} for {url}, not retrying")
                return None
        except requests.RequestException as e:
            delay = (2 ** attempt) + random.uniform(0, 1)
            LOG.warning(f"Request failed, retrying in {delay:.1f}s: {e}")
            time.sleep(delay)
    
    return None

def log_scraping_success_rates():
    """Log success rates for all scraping methods in a prominent format"""
    total_attempts = enhanced_scraping_stats["total_attempts"]
    if total_attempts == 0:
        LOG.info("SCRAPING SUCCESS RATES: No attempts made")
        return
    
    # Calculate overall success rate
    total_success = (enhanced_scraping_stats["requests_success"] + 
                    enhanced_scraping_stats["playwright_success"] + 
                    enhanced_scraping_stats["scrapingbee_success"])
    overall_rate = (total_success / total_attempts) * 100
    
    # Calculate individual success rates
    requests_attempts = enhanced_scraping_stats["by_method"]["requests"]["attempts"]
    requests_success = enhanced_scraping_stats["by_method"]["requests"]["successes"]
    requests_rate = (requests_success / requests_attempts * 100) if requests_attempts > 0 else 0
    
    playwright_attempts = enhanced_scraping_stats["by_method"]["playwright"]["attempts"]
    playwright_success = enhanced_scraping_stats["by_method"]["playwright"]["successes"]
    playwright_rate = (playwright_success / playwright_attempts * 100) if playwright_attempts > 0 else 0
    
    scrapingbee_attempts = enhanced_scraping_stats["by_method"]["scrapingbee"]["attempts"]
    scrapingbee_success = enhanced_scraping_stats["by_method"]["scrapingbee"]["successes"]
    scrapingbee_rate = (scrapingbee_success / scrapingbee_attempts * 100) if scrapingbee_attempts > 0 else 0
    
    # Log prominent success rate summary
    LOG.info("=" * 60)
    LOG.info("SCRAPING SUCCESS RATES")
    LOG.info("=" * 60)
    LOG.info(f"OVERALL SUCCESS: {overall_rate:.1f}% ({total_success}/{total_attempts})")
    LOG.info(f"TIER 1 (Requests): {requests_rate:.1f}% ({requests_success}/{requests_attempts})")
    LOG.info(f"TIER 2 (Playwright): {playwright_rate:.1f}% ({playwright_success}/{playwright_attempts})")
    LOG.info(f"TIER 3 (ScrapingBee): {scrapingbee_rate:.1f}% ({scrapingbee_success}/{scrapingbee_attempts})")
    
    if scrapingbee_attempts > 0:
        LOG.info(f"ScrapingBee Cost: ${scrapingbee_stats['cost_estimate']:.3f}")
    
    LOG.info("=" * 60)

def validate_scraped_content(content, url, domain):
    """Enhanced content validation for cleaned content"""
    if not content or len(content.strip()) < 100:
        return False, "Content too short"
    
    # Check content-to-boilerplate ratio
    sentences = [s.strip() for s in content.split('.') if s.strip()]
    if len(sentences) < 3:
        return False, "Insufficient sentences"
    
    # Check for repetitive content (often indicates scraping issues)
    words = content.lower().split()
    if len(words) > 0 and len(set(words)) / len(words) < 0.3:  # Less than 30% unique words
        return False, "Repetitive content detected"
    
    # Check if content is mostly technical/code-like
    technical_chars = len(re.findall(r'[{}();:=<>]', content))
    if technical_chars > len(content) * 0.1:  # More than 10% technical characters
        return False, "Content appears to be technical/code data"
    
    # REMOVED: Sentence structure validation that was rejecting legitimate articles
    
    return True, "Valid content"

# Create global session
scraping_session = create_scraping_session()

def is_insider_trading_article(title: str) -> bool:
    """
    Detect insider trading/institutional flow articles that are typically low-value
    Enhanced to catch more patterns
    """
    title_lower = title.lower()
    
    # Executive transactions
    insider_patterns = [
        # Existing patterns...
        r"\w+\s+(ceo|cfo|coo|president|director|officer|executive)\s+\w+\s+(sells?|buys?|purchases?)",
        r"(sells?|buys?|purchases?)\s+\$[\d,]+\.?\d*[km]?\s+(in\s+shares?|worth\s+of)",
        r"(insider|executive|officer|director|ceo|cfo)\s+(selling|buying|sold|bought)",
        
        # Institutional flow patterns  
        r"\w+\s+(capital|management|advisors?|investments?|llc|inc\.?)\s+(buys?|sells?|invests?|increases?|decreases?)",
        r"(invests?|buys?)\s+\$[\d,]+\.?\d*[km]?\s+in",
        r"shares?\s+(sold|bought)\s+by\s+",
        r"(increases?|decreases?|trims?|adds?\s+to)\s+(stake|position|holdings?)\s+in",
        
        # NEW: More aggressive patterns
        r"(acquires?|sells?)\s+\d+\s+shares?",
        r"(boosts?|cuts?|trims?)\s+(stake|holdings?|position)",
        r"(takes?|builds?)\s+\$[\d,]+\.?\d*[km]?\s+(stake|position)",
        r"(hedge fund|institutional|mutual fund)\s+.{0,20}(buys?|sells?|adds?|cuts?)",
        r"(13f|form\s*4|schedule\s*13d)\s+(filing|report)",
        r"(quarterly\s+)?(holdings?|portfolio)\s+(report|update|filing)",
        
        # Specific low-value phrases
        r"buys?\s+\d+\s+shares?",
        r"sells?\s+\d+\s+shares?", 
        r"shares?\s+(sold|purchased|bought)\s+by",
        r"\$[\d,]+\.?\d*[km]?\s+stake",
        r"(quarterly|q\d)\s+holdings?\s+(report|filing)",
        r"13f\s+filing",
        r"form\s+4\s+filing"
    ]
    
    # Check against patterns
    for pattern in insider_patterns:
        if re.search(pattern, title_lower):
            LOG.debug(f"INSIDER PATTERN MATCH: '{pattern}' in '{title[:50]}...'")
            return True
    
    # Enhanced small dollar amount detection
    small_amount_pattern = r"\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*([km])?"
    matches = re.findall(small_amount_pattern, title_lower)
    if matches and any(word in title_lower for word in ["sells", "buys", "purchases", "sold", "bought", "stake", "position"]):
        for amount_str, unit in matches:
            try:
                amount = float(amount_str.replace(",", ""))
                if unit.lower() == 'k':
                    amount *= 1000
                elif unit.lower() == 'm':
                    amount *= 1000000
                
                # Flag transactions under $100M as likely insider trading (raised threshold)
                if amount < 100000000:
                    LOG.debug(f"SMALL AMOUNT DETECTED: ${amount:,.0f} in '{title[:50]}...'")
                    return True
            except:
                continue
    
    return False

def safe_content_scraper(url: str, domain: str, scraped_domains: set) -> Tuple[Optional[str], str]:
    """
    Safe content scraper using requests only (no Playwright)
    """
    try:
        # Remove domain deduplication - scrape every URL
        # scraped_domains.add(domain)  # Optional: still track but don't block
        
        # Use the existing extract_article_content function
        content, error = extract_article_content(url, domain)
        
        if content:
            return content, f"Successfully scraped {len(content)} chars"
        else:
            return None, error or "Failed to extract content"
            
    except Exception as e:
        return None, f"Scraping error: {str(e)}"

def safe_content_scraper_with_3tier_fallback(url: str, domain: str, category: str, keyword: str, scraped_domains: set) -> Tuple[Optional[str], str]:
    """
    3-tier content scraper: requests → Playwright → Scrapfly with comprehensive tracking
    """
    global enhanced_scraping_stats
    
    # Check limits first
    if not _check_scraping_limit(category, keyword):
        if category == "company":
            return None, f"Company limit reached ({scraping_stats['company_scraped']}/{scraping_stats['limits']['company']})"
        elif category == "industry":
            keyword_count = scraping_stats["industry_scraped_by_keyword"].get(keyword, 0)
            return None, f"Industry keyword '{keyword}' limit reached ({keyword_count}/{scraping_stats['limits']['industry_per_keyword']})"
        elif category == "competitor":
            keyword_count = scraping_stats["competitor_scraped_by_keyword"].get(keyword, 0)
            return None, f"Competitor '{keyword}' limit reached ({keyword_count}/{scraping_stats['limits']['competitor_per_keyword']})"
    
    enhanced_scraping_stats["total_attempts"] += 1
    
    # TIER 1: Try standard requests-based scraping
    LOG.info(f"TIER 1 (Requests): Attempting {domain}")
    enhanced_scraping_stats["by_method"]["requests"]["attempts"] += 1
    
    content, error = safe_content_scraper(url, domain, scraped_domains)
    
    if content:
        enhanced_scraping_stats["requests_success"] += 1
        enhanced_scraping_stats["by_method"]["requests"]["successes"] += 1
        _update_scraping_stats(category, keyword, True)
        return content, f"TIER 1 SUCCESS: {len(content)} chars via requests"
    
    LOG.info(f"TIER 1 FAILED: {domain} - {error}")
    
    # TIER 2: Try Playwright fallback
    LOG.info(f"TIER 2 (Playwright): Attempting {domain}")
    enhanced_scraping_stats["by_method"]["playwright"]["attempts"] += 1
    
    playwright_content, playwright_error = extract_article_content_with_playwright(url, domain)
    
    if playwright_content:
        enhanced_scraping_stats["playwright_success"] += 1
        enhanced_scraping_stats["by_method"]["playwright"]["successes"] += 1
        _update_scraping_stats(category, keyword, True)
        return playwright_content, f"TIER 2 SUCCESS: {len(playwright_content)} chars via Playwright"
    
    LOG.info(f"TIER 2 FAILED: {domain} - {playwright_error}")
    
    # TIER 3: Try Scrapfly fallback (CHANGED from ScrapingBee)
    if SCRAPFLY_API_KEY:
        LOG.info(f"TIER 3 (Scrapfly): Attempting {domain}")
        
        scrapfly_content, scrapfly_error = scrape_with_scrapfly(url, domain)
        
        if scrapfly_content:
            enhanced_scraping_stats["scrapfly_success"] += 1
            _update_scraping_stats(category, keyword, True)
            return scrapfly_content, f"TIER 3 SUCCESS: {len(scrapfly_content)} chars via Scrapfly"
        
        LOG.info(f"TIER 3 FAILED: {domain} - {scrapfly_error}")
    else:
        LOG.info(f"TIER 3 SKIPPED: Scrapfly API key not configured")
    
    # All tiers failed
    enhanced_scraping_stats["total_failures"] += 1
    return None, f"ALL TIERS FAILED - Requests: {error}, Playwright: {playwright_error}, Scrapfly: {scrapfly_error if SCRAPFLY_API_KEY else 'not configured'}"

def log_enhanced_scraping_stats():
    """Log comprehensive scraping statistics across all methods"""
    total = enhanced_scraping_stats["total_attempts"]
    if total == 0:
        LOG.info("ENHANCED SCRAPING: No attempts made")
        return
    
    requests_success = enhanced_scraping_stats["requests_success"]
    playwright_success = enhanced_scraping_stats["playwright_success"]
    scrapfly_success = enhanced_scraping_stats["scrapfly_success"]  # ADD THIS
    scrapingbee_success = enhanced_scraping_stats["scrapingbee_success"]
    total_success = requests_success + playwright_success + scrapfly_success + scrapingbee_success
    
    overall_rate = (total_success / total) * 100
    
    LOG.info("=== ENHANCED SCRAPING FINAL STATS ===")
    LOG.info(f"OVERALL SUCCESS: {overall_rate:.1f}% ({total_success}/{total})")
    LOG.info(f"  TIER 1 (Requests): {requests_success} successes / {enhanced_scraping_stats['by_method']['requests']['attempts']} attempts")
    LOG.info(f"  TIER 2 (Playwright): {playwright_success} successes / {enhanced_scraping_stats['by_method']['playwright']['attempts']} attempts")
    LOG.info(f"  TIER 3 (Scrapfly): {scrapfly_success} successes / {enhanced_scraping_stats['by_method']['scrapfly']['attempts']} attempts")
    
    # Calculate tier-specific success rates
    for method, stats in enhanced_scraping_stats["by_method"].items():
        if stats["attempts"] > 0:
            rate = (stats["successes"] / stats["attempts"]) * 100
            LOG.info(f"  {method.upper()} RATE: {rate:.1f}%")

def log_scrapingbee_stats():
    """Enhanced ScrapingBee statistics logging"""
    if scrapingbee_stats["requests_made"] == 0:
        LOG.info("SCRAPINGBEE: No requests made this run")
        return
    
    success_rate = (scrapingbee_stats["successful"] / scrapingbee_stats["requests_made"]) * 100
    LOG.info(f"SCRAPINGBEE FINAL: {success_rate:.1f}% success rate ({scrapingbee_stats['successful']}/{scrapingbee_stats['requests_made']})")
    LOG.info(f"SCRAPINGBEE COST: ${scrapingbee_stats['cost_estimate']:.3f} estimated")
    
    if scrapingbee_stats["failed"] > 0:
        LOG.warning(f"SCRAPINGBEE: {scrapingbee_stats['failed']} requests failed")
    
    # Log top performing and failing domains
    successful_domains = [(domain, stats) for domain, stats in scrapingbee_stats["by_domain"].items() if stats["successes"] > 0]
    failed_domains = [(domain, stats) for domain, stats in scrapingbee_stats["by_domain"].items() if stats["successes"] == 0 and stats["attempts"] > 0]
    
    if successful_domains:
        LOG.info(f"SCRAPINGBEE SUCCESS DOMAINS: {len(successful_domains)} domains working")
    if failed_domains:
        LOG.info(f"SCRAPINGBEE FAILED DOMAINS: {len(failed_domains)} domains blocked/failed")

def log_scrapfly_stats():
    """Scrapfly statistics logging"""
    if scrapfly_stats["requests_made"] == 0:
        LOG.info("SCRAPFLY: No requests made this run")
        return
    
    success_rate = (scrapfly_stats["successful"] / scrapfly_stats["requests_made"]) * 100
    LOG.info(f"SCRAPFLY FINAL: {success_rate:.1f}% success rate ({scrapfly_stats['successful']}/{scrapfly_stats['requests_made']})")
    LOG.info(f"SCRAPFLY COST: ${scrapfly_stats['cost_estimate']:.3f} estimated")
    
    if scrapfly_stats["failed"] > 0:
        LOG.warning(f"SCRAPFLY: {scrapfly_stats['failed']} requests failed")
    
    # Log top performing and failing domains
    successful_domains = [(domain, stats) for domain, stats in scrapfly_stats["by_domain"].items() if stats["successes"] > 0]
    failed_domains = [(domain, stats) for domain, stats in scrapfly_stats["by_domain"].items() if stats["successes"] == 0 and stats["attempts"] > 0]
    
    if successful_domains:
        LOG.info(f"SCRAPFLY SUCCESS DOMAINS: {len(successful_domains)} domains working")
    if failed_domains:
        LOG.info(f"SCRAPFLY FAILED DOMAINS: {len(failed_domains)} domains blocked/failed")

def reset_enhanced_scraping_stats():
    """Reset enhanced scraping stats for new run"""
    global enhanced_scraping_stats, scrapfly_stats, scrapingbee_stats
    enhanced_scraping_stats = {
        "total_attempts": 0,
        "requests_success": 0,
        "playwright_success": 0,
        "scrapfly_success": 0,  # ADD THIS
        "scrapingbee_success": 0,
        "total_failures": 0,
        "by_method": {
            "requests": {"attempts": 0, "successes": 0},
            "playwright": {"attempts": 0, "successes": 0},
            "scrapfly": {"attempts": 0, "successes": 0},  # ADD THIS
            "scrapingbee": {"attempts": 0, "successes": 0}
        }
    }
    # Reset Scrapfly stats
    scrapfly_stats = {
        "requests_made": 0,
        "successful": 0,
        "failed": 0,
        "cost_estimate": 0.0,
        "by_domain": defaultdict(lambda: {"attempts": 0, "successes": 0})
    }
    # Keep ScrapingBee stats reset as well
    scrapingbee_stats = {
        "requests_made": 0,
        "successful": 0,
        "failed": 0,
        "cost_estimate": 0.0,
        "by_domain": defaultdict(lambda: {"attempts": 0, "successes": 0})
    }

def scrape_and_analyze_article_3tier(article: Dict, category: str, metadata: Dict, analysis_ticker: str) -> bool:
    """Scrape content and run AI analysis for a single article from specific ticker's perspective"""
    try:
        article_id = article["id"]
        resolved_url = article.get("resolved_url") or article.get("url")
        domain = article.get("domain", "unknown")
        title = article.get("title", "")
        url_hash = article.get("url_hash")
        
        # Check if this URL already has analysis from this ticker's perspective
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT scraped_content, ai_summary
                FROM found_url 
                WHERE url_hash = %s AND ai_analysis_ticker = %s
            """, (url_hash, analysis_ticker))
            existing_analysis = cur.fetchone()
        
        if (existing_analysis and 
            existing_analysis["scraped_content"] and 
            existing_analysis["ai_summary"]):
            
            LOG.info(f"REUSING ANALYSIS: Article {article_id} already analyzed from {analysis_ticker}'s perspective")
            return True
        
        # Get keyword for limit tracking
        if category == "company":
            keyword = analysis_ticker
        elif category == "competitor":
            keyword = article.get("competitor_ticker") or article.get("search_keyword", "unknown")
        else:
            keyword = article.get("search_keyword", "unknown")
        
        # Initialize content variables
        scraped_content = None
        scraping_error = None
        content_scraped_at = None
        scraping_failed = False
        ai_summary = None
        
        if resolved_url and resolved_url.startswith(('http://', 'https://')):
            scrape_domain = normalize_domain(urlparse(resolved_url).netloc.lower())
            
            if scrape_domain not in PAYWALL_DOMAINS and scrape_domain not in PROBLEMATIC_SCRAPE_DOMAINS:
                content, status = safe_content_scraper_with_3tier_fallback(
                    resolved_url, scrape_domain, category, keyword, set()
                )
                
                if content:
                    scraped_content = clean_null_bytes(content)
                    content_scraped_at = datetime.now(timezone.utc)
                    # Generate AI summary after successful scraping
                    ai_summary = generate_ai_individual_summary(scraped_content, title, analysis_ticker)
                    if ai_summary:
                        ai_summary = clean_null_bytes(ai_summary)
                        LOG.info(f"AI SUMMARY GENERATED for {analysis_ticker}: {len(ai_summary)} chars - '{ai_summary[:100]}...'")
                    else:
                        LOG.warning(f"AI SUMMARY FAILED for {analysis_ticker}: {title[:50]}...")
                        ai_summary = None
                else:
                    scraping_failed = True
                    scraping_error = clean_null_bytes(status or "")
                    return False
        
        # Store with analysis_ticker perspective
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO found_url (
                    url, resolved_url, url_hash, title, description, ticker, domain,
                    published_at, category, search_keyword, 
                    scraped_content, content_scraped_at, scraping_failed, scraping_error,
                    ai_summary, ai_analysis_ticker, competitor_ticker, triage_priority
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url_hash, ticker, COALESCE(ai_analysis_ticker, '')) 
                DO UPDATE SET
                    scraped_content = EXCLUDED.scraped_content,
                    content_scraped_at = EXCLUDED.content_scraped_at,
                    ai_summary = EXCLUDED.ai_summary,
                    updated_at = NOW()
            """, (
                article.get("url"), resolved_url, url_hash, title, 
                article.get("description"), analysis_ticker, domain, 
                article.get("published_at"), category, article.get("search_keyword"),
                scraped_content, content_scraped_at, scraping_failed, 
                clean_null_bytes(scraping_error or ""), clean_null_bytes(ai_summary or ""),
                analysis_ticker, article.get("competitor_ticker"),
                normalize_priority_to_int(article.get("triage_priority", 2))
            ))
        
        LOG.info(f"TICKER-SPECIFIC ANALYSIS: {title[:50]}... analyzed from {analysis_ticker}'s perspective with AI summary")
        return scraped_content is not None
        
    except Exception as e:
        LOG.error(f"Failed to analyze article {article.get('id')} from {analysis_ticker}'s perspective: {e}")
        return False

def _update_scraping_stats(category: str, keyword: str, success: bool):
    """Helper to update scraping statistics"""
    global scraping_stats
    
    if success:
        scraping_stats["successful_scrapes"] += 1
        
        if category == "company":
            scraping_stats["company_scraped"] += 1
            LOG.info(f"SCRAPING SUCCESS: Company {scraping_stats['company_scraped']}/{scraping_stats['limits']['company']} | Total: {scraping_stats['successful_scrapes']}")
        
        elif category == "industry":
            if keyword not in scraping_stats["industry_scraped_by_keyword"]:
                scraping_stats["industry_scraped_by_keyword"][keyword] = 0
            scraping_stats["industry_scraped_by_keyword"][keyword] += 1
            keyword_count = scraping_stats["industry_scraped_by_keyword"][keyword]
            LOG.info(f"SCRAPING SUCCESS: Industry '{keyword}' {keyword_count}/{scraping_stats['limits']['industry_per_keyword']} | Total: {scraping_stats['successful_scrapes']}")
        
        elif category == "competitor":
            # Use competitor_ticker as consolidation key
            if keyword not in scraping_stats["competitor_scraped_by_keyword"]:
                scraping_stats["competitor_scraped_by_keyword"][keyword] = 0
            scraping_stats["competitor_scraped_by_keyword"][keyword] += 1
            keyword_count = scraping_stats["competitor_scraped_by_keyword"][keyword]
            LOG.info(f"SCRAPING SUCCESS: Competitor '{keyword}' {keyword_count}/{scraping_stats['limits']['competitor_per_keyword']} | Total: {scraping_stats['successful_scrapes']}")
    
def _check_scraping_limit(category: str, keyword: str) -> bool:
    """Check if we can scrape more articles for this category/keyword"""
    global scraping_stats
    
    if category == "company":
        return scraping_stats["company_scraped"] < scraping_stats["limits"]["company"]
    
    elif category == "industry":
        keyword_count = scraping_stats["industry_scraped_by_keyword"].get(keyword, 0)
        return keyword_count < scraping_stats["limits"]["industry_per_keyword"]
    
    elif category == "competitor":
        # Use competitor_ticker as consolidation key
        keyword_count = scraping_stats["competitor_scraped_by_keyword"].get(keyword, 0)
        return keyword_count < scraping_stats["limits"]["competitor_per_keyword"]
    
    return False

def calculate_dynamic_scraping_limits(ticker: str) -> Dict[str, int]:
    """Calculate dynamic scraping limits based on actual keywords/competitors"""
    config = get_ticker_config(ticker)
    if not config:
        return {"company": 20, "industry_total": 0, "competitor_total": 0}
    
    # Get actual counts
    industry_keywords = config.get("industry_keywords", [])
    competitors = config.get("competitors", [])
    
    # Calculate totals: 5 articles per keyword/competitor
    industry_total = len(industry_keywords) * 5
    competitor_total = len(competitors) * 5
    
    LOG.info(f"DYNAMIC SCRAPING LIMITS for {ticker}:")
    LOG.info(f"  Company: 20")
    LOG.info(f"  Industry: {len(industry_keywords)} keywords × 5 = {industry_total}")
    LOG.info(f"  Competitor: {len(competitors)} competitors × 5 = {competitor_total}")
    LOG.info(f"  TOTAL: {20 + industry_total + competitor_total} articles max")
    
    return {
        "company": 20,
        "industry_total": industry_total,
        "competitor_total": competitor_total,
        "total_possible": 20 + industry_total + competitor_total
    }

# Update the ingest function to include AI summary generation
def ingest_feed_with_content_scraping(feed: Dict, category: str = "company", keywords: List[str] = None, 
                                       enable_ai_scoring: bool = True, max_ai_articles: int = None) -> Dict[str, int]:
    """
    Enhanced feed processing with AI summaries for scraped content and comprehensive Yahoo Finance resolution
    Flow: Check scraping limits -> Scrape resolved URLs -> AI analysis only on successful scrapes
    """
    global scraping_stats
    
    stats = {
        "processed": 0, "inserted": 0, "duplicates": 0, "blocked_spam": 0, "blocked_non_latin": 0,
        "content_scraped": 0, "content_failed": 0, "scraping_skipped": 0, "ai_reanalyzed": 0,
        "ai_scored": 0, "basic_scored": 0, "ai_summaries_generated": 0, "blocked_insider_trading": 0
    }
    
    scraped_domains = set()
    ai_processed_count = 0
    
    # Get the keyword for this feed (for limit tracking)
    feed_keyword = feed.get("search_keyword", "unknown")
    
    try:
        parsed = feedparser.parse(feed["url"])
        LOG.info(f"Processing feed [{category}]: {feed['name']} - {len(parsed.entries)} entries (AI: {'enabled' if enable_ai_scoring else 'disabled'})")
        
        # Sort entries by publication date (newest first) if available
        entries_with_dates = []
        for entry in parsed.entries:
            pub_date = None
            if hasattr(entry, "published_parsed"):
                pub_date = parse_datetime(entry.published_parsed)
            entries_with_dates.append((entry, pub_date))
        
        entries_with_dates.sort(key=lambda x: (x[1] or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        
        for entry, _ in entries_with_dates:
            stats["processed"] += 1
            
            url = getattr(entry, "link", None)
            title = getattr(entry, "title", "") or "No Title"
            raw_description = getattr(entry, "summary", "") if hasattr(entry, "summary") else ""
            
            # Filter description
            description = ""
            if raw_description and is_description_valuable(title, raw_description):
                description = raw_description
            
            # Quick spam checks
            if not url or contains_non_latin_script(title):
                stats["blocked_non_latin"] += 1
                continue
                
            if any(spam in title.lower() for spam in ["marketbeat", "newser", "khodrobank"]):
                stats["blocked_spam"] += 1
                continue
            
            # NEW: Insider trading check - skip these articles entirely
            if is_insider_trading_article(title):
                stats["blocked_insider_trading"] += 1
                LOG.debug(f"INSIDER TRADING BLOCKED: {title[:50]}...")
                continue

            # COMPREHENSIVE URL RESOLUTION
            final_resolved_url = None
            final_domain = None
            final_source_url = None
            
            # Step 1: Basic resolution using domain resolver
            resolved_url, domain, source_url = domain_resolver.resolve_url_and_domain(url, title)
            
            if not resolved_url or not domain:
                stats["blocked_spam"] += 1
                continue
            
            # Step 2: Check if this is a Yahoo Finance URL (direct or redirected)
            is_yahoo_finance = any(yahoo_domain in resolved_url for yahoo_domain in [
                "finance.yahoo.com", "ca.finance.yahoo.com", "uk.finance.yahoo.com"
            ])
            
            # Step 3: Handle Yahoo Finance resolution
            if is_yahoo_finance:
                LOG.info(f"YAHOO RESOLUTION: Attempting to resolve {resolved_url}")
                yahoo_original = extract_yahoo_finance_source_optimized(resolved_url)
                
                if yahoo_original:
                    # Successfully resolved Yahoo Finance to original source
                    final_resolved_url = yahoo_original
                    final_domain = normalize_domain(urlparse(yahoo_original).netloc.lower())
                    final_source_url = resolved_url  # Yahoo URL becomes the source
                    LOG.info(f"YAHOO SUCCESS: {resolved_url} → {yahoo_original}")
                else:
                    # Couldn't resolve Yahoo Finance, use Yahoo URL
                    final_resolved_url = resolved_url
                    final_domain = domain
                    final_source_url = source_url
                    LOG.warning(f"YAHOO FAILED: Could not resolve {resolved_url}")
            else:
                # Not a Yahoo Finance URL, use standard resolution
                final_resolved_url = resolved_url
                final_domain = domain
                final_source_url = source_url
            
            # Step 4: Generate hash for deduplication based on FINAL resolved URL
            url_hash = get_url_hash(url, final_resolved_url)
            
            try:
                with db() as conn, conn.cursor() as cur:
                    # Check for duplicates
                    cur.execute("""
                        SELECT id, ai_impact, ai_reasoning, quality_score 
                        FROM found_url 
                        WHERE url_hash = %s
                    """, (url_hash,))
                    existing_article = cur.fetchone()
                    
                    if existing_article:
                        # Handle re-analysis if needed (existing logic)
                        if (enable_ai_scoring and max_ai_articles and ai_processed_count < max_ai_articles and
                            (existing_article["ai_impact"] is None or existing_article["ai_reasoning"] is None)):
                            
                            quality_score, ai_impact, ai_reasoning, components = calculate_quality_score(
                                title=title, domain=final_domain, ticker=feed["ticker"],
                                description=description, category=category, keywords=keywords
                            )
                            
                            # Update with components
                            source_tier = components.get('source_tier') if components else None
                            event_multiplier = components.get('event_multiplier') if components else None
                            event_multiplier_reason = components.get('event_multiplier_reason') if components else None
                            relevance_boost = components.get('relevance_boost') if components else None
                            relevance_boost_reason = components.get('relevance_boost_reason') if components else None
                            numeric_bonus = components.get('numeric_bonus') if components else None
                            penalty_multiplier = components.get('penalty_multiplier') if components else None
                            penalty_reason = components.get('penalty_reason') if components else None
                            
                            cur.execute("""
                                UPDATE found_url 
                                SET quality_score = %s, ai_impact = %s, ai_reasoning = %s,
                                    source_tier = %s, event_multiplier = %s, event_multiplier_reason = %s,
                                    relevance_boost = %s, relevance_boost_reason = %s, numeric_bonus = %s,
                                    penalty_multiplier = %s, penalty_reason = %s
                                WHERE id = %s
                            """, (
                                quality_score, ai_impact, ai_reasoning,
                                source_tier, event_multiplier, event_multiplier_reason,
                                relevance_boost, relevance_boost_reason, numeric_bonus,
                                penalty_multiplier, penalty_reason,
                                existing_article["id"]
                            ))
                            
                            if cur.rowcount > 0:
                                stats["ai_reanalyzed"] += 1
                                ai_processed_count += 1
                        
                        stats["duplicates"] += 1
                        continue
                    
                    # Parse publish date
                    published_at = None
                    if hasattr(entry, "published_parsed"):
                        published_at = parse_datetime(entry.published_parsed)
                    
                    # Initialize content scraping variables
                    scraped_content = None
                    scraping_error = None
                    content_scraped_at = None
                    scraping_failed = False
                    ai_summary = None
                    should_use_ai = False
                    
                    # Attempt content scraping (respects 20/5/5 limits)
                    if final_resolved_url and final_resolved_url.startswith(('http://', 'https://')):
                        scrape_domain = normalize_domain(urlparse(final_resolved_url).netloc.lower())
                        
                        if scrape_domain in PAYWALL_DOMAINS:
                            stats["scraping_skipped"] += 1
                            LOG.info(f"Skipping paywall domain: {scrape_domain}")
                        else:
                            # Check scraping limits and attempt scraping
                            content, status = safe_content_scraper_with_3tier_fallback(
                                final_resolved_url, scrape_domain, category, feed_keyword, scraped_domains
                            )
                            
                            if content:
                                # Scraping successful - enable AI processing
                                scraped_content = content
                                content_scraped_at = datetime.now(timezone.utc)
                                stats["content_scraped"] += 1
                                should_use_ai = True
                                
                                # Generate AI summary from scraped content
                                ai_summary = generate_ai_individual_summary(scraped_content, title, feed["ticker"])
                                if ai_summary:
                                    stats["ai_summaries_generated"] += 1
                            else:
                                scraping_failed = True
                                scraping_error = status
                                stats["content_failed"] += 1
                    else:
                        stats["scraping_skipped"] += 1
                    
                    # Calculate quality score (AI only runs if scraping was successful)
                    if should_use_ai:
                        quality_score, ai_impact, ai_reasoning, components = calculate_quality_score(
                            title=title, domain=final_domain, ticker=feed["ticker"],
                            description=description, category=category, keywords=keywords
                        )
                        
                        source_tier = components.get('source_tier') if components else None
                        event_multiplier = components.get('event_multiplier') if components else None
                        event_multiplier_reason = components.get('event_multiplier_reason') if components else None
                        relevance_boost = components.get('relevance_boost') if components else None
                        relevance_boost_reason = components.get('relevance_boost_reason') if components else None
                        numeric_bonus = components.get('numeric_bonus') if components else None
                        penalty_multiplier = components.get('penalty_multiplier') if components else None
                        penalty_reason = components.get('penalty_reason') if components else None
                        
                        ai_processed_count += 1
                        stats["ai_scored"] += 1
                    else:
                        quality_score = _fallback_quality_score(title, final_domain, feed["ticker"], description, keywords)
                        ai_impact = None
                        ai_reasoning = None
                        source_tier = None
                        event_multiplier = None
                        event_multiplier_reason = None
                        relevance_boost = None
                        relevance_boost_reason = None
                        numeric_bonus = None
                        penalty_multiplier = None
                        penalty_reason = None
                        stats["basic_scored"] += 1
                    
                    display_content = scraped_content if scraped_content else description
                    
                    # Insert article with AI summary and comprehensive resolution data
                    cur.execute("""
                        INSERT INTO found_url (
                            url, resolved_url, url_hash, title, description,
                            feed_id, ticker, domain, quality_score, published_at,
                            category, search_keyword, original_source_url,
                            scraped_content, content_scraped_at, scraping_failed, scraping_error,
                            ai_impact, ai_reasoning, ai_summary,
                            source_tier, event_multiplier, event_multiplier_reason,
                            relevance_boost, relevance_boost_reason, numeric_bonus,
                            penalty_multiplier, penalty_reason
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        url, final_resolved_url, url_hash, title, display_content,
                        feed["id"], feed["ticker"], final_domain, quality_score, published_at,
                        category, feed.get("search_keyword"), final_source_url,
                        scraped_content, content_scraped_at, scraping_failed, scraping_error,
                        ai_impact, ai_reasoning, ai_summary,
                        source_tier, event_multiplier, event_multiplier_reason,
                        relevance_boost, relevance_boost_reason, numeric_bonus,
                        penalty_multiplier, penalty_reason
                    ))
                    
                    if cur.fetchone():
                        stats["inserted"] += 1
                        processing_type = "AI analysis" if should_use_ai else "basic processing"
                        content_info = f"with content + summary" if scraped_content and ai_summary else f"with content" if scraped_content else "no content"
                        
                        # Enhanced logging
                        resolution_info = ""
                        if is_yahoo_finance and yahoo_original:
                            resolution_info = f" (Yahoo→{get_or_create_formal_domain_name(final_domain)})"
                        elif final_source_url:
                            resolution_info = f" (via {get_or_create_formal_domain_name(normalize_domain(urlparse(final_source_url).netloc))})"
                        
                        LOG.info(f"Inserted [{category}] from {get_or_create_formal_domain_name(final_domain)}: {title[:60]}... ({processing_type}, {content_info}){resolution_info}")
                        
            except Exception as e:
                LOG.error(f"Database error for '{title[:50]}': {e}")
                continue
                
    except Exception as e:
        LOG.error(f"Feed processing error for {feed['name']}: {e}")
    
    # At the very end of the function, update the return statement:
    return {
        "processed": stats["processed"],
        "inserted": stats["inserted"],
        "duplicates": stats["duplicates"], 
        "blocked_spam": stats["blocked_spam"],
        "blocked_non_latin": stats["blocked_non_latin"],
        "content_scraped": stats["content_scraped"],
        "content_failed": stats["content_failed"],
        "scraping_skipped": stats["scraping_skipped"],
        "ai_reanalyzed": stats["ai_reanalyzed"],
        "ai_scored": stats["ai_scored"],
        "basic_scored": stats["basic_scored"],
        "ai_summaries_generated": stats["ai_summaries_generated"],
        "blocked_insider_trading": stats["blocked_insider_trading"]  # ADD THIS LINE
    }

def ingest_feed_basic_only(feed: Dict) -> Dict[str, int]:
    """Basic feed ingestion with FIXED deduplication count tracking"""
    stats = {
        "processed": 0, "inserted": 0, "duplicates": 0, "blocked_spam": 0, 
        "blocked_non_latin": 0, "limit_reached": 0, "blocked_insider_trading": 0,
        "yahoo_rejected": 0
    }
    
    category = feed.get("category", "company")
    
    # Use competitor_ticker for competitor feeds, search_keyword for others
    if category == "competitor":
        feed_keyword = feed.get("competitor_ticker", "unknown")
    else:
        feed_keyword = feed.get("search_keyword", "unknown")
    
    # Track processed URLs within this feed run to avoid counting duplicates
    processed_hashes = set()
    
    try:
        parsed = feedparser.parse(feed["url"])
        
        # Check if feed parsed successfully
        if not hasattr(parsed, 'entries') or not parsed.entries:
            LOG.warning(f"No entries found in feed: {feed['name']}")
            return stats
        
        # Sort entries by publication date (newest first) if available
        entries_with_dates = []
        for entry in parsed.entries:
            pub_date = None
            if hasattr(entry, "published_parsed"):
                pub_date = parse_datetime(entry.published_parsed)
            entries_with_dates.append((entry, pub_date))
        
        entries_with_dates.sort(key=lambda x: (x[1] or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        
        for entry, _ in entries_with_dates:
            try:
                stats["processed"] += 1
                
                url = getattr(entry, "link", None)
                title = getattr(entry, "title", "") or "No Title"
                raw_description = getattr(entry, "summary", "") if hasattr(entry, "summary") else ""
                
                # Filter description
                description = ""
                if raw_description and is_description_valuable(title, raw_description):
                    description = raw_description
                
                # Quick spam checks
                if not url or contains_non_latin_script(title):
                    stats["blocked_non_latin"] += 1
                    continue
                    
                if any(spam in title.lower() for spam in ["marketbeat", "newser", "khodrobank"]):
                    stats["blocked_spam"] += 1
                    continue
                
                # NEW: Insider trading check - skip these articles entirely
                if is_insider_trading_article(title):
                    stats["blocked_insider_trading"] += 1
                    LOG.debug(f"INSIDER TRADING BLOCKED: {title[:50]}...")
                    continue
                
                # COMPREHENSIVE URL RESOLUTION
                try:
                    resolved_url, domain, source_url = domain_resolver.resolve_url_and_domain(url, title)
                    
                    # CHECK FOR YAHOO REJECTION
                    if not resolved_url or not domain:
                        if "yahoo.com" in url.lower():
                            stats["yahoo_rejected"] += 1
                            LOG.debug(f"YAHOO REJECTED: {title[:50]}...")
                        else:
                            stats["blocked_spam"] += 1
                        continue
                except Exception as e:
                    LOG.warning(f"URL resolution failed for {url}: {e}")
                    stats["blocked_spam"] += 1
                    continue
                
                # Handle Yahoo Finance resolution
                final_resolved_url = resolved_url
                final_domain = domain
                final_source_url = source_url
                
                # Check if this is a Yahoo Finance URL
                is_yahoo_finance = any(yahoo_domain in resolved_url for yahoo_domain in [
                    "finance.yahoo.com", "ca.finance.yahoo.com", "uk.finance.yahoo.com"
                ])
                
                if is_yahoo_finance:
                    try:
                        yahoo_original = extract_yahoo_finance_source_optimized(resolved_url)
                        if yahoo_original:
                            final_resolved_url = yahoo_original
                            final_domain = normalize_domain(urlparse(yahoo_original).netloc.lower())
                            final_source_url = resolved_url
                    except Exception as e:
                        LOG.warning(f"Yahoo source extraction failed for {resolved_url}: {e}")
                
                # Generate hash for deduplication
                url_hash = get_url_hash(url, final_resolved_url)
                
                # CHECK FOR DUPLICATES WITHIN THIS RUN FIRST
                if url_hash in processed_hashes:
                    stats["duplicates"] += 1
                    LOG.debug(f"FEED DUPLICATE SKIPPED: {title[:50]}... (same URL within feed)")
                    continue
                
                # Add to processed set
                processed_hashes.add(url_hash)
                
                # NOW check ingestion limits with FIXED count logic
                with db() as conn, conn.cursor() as cur:
                    try:
                        # Check if URL already exists in database for this ticker
                        cur.execute("""
                            SELECT id FROM found_url 
                            WHERE url_hash = %s AND ticker = %s AND COALESCE(ai_analysis_ticker, '') = ''
                        """, (url_hash, feed["ticker"]))
                        if cur.fetchone():
                            stats["duplicates"] += 1
                            LOG.debug(f"DATABASE DUPLICATE SKIPPED: {title[:50]}... (already in database)")
                            continue
                        
                        # FIXED: Count existing unique URLs for this category/keyword combination
                        # This ensures we count TOTAL unique URLs (existing + new)
                        if category == "company":
                            cur.execute("""
                                SELECT COUNT(DISTINCT url_hash) as count FROM found_url 
                                WHERE ticker = %s AND category = 'company'
                                AND COALESCE(ai_analysis_ticker, '') = ''
                            """, (feed["ticker"],))
                        elif category == "industry":
                            cur.execute("""
                                SELECT COUNT(DISTINCT url_hash) as count FROM found_url 
                                WHERE ticker = %s AND category = 'industry' AND search_keyword = %s
                                AND COALESCE(ai_analysis_ticker, '') = ''
                            """, (feed["ticker"], feed_keyword))
                        elif category == "competitor":
                            cur.execute("""
                                SELECT COUNT(DISTINCT url_hash) as count FROM found_url 
                                WHERE ticker = %s AND category = 'competitor' AND competitor_ticker = %s
                                AND COALESCE(ai_analysis_ticker, '') = ''
                            """, (feed["ticker"], feed_keyword))
                        
                        result = cur.fetchone()
                        existing_count = result["count"] if result and result["count"] is not None else 0
                        
                        # FIXED: Check limits based on existing count from database
                        # Add 1 to existing count to account for this new article
                        projected_count = existing_count + 1
                        
                        # Check against limits
                        if category == "company" and projected_count > ingestion_stats["limits"]["company"]:
                            stats["limit_reached"] += 1
                            LOG.info(f"COMPANY LIMIT REACHED: {existing_count} existing + 1 new = {projected_count} > {ingestion_stats['limits']['company']}")
                            break
                        elif category == "industry" and projected_count > ingestion_stats["limits"]["industry_per_keyword"]:
                            stats["limit_reached"] += 1
                            LOG.info(f"INDUSTRY LIMIT REACHED for '{feed_keyword}': {existing_count} existing + 1 new = {projected_count} > {ingestion_stats['limits']['industry_per_keyword']}")
                            break
                        elif category == "competitor" and projected_count > ingestion_stats["limits"]["competitor_per_keyword"]:
                            stats["limit_reached"] += 1
                            LOG.info(f"COMPETITOR LIMIT REACHED for '{feed_keyword}': {existing_count} existing + 1 new = {projected_count} > {ingestion_stats['limits']['competitor_per_keyword']}")
                            break
                        
                        # COUNT THIS NEW UNIQUE URL for tracking
                        _update_ingestion_stats(category, feed_keyword)
                        
                        # Parse publish date
                        published_at = None
                        if hasattr(entry, "published_parsed"):
                            published_at = parse_datetime(entry.published_parsed)
                        
                        # Use basic scoring
                        domain_tier = _get_domain_tier(final_domain, title, description)
                        basic_quality_score = 50.0 + (domain_tier - 0.5) * 20
                        
                        # Add ticker mention bonus
                        if feed["ticker"].upper() in title.upper():
                            basic_quality_score += 10
                        
                        # Clamp to reasonable range
                        basic_quality_score = max(20.0, min(80.0, basic_quality_score))
                        
                        display_content = description
                        
                        # Clean all text fields to remove NULL bytes
                        clean_url = clean_null_bytes(url or "")
                        clean_resolved_url = clean_null_bytes(final_resolved_url or "")
                        clean_title = clean_null_bytes(title or "")
                        clean_description = clean_null_bytes(display_content or "")
                        clean_search_keyword = clean_null_bytes(feed.get("search_keyword") or "")
                        clean_source_url = clean_null_bytes(final_source_url or "")
                        clean_competitor_ticker = clean_null_bytes(feed.get("competitor_ticker") or "")
                        
                        # Insert with proper constraint handling
                        cur.execute("""
                            INSERT INTO found_url (
                                url, resolved_url, url_hash, title, description,
                                feed_id, ticker, domain, quality_score, published_at,
                                category, search_keyword, original_source_url,
                                competitor_ticker, ai_analysis_ticker
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (url_hash, ticker, COALESCE(ai_analysis_ticker, '')) 
                            DO UPDATE SET
                                updated_at = NOW()
                            RETURNING id
                        """, (
                            clean_url, clean_resolved_url, url_hash, clean_title, clean_description,
                            feed["id"], feed["ticker"], final_domain, basic_quality_score, published_at,
                            category, clean_search_keyword, clean_source_url,
                            clean_competitor_ticker, ''
                        ))
                        
                        result = cur.fetchone()
                        if result:
                            stats["inserted"] += 1
                            LOG.info(f"INSERTED [{category}]: Total unique now: {projected_count}/{ingestion_stats['limits'][category if category == 'company' else f'{category}_per_keyword']} - {title[:50]}...")
                        else:
                            LOG.warning(f"Insert returned no result for: {title[:30]}")
                            
                    except Exception as db_e:
                        import traceback
                        LOG.error(f"Database error processing article '{title[:30]}...': {type(db_e).__name__}: {str(db_e)}")
                        LOG.error(f"Full traceback: {traceback.format_exc()}")
                        continue
                            
            except Exception as entry_e:
                LOG.error(f"Error processing feed entry: {entry_e}")
                continue
                        
    except Exception as e:
        LOG.error(f"Feed processing error for {feed['name']}: {e}")
    
    # At the very end of the function, update the return statement:
    return {
        "processed": stats["processed"],
        "inserted": stats["inserted"], 
        "duplicates": stats["duplicates"],
        "blocked_spam": stats["blocked_spam"],
        "blocked_non_latin": stats["blocked_non_latin"],
        "limit_reached": stats["limit_reached"],
        "blocked_insider_trading": stats["blocked_insider_trading"],
        "yahoo_rejected": stats["yahoo_rejected"]  # ADD THIS LINE
    }

def _check_ingestion_limit_with_existing_count(category: str, keyword: str, existing_count: int) -> bool:
    """Check if we can ingest more articles considering existing count in database"""
    global ingestion_stats
    
    if category == "company":
        total_count = existing_count + ingestion_stats["company_ingested"]
        return total_count < ingestion_stats["limits"]["company"]
    
    elif category == "industry":
        current_keyword_count = ingestion_stats["industry_ingested_by_keyword"].get(keyword, 0)
        total_count = existing_count + current_keyword_count
        return total_count < ingestion_stats["limits"]["industry_per_keyword"]
    
    elif category == "competitor":
        current_keyword_count = ingestion_stats["competitor_ingested_by_keyword"].get(keyword, 0)
        total_count = existing_count + current_keyword_count
        return total_count < ingestion_stats["limits"]["competitor_per_keyword"]
    
    return False

# Update the database schema to include ai_summary field
def update_schema_for_ai_summary():
    """Add AI summary field to found_url table"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE found_url ADD COLUMN IF NOT EXISTS ai_summary TEXT;
        """)

# Updated article formatting function
def _format_article_html_with_ai_summary(article: Dict, category: str, ticker_metadata_cache: Dict = None) -> str:
    """
    Enhanced article HTML formatting with AI summaries and proper left-side headers
    """
    import html
    
    # Format timestamp for individual articles
    if article["published_at"]:
        pub_date = format_timestamp_est(article["published_at"])
    else:
        pub_date = "N/A"
    
    original_title = article["title"] or "No Title"
    resolved_domain = article["domain"] or "unknown"
    
    # Determine source and clean title based on domain type
    if "news.google.com" in resolved_domain or resolved_domain == "google-news-unresolved":
        title_result = extract_source_from_title_smart(original_title)
        
        if title_result[0] is None:
            return ""
        
        title, extracted_source = title_result
        
        if extracted_source:
            display_source = get_or_create_formal_domain_name(extracted_source)
        else:
            display_source = "Google News"
    else:
        title = original_title
        display_source = get_or_create_formal_domain_name(resolved_domain)
    
    # Additional title cleanup
    title = re.sub(r'\s*\$[A-Z]+\s*-?\s*', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    
    # Determine the actual link URL
    link_url = article["resolved_url"] or article.get("original_source_url") or article["url"]
    
    # Get company name for company articles
    ticker = article.get("ticker", "")
    company_name = ticker
    if ticker_metadata_cache and ticker in ticker_metadata_cache:
        company_name = ticker_metadata_cache[ticker].get("company_name", ticker)
    
    # Build header badges - LEFT SIDE POSITIONING
    header_badges = []
    
    # 1. FIRST BADGE: Category-specific (LEFT SIDE)
    if category == "company":
        header_badges.append(f'<span class="company-name-badge">🎯 {company_name}</span>')
    elif category == "competitor":
        comp_name = get_competitor_display_name(article.get('search_keyword'), article.get('competitor_ticker'))
        header_badges.append(f'<span class="competitor-badge">🏢 {comp_name}</span>')
    elif category == "industry" and article.get('search_keyword'):
        header_badges.append(f'<span class="industry-badge">🏭 {article["search_keyword"]}</span>')
    
    # 2. SECOND BADGE: Source name
    header_badges.append(f'<span class="source-badge">{display_source}</span>')
    
    # 3. Analysis badge if both content and summary exist
    analyzed_html = ""
    if article.get('scraped_content') and article.get('ai_summary'):
        analyzed_html = f'<span class="analyzed-badge">Analyzed</span>'
        header_badges.append(analyzed_html)
    
    # AI Summary section - check for ai_summary field
    ai_summary_html = ""
    if article.get("ai_summary"):
        clean_summary = html.escape(article["ai_summary"].strip())
        ai_summary_html = f"<br><div class='ai-summary'><strong>📊 Analysis:</strong> {clean_summary}</div>"
    
    # Get description and format it (only if no AI summary)
    description_html = ""
    if not article.get("ai_summary") and article.get("description"):
        description = article["description"].strip()
        description = html.unescape(description)
        description = re.sub(r'<[^>]+>', '', description)
        description = re.sub(r'\s+', ' ', description).strip()
        
        if len(description) > 500:
            description = description[:500] + "..."
        
        description = html.escape(description)
        description_html = f"<br><div class='description'>{description}</div>"
    
    return f"""
    <div class='article {category}'>
        <div class='article-header'>
            {' '.join(header_badges)}
        </div>
        <div class='article-content'>
            <a href='{link_url}' target='_blank'>{title}</a>
            <span class='meta'> | {pub_date}</span>
            {ai_summary_html}
            {description_html}
        </div>
    </div>
    """

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
    """Get ticker configuration from database with complete metadata extraction"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT ticker, name, industry_keywords, competitors, ai_generated,
                   sector, industry, sub_industry, sector_profile, aliases_brands_assets
            FROM ticker_config
            WHERE ticker = %s AND active = TRUE
        """, (ticker,))
        return cur.fetchone()

def get_or_create_ticker_metadata(ticker: str, force_refresh: bool = False) -> Dict:
    """Wrapper for backward compatibility"""
    return ticker_manager.get_or_create_metadata(ticker, force_refresh)

def build_feed_urls(ticker: str, keywords: Dict) -> List[Dict]:
    """Wrapper for backward compatibility"""
    return feed_manager.create_feeds_for_ticker(ticker, keywords)
    
def upsert_feed(url: str, name: str, ticker: str, category: str = "company", 
                retain_days: int = 90, search_keyword: str = None, 
                competitor_ticker: str = None) -> int:
    """Simplified feed upsert with category storage"""
    LOG.info(f"DEBUG: Upserting feed - ticker: {ticker}, name: {name}, category: {category}, search_keyword: {search_keyword}")
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO source_feed (url, name, ticker, category, retain_days, active, search_keyword, competitor_ticker)
            VALUES (%s, %s, %s, %s, %s, TRUE, %s, %s)
            ON CONFLICT (url) DO UPDATE SET 
                name = EXCLUDED.name, 
                category = EXCLUDED.category,
                active = TRUE
            RETURNING id;
        """, (url, name, ticker, category, retain_days, search_keyword, competitor_ticker))
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
    """Enhanced Yahoo Finance source extraction - handles all Yahoo Finance domains and author pages"""
    try:
        # Expand the domain check to include regional Yahoo Finance
        if not any(domain in url for domain in ["finance.yahoo.com", "ca.finance.yahoo.com", "uk.finance.yahoo.com"]):
            return None
        
        # Skip Yahoo author pages and video files
        if any(skip_pattern in url for skip_pattern in ["/author/", "yahoo-finance-video", ".mp4", ".avi", ".mov"]):
            LOG.info(f"Skipping Yahoo author/video page: {url}")
            return None
            
        LOG.info(f"Extracting Yahoo Finance source from: {url}")
        
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        if response.status_code != 200:
            LOG.warning(f"HTTP {response.status_code} when fetching Yahoo URL: {url}")
            return None
        
        html_content = response.text
        LOG.debug(f"Yahoo page content length: {len(html_content)}")
        
        # Try the most reliable patterns in order of preference
        extraction_patterns = [
            # Pattern 1: Standard providerContentUrl
            r'"providerContentUrl"\s*:\s*"([^"]*)"',
            # Pattern 2: Alternative sourceUrl
            r'"sourceUrl"\s*:\s*"([^"]*)"',
            # Pattern 3: originalUrl
            r'"originalUrl"\s*:\s*"([^"]*)"',
            # Pattern 4: Escaped JSON patterns
            r'\\+"providerContentUrl\\+"\s*:\s*\\+"([^\\]*?)\\+"'
        ]
        
        for i, pattern in enumerate(extraction_patterns):
            matches = re.findall(pattern, html_content)
            LOG.debug(f"Pattern {i+1} found {len(matches)} matches")
            
            for match in matches:
                try:
                    # Try different unescaping methods
                    candidate_urls = []
                    
                    # Method 1: JSON unescape
                    try:
                        unescaped_url = json.loads(f'"{match}"')
                        candidate_urls.append(unescaped_url)
                    except json.JSONDecodeError:
                        pass
                    
                    # Method 2: Simple replace
                    simple_unescaped = match.replace('\\/', '/').replace('\\"', '"')
                    candidate_urls.append(simple_unescaped)
                    
                    # Method 3: Raw match
                    candidate_urls.append(match)
                    
                    # Test each candidate
                    for candidate_url in candidate_urls:
                        try:
                            parsed = urlparse(candidate_url)
                            if (parsed.scheme in ['http', 'https'] and 
                                parsed.netloc and 
                                len(candidate_url) > 20 and
                                'finance.yahoo.com' not in candidate_url and
                                'ca.finance.yahoo.com' not in candidate_url and
                                not candidate_url.startswith('//') and
                                '.' in parsed.netloc and
                                # Enhanced validation to exclude problematic URLs
                                not candidate_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.mp4', '.avi', '.mov')) and
                                not '/api/' in candidate_url.lower() and
                                not 'yimg.com' in candidate_url.lower() and
                                not '/author/' in candidate_url.lower() and
                                not 'yahoo-finance-video' in candidate_url.lower()):
                                
                                LOG.info(f"Successfully extracted Yahoo source: {candidate_url}")
                                return candidate_url
                        except Exception as e:
                            LOG.debug(f"URL validation failed for {candidate_url}: {e}")
                            continue
                            
                except Exception as e:
                    LOG.debug(f"Processing match failed: {e}")
                    continue
        
        # Enhanced fallback patterns that exclude author pages and videos
        fallback_patterns = [
            # Only match URLs that are likely to be news articles (not author pages)
            r'https://(?!finance\.yahoo\.com|ca\.finance\.yahoo\.com|s\.yimg\.com)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/(?!author/)[^\s"<>]*(?:news|article|story|press|finance|business)[^\s"<>]*',
            r'https://stockstory\.org/[^\s"<>]*',
        ]
        
        for pattern in fallback_patterns:
            matches = re.finditer(pattern, html_content)
            for match in matches:
                candidate_url = match.group(0).rstrip('",')
                try:
                    parsed = urlparse(candidate_url)
                    if (parsed.scheme and parsed.netloc and
                        # Additional validation for fallback URLs
                        not candidate_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.css', '.js', '.mp4', '.avi', '.mov')) and
                        not '/api/' in candidate_url.lower() and
                        not '/author/' in candidate_url.lower() and
                        not 'yahoo-finance-video' in candidate_url.lower() and
                        len(candidate_url) > 30):  # Minimum reasonable URL length
                        LOG.info(f"Fallback extraction successful: {candidate_url}")
                        return candidate_url
                except Exception:
                    continue
        
        LOG.warning(f"No original source found for Yahoo URL: {url}")
        return None
        
    except Exception as e:
        LOG.error(f"Yahoo Finance source extraction failed for {url}: {e}")
        return None

def extract_source_from_title_smart(title: str) -> Tuple[str, Optional[str]]:
    """Extract source from title with simple regex"""
    if not title:
        return title, None
    
    # Simple patterns for common formats
    patterns = [
        r'\s*-\s*([^-]+?)(?:\s*\([^)]*\.gov[^)]*\))?$',  # " - Source" (ignore .gov suffix)
        r'\s*-\s*([^-]+)$',  # " - Source"
        r'\s*\|\s*([^|]+)$'   # " | Source"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            source = match.group(1).strip()
            clean_title = re.sub(pattern, '', title).strip()
            
            # FIXED: Allow 3 characters (for MSN, CNN, etc.) and basic validation
            if 2 < len(source) < 50 and not any(spam in source.lower() for spam in ["marketbeat", "newser", "khodrobank"]):
                return clean_title, source
    
    return title, None
    
def _get_domain_tier(domain: str, title: str = "", description: str = "") -> float:
    """Get domain authority tier with potential upgrades from content - FIXED minimum tier"""
    if not domain:
        return 0.5
    
    normalized_domain = normalize_domain(domain)
    base_tier = DOMAIN_TIERS.get(normalized_domain, 0.5)  # This is correct
    
    # Upgrade tier if aggregator content reveals higher-quality source
    if any(agg in normalized_domain for agg in ["yahoo", "msn", "google"]):
        combined_text = f"{title} || {description}".lower()
        
        for pattern, tier in SOURCE_TIER_HINTS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return max(base_tier, tier)
    
    # FIXED: Ensure Tier C and below get 0.5 minimum for scoring
    # Tier C (0.4) and below should be treated as 0.5 for scoring purposes
    return max(base_tier, 0.5) if base_tier < 0.5 else base_tier

def _is_spam_content(title: str, domain: str, description: str = "") -> bool:
    """Check if content appears to be spam - reuse existing spam logic"""
    if not title:
        return True
    
    # Use existing spam domain check
    if any(spam in domain.lower() for spam in SPAM_DOMAINS):
        return True
    
    # Use existing non-Latin script check
    if contains_non_latin_script(title):
        return True
    
    # Additional spam indicators
    spam_phrases = [
        "marketbeat", "newser", "khodrobank", "should you buy", "top stocks to",
        "best stocks", "stock picks", "hot stocks", "penny stocks"
    ]
    
    combined_text = f"{title} {description}".lower()
    return any(phrase in combined_text for phrase in spam_phrases)

def calculate_quality_score(
    title: str, 
    domain: str, 
    ticker: str,
    description: str = "",
    category: str = "company",
    keywords: List[str] = None
) -> Tuple[float, Optional[str], Optional[str], Optional[Dict]]:
    """No individual article scoring - return None values"""
    return None, None, None, None

def _fallback_quality_score(title: str, domain: str, ticker: str, description: str = "", keywords: List[str] = None) -> float:
    """Fallback scoring when AI is unavailable - GUARANTEED NO AI CALLS"""
    base_score = 50.0
    
    # Domain tier bonus
    domain_tier = _get_domain_tier(domain, title, description)
    base_score += (domain_tier - 0.5) * 20  # Scale tier to score impact
    
    # Ticker mention bonus
    if ticker and ticker.upper() in title.upper():
        base_score += 10
    
    # Keyword relevance bonus
    if keywords:
        title_lower = title.lower()
        keyword_matches = sum(1 for kw in keywords if kw.lower() in title_lower)
        base_score += min(keyword_matches * 5, 15)
    
    return max(20.0, min(80.0, base_score))

def generate_ai_individual_summary(scraped_content: str, title: str, ticker: str, description: str = "") -> Optional[str]:
    """Generate enhanced hedge fund analyst summary with specific financial context and materiality assessment"""
    if not OPENAI_API_KEY or not scraped_content or len(scraped_content.strip()) < 200:
        LOG.warning(f"AI summary generation skipped - API key: {bool(OPENAI_API_KEY)}, content length: {len(scraped_content) if scraped_content else 0}")
        return None
    
    try:
        config = get_ticker_config(ticker)
        company_name = config.get("name", ticker) if config else ticker
        sector = config.get("sector", "") if config else ""
        
        prompt = f"""You are a hedge-fund analyst. Write a 5–7 sentence summary that is 100% EXTRACTIVE.

Rules (hard):
- Use ONLY facts explicitly present in the provided text (no outside knowledge, no inference, no estimates).
- Include every stated number, date, percentage, price, share count, unit, and named entity relevant to the main event.
- If a detail is not stated in the text, do NOT mention it or imply it.
- Forbidden words/hedges: likely, may, could, should, appears, expect, estimate, assume, infer, suggests, catalysts, risks, second-order.
- No bullets, no labels, no quotes, no headlines; 5–7 sentences; each ≤ 28 words.

If the text lacks enough information for 5 sentences, write only as many factual sentences as the text supports (minimum 3), still ≤ 28 words each.

Article Title: {title}
Content Snippet: {description[:1000] if description else ""}
Full Content: {scraped_content[:10000]}

"""

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": OPENAI_MODEL,
            "input": prompt,
            "max_output_tokens": 15000,
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "low"},
            "truncation": "auto"
        }
        
        LOG.info(f"Generating AI summary for {ticker} - Content: {len(scraped_content)} chars, Title: {title[:50]}...")
        
        response = get_openai_session().post(OPENAI_API_URL, headers=headers, json=data, timeout=(10, 180))
        
        if response.status_code == 200:
            result = response.json()
            
            u = result.get("usage", {}) or {}
            LOG.info("AI Enhanced Summary usage — input:%s output:%s (cap:%s) status:%s reason:%s",
                     u.get("input_tokens"), u.get("output_tokens"),
                     result.get("max_output_tokens"),
                     result.get("status"),
                     (result.get("incomplete_details") or {}).get("reason"))
            
            summary = extract_text_from_responses(result)
            if summary and len(summary.strip()) > 10:
                LOG.info(f"Generated AI summary for {ticker}: {len(summary)} chars - '{summary[:100]}...'")
                return summary.strip()
            else:
                LOG.warning(f"AI summary empty or too short for {ticker}: '{summary}'")
                return None
        else:
            LOG.error(f"AI summary API error {response.status_code} for {ticker}: {response.text}")
            return None
            
    except Exception as e:
        LOG.error(f"AI enhanced summary generation failed for {ticker}: {e}")
        return None

def perform_ai_triage_batch(articles_by_category: Dict[str, List[Dict]], ticker: str) -> Dict[str, List[Dict]]:
    """
    Perform AI triage on batched articles to identify scraping candidates
    FIXED: Process industry/competitor articles by keyword/competitor separately
    """
    if not OPENAI_API_KEY:
        LOG.warning("OpenAI API key not configured - skipping triage")
        return {"company": [], "industry": [], "competitor": []}
    
    selected_results = {"company": [], "industry": [], "competitor": []}
    
    # Get ticker metadata for enhanced prompts
    config = get_ticker_config(ticker)
    company_name = config.get("name", ticker) if config else ticker
    
    # Company articles - process as single batch (limit: 20)
    company_articles = articles_by_category.get("company", [])
    if company_articles:
        LOG.info(f"Starting AI triage for company: {len(company_articles)} articles")
        try:
            selected = triage_company_articles_full(company_articles, ticker, company_name, {}, {})
            selected_results["company"] = selected
            LOG.info(f"AI triage company: selected {len(selected)} articles for scraping")
        except Exception as e:
            LOG.error(f"Company triage failed: {e}")
    
    # Industry articles - FIXED: Process by keyword separately (5 per keyword)
    industry_articles = articles_by_category.get("industry", [])
    if industry_articles:
        # Group by search_keyword
        industry_by_keyword = {}
        for idx, article in enumerate(industry_articles):
            keyword = article.get("search_keyword", "unknown")
            if keyword not in industry_by_keyword:
                industry_by_keyword[keyword] = []
            industry_by_keyword[keyword].append({"article": article, "original_idx": idx})
        
        LOG.info(f"Starting AI triage for industry: {len(industry_articles)} articles across {len(industry_by_keyword)} keywords")
        
        all_industry_selected = []
        for keyword, keyword_articles in industry_by_keyword.items():
            try:
                LOG.info(f"Processing industry keyword '{keyword}': {len(keyword_articles)} articles")
                triage_articles = [item["article"] for item in keyword_articles]
                selected = triage_industry_articles_full(triage_articles, ticker, {}, [])
                
                # Map back to original indices
                for selected_item in selected:
                    original_idx = keyword_articles[selected_item["id"]]["original_idx"]
                    selected_item["id"] = original_idx
                    all_industry_selected.append(selected_item)
                
                LOG.info(f"Industry keyword '{keyword}': selected {len(selected)} articles")
            except Exception as e:
                LOG.error(f"Industry triage failed for keyword '{keyword}': {e}")
        
        selected_results["industry"] = all_industry_selected
        LOG.info(f"AI triage industry: selected {len(all_industry_selected)} articles total")
    
    # Competitor articles - FIXED: Process by competitor separately (5 per competitor)
    competitor_articles = articles_by_category.get("competitor", [])
    if competitor_articles:
        # Group by competitor_ticker (primary) or search_keyword (fallback)
        competitor_by_entity = {}
        for idx, article in enumerate(competitor_articles):
            entity_key = article.get("competitor_ticker") or article.get("search_keyword", "unknown")
            if entity_key not in competitor_by_entity:
                competitor_by_entity[entity_key] = []
            competitor_by_entity[entity_key].append({"article": article, "original_idx": idx})
        
        LOG.info(f"Starting AI triage for competitor: {len(competitor_articles)} articles across {len(competitor_by_entity)} competitors")
        
        all_competitor_selected = []
        for entity_key, entity_articles in competitor_by_entity.items():
            try:
                LOG.info(f"Processing competitor '{entity_key}': {len(entity_articles)} articles")
                triage_articles = [item["article"] for item in entity_articles]
                selected = triage_competitor_articles_full(triage_articles, ticker, [], {})
                
                # Map back to original indices
                for selected_item in selected:
                    original_idx = entity_articles[selected_item["id"]]["original_idx"]
                    selected_item["id"] = original_idx
                    all_competitor_selected.append(selected_item)
                
                LOG.info(f"Competitor '{entity_key}': selected {len(selected)} articles")
            except Exception as e:
                LOG.error(f"Competitor triage failed for entity '{entity_key}': {e}")
        
        selected_results["competitor"] = all_competitor_selected
        LOG.info(f"AI triage competitor: selected {len(all_competitor_selected)} articles total")
    
    return selected_results

def rule_based_triage_score_company(title: str, domain: str) -> Tuple[int, str, str]:
    """
    Company-specific rule-based scoring based on AI triage logic
    Returns (score, reasoning, qb_level)
    """
    score = 0
    reasons = []
    
    title_lower = title.lower()
    
    # HIGH PRIORITY - Hard business events (60-80 points)
    high_events = [
        r'\b(acquires?|acquisition|merger|divests?|divestiture|spin-?off)\b',
        r'\b(bankruptcy|chapter 11|delist|delisting|halt|halted)\b',
        r'\b(guidance|preannounce|beats?|misses?|earnings|results|q[1-4])\b',
        r'\b(margin|backlog|contract|long-?term agreement|supply deal)\b',
        r'\b(price increase|price cut|capacity add|closure|curtailment)\b',
        r'\b(buyback|tender|equity offering|convertible|refinanc)\b',
        r'\b(rating change|approval|license)\b',
        r'\b(tariff|quota|sanction|fine|settlement)\b',
        r'\b(doj|ftc|sec|fda|usda|epa|osha|nhtsa|faa|fcc)\b'
    ]
    
    for pattern in high_events:
        if re.search(pattern, title_lower):
            score += 70
            reasons.append("hard business event")
            break
    
    # MEDIUM PRIORITY - Strategic developments (40-55 points)
    if score == 0:  # Only if no high-priority event found
        medium_events = [
            r'\b(investment|expansion)\b.*\$[\d,]+',  # Investment with $ amount
            r'\b(technology|product)\b.*\b(launch|deployment|ship)\b',
            r'\b(ceo|cfo|president|director)\b.*\b(change|resign|appoint|hire)\b',
            r'\b(partnership|joint venture|collaboration)\b',
            r'\b(facility|plant|factory)\b.*\b(opening|closing)\b'
        ]
        
        for pattern in medium_events:
            if re.search(pattern, title_lower):
                score += 50
                reasons.append("strategic development")
                break
    
    # LOW PRIORITY - Routine coverage (25-35 points)
    if score == 0:  # Only if no higher priority found
        low_events = [
            r'\b(analyst|rating|target|upgrade|downgrade)\b',
            r'\b(corporate announcement|operational)\b'
        ]
        
        for pattern in low_events:
            if re.search(pattern, title_lower):
                score += 30
                reasons.append("routine coverage")
                break
    
    # FINANCIAL SPECIFICITY BOOSTERS (+10-25 points)
    if re.search(r'\$[\d,]+\.?\d*\s*(million|billion|m|b)\b', title_lower):
        score += 20
        reasons.append("dollar amount")
        
    if re.search(r'\b\d+\.?\d*%\b', title):
        score += 15
        reasons.append("percentage")
    
    if re.search(r'\b(q[1-4]|\d{4}|\d+\s*(year|month|day)s?)\b', title_lower):
        score += 10
        reasons.append("timeline specificity")
    
    # EXCLUDE/PENALIZE PATTERNS (-30 to -50 points)
    exclude_patterns = [
        r'\b(top\s+\d+|best|should you|right now|reasons|prediction)\b',
        r'\b(if you.d invested|what to know|how to|why|analysis|outlook)\b',
        r'\b(announces)\b(?!.*\$)(?!.*\d+%)',  # "Announces" without concrete numbers
        r'\b(market size|cagr|forecast 20\d{2}|to reach \$.*by 20\d{2})\b',
        r'\b(market report|press release)\b(?!.*\$)(?!.*\d+%)'
    ]
    
    for pattern in exclude_patterns:
        if re.search(pattern, title_lower):
            score -= 40
            reasons.append("excluded pattern")
            break
    
    # DOMAIN BONUS
    domain_tier = DOMAIN_TIERS.get(normalize_domain(domain), 0.3)
    score += int(domain_tier * 25)
    reasons.append(f"domain tier {domain_tier}")
    
    final_score = max(0, min(100, score))
    
    # Determine QB level for company
    if final_score >= 70:
        qb_level = "QB: High"
    elif final_score >= 40:
        qb_level = "QB: Medium"
    else:
        qb_level = "QB: Low"
    
    return final_score, " + ".join(reasons), qb_level

def rule_based_triage_score_industry(title: str, domain: str, industry_keywords: List[str] = None) -> Tuple[int, str, str]:
    """
    Industry-specific rule-based scoring based on AI triage logic
    Returns (score, reasoning, qb_level)
    """
    score = 0
    reasons = []
    
    title_lower = title.lower()
    keywords = industry_keywords or []
    
    # HIGH PRIORITY - Policy/regulatory shocks with quantified impact (65-85 points)
    high_events = [
        r'\b(tariff|ban|quota|price control|regulatory change)\b.*\b\d+',
        r'\b(supply shock|inventory)\b.*\b(draw|build)\b.*\b\d+',
        r'\b(price cap|price floor|standard adopted)\b',
        r'\b(subsidy|credit|reimbursement change)\b.*\$',
        r'\b(safety requirement|environmental standard)\b.*\b(effective|deadline)\b',
        r'\b(trade agreement|export control|import restriction)\b'
    ]
    
    for pattern in high_events:
        if re.search(pattern, title_lower):
            score += 75
            reasons.append("policy/regulatory shock")
            break
    
    # MEDIUM PRIORITY - Sector developments (45-60 points)
    if score == 0:
        medium_events = [
            r'\b(infrastructure investment)\b.*\$[\d,]+',
            r'\b(industry consolidation)\b.*\$[\d,]+',
            r'\b(technology standards adoption)\b.*\b(implementation|schedule)\b',
            r'\b(labor agreement)\b.*\b(wage|benefit|cost)\b',
            r'\b(supply chain)\b.*\b(volume|pricing)\b',
            r'\b(capacity)\b.*\b(addition|reduction)\b.*\b(production|impact)\b'
        ]
        
        for pattern in medium_events:
            if re.search(pattern, title_lower):
                score += 52
                reasons.append("sector development")
                break
    
    # LOW PRIORITY - Broad trends (30-40 points)
    if score == 0:
        low_events = [
            r'\b(government initiative)\b.*\b(budget|implementation)\b',
            r'\b(economic indicator)\b.*\b(sector|industry)\b',
            r'\b(research finding)\b.*\b(quantified|impact)\b'
        ]
        
        for pattern in low_events:
            if re.search(pattern, title_lower):
                score += 35
                reasons.append("broad trend")
                break
    
    # INDUSTRY KEYWORD RELEVANCE BOOST (+15-25 points)
    if keywords:
        keyword_matches = 0
        for keyword in keywords:
            if keyword.lower() in title_lower:
                keyword_matches += 1
        
        if keyword_matches > 0:
            boost = min(keyword_matches * 8, 25)
            score += boost
            reasons.append(f"{keyword_matches} keyword matches")
    
    # SPECIFICITY BOOSTERS
    if re.search(r'\b\d+\.?\d*%\b.*\b(up|down|increase|decrease)\b', title_lower):
        score += 20
        reasons.append("percentage change")
        
    if re.search(r'\b(effective|deadline|implementation)\b.*\b20\d{2}\b', title_lower):
        score += 15
        reasons.append("implementation date")
    
    # EXCLUDE PATTERNS - Industry specific
    exclude_patterns = [
        r'\b(market size|cagr|forecast 20\d{2}|analysis report)\b',
        r'\b(industry outlook|future of|trends|sustainability)\b(?!.*\b(regulation|compliance)\b)',
        r'\b(academic research|consumer preference)\b(?!.*\b(policy|demand shift)\b)'
    ]
    
    for pattern in exclude_patterns:
        if re.search(pattern, title_lower):
            score -= 35
            reasons.append("excluded industry pattern")
            break
    
    # DOMAIN BONUS
    domain_tier = DOMAIN_TIERS.get(normalize_domain(domain), 0.3)
    score += int(domain_tier * 20)
    reasons.append(f"domain tier {domain_tier}")
    
    final_score = max(0, min(100, score))
    
    # Industry-specific QB levels (slightly different thresholds)
    if final_score >= 75:
        qb_level = "QB: High"
    elif final_score >= 45:
        qb_level = "QB: Medium"
    else:
        qb_level = "QB: Low"
    
    return final_score, " + ".join(reasons), qb_level

def rule_based_triage_score_competitor(title: str, domain: str, competitors: List[str] = None) -> Tuple[int, str, str]:
    """
    Competitor-specific rule-based scoring based on AI triage logic
    Returns (score, reasoning, qb_level)
    """
    score = 0
    reasons = []
    
    title_lower = title.lower()
    competitor_names = []
    
    # Extract competitor names from list
    if competitors:
        for comp in competitors:
            if isinstance(comp, dict):
                competitor_names.append(comp.get('name', '').lower())
            else:
                # Handle "Name (TICKER)" format
                name = comp.split('(')[0].strip().lower()
                competitor_names.append(name)
    
    # Check if any competitor is mentioned
    competitor_mentioned = False
    mentioned_competitor = ""
    for comp_name in competitor_names:
        if comp_name and comp_name in title_lower:
            competitor_mentioned = True
            mentioned_competitor = comp_name
            break
    
    if not competitor_mentioned:
        # If no competitor mentioned, default to low score
        score = 15
        reasons.append("no competitor match")
    else:
        # HIGH PRIORITY - Hard competitive events (65-85 points)
        high_events = [
            r'\b(capacity expansion|capacity reduction)\b.*\b\d+',
            r'\b(pricing action|price increase|price cut)\b.*\b\d+%',
            r'\b(customer win|customer loss|major customer)\b',
            r'\b(plant opening|plant closing)\b.*\b(output|capacity)\b',
            r'\b(asset sale|acquisition)\b.*\$[\d,]+',
            r'\b(restructuring|bankruptcy|chapter 11)\b',
            r'\b(breakthrough|launch)\b.*\b(ship date|deployment)\b',
            r'\b(supply agreement)\b.*\b(volume|capacity)\b',
            r'\b(market entry|market exit)\b.*\$[\d,]+'
        ]
        
        for pattern in high_events:
            if re.search(pattern, title_lower):
                score += 75
                reasons.append(f"hard competitive event ({mentioned_competitor})")
                break
        
        # MEDIUM PRIORITY - Strategic moves (45-60 points)
        if score == 0:
            medium_events = [
                r'\b(acquisition|partnership)\b.*\b(deal value|strategic)\b',
                r'\b(technology)\b.*\b(deployment|timeline)\b.*\b(competitive)\b',
                r'\b(ceo|cfo|division head)\b.*\b(change|succession)\b',
                r'\b(geographic expansion)\b.*\b(investment|market entry)\b',
                r'\b(regulatory approval)\b.*\b(timeline|capability)\b',
                r'\b(supply chain)\b.*\b(cost|availability|contract)\b'
            ]
            
            for pattern in medium_events:
                if re.search(pattern, title_lower):
                    score += 52
                    reasons.append(f"strategic move ({mentioned_competitor})")
                    break
        
        # LOW PRIORITY - Routine competitive intel (30-40 points)
        if score == 0:
            low_events = [
                r'\b(earnings)\b.*\b(guidance|beat|miss)\b',
                r'\b(analyst)\b.*\b(rating change|target)\b',
                r'\b(product announcement)\b.*\b(launch|timeline)\b'
            ]
            
            for pattern in low_events:
                if re.search(pattern, title_lower):
                    score += 35
                    reasons.append(f"routine intel ({mentioned_competitor})")
                    break
    
    # COMPETITIVE IMPACT BOOSTERS
    if re.search(r'\b(market share|losing ground|gaining share)\b', title_lower):
        score += 20
        reasons.append("market share impact")
        
    if re.search(r'\b(pricing power|cost structure|competitive advantage)\b', title_lower):
        score += 15
        reasons.append("competitive positioning")
    
    # FINANCIAL SPECIFICITY
    if re.search(r'\$[\d,]+\.?\d*\s*(million|billion)\b', title_lower):
        score += 18
        reasons.append("deal value")
        
    if re.search(r'\b\d+\.?\d*%\b.*\b(capacity|price|margin)\b', title_lower):
        score += 15
        reasons.append("quantified impact")
    
    # EXCLUDE PATTERNS - Competitor specific
    exclude_patterns = [
        r'\b(generic analyst commentary|stock performance)\b(?!.*\b(operational|guidance)\b)',
        r'\b(historical retrospective)\b(?!.*\b(competitive|forward)\b)',
        r'\b(technical analysis|trading analysis)\b',
        r'\b(market commentary)\b(?!.*\b(competitive|share)\b)'
    ]
    
    for pattern in exclude_patterns:
        if re.search(pattern, title_lower):
            score -= 30
            reasons.append("excluded competitor pattern")
            break
    
    # DOMAIN BONUS
    domain_tier = DOMAIN_TIERS.get(normalize_domain(domain), 0.3)
    score += int(domain_tier * 20)
    reasons.append(f"domain tier {domain_tier}")
    
    final_score = max(0, min(100, score))
    
    # Competitor-specific QB levels
    if final_score >= 70:
        qb_level = "QB: High"
    elif final_score >= 40:
        qb_level = "QB: Medium"
    else:
        qb_level = "QB: Low"
    
    return final_score, " + ".join(reasons), qb_level

def _apply_tiered_backfill_to_limits(articles: List[Dict], ai_selected: List[Dict], category: str, low_quality_domains: Set[str], target_limit: int) -> List[Dict]:
    """
    Apply backfill using AI → Quality → Category-specific QB scoring (100→0)
    AND store QB scores in database for ALL articles
    """
    # Step 1: Start with AI selections
    combined_selected = list(ai_selected)
    selected_indices = {item["id"] for item in ai_selected}
    
    # Step 2: Add quality domains (not already AI selected)
    quality_selected = []
    for idx, article in enumerate(articles):
        if idx in selected_indices:
            continue
            
        domain = normalize_domain(article.get("domain", ""))
        title = article.get("title", "").lower()
        
        if domain in low_quality_domains or is_insider_trading_article(title):
            continue
            
        if domain in QUALITY_DOMAINS:
            quality_selected.append({
                "id": idx,
                "scrape_priority": "MEDIUM",  # Updated to use HIGH/MEDIUM/LOW
                "likely_repeat": False,
                "repeat_key": "",
                "why": f"Quality domain: {domain}",
                "confidence": 0.8,
                "selection_method": "quality_domain"
            })
            selected_indices.add(idx)
    
    combined_selected.extend(quality_selected)
    
    # Step 3: Category-specific QB scoring and selection
    current_count = len(combined_selected)
    backfill_selected = []
    
    # Score ALL articles for QB analysis (regardless of selection)
    for idx, article in enumerate(articles):
        domain = normalize_domain(article.get("domain", ""))
        title = article.get("title", "")
        
        if not (domain in low_quality_domains or is_insider_trading_article(title.lower())):
            # Calculate QB score for every article
            if category == "company":
                qb_score, qb_reasoning, qb_level = rule_based_triage_score_company(title, domain)
            elif category == "industry":
                keywords = [article.get('search_keyword')] if article.get('search_keyword') else []
                qb_score, qb_reasoning, qb_level = rule_based_triage_score_industry(title, domain, keywords)
            elif category == "competitor":
                qb_score, qb_reasoning, qb_level = rule_based_triage_score_competitor(title, domain, [])
            else:
                qb_score, qb_reasoning, qb_level = rule_based_triage_score(title, domain)
            
            # Store QB scores in article object for database update
            article['qb_score'] = qb_score
            article['qb_level'] = qb_level
            article['qb_reasoning'] = qb_reasoning
    
    if current_count < target_limit:
        remaining_slots = target_limit - current_count
        
        # Score remaining articles for backfill selection
        scored_candidates = []
        for idx, article in enumerate(articles):
            if idx in selected_indices:
                continue
                
            domain = normalize_domain(article.get("domain", ""))
            title = article.get("title", "")
            
            if domain in low_quality_domains or is_insider_trading_article(title.lower()):
                continue
            
            # Use the QB scores we just calculated
            qb_score = article.get('qb_score', 0)
            qb_level = article.get('qb_level', 'QB: Low')
            qb_reasoning = article.get('qb_reasoning', '')
            
            scored_candidates.append({
                "id": idx,
                "article": article,
                "qb_score": qb_score,
                "qb_reasoning": qb_reasoning,
                "qb_level": qb_level,
                "domain": domain,
                "published_at": article.get("published_at")
            })
        
        # Sort by QB score descending (100 → 0), then by publication time
        scored_candidates.sort(key=lambda x: (
            -x["qb_score"],
            -(x["published_at"].timestamp() if x["published_at"] else 0)
        ))
        
        # Take top candidates up to remaining slots
        for candidate in scored_candidates[:remaining_slots]:
            # Convert QB level to scrape priority
            if candidate['qb_score'] >= 70:
                scrape_priority = "HIGH"
            elif candidate['qb_score'] >= 40:
                scrape_priority = "MEDIUM"
            else:
                scrape_priority = "LOW"
                
            backfill_selected.append({
                "id": candidate["id"],
                "scrape_priority": scrape_priority,  # Now using HIGH/MEDIUM/LOW strings
                "likely_repeat": False,
                "repeat_key": "",
                "why": f"{candidate['qb_level']}: {candidate['qb_reasoning']} (score: {candidate['qb_score']})",
                "confidence": 0.6,
                "selection_method": f"qb_score_{category}",
                "qb_score": candidate["qb_score"],
                "qb_level": candidate["qb_level"]
            })
        
        combined_selected.extend(backfill_selected)
        
        # Final sort by priority (HIGH=1, MEDIUM=2, LOW=3)
        priority_map = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}
        combined_selected.sort(key=lambda x: priority_map.get(x.get("scrape_priority", "LOW"), 3))
    
    # Enhanced logging
    ai_count = len(ai_selected)
    quality_count = len(quality_selected)
    qb_count = len(backfill_selected)
    
    LOG.info(f"Category-specific QB backfill {category}: {ai_count} AI + {quality_count} Quality + {qb_count} QB-{category} = {len(combined_selected)}/{target_limit}")
    
    return combined_selected

def perform_ai_triage_batch_with_enhanced_selection(articles_by_category: Dict[str, List[Dict]], ticker: str, target_limits: Dict[str, int] = None) -> Dict[str, List[Dict]]:
    """
    Enhanced triage with detailed logging for render logs
    """
    if not OPENAI_API_KEY:
        LOG.warning("OpenAI API key not configured - using existing triage data and quality domains only")
        return {"company": [], "industry": [], "competitor": []}
    
    selected_results = {"company": [], "industry": [], "competitor": []}
    
    # Get ticker metadata
    config = get_ticker_config(ticker)
    company_name = config.get("name", ticker) if config else ticker
    
    # DETAILED TRIAGE LOGGING FOR RENDER
    LOG.info("=== DETAILED TRIAGE BREAKDOWN ===")
    LOG.info(f"Ticker: {ticker} ({company_name})")
    
    total_ai_selected = 0
    total_quality_selected = 0
    total_qb_backfill = 0
    
    # COMPANY: Process with smart reuse and detailed logging
    company_articles = articles_by_category.get("company", [])
    if company_articles:
        LOG.info(f"COMPANY TRIAGE: Processing {len(company_articles)} articles")
        
        # Separate articles into those that need triage vs those already triaged
        needs_triage = []
        already_triaged = []
        
        for idx, article in enumerate(company_articles):
            # Check if article already has triage data and good quality
            if (article.get('ai_triage_selected') and 
                article.get('triage_priority') and 
                article.get('quality_score', 0) >= 40):
                
                already_triaged.append({
                    "id": idx,
                    "scrape_priority": article.get('triage_priority', 'MEDIUM'),
                    "likely_repeat": False,
                    "repeat_key": "",
                    "why": f"Previously triaged - {article.get('triage_reasoning', 'existing selection')}",
                    "confidence": 0.9,
                    "selection_method": "existing_triage"
                })
            else:
                needs_triage.append((idx, article))
        
        LOG.info(f"  COMPANY REUSE: {len(already_triaged)} already triaged")
        LOG.info(f"  COMPANY NEW: {len(needs_triage)} need new triage")
        
        # Start with existing triaged articles
        company_selected = list(already_triaged)
        
        # Run AI triage on new articles if needed
        remaining_slots = 20 - len(company_selected)
        if remaining_slots > 0 and needs_triage:
            try:
                triage_articles = [article for _, article in needs_triage]
                new_selected = triage_company_articles_full(triage_articles, ticker, company_name, {}, {})
                
                # Map back to original indices and add to selection
                for selected_item in new_selected[:remaining_slots]:
                    original_idx = needs_triage[selected_item["id"]][0]
                    selected_item["id"] = original_idx
                    selected_item["selection_method"] = "new_ai_triage"
                    company_selected.append(selected_item)
                
                LOG.info(f"  COMPANY AI: {len(new_selected)} selected from new AI triage")
                total_ai_selected += len(new_selected)
            except Exception as e:
                LOG.error(f"Company triage failed: {e}")
        
        # Apply enhanced selection (Quality + QB backfill)
        quality_domains_count = 0
        qb_backfill_count = 0
        
        for idx, article in enumerate(company_articles):
            if idx not in [item["id"] for item in company_selected]:
                domain = normalize_domain(article.get("domain", ""))
                if domain in QUALITY_DOMAINS:
                    quality_domains_count += 1
                else:
                    # This would be QB backfill
                    qb_backfill_count += 1
        
        total_quality_selected += quality_domains_count
        total_qb_backfill += min(qb_backfill_count, max(0, 20 - len(company_selected) - quality_domains_count))
        
        LOG.info(f"  COMPANY QUALITY: {quality_domains_count} quality domains")
        LOG.info(f"  COMPANY QB BACKFILL: {total_qb_backfill} QB score selections")
        LOG.info(f"  COMPANY TOTAL: {len(company_selected)} selected for scraping")
        
        selected_results["company"] = company_selected
    
    # INDUSTRY: Process EACH keyword separately with detailed logging
    industry_articles = articles_by_category.get("industry", [])
    if industry_articles:
        # Group by search_keyword
        industry_by_keyword = {}
        for idx, article in enumerate(industry_articles):
            keyword = article.get("search_keyword", "unknown")
            if keyword not in industry_by_keyword:
                industry_by_keyword[keyword] = []
            industry_by_keyword[keyword].append({"article": article, "original_idx": idx})
        
        LOG.info(f"INDUSTRY TRIAGE: Processing {len(industry_by_keyword)} keywords")
        
        industry_selected = []
        for keyword, keyword_articles in industry_by_keyword.items():
            if len(keyword_articles) == 0:
                continue
                
            LOG.info(f"  KEYWORD '{keyword}': {len(keyword_articles)} articles")
            
            # Separate into existing vs new for this keyword
            needs_triage = []
            already_triaged = []
            
            for item in keyword_articles:
                article = item["article"]
                original_idx = item["original_idx"]
                
                if (article.get('ai_triage_selected') and 
                    article.get('triage_priority') and 
                    article.get('quality_score', 0) >= 40):
                    
                    already_triaged.append({
                        "id": original_idx,
                        "scrape_priority": article.get('triage_priority', 'MEDIUM'),
                        "likely_repeat": False,
                        "repeat_key": "",
                        "why": f"Previously triaged - {article.get('triage_reasoning', 'existing selection')}",
                        "confidence": 0.9,
                        "selection_method": "existing_triage"
                    })
                else:
                    needs_triage.append(item)
            
            # Start with existing
            keyword_selected = list(already_triaged)
            
            # Run AI triage on new articles if needed (target: 5 per keyword)
            remaining_slots = 5 - len(keyword_selected)
            if remaining_slots > 0 and needs_triage:
                try:
                    triage_articles = [item["article"] for item in needs_triage]
                    new_selected = triage_industry_articles_full(triage_articles, ticker, {}, [])
                    
                    # Map back to original indices
                    for selected_item in new_selected[:remaining_slots]:
                        original_idx = needs_triage[selected_item["id"]]["original_idx"]
                        selected_item["id"] = original_idx
                        selected_item["selection_method"] = "new_ai_triage"
                        keyword_selected.append(selected_item)
                    
                    LOG.info(f"    AI TRIAGE: {len(new_selected)} selected")
                    total_ai_selected += len(new_selected)
                except Exception as e:
                    LOG.error(f"Industry triage failed for keyword '{keyword}': {e}")
            
            industry_selected.extend(keyword_selected)
            LOG.info(f"    TOTAL: {len(keyword_selected)}/5 selected for '{keyword}'")
        
        selected_results["industry"] = industry_selected
        LOG.info(f"INDUSTRY TOTAL: {len(industry_selected)} articles selected across all keywords")
    
    # COMPETITOR: Process EACH competitor separately with detailed logging
    competitor_articles = articles_by_category.get("competitor", [])
    if competitor_articles:
        # Group by competitor_ticker (primary) or search_keyword (fallback)
        competitor_by_entity = {}
        for idx, article in enumerate(competitor_articles):
            entity_key = article.get("competitor_ticker") or article.get("search_keyword", "unknown")
            if entity_key not in competitor_by_entity:
                competitor_by_entity[entity_key] = []
            competitor_by_entity[entity_key].append({"article": article, "original_idx": idx})
        
        LOG.info(f"COMPETITOR TRIAGE: Processing {len(competitor_by_entity)} competitors")
        
        competitor_selected = []
        for entity_key, entity_articles in competitor_by_entity.items():
            if len(entity_articles) == 0:
                continue
                
            LOG.info(f"  COMPETITOR '{entity_key}': {len(entity_articles)} articles")
            
            # Separate into existing vs new for this competitor
            needs_triage = []
            already_triaged = []
            
            for item in entity_articles:
                article = item["article"]
                original_idx = item["original_idx"]
                
                if (article.get('ai_triage_selected') and 
                    article.get('triage_priority') and 
                    article.get('quality_score', 0) >= 40):
                    
                    already_triaged.append({
                        "id": original_idx,
                        "scrape_priority": article.get('triage_priority', 'MEDIUM'),
                        "likely_repeat": False,
                        "repeat_key": "",
                        "why": f"Previously triaged - {article.get('triage_reasoning', 'existing selection')}",
                        "confidence": 0.9,
                        "selection_method": "existing_triage"
                    })
                else:
                    needs_triage.append(item)
            
            # Start with existing
            entity_selected = list(already_triaged)
            
            # Run AI triage on new articles if needed (target: 5 per competitor)
            remaining_slots = 5 - len(entity_selected)
            if remaining_slots > 0 and needs_triage:
                try:
                    triage_articles = [item["article"] for item in needs_triage]
                    new_selected = triage_competitor_articles_full(triage_articles, ticker, [], {})
                    
                    # Map back to original indices
                    for selected_item in new_selected[:remaining_slots]:
                        original_idx = needs_triage[selected_item["id"]]["original_idx"]
                        selected_item["id"] = original_idx
                        selected_item["selection_method"] = "new_ai_triage"
                        entity_selected.append(selected_item)
                    
                    LOG.info(f"    AI TRIAGE: {len(new_selected)} selected")
                    total_ai_selected += len(new_selected)
                except Exception as e:
                    LOG.error(f"Competitor triage failed for entity '{entity_key}': {e}")
            
            competitor_selected.extend(entity_selected)
            LOG.info(f"    TOTAL: {len(entity_selected)}/5 selected for '{entity_key}'")
        
        selected_results["competitor"] = competitor_selected
        LOG.info(f"COMPETITOR TOTAL: {len(competitor_selected)} articles selected across all competitors")
    
    # FINAL TRIAGE SUMMARY FOR RENDER LOGS
    total_selected = sum(len(items) for items in selected_results.values())
    LOG.info("=== TRIAGE SUMMARY ===")
    LOG.info(f"AI TRIAGE: {total_ai_selected} articles")
    LOG.info(f"QUALITY DOMAINS: {total_quality_selected} articles")
    LOG.info(f"QB BACKFILL: {total_qb_backfill} articles")
    LOG.info(f"TOTAL SELECTED: {total_selected} articles")
    LOG.info("=== END TRIAGE ===")
    
    return selected_results

def _safe_json_loads(s: str) -> Dict:
    """
    Robust JSON parsing:
      - strip code fences if any
      - find outermost JSON object
      - fallback to {} on failure
    """
    s = s.strip()
    
    # Strip ```json ... ``` fences if present
    if s.startswith("```"):
        s = s.strip("`")
        # after stripping backticks, content may start with 'json\n'
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    
    # Quick path - try to parse as-is
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # Fallback: slice from first '{' to last '}' (common when trailing notes appear)
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end + 1])
    except Exception:
        pass
    
    LOG.error("Failed to parse JSON. First 200 chars:\n" + s[:200])
    return {"selected_ids": [], "selected": [], "skipped": []}

def _make_triage_request_full(system_prompt: str, payload: dict) -> List[Dict]:
    """Make structured triage request to OpenAI with integer priority storage"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Extract target info for constraint enforcement
        target_cap = payload.get('target_cap', 10)
        total_items = len(payload.get('items', []))
        effective_cap = min(target_cap, total_items)
        
        # Updated schema to use integer priorities
        triage_schema = {
            "type": "object",
            "properties": {
                "selected_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "maxItems": effective_cap,
                    "minItems": effective_cap
                },
                "selected": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "scrape_priority": {"type": "integer", "minimum": 1, "maximum": 3},
                            "likely_repeat": {"type": "boolean"},
                            "repeat_key": {"type": "string"},
                            "why": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["id", "scrape_priority", "likely_repeat", "repeat_key", "why", "confidence"],
                        "additionalProperties": False
                    },
                    "maxItems": effective_cap,
                    "minItems": effective_cap
                },
                "skipped": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "scrape_priority": {"type": "integer", "minimum": 1, "maximum": 3},
                            "likely_repeat": {"type": "boolean"},
                            "repeat_key": {"type": "string"},
                            "why": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["id", "scrape_priority", "likely_repeat", "repeat_key", "why", "confidence"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["selected_ids", "selected", "skipped"],
            "additionalProperties": False
        }
        
        # Updated system prompt to specify integer priorities
        constrained_system_prompt = f"""{system_prompt}

CRITICAL SELECTION CONSTRAINT:
You MUST select EXACTLY {effective_cap} articles from the {total_items} provided.
Be highly selective. Choose only the {effective_cap} BEST articles based on the criteria above.

Your selected_ids array must contain exactly {effective_cap} integers.
Your selected array must contain exactly {effective_cap} objects.

PRIORITY SCALE (use integers):
- 1 = High priority (most important articles requiring immediate scraping)
- 2 = Medium priority (moderate importance)
- 3 = Low priority (lower importance, backup selections)

Return only valid JSON with integer scrape_priority values (1, 2, or 3)."""
        
        data = {
            "model": OPENAI_MODEL,
            "reasoning": {"effort": "medium"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "triage_results",
                    "schema": triage_schema,
                    "strict": True
                },
                "verbosity": "low"
            },
            "input": f"{constrained_system_prompt}\n\n{json.dumps(payload, separators=(',', ':'))}",
            "max_output_tokens": 20000,
            "truncation": "auto"
        }
        
        response = get_openai_session().post(OPENAI_API_URL, headers=headers, json=data, timeout=(10, 180))
        
        if response.status_code != 200:
            LOG.error(f"OpenAI triage API error {response.status_code}: {response.text}")
            return []
        
        result = response.json()
        content = extract_text_from_responses(result)
        
        if not content:
            LOG.error("OpenAI returned no text content for triage")
            return []
        
        try:
            triage_result = json.loads(content)
        except json.JSONDecodeError as e:
            LOG.error(f"JSON parsing failed for triage: {e}")
            return []
        
        selected_articles = []
        for selected_item in triage_result.get("selected", []):
            if selected_item["id"] < total_items:
                result_item = {
                    "id": selected_item["id"],
                    "scrape_priority": normalize_priority_to_int(selected_item["scrape_priority"]),
                    "why": selected_item["why"],
                    "confidence": selected_item["confidence"],
                    "likely_repeat": selected_item["likely_repeat"],
                    "repeat_key": selected_item["repeat_key"]
                }
                selected_articles.append(result_item)
        
        # Enforce the cap at the code level as final safeguard
        if len(selected_articles) > effective_cap:
            LOG.warning(f"Code-level cap enforcement: Trimming {len(selected_articles)} to {effective_cap}")
            selected_articles = selected_articles[:effective_cap]
        
        return selected_articles
        
    except Exception as e:
        LOG.error(f"Triage request failed: {str(e)}")
        return []

def triage_company_articles_full(articles: List[Dict], ticker: str, company_name: str, aliases_brands_assets: Dict, sector_profile: Dict) -> List[Dict]:
    """
    Enhanced company triage with explicit selection constraints
    """
    if not OPENAI_API_KEY or not articles:
        LOG.warning("No OpenAI API key or no articles to triage")
        return []

    # Prepare items for triage (title only, no domain quality info)
    items = []
    for i, article in enumerate(articles):
        items.append({
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        })

    target_cap = min(20, len(articles))  # Explicit cap for company articles
    
    payload = {
        "bucket": "company",
        "target_cap": target_cap,
        "ticker": ticker,
        "company_name": company_name,
        "items": items
    }

    system_prompt = f"""You are a financial analyst doing PRE-SCRAPE TRIAGE for COMPANY articles about {company_name} ({ticker}).

You MUST select EXACTLY {target_cap} articles from {len(articles)} total articles.
Focus SOLELY on title content. Ignore domain/source quality. Do not infer from outlet names embedded in titles.

SELECTION PRIORITY (choose the {target_cap} BEST):

HIGH PRIORITY - Hard business events with company as primary subject:
Event verbs: acquires, merger, divests, spin-off, bankruptcy, Chapter 11, delist, recall, halt, guidance, preannounce, beats, misses, earnings, margin, backlog, contract, long-term agreement, supply deal, price increase, price cut, capacity add, closure, curtailment, buyback, tender, equity offering, convertible, refinance, rating change, approval, license, tariff, quota, sanction, fine, settlement, DOJ, FTC, SEC, FDA, USDA, EPA, OSHA, NHTSA, FAA, FCC

MEDIUM PRIORITY - Strategic developments with specificity:
* Investment/expansion announcements with specific amounts or timelines
* Technology developments with ship dates or deployment specifics  
* Leadership changes (CEO, CFO, division heads) with effective dates
* Partnership announcements with scope/duration details
* Product launches with market entry dates and revenue targets
* Facility openings/closings with employment or capacity numbers

LOW PRIORITY - Routine coverage requiring backfill:
* Analyst coverage with material rating changes and revised targets
* Routine corporate announcements with minor operational impact
* Market commentary where company is mentioned with business context

EXCLUDE COMPLETELY:
* Listicles/opinion titles: "Top", "Best", "Should you", "Right now", "Reasons", "Prediction", "If you'd invested", "What to know", "How to", "Why", "Analysis", "Outlook"
* PR/TAM phrasing: "Announces" (without hard event verb), "Reports Market Size", "CAGR", "Forecast 20XX-20YY", "to reach $X by 2030", "Market Report", "Press Release" (unless paired with hard event verb AND concrete numbers)

CRITICAL: Be ruthlessly selective. Choose only the {target_cap} highest-priority articles. If fewer than {target_cap} articles meet the criteria, select the best available but never exceed {target_cap}.

The response will be automatically formatted as structured JSON with selected_ids, selected, and skipped arrays."""

    return _make_triage_request_full(system_prompt, payload)

def triage_industry_articles_full(articles: List[Dict], ticker: str, sector_profile: Dict, peers: List[str]) -> List[Dict]:
    """
    Enhanced industry triage with explicit selection constraints
    """
    if not OPENAI_API_KEY or not articles:
        LOG.warning("No OpenAI API key or no articles to triage")
        return []

    # Prepare items for triage
    items = []
    for i, article in enumerate(articles):
        items.append({
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        })

    # Extract industry keywords from the first article's search_keyword
    industry_keywords = [articles[0].get('search_keyword', '')] if articles else []
    target_cap = min(5, len(articles))

    payload = {
        "bucket": "industry",
        "target_cap": target_cap,
        "ticker": ticker,
        "industry_keywords": industry_keywords,
        "sector_profile": sector_profile,
        "items": items
    }

    system_prompt = f"""You are a financial analyst doing PRE-SCRAPE TRIAGE for INDUSTRY articles.

You MUST select EXACTLY {target_cap} articles from {len(articles)} total articles.
Focus SOLELY on title content. Ignore domain/source quality. Do not infer from outlet names embedded in titles.

SELECTION PRIORITY (choose the {target_cap} BEST):

HIGH PRIORITY - Policy/regulatory/benchmark shocks with quantified impact:
Event verbs: tariff, ban, quota, price control, regulatory change with date, supply shock, inventory draw/build with numbers, price cap/floor, standard adopted, subsidy/credit, reimbursement change, safety requirement, environmental standard, trade agreement, export control, import restriction
Must include: concrete numbers (%, $ amounts, effective dates, capacity figures, compliance costs)

MEDIUM PRIORITY - Sector developments with business implications:
* Large infrastructure investments affecting sector with budgets/timelines
* Industry consolidation with transaction values impacting capacity/pricing
* Technology standards adoption with implementation schedules
* Labor agreements affecting sector costs with wage/benefit specifics
* Supply chain changes affecting core inputs with volume/pricing data
* Capacity additions/reductions across sector with production impact

LOW PRIORITY - Broad trends requiring backfill:
* Government initiatives affecting sector without specific budgets/implementation dates
* Economic indicators with sector-specific implications and historical context
* Research findings with quantified sector impact and methodology details

EXCLUDE COMPLETELY:
* Market research/TAM reports: "Market Size", "CAGR", "Forecast 20XX-20YY", "to reach $X by 2030", "Market Report", "Analysis Report", "Industry Outlook" (unless tied to specific regulatory change with implementation dates)
* Listicles/generic content: "Top", "Best", "Trends", "Future of", "Analysis", "Outlook", "How to", "Why", "What to know"

CRITICAL: Be ruthlessly selective. Choose only the {target_cap} highest-priority articles. Never exceed {target_cap} selections.

The response will be automatically formatted as structured JSON with selected_ids, selected, and skipped arrays."""

    return _make_triage_request_full(system_prompt, payload)

def triage_competitor_articles_full(articles: List[Dict], ticker: str, peers: List[str], sector_profile: Dict) -> List[Dict]:
    """
    Enhanced competitor triage with explicit selection constraints
    """
    if not OPENAI_API_KEY or not articles:
        LOG.warning("No OpenAI API key or no articles to triage")
        return []

    # Prepare items for triage
    items = []
    for i, article in enumerate(articles):
        items.append({
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        })

    # Extract competitor names from peers
    competitors = [peer.split(' (')[0] if ' (' in peer else peer for peer in peers]
    target_cap = min(5, len(articles))

    payload = {
        "bucket": "competitor",
        "target_cap": target_cap,
        "ticker": ticker,
        "competitors": competitors,
        "sector_profile": sector_profile,
        "items": items
    }

    system_prompt = f"""You are a financial analyst doing PRE-SCRAPE TRIAGE for COMPETITOR articles.

You MUST select EXACTLY {target_cap} articles from {len(articles)} total articles.
Focus SOLELY on title content. Ignore domain/source quality. Do not infer from outlet names embedded in titles.

SELECTION PRIORITY (choose the {target_cap} BEST):

HIGH PRIORITY - Hard competitive events with quantified market impact:
Event verbs: capacity expansion/reduction with numbers, pricing actions with %, major customer win/loss, plant opening/closing with output figures, asset sale/acquisition with values, restructuring/bankruptcy, breakthrough/launch with ship dates, supply agreement with volumes, market entry/exit with investment amounts
Must include: specific competitor name + quantified business impact (%, $ amounts, capacity figures, timelines, customer names)

MEDIUM PRIORITY - Strategic competitive moves with business implications:
* Acquisitions/partnerships that change competitive landscape with deal values or strategic rationale
* Technology developments with deployment timelines affecting competitive positioning
* Management changes at key positions (CEO, division heads, key executives) with succession details
* Geographic expansion with market entry specifics and investment commitments
* Regulatory approvals enabling new capabilities or market access with timelines
* Supply chain agreements affecting costs or availability with contract terms

LOW PRIORITY - Routine competitive intelligence requiring backfill:
* Earnings releases with material guidance changes and specific numbers
* Analyst coverage with significant rating changes and revised price targets
* Product announcements with launch timelines and competitive positioning details
* Financial metrics showing competitive performance changes

EXCLUDE COMPLETELY:
* Generic analyst commentary without specific guidance/target changes
* Stock performance discussions without underlying operational drivers
* Listicles: "Top", "Best", "Should you", "Reasons", "Analysis", "Outlook", "How to", "Why"

CRITICAL: Be ruthlessly selective. Choose only the {target_cap} highest-priority articles. Never exceed {target_cap} selections.

The response will be automatically formatted as structured JSON with selected_ids, selected, and skipped arrays."""

    return _make_triage_request_full(system_prompt, payload)

# Pydantic models for structured response (keep existing)
from pydantic import BaseModel
from typing import List, Optional

class TriageItem(BaseModel):
    id: int
    scrape_priority: int
    likely_repeat: bool
    repeat_key: str
    why: str
    confidence: float

class TriageResponse(BaseModel):
    selected_ids: List[int]
    selected: List[TriageItem]
    skipped: List[TriageItem]

def get_competitor_display_name(search_keyword: str, competitor_ticker: str = None) -> str:
    """
    Get standardized competitor display name using database lookup with proper fallback
    Priority: competitor_ticker -> search_keyword -> fallback
    """
    
    # Input validation
    if competitor_ticker:
        competitor_ticker = competitor_ticker.strip().upper()
        if not re.match(r'^[A-Z]{1,5}$', competitor_ticker):
            LOG.warning(f"Invalid competitor ticker format: {competitor_ticker}")
            competitor_ticker = None
    
    if search_keyword:
        search_keyword = search_keyword.strip()
        if len(search_keyword) > 100 or not search_keyword:
            LOG.warning(f"Invalid search keyword: {search_keyword}")
            search_keyword = None
    
    # Try database lookup by ticker first (most reliable)
    if competitor_ticker:
        try:
            with db() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT company_name FROM competitor_metadata 
                    WHERE ticker = %s AND active = TRUE
                    LIMIT 1
                """, (competitor_ticker,))
                result = cur.fetchone()
                
                if result and result["company_name"]:
                    return result["company_name"]
        except Exception as e:
            LOG.debug(f"Database lookup failed for competitor {competitor_ticker}: {e}")
    
    # Try to get competitor name from source_feed table using competitor_ticker
    if competitor_ticker:
        try:
            with db() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT search_keyword FROM source_feed 
                    WHERE competitor_ticker = %s AND category = 'competitor' AND active = TRUE
                    LIMIT 1
                """, (competitor_ticker,))
                result = cur.fetchone()
                
                if result and result["search_keyword"]:
                    return result["search_keyword"]
        except Exception as e:
            LOG.debug(f"Source feed lookup failed for competitor {competitor_ticker}: {e}")
    
    # Fallback to search_keyword (should be company name for Google feeds)
    if search_keyword and not search_keyword.isupper():  # Likely a company name, not ticker
        return search_keyword
        
    # Final fallback - use ticker if that's all we have
    return competitor_ticker or search_keyword or "Unknown Competitor"

def get_url_hash(url: str, resolved_url: str = None) -> str:
    """Generate hash for URL deduplication, using resolved URL if available"""
    # Use resolved URL if available (this is the key part)
    primary_url = resolved_url or url
    url_lower = primary_url.lower()
    
    # Remove common parameters
    url_clean = re.sub(r'[?&](utm_|ref=|source=|siteid=|cid=|\.tsrc=).*', '', url_lower)
    url_clean = url_clean.rstrip('/')
    
    return hashlib.md5(url_clean.encode()).hexdigest()

def normalize_domain(domain: str) -> str:
    """Enhanced domain normalization for consistent storage"""
    if not domain:
        return None
    
    # Handle special consolidation cases first
    domain_mappings = {
        'ca.finance.yahoo.com': 'finance.yahoo.com',
        'uk.finance.yahoo.com': 'finance.yahoo.com', 
        'sg.finance.yahoo.com': 'finance.yahoo.com',
        'www.finance.yahoo.com': 'finance.yahoo.com',
        'yahoo.com/finance': 'finance.yahoo.com',
        'www.bloomberg.com': 'bloomberg.com',
        'www.reuters.com': 'reuters.com',
        'www.wsj.com': 'wsj.com',
        'www.cnbc.com': 'cnbc.com',
        'www.marketwatch.com': 'marketwatch.com',
        'www.seekingalpha.com': 'seekingalpha.com',
        'www.fool.com': 'fool.com',
        'www.tipranks.com': 'tipranks.com',
        'www.benzinga.com': 'benzinga.com',
        'www.barrons.com': 'barrons.com'
    }
    
    # Check for direct mapping first
    domain_lower = domain.lower().strip()
    if domain_lower in domain_mappings:
        return domain_mappings[domain_lower]
    
    # Standard normalization
    normalized = domain_lower
    if normalized.startswith('www.') and normalized != 'www.':
        normalized = normalized[4:]
    
    # Remove trailing slash
    normalized = normalized.rstrip('/')
    
    # Final check for any remaining mappings after normalization
    if normalized in domain_mappings:
        return domain_mappings[normalized]
    
    return normalized

class DomainResolver:
    def __init__(self):
        self._cache = {}
        self._common_mappings = {
            'reuters.com': 'Reuters',
            'bloomberg.com': 'Bloomberg News',
            'wsj.com': 'The Wall Street Journal',
            'cnbc.com': 'CNBC',
            'marketwatch.com': 'MarketWatch',
            'seekingalpha.com': 'Seeking Alpha',
            'fool.com': 'The Motley Fool',
            'tipranks.com': 'TipRanks',
            'benzinga.com': 'Benzinga',
            'investors.com': "Investor's Business Daily",
            'barrons.com': "Barron's",
            'ft.com': 'Financial Times',
            'theglobeandmail.com': 'The Globe and Mail',
        }

    def _resolve_publication_to_domain(self, publication_name: str) -> Optional[str]:
        """Resolve publication name to domain using database first, then AI fallback"""
        if not publication_name:
            return None
        
        clean_name = publication_name.lower().strip()
        
        # First, check if we already have this publication mapped in database
        try:
            with db() as conn, conn.cursor() as cur:
                # Look for existing mapping by formal_name (case insensitive)
                cur.execute("""
                    SELECT domain FROM domain_names 
                    WHERE LOWER(formal_name) = %s
                    LIMIT 1
                """, (clean_name,))
                result = cur.fetchone()
                
                if result:
                    return result["domain"]
                    
                # Also try variations (without "the", etc.)
                variations = [
                    clean_name.replace("the ", ""),
                    clean_name.replace(" the", ""),
                    clean_name + " news",
                    clean_name.replace(" news", "")
                ]
                
                for variation in variations:
                    if variation != clean_name:
                        cur.execute("""
                            SELECT domain FROM domain_names 
                            WHERE LOWER(formal_name) = %s
                            LIMIT 1
                        """, (variation,))
                        result = cur.fetchone()
                        if result:
                            return result["domain"]
                            
        except Exception as e:
            LOG.warning(f"Database lookup failed for '{publication_name}': {e}")
        
        # If not found in database, try AI resolution
        ai_domain = self._resolve_publication_to_domain_with_ai(publication_name)
        
        # If AI found a domain, store the mapping for future use
        if ai_domain:
            try:
                self._store_in_database(ai_domain, publication_name, True)
                LOG.info(f"Stored new mapping: '{publication_name}' -> '{ai_domain}'")
                return ai_domain
            except Exception as e:
                LOG.warning(f"Failed to store domain mapping: {e}")
                return ai_domain
        
        # FIXED: If AI fails, create fallback domain instead of returning None
        LOG.warning(f"AI failed to resolve '{publication_name}', creating fallback domain")
        return self._create_fallback_domain(publication_name)

    def _create_fallback_domain(self, publication_name: str) -> str:
        """Create a fallback domain when AI resolution fails"""
        if not publication_name:
            return "unknown-publication.com"
        
        # Clean the publication name
        clean_name = publication_name.lower().strip()
        
        # Remove common words
        words_to_remove = ['the', 'news', 'magazine', 'journal', 'times', 'post', 'daily', 'weekly']
        words = clean_name.split()
        filtered_words = [word for word in words if word not in words_to_remove]
        
        if not filtered_words:
            # If all words were removed, use the original
            filtered_words = words
        
        # Join words and clean up
        domain_base = ''.join(filtered_words[:3])  # Limit to first 3 words
        domain_base = ''.join(c for c in domain_base if c.isalnum())  # Remove special chars
        
        if len(domain_base) < 3:
            domain_base = "unknown"
        
        fallback_domain = f"{domain_base}.com"
        
        # Store in database as a fallback (not AI generated)
        try:
            self._store_in_database(fallback_domain, publication_name, False)
        except Exception as e:
            LOG.warning(f"Failed to store fallback domain {fallback_domain}: {e}")
        
        return fallback_domain

    def _resolve_publication_to_domain_with_ai(self, publication_name: str) -> Optional[str]:
        """Use AI to convert publication name to domain"""
        if not OPENAI_API_KEY or not publication_name:
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f'What is the primary domain name for the publication "{publication_name}"? Respond with just the domain (e.g., "reuters.com").'
            
            data = {
                "model": OPENAI_MODEL,
                "input": prompt,
                "max_output_tokens": 30,
                "reasoning": {"effort": "low"},
                "text": {"verbosity": "low"},
                "truncation": "auto"
            }
            
            response = get_openai_session().post(OPENAI_API_URL, headers=headers, json=data, timeout=(10, 180))
            if response.status_code == 200:
                result = response.json()
                
                # Log usage details
                u = result.get("usage", {}) or {}
                LOG.debug("Domain resolution usage — input:%s output:%s (cap:%s) status:%s",
                          u.get("input_tokens"), u.get("output_tokens"),
                          result.get("max_output_tokens"),
                          result.get("status"))
                
                domain = extract_text_from_responses(result).strip().lower()
                
                # Clean up common AI response patterns
                domain = domain.replace('"', '').replace("'", "").replace("www.", "")
                
                # Validate it looks like a domain
                if '.' in domain and len(domain) > 4 and len(domain) < 50 and not ' ' in domain:
                    normalized = normalize_domain(domain)
                    if normalized:
                        LOG.info(f"AI resolved '{publication_name}' -> '{normalized}'")
                        return normalized
            
        except Exception as e:
            LOG.warning(f"AI domain resolution failed for '{publication_name}': {e}")
        
        return None
    
    def resolve_url_and_domain(self, url, title=None):
        """Single method to resolve any URL to (final_url, domain, source_url)"""
        try:
            if "news.google.com" in url:
                return self._handle_google_news(url, title)
            elif any(yahoo_domain in url for yahoo_domain in ["finance.yahoo.com", "ca.finance.yahoo.com", "uk.finance.yahoo.com", "yahoo.com"]):
                return self._handle_yahoo_finance(url)
            else:
                return self._handle_direct_url(url)
        except Exception as e:
            LOG.warning(f"URL resolution failed for {url}: {e}")
            fallback_domain = urlparse(url).netloc.lower() if url else None
            return url, normalize_domain(fallback_domain), None
    
    def _resolve_google_news_url_advanced(self, url: str) -> Optional[str]:
        """
        Advanced Google News URL resolution using internal API
        Falls back to existing method if this fails
        """
        try:
            # Only attempt for Google News URLs
            if "news.google.com" not in url:
                return None
                
            headers = {
                'content-type': 'application/x-www-form-urlencoded;charset=UTF-8',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            }
            
            # Get the page content
            resp = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Find the c-wiz element with data-p attribute
            c_wiz = soup.select_one('c-wiz[data-p]')
            if not c_wiz:
                return None
                
            data_p = c_wiz.get('data-p')
            if not data_p:
                return None
                
            # Parse the embedded JSON
            obj = json.loads(data_p.replace('%.@.', '["garturlreq",'))
            
            # Prepare the payload for the internal API
            payload = {
                'f.req': json.dumps([[['Fbv4je', json.dumps(obj[:-6] + obj[-2:]), 'null', 'generic']]])
            }
            
            # Make the API call
            api_url = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
            response = requests.post(api_url, headers=headers, data=payload, timeout=10)
            
            if response.status_code != 200:
                return None
                
            # Parse the response
            response_text = response.text.replace(")]}'", "")
            response_json = json.loads(response_text)
            array_string = response_json[0][2]
            article_url = json.loads(array_string)[1]
            
            if article_url and len(article_url) > 10:  # Basic validation
                LOG.info(f"Advanced Google News resolution: {article_url}")
                return article_url
                
        except Exception as e:
            LOG.debug(f"Advanced Google News resolution failed for {url}: {e}")
            return None
        
        return None
    
    def get_formal_name(self, domain):
        """Get formal name for domain with caching"""
        if not domain:
            return "Unknown"
        
        clean_domain = normalize_domain(domain)
        if not clean_domain:
            return "Unknown"
        
        # Check cache
        if clean_domain in self._cache:
            return self._cache[clean_domain]
        
        # Check database
        formal_name = self._get_from_database(clean_domain)
        if formal_name:
            self._cache[clean_domain] = formal_name
            return formal_name
        
        # Check common mappings
        if clean_domain in self._common_mappings:
            formal_name = self._common_mappings[clean_domain]
            self._store_in_database(clean_domain, formal_name, False)
            self._cache[clean_domain] = formal_name
            return formal_name
        
        # Use AI as last resort
        if OPENAI_API_KEY:
            formal_name = self._get_from_ai(clean_domain)
            if formal_name:
                self._store_in_database(clean_domain, formal_name, True)
                self._cache[clean_domain] = formal_name
                return formal_name
        
        # Fallback
        fallback = clean_domain.replace('.com', '').replace('.org', '').title()
        self._store_in_database(clean_domain, fallback, False)
        self._cache[clean_domain] = fallback
        return fallback
    
    def _handle_google_news(self, url, title):
        """Handle Google News URL resolution with advanced and fallback methods"""
        
        # Try advanced resolution first
        advanced_url = self._resolve_google_news_url_advanced(url)
        if advanced_url:
            domain = normalize_domain(urlparse(advanced_url).netloc.lower())
            if not self._is_spam_domain(domain):
                LOG.info(f"Advanced resolution: {url} -> {advanced_url}")
                return advanced_url, domain, None
            else:
                LOG.info(f"SPAM REJECTED: Advanced resolution found spam domain {domain}")
                return None, None, None  # Reject entirely, don't fall back
        
        # Fall back to direct resolution method
        try:
            response = requests.get(url, timeout=10, allow_redirects=True, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            final_url = response.url
            
            if final_url != url and "news.google.com" not in final_url:
                domain = normalize_domain(urlparse(final_url).netloc.lower())
                if not self._is_spam_domain(domain):
                    LOG.info(f"Direct resolution: {url} -> {final_url}")
                    return final_url, domain, None
                else:
                    LOG.info(f"SPAM REJECTED: Direct resolution found spam domain {domain}")
                    return None, None, None  # Reject entirely
        except:
            pass
        
        # Title extraction fallback - also check for spam
        if title and not contains_non_latin_script(title):
            clean_title, source = extract_source_from_title_smart(title)
            if source and not self._is_spam_source(source):
                resolved_domain = self._resolve_publication_to_domain(source)
                if resolved_domain:
                    if not self._is_spam_domain(resolved_domain):
                        LOG.info(f"Title resolution: {source} -> {resolved_domain}")
                        return url, resolved_domain, None
                    else:
                        LOG.info(f"SPAM REJECTED: Title resolution found spam domain {resolved_domain}")
                        return None, None, None
                else:
                    LOG.warning(f"Could not resolve publication '{source}' to domain")
                    return url, "google-news-unresolved", None
        
        return url, "google-news-unresolved", None
    
    def _handle_yahoo_finance(self, url):
        """Handle Yahoo Finance URL resolution - reject all failures"""
        original_source = extract_yahoo_finance_source_optimized(url)
        if original_source:
            domain = normalize_domain(urlparse(original_source).netloc.lower())
            if not self._is_spam_domain(domain):
                LOG.info(f"YAHOO SUCCESS: Resolved {url} -> {original_source}")
                return original_source, domain, url
        
        # REJECT ALL FAILED YAHOO URLs - don't return Yahoo URLs to system
        LOG.info(f"YAHOO REJECTED: Failed to resolve {url} - discarding")
        return None, None, None
    
    def _handle_direct_url(self, url):
        """Handle direct URL"""
        domain = normalize_domain(urlparse(url).netloc.lower())
        if self._is_spam_domain(domain):
            return None, None, None
        return url, domain, None
    
    def _is_spam_domain(self, domain):
        """Check if domain is spam"""
        if not domain:
            return True
        return any(spam in domain for spam in SPAM_DOMAINS)
    
    def _is_spam_source(self, source):
        """Check if source name is spam"""
        if not source:
            return True
        source_lower = source.lower()
        return any(spam in source_lower for spam in ["marketbeat", "newser", "khodrobank"])
    
    def _get_from_database(self, domain):
        """Get formal name from database"""
        try:
            with db() as conn, conn.cursor() as cur:
                cur.execute("SELECT formal_name FROM domain_names WHERE domain = %s", (domain,))
                result = cur.fetchone()
                return result["formal_name"] if result else None
        except:
            return None
    
    def _store_in_database(self, domain, formal_name, ai_generated):
        """Store formal name in database"""
        try:
            with db() as conn, conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO domain_names (domain, formal_name, ai_generated)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (domain) DO UPDATE
                    SET formal_name = EXCLUDED.formal_name, updated_at = NOW()
                """, (domain, formal_name, ai_generated))
        except Exception as e:
            LOG.warning(f"Failed to store domain mapping {domain}: {e}")
    
    def _get_from_ai(self, domain):
        """Get formal name from AI"""
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f'What is the formal publication name for "{domain}"? Respond with just the name.'
            
            data = {
                "model": OPENAI_MODEL,
                "input": prompt,
                "max_output_tokens": 50,
                "reasoning": {"effort": "low"},
                "text": {"verbosity": "low"},
                "truncation": "auto"
            }
            
            response = get_openai_session().post(OPENAI_API_URL, headers=headers, json=data, timeout=(10, 180))
            if response.status_code == 200:
                result = response.json()
                
                # Log usage details
                u = result.get("usage", {}) or {}
                LOG.debug("Formal name lookup usage — input:%s output:%s (cap:%s) status:%s",
                          u.get("input_tokens"), u.get("output_tokens"),
                          result.get("max_output_tokens"),
                          result.get("status"))
                
                name = extract_text_from_responses(result).strip()
                return name if 2 < len(name) < 100 else None
        except Exception as e:
            LOG.debug(f"AI formal name lookup failed for {domain}: {e}")
        return None

# Create global instance
domain_resolver = DomainResolver()

class FeedManager:
    @staticmethod
    def create_feeds_for_ticker(ticker: str, metadata: Dict) -> List[Dict]:
        """Create feeds only if under the limits - FIXED competitor counting logic with strict ticker requirement"""
        feeds = []
        company_name = metadata.get("company_name", ticker)
        
        LOG.info(f"CREATING FEEDS for {ticker} ({company_name}):")
        
        # Check existing feed counts by unique competitor (not by feed count)
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT category, COUNT(*) as count
                FROM source_feed 
                WHERE ticker = %s AND active = TRUE
                GROUP BY category
            """, (ticker,))
            
            existing_data = {row["category"]: row for row in cur.fetchall()}
            
            # Extract counts
            existing_company_count = existing_data.get('company', {}).get('count', 0)
            existing_industry_count = existing_data.get('industry', {}).get('count', 0)
            
            # Count unique competitors separately
            cur.execute("""
                SELECT COUNT(DISTINCT competitor_ticker) as unique_competitors
                FROM source_feed 
                WHERE ticker = %s AND category = 'competitor' AND active = TRUE
                AND competitor_ticker IS NOT NULL
            """, (ticker,))
            result = cur.fetchone()
            existing_competitor_entities = result["unique_competitors"] if result else 0
            
            LOG.info(f"  EXISTING FEEDS: Company={existing_company_count}, Industry={existing_industry_count}, Competitors={existing_competitor_entities} unique entities")
        
        # Company feeds - always ensure we have the core 2
        if existing_company_count < 2:
            company_feeds = [
                {
                    "url": f"https://news.google.com/rss/search?q=\"{requests.utils.quote(company_name)}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
                    "name": f"Google News: {company_name}",
                    "category": "company",
                    "search_keyword": company_name
                },
                {
                    "url": f"https://finance.yahoo.com/rss/headline?s={ticker}",
                    "name": f"Yahoo Finance: {ticker}",
                    "category": "company",
                    "search_keyword": ticker
                }
            ]
            feeds.extend(company_feeds)
            LOG.info(f"  COMPANY FEEDS: Adding {len(company_feeds)} (existing: {existing_company_count})")
        else:
            LOG.info(f"  COMPANY FEEDS: Skipping - already have {existing_company_count}")
        
        # Industry feeds - MAX 3 TOTAL (1 per keyword, max 3 keywords)
        if existing_industry_count < 3:
            available_slots = 3 - existing_industry_count
            industry_keywords = metadata.get("industry_keywords", [])[:available_slots]
            
            LOG.info(f"  INDUSTRY FEEDS: Can add {available_slots} more (existing: {existing_industry_count}, keywords available: {len(metadata.get('industry_keywords', []))})")
            
            for keyword in industry_keywords:
                feed = {
                    "url": f"https://news.google.com/rss/search?q=\"{requests.utils.quote(keyword)}\"+when:7d&hl=en-US&gl=US&ceid=US:en",
                    "name": f"Industry: {keyword}",
                    "category": "industry",
                    "search_keyword": keyword
                }
                feeds.append(feed)
                LOG.info(f"    INDUSTRY: {keyword}")
        else:
            LOG.info(f"  INDUSTRY FEEDS: Skipping - already at limit (3/3)")
        
        # Competitor feeds - MAX 3 UNIQUE COMPETITORS (each competitor can have multiple feeds but counts as 1 entity)
        if existing_competitor_entities < 3:
            available_competitor_slots = 3 - existing_competitor_entities
            competitors = metadata.get("competitors", [])[:available_competitor_slots]
            
            LOG.info(f"  COMPETITOR ENTITIES: Can add {available_competitor_slots} more competitors (existing: {existing_competitor_entities}, available: {len(metadata.get('competitors', []))})")
            
            # Get existing competitor tickers to avoid duplicates
            with db() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT competitor_ticker 
                    FROM source_feed 
                    WHERE ticker = %s AND category = 'competitor' AND active = TRUE
                    AND competitor_ticker IS NOT NULL
                """, (ticker,))
                existing_competitor_tickers = {row["competitor_ticker"] for row in cur.fetchall()}
            
            LOG.info(f"DEBUG: Processing {len(competitors)} competitors for {ticker}")
            for i, comp in enumerate(competitors):
                LOG.info(f"DEBUG: Competitor {i}: {comp} (type: {type(comp)})")
                
                comp_name = None
                comp_ticker = None
                
                if isinstance(comp, dict):
                    comp_name = comp.get('name', '')
                    comp_ticker = comp.get('ticker')
                    LOG.info(f"DEBUG: Dict competitor - Name: '{comp_name}', Ticker: '{comp_ticker}'")
                elif isinstance(comp, str):
                    LOG.info(f"DEBUG: String competitor: {comp}")
                    # Try to parse "Name (TICKER)" format
                    match = re.search(r'^(.+?)\s*\(([A-Z]{1,5})\)$', comp)
                    if match:
                        comp_name = match.group(1).strip()
                        comp_ticker = match.group(2)
                        LOG.info(f"DEBUG: Parsed competitor - Name: '{comp_name}', Ticker: '{comp_ticker}'")
                    else:
                        LOG.info(f"DEBUG: Skipping competitor - no ticker found in string: {comp}")
                        continue
                
                # Strict validation - REQUIRE both name and ticker
                if not comp_name or not comp_ticker:
                    LOG.info(f"DEBUG: Skipping competitor - missing name or ticker: name='{comp_name}', ticker='{comp_ticker}'")
                    continue
                    
                if comp_ticker.upper() == ticker.upper():
                    LOG.info(f"DEBUG: Skipping competitor - same as main ticker: {comp_ticker}")
                    continue
                    
                if comp_ticker in existing_competitor_tickers:
                    LOG.info(f"DEBUG: Skipping competitor - already exists: {comp_ticker}")
                    continue
                
                # Validate ticker format (1-5 uppercase letters)
                if not re.match(r'^[A-Z]{1,5}$', comp_ticker):
                    LOG.info(f"DEBUG: Skipping competitor - invalid ticker format: '{comp_ticker}'")
                    continue
                
                # Create feeds for this competitor (BOTH Google News and Yahoo Finance)
                LOG.info(f"DEBUG: Creating feeds for competitor {comp_name} ({comp_ticker})")
                comp_feeds = [
                    {
                        "url": f"https://news.google.com/rss/search?q=\"{requests.utils.quote(comp_name)}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
                        "name": f"Competitor: {comp_name}",
                        "category": "competitor",
                        "search_keyword": comp_name,
                        "competitor_ticker": comp_ticker
                    },
                    {
                        "url": f"https://finance.yahoo.com/rss/headline?s={comp_ticker}",
                        "name": f"Yahoo Competitor: {comp_name} ({comp_ticker})",
                        "category": "competitor",
                        "search_keyword": comp_name,
                        "competitor_ticker": comp_ticker
                    }
                ]
                
                feeds.extend(comp_feeds)
                LOG.info(f"    COMPETITOR: {comp_name} ({comp_ticker}) - 2 feeds (counts as 1 entity)")
        else:
            LOG.info(f"  COMPETITOR ENTITIES: Skipping - already at limit (3/3 unique competitors)")
        
        LOG.info(f"TOTAL FEEDS TO CREATE: {len(feeds)}")
        return feeds
    
    @staticmethod
    def store_feeds(ticker: str, feeds: List[Dict], retain_days: int = 90) -> List[int]:
        """Store feeds in database"""
        feed_ids = []
        
        with db() as conn, conn.cursor() as cur:
            for feed in feeds:
                cur.execute("""
                    INSERT INTO source_feed (url, name, ticker, retain_days, active, search_keyword, competitor_ticker)
                    VALUES (%s, %s, %s, %s, TRUE, %s, %s)
                    ON CONFLICT (url) DO UPDATE
                    SET name = EXCLUDED.name, ticker = EXCLUDED.ticker, active = TRUE
                    RETURNING id;
                """, (
                    feed["url"], feed["name"], ticker, retain_days,
                    feed.get("search_keyword"), feed.get("competitor_ticker")
                ))
                result = cur.fetchone()
                if result:
                    feed_ids.append(result["id"])
        
        return feed_ids

# Global instance
feed_manager = FeedManager()

class TickerManager:
    @staticmethod
    def get_or_create_metadata(ticker: str, force_refresh: bool = False) -> Dict:
        """Unified ticker metadata management - ALWAYS returns a valid dict"""
        # Check database first
        if not force_refresh:
            with db() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT ticker, name, industry_keywords, competitors, ai_generated
                    FROM ticker_config WHERE ticker = %s AND active = TRUE
                """, (ticker,))
                config = cur.fetchone()
                
                if config:
                    # Process competitors back to structured format
                    competitors = []
                    for comp_str in config.get("competitors", []):
                        match = re.search(r'^(.+?)\s*\(([A-Z]{1,5})\)$', comp_str)
                        if match:
                            competitors.append({"name": match.group(1).strip(), "ticker": match.group(2)})
                        else:
                            competitors.append({"name": comp_str, "ticker": None})
                    
                    return {
                        "company_name": config.get("name", ticker),
                        "industry_keywords": config.get("industry_keywords", []),
                        "competitors": competitors
                    }
        
        # Generate with AI
        ai_metadata = generate_enhanced_ticker_metadata_with_ai(ticker)
        
        # ALWAYS store something, even if AI failed
        if ai_metadata:
            TickerManager.store_metadata(ticker, ai_metadata)
            return ai_metadata
        else:
            # Create and store fallback metadata
            fallback_metadata = {
                "company_name": ticker,
                "industry_keywords": [],
                "competitors": []
            }
            TickerManager.store_metadata(ticker, fallback_metadata)
            return fallback_metadata
    
    @staticmethod
    def store_metadata(ticker: str, metadata: Dict):
        """Store enhanced ticker metadata in database with robust error handling"""
        
        # Handle None or invalid metadata
        if not metadata or not isinstance(metadata, dict):
            LOG.warning(f"Invalid or missing metadata for {ticker}, creating fallback")
            metadata = {
                "company_name": ticker,
                "industry_keywords": [],
                "competitors": [],
                "sector": "",
                "industry": "",
                "sub_industry": "",
                "sector_profile": {},
                "aliases_brands_assets": {}
            }
        
        # Ensure required fields exist with defaults
        metadata.setdefault("company_name", ticker)
        metadata.setdefault("industry_keywords", [])
        metadata.setdefault("competitors", [])
        metadata.setdefault("sector", "")
        metadata.setdefault("industry", "")
        metadata.setdefault("sub_industry", "")
        metadata.setdefault("sector_profile", {})
        metadata.setdefault("aliases_brands_assets", {})
        
        # Convert competitors to storage format
        competitors_for_db = []
        structured_competitors = []  # Keep structured format for competitor_metadata table
        
        try:
            for comp in metadata.get("competitors", []):
                if isinstance(comp, dict):
                    structured_competitors.append(comp)  # For new table
                    if comp.get('ticker'):
                        competitors_for_db.append(f"{comp['name']} ({comp['ticker']})")
                    else:
                        competitors_for_db.append(comp.get('name', 'Unknown'))
                else:
                    competitors_for_db.append(str(comp))
        except Exception as e:
            LOG.warning(f"Error processing competitors for {ticker}: {e}")
            competitors_for_db = []
            structured_competitors = []
        
        try:
            with db() as conn, conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ticker_config (
                        ticker, name, industry_keywords, competitors, ai_generated,
                        sector, industry, sub_industry, sector_profile, aliases_brands_assets
                    ) VALUES (%s, %s, %s, %s, TRUE, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker) DO UPDATE
                    SET name = EXCLUDED.name, 
                        industry_keywords = EXCLUDED.industry_keywords,
                        competitors = EXCLUDED.competitors, 
                        sector = EXCLUDED.sector,
                        industry = EXCLUDED.industry,
                        sub_industry = EXCLUDED.sub_industry,
                        sector_profile = EXCLUDED.sector_profile,
                        aliases_brands_assets = EXCLUDED.aliases_brands_assets,
                        updated_at = NOW()
                """, (
                    ticker, 
                    metadata.get("company_name", ticker),
                    metadata.get("industry_keywords", []), 
                    competitors_for_db,
                    metadata.get("sector", ""),
                    metadata.get("industry", ""),
                    metadata.get("sub_industry", ""),
                    json.dumps(metadata.get("sector_profile", {})),
                    json.dumps(metadata.get("aliases_brands_assets", {}))
                ))
                
                LOG.info(f"Successfully stored metadata for {ticker}")
                
        except Exception as e:
            LOG.error(f"Database error storing metadata for {ticker}: {e}")
            # Don't raise - we want the system to continue with fallback data
            return
        
        # Store competitor metadata in dedicated table (with error handling)
        if structured_competitors:
            try:
                store_competitor_metadata(ticker, structured_competitors)
            except Exception as e:
                LOG.warning(f"Failed to store competitor metadata for {ticker}: {e}")
                # Continue - this is not critical


# Global instances
ticker_manager = TickerManager()
feed_manager = FeedManager()

def generate_enhanced_ticker_metadata_with_ai(ticker: str, company_name: str = None, sector: str = "", industry: str = "") -> Optional[Dict]:
    """
    Enhanced AI generation with company context from ticker reference table
    """
    if company_name is None:
        company_name = ticker
    
    if not OPENAI_API_KEY:
        LOG.warning("Missing OPENAI_API_KEY; skipping metadata generation")
        return None

    system_prompt = """You are a financial analyst creating metadata for a hedge fund's stock monitoring system. Generate precise, actionable metadata that will be used for news article filtering and triage.

CRITICAL REQUIREMENTS:
- All competitors must be currently publicly traded with valid ticker symbols
- Industry keywords must be SPECIFIC enough to avoid false positives in news filtering, but not so narrow that they miss material news.
- Benchmarks must be sector-specific, not generic market indices
- All information must be factually accurate
- The company name MUST be the official legal name (e.g., "Prologis Inc" not "PLD")
- If any field is unknown, output an empty array for lists and omit optional fields. Never refuse; always return a valid JSON object.

INDUSTRY KEYWORDS (exactly 3):
- Must be SPECIFIC to the company's primary business
- Return proper capitalization format like "Digital Advertising"
- Avoid generic terms like "Technology", "Healthcare", "Energy", "Oil", "Services"
- Use compound terms or specific product categories
- Examples: "Smartphone Manufacturing" not "Technology", "Upstream Oil Production" not "Oil"

COMPETITORS (exactly 3):
- Must be direct business competitors, not just same-sector companies
- Must be currently publicly traded (check acquisition status)
- Format: "Company Name (TICKER)" - verify ticker is correct and current
- Exclude: Private companies, subsidiaries, companies acquired in last 2 years

Generate response in valid JSON format with all required fields. Be concise and precise."""

    # Enhanced prompt with context from ticker reference table
    context_info = f"Company: {company_name} ({ticker})"
    if sector:
        context_info += f", Sector: {sector}"
    if industry:
        context_info += f", Industry: {industry}"

    user_prompt = f"""Generate metadata for hedge fund news monitoring. Focus on precision to avoid irrelevant news articles.

{context_info}

Since we have basic company information, focus on generating specific industry keywords and direct competitors with accurate tickers.

CRITICAL: The "company_name" field should be: {company_name}

Required JSON format:
{{
    "ticker": "{ticker}",
    "company_name": "{company_name}",
    "sector": "{sector if sector else 'GICS Sector'}",
    "industry": "{industry if industry else 'GICS Industry'}",
    "sub_industry": "GICS Sub-Industry",
    "industry_keywords": ["keyword1", "keyword2", "keyword3"],
    "competitors": ["Company Name (TICKER)", "Company Name (TICKER)", "Company Name (TICKER)"],
    "sector_profile": {{
        "core_inputs": ["input1", "input2", "input3"],
        "core_channels": ["channel1", "channel2", "channel3"],
        "core_geos": ["geo1", "geo2", "geo3"],
        "benchmarks": ["benchmark1", "benchmark2", "benchmark3"]
    }},
    "aliases_brands_assets": {{
        "aliases": ["alias1", "alias2", "alias3"],
        "brands": ["brand1", "brand2", "brand3"],
        "assets": ["asset1", "asset2", "asset3"]
    }}
}}"""

    brief_user_prompt = f"""Generate compact JSON metadata for {company_name} ({ticker}):
{{
    "ticker": "{ticker}",
    "company_name": "{company_name}",
    "sector": "{sector if sector else 'GICS Sector'}",
    "industry": "{industry if industry else 'GICS Industry'}", 
    "sub_industry": "GICS Sub-Industry",
    "industry_keywords": ["k1", "k2", "k3"],
    "competitors": ["Co1 (TKR1)", "Co2 (TKR2)", "Co3 (TKR3)"],
    "sector_profile": {{"core_inputs": ["i1","i2","i3"], "core_channels": ["c1","c2","c3"], "core_geos": ["g1","g2","g3"], "benchmarks": ["b1","b2","b3"]}},
    "aliases_brands_assets": {{"aliases": ["a1","a2","a3"], "brands": ["b1","b2","b3"], "assets": ["as1","as2","as3"]}}
}}
Use exactly 3 items per list. Be brief and specific."""

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # First attempt with optimized settings
        data = {
            "model": OPENAI_MODEL,
            "input": f"{system_prompt}\n\n{user_prompt}",
            "max_output_tokens": 5000,
            "reasoning": {"effort": "medium"},
            "text": {
                "format": {"type": "json_object"},
                "verbosity": "low"
            },
            "truncation": "auto"
        }
        
        response = get_openai_session().post(OPENAI_API_URL, headers=headers, json=data, timeout=(10, 180))
        
        if response.status_code != 200:
            print(f"API error {response.status_code}: {response.text}")
            return None
        
        result = response.json()
        text = extract_text_from_responses(result)
        
        # Log usage details for debugging
        u = result.get("usage", {}) or {}
        LOG.info("Enhanced OpenAI usage — input:%s output:%s (cap:%s) status:%s reason:%s",
                 u.get("input_tokens"), u.get("output_tokens"),
                 result.get("max_output_tokens"),
                 result.get("status"),
                 (result.get("incomplete_details") or {}).get("reason"))
        
        # Check for any incomplete response
        status = result.get("status")
        incomplete = result.get("incomplete_details", {}) or {}
        
        # If no text and response incomplete, retry with smaller prompt
        if (not text) and status == "incomplete":
            LOG.warning(f"OpenAI response incomplete for {ticker} metadata (reason: {incomplete.get('reason')}); attempting smaller prompt")
            
            retry_data = {
                "model": OPENAI_MODEL,
                "input": f"{system_prompt}\n\n{brief_user_prompt}",
                "max_output_tokens": 1500,
                "reasoning": {"effort": "low"},
                "text": {
                    "format": {"type": "json_object"},
                    "verbosity": "low"
                },
                "truncation": "auto"
            }
            
            retry_response = get_openai_session().post(OPENAI_API_URL, headers=headers, json=retry_data, timeout=(10, 180))
            if retry_response.status_code == 200:
                retry_result = retry_response.json()
                
                # Log retry usage too
                retry_u = retry_result.get("usage", {}) or {}
                LOG.info("Enhanced OpenAI retry usage — input:%s output:%s (cap:%s) status:%s reason:%s",
                         retry_u.get("input_tokens"), retry_u.get("output_tokens"),
                         retry_result.get("max_output_tokens"),
                         retry_result.get("status"),
                         (retry_result.get("incomplete_details") or {}).get("reason"))
                
                text = extract_text_from_responses(retry_result)
                if text:
                    LOG.info(f"Enhanced brief prompt successful for {ticker}")
        
        if not text:
            print(f"No content returned for {ticker} even after retry")
            return None
        
        # Parse JSON with robust fallback
        metadata = parse_json_with_fallback(text, ticker)
        
        # Process the results using existing logic
        def _list3(x): 
            if isinstance(x, (list, tuple)):
                items = [item for item in list(x)[:3] if item]
                return items
            return []
        
        metadata.setdefault("ticker", ticker)
        metadata["name"] = company_name  # Use the provided company name
        metadata["company_name"] = company_name  # Ensure both fields are set
        
        metadata.setdefault("sector", sector)
        metadata.setdefault("industry", industry)
        metadata.setdefault("sub_industry", "")
        
        # Normalize lists (no padding)
        metadata["industry_keywords"] = _list3(metadata.get("industry_keywords", []))
        metadata["competitors"] = _list3(metadata.get("competitors", []))
        
        # Ensure nested objects exist
        sector_profile = metadata.setdefault("sector_profile", {})
        aliases_brands = metadata.setdefault("aliases_brands_assets", {})
        
        sector_profile["core_inputs"] = _list3(sector_profile.get("core_inputs", []))
        sector_profile["core_channels"] = _list3(sector_profile.get("core_channels", []))
        sector_profile["core_geos"] = _list3(sector_profile.get("core_geos", []))
        sector_profile["benchmarks"] = _list3(sector_profile.get("benchmarks", []))
        
        aliases_brands["aliases"] = _list3(aliases_brands.get("aliases", []))
        aliases_brands["brands"] = _list3(aliases_brands.get("brands", []))
        aliases_brands["assets"] = _list3(aliases_brands.get("assets", []))
        
        LOG.info(f"Enhanced metadata generated for {ticker}: {company_name}")
        return metadata
            
    except Exception as e:
        print(f"Error generating enhanced metadata for {ticker}: {e}")
        return None


def get_or_create_enhanced_ticker_metadata(ticker: str, force_refresh: bool = False) -> Dict:
    """Get ticker metadata with reference table lookup first, then AI enhancement"""
    
    # Step 1: Check ticker_reference table
    reference_data = get_ticker_reference(ticker)
    
    if reference_data and not force_refresh:
        # Use reference data as base
        metadata = {
            "ticker": ticker,
            "company_name": reference_data["company_name"],
            "name": reference_data["company_name"],  # Both fields for compatibility
            "sector": reference_data.get("sector", ""),
            "industry": reference_data.get("industry", ""),
            "industry_keywords": reference_data.get("industry_keywords", []),
            "competitors": reference_data.get("competitors", [])
        }
        
        # If reference data lacks AI-generated keywords/competitors, enhance with AI
        if (not reference_data.get("industry_keywords") or 
            not reference_data.get("competitors") or 
            not reference_data.get("ai_generated")):
            
            LOG.info(f"Enhancing {ticker} ({reference_data['company_name']}) with AI generation")
            ai_metadata = generate_enhanced_ticker_metadata_with_ai(
                ticker, 
                reference_data["company_name"],
                reference_data.get("sector", ""),
                reference_data.get("industry", "")
            )
            
            if ai_metadata:
                # Merge AI enhancements with reference data
                metadata.update({
                    "industry_keywords": ai_metadata.get("industry_keywords", []),
                    "competitors": ai_metadata.get("competitors", []),
                    "sector_profile": ai_metadata.get("sector_profile", {}),
                    "aliases_brands_assets": ai_metadata.get("aliases_brands_assets", {})
                })
                
                # Update reference table with AI enhancements
                update_ticker_reference_ai_data(ticker, metadata)
        
        return metadata
    
    # Step 2: Fall back to original AI generation (for unknown tickers)
    LOG.info(f"No reference data found for {ticker}, using original AI generation")
    return generate_enhanced_ticker_metadata_with_ai(ticker)


def get_ticker_reference(ticker: str) -> Optional[Dict]:
    """Get ticker reference data from database"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT ticker, country, company_name, industry, sector,
                   exchange, active, industry_keywords, competitors, ai_generated
            FROM ticker_reference
            WHERE ticker = %s AND active = TRUE
        """, (ticker,))
        return cur.fetchone()


def update_ticker_reference_ai_data(ticker: str, metadata: Dict):
    """Update reference table with AI-generated enhancements"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            UPDATE ticker_reference
            SET industry_keywords = %s,
                competitors = %s,
                ai_generated = TRUE,
                updated_at = NOW()
            WHERE ticker = %s
        """, (
            metadata.get("industry_keywords", []),
            metadata.get("competitors", []),
            ticker
        ))
        LOG.info(f"Updated {ticker} reference table with AI enhancements")

def validate_metadata(metadata):
    """
    Validate metadata quality and return warnings
    """
    warnings = []
    
    # Forbidden generic keywords
    forbidden_keywords = [
        'Technology', 'Healthcare', 'Energy', 'Oil', 'Services', 
        'Software', 'Hardware', 'Consumer', 'Financial', 'Industrial'
    ]
    
    for keyword in metadata.get('industry_keywords', []):
        if keyword in forbidden_keywords:
            warnings.append(f"Generic keyword detected: '{keyword}'")
    
    # Check competitor ticker format
    ticker_pattern = r'^.+\([A-Z0-9]{1,6}(?:\.[A-Z]{1,3})?\)$'
    for competitor in metadata.get('competitors', []):
        if not re.match(ticker_pattern, competitor):
            warnings.append(f"Invalid competitor format: '{competitor}'")
    
    # Check for generic benchmarks
    generic_benchmarks = ['S&P 500', 'NASDAQ', 'Dow Jones', 'Russell 2000']
    benchmarks = metadata.get('sector_profile', {}).get('benchmarks', [])
    
    generic_count = sum(1 for b in benchmarks if any(g in b for g in generic_benchmarks))
    if generic_count == len(benchmarks) and len(benchmarks) > 0:
        warnings.append("All benchmarks are generic market indices")
    
    # Check for generic core inputs
    generic_inputs = ['raw materials', 'capital', 'labor', 'technology', 'supply chain']
    core_inputs = metadata.get('sector_profile', {}).get('core_inputs', [])
    
    for inp in core_inputs:
        if inp.lower() in generic_inputs:
            warnings.append(f"Generic core input: '{inp}'")
    
    return warnings

def store_ticker_metadata(ticker, name, sector, industry, sub_industry, 
                         industry_keywords, competitors, sector_profile, aliases_brands_assets):
    """
    Store ticker metadata in database
    """
    try:
        with db() as conn:
            with conn.cursor() as cur:
                # Check if ticker already exists
                cur.execute("SELECT ticker FROM ticker_config WHERE ticker = %s", (ticker,))
                exists = cur.fetchone()
                
                if exists:
                    # Update existing
                    cur.execute("""
                        UPDATE ticker_config SET
                            name = %s,
                            sector = %s, 
                            industry = %s,
                            sub_industry = %s,
                            industry_keywords = %s,
                            competitors = %s,
                            sector_profile = %s,
                            aliases_brands_assets = %s,
                            ai_generated = TRUE,
                            updated_at = NOW()
                        WHERE ticker = %s
                    """, (
                        name, sector, industry, sub_industry,
                        industry_keywords, competitors,
                        json.dumps(sector_profile), json.dumps(aliases_brands_assets),
                        ticker
                    ))
                else:
                    # Insert new
                    cur.execute("""
                        INSERT INTO ticker_config (
                            ticker, name, sector, industry, sub_industry,
                            industry_keywords, competitors, sector_profile, 
                            aliases_brands_assets, ai_generated, active
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, TRUE)
                    """, (
                        ticker, name, sector, industry, sub_industry,
                        industry_keywords, competitors,
                        json.dumps(sector_profile), json.dumps(aliases_brands_assets)
                    ))
                
                conn.commit()
                return True
                
    except Exception as e:
        print(f"Database error storing {ticker}: {e}")
        return False

def resolve_google_news_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Wrapper for backward compatibility"""
    return domain_resolver.resolve_url_and_domain(url)

def get_or_create_formal_domain_name(domain: str) -> str:
    """Wrapper for backward compatibility"""
    return domain_resolver.get_formal_name(domain)

def cleanup_domain_data():
    """One-time script to clean up existing domain data"""
    
    # Mapping of publication names to actual domains
    cleanup_mappings = {
        'yahoo finance': 'finance.yahoo.com',
        'reuters': 'reuters.com',
        'bloomberg': 'bloomberg.com',
        'cnbc': 'cnbc.com',
        'forbes': 'forbes.com',
        'business insider': 'businessinsider.com',
        'the motley fool': 'fool.com',
        'seeking alpha': 'seekingalpha.com',
        'marketwatch': 'marketwatch.com',
        'nasdaq': 'nasdaq.com',
        'thestreet': 'thestreet.com',
        'benzinga': 'benzinga.com',
        'tipranks': 'tipranks.com',
        'morningstar': 'morningstar.com',
        'investor\'s business daily': 'investors.com',
        'zacks investment research': 'zacks.com',
        'stocktwits': 'stocktwits.com',
        'globenewswire': 'globenewswire.com',
        'business wire': 'businesswire.com',
        'pr newswire': 'prnewswire.com',
        'financialcontent': 'financialcontent.com',
        'insider monkey': 'insidermonkey.com',
        'gurufocus': 'gurufocus.com',
        'american banker': 'americanbanker.com',
        'thinkadvisor': 'thinkadvisor.com',
        'investmentnews': 'investmentnews.com',
        'the globe and mail': 'theglobeandmail.com',
        'financial news london': 'fnlondon.com',
        'steel market update': 'steelmarketupdate.com',
        'times of india': 'timesofindia.indiatimes.com',
        'business standard': 'business-standard.com',
        'fortune india': 'fortuneindia.com',
        'the new indian express': 'newindianexpress.com',
        'quiver quantitative': 'quiverquant.com',
        'stock titan': 'stocktitan.net',
        'modern healthcare': 'modernhealthcare.com'
    }
    
    with db() as conn, conn.cursor() as cur:
        total_updated = 0
        
        for old_domain, new_domain in cleanup_mappings.items():
            cur.execute("""
                UPDATE found_url 
                SET domain = %s 
                WHERE domain = %s
            """, (new_domain, old_domain))
            
            updated_count = cur.rowcount
            total_updated += updated_count
            
            if updated_count > 0:
                LOG.info(f"Updated {updated_count} records: '{old_domain}' -> '{new_domain}'")
        
        # Handle Yahoo regional consolidation
        cur.execute("""
            UPDATE found_url 
            SET domain = 'finance.yahoo.com' 
            WHERE domain IN ('ca.finance.yahoo.com', 'uk.finance.yahoo.com', 'sg.finance.yahoo.com')
        """)
        yahoo_updated = cur.rowcount
        total_updated += yahoo_updated
        
        if yahoo_updated > 0:
            LOG.info(f"Consolidated {yahoo_updated} Yahoo regional domains to finance.yahoo.com")
        
        # Handle duplicate benzinga entries
        cur.execute("""
            UPDATE found_url 
            SET domain = 'benzinga.com' 
            WHERE domain = 'benzinga'
        """)
        benzinga_updated = cur.rowcount
        total_updated += benzinga_updated
        
        if benzinga_updated > 0:
            LOG.info(f"Consolidated {benzinga_updated} benzinga entries to benzinga.com")
        
        LOG.info(f"Total domain cleanup: {total_updated} records updated")
        
        return total_updated

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
    """Format datetime to EST without time emoji"""
    if not dt:
        return "N/A"
    
    # Ensure we have a timezone-aware datetime
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    # Convert to Eastern Time
    eastern = pytz.timezone('US/Eastern')
    est_time = dt.astimezone(eastern)
    
    # Format without emoji
    time_part = est_time.strftime("%I:%M%p").lower().lstrip('0')
    date_part = est_time.strftime("%b %d")
    
    return f"{date_part}, {time_part} EST"

def is_description_valuable(title: str, description: str) -> bool:
    """Check if description adds value beyond the title"""
    if not description or len(description.strip()) < 10:
        return False
    
    # Clean both title and description for comparison
    clean_title = re.sub(r'[^\w\s]', ' ', title.lower()).strip()
    clean_desc = re.sub(r'[^\w\s]', ' ', description.lower()).strip()
    
    # Remove HTML tags from description
    clean_desc = re.sub(r'<[^>]+>', '', clean_desc)
    
    # Check for URL patterns (common in bad descriptions)
    url_patterns = [
        r'https?://',
        r'www\.',
        r'\.com',
        r'news\.google\.com',
        r'href=',
        r'CBM[A-Za-z0-9]{20,}',  # Google News encoded URLs
    ]
    
    for pattern in url_patterns:
        if re.search(pattern, description, re.IGNORECASE):
            return False
    
    # Check if description is just a truncated version of title
    title_words = set(clean_title.split())
    desc_words = set(clean_desc.split())
    
    # If description is mostly just title words, skip it
    if len(title_words) > 0:
        overlap = len(title_words.intersection(desc_words)) / len(title_words)
        if overlap > 0.8:  # 80% overlap suggests redundancy
            return False
    
    # Check if description starts with title (common redundancy)
    if clean_desc.startswith(clean_title[:min(len(clean_title), 50)]):
        return False
    
    # Check for single character descriptions
    if len(clean_desc.strip()) == 1:
        return False
    
    # Check for descriptions that are just fragments
    if len(clean_desc) < 30 and not clean_desc.endswith('.'):
        # Short fragments without proper ending are likely truncated/useless
        return False
    
    return True

def generate_ai_final_summaries(articles_by_ticker: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, Dict[str, str]]:
    """Generate AI summaries with enhanced financial context, industry analysis, and materiality focus"""
    if not OPENAI_API_KEY:
        return {}
    
    summaries = {}
    
    for ticker, categories in articles_by_ticker.items():
        company_articles = categories.get("company", [])
        competitor_articles = categories.get("competitor", [])
        industry_articles = categories.get("industry", [])
        
        if not company_articles and not industry_articles:
            continue
        
        config = get_ticker_config(ticker)
        company_name = config.get("name", ticker) if config else ticker
        
        # Get enhanced sector information
        sector = config.get("sector", "") if config else ""
        industry = config.get("industry", "") if config else ""
        financial_context = f"{company_name} operates in {sector}" if sector else f"{company_name}"
        if industry:
            financial_context += f" within the {industry} industry"
        
        # Get industry keywords for enhanced context
        industry_keywords = config.get("industry_keywords", []) if config else []
        
        # Company articles with content and ticker-specific analysis
        articles_with_content = [
            article for article in company_articles 
            if (article.get("scraped_content") and 
                article.get("ai_analysis_ticker") == ticker)
        ]
        
        # Industry articles with content and ticker-specific analysis
        industry_articles_with_content = [
            article for article in industry_articles 
            if (article.get("scraped_content") and 
                article.get("ai_analysis_ticker") == ticker)
        ]
        
        competitor_articles_with_content = [
            article for article in competitor_articles 
            if (article.get("scraped_content") and 
                article.get("ai_analysis_ticker") == ticker)
        ]
        
        LOG.info(f"Found {len(articles_with_content)} company articles with {ticker}-perspective analysis")
        LOG.info(f"Found {len(industry_articles_with_content)} industry articles with {ticker}-perspective analysis")
        LOG.info(f"Found {len(competitor_articles_with_content)} competitor articles with {ticker}-perspective analysis")
        
        ai_analysis_summary = ""
        
        if articles_with_content or industry_articles_with_content:
            # Company content summaries - USE AI SUMMARIES
            content_summaries = []
            for article in articles_with_content[:20]:
                title = article.get("title", "")
                ai_summary = article.get("ai_summary", "")  # CHANGED: Use AI summary instead of scraped content
                domain = article.get("domain", "")
                
                if ai_summary and len(ai_summary) > 50:  # CHANGED: Lower threshold for AI summaries
                    source_name = get_or_create_formal_domain_name(domain) if domain else "Unknown Source"
                    content_summaries.append(f"• {title} [{source_name}]: {ai_summary}")  # CHANGED: Use full AI summary
            
            # Industry content summaries with keyword context - USE AI SUMMARIES
            industry_content_summaries = []
            for article in industry_articles_with_content[:15]:
                title = article.get("title", "")
                ai_summary = article.get("ai_summary", "")  # CHANGED: Use AI summary
                domain = article.get("domain", "")
                keyword = article.get("search_keyword", "Industry")
                
                if ai_summary and len(ai_summary) > 50:  # CHANGED: Lower threshold
                    source_name = get_or_create_formal_domain_name(domain) if domain else "Unknown Source"
                    industry_content_summaries.append(f"• {title} [Industry: {keyword}] [{source_name}]: {ai_summary}")  # CHANGED: Use full AI summary
            
            # Competitor content summaries - USE AI SUMMARIES
            competitor_content_summaries = []
            for article in competitor_articles_with_content[:15]:
                title = article.get("title", "")
                ai_summary = article.get("ai_summary", "")  # CHANGED: Use AI summary
                domain = article.get("domain", "")
                
                if ai_summary and len(ai_summary) > 50:  # CHANGED: Lower threshold
                    source_name = get_or_create_formal_domain_name(domain) if domain else "Unknown Source"
                    competitor_content_summaries.append(f"• {title} [{source_name}]: {ai_summary}")  # CHANGED: Use full AI summary
            
            if content_summaries or industry_content_summaries:
                ai_text = "\n".join(content_summaries)
                industry_analysis = ""
                if industry_content_summaries:
                    industry_analysis = "\n\nINDUSTRY & SECTOR ANALYSIS:\n" + "\n".join(industry_content_summaries)
                competitor_analysis = ""
                if competitor_content_summaries:
                    competitor_analysis = "\n\nCOMPETITOR ANALYSIS:\n" + "\n".join(competitor_content_summaries)
                
                try:
                    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
                    
                    prompt = f"""You are a hedge fund analyst synthesizing deep content analysis into an investment thesis for {company_name} ({ticker}). Transform individual article analyses into cohesive strategic assessment.

ANALYSIS FRAMEWORK:
1. COMPANY FINANCIAL IMPACT: Developments affecting sales, margins, EBITDA, FCF, or growth, if present. Discuss M&A, debt issuance, buybacks, dividends, analyst actions, if present.
2. INDUSTRY/SECTOR DYNAMICS: Policy, regulatory, supply chain, or market developments affecting the sector and {company_name}'s position, if present.
3. COMPETITIVE DYNAMICS: Competitor actions impacting {company_name}'s market position, if present.
4. OPERATIONAL DEVELOPMENTS: Highlight capacity changes, strategic moves, regulatory impacts, if present.
5. MARKET POSITIONING: Evaluate brand strength, pricing power, customer relationships, if present.

CRITICAL REQUIREMENTS:
- Include SPECIFIC DATES: earnings dates, regulatory deadlines, investor days, conference dates, completion timelines, if present
- Report figures (%/$/units) exactly if present; no estimates/price math unless both numbers are in-text
- Synthesize quantitative metrics when available
- MATERIALITY ASSESSMENT: Compare dollar amounts to company scale where mentioned
- ANALYST ACTIONS: Include firm names and price targets as mentioned in content
- INDUSTRY IMPACT: Assess how sector developments affect {company_name}'s business model and profitability
- NEAR-TERM FOCUS: Emphasize next-term (<1 year) but note medium/long-term implications
- Include specific numbers when available and cite sources using formal domain names exactly as written and nothing else: {source_name}. Cite them in parentheses, e.g., (Business Wire).
- Assess competitor moves that could affect {company_name}'s performance
- Keep to 7-8 sentences maximum

FINANCIAL CONTEXT: {financial_context}
INDUSTRY KEYWORDS: {', '.join(industry_keywords) if industry_keywords else 'N/A'}

TARGET: {company_name} ({ticker})

COMPANY ARTICLE CONTENT ANALYSIS (sources in brackets):
{ai_text}{industry_analysis}{competitor_analysis}

Provide a strategic investment thesis integrating company developments, industry dynamics, and competitive positioning with specific dates, materiality context, and analyst price targets."""

                    data = {
                        "model": OPENAI_MODEL,
                        "input": prompt,
                        "max_output_tokens": 10000,
                        "reasoning": {"effort": "medium"},
                        "text": {"verbosity": "low"},
                        "truncation": "auto"
                    }
                    
                    response = get_openai_session().post(OPENAI_API_URL, headers=headers, json=data, timeout=(10, 180))
                    
                    if response.status_code == 200:
                        result = response.json()
                        ai_analysis_summary = extract_text_from_responses(result)
                        
                        u = result.get("usage", {}) or {}
                        LOG.info("Enhanced AI Analysis usage — input:%s output:%s (cap:%s) status:%s",
                                 u.get("input_tokens"), u.get("output_tokens"),
                                 result.get("max_output_tokens"),
                                 result.get("status"))
                    else:
                        LOG.warning(f"Enhanced AI analysis summary failed: {response.status_code}")
                        
                except Exception as e:
                    LOG.warning(f"Failed to generate enhanced AI analysis summary for {ticker}: {e}")
        
        summaries[ticker] = {
            "ai_analysis_summary": ai_analysis_summary,
            "company_name": company_name,
            "industry_articles_analyzed": len(industry_articles_with_content)
        }
    
    return summaries

def generate_ai_titles_summary(articles_by_ticker: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, Dict[str, str]]:
    """Generate AI summaries from company + industry article titles with enhanced financial focus"""
    if not OPENAI_API_KEY:
        return {}
    
    summaries = {}
    
    for ticker, categories in articles_by_ticker.items():
        company_articles = categories.get("company", [])
        competitor_articles = categories.get("competitor", [])
        industry_articles = categories.get("industry", [])  # Add industry articles
        
        if not company_articles and not industry_articles:
            continue
        
        config = get_ticker_config(ticker)
        company_name = config.get("name", ticker) if config else ticker
        
        # Get enhanced competitor information
        competitor_names = []
        if config and config.get("competitors"):
            for comp in config["competitors"]:
                if isinstance(comp, dict):
                    if comp.get('ticker'):
                        competitor_names.append(f"{comp['name']} ({comp['ticker']})")
                    else:
                        competitor_names.append(comp['name'])
                else:
                    # Handle string format "Name (TICKER)"
                    match = re.search(r'^(.+?)\s*\(([A-Z]{1,5})\)$', comp)
                    if match:
                        competitor_names.append(f"{match.group(1).strip()} ({match.group(2)})")
                    else:
                        competitor_names.append(comp)
        
        # Get enhanced industry keywords
        industry_keywords = config.get("industry_keywords", []) if config else []
        
        # Get enhanced sector information  
        sector_info = ""
        if config:
            sector = config.get("sector", "")
            industry = config.get("industry", "") 
            if sector and industry:
                sector_info = f"Sector: {sector}, Industry: {industry}"
            elif sector:
                sector_info = f"Sector: {sector}"
            elif industry:
                sector_info = f"Industry: {industry}"
        
        # Company titles
        titles_with_sources = []
        for article in company_articles[:20]:
            title = article.get("title", "")
            if title:
                domain = article.get("domain", "")
                source_name = get_or_create_formal_domain_name(domain) if domain else "Unknown Source"
                titles_with_sources.append(f"• {title} [{source_name}]")
        
        # Industry titles with keyword context
        industry_titles_with_sources = []
        for article in industry_articles[:10]:  # Add industry articles
            title = article.get("title", "")
            if title:
                domain = article.get("domain", "")
                keyword = article.get("search_keyword", "Industry")
                source_name = get_or_create_formal_domain_name(domain) if domain else "Unknown Source"
                industry_titles_with_sources.append(f"• {title} [Industry: {keyword}] [{source_name}]")
        
        # Competitor titles
        competitor_titles_with_sources = []
        for article in competitor_articles[:10]:
            title = article.get("title", "")
            if title:
                domain = article.get("domain", "")
                source_name = get_or_create_formal_domain_name(domain) if domain else "Unknown Source"
                competitor_titles_with_sources.append(f"• {title} [{source_name}]")
        
        titles_summary = ""
        
        if titles_with_sources or industry_titles_with_sources:
            titles_text = "\n".join(titles_with_sources)
            industry_text = ""
            if industry_titles_with_sources:
                industry_text = "\n\nINDUSTRY DEVELOPMENTS:\n" + "\n".join(industry_titles_with_sources)
            competitor_text = ""
            if competitor_titles_with_sources:
                competitor_text = "\n\nCOMPETITOR NEWS:\n" + "\n".join(competitor_titles_with_sources)
            
            try:
                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
                
                prompt = f"""You are a hedge fund analyst creating a daily executive summary for {company_name} ({ticker}). Analyze recent company and industry news headlines to assess near-term financial impact.

ANALYSIS FRAMEWORK:
1. COMPANY FINANCIAL IMPACT: Developments affecting sales, margins, EBITDA, FCF, or growth, if present. Discuss M&A, debt issuance, buybacks, dividends, analyst actions, if present.
2. INDUSTRY/SECTOR DYNAMICS: Policy, regulatory, supply chain, or market developments affecting the sector and {company_name}'s position, if present.
3. COMPETITIVE DYNAMICS: Competitor actions impacting {company_name}'s market position, if present.
4. OPERATIONAL DEVELOPMENTS: Highlight capacity changes, strategic moves, regulatory impacts, if present.
5. MARKET POSITIONING: Evaluate brand strength, pricing power, customer relationships, if present.

CRITICAL REQUIREMENTS:
- Include SPECIFIC DATES: earnings dates, regulatory deadlines, completion timelines, if present
- Report figures (%/$/units) exactly if present; no estimates/price math unless both numbers are in-text
- Synthesize quantitative metrics when available
- MATERIALITY ASSESSMENT: Compare dollar amounts to company scale where mentioned
- ANALYST ACTIONS: Include firm names and price targets as mentioned in headlines
- INDUSTRY IMPACT: Assess how sector developments affect {company_name}'s business model and profitability
- NEAR-TERM FOCUS: Emphasize next-term (<1 year) but note medium/long-term implications
- Include specific numbers when available and cite sources using formal domain names and nothing else: {source_name}. Cite them in parentheses, e.g., (Business Wire).
- Keep to 5-6 sentences maximum

TARGET COMPANY: {company_name} ({ticker})
INDUSTRY KEYWORDS: {', '.join(industry_keywords) if industry_keywords else 'None specified'}
KNOWN COMPETITORS: {', '.join(competitor_names) if competitor_names else 'None specified'}

COMPANY HEADLINES (sources provided in brackets):
{titles_text}{industry_text}{competitor_text}

Provide a comprehensive executive summary integrating company-specific news with relevant industry and competitive developments."""

                data = {
                    "model": OPENAI_MODEL,
                    "input": prompt,
                    "max_output_tokens": 10000,
                    "reasoning": {"effort": "medium"},
                    "text": {"verbosity": "low"},
                    "truncation": "auto"
                }
                
                response = get_openai_session().post(OPENAI_API_URL, headers=headers, json=data, timeout=(10, 180))
                
                if response.status_code == 200:
                    result = response.json()
                    titles_summary = extract_text_from_responses(result)
                    
                    u = result.get("usage", {}) or {}
                    LOG.info("Enhanced titles summary usage — input:%s output:%s (cap:%s) status:%s",
                             u.get("input_tokens"), u.get("output_tokens"),
                             result.get("max_output_tokens"),
                             result.get("status"))
                else:
                    LOG.warning(f"Enhanced titles summary failed: {response.status_code}")
                         
            except Exception as e:
                LOG.warning(f"Failed to generate enhanced titles summary for {ticker}: {e}")
        
        summaries[ticker] = {
            "titles_summary": titles_summary,
            "company_name": company_name,
            "industry_coverage": len(industry_titles_with_sources)
        }
    
    return summaries

def send_enhanced_quick_intelligence_email(articles_by_ticker: Dict[str, Dict[str, List[Dict]]], triage_results: Dict[str, Dict[str, List[Dict]]]) -> bool:
    """Quick email with metadata display removed"""
    try:
        current_time_est = format_timestamp_est(datetime.now(timezone.utc))
        
        ticker_list = ', '.join(articles_by_ticker.keys())
        
        company_summaries = generate_ai_titles_summary(articles_by_ticker)
        
        html = [
            "<html><head><style>",
            "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 13px; line-height: 1.6; color: #333; }",
            "h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }",
            "h2 { color: #34495e; margin-top: 25px; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; }",
            "h3 { color: #7f8c8d; margin-top: 15px; margin-bottom: 8px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }",
            ".article { margin: 8px 0; padding: 8px; border-left: 3px solid transparent; transition: all 0.3s; background-color: #fafafa; border-radius: 4px; }",
            ".company { border-left-color: #27ae60; }",
            ".industry { border-left-color: #f39c12; }",
            ".competitor { border-left-color: #e74c3c; }",
            ".company-summary { background-color: #f0f8ff; padding: 15px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #3498db; }",
            ".summary-title { font-weight: bold; color: #2c3e50; margin-bottom: 10px; font-size: 14px; }",
            ".summary-content { color: #34495e; line-height: 1.5; margin-bottom: 10px; }",
            ".company-name-badge { display: inline-block; padding: 2px 8px; margin-right: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e8f5e8; color: #2e7d32; border: 1px solid #a5d6a7; }",
            ".source-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e9ecef; color: #495057; }",
            ".quality-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e1f5fe; color: #0277bd; border: 1px solid #81d4fa; }",
            ".flagged-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }",
            ".ai-triage { display: inline-block; padding: 2px 8px; border-radius: 3px; font-weight: bold; font-size: 10px; margin-left: 5px; }",
            ".ai-high { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }",
            ".ai-medium { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }",
            ".ai-low { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }",
            ".qb-score { display: inline-block; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 10px; margin-left: 5px; }",
            ".qb-high { background-color: #c8e6c9; color: #2e7d32; border: 1px solid #a5d6a7; }",
            ".qb-medium { background-color: #fff3e0; color: #f57c00; border: 1px solid #ffcc02; }",
            ".qb-low { background-color: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }",
            ".competitor-badge { display: inline-block; padding: 2px 8px; margin-right: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fdeaea; color: #c53030; border: 1px solid #feb2b2; }",
            ".industry-badge { display: inline-block; padding: 2px 8px; margin-right: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fef5e7; color: #b7791f; border: 1px solid #f6e05e; }",
            ".summary { margin-top: 20px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }",
            ".ticker-section { margin-bottom: 40px; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            ".meta { color: #95a5a6; font-size: 11px; }",
            "a { color: #2980b9; text-decoration: none; }",
            "a:hover { text-decoration: underline; }",
            "</style></head><body>",
            f"<h1>🚀 Quick Intelligence Report: {ticker_list} - Triage Complete</h1>",
            f"<div class='summary'>",
            f"<strong>Generated:</strong> {current_time_est}<br>",
            f"<strong>🎯 Status:</strong> Articles ingested and triaged, AI analysis and scraping in progress...<br>",
            f"<strong>📊 Tickers Covered:</strong> {ticker_list}<br>",
            f"<strong>🤖 Selection Process:</strong> AI Triage → Quality Domains → Exclude Problematic → QB Score Backfill",
            "</div>"
        ]
        
        total_articles = 0
        total_selected = 0
        
        for ticker, categories in articles_by_ticker.items():
            ticker_count = sum(len(articles) for articles in categories.values())
            total_articles += ticker_count
            
            # Get company name from config for display
            config = get_ticker_config(ticker)
            full_company_name = config.get("name", ticker) if config else ticker
            
            ticker_selected = 0
            triage_data = triage_results.get(ticker, {})
            for category in ["company", "industry", "competitor"]:
                ticker_selected += len(triage_data.get(category, []))
            
            total_selected += ticker_selected
            
            html.append(f"<div class='ticker-section'>")
            html.append(f"<h2>📈 {ticker} ({full_company_name}) - {ticker_count} Total Articles</h2>")
            
            if ticker in company_summaries and company_summaries[ticker].get("titles_summary"):
                html.append("<div class='company-summary'>")
                html.append("<div class='summary-title'>📰 Executive Summary (Headlines Analysis)</div>")
                html.append(f"<div class='summary-content'>{company_summaries[ticker]['titles_summary']}</div>")
                html.append("</div>")
                html.append(f"<p><strong>✅ Selected for Analysis:</strong> {ticker_selected} articles</p>")
            
            # Process articles with quality domains first
            for category, articles in categories.items():
                if not articles:
                    continue
                
                category_triage = triage_data.get(category, [])
                selected_article_data = {item["id"]: item for item in category_triage}
                
                # Enhanced article sorting - Quality domains first
                enhanced_articles = []
                for idx, article in enumerate(articles):
                    domain = normalize_domain(article.get("domain", ""))
                    is_ai_selected = idx in selected_article_data
                    is_quality_domain = domain in QUALITY_DOMAINS
                    is_problematic = domain in PROBLEMATIC_SCRAPE_DOMAINS
                    
                    priority = 999
                    triage_reason = ""
                    ai_priority = "Low"
                    
                    if is_quality_domain and not is_problematic:
                        priority = 1  # Quality domains get top priority
                        triage_reason = "Quality domain auto-selected"
                    elif is_ai_selected:
                        ai_priority = normalize_priority_to_display(selected_article_data[idx].get("scrape_priority", "Low"))
                        triage_reason = selected_article_data[idx].get("why", "")
                        priority_map = {"High": 2, "Medium": 3, "Low": 4}
                        priority = priority_map.get(ai_priority, 4)
                    
                    enhanced_articles.append({
                        "article": article,
                        "idx": idx,
                        "priority": priority,
                        "is_ai_selected": is_ai_selected,
                        "is_quality_domain": is_quality_domain,
                        "is_problematic": is_problematic,
                        "triage_reason": triage_reason,
                        "ai_priority": ai_priority,
                        "published_at": article.get("published_at")
                    })
                
                # Sort by priority (quality first) and time
                enhanced_articles.sort(key=lambda x: (
                    x["priority"],
                    -(x["published_at"].timestamp() if x["published_at"] else 0)
                ))
                
                selected_count = len([a for a in enhanced_articles if a["is_ai_selected"] or (a["is_quality_domain"] and not a["is_problematic"])])
                
                category_icons = {
                    "company": "🎯",
                    "industry": "🏭", 
                    "competitor": "⚔️"
                }
                
                html.append(f"<h3>{category_icons.get(category, '📰')} {category.title()} ({len(articles)} articles, {selected_count} selected)</h3>")
                
                for enhanced_article in enhanced_articles[:50]:
                    article = enhanced_article["article"]
                    domain = article.get("domain", "unknown")
                    title = article.get("title", "No Title")
                    
                    header_badges = []
                    
                    # 1. First badge depends on category
                    if category == "company":
                        header_badges.append(f'<span class="company-name-badge">🎯 {full_company_name}</span>')
                    elif category == "competitor":
                        comp_name = get_competitor_display_name(article.get('search_keyword'), article.get('competitor_ticker'))
                        header_badges.append(f'<span class="competitor-badge">🏢 {comp_name}</span>')
                    elif category == "industry" and article.get('search_keyword'):
                        header_badges.append(f'<span class="industry-badge">🏭 {article["search_keyword"]}</span>')
                    
                    # 2. Domain name second
                    header_badges.append(f'<span class="source-badge">📰 {get_or_create_formal_domain_name(domain)}</span>')
                    
                    # 3. Quality badge third
                    if enhanced_article["is_quality_domain"]:
                        header_badges.append('<span class="quality-badge">⭐ Quality</span>')
                    
                    # 4. Flagged badge if selected
                    if enhanced_article["is_ai_selected"] or (enhanced_article["is_quality_domain"] and not enhanced_article["is_problematic"]):
                        header_badges.append('<span class="flagged-badge">🚩 Flagged</span>')
                    
                    # 5. AI Triage - normalized format
                    if enhanced_article["is_ai_selected"]:
                        ai_priority = enhanced_article["ai_priority"]
                        badge_class = f"ai-{ai_priority.lower()}"
                        ai_emoji = {"High": "🔥", "Medium": "⚡", "Low": "🔋"}.get(ai_priority, "🔋")
                        header_badges.append(f'<span class="ai-triage {badge_class}">{ai_emoji} AI: {ai_priority}</span>')
                    
                    # 6. QB Score last
                    qb_score = article.get('qb_score', 0)
                    if qb_score >= 70:
                        qb_class = "qb-high"
                        qb_level = "QB: High"
                        qb_emoji = "🏆"
                    elif qb_score >= 40:
                        qb_class = "qb-medium"
                        qb_level = "QB: Medium"
                        qb_emoji = "🥉"
                    else:
                        qb_class = "qb-low"
                        qb_level = "QB: Low"
                        qb_emoji = "📊"
                    header_badges.append(f'<span class="qb-score {qb_class}">{qb_emoji} {qb_level}</span>')
                    
                    # Publication time
                    pub_time = ""
                    if article.get("published_at"):
                        pub_time = format_timestamp_est(article["published_at"])
                    
                    html.append(f"""
                    <div class='article {category}'>
                        <div class='article-header'>
                            {' '.join(header_badges)}
                        </div>
                        <div class='article-content'>
                            <a href='{article.get("resolved_url") or article.get("url", "")}'>{title}</a>
                            <span class='meta'> | {pub_time}</span>
                        </div>
                    </div>
                    """)
            
            html.append("</div>")
        
        html.append(f"<div class='summary'>")
        html.append(f"<strong>📊 Total Articles:</strong> {total_articles}<br>")
        html.append(f"<strong>✅ Selected for Analysis:</strong> {total_selected}<br>")
        html.append(f"<strong>📄 Next:</strong> Full content analysis and hedge fund summaries in progress...")
        html.append("</div>")
        html.append("</body></html>")
        
        html_content = "".join(html)
        subject = f"🚀 Quick Intelligence: {ticker_list} - {total_selected} articles selected"
        
        return send_email(subject, html_content)
        
    except Exception as e:
        LOG.error(f"Enhanced quick intelligence email failed: {e}")
        return False


# ------------------------------------------------------------------------------
# Email Digest
# ------------------------------------------------------------------------------

# Updated email sending function with text attachment
def send_email(subject: str, html_body: str, to: str | None = None) -> bool:
    """Send email with HTML body only (no attachments)."""
    if not all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, EMAIL_FROM]):
        LOG.error("SMTP not fully configured")
        return False

    try:
        recipient = to or DIGEST_TO

        # multipart/alternative for text + HTML, wrapped in mixed not needed if no attachments
        msg = MIMEMultipart('alternative')
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = recipient

        # Plain-text fallback
        text_body = "This email contains HTML content. Please view in an HTML-capable email client."
        msg.attach(MIMEText(text_body, "plain", "utf-8"))

        # HTML body
        msg.attach(MIMEText(html_body, "html", "utf-8"))

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

def build_enhanced_digest_html(articles_by_ticker: Dict[str, Dict[str, List[Dict]]], period_days: int) -> Tuple[str, str]:
    """Enhanced digest with metadata display removed but keeping all badges/emojis"""
    
    company_summaries = generate_ai_final_summaries(articles_by_ticker)
    
    ticker_list = ', '.join(articles_by_ticker.keys())
    current_time_est = format_timestamp_est(datetime.now(timezone.utc))
    
    html = [
        "<html><head><style>",
        "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 13px; line-height: 1.6; color: #333; }",
        "h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }",
        "h2 { color: #34495e; margin-top: 25px; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; }",
        "h3 { color: #7f8c8d; margin-top: 15px; margin-bottom: 8px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }",
        ".article { margin: 8px 0; padding: 8px; border-left: 3px solid transparent; transition: all 0.3s; background-color: #fafafa; border-radius: 4px; }",
        ".article:hover { background-color: #f0f8ff; border-left-color: #3498db; }",
        ".article-header { margin-bottom: 5px; }",
        ".article-content { }",
        ".company { border-left-color: #27ae60; }",
        ".industry { border-left-color: #f39c12; }",
        ".competitor { border-left-color: #e74c3c; }",
        ".company-summary { background-color: #f0f8ff; padding: 15px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #3498db; }",
        ".summary-title { font-weight: bold; color: #2c3e50; margin-bottom: 10px; font-size: 14px; }",
        ".summary-content { color: #34495e; line-height: 1.5; margin-bottom: 10px; }",
        ".company-name-badge { display: inline-block; padding: 2px 8px; margin-right: 8px; border-radius: 5px; font-weight: bold; font-size: 10px; background-color: #e8f5e8; color: #2e7d32; border: 1px solid #a5d6a7; }",
        ".source-badge { display: inline-block; padding: 2px 8px; margin-left: 0px; margin-right: 8px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e9ecef; color: #495057; }",
        ".quality-badge { display: inline-block; padding: 2px 6px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e1f5fe; color: #0277bd; border: 1px solid #81d4fa; }",
        ".analyzed-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e3f2fd; color: #1565c0; border: 1px solid #90caf9; }",
        ".competitor-badge { display: inline-block; padding: 2px 8px; margin-right: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fdeaea; color: #c53030; border: 1px solid #feb2b2; }",
        ".industry-badge { display: inline-block; padding: 2px 8px; margin-right: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fef5e7; color: #b7791f; border: 1px solid #f6e05e; }",
        ".description { color: #6c757d; font-size: 11px; font-style: italic; margin-top: 5px; line-height: 1.4; display: block; }",
        ".ai-summary { color: #2c5aa0; font-size: 12px; margin-top: 8px; line-height: 1.4; background-color: #f8f9ff; padding: 8px; border-radius: 4px; border-left: 3px solid #3498db; }",
        ".meta { color: #95a5a6; font-size: 11px; }",
        ".ticker-section { margin-bottom: 40px; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        ".summary { margin-top: 20px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }",
        "a { color: #2980b9; text-decoration: none; }",
        "a:hover { text-decoration: underline; }",
        "</style></head><body>",
        f"<h1>📊 Stock Intelligence Report: {ticker_list}</h1>",
        f"<div class='summary'>",
        f"<strong>📅 Report Period:</strong> Last {period_days} days<br>",
        f"<strong>Generated:</strong> {current_time_est}<br>",
        f"<strong>🎯 Tickers Covered:</strong> {ticker_list}<br>",
        f"<strong>🤖 AI Features:</strong> Enhanced Content Analysis + Hedge Fund Summaries + Company Intelligence Synthesis",
        "</div>"
    ]
        
    for ticker, categories in articles_by_ticker.items():
        total_articles = sum(len(articles) for articles in categories.values())
        
        # Get company name for display
        config = get_ticker_config(ticker)
        company_name = config.get("name", ticker) if config else ticker
        
        html.append(f"<div class='ticker-section'>")
        html.append(f"<h2>📈 {ticker} ({company_name}) - {total_articles} Total Articles</h2>")
        
        # Add AI-generated summaries from scraped content
        if ticker in company_summaries and company_summaries[ticker].get("ai_analysis_summary"):
            html.append("<div class='company-summary'>")
            html.append("<div class='summary-title'>🎯 Investment Thesis (Deep Analysis Synthesis)</div>")
            html.append(f"<div class='summary-content'>{company_summaries[ticker]['ai_analysis_summary']}</div>")
            html.append("</div>")
        
        # Sort articles within each category - Quality domains first, then AI analyzed, then by time
        for category in ["company", "industry", "competitor"]:
            if category in categories and categories[category]:
                articles = categories[category]
                
                def sort_key(article):
                    domain = normalize_domain(article.get("domain", ""))
                    is_quality_domain = domain in QUALITY_DOMAINS
                    is_analyzed = bool(article.get('ai_summary') or article.get('ai_triage_selected'))
                    pub_time = article.get("published_at") or datetime.min.replace(tzinfo=timezone.utc)
                    
                    # Priority: 0 = quality domain, 1 = analyzed, 2 = other
                    if is_quality_domain:
                        priority = 0
                    elif is_analyzed:
                        priority = 1
                    else:
                        priority = 2
                    
                    return (priority, -pub_time.timestamp())
                
                sorted_articles = sorted(articles, key=sort_key)
                
                category_icons = {
                    "company": "🎯",
                    "industry": "🏭", 
                    "competitor": "⚔️"
                }
                
                html.append(f"<h3>{category_icons.get(category, '📰')} {category.title()} News ({len(articles)} articles)</h3>")
                for article in sorted_articles[:100]:
                    # Use simplified ticker metadata cache (just company name)
                    simple_cache = {ticker: {"company_name": company_name}}
                    html.append(_format_article_html_with_ai_summary(article, category, simple_cache))
        
        html.append("</div>")
    
    html.append("""
        <div style='margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; font-size: 11px; color: #6c757d;'>
            <strong>🤖 Enhanced AI Features:</strong><br>
            📊 Investment Thesis: AI synthesis of all company deep analysis<br>
            📰 Content Analysis: Full article scraping with intelligent extraction<br>
            💼 Hedge Fund Summaries: AI-generated analytical summaries for scraped content<br>
            🎯 Enhanced Selection: AI Triage → Quality Domains → Exclude Problematic → QB Score Backfill<br>
            ✅ "Analyzed" badge indicates articles with both scraped content and AI summary<br>
            ⭐ "Quality" badge indicates high-authority news sources
        </div>
        </body></html>
    """)
    
    html_content = "".join(html)
    return html_content

def fetch_digest_articles_with_enhanced_content(hours: int = 24, tickers: List[str] = None) -> Dict[str, Dict[str, List[Dict]]]:
    """Fetch categorized articles for digest with ticker-specific AI analysis"""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    days = int(hours / 24) if hours >= 24 else 0
    period_label = f"{days} days" if days > 0 else f"{hours:.0f} hours"
    
    with db() as conn, conn.cursor() as cur:
        # Enhanced query to get articles analyzed from each ticker's perspective - avoid duplicates
        if tickers:
            cur.execute("""
                SELECT DISTINCT ON (f.url_hash, f.ticker)
                    f.id, f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.published_at,
                    f.found_at, f.category, f.original_source_url,
                    f.search_keyword, f.ai_summary,
                    f.scraped_content, f.content_scraped_at, f.scraping_failed, f.scraping_error,
                    f.competitor_ticker, f.ai_triage_selected, f.triage_priority, f.triage_reasoning,
                    f.qb_score, f.qb_level, f.qb_reasoning, f.ai_analysis_ticker
                FROM found_url f
                WHERE f.found_at >= %s
                    AND (f.ticker = ANY(%s) OR f.ai_analysis_ticker = ANY(%s))
                ORDER BY f.url_hash, f.ticker, 
                    CASE WHEN f.ai_analysis_ticker IS NOT NULL THEN 0 ELSE 1 END,
                    COALESCE(f.published_at, f.found_at) DESC, f.found_at DESC
            """, (cutoff, tickers, tickers))
        else:
            cur.execute("""
                SELECT DISTINCT ON (f.url_hash, f.ticker)
                    f.id, f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.published_at,
                    f.found_at, f.category, f.original_source_url,
                    f.search_keyword, f.ai_summary,
                    f.scraped_content, f.content_scraped_at, f.scraping_failed, f.scraping_error,
                    f.competitor_ticker, f.ai_triage_selected, f.triage_priority, f.triage_reasoning,
                    f.qb_score, f.qb_level, f.qb_reasoning, f.ai_analysis_ticker
                FROM found_url f
                WHERE f.found_at >= %s
                ORDER BY f.url_hash, f.ticker, 
                    CASE WHEN f.ai_analysis_ticker IS NOT NULL THEN 0 ELSE 1 END,
                    COALESCE(f.published_at, f.found_at) DESC, f.found_at DESC
            """, (cutoff,))
        
        # Group articles by target ticker (the ticker we're analyzing for)
        articles_by_ticker = {}
        for row in cur.fetchall():
            # Use ai_analysis_ticker if available, otherwise use ticker
            target_ticker = row["ai_analysis_ticker"] or row["ticker"] or "UNKNOWN"
            category = row["category"] or "company"
            
            if target_ticker not in articles_by_ticker:
                articles_by_ticker[target_ticker] = {}
            if category not in articles_by_ticker[target_ticker]:
                articles_by_ticker[target_ticker][category] = []
            
            # Convert row to dict and add to results
            article_dict = dict(row)
            articles_by_ticker[target_ticker][category].append(article_dict)
            
            # Debug logging for AI summary presence
            if article_dict.get('ai_summary'):
                LOG.debug(f"DIGEST QUERY: Found AI summary for {target_ticker} - {len(article_dict['ai_summary'])} chars")
            else:
                LOG.debug(f"DIGEST QUERY: No AI summary for {target_ticker} article: {article_dict.get('title', 'No title')[:50]}")
        
        # Mark articles as sent (only new ones)
        total_to_mark = 0
        if tickers:
            cur.execute("""
                SELECT COUNT(DISTINCT (f.url_hash, f.ticker)) as count
                FROM found_url f
                WHERE f.found_at >= %s 
                AND (f.ticker = ANY(%s) OR f.ai_analysis_ticker = ANY(%s))
                AND NOT f.sent_in_digest
            """, (cutoff, tickers, tickers))
        else:
            cur.execute("""
                SELECT COUNT(DISTINCT (f.url_hash, f.ticker)) as count
                FROM found_url f
                WHERE f.found_at >= %s 
                AND NOT f.sent_in_digest
            """, (cutoff,))
        
        result = cur.fetchone()
        total_to_mark = result["count"] if result else 0
        
        if total_to_mark > 0:
            if tickers:
                cur.execute("""
                    UPDATE found_url
                    SET sent_in_digest = TRUE
                    WHERE found_at >= %s 
                    AND (ticker = ANY(%s) OR ai_analysis_ticker = ANY(%s))
                    AND NOT sent_in_digest
                """, (cutoff, tickers, tickers))
            else:
                cur.execute("""
                    UPDATE found_url
                    SET sent_in_digest = TRUE
                    WHERE found_at >= %s
                    AND NOT sent_in_digest
                """, (cutoff,))
            
            LOG.info(f"Marked {total_to_mark} articles as sent in digest")
        else:
            LOG.info("No new articles to mark as sent (smart reuse mode)")
    
    total_articles = sum(
        sum(len(arts) for arts in categories.values())
        for categories in articles_by_ticker.values()
    )
    
    if total_articles == 0:
        return {
            "status": "no_articles",
            "message": f"No quality articles found in the last {period_label}",
            "tickers": tickers or "all"
        }
    
    # Use the enhanced digest function
    html = build_enhanced_digest_html(articles_by_ticker, days if days > 0 else 1)
    
    # Enhanced subject with ticker list
    ticker_list = ', '.join(articles_by_ticker.keys())
    subject = f"📊 Stock Intelligence: {ticker_list} - {total_articles} articles"
    success = send_email(subject, html)
    
    # Count by category and content scraping
    category_counts = {"company": 0, "industry": 0, "competitor": 0}
    content_stats = {"scraped": 0, "failed": 0, "skipped": 0, "ai_summaries": 0}
    
    for ticker_cats in articles_by_ticker.values():
        for cat, arts in ticker_cats.items():
            category_counts[cat] = category_counts.get(cat, 0) + len(arts)
            for art in arts:
                if art.get('scraped_content'):
                    content_stats['scraped'] += 1
                elif art.get('scraping_failed'):
                    content_stats['failed'] += 1
                else:
                    content_stats['skipped'] += 1
                
                if art.get('ai_summary'):
                    content_stats['ai_summaries'] += 1
    
    LOG.info(f"DIGEST STATS: {content_stats['ai_summaries']} articles with AI summaries out of {total_articles} total")
    
    return {
        "status": "sent" if success else "failed",
        "articles": total_articles,
        "tickers": list(articles_by_ticker.keys()),
        "by_category": category_counts,
        "content_scraping_stats": content_stats,
        "recipient": DIGEST_TO
    }

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
    """Initialize database and generate AI-powered feeds for specified tickers - ENHANCED with limit checking"""
    require_admin(request)
    ensure_schema()
    
    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OpenAI API key not configured"}
    
    results = []
    LOG.info("=== INITIALIZATION STARTING ===")
    
    for ticker in body.tickers:
        LOG.info(f"=== INITIALIZING TICKER: {ticker} ===")
        
        # Get or generate metadata with AI
        keywords = get_or_create_ticker_metadata(ticker, force_refresh=body.force_refresh)
        
        # Build feed URLs for all categories - will check existing counts and only create what's needed
        feeds = feed_manager.create_feeds_for_ticker(ticker, keywords)
        
        if not feeds:
            LOG.info(f"=== {ticker}: No new feeds needed - already at limits ===")
            results.append({
                "ticker": ticker,
                "message": "No new feeds created - already at limits",
                "feeds_created": 0
            })
            continue
        
        # Create the feeds that are needed
        ticker_feed_count = 0
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
            ticker_feed_count += 1
        
        LOG.info(f"=== COMPLETED {ticker}: {ticker_feed_count} new feeds created ===")
    
    # Final summary logging with FIXED competitor entity counting
    LOG.info("=== INITIALIZATION COMPLETE ===")
    LOG.info("SUMMARY:")
    
    feeds_by_ticker = {}
    for result in results:
        if "feed" in result:  # Only count actual feed creations, not "no feeds needed" messages
            ticker = result['ticker']
            if ticker not in feeds_by_ticker:
                feeds_by_ticker[ticker] = {"company": 0, "industry": 0, "competitor": 0}
            category = result.get('category', 'company')
            feeds_by_ticker[ticker][category] += 1
    
    for ticker in body.tickers:
        if ticker in feeds_by_ticker:
            ticker_feeds = feeds_by_ticker[ticker]
            total_feeds = sum(ticker_feeds.values())
            # FIXED: Count competitor entities (divide by 2 since each competitor creates 2 feeds)
            competitor_entities = ticker_feeds['competitor'] // 2
            LOG.info(f"  {ticker}: {total_feeds} new feeds created")
            LOG.info(f"    Company: {ticker_feeds['company']}, Industry: {ticker_feeds['industry']}, Competitor: {competitor_entities} entities ({ticker_feeds['competitor']} feeds)")
        else:
            LOG.info(f"  {ticker}: 0 new feeds created (already at limits)")
    
    total_feeds_created = len([r for r in results if "feed" in r])
    
    # FIXED: Calculate competitor entities for return value
    competitor_entities_by_ticker = {}
    for ticker, feeds in feeds_by_ticker.items():
        competitor_entities_by_ticker[ticker] = {
            "company": feeds["company"],
            "industry": feeds["industry"], 
            "competitor_entities": feeds["competitor"] // 2,
            "competitor_feeds": feeds["competitor"]
        }
    
    return {
        "status": "initialized",
        "tickers": body.tickers,
        "feeds": [r for r in results if "feed" in r],  # Only return actual feed creations
        "summary": {
            "total_feeds_created": total_feeds_created,
            "feeds_by_ticker": competitor_entities_by_ticker,  # Use the fixed version
            "tickers_at_limit": [r["ticker"] for r in results if "message" in r]
        },
        "message": f"Generated {total_feeds_created} new feeds using AI-powered keyword analysis (respecting limits: max 5 industry, max 3 competitors)"
    }
 
@APP.post("/cron/ingest")
def cron_ingest(
    request: Request,
    minutes: int = Query(default=15, description="Time window in minutes"),
    tickers: List[str] = Query(default=None, description="Specific tickers to ingest")
):
    """
    Enhanced ingest with ticker-specific AI analysis perspective
    Each URL analyzed from the perspective of the target ticker
    """
    start_time = time.time()
    require_admin(request)
    ensure_schema()
    update_schema_for_content()
    update_schema_for_triage()
    update_schema_for_qb_scores()
    
    LOG.info("=== CRON INGEST STARTING (TICKER-SPECIFIC AI PERSPECTIVE) ===")
    LOG.info(f"Processing window: {minutes} minutes")
    LOG.info(f"Target tickers: {tickers or 'ALL'}")
    
    # Reset statistics
    reset_ingestion_stats()
    reset_scraping_stats()
    reset_enhanced_scraping_stats()
    
    # Calculate dynamic scraping limits for each ticker
    dynamic_limits = {}
    if tickers:
        for ticker in tickers:
            dynamic_limits[ticker] = calculate_dynamic_scraping_limits(ticker)
    
    # PHASE 1: Normal feed processing for new articles
    LOG.info("=== PHASE 1: PROCESSING FEEDS (NEW + EXISTING ARTICLES) ===")
    
    # Get feeds
    with db() as conn, conn.cursor() as cur:
        if tickers:
            cur.execute("""
                SELECT id, url, name, ticker, category, retain_days, search_keyword, competitor_ticker
                FROM source_feed
                WHERE active = TRUE AND ticker = ANY(%s)
                ORDER BY ticker, category, id
            """, (tickers,))
        else:
            cur.execute("""
                SELECT id, url, name, ticker, category, retain_days, search_keyword, competitor_ticker
                FROM source_feed
                WHERE active = TRUE
                ORDER BY ticker, category, id
            """)
        feeds = list(cur.fetchall())
    
    if not feeds:
        return {"status": "no_feeds", "message": "No active feeds found"}
    
    ingest_stats = {
        "total_processed": 0, 
        "total_inserted": 0, 
        "total_duplicates": 0, 
        "total_spam_blocked": 0, 
        "total_limit_reached": 0,
        "total_insider_trading_blocked": 0,
        "total_yahoo_rejected": 0  # ADD THIS
    }
    
    # Process feeds normally to get new articles
    for feed in feeds:
        try:
            stats = ingest_feed_basic_only(feed)
            ingest_stats["total_processed"] += stats["processed"]
            ingest_stats["total_inserted"] += stats["inserted"]
            ingest_stats["total_duplicates"] += stats["duplicates"]
            ingest_stats["total_spam_blocked"] += stats.get("blocked_spam", 0)
            ingest_stats["total_limit_reached"] += stats.get("limit_reached", 0)
            ingest_stats["total_insider_trading_blocked"] += stats.get("blocked_insider_trading", 0)
            ingest_stats["total_yahoo_rejected"] += stats.get("yahoo_rejected", 0)
        except Exception as e:
            LOG.error(f"Feed ingest failed for {feed['name']}: {e}")
            continue
    
    # Now get ALL articles from the timeframe for ticker-specific analysis
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    articles_by_ticker = {}
    
    with db() as conn, conn.cursor() as cur:
        if tickers:
            cur.execute("""
                SELECT id, url, resolved_url, title, domain, published_at, category, 
                       search_keyword, competitor_ticker, ticker, ai_triage_selected,
                       triage_priority, triage_reasoning, quality_score, ai_impact,
                       scraped_content, ai_summary, ai_analysis_ticker, url_hash
                FROM found_url 
                WHERE found_at >= %s AND ticker = ANY(%s)
                ORDER BY ticker, category, found_at DESC
            """, (cutoff, tickers))
        else:
            cur.execute("""
                SELECT id, url, resolved_url, title, domain, published_at, category, 
                       search_keyword, competitor_ticker, ticker, ai_triage_selected,
                       triage_priority, triage_reasoning, quality_score, ai_impact,
                       scraped_content, ai_summary, ai_analysis_ticker, url_hash
                FROM found_url 
                WHERE found_at >= %s
                ORDER BY ticker, category, found_at DESC
            """, (cutoff,))
        
        all_articles = list(cur.fetchall())
    
    # Organize articles by ticker and category
    for article in all_articles:
        ticker = article["ticker"]
        category = article["category"] or "company"
        
        if ticker not in articles_by_ticker:
            articles_by_ticker[ticker] = {"company": [], "industry": [], "competitor": []}
        
        articles_by_ticker[ticker][category].append(article)
    
    total_articles = len(all_articles)
    LOG.info(f"=== PHASE 1 COMPLETE: {ingest_stats['total_inserted']} new + {total_articles} total in timeframe ===")
    
    # PHASE 2: Pure AI triage (no enhanced selection per your request)
    LOG.info("=== PHASE 2: PURE AI TRIAGE (NO ENHANCED SELECTION) ===")
    triage_results = {}
    
    for ticker in articles_by_ticker.keys():
        LOG.info(f"Running pure AI triage for {ticker}")
        
        # USE PURE AI TRIAGE ONLY (as requested)
        selected_results = perform_ai_triage_batch(articles_by_ticker[ticker], ticker)
        triage_results[ticker] = selected_results
        
        # Update database with triage results
        for category, selected_items in selected_results.items():
            articles = articles_by_ticker[ticker][category]
            for item in selected_items:
                article_idx = item["id"]
                if article_idx < len(articles):
                    article_id = articles[article_idx]["id"]
                    with db() as conn, conn.cursor() as cur:
                        clean_priority = normalize_priority_to_int(item.get("scrape_priority", 2))
                        clean_reasoning = clean_null_bytes(item.get("why", ""))
                        
                        cur.execute("""
                            UPDATE found_url 
                            SET ai_triage_selected = TRUE, triage_priority = %s, triage_reasoning = %s
                            WHERE id = %s
                        """, (clean_priority, clean_reasoning, article_id))
    
    # PHASE 3: Send enhanced quick email
    LOG.info("=== PHASE 3: SENDING ENHANCED QUICK TRIAGE EMAIL ===")
    quick_email_sent = send_enhanced_quick_intelligence_email(articles_by_ticker, triage_results)
    LOG.info(f"Enhanced quick triage email sent: {quick_email_sent}")
    
    # PHASE 4: Ticker-specific content scraping and analysis (WITH HEARTBEATS)
    LOG.info("=== PHASE 4: TICKER-SPECIFIC CONTENT SCRAPING AND ANALYSIS ===")
    scraping_final_stats = {"scraped": 0, "failed": 0, "ai_analyzed": 0, "reused_existing": 0}
    
    # Count total articles to be processed for heartbeat tracking
    total_articles_to_process = 0
    for target_ticker in articles_by_ticker.keys():
        selected = triage_results.get(target_ticker, {})
        for category in ["company", "industry", "competitor"]:
            total_articles_to_process += len(selected.get(category, []))
    
    processed_count = 0
    LOG.info(f"Starting Phase 4: {total_articles_to_process} total articles to process across all tickers")
    
    for target_ticker in articles_by_ticker.keys():
        config = get_ticker_config(target_ticker)
        metadata = {
            "industry_keywords": config.get("industry_keywords", []) if config else [],
            "competitors": config.get("competitors", []) if config else []
        }
        
        selected = triage_results.get(target_ticker, {})
        
        for category in ["company", "industry", "competitor"]:
            category_selected = selected.get(category, [])
            
            for item in category_selected:
                processed_count += 1
                
                # Heartbeat every 5 articles to prevent Render timeout
                if processed_count % 5 == 0:
                    elapsed = time.time() - start_time
                    LOG.info(f"HEARTBEAT: Processing article {processed_count}/{total_articles_to_process} - {category} for {target_ticker} ({elapsed:.1f}s elapsed)")
                    LOG.info(f"HEARTBEAT: Progress {(processed_count/total_articles_to_process)*100:.1f}% - Scraped:{scraping_final_stats['scraped']}, Failed:{scraping_final_stats['failed']}, Reused:{scraping_final_stats['reused_existing']}")
                
                article_idx = item["id"]
                if article_idx < len(articles_by_ticker[target_ticker][category]):
                    article = articles_by_ticker[target_ticker][category][article_idx]
                    
                    # Get appropriate keyword for limit checking
                    if category == "company":
                        keyword = target_ticker
                    elif category == "competitor":
                        keyword = article.get("competitor_ticker") or article.get("search_keyword", "unknown")
                    else:
                        keyword = article.get("search_keyword", "unknown")
                    
                    if not _check_scraping_limit(category, keyword):
                        continue
                    
                    # CRITICAL: Analyze from target_ticker's perspective
                    success = scrape_and_analyze_article_3tier(article, category, metadata, target_ticker)
                    if success:
                        # Check if we reused existing data
                        if (article.get('scraped_content') and 
                            article.get('ai_summary') and 
                            article.get('ai_impact') and
                            article.get('ai_analysis_ticker') == target_ticker):
                            scraping_final_stats["reused_existing"] += 1
                        else:
                            scraping_final_stats["scraped"] += 1
                            scraping_final_stats["ai_analyzed"] += 1
                    else:
                        scraping_final_stats["failed"] += 1
                
                # Additional heartbeat for very large batches (every 10 articles)
                if processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    LOG.info(f"HEARTBEAT: {processed_count}/{total_articles_to_process} complete after {elapsed:.1f}s - keeping connection alive")
    
    # Final heartbeat before completion
    elapsed = time.time() - start_time
    LOG.info(f"PHASE 4 COMPLETE: All {total_articles_to_process} articles processed in {elapsed:.1f}s")
    LOG.info(f"=== PHASE 4 COMPLETE: {scraping_final_stats['scraped']} new + {scraping_final_stats['reused_existing']} reused ===")
    
    log_scraping_success_rates()
    log_enhanced_scraping_stats()
    log_scrapingbee_stats()
    log_scrapfly_stats()
    
    processing_time = time.time() - start_time
    LOG.info(f"=== CRON INGEST COMPLETE (TICKER-SPECIFIC ANALYSIS) - Total time: {processing_time:.1f}s ===")
    
    # Enhanced return with optimization stats
    total_scraping_attempts = enhanced_scraping_stats["total_attempts"]
    total_scraping_success = (enhanced_scraping_stats["requests_success"] + 
                             enhanced_scraping_stats["playwright_success"] + 
                             enhanced_scraping_stats.get("scrapfly_success", 0) +
                             enhanced_scraping_stats["scrapingbee_success"])
    overall_scraping_rate = (total_scraping_success / total_scraping_attempts * 100) if total_scraping_attempts > 0 else 0
    
    return {
        "status": "completed",
        "processing_time_seconds": round(processing_time, 1),
        "workflow": "ticker_specific_ai_analysis",
        "phase_1_ingest": {
            "total_processed": ingest_stats["total_processed"],
            "total_inserted": ingest_stats["total_inserted"],
            "total_duplicates": ingest_stats["total_duplicates"],
            "total_spam_blocked": ingest_stats["total_spam_blocked"],
            "total_limit_reached": ingest_stats["total_limit_reached"],
            "total_insider_trading_blocked": ingest_stats["total_insider_trading_blocked"],
            "total_yahoo_rejected": ingest_stats["total_yahoo_rejected"],  # ADD THIS
            "total_articles_in_timeframe": total_articles
        },
        "phase_2_triage": {
            "type": "pure_ai_triage_only",
            "tickers_processed": len(triage_results),
            "selections_by_ticker": {k: {cat: len(items) for cat, items in v.items()} for k, v in triage_results.items()}
        },
        "phase_3_quick_email": {"sent": quick_email_sent},
        "phase_4_scraping": {
            **scraping_final_stats,
            "overall_success_rate": f"{overall_scraping_rate:.1f}%",
            "tier_breakdown": {
                "requests_success": enhanced_scraping_stats["requests_success"],
                "playwright_success": enhanced_scraping_stats["playwright_success"],
                "scrapingbee_success": enhanced_scraping_stats["scrapingbee_success"],
                "total_attempts": total_scraping_attempts
            },
            "scrapingbee_cost": f"${scrapingbee_stats['cost_estimate']:.3f}",
            "dynamic_limits": dynamic_limits
        },
        "ticker_specific_analysis": f"Each URL analyzed from target ticker's perspective"
    }

def _update_ticker_stats(ticker_stats: Dict, total_stats: Dict, stats: Dict, category: str):
    """Helper to update statistics"""
    ticker_stats["inserted"] += stats["inserted"]
    ticker_stats["duplicates"] += stats["duplicates"]
    ticker_stats["blocked_spam"] += stats.get("blocked_spam", 0)
    ticker_stats["content_scraped"] += stats.get("content_scraped", 0)
    ticker_stats["content_failed"] += stats.get("content_failed", 0)
    ticker_stats["scraping_skipped"] += stats.get("scraping_skipped", 0)
    ticker_stats["ai_scored"] += stats.get("ai_scored", 0)
    ticker_stats["basic_scored"] += stats.get("basic_scored", 0)
    
    total_stats["feeds_processed"] += 1
    total_stats["total_inserted"] += stats["inserted"]
    total_stats["total_duplicates"] += stats["duplicates"]
    total_stats["total_blocked_spam"] += stats.get("blocked_spam", 0)
    total_stats["total_content_scraped"] += stats.get("content_scraped", 0)
    total_stats["total_content_failed"] += stats.get("content_failed", 0)
    total_stats["total_scraping_skipped"] += stats.get("scraping_skipped", 0)
    total_stats["total_ai_scored"] += stats.get("ai_scored", 0)
    total_stats["total_basic_scored"] += stats.get("basic_scored", 0)
    total_stats["by_category"][category] += stats["inserted"]

@APP.post("/cron/digest")
def cron_digest(
    request: Request,
    minutes: int = Query(default=1440, description="Time window in minutes"),
    tickers: List[str] = Query(default=None, description="Specific tickers for digest")
):
    """Generate and send email digest with content scraping data and AI summaries"""
    require_admin(request)
    ensure_schema()
    
    result = fetch_digest_articles_with_enhanced_content(minutes / 60, tickers)
    return result
    
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
    metadata = ticker_manager.get_or_create_metadata(body.ticker, force_refresh=True)
    
    # Rebuild feeds
    feeds = feed_manager.create_feeds_for_ticker(body.ticker, metadata)
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
    """Force digest with existing articles (for testing) - Enhanced with AI analysis"""
    require_admin(request)
    
    with db() as conn, conn.cursor() as cur:
        if body.tickers:
            cur.execute("""
                SELECT 
                    f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.quality_score, f.published_at,
                    f.found_at, f.category, f.related_ticker, f.original_source_url,
                    f.search_keyword, f.ai_impact, f.ai_reasoning
                FROM found_url f
                WHERE f.found_at >= %s
                    AND f.ticker = ANY(%s)
                ORDER BY f.ticker, f.category, COALESCE(f.published_at, f.found_at) DESC, f.found_at DESC
            """, (datetime.now(timezone.utc) - timedelta(days=7), body.tickers))
        else:
            cur.execute("""
                SELECT 
                    f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.quality_score, f.published_at,
                    f.found_at, f.category, f.related_ticker, f.original_source_url,
                    f.search_keyword, f.ai_impact, f.ai_reasoning
                FROM found_url f
                WHERE f.found_at >= %s
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
    
    # FIXED: Extract HTML from tuple and add empty text attachment
    html = build_enhanced_digest_html(articles_by_ticker, 7)
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

@APP.post("/admin/cleanup-domains")
def admin_cleanup_domains(request: Request):
    """One-time cleanup of domain data"""
    require_admin(request)
    
    updated_count = cleanup_domain_data()
    
    return {
        "status": "completed",
        "records_updated": updated_count,
        "message": "Domain data has been cleaned up. Publication names converted to actual domains."
    }

@APP.post("/admin/reset-ai-analysis")
def reset_ai_analysis(request: Request, tickers: List[str] = Query(default=None, description="Specific tickers to reset")):
    """Reset AI analysis data and force re-scoring of existing articles"""
    require_admin(request)
    
    with db() as conn, conn.cursor() as cur:
        # Clear existing AI analysis data
        if tickers:
            cur.execute("""
                UPDATE found_url 
                SET ai_impact = NULL, 
                    ai_reasoning = NULL,
                    quality_score = 50.0
                WHERE ticker = ANY(%s)
            """, (tickers,))
        else:
            cur.execute("""
                UPDATE found_url 
                SET ai_impact = NULL, 
                    ai_reasoning = NULL,
                    quality_score = 50.0
            """)
        
        reset_count = cur.rowcount
        
        LOG.info(f"Reset AI analysis for {reset_count} articles")
    
    return {
        "status": "ai_analysis_reset",
        "articles_reset": reset_count,
        "tickers": tickers or "all",
        "message": f"Cleared AI analysis data for {reset_count} articles. Run re-analysis to generate fresh scores."
    }

@APP.post("/admin/rerun-ai-analysis")
def rerun_ai_analysis(
    request: Request, 
    tickers: List[str] = Query(default=None, description="Specific tickers to re-analyze"),
    limit: int = Query(default=500, description="Max articles to process per run (no upper limit)")
):
    """Re-run AI analysis on existing articles that have NULL ai_impact - UNLIMITED PROCESSING"""
    require_admin(request)
    
    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OpenAI API key not configured"}
    
    # Get ticker metadata for keywords
    ticker_metadata_cache = {}
    if tickers:
        for ticker in tickers:
            config = get_ticker_config(ticker)
            if config:
                ticker_metadata_cache[ticker] = {
                    "industry_keywords": config.get("industry_keywords", []),
                    "competitors": config.get("competitors", [])
                }
    
    with db() as conn, conn.cursor() as cur:
        # Get articles that need AI analysis - NO LIMIT restriction
        if tickers:
            cur.execute("""
                SELECT id, title, description, domain, ticker, category, search_keyword
                FROM found_url 
                WHERE ai_impact IS NULL 
                    AND ticker = ANY(%s)
                    AND quality_score >= 15
                ORDER BY found_at DESC
                LIMIT %s
            """, (tickers, limit))
        else:
            cur.execute("""
                SELECT id, title, description, domain, ticker, category, search_keyword
                FROM found_url 
                WHERE ai_impact IS NULL 
                    AND quality_score >= 15
                ORDER BY found_at DESC
                LIMIT %s
            """, (limit,))
        
        articles = list(cur.fetchall())
    
    if not articles:
        return {
            "status": "no_articles",
            "message": "No articles found that need AI re-analysis"
        }
    
    processed = 0
    updated = 0
    errors = 0
    
    LOG.info(f"Starting AI re-analysis of {len(articles)} articles (limit={limit})")
    
    for article in articles:
        try:
            processed += 1
            
            # Get appropriate keywords based on category
            ticker = article["ticker"]
            category = article["category"] or "company"
            
            if ticker in ticker_metadata_cache:
                metadata = ticker_metadata_cache[ticker]
                if category == "industry":
                    keywords = metadata.get("industry_keywords", [])
                elif category == "competitor":
                    keywords = metadata.get("competitors", [])
                else:
                    keywords = []
            else:
                keywords = []
            
            # Run AI analysis - UPDATED to handle 4-parameter return
            quality_score, ai_impact, ai_reasoning, components = calculate_quality_score(
                title=article["title"],
                domain=article["domain"],
                ticker=ticker,
                description=article["description"] or "",
                category=category,
                keywords=keywords
            )
            
            # Extract components for database storage
            source_tier = components.get('source_tier') if components else None
            event_multiplier = components.get('event_multiplier') if components else None
            event_multiplier_reason = components.get('event_multiplier_reason') if components else None
            relevance_boost = components.get('relevance_boost') if components else None
            relevance_boost_reason = components.get('relevance_boost_reason') if components else None
            numeric_bonus = components.get('numeric_bonus') if components else None
            penalty_multiplier = components.get('penalty_multiplier') if components else None
            penalty_reason = components.get('penalty_reason') if components else None
            
            # Update the article with ALL scoring data
            with db() as conn, conn.cursor() as cur:
                cur.execute("""
                    UPDATE found_url 
                    SET quality_score = %s, ai_impact = %s, ai_reasoning = %s,
                        source_tier = %s, event_multiplier = %s, event_multiplier_reason = %s,
                        relevance_boost = %s, relevance_boost_reason = %s, numeric_bonus = %s,
                        penalty_multiplier = %s, penalty_reason = %s
                    WHERE id = %s
                """, (
                    quality_score, ai_impact, ai_reasoning,
                    source_tier, event_multiplier, event_multiplier_reason,
                    relevance_boost, relevance_boost_reason, numeric_bonus,
                    penalty_multiplier, penalty_reason,
                    article["id"]
                ))
                
                if cur.rowcount > 0:
                    updated += 1
            
            # Progress logging every 25 articles
            if processed % 25 == 0:
                LOG.info(f"Progress: {processed}/{len(articles)} articles processed, {updated} updated")
            
            # Small delay to avoid overwhelming OpenAI API
            time.sleep(0.1)
            
        except Exception as e:
            errors += 1
            LOG.error(f"Failed to re-analyze article {article['id']}: {e}")
            continue
    
    return {
        "status": "completed",
        "processed": processed,
        "updated": updated,
        "errors": errors,
        "limit_used": limit,
        "tickers": tickers or "all",
        "message": f"Re-analyzed {updated} articles with fresh AI scoring"
    }

@APP.get("/admin/test-yahoo-resolution")
def test_yahoo_resolution(request: Request, url: str = Query(...)):
    """Test Yahoo Finance URL resolution"""
    require_admin(request)
    
    result = domain_resolver.resolve_url_and_domain(url, "Test Title")
    return {
        "original_url": url,
        "resolved_url": result[0],
        "domain": result[1],
        "source_url": result[2]
    }

@APP.post("/admin/import-ticker-reference")
def import_ticker_reference(request: Request, file_path: str = Body(..., embed=True)):
    """Import ticker reference data from CSV file"""
    require_admin(request)

    file_path = r"C:\Users\14166\Desktop\QuantBrief\data\ticker_reference.csv"
    import csv
    import os
    from datetime import datetime
    
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"File not found: {file_path}"}
    
    # Ensure table exists
    create_ticker_reference_table()
    
    imported = 0
    updated = 0
    errors = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            with db() as conn, conn.cursor() as cur:
                for row_num, row in enumerate(reader, start=2):
                    try:
                        # Parse arrays from string
                        industry_keywords = []
                        competitors = []
                        
                        if row.get('industry_keywords'):
                            industry_keywords = [kw.strip() for kw in row['industry_keywords'].split(',')]
                        
                        if row.get('competitors'):
                            competitors = [comp.strip() for comp in row['competitors'].split(',')]
                        
                        # Convert boolean
                        active = row.get('active', 'TRUE').upper() in ('TRUE', '1', 'YES')
                        ai_generated = row.get('ai_generated', 'FALSE').upper() in ('TRUE', '1', 'YES')
                        
                        cur.execute("""
                            INSERT INTO ticker_reference (
                                ticker, country, company_name, industry, sector, 
                                yahoo_ticker, exchange, active, industry_keywords, 
                                competitors, ai_generated
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (ticker) DO UPDATE SET
                                country = EXCLUDED.country,
                                company_name = EXCLUDED.company_name,
                                industry = EXCLUDED.industry,
                                sector = EXCLUDED.sector,
                                yahoo_ticker = EXCLUDED.yahoo_ticker,
                                exchange = EXCLUDED.exchange,
                                active = EXCLUDED.active,
                                industry_keywords = EXCLUDED.industry_keywords,
                                competitors = EXCLUDED.competitors,
                                updated_at = NOW()
                        """, (
                            row['ticker'], row['country'], row['company_name'],
                            row.get('industry'), row.get('sector'),
                            row.get('yahoo_ticker', row['ticker']), row.get('exchange'),
                            active, industry_keywords, competitors, ai_generated
                        ))
                        
                        if cur.rowcount == 1:
                            imported += 1
                        else:
                            updated += 1
                            
                    except Exception as e:
                        errors.append(f"Row {row_num}: {str(e)}")
                        
        return {
            "status": "completed",
            "imported": imported,
            "updated": updated,
            "errors": errors[:10],  # Limit error display
            "total_errors": len(errors)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
            metadata = ticker_manager.get_or_create_metadata(ticker)
            feeds = feed_manager.create_feeds_for_ticker(ticker, metadata)
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
