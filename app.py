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
    "seekingalpha.com",
    "theglobeandmail.com",
    "apnews.com",
    "reuters.com",
}

# Known paywall domains to skip during content scraping
PAYWALL_DOMAINS = {
    # Hard paywalls only
    "wsj.com", "www.wsj.com",
    "ft.com", "www.ft.com",
    "economist.com", "www.economist.com",
    "nytimes.com", "www.nytimes.com",
    "washingtonpost.com", "www.washingtonpost.com",
    
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
    """Complete database schema initialization with all required tables"""
    with db() as conn:
        with conn.cursor() as cur:
            # Create found_url table with all required columns
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
                    quality_score DECIMAL(5,2) DEFAULT 50.0,
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
                    ai_impact VARCHAR(20),
                    ai_reasoning TEXT,
                    ai_summary TEXT,
                    source_tier DECIMAL(3,2),
                    event_multiplier DECIMAL(3,2),
                    event_multiplier_reason TEXT,
                    relevance_boost DECIMAL(3,2),
                    relevance_boost_reason TEXT,
                    numeric_bonus DECIMAL(3,2),
                    penalty_multiplier DECIMAL(3,2),
                    penalty_reason TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                
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
                
                -- Add any missing columns to existing tables
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS ai_impact VARCHAR(20);
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS ai_reasoning TEXT;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS ai_summary TEXT;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS source_tier DECIMAL(3,2);
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS event_multiplier DECIMAL(3,2);
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS event_multiplier_reason TEXT;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS relevance_boost DECIMAL(3,2);
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS relevance_boost_reason TEXT;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS numeric_bonus DECIMAL(3,2);
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS penalty_multiplier DECIMAL(3,2);
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS penalty_reason TEXT;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS scraped_content TEXT;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS content_scraped_at TIMESTAMP;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS scraping_failed BOOLEAN DEFAULT FALSE;
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS scraping_error TEXT;
                ALTER TABLE source_feed ADD COLUMN IF NOT EXISTS category VARCHAR(20) DEFAULT 'company';
                ALTER TABLE found_url ADD COLUMN IF NOT EXISTS competitor_ticker VARCHAR(10);
                
                -- Essential indexes
                CREATE INDEX IF NOT EXISTS idx_found_url_hash ON found_url(url_hash);
                CREATE INDEX IF NOT EXISTS idx_found_url_ticker_published ON found_url(ticker, published_at DESC);
                CREATE INDEX IF NOT EXISTS idx_found_url_digest ON found_url(sent_in_digest, found_at DESC);
                CREATE INDEX IF NOT EXISTS idx_ticker_config_active ON ticker_config(active);
                CREATE INDEX IF NOT EXISTS idx_source_feed_ticker_category ON source_feed(ticker, category, active);
                CREATE INDEX IF NOT EXISTS idx_found_url_source_tier ON found_url(source_tier);
                CREATE INDEX IF NOT EXISTS idx_found_url_event_multiplier ON found_url(event_multiplier);
                CREATE INDEX IF NOT EXISTS idx_found_url_quality_score ON found_url(quality_score);
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

def update_schema_for_triage():
    """Add triage fields to found_url table"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE found_url ADD COLUMN IF NOT EXISTS ai_triage_selected BOOLEAN DEFAULT FALSE;
            ALTER TABLE found_url ADD COLUMN IF NOT EXISTS triage_priority INTEGER;
            ALTER TABLE found_url ADD COLUMN IF NOT EXISTS triage_reasoning TEXT;
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
    Enhanced article extraction with intelligent delay management
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
        content = article.text.strip()
        
        if not content:
            return None, "No content extracted"
        
        # Enhanced cookie banner detection
        cookie_indicators = [
            "we use cookies", "accept all cookies", "cookie policy",
            "privacy policy and terms of service", "consent to the use",
            "personalizing content and advertising", "marketing cookies",
            "essential cookies", "functional cookies", "deny optional",
            "accept all or closing out of this banner", "revised from time to time"
        ]
        
        content_lower = content.lower()
        cookie_count = sum(1 for indicator in cookie_indicators if indicator in content_lower)
        
        # If multiple cookie indicators and content is short, likely a cookie page
        if cookie_count >= 3 and len(content) < 800:
            return None, "Cookie consent page detected"
        
        # If content starts with cookie text, likely a banner page
        cookie_start_indicators = [
            "we use cookies to understand",
            "this site uses cookies",
            "by continuing to use this site"
        ]
        if any(content_lower.startswith(indicator) for indicator in cookie_start_indicators):
            return None, "Cookie banner content detected"
        
        # Enhanced paywall indicators (try scraping first, then check)
        paywall_indicators = [
            "subscribe to continue", "unlock this story", "premium content",
            "sign up to read", "become a member", "subscription required",
            "create free account", "log in to continue", "paywall",
            "this article is for subscribers", "register to read",
            "403 forbidden", "401 unauthorized", "access denied",
            "upgrade to premium", "become a subscriber"
        ]
        
        if any(indicator in content_lower for indicator in paywall_indicators):
            return None, "Paywall content detected"
        
        # Check for error pages
        error_indicators = [
            "404", "page not found", "forbidden",
            "this page doesn't exist", "error occurred",
            "internal server error", "service unavailable"
        ]
        
        if any(error in content_lower for error in error_indicators):
            return None, "Error page detected"
        
        # Validate content quality
        is_valid, validation_msg = validate_scraped_content(content, url, domain)
        if not is_valid:
            return None, validation_msg
        
        # Store full content (no length limit)
        LOG.info(f"Successfully extracted {len(content)} chars from {domain}")
        return content, None
        
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
    Memory-optimized article extraction using Playwright for JavaScript-heavy sites
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
                    '--disable-dev-shm-usage',  # Use /tmp instead of /dev/shm for lower memory
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--memory-pressure-off',
                    '--disable-background-timer-throttling',
                    '--disable-renderer-backgrounding',
                    '--disable-extensions',
                    '--disable-plugins',
                    '--disable-images',  # Don't load images to save memory
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
                content = None
                extraction_method = None
                
                # Method 1: Try article tag first
                try:
                    article_element = page.query_selector('article')
                    if article_element:
                        content = article_element.inner_text()
                        extraction_method = "article tag"
                except Exception:
                    pass
                
                # Method 2: Try main content selectors
                if not content or len(content.strip()) < 200:
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
                                    content = temp_content
                                    extraction_method = f"selector: {selector}"
                                    break
                        except Exception:
                            continue
                
                # Method 3: Smart body text extraction (removes navigation/ads)
                if not content or len(content.strip()) < 200:
                    try:
                        content = page.evaluate("""
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
                            content = page.evaluate("() => document.body ? document.body.innerText : ''")
                            extraction_method = "fallback body text"
                        except Exception:
                            content = None
                
            except Exception as e:
                LOG.warning(f"PLAYWRIGHT: Navigation/extraction failed for {domain}: {str(e)}")
                content = None
            finally:
                # Always close browser to free memory
                try:
                    browser.close()
                except Exception:
                    pass
            
            if not content or len(content.strip()) < 100:
                LOG.warning(f"PLAYWRIGHT FAILED: {domain} -> Insufficient content extracted")
                return None, "Insufficient content extracted"
            
            # Enhanced content validation
            is_valid, validation_msg = validate_scraped_content(content, url, domain)
            if not is_valid:
                LOG.warning(f"PLAYWRIGHT FAILED: {domain} -> {validation_msg}")
                return None, validation_msg
            
            # Check for common error pages or blocking messages
            content_lower = content.lower()
            error_indicators = [
                "403 forbidden", "access denied", "captcha", "robot", "bot detection",
                "please verify you are human", "cloudflare", "rate limit", "blocked",
                "security check", "unusual traffic"
            ]
            
            if any(indicator in content_lower for indicator in error_indicators):
                LOG.warning(f"PLAYWRIGHT FAILED: {domain} -> Error page detected")
                return None, "Error page or blocking detected"
            
            LOG.info(f"PLAYWRIGHT SUCCESS: {domain} -> {len(content)} chars extracted via {extraction_method}")
            return content.strip(), None
            
    except Exception as e:
        error_msg = f"Playwright extraction failed: {str(e)}"
        LOG.error(f"PLAYWRIGHT ERROR: {domain} -> {error_msg}")
        return None, error_msg
    finally:
        # Add this cleanup
        import gc
        gc.collect()

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
    """Enhanced scraping with exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = scraping_session.get(url, timeout=15)
            if response.status_code == 200:
                return response
            elif response.status_code in [429, 503]:
                # Rate limited - wait longer
                delay = (2 ** attempt) + random.uniform(0, 1)
                LOG.info(f"Rate limited, waiting {delay:.1f}s")
                time.sleep(delay)
            else:
                return None
        except requests.RequestException as e:
            delay = (2 ** attempt) + random.uniform(0, 1)
            LOG.warning(f"Request failed, retrying in {delay:.1f}s: {e}")
            time.sleep(delay)
    
    return None

def validate_scraped_content(content, url, domain):
    """Multi-step content validation"""
    if not content or len(content.strip()) < 100:
        return False, "Content too short"
    
    # Check content-to-boilerplate ratio
    sentences = content.split('.')
    if len(sentences) < 3:
        return False, "Insufficient sentences"
    
    # Check for repetitive content (often indicates scraping issues)
    words = content.lower().split()
    if len(words) > 0 and len(set(words)) / len(words) < 0.3:  # Less than 30% unique words
        return False, "Repetitive content detected"
    
    return True, "Valid content"

# Create global session
scraping_session = create_scraping_session()

def is_insider_trading_article(title: str) -> bool:
    """
    Detect insider trading/institutional flow articles that are typically low-value
    """
    title_lower = title.lower()
    
    # Insider trading patterns
    insider_patterns = [
        # Executive transactions
        r"\w+\s+(ceo|cfo|coo|president|director|officer|executive)\s+\w+\s+(sells?|buys?|purchases?)",
        r"(sells?|buys?|purchases?)\s+\$[\d,]+\.?\d*[km]?\s+(in\s+shares?|worth\s+of)",
        r"(insider|executive|officer|director|ceo|cfo)\s+(selling|buying|sold|bought)",
        
        # Institutional flow patterns  
        r"\w+\s+(capital|management|advisors?|investments?|llc|inc\.?)\s+(buys?|sells?|invests?|increases?|decreases?)",
        r"(invests?|buys?)\s+\$[\d,]+\.?\d*[km]?\s+in",
        r"shares?\s+(sold|bought)\s+by\s+",
        r"(increases?|decreases?|trims?|adds?\s+to)\s+(stake|position|holdings?)\s+in",
        
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
            return True
    
    # Additional heuristics
    # Small dollar amounts (under $50M) with "sells" or "buys"
    small_amount_pattern = r"\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*([km])?"
    matches = re.findall(small_amount_pattern, title_lower)
    if matches and any(word in title_lower for word in ["sells", "buys", "purchases", "sold", "bought"]):
        for amount_str, unit in matches:
            try:
                amount = float(amount_str.replace(",", ""))
                if unit.lower() == 'k':
                    amount *= 1000
                elif unit.lower() == 'm':
                    amount *= 1000000
                
                # Flag transactions under $50M as likely insider trading
                if amount < 50000000:
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

def safe_content_scraper_with_playwright(url: str, domain: str, category: str, keyword: str, scraped_domains: set) -> Tuple[Optional[str], str]:
    """
    Enhanced scraper with per-keyword limits - only successful scrapes count toward limit
    Updated for new scraping logic: 20 company + 5×keywords + 5×competitors
    """
    global scraping_stats
    
    # Check limits based on category
    if not _check_scraping_limit(category, keyword):
        if category == "company":
            return None, f"Company limit reached ({scraping_stats['company_scraped']}/{scraping_stats['limits']['company']})"
        elif category == "industry":
            keyword_count = scraping_stats["industry_scraped_by_keyword"].get(keyword, 0)
            return None, f"Industry keyword '{keyword}' limit reached ({keyword_count}/{scraping_stats['limits']['industry_per_keyword']})"
        elif category == "competitor":
            keyword_count = scraping_stats["competitor_scraped_by_keyword"].get(keyword, 0)
            return None, f"Competitor '{keyword}' limit reached ({keyword_count}/{scraping_stats['limits']['competitor_per_keyword']})"
    
    # First try standard method
    content, error = safe_content_scraper(url, domain, scraped_domains)
    
    # If successful with requests, count it and return
    if content:
        _update_scraping_stats(category, keyword, True)
        return content, f"Successfully scraped {len(content)} chars with requests"
    
    # Log failed attempt but don't count toward limit
    scraping_stats["failed_scrapes"] += 1
    
    # Try Playwright fallback for ANY domain (not just high-value ones)
    LOG.info(f"Trying Playwright fallback for: {domain} (failed attempts don't count toward limit)")
    
    # Update Playwright stats (for monitoring, not limits)
    playwright_stats["attempted"] += 1
    normalized_domain = normalize_domain(domain)
    playwright_stats["by_domain"][normalized_domain]["attempts"] += 1
    
    playwright_content, playwright_error = extract_article_content_with_playwright(url, domain)
    
    if playwright_content:
        # SUCCESS - count toward limit
        playwright_stats["successful"] += 1
        playwright_stats["by_domain"][normalized_domain]["successes"] += 1
        
        _update_scraping_stats(category, keyword, True)
        
        # Log stats every 10 attempts
        if playwright_stats["attempted"] % 10 == 0:
            log_playwright_stats()
            
        return playwright_content, f"Playwright success: {len(playwright_content)} chars"
    else:
        # FAILURE - don't count toward limit
        playwright_stats["failed"] += 1
        scraping_stats["failed_scrapes"] += 1
        
        # Log stats every 10 attempts
        if playwright_stats["attempted"] % 10 == 0:
            log_playwright_stats()
            
        return None, f"Both methods failed - Requests: {error}, Playwright: {playwright_error} (not counted toward limit)"

def safe_content_scraper_with_playwright_limited(url: str, domain: str, category: str, keyword: str, scraped_domains: set) -> Tuple[Optional[str], str]:
    """
    Enhanced scraper with per-keyword limits - only successful scrapes count toward limit
    """
    global scraping_stats
    
    # Check limits based on category
    if category == "company":
        if scraping_stats["company_scraped"] >= scraping_stats["limits"]["company"]:
            return None, f"Company limit reached ({scraping_stats['company_scraped']}/{scraping_stats['limits']['company']})"
    
    elif category == "industry":
        keyword_count = scraping_stats["industry_scraped_by_keyword"].get(keyword, 0)
        if keyword_count >= scraping_stats["limits"]["industry_per_keyword"]:
            return None, f"Industry keyword '{keyword}' limit reached ({keyword_count}/{scraping_stats['limits']['industry_per_keyword']})"
    
    elif category == "competitor":
        keyword_count = scraping_stats["competitor_scraped_by_keyword"].get(keyword, 0)
        if keyword_count >= scraping_stats["limits"]["competitor_per_keyword"]:
            return None, f"Competitor '{keyword}' limit reached ({keyword_count}/{scraping_stats['limits']['competitor_per_keyword']})"
    
    # First try your existing method
    content, error = safe_content_scraper(url, domain, scraped_domains)
    
    # If successful with requests, count it and return
    if content:
        _update_scraping_stats(category, keyword, True)
        return content, f"Successfully scraped {len(content)} chars with requests"
    
    # Log failed attempt but don't count toward limit
    scraping_stats["failed_scrapes"] += 1
    
    # Try Playwright fallback for ANY domain (not just high-value ones)
    LOG.info(f"Trying Playwright fallback for: {domain} (failed attempts don't count toward limit)")
    
    # Update Playwright stats (for monitoring, not limits)
    playwright_stats["attempted"] += 1
    normalized_domain = normalize_domain(domain)
    playwright_stats["by_domain"][normalized_domain]["attempts"] += 1
    
    playwright_content, playwright_error = extract_article_content_with_playwright(url, domain)
    
    if playwright_content:
        # SUCCESS - count toward limit
        playwright_stats["successful"] += 1
        playwright_stats["by_domain"][normalized_domain]["successes"] += 1
        
        _update_scraping_stats(category, keyword, True)
        
        # Log stats every 10 attempts
        if playwright_stats["attempted"] % 10 == 0:
            log_playwright_stats()
            
        return playwright_content, f"Playwright success: {len(playwright_content)} chars"
    else:
        # FAILURE - don't count toward limit
        playwright_stats["failed"] += 1
        scraping_stats["failed_scrapes"] += 1
        
        # Log stats every 10 attempts
        if playwright_stats["attempted"] % 10 == 0:
            log_playwright_stats()
            
        return None, f"Both methods failed - Requests: {error}, Playwright: {playwright_error} (not counted toward limit)"

def scrape_and_analyze_article(article: Dict, category: str, metadata: Dict, ticker: str) -> bool:
    """Scrape content and run AI analysis for a single article with dynamic limits"""
    try:
        article_id = article["id"]
        resolved_url = article.get("resolved_url") or article.get("url")
        domain = article.get("domain", "unknown")
        title = article.get("title", "")
        
        # FIXED:
        if category == "company":
            keyword = ticker
        elif category == "competitor":
            # Use competitor_ticker for limit consolidation
            keyword = article.get("competitor_ticker") or article.get("search_keyword", "unknown")
        else:
            keyword = article.get("search_keyword", "unknown")
        
        # Attempt content scraping with limits
        scraped_content = None
        scraping_error = None
        content_scraped_at = None
        scraping_failed = False
        ai_summary = None
        
        if resolved_url and resolved_url.startswith(('http://', 'https://')):
            scrape_domain = normalize_domain(urlparse(resolved_url).netloc.lower())
            
            if scrape_domain not in PAYWALL_DOMAINS:
                # Use the enhanced scraper with limits
                content, status = safe_content_scraper_with_playwright_limited(
                    resolved_url, scrape_domain, category, keyword, set()
                )
                
                if content:
                    # Scraping successful
                    scraped_content = content
                    content_scraped_at = datetime.now(timezone.utc)
                    
                    # Generate AI summary from scraped content
                    ai_summary = generate_ai_summary(scraped_content, title, ticker)
                else:
                    scraping_failed = True
                    scraping_error = status
                    return False  # Failed scraping doesn't count as success
        
        # Run AI quality scoring
        keywords = []
        if category == "industry":
            keywords = metadata.get("industry_keywords", [])
        elif category == "competitor":
            keywords = metadata.get("competitors", [])
        
        quality_score, ai_impact, ai_reasoning, components = calculate_quality_score(
            title=title,
            domain=domain,
            ticker=ticker,
            description=article.get("description", ""),
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
        
        # Update the article in database
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE found_url 
                SET scraped_content = %s, content_scraped_at = %s, scraping_failed = %s, scraping_error = %s,
                    ai_summary = %s, quality_score = %s, ai_impact = %s, ai_reasoning = %s,
                    source_tier = %s, event_multiplier = %s, event_multiplier_reason = %s,
                    relevance_boost = %s, relevance_boost_reason = %s, numeric_bonus = %s,
                    penalty_multiplier = %s, penalty_reason = %s
                WHERE id = %s
            """, (
                scraped_content, content_scraped_at, scraping_failed, scraping_error,
                ai_summary, quality_score, ai_impact, ai_reasoning,
                source_tier, event_multiplier, event_multiplier_reason,
                relevance_boost, relevance_boost_reason, numeric_bonus,
                penalty_multiplier, penalty_reason,
                article_id
            ))
        
        LOG.info(f"Scraped and analyzed with limits: {title[:50]}... (Score: {quality_score:.1f}, Content: {'Yes' if scraped_content else 'No'})")
        return scraped_content is not None  # Only return True if we actually got content
        
    except Exception as e:
        LOG.error(f"Failed to scrape and analyze article {article.get('id')} with limits: {e}")
        return False

def _update_scraping_stats(category: str, keyword: str, success: bool):
    """Helper to update scraping statistics"""
    global scraping_stats
    
    if success:
        scraping_stats["successful_scrapes"] += 1
        
        if category == "company":
            scraping_stats["company_scraped"] += 1
            # ... existing logging
        
        elif category == "industry":
            if keyword not in scraping_stats["industry_scraped_by_keyword"]:
                scraping_stats["industry_scraped_by_keyword"][keyword] = 0
            scraping_stats["industry_scraped_by_keyword"][keyword] += 1
            # ... existing logging
        
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
    Enhanced feed processing with AI summaries for scraped content
    Flow: Check scraping limits -> Scrape resolved URLs -> AI analysis only on successful scrapes
    """
    global scraping_stats
    
    stats = {
        "processed": 0, 
        "inserted": 0, 
        "duplicates": 0, 
        "blocked_spam": 0, 
        "blocked_non_latin": 0,
        "content_scraped": 0,
        "content_failed": 0,
        "scraping_skipped": 0,
        "ai_reanalyzed": 0,
        "ai_scored": 0,
        "basic_scored": 0,
        "ai_summaries_generated": 0
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
            
            # URL resolution
            resolved_url, domain, source_url = domain_resolver.resolve_url_and_domain(url, title)
            if not resolved_url or not domain:
                stats["blocked_spam"] += 1
                continue
            
            # Handle Google->Yahoo redirects
            is_google_to_yahoo = (
                "news.google.com" in url and 
                "finance.yahoo.com" in url and 
                resolved_url != url
            )
            
            if is_google_to_yahoo:
                original_source = extract_yahoo_finance_source_optimized(resolved_url)
                if original_source:
                    final_resolved_url = original_source
                    final_domain = normalize_domain(urlparse(original_source).netloc.lower())
                    final_source_url = resolved_url
                else:
                    final_resolved_url = resolved_url
                    final_domain = domain
                    final_source_url = source_url
            else:
                final_resolved_url = resolved_url
                final_domain = domain
                final_source_url = source_url
            
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
                            content, status = safe_content_scraper_with_playwright_limited(
                                final_resolved_url, scrape_domain, category, feed_keyword, scraped_domains
                            )
                            
                            if content:
                                # Scraping successful - enable AI processing
                                scraped_content = content
                                content_scraped_at = datetime.now(timezone.utc)
                                stats["content_scraped"] += 1
                                should_use_ai = True
                                
                                # Generate AI summary from scraped content
                                ai_summary = generate_ai_summary(scraped_content, title, feed["ticker"])
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
                    
                    # Insert article with AI summary
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
                        source_info = "via Google→Yahoo" if is_google_to_yahoo else "direct"
                        LOG.info(f"Inserted [{category}] from {domain_resolver.get_formal_name(final_domain)}: {title[:60]}... ({processing_type}, {content_info}) ({source_info})")
                        
            except Exception as e:
                LOG.error(f"Database error for '{title[:50]}': {e}")
                continue
                
    except Exception as e:
        LOG.error(f"Feed processing error for {feed['name']}: {e}")
    
    return stats

def ingest_feed_basic_only(feed: Dict) -> Dict[str, int]:
    """Basic feed ingestion with per-category limits during ingestion (50/25/25) - COUNT ONLY UNIQUE URLs"""
    stats = {"processed": 0, "inserted": 0, "duplicates": 0, "blocked_spam": 0, "blocked_non_latin": 0, "limit_reached": 0}
    
    category = feed.get("category", "company")
    
    # Use competitor_ticker for competitor feeds, search_keyword for others
    if category == "competitor":
        feed_keyword = feed.get("competitor_ticker", "unknown")
    else:
        feed_keyword = feed.get("search_keyword", "unknown")
    
    try:
        parsed = feedparser.parse(feed["url"])
        
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
            
            # Check limit BEFORE processing - this now includes ALL URLs
            if not _check_ingestion_limit(category, feed_keyword):
                stats["limit_reached"] += 1
                LOG.info(f"INGESTION LIMIT REACHED: Stopping ingestion for {category} '{feed_keyword}'")
                break
            
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
            
            # URL resolution
            resolved_url, domain, source_url = domain_resolver.resolve_url_and_domain(url, title)
            if not resolved_url or not domain:
                stats["blocked_spam"] += 1
                continue
            
            # Handle Google->Yahoo redirects
            is_google_to_yahoo = (
                "news.google.com" in url and 
                "finance.yahoo.com" in url and 
                resolved_url != url
            )
            
            if is_google_to_yahoo:
                original_source = extract_yahoo_finance_source_optimized(resolved_url)
                if original_source:
                    final_resolved_url = original_source
                    final_domain = normalize_domain(urlparse(original_source).netloc.lower())
                    final_source_url = resolved_url
                else:
                    final_resolved_url = resolved_url
                    final_domain = domain
                    final_source_url = source_url
            else:
                final_resolved_url = resolved_url
                final_domain = domain
                final_source_url = source_url
            
            url_hash = get_url_hash(url, final_resolved_url)
            
            try:
                with db() as conn, conn.cursor() as cur:
                    # Check for duplicates
                    cur.execute("SELECT id FROM found_url WHERE url_hash = %s", (url_hash,))
                    if cur.fetchone():
                        stats["duplicates"] += 1
                        # FIXED: DO NOT count duplicates toward limit - they are not unique URLs
                        LOG.debug(f"DUPLICATE SKIPPED: {title[:50]}... (not counted toward limit)")
                        continue
                    
                    # Parse publish date
                    published_at = None
                    if hasattr(entry, "published_parsed"):
                        published_at = parse_datetime(entry.published_parsed)
                    
                    # Use truly basic scoring - NO AI CALLS WHATSOEVER
                    domain_tier = _get_domain_tier(final_domain, title, description)
                    basic_quality_score = 50.0 + (domain_tier - 0.5) * 20
                    
                    # Add ticker mention bonus
                    if feed["ticker"].upper() in title.upper():
                        basic_quality_score += 10
                    
                    # Clamp to reasonable range
                    basic_quality_score = max(20.0, min(80.0, basic_quality_score))
                    
                    display_content = description
                    
                    # Insert article with basic data only
                    cur.execute("""
                        INSERT INTO found_url (
                            url, resolved_url, url_hash, title, description,
                            feed_id, ticker, domain, quality_score, published_at,
                            category, search_keyword, original_source_url,
                            competitor_ticker
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        url, final_resolved_url, url_hash, title, display_content,
                        feed["id"], feed["ticker"], final_domain, basic_quality_score, published_at,
                        category, feed.get("search_keyword"), final_source_url,
                        feed.get("competitor_ticker")
                    ))
                    
                    if cur.fetchone():
                        stats["inserted"] += 1
                        # ONLY count UNIQUE URLs toward limit
                        _update_ingestion_stats(category, feed_keyword)
                        LOG.debug(f"UNIQUE URL COUNTED: {title[:50]}...")
                        
            except Exception as e:
                LOG.error(f"Database error for '{title[:50]}': {e}")
                continue
                
    except Exception as e:
        LOG.error(f"Feed processing error for {feed['name']}: {e}")
    
    return stats

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
    Enhanced article HTML formatting with AI summaries and database-based competitor names - NO REDUNDANT KEYWORDS
    """
    import html
    
    # Format timestamp for individual articles
    if article["published_at"]:
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
    
    # Quality score styling - only show if AI analyzed
    score_html = ""
    analyzed_html = ""
    impact_html = ""
    quality_score = article.get("quality_score")
    ai_impact = article.get("ai_impact")
    
    if ai_impact is not None and quality_score is not None:
        # This article was AI analyzed - show score
        score_class = "high-score" if quality_score >= 70 else "med-score" if quality_score >= 40 else "low-score"
        score_html = f'<span class="score {score_class}">Score: {quality_score:.0f}</span>'
        
        # Show impact next to score with appropriate styling
        impact_class = {
            "positive": "impact-positive", 
            "negative": "impact-negative", 
            "mixed": "impact-mixed", 
            "neutral": "impact-neutral"
        }.get(ai_impact.lower(), "impact-neutral")
        impact_html = f'<span class="impact {impact_class}">{ai_impact}</span>'
        
        # Show "Analyzed" badge if we have scraped content and AI summary
        if article.get('scraped_content') and article.get('ai_summary'):
            analyzed_html = f'<span class="analyzed-badge">Analyzed</span>'
    
    # Build metadata badges for category-specific information with DATABASE LOOKUP
    metadata_badges = []
    
    if category == "competitor":
        # Use competitor_ticker first (most reliable), fallback to search_keyword
        competitor_ticker = article.get('competitor_ticker')
        search_keyword = article.get('search_keyword')
        
        competitor_name = get_competitor_display_name(search_keyword, competitor_ticker)
        metadata_badges.append(f'<span class="competitor-badge">🏢 {competitor_name}</span>')
        
    elif category == "industry" and article.get('search_keyword'):
        industry_keyword = article['search_keyword']
        metadata_badges.append(f'<span class="industry-badge">🏭 {industry_keyword}</span>')
    
    enhanced_metadata = "".join(metadata_badges)
    
    # AI Summary section (replaces scraped content display)
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
            <span class='source-badge'>{display_source}</span>
            {enhanced_metadata}
            {score_html}
            {impact_html}
            {analyzed_html}
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
    """Get ticker configuration from database"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT ticker, name, industry_keywords, competitors, ai_generated
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

# Update the calculate_quality_score function to return components
def calculate_quality_score(
    title: str, 
    domain: str, 
    ticker: str,
    description: str = "",
    category: str = "company",
    keywords: List[str] = None
) -> Tuple[float, Optional[str], Optional[str], Optional[Dict]]:
    """Calculate quality score using component-based AI scoring - UPDATED to return components
    Returns: (score, impact, reasoning, components)
    """
    
    # Debug logging
    LOG.info(f"SCORING: {category} article for {ticker}: '{title[:30]}...'")
    
    # Pre-filter spam
    if _is_spam_content(title, domain, description):
        return 0.0, "Negative", "Spam content detected", None
     
    # If no OpenAI key, use fallback scoring
    if not OPENAI_API_KEY or not title.strip():
        score = _fallback_quality_score(title, domain, ticker, description, keywords)
        return score, None, None, None
    
    # Use category-specific AI component extraction
    try:
        if category.lower() in ["company", "company_news", "comp"]:
            score, impact, reasoning, components = _ai_quality_score_company_components(title, domain, ticker, description, keywords)
        elif category.lower() in ["industry", "industry_market", "market"]:
            score, impact, reasoning, components = _ai_quality_score_industry_components(title, domain, ticker, description, keywords)
        elif category.lower() in ["competitor", "competitor_intel", "competition"]:
            score, impact, reasoning, components = _ai_quality_score_competitor_components(title, domain, ticker, description, keywords)
        else:
            # Fallback for unknown categories
            score = _fallback_quality_score(title, domain, ticker, description, keywords)
            return score, None, None, None
            
        LOG.info(f"Component-based score [{ticker}] '{title[:50]}...' from {domain}: {score:.1f} ({category})")
        return max(0.0, min(100.0, score)), impact, reasoning, components
    except Exception as e:
        LOG.warning(f"AI component scoring failed for '{title[:50]}...': {e}")
        score = _fallback_quality_score(title, domain, ticker, description, keywords)
        return score, None, None, None

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

def generate_ai_summary(scraped_content: str, title: str, ticker: str, description: str = "") -> Optional[str]:
    """Generate hedge fund analyst summary from scraped content using enhanced prompt"""
    if not OPENAI_API_KEY or not scraped_content or len(scraped_content.strip()) < 200:
        return None
    
    try:
        prompt = f"""As a hedge-fund analyst, summarize the article about {ticker} in 3-5 sentences.

Use this decision logic:
1) Source text = scraped_content.
2) Extract: (a) WHAT happened (entity + action), (b) MAGNITUDE (only numbers actually present: EPS/rev/%, $ capex, units/capacity, price moves, dates), (c) WHY it matters (cost/price/volume/regulatory effect), (d) TIMING/CATALYSTS (deadlines, commissioning, votes, guidance dates), (e) NET TAKEAWAY (implication for economics/positioning).
3) If text is PR/opinion/preview or lacks hard numbers, state that plainly and keep to verifiable facts.
4) Do not invent data or compare to prior periods unless stated. No recommendations.

Write 3-5 sentences, crisp and factual.

ticker: {ticker}
title: {title}
description_snippet: {description[:500] if description else ""}
scraped_content: {scraped_content[:2000]}"""

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a hedge fund analyst. Provide concise, analytical summaries focusing on financial implications."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.3
        }
        
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            summary = result["choices"][0]["message"]["content"].strip()
            LOG.info(f"Generated enhanced AI summary for {ticker}: {len(summary)} chars")
            return summary
        else:
            LOG.warning(f"AI summary failed: {response.status_code}")
            return None
            
    except Exception as e:
        LOG.warning(f"AI summary generation failed: {e}")
        return None

def perform_ai_triage_batch(articles_by_category: Dict[str, List[Dict]], ticker: str) -> Dict[str, List[Dict]]:
    """
    Perform AI triage on batched articles to identify scraping candidates
    Returns dict with category -> list of selected article data (including priority, reasoning)
    """
    if not OPENAI_API_KEY:
        LOG.warning("OpenAI API key not configured - skipping triage")
        return {"company": [], "industry": [], "competitor": []}
    
    selected_results = {"company": [], "industry": [], "competitor": []}
    
    # Get ticker metadata for enhanced prompts
    config = get_ticker_config(ticker)
    company_name = config.get("name", ticker) if config else ticker
    
    # Build enhanced metadata
    aliases_brands_assets = {"aliases": [], "brands": [], "assets": []}
    sector_profile = {"core_inputs": [], "core_channels": [], "core_geos": [], "benchmarks": []}
    
    if config:
        try:
            if config.get("aliases_brands_assets"):
                aliases_brands_assets = json.loads(config["aliases_brands_assets"]) if isinstance(config["aliases_brands_assets"], str) else config["aliases_brands_assets"]
            if config.get("sector_profile"):
                sector_profile = json.loads(config["sector_profile"]) if isinstance(config["sector_profile"], str) else config["sector_profile"]
        except:
            pass
    
    # Process each category
    for category, articles in articles_by_category.items():
        if not articles:
            continue
            
        LOG.info(f"Starting AI triage for {category}: {len(articles)} articles")
        
        try:
            if category == "company":
                selected = triage_company_articles_full(articles, ticker, company_name, aliases_brands_assets, sector_profile)
            elif category == "industry":
                peers = config.get("competitors", []) if config else []
                selected = triage_industry_articles_full(articles, ticker, sector_profile, peers)
            elif category == "competitor":
                peers = config.get("competitors", []) if config else []
                selected = triage_competitor_articles_full(articles, ticker, peers, sector_profile)
            else:
                selected = []
                
            selected_results[category] = selected
            LOG.info(f"AI triage {category}: selected {len(selected)} articles for scraping")
            
        except Exception as e:
            LOG.error(f"AI triage failed for {category}: {e}")
            selected_results[category] = []
    
    return selected_results

def _apply_tiered_backfill_to_limits(articles: List[Dict], ai_selected: List[Dict], category: str, low_quality_domains: Set[str], target_limit: int) -> List[Dict]:
    """
    Apply tiered backfill to reach target limits using domain rankings
    NEVER reduce AI + Quality selections, only backfill UP TO limit
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
        
        # Skip low-quality domains and insider trading
        if domain in low_quality_domains or is_insider_trading_article(title):
            continue
            
        if domain in QUALITY_DOMAINS:
            quality_selected.append({
                "id": idx,
                "scrape_priority": 2,
                "likely_repeat": False,
                "repeat_key": "",
                "why": f"Quality domain: {domain}",
                "confidence": 0.8
            })
            selected_indices.add(idx)
    
    combined_selected.extend(quality_selected)
    
    # Step 3: Tiered backfill ONLY if under target (never reduce)
    current_count = len(combined_selected)
    backfill_selected = []
    
    if current_count < target_limit:
        remaining_slots = target_limit - current_count
        
        # Group remaining articles by domain tier
        tier_groups = {}
        for idx, article in enumerate(articles):
            if idx in selected_indices:
                continue
                
            domain = normalize_domain(article.get("domain", ""))
            title = article.get("title", "").lower()
            
            # Skip low-quality domains and insider trading
            if domain in low_quality_domains or is_insider_trading_article(title):
                continue
            
            tier = DOMAIN_TIERS.get(domain, 0.3)  # Default tier for unknown domains
            
            if tier not in tier_groups:
                tier_groups[tier] = []
            tier_groups[tier].append({
                "id": idx,
                "article": article,
                "tier": tier,
                "domain": domain
            })
        
        # Fill remaining slots starting from highest tier
        for tier in sorted(tier_groups.keys(), reverse=True):  # High to low (1.0 → 0.1)
            if len(backfill_selected) >= remaining_slots:
                break
                
            tier_articles = tier_groups[tier]
            
            # Sort by publication time within tier (newest first)
            tier_articles.sort(
                key=lambda x: x["article"].get("published_at") or datetime.min.replace(tzinfo=timezone.utc), 
                reverse=True
            )
            
            slots_available = remaining_slots - len(backfill_selected)
            for article_data in tier_articles[:slots_available]:
                backfill_selected.append({
                    "id": article_data["id"],
                    "scrape_priority": 3,
                    "likely_repeat": False,
                    "repeat_key": "",
                    "why": f"Tier {article_data['tier']} backfill from {article_data['domain']}",
                    "confidence": 0.4
                })
        
        combined_selected.extend(backfill_selected)
    
    # Final sort by priority (lower number = higher priority)
    combined_selected.sort(key=lambda x: x.get("scrape_priority", 5))
    
    # NEVER REDUCE - return all AI + Quality + Backfill selections
    final_selected = combined_selected  # REMOVED the [:target_limit] slice
    
    # Enhanced logging
    ai_count = len(ai_selected)
    quality_count = len(quality_selected)
    backfill_count = len(backfill_selected)
    
    LOG.info(f"Tiered backfill {category}: {ai_count} AI + {quality_count} Quality + {backfill_count} Backfill = {len(final_selected)}/{target_limit}")
    
    return final_selected

def perform_ai_triage_batch_with_tiered_backfill(articles_by_category: Dict[str, List[Dict]], ticker: str, target_limits: Dict[str, int] = None) -> Dict[str, List[Dict]]:
    """
    Enhanced triage with tiered backfill to reach target limits
    """
    if not OPENAI_API_KEY:
        LOG.warning("OpenAI API key not configured - using quality domains and backfill only")
        return {"company": [], "industry": [], "competitor": []}
    
    # Default limits if not provided
    if not target_limits:
        target_limits = {"company": 20, "industry": 15, "competitor": 15}
    
    selected_results = {"company": [], "industry": [], "competitor": []}
    
    # Low-quality domains to avoid in manual triage
    LOW_QUALITY_DOMAINS = {
        "defense-world.net", "defensenews.com", "defenseworld.net",
        "zacks.com", "thefly.com", "accesswire.com", "streetinsider.com"
    }
    
    # Get ticker metadata for enhanced prompts
    config = get_ticker_config(ticker)
    company_name = config.get("name", ticker) if config else ticker
    
    # Build enhanced metadata
    aliases_brands_assets = {"aliases": [], "brands": [], "assets": []}
    sector_profile = {"core_inputs": [], "core_channels": [], "core_geos": [], "benchmarks": []}
    
    if config:
        try:
            if config.get("aliases_brands_assets"):
                aliases_brands_assets = json.loads(config["aliases_brands_assets"]) if isinstance(config["aliases_brands_assets"], str) else config["aliases_brands_assets"]
            if config.get("sector_profile"):
                sector_profile = json.loads(config["sector_profile"]) if isinstance(config["sector_profile"], str) else config["sector_profile"]
        except:
            pass
    
    # COMPANY: Process with backfill to limit
    if "company" in articles_by_category and articles_by_category["company"]:
        articles = articles_by_category["company"]
        target = target_limits.get("company", 20)
        LOG.info(f"Starting triage for company: {len(articles)} articles (target: {target})")
        
        try:
            ai_selected = triage_company_articles_full(articles, ticker, company_name, aliases_brands_assets, sector_profile)
        except Exception as e:
            LOG.error(f"AI triage failed for company: {e}")
            ai_selected = []
        
        # Apply tiered backfill to reach target
        selected_results["company"] = _apply_tiered_backfill_to_limits(
            articles, ai_selected, "company", LOW_QUALITY_DOMAINS, target
        )
    
    # INDUSTRY: Process by keyword batches with backfill
    if "industry" in articles_by_category and articles_by_category["industry"]:
        articles = articles_by_category["industry"]
        target = target_limits.get("industry", 15)
        LOG.info(f"Starting triage for industry: {len(articles)} articles (target: {target})")
        
        # Group articles by search_keyword
        articles_by_keyword = {}
        for idx, article in enumerate(articles):
            keyword = article.get("search_keyword", "unknown")
            if keyword not in articles_by_keyword:
                articles_by_keyword[keyword] = []
            articles_by_keyword[keyword].append((idx, article))
        
        # Calculate per-keyword target (distribute total target across keywords)
        num_keywords = len(articles_by_keyword)
        per_keyword_target = max(1, target // num_keywords) if num_keywords > 0 else target
        
        # Process each keyword separately
        all_industry_selected = []
        for keyword, keyword_articles in articles_by_keyword.items():
            if not keyword_articles:
                continue
                
            LOG.info(f"  Processing industry keyword '{keyword}': {len(keyword_articles)} articles (target: {per_keyword_target})")
            
            # Create mini-article list for this keyword
            keyword_article_list = [article for idx, article in keyword_articles]
            
            try:
                peers = config.get("competitors", []) if config else []
                keyword_selected = triage_industry_articles_full(keyword_article_list, ticker, sector_profile, peers)
                
                # Convert mini-indices back to full indices
                for item in keyword_selected:
                    if item["id"] < len(keyword_articles):
                        original_idx = keyword_articles[item["id"]][0]
                        item["id"] = original_idx
                        item["why"] = f"[{keyword}] {item['why']}"
                
                # Apply tiered backfill for this keyword
                keyword_selected_with_backfill = _apply_tiered_backfill_to_limits(
                    keyword_article_list, keyword_selected, "industry", LOW_QUALITY_DOMAINS, per_keyword_target
                )
                
                # Convert back to full article indices for final results
                for item in keyword_selected_with_backfill:
                    if item["id"] < len(keyword_articles):
                        original_idx = keyword_articles[item["id"]][0]
                        item["id"] = original_idx
                        if not item["why"].startswith(f"[{keyword}]"):
                            item["why"] = f"[{keyword}] {item['why']}"
                
                all_industry_selected.extend(keyword_selected_with_backfill)
                LOG.info(f"    Keyword '{keyword}' final selection: {len(keyword_selected_with_backfill)} articles")
                
            except Exception as e:
                LOG.error(f"AI triage failed for industry keyword '{keyword}': {e}")
        
        selected_results["industry"] = all_industry_selected
    
    # COMPETITOR: Process by competitor batches with backfill
    if "competitor" in articles_by_category and articles_by_category["competitor"]:
        articles = articles_by_category["competitor"]
        target = target_limits.get("competitor", 15)
        LOG.info(f"Starting triage for competitor: {len(articles)} articles (target: {target})")
        
        # Group articles by competitor_ticker
        articles_by_competitor = {}
        for idx, article in enumerate(articles):
            search_keyword = article.get("search_keyword", "unknown")
            
            try:
                with db() as conn, conn.cursor() as cur:
                    cur.execute("""
                        SELECT DISTINCT competitor_ticker 
                        FROM source_feed 
                        WHERE search_keyword = %s AND category = 'competitor' AND active = TRUE 
                        AND competitor_ticker IS NOT NULL
                        LIMIT 1
                    """, (search_keyword,))
                    result = cur.fetchone()
                    
                    consolidation_key = result["competitor_ticker"] if result else search_keyword
            except Exception:
                consolidation_key = search_keyword
            
            if consolidation_key not in articles_by_competitor:
                articles_by_competitor[consolidation_key] = []
            articles_by_competitor[consolidation_key].append((idx, article))
        
        # Calculate per-competitor target
        num_competitors = len(articles_by_competitor)
        per_competitor_target = max(1, target // num_competitors) if num_competitors > 0 else target
        
        # Process each competitor separately
        all_competitor_selected = []
        for competitor, competitor_articles in articles_by_competitor.items():
            if not competitor_articles:
                continue
                
            LOG.info(f"  Processing competitor '{competitor}': {len(competitor_articles)} articles (target: {per_competitor_target})")
            
            # Create mini-article list for this competitor
            competitor_article_list = [article for idx, article in competitor_articles]
            
            try:
                peers = config.get("competitors", []) if config else []
                competitor_selected = triage_competitor_articles_full(competitor_article_list, ticker, peers, sector_profile)
                
                # Convert mini-indices back to full indices
                for item in competitor_selected:
                    if item["id"] < len(competitor_articles):
                        original_idx = competitor_articles[item["id"]][0]
                        item["id"] = original_idx
                        item["why"] = f"[{competitor}] {item['why']}"
                
                # Apply tiered backfill for this competitor
                competitor_selected_with_backfill = _apply_tiered_backfill_to_limits(
                    competitor_article_list, competitor_selected, "competitor", LOW_QUALITY_DOMAINS, per_competitor_target
                )
                
                # Convert back to full article indices for final results
                for item in competitor_selected_with_backfill:
                    if item["id"] < len(competitor_articles):
                        original_idx = competitor_articles[item["id"]][0]
                        item["id"] = original_idx
                        if not item["why"].startswith(f"[{competitor}]"):
                            item["why"] = f"[{competitor}] {item['why']}"
                
                all_competitor_selected.extend(competitor_selected_with_backfill)
                LOG.info(f"    Competitor '{competitor}' final selection: {len(competitor_selected_with_backfill)} articles")
                
            except Exception as e:
                LOG.error(f"AI triage failed for competitor '{competitor}': {e}")
        
        selected_results["competitor"] = all_competitor_selected
    
    # Final summary
    total_selected = sum(len(items) for items in selected_results.values())
    total_target = sum(target_limits.values())
    LOG.info(f"=== TRIAGE COMPLETE WITH BACKFILL: {total_selected}/{total_target} articles selected ===")
    
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
    """
    Make structured triage request to OpenAI with enhanced error handling and validation
    """
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Define JSON schema for structured output
        triage_schema = {
            "name": "triage_results",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "selected_ids": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "selected": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "scrape_priority": {"type": "integer", "minimum": 1, "maximum": 5},
                                "likely_repeat": {"type": "boolean"},
                                "repeat_key": {"type": "string"},
                                "why": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            },
                            "required": ["id", "scrape_priority", "likely_repeat", "repeat_key", "why", "confidence"],
                            "additionalProperties": False
                        }
                    },
                    "skipped": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "scrape_priority": {"type": "integer", "minimum": 3, "maximum": 5},
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
        }
        
        data = {
            "model": OPENAI_MODEL,
            "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": triage_schema},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, separators=(",", ":"))}
            ],
            "max_completion_tokens": 5000,
        }
        
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=60)
        
        if response.status_code != 200:
            LOG.error(f"OpenAI triage API error {response.status_code}: {response.text}")
            return []
        
        result = response.json()
        content = result["choices"][0]["message"]["content"] or ""
        
        # Parse JSON - should be clean with structured output
        try:
            triage_result = json.loads(content)
        except json.JSONDecodeError as e:
            LOG.error(f"JSON parsing failed despite structured output: {e}")
            LOG.error(f"Response content: {content[:500]}")
            return []
        
        # Validate response completeness
        total_items = len(payload.get('items', []))
        selected_count = len(triage_result.get("selected_ids", []))
        accounted_items = len(triage_result.get("selected", [])) + len(triage_result.get("skipped", []))
        target_cap = payload.get('target_cap', 0)
        
        # Enhanced validation logging
        if accounted_items != total_items:
            LOG.warning(f"Triage accounting mismatch: {accounted_items} processed vs {total_items} total items")
        
        if selected_count != min(target_cap, total_items):
            LOG.warning(f"Triage selection mismatch: {selected_count} selected vs {min(target_cap, total_items)} expected")
        
        # Validate all selected IDs are valid
        invalid_ids = [id for id in triage_result.get("selected_ids", []) if id >= total_items]
        if invalid_ids:
            LOG.error(f"Invalid selected IDs: {invalid_ids} (max valid ID: {total_items-1})")
            return []
        
        LOG.info(f"Triage completed - Bucket: {payload.get('bucket', 'unknown')}, "
                f"Selected: {selected_count}/{target_cap}, Total processed: {total_items}")
        
        # Convert to expected format for existing code integration
        selected_articles = []
        original_articles = payload.get('items', [])
        
        for selected_item in triage_result.get("selected", []):
            if selected_item["id"] < len(original_articles):
                result_item = {
                    "id": selected_item["id"],
                    "scrape_priority": selected_item["scrape_priority"],
                    "why": selected_item["why"],
                    "confidence": selected_item["confidence"],
                    "likely_repeat": selected_item["likely_repeat"],
                    "repeat_key": selected_item["repeat_key"]
                }
                selected_articles.append(result_item)
            else:
                LOG.warning(f"Selected ID {selected_item['id']} out of range for {len(original_articles)} items")
        
        return selected_articles
        
    except Exception as e:
        LOG.error(f"Triage request failed for {payload.get('bucket', 'unknown')} bucket: {str(e)}")
        return []

# Update the individual triage functions to use the new _full suffix

def triage_company_articles_full(articles: List[Dict], ticker: str, company_name: str, limit: int = 30) -> List[Dict]:
    """
    Enhanced company triage focusing solely on title relevance with mandatory fill-to-cap
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

    payload = {
        "bucket": "company",
        "target_cap": limit,
        "ticker": ticker,
        "company_name": company_name,
        "items": items
    }

    system_prompt = """You are a financial analyst doing PRE-SCRAPE TRIAGE for COMPANY articles.

Focus SOLELY on title content. Ignore domain/source quality. Do not infer from outlet names embedded in titles.

COMPANY RELEVANCE (Title-Only Assessment):

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
* Articles primarily about other companies with target company mentioned in passing
* Historical retrospectives without forward-looking implications
* Generic trading/technical analysis without business fundamentals
* Promotional content or vendor announcements

TITLE SPECIFICITY BOOSTERS (within priority band):
* $ figures, percentages, unit magnitudes, dates/timelines, capacity numbers
* Company name appears close to event verb (same clause)
* Concrete operational details (facility names, product specifics, customer names)
* Geographic specificity (plant locations, market regions, regulatory jurisdictions)
* Quantified business metrics (revenue guidance, margin targets, headcount changes)

FILL POLICY:
* Select exactly target_cap items unless there are fewer than target_cap total items
* Backfill ladder if insufficient High items: High → Medium → Low → (last resort) items matching ticker + relevant business terms
* When backfilling from lower categories, mark reason as "backfill" and cite qualifying title tokens
* Maintain consistent scoring for identical titles regardless of source

REQUIRED RESPONSE FIELDS:
selected_ids: int[]
selected: [{id: int, scrape_priority: int (1-5), why: string ≤180, confidence: float 0.1-1.0, likely_repeat: bool false, repeat_key: ""}]
skipped: [same structure]

For repeat coverage: explain why each repeat adds value based on title content differences, not source.

The response will be automatically formatted as structured JSON with selected_ids, selected, and skipped arrays."""

    return _make_triage_request_full(system_prompt, payload)


def triage_industry_articles_full(articles: List[Dict], ticker: str, industry_keywords: List[str], sector_profile: Dict, limit: int = 30) -> List[Dict]:
    """
    Enhanced industry triage focusing on sector-specific relevance with mandatory fill-to-cap
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

    payload = {
        "bucket": "industry",
        "target_cap": limit,
        "ticker": ticker,
        "industry_keywords": industry_keywords,
        "sector_profile": sector_profile,
        "items": items
    }

    system_prompt = """You are a financial analyst doing PRE-SCRAPE TRIAGE for INDUSTRY articles.

Focus SOLELY on title content. Ignore domain/source quality. Do not infer from outlet names embedded in titles.

INDUSTRY RELEVANCE (Title-Only Assessment):

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
* Local project news without broader sector implications
* Generic sustainability/ESG discussions without regulatory requirements or compliance costs
* Academic research without immediate commercial application or policy relevance
* Vendor announcements without material market impact or adoption timeline
* Consumer preference studies without regulatory or demand shift implications

TITLE SPECIFICITY BOOSTERS (within priority band):
* Percentage changes in key inputs (steel prices up 15%, oil supply down 8%)
* Dollar amounts for industry investments, penalties, or compliance costs
* Effective dates for regulations, policy changes, or standard implementations
* Capacity numbers for new facilities, closures, or production changes
* Geographic scope with specific markets, regions, or jurisdictions mentioned
* Commodity/input price levels with historical context (highest since, lowest in X years)

FILL POLICY:
* Select exactly target_cap items unless there are fewer than target_cap total items
* Backfill ladder if insufficient High items: High → Medium → Low → (last resort) items matching industry_keywords + concrete business terms
* When backfilling, mark reason as "backfill" and cite qualifying title tokens (e.g., "healthcare + regulation", "energy + infrastructure")
* Maintain consistent scoring for identical titles regardless of source

REQUIRED RESPONSE FIELDS:
selected_ids: int[]
selected: [{id: int, scrape_priority: int (1-5), why: string ≤180, confidence: float 0.1-1.0, likely_repeat: bool false, repeat_key: ""}]
skipped: [same structure]

SECTOR CONTEXT PRIORITY:
* core_inputs: Raw materials/components critical to operations (prioritize price/supply/regulatory changes)
* core_channels: Primary markets/customer segments (prioritize demand/access/regulatory changes)  
* core_geos: Key geographic markets (prioritize policy changes, trade issues, regulatory shifts)
* benchmarks: Industry indices/pricing mechanisms (prioritize methodology changes, significant moves)

For repeat coverage: explain why each repeat adds value based on different regulatory/policy aspects mentioned in title.

The response will be automatically formatted as structured JSON with selected_ids, selected, and skipped arrays."""

    return _make_triage_request_full(system_prompt, payload)

def triage_competitor_articles_full(articles: List[Dict], ticker: str, competitors: List[str], sector_profile: Dict, limit: int = 30) -> List[Dict]:
    """
    Enhanced competitor triage focusing on competitive dynamics with mandatory fill-to-cap
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

    payload = {
        "bucket": "competitor",
        "target_cap": limit,
        "ticker": ticker,
        "competitors": competitors,
        "sector_profile": sector_profile,
        "items": items
    }

    system_prompt = """You are a financial analyst doing PRE-SCRAPE TRIAGE for COMPETITOR articles.

Focus SOLELY on title content. Ignore domain/source quality. Do not infer from outlet names embedded in titles.

COMPETITOR RELEVANCE (Title-Only Assessment):

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
* Historical retrospectives without forward-looking competitive implications
* Technical/trading analysis focused on stock price movements without business fundamentals
* Market commentary mentioning competitors in passing without competitive focus
* PR/TAM phrasing: "Announces" (without hard event), "Market Report", "Press Release", "Analysis Report" (unless with hard event verb + quantified impact)
* Articles about non-competing companies or companies outside direct competitive scope

TITLE SPECIFICITY BOOSTERS (within priority band):
* Competitor name positioned close to event verb in title structure
* Percentage changes (pricing actions, capacity changes, market share shifts, margin impacts)
* Dollar amounts (investments, deal values, cost savings, revenue impacts)
* Timelines and effective dates (plant openings, product launches, agreement terms)
* Specific customer, facility, product, or market segment names
* Competitive positioning language (market leader, gaining share, losing ground)

COMPETITIVE IMPACT ASSESSMENT CRITERIA:
* Will this change market share dynamics or competitive positioning?
* Could this affect industry pricing power or cost structures?
* Does this represent a capacity, capability, or geographic advantage/disadvantage?
* Are there specific numbers indicating competitive advantage shifts?
* Would this information influence strategic competitive responses?

FILL POLICY:
* Select exactly target_cap items unless there are fewer than target_cap total items  
* Backfill ladder if insufficient High items: High → Medium → Low → (last resort) items matching competitor names + relevant business terms
* When backfilling, mark reason as "backfill" and cite qualifying elements (e.g., "competitor name + earnings beat", "competitor name + expansion")
* Maintain consistent scoring for identical titles regardless of source

REQUIRED RESPONSE FIELDS:
selected_ids: int[]
selected: [{id: int, scrape_priority: int (1-5), why: string ≤180, confidence: float 0.1-1.0, likely_repeat: bool false, repeat_key: ""}]
skipped: [same structure]

For repeat coverage: explain why each repeat adds different competitive intelligence value based on distinct aspects mentioned in title content.

The response will be automatically formatted as structured JSON with selected_ids, selected, and skipped arrays."""

    return _make_triage_request_full(system_prompt, payload)

def _make_triage_request_full(system_prompt: str, payload: dict) -> List[Dict]:
    """
    Make structured triage request to OpenAI with enhanced error handling and validation
    """
    try:
        response = openai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload)}
            ],
            response_format=TriageResponse,
            temperature=0
        )
        
        if response.choices and response.choices[0].message.parsed:
            triage_result = response.choices[0].message.parsed
            
            # Validate response completeness
            total_items = len(payload.get('items', []))
            selected_count = len(triage_result.selected_ids)
            accounted_items = len(triage_result.selected) + len(triage_result.skipped)
            target_cap = payload.get('target_cap', 0)
            
            # Enhanced validation logging
            if accounted_items != total_items:
                LOG.warning(f"Triage accounting mismatch: {accounted_items} processed vs {total_items} total items")
            
            if selected_count != min(target_cap, total_items):
                LOG.warning(f"Triage selection mismatch: {selected_count} selected vs {min(target_cap, total_items)} expected")
            
            # Validate all selected IDs are valid
            invalid_ids = [id for id in triage_result.selected_ids if id >= total_items]
            if invalid_ids:
                LOG.error(f"Invalid selected IDs: {invalid_ids} (max valid ID: {total_items-1})")
                return []
            
            LOG.info(f"Triage completed - Bucket: {payload.get('bucket', 'unknown')}, "
                    f"Selected: {selected_count}/{target_cap}, Total processed: {total_items}")
            
            # Convert to expected format for existing code integration
            selected_articles = []
            original_articles = payload.get('items', [])
            
            for selected_item in triage_result.selected:
                if selected_item.id < len(original_articles):
                    article_copy = original_articles[selected_item.id].copy()
                    article_copy.update({
                        'ai_triage_priority': selected_item.scrape_priority,
                        'ai_triage_reason': selected_item.why,
                        'ai_triage_confidence': selected_item.confidence,
                        'likely_repeat': selected_item.likely_repeat,
                        'repeat_key': selected_item.repeat_key or ''
                    })
                    selected_articles.append(article_copy)
                else:
                    LOG.warning(f"Selected ID {selected_item.id} out of range for {len(original_articles)} items")
            
            return selected_articles
        
        LOG.error("No valid parsed response from OpenAI triage")
        return []
        
    except Exception as e:
        LOG.error(f"Triage request failed for {payload.get('bucket', 'unknown')} bucket: {str(e)}")
        return []


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

def create_ai_evaluation_text(articles_by_ticker: Dict[str, Dict[str, List[Dict]]]) -> str:
    """Create structured text file for AI evaluation of scoring quality with full metadata and AI summaries"""
    
    text_lines = []
    text_lines.append("STOCK NEWS AGGREGATOR - AI SCORING EVALUATION DATA")
    text_lines.append("=" * 60)
    text_lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    text_lines.append("")
    text_lines.append("PURPOSE: Evaluate quality scoring algorithm performance")
    text_lines.append("INSTRUCTIONS: Analyze patterns in scoring components vs article content")
    text_lines.append("=" * 60)
    text_lines.append("")
    
    article_count = 0
    
    for ticker, categories in articles_by_ticker.items():
        text_lines.append(f"TICKER: {ticker}")
        text_lines.append("-" * 40)
        
        # Add COMPLETE enhanced metadata for each ticker
        config = get_ticker_config(ticker)
        if config:
            text_lines.append("COMPLETE TICKER METADATA:")
            text_lines.append(f"Company Name: {config.get('name', ticker)}")
            text_lines.append(f"Sector: {config.get('sector', 'N/A')}")
            text_lines.append(f"Industry: {config.get('industry', 'N/A')}")
            text_lines.append(f"Sub-Industry: {config.get('sub_industry', 'N/A')}")
            
            # Industry keywords
            if config.get("industry_keywords"):
                text_lines.append(f"Industry Keywords: {', '.join(config['industry_keywords'])}")
            
            # Competitors with full details
            if config.get("competitors"):
                text_lines.append("Competitors:")
                for i, comp in enumerate(config['competitors'], 1):
                    text_lines.append(f"  {i}. {comp}")
            
            # Sector profile with full details
            sector_profile = config.get("sector_profile")
            if sector_profile:
                try:
                    if isinstance(sector_profile, str):
                        sector_data = json.loads(sector_profile)
                    else:
                        sector_data = sector_profile
                    
                    text_lines.append("SECTOR PROFILE:")
                    if sector_data.get("core_inputs"):
                        text_lines.append(f"  Core Inputs: {', '.join(sector_data['core_inputs'])}")
                    if sector_data.get("core_geos"):
                        text_lines.append(f"  Core Geographies: {', '.join(sector_data['core_geos'])}")
                    if sector_data.get("core_channels"):
                        text_lines.append(f"  Core Channels: {', '.join(sector_data['core_channels'])}")
                    if sector_data.get("benchmarks"):
                        text_lines.append(f"  Benchmarks: {', '.join(sector_data['benchmarks'])}")
                except Exception as e:
                    text_lines.append(f"SECTOR PROFILE: Error parsing - {e}")
            
            # Aliases, brands, and assets with full details
            aliases_brands = config.get("aliases_brands_assets")
            if aliases_brands:
                try:
                    if isinstance(aliases_brands, str):
                        alias_data = json.loads(aliases_brands)
                    else:
                        alias_data = aliases_brands
                    
                    text_lines.append("ALIASES, BRANDS & ASSETS:")
                    if alias_data.get("aliases"):
                        text_lines.append(f"  Aliases: {', '.join(alias_data['aliases'])}")
                    if alias_data.get("brands"):
                        text_lines.append(f"  Brands: {', '.join(alias_data['brands'])}")
                    if alias_data.get("assets"):
                        text_lines.append(f"  Key Assets: {', '.join(alias_data['assets'])}")
                except Exception as e:
                    text_lines.append(f"ALIASES/BRANDS: Error parsing - {e}")
            
            text_lines.append("")
        
        for category, articles in categories.items():
            if not articles:
                continue
                
            text_lines.append(f"CATEGORY: {category.upper()}")
            text_lines.append("")
            
            for article in articles:
                article_count += 1
                
                text_lines.append(f"ARTICLE {article_count}:")
                text_lines.append(f"Type: {category}")
                text_lines.append(f"Ticker: {ticker}")
                text_lines.append(f"Title: {article.get('title', 'No Title')}")
                text_lines.append(f"Domain: {article.get('domain', 'unknown')}")
                text_lines.append(f"URL: {article.get('resolved_url') or article.get('url', 'N/A')}")
                
                if article.get('published_at'):
                    text_lines.append(f"Published: {article['published_at']}")
                
                if article.get('search_keyword'):
                    text_lines.append(f"Search Keyword: {article['search_keyword']}")
                
                if article.get('competitor_ticker'):
                    text_lines.append(f"Competitor Ticker: {article['competitor_ticker']}")
                
                text_lines.append("")
                
                # AI Analysis section with ALL components
                text_lines.append("AI ANALYSIS:")
                quality_score = article.get('quality_score', 0)
                text_lines.append(f"Final Score: {quality_score:.1f}")
                
                # Show ALL scoring components for verification
                if article.get('source_tier'):
                    text_lines.append(f"Source Tier: {article['source_tier']} (Domain: {article.get('domain', 'unknown')})")
                if article.get('event_multiplier'):
                    text_lines.append(f"Event Multiplier: {article['event_multiplier']} - {article.get('event_multiplier_reason', '')}")
                if article.get('relevance_boost'):
                    text_lines.append(f"Relevance Boost: {article['relevance_boost']} - {article.get('relevance_boost_reason', '')}")
                if article.get('numeric_bonus'):
                    text_lines.append(f"Numeric Bonus: {article['numeric_bonus']}")
                if article.get('penalty_multiplier'):
                    text_lines.append(f"Penalty Multiplier: {article['penalty_multiplier']} - {article.get('penalty_reason', '')}")
                
                # Calculation verification
                if all(article.get(field) is not None for field in ['source_tier', 'event_multiplier', 'relevance_boost', 'penalty_multiplier']):
                    calculated = ((100 * article['source_tier'] * article['event_multiplier'] * article['relevance_boost']) * 
                                article['penalty_multiplier'] + article.get('numeric_bonus', 0))
                    text_lines.append(f"CALCULATION CHECK: (100 × {article['source_tier']} × {article['event_multiplier']} × {article['relevance_boost']}) × {article['penalty_multiplier']} + {article.get('numeric_bonus', 0)} = {calculated:.1f}")
                
                ai_impact = article.get('ai_impact', 'N/A')
                ai_reasoning = article.get('ai_reasoning', 'N/A')
                text_lines.append(f"Impact: {ai_impact}")
                text_lines.append(f"Reasoning: {ai_reasoning}")
                
                # Triage information
                if article.get('ai_triage_selected'):
                    text_lines.append(f"AI Triage: SELECTED (Priority: {article.get('triage_priority', 'N/A')})")
                    if article.get('triage_reasoning'):
                        text_lines.append(f"Triage Reasoning: {article['triage_reasoning']}")
                else:
                    text_lines.append(f"AI Triage: NOT SELECTED")
                
                text_lines.append("")
                
                # AI SUMMARY (if available)
                if article.get('ai_summary'):
                    text_lines.append("AI HEDGE FUND SUMMARY:")
                    text_lines.append(article['ai_summary'])
                    text_lines.append("")
                
                # Original description
                if article.get('description'):
                    text_lines.append("ORIGINAL DESCRIPTION:")
                    text_lines.append(article['description'])
                    text_lines.append("")
                
                # Scraped content
                if article.get('scraped_content'):
                    text_lines.append("SCRAPED CONTENT:")
                    text_lines.append(article['scraped_content'])
                elif article.get('scraping_error'):
                    text_lines.append("SCRAPING FAILED:")
                    text_lines.append(article['scraping_error'])
                else:
                    text_lines.append("SCRAPED CONTENT: Not attempted or not available")
                
                text_lines.append("")
                text_lines.append("-" * 80)
                text_lines.append("")
    
    text_lines.append(f"\nTOTAL ARTICLES: {article_count}")
    text_lines.append("END OF EVALUATION DATA")
    
    return "\n".join(text_lines)

def create_triage_evaluation_text(articles_by_ticker: Dict[str, Dict[str, List[Dict]]], triage_results: Dict[str, Dict[str, List[Dict]]], ticker_metadata_cache: Dict) -> str:
    """Create structured text file for AI triage evaluation and improvement"""
    
    text_lines = []
    text_lines.append("STOCK NEWS AGGREGATOR - AI TRIAGE EVALUATION DATA")
    text_lines.append("=" * 60)
    text_lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    text_lines.append("")
    text_lines.append("PURPOSE: Evaluate triage algorithm performance and improve selection logic")
    text_lines.append("INSTRUCTIONS: Analyze triage reasoning vs article titles for prompt optimization")
    text_lines.append("=" * 60)
    text_lines.append("")
    
    article_count = 0
    
    for ticker, categories in articles_by_ticker.items():
        text_lines.append(f"TICKER: {ticker}")
        text_lines.append("-" * 40)
        
        # Add enhanced metadata for each ticker
        if ticker in ticker_metadata_cache:
            metadata = ticker_metadata_cache[ticker]
            text_lines.append("ENHANCED METADATA:")
            text_lines.append(f"Company Name: {metadata.get('company_name', ticker)}")
            text_lines.append(f"Sector: {metadata.get('sector', 'N/A')}")
            
            # Industry keywords
            if metadata.get("industry_keywords"):
                text_lines.append(f"Industry Keywords: {', '.join(metadata['industry_keywords'])}")
            
            # Competitors
            if metadata.get("competitors"):
                comp_names = [comp["name"] if isinstance(comp, dict) else comp for comp in metadata['competitors']]
                text_lines.append(f"Competitors: {', '.join(comp_names)}")
            
            text_lines.append("")
        
        triage_data = triage_results.get(ticker, {})
        
        for category, articles in categories.items():
            if not articles:
                continue
                
            text_lines.append(f"CATEGORY: {category.upper()}")
            text_lines.append("")
            
            # Get triage results for this category
            category_triage = triage_data.get(category, [])
            selected_article_data = {item["id"]: item for item in category_triage}
            
            for idx, article in enumerate(articles):
                article_count += 1
                
                text_lines.append(f"ARTICLE {article_count}:")
                text_lines.append(f"Type: {category}")
                text_lines.append(f"Ticker: {ticker}")
                text_lines.append(f"Index: {idx}")
                text_lines.append(f"Title: {article.get('title', 'No Title')}")
                text_lines.append(f"Domain: {article.get('domain', 'unknown')}")
                
                # Check domain tier
                domain = normalize_domain(article.get("domain", ""))
                source_tier = _get_domain_tier(domain, article.get("title", ""), article.get("description", ""))
                text_lines.append(f"Source Tier: {source_tier}")
                text_lines.append(f"Quality Domain: {'YES' if domain in QUALITY_DOMAINS else 'NO'}")
                
                if article.get('published_at'):
                    text_lines.append(f"Published: {article['published_at']}")
                
                if article.get('search_keyword'):
                    text_lines.append(f"Search Keyword: {article['search_keyword']}")
                
                text_lines.append("")
                
                # AI Triage Results
                text_lines.append("AI TRIAGE RESULTS:")
                if idx in selected_article_data:
                    triage_item = selected_article_data[idx]
                    priority = triage_item.get("scrape_priority", 5)
                    confidence = triage_item.get("confidence", 0.0)
                    reason = triage_item.get("why", "")
                    
                    text_lines.append(f"SELECTED: YES")
                    text_lines.append(f"Priority: P{priority}")
                    text_lines.append(f"Confidence: {confidence:.2f}")
                    text_lines.append(f"AI Reasoning: {reason}")
                else:
                    text_lines.append(f"SELECTED: NO")
                    text_lines.append(f"AI Reasoning: Not selected by triage algorithm")
                
                # Final selection status
                will_scrape = (idx in selected_article_data) or (domain in QUALITY_DOMAINS)
                text_lines.append(f"FINAL STATUS: {'WILL SCRAPE' if will_scrape else 'SKIPPED'}")
                
                text_lines.append("")
                
                # Original description for context
                if article.get('description'):
                    text_lines.append("ORIGINAL DESCRIPTION:")
                    text_lines.append(article['description'])
                else:
                    text_lines.append("ORIGINAL DESCRIPTION: Not available")
                
                text_lines.append("")
                text_lines.append("-" * 80)
                text_lines.append("")
    
    text_lines.append(f"\nTOTAL ARTICLES EVALUATED: {article_count}")
    text_lines.append("END OF TRIAGE EVALUATION DATA")
    
    return "\n".join(text_lines)

def get_competitor_display_name(search_keyword: str, competitor_ticker: str = None) -> str:
    """
    Get standardized competitor display name using database lookup
    Priority: competitor_ticker -> search_keyword -> fallback
    """
    
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
    
    # Fallback to search_keyword (should be company name for Google feeds)
    if search_keyword and not search_keyword.isupper():  # Likely a company name, not ticker
        return search_keyword
        
    # Final fallback
    return competitor_ticker or search_keyword or "Unknown Competitor"

def _ai_quality_score_company_components(title: str, domain: str, ticker: str, description: str = "", keywords: List[str] = None) -> Tuple[float, str, str, Dict]:
    """Enhanced AI-powered component extraction for company articles"""
    
    source_tier = _get_domain_tier(domain, title, description)
    desc_snippet = description[:500] if description and description.lower() != title.lower().strip() else ""
    
    # Get enhanced metadata for this ticker if available
    config = get_ticker_config(ticker)
    company_name = config.get("name", ticker) if config else ticker
    
    # Build brand alias regex if we have enhanced metadata stored
    brand_alias_regex = ""
    # This would be populated from the enhanced metadata if stored
    
    schema = {
        "name": "company_news_components",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "event_multiplier": {"type": "number", "minimum": 0.5, "maximum": 2.0},
                "event_multiplier_reason": {"type": "string"},
                "relevance_boost": {"type": "number", "minimum": 1.0, "maximum": 1.3},
                "relevance_boost_reason": {"type": "string"},
                "numeric_bonus": {"type": "number", "minimum": 0.0, "maximum": 0.4},
                "penalty_multiplier": {"type": "number", "minimum": 0.5, "maximum": 1.0},
                "penalty_reason": {"type": "string"},
                "impact_on_main": {"type": "string", "enum": ["Positive", "Negative", "Mixed", "Neutral"]},
                "reason_short": {"type": "string"},
                "debug_tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["event_multiplier", "event_multiplier_reason", "relevance_boost", "relevance_boost_reason", "numeric_bonus", "penalty_multiplier", "penalty_reason", "impact_on_main", "reason_short", "debug_tags"],
            "additionalProperties": False
        }
    }
    
    system_prompt = f"""You are a hedge-fund news scorer for COMPANY context affecting {company_name}. Base ALL judgments ONLY on title and description_snippet. Do NOT compute a final score. Return STRICT JSON exactly as specified.

INPUTS PROVIDED:
- bucket = "company_news"
- company_ticker = "{ticker}"
- company_name = "{company_name}"
- title = "{title}"
- description_snippet = "{desc_snippet}"
- source_tier = {source_tier} (already calculated)

GENERAL RELEVANCE FACTORS (APPLY IN ORDER; USE FIRST MATCH):
- 1.3 Direct company match: title/desc matches ticker/company_name OR a named asset/subsidiary unique to the firm.
- 1.2 First-order exposure: named counterparty contract/customer/supplier that binds economics to the firm (with $$, units, volumes, pricing, or %, OR a named regulator decision on the firm).
- 1.1 Second-order but specific: sector/policy/benchmark/geography explicitly tied to the firm's core inputs/channels/geos.
- 1.0 Vague/off-sector: generic market commentary, far geographies, or missing a clear tie to firm economics.

EVENT MULTIPLIER (PICK ONE):
- 2.0 Hard corporate actions: M&A (acquires/divests/spin), bankruptcy/Chapter 11, delist/halt/recall, large asset closure, definitive regulatory fines/settlements with $.
- 1.8 Capital actions: major buyback start/upsizing, dividend initiation/meaningful change, debt/equity issuance, rating change with outlook.
- 1.6 Binding regulatory/procedural decisions affecting the company (approval/denial, tariff specific to the firm).
- 1.5 Signed commercial contracts/backlog/LOIs with named counterparties and $$ or units.
- 1.4 Earnings/guidance (scheduled reports, pre-announcements) with numbers.
- 1.2-1.1 Management changes, product launches without numbers, notable partnerships w/o $$.
- 1.0 Miscellaneous updates with unclear financial impact.
- 0.9 Institutional flows (13F, small stake changes) with no activism/merger intent.
- 0.6 Price-move explainers/recaps/opinion/education; previews with no new facts.
- 0.5 Routine PR (awards, CSR, conference attendance) with no economic detail.

NUMERIC BONUS (COMPANY) --- CAP = +0.40:
- +0.20 Per clearly material FINANCIAL figure (EPS, revenue, margin, FCF, capex, buyback $, guidance delta, unit/capacity adds/cuts) that is NEW in this item (max 2 such adds).
- +0.10 Additional supporting numeric detail (mix, ASP, utilization, backlog change) that ties to economics.
- +0.00 Trivial counts (e.g., "added 879 shares"), headcounts, vague % with no base.

PENALTIES (COMPANY):
- 0.5 PR/sponsored/marketing tone or PR domains; "market size will reach ... CAGR ..."
- 0.6 Question/listicle/prediction ("Top X...", "Should you...", "What to watch")
- 1.0 Otherwise

IMPACT DIRECTION (COMPANY):
- Positive: beats/raises; upgrades; accretive deals; favorable rulings; capacity adds with demand.
- Negative: misses/cuts; downgrades; dilutive raises; adverse rulings; closures/recalls.
- Mixed: e.g., guide up YoY but down QoQ; cost up but price up; EPS beat on lower quality.
- Neutral: minor 10b5-1 sales (<5% holdings), routine housekeeping, ceremonial PR.

DEFENSIVE CLAMPS (ALWAYS APPLY):
- If title indicates price-move recap without concrete action → event=0.6; numeric_bonus=0.
- If 13F/holding tweaks only → event=0.9; numeric_bonus=0.
- If "Rule 10b5-1" and <5% of reported holdings → event ≤1.1; impact=Neutral.
- If data insufficient (no event keywords & no numbers) → event ≤1.1; numeric_bonus=0.

RETURN STRICT JSON ONLY."""

    user_payload = {
        "bucket": "company_news",
        "company_ticker": ticker,
        "company_name": company_name,
        "title": title,
        "description_snippet": desc_snippet,
        "source_tier": source_tier
    }
    
    return _make_ai_component_request(system_prompt, user_payload, schema, source_tier)

def _ai_quality_score_industry_components(title: str, domain: str, ticker: str, description: str = "", keywords: List[str] = None) -> Tuple[float, str, str, Dict]:
    """Enhanced AI-powered component extraction for industry articles"""
    
    source_tier = _get_domain_tier(domain, title, description)
    desc_snippet = description[:500] if description and description.lower() != title.lower().strip() else ""
    
    # Get enhanced metadata for this ticker if available
    config = get_ticker_config(ticker)
    company_name = config.get("name", ticker) if config else ticker
    
    # Build sector profile if we have enhanced metadata
    sector_profile = {}
    # This would be populated from enhanced metadata if stored
    
    schema = {
        "name": "industry_market_components",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "event_multiplier": {"type": "number", "minimum": 0.6, "maximum": 1.6},
                "event_multiplier_reason": {"type": "string"},
                "relevance_boost": {"type": "number", "minimum": 1.0, "maximum": 1.2},
                "relevance_boost_reason": {"type": "string"},
                "numeric_bonus": {"type": "number", "minimum": 0.0, "maximum": 0.3},
                "penalty_multiplier": {"type": "number", "minimum": 0.5, "maximum": 1.0},
                "penalty_reason": {"type": "string"},
                "impact_on_main": {"type": "string", "enum": ["Positive", "Negative", "Mixed", "Neutral"]},
                "reason_short": {"type": "string"},
                "debug_tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["event_multiplier", "event_multiplier_reason", "relevance_boost", "relevance_boost_reason", "numeric_bonus", "penalty_multiplier", "penalty_reason", "impact_on_main", "reason_short", "debug_tags"],
            "additionalProperties": False
        }
    }
    
    system_prompt = f"""You are a hedge-fund news scorer for INDUSTRY context affecting {company_name}. Base ALL judgments ONLY on title and description_snippet. Do NOT compute a final score. Return STRICT JSON exactly as specified.

INPUTS PROVIDED:
- bucket = "industry_market"
- target_company_ticker = "{ticker}"
- company_name = "{company_name}"
- title = "{title}"
- description_snippet = "{desc_snippet}"
- industry_keywords = {keywords or []}
- sector_profile = {sector_profile}

GENERAL RELEVANCE FACTORS (INDUSTRY):
- 1.2 Article explicitly mentions any of sector_profile.core_inputs / benchmarks / core_geos / core_channels in a way that clearly affects sector economics (cost, price, availability, demand).
- 1.1 Sector-matching but adjacent geo/channel (e.g., EU policy for US-centric firm) or early-stage proposals with plausible path to effect.
- 1.0 Generic macro/sustainability platitudes or unrelated sub-sector news.

EVENT MULTIPLIER (INDUSTRY):
- 1.6 Enacted policy/regulation with effective dates/geos (tariffs, standards, reimbursement, safety/emissions rules) that shape sector economics.
- 1.5 Input/commodity/benchmark supply-demand or price shocks tied to core_inputs/benchmarks (e.g., ore/energy shortage, index spike), or infrastructure funding passed.
- 1.4 Ecosystem capacity/capex/standardization (new mills/lines, grid or logistics expansions, standards adoption) with implications for cost/throughput.
- 1.1 Reputable research/indices (PMI sub-indices, monthly production, government stats) directly relevant.
- 1.0 Sector commentary without new data.
- 0.6 Market-size/vendor marketing/CSR with no quantified economics.

NUMERIC BONUS (INDUSTRY) --- CAP = +0.30:
- +0.15 Clear magnitude on policy/price/capacity (e.g., tariff %; index +X%; capacity +Y units; capex $ with commissioning date).
- +0.10 Secondary corroboration (inventory days, utilization, lead times).
- +0.05 Concrete effective date/time-to-impact (e.g., "effective Jan 1, 2026").
- +0.00 Vague "billions by 2030" TAM claims without sources/method.

PENALTIES (INDUSTRY):
- 0.5 Vendor PR/marketing puff or TAM-CAGR boilerplate.
- 0.6 Opinion/listicle/forecast pieces without cited data.
- 1.0 Otherwise.

IMPACT DIRECTION (ON TARGET COMPANY):
- Positive: inputs down; favorable tariffs/subsidies; demand-side stimulus in core channels; supportive benchmark moves.
- Negative: inputs up/shortages; adverse tariffs/quotas; regulatory burdens increasing costs; unfavorable benchmark shifts.
- Mixed: opposing forces (e.g., input up but prices up).
- Neutral: data points not yet directional.

DEFENSIVE CLAMPS:
- If core_inputs/benchmarks/geos/channels not mentioned and no numbers → relevance=1.0; event ≤1.0; numeric_bonus=0.
- Title/desc only: cap numeric_bonus at +0.20; require explicit units/%/$ to award any bonus.

RETURN STRICT JSON ONLY."""

    user_payload = {
        "bucket": "industry_market",
        "target_company_ticker": ticker,
        "title": title,
        "description_snippet": desc_snippet,
        "industry_keywords": keywords or [],
        "sector_profile": sector_profile
    }
    
    return _make_ai_component_request(system_prompt, user_payload, schema, source_tier)

def _ai_quality_score_competitor_components(title: str, domain: str, ticker: str, description: str = "", keywords: List[str] = None) -> Tuple[float, str, str, Dict]:
    """Enhanced AI-powered component extraction for competitor articles"""
    
    source_tier = _get_domain_tier(domain, title, description)
    desc_snippet = description[:500] if description and description.lower() != title.lower().strip() else ""
    
    # Get enhanced metadata for this ticker if available
    config = get_ticker_config(ticker)
    company_name = config.get("name", ticker) if config else ticker
    
    # Build competitor whitelist from stored metadata
    competitor_whitelist = []
    if config and config.get("competitors"):
        for comp_str in config["competitors"]:
            # Extract ticker from "Name (TICKER)" format
            match = re.search(r'\(([A-Z]{1,5})\)', comp_str)
            if match:
                competitor_whitelist.append(match.group(1))
    
    schema = {
        "name": "competitor_intel_components",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "event_multiplier": {"type": "number", "minimum": 0.5, "maximum": 1.7},
                "event_multiplier_reason": {"type": "string"},
                "relevance_boost": {"type": "number", "minimum": 1.0, "maximum": 1.2},
                "relevance_boost_reason": {"type": "string"},
                "numeric_bonus": {"type": "number", "minimum": 0.0, "maximum": 0.35},
                "penalty_multiplier": {"type": "number", "minimum": 0.5, "maximum": 1.0},
                "penalty_reason": {"type": "string"},
                "impact_on_main": {"type": "string", "enum": ["Positive", "Negative", "Mixed", "Neutral"]},
                "reason_short": {"type": "string"},
                "debug_tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["event_multiplier", "event_multiplier_reason", "relevance_boost", "relevance_boost_reason", "numeric_bonus", "penalty_multiplier", "penalty_reason", "impact_on_main", "reason_short", "debug_tags"],
            "additionalProperties": False
        }
    }
    
    system_prompt = f"""You are a hedge-fund news scorer for COMPETITIVE intelligence affecting {company_name}. Base ALL judgments ONLY on title and description_snippet. Do NOT compute a final score. Return STRICT JSON exactly as specified.

INPUTS PROVIDED:
- bucket = "competitor_intel"
- target_company_ticker = "{ticker}"
- company_name = "{company_name}"
- title = "{title}"
- description_snippet = "{desc_snippet}"
- competitor_whitelist = {competitor_whitelist}

GENERAL RELEVANCE FACTORS (COMPETITOR):
- 1.2 The SUBJECT of the article is in competitor_whitelist AND explicit competitive implications are stated (price/capacity/share/customer win/loss).
- 1.1 Close adjacent peer (same sub-industry) with explicit implications, but not in whitelist.
- 1.0 Vague peer reference, editorial/preview, or off-universe.

EVENT MULTIPLIER (COMPETITOR):
- 1.7 Rival hard events likely to shift share/price: M&A, major capacity adds/cuts, price hikes/cuts, plant shutdown/curtailment, large customer win/loss with $$.
- 1.6 Capital structure stress/advantage: punitive refi, large equity raise, rating cut to HY, covenant issues; or major cost advantage emergence.
- 1.4 Rival guidance with numbers, product/pricing launch with explicit $$/units, meaningful opex/capex changes.
- 1.1 Management changes or announcements without tangible economics.
- 1.0 Miscellaneous/unclear impact on competitive landscape.
- 0.9 Holdings/13F flows about the peer (non-activist).
- 0.6 Opinion/preview/"why it moved" without new facts.
- 0.5 Routine PR by the peer with no economics.

NUMERIC BONUS (COMPETITOR) --- CAP = +0.35:
- +0.20 Concrete economics: announced capacity (units/Mt/MW), explicit price change %, contract value, guide deltas that plausibly affect the target company's market.
- +0.10 Secondary figures (utilization, mix, input cost changes) that inform rivalry economics.
- +0.00 Trivial numbers (share counts, social metrics).

PENALTIES (COMPETITOR):
- 0.5 Press release/sponsored content or vendor marketing.
- 0.6 Question/listicle/preview framing.
- 1.0 Otherwise.

IMPACT DIRECTION (ON TARGET COMPANY):
- Positive: rival capacity cuts/outages; rival distress; rival price increases; loss of a key customer by the rival that the target might win.
- Negative: rival capacity adds; under-cut pricing; rival wins a key customer from target; rival cost breakthrough.
- Mixed: simultaneous adds and shutdowns, or price up but input costs fall for both.
- Neutral: editorial notes, small sponsorships, or ambiguous previews.

DEFENSIVE CLAMPS:
- If the subject company is NOT a plausible peer (fails whitelist and not same sub-industry) → relevance=1.0; event ≤1.1; numeric_bonus=0.
- If only title/desc and no concrete economics → event ≤1.1; numeric_bonus=0.

RETURN STRICT JSON ONLY."""

    user_payload = {
        "bucket": "competitor_intel",
        "target_company_ticker": ticker,
        "title": title,
        "description_snippet": desc_snippet,
        "competitor_whitelist": competitor_whitelist
    }
    
    return _make_ai_component_request(system_prompt, user_payload, schema, source_tier)

def calculate_score_from_components(components: Dict) -> float:
    """
    Calculate quality score from AI-provided components using our own formula
    Formula: (100 × source_tier × event_multiplier × relevance_boost) × penalty_multiplier + numeric_bonus
    """
    try:
        source_tier = float(components.get('source_tier', 0.5))
        event_multiplier = float(components.get('event_multiplier', 1.0))
        relevance_boost = float(components.get('relevance_boost', 1.0))
        numeric_bonus = float(components.get('numeric_bonus', 0.0))
        penalty_multiplier = float(components.get('penalty_multiplier', 1.0))
        
        # Validate ranges
        source_tier = max(0.1, min(1.0, source_tier))
        event_multiplier = max(0.1, min(2.0, event_multiplier))
        relevance_boost = max(0.5, min(1.5, relevance_boost))
        numeric_bonus = max(0.0, min(1.0, numeric_bonus))
        penalty_multiplier = max(0.1, min(1.0, penalty_multiplier))
        
        # Calculate score using exact formula - penalty applies to base score, then add numeric bonus
        base_score = 100 * source_tier * event_multiplier * relevance_boost
        penalized_score = base_score * penalty_multiplier
        final_score_raw = penalized_score + numeric_bonus
        
        # Clamp to valid range
        final_score = max(0.0, min(100.0, final_score_raw))
        
        LOG.info(f"COMPONENT CALCULATION: (100 × {source_tier} × {event_multiplier} × {relevance_boost}) × {penalty_multiplier} + {numeric_bonus} = {final_score:.1f}")
        
        return final_score
        
    except (ValueError, TypeError) as e:
        LOG.error(f"Invalid components for score calculation: {e}, using fallback score 50.0")
        return 50.0

def _make_ai_component_request(system_prompt: str, user_payload: Dict, schema: Dict, source_tier: float) -> Tuple[float, str, str, Dict]:
    """Make AI request for components only, calculate score ourselves"""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": OPENAI_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)}
        ],
        "response_format": {"type": "json_schema", "json_schema": schema},
        "max_completion_tokens": 300
    }
    
    response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=20)
    
    if response.status_code != 200:
        LOG.warning(f"OpenAI API error {response.status_code}: {response.text[:200]}")
        raise Exception(f"API error: {response.status_code}")
    
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    
    # Extract components with reasons
    components = {
        'source_tier': source_tier,  # Use our calculated source tier
        'event_multiplier': parsed.get('event_multiplier', 1.0),
        'event_multiplier_reason': parsed.get('event_multiplier_reason', ''),
        'relevance_boost': parsed.get('relevance_boost', 1.0),
        'relevance_boost_reason': parsed.get('relevance_boost_reason', ''),
        'numeric_bonus': parsed.get('numeric_bonus', 0.0),
        'penalty_multiplier': parsed.get('penalty_multiplier', 1.0),
        'penalty_reason': parsed.get('penalty_reason', '')
    }
    
    # Calculate score using our own formula
    calculated_score = calculate_score_from_components(components)
    
    impact = parsed.get("impact_on_main", "Unclear")
    reason = parsed.get("reason_short", "")
    
    # Log the components for transparency
    LOG.info(f"AI COMPONENTS:")
    LOG.info(f"  Source tier: {source_tier} (calculated)")
    LOG.info(f"  Event multiplier: {components['event_multiplier']} - {components['event_multiplier_reason']}")
    LOG.info(f"  Relevance boost: {components['relevance_boost']} - {components['relevance_boost_reason']}")
    LOG.info(f"  Numeric bonus: {components['numeric_bonus']}")
    LOG.info(f"  Penalty: {components['penalty_multiplier']} - {components['penalty_reason']}")
    LOG.info(f"  OUR CALCULATED SCORE: {calculated_score:.1f}")
    LOG.info(f"  Impact: {impact} | Reason: {reason}")
    
    return calculated_score, impact, reason, components

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
            'finance.yahoo.com': 'Yahoo Finance',
            'ca.finance.yahoo.com': 'Yahoo Finance',
            'uk.finance.yahoo.com': 'Yahoo Finance',
            'seekingalpha.com': 'Seeking Alpha',
            'fool.com': 'The Motley Fool',
            'tipranks.com': 'TipRanks',
            'benzinga.com': 'Benzinga',
            'investors.com': "Investor's Business Daily",
            'barrons.com': "Barron's",
            'ft.com': 'Financial Times'
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
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20
            }
            
            response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=15)
            if response.status_code == 200:
                result = response.json()
                domain = result["choices"][0]["message"]["content"].strip().lower()
                
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
            elif "finance.yahoo.com" in url:
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
        
        # Fall back to existing method (direct resolution + title extraction)
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
        except:
            pass
        
        # Existing title extraction fallback
        if title and not contains_non_latin_script(title):
            clean_title, source = extract_source_from_title_smart(title)
            if source and not self._is_spam_source(source):
                resolved_domain = self._resolve_publication_to_domain(source)
                if resolved_domain:
                    LOG.info(f"Title resolution: {source} -> {resolved_domain}")
                    return url, resolved_domain, None
                else:
                    LOG.warning(f"Could not resolve publication '{source}' to domain")
                    return url, "google-news-unresolved", None
        
        return url, "google-news-unresolved", None
    
    def _handle_yahoo_finance(self, url):
        """Handle Yahoo Finance URL resolution"""
        original_source = extract_yahoo_finance_source_optimized(url)
        if original_source:
            domain = normalize_domain(urlparse(original_source).netloc.lower())
            if not self._is_spam_domain(domain):
                return original_source, domain, url
        
        return url, normalize_domain(urlparse(url).netloc.lower()), None
    
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
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 30
            }
            
            response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=15)
            if response.status_code == 200:
                result = response.json()
                name = result["choices"][0]["message"]["content"].strip()
                return name if 2 < len(name) < 100 else None
        except:
            pass
        return None

# Create global instance
domain_resolver = DomainResolver()

class FeedManager:
    @staticmethod
    def create_feeds_for_ticker(ticker: str, metadata: Dict) -> List[Dict]:
        """Create feeds only if under the limits - FIXED competitor counting logic"""
        feeds = []
        company_name = metadata.get("company_name", ticker)
        
        LOG.info(f"CREATING FEEDS for {ticker} ({company_name}):")
        
        # Check existing feed counts by unique competitor (not by feed count)
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT category, COUNT(*) as count,
                       COUNT(DISTINCT COALESCE(competitor_ticker, search_keyword)) as unique_competitors
                FROM source_feed 
                WHERE ticker = %s AND active = TRUE
                GROUP BY category
            """, (ticker,))
            
            existing_data = {row["category"]: row for row in cur.fetchall()}
            
            # Extract counts
            existing_company_count = existing_data.get('company', {}).get('count', 0)
            existing_industry_count = existing_data.get('industry', {}).get('count', 0)
            existing_competitor_entities = existing_data.get('competitor', {}).get('unique_competitors', 0)  # Count unique competitors, not feeds
            
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
                if isinstance(comp, dict):
                    comp_name = comp.get('name', '')
                    comp_ticker = comp.get('ticker')
                    LOG.info(f"DEBUG: Competitor details - Name: '{comp_name}', Ticker: '{comp_ticker}'")
                    
                    if comp_ticker and comp_ticker.upper() != ticker.upper() and comp_name:
                        LOG.info(f"DEBUG: Creating feeds for competitor {comp_name} ({comp_ticker})")
                        # BOTH feeds now use company name as search_keyword for consistency
                        comp_feeds = [
                            {
                                "url": f"https://news.google.com/rss/search?q=\"{requests.utils.quote(comp_name)}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
                                "name": f"Competitor: {comp_name}",
                                "category": "competitor",
                                "search_keyword": comp_name,  # Always use company name
                                "competitor_ticker": comp_ticker
                            },
                            {
                                "url": f"https://finance.yahoo.com/rss/headline?s={comp_ticker}",
                                "name": f"Yahoo Competitor: {comp_name} ({comp_ticker})",
                                "category": "competitor",
                                "search_keyword": comp_name,  # Changed: use company name instead of ticker
                                "competitor_ticker": comp_ticker
                            }
                        ]
                        feeds.extend(comp_feeds)
                        LOG.info(f"    COMPETITOR: {comp_name} ({comp_ticker}) - 2 feeds (counts as 1 entity)")
                    else:
                        LOG.info(f"DEBUG: Skipping competitor - ticker:{comp_ticker}, name:'{comp_name}', ticker_check:{comp_ticker and comp_ticker.upper() != ticker.upper() if comp_ticker else 'No ticker'}")
                elif isinstance(comp, str):
                    LOG.info(f"DEBUG: Competitor is string format: {comp}")
                    # Try to parse "Name (TICKER)" format
                    match = re.search(r'^(.+?)\s*\(([A-Z]{1,5})\)$', comp)
                    if match:
                        comp_name = match.group(1).strip()
                        comp_ticker = match.group(2)
                        LOG.info(f"DEBUG: Parsed competitor - Name: '{comp_name}', Ticker: '{comp_ticker}'")
                        
                        if comp_ticker and comp_ticker.upper() != ticker.upper() and comp_name:
                            LOG.info(f"DEBUG: Creating feeds for parsed competitor {comp_name} ({comp_ticker})")
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
                            LOG.info(f"DEBUG: Skipping parsed competitor - ticker:{comp_ticker}, name:'{comp_name}'")
                    else:
                        LOG.info(f"DEBUG: Could not parse competitor string: {comp}")
                else:
                    LOG.info(f"DEBUG: Competitor not a dict or string: {comp}")
        else:
            LOG.info(f"  COMPETITOR ENTITIES: Skipping - already at limit (3/3 unique competitors)")
    
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
        ai_metadata = generate_ticker_metadata_with_ai(ticker)
        
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

def generate_ticker_metadata_with_ai(ticker, company_name=None):
    """
    Generate comprehensive ticker metadata using OpenAI with improved validation
    """
    if company_name is None:
        company_name = ticker  # Use ticker as fallback
    system_prompt = """You are a financial analyst creating metadata for a hedge fund's stock monitoring system. Generate precise, actionable metadata that will be used for news article filtering and triage.

CRITICAL REQUIREMENTS:
- All competitors must be currently publicly traded with valid ticker symbols
- Industry keywords must be SPECIFIC enough to avoid false positives in news filtering
- Benchmarks must be sector-specific, not generic market indices
- All information must be factually accurate as of 2024

INDUSTRY KEYWORDS (exactly 3):
- Must be SPECIFIC to the company's primary business
- Avoid generic terms like "Technology", "Healthcare", "Energy", "Oil", "Services"
- Use compound terms or specific product categories
- Examples: "Smartphone Manufacturing" not "Technology", "Upstream Oil Production" not "Oil"
- Test: Would this keyword appear in articles about direct competitors but NOT unrelated companies?

COMPETITORS (exactly 3):
- Must be direct business competitors, not just same-sector companies
- Must be currently publicly traded (check acquisition status)
- Format: "Company Name (TICKER)" - verify ticker is correct and current
- Exclude: Private companies, subsidiaries, companies acquired in last 2 years
- Focus on companies competing for same customers/market share

SECTOR PROFILE REQUIREMENTS:
core_inputs: Specific materials/resources unique to this industry (not "raw materials", "capital", "labor")
core_channels: Specific distribution/sales channels
core_geos: Primary revenue geographic regions (3 max)
benchmarks: Sector-specific indices/commodities, NOT S&P 500/NASDAQ unless no sector alternative exists

ALIASES/BRANDS/ASSETS:
- Include only well-known brand names that appear in financial news
- Assets should be major facilities/operations mentioned in earnings calls
- Avoid internal product codes or minor brands

VALIDATION CHECKLIST:
□ All competitor tickers are valid and current
□ No industry keyword would match unrelated news
□ Benchmarks are sector-specific
□ Core inputs are materially specific to the business
□ All information is factually accurate

Generate response in valid JSON format with all required fields."""

    user_prompt = f"""Generate metadata for hedge fund news monitoring. Focus on precision to avoid irrelevant news articles.

Ticker: {ticker}
Company: {company_name}
Current date: September 2025

Required JSON format:
{{
    "ticker": "{ticker}",
    "name": "{company_name}",
    "sector": "GICS Sector",
    "industry": "GICS Industry",
    "sub_industry": "GICS Sub-Industry",
    "industry_keywords": ["keyword1", "keyword2", "keyword3"],
    "competitors": ["Company Name (TICKER)", "Company Name (TICKER)", "Company Name (TICKER)"],
    "sector_profile": {{
        "core_inputs": ["input1", "input2", "input3"],
        "core_channels": ["channel1", "channel2"],
        "core_geos": ["geo1", "geo2", "geo3"],
        "benchmarks": ["benchmark1", "benchmark2"]
    }},
    "aliases_brands_assets": {{
        "aliases": ["alias1", "alias2"],
        "brands": ["brand1", "brand2", "brand3"],
        "assets": ["asset1", "asset2"]
    }}
}}"""

    try:
        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        metadata = json.loads(response.choices[0].message.content)
        
        # Validation
        validation_errors = validate_metadata(metadata)
        if validation_errors:
            print(f"⚠️ Validation warnings for {ticker}:")
            for error in validation_errors:
                print(f"  - {error}")
        
        # Store in database
        stored_successfully = store_ticker_metadata(
            ticker=metadata['ticker'],
            name=metadata['name'],
            sector=metadata['sector'],
            industry=metadata['industry'],
            sub_industry=metadata['sub_industry'],
            industry_keywords=metadata['industry_keywords'],
            competitors=metadata['competitors'],
            sector_profile=metadata['sector_profile'],
            aliases_brands_assets=metadata['aliases_brands_assets']
        )
        
        if stored_successfully:
            print(f"✅ Generated and stored metadata for {ticker}")
            return metadata
        else:
            print(f"❌ Failed to store metadata for {ticker}")
            return None
            
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON response for {ticker}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error generating metadata for {ticker}: {e}")
        return None

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

# ------------------------------------------------------------------------------
# Email Digest
# ------------------------------------------------------------------------------
def build_digest_html(articles_by_ticker: Dict[str, Dict[str, List[Dict]]], period_days: int) -> Tuple[str, str]:
    """Build HTML email digest and return both HTML and text export"""
    
    # Load ticker metadata for competitor names
    ticker_metadata_cache = {}
    for ticker in articles_by_ticker.keys():
        config = get_ticker_config(ticker)
        if config:
            # Convert competitors back to dict format for the helper function
            competitors = []
            for comp_str in config.get("competitors", []):
                match = re.search(r'^(.+?)\s*\(([A-Z]{1,5})\)$', comp_str)
                if match:
                    competitors.append({"name": match.group(1).strip(), "ticker": match.group(2)})
                else:
                    competitors.append({"name": comp_str, "ticker": None})
            
            ticker_metadata_cache[ticker] = {
                "industry_keywords": config.get("industry_keywords", []),
                "competitors": competitors,
                "company_name": config.get("name", ticker),
                "sector": config.get("sector", ""),
                "industry": config.get("industry", ""),
                "sub_industry": config.get("sub_industry", ""),
                "sector_profile": config.get("sector_profile"),
                "aliases_brands_assets": config.get("aliases_brands_assets")
            }
    
    # Generate text export for AI evaluation
    text_export = create_ai_evaluation_text(articles_by_ticker)
    
    # Use the new timestamp formatting
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
        ".description { color: #6c757d; font-size: 11px; font-style: italic; margin-top: 5px; line-height: 1.4; display: block; }",
        ".ai-summary { color: #2c5aa0; font-size: 12px; margin-top: 8px; line-height: 1.4; background-color: #f8f9ff; padding: 8px; border-radius: 4px; border-left: 3px solid #3498db; }",
        ".company { border-left-color: #27ae60; }",
        ".industry { border-left-color: #f39c12; }",
        ".competitor { border-left-color: #e74c3c; }",
        ".meta { color: #95a5a6; font-size: 11px; }",
        ".score { display: inline-block; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 10px; margin-left: 5px; }",
        ".high-score { background-color: #d4edda; color: #155724; }",
        ".med-score { background-color: #fff3cd; color: #856404; }",
        ".low-score { background-color: #f8d7da; color: #721c24; }",
        ".impact { display: inline-block; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 10px; margin-left: 5px; }",
        ".impact-positive { background-color: #d4edda; color: #155724; }",
        ".impact-negative { background-color: #f8d7da; color: #721c24; }",
        ".impact-mixed { background-color: #fff3cd; color: #856404; }",
        ".impact-neutral { background-color: #e2e3e5; color: #383d41; }",
        ".analyzed-badge { display: inline-block; padding: 2px 6px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e3f2fd; color: #1565c0; border: 1px solid #90caf9; }",
        ".source-badge { display: inline-block; padding: 2px 6px; margin-left: 8px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e9ecef; color: #495057; }",
        ".competitor-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fdeaea; color: #c53030; border: 1px solid #feb2b2; max-width: 200px; white-space: nowrap; overflow: visible; }",
        ".industry-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fef5e7; color: #b7791f; border: 1px solid #f6e05e; max-width: 200px; white-space: nowrap; overflow: visible; }",
        ".sector-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e8f5e8; color: #2e7d32; border: 1px solid #a5d6a7; }",
        ".geography-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #f3e5f5; color: #7b1fa2; border: 1px solid #ce93d8; }",
        ".alias-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fff3e0; color: #f57c00; border: 1px solid #ffcc02; }",
        ".brand-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fce4ec; color: #ad1457; border: 1px solid #f8bbd9; }",
        ".asset-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e8eaf6; color: #3f51b5; border: 1px solid #c5cae9; }",
        ".keywords { background-color: #f8f9fa; padding: 15px; margin: 15px 0; border-radius: 8px; font-size: 11px; border-left: 4px solid #3498db; }",
        "a { color: #2980b9; text-decoration: none; }",
        "a:hover { text-decoration: underline; }",
        ".summary { margin-top: 20px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }",
        ".ticker-section { margin-bottom: 40px; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "</style></head><body>",
        f"<h1>Stock Intelligence Report</h1>",
        f"<div class='summary'>",
        f"<strong>Report Period:</strong> Last {period_days} days<br>",
        f"<strong>Generated:</strong> {current_time_est}<br>",
        f"<strong>Tickers Covered:</strong> {', '.join(articles_by_ticker.keys())}<br>",
        f"<strong>AI Features:</strong> Content Analysis + Hedge Fund Summaries",
        "</div>"
    ]
    
    for ticker, categories in articles_by_ticker.items():
        total_articles = sum(len(articles) for articles in categories.values())
        
        html.append(f"<div class='ticker-section'>")
        html.append(f"<h2>{ticker} - {total_articles} Total Articles</h2>")
        
        # Enhanced keyword information with full metadata
        if ticker in ticker_metadata_cache:
            metadata = ticker_metadata_cache[ticker]
            html.append("<div class='keywords'>")
            html.append(f"<strong>🤖 AI-Powered Monitoring Configuration:</strong><br><br>")
            
            # Company details
            if metadata.get("company_name"):
                html.append(f"<strong>Company:</strong> {metadata['company_name']}<br>")
            
            # Sector information
            if metadata.get("sector"):
                html.append(f"<strong>Sector:</strong> <span class='sector-badge'>🏭 {metadata['sector']}</span><br>")
            if metadata.get("industry"):
                html.append(f"<strong>Industry:</strong> {metadata['industry']}<br>")
            if metadata.get("sub_industry"):
                html.append(f"<strong>Sub-Industry:</strong> {metadata['sub_industry']}<br>")
            
            # Keywords and competitors
            if metadata.get("industry_keywords"):
                industry_badges = [f'<span class="industry-badge">🏭 {kw}</span>' for kw in metadata['industry_keywords']]
                html.append(f"<strong>Industry Keywords:</strong> {' '.join(industry_badges)}<br>")
            
            if metadata.get("competitors"):
                competitor_badges = [f'<span class="competitor-badge">🏢 {comp["name"] if isinstance(comp, dict) else comp}</span>' for comp in metadata['competitors']]
                html.append(f"<strong>Competitors:</strong> {' '.join(competitor_badges)}<br>")
            
            # Enhanced metadata from sector profile
            sector_profile = metadata.get("sector_profile")
            if sector_profile:
                try:
                    if isinstance(sector_profile, str):
                        sector_data = json.loads(sector_profile)
                    else:
                        sector_data = sector_profile
                    
                    if sector_data.get("core_geos"):
                        geo_badges = [f'<span class="geography-badge">🌍 {geo}</span>' for geo in sector_data["core_geos"][:5]]
                        html.append(f"<strong>Core Geographies:</strong> {' '.join(geo_badges)}<br>")
                    
                    if sector_data.get("core_inputs"):
                        input_badges = [f'<span class="sector-badge">⚡ {inp}</span>' for inp in sector_data["core_inputs"][:5]]
                        html.append(f"<strong>Core Inputs:</strong> {' '.join(input_badges)}<br>")
                    
                    if sector_data.get("core_channels"):
                        channel_badges = [f'<span class="geography-badge">📊 {ch}</span>' for ch in sector_data["core_channels"][:5]]
                        html.append(f"<strong>Core Channels:</strong> {' '.join(channel_badges)}<br>")
                        
                except Exception as e:
                    LOG.warning(f"Error parsing sector profile for {ticker}: {e}")
            
            # Aliases, brands, and assets
            aliases_brands = metadata.get("aliases_brands_assets")
            if aliases_brands:
                try:
                    if isinstance(aliases_brands, str):
                        alias_data = json.loads(aliases_brands)
                    else:
                        alias_data = aliases_brands
                    
                    if alias_data.get("aliases"):
                        alias_badges = [f'<span class="alias-badge">🏷️ {alias}</span>' for alias in alias_data["aliases"][:5]]
                        html.append(f"<strong>Aliases:</strong> {' '.join(alias_badges)}<br>")
                    
                    if alias_data.get("brands"):
                        brand_badges = [f'<span class="brand-badge">🏪 {brand}</span>' for brand in alias_data["brands"][:5]]
                        html.append(f"<strong>Brands:</strong> {' '.join(brand_badges)}<br>")
                    
                    if alias_data.get("assets"):
                        asset_badges = [f'<span class="asset-badge">🏗️ {asset}</span>' for asset in alias_data["assets"][:5]]
                        html.append(f"<strong>Key Assets:</strong> {' '.join(asset_badges)}")
                        
                except Exception as e:
                    LOG.warning(f"Error parsing aliases/brands for {ticker}: {e}")
            
            html.append("</div>")
        
        # Sort articles within each category - prioritize triaged/analyzed articles, then by time
        for category in ["company", "industry", "competitor"]:
            if category in categories and categories[category]:
                articles = categories[category]
                
                # Sort: AI triaged/analyzed first, then quality domains, then by time
                def sort_key(article):
                    is_analyzed = bool(article.get('ai_summary') or article.get('ai_triage_selected'))
                    has_quality_domain = normalize_domain(article.get("domain", "")) in QUALITY_DOMAINS
                    pub_time = article.get("published_at") or datetime.min.replace(tzinfo=timezone.utc)
                    
                    # Priority: 0 = analyzed, 1 = quality domain, 2 = other
                    priority = 0 if is_analyzed else (1 if has_quality_domain else 2)
                    return (priority, -pub_time.timestamp())
                
                sorted_articles = sorted(articles, key=sort_key)
                
                html.append(f"<h3>{category.title()} News ({len(articles)} articles)</h3>")
                for article in sorted_articles[:100]:  # Show up to 100 articles
                    html.append(_format_article_html_with_ai_summary(article, category, ticker_metadata_cache))
        
        html.append("</div>")
    
    html.append("""
        <div style='margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; font-size: 11px; color: #6c757d;'>
            <strong>Enhanced AI Features:</strong><br>
            • Content Analysis: Full article scraping with intelligent extraction<br>
            • Hedge Fund Summaries: AI-generated analytical summaries for scraped content<br>
            • Component-Based Scoring: Transparent quality scoring with detailed reasoning<br>
            • "Analyzed" badge indicates articles with both scraped content and AI summary
        </div>
        </body></html>
    """)
    
    html_content = "".join(html)
    
    return html_content, text_export

# Updated email sending function with text attachment
def send_email(subject: str, html_body: str, text_attachment: str = None, to: str = None):
    """Send email with HTML body displayed properly and optional text attachment"""
    if not all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, EMAIL_FROM]):
        LOG.error("SMTP not fully configured")
        return False
    
    try:
        recipient = to or DIGEST_TO
        
        # Create multipart message with HTML as primary content
        msg = MIMEMultipart('mixed')  # Use 'mixed' for attachments
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = recipient
        
        # Create multipart alternative for HTML/text content
        msg_alternative = MIMEMultipart('alternative')
        
        # Add text version (fallback for very old email clients)
        text_body = "This email contains HTML content. Please view in an HTML-capable email client."
        msg_alternative.attach(MIMEText(text_body, "plain", "utf-8"))
        
        # Add HTML body (this will be displayed by modern email clients)
        msg_alternative.attach(MIMEText(html_body, "html", "utf-8"))
        
        # Attach the alternative part to main message
        msg.attach(msg_alternative)
        
        # Add text attachment if provided and not empty
        if text_attachment and len(text_attachment.strip()) > 0:
            try:
                # Create text attachment
                attachment = MIMEText(text_attachment, "plain", "utf-8")
                attachment.add_header(
                    "Content-Disposition", 
                    "attachment", 
                    filename="ai_evaluation_data.txt"
                )
                msg.attach(attachment)
                LOG.info(f"Added text attachment: {len(text_attachment)} characters")
            except Exception as e:
                LOG.warning(f"Failed to add text attachment: {e}")
                # Continue without attachment rather than failing
        
        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            if SMTP_STARTTLS:
                server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, [recipient], msg.as_string())
        
        attachment_info = f" with attachment ({len(text_attachment)} chars)" if text_attachment else " (no attachment)"
        LOG.info(f"Email sent{attachment_info} to {recipient}")
        return True
        
    except Exception as e:
        LOG.error(f"Email send failed: {e}")
        return False

def send_quick_ingest_email_with_triage(articles_by_ticker: Dict[str, Dict[str, List[Dict]]], triage_results: Dict[str, Dict[str, List[Dict]]]) -> bool:
    """Send enhanced quick email with domain, title, triage results - REMOVED REDUNDANT KEYWORD SECTIONS"""
    try:
        current_time_est = format_timestamp_est(datetime.now(timezone.utc))
        
        # Load ticker metadata for competitor names and enhanced info
        ticker_metadata_cache = {}
        for ticker in articles_by_ticker.keys():
            config = get_ticker_config(ticker)
            if config:
                # Convert competitors back to dict format for the helper function
                competitors = []
                for comp_str in config.get("competitors", []):
                    match = re.search(r'^(.+?)\s*\(([A-Z]{1,5})\)$', comp_str)
                    if match:
                        competitors.append({"name": match.group(1).strip(), "ticker": match.group(2)})
                    else:
                        competitors.append({"name": comp_str, "ticker": None})
                
                ticker_metadata_cache[ticker] = {
                    "industry_keywords": config.get("industry_keywords", []),
                    "competitors": competitors,
                    "sector": config.get("sector", ""),
                    "company_name": config.get("name", ticker)
                }
        
        # Generate triage evaluation text for attachment
        triage_evaluation_text = create_triage_evaluation_text(articles_by_ticker, triage_results, ticker_metadata_cache)
        
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
            ".meta { color: #95a5a6; font-size: 11px; }",
            ".triage { display: inline-block; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 10px; margin-left: 8px; }",
            ".triage-selected { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }",
            ".triage-p1 { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }",
            ".triage-p2 { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }",
            ".triage-p3 { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }",
            ".triage-p4 { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }",
            ".triage-p5 { background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }",
            ".triage-skipped { background-color: #f8f9fa; color: #6c757d; border: 1px solid #dee2e6; }",
            ".triage-quality { background-color: #e8f5e8; color: #2e7d32; border: 1px solid #a5d6a7; }",
            ".selected-for-scrape { background-color: #e8f5e8 !important; }",
            ".quality-domain-selected { background-color: #e3f2fd !important; }",
            ".source-badge { display: inline-block; padding: 2px 6px; margin-left: 8px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e9ecef; color: #495057; }",
            ".competitor-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fdeaea; color: #c53030; border: 1px solid #feb2b2; max-width: 200px; white-space: nowrap; overflow: visible; }",
            ".industry-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fef5e7; color: #b7791f; border: 1px solid #f6e05e; max-width: 200px; white-space: nowrap; overflow: visible; }",
            ".sector-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e8f5e8; color: #2e7d32; border: 1px solid #a5d6a7; }",
            ".keywords { background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 11px; }",
            ".summary { margin-top: 20px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }",
            ".ticker-section { margin-bottom: 40px; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            "</style></head><body>",
            f"<h1>Quick Intelligence Report - Triage Complete</h1>",
            f"<div class='summary'>",
            f"<strong>Generated:</strong> {current_time_est}<br>",
            f"<strong>Status:</strong> Articles ingested and triaged, AI analysis and scraping in progress...<br>",
            f"<strong>Tickers Covered:</strong> {', '.join(articles_by_ticker.keys())}<br>",
            f"<strong>AI Triage:</strong> Priority scoring complete + Quality domain selections",
            "</div>"
        ]
        
        total_articles = 0
        total_selected = 0
        total_quality_domains = 0
        
        for ticker, categories in articles_by_ticker.items():
            ticker_count = sum(len(articles) for articles in categories.values())
            total_articles += ticker_count
            
            # Count selected articles for this ticker
            ticker_selected = 0
            triage_data = triage_results.get(ticker, {})
            for category in ["company", "industry", "competitor"]:
                ticker_selected += len(triage_data.get(category, []))
            
            # Count quality domain articles
            ticker_quality_domains = 0
            for category, articles in categories.items():
                for article in articles:
                    domain = normalize_domain(article.get("domain", ""))
                    if domain in QUALITY_DOMAINS:
                        ticker_quality_domains += 1
            
            total_selected += ticker_selected
            total_quality_domains += ticker_quality_domains
            
            html.append(f"<div class='ticker-section'>")
            html.append(f"<h2>{ticker} - {ticker_count} Total Articles</h2>")
            html.append(f"<p><strong>AI Triage Selected:</strong> {ticker_selected} articles | <strong>Quality Domains:</strong> {ticker_quality_domains} articles</p>")
            
            # Enhanced keyword information with metadata (similar to final email)
            if ticker in ticker_metadata_cache:
                metadata = ticker_metadata_cache[ticker]
                html.append("<div class='keywords'>")
                html.append(f"<strong>AI-Powered Monitoring Keywords:</strong><br>")
                
                if metadata.get("industry_keywords"):
                    industry_badges = [f'<span class="industry-badge">{kw}</span>' for kw in metadata['industry_keywords']]
                    html.append(f"<strong>Industry:</strong> {' '.join(industry_badges)}<br>")
                
                if metadata.get("competitors"):
                    competitor_badges = [f'<span class="competitor-badge">{comp["name"] if isinstance(comp, dict) else comp}</span>' for comp in metadata['competitors']]
                    html.append(f"<strong>Competitors:</strong> {' '.join(competitor_badges)}<br>")
                
                # Sector information
                if metadata.get("sector"):
                    html.append(f"<strong>Sector:</strong> <span class='sector-badge'>{metadata['sector']}</span>")
                
                html.append("</div>")
            
            for category, articles in categories.items():
                if not articles:
                    continue
                
                # Get triage results for this category
                category_triage = triage_data.get(category, [])
                selected_article_data = {item["id"]: item for item in category_triage}
                
                # REMOVED: category-specific keywords section - no more redundant "Monitoring Keywords" or "Monitoring Competitors"
                
                # Create combined list with triage priority and quality domain sorting
                enhanced_articles = []
                for idx, article in enumerate(articles):
                    domain = normalize_domain(article.get("domain", ""))
                    is_ai_selected = idx in selected_article_data
                    is_quality_domain = domain in QUALITY_DOMAINS
                    
                    priority = 999  # Default low priority
                    triage_reason = ""
                    
                    if is_ai_selected:
                        priority = selected_article_data[idx].get("scrape_priority", 5)
                        triage_reason = selected_article_data[idx].get("why", "")
                    elif is_quality_domain:
                        priority = 2.5  # Between P2 and P3 for quality domains
                        triage_reason = "Quality domain auto-selected"
                    
                    enhanced_articles.append({
                        "article": article,
                        "idx": idx,
                        "priority": priority,
                        "is_ai_selected": is_ai_selected,
                        "is_quality_domain": is_quality_domain,
                        "triage_reason": triage_reason,
                        "published_at": article.get("published_at")
                    })
                
                # Sort: AI triage + quality domains first (by priority), then by publication time
                enhanced_articles.sort(key=lambda x: (
                    x["priority"],  # Lower priority number = higher priority
                    -(x["published_at"].timestamp() if x["published_at"] else 0)  # Newer articles first
                ))
                
                selected_count = len([a for a in enhanced_articles if a["is_ai_selected"] or a["is_quality_domain"]])
                html.append(f"<h3>{category.title()} ({len(articles)} articles, {selected_count} selected)</h3>")
                
                for enhanced_article in enhanced_articles[:100]:  # Show up to 100 per category
                    article = enhanced_article["article"]
                    domain = article.get("domain", "unknown")
                    title = article.get("title", "No Title")
                    
                    # Add keyword badge for this specific article
                    article_keyword_badge = ""
                    if category in ["industry", "competitor"] and article.get('search_keyword'):
                        keyword = article['search_keyword']
                        if category == "industry":
                            article_keyword_badge = f'<span class="industry-badge">{keyword}</span>'
                        elif category == "competitor":
                            # Get full competitor name using database lookup
                            comp_name = get_competitor_display_name(
                                keyword, 
                                article.get('competitor_ticker')
                            )
                            article_keyword_badge = f'<span class="competitor-badge">{comp_name}</span>'
                    
                    # Determine article class and triage badge
                    article_class = f"article {category}"
                    triage_badge = ""
                    
                    if enhanced_article["is_ai_selected"] and enhanced_article["is_quality_domain"]:
                        article_class += " selected-for-scrape"
                        priority = int(enhanced_article["priority"])
                        triage_badge = f'<span class="triage triage-p{priority}" title="{enhanced_article["triage_reason"]}">AI P{priority}</span><span class="triage triage-quality">Quality</span>'
                    elif enhanced_article["is_ai_selected"]:
                        article_class += " selected-for-scrape"
                        priority = int(enhanced_article["priority"])
                        triage_badge = f'<span class="triage triage-p{priority}" title="{enhanced_article["triage_reason"]}">AI P{priority}</span>'
                    elif enhanced_article["is_quality_domain"]:
                        article_class += " quality-domain-selected"
                        triage_badge = '<span class="triage triage-quality">Quality Domain</span>'
                    else:
                        triage_badge = '<span class="triage triage-skipped">Skipped</span>'
                    
                    # Format publication time
                    pub_time = ""
                    if article.get("published_at"):
                        pub_time = format_timestamp_est(article["published_at"])
                    
                    html.append(f"""
                    <div class='{article_class}'>
                        <div class='article-header'>
                            <span class='source-badge'>{get_or_create_formal_domain_name(domain)}</span>
                            {article_keyword_badge}
                            {triage_badge}
                        </div>
                        <div class='article-content'>
                            <a href='{article.get("resolved_url") or article.get("url", "")}'>{title}</a>
                            <span class='meta'> | {pub_time}</span>
                        </div>
                    </div>
                    """)
            
            html.append("</div>")
        
        total_to_scrape = total_selected + total_quality_domains
        html.append(f"<div class='summary'>")
        html.append(f"<strong>Total Articles:</strong> {total_articles}<br>")
        html.append(f"<strong>AI Triage Selected:</strong> {total_selected}<br>")
        html.append(f"<strong>Quality Domain Selected:</strong> {total_quality_domains}<br>")
        html.append(f"<strong>Total for Scraping:</strong> {total_to_scrape}<br>")
        html.append(f"<strong>Next:</strong> Full content analysis and hedge fund summaries in progress...")
        html.append("</div>")
        html.append("</body></html>")
        
        html_content = "".join(html)
        subject = f"Quick Intelligence: {total_to_scrape} articles selected for analysis"
        
        return send_email(subject, html_content, triage_evaluation_text)
        
    except Exception as e:
        LOG.error(f"Quick triage email send failed: {e}")
        return False

def fetch_digest_articles_with_content(hours: int = 24, tickers: List[str] = None) -> Dict[str, Dict[str, List[Dict]]]:
    """Fetch categorized articles for digest with content scraping data and AI summaries"""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    days = int(hours / 24) if hours >= 24 else 0
    period_label = f"{days} days" if days > 0 else f"{hours:.0f} hours"
    
    with db() as conn, conn.cursor() as cur:
        # Enhanced query to include AI summary field
        if tickers:
            cur.execute("""
                SELECT 
                    f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.quality_score, f.published_at,
                    f.found_at, f.category, f.original_source_url,
                    f.search_keyword, f.ai_impact, f.ai_reasoning, f.ai_summary,
                    f.scraped_content, f.content_scraped_at, f.scraping_failed, f.scraping_error,
                    f.source_tier, f.event_multiplier, f.event_multiplier_reason,
                    f.relevance_boost, f.relevance_boost_reason, f.numeric_bonus,
                    f.penalty_multiplier, f.penalty_reason
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
                    f.found_at, f.category, f.original_source_url,
                    f.search_keyword, f.ai_impact, f.ai_reasoning, f.ai_summary,
                    f.scraped_content, f.content_scraped_at, f.scraping_failed, f.scraping_error,
                    f.source_tier, f.event_multiplier, f.event_multiplier_reason,
                    f.relevance_boost, f.relevance_boost_reason, f.numeric_bonus,
                    f.penalty_multiplier, f.penalty_reason
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
    
    # Use the new digest function with text export
    html, text_export = build_digest_html(articles_by_ticker, days if days > 0 else 1)
    
    tickers_str = ', '.join(articles_by_ticker.keys())
    subject = f"Stock Intelligence: {tickers_str} - {total_articles} articles"
    success = send_email(subject, html, text_export)
    
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
    
    return {
        "status": "sent" if success else "failed",
        "articles": total_articles,
        "tickers": list(articles_by_ticker.keys()),
        "by_category": category_counts,
        "content_scraping_stats": content_stats,
        "recipient": DIGEST_TO
    }

def _format_article_html(article: Dict, category: str) -> str:
    """Format article HTML with AI analysis display"""
    import html
    
    # Format timestamp for individual articles
    if article["published_at"]:
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
    
    # Quality score styling
    score = article["quality_score"]
    score_class = "high-score" if score >= 70 else "med-score" if score >= 40 else "low-score"
    
    # Impact styling - FIXED: Handle None values
    ai_impact = (article.get("ai_impact") or "").lower()
    impact_class = f"impact-{ai_impact}" if ai_impact in ["positive", "negative", "mixed", "unclear"] else "impact-unclear"
    impact_display = ai_impact.title() if ai_impact else "N/A"
    
    # AI reasoning - FIXED: Handle None values
    ai_reasoning = (article.get("ai_reasoning") or "").strip()
    
    # Build metadata badges for category-specific information
    metadata_badges = []
    
    if category == "competitor" and article.get('search_keyword'):
        competitor_name = article['search_keyword']
        metadata_badges.append(f'<span class="competitor-badge">🏢 {competitor_name}</span>')
    elif category == "industry" and article.get('search_keyword'):
        industry_keyword = article['search_keyword']
        metadata_badges.append(f'<span class="industry-badge">🏭 {industry_keyword}</span>')
    
    enhanced_metadata = "".join(metadata_badges)
    
    # Get description and format it
    description = article.get("description", "").strip()
    description_html = ""
    if description:
        description = html.unescape(description)
        description = re.sub(r'<[^>]+>', '', description)
        description = re.sub(r'\s+', ' ', description).strip()
        
        if len(description) > 500:
            description = description[:500] + "..."
        
        description = html.escape(description)
        description_html = f"<br><div class='description'>{description}</div>"
    
    # Build analysis section for company articles with AI data
    analysis_html = ""
    if category == "company" and (ai_impact or ai_reasoning):
        analysis_html = f"""
        <div class='article-analysis'>
            <strong>🤖 AI Analysis:</strong>
            {f'<span class="impact {impact_class}">{impact_display}</span>' if ai_impact else ''}
            {f'<div class="ai-reasoning">{html.escape(ai_reasoning)}</div>' if ai_reasoning else ''}
        </div>
        """
    
    return f"""
    <div class='article {category}'>
        <div class='article-header'>
            <span class='source-badge'>{display_source}</span>
            {enhanced_metadata}
            <span class='score {score_class}'>Score: {score:.0f}</span>
            {f'<span class="impact {impact_class}">{impact_display}</span>' if ai_impact and category == "company" else ''}
        </div>
        <div class='article-content'>
            <a href='{link_url}' target='_blank'>{title}</a>
            <span class='meta'> | {pub_date}</span>
            {description_html}
            {analysis_html}
        </div>
    </div>
    """
    
def fetch_digest_articles(hours: int = 24, tickers: List[str] = None) -> Dict[str, Dict[str, List[Dict]]]:
    """Fetch categorized articles for digest with content scraping data"""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    days = int(hours / 24) if hours >= 24 else 0
    period_label = f"{days} days" if days > 0 else f"{hours:.0f} hours"
    
    with db() as conn, conn.cursor() as cur:
        # Enhanced query to include content scraping fields
        if tickers:
            cur.execute("""
                SELECT 
                    f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.quality_score, f.published_at,
                    f.found_at, f.category, f.related_ticker, f.original_source_url,
                    f.search_keyword, f.ai_impact, f.ai_reasoning,
                    f.scraped_content, f.content_scraped_at, f.scraping_failed, f.scraping_error
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
                    f.search_keyword, f.ai_impact, f.ai_reasoning,
                    f.scraped_content, f.content_scraped_at, f.scraping_failed, f.scraping_error
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
    
    # FIXED: build_digest_html now returns tuple, extract first element
    html, text_export = build_digest_html(articles_by_ticker, days if days > 0 else 1)
    
    tickers_str = ', '.join(articles_by_ticker.keys())
    subject = f"Stock Intelligence: {tickers_str} - {total_articles} articles"
    # FIXED: Add empty text attachment parameter
    success = send_email(subject, html, "")
    
    # Count by category and content scraping
    category_counts = {"company": 0, "industry": 0, "competitor": 0}
    content_stats = {"scraped": 0, "failed": 0, "skipped": 0}
    
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
    
    # Final summary logging
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
            LOG.info(f"  {ticker}: {sum(ticker_feeds.values())} new feeds created")
            LOG.info(f"    Company: {ticker_feeds['company']}, Industry: {ticker_feeds['industry']}, Competitor: {ticker_feeds['competitor']}")
        else:
            LOG.info(f"  {ticker}: 0 new feeds created (already at limits)")
    
    total_feeds_created = len([r for r in results if "feed" in r])
    
    return {
        "status": "initialized",
        "tickers": body.tickers,
        "feeds": [r for r in results if "feed" in r],  # Only return actual feed creations
        "summary": {
            "total_feeds_created": total_feeds_created,
            "feeds_by_ticker": feeds_by_ticker,
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
    Enhanced ingest with separate ingestion (50/25/25) and dynamic scraping limits:
    1. Ingest URLs from feeds with strict limits (50 company, 25 per industry keyword, 25 per competitor)
    2. Perform AI triage + quality domain selection
    3. Send enhanced quick email with triage results
    4. Scrape selected articles with dynamic limits (20 + 5×keywords + 5×competitors)
    5. Send final email with full analysis
    """
    require_admin(request)
    ensure_schema()
    update_schema_for_content()
    update_schema_for_triage()
    
    LOG.info("=== CRON INGEST STARTING (INGESTION 50/25/25 + DYNAMIC SCRAPING) ===")
    LOG.info(f"Processing window: {minutes} minutes")
    LOG.info(f"Target tickers: {tickers or 'ALL'}")
    
    # Reset both ingestion and scraping stats
    reset_ingestion_stats()
    reset_scraping_stats()
    
    # Calculate dynamic scraping limits for each ticker
    dynamic_limits = {}
    if tickers:
        for ticker in tickers:
            dynamic_limits[ticker] = calculate_dynamic_scraping_limits(ticker)
    
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
    
    LOG.info(f"=== PHASE 1: INGESTING URLS WITH LIMITS (50/25/25) FROM {len(feeds)} FEEDS ===")
    
    # PHASE 1: Ingest URLs with strict limits
    articles_by_ticker = {}
    ingest_stats = {"total_processed": 0, "total_inserted": 0, "total_duplicates": 0, "total_spam_blocked": 0, "total_limit_reached": 0}
    
    for feed in feeds:
        try:
            stats = ingest_feed_basic_only(feed)
            
            # Collect articles for triage
            ticker = feed["ticker"]
            category = feed.get("category", "company")
            
            if ticker not in articles_by_ticker:
                articles_by_ticker[ticker] = {"company": [], "industry": [], "competitor": []}
            
            # Get recently inserted articles for this feed
            with db() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT id, url, resolved_url, title, domain, published_at, category, search_keyword
                    FROM found_url 
                    WHERE feed_id = %s AND found_at >= %s
                    ORDER BY found_at DESC
                """, (feed["id"], datetime.now(timezone.utc) - timedelta(minutes=minutes)))
                
                feed_articles = list(cur.fetchall())
                articles_by_ticker[ticker][category].extend(feed_articles)
            
            ingest_stats["total_processed"] += stats["processed"]
            ingest_stats["total_inserted"] += stats["inserted"]
            ingest_stats["total_duplicates"] += stats["duplicates"]
            ingest_stats["total_spam_blocked"] += stats.get("blocked_spam", 0)
            ingest_stats["total_limit_reached"] += stats.get("limit_reached", 0)
            
        except Exception as e:
            LOG.error(f"Feed ingest failed for {feed['name']}: {e}")
            continue
    
    # Log final ingestion statistics
    LOG.info("=== INGESTION LIMITS FINAL STATUS ===")
    LOG.info(f"Company: {ingestion_stats['company_ingested']}/{ingestion_stats['limits']['company']}")
    for keyword, count in ingestion_stats["industry_ingested_by_keyword"].items():
        LOG.info(f"Industry '{keyword}': {count}/{ingestion_stats['limits']['industry_per_keyword']}")
    for keyword, count in ingestion_stats["competitor_ingested_by_keyword"].items():
        LOG.info(f"Competitor '{keyword}': {count}/{ingestion_stats['limits']['competitor_per_keyword']}")
    
    LOG.info(f"=== PHASE 1 COMPLETE: {ingest_stats['total_inserted']} articles ingested (limits enforced) ===")
    
    # PHASE 2: Enhanced AI Triage + Quality Domains + Tiered Backfill
    LOG.info("=== PHASE 2: ENHANCED AI TRIAGE WITH TIERED BACKFILL ===")
    triage_results = {}
    
    for ticker in articles_by_ticker.keys():
        LOG.info(f"Running enhanced triage with backfill for {ticker}")
        
        # Calculate dynamic limits for this ticker
        config = get_ticker_config(ticker)
        industry_keywords = config.get("industry_keywords", []) if config else []
        competitors = config.get("competitors", []) if config else []
        
        target_limits = {
            "company": 20,
            "industry": len(industry_keywords) * 5,  # 5 per keyword
            "competitor": len(competitors) * 5        # 5 per competitor
        }
        
        selected_results = perform_ai_triage_batch_with_tiered_backfill(
            articles_by_ticker[ticker], 
            ticker, 
            target_limits
        )
        triage_results[ticker] = selected_results
        
        # Update database with triage results
        for category, selected_items in selected_results.items():
            articles = articles_by_ticker[ticker][category]
            for item in selected_items:
                article_idx = item["id"]
                if article_idx < len(articles):
                    article_id = articles[article_idx]["id"]
                    with db() as conn, conn.cursor() as cur:
                        cur.execute("""
                            UPDATE found_url 
                            SET ai_triage_selected = TRUE, triage_priority = %s, triage_reasoning = %s
                            WHERE id = %s
                        """, (item.get("scrape_priority", 5), item.get("why", ""), article_id))
    
    # PHASE 3: Send enhanced quick email with triage results
    LOG.info("=== PHASE 3: SENDING ENHANCED QUICK TRIAGE EMAIL ===")
    quick_email_sent = send_quick_ingest_email_with_triage(articles_by_ticker, triage_results)
    LOG.info(f"Enhanced quick triage email sent: {quick_email_sent}")
    
    # PHASE 4: Scrape selected articles with DYNAMIC limits (20 + 5×keywords + 5×competitors)
    LOG.info("=== PHASE 4: SCRAPING SELECTED ARTICLES WITH DYNAMIC LIMITS ===")
    scraping_final_stats = {"scraped": 0, "failed": 0, "ai_analyzed": 0}
    
    for ticker in articles_by_ticker.keys():
        config = get_ticker_config(ticker)
        metadata = {
            "industry_keywords": config.get("industry_keywords", []) if config else [],
            "competitors": config.get("competitors", []) if config else []
        }
        
        selected = triage_results.get(ticker, {})
        
        # Company articles (limit 20)
        company_selected = selected.get("company", [])
        LOG.info(f"SCRAPING {ticker} Company: {len(company_selected)} articles selected")
        for item in company_selected:
            if not _check_scraping_limit("company", ticker):
                LOG.info(f"Company scraping limit reached for {ticker}")
                break
                
            article_idx = item["id"]
            if article_idx < len(articles_by_ticker[ticker]["company"]):
                article = articles_by_ticker[ticker]["company"][article_idx]
                success = scrape_and_analyze_article(article, "company", metadata, ticker)
                if success:
                    scraping_final_stats["scraped"] += 1
                    scraping_final_stats["ai_analyzed"] += 1
                else:
                    scraping_final_stats["failed"] += 1
        
        # Industry articles (5 per keyword)
        industry_selected = selected.get("industry", [])
        LOG.info(f"SCRAPING {ticker} Industry: {len(industry_selected)} articles selected")
        for item in industry_selected:
            article_idx = item["id"]
            if article_idx < len(articles_by_ticker[ticker]["industry"]):
                article = articles_by_ticker[ticker]["industry"][article_idx]
                keyword = article.get("search_keyword", "unknown")
                
                if not _check_scraping_limit("industry", keyword):
                    continue  # Skip this article, try next
                
                success = scrape_and_analyze_article(article, "industry", metadata, ticker)
                if success:
                    scraping_final_stats["scraped"] += 1
                    scraping_final_stats["ai_analyzed"] += 1
                else:
                    scraping_final_stats["failed"] += 1
        
        # Competitor articles (5 per competitor)
        competitor_selected = selected.get("competitor", [])
        LOG.info(f"SCRAPING {ticker} Competitor: {len(competitor_selected)} articles selected")
        for item in competitor_selected:
            article_idx = item["id"]
            if article_idx < len(articles_by_ticker[ticker]["competitor"]):
                article = articles_by_ticker[ticker]["competitor"][article_idx]
                keyword = article.get("search_keyword", "unknown")
                
                if not _check_scraping_limit("competitor", keyword):
                    continue  # Skip this article, try next
                
                success = scrape_and_analyze_article(article, "competitor", metadata, ticker)
                if success:
                    scraping_final_stats["scraped"] += 1
                    scraping_final_stats["ai_analyzed"] += 1
                else:
                    scraping_final_stats["failed"] += 1
    
    LOG.info(f"=== PHASE 4 COMPLETE: {scraping_final_stats['scraped']} articles scraped and analyzed ===")
    
    # Log final scraping statistics
    LOG.info("=== SCRAPING LIMITS FINAL STATUS ===")
    LOG.info(f"Company: {scraping_stats['company_scraped']}/{scraping_stats['limits']['company']}")
    for keyword, count in scraping_stats["industry_scraped_by_keyword"].items():
        LOG.info(f"Industry '{keyword}': {count}/{scraping_stats['limits']['industry_per_keyword']}")
    for keyword, count in scraping_stats["competitor_scraped_by_keyword"].items():
        LOG.info(f"Competitor '{keyword}': {count}/{scraping_stats['limits']['competitor_per_keyword']}")
    
    # PHASE 5: Send final comprehensive email
    LOG.info("=== PHASE 5: SENDING FINAL COMPREHENSIVE EMAIL ===")
    final_digest_result = fetch_digest_articles_with_content(minutes / 60, list(articles_by_ticker.keys()) if articles_by_ticker else None)
    LOG.info(f"Final comprehensive email status: {final_digest_result.get('status', 'unknown')}")
    
    # Clean old articles
    cutoff = datetime.now(timezone.utc) - timedelta(days=DEFAULT_RETAIN_DAYS)
    with db() as conn, conn.cursor() as cur:
        if tickers:
            cur.execute("DELETE FROM found_url WHERE found_at < %s AND ticker = ANY(%s)", (cutoff, tickers))
        else:
            cur.execute("DELETE FROM found_url WHERE found_at < %s", (cutoff,))
        deleted = cur.rowcount
    
    LOG.info("=== CRON INGEST COMPLETE ===")
    
    return {
        "status": "completed",
        "workflow": "5_phase_ingestion_50_25_25_dynamic_scraping",
        "phase_1_ingest": {
            **ingest_stats,
            "ingestion_limits_status": {
                "company": f"{ingestion_stats['company_ingested']}/{ingestion_stats['limits']['company']}",
                "industry_by_keyword": {k: f"{v}/{ingestion_stats['limits']['industry_per_keyword']}" for k, v in ingestion_stats["industry_ingested_by_keyword"].items()},
                "competitor_by_keyword": {k: f"{v}/{ingestion_stats['limits']['competitor_per_keyword']}" for k, v in ingestion_stats["competitor_ingested_by_keyword"].items()}
            }
        },
        "phase_2_triage": {
            "tickers_processed": len(triage_results),
            "selections_by_ticker": {k: {cat: len(items) for cat, items in v.items()} for k, v in triage_results.items()}
        },
        "phase_3_quick_email": {"sent": quick_email_sent},
        "phase_4_scraping": {
            **scraping_final_stats,
            "scraping_limits_status": {
                "company": f"{scraping_stats['company_scraped']}/{scraping_stats['limits']['company']}",
                "industry_by_keyword": {k: f"{v}/{scraping_stats['limits']['industry_per_keyword']}" for k, v in scraping_stats["industry_scraped_by_keyword"].items()},
                "competitor_by_keyword": {k: f"{v}/{scraping_stats['limits']['competitor_per_keyword']}" for k, v in scraping_stats["competitor_scraped_by_keyword"].items()}
            },
            "dynamic_limits": dynamic_limits
        },
        "phase_5_final_email": final_digest_result,
        "cleanup": {"old_articles_deleted": deleted},
        "message": f"Ingested {ingest_stats['total_inserted']} articles (50/25/25 limits), scraped {scraping_final_stats['scraped']} articles (dynamic limits: 20 + 5×keywords + 5×competitors)"
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
    
    result = fetch_digest_articles_with_content(minutes / 60, tickers)
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
                    AND f.quality_score >= 15
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
    
    # FIXED: Extract HTML from tuple and add empty text attachment
    html, text_export = build_digest_html(articles_by_ticker, 7)
    tickers_str = ', '.join(articles_by_ticker.keys())
    subject = f"FULL Stock Intelligence: {tickers_str} - {total_articles} articles"
    success = send_email(subject, html, "")
    
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
