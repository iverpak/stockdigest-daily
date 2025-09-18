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
    "businesswire.com", "prnewswire.com", "globenewswire.com",
    "insidermoneky.com", "seekingalpha.com/pro"
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

# Domain authority tiers for AI scoring
DOMAIN_TIERS = {
    # Tier A (1.0) - Premier financial news
    "reuters.com": 1.0, 
    "wsj.com": 1.0, 
    "ft.com": 1.0,
    "bloomberg.com": 1.0,
    
    # Tier A- (0.9) - Major wire services
    "apnews.com": 0.9,
    
    # Tier B (0.7) - Strong trade/tech press
    "techcrunch.com": 0.7, 
    "cnbc.com": 0.7, 
    "marketwatch.com": 0.7,
    "barrons.com": 0.7,
    
    # Tier C (0.4) - Aggregators and opinion sites
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
    "investors.com": 0.4,
    
    # Tier D (0.2) - PR wires
    "globenewswire.com": 0.2, 
    "prnewswire.com": 0.2, 
    "businesswire.com": 0.2,
    "openpr.com": 0.2, 
    "financialcontent.com": 0.2,
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

# Add this near your other global variables and functions
scraping_stats = {
    "successful_scrapes": 0,
    "failed_scrapes": 0,
    "company_scraped": 0,
    "industry_scraped_by_keyword": {},
    "competitor_scraped_by_keyword": {},
    "limits": {
        "company": 20,
        "industry_per_keyword": 5,
        "competitor_per_keyword": 5
    }
}

def reset_scraping_stats():
    """Reset scraping stats for new run"""
    global scraping_stats
    scraping_stats = {
        "successful_scrapes": 0,
        "failed_scrapes": 0,
        "company_scraped": 0,
        "industry_scraped_by_keyword": {},
        "competitor_scraped_by_keyword": {},
        "limits": {
            "company": 20,
            "industry_per_keyword": 5,
            "competitor_per_keyword": 5
        }
    }

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

def safe_content_scraper_with_playwright(url: str, domain: str, scraped_domains: set) -> Tuple[Optional[str], str]:
    """
    Enhanced scraper with Playwright fallback for failed requests (updated for triage workflow)
    """
    global playwright_stats
    
    # First try your existing method
    content, error = safe_content_scraper(url, domain, scraped_domains)
    
    # If successful, return immediately
    if content:
        return content, f"Successfully scraped {len(content)} chars with requests"
    
    # Expanded high-value domains that justify Playwright overhead
    high_value_domains = {
        'finance.yahoo.com', 'fool.com', 'barrons.com', 'tipranks.com', 
        'msn.com', 'zacks.com', 'nasdaq.com', 'theglobeandmail.com',
        'cnbc.com', 'benzinga.com', 'businessinsider.com', 'marketwatch.com',
        'investopedia.com', 'forbes.com', 'reuters.com', 'insidermonkey.com',
        'tradingview.com', 'barchart.com', 'apnews.com', 'bloomberg.com',
        'investors.com', 'investing.com', 'seekingalpha.com', 'businesswire.com',
        '247wallst.com', 'theinformation.com', 'thestreet.com', 'gurufocus.com',
        'aol.com', 'cbsnews.com', 'entrepreneur.com', 'foxbusiness.com'
    }
    
    normalized_domain = normalize_domain(domain)
    if normalized_domain in high_value_domains:
        LOG.info(f"Trying Playwright fallback for high-value domain: {domain}")
        
        # Update stats
        playwright_stats["attempted"] += 1
        playwright_stats["by_domain"][normalized_domain]["attempts"] += 1
        
        playwright_content, playwright_error = extract_article_content_with_playwright(url, domain)
        
        if playwright_content:
            playwright_stats["successful"] += 1
            playwright_stats["by_domain"][normalized_domain]["successes"] += 1
            
            # Log stats every 10 attempts
            if playwright_stats["attempted"] % 10 == 0:
                log_playwright_stats()
                
            return playwright_content, f"Playwright success: {len(playwright_content)} chars"
        else:
            playwright_stats["failed"] += 1
            
            # Log stats every 10 attempts
            if playwright_stats["attempted"] % 10 == 0:
                log_playwright_stats()
                
            return None, f"Both methods failed - Requests: {error}, Playwright: {playwright_error}"
    
    # For low-value domains, don't waste time on Playwright
    return None, f"Requests failed: {error} (Playwright not attempted for this domain)"

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
    """Scrape content and run AI analysis for a single article"""
    try:
        article_id = article["id"]
        resolved_url = article.get("resolved_url") or article.get("url")
        domain = article.get("domain", "unknown")
        title = article.get("title", "")
        
        # Attempt content scraping
        scraped_content = None
        scraping_error = None
        content_scraped_at = None
        scraping_failed = False
        ai_summary = None
        
        if resolved_url and resolved_url.startswith(('http://', 'https://')):
            scrape_domain = normalize_domain(urlparse(resolved_url).netloc.lower())
            
            if scrape_domain not in PAYWALL_DOMAINS:
                # Use the enhanced scraper with Playwright fallback
                content, status = safe_content_scraper_with_playwright(resolved_url, scrape_domain, set())
                
                if content:
                    scraped_content = content
                    content_scraped_at = datetime.now(timezone.utc)
                    
                    # Generate AI summary from scraped content
                    ai_summary = generate_ai_summary(scraped_content, title, ticker)
                else:
                    scraping_failed = True
                    scraping_error = status
        
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
        
        LOG.info(f"Scraped and analyzed: {title[:50]}... (Score: {quality_score:.1f}, Content: {'Yes' if scraped_content else 'No'})")
        return True
        
    except Exception as e:
        LOG.error(f"Failed to scrape and analyze article {article.get('id')}: {e}")
        return False

def _update_scraping_stats(category: str, keyword: str, success: bool):
    """Helper to update scraping statistics"""
    global scraping_stats
    
    if success:
        scraping_stats["successful_scrapes"] += 1
        
        # Calculate overall success rate
        total_attempts = scraping_stats["successful_scrapes"] + scraping_stats["failed_scrapes"]
        success_rate = (scraping_stats["successful_scrapes"] / total_attempts) * 100 if total_attempts > 0 else 0
        
        if category == "company":
            scraping_stats["company_scraped"] += 1
            LOG.info(f"SCRAPING SUCCESS: Company {scraping_stats['company_scraped']}/{scraping_stats['limits']['company']} | Total: {scraping_stats['successful_scrapes']} ({success_rate:.0f}% scrape success overall)")
        
        elif category == "industry":
            if keyword not in scraping_stats["industry_scraped_by_keyword"]:
                scraping_stats["industry_scraped_by_keyword"][keyword] = 0
            scraping_stats["industry_scraped_by_keyword"][keyword] += 1
            keyword_count = scraping_stats["industry_scraped_by_keyword"][keyword]
            LOG.info(f"SCRAPING SUCCESS: Industry '{keyword}' {keyword_count}/{scraping_stats['limits']['industry_per_keyword']} | Total: {scraping_stats['successful_scrapes']} ({success_rate:.0f}% scrape success overall)")
        
        elif category == "competitor":
            if keyword not in scraping_stats["competitor_scraped_by_keyword"]:
                scraping_stats["competitor_scraped_by_keyword"][keyword] = 0
            scraping_stats["competitor_scraped_by_keyword"][keyword] += 1
            keyword_count = scraping_stats["competitor_scraped_by_keyword"][keyword]
            LOG.info(f"SCRAPING SUCCESS: Competitor '{keyword}' {keyword_count}/{scraping_stats['limits']['competitor_per_keyword']} | Total: {scraping_stats['successful_scrapes']} ({success_rate:.0f}% scrape success overall)")

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
    """Basic feed ingestion without AI processing or scraping"""
    stats = {"processed": 0, "inserted": 0, "duplicates": 0, "blocked_spam": 0, "blocked_non_latin": 0}
    
    try:
        parsed = feedparser.parse(feed["url"])
        
        for entry in parsed.entries:
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
                    cur.execute("SELECT id FROM found_url WHERE url_hash = %s", (url_hash,))
                    if cur.fetchone():
                        stats["duplicates"] += 1
                        continue
                    
                    # Parse publish date
                    published_at = None
                    if hasattr(entry, "published_parsed"):
                        published_at = parse_datetime(entry.published_parsed)
                    
                    # Basic quality score (no AI)
                    basic_quality_score = _fallback_quality_score(title, final_domain, feed["ticker"], description, [])
                    
                    display_content = description
                    category = feed.get("category", "company")
                    
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
    Enhanced article HTML formatting with AI summaries and better competitor names
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
    
    # Build metadata badges for category-specific information with better names
    metadata_badges = []
    
    if category == "competitor" and article.get('search_keyword'):
        # Use the improved competitor name function
        competitor_name = get_competitor_display_name(
            article['search_keyword'], 
            article.get('competitor_ticker'),
            ticker_metadata_cache or {}
        )
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
    """
    Fixed Yahoo Finance source extraction - properly handles HTML parsing
    """
    try:
        if "finance.yahoo.com" not in url:
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
                                not candidate_url.startswith('//') and
                                '.' in parsed.netloc):
                                
                                LOG.info(f"Successfully extracted Yahoo source: {candidate_url}")
                                return candidate_url
                        except Exception as e:
                            LOG.debug(f"URL validation failed for {candidate_url}: {e}")
                            continue
                            
                except Exception as e:
                    LOG.debug(f"Processing match failed: {e}")
                    continue
        
        # Fallback: Look for any reasonable URLs in the content
        fallback_patterns = [
            r'https://(?!finance\.yahoo\.com)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/[^\s"<>]*(?:news|article|story|press)[^\s"<>]*',
            r'https://stockstory\.org/[^\s"<>]*',
        ]
        
        for pattern in fallback_patterns:
            matches = re.finditer(pattern, html_content)
            for match in matches:
                candidate_url = match.group(0).rstrip('",')
                try:
                    parsed = urlparse(candidate_url)
                    if parsed.scheme and parsed.netloc:
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
    """Fallback scoring when AI is unavailable"""
    base_score = 50.0
    
    # Domain tier bonus
    domain_tier = _get_domain_tier(domain, title, description)
    base_score += (domain_tier - 0.5) * 20  # Scale tier to score impact
    
    # Ticker mention bonus
    if ticker.upper() in title.upper():
        base_score += 10
    
    # Keyword relevance bonus
    if keywords:
        title_lower = title.lower()
        keyword_matches = sum(1 for kw in keywords if kw.lower() in title_lower)
        base_score += min(keyword_matches * 5, 15)
    
    return max(10.0, min(90.0, base_score))

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
                selected = _triage_company_articles_full(articles, ticker, company_name, aliases_brands_assets, sector_profile)
            elif category == "industry":
                peers = config.get("competitors", []) if config else []
                selected = _triage_industry_articles_full(articles, ticker, sector_profile, peers)
            elif category == "competitor":
                peers = config.get("competitors", []) if config else []
                selected = _triage_competitor_articles_full(articles, ticker, peers, sector_profile)
            else:
                selected = []
                
            selected_results[category] = selected
            LOG.info(f"AI triage {category}: selected {len(selected)} articles for scraping")
            
        except Exception as e:
            LOG.error(f"AI triage failed for {category}: {e}")
            selected_results[category] = []
    
    return selected_results

def _make_triage_request_full(system_prompt: str, payload: Dict) -> List[Dict]:
    """Make API request to OpenAI for triage and return full results"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": OPENAI_MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload)}
            ],
            "max_completion_tokens": 1000
        }
        
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code != 200:
            LOG.error(f"OpenAI triage API error {response.status_code}: {response.text}")
            return []
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Parse the JSON response
        triage_result = json.loads(content)
        
        # Sort by priority (P1=1 is highest) then return full data
        selected = triage_result.get("selected", [])
        selected.sort(key=lambda x: x.get("scrape_priority", 5))
        
        # Return full article data with triage info
        return selected
        
    except Exception as e:
        LOG.error(f"Triage request failed: {e}")
        return []

# Update the individual triage functions to use the new _full suffix
def _triage_company_articles_full(articles: List[Dict], ticker: str, company_name: str, aliases_brands_assets: Dict, sector_profile: Dict) -> List[Dict]:
    """Triage company articles using optimized prompt - returns full results"""
    
    # Build items for API (same as before)
    items = []
    for i, article in enumerate(articles):
        domain = article.get("domain", "unknown")
        source_tier = _get_domain_tier(domain, article.get("title", ""), article.get("description", ""))
        
        items.append({
            "id": i,
            "url": article.get("resolved_url") or article.get("url", ""),
            "title": article.get("title", ""),
            "domain": domain,
            "published_at": article.get("published_at").isoformat() if article.get("published_at") else "2024-01-01T00:00:00Z",
            "source_tier": round(source_tier, 2),
            "repeat_key": "",
            "likely_repeat": False,
            "primary_id": i
        })
    
    # Build the API payload (same as before)
    payload = {
        "bucket": "company",
        "target_cap": 30,
        "ticker": ticker,
        "company_name": company_name,
        "aliases_brands_assets": aliases_brands_assets,
        "sector_profile": sector_profile,
        "items": items
    }
    
    # Same system prompt as before
    system_prompt = f"""You are a hedge-fund news router doing PRE-SCRAPE TRIAGE for COMPANY items.

Assume NO article body. Base ALL judgments ONLY on: title, domain, source_tier, and the provided metadata.

Goal:
- Choose up to target_cap items to "scrape".
- Down-prioritize likely repeats (based on repeat_key/likely_repeat/primary_id) without full canonicalization.
- Apply cross-sector logic; lean on metadata for relevance.

RELEVANCE (COMPANY)
High: title matches {{ticker | company_name | aliases | brands | assets}} OR named management/board/owner/SEC forms tied to the company.
Medium: named counterparty contract/customer/supplier with $$, units, prices, %, or formal regulator decision specific to the company.
Low: generic market talk with no clear tie to company economics.

EVENT SIGNALS (COMPANY; any sector)
Hard actions: acquires|merger|divests|spinoff|bankruptcy|Chapter 11|delist|recall|halt
Capital/structure: buyback|tender|special dividend|equity offering|convertible|refinance|rating (upgrade|downgrade|outlook)
Operations/econ: guidance|preannounce|beats|misses|earnings|margin|backlog|contract|long-term agreement|supply deal|price increase|price cut|capacity add|closure|curtailment|capex plan
Regulatory/legal: approval|license|tariff|quota|sanction|fine|settlement|DOJ|FTC|SEC|antitrust

NUMERIC CUES (COMPANY)
- Financial: EPS/revenue %, margin %, FCF, guidance deltas, buyback $, offering size/pricing, ratings level.
- Operating: capacity/units (MW/Mt/kt/GWh/lines/rigs/aircraft/rooms), utilization %, ASP/price %, contract value/term, commissioning/open dates.
- Dates: effective/close dates, Q# YYYY.
- Ownership: insider % or $ size (material only), stake size if activist/strategic.

SPECIAL CLAMPS
- 13F/position micro-changes without activism/merger intent ⇒ low priority.
- Price-move explainers/listicles ("Why X rose", "Top Y...") ⇒ low priority unless new numbers.
- PR domains are allowed; down-prioritize only if unnumbered or likely_repeat.

REPEAT HANDLING (lightweight)
If likely_repeat=true and id ≠ primary_id, set lower priority (e.g., P4--P5) unless this item has a clearly higher source_tier than its primary.

PRIORITY BANDS (1 best → 5 lowest)
P1: hard event + concrete numbers + high relevance
P2: hard event but soft numbers OR clear economic implication (pricing/capacity) with high relevance
P3: moderate signal (contract/guide mention) with some specificity
P4: weak/remote/editorial or small ownership flows
P5: likely_repeat or listicle/recap style with no data

OUTPUT (STRICT JSON)
{{
"selected_ids": [ <id>, ... ],
"selected": [
{{ "id": <id>, "scrape_priority": 1|2|3|4|5, "likely_repeat": true|false, "repeat_key": "...", "why": "<=120 chars>", "confidence": 0.0..1.0 }}
],
"skipped": [
{{ "id": <id>, "scrape_priority": 3|4|5, "likely_repeat": true|false, "repeat_key": "...", "why": "<=120 chars>", "confidence": 0.0..1.0 }}
]
}}"""

    return _make_triage_request_full(system_prompt, payload)

def _triage_industry_articles_full(articles: List[Dict], ticker: str, sector_profile: Dict, peers: List[str]) -> List[Dict]:
    """Triage industry articles using optimized prompt - returns full results"""
    
    # Build items for API
    items = []
    for i, article in enumerate(articles):
        domain = article.get("domain", "unknown")
        source_tier = _get_domain_tier(domain, article.get("title", ""), article.get("description", ""))
        
        items.append({
            "id": i,
            "url": article.get("resolved_url") or article.get("url", ""),
            "title": article.get("title", ""),
            "domain": domain,
            "published_at": article.get("published_at").isoformat() if article.get("published_at") else "2024-01-01T00:00:00Z",
            "source_tier": round(source_tier, 2),
            "repeat_key": "",
            "likely_repeat": False,
            "primary_id": i
        })
    
    payload = {
        "bucket": "industry",
        "target_cap": 20,
        "ticker": ticker,
        "sector_profile": sector_profile,
        "items": items
    }
    
    system_prompt = f"""You are a hedge-fund news router doing PRE-SCRAPE TRIAGE for INDUSTRY items.

Assume NO article body. Use only title, domain, source_tier, and sector_profile.

RELEVANCE (INDUSTRY)
High: title explicitly mentions sector_profile.core_inputs OR benchmarks OR enacted policy/standards/regulation in sector_profile.core_geos, with plausible impact on cost/price/demand.
Medium: sector-matching but adjacent geo/channel OR credible research/indices directly tied to the sector.
Low: generic macro/sustainability platitudes, vendor puff, or unrelated sub-sectors.

EVENT SIGNALS (INDUSTRY --- different from Company/Competitor)
Policy/regulation (enacted or final): tariff|quota|CBAM|EPA|FDA|FCC|FAA|NHTSA|PJM|FERC|ITC|DOE|Treasury guidance|subsidy/credit rules|reimbursement codes|safety/standards adoption
Benchmark/input shocks: HRC|LME/COMEX metals|Henry Hub/TTF|Brent/WTI|freight indices|lithium/cobalt/uranium prices|capacity factor/curtailment metrics
Ecosystem/capacity/logistics: new plants/lines/mills|grid/transmission|spectrum allocation|port/rail constraints|export bans/quotas|waivers
Authoritative data: government/agency prints (inventory/production/PMI sub-indices), methodology changes in benchmarks

NUMERIC CUES (INDUSTRY)
- Policy: effective dates, phase-in schedules, rate/percentage levels, quota tonnage.
- Benchmarks/inputs: index level or % move, spread changes, inventory days, lead times.
- Capacity/logistics: nameplate units, capex $, commissioning timelines.
- Research/indices: month-over-month/YoY changes, diffusion index levels.

SPECIAL CLAMPS
- Remarks/op-eds without enacted dates or magnitudes → lower priority than enacted policy or benchmark prints.
- Vendor marketing/TAM-CAGR fluff without sources → low priority unless it includes sector_profile benchmarks/inputs with numbers.
- PR allowed; down-prioritize only if unnumbered or likely_repeat.

PRIORITY BANDS
P1: enacted policy with effective dates/levels OR benchmark/input shock with magnitude
P2: capacity/logistics/standardization with quantified units/timelines
P3: authoritative indices/research directly tied to sector_profile
P4: early proposals/adjacent geo/channel without numbers
P5: likely_repeat/vendor puff/unsourced TAM claims

OUTPUT (STRICT JSON)
{{
"selected_ids": [ <id>, ... ],
"selected": [
{{ "id": <id>, "scrape_priority": 1|2|3|4|5, "likely_repeat": true|false, "repeat_key": "...", "why": "<=120 chars>", "confidence": 0.0..1.0 }}
],
"skipped": [
{{ "id": <id>, "scrape_priority": 3|4|5, "likely_repeat": true|false, "repeat_key": "...", "why": "<=120 chars>", "confidence": 0.0..1.0 }}
]
}}"""

    return _make_triage_request_full(system_prompt, payload)

def _triage_competitor_articles_full(articles: List[Dict], ticker: str, peers: List[str], sector_profile: Dict) -> List[Dict]:
    """Triage competitor articles using optimized prompt - returns full results"""
    
    # Extract just ticker symbols from competitors for the whitelist
    peer_tickers = []
    for peer in peers:
        if isinstance(peer, str):
            # Extract ticker from "Name (TICKER)" format
            match = re.search(r'\(([A-Z]{1,5})\)', peer)
            if match:
                peer_tickers.append(match.group(1))
    
    # Build items for API
    items = []
    for i, article in enumerate(articles):
        domain = article.get("domain", "unknown")
        source_tier = _get_domain_tier(domain, article.get("title", ""), article.get("description", ""))
        
        items.append({
            "id": i,
            "url": article.get("resolved_url") or article.get("url", ""),
            "title": article.get("title", ""),
            "domain": domain,
            "published_at": article.get("published_at").isoformat() if article.get("published_at") else "2024-01-01T00:00:00Z",
            "source_tier": round(source_tier, 2),
            "repeat_key": "",
            "likely_repeat": False,
            "primary_id": i
        })
    
    payload = {
        "bucket": "competitor",
        "target_cap": 20,
        "ticker": ticker,
        "peers": peer_tickers,
        "sector_profile": sector_profile,
        "items": items
    }
    
    system_prompt = f"""You are a hedge-fund news router doing PRE-SCRAPE TRIAGE for COMPETITOR items.

Assume NO article body. Use only title, domain, source_tier, peers, and sector_profile.

RELEVANCE (COMPETITOR)
High: SUBJECT in peers[] AND title implies capacity|price|guidance|contract|shutdown|financing that can alter competitive dynamics (share, pricing power, cost).
Medium: adjacent peer (same sub-industry) with explicit implications.
Low: unclear peer fit or editorial.

EVENT SIGNALS (COMPETITOR)
Rival hard events: M&A|capacity add/cut|plant open|shutdown|outage|strike|price hike/cut|major customer win/loss
Financing/health: equity raise|distressed refi|covenant stress|rating cut to HY|asset sale to raise cash
Guidance/ops: guide raise/cut with numbers|ASP/mix change|cost breakthrough|input contract locked/renegotiated

NUMERIC CUES (COMPETITOR)
- Capacity/throughput units (MW/Mt/kt/lines/rigs/aircraft), utilization %, price change %, contract $/tenor, guide deltas.
- Cost/inputs if explicit (hedge locked %, surcharge %, input price %).
- Dates (commissioning/restart/effective).

SPECIAL CLAMPS
- Non-activist 13F/position stories about peers → low priority.
- PR allowed; down-prioritize if unnumbered or likely_repeat.

PRIORITY BANDS
P1: rival hard event + numbers (clear impact on competition)
P2: rival hard event with soft numbers OR clear price/capacity signal
P3: peer guidance with figures; meaningful customer wins/losses
P4: weak/remote/editorial
P5: likely_repeat

OUTPUT (STRICT JSON)
{{
"selected_ids": [ <id>, ... ],
"selected": [
{{ "id": <id>, "scrape_priority": 1|2|3|4|5, "likely_repeat": true|false, "repeat_key": "...", "why": "<=120 chars>", "confidence": 0.0..1.0 }}
],
"skipped": [
{{ "id": <id>, "scrape_priority": 3|4|5, "likely_repeat": true|false, "repeat_key": "...", "why": "<=120 chars>", "confidence": 0.0..1.0 }}
]
}}"""

    return _make_triage_request_full(system_prompt, payload)

def _make_triage_request_full(system_prompt: str, payload: Dict) -> List[Dict]:
    """Make API request to OpenAI for triage and return full results"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": OPENAI_MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload)}
            ],
            "max_completion_tokens": 1000
        }
        
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code != 200:
            LOG.error(f"OpenAI triage API error {response.status_code}: {response.text}")
            return []
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Parse the JSON response
        triage_result = json.loads(content)
        
        # Sort by priority (P1=1 is highest) then return full data
        selected = triage_result.get("selected", [])
        selected.sort(key=lambda x: x.get("scrape_priority", 5))
        
        # Return full article data with triage info
        return selected
        
    except Exception as e:
        LOG.error(f"Triage request failed: {e}")
        return []

def create_ai_evaluation_text(articles_by_ticker: Dict[str, Dict[str, List[Dict]]]) -> str:
    """Create structured text file for AI evaluation of scoring quality"""
    
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
        
        # Add enhanced metadata for each ticker
        config = get_ticker_config(ticker)
        if config:
            text_lines.append("ENHANCED METADATA:")
            text_lines.append(f"Company Name: {config.get('name', ticker)}")
            text_lines.append(f"Sector: {config.get('sector', 'N/A')}")
            text_lines.append(f"Industry: {config.get('industry', 'N/A')}")
            text_lines.append(f"Sub-Industry: {config.get('sub_industry', 'N/A')}")
            
            # Industry keywords
            if config.get("industry_keywords"):
                text_lines.append(f"Industry Keywords: {', '.join(config['industry_keywords'])}")
            
            # Competitors
            if config.get("competitors"):
                text_lines.append(f"Competitors: {', '.join(config['competitors'])}")
            
            # Sector profile
            sector_profile = config.get("sector_profile")
            if sector_profile:
                try:
                    if isinstance(sector_profile, str):
                        sector_data = json.loads(sector_profile)
                    else:
                        sector_data = sector_profile
                    
                    if sector_data.get("core_inputs"):
                        text_lines.append(f"Core Inputs: {', '.join(sector_data['core_inputs'])}")
                    if sector_data.get("core_geos"):
                        text_lines.append(f"Core Geographies: {', '.join(sector_data['core_geos'])}")
                    if sector_data.get("core_channels"):
                        text_lines.append(f"Core Channels: {', '.join(sector_data['core_channels'])}")
                except:
                    pass
            
            # Aliases and brands
            aliases_brands = config.get("aliases_brands_assets")
            if aliases_brands:
                try:
                    if isinstance(aliases_brands, str):
                        alias_data = json.loads(aliases_brands)
                    else:
                        alias_data = aliases_brands
                    
                    if alias_data.get("aliases"):
                        text_lines.append(f"Aliases: {', '.join(alias_data['aliases'])}")
                    if alias_data.get("brands"):
                        text_lines.append(f"Brands: {', '.join(alias_data['brands'])}")
                except:
                    pass
            
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
                
                if article.get('published_at'):
                    text_lines.append(f"Published: {article['published_at']}")
                
                if article.get('search_keyword'):
                    text_lines.append(f"Search Keyword: {article['search_keyword']}")
                
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
                else:
                    text_lines.append("SCRAPED CONTENT: Not available")
                
                text_lines.append("")
                text_lines.append("-" * 80)
                text_lines.append("")
    
    text_lines.append(f"\nTOTAL ARTICLES: {article_count}")
    text_lines.append("END OF EVALUATION DATA")
    
    return "\n".join(text_lines)

def get_competitor_display_name(search_keyword: str, competitor_ticker: str, ticker_metadata_cache: Dict) -> str:
    """Get full company name for competitor display in emails"""
    if not search_keyword:
        return competitor_ticker or "Unknown Competitor"
    
    # For all tickers, check if we have metadata with competitors
    for ticker, metadata in ticker_metadata_cache.items():
        competitors = metadata.get("competitors", [])
        for comp in competitors:
            if isinstance(comp, dict):
                # FIXED: Handle None values from .get() calls
                comp_name = comp.get("name") or ""
                comp_ticker = comp.get("ticker") or ""
                
                # Check if this competitor matches either by name or ticker
                if (comp_name.lower() == search_keyword.lower() or 
                    comp_ticker.lower() == (competitor_ticker or "").lower()):
                    return comp_name if comp_name else search_keyword
            else:
                # Old format - string that might contain both name and ticker
                if comp and search_keyword.lower() in comp.lower():
                    # Extract just the name part (before any parentheses)
                    name_match = re.match(r'^([^(]+)', comp)
                    if name_match:
                        return name_match.group(1).strip()
    
    # Fallback to search keyword (which should be the company name for Google feeds)
    return search_keyword

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
        """Create feeds only if under the limits - CHECK EXISTING COUNTS FIRST"""
        feeds = []
        company_name = metadata.get("company_name", ticker)
        
        LOG.info(f"CREATING FEEDS for {ticker} ({company_name}):")
        
        # Check existing feed counts
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT category, COUNT(*) as count
                FROM source_feed 
                WHERE ticker = %s AND active = TRUE
                GROUP BY category
            """, (ticker,))
            
            existing_counts = {row["category"]: row["count"] for row in cur.fetchall()}
            
            LOG.info(f"  EXISTING FEEDS: Company={existing_counts.get('company', 0)}, Industry={existing_counts.get('industry', 0)}, Competitor={existing_counts.get('competitor', 0)}")
        
        # Company feeds - always ensure we have the core 2
        existing_company_count = existing_counts.get('company', 0)
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
        
        # Industry feeds - MAX 5 TOTAL
        existing_industry_count = existing_counts.get('industry', 0)
        if existing_industry_count < 5:
            available_slots = 5 - existing_industry_count
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
            LOG.info(f"  INDUSTRY FEEDS: Skipping - already at limit (5/5)")
        
        # Competitor feeds - MAX 3 COMPETITORS (6 total feeds = 3 Google + 3 Yahoo)
        existing_competitor_count = existing_counts.get('competitor', 0)
        # Each competitor creates 2 feeds (Google + Yahoo), so divide by 2 for competitor count
        existing_competitor_entities = existing_competitor_count // 2
        
        if existing_competitor_entities < 3:
            available_competitor_slots = 3 - existing_competitor_entities
            competitors = metadata.get("competitors", [])[:available_competitor_slots]
            
            LOG.info(f"  COMPETITOR FEEDS: Can add {available_competitor_slots} more competitors (existing: {existing_competitor_entities}, available: {len(metadata.get('competitors', []))})")
            
            for comp in competitors:
                if isinstance(comp, dict):
                    comp_name = comp.get('name', '')
                    comp_ticker = comp.get('ticker')
                    
                    if comp_ticker and comp_ticker.upper() != ticker.upper() and comp_name:
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
                                "search_keyword": comp_ticker,
                                "competitor_ticker": comp_ticker
                            }
                        ]
                        feeds.extend(comp_feeds)
                        LOG.info(f"    COMPETITOR: {comp_name} ({comp_ticker}) - 2 feeds")
        else:
            LOG.info(f"  COMPETITOR FEEDS: Skipping - already at limit (3/3 competitors)")
        
        LOG.info(f"TOTAL NEW FEEDS for {ticker}: {len(feeds)}")
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
        """Unified ticker metadata management"""
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
        TickerManager.store_metadata(ticker, ai_metadata)
        return ai_metadata
    
    @staticmethod
    def store_metadata(ticker: str, metadata: Dict):
        """Store enhanced ticker metadata in database"""
        # Convert competitors to storage format
        competitors_for_db = []
        for comp in metadata.get("competitors", []):
            if isinstance(comp, dict):
                if comp.get('ticker'):
                    competitors_for_db.append(f"{comp['name']} ({comp['ticker']})")
                else:
                    competitors_for_db.append(comp['name'])
            else:
                competitors_for_db.append(str(comp))
        
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

# Global instances
ticker_manager = TickerManager()
feed_manager = FeedManager()

def generate_ticker_metadata_with_ai(ticker: str) -> Dict[str, Any]:
    """Enhanced AI metadata generation with sector profile and aliases"""
    if not OPENAI_API_KEY:
        LOG.error("OpenAI API key not configured")
        return {"industry_keywords": [], "competitors": [], "company_name": ticker}

    prompt = f"""You are a financial analyst. Return STRICT JSON ONLY.

For stock ticker "{ticker}":

1) Confirm the company and provide sector context and peers.
2) Output fields exactly as below. Avoid generic terms. Prefer GICS-style naming.
3) Peers MUST be public, primarily competing in the same core business.
4) Add sector_profile so Industry routing can reason about inputs/channels/geos/benchmarks.
5) Add aliases/brands/assets for better Company relevance.
6) Include confidence fields to allow post-run validation.

JSON schema to return:
{{
    "company_name": "Company Name",
    "ticker": "{ticker}",
    "sector": "GICS Sector", 
    "industry": "GICS Industry",
    "sub_industry": "GICS Sub-Industry",
    "industry_keywords": ["2-3 word term", "...", "...", "...", "..."],
    "competitors": [
        {{"name": "Competitor 1", "ticker": "TICK1", "confidence": 0.9}},
        {{"name": "Competitor 2", "ticker": "TICK2", "confidence": 0.8}},
        {{"name": "Competitor 3", "ticker": "TICK3", "confidence": 0.7}}
    ],
    "sector_profile": {{
        "core_inputs": ["commodity/input", "key cost driver", "..."],
        "core_channels": ["end-market channel 1", "end-market 2"],
        "core_geos": ["primary region(s) of ops/sales"],
        "benchmarks": ["index/benchmark used in pricing or regulation"]
    }},
    "aliases_brands_assets": {{
        "aliases": ["legal DBA", "common shortened names"],
        "brands": ["notable product lines/brands"], 
        "assets": ["plants/facilities/major projects if named"]
    }}
}}"""

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a financial analyst. Provide accurate stock information in valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 400,
            "response_format": {"type": "json_object"}
        }

        LOG.info(f"Generating enhanced metadata for {ticker} with single AI call")
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code != 200:
            LOG.error(f"OpenAI API error: {response.status_code}")
            return {"industry_keywords": [], "competitors": [], "company_name": ticker}

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        metadata = json.loads(content)
        
        # Validate and clean data - keep existing format for compatibility
        company_name = metadata.get("company_name", ticker)
        industry_keywords = [kw.title() for kw in metadata.get("industry_keywords", [])[:5]]
        competitors = []
        
        for comp in metadata.get("competitors", [])[:3]:
            if isinstance(comp, dict) and comp.get("name") and comp.get("ticker"):
                if comp["ticker"].upper() != ticker.upper():  # Prevent self-reference
                    competitors.append({
                        "name": comp["name"],
                        "ticker": comp["ticker"],
                        "confidence": comp.get("confidence", 0.8)
                    })
        
        # Store enhanced metadata in a way that's backward compatible
        enhanced_metadata = {
            "company_name": company_name,
            "industry_keywords": industry_keywords,
            "competitors": competitors,
            # New enhanced fields
            "sector_profile": metadata.get("sector_profile", {}),
            "aliases_brands_assets": metadata.get("aliases_brands_assets", {}),
            "sector": metadata.get("sector", ""),
            "industry": metadata.get("industry", ""),
            "sub_industry": metadata.get("sub_industry", "")
        }
        
        # ENHANCED LOGGING
        LOG.info(f"=== ENHANCED AI METADATA GENERATED for {ticker} ===")
        LOG.info(f"Company Name: {company_name}")
        LOG.info(f"Sector: {metadata.get('sector', 'N/A')}")
        LOG.info(f"Industry Keywords ({len(industry_keywords)}):")
        for i, keyword in enumerate(industry_keywords, 1):
            LOG.info(f"  {i}. {keyword}")
        
        LOG.info(f"Competitors ({len(competitors)}):")
        for i, comp in enumerate(competitors, 1):
            LOG.info(f"  {i}. {comp['name']} ({comp['ticker']}) - Confidence: {comp.get('confidence', 0.8)}")
        
        # Log enhanced fields
        sector_profile = metadata.get("sector_profile", {})
        if sector_profile:
            LOG.info(f"Sector Profile:")
            LOG.info(f"  Core Inputs: {sector_profile.get('core_inputs', [])}")
            LOG.info(f"  Core Channels: {sector_profile.get('core_channels', [])}")
            LOG.info(f"  Core Geos: {sector_profile.get('core_geos', [])}")
            LOG.info(f"  Benchmarks: {sector_profile.get('benchmarks', [])}")
        
        LOG.info(f"=== ENHANCED METADATA COMPLETE for {ticker} ===")
        
        return enhanced_metadata

    except Exception as e:
        LOG.error(f"AI metadata generation failed for {ticker}: {e}")
        return {"industry_keywords": [], "competitors": [], "company_name": ticker}

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
# Enhanced CSS styles to be added to the build_digest_html function
# Update CSS to include analyzed badge styling
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
                "competitors": competitors
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
        ".keywords { background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 11px; }",
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
        
        # Enhanced keyword information with metadata
        if ticker in ticker_metadata_cache:
            metadata = ticker_metadata_cache[ticker]
            html.append("<div class='keywords'>")
            html.append(f"<strong>🤖 AI-Powered Monitoring Keywords:</strong><br>")
            
            if metadata.get("industry_keywords"):
                industry_badges = [f'<span class="industry-badge">🏭 {kw}</span>' for kw in metadata['industry_keywords']]
                html.append(f"<strong>Industry:</strong> {' '.join(industry_badges)}<br>")
            
            if metadata.get("competitors"):
                competitor_badges = [f'<span class="competitor-badge">🏢 {comp["name"] if isinstance(comp, dict) else comp}</span>' for comp in metadata['competitors']]
                html.append(f"<strong>Competitors:</strong> {' '.join(competitor_badges)}<br>")
            
            # Enhanced metadata display
            config = get_ticker_config(ticker)
            if config:
                # Sector information
                if config.get("sector"):
                    html.append(f"<strong>Sector:</strong> <span class='sector-badge'>🏭 {config['sector']}</span><br>")
                
                # Core geographies from sector profile
                sector_profile = config.get("sector_profile")
                if sector_profile and isinstance(sector_profile, str):
                    try:
                        sector_data = json.loads(sector_profile)
                        if sector_data.get("core_geos"):
                            geo_badges = [f'<span class="geography-badge">🌍 {geo}</span>' for geo in sector_data["core_geos"][:3]]
                            html.append(f"<strong>Core Regions:</strong> {' '.join(geo_badges)}<br>")
                    except:
                        pass
                
                # Aliases/brands from enhanced metadata
                aliases_brands = config.get("aliases_brands_assets")
                if aliases_brands and isinstance(aliases_brands, str):
                    try:
                        alias_data = json.loads(aliases_brands)
                        if alias_data.get("aliases"):
                            alias_badges = [f'<span class="alias-badge">🏷️ {alias}</span>' for alias in alias_data["aliases"][:3]]
                            html.append(f"<strong>Aliases:</strong> {' '.join(alias_badges)}")
                    except:
                        pass
            
            html.append("</div>")
        
        # Company News Section
        if "company" in categories and categories["company"]:
            html.append(f"<h3>Company News ({len(categories['company'])} articles)</h3>")
            for article in categories["company"][:50]:
                html.append(_format_article_html_with_ai_summary(article, "company", ticker_metadata_cache))
        
        # Industry News Section
        if "industry" in categories and categories["industry"]:
            html.append(f"<h3>Industry & Market News ({len(categories['industry'])} articles)</h3>")
            for article in categories["industry"][:50]:
                html.append(_format_article_html_with_ai_summary(article, "industry", ticker_metadata_cache))
        
        # Competitor News Section
        if "competitor" in categories and categories["competitor"]:
            html.append(f"<h3>Competitor Intelligence ({len(categories['competitor'])} articles)</h3>")
            for article in categories["competitor"][:50]:
                html.append(_format_article_html_with_ai_summary(article, "competitor", ticker_metadata_cache))
        
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
def send_email(subject: str, html_body: str, text_attachment: str, to: str = None):
    """Send email with text file attachment for AI evaluation"""
    if not all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, EMAIL_FROM]):
        LOG.error("SMTP not fully configured")
        return False
    
    try:
        recipient = to or DIGEST_TO
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = recipient
        
        # Create the email body
        body_part = MIMEMultipart("alternative")
        body_part.attach(MIMEText("Please view this email in HTML format.", "plain"))
        body_part.attach(MIMEText(html_body, "html"))
        msg.attach(body_part)
        
        # Add text file attachment
        attachment = MIMEText(text_attachment)
        attachment.add_header('Content-Disposition', 'attachment', filename='ai_scoring_evaluation.txt')
        msg.attach(attachment)
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            if SMTP_STARTTLS:
                server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, [recipient], msg.as_string())
        
        LOG.info(f"Email with text attachment sent to {recipient}")
        return True
        
    except Exception as e:
        LOG.error(f"Email send failed: {e}")
        return False

def send_quick_ingest_email_with_triage(articles_by_ticker: Dict[str, Dict[str, List[Dict]]], triage_results: Dict[str, Dict[str, List[Dict]]]) -> bool:
    """Send quick email with domain, title, and triage results before scraping"""
    try:
        current_time_est = format_timestamp_est(datetime.now(timezone.utc))
        
        html = [
            "<html><head><style>",
            "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 13px; line-height: 1.6; color: #333; }",
            "h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }",
            "h2 { color: #34495e; margin-top: 25px; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; }",
            "h3 { color: #7f8c8d; margin-top: 15px; margin-bottom: 8px; font-size: 14px; }",
            ".article { margin: 4px 0; padding: 6px; border-radius: 3px; }",
            ".article-header { margin-bottom: 3px; }",
            ".source { color: #6c757d; font-size: 11px; font-weight: bold; display: inline-block; }",
            ".title { color: #2c3e50; margin-top: 2px; }",
            ".triage { display: inline-block; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 10px; margin-left: 8px; }",
            ".triage-selected { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }",
            ".triage-p1 { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }",
            ".triage-p2 { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }",
            ".triage-p3 { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }",
            ".triage-p4 { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }",
            ".triage-p5 { background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }",
            ".triage-skipped { background-color: #f8f9fa; color: #6c757d; border: 1px solid #dee2e6; }",
            ".company { border-left: 3px solid #27ae60; background-color: #f8fff9; }",
            ".industry { border-left: 3px solid #f39c12; background-color: #fffdf8; }",
            ".competitor { border-left: 3px solid #e74c3c; background-color: #fef9f9; }",
            ".selected-for-scrape { background-color: #e8f5e8 !important; }",
            "</style></head><body>",
            f"<h1>Quick Triage Report</h1>",
            f"<p><strong>Generated:</strong> {current_time_est}</p>",
            f"<p><strong>Status:</strong> Articles ingested and triaged, scraping and AI analysis pending...</p>",
        ]
        
        total_articles = 0
        total_selected = 0
        
        for ticker, categories in articles_by_ticker.items():
            ticker_count = sum(len(articles) for articles in categories.values())
            total_articles += ticker_count
            
            # Count selected articles for this ticker
            ticker_selected = 0
            triage_data = triage_results.get(ticker, {})
            for category in ["company", "industry", "competitor"]:
                ticker_selected += len(triage_data.get(category, []))
            total_selected += ticker_selected
            
            html.append(f"<h2>{ticker} - {ticker_count} Articles Found ({ticker_selected} selected for scraping)</h2>")
            
            for category, articles in categories.items():
                if not articles:
                    continue
                
                # Get triage results for this category
                category_triage = triage_data.get(category, [])
                selected_article_data = {item["id"]: item for item in category_triage}
                
                html.append(f"<h3>{category.title()} ({len(articles)} articles, {len(category_triage)} selected)</h3>")
                
                for idx, article in enumerate(articles[:25]):  # Show up to 25 per category
                    domain = article.get("domain", "unknown")
                    title = article.get("title", "No Title")
                    
                    # Check if this article was selected for scraping
                    is_selected = idx in selected_article_data
                    article_class = f"article {category}"
                    if is_selected:
                        article_class += " selected-for-scrape"
                    
                    # Build triage badge
                    triage_badge = ""
                    if is_selected:
                        triage_item = selected_article_data[idx]
                        priority = triage_item.get("scrape_priority", 5)
                        confidence = triage_item.get("confidence", 0.0)
                        reason = triage_item.get("why", "")
                        
                        triage_badge = f'<span class="triage triage-p{priority}" title="{reason}">Triage P{priority} ({confidence:.1f})</span>'
                    else:
                        triage_badge = '<span class="triage triage-skipped">Skipped</span>'
                    
                    html.append(f"""
                    <div class='{article_class}'>
                        <div class='article-header'>
                            <span class='source'>{get_or_create_formal_domain_name(domain)}</span>
                            {triage_badge}
                        </div>
                        <div class='title'>{title}</div>
                    </div>
                    """)
        
        html.append(f"<p><strong>Total Articles:</strong> {total_articles} (Selected for scraping: {total_selected})</p>")
        html.append(f"<p><strong>Triage Legend:</strong> P1=Highest Priority, P2=High, P3=Medium, P4=Low, P5=Lowest Priority</p>")
        html.append("</body></html>")
        
        html_content = "".join(html)
        subject = f"Quick Triage: {total_selected}/{total_articles} articles selected - processing..."
        
        return send_email(subject, html_content, "")  # Empty text attachment
        
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
    
    html = build_digest_html(articles_by_ticker, days if days > 0 else 1)
    
    tickers_str = ', '.join(articles_by_ticker.keys())
    subject = f"Stock Intelligence: {tickers_str} - {total_articles} articles"
    success = send_email(subject, html)
    
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
    Optimized ingest with AI triage workflow:
    1. Ingest all URLs from feeds (no AI processing)
    2. Perform AI triage to select scraping candidates  
    3. Send quick email with triage results
    4. Scrape selected articles with limits (20/5/5)
    5. Send final email with full analysis
    """
    require_admin(request)
    ensure_schema()
    update_schema_for_content()
    update_schema_for_triage()
    
    LOG.info("=== CRON INGEST STARTING (TRIAGE WORKFLOW) ===")
    LOG.info(f"Processing window: {minutes} minutes")
    LOG.info(f"Target tickers: {tickers or 'ALL'}")
    
    # Reset scraping stats
    reset_scraping_stats()
    
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
    
    LOG.info(f"=== PHASE 1: INGESTING ALL URLS ({len(feeds)} feeds) ===")
    
    # PHASE 1: Ingest all URLs without AI processing or scraping
    articles_by_ticker = {}
    ingest_stats = {"total_processed": 0, "total_inserted": 0, "total_duplicates": 0, "total_spam_blocked": 0}
    
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
            
        except Exception as e:
            LOG.error(f"Feed ingest failed for {feed['name']}: {e}")
            continue
    
    LOG.info(f"=== PHASE 1 COMPLETE: {ingest_stats['total_inserted']} articles ingested ===")
    
    # PHASE 2: AI Triage
    LOG.info("=== PHASE 2: AI TRIAGE ===")
    triage_results = {}
    
    for ticker in articles_by_ticker.keys():
        LOG.info(f"Running triage for {ticker}")
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
                        cur.execute("""
                            UPDATE found_url 
                            SET ai_triage_selected = TRUE, triage_priority = %s, triage_reasoning = %s
                            WHERE id = %s
                        """, (item.get("scrape_priority", 5), item.get("why", ""), article_id))
    
    # PHASE 3: Send quick email with triage results
    LOG.info("=== PHASE 3: SENDING QUICK TRIAGE EMAIL ===")
    quick_email_sent = send_quick_ingest_email_with_triage(articles_by_ticker, triage_results)
    LOG.info(f"Quick triage email sent: {quick_email_sent}")
    
    # PHASE 4: Scrape selected articles with limits
    LOG.info("=== PHASE 4: SCRAPING SELECTED ARTICLES ===")
    scraping_stats = {"scraped": 0, "failed": 0, "ai_analyzed": 0}
    
    for ticker in articles_by_ticker.keys():
        config = get_ticker_config(ticker)
        metadata = {
            "industry_keywords": config.get("industry_keywords", []) if config else [],
            "competitors": config.get("competitors", []) if config else []
        }
        
        selected = triage_results.get(ticker, {})
        
        # Company articles (limit 20)
        company_selected = selected.get("company", [])[:20]
        for item in company_selected:
            article_idx = item["id"]
            if article_idx < len(articles_by_ticker[ticker]["company"]):
                article = articles_by_ticker[ticker]["company"][article_idx]
                success = scrape_and_analyze_article(article, "company", metadata, ticker)
                if success:
                    scraping_stats["scraped"] += 1
                    scraping_stats["ai_analyzed"] += 1
                else:
                    scraping_stats["failed"] += 1
        
        # Industry articles (limit 25 total across all keywords)
        industry_selected = selected.get("industry", [])[:25]
        for item in industry_selected:
            article_idx = item["id"]
            if article_idx < len(articles_by_ticker[ticker]["industry"]):
                article = articles_by_ticker[ticker]["industry"][article_idx]
                success = scrape_and_analyze_article(article, "industry", metadata, ticker)
                if success:
                    scraping_stats["scraped"] += 1
                    scraping_stats["ai_analyzed"] += 1
                else:
                    scraping_stats["failed"] += 1
        
        # Competitor articles (limit 15 total across all competitors)
        competitor_selected = selected.get("competitor", [])[:15]
        for item in competitor_selected:
            article_idx = item["id"]
            if article_idx < len(articles_by_ticker[ticker]["competitor"]):
                article = articles_by_ticker[ticker]["competitor"][article_idx]
                success = scrape_and_analyze_article(article, "competitor", metadata, ticker)
                if success:
                    scraping_stats["scraped"] += 1
                    scraping_stats["ai_analyzed"] += 1
                else:
                    scraping_stats["failed"] += 1
    
    LOG.info(f"=== PHASE 4 COMPLETE: {scraping_stats['scraped']} articles scraped and analyzed ===")
    
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
        "workflow": "5_phase_triage_with_dual_emails",
        "phase_1_ingest": ingest_stats,
        "phase_2_triage": {
            "tickers_processed": len(triage_results),
            "selections_by_ticker": {k: {cat: len(items) for cat, items in v.items()} for k, v in triage_results.items()}
        },
        "phase_3_quick_email": {"sent": quick_email_sent},
        "phase_4_scraping": scraping_stats,
        "phase_5_final_email": final_digest_result,
        "cleanup": {"old_articles_deleted": deleted},
        "message": f"Processed {ingest_stats['total_inserted']} articles, triaged and scraped {scraping_stats['scraped']} high-priority items, sent 2 emails"
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
