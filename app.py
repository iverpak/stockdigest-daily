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
    "yahoo.com/finance", "finance.yahoo.com",
    "businesswire.com", "prnewswire.com", "globenewswire.com",
    "tipranks.com", "www.tipranks.com", "tipranks",
    "simplywall.st", "www.simplywall.st", "simplywall",
    "dailyitem.com", "www.dailyitem.com",
    "marketscreener.com", "www.marketscreener.com", "marketscreener",
    "insidermoneky.com", "seekingalpha.com/pro", "fool.com"
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
    Enhanced scraper with Playwright fallback for failed requests
    """
    global playwright_stats
    
    # First try your existing method
    content, error = safe_content_scraper(url, domain, scraped_domains)
    
    # If successful, return immediately
    if content:
        return content, f"Successfully scraped {len(content)} chars with requests"
    
    # Expanded high-value domains that justify Playwright overhead
    high_value_domains = {
        # Top JavaScript redirect failures
        'finance.yahoo.com', 'fool.com', 'barrons.com', 'tipranks.com', 
        'msn.com', 'zacks.com', 'nasdaq.com', 'theglobeandmail.com',
        'cnbc.com', 'benzinga.com', 'businessinsider.com', 'marketwatch.com',
        'investopedia.com', 'forbes.com', 'reuters.com', 'insidermonkey.com',
        'tradingview.com', 'barchart.com', 'apnews.com', 'bloomberg.com',
        
        # Top fetch failure domains (server issues)
        'investors.com', 'investing.com', 'seekingalpha.com', 'businesswire.com',
        '247wallst.com', 'theinformation.com', 'thestreet.com', 'gurufocus.com',
        
        # Other valuable financial/tech domains
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

def ingest_feed_with_content_scraping(feed: Dict, category: str = "company", keywords: List[str] = None, 
                                       enable_ai_scoring: bool = True, max_ai_articles: int = None) -> Dict[str, int]:
    """
    Enhanced feed processing with selective AI scoring and content scraping
    - enable_ai_scoring: Whether to run AI analysis on articles
    - max_ai_articles: Maximum number of articles to process with AI per feed
    """
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
        "basic_scored": 0
    }
    
    scraped_domains = set()
    ai_processed_count = 0
    
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
        
        # Sort by date (newest first), then by original order
        entries_with_dates.sort(key=lambda x: (x[1] or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        
        for entry, _ in entries_with_dates:
            stats["processed"] += 1
            
            url = getattr(entry, "link", None)
            title = getattr(entry, "title", "") or "No Title"
            raw_description = getattr(entry, "summary", "") if hasattr(entry, "summary") else ""
            
            # Filter description - only keep if it adds value
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
            
            # SPECIAL CASE: Google Feed → Yahoo Finance redirect handling
            is_google_to_yahoo = (
                "news.google.com" in url and 
                "finance.yahoo.com" in url and 
                resolved_url != url
            )
            
            if is_google_to_yahoo:
                LOG.info(f"Google→Yahoo redirect detected: {url} → {resolved_url}")
                original_source = extract_yahoo_finance_source_optimized(resolved_url)
                if original_source:
                    final_resolved_url = original_source
                    final_domain = normalize_domain(urlparse(original_source).netloc.lower())
                    final_source_url = resolved_url
                    LOG.info(f"Yahoo source extracted: {original_source}")
                else:
                    final_resolved_url = resolved_url
                    final_domain = domain
                    final_source_url = source_url
            else:
                final_resolved_url = resolved_url
                final_domain = domain
                final_source_url = source_url
            
            # Generate hash based on FINAL resolved URL
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
                        # Handle existing article re-analysis if needed
                        if (enable_ai_scoring and max_ai_articles and ai_processed_count < max_ai_articles and
                            (existing_article["ai_impact"] is None or existing_article["ai_reasoning"] is None)):
                            LOG.info(f"Re-analyzing existing article: {title[:60]}... (missing AI data)")
                            
                            quality_score, ai_impact, ai_reasoning, components = calculate_quality_score(
                                title=title,
                                domain=final_domain, 
                                ticker=feed["ticker"],
                                description=description,
                                category=category,
                                keywords=keywords
                            )
                            
                            # Extract components
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
                    
                    # Determine if this article should get full AI processing
                    should_use_ai = (enable_ai_scoring and 
                                   (max_ai_articles is None or ai_processed_count < max_ai_articles))
                    
                    # Content scraping logic - only for AI-processed articles
                    scraped_content = None
                    scraping_error = None
                    content_scraped_at = None
                    scraping_failed = False
                    
                    if should_use_ai:
                        # Determine if we should scrape content
                        should_scrape = False
                        scrape_url = None
                        
                        # For Yahoo Finance URLs (direct or via Google redirect)
                        if (final_source_url and "finance.yahoo.com" in (url if not is_google_to_yahoo else final_source_url)):
                            should_scrape = True
                            scrape_url = final_resolved_url
                            
                        # For Google News resolved URLs (non-Yahoo)
                        elif ("news.google.com" in url and final_resolved_url != url and 
                              not is_google_to_yahoo and final_resolved_url.startswith(('http://', 'https://'))):
                            should_scrape = True
                            scrape_url = final_resolved_url
                        
                        if should_scrape and scrape_url:
                            scrape_domain = normalize_domain(urlparse(scrape_url).netloc.lower())
                            
                            if scrape_domain in PAYWALL_DOMAINS:
                                stats["scraping_skipped"] += 1
                                LOG.info(f"Skipping paywall domain: {scrape_domain}")
                            else:
                                content, status = safe_content_scraper_with_playwright(scrape_url, scrape_domain, scraped_domains)
                                
                                if content:
                                    scraped_content = content
                                    content_scraped_at = datetime.now(timezone.utc)
                                    stats["content_scraped"] += 1
                                    LOG.info(f"Content scraped: {title[:60]}... ({len(content)} chars)")
                                else:
                                    scraping_failed = True
                                    scraping_error = status
                                    stats["content_failed"] += 1
                                    LOG.warning(f"Content scraping failed: {status}")
                        else:
                            stats["scraping_skipped"] += 1
                    else:
                        stats["scraping_skipped"] += 1
                    
                    # Calculate quality score - AI or fallback
                    if should_use_ai:
                        quality_score, ai_impact, ai_reasoning, components = calculate_quality_score(
                            title=title,
                            domain=final_domain, 
                            ticker=feed["ticker"],
                            description=description,
                            category=category,
                            keywords=keywords
                        )
                        
                        # Extract components
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
                        LOG.info(f"AI scored: {title[:60]}... (Score: {quality_score:.1f}, Impact: {ai_impact})")
                    else:
                        # Use basic fallback scoring - NO AI analysis
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
                        LOG.debug(f"Basic scored: {title[:60]}... (Score: {quality_score:.1f})")
                    
                    # Use scraped content for display, fallback to description
                    display_content = scraped_content if scraped_content else description
                    
                    # Insert article with appropriate scoring data
                    cur.execute("""
                        INSERT INTO found_url (
                            url, resolved_url, url_hash, title, description,
                            feed_id, ticker, domain, quality_score, published_at,
                            category, search_keyword, original_source_url,
                            scraped_content, content_scraped_at, scraping_failed, scraping_error,
                            ai_impact, ai_reasoning,
                            source_tier, event_multiplier, event_multiplier_reason,
                            relevance_boost, relevance_boost_reason, numeric_bonus,
                            penalty_multiplier, penalty_reason
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        url, final_resolved_url, url_hash, title, display_content,
                        feed["id"], feed["ticker"], final_domain, quality_score, published_at,
                        category, feed.get("search_keyword"), final_source_url,
                        scraped_content, content_scraped_at, scraping_failed, scraping_error,
                        ai_impact, ai_reasoning,
                        source_tier, event_multiplier, event_multiplier_reason,
                        relevance_boost, relevance_boost_reason, numeric_bonus,
                        penalty_multiplier, penalty_reason
                    ))
                    
                    if cur.fetchone():
                        stats["inserted"] += 1
                        processing_type = "AI analysis" if should_use_ai else "basic processing"
                        content_info = f"with content" if scraped_content else "no content"
                        source_info = "via Google→Yahoo" if is_google_to_yahoo else "direct"
                        LOG.info(f"Inserted [{category}] from {domain_resolver.get_formal_name(final_domain)}: {title[:60]}... ({processing_type}, {content_info}) ({source_info})")
                        
            except Exception as e:
                LOG.error(f"Database error for '{title[:50]}': {e}")
                continue
                
    except Exception as e:
        LOG.error(f"Feed processing error for {feed['name']}: {e}")
    
    # Log processing summary
    LOG.info(f"Feed processing complete - AI articles: {stats['ai_scored']}, Basic articles: {stats['basic_scored']}, Content scraped: {stats['content_scraped']}")
    
    return stats
    
def _format_article_html_with_content(article: Dict, category: str) -> str:
    """
    Enhanced article HTML formatting that handles both AI-analyzed and basic articles
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
    quality_score = article.get("quality_score")
    ai_impact = article.get("ai_impact")
    
    if ai_impact is not None and quality_score is not None:
        # This article was AI analyzed - show score
        score_class = "high-score" if quality_score >= 70 else "med-score" if quality_score >= 40 else "low-score"
        score_html = f'<span class="score {score_class}">Score: {quality_score:.0f}</span>'
    
    # Build metadata badges for category-specific information
    metadata_badges = []
    
    if category == "competitor" and article.get('search_keyword'):
        competitor_name = article['search_keyword']
        metadata_badges.append(f'<span class="competitor-badge">🏢 {competitor_name}</span>')
    elif category == "industry" and article.get('search_keyword'):
        industry_keyword = article['search_keyword']
        metadata_badges.append(f'<span class="industry-badge">🏭 {industry_keyword}</span>')
    
    enhanced_metadata = "".join(metadata_badges)
    
    # Get content - prioritize scraped content, fall back to description
    content_to_display = ""
    content_source_indicator = ""
    
    if article.get("scraped_content"):
        content_to_display = article["scraped_content"]
        content_source_indicator = " [ANALYZED]"
    elif article.get("description"):
        content_to_display = article["description"]
        # Only show [DESC] if this was an AI-analyzed article, otherwise show nothing
        if ai_impact is not None:
            content_source_indicator = " [DESC]"
    
    description_html = ""
    if content_to_display:
        content_clean = html.unescape(content_to_display.strip())
        content_clean = re.sub(r'<[^>]+>', '', content_clean)
        content_clean = re.sub(r'\s+', ' ', content_clean).strip()
        
        if len(content_clean) > 800:
            content_clean = content_clean[:800] + "..."
        
        content_clean = html.escape(content_clean)
        description_html = f"<br><div class='description'>{content_clean}<em>{content_source_indicator}</em></div>"
    
    return f"""
    <div class='article {category}'>
        <div class='article-header'>
            <span class='source-badge'>{display_source}</span>
            {enhanced_metadata}
            {score_html}
        </div>
        <div class='article-content'>
            <a href='{link_url}' target='_blank'>{title}</a>
            <span class='meta'> | {pub_date}</span>
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

def _ai_quality_score_company_components(title: str, domain: str, ticker: str, description: str = "", keywords: List[str] = None) -> Tuple[float, str, str, Dict]:
    """AI-powered component extraction for company articles - returns components for our calculation"""
    
    source_tier = _get_domain_tier(domain, title, description)
    desc_snippet = description[:500] if description and description.lower() != title.lower().strip() else ""
    
    schema = {
        "name": "company_news_components",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "source_tier_input": {"type": "number"},
                "event_multiplier": {"type": "number", "minimum": 0.1, "maximum": 2.0},
                "event_multiplier_reason": {"type": "string"},
                "relevance_boost": {"type": "number", "minimum": 0.5, "maximum": 1.5},
                "relevance_boost_reason": {"type": "string"},
                "numeric_bonus": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "penalty_multiplier": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                "penalty_reason": {"type": "string"},
                "impact_on_main": {"type": "string", "enum": ["Positive", "Negative", "Mixed", "Unclear"]},
                "reason_short": {"type": "string"}
            },
            "required": ["source_tier_input", "event_multiplier", "event_multiplier_reason", "relevance_boost", "relevance_boost_reason", "numeric_bonus", "penalty_multiplier", "penalty_reason", "impact_on_main", "reason_short"],
            "additionalProperties": False
        }
    }
    
    system_prompt = f"""You are a hedge-fund news scorer. Return strict JSON ONLY (no prose).
Analyze the article and provide ONLY the scoring components. Do NOT calculate the final score.

INPUTS PROVIDED:
- source_tier = {source_tier} (already calculated - use this exact value)
- title = "{title}"
- description = "{desc_snippet}"
- company = {ticker}

PROVIDE THESE COMPONENTS:

source_tier_input: Use exactly {source_tier} (provided)

event_multiplier (choose ONE that best fits title+description):
2.0 = halt/shut/delist/recall/probed/sues/settles/guides/cuts/raises/acquires/divests/spin-off
1.6 = regulatory/policy directly affecting {ticker}
1.5 = contracts/commitments/backlog announcements for {ticker}
1.4 = earnings/guidance releases for {ticker}
1.1 = analyst rating/price-target changes for {ticker}
0.6 = opinion/education pieces about {ticker}
0.5 = routine PR announcements from {ticker}

relevance_boost:
1.3 = if title/description clearly mentions {ticker}, "Vistra", or {ticker}'s specific assets/operations
1.0 = if {ticker} not directly mentioned or unclear relevance

numeric_bonus: +0.1 for each concrete number in title+description (%, $, MW figures), max +0.3

penalty_multiplier (check title+description):
0.6 = if question/listicle/prediction format
0.5 = if PR-ish announcements
1.0 = otherwise

Provide impact_on_main based on shareholder impact, reason_short ≤140 chars.
DO NOT calculate any scores - just provide the components."""

    user_payload = {
        "bucket": "company_news",
        "company_ticker": ticker,
        "title": title,
        "description_snippet": desc_snippet,
        "source_tier": source_tier,
        "keywords": keywords or []
    }
    
    return _make_ai_component_request(system_prompt, user_payload, schema, source_tier)

def _ai_quality_score_industry_components(title: str, domain: str, ticker: str, description: str = "", keywords: List[str] = None) -> Tuple[float, str, str, Dict]:
    """AI-powered component extraction for industry articles"""
    
    source_tier = _get_domain_tier(domain, title, description)
    desc_snippet = description[:500] if description and description.lower() != title.lower().strip() else ""
    
    schema = {
        "name": "industry_market_components",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "source_tier_input": {"type": "number"},
                "event_multiplier": {"type": "number", "minimum": 0.1, "maximum": 2.0},
                "event_multiplier_reason": {"type": "string"},
                "relevance_boost": {"type": "number", "minimum": 0.5, "maximum": 1.5},
                "relevance_boost_reason": {"type": "string"},
                "numeric_bonus": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "penalty_multiplier": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                "penalty_reason": {"type": "string"},
                "impact_on_main": {"type": "string", "enum": ["Positive", "Negative", "Mixed", "Unclear"]},
                "reason_short": {"type": "string"}
            },
            "required": ["source_tier_input", "event_multiplier", "event_multiplier_reason", "relevance_boost", "relevance_boost_reason", "numeric_bonus", "penalty_multiplier", "penalty_reason", "impact_on_main", "reason_short"],
            "additionalProperties": False
        }
    }
    
    system_prompt = f"""You are a hedge-fund news scorer. Return strict JSON ONLY (no prose).
Analyze the article and provide ONLY the scoring components. Do NOT calculate the final score.

INPUTS PROVIDED:
- source_tier = {source_tier} (already calculated - use this exact value)
- title = "{title}"
- description = "{desc_snippet}"
- target_company = {ticker} (energy utility)
- industry_keywords = {keywords or []}

PROVIDE THESE COMPONENTS:

source_tier_input: Use exactly {source_tier} (provided)

event_multiplier (choose ONE that best fits title+description):
1.6 = policy/regulation shaping sector economics (energy policy, utility regulations, grid standards)
1.5 = commodity/input supply-demand changes (natural gas prices, coal supply, renewable capacity)
1.4 = large ecosystem deals/capex/standards (major power plant construction, grid infrastructure, energy storage)
1.1 = research/indices (energy sector reports, utility performance studies)
0.6 = PR/market-size advertisements (industry growth predictions, market reports)

relevance_boost (examine title+description+keywords):
1.1 = if energy/utility sector topic clearly relates to {ticker}'s business (power generation, energy markets, utility operations)
1.0 = if topic seems unrelated to energy utilities or too general

numeric_bonus: +0.1 for each concrete number in title+description (%, $, MW, capacity figures), max +0.3

penalty_multiplier (check title+description):
0.6 = if question/listicle/prediction format ("Should you...", "Best...", "Top...", "Will X happen?")
0.5 = if PR-ish ("announces expansion", "market size expected to grow", "unveils breakthrough")
1.0 = otherwise

Provide impact_on_main based on implications for {ticker}, reason_short ≤140 chars.
DO NOT calculate any scores - just provide the components."""

    user_payload = {
        "bucket": "industry_market",
        "target_company_ticker": ticker,
        "title": title,
        "description_snippet": desc_snippet,
        "source_tier": source_tier,
        "industry_keywords": keywords or []
    }
    
    return _make_ai_component_request(system_prompt, user_payload, schema, source_tier)

def _ai_quality_score_competitor_components(title: str, domain: str, ticker: str, description: str = "", keywords: List[str] = None) -> Tuple[float, str, str, Dict]:
    """AI-powered component extraction for competitor articles"""
    
    source_tier = _get_domain_tier(domain, title, description)
    desc_snippet = description[:500] if description and description.lower() != title.lower().strip() else ""
    
    schema = {
        "name": "competitor_intel_components",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "source_tier_input": {"type": "number"},
                "event_multiplier": {"type": "number", "minimum": 0.1, "maximum": 2.0},
                "event_multiplier_reason": {"type": "string"},
                "relevance_boost": {"type": "number", "minimum": 0.5, "maximum": 1.5},
                "relevance_boost_reason": {"type": "string"},
                "numeric_bonus": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "penalty_multiplier": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                "penalty_reason": {"type": "string"},
                "impact_on_main": {"type": "string", "enum": ["Positive", "Negative", "Mixed", "Unclear"]},
                "reason_short": {"type": "string"}
            },
            "required": ["source_tier_input", "event_multiplier", "event_multiplier_reason", "relevance_boost", "relevance_boost_reason", "numeric_bonus", "penalty_multiplier", "penalty_reason", "impact_on_main", "reason_short"],
            "additionalProperties": False
        }
    }
    
    system_prompt = f"""You are a hedge-fund news scorer. Return strict JSON ONLY (no prose).
Analyze the article and provide ONLY the scoring components. Do NOT calculate the final score.

INPUTS PROVIDED:
- source_tier = {source_tier} (already calculated - use this exact value)
- title = "{title}"
- description = "{desc_snippet}"
- target_company = {ticker}
- competitors = {keywords or []}

PROVIDE THESE COMPONENTS:

source_tier_input: Use exactly {source_tier} (provided)

event_multiplier (choose ONE that best fits title+description):
1.7 = rival hard events (M&A, delist, shutdown, major pricing moves by competitors)
1.6 = rival capital structure/asset sales that affect competitive positioning
1.4 = rival product launches/pricing changes/strategic moves
0.9 = institutional holdings changes (13F filings, fund movements)
0.6 = opinion pieces about competitors

relevance_boost:
1.2 = if a clear {ticker} competitor is the subject and competitive implications are obvious
1.0 = if competitor connection unclear or implications for {ticker} are vague

numeric_bonus: +0.1 for each concrete number in title+description (%, $, capacity figures), max +0.3

penalty_multiplier (check title+description):
0.6 = if question/listicle/prediction format
0.5 = if PR-ish announcements
1.0 = otherwise

Provide impact_on_main based on competitive implications for {ticker}, reason_short ≤140 chars.
DO NOT calculate any scores - just provide the components."""

    user_payload = {
        "bucket": "competitor_intel",
        "target_company_ticker": ticker,
        "title": title,
        "description_snippet": desc_snippet,
        "source_tier": source_tier,
        "competitor_context": keywords or []
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
        """Store ticker metadata in database"""
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
                INSERT INTO ticker_config (ticker, name, industry_keywords, competitors, ai_generated)
                VALUES (%s, %s, %s, %s, TRUE)
                ON CONFLICT (ticker) DO UPDATE
                SET name = EXCLUDED.name, industry_keywords = EXCLUDED.industry_keywords,
                    competitors = EXCLUDED.competitors, updated_at = NOW()
            """, (
                ticker, metadata.get("company_name", ticker),
                metadata.get("industry_keywords", []), competitors_for_db
            ))

# Global instance  
ticker_manager = TickerManager()

def generate_ticker_metadata_with_ai(ticker: str) -> Dict[str, Any]:
    """Streamlined AI metadata generation with single API call"""
    if not OPENAI_API_KEY:
        LOG.error("OpenAI API key not configured")
        return {"industry_keywords": [], "competitors": [], "company_name": ticker}

    prompt = f"""For stock ticker "{ticker}", provide:
1. Full company name (without Inc/Corp unless critical)
2. Five specific industry keywords (2-3 words each, avoid generic terms)
3. Three direct public competitors with tickers

JSON format:
{{
    "company_name": "Company Name",
    "industry_keywords": ["Keyword1", "Keyword2", "Keyword3", "Keyword4", "Keyword5"],
    "competitors": [
        {{"name": "Competitor 1", "ticker": "TICK1"}},
        {{"name": "Competitor 2", "ticker": "TICK2"}},
        {{"name": "Competitor 3", "ticker": "TICK3"}}
    ]
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
            "max_tokens": 200,
            "response_format": {"type": "json_object"}
        }

        LOG.info(f"Generating metadata for {ticker} with single AI call")
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code != 200:
            LOG.error(f"OpenAI API error: {response.status_code}")
            return {"industry_keywords": [], "competitors": [], "company_name": ticker}

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        metadata = json.loads(content)
        
        # Validate and clean data
        company_name = metadata.get("company_name", ticker)
        industry_keywords = [kw.title() for kw in metadata.get("industry_keywords", [])[:5]]
        competitors = []
        
        for comp in metadata.get("competitors", [])[:3]:
            if isinstance(comp, dict) and comp.get("name") and comp.get("ticker"):
                if comp["ticker"].upper() != ticker.upper():  # Prevent self-reference
                    competitors.append(comp)
        
        # ENHANCED LOGGING
        LOG.info(f"=== AI METADATA GENERATED for {ticker} ===")
        LOG.info(f"Company Name: {company_name}")
        LOG.info(f"Industry Keywords ({len(industry_keywords)}):")
        for i, keyword in enumerate(industry_keywords, 1):
            LOG.info(f"  {i}. {keyword}")
        
        LOG.info(f"Competitors ({len(competitors)}):")
        for i, comp in enumerate(competitors, 1):
            LOG.info(f"  {i}. {comp['name']} ({comp['ticker']})")
        
        if len(competitors) < 3:
            LOG.warning(f"Only {len(competitors)} valid competitors found for {ticker}")
        
        LOG.info(f"=== METADATA COMPLETE for {ticker} ===")
        
        return {
            "company_name": company_name,
            "industry_keywords": industry_keywords,
            "competitors": competitors
        }

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

def ingest_feed(feed: Dict, category: str = "company", keywords: List[str] = None) -> Dict[str, int]:
    """Simplified feed processing with smart description filtering and AI scoring"""
    stats = {"processed": 0, "inserted": 0, "duplicates": 0, "blocked_spam": 0, "blocked_non_latin": 0}
    
    try:
        parsed = feedparser.parse(feed["url"])
        LOG.info(f"Processing feed [{category}]: {feed['name']} - {len(parsed.entries)} entries")
        
        for entry in parsed.entries:
            stats["processed"] += 1
            
            url = getattr(entry, "link", None)
            title = getattr(entry, "title", "") or "No Title"
            raw_description = getattr(entry, "summary", "") if hasattr(entry, "summary") else ""
            
            # Filter description - only keep if it adds value
            description = ""
            if raw_description and is_description_valuable(title, raw_description):
                description = raw_description[:500]  # Keep original length limit
            
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
            
            # Check for duplicates
            url_hash = get_url_hash(resolved_url)
            
            try:
                with db() as conn, conn.cursor() as cur:
                    cur.execute("SELECT id FROM found_url WHERE url_hash = %s", (url_hash,))
                    if cur.fetchone():
                        stats["duplicates"] += 1
                        continue
                    
                    # Parse publish date
                    published_at = None
                    if hasattr(entry, "published_parsed"):
                        published_at = parse_datetime(entry.published_parsed)
                    
                    # Calculate quality score using AI for company articles
                    quality_score, ai_impact, ai_reasoning = calculate_quality_score(
                        title=title,
                        domain=domain, 
                        ticker=feed["ticker"],
                        description=description,
                        category=category,
                        keywords=keywords
                    )
                    
                    # Insert article with AI-calculated score and analysis
                    cur.execute("""
                        INSERT INTO found_url (
                            url, resolved_url, url_hash, title, description,
                            feed_id, ticker, domain, quality_score, published_at,
                            category, search_keyword, original_source_url,
                            ai_impact, ai_reasoning
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        url, resolved_url, url_hash, title, description,
                        feed["id"], feed["ticker"], domain, quality_score, published_at,
                        category, feed.get("search_keyword"), source_url,
                        ai_impact, ai_reasoning
                    ))
                    
                    if cur.fetchone():
                        stats["inserted"] += 1
                        LOG.info(f"Inserted [{category}] from {domain_resolver.get_formal_name(domain)}: {title[:60]}... (Score: {quality_score:.1f})")
                        
            except Exception as e:
                LOG.error(f"Database error for '{title[:50]}': {e}")
                continue
                
    except Exception as e:
        LOG.error(f"Feed processing error for {feed['name']}: {e}")
    
    return stats
    
# ------------------------------------------------------------------------------
# Email Digest
# ------------------------------------------------------------------------------
# Enhanced CSS styles to be added to the build_digest_html function
def build_digest_html(articles_by_ticker: Dict[str, Dict[str, List[Dict]]], period_days: int) -> str:
    """Build HTML email digest using the enhanced article formatting with content indicators"""
    # Get ticker metadata for display
    ticker_metadata = {}
    for ticker in articles_by_ticker.keys():
        config = get_ticker_config(ticker)
        if config:
            ticker_metadata[ticker] = {
                "industry_keywords": config.get("industry_keywords", []),
                "competitors": config.get("competitors", [])
            }
    
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
        f"<h1>Stock Intelligence Report - Content Scraping Test</h1>",
        f"<div class='summary'>",
        f"<strong>Report Period:</strong> Last {period_days} days<br>",
        f"<strong>Generated:</strong> {current_time_est}<br>",
        f"<strong>Tickers Covered:</strong> {', '.join(articles_by_ticker.keys())}<br>",
        f"<strong>Mode:</strong> Content Scraping Test (10 articles max, AI scoring disabled)",
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
                industry_badges = [f'<span class="industry-badge">🏭 {kw}</span>' for kw in metadata['industry_keywords']]
                html.append(f"<strong>Industry:</strong> {' '.join(industry_badges)}<br>")
            if metadata.get("competitors"):
                competitor_badges = [f'<span class="competitor-badge">🏢 {comp}</span>' for comp in metadata['competitors']]
                html.append(f"<strong>Competitors:</strong> {' '.join(competitor_badges)}")
            html.append("</div>")
        
        # Company News Section
        if "company" in categories and categories["company"]:
            html.append(f"<h3>Company News ({len(categories['company'])} articles)</h3>")
            for article in categories["company"][:50]:
                html.append(_format_article_html_with_content(article, "company"))
        
        # Industry News Section
        if "industry" in categories and categories["industry"]:
            html.append(f"<h3>Industry & Market News ({len(categories['industry'])} articles)</h3>")
            for article in categories["industry"][:50]:
                html.append(_format_article_html_with_content(article, "industry"))
        
        # Competitor News Section
        if "competitor" in categories and categories["competitor"]:
            html.append(f"<h3>Competitor Intelligence ({len(categories['competitor'])} articles)</h3>")
            for article in categories["competitor"][:50]:
                html.append(_format_article_html_with_content(article, "competitor"))
        
        html.append("</div>")
    
    html.append("""
        <div style='margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; font-size: 11px; color: #6c757d;'>
            <strong>About This Test Report:</strong><br>
            • Content Scraping Test: Yahoo Finance resolved URLs are scraped for full article content<br>
            • [SCRAPED] indicator shows articles where full content was extracted<br>
            • [DESC] indicator shows fallback to original description<br>
            • AI Quality Scoring temporarily disabled to focus on content extraction<br>
            • Limited to first 10 articles per feed run for safety testing<br>
            • Domain deduplication: each domain scraped max once per run<br>
            • 3-7 second delays between scraping requests for politeness<br>
            • Only company-category Yahoo Finance articles are scraped for now
        </div>
        </body></html>
    """)
    
    return "".join(html)

# Missing function 1: Enhanced digest fetching
def fetch_digest_articles_with_content(hours: int = 24, tickers: List[str] = None) -> Dict[str, Dict[str, List[Dict]]]:
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
    
    html = build_digest_html_with_content(articles_by_ticker, days if days > 0 else 1)
    
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
    
    html = build_digest_html_with_content(articles_by_ticker, days if days > 0 else 1)
    
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
    minutes: int = Query(default=15, description="Time window in minutes - optimized for 15min cycles"),
    tickers: List[str] = Query(default=None, description="Specific tickers to ingest")
):
    """
    Optimized ingest with selective AI scoring and content scraping
    - Company articles: First 20 get full AI processing
    - Industry articles: First 5 per keyword get AI processing  
    - Competitor articles: First 5 per keyword get AI processing
    - All others: Basic processing only
    """
    require_admin(request)
    ensure_schema()
    update_schema_for_content()
    
    LOG.info("=== CRON INGEST STARTING ===")
    LOG.info(f"Processing window: {minutes} minutes")
    LOG.info(f"Target tickers: {tickers or 'ALL'}")
    
    # Get feeds for specified tickers
    with db() as conn, conn.cursor() as cur:
        if tickers:
            cur.execute("""
                SELECT id, url, name, ticker, category, retain_days, search_keyword, competitor_ticker
                FROM source_feed
                WHERE active = TRUE AND ticker = ANY(%s)
                ORDER BY ticker, 
                         CASE category 
                           WHEN 'company' THEN 1 
                           WHEN 'industry' THEN 2 
                           WHEN 'competitor' THEN 3 
                           ELSE 4 END,
                         id
            """, (tickers,))
        else:
            cur.execute("""
                SELECT id, url, name, ticker, category, retain_days, search_keyword, competitor_ticker
                FROM source_feed
                WHERE active = TRUE
                ORDER BY ticker,
                         CASE category 
                           WHEN 'company' THEN 1 
                           WHEN 'industry' THEN 2 
                           WHEN 'competitor' THEN 3 
                           ELSE 4 END,
                         id
            """)
        feeds = list(cur.fetchall())
    
    if not feeds:
        LOG.warning("No active feeds found")
        return {"status": "no_feeds", "message": "No active feeds found for specified tickers"}
    
    # ENHANCED FEED LOGGING
    feeds_by_category = {"company": 0, "industry": 0, "competitor": 0}
    for feed in feeds:
        category = feed.get("category", "company")
        feeds_by_category[category] += 1

    LOG.info(f"=== FEEDS TO PROCESS: {len(feeds)} total ===")
    LOG.info(f"  Company feeds: {feeds_by_category['company']}")  
    LOG.info(f"  Industry feeds: {feeds_by_category['industry']}")
    LOG.info(f"  Competitor feeds: {feeds_by_category['competitor']}")

    total_stats = {
        "feeds_processed": 0,
        "total_inserted": 0,
        "total_duplicates": 0,
        "total_blocked_spam": 0,
        "total_content_scraped": 0,
        "total_content_failed": 0,
        "total_scraping_skipped": 0,
        "total_ai_scored": 0,
        "total_basic_scored": 0,
        "by_ticker": {},
        "by_category": {"company": 0, "industry": 0, "competitor": 0},
        "processing_limits": {
            "company_ai_limit": 20,
            "industry_ai_limit": 5,
            "competitor_ai_limit": 5
        }
    }
    
    # Group feeds by ticker for better processing
    feeds_by_ticker = {}
    for feed in feeds:
        ticker = feed["ticker"]
        if ticker not in feeds_by_ticker:
            feeds_by_ticker[ticker] = {"company": [], "industry": [], "competitor": []}
        
        category = feed.get("category", "company")
        feeds_by_ticker[ticker][category].append(feed)
    
    # Log ticker breakdown
    LOG.info("=== FEEDS BY TICKER ===")
    for ticker in feeds_by_ticker.keys():
        ticker_feed_counts = {
            "company": len(feeds_by_ticker[ticker]["company"]),
            "industry": len(feeds_by_ticker[ticker]["industry"]), 
            "competitor": len(feeds_by_ticker[ticker]["competitor"])
        }
        LOG.info(f"  {ticker}: Company={ticker_feed_counts['company']}, Industry={ticker_feed_counts['industry']}, Competitor={ticker_feed_counts['competitor']}")
    
    # Load ticker metadata once
    ticker_metadata_cache = {}
    LOG.info("=== LOADING TICKER METADATA ===")
    for ticker in feeds_by_ticker.keys():
        config = get_ticker_config(ticker)
        if config:
            ticker_metadata_cache[ticker] = {
                "industry_keywords": config.get("industry_keywords", []),
                "competitors": config.get("competitors", [])
            }
            LOG.info(f"  {ticker}: {len(config.get('industry_keywords', []))} industry keywords, {len(config.get('competitors', []))} competitors")
        else:
            LOG.warning(f"  {ticker}: No stored metadata found - using empty keywords")
            ticker_metadata_cache[ticker] = {
                "industry_keywords": [],
                "competitors": []
            }
    
    # Process each ticker's feeds with limits
    LOG.info("=== STARTING FEED PROCESSING ===")
    for ticker, ticker_feeds in feeds_by_ticker.items():
        LOG.info(f"=== PROCESSING TICKER: {ticker} ===")
        
        metadata = ticker_metadata_cache[ticker]
        ticker_stats = {
            "inserted": 0, 
            "duplicates": 0, 
            "blocked_spam": 0, 
            "content_scraped": 0,
            "content_failed": 0,
            "scraping_skipped": 0,
            "ai_scored": 0,
            "basic_scored": 0
        }
        
        # Track AI processing counts per category for this ticker
        company_ai_count = 0
        industry_ai_counts = {}  # Track per keyword
        competitor_ai_counts = {}  # Track per keyword
        
        # Process company feeds first (highest priority for AI)
        if ticker_feeds["company"]:
            LOG.info(f"  Processing {len(ticker_feeds['company'])} company feeds...")
            
        for feed in ticker_feeds["company"]:
            enable_ai = company_ai_count < 20  # First 20 company articles get AI
            max_ai = 20 - company_ai_count if enable_ai else 0
            
            LOG.info(f"    Company Feed: {feed['name']} (AI: {'enabled' if enable_ai else 'disabled'}, max: {max_ai})")
            
            stats = ingest_feed_with_content_scraping(
                feed=feed, 
                category="company", 
                keywords=[],  # Company news doesn't need additional keywords
                enable_ai_scoring=enable_ai,
                max_ai_articles=max_ai
            )
            
            company_ai_count += stats.get("ai_scored", 0)
            _update_ticker_stats(ticker_stats, total_stats, stats, "company")
            
            LOG.info(f"      Result: AI={stats.get('ai_scored', 0)}, Basic={stats.get('basic_scored', 0)}, Inserted={stats.get('inserted', 0)}")
        
        # Process industry feeds (5 per keyword)
        if ticker_feeds["industry"]:
            LOG.info(f"  Processing {len(ticker_feeds['industry'])} industry feeds...")
            
        for feed in ticker_feeds["industry"]:
            keyword = feed.get("search_keyword", "default")
            
            if keyword not in industry_ai_counts:
                industry_ai_counts[keyword] = 0
            
            enable_ai = industry_ai_counts[keyword] < 5  # First 5 per industry keyword
            max_ai = 5 - industry_ai_counts[keyword] if enable_ai else 0
            
            LOG.info(f"    Industry Feed: {feed['name']} - Keyword: {keyword} (AI: {'enabled' if enable_ai else 'disabled'}, max: {max_ai})")
            
            stats = ingest_feed_with_content_scraping(
                feed=feed,
                category="industry",
                keywords=metadata.get("industry_keywords", []),
                enable_ai_scoring=enable_ai,
                max_ai_articles=max_ai
            )
            
            industry_ai_counts[keyword] += stats.get("ai_scored", 0)
            _update_ticker_stats(ticker_stats, total_stats, stats, "industry")
            
            LOG.info(f"      Result: AI={stats.get('ai_scored', 0)}, Basic={stats.get('basic_scored', 0)}, Inserted={stats.get('inserted', 0)}")
        
        # Process competitor feeds (5 per competitor)
        if ticker_feeds["competitor"]:
            LOG.info(f"  Processing {len(ticker_feeds['competitor'])} competitor feeds...")
            
        for feed in ticker_feeds["competitor"]:
            keyword = feed.get("search_keyword", "default")
            
            if keyword not in competitor_ai_counts:
                competitor_ai_counts[keyword] = 0
            
            enable_ai = competitor_ai_counts[keyword] < 5  # First 5 per competitor
            max_ai = 5 - competitor_ai_counts[keyword] if enable_ai else 0
            
            LOG.info(f"    Competitor Feed: {feed['name']} - Keyword: {keyword} (AI: {'enabled' if enable_ai else 'disabled'}, max: {max_ai})")
            
            stats = ingest_feed_with_content_scraping(
                feed=feed,
                category="competitor",
                keywords=metadata.get("competitors", []),
                enable_ai_scoring=enable_ai,
                max_ai_articles=max_ai
            )
            
            competitor_ai_counts[keyword] += stats.get("ai_scored", 0)
            _update_ticker_stats(ticker_stats, total_stats, stats, "competitor")
            
            LOG.info(f"      Result: AI={stats.get('ai_scored', 0)}, Basic={stats.get('basic_scored', 0)}, Inserted={stats.get('inserted', 0)}")
        
        total_stats["by_ticker"][ticker] = ticker_stats
        LOG.info(f"=== TICKER {ticker} COMPLETE ===")
        LOG.info(f"  Total AI Processed: {ticker_stats['ai_scored']}")
        LOG.info(f"  Total Basic Processed: {ticker_stats['basic_scored']}")
        LOG.info(f"  Total Articles Inserted: {ticker_stats['inserted']}")
        LOG.info(f"  Content Scraped: {ticker_stats['content_scraped']}")
    
    # Clean old articles
    LOG.info("=== CLEANING OLD ARTICLES ===")
    cutoff = datetime.now(timezone.utc) - timedelta(days=DEFAULT_RETAIN_DAYS)
    with db() as conn, conn.cursor() as cur:
        if tickers:
            cur.execute("DELETE FROM found_url WHERE found_at < %s AND ticker = ANY(%s)", (cutoff, tickers))
        else:
            cur.execute("DELETE FROM found_url WHERE found_at < %s", (cutoff,))
        deleted = cur.rowcount
    
    total_stats["old_articles_deleted"] = deleted
    LOG.info(f"Deleted {deleted} old articles (older than {DEFAULT_RETAIN_DAYS} days)")
    
    total_stats["optimization_summary"] = {
        "ai_processed_articles": total_stats["total_ai_scored"],
        "basic_processed_articles": total_stats["total_basic_scored"],
        "content_analysis_rate": f"{total_stats['total_content_scraped']}/{total_stats['total_ai_scored']}" if total_stats['total_ai_scored'] > 0 else "0/0",
        "processing_mode": "Selective AI (Company: 20, Industry: 5/keyword, Competitor: 5/keyword)"
    }
    
    LOG.info("=== CRON INGEST COMPLETE ===")
    LOG.info(f"FINAL SUMMARY:")
    LOG.info(f"  Feeds Processed: {total_stats['feeds_processed']}")
    LOG.info(f"  Articles Inserted: {total_stats['total_inserted']}")
    LOG.info(f"  AI Analyzed: {total_stats['total_ai_scored']}")
    LOG.info(f"  Basic Processed: {total_stats['total_basic_scored']}")
    LOG.info(f"  Content Scraped: {total_stats['total_content_scraped']}")
    LOG.info(f"  Processing Efficiency: {total_stats['total_ai_scored']}AI + {total_stats['total_basic_scored']}Basic = {total_stats['total_ai_scored'] + total_stats['total_basic_scored']} total processed")
    
    return total_stats

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
    """Generate and send email digest with content scraping data"""
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
