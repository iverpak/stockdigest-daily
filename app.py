import os
import sys
import time
import logging
import traceback
import hashlib
import re
import pytz
import json
import openai
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple, Set, Union
from contextlib import contextmanager
from urllib.parse import urlparse, parse_qs, unquote, quote
import csv
import io
import newspaper
from newspaper import Article
import random
from urllib.robotparser import RobotFileParser

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, Request, HTTPException, Query, Body, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

import feedparser
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from bs4 import BeautifulSoup

import base64
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from collections import defaultdict

from playwright.async_api import async_playwright
import asyncio
import signal
from contextlib import contextmanager

import os
import tracemalloc
from functools import wraps
import threading
from threading import BoundedSemaphore
from concurrent.futures import ThreadPoolExecutor, as_completed

import aiohttp

import gc
import psutil
import tracemalloc
from memory_monitor import (
    memory_monitor,
    monitor_phase,
    resource_cleanup_context,
    full_resource_cleanup
)
import yfinance as yf

# Global session for OpenAI API calls with retries
_openai_session = None

import asyncio

# Global ticker processing lock
TICKER_PROCESSING_LOCK = asyncio.Lock()

# Concurrency Configuration and Semaphores
SCRAPE_BATCH_SIZE = int(os.getenv("SCRAPE_BATCH_SIZE", "5"))
OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "5"))
CLAUDE_MAX_CONCURRENCY = int(os.getenv("CLAUDE_MAX_CONCURRENCY", "5"))  # Claude concurrency
PLAYWRIGHT_MAX_CONCURRENCY = int(os.getenv("PLAYWRIGHT_MAX_CONCURRENCY", "5"))
SCRAPFLY_MAX_CONCURRENCY = int(os.getenv("SCRAPFLY_MAX_CONCURRENCY", "5"))
TRIAGE_MAX_CONCURRENCY = int(os.getenv("TRIAGE_MAX_CONCURRENCY", "5"))

# Global semaphores for concurrent processing
OPENAI_SEM = BoundedSemaphore(OPENAI_MAX_CONCURRENCY)
CLAUDE_SEM = BoundedSemaphore(CLAUDE_MAX_CONCURRENCY)  # Claude semaphore
PLAYWRIGHT_SEM = BoundedSemaphore(PLAYWRIGHT_MAX_CONCURRENCY)
SCRAPFLY_SEM = BoundedSemaphore(SCRAPFLY_MAX_CONCURRENCY)

# Async semaphores for async operations
_async_semaphores_initialized = False
async_openai_sem = None
async_claude_sem = None  # Claude async semaphore
async_playwright_sem = None
async_scrapfly_sem = None
async_triage_sem = None

def init_async_semaphores():
    """Initialize async semaphores - called once per event loop"""
    global _async_semaphores_initialized, async_openai_sem, async_claude_sem, async_playwright_sem, async_scrapfly_sem, async_triage_sem
    if not _async_semaphores_initialized:
        async_openai_sem = asyncio.Semaphore(OPENAI_MAX_CONCURRENCY)
        async_claude_sem = asyncio.Semaphore(CLAUDE_MAX_CONCURRENCY)  # Initialize Claude semaphore
        async_playwright_sem = asyncio.Semaphore(PLAYWRIGHT_MAX_CONCURRENCY)
        async_scrapfly_sem = asyncio.Semaphore(SCRAPFLY_MAX_CONCURRENCY)
        async_triage_sem = asyncio.Semaphore(TRIAGE_MAX_CONCURRENCY)
        _async_semaphores_initialized = True

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

@contextmanager
def timeout_handler(seconds: int):
    """Context manager for timeout handling on Unix systems"""
    def timeout_signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the old signal handler and cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def clean_null_bytes(text: str) -> str:
    """Remove NULL bytes and other problematic characters that cause PostgreSQL errors"""
    if not text:
        return text
    # Remove NULL bytes, control characters, and other problematic Unicode
    cleaned = text.replace('\x00', '').replace('\0', '')
    # Remove other control characters that can cause XML/database issues
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
    return cleaned

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

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/responses")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# Anthropic Claude Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
USE_CLAUDE_FOR_SUMMARIES = os.getenv("USE_CLAUDE_FOR_SUMMARIES", "true").lower() == "true"
USE_CLAUDE_FOR_METADATA = os.getenv("USE_CLAUDE_FOR_METADATA", "true").lower() == "true"

SCRAPFLY_API_KEY = os.getenv("SCRAPFLY_API_KEY")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Personal Access Token
GITHUB_REPO = os.getenv("GITHUB_REPO")    # e.g., "username/repo-name"
GITHUB_CSV_PATH = os.getenv("GITHUB_CSV_PATH", "data/ticker_reference.csv")  # Path in repo

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
# These domains are blocked at ingestion - articles never stored
SPAM_DOMAINS = {
    # Original spam domains
    "marketbeat.com", "www.marketbeat.com", "marketbeat",
    "newser.com", "www.newser.com", "newser",
    "khodrobank.com", "www.khodrobank.com", "khodrobank",
    "defenseworld.net", "www.defenseworld.net", "defenseworld",
    "defenseworld.com", "www.defenseworld.com", "defenseworld",
    "defense-world.net", "www.defense-world.net", "defense-world",
    "defensenews.com", "www.defensenews.com", "defensenews",
    "facebook.com", "www.facebook.com", "facebook",
    # Low-quality financial sites - block at ingestion
    "msn.com", "www.msn.com",
    "tipranks.com", "www.tipranks.com",
    "simplywall.st", "www.simplywall.st",
    "sharewise.com", "www.sharewise.com",
    "stockstory.org", "www.stockstory.org",
    "news.stocktradersdaily.com", "stocktradersdaily.com", "www.stocktradersdaily.com",
    "earlytimes.in", "www.earlytimes.in",
    "investing.com", "www.investing.com",
    "fool.com", "www.fool.com", "fool.ca", "www.fool.ca",
    "marketsmojo.com", "www.marketsmojo.com",
    "stocktitan.net", "www.stocktitan.net",
    "insidermonkey.com", "www.insidermonkey.com",
    "zacks.com", "www.zacks.com",
}

QUALITY_DOMAINS = {
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "barrons.com", "cnbc.com", "marketwatch.com",
    "theglobeandmail.com",
    "apnews.com",
}

# Domains that can be ingested but NOT scraped (heavy JS/bot protection)
PROBLEMATIC_SCRAPE_DOMAINS = {
    "defenseworld.net", "defense-world.net", "defensenews.com",
    "gurufocus.com", "www.gurufocus.com",
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
    "marketwatch.com", "www.marketwatch.com",
    "bloomberg.com", "www.bloomberg.com",

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

# Domains that should skip TIER 1 (requests) and go directly to TIER 2 (Playwright)
# These are typically JavaScript-heavy sites that timeout on standard requests
SKIP_TIER1_DOMAINS = {
    'businesswire.com',  # Slow to load, always requires Playwright
}

# FIXED: Ticker-specific ingestion stats to prevent race conditions
ticker_ingestion_stats = {}

def get_ticker_ingestion_stats(ticker: str) -> dict:
    """Get or create ticker-specific ingestion stats"""
    if ticker not in ticker_ingestion_stats:
        ticker_ingestion_stats[ticker] = {
            "company_ingested": 0,
            "industry_ingested_by_keyword": {},
            "competitor_ingested_by_keyword": {},
            "limits": {
                "company": 50,
                "industry_per_keyword": 25,
                "competitor_per_keyword": 25
            }
        }
    return ticker_ingestion_stats[ticker]

# Legacy global stats for backward compatibility (deprecated)
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

# FIXED: Ticker-specific scraping stats to prevent race conditions
ticker_scraping_stats = {}

def get_ticker_scraping_stats(ticker: str) -> dict:
    """Get or create ticker-specific scraping stats"""
    if ticker not in ticker_scraping_stats:
        ticker_scraping_stats[ticker] = {
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
    return ticker_scraping_stats[ticker]

# Legacy global stats for backward compatibility (deprecated)
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


# Global Scrapfly statistics tracking  
scrapfly_stats = {
    "requests_made": 0,
    "successful": 0,
    "failed": 0,
    "cost_estimate": 0.0,
    "by_domain": defaultdict(lambda: {"attempts": 0, "successes": 0})
}


# Enhanced scraping statistics to track all methods
enhanced_scraping_stats = {
    "total_attempts": 0,
    "requests_success": 0,
    "playwright_success": 0,
    "scrapfly_success": 0,  # ADD THIS
    "total_failures": 0,
    "by_method": {
        "requests": {"attempts": 0, "successes": 0},
        "playwright": {"attempts": 0, "successes": 0},
        "scrapfly": {"attempts": 0, "successes": 0},  # ADD THIS
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
    """FRESH DATABASE - Complete schema creation for new architecture"""
    LOG.info("🔄 Creating complete database schema for NEW ARCHITECTURE")

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                -- Articles table: ticker-agnostic content storage
                CREATE TABLE IF NOT EXISTS articles (
                    id SERIAL PRIMARY KEY,
                    url_hash VARCHAR(32) UNIQUE NOT NULL,
                    url TEXT NOT NULL,
                    resolved_url TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    domain VARCHAR(255),
                    published_at TIMESTAMP,
                    scraped_content TEXT,
                    content_scraped_at TIMESTAMP,
                    scraping_failed BOOLEAN DEFAULT FALSE,
                    scraping_error TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );

                -- Add financial data columns to ticker_reference if they don't exist
                ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS financial_last_price NUMERIC(15, 2);
                ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS financial_price_change_pct NUMERIC(10, 4);
                ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS financial_yesterday_return_pct NUMERIC(10, 4);
                ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS financial_ytd_return_pct NUMERIC(10, 4);
                ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS financial_market_cap NUMERIC(20, 2);
                ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS financial_enterprise_value NUMERIC(20, 2);
                ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS financial_volume NUMERIC(15, 0);
                ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS financial_avg_volume NUMERIC(15, 0);
                ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS financial_analyst_target NUMERIC(15, 2);
                ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS financial_analyst_range_low NUMERIC(15, 2);
                ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS financial_analyst_range_high NUMERIC(15, 2);
                ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS financial_analyst_count INTEGER;
                ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS financial_analyst_recommendation VARCHAR(50);
                ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS financial_snapshot_date DATE;

                -- NEW ARCHITECTURE: Feeds table (category-neutral, shareable feeds)
                CREATE TABLE IF NOT EXISTS feeds (
                    id SERIAL PRIMARY KEY,
                    url VARCHAR(2048) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    search_keyword VARCHAR(255),
                    competitor_ticker VARCHAR(10),
                    retain_days INTEGER DEFAULT 90,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );

                -- NEW ARCHITECTURE: Ticker-Feed relationships with per-relationship categories
                CREATE TABLE IF NOT EXISTS ticker_feeds (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    feed_id INTEGER NOT NULL REFERENCES feeds(id) ON DELETE CASCADE,
                    category VARCHAR(20) NOT NULL DEFAULT 'company',
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(ticker, feed_id)
                );

                -- Ticker-Articles relationship table
                CREATE TABLE IF NOT EXISTS ticker_articles (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
                    category VARCHAR(20) DEFAULT 'company',
                    feed_id INTEGER REFERENCES feeds(id) ON DELETE SET NULL,
                    search_keyword TEXT,
                    competitor_ticker VARCHAR(10),
                    sent_in_digest BOOLEAN DEFAULT FALSE,
                    ai_summary TEXT,
                    ai_model VARCHAR(50),
                    found_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(ticker, article_id)
                );

                -- Add ai_summary and ai_model columns to ticker_articles if they don't exist (for existing databases)
                ALTER TABLE ticker_articles ADD COLUMN IF NOT EXISTS ai_summary TEXT;
                ALTER TABLE ticker_articles ADD COLUMN IF NOT EXISTS ai_model VARCHAR(50);

                -- ticker_reference table - EXACT match to CSV structure
                CREATE TABLE IF NOT EXISTS ticker_reference (
                    ticker VARCHAR(20) PRIMARY KEY,
                    country VARCHAR(5),
                    company_name VARCHAR(255),
                    industry VARCHAR(255),
                    sector VARCHAR(255),
                    sub_industry VARCHAR(255),
                    exchange VARCHAR(20),
                    currency VARCHAR(3),
                    market_cap_category VARCHAR(20),
                    active BOOLEAN DEFAULT TRUE,
                    is_etf BOOLEAN DEFAULT FALSE,
                    yahoo_ticker VARCHAR(20),
                    industry_keyword_1 VARCHAR(255),
                    industry_keyword_2 VARCHAR(255),
                    industry_keyword_3 VARCHAR(255),
                    ai_generated BOOLEAN DEFAULT FALSE,
                    ai_enhanced_at TIMESTAMP,
                    competitor_1_name VARCHAR(255),
                    competitor_1_ticker VARCHAR(20),
                    competitor_2_name VARCHAR(255),
                    competitor_2_ticker VARCHAR(20),
                    competitor_3_name VARCHAR(255),
                    competitor_3_ticker VARCHAR(20),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    data_source VARCHAR(50) DEFAULT 'csv_import',
                    last_github_sync TIMESTAMP,
                    financial_last_price NUMERIC(15, 2),
                    financial_price_change_pct NUMERIC(10, 4),
                    financial_yesterday_return_pct NUMERIC(10, 4),
                    financial_ytd_return_pct NUMERIC(10, 4),
                    financial_market_cap NUMERIC(20, 2),
                    financial_enterprise_value NUMERIC(20, 2),
                    financial_volume NUMERIC(15, 0),
                    financial_avg_volume NUMERIC(15, 0),
                    financial_analyst_target NUMERIC(15, 2),
                    financial_analyst_range_low NUMERIC(15, 2),
                    financial_analyst_range_high NUMERIC(15, 2),
                    financial_analyst_count INTEGER,
                    financial_analyst_recommendation VARCHAR(50),
                    financial_snapshot_date DATE
                );

                CREATE TABLE IF NOT EXISTS domain_names (
                    domain VARCHAR(255) PRIMARY KEY,
                    formal_name VARCHAR(255) NOT NULL,
                    ai_generated BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS competitor_metadata (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    competitor_name VARCHAR(255) NOT NULL,
                    competitor_ticker VARCHAR(20),
                    relationship_type VARCHAR(50) DEFAULT 'competitor',
                    ai_generated BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(ticker, competitor_name)
                );

                -- All indexes for performance
                CREATE INDEX IF NOT EXISTS idx_articles_url_hash ON articles(url_hash);
                CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain);
                CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at DESC);

                CREATE INDEX IF NOT EXISTS idx_feeds_url ON feeds(url);
                CREATE INDEX IF NOT EXISTS idx_feeds_active ON feeds(active);

                CREATE INDEX IF NOT EXISTS idx_ticker_feeds_ticker ON ticker_feeds(ticker);
                CREATE INDEX IF NOT EXISTS idx_ticker_feeds_feed_id ON ticker_feeds(feed_id);
                CREATE INDEX IF NOT EXISTS idx_ticker_feeds_category ON ticker_feeds(category);
                CREATE INDEX IF NOT EXISTS idx_ticker_feeds_active ON ticker_feeds(active);

                CREATE INDEX IF NOT EXISTS idx_ticker_articles_ticker ON ticker_articles(ticker);
                CREATE INDEX IF NOT EXISTS idx_ticker_articles_article_id ON ticker_articles(article_id);
                CREATE INDEX IF NOT EXISTS idx_ticker_articles_category ON ticker_articles(category);
                CREATE INDEX IF NOT EXISTS idx_ticker_articles_sent_in_digest ON ticker_articles(sent_in_digest);
                CREATE INDEX IF NOT EXISTS idx_ticker_articles_found_at ON ticker_articles(found_at DESC);

                CREATE INDEX IF NOT EXISTS idx_ticker_reference_ticker ON ticker_reference(ticker);
                CREATE INDEX IF NOT EXISTS idx_ticker_reference_active ON ticker_reference(active);
                CREATE INDEX IF NOT EXISTS idx_ticker_reference_company_name ON ticker_reference(company_name);

                -- JOB QUEUE: Batch tracking table
                CREATE TABLE IF NOT EXISTS ticker_processing_batches (
                    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    status VARCHAR(20) DEFAULT 'queued' CHECK (status IN
                        ('queued', 'processing', 'completed', 'failed', 'cancelled')),
                    total_jobs INT NOT NULL,
                    completed_jobs INT DEFAULT 0,
                    failed_jobs INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_by VARCHAR(100) DEFAULT 'api',
                    config JSONB,
                    error_summary TEXT
                );

                -- JOB QUEUE: Individual ticker job tracking
                CREATE TABLE IF NOT EXISTS ticker_processing_jobs (
                    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    batch_id UUID NOT NULL REFERENCES ticker_processing_batches(batch_id) ON DELETE CASCADE,
                    ticker VARCHAR(20) NOT NULL,

                    -- Execution state
                    status VARCHAR(20) DEFAULT 'queued' CHECK (status IN
                        ('queued', 'processing', 'completed', 'failed', 'cancelled', 'timeout')),
                    phase VARCHAR(50),
                    progress INT DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),

                    -- Results
                    result JSONB,
                    error_message TEXT,
                    error_stacktrace TEXT,

                    -- Retry logic
                    retry_count INT DEFAULT 0,
                    max_retries INT DEFAULT 2,
                    last_retry_at TIMESTAMP,

                    -- Resource tracking
                    worker_id VARCHAR(100),
                    memory_mb FLOAT,
                    duration_seconds FLOAT,

                    -- Audit trail
                    created_at TIMESTAMP DEFAULT NOW(),
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT NOW(),

                    -- Configuration
                    config JSONB,

                    -- Timeout protection
                    timeout_at TIMESTAMP
                );

                -- JOB QUEUE: Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_jobs_status_queued ON ticker_processing_jobs(status, created_at)
                    WHERE status = 'queued';
                CREATE INDEX IF NOT EXISTS idx_jobs_status_processing ON ticker_processing_jobs(status, timeout_at)
                    WHERE status = 'processing';
                CREATE INDEX IF NOT EXISTS idx_jobs_batch ON ticker_processing_jobs(batch_id);
                CREATE INDEX IF NOT EXISTS idx_jobs_ticker ON ticker_processing_jobs(ticker);
                CREATE INDEX IF NOT EXISTS idx_batches_status ON ticker_processing_batches(status, created_at);

                -- EXECUTIVE SUMMARIES: Store final AI-generated summaries per ticker per date
                CREATE TABLE IF NOT EXISTS executive_summaries (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    summary_date DATE NOT NULL,
                    summary_text TEXT NOT NULL,
                    ai_provider VARCHAR(20) NOT NULL,
                    article_ids TEXT,
                    company_articles_count INTEGER,
                    industry_articles_count INTEGER,
                    competitor_articles_count INTEGER,
                    generated_at TIMESTAMP DEFAULT NOW(),

                    UNIQUE(ticker, summary_date)
                );

                CREATE INDEX IF NOT EXISTS idx_exec_summ_ticker_date ON executive_summaries(ticker, summary_date DESC);
            """)

    LOG.info("✅ Complete database schema created successfully with NEW ARCHITECTURE + JOB QUEUE")

def update_schema_for_content():
    """Deprecated - schema already created in ensure_schema()"""
    pass

def update_schema_for_qb_scores():
    """Deprecated - schema already created in ensure_schema()"""
    pass

# Helper Functions for New Schema
def insert_article_if_new(url_hash: str, url: str, title: str, description: str,
                          domain: str, published_at: datetime, resolved_url: str = None) -> Optional[int]:
    """Insert article if it doesn't exist, return article_id"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO articles (url_hash, url, resolved_url, title, description, domain, published_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url_hash) DO UPDATE SET updated_at = NOW()
            RETURNING id
        """, (url_hash, url, resolved_url, title, description, domain, published_at))
        result = cur.fetchone()

        if result:
            return result['id']
        else:
            # Handle UPDATE case - RETURNING id is NULL for ON CONFLICT DO UPDATE
            cur.execute("SELECT id FROM articles WHERE url_hash = %s", (url_hash,))
            result = cur.fetchone()
            return result['id'] if result else None

def link_article_to_ticker(article_id: int, ticker: str, category: str = None,
                          feed_id: int = None, search_keyword: str = None,
                          competitor_ticker: str = None) -> None:
    """Create relationship between article and ticker - category is immutable after first insert"""
    with db() as conn, conn.cursor() as cur:
        if category is not None:
            # INSERT mode: Set category on first insert
            cur.execute("""
                INSERT INTO ticker_articles (ticker, article_id, category, feed_id, search_keyword, competitor_ticker)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, article_id) DO UPDATE SET
                    search_keyword = EXCLUDED.search_keyword,
                    competitor_ticker = EXCLUDED.competitor_ticker
            """, (ticker, article_id, category, feed_id, search_keyword, competitor_ticker))
        else:
            # UPDATE mode: Only update metadata, don't touch category
            cur.execute("""
                UPDATE ticker_articles
                SET search_keyword = %s, competitor_ticker = %s
                WHERE ticker = %s AND article_id = %s
            """, (search_keyword, competitor_ticker, ticker, article_id))

def update_article_content(article_id: int, scraped_content: str = None, ai_summary: str = None,
                          ai_model: str = None, scraping_failed: bool = False, scraping_error: str = None) -> None:
    """Update article with scraped content and error status (ai_summary/ai_model params ignored, kept for backward compatibility)"""
    with db() as conn, conn.cursor() as cur:
        updates = []
        params = []

        if scraped_content is not None:
            updates.append("scraped_content = %s")
            params.append(scraped_content)
            updates.append("content_scraped_at = NOW()")

        # NOTE: ai_summary and ai_model now stored in ticker_articles (POV-specific)
        # These parameters kept for backward compatibility but ignored

        if scraping_failed:
            updates.append("scraping_failed = %s")
            params.append(scraping_failed)

        if scraping_error is not None:
            updates.append("scraping_error = %s")
            params.append(scraping_error)

        updates.append("updated_at = NOW()")
        params.append(article_id)

        if updates:
            cur.execute(f"""
                UPDATE articles SET {', '.join(updates)} WHERE id = %s
            """, params)

def update_ticker_article_summary(ticker: str, article_id: int, ai_summary: str, ai_model: str) -> None:
    """Update ticker-specific AI summary (POV-based analysis)"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            UPDATE ticker_articles
            SET ai_summary = %s, ai_model = %s
            WHERE ticker = %s AND article_id = %s
        """, (ai_summary, ai_model, ticker, article_id))

def save_executive_summary(ticker: str, summary_text: str, ai_provider: str,
                          article_ids: List[int], company_count: int,
                          industry_count: int, competitor_count: int) -> None:
    """
    Store/update executive summary for ticker on current date.
    Overwrites if run multiple times same day.

    Args:
        ticker: Target company ticker (e.g., "NVDA")
        summary_text: Generated executive summary
        ai_provider: "claude" or "openai"
        article_ids: List of article IDs included in summary
        company_count: Number of company articles analyzed
        industry_count: Number of industry articles analyzed
        competitor_count: Number of competitor articles analyzed
    """
    with db() as conn, conn.cursor() as cur:
        article_ids_json = json.dumps(article_ids)

        cur.execute("""
            INSERT INTO executive_summaries
                (ticker, summary_date, summary_text, ai_provider, article_ids,
                 company_articles_count, industry_articles_count, competitor_articles_count)
            VALUES (%s, CURRENT_DATE, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, summary_date)
            DO UPDATE SET
                summary_text = EXCLUDED.summary_text,
                ai_provider = EXCLUDED.ai_provider,
                article_ids = EXCLUDED.article_ids,
                company_articles_count = EXCLUDED.company_articles_count,
                industry_articles_count = EXCLUDED.industry_articles_count,
                competitor_articles_count = EXCLUDED.competitor_articles_count,
                generated_at = NOW()
        """, (ticker, summary_text, ai_provider, article_ids_json,
              company_count, industry_count, competitor_count))

        LOG.info(f"✅ Saved executive summary for {ticker} on {datetime.now().date()} ({ai_provider}, {len(article_ids)} articles)")

# NEW FEED ARCHITECTURE V2 - Category per Relationship Functions
def upsert_feed_new_architecture(url: str, name: str, search_keyword: str = None,
                                competitor_ticker: str = None, retain_days: int = 90) -> int:
    """Insert/update feed in new architecture - NO CATEGORY (category is per-relationship)"""
    with db() as conn, conn.cursor() as cur:
        try:
            # Insert or get existing feed - NEVER overwrite existing feeds
            cur.execute("""
                INSERT INTO feeds (url, name, search_keyword, competitor_ticker, retain_days)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET
                    active = TRUE,
                    updated_at = NOW()
                RETURNING id;
            """, (url, name, search_keyword, competitor_ticker, retain_days))

            result = cur.fetchone()
            if result:
                feed_id = result['id']
                LOG.info(f"✅ Feed upserted: {name} (ID: {feed_id})")
                return feed_id
            else:
                raise Exception(f"Failed to upsert feed: {name}")

        except Exception as e:
            # Handle race condition
            if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
                LOG.warning(f"⚠️ Concurrent feed creation detected for {url}, retrieving existing feed")
                try:
                    conn.rollback()
                    cur.execute("SELECT id FROM feeds WHERE url = %s", (url,))
                    result = cur.fetchone()
                    if result:
                        feed_id = result['id']
                        LOG.info(f"✅ Retrieved existing feed: {name} (ID: {feed_id})")
                        return feed_id
                except Exception as recovery_error:
                    LOG.error(f"❌ Recovery attempt failed: {recovery_error}")
            raise e

def associate_ticker_with_feed_new_architecture(ticker: str, feed_id: int, category: str) -> bool:
    """Associate a ticker with a feed with SPECIFIC CATEGORY for this relationship"""
    with db() as conn, conn.cursor() as cur:
        try:
            cur.execute("""
                INSERT INTO ticker_feeds (ticker, feed_id, category)
                VALUES (%s, %s, %s)
                ON CONFLICT (ticker, feed_id) DO UPDATE SET
                    category = EXCLUDED.category,
                    active = TRUE,
                    updated_at = NOW()
            """, (ticker, feed_id, category))

            LOG.info(f"✅ Associated ticker {ticker} with feed {feed_id} as category '{category}'")
            return True
        except Exception as e:
            LOG.error(f"❌ Failed to associate ticker {ticker} with feed {feed_id}: {e}")
            return False

def create_feeds_for_ticker_new_architecture(ticker: str, metadata: dict) -> list:
    """Create feeds using new architecture with per-relationship categories"""
    feeds_created = []
    company_name = metadata.get("company_name", ticker)

    LOG.info(f"🔄 Creating feeds for {ticker} using NEW ARCHITECTURE (category-per-relationship)")

    # 1. Company feeds - will be associated with category="company"
    company_feeds = [
        {
            "url": f"https://news.google.com/rss/search?q=\"{company_name.replace(' ', '%20')}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
            "name": f"Google News: {company_name}",
            "search_keyword": company_name
        },
        {
            "url": f"https://finance.yahoo.com/rss/headline?s={ticker}",
            "name": f"Yahoo Finance: {ticker}",
            "search_keyword": ticker
        }
    ]

    for feed_config in company_feeds:
        try:
            feed_id = upsert_feed_new_architecture(
                url=feed_config["url"],
                name=feed_config["name"],
                search_keyword=feed_config["search_keyword"]
            )

            # Associate this feed with this ticker as "company" category
            if associate_ticker_with_feed_new_architecture(ticker, feed_id, "company"):
                feeds_created.append({
                    "feed_id": feed_id,
                    "config": {"category": "company", "name": feed_config["name"]}
                })

        except Exception as e:
            LOG.error(f"❌ Failed to create company feed for {ticker}: {e}")

    # 2. Industry feeds - will be associated with category="industry"
    industry_keywords = metadata.get("industry_keywords", [])[:3]
    for keyword in industry_keywords:
        try:
            feed_id = upsert_feed_new_architecture(
                url=f"https://news.google.com/rss/search?q=\"{keyword.replace(' ', '%20')}\"+when:7d&hl=en-US&gl=US&ceid=US:en",
                name=f"Industry: {keyword}",
                search_keyword=keyword
            )

            # Associate this feed with this ticker as "industry" category
            if associate_ticker_with_feed_new_architecture(ticker, feed_id, "industry"):
                feeds_created.append({
                    "feed_id": feed_id,
                    "config": {"category": "industry", "keyword": keyword}
                })

        except Exception as e:
            LOG.error(f"❌ Failed to create industry feed for {ticker}: {e}")

    # 3. Competitor feeds - will be associated with category="competitor"
    competitors = metadata.get("competitors", [])[:3]
    for comp in competitors:
        if isinstance(comp, dict) and comp.get('name') and comp.get('ticker'):
            comp_name = comp['name']
            comp_ticker = comp['ticker']

            try:
                # Google News competitor feed - neutral name, shareable
                feed_id = upsert_feed_new_architecture(
                    url=f"https://news.google.com/rss/search?q=\"{comp_name.replace(' ', '%20')}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
                    name=f"Google News: {comp_name}",  # Neutral name (no "Competitor:" prefix)
                    search_keyword=comp_name,
                    competitor_ticker=comp_ticker
                )

                # Associate this feed with this ticker as "competitor" category
                if associate_ticker_with_feed_new_architecture(ticker, feed_id, "competitor"):
                    feeds_created.append({
                        "feed_id": feed_id,
                        "config": {"category": "competitor", "name": comp_name}
                    })

                # Yahoo Finance competitor feed - neutral name, shareable
                feed_id = upsert_feed_new_architecture(
                    url=f"https://finance.yahoo.com/rss/headline?s={comp_ticker}",
                    name=f"Yahoo Finance: {comp_ticker}",  # Neutral name (no "Competitor:" prefix)
                    search_keyword=comp_name,
                    competitor_ticker=comp_ticker
                )

                # Associate this feed with this ticker as "competitor" category
                if associate_ticker_with_feed_new_architecture(ticker, feed_id, "competitor"):
                    feeds_created.append({
                        "feed_id": feed_id,
                        "config": {"category": "competitor", "name": comp_name}
                    })

            except Exception as e:
                LOG.error(f"❌ Failed to create competitor feeds for {ticker}: {e}")

    LOG.info(f"✅ Created {len(feeds_created)} feed associations for {ticker} using NEW ARCHITECTURE (category-per-relationship)")
    return feeds_created

def get_feeds_for_ticker_new_architecture(ticker: str) -> list:
    """Get all active feeds for a ticker with their per-relationship categories"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                f.id, f.url, f.name, f.search_keyword, f.competitor_ticker,
                tf.category, tf.active as association_active, tf.created_at as associated_at
            FROM feeds f
            JOIN ticker_feeds tf ON f.id = tf.feed_id
            WHERE tf.ticker = %s AND f.active = TRUE AND tf.active = TRUE
            ORDER BY tf.category, f.name
        """, (ticker,))

        return cur.fetchall()

def get_articles_for_ticker(ticker: str, hours: int = 24, sent_in_digest: bool = None) -> List[Dict]:
    """Get articles for a specific ticker within time window"""
    with db() as conn, conn.cursor() as cur:
        query = """
            SELECT a.*, ta.category, ta.sent_in_digest, ta.found_at, ta.competitor_ticker
            FROM articles a
            JOIN ticker_articles ta ON a.id = ta.article_id
            WHERE ta.ticker = %s
            AND ta.found_at >= NOW() - INTERVAL '%s hours'
        """
        params = [ticker, hours]

        if sent_in_digest is not None:
            query += " AND ta.sent_in_digest = %s"
            params.append(sent_in_digest)

        query += " ORDER BY a.published_at DESC"
        cur.execute(query, params)
        return cur.fetchall()

def mark_articles_sent_in_digest(ticker: str, article_ids: List[int]) -> None:
    """Mark articles as sent in digest for a specific ticker"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            UPDATE ticker_articles
            SET sent_in_digest = TRUE
            WHERE ticker = %s AND article_id = ANY(%s)
        """, (ticker, article_ids))

# Core Ticker Reference Functions
# 1. UPDATED SCHEMA - With 3 industry keyword columns and 6 competitor columns

# Helper function for backward compatibility
def ensure_ticker_reference_schema():
    """Backward compatibility wrapper - schema is now created in ensure_schema()"""
    pass  # Schema already created in ensure_schema()

def create_ticker_reference_table():
    """Backward compatibility wrapper - table is now created in ensure_schema()"""
    pass  # Table already created in ensure_schema()

# 2. INTERNATIONAL TICKER FORMAT VALIDATION
def validate_ticker_format(ticker: str) -> bool:
    """Validate ticker format supporting international exchanges and special formats"""
    if not ticker or len(ticker) > 25:  # Increased length for complex international tickers
        return False
    
    # Comprehensive regex patterns for different ticker formats
    patterns = [
        # Standard formats
        r'^[A-Z]{1,8}$',                          # US: MSFT, AAPL, GOOGL, BERKSHIRE
        r'^[A-Z0-9]{1,8}\.[A-Z]{1,4}$',          # International: RY.TO, BHP.AX, VOD.L, 005930.KS (Korean), 7203.T (Tokyo)
        
        # Special class/series formats
        r'^[A-Z]{1,6}-[A-Z]$',                   # Class shares: BRK-A, BRK-B
        r'^[A-Z]{1,6}-[A-Z]{2}$',               # Extended class: BRK-PA
        r'^[A-Z]{1,6}-[A-Z]\.[A-Z]{1,4}$',      # International class: TECK-A.TO
        
        # Rights, warrants, units
        r'^[A-Z]{1,6}\.R$',                      # Rights: AAPL.R
        r'^[A-Z]{1,6}\.W$',                      # Warrants: AAPL.W  
        r'^[A-Z]{1,6}\.U$',                      # Units: AAPL.U
        r'^[A-Z]{1,6}\.UN$',                     # Units: REI.UN (Canadian REITs)
        
        # Canadian specific formats
        r'^[A-Z]{1,6}\.TO$',                     # Toronto: RY.TO, TD.TO
        r'^[A-Z]{1,6}\.V$',                      # Vancouver: XXX.V
        r'^[A-Z]{1,6}\.CN$',                     # Canadian National: XXX.CN
        
        # Other international suffixes
        r'^[A-Z]{1,6}\.L$',                      # London: VOD.L, BP.L
        r'^[A-Z]{1,6}\.AX$',                     # Australia: BHP.AX, CBA.AX
        r'^[A-Z]{1,6}\.HK$',                     # Hong Kong: 0005.HK
        r'^\d{4}\.HK$',                          # Hong Kong numeric: 0700.HK
        
        # European formats
        r'^[A-Z]{1,6}\.DE$',                     # Germany: SAP.DE
        r'^[A-Z]{1,6}\.PA$',                     # Paris: MC.PA
        r'^[A-Z]{1,6}\.AS$',                     # Amsterdam: ASML.AS
        
        # ADR formats
        r'^[A-Z]{1,6}-ADR$',                     # ADRs: NVO-ADR
    ]
    
    ticker_upper = ticker.upper().strip()
    
    # Check against all patterns
    for pattern in patterns:
        if re.match(pattern, ticker_upper):
            return True
    
    return False

# 3. TICKER FORMAT NORMALIZATION
def normalize_ticker_format(ticker: str) -> str:
    """Normalize ticker to consistent format for storage and lookup"""
    if not ticker:
        return ""
    
    # Convert to uppercase and strip whitespace
    normalized = ticker.upper().strip()

    # CRITICAL FIX: Convert colon format to dot format BEFORE character filtering
    # Bloomberg/Reuters uses "ULVR:L", Yahoo Finance uses "ULVR.L"
    colon_to_dot_mappings = {
        ':L': '.L',      # London: ULVR:L → ULVR.L
        ':TO': '.TO',    # Toronto: RY:TO → RY.TO
        ':AX': '.AX',    # Australia: BHP:AX → BHP.AX
        ':HK': '.HK',    # Hong Kong: 0005:HK → 0005.HK
        ':DE': '.DE',    # Germany: SAP:DE → SAP.DE
        ':PA': '.PA',    # Paris: MC:PA → MC.PA
        ':AS': '.AS',    # Amsterdam: ASML:AS → ASML.AS
        ':KS': '.KS',    # Korea: 005930:KS → 005930.KS
        ':T': '.T',      # Tokyo: 7203:T → 7203.T
    }

    # Apply colon-to-dot conversions
    for colon_suffix, dot_suffix in colon_to_dot_mappings.items():
        if normalized.endswith(colon_suffix):
            normalized = normalized[:-len(colon_suffix)] + dot_suffix
            break

    # Remove quotes and invalid characters (keep only alphanumeric, dots, dashes)
    normalized = re.sub(r'[^A-Z0-9.-]', '', normalized)
    
    # Handle common exchange variations and map to Yahoo Finance standard
    exchange_mappings = {
        # Toronto Stock Exchange variations
        '.TSX': '.TO',
        '.TSE': '.TO', 
        '.TOR': '.TO',
        
        # Vancouver variations  
        '.VAN': '.V',
        '.VSE': '.V',
        
        # London variations
        '.LSE': '.L',
        '.LON': '.L',
        
        # Australian variations
        '.ASX': '.AX',
        '.AUS': '.AX',
        
        # German variations
        '.FRA': '.DE',
        '.XETRA': '.DE',
        
        # US exchange suffixes to remove
        '.NYSE': '',
        '.NASDAQ': '',
        '.OTC': '',
    }
    
    # Apply exchange mappings
    for old_suffix, new_suffix in exchange_mappings.items():
        if normalized.endswith(old_suffix):
            normalized = normalized[:-len(old_suffix)] + new_suffix
            break
    
    # Handle edge cases
    edge_case_mappings = {
        # Berkshire Hathaway
        'BRKA': 'BRK-A',
        'BRKB': 'BRK-B',
        
        # Common Canadian bank shortcuts
        'ROYALBANK.TO': 'RY.TO',
        'SCOTIABANK.TO': 'BNS.TO',
    }
    
    # Apply edge case mappings
    if normalized in edge_case_mappings:
        normalized = edge_case_mappings[normalized]
    
    # Replace underscores with dashes in base ticker only
    if '.' in normalized:
        parts = normalized.split('.')
        parts[0] = parts[0].replace('_', '-')
        normalized = '.'.join(parts)
    else:
        normalized = normalized.replace('_', '-')
    
    return normalized

# 4. TICKER VALIDATION HELPER FUNCTIONS
def get_ticker_exchange_info(ticker: str) -> Dict[str, str]:
    """Extract exchange and country information from ticker format"""
    normalized = normalize_ticker_format(ticker)
    
    exchange_info = {
        'ticker': normalized,
        'exchange': 'UNKNOWN',
        'country': 'UNKNOWN',
        'currency': 'UNKNOWN'
    }
    
    # Exchange mappings based on ticker suffix
    if '.TO' in normalized:
        exchange_info.update({'exchange': 'TSX', 'country': 'CA', 'currency': 'CAD'})
    elif '.V' in normalized:
        exchange_info.update({'exchange': 'TSXV', 'country': 'CA', 'currency': 'CAD'})
    elif '.L' in normalized:
        exchange_info.update({'exchange': 'LSE', 'country': 'UK', 'currency': 'GBP'})
    elif '.AX' in normalized:
        exchange_info.update({'exchange': 'ASX', 'country': 'AU', 'currency': 'AUD'})
    elif '.DE' in normalized:
        exchange_info.update({'exchange': 'XETRA', 'country': 'DE', 'currency': 'EUR'})
    elif '.PA' in normalized:
        exchange_info.update({'exchange': 'EPA', 'country': 'FR', 'currency': 'EUR'})
    elif '.HK' in normalized:
        exchange_info.update({'exchange': 'HKEX', 'country': 'HK', 'currency': 'HKD'})
    elif '.AS' in normalized:
        exchange_info.update({'exchange': 'AEX', 'country': 'NL', 'currency': 'EUR'})
    elif '.KS' in normalized:
        exchange_info.update({'exchange': 'KRX', 'country': 'KR', 'currency': 'KRW'})
    elif '.T' in normalized:
        exchange_info.update({'exchange': 'TSE', 'country': 'JP', 'currency': 'JPY'})
    else:
        # Assume US market for tickers without suffix
        exchange_info.update({'exchange': 'NASDAQ/NYSE', 'country': 'US', 'currency': 'USD'})
    
    return exchange_info

# 5. COMPREHENSIVE TICKER TESTING FUNCTION
def test_ticker_validation():
    """Test function to verify ticker validation works correctly"""
    test_cases = [
        # US Markets
        ('AAPL', True), ('MSFT', True), ('GOOGL', True), ('BRK-A', True), ('BRK-B', True),
        
        # Canadian Markets
        ('RY.TO', True), ('TD.TO', True), ('BNS.TO', True), ('BMO.TO', True), ('TECK-A.TO', True),
        ('SHOP.TO', True), ('CNQ.TO', True), ('ENB.TO', True), ('BCE.TO', True),
        
        # International Markets  
        ('VOD.L', True), ('BP.L', True), ('BHP.AX', True), ('SAP.DE', True),
        ('ASML.AS', True), ('MC.PA', True), ('0700.HK', True),
        
        # Special formats
        ('BRK-PA', True), ('TECK-A.TO', True), ('RCI-B.TO', True),
        
        # Invalid formats
        ('', False), ('TOOLONG12345', False), ('12345', False), ('ABC.INVALID', False),
        ('SPECIAL!@#', False), ('A', False),  # Too short single letter
    ]
    
    passed = 0
    failed = 0
    
    LOG.info("Testing ticker validation...")
    
    for ticker, expected in test_cases:
        result = validate_ticker_format(ticker)
        if result == expected:
            passed += 1
        else:
            failed += 1
            LOG.warning(f"VALIDATION TEST FAILED: {ticker} - Expected: {expected}, Got: {result}")
    
    LOG.info(f"Ticker validation tests: {passed} passed, {failed} failed")
    
    # Test normalization
    normalization_tests = [
        ('ry.to', 'RY.TO'),
        ('BRK A', 'BRK-A'),  
        ('TECK_A.TSX', 'TECK-A.TO'),
        ('  AAPL  ', 'AAPL'),
        ('brk-b', 'BRK-B'),
    ]
    
    LOG.info("Testing ticker normalization...")
    norm_passed = 0
    norm_failed = 0
    
    for input_ticker, expected_output in normalization_tests:
        result = normalize_ticker_format(input_ticker)
        if result == expected_output:
            norm_passed += 1
        else:
            norm_failed += 1
            LOG.warning(f"NORMALIZATION TEST FAILED: '{input_ticker}' - Expected: '{expected_output}', Got: '{result}'")
    
    LOG.info(f"Ticker normalization tests: {norm_passed} passed, {norm_failed} failed")
    
    return {"validation": {"passed": passed, "failed": failed}, "normalization": {"passed": norm_passed, "failed": norm_failed}}

# Optional: Add this function to test the implementation
def debug_ticker_processing(ticker: str):
    """Debug function to see how a ticker gets processed through the system"""
    LOG.info(f"=== DEBUGGING TICKER: {ticker} ===")
    LOG.info(f"Original: '{ticker}'")
    
    normalized = normalize_ticker_format(ticker)
    LOG.info(f"Normalized: '{normalized}'")
    
    is_valid = validate_ticker_format(normalized)
    LOG.info(f"Valid: {is_valid}")
    
    exchange_info = get_ticker_exchange_info(normalized)
    LOG.info(f"Exchange Info: {exchange_info}")
    
    return {
        'original': ticker,
        'normalized': normalized,
        'valid': is_valid,
        'exchange_info': exchange_info
    }

# Ticker Reference Data Management Functions
# 1. GET TICKER REFERENCE - Enhanced lookup with fallback logic
def get_ticker_reference(ticker: str):
    """Get ticker reference data from database with US/Canadian fallback logic"""
    if not ticker:
        return None
        
    normalized_ticker = normalize_ticker_format(ticker)
    
    with db() as conn, conn.cursor() as cur:
        # First try exact match
        cur.execute("""
            SELECT ticker, country, company_name, industry, sector, sub_industry,
                   exchange, currency, market_cap_category, yahoo_ticker, active, is_etf,
                   industry_keyword_1, industry_keyword_2, industry_keyword_3,
                   competitor_1_name, competitor_1_ticker,
                   competitor_2_name, competitor_2_ticker,
                   competitor_3_name, competitor_3_ticker,
                   ai_generated, ai_enhanced_at, created_at, updated_at, data_source
            FROM ticker_reference
            WHERE ticker = %s AND active = TRUE
        """, (normalized_ticker,))
        result = cur.fetchone()
        
        if result:
            return dict(result)
        
        # Fallback logic for Canadian tickers
        if not normalized_ticker.endswith('.TO') and len(normalized_ticker) <= 5:
            canadian_ticker = f"{normalized_ticker}.TO"
            cur.execute("""
                SELECT ticker, country, company_name, industry, sector, sub_industry,
                       exchange, currency, market_cap_category, yahoo_ticker, active, is_etf,
                       industry_keyword_1, industry_keyword_2, industry_keyword_3,
                       competitor_1_name, competitor_1_ticker,
                       competitor_2_name, competitor_2_ticker,
                       competitor_3_name, competitor_3_ticker,
                       ai_generated, ai_enhanced_at, created_at, updated_at, data_source
                FROM ticker_reference
                WHERE ticker = %s AND active = TRUE
            """, (canadian_ticker,))
            result = cur.fetchone()
            if result:
                return dict(result)
        
        # Fallback logic for US tickers
        if '.TO' in normalized_ticker:
            us_ticker = normalized_ticker.replace('.TO', '')
            cur.execute("""
                SELECT ticker, country, company_name, industry, sector, sub_industry,
                       exchange, currency, market_cap_category, yahoo_ticker, active, is_etf,
                       industry_keyword_1, industry_keyword_2, industry_keyword_3,
                       competitor_1_name, competitor_1_ticker,
                       competitor_2_name, competitor_2_ticker,
                       competitor_3_name, competitor_3_ticker,
                       ai_generated, ai_enhanced_at, created_at, updated_at, data_source
                FROM ticker_reference
                WHERE ticker = %s AND active = TRUE
            """, (us_ticker,))
            result = cur.fetchone()
            if result:
                LOG.info(f"Found US listing {us_ticker} for Canadian ticker {normalized_ticker}")
                return dict(result)
    
    return None

# ============================================================================
# FINANCIAL DATA FUNCTIONS (yfinance integration)
# ============================================================================

def format_financial_number(num):
    """Format large numbers with B/M/K suffixes"""
    if num is None or num == 0:
        return None

    try:
        num = float(num)
        if num >= 1e12:
            return f"${num/1e12:.2f}T"
        elif num >= 1e9:
            return f"${num/1e9:.2f}B"
        elif num >= 1e6:
            return f"${num/1e6:.2f}M"
        elif num >= 1e3:
            return f"${num/1e3:.2f}K"
        else:
            return f"${num:.2f}"
    except:
        return None

def format_financial_volume(num):
    """Format volume without dollar sign"""
    if num is None or num == 0:
        return None

    try:
        num = float(num)
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return f"{num:,.0f}"
    except:
        return None

def format_financial_percent(value, include_plus=True):
    """Format percentage with + or - sign"""
    if value is None:
        return None

    try:
        value = float(value)
        if value > 0 and include_plus:
            return f"+{value:.2f}%"
        return f"{value:.2f}%"
    except:
        return None

def get_stock_context(ticker: str, retries: int = 3, timeout: int = 10) -> Optional[Dict]:
    """
    Fetch financial data from yfinance with retry logic and timeout.
    Returns dict with 13 financial fields + snapshot_date, or None on failure.
    All-or-nothing: if critical fields missing, returns None.
    """

    for attempt in range(retries):
        try:
            LOG.info(f"Fetching financial data for {ticker} (attempt {attempt + 1}/{retries})")

            # Create a wrapper with timeout using threading
            result = {'data': None, 'error': None}

            def fetch_data():
                try:
                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.info
                    hist = ticker_obj.history(period="ytd")
                    result['data'] = (info, hist)
                except Exception as e:
                    result['error'] = e

            # Start fetch in thread with timeout
            fetch_thread = threading.Thread(target=fetch_data)
            fetch_thread.daemon = True
            fetch_thread.start()
            fetch_thread.join(timeout=timeout)

            if fetch_thread.is_alive():
                LOG.warning(f"yfinance timeout for {ticker} after {timeout}s (attempt {attempt + 1})")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return None

            # Check for errors
            if result['error']:
                raise result['error']

            if not result['data']:
                raise ValueError("No data returned from yfinance")

            info, hist = result['data']

            # Validate we got real data
            if not info or not isinstance(info, dict):
                raise ValueError(f"Invalid info data for {ticker}")

            # Get price data
            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            if not current_price:
                raise ValueError(f"No price data available for {ticker}")

            # Calculate price change percentage
            regular_market_change = info.get('regularMarketChangePercent', 0.0)

            # Calculate yesterday's return
            previous_close = info.get('previousClose')
            regular_market_previous_close = info.get('regularMarketPreviousClose')
            yesterday_return = None

            if previous_close and regular_market_previous_close:
                yesterday_return = ((previous_close - regular_market_previous_close) / regular_market_previous_close) * 100

            # Calculate YTD return
            ytd_return = None
            try:
                if not hist.empty and len(hist) > 0:
                    ytd_start = hist['Close'].iloc[0]
                    ytd_current = previous_close if previous_close else current_price
                    ytd_return = ((ytd_current - ytd_start) / ytd_start) * 100
            except Exception as e:
                LOG.warning(f"YTD calculation failed for {ticker}: {e}")

            # Get financial data
            market_cap = info.get('marketCap')
            enterprise_value = info.get('enterpriseValue')

            # Get volume data
            volume = info.get('volume')
            avg_volume = info.get('averageVolume')

            # Get analyst data
            target_mean = info.get('targetMeanPrice')
            target_low = info.get('targetLowPrice')
            target_high = info.get('targetHighPrice')
            num_analysts = info.get('numberOfAnalystOpinions')
            recommendation = info.get('recommendationKey', '').capitalize() if info.get('recommendationKey') else None

            # Validate critical fields (all-or-nothing)
            if not current_price or not market_cap:
                raise ValueError(f"Missing critical financial fields for {ticker}")

            # Build financial data dict
            financial_data = {
                'financial_last_price': float(current_price) if current_price else None,
                'financial_price_change_pct': float(regular_market_change) if regular_market_change else None,
                'financial_yesterday_return_pct': float(yesterday_return) if yesterday_return else None,
                'financial_ytd_return_pct': float(ytd_return) if ytd_return else None,
                'financial_market_cap': float(market_cap) if market_cap else None,
                'financial_enterprise_value': float(enterprise_value) if enterprise_value else None,
                'financial_volume': float(volume) if volume else None,
                'financial_avg_volume': float(avg_volume) if avg_volume else None,
                'financial_analyst_target': float(target_mean) if target_mean else None,
                'financial_analyst_range_low': float(target_low) if target_low else None,
                'financial_analyst_range_high': float(target_high) if target_high else None,
                'financial_analyst_count': int(num_analysts) if num_analysts else None,
                'financial_analyst_recommendation': recommendation,
                'financial_snapshot_date': datetime.now(pytz.timezone('America/Toronto')).strftime('%Y-%m-%d')
            }

            LOG.info(f"✅ Financial data retrieved for {ticker}: Price=${current_price:.2f}, MCap={format_financial_number(market_cap)}")
            return financial_data

        except Exception as e:
            LOG.warning(f"yfinance attempt {attempt + 1}/{retries} failed for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
            else:
                LOG.error(f"❌ yfinance failed after {retries} attempts for {ticker}")
                return None

    return None

# Backwards compatability ticker_reference to ticker_config
def get_ticker_config(ticker: str) -> Optional[Dict]:
    """Get ticker configuration from ticker_reference table with proper field conversion"""
    LOG.info(f"[DB_DEBUG] get_ticker_config() called with ticker='{ticker}'")

    with db() as conn, conn.cursor() as cur:
        # Check if ticker_reference table exists first (safe for fresh database)
        try:
            cur.execute("SELECT 1 FROM ticker_reference LIMIT 1")
        except Exception as e:
            LOG.info(f"[DB_DEBUG] ticker_reference table doesn't exist yet (fresh database): {e}")
            return None

        # Add debug query first
        cur.execute("SELECT COUNT(*) as count FROM ticker_reference WHERE ticker = %s", (ticker,))
        count_result = cur.fetchone()
        LOG.info(f"[DB_DEBUG] Found {count_result['count'] if count_result else 0} records for ticker '{ticker}'")
        
        cur.execute("""
            SELECT ticker, company_name,
                   industry_keyword_1, industry_keyword_2, industry_keyword_3,
                   competitor_1_name, competitor_1_ticker,
                   competitor_2_name, competitor_2_ticker,
                   competitor_3_name, competitor_3_ticker,
                   sector, industry, sub_industry,
                   financial_last_price, financial_price_change_pct,
                   financial_yesterday_return_pct, financial_ytd_return_pct,
                   financial_market_cap, financial_enterprise_value,
                   financial_volume, financial_avg_volume,
                   financial_analyst_target, financial_analyst_range_low,
                   financial_analyst_range_high, financial_analyst_count,
                   financial_analyst_recommendation, financial_snapshot_date
            FROM ticker_reference
            WHERE ticker = %s
        """, (ticker,))
        
        result = cur.fetchone()
        LOG.info(f"[DB_DEBUG] Raw database result for '{ticker}': {result}")
        if not result:
            LOG.info(f"[DB_DEBUG] No result found for ticker '{ticker}'")
            return None
        
        # Convert 3 separate keyword fields back to array format
        industry_keywords = []
        for i in range(1, 4):
            keyword = result.get(f"industry_keyword_{i}")
            if keyword and keyword.strip():
                industry_keywords.append(keyword.strip())
        
        # Convert 6 separate competitor fields back to structured format
        competitors = []
        for i in range(1, 4):
            name = result.get(f"competitor_{i}_name")
            ticker_field = result.get(f"competitor_{i}_ticker")
            if name and name.strip():
                comp = {"name": name.strip()}
                if ticker_field and ticker_field.strip():
                    comp["ticker"] = ticker_field.strip()
                competitors.append(comp)
        
        config = {
            "name": result["company_name"],
            "company_name": result["company_name"],  # Some functions expect this field name
            "industry_keywords": industry_keywords,
            "competitors": competitors,
            "sector": result.get("sector", ""),
            "industry": result.get("industry", ""),
            "sub_industry": result.get("sub_industry", "")
        }

        # Add financial data if available
        if result.get("financial_snapshot_date"):
            config.update({
                "financial_last_price": float(result["financial_last_price"]) if result.get("financial_last_price") else None,
                "financial_price_change_pct": float(result["financial_price_change_pct"]) if result.get("financial_price_change_pct") else None,
                "financial_yesterday_return_pct": float(result["financial_yesterday_return_pct"]) if result.get("financial_yesterday_return_pct") else None,
                "financial_ytd_return_pct": float(result["financial_ytd_return_pct"]) if result.get("financial_ytd_return_pct") else None,
                "financial_market_cap": float(result["financial_market_cap"]) if result.get("financial_market_cap") else None,
                "financial_enterprise_value": float(result["financial_enterprise_value"]) if result.get("financial_enterprise_value") else None,
                "financial_volume": float(result["financial_volume"]) if result.get("financial_volume") else None,
                "financial_avg_volume": float(result["financial_avg_volume"]) if result.get("financial_avg_volume") else None,
                "financial_analyst_target": float(result["financial_analyst_target"]) if result.get("financial_analyst_target") else None,
                "financial_analyst_range_low": float(result["financial_analyst_range_low"]) if result.get("financial_analyst_range_low") else None,
                "financial_analyst_range_high": float(result["financial_analyst_range_high"]) if result.get("financial_analyst_range_high") else None,
                "financial_analyst_count": int(result["financial_analyst_count"]) if result.get("financial_analyst_count") else None,
                "financial_analyst_recommendation": result.get("financial_analyst_recommendation"),
                "financial_snapshot_date": str(result["financial_snapshot_date"]) if result.get("financial_snapshot_date") else None
            })

        return config

# 2. STORE TICKER REFERENCE - With 6 competitor fields
def store_ticker_reference(ticker_data: dict) -> bool:
    """Store or update ticker reference data with 3 industry keywords + 6 competitor fields"""
    try:
        # Validate required fields
        required_fields = ['ticker', 'country', 'company_name']
        for field in required_fields:
            if not ticker_data.get(field):
                LOG.warning(f"Missing required field '{field}' for ticker reference")
                return False
        
        # Normalize ticker format
        ticker_data['ticker'] = normalize_ticker_format(ticker_data['ticker'])
        
        # Validate ticker format
        if not validate_ticker_format(ticker_data['ticker']):
            LOG.warning(f"Invalid ticker format: {ticker_data['ticker']}")
            return False
        
        # Set yahoo_ticker if not provided
        if not ticker_data.get('yahoo_ticker'):
            ticker_data['yahoo_ticker'] = ticker_data['ticker']
        
        # Clean text fields to remove NULL bytes
        text_fields = [
            'company_name', 'industry', 'sector', 'sub_industry', 'exchange',
            'industry_keyword_1', 'industry_keyword_2', 'industry_keyword_3',
            'competitor_1_name', 'competitor_2_name', 'competitor_3_name',
            'competitor_1_ticker', 'competitor_2_ticker', 'competitor_3_ticker'
        ]
        for field in text_fields:
            if ticker_data.get(field):
                ticker_data[field] = clean_null_bytes(str(ticker_data[field]))
        
        # Normalize competitor tickers
        competitor_ticker_fields = ['competitor_1_ticker', 'competitor_2_ticker', 'competitor_3_ticker']
        for field in competitor_ticker_fields:
            if ticker_data.get(field):
                ticker_data[field] = normalize_ticker_format(ticker_data[field])
                # Validate competitor ticker format
                if not validate_ticker_format(ticker_data[field]):
                    LOG.warning(f"Invalid competitor ticker format: {ticker_data[field]}")
                    ticker_data[field] = None  # Clear invalid ticker
        
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ticker_reference (
                    ticker, country, company_name, industry, sector, sub_industry,
                    exchange, currency, market_cap_category, yahoo_ticker, active, is_etf,
                    industry_keyword_1, industry_keyword_2, industry_keyword_3,
                    competitor_1_name, competitor_1_ticker,
                    competitor_2_name, competitor_2_ticker,
                    competitor_3_name, competitor_3_ticker,
                    ai_generated, data_source
                ) VALUES (
                    %(ticker)s, %(country)s, %(company_name)s, %(industry)s, %(sector)s, %(sub_industry)s,
                    %(exchange)s, %(currency)s, %(market_cap_category)s, %(yahoo_ticker)s, %(active)s, %(is_etf)s,
                    %(industry_keyword_1)s, %(industry_keyword_2)s, %(industry_keyword_3)s,
                    %(competitor_1_name)s, %(competitor_1_ticker)s,
                    %(competitor_2_name)s, %(competitor_2_ticker)s,
                    %(competitor_3_name)s, %(competitor_3_ticker)s,
                    %(ai_generated)s, %(data_source)s
                )
                ON CONFLICT (ticker) DO UPDATE SET
                    country = EXCLUDED.country,
                    company_name = EXCLUDED.company_name,
                    industry = EXCLUDED.industry,
                    sector = EXCLUDED.sector,
                    sub_industry = EXCLUDED.sub_industry,
                    exchange = EXCLUDED.exchange,
                    currency = EXCLUDED.currency,
                    market_cap_category = EXCLUDED.market_cap_category,
                    yahoo_ticker = EXCLUDED.yahoo_ticker,
                    active = EXCLUDED.active,
                    is_etf = EXCLUDED.is_etf,
                    industry_keyword_1 = EXCLUDED.industry_keyword_1,
                    industry_keyword_2 = EXCLUDED.industry_keyword_2,
                    industry_keyword_3 = EXCLUDED.industry_keyword_3,
                    competitor_1_name = EXCLUDED.competitor_1_name,
                    competitor_1_ticker = EXCLUDED.competitor_1_ticker,
                    competitor_2_name = EXCLUDED.competitor_2_name,
                    competitor_2_ticker = EXCLUDED.competitor_2_ticker,
                    competitor_3_name = EXCLUDED.competitor_3_name,
                    competitor_3_ticker = EXCLUDED.competitor_3_ticker,
                    ai_generated = EXCLUDED.ai_generated,
                    data_source = EXCLUDED.data_source,
                    updated_at = NOW()
            """, {
                'ticker': ticker_data['ticker'],
                'country': ticker_data['country'],
                'company_name': ticker_data['company_name'],
                'industry': ticker_data.get('industry'),
                'sector': ticker_data.get('sector'),
                'sub_industry': ticker_data.get('sub_industry'),
                'exchange': ticker_data.get('exchange'),
                'currency': ticker_data.get('currency'),
                'market_cap_category': ticker_data.get('market_cap_category'),
                'yahoo_ticker': ticker_data.get('yahoo_ticker', ticker_data['ticker']),
                'active': ticker_data.get('active', True),
                'is_etf': ticker_data.get('is_etf', False),
                'industry_keyword_1': ticker_data.get('industry_keyword_1'),
                'industry_keyword_2': ticker_data.get('industry_keyword_2'),
                'industry_keyword_3': ticker_data.get('industry_keyword_3'),
                'competitor_1_name': ticker_data.get('competitor_1_name'),
                'competitor_1_ticker': ticker_data.get('competitor_1_ticker'),
                'competitor_2_name': ticker_data.get('competitor_2_name'),
                'competitor_2_ticker': ticker_data.get('competitor_2_ticker'),
                'competitor_3_name': ticker_data.get('competitor_3_name'),
                'competitor_3_ticker': ticker_data.get('competitor_3_ticker'),
                'ai_generated': ticker_data.get('ai_generated', False),
                'data_source': ticker_data.get('data_source', 'api')
            })
            
            LOG.info(f"Successfully stored ticker reference: {ticker_data['ticker']} - {ticker_data['company_name']}")
            return True
            
    except Exception as e:
        LOG.error(f"Failed to store ticker reference for {ticker_data.get('ticker')}: {e}")
        return False

# 3. CSV IMPORT - With 6 competitor fields support
def import_ticker_reference_from_csv_content(csv_content: str):
    """Import ticker reference data from CSV with 3 industry keywords + 6 competitor fields - BULK OPTIMIZED"""
    ensure_ticker_reference_schema()
    
    imported = 0
    updated = 0
    errors = []
    skipped = 0
    
    try:
        LOG.debug(f"[CSV_IMPORT] Starting CSV import, content length: {len(csv_content) if csv_content else 'None'}")

        if not csv_content:
            LOG.error("[CSV_IMPORT] csv_content is None or empty!")
            return {
                "status": "error",
                "message": "CSV content is empty or None",
                "imported": 0, "updated": 0, "errors": []
            }

        csv_reader = csv.DictReader(io.StringIO(csv_content))

        # Validate CSV headers
        required_headers = ['ticker', 'country', 'company_name']
        missing_headers = [h for h in required_headers if h not in csv_reader.fieldnames]
        if missing_headers:
            LOG.error(f"[CSV_IMPORT] Missing headers: {missing_headers}")
            return {
                "status": "error",
                "message": f"Missing required CSV columns: {missing_headers}",
                "imported": 0, "updated": 0, "errors": []
            }

        LOG.info(f"[CSV_IMPORT] CSV headers found: {csv_reader.fieldnames}")
        LOG.debug(f"[CSV_IMPORT] Has legacy 'competitors' column: {'competitors' in csv_reader.fieldnames}")
        LOG.debug(f"[CSV_IMPORT] Has legacy 'industry_keywords' column: {'industry_keywords' in csv_reader.fieldnames}")
        
        # Collect all ticker data for bulk processing
        ticker_data_batch = []
        
        for row_num, row in enumerate(csv_reader, start=2):
            try:
                LOG.debug(f"[CSV_IMPORT] Processing row {row_num}, row type: {type(row)}")

                # Skip empty rows
                ticker_value = row.get('ticker', '')
                company_value = row.get('company_name', '')

                LOG.debug(f"[CSV_IMPORT] Row {row_num} - ticker type: {type(ticker_value)}, company type: {type(company_value)}")

                if not ticker_value or not str(ticker_value).strip() or not company_value or not str(company_value).strip():
                    skipped += 1
                    LOG.debug(f"[CSV_IMPORT] Row {row_num} skipped (empty ticker or company)")
                    continue

                # Build ticker data from CSV row
                ticker = str(ticker_value).strip()
                LOG.debug(f"[CSV_IMPORT] Row {row_num} processing ticker: {ticker}")
                
                # Build ticker_data with defensive string handling
                try:
                    country_val = row.get('country', '')
                    if country_val is None:
                        LOG.warning(f"[CSV_IMPORT] Row {row_num} has None country value")
                        country_val = ''

                    ticker_data = {
                        'ticker': ticker,
                        'country': str(country_val).strip().upper(),
                        'company_name': str(row.get('company_name', '')).strip(),
                        'industry': str(row.get('industry', '')).strip() or None,
                        'sector': str(row.get('sector', '')).strip() or None,
                        'sub_industry': str(row.get('sub_industry', '')).strip() or None,
                        'exchange': str(row.get('exchange', '')).strip() or None,
                        'currency': str(row.get('currency', '')).strip().upper() or None,
                        'market_cap_category': str(row.get('market_cap_category', '')).strip() or None,
                        'yahoo_ticker': str(row.get('yahoo_ticker', '')).strip() or ticker,
                        'active': str(row.get('active', 'true')).lower() in ('true', '1', 'yes', 'y', 't'),
                        'is_etf': str(row.get('is_etf', 'FALSE')).upper() in ('TRUE', '1', 'YES', 'Y'),
                        'data_source': 'csv_import',
                        'ai_generated': str(row.get('ai_generated', 'FALSE')).upper() in ('TRUE', '1', 'YES', 'Y')
                    }
                    LOG.debug(f"[CSV_IMPORT] Row {row_num} ticker_data built successfully")
                except Exception as e:
                    LOG.error(f"[CSV_IMPORT] Row {row_num} failed building ticker_data: {e}, row keys: {row.keys()}")
                    raise
                
                # Handle 3 industry keyword fields (with None safety)
                try:
                    ticker_data['industry_keyword_1'] = str(row.get('industry_keyword_1', '') or '').strip() or None
                    ticker_data['industry_keyword_2'] = str(row.get('industry_keyword_2', '') or '').strip() or None
                    ticker_data['industry_keyword_3'] = str(row.get('industry_keyword_3', '') or '').strip() or None
                    LOG.debug(f"[CSV_IMPORT] Row {row_num} industry keywords processed")
                except Exception as e:
                    LOG.error(f"[CSV_IMPORT] Row {row_num} failed processing industry keywords: {e}")
                    raise

                # Handle 6 competitor fields (with None safety)
                try:
                    ticker_data['competitor_1_name'] = str(row.get('competitor_1_name', '') or '').strip() or None
                    ticker_data['competitor_1_ticker'] = str(row.get('competitor_1_ticker', '') or '').strip() or None
                    ticker_data['competitor_2_name'] = str(row.get('competitor_2_name', '') or '').strip() or None
                    ticker_data['competitor_2_ticker'] = str(row.get('competitor_2_ticker', '') or '').strip() or None
                    ticker_data['competitor_3_name'] = str(row.get('competitor_3_name', '') or '').strip() or None
                    ticker_data['competitor_3_ticker'] = str(row.get('competitor_3_ticker', '') or '').strip() or None
                    LOG.debug(f"[CSV_IMPORT] Row {row_num} competitor fields processed")
                except Exception as e:
                    LOG.error(f"[CSV_IMPORT] Row {row_num} failed processing competitor fields: {e}")
                    raise
                
                # LEGACY SUPPORT: Handle old "competitors" field format
                competitors_field = row.get('competitors', '') or ''  # Handle None explicitly
                LOG.debug(f"[CSV_DEBUG] Row {row_num} competitors field type: {type(competitors_field)}, value: {repr(competitors_field)[:100]}")

                if (competitors_field and competitors_field.strip() and
                    not any(ticker_data.get(f'competitor_{i}_name') for i in range(1, 4))):
                    legacy_competitors = competitors_field.split(',')
                    for i, comp in enumerate(legacy_competitors[:3], 1):
                        comp = comp.strip()
                        if comp:
                            # Try to parse "Name (TICKER)" format
                            match = re.search(r'^(.+?)\s*\(([^)]+)\)$', comp)
                            if match:
                                name = match.group(1).strip()
                                ticker_comp = match.group(2).strip()
                                # Validate competitor ticker format
                                normalized_ticker = normalize_ticker_format(ticker_comp)
                                if validate_ticker_format(normalized_ticker):
                                    ticker_data[f'competitor_{i}_name'] = name
                                    ticker_data[f'competitor_{i}_ticker'] = normalized_ticker
                                else:
                                    LOG.warning(f"Invalid competitor ticker format: {ticker_comp}")
                                    ticker_data[f'competitor_{i}_name'] = name
                            else:
                                # Just a name without ticker
                                ticker_data[f'competitor_{i}_name'] = comp
                
                # LEGACY SUPPORT: Handle old "industry_keywords" field
                keywords_field = row.get('industry_keywords', '') or ''  # Handle None explicitly
                LOG.debug(f"[CSV_DEBUG] Row {row_num} industry_keywords field type: {type(keywords_field)}, value: {repr(keywords_field)[:100]}")

                if (keywords_field and keywords_field.strip() and
                    not any(ticker_data.get(f'industry_keyword_{i}') for i in range(1, 4))):
                    legacy_keywords = [kw.strip() for kw in keywords_field.split(',') if kw.strip()]
                    for i, keyword in enumerate(legacy_keywords[:3], 1):
                        ticker_data[f'industry_keyword_{i}'] = keyword
                
                # Clean NULL bytes from text fields
                text_fields = [
                    'company_name', 'industry', 'sector', 'sub_industry', 'exchange',
                    'industry_keyword_1', 'industry_keyword_2', 'industry_keyword_3',
                    'competitor_1_name', 'competitor_2_name', 'competitor_3_name',
                    'competitor_1_ticker', 'competitor_2_ticker', 'competitor_3_ticker'
                ]
                for field in text_fields:
                    if ticker_data.get(field):
                        ticker_data[field] = clean_null_bytes(str(ticker_data[field]))
                
                # Normalize competitor tickers
                competitor_ticker_fields = ['competitor_1_ticker', 'competitor_2_ticker', 'competitor_3_ticker']
                for field in competitor_ticker_fields:
                    if ticker_data.get(field):
                        ticker_data[field] = normalize_ticker_format(ticker_data[field])
                        # Validate competitor ticker format
                        if not validate_ticker_format(ticker_data[field]):
                            LOG.warning(f"Invalid competitor ticker format: {ticker_data[field]}")
                            ticker_data[field] = None  # Clear invalid ticker
                
                ticker_data_batch.append(ticker_data)
                    
            except Exception as e:
                errors.append(f"Row {row_num}: {str(e)}")
                continue
        
        # Bulk insert all valid data in single transaction
        if ticker_data_batch:
            try:
                with db() as conn, conn.cursor() as cur:
                   
                    # Prepare data tuples for bulk insert (fresh start, so everything is imported)
                    insert_data = []
                    imported = len(ticker_data_batch)
                    updated = 0
                    
                    for ticker_data in ticker_data_batch:

                        insert_data.append((
                            ticker_data['ticker'], ticker_data['country'], ticker_data['company_name'],
                            ticker_data.get('industry'), ticker_data.get('sector'), ticker_data.get('sub_industry'),
                            ticker_data.get('exchange'), ticker_data.get('currency'), ticker_data.get('market_cap_category'),
                            ticker_data.get('yahoo_ticker'), ticker_data.get('active', True), ticker_data.get('is_etf', False),
                            ticker_data.get('industry_keyword_1'), ticker_data.get('industry_keyword_2'), ticker_data.get('industry_keyword_3'),
                            ticker_data.get('competitor_1_name'), ticker_data.get('competitor_1_ticker'),
                            ticker_data.get('competitor_2_name'), ticker_data.get('competitor_2_ticker'),
                            ticker_data.get('competitor_3_name'), ticker_data.get('competitor_3_ticker'),
                            ticker_data.get('ai_generated', False), ticker_data.get('data_source', 'csv_import')
                        ))
                    
                    # Single bulk INSERT with ON CONFLICT handling
                    cur.executemany("""
                        INSERT INTO ticker_reference (
                            ticker, country, company_name, industry, sector, sub_industry,
                            exchange, currency, market_cap_category, yahoo_ticker, active, is_etf,
                            industry_keyword_1, industry_keyword_2, industry_keyword_3,
                            competitor_1_name, competitor_1_ticker,
                            competitor_2_name, competitor_2_ticker,
                            competitor_3_name, competitor_3_ticker,
                            ai_generated, data_source
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker) DO UPDATE SET
                            country = EXCLUDED.country,
                            company_name = EXCLUDED.company_name,
                            industry = EXCLUDED.industry,
                            sector = EXCLUDED.sector,
                            sub_industry = EXCLUDED.sub_industry,
                            exchange = EXCLUDED.exchange,
                            currency = EXCLUDED.currency,
                            market_cap_category = EXCLUDED.market_cap_category,
                            yahoo_ticker = EXCLUDED.yahoo_ticker,
                            active = EXCLUDED.active,
                            is_etf = EXCLUDED.is_etf,
                            industry_keyword_1 = EXCLUDED.industry_keyword_1,
                            industry_keyword_2 = EXCLUDED.industry_keyword_2,
                            industry_keyword_3 = EXCLUDED.industry_keyword_3,
                            competitor_1_name = EXCLUDED.competitor_1_name,
                            competitor_1_ticker = EXCLUDED.competitor_1_ticker,
                            competitor_2_name = EXCLUDED.competitor_2_name,
                            competitor_2_ticker = EXCLUDED.competitor_2_ticker,
                            competitor_3_name = EXCLUDED.competitor_3_name,
                            competitor_3_ticker = EXCLUDED.competitor_3_ticker,
                            ai_generated = EXCLUDED.ai_generated,
                            data_source = EXCLUDED.data_source,
                            updated_at = NOW()
                    """, insert_data)
                    
                    LOG.info(f"BULK INSERT COMPLETED: {len(insert_data)} records processed in single transaction")
                    
            except Exception as e:
                LOG.error(f"Bulk insert failed: {e}")
                return {
                    "status": "error", 
                    "message": f"Bulk insert failed: {str(e)}",
                    "imported": 0,
                    "updated": 0,
                    "errors": [str(e)]
                }
        else:
            LOG.warning("No valid ticker data found to import")
        
        LOG.info(f"CSV Import completed: {imported} imported, {updated} updated, {len(errors)} errors, {skipped} skipped")
        
        return {
            "status": "completed",
            "imported": imported,
            "updated": updated,
            "skipped": skipped,
            "errors": errors[:10],
            "total_errors": len(errors),
            "message": f"Successfully processed {imported + updated} ticker references ({imported} new, {updated} updated)"
        }
        
    except Exception as e:
        import traceback
        full_trace = traceback.format_exc()
        LOG.error(f"[CSV_IMPORT] CSV parsing failed with exception: {e}")
        LOG.error(f"[CSV_IMPORT] Full traceback:\n{full_trace}")
        return {
            "status": "error",
            "message": f"CSV parsing failed: {str(e)}",
            "imported": 0,
            "updated": 0,
            "errors": [str(e)]
        }

# 4. HELPER FUNCTIONS for data management
def get_all_ticker_references(limit: int = None, offset: int = 0, country_filter: str = None):
    """Get all ticker references with pagination and filtering"""
    with db() as conn, conn.cursor() as cur:
        # Build query based on filters
        where_clause = "WHERE active = TRUE"
        params = []
        
        if country_filter:
            where_clause += " AND country = %s"
            params.append(country_filter.upper())
        
        # Add pagination
        limit_clause = ""
        if limit:
            limit_clause = f" LIMIT {limit} OFFSET {offset}"
        
        cur.execute(f"""
            SELECT ticker, country, company_name, industry, sector, exchange, 
                   currency, market_cap_category, 
                   industry_keyword_1, industry_keyword_2, industry_keyword_3,
                   competitor_1_name, competitor_1_ticker,
                   competitor_2_name, competitor_2_ticker,
                   competitor_3_name, competitor_3_ticker,
                   ai_generated, ai_enhanced_at, created_at, updated_at
            FROM ticker_reference
            {where_clause}
            ORDER BY ticker
            {limit_clause}
        """, params)
        
        return list(cur.fetchall())

def count_ticker_references(country_filter: str = None) -> int:
    """Count total ticker references"""
    with db() as conn, conn.cursor() as cur:
        where_clause = "WHERE active = TRUE"
        params = []
        
        if country_filter:
            where_clause += " AND country = %s"
            params.append(country_filter.upper())
        
        cur.execute(f"""
            SELECT COUNT(*) as count
            FROM ticker_reference
            {where_clause}
        """, params)
        
        result = cur.fetchone()
        return result["count"] if result else 0

def delete_ticker_reference(ticker: str) -> bool:
    """Delete (deactivate) a ticker reference"""
    normalized_ticker = normalize_ticker_format(ticker)
    
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            UPDATE ticker_reference 
            SET active = FALSE, updated_at = NOW()
            WHERE ticker = %s
        """, (normalized_ticker,))
        
        if cur.rowcount > 0:
            LOG.info(f"Deactivated ticker reference: {normalized_ticker}")
            return True
        else:
            LOG.warning(f"Ticker reference not found: {normalized_ticker}")
            return False

# GitHub Integration Functions
# 1. FETCH CSV FROM GITHUB - Download latest version
def fetch_csv_from_github():
    """Download the latest ticker reference CSV from GitHub repository"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return {
            "status": "error",
            "message": "GitHub integration not configured. Set GITHUB_TOKEN and GITHUB_REPO environment variables."
        }
    
    try:
        # GitHub API URL for file content
        api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_CSV_PATH}"
        
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        LOG.info(f"Fetching CSV from GitHub: {GITHUB_REPO}/{GITHUB_CSV_PATH}")
        
        response = requests.get(api_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            file_data = response.json()

            # Get file metadata
            file_info = {
                "sha": file_data["sha"],
                "size": file_data["size"],
                "last_modified": file_data.get("last_modified"),
                "download_url": file_data["download_url"]
            }

            LOG.info(f"[GITHUB_FETCH] File size: {file_info['size']} bytes, SHA: {file_info['sha'][:8]}")

            # For files >1MB, GitHub's content field is empty - use download_url instead
            raw_content = file_data.get("content", "")

            if not raw_content or file_info["size"] > 1000000:
                LOG.info(f"[GITHUB_FETCH] Using download_url for large file ({file_info['size']} bytes)")

                # Fetch raw content directly via download_url
                download_response = requests.get(file_info["download_url"], timeout=30)

                if download_response.status_code == 200:
                    csv_content = download_response.text
                    LOG.info(f"[GITHUB_FETCH] Successfully downloaded {len(csv_content)} characters via download_url")
                else:
                    LOG.error(f"[GITHUB_FETCH] Failed to download from download_url: HTTP {download_response.status_code}")
                    return {
                        "status": "error",
                        "message": f"Failed to download file content: HTTP {download_response.status_code}",
                        "csv_content": None
                    }
            else:
                # Small file - decode from base64 content field
                LOG.info(f"[GITHUB_FETCH] Using base64 content field for small file")
                csv_content = base64.b64decode(raw_content).decode('utf-8')
            
            return {
                "status": "success",
                "csv_content": csv_content,
                "file_info": file_info,
                "message": f"Successfully downloaded {len(csv_content)} characters from GitHub"
            }
            
        elif response.status_code == 404:
            return {
                "status": "error",
                "message": f"CSV file not found at {GITHUB_REPO}/{GITHUB_CSV_PATH}. Please verify the path."
            }
        elif response.status_code == 401:
            return {
                "status": "error",
                "message": "GitHub authentication failed. Please check your GITHUB_TOKEN."
            }
        elif response.status_code == 403:
            return {
                "status": "error",
                "message": "GitHub API rate limit exceeded or insufficient permissions."
            }
        else:
            return {
                "status": "error",
                "message": f"GitHub API error {response.status_code}: {response.text}"
            }
            
    except requests.RequestException as e:
        LOG.error(f"Network error fetching CSV from GitHub: {e}")
        return {
            "status": "error",
            "message": f"Network error: {str(e)}"
        }
    except Exception as e:
        LOG.error(f"Unexpected error fetching CSV from GitHub: {e}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

# 2. EXPORT SQL TO CSV FORMAT - Generate CSV from current database
def export_ticker_references_to_csv():
    LOG.info("🚨 EXPORT FUNCTION CALLED - UPDATED VERSION WITH DEBUG 🚨")
    """Export all ticker references from database to CSV format with new structure"""
    try:
        with db() as conn, conn.cursor() as cur:
            # Get all ticker references with all fields
            cur.execute("""
                SELECT ticker, country, company_name, industry, sector, sub_industry,
                       exchange, currency, market_cap_category, active, is_etf, yahoo_ticker,
                       industry_keyword_1, industry_keyword_2, industry_keyword_3,
                       ai_generated, ai_enhanced_at,
                       competitor_1_name, competitor_1_ticker,
                       competitor_2_name, competitor_2_ticker,
                       competitor_3_name, competitor_3_ticker,
                       financial_last_price, financial_price_change_pct,
                       financial_yesterday_return_pct, financial_ytd_return_pct,
                       financial_market_cap, financial_enterprise_value,
                       financial_volume, financial_avg_volume,
                       financial_analyst_target, financial_analyst_range_low,
                       financial_analyst_range_high, financial_analyst_count,
                       financial_analyst_recommendation, financial_snapshot_date,
                       created_at, updated_at, data_source
                FROM ticker_reference
                ORDER BY ticker
            """)
            
            rows = cur.fetchall()
            LOG.info(f"DEBUG: Retrieved {len(rows)} rows from database")
            
            if not rows:
                LOG.error("DEBUG: No rows returned from database query")
                return {
                    "status": "error",
                    "message": "No ticker references found in database"
                }
            
            # Debug: Show first row structure
            LOG.info(f"DEBUG: First row sample: {dict(rows[0]) if rows else 'No rows'}")
            LOG.info(f"DEBUG: Row has {len(rows[0])} columns")
            
            # Build CSV content
            csv_buffer = io.StringIO()
            
            # Define CSV headers (matching your GitHub CSV structure)
            headers = [
                'ticker', 'country', 'company_name', 'industry', 'sector', 'sub_industry',
                'exchange', 'currency', 'market_cap_category', 'active', 'is_etf', 'yahoo_ticker',
                'industry_keyword_1', 'industry_keyword_2', 'industry_keyword_3',
                'ai_generated', 'ai_enhanced_at',
                'competitor_1_name', 'competitor_1_ticker',
                'competitor_2_name', 'competitor_2_ticker',
                'competitor_3_name', 'competitor_3_ticker',
                'financial_last_price', 'financial_price_change_pct',
                'financial_yesterday_return_pct', 'financial_ytd_return_pct',
                'financial_market_cap', 'financial_enterprise_value',
                'financial_volume', 'financial_avg_volume',
                'financial_analyst_target', 'financial_analyst_range_low',
                'financial_analyst_range_high', 'financial_analyst_count',
                'financial_analyst_recommendation', 'financial_snapshot_date',
                'created_at', 'updated_at', 'data_source'
            ]
            
            LOG.info(f"DEBUG: Header has {len(headers)} columns")
            
            writer = csv.writer(csv_buffer)
            writer.writerow(headers)
            
            # Write data rows with debug info
            rows_written = 0
            for i, row in enumerate(rows):
                try:
                    csv_row = []
                    
                    # Debug first few rows
                    if i < 3:
                        LOG.info(f"DEBUG: Processing row {i}: ticker={row.get('ticker', 'NO_TICKER')}")
                    
                    for header in headers:
                        value = row[header]
                        if value is None:
                            csv_row.append('')
                        elif isinstance(value, bool):
                            csv_row.append('TRUE' if value else 'FALSE')
                        elif isinstance(value, datetime):
                            csv_row.append(value.isoformat())
                        else:
                            csv_row.append(str(value))
                    
                    # Debug row length
                    if i < 3:
                        LOG.info(f"DEBUG: Row {i} has {len(csv_row)} values")
                    
                    writer.writerow(csv_row)
                    rows_written += 1
                    
                except Exception as row_error:
                    LOG.error(f"DEBUG: Error processing row {i}: {row_error}")
                    continue
            
            csv_content = csv_buffer.getvalue()
            csv_buffer.close()
            
            LOG.info(f"DEBUG: Wrote {rows_written} data rows to CSV")
            LOG.info(f"DEBUG: CSV content length: {len(csv_content)} characters")
            LOG.info(f"DEBUG: CSV starts with: {csv_content[:200]}...")
            
            return {
                "status": "success",
                "csv_content": csv_content,
                "ticker_count": len(rows),
                "message": f"Successfully exported {len(rows)} ticker references"
            }
            
    except Exception as e:
        LOG.error(f"Failed to export ticker references to CSV: {e}")
        return {
            "status": "error",
            "message": f"Export failed: {str(e)}"
        }

# 3. COMMIT CSV TO GITHUB - Push updated CSV back to repository
def commit_csv_to_github(csv_content: str, commit_message: str = None):
    """Push updated CSV content back to GitHub repository"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return {
            "status": "error",
            "message": "GitHub integration not configured. Set GITHUB_TOKEN and GITHUB_REPO environment variables."
        }
    
    if not csv_content:
        return {
            "status": "error",
            "message": "No CSV content provided"
        }
    
    try:
        # First, get the current file to obtain its SHA (required for updates)
        api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_CSV_PATH}"
        
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        # Get current file info
        LOG.info(f"Getting current file info from GitHub: {GITHUB_REPO}/{GITHUB_CSV_PATH}")
        response = requests.get(api_url, headers=headers, timeout=30)
        
        file_sha = None
        if response.status_code == 200:
            current_file = response.json()
            file_sha = current_file["sha"]
            LOG.info(f"Found existing file, SHA: {file_sha[:8]}")
        elif response.status_code == 404:
            LOG.info("File doesn't exist, will create new file")
        else:
            return {
                "status": "error",
                "message": f"Failed to get current file info: {response.status_code} - {response.text}"
            }
        
        # Prepare commit data
        if not commit_message:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            commit_message = f"[skip render] Update ticker reference data - {timestamp}"
        
        # Encode content to base64
        encoded_content = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
        
        commit_data = {
            "message": commit_message,
            "content": encoded_content,
        }
        
        # Include SHA for updates (not for new files)
        if file_sha:
            commit_data["sha"] = file_sha
        
        # Commit the file
        LOG.info(f"Committing CSV to GitHub: {len(csv_content)} characters")
        # Add retry logic for network timeouts AND SHA conflicts
        max_retries = 3
        sha_retry_count = 0
        max_sha_retries = 2  # Allow 2 SHA conflict retries

        for attempt in range(max_retries):
            try:
                LOG.info(f"GitHub commit attempt {attempt + 1}/{max_retries}")
                commit_response = requests.put(api_url, headers=headers, json=commit_data, timeout=120)

                # Handle 409 Conflict (SHA mismatch) - someone else committed
                if commit_response.status_code == 409 and sha_retry_count < max_sha_retries:
                    sha_retry_count += 1
                    LOG.warning(f"⚠️ GitHub SHA conflict detected (attempt {sha_retry_count}/{max_sha_retries})")
                    LOG.warning("   Another commit was made between GET and PUT. Re-fetching current SHA...")

                    # Re-fetch current file SHA
                    time.sleep(2)  # Brief pause before retry
                    refetch_response = requests.get(api_url, headers=headers, timeout=30)

                    if refetch_response.status_code == 200:
                        current_file = refetch_response.json()
                        new_sha = current_file["sha"]
                        LOG.info(f"   Refetched SHA: {new_sha[:8]} (was: {file_sha[:8] if file_sha else 'None'})")

                        # Update commit_data with new SHA
                        commit_data["sha"] = new_sha
                        file_sha = new_sha

                        # Retry the commit with new SHA
                        LOG.info(f"   Retrying commit with updated SHA...")
                        continue
                    else:
                        LOG.error(f"   Failed to refetch SHA: {refetch_response.status_code}")
                        return {
                            "status": "error",
                            "message": f"SHA conflict: failed to refetch file ({refetch_response.status_code})"
                        }

                # If not a retryable 409, break out of loop
                break

            except requests.Timeout as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                    LOG.warning(f"GitHub commit timeout (attempt {attempt + 1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"GitHub commit failed after {max_retries} timeout attempts")
                    return {
                        "status": "error",
                        "message": f"GitHub commit timed out after {max_retries} attempts"
                    }
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                    LOG.warning(f"GitHub commit network error (attempt {attempt + 1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise

        if commit_response.status_code in [200, 201]:
            commit_result = commit_response.json()

            LOG.info(f"✅ Successfully committed CSV to GitHub: {commit_result['commit']['sha'][:8]}")

            return {
                "status": "success",
                "commit_sha": commit_result['commit']['sha'],
                "file_sha": commit_result['content']['sha'],
                "commit_url": commit_result['commit']['html_url'],
                "message": f"Successfully updated {GITHUB_CSV_PATH} in {GITHUB_REPO}",
                "csv_size": len(csv_content),
                "sha_retries": sha_retry_count
            }
        else:
            error_msg = commit_response.text
            LOG.error(f"❌ Failed to commit CSV to GitHub: {commit_response.status_code} - {error_msg}")

            return {
                "status": "error",
                "message": f"GitHub commit failed ({commit_response.status_code}): {error_msg}",
                "response_body": error_msg[:500]  # Include partial response for debugging
            }
            
    except requests.RequestException as e:
        LOG.error(f"Network error committing CSV to GitHub: {e}")
        return {
            "status": "error",
            "message": f"Network error: {str(e)}"
        }
    except Exception as e:
        LOG.error(f"Unexpected error committing CSV to GitHub: {e}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

# 4. FULL SYNC OPERATIONS - High-level workflow functions
def sync_ticker_references_from_github():
    """Complete workflow: Fetch CSV from GitHub and update database"""
    LOG.info("=== Starting full sync from GitHub ===")
    
    # Step 1: Fetch CSV from GitHub
    fetch_result = fetch_csv_from_github()
    if fetch_result["status"] != "success":
        return fetch_result
    
    # Step 2: Import CSV into database
    import_result = import_ticker_reference_from_csv_content(fetch_result["csv_content"])
    
    # Combine results - Fix status logic
    sync_status = "success" if import_result["status"] == "completed" else "error"
    success_message = f"GitHub sync successful: {import_result.get('imported', 0)} imported, {import_result.get('updated', 0)} updated"
    
    return {
        "status": sync_status,
        "github_fetch": {
            "csv_size": len(fetch_result["csv_content"]),
            "file_sha": fetch_result["file_info"]["sha"][:8]
        },
        "database_import": {
            "imported": import_result.get("imported", 0),
            "updated": import_result.get("updated", 0),
            "skipped": import_result.get("skipped", 0),
            "errors": import_result.get("total_errors", 0)
        },
        "message": success_message if sync_status == "success" else f"GitHub sync failed: {import_result.get('message', 'Unknown error')}"
    }

def sync_ticker_references_to_github(commit_message: str = None):
    """Export database to CSV format (no GitHub commit)"""
    LOG.info("=== Exporting ticker references to CSV format ===")
    
    # Export database to CSV
    export_result = export_ticker_references_to_csv()
    
    return {
        "status": export_result["status"],
        "database_export": {
            "ticker_count": export_result.get("ticker_count", 0),
            "csv_size": len(export_result.get("csv_content", ""))
        },
        "message": f"Exported {export_result.get('ticker_count', 0)} ticker references to CSV format"
    }

# 5. SELECTIVE TICKER UPDATE - Update specific tickers only
def update_specific_tickers_on_github(tickers: list, commit_message: str = None):
    """Update specific tickers in CSV format (no GitHub commit)"""
    if not tickers:
        return {"status": "error", "message": "No tickers specified"}
    
    LOG.info(f"=== Updating specific tickers in CSV format: {tickers} ===")
    
    try:
        # Get current CSV from database export
        export_result = export_ticker_references_to_csv()
        if export_result["status"] != "success":
            return export_result
        
        # Parse and validate the updated data
        updated_csv_content = export_result["csv_content"]
        
        return {
            "status": "success", 
            "updated_tickers": len(tickers),
            "csv_content": updated_csv_content,
            "message": f"Updated {len(tickers)} tickers in CSV format"
        }
        
    except Exception as e:
        LOG.error(f"Failed to update specific tickers: {e}")
        return {"status": "error", "message": f"Update failed: {str(e)}"}

def store_competitor_metadata(ticker: str, competitors: List[Dict]) -> None:
    """Store competitor metadata in a dedicated table for better normalization"""
    with db() as conn, conn.cursor() as cur:
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
async def extract_article_content_with_playwright(url: str, domain: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Memory-optimized article extraction using Playwright with enhanced content cleaning (ASYNC VERSION)
    """
    try:
        # Check for known paywall domains first
        if normalize_domain(domain) in PAYWALL_DOMAINS:
            return None, f"Paywall domain: {domain}"
        
        LOG.info(f"PLAYWRIGHT: Starting browser for {domain}")
        
        async with async_playwright() as p:
            # Launch browser with memory optimization
            browser = await p.chromium.launch(
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
            page = await browser.new_page(
                viewport={'width': 1280, 'height': 720},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            # Set additional headers to look more human
            await page.set_extra_http_headers({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            })
            
            try:
                # REDUCED TIMEOUT - 15 seconds instead of 30
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                LOG.info(f"PLAYWRIGHT: Page loaded for {domain}, extracting content...")
                
                # Shorter wait for dynamic content
                await page.wait_for_timeout(1000)
                
                # Try multiple content extraction methods
                raw_content = None
                extraction_method = None
                
                # Method 1: Try article tag first
                try:
                    article_element = await page.query_selector('article')
                    if article_element:
                        raw_content = await article_element.inner_text()
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
                            element = await page.query_selector(selector)
                            if element:
                                temp_content = await element.inner_text()
                                if temp_content and len(temp_content.strip()) > 200:
                                    raw_content = temp_content
                                    extraction_method = f"selector: {selector}"
                                    break
                        except Exception:
                            continue
                
                # Method 3: Smart body text extraction (removes navigation/ads)
                if not raw_content or len(raw_content.strip()) < 200:
                    try:
                        raw_content = await page.evaluate("""
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
                            raw_content = await page.evaluate("() => document.body ? document.body.innerText : ''")
                            extraction_method = "fallback body text"
                        except Exception:
                            raw_content = None
                
            except Exception as e:
                LOG.warning(f"PLAYWRIGHT: Navigation/extraction failed for {domain}: {str(e)}")
                raw_content = None
            finally:
                # Always close browser to free memory
                try:
                    await browser.close()
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


async def scrape_with_scrapfly_async(url: str, domain: str, max_retries: int = 2) -> Tuple[Optional[str], Optional[str]]:
    """
    Async Scrapfly scraping with improved text extraction, content cleaning, and video URL filtering.
    """
    global scrapfly_stats, enhanced_scraping_stats

    def _host(u: str) -> str:
        h = (urlparse(u).hostname or "").lower()
        if h.startswith("www."):
            h = h[4:]
        return h

    def _matches(host: str, dom: str) -> bool:
        return host == dom or host.endswith("." + dom)

    # EARLY FILTER: Reject video URLs before any processing
    if "video.media.yql.yahoo.com" in url:
        LOG.warning(f"ASYNC SCRAPFLY: Rejecting video URL: {url}")
        return None, "Video URL not supported"

    # Local list for anti-bot domains
    LOCAL_SCRAPFLY_ANTIBOT = {
        "simplywall.st", "seekingalpha.com", "zacks.com", "benzinga.com",
        "cnbc.com", "investing.com", "gurufocus.com", "fool.com",
        "insidermonkey.com", "nasdaq.com", "markets.financialcontent.com",
        "thefly.com", "streetinsider.com", "accesswire.com",
        "247wallst.com", "barchart.com", "telecompaper.com",
        "news.stocktradersdaily.com", "sharewise.com"
        # Removed video.media.yql.yahoo.com since we filter it out earlier
    }

    for attempt in range(max_retries + 1):
        try:
            if not SCRAPFLY_API_KEY:
                return None, "Scrapfly API key not configured"

            if normalize_domain(domain) in PAYWALL_DOMAINS:
                return None, f"Paywall domain: {domain}"

            if attempt > 0:
                delay = 2 ** attempt
                LOG.info(f"ASYNC SCRAPFLY RETRY {attempt}/{max_retries} for {domain} after {delay}s delay")
                await asyncio.sleep(delay)

            LOG.info(f"ASYNC SCRAPFLY: Starting scrape for {domain} (attempt {attempt + 1})")

            # Update usage stats
            scrapfly_stats["requests_made"] += 1
            scrapfly_stats["cost_estimate"] += 0.002
            scrapfly_stats["by_domain"][domain]["attempts"] += 1
            enhanced_scraping_stats["by_method"]["scrapfly"]["attempts"] += 1

            host = _host(url)
            
            # Build Scrapfly params - CONVERT BOOLEANS TO STRINGS
            params = {
                "key": SCRAPFLY_API_KEY,
                "url": url,
                "render_js": "false",  # Convert boolean to string
                "country": "us",
                "cache": "false",      # Convert boolean to string
            }
            
            # Toggle anti-bot/JS only for local list, but not for video URLs
            if any(_matches(host, d) for d in LOCAL_SCRAPFLY_ANTIBOT) and "video.media" not in host:
                params["asp"] = "true"        # Convert boolean to string
                params["render_js"] = "true"  # Convert boolean to string

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get("https://api.scrapfly.io/scrape", params=params) as response:
                    if response.status == 200:
                        try:
                            result = await response.json()
                            html_content = result.get("result", {}).get("content", "") or ""
                        except Exception as json_error:
                            LOG.warning(f"ASYNC SCRAPFLY: JSON parsing failed for {domain}: {json_error}")
                            html_content = await response.text() or ""

                        # Extract text from HTML
                        raw_content = ""
                        if html_content:
                            try:
                                article = newspaper.Article(url)
                                article.set_html(html_content)
                                article.parse()
                                raw_content = article.text or ""
                            except Exception as e:
                                LOG.warning(f"ASYNC SCRAPFLY: Newspaper parse failed for {domain}: {e}")
                                raw_content = html_content

                        if not raw_content or len(raw_content.strip()) < 100:
                            LOG.warning(f"ASYNC SCRAPFLY: Insufficient raw content for {domain} (attempt {attempt + 1})")
                            continue

                        cleaned_content = clean_scraped_content(raw_content, url, domain)
                        if not cleaned_content or len(cleaned_content.strip()) < 100:
                            LOG.warning(f"ASYNC SCRAPFLY: Content too short after cleaning for {domain}")
                            continue

                        is_valid, validation_msg = validate_scraped_content(cleaned_content, url, domain)
                        if not is_valid:
                            LOG.warning(f"ASYNC SCRAPFLY: Content validation failed for {domain}: {validation_msg}")
                            break

                        # Success stats
                        scrapfly_stats["successful"] += 1
                        scrapfly_stats["by_domain"][domain]["successes"] += 1
                        enhanced_scraping_stats["by_method"]["scrapfly"]["successes"] += 1

                        raw_len = len(str(raw_content))
                        clean_len = len(cleaned_content)
                        reduction = ((raw_len - clean_len) / raw_len * 100) if raw_len > 0 else 0.0
                        LOG.info(f"ASYNC SCRAPFLY SUCCESS: {domain} -> {clean_len} chars (cleaned from {raw_len}, {reduction:.1f}% reduction)")
                        return cleaned_content, None

                    elif response.status == 422:
                        error_text = await response.text()
                        # Check if this is a video URL causing the error
                        if "video.media" in url:
                            LOG.warning(f"ASYNC SCRAPFLY: Video URL caused 422 error, skipping: {url}")
                            return None, "Video URL not supported"
                        else:
                            LOG.warning(f"ASYNC SCRAPFLY: 422 invalid parameters for {domain} body: {error_text[:500]}")
                        break

                    elif response.status == 429:
                        LOG.warning(f"ASYNC SCRAPFLY: 429 rate limited for {domain} (attempt {attempt + 1})")
                        if attempt < max_retries:
                            await asyncio.sleep(5)
                            continue

                    else:
                        error_text = await response.text()
                        req_id = response.headers.get("x-request-id") or response.headers.get("cf-ray")
                        LOG.warning(
                            f"ASYNC SCRAPFLY: HTTP {response.status} for {domain} "
                            f"(attempt {attempt + 1}) id={req_id} body: {error_text[:500]}"
                        )
                        if attempt < max_retries:
                            continue

        except Exception as e:
            if "video.media" in url:
                LOG.warning(f"ASYNC SCRAPFLY: Video URL caused exception, skipping: {url}")
                return None, "Video URL not supported"
            LOG.warning(f"ASYNC SCRAPFLY: Request error for {domain} (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                continue

    scrapfly_stats["failed"] += 1
    return None, f"Async Scrapfly failed after {max_retries + 1} attempts"

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

    # Stage 0: Remove NULL bytes and control characters FIRST
    content = clean_null_bytes(content)
    
    # Stage 1: Remove obvious binary/encoded data
    # Remove sequences that look like binary data or encoding artifacts
    content = re.sub(r'[Â¿Â½]{3,}.*?[Â¿Â½]{3,}', '', content)
    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]+', '', content)
    content = re.sub(r'[A-Za-z0-9+/]{50,}={0,2}', '', content)
    
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

    # Final NULL byte check
    content = clean_null_bytes(content)
    
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
    """Enhanced scraping with better timeout handling for slow domains"""
    domain = normalize_domain(urlparse(url).netloc.lower())
    
    # Longer timeout for known slow domains
    timeout = 30 if domain in ["businesswire.com", "globenewswire.com"] else 15
    
    for attempt in range(max_retries):
        try:
            response = scraping_session.get(url, timeout=timeout)
            if response.status_code == 200:
                return response
            elif response.status_code in [429, 500, 503, 504]:  # Added 504
                delay = (2 ** attempt) + random.uniform(0, 1)
                LOG.info(f"HTTP {response.status_code} error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                if attempt == max_retries - 1:
                    LOG.warning(f"Max retries reached for {url} after {response.status_code} errors")
                    return None
            else:
                LOG.warning(f"HTTP {response.status_code} for {url}, not retrying")
                return None
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                LOG.warning(f"Request failed, retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
            else:
                LOG.warning(f"Final retry failed for {url}: {e}")
    
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
                    enhanced_scraping_stats.get("scrapfly_success", 0))
    overall_rate = (total_success / total_attempts) * 100
    
    # Calculate individual success rates
    requests_attempts = enhanced_scraping_stats["by_method"]["requests"]["attempts"]
    requests_success = enhanced_scraping_stats["by_method"]["requests"]["successes"]
    requests_rate = (requests_success / requests_attempts * 100) if requests_attempts > 0 else 0
    
    playwright_attempts = enhanced_scraping_stats["by_method"]["playwright"]["attempts"]
    playwright_success = enhanced_scraping_stats["by_method"]["playwright"]["successes"]
    playwright_rate = (playwright_success / playwright_attempts * 100) if playwright_attempts > 0 else 0
    
    scrapfly_attempts = enhanced_scraping_stats["by_method"].get("scrapfly", {}).get("attempts", 0)
    scrapfly_success = enhanced_scraping_stats["by_method"].get("scrapfly", {}).get("successes", 0)
    scrapfly_rate = (scrapfly_success / scrapfly_attempts * 100) if scrapfly_attempts > 0 else 0
    
    # Log prominent success rate summary with per-tier percentages
    LOG.info("=" * 60)
    LOG.info("SCRAPING SUCCESS RATES")
    LOG.info("=" * 60)
    LOG.info(f"OVERALL SUCCESS: {overall_rate:.1f}% ({total_success}/{total_attempts})")
    LOG.info(f"TIER 1 (Requests): {requests_rate:.1f}% of tier attempts ({requests_success}/{requests_attempts})")
    LOG.info(f"TIER 2 (Playwright): {playwright_rate:.1f}% of tier attempts ({playwright_success}/{playwright_attempts})")
    LOG.info(f"TIER 3 (Scrapfly): {scrapfly_rate:.1f}% of tier attempts ({scrapfly_success}/{scrapfly_attempts})")

    if scrapfly_attempts > 0:
        LOG.info(f"Scrapfly Cost: ${scrapfly_stats['cost_estimate']:.3f}")
    
    LOG.info("=" * 60)

def validate_scraped_content(content, url, domain):
    """Enhanced content validation with relaxed requirements for quality domains"""
    if not content or len(content.strip()) < 100:
        return False, "Content too short"
    
    # RELAXED REQUIREMENTS for quality domains
    domain_normalized = normalize_domain(domain)
    is_quality_domain = domain_normalized in QUALITY_DOMAINS
    
    # Check content-to-boilerplate ratio
    sentences = [s.strip() for s in content.split('.') if s.strip()]
    min_sentences = 2 if is_quality_domain else 3  # RELAXED for quality domains
    
    if len(sentences) < min_sentences:
        return False, f"Insufficient sentences (need {min_sentences}, got {len(sentences)})"
    
    # Check for repetitive content
    words = content.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        min_ratio = 0.25 if is_quality_domain else 0.3  # RELAXED for quality domains
        if unique_ratio < min_ratio:
            return False, "Repetitive content detected"
    
    # Check if content is mostly technical/code-like
    technical_chars = len(re.findall(r'[{}();:=<>]', content))
    max_technical = 0.15 if is_quality_domain else 0.1  # RELAXED for quality domains
    if technical_chars > len(content) * max_technical:
        return False, "Content appears to be technical/code data"
    
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

async def safe_content_scraper_with_3tier_fallback_async(url: str, domain: str, category: str, keyword: str, scraped_domains: set) -> Tuple[Optional[str], str]:
    """
    Async 3-tier content scraper: requests → Playwright → Scrapfly with comprehensive tracking
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
    
    # Ensure async semaphores are initialized
    init_async_semaphores()
    
    enhanced_scraping_stats["total_attempts"] += 1

    # Check if domain should skip TIER 1 (known Playwright-only domains)
    if domain in SKIP_TIER1_DOMAINS:
        LOG.info(f"⚡ SKIP TIER 1: {domain} (known Playwright-only domain)")
        error = "Skipped TIER 1 (Playwright-only domain)"
    else:
        # TIER 1: Try standard requests-based scraping
        LOG.info(f"ASYNC TIER 1 (Requests): Attempting {domain}")
        enhanced_scraping_stats["by_method"]["requests"]["attempts"] += 1

        # Use thread pool for sync requests call
        import asyncio
        loop = asyncio.get_event_loop()

        try:
            content, error = await loop.run_in_executor(None, safe_content_scraper, url, domain, scraped_domains)

            if content:
                enhanced_scraping_stats["requests_success"] += 1
                enhanced_scraping_stats["by_method"]["requests"]["successes"] += 1
                update_scraping_stats(category, keyword, True)
                return content, f"ASYNC TIER 1 SUCCESS: {len(content)} chars via requests"
        except Exception as e:
            error = str(e)

        LOG.info(f"ASYNC TIER 1 FAILED: {domain} - {error}")
    
    # TIER 2: Try Playwright fallback
    LOG.info(f"ASYNC TIER 2 (Playwright): Attempting {domain}")
    enhanced_scraping_stats["by_method"]["playwright"]["attempts"] += 1
    
    async with async_playwright_sem:
        try:
            playwright_content, playwright_error = await extract_article_content_with_playwright(url, domain)
            
            if playwright_content:
                enhanced_scraping_stats["playwright_success"] += 1
                enhanced_scraping_stats["by_method"]["playwright"]["successes"] += 1
                update_scraping_stats(category, keyword, True)
                return playwright_content, f"ASYNC TIER 2 SUCCESS: {len(playwright_content)} chars via Playwright"
        except Exception as e:
            playwright_error = str(e)
    
    LOG.info(f"ASYNC TIER 2 FAILED: {domain} - {playwright_error}")
    
    # TIER 3: Try Scrapfly fallback
    if SCRAPFLY_API_KEY:
        LOG.info(f"ASYNC TIER 3 (Scrapfly): Attempting {domain}")
        
        async with async_scrapfly_sem:
            try:
                scrapfly_content, scrapfly_error = await scrape_with_scrapfly_async(url, domain)
                
                if scrapfly_content:
                    enhanced_scraping_stats["scrapfly_success"] += 1
                    update_scraping_stats(category, keyword, True)
                    return scrapfly_content, f"ASYNC TIER 3 SUCCESS: {len(scrapfly_content)} chars via Scrapfly"
            except Exception as e:
                scrapfly_error = str(e)
        
        LOG.info(f"ASYNC TIER 3 FAILED: {domain} - {scrapfly_error}")
    else:
        LOG.info(f"ASYNC TIER 3 SKIPPED: Scrapfly API key not configured")
        scrapfly_error = "not configured"
    
    # All tiers failed
    enhanced_scraping_stats["total_failures"] += 1
    return None, f"ALL ASYNC TIERS FAILED - Requests: {error}, Playwright: {playwright_error}, Scrapfly: {scrapfly_error if SCRAPFLY_API_KEY else 'not configured'}"

def log_enhanced_scraping_stats():
    """Log comprehensive scraping statistics across all methods"""
    total = enhanced_scraping_stats["total_attempts"]
    if total == 0:
        LOG.info("ENHANCED SCRAPING: No attempts made")
        return
    
    requests_success = enhanced_scraping_stats["requests_success"]
    playwright_success = enhanced_scraping_stats["playwright_success"]
    scrapfly_success = enhanced_scraping_stats["scrapfly_success"]
    total_success = requests_success + playwright_success + scrapfly_success
    
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
    global enhanced_scraping_stats, scrapfly_stats
    enhanced_scraping_stats = {
        "total_attempts": 0,
        "requests_success": 0,
        "playwright_success": 0,
        "scrapfly_success": 0,
        "total_failures": 0,
        "by_method": {
            "requests": {"attempts": 0, "successes": 0},
            "playwright": {"attempts": 0, "successes": 0},
            "scrapfly": {"attempts": 0, "successes": 0},
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

async def process_article_batch_async(articles_batch: List[Dict], categories: Union[str, List[str]], metadata: Dict, analysis_ticker: str) -> List[Dict]:
    """
    Process a batch of articles concurrently: scraping → AI summarization → database update
    Now supports per-article categories for POV-agnostic summarization
    Returns list of results for each article in the batch
    """
    batch_size = len(articles_batch)
    LOG.info(f"BATCH START: Processing {batch_size} articles from {analysis_ticker}'s perspective")

    # Normalize categories to list
    if isinstance(categories, str):
        categories = [categories] * batch_size

    # Build competitor name cache from metadata
    competitor_name_cache = {}
    for comp in metadata.get("competitors", []):
        if comp.get("ticker") and comp.get("name"):
            competitor_name_cache[comp["ticker"]] = comp["name"]

    target_company_name = metadata.get("company_name", analysis_ticker)

    results = []

    # Phase 1: Concurrent scraping for all articles in batch
    LOG.info(f"BATCH PHASE 1: Concurrent scraping of {batch_size} articles")

    scraping_tasks = []
    for i, article in enumerate(articles_batch):
        # Use individual article's category
        article_category = categories[i] if i < len(categories) else categories[0]
        task = scrape_single_article_async(article, article_category, metadata, analysis_ticker, i)
        scraping_tasks.append(task)

    # Execute all scraping tasks concurrently
    scraping_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)

    # Phase 2: Concurrent AI summarization with category-specific prompts
    successful_scrapes = []
    for i, result in enumerate(scraping_results):
        if isinstance(result, Exception):
            LOG.error(f"BATCH SCRAPING ERROR: Article {i} failed: {result}")
            results.append({
                "article_id": articles_batch[i]["id"],
                "success": False,
                "error": f"Scraping failed: {str(result)}",
                "scraped_content": None,
                "ai_summary": None,
                "ai_model": None
            })
        elif result["success"]:
            successful_scrapes.append((i, result))

    if successful_scrapes:
        LOG.info(f"BATCH PHASE 2: AI summarization of {len(successful_scrapes)} successful scrapes with category-specific prompts")

        ai_tasks = []
        for i, scrape_result in successful_scrapes:
            article_category = categories[i] if i < len(categories) else categories[0]
            task = generate_ai_summary_with_fallback(
                scrape_result["scraped_content"],
                articles_batch[i]["title"],
                analysis_ticker,
                articles_batch[i].get("description", ""),
                article_category,
                articles_batch[i],  # article_metadata
                target_company_name,
                competitor_name_cache
            )
            ai_tasks.append((i, task))

        # Execute AI summarization concurrently
        ai_results = await asyncio.gather(*[task for _, task in ai_tasks], return_exceptions=True)

        # Combine scraping and AI results
        for j, (original_idx, _) in enumerate(ai_tasks):
            scrape_result = next(result for i, result in successful_scrapes if i == original_idx)
            ai_result = ai_results[j]

            if isinstance(ai_result, Exception):
                LOG.error(f"BATCH AI ERROR: Article {original_idx} failed: {ai_result}")
                ai_summary = None
                ai_model = None
            else:
                # ai_result is tuple (summary, model_used)
                ai_summary, ai_model = ai_result if isinstance(ai_result, tuple) else (ai_result, "unknown")

            results.append({
                "article_id": articles_batch[original_idx]["id"],
                "article_idx": original_idx,
                "success": True,
                "scraped_content": scrape_result["scraped_content"],
                "ai_summary": ai_summary,
                "ai_model": ai_model,
                "content_scraped_at": scrape_result["content_scraped_at"],
                "scraping_error": None
            })
    
    # Add failed scraping results
    for i, result in enumerate(scraping_results):
        if isinstance(result, Exception) or not result["success"]:
            if not any(r["article_id"] == articles_batch[i]["id"] for r in results):
                results.append({
                    "article_id": articles_batch[i]["id"],
                    "article_idx": i,
                    "success": False,
                    "error": str(result) if isinstance(result, Exception) else result.get("error", "Unknown error"),
                    "scraped_content": None,
                    "ai_summary": None
                })
    
    # Phase 3: Batch database update
    LOG.info(f"BATCH PHASE 3: Database update for {len(results)} articles")
    successful_updates = 0
    
    try:
        with db() as conn, conn.cursor() as cur:
            for result in results:
                if result["success"]:
                    article = articles_batch[result["article_idx"]]

                    clean_content = clean_null_bytes(result["scraped_content"]) if result["scraped_content"] else None
                    clean_summary = clean_null_bytes(result["ai_summary"]) if result["ai_summary"] else None

                    # First ensure article exists and get its ID
                    article_id = insert_article_if_new(
                        article.get("url_hash"), article.get("url"), article.get("title"),
                        article.get("description"), article.get("domain"),
                        article.get("published_at"), article.get("resolved_url")
                    )

                    # Update article with scraped content and error status
                    if article_id:
                        update_article_content(
                            article_id, clean_content, None,
                            None, False, None
                        )

                        # Ensure ticker relationship exists (don't pass category - it's immutable)
                        link_article_to_ticker(
                            article_id, analysis_ticker,
                            search_keyword=article.get("search_keyword"),
                            competitor_ticker=article.get("competitor_ticker")
                        )

                        # Update ticker-specific AI summary (POV-based)
                        if clean_summary and result.get("ai_model"):
                            update_ticker_article_summary(
                                analysis_ticker, article_id, clean_summary, result.get("ai_model")
                            )

                        successful_updates += 1
                else:
                    # Update with scraping failure
                    article = articles_batch[result["article_idx"]]

                    # First ensure article exists and get its ID
                    article_id = insert_article_if_new(
                        article.get("url_hash"), article.get("url"), article.get("title"),
                        article.get("description"), article.get("domain"),
                        article.get("published_at"), article.get("resolved_url")
                    )

                    # Update article with scraping failure
                    if article_id:
                        update_article_content(
                            article_id, None, None, None,
                            True, clean_null_bytes(result.get("error", ""))
                        )

                        # Ensure ticker relationship exists (don't pass category - it's immutable)
                        link_article_to_ticker(
                            article_id, analysis_ticker,
                            search_keyword=article.get("search_keyword"),
                            competitor_ticker=article.get("competitor_ticker")
                        )
        
        LOG.info(f"BATCH COMPLETE: {successful_updates}/{len(results)} articles successfully updated in database")
        
    except Exception as e:
        LOG.error(f"BATCH DATABASE ERROR: Failed to update batch results: {e}")
    
    return results

async def scrape_single_article_async(article: Dict, category: str, metadata: Dict, analysis_ticker: str, article_idx: int) -> Dict:
    """Scrape a single article asynchronously"""
    try:
        resolved_url = article.get("resolved_url") or article.get("url")
        domain = article.get("domain", "unknown")
        title = article.get("title", "")
        
        # Get keyword for limit tracking
        if category == "company":
            keyword = analysis_ticker
        elif category == "competitor":
            keyword = article.get("competitor_ticker", "unknown")
            if keyword == "unknown":
                keyword = article.get("search_keyword", "unknown")
        else:
            keyword = article.get("search_keyword", "unknown")
        
        if resolved_url and resolved_url.startswith(('http://', 'https://')):
            scrape_domain = normalize_domain(urlparse(resolved_url).netloc.lower())
            
            if scrape_domain not in PAYWALL_DOMAINS and scrape_domain not in PROBLEMATIC_SCRAPE_DOMAINS:
                content, status = await safe_content_scraper_with_3tier_fallback_async(
                    resolved_url, scrape_domain, category, keyword, set()
                )
                
                if content:
                    return {
                        "success": True,
                        "scraped_content": content,
                        "content_scraped_at": datetime.now(timezone.utc),
                        "error": None
                    }
                else:
                    return {
                        "success": False,
                        "error": status or "Failed to scrape content",
                        "scraped_content": None
                    }
            else:
                return {
                    "success": False,
                    "error": f"Skipped problematic domain: {scrape_domain}",
                    "scraped_content": None
                }
        else:
            return {
                "success": False,
                "error": "Invalid or missing URL",
                "scraped_content": None
            }
            
    except Exception as e:
        LOG.error(f"Article scraping failed for article {article_idx}: {e}")
        return {
            "success": False,
            "error": f"Exception during scraping: {str(e)}",
            "scraped_content": None
        }

async def generate_ai_summary_for_scraped_article(scraped_content: str, title: str, ticker: str, description: str = "") -> Optional[str]:
    """Generate AI summary for a scraped article"""
    try:
        return await generate_ai_individual_summary_async(scraped_content, title, ticker, description)
    except Exception as e:
        LOG.error(f"AI summary generation failed: {e}")
        return None

def update_scraping_stats(category: str, keyword: str, success: bool):
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
        LOG.warning(f"⚠️ No config found for {ticker} - using minimal limits")
        return {"company": 20, "industry_total": 0, "competitor_total": 0}

    # Get actual counts
    industry_keywords = config.get("industry_keywords", [])
    competitors = config.get("competitors", [])

    # DIAGNOSTIC: Log what we actually found
    LOG.info(f"📊 METADATA READ for {ticker}:")
    LOG.info(f"   AI Generated: {config.get('ai_generated', False)}")
    LOG.info(f"   Industry Keywords: {industry_keywords}")
    LOG.info(f"   Competitors: {competitors}")

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
                    # Check if article already exists
                    cur.execute("""
                        SELECT a.id FROM articles a
                        JOIN ticker_articles ta ON a.id = ta.article_id
                        WHERE a.url_hash = %s AND ta.ticker = %s
                    """, (url_hash, feed["ticker"]))
                    existing_article = cur.fetchone()

                    if existing_article:
                        # Article already exists for this ticker
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
                        # No QB fallback - triage handles selection
                        quality_score = None
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
                    
                    # Insert article and link to ticker
                    article_id = insert_article_if_new(
                        url_hash, url, title, display_content,
                        final_domain, published_at, final_resolved_url
                    )

                    if article_id:
                        # Update with scraped content and AI summary if available
                        if scraped_content or ai_summary:
                            update_article_content(
                                article_id, scraped_content, ai_summary,
                                scraping_failed, scraping_error
                            )

                        # Link article to ticker
                        link_article_to_ticker(
                            article_id, feed["ticker"], category,
                            feed["id"], feed.get("search_keyword"),
                            feed.get("competitor_ticker")
                        )

                    if article_id:
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
    
    # Use competitor_ticker for competitor feeds, search_keyword for others - FIXED: Consistent logic
    if category == "competitor":
        feed_keyword = feed.get("competitor_ticker")
        if not feed_keyword:
            feed_keyword = feed.get("search_keyword", "unknown")
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

                # CHECK SPAM DOMAINS AFTER RESOLUTION
                if final_domain and final_domain in SPAM_DOMAINS:
                    stats["blocked_spam"] += 1
                    LOG.debug(f"SPAM DOMAIN BLOCKED: {final_domain} - {title[:50]}...")
                    continue

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
                        # Check if article already exists
                        cur.execute("""
                            SELECT a.id FROM articles a
                            JOIN ticker_articles ta ON a.id = ta.article_id
                            WHERE a.url_hash = %s AND ta.ticker = %s
                        """, (url_hash, feed["ticker"]))
                        if cur.fetchone():
                            stats["duplicates"] += 1
                            LOG.debug(f"DATABASE DUPLICATE SKIPPED: {title[:50]}... (already in database)")
                            continue
                        
                        # Count existing articles for this category/keyword combination
                        if category == "company":
                            cur.execute("""
                                SELECT COUNT(DISTINCT a.id) as count FROM articles a
                                JOIN ticker_articles ta ON a.id = ta.article_id
                                WHERE ta.ticker = %s AND ta.category = 'company'
                            """, (feed["ticker"],))
                        elif category == "industry":
                            cur.execute("""
                                SELECT COUNT(DISTINCT a.id) as count FROM articles a
                                JOIN ticker_articles ta ON a.id = ta.article_id
                                WHERE ta.ticker = %s AND ta.category = 'industry' AND ta.search_keyword = %s
                            """, (feed["ticker"], feed_keyword))
                        elif category == "competitor":
                            cur.execute("""
                                SELECT COUNT(DISTINCT a.id) as count FROM articles a
                                JOIN ticker_articles ta ON a.id = ta.article_id
                                WHERE ta.ticker = %s AND ta.category = 'competitor' AND ta.competitor_ticker = %s
                            """, (feed["ticker"], feed_keyword))
                        
                        result = cur.fetchone()
                        existing_count = result["count"] if result and result["count"] is not None else 0

                        # Check if we've already hit the limit (BEFORE inserting this article)
                        limit_reached = False
                        if category == "company" and existing_count >= ingestion_stats["limits"]["company"]:
                            limit_reached = True
                            LOG.info(f"COMPANY LIMIT ALREADY REACHED: {existing_count} >= {ingestion_stats['limits']['company']} - stopping ingestion for this feed")
                            break
                        elif category == "industry" and existing_count >= ingestion_stats["limits"]["industry_per_keyword"]:
                            limit_reached = True
                            LOG.info(f"INDUSTRY LIMIT ALREADY REACHED for '{feed_keyword}': {existing_count} >= {ingestion_stats['limits']['industry_per_keyword']} - stopping ingestion")
                            break
                        elif category == "competitor" and existing_count >= ingestion_stats["limits"]["competitor_per_keyword"]:
                            limit_reached = True
                            LOG.info(f"COMPETITOR LIMIT ALREADY REACHED for '{feed_keyword}': {existing_count} >= {ingestion_stats['limits']['competitor_per_keyword']} - stopping ingestion")
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
                        
                        # Insert article if new, then link to ticker
                        article_id = insert_article_if_new(
                            url_hash, clean_url, clean_title, clean_description,
                            final_domain, published_at, clean_resolved_url
                        )

                        if article_id:
                            link_article_to_ticker(
                                article_id, feed["ticker"], category,
                                feed["id"], clean_search_keyword, clean_competitor_ticker
                            )
                            stats["inserted"] += 1
                            # Get current count after insertion
                            limit_key = 'company' if category == 'company' else f'{category}_per_keyword'
                            current_limit = ingestion_stats['limits'][limit_key]
                            LOG.info(f"INSERTED [{category}]: Article inserted (limit: {current_limit}) - {title[:50]}...")
                        else:
                            LOG.info(f"DUPLICATE SKIPPED: {title[:30]}")
                            
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
    """Deprecated - schema already created in ensure_schema()"""
    pass

# Updated article formatting function
def _format_article_html_with_ai_summary(article: Dict, category: str, ticker_metadata_cache: Dict = None,
                                         show_ai_analysis: bool = True, show_descriptions: bool = True) -> str:
    """
    Enhanced article HTML formatting with AI summaries and proper left-side headers

    Args:
        article: Article dictionary
        category: Article category (company/industry/competitor)
        ticker_metadata_cache: Cache of ticker metadata
        show_ai_analysis: If True, show AI analysis boxes (default True)
        show_descriptions: If True, show article descriptions (default True)
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
    header_badges.append(f'<span class="source-badge">📰 {display_source}</span>')

    # 3. AI Model badge (if AI summary exists and model is not "none")
    ai_model = article.get('ai_model') or ''
    ai_model_clean = ai_model.strip().lower() if ai_model else ''
    if ai_model_clean and ai_model_clean != 'none':
        header_badges.append(f'<span class="ai-model-badge">🤖 {ai_model}</span>')

    # 4. Quality badge for quality domains
    if normalize_domain(resolved_domain) in QUALITY_DOMAINS:
        header_badges.append('<span class="quality-badge">⭐ Quality</span>')

    # 5. Analysis badge if both content and summary exist
    analyzed_html = ""
    if article.get('scraped_content') and article.get('ai_summary'):
        analyzed_html = f'<span class="analyzed-badge">Analyzed</span>'
        header_badges.append(analyzed_html)
    
    # AI Summary section - check for ai_summary field (conditional based on show_ai_analysis)
    ai_summary_html = ""
    if show_ai_analysis and article.get("ai_summary"):
        clean_summary = html.escape(article["ai_summary"].strip())
        ai_summary_html = f"<br><div class='ai-summary'><strong>📊 Analysis:</strong> {clean_summary}</div>"

    # Get description and format it (only if no AI summary shown, conditional based on show_descriptions)
    description_html = ""
    if show_descriptions and not article.get("ai_summary") and article.get("description"):
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

def get_or_create_ticker_metadata(ticker: str, force_refresh: bool = False) -> Dict:
    """Wrapper for backward compatibility"""
    return ticker_manager.get_or_create_metadata(ticker, force_refresh)

def build_feed_urls(ticker: str, keywords: Dict) -> List[Dict]:
    """NEW ARCHITECTURE: Build feed URLs based on ticker metadata (without creating feeds)"""
    feeds = []
    company_name = keywords.get("company_name", ticker)

    # 1. Company feeds (2 feeds)
    feeds.extend([
        {
            "url": f"https://news.google.com/rss/search?q=\"{company_name.replace(' ', '%20')}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
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
    ])

    # 2. Industry feeds (up to 3)
    industry_keywords = keywords.get("industry_keywords", [])[:3]
    for keyword in industry_keywords:
        feeds.append({
            "url": f"https://news.google.com/rss/search?q=\"{keyword.replace(' ', '%20')}\"+when:7d&hl=en-US&gl=US&ceid=US:en",
            "name": f"Industry: {keyword}",
            "category": "industry",
            "search_keyword": keyword
        })

    # 3. Competitor feeds (up to 3)
    competitors = keywords.get("competitors", [])[:3]
    for comp in competitors:
        if isinstance(comp, dict) and comp.get('name') and comp.get('ticker'):
            comp_name = comp['name']
            comp_ticker = comp['ticker']

            feeds.extend([
                {
                    "url": f"https://news.google.com/rss/search?q=\"{comp_name.replace(' ', '%20')}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
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
            ])

    return feeds
    
def upsert_feed(url: str, name: str, ticker: str, category: str = "company",
                retain_days: int = 90, search_keyword: str = None,
                competitor_ticker: str = None) -> int:
    """NEW ARCHITECTURE V2: Feed upsert with category per ticker-feed relationship"""
    LOG.info(f"DEBUG: Upserting feed (NEW ARCHITECTURE V2) - ticker: {ticker}, name: {name}, category: {category}, search_keyword: {search_keyword}")

    try:
        # Use NEW ARCHITECTURE V2 functions directly (no import needed)
        # Create/get feed in new architecture (NO CATEGORY in feed itself)
        feed_id = upsert_feed_new_architecture(
            url=url,
            name=name,
            search_keyword=search_keyword,
            competitor_ticker=competitor_ticker,
            retain_days=retain_days
        )

        # Associate ticker with feed WITH SPECIFIC CATEGORY for this relationship
        if associate_ticker_with_feed_new_architecture(ticker, feed_id, category):
            LOG.info(f"DEBUG: Feed upsert SUCCESS (NEW ARCHITECTURE V2) - ID: {feed_id}, category: {category}")
            return feed_id
        else:
            LOG.error(f"Failed to associate ticker {ticker} with feed {feed_id}")
            return -1

    except Exception as e:
        LOG.error(f"Error upserting feed for {ticker} (NEW ARCHITECTURE): {e}")
        return -1

def list_active_feeds(tickers: List[str] = None) -> List[Dict]:
    """NEW ARCHITECTURE V2: Get all active feeds with per-relationship categories"""
    with db() as conn, conn.cursor() as cur:
        if tickers:
            cur.execute("""
                SELECT f.id, f.url, f.name, tf.ticker, f.retain_days, tf.category, f.search_keyword, f.competitor_ticker
                FROM feeds f
                JOIN ticker_feeds tf ON f.id = tf.feed_id
                WHERE f.active = TRUE AND tf.active = TRUE AND tf.ticker = ANY(%s)
                ORDER BY tf.ticker, tf.category, f.id
            """, (tickers,))
        else:
            cur.execute("""
                SELECT f.id, f.url, f.name, tf.ticker, f.retain_days, tf.category, f.search_keyword, f.competitor_ticker
                FROM feeds f
                JOIN ticker_feeds tf ON f.id = tf.feed_id
                WHERE f.active = TRUE AND tf.active = TRUE
                ORDER BY tf.ticker, tf.category, f.id
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
    """Enhanced Yahoo Finance source extraction - handles all Yahoo Finance domains and filters out video URLs"""
    try:
        # Expand the domain check to include regional Yahoo Finance
        if not any(domain in url for domain in ["finance.yahoo.com", "ca.finance.yahoo.com", "uk.finance.yahoo.com"]):
            return None
        
        # Skip Yahoo author pages, video files, media.zenfs.com redirects, AND video.media.yql.yahoo.com
        skip_patterns = [
            "/author/", "yahoo-finance-video", ".mp4", ".avi", ".mov",
            "video.media.yql.yahoo.com",  # Block video URLs
            "media.zenfs.com"  # Block media.zenfs.com redirects
        ]
        if any(skip_pattern in url for skip_pattern in skip_patterns):
            LOG.info(f"Skipping Yahoo video/author/zenfs page: {url}")
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
            # Pattern 2: Escaped JSON patterns
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
                                'video.media.yql.yahoo.com' not in candidate_url and  # Block video URLs
                                'media.zenfs.com' not in candidate_url and  # Block media.zenfs.com redirects
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

        # If no redirect found, keep the original Yahoo URL
        LOG.info(f"No redirect found, keeping original Yahoo URL: {url}")
        return url
        
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

# QB fallback scoring removed - triage now handles all article selection

async def generate_ai_individual_summary_async(scraped_content: str, title: str, ticker: str, description: str = "") -> Optional[str]:
    """Async version of AI summary generation with semaphore control"""
    if not OPENAI_API_KEY or not scraped_content or len(scraped_content.strip()) < 200:
        LOG.warning(f"AI summary generation skipped - API key: {bool(OPENAI_API_KEY)}, content length: {len(scraped_content) if scraped_content else 0}")
        return None
    
    # Ensure async semaphores are initialized
    init_async_semaphores()
    
    async with async_openai_sem:
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
            
            # Use asyncio-compatible HTTP client
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
                async with session.post(OPENAI_API_URL, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        u = result.get("usage", {}) or {}
                        LOG.info("AI Enhanced Summary usage – input:%s output:%s (cap:%s) status:%s reason:%s",
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
                        error_text = await response.text()
                        LOG.error(f"AI summary API error {response.status} for {ticker}: {error_text}")
                        return None
                        
        except Exception as e:
            LOG.error(f"AI enhanced summary generation failed for {ticker}: {e}")
            return None

# ===== CATEGORIZED AI SUMMARIZATION WITH CLAUDE PRIMARY, OPENAI FALLBACK =====

# Content character limit for AI summarization
CONTENT_CHAR_LIMIT = 10000

async def generate_claude_article_summary(company_name: str, ticker: str, title: str, scraped_content: str) -> Optional[str]:
    """Generate Claude summary for company article - POV agnostic"""
    if not ANTHROPIC_API_KEY or not scraped_content or len(scraped_content.strip()) < 200:
        return None

    init_async_semaphores()
    async with async_claude_sem:
        try:
            prompt = f"""You are a hedge fund analyst writing a factual memo on {company_name} ({ticker}). Analyze this article and write a summary using ONLY facts explicitly stated in the text.

**Focus:** This article is about {company_name}'s operations, financials, and strategic actions. Extract all material facts about {company_name}.

**Content Priority (address only what article contains):**
- Financial metrics: Revenue, margins, EBITDA, FCF, growth rates, guidance with exact time periods
- Strategic actions: M&A, partnerships, products, capacity changes, buybacks, dividends with dollar amounts and dates
- Competitive dynamics: How competitors are discussed in relation to {company_name}
- Industry developments: Regulatory changes, supply chain shifts, sector trends affecting {company_name}
- Analyst actions: Firm name, rating, price target, rationale for {company_name}
- Administrative: Earnings dates, regulatory deadlines, completion timelines

**Structure (no headers in output):**
Write 2-6 paragraphs in natural prose. Scale to article depth. Lead with most material information.

**Hard Rules:**
- Every number MUST have: time period, units, comparison basis
- Cite sources in parentheses using domain name only: (Reuters), (Business Wire)
- FORBIDDEN words: may, could, likely, appears, positioned, poised, expect (unless quoting), estimate (unless quoting), assume, suggests, catalyst
- NO inference beyond explicit guidance/commentary
- Each sentence must add new factual information

TARGET: {company_name} ({ticker})
TITLE: {title}
CONTENT: {scraped_content[:CONTENT_CHAR_LIMIT]}"""

            headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            data = {"model": ANTHROPIC_MODEL, "max_tokens": 8192, "messages": [{"role": "user", "content": prompt}]}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
                async with session.post(ANTHROPIC_API_URL, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        summary = result.get("content", [{}])[0].get("text", "")
                        if summary and len(summary.strip()) > 10:
                            LOG.info(f"Claude company summary: {ticker} ({len(summary)} chars)")
                            return summary.strip()
                    else:
                        LOG.error(f"Claude company API error {response.status}")
        except Exception as e:
            LOG.error(f"Claude company summary failed for {ticker}: {e}")
    return None


async def generate_claude_competitor_article_summary(competitor_name: str, competitor_ticker: str, target_company: str, target_ticker: str, title: str, scraped_content: str) -> Optional[str]:
    """Generate Claude summary for competitor article with target company POV"""
    if not ANTHROPIC_API_KEY or not scraped_content or len(scraped_content.strip()) < 200:
        return None

    init_async_semaphores()
    async with async_claude_sem:
        try:
            prompt = f"""You are a hedge fund analyst evaluating how {competitor_name} ({competitor_ticker}) developments affect {target_company} ({target_ticker}) investors. Analyze this article and write a summary using ONLY facts explicitly stated in the text.

**Focus:** This article is about {competitor_name}'s operations, but your analysis must explain competitive implications for {target_company}. Extract facts about {competitor_name} AND assess impact on {target_company}'s competitive position.

**Content Priority (address only what article contains):**
- Financial metrics: {competitor_name}'s revenue, margins, EBITDA, FCF, growth rates, guidance with exact time periods → Explain competitive pressure/opportunity for {target_company}
- Strategic actions: {competitor_name}'s M&A, partnerships, products, capacity changes → Assess how this shifts competitive landscape for {target_company}
- Market positioning: {competitor_name}'s market share gains/losses, pricing actions → Quantify threat/benefit to {target_company}'s position
- Technology/products: {competitor_name}'s launches, capabilities → Compare to {target_company}'s offerings, identify competitive gaps/advantages
- Analyst actions: Firm name, rating, price target on {competitor_name} → Contextualize relative to {target_company}'s valuation/sentiment
- Operational changes: {competitor_name}'s capacity, efficiency, cost structure → Implications for {target_company}'s competitive cost position

**Structure (no headers in output):**
Write 2-6 paragraphs in natural prose. Scale to article depth. Lead with competitive implications for {target_company}, then supporting {competitor_name} facts.

**Competitive Impact Framework:**
- If {competitor_name} gains advantage → explain pressure on {target_company} (market share, pricing power, margins)
- If {competitor_name} faces challenges → explain opportunity for {target_company} (market share capture, pricing leverage)
- If neutral development → explain why it doesn't change {target_company}'s competitive position

**Hard Rules:**
- Every number MUST have: time period, units, comparison basis
- Cite sources in parentheses using domain name only: (Reuters), (Bloomberg)
- FORBIDDEN words: may, could, likely, appears, positioned, poised, expect (unless quoting), estimate (unless quoting), assume, suggests, catalyst
- NO inference beyond explicit guidance/commentary
- Each paragraph must connect {competitor_name} facts to {target_company} competitive impact

TARGET COMPANY: {target_company} ({target_ticker})
COMPETITOR: {competitor_name} ({competitor_ticker})
TITLE: {title}
CONTENT: {scraped_content[:CONTENT_CHAR_LIMIT]}"""

            headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            data = {"model": ANTHROPIC_MODEL, "max_tokens": 8192, "messages": [{"role": "user", "content": prompt}]}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
                async with session.post(ANTHROPIC_API_URL, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        summary = result.get("content", [{}])[0].get("text", "")
                        if summary and len(summary.strip()) > 10:
                            LOG.info(f"Claude competitor summary: {competitor_ticker} ({len(summary)} chars)")
                            return summary.strip()
                    else:
                        LOG.error(f"Claude competitor API error {response.status}")
        except Exception as e:
            LOG.error(f"Claude competitor summary failed for {competitor_ticker}: {e}")
    return None


async def generate_claude_industry_article_summary(industry_keyword: str, target_company: str, target_ticker: str, title: str, scraped_content: str) -> Optional[str]:
    """Generate Claude summary for industry article with target company POV"""
    if not ANTHROPIC_API_KEY or not scraped_content or len(scraped_content.strip()) < 200:
        return None

    init_async_semaphores()
    async with async_claude_sem:
        try:
            prompt = f"""You are a hedge fund analyst evaluating how {industry_keyword} sector developments affect {target_company} ({target_ticker}). Analyze this article and write a summary using ONLY facts explicitly stated in the text.

**Focus:** This article contains {industry_keyword} industry insights, but your analysis must explain specific implications for {target_company}'s operations, costs, demand, or competitive position.

**Content Priority (address only what article contains):**
- Market dynamics: TAM/SAM sizing, growth rates, adoption trends with specific figures → How this affects {target_company}'s addressable market and growth runway
- Technology developments: Product launches, performance benchmarks, standards adoption → Impact on {target_company}'s product roadmap, R&D priorities, or competitive differentiation
- Competitive landscape: Market share data, company positioning, partnerships → Where {target_company} stands relative to sector trends
- Financial metrics: Aggregate sector revenue/growth OR company metrics when comparing → {target_company}'s performance vs. sector benchmarks
- Regulatory/policy: Government actions, standards, trade restrictions with dates and amounts → Compliance costs, competitive advantages, or operational constraints for {target_company}
- Supply chain: Manufacturing capacity, component availability, pricing trends → Impact on {target_company}'s input costs, production capacity, or delivery timelines

**Structure (no headers in output):**
Write 2-6 paragraphs in natural prose. Scale to article depth. Lead with {target_company}-specific implications, then supporting sector facts.

**Impact Analysis Framework:**
- Supply-side changes → explain effect on {target_company}'s costs, capacity, or supply security
- Demand-side changes → explain effect on {target_company}'s revenue opportunities or pricing power
- Regulatory changes → explain {target_company}'s compliance burden or competitive positioning shift
- Competitive dynamics → explain where {target_company} wins/loses from sector trends
- Technology shifts → explain whether {target_company} is ahead/behind the curve

**Hard Rules:**
- Every number MUST have: time period, units, comparison basis
- Cite sources in parentheses using domain name only: (WSJ), (FT)
- FORBIDDEN words: may, could, likely, appears, positioned, expect (unless quoting), estimate (unless quoting), catalyst
- NO inference beyond explicit projections/guidance
- Each paragraph must connect sector facts to {target_company}-specific impact

TARGET COMPANY: {target_company} ({target_ticker})
SECTOR FOCUS: {industry_keyword}
TITLE: {title}
CONTENT: {scraped_content[:CONTENT_CHAR_LIMIT]}"""

            headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            data = {"model": ANTHROPIC_MODEL, "max_tokens": 8192, "messages": [{"role": "user", "content": prompt}]}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
                async with session.post(ANTHROPIC_API_URL, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        summary = result.get("content", [{}])[0].get("text", "")
                        if summary and len(summary.strip()) > 10:
                            LOG.info(f"Claude industry summary: {industry_keyword} for {target_ticker} ({len(summary)} chars)")
                            return summary.strip()
                    else:
                        LOG.error(f"Claude industry API error {response.status}")
        except Exception as e:
            LOG.error(f"Claude industry summary failed for {industry_keyword}/{target_ticker}: {e}")
    return None


async def generate_openai_article_summary(company_name: str, ticker: str, title: str, scraped_content: str) -> Optional[str]:
    """OpenAI fallback for company article"""
    if not OPENAI_API_KEY or not scraped_content or len(scraped_content.strip()) < 200:
        return None

    init_async_semaphores()
    async with async_openai_sem:
        try:
            prompt = f"""You are a hedge fund analyst writing a factual memo on {company_name} ({ticker}). Write a summary using ONLY facts explicitly stated.

**Focus:** Extract material facts about {company_name}'s operations, financials, strategic actions.

**Include:** Financial metrics with time periods, strategic actions with amounts/dates, analyst actions, administrative dates.

**Hard Rules:**
- Every number needs time period, units, comparison basis
- Cite sources in parentheses (domain name only)
- NO speculation words: may, could, likely, appears, positioned
- 4-8 sentences, no preamble

TARGET: {company_name} ({ticker})
TITLE: {title}
CONTENT: {scraped_content[:CONTENT_CHAR_LIMIT]}"""

            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            data = {"model": OPENAI_MODEL, "input": prompt, "max_output_tokens": 8000, "reasoning": {"effort": "medium"}, "text": {"verbosity": "low"}, "truncation": "auto"}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
                async with session.post(OPENAI_API_URL, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        summary = extract_text_from_responses(result)
                        if summary and len(summary.strip()) > 10:
                            LOG.info(f"OpenAI company summary: {ticker} ({len(summary)} chars)")
                            return summary.strip()
                    else:
                        LOG.error(f"OpenAI company API error {response.status}")
        except Exception as e:
            LOG.error(f"OpenAI company summary failed for {ticker}: {e}")
    return None


async def generate_openai_competitor_article_summary(competitor_name: str, competitor_ticker: str, target_company: str, target_ticker: str, title: str, scraped_content: str) -> Optional[str]:
    """OpenAI fallback for competitor article with target company POV"""
    if not OPENAI_API_KEY or not scraped_content or len(scraped_content.strip()) < 200:
        return None

    init_async_semaphores()
    async with async_openai_sem:
        try:
            prompt = f"""You are a hedge fund analyst evaluating how {competitor_name} ({competitor_ticker}) developments affect {target_company} ({target_ticker}) investors. Analyze this article and write a summary using ONLY facts explicitly stated in the text.

**Focus:** This article is about {competitor_name}'s operations, but your analysis must explain competitive implications for {target_company}. Extract facts about {competitor_name} AND assess impact on {target_company}'s competitive position.

**Content Priority (address only what article contains):**
- Financial metrics: {competitor_name}'s revenue, margins, EBITDA, FCF, growth rates, guidance with exact time periods → Explain competitive pressure/opportunity for {target_company}
- Strategic actions: {competitor_name}'s M&A, partnerships, products, capacity changes → Assess how this shifts competitive landscape for {target_company}
- Market positioning: {competitor_name}'s market share gains/losses, pricing actions → Quantify threat/benefit to {target_company}'s position
- Technology/products: {competitor_name}'s launches, capabilities → Compare to {target_company}'s offerings, identify competitive gaps/advantages
- Analyst actions: Firm name, rating, price target on {competitor_name} → Contextualize relative to {target_company}'s valuation/sentiment
- Operational changes: {competitor_name}'s capacity, efficiency, cost structure → Implications for {target_company}'s competitive cost position

**Structure (no headers in output):**
Write 2-6 paragraphs in natural prose. Scale to article depth. Lead with competitive implications for {target_company}, then supporting {competitor_name} facts.

**Competitive Impact Framework:**
- If {competitor_name} gains advantage → explain pressure on {target_company} (market share, pricing power, margins)
- If {competitor_name} faces challenges → explain opportunity for {target_company} (market share capture, pricing leverage)
- If neutral development → explain why it doesn't change {target_company}'s competitive position

**Hard Rules:**
- Every number MUST have: time period, units, comparison basis
- Cite sources in parentheses using domain name only: (Reuters), (Bloomberg)
- FORBIDDEN words: may, could, likely, appears, positioned, poised, expect (unless quoting), estimate (unless quoting), assume, suggests, catalyst
- NO inference beyond explicit guidance/commentary
- Each paragraph must connect {competitor_name} facts to {target_company} competitive impact

TARGET COMPANY: {target_company} ({target_ticker})
COMPETITOR: {competitor_name} ({competitor_ticker})
TITLE: {title}
CONTENT: {scraped_content[:CONTENT_CHAR_LIMIT]}"""

            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            data = {"model": OPENAI_MODEL, "input": prompt, "max_output_tokens": 8000, "reasoning": {"effort": "medium"}, "text": {"verbosity": "low"}, "truncation": "auto"}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
                async with session.post(OPENAI_API_URL, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        summary = extract_text_from_responses(result)
                        if summary and len(summary.strip()) > 10:
                            LOG.info(f"OpenAI competitor summary: {competitor_ticker} ({len(summary)} chars)")
                            return summary.strip()
                    else:
                        LOG.error(f"OpenAI competitor API error {response.status}")
        except Exception as e:
            LOG.error(f"OpenAI competitor summary failed for {competitor_ticker}: {e}")
    return None


async def generate_openai_industry_article_summary(industry_keyword: str, target_company: str, target_ticker: str, title: str, scraped_content: str) -> Optional[str]:
    """OpenAI fallback for industry article with target company POV"""
    if not OPENAI_API_KEY or not scraped_content or len(scraped_content.strip()) < 200:
        return None

    init_async_semaphores()
    async with async_openai_sem:
        try:
            prompt = f"""You are a hedge fund analyst evaluating how {industry_keyword} sector developments affect {target_company} ({target_ticker}). Analyze this article and write a summary using ONLY facts explicitly stated in the text.

**Focus:** This article contains {industry_keyword} industry insights, but your analysis must explain specific implications for {target_company}'s operations, costs, demand, or competitive position.

**Content Priority (address only what article contains):**
- Market dynamics: TAM/SAM sizing, growth rates, adoption trends with specific figures → How this affects {target_company}'s addressable market and growth runway
- Technology developments: Product launches, performance benchmarks, standards adoption → Impact on {target_company}'s product roadmap, R&D priorities, or competitive differentiation
- Competitive landscape: Market share data, company positioning, partnerships → Where {target_company} stands relative to sector trends
- Financial metrics: Aggregate sector revenue/growth OR company metrics when comparing → {target_company}'s performance vs. sector benchmarks
- Regulatory/policy: Government actions, standards, trade restrictions with dates and amounts → Compliance costs, competitive advantages, or operational constraints for {target_company}
- Supply chain: Manufacturing capacity, component availability, pricing trends → Impact on {target_company}'s input costs, production capacity, or delivery timelines

**Structure (no headers in output):**
Write 2-6 paragraphs in natural prose. Scale to article depth. Lead with {target_company}-specific implications, then supporting sector facts.

**Impact Analysis Framework:**
- Supply-side changes → explain effect on {target_company}'s costs, capacity, or supply security
- Demand-side changes → explain effect on {target_company}'s revenue opportunities or pricing power
- Regulatory changes → explain {target_company}'s compliance burden or competitive positioning shift
- Competitive dynamics → explain where {target_company} wins/loses from sector trends
- Technology shifts → explain whether {target_company} is ahead/behind the curve

**Hard Rules:**
- Every number MUST have: time period, units, comparison basis
- Cite sources in parentheses using domain name only: (WSJ), (FT)
- FORBIDDEN words: may, could, likely, appears, positioned, expect (unless quoting), estimate (unless quoting), catalyst
- NO inference beyond explicit projections/guidance
- Each paragraph must connect sector facts to {target_company}-specific impact

TARGET COMPANY: {target_company} ({target_ticker})
SECTOR FOCUS: {industry_keyword}
TITLE: {title}
CONTENT: {scraped_content[:CONTENT_CHAR_LIMIT]}"""

            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            data = {"model": OPENAI_MODEL, "input": prompt, "max_output_tokens": 8000, "reasoning": {"effort": "medium"}, "text": {"verbosity": "low"}, "truncation": "auto"}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
                async with session.post(OPENAI_API_URL, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        summary = extract_text_from_responses(result)
                        if summary and len(summary.strip()) > 10:
                            LOG.info(f"OpenAI industry summary: {industry_keyword} ({len(summary)} chars)")
                            return summary.strip()
                    else:
                        LOG.error(f"OpenAI industry API error {response.status}")
        except Exception as e:
            LOG.error(f"OpenAI industry summary failed for {industry_keyword}: {e}")
    return None


async def generate_claude_summary(scraped_content: str, title: str, ticker: str, category: str,
                                  article_metadata: dict, target_company_name: str,
                                  competitor_name_cache: dict) -> Optional[str]:
    """Route to appropriate Claude function based on category"""
    if category == "company":
        return await generate_claude_article_summary(target_company_name, ticker, title, scraped_content)
    elif category == "competitor":
        competitor_ticker = article_metadata.get("competitor_ticker")
        if not competitor_ticker:
            return None
        competitor_name = competitor_name_cache.get(competitor_ticker, competitor_ticker)
        return await generate_claude_competitor_article_summary(competitor_name, competitor_ticker, target_company_name, ticker, title, scraped_content)
    elif category == "industry":
        industry_keyword = article_metadata.get("search_keyword", "this industry")
        return await generate_claude_industry_article_summary(industry_keyword, target_company_name, ticker, title, scraped_content)
    return None


async def generate_openai_summary(scraped_content: str, title: str, ticker: str, category: str,
                                  article_metadata: dict, target_company_name: str,
                                  competitor_name_cache: dict) -> Optional[str]:
    """Route to appropriate OpenAI function based on category"""
    if category == "company":
        return await generate_openai_article_summary(target_company_name, ticker, title, scraped_content)
    elif category == "competitor":
        competitor_ticker = article_metadata.get("competitor_ticker")
        if not competitor_ticker:
            return None
        competitor_name = competitor_name_cache.get(competitor_ticker, competitor_ticker)
        return await generate_openai_competitor_article_summary(competitor_name, competitor_ticker, target_company_name, ticker, title, scraped_content)
    elif category == "industry":
        industry_keyword = article_metadata.get("search_keyword", "this industry")
        return await generate_openai_industry_article_summary(industry_keyword, target_company_name, ticker, title, scraped_content)
    return None


async def generate_ai_summary_with_fallback(scraped_content: str, title: str, ticker: str, description: str,
                                           category: str, article_metadata: dict, target_company_name: str,
                                           competitor_name_cache: dict) -> tuple[Optional[str], str]:
    """Main entry point: Try Claude first, fallback to OpenAI. Returns (summary, model_used)"""
    model_used = "none"
    summary = None

    # Try Claude first (if enabled and API key available)
    if USE_CLAUDE_FOR_SUMMARIES and ANTHROPIC_API_KEY:
        try:
            summary = await generate_claude_summary(
                scraped_content, title, ticker, category,
                article_metadata, target_company_name, competitor_name_cache
            )
            if summary:
                model_used = "Claude"
                return summary, model_used
            else:
                LOG.warning(f"Claude returned no summary for {ticker}, falling back to OpenAI")
        except Exception as e:
            LOG.warning(f"Claude summarization failed for {ticker}, falling back to OpenAI: {e}")

    # Fallback to OpenAI
    if OPENAI_API_KEY:
        try:
            summary = await generate_openai_summary(
                scraped_content, title, ticker, category,
                article_metadata, target_company_name, competitor_name_cache
            )
            if summary:
                model_used = "OpenAI"
                return summary, model_used
        except Exception as e:
            LOG.error(f"OpenAI summarization also failed for {ticker}: {e}")

    return None, "none"

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
    sector = config.get("sector", "") if config else ""

    # Build peers list for industry triage
    competitors = [(config.get(f"competitor_{i}_name"), config.get(f"competitor_{i}_ticker")) for i in range(1, 4) if config.get(f"competitor_{i}_name")] if config else []
    peers = []
    for comp_name, comp_ticker in competitors:
        if comp_name and comp_ticker:
            peers.append(f"{comp_name} ({comp_ticker})")
        elif comp_name:
            peers.append(comp_name)

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
                selected = triage_industry_articles_full(triage_articles, ticker, company_name, sector, peers)
                
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
    sector = config.get("sector", "") if config else ""

    # Build peers list for industry triage
    competitors = [(config.get(f"competitor_{i}_name"), config.get(f"competitor_{i}_ticker")) for i in range(1, 4) if config.get(f"competitor_{i}_name")] if config else []
    peers = []
    for comp_name, comp_ticker in competitors:
        if comp_name and comp_ticker:
            peers.append(f"{comp_name} ({comp_ticker})")
        elif comp_name:
            peers.append(comp_name)

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
        
        # Run AI triage on new articles if needed (NO LIMIT - process all AI selections)
        if needs_triage:
            try:
                triage_articles = [article for _, article in needs_triage]
                new_selected = triage_company_articles_full(triage_articles, ticker, company_name, {}, {})

                # Map back to original indices and add to selection (NO CAP - take all AI selections)
                for selected_item in new_selected:
                    original_idx = needs_triage[selected_item["id"]][0]
                    selected_item["id"] = original_idx
                    selected_item["selection_method"] = "new_ai_triage"
                    company_selected.append(selected_item)

                LOG.info(f"  COMPANY AI: {len(new_selected)} selected from new AI triage (no cap applied)")
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

            # Run AI triage on new articles if needed (NO LIMIT - process all AI selections)
            if needs_triage:
                try:
                    triage_articles = [item["article"] for item in needs_triage]
                    new_selected = triage_industry_articles_full(triage_articles, ticker, company_name, sector, peers)

                    # Map back to original indices (NO CAP - take all AI selections)
                    for selected_item in new_selected:
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

            # Run AI triage on new articles if needed (NO LIMIT - process all AI selections)
            if needs_triage:
                try:
                    triage_articles = [item["article"] for item in needs_triage]
                    new_selected = triage_competitor_articles_full(triage_articles, ticker, [], {})

                    # Map back to original indices (NO CAP - take all AI selections)
                    for selected_item in new_selected:
                        original_idx = needs_triage[selected_item["id"]]["original_idx"]
                        selected_item["id"] = original_idx
                        selected_item["selection_method"] = "new_ai_triage"
                        entity_selected.append(selected_item)

                    LOG.info(f"    AI TRIAGE: {len(new_selected)} selected (no cap applied)")
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

async def triage_company_articles_full(articles: List[Dict], ticker: str, company_name: str, aliases_brands_assets: Dict, sector_profile: Dict) -> List[Dict]:
    """Enhanced company triage with explicit selection constraints and embedded HTTP logic (ASYNC)"""
    if not OPENAI_API_KEY or not articles:
        LOG.warning("No OpenAI API key or no articles to triage")
        return []

    # Prepare items for triage (title + description)
    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        # Add description if available
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(20, len(articles))  # Explicit cap for company articles

    payload = {
        "bucket": "company",
        "target_cap": target_cap,
        "ticker": ticker,
        "company_name": company_name,
        "items": items
    }

    # Triage schema - OpenAI now supports priority 1-3 (restored tier 3)
    triage_schema = {
        "type": "object",
        "properties": {
            "selected_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": target_cap,
                "minItems": 0
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
                "maxItems": target_cap,
                "minItems": 0
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

    system_prompt = f"""You are a financial analyst selecting the {target_cap} most important articles about {company_name} ({ticker}) from {len(articles)} candidates based ONLY on titles and descriptions.

CRITICAL: Select UP TO {target_cap} articles, fewer if uncertain.

If you're unsure whether an article is relevant to {company_name}, assign 0 points rather than selecting it. Only select articles you are confident about.

SELECT (choose up to {target_cap}):

TIER 1 - Hard corporate events:
- Financial: "beats," "misses," "earnings," "revenue," "guidance," "margin," "profit," "loss," "EPS," "sales"
- Capital: "buyback," "dividend," "debt," "bond," "offering," "raises," "refinance," "converts"
- M&A: "acquires," "buys," "sells," "divests," "stake," "merger," "spin-off," "joint venture"
- Regulatory: "FDA," "SEC," "DOJ," "FTC," "investigation," "fine," "settlement," "approval," "lawsuit," "probe"
- Operations: "accident," "disaster," "halt," "shutdown," "recall," "expansion," "closure," "layoffs," "strike"
- Products: "approval," "launch," "recall," "trial results," "patent" WITH specific product/drug names
- Contracts: Dollar amounts in title (e.g., "$500M contract," "£2B deal")
- Ratings: "upgrade," "downgrade" WITH analyst firm name (e.g., "BofA upgrades," "Moody's cuts")

TIER 2 - Strategic developments:
- Leadership: CEO, CFO, President, CTO WITH "appoints," "names," "resigns," "retires," "replaces"
- Partnerships: Named partner companies (e.g., "{company_name} partners with [Other Company]")
- Technology: Specific tech/platform names WITH "launches," "announces," "deploys"
- Facilities: Plant/office/branch/store openings, closures WITH locations and capacity/headcount numbers
- Clinical: Trial phases, enrollment milestones, data releases (pharma/biotech)
- Spectrum/Licenses: Acquisitions, renewals WITH specific bands/regions (telecom)
- Geographic: Market entry/exit WITH investment levels or unit counts

TIER 3 - Context (ONLY if quota unfilled):
- Analyst coverage WITH price targets visible in title
- Industry awards, certifications if indicative of competitive position
- Routine announcements WITH material operational details

REJECT COMPLETELY - Never select:
- Generic lists: "Top," "Best," "Should You Buy," "Stocks to Watch," "X Stocks to"
- Roundups: "Sector Update," "Stock Movers," "Trending Stocks," "Biggest Analyst Calls"
- Clickbait: "This Could," "Why," "What Investors Need to Know," "How to"
- Future speculation: "Heading to," "Could reach $X," "Price Prediction 20XX"
- Historical: "If you'd invested," "20 years ago," "Where would $1000 be"
- Market reports: "Market to reach," "CAGR," "Forecast 20XX-20YY"
- Quote pages: "Stock Price | Live Quotes & Charts | [Exchange]"
- Other companies: If title clearly focuses on different company, reject

DISAMBIGUATION - Avoid confusion:
- If title leads with different company name, likely not about {company_name}
- If {company_name} only appears as news source attribution, not subject
- For common words (Oracle, Amazon, Apple), verify context matches your company

SCRAPE PRIORITY (assign integer 1-3):
1 = Tier 1 (financial results, M&A, regulatory, disasters, major contracts)
2 = Tier 2 (leadership, partnerships, product launches, facilities)
3 = Tier 3 (analyst coverage, awards, routine announcements)

For each article assess:
- likely_repeat: Same event as another selected article?
- repeat_key: Event identifier (e.g., "q2_earnings_2025," "ceo_change_sept_2025")
- confidence: 0.0-1.0, certainty this is specifically about {company_name}

CRITICAL CONSTRAINT: Return UP TO {target_cap} articles. Select fewer if you're uncertain about relevance.

CONSERVATIVE STANDARD: If you're unsure whether an article is directly relevant to {company_name}, skip it entirely. Only select articles you are confident about."""

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
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
            "input": f"{system_prompt}\n\n{json.dumps(payload, separators=(',', ':'))}",
            "max_output_tokens": 20000,
            "truncation": "auto"
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            async with session.post(OPENAI_API_URL, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    LOG.error(f"OpenAI triage API error {response.status}: {error_text}")
                    return []
                
                result = await response.json()
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
                    if selected_item["id"] < len(articles):
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
                if len(selected_articles) > target_cap:
                    LOG.warning(f"Code-level cap enforcement: Trimming {len(selected_articles)} to {target_cap}")
                    selected_articles = selected_articles[:target_cap]
                
                return selected_articles
        
    except Exception as e:
        LOG.error(f"Async triage request failed: {str(e)}")
        return []

async def triage_industry_articles_full(articles: List[Dict], ticker: str, company_name: str, sector: str, peers: List[str]) -> List[Dict]:
    """Enhanced industry triage with explicit selection constraints and embedded HTTP logic (ASYNC)"""
    if not OPENAI_API_KEY or not articles:
        LOG.warning("No OpenAI API key or no articles to triage")
        return []

    # Prepare items for triage (title + description)
    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        # Add description if available
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(5, len(articles))

    payload = {
        "bucket": "industry",
        "target_cap": target_cap,
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "peers": peers,
        "items": items
    }

    # Triage schema - OpenAI now supports priority 1-3 (restored tier 3)
    triage_schema = {
        "type": "object",
        "properties": {
            "selected_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": target_cap,
                "minItems": 0
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
                "maxItems": target_cap,
                "minItems": 0
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

    # Format peer list for display
    peers_display = ', '.join(peers[:5]) if peers else 'None'

    system_prompt = f"""You are a financial analyst selecting the {target_cap} most important INDUSTRY articles from {len(articles)} candidates based ONLY on titles and descriptions.

TARGET COMPANY: {company_name} ({ticker})
SECTOR: {sector}
KNOWN PEERS: {peers_display}

INDUSTRY CONTEXT: Select articles about industry trends and developments that are relevant to {company_name}'s competitive landscape. These should be sector-wide insights affecting {company_name} and its peers, not articles solely about non-peer companies.

CRITICAL: Select UP TO {target_cap} articles, fewer if uncertain. If you're unsure whether an article is relevant to {company_name}'s industry position, DO NOT select it. Only select articles you are confident have industry implications for {company_name}.

SELECT (choose up to {target_cap}):

TIER 1 - Hard industry events with quantified impact (scrape_priority=1):
- Regulatory/Policy: New laws, rules, tariffs, bans, quotas WITH specific rates/dates/costs affecting {sector}
- Pricing: Commodity/service prices, reimbursement rates WITH specific figures affecting {company_name} sector
- Supply/Demand: Production disruptions, capacity changes WITH volume/value numbers impacting {sector}
- Standards: New requirements, certifications, compliance rules WITH deadlines/costs for {sector}
- Trade: Agreements, restrictions, sanctions WITH affected volumes or timelines for {sector}
- Financial: Interest rates, capital requirements, reserve rules affecting {sector}

TIER 2 - Strategic sector developments (scrape_priority=2):
- Major capacity additions/closures WITH specific impact (e.g., "500MW," "1M units/year") in {sector}
- Industry consolidation WITH transaction values and market share implications for {sector}
- Technology adoption WITH implementation timelines and cost impacts in {sector}
- Labor agreements WITH wage/benefit details affecting {sector}
- Infrastructure investments WITH budgets and completion dates for {sector}
- Patent expirations, generic approvals, technology shifts WITH market impact on {sector}
- Major peer company announcements revealing sector-wide trends (from peers: {peers_display})

TIER 3 - Sector context (ONLY if quota unfilled):
- Economic indicators directly affecting {sector} WITH specific data
- Government initiatives WITH allocated budgets (not vague "plans")
- Research findings WITH quantified sector implications

REJECT COMPLETELY - Never select:
- Market research reports: "Market to reach," "CAGR," "Forecast 20XX-20YY," "TAM," "Industry Report"
- Generic trends: "Top Trends," "Future of," "Outlook," "What to Expect in [Year]"
- Articles ONLY about non-peer companies without clear {sector} implications
- Pure company-specific news: Single-company earnings, appointments WITHOUT sector implications
- Listicles: "X Best," "How to," "Why You Should," "Reasons to"
- Opinion: "Analysis," "Commentary," "Perspective" (without hard data)
- Small company routine announcements: Financing rounds, junior partnerships, minor appointments
- Articles unrelated to {company_name}'s competitive landscape

INCLUDE when company news has sector implications for {company_name}:
✓ Major peer company action indicating sector direction
✓ Company action revealing regulatory/policy impacts on {company_name} and peers
✓ Production disruption at major player affecting {sector} supply/pricing
✓ Technology deployment showing sector-wide adoption affecting {company_name}
✓ Company data revealing {sector} cost/margin trends

SCRAPE PRIORITY (assign integer 1-3):
1 = Tier 1 (regulatory, pricing, supply shocks WITH numbers)
2 = Tier 2 (capacity, consolidation, policy WITH budgets, peer company moves with sector implications)
3 = Tier 3 (economic indicators, sector context)

For each article assess:
- likely_repeat: Same sector event covered by multiple outlets?
- repeat_key: Event identifier (e.g., "banking_regulation_sept_2025")
- confidence: 0.0-1.0, certainty this has implications for {company_name} and its competitive position

CRITICAL CONSTRAINT: Return UP TO {target_cap} articles. Select fewer if you're uncertain about relevance.

CONSERVATIVE STANDARD: If you're unsure whether an article is relevant to {company_name}'s competitive landscape in {sector}, skip it entirely. Only select articles you are confident about."""

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
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
            "input": f"{system_prompt}\n\n{json.dumps(payload, separators=(',', ':'))}",
            "max_output_tokens": 20000,
            "truncation": "auto"
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            async with session.post(OPENAI_API_URL, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    LOG.error(f"OpenAI triage API error {response.status}: {error_text}")
                    return []
                
                result = await response.json()
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
                    if selected_item["id"] < len(articles):
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
                if len(selected_articles) > target_cap:
                    LOG.warning(f"Code-level cap enforcement: Trimming {len(selected_articles)} to {target_cap}")
                    selected_articles = selected_articles[:target_cap]
                
                return selected_articles
        
    except Exception as e:
        LOG.error(f"Async triage request failed: {str(e)}")
        return []

async def triage_competitor_articles_full(articles: List[Dict], ticker: str, competitor_name: str, competitor_ticker: str) -> List[Dict]:
    """Enhanced competitor triage with explicit selection constraints and embedded HTTP logic (ASYNC)"""
    if not OPENAI_API_KEY or not articles:
        LOG.warning("No OpenAI API key or no articles to triage")
        return []

    # Prepare items for triage (title + description)
    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        # Add description if available
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(5, len(articles))

    payload = {
        "bucket": "competitor",
        "target_cap": target_cap,
        "ticker": ticker,
        "competitor_name": competitor_name,
        "competitor_ticker": competitor_ticker,
        "items": items
    }

    # Triage schema - OpenAI now supports priority 1-3 (restored tier 3)
    triage_schema = {
        "type": "object",
        "properties": {
            "selected_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": target_cap,
                "minItems": 0
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
                "maxItems": target_cap,
                "minItems": 0
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

    system_prompt = f"""You are a financial analyst for {ticker} investors selecting the {target_cap} most important articles about competitor {competitor_name} ({competitor_ticker}) from {len(articles)} candidates based ONLY on titles and descriptions.

CRITICAL: Select UP TO {target_cap} articles, fewer if uncertain.

If you're unsure whether an article is relevant to {competitor_name} ({competitor_ticker}), assign 0 points rather than selecting it. Only select articles you are confident about.

SELECT (choose up to {target_cap}):

TIER 1 - Hard corporate events:
- Financial: "beats," "misses," "earnings," "revenue," "guidance," "margin," "profit," "loss," "EPS," "sales"
- Capital: "buyback," "dividend," "debt," "bond," "offering," "raises," "refinance," "converts"
- M&A: "acquires," "buys," "sells," "divests," "stake," "merger," "spin-off," "joint venture"
- Regulatory: "FDA," "SEC," "DOJ," "FTC," "investigation," "fine," "settlement," "approval," "lawsuit," "probe"
- Operations: "accident," "disaster," "halt," "shutdown," "recall," "expansion," "closure," "layoffs," "strike"
- Products: "approval," "launch," "recall," "trial results," "patent" WITH specific product/drug names
- Contracts: Dollar amounts in title (e.g., "$500M contract," "£2B deal")
- Ratings: "upgrade," "downgrade" WITH analyst firm name (e.g., "BofA upgrades," "Moody's cuts")

TIER 2 - Strategic developments:
- Leadership: CEO, CFO, President, CTO WITH "appoints," "names," "resigns," "retires," "replaces"
- Partnerships: Named partner companies (e.g., "{competitor_name} partners with [Other Company]")
- Technology: Specific tech/platform names WITH "launches," "announces," "deploys"
- Facilities: Plant/office/branch/store openings, closures WITH locations and capacity/headcount numbers
- Clinical: Trial phases, enrollment milestones, data releases (pharma/biotech)
- Spectrum/Licenses: Acquisitions, renewals WITH specific bands/regions (telecom)
- Geographic: Market entry/exit WITH investment levels or unit counts

TIER 3 - Context (ONLY if quota unfilled):
- Analyst coverage of {competitor_name} WITH price targets
- Industry awards, certifications for {competitor_name}
- Routine announcements WITH material details

REJECT COMPLETELY - Never select:
- Generic lists: "Top," "Best," "Should You Buy," "Stocks to Watch," "X Stocks to"
- Roundups: "Sector Update," "Stock Movers," "Trending Stocks," "Biggest Analyst Calls"
- Clickbait: "This Could," "Why," "What Investors Need to Know," "How to"
- Future speculation: "Heading to," "Could reach $X," "Price Prediction 20XX"
- Historical: "If you'd invested," "20 years ago," "Where would $1000 be"
- Market reports: "Market to reach," "CAGR," "Forecast 20XX-20YY"
- Quote pages: "Stock Price | Live Quotes & Charts | [Exchange]"
- Other companies: If title clearly focuses on different company, reject

DISAMBIGUATION - Avoid confusion:
- If title leads with different company name, likely not about {competitor_name}
- If {competitor_name} only appears as news source attribution, not subject
- For common words (Oracle, Amazon, Apple), verify context matches your company

SCRAPE PRIORITY (assign integer 1-3):
1 = Tier 1 (financial results, M&A, regulatory, disasters, major contracts)
2 = Tier 2 (leadership, partnerships, product launches, facilities)
3 = Tier 3 (analyst coverage, awards, routine announcements)

For each article assess:
- likely_repeat: Same event as another selected article?
- repeat_key: Event identifier (e.g., "q2_earnings_2025," "ceo_change_sept_2025")
- confidence: 0.0-1.0, certainty this is specifically about {competitor_name}

CRITICAL CONSTRAINT: Return UP TO {target_cap} articles. Select fewer if you're uncertain about relevance.

CONSERVATIVE STANDARD: If you're unsure whether an article is directly relevant to {competitor_name} ({competitor_ticker}), skip it entirely. Only select articles you are confident about."""

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
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
            "input": f"{system_prompt}\n\n{json.dumps(payload, separators=(',', ':'))}",
            "max_output_tokens": 20000,
            "truncation": "auto"
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            async with session.post(OPENAI_API_URL, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    LOG.error(f"OpenAI triage API error {response.status}: {error_text}")
                    return []
                
                result = await response.json()
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
                    if selected_item["id"] < len(articles):
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
                if len(selected_articles) > target_cap:
                    LOG.warning(f"Code-level cap enforcement: Trimming {len(selected_articles)} to {target_cap}")
                    selected_articles = selected_articles[:target_cap]
                
                return selected_articles
        
    except Exception as e:
        LOG.error(f"Async triage request failed: {str(e)}")
        return []

# ===== CLAUDE TRIAGE FUNCTIONS =====

async def triage_company_articles_claude(articles: List[Dict], ticker: str, company_name: str) -> List[Dict]:
    """Claude-based company triage - parallel to OpenAI triage"""
    if not ANTHROPIC_API_KEY or not articles:
        LOG.warning("No Anthropic API key or no articles to triage")
        return []

    # Prepare items for triage (title + description)
    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        # Add description if available
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(20, len(articles))

    system_prompt = f"""You are a financial analyst selecting the {target_cap} most important articles about {company_name} ({ticker}) from {len(articles)} candidates based ONLY on titles and descriptions.

CRITICAL: Select UP TO {target_cap} articles, fewer if uncertain.

SELECT (choose up to {target_cap}):

TIER 1 - Hard corporate events:
- Financial: "beats," "misses," "earnings," "revenue," "guidance," "margin," "profit," "loss," "EPS," "sales"
- Capital: "buyback," "dividend," "debt," "bond," "offering," "raises," "refinance," "converts"
- M&A: "acquires," "buys," "sells," "divests," "stake," "merger," "spin-off," "joint venture"
- Regulatory: "FDA," "SEC," "DOJ," "FTC," "investigation," "fine," "settlement," "approval," "lawsuit," "probe"
- Operations: "accident," "disaster," "halt," "shutdown," "recall," "expansion," "closure," "layoffs," "strike"
- Products: "approval," "launch," "recall," "trial results," "patent" WITH specific product/drug names
- Contracts: Dollar amounts in title (e.g., "$500M contract," "£2B deal")
- Ratings: "upgrade," "downgrade" WITH analyst firm name (e.g., "BofA upgrades," "Moody's cuts")

TIER 2 - Strategic developments:
- Leadership: CEO, CFO, President, CTO WITH "appoints," "names," "resigns," "retires," "replaces"
- Partnerships: Named partner companies (e.g., "{company_name} partners with [Other Company]")
- Technology: Specific tech/platform names WITH "launches," "announces," "deploys"
- Facilities: Plant/office/branch/store openings, closures WITH locations and capacity/headcount numbers
- Clinical: Trial phases, enrollment milestones, data releases (pharma/biotech)
- Spectrum/Licenses: Acquisitions, renewals WITH specific bands/regions (telecom)
- Geographic: Market entry/exit WITH investment levels or unit counts

TIER 3 - Context (ONLY if quota unfilled):
- Analyst coverage WITH price targets visible in title
- Industry awards, certifications if indicative of competitive position
- Routine announcements WITH material operational details

REJECT COMPLETELY - Never select:
- Generic lists: "Top," "Best," "Should You Buy," "Stocks to Watch," "X Stocks to"
- Roundups: "Sector Update," "Stock Movers," "Trending Stocks," "Biggest Analyst Calls"
- Clickbait: "This Could," "Why," "What Investors Need to Know," "How to"
- Future speculation: "Heading to," "Could reach $X," "Price Prediction 20XX"
- Historical: "If you'd invested," "20 years ago," "Where would $1000 be"
- Market reports: "Market to reach," "CAGR," "Forecast 20XX-20YY"
- Quote pages: "Stock Price | Live Quotes & Charts | [Exchange]"
- Other companies: If title clearly focuses on different company, reject

DISAMBIGUATION - Avoid confusion:
- If title leads with different company name, likely not about {company_name}
- If {company_name} only appears as news source attribution, not subject
- For common words (Oracle, Amazon, Apple), verify context matches your company

SCRAPE PRIORITY (assign integer 1-3):
1 = Tier 1 (financial results, M&A, regulatory, disasters, major contracts)
2 = Tier 2 (leadership, partnerships, product launches, facilities)
3 = Tier 3 (analyst coverage, awards, routine announcements)

Return a JSON array of selected articles. Each must have:
[{{"id": 0, "scrape_priority": 1, "why": "brief reason"}}]

CRITICAL CONSTRAINT: Return UP TO {target_cap} articles. Select fewer if you're uncertain about relevance.

Articles: {json.dumps(items, separators=(',', ':'))}"""

    try:
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        data = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": system_prompt
                }
            ]
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            async with session.post(ANTHROPIC_API_URL, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    LOG.error(f"Claude triage API error {response.status}: {error_text}")
                    return []

                result = await response.json()
                content = result.get("content", [{}])[0].get("text", "")

                if not content:
                    LOG.error("Claude returned no text content for triage")
                    return []

                # Extract JSON from response
                try:
                    # Try direct parse first
                    triage_result = json.loads(content)
                except json.JSONDecodeError:
                    # Try extracting JSON array from text
                    import re
                    match = re.search(r'\[.*\]', content, re.DOTALL)
                    if match:
                        try:
                            triage_result = json.loads(match.group(0))
                        except:
                            LOG.error(f"Claude JSON extraction failed")
                            return []
                    else:
                        LOG.error(f"Claude returned non-JSON content")
                        return []

                # Validate and return selected articles
                selected_articles = []
                for item in triage_result:
                    if isinstance(item, dict) and "id" in item and "scrape_priority" in item:
                        article_id = item["id"]
                        if 0 <= article_id < len(articles):
                            selected_articles.append({
                                "id": article_id,
                                "scrape_priority": item["scrape_priority"],
                                "why": item.get("why", ""),
                                "confidence": 0.8,  # Default confidence for Claude
                                "likely_repeat": False,
                                "repeat_key": ""
                            })

                # Cap at target
                if len(selected_articles) > target_cap:
                    LOG.warning(f"Claude selected {len(selected_articles)}, capping to {target_cap}")
                    selected_articles = selected_articles[:target_cap]

                LOG.info(f"Claude triage company: selected {len(selected_articles)}/{len(articles)} articles")
                return selected_articles

    except Exception as e:
        LOG.error(f"Claude triage request failed: {str(e)}")
        return []

async def triage_industry_articles_claude(articles: List[Dict], ticker: str, company_name: str, sector: str, peers: List[str]) -> List[Dict]:
    """Claude-based industry keyword triage"""
    if not ANTHROPIC_API_KEY or not articles:
        return []

    # Prepare items
    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(5, len(articles))
    peers_display = ', '.join(peers[:5]) if peers else 'None'

    system_prompt = f"""You are a financial analyst selecting the {target_cap} most important INDUSTRY articles from {len(articles)} candidates that have demonstrable impact on {company_name} ({ticker}).

TARGET COMPANY: {company_name} ({ticker})
SECTOR: {sector}
KNOWN PEERS: {peers_display}

CRITICAL: Select UP TO {target_cap} articles, fewer if uncertain. ONLY select articles where {ticker}-specific impact is clear and material.

SELECT (choose up to {target_cap}):

TIER 1 - Direct operational impact on {ticker}:
- Regulatory/Policy affecting {ticker}'s operations: New laws, rules, tariffs, bans WITH specific rates/dates/costs that apply to {ticker}
- Input costs for {ticker}: Commodity/component prices, labor rates WITH figures affecting {ticker}'s cost structure
- Supply disruptions affecting {ticker}: Production stoppages, capacity constraints WITH volume impacts on {ticker}'s supply chain
- Demand drivers for {ticker}: End-market trends, customer budget changes WITH implications for {ticker}'s revenue
- Compliance requirements for {ticker}: New standards, certifications WITH deadlines/costs {ticker} must meet
- Trade impacts on {ticker}: Tariffs, quotas, sanctions WITH specific effect on {ticker}'s markets or costs

TIER 2 - Competitive positioning impact on {ticker}:
- Capacity changes affecting {ticker}: Major additions/closures WITH impact on {ticker}'s market share or pricing power
- Technology adoption revealing {ticker}'s position: Industry shifts WITH data showing {ticker} ahead/behind
- Industry consolidation affecting {ticker}: M&A changing competitive landscape for {ticker}
- Standards evolution affecting {ticker}: Protocol/spec changes WITH {ticker}'s compliance position or advantage
- Infrastructure investment affecting {ticker}: Government/private buildouts WITH impact on {ticker}'s business
- Sector financial trends revealing {ticker}'s performance: Industry margins, growth rates WITH {ticker} comparison context

TIER 3 - Material sector context for {ticker} (ONLY if quota unfilled):
- Economic indicators directly affecting {ticker}'s end markets WITH quantified demand/pricing impact
- Government initiatives WITH budgets that create opportunities/risks for {ticker}
- Research findings WITH specific implications for {ticker}'s technology or market position

REJECT COMPLETELY - Never select:
- Market research reports: "Market to reach," "CAGR," "Forecast 20XX-20YY," "TAM" (unless {ticker} explicitly discussed)
- Generic trends: "Top Trends," "Future of," "Outlook" (no {ticker} connection)
- Pure competitor news: Single-company earnings, appointments WITHOUT {ticker} competitive context
- Listicles: "X Best," "How to," "Why You Should" (not actionable for {ticker})
- Opinion pieces: "Analysis," "Commentary" WITHOUT hard data affecting {ticker}
- Routine announcements: Small company moves with no {ticker} relevance

VALIDATION - Article must pass this test:
✓ Can you explain in ONE sentence how this affects {ticker}'s operations, costs, revenues, or competitive position?
✗ If NO clear {ticker} connection → REJECT

Examples of GOOD selections:
✓ "EU mandates 30% recycled content by 2026" → {ticker} must adjust supply chain, compliance costs quantified
✓ "Semiconductor equipment lead times extend to 18 months" → {ticker} capacity expansion delayed, competitive impact
✓ "Copper prices hit $12,000/ton" → {ticker} input costs rise X%, margin pressure
✓ "FDA approves competitor drug" → {ticker} market share threatened in $XXB segment

Examples of BAD selections (REJECT these):
✗ "Sector employment rises 5%" → No specific {ticker} operational impact
✗ "Industry to grow 12% CAGR through 2030" → Generic forecast, no {ticker} positioning
✗ "Small biotech raises $50M Series B" → Irrelevant to {ticker}
✗ "Expert predicts AI transformation" → Opinion, no actionable {ticker} impact

SCRAPE PRIORITY (assign integer 1-3):
1 = Tier 1 (direct operational impact on {ticker} WITH numbers)
2 = Tier 2 (competitive positioning impact on {ticker} WITH data)
3 = Tier 3 (material sector context for {ticker})

Return JSON array: [{{"id": 0, "scrape_priority": 1, "why": "Explain {ticker} impact in 10 words"}}]

CRITICAL CONSTRAINT: Return UP TO {target_cap} articles. Every selection must have clear {ticker} relevance. If uncertain, DO NOT SELECT.

Articles: {json.dumps(items, separators=(',', ':'))}"""

    try:
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        data = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": system_prompt}]
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            async with session.post(ANTHROPIC_API_URL, headers=headers, json=data) as response:
                if response.status != 200:
                    LOG.error(f"Claude industry triage error {response.status}")
                    return []

                result = await response.json()
                content = result.get("content", [{}])[0].get("text", "")

                try:
                    triage_result = json.loads(content)
                except:
                    import re
                    match = re.search(r'\[.*\]', content, re.DOTALL)
                    if match:
                        triage_result = json.loads(match.group(0))
                    else:
                        return []

                selected_articles = []
                for item in triage_result:
                    if isinstance(item, dict) and "id" in item:
                        if 0 <= item["id"] < len(articles):
                            selected_articles.append({
                                "id": item["id"],
                                "scrape_priority": item.get("scrape_priority", 2),
                                "why": item.get("why", ""),
                                "confidence": 0.8,
                                "likely_repeat": False,
                                "repeat_key": ""
                            })

                if len(selected_articles) > target_cap:
                    selected_articles = selected_articles[:target_cap]

                return selected_articles

    except Exception as e:
        LOG.error(f"Claude industry triage failed: {str(e)}")
        return []

async def triage_competitor_articles_claude(articles: List[Dict], ticker: str, competitor_name: str, competitor_ticker: str) -> List[Dict]:
    """Claude-based competitor triage"""
    if not ANTHROPIC_API_KEY or not articles:
        return []

    # Prepare items
    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(5, len(articles))

    system_prompt = f"""You are a financial analyst for {ticker} investors selecting the {target_cap} most important articles about competitor {competitor_name} ({competitor_ticker}) from {len(articles)} candidates based ONLY on titles and descriptions.

CRITICAL: Select UP TO {target_cap} articles, fewer if uncertain.

SELECT (choose up to {target_cap}):

TIER 1 - Hard corporate events:
- Financial: "beats," "misses," "earnings," "revenue," "guidance," "margin," "profit," "loss," "EPS," "sales"
- Capital: "buyback," "dividend," "debt," "bond," "offering," "raises," "refinance," "converts"
- M&A: "acquires," "buys," "sells," "divests," "stake," "merger," "spin-off," "joint venture"
- Regulatory: "FDA," "SEC," "DOJ," "FTC," "investigation," "fine," "settlement," "approval," "lawsuit," "probe"
- Operations: "accident," "disaster," "halt," "shutdown," "recall," "expansion," "closure," "layoffs," "strike"
- Products: "approval," "launch," "recall," "trial results," "patent" WITH specific product/drug names
- Contracts: Dollar amounts in title (e.g., "$500M contract," "£2B deal")
- Ratings: "upgrade," "downgrade" WITH analyst firm name (e.g., "BofA upgrades," "Moody's cuts")

TIER 2 - Strategic developments:
- Leadership: CEO, CFO, President, CTO WITH "appoints," "names," "resigns," "retires," "replaces"
- Partnerships: Named partner companies (e.g., "{competitor_name} partners with [Other Company]")
- Technology: Specific tech/platform names WITH "launches," "announces," "deploys"
- Facilities: Plant/office/branch/store openings, closures WITH locations and capacity/headcount numbers
- Clinical: Trial phases, enrollment milestones, data releases (pharma/biotech)
- Spectrum/Licenses: Acquisitions, renewals WITH specific bands/regions (telecom)
- Geographic: Market entry/exit WITH investment levels or unit counts

TIER 3 - Context (ONLY if quota unfilled):
- Analyst coverage WITH price targets visible in title
- Industry awards, certifications if indicative of competitive position
- Routine announcements WITH material operational details

REJECT COMPLETELY - Never select:
- Generic lists: "Top," "Best," "Should You Buy," "Stocks to Watch," "X Stocks to"
- Roundups: "Sector Update," "Stock Movers," "Trending Stocks," "Biggest Analyst Calls"
- Clickbait: "This Could," "Why," "What Investors Need to Know," "How to"
- Future speculation: "Heading to," "Could reach $X," "Price Prediction 20XX"
- Historical: "If you'd invested," "20 years ago," "Where would $1000 be"
- Market reports: "Market to reach," "CAGR," "Forecast 20XX-20YY"
- Quote pages: "Stock Price | Live Quotes & Charts | [Exchange]"
- Other companies: If title clearly focuses on different company, reject

DISAMBIGUATION - Avoid confusion:
- If title leads with different company name, likely not about {competitor_name}
- If {competitor_name} only appears as news source attribution, not subject
- For common words (Oracle, Amazon, Apple), verify context matches your company

SCRAPE PRIORITY (assign integer 1-3):
1 = Tier 1 (financial results, M&A, regulatory, disasters, major contracts)
2 = Tier 2 (leadership, partnerships, product launches, facilities)
3 = Tier 3 (analyst coverage, awards, routine announcements)

Return JSON array: [{{"id": 0, "scrape_priority": 1, "why": "reason text"}}]

CRITICAL CONSTRAINT: Return UP TO {target_cap} articles. Select fewer if you're uncertain about relevance.

Articles: {json.dumps(items, separators=(',', ':'))}"""

    try:
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        data = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": system_prompt}]
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            async with session.post(ANTHROPIC_API_URL, headers=headers, json=data) as response:
                if response.status != 200:
                    LOG.error(f"Claude competitor triage error {response.status}")
                    return []

                result = await response.json()
                content = result.get("content", [{}])[0].get("text", "")

                try:
                    triage_result = json.loads(content)
                except:
                    import re
                    match = re.search(r'\[.*\]', content, re.DOTALL)
                    if match:
                        triage_result = json.loads(match.group(0))
                    else:
                        return []

                selected_articles = []
                for item in triage_result:
                    if isinstance(item, dict) and "id" in item:
                        if 0 <= item["id"] < len(articles):
                            selected_articles.append({
                                "id": item["id"],
                                "scrape_priority": item.get("scrape_priority", 2),
                                "why": item.get("why", ""),
                                "confidence": 0.8,
                                "likely_repeat": False,
                                "repeat_key": ""
                            })

                if len(selected_articles) > target_cap:
                    selected_articles = selected_articles[:target_cap]

                return selected_articles

    except Exception as e:
        LOG.error(f"Claude competitor triage failed: {str(e)}")
        return []

# ===== END CLAUDE TRIAGE FUNCTIONS =====

# ===== DUAL SCORING LOGIC (OpenAI + Claude) =====

async def merge_triage_scores(
    openai_results: List[Dict],
    claude_results: List[Dict],
    articles: List[Dict],
    target_cap: int,
    category_type: str,
    category_key: str
) -> List[Dict]:
    """
    Merge OpenAI and Claude triage results by combined score.
    Returns top N articles by combined score (openai_score + claude_score).
    Handles fallback if one API fails (use single API's selections).
    """
    # Build URL-based lookup for matching articles across APIs
    url_scores = {}  # url -> {"openai": score, "claude": score, "article": article_obj}

    # Process OpenAI results
    for result in openai_results:
        article_id = result["id"]
        if article_id < len(articles):
            article = articles[article_id]
            url = article.get("url", "")
            if url:
                if url not in url_scores:
                    url_scores[url] = {"openai": 0, "claude": 0, "article": article, "id": article_id}
                # Score: 1=high, 2=medium, 3=low -> convert to reverse (high=3, low=1)
                priority = result.get("scrape_priority", 2)
                url_scores[url]["openai"] = 4 - priority  # 3, 2, 1
                url_scores[url]["why_openai"] = result.get("why", "")

    # Process Claude results
    for result in claude_results:
        article_id = result["id"]
        if article_id < len(articles):
            article = articles[article_id]
            url = article.get("url", "")
            if url:
                if url not in url_scores:
                    url_scores[url] = {"openai": 0, "claude": 0, "article": article, "id": article_id}
                priority = result.get("scrape_priority", 2)
                url_scores[url]["claude"] = 4 - priority
                url_scores[url]["why_claude"] = result.get("why", "")

    # Track stats for logging
    total_unique_before_filter = len(url_scores)

    # Filter out problematic domains BEFORE ranking
    filtered_scores = {}
    blocked_count = 0
    for url, data in url_scores.items():
        article = data["article"]
        domain = article.get("domain", "")
        if domain in PROBLEMATIC_SCRAPE_DOMAINS:
            blocked_count += 1
            LOG.debug(f"Blocked {domain} from triage selection (problematic domain)")
            continue
        filtered_scores[url] = data

    # Calculate combined scores and sort
    scored_articles = []
    for url, data in filtered_scores.items():
        combined_score = data["openai"] + data["claude"]
        if combined_score > 0:  # At least one API selected it
            scored_articles.append({
                "url": url,
                "article": data["article"],
                "id": data["id"],
                "openai_score": data["openai"],
                "claude_score": data["claude"],
                "combined_score": combined_score,
                "why_openai": data.get("why_openai", ""),
                "why_claude": data.get("why_claude", "")
            })

    # Sort by combined score (descending), then by timestamp (recent first)
    scored_articles.sort(key=lambda x: (
        -x["combined_score"],
        -x["article"].get("published_at", datetime.min).timestamp() if x["article"].get("published_at") else 0
    ))

    # Take top N
    top_articles = scored_articles[:target_cap]

    # Enhanced logging with domain filter transparency
    if blocked_count > 0:
        LOG.info(f"  Dual scoring {category_type}/{category_key}: {len(openai_results)} OpenAI + {len(claude_results)} Claude = {total_unique_before_filter} unique ({blocked_count} blocked by domain filter) → {len(scored_articles)} remaining → top {len(top_articles)}")
    else:
        LOG.info(f"  Dual scoring {category_type}/{category_key}: {len(openai_results)} OpenAI + {len(claude_results)} Claude = {len(scored_articles)} unique → top {len(top_articles)}")

    # Return in format expected by downstream code
    result = []
    for item in top_articles:
        result.append({
            "id": item["id"],
            "scrape_priority": 1 if item["combined_score"] >= 5 else (2 if item["combined_score"] >= 3 else 3),
            "why": f"OpenAI: {item['why_openai'][:50]}... Claude: {item['why_claude'][:50]}..." if item['why_openai'] and item['why_claude'] else (item['why_openai'] or item['why_claude']),
            "confidence": 0.9 if (item["openai_score"] > 0 and item["claude_score"] > 0) else 0.7,
            "likely_repeat": False,
            "repeat_key": "",
            "openai_score": item["openai_score"],
            "claude_score": item["claude_score"],
            "combined_score": item["combined_score"]
        })

    return result

async def perform_ai_triage_with_dual_scoring_async(
    articles_by_category: Dict[str, List[Dict]],
    ticker: str,
    triage_batch_size: int = 5  # Updated default to 5
) -> Dict[str, List[Dict]]:
    """
    Dual scoring triage: OpenAI + Claude run concurrently, results merged by combined score.
    Replaces perform_ai_triage_with_batching_async with dual API support.
    """
    selected_results = {"company": [], "industry": [], "competitor": []}

    if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
        LOG.warning("No AI API keys configured - skipping triage")
        return selected_results

    # Ensure async semaphores are initialized
    init_async_semaphores()

    # Get ticker metadata
    config = get_ticker_config(ticker)
    company_name = config.get("name", ticker) if config else ticker
    sector = config.get("sector", "") if config else ""
    industry_keywords = [config.get(f"industry_keyword_{i}") for i in range(1, 4) if config.get(f"industry_keyword_{i}")]
    competitors = [(config.get(f"competitor_{i}_name"), config.get(f"competitor_{i}_ticker")) for i in range(1, 4) if config.get(f"competitor_{i}_name")]

    # Build peers list for industry triage
    peers = []
    for comp_name, comp_ticker in competitors:
        if comp_name and comp_ticker:
            peers.append(f"{comp_name} ({comp_ticker})")
        elif comp_name:
            peers.append(comp_name)

    LOG.info(f"=== DUAL SCORING TRIAGE (OpenAI + Claude): batch_size={triage_batch_size} ===")

    # Collect ALL dual-triage operations
    all_triage_operations = []

    # Company operations
    company_articles = articles_by_category.get("company", [])
    if company_articles:
        all_triage_operations.append({
            "type": "company",
            "key": "company",
            "articles": company_articles,
            "target_cap": min(20, len(company_articles)),
            "openai_func": triage_company_articles_full,
            "openai_args": (company_articles, ticker, company_name, {}, {}),
            "claude_func": triage_company_articles_claude,
            "claude_args": (company_articles, ticker, company_name)
        })

    # Industry operations (one per keyword)
    industry_articles = articles_by_category.get("industry", [])
    if industry_articles:
        industry_by_keyword = {}
        for idx, article in enumerate(industry_articles):
            keyword = article.get("search_keyword", "unknown")
            if keyword not in industry_by_keyword:
                industry_by_keyword[keyword] = []
            industry_by_keyword[keyword].append({"article": article, "original_idx": idx})

        for keyword, keyword_articles in industry_by_keyword.items():
            triage_articles = [item["article"] for item in keyword_articles]
            all_triage_operations.append({
                "type": "industry",
                "key": keyword,
                "articles": triage_articles,
                "target_cap": min(5, len(triage_articles)),
                "openai_func": triage_industry_articles_full,
                "openai_args": (triage_articles, ticker, company_name, sector, peers),
                "claude_func": triage_industry_articles_claude,
                "claude_args": (triage_articles, ticker, company_name, sector, peers),
                "index_mapping": keyword_articles
            })

    # Competitor operations (one per competitor)
    competitor_articles = articles_by_category.get("competitor", [])
    if competitor_articles:
        competitor_by_entity = {}
        for idx, article in enumerate(competitor_articles):
            entity_key = article.get("competitor_ticker") or article.get("search_keyword", "unknown")
            if entity_key not in competitor_by_entity:
                competitor_by_entity[entity_key] = []
            competitor_by_entity[entity_key].append({"article": article, "original_idx": idx})

        for entity_key, entity_articles in competitor_by_entity.items():
            triage_articles = [item["article"] for item in entity_articles]
            # Get competitor name and ticker from article metadata
            competitor_ticker = entity_articles[0]["article"].get("competitor_ticker", entity_key)
            competitor_name = entity_articles[0]["article"].get("search_keyword", entity_key)
            all_triage_operations.append({
                "type": "competitor",
                "key": entity_key,
                "articles": triage_articles,
                "target_cap": min(5, len(triage_articles)),
                "openai_func": triage_competitor_articles_full,
                "openai_args": (triage_articles, ticker, competitor_name, competitor_ticker),
                "claude_func": triage_competitor_articles_claude,
                "claude_args": (triage_articles, ticker, competitor_name, competitor_ticker),
                "index_mapping": entity_articles
            })

    total_operations = len(all_triage_operations)
    LOG.info(f"Total dual-triage operations: {total_operations}")

    if total_operations == 0:
        return selected_results

    # Process operations in batches
    for batch_start in range(0, total_operations, triage_batch_size):
        batch_end = min(batch_start + triage_batch_size, total_operations)
        batch = all_triage_operations[batch_start:batch_end]
        batch_num = (batch_start // triage_batch_size) + 1
        total_batches = (total_operations + triage_batch_size - 1) // triage_batch_size

        LOG.info(f"BATCH {batch_num}/{total_batches}: Processing {len(batch)} operations (OpenAI + Claude concurrent):")
        for op in batch:
            LOG.info(f"  - {op['type']}: {op['key']} ({len(op['articles'])} articles)")

        # Create dual tasks for each operation (OpenAI + Claude in parallel)
        batch_tasks = []
        for op in batch:
            async def run_dual_triage(operation):
                # Run OpenAI and Claude concurrently with semaphore control
                openai_task = None
                claude_task = None

                if OPENAI_API_KEY:
                    async def run_openai():
                        async with async_openai_sem:
                            return await operation["openai_func"](*operation["openai_args"])
                    openai_task = run_openai()

                if ANTHROPIC_API_KEY:
                    async def run_claude():
                        async with async_claude_sem:
                            return await operation["claude_func"](*operation["claude_args"])
                    claude_task = run_claude()

                # Wait for both to complete
                tasks_to_run = [t for t in [openai_task, claude_task] if t]
                results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

                # Extract results
                openai_result = results[0] if OPENAI_API_KEY and len(results) > 0 else []
                claude_result = results[1] if ANTHROPIC_API_KEY and len(results) > 1 else (results[0] if not OPENAI_API_KEY and len(results) > 0 else [])

                # Handle exceptions
                if isinstance(openai_result, Exception):
                    LOG.error(f"OpenAI triage failed for {operation['type']}/{operation['key']}: {openai_result}")
                    openai_result = []
                if isinstance(claude_result, Exception):
                    LOG.error(f"Claude triage failed for {operation['type']}/{operation['key']}: {claude_result}")
                    claude_result = []

                # Merge scores
                merged = await merge_triage_scores(
                    openai_result,
                    claude_result,
                    operation["articles"],
                    operation["target_cap"],
                    operation["type"],
                    operation["key"]
                )

                return merged

            task = run_dual_triage(op)
            batch_tasks.append((op, task))

        # Execute batch concurrently
        batch_results = await asyncio.gather(*[task for _, task in batch_tasks], return_exceptions=True)

        # Process results
        for i, result in enumerate(batch_results):
            op = batch_tasks[i][0]

            if isinstance(result, Exception):
                LOG.error(f"Dual triage failed for {op['type']}/{op['key']}: {result}")
                continue

            # Map results back to original indices and add to selected_results
            if op["type"] == "company":
                selected_results["company"].extend(result)
                LOG.info(f"  ✓ Company: selected {len(result)} articles")

            elif op["type"] == "industry":
                # Map back to original indices
                for selected_item in result:
                    original_idx = op["index_mapping"][selected_item["id"]]["original_idx"]
                    selected_item["id"] = original_idx
                selected_results["industry"].extend(result)
                LOG.info(f"  ✓ Industry '{op['key']}': selected {len(result)} articles")

            elif op["type"] == "competitor":
                # Map back to original indices
                for selected_item in result:
                    original_idx = op["index_mapping"][selected_item["id"]]["original_idx"]
                    selected_item["id"] = original_idx
                selected_results["competitor"].extend(result)
                LOG.info(f"  ✓ Competitor '{op['key']}': selected {len(result)} articles")

        LOG.info(f"BATCH {batch_num} COMPLETE")

    LOG.info(f"DUAL SCORING TRIAGE COMPLETE:")
    LOG.info(f"  Company: {len(selected_results['company'])} selected")
    LOG.info(f"  Industry: {len(selected_results['industry'])} selected")
    LOG.info(f"  Competitor: {len(selected_results['competitor'])} selected")

    return selected_results

# ===== END DUAL SCORING LOGIC =====

async def perform_ai_triage_with_batching_async(
    articles_by_category: Dict[str, List[Dict]],
    ticker: str,
    triage_batch_size: int = 2
) -> Dict[str, List[Dict]]:
    """
    Perform AI triage with true cross-category async batching
    """
    if not OPENAI_API_KEY:
        LOG.warning("OpenAI API key not configured - skipping triage")
        return {"company": [], "industry": [], "competitor": []}
    
    # Ensure async semaphores are initialized
    init_async_semaphores()
    
    selected_results = {"company": [], "industry": [], "competitor": []}

    # Get ticker metadata for enhanced prompts
    config = get_ticker_config(ticker)
    company_name = config.get("name", ticker) if config else ticker
    sector = config.get("sector", "") if config else ""

    # Build peers list for industry triage
    competitors = [(config.get(f"competitor_{i}_name"), config.get(f"competitor_{i}_ticker")) for i in range(1, 4) if config.get(f"competitor_{i}_name")] if config else []
    peers = []
    for comp_name, comp_ticker in competitors:
        if comp_name and comp_ticker:
            peers.append(f"{comp_name} ({comp_ticker})")
        elif comp_name:
            peers.append(comp_name)

    LOG.info(f"=== TRUE CROSS-CATEGORY ASYNC TRIAGE: batch_size={triage_batch_size} ===")
    
    # Collect ALL triage operations into a single list
    all_triage_operations = []
    
    # Add company operation
    company_articles = articles_by_category.get("company", [])
    if company_articles:
        all_triage_operations.append({
            "type": "company",
            "key": "company",
            "articles": company_articles,
            "task_func": triage_company_articles_full,
            "args": (company_articles, ticker, company_name, {}, {})
        })
    
    # Add industry operations (one per keyword)
    industry_articles = articles_by_category.get("industry", [])
    if industry_articles:
        # Group by search_keyword
        industry_by_keyword = {}
        for idx, article in enumerate(industry_articles):
            keyword = article.get("search_keyword", "unknown")
            if keyword not in industry_by_keyword:
                industry_by_keyword[keyword] = []
            industry_by_keyword[keyword].append({"article": article, "original_idx": idx})
        
        for keyword, keyword_articles in industry_by_keyword.items():
            triage_articles = [item["article"] for item in keyword_articles]
            all_triage_operations.append({
                "type": "industry",
                "key": keyword,
                "articles": triage_articles,
                "task_func": triage_industry_articles_full,
                "args": (triage_articles, ticker, company_name, sector, peers),
                "index_mapping": keyword_articles  # For mapping back to original indices
            })
    
    # Add competitor operations (one per competitor)
    competitor_articles = articles_by_category.get("competitor", [])
    if competitor_articles:
        # Group by competitor_ticker (primary) or search_keyword (fallback)
        competitor_by_entity = {}
        for idx, article in enumerate(competitor_articles):
            entity_key = article.get("competitor_ticker") or article.get("search_keyword", "unknown")
            if entity_key not in competitor_by_entity:
                competitor_by_entity[entity_key] = []
            competitor_by_entity[entity_key].append({"article": article, "original_idx": idx})
        
        for entity_key, entity_articles in competitor_by_entity.items():
            triage_articles = [item["article"] for item in entity_articles]
            all_triage_operations.append({
                "type": "competitor",
                "key": entity_key,
                "articles": triage_articles,
                "task_func": triage_competitor_articles_full,
                "args": (triage_articles, ticker, [], {}),
                "index_mapping": entity_articles  # For mapping back to original indices
            })
    
    total_operations = len(all_triage_operations)
    LOG.info(f"Total triage operations to process: {total_operations}")
    
    if total_operations == 0:
        return selected_results
    
    # Process operations in cross-category batches
    for batch_start in range(0, total_operations, triage_batch_size):
        batch_end = min(batch_start + triage_batch_size, total_operations)
        batch = all_triage_operations[batch_start:batch_end]
        batch_num = (batch_start // triage_batch_size) + 1
        total_batches = (total_operations + triage_batch_size - 1) // triage_batch_size
        
        LOG.info(f"BATCH {batch_num}/{total_batches}: Processing {len(batch)} operations concurrently:")
        for op in batch:
            LOG.info(f"  - {op['type']}: {op['key']} ({len(op['articles'])} articles)")
        
        # Create tasks for this batch with semaphore control
        batch_tasks = []
        for op in batch:
            async def run_with_semaphore(operation):
                async with async_triage_sem:
                    return await operation["task_func"](*operation["args"])
            
            task = run_with_semaphore(op)
            batch_tasks.append((op, task))
        
        # Execute batch concurrently
        batch_results = await asyncio.gather(*[task for _, task in batch_tasks], return_exceptions=True)
        
        # Process results
        for i, result in enumerate(batch_results):
            op = batch_tasks[i][0]
            
            if isinstance(result, Exception):
                LOG.error(f"Triage failed for {op['type']}/{op['key']}: {result}")
                continue
            
            # Map results back to original indices and add to selected_results
            if op["type"] == "company":
                selected_results["company"].extend(result)
                LOG.info(f"  ✓ Company: selected {len(result)} articles")
            
            elif op["type"] == "industry":
                # Map back to original indices
                for selected_item in result:
                    original_idx = op["index_mapping"][selected_item["id"]]["original_idx"]
                    selected_item["id"] = original_idx
                selected_results["industry"].extend(result)
                LOG.info(f"  ✓ Industry '{op['key']}': selected {len(result)} articles")
            
            elif op["type"] == "competitor":
                # Map back to original indices
                for selected_item in result:
                    original_idx = op["index_mapping"][selected_item["id"]]["original_idx"]
                    selected_item["id"] = original_idx
                selected_results["competitor"].extend(result)
                LOG.info(f"  ✓ Competitor '{op['key']}': selected {len(result)} articles")
        
        LOG.info(f"BATCH {batch_num} COMPLETE")
    
    LOG.info(f"CROSS-CATEGORY TRIAGE COMPLETE:")
    LOG.info(f"  Company: {len(selected_results['company'])} selected")
    LOG.info(f"  Industry: {len(selected_results['industry'])} selected") 
    LOG.info(f"  Competitor: {len(selected_results['competitor'])} selected")
    
    return selected_results

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
    Priority: ticker_reference.company_name -> competitor_metadata -> search_keyword -> fallback

    ALWAYS uses competitor_ticker to lookup company_name in ticker_reference table.
    Never relies on search_keyword for display (search_keyword is feed query parameter, not display name).
    """

    # Input validation
    if competitor_ticker:
        competitor_ticker = normalize_ticker_format(competitor_ticker)
        if not validate_ticker_format(competitor_ticker):
            LOG.warning(f"Invalid competitor ticker format: {competitor_ticker}")
            competitor_ticker = None

    if search_keyword:
        search_keyword = search_keyword.strip()
        if len(search_keyword) > 100 or not search_keyword:
            LOG.warning(f"Invalid search keyword: {search_keyword}")
            search_keyword = None

    # STEP 1: Try ticker_reference table first (most comprehensive - 6,178 tickers)
    if competitor_ticker:
        try:
            with db() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT company_name FROM ticker_reference
                    WHERE ticker = %s AND active = TRUE
                    LIMIT 1
                """, (competitor_ticker,))
                result = cur.fetchone()

                if result and result["company_name"]:
                    return result["company_name"]
        except Exception as e:
            LOG.debug(f"ticker_reference lookup failed for competitor {competitor_ticker}: {e}")

    # STEP 2: Try competitor_metadata table as backup
    if competitor_ticker:
        try:
            with db() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT competitor_name FROM competitor_metadata
                    WHERE competitor_ticker = %s AND active = TRUE
                    LIMIT 1
                """, (competitor_ticker,))
                result = cur.fetchone()

                if result and result["competitor_name"]:
                    return result["competitor_name"]
        except Exception as e:
            LOG.debug(f"competitor_metadata lookup failed for competitor {competitor_ticker}: {e}")

    # STEP 3: Fallback to search_keyword ONLY if it looks like a company name (not ticker)
    if search_keyword and not search_keyword.isupper():  # Likely a company name, not ticker
        return search_keyword

    # STEP 4: Final fallback - use ticker if that's all we have
    return competitor_ticker or search_keyword or "Unknown Competitor"

def determine_article_category_for_ticker(article_row: dict, viewing_ticker: str) -> str:
    """
    Determine the correct article category based on the viewing ticker.
    Same article can be 'company' for one ticker and 'competitor' for another.
    """
    try:
        # Get ticker metadata to determine relationships
        ticker_config = get_ticker_reference(viewing_ticker)
        if not ticker_config:
            # Fallback to stored category if no metadata
            LOG.debug(f"CATEGORIZATION: No metadata for {viewing_ticker}, using original category")
            return article_row.get("category", "company")

        # Get company name for the viewing ticker
        viewing_company_name = ticker_config.get("company_name", viewing_ticker)

        # Get article's search keyword and competitor ticker
        search_keyword = article_row.get("search_keyword", "")
        competitor_ticker = article_row.get("competitor_ticker", "")
        original_category = article_row.get("category", "company")
        article_title = article_row.get("title", "")[:50]

        LOG.debug(f"CATEGORIZATION: {viewing_ticker} | Company: '{viewing_company_name}' | Article: '{article_title}' | Search: '{search_keyword}' | Original: '{original_category}'")

        # If this article mentions the viewing ticker's company, it's company news
        if search_keyword and viewing_company_name:
            # Check if search keyword matches company name (case insensitive)
            if viewing_company_name.lower() in search_keyword.lower() or \
               search_keyword.lower() in viewing_company_name.lower():
                LOG.debug(f"CATEGORIZATION: {viewing_ticker} | MATCH FOUND - Converting '{original_category}' → 'company' for '{search_keyword}'")
                return "company"

        # If the competitor_ticker matches viewing ticker, this is company news
        if competitor_ticker == viewing_ticker:
            LOG.debug(f"CATEGORIZATION: {viewing_ticker} | COMPETITOR TICKER MATCH - Converting '{original_category}' → 'company'")
            return "company"

        # If this is from a competitor feed but about viewing ticker's company, it's company news
        if original_category == "competitor":
            # Check if this competitor article is actually about the viewing ticker's company
            competitors_list = ticker_config.get("competitors", [])
            article_company_mentioned = False

            for comp in competitors_list:
                if isinstance(comp, dict):
                    comp_name = comp.get("name", "")
                    comp_ticker = comp.get("ticker", "")
                    # If article mentions a known competitor, keep it as competitor
                    if comp_name and (comp_name.lower() in search_keyword.lower()):
                        article_company_mentioned = True
                        break
                    if comp_ticker and comp_ticker == competitor_ticker:
                        article_company_mentioned = True
                        break

            # If competitor article doesn't mention known competitors, might be about viewing company
            if not article_company_mentioned and search_keyword:
                if viewing_company_name.lower() in search_keyword.lower():
                    LOG.debug(f"CATEGORIZATION: {viewing_ticker} | COMPETITOR ARTICLE ABOUT COMPANY - Converting 'competitor' → 'company'")
                    return "company"

        # Default: use original category from database
        LOG.debug(f"CATEGORIZATION: {viewing_ticker} | NO MATCH - Keeping original category '{original_category}'")
        return original_category

    except Exception as e:
        LOG.warning(f"Error determining category for {viewing_ticker}: {e}")
        return article_row.get("category", "company")

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
        """Handle Yahoo Finance URL resolution - keep Yahoo URLs when resolution fails"""
        original_source = extract_yahoo_finance_source_optimized(url)
        if original_source:
            domain = normalize_domain(urlparse(original_source).netloc.lower())
            if not self._is_spam_domain(domain):
                LOG.info(f"YAHOO SUCCESS: Resolved {url} -> {original_source}")
                return original_source, domain, url
            else:
                LOG.info(f"SPAM REJECTED: Yahoo resolution found spam domain {domain}")
                return None, None, None
        
        # UPDATED: Keep Yahoo Finance URLs when resolution fails (they're easy to scrape)
        yahoo_domain = normalize_domain(urlparse(url).netloc.lower())
        LOG.info(f"YAHOO RESOLUTION FAILED: Keeping Yahoo URL for direct scraping: {url}")
        return url, yahoo_domain, None
    
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
        """Get formal name from AI with improved prompt"""
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

            prompt = f'''Extract the formal publication name for "{domain}".

Requirements:
- Use proper capitalization and spacing
- Return company/publication name only (no domain extension)
- Format as it would appear in a citation

Examples:
- "reuters.com" → "Reuters"
- "prnewswire.co.uk" → "PR Newswire"
- "theglobeandmail.com" → "The Globe and Mail"
- "businessmodelanalyst.com" → "Business Model Analyst"
- "arabamericannews.com" → "Arab American News"

Domain: {domain}
Response:'''

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
    def create_feeds_for_ticker_enhanced(ticker: str, metadata: Dict) -> List[Dict]:
        """Enhanced feed creation with proper international ticker support"""
        feeds = []
        company_name = metadata.get("company_name", ticker)
        
        # CRITICAL: Clear any global state that might affect this ticker's processing
        import gc
        gc.collect()

        # CRITICAL DEBUG: Log exactly what data this ticker is receiving
        LOG.info(f"FEED GENERATION DEBUG for {ticker}:")
        LOG.info(f"  Company name: {metadata.get('company_name', 'MISSING')}")
        LOG.info(f"  Competitors: {metadata.get('competitors', [])}")
        LOG.info(f"  Metadata keys: {list(metadata.keys())}")

        LOG.info(f"CREATING FEEDS for {ticker} ({company_name}):")
        
        # Check existing feed counts by unique competitor
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT tf.category, COUNT(*) as count
                FROM feeds f
                JOIN ticker_feeds tf ON f.id = tf.feed_id
                WHERE tf.ticker = %s AND f.active = TRUE AND tf.active = TRUE
                GROUP BY tf.category
            """, (ticker,))
            
            existing_data = {row["category"]: row for row in cur.fetchall()}
            
            # Extract counts
            existing_company_count = existing_data.get('company', {}).get('count', 0)
            existing_industry_count = existing_data.get('industry', {}).get('count', 0)
            
            # Count unique competitors separately
            cur.execute("""
                SELECT COUNT(DISTINCT f.competitor_ticker) as unique_competitors
                FROM feeds f
                JOIN ticker_feeds tf ON f.id = tf.feed_id
                WHERE tf.ticker = %s AND tf.category = 'competitor' AND f.active = TRUE AND tf.active = TRUE
                AND f.competitor_ticker IS NOT NULL
            """, (ticker,))
            result = cur.fetchone()
            existing_competitor_entities = result["unique_competitors"] if result else 0
            
            LOG.info(f"  EXISTING FEEDS: Company={existing_company_count}, Industry={existing_industry_count}, Competitors={existing_competitor_entities} unique entities")
        
        # Company feeds - always ensure we have the core 2
        if existing_company_count < 2:
            # CRITICAL DEBUG: Log exactly what company data is being used
            LOG.info(f"[{ticker}] CREATING COMPANY FEEDS: company_name='{company_name}', ticker='{ticker}'")

            # CRITICAL DEBUG: Log exactly what variables are being used for company feeds
            LOG.info(f"[COMPANY_FEED_DEBUG] {ticker} creating company feeds with:")
            LOG.info(f"[COMPANY_FEED_DEBUG]   company_name='{company_name}'")
            LOG.info(f"[COMPANY_FEED_DEBUG]   ticker='{ticker}'")

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
        
        # Industry feeds - MAX 3 TOTAL
        if existing_industry_count < 3:
            available_slots = 3 - existing_industry_count
            industry_keywords = metadata.get("industry_keywords", [])[:available_slots]
            
            for keyword in industry_keywords:
                feed = {
                    "url": f"https://news.google.com/rss/search?q=\"{requests.utils.quote(keyword)}\"+when:7d&hl=en-US&gl=US&ceid=US:en",
                    "name": f"Industry: {keyword}",
                    "category": "industry",
                    "search_keyword": keyword
                }
                feeds.append(feed)
        
        # Enhanced competitor feeds with proper international ticker validation
        if existing_competitor_entities < 3:
            available_competitor_slots = 3 - existing_competitor_entities
            competitors = metadata.get("competitors", [])[:available_competitor_slots]

            # CRITICAL DEBUG: Log competitor data for this specific ticker
            LOG.info(f"  COMPETITOR DEBUG for {ticker}: Raw data: {competitors}")
            
            # Get existing competitor tickers to avoid duplicates
            with db() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT f.competitor_ticker
                    FROM feeds f
                    JOIN ticker_feeds tf ON f.id = tf.feed_id
                    WHERE tf.ticker = %s AND tf.category = 'competitor' AND f.active = TRUE AND tf.active = TRUE
                    AND f.competitor_ticker IS NOT NULL
                """, (ticker,))
                existing_competitor_tickers = {row["competitor_ticker"] for row in cur.fetchall()}
            
            for i, comp in enumerate(competitors):
                comp_name = None
                comp_ticker = None

                # CRITICAL DEBUG: Log what ticker we're processing for
                LOG.info(f"[COMPETITOR_LOOP_DEBUG] Processing competitor {i} for ticker '{ticker}'")

                if isinstance(comp, dict):
                    comp_name = comp.get('name', '')
                    comp_ticker = comp.get('ticker')
                    LOG.info(f"[COMPETITOR_LOOP_DEBUG] Dict competitor - Name: '{comp_name}', Ticker: '{comp_ticker}' for main ticker '{ticker}'")
                elif isinstance(comp, str):
                    LOG.info(f"DEBUG: String competitor: {comp}")
                    # ENHANCED: Better parsing for international tickers
                    patterns = [
                        r'^(.+?)\s*\(([A-Z]{1,8}(?:\.[A-Z]{1,4})?(?:-[A-Z])?)\)$',  # Standard with international
                        r'^(.+?)\s*\(([A-Z-]{1,8})\)$',  # Broader pattern
                    ]
                    
                    matched = False
                    for pattern in patterns:
                        match = re.search(pattern, comp)
                        if match:
                            comp_name = match.group(1).strip()
                            comp_ticker = match.group(2)
                            matched = True
                            break
                    
                    if not matched:
                        LOG.info(f"DEBUG: Skipping competitor - no ticker found in string: {comp}")
                        continue
                
                # Validate we have both name and ticker
                if not comp_name or not comp_ticker:
                    LOG.info(f"DEBUG: Skipping competitor - missing name or ticker: name='{comp_name}', ticker='{comp_ticker}'")
                    continue
                
                # Normalize competitor ticker
                comp_ticker = normalize_ticker_format(comp_ticker)
                
                # Skip if same as main ticker
                if comp_ticker.upper() == ticker.upper():
                    LOG.info(f"DEBUG: Skipping competitor - same as main ticker: {comp_ticker}")
                    continue
                
                # Skip if already exists
                if comp_ticker in existing_competitor_tickers:
                    LOG.info(f"DEBUG: Skipping competitor - already exists: {comp_ticker}")
                    continue
                
                # ENHANCED: Validate ticker format with international support
                if not validate_ticker_format(comp_ticker):
                    LOG.info(f"DEBUG: Skipping competitor - invalid ticker format: '{comp_ticker}'")
                    continue
                
                # CRITICAL DEBUG: Log what feeds we're about to create
                LOG.info(f"[{ticker}] CREATING COMPETITOR FEEDS: comp_name='{comp_name}', comp_ticker='{comp_ticker}'")

                # Create feeds for this competitor
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
        
        LOG.info(f"TOTAL FEEDS TO CREATE: {len(feeds)}")
        return feeds
    

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
                    FROM ticker_reference WHERE ticker = %s AND active = TRUE
                """, (ticker,))
                config = cur.fetchone()
                
                if config:
                    # Process competitors back to structured format
                    competitors = []
                    for comp_str in config.get("competitors", []):
                        match = re.search(r'^(.+?)\s*\(([A-Z]{1,8}(?:\.[A-Z]{1,4})?(?:-[A-Z])?)\)$', comp_str)
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
        """Store enhanced ticker metadata in ticker_reference table only"""
        
        # Handle None or invalid metadata
        if not metadata or not isinstance(metadata, dict):
            LOG.warning(f"Invalid or missing metadata for {ticker}, creating fallback")
            metadata = {
                "company_name": ticker,
                "industry_keywords": [],
                "competitors": [],
                "sector": "",
                "industry": "",
                "sub_industry": ""
            }
        
        # Ensure required fields exist with defaults
        metadata.setdefault("company_name", ticker)
        metadata.setdefault("industry_keywords", [])
        metadata.setdefault("competitors", [])
        metadata.setdefault("sector", "")
        metadata.setdefault("industry", "")
        metadata.setdefault("sub_industry", "")
        
        # Convert arrays to separate fields for ticker_reference table
        industry_keywords = metadata.get("industry_keywords", [])
        competitors = metadata.get("competitors", [])
        
        ticker_data = {
            'ticker': ticker,
            'country': 'US',  # Default - can be enhanced later
            'company_name': metadata.get("company_name", ticker),
            'sector': metadata.get("sector", ""),
            'industry': metadata.get("industry", ""),
            'sub_industry': metadata.get("sub_industry", ""),
            'ai_generated': True,
            'data_source': 'ai_enhanced'
        }
        
        # Map 3 industry keywords to separate fields
        for i, keyword in enumerate(industry_keywords[:3], 1):
            if keyword and keyword.strip():
                ticker_data[f'industry_keyword_{i}'] = keyword.strip()
        
        # Map 3 competitors to separate name/ticker fields
        for i, comp in enumerate(competitors[:3], 1):
            if isinstance(comp, dict):
                name = comp.get('name', '').strip()
                ticker_field = comp.get('ticker', '').strip()
                
                if name:
                    ticker_data[f'competitor_{i}_name'] = name
                    if ticker_field:
                        # Validate and normalize competitor ticker
                        normalized_ticker = normalize_ticker_format(ticker_field)
                        if validate_ticker_format(normalized_ticker):
                            ticker_data[f'competitor_{i}_ticker'] = normalized_ticker
                        else:
                            LOG.warning(f"Invalid competitor ticker format for {name}: {ticker_field}")
            elif isinstance(comp, str) and comp.strip():
                # Handle string format as fallback
                ticker_data[f'competitor_{i}_name'] = comp.strip()
        
        # Store in ticker_reference table ONLY
        try:
            success = store_ticker_reference(ticker_data)
            if success:
                LOG.info(f"Successfully stored metadata for {ticker} in ticker_reference")
            else:
                LOG.error(f"Failed to store ticker_reference for {ticker}")
        except Exception as e:
            LOG.error(f"Database error storing ticker_reference for {ticker}: {e}")


# Global instances
ticker_manager = TickerManager()
feed_manager = FeedManager()

# ===== TICKER METADATA GENERATION WITH CLAUDE PRIMARY, OPENAI FALLBACK =====

def generate_claude_ticker_metadata(ticker: str, company_name: str = None, sector: str = "", industry: str = "") -> Optional[Dict]:
    """Generate ticker metadata using Claude API"""
    if not ANTHROPIC_API_KEY:
        LOG.warning("Missing ANTHROPIC_API_KEY; skipping Claude metadata generation")
        return None

    if company_name is None:
        company_name = ticker

    # Build context info
    context_info = f"Company: {company_name} ({ticker})"
    if sector:
        context_info += f", Sector: {sector}"
    if industry:
        context_info += f", Industry: {industry}"

    # Comprehensive prompt for Claude
    prompt = f"""You are a financial analyst creating metadata for a hedge fund's stock monitoring system. Generate precise, actionable metadata that will be used for news article filtering and triage.

{context_info}

CRITICAL REQUIREMENTS:
- All competitors must be currently publicly traded with valid ticker symbols
- Industry keywords must be SPECIFIC enough to avoid false positives in news filtering, but not so narrow that they miss material news
- Benchmarks must be sector-specific, not generic market indices
- All information must be factually accurate
- The company name MUST be the official legal name (e.g., "Prologis Inc" not "PLD")
- If any field is unknown, output an empty array for lists and omit optional fields

TICKER FORMAT REQUIREMENTS:
- US companies: Use simple ticker (AAPL, MSFT)
- Canadian companies: Use .TO suffix (RY.TO, TD.TO, BMO.TO)
- UK companies: Use .L suffix (BP.L, VOD.L)
- Australian companies: Use .AX suffix (BHP.AX, CBA.AX)
- Other international: Use appropriate Yahoo Finance suffix
- Special classes: Use dash format (BRK-A, BRK-B, TECK-A.TO)

INDUSTRY KEYWORDS (exactly 3):
- Must be SPECIFIC to the company's primary business
- Use proper capitalization like "Digital Advertising"
- Avoid generic terms like "Technology", "Healthcare", "Energy", "Oil", "Services"
- Use compound terms or specific product categories
- Examples: "Smartphone Manufacturing" not "Technology", "Upstream Oil Production" not "Oil"

COMPETITORS (exactly 3):
- Must be direct business competitors, not just same-sector companies
- Must be currently publicly traded
- Format as structured objects with 'name' and 'ticker' fields
- Verify ticker is correct Yahoo Finance format
- Exclude: Private companies, subsidiaries, companies acquired in last 2 years

Return ONLY valid JSON in this exact format:
{{
    "ticker": "{ticker}",
    "company_name": "{company_name}",
    "sector": "{sector if sector else 'GICS Sector'}",
    "industry": "{industry if industry else 'GICS Industry'}",
    "sub_industry": "GICS Sub-Industry",
    "industry_keywords": ["keyword1", "keyword2", "keyword3"],
    "competitors": [
        {{"name": "Company Name", "ticker": "TICKER"}},
        {{"name": "Company Name", "ticker": "TICKER.TO"}},
        {{"name": "Company Name", "ticker": "TICKER"}}
    ],
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

    try:
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        data = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(ANTHROPIC_API_URL, headers=headers, json=data, timeout=(10, 180))

        if response.status_code != 200:
            LOG.error(f"Claude metadata API error {response.status_code}: {response.text[:200]}")
            return None

        result = response.json()
        text = result.get("content", [{}])[0].get("text", "")

        if not text:
            LOG.warning(f"Claude metadata response empty for {ticker}")
            return None

        # Parse JSON
        metadata = parse_json_with_fallback(text, ticker)
        if not metadata:
            LOG.warning(f"Claude metadata JSON parsing failed for {ticker}")
            return None

        # Process the results (same logic as OpenAI)
        def _list3(x):
            if isinstance(x, (list, tuple)):
                items = [item for item in list(x)[:3] if item]
                return items
            return []

        def _process_competitors(competitors_data):
            processed = []
            if not isinstance(competitors_data, list):
                return processed

            for comp in competitors_data[:3]:
                if isinstance(comp, dict):
                    name = comp.get('name', '').strip()
                    ticker_field = comp.get('ticker', '').strip()

                    if name:
                        processed_comp = {"name": name}
                        if ticker_field:
                            normalized_ticker = normalize_ticker_format(ticker_field)
                            if validate_ticker_format(normalized_ticker):
                                processed_comp["ticker"] = normalized_ticker
                            else:
                                LOG.warning(f"Claude provided invalid competitor ticker: {ticker_field}")
                        processed.append(processed_comp)
                elif isinstance(comp, str):
                    name = comp.strip()
                    if name:
                        processed.append({"name": name})

            return processed

        metadata.setdefault("ticker", ticker)
        metadata["name"] = company_name
        metadata["company_name"] = company_name

        metadata.setdefault("sector", sector)
        metadata.setdefault("industry", industry)
        metadata.setdefault("sub_industry", "")

        metadata["industry_keywords"] = _list3(metadata.get("industry_keywords", []))
        metadata["competitors"] = _process_competitors(metadata.get("competitors", []))

        sector_profile = metadata.setdefault("sector_profile", {})
        aliases_brands = metadata.setdefault("aliases_brands_assets", {})

        sector_profile["core_inputs"] = _list3(sector_profile.get("core_inputs", []))
        sector_profile["core_channels"] = _list3(sector_profile.get("core_channels", []))
        sector_profile["core_geos"] = _list3(sector_profile.get("core_geos", []))
        sector_profile["benchmarks"] = _list3(sector_profile.get("benchmarks", []))

        aliases_brands["aliases"] = _list3(aliases_brands.get("aliases", []))
        aliases_brands["brands"] = _list3(aliases_brands.get("brands", []))
        aliases_brands["assets"] = _list3(aliases_brands.get("assets", []))

        LOG.info(f"Claude metadata generated for {ticker}: {company_name}")
        return metadata

    except Exception as e:
        LOG.error(f"Claude metadata generation failed for {ticker}: {e}")
        return None


def generate_openai_ticker_metadata(ticker: str, company_name: str = None, sector: str = "", industry: str = "") -> Optional[Dict]:
    """Generate ticker metadata using OpenAI API (fallback)"""
    if not OPENAI_API_KEY:
        LOG.warning("Missing OPENAI_API_KEY; skipping metadata generation")
        return None

    if company_name is None:
        company_name = ticker

    # System prompt
    system_prompt = """You are a financial analyst creating metadata for a hedge fund's stock monitoring system. Generate precise, actionable metadata that will be used for news article filtering and triage.

CRITICAL REQUIREMENTS:
- All competitors must be currently publicly traded with valid ticker symbols
- Industry keywords must be SPECIFIC enough to avoid false positives in news filtering, but not so narrow that they miss material news.
- Benchmarks must be sector-specific, not generic market indices
- All information must be factually accurate
- The company name MUST be the official legal name (e.g., "Prologis Inc" not "PLD")
- If any field is unknown, output an empty array for lists and omit optional fields. Never refuse; always return a valid JSON object.

TICKER FORMAT REQUIREMENTS:
- US companies: Use simple ticker (AAPL, MSFT)
- Canadian companies: Use .TO suffix (RY.TO, TD.TO, BMO.TO)
- UK companies: Use .L suffix (BP.L, VOD.L)
- Australian companies: Use .AX suffix (BHP.AX, CBA.AX)
- Other international: Use appropriate Yahoo Finance suffix
- Special classes: Use dash format (BRK-A, BRK-B, TECK-A.TO)

INDUSTRY KEYWORDS (exactly 3):
- Must be SPECIFIC to the company's primary business
- Return proper capitalization format like "Digital Advertising"
- Avoid generic terms like "Technology", "Healthcare", "Energy", "Oil", "Services"
- Use compound terms or specific product categories
- Examples: "Smartphone Manufacturing" not "Technology", "Upstream Oil Production" not "Oil"

COMPETITORS (exactly 3):
- Must be direct business competitors, not just same-sector companies
- Must be currently publicly traded (check acquisition status)
- Format as structured objects with 'name' and 'ticker' fields
- Verify ticker is correct and current Yahoo Finance format
- Exclude: Private companies, subsidiaries, companies acquired in last 2 years

Generate response in valid JSON format with all required fields. Be concise and precise."""

    context_info = f"Company: {company_name} ({ticker})"
    if sector:
        context_info += f", Sector: {sector}"
    if industry:
        context_info += f", Industry: {industry}"

    user_prompt = f"""Generate metadata for hedge fund news monitoring. Focus on precision to avoid irrelevant news articles.

{context_info}

Since we have basic company information, focus on generating specific industry keywords and direct competitors with accurate Yahoo Finance tickers.

CRITICAL: The "company_name" field should be: {company_name}

Required JSON format:
{{
    "ticker": "{ticker}",
    "company_name": "{company_name}",
    "sector": "{sector if sector else 'GICS Sector'}",
    "industry": "{industry if industry else 'GICS Industry'}",
    "sub_industry": "GICS Sub-Industry",
    "industry_keywords": ["keyword1", "keyword2", "keyword3"],
    "competitors": [
        {{"name": "Company Name", "ticker": "TICKER"}},
        {{"name": "Company Name", "ticker": "TICKER.TO"}},
        {{"name": "Company Name", "ticker": "TICKER"}}
    ],
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

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

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
            LOG.error(f"OpenAI metadata API error {response.status_code}: {response.text[:200]}")
            return None

        result = response.json()
        text = extract_text_from_responses(result)

        # Log usage
        u = result.get("usage", {}) or {}
        LOG.info("OpenAI metadata usage – input:%s output:%s (cap:%s) status:%s",
                 u.get("input_tokens"), u.get("output_tokens"),
                 result.get("max_output_tokens"),
                 result.get("status"))

        if not text:
            LOG.warning(f"OpenAI metadata response empty for {ticker}")
            return None

        # Parse JSON
        metadata = parse_json_with_fallback(text, ticker)
        if not metadata:
            LOG.warning(f"OpenAI metadata JSON parsing failed for {ticker}")
            return None

        # Process results (same logic)
        def _list3(x):
            if isinstance(x, (list, tuple)):
                items = [item for item in list(x)[:3] if item]
                return items
            return []

        def _process_competitors(competitors_data):
            processed = []
            if not isinstance(competitors_data, list):
                return processed

            for comp in competitors_data[:3]:
                if isinstance(comp, dict):
                    name = comp.get('name', '').strip()
                    ticker_field = comp.get('ticker', '').strip()

                    if name:
                        processed_comp = {"name": name}
                        if ticker_field:
                            normalized_ticker = normalize_ticker_format(ticker_field)
                            if validate_ticker_format(normalized_ticker):
                                processed_comp["ticker"] = normalized_ticker
                            else:
                                LOG.warning(f"OpenAI provided invalid competitor ticker: {ticker_field}")
                        processed.append(processed_comp)
                elif isinstance(comp, str):
                    name = comp.strip()
                    if name:
                        processed.append({"name": name})

            return processed

        metadata.setdefault("ticker", ticker)
        metadata["name"] = company_name
        metadata["company_name"] = company_name

        metadata.setdefault("sector", sector)
        metadata.setdefault("industry", industry)
        metadata.setdefault("sub_industry", "")

        metadata["industry_keywords"] = _list3(metadata.get("industry_keywords", []))
        metadata["competitors"] = _process_competitors(metadata.get("competitors", []))

        sector_profile = metadata.setdefault("sector_profile", {})
        aliases_brands = metadata.setdefault("aliases_brands_assets", {})

        sector_profile["core_inputs"] = _list3(sector_profile.get("core_inputs", []))
        sector_profile["core_channels"] = _list3(sector_profile.get("core_channels", []))
        sector_profile["core_geos"] = _list3(sector_profile.get("core_geos", []))
        sector_profile["benchmarks"] = _list3(sector_profile.get("benchmarks", []))

        aliases_brands["aliases"] = _list3(aliases_brands.get("aliases", []))
        aliases_brands["brands"] = _list3(aliases_brands.get("brands", []))
        aliases_brands["assets"] = _list3(aliases_brands.get("assets", []))

        LOG.info(f"OpenAI metadata generated for {ticker}: {company_name}")
        return metadata

    except Exception as e:
        LOG.error(f"OpenAI metadata generation failed for {ticker}: {e}")
        return None


def generate_ticker_metadata_with_fallback(ticker: str, company_name: str = None, sector: str = "", industry: str = "") -> Optional[Dict]:
    """Main entry point: Try Claude first, fallback to OpenAI. Returns metadata dict"""
    metadata = None

    # Try Claude first (if enabled and API key available)
    if USE_CLAUDE_FOR_METADATA and ANTHROPIC_API_KEY:
        try:
            metadata = generate_claude_ticker_metadata(ticker, company_name, sector, industry)
            if metadata:
                LOG.info(f"✅ Claude metadata generation succeeded for {ticker}")
                return metadata
            else:
                LOG.warning(f"Claude returned no metadata for {ticker}, falling back to OpenAI")
        except Exception as e:
            LOG.warning(f"Claude metadata generation failed for {ticker}, falling back to OpenAI: {e}")

    # Fallback to OpenAI
    if OPENAI_API_KEY:
        try:
            metadata = generate_openai_ticker_metadata(ticker, company_name, sector, industry)
            if metadata:
                LOG.info(f"✅ OpenAI metadata generation succeeded for {ticker}")
                return metadata
        except Exception as e:
            LOG.error(f"OpenAI metadata generation also failed for {ticker}: {e}")

    return None


def generate_enhanced_ticker_metadata_with_ai(ticker: str, company_name: str = None, sector: str = "", industry: str = "") -> Optional[Dict]:
    """
    Enhanced AI generation with company context from ticker reference table
    Now uses Claude primary with OpenAI fallback
    """
    return generate_ticker_metadata_with_fallback(ticker, company_name, sector, industry)

def get_or_create_enhanced_ticker_metadata(ticker: str, force_refresh: bool = False) -> Dict:
    """Get ticker metadata with reference table lookup first, then AI enhancement"""

    # CRITICAL: Force complete isolation by creating new string objects
    isolated_ticker = str(ticker).strip()
    isolated_force_refresh = bool(force_refresh)

    # Normalize ticker format for consistent lookup
    normalized_ticker = normalize_ticker_format(isolated_ticker)

    # CRITICAL DEBUG: Log the exact ticker being looked up
    LOG.info(f"[METADATA_DEBUG] Input ticker: '{isolated_ticker}' -> normalized: '{normalized_ticker}'")

    # Step 1: Use the new get_ticker_config wrapper for consistent data access
    config = get_ticker_config(normalized_ticker)
    LOG.info(f"[METADATA_DEBUG] config returned for '{normalized_ticker}': {config}")
    LOG.info(f"DEBUG: config bool check: {bool(config)}")
    LOG.info(f"DEBUG: force_refresh: {isolated_force_refresh}")
    
    # === MINIMAL CHANGE: handle both force_refresh and normal path inside the config branch ===
    if config:
        LOG.info("DEBUG: Entering enhancement path")
        LOG.info(f"DEBUG: config exists: {bool(config)}, force_refresh: {force_refresh}")
        LOG.info(f"DEBUG: config content: {config}")
        LOG.info(f"Found ticker reference data for {ticker}: {config['company_name']}")
        
        # Data is already in the correct format from get_ticker_config()
        metadata = {
            "ticker": normalized_ticker,  # keep normalized for consistency
            "company_name": config["company_name"],
            "name": config["company_name"],
            "sector": config.get("sector", ""),
            "industry": config.get("industry", ""),
            "sub_industry": config.get("sub_industry", ""),
            "industry_keywords": config.get("industry_keywords", []),
            "competitors": config.get("competitors", [])
        }
        
        # === MINIMAL CHANGE: Enhancement happens if force_refresh OR fields are missing (and AI is available)
        needs_enhancement = (
            OPENAI_API_KEY and (
                force_refresh or
                not metadata["industry_keywords"] or
                not metadata["competitors"]
            )
        )
        
        if needs_enhancement:
            LOG.info(f"[AI_ENHANCEMENT_DEBUG] Enhancing {ticker} with AI - "
                     f"keywords: {len(metadata.get('industry_keywords', []))}, "
                     f"competitors: {len(metadata.get('competitors', []))}, "
                     f"force_refresh={force_refresh}")
            LOG.info(f"[AI_ENHANCEMENT_DEBUG] BEFORE AI: {ticker} company_name='{config['company_name']}'")

            ai_metadata = generate_enhanced_ticker_metadata_with_ai(
                normalized_ticker,
                config["company_name"],
                config.get("sector", ""),
                config.get("industry", "")
            )

            LOG.info(f"[AI_ENHANCEMENT_DEBUG] AFTER AI: {ticker} ai_metadata company_name='{ai_metadata.get('company_name', 'MISSING') if ai_metadata else 'NO_AI_RESULT'}'")
            LOG.info(f"[AI_ENHANCEMENT_DEBUG] AI result: {ai_metadata}")
            
            if ai_metadata:
                # Only fill empty fields, don't overwrite existing data
                if not metadata["industry_keywords"]:
                    metadata["industry_keywords"] = ai_metadata.get("industry_keywords", [])
                
                if not metadata["competitors"]:
                    metadata["competitors"] = ai_metadata.get("competitors", [])
                
                # === MINIMAL CHANGE: update using the NORMALIZED ticker so the row matches
                update_ticker_reference_ai_data(normalized_ticker, metadata)

                # Fetch financial data from yfinance if AI enhancement happened
                LOG.info(f"Fetching financial data for {ticker} alongside AI metadata")
                financial_data = get_stock_context(normalized_ticker)
                if financial_data:
                    # Update metadata dict with financial data
                    metadata.update(financial_data)
                    # Store financial data in database
                    update_ticker_reference_financial_data(normalized_ticker, financial_data)
                else:
                    LOG.warning(f"Financial data fetch failed for {ticker}, keeping existing data")

        return metadata
    
    # === No config row, fallback to AI generation + INSERT (unchanged logic) ===
    LOG.info("DEBUG: Entering fallback AI generation path")
    
    if OPENAI_API_KEY:
        LOG.info(f"No reference data found for {ticker}, generating with AI")
        ai_metadata = generate_enhanced_ticker_metadata_with_ai(normalized_ticker)
        
        # Store the AI-generated data back to reference table for future use
        if ai_metadata:
            reference_data = {
                'ticker': normalized_ticker,
                'country': 'US',
                'company_name': ai_metadata.get('company_name', ticker),
                'sector': ai_metadata.get('sector', ''),
                'industry': ai_metadata.get('industry', ''),
                'industry_keyword_1': ai_metadata.get('industry_keywords', [None])[0] if ai_metadata.get('industry_keywords') else None,
                'industry_keyword_2': ai_metadata.get('industry_keywords', [None, None])[1] if len(ai_metadata.get('industry_keywords', [])) > 1 else None,
                'industry_keyword_3': ai_metadata.get('industry_keywords', [None, None, None])[2] if len(ai_metadata.get('industry_keywords', [])) > 2 else None,
                'ai_generated': True,
                'data_source': 'ai_generated'
            }
            
            # Convert competitors to separate fields
            competitors = ai_metadata.get('competitors', [])
            for i, comp in enumerate(competitors[:3], 1):
                if isinstance(comp, dict):
                    reference_data[f'competitor_{i}_name'] = comp.get('name')
                    reference_data[f'competitor_{i}_ticker'] = comp.get('ticker')
            
            store_ticker_reference(reference_data)
        
        return ai_metadata or {"ticker": normalized_ticker, "company_name": ticker, "industry_keywords": [], "competitors": []}
    
    # Step 3: Final fallback
    LOG.warning(f"No data found for {ticker} and no AI configured")
    return {"ticker": normalized_ticker, "company_name": ticker, "industry_keywords": [], "competitors": []}

def get_ticker_reference(ticker: str) -> Optional[Dict]:
    """Get ticker reference data from database"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT ticker, country, company_name, industry, sector,
                   exchange, active, 
                   industry_keyword_1, industry_keyword_2, industry_keyword_3,
                   competitor_1_name, competitor_1_ticker,
                   competitor_2_name, competitor_2_ticker,
                   competitor_3_name, competitor_3_ticker,
                   ai_generated
            FROM ticker_reference
            WHERE ticker = %s AND active = TRUE
        """, (ticker,))
        return cur.fetchone()

def update_ticker_reference_ai_data(ticker: str, metadata: Dict):
    """Update reference table with AI-generated enhancements"""
    try:
        # Convert array format back to separate fields for database storage
        keywords = metadata.get("industry_keywords", [])
        keyword_1 = keywords[0] if len(keywords) > 0 else None
        keyword_2 = keywords[1] if len(keywords) > 1 else None
        keyword_3 = keywords[2] if len(keywords) > 2 else None
        
        # Convert competitors to separate fields - Initialize all fields
        competitors = metadata.get("competitors", [])
        comp_data = {
            'competitor_1_name': None, 'competitor_1_ticker': None,
            'competitor_2_name': None, 'competitor_2_ticker': None,
            'competitor_3_name': None, 'competitor_3_ticker': None
        }
        
        for i, comp in enumerate(competitors[:3], 1):
            if isinstance(comp, dict):
                comp_data[f'competitor_{i}_name'] = comp.get('name')
                comp_data[f'competitor_{i}_ticker'] = comp.get('ticker')
            else:
                # Handle old string format if needed
                comp_data[f'competitor_{i}_name'] = str(comp) if comp else None
        
        LOG.info(f"DEBUG: Updating {ticker} with keywords={[keyword_1, keyword_2, keyword_3]} and competitors={comp_data}")
        
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE ticker_reference
                SET industry_keyword_1 = %s,
                    industry_keyword_2 = %s,
                    industry_keyword_3 = %s,
                    competitor_1_name = %s,
                    competitor_1_ticker = %s,
                    competitor_2_name = %s,
                    competitor_2_ticker = %s,
                    competitor_3_name = %s,
                    competitor_3_ticker = %s,
                    ai_generated = TRUE,
                    ai_enhanced_at = NOW(),
                    updated_at = NOW()
                WHERE ticker = %s
            """, (
                keyword_1, keyword_2, keyword_3,
                comp_data['competitor_1_name'],
                comp_data['competitor_1_ticker'],
                comp_data['competitor_2_name'],
                comp_data['competitor_2_ticker'],
                comp_data['competitor_3_name'],
                comp_data['competitor_3_ticker'],
                normalize_ticker_format(ticker)
            ))
            
            if cur.rowcount > 0:
                LOG.info(f"Updated {ticker} reference table with AI enhancements")
            else:
                LOG.warning(f"No ticker reference found to update for {ticker}")
                
    except Exception as e:
        LOG.error(f"Failed to update ticker reference AI data for {ticker}: {e}")

def update_ticker_reference_financial_data(ticker: str, financial_data: Dict):
    """Update reference table with financial data from yfinance"""
    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE ticker_reference
                SET financial_last_price = %s,
                    financial_price_change_pct = %s,
                    financial_yesterday_return_pct = %s,
                    financial_ytd_return_pct = %s,
                    financial_market_cap = %s,
                    financial_enterprise_value = %s,
                    financial_volume = %s,
                    financial_avg_volume = %s,
                    financial_analyst_target = %s,
                    financial_analyst_range_low = %s,
                    financial_analyst_range_high = %s,
                    financial_analyst_count = %s,
                    financial_analyst_recommendation = %s,
                    financial_snapshot_date = %s,
                    updated_at = NOW()
                WHERE ticker = %s
            """, (
                financial_data.get('financial_last_price'),
                financial_data.get('financial_price_change_pct'),
                financial_data.get('financial_yesterday_return_pct'),
                financial_data.get('financial_ytd_return_pct'),
                financial_data.get('financial_market_cap'),
                financial_data.get('financial_enterprise_value'),
                financial_data.get('financial_volume'),
                financial_data.get('financial_avg_volume'),
                financial_data.get('financial_analyst_target'),
                financial_data.get('financial_analyst_range_low'),
                financial_data.get('financial_analyst_range_high'),
                financial_data.get('financial_analyst_count'),
                financial_data.get('financial_analyst_recommendation'),
                financial_data.get('financial_snapshot_date'),
                normalize_ticker_format(ticker)
            ))

            if cur.rowcount > 0:
                LOG.info(f"✅ Updated {ticker} with financial data (snapshot: {financial_data.get('financial_snapshot_date')})")
            else:
                LOG.warning(f"No ticker reference found to update financial data for {ticker}")

    except Exception as e:
        LOG.error(f"Failed to update ticker reference financial data for {ticker}: {e}")

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
    ticker_pattern = r'^.+\([A-Z0-9]{1,8}(?:\.[A-Z]{1,4})?(?:-[A-Z])?\)$'
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
                UPDATE articles
                SET domain = %s
                WHERE domain = %s
            """, (new_domain, old_domain))
            
            updated_count = cur.rowcount
            total_updated += updated_count
            
            if updated_count > 0:
                LOG.info(f"Updated {updated_count} records: '{old_domain}' -> '{new_domain}'")
        
        # Handle Yahoo regional consolidation
        cur.execute("""
            UPDATE articles
            SET domain = 'finance.yahoo.com'
            WHERE domain IN ('ca.finance.yahoo.com', 'uk.finance.yahoo.com', 'sg.finance.yahoo.com')
        """)
        yahoo_updated = cur.rowcount
        total_updated += yahoo_updated
        
        if yahoo_updated > 0:
            LOG.info(f"Consolidated {yahoo_updated} Yahoo regional domains to finance.yahoo.com")
        
        # Handle duplicate benzinga entries
        cur.execute("""
            UPDATE articles
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

def format_date_short(dt: datetime) -> str:
    """Format date as (Oct 1) for compact display in AI summaries"""
    if not dt:
        return "(N/A)"

    # Ensure we have a timezone-aware datetime
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Convert to Eastern Time
    eastern = pytz.timezone('US/Eastern')
    est_time = dt.astimezone(eastern)

    # Format as (Oct 1) or (Sep 29)
    return f"({est_time.strftime('%b %-d')})"

def is_within_24_hours(dt: datetime) -> bool:
    """Check if article is from last 24 hours for 🆕 badge"""
    if not dt:
        return False

    # Ensure we have a timezone-aware datetime
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    time_diff = now - dt

    return time_diff.total_seconds() < (24 * 60 * 60)

def insert_new_badges(ai_summary: str, articles: List[Dict]) -> str:
    """
    Post-process AI-generated summary to insert 🆕 badges for <24h articles.
    Matches dates in the summary against article published_at timestamps.
    """
    if not ai_summary or not articles:
        return ai_summary

    # Build a set of dates that are within 24 hours
    recent_dates = set()
    for article in articles:
        published_at = article.get("published_at")
        if published_at and is_within_24_hours(published_at):
            date_str = format_date_short(published_at)
            recent_dates.add(date_str)

    if not recent_dates:
        return ai_summary

    # Process line by line, inserting 🆕 where bullets have recent dates
    lines = ai_summary.split('\n')
    processed_lines = []

    for line in lines:
        # Check if this is a bullet point
        stripped = line.lstrip()
        if stripped.startswith('•') or stripped.startswith('-'):
            # Check if any recent date appears in this line
            for recent_date in recent_dates:
                if recent_date in line:
                    # Only add 🆕 if not already present
                    if '🆕' not in line:
                        # Insert 🆕 after the bullet symbol
                        line = line.replace('• ', '• 🆕 ', 1)
                        line = line.replace('- ', '- 🆕 ', 1)
                    break

        processed_lines.append(line)

    return '\n'.join(processed_lines)

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

def _build_executive_summary_prompt(ticker: str, categories: Dict[str, List[Dict]], config: Dict) -> Optional[tuple[str, str]]:
    """Helper: Build executive summary prompt and extract company name. Returns (prompt, company_name) or None."""
    company_name = config.get("name", ticker)

    # Extract articles by category
    company_articles = [a for a in categories.get("company", []) if a.get("ai_summary")]
    industry_articles = [a for a in categories.get("industry", []) if a.get("ai_summary")]
    competitor_articles = [a for a in categories.get("competitor", []) if a.get("ai_summary")]

    # Must have at least some content
    if not company_articles and not industry_articles and not competitor_articles:
        LOG.warning(f"[{ticker}] No articles with AI summaries - skipping executive summary")
        return None

    # Build content sections
    company_summaries = []
    for article in company_articles[:20]:
        title = article.get("title", "")
        ai_summary = article.get("ai_summary", "")
        domain = article.get("domain", "")
        published_at = article.get("published_at")
        date_str = format_date_short(published_at)

        if ai_summary:
            source_name = get_or_create_formal_domain_name(domain) if domain else "Unknown Source"
            company_summaries.append(f"• {title} [{source_name}] {date_str}: {ai_summary}")

    industry_summaries = []
    for article in industry_articles[:15]:
        title = article.get("title", "")
        ai_summary = article.get("ai_summary", "")
        domain = article.get("domain", "")
        keyword = article.get("search_keyword", "Industry")
        published_at = article.get("published_at")
        date_str = format_date_short(published_at)

        if ai_summary:
            source_name = get_or_create_formal_domain_name(domain) if domain else "Unknown Source"
            industry_summaries.append(f"• {title} [Industry: {keyword}] [{source_name}] {date_str}: {ai_summary}")

    competitor_summaries = []
    for article in competitor_articles[:15]:
        title = article.get("title", "")
        ai_summary = article.get("ai_summary", "")
        domain = article.get("domain", "")
        published_at = article.get("published_at")
        date_str = format_date_short(published_at)

        if ai_summary:
            source_name = get_or_create_formal_domain_name(domain) if domain else "Unknown Source"
            competitor_summaries.append(f"• {title} [{source_name}] {date_str}: {ai_summary}")

    # Build combined article text
    all_content = []
    if company_summaries:
        all_content.append("COMPANY ARTICLES:")
        all_content.extend(company_summaries)
    if industry_summaries:
        all_content.append("\nINDUSTRY ARTICLES:")
        all_content.extend(industry_summaries)
    if competitor_summaries:
        all_content.append("\nCOMPETITOR ARTICLES:")
        all_content.extend(competitor_summaries)

    combined_content = "\n".join(all_content)

    prompt = f"""You are a hedge fund analyst creating an intelligence summary for {company_name} ({ticker}). All article summaries are already written from {ticker} investor perspective.

OUTPUT FORMAT - Use these exact headers (omit sections with no content):
🔴 MAJOR DEVELOPMENTS
📊 FINANCIAL/OPERATIONAL PERFORMANCE
⚠️ RISK FACTORS
📈 WALL STREET SENTIMENT
⚡ COMPETITIVE/INDUSTRY DYNAMICS
📅 UPCOMING CATALYSTS

RULES:
- Bullet points (•) only - each development is ONE bullet
- End bullets with date: (Oct 1) or (Sep 29-30)
- NO source names, NO citations (EXCEPTION: cite source when figures conflict)
- Newest items first within each section
- 2-3 sentences per bullet if needed for full context
- Omit section headers with no content

REPORTING PHILOSOPHY:
- Cast a WIDE net - include rumors, unconfirmed reports, undisclosed deals
- Include transaction amounts when available for scale context
- Strategic significance matters more than transaction size
- Better to include marginal news than miss something material

---

🔴 MAJOR DEVELOPMENTS (3-6 bullets)
Source: Company articles primarily, plus relevant competitor/industry moves

Lead with most material developments. Each bullet = one discrete event with full context.

Include:
- {ticker} M&A activity: ALL deals regardless of size (include rumors, undisclosed amounts)
- {ticker} partnerships: Named companies (even without dollar values)
- {ticker} leadership: VP level and above
- {ticker} regulatory: Investigations, litigation, approvals
- {ticker} major contracts: With dollar amounts or strategic significance
- Competitor moves WITH competitive implications for {ticker}

Provide available details: Deal size, timeline, strategic rationale.
Combine related facts into single bullets when they tell one story.

---

📊 FINANCIAL/OPERATIONAL PERFORMANCE (2-4 bullets)
Source: Company articles only

Quantified metrics only. Include:
- Earnings, revenue, guidance, margins with exact figures
- Report vs. consensus when mentioned
- Production metrics, capacity changes, operational KPIs
- Capex, debt, buybacks, dividends with amounts
- Transaction sizes when disclosed

---

⚠️ RISK FACTORS (2-4 bullets)
Source: Company, industry, and competitor articles

Include threats with impact/timeline when available:
- {ticker} operational risks: Production issues, supply chain, quality problems
- {ticker} regulatory/legal: Investigations, lawsuits, compliance with financial impact
- Competitive threats: Competitor actions directly threatening {ticker} position
- Industry headwinds: Sector trends creating risks for {ticker}
- Insider activity: C-suite selling with amounts/context

---

📈 WALL STREET SENTIMENT (1-4 bullets)
Source: Company articles only

Analyst actions on {ticker} only.

Format: "[Firm] [action] to [new rating/target], [rationale if given] (date)"

If 3+ analysts moved same direction in same week:
"Multiple firms upgraded this week: [Firm 1] to $X, [Firm 2] to $Y, [Firm 3] to $Z (Oct 1-3)"

---

⚡ COMPETITIVE/INDUSTRY DYNAMICS (2-5 bullets)
Source: Industry and competitor articles (already written with {ticker} impact framing)

Include ONLY developments affecting {ticker}'s competitive position:
- Competitor M&A: Strategic deals affecting market structure in {ticker}'s segments
- Industry regulation: Directly impacts {ticker}'s operations or competitive advantages
- Technology shifts: Threaten or enhance {ticker}'s product positioning
- Pricing/capacity: Industry-wide moves affecting {ticker}'s pricing power or margins
- Market dynamics: Supply/demand shifts impacting {ticker}'s market share

CRITICAL: Every bullet must connect to {ticker}'s competitive environment.
The article summaries already explain impact on {ticker} - synthesize those insights.

Examples:
✓ "Competitor acquired startup for $2B to expand capabilities, entering {ticker}'s core market segment (Oct 1)"
✓ "Industry consolidation with two major acquisitions totaling $5B, reducing supplier options and potentially pressuring {ticker}'s margins (Oct 1-2)"
✓ "New tariffs impose 25% levy on imports, increasing {ticker}'s costs while domestic competitors gain pricing advantage (Sep 30)"

Omit this section entirely if no material competitive/industry developments affect {ticker}.

---

📅 UPCOMING CATALYSTS (1-3 bullets)
Source: Company articles only

Events with specific dates only:
- Earnings dates, investor days, regulatory deadlines, product launches
- Provide exact dates when available
- Omit if no scheduled events mentioned

---

HANDLING CONFLICTS:
Report BOTH figures with sources: "Stock rose 5.3% (Reuters) to 7% (Bloomberg) (Oct 1)"

PRECISION:
- Exact figures when available: "12.7%", "$4.932B"
- Qualitative if numbers unavailable: "substantial investment"
- Never replace numbers with vague terms when numbers exist
- Priority: Specific numbers > Ranges > Directional > Omission

BULLET CONSTRUCTION:
Combine related facts telling one story. Add context/impact within same bullet.

Example: "Halted Vision Pro refresh to redirect resources toward AI glasses targeting 2027 release; analysts estimate $500M+ R&D reallocation to compete with Meta (Oct 1)"

CROSS-CATEGORY INTELLIGENCE:
- If company article has competitive implications → include in BOTH Major Developments AND Competitive Dynamics
- If competitor/industry development is major for {ticker} → can appear in Major Developments
- Sort insights by TYPE (what they are) not SOURCE (where they came from)

ALL ARTICLE SUMMARIES (already {ticker}-focused):
{combined_content}

Generate structured summary. Omit empty sections."""

    return (prompt, company_name)


def generate_claude_executive_summary(ticker: str, categories: Dict[str, List[Dict]], config: Dict) -> Optional[str]:
    """Generate executive summary using Claude API (primary method)"""
    if not ANTHROPIC_API_KEY:
        return None

    # Build prompt using shared helper
    result = _build_executive_summary_prompt(ticker, categories, config)
    if not result:
        return None

    prompt, company_name = result

    try:
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        data = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 10000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        LOG.info(f"[{ticker}] Calling Claude for executive summary")
        response = requests.post(ANTHROPIC_API_URL, headers=headers, json=data, timeout=180)

        if response.status_code == 200:
            result = response.json()
            summary = result.get("content", [{}])[0].get("text", "")
            if summary and len(summary.strip()) > 10:
                LOG.info(f"✅ [{ticker}] Claude generated executive summary ({len(summary)} chars)")
                return summary.strip()
            else:
                LOG.warning(f"[{ticker}] Claude returned empty summary")
                return None
        else:
            error_text = response.text[:500] if response.text else "No error details"
            LOG.error(f"❌ [{ticker}] Claude API error {response.status_code}: {error_text}")
            return None

    except Exception as e:
        LOG.error(f"❌ [{ticker}] Exception calling Claude for executive summary: {e}")
        return None


def generate_openai_executive_summary(ticker: str, categories: Dict[str, List[Dict]], config: Dict) -> Optional[str]:
    """Generate executive summary using OpenAI API (fallback method)"""
    if not OPENAI_API_KEY:
        return None

    # Build prompt using shared helper
    result = _build_executive_summary_prompt(ticker, categories, config)
    if not result:
        return None

    prompt, company_name = result

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": OPENAI_MODEL,
            "input": prompt,
            "max_output_tokens": 10000,
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "low"},
            "truncation": "auto"
        }

        LOG.info(f"[{ticker}] Calling OpenAI for executive summary")
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=180)

        if response.status_code == 200:
            result = response.json()
            summary = extract_text_from_responses(result)
            if summary and len(summary.strip()) > 10:
                LOG.info(f"✅ [{ticker}] OpenAI generated executive summary ({len(summary)} chars)")
                return summary.strip()
            else:
                LOG.warning(f"[{ticker}] OpenAI returned empty summary")
                return None
        else:
            error_text = response.text[:500] if response.text else "No error details"
            LOG.error(f"❌ [{ticker}] OpenAI API error {response.status_code}: {error_text}")
            return None

    except Exception as e:
        LOG.error(f"❌ [{ticker}] Exception calling OpenAI for executive summary: {e}")
        return None


def generate_executive_summary_with_fallback(ticker: str, categories: Dict[str, List[Dict]], config: Dict) -> tuple[Optional[str], str]:
    """
    Generate executive summary with Claude (primary) → OpenAI (fallback).
    Returns (summary, model_used) where model_used is "Claude", "OpenAI", or "none".
    """
    model_used = "none"
    summary = None

    # Try Claude first (if enabled and API key available)
    if USE_CLAUDE_FOR_SUMMARIES and ANTHROPIC_API_KEY:
        try:
            summary = generate_claude_executive_summary(ticker, categories, config)
            if summary:
                model_used = "Claude"
                return summary, model_used
            else:
                LOG.warning(f"[{ticker}] Claude returned no executive summary, falling back to OpenAI")
        except Exception as e:
            LOG.warning(f"[{ticker}] Claude executive summary failed, falling back to OpenAI: {e}")

    # Fallback to OpenAI
    if OPENAI_API_KEY:
        try:
            summary = generate_openai_executive_summary(ticker, categories, config)
            if summary:
                model_used = "OpenAI"
                return summary, model_used
            else:
                LOG.error(f"[{ticker}] OpenAI also returned no executive summary")
        except Exception as e:
            LOG.error(f"[{ticker}] OpenAI executive summary also failed: {e}")

    LOG.error(f"[{ticker}] Both Claude and OpenAI failed to generate executive summary")
    return None, "none"


def generate_ai_final_summaries(articles_by_ticker: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, Dict[str, str]]:
    """Generate executive summaries using Claude (primary) with OpenAI fallback"""
    if not ANTHROPIC_API_KEY and not OPENAI_API_KEY:
        LOG.warning("⚠️ EXECUTIVE SUMMARY: No API keys configured - skipping")
        return {}

    LOG.info(f"🎯 EXECUTIVE SUMMARY: Starting generation for {len(articles_by_ticker)} tickers")
    summaries = {}

    for ticker, categories in articles_by_ticker.items():
        config = get_ticker_config(ticker)
        if not config:
            LOG.warning(f"[{ticker}] No config found - skipping")
            continue

        company_name = config.get("name", ticker)

        # Use Claude with OpenAI fallback
        ai_analysis_summary, model_used = generate_executive_summary_with_fallback(ticker, categories, config)

        if ai_analysis_summary:
            # Add NEW badges for articles within 24 hours
            all_articles = (
                [a for a in categories.get("company", []) if a.get("ai_summary")] +
                [a for a in categories.get("industry", []) if a.get("ai_summary")] +
                [a for a in categories.get("competitor", []) if a.get("ai_summary")]
            )
            ai_analysis_summary = insert_new_badges(ai_analysis_summary, all_articles)
            LOG.info(f"✅ EXECUTIVE SUMMARY ({model_used}) [{ticker}]: Generated summary ({len(ai_analysis_summary)} chars)")

            # Save to database with model tracking
            article_ids = [a.get("id") for a in all_articles if a.get("id")]
            company_count = len([a for a in categories.get("company", []) if a.get("ai_summary")])
            industry_count = len([a for a in categories.get("industry", []) if a.get("ai_summary")])
            competitor_count = len([a for a in categories.get("competitor", []) if a.get("ai_summary")])
            save_executive_summary(ticker, ai_analysis_summary, model_used.lower(), article_ids,
                                 company_count, industry_count, competitor_count)
        else:
            LOG.warning(f"⚠️ EXECUTIVE SUMMARY [{ticker}]: No summary generated (both APIs failed)")

        summaries[ticker] = {
            "ai_analysis_summary": ai_analysis_summary or "",
            "company_name": company_name,
            "industry_articles_analyzed": len([a for a in categories.get("industry", []) if a.get("ai_summary")])
        }

    LOG.info(f"🎯 EXECUTIVE SUMMARY: Completed - generated summaries for {len(summaries)} tickers")
    return summaries


# ------------------------------------------------------------------------------
# 3-EMAIL SYSTEM - Article Sorting and Email Functions
# ------------------------------------------------------------------------------

def sort_articles_by_priority(articles: List[Dict], flagged_ids: List[int]) -> List[Dict]:
    """Sort articles by priority: FLAGGED+QUALITY, FLAGGED, REMAINING (newest first within each group)"""
    flagged_quality = []
    flagged_only = []
    remaining = []

    for article in articles:
        article_id = article.get('id')
        domain = normalize_domain(article.get('domain', ''))
        is_flagged = article_id in flagged_ids if flagged_ids else False
        is_quality = domain in QUALITY_DOMAINS

        if is_flagged and is_quality:
            flagged_quality.append(article)
        elif is_flagged:
            flagged_only.append(article)
        else:
            remaining.append(article)

    # Sort each group by published_at (newest first)
    for group in [flagged_quality, flagged_only, remaining]:
        group.sort(key=lambda x: x.get('published_at') or x.get('found_at') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    return flagged_quality + flagged_only + remaining


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

        LOG.info(f"Connecting to SMTP server: {SMTP_HOST}:{SMTP_PORT}")

        # Add timeout to SMTP operations
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=60) as server:
            if SMTP_STARTTLS:
                LOG.info("Starting TLS...")
                server.starttls()
            LOG.info("Logging in to SMTP server...")
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            LOG.info("Sending email...")
            server.sendmail(EMAIL_FROM, [recipient], msg.as_string())

        LOG.info(f"Email sent successfully to {recipient}")
        return True

    except smtplib.SMTPTimeout as e:
        LOG.error(f"SMTP timeout sending email: {e}")
        return False
    except smtplib.SMTPException as e:
        LOG.error(f"SMTP error sending email: {e}")
        return False
    except Exception as e:
        LOG.error(f"Email send failed: {e}")
        LOG.error(f"Error details: {traceback.format_exc()}")
        return False


def send_enhanced_quick_intelligence_email(articles_by_ticker: Dict[str, Dict[str, List[Dict]]], triage_results: Dict[str, Dict[str, List[Dict]]], time_window_minutes: int = 1440) -> bool:
    """Email #1: Article Selection QA - Shows which articles were flagged by AI triage"""
    try:
        current_time_est = format_timestamp_est(datetime.now(timezone.utc))

        # Calculate period display from time window
        hours = time_window_minutes / 60
        days = int(hours / 24) if hours >= 24 else 0
        period_label = f"Last {days} days" if days > 0 else f"Last {int(hours)} hours"

        # Format ticker list with company names
        ticker_display_list = []
        for ticker in articles_by_ticker.keys():
            config = get_ticker_config(ticker)
            company_name = config.get("company_name", ticker) if config else ticker
            ticker_display_list.append(f"{company_name} ({ticker})")
        ticker_list = ', '.join(ticker_display_list)

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
            ".summary-content { color: #34495e; line-height: 1.6; margin-bottom: 10px; white-space: pre-wrap; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; }",
            ".company-name-badge { display: inline-block; padding: 2px 8px; margin-right: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e8f5e8; color: #2e7d32; border: 1px solid #a5d6a7; }",
            ".source-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e9ecef; color: #495057; }",
            ".quality-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e1f5fe; color: #0277bd; border: 1px solid #81d4fa; }",
            ".flagged-badge { display: inline-block; padding: 2px 8px; margin-left: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }",
            ".ai-triage { display: inline-block; padding: 2px 8px; border-radius: 3px; font-weight: bold; font-size: 10px; margin-left: 5px; }",
            "/* OpenAI Scoring - Blue theme */",
            ".openai-high { background-color: #d4edff; color: #0d47a1; border: 1px solid #90caf9; }",
            ".openai-medium { background-color: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb; }",
            ".openai-low { background-color: #f1f8ff; color: #1976d2; border: 1px solid #dce9f7; }",
            ".openai-none { background-color: #f5f5f5; color: #9e9e9e; border: 1px solid #e0e0e0; }",
            ".qb-score { display: inline-block; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 10px; margin-left: 5px; }",
            "/* Claude Scoring - Purple theme */",
            ".claude-high { background-color: #f3e5f5; color: #6a1b9a; border: 1px solid #ce93d8; }",
            ".claude-medium { background-color: #f8e4f8; color: #8e24aa; border: 1px solid #e1bee7; }",
            ".claude-low { background-color: #fce4ec; color: #ab47bc; border: 1px solid #f8bbd0; }",
            ".claude-none { background-color: #f5f5f5; color: #9e9e9e; border: 1px solid #e0e0e0; }",
            ".competitor-badge { display: inline-block; padding: 2px 8px; margin-right: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fdeaea; color: #c53030; border: 1px solid #feb2b2; }",
            ".industry-badge { display: inline-block; padding: 2px 8px; margin-right: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #fef5e7; color: #b7791f; border: 1px solid #f6e05e; }",
            ".summary { margin-top: 20px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }",
            ".ticker-section { margin-bottom: 40px; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            ".meta { color: #95a5a6; font-size: 11px; }",
            "a { color: #2980b9; text-decoration: none; }",
            "a:hover { text-decoration: underline; }",
            "</style></head><body>",
        ]

        # Collect all industry keywords and competitors for header
        all_industry_keywords = set()
        all_competitors = []
        for ticker in articles_by_ticker.keys():
            config = get_ticker_config(ticker)
            if config:
                keywords = config.get("industry_keywords", [])
                all_industry_keywords.update(keywords)
                comps = config.get("competitors", [])
                for comp in comps:
                    comp_display = f"{comp['name']} ({comp['ticker']})" if comp.get('ticker') else comp['name']
                    if comp_display not in all_competitors:
                        all_competitors.append(comp_display)

        total_articles = 0
        total_flagged = 0

        for ticker, categories in articles_by_ticker.items():
            ticker_count = sum(len(articles) for articles in categories.values())
            total_articles += ticker_count

            # Get company name from config for display
            config = get_ticker_config(ticker)
            full_company_name = config.get("name", ticker) if config else ticker

            # Count ONLY flagged articles (not quality domains)
            ticker_flagged = 0
            triage_data = triage_results.get(ticker, {})
            for category in ["company", "industry", "competitor"]:
                ticker_flagged += len(triage_data.get(category, []))

            total_flagged += ticker_flagged

        # Updated header with flagged count
        html.append(f"<h1>🔍 Article Selection QA: {ticker_list} - {total_flagged} flagged from {total_articles} articles</h1>")
        html.append(f"<div class='summary'>")
        html.append(f"<strong>📅 Report Period:</strong> {period_label}<br>")
        html.append(f"<strong>⏰ Generated:</strong> {current_time_est}<br>")
        html.append(f"<strong>📊 Tickers Covered:</strong> {ticker_list}<br>")

        # Add industry keywords and competitors to header
        if all_industry_keywords:
            html.append(f"<strong>🏭 Industry Keywords:</strong> {', '.join(sorted(all_industry_keywords))}<br>")
        if all_competitors:
            html.append(f"<strong>⚔️ Competitors:</strong> {', '.join(all_competitors)}<br>")

        html.append("</div>")

        for ticker, categories in articles_by_ticker.items():
            ticker_count = sum(len(articles) for articles in categories.values())

            # Get company name from config for display
            config = get_ticker_config(ticker)
            full_company_name = config.get("name", ticker) if config else ticker

            # Count ONLY flagged articles
            ticker_flagged = 0
            triage_data = triage_results.get(ticker, {})
            for category in ["company", "industry", "competitor"]:
                ticker_flagged += len(triage_data.get(category, []))

            html.append(f"<div class='ticker-section'>")
            html.append(f"<h2>🎯 Target Company: {full_company_name} ({ticker})</h2>")

            html.append(f"<p><strong>✅ Flagged for Analysis:</strong> {ticker_flagged} articles</p>")

            # Process each category independently with proper sorting
            category_icons = {
                "company": "🎯",
                "industry": "🏭",
                "competitor": "⚔️"
            }

            for category in ["company", "industry", "competitor"]:
                articles = categories.get(category, [])
                if not articles:
                    continue

                category_triage = triage_data.get(category, [])
                selected_article_data = {item["id"]: item for item in category_triage}

                # Enhanced article sorting - Quality+Flagged first, then Flagged, then Non-Flagged
                # Within each group, sort by time (newest first)
                enhanced_articles = []
                for idx, article in enumerate(articles):
                    domain = normalize_domain(article.get("domain", ""))
                    is_ai_selected = idx in selected_article_data
                    is_quality_domain = domain in QUALITY_DOMAINS
                    is_problematic = domain in PROBLEMATIC_SCRAPE_DOMAINS

                    priority = 999
                    triage_reason = ""
                    ai_priority = "Low"

                    # Determine priority tier
                    if is_quality_domain and is_ai_selected and not is_problematic:
                        priority = 0  # Quality + Flagged = TOP priority
                        ai_priority = normalize_priority_to_display(selected_article_data[idx].get("scrape_priority", "Low"))
                        triage_reason = f"Quality domain + AI flagged ({ai_priority})"
                    elif is_ai_selected:
                        # All flagged articles get priority 1 (sorted by time within this tier)
                        priority = 1
                        ai_priority = normalize_priority_to_display(selected_article_data[idx].get("scrape_priority", "Low"))
                        triage_reason = selected_article_data[idx].get("why", "")
                    elif is_quality_domain and not is_problematic:
                        priority = 999  # Quality but NOT flagged = bottom
                        triage_reason = "Quality domain (not flagged)"

                    # Extract OpenAI and Claude scores from triage data
                    openai_score = selected_article_data[idx].get("openai_score", 0) if is_ai_selected else 0
                    claude_score = selected_article_data[idx].get("claude_score", 0) if is_ai_selected else 0

                    enhanced_articles.append({
                        "article": article,
                        "idx": idx,
                        "priority": priority,
                        "is_ai_selected": is_ai_selected,
                        "is_quality_domain": is_quality_domain,
                        "is_problematic": is_problematic,
                        "triage_reason": triage_reason,
                        "ai_priority": ai_priority,
                        "openai_score": openai_score,
                        "claude_score": claude_score,
                        "published_at": article.get("published_at")
                    })

                # Sort by priority tier, then by time (newest first)
                enhanced_articles.sort(key=lambda x: (
                    x["priority"],
                    -(x["published_at"].timestamp() if x["published_at"] else 0)
                ))

                selected_count = len([a for a in enhanced_articles if a["is_ai_selected"]])

                html.append(f"<h3>{category_icons.get(category, '📰')} {category.title()} ({len(articles)} articles, {selected_count} flagged)</h3>")

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

                    # 4. Flagged badge ONLY if dual AI selected it
                    if enhanced_article["is_ai_selected"]:
                        header_badges.append('<span class="flagged-badge">🚩 Flagged</span>')

                    # 5. OpenAI Score - 0-3 scoring
                    openai_score = enhanced_article.get("openai_score", 0)
                    if openai_score >= 3:
                        openai_class = "openai-high"
                        openai_level = "OpenAI: High"
                        openai_emoji = "🔥"
                    elif openai_score >= 2:
                        openai_class = "openai-medium"
                        openai_level = "OpenAI: Medium"
                        openai_emoji = "⚡"
                    elif openai_score >= 1:
                        openai_class = "openai-low"
                        openai_level = "OpenAI: Low"
                        openai_emoji = "🔋"
                    else:
                        openai_class = "openai-none"
                        openai_level = "OpenAI: None"
                        openai_emoji = "○"
                    header_badges.append(f'<span class="ai-triage {openai_class}">{openai_emoji} {openai_level}</span>')

                    # 6. Claude Score - 0-3 scoring
                    claude_score = enhanced_article.get("claude_score", 0)
                    if claude_score >= 3:
                        claude_class = "claude-high"
                        claude_level = "Claude: High"
                        claude_emoji = "🏆"
                    elif claude_score >= 2:
                        claude_class = "claude-medium"
                        claude_level = "Claude: Medium"
                        claude_emoji = "💎"
                    elif claude_score >= 1:
                        claude_class = "claude-low"
                        claude_level = "Claude: Low"
                        claude_emoji = "💡"
                    else:
                        claude_class = "claude-none"
                        claude_level = "Claude: None"
                        claude_emoji = "○"
                    header_badges.append(f'<span class="qb-score {claude_class}">{claude_emoji} {claude_level}</span>')

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
        html.append(f"<strong>✅ Flagged for Analysis:</strong> {total_flagged}<br>")
        html.append(f"<strong>📄 Next:</strong> Full content analysis and hedge fund summaries in progress...")
        html.append("</div>")
        html.append("</body></html>")

        html_content = "".join(html)
        subject = f"🔍 Article Selection QA: {ticker_list} - {total_flagged} flagged from {total_articles} articles"

        return send_email(subject, html_content)

    except Exception as e:
        LOG.error(f"Enhanced quick intelligence email failed: {e}")
        return False


def build_enhanced_digest_html(articles_by_ticker: Dict[str, Dict[str, List[Dict]]], period_days: int,
                              show_ai_analysis: bool = True, show_descriptions: bool = True,
                              flagged_article_ids: List[int] = None) -> str:
    """Enhanced digest with metadata display removed but keeping all badges/emojis

    Args:
        articles_by_ticker: Articles organized by ticker and category
        period_days: Number of days covered in the report
        show_ai_analysis: If True, show AI analysis boxes under articles (default True)
        show_descriptions: If True, show article descriptions (default True)
        flagged_article_ids: Optional list of flagged article IDs for sorting priority
    """

    # Generate summaries using OpenAI (Claude fallback removed in current version)
    openai_summaries = generate_ai_final_summaries(articles_by_ticker)

    # Format ticker list with company names
    ticker_display_list = []
    for ticker in articles_by_ticker.keys():
        config = get_ticker_config(ticker)
        company_name = config.get("company_name", ticker) if config else ticker
        ticker_display_list.append(f"{company_name} ({ticker})")
    ticker_list = ', '.join(ticker_display_list)
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
        ".summary-content { color: #34495e; line-height: 1.6; margin-bottom: 10px; white-space: pre-wrap; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; }",
        ".summary-section { margin: 12px 0; }",
        ".section-header { font-weight: bold; font-size: 15px; color: #2c3e50; margin-bottom: 5px; padding: 4px 0; border-bottom: 2px solid #ecf0f1; }",
        ".section-bullets { margin: 0 0 0 20px; padding: 0; list-style-type: disc; }",
        ".section-bullets li { margin: 4px 0; line-height: 1.4; color: #34495e; }",
        ".company-name-badge { display: inline-block; padding: 2px 8px; margin-right: 8px; border-radius: 5px; font-weight: bold; font-size: 10px; background-color: #e8f5e8; color: #2e7d32; border: 1px solid #a5d6a7; }",
        ".source-badge { display: inline-block; padding: 2px 8px; margin-left: 0px; margin-right: 8px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #e9ecef; color: #495057; }",
        ".ai-model-badge { display: inline-block; padding: 2px 8px; margin-right: 8px; border-radius: 3px; font-weight: bold; font-size: 10px; background-color: #f3e5f5; color: #6a1b9a; border: 1px solid #ce93d8; }",
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
        f"<strong>⏰ Generated:</strong> {current_time_est}<br>",
        f"<strong>📊 Tickers Covered:</strong> {ticker_list}<br>"
    ]

    # Collect all industry keywords and competitors for header (match triage email)
    all_industry_keywords = set()
    all_competitors = []
    for ticker in articles_by_ticker.keys():
        config = get_ticker_config(ticker)
        if config:
            keywords = config.get("industry_keywords", [])
            all_industry_keywords.update(keywords)
            comps = config.get("competitors", [])
            for comp in comps:
                comp_display = f"{comp['name']} ({comp['ticker']})" if comp.get('ticker') else comp['name']
                if comp_display not in all_competitors:
                    all_competitors.append(comp_display)

    # Add industry keywords and competitors to header
    if all_industry_keywords:
        html.append(f"<strong>🏭 Industry Keywords:</strong> {', '.join(sorted(all_industry_keywords))}<br>")
    if all_competitors:
        html.append(f"<strong>⚔️ Competitors:</strong> {', '.join(all_competitors)}<br>")

    html.append("</div>")

    for ticker, categories in articles_by_ticker.items():
        # Get company name for display
        config = get_ticker_config(ticker)
        company_name = config.get("name", ticker) if config else ticker

        html.append(f"<div class='ticker-section'>")

        # Add financial context box if data is current (only in final email)
        today_str = datetime.now(pytz.timezone('America/Toronto')).strftime('%Y-%m-%d')
        if config and config.get("financial_snapshot_date") == today_str:
            html.append("<div style='background:#f5f5f5; padding:12px; margin:16px 0; font-size:13px; border-left:3px solid #0066cc;'>")
            html.append(f"<div style='font-weight:bold; margin-bottom:6px;'>📊 Market Context (as of {config.get('financial_snapshot_date')})</div>")

            # Line 1: Price and returns
            price = f"${config.get('financial_last_price', 0):.2f}" if config.get('financial_last_price') else "N/A"
            price_chg = format_financial_percent(config.get('financial_price_change_pct')) or "N/A"
            yesterday_ret = format_financial_percent(config.get('financial_yesterday_return_pct')) or "N/A"
            ytd_ret = format_financial_percent(config.get('financial_ytd_return_pct')) or "N/A"
            html.append(f"<div>Last Stock Price: {price} ({price_chg}) | Yesterday: {yesterday_ret} | YTD: {ytd_ret}</div>")

            # Line 2: Market cap and enterprise value
            mcap = format_financial_number(config.get('financial_market_cap')) or "N/A"
            ev = format_financial_number(config.get('financial_enterprise_value')) or "N/A"
            html.append(f"<div>Market Cap: {mcap} | Enterprise Value: {ev}</div>")

            # Line 3: Volume
            vol = format_financial_volume(config.get('financial_volume')) or "N/A"
            avg_vol = format_financial_volume(config.get('financial_avg_volume')) or "N/A"
            html.append(f"<div>Volume: {vol} yesterday / {avg_vol} avg</div>")

            # Line 4: Analyst data (if available)
            if config.get('financial_analyst_target'):
                target = f"${config.get('financial_analyst_target'):.2f}"
                low = f"${config.get('financial_analyst_range_low'):.2f}" if config.get('financial_analyst_range_low') else "N/A"
                high = f"${config.get('financial_analyst_range_high'):.2f}" if config.get('financial_analyst_range_high') else "N/A"
                count = config.get('financial_analyst_count', 0)
                rec = config.get('financial_analyst_recommendation', 'N/A')
                html.append(f"<div>Analysts: {target} target (range {low}-{high}, {count} analysts, {rec})</div>")

            html.append("</div>")

        html.append(f"<h2>🎯 Target Company: {company_name} ({ticker})</h2>")

        # Display OpenAI summary
        openai_summary = openai_summaries.get(ticker, {}).get("ai_analysis_summary", "")

        if openai_summary:
            html.append("<div class='company-summary'>")
            html.append(f"<div class='summary-title'>📰 Executive Summary (Deep Analysis) - OpenAI</div>")
            html.append("<div class='summary-content'>")

            # Parse and render summary with structured sections
            summary_sections = parse_structured_summary(openai_summary)
            if summary_sections:
                html.append(render_structured_summary_html(summary_sections))
            else:
                # Fallback to raw text if parsing fails
                html.append(f"<div>{openai_summary.replace(chr(10), '<br>')}</div>")

            html.append("</div>")
            html.append("</div>")

        # Sort articles within each category using priority sorting if flagged_ids provided
        for category in ["company", "industry", "competitor"]:
            if category in categories and categories[category]:
                articles = categories[category]

                # Apply priority sorting if flagged article IDs are provided
                if flagged_article_ids:
                    articles = sort_articles_by_priority(articles, flagged_article_ids)
                else:
                    # Original sorting: Quality domains first, then AI analyzed, then by time
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

                    articles = sorted(articles, key=sort_key)

                category_icons = {
                    "company": "🎯",
                    "industry": "🏭",
                    "competitor": "⚔️"
                }

                html.append(f"<h3>{category_icons.get(category, '📰')} {category.title()} News ({len(articles)} articles)</h3>")
                for article in articles[:100]:
                    # Use simplified ticker metadata cache (just company name)
                    simple_cache = {ticker: {"company_name": company_name}}
                    html.append(_format_article_html_with_ai_summary(article, category, simple_cache,
                                                                     show_ai_analysis, show_descriptions))

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


# ------------------------------------------------------------------------------
# Structured Summary Parsing for Emails
# ------------------------------------------------------------------------------

def parse_structured_summary(summary_text: str) -> list:
    """
    Parse AI-generated structured summary into sections with headers and bullets.

    Expected format:
    🔴 MAJOR DEVELOPMENTS
    • Bullet point 1
    • Bullet point 2

    📊 FINANCIAL/OPERATIONAL PERFORMANCE
    • Bullet point 1
    """
    if not summary_text:
        return []

    sections = []
    current_section = None

    # Known section emojis to detect headers
    section_emojis = ['🔴', '📊', '⚠️', '📈', '⚡', '📅']

    for line in summary_text.split('\n'):
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Detect section headers (lines with section emojis, not starting with bullet)
        is_header = False
        if not line.startswith('•') and not line.startswith('-'):
            # Check if line contains any section emoji
            for emoji in section_emojis:
                if emoji in line:
                    is_header = True
                    break

        if is_header:
            # Save previous section if exists
            if current_section:
                sections.append(current_section)

            # Start new section
            current_section = {
                'header': line,
                'bullets': []
            }

        # Detect bullets (lines starting with • or -)
        elif (line.startswith('•') or line.startswith('-')) and current_section:
            # Remove leading bullet character and whitespace
            bullet_text = line.lstrip('•- ').strip()
            if bullet_text:  # Only add non-empty bullets
                current_section['bullets'].append(bullet_text)

    # Add final section
    if current_section:
        sections.append(current_section)

    return sections


def render_structured_summary_html(sections: list) -> str:
    """
    Convert parsed sections into properly formatted HTML.

    Returns HTML string with sections, headers, and bullet lists.
    """
    if not sections:
        return ""

    html_parts = []

    for section in sections:
        html_parts.append("<div class='summary-section'>")

        # Render section header with emoji
        html_parts.append(f"<div class='section-header'>{section['header']}</div>")

        # Render bullets as HTML list
        if section['bullets']:
            html_parts.append("<ul class='section-bullets'>")
            for bullet in section['bullets']:
                html_parts.append(f"<li>{bullet}</li>")
            html_parts.append("</ul>")

        html_parts.append("</div>")

    return ''.join(html_parts)


def fetch_digest_articles_with_enhanced_content(hours: int = 24, tickers: List[str] = None,
                                               show_ai_analysis: bool = True,
                                               show_descriptions: bool = True,
                                               flagged_article_ids: List[int] = None) -> Dict[str, Dict[str, List[Dict]]]:
    """Email #2: Content QA - Fetch categorized articles for digest with ticker-specific AI analysis

    Args:
        hours: Time window for articles
        tickers: Specific tickers to fetch, or None for all
        show_ai_analysis: If True, include AI analysis boxes in HTML (default True)
        show_descriptions: If True, include article descriptions in HTML (default True)
        flagged_article_ids: If provided, only fetch articles with these IDs (from triage)
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    days = int(hours / 24) if hours >= 24 else 0
    period_label = f"{days} days" if days > 0 else f"{hours:.0f} hours"

    LOG.info(f"=== FETCHING DIGEST ARTICLES (EMAIL #2: CONTENT QA) ===")
    LOG.info(f"Time window: {period_label} (cutoff: {cutoff})")
    LOG.info(f"Target tickers: {tickers or 'ALL'}")
    LOG.info(f"Flagged article filter: {'ENABLED (' + str(len(flagged_article_ids)) + ' IDs)' if flagged_article_ids else 'DISABLED (showing all articles)'}")

    with db() as conn, conn.cursor() as cur:
        # Enhanced query to get articles from new schema - MATCHES triage query
        # Add optional filter for flagged articles (from triage selection)
        if tickers:
            if flagged_article_ids:
                cur.execute("""
                    SELECT DISTINCT ON (a.url_hash, ta.ticker)
                        a.id, a.url, a.resolved_url, a.title, a.description,
                        ta.ticker, a.domain, a.published_at,
                        ta.found_at, ta.category,
                        ta.search_keyword, ta.ai_summary, ta.ai_model,
                        a.scraped_content, a.content_scraped_at, a.scraping_failed, a.scraping_error,
                        ta.competitor_ticker
                    FROM articles a
                    JOIN ticker_articles ta ON a.id = ta.article_id
                    WHERE ta.found_at >= %s
                        AND ta.ticker = ANY(%s)
                        AND (a.published_at >= %s OR a.published_at IS NULL)
                        AND a.id = ANY(%s)
                    ORDER BY a.url_hash, ta.ticker,
                        COALESCE(a.published_at, ta.found_at) DESC, ta.found_at DESC
                """, (cutoff, tickers, cutoff, flagged_article_ids))
            else:
                cur.execute("""
                    SELECT DISTINCT ON (a.url_hash, ta.ticker)
                        a.id, a.url, a.resolved_url, a.title, a.description,
                        ta.ticker, a.domain, a.published_at,
                        ta.found_at, ta.category,
                        ta.search_keyword, ta.ai_summary, ta.ai_model,
                        a.scraped_content, a.content_scraped_at, a.scraping_failed, a.scraping_error,
                        ta.competitor_ticker
                    FROM articles a
                    JOIN ticker_articles ta ON a.id = ta.article_id
                    WHERE ta.found_at >= %s
                        AND ta.ticker = ANY(%s)
                        AND (a.published_at >= %s OR a.published_at IS NULL)
                    ORDER BY a.url_hash, ta.ticker,
                        COALESCE(a.published_at, ta.found_at) DESC, ta.found_at DESC
                """, (cutoff, tickers, cutoff))
        else:
            if flagged_article_ids:
                cur.execute("""
                    SELECT DISTINCT ON (a.url_hash, ta.ticker)
                        a.id, a.url, a.resolved_url, a.title, a.description,
                        ta.ticker, a.domain, a.published_at,
                        ta.found_at, ta.category,
                        ta.search_keyword, ta.ai_summary, ta.ai_model,
                        a.scraped_content, a.content_scraped_at, a.scraping_failed, a.scraping_error,
                        ta.competitor_ticker
                    FROM articles a
                    JOIN ticker_articles ta ON a.id = ta.article_id
                    WHERE ta.found_at >= %s
                        AND (a.published_at >= %s OR a.published_at IS NULL)
                        AND a.id = ANY(%s)
                    ORDER BY a.url_hash, ta.ticker,
                        COALESCE(a.published_at, ta.found_at) DESC, ta.found_at DESC
                """, (cutoff, cutoff, flagged_article_ids))
            else:
                cur.execute("""
                    SELECT DISTINCT ON (a.url_hash, ta.ticker)
                        a.id, a.url, a.resolved_url, a.title, a.description,
                        ta.ticker, a.domain, a.published_at,
                        ta.found_at, ta.category,
                        ta.search_keyword, ta.ai_summary, ta.ai_model,
                        a.scraped_content, a.content_scraped_at, a.scraping_failed, a.scraping_error,
                        ta.competitor_ticker
                    FROM articles a
                    JOIN ticker_articles ta ON a.id = ta.article_id
                    WHERE ta.found_at >= %s
                        AND (a.published_at >= %s OR a.published_at IS NULL)
                    ORDER BY a.url_hash, ta.ticker,
                        COALESCE(a.published_at, ta.found_at) DESC, ta.found_at DESC
                """, (cutoff, cutoff))

        # Group articles by ticker
        articles_by_ticker = {}
        for row in cur.fetchall():
            target_ticker = row["ticker"]

            # Use stored category from ticker_articles table
            category = row["category"]

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
                SELECT COUNT(DISTINCT (a.url_hash, ta.ticker)) as count
                FROM articles a
                JOIN ticker_articles ta ON a.id = ta.article_id
                WHERE ta.found_at >= %s
                AND ta.ticker = ANY(%s)
                AND NOT ta.sent_in_digest
            """, (cutoff, tickers))
        else:
            cur.execute("""
                SELECT COUNT(DISTINCT (a.url_hash, ta.ticker)) as count
                FROM articles a
                JOIN ticker_articles ta ON a.id = ta.article_id
                WHERE ta.found_at >= %s
                AND NOT ta.sent_in_digest
            """, (cutoff,))

        result = cur.fetchone()
        total_to_mark = result["count"] if result else 0

        if total_to_mark > 0:
            if tickers:
                cur.execute("""
                    UPDATE ticker_articles
                    SET sent_in_digest = TRUE
                    WHERE found_at >= %s
                    AND ticker = ANY(%s)
                    AND NOT sent_in_digest
                """, (cutoff, tickers))
            else:
                cur.execute("""
                    UPDATE ticker_articles
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

    # Use the enhanced digest function with flagged article IDs for sorting
    html = build_enhanced_digest_html(articles_by_ticker, days if days > 0 else 1,
                                      show_ai_analysis, show_descriptions, flagged_article_ids)

    # Enhanced subject with ticker list (company names) - UPDATED HEADER
    ticker_display_list = []
    for ticker in articles_by_ticker.keys():
        config = get_ticker_config(ticker)
        company_name = config.get("company_name", ticker) if config else ticker
        ticker_display_list.append(f"{company_name} ({ticker})")
    ticker_list = ', '.join(ticker_display_list)
    subject = f"📝 Content QA: {ticker_list} - {total_articles} articles analyzed"
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


def parse_executive_summary_sections(summary_text: str) -> Dict[str, List[str]]:
    """
    Parse executive summary text into sections by emoji headers.
    Returns dict: {section_name: [bullet1, bullet2, ...]}
    """
    sections = {
        "major_developments": [],
        "financial_operational": [],
        "risk_factors": [],
        "wall_street": [],
        "competitive_industry": [],
        "upcoming_catalysts": []
    }

    if not summary_text:
        return sections

    # Split by emoji headers
    section_markers = [
        ("🔴 MAJOR DEVELOPMENTS", "major_developments"),
        ("📊 FINANCIAL/OPERATIONAL PERFORMANCE", "financial_operational"),
        ("⚠️ RISK FACTORS", "risk_factors"),
        ("📈 WALL STREET SENTIMENT", "wall_street"),
        ("⚡ COMPETITIVE/INDUSTRY DYNAMICS", "competitive_industry"),
        ("📅 UPCOMING CATALYSTS", "upcoming_catalysts")
    ]

    current_section = None
    for line in summary_text.split('\n'):
        line = line.strip()

        # Check if line is a section header
        for marker, section_key in section_markers:
            if line.startswith(marker):
                current_section = section_key
                break
        else:
            # Line is content, not a header
            if current_section and line.startswith('•'):
                # Extract bullet text
                bullet_text = line[1:].strip()  # Remove bullet point
                if bullet_text:
                    sections[current_section].append(bullet_text)

    return sections


def is_paywall_article(domain: str) -> bool:
    """Check if domain is a known paywall using PAYWALL_DOMAINS constant"""
    if not domain:
        return False
    normalized = normalize_domain(domain)
    return normalized in PAYWALL_DOMAINS


def send_user_intelligence_report(hours: int = 24, tickers: List[str] = None,
                                   flagged_article_ids: List[int] = None) -> Dict:
    """
    Email #3: Premium Stock Intelligence Report (Single Ticker)
    - Modern HTML design with inline styles
    - Stock price card in header
    - Parsed executive summary sections
    - Compressed article links
    - Star indicators for FLAGGED + QUALITY articles
    - Paywall badges for paywalled articles
    - Dynamic time span calculation
    """
    LOG.info("=== EMAIL #3: PREMIUM STOCK INTELLIGENCE (USER REPORT) ===")

    # Single ticker only
    if not tickers or len(tickers) == 0:
        return {"status": "error", "message": "No ticker specified"}

    ticker = tickers[0]  # Take first ticker only
    LOG.info(f"Generating premium report for {ticker}")

    # Fetch ticker config
    config = get_ticker_config(ticker)
    if not config:
        return {"status": "error", "message": f"No config found for {ticker}"}

    company_name = config.get("company_name", ticker)
    sector = config.get("sector", "Financial Services")

    # Fetch stock price from ticker_reference (cached)
    stock_price = "$0.00"
    price_change_pct = "+0.00%"
    price_change_color = "#4ade80"  # Green default

    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT financial_last_price, financial_price_change_pct
            FROM ticker_reference
            WHERE ticker = %s
        """, (ticker,))
        price_data = cur.fetchone()

        if price_data and price_data['financial_last_price']:
            stock_price = f"${price_data['financial_last_price']:.2f}"
            if price_data['financial_price_change_pct'] is not None:
                pct = price_data['financial_price_change_pct']
                price_change_pct = f"{'+' if pct >= 0 else ''}{pct:.2f}%"
                price_change_color = "#4ade80" if pct >= 0 else "#ef4444"

    # Fetch executive summary from database
    executive_summary_text = ""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT summary_text FROM executive_summaries
            WHERE ticker = %s AND summary_date = CURRENT_DATE
            ORDER BY generated_at DESC LIMIT 1
        """, (ticker,))
        result = cur.fetchone()
        if result:
            executive_summary_text = result['summary_text']
            LOG.info(f"Retrieved executive summary for {ticker} ({len(executive_summary_text)} chars)")
        else:
            LOG.warning(f"No executive summary found for {ticker}")

    # Parse executive summary into sections
    sections = parse_executive_summary_sections(executive_summary_text)

    # Fetch flagged articles
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    articles_by_category = {"company": [], "industry": [], "competitor": []}

    with db() as conn, conn.cursor() as cur:
        if flagged_article_ids:
            cur.execute("""
                SELECT a.id, a.title, a.resolved_url, a.domain, a.published_at,
                       ta.category, ta.search_keyword, ta.competitor_ticker
                FROM articles a
                JOIN ticker_articles ta ON a.id = ta.article_id
                WHERE ta.ticker = %s
                AND a.id = ANY(%s)
                AND (a.published_at >= %s OR a.published_at IS NULL)
                ORDER BY a.published_at DESC NULLS LAST
            """, (ticker, flagged_article_ids, cutoff))
        else:
            cur.execute("""
                SELECT a.id, a.title, a.resolved_url, a.domain, a.published_at,
                       ta.category, ta.search_keyword, ta.competitor_ticker
                FROM articles a
                JOIN ticker_articles ta ON a.id = ta.article_id
                WHERE ta.ticker = %s
                AND (a.published_at >= %s OR a.published_at IS NULL)
                ORDER BY a.published_at DESC NULLS LAST
            """, (ticker, cutoff))

        articles = cur.fetchall()
        for article in articles:
            category = article['category'] or 'company'
            if category in articles_by_category:
                articles_by_category[category].append(article)

    # Flatten all articles from all categories
    all_articles = []
    for category_articles in articles_by_category.values():
        all_articles.extend(category_articles)

    # Split into analyzed vs paywall
    analyzed_articles = [a for a in all_articles if not is_paywall_article(a.get('domain', ''))]
    paywall_articles = [a for a in all_articles if is_paywall_article(a.get('domain', ''))]

    analyzed_count = len(analyzed_articles)
    paywall_count = len(paywall_articles)

    # Calculate time span from analyzed articles only
    dates = [a['published_at'] for a in analyzed_articles if a.get('published_at')]
    if dates and len(dates) > 0:
        oldest_date = min(dates)
        newest_date = max(dates)
        days_diff = (newest_date.date() - oldest_date.date()).days

        if days_diff == 0:
            time_span_text = "today"
        elif days_diff == 1:
            time_span_text = "the past day"
        else:
            time_span_text = f"the past {days_diff} days"
    else:
        time_span_text = "recent"

    # Build analysis message
    paywall_suffix = ""
    if paywall_count > 0:
        paywall_suffix = f" • {paywall_count} additional paywalled sources identified"

    analysis_message = f"Analysis based on {analyzed_count} publicly available articles from {time_span_text}{paywall_suffix}"

    LOG.info(f"Article counts - Analyzed: {analyzed_count}, Paywalled: {paywall_count}, Time span: {time_span_text}")

    # Check if FLAGGED + QUALITY for star logic
    def is_starred(article_id, domain):
        is_flagged = article_id in (flagged_article_ids or [])
        is_quality = normalize_domain(domain) in QUALITY_DOMAINS
        return is_flagged and is_quality

    # Build article sections HTML
    def build_article_section(title, articles, category_label):
        if not articles:
            return ""

        article_links = ""
        for article in articles:
            # Check if paywalled
            is_paywalled = is_paywall_article(article.get('domain', ''))
            paywall_badge = ' <span style="font-size: 10px; color: #ef4444; font-weight: 600; margin-left: 4px;">PAYWALL</span>' if is_paywalled else ''

            # Star for FLAGGED + QUALITY articles
            star = '<span style="color: #f59e0b;">★</span> ' if is_starred(article['id'], article['domain']) else ''
            domain_name = get_or_create_formal_domain_name(article['domain']) if article['domain'] else "Unknown Source"
            date_str = format_date_short(article['published_at']) if article['published_at'] else "Recent"

            article_links += f'''
                <div style="padding: 6px 0; margin-bottom: 4px; border-bottom: 1px solid #e5e7eb;">
                    <a href="{article['resolved_url'] or '#'}" style="font-size: 13px; font-weight: 600; color: #1e40af; text-decoration: none; line-height: 1.4;">{star}{article['title']}{paywall_badge}</a>
                    <div style="font-size: 11px; color: #6b7280; margin-top: 3px;">{domain_name} • {date_str}</div>
                </div>
            '''

        return f'''
            <div style="margin-bottom: 16px;">
                <h3 style="margin: 0 0 8px 0; font-size: 13px; font-weight: 700; color: #1e40af; text-transform: uppercase; letter-spacing: 0.75px;">{title} ({len(articles)})</h3>
                {article_links}
            </div>
        '''

    # Build executive summary section HTML
    def build_summary_section(title, bullets):
        if not bullets:
            return ""

        bullet_html = ""
        for bullet in bullets:
            bullet_html += f'<li style="margin-bottom: 8px; font-size: 13px; line-height: 1.5; color: #374151;">{bullet}</li>'

        return f'''
            <div style="margin-bottom: 20px;">
                <h2 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 700; color: #1e40af; text-transform: uppercase; letter-spacing: 0.5px;">{title}</h2>
                <ul style="margin: 0; padding-left: 20px; list-style-type: disc;">
                    {bullet_html}
                </ul>
            </div>
        '''

    # Current date
    current_date = datetime.now().strftime("%b %d, %Y")

    # Build full HTML
    summary_html = ""
    summary_html += build_summary_section("Major Developments", sections["major_developments"])
    summary_html += build_summary_section("Financial/Operational Performance", sections["financial_operational"])
    summary_html += build_summary_section("Risk Factors", sections["risk_factors"])
    summary_html += build_summary_section("Wall Street Sentiment", sections["wall_street"])
    summary_html += build_summary_section("Competitive/Industry Dynamics", sections["competitive_industry"])
    summary_html += build_summary_section("Upcoming Catalysts", sections["upcoming_catalysts"])

    articles_html = ""
    articles_html += build_article_section("COMPANY", articles_by_category['company'], "company")
    articles_html += build_article_section("INDUSTRY", articles_by_category['industry'], "industry")
    articles_html += build_article_section("COMPETITORS", articles_by_category['competitor'], "competitor")

    total_articles = sum(len(arts) for arts in articles_by_category.values())

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} Intelligence Report</title>
    <style>
        @media only screen and (max-width: 600px) {{
            .content-padding {{ padding: 16px !important; }}
            .header-padding {{ padding: 16px 20px !important; }}
            .price-box {{ padding: 8px 10px !important; }}
            .company-name {{ font-size: 20px !important; }}
        }}
    </style>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f8f9fa; color: #212529;">

    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td align="center" style="padding: 40px 20px;">

                <table role="presentation" style="max-width: 700px; width: 100%; background-color: #ffffff; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-collapse: collapse; border-radius: 8px; overflow: hidden;">

                    <!-- Header -->
                    <tr>
                        <td class="header-padding" style="padding: 20px 24px; background-color: #1e40af; background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); color: #ffffff;">
                            <table role="presentation" style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td style="width: 65%;">
                                        <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 6px; opacity: 0.85; font-weight: 600;">STOCK INTELLIGENCE</div>
                                        <h1 class="company-name" style="margin: 0; font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">{company_name}</h1>
                                        <div style="margin-top: 6px; font-size: 13px; opacity: 0.9; font-weight: 500;">{ticker} • {sector}</div>
                                    </td>
                                    <td align="right" style="vertical-align: top; width: 35%;">
                                        <div class="price-box" style="background-color: rgba(255,255,255,0.15); backdrop-filter: blur(10px); padding: 10px 14px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.2);">
                                            <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; opacity: 0.75;">{current_date}</div>
                                            <div style="font-size: 20px; font-weight: 700; line-height: 1;">{stock_price}</div>
                                            <div style="font-size: 13px; color: {price_change_color}; font-weight: 700; margin-top: 3px;">{price_change_pct}</div>
                                        </div>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Content -->
                    <tr>
                        <td class="content-padding" style="padding: 24px;">

                            <!-- Executive Summary -->
                            {summary_html}

                            <!-- Transition to Sources -->
                            <div style="margin: 32px 0 20px 0; padding: 12px 16px; background-color: #eff6ff; border-left: 4px solid #1e40af; border-radius: 4px;">
                                <p style="margin: 0; font-size: 12px; color: #1e40af; font-weight: 600; line-height: 1.4;">
                                    {analysis_message}
                                </p>
                            </div>

                            <!-- Divider -->
                            <div style="height: 2px; background: linear-gradient(90deg, #1e40af 0%, #e5e7eb 100%); margin-bottom: 20px;"></div>

                            <!-- Source Articles -->
                            <div style="margin-bottom: 0;">
                                <h2 style="margin: 0 0 16px 0; font-size: 14px; font-weight: 700; color: #1e40af; text-transform: uppercase; letter-spacing: 0.5px;">Source Articles</h2>
                                {articles_html}
                            </div>

                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #1e40af; background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); padding: 16px 24px; color: rgba(255,255,255,0.9);">
                            <table role="presentation" style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td>
                                        <div style="font-size: 14px; font-weight: 600; color: #ffffff; margin-bottom: 4px;">QuantBrief</div>
                                        <div style="font-size: 12px; opacity: 0.8; margin-bottom: 8px;">Stock Intelligence Delivered Daily</div>
                                        <div style="font-size: 10px; opacity: 0.7; line-height: 1.4; margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.2);">
                                            This report is for informational purposes only and does not constitute investment advice, a recommendation, or an offer to buy or sell securities. Please consult a financial advisor before making investment decisions.
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

    subject = f"📊 Stock Intelligence: {company_name} ({ticker}) - {analyzed_count} articles analyzed"
    success = send_email(subject, html)

    LOG.info(f"📧 Email #3 (Premium Intelligence): {'✅ SENT' if success else '❌ FAILED'} to {DIGEST_TO}")

    return {
        "status": "sent" if success else "failed",
        "articles_analyzed": analyzed_count,
        "articles_paywalled": paywall_count,
        "articles_total": analyzed_count + paywall_count,
        "ticker": ticker,
        "recipient": DIGEST_TO,
        "email_type": "premium_stock_intelligence"
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
# JOB QUEUE MODELS & INFRASTRUCTURE
# ------------------------------------------------------------------------------

class JobSubmitRequest(BaseModel):
    tickers: List[str]
    minutes: int = 1440
    batch_size: int = 3
    triage_batch_size: int = 3

# ------------------------------------------------------------------------------
# JOB QUEUE SYSTEM - PostgreSQL-Based Background Processing
# ------------------------------------------------------------------------------

# Circuit Breaker for System-Wide Failure Detection
class CircuitBreaker:
    """Detect and halt processing on systematic failures"""
    def __init__(self, failure_threshold=3, reset_timeout=300):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = None
        self.state = 'closed'  # closed = working, open = failing
        self.lock = threading.Lock()

    def record_failure(self, error_type: str, error_msg: str):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                LOG.critical(f"🚨 CIRCUIT BREAKER OPEN: {self.failure_count} consecutive failures")
                LOG.critical(f"   Last error: {error_type}: {error_msg}")
                # TODO: Send alert email in production

    def record_success(self):
        with self.lock:
            # Reset if we've been in open state long enough
            if self.state == 'open' and self.last_failure_time:
                if time.time() - self.last_failure_time > self.reset_timeout:
                    LOG.info("✅ Circuit breaker CLOSED: Resuming after timeout")
                    self.state = 'closed'
                    self.failure_count = 0
            elif self.state == 'closed':
                # Gradual recovery - reduce count on success
                self.failure_count = max(0, self.failure_count - 1)

    def is_open(self):
        with self.lock:
            return self.state == 'open'

    def reset(self):
        with self.lock:
            self.state = 'closed'
            self.failure_count = 0
            self.last_failure_time = None
            LOG.info("🔄 Circuit breaker manually reset")

# Global circuit breaker instance
job_circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=300)

# Job Queue Worker State
_job_worker_running = False
_job_worker_thread = None

# Forward declarations - these reference functions defined later in the file
# We use globals() to avoid circular imports

async def process_ingest_phase(job_id: str, ticker: str, minutes: int, batch_size: int, triage_batch_size: int):
    """Wrapper for ingest logic with error handling and progress tracking"""
    try:
        # Call the actual cron_ingest function which is defined later
        cron_ingest_func = globals().get('cron_ingest')
        if not cron_ingest_func:
            raise RuntimeError("cron_ingest function not yet defined")

        # Create mock request
        class MockRequest:
            def __init__(self):
                self.headers = {"x-admin-token": ADMIN_TOKEN}

        LOG.info(f"[JOB {job_id}] Calling cron_ingest for {ticker}...")

        result = await cron_ingest_func(
            MockRequest(),
            minutes=minutes,
            tickers=[ticker],
            batch_size=batch_size,
            triage_batch_size=triage_batch_size
        )

        LOG.info(f"[JOB {job_id}] cron_ingest completed for {ticker}")

        # Extract and store flagged articles in job config
        if result and isinstance(result, dict):
            phase2 = result.get('phase_2_triage', {})
            flagged_articles = phase2.get('flagged_articles', [])

            if flagged_articles:
                with db() as conn, conn.cursor() as cur:
                    cur.execute("""
                        UPDATE ticker_processing_jobs
                        SET config = jsonb_set(COALESCE(config, '{}'), '{flagged_articles}', %s::jsonb)
                        WHERE job_id = %s
                    """, (json.dumps(flagged_articles), job_id))
                LOG.info(f"[JOB {job_id}] Stored {len(flagged_articles)} flagged article IDs in job config")
            else:
                LOG.warning(f"[JOB {job_id}] No flagged articles found in triage results")

        return result

    except Exception as e:
        LOG.error(f"[JOB {job_id}] INGEST FAILED for {ticker}: {e}")
        LOG.error(f"[JOB {job_id}] Stacktrace: {traceback.format_exc()}")
        raise

async def process_digest_phase(job_id: str, ticker: str, minutes: int, flagged_article_ids: List[int] = None):
    """Wrapper for digest logic with error handling - sends Stock Intelligence Email with executive summary"""
    try:
        # CRITICAL: fetch_digest_articles_with_enhanced_content sends the Stock Intelligence Email
        # which includes the executive summary via generate_ai_final_summaries()
        fetch_digest_func = globals().get('fetch_digest_articles_with_enhanced_content')
        if not fetch_digest_func:
            raise RuntimeError("fetch_digest_articles_with_enhanced_content not yet defined")

        LOG.info(f"[JOB {job_id}] Calling fetch_digest (will send Stock Intelligence Email) for {ticker}...")
        if flagged_article_ids:
            LOG.info(f"[JOB {job_id}] Filtering to {len(flagged_article_ids)} flagged articles from triage")

        result = fetch_digest_func(
            minutes / 60,
            [ticker],
            show_ai_analysis=True,
            show_descriptions=True,
            flagged_article_ids=flagged_article_ids
        )

        LOG.info(f"[JOB {job_id}] fetch_digest completed for {ticker} - Email sent: {result.get('status') == 'sent'}")
        return result

    except Exception as e:
        LOG.error(f"[JOB {job_id}] DIGEST FAILED for {ticker}: {e}")
        LOG.error(f"[JOB {job_id}] Stacktrace: {traceback.format_exc()}")
        raise

async def process_commit_phase(job_id: str, ticker: str, batch_id: str = None, is_last_job: bool = False):
    """Wrapper for commit logic with error handling - includes job_id for idempotency"""
    try:
        # Call the actual GitHub commit function (it's named safe_incremental_commit, not admin_safe_incremental_commit)
        commit_func = globals().get('safe_incremental_commit')
        if not commit_func:
            raise RuntimeError("safe_incremental_commit not yet defined")

        class MockRequest:
            def __init__(self):
                self.headers = {"x-admin-token": ADMIN_TOKEN}

        class CommitBody(BaseModel):
            tickers: List[str]
            job_id: Optional[str] = None  # Pass job_id for idempotency tracking
            skip_render: Optional[bool] = True  # Control [skip render] flag

        LOG.info(f"[JOB {job_id}] Calling GitHub commit for {ticker}...")

        # Skip render for all jobs EXCEPT the last one in batch
        skip_render = not is_last_job
        if is_last_job:
            LOG.info(f"[JOB {job_id}] ⚠️ LAST JOB IN BATCH - Render will deploy after this commit")
        else:
            LOG.info(f"[JOB {job_id}] [skip render] flag enabled - no deployment")

        # Convert job_id to string (PostgreSQL returns UUID objects)
        result = await commit_func(MockRequest(), CommitBody(tickers=[ticker], job_id=str(job_id), skip_render=skip_render))

        LOG.info(f"[JOB {job_id}] GitHub commit completed for {ticker}")
        return result

    except Exception as e:
        LOG.error(f"[JOB {job_id}] COMMIT FAILED for {ticker}: {e}")
        LOG.error(f"[JOB {job_id}] Stacktrace: {traceback.format_exc()}")
        raise

def get_worker_id():
    """Get unique worker ID (Render instance or hostname)"""
    return os.getenv('RENDER_INSTANCE_ID') or os.getenv('HOSTNAME') or 'worker-local'

def update_job_status(job_id: str, status: str = None, phase: str = None, progress: int = None,
                     error_message: str = None, error_stacktrace: str = None, result: dict = None,
                     memory_mb: float = None, duration_seconds: float = None):
    """Update job status in database"""
    updates = ["last_updated = NOW()"]
    params = []

    if status:
        updates.append("status = %s")
        params.append(status)

        if status == 'processing':
            updates.append("started_at = NOW()")
        elif status in ('completed', 'failed', 'timeout', 'cancelled'):
            updates.append("completed_at = NOW()")

    if phase:
        updates.append("phase = %s")
        params.append(phase)

    if progress is not None:
        updates.append("progress = %s")
        params.append(progress)

    if error_message:
        updates.append("error_message = %s")
        params.append(error_message)

    if error_stacktrace:
        updates.append("error_stacktrace = %s")
        params.append(error_stacktrace)

    if result:
        updates.append("result = %s")
        params.append(json.dumps(result))

    if memory_mb is not None:
        updates.append("memory_mb = %s")
        params.append(memory_mb)

    if duration_seconds is not None:
        updates.append("duration_seconds = %s")
        params.append(duration_seconds)

    params.append(job_id)

    with db() as conn, conn.cursor() as cur:
        cur.execute(f"""
            UPDATE ticker_processing_jobs
            SET {', '.join(updates)}
            WHERE job_id = %s
        """, params)

def get_next_queued_job():
    """Get next queued job with atomic claim (prevents race conditions)"""
    with db() as conn, conn.cursor() as cur:
        try:
            cur.execute("""
                UPDATE ticker_processing_jobs
                SET status = 'processing',
                    started_at = NOW(),
                    worker_id = %s,
                    last_updated = NOW()
                WHERE job_id = (
                    SELECT job_id FROM ticker_processing_jobs
                    WHERE status = 'queued'
                    ORDER BY created_at
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING *
            """, (get_worker_id(),))

            job = cur.fetchone()
            if job:
                LOG.info(f"📋 Claimed job {job['job_id']} for ticker {job['ticker']}")
            return dict(job) if job else None

        except Exception as e:
            LOG.error(f"Error claiming job: {e}")
            return None

async def process_ticker_job(job: dict):
    """Process a single ticker job (ingest + digest + commit)"""
    job_id = job['job_id']
    ticker = job['ticker']

    # psycopg returns JSONB as dict, not string
    config = job['config'] if isinstance(job['config'], dict) else {}

    minutes = config.get('minutes', 1440)
    batch_size = config.get('batch_size', 3)
    triage_batch_size = config.get('triage_batch_size', 3)

    start_time = time.time()
    memory_start = memory_monitor.get_current_mb() if hasattr(memory_monitor, 'get_current_mb') else 0

    LOG.info(f"🚀 [JOB {job_id}] Starting processing for {ticker}")
    LOG.info(f"   Config: minutes={minutes}, batch={batch_size}, triage_batch={triage_batch_size}")

    try:
        # NOTE: TICKER_PROCESSING_LOCK is acquired inside cron_ingest/cron_digest
        # We don't acquire it here to avoid deadlock (lock is not reentrant)

        # Check if job was cancelled before starting
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT status FROM ticker_processing_jobs WHERE job_id = %s", (job_id,))
            current_status = cur.fetchone()
            if current_status and current_status['status'] == 'cancelled':
                LOG.warning(f"🚫 [JOB {job_id}] Job cancelled before starting, exiting")
                return

        # PHASE 1: Ingest (already implemented in /cron/ingest)
        update_job_status(job_id, phase='ingest_start', progress=10)
        LOG.info(f"📥 [JOB {job_id}] Phase 1: Ingest starting...")

        # Call ingest logic (will be defined later in file)
        # We can't import it here due to circular dependency
        # So we'll call it by name after it's defined
        ingest_result = await process_ingest_phase(
            job_id=job_id,
            ticker=ticker,
            minutes=minutes,
            batch_size=batch_size,
            triage_batch_size=triage_batch_size
        )

        update_job_status(job_id, phase='ingest_complete', progress=60)

        # Log detailed ingest stats
        if ingest_result:
            LOG.info(f"✅ [JOB {job_id}] Phase 1: Ingest complete")
            if isinstance(ingest_result, dict):
                phase1 = ingest_result.get('phase_1_ingest', {})
                phase2 = ingest_result.get('phase_2_triage', {})
                phase4 = ingest_result.get('phase_4_async_batch_scraping', {})

                if phase1:
                    LOG.info(f"   Articles: New({phase1.get('total_inserted', 0)}) Total({phase1.get('total_articles_in_timeframe', 0)})")

                if phase2 and phase2.get('selections_by_ticker'):
                    sel = phase2['selections_by_ticker'].get(ticker, {})
                    LOG.info(f"   Triage: Company({sel.get('company', 0)}) Industry({sel.get('industry', 0)}) Competitor({sel.get('competitor', 0)})")

                if phase4:
                    LOG.info(f"   Scraping: New({phase4.get('scraped', 0)}) Reused({phase4.get('reused_existing', 0)}) Success({phase4.get('overall_success_rate', 'N/A')})")
        else:
            LOG.info(f"✅ [JOB {job_id}] Phase 1: Ingest complete (no detailed stats)")

        # Check if cancelled after Phase 1
        # Also re-fetch config to get flagged_articles that were stored during ingest
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT status, config FROM ticker_processing_jobs WHERE job_id = %s", (job_id,))
            job_status = cur.fetchone()
            if job_status and job_status['status'] == 'cancelled':
                LOG.warning(f"🚫 [JOB {job_id}] Job cancelled after Phase 1, exiting")
                return

            # Re-fetch flagged_articles that were stored during ingest phase
            updated_config = job_status['config'] if job_status and isinstance(job_status['config'], dict) else {}
            flagged_article_ids = updated_config.get('flagged_articles', [])

            if flagged_article_ids:
                LOG.info(f"📋 [JOB {job_id}] Retrieved {len(flagged_article_ids)} flagged article IDs from ingest phase")
            else:
                LOG.warning(f"⚠️ [JOB {job_id}] No flagged articles found in config after ingest")

        # PHASE 2: Digest (already implemented in /cron/digest)
        update_job_status(job_id, phase='digest_start', progress=65)
        LOG.info(f"📧 [JOB {job_id}] Phase 2: Digest starting...")

        # Call digest function (defined later in file) - pass flagged articles from triage
        digest_result = await process_digest_phase(
            job_id=job_id,
            ticker=ticker,
            minutes=minutes,
            flagged_article_ids=flagged_article_ids
        )

        update_job_status(job_id, phase='digest_complete', progress=95)

        # Log detailed digest stats
        if digest_result:
            LOG.info(f"✅ [JOB {job_id}] Phase 2: Digest complete")
            if isinstance(digest_result, dict):
                LOG.info(f"   Status: {digest_result.get('status', 'unknown')}")
                LOG.info(f"   Articles: {digest_result.get('articles', 0)}")
        else:
            LOG.info(f"✅ [JOB {job_id}] Phase 2: Digest complete (no detailed stats)")

        # Check if cancelled after Phase 2
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT status FROM ticker_processing_jobs WHERE job_id = %s", (job_id,))
            current_status = cur.fetchone()
            if current_status and current_status['status'] == 'cancelled':
                LOG.warning(f"🚫 [JOB {job_id}] Job cancelled after Phase 2, exiting")
                return

        # EMAIL #3: USER INTELLIGENCE REPORT (no AI analysis, no descriptions)
        update_job_status(job_id, phase='user_report', progress=97)
        LOG.info(f"📧 [JOB {job_id}] Sending Email #3: User Intelligence Report...")

        try:
            user_report_result = send_user_intelligence_report(
                hours=int(minutes/60),
                tickers=[ticker],
                flagged_article_ids=flagged_article_ids  # Filter to same articles as Email #2
            )
            if user_report_result:
                LOG.info(f"✅ [JOB {job_id}] Email #3 sent successfully")
                if isinstance(user_report_result, dict):
                    LOG.info(f"   Status: {user_report_result.get('status', 'unknown')}")
            else:
                LOG.warning(f"⚠️ [JOB {job_id}] Email #3 returned no result")
        except Exception as e:
            LOG.error(f"❌ [JOB {job_id}] Email #3 failed: {e}")
            # Continue to GitHub commit even if Email #3 fails (Option A)

        # Check if cancelled after Phase 3
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT status FROM ticker_processing_jobs WHERE job_id = %s", (job_id,))
            current_status = cur.fetchone()
            if current_status and current_status['status'] == 'cancelled':
                LOG.warning(f"🚫 [JOB {job_id}] Job cancelled after Email #3, exiting")
                return

        # COMMIT METADATA TO GITHUB after all emails sent
        # This ensures GitHub commit doesn't trigger server restart before emails are sent
        update_job_status(job_id, phase='commit_metadata', progress=99)
        LOG.info(f"💾 [JOB {job_id}] Committing AI metadata to GitHub after final email...")

        try:
            # Check if this is the last job in the batch (to control [skip render] flag)
            batch_id = job.get('batch_id')
            is_last_job = False

            if batch_id:
                with db() as conn, conn.cursor() as cur:
                    # Count remaining jobs in batch (queued + processing, excluding this one)
                    cur.execute("""
                        SELECT COUNT(*) as remaining
                        FROM ticker_processing_jobs
                        WHERE batch_id = %s
                        AND status IN ('queued', 'processing')
                        AND job_id != %s
                    """, (batch_id, job_id))
                    result = cur.fetchone()
                    remaining_jobs = result['remaining'] if result else 0

                    if remaining_jobs == 0:
                        is_last_job = True
                        LOG.info(f"[JOB {job_id}] 🎯 This is the LAST job in batch {batch_id}")

            await process_commit_phase(
                job_id=job_id,
                ticker=ticker,
                batch_id=batch_id,
                is_last_job=is_last_job
            )
            LOG.info(f"✅ [JOB {job_id}] Metadata committed to GitHub successfully")
        except Exception as e:
            LOG.error(f"⚠️ [JOB {job_id}] GitHub commit failed (non-fatal): {e}")
            # Don't fail the job if commit fails - continue processing

        # PHASE 3: Complete
        update_job_status(job_id, phase='finalizing', progress=99)
        LOG.info(f"✅ [JOB {job_id}] Finalizing job...")

        # Calculate final metrics
        duration = time.time() - start_time
        memory_end = memory_monitor.get_current_mb() if hasattr(memory_monitor, 'get_current_mb') else 0
        memory_used = max(0, memory_end - memory_start)

        # Mark complete
        result = {
            "ticker": ticker,
            "ingest": ingest_result,
            "digest": digest_result,
            "metadata_committed": True,  # Committed after Phase 1
            "duration_seconds": duration,
            "memory_mb": memory_used
        }

        update_job_status(
            job_id,
            status='completed',
            phase='complete',
            progress=100,
            result=result,
            duration_seconds=duration,
            memory_mb=memory_used
        )

        LOG.info(f"✅ [JOB {job_id}] COMPLETED in {duration:.1f}s (memory: {memory_used:.1f}MB)")

        # Record success with circuit breaker
        job_circuit_breaker.record_success()

        # Update batch counters
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE ticker_processing_batches
                SET completed_jobs = completed_jobs + 1
                WHERE batch_id = %s
            """, (job['batch_id'],))

        return result

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        duration = time.time() - start_time

        LOG.error(f"❌ [JOB {job_id}] FAILED after {duration:.1f}s: {error_msg}")
        LOG.error(f"   Stacktrace: {error_trace}")

        # Determine if this is a system-wide failure
        is_system_failure = any(keyword in error_msg.lower() for keyword in [
            'database', 'connection', 'psycopg', 'timeout', 'memory'
        ])

        if is_system_failure:
            job_circuit_breaker.record_failure(type(e).__name__, error_msg)
        else:
            # Ticker-specific failure, not system-wide
            job_circuit_breaker.record_success()

        update_job_status(
            job_id,
            status='failed',
            error_message=error_msg[:1000],  # Limit size
            error_stacktrace=error_trace[:5000],
            duration_seconds=duration
        )

        # Update batch counters
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE ticker_processing_batches
                SET failed_jobs = failed_jobs + 1
                WHERE batch_id = %s
            """, (job['batch_id'],))

        raise

def job_worker_loop():
    """Background worker that polls database for jobs"""
    global _job_worker_running

    LOG.info(f"🔧 Job worker started (worker_id: {get_worker_id()})")

    while _job_worker_running:
        try:
            # Check circuit breaker
            if job_circuit_breaker.is_open():
                LOG.warning("⚠️ Circuit breaker is OPEN, skipping job polling")
                time.sleep(30)
                continue

            # Get next job
            job = get_next_queued_job()

            if job:
                # Process job (blocks until complete)
                asyncio.run(process_ticker_job(job))
            else:
                # No jobs available, sleep and poll again
                time.sleep(10)

        except KeyboardInterrupt:
            LOG.info("🛑 Job worker received interrupt signal")
            break

        except Exception as e:
            LOG.error(f"💥 Job worker error: {e}")
            LOG.error(traceback.format_exc())
            time.sleep(30)  # Back off on errors

    LOG.info("🔚 Job worker stopped")

def start_job_worker():
    """Start the background job worker thread"""
    global _job_worker_running, _job_worker_thread

    if _job_worker_running:
        LOG.warning("Job worker already running")
        return

    _job_worker_running = True
    _job_worker_thread = threading.Thread(target=job_worker_loop, daemon=True, name="JobWorker")
    _job_worker_thread.start()

    LOG.info("✅ Job worker thread started")

def stop_job_worker():
    """Stop the background job worker thread"""
    global _job_worker_running

    if not _job_worker_running:
        return

    _job_worker_running = False
    LOG.info("⏸️ Job worker stopping...")

    if _job_worker_thread:
        _job_worker_thread.join(timeout=10)
        LOG.info("✅ Job worker stopped")

def timeout_watchdog_loop():
    """Monitor for timed-out jobs"""
    LOG.info("⏰ Timeout watchdog started")

    while _job_worker_running:
        try:
            time.sleep(60)  # Check every minute

            with db() as conn, conn.cursor() as cur:
                # Find jobs that exceeded timeout
                cur.execute("""
                    UPDATE ticker_processing_jobs
                    SET status = 'timeout',
                        error_message = 'Job exceeded timeout limit',
                        completed_at = NOW()
                    WHERE status = 'processing'
                    AND timeout_at < NOW()
                    RETURNING job_id, ticker, worker_id
                """)

                timed_out = cur.fetchall()
                for job in timed_out:
                    LOG.error(f"⏰ JOB TIMEOUT: {job['job_id']} (ticker: {job['ticker']}, worker: {job['worker_id']})")

                    # Update batch counters
                    cur.execute("""
                        UPDATE ticker_processing_batches
                        SET failed_jobs = failed_jobs + 1
                        WHERE batch_id = (
                            SELECT batch_id FROM ticker_processing_jobs WHERE job_id = %s
                        )
                    """, (job['job_id'],))

        except Exception as e:
            LOG.error(f"Timeout watchdog error: {e}")
            time.sleep(30)

    LOG.info("⏰ Timeout watchdog stopped")

# Start workers on app startup
@APP.on_event("startup")
async def startup_event():
    """Initialize job queue system on startup"""
    worker_id = get_worker_id()
    LOG.info("=" * 80)
    LOG.info(f"🚀 FastAPI STARTUP EVENT - Worker: {worker_id}")
    LOG.info(f"   Python: {sys.version}")
    LOG.info(f"   Platform: {sys.platform}")
    LOG.info(f"   Environment: Render.com" if os.getenv('RENDER') else "   Environment: Local")
    LOG.info(f"   Port: {os.getenv('PORT', '10000')}")
    LOG.info(f"   Memory: {memory_monitor.get_current_mb() if hasattr(memory_monitor, 'get_current_mb') else 'N/A'} MB")
    LOG.info("=" * 80)
    LOG.info("🔧 Initializing job queue system...")

    # Reclaim orphaned jobs from previous worker instance (handles Render restarts)
    try:
        with db() as conn, conn.cursor() as cur:
            # First, log ALL processing jobs to understand restart impact
            cur.execute("""
                SELECT job_id, ticker, phase, progress, worker_id,
                       EXTRACT(EPOCH FROM (NOW() - started_at)) / 60 AS minutes_running
                FROM ticker_processing_jobs
                WHERE status = 'processing'
                ORDER BY started_at
            """)
            all_processing = cur.fetchall()

            if all_processing:
                LOG.warning(f"⚠️ STARTUP: Found {len(all_processing)} jobs in 'processing' state:")
                for job in all_processing:
                    LOG.info(f"   → {job['ticker']} ({job['phase']}, {job['progress']}%, {job['minutes_running']:.1f}min, worker: {job['worker_id']})")

            # Reclaim jobs that were processing but worker died (older than 5 minutes = definitely orphaned)
            cur.execute("""
                UPDATE ticker_processing_jobs
                SET status = 'queued',
                    started_at = NULL,
                    worker_id = NULL,
                    phase = 'restart_recovery',
                    progress = 0,
                    last_updated = NOW(),
                    error_message = COALESCE(error_message, '') || ' | Server restart detected, job reclaimed'
                WHERE status = 'processing'
                AND started_at < NOW() - INTERVAL '5 minutes'
                RETURNING job_id, ticker, phase AS old_phase, progress AS old_progress
            """)

            orphaned = cur.fetchall()
            if orphaned:
                LOG.warning(f"🔄 RECLAIMED {len(orphaned)} orphaned jobs (>5min old, server likely restarted):")
                for job in orphaned:
                    LOG.info(f"   → {job['ticker']} was at {job['old_phase']} ({job['old_progress']}%), now queued for retry")

            # Also check for jobs processing <5 minutes (possible crash mid-job)
            cur.execute("""
                SELECT COUNT(*) as recent_count
                FROM ticker_processing_jobs
                WHERE status = 'processing'
                AND started_at >= NOW() - INTERVAL '5 minutes'
            """)
            recent_result = cur.fetchone()
            if recent_result and recent_result['recent_count'] > 0:
                LOG.warning(f"⚠️ {recent_result['recent_count']} jobs started <5min ago still marked 'processing'")
                LOG.warning("   These will NOT be reclaimed yet (might still be running on old worker)")
                LOG.warning("   Timeout watchdog will mark them as 'timeout' if they exceed 45 minutes")

            if not orphaned and not all_processing:
                LOG.info("✅ No orphaned jobs found - clean startup")

    except Exception as e:
        LOG.error(f"❌ Failed to reclaim orphaned jobs: {e}")
        LOG.error(f"   Stacktrace: {traceback.format_exc()}")

    start_job_worker()

    # Start timeout watchdog in separate thread
    timeout_thread = threading.Thread(target=timeout_watchdog_loop, daemon=True, name="TimeoutWatchdog")
    timeout_thread.start()

    LOG.info("✅ Job queue system initialized")

@APP.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    worker_id = get_worker_id()
    LOG.info("=" * 80)
    LOG.info(f"🛑 FastAPI SHUTDOWN EVENT - Worker: {worker_id}")
    LOG.info(f"   Reason: Unknown (check Render logs)")
    LOG.info(f"   Memory: {memory_monitor.get_current_mb() if hasattr(memory_monitor, 'get_current_mb') else 'N/A'} MB")

    # Log any jobs currently processing (will be orphaned after shutdown)
    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT job_id, ticker, phase, progress,
                       EXTRACT(EPOCH FROM (NOW() - started_at)) / 60 AS minutes_running
                FROM ticker_processing_jobs
                WHERE status = 'processing'
                AND worker_id = %s
            """, (worker_id,))
            active_jobs = cur.fetchall()

            if active_jobs:
                LOG.warning(f"⚠️ SHUTDOWN: {len(active_jobs)} jobs were still processing:")
                for job in active_jobs:
                    LOG.warning(f"   → {job['ticker']} ({job['phase']}, {job['progress']}%, {job['minutes_running']:.1f}min)")
                LOG.warning("   These jobs will be reclaimed on next startup (>5min threshold)")
            else:
                LOG.info("✅ No active jobs at shutdown")
    except Exception as e:
        LOG.error(f"Failed to check active jobs during shutdown: {e}")

    LOG.info("=" * 80)
    stop_job_worker()

# ------------------------------------------------------------------------------
# Pydantic Request Models
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
    triage_batch_size: int = 2

# ------------------------------------------------------------------------------
# API Routes
# ------------------------------------------------------------------------------
@APP.get("/")
def root():
    return {"status": "ok", "service": "Quantbrief Stock News Aggregator"}

@APP.get("/health")
def health_check():
    """
    Health check endpoint for Render and monitoring services.

    Returns worker status and last activity to prevent idle timeout.
    Render pings this endpoint to verify the service is alive.
    """
    global _job_worker_running

    # Check worker thread is alive
    worker_alive = _job_worker_running and (_job_worker_thread is not None and _job_worker_thread.is_alive())

    # Check database connectivity
    db_healthy = False
    db_error = None
    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            db_healthy = True
    except Exception as e:
        db_error = str(e)

    # Check for active jobs
    active_jobs = 0
    recent_completions = 0
    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM ticker_processing_jobs WHERE status = 'processing'")
            active_jobs = cur.fetchone()['count']

            cur.execute("""
                SELECT COUNT(*) as count FROM ticker_processing_jobs
                WHERE status = 'completed' AND completed_at > NOW() - INTERVAL '10 minutes'
            """)
            recent_completions = cur.fetchone()['count']
    except:
        pass

    # Overall health status
    is_healthy = worker_alive and db_healthy
    status_code = 200 if is_healthy else 503

    # Get memory usage
    memory_mb = memory_monitor.get_current_mb() if hasattr(memory_monitor, 'get_current_mb') else None

    response = {
        "status": "healthy" if is_healthy else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "worker": {
            "running": _job_worker_running,
            "thread_alive": worker_alive,
            "worker_id": get_worker_id()
        },
        "database": {
            "connected": db_healthy,
            "error": db_error
        },
        "jobs": {
            "active": active_jobs,
            "recent_completions_10min": recent_completions
        },
        "circuit_breaker": {
            "state": job_circuit_breaker.state,
            "failure_count": job_circuit_breaker.failure_count
        },
        "system": {
            "memory_mb": memory_mb,
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "render_instance": os.getenv('RENDER_INSTANCE_ID', 'not_render')
        }
    }

    return JSONResponse(content=response, status_code=status_code)

@APP.get("/debug/auth")
def debug_auth(request: Request):
    """Debug endpoint to check authentication headers"""
    return {
        "x-admin-token": request.headers.get("x-admin-token"),
        "authorization": request.headers.get("authorization"),
        "expected_token_prefix": ADMIN_TOKEN[:10] + "..." if ADMIN_TOKEN else "None",
        "token_length": len(ADMIN_TOKEN) if ADMIN_TOKEN else 0
    }

# ------------------------------------------------------------------------------
# JOB QUEUE API ENDPOINTS
# ------------------------------------------------------------------------------

@APP.post("/jobs/submit")
async def submit_job_batch(request: Request, body: JobSubmitRequest):
    """Submit a batch of tickers for server-side processing"""
    require_admin(request)

    if not body.tickers:
        raise HTTPException(status_code=400, detail="No tickers specified")

    # Check queue capacity
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) as queued_count
            FROM ticker_processing_jobs
            WHERE status IN ('queued', 'processing')
        """)

        queued_count = cur.fetchone()['queued_count']

        if queued_count > 100:
            raise HTTPException(
                status_code=429,
                detail=f"Job queue is full ({queued_count} jobs pending). Try again later."
            )

    # Create batch
    batch_id = None
    job_ids = []

    with db() as conn, conn.cursor() as cur:
        # Create batch record
        cur.execute("""
            INSERT INTO ticker_processing_batches (total_jobs, created_by, config)
            VALUES (%s, %s, %s)
            RETURNING batch_id
        """, (len(body.tickers), 'powershell', json.dumps({
            "minutes": body.minutes,
            "batch_size": body.batch_size,
            "triage_batch_size": body.triage_batch_size
        })))

        batch_id = cur.fetchone()['batch_id']

        # Create individual jobs with timeout
        timeout_minutes = 45  # 45 minutes per ticker
        timeout_at = datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)

        for ticker in body.tickers:
            cur.execute("""
                INSERT INTO ticker_processing_jobs (
                    batch_id, ticker, config, timeout_at
                )
                VALUES (%s, %s, %s, %s)
                RETURNING job_id
            """, (batch_id, ticker, json.dumps({
                "minutes": body.minutes,
                "batch_size": body.batch_size,
                "triage_batch_size": body.triage_batch_size
            }), timeout_at))

            job_ids.append(str(cur.fetchone()['job_id']))

    LOG.info(f"📦 Batch {batch_id} created: {len(body.tickers)} tickers submitted")

    return {
        "status": "submitted",
        "batch_id": str(batch_id),
        "job_ids": job_ids,
        "tickers": body.tickers,
        "total_jobs": len(body.tickers),
        "message": f"Processing started server-side for {len(body.tickers)} tickers"
    }

@APP.get("/jobs/batch/{batch_id}")
async def get_batch_status(request: Request, batch_id: str):
    """Get status of all jobs in a batch"""
    require_admin(request)

    with db() as conn, conn.cursor() as cur:
        # Get batch info
        cur.execute("""
            SELECT * FROM ticker_processing_batches WHERE batch_id = %s
        """, (batch_id,))

        batch = cur.fetchone()
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")

        # Get all jobs in batch
        cur.execute("""
            SELECT job_id, ticker, status, phase, progress,
                   error_message, started_at, completed_at,
                   duration_seconds, memory_mb
            FROM ticker_processing_jobs
            WHERE batch_id = %s
            ORDER BY created_at
        """, (batch_id,))

        jobs = [dict(row) for row in cur.fetchall()]

    # Calculate overall progress
    total_progress = sum(j['progress'] for j in jobs)
    avg_progress = total_progress // len(jobs) if jobs else 0

    completed = len([j for j in jobs if j['status'] == 'completed'])
    failed = len([j for j in jobs if j['status'] in ('failed', 'timeout')])
    processing = len([j for j in jobs if j['status'] == 'processing'])
    queued = len([j for j in jobs if j['status'] == 'queued'])

    # Determine batch status
    batch_status = batch['status']
    if completed + failed == len(jobs):
        batch_status = 'completed'
    elif processing > 0 or completed > 0:
        batch_status = 'processing'
    else:
        batch_status = 'queued'

    # Update batch status if changed
    if batch_status != batch['status']:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE ticker_processing_batches
                SET status = %s,
                    completed_jobs = %s,
                    failed_jobs = %s
                WHERE batch_id = %s
            """, (batch_status, completed, failed, batch_id))

    return {
        "batch_id": batch_id,
        "status": batch_status,
        "total_tickers": len(jobs),
        "completed": completed,
        "failed": failed,
        "processing": processing,
        "queued": queued,
        "overall_progress": avg_progress,
        "created_at": batch['created_at'].isoformat() if batch['created_at'] else None,
        "jobs": [{
            "job_id": str(j['job_id']),
            "ticker": j['ticker'],
            "status": j['status'],
            "phase": j['phase'],
            "progress": j['progress'],
            "error_message": j['error_message'],
            "duration_seconds": j['duration_seconds'],
            "memory_mb": j['memory_mb']
        } for j in jobs]
    }

@APP.get("/jobs/active-batches")
async def get_active_batches(request: Request):
    """Get all active batches with their job details"""
    require_admin(request)

    try:
        with db() as conn, conn.cursor() as cur:
            # First get batches with active jobs
            cur.execute("""
                SELECT
                    b.batch_id,
                    b.status as batch_status,
                    b.created_at,
                    b.total_jobs,
                    b.completed_jobs,
                    b.failed_jobs
                FROM ticker_processing_batches b
                WHERE b.created_at > NOW() - INTERVAL '24 hours'
                  AND EXISTS (
                      SELECT 1 FROM ticker_processing_jobs j
                      WHERE j.batch_id = b.batch_id
                        AND j.status IN ('queued', 'processing')
                  )
                ORDER BY b.created_at DESC
            """)
            batches = cur.fetchall()

            result = []
            for batch in batches:
                # Get jobs for this batch
                cur.execute("""
                    SELECT job_id, ticker, status, phase, progress
                    FROM ticker_processing_jobs
                    WHERE batch_id = %s
                    ORDER BY created_at
                """, (batch['batch_id'],))
                jobs = cur.fetchall()

                result.append({
                    "batch_id": str(batch['batch_id']),
                    "batch_status": batch['batch_status'],
                    "created_at": batch['created_at'].isoformat() if batch['created_at'] else None,
                    "total_jobs": batch['total_jobs'],
                    "completed_jobs": batch['completed_jobs'],
                    "failed_jobs": batch['failed_jobs'],
                    "jobs": [{"job_id": str(j['job_id']), "ticker": j['ticker'], "status": j['status'], "phase": j['phase'], "progress": j['progress']} for j in jobs]
                })

            return {
                "active_batches": len(result),
                "batches": result
            }
    except Exception as e:
        LOG.error(f"Error in /jobs/active-batches: {e}")
        LOG.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@APP.get("/jobs/stats")
async def get_job_stats(request: Request):
    """Get overall job queue statistics"""
    require_admin(request)

    with db() as conn, conn.cursor() as cur:
        # Job counts by status
        cur.execute("""
            SELECT status, COUNT(*) as count
            FROM ticker_processing_jobs
            GROUP BY status
        """)
        status_counts = {row['status']: row['count'] for row in cur.fetchall()}

        # Recent completions (last hour)
        cur.execute("""
            SELECT COUNT(*) as count,
                   AVG(duration_seconds) as avg_duration,
                   AVG(memory_mb) as avg_memory
            FROM ticker_processing_jobs
            WHERE status = 'completed'
            AND completed_at > NOW() - INTERVAL '1 hour'
        """)
        recent = cur.fetchone()

        # Active batches
        cur.execute("""
            SELECT COUNT(*) as count
            FROM ticker_processing_batches
            WHERE status IN ('queued', 'processing')
        """)
        active_batches = cur.fetchone()['count']

    return {
        "status_counts": status_counts,
        "recent_completions_1h": recent['count'] or 0,
        "avg_duration_seconds": float(recent['avg_duration']) if recent['avg_duration'] else None,
        "avg_memory_mb": float(recent['avg_memory']) if recent['avg_memory'] else None,
        "active_batches": active_batches,
        "circuit_breaker_state": job_circuit_breaker.state,
        "circuit_breaker_failures": job_circuit_breaker.failure_count,
        "worker_id": get_worker_id()
    }

@APP.get("/jobs/{job_id}")
async def get_job_detail(request: Request, job_id: str):
    """Get detailed status of a single job"""
    require_admin(request)

    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT j.*, b.batch_id
            FROM ticker_processing_jobs j
            JOIN ticker_processing_batches b ON j.batch_id = b.batch_id
            WHERE j.job_id = %s
        """, (job_id,))

        job = cur.fetchone()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "job_id": str(job['job_id']),
            "batch_id": str(job['batch_id']),
            "ticker": job['ticker'],
            "status": job['status'],
            "phase": job['phase'],
            "progress": job['progress'],
            "result": job['result'],
            "error_message": job['error_message'],
            "error_stacktrace": job['error_stacktrace'],
            "retry_count": job['retry_count'],
            "worker_id": job['worker_id'],
            "memory_mb": job['memory_mb'],
            "duration_seconds": job['duration_seconds'],
            "created_at": job['created_at'].isoformat() if job['created_at'] else None,
            "started_at": job['started_at'].isoformat() if job['started_at'] else None,
            "completed_at": job['completed_at'].isoformat() if job['completed_at'] else None,
            "config": job['config']
        }

@APP.post("/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str):
    """Cancel a specific job"""
    require_admin(request)

    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            UPDATE ticker_processing_jobs
            SET status = 'cancelled',
                error_message = 'Cancelled by user',
                completed_at = NOW(),
                last_updated = NOW()
            WHERE job_id = %s
            AND status IN ('queued', 'processing')
            RETURNING ticker, status, phase
        """, (job_id,))

        result = cur.fetchone()
        if result:
            LOG.warning(f"🚫 Job {job_id} cancelled by user (ticker: {result['ticker']}, was in: {result['phase']})")

            # Update batch counters
            cur.execute("""
                UPDATE ticker_processing_batches
                SET failed_jobs = failed_jobs + 1
                WHERE batch_id = (
                    SELECT batch_id FROM ticker_processing_jobs WHERE job_id = %s
                )
            """, (job_id,))

            return {
                "status": "cancelled",
                "job_id": job_id,
                "ticker": result['ticker'],
                "was_in_phase": result['phase']
            }
        else:
            # Check if job exists but already completed/failed
            cur.execute("""
                SELECT ticker, status FROM ticker_processing_jobs WHERE job_id = %s
            """, (job_id,))

            existing = cur.fetchone()
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Job already {existing['status']}, cannot cancel"
                )
            else:
                raise HTTPException(status_code=404, detail="Job not found")

@APP.post("/jobs/batch/{batch_id}/cancel")
async def cancel_batch(request: Request, batch_id: str):
    """Cancel all jobs in a batch"""
    require_admin(request)

    try:
        with db() as conn, conn.cursor() as cur:
            # Cancel all queued/processing jobs in the batch
            cur.execute("""
                UPDATE ticker_processing_jobs
                SET status = 'cancelled',
                    error_message = 'Batch cancelled by user',
                    completed_at = NOW(),
                    last_updated = NOW()
                WHERE batch_id = %s
                AND status IN ('queued', 'processing')
                RETURNING job_id, ticker
            """, (batch_id,))

            cancelled_jobs = cur.fetchall()

            if cancelled_jobs:
                LOG.warning(f"🚫 Batch {batch_id} cancelled by user ({len(cancelled_jobs)} jobs)")
                for job in cancelled_jobs:
                    LOG.info(f"   Cancelled: {job['ticker']} (job_id: {job['job_id']})")

                # Update batch status
                cur.execute("""
                    UPDATE ticker_processing_batches
                    SET status = 'cancelled',
                        failed_jobs = failed_jobs + %s
                    WHERE batch_id = %s
                """, (len(cancelled_jobs), batch_id))

                return {
                    "status": "cancelled",
                    "batch_id": batch_id,
                    "jobs_cancelled": len(cancelled_jobs),
                    "tickers": [j['ticker'] for j in cancelled_jobs]
                }
            else:
                # Check if batch exists
                cur.execute("""
                    SELECT status, total_jobs FROM ticker_processing_batches WHERE batch_id = %s
                """, (batch_id,))

                batch = cur.fetchone()
                if batch:
                    return {
                        "status": "no_jobs_to_cancel",
                        "message": f"Batch is already {batch['status']}, no jobs to cancel",
                        "batch_id": batch_id
                    }
                else:
                    raise HTTPException(status_code=404, detail="Batch not found")
    except Exception as e:
        LOG.error(f"Error cancelling batch {batch_id}: {e}")
        LOG.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@APP.post("/jobs/circuit-breaker/reset")
async def reset_circuit_breaker(request: Request):
    """Manually reset the circuit breaker"""
    require_admin(request)

    job_circuit_breaker.reset()

    return {
        "status": "reset",
        "message": "Circuit breaker has been manually reset"
    }

# ------------------------------------------------------------------------------
# ADMIN ENDPOINTS (Existing)
# ------------------------------------------------------------------------------

@APP.post("/admin/migrate-feeds")
async def admin_migrate_feeds(request: Request):
    """NEW ARCHITECTURE V2: Verify feeds + ticker_feeds architecture is ready"""
    require_admin(request)

    try:
        # Step 1: Ensure schema is up to date (includes feeds + ticker_feeds tables)
        ensure_schema()

        # Step 2: Verify new architecture exists
        with db() as conn, conn.cursor() as cur:
            # Verify new architecture exists
            cur.execute("SELECT COUNT(*) as feed_count FROM feeds")
            new_feed_count = cur.fetchone()['feed_count']

            cur.execute("SELECT COUNT(*) as association_count FROM ticker_feeds")
            association_count = cur.fetchone()['association_count']

            LOG.info(f"✅ NEW ARCHITECTURE V2 verified: {new_feed_count} feeds, {association_count} associations")

            return {
                "status": "success",
                "message": "NEW ARCHITECTURE V2 (category-per-relationship) is active",
                "feeds": new_feed_count,
                "associations": association_count
            }

    except Exception as e:
        LOG.error(f"❌ Migration failed: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/admin/fix-foreign-key")
async def admin_fix_foreign_key(request: Request):
    """Fix the found_url foreign key constraint to point to feeds table (DEPRECATED)"""
    require_admin(request)

    try:
        # This endpoint is deprecated - foreign keys are handled in ensure_schema()

        return {
            "status": "success",
            "message": "Foreign key constraint fixed successfully"
        }

    except Exception as e:
        LOG.error(f"❌ Foreign key fix failed: {e}")
        return {"status": "error", "message": str(e)}


@APP.post("/admin/init")
async def admin_init(request: Request, body: InitRequest):
    async with TICKER_PROCESSING_LOCK:
        """Initialize feeds for specified tickers using NEW ARCHITECTURE"""
        require_admin(request)

        # Global state to track if initialization already happened in this session
        global _schema_initialized, _github_synced
        if '_schema_initialized' not in globals():
            _schema_initialized = False
        if '_github_synced' not in globals():
            _github_synced = False

        # NEW ARCHITECTURE V2: Use functions directly from app.py (no import needed)

        LOG.info("=== INITIALIZATION STARTING ===")

        # CRITICAL: Clear any global state that might contaminate between tickers
        LOG.info("=== CLEARING GLOBAL STATE FOR FRESH TICKER PROCESSING ===")
        import gc
        gc.collect()  # Force garbage collection to clear any lingering objects

        # STEP 1: Ensure database schema is up to date (ONCE per session)
        if not _schema_initialized:
            LOG.info("=== ENSURING DATABASE SCHEMA (NEW FEED ARCHITECTURE) ===")
            ensure_schema()
            _schema_initialized = True
        else:
            LOG.info("=== SCHEMA ALREADY INITIALIZED - SKIPPING ===")

        # STEP 2: Import CSV from GitHub (ONCE per session)
        if not _github_synced:
            LOG.info("=== INITIALIZATION: Syncing ticker reference from GitHub ===")
            github_sync_result = sync_ticker_references_from_github()
            _github_synced = True
        else:
            LOG.info("=== GITHUB SYNC ALREADY COMPLETED - SKIPPING ===")
            github_sync_result = {"status": "skipped", "message": "Already synced in this session"}

        if github_sync_result["status"] != "success":
            LOG.warning(f"GitHub sync failed: {github_sync_result.get('message', 'Unknown error')}")
        else:
            LOG.info(f"GitHub sync successful: {github_sync_result.get('message', 'Completed')}")

        results = []

        # CRITICAL: Process each ticker in complete isolation using NEW ARCHITECTURE
        for ticker in body.tickers:
            # STEP 1: Create isolated ticker variable to prevent corruption
            isolated_ticker = str(ticker).strip()  # Force new string object

            # CRITICAL: Clear any residual state between ticker processing
            LOG.info(f"=== PROCESSING {isolated_ticker} - NEW ARCHITECTURE ===")
            import gc
            gc.collect()  # Force garbage collection between each ticker

            LOG.info(f"=== INITIALIZING TICKER: {isolated_ticker} ===")

            try:
                # STEP 2: Get or generate metadata with enhanced ticker reference integration
                # CRITICAL: Force refresh to ensure no cached contamination from previous tickers
                metadata = get_or_create_enhanced_ticker_metadata(isolated_ticker, force_refresh=True)

                # CRITICAL DEBUG: Log what metadata was actually returned
                LOG.info(f"[NEW_ARCH_DEBUG] {isolated_ticker} metadata returned: {metadata}")
                LOG.info(f"[NEW_ARCH_DEBUG] {isolated_ticker} company_name in metadata: '{metadata.get('company_name', 'MISSING')}'")

                # STEP 3: Create feeds using NEW MANY-TO-MANY ARCHITECTURE
                feeds_created = create_feeds_for_ticker_new_architecture(isolated_ticker, metadata)

                if not feeds_created:
                    LOG.info(f"=== {isolated_ticker}: No new feeds created ===")
                    results.append({
                        "ticker": isolated_ticker,
                        "message": "No new feeds created",
                        "feeds_created": 0
                    })
                    continue

                # STEP 4: Process results from new architecture
                LOG.info(f"✅ Successfully created {len(feeds_created)} feeds for {isolated_ticker} using NEW ARCHITECTURE")

                results.append({
                    "ticker": isolated_ticker,
                    "message": f"Successfully created feeds using new architecture",
                    "feeds_created": len(feeds_created),
                    "details": [{"feed_id": f["feed_id"], "category": f["config"].get("category", "unknown")} for f in feeds_created]
                })

            except Exception as e:
                LOG.error(f"❌ Failed to create feeds for {isolated_ticker}: {e}")
                results.append({
                    "ticker": isolated_ticker,
                    "message": f"Failed to create feeds: {str(e)}",
                    "feeds_created": 0
                })

        # Return results
        LOG.info("=== INITIALIZATION COMPLETE ===")
        return {
            "status": "success",
            "message": "Feed initialization completed using NEW ARCHITECTURE",
            "results": results,
            "total_tickers": len(body.tickers),
            "successful": len([r for r in results if r["feeds_created"] > 0])
        }

def process_feeds_sequentially(feeds: List[Dict]) -> Dict[str, int]:
    """
    Process a list of feeds sequentially (used for Google→Yahoo pairs).
    Returns aggregated stats from all feeds.
    """
    aggregated_stats = {
        "processed": 0,
        "inserted": 0,
        "duplicates": 0,
        "blocked_spam": 0,
        "limit_reached": 0,
        "blocked_insider_trading": 0,
        "yahoo_rejected": 0
    }

    for feed in feeds:
        try:
            stats = ingest_feed_basic_only(feed)
            aggregated_stats["processed"] += stats["processed"]
            aggregated_stats["inserted"] += stats["inserted"]
            aggregated_stats["duplicates"] += stats["duplicates"]
            aggregated_stats["blocked_spam"] += stats.get("blocked_spam", 0)
            aggregated_stats["limit_reached"] += stats.get("limit_reached", 0)
            aggregated_stats["blocked_insider_trading"] += stats.get("blocked_insider_trading", 0)
            aggregated_stats["yahoo_rejected"] += stats.get("yahoo_rejected", 0)
        except Exception as e:
            LOG.error(f"[{feed.get('ticker', 'UNKNOWN')}] Sequential feed processing failed for {feed['name']}: {e}")
            continue

    return aggregated_stats

@APP.post("/cron/ingest")
async def cron_ingest(
    request: Request,
    minutes: int = Query(default=15, description="Time window in minutes"),
    tickers: List[str] = Query(default=None, description="Specific tickers to ingest"),
    batch_size: int = Query(default=None, description="Batch size for concurrent processing"),
    triage_batch_size: int = Query(default=2, description="Batch size for triage processing")
):
    """Enhanced ingest with comprehensive memory monitoring and async batch processing"""
    async with TICKER_PROCESSING_LOCK:
        # Set batch size from parameter or environment variable
        global SCRAPE_BATCH_SIZE
        if batch_size is not None:
            SCRAPE_BATCH_SIZE = max(1, min(batch_size, 10))  # Limit between 1-10
            LOG.info(f"BATCH SIZE OVERRIDE: Using batch_size={SCRAPE_BATCH_SIZE} from API parameter")
        
        start_time = time.time()
        require_admin(request)
        ensure_schema()
        
        # Initialize memory monitoring with error handling
        try:
            memory_monitor.start_monitoring()
            memory_monitor.take_snapshot("CRON_INGEST_START")
            LOG.info("=== CRON INGEST STARTING (WITH MEMORY MONITORING & ASYNC BATCHES) ===")
        except Exception as e:
            LOG.error(f"Memory monitoring failed to start: {e}")
            LOG.info("=== CRON INGEST STARTING (WITHOUT MEMORY MONITORING) ===")
        
        LOG.info(f"Processing window: {minutes} minutes")
        LOG.info(f"Target tickers: {tickers or 'ALL'}")
        LOG.info(f"Batch size: {SCRAPE_BATCH_SIZE}")
        
        try:
            # Reset statistics
            reset_ingestion_stats()
            reset_scraping_stats()
            reset_enhanced_scraping_stats()
            memory_monitor.take_snapshot("STATS_RESET")
            
            # Initialize async semaphores
            init_async_semaphores()
            
            # Calculate dynamic scraping limits for each ticker using enhanced database
            dynamic_limits = {}
            if tickers:
                for ticker in tickers:
                    dynamic_limits[ticker] = calculate_dynamic_scraping_limits(ticker)
                    LOG.info(f"DYNAMIC SCRAPING LIMITS for {ticker}: {dynamic_limits[ticker]}")
            
            memory_monitor.take_snapshot("LIMITS_CALCULATED")
            
            # PHASE 1: Process feeds for new articles
            LOG.info("=== PHASE 1: PROCESSING FEEDS (NEW + EXISTING ARTICLES) ===")
            memory_monitor.take_snapshot("PHASE1_START")
            
            # Get feeds using NEW ARCHITECTURE (feeds + ticker_feeds)
            with resource_cleanup_context("database_connection"):
                with db() as conn, conn.cursor() as cur:
                    if tickers:
                        cur.execute("""
                            SELECT f.id, f.url, f.name, tf.ticker, tf.category, f.retain_days, f.search_keyword, f.competitor_ticker
                            FROM feeds f
                            JOIN ticker_feeds tf ON f.id = tf.feed_id
                            WHERE f.active = TRUE AND tf.active = TRUE AND tf.ticker = ANY(%s)
                            ORDER BY tf.ticker, tf.category, f.id
                        """, (tickers,))
                    else:
                        cur.execute("""
                            SELECT f.id, f.url, f.name, tf.ticker, tf.category, f.retain_days, f.search_keyword, f.competitor_ticker
                            FROM feeds f
                            JOIN ticker_feeds tf ON f.id = tf.feed_id
                            WHERE f.active = TRUE AND tf.active = TRUE
                            ORDER BY tf.ticker, tf.category, f.id
                        """)
                    feeds = list(cur.fetchall())
            
            if not feeds:
                memory_monitor.take_snapshot("NO_FEEDS_FOUND")
                return {"status": "no_feeds", "message": "No active feeds found"}
            
            memory_monitor.take_snapshot("FEEDS_LOADED")
            
            # DEBUG: Check what feeds exist before processing (NEW ARCHITECTURE)
            with resource_cleanup_context("debug_feed_check"):
                with db() as conn, conn.cursor() as cur:
                    if tickers:
                        cur.execute("""
                            SELECT tf.ticker, tf.category, COUNT(*) as count,
                                   STRING_AGG(f.name, ' | ') as feed_names
                            FROM feeds f
                            JOIN ticker_feeds tf ON f.id = tf.feed_id
                            WHERE tf.ticker = ANY(%s) AND f.active = TRUE AND tf.active = TRUE
                            GROUP BY tf.ticker, tf.category
                            ORDER BY tf.ticker, tf.category
                        """, (tickers,))
                    else:
                        cur.execute("""
                            SELECT tf.ticker, tf.category, COUNT(*) as count,
                                   STRING_AGG(f.name, ' | ') as feed_names
                            FROM feeds f
                            JOIN ticker_feeds tf ON f.id = tf.feed_id
                            WHERE f.active = TRUE AND tf.active = TRUE
                            GROUP BY tf.ticker, tf.category
                            ORDER BY tf.ticker, tf.category
                        """)
                    
                    feed_debug = list(cur.fetchall())
                    LOG.info("=== FEED DEBUG BEFORE PROCESSING ===")
                    for feed_row in feed_debug:
                        LOG.info(f"  {feed_row['ticker']} | {feed_row['category']} | Count: {feed_row['count']} | Names: {feed_row['feed_names']}")
                    LOG.info("=== END FEED DEBUG ===")
            
            ingest_stats = {
                "total_processed": 0,
                "total_inserted": 0,
                "total_duplicates": 0,
                "total_spam_blocked": 0,
                "total_limit_reached": 0,
                "total_insider_trading_blocked": 0,
                "total_yahoo_rejected": 0
            }

            # ASYNC FEED PROCESSING: Group feeds by strategy
            LOG.info("=== ASYNC FEED PROCESSING: Grouping feeds by strategy ===")

            # Group feeds by ticker and category
            company_feeds = []       # Will be processed: Google→Yahoo sequentially per ticker
            industry_feeds = []      # Will be processed: All in parallel (Google only)
            competitor_feeds = []    # Will be processed: Google→Yahoo sequentially per competitor

            for feed in feeds:
                category = feed.get('category', 'company')
                if category == 'company':
                    company_feeds.append(feed)
                elif category == 'industry':
                    industry_feeds.append(feed)
                elif category == 'competitor':
                    competitor_feeds.append(feed)

            LOG.info(f"Feed groups - Company: {len(company_feeds)}, Industry: {len(industry_feeds)}, Competitor: {len(competitor_feeds)}")

            # Further group company and competitor feeds by ticker for sequential Google→Yahoo processing
            company_by_ticker = {}  # {ticker: [google_feed, yahoo_feed]}
            for feed in company_feeds:
                ticker = feed.get('ticker')
                if ticker not in company_by_ticker:
                    company_by_ticker[ticker] = []
                company_by_ticker[ticker].append(feed)

            # Sort each ticker's feeds: Google first, then Yahoo
            for ticker in company_by_ticker:
                company_by_ticker[ticker].sort(key=lambda f: 0 if 'google' in f['url'].lower() else 1)

            # Group competitor feeds by (ticker, competitor_ticker) for sequential processing
            competitor_by_key = {}  # {(ticker, competitor_ticker): [google_feed, yahoo_feed]}
            for feed in competitor_feeds:
                ticker = feed.get('ticker')
                comp_ticker = feed.get('competitor_ticker', 'unknown')
                key = (ticker, comp_ticker)
                if key not in competitor_by_key:
                    competitor_by_key[key] = []
                competitor_by_key[key].append(feed)

            # Sort each competitor's feeds: Google first, then Yahoo
            for key in competitor_by_key:
                competitor_by_key[key].sort(key=lambda f: 0 if 'google' in f['url'].lower() else 1)

            # Process all groups in parallel using ThreadPoolExecutor
            LOG.info("=== Starting parallel feed processing with grouped strategy ===")
            processing_start_time = time.time()

            with ThreadPoolExecutor(max_workers=15) as executor:
                futures = []

                # Submit company feed groups (sequential Google→Yahoo within each ticker)
                for ticker, ticker_feeds in company_by_ticker.items():
                    future = executor.submit(process_feeds_sequentially, ticker_feeds)
                    futures.append(("company", ticker, future))
                    LOG.info(f"Submitted company feeds for {ticker}: {len(ticker_feeds)} feeds (Google→Yahoo sequential)")

                # Submit industry feeds (all parallel, Google only)
                for feed in industry_feeds:
                    future = executor.submit(ingest_feed_basic_only, feed)
                    futures.append(("industry", feed.get('search_keyword', 'unknown'), future))
                LOG.info(f"Submitted {len(industry_feeds)} industry feeds (all parallel)")

                # Submit competitor feed groups (sequential Google→Yahoo within each competitor)
                for (ticker, comp_ticker), comp_feeds in competitor_by_key.items():
                    future = executor.submit(process_feeds_sequentially, comp_feeds)
                    futures.append(("competitor", f"{ticker}/{comp_ticker}", future))
                    LOG.info(f"Submitted competitor feeds for {ticker}/{comp_ticker}: {len(comp_feeds)} feeds (Google→Yahoo sequential)")

                # Collect results as they complete
                completed_count = 0
                for future_type, identifier, future in futures:
                    try:
                        stats = future.result()
                        ingest_stats["total_processed"] += stats["processed"]
                        ingest_stats["total_inserted"] += stats["inserted"]
                        ingest_stats["total_duplicates"] += stats["duplicates"]
                        ingest_stats["total_spam_blocked"] += stats.get("blocked_spam", 0)
                        ingest_stats["total_limit_reached"] += stats.get("limit_reached", 0)
                        ingest_stats["total_insider_trading_blocked"] += stats.get("blocked_insider_trading", 0)
                        ingest_stats["total_yahoo_rejected"] += stats.get("yahoo_rejected", 0)

                        completed_count += 1
                        LOG.info(f"✅ Completed {future_type} feed group: {identifier} ({completed_count}/{len(futures)})")

                        # Memory monitoring every 5 completions
                        if completed_count % 5 == 0:
                            memory_monitor.take_snapshot(f"FEED_PROCESSING_{completed_count}")
                            current_memory = memory_monitor.get_memory_info()
                            if current_memory["memory_mb"] > 500:  # 500MB threshold
                                LOG.warning(f"High memory during feed processing: {current_memory['memory_mb']:.1f}MB")
                                memory_monitor.force_garbage_collection()

                    except Exception as e:
                        LOG.error(f"❌ Failed {future_type} feed group: {identifier} - {e}")
                        continue

            processing_duration = time.time() - processing_start_time
            LOG.info(f"=== ASYNC FEED PROCESSING COMPLETE: {processing_duration:.2f} seconds ({len(futures)} groups processed) ===")
            
            memory_monitor.take_snapshot("PHASE1_FEEDS_COMPLETE")
            
            # Now get ALL articles from the timeframe for ticker-specific analysis
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            articles_by_ticker = {}

            with resource_cleanup_context("database_connection"):
                with db() as conn, conn.cursor() as cur:
                    if tickers:
                        cur.execute("""
                            WITH ranked_articles AS (
                                SELECT a.id, a.url, a.resolved_url, a.title, a.domain, a.published_at,
                                       ta.category, ta.search_keyword, ta.competitor_ticker, ta.ticker,
                                       a.scraped_content, ta.ai_summary, a.url_hash,
                                       ROW_NUMBER() OVER (
                                           PARTITION BY ta.ticker,
                                               CASE
                                                   WHEN ta.category = 'company' THEN ta.category
                                                   WHEN ta.category = 'industry' THEN ta.search_keyword
                                                   WHEN ta.category = 'competitor' THEN ta.competitor_ticker
                                               END
                                           ORDER BY a.published_at DESC NULLS LAST, ta.found_at DESC
                                       ) as rn
                                FROM articles a
                                JOIN ticker_articles ta ON a.id = ta.article_id
                                WHERE ta.ticker = ANY(%s)
                                AND (a.published_at >= %s OR a.published_at IS NULL)
                            )
                            SELECT id, url, resolved_url, title, domain, published_at,
                                   category, search_keyword, competitor_ticker, ticker,
                                   scraped_content, ai_summary, url_hash
                            FROM ranked_articles
                            WHERE (category = 'company' AND rn <= 50)
                               OR (category = 'industry' AND rn <= 25)
                               OR (category = 'competitor' AND rn <= 25)
                            ORDER BY ticker, category, rn
                        """, (tickers, cutoff))
                    else:
                        cur.execute("""
                            WITH ranked_articles AS (
                                SELECT a.id, a.url, a.resolved_url, a.title, a.domain, a.published_at,
                                       ta.category, ta.search_keyword, ta.competitor_ticker, ta.ticker,
                                       a.scraped_content, ta.ai_summary, a.url_hash,
                                       ROW_NUMBER() OVER (
                                           PARTITION BY ta.ticker,
                                               CASE
                                                   WHEN ta.category = 'company' THEN ta.category
                                                   WHEN ta.category = 'industry' THEN ta.search_keyword
                                                   WHEN ta.category = 'competitor' THEN ta.competitor_ticker
                                               END
                                           ORDER BY a.published_at DESC NULLS LAST, ta.found_at DESC
                                       ) as rn
                                FROM articles a
                                JOIN ticker_articles ta ON a.id = ta.article_id
                                WHERE (a.published_at >= %s OR a.published_at IS NULL)
                            )
                            SELECT id, url, resolved_url, title, domain, published_at,
                                   category, search_keyword, competitor_ticker, ticker,
                                   scraped_content, ai_summary, url_hash
                            FROM ranked_articles
                            WHERE (category = 'company' AND rn <= 50)
                               OR (category = 'industry' AND rn <= 25)
                               OR (category = 'competitor' AND rn <= 25)
                            ORDER BY ticker, category, rn
                        """, (cutoff,))

                    all_articles = list(cur.fetchall())

            memory_monitor.take_snapshot("ARTICLES_LOADED")

            # Log filtering statistics
            with db() as conn, conn.cursor() as cur:
                if tickers:
                    cur.execute("""
                        SELECT
                            ta.ticker,
                            ta.category,
                            COUNT(*) FILTER (WHERE a.published_at >= %s OR a.published_at IS NULL) as within_period,
                            COUNT(*) FILTER (WHERE a.published_at < %s) as outside_period
                        FROM ticker_articles ta
                        JOIN articles a ON ta.article_id = a.id
                        WHERE ta.ticker = ANY(%s)
                        GROUP BY ta.ticker, ta.category
                        ORDER BY ta.ticker, ta.category
                    """, (cutoff, cutoff, tickers))
                else:
                    cur.execute("""
                        SELECT
                            ta.ticker,
                            ta.category,
                            COUNT(*) FILTER (WHERE a.published_at >= %s OR a.published_at IS NULL) as within_period,
                            COUNT(*) FILTER (WHERE a.published_at < %s) as outside_period
                        FROM ticker_articles ta
                        JOIN articles a ON ta.article_id = a.id
                        GROUP BY ta.ticker, ta.category
                        ORDER BY ta.ticker, ta.category
                    """, (cutoff, cutoff))

                filter_stats = cur.fetchall()
                for stat in filter_stats:
                    if stat['outside_period'] > 0:
                        LOG.info(f"📅 [{stat['ticker']}] {stat['category']}: {stat['within_period']} within report period, {stat['outside_period']} excluded (published outside {minutes}min window)")

            # Organize articles by ticker and category
            for article in all_articles:
                ticker = article["ticker"]
                category = article["category"] or "company"

                if ticker not in articles_by_ticker:
                    articles_by_ticker[ticker] = {"company": [], "industry": [], "competitor": []}

                articles_by_ticker[ticker][category].append(article)

            total_articles = len(all_articles)
            LOG.info(f"=== PHASE 1 COMPLETE: {ingest_stats['total_inserted']} new + {total_articles} total in timeframe ===")
            memory_monitor.take_snapshot("PHASE1_COMPLETE")
            
            # PHASE 2: Pure AI triage
            LOG.info("=== PHASE 2: PURE AI TRIAGE ===")
            memory_monitor.take_snapshot("PHASE2_START")
            triage_results = {}
            
            for ticker in articles_by_ticker.keys():
                LOG.info(f"Running pure AI triage for {ticker}")
                memory_monitor.take_snapshot(f"TRIAGE_START_{ticker}")
                
                with resource_cleanup_context("ai_triage"):
                    # Use dual scoring triage (OpenAI + Claude)
                    selected_results = await perform_ai_triage_with_dual_scoring_async(articles_by_ticker[ticker], ticker, triage_batch_size)
                    triage_results[ticker] = selected_results

                memory_monitor.take_snapshot(f"TRIAGE_COMPLETE_{ticker}")

                # Build flagged articles list (will be stored by process_ingest_phase)
                flagged_articles = []
                with resource_cleanup_context("database_connection"):
                    for category, selected_items in selected_results.items():
                        articles = articles_by_ticker[ticker][category]
                        LOG.info(f"  Building flagged list for {category}: {len(selected_items)} selected from {len(articles)} articles")
                        for item in selected_items:
                            article_idx = item["id"]
                            if article_idx < len(articles):
                                article = articles[article_idx]
                                article_id = article.get("id")
                                if article_id:
                                    flagged_articles.append(article_id)
                                else:
                                    LOG.warning(f"  Article at index {article_idx} in {category} has no ID!")

                LOG.info(f"✅ Built flagged articles list: {len(flagged_articles)} article IDs for {ticker}")
                if flagged_articles:
                    LOG.info(f"   Sample IDs: {flagged_articles[:5]}...")

            memory_monitor.take_snapshot("PHASE2_COMPLETE")
            
            # PHASE 3: Send enhanced quick email
            LOG.info("=== PHASE 3: SENDING ENHANCED QUICK TRIAGE EMAIL ===")
            memory_monitor.take_snapshot("PHASE3_START")

            with resource_cleanup_context("email_sending"):
                quick_email_sent = send_enhanced_quick_intelligence_email(articles_by_ticker, triage_results, minutes)
            
            LOG.info(f"Enhanced quick triage email sent: {quick_email_sent}")
            memory_monitor.take_snapshot("PHASE3_COMPLETE")
            
            # PHASE 4: Ticker-specific content scraping and analysis (WITH ASYNC BATCH PROCESSING)
            LOG.info("=== PHASE 4: TICKER-SPECIFIC CONTENT SCRAPING AND ANALYSIS (ASYNC BATCHES) ===")
            memory_monitor.take_snapshot("PHASE4_START")
            scraping_final_stats = {"scraped": 0, "failed": 0, "ai_analyzed": 0, "reused_existing": 0}
            
            # Count total articles to be processed for heartbeat tracking
            total_articles_to_process = 0
            for target_ticker in articles_by_ticker.keys():
                selected = triage_results.get(target_ticker, {})
                for category in ["company", "industry", "competitor"]:
                    total_articles_to_process += len(selected.get(category, []))
            
            processed_count = 0
            LOG.info(f"Starting Phase 4: {total_articles_to_process} total articles to process in batches of {SCRAPE_BATCH_SIZE}")
            
            # Track which tickers were successfully processed
            successfully_processed_tickers = set()
            
            for target_ticker in articles_by_ticker.keys():
                memory_monitor.take_snapshot(f"TICKER_START_{target_ticker}")
                
                config = get_ticker_config(target_ticker)
                metadata = {
                    "industry_keywords": config.get("industry_keywords", []) if config else [],
                    "competitors": config.get("competitors", []) if config else []
                }
                
                ticker_success_count = 0
                selected = triage_results.get(target_ticker, {})
                
                # Collect all selected articles for this ticker across categories
                all_selected_articles = []
                for category in ["company", "industry", "competitor"]:
                    category_selected = selected.get(category, [])
                    
                    for item in category_selected:
                        article_idx = item["id"]
                        if article_idx < len(articles_by_ticker[target_ticker][category]):
                            article = articles_by_ticker[target_ticker][category][article_idx]
                            all_selected_articles.append({
                                "article": article,
                                "category": category,
                                "item": item
                            })
                
                # Process articles in batches
                total_batches = (len(all_selected_articles) + SCRAPE_BATCH_SIZE - 1) // SCRAPE_BATCH_SIZE
                LOG.info(f"TICKER {target_ticker}: Processing {len(all_selected_articles)} articles in {total_batches} batches of {SCRAPE_BATCH_SIZE}")
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * SCRAPE_BATCH_SIZE
                    end_idx = min(start_idx + SCRAPE_BATCH_SIZE, len(all_selected_articles))
                    batch = all_selected_articles[start_idx:end_idx]
                    
                    LOG.info(f"TICKER {target_ticker}: Processing batch {batch_num + 1}/{total_batches} ({len(batch)} articles)")
                    
                    # Prepare batch for processing
                    batch_articles = []
                    batch_categories = []
                    for selected_article in batch:
                        batch_articles.append(selected_article["article"])
                        batch_categories.append(selected_article["category"])
                    
                    # Process batch asynchronously with per-article categories
                    # Add company_name to metadata for AI summarization
                    metadata["company_name"] = config.get("company_name", target_ticker) if config else target_ticker
                    try:
                        batch_results = await process_article_batch_async(
                            batch_articles,
                            batch_categories,  # Pass all categories for per-article processing
                            metadata,
                            target_ticker
                        )
                        
                        # Update statistics based on batch results
                        for result in batch_results:
                            processed_count += 1
                            
                            if result["success"]:
                                ticker_success_count += 1
                                if result.get("scraped_content") and result.get("ai_summary"):
                                    scraping_final_stats["scraped"] += 1
                                    scraping_final_stats["ai_analyzed"] += 1
                                elif result.get("scraped_content"):
                                    scraping_final_stats["reused_existing"] += 1
                            else:
                                scraping_final_stats["failed"] += 1
                        
                        # MEMORY MONITORING: Every batch
                        elapsed = time.time() - start_time
                        memory_monitor.take_snapshot(f"BATCH_{batch_num + 1}_COMPLETE")
                        current_memory = memory_monitor.get_memory_info()
                        
                        LOG.info(f"BATCH COMPLETE: {batch_num + 1}/{total_batches} for {target_ticker} - {processed_count}/{total_articles_to_process} total ({elapsed:.1f}s elapsed)")
                        LOG.info(f"BATCH STATS: Scraped:{scraping_final_stats['scraped']}, Failed:{scraping_final_stats['failed']}, Reused:{scraping_final_stats['reused_existing']}")
                        LOG.info(f"MEMORY: {current_memory['memory_mb']:.1f}MB, CPU: {current_memory['cpu_percent']:.1f}%")
                        
                        # Force garbage collection if memory gets high
                        if current_memory["memory_mb"] > 800:  # 800MB threshold
                            LOG.warning(f"HIGH MEMORY USAGE: {current_memory['memory_mb']:.1f}MB - forcing garbage collection")
                            gc_stats = memory_monitor.force_garbage_collection()
                            memory_monitor.take_snapshot(f"BATCH_POST_GC_{batch_num + 1}")
                            LOG.info(f"GC: Freed {gc_stats['objects_freed']} objects")
                            
                    except Exception as e:
                        LOG.error(f"BATCH PROCESSING ERROR: Batch {batch_num + 1} for {target_ticker} failed: {e}")
                        # Mark articles in failed batch as failed
                        for _ in batch:
                            processed_count += 1
                            scraping_final_stats["failed"] += 1
                        continue
                
                # Track tickers that had successful processing
                if ticker_success_count > 0:
                    successfully_processed_tickers.add(target_ticker)
                
                LOG.info(f"TICKER {target_ticker} COMPLETE: {ticker_success_count} successful articles")
                memory_monitor.take_snapshot(f"TICKER_COMPLETE_{target_ticker}")
            
            # Final heartbeat before completion
            elapsed = time.time() - start_time
            LOG.info(f"PHASE 4 COMPLETE: All {total_articles_to_process} articles processed in batches in {elapsed:.1f}s")
            LOG.info(f"=== PHASE 4 COMPLETE: {scraping_final_stats['scraped']} new + {scraping_final_stats['reused_existing']} reused ===")
            memory_monitor.take_snapshot("PHASE4_COMPLETE")

            # Consolidated scraping statistics
            total_attempts = enhanced_scraping_stats["total_attempts"]
            if total_attempts > 0:
                total_success = (enhanced_scraping_stats["requests_success"] +
                                enhanced_scraping_stats["playwright_success"] +
                                enhanced_scraping_stats.get("scrapfly_success", 0))
                overall_rate = (total_success / total_attempts) * 100

                requests_attempts = enhanced_scraping_stats["by_method"]["requests"]["attempts"]
                requests_success = enhanced_scraping_stats["by_method"]["requests"]["successes"]
                requests_rate = (requests_success / requests_attempts * 100) if requests_attempts > 0 else 0

                playwright_attempts = enhanced_scraping_stats["by_method"]["playwright"]["attempts"]
                playwright_success = enhanced_scraping_stats["by_method"]["playwright"]["successes"]
                playwright_rate = (playwright_success / playwright_attempts * 100) if playwright_attempts > 0 else 0

                scrapfly_attempts = enhanced_scraping_stats["by_method"].get("scrapfly", {}).get("attempts", 0)
                scrapfly_success = enhanced_scraping_stats["by_method"].get("scrapfly", {}).get("successes", 0)
                scrapfly_rate = (scrapfly_success / scrapfly_attempts * 100) if scrapfly_attempts > 0 else 0

                LOG.info("=" * 60)
                LOG.info("SCRAPING SUCCESS RATES")
                LOG.info("=" * 60)
                LOG.info(f"OVERALL SUCCESS: {overall_rate:.1f}% ({total_success}/{total_attempts})")
                LOG.info(f"TIER 1 (Requests): {requests_rate:.1f}% ({requests_success}/{requests_attempts})")
                LOG.info(f"TIER 2 (Playwright): {playwright_rate:.1f}% ({playwright_success}/{playwright_attempts})")
                LOG.info(f"TIER 3 (Scrapfly): {scrapfly_rate:.1f}% ({scrapfly_success}/{scrapfly_attempts})")
                if scrapfly_stats["requests_made"] > 0:
                    LOG.info(f"Scrapfly Cost: ${scrapfly_stats['cost_estimate']:.3f}")
                LOG.info("=" * 60)
            
            # CRITICAL: Final cleanup before returning response
            LOG.info("=== PERFORMING FINAL CLEANUP ===")
            memory_monitor.take_snapshot("BEFORE_FINAL_CLEANUP")
            
            try:
                await full_resource_cleanup()  # Single call with await
                memory_monitor.take_snapshot("AFTER_FINAL_CLEANUP")
                LOG.info("=== FINAL CLEANUP COMPLETE ===")
            except Exception as cleanup_error:
                LOG.error(f"Error during final cleanup: {cleanup_error}")
                memory_monitor.take_snapshot("CLEANUP_ERROR")
            
            processing_time = time.time() - start_time
            LOG.info(f"=== CRON INGEST COMPLETE - Total time: {processing_time:.1f}s ===")
            
            # Calculate scraping stats
            total_scraping_attempts = enhanced_scraping_stats["total_attempts"]
            total_scraping_success = (enhanced_scraping_stats["requests_success"] +
                                     enhanced_scraping_stats["playwright_success"] +
                                     enhanced_scraping_stats.get("scrapfly_success", 0))
            overall_scraping_rate = (total_scraping_success / total_scraping_attempts * 100) if total_scraping_attempts > 0 else 0
            
            # Stop memory monitoring and get summary
            memory_summary = memory_monitor.stop_monitoring()
            
            # Prepare response with monitoring data
            response = {
                "status": "completed",
                "processing_time_seconds": round(processing_time, 1),
                "workflow": "enhanced_ticker_reference_with_async_batch_processing",
                "batch_size_used": SCRAPE_BATCH_SIZE,

                "phase_1_ingest": {
                    "total_processed": ingest_stats["total_processed"],
                    "total_inserted": ingest_stats["total_inserted"],
                    "total_duplicates": ingest_stats["total_duplicates"],
                    "total_spam_blocked": ingest_stats["total_spam_blocked"],
                    "total_limit_reached": ingest_stats["total_limit_reached"],
                    "total_insider_trading_blocked": ingest_stats["total_insider_trading_blocked"],
                    "total_yahoo_rejected": ingest_stats["total_yahoo_rejected"],
                    "total_articles_in_timeframe": total_articles
                },
                "phase_2_triage": {
                    "type": "pure_ai_triage_only",
                    "tickers_processed": len(triage_results),
                    "selections_by_ticker": {k: {cat: len(items) for cat, items in v.items()} for k, v in triage_results.items()},
                    "flagged_articles": flagged_articles  # Article IDs flagged during triage
                },
                "phase_3_quick_email": {"sent": quick_email_sent},
                "phase_4_async_batch_scraping": {
                    **scraping_final_stats,
                    "overall_success_rate": f"{overall_scraping_rate:.1f}%",
                    "tier_breakdown": {
                        "requests_success": enhanced_scraping_stats["requests_success"],
                        "playwright_success": enhanced_scraping_stats["playwright_success"],
                        "scrapfly_success": enhanced_scraping_stats.get("scrapfly_success", 0),
                        "total_attempts": total_scraping_attempts
                    },
                    "scrapfly_cost": f"${scrapfly_stats['cost_estimate']:.3f}",
                    "dynamic_limits": dynamic_limits,
                    "batch_processing": {
                        "batch_size": SCRAPE_BATCH_SIZE,
                        "total_articles": total_articles_to_process,
                        "articles_per_batch": SCRAPE_BATCH_SIZE,
                        "estimated_batches": total_articles_to_process // SCRAPE_BATCH_SIZE if total_articles_to_process > 0 else 0
                    }
                },
                "successfully_processed_tickers": list(successfully_processed_tickers),
                "message": f"Processing completed successfully with async batch processing (batch_size={SCRAPE_BATCH_SIZE})",
                "github_sync_required": len(successfully_processed_tickers) > 0,

                # Memory monitoring data
                "memory_monitoring": {
                    "enabled": True,
                    "total_snapshots": len(memory_monitor.snapshots),
                    "memory_summary": memory_summary,
                    "peak_memory_mb": max([s.get("memory_mb", 0) for s in memory_monitor.snapshots]) if memory_monitor.snapshots else 0,
                    "final_memory_mb": memory_summary.get("final_memory_mb") if memory_summary else None,
                    "total_memory_change_mb": memory_summary.get("total_change_mb") if memory_summary else None
                }
            }
            
            return response
            
        except Exception as e:
            # CRITICAL ERROR HANDLING WITH MEMORY SNAPSHOT
            LOG.error(f"CRITICAL ERROR in cron_ingest: {str(e)}")
            memory_monitor.take_snapshot("CRITICAL_ERROR")
            
            # Get detailed memory info at crash
            current_memory = memory_monitor.get_memory_info()
            tracemalloc_info = memory_monitor.get_tracemalloc_top(20)
            
            LOG.error(f"MEMORY AT CRASH: {current_memory}")
            LOG.error(f"TOP MEMORY ALLOCATIONS AT CRASH: {tracemalloc_info}")
            
            # Emergency cleanup
            try:
                cleanup_result = await full_resource_cleanup()
                LOG.info(f"Emergency cleanup completed: {cleanup_result}")
            except Exception as cleanup_error:
                LOG.error(f"Error during emergency cleanup: {cleanup_error}")
            
            return {
                "status": "critical_error",
                "message": str(e),
                "batch_size_used": SCRAPE_BATCH_SIZE,
                "memory_monitoring": {
                    "snapshots_taken": len(memory_monitor.snapshots),
                    "memory_at_crash": current_memory,
                    "top_allocations": tracemalloc_info[:5],  # Top 5 memory allocations
                    "error_occurred": True
                }
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
async def cron_digest(
    request: Request,
    minutes: int = Query(default=1440, description="Time window in minutes"),
    tickers: List[str] = Query(default=None, description="Specific tickers for digest")
):
    async with TICKER_PROCESSING_LOCK:
        """Generate and send email digest with content scraping data and AI summaries"""
        require_admin(request)
        ensure_schema()

        try:
            LOG.info(f"=== DIGEST GENERATION STARTING ===")
            LOG.info(f"Time window: {minutes} minutes, Tickers: {tickers}")

            # Use the existing enhanced digest function that sends emails
            LOG.info("Calling enhanced digest function...")
            result = fetch_digest_articles_with_enhanced_content(minutes / 60, tickers)

            # The function returns a detailed result dict, let's pass it through with additional metadata
            if isinstance(result, dict):
                result["minutes"] = minutes
                result["requested_tickers"] = tickers
                LOG.info(f"Digest result: {result.get('status', 'unknown')} - {result.get('articles', 0)} articles")
                return result
            else:
                LOG.error(f"Unexpected result type from digest function: {type(result)}")
                return {
                    "status": "error",
                    "message": "Unexpected result from digest generation",
                    "minutes": minutes,
                    "tickers": tickers
                }

        except Exception as e:
            LOG.error(f"Digest generation failed: {e}")
            LOG.error(f"Error details: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": str(e),
                "tickers": tickers,
                "minutes": minutes
            }
        
# FIXED: Updated endpoint with proper request body handling
@APP.post("/admin/clean-feeds")
async def clean_old_feeds(request: Request, body: CleanFeedsRequest):
    async with TICKER_PROCESSING_LOCK:
        """Clean old Reddit/Twitter feeds from database"""
        require_admin(request)
        
        with db() as conn, conn.cursor() as cur:
            # Check if tables exist first (safe for fresh database)
            try:
                cur.execute("SELECT 1 FROM ticker_feeds LIMIT 1")
                cur.execute("SELECT 1 FROM feeds LIMIT 1")
            except Exception as e:
                LOG.info(f"📋 Tables don't exist yet (fresh database) - clean-feeds skipped: {e}")
                return {"status": "skipped", "message": "Tables don't exist yet - nothing to clean", "deleted": 0}

            # Delete feeds that contain Reddit, Twitter, SEC, StockTwits
            cleanup_patterns = [
                "Reddit", "Twitter", "SEC EDGAR", "StockTwits",
                "r/investing", "r/stocks", "r/SecurityAnalysis",
                "r/ValueInvesting", "r/energy", "@TalenEnergy"
            ]

            total_deleted = 0
            if body.tickers:
                # NEW ARCHITECTURE: Remove ticker-feed associations for specific tickers
                for pattern in cleanup_patterns:
                    # First, remove ticker-feed associations for feeds matching the pattern
                    cur.execute("""
                        DELETE FROM ticker_feeds
                        WHERE ticker = ANY(%s) AND feed_id IN (
                            SELECT id FROM feeds WHERE name LIKE %s
                        )
                    """, (body.tickers, f"%{pattern}%"))
                    total_deleted += cur.rowcount

                    # Then, delete feeds that have no more associations
                    cur.execute("""
                        DELETE FROM feeds
                        WHERE name LIKE %s AND id NOT IN (
                            SELECT DISTINCT feed_id FROM ticker_feeds WHERE active = TRUE
                        )
                    """, (f"%{pattern}%",))
                    total_deleted += cur.rowcount
            else:
                # NEW ARCHITECTURE: Delete all feeds matching patterns (and their associations)
                for pattern in cleanup_patterns:
                    # First remove all ticker-feed associations for these feeds
                    cur.execute("""
                        DELETE FROM ticker_feeds
                        WHERE feed_id IN (SELECT id FROM feeds WHERE name LIKE %s)
                    """, (f"%{pattern}%",))

                    # Then delete the feeds themselves
                    cur.execute("""
                        DELETE FROM feeds
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

@APP.get("/admin/domain-names")
def get_domain_names(request: Request, limit: int = 1000, offset: int = 0):
    """Export domain names database (domain → formal name mappings)"""
    require_admin(request)

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT domain, formal_name, ai_generated, created_at, updated_at
                FROM domain_names
                ORDER BY domain
                LIMIT %s OFFSET %s
            """, (limit, offset))

            domains = cur.fetchall()

            # Get total count
            cur.execute("SELECT COUNT(*) as total FROM domain_names")
            total = cur.fetchone()['total']

            return {
                "status": "success",
                "total": total,
                "returned": len(domains),
                "limit": limit,
                "offset": offset,
                "domains": [dict(d) for d in domains]
            }
    except Exception as e:
        LOG.error(f"Failed to fetch domain names: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@APP.post("/admin/regenerate-metadata")
async def regenerate_metadata(request: Request, body: RegenerateMetadataRequest):
    """Force regeneration of AI metadata for a ticker"""
    async with TICKER_PROCESSING_LOCK:
        require_admin(request)

        if not OPENAI_API_KEY:
            return {"status": "error", "message": "OpenAI API key not configured"}

        LOG.info(f"Regenerating metadata for {body.ticker}")
        metadata = ticker_manager.get_or_create_metadata(body.ticker)

        # Rebuild feeds using NEW ARCHITECTURE V2
        feeds_created = create_feeds_for_ticker_new_architecture(body.ticker, metadata)

        return {
            "status": "regenerated",
            "ticker": body.ticker,
            "metadata": metadata,
            "feeds_created": len(feeds_created)
        }


@APP.post("/admin/force-digest")
def force_digest(request: Request, body: ForceDigestRequest):
    """Force digest with existing articles (for testing) - Enhanced with AI analysis"""
    require_admin(request)
    
    with db() as conn, conn.cursor() as cur:
        if body.tickers:
            cur.execute("""
                SELECT
                    a.url, a.resolved_url, a.title, a.description,
                    ta.ticker, a.domain, a.published_at,
                    ta.found_at, ta.category,
                    ta.search_keyword
                FROM articles a
                JOIN ticker_articles ta ON a.id = ta.article_id
                WHERE ta.found_at >= %s
                    AND ta.ticker = ANY(%s)
                ORDER BY ta.ticker, ta.category, COALESCE(a.published_at, ta.found_at) DESC, ta.found_at DESC
            """, (datetime.now(timezone.utc) - timedelta(days=7), body.tickers))
        else:
            cur.execute("""
                SELECT
                    a.url, a.resolved_url, a.title, a.description,
                    ta.ticker, a.domain, a.published_at,
                    ta.found_at, ta.category,
                    ta.search_keyword
                FROM articles a
                JOIN ticker_articles ta ON a.id = ta.article_id
                WHERE ta.found_at >= %s
                ORDER BY ta.ticker, ta.category, COALESCE(a.published_at, ta.found_at) DESC, ta.found_at DESC
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
        cur.execute("DELETE FROM ticker_articles")
        cur.execute("DELETE FROM articles")
        deleted_stats["articles"] = cur.rowcount
        
        # Delete all feeds (NEW ARCHITECTURE)
        cur.execute("DELETE FROM ticker_feeds")
        deleted_stats["ticker_feeds"] = cur.rowcount
        cur.execute("DELETE FROM feeds")
        deleted_stats["feeds"] = cur.rowcount
        
        # Delete all ticker configurations (keywords, competitors, etc.)
        cur.execute("DELETE FROM ticker_reference")  
        deleted_stats["ticker_references"] = cur.rowcount
        
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
        # Check if tables exist first (safe for fresh database)
        try:
            cur.execute("SELECT 1 FROM ticker_articles LIMIT 1")
            cur.execute("SELECT 1 FROM ticker_feeds LIMIT 1")
            cur.execute("SELECT 1 FROM feeds LIMIT 1")
            cur.execute("SELECT 1 FROM ticker_reference LIMIT 1")
        except Exception as e:
            LOG.info(f"📋 Tables don't exist yet (fresh database) - wipe-ticker skipped: {e}")
            return {"status": "skipped", "message": "Tables don't exist yet - nothing to wipe", "ticker": ticker, "deleted": {}}

        deleted_stats = {}

        # Delete articles for this ticker
        cur.execute("DELETE FROM ticker_articles WHERE ticker = %s", (ticker,))
        deleted_stats["articles"] = cur.rowcount
        
        # Delete feeds for this ticker (NEW ARCHITECTURE)
        cur.execute("DELETE FROM ticker_feeds WHERE ticker = %s", (ticker,))
        deleted_stats["ticker_feeds"] = cur.rowcount

        # Delete orphaned feeds (feeds with no ticker associations)
        cur.execute("""
            DELETE FROM feeds
            WHERE id NOT IN (SELECT DISTINCT feed_id FROM ticker_feeds WHERE active = TRUE)
        """)
        deleted_stats["orphaned_feeds"] = cur.rowcount
        
        # Delete ticker configuration
        cur.execute("DELETE FROM ticker_reference WHERE ticker = %s", (ticker,))
        deleted_stats["ticker_reference"] = cur.rowcount
        
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
async def get_stats(
    request: Request,
    tickers: List[str] = Query(default=None, description="Filter stats by tickers")
):
    async with TICKER_PROCESSING_LOCK:
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
                        COUNT(DISTINCT ta.ticker) as tickers,
                        COUNT(DISTINCT a.domain) as domains,
                        MAX(a.published_at) as latest_article
                    FROM articles a
                    JOIN ticker_articles ta ON a.id = ta.article_id
                    WHERE ta.found_at > NOW() - INTERVAL '7 days'
                        AND ta.ticker = ANY(%s)
                """, (tickers,))
                stats = dict(cur.fetchone())
                
                # Stats by category
                cur.execute("""
                    SELECT ta.category, COUNT(*) as count
                    FROM ticker_articles ta
                    WHERE ta.found_at > NOW() - INTERVAL '7 days'
                        AND ta.ticker = ANY(%s)
                    GROUP BY ta.category
                    ORDER BY ta.category
                """, (tickers,))
                stats["by_category"] = list(cur.fetchall())
                
                # Top domains
                cur.execute("""
                    SELECT a.domain, COUNT(*) as count
                    FROM articles a
                    JOIN ticker_articles ta ON a.id = ta.article_id
                    WHERE ta.found_at > NOW() - INTERVAL '7 days'
                        AND ta.ticker = ANY(%s)
                    GROUP BY a.domain
                    ORDER BY count DESC
                    LIMIT 10
                """, (tickers,))
                stats["top_domains"] = list(cur.fetchall())
                
                # Articles by ticker and category
                cur.execute("""
                    SELECT ta.ticker, ta.category, COUNT(*) as count
                    FROM ticker_articles ta
                    WHERE ta.found_at > NOW() - INTERVAL '7 days'
                        AND ta.ticker = ANY(%s)
                    GROUP BY ta.ticker, ta.category
                    ORDER BY ta.ticker, ta.category
                """, (tickers,))
                stats["by_ticker_category"] = list(cur.fetchall())
            else:
                # Article stats
                cur.execute("""
                    SELECT
                        COUNT(*) as total_articles,
                        COUNT(DISTINCT ta.ticker) as tickers,
                        COUNT(DISTINCT a.domain) as domains,
                        MAX(a.published_at) as latest_article
                    FROM articles a
                    JOIN ticker_articles ta ON a.id = ta.article_id
                    WHERE ta.found_at > NOW() - INTERVAL '7 days'
                """)
                stats = dict(cur.fetchone())
                
                # Stats by category
                cur.execute("""
                    SELECT ta.category, COUNT(*) as count
                    FROM ticker_articles ta
                    WHERE ta.found_at > NOW() - INTERVAL '7 days'
                    GROUP BY ta.category
                    ORDER BY ta.category
                """)
                stats["by_category"] = list(cur.fetchall())
                
                # Top domains
                cur.execute("""
                    SELECT a.domain, COUNT(*) as count
                    FROM articles a
                    JOIN ticker_articles ta ON a.id = ta.article_id
                    WHERE ta.found_at > NOW() - INTERVAL '7 days'
                    GROUP BY a.domain
                    ORDER BY count DESC
                    LIMIT 10
                """)
                stats["top_domains"] = list(cur.fetchall())
                
                # Articles by ticker and category
                cur.execute("""
                    SELECT ta.ticker, ta.category, COUNT(*) as count
                    FROM ticker_articles ta
                    WHERE ta.found_at > NOW() - INTERVAL '7 days'
                    GROUP BY ta.ticker, ta.category
                    ORDER BY ta.ticker, ta.category
                """)
                stats["by_ticker_category"] = list(cur.fetchall())
            
            # Check AI metadata status
            cur.execute("""
                SELECT COUNT(*) as total, 
                       COUNT(CASE WHEN ai_generated THEN 1 END) as ai_generated
                FROM ticker_reference
                WHERE active = TRUE
            """)
            ai_stats = cur.fetchone()
            stats["ai_metadata"] = ai_stats
        
        return stats

@APP.post("/admin/reset-digest-flags")
async def reset_digest_flags(request: Request, body: ResetDigestRequest):
    async with TICKER_PROCESSING_LOCK:
        """Reset sent_in_digest flags for testing"""
        require_admin(request)
        # Note: ensure_schema() not needed for simple UPDATE operation

        with db() as conn, conn.cursor() as cur:
            # Check if ticker_articles table exists first (safe for fresh database)
            try:
                cur.execute("SELECT 1 FROM ticker_articles LIMIT 1")
            except Exception as e:
                LOG.info(f"📋 ticker_articles table doesn't exist yet (fresh database) - reset-digest-flags skipped: {e}")
                return {"status": "skipped", "message": "ticker_articles table doesn't exist yet - nothing to reset", "articles_reset": 0, "tickers": body.tickers or "all"}

            if body.tickers:
                cur.execute("UPDATE ticker_articles SET sent_in_digest = FALSE WHERE ticker = ANY(%s)", (body.tickers,))
            else:
                cur.execute("UPDATE ticker_articles SET sent_in_digest = FALSE")
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
        # AI analysis is no longer stored in database with new schema
        # This operation is no longer needed
        reset_count = 0
        
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
                SELECT a.id, a.title, a.description, a.domain, ta.ticker, ta.category, ta.search_keyword
                FROM articles a
                JOIN ticker_articles ta ON a.id = ta.article_id
                WHERE ta.ticker = ANY(%s)
                ORDER BY ta.found_at DESC
                LIMIT %s
            """, (tickers, limit))
        else:
            cur.execute("""
                SELECT a.id, a.title, a.description, a.domain, ta.ticker, ta.category, ta.search_keyword
                FROM articles a
                JOIN ticker_articles ta ON a.id = ta.article_id
                ORDER BY ta.found_at DESC
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
            
            # AI analysis is no longer stored in database with new schema
            # Analysis results are calculated on-demand
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

# Pydantic models for request bodies
class TickerReferenceRequest(BaseModel):
    ticker: str
    country: str
    company_name: str
    industry: Optional[str] = None
    sector: Optional[str] = None
    sub_industry: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    market_cap_category: Optional[str] = None
    yahoo_ticker: Optional[str] = None
    active: bool = True
    is_etf: bool = False
    industry_keyword_1: Optional[str] = None
    industry_keyword_2: Optional[str] = None
    industry_keyword_3: Optional[str] = None
    competitor_1_name: Optional[str] = None
    competitor_1_ticker: Optional[str] = None
    competitor_2_name: Optional[str] = None
    competitor_2_ticker: Optional[str] = None
    competitor_3_name: Optional[str] = None
    competitor_3_ticker: Optional[str] = None

class GitHubSyncRequest(BaseModel):
    commit_message: Optional[str] = None

class UpdateTickersRequest(BaseModel):
    tickers: List[str]
    commit_message: Optional[str] = None
    job_id: Optional[str] = None  # For idempotency tracking
    skip_render: Optional[bool] = True  # Default: skip render to prevent auto-deployment

# 1. GITHUB SYNC ENDPOINTS
@APP.post("/admin/sync-ticker-reference-from-github")
def sync_from_github(request: Request):
    """Sync ticker reference data FROM GitHub TO database"""
    require_admin(request)
    
    result = sync_ticker_references_from_github()
    return result

@APP.post("/admin/sync-ticker-reference-to-github")
def sync_to_github(request: Request, body: GitHubSyncRequest):
    """Sync ticker reference data FROM database TO GitHub"""
    require_admin(request)
    
    result = sync_ticker_references_to_github(body.commit_message)
    return result

@APP.post("/admin/update-specific-tickers-on-github")
def update_tickers_on_github(request: Request, body: UpdateTickersRequest):
    """Update only specific tickers on GitHub (for post-processing sync)"""
    require_admin(request)
    
    if not body.tickers:
        return {"status": "error", "message": "No tickers specified"}
    
    result = update_specific_tickers_on_github(body.tickers, body.commit_message)
    return result

# 2. CSV UPLOAD ENDPOINT
@APP.post("/admin/upload-ticker-csv")
async def upload_ticker_csv(request: Request, file: UploadFile = File(...)):
    """Upload CSV file and import ticker reference data"""
    require_admin(request)
    
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    try:
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        result = import_ticker_reference_from_csv_content(csv_content)
        return result
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV file must be UTF-8 encoded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")

# 3. INDIVIDUAL TICKER MANAGEMENT
@APP.get("/admin/ticker-reference/{ticker}")
async def get_ticker_reference_endpoint(request: Request, ticker: str):
    async with TICKER_PROCESSING_LOCK:
        """Get specific ticker reference data"""
        require_admin(request)
        
        ticker_data = get_ticker_reference(ticker)
        if ticker_data:
            return {
                "status": "found",
                "ticker": ticker,
                "data": ticker_data
            }
        else:
            return {
                "status": "not_found",
                "ticker": ticker,
                "message": "Ticker reference not found"
            }

@APP.post("/admin/ticker-reference")
def add_ticker_reference_endpoint(request: Request, body: TickerReferenceRequest):
    """Add or update a single ticker reference manually"""
    require_admin(request)
    
    ticker_data = {
        'ticker': body.ticker,
        'country': body.country,
        'company_name': body.company_name,
        'industry': body.industry,
        'sector': body.sector,
        'sub_industry': body.sub_industry,
        'exchange': body.exchange,
        'currency': body.currency,
        'market_cap_category': body.market_cap_category,
        'yahoo_ticker': body.yahoo_ticker,
        'active': body.active,
        'is_etf': body.is_etf,
        'industry_keyword_1': body.industry_keyword_1,
        'industry_keyword_2': body.industry_keyword_2,
        'industry_keyword_3': body.industry_keyword_3,
        'competitor_1_name': body.competitor_1_name,
        'competitor_1_ticker': body.competitor_1_ticker,
        'competitor_2_name': body.competitor_2_name,
        'competitor_2_ticker': body.competitor_2_ticker,
        'competitor_3_name': body.competitor_3_name,
        'competitor_3_ticker': body.competitor_3_ticker,
        'data_source': 'manual_api'
    }
    
    success = store_ticker_reference(ticker_data)
    
    if success:
        return {
            "status": "success",
            "ticker": body.ticker,
            "message": f"Successfully stored ticker reference for {body.ticker}"
        }
    else:
        return {
            "status": "error",
            "ticker": body.ticker,
            "message": "Failed to store ticker reference"
        }

@APP.delete("/admin/ticker-reference/{ticker}")
def delete_ticker_reference_endpoint(request: Request, ticker: str):
    """Delete (deactivate) a ticker reference"""
    require_admin(request)
    
    success = delete_ticker_reference(ticker)
    
    if success:
        return {
            "status": "success",
            "ticker": ticker,
            "message": f"Successfully deactivated ticker reference for {ticker}"
        }
    else:
        return {
            "status": "error",
            "ticker": ticker,
            "message": "Ticker reference not found or already inactive"
        }

# 4. BULK TICKER OPERATIONS
@APP.get("/admin/ticker-references")
def list_ticker_references(
    request: Request,
    limit: int = Query(default=50, description="Number of records to return"),
    offset: int = Query(default=0, description="Number of records to skip"),
    country: Optional[str] = Query(default=None, description="Filter by country (US, CA, etc.)")
):
    """List ticker references with pagination and filtering"""
    require_admin(request)
    
    if limit > 1000:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 1000")
    
    try:
        ticker_references = get_all_ticker_references(limit, offset, country)
        total_count = count_ticker_references(country)
        
        return {
            "status": "success",
            "data": ticker_references,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total_count,
                "returned": len(ticker_references)
            },
            "filter": {
                "country": country
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve ticker references: {str(e)}")

@APP.get("/admin/ticker-references/stats")
def get_ticker_reference_stats(request: Request):
    """Get statistics about ticker reference data"""
    require_admin(request)
    
    try:
        with db() as conn, conn.cursor() as cur:
            # Total counts by country
            cur.execute("""
                SELECT country, COUNT(*) as count
                FROM ticker_reference
                WHERE active = TRUE
                GROUP BY country
                ORDER BY count DESC
            """)
            by_country = list(cur.fetchall())
            
            # AI enhancement status
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN ai_generated THEN 1 END) as ai_enhanced,
                    COUNT(CASE WHEN industry_keyword_1 IS NOT NULL THEN 1 END) as has_keywords,
                    COUNT(CASE WHEN competitor_1_name IS NOT NULL THEN 1 END) as has_competitors
                FROM ticker_reference
                WHERE active = TRUE
            """)
            ai_stats = dict(cur.fetchone())
            
            # Recent updates
            cur.execute("""
                SELECT COUNT(*) as count
                FROM ticker_reference
                WHERE active = TRUE AND updated_at > NOW() - INTERVAL '7 days'
            """)
            recent_updates = cur.fetchone()["count"]
            
            return {
                "status": "success",
                "stats": {
                    "total_active_tickers": sum(row["count"] for row in by_country),
                    "by_country": by_country,
                    "ai_enhancement": {
                        "total_tickers": ai_stats["total"],
                        "ai_enhanced": ai_stats["ai_enhanced"],
                        "has_industry_keywords": ai_stats["has_keywords"],
                        "has_competitors": ai_stats["has_competitors"]
                    },
                    "recent_updates_7_days": recent_updates
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# 5. TICKER VALIDATION AND TESTING
@APP.post("/admin/validate-ticker-format")
def validate_ticker_endpoint(request: Request, ticker: str = Body(..., embed=True)):
    """Test ticker format validation"""
    require_admin(request)
    
    normalized = normalize_ticker_format(ticker)
    is_valid = validate_ticker_format(normalized)
    exchange_info = get_ticker_exchange_info(normalized)
    
    return {
        "status": "success",
        "original_ticker": ticker,
        "normalized_ticker": normalized,
        "is_valid": is_valid,
        "exchange_info": exchange_info
    }

@APP.post("/admin/test-ticker-validation")
def test_ticker_validation_endpoint(request: Request):
    """Run comprehensive ticker validation tests"""
    require_admin(request)
    
    test_results = test_ticker_validation()
    return {
        "status": "success",
        "test_results": test_results
    }

# 6. ENHANCED METADATA INTEGRATION
@APP.post("/admin/enhance-ticker-with-ai")
def enhance_ticker_with_ai_endpoint(request: Request, ticker: str = Body(..., embed=True)):
    """Enhance a specific ticker with AI-generated metadata"""
    require_admin(request)
    
    if not OPENAI_API_KEY:
        return {"status": "error", "message": "OpenAI API key not configured"}
    
    try:
        # Get current ticker data
        ticker_data = get_ticker_reference(ticker)
        if not ticker_data:
            return {
                "status": "error",
                "message": f"Ticker {ticker} not found in reference database"
            }
        
        # Generate enhanced metadata
        enhanced_metadata = get_or_create_enhanced_ticker_metadata(ticker)
        
        return {
            "status": "success",
            "ticker": ticker,
            "original_data": ticker_data,
            "enhanced_metadata": enhanced_metadata,
            "message": f"Successfully enhanced {ticker} with AI-generated metadata"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "ticker": ticker,
            "message": f"Failed to enhance ticker: {str(e)}"
        }

# 7. WORKFLOW INTEGRATION ENDPOINTS
@APP.post("/admin/prepare-ticker-for-processing")
def prepare_ticker_for_processing(request: Request, ticker: str = Body(..., embed=True)):
    """Complete workflow: Sync from GitHub -> Get ticker metadata -> Ready for processing"""
    require_admin(request)
    
    try:
        # Step 1: Sync latest data from GitHub
        LOG.info(f"Preparing {ticker} for processing - syncing from GitHub")
        sync_result = sync_ticker_references_from_github()
        
        if sync_result["status"] != "success":
            return {
                "status": "error",
                "message": f"GitHub sync failed: {sync_result.get('message', 'Unknown error')}"
            }
        
        # Step 2: Get ticker data with enhancement
        ticker_metadata = get_or_create_enhanced_ticker_metadata(ticker)
        
        # Step 3: Check if ticker has required data
        ticker_reference = get_ticker_reference(ticker)
        
        return {
            "status": "ready",
            "ticker": ticker,
            "github_sync": {
                "imported": sync_result.get("database_import", {}).get("imported", 0),
                "updated": sync_result.get("database_import", {}).get("updated", 0)
            },
            "ticker_reference": ticker_reference,
            "enhanced_metadata": ticker_metadata,
            "message": f"Ticker {ticker} is ready for processing"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "ticker": ticker,
            "message": f"Preparation failed: {str(e)}"
        }

@APP.post("/admin/finalize-ticker-processing")
def finalize_ticker_processing(request: Request, body: UpdateTickersRequest):
    """Complete workflow: Update specific processed tickers back to GitHub"""
    require_admin(request)
    
    if not body.tickers:
        return {"status": "error", "message": "No tickers specified"}
    
    try:
        # Update the processed tickers back to GitHub
        result = update_specific_tickers_on_github(body.tickers, body.commit_message)
        
        return {
            "status": result["status"],
            "processed_tickers": body.tickers,
            "github_update": result,
            "message": f"Finalized processing for {len(body.tickers)} tickers"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "tickers": body.tickers,
            "message": f"Finalization failed: {str(e)}"
        }

@APP.post("/admin/sync-processed-tickers-to-github")
def sync_processed_tickers_to_github(request: Request, body: UpdateTickersRequest):
    """Prepare processed ticker data for sync (no GitHub commit)"""
    require_admin(request)
    
    if not body.tickers:
        return {"status": "error", "message": "No tickers specified"}
    
    LOG.info(f"=== PREPARING {len(body.tickers)} PROCESSED TICKERS FOR SYNC ===")
    
    try:
        # Get current successfully processed tickers that have AI enhancements
        successfully_enhanced_tickers = []
        
        with db() as conn, conn.cursor() as cur:
            # Check which tickers actually have AI enhancements
            cur.execute("""
                SELECT DISTINCT ticker 
                FROM ticker_reference 
                WHERE ticker = ANY(%s) 
                AND ai_generated = TRUE 
                AND (industry_keyword_1 IS NOT NULL OR competitor_1_name IS NOT NULL)
            """, (body.tickers,))
            
            successfully_enhanced_tickers = [row["ticker"] for row in cur.fetchall()]
        
        if not successfully_enhanced_tickers:
            return {
                "status": "no_changes", 
                "message": "No tickers found with AI enhancements",
                "requested_tickers": body.tickers
            }
        
        # Export the enhanced data to CSV format
        export_result = export_ticker_references_to_csv()
        
        return {
            "status": "ready_for_sync",
            "requested_tickers": body.tickers,
            "enhanced_tickers": successfully_enhanced_tickers,
            "csv_content": export_result.get("csv_content", ""),
            "message": f"Prepared {len(successfully_enhanced_tickers)} enhanced tickers for sync"
        }
        
    except Exception as e:
        LOG.error(f"Failed to prepare processed tickers: {e}")
        return {
            "status": "error",
            "message": f"Preparation failed: {str(e)}",
            "tickers": body.tickers
        }

@APP.get("/admin/memory")
async def get_memory_info():
    """Get current memory usage"""
    return memory_monitor.get_memory_info()

@APP.get("/admin/debug/digest-check/{ticker}")
async def debug_digest_check(request: Request, ticker: str):
    """Debug endpoint to check digest data for a specific ticker"""
    require_admin(request)

    try:
        with db() as conn, conn.cursor() as cur:
            # Check if tables exist first (safe for fresh database)
            try:
                cur.execute("SELECT 1 FROM articles LIMIT 1")
                cur.execute("SELECT 1 FROM ticker_articles LIMIT 1")
            except Exception as e:
                LOG.info(f"📋 Tables don't exist yet (fresh database) - digest-check skipped: {e}")
                return {"status": "skipped", "message": "Tables don't exist yet - cannot check digest", "ticker": ticker}

            # Check articles for this ticker in the last 24 hours
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

            cur.execute("""
                SELECT COUNT(*) as total_articles,
                       COUNT(CASE WHEN ta.ai_summary IS NOT NULL THEN 1 END) as with_ai_summary,
                       COUNT(CASE WHEN a.scraped_content IS NOT NULL THEN 1 END) as with_content,
                       COUNT(CASE WHEN ta.sent_in_digest = TRUE THEN 1 END) as already_sent
                FROM articles a
                JOIN ticker_articles ta ON a.id = ta.article_id
                WHERE ta.ticker = %s
                AND ta.found_at >= %s
            """, (ticker, cutoff))

            stats = dict(cur.fetchone())

            # Get sample articles
            cur.execute("""
                SELECT a.id, a.title, ta.category,
                       ta.ai_summary IS NOT NULL as has_ai_summary,
                       a.scraped_content IS NOT NULL as has_content,
                       ta.sent_in_digest
                FROM articles a
                JOIN ticker_articles ta ON a.id = ta.article_id
                WHERE ta.ticker = %s
                AND ta.found_at >= %s
                ORDER BY ta.found_at DESC
                LIMIT 5
            """, (ticker, cutoff))

            sample_articles = [dict(row) for row in cur.fetchall()]

            return {
                "ticker": ticker,
                "time_window": "24 hours",
                "stats": stats,
                "sample_articles": sample_articles,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    except Exception as e:
        return {
            "error": str(e),
            "ticker": ticker
        }

@APP.get("/admin/memory-snapshots")
async def get_memory_snapshots():
    """Get all memory snapshots"""
    return {
        "snapshots": memory_monitor.snapshots,
        "tracemalloc_top": memory_monitor.get_tracemalloc_top() if memory_monitor.tracemalloc_started else []
    }

@APP.post("/admin/force-cleanup")
async def force_cleanup():
    """Force garbage collection"""
    cleanup_result = memory_monitor.force_garbage_collection()
    return {
        "status": "success", 
        "cleanup_result": cleanup_result
    }

@APP.post("/admin/commit-csv-to-github")
async def commit_csv_to_github_endpoint():
    """HTTP endpoint to export DB to CSV and commit to GitHub"""
    try:
        # Step 1: Export database to CSV
        export_result = export_ticker_references_to_csv()
        if export_result["status"] != "success":
            return export_result
        
        # Step 2: Commit CSV to GitHub
        commit_result = commit_csv_to_github(export_result["csv_content"])
        
        return {
            "status": commit_result["status"],
            "export_info": {
                "ticker_count": export_result["ticker_count"],
                "csv_size": len(export_result["csv_content"])
            },
            "github_commit": commit_result,
            "message": f"Exported {export_result['ticker_count']} tickers and committed to GitHub"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@APP.post("/admin/update-domain-names")
async def update_domain_formal_names(request: Request):
    """Batch update domain formal names using Claude API"""
    require_admin(request)

    try:
        LOG.info("=== BATCH UPDATING DOMAIN FORMAL NAMES ===")

        # Step 1: Fetch all domains from database
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT domain FROM domain_names ORDER BY domain;")
            domains = [row["domain"] for row in cur.fetchall()]

        if not domains:
            return {"status": "error", "message": "No domains found in database"}

        LOG.info(f"Found {len(domains)} domains to process")

        # Step 2: Build Claude API prompt
        domain_list = "\n".join([f"- {domain}" for domain in domains])

        prompt = f"""You are a domain name expert. Below is a list of domain names. For EACH domain, provide ONLY the formal brand/publication name as it should appear in professional contexts.

Rules:
1. Return EXACTLY one name per domain
2. Use proper capitalization (e.g., "The Wall Street Journal" not "wall street journal")
3. For news outlets, include "The" if official (e.g., "The Guardian")
4. For companies, use official brand name (e.g., "Bloomberg" not "Bloomberg LP")
5. Do NOT include domain extensions (.com, .net, etc.) in the formal name
6. If a domain is unknown or generic, return the domain name as-is with proper capitalization
7. IMPORTANT: Use only basic ASCII characters (A-Z, a-z, 0-9, spaces, hyphens). Convert accented characters to their base form (é→e, ü→u, ñ→n, etc.)

Format your response as a JSON object where keys are domains and values are formal names:
{{
  "domain1.com": "Formal Name 1",
  "domain2.com": "Formal Name 2"
}}

Domains:
{domain_list}
"""

        # Step 3: Call Claude API
        LOG.info("Calling Claude API...")

        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text
        LOG.info("Claude API response received")
        LOG.info(f"Response preview (first 200 chars): {response_text[:200]}")

        # Step 4: Extract JSON from response
        import json
        import re

        json_content = None

        # Try to extract JSON from code block first
        if '```json' in response_text:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_content = json_match.group(1).strip()
                LOG.info("Extracted JSON from ```json code block")
            else:
                LOG.warning("Found ```json marker but couldn't extract content")

        # If no code block, try to find raw JSON object
        if not json_content and '{' in response_text:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_content = json_match.group(0).strip()
                LOG.info("Extracted raw JSON object")
            else:
                LOG.warning("Found '{' but couldn't extract JSON object")

        # Validate we have content
        if not json_content or not json_content.strip():
            LOG.error(f"No valid JSON content extracted. Full response:\n{response_text}")
            return {"status": "error", "message": "Could not extract valid JSON from Claude response", "raw_response": response_text[:500]}

        LOG.info(f"JSON content length: {len(json_content)} chars")
        LOG.info(f"JSON content preview: {json_content[:200]}")

        # Parse JSON
        try:
            domain_mappings = json.loads(json_content)
        except json.JSONDecodeError as e:
            LOG.error(f"JSON parsing failed: {e}")
            LOG.error(f"JSON content that failed to parse:\n{json_content[:1000]}")
            return {"status": "error", "message": f"Invalid JSON from Claude: {str(e)}", "json_preview": json_content[:500]}

        # Step 5: Update database
        LOG.info("Updating database...")
        update_count = 0
        error_count = 0

        with db() as conn, conn.cursor() as cur:
            for domain in domains:
                formal_name = domain_mappings.get(domain)

                if not formal_name:
                    LOG.warning(f"Missing formal name for: {domain}")
                    error_count += 1
                    continue

                cur.execute("""
                    UPDATE domain_names
                    SET formal_name = %s
                    WHERE domain = %s
                """, (formal_name, domain))

                update_count += 1
                LOG.info(f"✅ {domain} → {formal_name}")

        LOG.info(f"Update complete: {update_count} updated, {error_count} errors")

        return {
            "status": "success",
            "total_domains": len(domains),
            "updated": update_count,
            "errors": error_count
        }

    except Exception as e:
        LOG.error(f"Domain name update failed: {e}")
        LOG.error(f"Error details: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

@APP.post("/admin/safe-incremental-commit")
async def safe_incremental_commit(request: Request, body: UpdateTickersRequest):
    """Safely commit individual tickers as they complete processing"""
    require_admin(request)

    if not body.tickers:
        return {"status": "error", "message": "No tickers specified"}

    LOG.info(f"=== SAFE INCREMENTAL COMMIT: {len(body.tickers)} TICKERS ===")

    try:
        # Step 1: Verify all tickers have AI-generated metadata
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT ticker, ai_generated, industry_keyword_1, competitor_1_name,
                       ai_enhanced_at, updated_at
                FROM ticker_reference
                WHERE ticker = ANY(%s)
                ORDER BY ticker
            """, (body.tickers,))

            ticker_status = {}
            for row in cur.fetchall():
                ticker = row["ticker"]
                has_ai_data = (row["ai_generated"] and
                             (row["industry_keyword_1"] or row["competitor_1_name"]))
                ticker_status[ticker] = {
                    "has_ai_metadata": has_ai_data,
                    "ai_enhanced_at": row["ai_enhanced_at"],
                    "updated_at": row["updated_at"]
                }

        # Step 2: Filter to only commit tickers with AI metadata
        valid_tickers = [t for t, status in ticker_status.items() if status["has_ai_metadata"]]
        invalid_tickers = [t for t, status in ticker_status.items() if not status["has_ai_metadata"]]

        if not valid_tickers:
            return {
                "status": "no_changes",
                "message": "No tickers have AI-generated metadata to commit",
                "invalid_tickers": invalid_tickers,
                "ticker_status": ticker_status
            }

        # Step 3: Create backup before commit (include job_id for idempotency)
        # [skip render] controlled by skip_render flag (default: True)
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id_suffix = f" [job:{body.job_id[:8]}]" if body.job_id else ""

        # Add [skip render] prefix only if skip_render=True
        skip_prefix = "[skip render] " if body.skip_render else ""
        commit_message = f"{skip_prefix}Incremental update: {', '.join(valid_tickers)} - {backup_timestamp}{job_id_suffix}"

        if not body.skip_render:
            LOG.warning(f"⚠️ RENDER DEPLOYMENT WILL BE TRIGGERED by this commit")
            LOG.warning(f"   Commit message: {commit_message}")
        else:
            LOG.info(f"✅ [skip render] enabled - no deployment will be triggered")

        # Step 4: Export and commit with retry logic
        LOG.info(f"Exporting {len(valid_tickers)} enhanced tickers: {valid_tickers}")
        export_result = export_ticker_references_to_csv()

        if export_result["status"] != "success":
            return {
                "status": "export_failed",
                "message": export_result["message"],
                "valid_tickers": valid_tickers,
                "invalid_tickers": invalid_tickers
            }

        LOG.info(f"Committing to GitHub with message: {commit_message}")

        # Wrap GitHub commit in try/except to make it non-fatal
        try:
            commit_result = commit_csv_to_github(export_result["csv_content"], commit_message)

            if commit_result["status"] == "success":
                # Step 5: Update commit tracking in database (with column existence check)
                try:
                    with db() as conn, conn.cursor() as cur:
                        # Verify column exists before attempting update (bulletproofing for schema mismatches)
                        cur.execute("""
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_schema = 'public'
                            AND table_name = 'ticker_reference'
                            AND column_name = 'last_github_sync'
                        """)

                        if cur.fetchone():
                            cur.execute("""
                                UPDATE ticker_reference
                                SET last_github_sync = %s
                                WHERE ticker = ANY(%s)
                            """, (datetime.now(timezone.utc), valid_tickers))

                            LOG.info(f"✅ Updated GitHub sync timestamp for {len(valid_tickers)} tickers")
                        else:
                            LOG.warning("⚠️ Column 'last_github_sync' does not exist in ticker_reference table")
                            LOG.warning("   CSV committed successfully, but sync timestamp not recorded")
                            LOG.warning("   Run: ALTER TABLE ticker_reference ADD COLUMN last_github_sync TIMESTAMP;")

                except Exception as db_error:
                    LOG.error(f"⚠️ Failed to update last_github_sync timestamp: {db_error}")
                    LOG.warning("   CSV was committed to GitHub successfully, but DB timestamp update failed")
                    # Don't fail the entire operation - GitHub commit succeeded

        except Exception as commit_error:
            LOG.error(f"⚠️ GitHub commit failed (non-fatal): {commit_error}")
            commit_result = {
                "status": "error",
                "message": f"GitHub commit failed: {str(commit_error)}"
            }

        return {
            "status": commit_result["status"],
            "message": f"Successfully committed {len(valid_tickers)} tickers",
            "committed_tickers": valid_tickers,
            "skipped_tickers": invalid_tickers,
            "ticker_status": ticker_status,
            "export_info": {
                "total_tickers_in_csv": export_result["ticker_count"],
                "csv_size": len(export_result["csv_content"])
            },
            "commit_info": commit_result
        }

    except Exception as e:
        LOG.error(f"Safe incremental commit failed: {e}")
        LOG.error(f"Error details: {traceback.format_exc()}")
        return {
            "status": "error",
            "message": f"Incremental commit failed: {str(e)}",
            "requested_tickers": body.tickers
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
        # Initialize feeds using NEW ARCHITECTURE V2
        ensure_schema()
        for ticker in body.tickers:
            metadata = ticker_manager.get_or_create_metadata(ticker)
            create_feeds_for_ticker_new_architecture(ticker, metadata)
        
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
