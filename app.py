import os
import sys
import time
import logging
import traceback
import hashlib
import re
import pytz
import json
import secrets
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
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr

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

# REMOVED: Playwright no longer used (Scrapfly-only as of Oct 2025)
# from playwright.async_api import async_playwright
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

# ============================================================================
# AIOHTTP CONNECTION MANAGEMENT - Prevents Connection Exhaustion & Deadlock
# ============================================================================
# Problem: Creating new session per API call leads to:
# - 180+ sessions with 3 concurrent tickers
# - File descriptor exhaustion → server freeze
# - TCP handshake overhead (slow)
#
# Solution: Thread-local connectors + thread-local sessions
# - One connector per thread (isolated event loop per thread)
# - One session per thread (reused for all calls in that thread)
# - Cleanup on thread exit (prevents resource leaks)
#
# FIX (Oct 2025): Made connector thread-local to prevent "Event loop is closed"
# errors when multiple ThreadPoolExecutor workers use asyncio.run() concurrently.
# Each thread gets its own connector bound to its own event loop.
# ============================================================================

# Thread-local storage for connectors and sessions (one per thread)
_thread_local = threading.local()

def _get_or_create_connector():
    """
    Get or create HTTP connector for current thread (thread-safe lazy initialization).

    Each ThreadPoolExecutor worker thread gets its own connector, which prevents
    "Event loop is closed" errors when multiple threads run asyncio.run() concurrently.

    The connector is bound to the thread's event loop and is automatically cleaned up
    when the thread exits.
    """
    if not hasattr(_thread_local, 'connector') or _thread_local.connector is None:
        import threading
        thread_name = threading.current_thread().name
        _thread_local.connector = aiohttp.TCPConnector(
            limit=100,              # Total connections across all hosts (per thread)
            limit_per_host=30,      # Max 30 concurrent per host (OpenAI/Anthropic/ScrapFly)
            ttl_dns_cache=300,      # Cache DNS for 5 minutes
            enable_cleanup_closed=True  # Clean up closed connections
        )
        LOG.debug(f"🔌 Created new HTTP connector for thread: {thread_name}")
    return _thread_local.connector

def get_http_session() -> aiohttp.ClientSession:
    """
    Get or create aiohttp session for current thread.

    Returns cached session if exists, creates new one if not.
    Session is reused for all HTTP calls in the thread (connection pooling).

    Benefits:
    - Prevents connection exhaustion (max 100 connections per thread)
    - Faster (reuses TCP connections, no handshake overhead)
    - Lower memory usage (one session per thread vs hundreds)
    - Thread-safe (each thread has its own session + connector)
    """
    if not hasattr(_thread_local, 'session') or _thread_local.session is None or _thread_local.session.closed:
        _thread_local.session = aiohttp.ClientSession(
            connector=_get_or_create_connector(),
            connector_owner=False  # Don't close connector when session closes
        )
    return _thread_local.session

async def cleanup_http_session():
    """
    Close thread-local session and connector (call before thread exits).

    This ensures proper cleanup of HTTP resources when a ThreadPoolExecutor
    worker thread completes processing a job.
    """
    # Close session first
    if hasattr(_thread_local, 'session') and _thread_local.session is not None:
        if not _thread_local.session.closed:
            await _thread_local.session.close()
        _thread_local.session = None

    # Close connector second (session must be closed first)
    if hasattr(_thread_local, 'connector') and _thread_local.connector is not None:
        if not _thread_local.connector.closed:
            await _thread_local.connector.close()
        _thread_local.connector = None

# ============================================================================
import yfinance as yf
from collections import deque

# Global session for OpenAI API calls with retries
_openai_session = None

# Polygon.io rate limiter (5 calls/minute for free tier)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
POLYGON_RATE_LIMIT = 5  # calls per minute
POLYGON_CALL_TIMES = deque(maxlen=POLYGON_RATE_LIMIT)  # Track last N call times

import asyncio

# Global ticker processing lock
TICKER_PROCESSING_LOCK = asyncio.Lock()

# Concurrency Configuration and Semaphores
SCRAPE_BATCH_SIZE = int(os.getenv("SCRAPE_BATCH_SIZE", "5"))
OPENAI_MAX_CONCURRENCY = int(os.getenv("OPENAI_MAX_CONCURRENCY", "5"))
CLAUDE_MAX_CONCURRENCY = int(os.getenv("CLAUDE_MAX_CONCURRENCY", "5"))  # Claude concurrency
# REMOVED: Playwright no longer used (Scrapfly-only as of Oct 2025)
# PLAYWRIGHT_MAX_CONCURRENCY = int(os.getenv("PLAYWRIGHT_MAX_CONCURRENCY", "5"))
SCRAPFLY_MAX_CONCURRENCY = int(os.getenv("SCRAPFLY_MAX_CONCURRENCY", "5"))
TRIAGE_MAX_CONCURRENCY = int(os.getenv("TRIAGE_MAX_CONCURRENCY", "5"))
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))  # Parallel ticker processing

# Global semaphores for concurrent processing (thread-safe, work across event loops)
# These work in both sync and async contexts (with minor blocking during acquisition)
OPENAI_SEM = BoundedSemaphore(OPENAI_MAX_CONCURRENCY)
CLAUDE_SEM = BoundedSemaphore(CLAUDE_MAX_CONCURRENCY)
# REMOVED: Playwright no longer used (Scrapfly-only as of Oct 2025)
# PLAYWRIGHT_SEM = BoundedSemaphore(PLAYWRIGHT_MAX_CONCURRENCY)
SCRAPFLY_SEM = BoundedSemaphore(SCRAPFLY_MAX_CONCURRENCY)
TRIAGE_SEM = BoundedSemaphore(TRIAGE_MAX_CONCURRENCY)

# NOTE: Removed asyncio semaphores (2025-10-07)
# Reason: asyncio.Semaphore binds to specific event loop, causing errors when
# using asyncio.run() in ThreadPoolExecutor (job queue system).
# Solution: Use threading.BoundedSemaphore which works across threads and event loops.
# Trade-off: Tiny blocking overhead (~microseconds) vs 100% async, but negligible impact.

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
APP = FastAPI(title="StockDigest Stock Intelligence Platform")

# Templates setup for landing page
templates = Jinja2Templates(directory="templates")
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
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
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

# Legal document versions
TERMS_VERSION = "1.0"
PRIVACY_VERSION = "1.0"
TERMS_LAST_UPDATED = "October 7, 2025"
PRIVACY_LAST_UPDATED = "October 7, 2025"

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

# Domains that should skip TIER 1 (requests) and go directly to TIER 2 (Scrapfly)
# These are typically JavaScript-heavy sites that timeout on standard requests
# UPDATED 2025-10-06: Now skips to Scrapfly (Playwright commented out)
SKIP_TIER1_DOMAINS = {
    'businesswire.com',  # Slow to load, JavaScript-heavy
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
# Database helpers with Connection Pooling
# ------------------------------------------------------------------------------

# Global connection pool (initialized at startup)
DB_POOL = None

def init_connection_pool():
    """Initialize connection pool at application startup with retry logic"""
    global DB_POOL
    if DB_POOL is not None:
        LOG.warning("Connection pool already initialized")
        return

    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")

    try:
        from psycopg_pool import ConnectionPool
    except ImportError:
        LOG.error("❌ psycopg_pool not installed! Install with: pip install psycopg[pool]")
        raise

    # Retry logic: 3 attempts with exponential backoff
    max_attempts = 3
    backoff_seconds = [2, 5, 10]  # Wait 2s, 5s, 10s between attempts

    for attempt in range(1, max_attempts + 1):
        try:
            LOG.info(f"🔄 Connection pool initialization attempt {attempt}/{max_attempts}...")

            DB_POOL = ConnectionPool(
                DATABASE_URL,
                min_size=5,      # Minimum connections to keep open
                max_size=80,     # Maximum connections (under 100 DB limit for Basic-1GB)
                timeout=30,      # Wait up to 30s for a connection
                max_idle=300,    # Close idle connections after 5 minutes
                kwargs={"row_factory": dict_row}
            )

            # Test the pool with timeout
            LOG.info("   Testing pool connection...")
            with DB_POOL.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")

            LOG.info(f"✅ Database connection pool initialized (min: 5, max: 80) on attempt {attempt}")
            return  # Success!

        except Exception as e:
            error_msg = str(e)
            LOG.error(f"❌ Pool initialization attempt {attempt}/{max_attempts} failed: {error_msg}")

            # Clean up failed pool
            if DB_POOL is not None:
                try:
                    DB_POOL.close()
                except:
                    pass
                DB_POOL = None

            # If not last attempt, wait and retry
            if attempt < max_attempts:
                wait_time = backoff_seconds[attempt - 1]
                LOG.warning(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # Last attempt failed - raise exception to fail startup
                LOG.error(f"💥 CRITICAL: Failed to initialize connection pool after {max_attempts} attempts")
                LOG.error(f"   Database may be down or unreachable")
                LOG.error(f"   Last error: {error_msg}")
                raise RuntimeError(f"Cannot start application without database connection pool (tried {max_attempts} times)")

def close_connection_pool():
    """Close connection pool at application shutdown"""
    global DB_POOL
    if DB_POOL is not None:
        try:
            DB_POOL.close()
            LOG.info("✅ Database connection pool closed")
            DB_POOL = None
        except Exception as e:
            LOG.error(f"❌ Error closing connection pool: {e}")

@contextmanager
def db():
    """Get database connection from pool (or create direct connection if pool not initialized)"""
    global DB_POOL

    # Use pool if available (production)
    if DB_POOL is not None:
        with DB_POOL.connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    else:
        # Fallback to direct connection (development/testing)
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

def with_deadlock_retry(max_retries=100):
    """
    Decorator to automatically retry database operations on deadlock detection.

    PostgreSQL deadlocks are transient - one transaction is automatically killed
    to break the cycle, and retrying immediately almost always succeeds.

    Args:
        max_retries: Maximum retry attempts (default 100, effectively unlimited for deadlocks)

    Usage:
        @with_deadlock_retry()
        def my_db_operation(...):
            with db() as conn:
                # database operations
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except psycopg.errors.DeadlockDetected as e:
                    if attempt < max_retries - 1:
                        # Exponential backoff capped at 1 second
                        delay = min(0.1 * (2 ** attempt), 1.0)
                        LOG.warning(
                            f"⚠️ Deadlock detected in {func.__name__} (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {delay:.1f}s: {str(e)[:100]}"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        LOG.error(f"💀 Deadlock in {func.__name__} failed after {max_retries} retries")
                        raise
            return None  # Should never reach here
        return wrapper
    return decorator

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
                    company_name VARCHAR(255),
                    retain_days INTEGER DEFAULT 90,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );

                -- Add company_name column if it doesn't exist (migration)
                ALTER TABLE feeds ADD COLUMN IF NOT EXISTS company_name VARCHAR(255);

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

                -- Add relevance scoring columns for industry article gate (Oct 2025)
                ALTER TABLE ticker_articles ADD COLUMN IF NOT EXISTS relevance_score NUMERIC(3,1);
                ALTER TABLE ticker_articles ADD COLUMN IF NOT EXISTS relevance_reason TEXT;
                ALTER TABLE ticker_articles ADD COLUMN IF NOT EXISTS is_rejected BOOLEAN DEFAULT FALSE;

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

                -- Beta users table for landing page signups
                CREATE TABLE IF NOT EXISTS beta_users (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    ticker1 VARCHAR(10) NOT NULL,
                    ticker2 VARCHAR(10) NOT NULL,
                    ticker3 VARCHAR(10) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    status VARCHAR(50) DEFAULT 'pending',
                    terms_version VARCHAR(10) DEFAULT '1.0',
                    terms_accepted_at TIMESTAMPTZ,
                    privacy_version VARCHAR(10) DEFAULT '1.0',
                    privacy_accepted_at TIMESTAMPTZ
                );

                CREATE INDEX IF NOT EXISTS idx_beta_users_status ON beta_users(status);
                CREATE INDEX IF NOT EXISTS idx_beta_users_created_at ON beta_users(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_beta_users_email ON beta_users(email);

                -- Add terms versioning columns to existing beta_users table (safe migration)
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                 WHERE table_name='beta_users' AND column_name='terms_version') THEN
                        ALTER TABLE beta_users
                        ADD COLUMN terms_version VARCHAR(10) DEFAULT '1.0',
                        ADD COLUMN terms_accepted_at TIMESTAMPTZ,
                        ADD COLUMN privacy_version VARCHAR(10) DEFAULT '1.0',
                        ADD COLUMN privacy_accepted_at TIMESTAMPTZ;
                    END IF;
                END $$;

                -- Create index AFTER adding columns
                CREATE INDEX IF NOT EXISTS idx_beta_users_terms_version ON beta_users(terms_version);

                -- Unsubscribe tokens table
                CREATE TABLE IF NOT EXISTS unsubscribe_tokens (
                    id SERIAL PRIMARY KEY,
                    user_email VARCHAR(255) NOT NULL,
                    token VARCHAR(64) UNIQUE NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    used_at TIMESTAMPTZ,
                    ip_address VARCHAR(45),
                    user_agent TEXT,
                    FOREIGN KEY (user_email) REFERENCES beta_users(email) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_unsubscribe_tokens_token ON unsubscribe_tokens(token);
                CREATE INDEX IF NOT EXISTS idx_unsubscribe_tokens_email ON unsubscribe_tokens(user_email);
                CREATE INDEX IF NOT EXISTS idx_unsubscribe_tokens_used ON unsubscribe_tokens(used_at) WHERE used_at IS NULL;

                -- Daily Email Queue: Workflow for beta user emails
                CREATE TABLE IF NOT EXISTS email_queue (
                    ticker VARCHAR(10) PRIMARY KEY,
                    company_name VARCHAR(255),
                    recipients TEXT[],  -- PostgreSQL array of email addresses
                    email_html TEXT,    -- Contains {{UNSUBSCRIBE_TOKEN}} placeholder
                    email_subject VARCHAR(500),
                    article_count INTEGER,
                    status VARCHAR(50) NOT NULL DEFAULT 'queued',
                    previous_status VARCHAR(50),  -- Tracks status before cancellation (for smart restore)
                    error_message TEXT,
                    is_production BOOLEAN DEFAULT TRUE,
                    heartbeat TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    sent_at TIMESTAMPTZ,
                    CONSTRAINT valid_status CHECK (status IN ('queued', 'processing', 'ready', 'failed', 'sent', 'cancelled'))
                );

                CREATE INDEX IF NOT EXISTS idx_email_queue_status ON email_queue(status);
                CREATE INDEX IF NOT EXISTS idx_email_queue_sent_at ON email_queue(sent_at);
                CREATE INDEX IF NOT EXISTS idx_email_queue_created_at ON email_queue(created_at);
                CREATE INDEX IF NOT EXISTS idx_email_queue_is_production ON email_queue(is_production);
                CREATE INDEX IF NOT EXISTS idx_email_queue_heartbeat ON email_queue(heartbeat) WHERE status = 'processing';

                -- System Configuration: UI-configurable settings
                CREATE TABLE IF NOT EXISTS system_config (
                    key VARCHAR(100) PRIMARY KEY,
                    value TEXT NOT NULL,
                    description TEXT,
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_by VARCHAR(100)
                );

                -- Initialize default lookback window (1 day = 1440 minutes)
                INSERT INTO system_config (key, value, description, updated_by)
                VALUES ('lookback_minutes', '1440', 'Article lookback window in minutes (1 day default)', 'system')
                ON CONFLICT (key) DO NOTHING;

                CREATE INDEX IF NOT EXISTS idx_system_config_key ON system_config(key);
            """)

    LOG.info("✅ Complete database schema created successfully with NEW ARCHITECTURE + JOB QUEUE + BETA USERS")

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

@with_deadlock_retry()
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

@with_deadlock_retry()
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

@with_deadlock_retry()
def update_ticker_article_summary(ticker: str, article_id: int, ai_summary: str, ai_model: str) -> None:
    """Update ticker-specific AI summary (POV-based analysis)"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            UPDATE ticker_articles
            SET ai_summary = %s, ai_model = %s
            WHERE ticker = %s AND article_id = %s
        """, (ai_summary, ai_model, ticker, article_id))

@with_deadlock_retry()
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
                                competitor_ticker: str = None, company_name: str = None, retain_days: int = 90) -> int:
    """Insert/update feed in new architecture - NO CATEGORY (category is per-relationship)"""
    with db() as conn, conn.cursor() as cur:
        try:
            # Insert or get existing feed - NEVER overwrite existing feeds
            cur.execute("""
                INSERT INTO feeds (url, name, search_keyword, competitor_ticker, company_name, retain_days)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET
                    active = TRUE,
                    company_name = COALESCE(EXCLUDED.company_name, feeds.company_name),
                    updated_at = NOW()
                RETURNING id;
            """, (url, name, search_keyword, competitor_ticker, company_name, retain_days))

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
    """Create feeds using new architecture with per-relationship categories

    Supports fallback configs (no full ticker data):
    - Always creates Google News feeds (works with any text)
    - Only creates Yahoo Finance feeds if has_full_config=True
    """
    feeds_created = []
    company_name = metadata.get("company_name", ticker)
    use_google_only = metadata.get("use_google_only", False)
    has_full_config = metadata.get("has_full_config", True)  # Default to True for backward compatibility

    if use_google_only or not has_full_config:
        LOG.info(f"🔄 Creating feeds for {ticker} using GOOGLE NEWS ONLY (no full config available)")
    else:
        LOG.info(f"🔄 Creating feeds for {ticker} using NEW ARCHITECTURE (Google News + Yahoo Finance)")

    # 1. Company feeds - ALWAYS create Google News, optionally create Yahoo Finance
    company_feeds = [
        {
            "url": f"https://news.google.com/rss/search?q=\"{company_name.replace(' ', '%20')}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
            "name": f"Google News: {company_name}",
            "search_keyword": company_name,
            "source": "google"
        }
    ]

    # Only add Yahoo Finance if we have full config
    if not use_google_only and has_full_config:
        company_feeds.append({
            "url": f"https://finance.yahoo.com/rss/headline?s={ticker}",
            "name": f"Yahoo Finance: {ticker}",
            "search_keyword": ticker,
            "source": "yahoo"
        })
    else:
        LOG.info(f"⏭️ Skipping Yahoo Finance feed for {ticker} (using Google News only)")

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
                    "config": {"category": "company", "name": feed_config["name"], "source": feed_config["source"]}
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
    # Supports competitors WITHOUT tickers (private companies) - Google News only
    competitors = metadata.get("competitors", [])[:3]
    for comp in competitors:
        if isinstance(comp, dict) and comp.get('name'):
            comp_name = comp['name']
            comp_ticker = comp.get('ticker')  # May be None for private companies

            try:
                # ALWAYS create Google News feed (works for any company name)
                feed_id = upsert_feed_new_architecture(
                    url=f"https://news.google.com/rss/search?q=\"{comp_name.replace(' ', '%20')}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
                    name=f"Google News: {comp_name}",  # Neutral name (no "Competitor:" prefix)
                    search_keyword=comp_name,
                    competitor_ticker=comp_ticker,  # Can be None
                    company_name=comp_name  # Full company name for display
                )

                # Associate this feed with this ticker as "competitor" category
                if associate_ticker_with_feed_new_architecture(ticker, feed_id, "competitor"):
                    feeds_created.append({
                        "feed_id": feed_id,
                        "config": {"category": "competitor", "name": comp_name, "source": "google"}
                    })

                # ONLY create Yahoo Finance feed if competitor has a ticker (public company)
                if comp_ticker:
                    feed_id = upsert_feed_new_architecture(
                        url=f"https://finance.yahoo.com/rss/headline?s={comp_ticker}",
                        name=f"Yahoo Finance: {comp_ticker}",  # Neutral name (no "Competitor:" prefix)
                        search_keyword=comp_name,
                        competitor_ticker=comp_ticker,
                        company_name=comp_name  # Full company name for display
                    )

                    # Associate this feed with this ticker as "competitor" category
                    if associate_ticker_with_feed_new_architecture(ticker, feed_id, "competitor"):
                        feeds_created.append({
                            "feed_id": feed_id,
                            "config": {"category": "competitor", "name": comp_name, "source": "yahoo"}
                        })
                else:
                    LOG.info(f"⏭️ Competitor {comp_name} has no ticker - using Google News only (private company)")

            except Exception as e:
                LOG.error(f"❌ Failed to create competitor feeds for {comp_name}: {e}")

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

        # Crypto pairs (Yahoo Finance format)
        r'^[A-Z0-9]{2,10}-USD$',                 # Crypto to USD: BTC-USD, ETH-USD, BNB-USD
        r'^[A-Z0-9]{2,10}-[A-Z]{3}$',           # Crypto pairs: BTC-EUR, ETH-GBP

        # Forex pairs (Yahoo Finance format)
        r'^[A-Z]{6}=X$',                         # Forex: EURUSD=X, GBPUSD=X, CADJPY=X
        r'^[A-Z]{3}=X$',                         # Single currency to USD: CAD=X, EUR=X

        # Market indices (Yahoo Finance format)
        r'^\^[A-Z0-9]{2,8}$',                    # Indices: ^GSPC, ^DJI, ^IXIC, ^FTSE

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

    # Remove quotes and invalid characters (keep alphanumeric, dots, dashes, carets, equals)
    # Keep: letters, numbers, dot (.), dash (-), caret (^), equals (=)
    normalized = re.sub(r'[^A-Z0-9.\-\^=]', '', normalized)
    
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

def _wait_for_polygon_rate_limit():
    """
    Rate limiter for Polygon.io API (5 calls/minute for free tier).
    Blocks until safe to make next call.
    """
    now = time.time()

    # Remove calls older than 60 seconds
    while POLYGON_CALL_TIMES and now - POLYGON_CALL_TIMES[0] > 60:
        POLYGON_CALL_TIMES.popleft()

    # If at limit, wait until oldest call expires
    if len(POLYGON_CALL_TIMES) >= POLYGON_RATE_LIMIT:
        sleep_time = 60 - (now - POLYGON_CALL_TIMES[0]) + 1  # +1 second buffer
        if sleep_time > 0:
            LOG.info(f"⏳ Polygon.io rate limit reached, waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    # Record this call
    POLYGON_CALL_TIMES.append(time.time())

def get_stock_context_polygon(ticker: str) -> Optional[Dict]:
    """
    Fetch financial data from Polygon.io as fallback when yfinance fails.
    Free tier: 5 API calls/minute.

    Returns dict with price and yesterday's return, or None on failure.
    """
    if not POLYGON_API_KEY:
        LOG.warning("POLYGON_API_KEY not set, skipping Polygon.io fallback")
        return None

    try:
        _wait_for_polygon_rate_limit()

        # Get previous close (snapshot endpoint)
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
        params = {"apiKey": POLYGON_API_KEY}

        LOG.info(f"📊 Fetching from Polygon.io: {ticker}")
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            LOG.warning(f"Polygon.io API error {response.status_code}: {response.text[:200]}")
            return None

        data = response.json()

        if data.get("status") != "OK" or not data.get("results"):
            LOG.warning(f"Polygon.io no data for {ticker}")
            return None

        result = data["results"][0]

        # Extract price data
        close_price = result.get("c")  # Close price
        open_price = result.get("o")   # Open price

        if not close_price:
            LOG.warning(f"Polygon.io missing price data for {ticker}")
            return None

        # Calculate yesterday's return (close vs open)
        yesterday_return = None
        if close_price and open_price:
            yesterday_return = ((close_price - open_price) / open_price) * 100

        # Build financial data dict (minimal - just what Email #3 needs)
        financial_data = {
            'financial_last_price': float(close_price),
            'financial_price_change_pct': float(yesterday_return) if yesterday_return else None,
            'financial_yesterday_return_pct': float(yesterday_return) if yesterday_return else None,
            'financial_ytd_return_pct': None,  # Not available from Polygon free tier
            'financial_market_cap': None,
            'financial_enterprise_value': None,
            'financial_volume': float(result.get("v")) if result.get("v") else None,
            'financial_avg_volume': None,
            'financial_analyst_target': None,
            'financial_analyst_range_low': None,
            'financial_analyst_range_high': None,
            'financial_analyst_count': None,
            'financial_analyst_recommendation': None,
            'financial_snapshot_date': datetime.now(pytz.timezone('America/Toronto')).strftime('%Y-%m-%d')
        }

        LOG.info(f"✅ Polygon.io data retrieved for {ticker}: Price=${close_price:.2f}, Return={yesterday_return:.2f}%")
        return financial_data

    except Exception as e:
        LOG.error(f"❌ Polygon.io failed for {ticker}: {e}")
        return None

def fetch_company_name_from_polygon(ticker: str) -> Optional[str]:
    """
    Fetch company name from Polygon.io ticker details API.
    Free tier: 5 API calls/minute.
    Returns company name or None if unavailable.
    """
    if not POLYGON_API_KEY:
        LOG.warning("POLYGON_API_KEY not set, skipping Polygon.io company name fetch")
        return None

    try:
        _wait_for_polygon_rate_limit()

        # Ticker details endpoint provides company name
        url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
        params = {"apiKey": POLYGON_API_KEY}

        LOG.info(f"📊 Fetching company name from Polygon.io for {ticker}...")
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            results = data.get('results', {})
            company_name = results.get('name')

            if company_name and company_name != ticker:
                LOG.info(f"✅ Polygon.io company name for {ticker}: {company_name}")
                return company_name
            else:
                LOG.warning(f"⚠️ Polygon.io returned no valid company name for {ticker}")
                return None
        else:
            LOG.warning(f"⚠️ Polygon.io company name fetch failed for {ticker}: HTTP {response.status_code}")
            return None

    except Exception as e:
        LOG.error(f"❌ Polygon.io company name fetch failed for {ticker}: {e}")
        return None

def fetch_company_name_from_yfinance(ticker: str, timeout: int = 10) -> Optional[str]:
    """
    Fetch company name from yfinance as a lightweight fallback.
    Returns longName or shortName, or None if unavailable.
    """
    try:
        LOG.info(f"Fetching company name from yfinance for {ticker}...")
        result = {'data': None, 'error': None}

        def fetch_data():
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                result['data'] = info
            except Exception as e:
                result['error'] = e

        # Start fetch in thread with timeout
        fetch_thread = threading.Thread(target=fetch_data)
        fetch_thread.daemon = True
        fetch_thread.start()
        fetch_thread.join(timeout=timeout)

        if fetch_thread.is_alive():
            LOG.warning(f"yfinance company name fetch timeout for {ticker} after {timeout}s")
            return None

        if result['error']:
            LOG.warning(f"yfinance company name fetch error for {ticker}: {result['error']}")
            return None

        if not result['data']:
            return None

        info = result['data']
        company_name = info.get('longName') or info.get('shortName') or info.get('name')

        if company_name and company_name != ticker:
            LOG.info(f"✅ yfinance company name for {ticker}: {company_name}")
            return company_name
        else:
            LOG.warning(f"⚠️ yfinance returned ticker symbol as company name for {ticker}")
            return None

    except Exception as e:
        LOG.error(f"❌ Failed to fetch company name from yfinance for {ticker}: {e}")
        return None

def get_stock_context(ticker: str, retries: int = 3, timeout: int = 10) -> Optional[Dict]:
    """
    Fetch financial data with yfinance (primary) and Polygon.io (fallback).

    Returns dict with price + yesterday's return (required for Email #3).
    Other fields (market cap, analysts) are optional extras.

    Validation: Only requires price data (not market cap).
    This supports forex (EURUSD=X), indices (^GSPC), crypto (BTC-USD), stocks (AAPL).
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

            # Validate critical fields (RELAXED: only require price, not market cap)
            # This allows forex (EURUSD=X), indices (^GSPC), crypto to work
            if not current_price:
                raise ValueError(f"Missing price data for {ticker}")

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

            mcap_str = format_financial_number(market_cap) if market_cap else "N/A"
            LOG.info(f"✅ yfinance data retrieved for {ticker}: Price=${current_price:.2f}, MCap={mcap_str}")
            return financial_data

        except Exception as e:
            LOG.warning(f"yfinance attempt {attempt + 1}/{retries} failed for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
            else:
                LOG.error(f"❌ yfinance failed after {retries} attempts for {ticker}")
                # Don't return None yet - try Polygon.io fallback

    # yfinance failed, try Polygon.io as fallback
    LOG.info(f"🔄 Trying Polygon.io fallback for {ticker}...")
    polygon_data = get_stock_context_polygon(ticker)

    if polygon_data:
        LOG.info(f"✅ Polygon.io fallback succeeded for {ticker}")
        return polygon_data
    else:
        LOG.error(f"❌ Both yfinance and Polygon.io failed for {ticker}")
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
            LOG.warning(f"⚠️ ticker_reference table doesn't exist yet, using fallback config for {ticker}")
            # Return fallback config instead of None
            return {
                'ticker': ticker,
                'name': ticker,
                'company_name': ticker,
                'industry_keywords': [],
                'competitors': [],
                'sector': 'Unknown',
                'industry': 'Unknown',
                'sub_industry': '',
                'has_full_config': False,
                'use_google_only': True
            }

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

        # Comprehensive error logging for coroutine debugging
        try:
            result = cur.fetchone()
            LOG.info(f"[DB_DEBUG] Raw database result for '{ticker}': {result}")
            LOG.info(f"[DB_DEBUG] Result type: {type(result)}")
        except Exception as e:
            LOG.error(f"[DB_DEBUG] ❌ fetchone() failed with {type(e).__name__}: {e}")
            LOG.error(f"[DB_DEBUG] Cursor type: {type(cur)}")
            LOG.error(f"[DB_DEBUG] Connection type: {type(conn)}")
            import traceback
            LOG.error(f"[DB_DEBUG] Traceback:\n{traceback.format_exc()}")
            raise  # Re-raise to surface the error

        if not result:
            LOG.warning(f"⚠️ No config found for {ticker} - using fallback config (Google News only)")
            # Return fallback config instead of None to prevent crashes
            return {
                'ticker': ticker,
                'name': ticker,
                'company_name': ticker,  # Use ticker as display name
                'industry_keywords': [],
                'competitors': [],
                'sector': 'Unknown',
                'industry': 'Unknown',
                'sub_industry': '',
                'has_full_config': False,  # Flag for downstream logic
                'use_google_only': True    # Skip Yahoo Finance feeds
            }
        
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

def commit_ticker_reference_to_github():
    """
    Daily cron job function: Export ticker_reference from database and commit to GitHub.
    Called by: python app.py commit (Render cron at 6:30 AM EST)

    Unlike incremental commits during job runs, this triggers a Render deployment.
    """
    LOG.info("🔄 ============================================")
    LOG.info("🔄 DAILY GITHUB COMMIT - ticker_reference.csv")
    LOG.info("🔄 ============================================")

    try:
        # Step 1: Export ticker_reference from database to CSV
        LOG.info("📤 Step 1: Exporting ticker_reference from database...")
        export_result = export_ticker_references_to_csv()

        if export_result["status"] != "success":
            LOG.error(f"❌ Export failed: {export_result.get('message', 'Unknown error')}")
            return {
                "status": "error",
                "message": f"CSV export failed: {export_result.get('message')}",
                "rows": 0
            }

        csv_content = export_result["csv_content"]
        ticker_count = export_result.get("ticker_count", 0)
        LOG.info(f"✅ Exported {ticker_count} tickers ({len(csv_content)} chars)")

        # Step 2: Commit to GitHub (WITHOUT [skip render])
        LOG.info("📤 Step 2: Committing to GitHub (triggers deployment)...")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        commit_message = f"Daily ticker reference update - {timestamp}"

        commit_result = commit_csv_to_github(csv_content, commit_message)

        if commit_result["status"] != "success":
            LOG.error(f"❌ GitHub commit failed: {commit_result.get('message', 'Unknown error')}")
            return {
                "status": "error",
                "message": f"GitHub commit failed: {commit_result.get('message')}",
                "rows": ticker_count
            }

        LOG.info(f"✅ GitHub commit successful")
        LOG.info(f"   Commit SHA: {commit_result.get('commit_sha', 'N/A')[:8]}")
        LOG.info(f"   Tickers: {ticker_count}")
        LOG.info(f"   CSV size: {len(csv_content)} chars")
        LOG.info(f"⚠️  Render deployment will start in ~10 seconds")

        return {
            "status": "success",
            "message": "Ticker reference committed to GitHub successfully",
            "rows": ticker_count,
            "commit_sha": commit_result.get("commit_sha"),
            "commit_url": commit_result.get("commit_url"),
            "timestamp": timestamp
        }

    except Exception as e:
        LOG.error(f"❌ Daily GitHub commit failed: {e}")
        LOG.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "rows": 0
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

# ============================================================================
# PLAYWRIGHT REMOVED (Oct 2025) - Replaced with 3-Tier Scrapfly Architecture
# ============================================================================
# Playwright caused indefinite hangs and required 100MB+ overhead per ticker.
# NEW: 3-tier fallback: Scrapfly scrape+extract → Scrapfly HTML → Free
# ============================================================================

# Helper functions for tier implementations
def _host(u: str) -> str:
    """Extract hostname from URL"""
    h = (urlparse(u).hostname or "").lower()
    if h.startswith("www."):
        h = h[4:]
    return h

def _matches(host: str, dom: str) -> bool:
    """Check if host matches domain (exact or subdomain)"""
    return host == dom or host.endswith("." + dom)

# Domains requiring anti-bot protection (used in Tier 1 and Tier 2)
ANTIBOT_DOMAINS = {
    "simplywall.st", "seekingalpha.com", "zacks.com", "benzinga.com",
    "cnbc.com", "investing.com", "gurufocus.com", "fool.com",
    "insidermonkey.com", "nasdaq.com", "markets.financialcontent.com",
    "thefly.com", "streetinsider.com", "accesswire.com",
    "247wallst.com", "barchart.com", "telecompaper.com",
    "news.stocktradersdaily.com", "sharewise.com"
}


async def scrape_with_scrapfly_html_only(url: str, domain: str, max_retries: int = 2) -> Tuple[Optional[str], Optional[str]]:
    """
    TIER 2: Scrapfly Web Scraping API + newspaper3k parsing
    Uses GET /scrape endpoint to fetch HTML (bypasses anti-bot), then we parse with newspaper3k.
    Cost: ~$0.0003 per article (1 API credit for basic scrape)
    Success rate: ~85% (Scrapfly handles anti-bot, newspaper3k parses)

    This tier is ONLY called when free scraping fails (anti-bot protection, JavaScript sites, etc.)
    """
    global scrapfly_stats, enhanced_scraping_stats

    if "video.media.yql.yahoo.com" in url:
        LOG.warning(f"SCRAPFLY: Rejecting video URL: {url}")
        return None, "Video URL not supported"

    for attempt in range(max_retries + 1):
        try:
            if not SCRAPFLY_API_KEY:
                return None, "Scrapfly API key not configured"

            if normalize_domain(domain) in PAYWALL_DOMAINS:
                return None, f"Paywall domain: {domain}"

            if attempt > 0:
                delay = 2 ** attempt
                LOG.info(f"SCRAPFLY RETRY {attempt}/{max_retries} for {domain} after {delay}s delay")
                await asyncio.sleep(delay)

            LOG.info(f"SCRAPFLY: Attempting {domain} (attempt {attempt + 1})")

            # Update usage stats (1 credit = ~$0.0003)
            scrapfly_stats["requests_made"] += 1
            scrapfly_stats["cost_estimate"] += 0.0003
            scrapfly_stats["by_domain"][domain]["attempts"] += 1
            enhanced_scraping_stats["by_method"]["scrapfly"]["attempts"] += 1

            # Build params for Web Scraping API without extraction
            params = {
                "key": SCRAPFLY_API_KEY,
                "url": url,
                "country": "us"
            }

            # Enable anti-bot protection for known tough domains
            host = _host(url)
            if any(_matches(host, d) for d in ANTIBOT_DOMAINS) and "video.media" not in host:
                params["asp"] = "true"
                params["render_js"] = "true"

            session = get_http_session()
            async with session.get("https://api.scrapfly.io/scrape", params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        try:
                            result = await response.json()
                            html_content = result.get("result", {}).get("content", "")

                            if not html_content or len(html_content) < 500:
                                LOG.warning(f"SCRAPFLY: Insufficient HTML for {domain} ({len(html_content)} bytes)")
                                if attempt < max_retries:
                                    continue
                                return None, "Insufficient HTML"

                            # Parse with newspaper3k
                            article = newspaper.Article(url)
                            article.set_html(html_content)
                            article.parse()

                            text = article.text.strip()

                            if not text or len(text) < 100:
                                LOG.warning(f"SCRAPFLY: newspaper3k extraction empty for {domain} ({len(text)} chars)")
                                if attempt < max_retries:
                                    continue
                                return None, "Extraction returned insufficient content"

                            # Minimal cleaning
                            cleaned_content = clean_scraped_content(text, url, domain)

                            if not cleaned_content or len(cleaned_content) < 100:
                                LOG.warning(f"SCRAPFLY: Content too short after cleaning for {domain}")
                                if attempt < max_retries:
                                    continue
                                return None, "Content too short after cleaning"

                            # Validation
                            is_valid, validation_msg = validate_scraped_content(cleaned_content, url, domain)
                            if not is_valid:
                                LOG.warning(f"SCRAPFLY: Validation failed for {domain}: {validation_msg}")
                                if attempt < max_retries:
                                    continue
                                return None, f"Validation failed: {validation_msg}"

                            # Track tier success
                            scrapfly_stats["successful"] += 1
                            scrapfly_stats["by_domain"][domain]["successes"] += 1
                            enhanced_scraping_stats["by_method"]["scrapfly"]["successes"] += 1
                            enhanced_scraping_stats["scrapfly_success"] += 1

                            LOG.info(f"✅ SCRAPFLY SUCCESS: {domain} -> {len(cleaned_content)} chars")
                            return cleaned_content, None

                        except Exception as e:
                            LOG.warning(f"SCRAPFLY: Error processing for {domain}: {e}")
                            if attempt < max_retries:
                                continue
                            return None, str(e)

                    elif response.status == 402:
                        LOG.error(f"SCRAPFLY: Payment required for {domain}")
                        return None, "Scrapfly quota exceeded"

                    elif response.status == 422:
                        error_text = await response.text()
                        LOG.warning(f"SCRAPFLY: 422 for {domain}: {error_text[:500]}")
                        return None, "Invalid parameters"

                    elif response.status == 429:
                        LOG.warning(f"SCRAPFLY: Rate limited for {domain}")
                        if attempt < max_retries:
                            await asyncio.sleep(5)
                            continue
                        return None, "Rate limited"

                    else:
                        error_text = await response.text()
                        LOG.warning(f"SCRAPFLY: HTTP {response.status} for {domain}: {error_text[:500]}")
                        if attempt < max_retries:
                            continue
                        return None, f"HTTP {response.status}"

        except Exception as e:
            LOG.warning(f"SCRAPFLY: Request error for {domain} (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                continue
            return None, str(e)

    # If we got here, all retries failed
    scrapfly_stats["failed"] += 1
    return None, f"Scrapfly HTML fetch failed after {max_retries + 1} attempts"


async def scrape_with_requests_free(url: str, domain: str) -> Tuple[Optional[str], Optional[str]]:
    """
    TIER 1: Free scraping (FIRST tier - tries this before paid Scrapfly)
    Uses direct HTTP request + newspaper3k parser.
    Cost: $0
    Success rate: ~70% (works for simple sites without anti-bot)
    """
    global enhanced_scraping_stats

    try:
        LOG.info(f"FREE SCRAPER: Starting for {domain}")

        # Track tier attempts
        enhanced_scraping_stats["by_method"]["requests"]["attempts"] += 1

        # Check paywall domains
        if normalize_domain(domain) in PAYWALL_DOMAINS:
            return None, f"Paywall domain: {domain}"

        session = get_http_session()
        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/"
        }

        async with session.get(url, headers=headers, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    LOG.warning(f"FREE: HTTP {response.status} for {domain}")
                    return None, f"HTTP {response.status}"

                html = await response.text()

                if not html or len(html) < 500:
                    LOG.warning(f"FREE: Insufficient HTML for {domain} ({len(html)} bytes)")
                    return None, "Insufficient HTML"

                # Use newspaper3k to parse
                article = newspaper.Article(url)
                article.set_html(html)
                article.parse()

                text = article.text.strip()

                if not text or len(text) < 100:
                    LOG.warning(f"FREE: newspaper3k extraction empty for {domain} ({len(text)} chars)")
                    return None, "Extraction returned insufficient content"

                # Minimal cleaning
                cleaned_content = clean_scraped_content(text, url, domain)

                if not cleaned_content or len(cleaned_content) < 100:
                    LOG.warning(f"FREE: Content too short after cleaning for {domain}")
                    return None, "Content too short after cleaning"

                # Validation
                is_valid, validation_msg = validate_scraped_content(cleaned_content, url, domain)
                if not is_valid:
                    LOG.warning(f"FREE: Validation failed for {domain}: {validation_msg}")
                    return None, f"Validation failed: {validation_msg}"

                # Track tier success
                enhanced_scraping_stats["by_method"]["requests"]["successes"] += 1
                enhanced_scraping_stats["requests_success"] += 1

                LOG.info(f"✅ FREE SUCCESS: {domain} -> {len(cleaned_content)} chars")
                return cleaned_content, None

    except Exception as e:
        LOG.warning(f"FREE: Error for {domain}: {e}")
        return None, str(e)


async def scrape_with_scrapfly_async(url: str, domain: str, max_retries: int = 2) -> Tuple[Optional[str], Optional[str]]:
    """
    2-TIER SCRAPING ARCHITECTURE (REVERTED TO OLD STRUCTURE):

    Tier 1: Free requests + newspaper3k (handles ~70% of sites, $0 cost)
      ↓ If fails
    Tier 2: Scrapfly Web Scraping API + newspaper3k (anti-bot bypass, $0.0003 cost)

    This minimizes Scrapfly API calls and stays well under the 5/min concurrency limit.

    Returns: (content, error_message)
    """
    global enhanced_scraping_stats

    LOG.info(f"SCRAPFLY: Starting extraction for {domain}")

    # Track total scraping attempt
    enhanced_scraping_stats["total_attempts"] += 1

    # Try Tier 1: Free scraping FIRST
    content, error1 = await scrape_with_requests_free(url, domain)

    if content:
        return content, None

    LOG.warning(f"⚠️ TIER 1 (Free) failed for {domain}: {error1}")
    LOG.info(f"🔄 Falling back to TIER 2 (Scrapfly) for {domain}")

    # Try Tier 2: Scrapfly (only for tough sites)
    if SCRAPFLY_API_KEY:
        content, error2 = await scrape_with_scrapfly_html_only(url, domain, max_retries)

        if content:
            return content, None

        LOG.warning(f"⚠️ TIER 2 (Scrapfly) failed for {domain}: {error2}")
        LOG.error(f"❌ ALL TIERS FAILED for {domain} - Free: {error1}, Scrapfly: {error2}")
        enhanced_scraping_stats["total_failures"] += 1
        return None, f"All methods failed (Free: {error1}, Scrapfly: {error2})"
    else:
        LOG.error(f"❌ Scraping failed for {domain} - Free tier failed and no Scrapfly key configured")
        enhanced_scraping_stats["total_failures"] += 1
        return None, f"Free scraping failed: {error1}"


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
    Minimal content cleaning for Scrapfly Extraction API output.
    Scrapfly already removes HTML, ads, navigation - we just need light sanitization.

    IMPORTANT: Preserves financial data (numbers, percentages, financial codes).
    """
    if not content:
        return ""

    original_length = len(content)

    # Stage 1: Remove NULL bytes and control characters
    content = clean_null_bytes(content)
    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]+', '', content)

    # Stage 2: Remove encoding artifacts (rarely needed with Scrapfly)
    content = re.sub(r'[Â¿Â½]{3,}', '', content)  # Binary encoding artifacts

    # Stage 3: Clean up any leftover HTML entities (Scrapfly usually handles this)
    content = re.sub(r'&[a-zA-Z0-9#]+;', ' ', content)

    # Stage 4: Remove obvious boilerplate phrases (minimal)
    boilerplate_patterns = [
        r'Advertisement\s*',
        r'Sponsored Content\s*',
        r'Continue Reading\s*',
    ]

    for pattern in boilerplate_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)

    # Stage 5: Normalize whitespace (preserve paragraphs)
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Max 2 consecutive line breaks
    content = re.sub(r'[ \t]+', ' ', content)  # Normalize spaces/tabs to single space
    content = re.sub(r'\n +', '\n', content)  # Remove leading spaces
    content = re.sub(r' +\n', '\n', content)  # Remove trailing spaces

    # Final cleanup
    content = content.strip()
    content = clean_null_bytes(content)  # Final NULL byte check

    # Log cleaning effectiveness
    final_length = len(content)
    reduction = ((original_length - final_length) / original_length * 100) if original_length > 0 else 0

    if reduction > 20:  # Only log if significant reduction
        LOG.debug(f"Minimal cleaning: {original_length} → {final_length} chars ({reduction:.1f}% reduction)")

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
    LOG.info("SCRAPING SUCCESS RATES (2-Tier Architecture)")
    LOG.info("=" * 60)
    LOG.info(f"OVERALL SUCCESS: {overall_rate:.1f}% ({total_success}/{total_attempts} total articles)")
    LOG.info(f"TIER 1 (Free):     {requests_rate:.1f}% ({requests_success}/{requests_attempts} attempted)")
    LOG.info(f"TIER 2 (Scrapfly): {scrapfly_rate:.1f}% ({scrapfly_success}/{scrapfly_attempts} attempted)")
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

async def safe_content_scraper_with_scrapfly_only_async(url: str, domain: str, category: str, keyword: str, scraped_domains: set) -> Tuple[Optional[str], str]:
    """
    Scrapfly-only content scraper with Extraction API.
    No Tier 1 requests, no Playwright - Scrapfly handles everything.
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

    # SCRAPFLY-ONLY: No tier fallback needed
    if not SCRAPFLY_API_KEY:
        LOG.error("SCRAPFLY: API key not configured")
        enhanced_scraping_stats["total_failures"] += 1
        return None, "Scrapfly API key not configured"

    LOG.info(f"SCRAPFLY: Starting extraction for {domain}")

    try:
        scrapfly_content, scrapfly_error = await scrape_with_scrapfly_async(url, domain)

        if scrapfly_content:
            enhanced_scraping_stats["scrapfly_success"] += 1
            update_scraping_stats(category, keyword, True)
            return scrapfly_content, f"SCRAPFLY SUCCESS: {len(scrapfly_content)} chars (extraction API)"
        else:
            # Scrapfly failed
            enhanced_scraping_stats["total_failures"] += 1
            return None, f"SCRAPFLY FAILED: {scrapfly_error}"

    except Exception as e:
        LOG.error(f"SCRAPFLY: Unexpected error for {domain}: {e}")
        enhanced_scraping_stats["total_failures"] += 1
        return None, f"SCRAPFLY ERROR: {str(e)}"

def log_enhanced_scraping_stats():
    """Log Scrapfly-only scraping statistics"""
    total = enhanced_scraping_stats["total_attempts"]
    if total == 0:
        LOG.info("SCRAPING STATS: No attempts made")
        return

    scrapfly_success = enhanced_scraping_stats["scrapfly_success"]
    success_rate = (scrapfly_success / total) * 100

    LOG.info("=" * 60)
    LOG.info("SCRAPFLY SCRAPING STATS (Extraction API)")
    LOG.info("=" * 60)
    LOG.info(f"SUCCESS RATE: {success_rate:.1f}% ({scrapfly_success}/{total})")
    LOG.info(f"COST ESTIMATE: ${scrapfly_stats['cost_estimate']:.2f}")
    LOG.info("=" * 60)


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

    # Phase 1.5: Industry article relevance scoring (NEW - Oct 2025)
    # For industry articles only, score relevance after scraping but before AI summarization
    relevance_scores = {}  # {article_index: {"score": float, "reason": str, "is_rejected": bool, "provider": str}}

    for i, result in enumerate(scraping_results):
        if isinstance(result, Exception) or not result["success"]:
            continue

        article_category = categories[i] if i < len(categories) else categories[0]

        # Only score industry articles
        if article_category == "industry" and result.get("scraped_content"):
            article = articles_batch[i]
            industry_keyword = article.get("search_keyword", "unknown industry")

            try:
                LOG.info(f"[{analysis_ticker}] 📊 Scoring relevance for industry article {i}: {article.get('title', 'No title')[:60]}...")

                relevance_result = await score_industry_article_relevance(
                    ticker=analysis_ticker,
                    company_name=target_company_name,
                    industry_keyword=industry_keyword,
                    title=article.get("title", ""),
                    scraped_content=result["scraped_content"]
                )

                relevance_scores[i] = relevance_result

                # Log score
                score = relevance_result["score"]
                is_rejected = relevance_result["is_rejected"]
                provider = relevance_result["provider"]
                status = "REJECTED" if is_rejected else "ACCEPTED"

                LOG.info(f"[{analysis_ticker}] {'✗' if is_rejected else '✓'} Article {i} scored {score:.1f}/10 [{status}] via {provider}")
                LOG.debug(f"   Reason: {relevance_result['reason']}")

                # Store relevance score in database immediately
                article_id = article.get("id")
                if article_id:
                    with db() as conn, conn.cursor() as cur:
                        cur.execute("""
                            UPDATE ticker_articles
                            SET relevance_score = %s,
                                relevance_reason = %s,
                                is_rejected = %s
                            WHERE ticker = %s AND article_id = %s
                        """, (
                            score,
                            relevance_result["reason"],
                            is_rejected,
                            analysis_ticker,
                            article_id
                        ))
                        LOG.debug(f"   Saved relevance score to database for article {article_id}")

            except Exception as e:
                LOG.error(f"[{analysis_ticker}] ❌ Relevance scoring failed for article {i}: {e}")
                # Don't block processing on scoring failure - treat as accepted
                relevance_scores[i] = {
                    "score": 5.0,
                    "reason": f"Scoring failed: {str(e)}",
                    "is_rejected": False,
                    "provider": "error"
                }

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
            # Check if article was rejected by relevance gate
            if i in relevance_scores and relevance_scores[i]["is_rejected"]:
                article_category = categories[i] if i < len(categories) else categories[0]
                LOG.info(f"[{analysis_ticker}] ⏭️ Skipping AI summary for rejected {article_category} article {i} (score: {relevance_scores[i]['score']:.1f}/10)")
                # Add to results as successful scrape but no AI summary
                results.append({
                    "article_id": articles_batch[i]["id"],
                    "article_idx": i,
                    "success": True,
                    "scraped_content": result["scraped_content"],
                    "ai_summary": None,  # Rejected articles get no summary
                    "ai_model": None,
                    "content_scraped_at": result["content_scraped_at"],
                    "scraping_error": None
                })
            else:
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

                    # Articles already exist - use their IDs directly
                    article_id = article.get("id")

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

                    # Articles already exist - use their IDs directly
                    article_id = article.get("id")

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
                content, status = await safe_content_scraper_with_scrapfly_only_async(
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

            # FEED-TYPE-AWARE URL RESOLUTION
            final_resolved_url = None
            final_domain = None
            final_source_url = None

            # Determine feed type by checking feed URL (once per feed, not per article)
            feed_url = feed.get("url", "")
            is_yahoo_feed = "finance.yahoo.com" in feed_url
            is_google_feed = "news.google.com" in feed_url

            if is_yahoo_feed:
                # YAHOO FEED: Resolve immediately (existing logic)
                resolved_url, domain, source_url = domain_resolver._handle_yahoo_finance(url)

                if not resolved_url or not domain:
                    stats["blocked_spam"] += 1
                    continue

                final_resolved_url = resolved_url
                final_domain = domain
                final_source_url = source_url
                LOG.info(f"YAHOO FEED RESOLVED: {url[:60]} → {final_domain}")

            elif is_google_feed:
                # GOOGLE FEED: Defer resolution until after triage
                # Extract domain from title for spam check and deduplication
                clean_title, source_name = extract_source_from_title_smart(title)

                if source_name and not domain_resolver._is_spam_source(source_name):
                    domain = domain_resolver._resolve_publication_to_domain(source_name)

                    if domain and domain_resolver._is_spam_domain(domain):
                        stats["blocked_spam"] += 1
                        LOG.info(f"SPAM REJECTED: Google News → {domain} (from title: {title[:40]})")
                        continue
                else:
                    # Could not extract valid domain from title - skip this article
                    stats["blocked_spam"] += 1
                    LOG.warning(f"GOOGLE NEWS: Could not extract domain from title: {title[:60]}")
                    continue

                # Store with NULL resolved_url (will resolve after triage)
                final_resolved_url = None  # Deferred resolution
                final_domain = domain
                final_source_url = None
                LOG.info(f"GOOGLE FEED DEFERRED: {title[:60]} → {domain}")

            else:
                # Direct URL or unknown feed type - use standard resolution
                resolved_url, domain, source_url = domain_resolver._handle_direct_url(url)

                if not resolved_url or not domain:
                    stats["blocked_spam"] += 1
                    continue

                final_resolved_url = resolved_url
                final_domain = domain
                final_source_url = source_url

            # Generate hash for deduplication (supports both resolved URLs and Google News title-based)
            url_hash = get_url_hash(url, final_resolved_url, final_domain, title)
            
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
                        if final_source_url:
                            resolution_info = f" (via {get_or_create_formal_domain_name(normalize_domain(urlparse(final_source_url).netloc))})"
                        elif not final_resolved_url and is_google_feed:
                            resolution_info = f" (Google News - deferred resolution)"

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
                
                # FEED-TYPE-AWARE URL RESOLUTION (same as main ingestion)
                final_resolved_url = None
                final_domain = None
                final_source_url = None

                # Determine feed type
                feed_url = feed.get("url", "")
                is_yahoo_feed = "finance.yahoo.com" in feed_url
                is_google_feed = "news.google.com" in feed_url

                if is_yahoo_feed:
                    # YAHOO FEED: Resolve immediately
                    try:
                        resolved_url, domain, source_url = domain_resolver._handle_yahoo_finance(url)

                        if not resolved_url or not domain:
                            stats["yahoo_rejected"] += 1
                            continue

                        final_resolved_url = resolved_url
                        final_domain = domain
                        final_source_url = source_url
                    except Exception as e:
                        LOG.warning(f"Yahoo resolution failed for {url}: {e}")
                        stats["yahoo_rejected"] += 1
                        continue

                elif is_google_feed:
                    # GOOGLE FEED: Defer resolution
                    clean_title, source_name = extract_source_from_title_smart(title)

                    if source_name and not domain_resolver._is_spam_source(source_name):
                        domain = domain_resolver._resolve_publication_to_domain(source_name)

                        if domain and domain_resolver._is_spam_domain(domain):
                            stats["blocked_spam"] += 1
                            continue
                    else:
                        stats["blocked_spam"] += 1
                        continue

                    final_resolved_url = None  # Deferred
                    final_domain = domain
                    final_source_url = None

                else:
                    # Direct URL
                    try:
                        resolved_url, domain, source_url = domain_resolver._handle_direct_url(url)

                        if not resolved_url or not domain:
                            stats["blocked_spam"] += 1
                            continue

                        final_resolved_url = resolved_url
                        final_domain = domain
                        final_source_url = source_url
                    except Exception as e:
                        LOG.warning(f"URL resolution failed for {url}: {e}")
                        stats["blocked_spam"] += 1
                        continue

                # CHECK SPAM DOMAINS AFTER RESOLUTION
                if final_domain and final_domain in SPAM_DOMAINS:
                    stats["blocked_spam"] += 1
                    LOG.debug(f"SPAM DOMAIN BLOCKED: {final_domain} - {title[:50]}...")
                    continue

                # Generate hash for deduplication (supports both resolved URLs and Google News)
                url_hash = get_url_hash(url, final_resolved_url, final_domain, title)
                
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
                        clean_resolved_url = clean_null_bytes(final_resolved_url) if final_resolved_url else None
                        clean_title = clean_null_bytes(title or "")
                        clean_description = clean_null_bytes(display_content or "")
                        clean_search_keyword = clean_null_bytes(feed.get("search_keyword") or "")
                        clean_source_url = clean_null_bytes(final_source_url) if final_source_url else None
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

    # 6. Relevance score badge for industry articles (NEW - Oct 2025)
    relevance_badge_html = ""
    relevance_reason_html = ""
    if category == "industry" and article.get('relevance_score') is not None:
        score = article.get('relevance_score')
        is_rejected = article.get('is_rejected', False)
        reason = article.get('relevance_reason', 'No reason provided')

        # Color and label based on rejection status
        if is_rejected:
            badge_style = "background-color: #fee; color: #c53030; border: 1px solid #fc8181;"
            badge_icon = "✗"
            badge_label = f"REJECTED ({score:.1f}/10)"
        else:
            badge_style = "background-color: #e6ffed; color: #22543d; border: 1px solid #9ae6b4;"
            badge_icon = "✓"
            badge_label = f"ACCEPTED ({score:.1f}/10)"

        relevance_badge_html = f'<span style="display: inline-block; padding: 2px 8px; margin-right: 5px; border-radius: 3px; font-weight: bold; font-size: 10px; {badge_style}">{badge_icon} {badge_label}</span>'
        header_badges.append(relevance_badge_html)

        # Relevance reason shown below badges, above AI summary
        reason_escaped = html.escape(reason)
        relevance_reason_html = f"<br><div style='color: #718096; font-size: 11px; font-style: italic; margin-top: 4px; padding: 6px 8px; background-color: #f7fafc; border-left: 3px solid {'#fc8181' if is_rejected else '#9ae6b4'}; border-radius: 3px;'><strong>Relevance:</strong> {reason_escaped}</div>"

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
            {relevance_reason_html}
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

    # SEMAPHORE DISABLED: Prevents threading deadlock with concurrent tickers
    # with OPENAI_SEM:
    if True:  # Maintain indentation
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
            session = get_http_session()
            async with session.post(OPENAI_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=180)) as response:
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

    # SEMAPHORE DISABLED: Prevents threading deadlock with concurrent tickers
    # with CLAUDE_SEM:
    if True:  # Maintain indentation
        try:
            # System prompt (cached - instructions that repeat across articles)
            system_prompt = f"""You are a hedge fund analyst writing a factual memo on {company_name} ({ticker}). Analyze articles and write summaries using ONLY facts explicitly stated in the text.

**Focus:** Extract all material facts about {company_name} from the article.

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
- Each sentence must add new factual information"""

            # User content (variable - changes per article)
            user_content = f"""TARGET: {company_name} ({ticker})
TITLE: {title}
CONTENT: {scraped_content[:CONTENT_CHAR_LIMIT]}"""

            headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            data = {
                "model": ANTHROPIC_MODEL,
                "max_tokens": 8192,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}]
            }

            session = get_http_session()
            async with session.post(ANTHROPIC_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=180)) as response:
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

    # SEMAPHORE DISABLED: Prevents threading deadlock with concurrent tickers
    # with CLAUDE_SEM:
    if True:  # Maintain indentation
        try:
            # System prompt (cached - competitive analysis framework)
            system_prompt = f"""You are a hedge fund analyst evaluating how {competitor_name} ({competitor_ticker}) developments affect {target_company} ({target_ticker}) investors. Analyze articles and write summaries using ONLY facts explicitly stated in the text.

**Focus:** Extract facts about {competitor_name} AND assess impact on {target_company}'s competitive position.

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
- Each paragraph must connect {competitor_name} facts to {target_company} competitive impact"""

            # User content (variable - changes per article)
            user_content = f"""TARGET COMPANY: {target_company} ({target_ticker})
COMPETITOR: {competitor_name} ({competitor_ticker})
TITLE: {title}
CONTENT: {scraped_content[:CONTENT_CHAR_LIMIT]}"""

            headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            data = {
                "model": ANTHROPIC_MODEL,
                "max_tokens": 8192,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}]
            }

            session = get_http_session()
            async with session.post(ANTHROPIC_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=180)) as response:
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


async def score_industry_article_relevance_claude(
    ticker: str,
    company_name: str,
    industry_keyword: str,
    title: str,
    scraped_content: str
) -> Optional[Dict]:
    """
    Score industry article relevance to target company (0-10 scale) using Claude.
    Returns {"score": float, "reason": str} or None on error.
    Uses prompt caching for cost savings.
    """
    if not ANTHROPIC_API_KEY or not scraped_content or len(scraped_content.strip()) < 100:
        return None

    # SEMAPHORE DISABLED: Prevents threading deadlock with concurrent tickers
    # with CLAUDE_SEM:
    if True:  # Maintain indentation
        try:
            # System prompt (cached - relevance scoring framework)
            system_prompt = f"""You are a hedge fund analyst evaluating whether an industry article is ACTUALLY relevant to {company_name} ({ticker}), not just tangentially related to the {industry_keyword} keyword.

**Your Task:**
Rate this article's relevance to {company_name} on a 0-10 scale and explain why in 1-2 sentences.

**Scoring Rubric:**

**TIER 1: Highly Relevant (8-10)**
- Article discusses {industry_keyword} trends/events that DIRECTLY affect {company_name}'s:
  • Core business operations or revenue drivers
  • Cost structure or profit margins
  • Competitive positioning or market share
  • Regulatory environment or compliance requirements
  • Supply chain or key partnerships
- Specific companies/competitors in {company_name}'s sector are named with material developments
- Market data (TAM, growth rates, pricing) clearly impacts {company_name}'s addressable market
- Technology/product developments affect {company_name}'s product roadmap or R&D priorities

**TIER 2: Moderately Relevant (5-7)**
- Article covers {industry_keyword} sector broadly with some relevance to {company_name}:
  • General sector trends that indirectly affect {company_name}
  • Broader economic/regulatory factors affecting the {industry_keyword} industry
  • Competitor news with sector-wide implications (not just single company updates)
  • Industry research/analysis that provides competitive context
- Mentions multiple companies in the sector, including peers of {company_name}
- Discusses market opportunities or challenges affecting the sector as a whole

**TIER 3: Tangentially Related (2-4)**
- Article mentions {industry_keyword} but connection to {company_name} is weak:
  • Generic industry commentary without specific company or market impacts
  • Single competitor news with no broader sector implications
  • Distant geographic markets where {company_name} has minimal presence
  • Industry mentioned in passing, not the article's focus
  • Opinion pieces or trend lists without hard data
- Very limited actionable intelligence for {company_name} investors

**TIER 4: Not Relevant (0-1)**
- Article is NOT actually about {industry_keyword} or {company_name}:
  • Keyword appears by coincidence (company name collision, unrelated usage)
  • Different industry using similar terminology
  • Only mentions sector in generic context (e.g., "like companies in {industry_keyword}...")
  • Exclusively about unrelated companies/topics
  • Marketing content, press releases without news value

**Critical Filters - Always score 0-1 if:**
- Article is primarily about a different industry (keyword match is coincidental)
- Only connection is a passing mention or analogy
- Content is pure speculation without factual basis
- Advertorial or promotional content without news substance
- Company name collision (e.g., "Apple" the fruit vs AAPL the company)

**Output Format:**
Return JSON only:
{{
  "score": <float 0.0-10.0>,
  "reason": "<1-2 sentence explanation of score>"
}}

**Examples:**

TIER 1 (Score: 9.0):
{{
  "score": 9.0,
  "reason": "Article provides detailed Q3 2024 sales data for electric vehicles in North America with specific market share figures for major manufacturers, directly relevant to {company_name}'s competitive position and revenue forecasts."
}}

TIER 2 (Score: 6.0):
{{
  "score": 6.0,
  "reason": "Discusses broader semiconductor supply chain constraints affecting automotive industry, which impacts {company_name} but article lacks specific details about their products or tier suppliers."
}}

TIER 3 (Score: 3.0):
{{
  "score": 3.0,
  "reason": "Generic trend piece about future of {industry_keyword} with vague predictions and no specific companies, data, or actionable insights for {company_name} investors."
}}

TIER 4 (Score: 0.0):
{{
  "score": 0.0,
  "reason": "Article about {industry_keyword} in a completely different context (e.g., banking software when we're tracking industrial manufacturing); keyword match is coincidental."
}}

**Your Goal:** Be STRICT. Only scores of 5+ should proceed to expensive AI summarization. When in doubt, score lower."""

            # User content (variable, not cached)
            user_content = f"""**Article Title:** {title}

**Industry Keyword:** {industry_keyword}

**Article Content:**
{scraped_content[:8000]}

Rate this article's relevance to {company_name} ({ticker}) on a 0-10 scale. Return JSON only."""

            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            data = {
                "model": ANTHROPIC_MODEL,
                "max_tokens": 512,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}]
            }

            session = get_http_session()
            async with session.post(ANTHROPIC_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        LOG.error(f"Claude relevance scoring error {response.status} for {ticker}: {error_text[:500]}")
                        return None

                    result = await response.json()
                    LOG.info(f"Claude response for {ticker}: {str(result)[:500]}")
                    content = result.get("content", [{}])[0].get("text", "")

                    if not content:
                        LOG.error(f"Claude returned empty content for {ticker}. Full response: {result}")
                        return None

                    try:
                        # Strip markdown code blocks if present (Claude wraps JSON in ```json...```)
                        content_clean = content.strip()
                        if content_clean.startswith("```"):
                            # Remove first line (```json) and last line (```)
                            lines = content_clean.split("\n")
                            if len(lines) >= 3:
                                content_clean = "\n".join(lines[1:-1])

                        parsed = json.loads(content_clean)
                        score = float(parsed.get("score", 0.0))
                        reason = parsed.get("reason", "No reason provided")

                        # Validate score range
                        if not (0.0 <= score <= 10.0):
                            LOG.warning(f"Claude returned invalid score {score} for {ticker}, clamping to 0-10")
                            score = max(0.0, min(10.0, score))

                        return {"score": score, "reason": reason}
                    except (json.JSONDecodeError, ValueError) as e:
                        LOG.error(f"Failed to parse Claude relevance score for {ticker}: {e}. Content: '{content[:200]}'")
                        return None

        except Exception as e:
            LOG.error(f"Claude relevance scoring failed for {ticker}: {str(e)}")
            return None

    return None


async def score_industry_article_relevance_openai(
    ticker: str,
    company_name: str,
    industry_keyword: str,
    title: str,
    scraped_content: str
) -> Optional[Dict]:
    """
    Score industry article relevance to target company (0-10 scale) using OpenAI.
    Fallback when Claude is unavailable.
    Returns {"score": float, "reason": str} or None on error.
    """
    if not OPENAI_API_KEY or not scraped_content or len(scraped_content.strip()) < 100:
        return None

    # SEMAPHORE DISABLED: Prevents threading deadlock with concurrent tickers
    # with OPENAI_SEM:
    if True:  # Maintain indentation
        try:
            system_prompt = f"""You are a hedge fund analyst evaluating whether an industry article is ACTUALLY relevant to {company_name} ({ticker}), not just tangentially related to the {industry_keyword} keyword.

Rate this article's relevance on a 0-10 scale:
- 8-10: Highly relevant (directly affects {company_name}'s operations, costs, revenue, or competitive position)
- 5-7: Moderately relevant (sector trends with indirect impact on {company_name})
- 2-4: Tangentially related (weak connection, limited actionable intelligence)
- 0-1: Not relevant (coincidental keyword match, different industry, or no real connection)

Return JSON only:
{{
  "score": <float 0.0-10.0>,
  "reason": "<1-2 sentence explanation>"
}}

Be STRICT. Only scores of 5+ should proceed to expensive AI summarization."""

            user_content = f"""**Article Title:** {title}

**Industry Keyword:** {industry_keyword}

**Article Content:**
{scraped_content[:8000]}

Rate this article's relevance to {company_name} ({ticker}) on a 0-10 scale. Return JSON only."""

            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

            # Define JSON schema for structured output
            relevance_schema = {
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                    "reason": {"type": "string"}
                },
                "required": ["score", "reason"],
                "additionalProperties": False
            }

            data = {
                "model": OPENAI_MODEL,
                "input": f"{system_prompt}\n\n{user_content}",
                "max_output_tokens": 256,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "relevance_score",
                        "schema": relevance_schema,
                        "strict": True
                    }
                },
                "truncation": "auto"
            }

            session = get_http_session()
            async with session.post(OPENAI_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        LOG.error(f"OpenAI relevance scoring error {response.status} for {ticker}: {error_text[:500]}")
                        return None

                    result = await response.json()
                    LOG.info(f"OpenAI response for {ticker}: {str(result)[:500]}")
                    content = result.get("text", "")

                    if not content:
                        LOG.error(f"OpenAI returned empty content for {ticker}. Full response: {result}")
                        return None

                    try:
                        parsed = json.loads(content)
                        score = float(parsed.get("score", 0.0))
                        reason = parsed.get("reason", "No reason provided")

                        # Validate score range
                        if not (0.0 <= score <= 10.0):
                            LOG.warning(f"OpenAI returned invalid score {score} for {ticker}, clamping to 0-10")
                            score = max(0.0, min(10.0, score))

                        return {"score": score, "reason": reason}
                    except (json.JSONDecodeError, ValueError) as e:
                        LOG.error(f"Failed to parse OpenAI relevance score for {ticker}: {e}. Content: '{content[:200]}'")
                        return None

        except Exception as e:
            LOG.error(f"OpenAI relevance scoring failed for {ticker}: {str(e)}")
            return None

    return None


async def score_industry_article_relevance(
    ticker: str,
    company_name: str,
    industry_keyword: str,
    title: str,
    scraped_content: str,
    threshold: float = 6.0
) -> Dict:
    """
    Main wrapper for industry article relevance scoring.
    Tries Claude first, falls back to OpenAI if needed.

    Args:
        ticker: Stock ticker (e.g., "AAPL")
        company_name: Company name (e.g., "Apple Inc.")
        industry_keyword: Industry keyword being evaluated (e.g., "Cloud Computing")
        title: Article title
        scraped_content: Full article text
        threshold: Minimum score to accept (default 6.0, rejects ≤6)

    Returns:
        {
            "score": float (0.0-10.0),
            "reason": str,
            "is_rejected": bool (True if score <= threshold),
            "provider": str ("claude" or "openai" or "error")
        }
    """
    # Try Claude first (primary)
    result = await score_industry_article_relevance_claude(
        ticker, company_name, industry_keyword, title, scraped_content
    )

    if result:
        result["is_rejected"] = result["score"] <= threshold
        result["provider"] = "claude"
        return result

    # Fallback to OpenAI
    LOG.warning(f"Claude relevance scoring unavailable for {ticker}, trying OpenAI fallback")
    result = await score_industry_article_relevance_openai(
        ticker, company_name, industry_keyword, title, scraped_content
    )

    if result:
        result["is_rejected"] = result["score"] <= threshold
        result["provider"] = "openai"
        return result

    # Both failed
    LOG.error(f"Both Claude and OpenAI relevance scoring failed for {ticker}")
    return {
        "score": 0.0,
        "reason": "AI scoring unavailable (both providers failed)",
        "is_rejected": True,
        "provider": "error"
    }


async def generate_claude_industry_article_summary(industry_keyword: str, target_company: str, target_ticker: str, title: str, scraped_content: str) -> Optional[str]:
    """Generate Claude summary for industry article with target company POV"""
    if not ANTHROPIC_API_KEY or not scraped_content or len(scraped_content.strip()) < 200:
        return None

    # SEMAPHORE DISABLED: Prevents threading deadlock with concurrent tickers
    # with CLAUDE_SEM:
    if True:  # Maintain indentation
        try:
            # System prompt (cached - industry analysis framework)
            system_prompt = f"""You are a hedge fund analyst evaluating how {industry_keyword} sector developments affect {target_company} ({target_ticker}). Analyze articles and write summaries using ONLY facts explicitly stated in the text.

**Focus:** Extract {industry_keyword} industry insights and explain specific implications for {target_company}'s operations, costs, demand, or competitive position.

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
- Each paragraph must connect sector facts to {target_company}-specific impact"""

            # User content (variable - changes per article)
            user_content = f"""TARGET COMPANY: {target_company} ({target_ticker})
SECTOR FOCUS: {industry_keyword}
TITLE: {title}
CONTENT: {scraped_content[:CONTENT_CHAR_LIMIT]}"""

            headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            data = {
                "model": ANTHROPIC_MODEL,
                "max_tokens": 8192,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}]
            }

            session = get_http_session()
            async with session.post(ANTHROPIC_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=180)) as response:
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

    # SEMAPHORE DISABLED: Prevents threading deadlock with concurrent tickers
    # with OPENAI_SEM:
    if True:  # Maintain indentation
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

            session = get_http_session()
            async with session.post(OPENAI_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=180)) as response:
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

    # SEMAPHORE DISABLED: Prevents threading deadlock with concurrent tickers
    # with OPENAI_SEM:
    if True:  # Maintain indentation
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

            session = get_http_session()
            async with session.post(OPENAI_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=180)) as response:
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

    # SEMAPHORE DISABLED: Prevents threading deadlock with concurrent tickers
    # with OPENAI_SEM:
    if True:  # Maintain indentation
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

            session = get_http_session()
            async with session.post(OPENAI_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=180)) as response:
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

PRIMARY CRITERION: Is this article SPECIFICALLY about {company_name}? If unclear, skip it.

SELECT (choose up to {target_cap}):

TIER 1 - Hard corporate events (scrape_priority=1):
- Financial: "beats," "misses," "earnings," "revenue," "guidance," "margin," "profit," "loss," "EPS," "sales"
- Capital: "buyback," "dividend," "debt," "bond," "offering," "raises," "refinance," "converts"
- M&A: "acquires," "buys," "sells," "divests," "stake," "merger," "spin-off," "joint venture"
- Regulatory: "FDA," "SEC," "DOJ," "FTC," "investigation," "fine," "settlement," "approval," "lawsuit," "probe"
- Operations: "accident," "disaster," "halt," "shutdown," "recall," "expansion," "closure," "layoffs," "strike"
- Products: "approval," "launch," "recall," "trial results," "patent" WITH specific product/drug names
- Contracts: Dollar amounts in title (e.g., "$500M contract," "£2B deal")
- Ratings: "upgrade," "downgrade" WITH analyst firm name (e.g., "BofA upgrades," "Moody's cuts")
- Price movements: "surges," "jumps," "plunges," "drops" WITH percentage or specific price levels

TIER 2 - Strategic developments and analysis (scrape_priority=2):
- Leadership: CEO, CFO, President, CTO WITH "appoints," "names," "resigns," "retires," "replaces"
- Partnerships: Named partner companies (e.g., "{company_name} partners with [Other Company]")
- Technology: Specific tech/platform names WITH "launches," "announces," "deploys"
- Facilities: Plant/office/branch/store openings, closures WITH locations and capacity/headcount numbers
- Clinical: Trial phases, enrollment milestones, data releases (pharma/biotech)
- Spectrum/Licenses: Acquisitions, renewals WITH specific bands/regions (telecom)
- Geographic: Market entry/exit WITH investment levels or unit counts
- Investment theses: Articles analyzing bull/bear cases, valuation, competitive position
- Executive interviews: CEO, CFO, founder interviews discussing strategy, outlook, vision
- Stock performance analysis: "Why [ticker] stock," "[Company] stock analysis," explaining recent moves
- Industry positioning: "{company_name}'s role in [trend]," competitive advantages/disadvantages

TIER 3 - Context and market intelligence (scrape_priority=3):
- Analyst coverage WITH price targets, ratings, or detailed notes
- Industry awards, certifications indicating competitive position
- Market opportunity sizing specific to {company_name}
- Routine announcements WITH material operational details
- Technical analysis from any source WITH specific price targets or chart patterns

ANALYTICAL CONTENT - Include these question-based titles:
✓ "Why {company_name} stock [moved/performed]..." (explaining actual events)
✓ "Can {company_name} [achieve/sustain/compete]..." (analyzing capability)
✓ "Should you buy {company_name}..." (investment thesis with specific reasoning)
✓ "What's next for {company_name}..." (forward-looking analysis based on recent events)
✓ "[Company] stock: [Question about valuation/growth/strategy]" (substantive analysis)

REJECT COMPLETELY - Never select:
- Generic watchlists: "Top 5 stocks," "Best dividend picks," "Stocks to watch this week"
- Sector roundups: "Tech sector movers," "Healthcare stocks rally," "Energy update"
- Unrelated listicles: Articles where {company_name} is mentioned in passing among many tickers
- Pure clickbait: "This stock could 10x" (without specific company thesis)
- Historical what-ifs: "If you'd invested $1000 in 2010," "Where would you be today"
- Distant predictions: "Price prediction 2030," "Could reach $X by 2035" (without near-term catalyst)
- Market research only: "Industry to reach $XB by 20XX" (unless specifically about {company_name}'s role)
- Quote pages: "Stock Price | Live Quotes & Charts | [Exchange]"
- Attribution confusion: Articles where {company_name} is the news SOURCE not the SUBJECT

DISAMBIGUATION - Critical for accuracy:
- If title leads with different company name, likely not about {company_name} (reject unless comparative)
- If {company_name} only appears as news source attribution (e.g., "According to {company_name}..."), reject
- For common words (Oracle, Amazon, Apple, Crown), verify context matches YOUR company
- Multi-company articles: Only select if {company_name} is primary focus (≥50% of title/description)

SCRAPE PRIORITY (assign integer 1-3):
1 = Tier 1 (financial results, M&A, regulatory, disasters, major contracts, price moves)
2 = Tier 2 (leadership, partnerships, launches, analysis, interviews, performance explainers)
3 = Tier 3 (analyst coverage, awards, market sizing, technical analysis)

For each article assess:
- likely_repeat: Same event as another selected article?
- repeat_key: Event identifier (e.g., "q2_earnings_2025," "ceo_change_sept_2025")
- confidence: 0.0-1.0, certainty this is specifically about {company_name}

SELECTION STANDARD:
- When uncertain if article is about {company_name}, skip it
- Prioritize relevance over domain prestige
- A niche trade publication covering {company_name} specifically > major outlet mentioning tangentially
- Only select if confident the article provides actionable intelligence about {company_name}

CRITICAL CONSTRAINT: Return UP TO {target_cap} articles. Select fewer if uncertain about relevance."""

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

INDUSTRY CONTEXT: Select articles about sector-wide trends, regulatory changes, supply/demand shifts, and competitive dynamics that affect {company_name}'s business environment. These should provide competitive intelligence, not just mention the sector in passing.

CRITICAL: Select UP TO {target_cap} articles, fewer if uncertain.

PRIMARY CRITERION: Does this article reveal information about {company_name}'s competitive landscape, regulatory environment, or market opportunity?

SELECT (choose up to {target_cap}):

TIER 1 - Hard industry events with quantified impact (scrape_priority=1):
- Regulatory/Policy: New laws, rules, tariffs, bans, quotas WITH specific rates/dates/costs affecting {sector}
- Pricing: Commodity/service prices, reimbursement rates WITH specific figures affecting {company_name} sector
- Supply/Demand: Production disruptions, capacity changes WITH volume/value numbers impacting {sector}
- Standards: New requirements, certifications, compliance rules WITH deadlines/costs for {sector}
- Trade: Agreements, restrictions, sanctions WITH affected volumes or timelines for {sector}
- Financial: Interest rates, capital requirements, reserve rules affecting {sector}
- Technology shifts: Standards changes, platform migrations, protocol updates affecting {sector}

TIER 2 - Strategic sector developments (scrape_priority=2):
- Major capacity additions/closures WITH impact metrics (e.g., "500MW," "1M units/year") in {sector}
- Industry consolidation WITH transaction values and market share implications
- Technology adoption WITH implementation timelines and cost/efficiency impacts
- Labor agreements WITH wage/benefit details affecting {sector} economics
- Infrastructure investments WITH budgets and completion dates for {sector}
- Patent expirations, generic approvals, licensing changes WITH market impact
- Major peer announcements revealing sector trends (from peers: {peers_display})
- Geographic expansion patterns: Multiple companies entering/exiting same markets
- Competitive dynamics: Pricing wars, margin pressure, customer switching patterns

TIER 3 - Market intelligence and context (scrape_priority=3):
- Market opportunity sizing WITH credible TAM/SAM figures for {company_name}'s addressable market
- Economic indicators directly affecting {sector} WITH specific data points
- Government funding/initiatives WITH allocated budgets (not vague "plans")
- Research findings WITH quantified sector implications
- Adoption metrics: Customer/user growth rates, penetration figures for {sector}
- Cost structure changes: Input prices, labor costs, logistics affecting {sector}
- Analyst sector reports WITH specific company mentions or competitive comparisons

ANALYTICAL CONTENT - Include sector analysis:
✓ "Why [sector] companies are [performing/struggling]..." (explaining macro trends)
✓ "[Sector] industry faces [challenge/opportunity]..." (structural shifts)
✓ "Major players in [sector] [taking action]..." (coordinated industry moves)
✓ "[Peer company] success/failure signals [trend]..." (competitive intelligence)
✓ "Can [sector] sustain [growth/margins/demand]..." (industry viability questions)

REJECT COMPLETELY - Never select:
- Generic market research: "Industry to reach $XB by 20YY" WITHOUT {company_name} positioning
- Pure trend pieces: "Top 5 trends in [sector]," "Future of [industry]" WITHOUT specifics
- Single company news: Peer earnings, appointments WITHOUT broader sector implications
- Unrelated companies: Articles about non-peer companies outside {company_name}'s competitive set
- Distant forecasts: "20XX outlook," "Next decade in [sector]" WITHOUT near-term catalysts
- Pure opinion: "Analysis," "Commentary" WITHOUT hard data or specific sector insights
- Small company routine news: Junior partnerships, minor financing rounds
- Attribution confusion: Industry trade group reports where {sector} is just mentioned

COMPETITIVE INTELLIGENCE - Include when:
✓ Peer company action indicates sector direction (new product categories, pricing changes, geographic priorities)
✓ Regulatory action against competitor reveals industry-wide risks
✓ Supply chain disruption at major player affects {sector} availability/pricing
✓ Technology deployment shows sector-wide adoption affecting {company_name}
✓ Financial performance reveals {sector} cost/margin/demand trends
✓ Multiple peers make similar strategic moves (consolidation wave, geographic expansion)

SCRAPE PRIORITY (assign integer 1-3):
1 = Tier 1 (regulatory, pricing, supply shocks WITH numbers)
2 = Tier 2 (capacity, consolidation, policy WITH budgets, peer moves with sector implications)
3 = Tier 3 (TAM sizing, economic indicators, adoption metrics, sector reports)

For each article assess:
- likely_repeat: Same sector event covered by multiple outlets?
- repeat_key: Event identifier (e.g., "eu_ai_regulation_oct2025," "chip_shortage_q4")
- confidence: 0.0-1.0, certainty this has implications for {company_name}'s competitive position

SELECTION STANDARD:
- Prioritize sector-specific insights over general business news
- A trade publication covering {sector} specifically > major outlet with tangential mention
- Include if article reveals something material about {company_name}'s operating environment
- Skip if industry mention is generic or doesn't connect to {company_name}'s business

CRITICAL CONSTRAINT: Return UP TO {target_cap} articles. Select fewer if uncertain about sector relevance to {company_name}."""

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

PRIMARY CRITERION: Is this article SPECIFICALLY about {competitor_name}? If unclear, skip it.

SELECT (choose up to {target_cap}):

TIER 1 - Hard corporate events (scrape_priority=1):
- Financial: "beats," "misses," "earnings," "revenue," "guidance," "margin," "profit," "loss," "EPS," "sales"
- Capital: "buyback," "dividend," "debt," "bond," "offering," "raises," "refinance," "converts"
- M&A: "acquires," "buys," "sells," "divests," "stake," "merger," "spin-off," "joint venture"
- Regulatory: "FDA," "SEC," "DOJ," "FTC," "investigation," "fine," "settlement," "approval," "lawsuit," "probe"
- Operations: "accident," "disaster," "halt," "shutdown," "recall," "expansion," "closure," "layoffs," "strike"
- Products: "approval," "launch," "recall," "trial results," "patent" WITH specific product/drug names
- Contracts: Dollar amounts in title (e.g., "$500M contract," "£2B deal")
- Ratings: "upgrade," "downgrade" WITH analyst firm name (e.g., "BofA upgrades," "Moody's cuts")
- Price movements: "surges," "jumps," "plunges," "drops" WITH percentage or specific levels

TIER 2 - Strategic competitive intelligence (scrape_priority=2):
- Leadership: CEO, CFO, President WITH "appoints," "names," "resigns," "retires," "replaces"
- Partnerships: Named partners (competitive threats or ecosystem plays)
- Technology: New products/platforms WITH "launches," "announces," "deploys"
- Facilities: Capacity expansions/closures WITH locations and scale
- Geographic: Market entry/exit WITH investment levels (competitive overlap with {ticker})
- Pricing: Price changes, discounting, bundling affecting competitive position
- Customer wins/losses: Major accounts, market share shifts
- Executive interviews: Strategy, roadmap, competitive positioning statements

TIER 3 - Competitive context (scrape_priority=3):
- Analyst coverage WITH price targets or competitive comparisons to {ticker}
- Performance analysis: "Why {competitor_name} [outperformed/underperformed]..."
- Strategic questions: "Can {competitor_name} compete with..." "Will {competitor_name} disrupt..."
- Awards, certifications affecting competitive positioning
- Technical analysis WITH specific price levels

ANALYTICAL CONTENT - Include competitor analysis:
✓ "Why {competitor_name} stock [moved]..." (understanding competitive threats/opportunities)
✓ "{competitor_name} vs {ticker}" or competitive comparisons
✓ "Can {competitor_name} [challenge/overtake/sustain]..." (competitive capability)
✓ "{competitor_name} strategy in [market/product]..." (strategic intelligence)
✓ "What {competitor_name}'s [move] means for..." (competitive implications)

REJECT COMPLETELY - Never select:
- Generic lists: "Top dividend stocks," "Best performers," "Stocks to watch"
- Sector roundups: "Tech movers," "Healthcare rally" (unless {competitor_name} is primary focus)
- Unrelated mentions: {competitor_name} listed among many tickers without focus
- Pure speculation: "Could 10x" WITHOUT specific competitive thesis
- Historical: "If you'd invested," "20 years of returns"
- Distant predictions: "2035 price prediction" WITHOUT near-term catalysts
- Market research: "Industry forecast" (unless specifically about {competitor_name})
- Quote pages: "Stock Price | Charts | [Exchange]"

DISAMBIGUATION - Critical accuracy:
- If title leads with different company, likely not about {competitor_name}
- If {competitor_name} is just news source attribution, reject
- For common names, verify context matches YOUR competitor
- Multi-company: Only select if {competitor_name} is ≥50% of focus

SCRAPE PRIORITY (assign integer 1-3):
1 = Tier 1 (financials, M&A, regulatory, disasters, contracts, price moves)
2 = Tier 2 (leadership, launches, partnerships, pricing, customer wins, interviews)
3 = Tier 3 (analyst coverage, performance analysis, competitive questions)

For each article assess:
- likely_repeat: Same event as another selected article?
- repeat_key: Event identifier (e.g., "q3_earnings_2025," "product_launch_oct")
- confidence: 0.0-1.0, certainty this is specifically about {competitor_name}

SELECTION STANDARD:
- When uncertain if about {competitor_name}, skip it
- Prioritize competitive intelligence relevance to {ticker}
- Niche source covering {competitor_name} specifically > major outlet tangential mention
- Only select if provides actionable intelligence about {competitor_name}

CRITICAL CONSTRAINT: Return UP TO {target_cap} articles. Select fewer if uncertain about relevance."""

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

PRIMARY CRITERION: Is this article SPECIFICALLY about {company_name}? If unclear, skip it.

SELECT (choose up to {target_cap}):

TIER 1 - Hard corporate events (scrape_priority=1):
- Financial: "beats," "misses," "earnings," "revenue," "guidance," "margin," "profit," "loss," "EPS," "sales"
- Capital: "buyback," "dividend," "debt," "bond," "offering," "raises," "refinance," "converts"
- M&A: "acquires," "buys," "sells," "divests," "stake," "merger," "spin-off," "joint venture"
- Regulatory: "FDA," "SEC," "DOJ," "FTC," "investigation," "fine," "settlement," "approval," "lawsuit," "probe"
- Operations: "accident," "disaster," "halt," "shutdown," "recall," "expansion," "closure," "layoffs," "strike"
- Products: "approval," "launch," "recall," "trial results," "patent" WITH specific product/drug names
- Contracts: Dollar amounts in title (e.g., "$500M contract," "£2B deal")
- Ratings: "upgrade," "downgrade" WITH analyst firm name (e.g., "BofA upgrades," "Moody's cuts")
- Price movements: "surges," "jumps," "plunges," "drops" WITH percentage or specific price levels

TIER 2 - Strategic developments and analysis (scrape_priority=2):
- Leadership: CEO, CFO, President, CTO WITH "appoints," "names," "resigns," "retires," "replaces"
- Partnerships: Named partner companies (e.g., "{company_name} partners with [Other Company]")
- Technology: Specific tech/platform names WITH "launches," "announces," "deploys"
- Facilities: Plant/office/branch/store openings, closures WITH locations and capacity/headcount numbers
- Clinical: Trial phases, enrollment milestones, data releases (pharma/biotech)
- Spectrum/Licenses: Acquisitions, renewals WITH specific bands/regions (telecom)
- Geographic: Market entry/exit WITH investment levels or unit counts
- Investment theses: Articles analyzing bull/bear cases, valuation, competitive position
- Executive interviews: CEO, CFO, founder interviews discussing strategy, outlook, vision
- Stock performance analysis: "Why [ticker] stock," "[Company] stock analysis," explaining recent moves
- Industry positioning: "{company_name}'s role in [trend]," competitive advantages/disadvantages

TIER 3 - Context and market intelligence (scrape_priority=3):
- Analyst coverage WITH price targets, ratings, or detailed notes
- Industry awards, certifications indicating competitive position
- Market opportunity sizing specific to {company_name}
- Routine announcements WITH material operational details
- Technical analysis from any source WITH specific price targets or chart patterns

ANALYTICAL CONTENT - Include these question-based titles:
✓ "Why {company_name} stock [moved/performed]..." (explaining actual events)
✓ "Can {company_name} [achieve/sustain/compete]..." (analyzing capability)
✓ "Should you buy {company_name}..." (investment thesis with specific reasoning)
✓ "What's next for {company_name}..." (forward-looking analysis based on recent events)
✓ "[Company] stock: [Question about valuation/growth/strategy]" (substantive analysis)

REJECT COMPLETELY - Never select:
- Generic watchlists: "Top 5 stocks," "Best dividend picks," "Stocks to watch this week"
- Sector roundups: "Tech sector movers," "Healthcare stocks rally," "Energy update"
- Unrelated listicles: Articles where {company_name} is mentioned in passing among many tickers
- Pure clickbait: "This stock could 10x" (without specific company thesis)
- Historical what-ifs: "If you'd invested $1000 in 2010," "Where would you be today"
- Distant predictions: "Price prediction 2030," "Could reach $X by 2035" (without near-term catalyst)
- Market research only: "Industry to reach $XB by 20XX" (unless specifically about {company_name}'s role)
- Quote pages: "Stock Price | Live Quotes & Charts | [Exchange]"
- Attribution confusion: Articles where {company_name} is the news SOURCE not the SUBJECT

DISAMBIGUATION - Critical for accuracy:
- If title leads with different company name, likely not about {company_name} (reject unless comparative)
- If {company_name} only appears as news source attribution (e.g., "According to {company_name}..."), reject
- For common words (Oracle, Amazon, Apple, Crown), verify context matches YOUR company
- Multi-company articles: Only select if {company_name} is primary focus (≥50% of title/description)

SCRAPE PRIORITY (assign integer 1-3):
1 = Tier 1 (financial results, M&A, regulatory, disasters, major contracts, price moves)
2 = Tier 2 (leadership, partnerships, launches, analysis, interviews, performance explainers)
3 = Tier 3 (analyst coverage, awards, market sizing, technical analysis)

SELECTION STANDARD:
- When uncertain if article is about {company_name}, skip it
- Prioritize relevance over domain prestige
- A niche trade publication covering {company_name} specifically > major outlet mentioning tangentially
- Only select if confident the article provides actionable intelligence about {company_name}

Return a JSON array of selected articles. Each must have:
[{{"id": 0, "scrape_priority": 1, "why": "brief reason"}}]

CRITICAL CONSTRAINT: Return UP TO {target_cap} articles. Select fewer if uncertain about relevance."""

    # Separate variable content for better caching (articles data changes per call)
    user_content = f"Articles: {json.dumps(items, separators=(',', ':'))}"

    try:
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",  # Updated for prompt caching
            "content-type": "application/json"
        }

        data = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_content
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

                # Sort by priority (1=HIGH, 2=MEDIUM, 3=LOW), then by recency
                selected_articles.sort(key=lambda x: (
                    x["scrape_priority"],  # Sort ascending: 1, 2, 3 (HIGH→MEDIUM→LOW)
                    -articles[x["id"]].get("published_at", datetime.min).timestamp() if articles[x["id"]].get("published_at") else 0
                ))

                # Cap at target after sorting (LOW-priority articles cut first)
                if len(selected_articles) > target_cap:
                    LOG.warning(f"Claude selected {len(selected_articles)}, capping to top {target_cap} by priority")
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

    system_prompt = f"""You are a financial analyst selecting the {target_cap} most important INDUSTRY articles from {len(articles)} candidates based ONLY on titles and descriptions.

TARGET COMPANY: {company_name} ({ticker})
SECTOR: {sector}
KNOWN PEERS: {peers_display}

INDUSTRY CONTEXT: Select articles about sector-wide trends, regulatory changes, supply/demand shifts, and competitive dynamics that affect {company_name}'s business environment. These should provide competitive intelligence, not just mention the sector in passing.

CRITICAL: Select UP TO {target_cap} articles, fewer if uncertain.

PRIMARY CRITERION: Does this article reveal information about {company_name}'s competitive landscape, regulatory environment, or market opportunity?

SELECT (choose up to {target_cap}):

TIER 1 - Hard industry events with quantified impact (scrape_priority=1):
- Regulatory/Policy: New laws, rules, tariffs, bans, quotas WITH specific rates/dates/costs affecting {sector}
- Pricing: Commodity/service prices, reimbursement rates WITH specific figures affecting {company_name} sector
- Supply/Demand: Production disruptions, capacity changes WITH volume/value numbers impacting {sector}
- Standards: New requirements, certifications, compliance rules WITH deadlines/costs for {sector}
- Trade: Agreements, restrictions, sanctions WITH affected volumes or timelines for {sector}
- Financial: Interest rates, capital requirements, reserve rules affecting {sector}
- Technology shifts: Standards changes, platform migrations, protocol updates affecting {sector}

TIER 2 - Strategic sector developments (scrape_priority=2):
- Major capacity additions/closures WITH impact metrics (e.g., "500MW," "1M units/year") in {sector}
- Industry consolidation WITH transaction values and market share implications
- Technology adoption WITH implementation timelines and cost/efficiency impacts
- Labor agreements WITH wage/benefit details affecting {sector} economics
- Infrastructure investments WITH budgets and completion dates for {sector}
- Patent expirations, generic approvals, licensing changes WITH market impact
- Major peer announcements revealing sector trends (from peers: {peers_display})
- Geographic expansion patterns: Multiple companies entering/exiting same markets
- Competitive dynamics: Pricing wars, margin pressure, customer switching patterns

TIER 3 - Market intelligence and context (scrape_priority=3):
- Market opportunity sizing WITH credible TAM/SAM figures for {company_name}'s addressable market
- Economic indicators directly affecting {sector} WITH specific data points
- Government funding/initiatives WITH allocated budgets (not vague "plans")
- Research findings WITH quantified sector implications
- Adoption metrics: Customer/user growth rates, penetration figures for {sector}
- Cost structure changes: Input prices, labor costs, logistics affecting {sector}
- Analyst sector reports WITH specific company mentions or competitive comparisons

ANALYTICAL CONTENT - Include sector analysis:
✓ "Why [sector] companies are [performing/struggling]..." (explaining macro trends)
✓ "[Sector] industry faces [challenge/opportunity]..." (structural shifts)
✓ "Major players in [sector] [taking action]..." (coordinated industry moves)
✓ "[Peer company] success/failure signals [trend]..." (competitive intelligence)
✓ "Can [sector] sustain [growth/margins/demand]..." (industry viability questions)

REJECT COMPLETELY - Never select:
- Generic market research: "Industry to reach $XB by 20YY" WITHOUT {company_name} positioning
- Pure trend pieces: "Top 5 trends in [sector]," "Future of [industry]" WITHOUT specifics
- Single company news: Peer earnings, appointments WITHOUT broader sector implications
- Unrelated companies: Articles about non-peer companies outside {company_name}'s competitive set
- Distant forecasts: "20XX outlook," "Next decade in [sector]" WITHOUT near-term catalysts
- Pure opinion: "Analysis," "Commentary" WITHOUT hard data or specific sector insights
- Small company routine news: Junior partnerships, minor financing rounds
- Attribution confusion: Industry trade group reports where {sector} is just mentioned

COMPETITIVE INTELLIGENCE - Include when:
✓ Peer company action indicates sector direction (new product categories, pricing changes, geographic priorities)
✓ Regulatory action against competitor reveals industry-wide risks
✓ Supply chain disruption at major player affects {sector} availability/pricing
✓ Technology deployment shows sector-wide adoption affecting {company_name}
✓ Financial performance reveals {sector} cost/margin/demand trends
✓ Multiple peers make similar strategic moves (consolidation wave, geographic expansion)

SCRAPE PRIORITY (assign integer 1-3):
1 = Tier 1 (regulatory, pricing, supply shocks WITH numbers)
2 = Tier 2 (capacity, consolidation, policy WITH budgets, peer moves with sector implications)
3 = Tier 3 (TAM sizing, economic indicators, adoption metrics, sector reports)

SELECTION STANDARD:
- Prioritize sector-specific insights over general business news
- A trade publication covering {sector} specifically > major outlet with tangential mention
- Include if article reveals something material about {company_name}'s operating environment
- Skip if industry mention is generic or doesn't connect to {company_name}'s business

Return JSON array: [{{"id": 0, "scrape_priority": 1, "why": "brief reason"}}]

CRITICAL CONSTRAINT: Return UP TO {target_cap} articles. Select fewer if uncertain about sector relevance to {company_name}."""

    # Separate variable content for better caching
    user_content = f"Articles: {json.dumps(items, separators=(',', ':'))}"

    try:
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",  # Updated for prompt caching
            "content-type": "application/json"
        }

        data = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 2048,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}]
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

                # Sort by priority (1=HIGH, 2=MEDIUM, 3=LOW), then by recency
                selected_articles.sort(key=lambda x: (
                    x["scrape_priority"],  # Sort ascending: 1, 2, 3 (HIGH→MEDIUM→LOW)
                    -articles[x["id"]].get("published_at", datetime.min).timestamp() if articles[x["id"]].get("published_at") else 0
                ))

                # Cap at target after sorting (LOW-priority articles cut first)
                if len(selected_articles) > target_cap:
                    LOG.warning(f"Claude industry selected {len(selected_articles)}, capping to top {target_cap} by priority")
                    selected_articles = selected_articles[:target_cap]

                LOG.info(f"Claude triage industry: selected {len(selected_articles)}/{len(articles)} articles")
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

PRIMARY CRITERION: Is this article SPECIFICALLY about {competitor_name}? If unclear, skip it.

SELECT (choose up to {target_cap}):

TIER 1 - Hard corporate events (scrape_priority=1):
- Financial: "beats," "misses," "earnings," "revenue," "guidance," "margin," "profit," "loss," "EPS," "sales"
- Capital: "buyback," "dividend," "debt," "bond," "offering," "raises," "refinance," "converts"
- M&A: "acquires," "buys," "sells," "divests," "stake," "merger," "spin-off," "joint venture"
- Regulatory: "FDA," "SEC," "DOJ," "FTC," "investigation," "fine," "settlement," "approval," "lawsuit," "probe"
- Operations: "accident," "disaster," "halt," "shutdown," "recall," "expansion," "closure," "layoffs," "strike"
- Products: "approval," "launch," "recall," "trial results," "patent" WITH specific product/drug names
- Contracts: Dollar amounts in title (e.g., "$500M contract," "£2B deal")
- Ratings: "upgrade," "downgrade" WITH analyst firm name (e.g., "BofA upgrades," "Moody's cuts")
- Price movements: "surges," "jumps," "plunges," "drops" WITH percentage or specific levels

TIER 2 - Strategic competitive intelligence (scrape_priority=2):
- Leadership: CEO, CFO, President WITH "appoints," "names," "resigns," "retires," "replaces"
- Partnerships: Named partners (competitive threats or ecosystem plays)
- Technology: New products/platforms WITH "launches," "announces," "deploys"
- Facilities: Capacity expansions/closures WITH locations and scale
- Geographic: Market entry/exit WITH investment levels (competitive overlap with {ticker})
- Pricing: Price changes, discounting, bundling affecting competitive position
- Customer wins/losses: Major accounts, market share shifts
- Executive interviews: Strategy, roadmap, competitive positioning statements

TIER 3 - Competitive context (scrape_priority=3):
- Analyst coverage WITH price targets or competitive comparisons to {ticker}
- Performance analysis: "Why {competitor_name} [outperformed/underperformed]..."
- Strategic questions: "Can {competitor_name} compete with..." "Will {competitor_name} disrupt..."
- Awards, certifications affecting competitive positioning
- Technical analysis WITH specific price levels

ANALYTICAL CONTENT - Include competitor analysis:
✓ "Why {competitor_name} stock [moved]..." (understanding competitive threats/opportunities)
✓ "{competitor_name} vs {ticker}" or competitive comparisons
✓ "Can {competitor_name} [challenge/overtake/sustain]..." (competitive capability)
✓ "{competitor_name} strategy in [market/product]..." (strategic intelligence)
✓ "What {competitor_name}'s [move] means for..." (competitive implications)

REJECT COMPLETELY - Never select:
- Generic lists: "Top dividend stocks," "Best performers," "Stocks to watch"
- Sector roundups: "Tech movers," "Healthcare rally" (unless {competitor_name} is primary focus)
- Unrelated mentions: {competitor_name} listed among many tickers without focus
- Pure speculation: "Could 10x" WITHOUT specific competitive thesis
- Historical: "If you'd invested," "20 years of returns"
- Distant predictions: "2035 price prediction" WITHOUT near-term catalysts
- Market research: "Industry forecast" (unless specifically about {competitor_name})
- Quote pages: "Stock Price | Charts | [Exchange]"

DISAMBIGUATION - Critical accuracy:
- If title leads with different company, likely not about {competitor_name}
- If {competitor_name} is just news source attribution, reject
- For common names, verify context matches YOUR competitor
- Multi-company: Only select if {competitor_name} is ≥50% of focus

SCRAPE PRIORITY (assign integer 1-3):
1 = Tier 1 (financials, M&A, regulatory, disasters, contracts, price moves)
2 = Tier 2 (leadership, launches, partnerships, pricing, customer wins, interviews)
3 = Tier 3 (analyst coverage, performance analysis, competitive questions)

SELECTION STANDARD:
- When uncertain if about {competitor_name}, skip it
- Prioritize competitive intelligence relevance to {ticker}
- Niche source covering {competitor_name} specifically > major outlet tangential mention
- Only select if provides actionable intelligence about {competitor_name}

Return JSON array: [{{"id": 0, "scrape_priority": 1, "why": "brief reason"}}]

CRITICAL CONSTRAINT: Return UP TO {target_cap} articles. Select fewer if uncertain about relevance."""

    # Separate variable content for better caching
    user_content = f"Articles: {json.dumps(items, separators=(',', ':'))}"

    try:
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",  # Updated for prompt caching
            "content-type": "application/json"
        }

        data = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 2048,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}]
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

                # Sort by priority (1=HIGH, 2=MEDIUM, 3=LOW), then by recency
                selected_articles.sort(key=lambda x: (
                    x["scrape_priority"],  # Sort ascending: 1, 2, 3 (HIGH→MEDIUM→LOW)
                    -articles[x["id"]].get("published_at", datetime.min).timestamp() if articles[x["id"]].get("published_at") else 0
                ))

                # Cap at target after sorting (LOW-priority articles cut first)
                if len(selected_articles) > target_cap:
                    LOG.warning(f"Claude competitor selected {len(selected_articles)}, capping to top {target_cap} by priority")
                    selected_articles = selected_articles[:target_cap]

                LOG.info(f"Claude triage competitor: selected {len(selected_articles)}/{len(articles)} articles")
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

async def perform_ai_triage_with_fallback_async(
    articles_by_category: Dict[str, List[Dict]],
    ticker: str,
    triage_batch_size: int = 5  # Updated default to 5
) -> Dict[str, List[Dict]]:
    """
    Claude-first triage with OpenAI fallback: Claude runs first, OpenAI only on error.
    Replaces dual scoring with sequential fallback for cost savings and simplicity.
    """
    selected_results = {"company": [], "industry": [], "competitor": []}

    if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
        LOG.warning("No AI API keys configured - skipping triage")
        return selected_results

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

    LOG.info(f"=== TRIAGE WITH FALLBACK (Claude primary, OpenAI fallback): batch_size={triage_batch_size} ===")

    # Collect ALL triage operations
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
    LOG.info(f"Total triage operations: {total_operations}")

    if total_operations == 0:
        return selected_results

    # Track API usage statistics
    claude_success_count = 0
    openai_fallback_count = 0
    both_failed_count = 0

    # Process operations in batches
    for batch_start in range(0, total_operations, triage_batch_size):
        batch_end = min(batch_start + triage_batch_size, total_operations)
        batch = all_triage_operations[batch_start:batch_end]
        batch_num = (batch_start // triage_batch_size) + 1
        total_batches = (total_operations + triage_batch_size - 1) // triage_batch_size

        LOG.info(f"BATCH {batch_num}/{total_batches}: Processing {len(batch)} operations (Claude primary, OpenAI fallback):")
        for op in batch:
            LOG.info(f"  - {op['type']}: {op['key']} ({len(op['articles'])} articles)")

        # Create fallback tasks for each operation (Claude → OpenAI on error)
        batch_tasks = []
        for op in batch:
            async def run_triage_with_fallback(operation):
                """Try Claude first, fallback to OpenAI on error"""
                result = []
                api_used = None

                # Try Claude first
                if ANTHROPIC_API_KEY:
                    try:
                        claude_result = await operation["claude_func"](*operation["claude_args"])
                        if claude_result is not None:  # Success (even if empty list)
                            api_used = "claude"
                            result = claude_result
                            LOG.info(f"  ✓ Claude succeeded for {operation['type']}/{operation['key']}: {len(result)} articles")
                            return result, api_used
                        else:
                            LOG.warning(f"  ⚠️ Claude returned None for {operation['type']}/{operation['key']}, falling back to OpenAI")
                    except Exception as e:
                        LOG.warning(f"  ⚠️ Claude failed for {operation['type']}/{operation['key']}: {e}, falling back to OpenAI")

                # Fallback to OpenAI
                if OPENAI_API_KEY:
                    try:
                        openai_result = await operation["openai_func"](*operation["openai_args"])
                        api_used = "openai"
                        result = openai_result if openai_result is not None else []
                        LOG.info(f"  ✓ OpenAI fallback succeeded for {operation['type']}/{operation['key']}: {len(result)} articles")
                        return result, api_used
                    except Exception as e:
                        LOG.error(f"  ❌ OpenAI fallback failed for {operation['type']}/{operation['key']}: {e}")

                # Both failed
                LOG.error(f"  ❌ Both Claude and OpenAI failed for {operation['type']}/{operation['key']}")
                return [], None

            task = run_triage_with_fallback(op)
            batch_tasks.append((op, task))

        # Execute batch concurrently
        batch_results = await asyncio.gather(*[task for _, task in batch_tasks], return_exceptions=True)

        # Process results and track API usage
        for i, result_tuple in enumerate(batch_results):
            op = batch_tasks[i][0]

            if isinstance(result_tuple, Exception):
                LOG.error(f"Triage failed for {op['type']}/{op['key']}: {result_tuple}")
                both_failed_count += 1
                continue

            # Unpack result and API used
            result, api_used = result_tuple

            # Track API usage
            if api_used == "claude":
                claude_success_count += 1
            elif api_used == "openai":
                openai_fallback_count += 1
            else:
                both_failed_count += 1

            # Add score fields based on which API was used (1=High, 2=Medium, 3=Low)
            for item in result:
                if api_used == "claude":
                    item["claude_score"] = item.get("scrape_priority", 0)  # Use priority directly (1=High, 2=Med, 3=Low)
                    item["openai_score"] = 0  # OpenAI didn't score this
                elif api_used == "openai":
                    item["openai_score"] = item.get("scrape_priority", 0)  # Use priority directly
                    item["claude_score"] = 0  # Claude didn't score this
                else:
                    # Both failed
                    item["openai_score"] = 0
                    item["claude_score"] = 0

            # Map results back to original indices and add to selected_results
            if op["type"] == "company":
                selected_results["company"].extend(result)

            elif op["type"] == "industry":
                # Map back to original indices
                for selected_item in result:
                    original_idx = op["index_mapping"][selected_item["id"]]["original_idx"]
                    selected_item["id"] = original_idx
                selected_results["industry"].extend(result)

            elif op["type"] == "competitor":
                # Map back to original indices
                for selected_item in result:
                    original_idx = op["index_mapping"][selected_item["id"]]["original_idx"]
                    selected_item["id"] = original_idx
                selected_results["competitor"].extend(result)

        LOG.info(f"BATCH {batch_num} COMPLETE")

    LOG.info(f"TRIAGE WITH FALLBACK COMPLETE:")
    LOG.info(f"  Company: {len(selected_results['company'])} selected")
    LOG.info(f"  Industry: {len(selected_results['industry'])} selected")
    LOG.info(f"  Competitor: {len(selected_results['competitor'])} selected")
    LOG.info(f"  API Usage: Claude primary: {claude_success_count}/{total_operations} | OpenAI fallback: {openai_fallback_count}/{total_operations} | Both failed: {both_failed_count}/{total_operations}")

    return selected_results

# ===== END FALLBACK TRIAGE LOGIC =====

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
                # SEMAPHORE DISABLED: Prevents threading deadlock with concurrent tickers
                # with TRIAGE_SEM:
                if True:  # Maintain indentation
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
    Priority: ticker_reference.company_name -> search_keyword -> fallback

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
                    WHERE ticker = %s
                    LIMIT 1
                """, (competitor_ticker,))
                result = cur.fetchone()

                if result and result["company_name"]:
                    return result["company_name"]
        except Exception as e:
            LOG.debug(f"ticker_reference lookup failed for competitor {competitor_ticker}: {e}")

    # STEP 2: Fallback to search_keyword ONLY if it looks like a company name (not ticker)
    if search_keyword and not search_keyword.isupper():  # Likely a company name, not ticker
        return search_keyword

    # STEP 3: Final fallback - use ticker if that's all we have
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

def extract_title_words_normalized(title: str, num_words: int = 10) -> str:
    """
    Extract first N words from title, normalized and concatenated (no spaces)
    Used for Google News deduplication when resolved URL not available

    Example: "Tesla Stock Rises 5% After Earnings" → "teslastockrises5afterearnings"
    """
    if not title:
        return ""

    # Remove all non-alphanumeric characters except spaces
    clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', title.lower())

    # Split into words, take first N
    words = clean_title.split()[:num_words]

    # Join without spaces
    return ''.join(words)

def get_url_hash(url: str, resolved_url: str = None, domain: str = None, title: str = None) -> str:
    """
    Generate hash for URL deduplication (simple URL-based)

    Uses resolved_url if available (Yahoo Finance URLs), otherwise uses original URL.
    Cross-feed duplicates (Google News + Yahoo → same article) are caught later in
    post-resolution deduplication (Phase 1.5) before AI analysis.
    """
    # Use resolved URL if available (Yahoo Finance URLs that were resolved during ingestion)
    if resolved_url:
        url_clean = re.sub(r'[?&](utm_|ref=|source=|siteid=|cid=|\.tsrc=).*', '', resolved_url.lower())
        url_clean = url_clean.rstrip('/')
        return hashlib.md5(url_clean.encode()).hexdigest()

    # Otherwise use original URL (Google News URLs, Direct URLs)
    url_clean = re.sub(r'[?&](utm_|ref=|source=|siteid=|cid=|\.tsrc=).*', '', url.lower())
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
                return article_url

        except Exception as e:
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

        LOG.info(f"🔍 [GOOGLE_NEWS] Attempting resolution for: {url[:100]}...")

        # Try advanced resolution first
        LOG.info(f"🔄 [GOOGLE_NEWS] Trying advanced API resolution...")
        advanced_url = self._resolve_google_news_url_advanced(url)
        if advanced_url:
            domain = normalize_domain(urlparse(advanced_url).netloc.lower())
            if not self._is_spam_domain(domain):
                LOG.info(f"✅ [GOOGLE_NEWS] Advanced resolution: {url[:80]} -> {advanced_url}")
                return advanced_url, domain, None
            else:
                LOG.info(f"SPAM REJECTED: Advanced resolution found spam domain {domain}")
                return None, None, None  # Reject entirely, don't fall back
        else:
            LOG.info(f"❌ [GOOGLE_NEWS] Advanced API resolution returned None")

        # Fall back to direct resolution method
        LOG.info(f"🔄 [GOOGLE_NEWS] Trying direct HTTP redirect...")
        try:
            response = requests.get(url, timeout=10, allow_redirects=True, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            final_url = response.url

            if final_url != url and "news.google.com" not in final_url:
                domain = normalize_domain(urlparse(final_url).netloc.lower())
                if not self._is_spam_domain(domain):
                    LOG.info(f"✅ [GOOGLE_NEWS] Direct resolution: {url[:80]} -> {final_url}")
                    return final_url, domain, None
                else:
                    LOG.info(f"SPAM REJECTED: Direct resolution found spam domain {domain}")
                    return None, None, None  # Reject entirely
            else:
                LOG.info(f"❌ [GOOGLE_NEWS] Direct HTTP didn't redirect (final_url still Google News)")
        except Exception as e:
            LOG.info(f"❌ [GOOGLE_NEWS] Direct HTTP failed: {str(e)[:100]}")

        # Title extraction fallback - also check for spam
        LOG.info(f"🔄 [GOOGLE_NEWS] Trying title extraction from: '{title[:60] if title else 'NO TITLE'}...'")
        if title and not contains_non_latin_script(title):
            clean_title, source = extract_source_from_title_smart(title)
            LOG.info(f"   Title parser extracted source: '{source}'")
            if source and not self._is_spam_source(source):
                resolved_domain = self._resolve_publication_to_domain(source)
                LOG.info(f"   Domain resolver returned: '{resolved_domain}'")
                if resolved_domain:
                    if not self._is_spam_domain(resolved_domain):
                        LOG.info(f"✅ [GOOGLE_NEWS] Title resolution: {source} -> {resolved_domain}")
                        return url, resolved_domain, None
                    else:
                        LOG.info(f"SPAM REJECTED: Title resolution found spam domain {resolved_domain}")
                        return None, None, None
                else:
                    LOG.warning(f"❌ [GOOGLE_NEWS] Could not resolve publication '{source}' to domain")
                    return url, "google-news-unresolved", None
            else:
                LOG.info(f"❌ [GOOGLE_NEWS] Title extraction: No valid source extracted or spam source")
        else:
            LOG.info(f"❌ [GOOGLE_NEWS] Title extraction: Title is empty or contains non-Latin script")

        LOG.warning(f"❌ [GOOGLE_NEWS] ALL 3 RESOLUTION METHODS FAILED for: {url[:100]}")
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
                    SELECT ticker, company_name,
                           industry_keyword_1, industry_keyword_2, industry_keyword_3,
                           competitor_1_name, competitor_1_ticker,
                           competitor_2_name, competitor_2_ticker,
                           competitor_3_name, competitor_3_ticker,
                           ai_generated
                    FROM ticker_reference WHERE ticker = %s AND active = TRUE
                """, (ticker,))
                config = cur.fetchone()

                if config:
                    # Reconstruct industry_keywords array from individual columns
                    industry_keywords = []
                    for i in range(1, 4):
                        kw = config.get(f"industry_keyword_{i}")
                        if kw:
                            industry_keywords.append(kw)

                    # Reconstruct competitors array from individual columns
                    competitors = []
                    for i in range(1, 4):
                        comp_name = config.get(f"competitor_{i}_name")
                        comp_ticker = config.get(f"competitor_{i}_ticker")
                        if comp_name:
                            competitors.append({
                                "name": comp_name,
                                "ticker": comp_ticker if comp_ticker else None
                            })

                    return {
                        "company_name": config.get("company_name", ticker),
                        "industry_keywords": industry_keywords,
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

        # CRITICAL GUARD: Never store fallback data (company_name == ticker)
        # This prevents overwriting good CSV data with AI fallback data
        if metadata and metadata.get("company_name") == ticker:
            LOG.warning(f"⚠️ Refusing to store fallback metadata for {ticker} (company_name == ticker). This prevents database corruption.")
            return

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
- Must be DIRECT business competitors where the MAJORITY of both companies' revenues/operations compete in the same markets with similar products or services
- NOT companies with only minor product overlap (e.g., Autodesk is NOT a Figma competitor despite both having design tools)
- NOT companies in the same sector but serving different customers or markets
- Prefer publicly traded companies with tickers when possible
- For private companies: Include name but omit or set ticker to empty string
- Company names should be the common/brand name ONLY (e.g., "Canva" not "Canva Pty Ltd", "Adobe" not "Adobe Inc")
- Format as structured objects with 'name' and 'ticker' fields
- Verify ticker is correct Yahoo Finance format (if provided)
- Exclude: Subsidiaries, companies acquired in last 2 years

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
- Must be DIRECT business competitors where the MAJORITY of both companies' revenues/operations compete in the same markets with similar products or services
- NOT companies with only minor product overlap (e.g., Autodesk is NOT a Figma competitor despite both having design tools)
- NOT companies in the same sector but serving different customers or markets
- Prefer publicly traded companies with tickers when possible
- For private companies: Include name but omit or set ticker to empty string
- Company names should be the common/brand name ONLY (e.g., "Canva" not "Canva Pty Ltd", "Adobe" not "Adobe Inc")
- Format as structured objects with 'name' and 'ticker' fields
- Verify ticker is correct and current Yahoo Finance format (if provided)
- Exclude: Subsidiaries, companies acquired in last 2 years

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
    
    # === No config row, fallback to AI generation + INSERT ===
    LOG.info("DEBUG: Entering fallback AI generation path")

    if OPENAI_API_KEY:
        # CRITICAL: Try to fetch company name from multiple sources BEFORE calling AI without a hint
        LOG.info(f"⚠️ No reference data found for {ticker}, trying external sources for company name...")

        # Try yfinance first (fast, free, but gets throttled easily)
        company_name_from_source = fetch_company_name_from_yfinance(normalized_ticker)
        source_used = "yfinance"

        # If yfinance fails, try Polygon.io (slower, rate limited, but more reliable)
        if not company_name_from_source:
            LOG.info(f"⚠️ yfinance failed for {ticker}, trying Polygon.io fallback...")
            company_name_from_source = fetch_company_name_from_polygon(normalized_ticker)
            source_used = "Polygon.io"

        if company_name_from_source:
            LOG.info(f"✅ {source_used} provided company name: {company_name_from_source}")
            # Call AI with company name hint from external source
            ai_metadata = generate_enhanced_ticker_metadata_with_ai(
                normalized_ticker,
                company_name=company_name_from_source
            )
        else:
            LOG.warning(f"⚠️ Both yfinance and Polygon.io failed for {ticker}, calling AI without hint")
            # Last resort: Call AI without company name hint
            ai_metadata = generate_enhanced_ticker_metadata_with_ai(normalized_ticker)

        # Validate AI response - warn if company_name == ticker, but ALLOW it
        if ai_metadata and ai_metadata.get('company_name') == normalized_ticker:
            LOG.warning(f"⚠️ AI returned ticker symbol as company name for {ticker}")
            LOG.warning(f"   This means: 1) CSV not loaded, 2) yfinance failed, 3) Polygon.io failed, 4) AI failed")
            LOG.warning(f"   Continuing with company_name='{normalized_ticker}' to avoid crash")

        # Store the AI-generated data back to reference table for future use
        if ai_metadata:
            reference_data = {
                'ticker': normalized_ticker,
                'country': 'US',
                'company_name': ai_metadata.get('company_name', normalized_ticker),  # Fallback to ticker if missing
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

        return ai_metadata or {"ticker": normalized_ticker, "company_name": normalized_ticker, "industry_keywords": [], "competitors": []}

    # Step 3: Final fallback - NO AI configured (use ticker as company name)
    LOG.warning(f"⚠️ No data found for {ticker} and no AI configured")
    LOG.warning(f"   Using ticker symbol as company name to avoid crash")
    return {"ticker": normalized_ticker, "company_name": normalized_ticker, "industry_keywords": [], "competitors": []}

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
    """UPSERT reference table with AI-generated enhancements (INSERT if new, UPDATE if exists)"""
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

        LOG.info(f"DEBUG: UPSERT {ticker} with keywords={[keyword_1, keyword_2, keyword_3]} and competitors={comp_data}")

        with db() as conn, conn.cursor() as cur:
            # UPSERT: INSERT new ticker or UPDATE existing one
            cur.execute("""
                INSERT INTO ticker_reference (
                    ticker, company_name, sector, industry, sub_industry,
                    industry_keyword_1, industry_keyword_2, industry_keyword_3,
                    competitor_1_name, competitor_1_ticker,
                    competitor_2_name, competitor_2_ticker,
                    competitor_3_name, competitor_3_ticker,
                    ai_generated, ai_enhanced_at, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, NOW(), NOW(), NOW())
                ON CONFLICT (ticker) DO UPDATE SET
                    industry_keyword_1 = EXCLUDED.industry_keyword_1,
                    industry_keyword_2 = EXCLUDED.industry_keyword_2,
                    industry_keyword_3 = EXCLUDED.industry_keyword_3,
                    competitor_1_name = EXCLUDED.competitor_1_name,
                    competitor_1_ticker = EXCLUDED.competitor_1_ticker,
                    competitor_2_name = EXCLUDED.competitor_2_name,
                    competitor_2_ticker = EXCLUDED.competitor_2_ticker,
                    competitor_3_name = EXCLUDED.competitor_3_name,
                    competitor_3_ticker = EXCLUDED.competitor_3_ticker,
                    ai_generated = TRUE,
                    ai_enhanced_at = NOW(),
                    updated_at = NOW()
            """, (
                normalize_ticker_format(ticker),
                metadata.get('company_name', ticker),
                metadata.get('sector') if metadata.get('sector') not in ['Unknown', ''] else None,
                metadata.get('industry') if metadata.get('industry') not in ['Unknown', ''] else None,
                metadata.get('sub_industry', ''),
                keyword_1, keyword_2, keyword_3,
                comp_data['competitor_1_name'],
                comp_data['competitor_1_ticker'],
                comp_data['competitor_2_name'],
                comp_data['competitor_2_ticker'],
                comp_data['competitor_3_name'],
                comp_data['competitor_3_ticker']
            ))

            LOG.info(f"✅ UPSERT successful for {ticker} reference table with AI enhancements")

    except Exception as e:
        LOG.error(f"Failed to UPSERT ticker reference AI data for {ticker}: {e}")

def update_ticker_reference_financial_data(ticker: str, financial_data: Dict):
    """UPSERT reference table with financial data from yfinance (INSERT if new, UPDATE if exists)"""
    try:
        with db() as conn, conn.cursor() as cur:
            # UPSERT: INSERT new ticker or UPDATE existing one with financial data
            cur.execute("""
                INSERT INTO ticker_reference (
                    ticker, company_name,
                    financial_last_price, financial_price_change_pct, financial_yesterday_return_pct,
                    financial_ytd_return_pct, financial_market_cap, financial_enterprise_value,
                    financial_volume, financial_avg_volume, financial_analyst_target,
                    financial_analyst_range_low, financial_analyst_range_high, financial_analyst_count,
                    financial_analyst_recommendation, financial_snapshot_date,
                    created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (ticker) DO UPDATE SET
                    financial_last_price = EXCLUDED.financial_last_price,
                    financial_price_change_pct = EXCLUDED.financial_price_change_pct,
                    financial_yesterday_return_pct = EXCLUDED.financial_yesterday_return_pct,
                    financial_ytd_return_pct = EXCLUDED.financial_ytd_return_pct,
                    financial_market_cap = EXCLUDED.financial_market_cap,
                    financial_enterprise_value = EXCLUDED.financial_enterprise_value,
                    financial_volume = EXCLUDED.financial_volume,
                    financial_avg_volume = EXCLUDED.financial_avg_volume,
                    financial_analyst_target = EXCLUDED.financial_analyst_target,
                    financial_analyst_range_low = EXCLUDED.financial_analyst_range_low,
                    financial_analyst_range_high = EXCLUDED.financial_analyst_range_high,
                    financial_analyst_count = EXCLUDED.financial_analyst_count,
                    financial_analyst_recommendation = EXCLUDED.financial_analyst_recommendation,
                    financial_snapshot_date = EXCLUDED.financial_snapshot_date,
                    updated_at = NOW()
            """, (
                normalize_ticker_format(ticker),
                ticker,  # Use ticker as fallback company_name if creating new row
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
                financial_data.get('financial_snapshot_date')
            ))

            LOG.info(f"✅ UPSERT successful for {ticker} with financial data (snapshot: {financial_data.get('financial_snapshot_date')})")

    except Exception as e:
        LOG.error(f"Failed to UPSERT ticker reference financial data for {ticker}: {e}")

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

def _build_executive_summary_prompt(ticker: str, categories: Dict[str, List[Dict]], config: Dict) -> Optional[tuple[str, str, str]]:
    """Helper: Build executive summary prompt and extract company name. Returns (system_prompt, user_content, company_name) or None."""
    company_name = config.get("name", ticker)

    # Collect ALL flagged articles across all categories
    all_flagged_articles = []

    # Company articles
    for article in categories.get("company", []):
        if article.get("ai_summary"):
            article['_category'] = 'COMPANY'
            article['_category_tag'] = '[COMPANY]'
            all_flagged_articles.append(article)

    # Industry articles
    for article in categories.get("industry", []):
        if article.get("ai_summary"):
            keyword = article.get("search_keyword", "Industry")
            article['_category'] = 'INDUSTRY'
            article['_category_tag'] = f'[INDUSTRY - {keyword}]'
            all_flagged_articles.append(article)

    # Competitor articles
    for article in categories.get("competitor", []):
        if article.get("ai_summary"):
            article['_category'] = 'COMPETITOR'
            article['_category_tag'] = '[COMPETITOR]'
            all_flagged_articles.append(article)

    # Must have at least some content
    if not all_flagged_articles:
        LOG.warning(f"[{ticker}] No articles with AI summaries - skipping executive summary")
        return None

    # Sort all articles globally by published_at DESC (newest first)
    all_flagged_articles.sort(
        key=lambda x: x.get("published_at") or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True
    )

    # Build unified timeline with category tags
    unified_timeline = []
    for article in all_flagged_articles[:50]:  # Limit to 50 most recent
        title = article.get("title", "")
        ai_summary = article.get("ai_summary", "")
        domain = article.get("domain", "")
        published_at = article.get("published_at")
        date_str = format_date_short(published_at)
        category_tag = article.get("_category_tag", "[UNKNOWN]")

        if ai_summary:
            source_name = get_or_create_formal_domain_name(domain) if domain else "Unknown Source"
            unified_timeline.append(f"• {category_tag} {title} [{source_name}] {date_str}: {ai_summary}")

    user_content = "UNIFIED ARTICLE TIMELINE (newest to oldest):\n" + "\n".join(unified_timeline)

    # System instructions (ticker-specific but cacheable per ticker)
    system_prompt = f"""You are a hedge fund analyst creating an intelligence summary for {company_name} ({ticker}). All article summaries are already written from {ticker} investor perspective.

INPUT FORMAT:
Articles are provided in a UNIFIED TIMELINE sorted newest to oldest. Each article has a category tag:
- [COMPANY] = Articles directly about {ticker}
- [INDUSTRY - keyword] = Industry/sector articles relevant to {ticker}
- [COMPETITOR] = Articles about {ticker}'s competitors

OUTPUT FORMAT - Use these exact headers (omit sections with no content):
DO NOT use markdown headers (##) or add title lines. Start directly with emoji headers EXACTLY as shown:
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
- Include all material developments, but keep bullets concise
- If uncertain about materiality, include it - but in ONE sentence
- Include transaction amounts when available for scale context
- Strategic significance matters more than transaction size

---

🔴 MAJOR DEVELOPMENTS (3-6 bullets)
Source: Primarily [COMPANY] articles, plus relevant [INDUSTRY] and [COMPETITOR] articles with competitive implications

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
Source: [COMPANY] articles only

Quantified metrics only. Include:
- Earnings, revenue, guidance, margins with exact figures
- Report vs. consensus when mentioned
- Production metrics, capacity changes, operational KPIs
- Capex, debt, buybacks, dividends with amounts
- Transaction sizes when disclosed

---

⚠️ RISK FACTORS (2-4 bullets)
Source: [COMPANY], [INDUSTRY], and [COMPETITOR] articles

Include threats with impact/timeline when available:
- {ticker} operational risks: Production issues, supply chain, quality problems
- {ticker} regulatory/legal: Investigations, lawsuits, compliance with financial impact
- Competitive threats: Competitor actions directly threatening {ticker} position
- Industry headwinds: Sector trends creating risks for {ticker}
- Insider activity: C-suite selling with amounts/context

---

📈 WALL STREET SENTIMENT (1-4 bullets)
Source: [COMPANY] articles only

Analyst actions on {ticker} only.

Format: "[Firm] [action] to [new rating/target], [rationale if given] (date)"

If 3+ analysts moved same direction in same week:
"Multiple firms upgraded this week: [Firm 1] to $X, [Firm 2] to $Y, [Firm 3] to $Z (Oct 1-3)"

---

⚡ COMPETITIVE/INDUSTRY DYNAMICS (2-5 bullets)
Source: [INDUSTRY] and [COMPETITOR] articles (already written with {ticker} impact framing)

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
Source: [COMPANY] articles only

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

Generate structured summary. Omit empty sections."""

    return (system_prompt, user_content, company_name)


def generate_claude_executive_summary(ticker: str, categories: Dict[str, List[Dict]], config: Dict) -> Optional[str]:
    """Generate executive summary using Claude API (primary method)"""
    if not ANTHROPIC_API_KEY:
        return None

    # Build prompt using shared helper
    result = _build_executive_summary_prompt(ticker, categories, config)
    if not result:
        return None

    system_prompt, user_content, company_name = result

    try:
        # Log prompt sizes for debugging 520 errors
        system_tokens_est = len(system_prompt) // 4
        user_tokens_est = len(user_content) // 4
        total_tokens_est = system_tokens_est + user_tokens_est
        LOG.info(f"[{ticker}] Executive summary prompt size: system={len(system_prompt)} chars (~{system_tokens_est} tokens), user={len(user_content)} chars (~{user_tokens_est} tokens), total=~{total_tokens_est} tokens")

        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",  # Updated for prompt caching
            "content-type": "application/json"
        }

        data = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 10000,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_content
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

    system_prompt, user_content, company_name = result

    # For OpenAI, combine system and user content (format: instructions + article summaries)
    prompt = f"{system_prompt}\n\nALL ARTICLE SUMMARIES:\n{user_content}"

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


async def generate_ai_final_summaries(articles_by_ticker: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, Dict[str, str]]:
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
            # NEW badges removed per user request (Oct 2025)
            all_articles = (
                [a for a in categories.get("company", []) if a.get("ai_summary")] +
                [a for a in categories.get("industry", []) if a.get("ai_summary")] +
                [a for a in categories.get("competitor", []) if a.get("ai_summary")]
            )
            # ai_analysis_summary = insert_new_badges(ai_analysis_summary, all_articles)  # DISABLED
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
            "industry_articles_analyzed": len([a for a in categories.get("industry", []) if a.get("ai_summary")]),
            "model_used": model_used  # Track which AI model generated the summary
        }

    LOG.info(f"🎯 EXECUTIVE SUMMARY: Completed - generated summaries for {len(summaries)} tickers")
    return summaries


# ------------------------------------------------------------------------------
# 3-EMAIL SYSTEM - Article Sorting and Email Functions
# ------------------------------------------------------------------------------

def sort_articles_chronologically(articles: List[Dict]) -> List[Dict]:
    """Sort articles by published_at DESC (newest first), regardless of quality or flagged status"""
    return sorted(
        articles,
        key=lambda x: x.get('published_at') or x.get('found_at') or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True
    )


def send_email(subject: str, html_body: str, to: str | None = None, bcc: str | None = None) -> bool:
    """Send email with HTML body only (no attachments). Supports BCC (hidden from recipients)."""
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
        # NOTE: BCC header is NOT added to message (hidden from recipients)

        # Plain-text fallback
        text_body = "This email contains HTML content. Please view in an HTML-capable email client."
        msg.attach(MIMEText(text_body, "plain", "utf-8"))

        # HTML body
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        LOG.info(f"Connecting to SMTP server: {SMTP_HOST}:{SMTP_PORT}")

        # Build recipient list (includes BCC, but not shown in headers)
        recipients = [recipient]
        if bcc:
            recipients.append(bcc)
            LOG.info(f"BCC enabled: {bcc} (hidden from recipient)")

        # Add timeout to SMTP operations
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=60) as server:
            if SMTP_STARTTLS:
                LOG.info("Starting TLS...")
                server.starttls()
            LOG.info("Logging in to SMTP server...")
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            LOG.info("Sending email...")
            server.sendmail(EMAIL_FROM, recipients, msg.as_string())

        LOG.info(f"Email sent successfully to {recipient}" + (f" (BCC: {bcc})" if bcc else ""))
        return True

    except smtplib.SMTPException as e:
        LOG.error(f"SMTP error sending email: {e}")
        return False
    except Exception as e:
        LOG.error(f"Email send failed: {e}")
        LOG.error(f"Error details: {traceback.format_exc()}")
        return False


def send_beta_signup_notification(name: str, email: str, ticker1: str, ticker2: str, ticker3: str) -> bool:
    """
    Send admin notification email for new beta signup.
    Returns True if email sent successfully, False otherwise.
    """
    try:
        # Get company names for better readability
        companies = []
        for ticker in [ticker1, ticker2, ticker3]:
            config = get_ticker_config(ticker)
            if config:
                companies.append(f"{ticker} ({config.get('company_name', 'Unknown')})")
            else:
                companies.append(ticker)

        html_body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px;">
            <h2 style="color: #1e40af;">🎉 New Beta User Signed Up!</h2>

            <div style="background: #f3f4f6; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <p style="margin: 8px 0;"><strong>Name:</strong> {name}</p>
                <p style="margin: 8px 0;"><strong>Email:</strong> {email}</p>
                <p style="margin: 8px 0;"><strong>Tracking:</strong></p>
                <ul style="margin: 8px 0; padding-left: 20px;">
                    <li>{companies[0]}</li>
                    <li>{companies[1]}</li>
                    <li>{companies[2]}</li>
                </ul>
                <p style="margin: 8px 0; color: #6b7280; font-size: 14px;">
                    <strong>Signed up:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p EST')}
                </p>
            </div>

            <p style="color: #6b7280; font-size: 14px;">
                This user will be included in tomorrow's 7 AM processing run.
            </p>
        </div>
        """

        subject = f"🎉 New Beta User: {name} tracking {ticker1}, {ticker2}, {ticker3}"

        return send_email(subject=subject, html_body=html_body, to=ADMIN_EMAIL)

    except Exception as e:
        LOG.error(f"Failed to send beta signup notification: {e}")
        return False


def export_beta_users_to_csv() -> int:
    """
    Export active beta users to CSV for daily processing.
    Called daily at 11:00 PM EST via Render cron.

    Returns: Number of users exported
    """
    output_path = "data/user_tickers.csv"

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT name, email, ticker1, ticker2, ticker3
                FROM beta_users
                WHERE status = 'active'
                ORDER BY email
            """)
            users = cur.fetchall()

        # Write CSV with header
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'email', 'ticker1', 'ticker2', 'ticker3'])
            for user in users:
                writer.writerow([user['name'], user['email'], user['ticker1'], user['ticker2'], user['ticker3']])

        LOG.info(f"✅ Exported {len(users)} active beta users to {output_path}")
        return len(users)

    except Exception as e:
        LOG.error(f"❌ CSV export failed: {e}")
        LOG.error(traceback.format_exc())
        raise


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

                # Chronological sorting - newest to oldest within category
                enhanced_articles = []
                for idx, article in enumerate(articles):
                    domain = normalize_domain(article.get("domain", ""))
                    is_ai_selected = idx in selected_article_data
                    is_quality_domain = domain in QUALITY_DOMAINS
                    is_problematic = domain in PROBLEMATIC_SCRAPE_DOMAINS

                    # Extract OpenAI and Claude scores from triage data
                    openai_score = selected_article_data[idx].get("openai_score", 0) if is_ai_selected else 0
                    claude_score = selected_article_data[idx].get("claude_score", 0) if is_ai_selected else 0

                    enhanced_articles.append({
                        "article": article,
                        "idx": idx,
                        "is_ai_selected": is_ai_selected,
                        "is_quality_domain": is_quality_domain,
                        "is_problematic": is_problematic,
                        "openai_score": openai_score,
                        "claude_score": claude_score,
                        "published_at": article.get("published_at")
                    })

                # Sort chronologically (newest first)
                enhanced_articles.sort(key=lambda x: (
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
                        # Use company_name from article metadata (populated from feeds.company_name)
                        comp_name = article.get('company_name') or get_competitor_display_name(article.get('search_keyword'), article.get('competitor_ticker'))
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

                    # 5. OpenAI Score - 1=High, 2=Medium, 3=Low, 0=None
                    openai_score = enhanced_article.get("openai_score", 0)
                    if openai_score == 1:
                        openai_class = "openai-high"
                        openai_level = "OpenAI: High"
                        openai_emoji = "🔥"
                    elif openai_score == 2:
                        openai_class = "openai-medium"
                        openai_level = "OpenAI: Medium"
                        openai_emoji = "⚡"
                    elif openai_score >= 3:
                        openai_class = "openai-low"
                        openai_level = "OpenAI: Low"
                        openai_emoji = "🔋"
                    else:
                        openai_class = "openai-none"
                        openai_level = "OpenAI: None"
                        openai_emoji = "○"
                    header_badges.append(f'<span class="ai-triage {openai_class}">{openai_emoji} {openai_level}</span>')

                    # 6. Claude Score - 1=High, 2=Medium, 3=Low, 0=None
                    claude_score = enhanced_article.get("claude_score", 0)
                    if claude_score == 1:
                        claude_class = "claude-high"
                        claude_level = "Claude: High"
                        claude_emoji = "🏆"
                    elif claude_score == 2:
                        claude_class = "claude-medium"
                        claude_level = "Claude: Medium"
                        claude_emoji = "💎"
                    elif claude_score >= 3:
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


async def build_enhanced_digest_html(articles_by_ticker: Dict[str, Dict[str, List[Dict]]], period_days: int,
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

    # Generate summaries using Claude (primary) with OpenAI fallback
    openai_summaries = await generate_ai_final_summaries(articles_by_ticker)  # Legacy variable name, actually uses Claude→OpenAI fallback

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

        # Display executive summary (Claude primary, OpenAI fallback)
        openai_summary = openai_summaries.get(ticker, {}).get("ai_analysis_summary", "")
        model_used = openai_summaries.get(ticker, {}).get("model_used", "AI")  # Get actual model used

        if openai_summary:
            html.append("<div class='company-summary'>")
            html.append(f"<div class='summary-title'>📰 Executive Summary (Deep Analysis) - {model_used}</div>")
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

        # Sort articles chronologically within each category (newest first)
        for category in ["company", "industry", "competitor"]:
            if category in categories and categories[category]:
                articles = categories[category]

                # Simple chronological sorting (newest first)
                articles = sort_articles_chronologically(articles)

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


async def fetch_digest_articles_with_enhanced_content(hours: int = 24, tickers: List[str] = None,
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
    if flagged_article_ids is not None and len(flagged_article_ids) > 0:
        LOG.info(f"Flagged article filter: ENABLED ({len(flagged_article_ids)} IDs)")
    else:
        LOG.warning(f"Flagged article filter: DISABLED - using relevance_score >= 7.0 filter as fallback")

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
                        ta.competitor_ticker,
                        ta.relevance_score, ta.relevance_reason, ta.is_rejected
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
                # ERROR: No flagged IDs - don't pull ALL articles
                LOG.error(f"❌ Email #2: No flagged_article_ids provided for {tickers}")
                # Return empty query instead of all articles
                cur.execute("SELECT NULL LIMIT 0")
        else:
            if flagged_article_ids:
                cur.execute("""
                    SELECT DISTINCT ON (a.url_hash, ta.ticker)
                        a.id, a.url, a.resolved_url, a.title, a.description,
                        ta.ticker, a.domain, a.published_at,
                        ta.found_at, ta.category,
                        ta.search_keyword, ta.ai_summary, ta.ai_model,
                        a.scraped_content, a.content_scraped_at, a.scraping_failed, a.scraping_error,
                        ta.competitor_ticker,
                        ta.relevance_score, ta.relevance_reason, ta.is_rejected
                    FROM articles a
                    JOIN ticker_articles ta ON a.id = ta.article_id
                    WHERE ta.found_at >= %s
                        AND (a.published_at >= %s OR a.published_at IS NULL)
                        AND a.id = ANY(%s)
                    ORDER BY a.url_hash, ta.ticker,
                        COALESCE(a.published_at, ta.found_at) DESC, ta.found_at DESC
                """, (cutoff, cutoff, flagged_article_ids))
            else:
                # ERROR: No flagged IDs - don't pull ALL articles
                LOG.error(f"❌ Email #2: No flagged_article_ids provided (all tickers)")
                # Return empty query instead of all articles
                cur.execute("SELECT NULL LIMIT 0")

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
    html = await build_enhanced_digest_html(articles_by_ticker, days if days > 0 else 1,
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


def build_executive_summary_html(sections: Dict[str, List[str]]) -> str:
    """
    Convert executive summary sections dict into HTML string.
    Used by Jinja2 template.
    """
    def build_section(title: str, bullets: List[str]) -> str:
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

    html = ""
    html += build_section("Major Developments", sections.get("major_developments", []))
    html += build_section("Financial/Operational Performance", sections.get("financial_operational", []))
    html += build_section("Risk Factors", sections.get("risk_factors", []))
    html += build_section("Wall Street Sentiment", sections.get("wall_street", []))
    html += build_section("Competitive/Industry Dynamics", sections.get("competitive_industry", []))
    html += build_section("Upcoming Catalysts", sections.get("upcoming_catalysts", []))

    return html


def build_articles_html(articles_by_category: Dict[str, List[Dict]]) -> str:
    """
    Convert articles by category into HTML string for email template.
    """
    def build_category_section(title: str, articles: List[Dict], category: str) -> str:
        if not articles:
            return ""

        article_links = ""
        for article in articles:
            # Check if article is paywalled
            is_paywalled = is_paywall_article(article.get('domain', ''))
            paywall_badge = ' <span style="font-size: 10px; color: #ef4444; font-weight: 600; margin-left: 4px;">PAYWALL</span>' if is_paywalled else ''

            # Check if article is new (< 24 hours)
            # NEW badge removed per user request (Oct 2025)
            # is_new = False
            # if article.get('published_at'):
            #     published_at = article['published_at']
            #     if published_at.tzinfo is None:
            #         published_at = published_at.replace(tzinfo=timezone.utc)
            #     age_hours = (datetime.now(timezone.utc) - published_at).total_seconds() / 3600
            #     is_new = age_hours < 24
            # new_badge = '🆕 ' if is_new else ''

            # Star for FLAGGED + QUALITY articles
            domain = article.get('domain', '')
            is_quality = domain.lower() in [
                'wsj.com', 'bloomberg.com', 'reuters.com', 'ft.com', 'barrons.com',
                'cnbc.com', 'forbes.com', 'marketwatch.com', 'seekingalpha.com'
            ]
            # Assume article is flagged if it's in the list (this function only gets flagged articles)
            star = '<span style="color: #f59e0b;">★</span> ' if is_quality else ''

            domain_name = get_or_create_formal_domain_name(domain) if domain else "Unknown Source"
            date_str = format_date_short(article['published_at']) if article.get('published_at') else "Recent"

            article_links += f'''
                <div style="padding: 6px 0; margin-bottom: 4px; border-bottom: 1px solid #e5e7eb;">
                    <a href="{article.get('resolved_url', '#')}" style="font-size: 13px; font-weight: 600; color: #1e40af; text-decoration: none; line-height: 1.4;">{star}{article.get('title', 'Untitled')}{paywall_badge}</a>
                    <div style="font-size: 11px; color: #6b7280; margin-top: 3px;">{domain_name} • {date_str}</div>
                </div>
            '''

        return f'''
            <div style="margin-bottom: 16px;">
                <h3 style="margin: 0 0 8px 0; font-size: 13px; font-weight: 700; color: #1e40af; text-transform: uppercase; letter-spacing: 0.75px;">{title} ({len(articles)})</h3>
                {article_links}
            </div>
        '''

    html = ""
    html += build_category_section("COMPANY", articles_by_category.get('company', []), "company")
    html += build_category_section("INDUSTRY", articles_by_category.get('industry', []), "industry")
    html += build_category_section("COMPETITORS", articles_by_category.get('competitor', []), "competitor")

    return html


def generate_email_html_core(
    ticker: str,
    hours: int = 24,
    flagged_article_ids: List[int] = None,
    recipient_email: str = None
) -> Dict[str, any]:
    """
    CORE Email #3 generation function - shared by both test and production workflows.

    Generates Premium Stock Intelligence Report HTML with executive summary and article links.
    Articles are pre-sorted by SQL (ORDER BY published_at DESC) within each category.

    Args:
        ticker: Stock ticker symbol
        hours: Lookback window in hours (default: 24)
        flagged_article_ids: List of article IDs flagged by AI triage
        recipient_email:
            - If provided: Generate real unsubscribe token (for test/immediate send)
            - If None: Use placeholder {{UNSUBSCRIBE_TOKEN}} (for production multi-recipient)

    Returns:
        {
            "html": Full HTML email string,
            "subject": Email subject line,
            "company_name": Company name,
            "article_count": Number of articles analyzed
        }
    """
    LOG.info(f"Generating Email #3 for {ticker} (recipient: {recipient_email or 'placeholder'})")

    # Fetch ticker config
    config = get_ticker_config(ticker)
    if not config:
        LOG.error(f"No config found for {ticker}")
        return None

    company_name = config.get("company_name", ticker)
    sector = config.get("sector")
    sector_display = f" • {sector}" if sector and sector.strip() else ""

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
        else:
            LOG.warning(f"No executive summary found for {ticker}")

    # Parse executive summary into sections
    sections = parse_executive_summary_sections(executive_summary_text)

    # Fetch flagged articles (already sorted by SQL)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    articles_by_category = {"company": [], "industry": [], "competitor": []}

    with db() as conn, conn.cursor() as cur:
        if flagged_article_ids is not None and len(flagged_article_ids) > 0:
            # Use flagged list (normal case)
            cur.execute("""
                SELECT a.id, a.title, a.resolved_url, a.domain, a.published_at,
                       ta.category, ta.search_keyword, ta.competitor_ticker,
                       ta.relevance_score, ta.relevance_reason, ta.is_rejected
                FROM articles a
                JOIN ticker_articles ta ON a.id = ta.article_id
                WHERE ta.ticker = %s
                AND a.id = ANY(%s)
                AND (a.published_at >= %s OR a.published_at IS NULL)
                AND (ta.is_rejected = FALSE OR ta.is_rejected IS NULL)
                ORDER BY a.published_at DESC NULLS LAST
            """, (ticker, flagged_article_ids, cutoff))
        else:
            # ERROR: No flagged_article_ids provided - should never happen in production
            LOG.error(f"[{ticker}] ❌ CRITICAL: No flagged_article_ids provided to Email #3!")
            LOG.error(f"[{ticker}] This should never happen - triage should always produce a list (even if empty)")
            # Return empty result instead of pulling all articles
            articles = []

        if flagged_article_ids is not None and len(flagged_article_ids) > 0:
            articles = cur.fetchall()
        # else: articles already set to [] above

        # Group articles by category (preserves SQL sort order: newest first)
        for article in articles:
            category = article['category']
            if category in articles_by_category:
                articles_by_category[category].append(article)

    # Calculate metrics
    analyzed_count = sum(len(arts) for arts in articles_by_category.values())
    paywall_count = sum(
        1 for articles in articles_by_category.values()
        for a in articles
        if is_paywall_article(a.get('domain', ''))
    )

    # Current date
    eastern = pytz.timezone('US/Eastern')
    current_date = datetime.now(timezone.utc).astimezone(eastern).strftime("%b %d, %Y")

    # Build HTML sections
    summary_html = build_executive_summary_html(sections)
    articles_html = build_articles_html(articles_by_category)

    # Analysis message
    lookback_days = hours // 24 if hours >= 24 else 1
    analysis_message = f"Analysis based on {analyzed_count} articles from the past {lookback_days} {'days' if lookback_days > 1 else 'day'}."
    if paywall_count > 0:
        analysis_message += f" {paywall_count} {'article' if paywall_count == 1 else 'articles'} behind paywalls (titles shown)."

    # Unsubscribe URL - real token for test, placeholder for production
    if recipient_email:
        # TEST MODE: Generate real token for immediate send
        unsubscribe_token = get_or_create_unsubscribe_token(recipient_email)
        if unsubscribe_token:
            unsubscribe_url = f"https://stockdigest.app/unsubscribe?token={unsubscribe_token}"
        else:
            unsubscribe_url = "https://stockdigest.app/unsubscribe"
            LOG.warning(f"No unsubscribe token for {recipient_email}, using generic link")
    else:
        # PRODUCTION MODE: Use placeholder for multi-recipient sending later
        unsubscribe_url = "{{UNSUBSCRIBE_TOKEN}}"

    # Build full HTML (same template for both test and production)
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} Intelligence Report</title>
    <style>
        @media only screen and (max-width: 600px) {{
            .content-padding {{ padding: 16px !important; }}
            .header-padding {{ padding: 16px 20px 25px 20px !important; }}
            .price-box {{ padding: 8px 10px !important; }}
            .company-name {{ font-size: 20px !important; }}
        }}
    </style>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f8f9fa; color: #212529;">

    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td align="center" style="padding: 40px 20px;">

                <table role="presentation" style="max-width: 700px; width: 100%; background-color: #ffffff; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-collapse: collapse; border-radius: 8px; overflow: visible;">

                    <!-- Header -->
                    <tr>
                        <td class="header-padding" style="padding: 18px 24px 30px 24px; background-color: #1e40af; background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); color: #ffffff; border-radius: 8px 8px 0 0;">
                            <table role="presentation" style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td style="width: 58%;">
                                        <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; opacity: 0.85; font-weight: 600; color: #ffffff;">STOCK INTELLIGENCE</div>
                                    </td>
                                    <td align="right" style="width: 42%;">
                                        <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; opacity: 0.85; font-weight: 600; color: #ffffff;">{current_date} • Last Close</div>
                                    </td>
                                </tr>
                                <tr>
                                    <td style="width: 58%; vertical-align: bottom; padding-bottom: 4px;">
                                        <h1 class="company-name" style="margin: 0; font-size: 28px; font-weight: 700; letter-spacing: -0.5px; line-height: 1; color: #ffffff;">{company_name}</h1>
                                        <div style="margin-top: 8px; font-size: 13px; opacity: 0.9; font-weight: 500; color: #ffffff;">{ticker}{sector_display}</div>
                                    </td>
                                    <td align="right" style="vertical-align: bottom; width: 42%; padding-bottom: 4px;">
                                        <div style="display: inline-block; text-align: right;">
                                            <div style="font-size: 28px; font-weight: 700; line-height: 1; margin-bottom: 2px; color: #ffffff;">{stock_price}</div>
                                            <div style="font-size: 14px; color: {price_change_color}; font-weight: 600; margin-top: 8px;">{price_change_pct}</div>
                                        </div>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Content -->
                    <tr>
                        <td class="content-padding" style="padding: 24px 24px 24px 24px;">

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
                                        <div style="font-size: 14px; font-weight: 600; color: #ffffff; margin-bottom: 4px;">StockDigest</div>
                                        <div style="font-size: 12px; opacity: 0.8; margin-bottom: 8px; color: #ffffff;">Stock Intelligence Delivered Daily</div>

                                        <!-- Legal Disclaimer -->
                                        <div style="font-size: 10px; opacity: 0.7; line-height: 1.4; margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.2); color: #ffffff;">
                                            This report is for informational purposes only and does not constitute investment advice, a recommendation, or an offer to buy or sell securities. Please consult a financial advisor before making investment decisions.
                                        </div>

                                        <!-- Links -->
                                        <div style="font-size: 11px; margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.2);">
                                            <a href="https://stockdigest.app/terms-of-service" style="color: #ffffff; text-decoration: none; opacity: 0.9; margin-right: 12px;">Terms of Service</a>
                                            <span style="color: rgba(255,255,255,0.5); margin-right: 12px;">|</span>
                                            <a href="https://stockdigest.app/privacy-policy" style="color: #ffffff; text-decoration: none; opacity: 0.9; margin-right: 12px;">Privacy Policy</a>
                                            <span style="color: rgba(255,255,255,0.5); margin-right: 12px;">|</span>
                                            <a href="mailto:stockdigest.research@gmail.com" style="color: #ffffff; text-decoration: none; opacity: 0.9; margin-right: 12px;">Contact</a>
                                            <span style="color: rgba(255,255,255,0.5); margin-right: 12px;">|</span>
                                            <a href="{unsubscribe_url}" style="color: #ffffff; text-decoration: none; opacity: 0.9;">Unsubscribe</a>
                                        </div>

                                        <!-- Copyright -->
                                        <div style="font-size: 10px; opacity: 0.6; margin-top: 12px; color: #ffffff;">
                                            © 2025 StockDigest. All rights reserved.
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

    return {
        "html": html,
        "subject": subject,
        "company_name": company_name,
        "article_count": analyzed_count
    }

def save_email_to_queue(ticker: str, recipients: List[str], hours: int = 24, flagged_article_ids: List[int] = None):
    """Save Email #3 to email_queue table for daily workflow"""
    LOG.info(f"[{ticker}] 💾 Saving Email #3 to queue for {len(recipients)} recipients")

    # Generate HTML using unified core function
    email_data = generate_email_html_core(
        ticker=ticker,
        hours=hours,
        flagged_article_ids=flagged_article_ids,
        recipient_email=None  # Use placeholder {{UNSUBSCRIBE_TOKEN}} for production
    )
    if not email_data:
        LOG.error(f"[{ticker}] ❌ Failed to generate email HTML")
        return False

    html = email_data['html']
    subject = email_data['subject']
    company_name = email_data['company_name']
    article_count = email_data['article_count']

    # Save to email_queue table
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO email_queue (
                ticker, company_name, recipients, email_html, email_subject,
                article_count, status, is_production, heartbeat, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (ticker) DO UPDATE
            SET company_name = EXCLUDED.company_name,
                recipients = EXCLUDED.recipients,
                email_html = EXCLUDED.email_html,
                email_subject = EXCLUDED.email_subject,
                article_count = EXCLUDED.article_count,
                status = 'ready',
                error_message = NULL,
                is_production = EXCLUDED.is_production,
                heartbeat = NOW(),
                updated_at = NOW()
        """, (ticker, company_name, recipients, html, subject, article_count, 'ready', True))
        conn.commit()

    LOG.info(f"[{ticker}] ✅ Email #3 saved to queue (status=ready)")

    # Send preview to admin
    try:
        preview_subject = f"[PREVIEW] {subject}"
        send_email(preview_subject, html, to=DIGEST_TO)
        LOG.info(f"[{ticker}] ✅ Preview sent to admin")
    except Exception as e:
        LOG.error(f"[{ticker}] ❌ Failed to send preview to admin: {e}")

    return True


def send_user_intelligence_report(hours: int = 24, tickers: List[str] = None,
                                   flagged_article_ids: List[int] = None,
                                   recipient_email: str = None) -> Dict:
    """
    TEST wrapper: Email #3 - Premium Stock Intelligence Report (Single Ticker).
    Generates email with real unsubscribe token and sends immediately.

    Used by test runs for immediate email delivery.
    Production uses generate_user_intelligence_report_html() instead.

    Returns: {"status": "sent" | "failed", "articles_analyzed": X, ...}
    """
    LOG.info("=== EMAIL #3: PREMIUM STOCK INTELLIGENCE (TEST MODE) ===")

    # Single ticker only
    if not tickers or len(tickers) == 0:
        return {"status": "error", "message": "No ticker specified"}

    ticker = tickers[0]  # Take first ticker only
    LOG.info(f"[TEST] Generating premium report for {ticker} → {recipient_email or DIGEST_TO}")

    # Call core function with real token (recipient_email provided)
    email_data = generate_email_html_core(
        ticker=ticker,
        hours=hours,
        flagged_article_ids=flagged_article_ids,
        recipient_email=recipient_email or DIGEST_TO  # Real token for test
    )

    if not email_data:
        return {"status": "error", "message": "Failed to generate email HTML"}

    # Send email immediately (test mode)
    success = send_email(email_data['subject'], email_data['html'], to=recipient_email or DIGEST_TO)

    LOG.info(f"📧 Email #3 (Premium Intelligence): {'✅ SENT' if success else '❌ FAILED'} to {recipient_email or DIGEST_TO}")

    return {
        "status": "sent" if success else "failed",
        "articles_analyzed": email_data['article_count'],
        "ticker": ticker,
        "recipient": recipient_email or DIGEST_TO,
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
_worker_heartbeat_monitor_thread = None
_worker_restart_count = 0
_last_worker_activity = None

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

async def resolve_google_news_url_with_scrapfly(url: str, ticker: str) -> tuple[Optional[str], Optional[str]]:
    """
    Use ScrapFly to resolve Google News redirects by fetching the final URL

    Uses ASP (Anti-Scraping Protection) + JS rendering to handle Google's anti-bot measures.
    ScrapFly follows redirects and returns the final URL after executing any JavaScript.

    Cost: ~$0.005-0.010 per request (higher due to ASP + JS rendering)
    Success rate: ~95% (ScrapFly designed specifically for Google scraping)

    Reference: https://scrapfly.io/blog/how-to-scrape-google-search/

    Returns:
        (resolved_url, error_message): URL if successful, error message if failed
    """
    try:
        if not SCRAPFLY_API_KEY:
            return None, "No ScrapFly API key configured"

        # Build params with anti-bot bypass for Google News
        # ASP (Anti-Scraping Protection) + JS rendering specifically recommended for Google
        params = {
            "key": SCRAPFLY_API_KEY,
            "url": url,
            "country": "us",
            "asp": "true",        # Anti-bot bypass (handles Google's anti-scraping)
            "render_js": "true",  # JavaScript execution (required for redirects)
        }

        session = get_http_session()
        async with session.get("https://api.scrapfly.io/scrape", params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    result = await response.json()
                    final_url = result.get("result", {}).get("url")

                    if final_url and "news.google.com" not in final_url:
                        return final_url, None

                    return None, "Still Google News URL after resolution"

                # Non-200 status
                error_text = await response.text()
                return None, f"HTTP {response.status}: {error_text[:100]}"

    except asyncio.TimeoutError:
        return None, "Timeout (15s exceeded)"
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:100]}"

async def resolve_flagged_google_news_urls(ticker: str, flagged_article_ids: List[int]) -> List[int]:
    """
    Resolve Google News URLs for flagged articles only (after triage, before digest)

    Resolution methods (NO title extraction - domain already extracted during ingestion):
    1. Advanced API resolution (Google internal API) - Free, fast when works
    2. Direct HTTP redirect (follow redirects) - Free, works for simple redirects
    3. ScrapFly resolution (paid) - ~$0.005-0.010/URL, 95% success rate (ASP + JS rendering)
    4. If all fail: Keep Google News URL with existing domain from DB

    Then check if resolved to Yahoo Finance:
    - If Yahoo: Extract original source (Google → Yahoo → Original chain)
    - If not Yahoo: Use resolved URL directly

    Benefits:
    - Only resolves 20-30 URLs (flagged articles) vs 150 URLs (all articles)
    - Spread out over time (no burst of concurrent requests)
    - Natural request pattern (like a human clicking links)
    - Domain from title already available for Email #1 and deduplication

    Returns:
    - Updated flagged_article_ids list with duplicates removed
    """
    LOG.info(f"[{ticker}] 🔗 Phase 1.5: Resolving Google News URLs for flagged articles...")

    # Get unresolved Google News articles
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, url, title, domain
            FROM articles
            WHERE id = ANY(%s)
            AND url LIKE '%%news.google.com%%'
            AND resolved_url IS NULL
        """, (flagged_article_ids,))

        unresolved = cur.fetchall()

    if not unresolved:
        LOG.info(f"[{ticker}] ✅ No Google News URLs need resolution (all already resolved or no Google News articles)")
        return flagged_article_ids

    total = len(unresolved)
    LOG.info(f"[{ticker}] 📋 Found {total} unresolved Google News URLs")
    LOG.info(f"[{ticker}] 🔄 Starting resolution process (Advanced API + Direct HTTP only)...")

    resolved_count = 0
    failed_count = 0
    yahoo_chain_count = 0  # Track Google → Yahoo → Final chains

    for idx, article in enumerate(unresolved, 1):
        article_id = article['id']
        url = article['url']
        title = article['title']
        existing_domain = article['domain']

        resolved_url = None
        resolution_method = None
        scrapfly_error = None

        try:
            # METHOD 1: Try Advanced API resolution first (silent)
            resolved_url = domain_resolver._resolve_google_news_url_advanced(url)
            if resolved_url:
                resolution_method = "Tier 1 (Advanced API)"

            # METHOD 2: Try direct HTTP redirect (silent)
            if not resolved_url:
                try:
                    response = requests.get(url, timeout=10, allow_redirects=True, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    final_url = response.url
                    if final_url != url and "news.google.com" not in final_url:
                        resolved_url = final_url
                        resolution_method = "Tier 2 (Direct HTTP)"
                except:
                    pass

            # METHOD 3: Try ScrapFly resolution (capture error if fails)
            if not resolved_url:
                resolved_url, scrapfly_error = await resolve_google_news_url_with_scrapfly(url, ticker)
                if resolved_url:
                    resolution_method = "Tier 3 (ScrapFly)"

            # If all methods failed, log and skip
            if not resolved_url:
                failed_count += 1
                # Log Tier 3 failure with error message
                if scrapfly_error:
                    LOG.error(f"[{ticker}] ❌ [{idx}/{total}] Tier 3 (ScrapFly) failed → {scrapfly_error}")
                continue

            # Check if resolved to Yahoo Finance → Extract original source
            is_yahoo_finance = any(yahoo_domain in resolved_url for yahoo_domain in [
                "finance.yahoo.com", "ca.finance.yahoo.com", "uk.finance.yahoo.com"
            ])

            if is_yahoo_finance:
                yahoo_original = extract_yahoo_finance_source_optimized(resolved_url)
                if yahoo_original:
                    final_resolved_url = yahoo_original
                    final_domain = normalize_domain(urlparse(yahoo_original).netloc.lower())
                    yahoo_chain_count += 1
                else:
                    final_resolved_url = resolved_url
                    final_domain = normalize_domain(urlparse(resolved_url).netloc.lower())
            else:
                final_resolved_url = resolved_url
                final_domain = normalize_domain(urlparse(resolved_url).netloc.lower())

            # Update database (no source_url)
            with db() as conn, conn.cursor() as cur:
                cur.execute("""
                    UPDATE articles
                    SET resolved_url = %s, domain = %s
                    WHERE id = %s
                """, (final_resolved_url, final_domain, article_id))

            # Store domain mapping for future use (if domain changed from fallback)
            # This prevents future ScrapFly calls for the same publication
            if existing_domain != final_domain and existing_domain.endswith('.com'):
                # Extract publication name from title
                clean_title, source_name = extract_source_from_title_smart(title)

                if source_name:
                    try:
                        domain_resolver._store_in_database(final_domain, source_name, ai_generated=False)
                        LOG.info(f"[{ticker}] 💾 Stored mapping: '{source_name}' → '{final_domain}' (learned from resolution)")
                    except Exception as e:
                        LOG.warning(f"[{ticker}] Failed to store domain mapping: {e}")

            resolved_count += 1
            # Show full resolved URL (not just domain)
            LOG.info(f"[{ticker}] ✅ [{idx}/{total}] {resolution_method} → {final_resolved_url}")

        except Exception as e:
            failed_count += 1
            LOG.error(f"[{ticker}] ❌ [{idx}/{total}] Resolution failed: {str(e)[:100]}")

    # Summary
    LOG.info(f"[{ticker}] {'='*60}")
    LOG.info(f"[{ticker}] 📊 Resolution Summary:")
    LOG.info(f"[{ticker}]    Total processed: {total}")
    LOG.info(f"[{ticker}]    ✅ Succeeded: {resolved_count} ({resolved_count/total*100:.1f}%)")
    LOG.info(f"[{ticker}]    ❌ Failed: {failed_count} ({failed_count/total*100:.1f}%)")
    LOG.info(f"[{ticker}]    🔗 Google→Yahoo→Final chains: {yahoo_chain_count}")
    LOG.info(f"[{ticker}] {'='*60}")

    # ============================================================================
    # POST-RESOLUTION DEDUPLICATION
    # ============================================================================
    # After resolution, check if multiple articles resolved to the SAME final URL
    # This catches: Google News → Source A, Yahoo Finance → Source A (same resolved_url)
    # Keep the best article, remove duplicates from flagged list
    # NOTE: System uses ID lists (not database column) to track flagged articles
    # ============================================================================
    LOG.info(f"[{ticker}] 🔍 Checking for duplicate resolved URLs...")

    # Return early if no flagged articles
    if not flagged_article_ids:
        LOG.info(f"[{ticker}] ✅ No flagged articles to deduplicate")
        return flagged_article_ids or []  # Return empty list if None

    with db() as conn, conn.cursor() as cur:
        # Find flagged articles with duplicate resolved_url (use ID list, not database column)
        cur.execute("""
            SELECT resolved_url, array_agg(a.id ORDER BY a.published_at DESC) as article_ids,
                   array_agg(a.domain ORDER BY a.published_at DESC) as domains,
                   array_agg(a.title ORDER BY a.published_at DESC) as titles
            FROM articles a
            WHERE a.id = ANY(%s)
            AND a.resolved_url IS NOT NULL
            GROUP BY resolved_url
            HAVING COUNT(*) > 1
        """, (flagged_article_ids,))

        duplicates = cur.fetchall()

        if duplicates:
            removed_count = 0
            removed_ids = []

            for dup in duplicates:
                resolved_url = dup['resolved_url']
                article_ids = dup['article_ids']
                domains = dup['domains']
                titles = dup['titles']

                # Keep first article (newest by published_at), remove the rest from ID list
                keep_id = article_ids[0]
                remove_ids = article_ids[1:]

                LOG.info(f"[{ticker}] 🔄 Duplicate resolved URL: {resolved_url[:80]}...")
                LOG.info(f"[{ticker}]    ✅ Keeping: ID {keep_id} ({domains[0]}) - {titles[0][:60]}...")

                for idx, remove_id in enumerate(remove_ids, 1):
                    LOG.info(f"[{ticker}]    ❌ Removing: ID {remove_id} ({domains[idx]}) - {titles[idx][:60]}...")
                    removed_ids.append(remove_id)
                    removed_count += 1

            # Remove duplicates from Python list (not database - no flagged column exists)
            flagged_article_ids = [aid for aid in flagged_article_ids if aid not in removed_ids]

            LOG.info(f"[{ticker}] ✅ Removed {removed_count} duplicate articles from flagged list")
        else:
            LOG.info(f"[{ticker}] ✅ No duplicate resolved URLs found")

    return flagged_article_ids

async def process_digest_phase(job_id: str, ticker: str, minutes: int, flagged_article_ids: List[int] = None):
    """Wrapper for digest logic with error handling - sends Stock Intelligence Email with executive summary

    NEW (Oct 2025): Scraping happens HERE in digest phase, AFTER Phase 1.5 URL resolution
    """
    try:
        # ============================================================================
        # PHASE 4: CONTENT SCRAPING (MOVED FROM INGEST PHASE - Oct 2025)
        # ============================================================================
        # Now runs AFTER Phase 1.5 Google URL resolution, ensuring all URLs are resolved
        # before scraping attempts
        # ============================================================================

        if flagged_article_ids:
            LOG.info(f"[{ticker}] 📄 [JOB {job_id}] Phase 4: Scraping {len(flagged_article_ids)} flagged articles...")

            # Get flagged articles that need scraping
            # NOTE: Articles with resolved_url = NULL are included (happens when resolution failed)
            # The scraper will fall back to the original URL, which may fail but won't crash
            with db() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT a.id, a.url, a.url_hash, a.resolved_url, a.title, a.description,
                           a.domain, a.published_at,
                           ta.category, ta.search_keyword, ta.competitor_ticker
                    FROM articles a
                    JOIN ticker_articles ta ON a.id = ta.article_id
                    WHERE a.id = ANY(%s)
                    AND ta.ticker = %s
                    AND a.scraped_content IS NULL
                    AND a.scraping_failed = FALSE
                    ORDER BY a.published_at DESC NULLS LAST
                """, (flagged_article_ids, ticker))

                articles_to_scrape = cur.fetchall()

            if articles_to_scrape:
                LOG.info(f"[{ticker}] 🔍 Found {len(articles_to_scrape)} articles needing scraping")

                # Get ticker metadata for AI analysis
                config = get_ticker_config(ticker)
                metadata = {
                    "industry_keywords": config.get("industry_keywords", []) if config else [],
                    "competitors": config.get("competitors", []) if config else [],
                    "company_name": config.get("company_name", ticker) if config else ticker
                }

                # Process articles in batches using existing batch scraping logic
                BATCH_SIZE = 5
                total_scraped = 0
                total_failed = 0

                for i in range(0, len(articles_to_scrape), BATCH_SIZE):
                    batch = articles_to_scrape[i:i + BATCH_SIZE]
                    batch_num = (i // BATCH_SIZE) + 1
                    total_batches = (len(articles_to_scrape) + BATCH_SIZE - 1) // BATCH_SIZE

                    LOG.info(f"[{ticker}] 📦 Processing scraping batch {batch_num}/{total_batches} ({len(batch)} articles)")

                    # Convert to format expected by process_article_batch_async
                    batch_articles = [dict(row) for row in batch]
                    batch_categories = [row['category'] for row in batch]

                    try:
                        # Use existing batch scraping function
                        batch_results = await process_article_batch_async(
                            batch_articles,
                            batch_categories,
                            metadata,
                            ticker
                        )

                        # Count successes and failures
                        for result in batch_results:
                            if result["success"] and result.get("scraped_content"):
                                total_scraped += 1
                            else:
                                total_failed += 1

                        LOG.info(f"[{ticker}] ✅ Batch {batch_num}/{total_batches} complete: {total_scraped} scraped, {total_failed} failed")

                    except Exception as e:
                        LOG.error(f"[{ticker}] ❌ Batch {batch_num} scraping error: {e}")
                        total_failed += len(batch)

                LOG.info(f"[{ticker}] 📊 Scraping complete: {total_scraped} successful, {total_failed} failed")
            else:
                LOG.info(f"[{ticker}] ✅ All flagged articles already scraped (or scraping previously failed)")
        else:
            LOG.info(f"[{ticker}] ⚠️ No flagged articles to scrape")

        # ============================================================================
        # PHASE 5: EMAIL #2 GENERATION (CONTENT QA)
        # ============================================================================
        # Now that articles are scraped, generate Email #2 with AI summaries
        # ============================================================================

        # CRITICAL: fetch_digest_articles_with_enhanced_content sends the Stock Intelligence Email
        # which includes the executive summary via generate_ai_final_summaries()
        fetch_digest_func = globals().get('fetch_digest_articles_with_enhanced_content')
        if not fetch_digest_func:
            raise RuntimeError("fetch_digest_articles_with_enhanced_content not yet defined")

        LOG.info(f"[JOB {job_id}] Calling fetch_digest (will send Stock Intelligence Email) for {ticker}...")
        if flagged_article_ids:
            LOG.info(f"[JOB {job_id}] Filtering to {len(flagged_article_ids)} flagged articles from triage")

        result = await fetch_digest_func(
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

        # ALWAYS skip render for job queue commits (deployment only via 6:30 AM cron or manual button)
        skip_render = True
        LOG.info(f"[JOB {job_id}] [skip render] flag enabled - no deployment (job queue mode)")

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
                LOG.info(f"[{job['ticker']}] 📋 Claimed job {job['job_id']} for ticker {job['ticker']}")
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

    LOG.info(f"[{ticker}] 🚀 [JOB {job_id}] Starting processing for {ticker}")
    LOG.info(f"   Config: minutes={minutes}, batch={batch_size}, triage_batch={triage_batch_size}")

    try:
        # NOTE: TICKER_PROCESSING_LOCK removed from cron_ingest/cron_digest for parallel processing
        # We don't acquire it here to avoid deadlock (lock is not reentrant)

        # Check if job was cancelled before starting
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT status FROM ticker_processing_jobs WHERE job_id = %s", (job_id,))
            current_status = cur.fetchone()
            if current_status and current_status['status'] == 'cancelled':
                LOG.warning(f"[{ticker}] 🚫 [JOB {job_id}] Job cancelled before starting, exiting")
                return

        # PHASE 0: SAFETY CHECK - Ensure feeds exist (failsafe)
        # This is a defensive check in case /jobs/submit initialization somehow failed
        LOG.info(f"[{ticker}] 🔍 [JOB {job_id}] Phase 0: Checking feeds exist...")

        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as count FROM ticker_feeds
                WHERE ticker = %s AND active = TRUE
            """, (ticker,))
            feed_count = cur.fetchone()['count']

            if feed_count == 0:
                LOG.warning(f"[{ticker}] ⚠️ [JOB {job_id}] No feeds found! Creating now (failsafe)...")

                # Get or create metadata
                metadata = get_or_create_enhanced_ticker_metadata(ticker, force_refresh=True)
                LOG.info(f"[{ticker}] 📋 [JOB {job_id}] Metadata: company={metadata.get('company_name', 'N/A')}")

                # Create feeds
                feeds_created = create_feeds_for_ticker_new_architecture(ticker, metadata)

                if feeds_created:
                    LOG.info(f"[{ticker}] ✅ [JOB {job_id}] Created {len(feeds_created)} feeds (failsafe recovery)")
                else:
                    LOG.error(f"[{ticker}] ❌ [JOB {job_id}] Failed to create feeds - job will likely fail")
            else:
                LOG.info(f"[{ticker}] ✅ [JOB {job_id}] {feed_count} feeds verified")

        # PHASE 1: Ingest (already implemented in /cron/ingest)
        update_job_status(job_id, phase='ingest_start', progress=10)
        LOG.info(f"[{ticker}] 📥 [JOB {job_id}] Phase 1: Ingest starting...")

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
            LOG.info(f"[{ticker}] ✅ [JOB {job_id}] Phase 1: Ingest complete")
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
            LOG.info(f"[{ticker}] ✅ [JOB {job_id}] Phase 1: Ingest complete (no detailed stats)")

        # Check if cancelled after Phase 1
        # Also re-fetch config to get flagged_articles that were stored during ingest
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT status, config FROM ticker_processing_jobs WHERE job_id = %s", (job_id,))
            job_status = cur.fetchone()
            if job_status and job_status['status'] == 'cancelled':
                LOG.warning(f"[{ticker}] 🚫 [JOB {job_id}] Job cancelled after Phase 1, exiting")
                return

            # Re-fetch flagged_articles that were stored during ingest phase
            updated_config = job_status['config'] if job_status and isinstance(job_status['config'], dict) else {}
            flagged_article_ids = updated_config.get('flagged_articles', [])

            if flagged_article_ids:
                LOG.info(f"[{ticker}] 📋 [JOB {job_id}] Retrieved {len(flagged_article_ids)} flagged article IDs from ingest phase")
            else:
                LOG.warning(f"[{ticker}] ⚠️ [JOB {job_id}] No flagged articles found in config after ingest")

        # PHASE 1.5: Resolve Google News URLs for flagged articles (NEW!)
        update_job_status(job_id, phase='resolution_start', progress=62)
        LOG.info(f"[{ticker}] 🔗 [JOB {job_id}] Phase 1.5: Google News URL resolution starting...")

        if flagged_article_ids:
            # Resolve URLs and get deduplicated list back
            flagged_article_ids = await resolve_flagged_google_news_urls(ticker, flagged_article_ids)
            LOG.info(f"[{ticker}] 📋 [JOB {job_id}] After resolution & deduplication: {len(flagged_article_ids or [])} flagged articles remain")
        else:
            LOG.warning(f"[{ticker}] ⚠️ [JOB {job_id}] No flagged articles to resolve")

        update_job_status(job_id, phase='resolution_complete', progress=64)

        # PHASE 2: Digest (already implemented in /cron/digest)
        update_job_status(job_id, phase='digest_start', progress=65)
        LOG.info(f"[{ticker}] 📧 [JOB {job_id}] Phase 2: Digest starting...")

        # Call digest function (defined later in file) - pass deduplicated flagged articles
        digest_result = await process_digest_phase(
            job_id=job_id,
            ticker=ticker,
            minutes=minutes,
            flagged_article_ids=flagged_article_ids
        )

        update_job_status(job_id, phase='digest_complete', progress=95)

        # Log detailed digest stats
        if digest_result:
            LOG.info(f"[{ticker}] ✅ [JOB {job_id}] Phase 2: Digest complete")
            if isinstance(digest_result, dict):
                LOG.info(f"   Status: {digest_result.get('status', 'unknown')}")
                LOG.info(f"   Articles: {digest_result.get('articles', 0)}")
        else:
            LOG.info(f"[{ticker}] ✅ [JOB {job_id}] Phase 2: Digest complete (no detailed stats)")

        # Check if cancelled after Phase 2
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT status FROM ticker_processing_jobs WHERE job_id = %s", (job_id,))
            current_status = cur.fetchone()
            if current_status and current_status['status'] == 'cancelled':
                LOG.warning(f"[{ticker}] 🚫 [JOB {job_id}] Job cancelled after Phase 2, exiting")
                return

        # Log resource usage after digest
        memory_after_digest = memory_monitor.get_current_mb() if hasattr(memory_monitor, 'get_current_mb') else 0
        if DB_POOL:
            try:
                pool_stats = DB_POOL.get_stats()
                LOG.info(f"[{ticker}] 📊 Resource Status: Memory={memory_after_digest:.1f}MB, DB Pool={pool_stats.get('pool_size', 0)}/{pool_stats.get('pool_max', 0)} connections")
            except:
                LOG.info(f"[{ticker}] 📊 Resource Status: Memory={memory_after_digest:.1f}MB")
        else:
            LOG.info(f"[{ticker}] 📊 Resource Status: Memory={memory_after_digest:.1f}MB")

        # EMAIL #3: USER INTELLIGENCE REPORT (no AI analysis, no descriptions)
        update_job_status(job_id, phase='user_report', progress=97)

        # Check mode to determine Email #3 behavior
        mode = config.get('mode', 'test')
        recipients = config.get('recipients', [])

        try:
            if mode == 'daily':
                # DAILY WORKFLOW: Save Email #3 to queue for admin review
                LOG.info(f"[{ticker}] 💾 [JOB {job_id}] Saving Email #3 to queue ({len(recipients)} recipients)...")
                success = save_email_to_queue(
                    ticker=ticker,
                    recipients=recipients,
                    hours=int(minutes/60),
                    flagged_article_ids=flagged_article_ids
                )
                if success:
                    LOG.info(f"[{ticker}] ✅ [JOB {job_id}] Email #3 saved to queue (status=ready)")
                else:
                    LOG.error(f"[{ticker}] ❌ [JOB {job_id}] Failed to save Email #3 to queue")
            else:
                # TEST WORKFLOW: Send Email #3 immediately to admin
                LOG.info(f"[{ticker}] 📧 [JOB {job_id}] Sending Email #3 immediately (test mode)...")
                user_report_result = send_user_intelligence_report(
                    hours=int(minutes/60),
                    tickers=[ticker],
                    flagged_article_ids=flagged_article_ids,
                    recipient_email=DIGEST_TO
                )
                if user_report_result:
                    LOG.info(f"[{ticker}] ✅ [JOB {job_id}] Email #3 sent successfully")
                    if isinstance(user_report_result, dict):
                        LOG.info(f"   Status: {user_report_result.get('status', 'unknown')}")
                else:
                    LOG.warning(f"[{ticker}] ⚠️ [JOB {job_id}] Email #3 returned no result")
        except Exception as e:
            LOG.error(f"[{ticker}] ❌ [JOB {job_id}] Email #3 failed: {e}")
            # Continue to GitHub commit even if Email #3 fails (Option A)

        # Check if cancelled after Phase 3
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT status FROM ticker_processing_jobs WHERE job_id = %s", (job_id,))
            current_status = cur.fetchone()
            if current_status and current_status['status'] == 'cancelled':
                LOG.warning(f"[{ticker}] 🚫 [JOB {job_id}] Job cancelled after Email #3, exiting")
                return

        # COMMIT METADATA TO GITHUB after all emails sent
        # This ensures GitHub commit doesn't trigger server restart before emails are sent
        update_job_status(job_id, phase='commit_metadata', progress=99)
        LOG.info(f"[{ticker}] 💾 [JOB {job_id}] Committing AI metadata to GitHub after final email...")

        try:
            # Commit to GitHub (always with [skip render] for job queue)
            batch_id = job.get('batch_id')

            await process_commit_phase(
                job_id=job_id,
                ticker=ticker,
                batch_id=batch_id,
                is_last_job=False  # Not used anymore - always skip render
            )
            LOG.info(f"[{ticker}] ✅ [JOB {job_id}] Metadata committed to GitHub successfully")
        except Exception as e:
            LOG.error(f"[{ticker}] ⚠️ [JOB {job_id}] GitHub commit failed (non-fatal): {e}")
            # Don't fail the job if commit fails - continue processing

        # PHASE 3: Complete
        update_job_status(job_id, phase='finalizing', progress=99)
        LOG.info(f"[{ticker}] ✅ [JOB {job_id}] Finalizing job...")

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

        # Log final resource stats
        if DB_POOL:
            try:
                pool_stats = DB_POOL.get_stats()
                LOG.info(f"[{ticker}] ✅ [JOB {job_id}] COMPLETED in {duration:.1f}s (memory: {memory_used:.1f}MB, DB pool: {pool_stats.get('pool_size', 0)}/{pool_stats.get('pool_max', 0)})")
            except:
                LOG.info(f"[{ticker}] ✅ [JOB {job_id}] COMPLETED in {duration:.1f}s (memory: {memory_used:.1f}MB)")
        else:
            LOG.info(f"[{ticker}] ✅ [JOB {job_id}] COMPLETED in {duration:.1f}s (memory: {memory_used:.1f}MB)")

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

        LOG.error(f"[{ticker}] ❌ [JOB {job_id}] FAILED after {duration:.1f}s: {error_msg}")
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

    finally:
        # Cleanup thread-local HTTP session and connector
        try:
            await cleanup_http_session()
            LOG.debug(f"[{ticker}] 🧹 Cleaned up HTTP session + connector for thread")
        except Exception as cleanup_error:
            LOG.warning(f"[{ticker}] Failed to cleanup HTTP session/connector: {cleanup_error}")

def job_worker_loop():
    """Background worker that polls database for jobs and processes them concurrently"""
    global _job_worker_running
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    LOG.info(f"🔧 Job worker started (worker_id: {get_worker_id()}, max_concurrent_jobs: {MAX_CONCURRENT_JOBS})")

    # CRITICAL: Load CSV from GitHub BEFORE processing any jobs
    LOG.info("🔄 Syncing ticker reference from GitHub (job worker initialization)...")
    try:
        github_sync_result = sync_ticker_references_from_github()
        if github_sync_result["status"] == "success":
            LOG.info(f"✅ GitHub sync successful: {github_sync_result.get('message', 'Completed')}")
        else:
            LOG.error(f"❌ GitHub sync failed: {github_sync_result.get('message', 'Unknown error')}")
            LOG.error("⚠️ Worker will continue, but tickers may have incorrect data!")
    except Exception as e:
        LOG.error(f"❌ GitHub sync crashed: {e}")
        LOG.error("⚠️ Worker will continue, but tickers may have incorrect data!")

    # Create thread pool for concurrent job processing
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="TickerWorker") as executor:
        active_futures = {}  # Map of future -> job_id

        while _job_worker_running:
            try:
                # Check circuit breaker
                if job_circuit_breaker.is_open():
                    LOG.warning("⚠️ Circuit breaker is OPEN, skipping job polling")
                    time.sleep(30)
                    continue

                # Poll for new jobs if we have capacity
                while len(active_futures) < MAX_CONCURRENT_JOBS:
                    job = get_next_queued_job()

                    if job:
                        # Submit job to thread pool
                        future = executor.submit(asyncio.run, process_ticker_job(job))
                        active_futures[future] = job['job_id']
                        LOG.info(f"📤 [JOB {job['job_id']}] Submitted to worker pool ({len(active_futures)}/{MAX_CONCURRENT_JOBS} active)")
                    else:
                        # No more jobs available
                        break

                # Wait for at least one job to complete (or timeout)
                if active_futures:
                    done_futures, _ = wait(active_futures.keys(), timeout=5, return_when=FIRST_COMPLETED)

                    # Process completed jobs
                    for done in done_futures:
                        job_id = active_futures.pop(done)
                        try:
                            done.result()  # Raises exception if job failed
                            LOG.info(f"✅ [JOB {job_id}] Completed successfully ({len(active_futures)}/{MAX_CONCURRENT_JOBS} active)")
                        except Exception as e:
                            LOG.error(f"❌ [JOB {job_id}] Failed with exception: {e}")
                else:
                    # No active jobs, sleep and poll again
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

def restart_worker_thread():
    """Restart the worker thread (called by heartbeat monitor when worker is frozen)"""
    global _job_worker_running, _job_worker_thread, _worker_restart_count

    LOG.error("🔄 Worker thread appears frozen - attempting restart...")

    # Stop the frozen worker
    try:
        _job_worker_running = False
        if _job_worker_thread and _job_worker_thread.is_alive():
            # Give it 5 seconds to gracefully stop
            _job_worker_thread.join(timeout=5)
            if _job_worker_thread.is_alive():
                LOG.error("⚠️ Worker thread did not stop gracefully")
    except Exception as e:
        LOG.error(f"Error stopping worker: {e}")

    # Increment restart counter
    _worker_restart_count += 1

    # If too many restarts, exit the process (let Render restart everything)
    if _worker_restart_count >= 3:
        LOG.critical(f"💀 Worker has been restarted {_worker_restart_count} times - exiting process!")
        LOG.critical("Render will automatically restart the service with a clean slate")
        os._exit(1)  # Force exit (bypasses cleanup, triggers Render restart)

    # Restart the worker thread
    try:
        _job_worker_running = True
        _job_worker_thread = threading.Thread(target=job_worker_loop, daemon=True, name=f"JobWorker-Restart-{_worker_restart_count}")
        _job_worker_thread.start()
        LOG.warning(f"✅ Worker thread restarted (attempt {_worker_restart_count}/3)")
    except Exception as e:
        LOG.error(f"Failed to restart worker thread: {e}")
        os._exit(1)  # Can't restart worker - exit process

def worker_heartbeat_monitor_loop():
    """Monitor worker health and restart if frozen"""
    global _last_worker_activity

    LOG.info("💓 Worker heartbeat monitor started (checks every 60 seconds, 5-minute threshold)")

    while True:
        try:
            time.sleep(60)  # Check every minute

            # Check if worker thread is alive
            if not _job_worker_running or not _job_worker_thread or not _job_worker_thread.is_alive():
                LOG.warning("⚠️ Worker thread is not running - attempting restart...")
                restart_worker_thread()
                continue

            # Check for recent worker activity
            with db() as conn, conn.cursor() as cur:
                # Look for any job that has been updated in the last 5 minutes
                cur.execute("""
                    SELECT MAX(last_updated) as latest_activity
                    FROM ticker_processing_jobs
                    WHERE status IN ('processing', 'queued')
                """)
                row = cur.fetchone()
                latest_activity = row['latest_activity'] if row else None

                if latest_activity:
                    _last_worker_activity = latest_activity
                    # Make timezone-aware if it's naive (database returns naive timestamps)
                    if latest_activity.tzinfo is None:
                        latest_activity = latest_activity.replace(tzinfo=timezone.utc)
                    time_since_activity = (datetime.now(timezone.utc) - latest_activity).total_seconds() / 60

                    # If no activity for 5 minutes AND there are jobs to process, worker is frozen
                    if time_since_activity > 5:
                        # Double-check there are actually jobs waiting
                        cur.execute("""
                            SELECT COUNT(*) as count
                            FROM ticker_processing_jobs
                            WHERE status = 'queued'
                        """)
                        queued_count = cur.fetchone()['count']

                        if queued_count > 0:
                            LOG.error(f"🚨 Worker frozen! {queued_count} jobs queued, no activity for {time_since_activity:.1f} minutes")
                            restart_worker_thread()

        except Exception as e:
            LOG.error(f"Worker heartbeat monitor error: {e}")
            LOG.error(traceback.format_exc())
            time.sleep(60)  # Continue monitoring even if error

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
def job_queue_reclaim_loop():
    """
    Job queue reclaim thread - monitors for dead workers via stale heartbeat.
    Runs every 60 seconds and requeues jobs with stale last_updated (>3 minutes).

    CRITICAL: Prevents jobs from getting stuck forever during rolling deployments.
    When Render kills OLD worker mid-job, this thread detects stale heartbeat and requeues.
    """
    LOG.info("🔄 Job queue reclaim thread started (checks every 60s, requeues after 3min stale heartbeat)")

    while True:
        try:
            time.sleep(60)  # Check every minute

            with db() as conn, conn.cursor() as cur:
                # Find jobs with stale heartbeat (no update in 3 minutes = worker likely dead)
                cur.execute("""
                    UPDATE ticker_processing_jobs
                    SET status = 'queued',
                        started_at = NULL,
                        worker_id = NULL,
                        phase = 'reclaimed_dead_worker',
                        progress = 0,
                        error_message = COALESCE(error_message, '') || ' | Reclaimed: Dead worker detected (heartbeat stale >3min)',
                        last_updated = NOW()
                    WHERE status = 'processing'
                    AND last_updated < NOW() - INTERVAL '3 minutes'
                    RETURNING job_id, ticker, worker_id, phase AS old_phase, progress AS old_progress,
                              EXTRACT(EPOCH FROM (NOW() - last_updated)) / 60 AS minutes_stale
                """)

                reclaimed = cur.fetchall()

                if reclaimed:
                    conn.commit()
                    LOG.warning(f"🔄 Job queue reclaim thread reclaimed {len(reclaimed)} jobs with stale heartbeat:")
                    for job in reclaimed:
                        LOG.info(f"   → {job['ticker']} (job_id: {job['job_id']}, worker: {job['worker_id']}, "
                                f"was {job['old_phase']} at {job['old_progress']}%, stale for {job['minutes_stale']:.1f}min)")

                        # Update batch counters
                        cur.execute("""
                            SELECT batch_id FROM ticker_processing_jobs WHERE job_id = %s
                        """, (job['job_id'],))
                        batch_result = cur.fetchone()

                        if batch_result:
                            # Note: Don't increment failed_jobs counter - job is being requeued for retry
                            LOG.info(f"   → Job {job['job_id']} requeued in batch {batch_result['batch_id']}")

                    conn.commit()

        except Exception as e:
            LOG.error(f"❌ Job queue reclaim thread error: {e}")
            LOG.error(traceback.format_exc())
            # Continue running despite errors

def email_queue_watchdog_loop():
    """
    Email queue watchdog - monitors for stalled jobs.
    Runs every 60 seconds and marks jobs with stale heartbeat as failed.
    """
    LOG.info("🐕 Email queue watchdog started (checks every 60s, kills after 3min stale heartbeat)")

    while True:
        try:
            time.sleep(60)  # Check every minute

            with db() as conn, conn.cursor() as cur:
                # Find jobs with stale heartbeat (no update in 3 minutes)
                cur.execute("""
                    UPDATE email_queue
                    SET status = 'failed',
                        error_message = 'Processing stalled (no heartbeat for 3 minutes)',
                        updated_at = NOW()
                    WHERE status = 'processing'
                    AND (heartbeat IS NULL OR heartbeat < NOW() - INTERVAL '3 minutes')
                    RETURNING ticker
                """)

                stalled = cur.fetchall()

                if stalled:
                    conn.commit()
                    LOG.warning(f"⚠️ Email queue watchdog killed {len(stalled)} stalled jobs:")
                    for row in stalled:
                        LOG.info(f"   → {row['ticker']}")

        except Exception as e:
            LOG.error(f"❌ Email queue watchdog error: {e}")
            # Continue running despite errors


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

    # Initialize connection pool BEFORE starting job worker
    LOG.info("=" * 80)
    LOG.info("🔄 Initializing database connection pool...")
    try:
        init_connection_pool()
        LOG.info("=" * 80)
    except Exception as e:
        LOG.error("=" * 80)
        LOG.error(f"💥 STARTUP FAILED: Cannot initialize database connection pool")
        LOG.error(f"   Error: {e}")
        LOG.error(f"   Stacktrace: {traceback.format_exc()}")
        LOG.error("=" * 80)
        LOG.error("🚨 APPLICATION WILL NOT START - Database connection required for parallel processing")
        LOG.error("   Check database status and retry")
        LOG.error("=" * 80)
        raise  # Re-raise to fail FastAPI startup
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

            # Reclaim jobs that were processing but worker died (older than 3 minutes = definitely orphaned)
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
                AND started_at < NOW() - INTERVAL '3 minutes'
                RETURNING job_id, ticker, phase AS old_phase, progress AS old_progress
            """)

            orphaned = cur.fetchall()
            if orphaned:
                LOG.warning(f"🔄 RECLAIMED {len(orphaned)} orphaned jobs (>3min old, server likely restarted):")
                for job in orphaned:
                    LOG.info(f"   → {job['ticker']} was at {job['old_phase']} ({job['old_progress']}%), now queued for retry")

            # Also check for jobs processing <3 minutes (possible crash mid-job)
            cur.execute("""
                SELECT COUNT(*) as recent_count
                FROM ticker_processing_jobs
                WHERE status = 'processing'
                AND started_at >= NOW() - INTERVAL '3 minutes'
            """)
            recent_result = cur.fetchone()
            if recent_result and recent_result['recent_count'] > 0:
                LOG.warning(f"⚠️ {recent_result['recent_count']} jobs started <3min ago still marked 'processing'")
                LOG.warning("   These will NOT be reclaimed yet (might still be running on old worker)")
                LOG.warning("   Job queue reclaim thread will requeue them if heartbeat becomes stale (>3min)")

            if not orphaned and not all_processing:
                LOG.info("✅ No orphaned jobs found - clean startup")

    except Exception as e:
        LOG.error(f"❌ Failed to reclaim orphaned jobs: {e}")
        LOG.error(f"   Stacktrace: {traceback.format_exc()}")

    # Email Queue Recovery: Mark stuck processing jobs as failed
    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT ticker, heartbeat
                FROM email_queue
                WHERE status = 'processing'
            """)
            stuck_emails = cur.fetchall()

            if stuck_emails:
                LOG.warning(f"⚠️ Found {len(stuck_emails)} email queue jobs in 'processing' state")

                # Mark as failed (server restarted during processing)
                cur.execute("""
                    UPDATE email_queue
                    SET status = 'failed',
                        error_message = 'Server restarted during processing',
                        updated_at = NOW()
                    WHERE status = 'processing'
                    RETURNING ticker
                """)

                failed = cur.fetchall()
                if failed:
                    LOG.warning(f"🔄 Marked {len(failed)} email queue jobs as failed:")
                    for row in failed:
                        LOG.info(f"   → {row['ticker']}")

                conn.commit()
            else:
                LOG.info("✅ No stuck email queue jobs found - clean startup")

    except Exception as e:
        LOG.error(f"❌ Failed to recover email queue: {e}")
        LOG.error(f"   Stacktrace: {traceback.format_exc()}")

    start_job_worker()

    # Start job queue reclaim thread (monitors for dead workers via stale heartbeat)
    job_reclaim_thread = threading.Thread(target=job_queue_reclaim_loop, daemon=True, name="JobQueueReclaimThread")
    job_reclaim_thread.start()

    # Start email queue watchdog in separate thread
    email_watchdog_thread = threading.Thread(target=email_queue_watchdog_loop, daemon=True, name="EmailQueueWatchdog")
    email_watchdog_thread.start()

    # Start timeout watchdog in separate thread
    timeout_thread = threading.Thread(target=timeout_watchdog_loop, daemon=True, name="TimeoutWatchdog")
    timeout_thread.start()

    # Start worker heartbeat monitor (restarts worker if frozen)
    global _worker_heartbeat_monitor_thread
    _worker_heartbeat_monitor_thread = threading.Thread(target=worker_heartbeat_monitor_loop, daemon=True, name="WorkerHeartbeatMonitor")
    _worker_heartbeat_monitor_thread.start()

    LOG.info("✅ Job queue system initialized (4 watchdog threads running + worker heartbeat monitor)")

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

    # Note: Thread-local HTTP connectors are cleaned up automatically when threads exit
    # (via cleanup_http_session() at end of each job + thread-local storage garbage collection)

    # Close connection pool
    LOG.info("🔄 Closing database connection pool...")
    close_connection_pool()

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
@APP.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Serve the StockDigest beta signup landing page"""
    return templates.TemplateResponse("signup.html", {"request": request})


@APP.get("/terms-of-service", response_class=HTMLResponse)
async def terms_page(request: Request):
    """Serve Terms of Service page"""
    return templates.TemplateResponse("terms_of_service.html", {"request": request})


@APP.get("/privacy-policy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    """Serve Privacy Policy page"""
    return templates.TemplateResponse("privacy_policy.html", {"request": request})


@APP.get("/unsubscribe", response_class=HTMLResponse)
async def unsubscribe_page(request: Request, token: str = Query(...)):
    """
    Handle unsubscribe requests via token link.
    Idempotent: Can be called multiple times safely.
    """
    LOG.info(f"Unsubscribe request with token: {token[:10]}...")

    try:
        with db() as conn, conn.cursor() as cur:
            # Validate token and get user info
            cur.execute("""
                SELECT ut.user_email, ut.used_at, bu.name, bu.status
                FROM unsubscribe_tokens ut
                JOIN beta_users bu ON ut.user_email = bu.email
                WHERE ut.token = %s
            """, (token,))
            result = cur.fetchone()

            if not result:
                LOG.warning(f"Invalid unsubscribe token: {token[:10]}...")
                return HTMLResponse("""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Invalid Link - StockDigest</title>
                        <style>
                            body {
                                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                                background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
                                min-height: 100vh;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                margin: 0;
                                padding: 20px;
                            }
                            .container {
                                background: white;
                                padding: 40px;
                                border-radius: 12px;
                                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                                max-width: 500px;
                                text-align: center;
                            }
                            h1 { color: #dc2626; margin-bottom: 16px; }
                            p { color: #374151; line-height: 1.6; }
                            a { color: #1e40af; text-decoration: none; }
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>❌ Invalid Unsubscribe Link</h1>
                            <p>This unsubscribe link is invalid or has expired.</p>
                            <p>If you need assistance, please contact us at <a href="mailto:stockdigest.research@gmail.com">stockdigest.research@gmail.com</a></p>
                            <p style="margin-top: 24px;"><a href="/">← Return to Home</a></p>
                        </div>
                    </body>
                    </html>
                """, status_code=404)

            email = result['user_email']
            name = result['name']
            already_cancelled = result['status'] == 'cancelled'
            already_used = result['used_at'] is not None

            # Capture request metadata for security tracking
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get('user-agent', '')

            if not already_cancelled:
                # Mark user as unsubscribed
                cur.execute("""
                    UPDATE beta_users
                    SET status = 'cancelled'
                    WHERE email = %s
                """, (email,))
                LOG.info(f"Unsubscribed user: {email}")

            if not already_used:
                # Mark token as used
                cur.execute("""
                    UPDATE unsubscribe_tokens
                    SET used_at = NOW(), ip_address = %s, user_agent = %s
                    WHERE token = %s
                """, (ip_address, user_agent, token))
                LOG.info(f"Marked token as used for {email}")

            conn.commit()

            # Return success page
            return HTMLResponse(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Unsubscribed - StockDigest</title>
                    <style>
                        body {{
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
                            min-height: 100vh;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin: 0;
                            padding: 20px;
                        }}
                        .container {{
                            background: white;
                            padding: 40px;
                            border-radius: 12px;
                            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                            max-width: 500px;
                            text-align: center;
                        }}
                        h1 {{ color: #059669; margin-bottom: 16px; }}
                        p {{ color: #374151; line-height: 1.6; margin-bottom: 12px; }}
                        .email {{ background: #f3f4f6; padding: 8px 12px; border-radius: 4px; font-family: monospace; }}
                        a {{ color: #1e40af; text-decoration: none; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>✅ Successfully Unsubscribed</h1>
                        <p><strong>{name}</strong>, you've been unsubscribed from StockDigest.</p>
                        <p class="email">{email}</p>
                        <p style="margin-top: 24px;">You will no longer receive daily stock intelligence reports.</p>
                        <p style="margin-top: 16px; font-size: 14px; color: #6b7280;">
                            Changed your mind? <a href="/">Re-subscribe here</a>
                        </p>
                        <p style="margin-top: 24px; font-size: 14px; color: #6b7280;">
                            Questions? <a href="mailto:stockdigest.research@gmail.com">Contact us</a>
                        </p>
                    </div>
                </body>
                </html>
            """)

    except Exception as e:
        LOG.error(f"Error processing unsubscribe: {e}")
        LOG.error(traceback.format_exc())

        return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head><title>Error - StockDigest</title></head>
            <body>
                <h1>Something went wrong</h1>
                <p>We couldn't process your unsubscribe request. Please try again or contact support.</p>
                <p><a href="mailto:stockdigest.research@gmail.com">stockdigest.research@gmail.com</a></p>
            </body>
            </html>
        """, status_code=500)


# ------------------------------------------------------------------------------
# Beta Signup API Endpoints
# ------------------------------------------------------------------------------

@APP.get("/api/validate-ticker")
async def validate_ticker_endpoint(ticker: str = Query(..., min_length=1, max_length=10)):
    """
    Validate ticker with smart Canadian .TO suggestions.
    Public endpoint (no auth required) for live form validation.

    Returns:
    - valid=True: Ticker found in database
    - valid=False + suggestion: Ticker not found, but .TO variant exists
    - valid=False: No matches found
    """
    try:
        # Normalize ticker format (handles case, removes quotes, etc)
        normalized = normalize_ticker_format(ticker)

        # TIER 1: Format validation (reject obvious garbage immediately)
        if not validate_ticker_format(normalized):
            return {
                "valid": False,
                "message": "Invalid ticker format"
            }

        # TIER 2: Database whitelist (only allow approved tickers)
        config = get_ticker_config(normalized)

        # Check if ticker exists in database (has_full_config=True means it's real, False means fallback)
        if config and config.get('has_full_config', True):
            # Real ticker found in database ✓
            return {
                "valid": True,
                "ticker": normalized,
                "company_name": config.get("company_name", "Unknown"),
                "exchange": config.get("exchange", "Unknown"),
                "country": config.get("country", "Unknown")
            }

        # No exact match - try Canadian variant (.TO suffix)
        # Only if: no existing suffix AND ticker is 2-5 chars (typical Canadian ticker length)
        if '.' not in normalized and 2 <= len(normalized) <= 5:
            canadian_ticker = f"{normalized}.TO"
            canadian_config = get_ticker_config(canadian_ticker)

            # Check if Canadian variant exists in database (not just fallback)
            if canadian_config and canadian_config.get('has_full_config', True):
                # Found Canadian variant - suggest it
                return {
                    "valid": False,
                    "suggestion": {
                        "ticker": canadian_ticker,
                        "company_name": canadian_config.get("company_name", "Unknown"),
                        "exchange": canadian_config.get("exchange", "TSX"),
                        "country": canadian_config.get("country", "Canada"),
                        "message": f"Did you mean {canadian_ticker}?"
                    }
                }

        # No matches found in database - ticker not recognized
        return {
            "valid": False,
            "message": "Ticker not recognized"
        }

    except Exception as e:
        LOG.error(f"Ticker validation error for '{ticker}': {e}")
        return {
            "valid": False,
            "message": "Validation error. Please try again."
        }


class BetaSignupRequest(BaseModel):
    """Pydantic model for beta signup form"""
    name: str
    email: str
    ticker1: str
    ticker2: str
    ticker3: str


def generate_unsubscribe_token(email: str) -> str:
    """
    Generate cryptographically secure unsubscribe token for user.
    Returns: 43-character URL-safe token
    """
    token = secrets.token_urlsafe(32)  # 32 bytes = 43 chars base64

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO unsubscribe_tokens (user_email, token)
                VALUES (%s, %s)
                ON CONFLICT (token) DO NOTHING
                RETURNING token
            """, (email, token))
            result = cur.fetchone()

            if not result:
                # Token collision (astronomically rare), retry
                LOG.warning(f"Token collision for {email}, regenerating")
                return generate_unsubscribe_token(email)

            conn.commit()
            LOG.info(f"Generated unsubscribe token for {email}")
            return token
    except Exception as e:
        LOG.error(f"Error generating unsubscribe token: {e}")
        raise


def get_or_create_unsubscribe_token(email: str) -> str:
    """
    Get existing unsubscribe token or create new one.
    Reuses token if user hasn't unsubscribed yet.
    Returns empty string if email is not a beta user (admin/test emails).
    """
    try:
        with db() as conn, conn.cursor() as cur:
            # First, check if email exists in beta_users (required for foreign key)
            cur.execute("""
                SELECT email FROM beta_users WHERE email = %s
            """, (email,))
            user_exists = cur.fetchone()

            if not user_exists:
                LOG.warning(f"Email {email} not in beta_users, skipping unsubscribe token generation")
                return ""  # Return empty string for admin/test emails

            # Check if unused token exists
            cur.execute("""
                SELECT token FROM unsubscribe_tokens
                WHERE user_email = %s AND used_at IS NULL
                ORDER BY created_at DESC LIMIT 1
            """, (email,))
            result = cur.fetchone()

            if result:
                return result['token']

            # No unused token found, generate new one
            return generate_unsubscribe_token(email)
    except Exception as e:
        LOG.error(f"Error getting unsubscribe token: {e}")
        # Fallback: return empty string (email will have generic unsubscribe link)
        return ""


@APP.post("/api/beta-signup")
async def beta_signup_endpoint(signup: BetaSignupRequest):
    """
    Handle beta sign-up form submissions with strict ticker validation.
    Public endpoint (no auth required).
    """
    try:
        # Validate and clean input
        name = signup.name.strip()
        email = signup.email.strip().lower()

        # Validate name
        if not name or len(name) > 255:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Name must be 1-255 characters"}
            )

        # Validate email format
        import re
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, email):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Invalid email format"}
            )

        # Normalize and validate tickers
        tickers = []
        ticker_data = []
        for ticker_input in [signup.ticker1, signup.ticker2, signup.ticker3]:
            normalized = normalize_ticker_format(ticker_input.strip())
            config = get_ticker_config(normalized)

            if not config:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Ticker '{ticker_input}' not recognized"}
                )

            tickers.append(normalized)
            ticker_data.append({
                "ticker": normalized,
                "company": config.get("company_name", "Unknown")
            })

        # Check for duplicate email
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT email FROM beta_users WHERE email = %s", (email,))
            if cur.fetchone():
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "Email already registered"}
                )

        # Save to database with terms tracking
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO beta_users (
                    name, email, ticker1, ticker2, ticker3, created_at, status,
                    terms_version, terms_accepted_at, privacy_version, privacy_accepted_at
                )
                VALUES (%s, %s, %s, %s, %s, NOW(), 'active', %s, NOW(), %s, NOW())
            """, (name, email, tickers[0], tickers[1], tickers[2], TERMS_VERSION, PRIVACY_VERSION))
            conn.commit()

        # Generate unsubscribe token
        try:
            unsubscribe_token = generate_unsubscribe_token(email)
            LOG.info(f"Created unsubscribe token for {email}")
        except Exception as e:
            LOG.error(f"Failed to create unsubscribe token for {email}: {e}")
            # Don't block signup if token generation fails

        # Send admin notification email
        send_beta_signup_notification(name, email, tickers[0], tickers[1], tickers[2])

        LOG.info(f"✅ New beta user: {email} tracking {tickers[0]}, {tickers[1]}, {tickers[2]}")

        return {
            "status": "success",
            "message": "Welcome to StockDigest beta!",
            "tickers": ticker_data
        }

    except Exception as e:
        LOG.error(f"Beta signup error: {e}")
        LOG.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Server error. Please try again."}
        )


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
        },
        "heartbeat_monitor": {
            "restart_count": _worker_restart_count,
            "last_activity": _last_worker_activity.isoformat() if _last_worker_activity else None
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

    # AUTO-INITIALIZE: Ensure CSV sync + feeds exist for all tickers
    # This matches the PowerShell test workflow (calls /admin/init before /jobs/submit)
    LOG.info(f"🔧 Auto-initializing {len(body.tickers)} tickers before queueing...")

    try:
        init_result = await admin_init(request, InitRequest(
            tickers=body.tickers,
            force_refresh=False  # Use cache if already initialized this session
        ))

        if init_result.get('status') == 'success':
            total_feeds = init_result.get('total_feeds_created', 0)
            LOG.info(f"✅ Initialization complete: {total_feeds} feeds created/verified")
        else:
            LOG.warning(f"⚠️ Initialization had issues: {init_result}")
    except Exception as e:
        LOG.error(f"❌ Initialization failed: {e}")
        # Don't fail the entire batch - safety check in process_ticker_job will catch this
        LOG.warning(f"⚠️ Continuing despite init failure - individual jobs will retry")

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
                                   a.scraped_content, ta.ai_summary, a.url_hash, f.company_name,
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
                            LEFT JOIN feeds f ON ta.feed_id = f.id
                            WHERE ta.ticker = ANY(%s)
                            AND (a.published_at >= %s OR a.published_at IS NULL)
                        )
                        SELECT id, url, resolved_url, title, domain, published_at,
                               category, search_keyword, competitor_ticker, ticker,
                               scraped_content, ai_summary, url_hash, company_name
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
                                   a.scraped_content, ta.ai_summary, a.url_hash, f.company_name,
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
                            LEFT JOIN feeds f ON ta.feed_id = f.id
                            WHERE (a.published_at >= %s OR a.published_at IS NULL)
                        )
                        SELECT id, url, resolved_url, title, domain, published_at,
                               category, search_keyword, competitor_ticker, ticker,
                               scraped_content, ai_summary, url_hash, company_name
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
                # Use fallback triage (Claude primary, OpenAI fallback)
                selected_results = await perform_ai_triage_with_fallback_async(articles_by_ticker[ticker], ticker, triage_batch_size)
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

        # ============================================================================
        # PHASE 4 DISABLED - MOVED TO DIGEST PHASE (Oct 2025)
        # ============================================================================
        # REASON: Phase 4 (scraping) was running BEFORE Phase 1.5 (Google URL resolution),
        # causing all Google News URLs to fail scraping. Scraping now happens ONLY in
        # process_digest_phase() which runs AFTER Phase 1.5, ensuring URLs are resolved first.
        #
        # TIMELINE FIX:
        #   OLD (BROKEN): Phase 1 → Phase 4 (scrape unresolved URLs) → Phase 1.5 (resolve too late)
        #   NEW (FIXED):  Phase 1 → Phase 1.5 (resolve) → Digest Phase (scrape resolved URLs)
        #
        # See: fetch_digest_articles_with_enhanced_content() for scraping implementation
        # ============================================================================

        # PHASE 4: Ticker-specific content scraping and analysis (WITH ASYNC BATCH PROCESSING)
        # COMMENTED OUT - NOW HAPPENS IN DIGEST PHASE
        # LOG.info("=== PHASE 4: TICKER-SPECIFIC CONTENT SCRAPING AND ANALYSIS (ASYNC BATCHES) ===")
        # memory_monitor.take_snapshot("PHASE4_START")
        scraping_final_stats = {"scraped": 0, "failed": 0, "ai_analyzed": 0, "reused_existing": 0}

        # PHASE 4 CODE BLOCK COMMENTED OUT (150+ lines)
        # Scraping now happens in fetch_digest_articles_with_enhanced_content()
        # which is called during process_digest_phase() AFTER Phase 1.5 URL resolution

        # Initialize variables needed for response structure
        total_articles_to_process = 0
        # for target_ticker in articles_by_ticker.keys():
        #     selected = triage_results.get(target_ticker, {})
        #     for category in ["company", "industry", "competitor"]:
        #         total_articles_to_process += len(selected.get(category, []))

        # processed_count = 0
        # LOG.info(f"Starting Phase 4: {total_articles_to_process} total articles to process in batches of {SCRAPE_BATCH_SIZE}")

        # Track which tickers were successfully processed
        successfully_processed_tickers = set()

        # [PHASE 4 SCRAPING LOOP DISABLED - 150+ lines commented out]
        # The entire scraping loop has been removed because it ran BEFORE Phase 1.5 URL resolution.
        # Scraping now happens in fetch_digest_articles_with_enhanced_content() during digest phase.
        # See process_digest_phase() which calls fetch_digest_articles_with_enhanced_content()

        # Skip Phase 4 - scraping happens in digest phase after URL resolution
        LOG.info("=== PHASE 4 SKIPPED - Scraping moved to Digest Phase (after URL resolution) ===")
        memory_monitor.take_snapshot("PHASE4_SKIPPED")

        # [END OF PHASE 4 BLOCK - 150+ lines of scraping code removed]
        
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
        
        # Stop memory monitoring and get summary
        memory_summary = memory_monitor.stop_monitoring()

        # Prepare response with monitoring data
        response = {
            "status": "completed",
            "processing_time_seconds": round(processing_time, 1),
            "workflow": "ingest_triage_email1_only",

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
            "phase_4_scraping": {
                "status": "disabled_moved_to_digest_phase",
                "reason": "Phase 4 scraping removed from ingest - now happens in digest phase after URL resolution",
                "note": "Scraping happens in fetch_digest_articles_with_enhanced_content() during process_digest_phase()"
            },
            "successfully_processed_tickers": list(successfully_processed_tickers),
            "message": "Ingest phase completed - scraping deferred to digest phase (after URL resolution)",
            "github_sync_required": False,  # No scraping happened, so no new content to sync

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
    """Generate and send email digest with content scraping data and AI summaries"""
    require_admin(request)
    ensure_schema()

    try:
        LOG.info(f"=== DIGEST GENERATION STARTING ===")
        LOG.info(f"Time window: {minutes} minutes, Tickers: {tickers}")

        # Use the existing enhanced digest function that sends emails
        LOG.info("Calling enhanced digest function...")
        result = await fetch_digest_articles_with_enhanced_content(minutes / 60, tickers)

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


@APP.post("/admin/export-user-csv")
async def admin_export_user_csv(request: Request):
    """
    Manually trigger CSV export of beta users.
    Requires X-Admin-Token authentication.
    """
    # Validate admin token
    admin_token = request.headers.get("X-Admin-Token")
    if not admin_token or admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        count = export_beta_users_to_csv()
        return {
            "status": "success",
            "message": f"Exported {count} active beta users to CSV",
            "file_path": "data/user_tickers.csv",
            "exported_at": datetime.now().isoformat()
        }
    except Exception as e:
        LOG.error(f"CSV export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@APP.post("/admin/force-digest")
async def force_digest(request: Request, body: ForceDigestRequest):
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
    html = await build_enhanced_digest_html(articles_by_ticker, 7)
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
            model="claude-sonnet-4-5-20250929",
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
# ADMIN DASHBOARD ENDPOINTS
# ------------------------------------------------------------------------------

def check_admin_token(token: str) -> bool:
    """Validate admin token from query param or header"""
    admin_token = os.getenv('ADMIN_TOKEN')
    return token == admin_token if admin_token else False

def get_lookback_minutes() -> int:
    """
    Get configured lookback window from system_config table.
    Used by production workflows (cron jobs and admin bulk actions).
    Test portal (/admin/test) has separate hardcoded settings.

    Returns:
        int: Lookback window in minutes (default: 1440 = 1 day)
    """
    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT value FROM system_config WHERE key = 'lookback_minutes'")
            row = cur.fetchone()
            if row:
                minutes = int(row['value'])  # Fixed: use dict access, not tuple index
                # Validate: must be at least 60 minutes (1 hour)
                if minutes < 60:
                    LOG.warning(f"⚠️ Invalid lookback_minutes in system_config: {minutes} < 60, using default 1440")
                    return 1440
                return minutes
            else:
                LOG.warning("⚠️ No lookback_minutes in system_config, using default 1440 (1 day)")
                return 1440  # 1 day default
    except Exception as e:
        LOG.error(f"Failed to fetch lookback_minutes: {e}, using default 1440")
        return 1440  # 1 day default

@APP.get("/admin")
def admin_dashboard(request: Request, token: str = Query(...)):
    """Admin dashboard landing page"""
    if not check_admin_token(token):
        return HTMLResponse("Unauthorized", status_code=401)

    return templates.TemplateResponse("admin.html", {
        "request": request,
        "token": token
    })

@APP.get("/admin/users")
def admin_users_page(request: Request, token: str = Query(...)):
    """Beta user management page"""
    if not check_admin_token(token):
        return HTMLResponse("Unauthorized", status_code=401)

    return templates.TemplateResponse("admin_users.html", {
        "request": request,
        "token": token
    })

@APP.get("/admin/queue")
def admin_queue_page(request: Request, token: str = Query(...)):
    """Email queue management page"""
    if not check_admin_token(token):
        return HTMLResponse("Unauthorized", status_code=401)

    return templates.TemplateResponse("admin_queue.html", {
        "request": request,
        "token": token
    })

@APP.get("/admin/test")
def admin_test_page(request: Request, token: str = Query(...)):
    """Test runner page - web-based version of setup_job_queue.ps1"""
    if not check_admin_token(token):
        return HTMLResponse("Unauthorized", status_code=401)

    return templates.TemplateResponse("admin_test.html", {
        "request": request,
        "token": token
    })

@APP.get("/admin/settings")
def admin_settings_page(request: Request, token: str = Query(...)):
    """Admin settings page - Lookback window and GitHub commit"""
    if not check_admin_token(token):
        return HTMLResponse("Unauthorized", status_code=401)

    return templates.TemplateResponse("admin_settings.html", {
        "request": request,
        "token": token
    })

# Admin API endpoints
@APP.get("/api/admin/stats")
def get_admin_stats(token: str = Query(...)):
    """Get dashboard stats - uses unified queue logic to match /admin/queue page"""
    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            # Count pending users
            cur.execute("SELECT COUNT(*) as count FROM beta_users WHERE status = 'pending'")
            pending_users = cur.fetchone()['count']

            # Count active users
            cur.execute("SELECT COUNT(*) as count FROM beta_users WHERE status = 'active'")
            active_users = cur.fetchone()['count']

            # Get unified queue status (same logic as /api/queue-status)
            # Query 1: Get active processing jobs (mode='daily')
            cur.execute("""
                SELECT ticker, status
                FROM ticker_processing_jobs
                WHERE config->>'mode' = 'daily'
                AND status IN ('queued', 'processing')
            """)
            active_jobs = cur.fetchall()

            # Query 2: Get email queue
            cur.execute("""
                SELECT ticker, status, sent_at
                FROM email_queue
            """)
            email_queue_rows = cur.fetchall()

            # Build unified status dict (same merging logic as /api/queue-status)
            tickers_dict = {}

            # Add email queue entries
            for row in email_queue_rows:
                tickers_dict[row['ticker']] = row['status']

            # Override with active jobs (processing takes precedence)
            for job in active_jobs:
                tickers_dict[job['ticker']] = 'processing'

            # Count by status
            ready_emails = sum(1 for status in tickers_dict.values() if status == 'ready')

            # Count sent today (from email_queue only)
            cur.execute("SELECT COUNT(*) as count FROM email_queue WHERE status = 'sent' AND sent_at >= CURRENT_DATE")
            sent_today = cur.fetchone()['count']

            return {
                "status": "success",
                "pending_users": pending_users,
                "active_users": active_users,
                "ready_emails": ready_emails,
                "sent_today": sent_today
            }
    except Exception as e:
        LOG.error(f"Failed to get admin stats: {e}")
        return {"status": "error", "message": str(e)}

@APP.get("/api/admin/users")
def get_all_users(token: str = Query(...)):
    """Get all beta users"""
    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, email, ticker1, ticker2, ticker3, status,
                       created_at, terms_accepted_at, privacy_accepted_at
                FROM beta_users
                ORDER BY
                    CASE status
                        WHEN 'pending' THEN 1
                        WHEN 'active' THEN 2
                        WHEN 'paused' THEN 3
                        WHEN 'cancelled' THEN 4
                    END,
                    created_at DESC
            """)
            users = cur.fetchall()

            return {
                "status": "success",
                "users": users
            }
    except Exception as e:
        LOG.error(f"Failed to get users: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/admin/approve-user")
async def approve_user(request: Request):
    """Approve a pending user"""
    body = await request.json()
    token = body.get('token')
    email = body.get('email')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE beta_users
                SET status = 'active'
                WHERE email = %s
            """, (email,))
            conn.commit()

            LOG.info(f"✅ Approved user: {email}")
            return {"status": "success", "message": f"Approved {email}"}
    except Exception as e:
        LOG.error(f"Failed to approve user: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/admin/reject-user")
async def reject_user(request: Request):
    """Reject a pending user"""
    body = await request.json()
    token = body.get('token')
    email = body.get('email')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE beta_users
                SET status = 'cancelled'
                WHERE email = %s
            """, (email,))
            conn.commit()

            LOG.info(f"❌ Rejected user: {email}")
            return {"status": "success", "message": f"Rejected {email}"}
    except Exception as e:
        LOG.error(f"Failed to reject user: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/admin/pause-user")
async def pause_user(request: Request):
    """Pause an active user"""
    body = await request.json()
    token = body.get('token')
    email = body.get('email')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE beta_users
                SET status = 'paused'
                WHERE email = %s
            """, (email,))
            conn.commit()

            LOG.info(f"⏸️ Paused user: {email}")
            return {"status": "success", "message": f"Paused {email}"}
    except Exception as e:
        LOG.error(f"Failed to pause user: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/admin/cancel-user")
async def cancel_user(request: Request):
    """Cancel a user (same as unsubscribe)"""
    body = await request.json()
    token = body.get('token')
    email = body.get('email')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE beta_users
                SET status = 'cancelled'
                WHERE email = %s
            """, (email,))
            conn.commit()

            LOG.info(f"🗑️ Cancelled user: {email}")
            return {"status": "success", "message": f"Cancelled {email}"}
    except Exception as e:
        LOG.error(f"Failed to cancel user: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/admin/reactivate-user")
async def reactivate_user(request: Request):
    """Reactivate a paused or cancelled user"""
    body = await request.json()
    token = body.get('token')
    email = body.get('email')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE beta_users
                SET status = 'active'
                WHERE email = %s
            """, (email,))
            conn.commit()

            LOG.info(f"▶️ Reactivated user: {email}")
            return {"status": "success", "message": f"Reactivated {email}"}
    except Exception as e:
        LOG.error(f"Failed to reactivate user: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/admin/delete-user")
async def delete_user(request: Request):
    """Permanently delete a user from the database (also deletes unsubscribe tokens via CASCADE)"""
    body = await request.json()
    token = body.get('token')
    email = body.get('email')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            # Delete user (unsubscribe_tokens will be auto-deleted via ON DELETE CASCADE)
            cur.execute("""
                DELETE FROM beta_users
                WHERE email = %s
            """, (email,))

            deleted_count = cur.rowcount
            conn.commit()

            if deleted_count == 0:
                return {"status": "error", "message": f"User {email} not found"}

            LOG.info(f"🗑️ Permanently deleted user: {email}")
            return {"status": "success", "message": f"Deleted {email}"}
    except Exception as e:
        LOG.error(f"Failed to delete user: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/admin/restart-worker")
async def restart_worker_api(request: Request):
    """Restart worker thread only (gentle, 0 downtime)"""
    body = await request.json()
    token = body.get('token')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        LOG.warning("🔄 Admin requested worker thread restart")
        restart_worker_thread()
        return {"status": "success", "message": "Worker thread restarted successfully"}
    except Exception as e:
        LOG.error(f"Failed to restart worker: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/admin/restart-server")
async def restart_server_api(request: Request):
    """Force full server restart (exits process, Render auto-restarts, ~10-20s downtime)"""
    body = await request.json()
    token = body.get('token')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    LOG.critical("💀 Admin requested FULL SERVER RESTART - exiting process in 2 seconds")
    LOG.critical("Render will automatically detect the crash and restart the service")

    # Give time for response to be sent
    import asyncio
    await asyncio.sleep(2)

    # Force exit (Render will restart the entire service)
    os._exit(1)

# Email Queue API endpoints
@APP.get("/api/queue-status")
def get_queue_status(token: str = Query(...)):
    """
    Get unified queue status - combines active job processing AND email queue.

    Shows:
    - Active jobs from ticker_processing_jobs (mode='daily', status='processing')
    - Completed emails from email_queue (ready/sent/failed/cancelled)

    This provides end-to-end visibility from job start to email delivery.
    """
    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            # Query 1: Get active processing jobs (mode='daily')
            cur.execute("""
                SELECT
                    j.ticker,
                    j.status as job_status,
                    j.phase,
                    j.progress,
                    j.worker_id,
                    j.started_at,
                    j.last_updated,
                    j.error_message,
                    j.config->>'recipients' as recipients_json,
                    EXTRACT(EPOCH FROM (NOW() - j.started_at)) / 60 AS minutes_running
                FROM ticker_processing_jobs j
                WHERE j.config->>'mode' = 'daily'
                AND j.status IN ('queued', 'processing')
                ORDER BY j.created_at
            """)
            active_jobs = cur.fetchall()

            # Query 2: Get email queue (completed emails)
            cur.execute("""
                SELECT ticker, company_name, recipients, email_subject,
                       article_count, status, error_message, heartbeat,
                       created_at, updated_at, sent_at
                FROM email_queue
                ORDER BY
                    CASE status
                        WHEN 'processing' THEN 1
                        WHEN 'ready' THEN 2
                        WHEN 'failed' THEN 3
                        WHEN 'sent' THEN 4
                        WHEN 'cancelled' THEN 5
                    END,
                    ticker
            """)
            email_queue_rows = cur.fetchall()

            # Build unified ticker list
            tickers_dict = {}

            # Add email queue entries
            for row in email_queue_rows:
                tickers_dict[row['ticker']] = {
                    "ticker": row['ticker'],
                    "company_name": row.get('company_name'),
                    "recipients": row.get('recipients'),
                    "email_subject": row.get('email_subject'),
                    "article_count": row.get('article_count'),
                    "status": row['status'],  # ready, sent, failed, cancelled
                    "error_message": row.get('error_message'),
                    "heartbeat": row.get('heartbeat'),
                    "created_at": row.get('created_at'),
                    "updated_at": row.get('updated_at'),
                    "sent_at": row.get('sent_at'),
                    "source": "email_queue",
                    "progress": None,  # Email queue doesn't have progress
                    "phase": None
                }

            # Add active jobs (override if ticker exists in email_queue)
            for job in active_jobs:
                ticker = job['ticker']

                # Parse recipients from JSON if available
                recipients_json = job.get('recipients_json')
                recipients = json.loads(recipients_json) if recipients_json else None

                tickers_dict[ticker] = {
                    "ticker": ticker,
                    "company_name": None,  # Will be filled when email generated
                    "recipients": recipients,
                    "email_subject": None,
                    "article_count": None,
                    "status": "processing",  # Active job
                    "error_message": job.get('error_message'),
                    "heartbeat": job.get('last_updated'),
                    "created_at": job.get('started_at'),
                    "updated_at": job.get('last_updated'),
                    "sent_at": None,
                    "source": "job_queue",
                    "progress": job.get('progress'),  # 0-100%
                    "phase": job.get('phase'),  # e.g. "ingest_complete", "digest_start"
                    "minutes_running": round(job.get('minutes_running', 0), 1) if job.get('minutes_running') else 0
                }

            # Convert dict to list and sort
            tickers_list = list(tickers_dict.values())
            tickers_list.sort(key=lambda x: (
                1 if x['status'] == 'processing' else
                2 if x['status'] == 'ready' else
                3 if x['status'] == 'failed' else
                4 if x['status'] == 'sent' else 5,
                x['ticker']
            ))

            # Calculate stats
            stats = {
                "processing": sum(1 for t in tickers_list if t['status'] == 'processing'),
                "ready": sum(1 for t in tickers_list if t['status'] == 'ready'),
                "failed": sum(1 for t in tickers_list if t['status'] == 'failed'),
                "sent": sum(1 for t in tickers_list if t['status'] == 'sent'),
                "cancelled": sum(1 for t in tickers_list if t['status'] == 'cancelled')
            }

            return {
                "status": "success",
                "tickers": tickers_list,
                # Flatten stats for frontend (expects data.ready, not data.stats.ready)
                "processing": stats["processing"],
                "ready": stats["ready"],
                "failed": stats["failed"],
                "sent": stats["sent"],
                "cancelled": stats["cancelled"]
            }
    except Exception as e:
        LOG.error(f"Failed to get queue status: {e}")
        LOG.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

@APP.post("/api/send-all-ready")
async def send_all_ready_api(request: Request):
    """Send all ready emails"""
    body = await request.json()
    token = body.get('token')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    result = send_all_ready_emails_impl()
    return result

@APP.post("/api/fix-inconsistent-emails")
async def fix_inconsistent_emails_api(request: Request):
    """Fix emails that have sent_at set but status is still 'ready' (data inconsistency)"""
    body = await request.json()
    token = body.get('token')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            # Find inconsistent emails
            cur.execute("""
                SELECT ticker, status, sent_at
                FROM email_queue
                WHERE status = 'ready' AND sent_at IS NOT NULL
            """)
            inconsistent = cur.fetchall()

            # Fix them by setting status to 'sent'
            cur.execute("""
                UPDATE email_queue
                SET status = 'sent'
                WHERE status = 'ready' AND sent_at IS NOT NULL
            """)
            fixed_count = cur.rowcount
            conn.commit()

            LOG.warning(f"🔧 Fixed {fixed_count} emails with inconsistent status (ready but already sent)")

            return {
                "status": "success",
                "fixed_count": fixed_count,
                "fixed_tickers": [e['ticker'] for e in inconsistent],
                "message": f"Fixed {fixed_count} emails that were already sent but had status='ready'"
            }
    except Exception as e:
        LOG.error(f"Failed to fix inconsistent emails: {e}")
        return {"status": "error", "message": str(e)}
@APP.post("/api/rerun-ticker")
async def rerun_ticker_api(request: Request):
    """Rerun single ticker - uses unified job queue"""
    body = await request.json()
    token = body.get('token')
    ticker = body.get('ticker')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    if not ticker:
        return {"status": "error", "message": "Ticker required"}

    try:
        # Get recipients from email_queue
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT recipients FROM email_queue WHERE ticker = %s
            """, (ticker,))
            row = cur.fetchone()

            if not row or not row['recipients']:
                return {"status": "error", "message": f"Ticker {ticker} not found in queue"}

            recipients = row['recipients']

        # Submit to job queue system
        with db() as conn, conn.cursor() as cur:
            # Create batch record
            cur.execute("""
                INSERT INTO ticker_processing_batches (total_jobs, created_by, config)
                VALUES (%s, %s, %s)
                RETURNING batch_id
            """, (1, 'admin_ui_single', json.dumps({
                "minutes": get_lookback_minutes(),
                "batch_size": 3,
                "triage_batch_size": 3,
                "mode": "daily"
            })))

            batch_id = cur.fetchone()['batch_id']

            # Create single job
            timeout_at = datetime.now(timezone.utc) + timedelta(minutes=45)
            cur.execute("""
                INSERT INTO ticker_processing_jobs (
                    batch_id, ticker, config, timeout_at
                )
                VALUES (%s, %s, %s, %s)
            """, (batch_id, ticker, json.dumps({
                "minutes": get_lookback_minutes(),
                "batch_size": 3,
                "triage_batch_size": 3,
                "mode": "daily",
                "recipients": recipients
            }), timeout_at))

            conn.commit()

        LOG.info(f"[{ticker}] 🔄 Re-run triggered by admin (batch {batch_id})")
        return {
            "status": "success",
            "ticker": ticker,
            "batch_id": str(batch_id),
            "message": "Processing started. Check dashboard in 2-3 minutes."
        }

    except Exception as e:
        LOG.error(f"Failed to rerun ticker {ticker}: {e}")
        return {"status": "error", "message": str(e)}


@APP.post("/api/retry-failed-and-cancelled")
async def retry_failed_and_cancelled_api(request: Request):
    """Retry all failed and cancelled tickers (non-ready only) - uses existing job queue"""
    body = await request.json()
    token = body.get('token')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        # CRITICAL: Load CSV from GitHub BEFORE processing
        try:
            sync_ticker_references_from_github()
        except Exception as e:
            LOG.error(f"❌ CSV sync failed: {e} - continuing anyway")

        # Get all non-ready tickers (failed + cancelled)  with their recipients
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT ticker, company_name, recipients, status
                FROM email_queue
                WHERE status IN ('failed', 'cancelled')
                ORDER BY ticker
            """)
            retry_rows = cur.fetchall()

        if not retry_rows:
            return {
                "status": "success",
                "ticker_count": 0,
                "message": "No failed or cancelled tickers to retry"
            }

        # Build ticker_recipients dict
        ticker_recipients = {
            row['ticker']: row['recipients']
            for row in retry_rows
        }

        # Submit to existing job queue system
        tickers_list = sorted(list(ticker_recipients.keys()))

        with db() as conn, conn.cursor() as cur:
            # Create batch record
            cur.execute("""
                INSERT INTO ticker_processing_batches (total_jobs, created_by, config)
                VALUES (%s, %s, %s)
                RETURNING batch_id
            """, (len(tickers_list), 'admin_ui_retry', json.dumps({
                "minutes": get_lookback_minutes(),
                "batch_size": 3,
                "triage_batch_size": 3,
                "mode": "daily"
            })))

            batch_id = cur.fetchone()['batch_id']

            # Create individual jobs with mode='daily' and recipients
            timeout_at = datetime.now(timezone.utc) + timedelta(minutes=45)
            for ticker in tickers_list:
                cur.execute("""
                    INSERT INTO ticker_processing_jobs (
                        batch_id, ticker, config, timeout_at
                    )
                    VALUES (%s, %s, %s, %s)
                """, (batch_id, ticker, json.dumps({
                    "minutes": get_lookback_minutes(),
                    "batch_size": 3,
                    "triage_batch_size": 3,
                    "mode": "daily",
                    "recipients": ticker_recipients[ticker]
                }), timeout_at))

            conn.commit()

        LOG.info(f"🔄 Retry failed & cancelled: {len(tickers_list)} tickers (batch {batch_id})")

        # Build response with ticker details
        affected_tickers = [
            {
                "ticker": row['ticker'],
                "company_name": row['company_name'],
                "status": row['status']
            }
            for row in retry_rows
        ]

        return {
            "status": "success",
            "batch_id": str(batch_id),
            "ticker_count": len(ticker_recipients),
            "affected_tickers": affected_tickers,
            "tickers": tickers_list,
            "message": f"Retrying {len(ticker_recipients)} failed/cancelled tickers. Processing time: ~15-30 minutes."
        }

    except Exception as e:
        LOG.error(f"Failed to retry failed & cancelled tickers: {e}")
        return {"status": "error", "message": str(e)}


@APP.post("/api/rerun-all-queue")
async def rerun_all_queue_api(request: Request):
    """Rerun ALL tickers in queue from scratch (regardless of status) - uses existing job queue"""
    body = await request.json()
    token = body.get('token')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        # CRITICAL: Load CSV from GitHub BEFORE processing
        try:
            sync_ticker_references_from_github()
        except Exception as e:
            LOG.error(f"❌ CSV sync failed: {e} - continuing anyway")

        # Get ALL tickers with their recipients and status
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT ticker, company_name, recipients, status
                FROM email_queue
                ORDER BY ticker
            """)
            all_rows = cur.fetchall()

        if not all_rows:
            return {
                "status": "success",
                "ticker_count": 0,
                "message": "No tickers in queue to rerun"
            }

        # Build ticker_recipients dict
        ticker_recipients = {
            row['ticker']: row['recipients']
            for row in all_rows
        }

        # Submit to existing job queue system
        tickers_list = sorted(list(ticker_recipients.keys()))

        with db() as conn, conn.cursor() as cur:
            # Create batch record
            cur.execute("""
                INSERT INTO ticker_processing_batches (total_jobs, created_by, config)
                VALUES (%s, %s, %s)
                RETURNING batch_id
            """, (len(tickers_list), 'admin_ui_rerun_all', json.dumps({
                "minutes": get_lookback_minutes(),
                "batch_size": 3,
                "triage_batch_size": 3,
                "mode": "daily"
            })))

            batch_id = cur.fetchone()['batch_id']

            # Create individual jobs with mode='daily' and recipients
            timeout_at = datetime.now(timezone.utc) + timedelta(minutes=45)
            for ticker in tickers_list:
                cur.execute("""
                    INSERT INTO ticker_processing_jobs (
                        batch_id, ticker, config, timeout_at
                    )
                    VALUES (%s, %s, %s, %s)
                """, (batch_id, ticker, json.dumps({
                    "minutes": get_lookback_minutes(),
                    "batch_size": 3,
                    "triage_batch_size": 3,
                    "mode": "daily",
                    "recipients": ticker_recipients[ticker]
                }), timeout_at))

            conn.commit()

        LOG.info(f"🔄 Re-run ALL queue: {len(tickers_list)} tickers (batch {batch_id})")

        # Build response with status breakdown
        status_counts = {}
        for row in all_rows:
            status = row['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "status": "success",
            "batch_id": str(batch_id),
            "ticker_count": len(ticker_recipients),
            "status_breakdown": status_counts,
            "message": f"Reprocessing ALL {len(ticker_recipients)} tickers from scratch. Processing time: ~30-60 minutes."
        }

    except Exception as e:
        LOG.error(f"Failed to rerun all queue: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/undo-cancel-ready-emails")
async def undo_cancel_ready_emails_api(request: Request):
    """Undo cancellation - restore ALL cancelled emails to their previous status (smart restore)"""
    body = await request.json()
    token = body.get('token')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            # Get affected emails for response
            cur.execute("""
                SELECT ticker, company_name, status, previous_status
                FROM email_queue
                WHERE status = 'cancelled'
            """)
            affected_emails = cur.fetchall()

            # Smart restore: Use previous_status if available, otherwise default to 'ready'
            cur.execute("""
                UPDATE email_queue
                SET status = COALESCE(previous_status, 'ready'),
                    previous_status = NULL,
                    updated_at = NOW()
                WHERE status = 'cancelled'
            """)
            restored_count = cur.rowcount
            conn.commit()

            LOG.info(f"✅ Restored {restored_count} cancelled emails to previous status")

            # Build response with restoration details
            restored_tickers = [
                {
                    "ticker": email['ticker'],
                    "company_name": email['company_name'],
                    "restored_to": email['previous_status'] or 'ready',
                    "was_cancelled": True
                }
                for email in affected_emails
            ]

            return {
                "status": "success",
                "restored_count": restored_count,
                "restored_tickers": restored_tickers,
                "message": f"Restored {restored_count} cancelled emails to previous status"
            }
    except Exception as e:
        LOG.error(f"Failed to restore cancelled emails: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/generate-user-reports")
async def generate_user_reports_api(request: Request):
    """Generate reports for selected users only (bulk processing) - uses existing job queue"""
    body = await request.json()
    token = body.get('token')
    user_emails = body.get('user_emails', [])

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    if not user_emails:
        return {"status": "error", "message": "No users selected"}

    try:
        # CRITICAL: Load CSV from GitHub BEFORE processing
        try:
            sync_ticker_references_from_github()
        except Exception as e:
            LOG.error(f"❌ CSV sync failed: {e} - continuing anyway")

        # Load tickers for selected users only and build ticker → recipients mapping
        ticker_recipients = {}

        with db() as conn, conn.cursor() as cur:
            # Use parameterized query to fetch selected users
            placeholders = ','.join(['%s'] * len(user_emails))
            cur.execute(f"""
                SELECT name, email, ticker1, ticker2, ticker3
                FROM beta_users
                WHERE email IN ({placeholders})
                AND status = 'active'
                ORDER BY created_at
            """, user_emails)
            users = cur.fetchall()

            if not users:
                return {
                    "status": "error",
                    "message": "No active users found with selected emails"
                }

            LOG.info(f"Found {len(users)} selected users")

            # Deduplicate tickers and build recipient mapping
            for user in users:
                email = user['email']
                tickers = [user['ticker1'], user['ticker2'], user['ticker3']]
                for ticker in tickers:
                    if ticker:
                        ticker = ticker.upper().strip()
                        if ticker not in ticker_recipients:
                            ticker_recipients[ticker] = []
                        if email not in ticker_recipients[ticker]:
                            ticker_recipients[ticker].append(email)

            LOG.info(f"Dedup complete: {len(ticker_recipients)} unique tickers from {len(users)} users")

        # Submit to existing job queue system
        tickers_list = sorted(list(ticker_recipients.keys()))

        with db() as conn, conn.cursor() as cur:
            # Create batch record
            cur.execute("""
                INSERT INTO ticker_processing_batches (total_jobs, created_by, config)
                VALUES (%s, %s, %s)
                RETURNING batch_id
            """, (len(tickers_list), 'admin_ui', json.dumps({
                "minutes": get_lookback_minutes(),
                "batch_size": 3,
                "triage_batch_size": 3,
                "mode": "daily"  # CRITICAL: Daily workflow mode
            })))

            batch_id = cur.fetchone()['batch_id']

            # Create individual jobs with mode='daily' and recipients
            timeout_at = datetime.now(timezone.utc) + timedelta(minutes=45)
            for ticker in tickers_list:
                cur.execute("""
                    INSERT INTO ticker_processing_jobs (
                        batch_id, ticker, config, timeout_at
                    )
                    VALUES (%s, %s, %s, %s)
                """, (batch_id, ticker, json.dumps({
                    "minutes": get_lookback_minutes(),
                    "batch_size": 3,
                    "triage_batch_size": 3,
                    "mode": "daily",  # CRITICAL: Daily workflow mode
                    "recipients": ticker_recipients[ticker]  # CRITICAL: Recipients for Email #3
                }), timeout_at))

            conn.commit()

        LOG.info(f"📊 Batch {batch_id} created for {len(users)} users: {len(tickers_list)} unique tickers (mode=daily)")

        return {
            "status": "success",
            "batch_id": str(batch_id),
            "user_count": len(users),
            "ticker_count": len(tickers_list),
            "tickers": tickers_list,
            "message": f"Processing {len(tickers_list)} unique tickers from {len(users)} selected users. Check back in 10-20 minutes."
        }

    except Exception as e:
        LOG.error(f"Failed to generate user reports: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/generate-all-reports")
async def generate_all_reports_api(request: Request):
    """Generate reports for ALL active users - uses existing job queue"""
    body = await request.json()
    token = body.get('token')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        # CRITICAL: Load CSV from GitHub BEFORE processing
        try:
            sync_ticker_references_from_github()
        except Exception as e:
            LOG.error(f"❌ CSV sync failed: {e} - continuing anyway")

        # Load all active beta users (same as process_daily_workflow)
        ticker_recipients = load_active_beta_users()

        if not ticker_recipients:
            return {
                "status": "error",
                "message": "No active beta users found"
            }

        # Submit to existing job queue system
        tickers_list = sorted(list(ticker_recipients.keys()))

        with db() as conn, conn.cursor() as cur:
            # Create batch record
            cur.execute("""
                INSERT INTO ticker_processing_batches (total_jobs, created_by, config)
                VALUES (%s, %s, %s)
                RETURNING batch_id
            """, (len(tickers_list), 'admin_ui', json.dumps({
                "minutes": get_lookback_minutes(),
                "batch_size": 3,
                "triage_batch_size": 3,
                "mode": "daily"  # CRITICAL: Daily workflow mode
            })))

            batch_id = cur.fetchone()['batch_id']

            # Create individual jobs with mode='daily' and recipients
            timeout_at = datetime.now(timezone.utc) + timedelta(minutes=45)
            for ticker in tickers_list:
                cur.execute("""
                    INSERT INTO ticker_processing_jobs (
                        batch_id, ticker, config, timeout_at
                    )
                    VALUES (%s, %s, %s, %s)
                """, (batch_id, ticker, json.dumps({
                    "minutes": get_lookback_minutes(),
                    "batch_size": 3,
                    "triage_batch_size": 3,
                    "mode": "daily",  # CRITICAL: Daily workflow mode
                    "recipients": ticker_recipients[ticker]  # CRITICAL: Recipients for Email #3
                }), timeout_at))

            conn.commit()

        LOG.info(f"📊 Batch {batch_id} created for all active users: {len(tickers_list)} unique tickers (mode=daily)")

        return {
            "status": "success",
            "batch_id": str(batch_id),
            "ticker_count": len(tickers_list),
            "tickers": tickers_list,
            "message": f"Processing {len(tickers_list)} unique tickers from all active users. This will take approximately 30-60 minutes."
        }

    except Exception as e:
        LOG.error(f"Failed to generate all reports: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/clear-all-reports")
async def clear_all_reports_api(request: Request):
    """Clear all queue entries (manual trigger equivalent to: python app.py cleanup)"""
    body = await request.json()
    token = body.get('token')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            # Delete all email queue entries
            cur.execute("DELETE FROM email_queue")
            deleted_count = cur.rowcount
            conn.commit()

        LOG.info(f"🗑️ Cleared all reports: {deleted_count} entries deleted")

        return {
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} queue entries"
        }

    except Exception as e:
        LOG.error(f"Failed to clear all reports: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/commit-ticker-csv")
async def commit_ticker_csv_api(request: Request):
    """
    Manually commit ticker_reference.csv to GitHub.
    Triggers Render deployment (~2-3 min downtime).
    Admin-only endpoint.
    """
    body = await request.json()
    token = body.get('token')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        LOG.info("🔄 Manual GitHub commit requested via admin dashboard")
        result = commit_ticker_reference_to_github()

        if result['status'] == 'success':
            LOG.info(f"✅ Manual commit successful: {result.get('commit_sha', 'N/A')[:8]}")
            return {
                "status": "success",
                "message": result['message'],
                "rows": result['rows'],
                "commit_sha": result.get('commit_sha'),
                "commit_url": result.get('commit_url'),
                "timestamp": result.get('timestamp')
            }
        else:
            LOG.error(f"❌ Manual commit failed: {result.get('message')}")
            return {
                "status": "error",
                "message": result.get('message', 'Unknown error'),
                "rows": result.get('rows', 0)
            }

    except Exception as e:
        LOG.error(f"Manual GitHub commit failed: {e}")
        LOG.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

@APP.get("/api/get-lookback-window")
async def get_lookback_window_api(token: str = Query(...)):
    """Get current production lookback window from system_config"""
    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        minutes = get_lookback_minutes()
        hours = minutes / 60
        days = int(hours / 24) if hours >= 24 else 0
        label = f"{days} days" if days > 0 else f"{int(hours)} hours"

        return {
            "status": "success",
            "minutes": minutes,
            "label": label
        }

    except Exception as e:
        LOG.error(f"Failed to get lookback window: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/set-lookback-window")
async def set_lookback_window_api(
    token: str = Query(...),
    minutes: int = Query(...)
):
    """Update production lookback window in system_config"""
    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    # Validation: 1 hour to 7 days
    if minutes < 60:
        return {
            "status": "error",
            "message": "Lookback must be at least 60 minutes (1 hour)"
        }

    if minutes > 10080:
        return {
            "status": "error",
            "message": "Lookback cannot exceed 10,080 minutes (7 days)"
        }

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE system_config
                SET value = %s,
                    updated_at = NOW(),
                    updated_by = 'admin_dashboard'
                WHERE key = 'lookback_minutes'
            """, (str(minutes),))
            conn.commit()

        hours = minutes / 60
        days = int(hours / 24) if hours >= 24 else 0
        label = f"{days} days" if days > 0 else f"{int(hours)} hours"

        LOG.info(f"✅ Production lookback window updated to {minutes} minutes ({label}) via admin dashboard")

        return {
            "status": "success",
            "message": f"Lookback window updated to {label}",
            "minutes": minutes,
            "label": label
        }

    except Exception as e:
        LOG.error(f"Failed to update lookback window: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/cancel-ready-emails")
async def cancel_ready_emails_api(request: Request):
    """Cancel ready emails - prevent 8:30am auto-send (tracks previous_status for smart restore)"""
    body = await request.json()
    token = body.get('token')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            # Get affected emails for response
            cur.execute("""
                SELECT ticker, company_name, recipients, status
                FROM email_queue
                WHERE status IN ('ready', 'processing')
            """)
            affected_emails = cur.fetchall()

            # Track previous_status before cancelling
            cur.execute("""
                UPDATE email_queue
                SET previous_status = status,
                    status = 'cancelled',
                    updated_at = NOW()
                WHERE status IN ('ready', 'processing')
            """)
            cancelled_count = cur.rowcount
            conn.commit()

            LOG.warning(f"⛔ CANCEL READY EMAILS: Cancelled {cancelled_count} emails (prevents 8:30am send)")

            # Build response with affected tickers
            affected_tickers = [
                {
                    "ticker": email['ticker'],
                    "company_name": email['company_name'],
                    "recipient_count": len(email['recipients']) if email['recipients'] else 0,
                    "previous_status": email['status']
                }
                for email in affected_emails
            ]

            return {
                "status": "success",
                "cancelled_count": cancelled_count,
                "affected_tickers": affected_tickers,
                "message": f"Cancelled {cancelled_count} ready emails (prevents 8:30am auto-send)"
            }
    except Exception as e:
        LOG.error(f"Cancel ready emails failed: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/cancel-in-progress-runs")
async def cancel_in_progress_runs_api(request: Request):
    """Cancel all in-progress ticker processing jobs (stops current runs)"""
    body = await request.json()
    token = body.get('token')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            # Get all active batches with their jobs
            cur.execute("""
                SELECT
                    b.batch_id,
                    b.created_by,
                    b.created_at
                FROM ticker_processing_batches b
                WHERE EXISTS (
                    SELECT 1 FROM ticker_processing_jobs j
                    WHERE j.batch_id = b.batch_id
                      AND j.status IN ('queued', 'processing')
                )
                ORDER BY b.created_at DESC
            """)
            active_batches = cur.fetchall()

            if not active_batches:
                return {
                    "status": "success",
                    "cancelled_jobs": 0,
                    "cancelled_batches": 0,
                    "message": "No active jobs to cancel"
                }

            # Get detailed job information for response
            all_jobs = []
            for batch in active_batches:
                cur.execute("""
                    SELECT job_id, ticker, status, phase, progress
                    FROM ticker_processing_jobs
                    WHERE batch_id = %s
                      AND status IN ('queued', 'processing')
                """, (batch['batch_id'],))
                jobs = cur.fetchall()
                all_jobs.extend([{
                    "batch_id": str(batch['batch_id']),
                    "batch_created_by": batch['created_by'],
                    "ticker": j['ticker'],
                    "status": j['status'],
                    "phase": j['phase'],
                    "progress": j['progress']
                } for j in jobs])

            # Cancel all active jobs across all batches
            total_cancelled = 0
            for batch in active_batches:
                cur.execute("""
                    UPDATE ticker_processing_jobs
                    SET status = 'cancelled',
                        error_message = 'Cancelled by admin via UI',
                        last_updated = NOW()
                    WHERE batch_id = %s
                      AND status IN ('queued', 'processing')
                """, (batch['batch_id'],))
                total_cancelled += cur.rowcount

            conn.commit()

            LOG.warning(f"🚫 CANCEL IN PROGRESS RUNS: Cancelled {total_cancelled} jobs across {len(active_batches)} batches")

            return {
                "status": "success",
                "cancelled_jobs": total_cancelled,
                "cancelled_batches": len(active_batches),
                "affected_jobs": all_jobs,
                "message": f"Cancelled {total_cancelled} in-progress jobs across {len(active_batches)} batches"
            }
    except Exception as e:
        LOG.error(f"Cancel in-progress runs failed: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/cancel-ticker")
async def cancel_ticker_api(request: Request):
    """Cancel individual ticker"""
    body = await request.json()
    token = body.get('token')
    ticker = body.get('ticker')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE email_queue
                SET status = 'cancelled', updated_at = NOW()
                WHERE ticker = %s AND status = 'ready'
            """, (ticker,))
            conn.commit()

            LOG.info(f"❌ Cancelled ticker: {ticker}")
            return {"status": "success", "ticker": ticker, "message": f"Cancelled {ticker}"}
    except Exception as e:
        LOG.error(f"Failed to cancel ticker: {e}")
        return {"status": "error", "message": str(e)}

@APP.post("/api/send-ticker")
async def send_ticker_api(request: Request):
    """Send individual ticker email"""
    body = await request.json()
    token = body.get('token')
    ticker = body.get('ticker')

    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    if not ticker:
        return {"status": "error", "message": "Ticker required"}

    try:
        with db() as conn, conn.cursor() as cur:
            # Get ready email for this ticker
            cur.execute("""
                SELECT ticker, company_name, recipients, email_html, email_subject, article_count
                FROM email_queue
                WHERE ticker = %s AND status = 'ready' AND sent_at IS NULL
            """, (ticker,))

            email = cur.fetchone()

            if not email:
                return {"status": "error", "message": f"No ready email found for {ticker}"}

            recipients = email['recipients']
            admin_email = os.getenv('ADMIN_EMAIL', 'stockdigest.research@gmail.com')

            # Send to each recipient with unique unsubscribe token
            for recipient in recipients:
                # Generate unique unsubscribe token for this recipient
                unsubscribe_token = get_or_create_unsubscribe_token(recipient)

                # Replace placeholder with full unsubscribe URL
                unsubscribe_url = f"https://stockdigest.app/unsubscribe?token={unsubscribe_token}" if unsubscribe_token else "https://stockdigest.app/unsubscribe"
                final_html = email['email_html'].replace(
                    '{{UNSUBSCRIBE_TOKEN}}',
                    unsubscribe_url
                )

                # Send email
                success = send_email_with_dry_run(
                    subject=email['email_subject'],
                    html=final_html,
                    to=recipient,
                    bcc=admin_email
                )

                if not success:
                    raise Exception(f"Failed to send to {recipient}")

                LOG.info(f"✅ Sent {ticker} to {recipient}")

            # Mark as sent
            cur.execute("""
                UPDATE email_queue
                SET status = 'sent', sent_at = NOW()
                WHERE ticker = %s
            """, (ticker,))
            conn.commit()

            LOG.info(f"✅ {ticker} sent to {len(recipients)} recipients")

            return {
                "status": "success",
                "ticker": ticker,
                "recipients_count": len(recipients),
                "message": f"Sent {ticker} to {len(recipients)} recipients"
            }

    except Exception as e:
        LOG.error(f"Failed to send ticker {ticker}: {e}")
        return {"status": "error", "message": str(e)}

@APP.get("/api/view-email/{ticker}")
def view_email_api(ticker: str, token: str = Query(...)):
    """View Email #3 (Final User Email) preview"""
    if not check_admin_token(token):
        return HTMLResponse("Unauthorized", status_code=401)

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT email_html
                FROM email_queue
                WHERE ticker = %s
            """, (ticker,))
            row = cur.fetchone()

            if not row or not row['email_html']:
                return HTMLResponse("Email not found", status_code=404)

            return HTMLResponse(row['email_html'])
    except Exception as e:
        LOG.error(f"Failed to view email: {e}")
        return HTMLResponse(f"Error: {str(e)}", status_code=500)

@APP.get("/api/view-email-1/{ticker}")
async def view_email_1_api(ticker: str, token: str = Query(...)):
    """View Email #1 (Article Selection QA) preview - regenerated on-demand from database"""
    if not check_admin_token(token):
        return HTMLResponse("Unauthorized", status_code=401)

    try:
        config = get_ticker_config(ticker)
        if not config:
            return HTMLResponse(f"<h1>Ticker {ticker} not found</h1>", status_code=404)

        hours = get_lookback_minutes() / 60
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Query ALL articles (not just flagged) to show triage results
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT
                    a.id, a.title, a.url, a.resolved_url, a.published_at, a.domain,
                    ta.category, ta.relevance_score, ta.category_score,
                    ta.search_keyword, ta.competitor_ticker
                FROM articles a
                JOIN ticker_articles ta ON a.id = ta.article_id
                WHERE ta.ticker = %s AND a.published_at >= %s
                ORDER BY a.published_at DESC
            """, (ticker, cutoff))
            articles = [dict(row) for row in cur.fetchall()]

        if not articles:
            return HTMLResponse(f"<h1>No articles for {ticker}</h1><p>Run processing first.</p>")

        # Count flagged vs total by category (flagged = relevance_score >= 7.0)
        stats = {"company": {"total": 0, "flagged": 0}, "industry": {"total": 0, "flagged": 0}, "competitor": {"total": 0, "flagged": 0}}
        for a in articles:
            cat = a['category']
            if cat in stats:
                stats[cat]["total"] += 1
                # Article is "flagged" if relevance_score >= 7.0
                if a.get('relevance_score') and a['relevance_score'] >= 7.0:
                    stats[cat]["flagged"] += 1
                    a['flagged'] = True  # Add flag for template
                else:
                    a['flagged'] = False

        # Generate simple HTML preview
        html = f"""
        <html><head><style>body{{font-family:sans-serif;padding:20px;}}table{{border-collapse:collapse;width:100%;}}th,td{{border:1px solid #ddd;padding:8px;text-align:left;}}th{{background:#1e40af;color:white;}}
        .flagged{{background:#d1fae5;}}
        .score{{font-weight:bold;color:#1e40af;}}</style></head>
        <body>
        <h1>📋 Email #1: Article Selection QA - {ticker}</h1>
        <p><strong>Time Period:</strong> Last {int(hours)} hours | <strong>Company:</strong> {config.get('name', ticker)}</p>
        <h2>Summary</h2>
        <ul>
        <li><strong>Company:</strong> {stats['company']['flagged']} flagged / {stats['company']['total']} total</li>
        <li><strong>Industry:</strong> {stats['industry']['flagged']} flagged / {stats['industry']['total']} total</li>
        <li><strong>Competitor:</strong> {stats['competitor']['flagged']} flagged / {stats['competitor']['total']} total</li>
        </ul>
        <h2>Articles (Flagged articles highlighted)</h2>
        <table>
        <tr><th>Category</th><th>Title</th><th>Domain</th><th>Relevance</th><th>Category Score</th><th>Published</th></tr>
        """

        for a in articles:
            row_class = 'class="flagged"' if a['flagged'] else ''
            pub_date = format_date_short(a['published_at']) if a['published_at'] else 'N/A'
            html += f"""
            <tr {row_class}>
                <td>{a['category']}</td>
                <td>{a['title'][:80]}...</td>
                <td>{a['domain']}</td>
                <td class="score">{a['relevance_score']}/10</td>
                <td class="score">{a['category_score']}/10</td>
                <td>{pub_date}</td>
            </tr>
            """

        html += "</table></body></html>"
        return HTMLResponse(html)

    except Exception as e:
        LOG.error(f"Failed to generate Email #1: {e}")
        return HTMLResponse(f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

@APP.get("/api/view-email-2/{ticker}")
async def view_email_2_api(ticker: str, token: str = Query(...)):
    """View Email #2 (Content QA) preview - regenerated on-demand"""
    if not check_admin_token(token):
        return HTMLResponse("Unauthorized", status_code=401)

    try:
        hours = get_lookback_minutes() / 60

        # Query FLAGGED articles with AI summaries (flagged = relevance_score >= 7.0)
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT
                    a.id, a.title, a.url, a.resolved_url, a.published_at, a.domain,
                    a.description, a.scraped_content, ta.ai_summary, ta.ai_model,
                    ta.category, ta.relevance_score, ta.search_keyword, ta.competitor_ticker
                FROM articles a
                JOIN ticker_articles ta ON a.id = ta.article_id
                WHERE ta.ticker = %s
                AND ta.relevance_score >= 7.0
                AND a.published_at >= NOW() - INTERVAL '%s hours'
                ORDER BY a.published_at DESC
            """, (ticker, hours))
            flagged = [dict(row) for row in cur.fetchall()]

        if not flagged:
            return HTMLResponse(f"<h1>No flagged articles for {ticker}</h1><p>Run processing first.</p>")

        # Use existing template
        flagged_ids = [a['id'] for a in flagged]
        articles_dict = await fetch_digest_articles_with_enhanced_content(
            hours=int(hours), tickers=[ticker], flagged_article_ids=flagged_ids
        )

        html = templates.get_template("email_template.html").render(
            articles_by_ticker=articles_dict,
            time_period=f"{int(hours)} hours",
            current_time=format_timestamp_est(datetime.now(timezone.utc))
        )
        return HTMLResponse(html)

    except Exception as e:
        LOG.error(f"Failed to generate Email #2: {e}")
        return HTMLResponse(f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

# ------------------------------------------------------------------------------
# DAILY WORKFLOW PROCESSING FUNCTIONS
# ------------------------------------------------------------------------------

def load_active_beta_users() -> Dict[str, List[str]]:
    """
    Load active beta users and deduplicate tickers.
    Returns: {ticker: [list of recipient emails]}
    """
    LOG.info("Loading active beta users...")

    ticker_recipients = {}

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT name, email, ticker1, ticker2, ticker3
                FROM beta_users
                WHERE status = 'active'
                ORDER BY created_at
            """)
            users = cur.fetchall()

            LOG.info(f"Found {len(users)} active beta users")

            for user in users:
                email = user['email']
                tickers = [user['ticker1'], user['ticker2'], user['ticker3']]

                for ticker in tickers:
                    if ticker:
                        ticker = ticker.upper().strip()
                        if ticker not in ticker_recipients:
                            ticker_recipients[ticker] = []

                        # Deduplicate emails
                        if email not in ticker_recipients[ticker]:
                            ticker_recipients[ticker].append(email)

            LOG.info(f"Dedup complete: {len(ticker_recipients)} unique tickers")
            for ticker, emails in ticker_recipients.items():
                LOG.info(f"  {ticker}: {len(emails)} recipients")

            return ticker_recipients

    except Exception as e:
        LOG.error(f"Failed to load beta users: {e}")
        return {}



def send_admin_notification(results: Dict):
    """Send admin notification after processing completes"""
    LOG.info("Sending admin notification...")

    total = results['total']
    succeeded = results['succeeded']
    failed = results['failed']

    # Get failed tickers
    failed_tickers = []
    for r in results['results']:
        if isinstance(r, dict) and r.get('status') == 'failed':
            failed_tickers.append({
                'ticker': r.get('ticker'),
                'error': r.get('error', 'unknown')
            })

    failed_list_html = ""
    if failed_tickers:
        for ft in failed_tickers:
            failed_list_html += f"<li>{ft['ticker']} - {ft['error']}</li>"
        failed_list_html = f"<ul>{failed_list_html}</ul>"
    else:
        failed_list_html = "<p>None</p>"

    admin_email = os.getenv('ADMIN_EMAIL', 'stockdigest.research@gmail.com')
    admin_token = os.getenv('ADMIN_TOKEN', '')
    dashboard_url = f"https://stockdigest.app/admin/queue?token={admin_token}"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); color: white; padding: 20px; border-radius: 8px; }}
            .stat {{ display: inline-block; margin: 10px 20px 10px 0; }}
            .stat-value {{ font-size: 32px; font-weight: bold; }}
            .stat-label {{ font-size: 12px; opacity: 0.8; }}
            .success {{ color: #10b981; }}
            .danger {{ color: #ef4444; }}
            .btn {{ display: inline-block; background: #1e40af; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Queue Ready - {datetime.now().strftime('%B %d, %Y')}</h1>
                <p>Processing completed at {datetime.now().strftime('%I:%M %p')} ET</p>
            </div>

            <div style="margin: 20px 0;">
                <div class="stat">
                    <div class="stat-value success">✅ {succeeded}</div>
                    <div class="stat-label">Ready</div>
                </div>
                <div class="stat">
                    <div class="stat-value danger">❌ {failed}</div>
                    <div class="stat-label">Failed</div>
                </div>
            </div>

            <a href="{dashboard_url}" class="btn">Review Dashboard →</a>

            <p style="margin-top: 20px; color: #6b7280; font-size: 14px;">
                Auto-send scheduled: 8:30 AM
            </p>

            <h3>Failed Tickers</h3>
            {failed_list_html}
        </div>
    </body>
    </html>
    """

    subject = f"[ADMIN] Queue Ready - {datetime.now().strftime('%B %d, %Y')}"
    send_email(subject, html, to=admin_email)
    LOG.info(f"Admin notification sent to {admin_email}")


def send_email_with_dry_run(subject: str, html: str, to, bcc=None) -> bool:
    """
    Email sending wrapper with DRY_RUN mode support.
    In DRY_RUN mode, all emails redirect to admin.
    """
    dry_run = os.getenv('DRY_RUN', 'false').lower() == 'true'
    admin_email = os.getenv('ADMIN_EMAIL', 'stockdigest.research@gmail.com')

    if dry_run:
        LOG.warning(f"🧪 DRY_RUN: Redirecting email to {admin_email}")
        LOG.warning(f"   Original TO: {to}")
        LOG.warning(f"   Original BCC: {bcc}")

        # Override recipients
        actual_to = admin_email
        actual_bcc = None
        subject = f"[DRY RUN] {subject}"
    else:
        actual_to = to
        actual_bcc = bcc

    return send_email(subject, html, to=actual_to, bcc=actual_bcc)


def send_all_ready_emails_impl() -> Dict:
    """
    Send all emails with status='ready' that haven't been sent yet.
    Replaces {{UNSUBSCRIBE_TOKEN}} placeholder with unique token per recipient.
    """
    LOG.info("=== SENDING ALL READY EMAILS ===")

    try:
        with db() as conn, conn.cursor() as cur:
            # Get all ready emails not yet sent
            cur.execute("""
                SELECT ticker, company_name, recipients, email_html, email_subject, article_count
                FROM email_queue
                WHERE status = 'ready'
                AND sent_at IS NULL
                AND is_production = TRUE
                ORDER BY ticker
            """)

            emails = cur.fetchall()

            if not emails:
                LOG.info("No emails to send")
                return {
                    'status': 'success',
                    'sent_count': 0,
                    'message': 'No emails ready to send'
                }

            sent_count = 0
            failed_tickers = []
            admin_email = os.getenv('ADMIN_EMAIL', 'stockdigest.research@gmail.com')

            for email in emails:
                ticker = email['ticker']
                recipients = email['recipients']

                try:
                    # Send to each recipient with unique unsubscribe token
                    for recipient in recipients:
                        # Generate unique unsubscribe token for this recipient
                        token = get_or_create_unsubscribe_token(recipient)

                        # Replace placeholder with full unsubscribe URL
                        unsubscribe_url = f"https://stockdigest.app/unsubscribe?token={token}" if token else "https://stockdigest.app/unsubscribe"
                        final_html = email['email_html'].replace(
                            '{{UNSUBSCRIBE_TOKEN}}',
                            unsubscribe_url
                        )

                        # Send email
                        success = send_email_with_dry_run(
                            subject=email['email_subject'],
                            html=final_html,
                            to=recipient,
                            bcc=admin_email
                        )

                        if not success:
                            raise Exception(f"Failed to send to {recipient}")

                        LOG.info(f"✅ Sent {ticker} to {recipient}")

                    # Mark as sent
                    cur.execute("""
                        UPDATE email_queue
                        SET status = 'sent', sent_at = NOW()
                        WHERE ticker = %s
                    """, (ticker,))
                    conn.commit()

                    sent_count += 1
                    LOG.info(f"✅ {ticker} sent to {len(recipients)} recipients")

                except Exception as e:
                    failed_tickers.append(ticker)
                    LOG.error(f"❌ Failed to send {ticker}: {e}")

            LOG.info(f"Send complete: {sent_count} sent, {len(failed_tickers)} failed")

            return {
                'status': 'success',
                'sent_count': sent_count,
                'failed_count': len(failed_tickers),
                'failed_tickers': failed_tickers
            }

    except Exception as e:
        LOG.error(f"Send all ready emails failed: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }


# ------------------------------------------------------------------------------
# CRON JOB HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def cleanup_old_queue_entries():
    """
    6:00 AM: Delete old email queue entries (safety measure).
    Prevents stale test emails from being sent.
    """
    LOG.info("🧹 Cleaning up old queue entries...")

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                DELETE FROM email_queue
                WHERE created_at < CURRENT_DATE
                OR (is_production = FALSE AND created_at < NOW() - INTERVAL '1 day')
            """)
            deleted = cur.rowcount
            conn.commit()

        LOG.info(f"✅ Cleanup complete: {deleted} old entries deleted")

    except Exception as e:
        LOG.error(f"❌ Cleanup failed: {e}")
        raise


def process_daily_workflow():
    """
    7:00 AM: Process all active beta users via job queue system.
    Generates emails and queues for 8:30 AM send.
    """
    LOG.info("="*80)
    LOG.info("=== DAILY WORKFLOW START ===")
    LOG.info("="*80)

    try:
        # CRITICAL: Load CSV from GitHub BEFORE processing (same as test workflow)
        try:
            sync_ticker_references_from_github()
        except Exception as e:
            LOG.error(f"❌ CSV sync failed: {e} - continuing anyway")

        # Load beta users
        ticker_recipients = load_active_beta_users()

        if not ticker_recipients:
            LOG.warning("No active beta users found")
            return

        # Submit to existing job queue system (same as admin UI buttons)
        tickers_list = sorted(list(ticker_recipients.keys()))

        with db() as conn, conn.cursor() as cur:
            # Create batch record
            cur.execute("""
                INSERT INTO ticker_processing_batches (total_jobs, created_by, config)
                VALUES (%s, %s, %s)
                RETURNING batch_id
            """, (len(tickers_list), 'cron_job', json.dumps({
                "minutes": get_lookback_minutes(),
                "batch_size": 3,
                "triage_batch_size": 3,
                "mode": "daily"  # CRITICAL: Daily workflow mode
            })))

            batch_id = cur.fetchone()['batch_id']

            # Create individual jobs with mode='daily' and recipients
            timeout_at = datetime.now(timezone.utc) + timedelta(minutes=45)
            for ticker in tickers_list:
                cur.execute("""
                    INSERT INTO ticker_processing_jobs (
                        batch_id, ticker, config, timeout_at
                    )
                    VALUES (%s, %s, %s, %s)
                """, (batch_id, ticker, json.dumps({
                    "minutes": get_lookback_minutes(),
                    "batch_size": 3,
                    "triage_batch_size": 3,
                    "mode": "daily",  # CRITICAL: Daily workflow mode
                    "recipients": ticker_recipients[ticker]  # CRITICAL: Recipients for Email #3
                }), timeout_at))

            conn.commit()

        LOG.info(f"✅ Batch {batch_id} created for {len(tickers_list)} unique tickers (mode=daily)")
        LOG.info(f"   Jobs will be processed by background worker")
        LOG.info("✅ Daily workflow complete (jobs submitted to queue)")

    except Exception as e:
        LOG.error(f"❌ Daily workflow failed: {e}")
        LOG.error(f"Traceback: {traceback.format_exc()}")
        raise


def auto_send_cron_job():
    """
    8:30 AM: Auto-send all ready emails to users.
    Only runs if admin hasn't manually sent already.
    """
    LOG.info("="*80)
    LOG.info("=== 8:30 AM AUTO-SEND ===")
    LOG.info("="*80)

    try:
        with db() as conn, conn.cursor() as cur:
            # Check if there are any ready emails not yet sent
            cur.execute("""
                SELECT COUNT(*) as count
                FROM email_queue
                WHERE status = 'ready'
                AND sent_at IS NULL
                AND is_production = TRUE
            """)

            count = cur.fetchone()['count']

            if count == 0:
                LOG.info("No emails to send (admin may have sent manually)")
                return

            LOG.info(f"Auto-sending {count} ready emails")

            # Use same function as manual send
            result = send_all_ready_emails_impl()

            # Generate stats report
            # generate_stats_report()  # TODO: Implement stats report

            LOG.info(f"✅ Auto-send complete: {result['sent_count']} sent")

    except Exception as e:
        LOG.error(f"❌ Auto-send failed: {e}")
        raise


def export_beta_users_csv():
    """
    11:59 PM: Export beta users to CSV for backup.
    Optionally commit to GitHub.
    """
    LOG.info("📄 Exporting beta users to CSV...")

    try:
        import csv
        from datetime import datetime

        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f'beta_users_{timestamp}.csv'

        # Create backups directory if needed
        os.makedirs('/tmp/backups', exist_ok=True)
        filepath = f'/tmp/backups/{filename}'

        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT name, email, ticker1, ticker2, ticker3, status,
                       created_at, terms_accepted_at, privacy_accepted_at
                FROM beta_users
                ORDER BY created_at DESC
            """)

            rows = cur.fetchall()

        # Write CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'name', 'email', 'ticker1', 'ticker2', 'ticker3',
                'status', 'created_at', 'terms_accepted_at', 'privacy_accepted_at'
            ])

            for row in rows:
                writer.writerow([
                    row['name'],
                    row['email'],
                    row['ticker1'],
                    row['ticker2'],
                    row['ticker3'],
                    row['status'],
                    row['created_at'],
                    row['terms_accepted_at'],
                    row['privacy_accepted_at']
                ])

        LOG.info(f"✅ CSV export complete: {len(rows)} users → {filepath}")

        # TODO: Optionally commit to GitHub
        # github_commit_csv(filepath, filename)

        return filepath

    except Exception as e:
        LOG.error(f"❌ CSV export failed: {e}")
        raise


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
    import sys

    # Check if we're running a cron function
    if len(sys.argv) > 1:
        func_name = sys.argv[1]

        if func_name == "cleanup":
            cleanup_old_queue_entries()
        elif func_name == "process":
            process_daily_workflow()
        elif func_name == "send":
            auto_send_cron_job()
        elif func_name == "export":
            export_beta_users_csv()
        elif func_name == "commit":
            # Daily GitHub commit (triggers deployment)
            result = commit_ticker_reference_to_github()
            if result['status'] == 'success':
                print(f"✅ {result['message']}")
                print(f"   Rows: {result['rows']}")
                print(f"   Commit: {result.get('commit_sha', 'N/A')[:8]}")
                print(f"   URL: {result.get('commit_url', 'N/A')}")
                sys.exit(0)
            else:
                print(f"❌ Error: {result['message']}")
                sys.exit(1)
        else:
            print(f"Unknown function: {func_name}")
            print("Available functions: cleanup, process, send, export, commit")
            sys.exit(1)
    else:
        # Normal server startup
        import uvicorn
        port = int(os.getenv("PORT", "10000"))
        uvicorn.run(APP, host="0.0.0.0", port=port)
