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

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

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

# Quality filtering keywords
SPAM_DOMAINS = {
    "marketbeat.com", "www.marketbeat.com",
    "newser.com", "www.newser.com",
    "khodrobank.com", "www.khodrobank.com",
    "Ø®ÙˆØ¯Ø±Ùˆ Ø¨Ø§Ù†Ú©"
}

QUALITY_DOMAINS = {
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "barrons.com", "cnbc.com", "marketwatch.com",
    "yahoo.com/finance", "finance.yahoo.com",
    "businesswire.com", "prnewswire.com", "globenewswire.com"
}

# ------------------------------------------------------------------------------
# OpenAI Integration for Dynamic Keyword Generation
# ------------------------------------------------------------------------------
def generate_ticker_metadata_with_ai(ticker: str) -> Dict[str, List[str]]:
    """Use OpenAI to generate industry keywords and competitors for a ticker"""
    if not OPENAI_API_KEY:
        LOG.error("OpenAI API key not configured")
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
    - Be specific rather than generic (e.g., "data center power" instead of just "technology")
    
    Return ONLY the JSON object, no additional text.
    """
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a financial analyst expert who provides accurate information about companies and their industries."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Clean up the response and parse JSON
        content = content.strip()
        # Remove any markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        metadata = json.loads(content)
        
        # Validate and limit the results
        return {
            "industry_keywords": metadata.get("industry_keywords", [])[:5],
            "competitors": metadata.get("competitors", [])[:5]
        }
        
    except Exception as e:
        LOG.error(f"OpenAI API error for ticker {ticker}: {e}")
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
    """Build feed URLs for different categories"""
    feeds = []
    
    # Company-specific feeds
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
        },
        {
            "url": f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
            "name": f"Yahoo Finance News: {ticker}",
            "category": "company"
        }
    ])
    
    # Industry feeds
    for keyword in keywords["industry"][:5]:
        # URL encode the keyword for safety
        keyword_encoded = requests.utils.quote(keyword)
        feeds.append({
            "url": f"https://news.google.com/rss/search?q={keyword_encoded}+when:7d&hl=en-US&gl=US&ceid=US:en",
            "name": f"Industry: {keyword}",
            "category": "industry"
        })
    
    # Competitor feeds
    for competitor in keywords["competitors"][:5]:
        # Clean and encode competitor name
        comp_clean = competitor.replace("(", "").replace(")", "")
        comp_encoded = requests.utils.quote(comp_clean)
        feeds.append({
            "url": f"https://news.google.com/rss/search?q={comp_encoded}+when:7d&hl=en-US&gl=US&ceid=US:en",
            "name": f"Competitor: {competitor}",
            "category": "competitor"
        })
    
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
def resolve_google_news_url(url: str) -> Tuple[str, str]:
    """Resolve Google News redirect URL to actual article URL"""
    try:
        # Extract actual URL from Google News redirect
        if "news.google.com" in url and "/articles/" in url:
            response = requests.get(url, timeout=10, allow_redirects=True)
            return response.url, urlparse(response.url).netloc
        
        # For direct Google redirect URLs
        if "google.com/url" in url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            if 'url' in params:
                actual_url = params['url'][0]
                return actual_url, urlparse(actual_url).netloc
            elif 'q' in params:
                actual_url = params['q'][0]
                return actual_url, urlparse(actual_url).netloc
        
        # Already a direct URL
        return url, urlparse(url).netloc
    except Exception as e:
        LOG.warning(f"Failed to resolve URL {url}: {e}")
        return url, urlparse(url).netloc

def calculate_quality_score(
    title: str, 
    domain: str, 
    ticker: str,
    description: str = "",
    category: str = "company",
    keywords: List[str] = None
) -> float:
    """Calculate article quality score with category weighting"""
    score = 50.0  # Base score
    
    # Check for spam
    spam_indicators = [
        "marketbeat", "newser", "khodrobank", "Ø®ÙˆØ¯Ø±Ùˆ Ø¨Ø§Ù†Ú©"
    ]
    
    content_to_check = f"{title} {domain} {description}".lower()
    if any(spam in content_to_check for spam in spam_indicators):
        return 0.0
    
    # Domain quality
    if domain in SPAM_DOMAINS:
        return 0.0
    
    if domain in QUALITY_DOMAINS:
        score += 30
    elif any(q in domain for q in ["reuters", "bloomberg", "wsj", "ft", "cnbc"]):
        score += 25
    
    # Category-specific scoring
    if category == "company":
        # Direct company news gets higher weight
        if ticker in (title or "").upper():
            score += 15
        if keywords and any(kw.lower() in content_to_check for kw in keywords[:2]):
            score += 10
    elif category == "industry":
        # Industry news moderate weight
        score += 5
        if keywords and any(kw.lower() in content_to_check for kw in keywords):
            score += 10
    elif category == "competitor":
        # Competitor news lower base weight but can be valuable
        score += 3
        if keywords and any(kw.lower() in content_to_check for kw in keywords):
            score += 7
    
    # Negative signals
    spam_keywords = ["sponsored", "advertisement", "promoted", "partner content"]
    if any(kw in (title or "").lower() for kw in spam_keywords):
        score -= 30
    
    # Length and substance
    if len(title or "") > 20:
        score += 5
    if len(description or "") > 50:
        score += 5
    
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
    """Process a single feed and store articles with category"""
    stats = {"processed": 0, "inserted": 0, "duplicates": 0, "low_quality": 0}
    
    try:
        parsed = feedparser.parse(feed["url"])
        LOG.info(f"Processing {feed['name']}: {len(parsed.entries)} entries")
        
        for entry in parsed.entries:
            stats["processed"] += 1
            
            original_url = getattr(entry, "link", None)
            if not original_url:
                continue
            
            resolved_url, domain = resolve_google_news_url(original_url)
            url_hash = get_url_hash(resolved_url)
            
            title = getattr(entry, "title", "") or "No Title"
            description = getattr(entry, "summary", "")[:500] if hasattr(entry, "summary") else ""
            
            # Calculate quality score with category context
            quality_score = calculate_quality_score(
                title, domain, feed["ticker"], description, category, keywords
            )
            
            if quality_score < 20:
                stats["low_quality"] += 1
                LOG.debug(f"Skipping low quality: {title} (score: {quality_score})")
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
                        if re.match(r'^[A-Z]{2,5}$', comp_name):
                            related_ticker = comp_name
                    
                    cur.execute("""
                        INSERT INTO found_url (
                            url, resolved_url, url_hash, title, description,
                            feed_id, ticker, domain, quality_score, published_at,
                            category, related_ticker
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        original_url, resolved_url, url_hash, title, description,
                        feed["id"], feed["ticker"], domain, quality_score, published_at,
                        category, related_ticker
                    ))
                    
                    if cur.fetchone():
                        stats["inserted"] += 1
                        LOG.debug(f"Inserted [{category}]: {title[:50]}...")
                        
            except Exception as e:
                LOG.error(f"Database insert error for '{title[:50]}': {e}")
                continue
                
    except Exception as e:
        LOG.error(f"Feed processing error for {feed['name']}: {e}")
    
    return stats

# ------------------------------------------------------------------------------
# Email Digest
# ------------------------------------------------------------------------------
def build_digest_html(articles_by_ticker: Dict[str, Dict[str, List[Dict]]], period_days: int) -> str:
    """Build HTML email digest with categorized sections"""
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
        "a { color: #2980b9; text-decoration: none; }",
        "a:hover { text-decoration: underline; }",
        ".summary { margin-top: 20px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }",
        ".ticker-section { margin-bottom: 40px; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "</style></head><body>",
        f"<h1>📊 Quantbrief Stock Intelligence Report</h1>",
        f"<div class='summary'>",
        f"<strong>Report Period:</strong> Last {period_days} days<br>",
        f"<strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC<br>",
        f"<strong>Tickers Covered:</strong> {', '.join(articles_by_ticker.keys())}",
        "</div>"
    ]
    
    for ticker, categories in articles_by_ticker.items():
        total_articles = sum(len(articles) for articles in categories.values())
        
        html.append(f"<div class='ticker-section'>")
        html.append(f"<h2>📈 {ticker} - {total_articles} Total Articles</h2>")
        
        # Company News Section
        if "company" in categories and categories["company"]:
            html.append(f"<h3>🏢 Company News ({len(categories['company'])} articles)</h3>")
            for article in categories["company"][:30]:
                html.append(_format_article_html(article, "company"))
        
        # Industry News Section
        if "industry" in categories and categories["industry"]:
            html.append(f"<h3>🏭 Industry & Market News ({len(categories['industry'])} articles)</h3>")
            for article in categories["industry"][:20]:
                html.append(_format_article_html(article, "industry"))
        
        # Competitor News Section
        if "competitor" in categories and categories["competitor"]:
            html.append(f"<h3>🎯 Competitor Intelligence ({len(categories['competitor'])} articles)</h3>")
            for article in categories["competitor"][:20]:
                html.append(_format_article_html(article, "competitor"))
        
        html.append("</div>")
    
    html.append("""
        <div style='margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; font-size: 11px; color: #6c757d;'>
            <strong>About This Report:</strong><br>
            • <span style='color: #27ae60;'>Company News</span>: Direct mentions and updates about the ticker<br>
            • <span style='color: #f39c12;'>Industry News</span>: Sector trends and market dynamics<br>
            • <span style='color: #e74c3c;'>Competitor Intelligence</span>: Updates on key competitors<br>
            • Quality scores: Higher scores indicate more reputable sources and relevant content (20-100 scale)<br>
            • Keywords generated by AI analysis for comprehensive coverage
        </div>
        </body></html>
    """)
    
    return "".join(html)

def _format_article_html(article: Dict, category: str) -> str:
    """Format a single article for HTML display"""
    pub_date = article["published_at"].strftime("%m/%d %H:%M") if article["published_at"] else "N/A"
    
    title = article["title"] or "No Title"
    # Clean up title
    suffixes_to_remove = [
        " - MarketBeat", " - Newser", " - TipRanks", " - MSN", 
        " - The Daily Item", " - MarketScreener", " - Seeking Alpha",
        " - simplywall.st", " - Investopedia", " - Ø®ÙˆØ¯Ø±Ùˆ Ø¨Ø§Ù†Ú©"
    ]
    
    for suffix in suffixes_to_remove:
        if title.endswith(suffix):
            title = title[:-len(suffix)].strip()
    
    title = re.sub(r'\s*\$[A-Z]+\s*-?\s*', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    
    domain = article["domain"] or "unknown"
    domain = domain.replace("www.", "")
    
    score = article["quality_score"]
    score_class = "high-score" if score >= 70 else "med-score" if score >= 40 else "low-score"
    
    related = f" | Related: {article.get('related_ticker', '')}" if article.get('related_ticker') else ""
    
    return f"""
    <div class='article {category}'>
        <a href='{article["resolved_url"] or article["url"]}' target='_blank'>{title}</a>
        <span class='meta'> | {domain} | {pub_date}</span>
        <span class='score {score_class}'>Score: {score:.0f}</span>
        {related}
    </div>
    """

def fetch_digest_articles(hours: int = 24, tickers: List[str] = None) -> Dict[str, Dict[str, List[Dict]]]:
    """Fetch categorized articles for digest"""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    with db() as conn, conn.cursor() as cur:
        # Build query based on tickers
        if tickers:
            cur.execute("""
                SELECT 
                    f.url, f.resolved_url, f.title, f.description,
                    f.ticker, f.domain, f.quality_score, f.published_at,
                    f.found_at, f.category, f.related_ticker
                FROM found_url f
                WHERE f.found_at >= %s
                    AND f.quality_score >= 20
                    AND NOT f.sent_in_digest
                    AND f.ticker = ANY(%s)
                ORDER BY f.ticker, f.category, f.quality_score DESC, f.published_at DESC
            """, (
