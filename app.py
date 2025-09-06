# app.py
# QuantBrief Daily - minimal, robust FastAPI service
# - Ingests Google News RSS for tickers/companies in your watchlist
# - Resolves Google News redirect to the publisher URL
# - Filters low-quality sources (MarketBeat/Newser) before insert
# - Summarizes with OpenAI (chat.completions) using OPENAI_MODEL
# - Emails a clean, linked digest via Mailgun
#
# Env needed:
#   DATABASE_URL
#   OPENAI_API_KEY
#   OPENAI_MODEL               (e.g., gpt-5-mini)
#   OPENAI_MAX_OUTPUT_TOKENS   (e.g., 240)
#   MAILGUN_API_KEY
#   MAILGUN_DOMAIN
#   MAILGUN_FROM               (e.g., "QuantBrief Daily <daily@mg.example.com>")
#   ADMIN_TOKEN
#   OWNER_EMAIL                (optional: seeded as default recipient if present)
#
# Endpoints:
#   GET  /admin/health
#   GET  /admin/test_openai
#   POST /admin/init
#   POST /cron/ingest?minutes=1440
#   POST /cron/digest?minutes=1440
#
# Notes:
# - Requires requirements from previous step (fastapi, uvicorn, psycopg, openai==1.51.0, requests, feedparser, beautifulsoup4, trafilatura, lxml<5.2.0)
# - No 'proxies' passed to OpenAI client (avoids TypeError in SDK).

from __future__ import annotations

import os
import re
import time
import hmac
import json
import html
import hashlib
import logging
from typing import Iterable, Tuple, Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, quote_plus

import requests
import feedparser
from bs4 import BeautifulSoup

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, Request, HTTPException, Depends

from openai import OpenAI

# -----------------------------------------------------------------------------
# Config & Globals
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("quantbrief")

DATABASE_URL = os.environ.get("DATABASE_URL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
OPENAI_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "240"))
MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = os.environ.get("MAILGUN_DOMAIN", "")
MAILGUN_FROM = os.environ.get("MAILGUN_FROM", "QuantBrief Daily <daily@example.com>")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
OWNER_EMAIL = os.environ.get("OWNER_EMAIL")  # optional

# Important: do NOT pass proxies to the OpenAI client; some environments error out.
_openai = OpenAI(api_key=OPENAI_API_KEY)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

BANNED_HOSTS = {
    "news.google.com",
    "marketbeat.com", "www.marketbeat.com",
    "newser.com", "www.newser.com",
}

app = FastAPI(title="QuantBrief Daily")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def hostname_of(u: str) -> str:
    try:
        return (urlparse(u).hostname or "").lower()
    except Exception:
        return ""

def http_get(url: str, timeout: int = 15) -> Optional[requests.Response]:
    try:
        return requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except Exception as e:
        log.warning("http_get failed: %s", e)
        return None

def resolve_google_news_url(url: str) -> Optional[str]:
    """Follow Google News redirect to the original publisher. Returns final URL or None."""
    try:
        if "news.google.com" not in url:
            return url
        r = requests.get(url, allow_redirects=True, timeout=12, headers={"User-Agent": USER_AGENT})
        final = r.url.strip()
        # Some GN articles require second hit due to meta-refresh; do a quick follow once more.
        if "news.google.com" in hostname_of(final):
            r2 = requests.get(final, allow_redirects=True, timeout=10, headers={"User-Agent": USER_AGENT})
            final = r2.url.strip()
        if "news.google.com" in hostname_of(final):
            return None
        return final
    except Exception as e:
        log.warning("resolve_google_news_url error: %s", e)
        return None

def is_banned_link(url: str) -> bool:
    h = hostname_of(url)
    return h in BANNED_HOSTS

def extract_main_text(html_text: str) -> str:
    """Very light HTML -> text cleaner."""
    soup = BeautifulSoup(html_text or "", "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = " ".join((soup.get_text(" ") or "").split())
    # truncate to keep input small for summarization
    return text[:8000]

def fetch_title_and_publisher(final_url: str) -> Tuple[str, str, Optional[str]]:
    """Fetch page to get <title> and best-effort publisher; return (title, publisher, html)."""
    r = http_get(final_url, timeout=15)
    if not r or r.status_code >= 400:
        return "", hostname_of(final_url), None
    html_doc = r.text
    soup = BeautifulSoup(html_doc, "lxml")
    # Title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
    if not title:
        title = final_url

    # Publisher guess
    publisher = ""
    og_site = soup.find("meta", property="og:site_name")
    if og_site and og_site.get("content"):
        publisher = og_site["content"].strip()
    if not publisher:
        publisher = hostname_of(final_url)

    return title, publisher, html_doc

def summarize_text(title: str, page_text: str) -> str:
    """Short summary using chat.completions. No temperature override (gpt-5-mini defaults only)."""
    try:
        prompt = (
            "Summarize the following news for a professional investor. "
            "One concise sentence; avoid hype; include concrete facts if present.\n\n"
            f"TITLE: {title}\n\n"
            f"TEXT: {page_text[:4000]}"
        )
        resp = _openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You summarize financial news crisply for experts."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=min(OPENAI_MAX_OUTPUT_TOKENS, 300),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning("summarize_text error: %s", e)
        return ""

def render_digest_html(window_minutes: int, window_end_iso: str, rows: List[Dict[str, Any]]) -> str:
    items_html = []
    for a in rows:
        t = html.escape((a.get("title") or "").strip())
        p = html.escape((a.get("publisher") or "").strip())
        u = html.escape((a.get("url") or "").strip())
        s = html.escape((a.get("summary") or "").strip())
        if not t or not u:
            continue
        link_text = f"{t} — {p}" if p else t
        items_html.append(
            f'<li style="margin:10px 0">'
            f'<a href="{u}" style="font-weight:600; text-decoration:none; color:#1a73e8">{link_text}</a>'
            f'{f"<div style=\'color:#555; margin-top:4px\'>- {s}</div>" if s else ""}'
            f"</li>"
        )
    header_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    body = (
        f'<h1 style="margin:0 0 6px 0">QuantBrief Daily — {header_ts}</h1>'
        f'<div style="color:#666; margin:0 0 18px 0">'
        f'Window: last {window_minutes} minutes ending {window_end_iso}'
        f"</div>"
        f"<ul style='padding-left:20px; margin:0'>{''.join(items_html) or '<li>No items</li>'}</ul>"
        f"<p style='color:#777; margin-top:18px'>Sources are links only. We don’t republish paywalled content.</p>"
    )
    return body

def send_mailgun(recipients: List[str], subject: str, html_body: str) -> Tuple[bool, str]:
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN and MAILGUN_FROM):
        return False, "Mailgun env missing"
    try:
        resp = requests.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={
                "from": MAILGUN_FROM,
                "to": recipients,
                "subject": subject,
                "html": html_body,
            },
            timeout=15,
        )
        if resp.status_code >= 400:
            return False, f"{resp.status_code}: {resp.text}"
        return True, "OK"
    except Exception as e:
        return False, str(e)

# -----------------------------------------------------------------------------
# DB Schema init / migrations
# -----------------------------------------------------------------------------

INIT_SQL = """
-- core tables

CREATE TABLE IF NOT EXISTS public.article (
  id            BIGSERIAL PRIMARY KEY,
  url           TEXT NOT NULL,
  canonical_url TEXT,
  title         TEXT,
  publisher     TEXT,
  published_at  TIMESTAMPTZ,
  sha256        CHAR(64) NOT NULL,
  raw_html      TEXT,
  clean_text    TEXT,
  created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_article_sha256_unique ON public.article(sha256);
CREATE UNIQUE INDEX IF NOT EXISTS idx_article_url_unique ON public.article(url);

CREATE TABLE IF NOT EXISTS public.article_nlp (
  id         BIGSERIAL PRIMARY KEY,
  article_id BIGINT NOT NULL REFERENCES public.article(id) ON DELETE CASCADE,
  model      TEXT NOT NULL,
  summary    TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_article_nlp_article ON public.article_nlp(article_id);

CREATE TABLE IF NOT EXISTS public.recipients (
  id         BIGSERIAL PRIMARY KEY,
  email      TEXT NOT NULL UNIQUE,
  enabled    BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.source_feed (
  id              BIGSERIAL PRIMARY KEY,
  kind            TEXT NOT NULL DEFAULT 'rss',
  url             TEXT NOT NULL UNIQUE,
  active          BOOLEAN NOT NULL DEFAULT TRUE,
  name            TEXT,
  period_minutes  INTEGER NOT NULL DEFAULT 60,
  last_checked_at TIMESTAMPTZ NULL
);

CREATE TABLE IF NOT EXISTS public.watchlist (
  id         BIGSERIAL PRIMARY KEY,
  symbol     TEXT UNIQUE,
  company    TEXT,
  enabled    BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Keep article_tag loose to avoid FK pain; only use when present. Defaults allow inserts.
CREATE TABLE IF NOT EXISTS public.article_tag (
  article_id   BIGINT NOT NULL REFERENCES public.article(id) ON DELETE CASCADE,
  equity_id    BIGINT NULL,
  watchlist_id BIGINT NULL,
  tag_kind     SMALLINT NOT NULL DEFAULT 1,
  PRIMARY KEY (article_id, tag_kind)
);

-- Delivery logs
CREATE TABLE IF NOT EXISTS public.delivery_log (
  id             BIGSERIAL PRIMARY KEY,
  run_date       DATE NOT NULL,
  run_started_at TIMESTAMPTZ,
  run_ended_at   TIMESTAMPTZ,
  sent_to        TEXT,           -- comma list
  items          INTEGER NOT NULL DEFAULT 0,
  summarized     INTEGER NOT NULL DEFAULT 0,
  created_at     TIMESTAMPTZ DEFAULT now()
);

-- helpful indexes
CREATE INDEX IF NOT EXISTS idx_article_published ON public.article(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_article_publisher ON public.article(publisher);
"""

def migrate(conn):
    conn.execute(INIT_SQL)

    # ensure columns exist across upgrades
    conn.execute("""
        ALTER TABLE public.recipients
            ADD COLUMN IF NOT EXISTS enabled BOOLEAN NOT NULL DEFAULT TRUE;
    """)
    conn.execute("""
        ALTER TABLE public.article_tag
            ADD COLUMN IF NOT EXISTS watchlist_id BIGINT NULL,
            ADD COLUMN IF NOT EXISTS tag_kind SMALLINT NOT NULL DEFAULT 1;
    """)
    conn.execute("""
        UPDATE public.article_tag SET tag_kind = 1 WHERE tag_kind IS NULL;
    """)

    # basic seed: recipient + watchlist + google feed
    if OWNER_EMAIL:
        conn.execute(
            "INSERT INTO public.recipients(email, enabled) VALUES (%s, TRUE) ON CONFLICT (email) DO UPDATE SET enabled=EXCLUDED.enabled",
            (OWNER_EMAIL,),
        )
    conn.execute(
        "INSERT INTO public.recipients(email, enabled) VALUES (%s, TRUE) ON CONFLICT (email) DO NOTHING",
        ("you@example.com",),
    )
    # seed a watchlist if empty
    rows = conn.execute("SELECT COUNT(*) AS c FROM public.watchlist").fetchone()
    if rows and rows["c"] == 0:
        conn.execute(
            "INSERT INTO public.watchlist(symbol, company, enabled) VALUES (%s,%s,TRUE)",
            ("TLN", "Talen Energy"),
        )

    # seed a google news feed for TLN if not present
    gn_url = "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN&hl=en-US&gl=US&ceid=US:en"
    conn.execute("""
        INSERT INTO public.source_feed(kind, url, active, name, period_minutes)
        VALUES ('rss', %s, TRUE, 'Google News: Talen Energy', 60)
        ON CONFLICT (url) DO UPDATE
        SET kind=EXCLUDED.kind, active=EXCLUDED.active, name=EXCLUDED.name, period_minutes=EXCLUDED.period_minutes
    """, (gn_url,))

# -----------------------------------------------------------------------------
# Ingest
# -----------------------------------------------------------------------------

def load_watch_terms(conn) -> List[Tuple[int, str, str]]:
    res = conn.execute("SELECT id, symbol, company FROM public.watchlist WHERE enabled = TRUE").fetchall()
    return [(r["id"], (r["symbol"] or "").strip(), (r["company"] or "").strip()) for r in res]

def active_rss_feeds(conn) -> List[Dict[str, Any]]:
    res = conn.execute("SELECT id, url, name FROM public.source_feed WHERE active = TRUE AND kind = 'rss'").fetchall()
    return list(res)

def gen_queries_from_watchlist(terms: List[Tuple[int, str, str]]) -> List[str]:
    # If dedicated feeds exist in source_feed we use them; otherwise create inline GN queries.
    queries = []
    for _, sym, comp in terms:
        qparts = []
        if comp:
            qparts.append(f'("{comp}")')
        if sym:
            qparts.append(sym)
        if not qparts:
            continue
        q = " OR ".join(qparts)
        url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"
        queries.append(url)
    return queries

def ingest_google_news(conn, minutes: int) -> Tuple[int, int]:
    watch_terms = load_watch_terms(conn)
    feeds = active_rss_feeds(conn)
    if not feeds:
        # derive Google News feeds from watchlist
        feeds = [{"id": None, "url": u, "name": "Google News"} for u in gen_queries_from_watchlist(watch_terms)]

    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(minutes=minutes)

    found = 0
    inserted = 0

    for f in feeds:
        url = f["url"]
        log.info("Parsing feed: %s", url)
        fp = feedparser.parse(url)
        for entry in fp.entries:
            raw_link = (getattr(entry, "link", "") or "").strip()
            title = (getattr(entry, "title", "") or "").strip()

            # Published time
            published_at = None
            try:
                if getattr(entry, "published_parsed", None):
                    published_at = datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)
            except Exception:
                published_at = None

            # skip outside window
            if published_at and (published_at < window_start or published_at > window_end):
                continue

            # Resolve to publisher
            final_url = resolve_google_news_url(raw_link)
            if not final_url:
                continue
            if is_banned_link(final_url):
                continue

            # Fetch Title/Publisher if the feed's title is generic
            pub_title, publisher, html_doc = fetch_title_and_publisher(final_url)
            if title.lower() == "google news" or not title:
                title = pub_title or title or final_url

            # derive text for summary
            clean_text = ""
            if html_doc:
                clean_text = extract_main_text(html_doc)

            # Insert article (dedupe by sha256(url))
            art_sha = sha256_hex(final_url)
            try:
                row = conn.execute("""
                    INSERT INTO public.article (url, canonical_url, title, publisher, published_at, sha256, raw_html, clean_text)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (sha256) DO NOTHING
                    RETURNING id
                """, (final_url, final_url, title, publisher, published_at, art_sha, html_doc, clean_text)).fetchone()
            except Exception as e:
                log.warning("article insert failed: %s", e)
                row = None

            found += 1
            if not row:
                continue  # duplicate
            article_id = row["id"]
            inserted += 1

            # Summarize and store
            summary = summarize_text(title, clean_text or title)
            if summary:
                try:
                    conn.execute("""
                        INSERT INTO public.article_nlp (article_id, model, summary) VALUES (%s,%s,%s)
                        ON CONFLICT DO NOTHING
                    """, (article_id, OPENAI_MODEL, summary))
                except Exception as e:
                    log.warning("Summarize failed for article %s: %s", article_id, e)

    return found, inserted

# -----------------------------------------------------------------------------
# Digest
# -----------------------------------------------------------------------------

def build_items_for_digest(conn, minutes: int) -> List[Dict[str, Any]]:
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(minutes=minutes)
    rows = conn.execute("""
        SELECT a.id, a.title, a.publisher, a.url,
               COALESCE(n.summary, '') AS summary,
               a.published_at
        FROM public.article a
        LEFT JOIN public.article_nlp n ON n.article_id = a.id
        WHERE (a.published_at IS NULL OR (a.published_at >= %s AND a.published_at <= %s))
        ORDER BY COALESCE(a.published_at, a.created_at) DESC, a.id DESC
        LIMIT 200
    """, (window_start, window_end)).fetchall()
    return list(rows)

def send_digest(conn, minutes: int) -> Tuple[bool, str, int, int]:
    recipients = [r["email"] for r in conn.execute("SELECT email FROM public.recipients WHERE enabled = TRUE ORDER BY id").fetchall()]
    if not recipients:
        return False, "No recipients enabled", 0, 0

    items = build_items_for_digest(conn, minutes=minutes)
    window_end = datetime.now(timezone.utc)
    html_body = render_digest_html(minutes, window_end.strftime("%Y-%m-%d %H:%M"), items)

    subject = f"QuantBrief Daily — {window_end.strftime('%Y-%m-%d %H:%M')}"
    ok, msg = send_mailgun(recipients, subject, html_body)

    # log
    try:
        conn.execute("""
            INSERT INTO public.delivery_log (run_date, run_started_at, run_ended_at, sent_to, items, summarized)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (window_end.date(), window_end, window_end, ",".join(recipients), len(items), sum(1 for r in items if r.get("summary"))))
    except Exception as e:
        log.warning("delivery_log insert failed: %s", e)

    return ok, msg, len(items), sum(1 for r in items if r.get("summary"))

# -----------------------------------------------------------------------------
# Security
# -----------------------------------------------------------------------------

def require_admin(request: Request):
    token = request.headers.get("x-admin-token")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/admin/health")
def admin_health():
    try:
        with get_conn() as conn:
            counts = {
                "articles": conn.execute("SELECT COUNT(*) c FROM public.article").fetchone()["c"],
                "summarized": conn.execute("SELECT COUNT(*) c FROM public.article_nlp").fetchone()["c"],
                "recipients": conn.execute("SELECT COUNT(*) c FROM public.recipients").fetchone()["c"],
                "watchlist": conn.execute("SELECT COUNT(*) c FROM public.watchlist").fetchone()["c"],
            }
    except Exception:
        counts = {}

    env = {
        "DATABASE_URL": bool(DATABASE_URL),
        "OPENAI_API_KEY": bool(OPENAI_API_KEY),
        "OPENAI_MODEL": bool(OPENAI_MODEL),
        "OPENAI_MAX_OUTPUT_TOKENS": bool(OPENAI_MAX_OUTPUT_TOKENS),
        "MAILGUN_API_KEY": bool(MAILGUN_API_KEY),
        "MAILGUN_DOMAIN": bool(MAILGUN_DOMAIN),
        "MAILGUN_FROM": bool(MAILGUN_FROM),
        "OWNER_EMAIL": bool(OWNER_EMAIL),
        "ADMIN_TOKEN": bool(ADMIN_TOKEN),
    }
    return {"env": env, "counts": counts}

@app.get("/admin/test_openai")
def admin_test_openai(_: bool = Depends(require_admin)):
    try:
        resp = _openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You reply with a single short word."},
                {"role": "user", "content": "Say OK"},
            ],
            max_tokens=5,
        )
        sample = (resp.choices[0].message.content or "").strip()
        return {"ok": True, "model": OPENAI_MODEL, "sample": sample}
    except Exception as e:
        return {"ok": False, "model": OPENAI_MODEL, "err": str(e)}

@app.post("/admin/init")
def admin_init(_: bool = Depends(require_admin)):
    with get_conn() as conn:
        migrate(conn)
    return {"ok": True, "msg": "Initialized."}

@app.post("/cron/ingest")
def cron_ingest(request: Request, _: bool = Depends(require_admin)):
    minutes = int(request.query_params.get("minutes", "1440"))
    with get_conn() as conn:
        found, ins = ingest_google_news(conn, minutes=minutes)
    return {"ok": True, "found_urls": found, "inserted": ins}

@app.post("/cron/digest")
def cron_digest(request: Request, _: bool = Depends(require_admin)):
    minutes = int(request.query_params.get("minutes", "1440"))
    with get_conn() as conn:
        ok, msg, total, summarized = send_digest(conn, minutes=minutes)
    status = 200 if ok else 500
    return {"ok": ok, "sent_to": "see logs", "items": total, "summarized": summarized, "msg": msg}
