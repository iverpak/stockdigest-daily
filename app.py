import os
import hashlib
import re
import json
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs, urlunparse

import requests
import feedparser
import trafilatura

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from jinja2 import Template

import psycopg
from psycopg.rows import dict_row

# -----------------------------
# Environment & clients
# -----------------------------

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "512"))
MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = os.getenv("MAILGUN_DOMAIN", "")
MAILGUN_FROM = os.getenv("MAILGUN_FROM", "QuantBrief Daily <daily@mg.example.com>")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

DATABASE_URL = os.getenv("DATABASE_URL", "")

def _ensure_sslmode(url: str) -> str:
    # Render Postgres often needs sslmode=require
    if "sslmode=" in url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}sslmode=require"

DB_URL = _ensure_sslmode(DATABASE_URL)

# OpenAI: use the Chat Completions API (works with openai==1.40.0)
from openai import OpenAI
_openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="QuantBrief Daily")

# -----------------------------
# Helpers
# -----------------------------

def admin_guard(req: Request):
    hdr = req.headers.get("x-admin-token", "")
    if not ADMIN_TOKEN or hdr != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

def now_utc():
    return datetime.now(timezone.utc)

def sha256_of(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def resolve_google_news_url(url: str) -> str:
    """
    For Google News RSS links, prefer the original publisher URL.
    Many entries look like ...news.google.com/...&url=https%3A%2F%2Foriginal.com%2Fstory
    """
    try:
        p = urlparse(url)
        if p.netloc.endswith("news.google.com"):
            qs = parse_qs(p.query)
            if "url" in qs and qs["url"]:
                return qs["url"][0]
        return url
    except Exception:
        return url

BAD_HOSTS = (
    "lh3.googleusercontent.com",
    "gstatic.com",
    "fonts.gstatic.com",
)
BAD_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")

def is_probably_junk(url: str) -> bool:
    try:
        p = urlparse(url)
        host = p.netloc.lower()
        path = p.path.lower()
        if host.endswith(BAD_HOSTS):
            return True
        if any(path.endswith(ext) for ext in BAD_EXTS):
            return True
        return False
    except Exception:
        return True

def extract_text(url: str) -> str | None:
    """
    Use trafilatura to fetch and extract article text.
    """
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=False, timeout=20)
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_recall=True,
        )
        if text:
            # squeeze multiple blank lines
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text or None
    except Exception:
        return None

def summarize_text(text: str) -> str:
    """
    Summarize with OpenAI. Avoid passing params that gpt-5-mini rejects.
    We do NOT set temperature/top_p/max_tokens to keep compatibility.
    """
    if not text:
        return ""

    # Keep prompt small; model output length is constrained by its defaults.
    head = text.strip()
    if len(head) > 4000:
        head = head[:4000]  # trim tokens roughly via chars

    prompt = (
        "You are a markets brief writer. Summarize the article in 2 crisp bullet points. "
        "Capture what matters for investors (price/volume, guidance, capex, regulatory, M&A, macro). "
        "Be neutral and specific. Do not include fluff.\n\n"
        f"Article:\n{head}\n\nBullets:"
    )

    try:
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You produce concise, finance-grade summaries."},
                {"role": "user", "content": prompt},
            ],
            # No temperature / top_p / max_tokens to avoid model-specific errors
        )
        out = resp.choices[0].message.content.strip()
        return out
    except Exception as e:
        return f"(summary unavailable: {e})"

# -----------------------------
# DB bootstrap
# -----------------------------

DDL = """
CREATE TABLE IF NOT EXISTS recipients (
  id BIGSERIAL PRIMARY KEY,
  email TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS source_feed (
  id BIGSERIAL PRIMARY KEY,
  name TEXT,
  url  TEXT NOT NULL UNIQUE,
  active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS article (
  id BIGSERIAL PRIMARY KEY,
  url           TEXT NOT NULL,
  canonical_url TEXT,
  title         TEXT,
  publisher     TEXT,
  published_at  TIMESTAMPTZ,
  sha256        CHAR(64) NOT NULL UNIQUE,
  raw_html      TEXT,
  clean_text    TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS delivery_log (
  id BIGSERIAL PRIMARY KEY,
  run_date    TIMESTAMPTZ NOT NULL DEFAULT now(),
  subject     TEXT,
  body        TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  recipients  TEXT,
  items       INTEGER NOT NULL DEFAULT 0
);

-- helpful indexes
CREATE INDEX IF NOT EXISTS idx_article_published ON article (published_at DESC);
"""

SEED_RECIPIENTS = """
INSERT INTO recipients (email) VALUES (%s)
ON CONFLICT (email) DO NOTHING;
"""

SEED_FEED_TLN = """
INSERT INTO source_feed (name, url, active)
VALUES (%s, %s, TRUE)
ON CONFLICT (url) DO NOTHING;
"""

def db() -> psycopg.Connection:
    if not DB_URL:
        raise RuntimeError("DATABASE_URL missing")
    return psycopg.connect(DB_URL, autocommit=True)

def init_db_and_seed(owner_email: str | None = None):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)

            # seed recipients
            default_rcpts = ["you@example.com"]
            if owner_email:
                default_rcpts.append(owner_email)
            for rcpt in {r for r in default_rcpts if r}:
                cur.execute(SEED_RECIPIENTS, (rcpt,))

            # seed one Google News feed for TLN/Talen Energy so ingestion has input
            # 7-day search; adjust locale as you like
            gnews_url = (
                "https://news.google.com/rss/search?"
                "q=%28%22Talen%20Energy%22%20OR%20TLN%29%20when%3A7d"
                "&hl=en-CA&gl=CA&ceid=CA%3Aen"
            )
            cur.execute(SEED_FEED_TLN, ("Google News — TLN", gnews_url))

# -----------------------------
# Ingestion
# -----------------------------

def ingest_feeds(minutes: int = 1440) -> dict:
    found = 0
    inserted = 0
    since = now_utc() - timedelta(minutes=minutes)

    with db() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, name, url FROM source_feed WHERE active = TRUE ORDER BY id;")
        feeds = cur.fetchall()

        for feed in feeds:
            try:
                parsed = feedparser.parse(feed["url"])
            except Exception:
                continue

            for entry in parsed.entries:
                # prefer original article for Google News
                link = entry.get("link") or entry.get("id") or ""
                if not link:
                    continue
                link = resolve_google_news_url(link)

                # skip junk/image CDNs
                if is_probably_junk(link):
                    continue

                # basic fields
                title = (entry.get("title") or "").strip() or None
                published = None
                if "published_parsed" in entry and entry.published_parsed:
                    published = datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)

                h = sha256_of(link)

                # de-dupe
                cur.execute("SELECT 1 FROM article WHERE sha256 = %s LIMIT 1;", (h,))
                if cur.fetchone():
                    continue

                found += 1

                # extract text (best effort)
                text = extract_text(link)

                publisher = domain_of(link)

                # store
                cur.execute(
                    """
                    INSERT INTO article (url, title, publisher, published_at, sha256, clean_text)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (sha256) DO NOTHING
                    """,
                    (link, title, publisher, published, h, text),
                )
                if cur.rowcount > 0:
                    inserted += 1

    return {"found_urls": found, "inserted": inserted}

# -----------------------------
# Digest (email)
# -----------------------------

EMAIL_TMPL = Template("""\
QuantBrief Daily — {{ run_date.strftime("%Y-%m-%d") }}
Window: last {{ minutes }}m ending {{ run_date.astimezone(user_tz).strftime("%H:%M %Z") }}

{% if items %}
{% for it in items -%}
{{ it.publisher or "Unknown" }}
{{ it.url }} — {{ it.published_at.isoformat() if it.published_at else "n/a" }}
• {{ it.summary|replace('\n','\n• ') }}
{% if not loop.last %}

{% endif -%}
{% endfor %}
{% else %}
No items found in the selected window.
{% endif %}

Sources are links only. We don’t republish paywalled content. Filters editable by owner.
""")

def collect_recent(minutes: int = 1440, limit: int = 40) -> list[dict]:
    since = now_utc() - timedelta(minutes=minutes)
    with db() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, url, title, publisher, published_at, clean_text
            FROM article
            WHERE (published_at IS NULL OR published_at >= %s)
            ORDER BY COALESCE(published_at, created_at) DESC
            LIMIT %s
            """,
            (since, limit),
        )
        return cur.fetchall()

def send_mailgun(recipients: list[str], subject: str, text: str) -> tuple[bool, str]:
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN and MAILGUN_FROM):
        return (False, "Mailgun not configured; skipped send")
    try:
        resp = requests.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={
                "from": MAILGUN_FROM,
                "to": recipients,
                "subject": subject,
                "text": text,
            },
            timeout=20,
        )
        if 200 <= resp.status_code < 300:
            return (True, "sent")
        return (False, f"mailgun error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        return (False, f"mailgun exception: {e}")

def make_digest(minutes: int = 1440) -> dict:
    # recipients
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT email FROM recipients ORDER BY id;")
        rcpts = [r[0] for r in cur.fetchall()]

    articles = collect_recent(minutes=minutes, limit=40)

    items = []
    for a in articles:
        # Summarize on the fly (only if we have text; otherwise, skip to avoid empty blurbs)
        summary = ""
        if a.get("clean_text"):
            summary = summarize_text(a["clean_text"])
        else:
            # Try a quick fallback fetch if missing (best effort)
            text = extract_text(a["url"])
            if text:
                summary = summarize_text(text)

        items.append(
            {
                "publisher": a.get("publisher"),
                "title": a.get("title"),
                "url": a.get("url"),
                "published_at": a.get("published_at"),
                "summary": summary or "No clean text available.",
            }
        )

    run_date = now_utc()
    subject = f"QuantBrief Daily — {run_date.strftime('%Y-%m-%d')}"
    # crude TZ: use America/Toronto if available; otherwise UTC
    # (FastAPI email body will still show UTC if pytz/zoneinfo not configured on Render)
    user_tz = timezone.utc
    body = EMAIL_TMPL.render(run_date=run_date, minutes=minutes, items=items, user_tz=user_tz)

    sent_ok, send_note = send_mailgun(rcpts, subject, body)

    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO delivery_log (run_date, subject, body, recipients, items) VALUES (now(), %s, %s, %s, %s)",
            (subject, body, ",".join(rcpts), len(items)),
        )

    return {
        "sent_to": ",".join(rcpts),
        "items": len(items),
        "note": send_note,
        "ok": sent_ok,
    }

# -----------------------------
# Routes
# -----------------------------

@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK"

@app.get("/admin/health", response_class=PlainTextResponse)
def admin_health(request: Request):
    admin_guard(request)
    env = {
        "DATABASE_URL": bool(DATABASE_URL),
        "OPENAI_API_KEY": bool(OPENAI_API_KEY),
        "OPENAI_MODEL": bool(OPENAI_MODEL),
        "OPENAI_MAX_OUTPUT_TOKENS": bool(OPENAI_MAX_OUTPUT_TOKENS),
        "MAILGUN_API_KEY": bool(MAILGUN_API_KEY),
        "MAILGUN_DOMAIN": bool(MAILGUN_DOMAIN),
        "MAILGUN_FROM": bool(MAILGUN_FROM),
        "OWNER_EMAIL": bool(os.getenv("OWNER_EMAIL", "")),
        "ADMIN_TOKEN": bool(ADMIN_TOKEN),
    }

    counts = {}
    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM article;")
            counts["articles"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM recipients;")
            counts["recipients"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM source_feed WHERE active = TRUE;")
            counts["active_feeds"] = cur.fetchone()[0]
    except Exception as e:
        counts["error"] = str(e)

    return "env\n---\n" + json.dumps(env, indent=2) + "\n\ncounts\n------\n" + "\n".join(
        f"{k}={v}" for k, v in counts.items()
    )

@app.get("/admin/test_openai", response_class=PlainTextResponse)
def admin_test_openai(request: Request):
    admin_guard(request)
    try:
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": "Respond with exactly: OK"},
            ],
        )
        txt = resp.choices[0].message.content.strip()
        ok = (txt == "OK")
        return f"   ok model      sample\n   -- -----      ------\n{str(ok):5} {OPENAI_MODEL:10} {txt}"
    except Exception as e:
        return f"   ok model      err\n   -- -----      ---\nFalse {OPENAI_MODEL:10} {e}"

@app.post("/admin/init", response_class=PlainTextResponse)
def admin_init(request: Request):
    admin_guard(request)
    try:
        init_db_and_seed(owner_email=os.getenv("OWNER_EMAIL"))
        return "Initialized."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Init error: {e}")

@app.post("/cron/ingest", response_class=PlainTextResponse)
def cron_ingest(request: Request, minutes: int = Query(1440)):
    admin_guard(request)
    try:
        res = ingest_feeds(minutes=minutes)
        return (
            "  ok found_urls inserted\n"
            "  -- ---------- --------\n"
            f"True{res['found_urls']:11}{res['inserted']:9}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest error: {e}")

@app.post("/cron/digest", response_class=PlainTextResponse)
def cron_digest(request: Request, minutes: int = Query(1440)):
    admin_guard(request)
    try:
        res = make_digest(minutes=minutes)
        return (
            "  ok sent_to                       items\n"
            "  -- -------                       -----\n"
            f"True {res['sent_to']:30} {res['items']:5}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Digest error: {e}")
