import os
import re
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

import httpx
import feedparser
from fastapi import FastAPI, Request, HTTPException, Query
from pydantic import BaseModel
from jinja2 import Template
import psycopg
from psycopg.rows import dict_row

from openai import OpenAI

# ------------------------------------------------------------------------------
# Config / Env
# ------------------------------------------------------------------------------

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("quantbrief")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

# Auth for admin endpoints
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "dev-token")

# Email (Mailgun example; adapt to your provider if needed)
MAILGUN_DOMAIN = os.getenv("MAILGUN_DOMAIN")            # e.g. "mg.quantbrief.ca"
MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY")
FROM_ADDR = os.getenv("FROM_ADDR", "QuantBrief Daily <daily@mg.quantbrief.ca>")
RECIPIENTS = os.getenv("RECIPIENTS", "you@example.com")  # comma-separated

# Summarization model
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Exclusions & allowlist
EXCLUDED_DOMAINS = {
    d.strip().lower()
    for d in os.getenv("EXCLUDED_DOMAINS", "marketbeat.com,www.marketbeat.com,newser.com,www.newser.com").split(",")
    if d.strip()
}
QUALITY_DOMAINS = {
    d.strip().lower()
    for d in os.getenv("QUALITY_DOMAINS", "").split(",")
    if d.strip()
}  # optional allowlist; if empty, it's ignored

# Default company search
DEFAULT_QUERY = os.getenv("NEWS_QUERY", "(Talen Energy) OR TLN")

# Google News locale (affects feed language and region)
GN_HL = os.getenv("GN_HL", "en-US")
GN_GL = os.getenv("GN_GL", "US")
GN_CEID = os.getenv("GN_CEID", "US:en")

# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------

app = FastAPI()


# ------------------------------------------------------------------------------
# DB Helpers
# ------------------------------------------------------------------------------

def db() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)


DDL_SQL = """
CREATE TABLE IF NOT EXISTS source_feed (
    id SERIAL PRIMARY KEY,
    kind TEXT NOT NULL,          -- 'google-news'
    url TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    name TEXT NOT NULL,          -- human label (e.g. "Google News")
    persist_days INTEGER NOT NULL DEFAULT 30,
    UNIQUE(kind, url)
);

CREATE TABLE IF NOT EXISTS found_url (
    id BIGSERIAL PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    publisher TEXT,
    published_at TIMESTAMPTZ,
    seen_first_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    seen_last_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS found_url_published_idx ON found_url (published_at DESC);

CREATE TABLE IF NOT EXISTS summary (
    id BIGSERIAL PRIMARY KEY,
    found_url_id BIGINT NOT NULL REFERENCES found_url(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(found_url_id)
);

CREATE TABLE IF NOT EXISTS send_log (
    id BIGSERIAL PRIMARY KEY,
    recipients TEXT NOT NULL,
    num_items INTEGER NOT NULL,
    num_summarized INTEGER NOT NULL,
    window_minutes INTEGER NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def ensure_schema():
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL_SQL)


def upsert_feed(kind: str, url: str, name: str, persist_days: int = 30):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO source_feed (kind, url, active, name, persist_days)
                VALUES (%s, %s, TRUE, %s, %s)
                ON CONFLICT (kind, url) DO UPDATE
                SET active = EXCLUDED.active,
                    name = EXCLUDED.name,
                    persist_days = EXCLUDED.persist_days
                """,
                (kind, url, name, persist_days),
            )


def insert_or_touch_url(u: Dict) -> Tuple[bool, Optional[int]]:
    """
    Insert a URL row if new; otherwise update seen_last_at.
    Returns (inserted, id or None if fetch fails)
    """
    with db() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                """
                INSERT INTO found_url (url, title, publisher, published_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE
                SET seen_last_at = NOW()
                RETURNING id
                """,
                (u["url"], u["title"], u.get("publisher"), u.get("published_at")),
            )
            row = cur.fetchone()
            return True, row["id"] if row else None
        except Exception as e:
            log.exception("insert_or_touch_url error for %s", u["url"])
            return False, None


def store_summary(found_url_id: int, model: str, text: str):
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO summary (found_url_id, model, summary_text)
            VALUES (%s, %s, %s)
            ON CONFLICT (found_url_id) DO UPDATE SET
                model = EXCLUDED.model,
                summary_text = EXCLUDED.summary_text,
                created_at = NOW()
            """,
            (found_url_id, model, text),
        )


def fetch_items_for_window(window_minutes: int, window_end: datetime) -> List[Dict]:
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.id, f.url, f.title, f.publisher, f.published_at, s.summary_text
            FROM found_url f
            LEFT JOIN summary s ON s.found_url_id = f.id
            WHERE f.seen_last_at >= %s
            ORDER BY COALESCE(f.published_at, f.seen_last_at) DESC
            """,
            (window_end - timedelta(minutes=window_minutes),),
        )
        return list(cur.fetchall())


def log_send(recipients: str, num_items: int, num_summarized: int, window_minutes: int, window_end: datetime):
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO send_log (recipients, num_items, num_summarized, window_minutes, window_end)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (recipients, num_items, num_summarized, window_minutes, window_end),
        )

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def require_admin(req: Request):
    token = req.headers.get("x-admin-token") or req.headers.get("authorization")
    if not token or token.replace("Bearer ", "").strip() != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def strip_utm(url: str) -> str:
    if "?" not in url:
        return url
    base, q = url.split("?", 1)
    # Keep only non-tracking params
    keep = []
    for kv in q.split("&"):
        k = kv.split("=", 1)[0].lower()
        if not k.startswith("utm_") and k not in {"gclid", "fbclid", "mc_cid", "mc_eid"}:
            keep.append(kv)
    return base if not keep else base + "?" + "&".join(keep)


def domain_of(url: str) -> str:
    m = re.match(r"^https?://([^/]+)", url)
    return (m.group(1).lower() if m else "").lstrip("www.")


def is_allowed(url: str) -> bool:
    d = domain_of(url)
    if d in EXCLUDED_DOMAINS or ("www." + d) in EXCLUDED_DOMAINS:
        return False
    if QUALITY_DOMAINS and d not in QUALITY_DOMAINS and ("www." + d) not in QUALITY_DOMAINS:
        return False
    return True


def build_google_news_feed(query: str, days: int = 7, excluded: Optional[List[str]] = None) -> str:
    """
    Build a Google News RSS URL with exclusions embedded in the query
    (so Google doesn't even return them).
    """
    excluded = excluded or []
    minus_sites = " ".join(f"-site:{s}" for s in excluded)
    q = f"{query} {minus_sites} when:{days}d".strip()
    # Manual encode: feedparser/httpx can handle normal URL encoding
    import urllib.parse as up
    qs = up.urlencode({
        "q": q,
        "hl": GN_HL,
        "gl": GN_GL,
        "ceid": GN_CEID
    })
    return f"https://news.google.com/rss/search?{qs}"


def parse_google_news(feed_url: str, minutes: int) -> List[Dict]:
    """
    Parse Google News RSS and return normalized articles:
    - url from <source href> (canonical publisher URL)
    - title from entry.title
    - publisher from entry.source.title (or domain)
    - published_at from entry.published_parsed if present
    Applies time window (minutes) and domain filters.
    """
    log.info("parsed feed: %s", feed_url)
    # We only fetch the RSS feed. We do not click every Google redirect link.
    resp = httpx.get(feed_url, timeout=30)
    resp.raise_for_status()
    parsed = feedparser.parse(resp.text)

    window_start = now_utc() - timedelta(minutes=minutes)
    out: List[Dict] = []

    for e in parsed.entries:
        # Canonical article URL is in <source href="...">
        src_href = None
        src_title = None
        try:
            # feedparser maps <source> to .source
            if hasattr(e, "source"):
                src_href = getattr(e.source, "href", None)
                src_title = getattr(e.source, "title", None)
        except Exception:
            pass

        link = src_href or getattr(e, "link", None)
        if not link:
            continue

        link = strip_utm(link)
        if not is_allowed(link):
            continue

        # Publisher
        publisher = src_title or domain_of(link) or "Unknown"

        # Title
        title = getattr(e, "title", "").strip() or domain_of(link)

        # Published time (if present)
        published_at = None
        if hasattr(e, "published_parsed") and e.published_parsed:
            try:
                published_at = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            except Exception:
                published_at = None

        # Time window filter: if we have published_at, use it; else pass (we'll keep)
        if published_at and published_at < window_start:
            continue

        out.append({
            "url": link,
            "title": title,
            "publisher": publisher,
            "published_at": published_at,
        })

    return out


# ------------------------------------------------------------------------------
# Summarization
# ------------------------------------------------------------------------------

_openai = OpenAI(api_key=OPENAI_API_KEY)

SUMMARY_SYSTEM = (
    "You are QuantBrief, a crisp financial news summarizer. "
    "For each item, write ONE short bullet (~20–35 words), neutral tone, "
    "focusing on what happened and why investors might care. No hype."
)

def summarize_one(title: str, publisher: str) -> str:
    prompt = (
        f"Title: {title}\n"
        f"Source: {publisher}\n\n"
        "Return one sentence (~20–35 words)."
    )
    try:
        resp = _openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Guardrails
        text = re.sub(r"\s+", " ", text)
        # Strip leading bullets/dashes if any
        text = re.sub(r"^[-•\u2022]\s*", "", text)
        return text
    except Exception as e:
        log.warning("OpenAI summarize error: %s", e)
        # Fallback: very short heuristic summary
        return f"{publisher}: {title}"


# ------------------------------------------------------------------------------
# Email
# ------------------------------------------------------------------------------

EMAIL_TEMPLATE = Template("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>QuantBrief Daily</title>
</head>
<body style="font-family: Arial, Helvetica, sans-serif; color:#111; line-height:1.5;">
  <h2 style="margin:0 0 8px 0;">QuantBrief Daily — {{ now_str }}</h2>
  <div style="color:#444; margin:0 0 16px 0;">
    Window: last {{ window_minutes }} minutes ending {{ window_end_iso }}
  </div>

  {% if items %}
  <ol style="padding-left: 18px;">
    {% for it in items %}
      <li style="margin-bottom:12px;">
        <div style="font-weight:600;">
          <a href="{{ it.url }}" target="_blank" rel="noopener noreferrer">{{ it.title }}</a>
          <span style="color:#666;"> — {{ it.publisher }}</span>
        </div>
        <div style="margin-top:4px;">- {{ it.summary }}</div>
      </li>
    {% endfor %}
  </ol>
  {% else %}
    <p>No items found for this window.</p>
  {% endif %}

  <hr style="margin:20px 0;">
  <div style="font-size:12px; color:#666;">
    Sources are links only. We don’t republish paywalled content.
  </div>
</body>
</html>
""".strip())


def send_email_html(subject: str, html: str, recipients: List[str]) -> Tuple[bool, str]:
    """
    Sends via Mailgun API. You can swap this out for SES/Sendgrid/etc.
    """
    if not MAILGUN_DOMAIN or not MAILGUN_API_KEY:
        log.warning("Mailgun not configured; skipping email send.")
        return False, "mailgun not configured"

    url = f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages"
    data = {
        "from": FROM_ADDR,
        "to": recipients,
        "subject": subject,
        "html": html,
    }
    try:
        r = httpx.post(url, auth=("api", MAILGUN_API_KEY), data=data, timeout=30)
        r.raise_for_status()
        return True, "ok"
    except Exception as e:
        log.exception("send_email error")
        return False, str(e)


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "time": now_utc().isoformat()}


class InitBody(BaseModel):
    query: Optional[str] = None
    days: Optional[int] = 7
    # Optionally override exclusions / allowlist for this service
    excluded_domains: Optional[List[str]] = None
    quality_domains: Optional[List[str]] = None


@app.post("/admin/init")
def admin_init(body: InitBody, request: Request):
    require_admin(request)
    ensure_schema()

    q = (body.query or DEFAULT_QUERY).strip()
    days = int(body.days or 7)

    # Allow runtime override of lists
    ex = body.excluded_domains if body.excluded_domains is not None else list(EXCLUDED_DOMAINS)
    allow = body.quality_domains if body.quality_domains is not None else list(QUALITY_DOMAINS)

    # Persist optional allowlist/exclusions as env-driven at runtime (not stored)
    log.info("init: query=%s days=%s excl=%s allow=%s", q, days, ex, allow)

    feed_url = build_google_news_feed(q, days=days, excluded=ex)
    upsert_feed(kind="google-news", url=feed_url, name="Google News", persist_days=max(30, days + 7))

    return {"initialized": True, "feed_url": feed_url}


@app.post("/cron/ingest")
def cron_ingest(request: Request, minutes: int = Query(1440, ge=5, le=60*24*30)):
    require_admin(request)
    ensure_schema()

    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, kind, url, active, name FROM source_feed WHERE active = TRUE")
        feeds = list(cur.fetchall())

    total_new = 0
    total_seen = 0
    for f in feeds:
        if f["kind"] != "google-news":
            continue
        items = parse_google_news(f["url"], minutes)
        # Dedup by URL within this batch as well
        seen_batch = set()
        for it in items:
            u = it["url"]
            if u in seen_batch:
                continue
            seen_batch.add(u)

            inserted, _id = insert_or_touch_url(it)
            if inserted:
                total_new += 1
            else:
                total_seen += 1

    return {"ok": True, "inserted": total_new, "touched": total_seen}


@app.post("/cron/summarize")
def cron_summarize(request: Request, minutes: int = Query(1440, ge=5, le=60*24*30)):
    require_admin(request)
    ensure_schema()

    window_end = now_utc()
    window_items = fetch_items_for_window(minutes, window_end)

    to_summarize = [w for w in window_items if not w["summary_text"]]
    summarized = 0

    for w in to_summarize:
        s = summarize_one(w["title"], w.get("publisher") or domain_of(w["url"]))
        store_summary(w["id"], OPENAI_MODEL, s)
        summarized += 1

    return {"ok": True, "summarized": summarized, "pending": len(to_summarize) - summarized}


@app.post("/cron/digest")
def cron_digest(request: Request, minutes: int = Query(1440, ge=5, le=60*24*30)):
    require_admin(request)
    ensure_schema()

    window_end = now_utc()
    items = fetch_items_for_window(minutes, window_end)

    # Ensure everything in-window has a summary
    for w in items:
        if not w["summary_text"]:
            s = summarize_one(w["title"], w.get("publisher") or domain_of(w["url"]))
            store_summary(w["id"], OPENAI_MODEL, s)
            w["summary_text"] = s

    # Format email
    formatted = []
    for w in items:
        # Apply domain filter again just before sending (in case URL changed)
        if not is_allowed(w["url"]):
            continue
        formatted.append({
            "url": w["url"],
            "title": w["title"],
            "publisher": w.get("publisher") or domain_of(w["url"]),
            "summary": w["summary_text"] or (w.get("publisher") or "") + ": " + w["title"],
        })

    now_str = window_end.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
    html = EMAIL_TEMPLATE.render(
        now_str=now_str,
        window_minutes=minutes,
        window_end_iso=window_end.isoformat(),
        items=formatted
    )
    subject = f"QuantBrief Daily — {now_str}"

    recips = [r.strip() for r in RECIPIENTS.split(",") if r.strip()]
    ok, msg = send_email_html(subject, html, recips)

    log_send(",".join(recips), num_items=len(formatted), num_summarized=len(formatted), window_minutes=minutes, window_end=window_end)

    return {"ok": ok, "message": msg, "sent_to": recips, "items": len(formatted)}


# Root (intentional 404 to match your logs)
@app.get("/")
def root():
    raise HTTPException(status_code=404, detail="Not Found")
