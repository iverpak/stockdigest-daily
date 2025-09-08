# app.py
import os
import re
import ssl
import json
import time
import math
import uuid
import smtplib
import logging
import datetime as dt
from typing import Optional, List, Tuple

import psycopg
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from email.message import EmailMessage

# Optional but handy
import requests
import feedparser

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOG = logging.getLogger("quantbrief")

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN") or os.getenv("ADMIN_SECRET") or "changeme"

DATABASE_URL = (
    os.getenv("DATABASE_URL")  # preferred
    or os.getenv("POSTGRES_URL")
    or os.getenv("PG_DSN")
)

# SMTP / email
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("EMAIL_FROM") or os.getenv("SMTP_FROM") or SMTP_USER
DIGEST_TO = os.getenv("DIGEST_TO") or os.getenv("EMAIL_TO") or os.getenv("ADMIN_EMAIL")
SMTP_SSL = os.getenv("SMTP_SSL", "0") == "1" or SMTP_PORT == 465
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "1") != "0"  # ignored if SMTP_SSL

DEFAULT_RETENTION_DAYS = int(os.getenv("DEFAULT_RETENTION_DAYS", "30"))

# ------------------------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------------------------
def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg.connect(DATABASE_URL, autocommit=True)

SCHEMA_SQL = """
-- Create base tables if they don't exist
CREATE TABLE IF NOT EXISTS source_feed (
    id            BIGSERIAL PRIMARY KEY,
    url           TEXT NOT NULL,
    name          TEXT,
    active        BOOLEAN NOT NULL DEFAULT TRUE,
    language      TEXT,
    retain_days   INTEGER, -- optional per-feed retention override (0 or NULL = use default)
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Prefer a UNIQUE INDEX to avoid duplicate constraint errors during upgrades
CREATE UNIQUE INDEX IF NOT EXISTS uq_source_feed_url ON source_feed(url);

-- Keep updated_at fresh
CREATE OR REPLACE FUNCTION set_source_feed_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at := NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_source_feed_updated_at ON source_feed;
CREATE TRIGGER trg_source_feed_updated_at
BEFORE UPDATE ON source_feed
FOR EACH ROW EXECUTE FUNCTION set_source_feed_updated_at();

CREATE TABLE IF NOT EXISTS found_url (
    id            BIGSERIAL PRIMARY KEY,
    url           TEXT NOT NULL,
    title         TEXT,
    feed_id       BIGINT REFERENCES source_feed(id) ON DELETE CASCADE,
    language      TEXT,
    published_at  TIMESTAMPTZ,
    found_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- composite uniqueness: the same URL can appear in different feeds, but only once per feed
CREATE UNIQUE INDEX IF NOT EXISTS uq_found_url_url_feed ON found_url(url, feed_id);

CREATE INDEX IF NOT EXISTS idx_found_url_found_at ON found_url(found_at);
CREATE INDEX IF NOT EXISTS idx_found_url_feed_id ON found_url(feed_id);
"""

def exec_sql_batch(conn, sql: str):
    with conn.cursor() as cur:
        cur.execute(sql)

def ensure_schema():
    with get_conn() as conn:
        exec_sql_batch(conn, SCHEMA_SQL)

def upsert_feed(url: str, name: Optional[str] = None,
                language: Optional[str] = None,
                retain_days: Optional[int] = None) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO source_feed (url, name, language, retain_days)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE
                SET name = COALESCE(EXCLUDED.name, source_feed.name),
                    language = COALESCE(EXCLUDED.language, source_feed.language),
                    retain_days = COALESCE(EXCLUDED.retain_days, source_feed.retain_days),
                    updated_at = NOW()
            RETURNING id
            """,
            (url, name, language, retain_days),
        )
        (feed_id,) = cur.fetchone()
        return int(feed_id)

def list_active_feeds():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, url, COALESCE(NULLIF(retain_days, 0), %s) AS effective_retain_days,
                   COALESCE(NULLIF(language, ''), 'en') AS language, name
            FROM source_feed
            WHERE active = TRUE
            """,
            (DEFAULT_RETENTION_DAYS,),
        )
        return cur.fetchall()

def insert_found_url(url: str, title: str, feed_id: int,
                     language: Optional[str], published_at: Optional[dt.datetime]) -> bool:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO found_url (url, title, feed_id, language, published_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (url, feed_id) DO NOTHING
            """,
            (url, title, feed_id, language, published_at),
        )
        return cur.rowcount > 0

def prune_old_found_urls(default_days: int = DEFAULT_RETENTION_DAYS) -> int:
    """
    Delete rows from found_url older than each feed's retention:
    COALESCE(NULLIF(source_feed.retain_days, 0), default_days)
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH params AS (
              SELECT %s::int AS default_days
            ),
            del AS (
              DELETE FROM found_url f
              USING source_feed s, params p
              WHERE f.feed_id = s.id
                AND f.found_at < NOW() - make_interval(days => COALESCE(NULLIF(s.retain_days, 0), p.default_days))
              RETURNING 1
            )
            SELECT COUNT(*) FROM del
            """,
            (default_days,),
        )
        (deleted,) = cur.fetchone()
        # robust logging: deleted might be None if no rows were matched, normalize:
        deleted = int(deleted or 0)
        LOG.warning("prune_old_found_urls: deleted=%d, default_days=%d, has_retain_days=%s",
                    deleted, default_days, "exists")
        return deleted

# ------------------------------------------------------------------------------
# Email / Digest
# ------------------------------------------------------------------------------
def send_smtp_email(subject: str, html: str, text: Optional[str] = None,
                    to_addrs: Optional[List[str]] = None) -> bool:
    recips = to_addrs or [a.strip() for a in (DIGEST_TO or "").split(",") if a.strip()]
    if not recips:
        LOG.warning("No recipients configured. Set DIGEST_TO or EMAIL_TO (comma-separated).")
        return False
    if not (SMTP_HOST and SMTP_FROM):
        LOG.error("SMTP not configured. Need SMTP_HOST and EMAIL_FROM/SMTP_FROM.")
        return False

    if text is None:
        text = re.sub(r"<[^>]+>", "", html or "")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = ", ".join(recips)
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

    ctx = ssl.create_default_context()
    try:
        if SMTP_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as s:
                if SMTP_USER and SMTP_PASS:
                    s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                if SMTP_STARTTLS:
                    s.starttls(context=ctx)
                if SMTP_USER and SMTP_PASS:
                    s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
        LOG.info("Email sent to %s (subject=%r)", msg["To"], subject)
        return True
    except Exception as e:
        LOG.exception("Email send failed: %s", e)
        return False

def build_digest(minutes: int) -> Tuple[str, str, str, int]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              f.url,
              COALESCE(NULLIF(f.title,''), f.url) AS title,
              s.name AS feed_name,
              f.published_at,
              f.found_at
            FROM found_url f
            LEFT JOIN source_feed s ON s.id = f.feed_id
            WHERE f.found_at >= NOW() - (%s || ' minutes')::interval
            ORDER BY f.published_at DESC NULLS LAST, f.found_at DESC
            """,
            (minutes,),
        )
        rows = cur.fetchall()

    count = len(rows)
    days = max(1, minutes // 1440)
    subject = f"Quantbrief digest — last {days} day{'s' if days != 1 else ''} ({count} item{'s' if count != 1 else ''})"

    if count == 0:
        html = "<p>No new items in the selected window.</p>"
        text = "No new items in the selected window."
        return subject, html, text, 0

    lis = []
    for (url, title, feed_name, published_at, found_at) in rows:
        meta = []
        if feed_name:
            meta.append(feed_name)
        if published_at:
            meta.append(f"published {published_at:%Y-%m-%d %H:%M}")
        if found_at:
            meta.append(f"found {found_at:%Y-%m-%d %H:%M}")
        suffix = " — " + " • ".join(meta) if meta else ""
        lis.append(f'<li><a href="{url}">{title}</a>{suffix}</li>')

    html = f"""
    <html><body>
      <h2>{subject}</h2>
      <ul>
        {''.join(lis)}
      </ul>
    </body></html>
    """
    text_lines = []
    for (url, title, feed_name, published_at, found_at) in rows:
        meta = []
        if feed_name:
            meta.append(feed_name)
        if published_at:
            meta.append(f"published {published_at:%Y-%m-%d %H:%M}")
        if found_at:
            meta.append(f"found {found_at:%Y-%m-%d %H:%M}")
        suffix = (" — " + " • ".join(meta)) if meta else ""
        text_lines.append(f"- {title} {url}{suffix}")
    text = f"{subject}\n\n" + "\n".join(text_lines)
    return subject, html, text, count

# ------------------------------------------------------------------------------
# Feed ingest
# ------------------------------------------------------------------------------
def _safe_dt(value) -> Optional[dt.datetime]:
    if not value:
        return None
    if isinstance(value, dt.datetime):
        return value
    # feedparser returns time.struct_time in published_parsed
    try:
        if hasattr(value, "tm_year"):  # struct_time-ish
            return dt.datetime.fromtimestamp(time.mktime(value), tz=dt.timezone.utc)
    except Exception:
        pass
    try:
        # if it’s an ISO string
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None

def fetch_feed_entries(feed_url: str, minutes: int) -> List[dict]:
    """
    Return a list of dicts: {url, title, published_at}
    """
    out = []
    try:
        # requests -> bytes -> feedparser
        r = requests.get(feed_url, timeout=20)
        r.raise_for_status()
        parsed = feedparser.parse(r.content)
        LOG.info("parsed feed: %s", feed_url)
        LOG.info("feed entries: %d", len(parsed.entries or []))
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)
        for e in parsed.entries or []:
            link = getattr(e, "link", None) or getattr(e, "id", None)
            title = getattr(e, "title", None) or (link or "")
            published_at = _safe_dt(getattr(e, "published_parsed", None)) or _safe_dt(getattr(e, "published", None))
            # allow items with no published date; we'll keep them
            if published_at and published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=dt.timezone.utc)
            # crude window check: include if no published_at OR published_at >= cutoff
            if (published_at is None) or (published_at >= cutoff):
                out.append({"url": link, "title": title, "published_at": published_at})
    except Exception as e:
        LOG.exception("fetch_feed_entries error for %s: %s", feed_url, e)
    return out

# ------------------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------------------
def require_admin(request: Request):
    token_hdr = request.headers.get("x-admin-token") or ""
    auth_hdr = request.headers.get("Authorization") or ""
    bearer = ""
    if auth_hdr.startswith("Bearer "):
        bearer = auth_hdr.split(" ", 1)[1].strip()

    if ADMIN_TOKEN and (token_hdr == ADMIN_TOKEN or bearer == ADMIN_TOKEN):
        return True
    raise HTTPException(status_code=401, detail="Unauthorized")

# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
app = FastAPI(title="Quantbrief")

@app.get("/")
def root():
    return {"ok": True, "service": "quantbrief", "message": "hi 👋"}

@app.post("/admin/init")
def admin_init(_: bool = Depends(require_admin)):
    ensure_schema()

    # Seed example Google News feeds (idempotent)
    seeds = [
        ("https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN&hl=en-US&gl=US&ceid=US:en", "Google News: Talen Energy / TLN (all)", "en", None),
        ("https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+-site:newser.com+-site:marketbeat.com+when:3d&hl=en-US&gl=US&ceid=US:en", "Google News: Talen Energy / TLN (3d, filtered)", "en", None),
        ("https://news.google.com/rss/search?q=%28Talen+Energy%29+OR+TLN+-site%3Anewser.com+-site%3Awww.marketbeat.com+-site%3Amarketbeat.com+-site%3Awww.newser.com+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen", "Google News: Talen Energy / TLN (7d, filtered A)", "en", None),
        ("https://news.google.com/rss/search?q=%28%28Talen+Energy%29+OR+TLN%29+-site%3Anewser.com+-site%3Awww.newser.com+-site%3Amarketbeat.com+-site%3Awww.marketbeat.com+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen", "Google News: Talen Energy / TLN (7d, filtered B)", "en", None),
    ]
    feed_ids = []
    for url, name, language, retain_days in seeds:
        fid = upsert_feed(url=url, name=name, language=language, retain_days=retain_days)
        feed_ids.append(fid)

    return {"ok": True, "feeds": feed_ids}

@app.post("/cron/ingest")
def cron_ingest(minutes: int = 1440, _: bool = Depends(require_admin)):
    ensure_schema()

    feeds = list_active_feeds()
    total_new = 0
    by_feed = []

    for (feed_id, url, eff_retain_days, language, name) in feeds:
        entries = fetch_feed_entries(url, minutes=minutes)
        new_count = 0
        for e in entries:
            if not e.get("url"):
                continue
            added = insert_found_url(
                url=e["url"],
                title=e.get("title") or e["url"],
                feed_id=feed_id,
                language=language,
                published_at=e.get("published_at"),
            )
            if added:
                new_count += 1
        by_feed.append({"feed_id": feed_id, "url": url, "added": new_count})
        total_new += new_count

    deleted = prune_old_found_urls(default_days=DEFAULT_RETENTION_DAYS)
    LOG.info("pruned %d old found_url rows (older than %d days)", int(deleted or 0), DEFAULT_RETENTION_DAYS)

    return {"ok": True, "new_items": total_new, "deleted": int(deleted or 0), "feeds": by_feed}

@app.post("/cron/digest")
def cron_digest(minutes: int = 1440, to: Optional[str] = None, _: bool = Depends(require_admin)):
    subject, html, text, count = build_digest(minutes)
    if count == 0:
        LOG.info("Digest has no items for last %d minutes; skipping send.", minutes)
        return {"ok": True, "sent": 0, "items": 0, "skipped": "no items"}
    recips = [a.strip() for a in (to or DIGEST_TO or "").split(",") if a.strip()]
    ok = send_smtp_email(subject, html, text, recips)
    return {"ok": bool(ok), "sent": len(recips), "items": count}

@app.post("/admin/test-email")
def admin_test_email(to: Optional[str] = None, _: bool = Depends(require_admin)):
    recips = [a.strip() for a in (to or DIGEST_TO or "").split(",") if a.strip()]
    ok = send_smtp_email("Quantbrief test email", "<p>Hello from Quantbrief!</p>", "Hello from Quantbrief!", recips)
    return {"ok": bool(ok), "to": recips}
