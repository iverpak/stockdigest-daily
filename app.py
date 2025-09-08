# app.py
import os
import re
import logging
from typing import Optional, Iterable
from datetime import datetime, timezone, timedelta

import psycopg
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import feedparser
import ssl
import smtplib
from email.message import EmailMessage

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOG = logging.getLogger("quantbrief")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
APP_NAME = os.getenv("APP_NAME", "quantbrief")

# Email / SMTP
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("EMAIL_FROM") or os.getenv("SMTP_FROM") or SMTP_USER
DIGEST_TO = os.getenv("DIGEST_TO") or os.getenv("EMAIL_TO") or os.getenv("ADMIN_EMAIL")
SMTP_SSL = os.getenv("SMTP_SSL", "0") == "1" or SMTP_PORT == 465
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "1") != "0"  # ignored if SMTP_SSL

DEFAULT_RETENTION_DAYS = int(os.getenv("DEFAULT_RETENTION_DAYS", "30"))

# Seed feeds (you can edit/add more)
SEED_FEEDS = [
    # Name, URL, retain_days
    ("Google News: Talen Energy (3d, filtered)",
     "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+-site:newser.com+-site:marketbeat.com+when:3d&hl=en-US&gl=US&ceid=US:en", 30),
    ("Google News: Talen Energy (all)",
     "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN&hl=en-US&gl=US&ceid=US:en", 30),
    ("Google News: Talen Energy (7d, filtered #1)",
     "https://news.google.com/rss/search?q=%28Talen+Energy%29+OR+TLN+-site%3Anewser.com+-site%3Awww.marketbeat.com+-site%3Amarketbeat.com+-site%3Awww.newser.com+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen", 30),
    ("Google News: Talen Energy (7d, filtered #2)",
     "https://news.google.com/rss/search?q=%28%28Talen+Energy%29+OR+TLN%29+-site%3Anewser.com+-site%3Awww.newser.com+-site%3Amarketbeat.com+-site%3Awww.marketbeat.com+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen", 30),
]

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title=APP_NAME)

def require_admin(req: Request):
    """Raise 401 unless the admin token matches."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfigured: ADMIN_TOKEN not set")
    h = req.headers.get("x-admin-token") or ""
    auth = (req.headers.get("authorization") or "").split()
    bearer = auth[1] if len(auth) == 2 and auth[0].lower() == "bearer" else ""
    if h != ADMIN_TOKEN and bearer != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------
def get_conn() -> psycopg.Connection:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    # Autocommit True so DDL batches don't get stuck on errors mid-transaction
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    return conn

def exec_sql(conn: psycopg.Connection, sql: str, params: Optional[Iterable] = None) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, params or ())

# -----------------------------------------------------------------------------
# Schema (idempotent)
# -----------------------------------------------------------------------------
SCHEMA_SQL = """
-- Tables
CREATE TABLE IF NOT EXISTS source_feed (
    id           BIGSERIAL PRIMARY KEY,
    url          TEXT NOT NULL,
    name         TEXT,
    retain_days  INTEGER,
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Ensure unique URL constraint exists (works even if table pre-existed)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_source_feed_url'
          AND conrelid = 'source_feed'::regclass
    ) THEN
        ALTER TABLE source_feed
        ADD CONSTRAINT uq_source_feed_url UNIQUE (url);
    END IF;
END$$;

-- Add missing columns safely
ALTER TABLE source_feed
    ADD COLUMN IF NOT EXISTS retain_days INTEGER,
    ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS name TEXT;

CREATE TABLE IF NOT EXISTS found_url (
    id            BIGSERIAL PRIMARY KEY,
    url           TEXT NOT NULL,
    title         TEXT,
    feed_id       BIGINT REFERENCES source_feed(id) ON DELETE SET NULL,
    language      TEXT,
    published_at  TIMESTAMPTZ,
    found_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Unique on URL to dedupe across feeds (adjust if you really want duplicates)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_found_url_url'
          AND conrelid = 'found_url'::regclass
    ) THEN
        ALTER TABLE found_url
        ADD CONSTRAINT uq_found_url_url UNIQUE (url);
    END IF;
END$$;

-- Add missing columns for forward-compat
ALTER TABLE found_url
    ADD COLUMN IF NOT EXISTS language TEXT,
    ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS found_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_found_url_found_at ON found_url(found_at);
CREATE INDEX IF NOT EXISTS idx_found_url_feed_id  ON found_url(feed_id);
"""

def bootstrap_schema_and_seed():
    with get_conn() as conn:
        exec_sql(conn, SCHEMA_SQL)

        # Seed feeds (upsert on URL)
        with conn.cursor() as cur:
            for name, url, days in SEED_FEEDS:
                cur.execute("""
                    INSERT INTO source_feed (url, name, retain_days, is_active)
                    VALUES (%s, %s, %s, TRUE)
                    ON CONFLICT (url) DO UPDATE
                    SET name = EXCLUDED.name,
                        retain_days = COALESCE(EXCLUDED.retain_days, source_feed.retain_days),
                        is_active = TRUE
                """, (url, name, days))
        LOG.info("Schema ensured; seed feeds upserted.")

# -----------------------------------------------------------------------------
# SMTP / Email helpers
# -----------------------------------------------------------------------------
def send_smtp_email(subject: str, html: str, text: Optional[str] = None, to_addrs: Optional[list[str]] = None) -> bool:
    recips = to_addrs or [a.strip() for a in (DIGEST_TO or "").split(",") if a.strip()]
    if not recips:
        LOG.warning("No recipients configured. Set DIGEST_TO or EMAIL_TO (comma-separated).")
        return False
    if not (SMTP_HOST and SMTP_FROM):
        LOG.error("SMTP not configured. Need SMTP_HOST and EMAIL_FROM/SMTP_FROM.")
        return False

    if text is None:
        text = re.sub(r"<[^>]+>", "", html)

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

def build_digest(minutes: int) -> tuple[str, str, str, int]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
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
        """, (minutes,))
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

# -----------------------------------------------------------------------------
# Ingest & Prune
# -----------------------------------------------------------------------------
def insert_found_url(cur: psycopg.Cursor, *, url: str, title: str, feed_id: Optional[int], language: Optional[str], published_at: Optional[datetime]):
    cur.execute("""
        INSERT INTO found_url (url, title, feed_id, language, published_at, found_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (url) DO UPDATE
        SET title = COALESCE(EXCLUDED.title, found_url.title),
            feed_id = COALESCE(EXCLUDED.feed_id, found_url.feed_id),
            language = COALESCE(EXCLUDED.language, found_url.language),
            published_at = COALESCE(EXCLUDED.published_at, found_url.published_at)
    """, (url, title, feed_id, language, published_at))

def parse_published(entry) -> Optional[datetime]:
    # Try rss 'published_parsed', fallback to 'updated_parsed'
    for key in ("published_parsed", "updated_parsed"):
        t = getattr(entry, key, None) or entry.get(key)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None

def ingest_once(minutes: int) -> dict:
    total_entries = 0
    inserted = 0

    with get_conn() as conn, conn.cursor() as cur:
        # read active feeds
        cur.execute("SELECT id, url, COALESCE(NULLIF(name, ''), url) AS name FROM source_feed WHERE is_active IS TRUE")
        feeds = cur.fetchall()

        for feed_id, url, name in feeds:
            parsed = feedparser.parse(url)
            entries = parsed.entries or []
            total_entries += len(entries)
            LOG.info("parsed feed: %s", url)
            LOG.info("feed entries: %d", len(entries))

            for e in entries:
                link = getattr(e, "link", None) or e.get("link") or e.get("id")
                if not link:
                    continue
                title = getattr(e, "title", None) or e.get("title") or link
                lang = getattr(e, "language", None) or parsed.get("language") or "en"
                published_at = parse_published(e)
                try:
                    insert_found_url(cur,
                                     url=link,
                                     title=title,
                                     feed_id=feed_id,
                                     language=lang,
                                     published_at=published_at)
                    inserted += 1
                except Exception as ex:
                    LOG.warning("ingest error for feed %s: %s", url, ex)

    # prune after ingest
    deleted = prune_old_found_urls()
    LOG.info("pruned %d old found_url rows (older than %d days)", deleted or 0, DEFAULT_RETENTION_DAYS)
    return {"feeds": len(feeds), "entries_seen": total_entries, "rows_upserted": inserted, "rows_pruned": deleted or 0}

def prune_old_found_urls(default_days: int = DEFAULT_RETENTION_DAYS) -> int:
    """
    Delete found_url rows older than per-feed retention, falling back to default_days.
    Returns number of rows deleted.
    """
    with get_conn() as conn, conn.cursor() as cur:
        # Delete with per-feed retention if available, otherwise default
        cur.execute("""
            WITH cutoffs AS (
                SELECT
                    f.id,
                    COALESCE(NULLIF(s.retain_days, 0), %s) AS keep_days
                FROM found_url f
                LEFT JOIN source_feed s ON s.id = f.feed_id
            )
            DELETE FROM found_url f
            USING cutoffs c
            WHERE c.id = f.id
              AND f.found_at < NOW() - make_interval(days => c.keep_days)
        """, (default_days,))
        deleted = cur.rowcount
        LOG.warning("prune_old_found_urls: deleted=%s, default_days=%s, has_retain_days=%s",
                    deleted, default_days, "exists")
        return deleted or 0

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": APP_NAME, "message": "Quantbrief is running."}

@app.post("/admin/init")
def admin_init(req: Request):
    require_admin(req)
    bootstrap_schema_and_seed()
    return {"ok": True}

@app.post("/cron/ingest")
def cron_ingest(req: Request, minutes: int = 1440):
    require_admin(req)
    result = ingest_once(minutes)
    return result

@app.post("/cron/digest")
def cron_digest(req: Request, minutes: int = 1440, to: Optional[str] = None):
    require_admin(req)
    subject, html, text, count = build_digest(minutes)
    if count == 0:
        LOG.info("Digest has no items for last %d minutes; skipping send.", minutes)
        return {"ok": True, "sent": 0, "items": 0, "skipped": "no items"}
    recips = [a.strip() for a in (to or DIGEST_TO or "").split(",") if a.strip()]
    ok = send_smtp_email(subject, html, text, recips)
    return {"ok": bool(ok), "sent": len(recips), "items": count}

@app.post("/admin/test-email")
def admin_test_email(req: Request, to: Optional[str] = None):
    require_admin(req)
    recips = [a.strip() for a in (to or DIGEST_TO or "").split(",") if a.strip()]
    ok = send_smtp_email("Quantbrief test email", "<p>Hello from Quantbrief!</p>", "Hello from Quantbrief!", recips)
    return {"ok": bool(ok), "to": recips}

# -----------------------------------------------------------------------------
# Uvicorn entrypoint (Render uses the start command, but this helps local dev)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
