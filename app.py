import os
import sys
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

import feedparser
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
LOG = logging.getLogger("quantbrief")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s:quantbrief:%(message)s"))
if not LOG.handlers:
    LOG.addHandler(handler)

# ------------------------------------------------------------------------------
# Config / Environment
# ------------------------------------------------------------------------------
APP = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    LOG.warning("DATABASE_URL is not set; the app will not be able to use Postgres.")

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme-admin-token")

# Prefer Mailgun envs if present; otherwise fall back to generic SMTP_*
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

EMAIL_FROM = _first(os.getenv("MAILGUN_FROM"), os.getenv("EMAIL_FROM"), os.getenv("SMTP_FROM"), SMTP_USERNAME)
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
DIGEST_TO = _first(os.getenv("DIGEST_TO"), ADMIN_EMAIL)

DEFAULT_RETAIN_DAYS = int(os.getenv("DEFAULT_RETAIN_DAYS", "30"))

# ------------------------------------------------------------------------------
# Permanent schema / migrations (idempotent)
# ------------------------------------------------------------------------------
SCHEMA_SQL = r"""
-- 1) Create tables if missing
CREATE TABLE IF NOT EXISTS source_feed (
  id           BIGSERIAL PRIMARY KEY,
  url          TEXT NOT NULL,
  name         TEXT,
  active       BOOLEAN NOT NULL DEFAULT TRUE,
  retain_days  INTEGER,
  language     TEXT NOT NULL DEFAULT 'en',
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS found_url (
  id           BIGSERIAL PRIMARY KEY,
  url          TEXT NOT NULL,
  title        TEXT,
  feed_id      BIGINT REFERENCES source_feed(id) ON DELETE CASCADE,
  language     TEXT NOT NULL DEFAULT 'en',
  published_at TIMESTAMPTZ,
  found_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2) Ensure columns that older deploys may lack (safe add)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name='source_feed' AND column_name='created_at') THEN
    ALTER TABLE source_feed ADD COLUMN created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name='source_feed' AND column_name='updated_at') THEN
    ALTER TABLE source_feed ADD COLUMN updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name='source_feed' AND column_name='active') THEN
    ALTER TABLE source_feed ADD COLUMN active BOOLEAN NOT NULL DEFAULT TRUE;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name='source_feed' AND column_name='retain_days') THEN
    ALTER TABLE source_feed ADD COLUMN retain_days INTEGER;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name='source_feed' AND column_name='language') THEN
    ALTER TABLE source_feed ADD COLUMN language TEXT;
    UPDATE source_feed SET language='en' WHERE language IS NULL;
    ALTER TABLE source_feed ALTER COLUMN language SET DEFAULT 'en';
    ALTER TABLE source_feed ALTER COLUMN language SET NOT NULL;
  END IF;

  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name='found_url' AND column_name='found_at') THEN
    ALTER TABLE found_url ADD COLUMN found_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name='found_url' AND column_name='language') THEN
    ALTER TABLE found_url ADD COLUMN language TEXT NOT NULL DEFAULT 'en';
  END IF;
END $$;

-- 3) Unique constraints / indexes (guarded)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='uq_source_feed_url') THEN
    ALTER TABLE source_feed ADD CONSTRAINT uq_source_feed_url UNIQUE (url);
  END IF;
END $$;

-- Optionally dedupe found_url by URL across feeds. If you prefer per-feed duplicates,
-- comment out this block.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='uq_found_url_url') THEN
    ALTER TABLE found_url ADD CONSTRAINT uq_found_url_url UNIQUE (url);
  END IF;
END $$;

-- Helpful index for pruning / digest queries
CREATE INDEX IF NOT EXISTS idx_found_url_feed_foundat ON found_url(feed_id, found_at DESC);
CREATE INDEX IF NOT EXISTS idx_found_url_pub ON found_url(published_at DESC);

-- 4) updated_at trigger on source_feed (idempotent)
CREATE OR REPLACE FUNCTION set_source_feed_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at := NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger
    WHERE tgname = 'tr_source_feed_updated_at'
  ) THEN
    CREATE TRIGGER tr_source_feed_updated_at
    BEFORE UPDATE ON source_feed
    FOR EACH ROW EXECUTE FUNCTION set_source_feed_updated_at();
  END IF;
END $$;
"""

# ------------------------------------------------------------------------------
# Seed feeds (idempotent via upsert)
# ------------------------------------------------------------------------------
SEED_FEEDS = [
    {
        "url": "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+-site:newser.com+-site:marketbeat.com+when:3d&hl=en-US&gl=US&ceid=US:en",
        "name": "Google News: Talen Energy OR TLN (3d, filtered)",
        "language": "en",
        "retain_days": 30,
    },
    {
        "url": "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN&hl=en-US&gl=US&ceid=US:en",
        "name": "Google News: Talen Energy OR TLN (all)",
        "language": "en",
        "retain_days": 30,
    },
    {
        "url": "https://news.google.com/rss/search?q=%28Talen+Energy%29+OR+TLN+-site%3Anewser.com+-site%3Awww.marketbeat.com+-site%3Amarketbeat.com+-site%3Awww.newser.com+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen",
        "name": "Google News: Talen Energy OR TLN (7d, filtered A)",
        "language": "en",
        "retain_days": 30,
    },
    {
        "url": "https://news.google.com/rss/search?q=%28%28Talen+Energy%29+OR+TLN%29+-site%3Anewser.com+-site%3Awww.newser.com+-site%3Amarketbeat.com+-site%3Awww.marketbeat.com+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen",
        "name": "Google News: ((Talen Energy) OR TLN) (7d, filtered B)",
        "language": "en",
        "retain_days": 30,
    },
]

# ------------------------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------------------------
@contextmanager
def db() -> psycopg.Connection:
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


def exec_sql_batch(conn: psycopg.Connection, sql: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql)


def ensure_schema() -> None:
    with db() as conn:
        exec_sql_batch(conn, SCHEMA_SQL)


def upsert_feed(
    url: str,
    name: Optional[str] = None,
    language: Optional[str] = None,
    retain_days: Optional[int] = None,
    active: Optional[bool] = True,
) -> int:
    ensure_schema()
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO source_feed (url, name, language, retain_days, active)
            VALUES (%s, COALESCE(%s,''), COALESCE(NULLIF(%s,''),'en'), %s, COALESCE(%s, TRUE))
            ON CONFLICT (url) DO UPDATE
              SET name = EXCLUDED.name,
                  language = EXCLUDED.language,
                  retain_days = EXCLUDED.retain_days,
                  active = EXCLUDED.active
            RETURNING id;
            """,
            (url, name, language, retain_days, active),
        )
        row = cur.fetchone()
        return int(row["id"])


def list_active_feeds() -> List[Dict[str, Any]]:
    ensure_schema()
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, url,
                   COALESCE(NULLIF(name,''), url) AS name,
                   COALESCE(NULLIF(language,''), 'en') AS language,
                   retain_days,
                   active
            FROM source_feed
            WHERE active IS TRUE
            ORDER BY id ASC
            """
        )
        return list(cur.fetchall())


def insert_found_url(
    url: str,
    title: Optional[str],
    feed_id: int,
    language: Optional[str],
    published_at: Optional[datetime],
    found_at: Optional[datetime] = None,
) -> None:
    ensure_schema()
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO found_url (url, title, feed_id, language, published_at, found_at)
            VALUES (%s, %s, %s, COALESCE(NULLIF(%s,''),'en'), %s, COALESCE(%s, NOW()))
            ON CONFLICT (url) DO NOTHING;
            """,
            (url, title, feed_id, language, published_at, found_at),
        )


def prune_old_found_urls(default_days: int = DEFAULT_RETAIN_DAYS) -> int:
    """
    Deletes rows older than per-feed retain_days; if NULL/0, falls back to `default_days`.
    Returns number of rows deleted.
    """
    ensure_schema()
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH del AS (
              DELETE FROM found_url f
              USING source_feed s
              WHERE s.id = f.feed_id
                AND f.found_at < NOW() - make_interval(days => CASE
                      WHEN s.retain_days IS NULL OR s.retain_days = 0 THEN %s
                      ELSE s.retain_days
                    END)
              RETURNING f.id
            )
            SELECT COUNT(*) AS deleted FROM del;
            """,
            (default_days,),
        )
        row = cur.fetchone()
        deleted = int(row["deleted"]) if row and row["deleted"] is not None else 0
        LOG.warning("prune_old_found_urls: deleted=%d, default_days=%d, has_retain_days=exists", deleted, default_days)
        return deleted


# ------------------------------------------------------------------------------
# Feed ingest
# ------------------------------------------------------------------------------
def parse_datetime(candidate) -> Optional[datetime]:
    """Try to normalize times from feedparser entries."""
    if not candidate:
        return None
    if isinstance(candidate, datetime):
        if candidate.tzinfo is None:
            return candidate.replace(tzinfo=timezone.utc)
        return candidate
    # feedparser provides .published_parsed / .updated_parsed as time.struct_time
    try:
        if hasattr(candidate, "tm_year"):
            return datetime.fromtimestamp(time.mktime(candidate), tz=timezone.utc)
    except Exception:
        pass
    # Fallback: try to parse string-ish
    try:
        return datetime.fromisoformat(str(candidate))
    except Exception:
        return None


def ingest_one_feed(feed: Dict[str, Any]) -> int:
    """
    Parse RSS and insert new links. Returns number of inserted (attempted) rows.
    """
    fid = int(feed["id"])
    furl = feed["url"]
    flang = feed.get("language") or "en"

    parsed = feedparser.parse(furl)
    LOG.info("parsed feed: %s", furl)
    LOG.info("feed entries: %d", len(parsed.entries))

    inserted = 0
    for e in parsed.entries:
        url = getattr(e, "link", None)
        if not url:
            continue
        title = getattr(e, "title", None)

        published_at = None
        # Prefer published, then updated
        if hasattr(e, "published_parsed") and e.published_parsed:
            published_at = parse_datetime(e.published_parsed)
        elif hasattr(e, "updated_parsed") and e.updated_parsed:
            published_at = parse_datetime(e.updated_parsed)

        try:
            insert_found_url(
                url=url,
                title=title,
                feed_id=fid,
                language=flang,
                published_at=published_at,
            )
            inserted += 1
        except Exception as ex:
            # Ignore duplicates etc., but log
            LOG.warning("ingest error for url %s: %s", url, str(ex))

    return inserted


# ------------------------------------------------------------------------------
# Email
# ------------------------------------------------------------------------------
def smtp_ready() -> bool:
    return bool(SMTP_HOST and SMTP_PORT and SMTP_USERNAME and SMTP_PASSWORD and EMAIL_FROM and DIGEST_TO)


def send_email(subject: str, html: str, text: Optional[str] = None, to: Optional[str] = None) -> None:
    if not smtp_ready():
        LOG.error("SMTP not configured. Need SMTP_HOST and EMAIL_FROM/SMTP_FROM.")
        return

    recipients = [to or DIGEST_TO]
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(recipients)

    if not text:
        # Simple plaintext fallback
        text = "Your email client does not support HTML. See HTML version."
    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        if SMTP_STARTTLS:
            s.starttls()
        s.login(SMTP_USERNAME, SMTP_PASSWORD)
        s.sendmail(EMAIL_FROM, recipients, msg.as_string())


def build_digest_html(rows: List[Dict[str, Any]], minutes: int) -> str:
    h = []
    h.append(f"<h2>Quantbrief digest — last {minutes} minutes</h2>")
    h.append("<ul>")
    for r in rows:
        title = r.get("title") or r.get("url")
        published = r.get("published_at")
        pub_s = published.strftime("%Y-%m-%d %H:%M") if isinstance(published, datetime) else ""
        h.append(f"<li><a href='{r['url']}'>{title}</a>"
                 f" <small>({pub_s})</small>"
                 f" <em>— {r.get('feed_name','')}</em></li>")
    h.append("</ul>")
    return "\n".join(h)


def fetch_recent_items(minutes: int) -> List[Dict[str, Any]]:
    ensure_schema()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.url, f.title, f.published_at, sf.name AS feed_name
            FROM found_url f
            JOIN source_feed sf ON sf.id = f.feed_id
            WHERE f.found_at >= %s
            ORDER BY COALESCE(f.published_at, f.found_at) DESC
            """,
            (cutoff,),
        )
        return list(cur.fetchall())


# ------------------------------------------------------------------------------
# Auth helper
# ------------------------------------------------------------------------------
def require_admin(req: Request) -> None:
    token = req.headers.get("x-admin-token")
    if not token:
        auth = req.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth.split(None, 1)[1]
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@APP.get("/")
def root():
    return PlainTextResponse("Quantbrief Daily — OK")


@APP.post("/admin/init")
def admin_init(request: Request):
    require_admin(request)
    ensure_schema()
    # Seed feeds
    ids = []
    for f in SEED_FEEDS:
        fid = upsert_feed(
            url=f["url"],
            name=f.get("name"),
            language=f.get("language", "en"),
            retain_days=f.get("retain_days"),
            active=True,
        )
        ids.append(fid)
    LOG.info("Schema ensured; seed feeds upserted.")
    return JSONResponse({"ok": True, "feed_ids": ids})


@APP.post("/cron/ingest")
def cron_ingest(request: Request, minutes: Optional[int] = 60 * 24 * 7):
    require_admin(request)
    ensure_schema()
    feeds = list_active_feeds()

    total_inserted = 0
    for f in feeds:
        try:
            total_inserted += ingest_one_feed(f)
        except Exception as ex:
            LOG.warning("ingest error for feed %s: %s", f.get("url"), str(ex))

    deleted = prune_old_found_urls(DEFAULT_RETAIN_DAYS)
    LOG.info("pruned %d old found_url rows (older than %d days)", deleted, DEFAULT_RETAIN_DAYS)
    return JSONResponse({"ok": True, "inserted": total_inserted, "deleted": deleted})


@APP.post("/cron/digest")
def cron_digest(request: Request, minutes: Optional[int] = 60 * 24 * 7):
    require_admin(request)
    ensure_schema()
    rows = fetch_recent_items(minutes or 1440)
    html = build_digest_html(rows, minutes or 1440)
    send_email(
        subject=f"Quantbrief digest — last {minutes} minutes ({len(rows)} items)",
        html=html,
        text=None,
        to=DIGEST_TO,
    )
    return JSONResponse({"ok": True, "items": len(rows), "to": DIGEST_TO})


@APP.post("/admin/test-email")
def admin_test_email(request: Request):
    require_admin(request)
    ensure_schema()
    send_email("Quantbrief test email", "<p>Hello from Quantbrief!</p>", "Hello from Quantbrief!", to=DIGEST_TO)
    return JSONResponse({"ok": True, "to": DIGEST_TO})


# ------------------------------------------------------------------------------
# Local run (optional)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(APP, host="0.0.0.0", port=port)
