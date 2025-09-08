import os
import logging
from datetime import datetime, timezone
from typing import Optional, List

import feedparser
from fastapi import FastAPI
from pydantic import BaseModel
import psycopg
from psycopg.rows import dict_row

LOG = logging.getLogger("quantbrief")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or os.getenv("DATABASE_CONNECTION_STRING")

app = FastAPI(title="quantbrief")

# ---------- DB helpers ----------

def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL env var is required")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def exec_sql_batch(conn, sql: str):
    """Run a big SQL batch with DO $$ blocks etc., committing or raising."""
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()

# ---------- Schema (idempotent) ----------

SCHEMA_SQL = """
-- Feeds table
CREATE TABLE IF NOT EXISTS source_feed (
    id            BIGSERIAL PRIMARY KEY,
    url           TEXT NOT NULL,
    name          TEXT,
    active        BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Enforce uniqueness with a unique index only (idempotent & safe)
CREATE UNIQUE INDEX IF NOT EXISTS uq_source_feed_url ON source_feed (url);

-- Found URLs table
CREATE TABLE IF NOT EXISTS found_url (
    id            BIGSERIAL PRIMARY KEY,
    url           TEXT NOT NULL,
    title         TEXT,
    feed_id       BIGINT REFERENCES source_feed(id) ON DELETE CASCADE,
    language      TEXT,
    published_at  TIMESTAMPTZ,
    found_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Backfill column if old table existed without it
ALTER TABLE found_url
    ADD COLUMN IF NOT EXISTS found_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

-- Helpful indexes
CREATE INDEX IF NOT EXISTS ix_found_url_feed_id ON found_url(feed_id);
CREATE INDEX IF NOT EXISTS ix_found_url_found_at ON found_url(found_at);
"""

# ---------- Models ----------

class AdminInitResult(BaseModel):
    status: str

class IngestResult(BaseModel):
    feeds_processed: int
    items_inserted: int
    items_skipped: int
    errors: int

# ---------- Utilities ----------

def prune_old_found_urls(retention_days: int = 30) -> int:
    """Delete found_url rows older than N days."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM found_url
            WHERE found_at < NOW() - ($1 || ' days')::INTERVAL
            """,
            (retention_days,),
        )
        deleted = cur.rowcount or 0
        conn.commit()
        return deleted

def upsert_source_feed(url: str, name: Optional[str] = None) -> int:
    """Ensure a feed exists; return its id."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM source_feed WHERE url = %s", (url,))
        row = cur.fetchone()
        if row:
            return row["id"]

        cur.execute(
            "INSERT INTO source_feed (url, name) VALUES (%s, %s) RETURNING id",
            (url, name),
        )
        feed_id = cur.fetchone()["id"]
        conn.commit()
        return feed_id

def insert_found_item(
    url: str,
    title: Optional[str],
    feed_id: int,
    language: Optional[str],
    published_at: Optional[datetime],
    found_at: Optional[datetime] = None,
) -> bool:
    """Insert one found item; returns True if inserted, False if duplicate."""
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                """
                INSERT INTO found_url (url, title, feed_id, language, published_at, found_at)
                VALUES (%s, %s, %s, %s, %s, COALESCE(%s, NOW()))
                ON CONFLICT DO NOTHING
                """,
                (url, title, feed_id, language, published_at, found_at),
            )
            inserted = cur.rowcount == 1
            conn.commit()
            return inserted
        except Exception as e:
            conn.rollback()
            LOG.warning("insert_found_item failed for %s: %s", url, e)
            raise

def parse_datetime(dt) -> Optional[datetime]:
    if not dt:
        return None
    if isinstance(dt, datetime):
        return dt
    # feedparser gives .published_parsed (time.struct_time)
    try:
        return datetime(*dt[:6], tzinfo=timezone.utc)
    except Exception:
        return None

# ---------- Endpoints ----------

@app.post("/admin/init", response_model=AdminInitResult)
def admin_init():
    with get_conn() as conn:
        exec_sql_batch(conn, SCHEMA_SQL)
    return AdminInitResult(status="ok")

@app.post("/cron/ingest", response_model=IngestResult)
def cron_ingest(retention_days: int = int(os.getenv("RETENTION_DAYS", "30"))):
    """
    Parse all active feeds in source_feed and store found items in found_url.
    Rolls back per-feed on failure to avoid 'current transaction is aborted'.
    """
    feeds_processed = 0
    items_inserted = 0
    items_skipped = 0
    errors = 0

    # Safety: ensure schema is in place (idempotent)
    with get_conn() as conn:
        exec_sql_batch(conn, SCHEMA_SQL)

    feeds: List[dict]
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, url, COALESCE(name, url) AS name FROM source_feed WHERE active = TRUE")
        feeds = cur.fetchall()

    for feed in feeds:
        feeds_processed += 1
        feed_id = feed["id"]
        feed_url = feed["url"]

        LOG.info("parsed feed: %s", feed_url)
        try:
            parsed = feedparser.parse(feed_url)
            entries = parsed.entries or []
            LOG.info("feed entries: %s", len(entries))

            for e in entries:
                link = getattr(e, "link", None) or getattr(e, "id", None)
                if not link:
                    items_skipped += 1
                    continue

                title = getattr(e, "title", None)
                lang = getattr(parsed.feed, "language", None) or getattr(e, "language", None)
                published = parse_datetime(getattr(e, "published_parsed", None)) or parse_datetime(getattr(e, "updated_parsed", None))

                try:
                    if insert_found_item(link, title, feed_id, lang, published):
                        items_inserted += 1
                    else:
                        items_skipped += 1
                except Exception:
                    errors += 1
                    # continue with other items

        except Exception as e:
            LOG.warning("ingest error for feed %s: %s", feed_url, e)
            errors += 1
            # continue with other feeds

    # Prune old items
    try:
        deleted = prune_old_found_urls(retention_days=retention_days)
        LOG.info("pruned %d old found_url rows (older than %d days)", deleted, retention_days)
    except Exception as e:
        LOG.warning("prune_old_found_urls failed: %s", e)
        errors += 1

    return IngestResult(
        feeds_processed=feeds_processed,
        items_inserted=items_inserted,
        items_skipped=items_skipped,
        errors=errors,
    )
