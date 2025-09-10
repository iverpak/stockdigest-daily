# app.py
import os
import re
import ssl
import smtplib
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import psycopg
from psycopg.rows import dict_row
from psycopg import errors as pg_errors
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse

APP_NAME = os.getenv("APP_NAME", "quantbrief")
BUILD_TAG = "quantbrief build 2025-09-10T00:58Z"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(APP_NAME)
logger.info(BUILD_TAG)

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    logger.warning("DATABASE_URL not set; app will start but DB calls will fail.")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587") or "587")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
MAIL_FROM = os.getenv("MAIL_FROM")
MAIL_TO = [e.strip() for e in os.getenv("MAIL_TO", "").split(",") if e.strip()]

app = FastAPI(title="Quantbrief Daily", version="1.1.6")

def get_conn() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, autocommit=False, row_factory=dict_row)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

# -----------------------------------------------------------------------------
# Schema & migrations (idempotent)
# -----------------------------------------------------------------------------
FEED_SEED: List[Tuple[str, str, bool]] = [
    (
        "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+-site:newser.com+-site:marketbeat.com+when:3d&hl=en-US&gl=US&ceid=US:en",
        "Talen TLN (3d, filtered)",
        True,
    ),
    (
        "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+when:7d&hl=en-US&gl=US&ceid=US:en",
        "Talen TLN (7d)",
        True,
    ),
]

def _ensure_table_shells(cur) -> None:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feed (
            id          BIGSERIAL PRIMARY KEY,
            name        TEXT NOT NULL,
            url         TEXT NOT NULL UNIQUE,
            active      BOOLEAN NOT NULL DEFAULT TRUE,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS source_feed (
            id          BIGINT PRIMARY KEY,
            url         TEXT UNIQUE,
            name        TEXT,
            active      BOOLEAN,
            created_at  TIMESTAMPTZ DEFAULT now()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS found_url (
            id            BIGSERIAL PRIMARY KEY,
            url           TEXT NOT NULL,
            url_canonical TEXT,
            host          TEXT,
            title         TEXT,
            slug          TEXT,
            summary       TEXT,
            published_at  TIMESTAMPTZ,
            score         DOUBLE PRECISION,
            feed_id       BIGINT,
            fingerprint   TEXT,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

def _ensure_columns(cur) -> None:
    cur.execute("ALTER TABLE feed ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE;")
    cur.execute("ALTER TABLE feed ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();")
    cur.execute("ALTER TABLE feed ADD COLUMN IF NOT EXISTS name TEXT;")
    cur.execute("UPDATE feed SET name = COALESCE(name, 'feed');")

    cur.execute("ALTER TABLE found_url  ADD COLUMN IF NOT EXISTS slug TEXT;")
    cur.execute("ALTER TABLE found_url  ADD COLUMN IF NOT EXISTS score DOUBLE PRECISION;")
    cur.execute("ALTER TABLE found_url  ADD COLUMN IF NOT EXISTS fingerprint TEXT;")
    cur.execute("ALTER TABLE found_url  ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_found_url_published_at ON found_url(published_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_found_url_feed_id ON found_url(feed_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_found_url_url ON found_url(url);")

def _drop_any_fk_on_found_url_feed(cur) -> None:
    cur.execute("""
    DO $$
    DECLARE r RECORD;
    BEGIN
      FOR r IN
        SELECT tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON kcu.constraint_name = tc.constraint_name
         AND kcu.table_name = tc.table_name
        WHERE tc.table_name = 'found_url'
          AND tc.constraint_type = 'FOREIGN KEY'
          AND kcu.column_name = 'feed_id'
      LOOP
        EXECUTE format('ALTER TABLE found_url DROP CONSTRAINT %I', r.constraint_name);
      END LOOP;
    END $$;
    """)

def _drop_fks_referencing_found_url_url(cur) -> None:
    """
    Drop ANY foreign key in any table that references found_url(url).
    This clears dependencies so we can drop UNIQUE(url).
    """
    cur.execute("""
    DO $$
    DECLARE r RECORD;
    BEGIN
      FOR r IN
        SELECT n.nspname AS schemaname,
               c.relname AS tablename,
               con.conname AS constraint_name
        FROM pg_constraint con
        JOIN pg_class c ON c.oid = con.conrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE con.contype = 'f'
          AND con.confrelid = 'public.found_url'::regclass
          AND con.confkey = ARRAY[
            (SELECT attnum FROM pg_attribute
              WHERE attrelid = 'public.found_url'::regclass
                AND attname = 'url')
          ]
      LOOP
        EXECUTE format('ALTER TABLE %I.%I DROP CONSTRAINT IF EXISTS %I', r.schemaname, r.tablename, r.constraint_name);
      END LOOP;
    END $$;
    """)

def _drop_unique_url_constraint_and_indexes(cur) -> None:
    # 1) Drop any dependent FKs to found_url(url)
    _drop_fks_referencing_found_url_url(cur)

    # 2) Drop ANY UNIQUE constraint that targets found_url(url) (not just one name), CASCADE to remove its backing index.
    cur.execute("""
    DO $$
    DECLARE r RECORD;
    BEGIN
      FOR r IN
        SELECT tc.constraint_name
          FROM information_schema.table_constraints tc
          JOIN information_schema.constraint_column_usage ccu
            ON tc.constraint_name = ccu.constraint_name
           AND tc.table_schema   = ccu.table_schema
         WHERE tc.table_schema='public'
           AND tc.table_name='found_url'
           AND tc.constraint_type='UNIQUE'
           AND ccu.column_name='url'
      LOOP
        EXECUTE format('ALTER TABLE public.found_url DROP CONSTRAINT IF EXISTS %I CASCADE', r.constraint_name);
      END LOOP;
    END $$;
    """)

    # 3) If any standalone UNIQUE index on (url) still exists, drop it.
    cur.execute("""
    DO $$
    DECLARE r RECORD;
    BEGIN
      FOR r IN
        SELECT indexname, indexdef
          FROM pg_indexes
         WHERE schemaname = 'public'
           AND tablename  = 'found_url'
      LOOP
        IF r.indexdef ILIKE '%%UNIQUE%%(url%%' THEN
          EXECUTE format('DROP INDEX IF EXISTS %I', r.indexname);
        END IF;
      END LOOP;
    END $$;
    """)

    # 4) Ensure a plain (non-unique) index for performance.
    cur.execute("CREATE INDEX IF NOT EXISTS idx_found_url_url ON found_url(url);")

def _migrate_summaries_fk_to_found_url_id(cur) -> None:
    """
    If a 'summaries' table exists and previously referenced found_url(url),
    migrate it to reference found_url(id) instead.
    """
    cur.execute("""
      SELECT COUNT(*) AS n
      FROM information_schema.tables
      WHERE table_schema='public' AND table_name='summaries';
    """)
    if (cur.fetchone() or {}).get("n", 0) == 0:
        return

    cur.execute("ALTER TABLE summaries ADD COLUMN IF NOT EXISTS found_url_id BIGINT;")
    cur.execute("""
        UPDATE summaries s
           SET found_url_id = fu.id
          FROM found_url fu
         WHERE s.found_url_id IS NULL
           AND s.url IS NOT NULL
           AND fu.url = s.url;
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_summaries_found_url_id ON summaries(found_url_id);")
    cur.execute("""
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_name='summaries'
          AND constraint_type='FOREIGN KEY'
          AND constraint_name='summaries_found_url_id_fkey'
      ) THEN
        EXECUTE '
          ALTER TABLE summaries
          ADD CONSTRAINT summaries_found_url_id_fkey
          FOREIGN KEY (found_url_id) REFERENCES found_url(id)
          ON DELETE SET NULL
          DEFERRABLE INITIALLY DEFERRED
        ';
      END IF;
    END $$;
    """)

def _dedupe_fingerprints(cur) -> None:
    cur.execute("""
        WITH ranked AS (
          SELECT id, fingerprint,
                 ROW_NUMBER() OVER (PARTITION BY fingerprint ORDER BY id) AS rn
          FROM found_url
          WHERE fingerprint IS NOT NULL
        )
        DELETE FROM found_url
        WHERE id IN (SELECT id FROM ranked WHERE rn > 1);
    """)

def _ensure_unique_fingerprint_index(cur) -> None:
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_fingerprint
        ON found_url (fingerprint);
    """)

def _seed_feeds_by_url(cur) -> None:
    cur.executemany(
        """
        INSERT INTO feed (url, name, active, created_at)
        VALUES (%s, %s, %s, now())
        ON CONFLICT (url) DO UPDATE
          SET name = EXCLUDED.name,
              active = EXCLUDED.active
        """,
        FEED_SEED,
    )

def _sync_feed_from_source_feed(cur) -> None:
    cur.execute("""
        INSERT INTO feed (url, name, active, created_at)
        SELECT sf.url,
               COALESCE(NULLIF(sf.name, ''), 'feed'),
               COALESCE(sf.active, TRUE),
               COALESCE(sf.created_at, now())
        FROM source_feed sf
        WHERE sf.url IS NOT NULL
        ON CONFLICT (url) DO UPDATE
          SET name = EXCLUDED.name,
              active = EXCLUDED.active,
              created_at = COALESCE(feed.created_at, EXCLUDED.created_at);
    """)

def _remap_found_url_feed_ids_from_source_feed(cur) -> None:
    cur.execute("""
        UPDATE found_url fu
           SET feed_id = f.id
          FROM source_feed sf
          JOIN feed f ON f.url = sf.url
         WHERE fu.feed_id = sf.id
           AND fu.feed_id IS NOT NULL
           AND fu.feed_id <> f.id;
    """)
    cur.execute("""
        UPDATE found_url fu
           SET feed_id = NULL
         WHERE fu.feed_id IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM feed f WHERE f.id = fu.feed_id);
    """)

def _add_fk_found_url_to_feed(cur) -> None:
    cur.execute("""
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1
          FROM information_schema.table_constraints
         WHERE table_name='found_url'
           AND constraint_type='FOREIGN KEY'
           AND constraint_name='found_url_feed_id_fkey'
      ) THEN
        EXECUTE '
          ALTER TABLE found_url
          ADD CONSTRAINT found_url_feed_id_fkey
          FOREIGN KEY (feed_id) REFERENCES feed(id)
          ON DELETE SET NULL
          DEFERRABLE INITIALLY DEFERRED
        ';
      END IF;
    END $$;
    """)

def ensure_schema_and_seed(conn) -> None:
    logger.info("Schema ensuring & compatibility pass…")
    with conn.cursor() as cur:
        _ensure_table_shells(cur)
        _ensure_columns(cur)

        # Temporarily drop FK on found_url.feed_id during remaps
        _drop_any_fk_on_found_url_feed(cur)

        # Remove ANY legacy UNIQUE(url) + its dependencies
        _drop_unique_url_constraint_and_indexes(cur)

        # Migrate summaries (if present) to found_url_id -> found_url(id)
        _migrate_summaries_fk_to_found_url_id(cur)

        # Fingerprint dedupe & unique index
        _dedupe_fingerprints(cur)
        _ensure_unique_fingerprint_index(cur)

        # Seed/Sync feeds and remap legacy feed ids
        _seed_feeds_by_url(cur)
        _sync_feed_from_source_feed(cur)
        _remap_found_url_feed_ids_from_source_feed(cur)

        # Re-add FK from found_url(feed_id) -> feed(id)
        _add_fk_found_url_to_feed(cur)

    conn.commit()
    logger.info("Schema ensured; seed feeds upserted.")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
UTM_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "gclid", "igshid"
}

def _strip_tracking_params(url: str) -> str:
    try:
        parts = urlsplit(url)
        q = [(k, v) for (k, v) in parse_qsl(parts.query, keep_blank_values=True) if k not in UTM_PARAMS]
        new_q = urlencode(q, doseq=True)
        return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, new_q, parts.fragment))
    except Exception:
        return url

def canonicalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    try:
        parsed = urlparse(u)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc.lower()
        path = re.sub(r"/{2,}", "/", parsed.path or "/")
        rebuilt = urlunparse((scheme, netloc, path, "", parsed.query, ""))
        return _strip_tracking_params(rebuilt)
    except Exception:
        return _strip_tracking_params(u)

def slugify(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    s = re.sub(r"[^\w\s-]", "", text).strip().lower()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:200]

def entry_published(entry) -> Optional[datetime]:
    if getattr(entry, "published_parsed", None):
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    if getattr(entry, "updated_parsed", None):
        return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
    return None

def fingerprint_for(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    import hashlib
    return hashlib.sha256(val.encode("utf-8")).hexdigest()

# -----------------------------------------------------------------------------
# Insert helper (row-scoped transaction)
# -----------------------------------------------------------------------------
def insert_found_url(conn, row: Dict[str, Any]) -> bool:
    sql = """
        INSERT INTO found_url
          (url, url_canonical, host, title, slug, summary, published_at, score, feed_id, fingerprint)
        VALUES
          (%(url)s, %(url_canonical)s, %(host)s, %(title)s, %(slug)s, %(summary)s, %(published_at)s, %(score)s, %(feed_id)s, %(fingerprint)s)
        ON CONFLICT (fingerprint) DO NOTHING
        RETURNING id;
    """
    try:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(sql, row)
                got = cur.fetchone()
        return bool(got)
    except pg_errors.UniqueViolation as ex:
        logger.debug(f"Insert skipped due to legacy unique(url) violation: {ex}")
        return False
    except Exception as ex:
        logger.exception(f"Insert failed (fingerprint={row.get('fingerprint')}): {ex}")
        return False

# -----------------------------------------------------------------------------
# Ingestion
# -----------------------------------------------------------------------------
def do_ingest(minutes: int = 7 * 24 * 60) -> Dict[str, Any]:
    since = now_utc() - timedelta(minutes=minutes)
    added = 0
    scanned = 0
    pruned = 0

    with get_conn() as conn:
        ensure_schema_and_seed(conn)

        with conn.cursor() as cur:
            cur.execute("SELECT id, url, name FROM feed WHERE active = TRUE ORDER BY id;")
            feeds = cur.fetchall()

        for f in feeds:
            feed_id = f["id"]
            feed_url = f["url"]
            logger.info(f"Parsing feed: {feed_url}")
            parsed = feedparser.parse(feed_url)
            entries = parsed.entries or []
            logger.info(f"Feed entries: {len(entries)}")

            for e in entries:
                scanned += 1
                link = getattr(e, "link", None) or ""
                title = getattr(e, "title", None)
                summary = getattr(e, "summary", None)
                pub_dt = entry_published(e)

                if pub_dt and pub_dt < since:
                    pruned += 1
                    continue

                can = canonicalize_url(link)
                host = urlparse(can).netloc.lower() if can else None
                fp_source = can or link or (title or "").strip()
                row = {
                    "url": link or can,
                    "url_canonical": can,
                    "host": host,
                    "title": title,
                    "slug": slugify(title),
                    "summary": summary,
                    "published_at": pub_dt,
                    "score": 0.0,
                    "feed_id": feed_id,
                    "fingerprint": fingerprint_for(fp_source),
                }

                if insert_found_url(conn, row):
                    added += 1

        conn.commit()

    return {"inserted": added, "scanned": scanned, "pruned": pruned}

# -----------------------------------------------------------------------------
# Email
# -----------------------------------------------------------------------------
def format_email_html(subject: str, rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    html_items = []
    text_items = []
    for r in rows:
        url = r["url"]
        url_canonical = r["url_canonical"]
        host = r["host"]
        title = r["title"]
        published_at = r["published_at"]
        disp = title or url_canonical or url
        pub_txt = published_at.isoformat() if isinstance(published_at, datetime) else ""
        html_items.append(
            f'<li><a href="{url}">{disp}</a> '
            f'<small>({host or ""})</small> <small>{pub_txt}</small></li>'
        )
        text_items.append(f"- {disp} [{host or ''}] {url} {pub_txt}")

    html = f"""<html><body>
    <h2>{subject}</h2>
    <ol>
    {''.join(html_items)}
    </ol>
    </body></html>"""

    text = subject + "\n\n" + "\n".join(text_items) + "\n"
    return html, text

def send_email(subject: str, html_body: str, text_body: str) -> None:
    if not (SMTP_HOST and SMTP_PORT and MAIL_FROM and MAIL_TO):
        logger.warning("SMTP not fully configured; skipping email send.")
        return
    msg = f"From: {MAIL_FROM}\r\n"
    msg += f"To: {', '.join(MAIL_TO)}\r\n"
    msg += f"Subject: {subject}\r\n"
    msg += 'MIME-Version: 1.0\r\n'
    msg += 'Content-Type: multipart/alternative; boundary="BOUNDARY"\r\n'
    msg += "\r\n"
    msg += "--BOUNDARY\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n"
    msg += text_body + "\r\n"
    msg += "--BOUNDARY\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
    msg += html_body + "\r\n"
    msg += "--BOUNDARY--\r\n"
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=context)
            if SMTP_USER and SMTP_PASS:
                server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(MAIL_FROM, MAIL_TO, msg.encode("utf-8"))
        logger.info(f"Email sent to {MAIL_TO} (subject='{subject}')")
    except Exception as ex:
        logger.warning(f"{APP_NAME}: test email failed: {ex}")

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return f"<h3>Quantbrief Daily is live ✔</h3><small>{BUILD_TAG}</small>"

@app.post("/admin/init")
def admin_init():
    with get_conn() as conn:
        ensure_schema_and_seed(conn)
    return JSONResponse({"ok": True, "message": "Schema ensured; seed feeds upserted.", "build": BUILD_TAG})

@app.post("/cron/ingest")
def cron_ingest(minutes: int = Query(7 * 24 * 60, ge=1, le=60 * 24 * 30)):
    res = do_ingest(minutes=minutes)
    return JSONResponse({"ok": True, **res, "build": BUILD_TAG})

@app.post("/cron/digest")
def cron_digest(minutes: int = Query(7 * 24 * 60, ge=1, le=60 * 24 * 30), limit: int = Query(25, ge=1, le=100)):
    since = now_utc() - timedelta(minutes=minutes)
    with get_conn() as conn:
        ensure_schema_and_seed(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, url, url_canonical, host, title, slug, summary, published_at, score, feed_id, fingerprint
                FROM found_url
                WHERE (published_at IS NULL OR published_at >= %s)
                ORDER BY COALESCE(published_at, created_at) DESC
                LIMIT %s
                """,
                (since, limit),
            )
            rows = cur.fetchall()

    subject = f"Quantbrief digest ({len(rows)} links)"
    html_body, text_body = format_email_html(subject, rows)
    send_email(subject, html_body, text_body)
    return JSONResponse({"ok": True, "count": len(rows), "build": BUILD_TAG})

@app.post("/admin/test-email")
def admin_test_email():
    subject = f"{APP_NAME} test email"
    html_body = "<b>Test</b> email body"
    text_body = "Test email body"
    send_email(subject, html_body, text_body)
    return JSONResponse({"ok": True, "message": "Test email attempted (check logs if not configured).", "build": BUILD_TAG})
