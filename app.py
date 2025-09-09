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
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse

# -----------------------------------------------------------------------------
# Config & logging
# -----------------------------------------------------------------------------
APP_NAME = os.getenv("APP_NAME", "quantbrief")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(APP_NAME)

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    logger.warning("DATABASE_URL not set; app will start but DB calls will fail.")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587") or "587")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
MAIL_FROM = os.getenv("MAIL_FROM")
MAIL_TO = [e.strip() for e in os.getenv("MAIL_TO", "").split(",") if e.strip()]

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title="Quantbrief Daily", version="1.1.0")


def get_conn() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, autocommit=False, row_factory=dict_row)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# -----------------------------------------------------------------------------
# Schema, migrations, and compatibility (all idempotent)
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
    # Canonical feeds table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feed (
            id          BIGSERIAL PRIMARY KEY,
            name        TEXT NOT NULL,
            url         TEXT NOT NULL UNIQUE,
            active      BOOLEAN NOT NULL DEFAULT TRUE,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

    # Legacy table retained for compat/mapping; don't enforce strict NOT NULL here
    cur.execute("""
        CREATE TABLE IF NOT EXISTS source_feed (
            id          BIGINT PRIMARY KEY,
            url         TEXT UNIQUE,
            name        TEXT,
            active      BOOLEAN,
            created_at  TIMESTAMPTZ DEFAULT now()
        );
    """)

    # Articles / links
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
    # Add columns that might be missing in older DBs
    cur.execute("ALTER TABLE feed       ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE;")
    cur.execute("ALTER TABLE feed       ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();")
    cur.execute("ALTER TABLE feed       ADD COLUMN IF NOT EXISTS name TEXT;")
    cur.execute("UPDATE feed SET name = COALESCE(name, 'feed');")

    cur.execute("ALTER TABLE found_url  ADD COLUMN IF NOT EXISTS slug TEXT;")
    cur.execute("ALTER TABLE found_url  ADD COLUMN IF NOT EXISTS score DOUBLE PRECISION;")
    cur.execute("ALTER TABLE found_url  ADD COLUMN IF NOT EXISTS fingerprint TEXT;")
    cur.execute("ALTER TABLE found_url  ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();")

    # indexes commonly used
    cur.execute("CREATE INDEX IF NOT EXISTS idx_found_url_published_at ON found_url(published_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_found_url_feed_id ON found_url(feed_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_found_url_url ON found_url(url);")


def _drop_any_fk_on_found_url_feed(cur) -> None:
    """
    Drop ANY FK that targets found_url(feed_id) — regardless of what it references.
    This clears old 'found_url.feed_id -> source_feed.id' constraints.
    """
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


def _ensure_unique_fingerprint_constraint(cur) -> None:
    """
    Ensure a NAMED UNIQUE CONSTRAINT (not just a partial index) so we can use:
      ON CONFLICT ON CONSTRAINT ux_found_url_fingerprint DO NOTHING
    Multi-NULLs are allowed by Postgres UNIQUE, so no need for partial index.
    """
    cur.execute("""
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_name='found_url'
          AND constraint_type='UNIQUE'
          AND constraint_name='ux_found_url_fingerprint'
      ) THEN
        EXECUTE 'ALTER TABLE found_url
                 ADD CONSTRAINT ux_found_url_fingerprint
                 UNIQUE (fingerprint)';
      END IF;
    END $$;
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
    """
    Make sure every URL present in legacy source_feed has a canonical row in feed.
    """
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
    """
    Map found_url.feed_id from legacy source_feed.id to canonical feed.id by matching URL.
    Then null out any remaining orphans that still don't match a feed row.
    """
    # 1) Remap using URL join (legacy id -> canonical id)
    cur.execute("""
        UPDATE found_url fu
           SET feed_id = f.id
          FROM source_feed sf
          JOIN feed f ON f.url = sf.url
         WHERE fu.feed_id = sf.id
           AND fu.feed_id IS NOT NULL
           AND fu.feed_id <> f.id;
    """)

    # 2) For any remaining non-null feed_id that doesn't match a feed row, null it
    cur.execute("""
        UPDATE found_url fu
           SET feed_id = NULL
         WHERE fu.feed_id IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM feed f WHERE f.id = fu.feed_id);
    """)


def _add_fk_found_url_to_feed(cur) -> None:
    """
    Add a deferrable FK from found_url(feed_id) -> feed(id), if missing.
    """
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

        # Critical: drop any legacy FK before we touch feed_id values
        _drop_any_fk_on_found_url_feed(cur)

        # Ensure we have a named unique constraint for fingerprint upsert
        _ensure_unique_fingerprint_constraint(cur)

        # Seed canonical feeds & mirror legacy URLs into 'feed'
        _seed_feeds_by_url(cur)
        _sync_feed_from_source_feed(cur)

        # Remap legacy found_url.feed_id -> canonical feed.id and clean orphans
        _remap_found_url_feed_ids_from_source_feed(cur)

        # Re-add the FK targeting feed(id)
        _add_fk_found_url_to_feed(cur)

    conn.commit()
    logger.info("Schema ensured; seed feeds upserted.")


# -----------------------------------------------------------------------------
# Ingestion utilities
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


def fingerprint_for(url_canonical: str) -> Optional[str]:
    if not url_canonical:
        return None
    import hashlib
    return hashlib.sha256(url_canonical.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Insert helpers
# -----------------------------------------------------------------------------
def insert_found_url(conn, row: Dict[str, Any]) -> bool:
    sql = """
        INSERT INTO found_url
          (url, url_canonical, host, title, slug, summary, published_at, score, feed_id, fingerprint)
        VALUES
          (%(url)s, %(url_canonical)s, %(host)s, %(title)s, %(slug)s, %(summary)s, %(published_at)s, %(score)s, %(feed_id)s, %(fingerprint)s)
        ON CONFLICT ON CONSTRAINT ux_found_url_fingerprint DO NOTHING
        RETURNING id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, row)
        got = cur.fetchone()
    return bool(got)


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
                    "fingerprint": fingerprint_for(can),
                }

                try:
                    if insert_found_url(conn, row):
                        added += 1
                except Exception as ex:
                    logger.exception(f"Failed insert for {link}: {ex}")

        conn.commit()

    return {"inserted": added, "scanned": scanned, "pruned": pruned}


# -----------------------------------------------------------------------------
# Email
# -----------------------------------------------------------------------------
def format_email_html(subject: str, rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    html_items = []
    text_items = []
    for r in rows:
        _id = r["id"]
        url = r["url"]
        url_canonical = r["url_canonical"]
        host = r["host"]
        title = r["title"]
        _slug = r["slug"]
        _summary = r["summary"]
        published_at = r["published_at"]
        score = r["score"]
        _feed_id = r["feed_id"]
        _fp = r["fingerprint"]

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
    return "<h3>Quantbrief Daily is live ✔</h3>"

@app.post("/admin/init")
def admin_init():
    with get_conn() as conn:
        ensure_schema_and_seed(conn)
    return JSONResponse({"ok": True, "message": "Schema ensured; seed feeds upserted."})

@app.post("/cron/ingest")
def cron_ingest(minutes: int = Query(7 * 24 * 60, ge=1, le=60 * 24 * 30)):
    res = do_ingest(minutes=minutes)
    return JSONResponse({"ok": True, **res})

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
    return JSONResponse({"ok": True, "count": len(rows)})

@app.post("/admin/test-email")
def admin_test_email():
    subject = f"{APP_NAME} test email"
    html_body = "<b>Test</b> email body"
    text_body = "Test email body"
    send_email(subject, html_body, text_body)
    return JSONResponse({"ok": True, "message": "Test email attempted (check logs if not configured)."})
