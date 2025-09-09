# app.py
import os
import re
import ssl
import smtplib
import logging
import textwrap
from typing import List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

from datetime import datetime, timezone, timedelta

import feedparser
import requests
import psycopg
from psycopg.rows import dict_row

from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse, JSONResponse

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
APP_NAME = os.getenv("APP_NAME", "quantbrief")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
# Render/Neon typically require TLS; psycopg v3 will negotiate, but allow override
PG_SSLMODE = os.getenv("PGSSLMODE", "prefer")

# Email settings
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", f"{APP_NAME}@no-reply.local")
# Comma-separated list of recipients
TO_EMAILS = [e.strip() for e in os.getenv("TO_EMAILS", "quantbrief.research@gmail.com").split(",") if e.strip()]

# Ingestion/dedupe tuning
DEDUPE_WINDOW_DAYS = int(os.getenv("DEDUPE_WINDOW_DAYS", "14"))
DEFAULT_MINUTES = 60 * 24 * 7  # one week
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.0"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "8.0"))
MAX_FEED_ITEMS = int(os.getenv("MAX_FEED_ITEMS", "200"))

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger(APP_NAME)
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
LOG.addHandler(handler)

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title=APP_NAME)

# -----------------------------------------------------------------------------
# DB schema & seeding
# -----------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feed (
    id           SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    url          TEXT NOT NULL UNIQUE,
    enabled      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS found_url (
    id             BIGSERIAL PRIMARY KEY,
    url            TEXT NOT NULL,
    url_canonical  TEXT,
    host           TEXT,
    title          TEXT,
    title_slug     TEXT, -- IMPORTANT: was missing before
    summary        TEXT,
    published_at   TIMESTAMPTZ,
    score          DOUBLE PRECISION DEFAULT 0.0,
    feed_id        INTEGER REFERENCES feed(id) ON DELETE SET NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Safety/uniqueness helpers
CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_url ON found_url(url);
CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_url_canonical ON found_url(url_canonical) WHERE url_canonical IS NOT NULL;

-- Dedupe accelerator: same site + slug in a recent window
CREATE INDEX IF NOT EXISTS ix_found_url_host_slug_time ON found_url (host, title_slug, published_at DESC);

-- Time queries
CREATE INDEX IF NOT EXISTS ix_found_url_published_at ON found_url(published_at DESC);
"""

SEED_FEEDS: List[Tuple[str, str]] = [
    (
        "Google News — Talen Energy (3d window, filtered)",
        "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+-site:newser.com+-site:marketbeat.com+when:3d&hl=en-US&gl=US&ceid=US:en",
    ),
    # You can add more seeds here if you like:
    # ("Example feed", "https://example.com/rss")
]

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def get_conn():
    # Note: psycopg v3 accepts sslmode in the URL; we allow override via env
    if "sslmode=" not in DATABASE_URL and PG_SSLMODE:
        sep = "&" if "?" in DATABASE_URL else "?"
        dsn = f"{DATABASE_URL}{sep}sslmode={PG_SSLMODE}"
    else:
        dsn = DATABASE_URL
    return psycopg.connect(dsn, row_factory=dict_row)

def exec_sql_batch(conn, sql: str):
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()

def ensure_schema_and_seed(conn):
    exec_sql_batch(conn, SCHEMA_SQL)
    # Seed feeds
    with conn.cursor() as cur:
        for name, url in SEED_FEEDS:
            cur.execute(
                """
                INSERT INTO feed (name, url, enabled)
                VALUES (%s, %s, TRUE)
                ON CONFLICT (url) DO UPDATE SET name = EXCLUDED.name
                """,
                (name, url),
            )
    conn.commit()
    LOG.info("Schema ensured; seed feeds upserted.")

_slug_re = re.compile(r"[^a-z0-9]+")
def slugify(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = _slug_re.sub("-", text).strip("-")
    # optional: collapse long slugs
    return text[:200]

def simplify_host(netloc: str) -> str:
    host = netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host

def canonicalize_url(url: str) -> str:
    """
    - Normalize scheme/host casing
    - Remove fragments
    - Keep meaningful query if it looks like an article link; strip common tracking
    """
    try:
        p = urlparse(url)
    except Exception:
        return url

    host = p.netloc.lower()
    scheme = (p.scheme or "https").lower()
    # Strip fragment
    fragment = ""
    query = p.query

    # Remove tracking params
    if query:
        q = parse_qs(query, keep_blank_values=True)
        for bad in ("utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id", "ncid", "clid", "gclid", "fbclid"):
            if bad in q:
                q.pop(bad, None)
        query = urlencode(q, doseq=True)

    return urlunparse((scheme, host, p.path, p.params, query, fragment))

def is_google_news_link(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return "news.google.com" in host
    except Exception:
        return False

def resolve_google_news(url: str) -> Optional[str]:
    """
    Google News RSS often wraps the real destination. Try to unwrap via:
    1) the 'url' query param if present
    2) following redirects (best effort)
    """
    try:
        p = urlparse(url)
        qs = parse_qs(p.query)
        if "url" in qs and qs["url"]:
            return qs["url"][0]
    except Exception:
        pass

    try:
        # Fallback: follow redirects
        with requests.get(url, timeout=HTTP_TIMEOUT, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as r:
            if r.history and r.url and not is_google_news_link(r.url):
                return r.url
    except Exception:
        pass
    return None

def extract_summary(entry) -> str:
    # Try 'summary', fallback to title
    for key in ("summary", "summary_detail", "description"):
        v = entry.get(key)
        if isinstance(v, dict):
            v = v.get("value")
        if v:
            return str(v)
    return entry.get("title", "")

def parse_time(entry) -> Optional[datetime]:
    # Prefer published_parsed, then updated_parsed
    if entry.get("published_parsed"):
        dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        return dt
    if entry.get("updated_parsed"):
        dt = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        return dt
    return None

def compute_score(title: str, summary: str, host: str) -> float:
    # Very light heuristic; customize as needed
    score = 0.0
    txt = f"{title} {summary}".lower()
    for kw in ("earnings", "guidance", "merger", "acquisition", "award", "contract", "upgrade", "downgrade", "bankruptcy"):
        if kw in txt:
            score += 0.7
    # de-emphasize press-release aggregators if you want
    if host.endswith(("globenewswire.com", "prnewswire.com", "businesswire.com")):
        score -= 0.2
    return max(score, 0.0)

# -----------------------------------------------------------------------------
# Ingestion
# -----------------------------------------------------------------------------
def fetch_and_ingest(conn, minutes: int, min_score: float) -> Tuple[int, int, int]:
    scanned = 0
    inserted = 0
    pruned = 0

    with conn.cursor() as cur:
        cur.execute("SELECT id, name, url FROM feed WHERE enabled = TRUE ORDER BY id;")
        feeds = cur.fetchall()

    time_cutoff = datetime.now(timezone.utc) - timedelta(days=DEDUPE_WINDOW_DAYS)

    for feed_row in feeds:
        feed_id = feed_row["id"]
        feed_name = feed_row["name"]
        feed_url = feed_row["url"]

        LOG.info("parsed feed: %s", feed_url)
        parsed = feedparser.parse(feed_url)
        entries = parsed.entries[:MAX_FEED_ITEMS]
        LOG.info("feed entries: %d", len(entries))

        for e in entries:
            scanned += 1
            raw_title = e.get("title", "").strip()
            summary = extract_summary(e)
            link = e.get("link") or e.get("id") or ""

            if not link:
                continue

            # Unwrap Google News if needed
            final_url = link
            if is_google_news_link(link):
                resolved = resolve_google_news(link)
                if not resolved:
                    LOG.info("skip blocked/invalid redirect: %s (%s)", link, urlparse(link).netloc)
                    continue
                final_url = resolved

            # Canonicalize and extract host
            final_url = final_url.strip()
            final_url_canon = canonicalize_url(final_url)
            host = simplify_host(urlparse(final_url_canon).netloc or urlparse(final_url).netloc)
            display_host = host

            title = raw_title or host
            slug = slugify(title)
            published_at = parse_time(e) or datetime.now(timezone.utc)

            score = compute_score(title, summary, host)

            # -------------------------------------------------------------
            # DUPLICATE CHECK (fixed to avoid IndeterminateDatatype on $4)
            # -------------------------------------------------------------
            # Only add the (host, title_slug, published_at>=time_cutoff) clause
            # when we actually have a slug. Bind a concrete timestamptz param.
            sql = [
                "SELECT 1",
                "FROM found_url",
                "WHERE (url = %s OR url = %s OR url_canonical = %s)"
            ]
            params = [final_url, final_url_canon, final_url_canon]

            if slug:
                sql.append("OR (host = %s AND title_slug = %s AND published_at >= %s)")
                params.extend([display_host, slug, time_cutoff])

            sql.append("LIMIT 1")

            with conn.cursor() as cur:
                cur.execute("\n".join(sql), params)
                if cur.fetchone():
                    LOG.info("skip duplicate: %s (host=%s slug=%s)", final_url, display_host, slug)
                    continue

            # Insert
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO found_url
                        (url, url_canonical, host, title, title_slug, summary, published_at, score, feed_id)
                    VALUES
                        (%s,  %s,            %s,   %s,    %s,         %s,      %s,           %s,    %s)
                    ON CONFLICT (url) DO NOTHING
                    """,
                    (
                        final_url,
                        final_url_canon,
                        display_host,
                        title,
                        slug,
                        summary,
                        published_at,
                        score,
                        feed_id,
                    ),
                )
                if cur.rowcount > 0:
                    inserted += 1

            # Optional pruning of very low score items (soft delete = skip insert above)
            # If you prefer an actual delete, implement here.

    conn.commit()
    return inserted, scanned, pruned

# -----------------------------------------------------------------------------
# Email
# -----------------------------------------------------------------------------
def send_email(to_addrs: List[str], subject: str, text_body: str):
    if not to_addrs:
        LOG.warning("No recipients configured; skipping email.")
        return

    msg = f"From: {FROM_EMAIL}\r\nTo: {', '.join(to_addrs)}\r\nSubject: {subject}\r\n"
    msg += "MIME-Version: 1.0\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n"
    msg += text_body

    context = ssl.create_default_context()
    if SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=15) as server:
            if SMTP_USERNAME:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(FROM_EMAIL, to_addrs, msg.encode("utf-8"))
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            if SMTP_USERNAME:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(FROM_EMAIL, to_addrs, msg.encode("utf-8"))

    LOG.info("Email sent to %s (subject='%s')", to_addrs, subject)

def render_digest_rows(rows) -> str:
    lines = []
    for r in rows:
        when = r["published_at"].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ") if r.get("published_at") else ""
        title = (r.get("title") or "").strip()
        host = r.get("host") or ""
        url = r.get("url") or r.get("url_canonical") or ""
        score = r.get("score") or 0.0
        lines.append(f"- [{when}] ({host}) [score {score:.2f}] {title}\n  {url}")
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return f"{APP_NAME} is online."

@app.post("/admin/init")
def admin_init():
    with get_conn() as conn:
        ensure_schema_and_seed(conn)
    return {"ok": True}

@app.post("/admin/test-email")
def admin_test_email():
    subject = "Quantbrief test email"
    body = "This is a test email from Quantbrief.\n\nIf you can read this, SMTP is configured."
    send_email(TO_EMAILS, subject, body)
    return {"ok": True}

@app.post("/cron/ingest")
def cron_ingest(
    minutes: int = Query(DEFAULT_MINUTES, ge=1, le=60 * 24 * 60),
    min_score: float = Query(DEFAULT_MIN_SCORE),
):
    with get_conn() as conn:
        ensure_schema_and_seed(conn)  # safe if already exists
        inserted, scanned, pruned = fetch_and_ingest(conn, minutes=minutes, min_score=min_score)
    return {"ok": True, "inserted": inserted, "scanned": scanned, "pruned": pruned}

@app.post("/cron/digest")
def cron_digest(
    minutes: int = Query(DEFAULT_MINUTES, ge=1, le=60 * 24 * 60),
    min_score: float = Query(DEFAULT_MIN_SCORE),
    limit: int = Query(1000, ge=1, le=2000),
):
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    with get_conn() as conn, conn.cursor() as cur:
        ensure_schema_and_seed(conn)  # safe if already exists
        cur.execute(
            """
            SELECT url, url_canonical, host, title, published_at, score
            FROM found_url
            WHERE published_at >= %s
              AND score >= %s
            ORDER BY published_at DESC
            LIMIT %s
            """,
            (since, min_score, limit),
        )
        rows = cur.fetchall()

    subject = f"Quantbrief digest — last {minutes} minutes ({len(rows)} items)"
    body = render_digest_rows(rows) or "(No items)"

    send_email(TO_EMAILS, subject, body)
    return {"ok": True, "count": len(rows)}

# -----------------------------------------------------------------------------
# Error handlers
# -----------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exceptions(request: Request, exc: Exception):
    LOG.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

# -----------------------------------------------------------------------------
# Local dev
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
