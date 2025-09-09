# app.py
import os
import re
import ssl
import smtplib
import logging
import hashlib
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
PG_SSLMODE = os.getenv("PGSSLMODE", "prefer")

# Email settings
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", f"{APP_NAME}@no-reply.local")
TO_EMAILS = [e.strip() for e in os.getenv("TO_EMAILS", "quantbrief.research@gmail.com").split(",") if e.strip()]

# Ingestion / scoring / dedupe
DEDUPE_WINDOW_DAYS = int(os.getenv("DEDUPE_WINDOW_DAYS", "14"))
DEFAULT_MINUTES = 60 * 24 * 7
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.0"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "8.0"))
MAX_FEED_ITEMS = int(os.getenv("MAX_FEED_ITEMS", "200"))

# Banned hosts (hard filter). Extend/override with env.
DEFAULT_BANNED = [
    "news.google.com",   # aggregator wrapper
    "marketbeat.com",    # explicitly banned in your query
    "newser.com",        # explicitly banned in your query
    "linkedin.com",
    "facebook.com",
    "youtube.com",
    "msn.com",
]
BANNED_HOSTS = {
    h.strip().lower()
    for h in (os.getenv("BANNED_HOSTS", ",".join(DEFAULT_BANNED))).split(",")
    if h.strip()
}

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
SCHEMA_TABLES = """
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
    title_slug     TEXT,
    summary        TEXT,
    published_at   TIMESTAMPTZ,
    score          DOUBLE PRECISION DEFAULT 0.0,
    feed_id        INTEGER,
    fingerprint    TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

SCHEMA_INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_url ON found_url(url);
CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_url_canonical
  ON found_url(url_canonical) WHERE url_canonical IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_fingerprint
  ON found_url(fingerprint) WHERE fingerprint IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_found_url_host_slug_time
  ON found_url (host, title_slug, published_at DESC);
CREATE INDEX IF NOT EXISTS ix_found_url_published_at
  ON found_url (published_at DESC);
"""

-- Uniqueness
CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_url ON found_url(url);
CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_url_canonical ON found_url(url_canonical) WHERE url_canonical IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_fingerprint ON found_url(fingerprint) WHERE fingerprint IS NOT NULL;

-- Performance helpers
CREATE INDEX IF NOT EXISTS ix_found_url_host_slug_time ON found_url (host, title_slug, published_at DESC);
CREATE INDEX IF NOT EXISTS ix_found_url_published_at ON found_url(published_at DESC);
"""

SEED_FEEDS: List[Tuple[str, str]] = [
    (
        "Google News — Talen Energy (3d window, filtered)",
        "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+-site:newser.com+-site:marketbeat.com+when:3d&hl=en-US&gl=US&ceid=US:en",
    ),
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_conn():
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
    # 1) Create tables (no indexes that depend on new columns)
    with conn.cursor() as cur:
        cur.execute(SCHEMA_TABLES)
    conn.commit()

    # 2) Migrations: add columns if missing (idempotent)
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE found_url ADD COLUMN IF NOT EXISTS url_canonical TEXT;")
        cur.execute("ALTER TABLE found_url ADD COLUMN IF NOT EXISTS title_slug TEXT;")
        cur.execute("ALTER TABLE found_url ADD COLUMN IF NOT EXISTS score DOUBLE PRECISION DEFAULT 0.0;")
        cur.execute("ALTER TABLE found_url ADD COLUMN IF NOT EXISTS feed_id INTEGER;")
        cur.execute("ALTER TABLE found_url ADD COLUMN IF NOT EXISTS fingerprint TEXT;")
    conn.commit()

    # 3) Now create indexes (column definitely exists)
    with conn.cursor() as cur:
        cur.execute(SCHEMA_INDEXES)
    conn.commit()

    # 4) Seed feeds
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
    return text[:200]

def simplify_host(netloc: str) -> str:
    host = (netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host

def canonicalize_url(url: str) -> str:
    try:
        p = urlparse(url)
    except Exception:
        return url

    scheme = (p.scheme or "https").lower()
    host = (p.netloc or "").lower()
    path = p.path or ""
    params = p.params or ""
    query = p.query or ""
    fragment = ""  # always drop

    # Strip common tracking params everywhere
    if query:
        q = parse_qs(query, keep_blank_values=True)
        for bad in ("utm_source","utm_medium","utm_campaign","utm_term","utm_content","utm_id",
                    "gclid","fbclid","ncid","clid","igshid","spm","xtor"):
            q.pop(bad, None)
        # Special: Google News junk query keys -> drop all, we never keep GN wrappers
        if host.endswith("news.google.com"):
            q = {}
        query = urlencode(q, doseq=True)

    return urlunparse((scheme, host, path, params, query, fragment))

def is_google_news_link(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return "news.google.com" in host
    except Exception:
        return False

def resolve_google_news(url: str) -> Optional[str]:
    """
    Try to unwrap a Google News RSS link.
    1) If there's a 'url=' param, use it.
    2) Otherwise we DO NOT keep the wrapper; if we can't unwrap, return None.
    """
    try:
        p = urlparse(url)
        qs = parse_qs(p.query)
        if "url" in qs and qs["url"]:
            return qs["url"][0]
    except Exception:
        pass
    # No reliable param? Hard fail -> skip this item.
    return None

def extract_summary(entry) -> str:
    for key in ("summary", "summary_detail", "description"):
        v = entry.get(key)
        if isinstance(v, dict):
            v = v.get("value")
        if v:
            return str(v)
    return entry.get("title", "")

def parse_time(entry) -> Optional[datetime]:
    if entry.get("published_parsed"):
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    if entry.get("updated_parsed"):
        return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
    return None

def compute_score(title: str, summary: str, host: str) -> float:
    score = 0.0
    txt = f"{title} {summary}".lower()
    for kw in ("earnings","guidance","merger","acquisition","award","contract","upgrade","downgrade","bankruptcy","buyback","repurchase"):
        if kw in txt:
            score += 0.7
    if host.endswith(("globenewswire.com","prnewswire.com","businesswire.com")):
        score -= 0.2
    return max(score, 0.0)

def is_banned_host(host: str) -> bool:
    host = (host or "").lower()
    base = host
    return base in BANNED_HOSTS or any(base.endswith("." + b) for b in BANNED_HOSTS)

def compute_fingerprint(url_canon: Optional[str], host: str, slug: str, published_at: datetime) -> str:
    """
    Prefer canonical URL. Fallback: host+slug+day — avoids dupes
    across runs, feeds, and GN wrapper variants.
    """
    if url_canon:
        key = url_canon
    else:
        day = (published_at or datetime.now(timezone.utc)).date().isoformat()
        key = f"{host}|{slug}|{day}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

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

    for feed_row in feeds:
        feed_id = feed_row["id"]
        feed_url = feed_row["url"]

        LOG.info("parsed feed: %s", feed_url)
        parsed = feedparser.parse(feed_url)
        entries = parsed.entries[:MAX_FEED_ITEMS]
        LOG.info("feed entries: %d", len(entries))

        for e in entries:
            scanned += 1
            link = e.get("link") or e.get("id") or ""
            if not link:
                continue

            # 1) STRICTLY unwrap/ban Google News
            if is_google_news_link(link):
                resolved = resolve_google_news(link)
                if not resolved:
                    LOG.info("skip (GN unresolvable): %s", link)
                    continue
                link = resolved

            # Canonicalize / host / filters
            link = link.strip()
            link_canon = canonicalize_url(link)
            host = simplify_host(urlparse(link_canon).netloc or urlparse(link).netloc)

            if is_banned_host(host):
                LOG.info("skip (banned host=%s): %s", host, link_canon or link)
                continue

            title = (e.get("title") or host).strip()
            summary = extract_summary(e)
            published_at = parse_time(e) or datetime.now(timezone.utc)
            slug = slugify(title)
            score = compute_score(title, summary, host)

            # Optional scoring gate
            if score < min_score:
                pruned += 1
                continue

            fingerprint = compute_fingerprint(link_canon or link, host, slug, published_at)

            # INSERT with hard dedupe on fingerprint (and url/url_canonical as backup)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO found_url
                        (url, url_canonical, host, title, title_slug, summary, published_at, score, feed_id, fingerprint)
                    VALUES
                        (%s,  %s,            %s,   %s,    %s,         %s,      %s,           %s,    %s,      %s)
                    ON CONFLICT (fingerprint) DO NOTHING
                    """,
                    (
                        link,
                        link_canon,
                        host,
                        title,
                        slug,
                        summary,
                        published_at,
                        score,
                        feed_id,
                        fingerprint,
                    ),
                )
                if cur.rowcount > 0:
                    inserted += 1

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
    """
    Hyperlink the title; no raw URLs. Dedup by a running set of fingerprints.
    """
    seen = set()
    lines = []
    for r in rows:
        fp = r.get("fingerprint")
        if fp in seen:
            continue
        seen.add(fp)

        when = r["published_at"].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ") if r.get("published_at") else ""
        title = (r.get("title") or "").strip()
        host = r.get("host") or ""
        url = r.get("url_canonical") or r.get("url") or ""
        score = float(r.get("score") or 0.0)

        # Markdown: Title hyperlinked, no naked link
        lines.append(f"- [{when}] ({host}) [score {score:.2f}] [{title}]({url})")
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
        ensure_schema_and_seed(conn)
        inserted, scanned, pruned = fetch_and_ingest(conn, minutes=minutes, min_score=min_score)
    return {"ok": True, "inserted": inserted, "scanned": scanned, "pruned": pruned}

@app.post("/cron/digest")
def cron_digest(
    minutes: int = Query(DEFAULT_MINUTES, ge=1, le=60 * 24 * 60),
    min_score: float = Query(DEFAULT_MIN_SCORE),
    limit: int = Query(1000, ge=1, le=5000),
):
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    with get_conn() as conn, conn.cursor() as cur:
        ensure_schema_and_seed(conn)
        cur.execute(
            """
            SELECT url, url_canonical, host, title, published_at, score, fingerprint
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
# Error handler
# -----------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exceptions(request: Request, exc: Exception):
    LOG.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

# -----------------------------------------------------------------------------
# Dev entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
