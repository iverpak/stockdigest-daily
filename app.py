import os
import re
import html
import ssl
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional, Dict, Any
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode

import feedparser
import requests
import psycopg
from psycopg.rows import dict_row

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOG = logging.getLogger("quantbrief")

# -----------------------------------------------------------------------------
# Config (env)
# -----------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required")

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

DEFAULT_MINUTES = int(os.getenv("DEFAULT_MINUTES", "10080"))  # 7 days (use 1440 in prod)
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.55"))
DEFAULT_RETAIN_DAYS = int(os.getenv("DEFAULT_RETAIN_DAYS", "30"))
MAX_FETCH_PER_RUN = int(os.getenv("MAX_FETCH_PER_RUN", "200"))
SLUG_DEDUP_DAYS = int(os.getenv("SLUG_DEDUP_DAYS", "21"))

HTTP_USER_AGENT = os.getenv(
    "HTTP_USER_AGENT",
    "Mozilla/5.0 (compatible; QuantbriefBot/1.0; +https://quantbrief)"
)

# -----------------------------------------------------------------------------
# Blocklist (applied to final landing host)
# -----------------------------------------------------------------------------
def _normalized_host(u: str) -> str:
    try:
        host = urlparse(u).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""

BLOCK_REDIRECT_HOSTS = {
    h.strip().lower().lstrip(".")
    for h in os.getenv(
        "BLOCK_REDIRECT_HOSTS",
        "marketbeat.com,chat.whatsapp.com,epayslip.grz.gov.zm,khodrobank.com"
    ).split(",")
    if h.strip()
}

# -----------------------------------------------------------------------------
# Email settings
# -----------------------------------------------------------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "1") not in ("0", "false", "False", "")
SMTP_FROM = os.getenv("SMTP_FROM", "").strip()  # e.g. "QuantBrief Daily <daily@mg.example.com>"
EMAIL_FROM = os.getenv("EMAIL_FROM", "").strip()
DIGEST_TO = os.getenv("DIGEST_TO", "").strip()  # comma-separated

# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------
def db() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, autocommit=True)

def exec_sql_batch(conn: psycopg.Connection, sql: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql)

# -----------------------------------------------------------------------------
# Schema (idempotent) + migrations (safe on old DBs)
# -----------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS source_feed (
  id            BIGSERIAL PRIMARY KEY,
  url           TEXT NOT NULL UNIQUE,
  name          TEXT NOT NULL,
  language      TEXT NOT NULL DEFAULT 'en',
  retain_days   INT  NOT NULL DEFAULT 30,
  active        BOOLEAN NOT NULL DEFAULT TRUE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Base table with all current columns (ok if exists with fewer columns)
CREATE TABLE IF NOT EXISTS found_url (
  id            BIGSERIAL PRIMARY KEY,
  url           TEXT NOT NULL UNIQUE,
  title         TEXT,
  title_slug    TEXT,
  host          TEXT,
  feed_id       BIGINT REFERENCES source_feed(id) ON DELETE CASCADE,
  language      TEXT,
  article_type  TEXT,
  score         DOUBLE PRECISION,
  published_at  TIMESTAMPTZ,
  found_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes on columns that have always existed in some form
CREATE INDEX IF NOT EXISTS idx_found_url_published_at ON found_url (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_found_url_score ON found_url (score DESC);
CREATE INDEX IF NOT EXISTS idx_found_url_feed ON found_url (feed_id);
-- NOTE: indexes on possibly-missing columns (host, title_slug) are created in MIGRATIONS_SQL
"""

MIGRATIONS_SQL = """
-- Ensure columns on source_feed
ALTER TABLE source_feed ADD COLUMN IF NOT EXISTS language TEXT;
UPDATE source_feed SET language='en' WHERE language IS NULL;
ALTER TABLE source_feed ALTER COLUMN language SET DEFAULT 'en';
ALTER TABLE source_feed ALTER COLUMN language SET NOT NULL;

ALTER TABLE source_feed ADD COLUMN IF NOT EXISTS retain_days INT;
UPDATE source_feed SET retain_days = 30 WHERE retain_days IS NULL;
ALTER TABLE source_feed ALTER COLUMN retain_days SET DEFAULT 30;
ALTER TABLE source_feed ALTER COLUMN retain_days SET NOT NULL;

ALTER TABLE source_feed ADD COLUMN IF NOT EXISTS active BOOLEAN;
UPDATE source_feed SET active = TRUE WHERE active IS NULL;
ALTER TABLE source_feed ALTER COLUMN active SET DEFAULT TRUE;
ALTER TABLE source_feed ALTER COLUMN active SET NOT NULL;

-- Ensure columns on found_url
ALTER TABLE found_url ADD COLUMN IF NOT EXISTS host TEXT;
ALTER TABLE found_url ADD COLUMN IF NOT EXISTS language TEXT;
ALTER TABLE found_url ADD COLUMN IF NOT EXISTS article_type TEXT;
ALTER TABLE found_url ADD COLUMN IF NOT EXISTS score DOUBLE PRECISION;
ALTER TABLE found_url ADD COLUMN IF NOT EXISTS title_slug TEXT;

-- Indexes that depend on newly added columns
CREATE INDEX IF NOT EXISTS idx_found_url_host ON found_url (host);
CREATE INDEX IF NOT EXISTS idx_found_url_title_slug ON found_url (title_slug);
"""

SEED_FEEDS: List[Dict[str, Any]] = [
    {
        "url": "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN&hl=en-US&gl=US&ceid=US:en",
        "name": "Google News: Talen Energy OR TLN (all)",
        "language": "en",
        "retain_days": DEFAULT_RETAIN_DAYS,
        "active": True,
    },
    {
        "url": "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+-site:newser.com+-site:marketbeat.com+when:3d&hl=en-US&gl=US&ceid=US:en",
        "name": "Google News: Talen Energy OR TLN (3d, filtered)",
        "language": "en",
        "retain_days": DEFAULT_RETAIN_DAYS,
        "active": True,
    },
    {
        "url": "https://news.google.com/rss/search?q=%28Talen+Energy%29+OR+TLN+-site%3Anewser.com+-site%3Awww.marketbeat.com+-site%3Amarketbeat.com+-site%3Awww.newser.com+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen",
        "name": "Google News: ((Talen Energy) OR TLN) (7d, filtered B)",
        "language": "en",
        "retain_days": DEFAULT_RETAIN_DAYS,
        "active": True,
    },
    {
        "url": "https://news.google.com/rss/search?q=%28%28Talen+Energy%29+OR+TLN%29+-site%3Anewser.com+-site%3Awww.newser.com+-site%3Amarketbeat.com+-site%3Awww.marketbeat.com+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen",
        "name": "Google News: ((Talen Energy) OR TLN) (7d, filtered A)",
        "language": "en",
        "retain_days": DEFAULT_RETAIN_DAYS,
        "active": True,
    },
]

def ensure_schema_and_seed(conn: psycopg.Connection) -> None:
    # Base schema first (safe no-op if the old table exists with fewer columns)
    exec_sql_batch(conn, SCHEMA_SQL)
    # Then apply migrations that add columns & indexes conditionally
    exec_sql_batch(conn, MIGRATIONS_SQL)
    # Upsert seed feeds
    with conn.cursor() as cur:
        for f in SEED_FEEDS:
            cur.execute(
                """
                INSERT INTO source_feed (url, name, language, retain_days, active)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET
                  name = EXCLUDED.name,
                  language = EXCLUDED.language,
                  retain_days = EXCLUDED.retain_days,
                  active = EXCLUDED.active
                """,
                (f["url"], f["name"], f["language"], f["retain_days"], f["active"]),
            )
    LOG.info("Schema ensured; seed feeds upserted.")

# -----------------------------------------------------------------------------
# Canonicalization / redirects / blocklist
# -----------------------------------------------------------------------------
def _strip_tracking_params(u: str) -> str:
    try:
        p = urlparse(u)
        q = parse_qs(p.query, keep_blank_values=False)
        cleaned = {
            k: v for k, v in q.items()
            if not (k.lower().startswith("utm_") or k.lower() in {"gclid","fbclid","mc_cid","mc_eid","yclid","igsh","s","ved"})
        }
        new_q = urlencode(cleaned, doseq=True)
        return urlunparse((p.scheme, p.netloc, p.path, p.params, new_q, p.fragment))
    except Exception:
        return u

def _google_news_target(u: str) -> Optional[str]:
    try:
        if _normalized_host(u) != "news.google.com":
            return None
        p = urlparse(u)
        q = parse_qs(p.query)
        for key in ("url","u"):
            if key in q and q[key]:
                return q[key][0]
    except Exception:
        pass
    return None

def _resolve_final_url(start_url: str, timeout: int = 8) -> str:
    headers = {"User-Agent": HTTP_USER_AGENT}
    target = _google_news_target(start_url) or start_url
    final = target
    try:
        with requests.Session() as s:
            r = s.head(target, allow_redirects=True, timeout=timeout, headers=headers)
            final = r.url or target
            if _normalized_host(final) == _normalized_host(target):
                r = s.get(target, allow_redirects=True, timeout=timeout, headers=headers, stream=True)
                final = r.url or final
                r.close()
    except Exception:
        pass
    return _strip_tracking_params(final)

def _is_blocked_redirect(start_url: str) -> Tuple[bool, str, str]:
    final_url = _resolve_final_url(start_url)
    host = _normalized_host(final_url)
    return (host in BLOCK_REDIRECT_HOSTS, final_url, host)

# -----------------------------------------------------------------------------
# Feed listing
# -----------------------------------------------------------------------------
def list_active_feeds(conn: psycopg.Connection) -> List[Dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, url, name, language, retain_days, active
            FROM source_feed
            WHERE active = TRUE
            ORDER BY id
            """
        )
        return list(cur.fetchall())

# -----------------------------------------------------------------------------
# Title slug (for cross-outlet de-dup)
# -----------------------------------------------------------------------------
_title_clean_re = re.compile(r"[^a-z0-9]+")
def _title_slug(title: str) -> str:
    t = (title or "").strip().lower()
    t = re.sub(r"\$[a-z0-9]+", "", t)
    t = _title_clean_re.sub("-", t)
    t = re.sub(r"-{2,}", "-", t).strip("-")
    return t

# -----------------------------------------------------------------------------
# Simple content-agnostic scoring (placeholder)
# -----------------------------------------------------------------------------
def simple_score(title: str, host: str) -> float:
    score = 0.5
    tlen = len(title or "")
    if tlen >= 40:
        score += 0.06
    elif tlen >= 25:
        score += 0.03
    good_bits = ("reuters", "bloomberg", "ft.com", "wsj.com", "seekingalpha", "heatmap.news", "bisnow", "utilitydive", "power-eng.com", "power-mag")
    if any(x in host for x in good_bits):
        score += 0.08
    return min(score, 0.99)

# -----------------------------------------------------------------------------
# Ingest & prune (with strong de-dup and blocklist on final host)
# -----------------------------------------------------------------------------
def fetch_and_ingest(conn: psycopg.Connection, minutes: int, min_score: float) -> Tuple[int, int, int]:
    feeds = list_active_feeds(conn)
    cutoff_dt = datetime.now(timezone.utc) - timedelta(minutes=minutes)

    headers = {"User-Agent": HTTP_USER_AGENT}

    inserted = 0
    scanned = 0
    blocked_redirects = 0

    # Run-level de-dup
    seen_urls: set = set()
    seen_slugs: set = set()

    for sf in feeds:
        url = sf["url"]
        language = sf.get("language") or "en"

        try:
            feed = feedparser.parse(url, request_headers=headers)
            entries = feed.get("entries") or []
            LOG.info("parsed feed: %s", url)
            LOG.info("feed entries: %d", len(entries))
        except Exception as e:
            LOG.warning("failed to parse feed %s: %s", url, e)
            continue

        for e in entries[:MAX_FETCH_PER_RUN]:
            raw_url = e.get("link") or e.get("id") or e.get("url")
            if not raw_url:
                continue

            # Quick initial host block
            if _normalized_host(raw_url) in BLOCK_REDIRECT_HOSTS:
                LOG.info("skip blocked host (initial): %s", raw_url)
                blocked_redirects += 1
                continue

            # Resolve final URL and block by final host
            blocked, final_url, final_host = _is_blocked_redirect(raw_url)
            if blocked or final_host == "" or final_host == "news.google.com":
                LOG.info("skip blocked/invalid redirect: %s -> %s (%s)", raw_url, final_url, final_host)
                blocked_redirects += 1
                continue

            title = (e.get("title") or "").strip()
            slug = _title_slug(title)

            # Published time from feed; if missing, fallback to now
            published = None
            for k in ("published_parsed", "updated_parsed"):
                tm = e.get(k)
                if tm:
                    published = datetime(*tm[:6], tzinfo=timezone.utc)
                    break
            if not published:
                published = datetime.now(timezone.utc)

            if published < cutoff_dt:
                continue

            # Run-level de-dup
            if final_url in seen_urls:
                continue
            if slug and slug in seen_slugs:
                continue

            # Score (placeholder)
            s = simple_score(title, final_host)
            if s < min_score:
                continue

            # DB-level de-dup by URL and recent title_slug
            with conn.cursor() as cur:
                # URL already exists?
                cur.execute("SELECT 1 FROM found_url WHERE url = %s", (final_url,))
                if cur.fetchone():
                    continue

                # Recent slug exists?
                cur.execute(
                    """
                    SELECT 1
                    FROM found_url
                    WHERE title_slug = %s
                      AND published_at >= NOW() - (%s * INTERVAL '1 day')
                    LIMIT 1
                    """,
                    (slug or None, SLUG_DEDUP_DAYS),
                )
                if cur.fetchone():
                    continue

                try:
                    cur.execute(
                        """
                        INSERT INTO found_url
                          (url, title, title_slug, host, feed_id, language, article_type, score, published_at, found_at)
                        VALUES
                          (%s,   %s,    %s,         %s,   %s,      %s,       %s,           %s,    %s,           NOW())
                        ON CONFLICT (url) DO NOTHING
                        """,
                        (
                            final_url,
                            title,
                            slug or None,
                            final_host,
                            sf["id"],
                            language,
                            "article",
                            s,
                            published,
                        ),
                    )
                    if cur.rowcount:
                        inserted += 1
                        seen_urls.add(final_url)
                        if slug:
                            seen_slugs.add(slug)
                except Exception as ex:
                    LOG.warning("insert failed for %s: %s", final_url, ex)

            scanned += 1

    # Prune old rows by per-feed retain_days (0 means use default)
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM found_url f
            USING source_feed s
            WHERE f.feed_id = s.id
              AND f.published_at < NOW() - (COALESCE(NULLIF(s.retain_days, 0), %s) * INTERVAL '1 day')
            """,
            (DEFAULT_RETAIN_DAYS,),
        )
        pruned = cur.rowcount or 0

    LOG.info(
        "ingest done: inserted=%d, fetched_scored=%d, blocked_redirects=%d, pruned=%d",
        inserted, scanned, blocked_redirects, pruned
    )
    return inserted, scanned, pruned

# -----------------------------------------------------------------------------
# Digest fetch (de-duplicated across outlets by title_slug/url)
# -----------------------------------------------------------------------------
def fetch_digest_rows(conn: psycopg.Connection, since_dt: datetime, min_score: float) -> List[Dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            WITH dedup AS (
              SELECT DISTINCT ON (COALESCE(title_slug, url))
                     f.id, f.title, f.url, f.host, f.article_type, COALESCE(f.score,0) AS score,
                     f.published_at, s.name AS feed_name
              FROM found_url f
              JOIN source_feed s ON s.id = f.feed_id
              WHERE f.published_at >= %s
                AND COALESCE(f.score, 0) >= %s
              ORDER BY COALESCE(title_slug, url), COALESCE(f.score,0) DESC, f.published_at DESC, f.id DESC
            )
            SELECT id, title, url, host, article_type, score, published_at, feed_name
            FROM dedup
            ORDER BY score DESC, published_at DESC, id DESC
            """,
            (since_dt, min_score),
        )
        return list(cur.fetchall())

def render_digest_html(rows: List[Dict[str, Any]], minutes: int) -> str:
    items = []
    for r in rows:
        t = html.escape(r.get("title") or "")
        u = html.escape(r.get("url") or "#")
        src = html.escape(r.get("feed_name") or "")
        ts = r.get("published_at")
        ts_iso = ts.astimezone(timezone.utc).isoformat() if isinstance(ts, datetime) else ""
        sc = r.get("score")
        hos = html.escape(r.get("host") or "")
        badge = f" • score {sc:.2f}" if isinstance(sc, (int, float)) else ""
        items.append(f'<li><a href="{u}">{t}</a> <span style="color:#666">({src} • {ts_iso}{badge} • {hos})</span></li>')
    body = "<ul>" + "\n".join(items) + "</ul>" if items else "<p>No items.</p>"
    return f"""
    <html>
      <body>
        <h2>Quantbrief digest — last {minutes} minutes</h2>
        {body}
      </body>
    </html>
    """

# -----------------------------------------------------------------------------
# Email
# -----------------------------------------------------------------------------
def _parse_sender_addr() -> Tuple[str, str]:
    header_from = SMTP_FROM or EMAIL_FROM
    if not header_from:
        raise RuntimeError("SMTP not configured. Need SMTP_HOST and EMAIL_FROM/SMTP_FROM.")
    m = re.search(r"<\s*([^>]+)\s*>", header_from)
    envelope_from = m.group(1).strip() if m else header_from.strip()
    return header_from, envelope_from

def _parse_recipients(to_env: str) -> List[str]:
    if not to_env:
        return []
    recips = []
    for part in to_env.split(","):
        p = part.strip()
        if not p:
            continue
        m = re.search(r"<\s*([^>]+)\s*>", p)
        recips.append(m.group(1).strip() if m else p)
    return recips

def send_email(subject: str, html_body: str, to_addrs: Optional[List[str]] = None) -> bool:
    if not SMTP_HOST:
        LOG.error("SMTP not configured. Need SMTP_HOST and EMAIL_FROM/SMTP_FROM.")
        return False

    header_from, envelope_from = _parse_sender_addr()
    to_list = to_addrs or _parse_recipients(DIGEST_TO)
    if not to_list:
        LOG.error("No recipients configured (DIGEST_TO empty and no to_addrs provided).")
        return False

    import smtplib
    from email.mime.text import MIMEText
    from email.utils import formatdate, make_msgid

    msg = MIMEText(html_body, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = header_from
    msg["To"] = ", ".join(to_list)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            if SMTP_STARTTLS:
                s.starttls(context=ssl.create_default_context())
            if SMTP_USERNAME and SMTP_PASSWORD:
                s.login(SMTP_USERNAME, SMTP_PASSWORD)
            s.sendmail(envelope_from, to_list, msg.as_string())
        LOG.info("Email sent to %s (subject='%s')", to_list, subject)
        return True
    except Exception as e:
        LOG.error("send_email failed: %s", e, exc_info=True)
        return False

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI()

def _require_admin(req: Request):
    tok = req.headers.get("x-admin-token") or ""
    if not tok and "authorization" in req.headers:
        auth = req.headers["authorization"]
        if auth.lower().startswith("bearer "):
            tok = auth.split(" ", 1)[1].strip()
    if not ADMIN_TOKEN or tok != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/", response_class=PlainTextResponse)
def root():
    return "quantbrief running"

@app.post("/admin/init", response_class=PlainTextResponse)
def admin_init(req: Request):
    _require_admin(req)
    with db() as conn:
        ensure_schema_and_seed(conn)
    return "ok"

@app.post("/cron/ingest", response_class=PlainTextResponse)
def cron_ingest(req: Request, minutes: Optional[int] = None, min_score: Optional[float] = None):
    _require_admin(req)
    minutes = int(minutes or DEFAULT_MINUTES)
    min_score = float(min_score or DEFAULT_MIN_SCORE)
    with db() as conn:
        ensure_schema_and_seed(conn)
        inserted, scanned, pruned = fetch_and_ingest(conn, minutes=minutes, min_score=min_score)
    return f"ok (inserted={inserted}, scanned={scanned}, pruned={pruned})"

@app.post("/cron/digest", response_class=PlainTextResponse)
def cron_digest(req: Request, minutes: Optional[int] = None, min_score: Optional[float] = None):
    _require_admin(req)
    minutes = int(minutes or DEFAULT_MINUTES)
    min_score = float(min_score or DEFAULT_MIN_SCORE)
    since_dt = datetime.now(timezone.utc) - timedelta(minutes=minutes)

    with db() as conn:
        ensure_schema_and_seed(conn)
        rows = fetch_digest_rows(conn, since_dt=since_dt, min_score=min_score)

    html_body = render_digest_html(rows, minutes)
    ok = send_email(f"Quantbrief digest — last {minutes} minutes ({len(rows)} items)", html_body)
    return "ok" if ok else "email failed"

@app.post("/admin/test-email", response_class=PlainTextResponse)
def admin_test_email(req: Request):
    _require_admin(req)
    html_body = "<b>It works</b>"
    ok = send_email("Quantbrief test email", html_body)
    return "ok" if ok else "failed"
