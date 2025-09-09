import os
import re
import smtplib
import hashlib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qs, urljoin

import feedparser
import psycopg
import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, HTMLResponse

APP_NAME = "quantbrief"
app = FastAPI(title=APP_NAME)

# ---------- Config ----------
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
FROM_EMAIL = os.environ.get("FROM_EMAIL") or "quantbrief=no-reply.local@mg.quantbrief.ca"
TO_EMAILS = [e.strip() for e in os.environ.get("TO_EMAILS", "").split(",") if e.strip()]
SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
RESOLVE_TIMEOUT = float(os.environ.get("RESOLVE_TIMEOUT", "8"))
MAX_FEED_ITEMS = int(os.environ.get("MAX_FEED_ITEMS", "100"))

# Primary topic example feed seeds (you can add your own)
DEFAULT_FEEDS = [
    # Filter out some notorious mirrors at the source, but we'll also filter in-app.
    "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+-site:newser.com+-site:marketbeat.com+when:3d&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+when:7d&hl=en-US&gl=US&ceid=US:en",
]

# Hosts to exclude from the digest
BANNED_HOSTS = {
    # noisy / mirrors / social
    "marketbeat.com", "newser.com", "linkedin.com", "facebook.com", "youtube.com", "youtu.be",
    "msn.com", "tipranks.com", "simplywall.st",
    # spammy mirrors seen in your logs
    "js.signavitae.com", "epayslip.grz.gov.zm", "mehadschools.ir", "krjc.org", "howcome.com.tw",
    "alumnimallrandgo.up.ac.za", "primatevets.org", "valueinvestorinsight.com", "giro.fr",
    "tw13zhi.com", "zyrofisherb2b.co.uk", "randgo", "clientele.co.za", "thjb.com.tw",
    "samsconsult.com", "westganews.net", "bluerewards.clientele.co.za",
    # finance rewrites (keep if you don't want Yahoo quote pages)
    "finance.yahoo.com", "uk.finance.yahoo.com", "ca.finance.yahoo.com", "yahoo.com",
    # allow Google News (we dedupe and display publisher if we can)
    # "news.google.com"  <-- intentionally NOT banned
}

# ---------- Schema ----------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feed (
    id          SERIAL PRIMARY KEY,
    url         TEXT UNIQUE NOT NULL,
    active      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS found_url (
    id            BIGSERIAL PRIMARY KEY,
    url           TEXT NOT NULL,
    url_canonical TEXT,
    host          TEXT,
    title         TEXT,
    title_slug    TEXT,
    summary       TEXT,
    published_at  TIMESTAMPTZ,
    score         DOUBLE PRECISION DEFAULT 0,
    feed_id       INTEGER REFERENCES feed(id) ON DELETE SET NULL,
    fingerprint   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Useful indexes
CREATE INDEX IF NOT EXISTS ix_found_url_published_at ON found_url(published_at DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS ix_found_url_host ON found_url(host);
CREATE INDEX IF NOT EXISTS ix_found_url_title_slug ON found_url(title_slug);
CREATE INDEX IF NOT EXISTS ix_found_url_url_canonical ON found_url(url_canonical);

-- Dedupe constraints
CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_url ON found_url(url);
CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_fingerprint
  ON found_url(fingerprint)
  WHERE fingerprint IS NOT NULL;

-- Seed feeds
"""

UPSERT_FEED_SQL = """
INSERT INTO feed (url, active)
VALUES (%s, TRUE)
ON CONFLICT (url) DO UPDATE SET active = EXCLUDED.active;
"""

INSERT_FOUND_SQL = """
INSERT INTO found_url (
    url, url_canonical, host, title, title_slug, summary,
    published_at, score, feed_id, fingerprint
) VALUES (
    %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s
)
ON CONFLICT DO NOTHING
RETURNING id;
"""

# ---------- Helpers ----------
def get_conn():
    return psycopg.connect(DATABASE_URL, autocommit=True)

def slugify(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = re.sub(r"[^\w\s-]", "", value.lower())
    v = re.sub(r"\s+", "-", v).strip("-")
    return v or None

def normalize_host(host: Optional[str]) -> Optional[str]:
    if not host:
        return None
    host = host.strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host

def is_banned_host(host: Optional[str]) -> bool:
    h = normalize_host(host)
    if not h:
        return False
    # Allow Google News to pass (we’ll still dedupe and show title link)
    if h == "news.google.com":
        return False
    return any(h == b or h.endswith("." + b) for b in BANNED_HOSTS)

def canon_url(u: str) -> str:
    """Normalize obvious tracking bits; for news.google.com keep as-is."""
    try:
        p = urlparse(u)
        if not p.scheme:
            return u
        # Strip common tracking params
        if p.query:
            q = parse_qs(p.query, keep_blank_values=False)
            for k in list(q.keys()):
                if k.lower() in {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "oc"}:
                    q.pop(k, None)
            query = "&".join(f"{k}={v[0]}" for k, v in q.items())
        else:
            query = ""
        # drop fragment
        p = p._replace(query=query, fragment="")
        return urlunparse(p)
    except Exception:
        return u

def resolve_url(u: str) -> Tuple[str, Optional[str]]:
    """
    Follow redirects to get a canonical URL. Returns (final_url, final_host).
    On error, returns (original, parsed_host).
    """
    try:
        r = requests.get(u, timeout=RESOLVE_TIMEOUT, allow_redirects=True, headers={"User-Agent": "quantbrief-bot/1.0"})
        final = r.url or u
        host = normalize_host(urlparse(final).netloc)
        return canon_url(final), host
    except Exception:
        p = urlparse(u)
        return canon_url(u), normalize_host(p.netloc)

def make_fingerprint(url: str, url_canonical: Optional[str], title: Optional[str]) -> str:
    base = (url_canonical or url or "") + "|" + (title or "")
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def as_utc(dt: Optional[datetime]) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def ensure_schema_and_seed(conn):
    app.logger.info("Schema ensuring...")
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        for feed_url in DEFAULT_FEEDS:
            cur.execute(UPSERT_FEED_SQL, (feed_url,))
    app.logger.info("Schema ensured; seed feeds upserted.")

# ---------- Ingest ----------
def fetch_and_ingest(conn, minutes: int = 7 * 24 * 60, min_score: float = 0.0) -> Tuple[int, int, int]:
    """Return (inserted, scanned, pruned)."""
    inserted = 0
    scanned = 0
    pruned = 0

    with conn.cursor() as cur:
        cur.execute("SELECT id, url, active FROM feed WHERE active = TRUE ORDER BY id;")
        feeds = cur.fetchall()

    for feed_id, feed_url, _active in feeds:
        app.logger.info(f"{APP_NAME}:parsed feed: {feed_url}")
        parsed = feedparser.parse(feed_url)
        entries = parsed.entries[:MAX_FEED_ITEMS]
        app.logger.info(f"{APP_NAME}:feed entries: {len(entries)}")
        for e in entries:
            scanned += 1
            raw_url = e.get("link") or e.get("id") or ""
            if not raw_url:
                pruned += 1
                continue

            # Normalize & (optionally) resolve
            href = canon_url(raw_url)
            title = (e.get("title") or "").strip()
            summary = (e.get("summary") or e.get("description") or "").strip() or None

            # published
            published_at: Optional[datetime] = None
            for key in ("published_parsed", "updated_parsed"):
                if e.get(key):
                    published_at = datetime(*e[key][:6], tzinfo=timezone.utc)
                    break

            # Try to resolve (only if not Google News)
            u_host = normalize_host(urlparse(href).netloc)
            if u_host and u_host != "news.google.com":
                final_url, final_host = resolve_url(href)
            else:
                final_url, final_host = href, u_host

            # Skip banned hosts
            if is_banned_host(final_host):
                pruned += 1
                continue

            title_slug = slugify(title)
            fp = make_fingerprint(final_url, final_url, title)

            try:
                with conn.cursor() as cur:
                    cur.execute(
                        INSERT_FOUND_SQL,
                        (
                            href,                         # url (original / feed link)
                            final_url,                    # url_canonical
                            final_host,                   # host
                            title or None,                # title
                            title_slug,                   # title_slug
                            summary,                      # summary
                            as_utc(published_at),         # published_at
                            float(e.get("score", 0.0)),   # score
                            feed_id,                      # feed_id
                            fp,                           # fingerprint
                        ),
                    )
                    if cur.rowcount > 0:
                        inserted += 1
                    else:
                        pruned += 1
            except Exception as ex:
                app.logger.warning(f"insert skipped due to error: {ex}")
                pruned += 1

    app.logger.info(f"{APP_NAME}:Ingest complete. inserted={inserted} scanned={scanned} pruned={pruned}")
    return inserted, scanned, pruned

# ---------- Digest selection & email formatting ----------
def select_digest_rows(conn, minutes: int, limit: int = 250):
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    banned = sorted(BANNED_HOSTS)
    placeholders = ",".join(["%s"] * len(banned)) if banned else None

    base_where = "published_at >= %s"
    params: List = [since]

    if banned:
        base_where += f" AND (host IS NULL OR host NOT IN ({placeholders}))"
        params.extend(banned)

    sql = f"""
        WITH ranked AS (
            SELECT
                id, url, url_canonical, host, title, title_slug, summary,
                published_at, score, feed_id, fingerprint,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(fingerprint, url_canonical, url, title_slug, id::text)
                    ORDER BY published_at DESC NULLS LAST, id DESC
                ) AS rn
            FROM found_url
            WHERE {base_where}
        )
        SELECT
            id, url, url_canonical, host, title, title_slug, summary,
            published_at, score, feed_id, fingerprint
        FROM ranked
        WHERE rn = 1
        ORDER BY published_at DESC NULLS LAST, id DESC
        LIMIT %s
    """
    params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()

def escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )

def format_email_html(subject: str, rows: List[Tuple]) -> Tuple[str, str]:
    # HTML body
    html_parts = [
        f"<h3>{escape_html(subject)}</h3>",
        "<ul>",
    ]
    # Plain text body
    text_lines = [subject]

    for r in rows:
        (_id, url, url_canonical, host, title, _slug, _summary, published_at, score, _feed_id, _fp) = r
        href = url_canonical or url
        p = urlparse(href)
        shown_host = normalize_host(p.netloc) or host or "—"
        ts = as_utc(published_at).strftime("%Y-%m-%d %H:%MZ")
        safe_title = escape_html(title or href)

        # Title is hyperlinked (your requirement)
        html_parts.append(
            f'<li>[{ts}] ({shown_host}) '
            f'[score {score:.2f}] '
            f'<a href="{escape_html(href)}">{safe_title}</a></li>'
        )

        # Plain text fallback: title then URL on next line
        text_lines.append(f"- [{ts}] ({shown_host}) [score {score:.2f}] {title or href}\n  {href}")

    html_parts.append("</ul>")
    return "\n".join(html_parts), "\n".join(text_lines)

def send_email(subject: str, html_body: str, text_body: str) -> None:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and TO_EMAILS):
        app.logger.warning("SMTP not fully configured; skipping email send.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = ", ".join(TO_EMAILS)

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(FROM_EMAIL, TO_EMAILS, msg.as_string())

# ---------- Routes ----------
@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK"

@app.post("/admin/init", response_class=PlainTextResponse)
def admin_init():
    with get_conn() as conn:
        ensure_schema_and_seed(conn)
    app.logger.info(f"{APP_NAME}:Schema ensured; seed feeds upserted.")
    return "ok"

@app.post("/cron/ingest", response_class=PlainTextResponse)
def cron_ingest(request: Request, minutes: int = 7 * 24 * 60, min_score: float = 0.0):
    with get_conn() as conn:
        ensure_schema_and_seed(conn)
        inserted, scanned, pruned = fetch_and_ingest(conn, minutes=minutes, min_score=min_score)
        return f"ingest done: inserted={inserted} scanned={scanned} pruned={pruned}"

@app.post("/cron/digest", response_class=PlainTextResponse)
def cron_digest(minutes: int = 7 * 24 * 60):
    with get_conn() as conn:
        ensure_schema_and_seed(conn)
        rows = select_digest_rows(conn, minutes=minutes, limit=1000)
        subject = f"Quantbrief digest — last {minutes} minutes ({len(rows)} items)"
        try:
            html_body, text_body = format_email_html(subject, rows)
        except Exception as ex:
            app.logger.error("Unhandled error in /cron/digest", exc_info=True)
            # still return 200 to avoid retries from Render; body explains
            return f"error formatting digest: {ex}"

    # Send email if SMTP is configured
    try:
        send_email(subject, html_body, text_body)
    except Exception as ex:
        app.logger.warning(f"email send failed: {ex}")

    return f"ok: {len(rows)} items"

@app.post("/admin/test-email", response_class=PlainTextResponse)
def admin_test_email():
    subject = "Quantbrief test email"
    html_body, text_body = format_email_html(subject, [])
    try:
        send_email(subject, html_body, text_body)
        app.logger.info(f"{APP_NAME}:Email sent to {TO_EMAILS} (subject='{subject}')")
    except Exception as ex:
        app.logger.warning(f"{APP_NAME}:test email failed: {ex}")
    return "ok"

# ---------- Simple HTML preview (nice for eyeballing formatting) ----------
@app.get("/admin/preview-digest", response_class=HTMLResponse)
def preview_digest(minutes: int = 7 * 24 * 60, limit: int = 250):
    with get_conn() as conn:
        ensure_schema_and_seed(conn)
        rows = select_digest_rows(conn, minutes=minutes, limit=limit)
        subject = f"Quantbrief digest — last {minutes} minutes ({len(rows)} items)"
        html_body, _ = format_email_html(subject, rows)
        return f"<!doctype html><meta charset='utf-8'><title>{escape_html(subject)}</title>{html_body}"
