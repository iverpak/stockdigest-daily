# app.py
import os
import re
import ssl
import hmac
import smtplib
import hashlib
import logging
import datetime as dt
from typing import Optional, Tuple, List
from urllib.parse import urlparse, parse_qs, unquote

import requests
import feedparser
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
import psycopg
from psycopg.rows import dict_row
from email.mime.text import MIMEText

# -------------------------
# Config & logging
# -------------------------
APP_NAME = os.getenv("APP_NAME", "quantbrief")
DATABASE_URL = os.getenv("DATABASE_URL")
FROM_EMAIL = os.getenv("FROM_EMAIL", "")
TO_EMAILS = [e.strip() for e in os.getenv("TO_EMAILS", "").split(",") if e.strip()]
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587") or "587")
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
RESOLVE_TIMEOUT = float(os.getenv("RESOLVE_TIMEOUT", "8"))
USER_AGENT = os.getenv("USER_AGENT", f"{APP_NAME}/1.0")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(APP_NAME)

BANNED_HOSTS = {
    "marketbeat.com",
    "newser.com",
}
extra_banned = {h.strip() for h in os.getenv("BANNED_HOSTS_EXTRA", "").split(",") if h.strip()}
BANNED_HOSTS |= extra_banned

app = FastAPI(title=APP_NAME)


# -------------------------
# Helpers
# -------------------------
def as_utc(d: Optional[dt.datetime]) -> Optional[dt.datetime]:
    if d is None:
        return None
    if d.tzinfo is None:
        return d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def normalize_host(netloc: str) -> str:
    return netloc.lower().lstrip("www.")


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:140]


def escape_html(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def canon_url(u: str) -> str:
    # Strip fragments & trivial query noise
    try:
        pr = urlparse(u)
        q = pr.query
        # keep all query; caller may already give us a clean link
        clean = pr._replace(fragment="").geturl()
        return clean
    except Exception:
        return u


def extract_google_news_target(u: str) -> Optional[str]:
    try:
        pr = urlparse(u)
        if pr.netloc.endswith("news.google.com"):
            qs = parse_qs(pr.query)
            if "url" in qs and qs["url"]:
                return unquote(qs["url"][0])
    except Exception:
        pass
    return None


def resolve_url(u: str) -> Tuple[str, Optional[str]]:
    # try to unwrap Google News redirect
    target = extract_google_news_target(u) or u
    target = canon_url(target)
    try:
        h = requests.head(
            target,
            timeout=RESOLVE_TIMEOUT,
            allow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        final = h.url or target
    except Exception:
        try:
            g = requests.get(
                target,
                timeout=RESOLVE_TIMEOUT,
                allow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            )
            final = g.url or target
        except Exception:
            final = target
    host = normalize_host(urlparse(final).netloc)
    return canon_url(final), host


def is_banned_host(host: str) -> bool:
    host = host.lower()
    return any(host == b or host.endswith("." + b) for b in BANNED_HOSTS)


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def compute_fingerprint(url_canonical: Optional[str], url: str, title: Optional[str]) -> Optional[str]:
    # Stable across re-ingests: canonical-url|title (fallbacks included)
    key = f"{url_canonical or url}|{(title or '').strip()}"
    if not key.strip():
        return None
    return sha1_hex(key)


# -------------------------
# DB: connections & migrations
# -------------------------
def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)


SCHEMA_MIGRATIONS: List[Tuple[str, str]] = [
    (
        "0001_base",
        """
        CREATE TABLE IF NOT EXISTS schema_version(
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS feed (
            id SERIAL PRIMARY KEY,
            url TEXT UNIQUE NOT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS found_url (
            id BIGSERIAL PRIMARY KEY,
            url TEXT NOT NULL,
            url_canonical TEXT,
            host TEXT,
            title TEXT,
            title_slug TEXT,
            summary TEXT,
            published_at TIMESTAMPTZ,
            score DOUBLE PRECISION DEFAULT 0,
            feed_id INTEGER REFERENCES feed(id) ON DELETE SET NULL,
            fingerprint TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS ix_found_url_published_at ON found_url(published_at DESC NULLS LAST);
        CREATE INDEX IF NOT EXISTS ix_found_url_host ON found_url(host);
        CREATE INDEX IF NOT EXISTS ix_found_url_title_slug ON found_url(title_slug);
        CREATE INDEX IF NOT EXISTS ix_found_url_url_canonical ON found_url(url_canonical);
        CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_url ON found_url(url);
        CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_fingerprint
          ON found_url(fingerprint) WHERE fingerprint IS NOT NULL;
        """,
    ),
    (
        "0002_seed_feeds",
        """
        INSERT INTO feed(url, active) VALUES
          ('https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+-site:newser.com+-site:marketbeat.com+when:3d&hl=en-US&gl=US&ceid=US:en', TRUE),
          ('https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+when:7d&hl=en-US&gl=US&ceid=US:en', TRUE)
        ON CONFLICT (url) DO UPDATE SET active=EXCLUDED.active;
        """,
    ),
]


def run_migrations(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS schema_version(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now());")
        cur.execute("SELECT version FROM schema_version;")
        applied = {r["version"] for r in cur.fetchall()}
        for version, sql in SCHEMA_MIGRATIONS:
            if version in applied:
                continue
            for stmt in [s.strip() for s in sql.strip().split(";") if s.strip()]:
                cur.execute(stmt + ";")
            cur.execute("INSERT INTO schema_version(version) VALUES (%s);", (version,))
            log.info("Applied migration %s", version)


# Back-compat: if older handlers still call this, keep it as a thin wrapper
def ensure_schema_and_seed(conn) -> None:
    log.info("Ensuring schema via migrations…")
    run_migrations(conn)


# -------------------------
# Ingest
# -------------------------
def parse_feed(url: str):
    # feedparser supports request_headers kwarg in dict form (for remote URLs)
    fp = feedparser.parse(url, request_headers={"User-Agent": USER_AGENT})
    return fp.entries or []


def parse_datetime_guess(entry) -> Optional[dt.datetime]:
    for k in ("published_parsed", "updated_parsed"):
        t = getattr(entry, k, None) or entry.get(k) if isinstance(entry, dict) else None
        if t:
            try:
                # struct_time → aware UTC
                ts = dt.datetime(*t[:6], tzinfo=dt.timezone.utc)
                return ts
            except Exception:
                pass
    # Try published string if present (feedparser often parses)
    txt = getattr(entry, "published", None) or getattr(entry, "updated", None) or ""
    try:
        # last resort: let feedparser date parser have done its job; else None
        return None
    except Exception:
        return None


def score_item(published_at: Optional[dt.datetime], title: str, host: str) -> float:
    base = 0.0
    if published_at:
        hours = (now_utc() - as_utc(published_at)).total_seconds() / 3600.0
        base += max(0.0, 96.0 - hours)  # fresher → higher
    if title:
        base += min(10.0, max(0.0, 0.05 * len(title)))  # small bump for descriptive titles
    if host.endswith("news.google.com"):
        base -= 5.0  # prefer resolved originals
    return round(base, 4)


def insert_found_url(conn, row: dict) -> bool:
    # First try conflict on URL; if unique-fingerprint trips, catch & ignore
    sql = """
        INSERT INTO found_url (url, url_canonical, host, title, title_slug, summary, published_at, score, feed_id, fingerprint)
        VALUES (%(url)s, %(url_canonical)s, %(host)s, %(title)s, %(title_slug)s, %(summary)s, %(published_at)s, %(score)s, %(feed_id)s, %(fingerprint)s)
        ON CONFLICT (url) DO NOTHING
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, row)
            return cur.rowcount > 0
    except psycopg.errors.UniqueViolation:
        # This can happen due to the unique partial index on fingerprint → treat as duplicate
        return False


def do_ingest(minutes: int) -> dict:
    scanned = 0
    inserted = 0
    pruned = 0  # reserved for future

    with get_conn() as conn:
        run_migrations(conn)
        # fetch active feeds
        with conn.cursor() as cur:
            cur.execute("SELECT id, url FROM feed WHERE active IS TRUE;")
            feeds = cur.fetchall()

        for f in feeds:
            feed_id, feed_url = f["id"], f["url"]
            log.info("Parsing feed: %s", feed_url)
            try:
                entries = parse_feed(feed_url)
            except Exception as ex:
                log.warning("Feed parse failed: %s (%s)", feed_url, ex)
                continue

            log.info("Feed entries: %s", len(entries))
            for e in entries:
                scanned += 1
                link = getattr(e, "link", None) or getattr(e, "id", None) or ""
                title = getattr(e, "title", "") or ""
                summary = getattr(e, "summary", None) or getattr(e, "description", None)
                published_at = parse_datetime_guess(e)

                # Resolve & filter
                final_url, host = resolve_url(link or "")
                if not final_url or not host or is_banned_host(host):
                    continue

                url_canonical = final_url
                fp = compute_fingerprint(url_canonical, link or url_canonical, title)
                title_slug = slugify(title) if title else None
                score = score_item(published_at, title, host)

                row = {
                    "url": link or final_url,
                    "url_canonical": url_canonical,
                    "host": host,
                    "title": title,
                    "title_slug": title_slug,
                    "summary": summary,
                    "published_at": as_utc(published_at),
                    "score": score,
                    "feed_id": feed_id,
                    "fingerprint": fp,
                }
                if insert_found_url(conn, row):
                    inserted += 1

    return {"inserted": inserted, "scanned": scanned, "pruned": pruned}


# -------------------------
# Digest & email
# -------------------------
def select_digest_rows(conn, minutes: int) -> List[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH base AS (
              SELECT *,
                     COALESCE(fingerprint, url) AS grp
              FROM found_url
              WHERE COALESCE(published_at, created_at) >= now() - (%s || ' minutes')::interval
            ),
            ranked AS (
              SELECT base.*,
                     ROW_NUMBER() OVER(
                       PARTITION BY grp
                       ORDER BY published_at DESC NULLS LAST, score DESC, id DESC
                     ) AS rn
              FROM base
            )
            SELECT * FROM ranked WHERE rn = 1
            ORDER BY COALESCE(published_at, created_at) DESC NULLS LAST, score DESC, id DESC
            """,
            (minutes,),
        )
        return cur.fetchall()


def format_email_html(subject: str, rows: List[dict]) -> Tuple[str, str]:
    html_parts = [
        f"<h2>{escape_html(subject)}</h2>",
        "<ul>",
    ]
    text_lines = [subject, ""]

    for r in rows:
        href = r.get("url_canonical") or r.get("url")
        host = normalize_host(urlparse(href).netloc) if href else (r.get("host") or "—")
        ts = r.get("published_at")
        ts_txt = as_utc(ts).strftime("%Y-%m-%d %H:%MZ") if ts else "—"
        score = r.get("score") or 0.0
        title = r.get("title") or href or "(untitled)"
        html_parts.append(
            f'<li>[{ts_txt}] ({escape_html(host)}) [score {score:.2f}] '
            f'<a href="{escape_html(href)}">{escape_html(title)}</a></li>'
        )
        text_lines.append(f"- [{ts_txt}] ({host}) [score {score:.2f}] {title}\n  {href}")

    html_parts.append("</ul>")
    return "\n".join(html_parts), "\n".join(text_lines)


def send_email(subject: str, html_body: str, text_body: str) -> None:
    if not (FROM_EMAIL and TO_EMAILS and SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS):
        log.warning("SMTP not fully configured; skipping email send.")
        return

    msg = MIMEText(text_body, "plain", _charset="utf-8")
    alt = MIMEText(html_body, "html", _charset="utf-8")

    # Build a multipart/alternative manually
    from email.mime.multipart import MIMEMultipart
    outer = MIMEMultipart("alternative")
    outer["Subject"] = subject
    outer["From"] = FROM_EMAIL
    outer["To"] = ", ".join(TO_EMAILS)
    outer["Reply-To"] = FROM_EMAIL
    outer.attach(msg)
    outer.attach(alt)

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
        server.starttls(context=context)
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(FROM_EMAIL, TO_EMAILS, outer.as_string())
    log.info("Email sent to %s", TO_EMAILS)


# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return f"{APP_NAME} up"


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
        return "ok"
    except Exception as e:
        return PlainTextResponse(f"not ok: {e}", status_code=500)


@app.post("/admin/init", response_class=PlainTextResponse)
def admin_init(request: Request):
    with get_conn() as conn:
        run_migrations(conn)
    log.info("Schema ensured; seed feeds upserted.")
    return "Schema ensured; seed feeds upserted."


@app.post("/cron/ingest", response_class=JSONResponse)
def cron_ingest(minutes: int = 7 * 24 * 60):
    res = do_ingest(minutes=minutes)
    log.info("Ingest complete. inserted=%s scanned=%s pruned=%s", res["inserted"], res["scanned"], res["pruned"])
    return res


@app.get("/admin/preview-digest", response_class=HTMLResponse)
def admin_preview_digest(minutes: int = 7 * 24 * 60):
    with get_conn() as conn:
        run_migrations(conn)
        rows = select_digest_rows(conn, minutes=minutes)
    subj = f"{APP_NAME} digest — last {minutes} minutes — {len(rows)} items"
    html, _ = format_email_html(subj, rows)
    return html


@app.post("/cron/digest", response_class=PlainTextResponse)
def cron_digest(minutes: int = 7 * 24 * 60):
    with get_conn() as conn:
        run_migrations(conn)
        rows = select_digest_rows(conn, minutes=minutes)
    subj = f"{APP_NAME} digest — last {minutes} minutes — {len(rows)} items"
    html, text = format_email_html(subj, rows)
    send_email(subj, html, text)
    return f"Digest attempted; items={len(rows)}"


@app.post("/admin/test-email", response_class=PlainTextResponse)
def admin_test_email():
    subject = f"{APP_NAME} test email"
    html_body = "<p>This is a test email from quantbrief.</p>"
    text_body = "This is a test email from quantbrief."
    try:
        send_email(subject, html_body, text_body)
        log.info("Email sent to %s (subject=%r)", TO_EMAILS, subject)
        return "Test email sent (or skipped if SMTP not configured)."
    except Exception as ex:
        log.warning("%s:test email failed: %s", APP_NAME, ex)
        return PlainTextResponse(f"Test email failed: {ex}", status_code=500)
