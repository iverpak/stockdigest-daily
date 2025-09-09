# app.py
import os
import re
import ssl
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
from email.mime.multipart import MIMEMultipart

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

BANNED_HOSTS = {"marketbeat.com", "newser.com"}
extra_banned = {h.strip() for h in os.getenv("BANNED_HOSTS_EXTRA", "").split(",") if h.strip()}
BANNED_HOSTS |= extra_banned

app = FastAPI(title=APP_NAME)

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
    try:
        pr = urlparse(u)
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
    target = extract_google_news_target(u) or u
    target = canon_url(target)
    try:
        h = requests.head(target, timeout=RESOLVE_TIMEOUT, allow_redirects=True, headers={"User-Agent": USER_AGENT})
        final = h.url or target
    except Exception:
        try:
            g = requests.get(target, timeout=RESOLVE_TIMEOUT, allow_redirects=True, headers={"User-Agent": USER_AGENT})
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
    key = f"{url_canonical or url}|{(title or '').strip()}"
    if not key.strip():
        return None
    return sha1_hex(key)

def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)

SCHEMA_MIGRATIONS: List[tuple[str, str]] = [
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
        CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_fingerprint ON found_url(fingerprint) WHERE fingerprint IS NOT NULL;
        """,
    ),
    (
        "0001a_compat_existing",
        """
        -- feed
        CREATE TABLE IF NOT EXISTS feed (id SERIAL PRIMARY KEY, url TEXT UNIQUE NOT NULL);
        ALTER TABLE feed ADD COLUMN IF NOT EXISTS active BOOLEAN;
        UPDATE feed SET active = TRUE WHERE active IS NULL;
        ALTER TABLE feed ALTER COLUMN active SET DEFAULT TRUE;
        ALTER TABLE feed ALTER COLUMN active SET NOT NULL;
        ALTER TABLE feed ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ;
        UPDATE feed SET created_at = now() WHERE created_at IS NULL;
        ALTER TABLE feed ALTER COLUMN created_at SET DEFAULT now();
        ALTER TABLE feed ALTER COLUMN created_at SET NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS ux_feed_url ON feed(url);

        -- found_url
        CREATE TABLE IF NOT EXISTS found_url (id BIGSERIAL PRIMARY KEY, url TEXT NOT NULL);
        ALTER TABLE found_url ADD COLUMN IF NOT EXISTS url_canonical TEXT;
        ALTER TABLE found_url ADD COLUMN IF NOT EXISTS host TEXT;
        ALTER TABLE found_url ADD COLUMN IF NOT EXISTS title TEXT;
        ALTER TABLE found_url ADD COLUMN IF NOT EXISTS title_slug TEXT;
        ALTER TABLE found_url ADD COLUMN IF NOT EXISTS summary TEXT;
        ALTER TABLE found_url ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ;
        ALTER TABLE found_url ADD COLUMN IF NOT EXISTS score DOUBLE PRECISION;
        ALTER TABLE found_url ADD COLUMN IF NOT EXISTS feed_id INTEGER;
        ALTER TABLE found_url ADD COLUMN IF NOT EXISTS fingerprint TEXT;
        ALTER TABLE found_url ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ;
        UPDATE found_url SET created_at = now() WHERE created_at IS NULL;
        ALTER TABLE found_url ALTER COLUMN created_at SET DEFAULT now();
        ALTER TABLE found_url ALTER COLUMN created_at SET NOT NULL;

        CREATE INDEX IF NOT EXISTS ix_found_url_published_at ON found_url(published_at DESC NULLS LAST);
        CREATE INDEX IF NOT EXISTS ix_found_url_host ON found_url(host);
        CREATE INDEX IF NOT EXISTS ix_found_url_title_slug ON found_url(title_slug);
        CREATE INDEX IF NOT EXISTS ix_found_url_url_canonical ON found_url(url_canonical);
        CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_url ON found_url(url);
        CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_fingerprint ON found_url(fingerprint) WHERE fingerprint IS NOT NULL;
        """,
    ),
    (
        "0001b_compat_feed_name",
        """
        ALTER TABLE feed ADD COLUMN IF NOT EXISTS name TEXT;
        ALTER TABLE feed ALTER COLUMN name SET DEFAULT 'feed';
        UPDATE feed SET name = 'feed' WHERE name IS NULL;
        """,
    ),
    # NEW: fix legacy FK to source_feed, mirror data, and repoint FK to feed
    (
        "0001c_fk_source_feed_compat",
        """
        -- If a legacy 'source_feed' exists, ensure it's present (no-op if already there)
        CREATE TABLE IF NOT EXISTS source_feed (
            id SERIAL PRIMARY KEY,
            name TEXT,
            url TEXT UNIQUE NOT NULL,
            active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT now()
        );

        -- Copy missing IDs from source_feed -> feed (preserve IDs so existing found_url.feed_id remains valid)
        INSERT INTO feed(id, url, active, created_at, name)
        SELECT sf.id, sf.url, COALESCE(sf.active, TRUE), COALESCE(sf.created_at, now()), COALESCE(sf.name, 'feed')
        FROM source_feed sf
        LEFT JOIN feed f ON f.id = sf.id
        WHERE f.id IS NULL
        ON CONFLICT (id) DO NOTHING;

        -- Create inactive placeholders for any dangling found_url.feed_id with no feed row
        INSERT INTO feed(id, url, active, created_at, name)
        SELECT DISTINCT fu.feed_id, 'legacy://feed/' || fu.feed_id, FALSE, now(), 'legacy'
        FROM found_url fu
        LEFT JOIN feed f ON f.id = fu.feed_id
        WHERE fu.feed_id IS NOT NULL AND f.id IS NULL
        ON CONFLICT (id) DO NOTHING;

        -- Re-point FK: drop any legacy FK (e.g., to source_feed) and add FK to feed(id)
        ALTER TABLE found_url DROP CONSTRAINT IF EXISTS found_url_feed_id_fkey;
        ALTER TABLE found_url
          ADD CONSTRAINT found_url_feed_id_fkey
          FOREIGN KEY (feed_id) REFERENCES feed(id) ON DELETE SET NULL;
        """,
    ),
    (
        "0002_seed_feeds",
        """
        INSERT INTO feed(url, active, name) VALUES
          ('https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+-site:newser.com+-site:marketbeat.com+when:3d&hl=en-US&gl=US&ceid=US:en',
           TRUE, 'Google News: Talen Energy / TLN (3d, excluding MarketBeat & Newser)'),
          ('https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+when:7d&hl=en-US&gl=US&ceid=US:en',
           TRUE, 'Google News: Talen Energy / TLN (7d)')
        ON CONFLICT (url) DO UPDATE
          SET active = EXCLUDED.active,
              name = COALESCE(feed.name, EXCLUDED.name);
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

def ensure_schema_and_seed(conn) -> None:
    log.info("Ensuring schema via migrations…")
    run_migrations(conn)

def parse_feed(url: str):
    return (feedparser.parse(url, request_headers={"User-Agent": USER_AGENT}).entries) or []

def parse_datetime_guess(entry) -> Optional[dt.datetime]:
    for k in ("published_parsed", "updated_parsed"):
        t = getattr(entry, k, None) or (entry.get(k) if isinstance(entry, dict) else None)
        if t:
            try:
                return dt.datetime(*t[:6], tzinfo=dt.timezone.utc)
            except Exception:
                pass
    return None

def score_item(published_at: Optional[dt.datetime], title: str, host: str) -> float:
    base = 0.0
    if published_at:
        hours = (now_utc() - as_utc(published_at)).total_seconds() / 3600.0
        base += max(0.0, 96.0 - hours)
    if title:
        base += min(10.0, max(0.0, 0.05 * len(title)))
    if host.endswith("news.google.com"):
        base -= 5.0
    return round(base, 4)

def insert_found_url(conn, row: dict) -> bool:
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
        return False

def do_ingest(minutes: int) -> dict:
    scanned = 0
    inserted = 0
    pruned = 0

    with get_conn() as conn:
        run_migrations(conn)

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

def select_digest_rows(conn, minutes: int) -> List[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH base AS (
              SELECT *,
                     COALESCE(fingerprint, url) AS grp
              FROM found_url
              WHERE COALESCE(published_at, created_at) >= now() - (%s * interval '1 minute')
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

def format_email_html(subject: str, rows: List[dict]) -> tuple[str, str]:
    html_parts = [f"<h2>{escape_html(subject)}</h2>", "<ul>"]
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

def send_email(subject: str, html_body: str, text_body: str) -> bool:
    if not (FROM_EMAIL and TO_EMAILS and SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS):
        log.warning("SMTP not fully configured; skipping email send.")
        return False

    outer = MIMEMultipart("alternative")
    outer["Subject"] = subject
    outer["From"] = FROM_EMAIL
    outer["To"] = ", ".join(TO_EMAILS)
    outer["Reply-To"] = FROM_EMAIL
    outer.attach(MIMEText(text_body, "plain", _charset="utf-8"))
    outer.attach(MIMEText(html_body, "html", _charset="utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
        server.starttls(context=context)
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(FROM_EMAIL, TO_EMAILS, outer.as_string())
    log.info("Email sent to %s", TO_EMAILS)
    return True

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
    sent = send_email(subj, html, text)
    return f"Digest {'sent' if sent else 'skipped'}; items={len(rows)}"

@app.post("/admin/test-email", response_class=PlainTextResponse)
def admin_test_email():
    subject = f"{APP_NAME} test email"
    html_body = "<p>This is a test email from quantbrief.</p>"
    text_body = "This is a test email from quantbrief."
    try:
        sent = send_email(subject, html_body, text_body)
        if sent:
            return "Test email sent."
        else:
            return "SMTP not fully configured; skipped sending."
    except Exception as ex:
        log.warning("%s: test email failed: %s", APP_NAME, ex)
        return PlainTextResponse(f"Test email failed: {ex}", status_code=500)
