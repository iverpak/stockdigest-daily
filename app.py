import os
import re
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import feedparser
import psycopg
import requests
from fastapi import FastAPI, Request
from email.message import EmailMessage
import smtplib

# -----------------------------------------------------------------------------
# Config & Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger("quantbrief")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required")

# Recipients (comma-separated)
RECIPIENTS = [e.strip() for e in os.getenv("QUANTBRIEF_RECIPIENTS", "").split(",") if e.strip()]
FROM_EMAIL = os.getenv("SMTP_FROM", "quantbrief@no-reply.local")
FROM_NAME = os.getenv("SMTP_FROM_NAME", "Quantbrief")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "true").lower() in ("1", "true", "yes")

USER_AGENT = os.getenv("USER_AGENT", "QuantbriefBot/1.0 (+https://quantbrief.ca)")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "6.0"))

# Allow adding/removing banned hosts via env
DEFAULT_BANNED = {
    # noisy/low-signal syndicators & social
    "marketbeat.com", "newser.com", "linkedin.com", "facebook.com", "youtube.com", "youtu.be",
    "msn.com", "tipranks.com", "simplywall.st",
    # spammy mirroring domains seen in your sample
    "js.signavitae.com", "epayslip.grz.gov.zm", "mehadschools.ir", "krjc.org", "howcome.com.tw",
    "alumnimallrandgo.up.ac.za", "primatevets.org", "valueinvestorinsight.com", "giro.fr",
    "tw13zhi.com", "zyrofisherb2b.co.uk", "randgo", "clientele.co.za", "thjb.com.tw",
    "samsconsult.com", "westganews.net",
    # finance quote spam
    "finance.yahoo.com", "uk.finance.yahoo.com", "ca.finance.yahoo.com", "yahoo.com",
    # aggregation that you don’t want in the email (we’ll resolve to the real host anyway)
    "news.google.com"
}
ENV_BANNED = {h.strip().lower() for h in os.getenv("QUANTBRIEF_BANNED_HOSTS", "").split(",") if h.strip()}
BANNED_HOSTS = DEFAULT_BANNED.union(ENV_BANNED)

SEED_FEEDS: List[Tuple[str, str]] = [
    # Example seed – your Google News query already tries to exclude MarketBeat/Newser,
    # but we still filter after resolving redirects.
    (
        "Google News — TLN",
        "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+when:7d&hl=en-US&gl=US&ceid=US:en",
    ),
]

# -----------------------------------------------------------------------------
# DB: schema (tables, migrations, indexes)
# -----------------------------------------------------------------------------
SCHEMA_TABLES = """
CREATE TABLE IF NOT EXISTS feed (
    id         SERIAL PRIMARY KEY,
    name       TEXT NOT NULL,
    url        TEXT NOT NULL UNIQUE,
    enabled    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
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

SCHEMA_MIGRATIONS = """
ALTER TABLE found_url ADD COLUMN IF NOT EXISTS url_canonical  TEXT;
ALTER TABLE found_url ADD COLUMN IF NOT EXISTS host           TEXT;
ALTER TABLE found_url ADD COLUMN IF NOT EXISTS title          TEXT;
ALTER TABLE found_url ADD COLUMN IF NOT EXISTS title_slug     TEXT;
ALTER TABLE found_url ADD COLUMN IF NOT EXISTS summary        TEXT;
ALTER TABLE found_url ADD COLUMN IF NOT EXISTS published_at   TIMESTAMPTZ;
ALTER TABLE found_url ADD COLUMN IF NOT EXISTS score          DOUBLE PRECISION DEFAULT 0.0;
ALTER TABLE found_url ADD COLUMN IF NOT EXISTS feed_id        INTEGER;
ALTER TABLE found_url ADD COLUMN IF NOT EXISTS fingerprint    TEXT;
"""

SCHEMA_INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_url ON found_url(url);
CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_url_canonical ON found_url(url_canonical);
CREATE UNIQUE INDEX IF NOT EXISTS ux_found_url_fingerprint ON found_url(fingerprint);

CREATE INDEX IF NOT EXISTS ix_found_url_host ON found_url(host);
CREATE INDEX IF NOT EXISTS ix_found_url_title_slug ON found_url(title_slug);
CREATE INDEX IF NOT EXISTS ix_found_url_published_at ON found_url(published_at DESC);
"""

def get_conn():
    return psycopg.connect(DATABASE_URL)

def exec_sql(conn, sql: str):
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()

def ensure_schema_and_seed(conn):
    # 1) Tables
    exec_sql(conn, SCHEMA_TABLES)
    # 2) Migrations (safe if already applied)
    exec_sql(conn, SCHEMA_MIGRATIONS)
    # 3) Indexes
    exec_sql(conn, SCHEMA_INDEXES)
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

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def normalize_host(host: Optional[str]) -> Optional[str]:
    if not host:
        return host
    h = host.strip().lower()
    if h.startswith("www."):
        h = h[4:]
    return h

def is_banned_host(host: Optional[str], title: Optional[str] = None) -> bool:
    h = normalize_host(host)
    if not h:
        return False
    # direct hostname ban
    if any(h == b or h.endswith("." + b) for b in BANNED_HOSTS):
        return True
    # crude " - Source" trailer in titles
    if title and " - " in title:
        src = title.rsplit(" - ", 1)[-1].strip().lower()
        src = src.replace("—", "-").replace("–", "-")
        src = src.replace(" ", "")
        for b in BANNED_HOSTS:
            base = b.replace(".", "")
            if src.endswith(base):
                return True
    return False

UTM_PARAMS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","oc","hl","gl","ceid","ved","usg"}
def strip_tracking(u: str) -> str:
    try:
        p = urlparse(u)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k not in UTM_PARAMS and not k.startswith("utm_")]
        clean = p._replace(query=urlencode(q, doseq=True))
        # remove trailing slash except root
        path = clean.path
        if path.endswith("/") and path != "/":
            clean = clean._replace(path=path.rstrip("/"))
        return urlunparse(clean)
    except Exception:
        return u

def slugify(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    s = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE)
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s[:120] if len(s) > 120 else s

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def resolve_canonical(url: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Returns (final_url, host, maybe_error)
    - If it's a Google News redirector, follow to publisher.
    - Otherwise, return a cleaned version of the same URL.
    """
    try:
        u = strip_tracking(url)
        host = normalize_host(urlparse(u).netloc)
        if host == "news.google.com":
            # Follow redirects to get publisher
            session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT
            # GET with redirects to capture final response.url quickly
            resp = session.get(u, allow_redirects=True, timeout=HTTP_TIMEOUT)
            final_url = strip_tracking(resp.url)
            final_host = normalize_host(urlparse(final_url).netloc)
            return final_url, final_host, None
        else:
            return u, normalize_host(urlparse(u).netloc), None
    except Exception as e:
        return url, normalize_host(urlparse(url).netloc), str(e)

def as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def parse_entry_published(entry) -> Optional[datetime]:
    # feedparser gives .published_parsed (time.struct_time) or .updated_parsed
    ts = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if ts:
        try:
            return datetime(*ts[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    # fallback to now
    return datetime.now(timezone.utc)

# -----------------------------------------------------------------------------
# Ingestion
# -----------------------------------------------------------------------------
def fetch_and_ingest(conn, minutes: int = 60*24*7, min_score: float = 0.0) -> Tuple[int, int, int]:
    inserted = 0
    scanned = 0
    pruned = 0

    headers = {"User-Agent": USER_AGENT}
    with conn.cursor() as cur:
        cur.execute("SELECT id, name, url FROM feed WHERE enabled = TRUE ORDER BY id")
        feeds = cur.fetchall()

    for feed_id, name, url in feeds:
        LOG.info("parsed feed: %s", url)
        d = feedparser.parse(url, request_headers=headers)
        entries = getattr(d, "entries", []) or []
        LOG.info("feed entries: %d", len(entries))
        for e in entries:
            scanned += 1
            link = getattr(e, "link", None)
            title = getattr(e, "title", None)
            if not link or not title:
                continue

            pub_dt = parse_entry_published(e)
            url_final, host_final, err = resolve_canonical(link)
            if err:
                LOG.debug("resolve error for %s: %s", link, err)

            # Now filter banned on final host OR title suffix
            if is_banned_host(host_final, title):
                pruned += 1
                continue

            # Fingerprint by canonical URL primarily; fall back to cleaned link
            fp_basis = url_final or strip_tracking(link)
            fingerprint = sha1_hex(fp_basis)

            row = {
                "url": link,
                "url_canonical": url_final,
                "host": host_final,
                "title": title,
                "title_slug": slugify(title),
                "summary": getattr(e, "summary", None) or getattr(e, "description", None),
                "published_at": pub_dt,
                "score": float(getattr(e, "score", 0.0) or 0.0),
                "feed_id": feed_id,
                "fingerprint": fingerprint,
            }

            # Insert (de-duplicates by url/url_canonical/fingerprint)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO found_url
                        (url, url_canonical, host, title, title_slug, summary,
                         published_at, score, feed_id, fingerprint)
                    VALUES (%(url)s, %(url_canonical)s, %(host)s, %(title)s, %(title_slug)s, %(summary)s,
                            %(published_at)s, %(score)s, %(feed_id)s, %(fingerprint)s)
                    ON CONFLICT (url) DO NOTHING
                    """,
                    row,
                )
                if cur.rowcount == 0 and row["url_canonical"]:
                    cur.execute(
                        """
                        INSERT INTO found_url
                            (url, url_canonical, host, title, title_slug, summary,
                             published_at, score, feed_id, fingerprint)
                        VALUES (%(url)s, %(url_canonical)s, %(host)s, %(title)s, %(title_slug)s, %(summary)s,
                                %(published_at)s, %(score)s, %(feed_id)s, %(fingerprint)s)
                        ON CONFLICT (url_canonical) DO NOTHING
                        """,
                        row,
                    )
                if cur.rowcount == 0:
                    cur.execute(
                        """
                        INSERT INTO found_url
                            (url, url_canonical, host, title, title_slug, summary,
                             published_at, score, feed_id, fingerprint)
                        VALUES (%(url)s, %(url_canonical)s, %(host)s, %(title)s, %(title_slug)s, %(summary)s,
                                %(published_at)s, %(score)s, %(feed_id)s, %(fingerprint)s)
                        ON CONFLICT (fingerprint) DO NOTHING
                        """,
                        row,
                    )
                if cur.rowcount > 0:
                    inserted += 1

    conn.commit()
    return inserted, scanned, pruned

# -----------------------------------------------------------------------------
# Digest (HTML hyperlinks + plain text fallback)
# -----------------------------------------------------------------------------
def select_digest_rows(conn, minutes: int, limit: int = 250):
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    # Build banned host list for SQL filtering
    banned = sorted(BANNED_HOSTS)
    placeholders = ",".join(["%s"] * len(banned)) if banned else None

    base_where = "published_at >= %s"
    params: List = [since]

    if banned:
        base_where += f" AND host NOT IN ({placeholders})"
        params.extend(banned)

    # De-duplicate within window by coalesce key
    sql = f"""
        WITH ranked AS (
            SELECT
                id, url, url_canonical, host, title, title_slug, summary,
                published_at, score, feed_id, fingerprint
              , ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(fingerprint, url_canonical, url, title_slug, id::text)
                    ORDER BY published_at DESC, id DESC
                ) AS rn
            FROM found_url
            WHERE {base_where}
        )
        SELECT *
        FROM ranked
        WHERE rn = 1
        ORDER BY published_at DESC
        LIMIT %s
    """
    params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return rows

def format_email_html(subject: str, rows: List[Tuple]) -> Tuple[str, str]:
    # HTML with hyperlinked title; plain-text fallback includes raw link.
    html_lines = [
        f"<h3>{subject}</h3>",
        "<ul>"
    ]
    text_lines = [subject]

    for r in rows:
        (_id, url, url_canonical, host, title, _slug, _summary, published_at, score, _feed_id, _fp) = r
        href = url_canonical or url
        shown_host = normalize_host(urlparse(href).netloc) or host or "—"
        ts = as_utc(published_at).strftime("%Y-%m-%d %H:%MZ")
        # HTML item with anchor
        html_lines.append(
            f'<li>[{ts}] ({shown_host}) [score {score:.2f}] '
            f'<a href="{href}">{escape_html(title or href)}</a></li>'
        )
        # Plain text fallback
        text_lines.append(f"- [{ts}] ({shown_host}) [score {score:.2f}] {title or href}\n  {href}")

    html_lines.append("</ul>")
    return "\n".join(html_lines), "\n".join(text_lines)

def escape_html(s: str) -> str:
    return (s
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))

def send_email(recipients: List[str], subject: str, html_body: str, text_body: str) -> None:
    if not recipients:
        LOG.warning("No recipients configured; skipping email send.")
        return
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
        LOG.warning("SMTP not fully configured; skipping email send.")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{FROM_NAME} <{FROM_EMAIL}>"
    msg["To"] = ", ".join(recipients)
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
        if SMTP_STARTTLS:
            s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

    LOG.info("Email sent to %s (subject=%r)", recipients, subject)

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title="Quantbrief")

@app.get("/")
def root():
    return {"ok": True, "service": "quantbrief", "time": datetime.now(timezone.utc).isoformat()}

@app.post("/admin/init")
def admin_init():
    try:
        with get_conn() as conn:
            ensure_schema_and_seed(conn)
        return {"ok": True, "message": "Schema ensured; seed feeds upserted."}
    except Exception as e:
        LOG.exception("Unhandled error in /admin/init")
        return {"ok": False, "error": str(e)}

@app.post("/cron/ingest")
def cron_ingest(minutes: int = 60*24*7, min_score: float = 0.0):
    try:
        with get_conn() as conn:
            ensure_schema_and_seed(conn)
            inserted, scanned, pruned = fetch_and_ingest(conn, minutes, min_score)
        LOG.info("Ingest complete. inserted=%d scanned=%d pruned=%d", inserted, scanned, pruned)
        return {"ok": True, "inserted": inserted, "scanned": scanned, "pruned": pruned}
    except Exception as e:
        LOG.exception("Unhandled error in /cron/ingest")
        return {"ok": False, "error": str(e)}

@app.post("/cron/digest")
def cron_digest(minutes: int = 60*24*7, limit: int = 250, subject: Optional[str] = None):
    try:
        with get_conn() as conn:
            ensure_schema_and_seed(conn)
            rows = select_digest_rows(conn, minutes, limit)

        subj = subject or f"Quantbrief digest — last {minutes} minutes ({len(rows)} items)"
        html_body, text_body = format_email_html(subj, rows)
        send_email(RECIPIENTS, subj, html_body, text_body)
        return {"ok": True, "count": len(rows)}
    except Exception as e:
        LOG.exception("Unhandled error in /cron/digest")
        return {"ok": False, "error": str(e)}

@app.post("/admin/test-email")
def admin_test_email():
    try:
        html = "<h3>Quantbrief test email</h3><p>If you can read this, SMTP is set up.</p>"
        send_email(RECIPIENTS or [FROM_EMAIL], "Quantbrief test email", html, "Quantbrief test email")
        return {"ok": True}
    except Exception as e:
        LOG.exception("Unhandled error in /admin/test-email")
        return {"ok": False, "error": str(e)}
