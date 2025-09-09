# app.py
import os
import re
import ssl
import smtplib
import time
import math
import json
import hashlib
import logging
import datetime as dt
from typing import Optional, List, Tuple, Dict
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode, urljoin, unquote
from email.message import EmailMessage
from email.utils import parseaddr, formataddr

import feedparser
import psycopg
from psycopg.rows import dict_row
import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

# ------------------------------------------------------------------------------
# Config & logging
# ------------------------------------------------------------------------------
LOG = logging.getLogger("quantbrief")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

APP_NAME = os.environ.get("APP_NAME", "quantbrief")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "changeme")

DATABASE_URL = os.environ.get("DATABASE_URL") or os.environ.get("PG_DSN") or ""
if not DATABASE_URL:
    # Render provides PG* env vars
    pg_host = os.environ.get("PGHOST")
    pg_user = os.environ.get("PGUSER")
    pg_pass = os.environ.get("PGPASSWORD")
    pg_db   = os.environ.get("PGDATABASE")
    pg_port = os.environ.get("PGPORT", "5432")
    if pg_host and pg_user and pg_pass and pg_db:
        DATABASE_URL = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USERNAME")
SMTP_PASS = os.environ.get("SMTP_PASSWORD")
SMTP_STARTTLS = os.environ.get("SMTP_STARTTLS", "1") not in ("0", "false", "False")
EMAIL_FROM = os.environ.get("EMAIL_FROM") or os.environ.get("SMTP_FROM")  # alias
DIGEST_TO = os.environ.get("DIGEST_TO") or os.environ.get("ADMIN_EMAIL")

DEFAULT_MINUTES = int(os.environ.get("DEFAULT_MINUTES", "10080"))  # 7 days
DEFAULT_RETAIN_DAYS = int(os.environ.get("DEFAULT_RETAIN_DAYS", "30"))
DEFAULT_MIN_SCORE = float(os.environ.get("DEFAULT_MIN_SCORE", "0.55"))
MAX_FETCH_PER_RUN = int(os.environ.get("MAX_FETCH_PER_RUN", "200"))  # guardrails

USER_AGENT = os.environ.get("USER_AGENT", f"{APP_NAME}/1.0 (+https://example.com)")

# ------------------------------------------------------------------------------
# DB schema (idempotent)
# ------------------------------------------------------------------------------
SCHEMA_SQL = r"""
-- Core tables
CREATE TABLE IF NOT EXISTS source_feed (
  id           BIGSERIAL PRIMARY KEY,
  url          TEXT NOT NULL UNIQUE,
  name         TEXT,
  language     TEXT NOT NULL DEFAULT 'en',
  retain_days  INTEGER NOT NULL DEFAULT 30,
  active       BOOLEAN NOT NULL DEFAULT TRUE,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION set_source_feed_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at := NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger
    WHERE tgname = 'trg_source_feed_updated_at'
  ) THEN
    CREATE TRIGGER trg_source_feed_updated_at
    BEFORE UPDATE ON source_feed
    FOR EACH ROW
    EXECUTE FUNCTION set_source_feed_updated_at();
  END IF;
END
$$;

CREATE TABLE IF NOT EXISTS found_url (
  id            BIGSERIAL PRIMARY KEY,
  url           TEXT NOT NULL,
  title         TEXT,
  feed_id       BIGINT NOT NULL REFERENCES source_feed(id) ON DELETE CASCADE,
  language      TEXT NOT NULL DEFAULT 'en',
  published_at  TIMESTAMPTZ NOT NULL,
  found_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (url)
);

CREATE INDEX IF NOT EXISTS idx_found_url_published ON found_url(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_found_url_feed_id ON found_url(feed_id);

-- Scoring / quality metrics
CREATE TABLE IF NOT EXISTS article_metrics (
  found_url_id   BIGINT PRIMARY KEY REFERENCES found_url(id) ON DELETE CASCADE,
  fetched_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  word_count     INTEGER,
  byline         BOOLEAN,
  content_date   TIMESTAMPTZ,
  entity_count   INTEGER,
  is_press_rel   BOOLEAN,
  is_job_or_quote BOOLEAN,
  score          REAL
);

-- Domain policy
CREATE TABLE IF NOT EXISTS domain_policy (
  domain  TEXT PRIMARY KEY,
  action  TEXT NOT NULL CHECK (action IN ('allow','demote','block')),
  score   REAL NOT NULL DEFAULT 0.0,
  notes   TEXT
);

-- Feedback (optional learning loop)
CREATE TABLE IF NOT EXISTS article_feedback (
  found_url_id BIGINT REFERENCES found_url(id) ON DELETE CASCADE,
  vote         SMALLINT NOT NULL, -- +1/-1
  reason       TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY(found_url_id, created_at)
);
"""

SEED_FEEDS = [
    # Your four Google News variants (keep, tweak as desired)
    ("https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN&hl=en-US&gl=US&ceid=US:en", "Google News: Talen Energy OR TLN (all)", "en", DEFAULT_RETAIN_DAYS),
    ("https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+-site:newser.com+-site:marketbeat.com+when:3d&hl=en-US&gl=US&ceid=US:en", "Google News: Talen Energy OR TLN (3d, filtered)", "en", DEFAULT_RETAIN_DAYS),
    ("https://news.google.com/rss/search?q=%28Talen+Energy%29+OR+TLN+-site%3Anewser.com+-site%3Awww.marketbeat.com+-site%3Amarketbeat.com+-site%3Awww.newser.com+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen", "Google News: ((Talen Energy) OR TLN) (7d, filtered B)", "en", DEFAULT_RETAIN_DAYS),
    ("https://news.google.com/rss/search?q=%28%28Talen+Energy%29+OR+TLN%29+-site%3Anewser.com+-site%3Awww.newser.com+-site%3Amarketbeat.com+-site%3Awww.marketbeat.com+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen", "Google News: ((Talen Energy) OR TLN) (7d, filtered A)", "en", DEFAULT_RETAIN_DAYS),
]

SEED_DOMAIN_POLICY = [
    # allow/boost (score in 0..1)
    ("reuters.com", "allow", 1.0, "Top tier wire"),
    ("ft.com", "allow", 0.9, "Financial Times"),
    ("wsj.com", "allow", 0.9, "Wall Street Journal"),
    ("bloomberg.com", "allow", 0.9, "Bloomberg"),
    ("utilitydive.com", "allow", 0.9, "Utility Dive"),
    ("power-eng.com", "allow", 0.8, "Power Engineering"),
    ("powermag.com", "allow", 0.8, "POWER Magazine"),
    ("datacenterdynamics.com", "allow", 0.8, "DCD"),
    ("heatmap.news", "allow", 0.8, "Heatmap"),
    ("spglobal.com", "allow", 0.8, "S&P Global"),
    ("pjm.com", "allow", 0.8, "PJM"),
    ("ferc.gov", "allow", 0.8, "Regulatory"),
    ("talenenergy.com", "allow", 0.7, "Company IR / PR"),
    ("seekingalpha.com", "allow", 0.6, "Analysis (mixed)"),
    ("eedaily.net", "allow", 0.7, "Politico Pro / E&E (if accessible)"),

    # demote
    ("yahoo.com", "demote", 0.35, "Mirrors/aggregation mixed"),
    ("finance.yahoo.com", "demote", 0.35, "Quotes/finance mirrors"),
    ("nasdaq.com", "demote", 0.35, "Mirrors/SEO"),
    ("simplywall.st", "demote", 0.3, "Automated analysis"),
    ("zacks.com", "demote", 0.3, "Screens/SEO"),
    ("tipranks.com", "demote", 0.25, "Screens/SEO"),

    # block
    ("marketbeat.com", "block", -1.0, "SEO & fund flows spam"),
    ("financialcontent.com", "block", -1.0, "Syndicated quote pages"),
    ("msn.com", "block", -1.0, "Mirrors"),
    ("facebook.com", "block", -1.0, "Social"),
    ("linkedin.com", "block", -1.0, "Jobs/social"),
    ("tiktok.com", "block", -1.0, "Social video"),
    ("indeed.com", "block", -1.0, "Jobs"),
    ("smartrecruiters.com", "block", -1.0, "Jobs"),
    ("youtube.com", "block", -1.0, "Video"),
]

# ------------------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------------------
app = FastAPI()

# ------------------------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------------------------
def db() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, autocommit=False, row_factory=dict_row)

def exec_sql_batch(conn: psycopg.Connection, sql: str):
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()

# ------------------------------------------------------------------------------
# Utilities: URL normalization, domain extraction, Google News target extraction
# ------------------------------------------------------------------------------
RE_UTM = re.compile(r"(utm_[^=&]+|cmp|gclid|fbclid|mc_cid|mc_eid|icid)=", re.I)

def strip_tracking(url: str) -> str:
    try:
        p = urlparse(url)
        qs = parse_qs(p.query, keep_blank_values=True)
        qs = {k: v for k, v in qs.items() if not RE_UTM.search(k)}
        q = urlencode(qs, doseq=True)
        return urlunparse((p.scheme, p.netloc, p.path, p.params, q, ""))
    except Exception:
        return url

def extract_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""

def google_news_target(url: str) -> Optional[str]:
    """
    If the URL is a Google News redirect/article link, try to extract the real target.
    """
    try:
        p = urlparse(url)
        if "news.google." in p.netloc:
            qs = parse_qs(p.query)
            if "url" in qs and qs["url"]:
                return unquote(qs["url"][0])
        return None
    except Exception:
        return None

def canonicalize_url(url: str) -> str:
    url = strip_tracking(url)
    tgt = google_news_target(url)
    if tgt:
        url = strip_tracking(tgt)
    # drop trailing slash if path != "/"
    p = urlparse(url)
    path = p.path
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return urlunparse((p.scheme, p.netloc.lower(), path, p.params, p.query, ""))

# ------------------------------------------------------------------------------
# Layer A: domain/URL heuristics
# ------------------------------------------------------------------------------
DROP_PATH_PATTERNS = [
    r"/jobs?", r"/careers?", r"/hiring", r"/apply", r"/job(s|board)",
    r"linkedin\.com", r"facebook\.com", r"tiktok\.com", r"youtube\.com",
    r"/watch", r"/video", r"/videos",
    r"/option(s)?-?chain", r"/quote", r"/interactive-?chart", r"/holders?",
    r"/financials?", r"/income-?statement", r"/historical",
    r"pressrelease", r"press-release", r"newswire", r"globenewswire", r"prnews",
]

TITLE_SPAM_PHRASES = [
    "buy zone", "ai powered", "momentum", "swing alert", "entry and exit",
    "weekly watchlist", "day trade", "price target &", "fast entry", "high accuracy",
    "profit recap", "trade signal", "trade tips", "buyback activity",
    "portfolio report", "big picture", "bear alert", "volume trigger",
]

def path_blacklisted(url: str) -> bool:
    u = url.lower()
    return any(re.search(p, u) for p in DROP_PATH_PATTERNS)

def title_spammy(title: str) -> bool:
    t = (title or "").lower()
    # very short or weird
    if len(t) < 12:
        return True
    return any(ph in t for ph in TITLE_SPAM_PHRASES)

# ------------------------------------------------------------------------------
# Domain policy helpers
# ------------------------------------------------------------------------------
def ensure_domain_policy(conn: psycopg.Connection):
    with conn.cursor() as cur:
        for d, action, score, notes in SEED_DOMAIN_POLICY:
            cur.execute("""
                INSERT INTO domain_policy(domain, action, score, notes)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (domain) DO UPDATE
                SET action=EXCLUDED.action, score=EXCLUDED.score, notes=EXCLUDED.notes
            """, (d, action, score, notes))
    conn.commit()

def get_domain_policy(conn: psycopg.Connection, domain: str) -> Tuple[str, float]:
    with conn.cursor() as cur:
        cur.execute("SELECT action, score FROM domain_policy WHERE domain=%s", (domain,))
        row = cur.fetchone()
        if row:
            return row["action"], float(row["score"])
    return ("demote", 0.4)  # default neutral-ish

# ------------------------------------------------------------------------------
# Content fetching & scoring (Layer C)
# ------------------------------------------------------------------------------
HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

def fetch_html(url: str, timeout: float = 5.0) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code >= 200 and r.status_code < 300 and "text/html" in r.headers.get("content-type",""):
            return r.text
        return None
    except Exception:
        return None

def extract_published_date(soup: BeautifulSoup) -> Optional[dt.datetime]:
    # schema.org or article meta
    meta_props = [
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "article:published_time"}),
        ("meta", {"itemprop": "datePublished"}),
        ("time", {"itemprop": "datePublished"}),
        ("time", {"datetime": True}),
    ]
    for tag, sel in meta_props:
        el = soup.find(tag, sel)
        if el:
            val = el.get("content") or el.get("datetime") or el.text
            try:
                return dt.datetime.fromisoformat(val.replace("Z","+00:00"))
            except Exception:
                pass
    return None

def extract_main_text(soup: BeautifulSoup) -> Tuple[str, int]:
    # Simple readability heuristic: paragraphs inside article/main/div with > 80 chars
    candidates = []
    for container in soup.find_all(["article", "main", "div", "section"]):
        text = " ".join(p.get_text(" ", strip=True) for p in container.find_all(["p","h2","h3","li"]))
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 400:
            candidates.append(text)
    if not candidates:
        # fallback: all paragraphs
        text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        text = re.sub(r"\s+", " ", text).strip()
        wc = len(text.split())
        return text, wc
    best = max(candidates, key=lambda t: len(t))
    return best, len(best.split())

def detect_byline(soup: BeautifulSoup, text: str) -> bool:
    by_meta = soup.find("meta", {"name": "author"}) or soup.find("meta", {"property": "author"})
    if by_meta and (by_meta.get("content") or "").strip():
        return True
    # visible "By John Smith"
    return bool(re.search(r"\bBy\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text))

def count_entities(text: str) -> int:
    # naive: count distinct capitalized multi-words not at sentence start
    words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text)
    return len(set(words))

def is_press_release(text: str) -> bool:
    t = text.lower()
    flags = ["for immediate release", "forward-looking statements", "non-gaap", "investor relations"]
    return any(f in t for f in flags)

def is_job_or_quote_page(text: str) -> bool:
    t = text.lower()
    if "apply now" in t or "responsibilities" in t or "qualifications" in t:
        return True
    if "options chain" in t or "interactive chart" in t or "bid" in t and "ask" in t:
        return True
    return False

def compute_score(domain_score: float, wc: int, byline: bool, ent: int, article_markup: bool,
                  pr_flag: bool, jq_flag: bool) -> float:
    score = 0.50*domain_score \
          + 0.25*min(1.0, wc/1200.0) \
          + 0.10*(1.0 if byline else 0.0) \
          + 0.10*min(1.0, ent/12.0) \
          + 0.05*(1.0 if article_markup else 0.0) \
          - 0.30*(1.0 if pr_flag else 0.0) \
          - 0.40*(1.0 if jq_flag else 0.0)
    # clamp 0..1
    return max(0.0, min(1.0, score))

def analyze_and_store(conn: psycopg.Connection, row: dict, domain_score: float):
    url = row["url"]
    html = fetch_html(url, timeout=5.0)
    if not html:
        # Store minimal metrics to avoid re-fetching every run
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO article_metrics(found_url_id, word_count, byline, entity_count,
                                            is_press_rel, is_job_or_quote, score)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (found_url_id) DO NOTHING
            """, (row["id"], None, False, None, False, False, 0.0))
        conn.commit()
        return

    soup = BeautifulSoup(html, "html.parser")
    text, wc = extract_main_text(soup)
    byline = detect_byline(soup, text)
    ents = count_entities(text)
    pr = is_press_release(text)
    jq = is_job_or_quote_page(text)
    # Article markup present?
    article_markup = bool(soup.find(["article"]) or soup.find("meta", {"property": "og:type", "content": "article"}))

    # try content date if missing
    content_dt = extract_published_date(soup)

    score = compute_score(domain_score, wc or 0, byline, ents, article_markup, pr, jq)

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO article_metrics(found_url_id, word_count, byline, content_date,
                                        entity_count, is_press_rel, is_job_or_quote, score)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (found_url_id) DO UPDATE
            SET fetched_at=NOW(),
                word_count=EXCLUDED.word_count,
                byline=EXCLUDED.byline,
                content_date=COALESCE(EXCLUDED.content_date, article_metrics.content_date),
                entity_count=EXCLUDED.entity_count,
                is_press_rel=EXCLUDED.is_press_rel,
                is_job_or_quote=EXCLUDED.is_job_or_quote,
                score=EXCLUDED.score
        """, (row["id"], wc, byline, content_dt, ents, pr, jq, score))
    conn.commit()

# ------------------------------------------------------------------------------
# Ingest helpers
# ------------------------------------------------------------------------------
def parse_entry_time(entry) -> Optional[dt.datetime]:
    # feedparser returns time structs; prefer published then updated
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return dt.datetime.fromtimestamp(time.mktime(t), tz=dt.timezone.utc)
    # sometimes only 'published' string exists; try iso parse
    val = entry.get("published") or entry.get("updated")
    if val:
        for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
            try:
                return dt.datetime.strptime(val, fmt).astimezone(dt.timezone.utc)
            except Exception:
                pass
    return None

def upsert_feed(conn: psycopg.Connection, url: str, name: str, language: str, retain_days: int) -> int:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO source_feed (url, name, language, retain_days, active)
            VALUES (%s,%s,%s,%s, TRUE)
            ON CONFLICT (url) DO UPDATE
            SET name=EXCLUDED.name, language=EXCLUDED.language, retain_days=EXCLUDED.retain_days, active=TRUE
            RETURNING id
        """, (url, name, language, retain_days))
        return int(cur.fetchone()["id"])

def list_active_feeds(conn: psycopg.Connection) -> List[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, url,
                   COALESCE(NULLIF(language,''),'en') AS language,
                   COALESCE(NULLIF(name,''), url) AS name,
                   retain_days
            FROM source_feed
            WHERE active = TRUE
            ORDER BY id
        """)
        return list(cur.fetchall())

def insert_found_url(conn: psycopg.Connection, url: str, title: str, feed_id: int,
                     language: str, published_at: dt.datetime) -> Optional[int]:
    with conn.cursor() as cur:
        try:
            cur.execute("""
                INSERT INTO found_url (url, title, feed_id, language, published_at, found_at)
                VALUES (%s,%s,%s,%s,%s,NOW())
                ON CONFLICT (url) DO NOTHING
                RETURNING id
            """, (url, title, feed_id, language, published_at))
            row = cur.fetchone()
            conn.commit()
            return int(row["id"]) if row else None
        except Exception as e:
            conn.rollback()
            LOG.warning("insert_found_url failed: %s", e)
            return None

def prune_old_found_urls(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH del AS (
                DELETE FROM found_url f
                USING source_feed s
                WHERE f.feed_id = s.id
                  AND f.published_at < NOW() - (s.retain_days || ' days')::interval
                RETURNING 1
            )
            SELECT count(*) AS c FROM del
        """)
        c = int(cur.fetchone()["c"])
    conn.commit()
    LOG.warning("prune_old_found_urls: deleted=%d", c)
    return c

# ------------------------------------------------------------------------------
# Email helper
# ------------------------------------------------------------------------------
def send_email(subject: str, html: str, to_addrs: list[str]) -> None:
    display_from = os.getenv("EMAIL_FROM") or os.getenv("SMTP_FROM") or "no-reply@mg.quantbrief.ca"
    name, addr = parseaddr(display_from)               # parses “Name <email>” or plain email
    envelope_from = os.getenv("SMTP_FROM") or addr     # MUST be bare email for SMTP

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((name or addr.split("@")[0], addr))  # pretty header
    msg["To"] = ", ".join(to_addrs)
    msg.set_content("HTML email requires an HTML-capable client.")
    msg.add_alternative(html, subtype="html")

    host = os.getenv("SMTP_HOST", "smtp.mailgun.org")
    port = int(os.getenv("SMTP_PORT", "587"))
    starttls = os.getenv("SMTP_STARTTLS", "1") not in ("0", "false", "False")

    with smtplib.SMTP(host, port, timeout=30) as s:
        if starttls:
            s.starttls()
        user = os.getenv("SMTP_USERNAME")
        pwd = os.getenv("SMTP_PASSWORD")
        if user and pwd:
            s.login(user, pwd)
        s.sendmail(envelope_from, to_addrs, msg.as_string())
# ------------------------------------------------------------------------------
# Digest rendering
# ------------------------------------------------------------------------------
def render_digest_html(items: List[dict], minutes: int) -> str:
    rows = []
    for it in items:
        ts = it["published_at"].strftime("%Y-%m-%d %H:%M")
        score = it.get("score")
        score_part = f" <span style='color:#888'>(score {score:.2f})</span>" if score is not None else ""
        title = (it.get("title") or "").replace("&", "&amp;").replace("<", "&lt;")
        url = it["url"]
        feed = it.get("feed_name") or "feed"
        rows.append(f"<li><a href='{url}'>{title}</a>{score_part} &mdash; "
                    f"<em>{feed}</em> <small>{ts}</small></li>")
    html = f"""
    <html><body>
    <h2>Quantbrief digest — last {minutes} minutes</h2>
    <ol>
      {''.join(rows)}
    </ol>
    </body></html>
    """
    return html

# ------------------------------------------------------------------------------
# Admin endpoints
# ------------------------------------------------------------------------------
def require_admin(req: Request):
    tok = req.headers.get("x-admin-token") or req.headers.get("authorization", "").replace("Bearer ","").strip()
    if not tok or tok != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")

@app.get("/")
def home():
    return PlainTextResponse("OK")

@app.post("/admin/init")
def admin_init(req: Request):
    require_admin(req)
    with db() as conn:
        exec_sql_batch(conn, SCHEMA_SQL)
        # seed feeds
        for url, name, lang, days in SEED_FEEDS:
            upsert_feed(conn, url, name, lang, days)
        ensure_domain_policy(conn)
    LOG.info("Schema ensured; seed feeds upserted.")
    return JSONResponse({"ok": True})

@app.post("/admin/test-email")
def admin_test_email(req: Request):
    require_admin(req)
    ok = send_email("Quantbrief test email", "<b>It works</b>")
    return JSONResponse({"ok": ok})

# ------------------------------------------------------------------------------
# Core: ingest + scoring + digest
# ------------------------------------------------------------------------------
@app.post("/cron/ingest")
def cron_ingest(req: Request, minutes: int = DEFAULT_MINUTES):
    require_admin(req)
    start = time.time()
    inserted = 0
    fetched = 0

    with db() as conn:
        feeds = list_active_feeds(conn)
        for f in feeds:
            url = f["url"]
            try:
                d = feedparser.parse(url)
                LOG.info("parsed feed: %s", url)
                LOG.info("feed entries: %d", len(d.entries))
            except Exception as e:
                LOG.warning("feed parse failed %s: %s", url, e)
                continue

            # intra-feed dedupe by normalized title
            seen_titles = set()

            for e in d.entries:
                pub = parse_entry_time(e)
                if not pub:
                    # try to skip entries without a reliable date (we enforce 7d window)
                    continue
                # enforce last N minutes at ingest time (tight window)
                if pub < dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes):
                    continue

                raw_link = e.get("link") or e.get("id") or ""
                if not raw_link:
                    continue

                link = canonicalize_url(raw_link)
                if path_blacklisted(link):
                    continue
                domain = extract_domain(link)
                action, dscore = get_domain_policy(conn, domain)
                if action == "block":
                    continue

                title = e.get("title") or ""
                norm_title = re.sub(r"\W+", " ", title.lower()).strip()
                if norm_title in seen_titles:
                    continue
                seen_titles.add(norm_title)

                if title_spammy(title):
                    # heavy demotion: keep only if domain is strongly allowed
                    if dscore < 0.75:
                        continue

                # Insert
                fid = insert_found_url(conn, link, title, f["id"], f["language"], pub)
                if fid:
                    inserted += 1

        # prune
        deleted = prune_old_found_urls(conn)

        # fetch & score a subset of newest rows missing metrics
        with conn.cursor() as cur:
            cur.execute("""
                SELECT f.id, f.url, f.title, f.published_at
                FROM found_url f
                LEFT JOIN article_metrics am ON am.found_url_id = f.id
                WHERE am.found_url_id IS NULL
                ORDER BY f.published_at DESC
                LIMIT %s
            """, (MAX_FETCH_PER_RUN,))
            todo = cur.fetchall()

        for row in todo:
            domain = extract_domain(row["url"])
            _, dscore = get_domain_policy(conn, domain)
            analyze_and_store(conn, row, dscore)
            fetched += 1

    LOG.info("ingest done: inserted=%d, fetched_scored=%d, pruned=%d, took=%.2fs",
             inserted, fetched, deleted, time.time() - start)
    return JSONResponse({"ok": True, "inserted": inserted, "fetched_scored": fetched, "pruned": deleted})

@app.post("/cron/digest")
def cron_digest(req: Request,
                minutes: int = DEFAULT_MINUTES,
                min_score: float = DEFAULT_MIN_SCORE,
                limit: int = 200,
                email: int = 1):
    require_admin(req)
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)

    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT f.url, f.title, f.published_at,
                   COALESCE(s.name, s.url) AS feed_name,
                   am.score
            FROM found_url f
            JOIN source_feed s ON s.id = f.feed_id
            LEFT JOIN article_metrics am ON am.found_url_id = f.id
            LEFT JOIN domain_policy dp ON dp.domain = split_part(split_part(f.url, '://', 2), '/', 1)
            WHERE f.published_at >= %s
              AND (COALESCE(am.score, dp.score, 0.0)) >= %s
            ORDER BY COALESCE(am.score, dp.score, 0.0) DESC, f.published_at DESC
            LIMIT %s
        """, (cutoff, min_score, limit))
        rows = list(cur.fetchall())

    html = render_digest_html(rows, minutes)
    if email and rows:
        subj = f"Quantbrief digest — last {minutes} minutes ({len(rows)} items)"
        send_email(subj, html)
    return PlainTextResponse(html, media_type="text/html")
