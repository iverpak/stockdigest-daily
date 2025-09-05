import os
import re
import hmac
import json
import time
import math
import hashlib
import logging
import textwrap
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs, unquote

import requests
import feedparser
import psycopg
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse, HTMLResponse
from jinja2 import Template
from dateutil import tz

# OpenAI SDK v1
from openai import OpenAI

# Optional but helpful content extractor
try:
    import trafilatura
except Exception:
    trafilatura = None

# -----------------------------------------------------------------------------
# Config / Env
# -----------------------------------------------------------------------------
APP_NAME = os.getenv("APP_NAME", "QuantBrief Daily")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # set in Render
TIMEZONE = os.getenv("APP_TZ", "America/Toronto")

DATABASE_URL = os.getenv("DATABASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "600"))

MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = os.getenv("MAILGUN_DOMAIN", "")
MAILGUN_FROM = os.getenv("MAILGUN_FROM", "QuantBrief Daily <daily@mg.quantbrief.ca>")

OWNER_EMAIL = os.getenv("OWNER_EMAIL", "quantbrief.research@gmail.com")

DEFAULT_COMPANY_NAME = os.getenv("DEFAULT_COMPANY_NAME", "Talen Energy")
DEFAULT_TICKER = os.getenv("DEFAULT_TICKER", "TLN")
DEFAULT_SECTOR = os.getenv("DEFAULT_SECTOR", "IPP / Power")

# Safety: HTTP timeouts
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "15"))

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title=APP_NAME)
logger = logging.getLogger("uvicorn.error")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def require_admin(request: Request):
    token = request.headers.get("x-admin-token") or request.headers.get("X-Admin-Token")
    if not token or not hmac.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Forbidden")

DB_DRIVER = "psycopg3"
try:
    import psycopg  # v3
except ModuleNotFoundError:
    DB_DRIVER = "psycopg2"
    import psycopg2 as psycopg  # type: ignore

def get_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    if DB_DRIVER == "psycopg3":
        return psycopg.connect(DATABASE_URL, autocommit=True)
    # psycopg2 fallback
    conn = psycopg.connect(DATABASE_URL, sslmode=os.getenv("PGSSLMODE", "require"))
    conn.autocommit = True
    return conn

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def to_tz(dt_utc: datetime, tz_name: str = TIMEZONE) -> datetime:
    try:
        zone = tz.gettz(tz_name)
        return dt_utc.astimezone(zone)
    except Exception:
        return dt_utc

def parse_google_news_original(link: str) -> str:
    """
    Google News RSS <link> often points to news.google.com, but includes the original via ?url=...
    Extract it if present; otherwise return the given link.
    """
    try:
        p = urlparse(link)
        if ("news.google.com" in (p.netloc or "")) and p.query:
            q = parse_qs(p.query)
            if "url" in q and q["url"]:
                return unquote(q["url"][0])
        return link
    except Exception:
        return link

def hostname_from_url(u: str) -> str:
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""

def now_utc():
    return datetime.now(timezone.utc)

# -----------------------------------------------------------------------------
# DB bootstrap & helpers
# -----------------------------------------------------------------------------
SCHEMA_SQL = """
-- Core tables (match your existing schema names/types)
CREATE TABLE IF NOT EXISTS public.equity (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  ticker TEXT NOT NULL UNIQUE,
  sector TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.watchlist (
  id BIGSERIAL PRIMARY KEY,
  equity_id BIGINT NOT NULL REFERENCES public.equity(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.recipients (
  id BIGSERIAL PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  name TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Articles table (you already have this; this is here for new installs)
CREATE TABLE IF NOT EXISTS public.article (
  id BIGSERIAL PRIMARY KEY,
  url TEXT NOT NULL,
  canonical_url TEXT,
  title TEXT,
  publisher TEXT,
  published_at TIMESTAMPTZ,
  sha256 CHAR(64) NOT NULL UNIQUE,
  raw_html TEXT,
  clean_text TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- link articles to equities
CREATE TABLE IF NOT EXISTS public.article_tag (
  id BIGSERIAL PRIMARY KEY,
  article_id BIGINT NOT NULL REFERENCES public.article(id) ON DELETE CASCADE,
  equity_id BIGINT NOT NULL REFERENCES public.equity(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(article_id, equity_id)
);

-- minimal NLP table to store our short summary + stance
CREATE TABLE IF NOT EXISTS public.article_nlp (
  id BIGSERIAL PRIMARY KEY,
  article_id BIGINT NOT NULL UNIQUE REFERENCES public.article(id) ON DELETE CASCADE,
  what_matters TEXT,
  stance TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Computed domain column (if not present)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name='article' AND column_name='domain'
  ) THEN
    EXECUTE $cmd$
      ALTER TABLE public.article
      ADD COLUMN domain TEXT
      GENERATED ALWAYS AS (
        lower(split_part(regexp_replace(coalesce(canonical_url, url), '^[a-z]+://', ''), '/', 1))
      ) STORED
    $cmd$;
  END IF;
END$$;

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_article_domain ON public.article (domain);
CREATE INDEX IF NOT EXISTS idx_article_published ON public.article (published_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_article_url_unique ON public.article (url);
"""

UPSERT_EQUITY_SQL = """
INSERT INTO public.equity (name, ticker, sector)
VALUES (%s, %s, %s)
ON CONFLICT (ticker) DO UPDATE
SET name=EXCLUDED.name,
    sector=EXCLUDED.sector
RETURNING id;
"""

UPSERT_RECIPIENT_SQL = """
INSERT INTO public.recipients (email, name)
VALUES (%s, %s)
ON CONFLICT (email) DO UPDATE
SET name=EXCLUDED.name
RETURNING id;
"""

INSERT_WATCHLIST_IF_NOT_EXISTS_SQL = """
INSERT INTO public.watchlist (equity_id)
SELECT %s
WHERE NOT EXISTS (SELECT 1 FROM public.watchlist WHERE equity_id=%s);
"""

UPSERT_ARTICLE_SQL = """
INSERT INTO public.article (url, canonical_url, title, publisher, published_at, sha256)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (sha256) DO NOTHING
RETURNING id;
"""

UPSERT_ARTICLE_TAG_SQL = """
INSERT INTO public.article_tag (article_id, equity_id)
VALUES (%s, %s)
ON CONFLICT (article_id, equity_id) DO NOTHING;
"""

UPDATE_ARTICLE_HTML_TEXT_SQL = """
UPDATE public.article
SET raw_html = COALESCE(raw_html, %s),
    clean_text = COALESCE(clean_text, %s)
WHERE id = %s;
"""

UPSERT_ARTICLE_NLP_SQL = """
INSERT INTO public.article_nlp (article_id, what_matters, stance)
VALUES (%s, %s, %s)
ON CONFLICT (article_id) DO UPDATE
SET what_matters=EXCLUDED.what_matters,
    stance=EXCLUDED.stance;
"""

SELECT_WATCHLIST_SQL = """
SELECT e.id, e.name, e.ticker
FROM public.watchlist w
JOIN public.equity e ON e.id = w.equity_id
ORDER BY e.ticker;
"""

SELECT_RECIPIENTS_SQL = "SELECT email, COALESCE(name,'') FROM public.recipients ORDER BY id;"

SELECT_UNSUMMARIZED_SQL = """
SELECT a.id, a.url, a.title, a.publisher, a.published_at, COALESCE(a.clean_text, ''), a.domain
FROM public.article a
LEFT JOIN public.article_nlp n ON n.article_id = a.id
JOIN public.article_tag t ON t.article_id = a.id
WHERE n.article_id IS NULL
  AND t.equity_id = %s
  AND a.published_at >= %s
  AND a.domain NOT LIKE 'lh%%.googleusercontent.com'
  AND a.domain NOT IN ('gstatic.com','fonts.gstatic.com','fonts.googleapis.com','maps.googleapis.com')
ORDER BY a.published_at DESC
LIMIT %s;
"""

SELECT_FOR_DIGEST_SQL = """
SELECT a.id, a.url, a.title, a.publisher, a.published_at, a.domain, n.what_matters, n.stance
FROM public.article a
JOIN public.article_nlp n ON n.article_id = a.id
JOIN public.article_tag t ON t.article_id = a.id
WHERE t.equity_id = %s
  AND a.published_at >= %s
  AND a.domain NOT LIKE 'lh%%.googleusercontent.com'
ORDER BY a.published_at DESC;
"""

DELETE_IMAGE_CDN_SQL = "DELETE FROM public.article WHERE domain LIKE 'lh%.googleusercontent.com';"
DELETE_IMAGE_EXT_SQL = "DELETE FROM public.article WHERE url ~* '\\.(png|jpe?g|gif|webp)(\\?.*)?$';"
DEDUP_URL_SQL = """
DELETE FROM public.article a
USING public.article b
WHERE a.url = b.url
  AND a.id > b.id;
"""

# -----------------------------------------------------------------------------
# Ingestion: Google News
# -----------------------------------------------------------------------------
def build_google_news_feeds(company_name: str, ticker: str):
    """
    Build a small set of Google News RSS searches that bias toward finance & IR sources
    but still pick up industry-level coverage.
    """
    base = "https://news.google.com/rss/search?q="
    tail = "&hl=en-US&gl=US&ceid=US:en"

    queries = [
        f'"{company_name}" OR "{ticker}"',
        f'"{ticker}" stock OR earnings OR guidance',
        f'"{company_name}" site:ir.{ticker.lower()}.com OR site:investors.{ticker.lower()}.com',
        # light industry context
        f'"{company_name}" OR "{ticker}" independent power producer OR electricity prices OR data center power',
    ]
    feeds = [base + requests.utils.quote(q) + tail for q in queries]
    return feeds

def fetch_rss(url: str):
    logger.info(f"Fetching RSS: {url}")
    # feedparser does HTTP for us
    return feedparser.parse(url)

def ingest_google_news(conn, equity_id: int, company_name: str, ticker: str, since_utc: datetime):
    feeds = build_google_news_feeds(company_name, ticker)
    inserted = 0
    for f in feeds:
        feed = fetch_rss(f)
        for e in feed.entries:
            try:
                # published
                if hasattr(e, "published_parsed") and e.published_parsed:
                    pub_dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                elif hasattr(e, "updated_parsed") and e.updated_parsed:
                    pub_dt = datetime(*e.updated_parsed[:6], tzinfo=timezone.utc)
                else:
                    pub_dt = now_utc()

                if pub_dt < since_utc:
                    continue

                link = e.link
                original_url = parse_google_news_original(link)
                if not original_url:
                    continue

                # filter obvious junk
                host = hostname_from_url(original_url)
                if host.startswith("lh") and host.endswith(".googleusercontent.com"):
                    continue

                title = (getattr(e, "title", "") or "").strip()
                publisher = ""
                if hasattr(e, "source") and hasattr(e.source, "title"):
                    publisher = (e.source.title or "").strip()
                if not publisher:
                    publisher = host or "Unknown"

                art_hash = sha256_hex(original_url)

                with conn.cursor() as cur:
                    cur.execute(
                        UPSERT_ARTICLE_SQL,
                        (original_url, link, title, publisher, pub_dt, art_hash),
                    )
                    row = cur.fetchone()
                    if row:
                        article_id = row[0]
                        cur.execute(UPSERT_ARTICLE_TAG_SQL, (article_id, equity_id))
                        inserted += 1
            except Exception as ex:
                logger.warning(f"RSS entry skipped: {ex}")
                continue
    return inserted

# -----------------------------------------------------------------------------
# Fetch & extract article text
# -----------------------------------------------------------------------------
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) QuantBriefBot/1.0"

def fetch_page(url: str) -> str:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT)
        if r.status_code >= 200 and r.status_code < 400:
            return r.text
    except Exception:
        pass
    return ""

def extract_text(html: str) -> str:
    if not html:
        return ""
    if trafilatura is None:
        # fallback: very naive strip of tags
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.S|re.I)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S|re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    try:
        txt = trafilatura.extract(html, include_comments=False, include_tables=False)
        if txt:
            return txt.strip()
    except Exception:
        pass
    # fallback again
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S|re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S|re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# -----------------------------------------------------------------------------
# OpenAI summarization
# -----------------------------------------------------------------------------
def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=OPENAI_API_KEY)

SUMMARY_SYSTEM = (
    "You are a terse, factual buy-side analyst. Summarize only what's material to investors. "
    "One line max. Strict format: '• What matters — <insight>. • Stance — Positive|Neutral|Negative'. "
    "If there is no substantive article text, reply exactly with 'Skip'."
)

def summarize_one(title: str, publisher: str, text: str) -> tuple[str, str]:
    """
    Returns (what_matters_line, stance). If not enough text, returns ('', '').
    """
    if not text or len(text) < 300:
        return ("", "")

    client = get_openai_client()
    prompt = f"Title: {title}\nPublisher: {publisher}\n\nArticle:\n{text[:8000]}"
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )
        out = resp.choices[0].message.content.strip()
        if out == "Skip":
            return ("", "")
        # Try to extract stance at the end: '• Stance — Neutral'
        stance = ""
        m = re.search(r"Stance\s*—\s*(Positive|Neutral|Negative)", out, flags=re.I)
        if m:
            stance = m.group(1).title()
        return (out, stance)
    except Exception as ex:
        logger.warning(f"OpenAI summarize error: {ex}")
        return ("", "")

# -----------------------------------------------------------------------------
# Email (Jinja2)
# -----------------------------------------------------------------------------
EMAIL_TMPL = Template(textwrap.dedent("""
<!doctype html>
<html>
  <body style="font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; line-height:1.45; color:#111;">
    <h2 style="margin:0 0 8px 0;">{{ app_name }} — {{ date_str }}</h2>
    <div style="font-size:13px; color:#666; margin-bottom:16px;">
      Window: last {{ window_hours }}h ending {{ end_local }} {{ tz_name }}
    </div>

    {% for block in blocks %}
      <h3 style="margin:16px 0 6px 0;">Company — {{ block.company }} ({{ block.ticker }})</h3>
      {% if block.items %}
        {% for it in block.items %}
          <div style="margin:10px 0;">
            <div style="font-weight:600">{{ it.domain }}</div>
            <div style="font-size:12px; color:#666;">{{ it.ts_local }}</div>
            <div style="margin-top:6px;">{{ it.line | e }}</div>
            <div style="font-size:12px; margin-top:6px;">
              <a href="{{ it.url }}" target="_blank" style="color:#0a58ca; text-decoration:none;">Original</a>
            </div>
          </div>
        {% endfor %}
      {% else %}
        <div style="color:#666; font-style:italic;">No items after filters.</div>
      {% endif %}
      <hr style="border:none; border-top:1px solid #e5e5e5; margin:16px 0;">
    {% endfor %}

    <div style="font-size:12px; color:#777;">
      Sources are links only. We don’t republish paywalled content. Filters editable by owner.
    </div>
  </body>
</html>
""").strip())

def send_mailgun(subject: str, html: str, to_list: list[tuple[str,str]]):
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN):
        logger.warning("Mailgun not configured; skipping email send.")
        return {"ok": False, "reason": "mailgun-not-configured"}

    url = f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages"
    results = []
    for (email, name) in to_list:
        data = {
            "from": MAILGUN_FROM,
            "to": f"{name} <{email}>" if name else email,
            "subject": subject,
            "html": html,
        }
        try:
            r = requests.post(url, auth=("api", MAILGUN_API_KEY), data=data, timeout=HTTP_TIMEOUT)
            ok = 200 <= r.status_code < 300
            if not ok:
                logger.warning(f"Mailgun error {r.status_code}: {r.text}")
            results.append({"email": email, "ok": ok})
        except Exception as ex:
            results.append({"email": email, "ok": False, "err": str(ex)})
    return {"ok": all(x.get("ok") for x in results), "results": results}

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK"

@app.get("/admin/health", response_class=PlainTextResponse)
def admin_health(request: Request):
    require_admin(request)
    env_flags = {
        "DATABASE_URL": bool(DATABASE_URL),
        "OPENAI_API_KEY": bool(OPENAI_API_KEY),
        "OPENAI_MODEL": bool(OPENAI_MODEL),
        "OPENAI_MAX_OUTPUT_TOKENS": bool(OPENAI_MAX_OUTPUT_TOKENS),
        "MAILGUN_API_KEY": bool(MAILGUN_API_KEY),
        "MAILGUN_DOMAIN": bool(MAILGUN_DOMAIN),
        "MAILGUN_FROM": bool(MAILGUN_FROM),
        "OWNER_EMAIL": bool(OWNER_EMAIL),
        "ADMIN_TOKEN": bool(ADMIN_TOKEN),
    }

    with get_db() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM public.article;")
        n_articles = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM public.article_nlp;")
        n_summ = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM public.recipients;")
        n_rcpt = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM public.watchlist;")
        n_watch = cur.fetchone()[0]

    text = "env\n---\n" + json.dumps(env_flags, indent=2)
    text += f"\n\ncounts\n------\narticles={n_articles}\nsummarized={n_summ}\nrecipients={n_rcpt}\nwatchlist={n_watch}\n"
    return text

@app.get("/admin/test_openai")
def admin_test_openai(request: Request):
    require_admin(request)
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a terse assistant."},
                {"role": "user", "content": "Say OK."},
            ],
            temperature=0,
            max_output_tokens=max(OPENAI_MAX_OUTPUT_TOKENS, 50),
        )
        msg = resp.choices[0].message.content.strip()
        return {"ok": True, "model": OPENAI_MODEL, "sample": "OK" if "OK" in msg.upper() else msg}
    except Exception as ex:
        return {"ok": False, "model": OPENAI_MODEL, "err": str(ex)}

@app.post("/admin/init")
def admin_init(request: Request):
    require_admin(request)
    with get_db() as conn, conn.cursor() as cur:
        # create schema objects
        cur.execute(SCHEMA_SQL)

        # seed equity & watchlist
        cur.execute(UPSERT_EQUITY_SQL, (DEFAULT_COMPANY_NAME, DEFAULT_TICKER, DEFAULT_SECTOR))
        eq_id = cur.fetchone()[0]
        cur.execute(INSERT_WATCHLIST_IF_NOT_EXISTS_SQL, (eq_id, eq_id))

        # seed recipient
        cur.execute(UPSERT_RECIPIENT_SQL, (OWNER_EMAIL, "Owner"))

        # quick cleanup (noisy domains/images)
        cur.execute(DELETE_IMAGE_CDN_SQL)
        cur.execute(DELETE_IMAGE_EXT_SQL)
        cur.execute(DEDUP_URL_SQL)

    return {"ok": True, "equity": {"id": eq_id, "name": DEFAULT_COMPANY_NAME, "ticker": DEFAULT_TICKER}}

@app.post("/cron/ingest")
def cron_ingest(request: Request, minutes: int = 1440):
    require_admin(request)
    since_utc = now_utc() - timedelta(minutes=minutes)
    inserted_total = 0

    with get_db() as conn, conn.cursor() as cur:
        cur.execute(SELECT_WATCHLIST_SQL)
        rows = cur.fetchall()
        for (eq_id, name, ticker) in rows:
            inserted = ingest_google_news(conn, eq_id, name, ticker, since_utc)
            inserted_total += inserted

        # cleanup after ingest
        cur.execute(DELETE_IMAGE_CDN_SQL)
        cur.execute(DELETE_IMAGE_EXT_SQL)
        cur.execute(DEDUP_URL_SQL)

    return {"ok": True, "inserted": inserted_total, "since_utc": since_utc.isoformat()}

@app.post("/cron/summarize")
def cron_summarize(request: Request, minutes: int = 1440, limit_per_equity: int = 30):
    require_admin(request)
    since_utc = now_utc() - timedelta(minutes=minutes)
    summarized = 0

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(SELECT_WATCHLIST_SQL)
            watch = cur.fetchall()

        for (eq_id, name, ticker) in watch:
            with conn.cursor() as cur:
                cur.execute(SELECT_UNSUMMARIZED_SQL, (eq_id, since_utc, limit_per_equity))
                rows = cur.fetchall()

            for (article_id, url, title, publisher, pub_at, clean_text, domain) in rows:
                raw_html = ""
                if not clean_text or len(clean_text) < 300:
                    raw_html = fetch_page(url)
                    clean_text = extract_text(raw_html)

                if not clean_text or len(clean_text) < 300:
                    # Not enough substance; skip NLP row
                    with conn.cursor() as cur:
                        cur.execute(UPDATE_ARTICLE_HTML_TEXT_SQL, (raw_html or None, clean_text or None, article_id))
                    continue

                wm_line, stance = summarize_one(title or "", publisher or domain or "", clean_text)
                if wm_line:
                    with conn.cursor() as cur:
                        cur.execute(UPDATE_ARTICLE_HTML_TEXT_SQL, (raw_html or None, clean_text or None, article_id))
                        cur.execute(UPSERT_ARTICLE_NLP_SQL, (article_id, wm_line, stance or None))
                    summarized += 1

    return {"ok": True, "summarized": summarized}

@app.post("/cron/digest")
def cron_digest(request: Request, minutes: int = 1440):
    require_admin(request)
    window_hours = max(1, minutes // 60)
    since_utc = now_utc() - timedelta(minutes=minutes)
    end_local = to_tz(now_utc(), TIMEZONE).strftime("%H:%M")

    blocks = []
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(SELECT_WATCHLIST_SQL)
        watch = cur.fetchall()

        for (eq_id, name, ticker) in watch:
            cur.execute(SELECT_FOR_DIGEST_SQL, (eq_id, since_utc))
            rows = cur.fetchall()
            items = []
            for (aid, url, title, publisher, pub_at, domain, wm_line, stance) in rows:
                if not wm_line:
                    continue
                ts_local = to_tz(pub_at, TIMEZONE).strftime("%Y-%m-%d %H:%M")
                items.append({
                    "id": aid,
                    "url": url,
                    "title": title or "",
                    "publisher": publisher or (domain or ""),
                    "domain": domain or (publisher or "source"),
                    "ts_local": ts_local,
                    "line": wm_line,
                    "stance": stance or "",
                })
            blocks.append({"company": name, "ticker": ticker, "items": items})

        # recipients
        cur.execute(SELECT_RECIPIENTS_SQL)
        recipients = cur.fetchall()

    # Render email
    today = to_tz(now_utc(), TIMEZONE).strftime("%Y-%m-%d")
    html = EMAIL_TMPL.render(
        app_name=APP_NAME,
        date_str=today,
        window_hours=window_hours,
        end_local=end_local,
        tz_name=TIMEZONE,
        blocks=blocks,
    )
    subject = f"{APP_NAME} — {today}"

    result = send_mailgun(subject, html, recipients)
    sent_to = [r[0] for r in recipients]
    return {"ok": result.get("ok", False), "sent_to": sent_to, "items": sum(len(b["items"]) for b in blocks)}

# Convenience: single endpoint to run ingest+summarize+digest
@app.post("/cron/run_all")
def cron_run_all(request: Request, minutes: int = 1440, limit_per_equity: int = 30):
    require_admin(request)
    r1 = cron_ingest(request, minutes=minutes)
    r2 = cron_summarize(request, minutes=minutes, limit_per_equity=limit_per_equity)
    r3 = cron_digest(request, minutes=minutes)
    return {"ingest": r1, "summarize": r2, "digest": r3}
