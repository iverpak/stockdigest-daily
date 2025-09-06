# app.py
import os
import re
import hmac
import json
import math
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs, urljoin

import requests
import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import psycopg
from psycopg.rows import dict_row

import trafilatura
from trafilatura.settings import use_config

from openai import OpenAI

# -----------------------------------------------------------------------------
# Environment / Config
# -----------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
OWNER_EMAIL = os.getenv("OWNER_EMAIL")  # optional
MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY")
MAILGUN_DOMAIN = os.getenv("MAILGUN_DOMAIN")
MAILGUN_FROM = os.getenv("MAILGUN_FROM")  # e.g. "QuantBrief <postmaster@yourdomain>"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "160"))
# Comma-separated hard blocklist (applies after resolving Google News):
BLOCKED_SITES = {
    d.strip().lower()
    for d in os.getenv("BLOCKED_SITES", "").split(",")
    if d.strip()
}

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

# OpenAI client — do NOT pass 'proxies' (breaks with 1.51.0)
_openai = OpenAI(api_key=OPENAI_API_KEY)

# Requests session
_http = requests.Session()
_http.headers.update({"User-Agent": "QuantBrief/1.0 (+https://render.com)"})
_http.timeout = (10, 20)

# Trafilatura config: speed up & avoid overfetch
tconfig = use_config()
tconfig.set("DEFAULT", "MIN_EXTRACTED_SIZE", "200")
tconfig.set("DEFAULT", "EXTRACTION_TIMEOUT", "10")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantbrief")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _require_admin(req: Request):
    token = req.headers.get("x-admin-token", "")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")


def _db():
    return psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)


def _now_utc():
    return datetime.now(timezone.utc)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _is_blocked_domain(domain: str) -> bool:
    if not domain:
        return False
    d = domain.lower()
    # match exact or subdomain (e.g., www.marketbeat.com)
    return any(d == b or d.endswith("." + b) for b in BLOCKED_SITES)


def _news_when_from_minutes(minutes: int) -> str:
    days = max(1, math.ceil(minutes / 1440))
    return f"when:{days}d"


def _follow_redirect_once(url: str) -> str | None:
    try:
        # First try HEAD with redirects
        r = _http.head(url, allow_redirects=True)
        if r.url and r.url != url:
            return r.url
    except Exception:
        pass
    try:
        # Then GET (some GNews pages render a meta refresh)
        r = _http.get(url, allow_redirects=True)
        if r.url and r.url != url:
            return r.url
        # meta refresh fallback
        soup = BeautifulSoup(r.text, "html.parser")
        # canonical
        link = soup.find("link", rel="canonical")
        if link and link.get("href"):
            return link["href"]
        # meta refresh pattern
        meta = soup.find("meta", attrs={"http-equiv": re.compile(r"refresh", re.I)})
        if meta and meta.get("content"):
            m = re.search(r'url=(.+)$', meta["content"], flags=re.I)
            if m:
                return m.group(1).strip()
        # an <a> with rel=nofollow (common on gnews landing)
        a = soup.find("a", attrs={"rel": re.compile(r"nofollow", re.I)})
        if a and a.get("href", "").startswith("http"):
            return a["href"]
    except Exception:
        pass
    return None


def resolve_google_news_link(raw_link: str) -> str:
    """
    Heuristics to turn a Google News 'articles/...' link into the real publisher link.
    """
    try:
        u = urlparse(raw_link)
        if "news.google.com" not in u.netloc.lower():
            return raw_link

        # 1) URL param sometimes carries the destination
        qs = parse_qs(u.query)
        if "url" in qs and qs["url"]:
            return qs["url"][0]

        # 2) follow redirects / parse landing
        final = _follow_redirect_once(raw_link)
        if final and "news.google.com" not in _domain_of(final):
            return final

        # 3) give up to the original link (we will block news.google.com below)
        return raw_link
    except Exception:
        return raw_link


def fetch_and_extract(url: str) -> tuple[str | None, str | None]:
    """
    Returns (clean_text, raw_html)
    """
    try:
        html = trafilatura.fetch_url(url, config=tconfig)
        if not html:
            return None, None
        text = trafilatura.extract(html, include_comments=False, include_images=False, config=tconfig)
        return (text, html)
    except Exception:
        return (None, None)


def openai_summarize(text: str) -> str:
    """
    Summarize with Chat Completions (gpt-5-mini compatible).
    Avoids passing temperature; tries with max_tokens and falls back if needed.
    """
    prompt = (
        "Summarize the following article in one crisp bullet (<= 24 words). "
        "Keep it factual and neutral. If details are missing, avoid speculation.\n\n"
        f"Article:\n{text[:6000]}"
    )
    messages = [
        {"role": "system", "content": "You are a concise financial news analyst."},
        {"role": "user", "content": prompt},
    ]

    # try with max_tokens first
    try:
        resp = _openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        # If max_tokens not supported or any 4xx param issue, retry without it
        estr = str(e)
        if "max_tokens" in estr or "max_completion_tokens" in estr or "Unsupported" in estr:
            resp = _openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
            )
            return (resp.choices[0].message.content or "").strip()
        raise


# -----------------------------------------------------------------------------
# DB bootstrap / migrations (idempotent)
# -----------------------------------------------------------------------------

DDL = [
    # articles
    """
    CREATE TABLE IF NOT EXISTS public.article (
        id            BIGSERIAL PRIMARY KEY,
        url           TEXT NOT NULL,
        canonical_url TEXT,
        title         TEXT,
        publisher     TEXT,
        published_at  TIMESTAMPTZ,
        sha256        CHAR(64) UNIQUE NOT NULL,
        raw_html      TEXT,
        clean_text    TEXT,
        created_at    TIMESTAMPTZ DEFAULT now()
    );
    """,
    # domain (optional column)
    """ALTER TABLE public.article ADD COLUMN IF NOT EXISTS domain TEXT;""",
    """CREATE UNIQUE INDEX IF NOT EXISTS idx_article_url_unique ON public.article (url);""",
    """CREATE INDEX IF NOT EXISTS idx_article_published ON public.article (published_at DESC);""",
    """CREATE INDEX IF NOT EXISTS idx_article_domain ON public.article (domain);""",

    # summarization store
    """
    CREATE TABLE IF NOT EXISTS public.article_nlp (
        id         BIGSERIAL PRIMARY KEY,
        article_id BIGINT NOT NULL REFERENCES public.article(id) ON DELETE CASCADE,
        model      TEXT,
        summary    TEXT,
        created_at TIMESTAMPTZ DEFAULT now(),
        UNIQUE (article_id, model)
    );
    """,

    # recipients
    """
    CREATE TABLE IF NOT EXISTS public.recipients (
        id         BIGSERIAL PRIMARY KEY,
        email      TEXT UNIQUE NOT NULL,
        enabled    BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ DEFAULT now()
    );
    """,

    # watchlist (simple)
    """
    CREATE TABLE IF NOT EXISTS public.watchlist (
        id         BIGSERIAL PRIMARY KEY,
        symbol     TEXT UNIQUE NOT NULL,
        company    TEXT,
        enabled    BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ DEFAULT now()
    );
    """,

    # basic tag table (we'll keep it flexible)
    """
    CREATE TABLE IF NOT EXISTS public.article_tag (
        article_id  BIGINT NOT NULL REFERENCES public.article(id) ON DELETE CASCADE,
        equity_id   BIGINT,
        watchlist_id BIGINT,
        tag_kind    INT NOT NULL DEFAULT 0,
        PRIMARY KEY (article_id, tag_kind, COALESCE(equity_id, 0), COALESCE(watchlist_id, 0))
    );
    """,

    # source feeds (optional; supports pre-seeded RSS)
    """
    CREATE TABLE IF NOT EXISTS public.source_feed (
        id             BIGSERIAL PRIMARY KEY,
        kind           TEXT NOT NULL DEFAULT 'rss',
        url            TEXT UNIQUE NOT NULL,
        active         BOOLEAN NOT NULL DEFAULT TRUE,
        name           TEXT,
        period_minutes INTEGER NOT NULL DEFAULT 60,
        last_checked_at TIMESTAMPTZ NULL
    );
    """,

    # delivery log
    """
    CREATE TABLE IF NOT EXISTS public.delivery_log (
        id             BIGSERIAL PRIMARY KEY,
        run_date       DATE NOT NULL DEFAULT CURRENT_DATE,
        run_started_at TIMESTAMPTZ,
        run_ended_at   TIMESTAMPTZ,
        sent_to        TEXT,
        items          INT NOT NULL DEFAULT 0,
        summarized     INT NOT NULL DEFAULT 0,
        created_at     TIMESTAMPTZ DEFAULT now()
    );
    """,
]

SEED_SQL = [
    # seed one example watch (TLN) if empty
    """
    INSERT INTO public.watchlist (symbol, company)
    SELECT 'TLN', 'Talen Energy Corporation'
    WHERE NOT EXISTS (SELECT 1 FROM public.watchlist WHERE symbol = 'TLN');
    """,
    # seed a Google News feed row (with stricter query) if not present
    """
    INSERT INTO public.source_feed (kind, url, active, name, period_minutes)
    VALUES (
        'rss',
        'https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+-site:newser.com+-site:marketbeat.com+' || %(when)s || '&hl=en-US&gl=US&ceid=US:en',
        TRUE,
        'Google News: Talen Energy',
        60
    )
    ON CONFLICT (url) DO NOTHING;
    """,
    # seed recipients (owner optional + example placeholder)
    """
    INSERT INTO public.recipients (email) VALUES ('you@example.com')
    ON CONFLICT (email) DO NOTHING;
    """,
]


def init_db():
    with _db() as conn:
        for stmt in DDL:
            conn.execute(stmt)

        # dynamic WHEN:3d for seed URL (default to 3d so first fill has some data)
        when = "when:3d"
        for stmt in SEED_SQL:
            if "%(when)s" in stmt:
                conn.execute(stmt, {"when": when})
            else:
                conn.execute(stmt)

        if OWNER_EMAIL:
            conn.execute(
                "INSERT INTO public.recipients (email) VALUES (%s) ON CONFLICT (email) DO NOTHING;",
                (OWNER_EMAIL,),
            )


# -----------------------------------------------------------------------------
# Ingestion
# -----------------------------------------------------------------------------

def _collect_rss_urls(minutes: int) -> list[tuple[int, str, str]]:
    """
    Return list of (feed_id, name, url) to scan.
    Strategy:
      1) use active rows from source_feed
      2) if none exist, build Google News queries for enabled watchlist symbols/companies
    """
    feeds: list[tuple[int, str, str]] = []
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, name, url FROM public.source_feed WHERE active = TRUE ORDER BY id"
        ).fetchall()
        for r in rows:
            feeds.append((r["id"], r["name"] or "", r["url"]))

        if not feeds:
            # fallback: build feeds from watchlist
            when = _news_when_from_minutes(minutes)
            wl = conn.execute(
                "SELECT id, symbol, company FROM public.watchlist WHERE enabled = TRUE ORDER BY id"
            ).fetchall()
            for w in wl:
                q = requests.utils.quote(f"({w['company']}) OR {w['symbol']} -site:newser.com -site:marketbeat.com {when}")
                url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
                feeds.append((0, f"Google News: {w['company']}", url))
    return feeds


def _published_at_of(entry) -> datetime | None:
    dt = None
    for k in ("published", "updated"):
        if hasattr(entry, k) and getattr(entry, k):
            try:
                dt = dateparser.parse(getattr(entry, k))
                break
            except Exception:
                pass
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ingest_google_news(minutes: int) -> tuple[int, int]:
    """
    Fetch active feeds, resolve links, extract text, and upsert into article table.
    Returns (found_urls, inserted_rows)
    """
    feeds = _collect_rss_urls(minutes)
    found = 0
    inserted = 0

    with _db() as conn:
        for _, feed_name, feed_url in feeds:
            logger.info("Fetching feed: %s", feed_url)
            fd = feedparser.parse(feed_url)
            for entry in fd.entries:
                raw_link = entry.get("link") or ""
                if not raw_link:
                    continue

                found += 1

                # Resolve Google News link (to real publisher) if needed
                final_url = resolve_google_news_link(raw_link)
                domain = _domain_of(final_url)

                # Hard blocklist
                if _is_blocked_domain(domain) or domain.endswith("news.google.com"):
                    logger.info("Skipping blocked domain: %s (%s)", domain, final_url)
                    continue

                # Extract metadata
                title = (entry.get("title") or "").strip()
                publisher = (entry.get("source", {}) or {}).get("title") if isinstance(entry.get("source"), dict) else None
                if not publisher:
                    publisher = domain
                published_at = _published_at_of(entry)

                # Fetch article content
                clean_text, raw_html = fetch_and_extract(final_url)

                # Build deterministic hash
                sh = _sha256(final_url)

                # Insert
                try:
                    res = conn.execute(
                        """
                        INSERT INTO public.article
                            (url, canonical_url, title, publisher, published_at, sha256, raw_html, clean_text, domain)
                        VALUES
                            (%s, NULL, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (sha256) DO NOTHING
                        RETURNING id
                        """,
                        (final_url, title, publisher, published_at, sh, raw_html, clean_text, domain),
                    )
                    row = res.fetchone()
                    if row:
                        inserted += 1
                        art_id = row["id"]
                        # Try to tag with watchlist (best-effort; table shape can vary)
                        try:
                            wl = conn.execute(
                                "SELECT id FROM public.watchlist WHERE enabled = TRUE ORDER BY id"
                            ).fetchall()
                            if wl:
                                conn.execute(
                                    "INSERT INTO public.article_tag (article_id, watchlist_id, tag_kind) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                                    (art_id, wl[0]["id"], 1),
                                )
                        except Exception as te:
                            logger.warning("Tag insert skipped: %s", te)
                except Exception as e:
                    logger.exception("Insert failed for %s: %s", final_url, e)

    return (found, inserted)


# -----------------------------------------------------------------------------
# Digest
# -----------------------------------------------------------------------------

def _collect_recipients() -> list[str]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT email FROM public.recipients WHERE enabled = TRUE ORDER BY id"
        ).fetchall()
        return [r["email"] for r in rows]


def _articles_since(minutes: int) -> list[dict]:
    cutoff = _now_utc() - timedelta(minutes=minutes)
    with _db() as conn:
        arts = conn.execute(
            """
            SELECT a.id, a.title, a.url, a.publisher, a.published_at, a.clean_text
            FROM public.article a
            WHERE (a.published_at IS NULL OR a.published_at >= %s)
            ORDER BY COALESCE(a.published_at, a.created_at) DESC
            LIMIT 200
            """,
            (cutoff,),
        ).fetchall()
        return arts


def _ensure_summaries(arts: list[dict]) -> tuple[int, list[tuple[dict, str]]]:
    """
    Ensures each article has a summary row for this model.
    Returns (num_summarized_now, [(article, summary), ...])
    """
    out: list[tuple[dict, str]] = []
    wrote = 0
    with _db() as conn:
        for a in arts:
            # check if already exists
            row = conn.execute(
                "SELECT summary FROM public.article_nlp WHERE article_id = %s AND model = %s",
                (a["id"], OPENAI_MODEL),
            ).fetchone()
            if row and row["summary"]:
                out.append((a, row["summary"]))
                continue

            # Summarize
            text = a["clean_text"] or a["title"] or a["url"]
            if not text:
                continue
            try:
                summary = openai_summarize(text)
            except Exception as e:
                logger.warning("Summarize failed for article %s: %s", a["id"], e)
                continue

            try:
                conn.execute(
                    "INSERT INTO public.article_nlp (article_id, model, summary) VALUES (%s, %s, %s) ON CONFLICT (article_id, model) DO UPDATE SET summary = EXCLUDED.summary",
                    (a["id"], OPENAI_MODEL, summary),
                )
                wrote += 1
                out.append((a, summary))
            except Exception as e:
                logger.warning("Store summary failed for article %s: %s", a["id"], e)
    return wrote, out


def _render_digest(minutes: int, pairs: list[tuple[dict, str]]) -> str:
    lines = []
    now = _now_utc().strftime("%Y-%m-%d %H:%M")
    lines.append(f"QuantBrief Daily — {now}")
    lines.append(f"Window: last {minutes} minutes ending {now}")

    for a, s in pairs:
        publisher = a["publisher"] or _domain_of(a["url"])
        title = a["title"] or a["url"]
        lines.append(f"{title} — {publisher}")
        lines.append(f"{s}")
    if not pairs:
        lines.append("No qualifying articles found in this window.")
    lines.append("Sources are links only. We don’t republish paywalled content.")
    return "\n".join(lines)


def _send_mailgun(to_emails: list[str], subject: str, text: str) -> bool:
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN and MAILGUN_FROM):
        logger.warning("Mailgun not configured; skipping send.")
        return False
    try:
        resp = _http.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={
                "from": MAILGUN_FROM,
                "to": to_emails,
                "subject": subject,
                "text": text,
            },
            timeout=(10, 30),
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error("Mailgun send failed: %s", e)
        return False


# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------

app = FastAPI(title="QuantBrief Daily", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/admin/health", response_class=PlainTextResponse)
def admin_health(req: Request):
    _require_admin(req)
    env_state = {
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

    with _db() as conn:
        counts = {
            "articles": conn.execute("SELECT COUNT(*) FROM public.article").fetchone()["count"],
            "summarized": conn.execute("SELECT COUNT(*) FROM public.article_nlp").fetchone()["count"],
            "recipients": conn.execute("SELECT COUNT(*) FROM public.recipients").fetchone()["count"],
            "watchlist": conn.execute("SELECT COUNT(*) FROM public.watchlist").fetchone()["count"],
        }

    out = []
    out.append("env")
    out.append("---")
    out.append(json.dumps(env_state, indent=2))
    out.append("\ncounts")
    out.append("------")
    for k, v in counts.items():
        out.append(f"{k}={v}")
    return "\n".join(out)


@app.get("/admin/test_openai", response_class=PlainTextResponse)
def admin_test_openai(req: Request):
    _require_admin(req)
    try:
        resp = _openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Reply with a single short phrase indicating readiness."},
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        ok = True
        sample = txt[:120]
    except Exception as e:
        ok = False
        sample = f"{type(e).__name__}: {e}"

    lines = []
    lines.append("   ok model      sample")
    lines.append("   -- -----      ------")
    lines.append(f"{str(ok):5} {OPENAI_MODEL:10} {sample}")
    return "\n".join(lines)


@app.post("/admin/init", response_class=PlainTextResponse)
def admin_init(req: Request):
    _require_admin(req)
    try:
        init_db()
        return "Initialized.\n"
    except Exception as e:
        logger.exception("Init error")
        raise HTTPException(status_code=500, detail=f"Init error: {e}")


@app.post("/cron/ingest", response_class=PlainTextResponse)
def cron_ingest(req: Request, minutes: int = 1440):
    _require_admin(req)
    try:
        found, ins = ingest_google_news(minutes=minutes)
        lines = []
        lines.append("")
        lines.append("  ok found_urls inserted")
        lines.append("  -- ---------- --------")
        lines.append(f"True {found:10d} {ins:8d}")
        return "\n".join(lines)
    except Exception as e:
        logger.exception("Ingest error")
        raise HTTPException(status_code=500, detail=f"Ingest error: {e}")


@app.post("/cron/digest", response_class=PlainTextResponse)
def cron_digest(req: Request, minutes: int = 1440):
    _require_admin(req)
    start = _now_utc()
    recipients = _collect_recipients()
    items = 0
    summarized_now = 0
    try:
        arts = _articles_since(minutes)
        summarized_now, pairs = _ensure_summaries(arts)
        items = len(pairs)
        digest_text = _render_digest(minutes, pairs)
        subj = f"QuantBrief Daily — {start.strftime('%Y-%m-%d %H:%M')}"

        sent_ok = _send_mailgun(recipients, subj, digest_text) if recipients else False

        # Log run
        with _db() as conn:
            conn.execute(
                """
                INSERT INTO public.delivery_log (run_date, run_started_at, run_ended_at, sent_to, items, summarized)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (start.date(), start, _now_utc(), ",".join(recipients), items, summarized_now),
            )

        return digest_text
    except Exception as e:
        logger.exception("Digest error")
        # still attempt to log the failure window
        try:
            with _db() as conn:
                conn.execute(
                    """
                    INSERT INTO public.delivery_log (run_date, run_started_at, run_ended_at, sent_to, items, summarized)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (start.date(), start, _now_utc(), ",".join(recipients), items, summarized_now),
                )
        except Exception as _:
            pass
        raise HTTPException(status_code=500, detail=f"Digest error: {e}")


@app.get("/", response_class=PlainTextResponse)
def root():
    return "QuantBrief Daily is running.\n"
