# app.py
# QuantBrief Daily — minimal FastAPI app with Google News ingest, OpenAI summaries,
# Mailgun emailing, and hard blocklist for noisy sources.
#
# Designed for:
#   - OPENAI Python SDK == 1.51.0 (see requirements.txt)
#   - httpx == 0.27.2  (to avoid 'proxies' kwarg errors inside OpenAI client)
#   - FastAPI 0.115.x, Uvicorn 0.30.x
#
# Notes:
# - Keeps your requested OPENAI_MODEL=gpt-5-mini and avoids passing temperature/other unsupported args.
# - Uses OpenAI Responses API first (with both max_completion_tokens and max_output_tokens fallbacks),
#   then falls back to Chat Completions if needed.
# - Resolves Google News links to the original publisher before inserting and applies a domain/publisher blocklist
#   at ingest-time and again before digest.
# - Contains idempotent "init" that creates/patches the tiny schema needed by this app, without
#   forcing destructive migrations.
# - If your DB already has the tables, this only adds any missing columns/indexes.
# - Admin endpoints require the "x-admin-token" header matching ADMIN_TOKEN.

from __future__ import annotations

import os
import re
import sys
import hmac
import json
import time
import hashlib
import logging
import datetime as dt
from typing import Optional, Tuple, List
from urllib.parse import urlparse, parse_qs, urljoin, urlsplit

import requests
import feedparser
from bs4 import BeautifulSoup

import psycopg
from psycopg.rows import dict_row

from fastapi import FastAPI, Request, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from openai import OpenAI

# -----------------------------------------------------------------------------
# Config & Globals
# -----------------------------------------------------------------------------

IS_RENDER = bool(os.environ.get("RENDER"))
PORT = int(os.environ.get("PORT", "8000"))

DATABASE_URL = os.environ.get("DATABASE_URL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

# Keep this set small; app now handles gpt-5-mini quirks.
OPENAI_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "120"))

MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = os.environ.get("MAILGUN_DOMAIN", "")
MAILGUN_FROM = os.environ.get("MAILGUN_FROM", "QuantBrief Daily <daily@mg.example.com>")

OWNER_EMAIL = os.environ.get("OWNER_EMAIL")  # optional; we'll still run without it.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

# These are enforced both at ingest and at digest (belt + suspenders).
BLOCKED_DOMAINS = {
    "marketbeat.com",
    "www.marketbeat.com",
    "newser.com",
    "www.newser.com",
}
BLOCKED_PUBLISHERS = {"marketbeat", "newser"}

# Default watchlist if DB empty (your TLN seed)
DEFAULT_WATCHLIST = [("TLN", "Talen Energy")]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
log = logging.getLogger("quantbrief")

# OpenAI client (DO NOT pass any 'proxies' argument; we pin httpx==0.27.2 in requirements)
if not OPENAI_API_KEY:
    log.warning("OPENAI_API_KEY is not set — OpenAI features will fail.")
_openai = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="QuantBrief Daily")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def require_admin(req: Request):
    token = req.headers.get("x-admin-token")
    if not token or not hmac.compare_digest(token, ADMIN_TOKEN or ""):
        raise HTTPException(status_code=403, detail="Forbidden")
    return True


def utcnow() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)


def domain_of(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    try:
        d = urlparse(u).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return None


def is_blocked(url: Optional[str], publisher: Optional[str]) -> bool:
    d = domain_of(url)
    if d and d in BLOCKED_DOMAINS:
        return True
    if publisher and publisher.strip().lower() in BLOCKED_PUBLISHERS:
        return True
    return False


def http_get(url: str, timeout: int = 12, allow_redirects: bool = True) -> requests.Response:
    # Simple requests wrapper with conservative timeouts.
    return requests.get(url, timeout=timeout, allow_redirects=allow_redirects, headers={
        "User-Agent": "quantbrief/1.0 (https://github.com/iverpak/quantbrief-daily)"
    })


def resolve_google_news_url(gn_url: str) -> Tuple[str, Optional[str]]:
    """
    Resolve a Google News RSS item URL to its original publisher URL.

    Strategy:
      1) Try extracting ?url=<encoded> if present.
      2) Otherwise, follow redirects — often final resp.url is the article.
      3) Try to infer publisher name from og:site_name if we fetch the page.

    Returns: (resolved_url, publisher_guess)
    """
    publisher_guess = None
    try:
        # Attempt 1: extract ?url=
        qs = parse_qs(urlsplit(gn_url).query or "")
        if "url" in qs and qs["url"]:
            candidate = qs["url"][0]
            # Normalize and return
            return candidate, None

        # Attempt 2: follow redirects
        r = http_get(gn_url, timeout=12, allow_redirects=True)
        final = r.url
        # Attempt light publisher guess from HTML og:site_name if we got HTML
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            tag = soup.find("meta", attrs={"property": "og:site_name"})
            if tag and tag.get("content"):
                publisher_guess = tag["content"].strip()
        except Exception:
            pass
        return final, publisher_guess
    except Exception as e:
        log.warning("Failed to resolve Google News URL %s: %s", gn_url, e)
        return gn_url, None


def html_to_text(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Fetch URL and extract:
      - publisher name (og:site_name fallback to domain)
      - title (og:title <title> fallback)
      - main text (very light extraction since we no longer rely on trafilatura here)
    We intentionally keep dependencies light and robust.
    """
    try:
        r = http_get(url, timeout=12, allow_redirects=True)
        html = r.text
        soup = BeautifulSoup(html, "html.parser")

        # Title
        title = None
        t1 = soup.find("meta", attrs={"property": "og:title"})
        if t1 and t1.get("content"):
            title = t1["content"].strip()
        if not title and soup.title and soup.title.text:
            title = soup.title.text.strip()

        # Publisher
        publisher = None
        p1 = soup.find("meta", attrs={"property": "og:site_name"})
        if p1 and p1.get("content"):
            publisher = p1["content"].strip()
        if not publisher:
            publisher = domain_of(url)

        # Main text (very simple, avoid noisy scripts/styles/nav)
        # Grab all <p> text as a rough body. This avoids heavier extractors.
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        body = "\n".join([x for x in paragraphs if x]) if paragraphs else None

        return publisher, title, body
    except Exception as e:
        log.warning("html_to_text failed for %s: %s", url, e)
        return None, None, None


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def summarize_text_or_title(text: Optional[str], title: Optional[str]) -> str:
    """
    Summarize with OpenAI. We avoid passing temperature/max_tokens that gpt-5-mini
    dislikes; we try Responses first with 'max_completion_tokens', then
    'max_output_tokens', then fallback to chat.completions.
    """
    if not OPENAI_API_KEY:
        return (title or "No title").strip()

    # Compose a short prompt
    core = (text or "").strip()
    if not core:
        core = (title or "").strip()
    if not core:
        core = "Summarize in 1 short bullet."

    prompt = (
        "Summarize the news item below in one crisp bullet (<= 30 words). "
        "No emojis, no tickers unless already in the text. Keep it factual.\n\n"
        f"TEXT:\n{core[:4000]}"
    )

    # Try Responses with 'max_completion_tokens'
    try:
        r = _openai.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_completion_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )
        out = (getattr(r, "output_text", None) or "").strip()
        if out:
            return out
    except Exception as e1:
        log.info("Responses (max_completion_tokens) failed: %s", e1)

    # Try Responses with 'max_output_tokens'
    try:
        r = _openai.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )
        out = (getattr(r, "output_text", None) or "").strip()
        if out:
            return out
    except Exception as e2:
        log.info("Responses (max_output_tokens) failed: %s", e2)

    # Fallback: Chat Completions (uses 'max_tokens')
    try:
        r = _openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise financial news summarizer."},
                {"role": "user", "content": prompt},
            ],
            # Do NOT pass temperature for gpt-5-mini (it threw errors before).
            max_tokens=min(OPENAI_MAX_OUTPUT_TOKENS, 256),
        )
        msg = r.choices[0].message.content.strip()
        return msg
    except Exception as e3:
        log.warning("All OpenAI summarization attempts failed: %s", e3)
        # fall back to a truncated title
        return (title or core)[:140].strip() or "Summary unavailable."


# -----------------------------------------------------------------------------
# Database helpers
# -----------------------------------------------------------------------------

def db() -> psycopg.Connection:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)
    return conn


def sql(conn: psycopg.Connection, stmt: str, params: Optional[dict | tuple] = None):
    try:
        return conn.execute(stmt, params or {})
    except Exception as e:
        log.debug("SQL failed: %s\n%s", e, stmt)
        raise


def safe_alter(conn: psycopg.Connection, stmt: str):
    try:
        conn.execute(stmt)
    except Exception as e:
        # swallow DDL errors (column exists, etc.)
        log.info("Schema patch skipped: %s", e)


def ensure_schema(conn: psycopg.Connection):
    # Required core tables
    conn.execute("""
    CREATE TABLE IF NOT EXISTS public.article (
        id            BIGSERIAL PRIMARY KEY,
        url           TEXT NOT NULL,
        canonical_url TEXT,
        title         TEXT,
        publisher     TEXT,
        published_at  TIMESTAMPTZ,
        sha256        CHAR(64) NOT NULL UNIQUE,
        raw_html      TEXT,
        clean_text    TEXT,
        created_at    TIMESTAMPTZ DEFAULT now()
    );
    """)
    # domain (optional helper)
    safe_alter(conn, """
    ALTER TABLE public.article
      ADD COLUMN IF NOT EXISTS domain TEXT;
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS public.article_nlp (
        article_id  BIGINT PRIMARY KEY REFERENCES public.article(id) ON DELETE CASCADE,
        summary     TEXT
    );
    """)
    # Optional 'model' column (won't be required)
    safe_alter(conn, "ALTER TABLE public.article_nlp ADD COLUMN IF NOT EXISTS model TEXT;")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS public.recipients (
        id         BIGSERIAL PRIMARY KEY,
        email      TEXT UNIQUE NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now()
    );
    """)
    safe_alter(conn, "ALTER TABLE public.recipients ADD COLUMN IF NOT EXISTS enabled BOOLEAN NOT NULL DEFAULT TRUE;")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS public.watchlist (
        id       BIGSERIAL PRIMARY KEY,
        symbol   TEXT,
        company  TEXT,
        enabled  BOOLEAN NOT NULL DEFAULT TRUE
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS public.delivery_log (
        id             BIGSERIAL PRIMARY KEY,
        run_date       DATE NOT NULL,
        run_started_at TIMESTAMPTZ,
        run_ended_at   TIMESTAMPTZ,
        created_at     TIMESTAMPTZ DEFAULT now(),
        sent_to        TEXT,
        items          INTEGER DEFAULT 0,
        summarized     INTEGER DEFAULT 0
    );
    """)

    # Optional mapping table if you want tags (best-effort; not required for core flow)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS public.article_tag (
        article_id   BIGINT REFERENCES public.article(id) ON DELETE CASCADE,
        equity_id    BIGINT,
        watchlist_id BIGINT,
        tag_kind     TEXT NOT NULL DEFAULT 'watchlist'
    );
    """)
    # Not enforcing PK or NOT NULLs here to keep compatibility.

    # Helpful indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_article_published ON public.article (published_at DESC);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_article_domain ON public.article (domain);")


def seed_defaults(conn: psycopg.Connection):
    # Seed watchlist if empty
    cur = conn.execute("SELECT COUNT(*) AS c FROM public.watchlist;")
    c = cur.fetchone()["c"]
    if c == 0:
        conn.executemany(
            "INSERT INTO public.watchlist (symbol, company, enabled) VALUES (%s, %s, TRUE) ON CONFLICT DO NOTHING;",
            DEFAULT_WATCHLIST,
        )

    # Seed recipients if empty
    cur = conn.execute("SELECT COUNT(*) AS c FROM public.recipients;")
    c = cur.fetchone()["c"]
    if c == 0:
        seed = OWNER_EMAIL or "you@example.com"
        conn.execute(
            "INSERT INTO public.recipients (email, enabled) VALUES (%s, TRUE) ON CONFLICT (email) DO NOTHING;",
            (seed,),
        )


def update_article_domain(conn: psycopg.Connection, article_id: int, url: str, canonical_url: Optional[str]):
    try:
        d = domain_of(canonical_url or url)
        conn.execute("UPDATE public.article SET domain = %s WHERE id = %s;", (d, article_id))
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Ingest: Google News (watchlist-driven)
# -----------------------------------------------------------------------------

GOOGLE_NEWS_RSS_TMPL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

def _make_queries(conn: psycopg.Connection) -> List[Tuple[int, str]]:
    """
    Build Google News query strings from watchlist.
    Returns list of (watchlist_id, query).
    """
    rows = conn.execute(
        "SELECT id, COALESCE(NULLIF(TRIM(symbol),''), NULL) AS sym, COALESCE(NULLIF(TRIM(company),''), NULL) AS co "
        "FROM public.watchlist WHERE enabled = TRUE;"
    ).fetchall()
    out = []
    for r in rows:
        parts = []
        if r["co"]:
            parts.append(f"({r['co']})")
        if r["sym"]:
            parts.append(r["sym"])
        if not parts:
            continue
        q = " OR ".join(parts)
        out.append((r["id"], q))
    return out


def ingest_google_news(conn: psycopg.Connection, minutes: int = 1440) -> Tuple[int, int]:
    """
    Pull Google News for each watchlist query and insert normalized articles,
    skipping blocked domains/publishers. Returns (found, inserted).
    """
    found = 0
    inserted = 0
    cutoff = utcnow() - dt.timedelta(minutes=minutes)

    for wl_id, q in _make_queries(conn):
        rss = GOOGLE_NEWS_RSS_TMPL.format(query=requests.utils.quote(q, safe=":+()"))
        log.info("Fetching RSS for watchlist %s: %s", wl_id, rss)
        feed = feedparser.parse(rss)

        for e in feed.entries:
            found += 1
            link = e.get("link")
            if not link:
                continue

            resolved_url, pub_guess = resolve_google_news_url(link)
            if is_blocked(resolved_url, pub_guess):
                continue

            # Published time if present
            published_at = None
            if hasattr(e, "published_parsed") and e.published_parsed:
                published_at = dt.datetime.fromtimestamp(time.mktime(e.published_parsed), tz=dt.timezone.utc)
            if not published_at:
                published_at = utcnow()

            # Pull basic HTML info
            publisher, title, body = html_to_text(resolved_url)
            if is_blocked(resolved_url, publisher):
                continue

            # Skip if older than cutoff (after resolution)
            if published_at < cutoff:
                continue

            # Insert
            sh = sha256_hex(resolved_url)
            try:
                row = conn.execute(
                    """
                    INSERT INTO public.article (url, canonical_url, title, publisher, published_at, sha256, clean_text)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (sha256) DO NOTHING
                    RETURNING id;
                    """,
                    (resolved_url, resolved_url, title, publisher, published_at, sh, body),
                ).fetchone()
            except Exception as e:
                log.warning("Insert article failed: %s", e)
                row = None

            if row and row.get("id"):
                inserted += 1
                aid = row["id"]
                update_article_domain(conn, aid, resolved_url, resolved_url)

                # Best-effort tag link
                try:
                    conn.execute(
                        "INSERT INTO public.article_tag (article_id, watchlist_id, tag_kind) VALUES (%s, %s, 'watchlist');",
                        (aid, wl_id),
                    )
                except Exception:
                    pass

    return found, inserted


# -----------------------------------------------------------------------------
# Digest & Email
# -----------------------------------------------------------------------------

def _select_recent_articles(conn: psycopg.Connection, minutes: int = 1440, limit: int = 100):
    cutoff = utcnow() - dt.timedelta(minutes=minutes)
    rows = conn.execute(
        """
        SELECT id, url, canonical_url, title, publisher, published_at, domain
        FROM public.article
        WHERE published_at >= %s
        ORDER BY published_at DESC
        LIMIT %s;
        """,
        (cutoff, limit),
    ).fetchall()

    # Apply final blocklist
    kept = []
    for r in rows:
        d = r.get("domain") or domain_of(r.get("canonical_url") or r.get("url"))
        pub = (r.get("publisher") or "").strip().lower()
        if d in BLOCKED_DOMAINS:
            continue
        if pub in BLOCKED_PUBLISHERS:
            continue
        kept.append(r)
    return kept


def _ensure_summaries(conn: psycopg.Connection, ids_and_urls: List[dict]) -> int:
    """
    For each article, ensure a short summary exists in article_nlp.
    Returns how many we summarized in this run.
    """
    summarized_now = 0
    for r in ids_and_urls:
        aid = r["id"]

        exists = conn.execute("SELECT 1 FROM public.article_nlp WHERE article_id = %s;", (aid,)).fetchone()
        if exists:
            continue

        # Load content for summary
        art = conn.execute(
            "SELECT title, clean_text, url, canonical_url FROM public.article WHERE id = %s;", (aid,)
        ).fetchone()
        if not art:
            continue

        s = summarize_text_or_title(art.get("clean_text"), art.get("title"))
        try:
            conn.execute(
                "INSERT INTO public.article_nlp (article_id, summary, model) VALUES (%s, %s, %s) "
                "ON CONFLICT (article_id) DO UPDATE SET summary = EXCLUDED.summary;",
                (aid, s, OPENAI_MODEL),
            )
        except Exception:
            # Best-effort without model column
            try:
                conn.execute(
                    "INSERT INTO public.article_nlp (article_id, summary) VALUES (%s, %s) "
                    "ON CONFLICT (article_id) DO UPDATE SET summary = EXCLUDED.summary;",
                    (aid, s),
                )
            except Exception as e2:
                log.warning("Failed to upsert article_nlp for %s: %s", aid, e2)
                continue

        summarized_now += 1

    return summarized_now


def _compose_email(conn: psycopg.Connection, minutes: int, rows: List[dict]) -> Tuple[str, str]:
    end_ts = utcnow().replace(microsecond=0).isoformat()
    subject = f"QuantBrief Daily — {end_ts}"
    header = f"Window: last {minutes} minutes ending {end_ts}"
    lines = [header]

    # Attach summaries
    idset = tuple([r["id"] for r in rows]) if rows else tuple()
    summaries = {}
    if idset:
        for s in conn.execute(
            "SELECT article_id, summary FROM public.article_nlp WHERE article_id = ANY(%s);", (list(idset),)
        ):
            summaries[s["article_id"]] = s["summary"]

    for r in rows:
        title = r.get("title") or "(no title)"
        pub = r.get("publisher") or (r.get("domain") or "")
        url = r.get("canonical_url") or r.get("url")

        # Display line
        display = f"{title} — {pub}"
        lines.append(display)

        # Summary bullet
        s = summaries.get(r["id"])
        if s:
            lines.append(f"- {s}")

    if not rows:
        lines.append("\n(no eligible articles found)")

    lines.append("\nSources are links only. We don’t republish paywalled content.")
    body = "\n".join(lines)
    return subject, body


def _send_mailgun(to_emails: List[str], subject: str, text_body: str) -> Tuple[bool, str]:
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN and MAILGUN_FROM):
        return False, "Mailgun env not configured"

    try:
        resp = requests.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={
                "from": MAILGUN_FROM,
                "to": to_emails,
                "subject": subject,
                "text": text_body,
            },
            timeout=15,
        )
        ok = resp.status_code < 300
        return ok, (resp.text if not ok else "OK")
    except Exception as e:
        return False, str(e)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/admin/health")
def admin_health(req: Request, _=Depends(require_admin)):
    try:
        with db() as conn:
            ensure_schema(conn)
            counts = {
                "articles": conn.execute("SELECT COUNT(*) AS c FROM public.article;").fetchone()["c"],
                "summarized": conn.execute("SELECT COUNT(*) AS c FROM public.article_nlp;").fetchone()["c"],
                "recipients": conn.execute("SELECT COUNT(*) AS c FROM public.recipients;").fetchone()["c"],
                "watchlist": conn.execute("SELECT COUNT(*) AS c FROM public.watchlist;").fetchone()["c"],
            }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"DB error: {e}"},
        )

    env = {
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
    return {"env": env, "counts": counts}


@app.get("/admin/test_openai")
def admin_test_openai(req: Request, _=Depends(require_admin)):
    try:
        txt = summarize_text_or_title("Markets steady as yields ease; oil up.", "Global markets mixed")
        return PlainTextResponse(f"{OPENAI_MODEL} OK\n- {txt}")
    except Exception as e:
        log.exception("OpenAI test failed")
        return JSONResponse(status_code=500, content={"ok": False, "model": OPENAI_MODEL, "err": str(e)})


@app.post("/admin/init")
def admin_init(req: Request, _=Depends(require_admin)):
    try:
        with db() as conn:
            ensure_schema(conn)
            seed_defaults(conn)
        return PlainTextResponse("Initialized.")
    except Exception as e:
        log.exception("Init failed")
        return JSONResponse(status_code=500, content={"error": f"Init error: {e}"})


@app.post("/cron/ingest")
def cron_ingest(
    req: Request,
    minutes: int = Query(1440, ge=30, le=43200),  # 30 minutes to 30 days cap
    _=Depends(require_admin),
):
    try:
        with db() as conn:
            ensure_schema(conn)
            found, inserted = ingest_google_news(conn, minutes=minutes)
        return {"ok": True, "found_urls": found, "inserted": inserted}
    except Exception as e:
        log.exception("Ingest failed")
        return JSONResponse(status_code=500, content={"error": f"Ingest error: {e}"})


@app.post("/cron/digest")
def cron_digest(req: Request, _=Depends(require_admin)):
    run_started = utcnow()
    try:
        with db() as conn:
            ensure_schema(conn)

            # Prepare recipients
            rcpts = [r["email"] for r in conn.execute("SELECT email FROM public.recipients WHERE enabled = TRUE;")]
            if not rcpts:
                rcpts = ["you@example.com"]

            # Recent articles and ensure summaries
            rows = _select_recent_articles(conn, minutes=1440, limit=200)
            summarized_now = _ensure_summaries(conn, rows)

            # Compose + send
            subject, body = _compose_email(conn, minutes=1440, rows=rows)
            ok, info = _send_mailgun(rcpts, subject, body)

            # Log the run
            run_ended = utcnow()
            try:
                conn.execute(
                    """
                    INSERT INTO public.delivery_log
                      (run_date, run_started_at, run_ended_at, sent_to, items, summarized)
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    (run_started.date(), run_started, run_ended, ",".join(rcpts), len(rows), summarized_now),
                )
            except Exception as e2:
                log.info("delivery_log insert skipped: %s", e2)

            if not ok:
                return JSONResponse(status_code=500, content={"error": f"Mailgun: {info}"})

            return {"ok": True, "sent_to": ",".join(rcpts), "items": len(rows), "summarized": summarized_now}
    except Exception as e:
        log.exception("Digest failed")
        return JSONResponse(status_code=500, content={"error": f"Digest error: {e}"})


# Root
@app.get("/")
def root():
    return PlainTextResponse("QuantBrief Daily — up.")


# For local debug
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
