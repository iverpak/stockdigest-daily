import os
import re
import json
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import requests
import feedparser
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from jinja2 import Template
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI

# ----------------------------
# Config & constants
# ----------------------------
APP_NAME = "QuantBrief Daily"

DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
OPENAI_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "240"))

MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY")
MAILGUN_DOMAIN = os.environ.get("MAILGUN_DOMAIN")
MAILGUN_FROM = os.environ.get("MAILGUN_FROM", f"daily@mg.example.com")

OWNER_EMAIL = os.environ.get("OWNER_EMAIL")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

# ban these sources everywhere, including subdomains
BANNED_PATTERNS = ("marketbeat.com", "newser.com")

# Optional: add any noisy CDNs, images, etc.
DISALLOW_EXT_REGEX = re.compile(r"\.(png|jpe?g|gif|webp|svg|ico)(\?.*)?$", re.I)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("quantbrief")

# OpenAI client (no proxies arg — keeps 1.51.0 happy)
_openai = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title=APP_NAME)


# ----------------------------
# DB helpers & schema
# ----------------------------
def db() -> psycopg.Connection:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL missing")
    return psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)


SCHEMA_SQL = """
-- articles
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

-- add a domain for quick filtering (computed)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='article' AND column_name='domain'
  ) THEN
    ALTER TABLE public.article
      ADD COLUMN domain TEXT GENERATED ALWAYS AS (
        lower(split_part(regexp_replace(coalesce(canonical_url, url), '^[a-z]+://', ''), '/', 1))
      ) STORED;
  END IF;
END$$;

-- indexes
CREATE INDEX IF NOT EXISTS idx_article_published ON public.article (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_article_domain    ON public.article (domain);
CREATE UNIQUE INDEX IF NOT EXISTS idx_article_url_unique ON public.article (url);

-- NLP per article
CREATE TABLE IF NOT EXISTS public.article_nlp (
  article_id BIGINT PRIMARY KEY REFERENCES public.article(id) ON DELETE CASCADE,
  model      TEXT,
  summary    TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- watchlist
CREATE TABLE IF NOT EXISTS public.watchlist (
  id       BIGSERIAL PRIMARY KEY,
  symbol   TEXT UNIQUE NOT NULL,
  company  TEXT,
  enabled  BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- recipients
CREATE TABLE IF NOT EXISTS public.recipients (
  id       BIGSERIAL PRIMARY KEY,
  email    TEXT UNIQUE NOT NULL,
  enabled  BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- topic tagging/links (make this flexible with a surrogate PK)
CREATE TABLE IF NOT EXISTS public.article_tag (
  id           BIGSERIAL PRIMARY KEY,
  article_id   BIGINT NOT NULL REFERENCES public.article(id) ON DELETE CASCADE,
  equity_id    BIGINT NULL REFERENCES public.equity(id) ON DELETE SET NULL,
  watchlist_id BIGINT NULL REFERENCES public.watchlist(id) ON DELETE CASCADE,
  tag_kind     TEXT NOT NULL DEFAULT 'watch',
  created_at   TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_article_tag_article ON public.article_tag (article_id);

-- delivery log (for digests)
CREATE TABLE IF NOT EXISTS public.delivery_log (
  id             BIGSERIAL PRIMARY KEY,
  run_date       DATE NOT NULL,
  run_started_at TIMESTAMPTZ,
  run_ended_at   TIMESTAMPTZ,
  sent_to        TEXT,
  items          INTEGER NOT NULL DEFAULT 0,
  summarized     INTEGER NOT NULL DEFAULT 0,
  created_at     TIMESTAMPTZ DEFAULT now()
);

-- list of feeds to poll
CREATE TABLE IF NOT EXISTS public.source_feed (
  id              BIGSERIAL PRIMARY KEY,
  kind            TEXT NOT NULL DEFAULT 'rss',
  url             TEXT UNIQUE NOT NULL,
  active          BOOLEAN NOT NULL DEFAULT TRUE,
  name            TEXT,
  period_minutes  INTEGER NOT NULL DEFAULT 60,
  last_checked_at TIMESTAMPTZ NULL,
  created_at      TIMESTAMPTZ DEFAULT now()
);
"""

SEED_SQL = """
-- seed TLN watchlist if empty
INSERT INTO public.watchlist (symbol, company) 
SELECT 'TLN', 'Talen Energy Corporation'
WHERE NOT EXISTS (SELECT 1 FROM public.watchlist);

-- seed recipients with OWNER_EMAIL if present
-- (safe to run repeatedly; unique constraint)
{owner_recipient}

-- seed a Google News RSS for TLN if none exists
INSERT INTO public.source_feed (kind, url, active, name, period_minutes)
SELECT 'rss',
       'https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN&hl=en-US&gl=US&ceid=US:en',
       TRUE,
       'Google News: Talen Energy',
       60
WHERE NOT EXISTS (
  SELECT 1 FROM public.source_feed WHERE url LIKE 'https://news.google.com/rss/search?q=(Talen+Energy)%'
);
"""


# ----------------------------
# Utility
# ----------------------------
def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hostname_of(u: str) -> str:
    try:
        return urlparse(u).hostname or ""
    except Exception:
        return ""


def strip_tracking(u: str) -> str:
    # Remove common trackers (very conservative)
    try:
        p = urlparse(u)
        if not p.scheme:
            return u
        base = f"{p.scheme}://{p.netloc}{p.path}"
        return base
    except Exception:
        return u


def is_banned_link(url: str) -> bool:
    host = hostname_of(url)
    return any(p in host for p in BANNED_PATTERNS)


def fetch_url(url: str) -> Optional[requests.Response]:
    try:
        if DISALLOW_EXT_REGEX.search(url):
            return None
        r = requests.get(url, timeout=12, headers={"User-Agent": USER_AGENT})
        if r.status_code >= 400:
            return None
        return r
    except Exception:
        return None


def extract_title_publisher(html: str, url: str) -> Tuple[Optional[str], Optional[str]]:
    """Try to get a clean title and publisher from the page."""
    try:
        soup = BeautifulSoup(html, "lxml")

        # <title>
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # OpenGraph/Meta
        og_t = soup.find("meta", property="og:title")
        if og_t and og_t.get("content"):
            title = og_t["content"].strip()

        # Publisher
        publisher = None
        og_site = soup.find("meta", property="og:site_name")
        if og_site and og_site.get("content"):
            publisher = og_site["content"].strip()

        if not publisher:
            # fallback: domain as publisher
            publisher = hostname_of(url).replace("www.", "")

        return (title, publisher)
    except Exception:
        return (None, hostname_of(url).replace("www.", ""))


def resolve_google_news_url(url: str) -> Optional[str]:
    """Return the publisher URL for a Google News entry (HTML wrapper) or None."""
    try:
        if "news.google.com" not in url:
            return url

        # Some GN links are redirect templates; resolve once
        r0 = requests.get(url, timeout=12, headers={"User-Agent": USER_AGENT}, allow_redirects=True)
        if r0.url and "news.google.com" not in hostname_of(r0.url):
            return r0.url

        r = r0 if r0 is not None else requests.get(url, timeout=12, headers={"User-Agent": USER_AGENT})
        if not r or r.status_code >= 400:
            return None

        soup = BeautifulSoup(r.text, "lxml")

        # 1) JSON-LD usually has canonical
        for s in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(s.string or "")
            except Exception:
                continue
            objs = data if isinstance(data, list) else [data]
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                cand = obj.get("url")
                if isinstance(cand, str) and "news.google.com" not in hostname_of(cand):
                    return cand
                meop = obj.get("mainEntityOfPage")
                if isinstance(meop, dict):
                    cand2 = meop.get("@id")
                    if isinstance(cand2, str) and "news.google.com" not in hostname_of(cand2):
                        return cand2

        # 2) First external anchor
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("/"):
                href = urljoin("https://news.google.com", href)
            if href.startswith("http") and "news.google.com" not in hostname_of(href):
                return href

        return None
    except Exception as e:
        log.warning("resolve_google_news_url error: %s", e)
        return None


# ----------------------------
# Summarization
# ----------------------------
def summarize_text(url: str, title: str, publisher: str, text: str) -> str:
    """Short, neutral, 1–2 sentence summary."""
    if not OPENAI_API_KEY:
        return ""
    # keep prompt tiny to avoid token pressure on mini models
    prompt = (
        "Summarize in 1–2 concise sentences, neutral tone, no hype, no emojis. "
        "Focus on the *new* information relevant to investors.\n\n"
        f"Title: {title}\n"
        f"Publisher: {publisher}\n"
        f"URL: {url}\n\n"
        f"Article:\n{text[:4000]}"
    )
    try:
        resp = _openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(OPENAI_MAX_OUTPUT_TOKENS, 240),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning("OpenAI summarize error: %s", e)
        return ""


# ----------------------------
# Email
# ----------------------------
EMAIL_HTML_TEMPLATE = Template(
    """
<h2 style="margin:0 0 12px 0;">{{ app_name }} — {{ when }}</h2>
<p style="margin:0 0 12px 0;color:#555;">
  Window: last {{ minutes }} minutes ending {{ window_end }}
</p>
<ul style="padding-left:20px;">
{% for it in items %}
  <li style="margin:10px 0;">
    <a href="{{ it.url | e }}" target="_blank" rel="noopener noreferrer">{{ it.title | e }}</a>
    {% if it.publisher %} — <em>{{ it.publisher | e }}</em>{% endif %}
    {% if it.summary %}
      <div style="margin-top:6px;color:#333;">{{ it.summary | e }}</div>
    {% endif %}
  </li>
{% endfor %}
</ul>
<p style="color:#888;">Sources are links only. We don’t republish paywalled content.</p>
"""
)

EMAIL_TEXT_TEMPLATE = Template(
    """{{ app_name }} — {{ when }}
Window: last {{ minutes }} minutes ending {{ window_end }}

{% for it in items -%}
- {{ it.title }} — {{ it.publisher or "" }}
  {{ it.url }}
  {{ it.summary or "" }}

{% endfor -%}
Sources are links only. We don’t republish paywalled content.
"""
)


def send_email(subject: str, items: List[Dict[str, Any]], minutes: int, recipients: List[str]):
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN and MAILGUN_FROM):
        log.info("Mailgun not configured; skipping send.")
        return

    now = datetime.now(timezone.utc)
    html = EMAIL_HTML_TEMPLATE.render(
        app_name=APP_NAME,
        when=now.strftime("%Y-%m-%d %H:%M"),
        minutes=minutes,
        window_end=now.isoformat(timespec="seconds"),
        items=items,
    )
    text = EMAIL_TEXT_TEMPLATE.render(
        app_name=APP_NAME,
        when=now.strftime("%Y-%m-%d %H:%M"),
        minutes=minutes,
        window_end=now.isoformat(timespec="seconds"),
        items=items,
    )

    r = requests.post(
        f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
        auth=("api", MAILGUN_API_KEY),
        data={
            "from": MAILGUN_FROM,
            "to": recipients,
            "subject": subject,
            "text": text,
            "html": html,
        },
        timeout=15,
    )
    r.raise_for_status()


# ----------------------------
# Ingest (RSS from source_feed)
# ----------------------------
def upsert_article(conn: psycopg.Connection, url: str, title: str, publisher: str,
                   published_at: Optional[datetime], raw_html: Optional[str], clean_text: Optional[str]) -> int:
    fingerprint = sha256_of(url)
    row = conn.execute(
        """
        INSERT INTO public.article (url, title, publisher, published_at, sha256, raw_html, clean_text)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (sha256) DO UPDATE
          SET title = COALESCE(EXCLUDED.title, public.article.title),
              publisher = COALESCE(EXCLUDED.publisher, public.article.publisher),
              published_at = COALESCE(EXCLUDED.published_at, public.article.published_at)
        RETURNING id;
        """,
        (url, title, publisher, published_at, fingerprint, raw_html, clean_text),
    ).fetchone()
    return int(row["id"])


def insert_tag_watch(conn: psycopg.Connection, article_id: int, watchlist_id: int):
    conn.execute(
        """
        INSERT INTO public.article_tag (article_id, watchlist_id, tag_kind)
        VALUES (%s, %s, 'watch')
        ON CONFLICT DO NOTHING;
        """,
        (article_id, watchlist_id),
    )


def set_article_summary(conn: psycopg.Connection, article_id: int, model: str, summary: str):
    conn.execute(
        """
        INSERT INTO public.article_nlp (article_id, model, summary)
        VALUES (%s, %s, %s)
        ON CONFLICT (article_id) DO UPDATE
          SET model = EXCLUDED.model,
              summary = EXCLUDED.summary;
        """,
        (article_id, model, summary),
    )


def ingest_google_news(conn: psycopg.Connection, minutes: int) -> Tuple[int, int]:
    """Ingest from active RSS feeds (source_feed), resolving Google News links to publisher."""
    now = datetime.now(timezone.utc)

    feeds = conn.execute(
        """
        SELECT id, url, name, period_minutes, last_checked_at
        FROM public.source_feed
        WHERE active = TRUE AND kind = 'rss'
          AND (last_checked_at IS NULL OR last_checked_at <= now() - (period_minutes || ' minutes')::interval)
        ORDER BY id;
        """
    ).fetchall()

    total_seen = 0
    inserted = 0

    # Watchlist map (for tagging)
    wl = conn.execute(
        "SELECT id, symbol, company FROM public.watchlist WHERE enabled = TRUE ORDER BY id"
    ).fetchall()
    watchlist = list(wl)

    for feed in feeds:
        url_feed: str = feed["url"]
        f = feedparser.parse(url_feed)
        log.info("parsed feed: %s entries=%s", url_feed, len(f.entries))
        for e in f.entries:
            total_seen += 1

            raw_link = e.get("link") or e.get("id") or ""
            if not raw_link:
                continue

            final_url = resolve_google_news_url(raw_link) or raw_link
            final_url = strip_tracking(final_url)

            # Skip unresolved GN or banned sources
            if "news.google.com" in hostname_of(final_url):
                continue
            if is_banned_link(final_url):
                continue
            if DISALLOW_EXT_REGEX.search(final_url):
                continue

            # Parse published date if present
            published_at = None
            try:
                if getattr(e, "published_parsed", None):
                    published_at = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            except Exception:
                published_at = None

            r = fetch_url(final_url)
            if not r:
                continue

            page_title, publisher = extract_title_publisher(r.text, final_url)

            # Fallbacks for bad feed items
            title = (e.get("title") or "").strip()
            if (not title) or (title.lower() == "google news"):
                title = page_title or final_url
            if (publisher or "").strip().lower() == "google news":
                publisher = hostname_of(final_url).replace("www.", "")

            # Store
            article_id = upsert_article(
                conn,
                url=final_url,
                title=title,
                publisher=publisher,
                published_at=published_at,
                raw_html=r.text[:250_000],
                clean_text=None,  # can add trafilatura later if you like
            )
            inserted += 1

            # Tag against all enabled watchlist symbols found in the title or URL
            low_t = f"{title} {final_url}".lower()
            for w in watchlist:
                sym = (w["symbol"] or "").lower()
                comp = (w["company"] or "").lower()
                if (sym and sym in low_t) or (comp and comp in low_t):
                    insert_tag_watch(conn, article_id, int(w["id"]))

            # Summarize (best-effort)
            summary = summarize_text(final_url, title, publisher, r.text)
            if summary:
                set_article_summary(conn, article_id, OPENAI_MODEL, summary)

        # Update last_checked_at even if nothing inserted
        conn.execute("UPDATE public.source_feed SET last_checked_at = now() WHERE id = %s", (feed["id"],))

    return total_seen, inserted


# ----------------------------
# Digest build & send
# ----------------------------
def build_items_for_digest(conn: psycopg.Connection, minutes: int) -> List[Dict[str, Any]]:
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(minutes=minutes)
    rows = conn.execute(
        """
        SELECT a.id, a.title, a.publisher, a.url,
               COALESCE(n.summary, '') AS summary,
               a.published_at
        FROM public.article a
        LEFT JOIN public.article_nlp n ON n.article_id = a.id
        WHERE (a.published_at IS NULL OR (a.published_at >= %s AND a.published_at <= %s))
          AND a.url !~* '(marketbeat\\.com|newser\\.com)'
          AND COALESCE(a.publisher,'') !~* '(marketbeat|newser)'
        ORDER BY COALESCE(a.published_at, a.created_at) DESC, a.id DESC
        LIMIT 200
        """,
        (window_start, window_end),
    ).fetchall()

    items = []
    for r in rows:
        items.append(
            dict(
                id=int(r["id"]),
                title=r["title"] or strip_tracking(r["url"]),
                publisher=(r["publisher"] or hostname_of(r["url"]).replace("www.", "")),
                url=r["url"],
                summary=r["summary"] or "",
                published_at=r["published_at"].isoformat() if r["published_at"] else None,
            )
        )
    return items


# ----------------------------
# Routes
# ----------------------------
def require_admin(req: Request):
    tok = req.headers.get("x-admin-token")
    if not ADMIN_TOKEN or tok != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.get("/admin/health", response_class=JSONResponse)
def admin_health():
    try:
        with db() as conn:
            counts = {
                "articles": conn.execute("SELECT COUNT(*) c FROM public.article").fetchone()["c"],
                "summarized": conn.execute("SELECT COUNT(*) c FROM public.article_nlp").fetchone()["c"],
                "recipients": conn.execute("SELECT COUNT(*) c FROM public.recipients").fetchone()["c"],
                "watchlist": conn.execute("SELECT COUNT(*) c FROM public.watchlist").fetchone()["c"],
            }
    except Exception as e:
        return JSONResponse(
            {
                "env": {
                    "DATABASE_URL": bool(DATABASE_URL),
                    "OPENAI_API_KEY": bool(OPENAI_API_KEY),
                    "OPENAI_MODEL": bool(OPENAI_MODEL),
                    "OPENAI_MAX_OUTPUT_TOKENS": bool(OPENAI_MAX_OUTPUT_TOKENS),
                    "MAILGUN_API_KEY": bool(MAILGUN_API_KEY),
                    "MAILGUN_DOMAIN": bool(MAILGUN_DOMAIN),
                    "MAILGUN_FROM": bool(MAILGUN_FROM),
                    "OWNER_EMAIL": bool(OWNER_EMAIL),
                    "ADMIN_TOKEN": bool(ADMIN_TOKEN),
                },
                "error": str(e),
            },
            status_code=500,
        )

    return JSONResponse(
        {
            "env": {
                "DATABASE_URL": bool(DATABASE_URL),
                "OPENAI_API_KEY": bool(OPENAI_API_KEY),
                "OPENAI_MODEL": bool(OPENAI_MODEL),
                "OPENAI_MAX_OUTPUT_TOKENS": bool(OPENAI_MAX_OUTPUT_TOKENS),
                "MAILGUN_API_KEY": bool(MAILGUN_API_KEY),
                "MAILGUN_DOMAIN": bool(MAILGUN_DOMAIN),
                "MAILGUN_FROM": bool(MAILGUN_FROM),
                "OWNER_EMAIL": bool(OWNER_EMAIL),
                "ADMIN_TOKEN": bool(ADMIN_TOKEN),
            },
            "counts": counts,
        }
    )


@app.get("/admin/test_openai", response_class=PlainTextResponse)
def admin_test_openai():
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY missing")
    try:
        r = _openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Reply 'OK' if you received this."}],
            max_tokens=8,
        )
        return r.choices[0].message.content or "OK"
    except Exception as e:
        raise HTTPException(500, f"OpenAI error: {e}")


@app.post("/admin/init", response_class=PlainTextResponse)
def admin_init(request: Request):
    require_admin(request)
    with db() as conn:
        conn.execute(SCHEMA_SQL)

        owner_clause = "SELECT 1 WHERE FALSE"
        if OWNER_EMAIL:
            owner_clause = f"INSERT INTO public.recipients (email, enabled) VALUES ('{OWNER_EMAIL}', TRUE) ON CONFLICT (email) DO NOTHING"

        conn.execute(SEED_SQL.format(owner_recipient=owner_clause))
    return "Initialized."


def _minutes_from_query(request: Request, default: int = 1440) -> int:
    try:
        m = int(request.query_params.get("minutes", str(default)))
        return max(30, min(60 * 24 * 60, m))  # 30 min to ~60 days clamp
    except Exception:
        return default


@app.post("/cron/ingest", response_class=JSONResponse)
def cron_ingest(request: Request):
    require_admin(request)
    minutes = _minutes_from_query(request, 1440)
    with db() as conn:
        found, inserted = ingest_google_news(conn, minutes=minutes)
    return JSONResponse({"ok": True, "found_urls": found, "inserted": inserted})


@app.post("/cron/digest", response_class=JSONResponse)
def cron_digest(request: Request):
    require_admin(request)
    minutes = _minutes_from_query(request, 1440)

    with db() as conn:
        start = datetime.now(timezone.utc)
        # log start
        conn.execute(
            "INSERT INTO public.delivery_log (run_date, run_started_at) VALUES (CURRENT_DATE, now())"
        )

        items = build_items_for_digest(conn, minutes)
        to_rows = conn.execute("SELECT email FROM public.recipients WHERE enabled = TRUE ORDER BY id").fetchall()
        recipients = [r["email"] for r in to_rows] or (["you@example.com"] if OWNER_EMAIL is None else [OWNER_EMAIL])

        subject = f"{APP_NAME} — {start.strftime('%Y-%m-%d %H:%M')}"
        send_email(subject, items, minutes, recipients)

        # finalize log
        conn.execute(
            """
            UPDATE public.delivery_log
               SET run_ended_at = now(),
                   sent_to      = %s,
                   items        = %s,
                   summarized   = %s
             WHERE id = (SELECT max(id) FROM public.delivery_log)
            """,
            (",".join(recipients), len(items), sum(1 for it in items if it.get("summary"))),
        )

    return JSONResponse({"ok": True, "sent_to": recipients, "items": len(items)})

