# app.py
from __future__ import annotations

import os
import sys
import hashlib
import logging
import datetime as dt
from math import ceil
from typing import List, Tuple, Dict, Any, Optional
from urllib.parse import urlsplit, urlunsplit, urlencode, parse_qs, quote_plus

import httpx
import feedparser
from bs4 import BeautifulSoup
from jinja2 import Template

import psycopg
from psycopg.rows import dict_row

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("quantbrief")

# -----------------------------------------------------------------------------
# Env
# -----------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "512"))

MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = os.getenv("MAILGUN_DOMAIN", "")
MAILGUN_FROM = os.getenv("MAILGUN_FROM", "QuantBrief Daily <daily@mg.example.com>")

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

if not DATABASE_URL:
    log.warning("DATABASE_URL missing")
if not OPENAI_API_KEY:
    log.warning("OPENAI_API_KEY missing")

# -----------------------------------------------------------------------------
# OpenAI client (1.51.0) — Chat Completions (no proxies, no temperature)
# -----------------------------------------------------------------------------
from openai import OpenAI
_openai = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# Constants / Filters
# -----------------------------------------------------------------------------
BAN_DOMAINS = {
    "marketbeat.com",
    "www.marketbeat.com",
    "newser.com",
    "www.newser.com",
    "news.google.com",       # we skip aggregator after resolution anyway
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI()


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------
def get_conn():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def exec_sql_batch(conn, sql: str) -> None:
    """
    Execute semicolon-separated SQL statements (psycopg3 can't run multi-stmt in one execute).
    Keep this simple: our SEED has no $$-functions or embedded semicolons in string literals.
    """
    for stmt in sql.split(";"):
        s = stmt.strip()
        if not s or s.startswith("--"):
            continue
        conn.execute(s)


# -----------------------------------------------------------------------------
# Seed / migrations - only ADD missing columns / indexes; never DROP
# -----------------------------------------------------------------------------
SEED_SQL = """
-- recipients safety columns
ALTER TABLE IF EXISTS public.recipients
  ADD COLUMN IF NOT EXISTS enabled boolean NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();

-- watchlist safety columns (symbol + company assumed existing from your DB)
ALTER TABLE IF EXISTS public.watchlist
  ADD COLUMN IF NOT EXISTS enabled boolean NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();

-- source_feed safety columns
CREATE TABLE IF NOT EXISTS public.source_feed (
  id bigserial PRIMARY KEY,
  kind text NOT NULL DEFAULT 'rss',
  url text UNIQUE NOT NULL,
  active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  name text,
  period_minutes integer NOT NULL DEFAULT 60,
  last_checked_at timestamptz NULL
);

-- article table exists in your DB; ensure helpful indexes
CREATE INDEX IF NOT EXISTS idx_article_published ON public.article (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_article_url ON public.article (url);
CREATE INDEX IF NOT EXISTS idx_article_publisher ON public.article (publisher);

-- article_nlp safety columns
CREATE TABLE IF NOT EXISTS public.article_nlp (
  id bigserial PRIMARY KEY,
  article_id bigint NOT NULL REFERENCES public.article(id) ON DELETE CASCADE,
  summary text,
  model text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_article_nlp_article ON public.article_nlp(article_id);

-- delivery_log safety columns
CREATE TABLE IF NOT EXISTS public.delivery_log (
  id bigserial PRIMARY KEY,
  run_date date NOT NULL,
  run_started_at timestamptz NULL,
  run_ended_at timestamptz NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  sent_to text,
  items integer NOT NULL DEFAULT 0,
  summarized integer NOT NULL DEFAULT 0
);

-- optional: seed initial watchlist TLN if empty
INSERT INTO public.watchlist (symbol, company, enabled)
SELECT 'TLN', 'Talen Energy Corporation', true
WHERE NOT EXISTS (SELECT 1 FROM public.watchlist);

-- seed a Google News feed for TLN if none present
INSERT INTO public.source_feed (kind, url, active, name, period_minutes)
SELECT
  'rss',
  'https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN&hl=en-US&gl=US&ceid=US:en',
  true,
  'Google News: Talen Energy',
  60
WHERE NOT EXISTS (SELECT 1 FROM public.source_feed WHERE url LIKE 'https://news.google.com/rss%');
"""


# -----------------------------------------------------------------------------
# Utility: Google News URL building / resolution
# -----------------------------------------------------------------------------
def add_when_days_to_gnews_url(url: str, days: int) -> str:
    """
    If this is a Google News 'search' RSS, add +when:Xd to the q= param (if missing).
    """
    try:
        parts = urlsplit(url)
        if "news.google.com" not in parts.netloc or "/rss/search" not in parts.path:
            return url
        qs = parse_qs(parts.query, keep_blank_values=True)
        q = qs.get("q", [""])[0]
        if "when:" not in q and days > 0:
            # Add site exclusions for banned outlets at the query-level too
            minus_sites = " ".join([f"-site:{d}" for d in BAN_DOMAINS if d != "news.google.com"])
            q_new = f"{q} {minus_sites} when:{days}d".strip()
            qs["q"] = [q_new]
            new_query = urlencode({k: v[0] for k, v in qs.items()})
            return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))
        return url
    except Exception:
        return url


def resolve_google_news(url: str) -> str:
    """
    Return the original article URL for Google News aggregator links.
    Strategy: url= param; else follow redirects.
    """
    try:
        netloc = urlsplit(url).netloc.lower()
        if "news.google.com" not in netloc:
            return url

        qs = parse_qs(urlsplit(url).query)
        if "url" in qs and qs["url"]:
            return qs["url"][0]

        with httpx.Client(follow_redirects=True, timeout=10, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(url)
            return str(resp.url)
    except Exception:
        return url


def url_domain(u: str) -> str:
    try:
        return urlsplit(u).netloc.lower()
    except Exception:
        return ""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


# -----------------------------------------------------------------------------
# Scrape helpers (very light)
# -----------------------------------------------------------------------------
def fetch_html(url: str) -> str:
    try:
        with httpx.Client(follow_redirects=True, timeout=15, headers={"User-Agent": USER_AGENT}) as client:
            r = client.get(url)
            if r.status_code == 200 and r.text:
                return r.text
    except Exception:
        pass
    return ""


def extract_title_from_html(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Prefer og:title
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            return og["content"].strip()
        if soup.title and soup.title.text:
            return soup.title.text.strip()
    except Exception:
        pass
    return ""


# -----------------------------------------------------------------------------
# OpenAI summarize
# -----------------------------------------------------------------------------
def summarize_article(url: str, title: str, snippet: str) -> str:
    """
    Use Chat Completions with OPENAI_MODEL. Avoid temperature; keep short.
    """
    prompt = (
        "Summarize the article in 1 concise bullet (≤25 words), focusing on the most material, "
        "investor-relevant point. Avoid hype. No ticker cashtags.\n\n"
        f"Title: {title or '(unknown)'}\n"
        f"URL: {url}\n"
        f"Snippet/Extract:\n{snippet[:2000]}"
    )

    try:
        resp = _openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You write terse, factual finance summaries."},
                {"role": "user", "content": prompt},
            ],
            # **Important**: some models reject non-default temperature; omit entirely.
            # Keep defaults; if needed we could set max_tokens, but some models reject it.
        )
        text = (resp.choices[0].message.content or "").strip()
        # Ensure a single '-' bullet prefix like your format
        if not text.startswith("-"):
            text = "- " + text
        return text
    except Exception as e:
        log.warning("OpenAI summarize failed: %s", e)
        return "- Summary unavailable."


# -----------------------------------------------------------------------------
# Email (Mailgun)
# -----------------------------------------------------------------------------
EMAIL_TEMPLATE = Template(
    """QuantBrief Daily — {{ when_local }}
Window: last {{ minutes }} minutes ending {{ now_iso }}

{% for it in items %}
<a href="{{ it.url }}">{{ it.title }}</a> — {{ it.publisher }}
{{ it.summary }}
{% endfor %}

Sources are links only. We don’t republish paywalled content.
"""
)


def send_email(to_list: List[str], subject: str, html_body: str, text_body: str = "") -> Tuple[bool, str]:
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN and MAILGUN_FROM):
        return False, "Mailgun env missing"

    try:
        with httpx.Client(timeout=15, auth=("api", MAILGUN_API_KEY)) as client:
            data = {
                "from": MAILGUN_FROM,
                "to": to_list,
                "subject": subject,
                "html": html_body,
            }
            if text_body:
                data["text"] = text_body
            r = client.post(f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages", data=data)
            if r.status_code >= 200 and r.status_code < 300:
                return True, "sent"
            return False, f"mailgun error: {r.status_code} {r.text}"
    except Exception as e:
        return False, str(e)


# -----------------------------------------------------------------------------
# Ingest logic
# -----------------------------------------------------------------------------
def build_feed_urls(conn, minutes: int) -> List[str]:
    """Use source_feed if present; else synthesize a Google News RSS from watchlist."""
    days = max(1, ceil(minutes / 1440)) if minutes and minutes > 0 else 1

    rows = conn.execute("SELECT url FROM public.source_feed WHERE active = TRUE AND kind = 'rss'").fetchall()
    urls = [r["url"] for r in rows]

    if not urls:
        # synthesize from watchlist
        wl = conn.execute("SELECT symbol, company FROM public.watchlist WHERE enabled = TRUE").fetchall()
        for row in wl:
            symbol = (row["symbol"] or "").strip()
            company = (row["company"] or "").strip()
            if not (symbol or company):
                continue
            query_bits = []
            if company:
                query_bits.append(f"({company})")
            if symbol:
                query_bits.append(symbol)
            # add banned sites at query level
            minus_sites = " ".join([f"-site:{d}" for d in BAN_DOMAINS if d != "news.google.com"])
            query = quote_plus(" OR ".join(query_bits) + " " + minus_sites + f" when:{days}d")
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            urls.append(url)
    else:
        # if an existing url is google news search, add when:Xd unless already present
        urls = [add_when_days_to_gnews_url(u, days) for u in urls]

    return urls


def ingest_rss_url(conn, url: str, cutoff_utc: Optional[dt.datetime]) -> Tuple[int, int]:
    """
    Returns (found, inserted)
    """
    found = inserted = 0
    log.info("parsed feed: %s", url)
    with httpx.Client(timeout=15, headers={"User-Agent": USER_AGENT}) as client:
        resp = client.get(url)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)

    for entry in parsed.entries:
        found += 1
        raw_link = getattr(entry, "link", "") or ""
        if not raw_link:
            continue

        # resolve aggregator to original
        final_url = resolve_google_news(raw_link)
        publisher = url_domain(final_url)

        # strong filter: skip banned and still-aggregator domains
        if not publisher or publisher in BAN_DOMAINS or "news.google.com" in publisher:
            continue

        # published time filter
        published_at = None
        try:
            # feedparser stores time struct in .published_parsed or .updated_parsed
            t = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
            if t:
                published_at = dt.datetime(*t[:6], tzinfo=dt.timezone.utc)
        except Exception:
            published_at = None

        if cutoff_utc and published_at and published_at < cutoff_utc:
            continue

        # Fetch HTML to improve title / summary seed
        html = fetch_html(final_url)
        extracted_title = extract_title_from_html(html)
        title = (getattr(entry, "title", "") or "").strip()
        if not title or title.lower() == "google news":
            title = extracted_title or publisher

        # very short snippet for the LLM (meta description if exists)
        snippet = ""
        if html:
            try:
                soup = BeautifulSoup(html, "html.parser")
                md = soup.find("meta", attrs={"name": "description"})
                if md and md.get("content"):
                    snippet = md["content"].strip()[:500]
            except Exception:
                pass

        # compute sha on final_url (dedupe on URL level)
        sha = sha256_text(final_url)

        # Insert article
        row_id = None
        try:
            rec = conn.execute(
                """
                INSERT INTO public.article (url, canonical_url, title, publisher, published_at, sha256, raw_html, clean_text)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (sha256) DO NOTHING
                RETURNING id
                """,
                (final_url, None, title, publisher, published_at, sha, html, None),
            ).fetchone()
            if rec and rec.get("id"):
                row_id = rec["id"]
                inserted += 1
            else:
                # already existed: fetch id
                got = conn.execute("SELECT id FROM public.article WHERE sha256 = %s", (sha,)).fetchone()
                if got:
                    row_id = got["id"]
        except Exception as e:
            log.warning("insert article failed (%s): %s", final_url, e)
            continue

        # Summarize if new (or missing summary)
        if row_id:
            has_nlp = conn.execute(
                "SELECT 1 FROM public.article_nlp WHERE article_id = %s LIMIT 1", (row_id,)
            ).fetchone()
            if not has_nlp:
                bullet = summarize_article(final_url, title, snippet)
                try:
                    conn.execute(
                        "INSERT INTO public.article_nlp (article_id, model, summary) VALUES (%s, %s, %s)",
                        (row_id, OPENAI_MODEL, bullet),
                    )
                except Exception as e:
                    log.warning("Summarize store failed for article %s: %s", row_id, e)

    return found, inserted


def ingest_google_news(conn, minutes: int) -> Tuple[int, int]:
    urls = build_feed_urls(conn, minutes)
    cutoff = None
    if minutes and minutes > 0:
        cutoff = now_utc() - dt.timedelta(minutes=minutes)

    total_found = 0
    total_inserted = 0
    for u in urls:
        f, i = ingest_rss_url(conn, u, cutoff)
        total_found += f
        total_inserted += i
    return total_found, total_inserted


# -----------------------------------------------------------------------------
# Digest (email)
# -----------------------------------------------------------------------------
def make_digest_items(conn, minutes: int) -> List[Dict[str, Any]]:
    cutoff = now_utc() - dt.timedelta(minutes=minutes)
    rows = conn.execute(
        """
        SELECT a.id, a.title, a.url, a.publisher, a.published_at,
               coalesce(n.summary, '') as summary
        FROM public.article a
        LEFT JOIN public.article_nlp n ON n.article_id = a.id
        WHERE a.published_at IS NULL OR a.published_at >= %s
        ORDER BY a.published_at DESC NULLS LAST, a.id DESC
        LIMIT 100
        """,
        (cutoff,),
    ).fetchall()

    items = []
    for r in rows:
        pub = (r["publisher"] or "").lower()
        if not pub or pub in BAN_DOMAINS or "news.google.com" in pub:
            continue
        title = (r["title"] or "").strip() or pub
        items.append(
            {
                "url": r["url"],
                "title": title,
                "publisher": pub,
                "summary": (r["summary"] or "- (no summary)").strip(),
            }
        )
    return items


def render_digest_email(items: List[Dict[str, Any]], minutes: int) -> str:
    return EMAIL_TEMPLATE.render(
        when_local=now_utc().astimezone().strftime("%Y-%m-%d %H:%M"),
        minutes=minutes,
        now_iso=now_utc().isoformat(timespec="seconds"),
        items=items,
    )


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
def require_admin(req: Request):
    tok = req.headers.get("x-admin-token") or req.headers.get("authorization") or ""
    if tok.startswith("Bearer "):
        tok = tok[7:]
    if ADMIN_TOKEN and tok == ADMIN_TOKEN:
        return
    if not ADMIN_TOKEN:
        # if no token configured, allow (dev)
        return
    raise HTTPException(status_code=403, detail="forbidden")


@app.get("/", response_class=PlainTextResponse)
def root():
    return "ok"


@app.get("/admin/health", response_class=PlainTextResponse)
def admin_health():
    # env present?
    env_ok = {
        "DATABASE_URL": bool(DATABASE_URL),
        "OPENAI_API_KEY": bool(OPENAI_API_KEY),
        "OPENAI_MODEL": bool(OPENAI_MODEL),
        "OPENAI_MAX_OUTPUT_TOKENS": bool(OPENAI_MAX_OUTPUT_TOKENS),
        "MAILGUN_API_KEY": bool(MAILGUN_API_KEY),
        "MAILGUN_DOMAIN": bool(MAILGUN_DOMAIN),
        "MAILGUN_FROM": bool(MAILGUN_FROM),
        "OWNER_EMAIL": bool(os.getenv("OWNER_EMAIL", "")),
        "ADMIN_TOKEN": bool(ADMIN_TOKEN),
    }

    counts = {}
    try:
        with get_conn() as conn:
            a = conn.execute("SELECT COUNT(*) AS c FROM public.article").fetchone()
            n = conn.execute("SELECT COUNT(*) AS c FROM public.article_nlp").fetchone()
            r = conn.execute("SELECT COUNT(*) AS c FROM public.recipients").fetchone()
            w = conn.execute("SELECT COUNT(*) AS c FROM public.watchlist").fetchone()
        counts = {
            "articles": a["c"],
            "summarized": n["c"],
            "recipients": r["c"],
            "watchlist": w["c"],
        }
    except Exception:
        counts = {"error": "db not reachable"}

    lines = [
        "env",
        "---",
        str({k: bool(v) for k, v in env_ok.items()}),
        "",
        "counts",
        "------",
        "\n".join([f"{k}={v}" for k, v in counts.items()]),
    ]
    return "\n".join(lines)


@app.get("/admin/test_openai", response_class=PlainTextResponse)
def admin_test_openai():
    try:
        resp = _openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Reply with exactly: OK"},
            ],
        )
        sample = (resp.choices[0].message.content or "").strip()
        ok = sample.upper().startswith("OK")
        lines = [
            "   ok model      sample",
            "   -- -----      ------",
            f"{str(bool(ok)):<5} {OPENAI_MODEL:<10} {sample[:40]}",
        ]
        return "\n".join(lines)
    except Exception as e:
        lines = [
            "   ok model      err",
            "   -- -----      ---",
            f"False {OPENAI_MODEL:<10} {str(e)[:120]}",
        ]
        return "\n".join(lines)


@app.post("/admin/init", response_class=PlainTextResponse)
def admin_init(req: Request):
    require_admin(req)
    owner_email = os.getenv("OWNER_EMAIL", "").strip()

    with get_conn() as conn:
        exec_sql_batch(conn, SEED_SQL)
        if owner_email:
            conn.execute(
                """
                INSERT INTO public.recipients (email, enabled)
                VALUES (%s, TRUE)
                ON CONFLICT (email) DO UPDATE SET enabled = EXCLUDED.enabled
                """,
                (owner_email,),
            )

    return "Initialized.\n"


@app.post("/cron/ingest", response_class=PlainTextResponse)
def cron_ingest(req: Request):
    require_admin(req)

    try:
        minutes = int(req.query_params.get("minutes", "1440"))
    except Exception:
        minutes = 1440

    with get_conn() as conn:
        found, inserted = ingest_google_news(conn, minutes)
    return f"""
  ok found_urls inserted
  -- ---------- --------
True          {found:<8} {inserted}
""".strip() + "\n"


@app.post("/cron/digest", response_class=PlainTextResponse)
def cron_digest(req: Request):
    require_admin(req)
    try:
        minutes = int(req.query_params.get("minutes", "1440"))
    except Exception:
        minutes = 1440

    run_start = now_utc()
    to_list: List[str] = []
    items: List[Dict[str, Any]] = []

    with get_conn() as conn:
        # recipients
        rows = conn.execute("SELECT email FROM public.recipients WHERE enabled = TRUE ORDER BY 1").fetchall()
        to_list = [r["email"] for r in rows]

        # items
        items = make_digest_items(conn, minutes)

        # render + send
        html = render_digest_email(items, minutes)
        subj = f"QuantBrief Daily — {run_start.astimezone().strftime('%Y-%m-%d %H:%M')}"
        ok, msg = send_email(to_list or ["you@example.com"], subj, html, html)

        # log delivery
        conn.execute(
            """
            INSERT INTO public.delivery_log (run_date, run_started_at, run_ended_at, sent_to, items, summarized)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                run_start.date(),
                run_start,
                now_utc(),
                ",".join(to_list) if to_list else "you@example.com",
                len(items),
                sum(1 for it in items if it.get("summary")),
            ),
        )

    if not to_list:
        to_list = ["you@example.com"]

    if items:
        return f"""
  ok sent_to                       items summarized
  -- -------                       ----- ----------
True {",".join(to_list)}     {len(items):<5} {sum(1 for it in items if it.get("summary")):<10}
""".strip() + "\n"
    else:
        return f"""
  ok sent_to                       items summarized
  -- -------                       ----- ----------
True {",".join(to_list)}     0     0
""".strip() + "\n"
