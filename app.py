import os
import re
import json
import time
import httpx
import feedparser
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import psycopg
from psycopg.rows import dict_row

# ------------------------------------------------------------------------------
# Config & logging
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)s:%(name)s:%(message)s",
)
log = logging.getLogger("quantbrief")

APP_NAME = os.getenv("APP_NAME", "quantbrief")

# Admin/API auth
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
AUTH_HEADER = "authorization"

# DB
DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    # render.com normally injects DATABASE_URL
    log.warning("DATABASE_URL not set; ensure it is provided in production.")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE", "1")

# Feeds / Query
QB_QUERY = os.getenv("QB_QUERY", "(Talen Energy) OR TLN")
QB_DAYS = int(os.getenv("QB_DAYS", "7"))
QB_EXCLUDE = [d.strip() for d in os.getenv("QB_EXCLUDE", "newser.com,www.newser.com,marketbeat.com,www.marketbeat.com").split(",") if d.strip()]
QB_ALLOW = [d.strip() for d in os.getenv("QB_ALLOW", "").split(",") if d.strip()]

# Email via Mailgun
MAILGUN_DOMAIN = os.getenv("MAILGUN_DOMAIN", "")
MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", f"QuantBrief Daily <daily@mg.{MAILGUN_DOMAIN}>") if MAILGUN_DOMAIN else os.getenv("FROM_EMAIL", "QuantBrief Daily <daily@example.com>")
TO_EMAILS = [e.strip() for e in os.getenv("TO_EMAILS", "you@example.com").split(",") if e.strip()]

# Server
PORT = int(os.getenv("PORT", "10000"))

# ------------------------------------------------------------------------------
# Database schema / migrations
# ------------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS source_feed (
    id            BIGSERIAL PRIMARY KEY,
    kind          TEXT NOT NULL,                   -- e.g., 'google-news'
    url           TEXT NOT NULL,
    active        BOOLEAN NOT NULL DEFAULT TRUE,
    name          TEXT NOT NULL DEFAULT 'Google News',
    persist_days  INTEGER NOT NULL DEFAULT 30,     -- keep found urls this many days
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Uniqueness ensures upsert works and avoids duplicates for the same feed.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'uq_source_feed_url'
    ) THEN
        ALTER TABLE source_feed ADD CONSTRAINT uq_source_feed_url UNIQUE (url);
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS found_url (
    id            BIGSERIAL PRIMARY KEY,
    url           TEXT NOT NULL,
    title         TEXT,
    feed_id       BIGINT REFERENCES source_feed(id) ON DELETE SET NULL,
    language      TEXT,
    published_at  TIMESTAMPTZ,
    found_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen     TIMESTAMPTZ,
    seen_count    INTEGER NOT NULL DEFAULT 1
);

-- Make url unique so the same link across feeds dedupes; we still track last_seen/seen_count.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'uq_found_url_url'
    ) THEN
        ALTER TABLE found_url ADD CONSTRAINT uq_found_url_url UNIQUE (url);
    END IF;
END $$;

-- Ensure columns exist when deploying over older versions
ALTER TABLE source_feed ADD COLUMN IF NOT EXISTS persist_days INTEGER NOT NULL DEFAULT 30;
ALTER TABLE found_url  ADD COLUMN IF NOT EXISTS feed_id BIGINT REFERENCES source_feed(id) ON DELETE SET NULL;
ALTER TABLE found_url  ADD COLUMN IF NOT EXISTS last_seen TIMESTAMPTZ;
ALTER TABLE found_url  ADD COLUMN IF NOT EXISTS seen_count INTEGER NOT NULL DEFAULT 1;

CREATE TABLE IF NOT EXISTS summaries (
    id         BIGSERIAL PRIMARY KEY,
    url        TEXT NOT NULL UNIQUE,
    summary    TEXT NOT NULL,
    model      TEXT,
    tokens     INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS digest_log (
    id          BIGSERIAL PRIMARY KEY,
    sent_to     TEXT NOT NULL,
    items       INTEGER NOT NULL,
    summarized  INTEGER NOT NULL,
    ok          BOOLEAN NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

def get_conn() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def exec_sql_batch(conn: psycopg.Connection, sql: str) -> None:
    with conn:
        with conn.cursor() as cur:
            for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                conn.execute(stmt + ";")

# ------------------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------------------

def now() -> datetime:
    return datetime.now(timezone.utc)

def domain_of(url: str) -> str:
    m = re.match(r"^https?://([^/]+)/?", url or "", flags=re.I)
    return (m.group(1) if m else "").lower()

def require_admin(request: Request) -> None:
    if not ADMIN_TOKEN:
        return  # no auth set; allow (development)
    auth = request.headers.get(AUTH_HEADER, "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Unauthorized")
    token = auth.split(" ", 1)[1].strip()
    if token != ADMIN_TOKEN:
        raise HTTPException(401, "Unauthorized")

# ------------------------------------------------------------------------------
# Feed management
# ------------------------------------------------------------------------------

def build_google_news_url(query: str, days: int, excl: List[str], allow: List[str]) -> str:
    """
    Builds a Google News RSS search with `when:X d` window and site filters.
    """
    site_filters = []
    for d in excl:
        if d:
            site_filters.append(f"-site:{d}")
    for d in allow:
        if d:
            site_filters.append(f"site:{d}")
    # Google News "when:7d" works inside the query text
    query_with_filters = f"({query}) {' '.join(site_filters)} when:{days}d".strip()
    q = quote_plus(query_with_filters)

    # en-US feed; change via env if needed later
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US%3Aen"

def upsert_feed(kind: str, url: str, name: str, persist_days: int = 30) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO source_feed (kind, url, active, name, persist_days, created_at, updated_at)
            VALUES (%s, %s, TRUE, %s, %s, NOW(), NOW())
            ON CONFLICT (url)
            DO UPDATE SET
                kind = EXCLUDED.kind,
                active = TRUE,
                name = EXCLUDED.name,
                persist_days = EXCLUDED.persist_days,
                updated_at = NOW()
            RETURNING id
            """,
            (kind, url, name, persist_days),
        )
        row = cur.fetchone()
        return int(row["id"])

# ------------------------------------------------------------------------------
# OpenAI – capability-aware wrapper (avoids 400 spam)
# ------------------------------------------------------------------------------

@dataclass
class _OpenAICaps:
    tokens_param: str = "max_tokens"   # or "max_completion_tokens"
    allow_temperature: bool = True
    probed: bool = False

_OPENAI_CAPS = _OpenAICaps()

def _probe_openai_caps() -> _OpenAICaps:
    global _OPENAI_CAPS
    if _OPENAI_CAPS.probed or not OPENAI_API_KEY:
        _OPENAI_CAPS.probed = True
        return _OPENAI_CAPS

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    base = {"model": OPENAI_MODEL, "messages": [{"role": "user", "content": "ok"}]}

    # tokens param
    for candidate in ("max_tokens", "max_completion_tokens"):
        try:
            r = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={**base, candidate: 1},
                timeout=15.0,
            )
            if r.status_code < 400:
                _OPENAI_CAPS.tokens_param = candidate
                break
            if "unsupported_parameter" in r.text and candidate in r.text:
                continue
        except Exception:
            pass

    # temperature support
    try:
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={**base, _OPENAI_CAPS.tokens_param: 1, "temperature": 0.5},
            timeout=15.0,
        )
        if r.status_code >= 400 and ("unsupported_value" in r.text and "temperature" in r.text):
            _OPENAI_CAPS.allow_temperature = False
    except Exception:
        pass

    _OPENAI_CAPS.probed = True
    return _OPENAI_CAPS

def _chat_complete(messages: List[Dict[str, str]], max_tokens: int = 240) -> str:
    if not OPENAI_API_KEY:
        return ""
    caps = _probe_openai_caps()

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": messages,
        caps.tokens_param: max_tokens,
    }

    # only include temperature if the model allows a non-default
    try:
        if caps.allow_temperature and OPENAI_TEMPERATURE not in ("", None):
            t = float(OPENAI_TEMPERATURE)
            if t != 1.0:
                payload["temperature"] = t
    except Exception:
        pass

    r = httpx.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30.0)
    r.raise_for_status()
    j = r.json()
    return (j["choices"][0]["message"]["content"] or "").strip()

def summarize_item(title: str, url: str) -> str:
    """
    Summarize briefly and factually from headline + link context.
    If OpenAI is unavailable, fall back to a compact headline line.
    """
    system = (
        "You are a financial news summarizer. Summaries must be 1–2 concise sentences,"
        " neutral tone, no hype, no emojis. Include the company/ticker if obvious."
    )
    user = f"Headline: {title}\nLink: {url}\nTask: Summarize key development and why it matters for investors."
    try:
        text = _chat_complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=160,
        )
        if text:
            return text
    except Exception as e:
        log.warning("OpenAI summarize error: %s", str(e))
    # Fallback
    dom = domain_of(url)
    return f"{title} — {dom}"

# ------------------------------------------------------------------------------
# Ingestion & summarization storage
# ------------------------------------------------------------------------------

def parse_feed(url: str) -> List[Dict[str, Any]]:
    log.info("parsed feed: %s", url)
    parsed = feedparser.parse(url)
    entries = []
    for e in parsed.entries:
        title = e.get("title", "").strip()
        link = e.get("link", "").strip()
        lang = (parsed.feed.get("language") or e.get("language") or "en").lower()
        published = None
        # Try multiple published fields
        for k in ("published_parsed", "updated_parsed"):
            if e.get(k):
                try:
                    published = datetime.fromtimestamp(time.mktime(e[k]), tz=timezone.utc)
                    break
                except Exception:
                    pass
        entries.append({"title": title, "url": link, "language": lang, "published_at": published})
    log.info("feed entries: %d", len(entries))
    return entries

def ingest_feed(feed_id: int, url: str) -> Tuple[int, int]:
    """
    Returns: (inserted_or_updated, skipped)
    """
    entries = parse_feed(url)
    inserted = 0
    skipped = 0
    with get_conn() as conn, conn.cursor() as cur:
        for it in entries:
            try:
                cur.execute(
                    """
                    INSERT INTO found_url (url, title, feed_id, language, published_at, found_at, last_seen, seen_count)
                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW(), 1)
                    ON CONFLICT (url)
                    DO UPDATE SET
                        title = COALESCE(EXCLUDED.title, found_url.title),
                        feed_id = COALESCE(EXCLUDED.feed_id, found_url.feed_id),
                        language = COALESCE(EXCLUDED.language, found_url.language),
                        published_at = COALESCE(EXCLUDED.published_at, found_url.published_at),
                        last_seen = NOW(),
                        seen_count = found_url.seen_count + 1
                    """,
                    (it["url"], it["title"], feed_id, it["language"], it["published_at"]),
                )
                inserted += 1
            except Exception as e:
                log.warning("ingest error for feed %s: %s", url, str(e))
                skipped += 1
    return inserted, skipped

def prune_old_found_urls():
    with get_conn() as conn, conn.cursor() as cur:
        # Each feed can keep links for its own persist_days; we conservatively use the minimum per url.
        cur.execute(
            """
            DELETE FROM found_url f
            WHERE f.found_at < NOW() - INTERVAL '1 day' * (
                SELECT MIN(persist_days) FROM source_feed s WHERE s.id = f.feed_id
            )
            """
        )

def ensure_summaries(urls: List[Tuple[str, str]]) -> int:
    """
    Given list of (url, title), create summaries for those that don't have one yet.
    Returns number summarized.
    """
    if not urls:
        return 0
    with get_conn() as conn, conn.cursor() as cur:
        summarized = 0
        for url, title in urls:
            cur.execute("SELECT 1 FROM summaries WHERE url = %s", (url,))
            if cur.fetchone():
                continue
            text = summarize_item(title, url)
            cur.execute(
                "INSERT INTO summaries (url, summary, model, tokens, created_at) VALUES (%s, %s, %s, %s, NOW())",
                (url, text, OPENAI_MODEL, None),
            )
            summarized += 1
        return summarized

# ------------------------------------------------------------------------------
# Email
# ------------------------------------------------------------------------------

def send_mailgun(subject: str, html: str, to_list: List[str]) -> bool:
    if not (MAILGUN_DOMAIN and MAILGUN_API_KEY):
        log.warning("Mailgun not configured; skipping email send.")
        return False
    mg_url = f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages"
    auth = ("api", MAILGUN_API_KEY)
    data = {
        "from": FROM_EMAIL,
        "to": to_list,
        "subject": subject,
        "html": html,
    }
    r = httpx.post(mg_url, auth=auth, data=data, timeout=30.0)
    ok = r.status_code < 400
    if not ok:
        log.warning("Mailgun send failed: %s %s", r.status_code, r.text)
    return ok

def build_digest_html(window_minutes: int, rows: List[Dict[str, Any]], summarized_count: int) -> str:
    window_end = now()
    window_start = window_end - timedelta(minutes=window_minutes)
    parts = []
    parts.append(f"<h2>QuantBrief Daily — {window_end.strftime('%Y-%m-%d %H:%M')} UTC</h2>")
    parts.append(f"<p><em>Window:</em> last {window_minutes} minutes ending {window_end.isoformat()}</p>")
    parts.append("<ul>")
    for r in rows:
        title = (r.get("title") or "").strip() or "[no title]"
        url = r["url"]
        summary = (r.get("summary") or "").strip()
        dom = domain_of(url)
        # Always show linked source; summary if present
        line = f"<li><a href='{url}' target='_blank' rel='noopener'>{title}</a>"
        if summary:
            line += f"<br/><small>{summary}</small>"
        line += f"<br/><small><em>{dom}</em></small></li>"
        parts.append(line)
    parts.append("</ul>")
    parts.append("<p><small>Sources are links only. We don’t republish paywalled content.</small></p>")
    return "\n".join(parts)

# ------------------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------------------

app = FastAPI(title=APP_NAME)

@app.get("/")
def root():
    return {"ok": True, "service": APP_NAME, "version": "1.0"}

@app.post("/admin/init")
def admin_init(request: Request):
    require_admin(request)
    log.info("init: query=%s days=%s excl=%s allow=%s", QB_QUERY, QB_DAYS, QB_EXCLUDE, QB_ALLOW)
    # Ensure schema/migrations
    with get_conn() as conn:
        exec_sql_batch(conn, SCHEMA_SQL)

    # Build + upsert one canonical Google News feed
    feed_url = build_google_news_url(QB_QUERY, QB_DAYS, QB_EXCLUDE, QB_ALLOW)
    feed_id = upsert_feed(kind="google-news", url=feed_url, name="Google News", persist_days=max(30, QB_DAYS + 7))

    return JSONResponse({"initialized": True, "feed_id": feed_id, "feed_url": feed_url})

@app.post("/cron/ingest")
def cron_ingest(request: Request, minutes: int = Query(60*24*7, ge=5, le=60*24*30)):
    """
    Ingest all active feeds (nothing to do with minutes; minutes is only used later for digest symmetry)
    """
    require_admin(request)
    inserted_total = 0
    skipped_total = 0
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, url, persist_days FROM source_feed WHERE active = TRUE")
        feeds = cur.fetchall()

    for f in feeds:
        ins, sk = ingest_feed(feed_id=f["id"], url=f["url"])
        inserted_total += ins
        skipped_total += sk

    prune_old_found_urls()
    return JSONResponse({"ok": True, "inserted_or_updated": inserted_total, "skipped": skipped_total})

@app.post("/cron/digest")
def cron_digest(request: Request, minutes: int = Query(60*24*7, ge=5, le=60*24*30)):
    require_admin(request)

    since = now() - timedelta(minutes=minutes)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                f.url,
                f.title,
                s.summary
            FROM found_url f
            LEFT JOIN summaries s ON s.url = f.url
            WHERE f.found_at >= %s
            ORDER BY COALESCE(f.published_at, f.found_at) DESC
            LIMIT 200
            """,
            (since,),
        )
        rows = cur.fetchall()

    # Make summaries for any missing
    to_sum = [(r["url"], r["title"] or "") for r in rows if not (r.get("summary") or "").strip()]
    summarized = ensure_summaries(to_sum)

    # Re-read rows to include fresh summaries
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                f.url,
                f.title,
                s.summary
            FROM found_url f
            LEFT JOIN summaries s ON s.url = f.url
            WHERE f.found_at >= %s
            ORDER BY COALESCE(f.published_at, f.found_at) DESC
            LIMIT 200
            """,
            (since,),
        )
        rows = cur.fetchall()

    subject = f"QuantBrief Daily — {now().strftime('%Y-%m-%d %H:%M')}"

    html = build_digest_html(minutes, rows, summarized)
    sent_ok = send_mailgun(subject, html, TO_EMAILS)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO digest_log (sent_to, items, summarized, ok, created_at) VALUES (%s, %s, %s, %s, NOW())",
            (",".join(TO_EMAILS), len(rows), summarized, sent_ok),
        )

    log.info("sent_to=%s items=%d summarized=%d sent_ok=%s", ",".join(TO_EMAILS), len(rows), summarized, sent_ok)
    return JSONResponse({"ok": True, "sent_to": TO_EMAILS, "items": len(rows), "summarized": summarized, "sent_ok": sent_ok})

# ------------------------------------------------------------------------------
# Local dev entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
