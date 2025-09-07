import os
import re
import json
import time
import math
import html
import httpx
import logging
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import psycopg
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
from xml.etree import ElementTree as ET

# -----------------------------------------------------------------------------
# Config / Env
# -----------------------------------------------------------------------------

APP_NAME = os.getenv("APP_NAME", "QuantBrief Daily")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE", "0.3")  # may be rejected by some models

MAILGUN_DOMAIN = os.getenv("MAILGUN_DOMAIN", "")
MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY", "")
MAIL_FROM = os.getenv("MAIL_FROM", "QuantBrief Daily <daily@example.com>")
MAIL_TO = os.getenv("MAIL_TO", "you@example.com")  # comma-separated

# Render/Heroku style: platform sets $PORT
PORT = int(os.getenv("PORT", "10000"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("quantbrief")

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------

app = FastAPI(title=APP_NAME)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def auth_required(req: Request):
    sent = req.headers.get("x-admin-token", "")
    if not ADMIN_TOKEN or sent != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def db() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, autocommit=True)


def exec_sql_batch(conn: psycopg.Connection, sql: str):
    # Split on semicolons that end a statement; ignore whitespace-only chunks.
    for stmt in [s.strip() for s in sql.split(";")]:
        if stmt:
            conn.execute(stmt)


def to_int(s: Optional[str], default: int) -> int:
    try:
        return int(s) if s is not None else default
    except Exception:
        return default


def list_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def minute_window(minutes: int) -> (datetime, datetime):
    end = _now()
    start = end - timedelta(minutes=minutes)
    return (start, end)


# -----------------------------------------------------------------------------
# DB schema (includes persist_days so your earlier error doesn't recur)
# -----------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS source_feed (
  id            BIGSERIAL PRIMARY KEY,
  kind          TEXT NOT NULL,            -- e.g., 'google-news'
  url           TEXT NOT NULL UNIQUE,
  name          TEXT,
  active        BOOLEAN NOT NULL DEFAULT TRUE,
  persist_days  INTEGER NOT NULL DEFAULT 30,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS found_url (
  id            BIGSERIAL PRIMARY KEY,
  url           TEXT NOT NULL UNIQUE,
  title         TEXT,
  feed_id       BIGINT REFERENCES source_feed(id) ON DELETE SET NULL,
  first_seen    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_seen     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  raw_html      TEXT,
  source_kind   TEXT,      -- redundant but handy in queries
  status        TEXT,      -- fetch/summarize status
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_found_url_last_seen ON found_url(last_seen DESC);

CREATE TABLE IF NOT EXISTS summaries (
  id            BIGSERIAL PRIMARY KEY,
  url           TEXT NOT NULL REFERENCES found_url(url) ON DELETE CASCADE,
  model         TEXT,
  summary       TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_summaries_url ON summaries(url);

CREATE TABLE IF NOT EXISTS sent_digest (
  id             BIGSERIAL PRIMARY KEY,
  window_start   TIMESTAMPTZ NOT NULL,
  window_end     TIMESTAMPTZ NOT NULL,
  recipients     TEXT NOT NULL,
  items_count    INTEGER NOT NULL,
  summarized     INTEGER NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


# -----------------------------------------------------------------------------
# Google News feed builder and parser
# -----------------------------------------------------------------------------

def build_google_news_url(query: str, days: int, excl: List[str], allow: List[str]) -> str:
    # Google News supports "when:7d", site: filters (+/-)
    terms = [f"({query})"] if query else []
    for a in allow:
        terms.append(f"site:{a}")
    for e in excl:
        terms.append(f"-site:{e}")
    terms.append(f"when:{days}d")
    q = " ".join(terms)
    return (
        "https://news.google.com/rss/search"
        f"?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US%3Aen"
    )


def parse_rss_entries(xml_text: str) -> List[Dict[str, str]]:
    # Minimal RSS parser for Google News
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    ns = {}
    # Build entries
    out: List[Dict[str, str]] = []
    for item in root.findall(".//item", ns):
        title_e = item.find("title")
        link_e = item.find("link")
        desc_e = item.find("description")
        title = title_e.text if title_e is not None else ""
        link = link_e.text if link_e is not None else ""
        desc = desc_e.text if desc_e is not None else ""
        out.append({"title": html.unescape(title or "").strip(),
                    "link": (link or "").strip(),
                    "desc": html.unescape(desc or "").strip()})
    return out


async def http_get(url: str, timeout: float = 30.0) -> httpx.Response:
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        return await client.get(url)


# -----------------------------------------------------------------------------
# OpenAI summarize with temperature fallback
# -----------------------------------------------------------------------------

def _chat_complete(messages: List[Dict[str, str]], max_tokens: int = 400) -> str:
    """
    Calls OpenAI Chat Completions, retrying without 'temperature' if the model rejects it.
    """
    if not OPENAI_API_KEY:
        return ""

    # parse optional temperature
    try:
        temp_val = float(OPENAI_TEMPERATURE) if OPENAI_TEMPERATURE.strip() != "" else None
    except Exception:
        temp_val = None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    def _req(include_temp: bool) -> str:
        payload: Dict[str, Any] = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # Only attach temperature if non-default AND caller asked us to
        if include_temp and temp_val is not None and temp_val != 1:
            payload["temperature"] = temp_val

        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        j = r.json()
        return (j["choices"][0]["message"]["content"] or "").strip()

    try:
        return _req(include_temp=True)
    except RuntimeError as e:
        msg = str(e)
        if "unsupported_value" in msg or "does not support" in msg:
            log.warning("OpenAI: model rejects temperature; retrying without it.")
            return _req(include_temp=False)
        raise


def summarize_text(text: str) -> str:
    if not text:
        return ""
    prompt = (
        "You are a terse financial/news summarizer. "
        "Summarize the key facts in 2–3 bullets. "
        "Avoid hype; include tickers only if explicitly present."
    )
    try:
        return _chat_complete(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text[:8000]},
            ],
            max_tokens=220,
        )
    except Exception as e:
        log.warning(f"OpenAI summarize error: {e}")
        return ""


# -----------------------------------------------------------------------------
# Mailgun email
# -----------------------------------------------------------------------------

def send_mail(subject: str, text_body: str, html_body: Optional[str] = None) -> bool:
    if not (MAILGUN_DOMAIN and MAILGUN_API_KEY and MAIL_FROM and MAIL_TO):
        log.warning("Mailgun not configured; skipping email.")
        return False

    to_list = list_csv(MAIL_TO)
    data = {
        "from": MAIL_FROM,
        "to": to_list,
        "subject": subject,
        "text": text_body,
    }
    if html_body:
        data["html"] = html_body

    try:
        r = httpx.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data=data,
            timeout=30.0,
        )
        if r.status_code >= 400:
            log.error(f"Mailgun send failed: {r.status_code} {r.text}")
            return False
        return True
    except Exception as e:
        log.error(f"Mailgun exception: {e}")
        return False


# -----------------------------------------------------------------------------
# Admin payload and upsert feed
# -----------------------------------------------------------------------------

class AdminInitIn(BaseModel):
    query: str = "(Talen Energy) OR TLN"
    days: int = 7
    exclude_sites: List[str] = ["newser.com", "marketbeat.com", "www.newser.com", "www.marketbeat.com"]
    allow_sites: List[str] = []


def upsert_feed(kind: str, url: str, name: str, persist_days: int = 30):
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO source_feed (kind, url, name, active, persist_days)
            VALUES (%s, %s, %s, TRUE, %s)
            ON CONFLICT (url) DO UPDATE
               SET name = EXCLUDED.name,
                   active = TRUE,
                   persist_days = EXCLUDED.persist_days
            """,
            (kind, url, name, persist_days),
        )


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    return PlainTextResponse("QuantBrief Daily is up. POST /admin/init, /cron/ingest, /cron/digest")


@app.post("/admin/init")
def admin_init(payload: AdminInitIn, request: Request):
    auth_required(request)
    days = max(1, int(payload.days or 7))
    excl = payload.exclude_sites or []
    allow = payload.allow_sites or []
    log.info(f"init: query={payload.query} days={days} excl={excl} allow={allow}")

    feed_url = build_google_news_url(payload.query, days, excl, allow)
    with db() as conn:
        exec_sql_batch(conn, SCHEMA_SQL)
    upsert_feed(kind="google-news", url=feed_url, name="Google News", persist_days=max(30, days + 7))
    log.info(f"parsed feed: {feed_url}")
    return JSONResponse({"ok": True, "feed": feed_url})


@app.post("/cron/ingest")
async def cron_ingest(request: Request, minutes: Optional[int] = 10080):
    auth_required(request)
    minutes = to_int(str(minutes), 10080)
    # get feeds
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, kind, url, persist_days FROM source_feed WHERE active = TRUE")
        feeds = cur.fetchall()

    inserted = 0
    for fid, kind, url, persist_days in feeds:
        try:
            if kind == "google-news":
                r = await http_get(url)
                entries = parse_rss_entries(r.text)
                log.info(f"feed entries: {len(entries)}")
                with db() as conn, conn.cursor() as cur:
                    for e in entries:
                        link = e.get("link", "").strip()
                        title = e.get("title", "").strip()
                        if not link:
                            continue
                        cur.execute(
                            """
                            INSERT INTO found_url (url, title, feed_id, last_seen, source_kind, status)
                            VALUES (%s, %s, %s, NOW(), %s, %s)
                            ON CONFLICT (url) DO UPDATE
                               SET last_seen = EXCLUDED.last_seen,
                                   title = COALESCE(EXCLUDED.title, found_url.title)
                            """,
                            (link, title, fid, kind, "seen"),
                        )
                        inserted += 1
                # cleanup old
                with db() as conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        DELETE FROM found_url
                        WHERE last_seen < NOW() - (%s || ' days')::INTERVAL
                        """,
                        (str(int(persist_days)),),
                    )
        except Exception as e:
            log.warning(f"ingest error for feed {url}: {e}")

    return JSONResponse({"ok": True, "inserted_or_updated": inserted})


@app.post("/cron/digest")
def cron_digest(request: Request, minutes: Optional[int] = 10080):
    auth_required(request)
    minutes = to_int(str(minutes), 10080)
    start, end = minute_window(minutes)

    # get items to summarize
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.url, f.title
            FROM found_url f
            LEFT JOIN summaries s ON s.url = f.url
            WHERE f.last_seen BETWEEN %s AND %s
              AND s.url IS NULL
            ORDER BY f.last_seen DESC
            LIMIT 200
            """,
            (start, end),
        )
        to_summarize = cur.fetchall()

    # summarize
    summarized = 0
    blocks: List[str] = []
    with db() as conn, conn.cursor() as cur:
        for url, title in to_summarize:
            text = f"{title}\n\n{url}"
            summ = summarize_text(text)
            if summ:
                cur.execute(
                    "INSERT INTO summaries (url, model, summary) VALUES (%s, %s, %s)",
                    (url, OPENAI_MODEL, summ),
                )
                summarized += 1

    # pull all items (even those summarized earlier)
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.url, f.title, COALESCE(s.summary, '') AS summary
            FROM found_url f
            LEFT JOIN summaries s ON s.url = f.url
            WHERE f.last_seen BETWEEN %s AND %s
            ORDER BY f.last_seen DESC
            LIMIT 300
            """,
            (start, end),
        )
        rows = cur.fetchall()

    items_count = len(rows)
    if items_count == 0:
        return JSONResponse({"ok": True, "sent": False, "reason": "no items in window"})

    # Build email body
    # If no summary for an item, show "(no summary)" to mirror your logs
    lines_txt: List[str] = []
    lines_html: List[str] = []
    for url, title, summ in rows:
        title = title or url
        s = summ.strip() if summ else "(no summary)"
        lines_txt.append(f"{title}\n  {url}\n{s}\n")
        item_html = (
            f"<p><b>{html.escape(title)}</b><br>"
            f'<a href="{html.escape(url)}">{html.escape(url)}</a><br>'
            f"{'<br>'.join(html.escape(x) for x in s.splitlines())}</p>"
        )
        lines_html.append(item_html)

    subject = f"{APP_NAME} — {end:%Y-%m-%d %H:%M} Window: last {minutes} minutes ending {end.isoformat(timespec='seconds')}"
    text_body = (
        f"{APP_NAME} — {end:%Y-%m-%d %H:%M} "
        f"Window: last {minutes} minutes ending {end.isoformat(timespec='seconds')}\n\n"
        + "\n".join(lines_txt)
        + "\nSources are links only. We don’t republish paywalled content."
    )
    html_body = (
        f"<h3>{html.escape(APP_NAME)} — {end:%Y-%m-%d %H:%M}</h3>"
        f"<p><i>Window: last {minutes} minutes ending {html.escape(end.isoformat(timespec='seconds'))}</i></p>"
        + "\n".join(lines_html)
        + "<p><i>Sources are links only. We don’t republish paywalled content.</i></p>"
    )

    sent_ok = send_mail(subject, text_body, html_body)

    # record send
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sent_digest (window_start, window_end, recipients, items_count, summarized)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (start, end, MAIL_TO, items_count, summarized),
        )

    log.info(f"sent_to={MAIL_TO} items={items_count} summarized={summarized} sent_ok={sent_ok}")
    return JSONResponse({"ok": True, "sent": bool(sent_ok), "to": MAIL_TO, "items": items_count, "summarized": summarized})


# -----------------------------------------------------------------------------
# Uvicorn entrypoint (Render runs: uvicorn app:app --host 0.0.0.0 --port $PORT)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
