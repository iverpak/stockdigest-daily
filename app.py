import os
import html
import httpx
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import psycopg
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
from xml.etree import ElementTree as ET

APP_NAME = os.getenv("APP_NAME", "QuantBrief Daily")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE", "0.3")  # will be dropped if model rejects it

MAILGUN_DOMAIN = os.getenv("MAILGUN_DOMAIN", "")
MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY", "")
MAIL_FROM = os.getenv("MAIL_FROM", "QuantBrief Daily <daily@example.com>")
MAIL_TO = os.getenv("MAIL_TO", "you@example.com")  # comma-separated

PORT = int(os.getenv("PORT", "10000"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("quantbrief")

app = FastAPI(title=APP_NAME)

# ---------- helpers

def _now() -> datetime:
    return datetime.now(timezone.utc)

def minute_window(minutes: int) -> Tuple[datetime, datetime]:
    end = _now()
    start = end - timedelta(minutes=minutes)
    return start, end

def to_int(s: Optional[str], default: int) -> int:
    try:
        return int(s) if s is not None else default
    except Exception:
        return default

def list_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def auth_required(req: Request):
    sent = req.headers.get("x-admin-token", "")
    if not ADMIN_TOKEN or sent != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

def db() -> psycopg.Connection:
    # autocommit True so each statement stands alone (avoids one failure aborting all)
    return psycopg.connect(DATABASE_URL, autocommit=True)

def exec_many(conn: psycopg.Connection, statements: List[str]):
    for stmt in statements:
        s = stmt.strip()
        if s:
            conn.execute(s)

# ---------- migrations (safe upgrades for already-existing tables)

MIGRATIONS: List[str] = [
    # source_feed
    """CREATE TABLE IF NOT EXISTS source_feed (
        id BIGSERIAL PRIMARY KEY,
        kind TEXT NOT NULL,
        url  TEXT NOT NULL UNIQUE
    )""",
    """ALTER TABLE source_feed ADD COLUMN IF NOT EXISTS name TEXT""",
    """ALTER TABLE source_feed ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE""",
    """ALTER TABLE source_feed ADD COLUMN IF NOT EXISTS persist_days INTEGER NOT NULL DEFAULT 30""",
    """ALTER TABLE source_feed ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()""",

    # found_url
    """CREATE TABLE IF NOT EXISTS found_url (
        id BIGSERIAL PRIMARY KEY,
        url TEXT NOT NULL UNIQUE
    )""",
    """ALTER TABLE found_url ADD COLUMN IF NOT EXISTS title TEXT""",
    """ALTER TABLE found_url ADD COLUMN IF NOT EXISTS feed_id BIGINT REFERENCES source_feed(id) ON DELETE SET NULL""",
    """ALTER TABLE found_url ADD COLUMN IF NOT EXISTS first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW()""",
    """ALTER TABLE found_url ADD COLUMN IF NOT EXISTS last_seen  TIMESTAMPTZ NOT NULL DEFAULT NOW()""",
    """ALTER TABLE found_url ADD COLUMN IF NOT EXISTS raw_html TEXT""",
    """ALTER TABLE found_url ADD COLUMN IF NOT EXISTS source_kind TEXT""",
    """ALTER TABLE found_url ADD COLUMN IF NOT EXISTS status TEXT""",
    """ALTER TABLE found_url ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()""",
    """CREATE INDEX IF NOT EXISTS ix_found_url_last_seen ON found_url(last_seen)""",

    # summaries
    """CREATE TABLE IF NOT EXISTS summaries (
        id BIGSERIAL PRIMARY KEY,
        url TEXT NOT NULL REFERENCES found_url(url) ON DELETE CASCADE,
        model TEXT,
        summary TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
    """CREATE INDEX IF NOT EXISTS ix_summaries_url ON summaries(url)""",

    # sent_digest
    """CREATE TABLE IF NOT EXISTS sent_digest (
        id BIGSERIAL PRIMARY KEY,
        window_start TIMESTAMPTZ NOT NULL,
        window_end   TIMESTAMPTZ NOT NULL,
        recipients   TEXT NOT NULL,
        items_count  INTEGER NOT NULL,
        summarized   INTEGER NOT NULL,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )""",
]

def ensure_schema():
    try:
        with db() as conn:
            exec_many(conn, MIGRATIONS)
    except Exception as e:
        log.error(f"schema migration failed: {e}")
        raise

@app.on_event("startup")
def on_startup():
    ensure_schema()

# ---------- feeds

def build_google_news_url(query: str, days: int, excl: List[str], allow: List[str]) -> str:
    terms = [f"({query})"] if query else []
    for a in allow:
        terms.append(f"site:{a}")
    for e in excl:
        terms.append(f"-site:{e}")
    terms.append(f"when:{days}d")
    q = " ".join(terms)
    return "https://news.google.com/rss/search?q=" + quote_plus(q) + "&hl=en-US&gl=US&ceid=US%3Aen"

def parse_rss_entries(xml_text: str) -> List[Dict[str, str]]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    out: List[Dict[str, str]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link  = (item.findtext("link") or "").strip()
        desc  = (item.findtext("description") or "").strip()
        out.append({"title": html.unescape(title), "link": link, "desc": html.unescape(desc)})
    return out

async def http_get(url: str, timeout: float = 30.0) -> httpx.Response:
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        return await client.get(url)

# ---------- OpenAI (temperature fallback)

def _chat_complete(messages: List[Dict[str, str]], max_tokens: int = 400) -> str:
    if not OPENAI_API_KEY:
        return ""

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    # Only send temperature if it's *not* the default (1) — and drop it if model rejects it.
    temp = None
    try:
        t = float(OPENAI_TEMPERATURE) if OPENAI_TEMPERATURE != "" else None
        if t is not None and t != 1:
            temp = t
    except Exception:
        pass

    def request_with(tokens_param: str | None, include_temp: bool) -> str:
        payload: Dict[str, Any] = {
            "model": OPENAI_MODEL,
            "messages": messages,
        }
        if tokens_param:
            payload[tokens_param] = max_tokens
        if include_temp and temp is not None:
            payload["temperature"] = temp

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

    # 1) Try: max_tokens (+ maybe temperature)
    try:
        return request_with(tokens_param="max_tokens", include_temp=True)
    except RuntimeError as e1:
        es1 = str(e1)

        # If temperature is unsupported, try again without it (still with max_tokens)
        if "unsupported_value" in es1 and "temperature" in es1:
            try:
                return request_with(tokens_param="max_tokens", include_temp=False)
            except RuntimeError as e1b:
                es1 = str(e1b)  # fall through to next block with updated error

        # If max_tokens is unsupported, switch to max_completion_tokens
        if "unsupported_parameter" in es1 and "max_tokens" in es1:
            try:
                return request_with(tokens_param="max_completion_tokens", include_temp=True)
            except RuntimeError as e2:
                es2 = str(e2)
                # And if temperature still offends here, drop it too
                if "unsupported_value" in es2 and "temperature" in es2:
                    return request_with(tokens_param="max_completion_tokens", include_temp=False)
                raise

        # Otherwise propagate original error
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
            [{"role": "system", "content": prompt}, {"role": "user", "content": text[:8000]}],
            max_tokens=220,
        )
    except Exception as e:
        log.warning(f"OpenAI summarize error: {e}")
        return ""

# ---------- Mail

def send_mail(subject: str, text_body: str, html_body: Optional[str] = None) -> bool:
    if not (MAILGUN_DOMAIN and MAILGUN_API_KEY and MAIL_FROM and MAIL_TO):
        log.warning("Mailgun not fully configured; skipping email send.")
        return False
    try:
        r = httpx.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={
                "from": MAIL_FROM,
                "to": list_csv(MAIL_TO),
                "subject": subject,
                "text": text_body,
                **({"html": html_body} if html_body else {}),
            },
            timeout=30.0,
        )
        if r.status_code >= 400:
            log.error(f"Mailgun send failed: {r.status_code} {r.text}")
            return False
        return True
    except Exception as e:
        log.error(f"Mailgun exception: {e}")
        return False

# ---------- models & db helpers

class AdminInitIn (BaseModel):
    query: str = "(Talen Energy) OR TLN"
    days: int = 7
    exclude_sites: List[str] = ["newser.com", "www.newser.com", "marketbeat.com", "www.marketbeat.com"]
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

# ---------- routes

@app.get("/")
def root():
    return PlainTextResponse("OK — QuantBrief Daily")

@app.get("/healthz")
def healthz():
    return PlainTextResponse("healthy")

@app.post("/admin/init")
def admin_init(payload: AdminInitIn, request: Request):
    auth_required(request)
    ensure_schema()  # make sure upgrades applied before writing
    days = max(1, int(payload.days or 7))
    excl = payload.exclude_sites or []
    allow = payload.allow_sites or []
    log.info(f"init: query={payload.query} days={days} excl={excl} allow={allow}")
    feed_url = build_google_news_url(payload.query, days, excl, allow)
    upsert_feed("google-news", feed_url, "Google News", persist_days=max(30, days + 7))
    log.info(f"parsed feed: {feed_url}")
    return JSONResponse({"ok": True, "feed": feed_url})

@app.post("/cron/ingest")
async def cron_ingest(request: Request, minutes: Optional[int] = 10080):
    auth_required(request)
    ensure_schema()
    minutes = to_int(str(minutes), 10080)

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
                            (link, title or link, fid, kind, "seen"),
                        )
                        inserted += 1
                # cleanup old
                with db() as conn, conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM found_url WHERE last_seen < NOW() - (%s || ' days')::INTERVAL",
                        (str(int(persist_days)),),
                    )
        except Exception as e:
            log.warning(f"ingest error for feed {url}: {e}")

    return JSONResponse({"ok": True, "inserted_or_updated": inserted})

@app.post("/cron/digest")
def cron_digest(request: Request, minutes: Optional[int] = 10080):
    auth_required(request)
    ensure_schema()
    minutes = to_int(str(minutes), 10080)
    start, end = minute_window(minutes)

    # summarize missing
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

    summarized = 0
    with db() as conn, conn.cursor() as cur:
        for url, title in to_summarize:
            content = f"{title or url}\n\n{url}"
            summ = summarize_text(content)
            if summ:
                cur.execute(
                    "INSERT INTO summaries (url, model, summary) VALUES (%s, %s, %s)",
                    (url, OPENAI_MODEL, summ),
                )
                summarized += 1

    # collect rows for the email
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

    if not rows:
        return JSONResponse({"ok": True, "sent": False, "reason": "no items in window"})

    # build email
    lines_txt: List[str] = []
    lines_html: List[str] = []
    for url, title, summ in rows:
        ttl = title or url
        s = (summ or "").strip() or "(no summary)"
        lines_txt.append(f"{ttl}\n  {url}\n{s}\n")
        item_html = (
            f"<p><b>{html.escape(ttl)}</b><br>"
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

    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sent_digest (window_start, window_end, recipients, items_count, summarized) VALUES (%s, %s, %s, %s, %s)",
            (start, end, MAIL_TO, len(rows), summarized),
        )

    log.info(f"sent_to={MAIL_TO} items={len(rows)} summarized={summarized} sent_ok={sent_ok}")
    return JSONResponse({"ok": True, "sent": bool(sent_ok), "to": MAIL_TO, "items": len(rows), "summarized": summarized})

# ---------- entry

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
