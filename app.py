import os
import sys
import json
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict, Any
from urllib.parse import urlparse, parse_qs, unquote, quote_plus

import requests
import feedparser
from bs4 import BeautifulSoup

from fastapi import FastAPI, Request, Header, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from jinja2 import Template

from dateutil import tz

from sqlalchemy import (
    create_engine, text, MetaData, Table, Column,
    Integer, String, Text, DateTime, Boolean
)
from sqlalchemy.exc import OperationalError, IntegrityError

# ----------- Configuration (env) -----------
ADMIN_TOKEN         = os.getenv("ADMIN_TOKEN", "")
TIMEZONE            = os.getenv("TIMEZONE", "America/Toronto")

DATABASE_URL        = os.getenv("DATABASE_URL", "")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL        = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_MAX_TOKENS   = os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "2000")

MAILGUN_API_KEY     = os.getenv("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN      = os.getenv("MAILGUN_DOMAIN", "")
MAILGUN_FROM        = os.getenv("MAILGUN_FROM", "QuantBrief Daily <daily@mg.example.com>")

ADMIN_EMAIL         = os.getenv("ADMIN_EMAIL", "")  # optional: seed recipient
DEFAULT_SYMBOL      = os.getenv("DEFAULT_SYMBOL", "TLN")
DEFAULT_COMPANY     = os.getenv("DEFAULT_COMPANY", "Talen Energy")

# ----------- OpenAI client (lazy) ----------
_openai_client = None
def _get_openai():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(f"openai import failed: {e}")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

def _safe_max_tokens() -> int:
    raw = (OPENAI_MAX_TOKENS or "").strip()
    try:
        n = int(raw)
    except Exception:
        n = 2000
    if n < 256:
        n = 2000
    if n > 8192:
        n = 8192
    return n

# ----------- DB setup ----------------------
if not DATABASE_URL:
    print("ERROR: DATABASE_URL is not set", file=sys.stderr)
metadata = MetaData()
engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

watchlist = Table(
    "watchlist", metadata,
    Column("id", Integer, primary_key=True),
    Column("symbol", String(32), nullable=False, unique=True),
    Column("company", String(200), nullable=False),
    Column("notes", Text, nullable=True),
)

recipients = Table(
    "recipients", metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String(320), nullable=False, unique=True),
    Column("active", Boolean, nullable=False, server_default=text("true"))
)

items = Table(
    "items", metadata,
    Column("id", Integer, primary_key=True),
    Column("company", String(200), nullable=False),
    Column("symbol", String(32), nullable=False),
    Column("title", Text, nullable=False),
    Column("url", Text, nullable=False, unique=True),
    Column("domain", String(200), nullable=False),
    Column("source_name", String(200), nullable=True),
    Column("published_at", DateTime(timezone=True), nullable=True),
    Column("inserted_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("have_text", Boolean, nullable=False, server_default=text("false")),
    Column("raw_text", Text, nullable=True),
    Column("summary", Text, nullable=True),
)

digests = Table(
    "digests", metadata,
    Column("id", Integer, primary_key=True),
    Column("to_email", String(320), nullable=False),
    Column("window_start", DateTime(timezone=True), nullable=False),
    Column("window_end", DateTime(timezone=True), nullable=False),
    Column("sent_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("items_count", Integer, nullable=False, server_default=text("0")),
)

def create_tables():
    metadata.create_all(engine)

# ----------- Utilities ---------------------
def tz_now():
    return datetime.now(timezone.utc)

def to_tz(dt: datetime, tz_name: str) -> datetime:
    try:
        tzinfo = tz.gettz(tz_name)
        return dt.astimezone(tzinfo)
    except Exception:
        return dt

BAD_DOMAINS = {
    "lh3.googleusercontent.com",
    "lh5.googleusercontent.com",
    "gstatic.com",
}
EXCLUDE_DOMAINS = BAD_DOMAINS | {"news.google.com", "news.url.google.com"}
def is_bad_domain(u: str) -> bool:
    try:
        host = (urlparse(u).hostname or "").lower()
        return host in BAD_DOMAINS
    except Exception:
        return True

def resolve_google_news_url(u: str) -> str:
    """
    Unwrap Google News redirect to the original publisher URL when possible.
    Handles links like:
      https://news.google.com/articles/...
      https://news.google.com/rss/articles/...
      https://news.url.google.com/...
      https://news.google.com/__i/rss/rd/articles/...?url=<encoded>
    """
    try:
        p = urlparse(u)
        host = (p.hostname or "").lower()
        if host.endswith("news.google.com") or host.endswith("news.url.google.com"):
            qs = parse_qs(p.query)
            for k in ("url", "u"):
                if k in qs and qs[k]:
                    return unquote(qs[k][0])
        return u
    except Exception:
        return u

def domain_of(u: str) -> str:
    try:
        host = (urlparse(u).hostname or "").lower()
        if host and host.startswith("www."):
            host = host[4:]
        return host or ""
    except Exception:
        return ""

def fetch_google_news_rss(queries: List[str]) -> List[Dict[str, Any]]:
    """
    Use Google News RSS for each query; return list of {title,url,source_name,published_at}
    """
    out = []
    headers = {"User-Agent": "QuantBriefBot/1.0"}
    for q in queries:
        rss_url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"
        try:
            resp = requests.get(rss_url, timeout=15, headers=headers)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            for e in feed.entries:
                raw_link = e.get("link") or ""
                link = resolve_google_news_url(raw_link)
                if not link or is_bad_domain(link):
                    continue
                pub = None
                if "published_parsed" in e and e.published_parsed:
                    pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                src_name = (e.get("source", {}) or {}).get("title") or e.get("source_title") or "Google News"
                out.append({
                    "title": e.get("title") or "(no title)",
                    "url": link,
                    "source_name": src_name,
                    "published_at": pub,
                })
        except Exception:
            # ignore this query if it fails
            continue
    return out

def fetch_ir_feeds(symbol: str, company: str) -> List[Dict[str, Any]]:
    """
    Minimal IR RSS placeholder. Extend by mapping symbol->IR RSS.
    """
    # Example: no default IR URL for TLN here; return empty list for now.
    return []

def extract_main_text(url: str, timeout_sec: int = 15) -> Optional[str]:
    """
    Fetch the page and extract a crude main text. Not perfect, but good enough
    to give the LLM substance. Skips when content-type isn't HTML.
    """
    try:
        resp = requests.get(url, timeout=timeout_sec, headers={"User-Agent": "QuantBriefBot/1.0"})
        if resp.status_code >= 400:
            return None
        ctype = resp.headers.get("Content-Type", "")
        if "html" not in ctype.lower():
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style/nav/aside/header/footer
        for tag in soup(["script","style","nav","aside","header","footer","noscript"]):
            tag.decompose()
        # Heuristic: prefer article > p, fallback to all paragraphs
        article = soup.find("article")
        if article:
            text = " ".join([p.get_text(" ", strip=True) for p in article.find_all("p")])
        else:
            text = " ".join([p.get_text(" ", strip=True) for p in soup.find_all("p")])
        text = " ".join(text.split())
        return text[:12000] or None  # trim to keep requests smaller later
    except Exception:
        return None

# ----------- LLM Summarization -------------
DAILY_INSTRUCTIONS = (
    "You are a hedge-fund analyst at QuantBrief. Be terse, factual, and skeptical. "
    "Separate facts from inference; if you must guess, prefix with 'Assumption:'. "
    "For daily digests, write 1–2 ultra-concise bullets: "
    "• What matters — the single most material point for investors. "
    "• Stance — Positive/Negative/Neutral. "
    "Do not mention Google News or aggregator mechanics; focus on the publisher’s point."
)

def llm_summarize(title: str, body: Optional[str], company: str, symbol: str, url: str) -> Optional[str]:
    """
    Returns a short two-bullet summary, or None on failure.
    """
    if not OPENAI_API_KEY:
        return None
    # If we have zero body, try to avoid hallucination; use title only with caution
    source = f"{domain_of(url)}"
    content_hint = (body or "")[:4000]
    prompt = (
        f"{DAILY_INSTRUCTIONS}\n\n"
        f"Company under coverage: {company} ({symbol}). Source: {source}\n"
        f"Title: {title}\n"
        f"Body (may be partial):\n{content_hint}\n\n"
        f"Return exactly two bullets: '• What matters — ...' and '• Stance — ...'."
    )
    try:
        client = _get_openai()
        r = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=min(_safe_max_tokens(), 220),
        )
        txt = getattr(r, "output_text", None)
        if not txt:
            return None
        # Light post-clean
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        # Keep only two bullets if more were returned
        bullets = [ln for ln in lines if ln.startswith("• ")]
        if len(bullets) >= 2:
            return "\n".join(bullets[:2])
        return "\n".join(lines[:2])
    except Exception:
        return None

# ----------- Mailgun -----------------------
def send_mailgun(subject: str, html: str, to_email: str) -> Dict[str, Any]:
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN and MAILGUN_FROM):
        return {"ok": False, "error": "mailgun not configured"}
    try:
        r = requests.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={
                "from": MAILGUN_FROM,
                "to": to_email,
                "subject": subject,
                "html": html,
            },
            timeout=20,
        )
        if r.status_code >= 400:
            return {"ok": False, "status": r.status_code, "body": r.text}
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": repr(e)}

# ----------- Templating --------------------
EMAIL_TMPL = Template("""
<!doctype html>
<html>
  <body style="font-family:Arial,Helvetica,sans-serif; line-height:1.4;">
    <h2 style="margin:0 0 8px 0;">QuantBrief Daily — {{ run_date }}</h2>
    <div style="color:#555; margin-bottom:16px;">
      Window: last {{ window_minutes }}h ending {{ window_end_local }}
    </div>
    {% for sym, company, rows in grouped %}
      <h3 style="margin:18px 0 8px 0;">Company — {{ company }} ({{ sym }})</h3>
      {% if not rows %}
        <div style="color:#777;">No items found in the selected window.</div>
      {% else %}
        {% for r in rows %}
          <div style="margin:10px 0; padding:10px; border:1px solid #eee; border-radius:8px;">
            <div style="font-weight:bold; margin-bottom:4px;">
              <a href="{{ r.url }}" style="text-decoration:none; color:#0b57d0;">{{ r.title }}</a>
            </div>
            <div style="color:#555; font-size:12px; margin-bottom:6px;">
              {{ r.domain }} — {{ r.published_at or "" }}
            </div>
            {% if r.summary %}
              <div style="white-space:pre-wrap;">{{ r.summary }}</div>
            {% else %}
              <div style="color:#777;">(link only)</div>
            {% endif %}
          </div>
        {% endfor %}
      {% endif %}
    {% endfor %}
    <div style="color:#888; font-size:12px; margin-top:24px;">
      Sources are links only. We don’t republish paywalled content. Filters editable by owner.
    </div>
  </body>
</html>
""".strip())

# ----------- FastAPI app -------------------
app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "app": "quantbrief-daily"}

def _require_admin(x_admin_token: Optional[str]) -> Optional[JSONResponse]:
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        return JSONResponse({"ok": False, "error": "forbidden"}, status_code=403)
    return None

@app.get("/admin/health")
def admin_health(x_admin_token: Optional[str] = Header(None)):
    guard = _require_admin(x_admin_token)
    if guard: return guard

    report: Dict[str, Any] = {}

    # ENV checks
    env_checks = {}
    for k in [
        "DATABASE_URL","OPENAI_API_KEY","OPENAI_MODEL","OPENAI_MAX_OUTPUT_TOKENS",
        "MAILGUN_API_KEY","MAILGUN_DOMAIN","MAILGUN_FROM"
    ]:
        env_checks[k] = bool(os.getenv(k))
    report["env"] = env_checks

    # DB connectivity
    try:
        with engine.begin() as conn:
            conn.execute(text("select 1"))
        report["db"] = {"ok": True}
    except Exception as e:
        report["db"] = {"ok": False, "error": repr(e)}

    # OpenAI quick call
    try:
        client = _get_openai()
        r = client.responses.create(
            model=OPENAI_MODEL,
            input="ping",
            max_output_tokens=min(_safe_max_tokens(), 20),
        )
        sample = getattr(r, "output_text", "") or "ok"
        report["openai"] = {"ok": True, "model": OPENAI_MODEL, "sample": sample[:50]}
    except Exception as e:
        report["openai"] = {"ok": False, "error": repr(e)}

    # Mailgun keys present?
    if env_checks["MAILGUN_API_KEY"] and env_checks["MAILGUN_DOMAIN"] and env_checks["MAILGUN_FROM"]:
        report["mailgun"] = {"ok": True}
    else:
        report["mailgun"] = {"ok": False, "error": "missing var(s)"}

    return report

@app.post("/admin/init")
def admin_init(x_admin_token: Optional[str] = Header(None)):
    guard = _require_admin(x_admin_token)
    if guard: return guard
    try:
        create_tables()
        seeded = {"watchlist": False, "recipients": False}
        with engine.begin() as conn:
            # seed watchlist
            r = conn.execute(watchlist.select().where(watchlist.c.symbol == DEFAULT_SYMBOL)).fetchone()
            if not r:
                conn.execute(watchlist.insert().values(symbol=DEFAULT_SYMBOL, company=DEFAULT_COMPANY))
                seeded["watchlist"] = True
            # seed recipient
            if ADMIN_EMAIL:
                r2 = conn.execute(recipients.select().where(recipients.c.email == ADMIN_EMAIL)).fetchone()
                if not r2:
                    conn.execute(recipients.insert().values(email=ADMIN_EMAIL, active=True))
                    seeded["recipients"] = True
        return {"ok": True, "seeded": seeded}
    except Exception as e:
        return JSONResponse({"ok": False, "error": repr(e)}, status_code=500)

@app.get("/admin/test_openai")
def admin_test_openai(x_admin_token: Optional[str] = Header(None)):
    guard = _require_admin(x_admin_token)
    if guard: return guard
    try:
        client = _get_openai()
        r = client.responses.create(
            model=OPENAI_MODEL,
            input="Return just 'OK'",
            max_output_tokens=min(_safe_max_tokens(), 10),
        )
        txt = getattr(r, "output_text", "").strip()
        return {"ok": True, "model": OPENAI_MODEL, "sample": ("OK" if txt else txt or "OK")}
    except Exception as e:
        return {"ok": False, "model": OPENAI_MODEL, "err": repr(e)}

def _default_queries_for(symbol: str, company: str) -> List[str]:
    # Tweak per sector if you like
    return [
        f"\"{company}\"",
        f"{symbol} stock",
        f"{company} data center power",
        f"{company} PJM",
        f"{company} ERCOT",
        f"{company} capacity market",
    ]

@app.post("/cron/ingest")
def cron_ingest(
    request: Request,
    x_admin_token: Optional[str] = Header(None),
    minutes: int = Query(1440)
):
    guard = _require_admin(x_admin_token)
    if guard: return guard
    window_end = tz_now()
    window_start = window_end - timedelta(minutes=minutes)

    created = 0
    seen = 0

    try:
        with engine.begin() as conn:
            wl = conn.execute(watchlist.select()).fetchall()
            for row in wl:
                symbol = row.symbol
                company = row.company

                # 1) Google News RSS
                entries = fetch_google_news_rss(_default_queries_for(symbol, company))

                # 2) Optional IR feeds
                entries += fetch_ir_feeds(symbol, company)

                # Store unique URLs only
                for e in entries:
                    url = e["url"]
                if not url:
                    continue
                host = domain_of(url)
                if not host or host in EXCLUDE_DOMAINS:
                    continue
                    pub = e.get("published_at")
                    # Skip obviously ancient items if they carry no pub date
                    if pub and (pub < window_start or pub > window_end):
                        # outside window; skip
                        continue

                    seen += 1
                    # Upsert-ish: ignore duplicates by URL unique constraint
                    try:
                        conn.execute(items.insert().values(
                            company=company,
                            symbol=symbol,
                            title=e.get("title") or "(no title)",
                            url=url,
                            domain=host,
                            source_name=e.get("source_name") or host,
                            published_at=pub,
                        ))
                        created += 1
                    except IntegrityError:
                        # already present
                        pass

    except Exception as e:
        return JSONResponse({"ok": False, "error": repr(e)}, status_code=500)

    return {"ok": True, "found_urls": seen, "inserted": created}

@app.post("/cron/digest")
def cron_digest(x_admin_token: Optional[str] = Header(None), minutes: int = Query(1440)):
    guard = _require_admin(x_admin_token)
    if guard: return guard
    window_end = tz_now()
    window_start = window_end - timedelta(minutes=minutes)

    # Load recipients
    with engine.begin() as conn:
        recips = [r.email for r in conn.execute(recipients.select().where(recipients.c.active == True)).fetchall()]
        wl = conn.execute(watchlist.select()).fetchall()

    grouped_render = []
    total_items = 0

    # Summarize up to N items per run to keep cost sane
    SUMMARIZE_LIMIT = 40
    summarized_count = 0

    with engine.begin() as conn:
        for row in wl:
            symbol = row.symbol
            company = row.company
rs = conn.execute(text("""
    select * from items
    where symbol = :sym
      and domain <> ALL(:excluded)
      and (published_at is null or (published_at >= :ws and published_at <= :we))
    order by coalesce(published_at, inserted_at) desc
    limit 80
"""), {
    "sym": symbol,
    "ws": window_start,
    "we": window_end,
    "excluded": list(EXCLUDE_DOMAINS),
}).mappings().all()

            # Try to fill raw_text and summary if missing (but cap work)
            for r in rs:
                if summarized_count >= SUMMARIZE_LIMIT:
                    break
                need_text = not r["have_text"]
                need_sum  = not r["summary"]

                if need_text or need_sum:
                    # skip summarization if this is a Google News domain or a bad domain
                    if r["domain"] in ("news.google.com", "news.url.google.com") or is_bad_domain(r["url"]):
                        continue

                page_text = r["raw_text"]
                if need_text:
                    page_text = extract_main_text(r["url"])
                    conn.execute(items.update().where(items.c.id == r["id"]).values(
                        have_text=bool(page_text),
                        raw_text=page_text
                    ))

                if need_sum:
                    # Only summarize if we have *some* text; otherwise keep link-only
                    if page_text:
                        s = llm_summarize(r["title"], page_text, company, symbol, r["url"])
                        if s:
                            conn.execute(items.update().where(items.c.id == r["id"]).values(summary=s))
                            summarized_count += 1

            # Prepare rows for rendering
            render_rows = []
            for r in rs:
                ts = r["published_at"]
                ts_local = None
                if ts:
                    ts_local = to_tz(ts, TIMEZONE).strftime("%Y-%m-%d %H:%M %Z")
                render_rows.append({
                    "title": r["title"],
                    "url": r["url"],
                    "domain": r["domain"],
                    "published_at": ts_local,
                    "summary": r["summary"],
                })
            total_items += len(render_rows)
            grouped_render.append((symbol, company, render_rows))

    # Build email HTML
    run_date = to_tz(window_end, TIMEZONE).strftime("%Y-%m-%d")
    window_end_local = to_tz(window_end, TIMEZONE).strftime("%H:%M %Z")
    html = EMAIL_TMPL.render(
        run_date=run_date,
        window_minutes=int(minutes/60),
        window_end_local=window_end_local,
        grouped=grouped_render,
    )

    # Send
    results = []
    for to in recips or ([ADMIN_EMAIL] if ADMIN_EMAIL else []):
        subject = f"QuantBrief Daily — {run_date}"
        results.append({"to": to, **send_mailgun(subject, html, to)})

        # Log digest
        try:
            with engine.begin() as conn:
                conn.execute(digests.insert().values(
                    to_email=to,
                    window_start=window_start,
                    window_end=window_end,
                    items_count=total_items,
                ))
        except Exception:
            pass

    return {
        "ok": True,
        "sent_to": [r.get("to") for r in results if r.get("ok")],
        "items": total_items,
        "summarized": summarized_count,
        "mailgun": results,
    }
