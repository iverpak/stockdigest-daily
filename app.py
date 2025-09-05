import os
import re
import hmac
import json
import time
import hashlib
import logging
import datetime as dt
from urllib.parse import urlparse, parse_qs, urlsplit, unquote
from typing import List, Dict, Any, Optional

import requests
import feedparser
import trafilatura
from dateutil import tz
from jinja2 import Template

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import PlainTextResponse, JSONResponse
from openai import OpenAI

# ----------------------------
# Basic app / logging
# ----------------------------
app = FastAPI(title="QuantBrief Daily")
log = logging.getLogger("uvicorn")
log.setLevel(logging.INFO)

# ----------------------------
# Env & constants
# ----------------------------
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "changeme")
DATABASE_URL = os.environ["DATABASE_URL"]  # must be set
OWNER_EMAIL = os.environ.get("OWNER_EMAIL", "you@example.com")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # you can set gpt-5-mini
OPENAI_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "300"))

MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = os.environ.get("MAILGUN_DOMAIN", "")
MAILGUN_FROM = os.environ.get("MAILGUN_FROM", "QuantBrief Daily <daily@example.com>")

DEFAULT_TZ = os.environ.get("DEFAULT_TZ", "America/Toronto")
DIGEST_HOUR_LOCAL = int(os.environ.get("DIGEST_HOUR_LOCAL", "8"))
DIGEST_MINUTE_LOCAL = int(os.environ.get("DIGEST_MINUTE_LOCAL", "30"))

# Known "image cdn / static" domains to ignore
NOISY_DOMAINS = {
    "lh3.googleusercontent.com",
    "gstatic.com",
    "fonts.gstatic.com",
    "fonts.googleapis.com",
    "maps.googleapis.com",
    "news.google.com"  # we try to expand; if not, skip
}

# ----------------------------
# DB helpers
# ----------------------------
def db() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)

def ensure_schema(conn: psycopg.Connection):
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS recipients (
          id BIGSERIAL PRIMARY KEY,
          email TEXT NOT NULL UNIQUE,
          enabled BOOLEAN NOT NULL DEFAULT TRUE,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS watchlist (
          id BIGSERIAL PRIMARY KEY,
          ticker TEXT NOT NULL UNIQUE,
          company TEXT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS article (
          id BIGSERIAL PRIMARY KEY,
          url TEXT NOT NULL,
          canonical_url TEXT,
          title TEXT,
          publisher TEXT,
          published_at TIMESTAMPTZ,
          sha256 CHAR(64) NOT NULL UNIQUE,
          raw_html TEXT,
          clean_text TEXT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS article_nlp (
          id BIGSERIAL PRIMARY KEY,
          article_id BIGINT NOT NULL REFERENCES article(id) ON DELETE CASCADE,
          summary TEXT,
          stance TEXT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE(article_id)
        );

        CREATE TABLE IF NOT EXISTS delivery_log (
          id BIGSERIAL PRIMARY KEY,
          sent_to TEXT NOT NULL,
          sent_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          items INT NOT NULL
        );

        -- helpful indexes
        CREATE INDEX IF NOT EXISTS idx_article_published ON article (published_at DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_article_url_unique ON article (url);
        """)
    # seed defaults if empty
    with conn.cursor() as cur:
        cur.execute("INSERT INTO recipients (email) VALUES (%s) ON CONFLICT DO NOTHING", (OWNER_EMAIL,))
        cur.execute("""
            INSERT INTO watchlist (ticker, company)
            VALUES ('TLN','Talen Energy')
            ON CONFLICT (ticker) DO NOTHING
        """)

def get_watchlist(conn: psycopg.Connection) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT ticker, COALESCE(company, ticker) AS company FROM watchlist ORDER BY ticker")
        return list(cur.fetchall())

def upsert_article(conn: psycopg.Connection, rec: Dict[str, Any]) -> Optional[int]:
    """
    rec keys: url, title, publisher, published_at (datetime or None), canonical_url?
    """
    sha = hashlib.sha256(rec["url"].encode("utf-8")).hexdigest()
    with conn.cursor() as cur:
        try:
            cur.execute("""
                INSERT INTO article (url, canonical_url, title, publisher, published_at, sha256)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (sha256) DO NOTHING
                RETURNING id
            """, (
                rec.get("url"),
                rec.get("canonical_url"),
                rec.get("title"),
                rec.get("publisher"),
                rec.get("published_at"),
                sha
            ))
            row = cur.fetchone()
            return row["id"] if row else None
        except Exception as e:
            log.exception("upsert_article failed: %s", e)
            return None

def list_unsummarized(conn: psycopg.Connection, window_minutes: int = 1440) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.*
            FROM article a
            LEFT JOIN article_nlp n ON n.article_id = a.id
            WHERE n.article_id IS NULL
              AND a.published_at >= now() - INTERVAL '%s minutes'
            ORDER BY a.published_at DESC NULLS LAST
        """, (window_minutes,))
        return list(cur.fetchall())

def store_summary(conn: psycopg.Connection, article_id: int, summary: str, stance: str):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO article_nlp (article_id, summary, stance)
            VALUES (%s, %s, %s)
            ON CONFLICT (article_id) DO UPDATE
              SET summary = EXCLUDED.summary,
                  stance = EXCLUDED.stance
        """, (article_id, summary, stance))

def list_recipients(conn: psycopg.Connection) -> List[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT email FROM recipients WHERE enabled = TRUE ORDER BY id")
        return [r["email"] for r in cur.fetchall()]

def list_recent_summaries(conn: psycopg.Connection, window_minutes: int = 1440) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.id, a.url, a.title, a.publisher, a.published_at, n.summary, n.stance
            FROM article a
            JOIN article_nlp n ON n.article_id = a.id
            WHERE a.published_at >= now() - INTERVAL '%s minutes'
            ORDER BY a.published_at DESC NULLS LAST
        """, (window_minutes,))
        return list(cur.fetchall())

# ----------------------------
# OpenAI lazy client (NO proxies kwarg)
# ----------------------------
_openai_client: Optional[OpenAI] = None

def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client

def clamp_tokens(n: int) -> int:
    # avoid "below minimum value" errors
    n = max(n, 128)
    # most models allow far more, but we keep this sane and cheap
    return min(n, 2000)

# ----------------------------
# Utilities
# ----------------------------
def as_utc(dt_like: Optional[dt.datetime]) -> Optional[dt.datetime]:
    if not dt_like:
        return None
    if dt_like.tzinfo is None:
        return dt_like.replace(tzinfo=tz.UTC)
    return dt_like.astimezone(tz.UTC)

def domain_of(u: str) -> str:
    try:
        return urlparse(u).hostname or ""
    except Exception:
        return ""

def now_local() -> dt.datetime:
    return dt.datetime.now(tz=tz.gettz(DEFAULT_TZ))

def expand_google_news_link(link: str) -> Optional[str]:
    # Google News sometimes has url= param embedded
    try:
        parsed = urlsplit(link)
        qs = parse_qs(parsed.query)
        if "url" in qs:
            return unquote(qs["url"][0])
    except Exception:
        pass
    return None

def extract_readable(url: str) -> Dict[str, Optional[str]]:
    """
    Fetch and extract main text. Avoids saving images/garbage.
    """
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=False)
        if not downloaded:
            return {"raw_html": None, "clean_text": None}
        clean = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        return {"raw_html": downloaded, "clean_text": clean}
    except Exception:
        return {"raw_html": None, "clean_text": None}

# ----------------------------
# Ingestion (GDELT + optional Google News RSS)
# ----------------------------
def ingest_gdelt_for_query(query: str, minutes: int = 1440) -> List[Dict[str, Any]]:
    """
    Use GDELT 2.1 DOC API: https://api.gdeltproject.org/api/v2/doc/doc
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "timespan": f"{minutes}m",
        "format": "json",
        "maxrecords": "100"
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        out = []
        for it in data.get("articles", []):
            link = it.get("url")
            titl = it.get("title")
            pub = it.get("sourceDomain")
            ts = it.get("seendate") or it.get("date")
            published = None
            try:
                if ts:
                    # GDELT times are like 2025-09-05 04:12:00
                    published = as_utc(dt.datetime.fromisoformat(ts.replace(" ", "T")))
            except Exception:
                published = None
            if link and domain_of(link) not in NOISY_DOMAINS:
                out.append({
                    "url": link,
                    "canonical_url": None,
                    "title": titl,
                    "publisher": pub,
                    "published_at": published
                })
        return out
    except Exception as e:
        log.warning("GDELT fetch failed for %s: %s", query, e)
        return []

def ingest_google_news_rss(query: str, minutes: int = 1440) -> List[Dict[str, Any]]:
    """
    Light RSS approach; we try to expand to the original site link and skip image/static domains.
    """
    q = requests.utils.quote(query)
    rss = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    out = []
    try:
        feed = feedparser.parse(rss)
        cutoff = dt.datetime.utcnow().replace(tzinfo=tz.UTC) - dt.timedelta(minutes=minutes)
        for e in feed.entries:
            link = e.link
            # expand aggregator to the source url if possible
            expanded = expand_google_news_link(link) or link
            d = domain_of(expanded)
            if not d or d in NOISY_DOMAINS:
                continue
            # compute published time
            published = None
            try:
                if hasattr(e, "published_parsed") and e.published_parsed:
                    published = as_utc(dt.datetime.fromtimestamp(time.mktime(e.published_parsed)))
            except Exception:
                published = None
            if published and published < cutoff:
                continue
            out.append({
                "url": expanded,
                "canonical_url": None,
                "title": e.title,
                "publisher": d,
                "published_at": published
            })
        return out
    except Exception as e:
        log.warning("Google News RSS failed for %s: %s", query, e)
        return []

def ingest_all(minutes: int = 1440) -> Dict[str, Any]:
    conn = db()
    ensure_schema(conn)
    watch = get_watchlist(conn)
    found, inserted = 0, 0

    # Build queries per ticker/company
    queries = []
    for w in watch:
        # Example: ("TLN" OR "Talen Energy") AND (news OR announcement OR deal)
        ticker = w["ticker"]
        comp = w["company"] or ticker
        queries.append(f'("{ticker}" OR "{comp}")')

    for q in queries:
        gdelt_items = ingest_gdelt_for_query(q, minutes=minutes)
        gnews_items = ingest_google_news_rss(q, minutes=minutes)
        items = gdelt_items + gnews_items
        for rec in items:
            found += 1
            if domain_of(rec["url"]) in NOISY_DOMAINS:
                continue
            art_id = upsert_article(conn, rec)
            if art_id:
                inserted += 1

    conn.close()
    return {"ok": True, "found_urls": found, "inserted": inserted}

# ----------------------------
# Summarize + Mail
# ----------------------------
SUMMARY_PROMPT = """You are a strict buy-side junior PM analyst. Given an article excerpt, produce two fields:

• What matters — 1–2 crisp sentences with the investor takeaway (no fluff).
• Stance — one of: Positive / Neutral / Negative, based only on the excerpt.

Return plain text with exactly these two bullets.
"""

def summarize_text(text: str, title: Optional[str]) -> Dict[str, str]:
    if not text:
        return {"summary": "I don’t have the source text. Please paste key excerpts.", "stance": "Neutral"}

    client = get_openai()
    max_tokens = clamp_tokens(OPENAI_MAX_OUTPUT_TOKENS)

    prompt = SUMMARY_PROMPT
    if title:
        prompt = f"Title: {title}\n\n" + prompt

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": "You summarize financial news conservatively and concisely."},
                {"role": "user", "content": f"{prompt}\n\n---\n{text}\n---"}
            ],
            max_output_tokens=max_tokens,
            temperature=0.2,
        )
        out = resp.output_text.strip()
        # crude split for stance line
        stance_match = re.search(r"Stance\s*—\s*(Positive|Neutral|Negative)", out, re.IGNORECASE)
        stance = stance_match.group(1).capitalize() if stance_match else "Neutral"
        # Keep only the two bullets
        return {"summary": out, "stance": stance}
    except Exception as e:
        return {"summary": f"(LLM error) {e}", "stance": "Neutral"}

def build_email_html(items: List[Dict[str, Any]]) -> str:
    local_now = now_local()
    window = f"last 24h ending {DIGEST_HOUR_LOCAL:02d}:{DIGEST_MINUTE_LOCAL:02d} {DEFAULT_TZ}"
    comp_line = "Companies — " + ", ".join(sorted({i.get("publisher") or domain_of(i.get("url","")) for i in items})) if items else "No companies"
    tmpl = Template("""
    <div style="font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:780px">
      <h2>QuantBrief Daily — {{ date_str }}</h2>
      <p><b>Window:</b> {{ window }}</p>
      {% for it in items %}
        <div style="margin:16px 0;padding:12px;border:1px solid #eee;border-radius:8px">
          <div style="font-size:14px;color:#666">{{ it['publisher'] or it['domain'] }} — {{ it['published_at'] or '' }}</div>
          <div style="font-size:16px;margin:6px 0">
            <a href="{{ it['url'] }}" target="_blank" style="text-decoration:none">{{ it['title'] or it['url'] }}</a>
          </div>
          <pre style="white-space:pre-wrap;font-family:inherit;margin:8px 0">{{ it['summary'] }}</pre>
        </div>
      {% endfor %}
      <hr/>
      <small>Sources are links only. We don’t republish paywalled content. Filters editable by owner.</small>
    </div>
    """)
    return tmpl.render(
        date_str=local_now.date().isoformat(),
        window=window,
        items=items
    )

def send_mail(subject: str, html: str, to_list: List[str]) -> Dict[str, Any]:
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN and MAILGUN_FROM):
        return {"ok": False, "err": "Mailgun env not set"}

    url = f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages"
    try:
        r = requests.post(url, auth=("api", MAILGUN_API_KEY), data={
            "from": MAILGUN_FROM,
            "to": to_list,
            "subject": subject,
            "html": html
        }, timeout=20)
        ok = r.status_code < 300
        return {"ok": ok, "status": r.status_code, "text": r.text[:500]}
    except Exception as e:
        return {"ok": False, "err": str(e)}

def run_summarize_and_send(window_minutes: int = 1440) -> Dict[str, Any]:
    conn = db()
    ensure_schema(conn)

    # 1) Pull unsummarized articles and extract text
    uns = list_unsummarized(conn, window_minutes=window_minutes)
    summarized = 0
    for a in uns:
        # skip noisy domains
        if domain_of(a["url"]) in NOISY_DOMAINS:
            continue
        # fetch only if we don't have clean_text
        if not a.get("clean_text"):
            ext = extract_readable(a["url"])
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE article SET raw_html=%s, clean_text=%s WHERE id=%s",
                    (ext["raw_html"], ext["clean_text"], a["id"])
                )
            a["clean_text"] = ext["clean_text"]
        s = summarize_text(a.get("clean_text") or "", a.get("title"))
        store_summary(conn, a["id"], s["summary"], s["stance"])
        summarized += 1

    # 2) Build digest list
    items = list_recent_summaries(conn, window_minutes=window_minutes)
    for it in items:
        it["domain"] = domain_of(it["url"])

    # 3) Send email
    to_list = list_recipients(conn)
    subject = f"QuantBrief Daily — {now_local().date().isoformat()}"
    html = build_email_html(items)
    mail_res = send_mail(subject, html, to_list)

    # 4) Log delivery
    with conn.cursor() as cur:
        cur.execute("INSERT INTO delivery_log (sent_to, items) VALUES (%s, %s)", (",".join(to_list), len(items)))

    conn.close()
    return {"ok": True, "sent_to": to_list, "items": len(items), "summarized": summarized, "mail": mail_res}

# ----------------------------
# Auth
# ----------------------------
def require_admin(request: Request):
    token = request.headers.get("x-admin-token")
    if not token or token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return "QuantBrief Daily — OK"

@app.get("/admin/health", dependencies=[Depends(require_admin)])
def admin_health():
    # Show which envs are present without leaking values
    present = {
        k: bool(os.environ.get(k))
        for k in ["DATABASE_URL","OPENAI_API_KEY","OPENAI_MODEL","OPENAI_MAX_OUTPUT_TOKENS",
                  "MAILGUN_API_KEY","MAILGUN_DOMAIN","MAILGUN_FROM","OWNER_EMAIL","ADMIN_TOKEN"]
    }
    # counts
    conn = db()
    ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) c FROM article")
        arts = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) c FROM article_nlp")
        sumd = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) c FROM recipients")
        recs = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) c FROM watchlist")
        w = cur.fetchone()["c"]
    conn.close()
    return PlainTextResponse(
        "env\n---\n" + json.dumps(present, indent=2) + "\n\ncounts\n------\n"
        + f"articles={arts}\nsummarized={sumd}\nrecipients={recs}\nwatchlist={w}\n"
    )

@app.get("/admin/test_openai", dependencies=[Depends(require_admin)])
def test_openai():
    try:
        client = get_openai()
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input="Say OK",
            max_output_tokens=clamp_tokens(OPENAI_MAX_OUTPUT_TOKENS),
            temperature=0
        )
        return PlainTextResponse(
            f"   ok model      sample\n   -- -----      ------\nTrue {OPENAI_MODEL} OK\n"
        )
    except Exception as e:
        return PlainTextResponse(
            "   ok model      err\n   -- -----      ---\nFalse "
            f"{OPENAI_MODEL} {str(e)[:200]}\n"
        )

@app.post("/admin/init", dependencies=[Depends(require_admin)])
def admin_init():
    try:
        conn = db()
        ensure_schema(conn)
        conn.close()
        return PlainTextResponse("Initialized.\n")
    except Exception as e:
        log.exception("admin/init failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cron/ingest", dependencies=[Depends(require_admin)])
def cron_ingest(minutes: Optional[int] = 1440):
    try:
        res = ingest_all(minutes=minutes or 1440)
        return PlainTextResponse(
            "  ok found_urls inserted\n  -- ---------- --------\n"
            f"True {res['found_urls']:>10} {res['inserted']:>8}\n"
        )
    except Exception as e:
        log.exception("cron/ingest failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cron/digest", dependencies=[Depends(require_admin)])
def cron_digest(minutes: Optional[int] = 1440):
    try:
        res = run_summarize_and_send(window_minutes=minutes or 1440)
        return PlainTextResponse(
            "  ok sent_to                       items summarized\n"
            "  -- -------                       ----- ----------\n"
            f"True {','.join(res['sent_to']):<28} {res['items']:>5} {res['summarized']:>10}\n"
        )
    except Exception as e:
        log.exception("cron/digest failed")
        raise HTTPException(status_code=500, detail=str(e))
