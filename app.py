import os
import re
import hmac
import json
import math
import time
import hashlib
import logging
import feedparser
import requests
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlencode
from typing import List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import PlainTextResponse, JSONResponse
from jinja2 import Template
from dateutil import tz
from trafilatura import extract as trafilatura_extract
from bs4 import BeautifulSoup

from openai import OpenAI   # modern SDK

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("quantbrief")

app = FastAPI(title="QuantBrief Daily")

# ---------- ENV ----------
DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
OPENAI_MAX_OUTPUT_TOKENS = os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "800")

MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN  = os.environ.get("MAILGUN_DOMAIN", "")
MAILGUN_FROM    = os.environ.get("MAILGUN_FROM", "QuantBrief Daily <daily@mg.example.com>")
OWNER_EMAIL     = os.environ.get("OWNER_EMAIL", "you@example.com")
ADMIN_TOKEN     = os.environ.get("ADMIN_TOKEN", "")

DEFAULT_TZ = os.environ.get("DEFAULT_TZ", "America/Toronto")
DIGEST_HOUR   = int(os.environ.get("DIGEST_HOUR", "8"))
DIGEST_MINUTE = int(os.environ.get("DIGEST_MINUTE", "30"))

# ---------- DB ----------
def db():
    return psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)

def ensure_schema():
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS app_user (
          id BIGSERIAL PRIMARY KEY,
          email TEXT UNIQUE NOT NULL,
          is_owner BOOLEAN NOT NULL DEFAULT false,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS recipients (
          id BIGSERIAL PRIMARY KEY,
          email TEXT UNIQUE NOT NULL,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
          id BIGSERIAL PRIMARY KEY,
          symbol TEXT NOT NULL,
          company TEXT,
          notes TEXT,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_watchlist (
          user_email TEXT NOT NULL,
          watch_id BIGINT NOT NULL,
          PRIMARY KEY (user_email, watch_id),
          FOREIGN KEY (watch_id) REFERENCES watchlist(id) ON DELETE CASCADE
        );
        """)
        # canonical article store
        cur.execute("""
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
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        # add computed domain column if missing
        cur.execute("""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='article' AND column_name='domain'
          ) THEN
            ALTER TABLE public.article
            ADD COLUMN domain TEXT GENERATED ALWAYS AS (
              lower(split_part(regexp_replace(coalesce(canonical_url, url), '^[a-z]+://', ''), '/', 1))
            ) STORED;
            CREATE INDEX IF NOT EXISTS idx_article_domain ON public.article(domain);
          END IF;
        END$$;
        """)
        cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_article_url_unique ON article (url);
        CREATE INDEX IF NOT EXISTS idx_article_published ON article (published_at DESC);
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS article_nlp (
          id BIGSERIAL PRIMARY KEY,
          article_id BIGINT REFERENCES article(id) ON DELETE CASCADE,
          stance TEXT,
          what_matters TEXT,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS delivery_log (
          id BIGSERIAL PRIMARY KEY,
          sent_to TEXT NOT NULL,
          items INT NOT NULL,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS source_feed (
          id BIGSERIAL PRIMARY KEY,
          name TEXT NOT NULL,
          kind TEXT NOT NULL, -- 'rss' | 'google'
          url TEXT NOT NULL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS gdelt_filter_set (
          id BIGSERIAL PRIMARY KEY,
          name TEXT NOT NULL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS gdelt_filter_rule (
          id BIGSERIAL PRIMARY KEY,
          set_id BIGINT REFERENCES gdelt_filter_set(id) ON DELETE CASCADE,
          include TEXT,   -- regex or keyword
          exclude TEXT    -- regex or keyword
        );
        """)

# ---------- UTIL ----------
def admin_guard(request: Request):
    token = request.headers.get("x-admin-token")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def clamp_max_output_tokens(val_str: str) -> int:
    try:
        v = int(val_str)
    except Exception:
        v = 800
    # avoid OpenAI “below minimum” errors
    v = max(64, min(v, 8000))
    return v

# ---------- OPENAI (lazy, no proxies kwarg) ----------
_client: Optional[OpenAI] = None

def get_openai() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=OPENAI_API_KEY)   # <-- FIX: no 'proxies=' here
    return _client

def llm_summarize(text: str) -> Tuple[str, str]:
    """
    Returns (what_matters, stance)
    """
    prompt = f"""
You are a buy-side junior PM assistant. Read the article text below and respond with two short fields:

[WHAT MATTERS]
• 1–3 bullet points of factual takeaways relevant to investors. Keep it concise.

[STANCE]
• One word: Positive / Negative / Neutral. If unclear, Neutral.

--- ARTICLE TEXT START ---
{text[:15000]}
--- ARTICLE TEXT END ---
"""
    client = get_openai()
    max_tokens = clamp_max_output_tokens(OPENAI_MAX_OUTPUT_TOKENS)

    # Use the Responses API; it matches the new SDK and lets us set max_output_tokens.
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        max_output_tokens=max_tokens,
    )
    out = resp.output_text or ""
    # naive parse
    wm_match = re.search(r"\[WHAT MATTERS\](.*?)(?:\[|$)", out, re.S | re.I)
    st_match = re.search(r"\[STANCE\](.*)", out, re.S | re.I)
    wm = (wm_match.group(1) if wm_match else out).strip()
    st = (st_match.group(1) if st_match else "Neutral").strip()
    st = st.split()[0].strip(":.").title() if st else "Neutral"
    if st not in ("Positive", "Negative", "Neutral"):
        st = "Neutral"
    return wm, st

# ---------- FEEDS ----------
GOOGLE_NEWS_Q = os.environ.get("GOOGLE_NEWS_Q", "Talen Energy|TLN|independent power producers|PJM|ERCOT")
GOOGLE_NEWS_LANG = os.environ.get("GOOGLE_NEWS_LANG", "en")
GOOGLE_NEWS_ED   = os.environ.get("GOOGLE_NEWS_ED", "us")

DEFAULT_RSS = [
    ("Company IR — Press", "rss", "https://www.globenewswire.com/RssFeed/orgclass/79/feedTitle/Company%20Press%20Releases"),
    ("PJM News",           "rss", "https://www.pjm.com/-/media/about-pjm/newsroom/rss/rss.ashx"),
    ("ERCOT News",         "rss", "https://www.ercot.com/rss/news"),
]

def google_news_urls(query: str) -> List[str]:
    # Google News RSS (topic search)
    q = query
    url = f"https://news.google.com/rss/search?{urlencode({'q': q, 'hl': GOOGLE_NEWS_LANG, 'gl': GOOGLE_NEWS_ED, 'ceid': f'{GOOGLE_NEWS_ED}:{GOOGLE_NEWS_LANG}'})}"
    return [url]

def feed_fetch(url: str) -> feedparser.FeedParserDict:
    return feedparser.parse(url)

def extract_main_text(url: str, html: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Fetch page and extract readable text.
    Returns (publisher, title, clean_text)
    """
    final_html = html
    if not final_html:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (QuantBriefBot)"})
            if "text/html" in r.headers.get("content-type", ""):
                final_html = r.text
            else:
                final_html = None
        except Exception:
            final_html = None

    publisher = urlparse(url).hostname or ""
    title = None
    clean_text = None

    if final_html:
        # Try trafilatura
        try:
            clean_text = trafilatura_extract(final_html)
        except Exception:
            clean_text = None
        # Title from HTML
        try:
            soup = BeautifulSoup(final_html, "lxml")
            t = soup.find("title")
            if t and t.text:
                title = t.text.strip()
        except Exception:
            pass

    return publisher, title, clean_text

# ---------- EMAIL ----------
def send_mailgun(to_email: str, subject: str, html: str) -> bool:
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN):
        log.error("Mailgun not configured")
        return False
    try:
        r = requests.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={
                "from": MAILGUN_FROM,
                "to": [to_email],
                "subject": subject,
                "html": html,
            },
            timeout=30,
        )
        if r.status_code >= 200 and r.status_code < 300:
            return True
        log.error("Mailgun error %s: %s", r.status_code, r.text)
    except Exception as e:
        log.exception("Mailgun exception: %s", e)
    return False

EMAIL_TEMPLATE = Template("""
<h3>QuantBrief Daily — {{ as_of.date() }}</h3>
<p><b>Window:</b> last 24h ending {{ window_end_tz.strftime("%H:%M %Z") }}</p>
{% for bucket in buckets %}
  <h4>Company — {{ bucket.company }} ({{ bucket.symbol }})</h4>
  {% for a in bucket.articles %}
    <p><b>{{ a.publisher or a.domain }}</b><br/>
    <a href="{{ a.url }}">{{ a.title or a.url }}</a><br/>
    <small>{{ a.published_at }}</small><br/>
    {% if a.what_matters %}
      • <b>What matters</b> — {{ a.what_matters }} • <b>Stance</b> — {{ a.stance or "Neutral" }}
    {% elif a.clean_text %}
      <i>Text extracted but not summarized.</i>
    {% else %}
      <i>I don’t have the source text. Open the link to view.</i>
    {% endif %}
    </p>
  {% endfor %}
{% endfor %}
<p><small>Sources are links only. We don’t republish paywalled content. Filters editable by owner.</small></p>
""".strip())

# ---------- ADMIN ----------
@app.get("/admin/health", response_class=PlainTextResponse)
def admin_health(request: Request, ok=Depends(admin_guard)):
    env_ok = {
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
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM article;")
        articles = cur.fetchone()["count"]
        cur.execute("SELECT count(*) FROM article_nlp;")
        summarized = cur.fetchone()["count"]
        cur.execute("SELECT count(*) FROM recipients;")
        recipients = cur.fetchone()["count"]
        cur.execute("SELECT count(*) FROM watchlist;")
        watch = cur.fetchone()["count"]

    lines = [
        "env",
        "---",
        json.dumps(env_ok, indent=2),
        "",
        "counts",
        "------",
        f"articles={articles}",
        f"summarized={summarized}",
        f"recipients={recipients}",
        f"watchlist={watch}",
        "",
    ]
    return "\n".join(lines)

@app.get("/admin/test_openai", response_class=PlainTextResponse)
def admin_test_openai(request: Request, ok=Depends(admin_guard)):
    try:
        client = get_openai()
        sample = client.responses.create(
            model=OPENAI_MODEL,
            input="Say OK",
            max_output_tokens=clamp_max_output_tokens(OPENAI_MAX_OUTPUT_TOKENS),
        )
        out = sample.output_text or ""
        return Template("""
   ok model      err
   -- -----      ---
{{ ok }} {{ model }} {{ err }}
""").render(ok=str(bool(out.strip().upper().startswith("OK"))), model=OPENAI_MODEL, err="")
    except Exception as e:
        return Template("""
   ok model      err
   -- -----      ---
False {{ model }} {{ err }}
""").render(model=OPENAI_MODEL, err=str(e))

@app.post("/admin/init")
def admin_init(request: Request, ok=Depends(admin_guard)):
    ensure_schema()
    with db() as conn, conn.cursor() as cur:
        # seed owner as recipient
        cur.execute("INSERT INTO recipients (email) VALUES (%s) ON CONFLICT (email) DO NOTHING;", (OWNER_EMAIL,))
        # seed basic watchlist (TLN)
        cur.execute("""
        INSERT INTO watchlist (symbol, company)
        VALUES ('TLN', 'Talen Energy')
        ON CONFLICT DO NOTHING;
        """)
        # seed feeds if empty
        cur.execute("SELECT COUNT(*) AS c FROM source_feed;")
        if cur.fetchone()["c"] == 0:
            for name, kind, url in DEFAULT_RSS:
                cur.execute("INSERT INTO source_feed (name, kind, url) VALUES (%s, %s, %s);", (name, kind, url))
            # google news search for the combined query
            for gurl in google_news_urls(GOOGLE_NEWS_Q):
                cur.execute("INSERT INTO source_feed (name, kind, url) VALUES (%s, %s, %s);",
                            ("Google News", "rss", gurl))
    return {"ok": True}

# ---------- CRON: INGEST ----------
@app.post("/cron/ingest")
def cron_ingest(request: Request, ok=Depends(admin_guard), minutes: int = 1440):
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    found = 0
    inserted = 0
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, name, kind, url FROM source_feed;")
        feeds = cur.fetchall()

        for f in feeds:
            fp = feed_fetch(f["url"])
            for e in fp.entries:
                # published
                published_at = None
                for k in ("published_parsed", "updated_parsed"):
                    if getattr(e, k, None):
                        published_at = datetime.fromtimestamp(time.mktime(getattr(e, k)), tz=timezone.utc)
                        break
                if not published_at:
                    published_at = datetime.now(timezone.utc)

                if published_at < since:
                    continue

                link = getattr(e, "link", None) or ""
                title = getattr(e, "title", None)
                if not link:
                    continue
                found += 1

                h = sha256(link)
                cur.execute("SELECT id FROM article WHERE sha256=%s;", (h,))
                row = cur.fetchone()
                if row:
                    continue

                publisher = urlparse(link).hostname or f["name"]
                clean_text = None
                raw_html = None
                try:
                    r = requests.get(link, timeout=20, headers={"User-Agent": "Mozilla/5.0 (QuantBriefBot)"})
                    if "text/html" in r.headers.get("content-type", ""):
                        raw_html = r.text
                        clean_text = trafilatura_extract(raw_html)
                except Exception:
                    pass

                cur.execute("""
                  INSERT INTO article (url, canonical_url, title, publisher, published_at, sha256, raw_html, clean_text)
                  VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id;
                """, (link, None, title, publisher, published_at, h, raw_html, clean_text))
                inserted += 1
    return {"ok": True, "found_urls": found, "inserted": inserted}

# ---------- CRON: DIGEST ----------
@app.post("/cron/digest")
def cron_digest(request: Request, ok=Depends(admin_guard)):
    now_utc = datetime.now(timezone.utc)
    local_tz = tz.gettz(DEFAULT_TZ)
    window_end_local = now_utc.astimezone(local_tz)

    # Build per-symbol buckets (simple: all articles in last 24h)
    since = now_utc - timedelta(days=1)

    with db() as conn, conn.cursor() as cur:
        # summarize anything lacking NLP
        cur.execute("""
        SELECT id, url, title, publisher, published_at, clean_text
        FROM article
        WHERE published_at >= %s
        ORDER BY published_at DESC
        """, (since,))
        arts = cur.fetchall()

        summarized = 0
        for a in arts:
            cur.execute("SELECT 1 FROM article_nlp WHERE article_id=%s;", (a["id"],))
            if cur.fetchone():
                continue
            wm, st = ("", "Neutral")
            if a["clean_text"]:
                try:
                    wm, st = llm_summarize(a["clean_text"])
                except Exception as e:
                    log.error("LLM summarize failed for %s: %s", a["url"], e)
                    wm, st = ("", "Neutral")
            cur.execute("INSERT INTO article_nlp (article_id, stance, what_matters) VALUES (%s,%s,%s);",
                        (a["id"], st, wm))
            summarized += 1

        # join nlp back
        cur.execute("""
        SELECT a.id, a.url, a.title, a.publisher, a.published_at, a.domain, n.what_matters, n.stance, a.clean_text
        FROM article a
        LEFT JOIN article_nlp n ON n.article_id = a.id
        WHERE a.published_at >= %s
        ORDER BY a.published_at DESC
        """, (since,))
        joined = cur.fetchall()

        # simple bucket: pretend watchlist is just TLN for now
        buckets = [{
            "company": "Talen Energy",
            "symbol": "TLN",
            "articles": joined
        }]

        html = EMAIL_TEMPLATE.render(
            as_of=now_utc.astimezone(local_tz),
            window_end_tz=window_end_local,
            buckets=buckets
        )

        sent = send_mailgun(OWNER_EMAIL, f"QuantBrief Daily — {now_utc.astimezone(local_tz).date()} — TLN", html)
        cur.execute("INSERT INTO delivery_log (sent_to, items) VALUES (%s,%s);", (OWNER_EMAIL, len(joined)))

    return {"ok": bool(sent), "sent_to": OWNER_EMAIL, "items": len(joined)}

# ---------- ERROR HANDLER ----------
@app.exception_handler(Exception)
async def on_err(request: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})
