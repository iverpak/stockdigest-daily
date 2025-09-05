import os
import re
import hmac
import json
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs, urljoin, unquote

import requests
import feedparser
import trafilatura
from dateutil import tz, parser as dateparser
from jinja2 import Template

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

import psycopg
from psycopg.rows import dict_row

from openai import OpenAI

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
APP_TZ = os.getenv("APP_TZ", "America/Toronto")
TZINFO = tz.gettz(APP_TZ)

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # you prefer this
OPENAI_MAX_OUTPUT_TOKENS = os.getenv("OPENAI_MAX_OUTPUT_TOKENS")  # we WON'T pass it
MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY", "")
MAILGUN_DOMAIN = os.getenv("MAILGUN_DOMAIN", "")
MAILGUN_FROM = os.getenv("MAILGUN_FROM", "")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# Google News locale
GN_HL = os.getenv("GOOGLE_NEWS_HL", "en-US")
GN_GL = os.getenv("GOOGLE_NEWS_GL", "US")
GN_CEID = os.getenv("GOOGLE_NEWS_CEID", "US:en")

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("quantbrief")

# Single OpenAI client (1.51.0). Do NOT pass unsupported args (e.g. proxies)
_openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Requests session with a realistic UA and redirects
http = requests.Session()
http.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
})
http.timeout = 20  # default per-request timeout via kwargs when used

app = FastAPI()


# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------
def db():
    return psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)


def ensure_schema(conn):
    # Core tables
    conn.execute("""
    CREATE TABLE IF NOT EXISTS recipients (
        id          BIGSERIAL PRIMARY KEY,
        email       TEXT UNIQUE NOT NULL,
        name        TEXT,
        enabled     BOOLEAN NOT NULL DEFAULT TRUE,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS watchlist (
        id          BIGSERIAL PRIMARY KEY,
        symbol      TEXT,
        company     TEXT,
        enabled     BOOLEAN NOT NULL DEFAULT TRUE,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)

    # Articles + tags + summaries
    conn.execute("""
    CREATE TABLE IF NOT EXISTS article (
        id            BIGSERIAL PRIMARY KEY,
        url           TEXT NOT NULL,
        canonical_url TEXT,
        title         TEXT,
        publisher     TEXT,
        published_at  TIMESTAMPTZ,
        sha256        CHAR(64) NOT NULL UNIQUE,
        raw_html      TEXT,
        clean_text    TEXT,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)

    # Domain (generated) if not present
    conn.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='article' AND column_name='domain'
        ) THEN
            ALTER TABLE public.article
            ADD COLUMN domain TEXT
            GENERATED ALWAYS AS (
              lower(split_part(regexp_replace(coalesce(canonical_url, url), '^[a-z]+://', ''), '/', 1))
            ) STORED;
        END IF;
    END$$;
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS article_tag (
        id            BIGSERIAL PRIMARY KEY,
        article_id    BIGINT NOT NULL REFERENCES article(id) ON DELETE CASCADE,
        watchlist_id  BIGINT NOT NULL REFERENCES watchlist(id) ON DELETE CASCADE,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE(article_id, watchlist_id)
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS article_nlp (
        id            BIGSERIAL PRIMARY KEY,
        article_id    BIGINT NOT NULL REFERENCES article(id) ON DELETE CASCADE,
        model         TEXT,
        stance        TEXT,
        summary       TEXT,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE(article_id)
    );
    """)

    # Delivery log with robust columns
    conn.execute("""
    CREATE TABLE IF NOT EXISTS delivery_log (
        id               BIGSERIAL PRIMARY KEY,
        run_date         DATE NOT NULL DEFAULT CURRENT_DATE,
        run_started_at   TIMESTAMPTZ DEFAULT NOW(),
        run_ended_at     TIMESTAMPTZ,
        created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        sent_to          TEXT,
        items            INTEGER NOT NULL DEFAULT 0,
        summarized       INTEGER NOT NULL DEFAULT 0
    );
    """)

    # Helpful indexes
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_article_url_unique ON article(url);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_article_published ON article(published_at DESC);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_article_domain ON article(domain);")


def seed_basics(conn):
    # Seed watchlist (keep your TLN example)
    conn.execute("""
    INSERT INTO watchlist (symbol, company)
    VALUES ('TLN', 'Talen Energy')
    ON CONFLICT DO NOTHING;
    """)
    # Recipients: keep yours; OWNER_EMAIL optional
    owner = os.getenv("OWNER_EMAIL")
    default_list = [e.strip() for e in os.getenv("DEFAULT_RECIPIENTS", "").split(",") if e.strip()]
    emails = set(default_list)
    if owner:
        emails.add(owner)
    # Always include your working address if provided separately (optional)
    for e in emails:
        conn.execute(
            "INSERT INTO recipients (email, enabled) VALUES (%s, TRUE) ON CONFLICT (email) DO NOTHING;",
            (e,)
        )


# -----------------------------------------------------------------------------
# Google News helpers
# -----------------------------------------------------------------------------
NEWS_GOOGLE_HOST = "news.google.com"
IMG_CDNS = {"lh3.googleusercontent.com", "gstatic.com", "fonts.gstatic.com", "fonts.googleapis.com", "maps.googleapis.com"}
IMG_EXT = re.compile(r"\.(png|jpe?g|gif|webp)(\?.*)?$", re.I)


def is_image_or_cdn(url: str) -> bool:
    try:
        u = urlparse(url)
        if u.netloc in IMG_CDNS:
            return True
        if IMG_EXT.search(u.path or ""):
            return True
        return False
    except Exception:
        return True


def resolve_google_news(url: str) -> str:
    """
    Tiny patch: take a news.google.com link and resolve the real publisher URL.
    Strategies:
      1) If 'url=' query param exists, use it.
      2) Follow redirects (GET, allow_redirects=True).
      3) As a last resort, return the original URL.
    """
    try:
        u = urlparse(url)
        if u.netloc == NEWS_GOOGLE_HOST:
            qs = parse_qs(u.query or "")
            if "url" in qs and qs["url"]:
                # URL param may be url-encoded
                return unquote(qs["url"][0])

        # Try to follow redirects (many GN links 302 to publisher)
        r = http.get(url, allow_redirects=True, timeout=20)
        final = r.url
        if final and urlparse(final).netloc != NEWS_GOOGLE_HOST:
            return final

        # If still on Google and a 'url=' is embedded (rss/articles pattern)
        qs = parse_qs(urlparse(final).query or "")
        if "url" in qs and qs["url"]:
            return unquote(qs["url"][0])

    except Exception as e:
        log.warning(f"resolve_google_news failed: {e}")

    return url  # fallback


def rss_url_for(company: str, symbol: str | None) -> str:
    # Query like: "Talen Energy" OR TLN site:news
    terms = [f'"{company}"']
    if symbol:
        terms.append(symbol)
    q = " OR ".join(terms)
    return (
        "https://news.google.com/rss/search?"
        f"q={requests.utils.quote(q)}"
        f"&hl={GN_HL}&gl={GN_GL}&ceid={GN_CEID}"
    )


# -----------------------------------------------------------------------------
# Ingest
# -----------------------------------------------------------------------------
def upsert_article(conn, url: str, canonical: str, title: str | None,
                   publisher: str | None, published_at: datetime | None,
                   raw_html: str | None, clean_text: str | None) -> int | None:
    if not canonical:
        canonical = url
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # Skip obvious images/CDNs
    if is_image_or_cdn(canonical):
        return None

    row = conn.execute(
        "SELECT id FROM article WHERE sha256 = %s;",
        (sha,)
    ).fetchone()
    if row:
        return row["id"]

    row = conn.execute(
        """
        INSERT INTO article (url, canonical_url, title, publisher, published_at, sha256, raw_html, clean_text)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (url, canonical, title, publisher, published_at, sha, raw_html, clean_text)
    ).fetchone()
    return row["id"] if row else None


def extract_clean_text(page_url: str, html: str | None) -> str | None:
    try:
        if not html:
            return None
        # Trafilatura extraction
        return trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            url=page_url
        )
    except Exception as e:
        log.warning(f"trafilatura extract failed: {e}")
        return None


def fetch_html(url: str) -> str | None:
    try:
        r = http.get(url, timeout=20)
        if "text/html" in (r.headers.get("Content-Type") or "") and r.status_code < 400:
            return r.text
    except Exception as e:
        log.warning(f"fetch_html failed: {e}")
    return None


def ingest_google_news(conn, minutes: int) -> tuple[int, int]:
    """
    Returns (found_urls, inserted).
    """
    now_utc = datetime.now(timezone.utc)
    since = now_utc - timedelta(minutes=minutes)

    wl = conn.execute(
        "SELECT id, symbol, company FROM watchlist WHERE enabled = TRUE ORDER BY id;"
    ).fetchall()

    found = 0
    inserted = 0

    for w in wl:
        feed_url = rss_url_for(w["company"], w["symbol"])
        try:
            resp = http.get(feed_url, timeout=20)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except Exception as e:
            log.warning(f"Feed fetch failed for {feed_url}: {e}")
            continue

        for entry in feed.entries:
            found += 1
            link = entry.get("link") or ""
            if not link:
                continue
            # Resolve GN → publisher
            resolved = resolve_google_news(link)

            # Timestamp
            ts = None
            if "published_parsed" in entry and entry.published_parsed:
                ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif "updated_parsed" in entry and entry.updated_parsed:
                ts = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            elif entry.get("published"):
                try:
                    ts = dateparser.parse(entry.published)
                    if ts and not ts.tzinfo:
                        ts = ts.replace(tzinfo=timezone.utc)
                except Exception:
                    ts = None

            if ts and ts < since:
                continue  # older than window

            title = entry.get("title")
            publisher = None
            if hasattr(entry, "source") and getattr(entry.source, "title", None):
                publisher = entry.source.title
            elif entry.get("author"):
                publisher = entry.get("author")

            # Fetch + extract
            html = fetch_html(resolved)
            clean = extract_clean_text(resolved, html)

            article_id = upsert_article(
                conn, link, resolved, title, publisher, ts, html, clean
            )
            if article_id:
                inserted += 1
                # Tag to watchlist
                conn.execute(
                    """
                    INSERT INTO article_tag (article_id, watchlist_id)
                    VALUES (%s, %s)
                    ON CONFLICT (article_id, watchlist_id) DO NOTHING;
                    """,
                    (article_id, w["id"])
                )

    # Quick cleanup of image/CDN rows if anything slipped
    conn.execute("DELETE FROM article WHERE domain LIKE 'lh%.googleusercontent.com';")
    conn.execute("DELETE FROM article WHERE url ~* '\\.(png|jpe?g|gif|webp)(\\?.*)?$';")

    return found, inserted


# -----------------------------------------------------------------------------
# Summaries + Mail
# -----------------------------------------------------------------------------
def summarize_text(text: str, title: str | None = None) -> str:
    """
    Use Chat Completions with gpt-5-mini; DO NOT pass temperature or max_tokens.
    """
    if not text:
        text = "No article body was available; please summarize from the title and source context only."

    system_msg = (
        "You are QuantBrief. Write one crisp, factual, finance-focused bullet (25–45 words). "
        "Lead with the action (what changed). Avoid hype, avoid advice. If numbers are present, keep 1–2 key ones."
    )
    user_msg = f"Title: {title or ''}\n\nArticle:\n{text[:8000]}"  # keep prompt modest

    resp = _openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def summarize_new(conn, minutes: int) -> int:
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    rows = conn.execute(
        """
        SELECT a.id, a.title, a.clean_text
        FROM article a
        LEFT JOIN article_nlp n ON n.article_id = a.id
        WHERE n.article_id IS NULL
          AND (a.published_at IS NULL OR a.published_at >= %s)
        ORDER BY a.created_at DESC
        LIMIT 25;
        """,
        (since,)
    ).fetchall()

    done = 0
    for r in rows:
        try:
            summary = summarize_text(r["clean_text"] or "", r["title"])
            conn.execute(
                "INSERT INTO article_nlp (article_id, model, summary) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
                (r["id"], OPENAI_MODEL, summary)
            )
            done += 1
        except Exception as e:
            log.warning(f"Summarize failed for article {r['id']}: {e}")
    return done


def render_email(conn, minutes: int) -> tuple[str, int]:
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    rows = conn.execute(
        """
        SELECT a.id, a.title, coalesce(a.canonical_url, a.url) AS link, a.publisher, a.published_at, n.summary
        FROM article a
        LEFT JOIN article_nlp n ON n.article_id = a.id
        WHERE (a.published_at IS NULL OR a.published_at >= %s)
        ORDER BY a.published_at DESC NULLS LAST, a.id DESC
        LIMIT 60;
        """,
        (since,)
    ).fetchall()

    tmpl = Template("""
    <div style="font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;line-height:1.4">
      <h2 style="margin:0 0 8px">QuantBrief Daily — {{ now_local }}</h2>
      <div style="color:#666;margin-bottom:14px">Window: last {{ minutes }} minutes ending {{ now_local }}</div>
      {% if items|length == 0 %}
        <p>No items found in the window.</p>
      {% else %}
        <ul style="padding-left:18px">
          {% for it in items %}
            <li style="margin-bottom:10px">
              <a href="{{ it.link }}" style="text-decoration:none;color:#0a58ca">{{ it.title or it.link }}</a>
              {% if it.publisher %}<span style="color:#666"> — {{ it.publisher }}</span>{% endif %}
              {% if it.summary %}<div style="margin-top:4px">{{ it.summary }}</div>{% endif %}
            </li>
          {% endfor %}
        </ul>
      {% endif %}
      <div style="margin-top:18px;color:#888;font-size:12px">Sources are links only. We don’t republish paywalled content.</div>
    </div>
    """.strip())

    now_local = datetime.now(TZINFO).strftime("%Y-%m-%d %H:%M")
    html = tmpl.render(now_local=now_local, minutes=minutes, items=rows)
    return html, len(rows)


def send_mailgun(to_list: list[str], subject: str, html: str) -> bool:
    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN and MAILGUN_FROM):
        log.warning("Mailgun env vars missing, skipping send.")
        return False
    try:
        r = requests.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={
                "from": MAILGUN_FROM,
                "to": to_list,
                "subject": subject,
                "html": html,
            },
            timeout=20,
        )
        if r.status_code < 300:
            return True
        log.warning(f"Mailgun send failed {r.status_code}: {r.text}")
    except Exception as e:
        log.warning(f"Mailgun exception: {e}")
    return False


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
def admin_ok(request: Request):
    tok = request.headers.get("x-admin-token") or request.headers.get("X-Admin-Token")
    if not tok or tok != ADMIN_TOKEN:
        raise HTTPException(403, "Forbidden")


@app.get("/admin/health", response_class=PlainTextResponse)
def admin_health(request: Request):
    admin_ok(request)
    out = []

    env_ok = {
        "DATABASE_URL": bool(DATABASE_URL),
        "OPENAI_API_KEY": bool(OPENAI_API_KEY),
        "OPENAI_MODEL": bool(OPENAI_MODEL),
        "OPENAI_MAX_OUTPUT_TOKENS": bool(OPENAI_MAX_OUTPUT_TOKENS),
        "MAILGUN_API_KEY": bool(MAILGUN_API_KEY),
        "MAILGUN_DOMAIN": bool(MAILGUN_DOMAIN),
        "MAILGUN_FROM": bool(MAILGUN_FROM),
        "OWNER_EMAIL": bool(os.getenv("OWNER_EMAIL")),
        "ADMIN_TOKEN": bool(ADMIN_TOKEN),
    }
    out.append("env\n---")
    out.append(json.dumps(env_ok, indent=2))

    with db() as conn:
        ensure_schema(conn)
        counts = {
            "articles": conn.execute("SELECT COUNT(*) c FROM article;").fetchone()["c"],
            "summarized": conn.execute("SELECT COUNT(*) c FROM article_nlp;").fetchone()["c"],
            "recipients": conn.execute("SELECT COUNT(*) c FROM recipients;").fetchone()["c"],
            "watchlist": conn.execute("SELECT COUNT(*) c FROM watchlist;").fetchone()["c"],
        }
    out.append("\ncounts\n------")
    out.extend([f"{k}={v}" for k, v in counts.items()])

    return "\n".join(out)


@app.get("/admin/test_openai", response_class=PlainTextResponse)
def admin_test_openai(request: Request):
    admin_ok(request)
    try:
        msg = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Reply with 'OK'"}],
        )
        sample = (msg.choices[0].message.content or "").strip()
        return f"   ok model      sample\n   -- -----      ------\nTrue {OPENAI_MODEL} {sample[:64]}"
    except Exception as e:
        return f"   ok model      err\n   -- -----      ---\nFalse {OPENAI_MODEL} {str(e)[:160]}"


@app.post("/admin/init", response_class=PlainTextResponse)
def admin_init(request: Request):
    admin_ok(request)
    with db() as conn:
        ensure_schema(conn)
        seed_basics(conn)
    return "Initialized."


@app.post("/cron/ingest", response_class=JSONResponse)
def cron_ingest(request: Request, minutes: int = 1440):
    admin_ok(request)
    with db() as conn:
        ensure_schema(conn)
        found, ins = ingest_google_news(conn, minutes=minutes)
    return {"ok": True, "found_urls": found, "inserted": ins}


@app.post("/cron/digest", response_class=JSONResponse)
def cron_digest(request: Request, minutes: int = 1440):
    admin_ok(request)
    started = datetime.now(timezone.utc)
    with db() as conn:
        ensure_schema(conn)
        summarized = summarize_new(conn, minutes=minutes)
        html, items = render_email(conn, minutes=minutes)
        # enabled recipients
        recs = conn.execute(
            "SELECT email FROM recipients WHERE enabled = TRUE ORDER BY id;"
        ).fetchall()
        to_list = [r["email"] for r in recs]

        subject = f"QuantBrief Daily — {datetime.now(TZINFO).strftime('%Y-%m-%d')}"
        sent_ok = send_mailgun(to_list, subject, html)

        conn.execute(
            """
            INSERT INTO delivery_log (run_date, run_started_at, run_ended_at, sent_to, items, summarized)
            VALUES (CURRENT_DATE, %s, NOW(), %s, %s, %s);
            """,
            (started, ",".join(to_list), items, summarized)
        )

    return {
        "ok": bool(sent_ok),
        "sent_to": ",".join(to_list),
        "items": items,
        "summarized": summarized
    }


@app.get("/", response_class=PlainTextResponse)
def root():
    return "QuantBrief: OK"
