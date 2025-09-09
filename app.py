import os
import re
import ssl
import smtplib
import logging
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
from urllib.parse import urlparse
from datetime import datetime, timezone, timedelta

import psycopg
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
import feedparser

# -----------------------------------------------------------------------------
# Config & Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger("quantbrief")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
DIGEST_TO = os.getenv("DIGEST_TO", ADMIN_EMAIL or "")
EMAIL_FROM = os.getenv("EMAIL_FROM") or os.getenv("SMTP_FROM") or os.getenv("SMTP_USERNAME") or "no-reply@example.com"

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "1") not in ("0", "false", "False", "")

DEFAULT_MINUTES = int(os.getenv("DEFAULT_MINUTES", "10080"))  # 7 days
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.55"))
DEFAULT_RETAIN_DAYS = int(os.getenv("DEFAULT_RETAIN_DAYS", "30"))
MAX_FETCH_PER_RUN = int(os.getenv("MAX_FETCH_PER_RUN", "200"))

DB_DSN = os.getenv("DATABASE_URL") or os.getenv("DB_DSN") or ""

# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------
def db_connect():
    if not DB_DSN:
        # Fallback to PG* envs
        dsn = f"host={os.getenv('PGHOST','localhost')} port={os.getenv('PGPORT','5432')} dbname={os.getenv('PGDATABASE','postgres')} user={os.getenv('PGUSER','postgres')} password={os.getenv('PGPASSWORD','')}"
    else:
        dsn = DB_DSN
    conn = psycopg.connect(dsn)
    conn.autocommit = True
    return conn

def exec_sql_batch(conn, sql: str):
    with conn.cursor() as cur:
        cur.execute(sql)

# -----------------------------------------------------------------------------
# Schema & Migrations
# -----------------------------------------------------------------------------
SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS source_feed (
  id            BIGSERIAL PRIMARY KEY,
  url           TEXT NOT NULL UNIQUE,
  name          TEXT,
  language      TEXT NOT NULL DEFAULT 'en',
  retain_days   INT  NOT NULL DEFAULT 30,
  active        BOOLEAN NOT NULL DEFAULT TRUE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS found_url (
  id            BIGSERIAL PRIMARY KEY,
  url           TEXT NOT NULL UNIQUE,
  title         TEXT,
  feed_id       BIGINT NOT NULL REFERENCES source_feed(id) ON DELETE CASCADE,
  language      TEXT,
  published_at  TIMESTAMPTZ,
  found_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  -- Newer columns may be added by MIGRATIONS_SQL:
  -- host TEXT, article_type TEXT, score DOUBLE PRECISION, quality_notes TEXT
  CONSTRAINT uq_found_url_url UNIQUE (url)
);

-- updated_at maintenance
CREATE OR REPLACE FUNCTION set_source_feed_updated_at()
RETURNS TRIGGER AS $FN$
BEGIN
  NEW.updated_at := NOW();
  RETURN NEW;
END
$FN$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_source_feed_updated_at ON source_feed;
CREATE TRIGGER trg_source_feed_updated_at
BEFORE INSERT OR UPDATE ON source_feed
FOR EACH ROW EXECUTE FUNCTION set_source_feed_updated_at();

CREATE INDEX IF NOT EXISTS idx_found_url_published_at ON found_url (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_found_url_found_at ON found_url (found_at DESC);
"""

MIGRATIONS_SQL = r"""
DO $$
BEGIN
  -- found_url.host
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='found_url' AND column_name='host'
  ) THEN
    ALTER TABLE found_url ADD COLUMN host TEXT;
    UPDATE found_url
       SET host = lower(regexp_replace(substring(url from '://([^/]+)'), '^www\.', ''))
     WHERE host IS NULL AND url IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_found_url_host ON found_url (host);
  END IF;

  -- found_url.article_type
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='found_url' AND column_name='article_type'
  ) THEN
    ALTER TABLE found_url ADD COLUMN article_type TEXT;
  END IF;

  -- found_url.score
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='found_url' AND column_name='score'
  ) THEN
    ALTER TABLE found_url ADD COLUMN score DOUBLE PRECISION;
    CREATE INDEX IF NOT EXISTS idx_found_url_score ON found_url (score DESC);
  END IF;

  -- found_url.quality_notes
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='found_url' AND column_name='quality_notes'
  ) THEN
    ALTER TABLE found_url ADD COLUMN quality_notes TEXT;
  END IF;

  -- found_url.published_at (if legacy DB lacked it)
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='found_url' AND column_name='published_at'
  ) THEN
    ALTER TABLE found_url ADD COLUMN published_at TIMESTAMPTZ;
    CREATE INDEX IF NOT EXISTS idx_found_url_published_at ON found_url (published_at DESC);
  END IF;

  -- found_url.found_at (if legacy DB lacked it)
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='found_url' AND column_name='found_at'
  ) THEN
    ALTER TABLE found_url ADD COLUMN found_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
    CREATE INDEX IF NOT EXISTS idx_found_url_found_at ON found_url (found_at DESC);
  END IF;

  -- source_feed.updated_at
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='source_feed' AND column_name='updated_at'
  ) THEN
    ALTER TABLE source_feed ADD COLUMN updated_at TIMESTAMPTZ;
  END IF;

  -- source_feed.language (hard default 'en')
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='source_feed' AND column_name='language'
  ) THEN
    ALTER TABLE source_feed ADD COLUMN language TEXT;
    UPDATE source_feed SET language = 'en' WHERE language IS NULL;
    ALTER TABLE source_feed ALTER COLUMN language SET DEFAULT 'en';
    ALTER TABLE source_feed ALTER COLUMN language SET NOT NULL;
  END IF;

END $$;
"""

# -----------------------------------------------------------------------------
# Seeding
# -----------------------------------------------------------------------------
DEFAULT_FEEDS = [
    # Your four TLN / Talen Energy Google News feeds
    ("Google News: Talen Energy OR TLN (3d, filtered)",
     "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+-site:newser.com+-site:marketbeat.com+when:3d&hl=en-US&gl=US&ceid=US:en",
     "en", 30),

    ("Google News: Talen Energy OR TLN (all)",
     "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN&hl=en-US&gl=US&ceid=US:en",
     "en", 30),

    ("Google News: ((Talen Energy) OR TLN) (7d, filtered A)",
     "https://news.google.com/rss/search?q=%28Talen+Energy%29+OR+TLN+-site%3Anewser.com+-site%3Awww.marketbeat.com+-site%3Amarketbeat.com+-site%3Awww.newser.com+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen",
     "en", 30),

    ("Google News: ((Talen Energy) OR TLN) (7d, filtered B)",
     "https://news.google.com/rss/search?q=%28%28Talen+Energy%29+OR+TLN%29+-site%3Anewser.com+-site%3Awww.newser.com+-site%3Amarketbeat.com+-site%3Awww.marketbeat.com+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen",
     "en", 30),
]

def upsert_feed(conn, url: str, name: str, language: str = "en", retain_days: int = 30):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO source_feed (url, name, language, retain_days, active)
            VALUES (%s, %s, %s, %s, TRUE)
            ON CONFLICT (url) DO UPDATE
               SET name = EXCLUDED.name,
                   language = EXCLUDED.language,
                   retain_days = EXCLUDED.retain_days,
                   active = TRUE
            RETURNING id
            """,
            (url, name, language, retain_days),
        )
        return cur.fetchone()[0]

def seed_default_feeds(conn):
    for name, url, language, retain_days in DEFAULT_FEEDS:
        upsert_feed(conn, url=url, name=name, language=language, retain_days=retain_days)
    LOG.info("Schema ensured; seed feeds upserted.")

def list_active_feeds(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, url, name, COALESCE(NULLIF(language,''),'en') AS lang,
                   COALESCE(NULLIF(retain_days,0), %s) AS retain_days
            FROM source_feed
            WHERE active = TRUE
            ORDER BY id ASC
            """,
            (DEFAULT_RETAIN_DAYS,),
        )
        return cur.fetchall()

# -----------------------------------------------------------------------------
# Quality scoring (soft heuristics, no hard blocks)
# -----------------------------------------------------------------------------
RE_POSITIVE = re.compile(
    r"(hedge\s*fund\s*letter|invest(or|ment)\s*letter|quarterly\s*(letter|commentary)|"
    r"13f|10-q|10-k|annual\s*report|earnings\s*(call|transcript)|"
    r"industry\s*(outlook|insight|analysis)|white\s*paper)",
    re.I,
)

RE_NEGATIVE = re.compile(
    r"(hiring|jobs?|job\s+posting|apply\s+now|linkedin|tiktok|facebook|reddit|youtube|"
    r"forum|message\s*board|coupon|promo|subscribe|"
    r"stock\s*forum|options\s*chain|chart|price\s*target|"
    r"ai\s*generated\s*(summary|content))",
    re.I,
)

RE_PRESS = re.compile(r"(press\s*release|pr\s+newswire|globe\s*newswire)", re.I)

HOST_REPUTATION = {
    # soft boosts
    "seekingalpha.com": 0.10,
    "bloomberg.com": 0.12,
    "reuters.com": 0.12,
    "ft.com": 0.12,
    "wsj.com": 0.12,
    "heatmap.news": 0.08,
    "powermag.com": 0.06,
    "power-eng.com": 0.06,
    "bisnow.com": 0.05,
    "politico.com": 0.06,
    "eedition" : -0.05,  # example pattern
    # soft penalties
    "msn.com": -0.18,
    "finance.yahoo.com": -0.15,
    "uk.finance.yahoo.com": -0.15,
    "ca.finance.yahoo.com": -0.15,
    "marketbeat.com": -0.35,
    "investing.com": -0.08,
    "tradingview.com": -0.05,
    "youtube.com": -0.30,
    "linkedin.com": -0.25,
    "facebook.com": -0.30,
    "tiktok.com": -0.40,
}

def normalize_host(url: str) -> str:
    try:
        h = urlparse(url).hostname or ""
        h = h.lower()
        if h.startswith("www."):
            h = h[4:]
        return h
    except Exception:
        return ""

def classify_article(title: str, host: str) -> str:
    t = title.lower()
    if RE_PRESS.search(t) or "press" in t and "release" in t:
        return "press_release"
    if any(s in host for s in ("linkedin.com", "facebook.com", "tiktok.com", "reddit.com", "youtube.com")):
        return "social_ugc"
    if any(s in host for s in ("smartrecruiters.com", "indeed.com", "glassdoor.", "greenhouse.io")) or re.search(r"\b(hiring|apply|careers?)\b", t):
        return "job_posting"
    if re.search(r"\b(presentation|deck|slides?)\b", t):
        return "investor_material"
    if RE_POSITIVE.search(t):
        return "analysis_insight"
    return "news_article"

def quality_score(title: str, summary: str, host: str, url: str) -> tuple[float, str]:
    base = 0.50
    notes = []

    t = title or ""
    s = summary or ""

    # Title / summary signals
    if RE_POSITIVE.search(t) or RE_POSITIVE.search(s):
        base += 0.18
        notes.append("positive_keywords")
    if RE_NEGATIVE.search(t) or RE_NEGATIVE.search(s):
        base -= 0.18
        notes.append("negative_keywords")

    # Host reputation (soft)
    delta = HOST_REPUTATION.get(host, 0.0)
    if delta != 0.0:
        base += delta
        notes.append(f"host_reputation({delta:+.2f})")

    # Domain type boosts/penalties
    if host.endswith(".gov") or host.endswith(".edu"):
        base += 0.10
        notes.append("gov/edu_boost")
    if any(seg in url for seg in ("/press/", "/press-release", "/news-releases/")):
        base -= 0.12
        notes.append("press_path_penalty")

    # Clamp
    score = max(0.0, min(1.0, base))
    return score, ", ".join(notes)

# -----------------------------------------------------------------------------
# Email
# -----------------------------------------------------------------------------
def send_email(subject: str, html: str, to_addrs: str | list[str]):
    if not (SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD and EMAIL_FROM):
        LOG.error("SMTP not configured. Need SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD and EMAIL_FROM/SMTP_FROM.")
        return False

    if isinstance(to_addrs, str):
        recipients = [x.strip() for x in to_addrs.split(",") if x.strip()]
    else:
        recipients = to_addrs

    msg = MIMEText(html, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(recipients)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
        if SMTP_STARTTLS:
            s.starttls(context=context)
        if SMTP_USERNAME:
            s.login(SMTP_USERNAME, SMTP_PASSWORD)
        # Envelope from must be a bare address; if EMAIL_FROM has a display name, extract the addr
        envelope_from = EMAIL_FROM
        if "<" in EMAIL_FROM and ">" in EMAIL_FROM:
            envelope_from = EMAIL_FROM.split("<", 1)[1].split(">", 1)[0].strip()
        s.sendmail(envelope_from, recipients, msg.as_string())
    LOG.info("Email sent to %s (subject='%s')", recipients, subject)
    return True

# -----------------------------------------------------------------------------
# Ingest / Digest
# -----------------------------------------------------------------------------
def ensure_schema_and_seed(conn):
    exec_sql_batch(conn, SCHEMA_SQL)
    exec_sql_batch(conn, MIGRATIONS_SQL)
    seed_default_feeds(conn)

def parse_published(entry) -> datetime:
    # feedparser gives published_parsed or updated_parsed as time.struct_time
    dt = None
    if getattr(entry, "published_parsed", None):
        dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    elif getattr(entry, "updated_parsed", None):
        dt = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    return dt

def insert_found_url(conn, url: str, title: str, feed_id: int, published_at: datetime,
                     host: str, article_type: str, score: float, quality_notes: str, language: str = "en") -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO found_url (url, title, feed_id, language, published_at, found_at, host, article_type, score, quality_notes)
            VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE
               SET title = EXCLUDED.title,
                   feed_id = EXCLUDED.feed_id,
                   language = EXCLUDED.language,
                   published_at = EXCLUDED.published_at,
                   host = EXCLUDED.host,
                   article_type = EXCLUDED.article_type,
                   score = EXCLUDED.score,
                   quality_notes = EXCLUDED.quality_notes,
                   found_at = GREATEST(found_url.found_at, NOW())
            """,
            (url, title, feed_id, language, published_at, host, article_type, score, quality_notes),
        )
    return True

def prune_old_found_urls(conn, default_days: int = DEFAULT_RETAIN_DAYS) -> int:
    with conn.cursor() as cur:
        # Use per-feed retain_days when present, else default
        cur.execute(
            """
            WITH cutoff AS (
              SELECT f.id,
                     (COALESCE(NULLIF(f.retain_days,0), %s)::text || ' days')::interval AS keep_for
              FROM source_feed f
              WHERE f.active = TRUE
            )
            DELETE FROM found_url u
            USING source_feed s, cutoff c
            WHERE u.feed_id = s.id
              AND s.id = c.id
              AND u.found_at < NOW() - c.keep_for
            RETURNING u.id
            """,
            (default_days,)
        )
        deleted = cur.rowcount or 0
    LOG.warning("prune_old_found_urls: deleted=%d", deleted)
    return deleted

def fetch_and_ingest(conn, minutes: int = DEFAULT_MINUTES, min_score: float = DEFAULT_MIN_SCORE):
    feeds = list_active_feeds(conn)
    cutoff_dt = datetime.now(timezone.utc) - timedelta(minutes=minutes)

    inserted = 0
    scanned = 0

    for (feed_id, feed_url, feed_name, lang, retain_days) in feeds:
        LOG.info("parsed feed: %s", feed_url)
        parsed = feedparser.parse(feed_url)
        entries = parsed.entries or []
        LOG.info("feed entries: %d", len(entries))

        for e in entries[:MAX_FETCH_PER_RUN]:
            url = getattr(e, "link", None)
            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""

            if not url:
                continue

            published_at = parse_published(e)
            if published_at < cutoff_dt:
                continue

            host = normalize_host(url)
            a_type = classify_article(title, host)
            score, notes = quality_score(title, summary, host, url)

            scanned += 1
            if score < min_score:
                continue

            if insert_found_url(conn, url, title, feed_id, published_at, host, a_type, score, notes, lang):
                inserted += 1

    pruned = prune_old_found_urls(conn, DEFAULT_RETAIN_DAYS)
    LOG.info("ingest done: inserted=%d, fetched_scored=%d, pruned=%d", inserted, scanned, pruned)
    return inserted, scanned

def fetch_digest_rows(conn, minutes: int, min_score: float):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                f.title,
                f.url,
                f.host,
                f.article_type,
                COALESCE(f.score, 0) AS score,
                f.published_at,
                s.name AS feed_name
            FROM found_url f
            JOIN source_feed s ON s.id = f.feed_id
            WHERE f.published_at >= NOW() - (%s::text || ' minutes')::interval
              AND COALESCE(f.score, 0) >= %s
            ORDER BY f.published_at DESC
            LIMIT 500
            """,
            (minutes, min_score),
        )
        return cur.fetchall()

def render_digest_html(rows, minutes: int):
    lines = [f"<h2>Quantbrief digest — last {minutes} minutes</h2>", "<ul>"]
    for (title, url, host, a_type, score, published_at, feed_name) in rows:
        ts = published_at.isoformat() if published_at else ""
        lines.append(
            f"<li><a href='{url}'>{title}</a> &mdash; "
            f"<i>{feed_name}</i> • {ts} • score {score:.2f} • {host or ''} • {a_type or ''}</li>"
        )
    lines.append("</ul>")
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI()

def require_admin(req: Request):
    token = req.headers.get("x-admin-token") or req.headers.get("authorization", "").replace("Bearer ", "")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

@app.get("/", response_class=PlainTextResponse)
def root():
    return "Quantbrief Daily OK"

@app.post("/admin/init")
def admin_init(request: Request):
    require_admin(request)
    with db_connect() as conn:
        ensure_schema_and_seed(conn)
    return JSONResponse({"ok": True})

@app.post("/cron/ingest")
def cron_ingest(request: Request, minutes: int = DEFAULT_MINUTES, min_score: float = DEFAULT_MIN_SCORE):
    require_admin(request)
    with db_connect() as conn:
        ensure_schema_and_seed(conn)  # safe if already exists
        inserted, scanned = fetch_and_ingest(conn, minutes=minutes, min_score=min_score)
    return JSONResponse({"inserted": inserted, "scored": scanned})

@app.post("/cron/digest")
def cron_digest(request: Request, minutes: int = DEFAULT_MINUTES, min_score: float = DEFAULT_MIN_SCORE):
    require_admin(request)
    with db_connect() as conn:
        ensure_schema_and_seed(conn)
        rows = fetch_digest_rows(conn, minutes=minutes, min_score=min_score)

    html = render_digest_html(rows, minutes)
    recipients = DIGEST_TO or ADMIN_EMAIL
    ok = send_email(f"Quantbrief digest — last {minutes} minutes ({len(rows)} items)", html, recipients)
    return JSONResponse({"sent": ok, "items": len(rows)})

@app.post("/admin/test-email")
def admin_test_email(request: Request):
    require_admin(request)
    to_addrs = DIGEST_TO or ADMIN_EMAIL or SMTP_USERNAME or "test@example.com"
    html = "<b>It works</b>"
    ok = send_email("Quantbrief test email", html, to_addrs)
    return JSONResponse({"sent": ok, "to": to_addrs})
