import os
import re
import json
import time
import math
import smtplib
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import feedparser
import psycopg
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr

# ---------------------------
# Config & Logging
# ---------------------------

LOG = logging.getLogger("quantbrief")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)s:quantbrief:%(message)s",
)

DATABASE_URL = os.getenv("DATABASE_URL")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
APP_NAME = os.getenv("APP_NAME", "quantbrief")

DEFAULT_MINUTES = int(os.getenv("DEFAULT_MINUTES", "10080"))  # 7 days
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.55"))
DEFAULT_RETAIN_DAYS = int(os.getenv("DEFAULT_RETAIN_DAYS", "30"))
MAX_FETCH_PER_RUN = int(os.getenv("MAX_FETCH_PER_RUN", "200"))

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "1") not in ("0", "false", "False")
EMAIL_FROM = os.getenv("EMAIL_FROM")  # Display header (e.g., 'QuantBrief Daily <daily@mg.example>')
SMTP_FROM = os.getenv("SMTP_FROM")    # Envelope MAIL FROM (e.g., 'daily@mg.example')
DIGEST_TO = os.getenv("DIGEST_TO", os.getenv("ADMIN_EMAIL", ""))

app = FastAPI(title="Quantbrief")

# ---------------------------
# DB Helpers
# ---------------------------

def get_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(DATABASE_URL, autocommit=True)

def exec_sql(conn, sql: str):
    with conn.cursor() as cur:
        cur.execute(sql)

# ---------------------------
# Schema (idempotent)
# ---------------------------

SCHEMA_SQL = r"""
-- Create tables if missing (fresh installs)
DO $$
BEGIN
  CREATE TABLE IF NOT EXISTS source_feed (
    id           BIGSERIAL PRIMARY KEY,
    url          TEXT NOT NULL,
    name         TEXT,
    language     TEXT NOT NULL DEFAULT 'en',
    retain_days  INTEGER,
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
  );

  CREATE TABLE IF NOT EXISTS found_url (
    id            BIGSERIAL PRIMARY KEY,
    url           TEXT NOT NULL,
    title         TEXT,
    feed_id       BIGINT REFERENCES source_feed(id) ON DELETE CASCADE,
    language      TEXT NOT NULL DEFAULT 'en',
    published_at  TIMESTAMPTZ,
    found_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    score         DOUBLE PRECISION,
    meta          JSONB
  );
END
$$;

-- Bring existing installs up-to-date (idempotent migrations)
DO $$
BEGIN
  -- source_feed columns
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='source_feed' AND column_name='language'
  ) THEN
    ALTER TABLE source_feed ADD COLUMN language TEXT NOT NULL DEFAULT 'en';
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='source_feed' AND column_name='retain_days'
  ) THEN
    ALTER TABLE source_feed ADD COLUMN retain_days INTEGER;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='source_feed' AND column_name='updated_at'
  ) THEN
    ALTER TABLE source_feed ADD COLUMN updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
  END IF;

  -- found_url columns
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='found_url' AND column_name='language'
  ) THEN
    ALTER TABLE found_url ADD COLUMN language TEXT NOT NULL DEFAULT 'en';
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='found_url' AND column_name='published_at'
  ) THEN
    ALTER TABLE found_url ADD COLUMN published_at TIMESTAMPTZ;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='found_url' AND column_name='found_at'
  ) THEN
    ALTER TABLE found_url ADD COLUMN found_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='found_url' AND column_name='score'
  ) THEN
    ALTER TABLE found_url ADD COLUMN score DOUBLE PRECISION;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='found_url' AND column_name='meta'
  ) THEN
    ALTER TABLE found_url ADD COLUMN meta JSONB;
  END IF;
END
$$;

-- Indexes & uniques (safe to re-run)
CREATE UNIQUE INDEX IF NOT EXISTS uq_source_feed_url ON source_feed (url);
CREATE UNIQUE INDEX IF NOT EXISTS uq_found_url_url   ON found_url (url);
CREATE INDEX IF NOT EXISTS idx_found_url_found_at    ON found_url (found_at);
CREATE INDEX IF NOT EXISTS idx_found_url_published_at ON found_url (published_at);
CREATE INDEX IF NOT EXISTS idx_found_url_feed_id     ON found_url (feed_id);

-- updated_at trigger
CREATE OR REPLACE FUNCTION set_source_feed_updated_at()
RETURNS trigger AS $f$ BEGIN NEW.updated_at := NOW(); RETURN NEW; END $f$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_source_feed_updated ON source_feed;
CREATE TRIGGER trg_source_feed_updated
BEFORE UPDATE ON source_feed
FOR EACH ROW EXECUTE FUNCTION set_source_feed_updated_at();
"""

# ---------------------------
# Feed seeding / upsert
# ---------------------------

SEED_FEEDS: List[Tuple[str, str, str, Optional[int]]] = [
    # name, url, language, retain_days
    ("Google News: Talen Energy OR TLN (3d, filtered)",
     "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN+-site:newser.com+-site:marketbeat.com+when:3d&hl=en-US&gl=US&ceid=US:en",
     "en", 7),
    ("Google News: Talen Energy OR TLN (all)",
     "https://news.google.com/rss/search?q=(Talen+Energy)+OR+TLN&hl=en-US&gl=US&ceid=US:en",
     "en", 7),
    ("Google News: ((Talen Energy) OR TLN) (7d, filtered A)",
     "https://news.google.com/rss/search?q=%28Talen+Energy%29+OR+TLN+-site%3Anewser.com+-site%3Awww.marketbeat.com+-site%3Amarketbeat.com+-site%3Awww.newser.com+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen",
     "en", 10),
    ("Google News: ((Talen Energy) OR TLN) (7d, filtered B)",
     "https://news.google.com/rss/search?q=%28%28Talen+Energy%29+OR+TLN%29+-site%3Anewser.com+-site%3Awww.newser.com+-site%3Amarketbeat.com+-site%3Awww.marketbeat.com+when%3A7d&hl=en-US&gl=US&ceid=US%3Aen",
     "en", 10),
]

def ensure_schema_and_seed():
    with get_db() as conn:
        exec_sql(conn, SCHEMA_SQL)
        with conn.cursor() as cur:
            for name, url, language, retain_days in SEED_FEEDS:
                cur.execute(
                    """
                    INSERT INTO source_feed (url, name, language, retain_days, is_active)
                    VALUES (%s, %s, %s, %s, TRUE)
                    ON CONFLICT (url) DO UPDATE
                    SET name = EXCLUDED.name,
                        language = COALESCE(EXCLUDED.language, source_feed.language),
                        retain_days = COALESCE(EXCLUDED.retain_days, source_feed.retain_days),
                        is_active = TRUE
                    RETURNING id
                    """,
                    (url, name, language, retain_days),
                )
        LOG.info("Schema ensured; seed feeds upserted.")

def list_active_feeds() -> List[Dict[str, Any]]:
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, url, COALESCE(NULLIF(language,''), 'en') AS language,
                   COALESCE(retain_days, %s) AS retain_days,
                   COALESCE(NULLIF(name,''), url) AS name
            FROM source_feed
            WHERE is_active = TRUE
            ORDER BY id
            """,
            (DEFAULT_RETAIN_DAYS,),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "url": r[1],
            "language": r[2],
            "retain_days": r[3],
            "name": r[4],
        }
        for r in rows
    ]

# ---------------------------
# URL tools
# ---------------------------

UTM_PREFIXES = {"utm_", "igshid", "ocid", "cmpid", "_hs", "spm", "yclid", "gclid", "fbclid"}

def canonicalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        q = [(k, v) for (k, v) in parse_qsl(p.query, keep_blank_values=True)
             if not any(k.lower().startswith(pref) for pref in UTM_PREFIXES)]
        qstr = urlencode(q, doseq=True)
        # remove fragments
        new = p._replace(query=qstr, fragment="")
        # strip default ports
        netloc = new.netloc
        if netloc.endswith(":80"):
            netloc = netloc[:-3]
        elif netloc.endswith(":443"):
            netloc = netloc[:-4]
        new = new._replace(netloc=netloc)
        return urlunparse(new)
    except Exception:
        return url

def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host
    except Exception:
        return ""

# ---------------------------
# Quality Scoring
# ---------------------------

# Hard block (skip entirely)
BLOCK_DOMAINS = {
    "marketbeat.com", "www.marketbeat.com",
    "newser.com", "www.newser.com",
    "msn.com", "www.msn.com",
    "sg.finance.yahoo.com", "uk.finance.yahoo.com", "ca.finance.yahoo.com",
    "finance.yahoo.com", "au.finance.yahoo.com",
    "futubull", "moomoo", "futunn",  # handled via substring check too
    "blueocean-eg.com", "reefoasisdiveclub.com", "azzurra-redsea.com",
    "alumnimallrandgo.up.ac.za", "epayslip.grz.gov.zm",
    "js.signavitae.com", "tw13zhi.com", "zyrofisherb2b.co.uk",
    "valueinvestorinsight.com",  # lots of spam clones
    "dolphinworldegypt.com",
}

# Domain weights (boost credible sources)
SOURCE_WEIGHTS = {
    "reuters.com": 0.95,
    "wsj.com": 0.95,
    "ft.com": 0.95,
    "bloomberg.com": 0.95,
    "politico.com": 0.85,
    "eenews.net": 0.85,
    "heatmap.news": 0.85,
    "powermag.com": 0.85,
    "power-eng.com": 0.80,
    "datacenterdynamics.com": 0.85,
    "power-technology.com": 0.80,
    "utilitydive.com": 0.85,
    "seekingalpha.com": 0.80,     # articles ok; data pages filtered via title rules
    "law360.com": 0.80,
    "whitecase.com": 0.75,
    "vinson-elkins.com": 0.75,
    "investopedia.com": 0.75,
    "theglobeandmail.com": 0.80,
    "industrialinfo.com": 0.75,
    "esgdive.com": 0.75,
    "dcd.global": 0.85,           # Data Center Dynamics alt
    "talenenergy.com": 0.60,      # press release (neutral/ok)
    "wraltechwire.com": 0.70,
    "citizensvoice.com": 0.70,
    "wpxi.com": 0.65,
}

# Strong title/url filters (drop)
DROP_PATTERNS = [
    r"\b(job|jobs|hiring|careers?|apply|internship)\b",
    r"\b(options?\s+chain|option\s+chain)\b",
    r"\b(stock\s+forum|discussion)\b",
    r"\b(interactive\s+stock\s+chart)\b",
    r"\b(dividend\s+history|p\/e|peg\s+ratios?|analyst\s+estimates?)\b",
    r"\b(etf|holdings)\b",
    r"facebook\.com|linkedin\.com|indeed\.com|ziprecruiter\.com|smartrecruiters\.com|tiktok\.com|youtube\.com",
    r"\b(msn|yahoo\s+finance)\b",
]

# Soft keyword boosts for genuine insight / institutional commentary
BOOST_KEYWORDS = [
    r"\b(hedge fund|letter to investors|quarterly commentary|industry outlook)\b",
    r"\b(PJM|FERC|SMR|nuclear|data centers?|PPA|RMR|capacity market)\b",
    r"\b(Amazon|AWS|Susquehanna)\b",
    r"\b(analysis|opinion|deep dive)\b",
]

def regex_any(patterns: List[str], s: str) -> bool:
    s_l = s.lower()
    return any(re.search(p, s_l) for p in patterns)

def score_article(url: str, title: str, feed_name: str) -> float:
    d = domain_of(url)
    host = d
    bare = host.replace("www.", "")

    # Block by obvious junk domains (hard or substring)
    if any(tok in host for tok in ("futubull", "moomoo", "futunn")):
        return 0.0
    if bare in BLOCK_DOMAINS:
        return 0.0

    t = (title or "").strip()
    if len(t) < 5:
        return 0.0

    # Drop if title/url matches strong junk patterns
    hay = f"{t} {url}".lower()
    if regex_any(DROP_PATTERNS, hay):
        return 0.0

    # Base score from source reputation
    base = SOURCE_WEIGHTS.get(bare, 0.5)

    # Penalize obvious scraper / aggregator paths
    if any(seg in hay for seg in ("quote&", "/quote/", "stock-quote", "/prices/", "/interactive/")):
        base -= 0.2

    # Penalize many hyphens/commas/dates that suggest ticker pages
    hyphen_pen = min(t.count("-"), 4) * 0.03
    base -= hyphen_pen

    # Boost if looks like analysis/industry insight
    if regex_any(BOOST_KEYWORDS, hay):
        base += 0.15

    # Slight boost if from feed title indicating 'filtered'
    if "filtered" in (feed_name or "").lower():
        base += 0.05

    # Bound to [0, 1]
    return max(0.0, min(1.0, base))

# ---------------------------
# Email
# ---------------------------

def _from_header_and_envelope() -> Tuple[str, str]:
    # Envelope MAIL FROM must be a plain address; header can be Name <addr>
    header_from = EMAIL_FROM
    if not header_from:
        # fallback to username if set
        if SMTP_USERNAME:
            header_from = formataddr((APP_NAME, SMTP_USERNAME))
        elif SMTP_FROM:
            header_from = formataddr((APP_NAME, SMTP_FROM))
        else:
            header_from = f"{APP_NAME} <noreply@localhost>"

    env_from = SMTP_FROM
    if not env_from:
        addr = parseaddr(header_from)[1]
        env_from = addr or SMTP_USERNAME or "noreply@localhost"

    return header_from, env_from

def send_email(subject: str, html: str, to_addrs: List[str]) -> None:
    if not (SMTP_HOST and SMTP_PORT and SMTP_USERNAME and SMTP_PASSWORD):
        raise RuntimeError("SMTP not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD.")

    to_addrs = [a.strip() for a in to_addrs if a and a.strip()]
    if not to_addrs:
        raise RuntimeError("No recipient to send to.")

    header_from, envelope_from = _from_header_and_envelope()

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = header_from
    msg["To"] = ", ".join(to_addrs)
    msg.attach(MIMEText(html, "html", "utf-8"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
        if SMTP_STARTTLS:
            s.starttls()
        s.login(SMTP_USERNAME, SMTP_PASSWORD)
        s.sendmail(envelope_from, to_addrs, msg.as_string())

# ---------------------------
# Pruning
# ---------------------------

def prune_old_found_urls(default_days: int = DEFAULT_RETAIN_DAYS) -> int:
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH del AS (
              DELETE FROM found_url f
              USING source_feed s
              WHERE f.feed_id = s.id
                AND f.found_at < NOW() - make_interval(days => COALESCE(NULLIF(s.retain_days, 0), %s))
              RETURNING 1
            )
            SELECT count(*) FROM del
            """,
            (default_days,),
        )
        row = cur.fetchone()
        deleted = int(row[0]) if row and row[0] is not None else 0
    LOG.warning("prune_old_found_urls: deleted=%d", deleted)
    return deleted

# ---------------------------
# Ingest
# ---------------------------

def parse_time(entry: Dict[str, Any]) -> Optional[datetime]:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        if entry.get(key):
            try:
                # feedparser returns time.struct_time
                t = entry[key]
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                continue
    # Try 'published' string if present (feedparser sometimes parses it)
    return None

def upsert_found_url(row: Dict[str, Any]) -> None:
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO found_url (url, title, feed_id, language, published_at, found_at, score, meta)
            VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s)
            ON CONFLICT (url) DO UPDATE
            SET title = COALESCE(EXCLUDED.title, found_url.title),
                feed_id = COALESCE(EXCLUDED.feed_id, found_url.feed_id),
                language = COALESCE(EXCLUDED.language, found_url.language),
                published_at = COALESCE(EXCLUDED.published_at, found_url.published_at),
                score = GREATEST(COALESCE(found_url.score, 0), COALESCE(EXCLUDED.score, 0)),
                meta = COALESCE(found_url.meta, '{}'::jsonb) || COALESCE(EXCLUDED.meta, '{}'::jsonb)
            """,
            (
                row["url"],
                row.get("title"),
                row.get("feed_id"),
                row.get("language") or "en",
                row.get("published_at"),
                row.get("score") or 0.0,
                json.dumps(row.get("meta") or {}),
            ),
        )

def fetch_and_ingest(minutes: int = DEFAULT_MINUTES, min_score: float = DEFAULT_MIN_SCORE) -> Tuple[int, int]:
    feeds = list_active_feeds()
    cutoff_dt = datetime.now(timezone.utc) - timedelta(minutes=minutes)

    inserted = 0
    scanned = 0

    for f in feeds:
        url = f["url"]
        name = f["name"]
        lang = f.get("language") or "en"
        feed_id = f["id"]

        try:
            parsed = feedparser.parse(url)
            entries = parsed.entries[:MAX_FETCH_PER_RUN]
            LOG.info("parsed feed: %s", url)
            LOG.info("feed entries: %d", len(entries))
        except Exception as e:
            LOG.warning("failed to parse feed %s: %s", url, e)
            continue

        for e in entries:
            scanned += 1
            link = e.get("link") or e.get("id")
            title = (e.get("title") or "").strip()
            if not link:
                continue

            link = canonicalize_url(link)
            d = domain_of(link)

            # Hard domain blocks
            bare = d.replace("www.", "")
            if bare in BLOCK_DOMAINS or any(tok in d for tok in ("futubull", "moomoo", "futunn")):
                continue

            published_at = parse_time(e)
            # Drop outside window when we have a timestamp
            if published_at and published_at < cutoff_dt:
                continue

            score = score_article(link, title, name)
            if score < min_score:
                continue

            row = {
                "url": link,
                "title": title,
                "feed_id": feed_id,
                "language": lang,
                "published_at": published_at,
                "score": score,
                "meta": {
                    "source": name,
                    "domain": d,
                },
            }

            try:
                upsert_found_url(row)
                inserted += 1
            except Exception as ex:
                # ignore dup races etc.
                LOG.warning("ingest upsert error (%s): %s", link, ex)

    return inserted, scanned

# ---------------------------
# Digest
# ---------------------------

def fetch_digest_rows(minutes: int, min_score: float) -> List[Tuple[str, str, Optional[datetime], str, float]]:
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.url, COALESCE(NULLIF(f.title,''), f.url) AS title,
                   f.published_at, COALESCE(NULLIF(s.name,''), s.url) AS feed_name,
                   COALESCE(f.score, 0)
            FROM found_url f
            JOIN source_feed s ON s.id = f.feed_id
            WHERE f.found_at >= NOW() - make_interval(mins => %s)
              AND (f.published_at IS NULL OR f.published_at >= NOW() - make_interval(mins => %s))
              AND COALESCE(f.score, 0) >= %s
            ORDER BY f.score DESC, COALESCE(f.published_at, f.found_at) DESC
            LIMIT 400
            """,
            (minutes, minutes, min_score),
        )
        return cur.fetchall()

def render_digest_html(rows: List[Tuple[str, str, Optional[datetime], str, float]], minutes: int) -> str:
    items = []
    for url, title, published_at, feed_name, score in rows:
        ts = published_at.isoformat() if published_at else ""
        items.append(f"<li><a href='{url}'>{title}</a> "
                     f"<small>({feed_name}{' • ' + ts if ts else ''} • score {score:.2f})</small></li>")
    return f"""
    <h2>Quantbrief digest — last {minutes} minutes</h2>
    <ol>
      {''.join(items)}
    </ol>
    """

# ---------------------------
# Auth
# ---------------------------

def check_admin(request: Request):
    tok = request.headers.get("x-admin-token") or request.headers.get("authorization", "").replace("Bearer ", "")
    if not ADMIN_TOKEN or tok != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------------------------
# Routes
# ---------------------------

@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK"

@app.post("/admin/init")
def admin_init(request: Request):
    check_admin(request)
    ensure_schema_and_seed()
    return {"ok": True}

@app.post("/cron/ingest")
def cron_ingest(request: Request, minutes: Optional[int] = None, min_score: Optional[float] = None):
    check_admin(request)
    minutes = minutes or DEFAULT_MINUTES
    min_score = min_score if min_score is not None else DEFAULT_MIN_SCORE

    t0 = time.time()
    inserted, scanned = fetch_and_ingest(minutes=minutes, min_score=min_score)
    deleted = prune_old_found_urls(DEFAULT_RETAIN_DAYS)
    took = time.time() - t0
    LOG.info("ingest done: inserted=%d, fetched_scored=%d, pruned=%d, took=%.2fs",
             inserted, scanned, deleted, took)
    return {"ok": True, "inserted": inserted, "scanned": scanned, "pruned": deleted, "took_sec": took}

@app.post("/cron/digest")
def cron_digest(request: Request, minutes: Optional[int] = None, min_score: Optional[float] = None):
    check_admin(request)
    minutes = minutes or DEFAULT_MINUTES
    min_score = min_score if min_score is not None else DEFAULT_MIN_SCORE

    rows = fetch_digest_rows(minutes, min_score)
    html = render_digest_html(rows, minutes)
    subject = f"Quantbrief digest — last {minutes} minutes ({len(rows)} items)"

    to_env = DIGEST_TO
    if not to_env:
        LOG.error("DIGEST_TO/ADMIN_EMAIL not set; cannot send digest")
        return {"ok": False, "error": "DIGEST_TO not set", "count": len(rows)}

    recipients = [a.strip() for a in to_env.split(",") if a.strip()]
    try:
        send_email(subject, html, recipients)
        LOG.info("Email sent to %s (subject='%s')", recipients, subject)
        return {"ok": True, "sent_to": recipients, "count": len(rows)}
    except Exception as e:
        LOG.error("send_email failed: %s", e)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

@app.post("/admin/test-email")
def admin_test_email(request: Request):
    check_admin(request)
    to_env = DIGEST_TO or os.getenv("ADMIN_EMAIL")
    if not to_env:
        raise HTTPException(status_code=400, detail="Set DIGEST_TO or ADMIN_EMAIL env var so I know where to send the test email.")
    to_addrs = [a.strip() for a in to_env.split(",") if a.strip()]
    html = "<b>It works</b> — Quantbrief can send email via your SMTP settings."
    try:
        send_email("Quantbrief test email", html, to_addrs)
        LOG.info("Email sent to %s (subject='Quantbrief test email')", to_addrs)
        return {"ok": True, "to": to_addrs}
    except Exception as e:
        LOG.error("send_email failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"send_email failed: {e}")
