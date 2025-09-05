# app.py
import os, sys, re, json, time, hmac, base64, hashlib
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode, urlsplit, urlunsplit, quote_plus

import requests
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
from jinja2 import Template

# ----------- Env & Constants -----------

DATABASE_URL          = os.getenv("DATABASE_URL")
ADMIN_TOKEN           = os.getenv("ADMIN_TOKEN", "changeme-admin")
TIMEZONE              = os.getenv("TIMEZONE", "America/Toronto")
MAILGUN_DOMAIN        = os.getenv("MAILGUN_DOMAIN")          # e.g. mg.quantbrief.ca
MAILGUN_API_KEY       = os.getenv("MAILGUN_API_KEY")
MAILGUN_FROM          = os.getenv("MAILGUN_FROM", "QuantBrief Daily <daily@mg.quantbrief.ca>")

GOOGLE_CSE_ID         = os.getenv("GOOGLE_CSE_ID")           # optional but recommended
GOOGLE_API_KEY        = os.getenv("GOOGLE_API_KEY")          # optional but recommended
MAX_CSE_RESULTS       = int(os.getenv("MAX_CSE_RESULTS", "25"))

OPENAI_MODEL          = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_MAX_OUTPUT     = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "700"))
MAX_SUMMARIES_PER_RUN = int(os.getenv("MAX_SUMMARIES_PER_RUN", "40"))

OWNER_EMAIL           = os.getenv("OWNER_EMAIL", "quantbrief.research@gmail.com")

MAILGUN_API_BASE      = "https://api.mailgun.net"

BLOCKED_HOSTS = {
    "news.google.com",
    "lh3.googleusercontent.com",
    "googleusercontent.com",
    "gstatic.com",
    "maps.google.com",
    "news.yahoo.com",
}

OK_CONTENT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")

# ----------- FastAPI -----------

app = FastAPI()

# ----------- DB Helpers -----------

def db():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def init_db():
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
        create table if not exists equity (
            id serial primary key,
            ticker text unique not null,
            name   text
        );
        create table if not exists subscriber (
            id serial primary key,
            email text unique not null,
            tz    text default 'America/Toronto',
            send_hour int default 8,
            send_min  int default 30
        );
        create table if not exists subscriber_watch (
            subscriber_id int references subscriber(id) on delete cascade,
            equity_id     int references equity(id) on delete cascade,
            primary key (subscriber_id, equity_id)
        );
        create table if not exists article (
            id bigserial primary key,
            url           text not null,
            canonical_url text,
            title         text,
            publisher     text,
            published_at  timestamptz,
            clean_text    text,
            source        text,           -- 'google_cse', etc
            created_at    timestamptz default now()
        );
        create index if not exists idx_article_pubtime on article(published_at desc);
        create unique index if not exists ux_article_canon on article((coalesce(canonical_url,url)));

        create table if not exists article_tag (
            article_id bigint references article(id) on delete cascade,
            equity_id  int references equity(id) on delete cascade,
            tag_kind   text,  -- 'company','peer','industry'
            primary key (article_id, equity_id, tag_kind)
        );

        create table if not exists article_nlp (
            article_id bigint primary key references article(id) on delete cascade,
            summary    text,
            stance     text,
            whatmatters text
        );
        """)
        conn.commit()

# ----------- Utilities -----------

def require_admin(req: Request):
    token = req.headers.get("x-admin-token")
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")

def local_now():
    # naive: just use UTC for internals, TZ for header display in email
    return datetime.now(timezone.utc)

def canonicalize_url(raw: str) -> str:
    try:
        u = urlsplit(raw)
        netloc = u.netloc.lower()
        path   = u.path
        query  = parse_qs(u.query, keep_blank_values=False)
        # common tracking params to drop
        for k in list(query.keys()):
            lk = k.lower()
            if lk.startswith("utm_") or lk in {"gclid","fbclid","igshid","mc_cid","mc_eid","oly_anon_id","oly_enc_id"}:
                query.pop(k, None)
        q = urlencode(query, doseq=True)
        return urlunsplit((u.scheme, netloc, path, q, ""))
    except Exception:
        return raw

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def is_blocked_host(host: str) -> bool:
    h = host.lower()
    if h in BLOCKED_HOSTS: return True
    # subdomain check (e.g. img.lh3.googleusercontent.com)
    return any(h.endswith("."+b) for b in BLOCKED_HOSTS)

def looks_like_text(content_type: str|None) -> bool:
    if not content_type: return False
    return any(ct in content_type for ct in OK_CONTENT_TYPES)

def fetch_page_text(url: str) -> tuple[str|None, str|None, datetime|None]:
    """
    Return (title, text, published_at) or (None,None,None) if not usable.
    Very light extraction using BeautifulSoup to stay dependency-light.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; QuantBriefBot/1.0; +https://quantbrief.ca/bot)"
        }
        r = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        ct = r.headers.get("Content-Type", "")
        if not looks_like_text(ct) or r.status_code >= 400:
            return (None, None, None)

        from bs4 import BeautifulSoup  # bs4 should be in requirements
        soup = BeautifulSoup(r.text, "html.parser")
        # title
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # remove script/style/nav/footer
        for tag in soup(["script","style","noscript","header","footer","svg","picture","form","aside"]):
            tag.decompose()

        # try to find 'article' text
        article_node = soup.find("article")
        text = (article_node.get_text("\n", strip=True) if article_node else soup.get_text("\n", strip=True))
        # collapse whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # naive published detection (optional)
        published = None
        dt_meta = soup.find("meta", {"property":"article:published_time"}) or soup.find("meta", {"name":"pubdate"})
        if dt_meta and (dtv := dt_meta.get("content")):
            try:
                published = datetime.fromisoformat(dtv.replace("Z","+00:00"))
            except Exception:
                published = None

        # sanity
        if not text or len(text) < 200:
            return (title, None, published)

        return (title, text, published)
    except Exception:
        return (None, None, None)

# ----------- Google Custom Search (preferred over Google News aggregator) -----------

def google_cse_search(queries: list[str], max_results: int = 25) -> list[dict]:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return []
    items = []
    seen = set()
    # CSE API caps num per call at 10; loop pages via start param
    remaining = max_results
    for q in queries:
        start = 1
        while remaining > 0 and start <= 41:  # up to 4 pages
            n = min(10, remaining)
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CSE_ID,
                "q": q,
                "num": n,
                "safe": "off",
            }
            try:
                resp = requests.get(url, params=params, timeout=15)
                data = resp.json()
                for it in data.get("items", []):
                    link = it.get("link")
                    if not link: continue
                    host = domain_of(link)
                    if is_blocked_host(host): 
                        continue
                    if link in seen: 
                        continue
                    seen.add(link)
                    items.append({
                        "url": link,
                        "title": it.get("title"),
                        "snippet": it.get("snippet"),
                        "publisher": host,
                        "source": "google_cse",
                    })
                    remaining -= 1
                # next page
                if "nextPage" in data.get("queries", {}):
                    start = data["queries"]["nextPage"][0].get("startIndex", start+10)
                else:
                    break
            except Exception:
                break
    return items

# ----------- OpenAI (summarization) -----------

_openai_client = None
def openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI  # ensure package installed
        _openai_client = OpenAI()
    return _openai_client

WHATMATTERS_PROMPT_TMPL = """You are a hedge-fund analyst at QuantBrief. Be terse, factual, and skeptical. Separate facts from inference; if you must guess, prefix with 'Assumption:'.

Source (truncated to relevant text):
---
{{TEXT}}
---

Return exactly 2 bullets:
• What matters — 1 short line on why this matters for investors (specific to the subject in the source).
• Stance — Positive / Negative / Neutral (choose one).
"""

def summarize_text_llm(text: str) -> tuple[str, str, str]:
    """
    Returns (summary_text, what_matters_line, stance_label)
    """
    prompt = Template(WHATMATTERS_PROMPT_TMPL).render(TEXT=text[:6000])
    cli = openai_client()
    r = cli.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        max_output_tokens=OPENAI_MAX_OUTPUT,
    )
    out = (r.output_text or "").strip()
    # Try to split out lines
    wm = ""
    stance = ""
    for line in out.splitlines():
        s = line.strip()
        if s.lower().startswith("• what matters") or s.lower().startswith("- what matters"):
            wm = s.split("—",1)[-1].strip() if "—" in s else s.split(":",1)[-1].strip()
        elif s.lower().startswith("• stance") or s.lower().startswith("- stance"):
            stance = s.split("—",1)[-1].strip() if "—" in s else s.split(":",1)[-1].strip()
    if not wm: wm = out[:200].replace("\n"," ")
    if not stance: stance = "Neutral"
    return (out, wm, stance)

# ----------- Ingestion & NLP Pipeline -----------

def upsert_article(conn, url, title, publisher, source, published_at, clean_text):
    canon = canonicalize_url(url)
    pub = publisher or domain_of(canon)
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        # skip blocked hosts at DB boundary too
        if is_blocked_host(domain_of(canon)):
            return None
        cur.execute("""
            insert into article (url, canonical_url, title, publisher, published_at, clean_text, source)
            values (%s,%s,%s,%s,%s,%s,%s)
            on conflict on constraint ux_article_canon do update
              set title = coalesce(excluded.title, article.title),
                  publisher = coalesce(excluded.publisher, article.publisher),
                  published_at = coalesce(excluded.published_at, article.published_at),
                  source = coalesce(excluded.source, article.source)
            returning id;
        """, (url, canon, title, pub, published_at, clean_text, source))
        row = cur.fetchone()
        conn.commit()
        return row[0] if row else None

def tag_article(conn, article_id: int, equity_id: int, tag_kind: str = "company"):
    with conn.cursor() as cur:
        cur.execute("""
            insert into article_tag (article_id, equity_id, tag_kind)
            values (%s,%s,%s)
            on conflict do nothing;
        """, (article_id, equity_id, tag_kind))
        conn.commit()

def ensure_summary(conn, article_id: int, text: str, force: bool = False):
    with conn.cursor() as cur:
        if not force:
            cur.execute("select 1 from article_nlp where article_id=%s", (article_id,))
            if cur.fetchone():
                return
    # call LLM
    summary, wm, stance = summarize_text_llm(text)
    with conn.cursor() as cur:
        cur.execute("""
            insert into article_nlp (article_id, summary, whatmatters, stance)
            values (%s,%s,%s,%s)
            on conflict (article_id) do update set summary=excluded.summary, whatmatters=excluded.whatmatters, stance=excluded.stance;
        """, (article_id, summary, wm, stance))
        conn.commit()

def ingest_for_equity(conn, equity_id: int, ticker: str, name: str):
    # build queries
    queries = [
        f"{name} {ticker} earnings",
        f"{name} {ticker} power market",
        f"{ticker} data center power deal",
        f"{name} project financing",
        f"{ticker} grid reliability",
        f"{name} permit OR PPA OR outage",
        f"{ticker} site:seekingalpha.com OR site:bloomberg.com OR site:wsj.com -subscription",  # many blocked, but we skip bad hosts later
    ]
    found = 0
    inserted = 0
    summarized = 0

    results = google_cse_search(queries, max_results=MAX_CSE_RESULTS)
    found = len(results)

    # cap summaries per run
    remaining_summ = MAX_SUMMARIES_PER_RUN

    for it in results:
        url = it["url"]
        host = domain_of(url)
        if is_blocked_host(host):
            continue

        title, text, published = fetch_page_text(url)
        if not text or len(text) < 200:
            # we skip if we can't extract meaningful text
            continue

        aid = upsert_article(
            conn=conn,
            url=url,
            title=title or it.get("title"),
            publisher=it.get("publisher") or host,
            source=it.get("source","google_cse"),
            published_at=published or datetime.now(timezone.utc),
            clean_text=text
        )
        if not aid:
            continue
        inserted += 1

        tag_article(conn, aid, equity_id, "company")

        if remaining_summ > 0:
            ensure_summary(conn, aid, text)
            summarized += 1
            remaining_summ -= 1

    return found, inserted, summarized

# ----------- Email (Mailgun) -----------

EMAIL_TMPL = Template("""<!doctype html>
<html>
  <body style="font-family: Arial, sans-serif; line-height:1.5; color:#111;">
    <h2 style="margin:0 0 6px 0;">QuantBrief Daily — {{date_str}}</h2>
    <div style="color:#666; font-size:13px; margin-bottom:12px;">Window: last 24h ending {{window_label}}</div>
    <h3 style="margin:18px 0 8px 0;">Company — {{company_name}} ({{ticker}})</h3>

    {% if items %}
      {% for it in items %}
        <div style="margin:12px 0; padding:10px 12px; border:1px solid #eee; border-radius:8px;">
          <div style="font-size:13px; color:#666;">
            <b>{{it.publisher}}</b> — <span>{{it.published_at}}</span>
          </div>
          <div style="margin:6px 0;">
            <a href="{{it.url}}" style="text-decoration:none; color:#0645ad; font-weight:bold;">{{it.title or it.url}}</a>
          </div>
          {% if it.whatmatters %}
            <div><b>What matters —</b> {{it.whatmatters}}</div>
          {% endif %}
          {% if it.stance %}
            <div><b>Stance —</b> {{it.stance}}</div>
          {% endif %}
        </div>
      {% endfor %}
    {% else %}
      <p>No items with summaries found in the last 24 hours.</p>
    {% endif %}

    <div style="margin-top:16px; color:#777; font-size:12px;">
      Sources are links only. We don’t republish paywalled content. Filters editable by owner.
    </div>
  </body>
</html>
""")

def send_email(to_addr: str, subject: str, html: str):
    if not (MAILGUN_DOMAIN and MAILGUN_API_KEY):
        raise RuntimeError("Mailgun not configured")
    url = f"{MAILGUN_API_BASE}/v3/{MAILGUN_DOMAIN}/messages"
    data = {
        "from": MAILGUN_FROM,
        "to": to_addr,
        "subject": subject,
        "html": html
    }
    r = requests.post(url, auth=("api", MAILGUN_API_KEY), data=data, timeout=15)
    if r.status_code >= 400:
        raise RuntimeError(f"mailgun send failed {r.status_code}: {r.text}")

# ----------- Routes -----------

@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK"

@app.post("/admin/init")
def admin_init(req: Request):
    require_admin(req)
    init_db()
    with db() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        # ensure owner subscriber
        cur.execute("insert into subscriber (email, tz, send_hour, send_min) values (%s,%s,%s,%s) on conflict do nothing returning id;",
                    (OWNER_EMAIL, TIMEZONE, 8, 30))
        if cur.rowcount:
            owner_id = cur.fetchone()[0]
        else:
            cur.execute("select id from subscriber where email=%s", (OWNER_EMAIL,))
            owner_id = cur.fetchone()[0]

        # ensure TLN equity as example
        cur.execute("insert into equity (ticker, name) values (%s,%s) on conflict (ticker) do update set name=excluded.name returning id;",
                    ("TLN", "Talen Energy"))
        eid = cur.fetchone()[0]

        # ensure watch
        cur.execute("""
            insert into subscriber_watch (subscriber_id, equity_id)
            values (%s,%s) on conflict do nothing;
        """, (owner_id, eid))
        conn.commit()
    return {"ok": True}

@app.get("/admin/test_openai")
def admin_test_openai(req: Request):
    require_admin(req)
    try:
        _ = openai_client()
        # tiny ping
        r = _ .responses.create(model=OPENAI_MODEL, input="Say OK", max_output_tokens=8)
        return {"ok": True, "model": OPENAI_MODEL, "sample": (r.output_text or "")[:50]}
    except Exception as e:
        return {"ok": False, "model": OPENAI_MODEL, "err": str(e)}

@app.post("/cron/ingest")
def cron_ingest(req: Request, minutes: int = 1440):
    require_admin(req)
    started = time.time()
    found = inserted = summarized = 0
    with db() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        # get all equities under watch (single-user MVP)
        cur.execute("""
            select distinct e.id, e.ticker, e.name
            from equity e
            join subscriber_watch w on w.equity_id=e.id
        """)
        equities = cur.fetchall()

    for e in equities:
        f,i,s = ingest_for_equity(conn, e["id"], e["ticker"], e["name"] or e["ticker"])
        found += f; inserted += i; summarized += s

    return {
        "ok": True,
        "elapsed_s": round(time.time()-started,2),
        "found_urls": found,
        "inserted": inserted,
        "summarized": summarized
    }

@app.post("/cron/digest")
def cron_digest(req: Request):
    require_admin(req)
    # compute 24h window ending at 08:30 TIMEZONE
    # For simplicity, use last 24h from now:
    t_end = datetime.now(timezone.utc)
    t_start = t_end - timedelta(hours=24)

    # load one equity (MVP: TLN)
    with db() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("select id, ticker, name from equity where ticker=%s", ("TLN",))
        row = cur.fetchone()
        if not row:
            return {"ok": False, "err": "No equity TLN found"}
        eid, ticker, name = row["id"], row["ticker"], row["name"] or row["ticker"]

        # Require we have an NLP summary AND exclude blocked/aggregator hosts
        cur.execute("""
            select a.id, a.url, a.canonical_url, a.title, a.publisher, a.published_at, n.whatmatters, n.stance
            from article a
            join article_tag t on t.article_id=a.id and t.equity_id=%s
            join article_nlp n on n.article_id=a.id
            where a.published_at between %s and %s
              and coalesce(a.clean_text,'') <> ''
              and a.publisher not ilike '%%googleusercontent.com%%'
              and a.publisher not ilike '%%news.google.com%%'
              and a.publisher not ilike '%%gstatic.com%%'
              and a.url not ilike '%%lh3.googleusercontent.com%%'
              and a.canonical_url not ilike '%%lh3.googleusercontent.com%%'
            order by a.published_at desc
            limit 100;
        """, (eid, t_start, t_end))
        arts = cur.fetchall()

        items = []
        for a in arts:
            items.append({
                "url": a["canonical_url"] or a["url"],
                "title": a["title"],
                "publisher": a["publisher"] or domain_of(a["canonical_url"] or a["url"]),
                "published_at": (a["published_at"] or t_end).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "whatmatters": a["whatmatters"],
                "stance": a["stance"],
            })

        # load recipient (owner)
        cur.execute("select email, tz, send_hour, send_min from subscriber where email=%s", (OWNER_EMAIL,))
        s = cur.fetchone()
        if not s:
            raise RuntimeError("owner subscriber missing")

    # Render email
    # show local label for header
    window_label = t_end.astimezone(timezone.utc).strftime("%H:%M UTC")
    date_str = t_end.astimezone(timezone.utc).strftime("%Y-%m-%d")

    html = EMAIL_TMPL.render(
        date_str=date_str,
        window_label=window_label,
        company_name=name,
        ticker=ticker,
        items=items
    )

    subject = f"QuantBrief Daily — {date_str} — {ticker}"
    send_email(OWNER_EMAIL, subject, html)

    return {"ok": True, "sent_to": OWNER_EMAIL, "items": len(items)}

# ---------- OPTIONAL: one-time purge endpoint (cleanup bad rows) ----------

@app.post("/admin/purge_blocked")
def purge_blocked(req: Request, days: int = 7):
    require_admin(req)
    cutoff = f"{days} days"
    with db() as conn, conn.cursor() as cur:
        cur.execute(f"""
            delete from article
            where created_at > now() - interval %s
              and (publisher ilike '%%googleusercontent.com%%'
                   or publisher ilike '%%news.google.com%%'
                   or publisher ilike '%%gstatic.com%%'
                   or url ilike '%%lh3.googleusercontent.com%%'
                   or coalesce(canonical_url,'') ilike '%%lh3.googleusercontent.com%%');
        """, (cutoff,))
        n1 = cur.rowcount
        cur.execute(f"""
            delete from article
            where created_at > now() - interval %s
              and (length(coalesce(clean_text,'')) < 120
                   or clean_text like '%%\\x89PNG%%'
                   or clean_text like '%%\\xFF\\xD8%%');
        """, (cutoff,))
        n2 = cur.rowcount
        conn.commit()
    return {"ok": True, "deleted_blocked": n1, "deleted_short_or_binary": n2}
