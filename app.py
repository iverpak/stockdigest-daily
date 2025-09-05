import os, re, json, traceback, datetime as dt, html as html_lib
from typing import List, Tuple, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, quote_plus, unquote

import httpx, feedparser, psycopg
from fastapi import FastAPI, Request, HTTPException
from jinja2 import Template
from readability import Document

# ---------- ENV ----------
DATABASE_URL    = os.getenv("DATABASE_URL")
MAILGUN_DOMAIN  = os.getenv("MAILGUN_DOMAIN")
MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY")
MAILGUN_FROM    = os.getenv("MAILGUN_FROM", "QuantBrief Daily <daily@mg.quantbrief.ca>")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-5-mini")
ADMIN_TOKEN     = os.getenv("ADMIN_TOKEN", "")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing")

app = FastAPI()

# ---------- BLOCK/ALLOW LISTS ----------
BLOCKED_HOSTS = {
    "news.google.com",
    "lh3.googleusercontent.com",
    "googleusercontent.com",
    "gstatic.com",
    "feedproxy.google.com",
    "consent.google.com",
    "accounts.google.com",
    "images.google.com",
}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico")

def is_blocked_host(host: Optional[str]) -> bool:
    if not host:
        return True
    host = host.lower()
    return any(host == h or host.endswith("." + h) for h in BLOCKED_HOSTS)

def is_image_url(u: str) -> bool:
    p = urlparse(u)
    if is_blocked_host(p.hostname):
        return True
    path = (p.path or "").lower()
    return path.endswith(IMAGE_EXT)

# ---------- SQL ----------
DDL = """
create table if not exists app_user (
  id serial primary key,
  email text unique not null,
  role text not null default 'owner',
  tz text not null default 'America/Toronto',
  created_at timestamptz default now()
);

create table if not exists equity (
  id serial primary key,
  ticker text not null,
  exchange text default 'US',
  name text,
  sector text, industry text,
  peers text[],
  unique (ticker, exchange)
);

create table if not exists user_watchlist (
  user_id int references app_user(id),
  equity_id int references equity(id),
  scope text not null default 'company',
  primary key (user_id, equity_id)
);

create table if not exists source_feed (
  id serial primary key,
  equity_id int references equity(id),
  kind text not null,   -- 'ir_press'
  url text not null,
  active boolean default true,
  created_at timestamptz default now()
);

create table if not exists gdelt_filter_set (
  id serial primary key,
  owner_user_id int references app_user(id),
  name text not null,
  scope text not null, -- 'company','peer','industry'
  locked boolean default true,
  is_active boolean default true,
  created_at timestamptz default now()
);

create table if not exists gdelt_filter_rule (
  id serial primary key,
  set_id int references gdelt_filter_set(id) on delete cascade,
  version int not null,
  query_bool text,
  domain_allowlist text[],
  domain_blocklist text[],
  include_keywords text[],
  exclude_keywords text[],
  entity_aliases jsonb,
  geos jsonb,
  created_at timestamptz default now()
);

create table if not exists article (
  id bigserial primary key,
  url text not null,
  canonical_url text,
  title text,
  publisher text,
  published_at timestamptz,
  sha256 char(64) not null unique,
  raw_html text,
  clean_text text,
  created_at timestamptz default now()
);

create table if not exists article_tag (
  article_id bigint references article(id) on delete cascade,
  equity_id int references equity(id),
  tag_kind text not null, -- 'company','peer','industry'
  primary key (article_id, equity_id, tag_kind)
);

create table if not exists article_nlp (
  article_id bigint primary key references article(id) on delete cascade,
  stance text,
  summary text,
  entities jsonb,
  quality_score numeric
);

create table if not exists delivery_log (
  id bigserial primary key,
  user_id int references app_user(id),
  run_date date not null,
  items_json jsonb,
  sent_at timestamptz default now(),
  unique (user_id, run_date)
);

create table if not exists gnews_query (
  id serial primary key,
  equity_id int references equity(id),
  scope text not null default 'industry',
  query text not null,
  active boolean default true,
  weight int default 10,
  created_at timestamptz default now()
);
"""

def db():
    return psycopg.connect(DATABASE_URL, autocommit=True)

# ---------- utils ----------
def normalize_url(u: str) -> str:
    p = urlparse(u)
    q = [(k,v) for k,v in parse_qsl(p.query, keep_blank_values=True)
         if not k.lower().startswith('utm_') and k.lower() not in ('gclid','fbclid')]
    new = p._replace(query=urlencode(q, doseq=True), fragment="")
    return urlunparse(new)

def sha256(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def now_local() -> dt.datetime:
    return dt.datetime.now().astimezone()

def is_gnews(u: str) -> bool:
    host = urlparse(u).hostname or ""
    return host.endswith("news.google.com")

def unwrap_gnews_url(u: str) -> str:
    """Pull the publisher URL out of a Google News redirect link if present."""
    if not is_gnews(u):
        return u
    p = urlparse(u)
    q = dict(parse_qsl(p.query))
    # Primary param Google uses
    for key in ("url","u","link","q"):
        if key in q and q[key].startswith("http"):
            return unquote(q[key])
    return u

def looks_garbled(s: str) -> bool:
    if not s: return True
    if "\x89PNG" in s or "\xFF\xD8\xFF" in s:
        return True
    bad = s.count("�")
    return bad > 10 or len(s) < 120

# ---------- admin: init/seed/config ----------
@app.post("/admin/init")
async def admin_init(req: Request):
    if req.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(401, "bad token")
    with db() as con:
        con.execute(DDL)
    return {"ok": True}

@app.post("/admin/seed_tln")
async def admin_seed_tln(req: Request):
    if req.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(401, "bad token")
    body = await req.json()
    owner_email = body.get("owner_email")
    ir_feed_url = body.get("ir_feed_url","")
    with db() as con:
        cur = con.execute(
            "insert into app_user(email,role) values (%s,'owner') on conflict (email) do update set role='owner' returning id",
            (owner_email,))
        owner_id = cur.fetchone()[0]
        cur = con.execute("""
            insert into equity(ticker,name,sector,industry,peers)
            values ('TLN','Talen Energy','Utilities','Independent Power Producers', ARRAY['NRG','NEP','DUK'])
            on conflict (ticker,exchange) do update set name=excluded.name
            returning id
        """)
        equity_id = cur.fetchone()[0]
        con.execute("insert into user_watchlist(user_id,equity_id,scope) values (%s,%s,'company') on conflict do nothing",
                    (owner_id,equity_id))
        if ir_feed_url:
            con.execute("insert into source_feed(equity_id,kind,url) values (%s,'ir_press',%s) on conflict do nothing",
                        (equity_id, ir_feed_url))
    return {"ok": True, "owner_id": owner_id}

@app.post("/admin/set_gdelt_filter")
async def admin_set_gdelt_filter(req: Request):
    if req.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(401, "bad token")
    body = await req.json()
    owner_email = body["owner_email"]
    name = body.get("name","TLN Company")
    with db() as con:
        cur = con.execute("select id from app_user where email=%s", (owner_email,))
        row = cur.fetchone()
        if not row: raise HTTPException(400,"owner not found")
        owner_id = row[0]
        cur = con.execute("""
            insert into gdelt_filter_set(owner_user_id,name,scope,locked,is_active)
            values (%s,%s,'company',true,true)
            returning id
        """,(owner_id,name))
        set_id = cur.fetchone()[0]
        con.execute("""
          insert into gdelt_filter_rule(set_id,version,query_bool,domain_allowlist,domain_blocklist,include_keywords,exclude_keywords,entity_aliases)
          values (
            %s,1,
            '("Talen Energy" OR ("Talen" NEAR/2 Energy) OR "Brandon Shores" OR "H.A. Wagner")',
            ARRAY[]::text[],
            ARRAY['x.com','facebook.com','news.google.com','lh3.googleusercontent.com','googleusercontent.com','gstatic.com'],
            ARRAY['PJM','ERCOT','IPP','data center','hyperscale','capacity auction','interconnect','outage','FERC','transmission'],
            ARRAY['gaming','fashion','esports','music'],
            '{"company_aliases":["Talen Energy","TLN"]}'::jsonb
          )
        """,(set_id,))
    return {"ok": True}

@app.post("/admin/set_gnews_queries")
async def admin_set_gnews_queries(req: Request):
    if req.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(401, "bad token")
    body = await req.json()
    ticker  = body["ticker"]
    queries = body["queries"]
    with db() as con:
        cur = con.execute("select id from equity where ticker=%s", (ticker,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(400, f"unknown ticker {ticker}")
        equity_id = row[0]
        con.execute("delete from gnews_query where equity_id=%s", (equity_id,))
        for q in queries:
            scope  = q.get("scope","industry")
            query  = q["query"]
            weight = int(q.get("weight",10))
            con.execute(
                "insert into gnews_query(equity_id,scope,query,weight) values (%s,%s,%s,%s)",
                (equity_id, scope, query, weight)
            )
    return {"ok": True}

@app.get("/admin/list_gnews_queries")
async def admin_list_gnews_queries(req: Request, ticker: str):
    if req.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(401, "bad token")
    with db() as con:
        cur = con.execute("""
          select q.id, e.ticker, q.scope, q.query, q.weight, q.active
          from gnews_query q join equity e on e.id=q.equity_id
          where e.ticker=%s order by q.weight desc, q.id
        """,(ticker,))
        rows = [dict(zip([d.name for d in cur.description], r)) for r in cur.fetchall()]
    return {"ok": True, "items": rows}

@app.get("/admin/test_openai")
async def admin_test_openai(req: Request):
    if req.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(401, "bad token")
    try:
        if not OPENAI_API_KEY:
            return {"ok": False, "err": "OPENAI_API_KEY missing"}
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        r = client.responses.create(model=OPENAI_MODEL, input="Say OK.")
        return {"ok": True, "model": OPENAI_MODEL, "sample": (r.output_text or "").strip()[:100]}
    except Exception as e:
        return {"ok": False, "model": OPENAI_MODEL, "err": str(e)}

@app.post("/admin/resummarize")
async def admin_resummarize(req: Request):
    if req.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(401, "bad token")
    q = req.query_params
    hours = int(q.get("hours", "48"))
    limit = int(q.get("limit", "150"))

    with db() as con:
        cur = con.execute("""
          select a.id, a.title, coalesce(a.clean_text, '')
          from article a
          where a.created_at > now() - (%s || ' hours')::interval
          order by a.id desc
          limit %s
        """, (hours, limit))
        rows = cur.fetchall()

    done = 0
    for aid, title, text in rows:
        try:
            if looks_garbled(text):
                continue
            summary = await summarize_article(aid, title or "", text or "")
            with db() as con:
                con.execute("""
                  insert into article_nlp(article_id, stance, summary)
                  values (%s, %s, %s)
                  on conflict (article_id) do update
                    set stance = excluded.stance,
                        summary = excluded.summary
                """, (aid, "Neutral", summary))
            done += 1
        except Exception as e:
            print(f"[resummarize] failed for {aid}: {e}")

    return {"ok": True, "resummarized": done, "hours": hours, "limit": limit}

@app.post("/admin/repair_gnews")
async def admin_repair_gnews(req: Request):
    if req.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(401, "bad token")
    q = req.query_params
    hours = int(q.get("hours", "96"))
    fixed = 0
    with db() as con:
        cur = con.execute("""
          select id, url, canonical_url from article
          where (publisher ilike '%%news.google.com%%' or url ilike '%%news.google.com%%' or canonical_url ilike '%%news.google.com%%'
                 or url ilike '%%lh3.googleusercontent.com%%' or canonical_url ilike '%%lh3.googleusercontent.com%%')
            and created_at > now() - (%s || ' hours')::interval
          order by id desc
          limit 800
        """, (hours,))
        rows = cur.fetchall()
    for aid, u, cu in rows:
        try:
            target = unwrap_gnews_url(cu or u)
            if is_image_url(target) or is_blocked_host(urlparse(target).hostname):
                continue
            title, publisher, clean, published_at, final_url = await fetch_text(target)
            host = urlparse(final_url).hostname or ""
            if is_blocked_host(host) or looks_garbled(clean):
                continue
            canon = normalize_url(final_url)
            with db() as con:
                con.execute("""
                  update article
                     set url=%s, canonical_url=%s, title=%s, publisher=%s, published_at=%s, clean_text=%s
                   where id=%s
                """, (target, canon, title, publisher, published_at, clean, aid))
            fixed += 1
        except Exception as e:
            print(f"[repair_gnews] {aid} failed: {e}")
    return {"ok": True, "fixed": fixed, "hours": hours}

# ---------- fetch & parse ----------
async def fetch_text(url: str, depth: int = 0) -> Tuple[str,str,str,dt.datetime,str]:
    """Return (title, publisher, clean_text, published_at, final_url)."""
    if depth > 3:
        return ("", urlparse(url).hostname or "", "", now_utc(), url)

    # unwrap & pre-filter
    url = unwrap_gnews_url(url)
    if is_image_url(url) or is_blocked_host((urlparse(url).hostname or "")):
        return ("", urlparse(url).hostname or "", "", now_utc(), url)

    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 QuantBriefBot/1.3",
        "Accept-Language": "en-US,en;q=0.9",
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=25) as h:
        r = await h.get(url, headers=headers)
        r.raise_for_status()
        html = r.text
        final_url = str(r.url)

    host = urlparse(final_url).hostname or ""
    if is_blocked_host(host):
        return ("", host, "", now_utc(), final_url)

    # If still on gNews, aggressively extract external link and refetch
    if is_gnews(final_url):
        # try ?url= inside the HTML
        for pat in [
            r'(?:[?&]url=)(https?%3A%2F%2F[^&"\'<> ]+)',
            r'data-n-au="(https?://[^"]+)"',
            r'"url"\s*:\s*"(https?://[^"]+)"',
            r'href="(https?://[^"]+)"'
        ]:
            m = re.search(pat, html)
            if m:
                cand = unquote(m.group(1))
                if not is_blocked_host(urlparse(cand).hostname or ""):
                    return await fetch_text(cand, depth+1)

    # Parse article DOM
    doc = Document(html)
    title = (doc.short_title() or "").strip()
    if not title:
        t = re.search(r"<title>(.*?)</title>", html, re.I|re.S)
        title = (t.group(1).strip() if t else "")

    content_html = doc.summary(html_partial=True) or ""
    publisher = host
    published_at = now_utc()
    clean = re.sub("<[^<]+?>"," ", content_html)
    clean = re.sub(r"\s+"," ", clean).strip()

    # fallbacks
    if not clean:
        og = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
        if og:
            clean = og.group(1).strip()

    # filter aggregator/gibberish
    if looks_garbled(clean) or "Comprehensive up-to-date news coverage" in clean:
        return (title, publisher, "", published_at, final_url)

    return title, publisher, clean, published_at, final_url

def rss_time(entry) -> dt.datetime:
    for k in ("published_parsed","updated_parsed","created_parsed"):
        t = getattr(entry, k, None)
        if t:
            return dt.datetime(*t[:6], tzinfo=dt.timezone.utc)
    return now_utc()

async def ingest_ir_feed(feed_url: str) -> int:
    parsed = feedparser.parse(feed_url)
    new_count=0
    with db() as con:
        for e in parsed.entries[:50]:
            raw = e.link
            if is_image_url(raw) or is_blocked_host(urlparse(raw).hostname or ""):
                continue
            url = normalize_url(unwrap_gnews_url(raw))
            if is_image_url(url) or is_blocked_host(urlparse(url).hostname or ""):
                continue
            key = sha256(url)
            published_at = rss_time(e)
            title = getattr(e, "title", "") or ""
            publisher = urlparse(url).hostname or ""
            try:
                con.execute("""
                  insert into article(url,canonical_url,title,publisher,published_at,sha256)
                  values (%s,%s,%s,%s,%s,%s)
                """,(url,url,title,publisher,published_at,key))
                new_count += 1
            except Exception:
                pass
    return new_count

# ---------- GDELT ----------
async def gdelt_search(q: str, minutes: int=60, maxrecords: int=50) -> List[str]:
    params = {
        "query": q,
        "mode": "ArtList",
        "format":"json",
        "timespan": f"{minutes}m",
        "maxrecords": str(maxrecords),
        "sort": "DateDesc",
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    async with httpx.AsyncClient(timeout=25) as h:
        r = await h.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    out=[]
    for row in data.get("articles", []):
        u = row.get("url","")
        if not u: continue
        u = normalize_url(unwrap_gnews_url(u))
        if is_image_url(u) or is_blocked_host(urlparse(u).hostname or ""):
            continue
        out.append(u)
    seen=set(); uniq=[]
    for u in out:
        if u not in seen:
            uniq.append(u); seen.add(u)
    return uniq

async def gdelt_urls_from_rules(rules: dict, minutes: int=60) -> List[str]:
    q = (rules or {}).get("query_bool") or ""
    if (rules or {}).get("include_keywords"):
        inc = " OR ".join([f'"{k}"' for k in rules["include_keywords"]])
        q = f"({q}) AND ({inc})" if q else inc
    urls = await gdelt_search(q, minutes=minutes, maxrecords=75)
    return urls[:40]

# ---------- Google News RSS (stricter unwrap) ----------
def google_news_rss_urls(query: str, max_items: int=40) -> List[str]:
    rss = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    parsed = feedparser.parse(rss)
    urls=[]
    for e in parsed.entries[:max_items]:
        # Prefer links that are HTML pages
        picked = None
        for L in e.get("links", []):
            href = L.get("href") or ""
            typ  = (L.get("type") or "").lower()
            if not href: continue
            if typ and not ("html" in typ):
                continue
            if is_image_url(href) or is_blocked_host(urlparse(href).hostname or ""):
                continue
            if "google." in (urlparse(href).hostname or ""):
                continue
            picked = href
            break

        if not picked:
            # Try entry.link
            link = e.get("link") or ""
            if link and not is_image_url(link):
                picked = unwrap_gnews_url(link)

        if not picked:
            # Try summary HTML anchors
            summ = e.get("summary") or e.get("description") or ""
            summ = html_lib.unescape(summ)
            for m in re.finditer(r'href="(https?://[^"]+)"', summ):
                cand = m.group(1)
                if is_image_url(cand): 
                    continue
                host = urlparse(cand).hostname or ""
                if "google." in host or is_blocked_host(host):
                    continue
                picked = cand
                break

        if not picked:
            # last resort: keep the gNews link; fetch_text() will try to unwrap
            picked = e.get("link") or ""

        if picked:
            picked = normalize_url(unwrap_gnews_url(picked))
            if not is_image_url(picked) and not is_blocked_host(urlparse(picked).hostname or ""):
                urls.append(picked)

    # de-dup
    seen=set(); out=[]
    for u in urls:
        if u not in seen:
            out.append(u); seen.add(u)
    return out

# ---------- Upsert & summarize ----------
async def upsert_article(url: str, equity_id: int, tag_kind: str) -> bool:
    url = normalize_url(unwrap_gnews_url(url))
    if is_image_url(url) or is_blocked_host(urlparse(url).hostname or ""):
        return False

    title, publisher, clean, published_at, final_url = await fetch_text(url)
    host = urlparse(final_url).hostname or ""

    # Drop if blocked/garbled/empty
    if is_blocked_host(host) or looks_garbled(clean):
        return False

    canon = normalize_url(final_url)
    key = sha256(canon)

    with db() as con:
        try:
            con.execute("""
              insert into article(url,canonical_url,title,publisher,published_at,sha256,raw_html,clean_text)
              values (%s,%s,%s,%s,%s,%s,%s,%s)
            """,(url, canon, title, host, published_at, key, None, clean))
        except Exception:
            pass
        cur = con.execute("select id from article where sha256=%s",(key,))
        row = cur.fetchone()
        if not row:
            return False
        aid = row[0]
        try:
            con.execute("insert into article_tag(article_id,equity_id,tag_kind) values (%s,%s,%s)", (aid,equity_id,tag_kind))
        except Exception:
            pass
    return True

def fallback_summary(title: str, text: str) -> str:
    blob = (text or "").strip()
    parts = re.split(r'(?<=[.!?])\s+', blob)
    gist = " ".join(parts[:3]).strip()
    if not gist:
        gist = (title or "Headline only")
    return f"• What matters — {gist}\n• Stance — Neutral"

async def summarize_article(aid: int, title: str, text: str) -> str:
    if not OPENAI_API_KEY or looks_garbled(text):
        return fallback_summary(title, text)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        system = "You are an equity analyst. Be terse, factual, skeptical."
        prompt = f"""Title: {title}

Source text (truncated):
{text[:8000]}

Return exactly:
• What matters — 2–3 short bullets (include numbers if present).
• Stance — Positive, Negative, or Neutral."""
        r = client.responses.create(
            model=OPENAI_MODEL,
            input=[{"role":"system","content":system},
                   {"role":"user","content":prompt}],
            max_output_tokens=280
        )
        out = (r.output_text or "").strip()
        return out or fallback_summary(title, text)
    except Exception as e:
        print(f"[summarize_article] OpenAI failed for article {aid}: {e}")
        traceback.print_exc()
        return fallback_summary(title, text)

# ---------- CRON: INGEST ----------
@app.get("/cron/ingest")
async def cron_ingest(req: Request):
    if req.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(401,"bad token")
    minutes = int(req.query_params.get("minutes", "60"))
    new_urls=set()

    with db() as con:
        cur = con.execute("select id from equity where ticker='TLN'")
        row = cur.fetchone()
        if not row:
            return {"ok": False, "msg":"TLN not seeded"}
        tln_id = row[0]
        feeds = [r[0] for r in con.execute("select url from source_feed where equity_id=%s and active", (tln_id,))]
        cur = con.execute("""
            select r.query_bool, r.domain_allowlist, r.domain_blocklist, r.include_keywords, r.exclude_keywords
            from gdelt_filter_set s
            join gdelt_filter_rule r on r.set_id=s.id
            where s.scope='company' and s.is_active
            order by r.version desc limit 1
        """)
        r = cur.fetchone()
        rules = {
            "query_bool": r[0] if r else '',
            "domain_allowlist": r[1] if r else [],
            "domain_blocklist": r[2] if r else [],
            "include_keywords": r[3] if r else [],
            "exclude_keywords": r[4] if r else [],
        }

    # IR feeds
    for f in feeds:
        try:
            await ingest_ir_feed(f)
        except Exception:
            pass

    # Existing hashes for dedupe
    with db() as con:
        cur_existing = con.execute("select sha256 from article")
        existing_hashes = {row[0] for row in cur_existing.fetchall()}

    # 1) GDELT
    try:
        urls = await gdelt_urls_from_rules(rules, minutes=minutes)
        new_urls.update(urls)
    except Exception:
        pass

    # 2) Google News RSS (from saved queries)
    with db() as con:
        cur = con.execute("select scope, query, weight from gnews_query where equity_id=%s and active order by weight desc, id",
                          (tln_id,))
        gq = cur.fetchall()

    gnews_batch = []
    for scope, qtext, weight in gq:
        try:
            urls = google_news_rss_urls(qtext, max_items=30)
            for u in urls:
                gnews_batch.append((u, scope, weight))
        except Exception:
            pass

    # Rank by weight then upsert unique
    gnews_batch.sort(key=lambda x: x[2], reverse=True)
    inserted = 0
    for u, scope, _w in gnews_batch:
        try:
            canon_key = sha256(normalize_url(unwrap_gnews_url(u)))
            if canon_key in existing_hashes:
                continue
            if await upsert_article(u, tln_id, scope):
                existing_hashes.add(canon_key)
                inserted += 1
        except Exception as e:
            print(f"[ingest gnews upsert] failed for {u}: {e}")

    # Also bring in a limited set from GDELT
    for u in list(new_urls)[:30]:
        try:
            canon_key = sha256(normalize_url(unwrap_gnews_url(u)))
            if canon_key in existing_hashes:
                continue
            if await upsert_article(u, tln_id, "company"):
                existing_hashes.add(canon_key)
                inserted += 1
        except Exception as e:
            print(f"[ingest upsert] failed for {u}: {e}")

    # Summarize recent articles (UPSERT)
    summarized = 0
    with db() as con:
        cur = con.execute("""
          select a.id, a.title, coalesce(a.clean_text, '')
          from article a
          where a.created_at > now() - interval '3 days'
          order by a.id desc
          limit 160
        """)
        rows = cur.fetchall()

    for aid, title, text in rows:
        try:
            if looks_garbled(text):
                continue
            summary = await summarize_article(aid, title or "", text or "")
            with db() as con:
                con.execute("""
                  insert into article_nlp(article_id, stance, summary)
                  values (%s, %s, %s)
                  on conflict (article_id) do update
                    set stance = excluded.stance,
                        summary = excluded.summary
                """, (aid, "Neutral", summary))
            summarized += 1
        except Exception as e:
            print(f"[ingest summarize] failed for {aid}: {e}")

    return {"ok": True, "found_urls": len(new_urls) + len(gnews_batch), "inserted": inserted, "summarized": summarized}

# ---------- CRON: DIGEST ----------
@app.get("/cron/digest")
async def cron_digest(req: Request):
    if req.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(401,"bad token")

    with db() as con:
        cur = con.execute("select id,email,tz from app_user where role='owner' limit 1")
        row = cur.fetchone()
        if not row: return {"ok": False, "msg":"no owner"}
        user_id, owner_email, user_tz = row

        cur = con.execute("""
          select e.id,e.ticker,e.name from equity e
          join user_watchlist w on w.equity_id=e.id
          where w.user_id=%s
        """,(user_id,))
        wl = cur.fetchall()
        if not wl: return {"ok": False, "msg":"watchlist empty"}
        equity_id, ticker, company_name = wl[0]

    end = now_local()
    start = end - dt.timedelta(days=1)
    run_date = end.date()

    with db() as con:
        cur = con.execute("""
          select a.id,
                 a.url,
                 a.canonical_url,
                 a.title,
                 a.publisher,
                 a.published_at,
                 coalesce(n.summary,'') as summary
          from article a
          left join article_nlp n on n.article_id=a.id
          join article_tag t on t.article_id=a.id and t.equity_id=%s
          where a.published_at between %s and %s
            and t.tag_kind in ('company','industry','peer')
            and coalesce(a.clean_text,'') <> ''
          order by a.published_at desc
        """,(equity_id, start, end))
        items = [dict(zip([d.name for d in cur.description], r)) for r in cur.fetchall()]

    none_found = not items

    # Template if available
    try:
        with open("email_template.html","r",encoding="utf-8") as f:
            tpl = Template(f.read())
        html = tpl.render(
            run_date=str(run_date),
            end_local=end.strftime("%Y-%m-%d %H:%M"),
            ticker=ticker,
            company_name=company_name,
            company_items=items,
            none_found=none_found
        )
    except Exception:
        items_html = "".join(
            (
              lambda href:
              f"<li><a href='{href}'><b>{(i.get('title') or href).strip()}</b></a>"
              f"<div style='font-size:12px;color:#666'>{i.get('publisher','')}"
              f"{' — ' + str(i.get('published_at')) if i.get('published_at') else ''}</div>"
              f"<div>{(i.get('summary') or '').replace('\n','<br>')}</div></li>"
            ) (i.get('canonical_url') or i.get('url'))
            for i in items
        )
        html = f"""
        <html><body style="font-family:Arial,Helvetica,sans-serif;color:#111;">
          <h2>QuantBrief Daily — {run_date}</h2>
          <div style="font-size:13px;color:#666;margin-bottom:18px;">
            Window: last 24h ending {end.strftime("%Y-%m-%d %H:%M")}
          </div>
          <h3>Company — {company_name} ({ticker})</h3>
          <ul>{items_html or '<li>No items found in the last 24 hours.</li>'}</ul>
          <hr><div style="font-size:12px;color:#666;">Sources are links only. We don’t republish paywalled content.</div>
        </body></html>
        """

    async with httpx.AsyncClient(auth=("api", MAILGUN_API_KEY), timeout=25) as h:
        r = await h.post(f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages", data={
            "from": MAILGUN_FROM,
            "to": owner_email,
            "subject": f"QuantBrief Daily — {run_date} — {ticker}",
            "html": html
        })
        r.raise_for_status()

    with db() as con:
        con.execute(
            "insert into delivery_log(user_id,run_date,items_json) values (%s,%s,%s::jsonb) on conflict do nothing",
            (user_id, run_date, json.dumps({"urls": [ (i.get("canonical_url") or i.get("url")) for i in items ]}))
        )
    return {"ok": True, "sent_to": owner_email, "items": len(items)}

# ---------- health ----------
@app.get("/")
def health():
    return {"ok": True}
