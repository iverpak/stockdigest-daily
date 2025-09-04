import os, re, json, traceback, datetime as dt
from typing import List
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, quote_plus

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

# ---------- APP ----------
app = FastAPI()

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
"""

def db():
    return psycopg.connect(DATABASE_URL, autocommit=True)

# ---------- utils ----------
def normalize_url(u: str) -> str:
    p = urlparse(u)
    q = [(k,v) for k,v in parse_qsl(p.query, keep_blank_values=True)
         if not k.lower().startswith('utm_') and k.lower() not in ('gclid','fbclid')]
    new = p._replace(query=urlencode(q, doseq=True))
    return urlunparse(new)

def sha256(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def now_local() -> dt.datetime:
    return dt.datetime.now().astimezone()

# ---------- admin ----------
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
            ARRAY['x.com','facebook.com'],
            ARRAY['PJM','ERCOT','IPP','data center','hyperscale','capacity auction','interconnect','outage','FERC','transmission'],
            ARRAY['gaming','fashion','esports','music'],
            '{"company_aliases":["Talen Energy","TLN"]}'::jsonb
          )
        """,(set_id,))
    return {"ok": True}

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

# ---------- fetch & parse ----------
async def fetch_text(url: str):
    """Return (title, publisher, clean_text, published_at, final_url)."""
    headers = {"User-Agent":"QuantBriefBot/1.0 (+https://quantbrief.ca)"}
    async with httpx.AsyncClient(follow_redirects=True, timeout=25) as h:
        r = await h.get(url, headers=headers)
        r.raise_for_status()
        html = r.text
        final_url = str(r.url)
    doc = Document(html)
    title = (doc.short_title() or "").strip()
    if not title:
        m = re.search(r"<title>(.*?)</title>", html, re.I|re.S)
        title = (m.group(1).strip() if m else "")
    content_html = doc.summary(html_partial=True)
    publisher = urlparse(final_url).hostname or ""
    published_at = now_utc()  # fallback
    clean = re.sub("<[^<]+?>"," ", content_html)
    clean = re.sub(r"\s+"," ", clean).strip()
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
            url = normalize_url(e.link)
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
        u = normalize_url(row.get("url",""))
        if u:
            out.append(u)
    # de-dup
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

# ---------- Google News RSS (fallback; no key) ----------
def google_news_rss_urls(query: str, max_items: int=40) -> List[str]:
    """Return article links from Google News RSS for a query."""
    rss = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    parsed = feedparser.parse(rss)
    urls=[]
    for e in parsed.entries[:max_items]:
        link = e.get("link")
        if link:
            urls.append(normalize_url(link))
    # de-dup
    seen=set(); out=[]
    for u in urls:
        if u not in seen:
            out.append(u); seen.add(u)
    return out

# ---------- Upsert & summarize ----------
async def upsert_article(url: str, equity_id: int, tag_kind: str):
    title, publisher, clean, published_at, final_url = await fetch_text(url)
    canon = normalize_url(final_url)
    key = sha256(canon)
    with db() as con:
        try:
            con.execute("""
              insert into article(url,canonical_url,title,publisher,published_at,sha256,raw_html,clean_text)
              values (%s,%s,%s,%s,%s,%s,%s,%s)
            """,(url, canon, title, publisher, published_at, key, None, clean))
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
        gist = (title or "No content")
    return f"• What matters — {gist}\n• Stance — Neutral"

async def summarize_article(aid: int, title: str, text: str) -> str:
    if not OPENAI_API_KEY:
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

    # IR feeds (if any)
    for f in feeds:
        try:
            await ingest_ir_feed(f)
        except Exception:
            pass

    # 1) GDELT
    try:
        urls = await gdelt_urls_from_rules(rules, minutes=minutes)
        new_urls.update(urls)
    except Exception:
        pass

    # 2) Google News RSS fallback (guarantee some items)
    if len(new_urls) < 5:
        q = '("Talen Energy" OR TLN OR "independent power" OR PJM OR ERCOT OR "power purchase agreement" OR "data center")'
        g_urls = google_news_rss_urls(q, max_items=50)
        new_urls.update(g_urls)

    # Upsert / tag (limit to avoid spam) and skip exact duplicates
    inserted=0
    with db() as con:
        cur_existing = con.execute("select sha256 from article")
        existing_hashes = {row[0] for row in cur_existing.fetchall()}
    for u in list(new_urls)[:30]:
        try:
            if sha256(normalize_url(u)) in existing_hashes:
                continue
            ok = await upsert_article(u, tln_id, "company")
            if ok:
                inserted += 1
        except Exception:
            pass

    # Summarize recent articles (UPSERT so you always get content)
    summarized = 0
    with db() as con:
        cur = con.execute("""
          select a.id, a.title, coalesce(a.clean_text, '')
          from article a
          where a.created_at > now() - interval '3 days'
          order by a.id desc
          limit 100
        """)
        rows = cur.fetchall()

    for aid, title, text in rows:
        try:
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

    return {"ok": True, "found_urls": len(new_urls), "inserted": inserted, "summarized": summarized}

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
          select a.id,a.url,a.title,a.publisher,a.published_at,coalesce(n.summary,'') as summary
          from article a
          left join article_nlp n on n.article_id=a.id
          join article_tag t on t.article_id=a.id and t.equity_id=%s
          where a.published_at between %s and %s and t.tag_kind in ('company')
          order by a.published_at desc
        """,(equity_id, start, end))
        company_items = [dict(zip([d.name for d in cur.description], r)) for r in cur.fetchall()]

    none_found = not company_items

    # Load email template (fallback inline if file missing)
    html = ""
    try:
        with open("email_template.html","r",encoding="utf-8") as f:
            tpl = Template(f.read())
        html = tpl.render(
            run_date=str(run_date),
            end_local=end.strftime("%Y-%m-%d %H:%M"),
            ticker=ticker,
            company_name=company_name,
            company_items=company_items,
            none_found=none_found
        )
    except Exception:
        # simple fallback template
        items_html = "".join(
            f"<li><a href='{i['url']}'><b>{i.get('title') or i['url']}</b></a>"
            f"<div style='font-size:12px;color:#666'>{i.get('publisher','')}"
            f"{' — ' + str(i.get('published_at')) if i.get('published_at') else ''}</div>"
            f"<div>{(i.get('summary') or '').replace('\n','<br>')}</div></li>"
            for i in company_items
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

    # send via Mailgun
    async with httpx.AsyncClient(auth=("api", MAILGUN_API_KEY), timeout=25) as h:
        r = await h.post(f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages", data={
            "from": MAILGUN_FROM,
            "to": owner_email,
            "subject": f"QuantBrief Daily — {run_date} — {ticker}",
            "html": html
        })
        r.raise_for_status()

    with db() as con:
        con.execute("insert into delivery_log(user_id,run_date,items_json) values (%s,%s,%s::jsonb) on conflict do nothing",
                    (user_id, run_date, json.dumps({"company": [c["url"] for c in company_items]})))
    return {"ok": True, "sent_to": owner_email, "items": len(company_items)}

# ---------- health ----------
@app.get("/")
def health():
    return {"ok": True}
