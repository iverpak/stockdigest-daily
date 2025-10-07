# StockDigest Daily Intelligence System - PRIMER v3.5

**Last Updated:** October 8, 2025
**Application File Size:** 15,000+ lines
**Total Endpoints:** 59 (39 admin + 8 job queue + 12 public)
**Database:** PostgreSQL with 14 core tables (includes legal compliance tables)
**Primary Language:** Python 3.11 (FastAPI framework)
**Legal:** Province of Ontario, Canada | CASL & PIPEDA Compliant | Terms v1.0 | Privacy v1.0
**Contact:** stockdigest.research@gmail.com
**New in v3.5:** **Semaphores DISABLED** (fixes deadlock) + 4 concurrent tickers stable + Connection pool: 80 + Executive summary cache_control + **Legal integration** (Terms/Privacy pages, unsubscribe tokens, Email #3 Jinja2 refactor) + **Relaxed ticker validation** (crypto, forex, indices supported) + **Polygon.io fallback** (5 calls/min free tier) + **"Last Close" label** in Email #3 header
**New in v3.4:** ~~Threading semaphores~~ (caused deadlock, removed in v3.5) + Scrapfly Tier 2 + Prompt caching + 24h NEW badge
**New in v3.3:** Parallel ticker processing + connection pooling
**New in v3.2:** Async feed ingestion (5.5x faster)

---

## WHAT THIS APPLICATION DOES

StockDigest is an **AI-powered financial news aggregation and analysis system** that:

1. **Ingests** RSS feeds from 100+ financial news sources (async, 5.5x faster)
2. **Scrapes** full article content using 2-tier fallback (Requests → Scrapfly)
3. **Triages** articles using dual AI scoring (OpenAI + Claude run in parallel)
4. **Summarizes** articles with Claude API (prompt caching enabled, 13% cost savings)
5. **Generates** 3 distinct QA emails per ticker with 24-hour NEW badges
6. **Stores** executive summaries in PostgreSQL for reuse
7. **Commits** processed data to GitHub repository for version control

**Target Users:** Investors, analysts, portfolio managers tracking specific stocks
**Processing Model:** Server-side job queue with parallel processing (2-5 concurrent tickers)

---

## ASYNC FEED INGESTION PERFORMANCE (NEW IN v3.2)

StockDigest now uses **grouped parallel processing** for RSS feed ingestion, dramatically improving performance while maintaining data integrity.

### Performance Improvement

**Before (Sequential):**
- 11 feeds processed one at a time
- ~5 seconds per feed
- **Total: ~55 seconds per ticker**

**After (Grouped Async):**
- Feeds grouped by strategy and processed in parallel
- Google→Yahoo pairs remain sequential (prevents duplicates)
- **Total: ~10 seconds per ticker**
- **Speedup: 5.5x faster!** 🚀

### Feed Structure (11 Feeds Per Ticker)

Each ticker has 11 RSS feeds organized as:

1. **Company Feeds (2 total):**
   - Google News: Company name search
   - Yahoo Finance: Ticker symbol

2. **Industry Feeds (3 total):**
   - Google News only (one feed per industry keyword)
   - No Yahoo feeds for industry

3. **Competitor Feeds (6 total):**
   - 3 competitors × 2 sources each (Google + Yahoo)

### Grouped Async Strategy

```
┌─────────────────────────────────┐
│  Group 1: Company Feeds         │
│  Google → Yahoo (sequential)    │ ← 10 seconds
└─────────────────────────────────┘
         ↓ Run in parallel
┌─────────────────────────────────┐
│  Group 2: Industry Feeds        │
│  Keyword 1, 2, 3 (all parallel) │ ← 5 seconds
└─────────────────────────────────┘
         ↓ Run in parallel
┌─────────────────────────────────┐
│  Group 3: Competitor Feeds      │
│  Comp1: Google → Yahoo (seq)    │
│  Comp2: Google → Yahoo (seq)    │ ← 10 seconds (max of 3)
│  Comp3: Google → Yahoo (seq)    │
└─────────────────────────────────┘

Final Processing Time: max(10s, 5s, 10s) = ~10 seconds
```

### Why Sequential Within Google/Yahoo Pairs?

**Problem:**
Yahoo Finance often redirects to original sources:
- Google finds: `https://reuters.com/article/123`
- Yahoo finds: `https://yahoo.com/finance/...` → redirects to `https://reuters.com/article/123`

**Solution:**
Process Google first, then Yahoo sequentially:
- Google article inserted with `url_hash` based on `reuters.com/article/123`
- Yahoo article resolves to same URL → deduplication catches it
- Prevents duplicate scraping/AI analysis (saves 20-40s + API costs)

**URL Hash Generation:**
```python
def get_url_hash(url: str, resolved_url: str = None) -> str:
    primary_url = resolved_url or url  # Uses RESOLVED URL
    url_clean = re.sub(r'[?&](utm_|ref=|...).*', '', url_lower)
    return hashlib.md5(url_clean.encode()).hexdigest()
```

### Implementation Details

**Technology:**
- `ThreadPoolExecutor` with `max_workers=15`
- Helper function: `process_feeds_sequentially()` (Line 13124)
- Main implementation: `/cron/ingest` endpoint (Line 13155)

**Database Safety:**
- Thread-safe: Each thread gets own DB connection via `@contextmanager`
- Deduplication: `ON CONFLICT (url_hash) DO UPDATE` prevents race conditions
- Connection limit: 11 concurrent << 22-97 available Postgres connections (no pool exhaustion)

**Safety Guarantees:**
✅ **No data corruption** - Sequential G→Y prevents duplicate articles
✅ **Thread-safe operations** - Isolated DB connections per thread
✅ **Error handling preserved** - Failed feeds don't block others
✅ **Stats aggregation maintained** - All metrics accurate
✅ **Memory monitoring continues** - Snapshots every 5 completions

### Monitoring Logs

When async processing runs, you'll see:
```
=== ASYNC FEED PROCESSING: Grouping feeds by strategy ===
Feed groups - Company: 2, Industry: 3, Competitor: 6

=== Starting parallel feed processing with grouped strategy ===
Submitted company feeds for AAPL: 2 feeds (Google→Yahoo sequential)
Submitted 3 industry feeds (all parallel)
Submitted competitor feeds for AAPL/MSFT: 2 feeds (Google→Yahoo sequential)
...

✅ Completed industry feed group: Technology (1/7)
✅ Completed company feed group: AAPL (2/7)
✅ Completed competitor feed group: AAPL/MSFT (3/7)
...

=== ASYNC FEED PROCESSING COMPLETE: 10.23 seconds (7 groups processed) ===
```

---

## 3-EMAIL QUALITY ASSURANCE WORKFLOW (NEW IN v3.0)

StockDigest now generates **3 distinct emails per ticker** during the digest phase, forming a complete QA pipeline. This is the most significant architectural change in v3.0.

### Email #1: Article Selection QA (60% Progress)
**Function:** `send_enhanced_quick_intelligence_email()` - Line 10353
**Subject Format:** `🔍 Article Selection QA: [Company Names] ([Tickers]) - [X] flagged from [Y] articles`

**Purpose:** Quick triage results to verify AI article selection quality

**Content:**
- Shows **ONLY flagged articles** (high relevance scores from AI triage)
- Displays dual AI scoring badges:
  - **Main score (0-10):** Overall relevance to ticker
  - **Category score (0-10):** Strength of category assignment (company/industry/competitor)
- Minimal metadata: title, publisher, timestamp
- **NO full content, NO descriptions**
- Sorted by priority (flagged+quality first, then flagged, then rest)

**Timing:** Sent at ~60% progress (end of ingest phase)

**Example Subject:**
```
🔍 Article Selection QA: General Motors (GM) - 12 flagged from 47 articles
```

### Email #2: Content QA (95% Progress)
**Function:** `fetch_digest_articles_with_enhanced_content()` - Line 10955
**Subject Format:** `📝 Content QA: [Tickers] - [X] articles analyzed`

**Purpose:** Full content review with AI analysis for internal QA

**Content:**
- Shows **ONLY flagged articles** (same filtering as Email #1)
- Full article content (title, description, full text)
- **AI Analysis boxes** with:
  - Key topics and themes
  - Relevance explanation
  - Sentiment indicators
  - Business impact assessment
- **Executive Summary section** (AI-generated overview of all flagged articles)
- Sorted by priority (same algorithm as Email #1)

**Timing:** Sent at ~95% progress (end of digest phase)

**Key Behavior:** Generates and **SAVES executive summary to database** via `save_executive_summary()` (Line 1050)

**Example Subject:**
```
📝 Content QA: GM, F - 18 articles analyzed
```

### Email #3: Premium Stock Intelligence Report (97% Progress) - **UPDATED in v3.5**
**Function:** `send_user_intelligence_report(hours, tickers, flagged_article_ids, recipient_email)` - Line 11873
**Template:** `email_intelligence_report.html` (Jinja2)
**Subject Format:** `📊 Stock Intelligence: [Company Name] ([Ticker]) - [X] articles analyzed`

**Purpose:** Premium user-facing intelligence report with legal disclaimers and professional design

**NEW in v3.5: Full Jinja2 Refactor**
- Replaced 100+ lines of inline HTML with clean Jinja2 template rendering
- Helper functions: `build_executive_summary_html()` (Line 11785), `build_articles_html()` (Line 11818)
- Maintainable, testable, professional code
- Legal compliance integrated (top disclaimer, footer disclaimer, unsubscribe links)

**Content:**
- **Top disclaimer banner:** "For informational purposes only. Not investment advice."
- **Modern HTML template** with gradient blue header
- **Stock price card** in header showing:
  - Today's date (email sent date)
  - Last close price (from `ticker_reference` cache or yfinance/Polygon.io)
  - Daily return with "Last Close" label for clarity (sleek, subtle styling)
- **Executive summary sections** rendered as 6 visual cards:
  1. 🔴 **Major Developments** (3-6 bullets)
  2. 📊 **Financial/Operational Performance** (2-4 bullets)
  3. ⚠️ **Risk Factors** (2-4 bullets)
  4. 📈 **Wall Street Sentiment** (1-4 bullets)
  5. ⚡ **Competitive/Industry Dynamics** (2-5 bullets)
  6. 📅 **Upcoming Catalysts** (1-3 bullets)
- **Compressed article links** at bottom organized by:
  - Company articles
  - Industry articles
  - Competitor articles
- **Visual indicators:**
  - **Star** (★) for FLAGGED + QUALITY articles
  - **NEW badge** (🆕) for articles published <24 hours ago
  - **PAYWALL badge** (red) for paywalled domains
- **Comprehensive footer:**
  - Legal disclaimer box
  - Links: Terms of Service | Privacy Policy | Contact | Unsubscribe (all full URLs)
  - Copyright notice
- Shows **ONLY flagged articles** (same filtering as Email #1 and #2)
- **NO AI analysis boxes, NO descriptions** (clean presentation)
- Uses `resolved_url` for all article links
- Hides sections with no content automatically

**Timing:** Sent at ~97% progress (after Email #2, before GitHub commit)

**Key Behavior:**
- Retrieves executive summary from `executive_summaries` table (no regeneration)
- Parses summary text by emoji headers via `parse_executive_summary_sections()` (Line 11733)
- **Generates unique unsubscribe token** via `get_or_create_unsubscribe_token(recipient_email)`
- **Single-ticker design only** (no multi-ticker support)
- **Requires `recipient_email` parameter** for proper unsubscribe functionality

**Template Variables:**
- `ticker`, `company_name`, `industry`, `current_date`
- `stock_price`, `price_change`, `price_change_color`
- `executive_summary_html` (pre-rendered HTML string)
- `articles_html` (pre-rendered HTML string)
- `total_articles`, `paywalled_count`, `lookback_days`
- `unsubscribe_url` (unique per user)

**Example Subject:**
```
📊 Stock Intelligence: General Motors (GM) - 12 articles analyzed
```

**Design Features:**
- Professional gradient header (blue theme)
- Stock price card with real-time data
- Legal disclaimer banners (top and footer)
- Visual section dividers with horizontal gradient lines
- Article cards with domain name + publication date
- Responsive email-client compatible design
- CASL/CAN-SPAM compliant unsubscribe link

### Flagged Article Filtering (CRITICAL CHANGE in v3.0)

**Email #2 and #3 show ONLY flagged articles** (those with high AI relevance scores).

- **Email #1:** Shows ALL articles but highlights which are flagged (dual AI scoring badges) - sorting logic at Line 10491-10539
- **Email #2:** Filters to flagged only (SQL filter at Line 10996: `AND a.id = ANY(%s)` for flagged_article_ids)
- **Email #3:** Filters to flagged only (parameter at Line 11212: `flagged_article_ids=flagged_article_ids`)

**OLD BEHAVIOR (v2.4):** "Selected" count included ALL QUALITY domain articles even if not flagged
**NEW BEHAVIOR (v3.0):** "Selected" count reflects **ONLY flagged articles** (counted at Line 10435-10441)

This ensures that low-relevance articles from premium sources (WSJ, Bloomberg) are excluded from Email #2 and #3 if the AI determines they're not relevant to the ticker. Email #1 shows all articles for complete QA visibility, with flagged articles clearly highlighted.

### Article Priority Sorting (NEW IN v3.0)

**Function:** `sort_articles_by_priority()` - Line 10278

Within each category (company/industry/competitor), articles are sorted by:
1. **FLAGGED + QUALITY domains** (newest first)
2. **FLAGGED only** (newest first)
3. **All remaining** (newest first)

This sorting is applied to **ALL 3 email reports** to ensure the most important content appears first.

**Quality Domains Defined:**
- Tier 1: WSJ, Bloomberg, Reuters, Financial Times, Barron's
- Tier 2: CNBC, Forbes, MarketWatch, Seeking Alpha
- Tier 3: Yahoo Finance, Business Insider, The Motley Fool
- Tier 4: Lower-quality domains (filtered out unless highly relevant)

### Executive Summary Database Storage (NEW IN v3.0)

**Table:** `executive_summaries` (Line 939)

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS executive_summaries (
    ticker TEXT NOT NULL,
    summary_date DATE NOT NULL,
    summary_text TEXT NOT NULL,
    ai_provider TEXT NOT NULL,
    article_ids INTEGER[],
    company_count INTEGER,
    industry_count INTEGER,
    competitor_count INTEGER,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, summary_date)  -- Overwrites on same-day re-runs
)
```

**Function:** `save_executive_summary()` - Line 1050
- Called during Email #2 generation
- Stores summary with metadata for reuse in Email #3
- Prevents redundant AI calls and ensures consistency between emails
- Automatic overwrite on same-day re-runs (UNIQUE constraint)

**Benefits:**
- 50% reduction in OpenAI API calls (no regeneration for Email #3)
- Guaranteed consistency between Email #2 and Email #3
- Historical executive summary tracking (can query past summaries)
- Supports multi-ticker summaries (article_ids array tracks all sources)

---

## ARCHITECTURE OVERVIEW

### Core Design Principles

1. **Single-File Monolith:** All logic in `app.py` (14,877 lines) - no microservices
2. **PostgreSQL-Centric:** Database as single source of truth (no Redis, no external queues)
3. **Job Queue Pattern:** Background worker polls database, eliminates HTTP 520 errors
4. **AI-First Approach:** Dual scoring (OpenAI + Claude), prompt caching enabled
5. **Anti-Bot Resilience:** 2-tier scraping (Requests → Scrapfly) - Playwright commented out

### Technology Stack

**Backend:**
- FastAPI (Python 3.11)
- PostgreSQL (primary database with connection pooling)
- Claude API (primary, v2024-10-22 with prompt caching)
- OpenAI API (triage + fallback for summaries)

**Scraping Tools:**
- newspaper3k (Tier 1, free, ~70% success)
- Scrapfly (Tier 2, $0.002/article, ~95% success)
- ~~Playwright~~ (commented out - caused hangs)

**Email & Templates:**
- Jinja2 templating
  - `email_intelligence_report.html` - Email #3 (Premium Intelligence Report) with legal disclaimers
  - `email_template.html` - Email #2 (Content QA) - Legacy template
  - `signup.html` - Beta landing page
  - `terms_of_service.html`, `privacy_policy.html` - Legal pages
- SMTP delivery (configurable)
- Toronto timezone (America/Toronto)

**Version Control:**
- GitHub integration via `safe_incremental_commit()` (Line 14707)
- Automatic daily commits of processed data

### Database Schema (14 Tables)

**Content Storage:**
- `articles` - Full article content with URL deduplication
- `ticker_articles` - Many-to-many ticker-article relationships with categorization
- `ticker_references` - Company metadata (name, exchange, industry)
- `feeds` - RSS feed sources (shareable across tickers)
- `ticker_feeds` - Many-to-many ticker-feed relationships with per-feed categories

**AI-Generated Content (NEW IN v3.0):**
- `executive_summaries` - Daily AI-generated summaries (Line 939)
  - Columns: ticker, summary_date, summary_text, ai_provider, article_ids, counts, generated_at
  - UNIQUE(ticker, summary_date) - overwrites on same-day re-runs
  - Generated during Email #2, reused in Email #3

**Beta User Management (UPDATED IN v3.5):**
- `beta_users` - Beta signup data with legal compliance tracking
  - Core fields: name, email, ticker1, ticker2, ticker3, status, created_at
  - **Legal tracking (NEW in v3.5):**
    - `terms_version` VARCHAR(10) DEFAULT '1.0' - Terms of Service version
    - `terms_accepted_at` TIMESTAMPTZ - When user accepted Terms
    - `privacy_version` VARCHAR(10) DEFAULT '1.0' - Privacy Policy version
    - `privacy_accepted_at` TIMESTAMPTZ - When user accepted Privacy
  - UNIQUE constraint on email
  - Status field: 'active' | 'paused' | 'cancelled'
  - Exported daily to `data/user_tickers.csv` for morning processing

**Unsubscribe System (NEW IN v3.5):**
- `unsubscribe_tokens` - Token-based unsubscribe for CASL/CAN-SPAM compliance
  - `token` VARCHAR(64) UNIQUE - Cryptographically secure (43-char URL-safe, 256-bit entropy)
  - `user_email` VARCHAR(255) - Foreign key to beta_users(email)
  - `created_at` TIMESTAMPTZ - Token generation time
  - `used_at` TIMESTAMPTZ - When token was used (NULL if unused)
  - `ip_address` VARCHAR(45) - Security tracking
  - `user_agent` TEXT - Security tracking
  - One token per user, reusable until unsubscribed
  - Indexed on token, email, and used_at

**Job Queue System:**
- `ticker_processing_batches` - Batch tracking (status, job counts, config)
- `ticker_processing_jobs` - Individual ticker jobs with full audit trail
  - Includes: retry logic, timeout protection, resource tracking, error stacktraces
  - Atomic job claiming via `FOR UPDATE SKIP LOCKED` (prevents race conditions)

**Other:**
- `domain_names` - Formal domain name mappings (AI-generated)
- `competitor_metadata` - Ticker competitor relationships
- `domain_strategies` - Scraping strategy overrides per domain

---

## JOB QUEUE SYSTEM (PRODUCTION ARCHITECTURE)

### The Problem It Solves

**Before (Broken):**
```
PowerShell → HTTP POST /cron/digest → 30 min processing → 520 timeout after 60-120s
```

**After (Production):**
```
PowerShell → POST /jobs/submit (<1s) → Instant response with batch_id
Background Worker → Process jobs → Update database
PowerShell → GET /jobs/batch/{id} (<1s) → Real-time status (poll every 20s)
```

The job queue system **decouples long-running ticker processing from HTTP request lifecycles**. All processing happens server-side in a background worker thread, with PowerShell polling for status instead of maintaining long HTTP connections.

### Key Components

**1. Background Worker Thread**
- Polls database every 10 seconds for queued jobs
- Processes jobs sequentially using `TICKER_PROCESSING_LOCK` (ensures ticker isolation)
- Updates progress in real-time (phase, progress %, memory, duration)
- Survives server restarts (state persists in PostgreSQL)
- **Starts automatically on FastAPI startup** (see `@APP.on_event("startup")`)

**2. Circuit Breaker**
- Detects 3+ consecutive **system failures** (DB crashes, memory exhaustion)
- Automatically halts processing when open (state: closed | open)
- Auto-closes after 5 minutes
- Does **NOT** trigger on individual ticker failures
- Manual reset: `POST /jobs/circuit-breaker/reset`

**3. Timeout Watchdog**
- Separate thread that monitors jobs every 60 seconds
- Marks jobs exceeding 45 minutes as timeout
- Updates batch counters automatically

**4. Retry Logic**
- Jobs can retry up to 2 times on transient failures
- Retry count tracked per job
- Exponential backoff between retries

### Job Processing Flow

```python
1. Client submits batch → Jobs created in database (status: queued)
2. Worker polls database → Claims job atomically (FOR UPDATE SKIP LOCKED)
3. Update status: processing, acquire TICKER_PROCESSING_LOCK
4. Phase 1: Ingest (RSS, AI triage, Email #1) → Update progress: 60%
5. Phase 2: Digest (scrape, AI analysis, Email #2) → Update progress: 95%
6. Email #3: User intelligence report (fetch summary from DB) → Update progress: 97%
7. Phase 3: GitHub commit → Update progress: 99%
8. Mark complete → Release lock → Poll for next job
```

**Email Timeline:**
- Email #1 (Article Selection QA): Sent at 60% progress (end of ingest phase)
- Email #2 (Content QA): Sent at 95% progress (end of digest phase, saves executive summary)
- Email #3 (Stock Intelligence): Sent at 97% progress (fetches executive summary from database)

### Production Features

✅ **Ticker Isolation** - Uses existing `TICKER_PROCESSING_LOCK`, ticker #2 never interrupts #1
✅ **Zero Infrastructure Cost** - No Redis, no Celery, just PostgreSQL
✅ **Resume Capability** - PowerShell can disconnect/reconnect (state in DB)
✅ **Full Audit Trail** - Stacktraces, timestamps, worker_id, memory, duration
✅ **Real-Time Progress** - See exactly what phase each ticker is in
✅ **Automatic Retries** - Up to 2 retries on transient failures
✅ **Circuit Breaker** - Detects system failures, prevents cascading errors
✅ **Timeout Protection** - Jobs auto-killed after 45 minutes

---

## API ENDPOINTS (56 TOTAL)

### Job Queue Endpoints (8 endpoints - Production)
- `POST /jobs/submit` - Submit batch of tickers for server-side processing
- `GET /jobs/batch/{batch_id}` - Get real-time status of all jobs in batch
- `GET /jobs/{job_id}` - Get detailed job status (includes stacktraces)
- `POST /jobs/circuit-breaker/reset` - Manually reset circuit breaker
- `GET /jobs/stats` - Queue statistics and worker health
- `GET /health` - Worker health check (prevents Render idle timeout)
- `POST /jobs/cancel/{job_id}` - Cancel a running job
- `GET /jobs/history` - Job execution history

### Admin Endpoints (39 endpoints - require X-Admin-Token header)

**Initialization & Setup:**
- `POST /admin/init` - Initialize ticker feeds and sync reference data
- `POST /admin/wipe-database` - Complete database reset
- `POST /admin/sync-ticker-reference` - Sync ticker reference from CSV

**Feed Management:**
- `POST /admin/clean-feeds` - Clean old articles beyond time window
- `POST /admin/add-feed` - Add new RSS feed
- `POST /admin/remove-feed` - Remove RSS feed
- `GET /admin/feeds` - List all feeds

**Ticker Operations:**
- `POST /admin/force-digest` - Generate digest emails for specific tickers
- `GET /admin/ticker-metadata/{ticker}` - Retrieve ticker configuration
- `POST /admin/update-ticker-metadata` - Update ticker configuration
- `POST /admin/reset-digest-flags` - Reset digest sent flags

**Article Management:**
- `POST /admin/flag-article` - Manually flag article as important
- `POST /admin/unflag-article` - Remove flag from article
- `GET /admin/flagged-articles/{ticker}` - List flagged articles for ticker

**Debugging & Diagnostics:**
- `GET /admin/debug/digest/{ticker}` - Debug digest generation (Line 14495)
- `GET /admin/debug/triage/{ticker}` - Debug AI triage results
- `GET /admin/debug/scrape/{url}` - Test scraping strategies

**And 25+ more admin endpoints...**

### Legacy Automation Endpoints (Direct HTTP - Subject to Timeouts)
- `POST /cron/ingest` - RSS feed processing and article discovery (Line 12695)
- `POST /cron/digest` - Email digest generation and delivery (Line 13303)
- `POST /admin/safe-incremental-commit` - GitHub commit (Line 14707)

### Public Endpoints (12 endpoints - No Authentication Required)
- `GET /` - Beta landing page with legal disclaimers (HTML)
  - Top disclaimer banner: "For informational purposes only"
  - Required Terms/Privacy checkbox before signup
  - Footer links: Terms | Privacy | Contact
- `GET /terms-of-service` - Terms of Service page (NEW in v3.5)
  - Province of Ontario, Canada jurisdiction
  - Contact: stockdigest.research@gmail.com
  - Last Updated: October 7, 2025 (v1.0)
- `GET /privacy-policy` - Privacy Policy page (NEW in v3.5)
  - PIPEDA compliant (Canadian privacy law)
  - GDPR/CCPA rights included
  - Last Updated: October 7, 2025 (v1.0)
- `GET /unsubscribe?token=xxx` - Token-based unsubscribe (NEW in v3.5)
  - Validates cryptographic token
  - Idempotent (safe to click multiple times)
  - Security tracking (IP, user agent)
  - Branded success/error HTML pages
  - CASL/CAN-SPAM compliant
- `GET /api/validate-ticker` - Live ticker validation with Canadian .TO suggestions
- `POST /api/beta-signup` - Beta user signup form submission
  - Now logs terms acceptance timestamp + version
  - Generates unsubscribe token automatically
- `GET /health` - Health check

---

## DATABASE-FIRST TRIAGE SELECTION (NEW IN v3.1)

### The Critical Change

**OLD BEHAVIOR (v3.0 and earlier):**
```sql
WHERE ta.found_at >= cutoff  -- Only articles discovered recently
AND (a.published_at >= cutoff OR NULL)
```

**NEW BEHAVIOR (v3.1):**
```sql
WHERE ta.ticker = %s
AND (a.published_at >= cutoff OR NULL)
-- NO found_at filter - database is source of truth
```

### How It Works

1. **RSS Feed Phase** - Discovers and ingests NEW articles (up to 50/25/25 per category)
   - Spam filtering happens during ingestion (before database)
   - Articles persist in database **indefinitely** (no automatic cleanup)
   - Ingestion limits: 50 company, 25 per industry keyword, 25 per competitor

2. **Triage Selection Phase** - Queries database for articles to triage
   - Pulls latest 50/25/25 articles by `published_at` (not `found_at`)
   - **Database is source of truth**, not RSS feed results
   - Articles ranked within each category/keyword partition
   - Lookback window (1 day, 7 days, 1 month) filters by publication date

### Benefits

✅ **RSS Feed Gaps Don't Matter** - Even if article disappears from feed, it's still in database
✅ **Slow-Moving Tickers Get Content** - Fills 50-article limit with older quality articles
✅ **Fast-Moving Tickers Get Latest** - NVDA fills limit with today's news only
✅ **Consistent Triage** - Same articles available day-to-day
✅ **No Article Loss** - Database accumulates complete history

### Example Scenarios

**Scenario 1: Fast-Moving Ticker (NVDA)**
- **Lookback:** 1 day
- **Database:** 150 articles published today
- **Triage:** Latest 50 articles from today (newest first)

**Scenario 2: Slow-Moving Ticker (Obscure Small Cap)**
- **Lookback:** 7 days
- **Database:** 30 articles published across 7 days
- **Triage:** All 30 articles (fills to 50 if possible, otherwise uses what's available)

**Scenario 3: RSS Feed Gap**
- **Day 1:** RSS returns 80 articles, 50 inserted to DB
- **Day 2:** RSS only returns 40 articles (20 disappeared!)
- **Old behavior:** Only 40 articles available for triage ❌
- **New behavior:** All 50 from Day 1 + 40 from Day 2 = 90 total in DB ✅

### Implementation

**Query Location:** Lines 12862-12916 in `cron_ingest()`
**Filters Applied:**
- `WHERE ta.ticker = %s` - Only articles for this ticker
- `AND (a.published_at >= cutoff OR NULL)` - Within lookback window
- `ORDER BY a.published_at DESC NULLS LAST` - Newest first
- `ROW_NUMBER() OVER (PARTITION BY category/keyword)` - Limit per partition

---

## KEY FUNCTION LOCATIONS (v3.1)

### 3-Email System
- `send_enhanced_quick_intelligence_email()` - **Line 10353** (Email #1: Article Selection QA)
- `fetch_digest_articles_with_enhanced_content()` - **Line 10955** (Email #2: Content QA)
- `send_user_intelligence_report(hours, tickers, flagged_article_ids, recipient_email)` - **Line 11873** (Email #3: Premium Intelligence Report)
  - **NEW in v3.5:** Jinja2 template refactor with legal disclaimers
  - **Requires:** `recipient_email` parameter for unsubscribe token generation
- `build_executive_summary_html(sections)` - **Line 11785** (Helper: Render summary sections as HTML)
- `build_articles_html(articles_by_category)` - **Line 11818** (Helper: Render article links as HTML)
- `parse_executive_summary_sections()` - **Line 11733** (Parse AI summary into 6 sections)
- `sort_articles_by_priority()` - **Line 10278** (Article priority sorting)
- `save_executive_summary()` - **Line 1050** (Executive summary database storage)
- `generate_openai_executive_summary()` - **Line 10069** (Executive summary AI prompt)

### Unsubscribe System (NEW in v3.5)
- `generate_unsubscribe_token(email)` - **Line 13055** (Generate cryptographic token)
- `get_or_create_unsubscribe_token(email)` - **Line 13085** (Get existing or create new token)
- `/unsubscribe` endpoint handler - **Line 12981** (Token validation + unsubscribe processing)

### Job Queue System
- `process_digest_phase()` - **Line 11393** (Main digest phase orchestrator)
- Background worker thread initialization - See `@APP.on_event("startup")`

### Content Processing
- `perform_ai_triage_batch()` - **Line 5679** (OpenAI AI triage)
- `rule_based_triage_score_company()` - **Line 5783** (Rule-based triage fallback)
- `rule_based_triage_score_industry()` - **Line 5886** (Industry triage fallback)
- `scrape_with_scrapfly_async()` - **Line 3172** (ScrapFly scraping)
- `clean_scraped_content()` - **Line 3346** (Content sanitization)
- `update_article_content()` - **Line 1010** (Update article in database)

### Legacy Endpoints
- `cron_ingest()` - **Line 12695** (RSS feed processing)
- `cron_digest()` - **Line 13303** (Digest generation)
- `safe_incremental_commit()` - **Line 14707** (GitHub commit)

### Database Operations
- `ensure_schema()` - **Line 700** (Schema initialization)
- `get_connection()` - Database connection pool management

---

## CONTENT PROCESSING PIPELINE

### Phase 1: Ingest (0-60% Progress)

**Step 1: RSS Feed Parsing**
- Fetches feeds from `ticker_feeds` table
- Deduplicates by URL hash
- Filters by time window (default: 24 hours)

**Step 2: AI Triage (OpenAI GPT-4o-mini)**
- Batch processing (2-3 articles per call)
- Scores relevance (0-10) and category strength (0-10)
- Categories: company, industry, competitor, market
- Flags high-scoring articles for inclusion

**Step 3: Email #1 Sent**
- Shows only flagged articles with AI scores
- Enables quick validation of AI selection quality

### Phase 2: Digest (60-95% Progress)

**Step 1: Content Scraping**
- 2-tier fallback approach:
  1. **newspaper3k** (Tier 1: Requests, free, ~70% success)
  2. **Scrapfly** (Tier 2: Premium API, $0.002/article, ~95% success)
- Domain-specific strategy overrides (skip Tier 1 for known JS-heavy sites)

**Step 2: AI Analysis (Claude Sonnet 4.5)**
- Generates detailed analysis for each flagged article
- Extracts key topics, themes, sentiment
- Assesses business impact
- Prompt caching enabled (90% discount on repeated prompts)

**Step 3: Executive Summary Generation**
- Claude Sonnet 4.5 generates overview of all flagged articles
- OpenAI GPT-4 fallback if Claude fails
- Saved to `executive_summaries` table (Line 1050)

**Step 4: Email #2 Sent**
- Full content + AI analysis boxes
- Executive summary section
- Internal QA review

### Phase 3: Email #3 Generation (95-97% Progress)

**Step 1: Fetch Executive Summary**
- Retrieves pre-generated summary from database
- No redundant AI calls

**Step 2: Email #3 Sent**
- Clean user-facing report
- No AI analysis boxes
- Same flagged articles as Email #2

### Phase 4: GitHub Commit (97-99% Progress)

**Step 1: Incremental Commit**
- Commits processed data to GitHub repository
- Includes article metadata, AI analysis, executive summaries
- Function: `safe_incremental_commit()` - Line 14707

**Step 2: Job Complete (100%)**
- Updates job status to 'completed'
- Releases `TICKER_PROCESSING_LOCK`
- Worker polls for next job

---

## ANTI-BOT & SCRAPING STRATEGIES

### Domain Strategy Detection

**Function:** `get_domain_strategy()` (check codebase for line number)

StockDigest maintains a domain-specific scraping strategy system:

**Tier 1: Direct Scraping (newspaper3k)**
- Most financial news sites (Yahoo Finance, CNBC, Reuters)
- Fast, reliable, respects robots.txt
- 80% success rate

**Tier 2: Playwright (Browser Automation)**
- Anti-bot domains (WSJ, Bloomberg, Barron's)
- Executes JavaScript, bypasses client-side blocking
- Slower but 95% success rate
- User-agent rotation, referrer spoofing

**Tier 3: Premium Proxies (ScrapingBee, ScrapFly)**
- Last resort for heavily protected domains
- Residential proxies, CAPTCHA solving
- 99% success rate but API costs
- Rate-limited to avoid abuse

### Rate Limiting & Concurrency

- **Semaphores:** **DISABLED as of v3.5** (threading.BoundedSemaphore caused deadlock with 3+ concurrent tickers)
- **API rate limiting:** Handled by API providers (429 errors retried automatically)
- **Domain-level delays:** 2-5 seconds between requests to same domain
- **Circuit breaker:** Halts processing after 3 consecutive system failures
- **Concurrent tickers:** Recommended 4, supports up to 5
- **Timeout protection:** 45-minute max per ticker job

---

## FINANCIAL DATA & TICKER VALIDATION (NEW IN v3.5)

### Relaxed Ticker Validation

StockDigest now supports a wide range of ticker formats beyond traditional US stocks.

**Supported Formats:**
- ✅ **Regular US stocks:** AAPL, MSFT, TSLA, NVDA
- ✅ **International stocks:** RY.TO (Canada), BP.L (London), SAP.DE (Germany), 005930.KS (Korea)
- ✅ **Cryptocurrency:** BTC-USD, ETH-USD, SOL-USD, BNB-USD
- ✅ **Forex pairs:** EURUSD=X, CADJPY=X, GBPUSD=X, CAD=X
- ✅ **Market indices:** ^GSPC (S&P 500), ^DJI (Dow), ^IXIC (NASDAQ)
- ✅ **ETFs:** SPY, VOO, QQQ, IVV
- ✅ **Class shares:** BRK-A, BRK-B, TECK-A.TO
- ✅ **Private company competitors:** No ticker required (Google News only)

**Key Changes:**
- `validate_ticker_format()` - 15+ regex patterns (Line 1479)
- `normalize_ticker_format()` - Preserves special characters: `^`, `=`, `-`, `.` (Line 1542)
- Fallback config prevents crashes for unknown tickers

**Special Character Handling:**
- Caret (^) for indices: `^GSPC` ✅
- Equals (=) for forex: `EURUSD=X` ✅
- Hyphen (-) for crypto: `BTC-USD` ✅
- Dot (.) for exchanges: `RY.TO` ✅

### Financial Data with Polygon.io Fallback

**2-Tier Data Fetching:**

```
┌─────────────────────────────────────────────────┐
│ 1. yfinance (Primary)                           │
│    ├─ Full data (13 fields)                     │
│    ├─ Includes: market cap, analysts, volume    │
│    ├─ 3 retries with exponential backoff        │
│    └─ ~48 calls/minute (undocumented limit)     │
└─────────────────────────────────────────────────┘
                    ↓ If fails
┌─────────────────────────────────────────────────┐
│ 2. Polygon.io (Fallback)                        │
│    ├─ Minimal data (price + daily return)       │
│    ├─ Free tier: 5 calls/minute                 │
│    ├─ Rate limited with automatic sleep         │
│    └─ Enough for Email #3 header card           │
└─────────────────────────────────────────────────┘
                    ↓ If fails
┌─────────────────────────────────────────────────┐
│ 3. Return None                                  │
│    Email #3 shows "N/A" for price data          │
└─────────────────────────────────────────────────┘
```

**Key Functions:**
- `get_stock_context()` - Main entry point (Line 1966)
- `get_stock_context_polygon()` - Polygon.io fallback (Line 1895)
- `_wait_for_polygon_rate_limit()` - Rate limiter for free tier (Line 1874)

**Validation Changes:**
- **Old:** Required both `currentPrice` AND `marketCap` (rejected forex/indices)
- **New:** Only requires `currentPrice` (market cap optional)
- **Result:** Forex (EURUSD=X) and Indices (^GSPC) now work! ✅

**Email #3 Requirements:**
- Only needs: `financial_last_price` and `financial_price_change_pct`
- Both yfinance and Polygon.io provide these fields
- Header displays: "+2.5% Last Close" (sleek label)

**Environment Setup:**
```bash
POLYGON_API_KEY=your_api_key_here  # Get free at polygon.io
```

**Rate Limiting:**
- Polygon.io free tier: 5 calls/minute
- Sliding window with automatic sleep
- Logs: `⏳ Polygon.io rate limit reached, waiting 12.3s...`

**Data Returned:**

| Field | yfinance | Polygon.io | Email #3 Needs |
|-------|----------|------------|----------------|
| Last Price | ✅ | ✅ | ✅ YES |
| Price Change % | ✅ | ✅ | ✅ YES |
| Market Cap | ✅ | ❌ | ❌ No |
| Volume | ✅ | ✅ | ❌ No |
| Analyst Data | ✅ | ❌ | ❌ No |

---

## POWERSHELL AUTOMATION WORKFLOW

### Production Workflow (Job Queue System)

**Script:** `.\scripts\setup_job_queue.ps1`

**Flow:**
1. Submit batch of tickers via `POST /jobs/submit`
2. Receive instant response with `batch_id`
3. Poll `GET /jobs/batch/{batch_id}` every 20 seconds
4. Display real-time progress:
   ```
   Progress: 75% | Completed: 3/4 | Failed: 0 | Current: CEG [digest_complete]
   ✅ RY.TO: completed (28.5min) [mem: 215MB]
   ✅ TD.TO: completed (30.2min) [mem: 234MB]
   🔄 CEG: processing [digest_complete - 75%]
   ⏳ SO: queued
   ```
5. Exit when batch complete or failed

**Benefits:**
- No HTTP 520 timeouts (instant responses)
- Resume capability (PowerShell can disconnect/reconnect)
- Real-time visibility into processing state
- Full error stacktraces on failure

### Legacy Workflow (Direct HTTP - Deprecated)

**Script:** `.\scripts\setup.ps1`

**Flow:**
1. `POST /cron/ingest` (subject to timeout)
2. `POST /cron/digest` (subject to timeout)
3. `POST /admin/safe-incremental-commit` (subject to timeout)

**Issues:**
- 520 errors after 60-120 seconds
- No progress visibility
- Cannot resume on disconnect
- Replaced by job queue system

---

## CONFIGURATION & ENVIRONMENT VARIABLES

### Required Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:port/dbname

# AI Provider
OPENAI_API_KEY=sk-...

# Authentication
ADMIN_TOKEN=your-secret-token

# Email (SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=StockDigest <your-email@gmail.com>
SMTP_TO=recipient@example.com

# Scraping (Optional - Premium Services)
SCRAPINGBEE_API_KEY=...
SCRAPFLY_API_KEY=...

# GitHub (Optional - for commits)
GITHUB_TOKEN=ghp_...
GITHUB_REPO=username/quantbrief-data
```

### Default Processing Parameters

```python
# Time window for article discovery
TIME_WINDOW_MINUTES = 1440  # 24 hours

# AI triage batch size
TRIAGE_BATCH_SIZE = 2-3  # articles per OpenAI call

# Scraping concurrency
MAX_CONCURRENT_SCRAPES = 5

# Job timeout
JOB_TIMEOUT_MINUTES = 45

# Circuit breaker
MAX_CONSECUTIVE_FAILURES = 3
CIRCUIT_BREAKER_COOLDOWN_MINUTES = 5

# Default tickers (if none specified)
DEFAULT_TICKERS = ["MO", "GM", "ODFL", "SO", "CVS"]
```

---

## MONITORING & DEBUGGING

### Job Queue Monitoring

**Check worker health:**
```bash
curl https://stockdigest.app/jobs/stats \
  -H "X-Admin-Token: $TOKEN"
```

**Response:**
```json
{
  "worker_status": "running",
  "circuit_breaker": "closed",
  "queued_jobs": 0,
  "processing_jobs": 1,
  "completed_jobs_24h": 12,
  "failed_jobs_24h": 0,
  "avg_processing_time_minutes": 28.5
}
```

**Check batch status:**
```bash
curl https://stockdigest.app/jobs/batch/{batch_id} \
  -H "X-Admin-Token: $TOKEN"
```

**Check specific job (includes full stacktrace):**
```bash
curl https://stockdigest.app/jobs/{job_id} \
  -H "X-Admin-Token: $TOKEN"
```

### SQL Debugging Queries

**See all active jobs:**
```sql
SELECT job_id, ticker, status, phase, progress,
       EXTRACT(EPOCH FROM (NOW() - started_at))/60 as minutes_running
FROM ticker_processing_jobs
WHERE status = 'processing';
```

**See recent failures:**
```sql
SELECT ticker, error_message, created_at
FROM ticker_processing_jobs
WHERE status IN ('failed', 'timeout')
  AND created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;
```

**Queue depth:**
```sql
SELECT COUNT(*) FROM ticker_processing_jobs WHERE status = 'queued';
```

**Executive summary history:**
```sql
SELECT ticker, summary_date, ai_provider,
       company_count, industry_count, competitor_count
FROM executive_summaries
WHERE ticker = 'GM'
ORDER BY summary_date DESC
LIMIT 10;
```

### Debug Endpoints

**Test digest generation:**
```bash
curl -X GET https://stockdigest.app/admin/debug/digest/GM \
  -H "X-Admin-Token: $TOKEN"
```

**Test AI triage:**
```bash
curl -X GET https://stockdigest.app/admin/debug/triage/GM \
  -H "X-Admin-Token: $TOKEN"
```

**Test scraping strategy:**
```bash
curl -X GET "https://stockdigest.app/admin/debug/scrape?url=https://example.com/article" \
  -H "X-Admin-Token: $TOKEN"
```

---

## COMMON ISSUES & TROUBLESHOOTING

### Issue: Jobs stuck in 'processing' state

**Cause:** Server restart during job execution
**Solution:**
```sql
-- Manually reset stuck jobs
UPDATE ticker_processing_jobs
SET status = 'failed',
    error_message = 'Job interrupted by server restart'
WHERE status = 'processing'
  AND started_at < NOW() - INTERVAL '1 hour';
```

### Issue: Circuit breaker open

**Cause:** 3+ consecutive system failures
**Solution:**
```bash
# Manual reset
curl -X POST https://stockdigest.app/jobs/circuit-breaker/reset \
  -H "X-Admin-Token: $TOKEN"
```

### Issue: Email #3 missing executive summary

**Cause:** Email #2 failed or didn't save summary to database
**Solution:**
```sql
-- Check if summary exists
SELECT * FROM executive_summaries
WHERE ticker = 'GM' AND summary_date = CURRENT_DATE;

-- If missing, re-run digest for that ticker
```

### Issue: Duplicate articles in emails

**Cause:** URL deduplication failure (rare)
**Solution:**
```sql
-- Find duplicates
SELECT url, COUNT(*)
FROM articles
GROUP BY url
HAVING COUNT(*) > 1;

-- Clean duplicates (keep newest)
DELETE FROM articles
WHERE id NOT IN (
  SELECT MAX(id) FROM articles GROUP BY url
);
```

### Issue: High memory usage

**Cause:** Too many concurrent tickers or memory leak
**Solution:**
- Check `memory_monitor.py` logs for resource tracking
- Recommended: `MAX_CONCURRENT_JOBS=4` (~1200MB total)
- Maximum: `MAX_CONCURRENT_JOBS=5` (~1500MB total, tight margins)
- Memory per ticker: ~300MB (down from 400MB with Playwright removed)
- Manual cleanup: Restart worker (server restart)

---

## VERSION HISTORY

### v3.5 (October 7, 2025) - Current
**Critical Fixes & Production Optimization:**
- **⚠️ SEMAPHORES DISABLED:** Removed all semaphore acquisitions
  - **Problem:** threading.BoundedSemaphore blocked threads, freezing async event loops → deadlock at 3+ tickers
  - **Symptom:** Jobs stuck at 10%-60% progress with no errors
  - **Solution:** Disabled all 11 semaphore acquisitions; APIs enforce their own rate limits
  - **Impact:** **4 concurrent tickers now stable** (previously deadlocked at 3)
- **Connection pool upgrade:** 45 → 80 max (for Basic-1GB database with 100 connection limit)
  - **Impact:** Supports 4-5 concurrent tickers comfortably
- **Executive summary cache_control:** Added proper system/user split with caching
  - **Fix:** Prevents Claude API 520 errors; ~2000 tokens cached per summary
- **NEW emoji spacing:** Added space after 🆕 for readability
- **Production config:** `MAX_CONCURRENT_JOBS=4` recommended (up to 5 supported)
- **Performance:** **4x speedup** (4 tickers: 30 min vs 120 min sequential)

**Legal Integration (October 7, 2025):**
- **Terms & Privacy pages:** Province of Ontario, Canada | CASL & PIPEDA compliant
- **Beta signup updates:** Logs terms acceptance timestamp + version (v1.0)
- **Token-based unsubscribe:** Cryptographic tokens (256-bit entropy), CASL/CAN-SPAM compliant
- **Email #3 Jinja2 refactor:** Replaced 100+ lines of inline HTML with clean template
  - New template: `email_intelligence_report.html`
  - Helper functions: `build_executive_summary_html()`, `build_articles_html()`
  - Legal disclaimers: Top banner + comprehensive footer
- **New database tables:** `unsubscribe_tokens` (security tracking)
- **Updated schema:** `beta_users` now tracks terms/privacy versions + acceptance timestamps
- **New public endpoints:** `/terms-of-service`, `/privacy-policy`, `/unsubscribe`

### v3.4 (October 7, 2025) - DEPRECATED
**Issues:** Threading semaphores caused deadlock with 3+ concurrent tickers (fixed in v3.5)
- **Scrapfly Tier 2:** Playwright commented out (caused hangs)
- **Prompt caching:** Claude API v2024-10-22 (~13% cost reduction)
- **24-hour NEW badge:** Shows green badge for articles <24 hours old
- **File size:** 15,000+ lines

### v3.3 (October 6, 2025)
**Major Changes:**
- Parallel ticker processing (2-5 concurrent tickers)
- Connection pooling (psycopg_pool)
- ThreadPoolExecutor job worker
- Smart GitHub commits (only last ticker triggers deployment)

### v3.2 (October 5, 2025)
**Major Changes:**
- Async feed ingestion (5.5x faster)
- Grouped parallel processing for RSS feeds

### v3.0 (October 5, 2025)
**Major Changes:**
- 3-Email QA workflow (Article Selection → Content QA → Stock Intelligence)
- Executive summary database storage (eliminates redundant AI calls)
- Flagged article filtering (all emails show only flagged content)
- Article priority sorting (flagged+quality first)
- File size: 14,877 lines (from 12,537)
- 56 total endpoints (from 53)

### v2.4 (Previous)
- 2-Email system (triage + digest)
- No executive summary storage (regenerated for each email)
- Mixed flagged/quality filtering
- File size: 12,537 lines
- 53 total endpoints

### v2.0-2.3
- Job queue system implementation
- Circuit breaker pattern
- Timeout watchdog
- Background worker thread

### v1.x
- Direct HTTP processing (legacy)
- Single digest email
- No job queue

---

## DEVELOPMENT NOTES

### Code Organization

**Single-file monolith strategy:**
- All functionality in `app.py` (14,877 lines)
- Sections organized by concern (database → scraping → AI → emails → endpoints)
- Extensive logging for debugging
- All functions use type hints

### Testing Strategy

**Manual testing via admin endpoints:**
- `/admin/debug/digest/{ticker}` - Test full digest pipeline
- `/admin/debug/triage/{ticker}` - Test AI triage only
- `/admin/debug/scrape?url=...` - Test scraping strategies

**SQL-based validation:**
- Check `ticker_processing_jobs` for job success/failure
- Check `executive_summaries` for AI-generated content
- Check `articles` for scraping success rate

### Performance Considerations

**Bottlenecks:**
1. **Claude/OpenAI API calls** (5-15s per batch)
   - Mitigated by prompt caching (90% discount on repeated prompts)
   - Mitigated by batch processing (2-3 articles per call)
2. **Scrapfly scraping** (3-5s per article for Tier 2)
   - Mitigated by domain strategy detection (70% use free Tier 1)
3. **Database queries** (1-5s for complex joins)
   - Mitigated by indexing on (ticker, published_at)

**Optimization strategies:**
- Async/await for I/O operations
- ~~Semaphore-based concurrency control~~ (disabled in v3.5, caused deadlock)
- Database connection pooling (80 max connections)
- Executive summary caching with prompt caching (v3.0/v3.5)
- 4 concurrent tickers (4x speedup vs sequential)

### Future Enhancements (Roadmap)

- [ ] Multi-language support (non-English articles)
- [ ] Historical trend analysis (price correlations)
- [ ] Custom alert thresholds (user-defined triggers)
- [ ] Web UI for job queue monitoring
- [ ] Webhook support (Slack, Discord, Teams)
- [ ] Article deduplication across tickers (global)

---

## QUICK REFERENCE CARD

**Start Processing:**
```powershell
.\scripts\setup_job_queue.ps1
```

**Check Status:**
```bash
curl https://stockdigest.app/jobs/stats -H "X-Admin-Token: $TOKEN"
```

**Reset Circuit Breaker:**
```bash
curl -X POST https://stockdigest.app/jobs/circuit-breaker/reset -H "X-Admin-Token: $TOKEN"
```

**Debug Ticker:**
```bash
curl https://stockdigest.app/admin/debug/digest/GM -H "X-Admin-Token: $TOKEN"
```

**View Executive Summaries:**
```sql
SELECT * FROM executive_summaries WHERE ticker = 'GM' ORDER BY summary_date DESC;
```

**Key Files:**
- `app.py` - Main application (15,000+ lines)
- `email_intelligence_report.html` - Email #3 Jinja2 template (NEW in v3.5)
- `email_template.html` - Email #2 Jinja2 template (legacy)
- `signup.html` - Beta landing page with legal disclaimers
- `terms_of_service.html` - Terms of Service page (v1.0)
- `privacy_policy.html` - Privacy Policy page (v1.0)
- `memory_monitor.py` - Resource tracking
- `data/ticker_reference.csv` - Ticker metadata
- `scripts/setup_job_queue.ps1` - Production automation

**Key Functions:**
- Email #1: Line 10353
- Email #2: Line 10955
- Email #3: Line 11873 (updated in v3.5)
- Article Sorting: Line 10278
- Executive Summary Save: Line 1050
- Digest Phase: Line 11626 (updated in v3.5)
- Unsubscribe Token Generation: Line 13055

---

**END OF PRIMER v3.0**
