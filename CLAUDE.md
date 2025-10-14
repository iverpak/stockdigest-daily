# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Information

**Name:** StockDigest
**Domain:** https://stockdigest.app
**GitHub:** https://github.com/iverpak/stockdigest-daily
**Database:** stockdigest-db (PostgreSQL on Render)
**Legal:** Province of Ontario, Canada | CASL & PIPEDA Compliant
**Contact:** stockdigest.research@gmail.com

## Development Commands

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server locally
uvicorn app:APP --host 0.0.0.0 --port 8000

# Run with auto-reload for development
uvicorn app:APP --reload --host 0.0.0.0 --port 8000
```

### PowerShell Automation Scripts

**NEW (Recommended): Job Queue System**
```powershell
# Execute via server-side job queue (no HTTP timeouts)
.\scripts\setup_job_queue.ps1
```

The job queue system decouples long-running processing from HTTP requests, eliminating 520 timeout errors. Processing happens server-side with real-time status polling.

### Daily Workflow Automation (NEW - October 2025)

**IMPORTANT:** See **[DAILY_WORKFLOW.md](DAILY_WORKFLOW.md)** for complete documentation.

**Beta User Email System** - Automated daily email delivery to beta users with admin review queue:

```bash
# Cron job functions (run via: python app.py <function>)
python app.py cleanup   # 6:00 AM - Delete old queue entries
python app.py commit    # 6:30 AM - Daily GitHub CSV commit (triggers deployment)
python app.py process   # 7:00 AM - Process all active beta users
python app.py send      # 8:30 AM - Auto-send emails to users
python app.py export    # 11:59 PM - Backup beta users to CSV
```

**Key Features:**
- Reads `beta_users` table (status='active')
- Deduplicates tickers across users
- Processes 4 concurrent tickers (recommended - ingest→digest→email generation)
- Queues emails for admin review at `/admin/queue`
- Auto-sends at 8:30 AM (or manual send anytime)
- Unique unsubscribe tokens per recipient
- DRY_RUN mode for safe testing

**Admin Dashboard:**
- `/admin` - Stats overview and navigation (4 cards: Users, Queue, Settings, Test)
- `/admin/users` - Beta user approval interface with bulk selection (Oct 2025)
- `/admin/queue` - Email queue management with 8 smart buttons and real-time counts (Oct 2025)
- `/admin/settings` - System configuration: Lookback window + GitHub CSV backup (Oct 2025)
- `/admin/test` - Web-based test runner (replaces PowerShell setup_job_queue.ps1) (Oct 2025)

**Safety Systems:**
- Startup recovery (requeues jobs stuck >3min at startup)
- **Job queue reclaim thread** (NEW - Oct 2025): Continuous monitoring, requeues jobs with stale heartbeat >3min
- Heartbeat monitoring (updates on every progress change via `last_updated` field)
- Email watchdog thread (marks email queue jobs failed after 3min stale heartbeat)
- Timeout watchdog thread (marks jobs timeout after 45min)
- DRY_RUN mode (redirects all emails to admin for testing)

**CRITICAL - Dead Worker Detection (Oct 2025):**
The job queue reclaim thread prevents jobs from getting stuck forever during Render rolling deployments.
When OLD worker is killed mid-job (SIGTERM), the reclaim thread detects stale `last_updated` timestamp
and automatically requeues the job for retry. Runs every 60 seconds, 3-minute threshold.

## Project Architecture

### Core Application Structure

**StockDigest** is a financial news aggregation and analysis system built with FastAPI. The architecture consists of:

- **Single-file monolithic design**: All functionality is contained in `app.py` (~18,700 lines)
- **PostgreSQL database**: Stores articles, ticker metadata, processing state, job queue, executive summaries, and beta users
- **Job queue system**: Background worker for reliable, resumable processing (eliminates HTTP 520 errors)
- **AI-powered content analysis**: Claude API (primary) with OpenAI fallback, prompt caching enabled (v2024-10-22)
- **Multi-source content scraping**: 2-tier fallback (newspaper3k → Scrapfly) - Playwright commented out for reliability
- **3-Email QA workflow**: Automated quality assurance pipeline with triage, content review, and user-facing reports
- **Beta landing page**: Professional signup page with live ticker validation and smart Canadian ticker suggestions

### Key Components

#### Data Models and Storage
- Ticker reference data stored in PostgreSQL with CSV backup (`data/ticker_reference.csv`)
- Articles table with deduplication via URL hashing
- Metadata tracking for company information and processing state
- Executive summaries table (`executive_summaries`) - stores daily AI-generated summaries with unique constraint on (ticker, summary_date)
- **Beta users table (`beta_users`)** - stores beta user signups with name, email, 3 tickers, status, and **legal tracking**:
  - `terms_version` (v1.0) - Terms of Service version accepted
  - `terms_accepted_at` - Timestamp when Terms accepted
  - `privacy_version` (v1.0) - Privacy Policy version accepted
  - `privacy_accepted_at` - Timestamp when Privacy accepted
- **Unsubscribe tokens table (`unsubscribe_tokens`)** - NEW (Oct 2025): Token-based unsubscribe system
  - Cryptographically secure 43-char tokens (256-bit entropy)
  - Security tracking: IP address, user agent, timestamps
  - One token per user, reusable until unsubscribed
  - CASL/CAN-SPAM compliant
- **Email queue table (`email_queue`)** - NEW (Oct 2025): Daily workflow email queue
  - Stores Email #3 HTML with {{UNSUBSCRIBE_TOKEN}} placeholder
  - Recipients array (multiple users per ticker)
  - Status workflow: processing → ready → sent
  - Heartbeat monitoring and watchdog protection
  - See [DAILY_WORKFLOW.md](DAILY_WORKFLOW.md) for details

#### Content Pipeline

**NEW (Production): Server-Side Job Queue**
1. **Job Submission** (`/jobs/submit`): Submit batch of tickers for processing
2. **Background Worker**: Polls database, processes jobs sequentially with full isolation
3. **Status Polling** (`/jobs/batch/{id}`): Real-time progress monitoring
4. Each job executes: Ingest Phase → Digest Phase (3 Emails) → GitHub Commit

**Processing Timeline per Ticker:**
- 0-60%: Ingest Phase - **Async feed parsing** (5.5x faster), AI triage, Email #1 (Article Selection QA)
- 60-95%: Digest Phase - Content scraping, AI analysis, Email #2 (Content QA)
- 95-97%: Email #3 Generation - User-facing intelligence report (fetches executive summary from database)
- 97-99%: GitHub Commit - Incremental commit to data repository
- 100%: Complete

#### Async Feed Ingestion (NEW - Production)

**Performance Optimization:**
Feed ingestion now uses **grouped parallel processing** instead of sequential, reducing processing time from ~55s to ~10s per ticker (5.5x speedup).

**Feed Structure (11 feeds per ticker):**
- **2 Company feeds:** Google News + Yahoo Finance (company name)
- **3 Industry feeds:** Google News only (3 industry keywords)
- **6 Competitor feeds:** 3 competitors × 2 sources (Google + Yahoo)

**Grouped Async Strategy:**
```
┌─ Group 1: Company ─────────────┐
│ Google → Yahoo (sequential)    │ 10s
└────────────────────────────────┘
           ↓ All groups run in parallel
┌─ Group 2: Industry ────────────┐
│ Keyword 1, 2, 3 (all parallel) │ 5s
└────────────────────────────────┘
           ↓
┌─ Group 3: Competitors ─────────┐
│ Comp1: Google → Yahoo (seq)    │
│ Comp2: Google → Yahoo (seq)    │ 10s (max of 3)
│ Comp3: Google → Yahoo (seq)    │
└────────────────────────────────┘

Total: max(10s, 5s, 10s) = ~10 seconds
```

**Why Sequential Within Google/Yahoo Pairs:**
- Yahoo Finance often redirects to original sources (e.g., yahoo.com → reuters.com)
- URL deduplication uses `resolved_url` for hash generation
- Sequential processing ensures Yahoo feed sees Google's articles already in DB
- Prevents duplicate scraping/AI calls (saves 20-40s + API costs per duplicate)

**Implementation Details:**
- Uses `ThreadPoolExecutor` with `max_workers=8` (reduced from 15 on Oct 14, 2025)
  - **Reason for reduction:** Prevents database connection contention
  - With 3 concurrent tickers: 3 × 8 = 24 peak connections (30% of 80-conn pool)
  - Old config: 3 × 15 = 45 peak connections (56% of pool) - too high
- Function: `process_feeds_sequentially()` (Line 17956)
- Database connections: Thread-safe (each thread gets own connection)
- Deduplication: `ON CONFLICT (url_hash)` prevents race conditions
- Connection pool: 11 concurrent feeds << 22-97 available Postgres connections

**Safety Guarantees:**
✅ No data corruption (sequential G→Y prevents duplicates)
✅ Thread-safe database operations
✅ All error handling preserved
✅ Stats aggregation maintained
✅ Memory monitoring continues

**Legacy: Direct HTTP Processing**
1. **Feed Ingestion** (`/cron/ingest` - Line 13155): Async RSS feed parsing with grouped strategy
2. **Content Scraping**: 2-tier fallback (newspaper3k → Scrapfly)
   - **Tier 1 (Requests):** Free, fast (~70% success rate)
   - **Tier 2 (Scrapfly):** Paid ($0.002/article), reliable (~95% success rate)
   - **Playwright:** Commented out (caused hangs on problematic domains like theglobeandmail.com)
3. **AI Triage**: Dual scoring (OpenAI + Claude run in parallel), results merged for better quality
4. **Digest Generation** (`/cron/digest`): Email compilation using Jinja2 templates

#### Rate Limiting and Concurrency
- **Semaphores DISABLED (as of Oct 2025)** to prevent threading deadlock
  - **Problem:** `threading.BoundedSemaphore` blocks threads, freezing async event loops
  - **Symptom:** 3+ concurrent tickers would deadlock (threads waiting for semaphores, can't release them)
  - **Solution:** Disabled all semaphore acquisitions (APIs enforce their own rate limits)
  - **Result:** 4 concurrent tickers run smoothly, occasional 429 errors handled gracefully
- Domain-specific scraping strategies defined in `get_domain_strategy()`
- User-agent rotation and referrer spoofing

### Memory Management

The `memory_monitor.py` module provides comprehensive resource tracking including:
- Database connection monitoring
- Async task lifecycle management
- Memory snapshot comparisons
- Connection pool utilization tracking

### API Endpoints

#### Public Endpoints (No Authentication)
- `GET /`: Beta landing page with legal disclaimers (HTML)
  - Top disclaimer banner: "For informational purposes only"
  - Required Terms/Privacy checkbox before signup
  - Footer links: Terms | Privacy | Contact
- `GET /terms-of-service`: Terms of Service page (NEW Oct 2025)
  - Province of Ontario, Canada jurisdiction
  - Contact: stockdigest.research@gmail.com
  - Last Updated: October 7, 2025 (v1.0)
- `GET /privacy-policy`: Privacy Policy page (NEW Oct 2025)
  - PIPEDA compliant (Canadian privacy law)
  - GDPR/CCPA rights included
  - Last Updated: October 7, 2025 (v1.0)
- `GET /unsubscribe?token=xxx`: Token-based unsubscribe (NEW Oct 2025)
  - Validates cryptographic token
  - Idempotent (safe to click multiple times)
  - Security tracking (IP, user agent)
  - Branded success/error HTML pages
  - CASL/CAN-SPAM compliant
- `GET /api/validate-ticker`: Live ticker validation with Canadian .TO suggestions
- `POST /api/beta-signup`: Beta user signup form submission
  - Now logs terms acceptance timestamp + version
  - Generates unsubscribe token automatically

#### Job Queue Endpoints (Production)
- `POST /jobs/submit`: Submit batch of tickers for server-side processing
- `GET /jobs/batch/{batch_id}`: Get real-time status of all jobs in batch
- `GET /jobs/{job_id}`: Get detailed job status (includes stacktraces)
- `POST /jobs/circuit-breaker/reset`: Manually reset circuit breaker
- `GET /jobs/stats`: Queue statistics and worker health
- `GET /health`: Worker health check (prevents Render idle timeout)

#### Admin Endpoints (require X-Admin-Token header)
- `POST /admin/init`: Initialize ticker feeds and sync reference data
- `POST /admin/clean-feeds`: Clean old articles beyond time window
- `POST /admin/force-digest`: Generate digest emails for specific tickers
- `POST /admin/wipe-database`: Complete database reset
- `GET /admin/ticker-metadata/{ticker}`: Retrieve ticker configuration
- **`POST /admin/export-user-csv`**: Export beta users to CSV for daily processing

**System Configuration Endpoints (NEW - Oct 2025):**
- **`GET /api/get-lookback-window`**: Get current production lookback window
- **`POST /api/set-lookback-window`**: Update production lookback window (60-10080 minutes)
- **`POST /api/commit-ticker-csv`**: Manually commit ticker_reference.csv to GitHub (triggers deployment)

**Email Queue Management (Oct 2025):**
- **`POST /api/generate-user-reports`**: Generate reports for selected users (bulk processing)
- **`POST /api/generate-all-reports`**: Generate reports for all active users (= `python app.py process`)
- **`POST /api/cancel-ready-emails`**: Cancel ready emails to prevent 8:30am auto-send (tracks previous_status)
- **`POST /api/undo-cancel-ready-emails`**: Smart restore cancelled emails to previous status
- **`POST /api/cancel-in-progress-runs`**: Cancel all ticker processing jobs (stops current runs)
- **`POST /api/rerun-all-queue`**: Reprocess all tickers regardless of status (fresh emails)
- **`POST /api/retry-failed-and-cancelled`**: Retry failed and cancelled tickers only (non-ready)
- **`POST /api/send-all-ready`**: Send all ready emails immediately
- **`POST /api/clear-all-reports`**: Delete all email queue entries (= `python app.py cleanup`)

**Test Runner (Oct 2025):**
- **`GET /admin/test`**: Web-based test runner page (replaces PowerShell script)

#### Legacy Automation Endpoints (Direct HTTP)
- `POST /cron/ingest`: RSS feed processing and article discovery (⚠️ Subject to HTTP timeouts)
- `POST /cron/digest`: Email digest generation and delivery (⚠️ Subject to HTTP timeouts)

### Configuration

#### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string (required)
- `OPENAI_API_KEY`: OpenAI API access (required for AI features)
- `SCRAPFLY_API_KEY`: ScrapFly API for Google News URL resolution (required - cost: ~$86/month)
- `ADMIN_TOKEN`: Authentication for admin endpoints
- Email configuration: `SMTP_*` variables for digest delivery
- `MAX_CONCURRENT_JOBS`: Number of tickers to process simultaneously (default: 2, recommended: 4)

#### Default Processing Parameters
- Time window: 1440 minutes (24 hours) - configurable per job
- Triage batch size: 2-3 articles per AI call
- Concurrent tickers: 4 recommended (controlled via `MAX_CONCURRENT_JOBS` env var)
- Semaphores: DISABLED (Oct 2025 - prevented threading deadlock)
- Default ticker set: ["MO", "GM", "ODFL", "SO", "CVS"]

### Database Schema

Key tables managed through schema initialization:

**Content Storage:**
- `articles`: Content storage with URL deduplication
- `ticker_articles`: Links articles to tickers with categorization
- `ticker_references`: Company metadata and exchange information
- `feeds`: RSS feed sources (shareable across tickers)
- `ticker_feeds`: Many-to-many ticker-feed relationships with per-relationship categories

**Beta User Management (October 2025 - Updated):**
- `beta_users`: Beta signup data with legal compliance tracking
  - Core fields: name, email, ticker1, ticker2, ticker3, status, created_at
  - **Legal tracking (NEW):**
    - `terms_version` VARCHAR(10) DEFAULT '1.0' - Terms of Service version
    - `terms_accepted_at` TIMESTAMPTZ - When user accepted Terms
    - `privacy_version` VARCHAR(10) DEFAULT '1.0' - Privacy Policy version
    - `privacy_accepted_at` TIMESTAMPTZ - When user accepted Privacy
  - UNIQUE constraint on email
  - Status field: 'active' | 'paused' | 'cancelled'
  - Exported daily to `data/user_tickers.csv` for morning processing

**Unsubscribe System (NEW - October 2025):**
- `unsubscribe_tokens`: Token-based unsubscribe for CASL/CAN-SPAM compliance
  - `token` VARCHAR(64) UNIQUE - Cryptographically secure (43-char URL-safe, 256-bit entropy)
  - `user_email` VARCHAR(255) - Foreign key to beta_users(email)
  - `created_at` TIMESTAMPTZ - Token generation time
  - `used_at` TIMESTAMPTZ - When token was used (NULL if unused)
  - `ip_address` VARCHAR(45) - Security tracking
  - `user_agent` TEXT - Security tracking
  - One token per user, reusable until unsubscribed
  - Indexed on token, email, and used_at

**Job Queue System (NEW):**
- `ticker_processing_batches`: Batch tracking (status, job counts, config)
- `ticker_processing_jobs`: Individual ticker jobs with full audit trail
  - Includes: retry logic, timeout protection, resource tracking, error stacktraces
  - Atomic job claiming via `FOR UPDATE SKIP LOCKED` (prevents race conditions)

**AI-Generated Content:**
- `executive_summaries`: Daily AI-generated summaries (Line 939)
  - Columns: ticker, summary_date, summary_text, ai_provider, article_ids, counts, generated_at
  - UNIQUE(ticker, summary_date) - overwrites on same-day re-runs
  - Generated during Email #2, reused in Email #3

**Other:**
- `domain_names`: Formal domain name mappings (AI-generated)
- `competitor_metadata`: Ticker competitor relationships

### Content Processing Strategy

#### Article Categorization
Articles are automatically categorized into:
- **Company**: Direct company mentions
- **Sector**: Industry-related content
- **Competitor**: Competitive landscape analysis
- **Market**: Broader market context

#### Quality Scoring
Multi-tier domain quality assessment:
- Tier 1: Premium financial sources (WSJ, Bloomberg, Reuters)
- Tier 2: Established business media
- Tier 3: General news sources
- Tier 4: Lower-quality domains with content filtering

#### Article Priority Sorting
Within each category (company/industry/competitor), articles are sorted by priority:
1. FLAGGED + QUALITY domains (newest first)
2. FLAGGED only (newest first)
3. All remaining (newest first)

This sorting is applied to all 3 email reports to ensure the most important content appears first.
Function: `sort_articles_by_priority()` - Line 10278

#### Database-First Triage Selection (v3.1)
**IMPORTANT:** Triage selection uses the database as source of truth, NOT RSS feed results.

**How it works:**
1. RSS feeds run to discover and ingest NEW articles (up to 50/25/25 per category)
2. Spam filtering happens during ingestion (before database insertion)
3. Triage queries database for latest 50/25/25 articles by `published_at` (not `found_at`)
4. Articles persist in database indefinitely (no automatic cleanup)

**Benefits:**
- RSS feed gaps don't affect triage (database has complete history)
- Slow-moving tickers fill limits with older quality articles
- Fast-moving tickers (NVDA) get latest news only
- Lookback window (1 day, 7 days, 1 month) determines publication date filter

**Query Logic:** Lines 12862-12916 in `cron_ingest()`
- Filters: `WHERE ta.ticker = %s AND (a.published_at >= cutoff OR NULL)`
- NO `found_at` filter - all articles in DB are considered
- Ranks by `published_at DESC` within each category/keyword partition

#### Google News URL Resolution (v3.7 - Oct 2025) ⭐ PRODUCTION-READY
**SUCCESS RATE: 100%** (12/12 URLs resolved in production testing)

**The Problem:**
Google News RSS feeds provide redirect URLs (`news.google.com/rss/articles/...`) that cannot be scraped directly. Articles from Google News feeds would fail content extraction, resulting in empty emails and no executive summaries (~50% content loss).

**The Solution: 3-Tier Resolution System**

**Phase 1.5: Deferred Resolution (62-64% progress)**
- Runs AFTER AI triage, BEFORE content scraping
- Only resolves **flagged articles** (~12-20 per ticker)
- 80% fewer API calls vs resolving all 150+ articles
- Function: `resolve_flagged_google_news_urls()` - Line 13198

**3-Tier Fallback Chain:**
```
Tier 1: Advanced API (Google internal API) - Free, fast
  ↓ (if fails)
Tier 2: Direct HTTP redirect - Free, simple
  ↓ (if fails)
Tier 3: ScrapFly (ASP + JS rendering) - Paid, reliable ✅
  ↓ (if fails)
Fallback: Keep Google News URL (user can still click)
```

**Tier 3: ScrapFly Resolution** (Line 13137)
- Function: `resolve_google_news_url_with_scrapfly()`
- Uses ASP (Anti-Scraping Protection) to bypass Google's anti-bot
- Uses JS rendering to handle client-side redirects
- Cost: ~$0.008 per URL resolution
- Success rate: **95-100%** (production validated)
- Reference: https://scrapfly.io/blog/how-to-scrape-google-search/

**Ingestion Phase** (Lines 4706-4727):
- Yahoo URLs: Resolve immediately (works well, no rate limits)
- Google URLs: Store with `resolved_url = NULL` (deferred until Phase 1.5)
- Extract domain from title for deduplication and Email #1
- Use `get_url_hash(url, NULL, domain, title)` for dedup

**Scraping Phase Integration** (Line 4470):
```python
# Scraper uses resolved URL when available, falls back to original
url_to_scrape = article.get("resolved_url") or article.get("url")
```

**Clean Logging:**
```
[VST] ✅ [1/12] Tier 3 (ScrapFly) → americanactionforum.org
[VST] ✅ [2/12] Tier 3 (ScrapFly) → sharewise.com
...
[VST] 📊 Resolution Summary: ✅ Succeeded: 12 (100.0%)
```

**Cost Analysis:**
- Per ticker: ~12 URLs × $0.008 = **$0.096**
- Per day (30 tickers): **$2.88/day**
- Per month: **~$86.40/month**
- **ROI**: Recovered ~35% of total content that was previously lost

**Benefits:**
✅ 100% resolution success rate (production validated)
✅ 80% fewer API calls (only flagged articles)
✅ 4 concurrent tickers process smoothly
✅ Clean, readable logs
✅ Minimal duplicate articles (domain + title matching)
✅ No database schema changes required

**Timing Impact:**
- Resolution adds: ~5-10 seconds per ticker
- Total processing time: ~30 minutes (unchanged)
- Resolutions happen naturally across concurrent tickers

### Email Template System

Uses Jinja2 templating with multiple templates:
- **`email_intelligence_report.html`** - Email #3 (Premium Intelligence Report)
  - Modern gradient header with stock price card
  - Top disclaimer banner: "For informational purposes only"
  - Executive summary sections (6 visual cards)
  - Article links with ★, 🆕, PAYWALL badges
  - Comprehensive footer legal disclaimer
  - Full URLs for Terms, Privacy, Contact, Unsubscribe
- **`email_template.html`** - Email #2 (Content QA)
- Responsive design for email clients
- Toronto timezone standardization (America/Toronto)

### 3-Email Quality Assurance Workflow

StockDigest generates 3 distinct emails per ticker during the digest phase, forming a complete QA pipeline:

#### Email #1: Article Selection QA (Line 10353)
**Function:** `send_enhanced_quick_intelligence_email()`
**Subject:** `🔍 Article Selection QA: [Company Names] ([Tickers]) - [X] flagged from [Y] articles`
**Purpose:** Quick triage results to verify AI article selection quality
**Content:**
- Shows ONLY flagged articles (high relevance scores from AI triage)
- Displays dual AI scoring badges:
  - Main score (0-10): Overall relevance to ticker
  - Category score (0-10): Strength of category assignment (company/industry/competitor)
- Minimal metadata: title, publisher, timestamp
- NO full content, NO descriptions
- Sorted by priority (flagged+quality first, then flagged, then rest)
**Timing:** Sent at ~60% progress (end of ingest phase)

#### Email #2: Content QA (Line 10955)
**Function:** `fetch_digest_articles_with_enhanced_content()` + template rendering
**Subject:** `📝 Content QA: [Tickers] - [X] articles analyzed`
**Purpose:** Full content review with AI analysis for internal QA
**Content:**
- Shows ONLY flagged articles (same filtering as Email #1)
- Full article content (title, description, full text)
- AI Analysis boxes with:
  - Key topics and themes
  - Relevance explanation
  - Sentiment indicators
  - Business impact assessment
- Executive Summary section (AI-generated overview of all flagged articles)
- Sorted by priority (same algorithm as Email #1)
**Timing:** Sent at ~95% progress (end of digest phase)
**Key Behavior:** Generates and SAVES executive summary to database via `save_executive_summary()` (Line 1050)

#### Email #3: Premium Stock Intelligence Report (Line 11873)
**Function:** `send_user_intelligence_report(hours, tickers, flagged_article_ids, recipient_email)`
**Template:** `email_intelligence_report.html` (Jinja2)
**Subject:** `📊 Stock Intelligence: [Company Name] ([Ticker]) - [X] articles analyzed`
**Purpose:** Premium user-facing intelligence report with legal disclaimers

**NEW (Oct 2025): Full Jinja2 Refactor**
- Replaced 100+ lines of inline HTML with clean Jinja2 template rendering
- Helper functions: `build_executive_summary_html()`, `build_articles_html()`
- Maintainable, testable, professional code

**Content:**
- **Top disclaimer banner:** "For informational purposes only. Not investment advice."
- **Modern HTML template** with gradient header
- **Stock price card** in header showing:
  - Today's date (email sent date)
  - Last close price (from `ticker_reference` cache or yfinance/Polygon.io)
  - Daily return with "Last Close" label for clarity
- **Executive summary sections** rendered as 6 visual cards:
  1. 🔴 Major Developments (3-6 bullets)
  2. 📊 Financial/Operational Performance (2-4 bullets)
  3. ⚠️ Risk Factors (2-4 bullets)
  4. 📈 Wall Street Sentiment (1-4 bullets)
  5. ⚡ Competitive/Industry Dynamics (2-5 bullets)
  6. 📅 Upcoming Catalysts (1-3 bullets)
- **Compressed article links** at bottom (Company/Industry/Competitors)
- **Visual indicators:**
  - **Star** (★) for FLAGGED + QUALITY articles
  - **NEW badge** (🆕) for articles published <24 hours ago
  - **PAYWALL badge** (red) for paywalled domains
- **Comprehensive footer:**
  - Legal disclaimer box
  - Links: Terms of Service | Privacy Policy | Contact | Unsubscribe (all full URLs)
  - Copyright notice
- Shows ONLY flagged articles (same filtering as Email #1 and #2)
- NO AI analysis boxes, NO descriptions (clean presentation)

**Timing:** Sent at ~97% progress (after Email #2, before GitHub commit)

**Key Behavior:**
- Retrieves executive summary from `executive_summaries` table
- Parses summary via `parse_executive_summary_sections()` (Line 11733)
- **Generates unique unsubscribe token** via `get_or_create_unsubscribe_token(recipient_email)`
- Uses `resolved_url` for all article links
- Hides empty sections automatically
- Single-ticker design only (no multi-ticker support)
- **Requires `recipient_email` parameter** for proper unsubscribe functionality

**Template Variables:**
- `ticker`, `company_name`, `industry`, `current_date`
- `stock_price`, `price_change`, `price_change_color`
- `executive_summary_html` (pre-rendered HTML string)
- `articles_html` (pre-rendered HTML string)
- `total_articles`, `paywalled_count`, `lookback_days`
- `unsubscribe_url` (unique per user)

#### Flagged Article Filtering
**CRITICAL:** Email #2 and #3 show ONLY flagged articles (those with high AI relevance scores).
- Email #1: Shows all articles, highlights which are flagged (dual AI scoring badges)
- Email #2: Filters to flagged only (SQL filter at Line 10996: `AND a.id = ANY(%s)`)
- Email #3: Filters to flagged only (parameter at Line 11212: `flagged_article_ids=flagged_article_ids`)

The "Selected" count in Email #1 reflects ONLY flagged articles, not all QUALITY domain articles.

#### Executive Summary Storage
**Table:** `executive_summaries` (Line 939)
**Columns:**
- ticker (text)
- summary_date (date)
- summary_text (text)
- ai_provider (text)
- article_ids (int[])
- company_count (int)
- industry_count (int)
- competitor_count (int)
- generated_at (timestamptz)
**Constraint:** UNIQUE(ticker, summary_date) - overwrites on same-day re-runs

**Function:** `save_executive_summary()` - Line 1050
- Called during Email #2 generation
- Stores summary with metadata for reuse in Email #3
- Prevents redundant AI calls and ensures consistency

## Job Queue System (Production Architecture)

### Overview

The job queue system eliminates HTTP 520 errors by decoupling long-running ticker processing from HTTP request lifecycles. All processing happens server-side in a background worker thread, with PowerShell polling for status instead of maintaining long HTTP connections.

### Architecture

**Before (Broken):**
```
PowerShell → HTTP → /cron/digest (30 min processing) → 520 timeout after 60-120s
```

**After (Production):**
```
PowerShell → /jobs/submit (<1s) → Instant response with batch_id
Background Worker → Process jobs → Update database
PowerShell → /jobs/batch/{id} (<1s) → Real-time status (poll every 20s)
```

### Key Components

**1. Background Worker Thread**
- Polls database every 10 seconds for queued jobs
- Processes jobs sequentially using `TICKER_PROCESSING_LOCK` (ensures ticker isolation)
- Updates progress in real-time (phase, progress %, memory, duration)
- Survives server restarts (state persists in PostgreSQL)

**2. Circuit Breaker**
- Detects 3+ consecutive **system failures** (DB crashes, memory exhaustion)
- Automatically halts processing when open (state: closed | open)
- Auto-closes after 5 minutes
- Does NOT trigger on individual ticker failures
- Manual reset: `POST /jobs/circuit-breaker/reset`

**3. Timeout Watchdog**
- Separate thread that monitors jobs every 60 seconds
- Marks jobs exceeding 45 minutes as timeout
- Updates batch counters automatically

**4. Retry Logic**
- Jobs can retry up to 2 times on transient failures
- Retry count tracked per job

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

### Usage Example

```powershell
# Submit batch
.\scripts\setup_job_queue.ps1

# Output:
# Progress: 75% | Completed: 3/4 | Failed: 0 | Current: CEG [digest_complete]
# ✅ RY.TO: completed (28.5min) [mem: 215MB]
# ✅ TD.TO: completed (30.2min) [mem: 234MB]
```

### Monitoring

```bash
# Check worker health
curl https://stockdigest.app/jobs/stats -H "X-Admin-Token: $TOKEN"

# Check batch status
curl https://stockdigest.app/jobs/batch/{batch_id} -H "X-Admin-Token: $TOKEN"

# Check specific job (includes full stacktrace)
curl https://stockdigest.app/jobs/{job_id} -H "X-Admin-Token: $TOKEN"
```

### SQL Queries

```sql
-- See all active jobs
SELECT job_id, ticker, status, phase, progress,
       EXTRACT(EPOCH FROM (NOW() - started_at))/60 as minutes_running
FROM ticker_processing_jobs WHERE status = 'processing';

-- See recent failures
SELECT ticker, error_message, created_at
FROM ticker_processing_jobs
WHERE status IN ('failed', 'timeout')
AND created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;

-- Queue depth
SELECT COUNT(*) FROM ticker_processing_jobs WHERE status = 'queued';
```

### Documentation

- `JOB_QUEUE_README.md` - Comprehensive system documentation
- `IMPLEMENTATION_SUMMARY.md` - Implementation details and testing guide

## Parallel Ticker Processing (v3.3 - Oct 2025)

### Overview

StockDigest now supports **concurrent ticker processing**, allowing 2-5 tickers to process simultaneously. This reduces total processing time when handling multiple tickers in a batch.

### Architecture Changes

**Before (Sequential):**
```
Ticker 1 → 30 min → Complete
Ticker 2 → 30 min → Complete
Ticker 3 → 30 min → Complete
Total: 90 minutes
```

**After (Parallel with MAX_CONCURRENT_JOBS=2):**
```
Ticker 1 ┐
Ticker 2 ┘ → 30 min → Complete
Ticker 3   → 30 min → Complete
Total: 60 minutes
```

### Key Components

**1. Connection Pooling (Lines 1050-1094)** ⭐ **UPDATED - Oct 14, 2025**
- Uses `psycopg_pool.ConnectionPool` for efficient connection reuse
- Configuration: `min_size=5, max_size=80` (under 100 DB limit for Basic-1GB)
- Supports up to 4-5 concurrent tickers comfortably
- Retry logic: 3 attempts with exponential backoff (2s, 5s, 10s)
- Fail-fast startup if pool cannot initialize
- **NEW: Per-Connection Timeouts** (prevents zombie transactions):
  - `idle_in_transaction_session_timeout = 300s` (5 min) - Auto-kills idle transactions
  - `lock_timeout = 30s` - Queries fail fast if can't acquire lock
  - `statement_timeout = 120s` (2 min) - Kills long-running queries
  - **Critical:** Prevents lock queue buildup that froze system for 1+ hour (Oct 14, 2025 incident)

**2. ThreadPoolExecutor Job Worker (Lines 12328-12367)**
- Concurrent job processing using Python's `ThreadPoolExecutor`
- Each ticker runs in isolated thread with own event loop (`asyncio.run()`)
- Polls database for jobs, submits to thread pool up to `MAX_CONCURRENT_JOBS`
- Uses `wait()` with `FIRST_COMPLETED` for efficient job completion handling
- Tracks active jobs: `{len(active_futures)}/{MAX_CONCURRENT_JOBS} active`

**3. Lock Removal**
- Removed `TICKER_PROCESSING_LOCK` from `/cron/ingest` and `/cron/digest`
- Tickers no longer block each other during processing
- Admin endpoints keep locks (not performance-critical)

**4. Ticker-Prefixed Logging**
- 49+ log statements updated with `[{ticker}]` prefix
- Easy filtering in Render logs: Search for `[RY.TO]` or `[SNAP]`
- Preserves `[JOB xxx]` correlation IDs
- Format: `[TICKER] 🚀 [JOB xxx] Starting processing for TICKER`

**5. Resource Monitoring**
- Connection pool stats logged: `DB Pool={active}/{max} connections`
- Memory tracking per ticker: `Memory={X}MB`
- Logged at digest completion and job completion
- Format: `[{ticker}] 📊 Resource Status: Memory=215MB, DB Pool=18/45`

**6. Thread-Local HTTP Connectors (Lines 88-147)** ⭐ **CRITICAL FIX - Oct 11, 2025**
- **Problem Solved:** "Event loop is closed" errors with 3+ concurrent tickers
- **Root Cause:** Global `aiohttp.TCPConnector` bound to first thread's event loop, became unusable when that loop closed
- **Solution:** Each `ThreadPoolExecutor` thread gets its own connector via `threading.local()`
- **Architecture:**
  - One connector per thread (isolated to thread's event loop)
  - One session per thread (uses thread's connector)
  - Automatic cleanup when job completes (`cleanup_http_session()`)
- **Scaling:**
  - `MAX_CONCURRENT_JOBS=3` → 3 connectors (300 max HTTP connections)
  - `MAX_CONCURRENT_JOBS=4` → 4 connectors (400 max HTTP connections)
  - No impact on database connections (separate system)
- **Benefits:**
  - ✅ Complete thread isolation (no cross-thread interference)
  - ✅ Scales automatically with concurrency setting
  - ✅ Fixes all "Event loop is closed" errors
  - ✅ Safe for 3-10 concurrent tickers
  - ✅ Production validated with 10-ticker runs
- **Trade-off:** Loses global connection pooling, but resources aren't the bottleneck (system runs at ~40% CPU, 60% memory with 4 concurrent)

**7. Automatic Deadlock Retry (Lines 1077-1114)** ⭐ **CRITICAL FIX - Oct 11, 2025**
- **Problem Solved:** Database deadlocks causing article loss with 3+ concurrent tickers
- **Root Cause:** Multiple threads writing to same database tables (articles, ticker_articles, domain_names) create circular lock dependencies
- **Key Insight:** Deadlocks occur even with DIFFERENT tickers (no shared data) due to shared database infrastructure (indexes, sequences, foreign keys)
- **Solution:** `@with_deadlock_retry()` decorator with unlimited retries (capped at 100 for safety)
- **Retry Strategy:**
  - Exponential backoff: 0.1s, 0.2s, 0.4s, 0.8s, capped at 1.0s max
  - PostgreSQL automatically kills one transaction to break deadlock
  - Retry succeeds immediately in 99% of cases
- **Applied to Critical Operations:**
  - `link_article_to_ticker()` - Most common deadlock point
  - `update_article_content()` - Scraping results
  - `update_ticker_article_summary()` - AI analysis
  - `save_executive_summary()` - Final summary
- **Impact:**
  - ✅ Zero article loss (previously ~90 articles lost per 10-ticker run)
  - ✅ Jobs complete successfully despite infrastructure contention
  - ✅ Minimal performance impact (~10-60 seconds total retry overhead per run)
  - ✅ Production validated with 10-ticker runs

**8. Stuck Transaction Monitor (Lines 16655-16730)** ⭐ **NEW - Oct 14, 2025**
- **Problem Solved:** Zombie transactions holding locks indefinitely (froze system Oct 14, 2025)
- **How It Works:** Watchdog thread checks every 60s for:
  - Idle-in-transaction connections >3 min
  - Queries waiting for locks >1 min
- **Logging Example:**
  ```
  ⚠️ Found 1 zombie transactions (idle >3min):
     → PID 900217: quantbrief_db_user (idle 300s) Query: SELECT COUNT...
     💡 These will be auto-killed by idle_in_transaction_session_timeout (300s)
  ```
- **Benefits:**
  - ✅ Early warning system (detect before cascade)
  - ✅ Detailed logging for debugging
  - ✅ Works with per-connection timeouts (item #1) for auto-recovery
- **Impact:** System now self-heals within 5 min max (was 1+ hour freeze)

### Environment Variables

```bash
MAX_CONCURRENT_JOBS=3  # Number of tickers to process simultaneously (default: 2, recommended: 3)
```

**Scaling:** (Updated Oct 14, 2025 - reduced from 4 to 3 after zombie transaction incident)
- Default: `2` (conservative, always safe)
- **Recommended: `3`** (optimal balance of speed and stability)
- Maximum: `4` (higher risk of lock contention, use with caution)
- Infrastructure: Standard 2GB RAM app, Basic-1GB DB (100 connections)

### Resource Requirements

**Per Ticker:** (Updated Oct 14, 2025 - max_workers reduced to 8)
- Memory: ~300MB peak (Scrapfly is lighter than Playwright)
- DB Connections: ~8-10 peak (max_workers=8, plus query overhead)
- Duration: ~25-30 minutes

**Recommended: 3 Concurrent Tickers** (UPDATED from 4)
- Memory: ~900MB (45% of 2GB - very comfortable) ✅
- DB Connections: ~24-30 peak (30-38% of 80 pool limit - very comfortable) ✅
- Duration: ~30 minutes (same as single ticker!)
- **3x faster than sequential** (90 min → 30 min for 3 tickers)
- **Why 3 not 4:** Safer margins after Oct 14, 2025 zombie transaction incident

**Maximum: 4 Concurrent Tickers** (use with caution)
- Memory: ~1200MB (60% of 2GB - acceptable) ⚠️
- DB Connections: ~32-40 peak (40-50% of 80 pool limit - acceptable) ⚠️
- Duration: ~30 minutes (no time benefit vs 3 concurrent)
- **Risk:** Higher chance of lock contention during autovacuum

### GitHub Commit Logic

**Smart Deployment Control:**
- Each ticker checks if it's the LAST job in batch
- Query: `COUNT(*) WHERE status IN ('queued', 'processing')`
- **NOT last:** `skip_render=TRUE` → `[skip render]` in commit (no deployment)
- **IS last:** `skip_render=FALSE` → Normal commit (triggers Render deployment)

**Example with 5 Tickers:**
```
RY.TO finishes → 4 remaining → [skip render]
TD.TO finishes → 3 remaining → [skip render]
VST finishes   → 2 remaining → [skip render]
CEG finishes   → 1 remaining → [skip render]
MO finishes    → 0 remaining → DEPLOYS! 🚀
```

**Result:** Only ONE deployment per batch, regardless of batch size (2, 5, or 100 tickers). **No scheduled commits needed!**

### Testing & Validation

**Phase 1: Single Ticker (Validation)**
```powershell
$TICKERS = @("RY.TO")
.\scripts\setup_job_queue.ps1
```
Verify: Connection pool initialized, worker starts with `max_concurrent_jobs: 4`

**Phase 2: 2 Parallel Tickers (Testing)**
```powershell
$TICKERS = @("RY.TO", "TD.TO")
.\scripts\setup_job_queue.ps1
```
Expected logs:
```
📤 [JOB abc] Submitted to worker pool (1/4 active)
📤 [JOB def] Submitted to worker pool (2/4 active)
[RY.TO] 🚀 Starting processing for RY.TO
[TD.TO] 🚀 Starting processing for TD.TO  ← Within SECONDS!
```

**Phase 3: 4 Parallel Tickers (Recommended Production)**
```powershell
$TICKERS = @("RY.TO", "TD.TO", "VST", "CEG")
```
Set `MAX_CONCURRENT_JOBS=4` in Render environment variables, then run script.
**Expected:** All 4 complete in ~30 minutes with comfortable resource margins.

### Expected Behavior

**Old (Sequential):**
```
19:28:50 - PLUG starts
19:33:00 - SNAP starts (4 min later) ❌
Total for 4 tickers: 120 minutes
```

**New (4 Concurrent):**
```
19:45:00 - RY.TO starts
19:45:03 - TD.TO starts (seconds later) ✅
19:45:05 - VST starts (seconds later) ✅
19:45:07 - CEG starts (seconds later) ✅
[RY.TO] 📊 Resource Status: Memory=280MB, DB Pool=18/80
[TD.TO] 📊 Resource Status: Memory=295MB, DB Pool=20/80
[VST] 📊 Resource Status: Memory=312MB, DB Pool=16/80
[CEG] 📊 Resource Status: Memory=305MB, DB Pool=19/80
Total for 4 tickers: ~30 minutes (4x faster!)
```

### Thread Safety

**Design Guarantees:**
- ✅ Each ticker job runs in isolated thread
- ✅ `asyncio.run()` creates independent event loop per thread
- ✅ Thread-local HTTP connectors (isolated per thread)
- ✅ Database connection pool is thread-safe (`psycopg_pool`)
- ✅ Automatic deadlock retry on all write operations
- ✅ Database atomic job claiming (`FOR UPDATE SKIP LOCKED`)
- ✅ No shared mutable state between ticker threads
- ✅ APIs enforce their own rate limits

### Troubleshooting

**Issue: Jobs processing sequentially instead of parallel**
- Check startup logs for: `🔧 Job worker started (max_concurrent_jobs: 4)`
- If missing or shows different number: Check Render environment variables
- Solution: Set `MAX_CONCURRENT_JOBS=4` in Render dashboard

**Issue: Connection pool errors**
- Check startup logs for: `✅ Database connection pool initialized (5-80 connections)`
- If failed: Database unavailable, check Render DB status
- Retry logic: 3 attempts before failing startup

**Issue: Database deadlocks**
- **Expected behavior:** Occasional deadlock warnings with automatic retry
- **Good:** `⚠️ Deadlock detected... retrying in 0.1s` (normal with concurrent processing)
- **Problem:** `💀 Deadlock failed after 100 retries` (extremely rare)
- **Solution:** This indicates a systemic issue, check database health

**Issue: Frequent API rate limit errors (429)**
- **Expected:** Occasional 429 errors are normal (APIs enforce their own limits)
- **Acceptable:** < 20 rate limit errors per batch
- **Problem:** > 50 rate limit errors per batch
- **Solution:** Reduce `MAX_CONCURRENT_JOBS` from `4` to `3`

**Issue: Memory exhaustion**
- Check logs for: `[{ticker}] 📊 Resource Status: Memory=XXX`
- If >1500MB total: Reduce `MAX_CONCURRENT_JOBS` to `3`
- Scrapfly is lightweight (~300MB per ticker)

### Performance Metrics

**Achieved (Oct 11, 2025):**
- ✅ **4 concurrent tickers processing smoothly** (recommended production config)
- ✅ Thread-local HTTP connectors (eliminates "Event loop is closed" errors)
- ✅ Automatic deadlock retry (zero article loss)
- ✅ Connection pool upgraded to 80 (supports up to 5 concurrent)
- ✅ Smart GitHub commits (only last ticker triggers deploy)
- ✅ Ticker-prefixed logging (easy filtering)
- ✅ **4x speedup:** 4 tickers in 30 min vs 120 min sequential
- ✅ **Production validated:** 10-ticker runs complete successfully

**Stable Production Configuration:**
- `MAX_CONCURRENT_JOBS=4`
- Memory usage: ~1200MB / 2GB (60%)
- DB connections: ~60 / 100 (60%)
- HTTP connections: ~400 (to external APIs, unlimited)
- Processing time: ~30 minutes per 4-ticker batch
- Deadlock retries: ~5-10 per ticker (auto-resolved)
- Article loss: 0 (previously ~90 per 10-ticker run)

## Development Notes

- The application uses extensive logging for debugging and monitoring
- All ticker symbols are normalized and validated against exchange patterns
- Robust error handling with fallback strategies for content extraction
- Built-in rate limiting and respect for robots.txt files
- **Job queue worker starts automatically on FastAPI startup** (see `@APP.on_event("startup")`)

## Financial Data & Ticker Validation (Oct 2025)

### Relaxed Ticker Validation

**Supported Ticker Formats:**
- ✅ **Regular US stocks:** AAPL, MSFT, TSLA
- ✅ **International stocks:** RY.TO, BP.L, SAP.DE, 005930.KS
- ✅ **Cryptocurrency:** BTC-USD, ETH-USD, SOL-USD
- ✅ **Forex pairs:** EURUSD=X, CADJPY=X, CAD=X
- ✅ **Market indices:** ^GSPC, ^DJI, ^IXIC
- ✅ **ETFs:** SPY, VOO, QQQ
- ✅ **Class shares:** BRK-A, BRK-B

**Key Functions:**
- `validate_ticker_format()` - Line 1479 (15+ regex patterns)
- `normalize_ticker_format()` - Line 1542 (preserves ^, =, -, .)

**Validation Changes:**
- Market cap no longer required (only price required)
- Supports forex and indices (which don't have market cap)
- Fallback config prevents crashes for unknown tickers

### Financial Data Fetching with Polygon.io Fallback

**Architecture (2-tier):**
```
1. yfinance (primary)
   ├─ Full data (13 fields including market cap, analysts)
   ├─ 3 retries with exponential backoff
   └─ ~48 calls/minute limit (undocumented)

2. Polygon.io (fallback - only if yfinance fails)
   ├─ Minimal data (price + daily return)
   ├─ Free tier: 5 calls/minute
   └─ Rate limited with automatic sleep
```

**Key Functions:**
- `get_stock_context()` - Line 1966 (main entry point)
- `get_stock_context_polygon()` - Line 1895 (Polygon.io fallback)
- `_wait_for_polygon_rate_limit()` - Line 1874 (rate limiter)

**Email #3 Requirements:**
- Only needs: `financial_last_price` and `financial_price_change_pct`
- Both providers supply these fields
- Header card displays: "Last Close" label for clarity

**Environment Variable:**
```bash
POLYGON_API_KEY=your_api_key_here  # Get free key at polygon.io
```

## Key Function Locations

**3-Email System:**
- `send_enhanced_quick_intelligence_email()` - Line 10353 (Email #1: Article Selection QA)
- `fetch_digest_articles_with_enhanced_content()` - Line 10955 (Email #2: Content QA)
- `send_user_intelligence_report(hours, tickers, flagged_article_ids, recipient_email)` - Line 11873 (Email #3: Premium Stock Intelligence)
  - **NEW (Oct 2025):** Jinja2 template refactor with legal disclaimers
  - **Requires:** `recipient_email` parameter for unsubscribe token generation
- `build_executive_summary_html(sections)` - Line 11785 (Helper: Render summary sections as HTML)
- `build_articles_html(articles_by_category)` - Line 11818 (Helper: Render article links as HTML)
- `parse_executive_summary_sections()` - Line 11733 (Parse AI summary into 6 sections)
- `generate_email_html_core()` - Line 12278 (Core Email #3 generation - shared by test and production)
- `save_executive_summary()` - Line 1050 (Executive summary database storage)
- `generate_openai_executive_summary()` - Line 10069 (Executive summary AI prompt)

**Unsubscribe System (NEW - Oct 2025):**
- `generate_unsubscribe_token(email)` - Line 13055 (Generate cryptographic token)
- `get_or_create_unsubscribe_token(email)` - Line 13085 (Get existing or create new token)
- `/unsubscribe` endpoint handler - Line 12981 (Token validation + unsubscribe processing)

**Job Queue System:**
- `process_digest_phase()` - Line 11626 (Main digest phase orchestrator)

**Triage & Ingestion:**
- `cron_ingest()` - Line 12730 (RSS feed processing & database-first triage)
- Database-first triage query - Lines 12862-12916 (Pulls from DB, not RSS)

**Digest & Commit:**
- `cron_digest()` - Line 13335 (Digest generation - called by job queue worker)
- `safe_incremental_commit()` - Line 14732 (GitHub commit)

## Claude API Prompt Caching (2024-10-22)

**Enabled:** October 2025
**API Version:** `2024-10-22` (upgraded from `2023-06-01`)
**Impact:** ~13% cost reduction per run (~$572/year savings for 50 tickers/day)

**How It Works:**
- System prompts marked with `cache_control: {"type": "ephemeral"}`
- First API call: Full cost
- Subsequent calls (within 5 min): 90% discount on cached portion
- Works perfectly with parallel ticker processing

**Functions Using Caching (7 total):**
1. `triage_company_articles_claude()` - ~900 tokens cached
2. `triage_industry_articles_claude()` - ~800 tokens cached
3. `triage_competitor_articles_claude()` - ~800 tokens cached
4. `generate_claude_article_summary()` - ~500 tokens cached
5. `generate_claude_competitor_article_summary()` - ~600 tokens cached
6. `generate_claude_industry_article_summary()` - ~600 tokens cached
7. `generate_claude_executive_summary()` - **~2000 tokens cached** (added Oct 2025)

**Cost Savings (50 tickers/morning):**
- Triage: $0.36/run saved
- Summaries: $1.15/run saved
- Total: ~$1.59/run (13% reduction)
- Monthly: ~$47.70 (30 runs)
- Yearly: ~$572

## Executive Summary AI Prompt (v3.2)

**Latest Update:** October 2025 - Refined for conciseness

**Reporting Philosophy Changes:**
- ❌ ~~"Cast a WIDE net - include rumors, unconfirmed reports, undisclosed deals"~~
- ❌ ~~"Better to include marginal news than miss something material"~~
- ✅ **NEW:** "Include all material developments, but keep bullets concise"
- ✅ **NEW:** "If uncertain about materiality, include it - but in ONE sentence"

**Key Characteristics:**
- ✅ Flexible bullet count ranges (3-6, 2-4, 1-4, 2-5) - no forced combining
- ✅ Enhanced guidance for competitive/industry dynamics
- ✅ Better Wall Street Sentiment formatting examples
- ✅ All explicit `{ticker}` references preserved in prompts

**Bullet Count Ranges:**
- 🔴 Major Developments: 3-6 bullets
- 📊 Financial/Operational: 2-4 bullets
- ⚠️ Risk Factors: 2-4 bullets
- 📈 Wall Street Sentiment: 1-4 bullets
- ⚡ Competitive/Industry: 2-5 bullets
- 📅 Upcoming Catalysts: 1-3 bullets

**Key Improvements:**
- AI writes to optimize clarity, not hit artificial word counts
- Multiple developments don't get combined inappropriately
- Competitive/Industry section can expand when needed (most important section)
- All explicit `{ticker}` references preserved in prompts

---

## Future Improvements & Optimizations

### Priority 1: Reliability Enhancements

#### 1. Job Phase Checkpoints (Prevent Duplicate Work on Restart)
**Current Issue:** Server restarts during processing cause jobs to restart from Phase 1, redoing completed work.

**Solution:**
```python
# Track phase completion in job config
config['completed_phases'] = ['ingest', 'resolution', 'digest']
config['emails_sent'] = ['email_1', 'email_2']

# Skip completed phases on restart
if 'digest' not in completed_phases:
    await process_digest_phase(...)
    mark_phase_complete('digest')

# Don't resend emails
if 'email_1' not in emails_sent:
    send_email_1()
    mark_email_sent('email_1')
```

**Benefits:**
- No duplicate emails during rolling deployments
- Faster recovery (skip completed work)
- Better idempotency
- Reuse scraped content and AI summaries

**Effort:** Medium (4-6 hours)

**Priority:** LOW (server restarts are rare, current recovery works)

---

### Priority 2: Performance Optimizations

#### 2. Batch ScrapFly Resolution Calls
**Current:** 1 API call per URL (~12 calls/ticker)
**Better:** Batch 5-10 URLs per call

**Savings:**
- 50-80% fewer ScrapFly API calls
- ~$40-70/month cost reduction (if supported by ScrapFly)

**Effort:** Medium (depends on ScrapFly API support)

#### 3. Cache Resolved URLs (24-hour TTL)
**Issue:** Same Google News URL might appear multiple times across feeds.

**Solution:**
```python
# Redis or database cache
SELECT resolved_url FROM url_cache
WHERE google_url = %s AND created_at > NOW() - INTERVAL '24 hours'
```

**Savings:**
- ~20% fewer ScrapFly calls
- ~$15-20/month cost reduction

**Effort:** Medium (3-4 hours with Redis, or use existing DB)

#### 4. Parallel Email Generation
**Current:** Email #2 and #3 generated sequentially
**Better:** Generate in parallel (independent operations)

**Savings:**
- 10-15 seconds per ticker
- 30-45 seconds for 3 concurrent tickers

**Effort:** Low (2 hours)

#### 5. Database Index Optimization
**Missing indexes that could speed up queries:**
```sql
CREATE INDEX IF NOT EXISTS idx_articles_resolved_url ON articles(resolved_url);
CREATE INDEX IF NOT EXISTS idx_ticker_articles_flagged ON ticker_articles(ticker, flagged) WHERE flagged = TRUE;
CREATE INDEX IF NOT EXISTS idx_domain_names_formal_name_lower ON domain_names(LOWER(formal_name));
```

**Benefits:**
- Faster deduplication queries
- Faster triage queries
- Faster domain lookups

**Effort:** Very Low (15 minutes)

---

### Priority 3: Code Organization & Maintainability

#### 7. Modularize 18,700-Line Monolith
**Current:** Single `app.py` file (hard to navigate, test, debug)

**Suggested Structure:**
```
app/
├── main.py              # FastAPI app, routes
├── ingestion.py         # RSS parsing, feed processing
├── resolution.py        # Google News URL resolution
├── scraping.py          # Content extraction
├── triage.py            # AI article scoring
├── emails.py            # Email generation (3 types)
├── jobs.py              # Job queue worker
├── database.py          # DB models, queries
└── utils.py             # Shared utilities
```

**Benefits:**
- Easier to navigate and debug
- Better testability (unit tests per module)
- Faster onboarding for new developers
- Reduced merge conflicts

**Effort:** High (2-3 days, but worth it)

#### 8. Centralize Email Sending Logic
**Current:** Multiple paths for email sending (test vs daily, Email #2 vs #3 confusion)

**Better:**
```python
class EmailOrchestrator:
    def send_email_1(self, ticker, articles):
        """Article Selection QA"""

    def send_email_2(self, ticker, articles):
        """Content QA"""

    def send_email_3(self, ticker, summary, mode='test'):
        """Premium Intelligence (test or queue)"""
```

**Benefits:**
- Clear email flow
- No duplicate email bugs
- Easier to modify email logic

**Effort:** Medium (4-5 hours)

---

### Priority 4: Data Quality Improvements

#### 9. Eliminate Fallback Domain Pollution
**Current:** AI fails → creates fake domain "nug.com" → corrects later

**Better:**
```python
# Store NULL domain with flag
domain = domain_resolver._resolve_publication_to_domain(source)
if not domain:
    INSERT articles (domain, needs_resolution) VALUES (NULL, TRUE)
```

**Benefits:**
- Cleaner database (no fake domains)
- Easier to identify unresolved publications
- Better for analytics

**Effort:** Low (2 hours)

#### 10. Smart Duplicate Detection During Ingestion
**Current:** Only catches exact URL duplicates
**Better:** Use fuzzy title matching + domain for near-duplicates

**Example:**
```python
# Catch near-duplicates
"Tesla Stock Rises 5%" vs "Tesla Stock Rises 5 Percent" → Same article
```

**Benefits:**
- Fewer duplicates in Email #1
- Less AI analysis waste
- Cleaner triage results

**Effort:** Medium (3-4 hours)

---

### Known Limitations (Design Trade-offs)

1. **Job Restart Recovery**
   - **Current:** Jobs restart from Phase 1 on server restart, redoing completed work
   - **Impact:** Duplicate emails (rare), wasted API credits, ~5-10 min delay
   - **Mitigation:** Server restarts are rare (deployments + occasional crashes)
   - **Fix Priority:** LOW (works acceptably, see Priority 1, Item 1 for improvement)

2. **Single-File Monolith**
   - **Current:** All functionality in one 18,700-line app.py file
   - **Impact:** Harder to navigate, test, and debug
   - **Mitigation:** Comprehensive logging, clear function naming
   - **Fix Priority:** MEDIUM (works well, but refactor would improve maintainability)

---

### Performance Benchmarks (Current)

**Per Ticker (4 concurrent):**
- Ingestion: ~10 seconds (5.5x faster with async feeds)
- Resolution: ~5-10 seconds (ScrapFly for ~12 URLs)
- Scraping: ~2-3 minutes (2-tier fallback)
- AI Analysis: ~1-2 minutes (triage + summaries)
- Email Generation: ~5 seconds
- **Total: ~25-30 minutes per ticker**

**API Costs (30 tickers/day):**
- ScrapFly Resolution: ~$86/month (100% success rate)
- Claude API: ~$350/month (with prompt caching: ~$300/month)
- OpenAI API: ~$150/month
- **Total: ~$536/month** (~$572 savings from caching)

**Potential Savings with Optimizations:**
- Batch ScrapFly: -$40-70/month
- URL Caching: -$15-20/month
- **Total Potential: ~$100/month savings** (18% reduction)