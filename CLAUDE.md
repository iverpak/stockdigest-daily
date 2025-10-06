# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

**OLD (Legacy): Direct HTTP Calls**
```powershell
# Direct HTTP orchestration (subject to 520 errors)
.\scripts\setup.ps1
```

The new job queue system decouples long-running processing from HTTP requests, eliminating 520 timeout errors. Processing happens server-side with real-time status polling.

## Project Architecture

### Core Application Structure

**QuantBrief** is a financial news aggregation and analysis system built with FastAPI. The architecture consists of:

- **Single-file monolithic design**: All functionality is contained in `app.py` (~14,877 lines)
- **PostgreSQL database**: Stores articles, ticker metadata, processing state, job queue, and executive summaries
- **Job queue system**: Background worker for reliable, resumable processing (eliminates HTTP 520 errors)
- **AI-powered content analysis**: Uses OpenAI API for article summarization and relevance scoring
- **Multi-source content scraping**: Supports various scraping strategies including Playwright for anti-bot domains
- **3-Email QA workflow**: Automated quality assurance pipeline with triage, content review, and user-facing reports

### Key Components

#### Data Models and Storage
- Ticker reference data stored in PostgreSQL with CSV backup (`data/ticker_reference.csv`)
- Articles table with deduplication via URL hashing
- Metadata tracking for company information and processing state
- Executive summaries table (`executive_summaries`) - stores daily AI-generated summaries with unique constraint on (ticker, summary_date)

#### Content Pipeline

**NEW (Production): Server-Side Job Queue**
1. **Job Submission** (`/jobs/submit`): Submit batch of tickers for processing
2. **Background Worker**: Polls database, processes jobs sequentially with full isolation
3. **Status Polling** (`/jobs/batch/{id}`): Real-time progress monitoring
4. Each job executes: Ingest Phase → Digest Phase (3 Emails) → GitHub Commit

**Processing Timeline per Ticker:**
- 0-60%: Ingest Phase - RSS feed parsing, AI triage, Email #1 (Article Selection QA)
- 60-95%: Digest Phase - Content scraping, AI analysis, Email #2 (Content QA)
- 95-97%: Email #3 Generation - User-facing intelligence report (fetches executive summary from database)
- 97-99%: GitHub Commit - Incremental commit to data repository
- 100%: Complete

**Legacy: Direct HTTP Processing**
1. **Feed Ingestion** (`/cron/ingest` - Line 12695): RSS feed parsing and article discovery
2. **Content Scraping**: Multi-strategy approach with fallbacks (newspaper3k → Playwright → ScrapingBee → ScrapFly)
3. **AI Triage**: OpenAI-powered relevance scoring and categorization
4. **Digest Generation** (`/cron/digest` - Line 13303): Email compilation using Jinja2 templates

#### Anti-Bot and Rate Limiting
- Domain-specific scraping strategies defined in `get_domain_strategy()`
- Playwright-based browser automation for challenging sites
- Built-in semaphore controls for concurrent processing
- User-agent rotation and referrer spoofing

### Memory Management

The `memory_monitor.py` module provides comprehensive resource tracking including:
- Database connection monitoring
- Playwright browser instance cleanup
- Async task lifecycle management
- Memory snapshot comparisons

### API Endpoints

#### Job Queue Endpoints (NEW - Production)
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

#### Legacy Automation Endpoints (Direct HTTP)
- `POST /cron/ingest`: RSS feed processing and article discovery (⚠️ Subject to HTTP timeouts)
- `POST /cron/digest`: Email digest generation and delivery (⚠️ Subject to HTTP timeouts)

### Configuration

#### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string (required)
- `OPENAI_API_KEY`: OpenAI API access (required for AI features)
- `ADMIN_TOKEN`: Authentication for admin endpoints
- Email configuration: `SMTP_*` variables for digest delivery

#### Default Processing Parameters
- Time window: 1440 minutes (24 hours)
- Triage batch size: 2-3 articles per AI call
- Scraping concurrency: Controlled via semaphores
- Default ticker set: ["MO", "GM", "ODFL", "SO", "CVS"]

### Database Schema

Key tables managed through schema initialization:

**Content Storage:**
- `articles`: Content storage with URL deduplication
- `ticker_articles`: Links articles to tickers with categorization
- `ticker_references`: Company metadata and exchange information
- `feeds`: RSS feed sources (shareable across tickers)
- `ticker_feeds`: Many-to-many ticker-feed relationships with per-relationship categories

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

### Email Template System

Uses Jinja2 templating (`email_template.html`) with:
- Responsive design for email clients
- Categorized article sections
- Publisher attribution and timestamps
- Toronto timezone standardization (America/Toronto)

### 3-Email Quality Assurance Workflow

QuantBrief generates 3 distinct emails per ticker during the digest phase, forming a complete QA pipeline:

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

#### Email #3: Premium Stock Intelligence Report (Line 11233)
**Function:** `send_user_intelligence_report()`
**Subject:** `📊 Stock Intelligence: [Company Name] ([Ticker]) - [X] articles`
**Purpose:** Premium user-facing intelligence report with modern HTML design
**Content:**
- **Modern HTML template** with gradient header and inline styles
- **Stock price card** in header (price, % change, date from `ticker_reference` cache)
- **Executive summary sections** parsed into 6 visual cards:
  1. 🔴 Major Developments (3-6 bullets)
  2. 📊 Financial/Operational Performance (2-4 bullets)
  3. ⚠️ Risk Factors (2-4 bullets)
  4. 📈 Wall Street Sentiment (1-4 bullets)
  5. ⚡ Competitive/Industry Dynamics (2-5 bullets)
  6. 📅 Upcoming Catalysts (1-3 bullets)
- **Compressed article links** at bottom (Company/Industry/Competitors)
- **Star indicators** (★) for FLAGGED + QUALITY articles only
- Shows ONLY flagged articles (same filtering as Email #1 and #2)
- NO AI analysis boxes, NO descriptions (clean presentation)
**Timing:** Sent at ~97% progress (after Email #2, before GitHub commit)
**Key Behavior:**
- Retrieves executive summary from `executive_summaries` table
- Parses text by emoji headers via `parse_executive_summary_sections()` (Line 11186)
- Uses `resolved_url` for all article links
- Hides empty sections automatically
- Single-ticker design only (no multi-ticker support)

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
curl https://quantbrief-daily.onrender.com/jobs/stats -H "X-Admin-Token: $TOKEN"

# Check batch status
curl https://quantbrief-daily.onrender.com/jobs/batch/{batch_id} -H "X-Admin-Token: $TOKEN"

# Check specific job (includes full stacktrace)
curl https://quantbrief-daily.onrender.com/jobs/{job_id} -H "X-Admin-Token: $TOKEN"
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

## Development Notes

- The application uses extensive logging for debugging and monitoring
- All ticker symbols are normalized and validated against exchange patterns
- Robust error handling with fallback strategies for content extraction
- Built-in rate limiting and respect for robots.txt files
- **Job queue worker starts automatically on FastAPI startup** (see `@APP.on_event("startup")`)

## Key Function Locations

**3-Email System:**
- `send_enhanced_quick_intelligence_email()` - Line 10353 (Email #1: Article Selection QA)
- `fetch_digest_articles_with_enhanced_content()` - Line 10955 (Email #2: Content QA)
- `send_user_intelligence_report()` - Line 11233 (Email #3: Premium Stock Intelligence)
- `parse_executive_summary_sections()` - Line 11186 (Parse AI summary into 6 sections)
- `sort_articles_by_priority()` - Line 10278 (Article priority sorting)
- `save_executive_summary()` - Line 1050 (Executive summary database storage)
- `generate_openai_executive_summary()` - Line 10069 (Executive summary AI prompt)

**Job Queue System:**
- `process_digest_phase()` - Line 11626 (Main digest phase orchestrator)

**Triage & Ingestion:**
- `cron_ingest()` - Line 12730 (RSS feed processing & database-first triage)
- Database-first triage query - Lines 12862-12916 (Pulls from DB, not RSS)

**Legacy Endpoints:**
- `cron_digest()` - Line 13335 (Digest generation)
- `safe_incremental_commit()` - Line 14732 (GitHub commit)

## Executive Summary AI Prompt (v3.1)

The executive summary prompt has been optimized for flexibility and quality:

**Removed:**
- ❌ Word count targets (100-150w, 80-120w, etc.)
- ❌ Prescriptive bullet limits that force combining unrelated facts

**Added:**
- ✅ Flexible bullet count ranges (3-6, 2-4, 1-4, 2-5)
- ✅ "Cast a WIDE net" philosophy - include rumors, undisclosed deals
- ✅ Enhanced guidance for competitive/industry dynamics
- ✅ Better Wall Street Sentiment formatting examples

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