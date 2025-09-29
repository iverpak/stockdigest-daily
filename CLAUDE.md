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

- **Single-file monolithic design**: All functionality is contained in `app.py` (~13,000+ lines)
- **PostgreSQL database**: Stores articles, ticker metadata, processing state, and job queue
- **Job queue system**: Background worker for reliable, resumable processing (eliminates HTTP 520 errors)
- **AI-powered content analysis**: Uses OpenAI API for article summarization and relevance scoring
- **Multi-source content scraping**: Supports various scraping strategies including Playwright for anti-bot domains

### Key Components

#### Data Models and Storage
- Ticker reference data stored in PostgreSQL with CSV backup (`data/ticker_reference.csv`)
- Articles table with deduplication via URL hashing
- Metadata tracking for company information and processing state

#### Content Pipeline

**NEW (Production): Server-Side Job Queue**
1. **Job Submission** (`/jobs/submit`): Submit batch of tickers for processing
2. **Background Worker**: Polls database, processes jobs sequentially with full isolation
3. **Status Polling** (`/jobs/batch/{id}`): Real-time progress monitoring
4. Each job executes: Ingest → Digest → GitHub Commit

**Legacy: Direct HTTP Processing**
1. **Feed Ingestion** (`/cron/ingest`): RSS feed parsing and article discovery
2. **Content Scraping**: Multi-strategy approach with fallbacks (newspaper3k → Playwright → ScrapingBee → ScrapFly)
3. **AI Triage**: OpenAI-powered relevance scoring and categorization
4. **Digest Generation** (`/cron/digest`): Email compilation using Jinja2 templates

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

### Email Template System

Uses Jinja2 templating (`email_template.html`) with:
- Responsive design for email clients
- Categorized article sections
- Publisher attribution and timestamps
- Toronto timezone standardization (America/Toronto)

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
4. Phase 1: Ingest (RSS, AI triage) → Update progress: 60%
5. Phase 2: Digest (scrape, AI analysis, emails) → Update progress: 95%
6. Phase 3: GitHub commit → Update progress: 99%
7. Mark complete → Release lock → Poll for next job
```

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