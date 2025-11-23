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
python app.py cleanup         # 6:00 AM EST - Delete old queue entries
python app.py check_filings   # 6:30 AM + Hourly (8:30 AM - 9:30 PM EST) - Check for new filings
python app.py commit          # 6:30 AM EST - Daily GitHub CSV commit (triggers deployment)
python app.py process         # 7:00 AM EST - Process all active beta users
python app.py send            # 8:30 AM EST - Auto-send emails to users
python app.py export          # 11:59 PM EST - Backup beta users to CSV (dual-file strategy with legal audit trail)
python app.py alerts          # Hourly (9 AM - 10 PM EST) - Real-time article alerts
```

**Render Cron Schedule (EST - UTC-5):**
```
Name: Cleanup               | Schedule: 0 11 * * *          | Command: python app.py cleanup
Name: Ticker CSV Commit     | Schedule: 30 11 * * *         | Command: python app.py commit
Name: Check Filings         | Schedule: 30 11,13-2 * * *    | Command: python app.py check_filings
Name: Daily Workflow        | Schedule: 0 12 * * *          | Command: python app.py process
Name: Auto-Send Emails      | Schedule: 30 13 * * *         | Command: python app.py send
Name: Hourly Alerts         | Schedule: 0 14-3 * * *        | Command: python app.py alerts
Name: Export Users          | Schedule: 59 4 * * *          | Command: python app.py export
```

**Note:** Cron times are updated manually for EST (UTC-5) and EDT (UTC-4) during daylight saving transitions.

**Key Features:**
- Reads `beta_users` table (status='active')
- Deduplicates tickers across users
- Processes 4 concurrent tickers (recommended - ingest→digest→email generation)
- Queues emails for admin review at `/admin/queue`
- Auto-sends at 8:30 AM (or manual send anytime)
- Unique unsubscribe tokens per recipient
- DRY_RUN mode for safe testing

**Daily vs Weekly Reports (NEW - November 2025):**

The system automatically generates different report types based on day of week:

**Schedule Logic:**
- **Monday:** Weekly reports (7-day lookback, full 6-section summary)
- **Tuesday-Sunday:** Daily reports (1-day lookback, abbreviated 2-section summary)

**Report Type Differences:**

| Feature | Daily (Tue-Sun) | Weekly (Mon) |
|---------|-----------------|--------------|
| **Lookback Window** | 1440 minutes (1 day) | 10080 minutes (7 days) |
| **Sections Shown** | 2 sections | 6 sections |
| **Bottom Line** | ✅ Shown | ✅ Shown |
| **Financial/Operational** | ✅ Shown | ✅ Shown |
| **Risk Factors** | ❌ Hidden | ✅ Shown |
| **Wall Street Sentiment** | ❌ Hidden | ✅ Shown |
| **Upside Scenario** | ❌ Hidden | ✅ Shown |
| **Downside Scenario** | ❌ Hidden | ✅ Shown |
| **Competitive/Industry** | ❌ Hidden | ✅ Shown |
| **Key Variables** | ❌ Hidden | ✅ Shown |
| **Upcoming Catalysts** | ❌ Hidden | ✅ Shown |

**Configuration:**
- Lookback windows stored in `system_config` table:
  - `daily_lookback_minutes`: 1440 (configurable via `/admin/settings`)
  - `weekly_lookback_minutes`: 10080 (configurable via `/admin/settings`)
- Report type auto-detected using Toronto timezone (America/Toronto)
- PowerShell scripts support explicit `report_type` override for testing

**Key Functions:**
- `get_report_type_and_lookback()` - Day-of-week detection (returns 'daily' or 'weekly' + minutes)
- `generate_email_html_core()` - Email generation with section filtering based on report_type
- All bulk processing endpoints propagate `report_type` through job configs

**Benefits:**
- ✅ **Daily:** Fast-moving news recap (reduces email fatigue)
- ✅ **Weekly:** Comprehensive analysis with strategic context
- ✅ **Automated:** Zero manual switching required
- ✅ **Database-backed:** Admins can adjust lookback windows without code changes

**Admin Dashboard:**
- `/admin` - Stats overview and navigation (4 cards: Users, Queue, Settings, Test)
- `/admin/users` - Beta user approval interface with bulk selection (Oct 2025)
- `/admin/queue` - Email queue management with 8 smart buttons + ♻️ Regenerate Email #3 button + 🔍 Quality Review button per ticker (Nov 2025)
- `/admin/settings` - System configuration: Lookback window + GitHub CSV backup (Oct 2025)
- `/admin/test` - Web-based test runner (replaces PowerShell setup_job_queue.ps1) (Oct 2025)
- `/admin/research` - **NEW (Oct 18, 2025):** Research tools for company profiles, transcripts, and press releases with modal viewer and bulk management

### Quality Review System (NEW - November 2025)

**Purpose:** AI-powered quality assurance to detect hallucinations, fabrications, and errors in executive summaries before sending to users.

**Two-Phase Architecture:**

**Phase 1: Article Verification**
- Verifies every sentence in executive summary is supported by article summaries
- Uses Gemini 2.5 Flash with comprehensive verification rules
- Detects 6 error types with severity levels

**Phase 2: Filing Context Verification**
- Verifies scenario contexts (Bottom Line, Upside, Downside) against 10-K/10-Q/Transcript
- Checks attribution accuracy ("per 10-K" must match 10-K content)
- Validates enrichment metadata (impact, sentiment, entity tags)

**Error Types Detected:**

🔴 **Critical Errors (Target: 0%):**
1. **Fabricated Number** - Specific numbers not found in any source article
2. **Fabricated Claim** - Events or facts not mentioned in sources
3. **Attribution Errors:**
   - WRONG: Incorrect source attribution
   - VAGUE: Too generic to verify ("per analyst" without firm name)
   - SPLIT: Multiple sources combined with single attribution
4. **Directional Error** - Opposite direction (e.g., "beat" vs "miss")

🟠 **Serious Errors (Target: <1%):**
5. **Company Confusion** - Competitor/supplier facts attributed to target company

🟡 **Minor Errors (Target: <5%):**
6. **Inference as Fact** - Conclusions stated without attribution

**API Endpoints:**
- **`POST /api/review-quality`** - Phase 1 only (article verification)
  - Parameters: `ticker`, `token`, `date` (optional, defaults to today with 12pm cutoff)
  - Processing: 5-10 seconds
  - Returns: Verification results + sends email report

- **`POST /api/review-all-quality`** - Phase 1 + Phase 2 (comprehensive review)
  - Same parameters as Phase 1
  - Processing: 10-15 seconds (includes filing context verification)
  - Email subject: "🔍 Quality Review: {ticker} - ✅ PASS" or "❌ FAIL (X critical errors)"

**Email Report Format:**
- **Summary Card:** Total sentences, supported/unsupported counts, critical error count
- **Phase 1 Results:** Sentence-by-sentence verification with status badges (✅ SUPPORTED | ⚠️ INFERENCE | 🔴 UNSUPPORTED)
- **Phase 2 Results:** Context verification for scenarios + enrichment metadata validation
- **Error Details:** Grouped by type with specific evidence and recommendations
- **Pass/Fail Status:** Overall assessment based on critical error count

**Key Functions:**
- `review_quality_phase1()` - Phase 1 verification logic (modules/quality_review.py)
- `review_quality_phase2()` - Phase 2 filing context verification (modules/quality_review_phase2.py)
- `generate_quality_review_email_html()` - Email template generation (modules/quality_review.py)

**Integration:**
- Button in `/admin/queue` for each ticker
- Runs after Email #3 generation (manual trigger)
- Always targets latest summary from database (no time-of-day dependency)
- Fetches executive summary from database using `summary_date` + `ticker`

**Benefits:**
✅ Catches hallucinations before user emails are sent
✅ Detects subtle errors (directional mistakes, vague attributions)
✅ Validates Phase 2 filing context against actual 10-K/10-Q content
✅ Detailed email report for rapid error correction
✅ <10-15 second processing time (fast feedback loop)

**Safety Systems (UPDATED - Nov 2025):**
- Startup recovery (requeues jobs stuck >3min at startup)
- **Two-phase timeout system**: 4-hour queue timeout, 45-min processing timeout (reset on claim)
- **Automatic retry on timeout**: Jobs retry up to 3 times before permanent failure
- **Freeze detection & recovery**: 5-min threshold, requeues jobs, exits for clean Render restart
- **Job queue reclaim thread**: Continuous monitoring, requeues jobs with stale heartbeat >3min
- Heartbeat monitoring (updates on every progress change via `last_updated` field)
- Email watchdog thread (marks email queue jobs failed after 3min stale heartbeat)
- DRY_RUN mode (redirects all emails to admin for testing)

**CRITICAL - Timeout & Freeze Recovery (Nov 2025):**
Jobs that timeout are automatically retried up to 3 times. If worker freezes (no activity for 5 min
with queued jobs), system requeues all processing jobs and exits via `os._exit(1)` for clean Render
restart. This prevents memory buildup from zombie threads and ensures all jobs complete.

## Project Architecture

### Core Application Structure

**StockDigest** is a financial news aggregation and analysis system built with FastAPI. The architecture consists of:

- **Single-file monolithic design**: All functionality is contained in `app.py` (~18,700 lines)
- **PostgreSQL database**: Stores articles, ticker metadata, processing state, job queue, executive summaries, and beta users
- **Job queue system**: Background worker for reliable, resumable processing (eliminates HTTP 520 errors)
- **AI-powered content analysis**: Claude API (primary) with OpenAI fallback, prompt caching enabled (v2023-06-01)
- **Multi-source content scraping**: 2-tier fallback (newspaper3k → Scrapfly) - Playwright commented out for reliability
- **3-Email QA workflow**: Automated quality assurance pipeline with triage, content review, and user-facing reports
- **Quality Review System**: AI-powered post-generation verification (Gemini 2.5 Flash) detects hallucinations and errors (NEW - Nov 2025)
- **Beta landing page**: Professional signup page with live ticker validation and smart Canadian ticker suggestions

### Key Components

#### Data Models and Storage
- **Schema initialization**: Automated at FastAPI startup with PostgreSQL advisory lock (prevents concurrent DDL execution)
  - Function: `ensure_schema()` at line 1411
  - Advisory lock ID: 123456 (prevents race conditions during rolling deployments)
  - Called once at startup (line 18329), removed from hot path (`admin_init()`) to eliminate lock contention
  - Idempotent DDL (`IF NOT EXISTS`) - safe to run multiple times
  - Non-blocking: If lock held by another process, returns early (fast startup)
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

**Quality Review Endpoints (NEW - November 2025):**
- **`POST /api/review-quality`**: Phase 1 quality review (article verification only)
  - Verifies executive summary against article summaries
  - Parameters: `ticker`, `token`, `date` (optional)
  - Processing: 5-10 seconds
  - Sends email report with detailed findings
- **`POST /api/review-all-quality`**: Comprehensive quality review (Phase 1 + Phase 2)
  - Phase 1: Article verification
  - Phase 2: Filing context verification (10-K/10-Q/Transcript)
  - Parameters: `ticker`, `token`, `date` (optional)
  - Processing: 10-15 seconds
  - Sends combined email report with both phases
- **12pm EST Cutoff Logic:** If `date` not specified:
  - Before 12pm EST → Reviews yesterday's summary
  - 12pm EST or later → Reviews today's summary

**Company Profiles & Research Summaries (UPDATED - Oct 19, 2025):**

StockDigest provides AI-powered research tools for analyzing SEC filings (10-K, 10-Q), earnings transcripts, and press releases.

### Admin Research Dashboard

**`GET /admin/research`** - Centralized research tools interface (NEW - Oct 18, 2025)
- **Location:** `https://stockdigest.app/admin/research?token=YOUR_ADMIN_TOKEN`
- **Features:**
  - Tabbed interface for Company Profiles, Transcripts, and Press Releases
  - Unified UI for all research workflows
  - Modal viewer for viewing profiles with markdown rendering
  - Bulk management tools (view all, delete, regenerate, email)

**SEC Filings Analysis (10-K and 10-Q):**

**MAJOR UPDATE (Oct 19, 2025):** Migrated to unified `sec_filings` table supporting 10-K, 10-Q, and investor presentations.

**Comprehensive Gemini Prompts:**
- **10-K Prompt:** 16-section comprehensive extraction (2,000-4,500 words)
  - Section 0: Filing metadata (fiscal year end, currency, accounting standard)
  - Section 3: EBITDA with approximation caveat, ASC 842 leases, ETR trends
  - Section 4: Goodwill by segment, segment EBITDA (rarely disclosed note)
  - Section 5: Complete debt schedule with covenant cushion analysis
  - Section 8: Comprehensive risk factor extraction with top 5 prioritization
  - Section 12: R&D capitalization policy, 3-year capex trends
  - Section 13: Strategic priorities + guidance (realistic framing)
  - Supports ALL industries and company sizes

- **10-Q Prompt:** 14-section quarterly extraction (2,000-5,000 words) - **READY FOR FUTURE USE**
  - YoY and YTD comparisons (QoQ noted as often unavailable)
  - Management tone analysis (confident, cautious, defensive, mixed)
  - New risks and material developments delta tracking
  - Momentum assessment (accelerating vs decelerating)
  - Share count change analysis
  - Guidance tracking (with caveat that it's rare in 10-Qs)

**Validation & Generation:**
- **`GET /api/fmp-validate-ticker?ticker=AAPL&type=profile`**: Validate ticker and fetch available 10-K filings from FMP
  - Returns: Array of `available_years` with fiscal year, filing date, and SEC.gov HTML URL
  - Uses FMP `/api/v3/sec_filings` endpoint (included in Starter plan)
  - Example response:
    ```json
    {
      "valid": true,
      "company_name": "Apple Inc.",
      "industry": "Consumer Electronics",
      "available_years": [
        {"year": 2024, "filing_date": "2024-11-01", "sec_html_url": "https://..."},
        {"year": 2023, "filing_date": "2023-11-03", "sec_html_url": "https://..."}
      ]
    }
    ```

- **`POST /api/admin/generate-company-profile`**: Generate AI company profile from 10-K filing (uses job queue)
  - **FMP Mode (recommended):** Send `sec_html_url` from validation response → Fetches HTML from SEC.gov
  - **File Upload Mode (fallback):** Send `file_content` (base64) + `file_name` → Extracts from PDF/TXT
  - Processing: 5-10 minutes (Gemini 2.5 Flash)
  - Returns: `job_id` for status polling via `/jobs/{job_id}`

**Profiles Management:**
- **`GET /api/admin/company-profiles`**: List all 10-K profiles from sec_filings table
  - Returns: Array of profiles with ticker, company_name, fiscal_year, markdown, char_count, metadata
  - Sorted by most recently generated first
  - Supports modal viewer in admin research page

- **`POST /api/admin/delete-company-profile`**: Delete 10-K profile(s)
  - Parameters: `ticker`, `token`
  - Deletes all 10-K filings for ticker (may be multiple years)
  - Returns: Success/error message

- **`POST /api/admin/regenerate-company-profile`**: Regenerate existing profile
  - Parameters: `ticker`, `token`
  - Currently: SEC.gov profiles only (file upload requires re-upload)
  - Returns: Info message with instructions

- **`POST /api/admin/email-company-profile`**: Email latest 10-K profile to admin
  - Parameters: `ticker`, `token`
  - Fetches most recent 10-K from sec_filings
  - Generates formatted email with stock price header
  - Sends to ADMIN_EMAIL

**Transcript Summaries:**

**MAJOR UPDATE (Oct 21, 2025):** Transcript prompt system redesigned for cleaner, more focused analysis:
- ❌ **Removed:** 10-K integration (no longer fetches 10-K profiles for context)
- ❌ **Removed:** Inline inference flagging (no more `(inference: explanation)` tags)
- ✅ **Kept:** Three-tier inference framework (Tier 0: Attribution, Tier 1: Sentiment tags, Tier 2: Synthesis)
- ✅ **Updated:** Section order (Operational Metrics before Major Developments)
- ✅ **Updated:** Target word count remains 3,000-6,000 words for rich transcripts
- 📊 **New sections:** Capital Allocation & Balance Sheet, Management Sentiment & Tone (expanded)
- Function: `_build_research_summary_prompt()` in app.py (line 15470)

**Section Flow (15 sections):**
1. 📌 Bottom Line | 2. 💰 Financial Results | 3. 📊 Operational Metrics | 4. 🏢 Major Developments
5. 📈 Guidance | 6. 🎯 Strategic Initiatives | 7. 💼 Management Sentiment & Tone | 8. ⚠️ Risk Factors & Headwinds
9. 🏭 Industry & Competitive Landscape | 10. 🔗 Related Entities | 11. 💡 Capital Allocation & Balance Sheet | 12. 💬 Q&A Highlights
13. 📈 Upside Scenario | 14. 📉 Downside Scenario | 15. 🔍 Key Variables to Monitor

- **`GET /api/fmp-validate-ticker?ticker=AAPL&type=transcript`**: Fetch available earnings transcripts from FMP
- **`GET /api/fmp-validate-ticker?ticker=AAPL&type=press_release`**: Fetch available press releases from FMP
- **`POST /api/admin/generate-transcript-summary`**: Generate AI summary (Claude) of transcript/press release
  - Parameters: `ticker`, `report_type` (transcript/press_release), `quarter`, `year`, `pr_date`
  - Synchronous processing (30-60 seconds)
  - Stores in `transcript_summaries` table

**Press Release Worker:**
- **Background Processing:** Press releases processed via job queue (like transcripts)
- **Worker Function:** `process_press_release_phase()` - handles `press_release_generation` jobs
- **Phase Routing:** Integrated into main job worker with phase detection
- **Processing Flow:** Fetch from FMP → Claude summary → Email to admin with `[INTERNAL]` tag → Save to DB
- **Email Subjects:**
  - Earnings: `[INTERNAL] TICKER Q3 2024 Earnings Release`
  - Other: `[INTERNAL] TICKER Press Release - Title`
- **Storage:** Saves to `company_releases` table (unified with 8-K)

**Key Functions (modules/company_profiles.py):**
- **`generate_sec_filing_profile_with_gemini()`**: NEW unified function for 10-K and 10-Q generation
  - Parameters: `ticker`, `content`, `config`, `filing_type` ('10-K' or '10-Q'), `fiscal_year`, `fiscal_quarter`, `gemini_api_key`
  - Automatically selects appropriate prompt (GEMINI_10K_PROMPT or GEMINI_10Q_PROMPT)
  - Model: gemini-2.5-flash
  - Temperature: 0.3 (consistent outputs)
  - **Max output tokens: 16000** (doubled from 8000 - supports comprehensive extraction)
  - Returns: profile_markdown + metadata (token counts, generation time)
  - Logs word count for validation

- **`generate_company_profile_with_gemini()`**: DEPRECATED (backward compatible)
  - Wrapper for `generate_sec_filing_profile_with_gemini()` with `filing_type='10-K'`
  - Maintains compatibility with existing code

- `fetch_sec_html_text(url)`: Fetch 10-K/10-Q HTML from SEC.gov and extract plain text
  - Uses proper User-Agent: "StockDigest/1.0 (stockdigest.research@gmail.com)"
  - BeautifulSoup HTML parsing with script/style removal
  - Returns cleaned plain text for AI processing

- `generate_company_profile_email()`: Create HTML email with profile preview and legal disclaimers
  - **Parameters for dynamic pricing:** `ytd_return_pct`, `ytd_return_color`, `market_status`, `return_label`
  - **Subject format:** `[INTERNAL] TICKER FY2024 10-K Report` or `[INTERNAL] TICKER Q3 2024 10-Q Report`
  - Supports 3-row price card with real-time data
  - Template: `email_research_report.html`

- `generate_transcript_email_v2()`: Create HTML email for transcripts and press releases
  - **Parameters for dynamic pricing:** `ytd_return_pct`, `ytd_return_color`, `market_status`, `return_label`
  - **Subject format:** `[INTERNAL] TICKER Q3 2024 Earnings Call Transcript`
  - Same pricing system as 10-K/10-Q emails
  - Template: `email_research_report.html`

- `get_filing_stock_data(ticker)`: **NEW (Oct 30, 2025)** - Unified stock pricing helper for ALL filing types
  - Returns: `stock_price`, `daily_return_pct`, `ytd_return_pct`, `price_change_color`, `ytd_return_color`, `market_status`, `return_label`
  - 3-tier fallback: yfinance → Polygon.io → ticker_reference cache → None
  - Shared by all 4 workers: 10-K, 10-Q, transcripts, press releases

**Comprehensive Prompts (modules/company_profiles.py):**
- **GEMINI_10K_PROMPT**: 16-section analysis (Sections 0-15)
  - Filing Metadata, Industry & Business Model, Revenue Model, Complete Financials
  - EBITDA extraction (disclosed or approximation with caveat)
  - Segment Performance, Complete Debt Schedule, Operational KPIs
  - Dependencies & Concentrations, Risk Factors (all from Item 1A)
  - Properties & Facilities, Legal Proceedings, Management & Governance
  - Capital Allocation (3-year history), Strategic Priorities & Outlook
  - Subsequent Events, Key Monitoring Variables
  - Target: 2,000-4,500 words

- **GEMINI_10Q_PROMPT**: 14-section quarterly analysis (Sections 0-13)
  - Filing Metadata, Quarterly Financial Performance, Segment Performance
  - Balance Sheet (QoQ and YoY), Debt Schedule Update, Cash Flow (QTD & YTD)
  - Operational Metrics, Guidance & Outlook, New Risks & Developments
  - Subsequent Events, Management Tone, Segment Trends, Liquidity
  - Summary of Key Changes (positive/negative developments, momentum check)
  - Target: 2,000-5,000 words

**User Workflow (20x faster than manual file upload):**
1. Navigate to `/admin/research` → Click "Company Profiles" tab
2. Enter ticker → Click "Validate Ticker"
3. FMP returns list of available 10-K years (dropdown auto-populates)
4. Select year from dropdown: "2023 (Filed: Nov 3, 2023)"
5. Click "Generate Profile (5-10 min)"
6. Backend fetches HTML from SEC.gov → Gemini generates comprehensive profile → Email sent
7. Profile saved to `sec_filings` table with UNIQUE(ticker, filing_type, fiscal_year, fiscal_quarter)
8. **NEW:** View all profiles in "View All Profiles" section below form
9. **NEW:** Click "View" to see profile in modal, "Email" to send to admin, "Delete" to remove

**8-K SEC Filings Analysis (NEW - Nov 8, 2025):**

StockDigest provides direct SEC Edgar scraping for 8-K filings (material event disclosures), bypassing FMP delays to access official SEC filings in real-time.

**Key Features:**
- ✅ Direct SEC Edgar scraping (no FMP dependencies)
- ✅ Last 10 8-Ks per ticker with quick preview (3KB header fetch)
- ✅ Smart content extraction (Exhibit 99.1 → main body fallback)
- ✅ Complete item code mapping (all 23 SEC item codes)
- ✅ CIK lookup with database caching
- ✅ Rate-limited (0.15s between requests = 6.67 req/sec, well under SEC's 10 req/sec limit)
- ✅ Gemini 2.5 Flash summary generation (800-1,500 words)
- ✅ Same email template as 10-K/10-Q (consistent branding)

**Material Event Item Codes (Most Common):**
- **Item 2.02:** Results of Operations and Financial Condition (earnings releases)
- **Item 1.01:** Entry into Material Agreement (debt deals, partnerships)
- **Item 2.01:** Completion of Acquisition or Disposition (M&A closings)
- **Item 8.01:** Other Events (catch-all for major announcements)
- **Item 5.02:** Departure/Appointment of Directors/Officers (C-suite changes)

**API Endpoints:**

**`GET /api/sec-validate-ticker?ticker=AAPL`** - Fetch last 10 8-Ks from SEC Edgar
- Returns: List of 8-Ks with quick-parsed titles and item codes
- Processing: ~1.5 seconds (10 × 0.15s rate limit delay)
- Example response:
  ```json
  {
    "valid": true,
    "company_name": "Apple Inc.",
    "cik": "0000320193",
    "available_8ks": [
      {
        "filing_date": "Jan 30, 2025",
        "accession_number": "0001193125-25-012345",
        "sec_html_url": "https://www.sec.gov/...",
        "title": "Results of Operations | Apple announces Q1 2024 results",
        "item_codes": "2.02, 9.01",
        "has_summary": false
      }
    ]
  }
  ```

**`POST /api/admin/generate-8k-summary`** - Generate AI summary (uses job queue)
- Parameters: `ticker`, `cik`, `accession_number`, `filing_date`, `filing_title`, `sec_html_url`, `item_codes`, `token`
- Processing: 5-10 minutes (Gemini 2.5 Flash)
- Returns: `job_id` for status polling

**`GET /api/admin/8k-filings?token=XXX`** - List all generated summaries
- Returns: Array of 8-K filings with ticker, date, title, items, summary text
- Sorted by filing date DESC

**`POST /api/admin/delete-8k-filing`** - Delete 8-K filing(s)
- Parameters: `ticker`, `token`
- Deletes ALL 8-K filings for ticker

**Key Functions (modules/company_profiles.py):**

**SEC Edgar Integration:**
- `get_cik_for_ticker(ticker)` - Lookup CIK from SEC API, cache in database
- `parse_sec_8k_filing_list(cik, count=10)` - Scrape last N 8-Ks from SEC Edgar
- `get_8k_html_url(documents_url)` - Parse documents index page for main HTML
- `quick_parse_8k_header(sec_html_url)` - Fast 3KB fetch for title + item codes
- `extract_8k_content(sec_html_url)` - Extract Exhibit 99.1 + main body

**AI Generation:**
- `generate_8k_summary_with_gemini(ticker, content, config, filing_date, item_codes, gemini_api_key)`
  - Simple, broad extraction prompt: "Extract ALL material information"
  - Preserves tables and charts in markdown
  - Target: 800-1,500 words depending on complexity
  - Max output tokens: 16,000 (same as 10-K)

**Job Queue Worker:**
- `process_8k_summary_phase(job)` - Handles `8k_summary_generation` jobs
- Phase routing: Integrated into main job worker
- Progress: 10% (extract) → 30% (Gemini) → 80% (save) → 95% (email) → 100%
- Email: Reuses 10-K/10-Q template with 8-K specific subject

**User Workflow:**
1. Navigate to `/admin/research` → Enter ticker → Click "Load Research Options"
2. Scroll to "8-K SEC Releases" section → Shows last 10 filings from SEC Edgar
3. Each filing displays: Date, Title (item description | parsed title), Items
4. Click "Generate Summary (5-10 min)" → Job queued
5. Receive email when complete with stock price card + markdown summary
6. View in Research Library → "8-K Filings" dropdown

**Database Schema:**
```sql
CREATE TABLE sec_8k_filings (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    company_name VARCHAR(255),
    cik VARCHAR(20),                     -- Cached CIK for future lookups
    accession_number VARCHAR(50) NOT NULL,  -- SEC unique ID
    filing_date DATE NOT NULL,
    filing_title VARCHAR(200) NOT NULL,  -- "Results of Operations | Apple announces..."
    item_codes VARCHAR(100),             -- "2.02, 9.01"
    sec_html_url TEXT NOT NULL,
    summary_text TEXT NOT NULL,          -- Gemini-generated summary
    ai_provider VARCHAR(20) NOT NULL,    -- 'gemini'
    ai_model VARCHAR(50),                -- 'gemini-2.5-flash'
    job_id VARCHAR(50),
    processing_duration_seconds INTEGER,
    monitored BOOLEAN DEFAULT FALSE,     -- Future automation support
    last_checked_at TIMESTAMPTZ,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT sec_8k_unique UNIQUE(ticker, accession_number)
);
```

**Rate Limiting & SEC.gov Compliance:**
- User-Agent: "StockDigest/1.0 (stockdigest.research@gmail.com)"
- Max 10 requests/second per IP (SEC requirement)
- Implementation: 0.15s delay between requests (conservative 6.67 req/sec)
- Respects robots.txt

**Cost Analysis:**
- Gemini API: ~$0.24 per summary (16K tokens output @ $0.015/1K)
- SEC Edgar: FREE (public data)
- Processing time: 5-10 minutes per summary

**Future Automation (Infrastructure Ready):**
- `monitored` field for tracking active tickers
- `last_checked_at` for cron monitoring
- Pattern ready: Copy FMP press release automation logic
- Planned: Hourly check for Item 2.02 (earnings) only

**Database Schema:**
- **`sec_filings`**: Unified storage for 10-K, 10-Q, and investor presentations (UPDATED Oct 19, 2025)
  - ticker, filing_type ('10-K', '10-Q', 'PRESENTATION'), fiscal_year, fiscal_quarter
  - filing_date, period_end_date, company_name, industry
  - profile_markdown (TEXT), profile_summary (TEXT), key_metrics (JSONB)
  - source_type ('fmp_sec', 'file_upload', 'gemini_multimodal'), source_file, sec_html_url
  - ai_provider, ai_model, generation_time_seconds, token_count_input, token_count_output
  - presentation_title, presentation_type, page_count, file_size_bytes (for presentations)
  - status, error_message, generated_at
  - **UNIQUE(ticker, filing_type, fiscal_year, fiscal_quarter)** - Stores ALL filings, not just latest
  - Indexes: ticker, filing_type, ticker+type, fiscal_year+quarter, status

- **`company_profiles` VIEW**: Backward compatibility (maps to sec_filings WHERE filing_type='10-K')
  - Allows existing queries to work unchanged during migration
  - Deprecated: Use sec_filings directly for new code

- **`transcript_summaries`**: Stores earnings transcript summaries
  - ticker, report_type ('transcript'), quarter, year, report_date
  - summary_text (TEXT), summary_json (JSONB), ai_provider (claude/gemini), ai_model
  - UNIQUE(ticker, report_type, quarter, year)

**Migration:**
- **Migration guide:** See `SEC_FILINGS_MIGRATION.md` for SQL migration script
- **Status:** ~10 existing 10-K profiles need migration from old company_profiles table
- **Backward compatibility:** VIEW ensures zero downtime during migration

**Cost Analysis:**
- Gemini API: FREE during experimental phase (will be ~$0.24/profile when pricing launches)
- FMP API: $0 (SEC filings included in Starter plan)
- Processing time: 5-10 minutes per profile (most time spent in Gemini generation)

**Future-Ready Features:**
- ✅ 10-Q support: Prompts ready, API endpoints TBD
- ✅ Investor presentations: Table schema ready, Gemini multimodal TBD
- ✅ Multi-material AI context: 10-K + 10-Q + presentations (commented code in app.py:14070-14120)

**System Configuration Endpoints (NEW - Oct 2025):**
- **`GET /api/get-lookback-window`**: Get current production lookback window
- **`POST /api/set-lookback-window`**: Update production lookback window (60-10080 minutes)
- **`POST /api/commit-ticker-csv`**: Manually commit ticker_reference.csv to GitHub (triggers deployment)

**Email Queue Management (Oct 2025):**
- **`POST /api/generate-user-reports`**: Generate reports for selected users (bulk processing)
- **`POST /api/generate-all-reports`**: Generate reports for all active users (= `python app.py process`)
- **`POST /api/regenerate-email`**: Regenerate Email #3 using existing articles (fixes bad summaries without reprocessing)
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
- `FMP_API_KEY`: Financial Modeling Prep API key (required - provides 10-K filings, transcripts, press releases)
- `GEMINI_API_KEY`: Google Gemini API key (required for company profiles - get from https://aistudio.google.com/app/apikey)
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
  - **Dual-file export strategy:**
    - **File 1:** `data/users/user_tickers.csv` - ACTIVE users only (5 fields) for 7 AM processing
      - Overwrites daily (stable filename)
      - Fields: name, email, ticker1, ticker2, ticker3
    - **File 2:** `data/users/beta_users_YYYYMMDD.csv` - ALL users (9 fields) for legal audit trail
      - New timestamped file daily (never deleted)
      - Fields: + status, created_at, terms_accepted_at, privacy_accepted_at
      - Automatic GitHub commits via REST API for immutable audit trail (CASL/PIPEDA compliance)
    - **Export Method:** Uses GitHub REST API (not git CLI)
      - Function: `commit_csv_to_github()` - accepts file_path parameter
      - Authentication: GITHUB_TOKEN environment variable
      - Works in Render cron environment (no git credentials needed)
      - Built-in retry logic and error handling

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

**System Configuration (NEW - November 2025):**
- `system_config`: Key-value store for runtime configuration
  - Fields: key (UNIQUE), value, description, updated_by, updated_at
  - **Daily/Weekly Report Settings:**
    - `daily_lookback_minutes`: 1440 (default) - Lookback window for daily reports (Tue-Sun)
    - `weekly_lookback_minutes`: 10080 (default) - Lookback window for weekly reports (Mon)
  - Configurable via `/admin/settings` interface
  - Used by `get_report_type_and_lookback()` for day-of-week detection

**AI-Generated Content:**
- `executive_summaries`: Daily AI-generated summaries (Line 939)
  - Columns: ticker, summary_date, summary_text, ai_provider, article_ids, counts, generated_at
  - UNIQUE(ticker, summary_date) - overwrites on same-day re-runs
  - Generated during Email #2, reused in Email #3

**Company Releases (NEW - November 2025):**
- `company_releases`: **Unified storage for FMP press releases AND 8-K SEC filings**
  - **Replaces deprecated:** `press_releases` and `parsed_press_releases` tables (dropped Nov 2025)
  - Columns: ticker, company_name, release_type, filing_date, report_title, source_id, source_type, summary_json, summary_html, summary_markdown, ai_provider, ai_model, fiscal_year, fiscal_quarter, exhibit_number, item_codes, generated_at
  - **Source types:**
    - `'fmp_press_release'` - FMP API press releases (source_id = NULL)
    - `'8k_exhibit'` - SEC 8-K exhibits (source_id = sec_8k_filings.id)
  - **UNIQUE(ticker, filing_date, report_title)** - Handles multiple exhibits per 8-K
  - **Single AI prompt:** Uses `8k_filing_prompt` for BOTH sources (no wasteful dual processing)
  - **Helper module:** `modules/company_releases.py` provides cron job helpers:
    - `db_has_any_fmp_releases_for_ticker()` - Silent init detection
    - `db_has_any_8k_for_ticker()` - Silent init detection
    - `db_check_fmp_release_exists()` - Deduplication
    - `db_check_8k_filing_exists()` - Deduplication (by accession_number)
    - `db_get_latest_fmp_release_datetime()` - Latest FMP release
    - `db_get_latest_8k_filing_date()` - Latest 8-K filing

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

Uses Jinja2 templating and inline HTML generation:
- **`email_research_report.html`** - Research documents (10-K, 10-Q, transcripts, presentations, press releases) **(UPDATED - Oct 30, 2025)**
  - **3-row price card** in gradient header (upgraded from 2-row):
    - Row 1: Last price with dynamic label (INTRADAY or LAST CLOSE)
    - Row 2: Daily return with dynamic label (TODAY or 1D)
    - Row 3: YTD return (newly added)
  - **Real-time pricing system** for all filing types:
    - yfinance (primary, 3 retries, 10s timeout)
    - Polygon.io (fallback, 5 calls/min)
    - ticker_reference cache (last resort for 7 AM closed market)
    - Helper function: `get_filing_stock_data(ticker)` - unified pricing logic
  - **Dynamic market detection:**
    - `is_market_open()` checks current time vs market hours
    - Labels adjust automatically: "INTRADAY" + "TODAY" vs "LAST CLOSE" + "1D"
  - **Graceful degradation:** If price unavailable, entire card hidden
  - Professional layout with styled sections
  - Comprehensive footer with legal disclaimers
  - Full URLs for Terms, Privacy, Contact
  - **Unified Styling (Oct 30, 2025):** All research emails now use consistent clean styling:
    - ✅ White background (no grey wrapper)
    - ✅ Compact header sizes (template default)
    - ✅ Minimal spacing (no extra padding)
    - ✅ 10-K/10-Q now match Transcripts/Press Releases
  - **Table Formatting (10-K/10-Q - Oct 30, 2025):** Consistent financial table layout:
    - `table-layout: fixed` forces equal distribution
    - First column (labels): **40% width**
    - Remaining columns: Split **60% equally**
    - Works with 2-10+ columns automatically
    - Example: 4 columns = Labels 40% | Data 20% + 20% + 20%
    - Email client support: Gmail, Outlook.com, Apple Mail (90-95% coverage)
- **Email #3 (Premium Intelligence Report)** - Inline HTML generation (no template)
  - Generated via `generate_email_html_core()` in app.py (line 18601)
  - Modern gradient header with stock price card
  - Top disclaimer banner: "For informational purposes only"
  - Executive summary sections (6 visual cards)
  - Article links with ★, 🆕, PAYWALL badges
  - Comprehensive footer legal disclaimer
  - Full URLs for Terms, Privacy, Contact, Unsubscribe
- **`email_hourly_alert.html`** - Hourly alert emails
- Responsive design for email clients
- Toronto timezone standardization (America/Toronto)

### Email Routing & Recipients

**Admin-Only Emails (Research Documents):**
All research documents (10-K, 10-Q, transcripts, press releases, 8-K) are sent ONLY to admin with `[INTERNAL]` subject tags:
- **Recipients:** Admin only (ADMIN_EMAIL environment variable)
- **Subject Format:** `[INTERNAL] TICKER <details>`
- **Purpose:** Legal audit trail, research archive, admin review
- **Templates:** `email_research_report.html`
- **Trigger:** Automated filings check (`check_filings` cron job) or manual generation
- **Examples:**
  - `[INTERNAL] AAPL FY2024 10-K Report`
  - `[INTERNAL] MSFT Q3 2024 Earnings Call Transcript`
  - `[INTERNAL] TSLA Press Release - Q4 Production Update`

**User-Facing Emails (Daily Intelligence):**
Users receive ONLY Email #3 (Premium Stock Intelligence Report) via daily workflow:
- **Recipients:** Active beta users from `beta_users` table
- **Subject Format:** `📊 Stock Intelligence: Company Name (TICKER) - X articles analyzed`
- **Purpose:** Daily/weekly investment intelligence
- **Generation:** Inline HTML (no template file)
- **Trigger:** 7 AM daily workflow (`process` cron job)
- **Routing:** Queued in `email_queue` table → sent at 8:30 AM (`send` cron job)
- **Content:** Executive summary + flagged article links (no full text)

**Internal QA Emails:**
Email #1 and #2 are sent ONLY to admin during ticker processing:
- **Email #1:** `🔍 Article Selection QA` - Shows AI triage results
- **Email #2:** `📝 Content QA` - Shows full article content + AI analysis
- **Recipients:** Admin only
- **Trigger:** Job queue processing (ingest phase and digest phase)

**Hourly Alert Emails:**
Real-time article alerts sent to active beta users:
- **Recipients:** Active beta users
- **Subject Format:** `📰 Hourly Alerts: TICKER1, TICKER2 (X articles) - HH:MM AM/PM`
- **Template:** `email_hourly_alert.html`
- **Trigger:** Hourly cron job (9 AM - 10 PM EST)
- **Content:** Cumulative articles from midnight to current hour

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

#### Email #3: Premium Stock Intelligence Report (Line 18601)
**Function:** `generate_email_html_core(ticker, hours, flagged_article_ids, recipient_email)` (inline HTML generation)
**Template:** Inline HTML (no Jinja2 template file)
**Subject:** `📊 Stock Intelligence: [Company Name] ([Ticker]) - [X] articles analyzed`
**Purpose:** Premium user-facing intelligence report with legal disclaimers

**Architecture:**
- Core generation: `generate_email_html_core()` at line 18601
- Test wrapper: `send_user_intelligence_report()` at line 18961
- Production: Called via job queue for daily workflow
- Helper functions: `build_executive_summary_html()`, `build_articles_html()`

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
- **Supported job phases:**
  - `ingest_and_digest` - Standard ticker processing (articles, emails)
  - `company_profile_generation` - 10-K profile generation (Gemini)
  - `transcript_generation` - Transcript summary generation (Claude)
  - `press_release_generation` - Press release summary generation (Claude) **(NEW - Oct 30, 2025)**

**2. Circuit Breaker**
- Detects 3+ consecutive **system failures** (DB crashes, memory exhaustion)
- Automatically halts processing when open (state: closed | open)
- Auto-closes after 5 minutes
- Does NOT trigger on individual ticker failures
- Manual reset: `POST /jobs/circuit-breaker/reset`

**3. Two-Phase Timeout System** (UPDATED - Nov 2025)
- **Queue Timeout (4 hours)**: Jobs can wait in queue up to 4 hours before being claimed
- **Processing Timeout (45 minutes)**: Reset when job is claimed, gives 45 min to complete
- Timeout watchdog monitors every 60 seconds
- **Automatic retry on timeout**: Jobs retry up to 3 times before permanent failure
- Queue timeout handling: Jobs stuck in queue > 4 hours are marked failed

**4. Freeze Detection & Recovery** (UPDATED - Nov 2025)
- Heartbeat monitor checks every 60 seconds
- **5-minute freeze threshold**: If no worker activity for 5 min with queued jobs → frozen
- On freeze: Requeues all processing jobs, then `os._exit(1)` for clean Render restart
- Prevents memory buildup from zombie threads

**5. Retry Logic**
- Jobs retry up to **3 times** on timeout before permanent failure
- Retry count tracked per job in `retry_count` column
- Logging: `[RETRY 1/3]`, `[RETRY 2/3]`, `[RETRY 3/3]`, `[FAILED]`

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
✅ **Automatic Retries** - Up to 3 retries on timeout before permanent failure
✅ **Circuit Breaker** - Detects system failures, prevents cascading errors
✅ **Two-Phase Timeout** - 4-hour queue timeout, 45-min processing timeout (reset on claim)
✅ **Freeze Recovery** - Auto-detects frozen workers, requeues jobs, exits for clean restart

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
- **Database schema initialization** (NEW - Nov 20, 2025):
  - Automated at app startup with PostgreSQL advisory lock (prevents concurrent DDL)
  - Advisory lock ensures only ONE process runs schema init during rolling deployments
  - Removed from hot path (`admin_init()`) to eliminate lock contention during concurrent test runs
  - Idempotent DDL design (`IF NOT EXISTS`) - safe for multiple startups

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

## Hourly Alerts System (NEW - October 2025)

**Purpose:** Real-time article delivery to beta users throughout the trading day.

### Overview

Lightweight, fast article alerting system that sends cumulative emails every hour from 9 AM - 10 PM EST.

**Key Characteristics:**
- ✅ No AI triage, no scraping, no analysis = fast processing (~2-5 min/hour)
- ✅ Cumulative sending (each email shows ALL articles from midnight to current hour)
- ✅ Stores in existing `articles` and `ticker_articles` tables
- ✅ 7 AM daily workflow automatically finds these articles (zero duplicate work!)
- ✅ One email per user with all 3 tickers jumbled together
- ✅ Sorted newest to oldest across all tickers

### Architecture

**Schedule:** Hourly cron job (9 AM - 10 PM EST = 14 emails per day)

**Processing Flow:**
```
1. Check time (9 AM - 10 PM EST) → Exit if outside window
2. Load active beta users from database
3. Deduplicate tickers across all users
4. For each unique ticker:
   - Parse RSS feeds (11 feeds per ticker)
   - Resolve Google News URLs (3-tier: Advanced API → HTTP → ScrapFly)
   - Filter spam domains (exclude Tier 4 only)
   - Store in articles table (ON CONFLICT skips duplicates)
   - Link to ticker in ticker_articles (flagged=FALSE)
5. For each user:
   - Query cumulative articles (midnight to now) for their 3 tickers
   - Generate HTML email using Jinja2 template
   - Send with unique unsubscribe token
   - BCC admin
```

**Email Format:**
- Subject: `📰 Hourly Alerts: JPM, AAPL, TSLA (47 articles) - 3:00 PM`
- Template: `templates/email_hourly_alert.html`
- Content:
  - `[JPM - Company]` ticker badge before each article
  - ★ Star for quality domains (WSJ, Bloomberg, Reuters, etc.)
  - PAYWALL badge (red, inline after title)
  - Domain name • Date • Time (EST)
  - Newest to oldest, jumbled across tickers

**Cron Setup (Render):**
```
Name: Hourly Alerts
Schedule: 0 * * * *
Command: python app.py alerts
```

### Database Integration

**Articles Storage:**
- Inserts into existing `articles` table with `ON CONFLICT (url_hash) DO NOTHING`
- Links to `ticker_articles` with `flagged=FALSE` (not triaged)
- 7 AM daily workflow finds these articles automatically
- No duplicate URL resolutions = cost savings (~$0.88/day offset)

**Cost Impact:**
- Resolution: ~10 URLs/hour × 14 hours = ~140 resolutions/day × $0.008 = **$1.12/day**
- Offset: 7 AM workflow skips ~110 duplicates/day = **-$0.88/day**
- **Net cost: ~$0.24/day = $7/month**

### Functions

**Core Processing:**
- `process_hourly_alerts()` - Line 22479 (Main orchestrator, checks time window)
- `insert_article_minimal()` - Line 22760 (Minimal insertion, no scraping)
- `link_article_to_ticker_minimal()` - Line 22790 (Link with flagged=FALSE)

**CLI Handler:**
- `python app.py alerts` - Line 22837

**Template:**
- `templates/email_hourly_alert.html` - Jinja2 template with ticker badges

### Features

✅ **Cumulative Display** - Users see full day's articles in each email
✅ **No Tracking Table** - Simple timestamp queries
✅ **Zero Duplicate Work** - Articles reused by daily workflow
✅ **Fast Processing** - No AI = ~2-5 min per hourly run
✅ **Same Unsubscribe** - Reuses existing token system
✅ **Quality/Paywall Badges** - Visual indicators preserved

## Automated Filings Check System (NEW - October 30, 2025)

**Purpose:** Real-time hedge fund research platform - catch new SEC filings, earnings transcripts, and press releases within hours of publication.

### Overview

Automated monitoring system that checks for new financial documents and queues them for AI analysis. Ensures latest filings are always available for the 7 AM daily workflow.

**Key Characteristics:**
- ✅ Monitors all active user tickers from `beta_users` table
- ✅ Checks 5 document types: 10-K, 10-Q, earnings transcripts, press releases, 8-K
- ✅ Compares latest from FMP/SEC APIs vs database records
- ✅ Queues jobs for missing/new documents (uses existing job queue system)
- ✅ Generated summaries emailed to admin with `[INTERNAL]` tag
- ✅ Guarantees latest filings available for 7 AM processing

### Schedule

**Cron Setup (Render):**
```
Name: Check Filings
Schedule: 30 6,8-21 * * *  # 6:30 AM, then hourly 8:30 AM - 9:30 PM EST
Command: python app.py check_filings
```

**Timing Strategy:**
- **6:30 AM:** Pre-processing check (catches overnight filings for 7 AM workflow)
- **Hourly 8:30 AM - 9:30 PM:** Real-time monitoring during market hours

### Document Type Logic

**10-K Filings:**
- Compare: Latest fiscal year from FMP vs latest in `sec_filings` table
- Queue if: FMP has newer fiscal year
- Example: FMP shows 2024, DB has 2023 → Queue 2024

**10-Q Filings:**
- Compare: Latest fiscal year + quarter from FMP vs `sec_filings`
- Queue if: FMP has newer quarter/year combination
- Example: FMP shows Q3 2024, DB has Q2 2024 → Queue Q3 2024

**Earnings Transcripts:**
- Compare: Latest quarter + year from FMP vs `transcript_summaries`
- Queue if: FMP has newer quarter/year combination
- Filter: `report_type = 'transcript'`

**Press Releases:**
- Check: Last 4 press releases from FMP
- Queue if: `report_date` not found in `transcript_summaries`
- Deduplication: By exact date match
- Efficient: Only checks recent PRs (covers 99% of cases)

**8-K SEC Filings (UPDATED - Nov 20, 2025):**
- Check: Last 3 8-K filings from SEC Edgar
- **Enrichment**: Cron workflow now matches manual workflow (lines 26985-27009)
  - Calls `get_8k_html_url()` to extract main 8-K HTML URL from documents page
  - Calls `quick_parse_8k_header()` to extract filing title + item codes
  - Rate limiting: 0.15s delay between requests (respects SEC 10 req/sec limit)
  - Error handling: Falls back to safe defaults ('8-K Filing', 'Unknown') if enrichment fails
- Queue if: Accession number not found in `company_releases` table (checks via `db_check_8k_filing_exists()`)
- Deduplication: By accession number (unique SEC identifier)
- **Fix (Nov 20, 2025)**: Previously missing enrichment step caused `NoneType` errors in logging

### Database Helper Functions

**New Functions (app.py):**
- `db_get_latest_10k(ticker)` → Returns latest 10-K fiscal year from DB
- `db_get_latest_10q(ticker)` → Returns latest 10-Q year + quarter from DB
- `db_get_latest_transcript(ticker)` → Returns latest transcript year + quarter
- `db_check_press_release_exists(ticker, date)` → Boolean check if PR exists
- `check_filings_for_ticker(ticker)` → Main checking logic for one ticker (lines 26798-27067)
  - **8-K enrichment logic**: Lines 26985-27009
  - Imports: `get_8k_html_url`, `quick_parse_8k_header` from `modules/company_profiles.py`
- `check_all_filings_cron()` → Wrapper that processes all active user tickers

**Company Profiles Module Functions (8-K enrichment):**
- `get_8k_html_url(documents_url)` → Parses documents page for main 8-K HTML URL (line 453)
- `quick_parse_8k_header(sec_html_url, rate_limit_delay=0.15)` → Extracts title + item codes from 8-K header (line 784)

**Company Releases Module Functions:**
- `db_has_any_8k_for_ticker(ticker)` → Check if ticker has ANY 8-K filings (used for silent init)
- `db_check_8k_filing_exists(ticker, filing_date, accession_number)` → Boolean check if 8-K exists (deduplication)

### Processing Flow

```
1. Query all active user tickers from beta_users table
2. Deduplicate ticker list
3. For EACH ticker:
   a. Fetch latest 10-K from FMP → Compare with DB → Queue if newer
   b. Fetch latest 10-Q from FMP → Compare with DB → Queue if newer
   c. Fetch latest transcript from FMP → Compare with DB → Queue if newer
   d. Fetch last 4 PRs from FMP → Check each against DB → Queue if missing
   e. Fetch last 3 8-Ks from SEC Edgar → **Enrich** (title + item codes) → Compare with DB → Queue if missing
4. Queued jobs processed by existing worker:
   - 10-K/10-Q: Gemini 2.5 Flash generation (5-10 min)
   - Transcripts: Claude generation (30-60 sec)
   - Press Releases: Claude generation (30-60 sec)
   - 8-K: Claude generation (30-60 sec)
5. Summaries emailed to admin with [INTERNAL] subject tag
6. Database updated (prevents duplicate processing on next check)
```

### Deduplication & Safety

**10-K/10-Q/Transcripts:**
- Primary key: `UNIQUE(ticker, filing_type, fiscal_year, fiscal_quarter)`
- Database prevents duplicates automatically
- Safe to run cron multiple times

**FMP Press Releases:**
- Primary key: `UNIQUE(ticker, filing_date, report_title)` in `company_releases` table
- Source type: `'fmp_press_release'`
- Matches by date AND title (supports multiple PRs per day)

**Silent Initialization (ALL 5 Filing Types):**
- **Purpose:** When a ticker is first monitored, save baseline documents and send admin notification emails tagged with `[INTERNAL]`
- **Applies to:** 10-K, 10-Q, Transcripts, Press Releases, 8-K
- **Detection:** Database-state-based (not time-based)
  - `db_has_any_10k_for_ticker(ticker)` - Check if ticker has ANY 10-K
  - `db_has_any_10q_for_ticker(ticker)` - Check if ticker has ANY 10-Q
  - `db_has_any_transcript_for_ticker(ticker)` - Check if ticker has ANY transcripts
  - `db_has_any_fmp_releases_for_ticker(ticker)` - Check if ticker has ANY FMP releases
  - `db_has_any_8k_for_ticker(ticker)` - Check if ticker has ANY 8-K filings
- **Behavior:**
  - First check: Save latest 1 of each type + send `[INTERNAL]` email to admin
  - Subsequent checks: Send `[INTERNAL]` email for any NEW filings
  - Works at ANY hour, not just 6:30 AM
- **Email Subjects:**
  - 10-K: `[INTERNAL] TICKER FY2024 10-K Report`
  - 10-Q: `[INTERNAL] TICKER Q3 2024 10-Q Report`
  - Transcript: `[INTERNAL] TICKER Q3 2024 Earnings Call Transcript`
  - Press Release (Earnings): `[INTERNAL] TICKER Q3 2024 Earnings Release`
  - Press Release (Other): `[INTERNAL] TICKER Press Release - Title`
  - 8-K: `[INTERNAL] TICKER 8-K - Title`
- **Benefits:**
  - ✅ Consistent logic across all 5 document types
  - ✅ Admin receives ALL filings for legal audit trail (CASL/PIPEDA)
  - ✅ `[INTERNAL]` tag prevents confusion with user-facing emails
  - ✅ Clean baseline in research folder
  - ✅ Real-time email alerts for all filings

### Benefits

✅ **Information Edge** - Summaries generated within 1-2 hours of publication
✅ **Zero Manual Work** - Fully automated monitoring
✅ **7 AM Guarantee** - Latest filings always available for morning workflow
✅ **Scalable** - Handles 100+ tickers easily (async FMP API calls)
✅ **Efficient** - Only generates new/missing documents
✅ **Database-Safe** - UNIQUE constraints prevent duplicates

### CLI Usage

```bash
# Run manually (useful for testing)
python app.py check_filings

# Expected output:
# 🔍 Checking filings for 15 active user tickers...
# ✅ AAPL: [INTERNAL] 10-K 2024 queued for generation
# ✅ MSFT: [INTERNAL] Transcript Q3 2024 queued for generation
# ⏭️ GOOGL: All filings up to date
# 📊 Summary: 3 jobs queued across 15 tickers (admin will receive [INTERNAL] emails)
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
- `save_executive_summary()` - Line 2051 (Executive summary database storage)
- `get_latest_summary_date(ticker)` - Line 2114 (Query for most recent summary - NEW Nov 2025)
  - Used by Regenerate and Quality Review endpoints
  - Queries: `ORDER BY generated_at DESC LIMIT 1`
  - Returns: Most recent `summary_date` for ticker
  - Eliminates time-of-day dependency (no 12pm cutoff)
- `generate_openai_executive_summary()` - Line 10069 (Executive summary AI prompt)

**Unsubscribe System (NEW - Oct 2025):**
- `generate_unsubscribe_token(email)` - Line 13055 (Generate cryptographic token)
- `get_or_create_unsubscribe_token(email)` - Line 13085 (Get existing or create new token)
- `/unsubscribe` endpoint handler - Line 12981 (Token validation + unsubscribe processing)

**Regenerate Email #3 (NEW - Oct 2025, Updated Nov 2025):**
- `POST /api/regenerate-email` - Line 32131 (Backend endpoint for regenerating Email #3)
  - Queries database for latest summary date (no time-of-day dependency)
  - Fetches existing flagged articles from that summary (preserves original article IDs)
  - Regenerates executive summary using Claude/OpenAI (Phase 1 + Phase 2)
  - Updates executive_summaries table with new JSON (both summary_text and summary_json)
  - Updates email_queue with new HTML
  - Sends preview to admin (Email #2 + Email #3)

**Hourly Alerts System (NEW - Oct 2025):**
- `process_hourly_alerts()` - Line 22479 (Main orchestrator, runs 9 AM - 10 PM EST)
- `insert_article_minimal()` - Line 22760 (Lightweight article insertion)
- `link_article_to_ticker_minimal()` - Line 22790 (Link articles without AI scoring)
- `templates/email_hourly_alert.html` - Cumulative alert email template

**Quality Review System (NEW - Nov 2025):**
- `review_quality_phase1()` - modules/quality_review.py (Phase 1: Article verification)
- `review_quality_phase2()` - modules/quality_review_phase2.py (Phase 2: Filing context verification)
- `generate_quality_review_email_html()` - modules/quality_review.py (Email report generation)
- `POST /api/review-quality` - Line 32513 (Phase 1 endpoint)
- `POST /api/review-all-quality` - Line 32658 (Phase 1 + Phase 2 endpoint)
- **Key Features:**
  - Gemini 2.5 Flash verification
  - 6 error types with severity levels (critical, serious, minor)
  - Sentence-by-sentence analysis with status badges
  - Database-driven date selection (queries for latest summary)

**Daily vs Weekly Reports (NEW - Nov 2025):**
- `get_report_type_and_lookback()` - Line 2142 (Day-of-week detection and lookback window retrieval)
  - Returns: `('daily', 1440)` or `('weekly', 10080)` based on day of week
  - Uses Toronto timezone (America/Toronto) for day detection
  - Queries `system_config` table for lookback windows
- `generate_email_html_core()` - Line 18601 (Email #3 generation with section filtering)
  - Parameter: `report_type` ('daily' or 'weekly')
  - Daily reports: Hide 4 sections (upside_scenario, downside_scenario, key_variables, upcoming_catalysts)
  - Weekly reports: Show all 6 sections
- **Database Schema:**
  - `system_config.daily_lookback_minutes` - Configurable via `/admin/settings` (default: 1440)
  - `system_config.weekly_lookback_minutes` - Configurable via `/admin/settings` (default: 10080)
- **Updated Endpoints:**
  - All bulk processing endpoints propagate `report_type` through job configs
  - `/jobs/submit` supports optional `report_type` field (falls back to day-of-week detection)
  - `POST /api/generate-all-reports` - Uses day-of-week detection
  - `POST /api/retry-failed-and-cancelled` - Uses day-of-week detection
  - `POST /api/rerun-all-queue` - Uses day-of-week detection

**Job Queue System:**
- `process_digest_phase()` - Line 11626 (Main digest phase orchestrator)

**Triage & Ingestion:**
- `cron_ingest()` - Line 12730 (RSS feed processing & database-first triage)
- Database-first triage query - Lines 12862-12916 (Pulls from DB, not RSS)

**Digest & Commit:**
- `cron_digest()` - Line 13335 (Digest generation - called by job queue worker)
- `safe_incremental_commit()` - Line 14732 (GitHub commit)

## Claude API Prompt Caching (2023-06-01)

**Enabled:** October 2025
**API Version:** `2023-06-01` (supports prompt caching)
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

## Executive Summary AI Prompt (v3.4)

**Latest Update:** November 2025 - Entity Tags, Retail Filter, Quality Review System

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

### November 2025 Enhancements

**1. Entity Tags & Category Labeling**
- **Entity Reference Lists:** Phase 2 prompt now includes entity reference lists (customers, suppliers, partners, competitors)
- **Inline Category Tags:** All bullets in competitive/industry section now include inline tags: `[CUSTOMER]`, `[SUPPLIER]`, `[PARTNER]`, `[COMPETITOR]`
- **Domain URL Replacement:** Post-processing automatically replaces domain URLs with formal publication names
  - Example: `"reuters.com"` → `"Reuters"`, `"wsj.com"` → `"The Wall Street Journal"`
  - Function: `post_process_executive_summary()` in app.py
- **Tag Formatting:** All brackets removed from category tags for cleaner display
- **Benefits:** Easier to identify relationships, better context for readers

**2. Retail Filter Extension**
- Upstream/downstream articles now filtered using same retail spam logic as company articles
- Prevents low-quality retail spam from polluting industry/competitor sections
- Blue ticker badges added to company articles in Email #3 for visual distinction

**3. AI Temperature Standardization**
- All AI functions now use consistent temperature settings across Claude and OpenAI
- Ensures reproducible outputs and consistent quality
- Applied to: triage, summaries, executive summary generation

**4. Value Chain Intelligence Enhancement**
- Better tracking of `value_chain_type` field (prevents NULL overwrites)
- Improved email formatting for value chain relationships
- Enhanced upstream/downstream flagging logic in Email #1

**5. Email #3 Improvements**
- Ticker badges with color coding for article categories
- Impact-based sorting (high impact articles appear first)
- Metadata temperature fixes (consistent AI behavior)

### Phase 2: Scenario Context Enrichment (October 2025)

**Purpose:** Enrich Phase 1 executive summary scenarios (Bottom Line, Upside, Downside) with synthesized context from the latest 10-K filing.

**Two-Phase Architecture:**

**Phase 1: Article Processing**
- Processes articles only (no filing data)
- Generates initial executive summary with 6 sections
- Includes content sufficiency checks (counts directional signals before generating scenarios)
- Output: Full JSON with all sections and bullets

**Phase 2: Filing Context Enrichment**
- Receives Phase 1 JSON + latest 10-K profile
- Generates 50-75 word context syntheses for 3 scenarios:
  - `bottom_line_context`: Strategic positioning from 10-K
  - `upside_scenario_context`: Growth drivers from 10-K
  - `downside_scenario_context`: Risk factors from 10-K
- **CRITICAL:** Completes bullet enrichments FIRST, then scenario contexts (prevents mixing)
- Output: Enrichments dict + scenario_contexts dict (separate)

**Prompts:**
- Phase 1: `modules/_build_executive_summary_prompt_phase1` (lines 806-976)
  - Content sufficiency checks for scenarios (counts directional signals)
  - Content sufficiency checks for key variables (counts forward-looking themes)
- Phase 2: `modules/_build_executive_summary_prompt_phase2` (lines 111-177, 1435-1474)
  - Updated output format with scenario context fields
  - 4-step synthesis process with task completion order

**Implementation Details:**

**Parsing Logic** (`modules/executive_summary_phase2.py:412-448`)
- Separates scenario contexts from bullet enrichments
- Handles 3 JSON format variations from Claude API
- Example:
  ```python
  enrichments = {}  # Bullet-level enrichments (context_source, impact, sentiment, etc.)
  scenario_contexts = {
      "bottom_line_context": "50-75 word synthesis...",
      "upside_scenario_context": "50-75 word synthesis...",
      "downside_scenario_context": "50-75 word synthesis..."
  }
  ```

**Merge Function** (`modules/executive_summary_phase2.py:715-730`)
- Adds context fields to scenario sections in Phase 1 JSON
- Example merged structure:
  ```python
  {
    "sections": {
      "bottom_line": {
        "content": "Phase 1 bottom line paragraph",
        "context": "Phase 2 10-K synthesis"  # NEW
      },
      "upside_scenario": {
        "content": "Phase 1 upside paragraph",
        "context": "Phase 2 10-K synthesis"  # NEW
      }
    }
  }
  ```

**Email Display** (`modules/executive_summary_phase1.py`)
- Email #3 Converter: `convert_phase1_to_sections_dict()` - User-facing format (lines 465-565)
- Email #2 Converter: `convert_phase1_to_enhanced_sections()` - QA format with hints (lines 609-713)
- Both converters use bullet_id matching for date appending
- Post-processing: `bold_labels=True` parameter on `build_section()` calls
- `bold_bullet_labels()` function auto-replaces `Context:` → `<strong>Context:</strong>`

**Database Storage:**
- Scenario contexts stored in `executive_summaries` table (part of full JSON)
- Retrieved during Email #3 generation
- Persists across email regenerations

**Key Functions:**
- `generate_phase2_enrichments_with_claude()` - Phase 2 generation
- `_parse_phase2_json_response()` - Parsing logic
- `merge_phase1_and_phase2()` - Phase 1+2 merge function
- `convert_phase1_to_sections_dict()` - Email #3 converter (NEW - uses bullet_id matching)
- `convert_phase1_to_enhanced_sections()` - Email #2 converter (NEW - uses bullet_id matching)

**Benefits:**
- ✅ Richer context for investment decisions
- ✅ 10-K insights integrated seamlessly with article-driven analysis
- ✅ Consistent display across Email #2 (QA) and Email #3 (user-facing)
- ✅ Unified bold formatting via post-processing
- ✅ Backward compatible (works even if no 10-K exists)

### Phase 3: Context Integration + Length Enforcement (November 2025)

**MAJOR SIMPLIFICATION:** Phase 3 now returns JSON (not markdown) with only integrated content. All restructuring logic removed.

**Purpose:** Mechanically weave Phase 2 context into Phase 1 content and enforce length limits. Phase 3 is purely editorial - no new information, no restructuring.

**Three-Phase Architecture (UPDATED):**

**Phase 1: Article Theme Extraction**
- Input: Article summaries by category
- Output: JSON with 10 sections (bullets have: bullet_id, topic_label, content, filing_hints)
- Scope: Articles ONLY

**Phase 2: Filing Context Enrichment**
- Input: Phase 1 JSON + latest 10-K/10-Q/Transcript
- Output: Enrichments (impact, sentiment, reason, entity, context) for each bullet
- Scope: Filing data ONLY
- Result: Phase 1+2 merged JSON with all metadata

**Phase 3: Context Integration (NEW)**
- Input: Phase 1+2 merged JSON
- Output: JSON with only (bullet_id, topic_label, content_integrated)
- Scope: Context weaving + length enforcement ONLY
- No restructuring, no thesis extraction, no new content
- **AI Provider:** Claude Sonnet 4.5 (primary) with Gemini 2.5 Pro fallback
- **Cost Impact:** +$135-210/month vs Gemini-primary (acceptable for quality improvement)

**Unified Bullet Format (All Emails):**
```
**[Entity] Topic • Sentiment (reason)**
Integrated paragraph with Phase 1 + Phase 2 context woven together (Nov 04)
  Filing hints: 10-K (Section A)  ← Email #2 only
  ID: bullet_id                    ← Email #2 only
```

**Key Improvements (November 2025):**

1. **Bullet ID Matching System**
   - All merging uses `bullet_id` (not index or topic_label)
   - Robust, order-independent, safe with duplicates
   - Function: `merge_phase3_with_phase2()` in Phase 2 module

2. **Shared Utilities** (`modules/executive_summary_utils.py`)
   - `format_bullet_header()`: Universal formatter for all bullet sections
   - `add_dates_to_email_sections()`: Date appending via bullet_id matching
   - Zero code duplication across Email #2, #3, #4

3. **Simplified Phase 3 Prompt** (`modules/_build_executive_summary_prompt_phase3_new`)
   - ❌ Removed: ALL restructuring (thesis extraction, bullet creation)
   - ✅ Focus: Context integration + length enforcement ONLY
   - 3 integration rules:
     1. Avoid parenthetical overload (limit to 1-2 metrics, use semicolons)
     2. Relevance upfront ("why this matters" at start, not buried)
     3. Consolidated attribution (group sources at end)
   - Length limits: Bottom Line ≤150w, Upside/Downside 80-160w

4. **Email Converter Redesign**
   - Both converters return `Dict[str, List[Dict]]` with `{'bullet_id': '...', 'formatted': '...'}`
   - Use shared utilities for formatting and dates
   - Backward compatible (HTML builder handles both old/new formats)

**Implementation Details:**

**Phase 3 Generation** (`modules/executive_summary_phase3.py:107-210`)
```python
def generate_executive_summary_phase3(ticker, phase2_merged_json, anthropic_api_key, gemini_api_key):
    # 1. Load simplified prompt
    # 2. Call Claude API (primary) - with prompt caching for cost savings
    # 3. If Claude fails: Fall back to Gemini 2.5 Pro
    # 4. Parse JSON response (handles 3 formats: plain, markdown blocks, text+JSON)
    # 5. merge_phase3_with_phase2(phase2_json, phase3_json)  # bullet_id matching
    # 6. Return final merged JSON
```

**Merge Function** (`modules/executive_summary_phase2.py:865-922`)
```python
def merge_phase3_with_phase2(phase2_json, phase3_json):
    # Build lookup by bullet_id from Phase 3
    phase3_map = {b['bullet_id']: b for b in phase3_bullets}

    # Overlay integrated content onto Phase 2 bullets
    for bullet in phase2_bullets:
        if bullet['bullet_id'] in phase3_map:
            bullet['content'] = phase3_map[bullet['bullet_id']]['content']

    # Preserves all Phase 2 metadata (impact, sentiment, dates, hints)
```

**Email Converters** (`modules/executive_summary_phase1.py`)
- `convert_phase1_to_enhanced_sections()`: Email #2 with filing hints (lines 609-713)
- `convert_phase1_to_sections_dict()`: Email #3 user-facing (lines 465-565)
- Both use bullet_id matching for date appending

**Data Flow:**
```
Articles → Phase 1 (synthesis)
         → Phase 2 (enrichment + metadata)
         → Phase 3 (context integration)
         → merge_phase3_with_phase2() [bullet_id matching]
         → Email converters [shared utilities]
         → HTML builder [handles both formats]
```

**Benefits:**
- ✅ Simplified Phase 3 = more predictable, reliable context integration
- ✅ Bullet ID matching = robust merging even if bullets reordered
- ✅ Shared utilities = zero code duplication
- ✅ Backward compatible = safe deployment
- ✅ 3,373 lines of redundant code flagged for deletion

**Deprecated Functions (Delete After Email #2 Migration):**
- `_build_executive_summary_prompt()` (2,609 lines) - OLD prompt system
- `generate_claude_executive_summary()` (79 lines) - OLD generation
- `generate_openai_executive_summary()` (54 lines) - OLD generation
- `generate_executive_summary_with_fallback()` (36 lines) - OLD wrapper
- `generate_ai_final_summaries()` (60 lines) - OLD async wrapper
- Phase 3 markdown functions (415 lines) - OLD markdown-based system

**Key Functions:**
- `generate_executive_summary_phase3()` - Main Phase 3 entry point (returns merged JSON)
- `merge_phase3_with_phase2()` - Bullet ID-based merge function
- `format_bullet_header()` - Universal bullet formatter
- `add_dates_to_email_sections()` - Date appending via bullet_id matching

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
---

## Migration History

### November 2025: Company Releases Migration

**Objective:** Unify FMP press releases and 8-K SEC filings into single `company_releases` table, eliminating wasteful dual-prompt processing.

**Changes:**

**Deleted (342 lines):**
- `modules/press_releases.py` (115 lines) - All functions queried dropped `press_releases` table
- `modules/company_profiles.py` (227 lines) - 3 deprecated functions:
  - `save_parsed_press_release_to_database()` - Queried dropped `parsed_press_releases` table
  - `get_parsed_press_releases_for_ticker()` - Queried dropped `parsed_press_releases` table
  - `delete_parsed_press_release()` - Queried dropped `parsed_press_releases` table

**Added:**
- `modules/company_releases.py` (211 lines) - Unified helper module with 6 functions for both FMP and 8-K
- Database table: `company_releases` - Single table for both sources

**Modified:**
- `app.py` (16 fixes):
  - FMP worker: Skip old `press_releases` table, save directly to `company_releases`
  - 8-K worker: Already saving to `company_releases` (no changes needed)
  - Manual generation endpoint: Updated to save to `company_releases`
  - Cascade delete endpoints: Updated to query `company_releases`
  - Email endpoints: Updated to query `company_releases`
  - Load Research Options: Updated to query `company_releases`
  - Removed deprecated function imports
- `templates/admin_research.html`: Removed "Press Releases" dropdown, unified view in "Company Releases"

**Architecture Before:**
```
FMP API → press_releases table (transcript prompt)
        → parsed_press_releases table (8k_filing_prompt)

8-K Edgar → sec_8k_filings table
          → parsed_press_releases table (8k_filing_prompt)
```

**Architecture After (November 2025):**
```
FMP API → company_releases table (8k_filing_prompt, source_type='fmp_press_release')

8-K Edgar → sec_8k_filings table
          → company_releases table (8k_filing_prompt, source_type='8k_exhibit')
```

**Benefits:**
- ✅ Single AI prompt for both sources (eliminated wasteful dual processing)
- ✅ Unified viewing in Research Library
- ✅ Cleaner codebase (-200 net lines)
- ✅ Better organization (separate FMP vs 8-K helpers)
- ✅ No duplicate 8-K processing

**Commit:** `70abd65` (November 20, 2025)
**Deployed:** Production

