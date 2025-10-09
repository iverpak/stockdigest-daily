# StockDigest Daily Workflow System

**Version:** 1.0
**Created:** October 8, 2025
**Status:** Production Ready
**Purpose:** Automated daily email delivery to beta users with admin review queue

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Database Schema](#database-schema)
4. [Admin Dashboard](#admin-dashboard)
5. [Daily Timeline](#daily-timeline)
6. [API Endpoints](#api-endpoints)
7. [Core Functions](#core-functions)
8. [Safety Systems](#safety-systems)
9. [Cron Jobs Setup](#cron-jobs-setup)
10. [Testing Guide](#testing-guide)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The Daily Workflow System automates the process of:
1. Reading active beta users from the database
2. Deduplicating tickers across users
3. Processing tickers (RSS feeds → AI triage → Email generation)
4. Queuing emails for admin review
5. Auto-sending emails at 8:30 AM (or manual send)

### Key Features

✅ **Admin Review Queue** - All emails sent to admin for preview before user delivery
✅ **Multiple Recipients** - Single ticker generates one email sent to multiple users
✅ **Unique Unsubscribe Tokens** - Each recipient gets personalized unsubscribe link
✅ **DRY_RUN Mode** - Test safely by redirecting all emails to admin
✅ **Server Restart Recovery** - Automatically handles crashed jobs
✅ **Heartbeat Monitoring** - Detects stalled processing
✅ **Manual Override** - Admin can send early, re-run, or cancel anytime

---

## Architecture

### Workflow Separation

**Your existing test workflow (UNCHANGED):**
```
PowerShell → setup_job_queue.ps1
          → ticker_processing_jobs table
          → Emails #1, #2, #3 → YOUR EMAIL ONLY
```

**New production workflow (SEPARATE):**
```
6:00 AM Cron → Cleanup old queue entries
7:00 AM Cron → Load beta_users (status='active')
             → Deduplicate tickers
             → Process 3 concurrent tickers
             → Save to email_queue (status='ready')
             → Send previews to admin
             → Send admin notification

8:30 AM Cron → Send emails to users
             → Replace {{UNSUBSCRIBE_TOKEN}}
             → BCC admin on all sends
             → Mark as sent
```

**Zero cross-contamination** - The systems use different tables and never interfere with each other.

---

## Database Schema

### `email_queue` Table

```sql
CREATE TABLE email_queue (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255),
    recipients TEXT[],  -- PostgreSQL array of email addresses
    email_html TEXT,    -- Contains {{UNSUBSCRIBE_TOKEN}} placeholder
    email_subject VARCHAR(500),
    article_count INTEGER,
    status VARCHAR(50) NOT NULL DEFAULT 'queued',
    error_message TEXT,
    is_production BOOLEAN DEFAULT TRUE,
    heartbeat TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    sent_at TIMESTAMPTZ,
    CONSTRAINT valid_status CHECK (status IN ('queued', 'processing', 'ready', 'failed', 'sent', 'cancelled'))
);
```

### Status Values

| Status | Meaning | Can Send? | Can Re-run? |
|--------|---------|-----------|-------------|
| `processing` | Currently being processed | No | Yes (cancels current) |
| `ready` | Email generated, ready to send | Yes | Yes |
| `failed` | Processing failed | No | Yes |
| `sent` | Email sent to recipients | No | Yes (generates new) |
| `cancelled` | Admin cancelled send | No | Yes |

### `beta_users` Table Updates

```sql
-- Default status changed from 'active' to 'pending'
status VARCHAR(50) DEFAULT 'pending'

-- Status values:
-- 'pending'   - Awaiting admin approval
-- 'active'    - Receiving daily emails
-- 'paused'    - Temporarily stopped
-- 'cancelled' - Unsubscribed
```

---

## Admin Dashboard

### Three Pages

#### 1. Landing Page: `/admin`

**URL:** `https://stockdigest.app/admin?token=YOUR_ADMIN_TOKEN`

**Features:**
- Real-time stats (pending users, active users, ready emails, sent today)
- Alert banner for pending items
- Navigation to users and queue pages
- Auto-refreshes every 30 seconds

#### 2. Beta User Management: `/admin/users`

**URL:** `https://stockdigest.app/admin/users?token=YOUR_ADMIN_TOKEN`

**Features:**
- **Bulk Selection** (NEW - Oct 2025)
  - Checkboxes on all active users
  - Select multiple users for batch processing
  - Real-time ticker counting: "X users selected (Y unique tickers)"
  - Sticky bulk actions bar appears when users selected
  - 📊 "Generate Reports" button with confirmation dialog
- **Pending Approvals Section** - Review new signups
  - Approve button → Sets status='active'
  - Reject button → Sets status='cancelled'
- **Active Users Section** - Manage current subscribers
  - Pause button → Sets status='paused'
  - Cancel button → Sets status='cancelled'
- **Inactive Users Section** - View paused/cancelled
  - Reactivate button → Sets status='active'

**User Card Shows:**
- Name, email
- 3 tickers
- Signup date
- Terms acceptance timestamp

#### 3. Email Queue: `/admin/queue`

**URL:** `https://stockdigest.app/admin/queue?token=YOUR_ADMIN_TOKEN`

**Features:**
- **Stats Grid** - Ready, Failed, Processing, Sent counts
- **7 Global Buttons:**
  - 📊 GENERATE ALL REPORTS - Manual trigger (= `python app.py process`)
  - ✉️ SEND ALL READY EMAILS - Send immediately
  - 🔄 RE-RUN QUEUE - Reprocess all queue entries (ready, failed, cancelled)
  - 🔄 RE-RUN ALL FAILED - Retry only failed tickers
  - ✅ APPROVE ALL CANCELLED - Restore cancelled emails
  - 🗑️ CLEAR ALL REPORTS - Delete all queue entries (= `python app.py cleanup`)
  - 🛑 EMERGENCY STOP ALL - Cancel all ready sends
- **Per-Ticker Cards:**
  - 👁️ View - Preview email HTML
  - 🔄 Re-run - Reprocess ticker
  - ❌ Cancel - Remove from queue
- **Auto-send countdown** - Shows time until 8:30 AM
- **Auto-refresh** - Every 30 seconds
- **Confirmation dialogs** - All buttons show impact counts and time estimates

---

## Daily Timeline

### 6:00 AM - Cleanup

**Function:** `cleanup_old_queue_entries()`
**Purpose:** Delete stale test emails before production run

```python
DELETE FROM email_queue
WHERE created_at < CURRENT_DATE
OR (is_production = FALSE AND created_at < NOW() - INTERVAL '1 day')
```

**Safety:** Prevents yesterday's test emails from being sent today.

---

### 7:00 AM - Processing

**Function:** `process_daily_workflow()`
**Duration:** ~30 minutes (for 30 tickers)

**Step 1: Load Active Beta Users**
```python
SELECT name, email, ticker1, ticker2, ticker3
FROM beta_users
WHERE status = 'active'
```

**Step 2: Deduplicate Tickers**
```python
# Input:
# John: AAPL, TSLA, MSFT
# Jane: AAPL, NVDA, GOOGL

# Output:
{
    "AAPL": ["john@email.com", "jane@email.com"],
    "TSLA": ["john@email.com"],
    "MSFT": ["john@email.com"],
    "NVDA": ["jane@email.com"],
    "GOOGL": ["jane@email.com"]
}
# Result: 5 unique tickers to process
```

**Step 3: Process Tickers (3 concurrent)**

For each ticker:
1. **Phase 1: Ingest (0-60%)**
   - Initialize feeds
   - Run RSS feed parsing (async, 5.5x faster)
   - Run AI triage (dual scoring)
   - **Send Email #1 to admin** (Article Selection QA)
   - Update heartbeat

2. **Phase 2: Digest (60-95%)**
   - Scrape article content (2-tier fallback)
   - Run AI analysis (Claude + OpenAI)
   - Generate executive summary
   - **Send Email #2 to admin** (Content QA)
   - Save executive summary to database
   - Update heartbeat

3. **Phase 3: Email Generation (95-97%)**
   - Fetch executive summary from database
   - Generate Email #3 HTML
   - Replace unsubscribe link with `{{UNSUBSCRIBE_TOKEN}}` placeholder
   - Save to `email_queue` (status='ready')
   - **Send Email #3 preview to admin** (no [PREVIEW] prefix)
   - Update heartbeat

**Step 4: Send Admin Notification**

Email sent to admin with:
- ✅ Ready: X emails
- ❌ Failed: Y emails
- Link to review dashboard
- Auto-send time (8:30 AM)
- List of failed tickers with error messages

---

### 7:00-8:30 AM - Admin Review Window

**Admin receives 4 emails per ticker:**
1. Email #1 (Article Selection QA)
2. Email #2 (Content QA)
3. Email #3 Preview (what users will see)
4. Admin notification (processing summary)

**Admin can:**
- Do nothing → emails auto-send at 8:30 AM
- Click "SEND ALL READY EMAILS" → send early
- Re-run individual tickers that look bad
- Re-run all failed tickers
- Cancel individual emails
- Emergency stop all sends

---

### 8:30 AM - Auto-Send

**Function:** `auto_send_cron_job()`
**Duration:** ~2 minutes (for 30 emails)

**Logic:**
```sql
SELECT * FROM email_queue
WHERE status = 'ready'
AND sent_at IS NULL
AND is_production = TRUE
AND created_at >= CURRENT_DATE
```

**For each email:**
1. For each recipient in recipients array:
   - Generate unique unsubscribe token
   - Replace `{{UNSUBSCRIBE_TOKEN}}` with real token
   - Send email to recipient
   - BCC: admin email
2. Mark as sent: `status='sent', sent_at=NOW()`

**Safety Checks:**
- Only sends `is_production=TRUE` emails
- Only sends emails created today
- Only sends emails with `sent_at IS NULL`
- Can be run multiple times safely (idempotent)

---

### 11:59 PM - Backup

**Function:** `export_beta_users_csv()`
**Purpose:** Daily backup of beta user data

**Output:** `/tmp/backups/beta_users_YYYYMMDD.csv`

**Columns:**
- name, email, ticker1, ticker2, ticker3
- status, created_at
- terms_accepted_at, privacy_accepted_at

**Future:** Optionally commit to GitHub repo for version control.

---

## API Endpoints

### Authentication

All endpoints require admin token via query parameter:
```
?token=YOUR_ADMIN_TOKEN
```

Validated by: `check_admin_token(token)` function

---

### User Management Endpoints

#### `GET /api/admin/stats`

Returns dashboard statistics.

**Response:**
```json
{
  "status": "success",
  "pending_users": 5,
  "active_users": 25,
  "ready_emails": 28,
  "sent_today": 0
}
```

#### `GET /api/admin/users`

Returns all beta users.

**Response:**
```json
{
  "status": "success",
  "users": [
    {
      "id": 1,
      "name": "John Smith",
      "email": "john@email.com",
      "ticker1": "AAPL",
      "ticker2": "TSLA",
      "ticker3": "MSFT",
      "status": "pending",
      "created_at": "2025-10-07T15:30:00Z",
      "terms_accepted_at": "2025-10-07T15:30:00Z"
    }
  ]
}
```

#### `POST /api/admin/approve-user`

Approve a pending user.

**Request:**
```json
{
  "token": "YOUR_ADMIN_TOKEN",
  "email": "john@email.com"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Approved john@email.com"
}
```

**Other user endpoints:**
- `POST /api/admin/reject-user` - Set status='cancelled'
- `POST /api/admin/pause-user` - Set status='paused'
- `POST /api/admin/cancel-user` - Set status='cancelled'
- `POST /api/admin/reactivate-user` - Set status='active'

---

### Email Queue Endpoints

#### `GET /api/queue-status`

Returns email queue status.

**Response:**
```json
{
  "status": "success",
  "stats": {
    "ready": 28,
    "failed": 2,
    "processing": 0,
    "sent": 0,
    "cancelled": 0
  },
  "tickers": [
    {
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "recipients": ["user1@email.com", "user2@email.com"],
      "email_subject": "📊 Stock Intelligence: Apple Inc. (AAPL) - 25 articles analyzed",
      "article_count": 25,
      "status": "ready",
      "error_message": null,
      "heartbeat": "2025-10-08T07:15:00Z",
      "created_at": "2025-10-08T07:00:00Z",
      "updated_at": "2025-10-08T07:15:00Z",
      "sent_at": null
    }
  ]
}
```

#### `POST /api/send-all-ready`

Send all ready emails immediately.

**Request:**
```json
{
  "token": "YOUR_ADMIN_TOKEN"
}
```

**Response:**
```json
{
  "status": "success",
  "sent_count": 28,
  "failed_count": 0,
  "failed_tickers": []
}
```

**Behavior:**
- Sends to all recipients with unique unsubscribe tokens
- BCCs admin on all sends
- Marks as sent (status='sent', sent_at=NOW())
- Can be called multiple times (only sends unsent emails)

#### `POST /api/emergency-stop`

Cancel all pending sends.

**Request:**
```json
{
  "token": "YOUR_ADMIN_TOKEN"
}
```

**Response:**
```json
{
  "status": "success",
  "cancelled_count": 28,
  "message": "Emergency stop: 28 emails cancelled"
}
```

**Behavior:**
```sql
UPDATE email_queue
SET status = 'cancelled', updated_at = NOW()
WHERE status IN ('ready', 'processing')
```

#### `POST /api/approve-all-cancelled`

Restore cancelled emails to ready status.

**Response:**
```json
{
  "status": "success",
  "approved_count": 25,
  "message": "25 emails changed from cancelled to ready"
}
```

#### `POST /api/cancel-ticker`

Cancel individual ticker.

**Request:**
```json
{
  "token": "YOUR_ADMIN_TOKEN",
  "ticker": "AAPL"
}
```

#### `GET /api/view-email/{ticker}?token=...`

View email HTML preview in browser.

**Returns:** HTML content (opens in new tab)

---

## Core Functions

### 1. `load_active_beta_users()`

**Location:** Line 17569
**Purpose:** Load active beta users and deduplicate tickers
**Returns:** `Dict[str, List[str]]` - {ticker: [emails]}

**Logic:**
```python
ticker_recipients = {}

for user in active_users:
    for ticker in [user.ticker1, user.ticker2, user.ticker3]:
        if ticker not in ticker_recipients:
            ticker_recipients[ticker] = []
        if email not in ticker_recipients[ticker]:
            ticker_recipients[ticker].append(email)
```

---

### 2. `generate_email_html_for_queue()`

**Location:** Line 17615
**Purpose:** Generate Email #3 HTML with placeholder token
**Returns:** `{"status": "success", "html": "...", "subject": "...", ...}`

**Key Feature:** Intercepts `send_user_intelligence_report()` to capture HTML before sending, then replaces unsubscribe token with `{{UNSUBSCRIBE_TOKEN}}` placeholder.

---

### 3. `process_ticker_for_daily_workflow()`

**Location:** Line 17709
**Purpose:** Process single ticker (ingest → digest → email generation)
**Timeout:** 10 minutes
**Returns:** `{"status": "success|failed", "ticker": "...", "duration": ...}`

**Phases:**
1. Insert/update email_queue (status='processing')
2. Initialize feeds
3. Run ingest (sends Email #1)
4. Update heartbeat
5. Run digest (sends Email #2, saves executive summary)
6. Update heartbeat
7. Generate Email #3 HTML
8. Save to email_queue (status='ready')
9. Send preview to admin

**Error Handling:**
- Try/catch wraps entire process
- On failure: Mark as failed in email_queue
- Logs full stacktrace

---

### 4. `process_all_tickers_daily()`

**Location:** Line 17857
**Purpose:** Process all tickers with 3 concurrent workers
**Returns:** `{"status": "success", "total": 30, "succeeded": 28, "failed": 2}`

**Concurrency:**
```python
semaphore = asyncio.Semaphore(3)  # Max 3 concurrent

async def process_with_semaphore(ticker, recipients):
    async with semaphore:
        return await asyncio.wait_for(
            process_ticker_for_daily_workflow(ticker, recipients),
            timeout=600  # 10 minutes
        )
```

---

### 5. `send_all_ready_emails_impl()`

**Location:** Line 18023
**Purpose:** Send all ready emails with unique unsubscribe tokens
**Returns:** `{"status": "success", "sent_count": 28, "failed_count": 0}`

**Logic:**
```python
for email in ready_emails:
    for recipient in email['recipients']:
        # Generate unique token
        token = get_or_create_unsubscribe_token(recipient)

        # Replace placeholder
        final_html = email['email_html'].replace(
            '{{UNSUBSCRIBE_TOKEN}}',
            token
        )

        # Send with DRY_RUN check
        send_email_with_dry_run(
            subject=email['email_subject'],
            html=final_html,
            to=recipient,
            bcc=admin_email
        )
```

---

### 6. `send_email_with_dry_run()`

**Location:** Line 17999
**Purpose:** Email sending wrapper with DRY_RUN mode

**DRY_RUN Mode:**
```python
if os.getenv('DRY_RUN', 'false').lower() == 'true':
    # Override recipients
    actual_to = admin_email
    actual_bcc = None
    subject = f"[DRY RUN] {subject}"
```

**Usage:** Set `DRY_RUN=true` in Render environment for testing.

---

## Safety Systems

### Layered Defense Architecture (4 Levels)

**UPDATED: October 9, 2025** - Enhanced dead worker detection for Render rolling deployments

StockDigest uses a 4-layer defense system to ensure jobs never get stuck:

1. **Startup Recovery** (one-time, 3-min threshold)
2. **Job Queue Reclaim Thread** (continuous, 3-min threshold) ⭐ **NEW**
3. **Email Watchdog Thread** (continuous, 3-min threshold)
4. **Timeout Watchdog Thread** (continuous, 45-min limit)

---

### 1. Startup Recovery

**Location:** Line 13806-13841 (Job Queue), Line 13847-13882 (Email Queue)
**Function:** `startup_event()` - Reclaims orphaned jobs from crashed workers
**Trigger:** Server restart (Render deployment, crash, manual restart)
**Threshold:** 3 minutes

**Job Queue Recovery:**
```python
# Requeue jobs stuck >3 minutes (worker likely dead)
UPDATE ticker_processing_jobs
SET status = 'queued',
    started_at = NULL,
    worker_id = NULL,
    phase = 'restart_recovery',
    error_message = '... | Server restart detected, job reclaimed'
WHERE status = 'processing'
AND started_at < NOW() - INTERVAL '3 minutes'
```

**Email Queue Recovery:**
```python
# Mark stuck email jobs as failed (server restarted during processing)
UPDATE email_queue
SET status = 'failed',
    error_message = 'Server restarted during processing'
WHERE status = 'processing'
```

**Why 3 minutes?** Balances fast recovery vs avoiding false positives during normal processing.

---

### 2. Job Queue Reclaim Thread ⭐ **NEW - Oct 2025**

**Location:** Line 13665-13721
**Function:** `job_queue_reclaim_loop()`
**Check Frequency:** Every 60 seconds
**Threshold:** 3 minutes of stale `last_updated` timestamp

**Purpose:** **CRITICAL - Prevents jobs from getting stuck forever during Render rolling deployments.**

**The Problem:**
During Render rolling deployments, 2 instances run simultaneously:
1. NEW instance receives job submission request
2. OLD instance claims the job (load balancer routing)
3. Render sends SIGTERM to OLD instance 9 seconds later
4. Job dies at 0.2 minutes old
5. Startup recovery threshold was 5 minutes → Job stuck forever ❌

**The Solution:**
Continuous monitoring thread detects dead workers via stale heartbeat and requeues jobs:

```python
UPDATE ticker_processing_jobs
SET status = 'queued',
    started_at = NULL,
    worker_id = NULL,
    phase = 'reclaimed_dead_worker',
    error_message = '... | Reclaimed: Dead worker detected (heartbeat stale >3min)'
WHERE status = 'processing'
AND last_updated < NOW() - INTERVAL '3 minutes'
```

**Key Features:**
- Runs continuously (not just at startup)
- Requeues jobs (not cancels them) for automatic retry
- Logs detailed reclaim info: ticker, job_id, worker_id, phase, progress, stale duration
- Started automatically in startup event (daemon thread)

**Example Log Output:**
```
🔄 Job queue reclaim thread reclaimed 1 jobs with stale heartbeat:
   → AMD (job_id: bf2e83d8..., worker: srv-d2t161odl3ps73fvfqv0-x98bg,
      was digest_complete at 95%, stale for 3.1min)
   → Job bf2e83d8... requeued in batch abc123
```

---

### 3. Heartbeat System

**For Job Queue (`ticker_processing_jobs`):**
- **Update Frequency:** On every progress change via `update_job_status()`
- **Field:** `last_updated` timestamp
- **Location:** Line 13158 - automatically includes `last_updated = NOW()` in every update

**For Email Queue (`email_queue`):**
- **Update Frequency:** After each phase (ingest, digest, email generation)
- **Field:** `heartbeat` timestamp
- **Locations:** Lines 18529, 18551, 18581

**Update Logic (Email Queue):**
```python
with db() as conn, conn.cursor() as cur:
    cur.execute("""
        UPDATE email_queue
        SET heartbeat = NOW()
        WHERE ticker = %s
    """, (ticker,))
```

**Purpose:** Both watchdog threads use timestamps to detect stalled jobs.

---

### 4. Email Queue Watchdog Thread

**Location:** Line 13722-13754
**Function:** `email_queue_watchdog_loop()`
**Check Frequency:** Every 60 seconds
**Threshold:** 3 minutes of stale heartbeat (UPDATED from 5 minutes)

**Logic:**
```python
UPDATE email_queue
SET status = 'failed',
    error_message = 'Processing stalled (no heartbeat for 3 minutes)'
WHERE status = 'processing'
AND (heartbeat IS NULL OR heartbeat < NOW() - INTERVAL '3 minutes')
```

**Starts:** Automatically on FastAPI startup (daemon thread)

**Purpose:** Detects stalled production jobs during daily workflow processing.

---

### 5. Timeout Watchdog Thread

**Location:** Line 13625-13663
**Function:** `timeout_watchdog_loop()`
**Check Frequency:** Every 60 seconds
**Threshold:** 45 minutes (job-specific `timeout_at` field)

**Logic:**
```python
UPDATE ticker_processing_jobs
SET status = 'timeout',
    error_message = 'Job exceeded timeout limit',
    completed_at = NOW()
WHERE status = 'processing'
AND timeout_at < NOW()
```

**Purpose:** Catches jobs that run abnormally long (last line of defense).

---

### 6. Cleanup Safety

**Function:** `cleanup_old_queue_entries()`
**Purpose:** Prevents stale test emails from being sent

**Deletes:**
- Emails created before today
- Test emails (`is_production=FALSE`) older than 1 day

**Does NOT delete:**
- Today's production emails
- Emails with status='sent' (already sent)

---

### 7. DRY_RUN Mode

**Environment Variable:** `DRY_RUN=true`

**Behavior:**
- All emails redirect to `ADMIN_EMAIL`
- Subject prefixed with `[DRY RUN]`
- Original TO/BCC logged to console
- No emails sent to users

**Use Case:** Testing beta workflow without spamming users.

---

## Cron Jobs Setup

### Render Dashboard Configuration

**Go to:** Your service → Settings → Cron Jobs

---

### Cron Job 1: Daily Cleanup

```
Name: Daily Cleanup
Schedule: 0 10 * * *  (6:00 AM EST = 10:00 AM UTC)
Command: python app.py cleanup
```

**Purpose:** Delete old queue entries
**Duration:** <1 second
**Logs:** Check for "✅ Cleanup complete: X old entries deleted"

---

### Cron Job 2: Daily Processing

```
Name: Daily Processing
Schedule: 0 11 * * *  (7:00 AM EST = 11:00 AM UTC)
Command: python app.py process
```

**Purpose:** Process all active beta users
**Duration:** ~30 minutes (for 30 tickers)
**Logs:**
- Look for "Loading active beta users..."
- "Processing X tickers (max 3 concurrent)"
- "[TICKER] ✅ Daily workflow complete"
- "✅ Daily workflow complete"

**Failures:** Check for "[TICKER] ❌ Daily workflow failed"

---

### Cron Job 3: Auto-Send

```
Name: Auto Send Emails
Schedule: 30 12 * * *  (8:30 AM EST = 12:30 PM UTC)
Command: python app.py send
```

**Purpose:** Send all ready emails to users
**Duration:** ~2 minutes (for 30 emails)
**Logs:**
- "Auto-sending X ready emails"
- "✅ Sent TICKER to recipient@email.com"
- "✅ Auto-send complete: X sent"

**Skip Logic:** If admin already sent manually, logs "No emails to send"

---

### Cron Job 4: CSV Backup

```
Name: Daily CSV Backup
Schedule: 59 3 * * *  (11:59 PM EST = 3:59 AM UTC next day)
Command: python app.py export
```

**Purpose:** Export beta users to CSV
**Duration:** <1 second
**Output:** `/tmp/backups/beta_users_YYYYMMDD.csv`

---

### Timezone Considerations

**IMPORTANT:** Render cron jobs run in **UTC timezone**.

**Conversion Table (EST → UTC):**
- 6:00 AM EST = **10:00 AM UTC** → Use schedule: `0 10 * * *`
- 7:00 AM EST = **11:00 AM UTC** → Use schedule: `0 11 * * *`
- 8:30 AM EST = **12:30 PM UTC** → Use schedule: `30 12 * * *`
- 11:59 PM EST = **3:59 AM UTC (next day)** → Use schedule: `59 3 * * *`

**During EDT (Daylight Saving Time, Mar-Nov):** Add 1 hour to UTC times above.

The schedules above are already configured for UTC - use them as-is in Render cron jobs.

---

## Testing Guide

### Phase 1: Test with Your Own Email (DRY_RUN)

**Setup:**
1. Set `DRY_RUN=true` in Render environment
2. Add yourself as a beta user:
```sql
INSERT INTO beta_users (name, email, ticker1, ticker2, ticker3, status, terms_accepted_at, privacy_accepted_at)
VALUES ('Your Name', 'your@email.com', 'AAPL', 'TSLA', 'MSFT', 'pending', NOW(), NOW());
```
3. Approve yourself in `/admin/users`

**Test Processing:**
```bash
python app.py cleanup   # Should delete 0 (no old entries)
python app.py process   # Should process 3 tickers (AAPL, TSLA, MSFT)
```

**Expected Results:**
- 4 emails per ticker (Emails #1, #2, #3 preview, admin notification)
- All emails sent to YOU only (DRY_RUN redirects)
- Check `/admin/queue` - should see 3 tickers with status='ready'

**Test Sending:**
```bash
python app.py send  # Should send 3 emails to YOU
```

**Expected Results:**
- 3 emails sent to YOU (prefixed with [DRY RUN])
- BCC to you as well
- Check `/admin/queue` - all 3 tickers now status='sent'

---

### Phase 2: Test Dashboard

**Access:**
```
https://stockdigest.app/admin?token=YOUR_ADMIN_TOKEN
```

**Test User Management:**
1. Add another test user (status='pending')
2. Click "Approve" - should change to 'active'
3. Click "Pause" - should change to 'paused'
4. Click "Reactivate" - should change back to 'active'
5. Click "Cancel" - should change to 'cancelled'

**Test Bulk Selection (NEW):**
1. Go to `/admin/users`
2. Check boxes next to 2 active users (including your test account)
3. Sticky bar appears showing "2 users selected (X unique tickers)"
4. Click "📊 Generate Reports"
5. Confirm dialog shows ticker list and time estimate
6. Wait 10-15 minutes
7. Go to `/admin/queue` - should see unique tickers with status='ready'

**Test Queue Management:**
1. **Option A:** Run `python app.py process` from terminal
2. **Option B:** Click "📊 GENERATE ALL REPORTS" in `/admin/queue`
3. Click "View" on a ticker - should open email preview
4. Click "Re-run" on a ticker - should reprocess (watch logs)
5. Click "Cancel" on a ticker - should change status to 'cancelled'
6. Click "APPROVE ALL CANCELLED" - should change back to 'ready'
7. Click "EMERGENCY STOP ALL" - should cancel all
8. Click "APPROVE ALL CANCELLED" - should restore all

**Test Sending:**
1. Click "✉️ SEND ALL READY EMAILS"
2. Should send all emails to YOU (DRY_RUN)
3. Verify BCC works (you should receive all)
4. Check status changed to 'sent'

**Test Cleanup (NEW):**
1. Click "🗑️ CLEAR ALL REPORTS"
2. Confirm dialog shows breakdown by status
3. All queue entries deleted
4. Ready for next test run

---

### Phase 3: Test with Real Beta User (Still DRY_RUN)

**Setup:**
1. Keep `DRY_RUN=true` (safety)
2. Add a friend's email as beta user
3. Approve them in `/admin/users`
4. Run `python app.py process`

**Expected Results:**
- If you share a ticker, that ticker should have 2 recipients
- Both of you in the `recipients` array
- Preview sent to YOU only (admin)
- Final emails sent to YOU only (DRY_RUN redirect)

**Verify:**
```sql
SELECT ticker, recipients FROM email_queue WHERE ticker = 'SHARED_TICKER';
-- Should show: {"your@email.com", "friend@email.com"}
```

---

### Phase 4: Production (DRY_RUN=false)

**⚠️ ONLY WHEN CONFIDENT**

**Setup:**
1. Set `DRY_RUN=false` in Render
2. Test with 1-2 users first
3. Monitor `/admin/queue` closely

**First Production Run:**
```bash
# Process just 1 user's tickers
python app.py process  # Processes all active users

# Review in dashboard before sending
# Go to /admin/queue
# Click "View" on each ticker to verify

# Send when ready
python app.py send
```

**Verify:**
- Users receive emails
- Unsubscribe links work (each unique)
- You receive BCC copies
- Dashboard shows status='sent'

---

## Troubleshooting

### Issue: No Active Beta Users Found

**Symptom:**
```
Loading active beta users...
Found 0 active beta users
```

**Cause:** All users have status='pending' or 'cancelled'

**Fix:**
```sql
-- Check user statuses
SELECT email, status FROM beta_users;

-- Approve users
UPDATE beta_users SET status='active' WHERE email='user@email.com';

-- Or use dashboard: /admin/users → Click "Approve"
```

---

### Issue: Tickers Timing Out

**Symptom:**
```
[TICKER] ⏱️ Timeout after 10 minutes
```

**Causes:**
1. API rate limits (OpenAI/Claude)
2. Slow scraping
3. Too many articles

**Fixes:**
1. Check API usage in OpenAI/Claude dashboards
2. Re-run the ticker (may succeed on retry)
3. Increase timeout (currently 600 seconds)

---

### Issue: Email Queue Stuck in 'processing'

**Symptom:**
Dashboard shows ticker stuck at 'processing' for >10 minutes

**Causes:**
1. Server crashed during processing
2. Heartbeat not updating
3. Watchdog not running

**Fixes:**
1. **Automatic:** Watchdog kills after 5 min (check logs)
2. **Manual:** Restart server (triggers startup recovery)
3. **Emergency:**
```sql
UPDATE email_queue
SET status='failed', error_message='Manual recovery'
WHERE status='processing' AND ticker='STUCK_TICKER';
```

---

### Issue: Preview Emails Not Received

**Symptom:**
Processing completes but admin doesn't receive Email #1, #2, #3 previews

**Causes:**
1. SMTP credentials wrong
2. Email going to spam
3. `ADMIN_EMAIL` not set

**Fixes:**
1. Check Render environment: `ADMIN_EMAIL=stockdigest.research@gmail.com`
2. Check spam folder
3. Test email sending:
```python
from app import send_email
send_email("Test", "<h1>Test</h1>", to="your@email.com")
```

---

### Issue: Unsubscribe Links Don't Work

**Symptom:**
Users click unsubscribe, get "Invalid token" error

**Causes:**
1. Token not generated
2. Email not in beta_users table
3. Token expired/used

**Debug:**
```sql
-- Check if token exists
SELECT * FROM unsubscribe_tokens WHERE user_email='user@email.com';

-- Check if user exists
SELECT * FROM beta_users WHERE email='user@email.com';

-- Manually create token
INSERT INTO unsubscribe_tokens (user_email, token)
VALUES ('user@email.com', 'RANDOM_TOKEN_HERE');
```

---

### Issue: DRY_RUN Not Working

**Symptom:**
Emails sent to users even with `DRY_RUN=true`

**Causes:**
1. Environment variable not set in Render
2. Typo in env var name
3. Value is not exactly 'true'

**Fixes:**
1. Verify in Render dashboard: `DRY_RUN=true` (lowercase)
2. Check logs for:
```
🧪 DRY_RUN: Redirecting email to admin@email.com
```
3. Test locally:
```bash
export DRY_RUN=true
python app.py send
```

---

### Issue: Auto-Send Not Triggering

**Symptom:**
8:30 AM passes, no emails sent

**Causes:**
1. Cron job not configured in Render
2. Server timezone wrong
3. No emails with status='ready'

**Fixes:**
1. Check Render Cron Jobs tab (should show "Auto Send Emails")
2. Check cron run history (Render shows last runs)
3. Verify timezone:
```bash
date  # Should show correct time
```
4. Manual trigger:
```bash
python app.py send
```

---

### Issue: Emergency Stop Doesn't Work

**Symptom:**
Click "EMERGENCY STOP", emails still send

**Causes:**
1. Emails already sent (status='sent')
2. Auto-send cron ran before emergency stop
3. Browser cache (dashboard not refreshing)

**Fixes:**
1. Check status in database:
```sql
SELECT ticker, status, sent_at FROM email_queue;
```
2. If emails already sent, too late (they're gone)
3. Hard refresh dashboard (Cmd+Shift+R)

---

### Issue: Re-run Doesn't Work

**Symptom:**
Click "Re-run", ticker stays in same state

**Causes:**
1. Processing failed silently
2. Feeds not initialized
3. API keys missing

**Fixes:**
1. Check Render logs for errors:
```
[TICKER] ❌ Daily workflow failed: ...
```
2. Look for specific error:
   - "No config found" → Initialize ticker metadata
   - "No executive summary" → Digest phase failed
   - "API error" → Check OpenAI/Claude keys

---

## Advanced Topics

### Multiple Sends Per Day

**Scenario:** Admin wants to send emails at 7 AM and 1 PM

**Solution:**
1. Add second cron job:
```
Name: Afternoon Processing
Schedule: 0 12 * * *  # 12 PM UTC = 8 AM ET
Command: python app.py process
```
2. Add second send job:
```
Name: Afternoon Send
Schedule: 0 13 * * *  # 1 PM UTC = 9 AM ET
Command: python app.py send
```

**Cleanup runs once at 6 AM only.**

---

### Testing Without Sending

**Use Case:** Process tickers, generate emails, but don't send

**Solution:**
1. Set `DRY_RUN=true`
2. Run `python app.py process`
3. Review in `/admin/queue`
4. Don't run `python app.py send`
5. Next day, 6 AM cleanup deletes them

---

### Skipping Auto-Send

**Use Case:** Admin reviews queue, decides to skip sending today

**Solution:**
1. Go to `/admin/queue`
2. Click "EMERGENCY STOP ALL"
3. 8:30 AM cron runs, finds 0 ready emails, skips
4. Tomorrow, 6 AM cleanup deletes cancelled emails

---

### Re-sending Same Email

**Use Case:** User reports they didn't receive email

**Solution:**
1. Check if sent:
```sql
SELECT status, sent_at FROM email_queue WHERE ticker='TICKER';
```
2. If status='sent', re-run to generate new email:
   - Click "Re-run" in `/admin/queue`
   - Wait for completion
   - Click "SEND ALL READY EMAILS"

**Note:** Re-running overwrites the previous email HTML.

---

## Production Checklist

Before going live:

- [ ] `DRY_RUN=true` set in Render (test mode)
- [ ] `ADMIN_EMAIL` set correctly
- [ ] `ADMIN_TOKEN` set (long random string)
- [ ] All 4 cron jobs configured in Render
- [ ] Beta users approved in `/admin/users`
- [ ] Test processing with your email only
- [ ] Test dashboard buttons work
- [ ] Test unsubscribe links work
- [ ] Verify emails not going to spam
- [ ] Set `DRY_RUN=false` when confident
- [ ] Monitor first live send closely

---

## Support

**Dashboard:** `https://stockdigest.app/admin?token=YOUR_ADMIN_TOKEN`
**Contact:** stockdigest.research@gmail.com
**Logs:** Render.com → Your Service → Logs tab
**Database:** Direct SQL access via Render dashboard

---

**Last Updated:** October 8, 2025
**Version:** 1.0
**Status:** Production Ready
