# Scheduler Implementation Plan

**Status:** ✅ IMPLEMENTED (Simplified Architecture)
**Last Updated:** November 26, 2025

---

## Overview

**SIMPLIFIED ARCHITECTURE (Nov 26, 2025):** The unified scheduler handles only **4 report-related functions**. Memory-intensive jobs run as **separate Render crons** for isolation.

### What the Scheduler Handles (4 functions)
1. **Cleanup** - Delete old queue entries (offset before processing)
2. **Process** - Run/queue reports (daily @ 7am, weekly @ 2am)
3. **Send** - Send ready emails (daily @ 8:30am, weekly @ 7:30am)
4. **Export** - Nightly CSV backup (11:59pm)

### What Runs as Separate Crons (3 jobs)
1. **Morning Filings Check** - 6:00 AM EST daily
2. **Hourly Filings Check** - 8:30 AM - 10:30 PM EST (:30 each hour)
3. **Hourly Alerts** - 9:00 AM - 11:00 PM EST (:00 each hour)

**Why Separate?** These are memory-intensive operations that have crashed before. Separating them prevents memory spikes from affecting report processing.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RENDER CRON JOBS (4 total)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  UNIFIED SCHEDULER (*/30 * * * *)                                │    │
│  │  python app.py scheduler                                         │    │
│  │                                                                   │    │
│  │  Handles: Cleanup → Process → Send → Export                      │    │
│  │  - Timezone-aware (America/Toronto)                              │    │
│  │  - Day-of-week detection (Mon=weekly, Tue-Fri=daily)            │    │
│  │  - Database-configurable times                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  MORNING FILINGS CHECK (0 11 * * *)                              │    │
│  │  python app.py check_filings                                     │    │
│  │  6:00 AM EST daily - Standard tier for memory                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  HOURLY FILINGS CHECK (30 13-3 * * *)                            │    │
│  │  python app.py check_filings                                     │    │
│  │  8:30 AM - 10:30 PM EST - Standard tier for memory               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  HOURLY ALERTS (0 14-4 * * *)                                    │    │
│  │  python app.py alerts                                            │    │
│  │  9:00 AM - 11:00 PM EST - Standard tier for memory               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Database Tables

**`schedule_config`** - Per-day schedule configuration:
```sql
CREATE TABLE schedule_config (
    day_of_week INTEGER PRIMARY KEY,  -- 0=Monday, 6=Sunday
    report_type VARCHAR(10) NOT NULL,  -- 'daily', 'weekly', 'none'
    process_time TIME,                 -- When to start processing
    send_time TIME,                    -- When to send emails
    cleanup_offset_minutes INTEGER DEFAULT 60,
    filings_offset_minutes INTEGER DEFAULT 60,  -- Legacy, not used by scheduler
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Default Schedule:**
| Day | Report | Process | Send | Cleanup Offset |
|-----|--------|---------|------|----------------|
| Monday | weekly | 02:00 | 07:30 | 60 min |
| Tuesday | daily | 07:00 | 08:30 | 60 min |
| Wednesday | daily | 07:00 | 08:30 | 60 min |
| Thursday | daily | 07:00 | 08:30 | 60 min |
| Friday | daily | 07:00 | 08:30 | 60 min |
| Saturday | none | - | - | 60 min |
| Sunday | none | - | - | 60 min |

**Note:** `schedule_hourly_config` table still exists but is NOT used by the scheduler. Hourly jobs run as separate crons.

---

## How It Works

### Scheduler Flow

1. Get current Toronto time (auto-handles EST/EDT via `pytz.timezone('America/Toronto')`)
2. Load day's schedule config from database
3. Calculate cleanup time (process_time - cleanup_offset)
4. Check if current time is within ±15 min window of each task:
   - Cleanup → Process → Send
5. At 11:59 PM, run nightly backup (export)
6. Log summary of tasks run

### Time Window Logic

- Scheduler runs every 30 minutes (`*/30 * * * *`)
- Each task has a ±15 minute window
- With 30-min intervals, tasks only trigger once per window
- Example: Process at 7:00 AM
  - 6:30 scheduler: 7:00 outside ±15 min window → skip
  - 7:00 scheduler: 7:00 within ±15 min window → RUN
  - 7:30 scheduler: 7:00 outside ±15 min window → skip

### Report Type Override

When scheduler calls `process_daily_workflow()`, it passes `force_report_type` from the `schedule_config` table. This ensures UI settings are respected, overriding the default day-of-week detection.

---

## Deployment Instructions

### Step 1: Deploy Code

Push the updated `app.py` to trigger Render deployment.

### Step 2: Create/Update Render Crons

**Unified Scheduler:**
- Name: `Weavara-Scheduler`
- Schedule: `*/30 * * * *`
- Command: `python app.py scheduler`
- Tier: Starter (lightweight)

**Morning Filings Check:**
- Name: `Weavara-Morning-Filings`
- Schedule: `0 11 * * *` (6am EST = 11:00 UTC)
- Command: `python app.py check_filings`
- Tier: **Standard** (memory-intensive)

**Hourly Filings Check:**
- Name: `Weavara-Hourly-Filings`
- Schedule: `30 13-3 * * *` (8:30am-10:30pm EST = 13:30-03:30 UTC)
- Command: `python app.py check_filings`
- Tier: **Standard** (memory-intensive)

**Hourly Alerts:**
- Name: `Weavara-Hourly-Alerts`
- Schedule: `0 14-4 * * *` (9am-11pm EST = 14:00-04:00 UTC)
- Command: `python app.py alerts`
- Tier: **Standard** (memory-intensive)

### Step 3: Suspend Old Crons

Suspend (don't delete) these legacy cron jobs:
- Weavara-Daily CSV Backup (now handled by scheduler)
- Weavara-Auto Send Emails (now handled by scheduler)
- Weavara-Daily Cleanup (now handled by scheduler)
- Weavara-Daily Processing (now handled by scheduler)

---

## Rollback Instructions

If something goes wrong:

1. **Suspend** the new `Weavara-Scheduler` cron
2. **Resume** these old cron jobs:
   - Weavara-Daily CSV Backup
   - Weavara-Auto Send Emails
   - Weavara-Daily Cleanup
   - Weavara-Daily Processing
3. Keep the separate filings/alerts crons running (unchanged)

The old CLI commands still work:
```bash
python app.py cleanup
python app.py process
python app.py send
python app.py export
python app.py check_filings
python app.py alerts
```

---

## Admin UI

Access at: `/admin/schedule?token=YOUR_TOKEN`

**Features:**
- Configure process and send times per day of week
- Set report type (daily/weekly/none) per day
- Adjust cleanup offset (minutes before processing)
- Real-time Toronto time display
- Info box showing separate cron schedules

**Note:** Hourly jobs are NOT configurable in the UI since they run as separate crons. The UI displays their schedules for reference.

---

## Cron Comparison

### Before (7 Render Crons - Unified)
| Cron | Schedule | Issues |
|------|----------|--------|
| All in one scheduler | `*/30 * * * *` | Memory-intensive jobs could crash report processing |

### After (4 Render Crons - Separated)
| Cron | Schedule | Tier | Purpose |
|------|----------|------|---------|
| Scheduler | `*/30 * * * *` | Starter | Report workflow |
| Morning Filings | `0 11 * * *` | Standard | 6am daily check |
| Hourly Filings | `30 13-3 * * *` | Standard | 8:30am-10:30pm |
| Hourly Alerts | `0 14-4 * * *` | Standard | 9am-11pm |

**Benefits:**
- ✅ Memory isolation (intensive jobs can't crash reports)
- ✅ Automatic EST/EDT handling for scheduler
- ✅ Different schedules per day of week
- ✅ Database-configurable times via admin UI
- ✅ Independent scaling (can upgrade filings/alerts tier separately)

---

## Functions Reference

### Scheduler Functions (app.py)

| Function | Purpose |
|----------|---------|
| `run_scheduler()` | Main scheduler - runs cleanup, process, send, export |
| `get_schedule_config_for_day(day)` | Fetch schedule config from database |
| `is_within_time_window(current, target, window)` | Check if within ±N min of target |
| `cleanup_old_queue_entries()` | Delete old email queue entries |
| `process_daily_workflow(force_report_type)` | Queue reports for all active users |
| `auto_send_cron_job()` | Send all ready emails |
| `export_beta_users_csv()` | Nightly CSV backup |

### Separate Cron Functions (app.py)

| Function | Cron Command | Purpose |
|----------|--------------|---------|
| `check_all_filings_cron()` | `python app.py check_filings` | Check for new SEC filings |
| `process_hourly_alerts()` | `python app.py alerts` | Send hourly article alerts |

---

## Changelog

**November 26, 2025 - Simplified Architecture**
- Removed hourly jobs from unified scheduler
- Removed morning filings check from scheduler
- Scheduler now only handles: cleanup, process, send, export
- Memory-intensive jobs run as separate Render crons
- Updated admin UI to show separate cron schedules
- Removed filings offset setting (no longer used)

**November 26, 2025 - Initial Implementation**
- Created `schedule_config` and `schedule_hourly_config` tables
- Implemented `run_scheduler()` with timezone-aware logic
- Added `python app.py scheduler` CLI command
- Added admin UI at `/admin/schedule`
