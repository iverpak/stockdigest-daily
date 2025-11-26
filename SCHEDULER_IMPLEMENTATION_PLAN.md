# Scheduler Implementation Plan

**Status:** ✅ IMPLEMENTED
**Last Updated:** November 26, 2025

---

## Overview

Replaced 7 separate Render cron jobs with 1 timezone-aware scheduler that automatically handles EST/EDT transitions and allows schedule configuration via database.

---

## What Was Built

### Database Tables

**`schedule_config`** - Per-day schedule configuration:
```sql
CREATE TABLE schedule_config (
    day_of_week INTEGER PRIMARY KEY,  -- 0=Monday, 6=Sunday
    report_type VARCHAR(10) NOT NULL,  -- 'daily', 'weekly', 'none'
    process_time TIME,                 -- When to start processing
    send_time TIME,                    -- When to send emails
    cleanup_offset_minutes INTEGER DEFAULT 60,
    filings_offset_minutes INTEGER DEFAULT 60,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Default values:**
| Day | Report | Process | Send | Cleanup/Filings Offset |
|-----|--------|---------|------|------------------------|
| Monday | weekly | 02:00 | 07:30 | 60 min |
| Tuesday | daily | 07:00 | 08:30 | 60 min |
| Wednesday | daily | 07:00 | 08:30 | 60 min |
| Thursday | daily | 07:00 | 08:30 | 60 min |
| Friday | daily | 07:00 | 08:30 | 60 min |
| Saturday | none | - | - | 60 min |
| Sunday | none | - | - | 60 min |

**`schedule_hourly_config`** - Hourly job configuration:
```sql
CREATE TABLE schedule_hourly_config (
    job_type VARCHAR(50) PRIMARY KEY,  -- 'filings_check', 'alerts', 'backup'
    start_hour INTEGER NOT NULL,       -- Start hour (0-23)
    end_hour INTEGER NOT NULL,         -- End hour (0-23)
    run_on_half_hour BOOLEAN DEFAULT FALSE,
    enabled BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Default values:**
| Job Type | Start | End | Runs At | Enabled |
|----------|-------|-----|---------|---------|
| filings_check | 8 | 20 | :30 | TRUE |
| alerts | 9 | 21 | :00 | TRUE |
| backup | 23 | 23 | :00 | TRUE |

### Functions Added

**`get_schedule_config_for_day(day_of_week)`** - Fetch schedule config for a day
**`get_hourly_job_config(job_type)`** - Fetch hourly job config
**`is_within_time_window(current_time, target_time, window_minutes=15)`** - Check if within ±15 min
**`is_hourly_job_time(current_hour, current_minute, job_config)`** - Check if hourly job should run
**`run_scheduler()`** - Main scheduler function

### CLI Command

```bash
python app.py scheduler
```

---

## How It Works

### Scheduler Flow

1. Get current Toronto time (auto-handles EST/EDT via `pytz.timezone('America/Toronto')`)
2. Load day's schedule config from database
3. For daily workflow tasks (cleanup, morning filings, process, send):
   - Calculate target times based on config
   - Check if current time is within ±15 min window
   - Run task if in window
4. For hourly jobs (filings check, alerts, backup):
   - Check if within configured hour range
   - Check if current minute matches (:00 or :30)
   - Run task if conditions met
5. Log summary of tasks run

### Time Window Logic

- Scheduler runs every 30 minutes (`*/30 * * * *`)
- Each task has a ±15 minute window
- With 30-min intervals, tasks only trigger once per window
- Example: Process at 7:00 AM
  - 6:30 scheduler: 7:00 outside ±15 min window → skip
  - 7:00 scheduler: 7:00 within ±15 min window → RUN
  - 7:30 scheduler: 7:00 outside ±15 min window → skip

### Hourly Jobs

- **filings_check**: Runs at :30 (minute 15-44) within 8am-8pm
- **alerts**: Runs at :00 (minute 0-14 or 45-59) within 9am-9pm
- **backup**: Runs at :00 only at 11pm

---

## Deployment Instructions

### Step 1: Deploy Code

Push the updated `app.py` to trigger Render deployment. The schema will auto-create on startup.

### Step 2: Create New Render Cron

1. Go to Render Dashboard → New → Cron Job
2. Configure:
   - Name: `Weavara-Scheduler`
   - Schedule: `*/30 * * * *`
   - Command: `python app.py scheduler`
   - Service: Same as your web service

### Step 3: Suspend Old Crons

Suspend (don't delete) these 7 cron jobs:
- Weavara-Daily CSV Backup
- Weavara-Auto Send Emails
- Weavara-Daily Cleanup
- Weavara-Hourly Filings Check
- Weavara-Daily Processing
- Weavara-Morning Filings Check
- Weavara-Hourly Alerts

### Step 4: Monitor

Watch the scheduler logs for a few days to verify correct behavior.

---

## Rollback Instructions

If something goes wrong:

1. **Suspend** the new `Weavara-Scheduler` cron
2. **Resume** all 7 old cron jobs
3. Everything works exactly as before (code unchanged)

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

## Testing

### Manual Test

Run scheduler manually to see what would execute:
```bash
python app.py scheduler
```

Output shows:
- Current Toronto time
- Day of week
- Schedule config for today
- Derived times (cleanup, filings)
- Which tasks ran (if any)

### Logic Tests

Time window and hourly job logic tested with assertions:
- `is_within_time_window()` - 6 test cases
- `is_hourly_job_time()` - 12 test cases

All tests passed.

---

## Future: Admin UI

A `/admin/schedule` page can be built to edit these database tables. The scheduler already reads from the database, so UI changes take effect immediately.

**Proposed UI:**
```
┌─────────────────────────────────────────────────────────────────┐
│  📅 Schedule Configuration                                      │
├─────────────────────────────────────────────────────────────────┤
│  Day         Report Type     Process Time     Send Time         │
│  ───────────────────────────────────────────────────────────    │
│  Monday      [Weekly ▼]      [ 2:00 AM ]      [ 7:30 AM ]      │
│  Tuesday     [Daily  ▼]      [ 7:00 AM ]      [ 8:30 AM ]      │
│  ...                                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cron Comparison

### Before (7 Render Crons)
| Cron | UTC Time | EST Time | Command |
|------|----------|----------|---------|
| Daily Cleanup | 10:00 | 5:00 AM | `python app.py cleanup` |
| Morning Filings | 11:00 | 6:00 AM | `python app.py check_filings` |
| Daily Processing | 12:00 | 7:00 AM | `python app.py process` |
| Auto Send | 13:30 | 8:30 AM | `python app.py send` |
| Hourly Filings | 13-23,0-1 :30 | 8:30am-8:30pm | `python app.py check_filings` |
| Hourly Alerts | 14-23,0-3 :00 | 9am-10pm | `python app.py alerts` |
| CSV Backup | 4:59 | 11:59 PM | `python app.py export` |

**Problems:**
- Manual EST/EDT updates twice per year
- Different times for weekly vs daily not possible
- Schedule changes require Render dashboard edits

### After (1 Render Cron)
| Cron | Schedule | Command |
|------|----------|---------|
| Scheduler | `*/30 * * * *` | `python app.py scheduler` |

**Benefits:**
- Automatic EST/EDT handling
- Different schedules per day of week
- Database-configurable (future admin UI)
- Single cron to manage

---

## Code Locations

- **Schema**: `app.py` lines 2364-2421 (in `ensure_schema()`)
- **Functions**: `app.py` lines 30789-31087
- **CLI**: `app.py` lines 31906-31912

---

## Changelog

**November 26, 2025 - Initial Implementation**
- Created `schedule_config` and `schedule_hourly_config` tables
- Implemented `run_scheduler()` with timezone-aware logic
- Added `python app.py scheduler` CLI command
- Tested time window and hourly job logic
