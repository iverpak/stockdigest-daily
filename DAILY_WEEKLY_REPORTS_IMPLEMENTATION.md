# Daily vs Weekly Reports Implementation Tracker

**Date Started:** November 20, 2025
**Last Updated:** November 20, 2025
**Status:** 40% Complete (Core Infrastructure Done)

---

## 📋 Overview

Implementing Monday weekly reports (7 days) vs Tuesday-Sunday daily reports (1 day). Users still receive same content quality, but daily reports hide 4 sections for cleaner UX.

### Key Changes
- **Monday:** "WEAVARA WEEKLY INTELLIGENCE" - 7 days lookback, all 6 sections
- **Tuesday-Sunday:** "WEAVARA DAILY BRIEF" - 1 day lookback, hide 4 sections

---

## ✅ COMPLETED (Commit: cdbdf34)

### 1. Database Schema Changes ✅
**File:** `app.py` (lines 2243-2264 in `ensure_schema()`)

```sql
-- Added to system_config
INSERT INTO system_config (key, value)
VALUES ('daily_lookback_minutes', '1440') ON CONFLICT DO NOTHING;

INSERT INTO system_config (key, value)
VALUES ('weekly_lookback_minutes', '10080') ON CONFLICT DO NOTHING;

-- Added to email_queue table
ALTER TABLE email_queue ADD COLUMN report_type VARCHAR(10) DEFAULT 'daily';
ALTER TABLE email_queue ADD COLUMN summary_date DATE;
CREATE INDEX idx_email_queue_report_type ON email_queue(report_type);
```

**Status:** ✅ Will apply on next deployment

---

### 2. Helper Functions Created ✅
**File:** `app.py` (lines 23406-23492)

**Functions:**
1. `get_daily_lookback_minutes()` → Returns 1440 (1 day)
2. `get_weekly_lookback_minutes()` → Returns 10080 (7 days)
3. `get_report_type_and_lookback(force_type=None)` → Returns `(report_type, minutes)`
   - Monday = `('weekly', 10080)`
   - Tuesday-Sunday = `('daily', 1440)`
   - `force_type` parameter for testing

**Error Handling:** Defaults to `('daily', 1440)` on failure

---

### 3. Production Workflow Updated ✅
**File:** `app.py` (line 30049 - `process_daily_workflow()`)

**Changes:**
```python
# OLD:
"minutes": get_lookback_minutes()  # Always 7 days

# NEW:
report_type, lookback_minutes = get_report_type_and_lookback()
config = {
    "minutes": lookback_minutes,
    "report_type": report_type,  # NEW: 'daily' or 'weekly'
    "mode": "daily",
    ...
}
```

**Logs Added:**
- `📅 Report Type: DAILY` or `WEEKLY`
- `⏱️ Lookback: 1440 minutes (1.0 days)` or `10080 minutes (7.0 days)`

---

### 4. Bulk Operation Endpoints Updated ✅
**Files Updated:**

**A. `/api/generate-user-reports` (line 28838)**
- Added `get_report_type_and_lookback()` call
- Passes `report_type` in job config

**B. `/api/generate-all-reports` (line 28922)**
- Added `get_report_type_and_lookback()` call
- Passes `report_type` in job config

**Both endpoints now:**
- Determine report type based on current day of week
- Log report type and lookback minutes
- Pass `report_type` in both batch and job configs

---

## 🚧 IN PROGRESS

### 5. Job Worker & Email Generation (NEXT - CRITICAL)

**A. Extract report_type in Job Worker**
**File:** `app.py` (around line 17415 in `process_digest_phase()`)

**Need to add:**
```python
config = job.get('config', {})
report_type = config.get('report_type', 'daily')  # Default to daily
```

**B. Pass report_type to Email Generation**
**File:** `app.py` (line 17550 in Email #3 generation)

**Change:**
```python
# OLD:
email3_data = generate_email_html_core(
    ticker=ticker,
    hours=int(minutes/60),
    recipient_email=None
)

# NEW:
email3_data = generate_email_html_core(
    ticker=ticker,
    hours=int(minutes/60),
    recipient_email=None,
    report_type=report_type  # NEW PARAMETER
)
```

---

### 6. Update generate_email_html_core() (CRITICAL)
**File:** `app.py` (line 14256)

**Changes Needed:**

**A. Add Parameter:**
```python
def generate_email_html_core(
    ticker: str,
    hours: int = 24,
    recipient_email: str = None,
    report_type: str = 'daily'  # NEW
) -> Dict[str, any]:
```

**B. Section Filtering:**
```python
# Parse all 6 sections (AI always generates them)
sections = convert_phase1_to_sections_dict(summary_json)

# Filter for daily reports
if report_type == 'daily':
    sections.pop('upside_scenario', None)
    sections.pop('downside_scenario', None)
    sections.pop('key_variables', None)
    sections.pop('upcoming_catalysts', None)
```

**C. Header Branding:**
```python
if report_type == 'weekly':
    header_title = "WEAVARA WEEKLY INTELLIGENCE"
    header_subtitle = "Weekly Intelligence Delivered Every Monday"
    subject_prefix = "📊 Weekly Intelligence"
else:  # daily
    header_title = "WEAVARA DAILY BRIEF"
    header_subtitle = "Daily Brief Delivered Tuesday-Sunday"
    subject_prefix = "📊 Daily Brief"
```

**D. Update HTML Template:**
- Line ~14514: Replace "Stock Intelligence Delivered Daily" with dynamic subtitle
- Line ~14552: Update subject line to use `subject_prefix`

---

### 7. Email Queue Storage Update
**File:** `app.py` (line 17564 - INSERT INTO email_queue)

**Change:**
```python
# OLD:
INSERT INTO email_queue (
    ticker, company_name, recipients, email_html, email_subject,
    article_count, status, is_production, heartbeat, created_at
)

# NEW:
INSERT INTO email_queue (
    ticker, company_name, recipients, email_html, email_subject,
    article_count, status, is_production, report_type, summary_date,
    heartbeat, created_at
)
VALUES (..., %s, CURRENT_DATE, NOW(), NOW())

# Add to VALUES tuple:
..., report_type, ...
```

**Also Update ON CONFLICT:**
```python
ON CONFLICT (ticker) DO UPDATE
SET ...,
    report_type = EXCLUDED.report_type,
    summary_date = EXCLUDED.summary_date,
    ...
```

---

### 8. Regenerate Email #3 Endpoint
**File:** `app.py` (line 27870 - `/api/regenerate-email`)

**Changes Needed:**

**A. Fetch report_type from email_queue:**
```python
# Around line 27908, after fetching article_ids:
cur.execute("""
    SELECT article_ids, report_type, summary_date
    FROM email_queue
    WHERE ticker = %s
    ORDER BY created_at DESC LIMIT 1
""", (ticker,))

row = cur.fetchone()
report_type = row['report_type'] if row and row['report_type'] else 'daily'
```

**B. Pass to email generation (around line 28155):**
```python
email3_data = generate_email_html_core(
    ticker=ticker,
    hours=int(minutes/60),
    recipient_email=None,
    report_type=report_type  # Use stored report_type
)
```

---

### 9. Email #1 and #2 Subject Updates

**A. Email #1 Subject (line 12823)**
```python
# OLD:
subject = f"🔍 Article Selection QA: {ticker_list} - {total_flagged} flagged..."

# NEW:
report_type = config.get('report_type', 'daily')
report_label = f"({report_type.upper()})"
subject = f"🔍 Article Selection QA {report_label}: {ticker_list} - {total_flagged} flagged..."
```

**Note:** Need to pass `config` parameter to email #1 function

**B. Email #2 Subject (line 13439)**
```python
# OLD:
subject = f"📝 Content QA: {ticker_list} - {total_articles} articles..."

# NEW:
report_type = config.get('report_type', 'daily')
report_label = f"({report_type.upper()})"
subject = f"📝 Content QA {report_label}: {ticker_list} - {total_articles} articles..."
```

**Note:** Need to pass `config` parameter to email #2 function

---

## 🔲 TODO (Not Started)

### 10. Admin Queue Page - Report Type Badge
**File:** `templates/admin_queue.html`

**Add badge in queue display:**
```html
<!-- Near ticker display -->
<span class="badge badge-daily" v-if="item.report_type === 'daily'">DAILY</span>
<span class="badge badge-weekly" v-if="item.report_type === 'weekly'">WEEKLY</span>
```

**CSS Styling:**
```css
.badge-daily {
    background-color: #0066cc;
    color: white;
    padding: 2px 6px;
    border-radius: 3px;
}
.badge-weekly {
    background-color: #6600cc;
    color: white;
    padding: 2px 6px;
    border-radius: 3px;
}
```

**Backend:** Ensure `/api/admin/queue` endpoint returns `report_type` field

---

### 11. Admin Settings UI - Two Lookback Windows
**File:** `templates/admin_settings.html`

**Add two input sections:**
```html
<!-- Daily Lookback -->
<div class="setting-item">
    <label>Daily Lookback Window (Tuesday-Sunday)</label>
    <input id="dailyLookbackMinutes" type="number" min="60" max="10080" value="1440">
    <span>minutes (1 day)</span>
    <button onclick="saveDailyLookback()">Save Daily Lookback</button>
</div>

<!-- Weekly Lookback -->
<div class="setting-item">
    <label>Weekly Lookback Window (Monday)</label>
    <input id="weeklyLookbackMinutes" type="number" min="60" max="10080" value="10080">
    <span>minutes (7 days)</span>
    <button onclick="saveWeeklyLookback()">Save Weekly Lookback</button>
</div>
```

**JavaScript Functions:**
```javascript
async function loadLookbackSettings() {
    // Fetch both daily and weekly
    const dailyResp = await fetch('/api/get-daily-lookback-window?token=' + TOKEN);
    const weeklyResp = await fetch('/api/get-weekly-lookback-window?token=' + TOKEN);
    // Populate inputs
}

async function saveDailyLookback() {
    const minutes = document.getElementById('dailyLookbackMinutes').value;
    await fetch('/api/set-daily-lookback-window?token=' + TOKEN + '&minutes=' + minutes, {method: 'POST'});
}

async function saveWeeklyLookback() {
    const minutes = document.getElementById('weeklyLookbackMinutes').value;
    await fetch('/api/set-weekly-lookback-window?token=' + TOKEN + '&minutes=' + minutes, {method: 'POST'});
}
```

---

### 12. Admin API Endpoints (4 New)
**File:** `app.py` (add after line 29233)

**A. Get Daily Lookback:**
```python
@APP.get("/api/get-daily-lookback-window")
async def get_daily_lookback_window_api(token: str = Query(...)):
    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    try:
        minutes = get_daily_lookback_minutes()
        hours = minutes / 60
        days = int(hours / 24) if hours >= 24 else 0
        label = f"{days} days" if days > 0 else f"{int(hours)} hours"

        return {"status": "success", "minutes": minutes, "label": label}
    except Exception as e:
        LOG.error(f"Failed to get daily lookback: {e}")
        return {"status": "error", "message": str(e)}
```

**B. Set Daily Lookback:**
```python
@APP.post("/api/set-daily-lookback-window")
async def set_daily_lookback_window_api(token: str = Query(...), minutes: int = Query(...)):
    if not check_admin_token(token):
        return {"status": "error", "message": "Unauthorized"}

    if minutes < 60 or minutes > 10080:
        return {"status": "error", "message": "Lookback must be 60-10080 minutes"}

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE system_config
                SET value = %s, updated_at = NOW()
                WHERE key = 'daily_lookback_minutes'
            """, (str(minutes),))
            conn.commit()

        return {"status": "success", "minutes": minutes}
    except Exception as e:
        LOG.error(f"Failed to set daily lookback: {e}")
        return {"status": "error", "message": str(e)}
```

**C. Get Weekly Lookback:** (Same pattern as daily)

**D. Set Weekly Lookback:** (Same pattern as daily)

---

### 13. Test Runner Dropdown
**File:** `templates/admin_test.html`

**Add Dropdown:**
```html
<div class="form-group">
    <label>Report Type (for testing):</label>
    <select id="reportType">
        <option value="daily">Daily Brief (1 day lookback)</option>
        <option value="weekly">Weekly Intelligence (7 day lookback)</option>
    </select>
</div>
```

**JavaScript Update:**
```javascript
// In job submission function
const reportType = document.getElementById('reportType').value;

// Add to job config
config.report_type = reportType;
```

**Backend:** Test runner endpoint needs to accept and use `report_type` override:
```python
report_type = request_data.get('report_type', None)
report_type, lookback_minutes = get_report_type_and_lookback(force_type=report_type)
```

---

### 14. Branding Updates (StockDigest → Weavara)

**Files to Update:**

**A. Email #3 Template (app.py line 14256+)**
- Line ~14513: "StockDigest" → "Weavara"
- Line ~14514: "Stock Intelligence Delivered Daily" → Dynamic based on report_type
- Header and footer branding

**B. Hourly Alerts Template**
- File: `templates/email_hourly_alert.html`
- Search for "StockDigest", replace with "Weavara"

**C. Research Emails Template**
- File: `templates/email_research_report.html`
- Search for "StockDigest", replace with "Weavara"

**D. Email #1 and #2 (if applicable)**
- Check for any StockDigest branding in internal QA emails

---

## 🧪 Testing Checklist

### Manual Testing Required:

- [ ] **Monday Test (Weekly):**
  - [ ] Run `python app.py process` on Monday
  - [ ] Verify lookback = 10080 minutes in logs
  - [ ] Verify Email #3 subject: "📊 Weekly Intelligence: ..."
  - [ ] Verify Email #3 header: "WEAVARA WEEKLY INTELLIGENCE"
  - [ ] Verify all 6 sections shown in email
  - [ ] Verify `email_queue.report_type = 'weekly'`

- [ ] **Tuesday-Sunday Test (Daily):**
  - [ ] Run on any non-Monday
  - [ ] Verify lookback = 1440 minutes in logs
  - [ ] Verify Email #3 subject: "📊 Daily Brief: ..."
  - [ ] Verify Email #3 header: "WEAVARA DAILY BRIEF"
  - [ ] Verify 4 sections hidden (upside, downside, key vars, catalysts)
  - [ ] Verify `email_queue.report_type = 'daily'`

- [ ] **Test Runner Override:**
  - [ ] Select "Weekly" dropdown on Tuesday
  - [ ] Verify weekly report generated despite being Tuesday
  - [ ] Select "Daily" dropdown on any day
  - [ ] Verify daily report generated

- [ ] **Regenerate Email #3:**
  - [ ] Regenerate a daily report
  - [ ] Verify sections remain hidden
  - [ ] Regenerate a weekly report
  - [ ] Verify all sections still shown

- [ ] **Admin Settings:**
  - [ ] Change daily lookback to 2880 (2 days)
  - [ ] Verify next Tuesday-Sunday run uses 2 days
  - [ ] Change weekly lookback to 14400 (10 days)
  - [ ] Verify next Monday run uses 10 days

---

## 📊 Implementation Progress

**Overall:** 40% Complete

| Component | Status | Files Changed |
|-----------|--------|---------------|
| Database Schema | ✅ Done | app.py (ensure_schema) |
| Helper Functions | ✅ Done | app.py (3 functions) |
| Production Workflow | ✅ Done | app.py (process_daily_workflow) |
| Bulk Endpoints (2/3) | ✅ Done | app.py (generate-user-reports, generate-all-reports) |
| Job Worker Updates | 🚧 Next | app.py (process_digest_phase) |
| Email Generation | 🚧 Next | app.py (generate_email_html_core) |
| Email Queue Storage | 🔲 Todo | app.py (INSERT statement) |
| Regenerate Email | 🔲 Todo | app.py (/api/regenerate-email) |
| Email #1/#2 Subjects | 🔲 Todo | app.py (2 locations) |
| Admin Queue Badge | 🔲 Todo | templates/admin_queue.html |
| Admin Settings UI | 🔲 Todo | templates/admin_settings.html |
| Admin API Endpoints | 🔲 Todo | app.py (4 new endpoints) |
| Test Runner Dropdown | 🔲 Todo | templates/admin_test.html |
| Branding Updates | 🔲 Todo | 3+ templates |

---

## 🔧 Quick Reference

**Day of Week Detection:**
```python
import pytz
eastern = pytz.timezone('America/Toronto')
now = datetime.now(eastern)
day_of_week = now.weekday()  # 0=Monday, 6=Sunday
```

**Config Structure:**
```python
job_config = {
    "minutes": 1440 or 10080,
    "report_type": "daily" or "weekly",
    "mode": "daily",  # For email queue (always 'daily')
    "recipients": [...],
    ...
}
```

**Sections to Hide in Daily:**
- `upside_scenario`
- `downside_scenario`
- `key_variables`
- `upcoming_catalysts`

---

## 🚀 Deployment Notes

1. Schema changes apply automatically on next app restart (via `ensure_schema()`)
2. Existing `email_queue` rows get `report_type='daily'` default (safe)
3. No backward compatibility issues (defaults to 'daily' everywhere)
4. Can test immediately after deployment using Test Runner

---

**Last Updated:** November 20, 2025
**Next Session:** Start with Job Worker updates (Section 5)
