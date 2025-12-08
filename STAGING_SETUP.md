# Staging Environment Setup Guide

This document outlines how to set up a staging environment for Weavara.

## Overview

```
PRODUCTION                          STAGING
──────────────────────────────────  ──────────────────────────────────
Render Web Service (main branch)    Render Web Service (staging branch)
Render PostgreSQL (quantbrief-db)   Render PostgreSQL (weavara-db-staging)
7 Cron jobs                         No crons (manual via /admin/cron)
Real beta users                     Test accounts (whitelisted emails)
```

## Cost Estimate

| Item | Monthly Cost |
|------|--------------|
| Staging PostgreSQL (Basic 1GB) | $7 |
| Staging Web Service (Starter) | $7 |
| Extra API usage (testing) | ~$5-10 |
| **Total** | **~$20-25/month** |

---

## Git Workflow

**Stay on `main` branch always.** Push to different remotes to control deployments:

```bash
# Deploy to STAGING (for testing)
git push origin main:staging

# Deploy to PRODUCTION (when ready)
git push origin main
```

**Why this approach:**
- Never switch branches (no confusion about which branch you're on)
- Production only gets code when you explicitly push to `main`
- Simple mental model: `main:staging` = test, `main` = ship

---

## Step 1: Create Staging Branch

```bash
# One-time setup: create staging branch
git checkout main
git pull origin main
git push origin main:staging
```

This creates the `staging` branch on GitHub from your current `main`.

---

## Step 2: Create Staging Database

1. Go to Render Dashboard → **New** → **PostgreSQL**
2. Configure:
   - **Name:** `weavara-db-staging`
   - **Database:** `weavara_staging`
   - **User:** `weavara_staging_user`
   - **Region:** Oregon (same as production)
   - **Plan:** Basic ($7/month, 1GB)
3. Click **Create Database**
4. Copy the **Internal Database URL** for Step 3

---

## Step 3: Create Staging Web Service

1. Go to Render Dashboard → **New** → **Web Service**
2. Connect to GitHub repo: `iverpak/weavara-daily`
3. Configure:
   - **Name:** `weavara-staging`
   - **Branch:** `staging` (NOT main)
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:APP --host 0.0.0.0 --port $PORT`
   - **Plan:** Starter ($7/month)
   - **Auto-Deploy:** Yes (deploys when staging branch updated)

---

## Step 4: Environment Variables for Staging

Copy from production, with these modifications:

### Core Settings (MODIFY)

| Variable | Value | Notes |
|----------|-------|-------|
| `DATABASE_URL` | `postgresql://weavara_staging_user:...` | **Use staging DB URL** |
| `STAGING_MODE` | `true` | **NEW - enables email whitelist** |
| `MAX_CONCURRENT_JOBS` | `2` | Lower for staging |

### API Keys (SAME AS PRODUCTION)

| Variable | Notes |
|----------|-------|
| `ADMIN_EMAIL` | Same (weavara.research@gmail.com) |
| `ADMIN_TOKEN` | Same |
| `ANTHROPIC_API_KEY` | Same |
| `OPENAI_API_KEY` | Same |
| `GEMINI_API_KEY` | Same |
| `FMP_API_KEY` | Same |
| `POLYGON_API_KEY` | Same |
| `SCRAPFLY_API_KEY` | Same |

### Email Settings (SAME AS PRODUCTION)

| Variable | Notes |
|----------|-------|
| `MAILGUN_API_KEY` | Same |
| `MAILGUN_DOMAIN` | Same |
| `MAILGUN_FROM` | Same |
| `SMTP_*` | Same |

### Other Settings

| Variable | Value |
|----------|-------|
| `TZ_DEFAULT` | `America/Toronto` |
| `PYTHON_VERSION` | `3.12.5` |

---

## Step 5: Code Changes Required

### 5.1 Email Whitelist Safety

Prevents accidental emails to real users in staging.

**Whitelisted emails:**
- `weavara.research@gmail.com`
- `ilia.verpakhovski@gmail.com`
- `timelesstalesvisualized@gmail.com`
- `stockdigest.research@gmail.com`

**Implementation:** Add to `app.py`:

```python
# Staging mode - only allow emails to whitelisted addresses
STAGING_MODE = os.getenv("STAGING_MODE", "false").lower() == "true"
STAGING_ALLOWED_EMAILS = [
    "weavara.research@gmail.com",
    "ilia.verpakhovski@gmail.com",
    "timelesstalesvisualized@gmail.com",
    "stockdigest.research@gmail.com",
]

def is_email_allowed_in_staging(email: str) -> bool:
    """Check if email is allowed in staging mode."""
    if not STAGING_MODE:
        return True  # Production - allow all
    return email.lower() in [e.lower() for e in STAGING_ALLOWED_EMAILS]
```

**In `send_email()` function:**

```python
# STAGING SAFETY: Block emails to non-whitelisted addresses
if STAGING_MODE and not is_email_allowed_in_staging(recipient):
    LOG.warning(f"⚠️ STAGING MODE: Blocked email to {recipient} (not in whitelist)")
    return False
```

### 5.2 Staging Banner in Admin UI

Red banner on all admin pages when in staging mode.

**Add to each admin template after `<body>`:**

```html
{% if staging_mode %}
<div style="background: #ff6b6b; color: white; padding: 8px; text-align: center; font-weight: bold;">
    ⚠️ STAGING ENVIRONMENT - Not Production
</div>
{% endif %}
```

**Pass to templates in route handlers:**

```python
return templates.TemplateResponse("admin_queue.html", {
    "request": request,
    "staging_mode": STAGING_MODE,
    # ... other context ...
})
```

### 5.3 Cron Jobs Admin Page (NEW)

New `/admin/cron` page for manually triggering cron functions in staging.

**Layout:** 3x3 grid of cards

| 🗑️ Cleanup | ⚙️ Process | 📧 Send |
|------------|------------|---------|
| Delete old queue entries | Generate reports for all users | Send all ready emails |

| 🔍 Check Filings | 📰 Hourly Alerts | 💾 Export Users |
|------------------|------------------|-----------------|
| Check for new SEC filings | Ingest articles + admin alert | Backup users to CSV |

| 🕐 Scheduler | | |
|--------------|---|---|
| Run master scheduler | | |

**API Endpoints:**
- `POST /api/cron/cleanup`
- `POST /api/cron/process`
- `POST /api/cron/send`
- `POST /api/cron/check-filings`
- `POST /api/cron/alerts`
- `POST /api/cron/export`
- `POST /api/cron/scheduler`

**Dashboard:** Add card linking to `/admin/cron`:
> 🕐 **Cron Jobs** - Manually run scheduled tasks

---

## Step 6: Initialize Staging Database

After staging web service deploys:

1. App automatically runs `ensure_schema()` on startup
2. All tables created

**Add test users via Admin API:**

```bash
curl -X POST "https://weavara-staging.onrender.com/api/beta-signup" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test User 1",
    "email": "weavara.research@gmail.com",
    "tickers": ["AAPL", "NVDA", "TSLA"]
  }'
```

Or copy existing admin/test users from production database.

---

## Step 7: Daily Development Workflow

```bash
# 1. Make changes in Codespaces (stay on main branch)
git add . && git commit -m "Your changes"

# 2. Deploy to staging for testing
git push origin main:staging

# 3. Test on staging
#    - URL: https://weavara-staging.onrender.com
#    - Admin: https://weavara-staging.onrender.com/admin?token=XXX
#    - Look for red "STAGING" banner

# 4. When verified, deploy to production
git push origin main
```

### Manual Testing via /admin/cron

Staging has no cron jobs. Use the Cron Jobs admin page:

1. Go to `https://weavara-staging.onrender.com/admin/cron?token=XXX`
2. Click buttons to manually trigger:
   - Cleanup, Process, Send (daily workflow)
   - Check Filings (SEC monitoring)
   - Hourly Alerts (article ingestion)
   - Export Users (backup)
3. Check Render logs for output

---

## Database Migrations

### What `ensure_schema()` handles automatically:
- New tables (`CREATE TABLE IF NOT EXISTS`)
- New columns (`ALTER TABLE ADD COLUMN IF NOT EXISTS`)

### What requires manual migration:
- Dropping columns
- Renaming columns
- Changing column types
- Data migrations

### For staging:
Staging DB is disposable. For major schema changes, just drop and recreate:
1. Delete `weavara-db-staging` in Render
2. Create new database
3. Update `DATABASE_URL` env var
4. Redeploy (schema auto-creates)

### For production:
Requires careful planning:
1. Test migration on staging first
2. Run during low-traffic time
3. Consider multi-deploy approach (add new column → migrate data → drop old)

---

## Rollback Strategy

If you push something broken to production:

**Option 1: Render Rollback**
- Go to Render → Web Service → Deploys
- Click "Rollback" on previous working deploy

**Option 2: Git Revert**
```bash
git revert HEAD
git push origin main
```

---

## Checklist

### One-Time Setup
- [ ] Create `staging` branch: `git push origin main:staging`
- [ ] Create Render PostgreSQL (`weavara-db-staging`)
- [ ] Create Render Web Service (`weavara-staging`) on `staging` branch
- [ ] Copy environment variables, set `DATABASE_URL` and `STAGING_MODE=true`
- [ ] Add email whitelist code to `app.py`
- [ ] Add staging banner to admin templates
- [ ] Create `/admin/cron` page and API endpoints
- [ ] Add Cron Jobs card to admin dashboard
- [ ] Add test users to staging database
- [ ] Test full flow: process → send → verify email received

### Per-Feature Testing
- [ ] Push to staging: `git push origin main:staging`
- [ ] Verify staging auto-deploys (check Render)
- [ ] Test feature via admin UI or `/admin/cron`
- [ ] Verify emails only go to whitelisted addresses
- [ ] When verified, push to production: `git push origin main`

---

## Troubleshooting

### "Table does not exist" errors
- Wait for web service to fully deploy (runs `ensure_schema()` on startup)
- Check Render logs for schema initialization

### Emails not sending in staging
- Verify `STAGING_MODE=true` is set
- Check if recipient is in `STAGING_ALLOWED_EMAILS`
- Check Render logs for "STAGING MODE: Blocked email" warnings

### Can't tell which environment I'm in
- Look for red "STAGING ENVIRONMENT" banner in admin UI
- Check URL: `weavara-staging.onrender.com` vs `weavara.io`

### Schema out of sync
- Staging: Drop and recreate database
- Production: Write manual migration SQL, test on staging first

---

## URLs

| Environment | URL |
|-------------|-----|
| Production | `https://weavara.io` |
| Staging | `https://weavara-staging.onrender.com` |
| Production Admin | `https://weavara.io/admin?token=XXX` |
| Staging Admin | `https://weavara-staging.onrender.com/admin?token=XXX` |
| Staging Cron Runner | `https://weavara-staging.onrender.com/admin/cron?token=XXX` |
