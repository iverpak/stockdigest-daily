# Staging Environment Setup Guide

This document outlines how to set up a staging environment for Weavara.

## Overview

```
PRODUCTION                          STAGING
──────────────────────────────────  ──────────────────────────────────
Render Web Service (main branch)    Render Web Service (staging branch)
Render PostgreSQL (Weavara-db)      Render PostgreSQL (Weavara-db-staging)
7 Cron jobs                         No crons (manual testing only)
Real beta users                     Test accounts (your emails only)
```

## Cost Estimate

| Item | Monthly Cost |
|------|--------------|
| Staging PostgreSQL (Basic 1GB) | $7 |
| Staging Web Service (Starter) | $7 |
| Extra API usage (testing) | ~$5-10 |
| **Total** | **~$20-25/month** |

---

## Step 1: Create Staging Branch

Run these commands locally (or in Codespaces):

```bash
# Create and push staging branch
git checkout main
git pull origin main
git checkout -b staging
git push -u origin staging
```

**Git Workflow:**
```
feature-branch → staging (test) → main (production)
```

---

## Step 2: Create Staging Database

1. Go to Render Dashboard → **New** → **PostgreSQL**
2. Configure:
   - **Name:** `Weavara-db-staging`
   - **Database:** `weavara_staging`
   - **User:** `weavara_staging_user`
   - **Region:** Oregon (same as production)
   - **Plan:** Basic ($7/month, 1GB)
3. Click **Create Database**
4. Copy the **Internal Database URL** for Step 3

---

## Step 3: Create Staging Web Service

1. Go to Render Dashboard → **New** → **Web Service**
2. Connect to the same GitHub repo: `iverpak/stockdigest-daily`
3. Configure:
   - **Name:** `Weavara-staging`
   - **Branch:** `staging` (NOT main)
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:APP --host 0.0.0.0 --port $PORT`
   - **Plan:** Starter ($7/month)
   - **Auto-Deploy:** Yes

---

## Step 4: Environment Variables for Staging

Copy these from production, with modifications noted:

### Core Settings (MODIFY)

| Variable | Value | Notes |
|----------|-------|-------|
| `DATABASE_URL` | `postgresql://weavara_staging_user:...` | **Use staging DB URL** |
| `STAGING_MODE` | `true` | **NEW - enables email whitelist** |
| `MAX_CONCURRENT_JOBS` | `2` | Lower for staging |

### API Keys (SAME AS PRODUCTION)

| Variable | Notes |
|----------|-------|
| `ADMIN_EMAIL` | Same |
| `ADMIN_TOKEN` | Same (or different for extra safety) |
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
| `SMTP_HOST` | Same |
| `SMTP_PORT` | Same |
| `SMTP_USERNAME` | Same |
| `SMTP_PASSWORD` | Same |
| `SMTP_STARTTLS` | Same |

### GitHub Settings (SAME AS PRODUCTION)

| Variable | Notes |
|----------|-------|
| `GITHUB_REPO` | Same |
| `GITHUB_TOKEN` | Same |
| `GITHUB_CSV_PATH` | Same |

### Other Settings

| Variable | Value |
|----------|-------|
| `TZ_DEFAULT` | `America/Toronto` |
| `PYTHON_VERSION` | `3.12.5` |
| `DRY_RUN` | `false` (we want real emails to test accounts) |

---

## Step 5: Code Changes Required

### 5.1 Add Staging Mode Email Whitelist

Add this safety check to prevent accidental emails to real users.

**Location:** `app.py` (near other config constants, around line 845)

```python
# Staging mode - only allow emails to whitelisted addresses
STAGING_MODE = os.getenv("STAGING_MODE", "false").lower() == "true"
STAGING_ALLOWED_EMAILS = [
    "weavara.research@gmail.com",
    "weavara.research+test1@gmail.com",
    "weavara.research+test2@gmail.com",
    "weavara.research+test3@gmail.com",
    # Add more test emails as needed (Gmail + trick goes to same inbox)
]

def is_email_allowed_in_staging(email: str) -> bool:
    """Check if email is allowed in staging mode."""
    if not STAGING_MODE:
        return True  # Production - allow all
    return email.lower() in [e.lower() for e in STAGING_ALLOWED_EMAILS]
```

**Location:** `send_email()` function (around line 13215)

```python
def send_email(subject: str, html_body: str, to: str | None = None, bcc: str | None = None):
    # ... existing validation ...

    recipient = to or ADMIN_EMAIL

    # STAGING SAFETY: Block emails to non-whitelisted addresses
    if STAGING_MODE and not is_email_allowed_in_staging(recipient):
        LOG.warning(f"⚠️ STAGING MODE: Blocked email to {recipient} (not in whitelist)")
        return False

    # ... rest of function ...
```

### 5.2 Add Staging Indicator to Admin UI

Add a visual banner so you know which environment you're in.

**Location:** Each admin template (e.g., `templates/admin_queue.html`, `templates/admin_users.html`)

Add after `<body>` tag:

```html
{% if staging_mode %}
<div style="background: #ff6b6b; color: white; padding: 8px; text-align: center; font-weight: bold;">
    ⚠️ STAGING ENVIRONMENT - Not Production
</div>
{% endif %}
```

**Location:** Admin route handlers in `app.py`

Pass `staging_mode` to templates:

```python
return templates.TemplateResponse("admin_queue.html", {
    "request": request,
    "staging_mode": STAGING_MODE,
    # ... other context ...
})
```

---

## Step 6: Initialize Staging Database

After the staging web service deploys:

1. The app will automatically run `ensure_schema()` on startup
2. All tables will be created

To add test users, either:

**Option A: Via Admin API**
```bash
curl -X POST "https://weavara-staging.onrender.com/api/beta-signup" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test User 1",
    "email": "weavara.research+test1@gmail.com",
    "tickers": ["AAPL", "NVDA", "TSLA"]
  }'
```

**Option B: Direct SQL**
```sql
INSERT INTO users (name, email, status)
VALUES ('Test User 1', 'weavara.research+test1@gmail.com', 'active');

INSERT INTO user_tickers (user_id, ticker)
SELECT id, unnest(ARRAY['AAPL', 'NVDA', 'TSLA'])
FROM users WHERE email = 'weavara.research+test1@gmail.com';
```

---

## Step 7: Testing Workflow

### Daily Development Workflow

1. **Make changes** in Codespaces (on `main` branch)
2. **Push to staging** to test:
   ```bash
   git checkout staging
   git merge main
   git push
   ```
3. **Test on staging** via `https://weavara-staging.onrender.com`
4. **When verified**, push to production:
   ```bash
   git checkout main
   git push origin main
   ```

### Manual Testing (No Crons)

Staging has no cron jobs. Use admin endpoints to test:

```bash
# Test daily workflow
curl -X POST "https://weavara-staging.onrender.com/api/generate-all-reports" \
  -H "X-Admin-Token: YOUR_TOKEN"

# Test email sending
curl -X POST "https://weavara-staging.onrender.com/api/send-all-ready" \
  -H "X-Admin-Token: YOUR_TOKEN"

# Test filings check
curl -X POST "https://weavara-staging.onrender.com/api/check-filings" \
  -H "X-Admin-Token: YOUR_TOKEN"
```

Or use the Admin Test page: `https://weavara-staging.onrender.com/admin/test`

---

## Step 8: Weekly Sync (Optional)

If you add new database columns/tables, staging will auto-sync on next deploy (via `ensure_schema()`).

For major schema changes, you may need to:

1. Drop and recreate staging database, OR
2. Run manual migration SQL

**Recommended:** Do schema sync on weekends when production doesn't run reports.

---

## Checklist

### One-Time Setup
- [ ] Create `staging` branch in git
- [ ] Create Render PostgreSQL (`Weavara-db-staging`)
- [ ] Create Render Web Service (`Weavara-staging`) pointing to `staging` branch
- [ ] Copy environment variables (modify `DATABASE_URL`, add `STAGING_MODE=true`)
- [ ] Add email whitelist safety code
- [ ] Add staging indicator to admin UI
- [ ] Add 3-5 test users in staging database
- [ ] Test full flow (process → send → verify email received)

### Per-Feature Testing
- [ ] Push changes to `staging` branch
- [ ] Verify staging auto-deploys
- [ ] Test feature via admin endpoints or UI
- [ ] Verify emails only go to whitelisted addresses
- [ ] When verified, merge to `main` for production

---

## Troubleshooting

### "Table does not exist" errors
- Wait for web service to fully deploy (runs `ensure_schema()` on startup)
- Check Render logs for schema initialization

### Emails not sending
- Verify `STAGING_MODE=true` is set
- Check if recipient is in `STAGING_ALLOWED_EMAILS`
- Check Render logs for "STAGING MODE: Blocked email" warnings

### Can't tell which environment I'm in
- Look for red "STAGING ENVIRONMENT" banner in admin UI
- Check URL: `weavara-staging.onrender.com` vs `weavara-daily.onrender.com`

### Schema out of sync
- Staging runs same `ensure_schema()` as production
- New columns/tables created automatically on deploy
- For major changes, may need to recreate staging DB

---

## URLs

| Environment | URL |
|-------------|-----|
| Production | `https://weavara-daily.onrender.com` |
| Staging | `https://weavara-staging.onrender.com` |
| Production Admin | `https://weavara-daily.onrender.com/admin?token=XXX` |
| Staging Admin | `https://weavara-staging.onrender.com/admin?token=XXX` |
