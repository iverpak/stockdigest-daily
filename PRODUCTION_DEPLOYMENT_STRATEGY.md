# Production Deployment & Continuous Improvement Strategy

**Project:** StockDigest (stockdigest.app)
**Created:** November 2025
**Purpose:** Pre-beta launch deployment strategy, staging setup, and production safety guidelines

---

## Table of Contents

1. [Current System Assessment](#current-system-assessment)
2. [Staging Environment Strategy](#staging-environment-strategy)
3. [Feature Flags vs Staging](#feature-flags-vs-staging)
4. [Database Migration Strategy](#database-migration-strategy)
5. [Ad-hoc Report Concurrency](#ad-hoc-report-concurrency)
6. [Infrastructure Scaling Plan](#infrastructure-scaling-plan)
7. [Processing Schedule (2 AM vs 7 AM)](#processing-schedule-2-am-vs-7-am)
8. [Health Checks & Error Handling](#health-checks-error-handling)
9. [Protecting Dangerous Endpoints](#protecting-dangerous-endpoints)
10. [Pre-Beta Launch Checklist](#pre-beta-launch-checklist)

---

## Current System Assessment

### What's Working Well (95% Safe for Scheduled Batch)

✅ **Database job claiming** - Atomic `FOR UPDATE SKIP LOCKED` prevents race conditions
✅ **Connection pooling** - 5-80 connections, handles 4 concurrent tickers safely
✅ **Deadlock retry** - Automatic retry prevents article loss during concurrent processing
✅ **Heartbeat monitoring** - Detects stuck jobs via reclaim thread
✅ **Job queue system** - Eliminates HTTP 520 timeouts
✅ **Thread isolation** - Each ticker runs in isolated thread with own event loop

### What Needs Improvement (70% Safe for Ad-hoc)

⚠️ **No concurrency limits** - 8+ concurrent tickers = memory crash (2GB limit)
⚠️ **No user credit limits** - Abuse risk, cost explosion potential
⚠️ **No priority system** - Ad-hoc can interfere with scheduled morning batch
⚠️ **No staging environment** - All changes go directly to production
⚠️ **No email rate limiting** - SMTP spam filter risk
⚠️ **Direct auto-deploy** - Syntax errors crash entire app

---

## Staging Environment Strategy

### Setup Overview

**Two-Service Architecture:**
```
Production: stockdigest-production (Manual deploy only)
Staging:    stockdigest-staging (Auto-deploy from 'staging' branch)
```

### Workflow

```
1. Feature branch → staging branch (merge)
2. Auto-deploys to staging.stockdigest.app
3. Test thoroughly (run 10-ticker batch, verify all 3 emails)
4. Merge staging → main
5. Manual deploy to production (Render dashboard button)
6. Monitor logs for 30 min
7. If broken: Instant rollback (Render dashboard button)
```

### Staging Environment Setup (1-2 Hours)

**Step 1: Clone Production Service**
- Render Dashboard → stockdigest-production → "Duplicate"
- Name: `stockdigest-staging`
- Branch: `staging` (create new git branch)
- Auto-deploy: ✅ Enabled

**Step 2: Create Staging Database**
- Use Render's **Free PostgreSQL** tier (1GB, $0/month)
- Copy production schema (NOT data)
- Link to staging app via `DATABASE_URL` env var

**Step 3: Configure Environment Variables**

```bash
# Staging-specific env vars
ENVIRONMENT=staging
DRY_RUN=true  # CRITICAL: Redirects all emails to admin
ADMIN_EMAIL=you+staging@example.com
DATABASE_URL=<staging_db_url>
MAX_CONCURRENT_JOBS=2  # Lower than production (save resources)

# Copy from production
CLAUDE_API_KEY=<same>
OPENAI_API_KEY=<same>
SCRAPFLY_API_KEY=<same>
FMP_API_KEY=<same>
GEMINI_API_KEY=<same>
SMTP_*=<same>
```

**Step 4: Disable Cron Jobs in Staging**

| Cron Job | Production | Staging | Reasoning |
|----------|------------|---------|-----------|
| 6:00 AM Cleanup | ✅ Enabled | ❌ Disabled | No daily cleanup needed |
| 6:30 AM Check Filings | ✅ Enabled | ❌ Disabled | No duplicate filing checks |
| 7:00 AM Process Users | ✅ Enabled | ❌ Disabled | Don't send emails to real users! |
| 8:30 AM Send Emails | ✅ Enabled | ❌ Disabled | CRITICAL: No emails from staging |
| Hourly Alerts | ✅ Enabled | ❌ Disabled | No hourly emails |

**Staging Testing (Manual Trigger Only):**
- Use admin dashboard: `/admin/test`
- Or API: `POST /jobs/submit` with test tickers

**Step 5: Set Domain**
- Custom domain: `staging.stockdigest.app`
- Or use Render auto-generated: `stockdigest-staging.onrender.com`

### Cost Breakdown

| Item | Production | Staging | Total |
|------|------------|---------|-------|
| **App Service** | $25/month | $25/month | $50/month |
| **Database** | $0 (free tier) | $0 (free tier) | **$0/month** |

**No additional database costs!** (Render provides 1 free PostgreSQL per account)

### Staging Database Strategy

**For Beta:**
- ✅ Use Render's Free PostgreSQL (1GB storage)
- ✅ Reset staging DB weekly (keeps it small)
- ✅ Copy production schema only (not 1M+ articles)
- ✅ Test with 100-1000 sample articles

**Schema Copy Command:**
```bash
# Export production schema
pg_dump $PRODUCTION_DB_URL --schema-only > schema.sql

# Import to staging
psql $STAGING_DB_URL < schema.sql
```

---

## Feature Flags vs Staging

### Do You Need Both?

**Short Answer: No (for beta)**

**Staging branch handles 90% of use cases:**
- ✅ Test code changes
- ✅ Test prompt changes
- ✅ Test database schema changes
- ✅ Catch bugs before production

### When Feature Flags ARE Useful

**Scenario A: Gradual Rollout (Canary Testing)**
```
Problem: You rewrote Phase 1 prompt (major change, risky)

Without flags:
- Deploy to 100% of users at once
- If bad: All users get bad summaries

With flags:
- Enable for 10% of tickers (random selection)
- Compare quality: old (90%) vs new (10%)
- If good: Gradually increase to 25%, 50%, 100%
- If bad: Turn off flag (instant fix, no redeploy)
```

**Scenario B: Instant Emergency Rollback**
```
Without flags:
- Revert git commit → Wait 2-3 min for redeploy

With flags:
- Flip flag in Render dashboard → 2 seconds
```

**Scenario C: A/B Testing**
```python
# Example: Testing summary lengths
if user_id % 2 == 0:
    max_words = 800   # Shorter summaries
else:
    max_words = 1200  # Longer summaries
```

### Implementation (If Needed Later)

```python
# Environment variable
FEATURE_FLAGS = {
    "use_new_summary_prompt": False,
    "enable_phase3_integration": True,
    "max_concurrent_jobs": 4
}

# In code
def generate_executive_summary_phase1(...):
    if FEATURE_FLAGS.get("use_new_summary_prompt"):
        prompt = NEW_PHASE1_PROMPT
    else:
        prompt = PHASE1_PROMPT  # Stable
```

### Recommendation

**For Beta (0-100 users):**
- ❌ Skip feature flags (staging is enough)
- ✅ Staging catches 90% of issues
- ✅ Fast iteration (no flag management overhead)

**Add feature flags when:**
- You have 500+ users (can't break everyone at once)
- Doing major rewrites (Phase 4, new AI model)
- Want to charge for premium features

---

## Database Migration Strategy

### When Migrations Are Needed

| Change Type | Migration Required? | Example |
|-------------|---------------------|---------|
| Add column (optional) | ⚠️ Optional | `ADD COLUMN quality_score INT DEFAULT NULL` |
| Add column (required) | ✅ YES | `ADD COLUMN quality_score INT NOT NULL` (crashes without) |
| Remove column | ✅ YES | Code breaks if column missing |
| Add index | ✅ YES | Can lock table (use `CONCURRENTLY`) |
| Rename column | ✅ YES | Code expects old name |
| Change column type | ✅ YES | Can corrupt data |
| Add new table | ⚠️ Optional | Code handles gracefully |

### Migration Workflow

**Without migrations (current - risky):**
```
1. Staging: ALTER TABLE executive_summaries ADD COLUMN quality_score INT;
2. Staging: Code uses quality_score (works)
3. Deploy to production
4. 💥 ERROR: column "quality_score" does not exist
5. App crashes until manual ALTER TABLE in production
```

**With migrations (safe):**
```
1. Write migration: migrations/001_add_quality_score.sql
2. Run migration in staging (before code deploy)
3. Deploy code to staging (works, column exists)
4. Run migration in production (before code deploy)
5. Deploy code to production (works, column exists)
6. ✅ Zero downtime
```

### Example Migration Files

```sql
-- migrations/001_add_quality_score.sql
ALTER TABLE executive_summaries
ADD COLUMN IF NOT EXISTS quality_score INTEGER DEFAULT NULL;

-- migrations/002_add_index.sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_articles_published
ON articles(published_at DESC);

-- migrations/003_change_date_type.sql
-- Step 1: Add new column
ALTER TABLE press_releases
ADD COLUMN report_date_new TIMESTAMPTZ;

-- Step 2: Copy data
UPDATE press_releases
SET report_date_new = report_date::TIMESTAMPTZ;

-- Step 3: Drop old column (after code updated)
ALTER TABLE press_releases
DROP COLUMN report_date;

-- Step 4: Rename new column
ALTER TABLE press_releases
RENAME COLUMN report_date_new TO report_date;
```

### Recommendation

**For Beta:**
- ✅ Keep manual approach (document changes in `SCHEMA_CHANGES.md`)
- ✅ Before deploying: Check "Does this require ALTER TABLE?"
- ✅ Run SQL manually in production before code deploy

**After 100+ Users:**
- ✅ Adopt Alembic or Django migrations (automated)
- ✅ Migrations run automatically on deploy

---

## Ad-hoc Report Concurrency

### Current Safety Analysis

| Concern | Current Protection | 100% Safe? | Issue |
|---------|-------------------|------------|-------|
| Database race conditions | `FOR UPDATE SKIP LOCKED` | ✅ YES | Atomic job claiming |
| Database connections | Pool 5-80, ~30% @ 4 concurrent | ⚠️ MOSTLY | 10 ad-hoc + 4 scheduled = 70% |
| Memory exhaustion | 60% @ 4 concurrent | ⚠️ MOSTLY | 8 concurrent = 120% → crash |
| Claude API rate limits | Self-limiting (429 errors) | ⚠️ NO | 8 concurrent = frequent 429s |
| ScrapFly rate limits | Self-limiting | ⚠️ NO | 8 concurrent = $1.50/run vs $0.30 |
| SMTP rate limits | No limiting | ❌ NO | 100 emails/min = spam filters |
| Cost explosion | No limiting | ❌ NO | User spam = $500 API bill |

### Solution 1: Job Priority System

**Add priority column:**
```sql
ALTER TABLE ticker_processing_jobs
ADD COLUMN priority INTEGER DEFAULT 50;
```

**Priority values:**
```python
PRIORITY_SCHEDULED = 10  # Morning batch (highest)
PRIORITY_ADHOC = 50      # User ad-hoc (normal)
PRIORITY_RESEARCH = 30   # Admin research (medium)
```

**Update job claiming:**
```python
# Worker query (prioritize scheduled jobs)
SELECT * FROM ticker_processing_jobs
WHERE status = 'queued'
ORDER BY priority ASC, created_at ASC  # Priority first, then FIFO
FOR UPDATE SKIP LOCKED
LIMIT 1;
```

**Behavior:**
```
2:00 AM - Scheduled batch queues 12 jobs (priority=10)
2:01 AM - User triggers ad-hoc for AAPL (priority=50)
2:02 AM - Worker processes ALL priority=10 jobs first
2:47 AM - Scheduled done, now process AAPL (priority=50)
```

### Solution 2: Concurrency Limits by Job Type

```python
# Config
MAX_CONCURRENT_SCHEDULED = 4  # Morning batch
MAX_CONCURRENT_ADHOC = 2       # User ad-hoc
MAX_CONCURRENT_TOTAL = 5       # Hard limit

# Worker logic
def claim_next_job():
    scheduled_running = count_jobs(status='processing', priority=10)
    adhoc_running = count_jobs(status='processing', priority=50)

    # Don't exceed limits
    if scheduled_running >= MAX_CONCURRENT_SCHEDULED:
        return None  # Wait

    if adhoc_running >= MAX_CONCURRENT_ADHOC:
        # Only claim scheduled jobs
        return claim_job(priority=10)

    # Claim highest priority job
    return claim_job(order_by='priority ASC')
```

**Benefits:**
- ✅ Morning batch always gets 4 workers
- ✅ Ad-hoc limited to 2 workers (protects resources)
- ✅ Total never exceeds 5 (safe memory/connection limits)

### Solution 3: User Credit System

**Database schema:**
```sql
ALTER TABLE beta_users
ADD COLUMN credits_remaining INTEGER DEFAULT 5,
ADD COLUMN credits_reset_date DATE DEFAULT CURRENT_DATE;
```

**Monthly reset (cron job):**
```sql
UPDATE beta_users
SET credits_remaining = 5,
    credits_reset_date = CURRENT_DATE
WHERE credits_reset_date < DATE_TRUNC('month', CURRENT_DATE);
```

**API endpoint:**
```python
@APP.post("/api/user/run-adhoc-report")
async def run_adhoc_report(ticker: str, user_email: str):
    user = get_user(user_email)

    if user.credits_remaining <= 0:
        raise HTTPException(
            status_code=429,
            detail="No credits remaining. Resets on 1st of month."
        )

    # Deduct credit BEFORE queuing
    decrement_credits(user_email)

    job = submit_job(ticker, priority=PRIORITY_ADHOC)
    return {
        "job_id": job.id,
        "credits_remaining": user.credits_remaining - 1
    }
```

### Ad-hoc Limit Recommendations

**Launch with conservative limit:**
- ✅ **5 ad-hoc/month** (to start)
- ✅ Track usage for 2-3 months
- ✅ Adjust based on data

**Cost Analysis (15/month limit):**
- 100 users × 15 ad-hoc/month = 1,500 reports/month (worst case)
- Cost per report: ~$1.30 (Claude + ScrapFly + OpenAI)
- Worst case: $1,950/month
- **Realistic:** 100 users × 5 actual = 500 reports = **$650/month**

**Future pricing tiers:**
```
Free Beta:     5 ad-hoc/month   ($0/month)
Premium:      25 ad-hoc/month   ($29/month)
Professional: 100 ad-hoc/month  ($99/month)
```

---

## Infrastructure Scaling Plan

### Current Setup

- **App:** Render Standard (2GB RAM, shared CPU) - $25/month
- **Database:** Render Basic-1GB (100 connections, 1GB RAM) - $0/month
- **Processing:** 4 concurrent tickers, ~45 min for 12 tickers

### Bottleneck Analysis

| Phase | Duration | Bottleneck | Fixable with Hardware? |
|-------|----------|------------|------------------------|
| Feed Ingestion | ~10s | Network I/O (RSS) | ❌ No (external APIs) |
| URL Resolution | ~10s | ScrapFly API | ❌ No (API rate limits) |
| Content Scraping | ~2-3 min | ScrapFly API | ⚠️ Barely (still API-limited) |
| AI Triage | ~30s | Claude/OpenAI | ❌ No (API rate limits) |
| AI Summaries | ~1 min | Claude/OpenAI | ❌ No (API rate limits) |
| Email Generation | ~5s | CPU | ✅ Yes (faster CPU helps) |

**Conclusion:** ~80% of time is external API calls (not CPU/RAM)

### Scaling Triggers

| User Count | Unique Tickers | Concurrent | Hardware | Processing Time | Monthly Cost |
|------------|----------------|------------|----------|-----------------|--------------|
| **0-20 (current)** | ~12 | 4 | 2GB RAM | 45 min | $25 |
| **20-50** | ~60 | 8 | 4GB RAM + Standard DB | 3.75 hours | $135 |
| **50-150** | ~120 | 12 | 8GB RAM + Standard DB | 5 hours | $185 |
| **150-500** | ~600 | 20 | 16GB RAM + Pro DB + Redis | 15 hours | $450 |

### When to Upgrade

**Don't upgrade preemptively!**

Upgrade when processing time exceeds **2 hours** (around 30-40 users)

**Current setup handles:**
- 4 concurrent tickers comfortably
- Up to 20 users (60 unique tickers) with 2 AM processing
- Buffer: 2 AM start → 2:45 AM done → 8:30 AM send = 5.75 hours safety margin

### Better than Hardware: Code Optimizations (Free)

**Option 1: Parallel Phase 1 + Phase 2**
```python
# Current: Sequential
Phase 1: 2 min
Phase 2: 3 min
Phase 3: 1 min
Total: 6 min

# Better: Parallel (Phase 1 + Phase 2 independent)
Phase 1 + Phase 2 (parallel): 3 min
Phase 3: 1 min
Total: 4 min (33% faster)
```

**Option 2: Cache Resolved URLs**
- Avoid duplicate ScrapFly calls → **-$20/month**
- Redis or database cache (24-hour TTL)

**Option 3: Batch ScrapFly Requests**
- 10 URLs per call (if API supports) → **-30 seconds/ticker**

**Total potential speedup: 30-40%, $0 cost**

### Recommendation

**For Beta (0-100 users):**
1. ✅ Keep current hardware (4 concurrent works)
2. ✅ Move processing to 2 AM (solves time pressure)
3. ✅ Optimize code (parallel phases, cache URLs)
4. ❌ Don't upgrade hardware yet

**At 100-500 users:**
1. ✅ Upgrade to 4GB RAM ($85/month) for 8 concurrent
2. ✅ Upgrade database to Standard-3GB ($50/month)
3. ✅ Add Redis cache ($10/month)

---

## Processing Schedule (2 AM vs 7 AM)

### News Publishing Patterns

| Time Window | % of Daily Articles | Source Types |
|-------------|---------------------|--------------|
| 12 AM - 6 AM EST | ~15% | Asia markets, Europe open, overnight earnings |
| 6 AM - 9 AM EST | ~25% | Pre-market earnings, analyst upgrades |
| 9 AM - 4 PM EST | ~45% | Market hours, breaking news |
| 4 PM - 12 AM EST | ~15% | After-hours earnings, analysis |

### Arguments FOR 2 AM Processing

✅ **More buffer time** - 5.5 hours to fix issues before 8:30 AM send
✅ **Lower API congestion** - Fewer users hitting Claude/OpenAI at night
✅ **Newer articles by 8:30 AM** - Can run 7 AM feed refresh if needed
✅ **Can process more tickers** - 30-50 comfortably with 45+ min window
✅ **Cheaper/faster ScrapFly** - Less traffic, faster response times

### Arguments AGAINST 2 AM

⚠️ **Less news overnight** - Midnight-6 AM is slow for US markets (~15% of articles)
⚠️ **Breaking news after 2 AM missed** - 6 AM earnings not in report
⚠️ **Europe/Asia news delayed** - Publish during your night

### Data-Driven Decision

**If you process at 2 AM:** Catch midnight-2 AM articles (~7%)
**If you process at 7 AM:** Catch midnight-7 AM articles (~22%)

**Difference: ~15% more articles** (mostly Europe/Asia + pre-market)

### Hybrid Solution (Optional Complexity)

```
Stage 1: 2 AM - Main batch (feed ingest, AI triage, scraping, emails)
Stage 2: 7 AM - Quick refresh (feed ingest only, breaking news)
Stage 3: 8:30 AM - Send all emails (includes 6-7 AM breaking news)
```

**Benefits:** Main work done by 3 AM, catch 6-7 AM breaking news
**Complexity:** Two cron jobs, need to merge article lists

### Recommendation

**Process at 2 AM (simple, best for beta)**

Why:
1. Beta users care about **quality** > **6 AM breaking news**
2. Need **buffer time** to fix issues (more important than 15% more articles)
3. Most **material news** happens 9 AM - 4 PM (already captured next day)
4. **Pre-market earnings** usually released by midnight (in 2 AM batch)

**After beta:**
- Add 7 AM refresh if users request it
- Or offer "Breaking News Alerts" as separate product

---

## Health Checks & Error Handling

### Why Health Checks Matter

**Even after testing in staging, these can break in production:**

1. **Environment variable typo**
   - Staging: `CLAUDE_API_KEY=sk-ant-staging-123` ✅
   - Production: `CLAUDE_API_KEY=sk-ant-prod-12` (missing digit) ❌

2. **Database connection issue**
   - Staging DB: Online ✅
   - Production DB: Maintenance mode ❌

3. **Dependency version mismatch**
   - Staging: `anthropic==0.25.0` ✅
   - Production deploys: `anthropic==0.26.0` (breaking changes) ❌

4. **Memory/resource exhaustion**
   - Staging: 4GB RAM (plenty) ✅
   - Production: 2GB RAM (code uses 2.5GB) ❌

### Minimal Health Check (Best ROI)

```python
@APP.get("/health")
async def health_check():
    """
    Minimal health check (catches 90% of issues)
    """
    # 1. Database connectivity
    try:
        with get_db_connection() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        return JSONResponse(
            {"status": "unhealthy", "error": str(e)},
            status_code=503
        )

    # 2. Critical imports work
    try:
        from modules.executive_summary_phase1 import generate_executive_summary_phase1
    except Exception as e:
        return JSONResponse(
            {"status": "unhealthy", "error": str(e)},
            status_code=503
        )

    return {"status": "healthy"}
```

### Render Configuration

```yaml
# render.yaml (Infrastructure as Code)
services:
  - type: web
    name: stockdigest-production
    env: python
    healthCheckPath: /health
    autoDeploy: false  # Manual deploys only

  - type: web
    name: stockdigest-staging
    env: python
    healthCheckPath: /health
    autoDeploy: true   # Auto-deploy staging
```

**Benefit:** Render won't route traffic to unhealthy instance (keeps old version running)

### Graceful Shutdown Handler

```python
import signal
import sys

SHUTDOWN_REQUESTED = False

def graceful_shutdown(signum, frame):
    """
    Handle SIGTERM from Render (15-second warning)
    """
    global SHUTDOWN_REQUESTED
    print("🛑 Shutdown requested, finishing current jobs...")
    SHUTDOWN_REQUESTED = True

signal.signal(signal.SIGTERM, graceful_shutdown)

def job_worker_loop():
    while not SHUTDOWN_REQUESTED:
        job = claim_next_job()
        if job:
            process_job(job)

    print("✅ Worker shutdown complete")
    sys.exit(0)
```

**Benefits:**
- ✅ Current jobs finish (no duplicate work)
- ✅ No duplicate emails
- ✅ Clean handoff to new worker

**Current system:** Reclaim thread requeues stale jobs (works, but slower)

### Rollback Strategy

```bash
# Git tags for releases
git tag v1.2.3-production
git push --tags

# Rollback (if needed)
git revert HEAD
git push origin main
# Render deploys previous version
```

**Better:** Keep last 3 working commits tagged

### Recommendation

**Add minimal health check:**
- ✅ 10 minutes of work
- ✅ Catches 90% of production issues
- ✅ Peace of mind

**Skip advanced checks** (API testing, memory) - overkill for beta

---

## Protecting Dangerous Endpoints

### Current Risk

These admin endpoints can trigger destructive actions:
- `POST /admin/wipe-database` - Deletes all data
- `POST /api/commit-ticker-csv` - Triggers GitHub commit → Render deploy
- `POST /admin/init` - Reinitializes ticker feeds
- `POST /admin/clean-feeds` - Deletes old articles

### Solution: Environment-Based Protection

```python
# Environment variable
ENVIRONMENT = "production"  # or "staging"

# Decorator
def require_staging_only(func):
    def wrapper(*args, **kwargs):
        if ENVIRONMENT == "production":
            raise HTTPException(
                status_code=403,
                detail="This endpoint is disabled in production"
            )
        return func(*args, **kwargs)
    return wrapper

# Usage
@APP.post("/admin/wipe-database")
@require_staging_only
async def wipe_database():
    # Only works in staging
    ...

@APP.post("/api/commit-ticker-csv")
@require_staging_only
async def commit_ticker_csv():
    # Only works in staging (prevents accidental deploys)
    ...
```

### Endpoint Protection Levels

**Level 1: DISABLE in production (staging only)**
```python
@require_staging_only
- POST /admin/wipe-database
- POST /api/commit-ticker-csv
- POST /admin/force-regenerate-all
```

**Level 2: PROTECT with confirmation (production allowed, but careful)**
```python
@require_confirmation_token
- POST /admin/clean-feeds
- POST /admin/init
- POST /api/clear-all-reports
```

**Level 3: SAFE (no protection needed)**
```python
# No decorator needed
- POST /api/generate-user-reports
- POST /api/send-all-ready
- GET endpoints (read-only)
```

### Recommendation

**Add environment-based protection (30 min):**
1. Set `ENVIRONMENT=production` in Render env vars
2. Add `@require_staging_only` decorator
3. Protect 3-5 dangerous endpoints

**Don't delete endpoints** - might need for emergencies

---

## Pre-Beta Launch Checklist

### Week 1: Critical Safety (Before Beta Launch)

**Estimated Time:** 1-2 days
**Cost:** $0/month

- [ ] **Set up staging environment** (1-2 hours)
  - Clone production service in Render
  - Create free PostgreSQL for staging
  - Update env vars (ENVIRONMENT=staging, DRY_RUN=true)
  - Point to `staging` git branch
  - Disable all cron jobs

- [ ] **Move production runs to 2 AM** (5 min)
  - Update cron schedule: `0 2 * * *` (instead of `0 7 * * *`)
  - No code changes needed

- [ ] **Add minimal health check** (10 min)
  - Database connectivity check
  - Critical imports check
  - Update Render config: `healthCheckPath: /health`

- [ ] **Implement user credit system** (2-3 hours)
  - Add columns: `credits_remaining`, `credits_reset_date`
  - Add monthly reset cron
  - Update ad-hoc endpoint with credit check
  - Start with 5 credits/month

- [ ] **Add job priority system** (2 hours)
  - Add `priority` column to jobs table
  - Update job claiming query (ORDER BY priority)
  - Set PRIORITY_SCHEDULED=10, PRIORITY_ADHOC=50

- [ ] **Protect dangerous endpoints** (30 min)
  - Add `@require_staging_only` decorator
  - Protect: wipe-database, commit-ticker-csv
  - Set ENVIRONMENT=production in Render

### Week 2: Optimization (Nice to Have)

**Estimated Time:** 1 day
**Cost Savings:** ~$20/month

- [ ] **Optimize code** (3-4 hours)
  - Parallel Phase 1+2 (if possible)
  - Review async opportunities

- [ ] **Add URL resolution caching** (2-3 hours)
  - Redis or database cache (24-hour TTL)
  - Saves ~$20/month ScrapFly costs

- [ ] **Document staging workflow** (30 min)
  - Add to README: How to use staging
  - Deployment checklist

- [ ] **Test full workflow in staging** (1-2 hours)
  - Run 10-ticker batch
  - Verify all 3 emails
  - Test ad-hoc report
  - Verify DRY_RUN works (emails to admin only)

### Week 3: Monitoring & Polish

**Estimated Time:** 3-4 hours
**Cost:** $0/month

- [ ] **Add error tracking** (1 hour)
  - Set up Sentry free tier
  - Add to production only

- [ ] **Create rollback runbook** (30 min)
  - Document: How to rollback deployment
  - Git tag strategy

- [ ] **Set up monitoring dashboard** (1 hour)
  - Render built-in metrics
  - Track: Memory, CPU, response time

- [ ] **Final production test** (1-2 hours)
  - Deploy to production via staging workflow
  - Monitor for 1 hour
  - Verify health checks work

---

## Total Investment Summary

### Time Investment
- **Week 1 (Critical):** 1-2 days
- **Week 2 (Optimization):** 1 day
- **Week 3 (Monitoring):** 3-4 hours
- **Total:** 3-4 days over 3 weeks

### Cost Impact
- **Additional Monthly Cost:** $0 (free tier staging DB)
- **Cost Savings:** ~$20/month (URL caching)
- **Risk Reduction:** Prevents 90% of production disasters

### Key Improvements
✅ Staging environment (test before production)
✅ Health checks (prevent bad deploys)
✅ User credits (prevent abuse, cost control)
✅ Job priorities (scheduled > ad-hoc)
✅ 2 AM processing (5.5 hour safety buffer)
✅ Protected endpoints (no accidental wipes)

---

## Notes & Decisions

**Feature Flags:** Skipping for beta (staging is sufficient)
**Database Migrations:** Manual for beta, automate at 100+ users
**Infrastructure Upgrades:** Wait until 30-40 users (processing time > 2 hours)
**Concurrency Limits:** Add when implementing ad-hoc reports
**Advanced Health Checks:** Minimal check only (database + imports)

**Last Updated:** November 2025
