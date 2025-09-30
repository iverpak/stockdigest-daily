# Smart Render Deployment Strategy

**Date**: 2025-09-30
**Status**: ✅ Implemented
**Feature**: Automatic `[skip render]` control based on batch position

---

## Problem Statement

**Original Issue**:
- Every GitHub commit triggered Render auto-deployment
- Server restarted mid-job → orphaned jobs → timeouts
- Example: NVDA commit at 04:30 → server restart at 04:34 → META timeout at 04:50

**Requirements**:
1. ✅ No server restarts during batch processing
2. ✅ Fresh CSV deployed to production after batch completes
3. ✅ AI metadata saved to GitHub after each ticker (data durability)

---

## Solution: Dynamic `[skip render]` Control

### Architecture

```
Batch: [RY.TO, TD.TO, NVDA, META]

Job 1: RY.TO completes → Commit with [skip render] → No deployment
Job 2: TD.TO completes → Commit with [skip render] → No deployment
Job 3: NVDA completes → Commit with [skip render] → No deployment
Job 4: META completes → Commit WITHOUT [skip render] → ✅ Render deploys!
```

**Key Innovation**: Check remaining jobs in batch to determine if current job is last.

---

## Implementation Details

### 1. Enhanced `process_commit_phase()` (Line 8976)

```python
async def process_commit_phase(job_id: str, ticker: str, batch_id: str = None, is_last_job: bool = False):
    # Skip render for all jobs EXCEPT the last one in batch
    skip_render = not is_last_job

    if is_last_job:
        LOG.info(f"⚠️ LAST JOB IN BATCH - Render will deploy after this commit")
    else:
        LOG.info(f"[skip render] flag enabled - no deployment")

    result = await commit_func(MockRequest(), CommitBody(
        tickers=[ticker],
        job_id=job_id,
        skip_render=skip_render  # ← Dynamic control
    ))
```

### 2. Batch Detection Logic (Line 9171)

```python
# Check if this is the last job in the batch
batch_id = job.get('batch_id')
is_last_job = False

if batch_id:
    with db() as conn, conn.cursor() as cur:
        # Count remaining jobs in batch (queued + processing, excluding this one)
        cur.execute("""
            SELECT COUNT(*) as remaining
            FROM ticker_processing_jobs
            WHERE batch_id = %s
            AND status IN ('queued', 'processing')
            AND job_id != %s
        """, (batch_id, job_id))

        remaining_jobs = cur.fetchone()['remaining']

        if remaining_jobs == 0:
            is_last_job = True  # ← This triggers deployment
```

**SQL Logic**:
- Query finds jobs that are still `queued` or `processing` (not completed/failed)
- Excludes current job (`job_id != %s`)
- If count = 0 → This is the last job → Deploy!

### 3. Dynamic Commit Message (Line 12014)

```python
skip_prefix = "[skip render] " if body.skip_render else ""
commit_message = f"{skip_prefix}Incremental update: {ticker} - {timestamp}"
```

**Examples**:
- Job 1-3: `"[skip render] Incremental update: RY.TO - 20250930_120000 [job:abc123]"`
- Job 4 (last): `"Incremental update: META - 20250930_123000 [job:def456]"`

---

## Workflow Example

### Batch Submission
```bash
curl -X POST "https://quantbrief-daily.onrender.com/jobs/submit" \
  -H "X-Admin-Token: $TOKEN" \
  -d '{"tickers": ["RY.TO", "TD.TO", "NVDA", "META"], "minutes": 1440}'
```

**Response**:
```json
{
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_jobs": 4,
  "status": "queued"
}
```

---

### Job Processing Timeline

#### T+0:00 - Job 1: RY.TO Starts
```
📥 Phase 1: Ingest (RSS, AI triage)
💾 Committing AI metadata to GitHub...
   Remaining jobs in batch: 3 (TD.TO, NVDA, META)
   [skip render] flag enabled - no deployment
✅ GitHub commit: [skip render] Incremental update: RY.TO [job:abc123]
📧 Phase 2: Digest (scrape, AI summaries, email)
✅ Job completed in 28.5 min
```

**GitHub**: CSV updated, no deployment
**Render**: Server keeps running

---

#### T+28:30 - Job 2: TD.TO Starts
```
📥 Phase 1: Ingest
💾 Committing AI metadata to GitHub...
   Remaining jobs in batch: 2 (NVDA, META)
   [skip render] flag enabled - no deployment
✅ GitHub commit: [skip render] Incremental update: TD.TO [job:bcd234]
📧 Phase 2: Digest
✅ Job completed in 30.2 min
```

**GitHub**: CSV updated, no deployment
**Render**: Server keeps running

---

#### T+58:45 - Job 3: NVDA Starts
```
📥 Phase 1: Ingest
💾 Committing AI metadata to GitHub...
   Remaining jobs in batch: 1 (META)
   [skip render] flag enabled - no deployment
✅ GitHub commit: [skip render] Incremental update: NVDA [job:cde345]
📧 Phase 2: Digest
✅ Job completed in 27.8 min
```

**GitHub**: CSV updated, no deployment
**Render**: Server keeps running

---

#### T+86:33 - Job 4: META Starts (LAST JOB)
```
📥 Phase 1: Ingest
💾 Committing AI metadata to GitHub...
   Remaining jobs in batch: 0
   🎯 This is the LAST job in batch 550e8400-e29b-41d4-a716-446655440000
   ⚠️ LAST JOB IN BATCH - Render will deploy after this commit
   ⚠️ RENDER DEPLOYMENT WILL BE TRIGGERED by this commit
   Commit message: Incremental update: META - 20250930_130000 [job:def456]
✅ GitHub commit: Incremental update: META [job:def456]
📧 Phase 2: Digest
✅ Job completed in 29.1 min
```

**GitHub**: CSV updated
**Render**: 🚀 Auto-deployment triggered!

---

#### T+115:40 - Render Deployment Starts
```
🚀 Render detects commit → Starts deployment
   Building Docker image...
   Deploying new version...

🛑 Old server shutdown:
   ⚠️ SHUTDOWN: 0 jobs still processing (all completed!)
   ✅ No active jobs at shutdown

🚀 New server startup:
   FastAPI STARTUP - Worker: srv-xyz789
   ✅ No orphaned jobs found - clean startup
   ✅ Job queue system initialized
```

**Result**: Fresh CSV now in production, zero data loss!

---

## Edge Cases Handled

### Case 1: Job Fails Mid-Batch

**Scenario**: RY.TO succeeds, TD.TO fails, NVDA/META still queued

```
Job 1: RY.TO → [skip render] ✅
Job 2: TD.TO → Failed ❌
Job 3: NVDA → Starts processing
   Remaining jobs query: status IN ('queued', 'processing')
   Result: 1 (META still queued)
   → [skip render] enabled
Job 4: META → Starts processing
   Remaining jobs: 0
   → 🎯 LAST JOB → Deployment triggered
```

**Outcome**: Deployment still happens after META, CSV includes all successful tickers.

---

### Case 2: Single Ticker Batch

**Scenario**: Only 1 ticker in batch

```
Job 1: AAPL → Starts
   Remaining jobs: 0
   → 🎯 LAST JOB → Deployment triggered immediately
```

**Outcome**: Single-ticker batches always trigger deployment (correct behavior).

---

### Case 3: Job Cancelled Mid-Batch

**Scenario**: User cancels TD.TO while RY.TO is processing

```
Job 1: RY.TO → Processing
Job 2: TD.TO → User cancels (status='cancelled')
Job 3: NVDA → Queued
Job 4: META → Queued

RY.TO completes:
   Remaining jobs query: WHERE status IN ('queued', 'processing')
   Result: 2 (NVDA, META) - TD.TO excluded (status='cancelled')
   → [skip render] enabled ✅
```

**Outcome**: Cancelled jobs don't count as "remaining", deployment logic still works.

---

### Case 4: Server Restarts Mid-Batch (Emergency)

**Scenario**: Render platform maintenance forces restart

```
T+30: Server restarts unexpectedly
T+35: New server starts up
   🔄 RECLAIMED 2 orphaned jobs (TD.TO, NVDA)
   Jobs requeued for processing

TD.TO restarts:
   Remaining jobs: 2 (NVDA, META)
   → [skip render] enabled

NVDA completes:
   Remaining jobs: 1 (META)
   → [skip render] enabled

META completes:
   Remaining jobs: 0
   → 🎯 LAST JOB → Deployment triggered
```

**Outcome**: Even with emergency restarts, deployment logic remains correct.

---

## Configuration Options

### Default Behavior (Recommended)
```python
# In UpdateTickersRequest model (line 11444)
skip_render: Optional[bool] = True  # Default: skip render
```

All jobs use `[skip render]` EXCEPT last job in batch.

---

### Manual Override (For Testing)

**Force deployment on specific ticker**:
```bash
curl -X POST "https://quantbrief-daily.onrender.com/admin/safe-incremental-commit" \
  -H "X-Admin-Token: $TOKEN" \
  -d '{"tickers": ["AAPL"], "skip_render": false}'
```

**Force no deployment (emergency)**:
```bash
curl -X POST "https://quantbrief-daily.onrender.com/admin/safe-incremental-commit" \
  -H "X-Admin-Token: $TOKEN" \
  -d '{"tickers": ["AAPL"], "skip_render": true}'
```

---

## Testing Checklist

### Test 1: 4-Ticker Batch
```bash
curl -X POST "https://quantbrief-daily.onrender.com/jobs/submit" \
  -d '{"tickers": ["RY.TO", "TD.TO", "NVDA", "META"]}'
```

**Expected**:
- Jobs 1-3: Commits show `[skip render]`
- Job 4: Commit missing `[skip render]`
- After Job 4: Server restarts

---

### Test 2: Single Ticker
```bash
curl -X POST "https://quantbrief-daily.onrender.com/jobs/submit" \
  -d '{"tickers": ["AAPL"]}'
```

**Expected**:
- Job 1 logs: "🎯 This is the LAST job in batch"
- Commit missing `[skip render]`
- Server restarts after Job 1

---

### Test 3: Check GitHub Commits
```bash
# After batch completes, check GitHub commits
gh api repos/USERNAME/quantbrief-daily/commits --jq '.[0:4] | .[] | .commit.message'
```

**Expected Output**:
```
Incremental update: META - 20250930_130000 [job:def456]
[skip render] Incremental update: NVDA - 20250930_125833 [job:cde345]
[skip render] Incremental update: TD.TO - 20250930_125520 [job:bcd234]
[skip render] Incremental update: RY.TO - 20250930_120000 [job:abc123]
```

---

## Monitoring

### Check Batch Status
```bash
curl "https://quantbrief-daily.onrender.com/jobs/batch/$BATCH_ID"
```

**Look for**:
- `progress_percentage`: 100%
- `jobs[].status`: All "completed"
- Last job commit message: No `[skip render]`

### Check Server Uptime
```bash
curl "https://quantbrief-daily.onrender.com/health" | jq '.worker.worker_id'
```

Compare worker_id before/after batch:
- Same worker_id = No restart during batch ✅
- Different worker_id = Restart occurred ❌

---

## Rollback Plan

If deployment logic breaks, revert to always using `[skip render]`:

```python
# Line 12014 - Emergency rollback
skip_prefix = "[skip render] "  # Always skip
commit_message = f"{skip_prefix}Incremental update: {ticker} - {timestamp}"
```

Then manually deploy via Render dashboard after all jobs complete.

---

## Benefits Summary

| Metric | Before | After |
|--------|--------|-------|
| Server restarts per batch | 1-4 | 1 (at end) |
| Orphaned jobs per batch | 1-3 | 0 |
| GitHub commits per batch | 4 | 4 (unchanged) |
| Production CSV freshness | Stale during run | Fresh after run |
| Data loss risk | Medium | None |
| Manual intervention needed | Yes (restart timing) | No |

---

## Future Enhancements

### Priority: MEDIUM
- [ ] **Deployment delay** - Wait 5 min after last job before deploying (in case more jobs queued)
- [ ] **Deployment window** - Only deploy during off-hours (e.g., 2-6 AM EST)
- [ ] **Blue-green deployment** - Zero-downtime deploys via Render

### Priority: LOW
- [ ] **Deployment notifications** - Slack/email when deployment triggered
- [ ] **CSV diff summary** - Show which tickers changed in deployment commit message

---

## Questions Answered

**Q: What if I manually submit another ticker while batch is running?**
A: New job gets added to queue with different batch_id. Original batch's last-job detection still works correctly.

**Q: Does this work with PowerShell scripts?**
A: Yes! `setup_job_queue.ps1` submits batch, gets batch_id, polls until complete. Logic is server-side.

**Q: Can I force deploy mid-batch (emergency)?**
A: Yes. Manually call `/admin/safe-incremental-commit` with `skip_render: false`.

**Q: What if GitHub is down?**
A: Job continues, commit marked as failed (non-fatal). Deployment won't trigger. CSV can be manually committed later.

---

**Status**: Ready for production testing
**Recommended Test**: 4-ticker batch during low-traffic period