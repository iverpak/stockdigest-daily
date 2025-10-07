# Job Cancellation Guide

## Quick Reference

### Cancel Entire Batch (Most Common)

```powershell
# Get your batch_id from PowerShell output when you submit
# Or from Render logs: "📦 Batch {batch_id} created"

$APP = "https://stockdigest-daily.onrender.com"
$TOKEN = "a77774hhwef88f99sd9g883h23nsndfs9d8cnns9adh7asc9xcibjweorn"
$headers = @{ "X-Admin-Token" = $TOKEN }

$batch_id = "3a9ece66-f647-46cc-b32f-15a2ab2ea9f5"

# Cancel it
Invoke-RestMethod -Method Post "$APP/jobs/batch/$batch_id/cancel" -Headers $headers
```

**Response:**
```json
{
  "status": "cancelled",
  "batch_id": "3a9ece66...",
  "jobs_cancelled": 4,
  "tickers": ["RY.TO", "TD.TO", "VST", "CEG"]
}
```

---

### Cancel Single Job

```powershell
# Get job_id from batch status:
$status = Invoke-RestMethod -Method Get "$APP/jobs/batch/$batch_id" -Headers $headers
$job_id = $status.jobs[0].job_id

# Cancel specific job
Invoke-RestMethod -Method Post "$APP/jobs/$job_id/cancel" -Headers $headers
```

**Response:**
```json
{
  "status": "cancelled",
  "job_id": "abc123...",
  "ticker": "RY.TO",
  "was_in_phase": "ingest_complete"
}
```

---

## Common Workflows

### **Scenario 1: Wrong Tickers, Want to Start Over**

```powershell
# You submitted: RY.TO, TD.TO
# But meant: VST, CEG

# 1. Cancel current batch
Invoke-RestMethod -Method Post "$APP/jobs/batch/$batch_id/cancel" -Headers $headers

# 2. Update your script
# Edit setup_job_queue.ps1: $TICKERS = @("VST", "CEG")

# 3. Run new batch
.\scripts\setup_job_queue.ps1
```

---

### **Scenario 2: Code Change, Need Fresh Run**

```powershell
# You're testing scraping logic
# Made code change, want to rerun

# 1. Cancel current batch
Invoke-RestMethod -Method Post "$APP/jobs/batch/$batch_id/cancel" -Headers $headers

# 2. Deploy code change (git push)
# Render auto-deploys

# 3. Run new batch with same tickers
.\scripts\setup_job_queue.ps1
```

---

### **Scenario 3: One Ticker Stuck, Cancel Just That One**

```powershell
# Batch has 4 tickers: RY.TO (done), TD.TO (stuck), VST (queued), CEG (queued)

# Get batch status
$status = Invoke-RestMethod -Method Get "$APP/jobs/batch/$batch_id" -Headers $headers

# Find stuck job
$stuck = $status.jobs | Where-Object { $_.ticker -eq "TD.TO" }
$job_id = $stuck.job_id

# Cancel just TD.TO
Invoke-RestMethod -Method Post "$APP/jobs/$job_id/cancel" -Headers $headers

# VST and CEG will still process normally
```

---

### **Scenario 4: Cancel from PowerShell During Run**

```powershell
# You're watching progress in PowerShell
# Progress: 25% | Completed: 1/4 | Failed: 0 | Current: TD.TO [ingest_start]

# Press Ctrl+C to stop watching (job keeps running)

# In a new PowerShell window:
$batch_id = "..." # From original script output
Invoke-RestMethod -Method Post "$APP/jobs/batch/$batch_id/cancel" -Headers $headers
```

---

## What Happens When You Cancel?

### **Immediate Effects:**
1. Database marks job(s) as `status='cancelled'`
2. `error_message='Cancelled by user'` (or 'Batch cancelled by user')
3. Batch `failed_jobs` counter incremented

### **Worker Behavior:**
- **If job is queued:** Never starts, stays cancelled
- **If job is processing:** Finishes current phase, then exits
- **Cancellation checks happen at:**
  - Before starting
  - After Phase 1 (Ingest)
  - After Phase 2 (Digest)
  - NOT mid-phase (current operation completes)

### **Render Logs:**
```
🚫 Batch 3a9ece66... cancelled by user (3 jobs)
   Cancelled: TD.TO (job_id: abc123)
   Cancelled: VST (job_id: def456)
   Cancelled: CEG (job_id: ghi789)

🚫 [JOB abc123] Job cancelled after Phase 1, exiting
```

### **PowerShell:**
If you're still watching, you'll see:
```
Progress: 50% | Completed: 2/4 | Failed: 3 | Current: (none)
```

Failed count includes cancelled jobs.

---

## When Cancellation Won't Work

### **Job Already Completed**
```powershell
# Response (400 error):
{
  "detail": "Job already completed, cannot cancel"
}
```

**Solution:** Nothing to cancel, job is done.

### **Batch Not Found**
```powershell
# Response (404 error):
{
  "detail": "Batch not found"
}
```

**Solution:** Check your batch_id is correct.

### **No Jobs to Cancel**
```powershell
# Response (200, but...):
{
  "status": "no_jobs_to_cancel",
  "message": "Batch is already completed, no jobs to cancel"
}
```

**Solution:** All jobs already finished/failed/cancelled.

---

## Advanced: Check Before Cancelling

```powershell
# Get batch status
$status = Invoke-RestMethod -Method Get "$APP/jobs/batch/$batch_id" -Headers $headers

# See what's cancellable
$cancellable = $status.jobs | Where-Object {
    $_.status -eq "queued" -or $_.status -eq "processing"
}

Write-Host "Can cancel: $($cancellable.Count) jobs"
foreach ($job in $cancellable) {
    Write-Host "  - $($job.ticker) ($($job.status), $($job.phase))"
}

# Cancel if needed
if ($cancellable.Count -gt 0) {
    Invoke-RestMethod -Method Post "$APP/jobs/batch/$batch_id/cancel" -Headers $headers
}
```

---

## Comparison: Old vs New

### **Before (Manual DB Access Required)**
```sql
-- Had to SSH into database
UPDATE ticker_processing_jobs
SET status = 'cancelled'
WHERE batch_id = '...';
```

### **After (Simple API Call)**
```powershell
# One line
Invoke-RestMethod -Method Post "$APP/jobs/batch/$batch_id/cancel" -Headers $headers
```

---

## Safety Features

✅ **Can't cancel completed jobs** - Returns 400 error
✅ **Graceful exit** - Current phase finishes, no corruption
✅ **Audit trail** - Logged in Render, stored in DB
✅ **Batch counter updates** - failed_jobs incremented
✅ **No resource leaks** - Worker exits cleanly

---

## Troubleshooting

### **"Job still processing after cancel"**

**Why:** Worker finishes current phase before exiting.

**How long:**
- Phase 1 (Ingest): Up to 15 minutes
- Phase 2 (Digest): Up to 15 minutes
- Phase 3 (Commit): < 1 minute

**What to do:** Wait for current phase to complete. Check logs for:
```
🚫 [JOB xxx] Job cancelled after Phase X, exiting
```

### **"Cancelled job shows as failed, not cancelled"**

**This is correct!** Cancelled jobs count as "failed" in batch stats.

**In database:** `status='cancelled'`
**In batch summary:** `failed_jobs` includes cancelled

### **"Want to cancel faster (mid-phase)"**

**Not supported** - would cause data corruption.

**Workaround:** Restart Render service (nuclear option, kills worker immediately).

---

## Integration with Your Workflow

### **Rapid Testing Loop**
```powershell
# Test loop
while ($true) {
    # Run
    .\scripts\setup_job_queue.ps1

    # Check output
    # If wrong, Ctrl+C and cancel

    # Fix code
    git add . && git commit -m "Fix" && git push

    # Render deploys (~2 min)
    Start-Sleep -Seconds 120

    # Try again
}
```

---

## Summary

**Cancel batch:**
```powershell
Invoke-RestMethod -Method Post "$APP/jobs/batch/$batch_id/cancel" -Headers $headers
```

**That's it!** Worker exits gracefully, you can start fresh immediately.