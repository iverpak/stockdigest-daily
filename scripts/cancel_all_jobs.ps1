# QuantBrief - Cancel All Jobs Script
# Cancels all queued and processing jobs across all batches

# Configuration
$APP = "https://quantbrief-daily.onrender.com"
$TOKEN = "a77774hhwef88f99sd9g883h23nsndfs9d8cnns9adh7asc9xcibjweorn"

$headers = @{ "X-Admin-Token" = $TOKEN; "Content-Type" = "application/json" }

Write-Host "=== QUANTBRIEF JOB CANCELLATION ===" -ForegroundColor Red
Write-Host "This will cancel ALL queued and processing jobs" -ForegroundColor Yellow

# Get all active batches
Write-Host "`nFetching active batches..." -ForegroundColor Cyan

try {
    $stats = Invoke-RestMethod -Method Get "$APP/jobs/stats" -Headers $headers

    if ($stats.status_counts) {
        $queued = $stats.status_counts.queued
        $processing = $stats.status_counts.processing

        if ($queued -eq 0 -and $processing -eq 0) {
            Write-Host "`n✅ No active jobs to cancel!" -ForegroundColor Green
            Write-Host "   Queue is empty." -ForegroundColor Gray
            Write-Host "`nPress any key to exit..." -ForegroundColor White
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
            exit 0
        }

        Write-Host "`nFound active jobs:" -ForegroundColor Yellow
        Write-Host "  Queued: $queued" -ForegroundColor Gray
        Write-Host "  Processing: $processing" -ForegroundColor Gray
    }

} catch {
    Write-Host "`n⚠️ Could not fetch job stats: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "Continuing with cancellation attempt..." -ForegroundColor Gray
}

# Get all batches from the last 24 hours
Write-Host "`nSearching for recent batches..." -ForegroundColor Cyan

try {
    # Execute SQL query to find all recent batches with active jobs
    $sqlQuery = @"
SELECT DISTINCT b.batch_id, b.status,
       COUNT(CASE WHEN j.status IN ('queued', 'processing') THEN 1 END) as active_jobs
FROM ticker_processing_batches b
JOIN ticker_processing_jobs j ON b.batch_id = j.batch_id
WHERE b.created_at > NOW() - INTERVAL '24 hours'
  AND j.status IN ('queued', 'processing')
GROUP BY b.batch_id, b.status
"@

    $sqlBody = @{ query = $sqlQuery } | ConvertTo-Json
    $batches = Invoke-RestMethod -Method Post "$APP/admin/execute-sql" -Headers $headers -Body $sqlBody

    if (-not $batches.results -or $batches.results.Count -eq 0) {
        Write-Host "`n✅ No active batches found!" -ForegroundColor Green
        Write-Host "`nPress any key to exit..." -ForegroundColor White
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 0
    }

    Write-Host "`nFound $($batches.results.Count) batch(es) with active jobs:" -ForegroundColor Yellow
    foreach ($batch in $batches.results) {
        Write-Host "  Batch: $($batch.batch_id.Substring(0,8))... | Active jobs: $($batch.active_jobs)" -ForegroundColor Gray
    }

    # Confirm cancellation
    Write-Host "`n⚠️  WARNING: This will cancel all these jobs!" -ForegroundColor Red
    Write-Host "Press 'Y' to confirm, any other key to abort..." -ForegroundColor Yellow

    $confirmation = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    Write-Host ""

    if ($confirmation.Character -ne 'Y' -and $confirmation.Character -ne 'y') {
        Write-Host "`n❌ Cancellation aborted by user" -ForegroundColor Yellow
        Write-Host "`nPress any key to exit..." -ForegroundColor White
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 0
    }

    # Cancel each batch
    Write-Host "`n=== CANCELLING BATCHES ===" -ForegroundColor Red

    $totalCancelled = 0
    foreach ($batch in $batches.results) {
        $batch_id = $batch.batch_id

        try {
            Write-Host "`nCancelling batch $($batch_id.Substring(0,8))..." -ForegroundColor Cyan

            $result = Invoke-RestMethod -Method Post "$APP/jobs/batch/$batch_id/cancel" -Headers $headers

            if ($result.status -eq "cancelled") {
                $jobsCancelled = $result.jobs_cancelled
                $totalCancelled += $jobsCancelled

                Write-Host "  ✅ Cancelled $jobsCancelled job(s)" -ForegroundColor Green

                if ($result.tickers) {
                    Write-Host "     Tickers: $($result.tickers -join ', ')" -ForegroundColor Gray
                }
            } elseif ($result.status -eq "no_jobs_to_cancel") {
                Write-Host "  ℹ️  No jobs to cancel (already completed)" -ForegroundColor Gray
            }

        } catch {
            Write-Host "  ❌ Failed to cancel batch: $($_.Exception.Message)" -ForegroundColor Red
        }
    }

    Write-Host "`n=== CANCELLATION COMPLETE ===" -ForegroundColor Green
    Write-Host "Total jobs cancelled: $totalCancelled" -ForegroundColor Cyan

    # Show updated stats
    Write-Host "`nFetching updated queue stats..." -ForegroundColor Cyan

    try {
        $updatedStats = Invoke-RestMethod -Method Get "$APP/jobs/stats" -Headers $headers

        if ($updatedStats.status_counts) {
            Write-Host "`nQueue Status:" -ForegroundColor Yellow
            Write-Host "  Queued: $($updatedStats.status_counts.queued)" -ForegroundColor Gray
            Write-Host "  Processing: $($updatedStats.status_counts.processing)" -ForegroundColor Gray
            Write-Host "  Completed: $($updatedStats.status_counts.completed)" -ForegroundColor Green
            Write-Host "  Failed: $($updatedStats.status_counts.failed)" -ForegroundColor Red

            if ($updatedStats.status_counts.cancelled) {
                Write-Host "  Cancelled: $($updatedStats.status_counts.cancelled)" -ForegroundColor Yellow
            }
        }

    } catch {
        Write-Host "  ⚠️ Could not fetch updated stats" -ForegroundColor Yellow
    }

} catch {
    Write-Host "`n🚨 ERROR: $($_.Exception.Message)" -ForegroundColor Red

    # Fallback: Try to get stats and show manual cancellation command
    Write-Host "`nIf you have a specific batch_id, you can cancel it manually:" -ForegroundColor Yellow
    Write-Host "Invoke-RestMethod -Method Post `"`$APP/jobs/batch/`$batch_id/cancel`" -Headers `$headers" -ForegroundColor Gray
}

Write-Host "`n✅ Script complete!" -ForegroundColor Green
Write-Host "`nPress any key to exit..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")