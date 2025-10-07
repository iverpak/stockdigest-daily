# StockDigest - Cancel ALL Active Jobs Script
# Automatically finds and cancels all batches with active jobs

# Configuration
$APP = "https://stockdigest-daily.onrender.com"
$TOKEN = "a77774hhwef88f99sd9g883h23nsndfs9d8cnns9adh7asc9xcibjweorn"

$headers = @{ "X-Admin-Token" = $TOKEN; "Content-Type" = "application/json" }

Write-Host "=== QUANTBRIEF JOB CANCELLATION ===" -ForegroundColor Red
Write-Host "Searching for ALL active batches..." -ForegroundColor Cyan

try {
    # Get all active batches
    $activeBatches = Invoke-RestMethod -Method Get "$APP/jobs/active-batches" -Headers $headers

    if ($activeBatches.active_batches -eq 0) {
        Write-Host "`n✅ No active jobs to cancel!" -ForegroundColor Green
        Write-Host "   Queue is empty." -ForegroundColor Gray
        Write-Host "`nPress any key to exit..." -ForegroundColor White
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 0
    }

    # Show what will be cancelled
    Write-Host "`nFound $($activeBatches.active_batches) active batch(es):" -ForegroundColor Yellow
    Write-Host ""

    $totalActiveJobs = 0
    foreach ($batch in $activeBatches.batches) {
        $batchIdShort = $batch.batch_id.Substring(0, 8)
        Write-Host "  Batch: $batchIdShort..." -ForegroundColor Cyan
        Write-Host "    Total jobs: $($batch.total_jobs) | Completed: $($batch.completed_jobs) | Failed: $($batch.failed_jobs)" -ForegroundColor Gray

        $activeJobs = $batch.jobs | Where-Object { $_.status -in @('queued', 'processing') }
        if ($activeJobs) {
            $totalActiveJobs += $activeJobs.Count
            Write-Host "    Active jobs ($($activeJobs.Count)):" -ForegroundColor Yellow
            foreach ($job in $activeJobs) {
                $statusDisplay = $job.status
                if ($job.phase) { $statusDisplay += " [$($job.phase)]" }
                if ($job.progress) { $statusDisplay += " $($job.progress)%" }
                Write-Host "      - $($job.ticker): $statusDisplay" -ForegroundColor Gray
            }
        }
        Write-Host ""
    }

    # Confirm cancellation
    Write-Host "⚠️  WARNING: This will cancel ALL $totalActiveJobs active job(s) across $($activeBatches.active_batches) batch(es)!" -ForegroundColor Red
    Write-Host "Press 'Y' to confirm, any other key to abort: " -ForegroundColor Yellow -NoNewline

    $confirmation = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    Write-Host ""

    if ($confirmation.Character -ne 'Y' -and $confirmation.Character -ne 'y') {
        Write-Host "`n❌ Cancellation aborted" -ForegroundColor Yellow
        Write-Host "`nPress any key to exit..." -ForegroundColor White
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 0
    }

    # Cancel each batch
    Write-Host "`n=== CANCELLING BATCHES ===" -ForegroundColor Red

    $totalCancelled = 0
    foreach ($batch in $activeBatches.batches) {
        $batch_id = $batch.batch_id
        $batchIdShort = $batch_id.Substring(0, 8)

        try {
            Write-Host "`nCancelling batch $batchIdShort..." -ForegroundColor Cyan

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
            $errorMsg = $_.Exception.Message
            Write-Host "  ❌ Failed to cancel batch: $errorMsg" -ForegroundColor Red
        }
    }

    Write-Host "`n=== CANCELLATION COMPLETE ===" -ForegroundColor Green
    Write-Host "Total jobs cancelled: $totalCancelled" -ForegroundColor Cyan

    # Clean up saved batch_id file
    $logPath = "$PSScriptRoot\..\last_batch_id.txt"
    if (Test-Path $logPath) {
        Remove-Item $logPath -ErrorAction SilentlyContinue
        Write-Host "Cleared saved batch_id" -ForegroundColor Gray
    }

} catch {
    $errorMsg = $_.Exception.Message

    Write-Host "`n🚨 ERROR: $errorMsg" -ForegroundColor Red

    if ($errorMsg -match "404") {
        Write-Host "`n⚠️  The /jobs/active-batches endpoint may not be deployed yet." -ForegroundColor Yellow
        Write-Host "Wait 2-3 minutes for Render to deploy, then try again." -ForegroundColor Gray
    } elseif ($errorMsg -match "500") {
        Write-Host "`n⚠️  Server error. The endpoint needs to be fixed." -ForegroundColor Yellow
        Write-Host "Deploying fix now..." -ForegroundColor Gray
    }

    Write-Host "`n💡 FALLBACK: Cancel manually:" -ForegroundColor Cyan
    Write-Host "1. Check Render logs for: 📦 Batch {batch_id} created" -ForegroundColor Gray
    Write-Host "2. Run:" -ForegroundColor Gray
    Write-Host '   $batch_id = "YOUR-BATCH-ID"' -ForegroundColor Gray
    Write-Host '   Invoke-RestMethod -Method Post "$APP/jobs/batch/$batch_id/cancel" -Headers @{ "X-Admin-Token" = "' + $TOKEN + '" }' -ForegroundColor Gray
}

Write-Host "`n✅ Script complete!" -ForegroundColor Green
Write-Host "`nPress any key to exit..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")