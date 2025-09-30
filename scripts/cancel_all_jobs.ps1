# QuantBrief - Cancel Jobs Script
# Cancel a specific batch or the most recent batch

# Configuration
$APP = "https://quantbrief-daily.onrender.com"
$TOKEN = "a77774hhwef88f99sd9g883h23nsndfs9d8cnns9adh7asc9xcibjweorn"

$headers = @{ "X-Admin-Token" = $TOKEN; "Content-Type" = "application/json" }

Write-Host "=== QUANTBRIEF JOB CANCELLATION ===" -ForegroundColor Red

# Try to find batch_id from saved file
$logPath = "$PSScriptRoot\..\last_batch_id.txt"
$batch_id = $null

if (Test-Path $logPath) {
    try {
        $batch_id = (Get-Content $logPath -ErrorAction Stop).Trim()
        if ($batch_id) {
            Write-Host "`nFound saved batch_id from last run:" -ForegroundColor Green
            Write-Host "  $($batch_id.Substring(0, 13))..." -ForegroundColor Cyan
            Write-Host "`nUse this batch? [Y/n]: " -ForegroundColor Yellow -NoNewline
            $confirm = Read-Host

            if ($confirm -eq 'n' -or $confirm -eq 'N') {
                $batch_id = $null
            }
        }
    } catch {
        # File read failed
    }
}

# Manual entry if needed
if (-not $batch_id) {
    Write-Host "`nEnter batch_id to cancel: " -ForegroundColor Cyan -NoNewline
    $userInput = Read-Host

    if (-not $userInput) {
        Write-Host "`n❌ No batch_id provided." -ForegroundColor Red
        Write-Host "`n💡 TIP: Find batch_id in Render logs:" -ForegroundColor Yellow
        Write-Host "   Look for: 📦 Batch {batch_id} created" -ForegroundColor Gray
        Write-Host "`nPress any key to exit..." -ForegroundColor White
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 1
    }

    $batch_id = $userInput.Trim()
}

Write-Host "`n=== CHECKING BATCH STATUS ===" -ForegroundColor Cyan

try {
    # Get batch status first
    $status = Invoke-RestMethod -Method Get "$APP/jobs/batch/$batch_id" -Headers $headers

    Write-Host "`nBatch: $($batch_id.Substring(0, 13))..." -ForegroundColor Cyan
    Write-Host "  Status: $($status.batch_status)" -ForegroundColor Gray
    Write-Host "  Total jobs: $($status.total_tickers)" -ForegroundColor Gray
    Write-Host "  Completed: $($status.completed)" -ForegroundColor Green
    Write-Host "  Failed: $($status.failed)" -ForegroundColor $(if ($status.failed -gt 0) { "Red" } else { "Gray" })
    Write-Host "  Processing: $($status.processing)" -ForegroundColor Yellow
    Write-Host "  Queued: $($status.queued)" -ForegroundColor Gray

    # Check if there are any active jobs
    $activeJobs = $status.jobs | Where-Object { $_.status -in @('queued', 'processing') }

    if ($activeJobs.Count -eq 0) {
        Write-Host "`n✅ No active jobs to cancel in this batch!" -ForegroundColor Green
        Write-Host "   All jobs are already completed/failed/cancelled." -ForegroundColor Gray
        Write-Host "`nPress any key to exit..." -ForegroundColor White
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 0
    }

    # Show active jobs
    Write-Host "`nActive jobs ($($activeJobs.Count)):" -ForegroundColor Yellow
    foreach ($job in $activeJobs) {
        $statusDisplay = $job.status
        if ($job.phase) { $statusDisplay += " [$($job.phase)]" }
        if ($job.progress) { $statusDisplay += " $($job.progress)%" }
        Write-Host "  - $($job.ticker): $statusDisplay" -ForegroundColor Gray
    }

    # Confirm cancellation
    Write-Host "`n⚠️  Cancel $($activeJobs.Count) job(s)?" -ForegroundColor Red
    Write-Host "Press 'Y' to confirm, any other key to abort: " -ForegroundColor Yellow -NoNewline
    $confirmation = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    Write-Host ""

    if ($confirmation.Character -ne 'Y' -and $confirmation.Character -ne 'y') {
        Write-Host "`n❌ Cancellation aborted" -ForegroundColor Yellow
        Write-Host "`nPress any key to exit..." -ForegroundColor White
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 0
    }

    # Cancel the batch
    Write-Host "`n=== CANCELLING BATCH ===" -ForegroundColor Red

    $result = Invoke-RestMethod -Method Post "$APP/jobs/batch/$batch_id/cancel" -Headers $headers

    if ($result.status -eq "cancelled") {
        Write-Host "`n✅ SUCCESS: Cancelled $($result.jobs_cancelled) job(s)" -ForegroundColor Green

        if ($result.tickers) {
            Write-Host "   Tickers: $($result.tickers -join ', ')" -ForegroundColor Gray
        }

        # Clean up saved batch_id
        if (Test-Path $logPath) {
            Remove-Item $logPath -ErrorAction SilentlyContinue
            Write-Host "   Cleared saved batch_id" -ForegroundColor Gray
        }

    } elseif ($result.status -eq "no_jobs_to_cancel") {
        Write-Host "`n✅ Batch already completed, no jobs to cancel" -ForegroundColor Green
    }

} catch {
    $errorMsg = $_.Exception.Message

    Write-Host "`n🚨 ERROR: $errorMsg" -ForegroundColor Red

    if ($errorMsg -match "404") {
        Write-Host "`n❌ Batch not found. It may have been deleted or the ID is incorrect." -ForegroundColor Red
    } elseif ($errorMsg -match "400") {
        Write-Host "`n❌ Cannot cancel: Jobs may already be completed." -ForegroundColor Red
    } else {
        Write-Host "`n💡 Try cancelling manually with:" -ForegroundColor Yellow
        Write-Host '$batch_id = "' + $batch_id + '"' -ForegroundColor Gray
        Write-Host '$APP = "' + $APP + '"' -ForegroundColor Gray
        Write-Host '$TOKEN = "' + $TOKEN + '"' -ForegroundColor Gray
        Write-Host '$headers = @{ "X-Admin-Token" = $TOKEN }' -ForegroundColor Gray
        Write-Host 'Invoke-RestMethod -Method Post "$APP/jobs/batch/$batch_id/cancel" -Headers $headers' -ForegroundColor Gray
    }
}

Write-Host "`n✅ Script complete!" -ForegroundColor Green
Write-Host "`nPress any key to exit..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")