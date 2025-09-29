# QuantBrief - Cancel All Jobs Script
# Cancels jobs by reading batch_id from the output file or prompting for manual entry

# Configuration
$APP = "https://quantbrief-daily.onrender.com"
$TOKEN = "a77774hhwef88f99sd9g883h23nsndfs9d8cnns9adh7asc9xcibjweorn"

$headers = @{ "X-Admin-Token" = $TOKEN; "Content-Type" = "application/json" }

Write-Host "=== QUANTBRIEF JOB CANCELLATION ===" -ForegroundColor Red

# Option 1: Try to find batch_id from recent log file
$logPath = "$PSScriptRoot\..\last_batch_id.txt"
$batch_id = $null

if (Test-Path $logPath) {
    try {
        $batch_id = Get-Content $logPath -ErrorAction Stop
        Write-Host "`nFound saved batch_id: $($batch_id.Substring(0,13))..." -ForegroundColor Cyan
    } catch {
        # File exists but couldn't read
    }
}

# Option 2: Manual entry
if (-not $batch_id) {
    Write-Host "`nNo saved batch_id found." -ForegroundColor Yellow
    Write-Host "Enter batch_id to cancel (or press Enter to cancel ALL recent batches): " -ForegroundColor Cyan -NoNewline
    $userInput = Read-Host

    if ($userInput) {
        $batch_id = $userInput.Trim()
    }
}

# If we have a specific batch_id, cancel it
if ($batch_id) {
    Write-Host "`n=== CANCELLING BATCH ===" -ForegroundColor Yellow
    Write-Host "Batch ID: $batch_id" -ForegroundColor Gray

    try {
        # First, check batch status
        $status = Invoke-RestMethod -Method Get "$APP/jobs/batch/$batch_id" -Headers $headers

        $activeTickers = $status.jobs | Where-Object { $_.status -in @('queued', 'processing') }

        if ($activeTickers.Count -eq 0) {
            Write-Host "`n✅ No active jobs to cancel in this batch!" -ForegroundColor Green
            Write-Host "   All jobs are already completed/failed/cancelled." -ForegroundColor Gray
            Write-Host "`nPress any key to exit..." -ForegroundColor White
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
            exit 0
        }

        Write-Host "`nActive jobs in this batch:" -ForegroundColor Yellow
        foreach ($job in $activeTickers) {
            $statusDisplay = $job.status
            if ($job.phase) { $statusDisplay += " [$($job.phase)]" }
            Write-Host "  - $($job.ticker): $statusDisplay" -ForegroundColor Gray
        }

        # Confirm cancellation
        Write-Host "`n⚠️  Cancel $($activeTickers.Count) job(s)? Press 'Y' to confirm..." -ForegroundColor Yellow
        $confirmation = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        Write-Host ""

        if ($confirmation.Character -ne 'Y' -and $confirmation.Character -ne 'y') {
            Write-Host "`n❌ Cancellation aborted" -ForegroundColor Yellow
            Write-Host "`nPress any key to exit..." -ForegroundColor White
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
            exit 0
        }

        # Cancel the batch
        Write-Host "`nCancelling batch..." -ForegroundColor Cyan
        $result = Invoke-RestMethod -Method Post "$APP/jobs/batch/$batch_id/cancel" -Headers $headers

        if ($result.status -eq "cancelled") {
            Write-Host "`n✅ SUCCESS: Cancelled $($result.jobs_cancelled) job(s)" -ForegroundColor Green

            if ($result.tickers) {
                Write-Host "   Tickers: $($result.tickers -join ', ')" -ForegroundColor Gray
            }

            # Clean up saved batch_id
            if (Test-Path $logPath) {
                Remove-Item $logPath -ErrorAction SilentlyContinue
            }

        } elseif ($result.status -eq "no_jobs_to_cancel") {
            Write-Host "`n✅ Batch already completed, no jobs to cancel" -ForegroundColor Green
        }

    } catch {
        $errorMsg = $_.Exception.Message

        if ($errorMsg -match "404") {
            Write-Host "`n❌ Batch not found. It may have been deleted or completed long ago." -ForegroundColor Red
        } elseif ($errorMsg -match "400") {
            Write-Host "`n❌ Cannot cancel: Jobs may already be completed." -ForegroundColor Red
        } else {
            Write-Host "`n❌ ERROR: $errorMsg" -ForegroundColor Red
        }
    }

} else {
    # No batch_id provided - give instructions
    Write-Host "`n📋 HOW TO CANCEL JOBS:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Option 1: Get batch_id from PowerShell output when you submit jobs" -ForegroundColor Yellow
    Write-Host "   Look for: 'Batch submitted: 550e8400-e29b-41d4-a716-446655440000'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Option 2: Check Render logs for:" -ForegroundColor Yellow
    Write-Host "   📦 Batch {batch_id} created with {N} jobs" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Then run this script again and paste the batch_id when prompted." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "OR cancel via PowerShell directly:" -ForegroundColor Cyan
    Write-Host '   $batch_id = "YOUR-BATCH-ID"' -ForegroundColor Gray
    Write-Host '   Invoke-RestMethod -Method Post "$APP/jobs/batch/$batch_id/cancel" -Headers $headers' -ForegroundColor Gray
}

Write-Host "`n✅ Script complete!" -ForegroundColor Green
Write-Host "`nPress any key to exit..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")