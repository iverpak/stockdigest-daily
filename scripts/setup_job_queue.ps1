# StockDigest Job Queue - PowerShell Client
# Uses server-side job queue for reliable, resumable processing

# Configuration
$APP = "https://stockdigest.app"
$TOKEN = "a77774hhwef88f99sd9g883h23nsndfs9d8cnns9adh7asc9xcibjweorn"
$TICKERS = @("RY.TO", "TD.TO", "VST", "CEG")
$MINUTES = 4320  # Time window in minutes
$BATCH_SIZE = 5  # Scraping batch size
$TRIAGE_BATCH_SIZE = 3  # Triage batch size

$headers = @{ "X-Admin-Token" = $TOKEN; "Content-Type" = "application/json" }

Write-Host "=== STOCKDIGEST JOB QUEUE PROCESSING ===" -ForegroundColor Cyan
Write-Host "Window: $MINUTES min | Scraping Batch: $BATCH_SIZE | Triage Batch: $TRIAGE_BATCH_SIZE" -ForegroundColor Yellow
Write-Host "Tickers: $($TICKERS -join ', ')" -ForegroundColor Yellow

# ===== STEP 1: INITIALIZE TICKERS (ONE-TIME) =====
Write-Host "`n=== STEP 1: INDIVIDUAL TICKER INITIALIZATION ===" -ForegroundColor Cyan

try {
    # Clean and reset for ALL tickers at once
    $cleanBody = @{ tickers = $TICKERS } | ConvertTo-Json
    Invoke-RestMethod -Method Post "$APP/admin/clean-feeds" -Headers $headers -Body $cleanBody | Out-Null
    Write-Host "  Global clean-feeds: SUCCESS" -ForegroundColor Green

    $resetBody = @{ tickers = $TICKERS } | ConvertTo-Json
    Invoke-RestMethod -Method Post "$APP/admin/reset-digest-flags" -Headers $headers -Body $resetBody | Out-Null
    Write-Host "  Global reset-digest-flags: SUCCESS" -ForegroundColor Green

    # Initialize each ticker separately to prevent corruption
    $totalFeedsCreated = 0
    foreach ($singleTicker in $TICKERS) {
        Write-Host "  Initializing $singleTicker..." -ForegroundColor Yellow

        try {
            $singleInitBody = @{ tickers = @($singleTicker); force_refresh = $false} | ConvertTo-Json
            $singleInitResult = Invoke-RestMethod -Method Post "$APP/admin/init" -Headers $headers -Body $singleInitBody

            if ($singleInitResult.results -and $singleInitResult.results.Count -gt 0) {
                $feedsForTicker = ($singleInitResult.results | Measure-Object -Property feeds_created -Sum).Sum
                if ($feedsForTicker -gt 0) {
                    $totalFeedsCreated += $feedsForTicker
                    Write-Host "    ${singleTicker}: SUCCESS - $feedsForTicker feeds" -ForegroundColor Green
                } else {
                    Write-Host "    ${singleTicker}: No new feeds needed" -ForegroundColor Cyan
                }
            }

            Start-Sleep -Seconds 2

        } catch {
            Write-Host "    ${singleTicker}: FAILED - $($_.Exception.Message)" -ForegroundColor Red
            throw "Failed to initialize $singleTicker"
        }
    }

    Write-Host "  ✅ Initialization complete: $totalFeedsCreated feeds created" -ForegroundColor Green

} catch {
    Write-Host "  🚨 INITIALIZATION FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  ❌ Cannot continue without proper initialization" -ForegroundColor Red
    Write-Host "`n  Press any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# ===== STEP 2: SUBMIT JOB BATCH =====
Write-Host "`n=== STEP 2: SUBMITTING JOB BATCH ===" -ForegroundColor Cyan

try {
    $submitBody = @{
        tickers = $TICKERS
        minutes = $MINUTES
        batch_size = $BATCH_SIZE
        triage_batch_size = $TRIAGE_BATCH_SIZE
    } | ConvertTo-Json

    $batch = Invoke-RestMethod -Method Post "$APP/jobs/submit" -Headers $headers -Body $submitBody

    $batch_id = $batch.batch_id

    # Save batch_id for easy cancellation (try to write, but don't fail if permission denied)
    try {
        $batch_id | Out-File "$PSScriptRoot\..\last_batch_id.txt" -Encoding UTF8 -ErrorAction Stop
        Write-Host "  💾 Batch ID saved to last_batch_id.txt" -ForegroundColor Gray
    } catch {
        Write-Host "  ⚠️ Could not save batch_id.txt (permission issue)" -ForegroundColor Yellow
    }

    Write-Host "  ✅ Batch submitted: $batch_id" -ForegroundColor Green
    Write-Host "  Processing $($batch.total_jobs) tickers server-side..." -ForegroundColor Yellow
    Write-Host "  💡 To cancel: .\scripts\cancel_all_jobs.ps1" -ForegroundColor Gray

} catch {
    Write-Host "  🚨 JOB SUBMISSION FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "`n  Press any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# ===== STEP 3: POLL FOR COMPLETION =====
Write-Host "`n=== STEP 3: MONITORING PROGRESS ===" -ForegroundColor Cyan

$maxPolls = 90  # 90 polls × 20s = 30 minutes max
$pollInterval = 20  # Poll every 20 seconds

for ($i = 0; $i -lt $maxPolls; $i++) {
    Start-Sleep -Seconds $pollInterval

    try {
        $status = Invoke-RestMethod -Method Get "$APP/jobs/batch/$batch_id" -Headers $headers

        # Clear line and show progress
        Write-Host "`r  Progress: $($status.overall_progress)% | Completed: $($status.completed)/$($status.total_tickers) | Failed: $($status.failed) | Processing: $($status.processing)" -NoNewline -ForegroundColor Cyan

        # Show current ticker being processed
        $current = $status.jobs | Where-Object { $_.status -eq 'processing' } | Select-Object -First 1
        if ($current) {
            $phase_display = if ($current.phase) { $current.phase } else { "starting" }
            Write-Host " | Current: $($current.ticker) [$phase_display]" -NoNewline -ForegroundColor Yellow
        }

        # Check if all done
        if ($status.completed + $status.failed -eq $status.total_tickers) {
            Write-Host ""
            if ($status.failed -eq 0) {
                Write-Host "`n  ✅ BATCH COMPLETE: All $($status.completed) tickers succeeded!" -ForegroundColor Green
            } else {
                Write-Host "`n  ⚠️ BATCH COMPLETE: $($status.completed) succeeded, $($status.failed) failed" -ForegroundColor Yellow
            }
            break
        }

    } catch {
        Write-Host "`n  ⚠️ Status check failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

if ($i -eq $maxPolls) {
    Write-Host "`n  ⏰ Timeout reached after $([math]::Round($maxPolls * $pollInterval / 60, 1)) minutes" -ForegroundColor Yellow
    Write-Host "     Processing continues server-side. Check status at:" -ForegroundColor Gray
    Write-Host "     $APP/jobs/batch/$batch_id" -ForegroundColor Cyan
}

# ===== STEP 4: SHOW DETAILED RESULTS =====
Write-Host "`n=== STEP 4: DETAILED RESULTS ===" -ForegroundColor Cyan

try {
    $finalStatus = Invoke-RestMethod -Method Get "$APP/jobs/batch/$batch_id" -Headers $headers

    foreach ($job in $finalStatus.jobs) {
        $statusSymbol = switch ($job.status) {
            "completed" { "✅"; break }
            "failed" { "❌"; break }
            "timeout" { "⏰"; break }
            "processing" { "🔄"; break }
            "queued" { "⏳"; break }
            default { "❓" }
        }

        $color = switch ($job.status) {
            "completed" { "Green"; break }
            "failed" { "Red"; break }
            "timeout" { "Red"; break }
            "processing" { "Yellow"; break }
            "queued" { "Gray"; break }
            default { "White" }
        }

        $duration = if ($job.duration_seconds) { " ($([math]::Round($job.duration_seconds / 60, 1))min)" } else { "" }
        $memory = if ($job.memory_mb) { " [mem: $([math]::Round($job.memory_mb, 0))MB]" } else { "" }

        Write-Host "  $statusSymbol $($job.ticker): $($job.status)$duration$memory" -ForegroundColor $color

        if ($job.error_message) {
            Write-Host "      Error: $($job.error_message)" -ForegroundColor Red
        }
    }

    # Summary stats
    Write-Host "`n=== SUMMARY ===" -ForegroundColor Cyan
    Write-Host "  Total: $($finalStatus.total_tickers) tickers" -ForegroundColor White
    Write-Host "  Completed: $($finalStatus.completed)" -ForegroundColor Green
    Write-Host "  Failed: $($finalStatus.failed)" -ForegroundColor $(if ($finalStatus.failed -gt 0) { "Red" } else { "Gray" })
    Write-Host "  Processing: $($finalStatus.processing)" -ForegroundColor Yellow
    Write-Host "  Queued: $($finalStatus.queued)" -ForegroundColor Gray
    Write-Host "  Success Rate: $([math]::Round($finalStatus.completed / $finalStatus.total_tickers * 100, 1))%" -ForegroundColor Cyan

    # Check if batch is actually complete
    $stillRunning = $finalStatus.processing + $finalStatus.queued
    $allComplete = ($stillRunning -eq 0)

} catch {
    Write-Host "  ⚠️ Could not fetch final results: $($_.Exception.Message)" -ForegroundColor Yellow
    $allComplete = $false
}

# ===== STEP 5: JOB QUEUE STATS =====
Write-Host "`n=== JOB QUEUE STATS ===" -ForegroundColor Cyan

try {
    $stats = Invoke-RestMethod -Method Get "$APP/jobs/stats" -Headers $headers

    Write-Host "  Worker ID: $($stats.worker_id)" -ForegroundColor Gray
    Write-Host "  Circuit Breaker: $($stats.circuit_breaker_state)" -ForegroundColor $(if ($stats.circuit_breaker_state -eq 'open') { "Red" } else { "Green" })

    if ($stats.avg_duration_seconds) {
        Write-Host "  Avg Duration: $([math]::Round($stats.avg_duration_seconds / 60, 1)) minutes" -ForegroundColor Cyan
    }

    if ($stats.avg_memory_mb) {
        Write-Host "  Avg Memory: $([math]::Round($stats.avg_memory_mb, 0)) MB" -ForegroundColor Cyan
    }

} catch {
    Write-Host "  ⚠️ Could not fetch stats" -ForegroundColor Yellow
}

# Display accurate completion status
if ($allComplete) {
    Write-Host "`n=== PROCESSING COMPLETE ===" -ForegroundColor Green
    Write-Host "✅ All operations complete using server-side job queue!" -ForegroundColor Green
    Write-Host "   No more 520 errors - processing is fully decoupled from HTTP!" -ForegroundColor Cyan
} else {
    Write-Host "`n=== MONITORING STOPPED (JOBS STILL RUNNING) ===" -ForegroundColor Yellow
    Write-Host "⚠️ Some jobs are still processing server-side" -ForegroundColor Yellow
    Write-Host "   Check status at: $APP/jobs/batch/$batch_id" -ForegroundColor Cyan
    Write-Host "   Processing continues in background - no action needed!" -ForegroundColor Gray
}

# Pause to read results
Write-Host "`n🔍 Press any key to close..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
