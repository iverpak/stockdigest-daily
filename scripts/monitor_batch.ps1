# QuantBrief - Monitor Batch Script
# Reconnect to a running batch and watch progress in real-time

# Configuration
$APP = "https://quantbrief-daily.onrender.com"
$TOKEN = "a77774hhwef88f99sd9g883h23nsndfs9d8cnns9adh7asc9xcibjweorn"

$headers = @{ "X-Admin-Token" = $TOKEN; "Content-Type" = "application/json" }

Write-Host "=== QUANTBRIEF BATCH MONITORING ===" -ForegroundColor Cyan

# Try to find batch_id
$logPath = "$PSScriptRoot\..\last_batch_id.txt"
$batch_id = $null

# Option 1: Read from saved file
if (Test-Path $logPath) {
    try {
        $batch_id = Get-Content $logPath -ErrorAction Stop
        Write-Host "`nFound saved batch_id: $($batch_id.Substring(0,13))..." -ForegroundColor Green
    } catch {
        # File exists but couldn't read
    }
}

# Option 2: Manual entry
if (-not $batch_id) {
    Write-Host "`nNo saved batch_id found." -ForegroundColor Yellow
    Write-Host "Enter batch_id to monitor: " -ForegroundColor Cyan -NoNewline
    $userInput = Read-Host

    if (-not $userInput) {
        Write-Host "`n❌ No batch_id provided. Exiting." -ForegroundColor Red
        Write-Host "`nPress any key to exit..." -ForegroundColor White
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 1
    }

    $batch_id = $userInput.Trim()
}

Write-Host "`n=== MONITORING PROGRESS ===" -ForegroundColor Cyan
Write-Host "Batch ID: $batch_id" -ForegroundColor Gray
Write-Host "Press Ctrl+C to stop monitoring (jobs will continue running)" -ForegroundColor Yellow
Write-Host ""

$maxPolls = 90  # 90 polls × 20s = 30 minutes max
$pollInterval = 20  # Poll every 20 seconds

try {
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
                Write-Host (" | Current: $($current.ticker) [$phase_display]" + " " * 20) -NoNewline -ForegroundColor Yellow
            } else {
                # Clear the "Current:" part if no job is processing
                Write-Host (" " * 40) -NoNewline
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
            $errorMsg = $_.Exception.Message

            if ($errorMsg -match "404") {
                Write-Host "`n`n❌ Batch not found. It may have been deleted or completed long ago." -ForegroundColor Red
                break
            } else {
                Write-Host "`n  ⚠️ Status check failed: $errorMsg" -ForegroundColor Yellow
            }
        }
    }

    if ($i -eq $maxPolls) {
        Write-Host "`n`n  ⏰ Timeout reached after $([math]::Round($maxPolls * $pollInterval / 60, 1)) minutes" -ForegroundColor Yellow
        Write-Host "     Processing continues server-side. Run this script again to reconnect." -ForegroundColor Gray
    }

    # Show detailed results
    Write-Host "`n`n=== DETAILED RESULTS ===" -ForegroundColor Cyan

    try {
        $finalStatus = Invoke-RestMethod -Method Get "$APP/jobs/batch/$batch_id" -Headers $headers

        foreach ($job in $finalStatus.jobs) {
            $statusSymbol = switch ($job.status) {
                "completed" { "✅"; break }
                "failed" { "❌"; break }
                "timeout" { "⏰"; break }
                "processing" { "🔄"; break }
                "queued" { "⏳"; break }
                "cancelled" { "🚫"; break }
                default { "❓" }
            }

            $color = switch ($job.status) {
                "completed" { "Green"; break }
                "failed" { "Red"; break }
                "timeout" { "Red"; break }
                "processing" { "Yellow"; break }
                "queued" { "Gray"; break }
                "cancelled" { "Yellow"; break }
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

        if ($finalStatus.completed -gt 0) {
            Write-Host "  Success Rate: $([math]::Round($finalStatus.completed / $finalStatus.total_tickers * 100, 1))%" -ForegroundColor Cyan
        }

    } catch {
        Write-Host "  ⚠️ Could not fetch final results: $($_.Exception.Message)" -ForegroundColor Yellow
    }

} catch {
    Write-Host "`n`n🚨 ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n=== MONITORING COMPLETE ===" -ForegroundColor Green
Write-Host "💡 To cancel this batch: .\scripts\cancel_all_jobs.ps1" -ForegroundColor Gray

# Pause to read results
Write-Host "`n🔍 Press any key to close..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")