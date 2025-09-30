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

# Option 1: Check for active batches
Write-Host "`nSearching for active batches..." -ForegroundColor Yellow

try {
    $activeBatches = Invoke-RestMethod -Method Get "$APP/jobs/active-batches" -Headers $headers

    if ($activeBatches.active_batches -gt 0) {
        Write-Host "Found $($activeBatches.active_batches) active batch(es):" -ForegroundColor Green
        Write-Host ""

        # Show all active batches
        $batchList = @()
        $index = 1
        foreach ($batch in $activeBatches.batches) {
            $batchIdShort = $batch.batch_id.Substring(0, 8)
            $activeJobs = $batch.jobs | Where-Object { $_.status -in @('queued', 'processing') }

            Write-Host "  [$index] Batch: $batchIdShort..." -ForegroundColor Cyan
            Write-Host "      Completed: $($batch.completed_jobs)/$($batch.total_jobs) | Processing: $($activeJobs.Count)" -ForegroundColor Gray

            if ($activeJobs) {
                $tickers = ($activeJobs | Select-Object -First 3 | ForEach-Object { $_.ticker }) -join ", "
                if ($activeJobs.Count -gt 3) { $tickers += "..." }
                Write-Host "      Tickers: $tickers" -ForegroundColor Gray
            }
            Write-Host ""

            $batchList += $batch
            $index++
        }

        # If only one batch, auto-select it
        if ($activeBatches.active_batches -eq 1) {
            $batch_id = $batchList[0].batch_id
            Write-Host "Auto-selecting the only active batch..." -ForegroundColor Green
        } else {
            # Let user choose
            Write-Host "Enter batch number to monitor [1-$($batchList.Count)], or press Enter to monitor all: " -ForegroundColor Yellow -NoNewline
            $selection = Read-Host

            if ($selection -and $selection -match '^\d+$' -and [int]$selection -ge 1 -and [int]$selection -le $batchList.Count) {
                $batch_id = $batchList[[int]$selection - 1].batch_id
                Write-Host "Selected batch: $($batch_id.Substring(0, 13))..." -ForegroundColor Green
            } elseif (-not $selection) {
                Write-Host "Monitoring ALL active batches..." -ForegroundColor Green
                # Monitor all batches mode
                $monitorAll = $true
            } else {
                Write-Host "`n❌ Invalid selection" -ForegroundColor Red
                Write-Host "`nPress any key to exit..." -ForegroundColor White
                $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
                exit 1
            }
        }
    } else {
        Write-Host "No active batches found." -ForegroundColor Gray
    }
} catch {
    Write-Host "⚠️ Could not fetch active batches: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Option 2: Read from saved file if no batch selected yet
if (-not $batch_id -and -not $monitorAll) {
    if (Test-Path $logPath) {
        try {
            $batch_id = (Get-Content $logPath -ErrorAction Stop).Trim()
            if ($batch_id) {
                Write-Host "`nFound saved batch_id: $($batch_id.Substring(0,13))..." -ForegroundColor Green
            }
        } catch {
            # File read failed
        }
    }
}

# Option 3: Manual entry
if (-not $batch_id -and -not $monitorAll) {
    Write-Host "`nEnter batch_id to monitor: " -ForegroundColor Cyan -NoNewline
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

if ($monitorAll) {
    Write-Host "Monitoring ALL active batches" -ForegroundColor Gray
} else {
    Write-Host "Batch ID: $batch_id" -ForegroundColor Gray
}

Write-Host "Press Ctrl+C to stop monitoring (jobs will continue running)" -ForegroundColor Yellow
Write-Host ""

$maxPolls = 90  # 90 polls × 20s = 30 minutes max
$pollInterval = 20  # Poll every 20 seconds

try {
    for ($i = 0; $i -lt $maxPolls; $i++) {
        Start-Sleep -Seconds $pollInterval

        try {
            if ($monitorAll) {
                # Monitor all active batches
                $activeBatches = Invoke-RestMethod -Method Get "$APP/jobs/active-batches" -Headers $headers

                if ($activeBatches.active_batches -eq 0) {
                    Write-Host "`n`n  ✅ ALL BATCHES COMPLETE!" -ForegroundColor Green
                    break
                }

                # Clear and redraw
                $output = "`r"
                $totalCompleted = 0
                $totalJobs = 0
                $totalFailed = 0

                foreach ($batch in $activeBatches.batches) {
                    $totalCompleted += $batch.completed_jobs
                    $totalJobs += $batch.total_jobs
                    $totalFailed += $batch.failed_jobs
                }

                $totalProcessing = $totalJobs - $totalCompleted - $totalFailed
                $overallProgress = if ($totalJobs -gt 0) { [math]::Round(($totalCompleted / $totalJobs) * 100) } else { 0 }

                Write-Host "`r  Progress: $overallProgress% | Total: $totalCompleted/$totalJobs | Failed: $totalFailed | Active Batches: $($activeBatches.active_batches)" -NoNewline -ForegroundColor Cyan

            } else {
                # Monitor single batch
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

    if ($monitorAll) {
        # Show results for all batches
        try {
            $activeBatches = Invoke-RestMethod -Method Get "$APP/jobs/active-batches" -Headers $headers

            foreach ($batch in $activeBatches.batches) {
                $batchIdShort = $batch.batch_id.Substring(0, 8)
                Write-Host "`nBatch $batchIdShort:" -ForegroundColor Cyan

                foreach ($job in $batch.jobs) {
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

                    $progressDisplay = if ($job.progress) { " $($job.progress)%" } else { "" }
                    Write-Host "  $statusSymbol $($job.ticker): $($job.status)$progressDisplay" -ForegroundColor $color
                }
            }
        } catch {
            Write-Host "  ⚠️ Could not fetch final results: $($_.Exception.Message)" -ForegroundColor Yellow
        }

    } else {
        # Show results for single batch
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
    }

} catch {
    Write-Host "`n`n🚨 ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n=== MONITORING COMPLETE ===" -ForegroundColor Green
Write-Host "💡 To cancel jobs: .\scripts\cancel_all_jobs.ps1" -ForegroundColor Gray

# Pause to read results
Write-Host "`n🔍 Press any key to close..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")