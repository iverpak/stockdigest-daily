# IMPROVED SETUP.PS1 SNIPPET - ADD THIS TO YOUR SCRIPT

# Add this variable at the top
$incremental_commits = @()

# Replace the ticker processing section (around line 77-181) with this:
foreach ($ticker in $TICKERS) {
    Write-Host "`n[$ticker] Processing (batch_size=$BATCH_SIZE, triage_batch_size=$TRIAGE_BATCH_SIZE)..." -ForegroundColor Yellow

    try {
        # MAIN INGEST WITH BOTH BATCH SIZES
        Write-Host "  Starting ingest for $ticker with async triage batching..." -ForegroundColor Gray
        $ingest = Invoke-RestMethod -Method Post "$APP/cron/ingest?minutes=$MINUTES&tickers=$ticker&batch_size=$BATCH_SIZE&triage_batch_size=$TRIAGE_BATCH_SIZE" -Headers $headers -TimeoutSec 1800

        # ... (keep all your existing logging/stats code) ...

        # DIGEST (for this ticker only)
        Write-Host "  Starting digest for $ticker..." -ForegroundColor Gray
        $digest = Invoke-RestMethod -Method Post "$APP/cron/digest?minutes=$MINUTES&tickers=$ticker" -Headers $headers -TimeoutSec 900

        if ($digest.status -eq "sent") {
            Write-Host "  DIGEST SUCCESS: $($digest.articles) articles sent" -ForegroundColor Green
        } else {
            Write-Host "  DIGEST RESULT: $($digest.message)" -ForegroundColor Yellow
        }

        # *** NEW: INCREMENTAL COMMIT AFTER EACH SUCCESSFUL TICKER ***
        if ($ingest.status -eq "completed" -or $ingested -gt 0) {
            Write-Host "  Attempting incremental commit for $ticker..." -ForegroundColor Green

            try {
                $commitBody = @{ tickers = @($ticker) } | ConvertTo-Json
                $commitResult = Invoke-RestMethod -Method Post "$APP/admin/safe-incremental-commit" -Headers $headers -Body $commitBody -TimeoutSec 180

                if ($commitResult.status -eq "success") {
                    Write-Host "  INCREMENTAL COMMIT SUCCESS: $ticker committed to GitHub" -ForegroundColor Green
                    $incremental_commits += $ticker
                } else {
                    Write-Host "  INCREMENTAL COMMIT SKIPPED: $($commitResult.message)" -ForegroundColor Yellow
                }
            } catch {
                Write-Host "  INCREMENTAL COMMIT FAILED: $($_.Exception.Message)" -ForegroundColor Red
                # Continue processing other tickers even if commit fails
            }

            $all_processed_tickers += $ticker
            Write-Host "  $ticker PROCESSING SUCCESS" -ForegroundColor Green
        } else {
            Write-Host "  $ticker PROCESSING FAILED" -ForegroundColor Red
        }

        # Brief pause between tickers to prevent overlap
        Write-Host "  $ticker completed. Pausing before next ticker..." -ForegroundColor Gray
        Start-Sleep -Seconds 3

    } catch {
        Write-Host "  ERROR: $($_.Exception.Message)" -ForegroundColor Red

        if ($_.Exception.Response) {
            Write-Host "    HTTP Status: $($_.Exception.Response.StatusCode)" -ForegroundColor Red
        }

        # *** NEW: STILL ATTEMPT COMMIT FOR PARTIAL SUCCESS ***
        if ($ingested -gt 0) {
            Write-Host "  Attempting emergency commit due to partial success..." -ForegroundColor Yellow
            try {
                $commitBody = @{ tickers = @($ticker) } | ConvertTo-Json
                $commitResult = Invoke-RestMethod -Method Post "$APP/admin/safe-incremental-commit" -Headers $headers -Body $commitBody -TimeoutSec 120

                if ($commitResult.status -eq "success") {
                    Write-Host "  EMERGENCY COMMIT SUCCESS: $ticker data saved!" -ForegroundColor Green
                    $incremental_commits += $ticker
                }
            } catch {
                Write-Host "  Emergency commit also failed: $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
}

# Replace the final GitHub commit section with this enhanced summary:
Write-Host "`n=== ENHANCED PROCESSING SUMMARY ===" -ForegroundColor Cyan
Write-Host "  Successfully processed tickers: $($all_processed_tickers -join ',')" -ForegroundColor Green
Write-Host "  Incrementally committed tickers: $($incremental_commits -join ',')" -ForegroundColor Green
Write-Host "  Total incremental commits: $($incremental_commits.Count)" -ForegroundColor Green

if ($incremental_commits.Count -gt 0) {
    Write-Host "`n✅ METADATA PRESERVED: $($incremental_commits.Count) tickers committed individually" -ForegroundColor Green
    Write-Host "   Your AI-generated metadata is safe in GitHub!" -ForegroundColor Green
} else {
    Write-Host "`n⚠️  NO METADATA COMMITTED: All tickers failed or had no AI enhancements" -ForegroundColor Yellow
}

Write-Host "`n=== INCREMENTAL COMMIT STRATEGY COMPLETED ===" -ForegroundColor Green
Write-Host "✅ No more all-or-nothing commits!" -ForegroundColor Green
Write-Host "✅ Each successful ticker saves immediately!" -ForegroundColor Green
Write-Host "✅ Network timeouts won't lose your work!" -ForegroundColor Green