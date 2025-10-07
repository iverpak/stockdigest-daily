# Configuration
$APP = "https://stockdigest-daily.onrender.com"
$TOKEN = "a77774hhwef88f99sd9g883h23nsndfs9d8cnns9adh7asc9xcibjweorn"
$TICKERS = @("RY.TO", "TD.TO", "VST", "CEG")  # Updated to test new architecture with problematic ticker combinations
$MINUTES = 4320  # Time window in minutes
$BATCH_SIZE = 3  # Scraping batch size
$TRIAGE_BATCH_SIZE = 3  # NEW: Triage batch size for async processing
$RESET_AI = $false  # Reset AI scores before running - DISABLED

$headers = @{ "X-Admin-Token" = $TOKEN; "Content-Type" = "application/json" }

Write-Host "=== QUANTBRIEF NEW ARCHITECTURE: INITIALIZE ONCE, PROCESS SEQUENTIALLY ===" -ForegroundColor Cyan
Write-Host "Window: $MINUTES min | Scraping Batch: $BATCH_SIZE | Triage Batch: $TRIAGE_BATCH_SIZE | Reset: $RESET_AI | Tickers: $($TICKERS -join ',')" -ForegroundColor Yellow

$all_processed_tickers = @()
$incremental_commits = @()

# ===== CRITICAL FIX: INITIALIZE EACH TICKER SEPARATELY =====
Write-Host "`n=== INDIVIDUAL TICKER INITIALIZATION (PREVENTS CORRUPTION) ===" -ForegroundColor Cyan

try {
    # Clean and reset for ALL tickers at once (this is safe)
    $cleanBody = @{ tickers = $TICKERS } | ConvertTo-Json
    Invoke-RestMethod -Method Post "$APP/admin/clean-feeds" -Headers $headers -Body $cleanBody | Out-Null
    Write-Host "  Global clean-feeds: SUCCESS" -ForegroundColor Green
    
    $resetBody = @{ tickers = $TICKERS } | ConvertTo-Json
    Invoke-RestMethod -Method Post "$APP/admin/reset-digest-flags" -Headers $headers -Body $resetBody | Out-Null
    Write-Host "  Global reset-digest-flags: SUCCESS" -ForegroundColor Green
    
    # CRITICAL: Initialize each ticker SEPARATELY to prevent corruption
    $totalFeedsCreated = 0
    foreach ($singleTicker in $TICKERS) {
        Write-Host "  Initializing $singleTicker individually..." -ForegroundColor Yellow
        
        try {
            $singleInitBody = @{ tickers = @($singleTicker); force_refresh = $false} | ConvertTo-Json
            $singleInitResult = Invoke-RestMethod -Method Post "$APP/admin/init" -Headers $headers -Body $singleInitBody
            
            # NEW ARCHITECTURE: Check results array for feeds_created
            if ($singleInitResult.results -and $singleInitResult.results.Count -gt 0) {
                $feedsForTicker = ($singleInitResult.results | Measure-Object -Property feeds_created -Sum).Sum
                if ($feedsForTicker -gt 0) {
                    $totalFeedsCreated += $feedsForTicker
                    Write-Host "    ${singleTicker}: SUCCESS - $feedsForTicker feeds created (NEW ARCHITECTURE)" -ForegroundColor Green
                } else {
                    Write-Host "    ${singleTicker}: No new feeds needed" -ForegroundColor Cyan
                }
            } elseif ($singleInitResult.successful -gt 0) {
                # Fallback: Use successful count if results format changes
                Write-Host "    ${singleTicker}: SUCCESS - New architecture working" -ForegroundColor Green
                $totalFeedsCreated += 1  # Approximate count
            } else {
                Write-Host "    ${singleTicker}: No new feeds needed" -ForegroundColor Cyan
            }
            
            # Brief pause between ticker initializations
            Start-Sleep -Seconds 2
            
        } catch {
            Write-Host "    ${singleTicker}: FAILED - $($_.Exception.Message)" -ForegroundColor Red
            throw "Failed to initialize $singleTicker"
        }
    }
    
    Write-Host "  Individual initialization: SUCCESS - $totalFeedsCreated total feeds created" -ForegroundColor Green
    
    # Show GitHub sync results (should be same for all calls)
    if ($singleInitResult.github_sync) {
        $github_sync = $singleInitResult.github_sync
        if ($github_sync.status -eq "success") {
            $imported = if ($github_sync.database_import.imported) { $github_sync.database_import.imported } else { 0 }
            $updated = if ($github_sync.database_import.updated) { $github_sync.database_import.updated } else { 0 }
            Write-Host "  GitHub Import: SUCCESS - Imported($imported) Updated($updated)" -ForegroundColor Green
        }
    }
    
} catch {
    Write-Host "  🚨 INDIVIDUAL INITIALIZATION FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  🔍 FULL ERROR DETAILS:" -ForegroundColor Yellow
    Write-Host "     Status Code: $($_.Exception.Response.StatusCode)" -ForegroundColor Red
    Write-Host "     Response: $($_.Exception.Response)" -ForegroundColor Red
    Write-Host "     Inner Exception: $($_.Exception.InnerException)" -ForegroundColor Red
    Write-Host "  ❌ Cannot continue without proper initialization" -ForegroundColor Red
    Write-Host "  ⏸️  Script will pause so you can read this error..." -ForegroundColor Yellow
    Write-Host "     Press any key to exit..." -ForegroundColor White
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# ===== NOW PROCESS EACH TICKER SEQUENTIALLY (NO MORE ADMIN CALLS) =====
Write-Host "`n=== SEQUENTIAL TICKER PROCESSING ===" -ForegroundColor Cyan

foreach ($ticker in $TICKERS) {
    Write-Host "`n[$ticker] Processing (batch_size=$BATCH_SIZE, triage_batch_size=$TRIAGE_BATCH_SIZE)..." -ForegroundColor Yellow
  
    try {
        # MEMORY CHECK: Before ingest
        Write-Host "  Memory Check: Before ingest..." -ForegroundColor Gray
        try {
            $memoryBefore = Invoke-RestMethod -Uri "$APP/admin/memory" -Method Get -Headers $headers
            Write-Host "  Memory Before: $([math]::Round($memoryBefore.memory_mb, 1)) MB" -ForegroundColor Cyan
        } catch {
            Write-Host "  Memory Check: Could not check memory" -ForegroundColor Yellow
        }
      
        # MAIN INGEST WITH BOTH BATCH SIZES
        Write-Host "  Starting ingest for $ticker with async triage batching..." -ForegroundColor Gray
        $ingest = Invoke-RestMethod -Method Post "$APP/cron/ingest?minutes=$MINUTES&tickers=$ticker&batch_size=$BATCH_SIZE&triage_batch_size=$TRIAGE_BATCH_SIZE" -Headers $headers -TimeoutSec 1800
        
        # MEMORY CHECK: After ingest
        Write-Host "  Memory Check: After ingest..." -ForegroundColor Gray
        try {
            $memoryAfter = Invoke-RestMethod -Uri "$APP/admin/memory" -Method Get -Headers $headers
            $memoryUsed = [math]::Round($memoryAfter.memory_mb - $memoryBefore.memory_mb, 1)
            Write-Host "  Memory After: $([math]::Round($memoryAfter.memory_mb, 1)) MB (Used: +$memoryUsed MB)" -ForegroundColor Cyan
            
            if ($memoryAfter.memory_mb -gt 800) {
                Write-Host "  WARNING: High memory usage! Forcing cleanup..." -ForegroundColor Red
                try {
                    $cleanupResult = Invoke-RestMethod -Uri "$APP/admin/force-cleanup" -Method Post -Headers $headers
                    Write-Host "  Cleanup: SUCCESS" -ForegroundColor Green
                } catch {
                    Write-Host "  Cleanup failed: $($_.Exception.Message)" -ForegroundColor Red
                }
            }
        } catch {
            Write-Host "  Memory Check: Could not check memory after ingest" -ForegroundColor Yellow
        }
        
        # Phase 1: Enhanced ingestion stats
        $ingested = if ($ingest.phase_1_ingest -and $ingest.phase_1_ingest.total_inserted) { $ingest.phase_1_ingest.total_inserted } else { 0 }
        $total_in_timeframe = if ($ingest.phase_1_ingest -and $ingest.phase_1_ingest.total_articles_in_timeframe) { $ingest.phase_1_ingest.total_articles_in_timeframe } else { 0 }
        
        Write-Host "  Ingest Results: New($ingested) Total($total_in_timeframe)" -ForegroundColor White
        
        # Phase 2: Enhanced triage results with batch info
        if ($ingest.phase_2_triage -and $ingest.phase_2_triage.selections_by_ticker -and $ingest.phase_2_triage.selections_by_ticker.$ticker) {
            $selections = $ingest.phase_2_triage.selections_by_ticker.$ticker
            $company_sel = if ($selections.company) { $selections.company } else { 0 }
            $industry_sel = if ($selections.industry) { $selections.industry } else { 0 }
            $competitor_sel = if ($selections.competitor) { $selections.competitor } else { 0 }
            Write-Host "  Async Triage (batch=$TRIAGE_BATCH_SIZE): Company($company_sel) Industry($industry_sel) Competitor($competitor_sel)" -ForegroundColor Magenta
        }
        
        # Phase 4: Enhanced scraping results with batch info
        if ($ingest.phase_4_async_batch_scraping) {
            $scraping = $ingest.phase_4_async_batch_scraping
            $scraped = if ($scraping.scraped) { $scraping.scraped } else { 0 }
            $reused = if ($scraping.reused_existing) { $scraping.reused_existing } else { 0 }
            $ai_analyzed = if ($scraping.ai_analyzed) { $scraping.ai_analyzed } else { 0 }
            $success_rate = if ($scraping.overall_success_rate) { $scraping.overall_success_rate } else { "0%" }
            
            Write-Host "  Async Scraping (batch=$BATCH_SIZE): New($scraped) Reused($reused) AI($ai_analyzed) Success($success_rate)" -ForegroundColor Cyan
        }
        
        # DEBUG: Enhanced tracking with ingest response debugging
        Write-Host "  Debug: Checking ingest response for successfully_processed_tickers..." -ForegroundColor Gray
        if ($ingest.successfully_processed_tickers) {
            Write-Host "  Debug: Found successfully_processed_tickers: $($ingest.successfully_processed_tickers -join ',')" -ForegroundColor Gray
            $all_processed_tickers += $ingest.successfully_processed_tickers
            Write-Host "  $ticker PROCESSING SUCCESS" -ForegroundColor Green
        } else {
            Write-Host "  Debug: No successfully_processed_tickers field found" -ForegroundColor Yellow
            Write-Host "  Debug: Available ingest fields: $($ingest | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name)" -ForegroundColor Gray
            
            # Fallback: If processing completed without errors, consider it successful
            if ($ingest.status -eq "completed" -or $ingested -gt 0) {
                Write-Host "  Fallback: Marking $ticker as successful based on completion status" -ForegroundColor Yellow
                $all_processed_tickers += $ticker
                Write-Host "  $ticker PROCESSING SUCCESS (fallback)" -ForegroundColor Green
            } else {
                Write-Host "  $ticker PROCESSING FAILED" -ForegroundColor Red
            }
        }
        
        # DIGEST (for this ticker only) with extended timeout and comprehensive debugging
        Write-Host "  Starting digest for $ticker..." -ForegroundColor Gray
        Write-Host "  DEBUG: About to call digest endpoint at $(Get-Date)" -ForegroundColor Cyan
        Write-Host "  DEBUG: URL = $APP/cron/digest?minutes=$MINUTES&tickers=$ticker" -ForegroundColor Cyan
        Write-Host "  DEBUG: Timeout = 1800 seconds (30 minutes)" -ForegroundColor Cyan

        try {
            $digestStartTime = Get-Date
            Write-Host "  DEBUG: Making digest request at $digestStartTime..." -ForegroundColor Cyan

            $digest = Invoke-RestMethod -Method Post "$APP/cron/digest?minutes=$MINUTES&tickers=$ticker" -Headers $headers -TimeoutSec 1800

            $digestEndTime = Get-Date
            $digestDuration = ($digestEndTime - $digestStartTime).TotalSeconds
            Write-Host "  DEBUG: Digest request completed in $digestDuration seconds" -ForegroundColor Cyan

            if ($digest.status -eq "sent") {
                Write-Host "  DIGEST SUCCESS: $($digest.articles) articles sent" -ForegroundColor Green
            } else {
                Write-Host "  DIGEST RESULT: $($digest.message)" -ForegroundColor Yellow
                Write-Host "  DEBUG: Full digest response: $($digest | ConvertTo-Json -Depth 3)" -ForegroundColor Gray
            }
        } catch {
            $digestEndTime = Get-Date
            $digestDuration = ($digestEndTime - $digestStartTime).TotalSeconds
            Write-Host "  DIGEST FAILED after $digestDuration seconds: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "  DEBUG: Full error: $($_.Exception | Format-List * | Out-String)" -ForegroundColor Red

            # Continue processing even if digest fails
            Write-Host "  Continuing to next ticker despite digest failure..." -ForegroundColor Yellow
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
        }

        # Brief pause between tickers to prevent overlap and ensure clean separation
        Write-Host "  $ticker completed. Pausing 10 seconds for clean separation..." -ForegroundColor Gray
        Start-Sleep -Seconds 10
      
    } catch {
        Write-Host "  ERROR: $($_.Exception.Message)" -ForegroundColor Red

        if ($_.Exception.Response) {
            Write-Host "    HTTP Status: $($_.Exception.Response.StatusCode)" -ForegroundColor Red
        }

        # *** NEW: EMERGENCY COMMIT FOR PARTIAL SUCCESS ***
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

# ===== ENHANCED DEBUGGING SECTION =====
Write-Host "`n=== ENHANCED GITHUB COMMIT DEBUG ===" -ForegroundColor Cyan
Write-Host "  all_processed_tickers raw: $($all_processed_tickers | ConvertTo-Json)" -ForegroundColor Gray
Write-Host "  all_processed_tickers count: $($all_processed_tickers.Count)" -ForegroundColor Gray
Write-Host "  all_processed_tickers type: $($all_processed_tickers.GetType().Name)" -ForegroundColor Gray

if ($all_processed_tickers.Count -gt 0) {
    # CRITICAL FIX: Force array type with @() operator
    $uniqueTickers = @($all_processed_tickers | Sort-Object | Get-Unique)
    Write-Host "  uniqueTickers: $($uniqueTickers | ConvertTo-Json)" -ForegroundColor Gray
    Write-Host "  uniqueTickers count: $($uniqueTickers.Count)" -ForegroundColor Gray
    Write-Host "  uniqueTickers type: $($uniqueTickers.GetType().Name)" -ForegroundColor Gray
    
    # Validate ticker format
    foreach ($ticker in $uniqueTickers) {
        if ([string]::IsNullOrWhiteSpace($ticker)) {
            Write-Host "  WARNING: Empty ticker found!" -ForegroundColor Red
        } elseif ($ticker.Length -gt 10 -or $ticker -notmatch '^[A-Z]{1,8}(\.[A-Z]{1,4})?$') {
            Write-Host "  WARNING: Invalid ticker format: '$ticker'" -ForegroundColor Red
        } else {
            Write-Host "  Valid ticker: '$ticker'" -ForegroundColor Green
        }
    }
}

# ===== IMPROVED GITHUB COMMIT SECTION =====
if ($all_processed_tickers.Count -gt 0) {
    Write-Host "`n=== GITHUB COMMIT (ENHANCED METADATA) ===" -ForegroundColor Cyan
    
    try {
        # CRITICAL FIX: Force array type and validate tickers
        $validTickers = @()
        foreach ($ticker in ($all_processed_tickers | Sort-Object | Get-Unique)) {
            if (![string]::IsNullOrWhiteSpace($ticker) -and $ticker -match '^[A-Z]{1,8}(\.[A-Z]{1,4})?$') {
                $validTickers += $ticker.Trim().ToUpper()
            } else {
                Write-Host "  Skipping invalid ticker: '$ticker'" -ForegroundColor Yellow
            }
        }
        
        # Ensure we always have an array, even with one element
        $uniqueTickers = @($validTickers | Sort-Object | Get-Unique)
        
        if ($uniqueTickers.Count -eq 0) {
            Write-Host "  No valid tickers to commit after validation" -ForegroundColor Yellow
            return
        }
        
        Write-Host "  Valid tickers to commit: $($uniqueTickers -join ',')" -ForegroundColor Green
        Write-Host "  Tickers array count: $($uniqueTickers.Count)" -ForegroundColor Gray
        Write-Host "  Tickers array type: $($uniqueTickers.GetType().Name)" -ForegroundColor Gray
        
        $commitBody = @{ 
            tickers = $uniqueTickers  # This will always be an array now
            commit_message = "Enhanced processing: async triage (batch=$TRIAGE_BATCH_SIZE) + async scraping (batch=$BATCH_SIZE) for $($uniqueTickers -join ',')"
        }
        
        $commitBodyJson = $commitBody | ConvertTo-Json -Depth 3
        Write-Host "  Request body JSON:" -ForegroundColor Gray
        Write-Host "  $commitBodyJson" -ForegroundColor DarkGray
        
        # Test the endpoint first
        Write-Host "  Testing endpoint availability..." -ForegroundColor Gray
        try {
            $testResponse = Invoke-RestMethod -Uri "$APP/admin/ticker-references/stats" -Method Get -Headers $headers -TimeoutSec 30
            Write-Host "  Endpoint test: SUCCESS" -ForegroundColor Green
        } catch {
            Write-Host "  Endpoint test: FAILED - $($_.Exception.Message)" -ForegroundColor Red
            throw "API endpoint not accessible"
        }
        
        Write-Host "  Attempting GitHub sync..." -ForegroundColor Gray
        $commitResult = Invoke-RestMethod -Method Post "$APP/admin/sync-processed-tickers-to-github" -Headers $headers -Body $commitBodyJson -TimeoutSec 120
        
        Write-Host "  GitHub commit response:" -ForegroundColor Gray
        Write-Host "  $($commitResult | ConvertTo-Json -Depth 3)" -ForegroundColor DarkGray
        
        if ($commitResult.status -eq "ready_for_sync") {
            Write-Host "  SUCCESS: $($commitResult.enhanced_tickers.Count) enhanced tickers ready for sync" -ForegroundColor Green
            Write-Host "  Enhanced: $($commitResult.enhanced_tickers -join ',')" -ForegroundColor Cyan
        } else {
            Write-Host "  RESULT: $($commitResult.message)" -ForegroundColor Yellow
            Write-Host "  STATUS: $($commitResult.status)" -ForegroundColor Yellow
        }
        
    } catch {
        Write-Host "  GitHub commit failed: $($_.Exception.Message)" -ForegroundColor Red
        
        $statusCode = $null
        if ($_.Exception.Response) {
            $statusCode = $_.Exception.Response.StatusCode.value__
            Write-Host "    HTTP Status: $statusCode" -ForegroundColor Red
            
            try {
                $errorStream = $_.Exception.Response.GetResponseStream()
                $reader = New-Object System.IO.StreamReader($errorStream)
                $errorBody = $reader.ReadToEnd()
                Write-Host "    Error Details: $errorBody" -ForegroundColor Red
            } catch {
                Write-Host "    Could not read error response" -ForegroundColor Red
            }
        }
        
        # Additional debugging for 422 errors
        if ($statusCode -eq 422) {
            Write-Host "    422 indicates validation error - check request format" -ForegroundColor Red
            Write-Host "    Attempted request body was: $commitBodyJson" -ForegroundColor Red
        }
        
        # Try alternative CSV commit approach
        Write-Host "  Trying alternative CSV commit approach..." -ForegroundColor Yellow
        try {
            $csvCommitResult = Invoke-RestMethod -Method Post "$APP/admin/commit-csv-to-github" -Headers $headers -TimeoutSec 120
            
            if ($csvCommitResult.status -eq "success") {
                Write-Host "  CSV COMMIT SUCCESS: $($csvCommitResult.export_info.ticker_count) tickers committed" -ForegroundColor Green
            } else {
                Write-Host "  CSV COMMIT FAILED: $($csvCommitResult.message)" -ForegroundColor Red
            }
        } catch {
            Write-Host "  CSV commit also failed: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
} else {
    Write-Host "`n=== NO TICKERS TO COMMIT ===" -ForegroundColor Yellow
    Write-Host "  Reason: No successfully processed tickers found" -ForegroundColor Yellow
}

# ===== ENHANCED PROCESSING SUMMARY =====
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

Write-Host "`n=== ASYNC BATCH PROCESSING COMPLETED ===" -ForegroundColor Green
Write-Host "Async Triage: $TRIAGE_BATCH_SIZE concurrent OpenAI calls" -ForegroundColor Magenta
Write-Host "Async Scraping: $BATCH_SIZE concurrent article processing" -ForegroundColor Cyan
Write-Host "Individual ticker initialization prevents data corruption!" -ForegroundColor Yellow
Write-Host "✅ Incremental commits prevent metadata loss!" -ForegroundColor Green

# PAUSE SCRIPT TO READ ANY ERRORS
Write-Host "`n" -NoNewline
Write-Host "🔍 DEBUGGING MODE: " -ForegroundColor Yellow -NoNewline
Write-Host "Press any key to close PowerShell..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
