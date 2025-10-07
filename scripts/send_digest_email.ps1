# StockDigest - Send Digest Email Only
# Uses existing articles in database to generate and send Stock Intelligence email
# Does NOT re-scrape or re-ingest articles

# Configuration (matches setup_job_queue.ps1)
$APP = "https://stockdigest-daily.onrender.com"
$TOKEN = "a77774hhwef88f99sd9g883h23nsndfs9d8cnns9adh7asc9xcibjweorn"
$TICKERS = @("RY.TO", "TD.TO", "VST", "CEG")
$MINUTES = 4320  # Time window in minutes

$headers = @{ "X-Admin-Token" = $TOKEN; "Content-Type" = "application/json" }

Write-Host "=== QUANTBRIEF DIGEST EMAIL (Using Existing Data) ===" -ForegroundColor Cyan
Write-Host "Window: $MINUTES min" -ForegroundColor Yellow
Write-Host "Tickers: $($TICKERS -join ', ')" -ForegroundColor Yellow
Write-Host ""

# Process each ticker individually
foreach ($ticker in $TICKERS) {
    Write-Host "📧 Sending digest email for $ticker..." -ForegroundColor Yellow

    try {
        $url = "$APP/cron/digest?minutes=$MINUTES&tickers=$ticker"

        $result = Invoke-RestMethod -Method Post $url -Headers $headers -TimeoutSec 120

        if ($result.status -eq "sent") {
            Write-Host "  ✅ $ticker`: Email sent - $($result.articles) articles" -ForegroundColor Green
            Write-Host "     By category: Company=$($result.by_category.company) Industry=$($result.by_category.industry) Competitor=$($result.by_category.competitor)" -ForegroundColor Gray
            Write-Host "     AI Summaries: $($result.content_scraping_stats.ai_summaries)/$($result.articles)" -ForegroundColor Gray
        }
        elseif ($result.status -eq "no_articles") {
            Write-Host "  ⚠️ $ticker`: No articles found in time window" -ForegroundColor Yellow
        }
        else {
            Write-Host "  ❌ $ticker`: Email failed - $($result.status)" -ForegroundColor Red
        }

    } catch {
        Write-Host "  ❌ $ticker`: ERROR - $($_.Exception.Message)" -ForegroundColor Red
    }

    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "=== DIGEST EMAIL COMPLETE ===" -ForegroundColor Green
Write-Host "✅ Check your email for Stock Intelligence reports!" -ForegroundColor Green

# Pause to read results
Write-Host "`n🔍 Press any key to close..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")