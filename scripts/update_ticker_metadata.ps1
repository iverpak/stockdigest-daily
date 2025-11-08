# Script to update ticker metadata using Gemini 2.5 Pro (batches of 5)
# Supports test mode (first 10 batches) and full run mode

param(
    [switch]$TestMode = $true,
    [int]$MaxBatches = 0
)

$APP_URL = "https://stockdigest.app"
$ADMIN_TOKEN = "a77774hhwef88f99sd9g883h23nsndfs9d8cnns9adh7asc9xcibjweorn"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Updating Ticker Metadata" -ForegroundColor Cyan
Write-Host "Using: Gemini 2.5 Pro (batches of 5)" -ForegroundColor Cyan
if ($TestMode -or $MaxBatches -gt 0) {
    $batchLimit = if ($MaxBatches -gt 0) { $MaxBatches } else { 10 }
    Write-Host "Mode: TEST (first $batchLimit batches only)" -ForegroundColor Yellow
} else {
    Write-Host "Mode: FULL RUN (all batches)" -ForegroundColor Green
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

try {
    Write-Host "Calling Gemini API via server..." -ForegroundColor Yellow
    Write-Host ""

    # Build request body
    $requestBody = @{
        csv_file = "ticker_reference_1.csv"
        batch_size = 5
    }

    # Add max_batches if in test mode
    if ($TestMode) {
        $requestBody.max_batches = 10
    } elseif ($MaxBatches -gt 0) {
        $requestBody.max_batches = $MaxBatches
    }

    $response = Invoke-RestMethod -Uri "$APP_URL/admin/update-ticker-metadata-csv" `
        -Method Post `
        -Headers @{ "X-Admin-Token" = $ADMIN_TOKEN } `
        -ContentType "application/json" `
        -Body ($requestBody | ConvertTo-Json)

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Update Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Summary:" -ForegroundColor Cyan
    Write-Host "  CSV File:         $($response.csv_file)"
    Write-Host "  Total to process: $($response.total_to_process)"
    Write-Host "  Total batches:    $($response.total_batches)"
    Write-Host "  Batch size:       $($response.batch_size)"
    Write-Host "  Updated:          $($response.updated)" -ForegroundColor Green
    Write-Host "  Errors:           $($response.errors)" -ForegroundColor $(if ($response.errors -gt 0) { "Red" } else { "Green" })
    Write-Host ""

    # Show batch details
    if ($response.batches -and $response.batches.Count -gt 0) {
        Write-Host "Batch Results:" -ForegroundColor Cyan
        Write-Host ""

        foreach ($batch in $response.batches) {
            $batchNum = $batch.batch
            $status = $batch.status

            if ($status -eq "success") {
                $color = "Green"
                $icon = "✅"
                $details = "Updated: $($batch.updated), Errors: $($batch.errors), Time: $($batch.generation_time)"
            } else {
                $color = "Red"
                $icon = "❌"
                $details = "Error: $($batch.message)"
            }

            Write-Host "  $icon Batch $batchNum/$($response.total_batches): " -NoNewline -ForegroundColor $color
            Write-Host $details -ForegroundColor Gray
        }

        Write-Host ""
    }

    # Show next steps
    if ($TestMode -or $MaxBatches -gt 0) {
        Write-Host "========================================" -ForegroundColor Yellow
        Write-Host "TEST MODE COMPLETE" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Next Steps:" -ForegroundColor Cyan
        Write-Host "  1. Open data/ticker_reference_1.csv" -ForegroundColor White
        Write-Host "  2. Check first $($response.updated) rows for quality" -ForegroundColor White
        Write-Host "  3. If good, run full update:" -ForegroundColor White
        Write-Host "     .\scripts\update_ticker_metadata.ps1" -ForegroundColor Gray
        Write-Host ""
    } else {
        Write-Host "All done! Updated tickers saved to: data/ticker_reference_1.csv" -ForegroundColor Green
        Write-Host ""
    }

} catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red

    # Try to parse error response
    if ($_.ErrorDetails.Message) {
        try {
            $errorResponse = $_.ErrorDetails.Message | ConvertFrom-Json
            if ($errorResponse.message) {
                Write-Host "Details: $($errorResponse.message)" -ForegroundColor Yellow
            }
        } catch {
            # If can't parse, show raw error
            Write-Host "Details: $($_.ErrorDetails.Message)" -ForegroundColor Yellow
        }
    }

    Write-Host ""
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
