# Script to update domain formal names using Gemini 2.5 Flash (batches of 500)

$APP_URL = "https://stockdigest.app"
$ADMIN_TOKEN = "a77774hhwef88f99sd9g883h23nsndfs9d8cnns9adh7asc9xcibjweorn"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Updating Domain Formal Names" -ForegroundColor Cyan
Write-Host "Using: Gemini 2.5 Flash (batches of 500)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

try {
    Write-Host "Calling Gemini API via server..." -ForegroundColor Yellow
    Write-Host ""

    $response = Invoke-RestMethod -Uri "$APP_URL/admin/update-domain-names" `
        -Method Post `
        -Headers @{ "X-Admin-Token" = $ADMIN_TOKEN } `
        -ContentType "application/json"

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Update Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Summary:" -ForegroundColor Cyan
    Write-Host "  Total domains:  $($response.total_domains)"
    Write-Host "  Total batches:  $($response.total_batches)"
    Write-Host "  Batch size:     $($response.batch_size)"
    Write-Host "  Updated:        $($response.updated)" -ForegroundColor Green
    Write-Host "  Errors:         $($response.errors)" -ForegroundColor $(if ($response.errors -gt 0) { "Red" } else { "Green" })
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
