# StockDigest Beta Users - CSV Export Script
# Manually trigger CSV export of active beta users

# Configuration
$APP = "https://quantbrief-daily.onrender.com"
$TOKEN = "a77774hhwef88f99sd9g883h23nsndfs9d8cnns9adh7asc9xcibjweorn"

$headers = @{ "X-Admin-Token" = $TOKEN }

Write-Host "=== STOCKDIGEST BETA USERS CSV EXPORT ===" -ForegroundColor Cyan
Write-Host ""

try {
    Write-Host "Exporting beta users to CSV..." -ForegroundColor Yellow

    $result = Invoke-RestMethod -Method Post "$APP/admin/export-user-csv" -Headers $headers

    if ($result.status -eq "success") {
        Write-Host ""
        Write-Host "✅ SUCCESS" -ForegroundColor Green
        Write-Host "  Users exported: $($result.message)" -ForegroundColor White
        Write-Host "  File path: $($result.file_path)" -ForegroundColor White
        Write-Host "  Exported at: $($result.exported_at)" -ForegroundColor White
        Write-Host ""
        Write-Host "CSV file ready for 7 AM processing workflow!" -ForegroundColor Cyan
    } else {
        Write-Host "❌ EXPORT FAILED" -ForegroundColor Red
        Write-Host "  $($result.message)" -ForegroundColor White
    }
} catch {
    Write-Host ""
    Write-Host "❌ ERROR" -ForegroundColor Red
    Write-Host "  $($_.Exception.Message)" -ForegroundColor White

    if ($_.Exception.Response) {
        $statusCode = $_.Exception.Response.StatusCode.value__
        Write-Host "  Status Code: $statusCode" -ForegroundColor White

        if ($statusCode -eq 401) {
            Write-Host "  Authentication failed. Check your ADMIN_TOKEN." -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Cyan
