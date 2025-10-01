# Simple script to call the domain update endpoint

$APP_URL = "https://quantbrief-daily.onrender.com"
$ADMIN_TOKEN = "a77774hhwef88f99sd9g883h23nsndfs9d8cnns9adh7asc9xcibjweorn"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Updating Domain Formal Names" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

try {
    Write-Host "Calling Claude API via server..." -ForegroundColor Yellow

    $response = Invoke-RestMethod -Uri "$APP_URL/admin/update-domain-names" `
        -Method Post `
        -Headers @{ "X-Admin-Token" = $ADMIN_TOKEN } `
        -ContentType "application/json"

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Update Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Total domains: $($response.total_domains)"
    Write-Host "Updated: $($response.updated)" -ForegroundColor Green
    Write-Host "Errors: $($response.errors)" -ForegroundColor $(if ($response.errors -gt 0) { "Red" } else { "Green" })
    Write-Host ""

} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
