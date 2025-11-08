# Script to commit ticker_reference_1.csv from Render to GitHub
# Preserves work-in-progress during batch processing

param(
    [switch]$TriggerDeploy = $false
)

$APP_URL = "https://stockdigest.app"
$ADMIN_TOKEN = "a77774hhwef88f99sd9g883h23nsndfs9d8cnns9adh7asc9xcibjweorn"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Committing Ticker CSV to GitHub" -ForegroundColor Cyan
Write-Host "File: ticker_reference_1.csv" -ForegroundColor Cyan
if ($TriggerDeploy) {
    Write-Host "Mode: DEPLOY (will trigger Render deployment)" -ForegroundColor Yellow
} else {
    Write-Host "Mode: SAFE (skip render - no deployment)" -ForegroundColor Green
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

try {
    Write-Host "Committing CSV from Render to GitHub..." -ForegroundColor Yellow
    Write-Host ""

    # Build request body
    $requestBody = @{
        csv_file = "ticker_reference_1.csv"
        skip_render = -not $TriggerDeploy
    }

    $response = Invoke-RestMethod -Uri "$APP_URL/admin/commit-ticker-csv" `
        -Method Post `
        -Headers @{ "X-Admin-Token" = $ADMIN_TOKEN } `
        -ContentType "application/json" `
        -Body ($requestBody | ConvertTo-Json)

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Commit Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Summary:" -ForegroundColor Cyan
    Write-Host "  Status:         $($response.status)" -ForegroundColor Green
    Write-Host "  Message:        $($response.message)"
    Write-Host "  GitHub Path:    $($response.github_path)"
    Write-Host "  Commit Message: $($response.commit_message)"
    if ($response.skip_render) {
        Write-Host "  Deployment:     SKIPPED (safe mode)" -ForegroundColor Green
    } else {
        Write-Host "  Deployment:     TRIGGERED (Render will redeploy)" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Next Steps" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To get the updated CSV on your local machine:" -ForegroundColor White
    Write-Host "  1. Run: git pull origin main" -ForegroundColor Gray
    Write-Host "  2. Open: data\ticker_reference_1.csv" -ForegroundColor Gray
    Write-Host ""

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
