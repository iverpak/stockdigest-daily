# Debug version - shows what's happening step by step

Write-Host "===== DEBUG START =====" -ForegroundColor Cyan

# Test 1: Check if script is running
Write-Host "Step 1: Script is running..." -ForegroundColor Green

# Test 2: Check environment variables
Write-Host "Step 2: Checking environment variables..." -ForegroundColor Yellow
$dbUrl = $env:DATABASE_URL
$apiKey = $env:ANTHROPIC_API_KEY

if ($dbUrl) {
    $maskedUrl = $dbUrl.Substring(0, [Math]::Min(20, $dbUrl.Length)) + "..."
    Write-Host "  DATABASE_URL: $maskedUrl" -ForegroundColor Green
} else {
    Write-Host "  DATABASE_URL: NOT SET" -ForegroundColor Red
}

if ($apiKey) {
    Write-Host "  ANTHROPIC_API_KEY: SET" -ForegroundColor Green
} else {
    Write-Host "  ANTHROPIC_API_KEY: NOT SET" -ForegroundColor Red
}

# Test 3: Check psql
Write-Host "Step 3: Checking for psql..." -ForegroundColor Yellow
try {
    $psqlVersion = psql --version 2>&1
    Write-Host "  psql found: $psqlVersion" -ForegroundColor Green
} catch {
    Write-Host "  psql NOT found or error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 4: Try a simple psql query
Write-Host "Step 4: Testing database connection..." -ForegroundColor Yellow
if ($dbUrl) {
    try {
        $testQuery = "SELECT 1 as test;"
        Write-Host "  Running: psql `$DATABASE_URL -c `"$testQuery`"" -ForegroundColor Cyan
        $result = psql $dbUrl -t -A -c $testQuery 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Database connection: SUCCESS" -ForegroundColor Green
            Write-Host "  Result: $result" -ForegroundColor Green
        } else {
            Write-Host "  Database connection: FAILED (exit code $LASTEXITCODE)" -ForegroundColor Red
            Write-Host "  Error: $result" -ForegroundColor Red
        }
    } catch {
        Write-Host "  Database connection: ERROR" -ForegroundColor Red
        Write-Host "  Exception: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "  Skipping database test (no DATABASE_URL)" -ForegroundColor Yellow
}

# Test 5: Try fetching domains
Write-Host "Step 5: Testing domain fetch..." -ForegroundColor Yellow
if ($dbUrl) {
    try {
        $domainQuery = "SELECT domain FROM domain_names ORDER BY domain LIMIT 5;"
        Write-Host "  Running: psql `$DATABASE_URL -c `"$domainQuery`"" -ForegroundColor Cyan
        $domains = psql $dbUrl -t -A -c $domainQuery 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Domain fetch: SUCCESS" -ForegroundColor Green
            Write-Host "  First 5 domains:" -ForegroundColor Green
            $domains -split "`n" | Where-Object { $_.Trim() -ne "" } | ForEach-Object {
                Write-Host "    - $_" -ForegroundColor Cyan
            }
        } else {
            Write-Host "  Domain fetch: FAILED (exit code $LASTEXITCODE)" -ForegroundColor Red
            Write-Host "  Error: $domains" -ForegroundColor Red
        }
    } catch {
        Write-Host "  Domain fetch: ERROR" -ForegroundColor Red
        Write-Host "  Exception: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "===== DEBUG COMPLETE =====" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
