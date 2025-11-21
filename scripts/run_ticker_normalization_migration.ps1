# StockDigest - Ticker Normalization Migration Script
# Runs SQL migration to enforce ticker consistency across all tables

# Configuration
$DATABASE_URL = "postgresql://quantbrief_db_user:YOUR_PASSWORD_HERE@dpg-ctbjhvbtq21c73bvjr10-a.oregon-postgres.render.com/quantbrief_db"

Write-Host "=== STOCKDIGEST TICKER NORMALIZATION MIGRATION ===" -ForegroundColor Cyan
Write-Host ""

# Set environment variable from config
$env:DATABASE_URL = $DATABASE_URL

# Check if DATABASE_URL is set
if (-not $env:DATABASE_URL -or $env:DATABASE_URL -like "*YOUR_PASSWORD_HERE*") {
    Write-Host "❌ ERROR: DATABASE_URL not configured in script!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please update line 5 in this script:" -ForegroundColor Yellow
    Write-Host '  $DATABASE_URL = "postgresql://quantbrief_db_user:ACTUAL_PASSWORD@dpg-..."' -ForegroundColor Gray
    Write-Host ""
    Write-Host "Get the full URL from Render dashboard:" -ForegroundColor Yellow
    Write-Host "  Render → Your Database → Connection → External Database URL" -ForegroundColor Gray
    Write-Host "  Copy the entire URL and replace line 5" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor White
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Check if psql is installed
Write-Host "🔍 Checking for psql..." -ForegroundColor Cyan
try {
    $psqlVersion = psql --version 2>&1
    Write-Host "   ✅ Found: $psqlVersion" -ForegroundColor Green
} catch {
    Write-Host "   ❌ psql not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "You have two options:" -ForegroundColor Yellow
    Write-Host "1. Install PostgreSQL client (includes psql)" -ForegroundColor Gray
    Write-Host "   Download: https://www.postgresql.org/download/windows/" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Copy-paste the SQL directly (easier!):" -ForegroundColor Green
    Write-Host "   - Open: migrations/add_ticker_reference_constraints.sql" -ForegroundColor Gray
    Write-Host "   - Copy lines 22-185" -ForegroundColor Gray
    Write-Host "   - Paste into Render SQL Shell or pgAdmin" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor White
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Get migration file path
$migrationFile = "$PSScriptRoot\..\migrations\add_ticker_reference_constraints.sql"

# Check if migration file exists
if (-not (Test-Path $migrationFile)) {
    Write-Host "❌ ERROR: Migration file not found!" -ForegroundColor Red
    Write-Host "   Expected: $migrationFile" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor White
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host ""
Write-Host "📄 Migration file: migrations/add_ticker_reference_constraints.sql" -ForegroundColor Cyan
Write-Host ""
Write-Host "This migration will:" -ForegroundColor Yellow
Write-Host "  ✓ Normalize existing tickers (UPPER + TRIM)" -ForegroundColor Gray
Write-Host "  ✓ Add foreign key constraints" -ForegroundColor Gray
Write-Host "  ✓ Create performance indexes" -ForegroundColor Gray
Write-Host "  ✓ Validate data consistency" -ForegroundColor Gray
Write-Host ""
Write-Host "Safety notes:" -ForegroundColor Green
Write-Host "  • Runs inside a transaction (all-or-nothing)" -ForegroundColor Gray
Write-Host "  • Idempotent (safe to run multiple times)" -ForegroundColor Gray
Write-Host "  • Checks for violations before adding constraints" -ForegroundColor Gray
Write-Host "  • Won't break existing valid data" -ForegroundColor Gray
Write-Host ""

# Show database info (hide password)
$dbInfo = $env:DATABASE_URL -replace "://[^:]+:[^@]+@", "://***:***@"
Write-Host "🎯 Target database: $dbInfo" -ForegroundColor Cyan
Write-Host ""

# Confirm execution
Write-Host "⚠️  Ready to run migration!" -ForegroundColor Yellow
Write-Host "Press 'Y' to continue, any other key to abort: " -ForegroundColor Yellow -NoNewline

$confirmation = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Write-Host ""
Write-Host ""

if ($confirmation.Character -ne 'Y' -and $confirmation.Character -ne 'y') {
    Write-Host "❌ Migration aborted" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor White
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 0
}

# Run migration
Write-Host "=== RUNNING MIGRATION ===" -ForegroundColor Cyan
Write-Host ""

try {
    # Execute migration and capture output
    $output = Get-Content $migrationFile | psql $env:DATABASE_URL 2>&1

    # Display output
    foreach ($line in $output) {
        $lineStr = $line.ToString()

        # Color-code output
        if ($lineStr -match "^NOTICE:.*WARNING") {
            Write-Host $lineStr -ForegroundColor Red
        } elseif ($lineStr -match "^NOTICE:.*Successfully") {
            Write-Host $lineStr -ForegroundColor Green
        } elseif ($lineStr -match "^NOTICE:.*====") {
            Write-Host $lineStr -ForegroundColor Cyan
        } elseif ($lineStr -match "^NOTICE:") {
            Write-Host $lineStr -ForegroundColor Gray
        } elseif ($lineStr -match "^ERROR") {
            Write-Host $lineStr -ForegroundColor Red
        } elseif ($lineStr -match "UPDATE|INSERT|CREATE|ALTER") {
            Write-Host $lineStr -ForegroundColor Yellow
        } elseif ($lineStr -match "COMMIT") {
            Write-Host $lineStr -ForegroundColor Green
        } else {
            Write-Host $lineStr
        }
    }

    Write-Host ""
    Write-Host "=== MIGRATION COMPLETE ===" -ForegroundColor Green
    Write-Host ""

    # Check for warnings in output
    $hasWarnings = $output | Where-Object { $_ -match "WARNING.*not in ticker_reference" }

    if ($hasWarnings) {
        Write-Host "⚠️  WARNINGS DETECTED!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Some tickers in your database are not in ticker_reference." -ForegroundColor Yellow
        Write-Host "Foreign key constraints were NOT added for those tables." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Cyan
        Write-Host "1. Review the warnings above" -ForegroundColor Gray
        Write-Host "2. Either:" -ForegroundColor Gray
        Write-Host "   a) Add missing tickers to ticker_reference, OR" -ForegroundColor Gray
        Write-Host "   b) Delete invalid rows from affected tables" -ForegroundColor Gray
        Write-Host "3. Re-run this script to add constraints" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Example fix:" -ForegroundColor Cyan
        Write-Host '  # Add missing ticker to ticker_reference' -ForegroundColor Gray
        Write-Host '  INSERT INTO ticker_reference (ticker, company_name, ...) VALUES (''AAPL'', ''Apple Inc.'', ...);' -ForegroundColor Gray
        Write-Host ""
        Write-Host "  OR" -ForegroundColor Yellow
        Write-Host ""
        Write-Host '  # Delete invalid data' -ForegroundColor Gray
        Write-Host '  DELETE FROM company_releases WHERE ticker = ''invalid_ticker'';' -ForegroundColor Gray
    } else {
        Write-Host "✅ No warnings! All constraints added successfully." -ForegroundColor Green
        Write-Host ""
        Write-Host "Your database now enforces ticker consistency:" -ForegroundColor Cyan
        Write-Host "  • All tickers must exist in ticker_reference" -ForegroundColor Gray
        Write-Host "  • All tickers are normalized (UPPERCASE)" -ForegroundColor Gray
        Write-Host "  • Performance indexes created" -ForegroundColor Gray
    }

} catch {
    $errorMsg = $_.Exception.Message

    Write-Host ""
    Write-Host "🚨 ERROR: $errorMsg" -ForegroundColor Red
    Write-Host ""

    if ($errorMsg -match "does not exist") {
        Write-Host "⚠️  One of the tables doesn't exist yet." -ForegroundColor Yellow
        Write-Host "This is normal for a fresh database." -ForegroundColor Gray
        Write-Host ""
        Write-Host "The migration will automatically create tables when they're needed." -ForegroundColor Gray
    } elseif ($errorMsg -match "connection") {
        Write-Host "⚠️  Database connection failed." -ForegroundColor Yellow
        Write-Host "Check your DATABASE_URL is correct." -ForegroundColor Gray
    }

    Write-Host ""
    Write-Host "💡 ALTERNATIVE: Copy-paste method" -ForegroundColor Cyan
    Write-Host "1. Open: migrations/add_ticker_reference_constraints.sql" -ForegroundColor Gray
    Write-Host "2. Copy lines 22-185" -ForegroundColor Gray
    Write-Host "3. Paste into Render SQL Shell or pgAdmin" -ForegroundColor Gray
}

Write-Host ""
Write-Host "✅ Script complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
