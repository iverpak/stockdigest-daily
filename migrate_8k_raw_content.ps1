# Migration: Add raw_content column to sec_8k_filings table
# Run this in PowerShell on your local machine or Render shell

$DATABASE_URL = $env:DATABASE_URL

if (-not $DATABASE_URL) {
    Write-Host "❌ ERROR: DATABASE_URL environment variable not set" -ForegroundColor Red
    Write-Host ""
    Write-Host "Set it first:" -ForegroundColor Yellow
    Write-Host '  $env:DATABASE_URL = "postgresql://user:pass@host/database"' -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "🔄 Adding raw_content column to sec_8k_filings..." -ForegroundColor Cyan
Write-Host ""

# SQL to add the column (idempotent - only adds if missing)
$SQL = @"
-- Add raw_content column if it doesn't exist
DO `$`$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'sec_8k_filings'
        AND column_name = 'raw_content'
    ) THEN
        ALTER TABLE sec_8k_filings ADD COLUMN raw_content TEXT;
        RAISE NOTICE 'Column raw_content added successfully';
    ELSE
        RAISE NOTICE 'Column raw_content already exists - no changes needed';
    END IF;
END
`$`$;

-- Verify the schema
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'sec_8k_filings'
ORDER BY ordinal_position;
"@

# Execute SQL using psql
$SQL | psql $DATABASE_URL

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Migration completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "💡 You can now retry generating the 8-K summary for PLD" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Migration failed - see errors above" -ForegroundColor Red
    Write-Host ""
    exit 1
}
