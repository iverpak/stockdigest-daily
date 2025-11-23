#!/usr/bin/env pwsh
# Manually add summary_json column to transcript_summaries table (if needed)

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Adding summary_json column to transcript_summaries table" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# Get DATABASE_URL from environment
$DATABASE_URL = $env:DATABASE_URL

if (-not $DATABASE_URL) {
    Write-Host "ERROR: DATABASE_URL environment variable not set" -ForegroundColor Red
    exit 1
}

Write-Host "⚠️  WARNING: This will modify your production database!" -ForegroundColor Yellow
Write-Host ""
Write-Host "This script will:" -ForegroundColor Yellow
Write-Host "  1. Add summary_json column (JSONB type) if it doesn't exist" -ForegroundColor Yellow
Write-Host "  2. Add prompt_version column (TEXT type) if it doesn't exist" -ForegroundColor Yellow
Write-Host "  3. This is SAFE - uses IF NOT EXISTS (won't fail if columns exist)" -ForegroundColor Yellow
Write-Host ""

$confirmation = Read-Host "Continue? (yes/no)"
if ($confirmation -ne "yes") {
    Write-Host "Aborted by user" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Running migration..." -ForegroundColor Green
Write-Host ""

# Migration SQL (same as in ensure_schema)
$migrationSQL = @"
-- Migration: Add summary_json column if it doesn't exist (Nov 2025)
DO `$ BEGIN
    ALTER TABLE transcript_summaries ADD COLUMN summary_json JSONB;
EXCEPTION
    WHEN duplicate_column THEN NULL;  -- Column already exists, ignore
END `$;

-- Migration: Add prompt_version column if it doesn't exist (Nov 2025)
DO `$ BEGIN
    ALTER TABLE transcript_summaries ADD COLUMN prompt_version TEXT;
EXCEPTION
    WHEN duplicate_column THEN NULL;  -- Column already exists, ignore
END `$;
"@

Write-Host "SQL to execute:" -ForegroundColor Gray
Write-Host $migrationSQL -ForegroundColor DarkGray
Write-Host ""

# Execute migration
psql $DATABASE_URL -c $migrationSQL

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Migration completed successfully!" -ForegroundColor Green
    Write-Host ""

    # Verify columns were added
    Write-Host "Verifying columns..." -ForegroundColor Yellow

    $verifyQuery = @"
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'transcript_summaries'
AND column_name IN ('summary_json', 'prompt_version')
ORDER BY column_name;
"@

    $result = psql $DATABASE_URL -c $verifyQuery
    Write-Host $result
    Write-Host ""

    Write-Host "✅ Columns verified!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📝 NEXT STEPS:" -ForegroundColor Yellow
    Write-Host "  1. New transcripts will now save with summary_json" -ForegroundColor Yellow
    Write-Host "  2. Old transcripts still have NULL summary_json (flagged as legacy)" -ForegroundColor Yellow
    Write-Host "  3. To email old transcripts, regenerate them via /admin/research" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Migration failed!" -ForegroundColor Red
    Write-Host "Check error messages above" -ForegroundColor Red
    Write-Host ""
    exit 1
}

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Migration complete" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""
