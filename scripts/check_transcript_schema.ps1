#!/usr/bin/env pwsh
# Check if summary_json column exists in transcript_summaries table

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Checking transcript_summaries schema in production database" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# Get DATABASE_URL from environment
$DATABASE_URL = $env:DATABASE_URL

if (-not $DATABASE_URL) {
    Write-Host "ERROR: DATABASE_URL environment variable not set" -ForegroundColor Red
    exit 1
}

Write-Host "1. Checking if summary_json column exists..." -ForegroundColor Yellow
Write-Host ""

# Check column existence
$columnCheckQuery = @"
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'transcript_summaries'
AND column_name IN ('summary_json', 'prompt_version', 'summary_text')
ORDER BY column_name;
"@

Write-Host "Running query:" -ForegroundColor Gray
Write-Host $columnCheckQuery -ForegroundColor DarkGray
Write-Host ""

$columns = psql $DATABASE_URL -t -A -F'|' -c $columnCheckQuery

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to query database" -ForegroundColor Red
    exit 1
}

Write-Host "Columns found:" -ForegroundColor Green
Write-Host "Column Name          | Data Type | Nullable" -ForegroundColor Cyan
Write-Host "-------------------- | --------- | --------" -ForegroundColor Cyan

$hasJson = $false
$hasPromptVersion = $false
$hasSummaryText = $false

if ($columns) {
    foreach ($line in $columns -split "`n") {
        if ($line.Trim()) {
            $parts = $line -split '\|'
            $colName = $parts[0].Trim()
            $dataType = $parts[1].Trim()
            $nullable = $parts[2].Trim()

            Write-Host ("{0,-20} | {1,-9} | {2}" -f $colName, $dataType, $nullable)

            if ($colName -eq "summary_json") { $hasJson = $true }
            if ($colName -eq "prompt_version") { $hasPromptVersion = $true }
            if ($colName -eq "summary_text") { $hasSummaryText = $true }
        }
    }
} else {
    Write-Host "No columns found!" -ForegroundColor Red
}

Write-Host ""
Write-Host "Column Status:" -ForegroundColor Yellow
Write-Host "  summary_text:    $($hasSummaryText ? '✅ EXISTS' : '❌ MISSING')" -ForegroundColor $(if ($hasSummaryText) { 'Green' } else { 'Red' })
Write-Host "  summary_json:    $($hasJson ? '✅ EXISTS' : '❌ MISSING')" -ForegroundColor $(if ($hasJson) { 'Green' } else { 'Red' })
Write-Host "  prompt_version:  $($hasPromptVersion ? '✅ EXISTS' : '❌ MISSING')" -ForegroundColor $(if ($hasPromptVersion) { 'Green' } else { 'Red' })
Write-Host ""

# If summary_json exists, check how many transcripts have NULL vs data
if ($hasJson) {
    Write-Host "2. Checking summary_json data state..." -ForegroundColor Yellow
    Write-Host ""

    $jsonStateQuery = @"
SELECT
    COUNT(*) as total_transcripts,
    COUNT(summary_json) as has_json,
    COUNT(*) - COUNT(summary_json) as null_json,
    COUNT(CASE WHEN summary_json IS NOT NULL THEN 1 END) * 100.0 / COUNT(*) as percent_with_json
FROM transcript_summaries
WHERE report_type = 'transcript';
"@

    Write-Host "Running query:" -ForegroundColor Gray
    Write-Host $jsonStateQuery -ForegroundColor DarkGray
    Write-Host ""

    $stats = psql $DATABASE_URL -t -A -F'|' -c $jsonStateQuery

    if ($stats) {
        $parts = $stats -split '\|'
        $total = $parts[0].Trim()
        $hasJsonCount = $parts[1].Trim()
        $nullJsonCount = $parts[2].Trim()
        $percentWithJson = $parts[3].Trim()

        Write-Host "Transcript Statistics:" -ForegroundColor Green
        Write-Host "  Total transcripts:     $total" -ForegroundColor Cyan
        Write-Host "  With JSON data:        $hasJsonCount ($([math]::Round([decimal]$percentWithJson, 1))%)" -ForegroundColor $(if ([decimal]$percentWithJson -gt 50) { 'Green' } else { 'Yellow' })
        Write-Host "  With NULL JSON:        $nullJsonCount" -ForegroundColor $(if ([int]$nullJsonCount -gt 0) { 'Yellow' } else { 'Green' })
        Write-Host ""

        if ([int]$nullJsonCount -gt 0) {
            Write-Host "⚠️  WARNING: $nullJsonCount transcripts have NULL summary_json" -ForegroundColor Yellow
            Write-Host "   These will be flagged as 'legacy' and cannot be emailed" -ForegroundColor Yellow
            Write-Host ""
        }
    }

    # Show sample of recent transcripts
    Write-Host "3. Recent transcripts (last 5):" -ForegroundColor Yellow
    Write-Host ""

    $sampleQuery = @"
SELECT
    ticker,
    quarter,
    year,
    CASE
        WHEN summary_json IS NULL THEN 'NULL'
        WHEN summary_json::text = 'null' THEN 'JSON_NULL'
        ELSE 'HAS_JSON'
    END as json_status,
    LEFT(summary_text, 60) as text_preview,
    generated_at::date as gen_date
FROM transcript_summaries
WHERE report_type = 'transcript'
ORDER BY generated_at DESC
LIMIT 5;
"@

    Write-Host "Running query:" -ForegroundColor Gray
    Write-Host $sampleQuery -ForegroundColor DarkGray
    Write-Host ""

    $samples = psql $DATABASE_URL -c $sampleQuery
    Write-Host $samples
    Write-Host ""

} else {
    Write-Host "❌ PROBLEM IDENTIFIED:" -ForegroundColor Red
    Write-Host "   The summary_json column does NOT exist in production database" -ForegroundColor Red
    Write-Host "   This means ALL transcripts (even new ones) are being saved without JSON" -ForegroundColor Red
    Write-Host "   This causes ALL transcripts to be flagged as 'legacy'" -ForegroundColor Red
    Write-Host ""
    Write-Host "🔧 SOLUTION:" -ForegroundColor Yellow
    Write-Host "   The migration is in ensure_schema() but may not have run yet" -ForegroundColor Yellow
    Write-Host "   You need to restart the app to trigger the schema update" -ForegroundColor Yellow
    Write-Host "   OR manually run the migration (see next section)" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Analysis complete" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""
