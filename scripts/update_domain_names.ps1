# QuantBrief - Batch Update Domain Formal Names via Claude API
# This script fetches all domains from the database and uses Claude API to generate proper formal names

param(
    [string]$DatabaseUrl = $env:DATABASE_URL,
    [string]$AnthropicApiKey = $env:ANTHROPIC_API_KEY,
    [switch]$DryRun = $false
)

# Prevent window from closing on error
$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Domain Formal Name Batch Updater" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Validate environment variables
if (-not $DatabaseUrl) {
    Write-Host "ERROR: DATABASE_URL environment variable not set" -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

if (-not $AnthropicApiKey) {
    Write-Host "ERROR: ANTHROPIC_API_KEY environment variable not set" -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Check if psql is available
if (-not (Get-Command psql -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: psql command not found. Please install PostgreSQL client tools." -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host "Step 1: Fetching all domains from database..." -ForegroundColor Yellow
$query = "SELECT domain FROM domain_names ORDER BY domain;"
$domainsJson = psql $DatabaseUrl -t -A -F',' -c $query

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to fetch domains from database" -ForegroundColor Red
    exit 1
}

# Parse domains
$domains = $domainsJson -split "`n" | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_.Trim() }

if ($domains.Count -eq 0) {
    Write-Host "No domains found in database. Exiting." -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($domains.Count) domains" -ForegroundColor Green
Write-Host ""

# Build Claude API prompt with all domains
Write-Host "Step 2: Building Claude API request..." -ForegroundColor Yellow

$domainList = ($domains | ForEach-Object { "- $_" }) -join "`n"

$prompt = @"
You are a domain name expert. Below is a list of domain names. For EACH domain, provide ONLY the formal brand/publication name as it should appear in professional contexts.

Rules:
1. Return EXACTLY one name per domain
2. Use proper capitalization (e.g., "The Wall Street Journal" not "wall street journal")
3. For news outlets, include "The" if official (e.g., "The Guardian")
4. For companies, use official brand name (e.g., "Bloomberg" not "Bloomberg LP")
5. Do NOT include domain extensions (.com, .net, etc.) in the formal name
6. If a domain is unknown or generic, return the domain name as-is with proper capitalization

Format your response as a JSON object where keys are domains and values are formal names:
{
  "domain1.com": "Formal Name 1",
  "domain2.com": "Formal Name 2"
}

Domains:
$domainList
"@

$requestBody = @{
    model = "claude-sonnet-4-20250514"
    max_tokens = 8192
    messages = @(
        @{
            role = "user"
            content = $prompt
        }
    )
} | ConvertTo-Json -Depth 10

Write-Host "Step 3: Calling Claude API (this may take 20-30 seconds)..." -ForegroundColor Yellow

$headers = @{
    "x-api-key" = $AnthropicApiKey
    "content-type" = "application/json"
    "anthropic-version" = "2023-06-01"
}

try {
    $response = Invoke-RestMethod -Uri "https://api.anthropic.com/v1/messages" `
        -Method Post `
        -Headers $headers `
        -Body $requestBody

    $claudeResponse = $response.content[0].text
    Write-Host "Claude API response received" -ForegroundColor Green
    Write-Host ""

    # Extract JSON from response (Claude might wrap it in markdown)
    if ($claudeResponse -match '```json\s*([\s\S]*?)\s*```') {
        $jsonContent = $matches[1]
    } elseif ($claudeResponse -match '\{[\s\S]*\}') {
        $jsonContent = $matches[0]
    } else {
        Write-Host "ERROR: Could not extract JSON from Claude response" -ForegroundColor Red
        Write-Host "Raw response:" -ForegroundColor Yellow
        Write-Host $claudeResponse
        exit 1
    }

    $domainMappings = $jsonContent | ConvertFrom-Json

} catch {
    Write-Host "ERROR: Claude API call failed" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

Write-Host "Step 4: Updating database..." -ForegroundColor Yellow
$updateCount = 0
$errorCount = 0

foreach ($domain in $domains) {
    $formalName = $domainMappings.$domain

    if (-not $formalName) {
        Write-Host "  ⚠️  Missing formal name for: $domain" -ForegroundColor Yellow
        $errorCount++
        continue
    }

    # Escape single quotes for SQL
    $escapedFormalName = $formalName -replace "'", "''"

    $updateQuery = "UPDATE domain_names SET formal_name = '$escapedFormalName' WHERE domain = '$domain';"

    if ($DryRun) {
        Write-Host "  [DRY RUN] $domain → $formalName" -ForegroundColor Cyan
    } else {
        $result = psql $DatabaseUrl -c $updateQuery 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ $domain → $formalName" -ForegroundColor Green
            $updateCount++
        } else {
            Write-Host "  ❌ Failed to update $domain" -ForegroundColor Red
            Write-Host "     Error: $result" -ForegroundColor Red
            $errorCount++
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Update Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total domains: $($domains.Count)"
Write-Host "Updated: $updateCount" -ForegroundColor Green
Write-Host "Errors: $errorCount" -ForegroundColor $(if ($errorCount -gt 0) { "Red" } else { "Green" })

if ($DryRun) {
    Write-Host ""
    Write-Host "DRY RUN MODE - No changes were made to the database" -ForegroundColor Yellow
    Write-Host "Run without -DryRun flag to apply changes" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
