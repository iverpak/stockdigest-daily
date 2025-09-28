# Network Timeout Solutions for QuantBrief

## Problems Identified ❌

### 1. **All-or-Nothing GitHub Commits**
Your current setup.ps1 only commits metadata to GitHub if ALL 5 tickers succeed. When CVS timed out:
- ✅ MO, GM, ODFL, SO processed successfully
- ❌ CVS timed out during digest generation
- ❌ **ZERO** tickers committed to GitHub (lost all metadata!)

### 2. **Network Timeout Vulnerabilities**
- SMTP email sending: No timeout protection
- GitHub API calls: Limited retry logic
- PowerShell timeout: 120s may be too short for large datasets

### 3. **No Incremental Saves**
- All processing happens in memory
- If any step fails, entire run's metadata is lost
- No partial success recovery

## Solutions Implemented ✅

### 1. **Enhanced SMTP Timeout Protection**
```python
# app.py:8084 - Added 60s timeout + detailed logging
with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=60) as server:
```

### 2. **GitHub Commit Retry Logic**
```python
# app.py:1918 - Added 3-attempt retry with exponential backoff
max_retries = 3
for attempt in range(max_retries):
    try:
        commit_response = requests.put(api_url, headers=headers, json=commit_data, timeout=120)
```

### 3. **Incremental Commit Endpoint**
```http
POST /admin/safe-incremental-commit
```
- Commits individual tickers immediately after processing
- Validates AI metadata exists before committing
- Network timeout won't lose entire run's work

### 4. **Enhanced Error Handling**
- Stack trace logging for all failures
- Specific timeout vs general error detection
- Graceful degradation when services fail

## Implementation Guide 🔧

### Step 1: Update Your PowerShell Script
Replace your ticker processing loop with the code in:
```
/workspaces/quantbrief-daily/scripts/improved_setup_snippet.ps1
```

Key changes:
- Commit each ticker individually after success
- Preserve metadata even if later tickers fail
- Emergency commits for partial successes

### Step 2: Test the New Endpoints

**Test incremental commit:**
```bash
curl -X POST "https://quantbrief-daily.onrender.com/admin/safe-incremental-commit" \
     -H "X-Admin-Token: your_token" \
     -H "Content-Type: application/json" \
     -d '{"tickers": ["MO"]}'
```

**Debug digest issues:**
```bash
curl "https://quantbrief-daily.onrender.com/admin/debug/digest-check/CVS" \
     -H "X-Admin-Token: your_token"
```

### Step 3: Monitor Improvements

The enhanced logging will now show:
```
=== DIGEST GENERATION STARTING ===
Connecting to SMTP server: smtp.gmail.com:587
Starting TLS...
Logging in to SMTP server...
Sending email...
Email sent successfully to recipient@domain.com
```

```
GitHub commit attempt 1/3
GitHub commit attempt 2/3 (if needed)
Successfully committed CSV to GitHub: abc12345
```

## Network Timeout Prevention 🛡️

### For CVS-Specific Issues:
1. **Check article volume**: CVS may have more articles than other tickers
2. **Monitor digest size**: Large HTML emails take longer to send
3. **Email server limits**: Gmail/Outlook may rate limit large emails

### General Prevention:
- ✅ **Incremental commits** preserve work as you go
- ✅ **Retry logic** handles temporary network issues
- ✅ **Timeout protection** prevents infinite hangs
- ✅ **Enhanced logging** shows exactly where failures occur

## Testing Recommendations 🧪

### 1. Test CVS Individually:
```bash
# Test just CVS processing
curl -X POST "https://quantbrief-daily.onrender.com/cron/ingest?tickers=CVS&minutes=4320" \
     -H "X-Admin-Token: your_token"

# Then test digest
curl -X POST "https://quantbrief-daily.onrender.com/cron/digest?tickers=CVS&minutes=4320" \
     -H "X-Admin-Token: your_token"
```

### 2. Test Incremental Workflow:
```powershell
# Process one ticker, then commit
$ingest = Invoke-RestMethod -Method Post "$APP/cron/ingest?tickers=MO&minutes=4320" -Headers $headers
$digest = Invoke-RestMethod -Method Post "$APP/cron/digest?tickers=MO&minutes=4320" -Headers $headers
$commit = Invoke-RestMethod -Method Post "$APP/admin/safe-incremental-commit" -Headers $headers -Body '{"tickers":["MO"]}'
```

## Expected Benefits 📈

### Before (All-or-Nothing):
- ❌ 1 ticker fails → Lose ALL metadata
- ❌ Network timeout → Lose ALL work
- ❌ No visibility into failure points

### After (Incremental):
- ✅ 1 ticker fails → Keep other 4 tickers' metadata
- ✅ Network timeout → Retry with backoff
- ✅ Detailed logging shows exact failure points
- ✅ Emergency commits save partial work

## Emergency Recovery 🚨

If you lose metadata in the future:

1. **Check database**: Metadata is preserved in PostgreSQL
```sql
SELECT ticker, ai_generated, industry_keyword_1, competitor_1_name
FROM ticker_reference
WHERE ticker IN ('MO', 'GM', 'ODFL', 'SO', 'CVS');
```

2. **Manual commit**: Force commit current database state
```bash
curl -X POST "https://quantbrief-daily.onrender.com/admin/commit-csv-to-github" \
     -H "X-Admin-Token: your_token"
```

3. **Incremental recovery**: Commit specific tickers
```bash
curl -X POST "https://quantbrief-daily.onrender.com/admin/safe-incremental-commit" \
     -H "X-Admin-Token: your_token" \
     -d '{"tickers": ["MO", "GM", "ODFL", "SO"]}'
```

---

**The core issue was network timeout vulnerability in an all-or-nothing commit strategy. These improvements provide incremental saves, retry logic, and comprehensive error handling to prevent metadata loss.**