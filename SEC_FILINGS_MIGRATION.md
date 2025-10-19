# SEC Filings Table Migration Script

**Date:** October 19, 2025
**Purpose:** Migrate existing `company_profiles` data to unified `sec_filings` table
**Database:** PostgreSQL (stockdigest-db on Render)

---

## Pre-Migration Checklist

✅ **Verify Current State:**
```sql
-- Check how many profiles exist
SELECT COUNT(*) FROM company_profiles;

-- Preview existing data
SELECT ticker, company_name, fiscal_year, filing_date
FROM company_profiles
ORDER BY generated_at DESC
LIMIT 10;
```

✅ **Backup Current Data:**
```sql
-- Export to CSV (run this from psql or Render dashboard)
\copy (SELECT * FROM company_profiles) TO '/tmp/company_profiles_backup.csv' CSV HEADER;
```

---

## Migration Steps

### Step 1: Create New `sec_filings` Table

**NOTE:** This is already done in `app.py` (lines 1527-1605) and will be created on next deployment.
The table will be created automatically when the app starts.

Verify table exists:
```sql
SELECT * FROM information_schema.tables
WHERE table_name = 'sec_filings';
```

### Step 2: Migrate Existing Data

Run this SQL script to migrate all existing 10-K profiles:

```sql
-- Migrate company_profiles → sec_filings
INSERT INTO sec_filings (
    ticker,
    filing_type,
    fiscal_year,
    fiscal_quarter,
    filing_date,
    profile_markdown,
    profile_summary,
    key_metrics,
    source_type,
    source_file,
    sec_html_url,
    company_name,
    industry,
    ai_provider,
    ai_model,
    generation_time_seconds,
    token_count_input,
    token_count_output,
    status,
    error_message,
    generated_at
)
SELECT
    ticker,
    '10-K' AS filing_type,                  -- All existing are 10-Ks
    fiscal_year,
    NULL AS fiscal_quarter,                 -- 10-Ks don't have quarters
    filing_date,
    profile_markdown,
    profile_summary,
    key_metrics,
    CASE
        WHEN source_file LIKE '%SEC.gov%' THEN 'fmp_sec'
        WHEN source_file IS NULL THEN 'file_upload'
        ELSE 'file_upload'
    END AS source_type,
    source_file,
    NULL AS sec_html_url,                   -- Not stored in old table
    company_name,
    industry,
    ai_provider,
    gemini_model AS ai_model,               -- Column renamed
    generation_time_seconds,
    token_count_input,
    token_count_output,
    status,
    error_message,
    generated_at
FROM company_profiles
ON CONFLICT (ticker, filing_type, fiscal_year, fiscal_quarter) DO UPDATE SET
    -- Update if somehow already exists (shouldn't happen on first run)
    profile_markdown = EXCLUDED.profile_markdown,
    generated_at = EXCLUDED.generated_at;
```

### Step 3: Verify Migration

```sql
-- Check counts match
SELECT
    (SELECT COUNT(*) FROM company_profiles) AS old_count,
    (SELECT COUNT(*) FROM sec_filings WHERE filing_type = '10-K') AS new_count;

-- Compare sample records
SELECT
    cp.ticker,
    cp.fiscal_year,
    cp.company_name,
    sf.ticker AS sf_ticker,
    sf.fiscal_year AS sf_year,
    sf.filing_type
FROM company_profiles cp
FULL OUTER JOIN sec_filings sf
    ON cp.ticker = sf.ticker
    AND sf.filing_type = '10-K'
    AND cp.fiscal_year = sf.fiscal_year
LIMIT 10;
```

### Step 4: Test Backward Compatibility View

The `company_profiles` VIEW should still work:

```sql
-- This should return same data as before
SELECT ticker, company_name, fiscal_year, filing_date
FROM company_profiles
ORDER BY generated_at DESC
LIMIT 5;

-- Verify it's actually querying sec_filings
EXPLAIN SELECT * FROM company_profiles WHERE ticker = 'AAPL';
-- Should show "Seq Scan on sec_filings" in plan
```

---

## Post-Migration Verification

### Test All API Endpoints:

**1. View Profiles:**
```bash
curl "https://stockdigest.app/api/admin/company-profiles?token=YOUR_TOKEN"
```
Expected: Returns all 10-K profiles from `sec_filings`

**2. Email Profile:**
```bash
curl -X POST https://stockdigest.app/api/admin/email-company-profile \
  -H "Content-Type: application/json" \
  -d '{"token":"YOUR_TOKEN","ticker":"AAPL"}'
```
Expected: Sends email with latest 10-K profile

**3. Delete Profile:**
```bash
curl -X POST https://stockdigest.app/api/admin/delete-company-profile \
  -H "Content-Type: application/json" \
  -d '{"token":"YOUR_TOKEN","ticker":"TEST"}'
```
Expected: Deletes 10-K profiles for TEST ticker

**4. Executive Summary Injection (Internal Test):**
- Process any ticker that has a 10-K profile
- Check logs for: `✅ Loaded 10-K profile (FY20XX, XXX,XXX chars)`
- This confirms the executive summary is pulling from `sec_filings`

---

## Future: Enable Multi-Material Context (Optional)

When ready to test enhanced context (10-K + 10-Q + presentations), uncomment lines 14070-14120 in `app.py`:

```python
# FUTURE: Multi-material context injection (10-K + 10-Q + Presentation)
# Uncomment when ready to test enhanced context
```

This will inject multiple filings into Claude's context when generating executive summaries.

---

## Rollback Plan (If Needed)

If migration fails, you can restore from backup:

```sql
-- Drop new table
DROP TABLE IF EXISTS sec_filings CASCADE;

-- Restore from backup
\copy company_profiles FROM '/tmp/company_profiles_backup.csv' CSV HEADER;
```

---

## Clean-Up (After Confirming Migration Success)

**WAIT 7 DAYS** before dropping old table to ensure everything works.

Then, optionally:

```sql
-- OPTIONAL: Drop old table (keeps VIEW in place)
-- The VIEW will continue to work, querying sec_filings
DROP TABLE IF EXISTS company_profiles CASCADE;

-- If you drop the table, you may want to remove the VIEW too:
DROP VIEW IF EXISTS company_profiles;
```

**RECOMMENDATION:** Keep the VIEW indefinitely for backward compatibility.

---

## Summary

- ✅ New `sec_filings` table created automatically on deployment
- ✅ All 6 integration points updated to use `sec_filings`
- ✅ Backward compatibility VIEW preserves old queries
- ✅ Future-ready for 10-Q and presentation support
- ✅ Migration script preserves all existing data

**Next Steps:**
1. Deploy updated code
2. Run migration script (Step 2 above)
3. Verify migration (Step 3 above)
4. Test API endpoints
5. Wait 7 days, then optionally drop old table

---

## Migration Execution

**To execute the migration after deployment:**

1. SSH into Render shell or use database GUI
2. Copy the migration SQL from Step 2
3. Execute in PostgreSQL
4. Verify with Step 3 queries
5. Test API endpoints
6. Monitor logs for any errors

**Expected Results:**
- ~10 rows migrated (your current 10-K profiles)
- All API endpoints work identically
- New 10-Ks save to `sec_filings` automatically
- Ready for future 10-Q and presentation support
