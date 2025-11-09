# Migration: Add raw_content Column to sec_8k_filings

## Problem

The 8-K summary generation is failing with this error:
```
column "raw_content" of relation "sec_8k_filings" does not exist
```

## Root Cause

The code schema includes a `raw_content TEXT` column (line 1696 in app.py) to store the raw extracted 8-K content before AI formatting, but this column was never added to the production database.

The `ensure_schema()` function uses `CREATE TABLE IF NOT EXISTS`, which doesn't add missing columns to existing tables.

## Solution

Run the migration script to add the missing column.

### Option 1: Run via Render Shell (Recommended)

1. Go to Render Dashboard → Your Service → Shell
2. Run the migration:
   ```bash
   python migrate_add_8k_raw_content.py
   ```

3. Expected output:
   ```
   🔄 Connecting to database...
   🔍 Checking if raw_content column exists...
   📝 Adding raw_content column to sec_8k_filings...
   ✅ Successfully added raw_content column

   📋 Current schema for sec_8k_filings:
   ------------------------------------------------------------
     id                             integer              NOT NULL
     ticker                         character varying    NOT NULL
     company_name                   character varying    NULL
     cik                           character varying    NULL
     accession_number              character varying    NOT NULL
     filing_date                   date                 NOT NULL
     filing_title                  character varying    NOT NULL
     item_codes                    character varying    NULL
     sec_html_url                  text                 NOT NULL
     raw_content                   text                 NULL    ← NEW
     summary_text                  text                 NOT NULL
     ai_provider                   character varying    NOT NULL
     ai_model                      character varying    NULL
     job_id                        character varying    NULL
     processing_duration_seconds   integer              NULL
     monitored                     boolean              NULL
     last_checked_at              timestamp with tz    NULL
     generated_at                 timestamp with tz    NULL
   ------------------------------------------------------------

   ✅ Migration completed successfully!
   ```

### Option 2: Manual SQL (Alternative)

If you prefer to run SQL directly:

```sql
-- Check if column exists
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'sec_8k_filings'
AND column_name = 'raw_content';

-- Add column if it doesn't exist
ALTER TABLE sec_8k_filings
ADD COLUMN raw_content TEXT;

-- Verify
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'sec_8k_filings'
ORDER BY ordinal_position;
```

## What raw_content Stores

The 8-K generation workflow now has two outputs:

1. **raw_content** (TEXT) - Raw extracted content from SEC Edgar
   - Exhibit 99.1 press release (if available)
   - Main 8-K body (fallback)
   - Used for Email #1 (quick raw content review)

2. **summary_text** (TEXT) - Gemini-formatted summary
   - 800-1,500 words
   - Noise filtered, 90% retention
   - Used for Email #2 (formatted summary for research library)

## After Migration

1. The migration is idempotent (safe to run multiple times)
2. Existing 8-K records won't have `raw_content` (will be NULL)
3. New 8-K generations will populate both fields
4. You can retry the PLD 8-K generation immediately

## Retry PLD 8-K Generation

After running the migration:

1. Go to `/admin/research?token=YOUR_TOKEN`
2. Scroll to "8-K SEC Releases" section
3. Find the PLD filing you want to generate
4. Click "Generate Summary (5-10 min)"
5. You should receive both emails:
   - Email #1: Raw content from SEC
   - Email #2: Gemini-formatted summary

## Code References

- Table schema: `app.py:1696`
- Extract raw content: `app.py:23672`
- Format for Email #1: `app.py:23687-23705`
- Database INSERT: `app.py:23738-23764`
- Email #1 template: `templates/email_8k_raw_content.html`
