# Value Chain Migration Guide

**Date:** October 31, 2025
**Purpose:** Update StockDigest database and CSV for value chain integration

---

## Quick Start (3 Options)

### Option 1: Fresh Start (RECOMMENDED)
**Use if:** You want to start clean with new AI-generated metadata

```bash
# 1. Clear database
psql $DATABASE_URL -f migrations/clear_database.sql

# 2. Run app to recreate schema with new fields
python app.py

# 3. Run /admin/init for your tickers
# This will generate fresh metadata with value chain fields
```

---

### Option 2: Migrate Existing Data
**Use if:** You want to keep existing ticker data and add value chain fields

```bash
# 1. Add new columns to database
psql $DATABASE_URL -f migrations/add_value_chain_fields.sql

# 2. Export existing data
curl -X POST https://stockdigest.app/admin/commit-ticker-csv \
  -H "X-Admin-Token: $ADMIN_TOKEN"

# 3. Download CSV from GitHub
# data/ticker_reference.csv now has 39 columns

# 4. Add 8 empty columns to CSV (upstream_1/2_name/ticker, downstream_1/2_name/ticker)
# Use Excel/Google Sheets to add columns at positions 40-47

# 5. Re-run /admin/init to populate value chain fields via AI
# Or manually populate CSV and re-import
```

---

### Option 3: Manual Database Update (Advanced)
**Use if:** You want to run SQL directly without migration files

```sql
-- Connect to your database
psql $DATABASE_URL

-- Add value chain columns
ALTER TABLE ticker_reference
ADD COLUMN IF NOT EXISTS upstream_1_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS upstream_1_ticker VARCHAR(20),
ADD COLUMN IF NOT EXISTS upstream_2_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS upstream_2_ticker VARCHAR(20),
ADD COLUMN IF NOT EXISTS downstream_1_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS downstream_1_ticker VARCHAR(20),
ADD COLUMN IF NOT EXISTS downstream_2_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS downstream_2_ticker VARCHAR(20);

ALTER TABLE ticker_articles
ADD COLUMN IF NOT EXISTS value_chain_type VARCHAR(10)
CHECK (value_chain_type IN ('upstream', 'downstream'));

ALTER TABLE feeds
ADD COLUMN IF NOT EXISTS value_chain_type VARCHAR(10)
CHECK (value_chain_type IN ('upstream', 'downstream'));
```

---

## CSV Format Changes

### Old CSV (39 columns):
```
ticker,...,competitor_1_name,competitor_1_ticker,competitor_2_name,competitor_2_ticker,competitor_3_name,competitor_3_ticker,ai_generated,ai_enhanced_at
```

### New CSV (47 columns):
```
ticker,...,competitor_1_name,competitor_1_ticker,competitor_2_name,competitor_2_ticker,competitor_3_name,competitor_3_ticker,upstream_1_name,upstream_1_ticker,upstream_2_name,upstream_2_ticker,downstream_1_name,downstream_1_ticker,downstream_2_name,downstream_2_ticker,ai_generated,ai_enhanced_at
```

### New Columns (Positions 40-47):
| Position | Column Name | Type | Description | Example |
|----------|-------------|------|-------------|---------|
| 40 | upstream_1_name | VARCHAR(255) | First supplier name | Panasonic |
| 41 | upstream_1_ticker | VARCHAR(20) | First supplier ticker | 6752.T |
| 42 | upstream_2_name | VARCHAR(255) | Second supplier name | CATL |
| 43 | upstream_2_ticker | VARCHAR(20) | Second supplier ticker | 300750.SZ |
| 44 | downstream_1_name | VARCHAR(255) | First customer name | Microsoft |
| 45 | downstream_1_ticker | VARCHAR(20) | First customer ticker | MSFT |
| 46 | downstream_2_name | VARCHAR(255) | Second customer name | Apple |
| 47 | downstream_2_ticker | VARCHAR(20) | Second customer ticker | AAPL |

---

## Example CSV Rows

### Tesla (Upstream-heavy):
```csv
TSLA,US,Tesla Inc,Automotive,...,Panasonic,6752.T,CATL,300750.SZ,,,,
```
- Horizontal competitors: (columns 33-38)
- Upstream: Panasonic (6752.T), CATL (300750.SZ)
- Downstream: None (empty columns 44-47)

### Intel (Both):
```csv
INTC,US,Intel Corporation,Semiconductors,...,TSMC,TSM,,MSFT,AAPL,,
```
- Upstream: TSMC (TSM)
- Downstream: Microsoft (MSFT), Apple (AAPL)

### Netflix (No value chain):
```csv
NFLX,US,Netflix Inc,Entertainment,...,,,,,,,
```
- All value chain columns empty

---

## Verification Queries

After migration, verify the schema:

```sql
-- Check ticker_reference columns
SELECT column_name, data_type, character_maximum_length
FROM information_schema.columns
WHERE table_name = 'ticker_reference'
ORDER BY ordinal_position;

-- Check value chain data
SELECT ticker, company_name,
       upstream_1_name, upstream_1_ticker,
       upstream_2_name, upstream_2_ticker,
       downstream_1_name, downstream_1_ticker,
       downstream_2_name, downstream_2_ticker
FROM ticker_reference
WHERE upstream_1_name IS NOT NULL
   OR downstream_1_name IS NOT NULL;

-- Check ticker_articles value_chain_type
SELECT DISTINCT value_chain_type
FROM ticker_articles
WHERE value_chain_type IS NOT NULL;

-- Check feeds value_chain_type
SELECT name, value_chain_type
FROM feeds
WHERE value_chain_type IS NOT NULL
LIMIT 10;
```

---

## Testing Steps

1. **Run migration:**
   ```bash
   psql $DATABASE_URL -f migrations/add_value_chain_fields.sql
   ```

2. **Test with one ticker:**
   ```bash
   # Use /admin/init endpoint for Tesla
   curl -X POST "https://stockdigest.app/admin/init?tickers=TSLA" \
     -H "X-Admin-Token: $ADMIN_TOKEN"
   ```

3. **Verify metadata generated:**
   ```sql
   SELECT ticker, company_name,
          upstream_1_name, upstream_2_name,
          downstream_1_name, downstream_2_name
   FROM ticker_reference
   WHERE ticker = 'TSLA';
   ```

4. **Check feeds created (should be 19 total):**
   ```sql
   SELECT COUNT(*) FROM ticker_feeds WHERE ticker = 'TSLA';
   ```

5. **Verify feed breakdown:**
   ```sql
   SELECT f.name, tf.category, f.value_chain_type
   FROM ticker_feeds tf
   JOIN feeds f ON tf.feed_id = f.id
   WHERE tf.ticker = 'TSLA'
   ORDER BY tf.category, f.value_chain_type;
   ```

   Expected:
   - 2 company feeds (category: company)
   - 3 industry feeds (category: industry)
   - 6 competitor feeds (category: competitor)
   - 8 value chain feeds (category: value_chain, 4 upstream + 4 downstream)

---

## Rollback (If Something Goes Wrong)

If you need to undo the migration:

```sql
-- Remove value chain columns from ticker_reference
ALTER TABLE ticker_reference
DROP COLUMN IF EXISTS upstream_1_name,
DROP COLUMN IF EXISTS upstream_1_ticker,
DROP COLUMN IF EXISTS upstream_2_name,
DROP COLUMN IF EXISTS upstream_2_ticker,
DROP COLUMN IF EXISTS downstream_1_name,
DROP COLUMN IF EXISTS downstream_1_ticker,
DROP COLUMN IF EXISTS downstream_2_name,
DROP COLUMN IF EXISTS downstream_2_ticker;

-- Remove value_chain_type from ticker_articles
ALTER TABLE ticker_articles
DROP COLUMN IF EXISTS value_chain_type;

-- Remove value_chain_type from feeds
ALTER TABLE feeds
DROP COLUMN IF EXISTS value_chain_type;
```

---

## Common Issues

### Issue 1: "column already exists"
**Cause:** Migration already run
**Solution:** Safe to ignore (all ALTER statements use `IF NOT EXISTS`)

### Issue 2: CSV import fails with "wrong number of columns"
**Cause:** CSV missing new 8 value chain columns
**Solution:** Add 8 empty columns to CSV at positions 40-47

### Issue 3: Feeds not created for value chain companies
**Cause:** ticker_reference missing value chain data
**Solution:** Run `/admin/init` to regenerate metadata with AI

### Issue 4: "relation does not exist" error
**Cause:** Fresh database, tables not created yet
**Solution:** Run `python app.py` once to trigger `ensure_schema()`

---

## Support

If you encounter issues:
1. Check migration logs for errors
2. Verify database schema with verification queries
3. Test with single ticker first (TSLA recommended)
4. Check commit history: e921296 (latest bugfix)

**Migration Files:**
- `migrations/add_value_chain_fields.sql` - Schema updates
- `migrations/clear_database.sql` - Database reset
- `migrations/MIGRATION_GUIDE.md` - This file
