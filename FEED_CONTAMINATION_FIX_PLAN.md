# Feed Contamination Fix - Implementation Plan

**Date:** November 24, 2025
**Status:** Ready for Review
**Goal:** Eliminate cross-ticker feed contamination by moving `value_chain_type` to `ticker_feeds` table

---

## Executive Summary

### The Problem
Feeds are shared across tickers (by design), but `value_chain_type` is stored at the feed level. This causes contamination:
- INTC creates DELL feed with `value_chain_type='downstream'`
- SMCI tries to use same DELL feed as competitor
- SMCI inherits `value_chain_type='downstream'` from INTC ❌

### The Root Cause
```sql
-- Current (BROKEN):
feeds table:
  - value_chain_type VARCHAR(10)  ❌ Per-ticker data at feed level!

ticker_feeds table:
  - (missing value_chain_type)    ❌ Should be here!
```

### The Solution
Move `value_chain_type` from `feeds` (shared) to `ticker_feeds` (per-relationship).

---

## Architecture Comparison

### Current Schema (BROKEN)

**feeds table:**
- id, url, name, search_keyword
- competitor_ticker, company_name
- **value_chain_type** ❌ (causes contamination)
- retain_days, active, created_at, updated_at

**ticker_feeds table:**
- id, ticker, feed_id
- category
- active, created_at, updated_at
- (missing value_chain_type) ❌

### Target Schema (CORRECT)

**feeds table:**
- id, url, name, search_keyword
- competitor_ticker, company_name
- retain_days, active, created_at, updated_at
- ~~value_chain_type~~ ✅ (removed)

**ticker_feeds table:**
- id, ticker, feed_id
- category
- **value_chain_type** ✅ (added here)
- active, created_at, updated_at

---

## Evidence of Contamination

### Query Results Showing the Problem:

```sql
-- Feed #495 shared by INTC and SMCI:
SELECT f.id, f.name, f.value_chain_type, tf.ticker, tf.category
FROM feeds f
JOIN ticker_feeds tf ON f.id = tf.feed_id
WHERE f.competitor_ticker = 'DELL';

Results:
id=495 | Downstream: Dell... | downstream | INTC | value_chain  ✅ Correct for INTC
id=495 | Downstream: Dell... | downstream | SMCI | competitor   ❌ WRONG for SMCI!
```

### Scope of Contamination:

20+ feeds shared across multiple tickers with conflicting value_chain_type needs:
- AMZN feed: Used by 7 tickers (company, competitor, value_chain)
- MSFT feed: Used by 5 tickers (competitor, value_chain)
- NVDA feed: Used by 4 tickers (company, competitor, value_chain)

**Every shared feed is potentially contaminated.**

---

## Phase 1: Database Schema Changes

### Step 1.1: Add value_chain_type to ticker_feeds

```sql
ALTER TABLE ticker_feeds
ADD COLUMN value_chain_type VARCHAR(10)
CHECK (value_chain_type IN ('upstream', 'downstream') OR value_chain_type IS NULL);
```

**Why:** This is where per-ticker relationship metadata belongs.

**Safe:** Adding a nullable column doesn't break existing data.

---

### Step 1.2: Drop value_chain_type from feeds

```sql
ALTER TABLE feeds
DROP COLUMN value_chain_type;
```

**Why:** Remove the contamination source.

**Safe:** We're wiping feeds table anyway, so no data loss concern.

**Timing:** Do this BEFORE recreating feeds to ensure code can't use the old column.

---

## Phase 2: Code Changes

### Change 2.1: upsert_feed_new_architecture()

**Location:** `app.py` line ~2767

**Current Code:**
```python
def upsert_feed_new_architecture(url: str, name: str, search_keyword: str = None,
                                competitor_ticker: str = None, company_name: str = None, retain_days: int = 90,
                                value_chain_type: str = None) -> int:
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO feeds (url, name, search_keyword, competitor_ticker, company_name, retain_days, value_chain_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET
                active = TRUE,
                company_name = COALESCE(EXCLUDED.company_name, feeds.company_name),
                value_chain_type = COALESCE(EXCLUDED.value_chain_type, feeds.value_chain_type),
                updated_at = NOW()
            RETURNING id;
        """, (url, name, search_keyword, competitor_ticker, company_name, retain_days, value_chain_type))
```

**New Code:**
```python
def upsert_feed_new_architecture(url: str, name: str, search_keyword: str = None,
                                competitor_ticker: str = None, company_name: str = None, retain_days: int = 90,
                                value_chain_type: str = None) -> int:
    """
    IMPORTANT: value_chain_type parameter is IGNORED (kept for backward compatibility).
    This field now lives in ticker_feeds table, not feeds table.
    """
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO feeds (url, name, search_keyword, competitor_ticker, company_name, retain_days)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET
                active = TRUE,
                company_name = COALESCE(EXCLUDED.company_name, feeds.company_name),
                updated_at = NOW()
            RETURNING id;
        """, (url, name, search_keyword, competitor_ticker, company_name, retain_days))
```

**Key Changes:**
1. ✅ Keep `value_chain_type` parameter (backward compatibility - avoid updating 34 call sites)
2. ✅ Ignore the parameter value (don't insert into database)
3. ✅ Remove from INSERT columns
4. ✅ Remove from VALUES
5. ✅ Remove from ON CONFLICT UPDATE

**Why keep the parameter?**
- 34 places in code currently pass this parameter
- Silently ignoring it is safer than mass refactor
- Clean separation: accept but don't use

---

### Change 2.2: associate_ticker_with_feed_new_architecture()

**Location:** Search for this function (need to find exact line)

**Current Code:**
```python
def associate_ticker_with_feed_new_architecture(ticker: str, feed_id: int, category: str) -> bool:
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ticker_feeds (ticker, feed_id, category)
            VALUES (%s, %s, %s)
            ON CONFLICT (ticker, feed_id) DO UPDATE SET
                category = EXCLUDED.category,
                active = TRUE,
                updated_at = NOW()
            RETURNING id;
        """, (ticker, feed_id, category))
        return True
```

**New Code:**
```python
def associate_ticker_with_feed_new_architecture(ticker: str, feed_id: int, category: str,
                                               value_chain_type: str = None) -> bool:
    """
    Associate a ticker with a feed, storing relationship-specific metadata.

    Args:
        value_chain_type: 'upstream', 'downstream', or None
                         - Set when category='value_chain'
                         - NULL for company/industry/competitor categories
    """
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ticker_feeds (ticker, feed_id, category, value_chain_type)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (ticker, feed_id) DO UPDATE SET
                category = EXCLUDED.category,
                value_chain_type = EXCLUDED.value_chain_type,
                active = TRUE,
                updated_at = NOW()
            RETURNING id;
        """, (ticker, feed_id, category, value_chain_type))
        return True
```

**Key Changes:**
1. ✅ Add `value_chain_type` parameter
2. ✅ Add to INSERT columns
3. ✅ Add to VALUES
4. ✅ Add to ON CONFLICT UPDATE (always overwrite with new value)

**ON CONFLICT Decision: OVERWRITE**
- `value_chain_type = EXCLUDED.value_chain_type` (always use new value)
- **Why:** ticker_reference is source of truth
- **Alternative:** `COALESCE(ticker_feeds.value_chain_type, EXCLUDED.value_chain_type)` would preserve existing

---

### Change 2.3: Update 10 Call Sites

All calls to `associate_ticker_with_feed_new_architecture()` need to pass `value_chain_type`.

#### **Company Feeds (Line ~2882):**
```python
# BEFORE
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "company"):

# AFTER
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "company", value_chain_type=None):
```

#### **Industry Feeds (Line ~2902):**
```python
# BEFORE
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "industry"):

# AFTER
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "industry", value_chain_type=None):
```

#### **Competitor Feeds - Google News (Line ~2933):**
```python
# BEFORE
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "competitor"):

# AFTER
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "competitor", value_chain_type=None):
```

#### **Competitor Feeds - Yahoo Finance (Line ~2950):**
```python
# BEFORE
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "competitor"):

# AFTER
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "competitor", value_chain_type=None):
```

#### **Upstream Feeds - Google News (Line ~2985):**
```python
# BEFORE
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "value_chain"):

# AFTER
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "value_chain", value_chain_type='upstream'):
```

#### **Upstream Feeds - Yahoo Finance (Line ~3003):**
```python
# BEFORE
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "value_chain"):

# AFTER
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "value_chain", value_chain_type='upstream'):
```

#### **Downstream Feeds - Google News (Line ~3037):**
```python
# BEFORE
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "value_chain"):

# AFTER
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "value_chain", value_chain_type='downstream'):
```

#### **Downstream Feeds - Yahoo Finance (Line ~3055):**
```python
# BEFORE
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "value_chain"):

# AFTER
if associate_ticker_with_feed_new_architecture(ticker, feed_id, "value_chain", value_chain_type='downstream'):
```

**Total: 10 call sites to update**

**Pattern:**
- Company/Industry/Competitor: Pass `value_chain_type=None`
- Upstream: Pass `value_chain_type='upstream'`
- Downstream: Pass `value_chain_type='downstream'`

---

### Change 2.4: Find and Update Query References

**Search for:**
- `f.value_chain_type`
- `feeds.value_chain_type`
- Any JOINs or queries using this column

**Common pattern:**
```sql
-- BEFORE
SELECT f.id, f.name, f.value_chain_type
FROM feeds f
WHERE f.value_chain_type = 'downstream'

-- AFTER
SELECT f.id, f.name, tf.value_chain_type
FROM feeds f
JOIN ticker_feeds tf ON f.id = tf.feed_id
WHERE tf.value_chain_type = 'downstream'
```

**Action Required:**
1. Search entire codebase for references
2. Document each location
3. Update queries to use `ticker_feeds.value_chain_type`

---

## Phase 3: Data Cleanup & Recreation

### Step 3.1: Complete Database Wipe

```sql
-- Delete all feed associations
TRUNCATE TABLE ticker_feeds CASCADE;

-- Delete all feeds
TRUNCATE TABLE feeds CASCADE;

-- Wipe ticker_reference (will reimport from clean CSV)
TRUNCATE TABLE ticker_reference CASCADE;
```

**Why:** Start completely fresh with zero contamination.

**Safe:** You have clean source CSV to reimport.

---

### Step 3.2: Import Clean CSV

**Method 1: Via API**
```bash
POST /admin/init
Body: { "tickers": ["SMCI", "INTC", ...] }
```

**Method 2: Direct Database Import**
```python
# Via existing import function
import_ticker_reference_from_csv_content(csv_content)
```

**What happens:**
1. ticker_reference table populated from clean CSV
2. No feeds created yet (that's next step)

---

### Step 3.3: Recreate All Feeds

```bash
POST /admin/init
Body: { "tickers": ["SMCI", "INTC", "NVDA", ...] }
```

**This will:**
1. Call `get_or_create_enhanced_ticker_metadata()` for each ticker
2. Read clean data from ticker_reference
3. Call `create_feeds_for_ticker_new_architecture()`
4. Create feeds WITHOUT value_chain_type in feeds table ✅
5. Create ticker_feeds associations WITH value_chain_type ✅

---

## Phase 4: Validation

### Validation 4.1: Check DELL Feed (Cross-Ticker Test)

```sql
SELECT
    f.id,
    f.name,
    f.competitor_ticker,
    tf.ticker,
    tf.category,
    tf.value_chain_type
FROM feeds f
JOIN ticker_feeds tf ON f.id = tf.feed_id
WHERE f.competitor_ticker = 'DELL'
ORDER BY tf.ticker, tf.category;
```

**Expected Result:**
```
Feed ID | Name              | Entity | Ticker | Category    | value_chain_type
--------|-------------------|--------|--------|-------------|------------------
495     | Google News: Dell | DELL   | INTC   | value_chain | downstream       ✅
495     | Google News: Dell | DELL   | SMCI   | competitor  | NULL             ✅
496     | Yahoo Finance...  | DELL   | INTC   | value_chain | downstream       ✅
496     | Yahoo Finance...  | DELL   | SMCI   | competitor  | NULL             ✅
```

**Success Criteria:**
- ✅ Same feed_id used by both tickers
- ✅ INTC has value_chain_type='downstream'
- ✅ SMCI has value_chain_type=NULL
- ✅ NO CONTAMINATION!

---

### Validation 4.2: Verify Column Locations

```sql
-- This should ERROR (column doesn't exist):
SELECT value_chain_type FROM feeds LIMIT 1;
-- Expected: ERROR: column "value_chain_type" does not exist

-- This should WORK:
SELECT value_chain_type FROM ticker_feeds LIMIT 1;
-- Expected: Returns NULL or 'upstream' or 'downstream'
```

---

### Validation 4.3: Check for Contamination Patterns

```sql
-- Find any feeds shared by multiple tickers with different needs:
SELECT
    f.id,
    f.name,
    f.competitor_ticker,
    COUNT(DISTINCT tf.ticker) as ticker_count,
    COUNT(DISTINCT tf.value_chain_type) as unique_value_chain_types,
    STRING_AGG(DISTINCT tf.ticker || ':' || COALESCE(tf.value_chain_type, 'NULL'), ', ') as relationships
FROM feeds f
JOIN ticker_feeds tf ON f.id = tf.feed_id
WHERE f.competitor_ticker IS NOT NULL
GROUP BY f.id, f.name, f.competitor_ticker
HAVING COUNT(DISTINCT tf.ticker) > 1
ORDER BY ticker_count DESC
LIMIT 20;
```

**Expected Result:**
- Multiple tickers using same feed ✅
- Each with their own value_chain_type ✅
- No conflicts or contamination ✅

---

## Execution Order (Step-by-Step)

### **Part A: Schema Changes (Safe - Run First)**
1. ✅ Connect to database
2. ✅ Run: `ALTER TABLE ticker_feeds ADD COLUMN value_chain_type...`
3. ✅ Run: `ALTER TABLE feeds DROP COLUMN value_chain_type`
4. ✅ Verify columns exist/don't exist

### **Part B: Code Changes (Do NOT Deploy Yet)**
5. ✅ Modify `upsert_feed_new_architecture()` function
6. ✅ Modify `associate_ticker_with_feed_new_architecture()` function
7. ✅ Update 10 call sites to pass value_chain_type
8. ✅ Search and update all query references to feeds.value_chain_type
9. ✅ Review changes, test locally if possible

### **Part C: Deploy & Test**
10. ✅ Deploy code to production
11. ✅ Wipe feeds, ticker_feeds, ticker_reference tables
12. ✅ Import clean ticker_reference.csv
13. ✅ Run `/admin/init` for single test ticker (SMCI)
14. ✅ Run Validation 4.1 query
15. ✅ Check for contamination
16. ✅ If successful, run `/admin/init` for ALL tickers
17. ✅ Run Validation 4.3 (comprehensive contamination check)
18. ✅ Monitor logs for errors

---

## Risk Mitigation

### Risk 1: Code References We Miss
**Mitigation:** Search entire codebase for:
- `f.value_chain_type`
- `feeds.value_chain_type`
- `FROM feeds` (check all JOINs)

### Risk 2: ON CONFLICT Preserves Old Data
**Mitigation:** We're wiping tables, so no old data exists.

### Risk 3: Incomplete Migration
**Mitigation:** Run validation queries before processing all tickers.

### Risk 4: Breaking Other Features
**Mitigation:**
- Keep parameter in `upsert_feed_new_architecture()` for backward compatibility
- Test with single ticker first
- Monitor logs during full migration

---

## Rollback Plan (If Needed)

If something goes wrong:

### Step 1: Restore Schema
```sql
-- Re-add value_chain_type to feeds
ALTER TABLE feeds
ADD COLUMN value_chain_type VARCHAR(10)
CHECK (value_chain_type IN ('upstream', 'downstream') OR value_chain_type IS NULL);

-- Remove from ticker_feeds
ALTER TABLE ticker_feeds
DROP COLUMN value_chain_type;
```

### Step 2: Restore Code
- Revert git commit with code changes
- Redeploy previous version

### Step 3: Restore Data
- You still have clean CSV
- Can reimport and recreate feeds

---

## Success Criteria

✅ **Schema:**
- `ticker_feeds` has `value_chain_type` column
- `feeds` does NOT have `value_chain_type` column

✅ **Code:**
- `upsert_feed_new_architecture()` ignores value_chain_type parameter
- `associate_ticker_with_feed_new_architecture()` accepts and stores value_chain_type
- All 10 call sites updated to pass value_chain_type

✅ **Data:**
- DELL feed shared by INTC (downstream) and SMCI (competitor)
- Each ticker has correct value_chain_type in ticker_feeds
- Zero contamination across all shared feeds

✅ **Validation:**
- All validation queries pass
- No errors in logs
- Feeds work correctly for all tickers

---

## Open Questions / Decisions Needed

### Question 1: ON CONFLICT Behavior
**Current Recommendation:** OVERWRITE (always use new value from ticker_reference)

**Alternative:** PRESERVE (keep existing value if set)

**Decision:** OVERWRITE ✅

---

### Question 2: Search All References First?
Should we search entire codebase for `feeds.value_chain_type` references before making changes?

**Recommendation:** YES - do this as first step of Part B

**Action:** Use grep/IDE search for:
- `f.value_chain_type`
- `feeds.value_chain_type`
- Document all findings

---

### Question 3: Test Locally First?
Can we test these changes in a local environment before production?

**Recommendation:** If possible, yes

**Alternative:** Test with single ticker in production (SMCI)

---

## Timeline Estimate

**Part A (Schema):** 5 minutes
**Part B (Code Changes):** 2-3 hours
**Part C (Deploy & Test):** 1-2 hours
**Total:** 3-5 hours

---

## Appendix: Key Architectural Insights

### Why Feeds Are Shared (By Design)
- Same URL = Same feed (deduplication)
- Avoid parsing same RSS feed multiple times
- Reduce API calls and storage

### Why value_chain_type Must Be Per-Relationship
- DELL is different things to different tickers:
  - INTC: Downstream customer
  - SMCI: Competitor
  - HPE: Competitor
- The feed is about DELL (shared)
- The relationship type varies (per-ticker)

### Comparison to articles/ticker_articles Pattern
This fix makes feeds/ticker_feeds mirror the CORRECT pattern already used for articles:
- `articles` table: Shared content (title, url, domain)
- `ticker_articles` table: Per-ticker relationship (category, value_chain_type, competitor_ticker)

**We're applying the same pattern to feeds!**

---

## Files to Modify

1. **Database Schema:**
   - Run migrations directly via SQL

2. **app.py:**
   - `upsert_feed_new_architecture()` (~line 2767)
   - `associate_ticker_with_feed_new_architecture()` (search for it)
   - 10 call sites in `create_feeds_for_ticker_new_architecture()` (lines 2832-3067)
   - Any queries referencing `feeds.value_chain_type` (TBD - need to search)

3. **This Plan:**
   - `FEED_CONTAMINATION_FIX_PLAN.md` (this file)

---

## Next Steps

1. ✅ Review this plan in detail
2. ✅ Confirm decisions (ON CONFLICT, search references, etc.)
3. ✅ Search for all `feeds.value_chain_type` references
4. ✅ Execute Part A (schema changes)
5. ✅ Execute Part B (code changes)
6. ✅ Execute Part C (deploy & validate)

---

**End of Plan**
