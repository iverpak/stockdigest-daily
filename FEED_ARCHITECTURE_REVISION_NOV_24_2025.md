# Feed Architecture Revision - November 24, 2025
**Status:** 🔴 DISCUSSION IN PROGRESS
**Database State:** Wiped clean - fresh start opportunity
**Context:** Feed contamination investigation revealed deeper architectural issues

---

## 📋 Executive Summary

During investigation of feed contamination issues (AMD → Intel upstream changes), we discovered that:

1. **AI Enhancement is running daily** for existing tickers (should only run for NEW tickers)
2. **ticker_reference is being updated daily** by AI (should only update via CSV)
3. **Multiple feeds exist for same relationship** (AMD + Intel both "upstream" for SMCI)
4. **Relationships are changing unintentionally** due to `force_refresh=True` in `/admin/init`

**Root Cause:** AI enhancement was designed for initial population when CSV was incomplete. Now that we have 5000 fully populated tickers, AI should ONLY run for new tickers not in CSV.

---

## 🔍 Problems Identified

### Problem #1: Unintended AI Enhancement
**Location:** Line 20628 in `admin_init()`, Line 17828 in job processing

```python
metadata = get_or_create_enhanced_ticker_metadata(isolated_ticker, force_refresh=True)  # ❌
```

**What Happens:**
1. Every `/admin/init` call triggers AI enhancement
2. AI regenerates competitors, upstream, downstream (non-deterministic)
3. `update_ticker_reference_ai_data()` UPDATEs database (line 11912-11934)
4. `ai_enhanced_at` timestamp updates to NOW() (line 11933)
5. New relationships create new feeds, old feeds remain active

**Evidence:**
```sql
SELECT ai_enhanced_at FROM ticker_reference WHERE ticker = 'SMCI';
-- Result: 2025-11-24 05:19:23 (TODAY!)
```

**Impact:**
- AMD was upstream → AI regenerates → Intel is upstream → Repeat
- Multiple feeds exist for "upstream" relationship
- Relationships drift over time

---

### Problem #2: Multiple Feeds Per Relationship
**Location:** Line 2898 in `associate_ticker_with_feed_new_architecture()`

```python
INSERT INTO ticker_feeds (ticker, feed_id, category, value_chain_type)
VALUES (%s, %s, %s, %s)
ON CONFLICT (ticker, feed_id) DO UPDATE SET ...
```

**The Issue:**
- Conflict check: `UNIQUE(ticker, feed_id)`
- AMD feed_id=100, Intel feed_id=200
- No conflict because different feed_ids
- Both relationships coexist:
  ```sql
  (SMCI, 100, 'value_chain', 'upstream')  -- AMD
  (SMCI, 200, 'value_chain', 'upstream')  -- Intel
  ```

**Impact:**
- Accumulates orphaned relationships over time
- Unclear which is "current" upstream entity
- Queries return multiple upstream feeds

---

### Problem #3: No Enforcement of CSV as Source of Truth
**Current Flow:**
```
CSV → Database → AI overwrites → New feeds → Old feeds linger → Repeat daily
```

**Should Be:**
```
CSV → Database (one-time) → Feeds (one-time) → NEVER CHANGE (unless CSV updated)
```

**Gap:** No validation that database matches CSV, AI can override at any time

---

## 🎯 User Requirements (Clarified)

### Requirement #1: ticker_reference.csv is Source of Truth
- CSV is manually curated (or updated via controlled process)
- Database should always match CSV
- Updates to relationships should flow: CSV → Database → Feeds
- AI should NOT update existing tickers

### Requirement #2: Updates Are Allowed
- **NOT immutable** - relationships CAN change over time
- Example: Q1 upstream=AMD, Q2 upstream=Intel (legitimate business change)
- But changes must come from CSV updates, not AI regeneration

### Requirement #3: AI Only for New Tickers
- Currently have 5000 US tickers fully populated in CSV
- AI enhancement ONLY for:
  - New companies not in CSV
  - Foreign tickers not in CSV
  - Edge cases where CSV has no data
- If ticker exists in CSV/database, NEVER run AI

### Requirement #4: Support Multiple Entities
- Up to 2 upstream entities per ticker
- Up to 2 downstream entities per ticker
- Up to 3 competitors per ticker
- Schema already supports this (upstream_1/2, downstream_1/2, competitor_1/2/3)

### Requirement #5: Historical Context Preservation
- Old articles should retain their relationship context
- If AMD was upstream in January, those articles should still show "upstream"
- Even if Intel becomes upstream in February

---

## 🏗️ Proposed Architecture

### Design Principle: CSV-Driven Controlled Mutability

```
┌─────────────────────────────────────┐
│ ticker_reference.csv (GitHub)       │  ← Source of Truth
│ - Version controlled                │
│ - Manually curated                  │
└─────────────────────────────────────┘
              ↓ sync
┌─────────────────────────────────────┐
│ ticker_reference (Database)         │  ← Matches CSV
│ - Updates via CSV sync only         │
│ - AI only for NEW tickers           │
└─────────────────────────────────────┘
              ↓ /admin/init
┌─────────────────────────────────────┐
│ feeds + ticker_feeds (Database)     │  ← Syncs to ticker_reference
│ - Active feeds match current CSV    │
│ - Old feeds deactivated (preserved) │
└─────────────────────────────────────┘
              ↓ ingest
┌─────────────────────────────────────┐
│ articles + ticker_articles          │  ← Immutable historical record
│ - Never update feed_id              │
│ - Preserves historical context      │
└─────────────────────────────────────┘
```

---

## 💡 Proposed Solutions

### Solution #1: Prevent AI for Existing Tickers

**Change:** Check database FIRST, ignore `force_refresh` if ticker exists

```python
def get_or_create_enhanced_ticker_metadata(ticker: str, force_refresh: bool = False) -> Dict:
    # STEP 1: Check database FIRST (even if force_refresh=True)
    db_config = get_ticker_config(ticker)

    if db_config:
        # ✅ Ticker exists in database
        # ✅ Return database data (NO AI!)
        # ❌ Ignore force_refresh (database is source of truth)
        LOG.info(f"[{ticker}] Using existing data from ticker_reference (NO AI)")
        return build_metadata_from_config(db_config)

    # STEP 2: Ticker NOT in database (new ticker)
    if OPENAI_API_KEY:
        # ✅ Run AI enhancement ONCE
        LOG.info(f"[{ticker}] New ticker - running AI enhancement")
        ai_metadata = generate_enhanced_ticker_metadata_with_ai(ticker)

        # ✅ INSERT into database (never UPDATE existing)
        insert_ticker_reference(ticker, ai_metadata)

        return ai_metadata
```

**Benefits:**
- ✅ AI never runs for existing tickers
- ✅ Database is source of truth
- ✅ Eliminates daily regeneration bug
- ✅ Relationships stable unless CSV changes

---

### Solution #2: Feed Sync on /admin/init

**Change:** Sync feeds to match current ticker_reference

```python
def sync_feeds_to_ticker_reference(ticker: str):
    """
    Sync feeds to match current ticker_reference.
    Called by /admin/init.

    - Deactivates feeds no longer in ticker_reference
    - Creates/activates feeds from ticker_reference
    - Preserves inactive feeds (historical context)
    """
    # Read current state from database (NO AI!)
    db_config = get_ticker_config(ticker)

    # Extract desired relationships
    desired_upstream = [db_config['upstream_1_ticker'], db_config['upstream_2_ticker']]
    desired_downstream = [db_config['downstream_1_ticker'], db_config['downstream_2_ticker']]
    desired_competitors = [db_config['competitor_1_ticker'], ...competitor_3_ticker']

    # Sync each relationship type
    sync_value_chain_feeds(ticker, desired_upstream, 'upstream')
    sync_value_chain_feeds(ticker, desired_downstream, 'downstream')
    sync_competitor_feeds(ticker, desired_competitors)
```

**Sync Logic:**
```python
def sync_value_chain_feeds(ticker: str, desired_tickers: list, value_chain_type: str):
    """
    Sync value chain feeds to match desired state.
    """
    # 1. Get currently active feeds
    current_feeds = get_active_feeds(ticker, category='value_chain', value_chain_type=value_chain_type)
    current_tickers = [feed['feed_ticker'] for feed in current_feeds]

    # 2. Calculate diff
    removed = set(current_tickers) - set(desired_tickers)
    added = set(desired_tickers) - set(current_tickers)

    # 3. Deactivate removed
    for removed_ticker in removed:
        UPDATE ticker_feeds SET active = FALSE
        WHERE ticker = %s AND feed_ticker = %s AND value_chain_type = %s;
        LOG.info(f"[{ticker}] Deactivated {value_chain_type}: {removed_ticker}")

    # 4. Create/activate new
    for added_ticker in added:
        feed_id = create_or_get_feed(added_ticker)
        INSERT INTO ticker_feeds ... ON CONFLICT DO UPDATE SET active = TRUE;
        LOG.info(f"[{ticker}] Activated {value_chain_type}: {added_ticker}")
```

**Benefits:**
- ✅ Feeds always match ticker_reference
- ✅ Old feeds deactivated (not deleted - preserves history)
- ✅ New feeds created/reactivated
- ✅ Clean active feed list

---

### Solution #3: CSV Update Workflow

**When Admin Wants to Change Relationships:**

```
Step 1: Update ticker_reference.csv
  Change: upstream_1_ticker: AMD → INTC

Step 2: Commit to GitHub
  git add data/ticker_reference.csv
  git commit -m "SMCI: Change upstream from AMD to Intel (supplier switch)"
  git push origin main

Step 3: Sync to Database
  POST /admin/sync-github
  # Downloads CSV from GitHub
  # Compares with database
  # Updates changed fields only
  # Logs: "SMCI: upstream_1_ticker changed from AMD to INTC"

Step 4: Sync Feeds
  POST /admin/init {"tickers": ["SMCI"]}
  # Reads database: upstream_1_ticker = INTC
  # Compares with active feeds
  # Deactivates: AMD upstream feed (active=FALSE)
  # Creates/activates: INTC upstream feed (active=TRUE)
  # Logs: "SMCI: Deactivated upstream AMD, activated upstream INTC"
```

**Result:**
- ✅ ticker_reference.csv is source of truth (version controlled)
- ✅ Database matches CSV
- ✅ Feeds match database
- ✅ Old AMD articles preserve "upstream" context
- ✅ New INTC articles show "upstream" context
- ✅ Audit trail in git commits and application logs

---

## 📊 Immutability Matrix

| Component | Immutable? | Updates Via | Notes |
|-----------|------------|-------------|-------|
| **ticker_reference.csv** | No | Manual editing | Version controlled in GitHub |
| **ticker_reference (DB)** | No | CSV sync | Always matches CSV |
| **feeds** | Yes | Never | URL is immutable, ON CONFLICT reactivates |
| **ticker_feeds** | No | /admin/init | Syncs to ticker_reference |
| **ticker_articles** | Yes | Never | Historical preservation |
| **articles** | Yes | Never | Global content store |

---

## 🔧 Implementation Plan

### Phase 1: Fix AI Bypass ✅ CRITICAL
**Priority:** Immediate
**Effort:** 2-3 hours

**Changes:**
1. Modify `get_or_create_enhanced_ticker_metadata()` (line 11695)
   - Check database first
   - Skip AI if ticker exists
   - Ignore `force_refresh` parameter for existing tickers

2. Remove `force_refresh=True` from:
   - Line 20628: `/admin/init`
   - Line 17828: Job processing failsafe

**Testing:**
- Verify existing ticker doesn't trigger AI
- Verify new ticker triggers AI once
- Verify `ai_enhanced_at` doesn't update daily

---

### Phase 2: Feed Sync Logic ✅ IMPORTANT
**Priority:** High
**Effort:** 4-6 hours

**New Functions:**
1. `sync_feeds_to_ticker_reference(ticker)` - Main orchestrator
2. `sync_value_chain_feeds(ticker, tickers, type)` - Upstream/downstream sync
3. `sync_competitor_feeds(ticker, tickers)` - Competitor sync
4. `get_active_feeds(ticker, category, value_chain_type)` - Query helper
5. `deactivate_feed(ticker, feed_id)` - Deactivation helper

**Changes to Existing:**
- `/admin/init` calls `sync_feeds_to_ticker_reference()` after feed creation
- Log all sync actions (deactivated/activated feeds)

**Testing:**
- Initial run: Creates feeds from ticker_reference
- Second run: No changes (idempotent)
- After CSV update: Syncs changes correctly

---

### Phase 3: CSV Sync Enhancement ⚠️ NICE-TO-HAVE
**Priority:** Medium
**Effort:** 2-3 hours

**Enhancements:**
1. Add change detection in `sync_ticker_references_from_github()`
2. Log what changed (before/after values)
3. Count updates vs inserts

**Example Log:**
```
[SMCI] ticker_reference updated:
  upstream_1_ticker: AMD → INTC
  upstream_1_name: Advanced Micro Devices → Intel Corporation
[Summary] 1 ticker updated, 0 inserted, 4 fields changed
```

---

### Phase 4: Documentation 📝 NICE-TO-HAVE
**Priority:** Low
**Effort:** 1-2 hours

**Documents to Create:**
1. Admin guide: "How to Update Ticker Relationships"
2. Architecture diagram: CSV → Database → Feeds flow
3. Troubleshooting guide: Common issues and solutions

---

## ❓ Open Questions

### Question #1: AI Bypass
**Q:** Should AI NEVER run for existing tickers, even with `force_refresh=True`?
**User Answer:** *(pending)*
**Recommendation:** Yes - database should always be source of truth

---

### Question #2: /admin/init Behavior
**Q:** Should `/admin/init` always sync feeds to match ticker_reference?
**User Answer:** *(pending)*
**Options:**
- A: Always sync (recommended - ensures consistency)
- B: Only sync on first run (faster, but can drift)
- C: Separate endpoint `/admin/sync-feeds` for explicit sync

---

### Question #3: Old Feed Handling
**Q:** When relationships change, should old feeds be:
- A: Deactivated (active=FALSE) - preserves history ✅ Recommended
- B: Deleted - cleaner database
- C: Kept active - simpler logic

**User Answer:** *(pending)*
**Recommendation:** Deactivate (preserves historical context for old articles)

---

### Question #4: CSV Sync Updates
**Q:** Should CSV sync UPDATE existing ticker rows when values differ?
**User Answer:** *(pending)*
**Options:**
- A: Yes - CSV always wins (recommended)
- B: No - manual database updates preferred
- C: Hybrid - log conflicts, require confirmation

---

### Question #5: Idempotency
**Q:** Should `/admin/init` be safe to call multiple times?
**User Answer:** *(pending)*
**Recommendation:** Yes - should be idempotent (no side effects on repeated calls)

---

### Question #6: Separate Sync Endpoint
**Q:** Should we add `/admin/sync-feeds` separate from `/admin/init`?
**User Answer:** *(pending)*
**Options:**
- A: Keep in `/admin/init` (simpler)
- B: Separate endpoint (more control)

---

### Question #7: Immediate Cleanup
**Q:** For wiped database, should we:
- A: Implement full solution before any data load
- B: Load CSV first, then implement fixes
- C: Partial fix (AI bypass only), then full solution

**User Answer:** *(pending)*
**Recommendation:** Option A - implement full solution for clean start

---

## 🧪 Testing Strategy

### Test Scenario #1: Fresh Database
```
1. Wipe database
2. Upload ticker_reference.csv (5000 tickers)
3. Call /admin/init for 10 tickers
4. Verify:
   - No AI calls made (all data from CSV)
   - Feeds created from CSV data
   - ai_enhanced_at is NULL (not generated)
```

---

### Test Scenario #2: New Ticker
```
1. Add foreign ticker NOT in CSV (e.g., "SAP.DE")
2. Call /admin/init for SAP.DE
3. Verify:
   - AI enhancement runs (ticker not in database)
   - Data inserted into ticker_reference
   - Feeds created from AI metadata
   - ai_enhanced_at is set
4. Call /admin/init again for SAP.DE
5. Verify:
   - AI does NOT run (ticker now in database)
   - ai_enhanced_at unchanged
```

---

### Test Scenario #3: Relationship Change
```
1. Initial: SMCI upstream = AMD, NVDA
2. Update CSV: SMCI upstream = NVDA, INTC (drop AMD, add INTC)
3. Sync CSV to database
4. Call /admin/init for SMCI
5. Verify:
   - AMD feed deactivated (active=FALSE)
   - INTC feed created/activated
   - NVDA feed unchanged (still active)
6. Query old AMD articles
7. Verify:
   - Still show "upstream" context (not orphaned)
```

---

### Test Scenario #4: Idempotency
```
1. Call /admin/init for SMCI
2. Verify feeds created
3. Call /admin/init again for SMCI
4. Verify:
   - No duplicate feeds created
   - No AI calls made
   - Same active feeds as step 2
5. Repeat 10 times
6. Verify consistent state
```

---

## 📝 Next Steps

**Immediate:**
1. User reviews this document
2. User answers open questions
3. Confirm implementation approach

**Before Coding:**
1. Finalize all architectural decisions
2. Get explicit approval on:
   - AI bypass logic
   - Feed sync approach
   - CSV update workflow

**Implementation Order:**
1. Phase 1: AI bypass (blocks everything else)
2. Phase 2: Feed sync logic
3. Phase 3: CSV sync enhancements
4. Phase 4: Documentation

---

## 📚 Related Documents

- `NOVEMBER_24_2025_REFACTOR_COMPLETION.md` - Original contamination fix
- `FEED_CONTAMINATION_FIX_PLAN.md` - Initial refactor plan
- `VALUE_CHAIN_IMPLEMENTATION_STATUS.md` - Value chain feature docs
- `CLAUDE.md` - Overall architecture documentation

---

## 🔄 Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-11-24 | Claude | Initial draft - architectural discussion |

---

**Status:** 🔴 Awaiting user input on open questions before implementation
