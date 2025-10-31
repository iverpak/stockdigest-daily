# Value Chain Integration - Implementation Status

**Created:** October 31, 2025
**Status:** Phase 1 (Infrastructure) - 40% Complete (6/15 tasks done)
**Last Updated:** After commits 526ac74, 13cfaa6, 7d1b378, 52ff527, 848fe6c, c07d507

---

## Overview

Adding 2 upstream + 2 downstream companies (4 value chain companies) to feed monitoring system alongside 3 horizontal competitors (0-3 flexible).

**Goals:**
- Track suppliers (upstream) and customers (downstream) for material intelligence
- Separate horizontal competition from value chain dynamics
- Create new "Value Chain" section in all 3 emails

**Key Decisions:**
- Keep `competitor` category internally (no rename to `horizontal_competitor`)
- Add `value_chain_type` field to track upstream vs downstream
- Apply same limits as competitors (25 per company = 100 total possible)
- Write value chain-specific triage/summary prompts (not copying industry prompts)

---

## ✅ COMPLETED TASKS (6/15)

### Task 1: Update `generate_claude_ticker_metadata()` Prompt ✅
**File:** `app.py` lines 13332-13688
**Commit:** 526ac74

**Changes:**
1. Updated CRITICAL REQUIREMENTS: "competitors and value chain companies"
2. Renamed "COMPETITORS (exactly 3)" → "HORIZONTAL COMPETITORS (0-3)"
   - Flexible 0-3 count (quality over quantity)
   - Added bad match examples (AvalonBay/ELME, IBM/D-Wave, JPMorgan/Robinhood)
   - Excludes companies already in value chain
3. Added "VALUE CHAIN (0-2 upstream, 0-2 downstream)" section
   - Upstream: >10% COGS or critical enabler
   - Downstream: >5% revenue or demand proxy
   - Materiality ranking (highest % first)
4. Added 4 examples with working JSON:
   - Tesla: 3 horizontal + 2 upstream + 0 downstream
   - Intel: 3 horizontal + 1 upstream + 2 downstream
   - Denison Mines: 3 horizontal + 0 upstream + 2 downstream
   - Netflix: 3 horizontal + 0 upstream + 0 downstream
5. Updated JSON format:
   ```json
   {
     "horizontal_competitors": [
       {"name": "Company", "ticker": "TICKER"}
     ],
     "value_chain": {
       "upstream": [{"name": "Supplier", "ticker": "TICKER"}],
       "downstream": [{"name": "Customer", "ticker": "TICKER"}]
     }
   }
   ```

### Task 2: Create JSON Parsing Helper Function ✅
**File:** `app.py` lines 14295-14345
**Commit:** 13cfaa6

**New Function:** `parse_metadata_to_flat_fields(metadata: Dict) -> Dict`

**Returns 14 fields:**
- `competitor_1_name`, `competitor_1_ticker` (horizontal)
- `competitor_2_name`, `competitor_2_ticker` (horizontal)
- `competitor_3_name`, `competitor_3_ticker` (horizontal)
- `upstream_1_name`, `upstream_1_ticker`
- `upstream_2_name`, `upstream_2_ticker`
- `downstream_1_name`, `downstream_1_ticker`
- `downstream_2_name`, `downstream_2_ticker`

**Features:**
- Parses `horizontal_competitors` array (0-3)
- Parses `value_chain.upstream` array (0-2)
- Parses `value_chain.downstream` array (0-2)
- Backward compatible: Falls back to old `competitors` field
- Initializes all fields to None

**Updated Locations:**
1. `get_or_create_enhanced_ticker_metadata()` (line 14499-14501)
2. `update_ticker_reference_ai_data()` (line 14537-14538)

### Task 3: Update `generate_openai_ticker_metadata()` Prompt ✅
**File:** `app.py` lines 13801-14114
**Commit:** 526ac74

**Changes:**
- Same updates as Claude prompt for consistency
- Updated CRITICAL REQUIREMENTS
- Added HORIZONTAL COMPETITORS section
- Added VALUE CHAIN section with all 4 examples
- Updated JSON format specification

### Task 4: Update Database Schema ✅
**File:** `app.py` lines 1307-1315, 1335, 1375, 1422-1429
**Commit:** 52ff527

**Changes:**
1. **ticker_reference table:** Added 8 value chain columns
   - upstream_1_name, upstream_1_ticker
   - upstream_2_name, upstream_2_ticker
   - downstream_1_name, downstream_1_ticker
   - downstream_2_name, downstream_2_ticker
   - Both in CREATE TABLE and ALTER TABLE statements

2. **ticker_articles table:** Added value_chain_type column
   - VARCHAR(10) with CHECK constraint (upstream/downstream/NULL)

3. **feeds table:** Added value_chain_type column
   - VARCHAR(10) with CHECK constraint (upstream/downstream/NULL)

4. **SQL statements updated:**
   - `update_ticker_reference_ai_data()` - INSERT/UPDATE with 8 fields
   - `get_ticker_reference()` (2 functions) - SELECT with 8 fields

### Task 5: Update Feed Creation Logic ✅
**File:** `app.py` lines 2135-2435
**Commit:** 848fe6c

**Changes:**
1. **Updated `upsert_feed_new_architecture()`:**
   - Added `value_chain_type` parameter (default: None)
   - Stores value in feeds table
   - COALESCE update preserves existing values

2. **Added Section 4: Upstream Value Chain feeds**
   - 0-2 suppliers × 2 sources (Google + Yahoo)
   - Feed names: "Upstream: {Company Name}"
   - value_chain_type='upstream'
   - Category: 'value_chain'

3. **Added Section 5: Downstream Value Chain feeds**
   - 0-2 customers × 2 sources (Google + Yahoo)
   - Feed names: "Downstream: {Company Name}"
   - value_chain_type='downstream'
   - Category: 'value_chain'

**Result:** 19 feeds per ticker (was 11, now +73%)

### Task 6: Update CSV Import/Export ✅
**Files:** `app.py` lines 3591-3604, 3649-3650, 3667-3675, 3704-3756, 3990-3993, 4033-4036
**Commit:** c07d507

**Changes:**
1. **Import function (`import_ticker_reference_from_csv_content`):**
   - Added 8 value chain field parsing (lines 3591-3604)
   - Added to NULL byte cleaning list
   - Added ticker normalization/validation for value chain fields
   - Updated bulk INSERT statement (8 columns, 8 parameters, 8 UPDATE assignments)

2. **Export function (`export_ticker_references_to_csv`):**
   - Updated SELECT to include 8 value chain columns
   - Updated CSV headers list (47 columns total, was 39)

**CSV Column Order:**
- Industry keywords (3 fields)
- Horizontal competitors (6 fields)
- Value chain upstream (4 fields)
- Value chain downstream (4 fields)

---

## 🔄 REMAINING TASKS (9/15)

### PHASE 1: Core Infrastructure (0 remaining - COMPLETE ✅)

All Phase 1 tasks complete!

### PHASE 2: Processing Pipeline (6 remaining)

#### Task 7: Add value_chain Category to Ingestion Logic

**Changes Needed:**

1. **ticker_reference table:** Add 8 columns
   ```sql
   ALTER TABLE ticker_reference ADD COLUMN IF NOT EXISTS
       upstream_1_name VARCHAR(255),
       upstream_1_ticker VARCHAR(20),
       upstream_2_name VARCHAR(255),
       upstream_2_ticker VARCHAR(20),
       downstream_1_name VARCHAR(255),
       downstream_1_ticker VARCHAR(20),
       downstream_2_name VARCHAR(255),
       downstream_2_ticker VARCHAR(20);
   ```

2. **ticker_articles table:** Add 1 column
   ```sql
   ALTER TABLE ticker_articles ADD COLUMN IF NOT EXISTS
       value_chain_type VARCHAR(10) CHECK (value_chain_type IN ('upstream', 'downstream'));
   ```

3. **feeds table:** Add 1 column (optional but recommended)
   ```sql
   ALTER TABLE feeds ADD COLUMN IF NOT EXISTS
       value_chain_type VARCHAR(10) CHECK (value_chain_type IN ('upstream', 'downstream'));
   ```

4. **Update SQL statements:**
   - `update_ticker_reference_ai_data()` (line 14548-14589) - Add 8 value chain fields to INSERT/UPDATE
   - `get_ticker_reference()` (line ~14462) - Add 8 fields to SELECT
   - Any other functions that query ticker_reference

**Testing:**
- Run `/admin/init` for test ticker to verify schema changes work
- Check that all 14 fields populate correctly

---

#### Task 5: Update Feed Creation Logic
**File:** `app.py` - `create_feeds_for_ticker_new_architecture()` (line 2170)

**Current Logic:**
- Creates 2 company feeds (Google + Yahoo)
- Creates 3 industry feeds (Google only, 3 keywords)
- Creates 6 competitor feeds (3 competitors × 2 sources)
- Total: 11 feeds

**New Logic:**
- Keep existing 11 feeds
- Add 4 upstream feeds (2 suppliers × 2 sources)
- Add 4 downstream feeds (2 customers × 2 sources)
- Total: 19 feeds

**Implementation:**
```python
# After competitor feeds section (~line 2294):

# 4. Value Chain Upstream feeds (0-2 suppliers)
value_chain = metadata.get("value_chain", {})
upstream_companies = value_chain.get("upstream", [])[:2]

for upstream_comp in upstream_companies:
    if isinstance(upstream_comp, dict) and upstream_comp.get('name'):
        comp_name = upstream_comp['name']
        comp_ticker = upstream_comp.get('ticker')

        # Google News feed
        feed_id = upsert_feed_new_architecture(
            url=f"https://news.google.com/rss/search?q={comp_name}",
            name=f"Upstream: {comp_name}",
            competitor_ticker=comp_ticker,
            value_chain_type='upstream'  # NEW FIELD
        )
        associate_ticker_with_feed_new_architecture(ticker, feed_id, "value_chain")

        # Yahoo Finance feed (if ticker exists)
        if comp_ticker:
            feed_id = upsert_feed_new_architecture(
                url=f"https://finance.yahoo.com/rss/headline?s={comp_ticker}",
                name=f"Yahoo Finance: {comp_ticker}",
                competitor_ticker=comp_ticker,
                value_chain_type='upstream'  # NEW FIELD
            )
            associate_ticker_with_feed_new_architecture(ticker, feed_id, "value_chain")

# 5. Value Chain Downstream feeds (0-2 customers)
downstream_companies = value_chain.get("downstream", [])[:2]

for downstream_comp in downstream_companies:
    # Same logic as upstream but with value_chain_type='downstream'
```

**Update Locations:**
- `upsert_feed_new_architecture()` (line 2111) - Add `value_chain_type` parameter
- Feed naming: "Upstream: TSMC" vs "Downstream: Microsoft"

---

#### Task 6: Update CSV Import/Export
**Files:**
- `import_ticker_reference_from_csv_content()` (line 3340)
- `export_ticker_references_to_csv()` (line 3794)

**CSV Columns to Add:**
1. `upstream_1_name`
2. `upstream_1_ticker`
3. `upstream_2_name`
4. `upstream_2_ticker`
5. `downstream_1_name`
6. `downstream_1_ticker`
7. `downstream_2_name`
8. `downstream_2_ticker`

**Import Changes:**
- Parse 8 new columns from CSV
- Handle None values gracefully
- Legacy support: Continue supporting old `competitors` column

**Export Changes:**
- Add 8 columns to CSV output
- Column order: industry_keyword_1-3, competitor_1-3 (6 fields), upstream_1-2 (4 fields), downstream_1-2 (4 fields)

---

### PHASE 2: Processing Pipeline (6 tasks)

#### Task 7: Add value_chain Category to Ingestion Logic
**Files:** `app.py` - ingestion phase functions

**Changes:**
1. Add `value_chain_ingested_by_keyword` to stats tracking (line ~901)
2. Update limits: 25 per value chain company (same as competitor)
3. Add logging: "INGESTION: Value Chain 'TSMC' 12/25"
4. Update category handling throughout ingestion

**Key Functions:**
- `get_ticker_ingestion_stats()` (line 901)
- `_update_ingestion_stats()` (line 1017)
- `_check_ingestion_limit()` (line 1040)

---

#### Task 8: Create `triage_value_chain_articles_claude()` Function
**File:** `app.py` - Add new function after existing triage functions (~line 9000)

**Prompt Focus:**
```python
"""You are a financial analyst triaging news articles about companies in the value chain
(suppliers/customers) of {ticker} ({company_name}).

UPSTREAM (Suppliers): Focus on signals affecting {ticker}'s costs, supply security, technology access:
- Supply disruptions, capacity constraints, delivery delays
- Raw material/component price changes
- Technology shifts, new product launches
- Production issues, quality problems
- M&A activity, strategic partnerships
- Regulatory changes affecting supply

DOWNSTREAM (Customers): Focus on signals affecting {ticker}'s revenue, demand visibility:
- Order trends, inventory levels, demand forecasts
- Market share shifts, competitive pressures
- Capital spending plans, expansion/contraction
- End-market demand signals
- Customer financial health, credit risk
- Strategic changes, product mix shifts

SCORING CRITERIA:
- Direct materiality: Does this affect {ticker}'s financials in next 1-2 quarters?
- Quantifiable impact: Are specific numbers mentioned (prices, volumes, orders)?
- Strategic significance: Does this change competitive dynamics?

Rate 0-10 (8-10 = flagged, <8 = not flagged)
"""
```

**Implementation:**
- Apply prompt caching (`cache_control: {"type": "ephemeral"}`)
- Batch processing (2-3 articles per call)
- Pass `value_chain_type` context ('upstream' or 'downstream')
- Same scoring thresholds as competitor (8-10 flagged)

---

#### Task 9: Update Triage Routing
**File:** `app.py` - Triage dispatcher function

**Changes:**
- Add `value_chain` category to routing logic
- Route to `triage_value_chain_articles_claude()` for value_chain articles
- Keep `value_chain_type` metadata through triage

**Location:** Search for triage routing/dispatcher (~line 9546)

---

#### Task 10: Add value_chain to Scraping Routing
**File:** `app.py` - Scraping phase

**Changes:**
- Add `value_chain` category to scraping dispatcher
- Same 2-tier fallback (newspaper3k → Scrapfly)
- Same limits as competitor (8 per keyword)
- Update scraping stats tracking

**Functions:**
- `safe_content_scraper()` (line 4951)
- `_check_scraping_limit()` (line 5503)

---

#### Task 11: Create `generate_claude_value_chain_article_summary()` Function
**File:** `app.py` - Add after existing summary functions

**Prompt Focus:**
```python
"""Analyze this article about {value_chain_company} ({value_chain_type}) in relation to {ticker}.

UPSTREAM ANALYSIS (if supplier):
- Cost impact: How does this affect {ticker}'s input costs or margins?
- Supply security: Does this create supply risk or opportunity?
- Technology: Any technology shifts that affect {ticker}'s products?
- Strategic: M&A, partnerships, or competitive moves affecting {ticker}?

DOWNSTREAM ANALYSIS (if customer):
- Revenue impact: How does this affect {ticker}'s sales or pricing power?
- Demand signal: What does this say about end-market demand for {ticker}'s products?
- Market share: Any competitive dynamics affecting {ticker}'s position?
- Strategic: Customer changes that affect {ticker}'s relationship or opportunity?

Provide 2-3 sentence analysis focusing on materiality to {ticker}.
"""
```

**Implementation:**
- Apply prompt caching
- Pass `value_chain_type` context
- 2-3 sentence output (concise)

---

#### Task 12: Update Summary Routing
**File:** `app.py` - Summary dispatcher

**Changes:**
- Add `value_chain` category routing
- Route to `generate_claude_value_chain_article_summary()`
- Keep `value_chain_type` metadata

---

### PHASE 3: Email Display (3 tasks)

#### Task 13: Update Email #1 (Article Selection QA)
**File:** `app.py` - `send_enhanced_quick_intelligence_email()` (line 10353)

**Changes:**
- Add 4th section: "Value Chain" after Company/Industry/Competitor
- Group articles by upstream/downstream with company names as subheaders
- Show dual AI scores (relevance + category)
- Same priority sorting (flagged+quality → flagged → rest)

**Format:**
```
📦 Value Chain (12 selected from 47 total)

Upstream
• [TSMC] Supply chain constraints (⚡ 9.2/10 | 🎯 8.5/10)
• [ASML] Lithography breakthrough (⚡ 8.5/10 | 🎯 7.8/10)

Downstream
• [MSFT] Azure demand surge (⚡ 8.8/10 | 🎯 8.2/10)
```

---

#### Task 14: Update Email #2 (Content QA)
**File:** `app.py` - Email #2 template/generation

**Changes:**
- Add "Value Chain" section with full scraped content
- Group by upstream/downstream
- Show AI analysis boxes
- Same article display format

---

#### Task 15: Update Email #3 (Premium Intelligence)
**File:** `app.py` - `generate_email_html_core()` (line 18601)

**Changes:**
- Add "📦 Value Chain" section (or "⛓️ Value Chain")
- Compressed article links grouped by upstream/downstream
- Same visual indicators (★, 🆕, PAYWALL)

**Format:**
```html
<h2>📦 Value Chain</h2>

<h3>Upstream</h3>
<ul>
  <li><strong>[TSMC]</strong> Article about supply constraints ★</li>
  <li><strong>[ASML]</strong> New lithography technology</li>
</ul>

<h3>Downstream</h3>
<ul>
  <li><strong>[MSFT]</strong> Azure demand surge 🆕</li>
  <li><strong>[AAPL]</strong> iPhone orders reduced 15%</li>
</ul>
```

---

## FUTURE: Phase 4 - Executive Summary (Not in Scope)

Will eventually split section 5 into:
- "Competitive Landscape" (horizontal competitors)
- "Value Chain" (upstream/downstream + industry keywords)

Requires updates to:
- Phase 1 prompt (`modules/_build_executive_summary_prompt_phase1`)
- Phase 2 prompt (`modules/_build_executive_summary_prompt_phase2`)
- Both converters in `modules/executive_summary_phase1.py`
- Email rendering functions

---

## Testing Strategy

**Test Tickers:**
1. **Tesla (TSLA)** - Upstream-heavy
   - Horizontal: BYD, Ford, GM
   - Upstream: Panasonic, CATL
   - Downstream: None (B2C)

2. **Intel (INTC)** - Both upstream + downstream
   - Horizontal: AMD, NVDA, QCOM
   - Upstream: TSMC
   - Downstream: MSFT, AAPL

3. **Netflix (NFLX)** - No value chain
   - Horizontal: DIS, WBD, PARA
   - Upstream: None
   - Downstream: None

4. **Denison Mines (DNN)** - Downstream-heavy
   - Horizontal: CCJ, NXE, UUUU
   - Upstream: None
   - Downstream: LEU, CEG

**Test Workflow:**
1. Database wipe (fresh start)
2. `/admin/init` with test tickers
3. Verify metadata generation (check 14 fields populated)
4. Run ingestion for 1 ticker
5. Verify 19 feeds created (2+3+6+8)
6. Check all 3 emails display Value Chain section

---

## Cost & Performance Impact

**RSS Parsing:**
- Current: ~10 seconds/ticker (11 feeds)
- New: ~17 seconds/ticker (19 feeds)
- Impact: +70% parsing time, +7 seconds/ticker

**AI API Costs per Ticker:**
- Triage: +4 calls (value chain triage)
- Summaries: +4 calls (value chain summaries)
- Cost increase: ~$0.30-0.40/ticker/day

**Database:**
- Up to +100 articles/ticker (4 companies × 25 each)
- Total: ~300 articles/ticker (up from ~200)

**Email Length:**
- Email #1: +25-50% longer
- Email #2: +25-50% longer
- Email #3: +20-30% longer

---

## Implementation Notes

### Key Design Decisions

1. **Category Naming:**
   - Keep `competitor` internally (no rename)
   - Only change metadata field names and display labels

2. **Value Chain Type Tracking:**
   - Add `value_chain_type` field to ticker_articles and feeds
   - Enables grouping: "Upstream: TSMC" vs "Downstream: Microsoft"

3. **Article Limits:**
   - 25 per company (same as competitor)
   - 100 total possible with 4 value chain companies
   - Prevents email overload

4. **Triage Prompts:**
   - Value chain-specific prompts (not copied from industry)
   - Focus on supply/demand signals vs competitive positioning

### Backward Compatibility

- Old metadata format (`competitors` field) still supported
- `parse_metadata_to_flat_fields()` handles fallback
- Existing tickers won't break during migration

### Fresh DB Wipe Required

- User confirmed fresh database wipe once implementation complete
- Eliminates backward compatibility complexity
- Clean slate with new schema

---

## Next Steps

**Immediate Priority (Task 4):**
1. Update `ensure_schema()` function - Add 8 columns to ticker_reference
2. Add `value_chain_type` column to ticker_articles and feeds
3. Update all SQL SELECT/INSERT/UPDATE statements referencing ticker_reference
4. Test schema changes work correctly

**Then Continue With:**
- Task 5: Feed creation (19 feeds)
- Task 6: CSV import/export (14 fields)
- Task 7: Ingestion logic (value_chain category)

**Estimated Time Remaining:** 4-5 days

---

## Git Commits

- **526ac74** - Phase 1.1: Update Metadata Generation Prompts
- **13cfaa6** - Phase 1.2: Add JSON Parsing for Value Chain Structure

---

## Questions for User

1. Ready to continue with Task 4 (database schema)?
2. Any changes to the design before proceeding?
3. Want to test metadata generation first before continuing?
