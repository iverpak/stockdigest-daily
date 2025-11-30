# Workflow Refactor Implementation Plan

**Date:** November 30, 2025
**Status:** ✅ IMPLEMENTED
**Goal:** Clean up spaghetti architecture, fix broken Phase 1+2 generation, simplify workflow

---

## Problem Summary

The current code has Phase 1+2 AI generation **buried inside** `build_enhanced_digest_html()`, which is an email building function. A recent "optimization" (commit `766d9cc`) moved an early return to skip this function when `send_email_enabled=False`, which accidentally broke Phase 1+2 generation entirely.

**Root cause:** Poor separation of concerns - AI generation and email building are tangled together.

---

## Current Architecture (Broken)

```
INGEST PHASE
├─ RSS feeds → articles
├─ AI triage
├─ Send Email #1 ✓
└─ Return flagged_article_ids

DIGEST PHASE (process_digest_phase)
├─ Scrape flagged articles
├─ Call fetch_digest_articles_with_enhanced_content(send_email_enabled=False)
│   ├─ EARLY RETURN HERE ← Skips everything below!
│   ├─ [NEVER REACHED]: build_enhanced_digest_html()
│   │   ├─ [NEVER REACHED]: generate_ai_final_summaries()
│   │   │   ├─ Phase 1 AI generation
│   │   │   └─ Phase 2 AI generation
│   │   └─ Save to database
│   └─ Build Email #2 HTML
└─ Return digest_result (missing Phase 1+2!)

MAIN JOB (after digest)
├─ Try to load Phase 1+2 from database ← FAILS! Nothing saved!
├─ Phase 3 fails: "No Phase 2 JSON found"
├─ No Email #2 sent
└─ No Email #3 generated
```

---

## Target Architecture (Clean)

```
INGEST PHASE (no changes)
├─ RSS feeds → articles
├─ AI triage
├─ Send Email #1 ✓
└─ Return flagged_article_ids

URL RESOLUTION PHASE (no changes)
└─ Resolve Google News URLs for flagged articles

SCRAPE PHASE (simplified)
└─ Scrape content for flagged articles (no AI, no emails)

EXECUTIVE SUMMARY PHASE (NEW - separated from emails)
├─ Phase 1 AI: Articles → JSON
├─ Phase 2 AI: JSON + Filings → Enriched JSON
├─ Save Phase 1+2 to database
├─ Phase 3 AI: Enriched JSON → Integrated JSON
└─ Save Phase 3 to database

EMAIL PHASE (clean, separate)
├─ Build Email #2 HTML from Phase 3 JSON + articles
├─ Send Email #2 to admin ✓
├─ Build Email #3 HTML from Phase 3 JSON
└─ Queue Email #3 for users (daily mode) OR send to admin (test mode) ✓
```

---

## Key Changes

### 1. Remove `send_email_enabled` parameter

**Files:** `app.py`
**Lines:** ~13771, ~13989, ~16077

This parameter was a hack to defer Email #2. We'll remove it entirely and just call functions in the right order.

### 2. Extract AI generation from `build_enhanced_digest_html()`

**Current location:** Inside `build_enhanced_digest_html()` at line ~13373
**New location:** Direct call in `process_ticker_job()` after scraping

The function `generate_ai_final_summaries()` (line 12561) does Phase 1+2 generation. We'll call it directly instead of through the email builder.

### 3. Simplify `fetch_digest_articles_with_enhanced_content()`

**Current:** Does too many things (fetch articles, generate AI, build HTML, optionally send)
**New:** Only fetches articles and returns them. Renamed to `fetch_digest_articles()`.

### 4. Simplify `build_enhanced_digest_html()`

**Current:** Generates AI summaries if not provided, builds HTML
**New:** Only builds HTML from provided Phase 3 JSON (no AI generation inside)

### 5. Consolidate daily/test mode logic

**Current:** ~200 lines duplicated for daily vs test mode (lines 18371-18586 vs 18588-18722)
**New:** Single code path with one `if` statement for Email #3 destination

### 6. Rename `process_digest_phase()` to `process_scrape_phase()`

Better reflects what it actually does after refactor (just scraping, no AI).

---

## Detailed Implementation Steps

### Step 1: Create new `generate_executive_summary_all_phases()` function

Location: After `generate_ai_final_summaries()` (~line 12789)

```python
async def generate_executive_summary_all_phases(
    ticker: str,
    articles_by_ticker: Dict[str, Dict[str, List[Dict]]],
    config: Dict,
    report_type: str = 'daily'
) -> Dict:
    """
    Generate complete executive summary (Phase 1 + Phase 2 + Phase 3).

    Returns:
        Dict with keys:
        - phase3_json: Final merged JSON (Phase 1+2+3)
        - articles_by_ticker: Original articles (for Email #2)
        - success: bool
        - error: str (if failed)
    """
```

This function will:
1. Call `generate_ai_final_summaries()` for Phase 1+2
2. Save to database
3. Call `generate_executive_summary_phase3()` for Phase 3
4. Update database with Phase 3
5. Return complete result

### Step 2: Simplify `fetch_digest_articles_with_enhanced_content()`

Remove:
- `send_email_enabled` parameter
- All AI generation code (move to Step 1)
- All email sending code

Keep:
- Article fetching logic
- Category grouping
- Company releases fetching

Rename to: `fetch_digest_articles()`

### Step 3: Simplify `build_enhanced_digest_html()`

Remove:
- `generate_ai_final_summaries()` call
- `existing_summaries` parameter (no longer needed)
- The conditional logic at lines 13362-13373

Keep:
- HTML building from `phase3_json`
- Article formatting
- All styling

Require:
- `phase3_json` parameter (no longer optional)

### Step 4: Update `process_digest_phase()` → `process_scrape_phase()`

Remove:
- Call to `fetch_digest_articles_with_enhanced_content()`
- `send_email_enabled=False` logic

Keep:
- Article scraping logic (Phase 4 content scraping)

Return:
- Just scraping stats (no articles_by_ticker needed here)

### Step 5: Refactor `process_ticker_job()` main flow

New flow after scraping:

```python
# After scraping completes...

# PHASE: Fetch articles for AI processing
articles_by_ticker = await fetch_digest_articles(
    hours=minutes/60,
    tickers=[ticker],
    flagged_article_ids=flagged_article_ids
)

# PHASE: Generate executive summary (all 3 phases)
summary_result = await generate_executive_summary_all_phases(
    ticker=ticker,
    articles_by_ticker=articles_by_ticker,
    config=config,
    report_type=report_type
)

if not summary_result['success']:
    LOG.error(f"[{ticker}] Executive summary generation failed: {summary_result['error']}")
    # Handle error...

phase3_json = summary_result['phase3_json']

# PHASE: Send Email #2 (Content QA) to admin
email2_html = await build_enhanced_digest_html(
    articles_by_ticker=articles_by_ticker,
    period_days=days,
    phase3_json=phase3_json
)
email2_subject = f"QA Content Review {'(WEEKLY)' if report_type == 'weekly' else '(DAILY)'}: {company_name} ({ticker}) - {total_articles} articles"

if mode != 'test':
    # Save Email #2 to database for admin dashboard
    save_email2_to_queue(ticker, email2_html)

send_email(email2_subject, email2_html)
LOG.info(f"[{ticker}] ✅ Email #2 sent")

# PHASE: Generate and queue/send Email #3
email3_data = generate_email_html_core(
    ticker=ticker,
    hours=int(minutes/60),
    recipient_email=None,  # Placeholder for queue
    report_type=report_type
)

if mode == 'daily':
    # Queue for users (8:30 AM send)
    save_email3_to_queue(ticker, email3_data, recipients)
    LOG.info(f"[{ticker}] ✅ Email #3 queued for {len(recipients)} recipients")
else:
    # Test mode: send directly to admin
    send_email(email3_data['subject'] + " (TEST)", email3_data['html'])
    LOG.info(f"[{ticker}] ✅ Email #3 sent to admin (test mode)")
```

### Step 6: Delete duplicate daily/test mode blocks

Remove the duplicate ~200 line blocks:
- Lines 18371-18586 (daily mode Phase 3)
- Lines 18588-18722 (test mode Phase 3)

Replace with unified flow from Step 5.

### Step 7: Update regenerate endpoint

The `/api/regenerate-email` endpoint (line 29028) has its own Phase 1+2+3 generation.

Options:
A. Leave as-is (it works independently)
B. Refactor to use new `generate_executive_summary_all_phases()`

**Recommendation:** Option B for consistency, but can be Phase 2 of refactor.

For now, ensure it still works after our changes.

---

## Files Modified

| File | Changes |
|------|---------|
| `app.py` | Main refactor (~500 lines changed) |

## Functions Modified

| Function | Change |
|----------|--------|
| `fetch_digest_articles_with_enhanced_content()` | Simplify → rename to `fetch_digest_articles()` |
| `build_enhanced_digest_html()` | Remove AI generation, require `phase3_json` |
| `process_digest_phase()` | Simplify → rename to `process_scrape_phase()` |
| `process_ticker_job()` | New clean flow |
| `generate_ai_final_summaries()` | No changes (called directly now) |

## Functions Added

| Function | Purpose |
|----------|---------|
| `generate_executive_summary_all_phases()` | Orchestrate Phase 1+2+3 generation |

## Functions Removed

| Function | Reason |
|----------|--------|
| None | We're simplifying, not removing |

---

## Test/Daily Mode Differences (Preserved)

| Aspect | Daily Mode | Test Mode |
|--------|------------|-----------|
| Email #1 | Sent to admin | Sent to admin |
| Email #1 saved to DB | Yes | No |
| Email #2 | Sent to admin | Sent to admin |
| Email #2 saved to DB | Yes | No |
| Email #3 | Queued for users (8:30 AM) | Sent immediately to admin |
| Email #3 saved to DB | Yes (email_queue) | No |
| Executive summary saved | Yes | Yes |

---

## Report Type Logic (Preserved)

| Aspect | Daily Report (Tue-Sun) | Weekly Report (Mon) |
|--------|------------------------|---------------------|
| Lookback window | 1440 min (1 day) | 10080 min (7 days) |
| Email #3 sections | 2 sections shown | 6 sections shown |
| Subject label | "(DAILY)" | "(WEEKLY)" |

---

## Rollback Plan

If issues arise:
1. `git revert` the refactor commit
2. Original code preserved in git history

---

## Success Criteria

1. **Email #1** sent during ingest phase ✓
2. **Phase 1+2+3** executive summary generated and saved to database
3. **Email #2** sent to admin after Phase 3 with deduplication info
4. **Email #3** queued for users (daily) or sent to admin (test)
5. **Regenerate endpoint** still works
6. **Report type** logic preserved (daily vs weekly)
7. **Test/daily mode** differences preserved
8. No duplicate code blocks
9. Clear, linear flow in logs

---

## Estimated Scope

- Lines changed: ~500-600
- Lines removed (deduplication): ~200
- Net change: ~300-400 lines modified
- Risk: Medium (core workflow change)
- Testing: Run test mode for 3 tickers, verify all 3 emails

---

## Next Steps

1. Review this plan
2. Get go signal
3. Implement Step 1-7 in sequence
4. Test with 3 tickers in test mode
5. Verify Email #1, #2, #3 all work
6. Push to production
