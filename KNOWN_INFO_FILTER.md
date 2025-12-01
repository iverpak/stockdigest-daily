# Known Information Filter (Phase 1.5)

**Created:** December 1, 2025
**Status:** TEST MODE - Runs in parallel, emails findings only, does NOT modify production pipeline

## Overview

The Known Information Filter is a new phase inserted between Phase 1 (article synthesis) and Phase 2 (filing context enrichment). It identifies claims in article-derived bullets that are already known from SEC filings, enabling "delta reporting" - only reporting what's genuinely new.

### The Problem

Articles often recap old earnings data. For example, an article from Nov 27 might recap META's Oct 29 earnings results. This pollutes reports with stale information that "erodes credibility" - users see the same numbers they already know from earnings calls.

### The Solution

Phase 1.5 decomposes each bullet into atomic claims and checks each against the company's SEC filings (10-K, 10-Q, Transcript, 8-K). Claims found in filings are marked as KNOWN; claims not found are marked as NEW. Bullets are then:
- **KEEP** - All claims are NEW (pass through unchanged)
- **REWRITE** - Mix of KNOWN and NEW claims (rewrite to contain only NEW)
- **REMOVE** - All claims are KNOWN (nothing new to report)

## Architecture

```
Phase 1 (Article Synthesis)
    ↓
Phase 1.5 (Known Info Filter) ← NEW
    ↓
Phase 2 (Filing Context Enrichment)
    ↓
Phase 3 (Context Integration)
```

## Filing Sources

Phase 1.5 checks claims against these SEC filings:

| Source | Description | Storage Table |
|--------|-------------|---------------|
| **Transcript** | Latest earnings call | `transcript_summaries` |
| **10-Q** | Latest quarterly report | `sec_filings` |
| **10-K** | Latest annual report | `sec_filings` |
| **8-K** | Material events since last earnings | `company_releases` |

## 8-K Integration

### Time Window

8-K filings are included based on the last earnings call date:

- **Start:** After `transcript.report_date` (last earnings call)
- **End:** T-3 (3 days before today) - buffer for articles to cover the 8-K first
- **Max:** 90-day lookback (safety cap)
- **Fallback:** 90-day window if no transcript exists

### 3-Layer Filtering

8-K filings go through aggressive filtering to exclude legal boilerplate:

#### Layer 1: Item Code Filter

Only include 8-Ks with these material event codes:

| Code | Description |
|------|-------------|
| 1.01 | Entry into Material Definitive Agreement |
| 1.02 | Termination of Material Agreement |
| 2.01 | Completion of Acquisition/Disposition |
| 2.02 | Results of Operations (Earnings) |
| 2.03 | Creation of Direct Financial Obligation |
| 2.05 | Exit/Restructuring Costs |
| 2.06 | Material Impairments |
| 4.01 | Accountant Changes |
| 4.02 | Non-Reliance on Prior Financials |
| 5.01 | Changes in Control |
| 5.02 | Director/Officer Changes |
| 5.07 | Shareholder Vote Results |
| 7.01 | Regulation FD Disclosure |
| 8.01 | Other Material Events |
| Unknown | Apply Layer 2+3 filtering |

#### Layer 2: Exhibit Number Filter

Only include these exhibit types:

| Exhibit | Description |
|---------|-------------|
| `99.x` | Press releases (99.1, 99.2, etc.) |
| `MAIN` | 8-K body itself |
| `2.1` | Merger agreements |

**Excluded:** `1.1` (underwriting), `4.x` (indentures), `5.x` (legal opinions), `10.x` (contracts)

#### Layer 3: Title Keyword Filter

Exclude if title contains:
- "Legal Opinion"
- "Underwriting Agreement"
- "Indenture"
- "Officers' Certificate"
- "Notes Due"
- "Bylaws"
- "Dividend"

### Cap: 3 Exhibits Per Filing Date

To prevent token explosion from merger filings with 15+ exhibits (e.g., HBAN), we cap at 3 exhibits per filing date.

**Priority order:**
1. `MAIN` (8-K body)
2. `2.1` (merger agreements)
3. `99.x` by exhibit number ascending (99.1, 99.2, 99.3)

This uses a PostgreSQL window function:
```sql
ROW_NUMBER() OVER (
    PARTITION BY filing_date
    ORDER BY
        CASE
            WHEN exhibit_number = 'MAIN' THEN 0
            WHEN exhibit_number = '2.1' THEN 1
            ELSE 2
        END,
        exhibit_number ASC
) as rn
...
WHERE rn <= 3
```

## Section Handling

### Filtered Sections (claim analysis applied)

**Bullet sections:**
- `major_developments`
- `financial_performance`
- `risk_factors`
- `competitive_industry_dynamics`
- `key_variables`

**Paragraph sections:**
- `bottom_line`
- `upside_scenario`
- `downside_scenario`

### Exempt Sections (pass through unchanged)

- `wall_street_sentiment` - Analyst opinions ARE the news, even when citing known data
- `upcoming_catalysts` - Forward-looking editorial value

## Claim Classification

### KNOWN (filter out)

Information already in the filings OR stale information:
- Specific numbers (revenue, margins, EPS, guidance, capex)
- Events explicitly stated in filings
- Management quotes from transcripts
- Guidance figures from earnings calls
- Risk factors already disclosed in 10-K/10-Q
- Business model descriptions
- Historical comparisons already discussed
- Material events disclosed in 8-K filings (mergers, acquisitions, executive changes)
- **Prior quarter data:** Financial metrics from quarters before the current filing period (e.g., Q2 data when current filings are Q3)

### NEW (keep)

Information NOT in the filings AND temporally fresh:
- Market reaction (stock price movement, trading volume)
- Analyst actions (upgrades, downgrades, price targets)
- Third-party commentary (analyst quotes, expert opinions)
- Events occurring AFTER the latest filing date
- Rumors, speculation, breaking news
- Competitive developments not in company filings
- External market data
- **Specific competitor metrics** (growth rates, market share) NOT in company filings

### Paired Claims Rule

Comparisons must be treated as a unit:
- "AWS at 20% vs Azure at 30%" → if one side is NEW, keep BOTH
- Removing one side of a comparison makes it meaningless
- Either keep the full comparison or remove it entirely

### Materiality Test

Before marking as REWRITE (instead of REMOVE), apply this test:

> "Would a reader gain ACTIONABLE INSIGHT from ONLY the NEW claims?"

Mark as REMOVE if NEW claims are merely:
- Dates or timing details on otherwise KNOWN events
- Minor wording variations of KNOWN information
- Context that only supports KNOWN claims
- Less than 20% of the original content's substance

### What Counts as KNOWN

The **specific fact** must be in filings, not just the general topic:
- ❌ "Competition exists" does NOT make "Temu has 57% market share" KNOWN
- ❌ "We face regulatory risk" does NOT make "EU investigation in November 2025" KNOWN
- ✅ Only mark KNOWN if the specific data point, number, or fact appears in filings

## AI Implementation

### Primary: Gemini 2.5 Flash

- Uses concatenated prompt pattern (not `system_instruction`)
- Temperature: 0.0
- Max output tokens: 40,000

**Note:** Using `system_instruction` parameter caused empty responses with `finish_reason=1`. The fix was to concatenate system prompt + user content into a single `full_prompt`, matching the working pattern in `article_summaries.py` and `triage.py`.

### Fallback: Claude Sonnet 4.5

Re-enabled as fallback when Gemini fails. Uses same prompt with 8-K filings included.

## Output Format

```json
{
  "summary": {
    "total_bullets": 15,
    "total_paragraphs": 3,
    "kept": 3,
    "rewritten": 8,
    "removed": 4,
    "total_claims": 45,
    "known_claims": 28,
    "new_claims": 17
  },
  "bullets": [
    {
      "bullet_id": "FIN_001",
      "section": "financial_performance",
      "original_content": "...",
      "claims": [
        {
          "claim": "Q3 revenue $51.2B",
          "status": "KNOWN",
          "source": "10-Q Financial Highlights section",
          "source_type": "10-Q",
          "evidence": "Total net sales increased 11% to $158.9 billion in Q3 2024"
        },
        {
          "claim": "stock pulled back 25%",
          "status": "NEW",
          "source": null,
          "source_type": null,
          "evidence": null
        }
      ],
      "action": "REWRITE",
      "rewritten_content": "..."
    }
  ],
  "paragraphs": [...]
}
```

### Evidence Field

For KNOWN claims, the `evidence` field contains the actual quote or paraphrase from the filing that proves the claim is known. This enables:
- **Verification:** Confirm the AI correctly matched the claim to the filing
- **QA:** Catch over-aggressive KNOWN marking (e.g., vague match shouldn't make specific claim KNOWN)
- **Transparency:** See exactly what text in the filing supports the classification

**Email Display:**
```
❌ KNOWN: AWS growth at ~20%
   → "AWS segment revenue grew 19% year-over-year" (Transcript, FINANCIAL RESULTS)
```

## Integration Points

### app.py

Phase 1.5 is called in two places:

1. **`generate_ai_final_summaries()`** (lines 12720-12753) - Production/test runs
2. **`/api/regenerate-email`** endpoint (lines 29389-29422) - Regenerate runs

Both use try/except with non-blocking behavior:
```python
try:
    from modules.known_info_filter import filter_known_information, generate_known_info_filter_email

    filter_result = filter_known_information(
        ticker=ticker,
        phase1_json=json_output,
        db_func=db,
        gemini_api_key=GEMINI_API_KEY,
        anthropic_api_key=ANTHROPIC_API_KEY
    )

    if filter_result:
        filter_email_html = generate_known_info_filter_email(ticker, filter_result)
        send_email(subject=email_subject, html_body=filter_email_html, to=ADMIN_EMAIL)
except Exception as e:
    LOG.warning(f"[{ticker}] Phase 1.5: Test failed (non-blocking): {e}")
```

## Email Report

The filter generates an HTML email showing:

- **Header:** Ticker, timestamp, model used, generation time, filings used
- **Summary Stats:** Total items, kept/rewritten/removed counts, known/new claims
- **Paragraph Sections:** Original content, claims with status, action, rewritten content
- **Bullet Sections:** Same format as paragraphs

## Files

| File | Description |
|------|-------------|
| `modules/known_info_filter.py` | Main module (~900 lines) |
| `KNOWN_INFO_FILTER.md` | This documentation |

## Key Functions

| Function | Description |
|----------|-------------|
| `filter_known_information()` | Main entry point |
| `_fetch_filtered_8k_filings()` | Fetch 8-Ks with 3-layer filtering |
| `_build_filter_user_content()` | Build prompt with filings + 8-Ks |
| `_get_filings_info()` | Extract metadata for email |
| `_filter_known_info_gemini()` | Gemini 2.5 Flash implementation |
| `_filter_known_info_claude()` | Claude fallback (commented out) |
| `generate_known_info_filter_email()` | Generate HTML email report |

## Test Query

Run this in PostgreSQL to preview 8-K filtering:

```sql
WITH filtered_8k AS (
    SELECT
        ticker,
        filing_date,
        report_title,
        item_codes,
        exhibit_number,
        LENGTH(summary_markdown) as chars,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, filing_date
            ORDER BY
                CASE
                    WHEN exhibit_number = 'MAIN' THEN 0
                    WHEN exhibit_number = '2.1' THEN 1
                    ELSE 2
                END,
                exhibit_number ASC
        ) as rn
    FROM company_releases
    WHERE source_type = '8k_exhibit'
      AND (
          item_codes LIKE '%1.01%' OR item_codes LIKE '%2.02%' OR
          item_codes LIKE '%5.07%' OR item_codes LIKE '%7.01%' OR
          item_codes LIKE '%8.01%' OR item_codes = 'Unknown'
      )
      AND (exhibit_number LIKE '99%' OR exhibit_number = 'MAIN' OR exhibit_number = '2.1')
      AND report_title NOT ILIKE '%Legal Opinion%'
      AND report_title NOT ILIKE '%Indenture%'
      AND report_title NOT ILIKE '%Dividend%'
)
SELECT ticker, filing_date, report_title, item_codes, exhibit_number, chars
FROM filtered_8k
WHERE rn <= 3
ORDER BY ticker, filing_date DESC, rn;
```

## Future Enhancements

1. **Phase 2 Integration:** Add 8-K data to Phase 2's filing context (same filtering logic)
2. **Production Mode:** Apply filtered output to Phase 2 instead of original Phase 1 JSON
3. **Claude Fallback:** Re-enable after Gemini stability confirmed
4. **Metrics Dashboard:** Track KNOWN vs NEW claim ratios over time

## Commits

| Hash | Description |
|------|-------------|
| `ab1f011` | Fix Gemini MAX_TOKENS (40K output tokens) |
| `ef3ab1a` | Prevent AI from truncating claims lists |
| `a76d5a8` | Fix Gemini Flash empty response (concatenated prompt pattern) |
| `9b03d9c` | Add 8-K filing integration with 3-layer filtering |
| `8cb44dd` | Add 8-K to KNOWN section in prompt |
| `fdcd133` | Cap 8-K exhibits to 3 per filing date |
