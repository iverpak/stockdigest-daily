# Phase 3 Editorial Format - Work in Progress

**Date Started:** 2025-01-09
**Status:** Ready to Code - Simple Implementation
**Goal:** Add editorial markdown format (Email #4) as incremental enhancement to existing Email #3

---

## Overview

Phase 3 transforms Phase 1+2 merged JSON into scannable, professional editorial format.

**Key Principles:**
- Phase 3 is a FORMATTER, not an ANALYZER
- No new analysis, just rearrange existing content for readability
- Copy Email #3 pattern exactly (no reinventing the wheel)
- Sequential execution: Email #3 → Phase 3 generation → Email #4
- No environment variables, no feature flags, no delays
- Just send it

---

## Implementation Strategy - SIMPLE

### **Pattern: Copy Email #3 Exactly**

**Email #3 Flow:**
```
generate_email_html_core()
  → Fetch summary_text from DB
  → Parse JSON to sections dict
  → Build HTML
  → Return dict

send_user_intelligence_report()
  → Call generate_email_html_core()
  → Send via send_email()
  → Return status
```

**Email #4 Flow (SAME):**
```
generate_email_html_core_editorial()
  → Fetch editorial_markdown from DB
  → Parse markdown to sections dict
  → Build HTML (REUSE SAME CODE)
  → Return dict

send_editorial_intelligence_report()
  → Call generate_email_html_core_editorial()
  → Send via send_email()
  → Return status
```

---

## Code Locations in app.py

### **Email #3 Functions (Reference):**
- Line 21831: `generate_email_html_core()` - Core Email #3 generation
- Line 22194: `send_user_intelligence_report()` - Test wrapper for Email #3

### **Where to Add Code:**
- Line ~24288: After Email #3 sent → Add Phase 3 generation + Email #4 send
- Line ~22240: After `send_user_intelligence_report()` → Add Email #4 functions

---

## Database Changes

### **Simple - Just 1 Column:**
```sql
ALTER TABLE executive_summaries ADD COLUMN IF NOT EXISTS
  editorial_markdown TEXT;
```

That's it. No metadata columns needed initially.

---

## File Structure

```
modules/
  _build_executive_summary_prompt_phase3    # NEW - Prompt text (already drafted)
  executive_summary_phase3.py               # NEW - 3 functions (see below)

app.py
  # Line ~22240: Add 2 new functions (copy Email #3 pattern)
  # Line ~24288: Add Phase 3 generation + Email #4 send
```

---

## modules/executive_summary_phase3.py - Complete Spec

### **Function 1: Generate Phase 3 Markdown**
```python
def generate_executive_summary_phase3(
    ticker: str,
    merged_json: Dict,           # Output from merge_phase1_phase2()
    anthropic_api_key: str
) -> Optional[Dict]:
    """
    Generate Phase 3 editorial markdown from merged Phase 1+2 JSON.

    Args:
        ticker: Stock ticker
        merged_json: Complete merged JSON from Phase 2
        anthropic_api_key: Claude API key

    Returns:
        {
            "markdown": "## BOTTOM LINE\n\n...",
            "model_used": "claude-sonnet-4-5-20250929",
            "prompt_tokens": 15000,
            "completion_tokens": 3500,
            "generation_time_ms": 35000
        }
        Or None if failed
    """

    # 1. Load Phase 3 prompt from file
    prompt_path = os.path.join(os.path.dirname(__file__), '_build_executive_summary_prompt_phase3')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    # 2. Build user content (merged JSON as formatted string)
    user_content = json.dumps(merged_json, indent=2)

    # 3. Call Claude API
    headers = {
        "x-api-key": anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    data = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 16000,
        "temperature": 0.0,
        "system": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": user_content
            }
        ]
    }

    # 4. Make request with retry logic (same as Phase 1/2)
    response = requests.post("https://api.anthropic.com/v1/messages",
                           headers=headers, json=data, timeout=180)

    # 5. Parse response
    if response.status_code == 200:
        result = response.json()
        markdown = result.get("content", [{}])[0].get("text", "")

        return {
            "markdown": markdown,
            "model_used": "claude-sonnet-4-5-20250929",
            "prompt_tokens": result.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": result.get("usage", {}).get("output_tokens", 0),
            "generation_time_ms": 0  # Calculate if needed
        }
    else:
        LOG.error(f"[{ticker}] Phase 3 API error {response.status_code}: {response.text[:500]}")
        return None
```

### **Function 2: Parse Markdown to Sections Dict**
```python
def parse_phase3_markdown_to_sections(markdown: str) -> Dict[str, List[str]]:
    """
    Parse Phase 3 markdown back to sections dict format.

    Output format matches parse_executive_summary_sections() from app.py

    Converts:
    ## BOTTOM LINE
    [Thesis]
    Key Developments:
    - **Theme 1**: [content]

    To:
    {
      "bottom_line": ["[Thesis]\n\nKey Developments:\n- **Theme 1**: [content]"],
      "major_developments": ["**Topic** • Bullish (reason)\n\n[integrated paragraph]"],
      ...
    }
    """
    import re

    sections = {
        "bottom_line": [],
        "major_developments": [],
        "financial_operational": [],
        "risk_factors": [],
        "wall_street": [],
        "competitive_industry": [],
        "upcoming_catalysts": [],
        "upside_scenario": [],
        "downside_scenario": [],
        "key_variables": []
    }

    # Split by section headers (## SECTION NAME)
    section_pattern = r'^## (.+)$'
    lines = markdown.split('\n')

    current_section = None
    current_content = []

    for line in lines:
        match = re.match(section_pattern, line)
        if match:
            # Save previous section
            if current_section and current_content:
                content = '\n'.join(current_content).strip()
                if current_section == "BOTTOM LINE":
                    sections["bottom_line"] = [content]
                elif current_section == "MAJOR DEVELOPMENTS":
                    sections["major_developments"] = parse_bullet_section(content)
                elif current_section == "FINANCIAL/OPERATIONAL PERFORMANCE":
                    sections["financial_operational"] = parse_bullet_section(content)
                elif current_section == "RISK FACTORS":
                    sections["risk_factors"] = parse_bullet_section(content)
                elif current_section == "WALL STREET SENTIMENT":
                    sections["wall_street"] = parse_bullet_section(content)
                elif current_section == "COMPETITIVE/INDUSTRY DYNAMICS":
                    sections["competitive_industry"] = parse_bullet_section(content)
                elif current_section == "UPCOMING CATALYSTS":
                    sections["upcoming_catalysts"] = parse_bullet_section(content)
                elif current_section == "UPSIDE SCENARIO":
                    sections["upside_scenario"] = [content]
                elif current_section == "DOWNSIDE SCENARIO":
                    sections["downside_scenario"] = [content]
                elif current_section == "KEY VARIABLES TO MONITOR":
                    sections["key_variables"] = parse_variable_section(content)

            # Start new section
            current_section = match.group(1)
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_section and current_content:
        # (same logic as above)
        pass

    return sections


def parse_bullet_section(content: str) -> List[str]:
    """Parse bullet section into list of bullet strings"""
    # Split by **Topic** • Sentiment pattern
    bullet_pattern = r'^\*\*(.+?)\*\* • (.+)$'
    bullets = []
    current_bullet = []

    for line in content.split('\n'):
        if re.match(bullet_pattern, line):
            # Save previous bullet
            if current_bullet:
                bullets.append('\n'.join(current_bullet).strip())
            current_bullet = [line]
        else:
            if line.strip():
                current_bullet.append(line)

    # Save last bullet
    if current_bullet:
        bullets.append('\n'.join(current_bullet).strip())

    return bullets


def parse_variable_section(content: str) -> List[str]:
    """Parse Key Variables section"""
    # Split by ▸ marker
    variables = []
    for line in content.split('\n'):
        if line.startswith('▸'):
            variables.append(line[2:].strip())  # Remove ▸ marker
    return variables
```

### **Function 3: Save to Database**
```python
def save_editorial_summary(
    ticker: str,
    summary_date: date,
    editorial_markdown: str,
    metadata: Dict
) -> bool:
    """
    Save Phase 3 editorial markdown to database.

    Updates executive_summaries table with editorial_markdown column.
    """
    from app import db  # Import from app.py

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE executive_summaries
                SET editorial_markdown = %s
                WHERE ticker = %s AND summary_date = %s
            """, (editorial_markdown, ticker, summary_date))

            conn.commit()
            return True
    except Exception as e:
        LOG.error(f"[{ticker}] Failed to save editorial summary: {e}")
        return False
```

---

## app.py Changes

### **Change 1: Add Email #4 Functions (after line 22237)**

**Location:** Right after `send_user_intelligence_report()` function ends

```python
# Line ~22240 - Add these 2 functions

def generate_email_html_core_editorial(
    ticker: str,
    hours: int = 24,
    recipient_email: str = None
) -> Dict[str, any]:
    """
    Email #4 generation - SAME AS generate_email_html_core but uses editorial_markdown.

    Returns:
        {
            "html": Full HTML email string,
            "subject": Email subject line,
            "company_name": Company name,
            "article_count": Number of articles analyzed
        }
    """
    LOG.info(f"Generating Email #4 (Editorial) for {ticker} (recipient: {recipient_email or 'placeholder'})")

    # ========== COPY FROM generate_email_html_core (lines 21862-21900) ==========
    # Fetch ticker config
    config = get_ticker_config(ticker)
    if not config:
        LOG.error(f"No config found for {ticker}")
        return None

    company_name = config.get("company_name", ticker)
    sector = config.get("sector")
    sector_display = f" • {sector}" if sector and sector.strip() else ""

    # Fetch stock price from ticker_reference (SAME CODE)
    stock_price = "$0.00"
    price_change_pct = None
    price_change_color = "#4ade80"
    ytd_return_pct = None
    ytd_return_color = "#4ade80"

    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT financial_last_price, financial_price_change_pct, financial_ytd_return_pct
            FROM ticker_reference
            WHERE ticker = %s
        """, (ticker,))
        price_data = cur.fetchone()

        if price_data and price_data['financial_last_price']:
            stock_price = f"${price_data['financial_last_price']:.2f}"

            if price_data['financial_price_change_pct'] is not None:
                pct = price_data['financial_price_change_pct']
                price_change_pct = f"{'+' if pct >= 0 else ''}{pct:.2f}%"
                price_change_color = "#4ade80" if pct >= 0 else "#ef4444"

            if price_data['financial_ytd_return_pct'] is not None:
                ytd = price_data['financial_ytd_return_pct']
                ytd_return_pct = f"{'+' if ytd >= 0 else ''}{ytd:.2f}%"
                ytd_return_color = "#4ade80" if ytd >= 0 else "#ef4444"
    # ========== END COPY ==========

    # ========== DIFFERENT: Fetch editorial_markdown instead of summary_text ==========
    editorial_markdown = ""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT editorial_markdown FROM executive_summaries
            WHERE ticker = %s AND summary_date = CURRENT_DATE
            ORDER BY generated_at DESC LIMIT 1
        """, (ticker,))
        result = cur.fetchone()
        if result:
            editorial_markdown = result['editorial_markdown']
        else:
            LOG.warning(f"No editorial markdown found for {ticker}")
            return None

    # ========== DIFFERENT: Parse markdown instead of JSON ==========
    from modules.executive_summary_phase3 import parse_phase3_markdown_to_sections
    try:
        sections = parse_phase3_markdown_to_sections(editorial_markdown)
    except Exception as e:
        LOG.error(f"[{ticker}] Failed to parse Phase 3 markdown in Email #4: {e}")
        sections = {}
    # ========== END DIFFERENT ==========

    # ========== COPY FROM generate_email_html_core (lines 21924-22130) ==========
    # Fetch flagged articles (SAME CODE)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    articles_by_category = {"company": [], "industry": [], "competitor": [], "value_chain": []}

    # ... (copy rest of article fetching logic from generate_email_html_core)
    # ... (copy HTML building logic from generate_email_html_core)
    # ========== END COPY ==========

    # Change subject line
    subject = f"📝 [EDITORIAL BETA] Stock Intelligence: {company_name} ({ticker})"

    return {
        "html": html,
        "subject": subject,
        "company_name": company_name,
        "article_count": article_count
    }


def send_editorial_intelligence_report(
    hours: int = 24,
    tickers: List[str] = None,
    recipient_email: str = None
) -> Dict:
    """
    Email #4 wrapper - SAME AS send_user_intelligence_report but uses editorial version.

    Returns: {"status": "sent" | "failed", "articles_analyzed": X, ...}
    """
    LOG.info("=== EMAIL #4: EDITORIAL STOCK INTELLIGENCE ===")

    # Single ticker only
    if not tickers or len(tickers) == 0:
        return {"status": "error", "message": "No ticker specified"}

    ticker = tickers[0]
    LOG.info(f"Generating editorial report for {ticker} → {recipient_email or DIGEST_TO}")

    # Call editorial core function
    email_data = generate_email_html_core_editorial(
        ticker=ticker,
        hours=hours,
        recipient_email=recipient_email or DIGEST_TO
    )

    if not email_data:
        return {"status": "error", "message": "Failed to generate editorial email HTML"}

    # Send email immediately
    success = send_email(email_data['subject'], email_data['html'], to=recipient_email or DIGEST_TO)

    LOG.info(f"📧 Email #4 (Editorial): {'✅ SENT' if success else '❌ FAILED'} to {recipient_email or DIGEST_TO}")

    return {
        "status": "sent" if success else "failed",
        "articles_analyzed": email_data['article_count'],
        "ticker": ticker,
        "recipient": recipient_email or DIGEST_TO,
        "email_type": "editorial_stock_intelligence"
    }
```

---

### **Change 2: Add Phase 3 Generation + Email #4 Send (after line 24288)**

**Location:** Right after Email #3 is sent successfully

```python
# Line 24281-24292 - Email #3 sent
user_report_result = send_user_intelligence_report(
    hours=int(minutes/60),
    tickers=[ticker],
    flagged_article_ids=flagged_article_ids,
    recipient_email=DIGEST_TO
)
if user_report_result:
    LOG.info(f"[{ticker}] ✅ [JOB {job_id}] Email #3 sent successfully")
    if isinstance(user_report_result, dict):
        LOG.info(f"   Status: {user_report_result.get('status', 'unknown')}")

    # ========== NEW: Phase 3 Generation + Email #4 ==========
    LOG.info(f"[{ticker}] 🎨 [JOB {job_id}] Generating Phase 3 editorial format...")

    try:
        from modules.executive_summary_phase3 import (
            generate_executive_summary_phase3,
            save_editorial_summary
        )

        # Fetch merged JSON from database
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT summary_text FROM executive_summaries
                WHERE ticker = %s AND summary_date = CURRENT_DATE
                ORDER BY generated_at DESC LIMIT 1
            """, (ticker,))
            result = cur.fetchone()

        if result and result['summary_text']:
            # Parse merged JSON
            merged_json = json.loads(result['summary_text'])

            # Generate Phase 3 markdown
            phase3_result = generate_executive_summary_phase3(
                ticker=ticker,
                merged_json=merged_json,
                anthropic_api_key=ANTHROPIC_API_KEY
            )

            if phase3_result and phase3_result.get('markdown'):
                # Save to database
                success = save_editorial_summary(
                    ticker=ticker,
                    summary_date=datetime.now().date(),
                    editorial_markdown=phase3_result['markdown'],
                    metadata=phase3_result
                )

                if success:
                    LOG.info(f"[{ticker}] ✅ [JOB {job_id}] Phase 3 editorial saved to database")

                    # Send Email #4 immediately
                    LOG.info(f"[{ticker}] 📧 [JOB {job_id}] Sending Email #4 (Editorial)...")
                    editorial_result = send_editorial_intelligence_report(
                        hours=int(minutes/60),
                        tickers=[ticker],
                        recipient_email=DIGEST_TO
                    )

                    if editorial_result and editorial_result.get('status') == 'sent':
                        LOG.info(f"[{ticker}] ✅ [JOB {job_id}] Email #4 sent successfully")
                    else:
                        LOG.warning(f"[{ticker}] ⚠️ [JOB {job_id}] Email #4 send failed (non-fatal)")
                else:
                    LOG.error(f"[{ticker}] ❌ [JOB {job_id}] Failed to save Phase 3 to database")
            else:
                LOG.warning(f"[{ticker}] ⚠️ [JOB {job_id}] Phase 3 returned no markdown")
        else:
            LOG.warning(f"[{ticker}] ⚠️ [JOB {job_id}] No merged JSON found for Phase 3")

    except Exception as e:
        LOG.error(f"[{ticker}] ❌ [JOB {job_id}] Phase 3 generation/send failed: {e}", exc_info=True)
        # Non-fatal - Email #3 already sent, continue to GitHub commit
    # ========== END Phase 3 ==========

else:
    LOG.warning(f"[{ticker}] ⚠️ [JOB {job_id}] Email #3 returned no result")

# Continue to cancellation check and GitHub commit (line 24297)...
```

---

## Implementation Checklist

### **Step 1: Create Phase 3 Module**
- [ ] Create `modules/_build_executive_summary_prompt_phase3` (copy from PHASE3_PROMPT_DRAFT.md)
- [ ] Create `modules/executive_summary_phase3.py` with 3 functions:
  - [ ] `generate_executive_summary_phase3()`
  - [ ] `parse_phase3_markdown_to_sections()`
  - [ ] `save_editorial_summary()`

### **Step 2: Modify app.py**
- [ ] Line ~22240: Add `generate_email_html_core_editorial()` (copy Email #3, change 10 lines)
- [ ] Line ~22240: Add `send_editorial_intelligence_report()` (copy Email #3, change 5 lines)
- [ ] Line ~24292: Add Phase 3 generation + Email #4 send (after Email #3)

### **Step 3: Database Migration**
- [ ] Run SQL: `ALTER TABLE executive_summaries ADD COLUMN editorial_markdown TEXT;`

### **Step 4: Test**
- [ ] Test on single ticker (AAPL)
- [ ] Verify Email #3 still works
- [ ] Verify Email #4 is sent after Email #3
- [ ] Verify markdown parsing works
- [ ] Check email HTML matches Email #3 structure

---

## Key Implementation Details

### **Parse Markdown - Robust Regex Patterns**

**Section Header Detection:**
```python
# Match: ## SECTION NAME
section_pattern = r'^## (.+)$'
```

**Bullet Detection:**
```python
# Match: **Topic** • Sentiment (reason)
# Match: **[Entity] Topic** • Sentiment (reason)
bullet_pattern = r'^\*\*(\[.+?\] )?(.+?)\*\* • (.+)$'
```

**Variable Detection:**
```python
# Match: ▸ **Variable Name**: Description
variable_pattern = r'^▸ \*\*(.+?)\*\*: (.+)$'
```

### **Section Name Mapping**

```python
section_map = {
    "BOTTOM LINE": "bottom_line",
    "MAJOR DEVELOPMENTS": "major_developments",
    "FINANCIAL/OPERATIONAL PERFORMANCE": "financial_operational",
    "RISK FACTORS": "risk_factors",
    "WALL STREET SENTIMENT": "wall_street",
    "COMPETITIVE/INDUSTRY DYNAMICS": "competitive_industry",
    "UPCOMING CATALYSTS": "upcoming_catalysts",
    "UPSIDE SCENARIO": "upside_scenario",
    "DOWNSIDE SCENARIO": "downside_scenario",
    "KEY VARIABLES TO MONITOR": "key_variables"
}
```

---

## Testing Strategy

### **Test 1: Single Ticker**
```python
# Run job queue on AAPL
# Verify 2 emails received:
#   1. 📊 Stock Intelligence: Apple Inc. (AAPL)
#   2. 📝 [EDITORIAL BETA] Stock Intelligence: Apple Inc. (AAPL)
```

### **Test 2: Markdown Parsing**
```python
# Check database:
SELECT ticker, LENGTH(editorial_markdown), editorial_markdown
FROM executive_summaries
WHERE ticker = 'AAPL' AND summary_date = CURRENT_DATE;

# Verify markdown has proper structure:
# - ## BOTTOM LINE
# - **Topic** • Sentiment (reason)
# - Proper spacing
```

### **Test 3: Email HTML**
```python
# Compare Email #3 vs Email #4 HTML
# Should be identical structure:
# - Same header (stock price card)
# - Same section layout
# - Same footer (legal disclaimers)
# Only difference: Content (Email #3 = raw bullets, Email #4 = editorial format)
```

---

## Rollback Plan

If Email #4 has issues:

1. **Quick disable:** Comment out Phase 3 code block (lines 24292-24330)
2. **Email #3 unchanged:** Continues working normally
3. **No data loss:** editorial_markdown column just stays NULL

---

## Timeline Estimate

- **Step 1 (Module):** 15 minutes
- **Step 2 (app.py):** 20 minutes (mostly copy-paste)
- **Step 3 (DB):** 2 minutes
- **Step 4 (Test):** 15 minutes

**Total: ~1 hour implementation + testing**

---

## Open Questions (RESOLVED)

1. ✅ **Phase 3 placement:** After Email #3 sent (line 24292)
2. ✅ **Subject line:** `📝 [EDITORIAL BETA] Stock Intelligence: {ticker}`
3. ✅ **Implementation approach:** Copy-paste Email #3 functions
4. ✅ **Environment variables:** None - just send it
5. ✅ **Feature flags:** None - always generate Phase 3 after Email #3

---

## Success Criteria

- [ ] Phase 3 generates markdown for every ticker
- [ ] Email #4 sent after Email #3 (no failures)
- [ ] Markdown parsing works 100% of time
- [ ] Email #4 HTML matches Email #3 structure
- [ ] No performance impact on Email #3
- [ ] Clean logs (no errors in Phase 3 generation)

---

**Status:** Ready to implement
**Next Action:** Code the 3 changes (module + 2 app.py changes)
**Estimated Time:** 1 hour

---

**Last Updated:** 2025-01-09 (Simplified to copy Email #3 pattern)
