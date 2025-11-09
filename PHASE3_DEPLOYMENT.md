# Phase 3 Editorial Format - Deployment Guide

**Implementation Date:** 2025-01-09
**Status:** Ready to Deploy

---

## What Was Implemented

Phase 3 transforms Phase 1+2 merged JSON into scannable, professional editorial format (Email #4).

### Code Changes:
1. **modules/_build_executive_summary_prompt_phase3** - Phase 3 prompt file
2. **modules/executive_summary_phase3.py** - 3 functions (generate, parse, save)
3. **app.py lines 22240-22587** - Email #4 generation functions
4. **app.py lines 24642-24705** - Phase 3 integration in job processing

---

## Deployment Checklist

### Step 1: Run Database Migration (REQUIRED)

```sql
ALTER TABLE executive_summaries ADD COLUMN IF NOT EXISTS editorial_markdown TEXT;
```

**How to run:**
1. Go to Render Dashboard → Database
2. Connect via psql or Render shell
3. Run the SQL above
4. Verify: `\d executive_summaries` should show `editorial_markdown` column

---

### Step 2: Deploy to Render

1. Commit all changes:
```bash
git add .
git commit -m "Add Phase 3 Editorial Format (Email #4)

- Phase 3 formatter transforms Phase 1+2 JSON to editorial markdown
- Email #4 sent after Email #3 with [EDITORIAL BETA] subject
- Reuses Email #3 HTML template exactly
- No environment variables needed (always runs)
- Sequential: Email #3 → Phase 3 gen → Email #4

Files:
- modules/_build_executive_summary_prompt_phase3
- modules/executive_summary_phase3.py
- app.py (Email #4 functions + Phase 3 integration)

Database: Added editorial_markdown TEXT column to executive_summaries"

git push origin main
```

2. Render will auto-deploy (takes ~5 minutes)

---

### Step 3: Verify Deployment

1. **Check logs** after deployment:
```
# Look for Phase 3 initialization
grep "Phase 3" render-logs.txt
```

2. **Run test ticker** (manually trigger via job queue):
```
Submit AAPL job via /admin/test
```

3. **Expected logs:**
```
[AAPL] ✅ [JOB xxx] Email #3 sent successfully
[AAPL] 🎨 [JOB xxx] Generating Phase 3 editorial format...
[AAPL] ✅ [JOB xxx] Phase 3 editorial saved to database
[AAPL] 📧 [JOB xxx] Sending Email #4 (Editorial)...
[AAPL] ✅ [JOB xxx] Email #4 sent successfully
```

4. **Check inbox:** You should receive 2 emails:
   - 📊 Stock Intelligence: Apple Inc. (AAPL)
   - 📝 [EDITORIAL BETA] Stock Intelligence: Apple Inc. (AAPL)

5. **Verify database:**
```sql
SELECT ticker,
       LENGTH(summary_text) as phase12_len,
       LENGTH(editorial_markdown) as phase3_len,
       generated_at
FROM executive_summaries
WHERE ticker = 'AAPL'
  AND summary_date = CURRENT_DATE
ORDER BY generated_at DESC
LIMIT 1;
```

Should show both `summary_text` (Phase 1+2 JSON) and `editorial_markdown` (Phase 3 markdown) populated.

---

## How It Works

### Workflow:
```
Email #3 sent → Phase 3 generation (30-45s) → Email #4 sent
```

### Sequential Execution:
- No delays, no async, no threads
- Runs immediately after Email #3
- Non-fatal if Phase 3 fails (Email #3 already sent)

### What Phase 3 Does:
1. Fetches Phase 1+2 merged JSON from database
2. Calls Claude API with Phase 3 prompt (markdown formatter)
3. Saves markdown to `editorial_markdown` column
4. Parses markdown back to sections dict
5. Reuses Email #3 HTML builder exactly
6. Sends Email #4 with `[EDITORIAL BETA]` subject

---

## Performance Impact

- **Additional time per ticker:** 30-45 seconds (Claude API call)
- **API cost:** ~$0.40 per ticker (Phase 3 prompt caching enabled)
- **No impact on Email #3:** Sequential execution, Email #3 always completes first

---

## Rollback Plan

If Phase 3 causes issues:

### Quick Disable (Comment out integration):
Edit `app.py` around line 24642, comment out the Phase 3 block:
```python
# ========== NEW: Phase 3 Generation + Email #4 ==========
# LOG.info(f"[{ticker}] 🎨 [JOB {job_id}] Generating Phase 3 editorial format...")
# ... (comment entire block)
# ========== END Phase 3 ==========
```

Redeploy. Email #3 will continue working normally.

### Database Cleanup (if needed):
```sql
-- Clear editorial_markdown data
UPDATE executive_summaries SET editorial_markdown = NULL;

-- Or drop column entirely
ALTER TABLE executive_summaries DROP COLUMN editorial_markdown;
```

---

## Monitoring

### Key Metrics to Watch:

1. **Phase 3 Success Rate:**
```sql
SELECT
    COUNT(*) FILTER (WHERE editorial_markdown IS NOT NULL) as phase3_success,
    COUNT(*) as total,
    ROUND(100.0 * COUNT(*) FILTER (WHERE editorial_markdown IS NOT NULL) / COUNT(*), 1) as success_rate_pct
FROM executive_summaries
WHERE generated_at > NOW() - INTERVAL '24 hours';
```

Target: >95% success rate

2. **Phase 3 Generation Time:**
Check logs for: `Phase 3 markdown generated (X ms)`
Target: <45 seconds

3. **Email #4 Delivery Rate:**
Check logs for: `Email #4 sent successfully`
Target: 100% (should match Phase 3 success rate)

4. **API Cost:**
Check Anthropic dashboard for Phase 3 token usage
Expected: ~15K prompt tokens, ~3.5K completion tokens per ticker

---

## Troubleshooting

### Issue: Phase 3 generation fails

**Symptoms:**
```
[TICKER] ⚠️ [JOB xxx] Phase 3 returned no markdown
```

**Check:**
1. Claude API key valid?
2. Merged JSON exists in database?
3. Claude API rate limits?

**Fix:**
- Check Anthropic dashboard for API errors
- Verify `summary_text` column populated
- Add retry logic if rate limited

---

### Issue: Email #4 parsing fails

**Symptoms:**
```
[TICKER] Failed to parse Phase 3 markdown in Email #4
```

**Check:**
1. Markdown format from Claude valid?
2. Section headers match expected format?

**Debug:**
```sql
SELECT editorial_markdown
FROM executive_summaries
WHERE ticker = 'TICKER'
  AND summary_date = CURRENT_DATE
ORDER BY generated_at DESC LIMIT 1;
```

Verify markdown has `## BOTTOM LINE`, `## MAJOR DEVELOPMENTS`, etc.

---

### Issue: Email #4 not sent

**Symptoms:**
```
[TICKER] ⚠️ [JOB xxx] Email #4 send failed (non-fatal)
```

**Check:**
1. Email #4 HTML generated?
2. SMTP settings valid?
3. Recipient email valid?

**Non-fatal:** Email #3 already sent, so user still gets content.

---

## Success Criteria

After deploying to production:

- [ ] Database migration completed successfully
- [ ] Test ticker (AAPL) generates 2 emails (Email #3 + Email #4)
- [ ] Email #4 HTML structure matches Email #3 (same header, footer, layout)
- [ ] Markdown parsing works (sections render correctly)
- [ ] Phase 3 generation completes in <45 seconds
- [ ] No errors in logs for Phase 3 block
- [ ] Daily workflow continues working normally

---

## Future Enhancements (Post-Launch)

1. **A/B Testing:**
   - Track open rates: Email #3 vs Email #4
   - User feedback surveys
   - Decide which format to keep long-term

2. **Prompt Optimization:**
   - Iterate on Phase 3 prompt based on output quality
   - Adjust bolding density, section structure
   - Add more integration examples

3. **Performance:**
   - Batch Phase 3 generation (multiple tickers in one API call)
   - Cache Phase 3 prompt more aggressively
   - Parallel Email #3 + Phase 3 generation?

---

**Deployment Ready:** Yes ✅
**Risk Level:** Low (Email #3 unchanged, Phase 3 non-fatal)
**Estimated Downtime:** 0 minutes (rolling deploy)

**Contact:** See CLAUDE.md for support information
