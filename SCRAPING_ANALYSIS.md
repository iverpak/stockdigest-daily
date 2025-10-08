# StockDigest Scraping Analysis - Content Capture Review

**Date:** October 8, 2025
**Issue:** Concerns about missing relevant information in scraped articles

---

## Executive Summary

✅ **Good News:** No hard truncation or paragraph limits found
⚠️ **Content Loss Areas:** Aggressive cleaning filters removing potentially relevant content
🔍 **Main Issues:** Content validation rules and cleaning stages may be too strict

---

## 1. Scraping Architecture (2-Tier Fallback)

### Tier 1: Requests + newspaper3k (~70% success)
- **Method:** HTTP requests + newspaper3k parser
- **Speed:** Fast (~2-5 seconds)
- **Cost:** Free
- **Extracts:** `article.text` from newspaper3k library

### Tier 2: Scrapfly (~95% success)
- **Method:** Scrapfly API with anti-bot protection
- **Speed:** Moderate (~5-10 seconds)
- **Cost:** $0.002 per article
- **Extracts:** HTML → newspaper3k parser

**Playwright (Tier 2) - DISABLED since Oct 6, 2025**
- Reason: Caused indefinite hangs on problematic domains

---

## 2. Content Extraction Flow

```
RSS Feed → Article URL → Scraping (Tier 1 or 2) → Raw Content
    ↓
newspaper3k.Article.text (full article extraction)
    ↓
clean_scraped_content() (7 cleaning stages)
    ↓
validate_scraped_content() (quality checks)
    ↓
PostgreSQL (TEXT column, no length limit)
```

---

## 3. Potential Content Loss Points

### ⚠️ Issue #1: Aggressive Content Cleaning (Lines 3680-3811)

**7 Cleaning Stages:**

#### Stage 1: Binary/Encoded Data Removal
```python
# Removes sequences like: ÂżÂ½ÂżÂ½ÂżÂ½
content = re.sub(r'[Â¿Â½]{3,}.*?[Â¿Â½]{3,}', '', content)
```
**Risk:** Low - only removes corrupted encoding

#### Stage 2: HTML/CSS/JavaScript Remnants
```python
# Removes HTML tags newspaper3k missed
content = re.sub(r'<[^>]+>', '', content)
content = re.sub(r'\{[^}]*\}', '', content)  # CSS rules
```
**Risk:** Medium - `\{[^}]*\}` could remove valid text with curly braces

#### Stage 3: Technical Metadata Removal
```python
# Removes image metadata and technical codes
content = re.sub(r'[A-Z]{2,}\d+[A-Z]*\d*', '', content)  # Removes: RGB123, NASDAQ, etc.
content = re.sub(r'\d{8,}', '', content)  # Removes 8+ digit numbers
```
**Risk:** HIGH ⚠️
- **Problem:** `\d{8,}` removes important financial data like:
  - Market cap: $15,000,000
  - Revenue figures: $1,234,567,890
  - Share counts: 150,000,000 shares
- **Problem:** `[A-Z]{2,}\d+` removes tickers and financial codes:
  - NASDAQ-listed companies
  - SEC filings like "10K", "8K"

#### Stage 4: Navigation/UI Element Removal
```python
navigation_patterns = [
    r'Home\s*>\s*[^.]*',  # Breadcrumbs
    r'Share on [A-Za-z]+',  # Social sharing
    r'Related Articles?',
    r'Continue reading',
    r'Advertisement',
]
```
**Risk:** Low - appropriate removals

#### Stage 5: Cookie/Consent Text Removal
```python
cookie_patterns = [
    r'We use cookies[^.]*\.',
    r'Cookie policy[^.]*\.',
]
```
**Risk:** Low - appropriate removals

#### Stage 6: Whitespace Cleanup
```python
content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Max 2 line breaks
```
**Risk:** Low - preserves paragraph structure

#### Stage 7: Line-by-Line Filtering
```python
for line in lines:
    # Skip lines that are mostly special characters or numbers
    if len(re.sub(r'[a-zA-Z\s]', '', line)) > len(line) * 0.5:
        continue

    # Skip very short fragments that don't end properly
    if len(line) < 20 and not line.endswith(('.', '!', '?', ':')):
        continue
```
**Risk:** MEDIUM-HIGH ⚠️
- **Problem:** Skips lines with >50% special chars/numbers
  - Could remove financial data tables
  - Could remove bullet points with figures
- **Problem:** Skips lines <20 chars without proper ending
  - Could remove short but important statements
  - Could remove data points

---

### ⚠️ Issue #2: Content Validation Rules (Lines 3933-3963)

**Validation Checks:**

#### Check 1: Minimum Length
```python
if not content or len(content.strip()) < 100:
    return False, "Content too short"
```
**Risk:** Low - 100 chars is reasonable

#### Check 2: Minimum Sentences
```python
sentences = [s.strip() for s in content.split('.') if s.strip()]
min_sentences = 2 if is_quality_domain else 3
if len(sentences) < min_sentences:
    return False, f"Insufficient sentences"
```
**Risk:** LOW-MEDIUM
- **Problem:** Splits on periods (`.`), which breaks:
  - Financial abbreviations: "Inc.", "Corp.", "U.S."
  - Decimal numbers: "3.5%", "$12.50"
  - Abbreviations: "CEO John Smith, Jr."
- **Result:** Could overcount sentences and reject valid short articles

#### Check 3: Unique Word Ratio
```python
unique_ratio = len(set(words)) / len(words)
min_ratio = 0.25 if is_quality_domain else 0.3
if unique_ratio < min_ratio:
    return False, "Repetitive content detected"
```
**Risk:** Medium
- **Problem:** Financial articles repeat key terms:
  - "stock", "shares", "price", "revenue", "earnings"
  - Company name repeated frequently
- **Could reject:** Legitimate financial analysis with focused vocabulary

#### Check 4: Technical Character Filter
```python
technical_chars = len(re.findall(r'[{}();:=<>]', content))
if technical_chars > len(content) * 0.15:  # Quality domains
    return False, "Content appears to be technical/code data"
```
**Risk:** Low-Medium
- **Problem:** Could flag financial tables with comparison operators
- Example: "Revenue: $500M (up 15% >$435M last year)"

---

### ⚠️ Issue #3: newspaper3k Library Limitations

**Known Issues:**
1. **Paywall Detection:** May stop reading before full article
2. **JavaScript Rendering:** Doesn't execute JS (handled by Scrapfly Tier 2)
3. **Comment Sections:** Sometimes includes user comments
4. **Article Selection:** Picks "main" article div, might miss sidebars with key data

**Your Version:** 0.2.8 (latest is 0.2.8, so up to date)

---

## 4. Database Storage

✅ **No Truncation Here:**
```sql
CREATE TABLE articles (
    scraped_content TEXT,  -- PostgreSQL TEXT = unlimited length
)
```

**Storage Function:** `update_article_content()` (Line 1185)
- No truncation
- Stores full cleaned content

---

## 5. What Content Might Be Lost?

### High Risk Content Loss:

**1. Financial Numbers (Stage 3)**
```python
# REMOVES 8+ digit numbers
content = re.sub(r'\d{8,}', '', content)
```
**Lost:**
- "$15,000,000 revenue" → "$revenue"
- "150,000,000 shares" → "shares"
- "Market cap: $1,234,567,890" → "Market cap: $"

**2. Financial Codes (Stage 3)**
```python
# REMOVES: [A-Z]{2,}\d+
content = re.sub(r'[A-Z]{2,}\d+[A-Z]*\d*', '', content)
```
**Lost:**
- "NASDAQ100" → ""
- "SEC Form 10K" → "SEC Form "
- "Q3 2024" → (might be preserved if space before digit)

**3. Data-Heavy Lines (Stage 7)**
```python
# Skips lines with >50% special chars/numbers
if len(re.sub(r'[a-zA-Z\s]', '', line)) > len(line) * 0.5:
    continue
```
**Lost:**
- "Revenue: $500M (up 15%)" → SKIPPED (52% special chars)
- "EPS: $3.50 vs $2.75 est" → SKIPPED (50% special chars)
- Bullet points with data

**4. Short Important Statements (Stage 7)**
```python
# Skips lines <20 chars without proper punctuation
if len(line) < 20 and not line.endswith(('.', '!', '?', ':')):
    continue
```
**Lost:**
- "Stock up 25%" → SKIPPED (13 chars, no period)
- "Revenue beat" → SKIPPED (12 chars)

---

## 6. Recommendations

### 🔧 Quick Fixes (High Impact)

#### Fix #1: Preserve Financial Numbers
```python
# BEFORE (Line 3726):
content = re.sub(r'\d{8,}', '', content)  # Removes ALL 8+ digit numbers

# AFTER:
# Don't remove numbers in financial context
content = re.sub(r'(?<![0-9$,.])\d{8,}(?![0-9$,.])', '', content)
# Only removes standalone 8+ digit codes, preserves $1,234,567,890
```

#### Fix #2: Preserve Financial Codes
```python
# BEFORE (Line 3725):
content = re.sub(r'[A-Z]{2,}\d+[A-Z]*\d*', '', content)

# AFTER:
# Whitelist common financial terms
financial_codes = r'(?!Q[1-4]|10K|10Q|8K|S&P|NASDAQ|NYSE)'
content = re.sub(financial_codes + r'[A-Z]{2,}\d+[A-Z]*\d*', '', content)
```

#### Fix #3: Relax Line-by-Line Filtering
```python
# BEFORE (Line 3779):
if len(re.sub(r'[a-zA-Z\s]', '', line)) > len(line) * 0.5:
    continue

# AFTER:
# Increase threshold to 70% to preserve data-heavy lines
if len(re.sub(r'[a-zA-Z\s]', '', line)) > len(line) * 0.7:
    continue
```

#### Fix #4: Preserve Short Important Lines
```python
# BEFORE (Line 3783):
if len(line) < 20 and not line.endswith(('.', '!', '?', ':')):
    continue

# AFTER:
# Preserve lines with financial keywords even if short
financial_keywords = ['revenue', 'earnings', 'eps', 'stock', 'shares', '%', '$']
has_financial = any(kw in line.lower() for kw in financial_keywords)
if len(line) < 20 and not (line.endswith(('.', '!', '?', ':')) or has_financial):
    continue
```

---

### 📊 Testing Recommendations

**1. Before/After Logging**
Add logging to see what's being removed:

```python
# In clean_scraped_content() at Line 3806
if reduction > 30:  # Log if >30% content removed
    LOG.warning(f"⚠️ High content reduction: {domain} ({reduction:.1f}%)")
    LOG.warning(f"   Original: {original_length} chars → Final: {final_length} chars")
```

**2. Sample Article Analysis**
Test with known articles:
1. Pick 5 articles with lots of financial data
2. Compare newspaper3k output vs. cleaned output
3. Identify what's being stripped

---

## 7. No Issues Found

✅ **No Paragraph Limits:** newspaper3k extracts full article
✅ **No Database Truncation:** TEXT column = unlimited
✅ **No Hard Caps:** No max character or paragraph count
✅ **Proper Tier Fallback:** Scrapfly handles tough sites

---

## Conclusion

**Main Issue:** Content cleaning is too aggressive for financial content.

**Impact Areas:**
1. ⚠️ **HIGH:** Financial numbers (8+ digits) completely removed
2. ⚠️ **HIGH:** Data-heavy lines (>50% numbers/special chars) skipped
3. ⚠️ **MEDIUM:** Financial codes (NASDAQ, 10K) partially removed
4. ⚠️ **MEDIUM:** Short data points (<20 chars) skipped

**Recommendation:** Implement fixes #1-4 above to preserve financial data while still removing junk.

**Testing:** Add logging to monitor content reduction % per article.
