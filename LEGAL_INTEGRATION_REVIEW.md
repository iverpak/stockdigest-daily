# StockDigest Legal Integration Review

**Date:** October 7, 2025
**Status:** Ready for Implementation

---

## TEMPLATES SAVED ✅

All 4 HTML templates have been saved to `/templates/`:

1. ✅ `email_intelligence_report.html` - Email #3 (Premium Intelligence Report)
2. ✅ `landing_page.html` - Updated beta signup page with Terms checkbox
3. ✅ `terms_of_service.html` - Terms of Service page
4. ✅ `privacy_policy.html` - Privacy Policy page

---

## CURRENT STATE ANALYSIS

### Landing Page
- **Current:** Uses `signup.html` template (older version without disclaimers)
- **Route:** `@APP.get("/")` at line 12922
- **New Template:** `landing_page.html` adds:
  - Top disclaimer banner
  - Terms/Privacy checkbox (required before signup)
  - Footer links to Terms, Privacy, Contact
  - Same ticker validation functionality (already works)

### Email #3 (Premium Intelligence Report)
- **Current:** Uses inline HTML generation in `send_user_intelligence_report()` (line 11744)
- **Issue:** No legal disclaimers in header/footer
- **New Template:** `email_intelligence_report.html` adds:
  - Top disclaimer banner
  - Comprehensive footer disclaimer
  - Links to Terms, Privacy, Contact, Unsubscribe
  - **IMPORTANT:** Uses Jinja2 template variables (needs conversion)

### Beta Signup
- **Current:** Working beta signup at `/api/beta-signup` (line 13002)
- **Database:** `beta_users` table exists (line 1049)
- **Missing:** No timestamp for when user accepted Terms

### Legal Pages
- **Current:** Routes do NOT exist yet
- **Need to add:**
  - `GET /terms-of-service` → `terms_of_service.html`
  - `GET /privacy-policy` → `privacy_policy.html`

### Unsubscribe Functionality
- **Current:** Does NOT exist
- **Need to implement:**
  - Generate unique unsubscribe tokens per user
  - Store tokens in database (new table or column)
  - Create `GET /unsubscribe` endpoint
  - Add token to all email footers

---

## IMPLEMENTATION PLAN

### Phase 1: Update Landing Page ✅ (Easiest)

**File to modify:** `app.py` line 12925

**Change:**
```python
# OLD
return templates.TemplateResponse("signup.html", {"request": request})

# NEW
return templates.TemplateResponse("landing_page.html", {"request": request})
```

**Impact:**
- Users will see disclaimer banner
- Users must check Terms/Privacy box before submitting
- Footer links to Terms/Privacy will work (after Phase 2)

---

### Phase 2: Add Terms & Privacy Routes (Easy)

**File to modify:** `app.py` (add after line 12925)

**New routes needed:**
```python
@APP.get("/terms-of-service", response_class=HTMLResponse)
async def terms_page(request: Request):
    """Serve Terms of Service page"""
    return templates.TemplateResponse("terms_of_service.html", {"request": request})

@APP.get("/privacy-policy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    """Serve Privacy Policy page"""
    return templates.TemplateResponse("privacy_policy.html", {"request": request})
```

**Impact:**
- `/terms-of-service` and `/privacy-policy` pages will work
- All footer links across landing page, Terms, and Privacy will work

---

### Phase 3: Database Schema Updates (Medium)

**3A. Add Terms Acceptance Timestamp**

Update `beta_users` table schema (line 1049):
```sql
ALTER TABLE beta_users ADD COLUMN IF NOT EXISTS terms_accepted_at TIMESTAMPTZ;
```

**3B. Add Unsubscribe Tokens Table**

Create new table:
```sql
CREATE TABLE IF NOT EXISTS unsubscribe_tokens (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    token VARCHAR(64) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    used_at TIMESTAMPTZ,
    FOREIGN KEY (user_email) REFERENCES beta_users(email) ON DELETE CASCADE
);

CREATE INDEX idx_unsubscribe_tokens_token ON unsubscribe_tokens(token);
CREATE INDEX idx_unsubscribe_tokens_email ON unsubscribe_tokens(user_email);
```

**3C. Update Beta Signup Handler**

Modify `/api/beta-signup` endpoint (line 13002) to:
- Log `terms_accepted_at = NOW()`
- Generate unsubscribe token
- Store token in `unsubscribe_tokens` table

---

### Phase 4: Implement Unsubscribe Functionality (Medium)

**4A. Generate Tokens Function**

```python
import secrets

def generate_unsubscribe_token(email: str) -> str:
    """Generate and store unique unsubscribe token for user"""
    token = secrets.token_urlsafe(32)  # 32 bytes = 43 chars

    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO unsubscribe_tokens (user_email, token)
            VALUES (%s, %s)
            ON CONFLICT (token) DO NOTHING
            RETURNING token
        """, (email, token))
        result = cur.fetchone()
        if not result:
            # Token collision (extremely rare), retry
            return generate_unsubscribe_token(email)
        conn.commit()

    return token
```

**4B. Unsubscribe Endpoint**

```python
@APP.get("/unsubscribe")
async def unsubscribe_page(token: str = Query(...)):
    """Handle unsubscribe requests via token"""

    with db() as conn, conn.cursor() as cur:
        # Validate token and get email
        cur.execute("""
            SELECT user_email, used_at FROM unsubscribe_tokens
            WHERE token = %s
        """, (token,))
        result = cur.fetchone()

        if not result:
            return HTMLResponse("<h1>Invalid unsubscribe link</h1>", status_code=404)

        email = result['user_email']
        already_used = result['used_at'] is not None

        if not already_used:
            # Mark user as unsubscribed
            cur.execute("""
                UPDATE beta_users SET status = 'cancelled' WHERE email = %s
            """, (email,))

            # Mark token as used
            cur.execute("""
                UPDATE unsubscribe_tokens SET used_at = NOW() WHERE token = %s
            """, (token,))

            conn.commit()

        # Return confirmation page
        return HTMLResponse(f"""
            <h1>Successfully Unsubscribed</h1>
            <p>Email: {email}</p>
            <p>You will no longer receive daily digests.</p>
        """)
```

---

### Phase 5: Update Email #3 Template (Complex)

**Current Function:** `send_user_intelligence_report()` at line 11744

**Issue:** Currently uses inline HTML string building (lines 11958-12053)

**Solution Options:**

**Option A: Full Jinja2 Refactor (Recommended)**

Convert to use `email_intelligence_report.html` template:

```python
def send_user_intelligence_report(...):
    # ... existing code to fetch data ...

    # Build executive summary sections HTML
    executive_summary_html = build_summary_sections(sections)

    # Build articles HTML
    articles_html = build_articles_html(articles_by_category)

    # Get unsubscribe token
    unsubscribe_token = get_or_create_unsubscribe_token(user_email)
    unsubscribe_url = f"https://stockdigest.app/unsubscribe?token={unsubscribe_token}"

    # Render template
    html = templates.TemplateResponse(
        "email_intelligence_report.html",
        {
            "request": None,  # Not needed for email
            "ticker": ticker,
            "company_name": company_name,
            "industry": sector,
            "current_date": current_date,
            "stock_price": stock_price,
            "price_change": price_change_pct,
            "price_change_color": price_change_color,
            "executive_summary_html": executive_summary_html,
            "total_articles": total_articles,
            "paywalled_count": paywalled_count,
            "lookback_days": hours // 24,
            "articles_html": articles_html,
            "unsubscribe_url": unsubscribe_url
        }
    ).body.decode('utf-8')

    # Send email...
```

**Option B: Minimal Changes (Quick Fix)**

Just add disclaimer divs to existing inline HTML:

- Add top disclaimer banner after `<table>` opening
- Replace footer section with new legal disclaimer box
- Add Terms/Privacy/Unsubscribe links

---

## GLOBAL REPLACEMENTS NEEDED

Search and replace across ALL 4 HTML templates:

1. ✅ Already done in templates:
   - `support@stockdigest.app` → `stockdigest.research@gmail.com`
   - `[Your Jurisdiction]` → `Province of Ontario, Canada`
   - `[Your Country]` → `Canada`

2. **Email template only:**
   - Add unsubscribe URL: `{{ unsubscribe_url }}`
   - Ensure all links use full URLs: `https://stockdigest.app/...`

---

## LINK MAPPING VERIFICATION

### Landing Page (`landing_page.html`)
- ✅ Footer: `/terms-of-service` → Works after Phase 2
- ✅ Footer: `/privacy-policy` → Works after Phase 2
- ✅ Footer: `mailto:stockdigest.research@gmail.com` → Works now
- ✅ Checkbox: `/terms-of-service` (new tab) → Works after Phase 2
- ✅ Checkbox: `/privacy-policy` (new tab) → Works after Phase 2

### Terms Page (`terms_of_service.html`)
- ✅ Header: `← Back to Home` → `/` → Works now
- ✅ Body: Link to Privacy Policy → `/privacy-policy` → Works after Phase 2
- ✅ Contact: `stockdigest.research@gmail.com` → Works now
- ✅ Footer: All links → Work after Phase 2

### Privacy Page (`privacy_policy.html`)
- ✅ Header: `← Back to Home` → `/` → Works now
- ✅ Body: Link to Terms → `/terms-of-service` → Works after Phase 2
- ✅ Contact: `stockdigest.research@gmail.com` → Works now
- ✅ Footer: All links → Work after Phase 2

### Email Template (`email_intelligence_report.html`)
- ⚠️ Footer: `https://stockdigest.app/terms-of-service` → **MUST be full URL**
- ⚠️ Footer: `https://stockdigest.app/privacy-policy` → **MUST be full URL**
- ⚠️ Footer: `{{ unsubscribe_url }}` → **MUST implement token system**
- ✅ Footer: `mailto:stockdigest.research@gmail.com` → Works now

---

## TESTING CHECKLIST

### Phase 1 Testing (Landing Page)
- [ ] Navigate to `/` - should show new landing page
- [ ] Disclaimer banner visible at top
- [ ] Try submitting without checking Terms box → Should block
- [ ] Check Terms box → Submit should work
- [ ] Footer links to `/terms-of-service` and `/privacy-policy` → Should 404 until Phase 2

### Phase 2 Testing (Legal Pages)
- [ ] Navigate to `/terms-of-service` → Page loads
- [ ] Navigate to `/privacy-policy` → Page loads
- [ ] Click "Back to Home" on both pages → Returns to `/`
- [ ] Click Terms link from Privacy page → Goes to Terms
- [ ] Click Privacy link from Terms page → Goes to Privacy
- [ ] All contact emails show `stockdigest.research@gmail.com`

### Phase 3 Testing (Database)
- [ ] Signup new user → Check `beta_users.terms_accepted_at` is set
- [ ] Check `unsubscribe_tokens` table exists
- [ ] Signup creates token in `unsubscribe_tokens`

### Phase 4 Testing (Unsubscribe)
- [ ] Get unsubscribe token from database
- [ ] Navigate to `/unsubscribe?token=<token>`
- [ ] Should show "Successfully Unsubscribed"
- [ ] Check `beta_users.status` = 'cancelled'
- [ ] Check `unsubscribe_tokens.used_at` is set
- [ ] Try using same token again → Should still show success (idempotent)
- [ ] Try invalid token → Should show 404

### Phase 5 Testing (Email)
- [ ] Send test Email #3
- [ ] Check top disclaimer banner appears
- [ ] Check footer has legal disclaimer box
- [ ] Check all footer links work (full URLs)
- [ ] Click unsubscribe link → Should work

---

## RISK ASSESSMENT

### Low Risk ✅
- Phase 1: Landing page swap (old template still exists as backup)
- Phase 2: Adding new routes (doesn't affect existing functionality)

### Medium Risk ⚠️
- Phase 3: Database schema changes (use `IF NOT EXISTS`, safe)
- Phase 4: Unsubscribe functionality (new feature, isolated)

### High Risk ⚠️⚠️
- Phase 5: Email template refactor (touches core email generation)
  - **Recommendation:** Test thoroughly in staging
  - **Fallback:** Keep inline HTML as backup

---

## QUESTIONS FOR REVIEW

### 1. Email Contact
You mentioned using `stockdigest.research@gmail.com`. I've already updated all templates to use this.
- ✅ Confirmed: All 4 templates now use `stockdigest.research@gmail.com`

### 2. Unsubscribe Tokens
Currently no unsubscribe system exists. Options:
- **Option A:** Token-based (recommended - more secure, better UX)
- **Option B:** Email-based parameter (simpler but less secure)

**Recommendation:** Implement token-based system as outlined in Phase 4.

### 3. Email Template Strategy
The email template refactor is complex. Which approach?
- **Option A:** Full Jinja2 refactor (clean, maintainable)
- **Option B:** Minimal changes to inline HTML (faster, less risky)

**Recommendation:** Start with Option B for quick deployment, refactor to Option A later.

### 4. Terms Acceptance Logging
Should we:
- ✅ Log timestamp when user checks Terms box during signup
- ❓ Store which version of Terms they accepted (for legal protection)

**Recommendation:** At minimum log timestamp. Versioning can come later.

### 5. SMTP Configuration
Current SMTP uses environment variables. For unsubscribe emails:
- Should we send "You've been unsubscribed" confirmation email?
- Or just show web page confirmation?

**Recommendation:** Web page only (simpler, no SMTP dependency for unsubscribe).

### 6. Jurisdiction
I've set:
- Terms: `Province of Ontario, Canada`
- Privacy: `Canada` with PIPEDA note added

**Confirm:** Is this correct for your legal jurisdiction?

---

## NEXT STEPS

Ready to proceed with implementation? I recommend this order:

1. **Start with Phase 1 + 2** (Landing page + Legal routes) - Low risk, high visibility
2. **Then Phase 3 + 4** (Database + Unsubscribe) - Medium risk, important for compliance
3. **Finally Phase 5** (Email template) - High risk, test thoroughly

Let me know if you want to:
- Proceed with implementation in this order
- Discuss any of the questions above
- Modify the approach for any phase

I'm ready to code when you give the signal! 🚀
