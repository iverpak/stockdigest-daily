# StockDigest Legal Integration - Implementation Plan

**Date:** October 7, 2025
**Approach:** Full Jinja2 Refactor + Token-Based Unsubscribe + Terms Versioning
**Risk Level:** Medium (thorough testing required)

---

## PHASE 1: UPDATE SIGNUP PAGE (15 min)

### Task 1.1: Overwrite signup.html
**File:** `/templates/signup.html`
**Action:** Replace content with `landing_page.html`

**Why:** They're the same page, just landing_page.html has:
- Top disclaimer banner
- Terms/Privacy checkbox (required)
- Footer links to legal pages

**Testing:**
- [ ] Navigate to `/` → New disclaimers visible
- [ ] Try submit without checking Terms → Blocked
- [ ] Check Terms box → Submission works

---

## PHASE 2: ADD LEGAL PAGE ROUTES (15 min)

### Task 2.1: Add Terms/Privacy Routes
**File:** `app.py` (add after line 12925)

**Code:**
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

**Testing:**
- [ ] Navigate to `/terms-of-service` → Page loads
- [ ] Navigate to `/privacy-policy` → Page loads
- [ ] All footer links work across all 3 pages

---

## PHASE 3: DATABASE SCHEMA UPDATES (30 min)

### Task 3.1: Add Terms Versioning to beta_users
**File:** `app.py` (update `ensure_schema()` around line 1049)

**SQL to add:**
```sql
-- Add terms versioning columns
ALTER TABLE beta_users
ADD COLUMN IF NOT EXISTS terms_version VARCHAR(10) DEFAULT '1.0',
ADD COLUMN IF NOT EXISTS terms_accepted_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS privacy_version VARCHAR(10) DEFAULT '1.0',
ADD COLUMN IF NOT EXISTS privacy_accepted_at TIMESTAMPTZ;

-- Add index for version tracking
CREATE INDEX IF NOT EXISTS idx_beta_users_terms_version ON beta_users(terms_version);
```

### Task 3.2: Create Unsubscribe Tokens Table
**File:** `app.py` (add to `ensure_schema()`)

**SQL:**
```sql
-- Unsubscribe tokens table
CREATE TABLE IF NOT EXISTS unsubscribe_tokens (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    token VARCHAR(64) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    used_at TIMESTAMPTZ,
    ip_address VARCHAR(45),  -- For security tracking (optional)
    user_agent TEXT,         -- For security tracking (optional)
    FOREIGN KEY (user_email) REFERENCES beta_users(email) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_unsubscribe_tokens_token ON unsubscribe_tokens(token);
CREATE INDEX IF NOT EXISTS idx_unsubscribe_tokens_email ON unsubscribe_tokens(user_email);
CREATE INDEX IF NOT EXISTS idx_unsubscribe_tokens_used ON unsubscribe_tokens(used_at) WHERE used_at IS NULL;
```

**Testing:**
- [ ] Run app → Schema migrations execute
- [ ] Check database: `\d beta_users` shows new columns
- [ ] Check database: `\d unsubscribe_tokens` exists

---

## PHASE 4: UPDATE BETA SIGNUP HANDLER (30 min)

### Task 4.1: Add Constants for Current Versions
**File:** `app.py` (add near top with other constants around line 300)

**Code:**
```python
# Legal document versions
TERMS_VERSION = "1.0"
PRIVACY_VERSION = "1.0"
TERMS_LAST_UPDATED = "October 7, 2025"
PRIVACY_LAST_UPDATED = "October 7, 2025"
```

### Task 4.2: Add Token Generation Function
**File:** `app.py` (add before `/api/beta-signup` endpoint, around line 13000)

**Code:**
```python
import secrets

def generate_unsubscribe_token(email: str) -> str:
    """
    Generate cryptographically secure unsubscribe token for user.
    Returns: 43-character URL-safe token
    """
    token = secrets.token_urlsafe(32)  # 32 bytes = 43 chars base64

    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO unsubscribe_tokens (user_email, token)
                VALUES (%s, %s)
                ON CONFLICT (token) DO NOTHING
                RETURNING token
            """, (email, token))
            result = cur.fetchone()

            if not result:
                # Token collision (astronomically rare), retry
                LOG.warning(f"Token collision for {email}, regenerating")
                return generate_unsubscribe_token(email)

            conn.commit()
            LOG.info(f"Generated unsubscribe token for {email}")
            return token
    except Exception as e:
        LOG.error(f"Error generating unsubscribe token: {e}")
        raise

def get_or_create_unsubscribe_token(email: str) -> str:
    """
    Get existing unsubscribe token or create new one.
    Reuses token if user hasn't unsubscribed yet.
    """
    try:
        with db() as conn, conn.cursor() as cur:
            # Check if unused token exists
            cur.execute("""
                SELECT token FROM unsubscribe_tokens
                WHERE user_email = %s AND used_at IS NULL
                ORDER BY created_at DESC LIMIT 1
            """, (email,))
            result = cur.fetchone()

            if result:
                return result['token']

            # No unused token found, generate new one
            return generate_unsubscribe_token(email)
    except Exception as e:
        LOG.error(f"Error getting unsubscribe token: {e}")
        # Fallback: return empty string (email will have broken unsubscribe link but won't crash)
        return ""
```

### Task 4.3: Update Beta Signup Endpoint
**File:** `app.py` (modify `/api/beta-signup` around line 13002)

**Find this block:**
```python
cur.execute("""
    INSERT INTO beta_users (name, email, ticker1, ticker2, ticker3, status, created_at)
    VALUES (%s, %s, %s, %s, %s, 'active', NOW())
    ON CONFLICT (email) DO UPDATE
    SET ticker1 = EXCLUDED.ticker1, ticker2 = EXCLUDED.ticker2, ticker3 = EXCLUDED.ticker3
    RETURNING id, created_at
""", (name, email, tickers[0], tickers[1], tickers[2]))
```

**Replace with:**
```python
cur.execute("""
    INSERT INTO beta_users (
        name, email, ticker1, ticker2, ticker3, status, created_at,
        terms_version, terms_accepted_at, privacy_version, privacy_accepted_at
    )
    VALUES (%s, %s, %s, %s, %s, 'active', NOW(), %s, NOW(), %s, NOW())
    ON CONFLICT (email) DO UPDATE
    SET ticker1 = EXCLUDED.ticker1,
        ticker2 = EXCLUDED.ticker2,
        ticker3 = EXCLUDED.ticker3,
        terms_version = EXCLUDED.terms_version,
        terms_accepted_at = NOW(),
        privacy_version = EXCLUDED.privacy_version,
        privacy_accepted_at = NOW()
    RETURNING id, created_at
""", (name, email, tickers[0], tickers[1], tickers[2], TERMS_VERSION, PRIVACY_VERSION))
```

**Add after user insertion:**
```python
conn.commit()

# Generate unsubscribe token
try:
    unsubscribe_token = generate_unsubscribe_token(email)
    LOG.info(f"Created unsubscribe token for {email}")
except Exception as e:
    LOG.error(f"Failed to create unsubscribe token for {email}: {e}")
    # Don't block signup if token generation fails
```

**Testing:**
- [ ] Sign up new user
- [ ] Check DB: `terms_version = '1.0'`, `terms_accepted_at` set
- [ ] Check DB: `unsubscribe_tokens` has entry for user
- [ ] Re-signup same email → Tokens not duplicated

---

## PHASE 5: IMPLEMENT UNSUBSCRIBE ENDPOINT (45 min)

### Task 5.1: Add Unsubscribe Page Route
**File:** `app.py` (add after legal page routes, around line 12940)

**Code:**
```python
@APP.get("/unsubscribe", response_class=HTMLResponse)
async def unsubscribe_page(request: Request, token: str = Query(...)):
    """
    Handle unsubscribe requests via token link.
    Idempotent: Can be called multiple times safely.
    """
    LOG.info(f"Unsubscribe request with token: {token[:10]}...")

    try:
        with db() as conn, conn.cursor() as cur:
            # Validate token and get user info
            cur.execute("""
                SELECT ut.user_email, ut.used_at, bu.name, bu.status
                FROM unsubscribe_tokens ut
                JOIN beta_users bu ON ut.user_email = bu.email
                WHERE ut.token = %s
            """, (token,))
            result = cur.fetchone()

            if not result:
                LOG.warning(f"Invalid unsubscribe token: {token[:10]}...")
                return HTMLResponse("""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Invalid Link - StockDigest</title>
                        <style>
                            body {
                                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                                background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
                                min-height: 100vh;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                margin: 0;
                                padding: 20px;
                            }
                            .container {
                                background: white;
                                padding: 40px;
                                border-radius: 12px;
                                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                                max-width: 500px;
                                text-align: center;
                            }
                            h1 { color: #dc2626; margin-bottom: 16px; }
                            p { color: #374151; line-height: 1.6; }
                            a { color: #1e40af; text-decoration: none; }
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>❌ Invalid Unsubscribe Link</h1>
                            <p>This unsubscribe link is invalid or has expired.</p>
                            <p>If you need assistance, please contact us at <a href="mailto:stockdigest.research@gmail.com">stockdigest.research@gmail.com</a></p>
                            <p style="margin-top: 24px;"><a href="/">← Return to Home</a></p>
                        </div>
                    </body>
                    </html>
                """, status_code=404)

            email = result['user_email']
            name = result['name']
            already_cancelled = result['status'] == 'cancelled'
            already_used = result['used_at'] is not None

            # Capture request metadata for security tracking
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get('user-agent', '')

            if not already_cancelled:
                # Mark user as unsubscribed
                cur.execute("""
                    UPDATE beta_users
                    SET status = 'cancelled'
                    WHERE email = %s
                """, (email,))
                LOG.info(f"Unsubscribed user: {email}")

            if not already_used:
                # Mark token as used
                cur.execute("""
                    UPDATE unsubscribe_tokens
                    SET used_at = NOW(), ip_address = %s, user_agent = %s
                    WHERE token = %s
                """, (ip_address, user_agent, token))
                LOG.info(f"Marked token as used for {email}")

            conn.commit()

            # Return success page
            return HTMLResponse(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Unsubscribed - StockDigest</title>
                    <style>
                        body {{
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
                            min-height: 100vh;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin: 0;
                            padding: 20px;
                        }}
                        .container {{
                            background: white;
                            padding: 40px;
                            border-radius: 12px;
                            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                            max-width: 500px;
                            text-align: center;
                        }}
                        h1 {{ color: #059669; margin-bottom: 16px; }}
                        p {{ color: #374151; line-height: 1.6; margin-bottom: 12px; }}
                        .email {{ background: #f3f4f6; padding: 8px 12px; border-radius: 4px; font-family: monospace; }}
                        a {{ color: #1e40af; text-decoration: none; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>✅ Successfully Unsubscribed</h1>
                        <p><strong>{name}</strong>, you've been unsubscribed from StockDigest.</p>
                        <p class="email">{email}</p>
                        <p style="margin-top: 24px;">You will no longer receive daily stock intelligence reports.</p>
                        <p style="margin-top: 16px; font-size: 14px; color: #6b7280;">
                            Changed your mind? <a href="/">Re-subscribe here</a>
                        </p>
                        <p style="margin-top: 24px; font-size: 14px; color: #6b7280;">
                            Questions? <a href="mailto:stockdigest.research@gmail.com">Contact us</a>
                        </p>
                    </div>
                </body>
                </html>
            """)

    except Exception as e:
        LOG.error(f"Error processing unsubscribe: {e}")
        LOG.error(traceback.format_exc())

        return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head><title>Error - StockDigest</title></head>
            <body>
                <h1>Something went wrong</h1>
                <p>We couldn't process your unsubscribe request. Please try again or contact support.</p>
                <p><a href="mailto:stockdigest.research@gmail.com">stockdigest.research@gmail.com</a></p>
            </body>
            </html>
        """, status_code=500)
```

**Testing:**
- [ ] Get token from DB: `SELECT token FROM unsubscribe_tokens LIMIT 1;`
- [ ] Navigate to `/unsubscribe?token=<TOKEN>` → Success page
- [ ] Check DB: `beta_users.status = 'cancelled'`
- [ ] Check DB: `unsubscribe_tokens.used_at` is set
- [ ] Try same URL again → Still shows success (idempotent)
- [ ] Try invalid token → Shows error page

---

## PHASE 6: REFACTOR EMAIL #3 TO JINJA2 (60 min)

### Task 6.1: Update send_user_intelligence_report() Function
**File:** `app.py` (modify function at line 11744)

**Strategy:**
1. Keep all data fetching logic (ticker, prices, articles, executive summary)
2. Replace inline HTML generation with Jinja2 template rendering
3. Build helper functions for summary sections and article lists
4. Pass all data to template as context variables

**Code Changes:**

**Step 1: Add helper function for building executive summary HTML**
```python
def build_executive_summary_html(sections: Dict[str, List[str]]) -> str:
    """
    Convert executive summary sections dict into HTML string.
    Used by Jinja2 template.
    """
    def build_section(title: str, bullets: List[str]) -> str:
        if not bullets:
            return ""

        bullet_html = ""
        for bullet in bullets:
            bullet_html += f'<li style="margin-bottom: 8px; font-size: 13px; line-height: 1.5; color: #374151;">{bullet}</li>'

        return f'''
            <div style="margin-bottom: 20px;">
                <h2 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 700; color: #1e40af; text-transform: uppercase; letter-spacing: 0.5px;">{title}</h2>
                <ul style="margin: 0; padding-left: 20px; list-style-type: disc;">
                    {bullet_html}
                </ul>
            </div>
        '''

    html = ""
    html += build_section("Major Developments", sections.get("major_developments", []))
    html += build_section("Financial/Operational Performance", sections.get("financial_operational", []))
    html += build_section("Risk Factors", sections.get("risk_factors", []))
    html += build_section("Wall Street Sentiment", sections.get("wall_street", []))
    html += build_section("Competitive/Industry Dynamics", sections.get("competitive_industry", []))
    html += build_section("Upcoming Catalysts", sections.get("upcoming_catalysts", []))

    return html
```

**Step 2: Add helper function for building article HTML**
```python
def build_articles_html(articles_by_category: Dict[str, List[Dict]]) -> str:
    """
    Convert articles by category into HTML string for email template.
    """
    def build_category_section(title: str, articles: List[Dict], category: str) -> str:
        if not articles:
            return ""

        article_links = ""
        for article in articles:
            # Check if article is paywalled
            is_paywalled = is_paywall_domain(article.get('domain', ''))
            paywall_badge = ' <span style="font-size: 10px; color: #ef4444; font-weight: 600; margin-left: 4px;">PAYWALL</span>' if is_paywalled else ''

            # Check if article is new (< 24 hours)
            is_new = False
            if article.get('published_at'):
                age_hours = (datetime.now(timezone.utc) - article['published_at']).total_seconds() / 3600
                is_new = age_hours < 24
            new_badge = '🆕 ' if is_new else ''

            # Star for FLAGGED + QUALITY articles (need to check if article is starred)
            domain = article.get('domain', '')
            is_quality = domain.lower() in [
                'wsj.com', 'bloomberg.com', 'reuters.com', 'ft.com', 'barrons.com',
                'cnbc.com', 'forbes.com', 'marketwatch.com', 'seekingalpha.com'
            ]
            # Assume article is flagged if it's in the list (this function only gets flagged articles)
            star = '<span style="color: #f59e0b;">★</span> ' if is_quality else ''

            domain_name = get_or_create_formal_domain_name(domain) if domain else "Unknown Source"
            date_str = format_date_short(article['published_at']) if article.get('published_at') else "Recent"

            article_links += f'''
                <div style="padding: 6px 0; margin-bottom: 4px; border-bottom: 1px solid #e5e7eb;">
                    <a href="{article.get('resolved_url', '#')}" style="font-size: 13px; font-weight: 600; color: #1e40af; text-decoration: none; line-height: 1.4;">{star}{new_badge}{article.get('title', 'Untitled')}{paywall_badge}</a>
                    <div style="font-size: 11px; color: #6b7280; margin-top: 3px;">{domain_name} • {date_str}</div>
                </div>
            '''

        return f'''
            <div style="margin-bottom: 16px;">
                <h3 style="margin: 0 0 8px 0; font-size: 13px; font-weight: 700; color: #1e40af; text-transform: uppercase; letter-spacing: 0.75px;">{title} ({len(articles)})</h3>
                {article_links}
            </div>
        '''

    html = ""
    html += build_category_section("COMPANY", articles_by_category.get('company', []), "company")
    html += build_category_section("INDUSTRY", articles_by_category.get('industry', []), "industry")
    html += build_category_section("COMPETITORS", articles_by_category.get('competitor', []), "competitor")

    return html
```

**Step 3: Refactor send_user_intelligence_report() to use Jinja2**

Find the section that builds inline HTML (around line 11958) and replace with:

```python
    # Count paywalled articles
    paywalled_count = sum(1 for arts in articles_by_category.values()
                         for art in arts if is_paywall_domain(art.get('domain', '')))

    total_articles = sum(len(arts) for arts in articles_by_category.values())
    lookback_days = hours // 24 if hours >= 24 else 1

    # Build HTML sections
    executive_summary_html = build_executive_summary_html(sections)
    articles_html = build_articles_html(articles_by_category)

    # Get or create unsubscribe token
    # Note: We need to get the user's email from somewhere
    # For now, assume SMTP_TO contains recipient email
    # In production, you'd pass recipient email as parameter
    recipient_email = SMTP_TO  # TODO: Pass as parameter when sending to multiple users
    unsubscribe_token = get_or_create_unsubscribe_token(recipient_email)
    unsubscribe_url = f"https://stockdigest.app/unsubscribe?token={unsubscribe_token}"

    # Current date
    current_date = datetime.now().strftime("%b %d, %Y")

    # Analysis message
    analysis_message = f"Analysis based on {total_articles} publicly available articles from the past {lookback_days} days"
    if paywalled_count > 0:
        analysis_message += f" • {paywalled_count} additional paywalled sources identified"

    # Render Jinja2 template
    try:
        # Create a minimal request object for template rendering
        from starlette.datastructures import Headers
        fake_request = type('Request', (), {
            'headers': Headers({}),
            'url': type('URL', (), {'path': '/email'})()
        })()

        html = templates.TemplateResponse(
            "email_intelligence_report.html",
            {
                "request": fake_request,
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
                "lookback_days": lookback_days,
                "articles_html": articles_html,
                "unsubscribe_url": unsubscribe_url
            }
        ).body.decode('utf-8')
    except Exception as e:
        LOG.error(f"Error rendering email template: {e}")
        LOG.error(traceback.format_exc())
        return {"status": "error", "message": f"Template rendering failed: {str(e)}"}

    # Rest of email sending logic continues...
```

**Testing:**
- [ ] Trigger Email #3 generation
- [ ] Check email HTML has disclaimers in header/footer
- [ ] All template variables render correctly
- [ ] Unsubscribe link present and valid
- [ ] Executive summary sections display correctly
- [ ] Article links work and show proper badges (★, 🆕, PAYWALL)

---

## PHASE 7: UPDATE EMAIL SENDING LOGIC (15 min)

### Task 7.1: Add Recipient Email Parameter
**File:** `app.py`

The `send_user_intelligence_report()` function needs to accept recipient email as parameter so we can generate proper unsubscribe tokens.

**Update function signature:**
```python
def send_user_intelligence_report(
    hours: int = 24,
    tickers: List[str] = None,
    flagged_article_ids: List[int] = None,
    recipient_email: str = None  # NEW PARAMETER
) -> Dict:
```

**Update unsubscribe token generation:**
```python
# Get or create unsubscribe token
if recipient_email:
    unsubscribe_token = get_or_create_unsubscribe_token(recipient_email)
    unsubscribe_url = f"https://stockdigest.app/unsubscribe?token={unsubscribe_token}"
else:
    # Fallback for testing/admin emails
    unsubscribe_token = ""
    unsubscribe_url = "https://stockdigest.app/unsubscribe"
```

**Update callers:**

Find where `send_user_intelligence_report()` is called (probably in job queue or digest functions) and add recipient email:

```python
# In digest processing
user_report_result = send_user_intelligence_report(
    hours=time_window_minutes // 60,
    tickers=[ticker],
    flagged_article_ids=flagged_article_ids,
    recipient_email=user_email  # From beta_users table
)
```

---

## TESTING PLAN

### Comprehensive Test Suite

**1. Landing Page Tests**
- [ ] Navigate to `/` → New page loads with disclaimers
- [ ] Submit without Terms checkbox → Form blocked
- [ ] Submit with Terms checked → Success
- [ ] All footer links work

**2. Legal Pages Tests**
- [ ] `/terms-of-service` → Loads correctly
- [ ] `/privacy-policy` → Loads correctly
- [ ] Cross-navigation between pages works
- [ ] All contact emails are `stockdigest.research@gmail.com`
- [ ] "Back to Home" buttons work

**3. Database Tests**
```sql
-- Check beta_users schema
\d beta_users

-- Sign up new user, then check:
SELECT email, terms_version, terms_accepted_at, privacy_version, privacy_accepted_at
FROM beta_users
ORDER BY created_at DESC LIMIT 1;

-- Should show:
-- terms_version: 1.0
-- terms_accepted_at: [timestamp]
-- privacy_version: 1.0
-- privacy_accepted_at: [timestamp]

-- Check unsubscribe_tokens schema
\d unsubscribe_tokens

-- Check token created
SELECT user_email, token, created_at, used_at
FROM unsubscribe_tokens
ORDER BY created_at DESC LIMIT 1;

-- Should have token, used_at should be NULL
```

**4. Unsubscribe Tests**
```bash
# Get a token
TOKEN=$(psql $DATABASE_URL -t -c "SELECT token FROM unsubscribe_tokens WHERE used_at IS NULL LIMIT 1;" | tr -d ' ')

# Test unsubscribe
curl "https://stockdigest.app/unsubscribe?token=$TOKEN"

# Check user unsubscribed
psql $DATABASE_URL -c "SELECT email, status FROM beta_users WHERE email IN (SELECT user_email FROM unsubscribe_tokens WHERE token = '$TOKEN');"

# Should show status = 'cancelled'

# Check token marked as used
psql $DATABASE_URL -c "SELECT used_at FROM unsubscribe_tokens WHERE token = '$TOKEN';"

# Should show timestamp
```

**5. Email Template Tests**
- [ ] Trigger Email #3 generation (via admin endpoint or job queue)
- [ ] Verify email contains:
  - Top disclaimer banner
  - Company name, ticker, price
  - Executive summary sections (6 cards)
  - Article links with proper formatting
  - Footer with legal disclaimer box
  - All footer links use full URLs (`https://stockdigest.app/...`)
  - Unsubscribe link present and valid
- [ ] Click unsubscribe link → Should work

**6. Edge Case Tests**
- [ ] Re-signup with same email → Doesn't create duplicate tokens
- [ ] Invalid unsubscribe token → Shows error page
- [ ] Already used unsubscribe token → Still shows success (idempotent)
- [ ] User with status='cancelled' tries to re-signup → Allowed (reactivates)

---

## DEPLOYMENT CHECKLIST

**Pre-Deployment:**
- [ ] All code changes committed
- [ ] Environment variables set in Render:
  - `TERMS_VERSION = "1.0"`
  - `PRIVACY_VERSION = "1.0"`
- [ ] Database backup created

**Deployment:**
- [ ] Push to GitHub
- [ ] Wait for Render deployment
- [ ] Check deployment logs for schema migration success

**Post-Deployment:**
- [ ] Run test suite above
- [ ] Sign up test user
- [ ] Send test Email #3
- [ ] Test unsubscribe flow end-to-end
- [ ] Monitor logs for errors

**Rollback Plan:**
If issues occur:
- [ ] Revert to previous Git commit
- [ ] Re-deploy
- [ ] Database schema changes are additive (safe to keep)

---

## PRODUCTION NOTES

### When to Update Terms/Privacy Versions

**Minor Changes (1.0 → 1.1):**
- Typo fixes
- Clarifications
- Contact email changes
- Non-material updates

**Major Changes (1.0 → 2.0):**
- Liability changes
- Data usage changes
- Pricing changes
- Material policy changes
- Require re-acceptance from existing users

**Process:**
1. Update HTML files
2. Update "Last Updated" date
3. Increment `TERMS_VERSION` or `PRIVACY_VERSION` constant
4. New signups get new version automatically
5. For major changes: Consider email to existing users

### Security Considerations

**Unsubscribe Tokens:**
- 32-byte tokens = 256 bits of entropy = astronomically hard to guess
- One token per email (reused if unused)
- Tokens marked as "used" after unsubscribe (can't be reused)
- IP address and user agent logged for security tracking

**Rate Limiting:**
Consider adding rate limiting to unsubscribe endpoint if abuse becomes an issue.

### CASL Compliance (Canadian Anti-Spam Law)

✅ **Compliant:**
- Users must opt-in via checkbox during signup
- Terms acceptance timestamp logged
- One-click unsubscribe in every email
- Unsubscribe processed within 10 business days (instant in our case)
- Clear sender identification in emails

---

## ESTIMATED TIME

- **Phase 1:** 15 min (update signup page)
- **Phase 2:** 15 min (add legal routes)
- **Phase 3:** 30 min (database schema)
- **Phase 4:** 30 min (update signup handler)
- **Phase 5:** 45 min (unsubscribe endpoint)
- **Phase 6:** 60 min (email template refactor)
- **Phase 7:** 15 min (email sending logic)
- **Testing:** 60 min (comprehensive testing)

**Total:** ~4.5 hours

**Recommended Approach:**
- Day 1: Phases 1-4 (2 hours) + Testing
- Day 2: Phases 5-7 (2 hours) + Testing

---

## QUESTIONS / BLOCKERS

None identified. Ready to proceed! 🚀

Let me know when you want to start implementation!
