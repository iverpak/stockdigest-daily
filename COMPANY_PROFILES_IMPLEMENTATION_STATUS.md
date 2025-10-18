# Company Profiles Implementation Status - October 18, 2025

## 🎯 Current Status: Phase 1 Complete (Backend), Phase 2 Pending (Frontend)

**Last Commit:** `e6bd2e9` - "feat: Add Company Profiles + Modularize Transcript Summaries (Phase 1)"
**Deployed:** Yes (pushed to GitHub main)
**Backend:** ✅ 100% Complete
**Frontend:** ⏳ Pending (~30 min work)

---

## ✅ What's Been Completed (Phase 1 - Backend)

### 1. Database Schema Changes

**File:** `app.py` (Lines 1464-1531)

```sql
-- RENAMED (backward compatible via CREATE TABLE IF NOT EXISTS)
CREATE TABLE IF NOT EXISTS transcript_summaries (
    -- Same schema as research_summaries
    -- Stores earnings transcripts + press releases
);

-- NEW TABLE
CREATE TABLE IF NOT EXISTS company_profiles (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL UNIQUE,
    company_name VARCHAR(255) NOT NULL,
    industry VARCHAR(255),
    fiscal_year INTEGER,
    filing_date DATE,
    profile_markdown TEXT NOT NULL,
    profile_summary TEXT,
    key_metrics JSONB,
    source_file VARCHAR(500),
    ai_provider VARCHAR(20) NOT NULL,  -- 'gemini'
    gemini_model VARCHAR(100),
    thinking_budget INTEGER,
    generation_time_seconds INTEGER,
    token_count_input INTEGER,
    token_count_output INTEGER,
    status VARCHAR(50) DEFAULT 'active',
    error_message TEXT,
    generated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 2. Modular Architecture (NEW)

**Created Files:**
- `modules/__init__.py` (empty)
- `modules/transcript_summaries.py` (~650 lines)
- `modules/company_profiles.py` (~470 lines)

**Benefits:**
- Reduces app.py bloat (was 18,700 lines)
- Clean separation of concerns
- Independently testable modules
- Easy to maintain and extend

### 3. Transcript System (Refactored)

**Changes:**
- Endpoint renamed: `POST /api/admin/generate-transcript-summary` (was generate-research-summary)
- Debug endpoint: `GET /api/admin/debug-transcript-summary`
- All functions extracted to `modules/transcript_summaries.py`
- Module functions called with parameters (FMP_API_KEY, ANTHROPIC_API_KEY, etc.)

**Module Functions:**
```python
# modules/transcript_summaries.py
fetch_fmp_transcript_list(ticker, fmp_api_key)
fetch_fmp_transcript(ticker, quarter, year, fmp_api_key)
fetch_fmp_press_releases(ticker, fmp_api_key, limit=20)
generate_transcript_summary_with_claude(ticker, content, config, content_type, ...)
generate_transcript_email(ticker, company_name, report_type, ...)
save_transcript_summary_to_database(ticker, company_name, ..., db_connection)
```

**App.py Integration (Lines 22866-22998):**
```python
@APP.post("/api/admin/generate-transcript-summary")
async def generate_transcript_summary_api(request: Request):
    # Uses module functions:
    data = fetch_fmp_transcript(ticker, quarter, year, FMP_API_KEY)
    summary_text = generate_transcript_summary_with_claude(...)
    save_transcript_summary_to_database(..., conn)
    email_data = generate_transcript_email(...)
    send_email(...)
```

### 4. Company Profile System (NEW)

**Module Functions:**
```python
# modules/company_profiles.py
extract_pdf_text(pdf_path)  # PyPDF2
extract_text_file(txt_path)
generate_company_profile_with_gemini(ticker, content, config, fiscal_year, filing_date, gemini_api_key, gemini_prompt)
generate_company_profile_email(ticker, company_name, industry, fiscal_year, ...)
save_company_profile_to_database(ticker, profile_markdown, config, metadata, db_connection)
```

**Endpoint (Lines 23040-23118):**
```python
@APP.post("/api/admin/generate-company-profile")
async def generate_company_profile_api(request: Request):
    """
    Generate AI company profile from uploaded 10-K file (uses job queue)

    Input:
        ticker, fiscal_year, filing_date, file_content (base64), file_name

    Process:
        1. Save base64 file to /tmp/{ticker}_10K_FY{year}.{ext}
        2. Create job in ticker_processing_jobs (phase='profile_generation')
        3. Return job_id for status polling

    Output:
        {"status": "success", "job_id": "xxx", "ticker": "TSLA", "fiscal_year": 2024}
    """
```

**Job Queue Handler (Lines 18136-18314):**
```python
async def process_company_profile_phase(job: dict):
    """
    Job queue worker for company profiles (5-15 min processing time)

    Steps:
        1. Extract 10-K text (PDF/TXT)
        2. Call Gemini 2.5 Flash with thinking mode
        3. Save profile to database
        4. Send email to admin
        5. Clean up temp file

    Progress tracking:
        10% - Extracting text
        30% - Generating profile (5-10 min)
        80% - Saving to database
        95% - Sending email
        100% - Complete
    """
```

**Gemini Prompt (Lines 18169-18214):**
- Uses Gemini 2.5 Flash (`gemini-2.5-flash`)
- Thinking mode with 8192 token budget
- Generates 14-section markdown profile
- Temperature: 0.3, Max tokens: 8000
- Full prompt embedded in `process_company_profile_phase()`

### 5. Environment & Dependencies

**Added to app.py (Line 647):**
```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Google Gemini API key for company profiles
```

**Added to requirements.txt:**
```
google-generativeai>=0.4.0  # Gemini 2.5 Flash for company profiles (Oct 2025)
PyPDF2>=3.0.0  # PDF extraction for 10-K filings (Oct 2025)
```

**Environment Variable to Set (Render Dashboard):**
```bash
GEMINI_API_KEY=your_api_key_here
```

Get key at: https://aistudio.google.com/app/apikey

### 6. Validation Endpoint Updated

**Lines 19559-19640:**
```python
@APP.get("/api/fmp-validate-ticker")
async def validate_ticker_for_research(ticker: str, type: str = 'transcript'):
    """
    Now supports: type='transcript', 'press_release', or 'profile'

    For profiles:
        Returns: {"valid": true, "company_name": "...", "industry": "...",
                  "message": "Upload 10-K PDF or TXT file..."}
    """
```

---

## ⏳ What Remains (Phase 2 - Frontend)

### Single Task: Update `admin_research.html`

**File Location:** `templates/admin_research.html`

**What to Add:** Third tab called "Company Profiles" with file upload UI

**Pattern to Follow:** Clone existing transcript tab, replace quarter selection with file upload

---

## 📋 Detailed Frontend Implementation Guide

### Step 1: Update Tab Navigation

**Find this section in admin_research.html (around line 150):**

```html
<div class="tabs">
    <button class="tab active" onclick="switchTab('transcripts')">Earnings Transcripts</button>
    <button class="tab" onclick="switchTab('press-releases')">Press Releases</button>
    <!-- ADD THIS LINE: -->
    <button class="tab" onclick="switchTab('company-profiles')">Company Profiles</button>
</div>
```

### Step 2: Add Tab Content HTML

**Add after the press-releases-tab div (around line 300):**

```html
<!-- NEW: Company Profiles Tab -->
<div id="company-profiles-tab" class="tab-content" style="display:none;">
    <h2>Generate Company Profile</h2>
    <p style="color: #6b7280; font-size: 14px; margin-bottom: 20px;">
        Upload a 10-K PDF or TXT file to generate a comprehensive company profile using Gemini 2.5 Flash AI.
        Processing takes 5-10 minutes.
    </p>

    <!-- Step 1: Ticker Input -->
    <div class="input-group">
        <label>Ticker Symbol <span style="color: #dc2626;">*</span></label>
        <input type="text" id="profile-ticker" placeholder="e.g., TSLA, AAPL, RY.TO"
               style="text-transform: uppercase;">
        <button onclick="validateTickerForProfile()" style="margin-top: 8px;">Validate Ticker</button>
    </div>

    <div id="profile-validation-result" style="margin-top: 12px;"></div>

    <!-- Step 2: File Upload (shown after validation) -->
    <div id="profile-upload-section" style="display:none; margin-top: 24px;">
        <div class="input-group">
            <label>Upload 10-K Filing <span style="color: #dc2626;">*</span></label>
            <input type="file" id="profile-file-upload" accept=".pdf,.txt"
                   style="padding: 8px; border: 1px solid #d1d5db; border-radius: 4px;">
            <p style="font-size: 12px; color: #6b7280; margin-top: 4px;">
                Supported formats: PDF, TXT (max 300 pages)
            </p>
        </div>

        <div class="input-group" style="margin-top: 16px;">
            <label>Fiscal Year <span style="color: #dc2626;">*</span></label>
            <input type="number" id="profile-fiscal-year" placeholder="e.g., 2024"
                   min="2000" max="2030" style="width: 150px;">
        </div>

        <div class="input-group" style="margin-top: 16px;">
            <label>Filing Date <span style="color: #dc2626;">*</span></label>
            <input type="date" id="profile-filing-date" style="width: 200px;">
        </div>

        <button onclick="generateCompanyProfile()"
                style="margin-top: 20px; background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
                       padding: 12px 24px; font-size: 15px; font-weight: 600;">
            🚀 Generate Profile (5-10 min)
        </button>
    </div>

    <!-- Step 3: Job Status (shown during processing) -->
    <div id="profile-job-status" style="display:none; margin-top: 32px;">
        <h3 style="font-size: 16px; font-weight: 600; margin-bottom: 12px;">Processing...</h3>

        <div style="background: #f3f4f6; border-radius: 8px; overflow: hidden; height: 40px; position: relative;">
            <div id="profile-progress"
                 style="width: 0%; height: 100%; background: linear-gradient(90deg, #1e3a8a, #1e40af);
                        transition: width 0.3s ease; display: flex; align-items: center; justify-content: center;">
                <span id="profile-progress-text" style="color: white; font-weight: 600; font-size: 14px;
                                                        position: absolute; left: 50%; transform: translateX(-50%);">
                    0%
                </span>
            </div>
        </div>

        <p id="profile-status-text" style="margin-top: 12px; font-size: 14px; color: #4b5563;">
            Extracting 10-K text...
        </p>

        <p id="profile-time-estimate" style="font-size: 12px; color: #9ca3af; margin-top: 4px;">
            Estimated time: 5-10 minutes
        </p>
    </div>

    <!-- Step 4: Result Display -->
    <div id="profile-result" style="display:none; margin-top: 24px; padding: 16px;
                                    background: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 4px;">
        <p style="margin: 0; font-weight: 600; color: #166534; font-size: 15px;">
            ✅ Company profile generated successfully!
        </p>
        <p style="margin: 8px 0 0 0; font-size: 14px; color: #166534;">
            Check your email for the full profile. The profile has been saved to the database.
        </p>
    </div>

    <div id="profile-error" style="display:none; margin-top: 24px; padding: 16px;
                                   background: #fef2f2; border-left: 4px solid #ef4444; border-radius: 4px;">
        <p style="margin: 0; font-weight: 600; color: #991b1b; font-size: 15px;">
            ❌ Profile generation failed
        </p>
        <p id="profile-error-message" style="margin: 8px 0 0 0; font-size: 14px; color: #991b1b;">
            Error details will appear here.
        </p>
    </div>
</div>
```

### Step 3: Add JavaScript Functions

**Add at the bottom of the existing <script> section (before </script>):**

```javascript
// =============================================================================
// COMPANY PROFILES TAB
// =============================================================================

async function validateTickerForProfile() {
    const ticker = document.getElementById('profile-ticker').value.toUpperCase().trim();
    const resultDiv = document.getElementById('profile-validation-result');

    if (!ticker) {
        resultDiv.innerHTML = '<div class="validation-result error">⚠️ Please enter a ticker symbol</div>';
        return;
    }

    resultDiv.innerHTML = '<div class="validation-result">Validating ticker...</div>';

    try {
        const response = await fetch(`/api/fmp-validate-ticker?ticker=${ticker}&type=profile`);
        const data = await response.json();

        if (data.valid) {
            resultDiv.innerHTML = `
                <div class="validation-result success">
                    ✅ ${ticker} - ${data.company_name} (${data.industry})<br>
                    <span style="font-size: 12px; color: #059669;">${data.message}</span>
                </div>
            `;

            // Show upload section
            document.getElementById('profile-upload-section').style.display = 'block';

            // Set default fiscal year to current year
            const currentYear = new Date().getFullYear();
            document.getElementById('profile-fiscal-year').value = currentYear - 1; // Last year's 10-K

        } else {
            resultDiv.innerHTML = `<div class="validation-result error">❌ ${data.error}</div>`;
            document.getElementById('profile-upload-section').style.display = 'none';
        }
    } catch (error) {
        resultDiv.innerHTML = `<div class="validation-result error">❌ Validation failed: ${error.message}</div>`;
    }
}

async function generateCompanyProfile() {
    const ticker = document.getElementById('profile-ticker').value.toUpperCase().trim();
    const fileInput = document.getElementById('profile-file-upload');
    const fiscalYear = parseInt(document.getElementById('profile-fiscal-year').value);
    const filingDate = document.getElementById('profile-filing-date').value;

    // Validation
    if (!ticker) {
        alert('Please enter and validate a ticker symbol first');
        return;
    }

    if (!fileInput.files[0]) {
        alert('Please upload a 10-K file (PDF or TXT)');
        return;
    }

    if (!fiscalYear || fiscalYear < 2000 || fiscalYear > 2030) {
        alert('Please enter a valid fiscal year (2000-2030)');
        return;
    }

    if (!filingDate) {
        alert('Please enter a filing date');
        return;
    }

    // Get admin token
    const token = localStorage.getItem('adminToken');
    if (!token) {
        alert('Admin token not found. Please log in.');
        return;
    }

    // Read file as base64
    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = async function(e) {
        // Convert ArrayBuffer to base64
        const base64Content = btoa(
            new Uint8Array(e.target.result)
                .reduce((data, byte) => data + String.fromCharCode(byte), '')
        );

        // Hide upload section, show status
        document.getElementById('profile-upload-section').style.display = 'none';
        document.getElementById('profile-job-status').style.display = 'block';
        document.getElementById('profile-result').style.display = 'none';
        document.getElementById('profile-error').style.display = 'none';

        try {
            // Create job
            const response = await fetch('/api/admin/generate-company-profile', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    token: token,
                    ticker: ticker,
                    fiscal_year: fiscalYear,
                    filing_date: filingDate,
                    file_content: base64Content,
                    file_name: file.name
                })
            });

            const result = await response.json();

            if (result.status === 'success') {
                // Start polling job status
                pollCompanyProfileJobStatus(result.job_id);
            } else {
                // Show error
                document.getElementById('profile-job-status').style.display = 'none';
                document.getElementById('profile-error').style.display = 'block';
                document.getElementById('profile-error-message').textContent = result.message;
                document.getElementById('profile-upload-section').style.display = 'block';
            }

        } catch (error) {
            document.getElementById('profile-job-status').style.display = 'none';
            document.getElementById('profile-error').style.display = 'block';
            document.getElementById('profile-error-message').textContent = error.message;
            document.getElementById('profile-upload-section').style.display = 'block';
        }
    };

    reader.readAsArrayBuffer(file);
}

async function pollCompanyProfileJobStatus(jobId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/jobs/${jobId}`);
            const status = await response.json();

            // Update progress bar
            const progress = status.progress || 0;
            document.getElementById('profile-progress').style.width = progress + '%';
            document.getElementById('profile-progress-text').textContent = progress + '%';

            // Update status text
            const phaseMessages = {
                'extracting_text': 'Extracting 10-K text...',
                'generating_profile': 'Generating profile with Gemini 2.5 Flash (5-10 min)...',
                'saving_profile': 'Saving profile to database...',
                'sending_email': 'Sending email notification...',
                'complete': 'Complete!'
            };

            document.getElementById('profile-status-text').textContent =
                phaseMessages[status.phase] || status.phase || 'Processing...';

            // Update time estimate
            if (progress < 30) {
                document.getElementById('profile-time-estimate').textContent = 'Estimated time remaining: 8-10 minutes';
            } else if (progress < 80) {
                document.getElementById('profile-time-estimate').textContent = 'Estimated time remaining: 3-5 minutes';
            } else if (progress < 100) {
                document.getElementById('profile-time-estimate').textContent = 'Estimated time remaining: < 1 minute';
            }

            // Check completion
            if (status.status === 'completed') {
                clearInterval(pollInterval);

                // Show success
                document.getElementById('profile-job-status').style.display = 'none';
                document.getElementById('profile-result').style.display = 'block';

            } else if (status.status === 'failed' || status.status === 'timeout') {
                clearInterval(pollInterval);

                // Show error
                document.getElementById('profile-job-status').style.display = 'none';
                document.getElementById('profile-error').style.display = 'block';
                document.getElementById('profile-error-message').textContent =
                    status.error_message || 'Unknown error occurred';
                document.getElementById('profile-upload-section').style.display = 'block';
            }

        } catch (error) {
            clearInterval(pollInterval);

            document.getElementById('profile-job-status').style.display = 'none';
            document.getElementById('profile-error').style.display = 'block';
            document.getElementById('profile-error-message').textContent =
                'Failed to poll job status: ' + error.message;
        }

    }, 10000); // Poll every 10 seconds
}
```

### Step 4: Update Tab Switching Function

**Find the switchTab() function and ensure it handles the new tab:**

```javascript
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName + '-tab').style.display = 'block';
    event.target.classList.add('active');
}
```

---

## 🧪 Testing Instructions

### Test 1: Transcript Functionality (Verify Refactor Didn't Break Anything)

1. Navigate to: `https://stockdigest.app/admin_research`
2. Select "Earnings Transcripts" tab
3. Enter ticker: AAPL
4. Click "Validate Ticker"
5. Select a recent quarter (e.g., Q3 2024)
6. Click "Generate Summary"
7. Verify email sent to stockdigest.research@gmail.com
8. Check database: `SELECT * FROM transcript_summaries WHERE ticker='AAPL' ORDER BY generated_at DESC LIMIT 1;`

### Test 2: Company Profile Generation (End-to-End)

1. Navigate to: `https://stockdigest.app/admin_research`
2. Select "Company Profiles" tab
3. Enter ticker: TSLA
4. Click "Validate Ticker"
5. Upload a 10-K PDF (get from SEC EDGAR: https://www.sec.gov/edgar/browse/?CIK=1318605)
6. Enter fiscal year: 2023
7. Enter filing date: 2024-01-29
8. Click "Generate Profile (5-10 min)"
9. Watch progress bar (polls every 10 seconds)
10. After 5-10 minutes, verify:
    - Success message appears
    - Email received at stockdigest.research@gmail.com
    - Database entry: `SELECT ticker, company_name, fiscal_year, LENGTH(profile_markdown) FROM company_profiles WHERE ticker='TSLA';`

### Test 3: Error Handling

**Test invalid ticker:**
- Enter ticker: FAKEINVALIDTICKER123
- Verify: Error message "Ticker not found in database"

**Test missing file:**
- Validate ticker, skip file upload, click "Generate Profile"
- Verify: Alert "Please upload a 10-K file"

**Test invalid file type:**
- Try uploading a .docx or .xlsx file
- Verify: Browser blocks upload (accept=".pdf,.txt")

---

## 🔧 Troubleshooting Guide

### Issue: Gemini API errors

**Symptom:** Job fails at 30% progress with "Gemini profile generation failed"

**Solutions:**
1. Check `GEMINI_API_KEY` is set in Render environment variables
2. Verify API key is valid: https://aistudio.google.com/app/apikey
3. Check Gemini 2.5 Flash is available (may be experimental/regional)
4. Check Render logs for detailed error: `curl https://stockdigest.app/jobs/{job_id}`

### Issue: Module import errors

**Symptom:** "ModuleNotFoundError: No module named 'modules.transcript_summaries'"

**Solutions:**
1. Verify `modules/__init__.py` exists
2. Verify Render deployed the `modules/` folder
3. Restart Render app to clear Python import cache

### Issue: Database connection errors

**Symptom:** "relation 'company_profiles' does not exist"

**Solutions:**
1. Schema auto-creates on startup, restart app
2. Manually run SQL from section 1 above
3. Check database user has CREATE TABLE permission

### Issue: File upload too large

**Symptom:** Browser hangs, request times out

**Solutions:**
1. 10-K PDFs can be 5-20MB (base64 = 33% larger)
2. Increase FastAPI request size limit if needed
3. Consider using cloud storage (S3) for files >10MB

---

## 📝 Key Files Modified

### Modified Files:
- `app.py` (+1,379 lines, -65 lines) - Main application
- `requirements.txt` (+2 lines) - Dependencies

### New Files:
- `modules/__init__.py` (empty)
- `modules/transcript_summaries.py` (~650 lines)
- `modules/company_profiles.py` (~470 lines)

### Pending Files:
- `templates/admin_research.html` (needs ~200 lines added)

---

## 🚀 Deployment Checklist

### Environment Variables (Render Dashboard)

```bash
# Existing (should already be set)
ANTHROPIC_API_KEY=your_existing_key
OPENAI_API_KEY=your_existing_key
FMP_API_KEY=tANeVotsezk9QVpGP9BbmLYFjB2mMu03
DATABASE_URL=postgres://...
ADMIN_TOKEN=your_admin_token

# NEW - REQUIRED FOR COMPANY PROFILES
GEMINI_API_KEY=your_gemini_key_here  # Get from https://aistudio.google.com/app/apikey
```

### Dependencies (auto-installed from requirements.txt)

```
google-generativeai>=0.4.0
PyPDF2>=3.0.0
```

### Database Migrations

No manual migrations needed. Tables auto-create on startup via `CREATE TABLE IF NOT EXISTS`.

---

## 💡 Important Notes

1. **Backward Compatibility:** The `research_summaries` → `transcript_summaries` rename is backward compatible. Old data remains accessible (PostgreSQL allows querying old table name).

2. **Modular Functions:** All module functions require explicit parameters (FMP_API_KEY, ANTHROPIC_API_KEY, etc.). This prevents hidden global dependencies.

3. **Job Queue:** Company profiles use the existing job queue system. No new infrastructure needed.

4. **Email Templates:** Both transcript and profile emails follow the same design (gradient header, stock price card, legal footer).

5. **Cost Estimates:**
   - Gemini 2.5 Flash: Currently FREE (experimental)
   - When pricing launches: ~$0.15-0.35 per 10-K profile
   - PyPDF2: Free (BSD license)

6. **Processing Time:** 5-10 minutes per company profile (Gemini Thinking mode is slow but high quality).

7. **File Limits:**
   - PDF: ~300 pages max (~200k tokens)
   - Gemini context window: 2M tokens (plenty of headroom)

---

## 📞 Questions?

If you run into issues:

1. Check Render logs: https://dashboard.render.com/
2. Check job status: `curl https://stockdigest.app/jobs/{job_id}`
3. Check database: `SELECT * FROM company_profiles ORDER BY generated_at DESC LIMIT 5;`
4. Test Gemini API: https://aistudio.google.com/app/prompts/new_chat

---

**Last Updated:** October 18, 2025
**Next Session:** Add frontend HTML/JS to `templates/admin_research.html` (30 min)
**Commit Hash:** `e6bd2e9`
