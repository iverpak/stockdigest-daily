# 8-K JavaScript Completion Guide

## Summary

The backend for 8-K filings is **100% complete**. The frontend is **90% complete** (HTML structure added, JavaScript functions needed).

## What's Been Completed

### Backend (100%)
✅ Database: `sec_8k_filings` table created
✅ SEC Edgar scraping functions (CIK lookup, filing list, content extraction)
✅ Gemini summary generation
✅ API endpoints: validation, generation, library list, delete
✅ Job queue worker with email generation
✅ Phase routing in main worker

### Frontend HTML (100%)
✅ 8-K section added to Generate Research tab (line 286-292)
✅ 8-K option added to Research Library dropdown (line 374)
✅ FMP Press Release buttons fixed (side-by-side with flex, line 747)

## What Needs To Be Added (JavaScript Only)

Add these JavaScript functions to `templates/admin_research.html`:

### 1. Modify `loadTickerResearch()` function (around line 539)

Add this BEFORE the `try` block:

```javascript
// Fetch 8-Ks separately (SEC Edgar, not FMP)
let eightKData = null;
```

Add this AFTER line 557 (`currentTickerData = data;`):

```javascript
// Fetch 8-Ks from SEC Edgar
try {
    const eightKResponse = await fetch(`/api/sec-validate-ticker?ticker=${ticker}`);
    eightKData = await eightKResponse.json();
} catch (error) {
    console.error('Failed to fetch 8-Ks:', error);
    eight KData = { valid: false, available_8ks: [] };
}
```

### 2. Modify `displayTickerResearch()` function (around line 571)

Add this AFTER line 584:

```javascript
document.getElementById('count-8ks').textContent = (eightKData && eightKData.valid) ? eightKData.available_8ks.length : 0;
```

Add this AFTER line 590:

```javascript
populate8KList((eightKData && eightKData.valid) ? eightKData.available_8ks : [], data.ticker, eightKData?.cik);
```

### 3. Add `populate8KList()` function (after line 774)

```javascript
function populate8KList(items, ticker, cik) {
    const container = document.getElementById('list-8ks');
    if (items.length === 0) {
        container.innerHTML = '<p style="color: #6b7280;">No 8-K filings available from SEC Edgar.</p>';
        return;
    }

    container.innerHTML = items.map(item => {
        const displayTitle = item.title.length > 80 ? item.title.substring(0, 80) + '...' : item.title;

        if (item.has_summary) {
            return `
                <div style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="flex: 1;">
                            <p style="font-weight: 600; margin: 0 0 4px 0;">${item.filing_date}</p>
                            <p style="margin: 0 0 4px 0; font-size: 14px; color: #374151;">
                                ${displayTitle}
                            </p>
                            <p style="margin: 0; font-size: 13px; color: #6b7280;">
                                Items: ${item.item_codes}
                            </p>
                            <p style="margin: 4px 0 0 0; color: #059669; font-size: 13px;">✅ Summary Generated</p>
                        </div>
                        <div style="display: flex; gap: 8px; align-items: center;">
                            <a href="/admin/research?token=${token}&tab=library&type=8k_filings&ticker=${ticker}"
                               style="background: #1e40af; color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; font-size: 14px;">
                                View in Library →
                            </a>
                        </div>
                    </div>
                </div>
            `;
        } else if (item.error) {
            return `
                <div style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="flex: 1;">
                            <p style="font-weight: 600; margin: 0 0 4px 0;">${item.filing_date}</p>
                            <p style="margin: 0; color: #dc2626; font-size: 13px;">⚠️ Error: ${item.error}</p>
                        </div>
                    </div>
                </div>
            `;
        } else {
            return `
                <div style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="flex: 1;">
                            <p style="font-weight: 600; margin: 0 0 4px 0;">${item.filing_date}</p>
                            <p style="margin: 0 0 4px 0; font-size: 14px; color: #374151;">
                                ${displayTitle}
                            </p>
                            <p style="margin: 0; font-size: 13px; color: #6b7280;">
                                Items: ${item.item_codes}
                            </p>
                        </div>
                        <button onclick="generate8K('${ticker}', '${cik}', '${item.accession_number}', '${item.filing_date}', \`${item.title}\`, '${item.sec_html_url}', '${item.item_codes}')"
                                style="background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: 600;">
                            Generate Summary (5-10 min)
                        </button>
                    </div>
                </div>
            `;
        }
    }).join('');
}
```

### 4. Add `generate8K()` function (after the populate8KList function)

```javascript
async function generate8K(ticker, cik, accessionNumber, filingDate, filingTitle, secHtmlUrl, itemCodes) {
    alert(`Starting 8-K summary generation for ${ticker} (${filingDate}). This will take 5-10 minutes...`);

    try {
        const response = await fetch('/api/admin/generate-8k-summary', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                token: token,
                ticker: ticker,
                cik: cik,
                accession_number: accessionNumber,
                filing_date: filingDate,
                filing_title: filingTitle,
                sec_html_url: secHtmlUrl,
                item_codes: itemCodes
            })
        });

        const result = await response.json();

        if (result.status === 'success') {
            alert(`✅ 8-K summary job created! Job ID: ${result.job_id}\n\nProcessing will take 5-10 minutes. You'll receive an email when complete.`);
        } else {
            alert(`❌ Failed to create job: ${result.message}`);
        }
    } catch (error) {
        alert(`❌ Error: ${error.message}`);
    }
}
```

### 5. Add 8-K support to `switchResearchType()` function

Find the `switchResearchType()` function and add this case to the switch statement:

```javascript
case '8k_filings':
    headers = ['Ticker', 'Date', 'Title', 'Items', 'Model', 'Actions'];
    endpoint = '/api/admin/8k-filings';
    renderFunction = render8KFilingsTable;
    break;
```

### 6. Add `render8KFilingsTable()` function (in the Research Library section)

```javascript
function render8KFilingsTable(filings) {
    const tbody = document.getElementById('profiles-table-body');
    const searchTerm = document.getElementById('profiles-search').value.toLowerCase();

    const filtered = filings.filter(f =>
        f.ticker.toLowerCase().includes(searchTerm) ||
        (f.filing_title && f.filing_title.toLowerCase().includes(searchTerm))
    );

    tbody.innerHTML = filtered.map(filing => {
        const duration = filing.processing_duration_seconds
            ? `${(filing.processing_duration_seconds / 60).toFixed(1)}min`
            : 'N/A';

        return `
            <tr style="border-bottom: 1px solid #e5e7eb;">
                <td style="padding: 12px;">${filing.ticker}</td>
                <td style="padding: 12px;">${formatShortDate(filing.filing_date)}</td>
                <td style="padding: 12px;">${filing.filing_title.substring(0, 60)}...</td>
                <td style="padding: 12px;">${filing.item_codes}</td>
                <td style="padding: 12px;">
                    <span style="background: #3b82f6; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">
                        🟦 Gemini (${duration})
                    </span>
                </td>
                <td style="padding: 12px;">
                    <div style="display: flex; gap: 8px;">
                        <button onclick="view8KFiling('${filing.ticker}', '${filing.accession_number}')"
                                style="background: #1e40af; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer;">
                            View
                        </button>
                        <button onclick="delete8KFiling('${filing.ticker}')"
                                style="background: #dc2626; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer;">
                            Delete
                        </button>
                    </div>
                </td>
            </tr>
        `;
    }).join('');

    document.getElementById('profiles-count').textContent = `Showing ${filtered.length} of ${filings.length} filings`;
}

async function view8KFiling(ticker, accessionNumber) {
    // Fetch the specific filing
    const response = await fetch(`/api/admin/8k-filings?token=${token}`);
    const data = await response.json();

    if (data.status === 'success') {
        const filing = data.filings.find(f => f.ticker === ticker && f.accession_number === accessionNumber);
        if (filing) {
            // Show in modal
            document.getElementById('modal-title').textContent = `8-K Filing: ${ticker} - ${filing.filing_date}`;
            document.getElementById('modal-content').innerHTML = marked.parse(filing.summary_text);
            document.getElementById('profile-modal').style.display = 'block';
        }
    }
}

async function delete8KFiling(ticker) {
    if (!confirm(`Delete ALL 8-K filings for ${ticker}?`)) return;

    try {
        const response = await fetch('/api/admin/delete-8k-filing', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ token: token, ticker: ticker })
        });

        const result = await response.json();

        if (result.status === 'success') {
            alert(`✅ ${result.message}`);
            switchResearchType(); // Reload
        } else {
            alert(`❌ Error: ${result.message}`);
        }
    } catch (error) {
        alert(`❌ Error: ${error.message}`);
    }
}
```

## Implementation Instructions

1. Open `/workspaces/quantbrief-daily/templates/admin_research.html`
2. Add each JavaScript function in the locations specified above
3. Save the file
4. The 8-K feature will be fully functional

## Testing Checklist

- [ ] Enter ticker (e.g., "AAPL") → Click "Load Research Options"
- [ ] Verify 8-K section shows "10 available"
- [ ] Verify each 8-K shows: Date, Title, Items
- [ ] Click "Generate Summary" → Verify job created
- [ ] Wait 5-10 minutes → Check admin email for summary
- [ ] Go to Research Library → Select "8-K Filings"
- [ ] Verify table shows generated filings
- [ ] Click "View" → Modal displays summary
- [ ] Click "Delete" → Confirms and removes filing
