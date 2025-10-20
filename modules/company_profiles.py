# modules/company_profiles.py

"""
Company Profiles Module

Handles 10-K filing ingestion and company profile generation using Gemini 2.5 Flash AI.
"""

import google.generativeai as genai
import PyPDF2
import logging
import traceback
from typing import Dict, Optional
from datetime import datetime, timezone
import pytz

LOG = logging.getLogger(__name__)

# ==============================================================================
# GEMINI PROMPTS (10-K and 10-Q)
# ==============================================================================

GEMINI_10K_PROMPT = """You are creating a comprehensive Company Profile for {company_name} ({ticker}) from their Form 10-K filing.

EXTRACTION PHILOSOPHY:
- Extract ALL material data disclosed in the filing - err on the side of inclusion
- Use exact figures, names, dates, and terms as stated in filing
- Format tables for structured data (financials, debt schedules, segment data)
- This serves ALL industries and company sizes - extract what's disclosed
- Target: 5,000-8,000 words (comprehensive extraction)

---
COMPLETE 10-K DOCUMENT:
{full_10k_text}

---

## 0. FILING METADATA
- **Fiscal year end date:** MM/DD/YYYY (e.g., December 31, 2024 or September 30, 2024)
- **Reporting period:** Fiscal Year [YYYY]
- **Filing date:** [Date 10-K was filed with SEC]
- **Reporting currency:** USD, CAD, EUR, GBP, etc.
- **Accounting standard:** US GAAP or IFRS
- **Number of employees:** Total headcount (from Part I, Item 1)
- **Independent auditor:** Auditing firm name

## 1. INDUSTRY & BUSINESS MODEL
- Industry classification and business description
- All reportable segments: Name, revenue, profitability, purpose
- Business characteristics: B2B/B2C, recurring/transactional, regulated/merchant, asset intensity
- Geographic footprint: All countries of operation, facility locations, production sites
- Organizational structure: Subsidiaries, JVs, equity method investments
- Seasonality: Is business seasonal? Which quarters/periods are strongest?
- Employee count by geography/segment if disclosed

## 2. REVENUE MODEL
- Revenue by segment (all periods disclosed, typically 3 years)
- Revenue by geography (all periods disclosed)
- Revenue by product/service category (all periods disclosed)
- Revenue by customer type if disclosed
- Revenue recognition policy: Timing, contract terms, performance obligations
- Deferred revenue and contract assets/liabilities (amounts and trends)
- Customer concentration: All customers >10% with names and percentages
- Significant contracts: Government contracts, long-term agreements with terms

## 3. COMPLETE FINANCIAL STATEMENTS
**Income Statement (extract all periods provided, typically 3 years):**
- Revenue line items (product, service, segment breakdowns)
- Cost of revenue/COGS with components if disclosed
- Gross profit and gross margin %
- Operating expenses: R&D, SG&A, restructuring, impairments (all line items)
- **Stock-based compensation (SBC):** Total SBC expense and 3-year trend ($ and % of revenue)
- Operating income by segment if disclosed
- **EBITDA/Adjusted EBITDA:**
  - Extract from MD&A if company explicitly discloses with reconciliation table
  - If disclosed: Extract all adjustments (stock comp, restructuring, impairments, etc.) and EBITDA margin %
  - If NOT disclosed: Note "EBITDA not disclosed. Approximation: Calculate as Operating Income + Depreciation & Amortization (from cash flow statement). Note: This is an approximation and may differ from company's internal calculation."
  - This is the PRIMARY earnings metric to extract
- Interest income and interest expense (separately)
- Other income/expense items
- Equity in earnings of affiliates
- **Income tax provision:** Amount, effective tax rate, 3-year ETR trend with explanation for changes
- Net income and net income margin %
- EPS: Basic and diluted
- Shares outstanding: Basic and diluted (note any significant changes YoY)

**Balance Sheet (extract all periods, typically 2 years):**
- **Current assets:** Cash, marketable securities, receivables, inventory, prepaid, other
- **Non-current assets:**
  - PP&E: Gross PP&E, accumulated depreciation, net PP&E
  - **Operating lease right-of-use (ROU) assets** (ASC 842 disclosure)
  - Intangibles: Detail by type if disclosed (customer relationships, technology, trade names)
  - **Goodwill:** Total and by segment if disclosed (required by ASC 350)
  - Equity method investments
  - Other non-current assets
- **Current liabilities:** Payables, accrued expenses, current debt, **current operating lease liabilities**, deferred revenue, other
- **Non-current liabilities:**
  - Long-term debt (detail in Section 5)
  - **Non-current operating lease liabilities** (ASC 842)
  - Deferred taxes (detail amounts and changes)
  - Pension/OPEB liabilities
  - Other non-current liabilities
- **Equity:** Common stock, additional paid-in capital, retained earnings, accumulated OCI, treasury stock
- **Total assets, total liabilities, total equity**

**Cash Flow Statement (extract all periods, typically 3 years):**
- **Operating activities:**
  - Net income
  - Add back: Depreciation, amortization, stock-based compensation, deferred taxes
  - Working capital changes: Receivables, inventory, payables (detail each component separately)
  - Other operating items
  - **Net cash from operations**
- **Investing activities:**
  - Capital expenditures (note if broken out: maintenance vs growth capex)
  - Acquisitions (amounts and targets)
  - Investments and asset sales
  - **Net cash from investing**
- **Financing activities:**
  - Debt proceeds and repayments (separately)
  - Dividends paid
  - Stock repurchases (shares and amounts)
  - Stock issuances
  - **Net cash from financing**
- **Net change in cash**
- **Free cash flow:** If disclosed, extract. Otherwise calculate: Operating cash flow - Capital expenditures
- **FCF conversion:** FCF ÷ Net Income (calculate %)

## 4. SEGMENT PERFORMANCE (All Periods)
For each reportable segment extract:
- Revenue (3 years if available, with YoY % changes)
- Operating income/EBIT (3 years, with margins)
- **EBITDA/Adjusted EBITDA by segment** (if disclosed - note: this is RARELY disclosed per ASC 280)
- Operating/EBITDA margin % (calculate if not disclosed)
- **Assets allocated to segment** (required disclosure)
- **Goodwill by segment** (required disclosure per ASC 350)
- Capital expenditures by segment (if disclosed)
- Depreciation/amortization by segment (if disclosed)
- Key operating metrics: Volume, customers, utilization, capacity

## 5. DEBT SCHEDULE (Complete Detail)
Extract from debt footnote (typically Note on Debt or Long-Term Obligations):
- **Instrument name:** Senior Notes, Term Loan, Revolving Credit Facility, Convertible Notes, etc.
- **Principal amount outstanding** (face value)
- **Maturity date**
- **Interest rate/coupon:** Fixed rate (%), floating rate (LIBOR/SOFR + spread), or 0% for convertibles
- **Security/collateral:** Secured or unsecured, assets pledged if disclosed
- **Covenants:** Leverage ratios, interest coverage, minimum liquidity, restrictions on dividends/buybacks
- **Guarantees:** Parent guarantee, subsidiary guarantee, cross-default provisions

**Debt Maturity Schedule (extract table showing principal due by year):**
- Year 1 (within 12 months)
- Year 2
- Year 3
- Year 4
- Year 5
- Thereafter (beyond 5 years)
- Total debt outstanding

**Additional debt metrics:**
- Fair value vs carrying value if disclosed
- Weighted average interest rate on total debt
- Credit ratings if disclosed (S&P, Moody's, Fitch)
- Debt issuance and extinguishment activity during year
- Available capacity on revolving credit facilities

**Format as table:**
```
| Instrument | Principal | Maturity | Rate | Security | Covenants |
|------------|-----------|----------|------|----------|-----------|
| [Name]     | $X.XB     | MM/YYYY  | X.X% | Unsecured| Leverage <Xx |
```

## 6. OPERATIONAL METRICS (KPIs)
Extract all disclosed non-financial metrics (all periods available):
- Volume: Units sold, subscribers, members, customers, accounts
- Efficiency: Revenue per employee, revenue per unit, same-store sales, utilization %, occupancy %
- Quality: Retention %, churn %, NPS, defect rates, return rates
- Capacity: Production capacity, installed capacity, capacity utilization %
- Growth: New customer adds, organic growth %, acquired growth %
- Geographic/segment-specific KPIs
- Calculate YoY % changes for all metrics

## 7. DEPENDENCIES & CONCENTRATIONS
- **Customer concentration:** All named customers with % of revenue
- **Supplier concentration:** Critical suppliers, sole-source dependencies, named suppliers with importance
- **Raw materials:** Key inputs, pricing exposure, supply constraints
- **Geographic concentration:** % of revenue/production by country/region
- **Regulatory dependencies:** Licenses, permits, approvals required (FDA, FCC, EPA, etc.)
- **Technology dependencies:** Patents, IP, third-party licenses, royalty agreements
- **Infrastructure dependencies:** Distribution networks, platforms, utilities, logistics
- **Joint ventures:** All JV partners, ownership %, purpose, financial contribution

## 8. RISK FACTORS (Comprehensive)
Extract ALL risks from Item 1A Risk Factors section:
- Organize by category: Operational, Financial, Regulatory, Competitive, Strategic, Technology, Legal, Macroeconomic
- For each risk: Describe threat, potential financial impact if quantified, disclosed mitigation
- Flag risks with specific dollar amounts, percentages, or timelines
- Note any risks that materialized during the fiscal year
- Include litigation risks with case names and potential loss amounts
- Identify top 5 most material risks based on disclosure length and specificity

## 9. PROPERTIES & FACILITIES
Extract from Item 2 Properties:
- All manufacturing/production facilities: Location, purpose, owned vs leased, square footage
- Distribution centers/warehouses: Locations, owned vs leased
- Office locations: Corporate HQ, regional offices, R&D centers
- Retail locations if applicable: Store count by geography
- Total square footage owned vs leased
- **Operating leases:** Total lease expense, weighted average remaining lease term, discount rate (ASC 842 disclosures)
- Significant leases: Terms, expiration dates, renewal options

## 10. LEGAL PROCEEDINGS
Extract from Item 3 Legal Proceedings:
- All material litigation: Case name, court, filing date, allegations
- Potential loss amounts or ranges if disclosed
- Status: Pending, settled, judgment entered
- Regulatory investigations or enforcement actions
- Environmental liabilities and remediation costs
- Intellectual property disputes

## 11. MANAGEMENT & GOVERNANCE
- **Executive officers:** Names, titles, ages, tenure (from Part I, Item 1 or Part III, Item 10)
- **Board of directors:** Names, independence status, committee assignments if disclosed
- **Note:** Detailed executive compensation (salary, bonus, equity grants) is in the DEF 14A proxy statement, not the 10-K
- Share ownership: Officers and directors beneficial ownership %
- Related party transactions if disclosed

## 12. CAPITAL ALLOCATION (3-Year History)
- **Capital expenditures by category:** Maintenance vs growth (if disclosed), absolute $ and % of revenue
- **R&D spending:** Absolute $ and % of revenue, 3-year trend
- **R&D capitalization policy:** Does company expense R&D immediately or capitalize software/development costs? (per ASC 985-20 for software)
- **Acquisitions:** Target names, purchase price, rationale, goodwill recognized
- **Dividends:** Quarterly rate, annual total, payout ratio (dividends ÷ net income), dividend policy and restrictions
- **Share repurchases:** Authorization amounts, shares repurchased, average price, remaining authorization, treasury stock method
- **Debt activity:** Issuances, repayments, refinancings with amounts and terms
- **Management's stated capital allocation priorities** (from MD&A)

## 13. STRATEGIC PRIORITIES, OUTLOOK & GUIDANCE
- **Strategic initiatives:** Transformation programs, cost reduction, growth investments (from Item 1 Business or MD&A)
- **Any disclosed financial guidance or targets:**
  - Note: Formal guidance is more common in earnings releases, but extract if mentioned in 10-K
  - Revenue targets, EBITDA/margin targets, EPS goals, ROIC objectives
  - Timeline for achieving targets
- **Segment-specific strategies and priorities**
- **Management's view on industry trends and competitive position**
- **Capital allocation priorities going forward**
- **Investment thesis or value creation plan** if articulated by management

## 14. SUBSEQUENT EVENTS
Extract from Subsequent Events footnote (typically last note in financial statements):
- All events occurring between fiscal year-end and 10-K filing date
- Material transactions: Acquisitions, divestitures, debt activity
- Legal developments: Settlements, judgments, new litigation
- Operational changes: Facility closures, restructuring announcements
- Any events affecting comparability to prior periods
- Note: Can be 2-3 months of developments between year-end and filing

## 15. KEY MONITORING VARIABLES
Based on business model, risks, and strategy, synthesize:
- **Critical financial metrics to track:** EBITDA margins, FCF conversion, leverage ratios, ROIC, working capital trends
- **Operational KPIs signaling execution:** Volume growth, retention, capacity utilization, same-store sales
- **Risk factors most likely to materialize:** Which risks have highest probability and impact?
- **Strategic milestones with timelines:** Product launches, facility openings, debt maturities
- **Upcoming catalysts:** Contract renewals, regulatory decisions, patent expirations, competitive product launches
- **Red flags to watch:** Covenant cushion shrinking, customer concentration increasing, margin compression

OUTPUT FORMAT:
- Valid Markdown with ## headers for each section (0-15)
- Use tables extensively for financial data, debt schedules, segment data, multi-year comparisons
- Include ALL disclosed figures with units (%, $M, $B, units, headcount, square footage)
- Preserve exact terminology from filing (e.g., "Adjusted EBITDA" if that's what company calls it)
- Write "Not disclosed in 10-K" for sections with no data (but search thoroughly first)
- Calculate percentage changes (YoY %) for all financial and operational metrics
- Comprehensive extraction: 5,000-8,000 words

Generate the complete company profile now.
"""

GEMINI_10Q_PROMPT = """You are creating a comprehensive Quarterly Update for {company_name} ({ticker}) from their Form 10-Q filing for Q{quarter} {fiscal_year}.

EXTRACTION PHILOSOPHY:
- Extract ALL material data disclosed in the 10-Q - err on the side of inclusion
- Focus on year-over-year (YoY) comparisons: Current Q vs YoY Prior Q, YTD vs YTD Prior
- Note: Quarter-over-quarter (QoQ) data is often NOT disclosed in 10-Qs - extract if available but don't assume it exists
- Use exact figures, names, dates as stated in filing
- Format tables showing: Current Q | YoY Prior Q | YoY % Change | YTD Current | YTD Prior | YTD % Change
- This serves ALL industries and company sizes - extract what's disclosed
- Target: 3,000-5,000 words (comprehensive quarterly extraction)

---
COMPLETE 10-Q DOCUMENT:
{full_10q_text}

---

## 0. FILING METADATA
- **Fiscal quarter:** Q{quarter} {fiscal_year}
- **Period covered:** [Start date] to [End date] (e.g., July 1, 2024 to September 30, 2024)
- **Number of days in quarter:** [90/91/92/93 days] (important for daily rate calculations)
- **Filing date:** [Date 10-Q was filed with SEC]
- **Fiscal year end:** [Company's fiscal year end date - e.g., December 31 or September 30]
- **Reporting currency:** USD, CAD, EUR, etc.
- **Seasonality note:** Is this business seasonal? Which quarters are typically strongest/weakest?

## 1. QUARTERLY FINANCIAL PERFORMANCE
**Income Statement (extract all periods disclosed: Current Q, YoY Prior Q, YTD Current, YTD Prior):**
- Revenue line items (product, service, segment breakdowns)
- Cost of revenue/COGS
- Gross profit and gross margin %
- Operating expenses: R&D, SG&A, restructuring, impairments (all line items)
- **Stock-based compensation (SBC):** Total SBC expense, trend vs prior year
- Operating income and operating margin %
- **EBITDA/Adjusted EBITDA:**
  - Extract from MD&A if company explicitly discloses
  - If disclosed: Extract all adjustments and EBITDA margin %
  - If NOT disclosed: Note "EBITDA not disclosed. Approximation: Calculate as Operating Income + Depreciation & Amortization. This is an approximation."
  - Calculate YoY % changes for EBITDA and margin
  - This is the PRIMARY earnings metric to track
- Interest income and interest expense (separately)
- Other income/expense items
- Income tax provision and effective tax rate (compare to YoY prior Q)
- Net income and net income margin %
- EPS: Basic and diluted
- **Share count:** Basic and diluted shares outstanding
  - **Significant share count changes:** Note if shares increased >5% (dilution from equity raise, options) or decreased >5% (buybacks)
  - Calculate impact on EPS comparability

**Format as table:**
```
| Metric | Current Q | YoY Q | YoY % | YTD Current | YTD Prior | YTD % |
|--------|-----------|-------|-------|-------------|-----------|-------|
| Revenue | $X.XB    | $X.XB | +X%   | $X.XB       | $X.XB     | +X%   |
```

## 2. SEGMENT PERFORMANCE (YoY Trends)
For each reportable segment extract:
- **Revenue:** Current Q | YoY Prior Q | YoY % Change | YTD Current | YTD Prior | YTD % Change
- **Operating income/EBIT:** Same format as revenue
- **EBITDA by segment:** If disclosed (note: RARELY disclosed - extract only if explicitly shown)
- **Operating/EBITDA margin:** Current Q vs YoY Q (note basis point changes)
- **Volume/operational metrics:** Units, customers, subscribers (with YoY % changes)
- **Trend assessment:** Is segment accelerating, decelerating, or stable? (compare Q YoY % vs YTD YoY %)

## 3. BALANCE SHEET (Quarter-End vs Prior Quarter-End vs Year-End)
**Assets:**
- **Current assets:** Cash, marketable securities, receivables, inventory, prepaid
- **Non-current assets:** PP&E (net), operating lease ROU assets, intangibles, goodwill, investments

**Liabilities:**
- **Current liabilities:** Payables, accrued expenses, current debt, current lease liabilities, deferred revenue
- **Non-current liabilities:** Long-term debt, non-current lease liabilities, deferred taxes, other

**Equity:**
- Common stock, APIC, retained earnings, accumulated OCI, treasury stock
- Book value per share

**Format as table:**
```
| Item | Current Q | Prior Q | Year-End | QoQ Change | YE Change |
|------|-----------|---------|----------|------------|-----------|
| Cash | $X.XB     | $X.XB   | $X.XB    | +$XXM      | -$XXM     |
```

**Working Capital Analysis:**
- Calculate days metrics: Days inventory outstanding (DIO), days sales outstanding (DSO), days payables outstanding (DPO)
- Compare to YoY prior quarter
- Note any significant working capital build or release

## 4. DEBT SCHEDULE UPDATE (Complete Detail)
Extract from debt footnote (typically Note on Debt):
- **All outstanding debt instruments:**
  - Instrument name
  - Principal amount outstanding (current quarter-end)
  - Change from prior quarter: Issuances, repayments (amounts)
  - Maturity date
  - Interest rate/coupon (fixed or floating)
  - Security/collateral status

**Covenant Compliance:**
- **Critical ratios:** Leverage (Debt/EBITDA), interest coverage (EBITDA/Interest), minimum liquidity
- **Actual values vs covenant limits:** How much cushion? (e.g., "Leverage: 3.2x vs 4.0x limit = 0.8x cushion")
- Any covenant waivers or amendments during quarter?

**Debt Maturity Schedule (updated as of quarter-end):**
```
| Period | Amount Due |
|--------|------------|
| < 1 yr | $X.XB      |
| 1-2 yr | $X.XB      |
| 2-3 yr | $X.XB      |
| 3-4 yr | $X.XB      |
| 4-5 yr | $X.XB      |
| > 5 yr | $X.XB      |
| Total  | $X.XB      |
```

**Debt Activity During Quarter:**
- New debt issued: Amount, terms, use of proceeds
- Debt repaid: Amount, instrument
- Amendments or refinancings
- Credit rating changes if disclosed

## 5. CASH FLOW STATEMENT (QTD & YTD)
Extract for Current Quarter YTD and Prior Year YTD:

**Operating Activities:**
- Net income
- Add back: Depreciation, amortization, stock-based comp, deferred taxes
- **Working capital changes:** Detail each component (receivables, inventory, payables, accrued expenses)
- Other operating items
- **Net cash from operations**

**Investing Activities:**
- **Capital expenditures:** Maintenance vs growth if disclosed
- Acquisitions: Amounts and targets
- Asset sales/dispositions
- Investments in securities
- **Net cash from investing**

**Financing Activities:**
- Debt proceeds and repayments (separately)
- Dividends paid
- Stock repurchases: Shares and amounts
- Stock issuances: Employee plans, offerings
- **Net cash from financing**

**Calculate:**
- **Free cash flow (FCF):** Operating cash flow - Capex (for both QTD and YTD)
- **FCF conversion:** FCF ÷ Net Income (%)
- **YoY change in FCF:** Current YTD vs Prior YTD ($change and %change)

**Format as table showing YTD comparisons:**
```
| Item | YTD Current | YTD Prior | $ Change | % Change |
|------|-------------|-----------|----------|----------|
| OCF  | $X.XB       | $X.XB     | +$XXM    | +X%      |
| Capex| ($XXM)      | ($XXM)    | -$XXM    | -X%      |
| FCF  | $X.XB       | $X.XB     | +$XXM    | +X%      |
```

## 6. OPERATIONAL METRICS (YoY Comparisons)
Extract all disclosed non-financial KPIs:
- **Volume metrics:** Units, subscribers, customers, accounts
  - Current Q | YoY Prior Q | YoY % Change
  - YTD Current | YTD Prior | YTD % Change
- **Efficiency metrics:** Revenue per user (ARPU), revenue per employee, same-store sales, utilization %
- **Quality metrics:** Retention %, churn %, NPS
- **Capacity metrics:** Capacity utilization, occupancy rates
- **Customer metrics:** Gross adds, churn, net adds

**Analysis:** Do operational trends support or contradict financial results?

## 7. GUIDANCE & OUTLOOK (If Discussed in MD&A)
**Note:** 10-Qs often DO NOT contain formal financial guidance. Guidance is typically provided in earnings releases and calls, not in the 10-Q filing itself.

**IF guidance is discussed in MD&A, extract:**
- **Current full-year guidance:**
  - Revenue range
  - EBITDA/Adjusted EBITDA range
  - EPS range
  - Free cash flow range
  - Segment-specific guidance

- **Prior guidance** (from previous quarter):
  - Show what changed: Raised, lowered, maintained, narrowed range

- **Management commentary on outlook:**
  - What gives confidence? (improving metrics, backlog, demand signals)
  - What creates caution? (headwinds, uncertainties)
  - Key assumptions: Volume, pricing, costs, macro environment

**IF NO formal guidance in 10-Q:**
- Note: "No formal guidance provided in 10-Q. Guidance typically disclosed in earnings releases."
- Extract any qualitative outlook statements from MD&A

## 8. NEW RISKS & MATERIAL DEVELOPMENTS
Extract from Risk Factors update and MD&A:

**NEW Risks Added:**
- Any new risk factors added in this 10-Q not present in prior filings
- Why added now? What changed?

**Updated Risks:**
- Existing risks with new details or escalated language
- Risks that improved or de-escalated

**Material Developments During Quarter:**
- **Customer wins or losses:** Named customers, contract values, impacts
- **Supplier changes or disruptions:** Impact on operations or costs
- **Regulatory actions:** Investigations, approvals, fines, compliance matters
- **Litigation updates:** New lawsuits, settlements, judgments (with amounts)
- **Restructuring actions:** Headcount reductions, facility closures, severance charges
- **M&A activity:** Acquisitions, divestitures with purchase prices and rationale
- **Strategic pivots:** Program cancellations, strategy shifts
- **Technology/product developments:** Launches, delays, certifications
- **Management changes:** CEO, CFO, other C-suite departures or hires

## 9. SUBSEQUENT EVENTS (Since Quarter-End)
Extract from Subsequent Events footnote:
- All events occurring between quarter-end and 10-Q filing date (typically 30-45 days)
- Material transactions, legal developments, operational changes
- Events affecting forward comparability

## 10. MANAGEMENT COMMENTARY & TONE
Extract key statements from MD&A:

**What is management emphasizing?**
- Growth drivers highlighted
- Strategic priorities reiterated or changed
- Operational achievements celebrated

**Confident areas:**
- Where is tone optimistic?
- What metrics are improving and praised?
- What initiatives are working?

**Cautious areas:**
- What concerns or headwinds are discussed?
- What metrics are deteriorating?
- What external factors are blamed?

**Competitive environment:**
- Comments on competition, market share, win rates, pricing environment

**Macroeconomic commentary:**
- Views on interest rates, inflation, demand, supply chains, consumer spending

**Tone assessment:**
- Overall: Confident, cautious, defensive, or mixed?
- Has tone shifted vs prior quarter?

## 11. SEGMENT & GEOGRAPHIC TRENDS
Extract from segment disclosures and MD&A:
- Revenue by geography (if disclosed): YoY % changes
- Same-store sales or organic growth metrics
- **Price vs volume contribution to growth:** "Revenue grew X% driven by Y% volume and Z% pricing"
- Customer concentration: Any changes in top customer percentages?
- Product mix shifts: Which products/services growing fastest?
- Regional commentary: Which geographies outperforming or underperforming?

## 12. LIQUIDITY & CAPITAL RESOURCES
Extract from Liquidity section of MD&A:
- **Total liquidity:** Cash + marketable securities + available credit facilities
- **Liquidity bridge:** Beginning liquidity → OCF → Capex → Dividends → Buybacks → Debt activity → Ending liquidity
- **Covenant compliance:** Actual ratios vs limits, cushion to violation
- **Cash deployment plans:** Near-term use of cash (capex programs, dividends, buybacks, M&A pipeline)
- **Funding needs:** Upcoming debt maturities (next 12 months), working capital requirements
- **Credit ratings:** Any updates from S&P, Moody's, Fitch

## 13. SUMMARY OF KEY CHANGES
Synthesize quarter's developments:

**Top 3-5 Positive Developments:**
- Revenue acceleration, margin expansion, strong FCF, customer wins, successful cost actions, etc.

**Top 3-5 Negative Developments:**
- Revenue deceleration, margin compression, FCF decline, guidance cuts, customer losses, etc.

**Segment Trends:**
- Which segments are accelerating? (YoY % growth improving)
- Which are decelerating? (YoY % growth slowing)
- Which are stable?

**Risk Assessment:**
- **Risk escalations:** What got worse this quarter?
- **Risk de-escalations:** What improved?

**Strategic Shifts:**
- Any changes in capital allocation priorities?
- New strategic initiatives or pivots?

**Financial Health Assessment:**
- Is company improving, stable, or deteriorating?
- Based on: EBITDA trend, FCF trend, leverage trend, liquidity position

**Momentum Check:**
- Is quarter's YoY growth > YTD YoY growth? (accelerating)
- Is quarter's YoY growth < YTD YoY growth? (decelerating)
- Apply to revenue, EBITDA, FCF

OUTPUT FORMAT:
- Valid Markdown with ## headers for each section (0-13)
- Use comparison tables extensively showing YoY and YTD comparisons
- Calculate ALL YoY percentage changes for every metric
- Include ALL disclosed figures with units (%, $M, $B, units, days, headcount)
- Preserve exact terminology from filing
- Write "Not disclosed in 10-Q" only if truly absent
- Focus on CHANGES and TRENDS (what's different vs prior year?)
- Comprehensive extraction: 3,000-5,000 words

Generate the complete quarterly update now.
"""

GEMINI_INVESTOR_DECK_PROMPT = """You are analyzing an Investor Presentation/Deck for {company_name} ({ticker}).

Presentation Date: {presentation_date}
Deck Type: {deck_type}

APPROACH:
Analyze the deck PAGE BY PAGE in sequential order. Extract all readable content, describe visuals, and flag limitations.

CRITICAL LIMITATIONS YOU MUST ACKNOWLEDGE:
- Image-based charts/tables: You may see the visual but cannot extract precise data points
- Complex diagrams: Describe what you see, but specific values may be unreadable
- Design elements: Note emphasis (full-page treatment, large fonts) but may miss color meanings
- When data is not extractable: CLEARLY STATE THIS and flag for manual review if critical

YOUR JOB:
1. Extract all text content (100% capture expected)
2. Describe all visuals (charts, graphs, tables, images)
3. Extract data where readable (70-80% success rate expected for image-based content)
4. Flag pages where image-based data is critical but not fully extractable
5. Provide executive summary synthesizing deck's messages

---
COMPLETE INVESTOR DECK:
{full_deck_text}

---

## PAGE-BY-PAGE ANALYSIS

For each page, use this format:

### Page [#]: [Slide Title or "Untitled"]

**Content Type:** [Financial Data / Strategic Message / Operational Metrics / Market Overview / Product Info / Visual Only / Other]

**Text Content:**
[Extract ALL text from slide - titles, bullets, paragraphs, labels, footnotes]
[If no text: "No text content"]

**Data Extracted:**
[List all numbers, percentages, dates, targets, company names, metrics that are READABLE]
- Revenue: $X
- Growth rate: Y%
- Target: Z by Year
[If data is in image format and not fully readable: "Data present but in image format - extraction limited"]
[If no data: "No quantitative data"]

**Visuals Description:**
[Describe EVERYTHING you can see, even if you cannot extract precise values]

For charts:
- Type: Bar chart / Line graph / Pie chart / Waterfall / Scatter plot / Other
- What it shows: "Revenue by segment over 5 years"
- Axes: "X-axis: Years 2020-2025, Y-axis: Revenue in $M"
- Visible labels: List any data labels you CAN read
- Trend: Increasing / Decreasing / Flat / Mixed
- Data extraction status: "All values readable" OR "Values in image format - partial extraction only"

For tables:
- Dimensions: "5 columns × 10 rows"
- Headers: List column and row headers
- Content type: Financial data / Operational metrics / Comparison table
- Data extraction status: "Fully extracted" OR "Complex table in image format - recommend manual review"

For images/photos:
- Subject: What is shown (product, facility, people, concept illustration)
- Context: Any captions or labels
- Purpose: What message does this visual convey?

For diagrams/infographics:
- Type: Process flow / Organizational chart / Concept map / Timeline / Other
- Elements: Describe components and relationships
- Labels: Extract any readable text
- Extraction status: "Fully described" OR "Complex diagram - details may require manual review"

[If no visuals: "Text-only slide"]

**Key Takeaway:**
[One concise sentence: What is the main point of this slide?]

**Data Extraction Quality:**
- Text: Complete / Partial / None
- Numerical data: Complete / Partial / None / Not applicable
- Visual data: Complete / Partial / Limited by image format / Not applicable

**Manual Review Priority:** [Critical / High / Medium / Low / None]
[Critical: Key financial data or strategic information in image format that significantly impacts investment thesis]
[High: Important data in image format that adds material context]
[Medium: Supplementary data in image format]
[Low: Data is readable or not material to investment thesis]
[None: All content successfully extracted]

---

[Repeat for EVERY page in deck]

---

## EXECUTIVE SUMMARY

### 1. DECK OVERVIEW
- **Purpose:** [What is this deck for? Earnings / Deal announcement / Investor day / Conference / Strategic update]
- **Total pages:** [X pages, excluding cover/disclaimer]
- **Audience:** [Public investors / Analysts / Specific event attendees]
- **Date context:** [Quarterly earnings / Special announcement / Annual event]
- **Overall extraction success:** [X% of content fully extracted, Y pages flagged for manual review]

### 2. KEY MESSAGES (Top 5)
What are the 5 most important messages management wants investors to remember?
1. [Message with supporting data/slide references]
2. [Message with supporting data/slide references]
3. [Message with supporting data/slide references]
4. [Message with supporting data/slide references]
5. [Message with supporting data/slide references]

### 3. FINANCIAL HIGHLIGHTS
Consolidate all financial data mentioned across slides:

**Current Period Results:**
- Revenue: [Amount, growth rate, vs guidance/consensus if shown]
- EBITDA/Adjusted EBITDA: [Amount, margin %, vs guidance if shown]
- EPS: [Amount, vs consensus if shown]
- Cash flow: [Operating CF, Free CF if disclosed]
- Other key metrics: [Margins, ROIC, leverage ratios]

**Segment Performance:**
- [Segment 1]: Revenue $X, EBITDA $Y, Margin Z%
- [Segment 2]: Revenue $X, EBITDA $Y, Margin Z%
- [Continue for all segments]

**Guidance (if provided):**
- Full-year revenue: $X - $Y
- Full-year EBITDA: $X - $Y
- Full-year EPS: $X - $Y
- Other guidance: [Capex, FCF, segment-specific]
- Changes from prior guidance: [Raised / Lowered / Maintained / Narrowed]

**Balance Sheet & Liquidity:**
- Cash: $X
- Total debt: $X
- Net debt: $X
- Leverage ratio: X.Xx
- Liquidity: $X (cash + available facilities)

**Capital Allocation:**
- Dividends: $X (yield Y%)
- Buybacks: $X YTD, $Y remaining authorization
- Capex: $X
- M&A: [Any deals mentioned]

### 4. STRATEGIC PRIORITIES
What strategic initiatives did management emphasize?

**Primary Strategic Focus:** [1-2 sentence summary of overarching strategy]

**Key Initiatives:**
1. [Initiative name]: [Objective, timeline, success metrics, investment required]
2. [Initiative name]: [Objective, timeline, success metrics, investment required]
3. [Initiative name]: [Objective, timeline, success metrics, investment required]
[Continue for all disclosed initiatives]

**Capital Allocation Priorities:**
1. [Priority 1 with % of capital or $ amount]
2. [Priority 2 with % of capital or $ amount]
3. [Priority 3 with % of capital or $ amount]

### 5. FORWARD-LOOKING TARGETS
All targets, goals, and timelines mentioned:

**Near-term (Next 12 months):**
- [Target 1]: Achieve X by Quarter Y
- [Target 2]: Launch Y by Date Z
- [Target 3]: Reach Z metric by Year-end

**Medium-term (2-3 years):**
- [Target 1]: Revenue CAGR of X% through 20XX
- [Target 2]: EBITDA margin expansion to Y% by 20XX
- [Target 3]: Achieve Z milestone by 20XX

**Long-term (3-5 years):**
- [Target 1]: Market share of X% by 20XX
- [Target 2]: Return metric of Y% by 20XX
- [Target 3]: Strategic transformation complete by 20XX

### 6. OPERATIONAL METRICS & UNIT ECONOMICS
Non-financial KPIs disclosed:

**Volume/Growth Metrics:**
- [Metric 1]: Current value, growth rate, target
- [Metric 2]: Current value, growth rate, target
- [Metric 3]: Current value, growth rate, target

**Efficiency Metrics:**
- [Metric 1]: Current value, trend, benchmark
- [Metric 2]: Current value, trend, benchmark

**Unit Economics (if disclosed):**
- Customer acquisition cost (CAC): $X
- Lifetime value (LTV): $Y
- LTV/CAC ratio: Z.Zx
- Payback period: X months
- Average revenue per unit: $X
- Other: [Any disclosed unit economics]

### 7. RISKS & CHALLENGES ACKNOWLEDGED
What headwinds or challenges did management discuss?

**Disclosed Risks:**
1. [Risk 1]: [Description, potential impact, mitigation plan]
2. [Risk 2]: [Description, potential impact, mitigation plan]
3. [Risk 3]: [Description, potential impact, mitigation plan]

**Cautious Language Used:**
- [Quote or theme showing caution]
- [Quote or theme showing caution]

**Guidance Assumptions:**
- [Key assumption 1 that could change]
- [Key assumption 2 that could change]

### 8. MANAGEMENT TONE & EMPHASIS

**Overall Tone:** [Confident / Cautious / Defensive / Visionary / Mixed / Other]

**Evidence for tone assessment:**
- [Language patterns observed]
- [Topics emphasized vs downplayed]
- [Comparison to prior communications if known]

**What Got Most Airtime:**
- [Topic 1]: X pages/slides dedicated
- [Topic 2]: Y pages/slides dedicated
- [Topic 3]: Z pages/slides dedicated
[This reveals management's true priorities]

**What Got Full-Page Treatment:**
- Page X: [Topic] - signals high importance
- Page Y: [Topic] - signals high importance

**Notable Language:**
- Positive: [Words like "accelerating," "momentum," "confident," "record"]
- Cautious: [Words like "headwinds," "uncertainty," "monitoring," "challenged"]
- Action: [Words like "transforming," "investing," "pivoting," "exiting"]

### 9. NOTABLE OMISSIONS
What topics were NOT discussed that you might expect?

- [Expected topic 1]: Why notable? [Context on why absence is meaningful]
- [Expected topic 2]: Why notable? [Context on why absence is meaningful]
- [Expected topic 3]: Why notable? [Context on why absence is meaningful]

**Metrics/Initiatives Dropped:**
- [If comparing to prior deck] What metrics or initiatives present in prior communications are now absent?

### 10. DEAL-SPECIFIC DETAILS (If M&A/Partnership Deck)
[Skip this section if not applicable]

**Transaction Overview:**
- Target: [Company name and business description]
- Structure: [Acquisition / Merger / JV / Partnership / Investment]
- Valuation: [Purchase price, multiples, structure (cash/stock/earnout)]
- Funding: [Source of funds]
- Expected close: [Date and conditions]

**Strategic Rationale:**
- [Why this deal? Market expansion, product adjacency, vertical integration, other]
- [How does this strengthen competitive position?]

**Financial Impact:**
- Target financials: [Revenue, EBITDA, margins if disclosed]
- Synergies: [Cost synergies $X by Year Y, Revenue synergies $Z by Year W]
- Accretion/dilution: [EPS impact over time]
- Returns: [ROIC, IRR, payback period if disclosed]

**Integration Plan:**
- Timeline: [Key milestones]
- Leadership: [Who's running combined entity]
- Risks: [Integration risks acknowledged]

### 11. VISUAL EMPHASIS & DESIGN CUES

**Most Visually Prominent Slides:**
- Page X: [What was emphasized and how] - suggests this is highest priority message
- Page Y: [What was emphasized and how]

**Design Patterns Observed:**
- Color usage: [Green for positive, red for negative, other patterns]
- Size emphasis: [Large fonts used for specific metrics/messages]
- Repetition: [Themes that appear multiple times across slides]

**Infographics & Diagrams:**
- [Count] pages with complex visuals vs simple text/data
- [Note] any strategic frameworks, process flows, or concept illustrations

### 12. PAGES FLAGGED FOR MANUAL REVIEW

**Critical Priority (Essential for investment thesis):**
- Page X: [Reason - e.g., "Complex financial table with key segment data in image format"]
- Page Y: [Reason - e.g., "Detailed guidance assumptions in small-print table"]

**High Priority (Material context):**
- Page X: [Reason]
- Page Y: [Reason]

**Medium Priority (Supplementary data):**
- Page X: [Reason]

**Summary:** [X total pages flagged, Y are critical, Z are high priority]

### 13. COMPARISON TO PRIOR COMMUNICATIONS (If Context Available)

**Messaging Changes:**
- [What's more emphasized now vs prior quarter/year?]
- [What's less emphasized or absent?]
- [What's entirely new in messaging?]

**Guidance Changes:**
- [Compare current guidance to prior guidance]
- [Magnitude and direction of changes]
- [Management's explanation for changes]

**Metric/Target Changes:**
- [New metrics introduced]
- [Metrics no longer disclosed]
- [Targets raised, lowered, or abandoned]

### 14. INVESTMENT IMPLICATIONS

**What's Improving:**
- [Positive signal 1 with evidence]
- [Positive signal 2 with evidence]
- [Positive signal 3 with evidence]

**What's Deteriorating:**
- [Negative signal 1 with evidence]
- [Negative signal 2 with evidence]
- [Negative signal 3 with evidence]

**What's New Information:**
- [Disclosure 1 not previously available]
- [Disclosure 2 not previously available]
- [Disclosure 3 not previously available]

**Key Questions Raised:**
- [Question 1 that investors should investigate further]
- [Question 2 that investors should investigate further]
- [Question 3 that investors should investigate further]

**Overall Assessment:**
- Results vs expectations: [Beat / In-line / Miss / Mixed]
- Outlook: [Improving / Stable / Deteriorating / Uncertain]
- Management credibility: [Delivering on promises / Struggling to execute / Lowering bar]
- Investment thesis impact: [Strengthened / Neutral / Weakened]

---

## EXTRACTION QUALITY SUMMARY

**Content Successfully Extracted:**
- Text content: [X pages with complete text extraction]
- Numerical data: [Y data points extracted from Z total visible]
- Visual descriptions: [All A visuals described]

**Limitations Encountered:**
- Pages with image-based data: [X pages]
- Complex tables requiring manual review: [Y pages]
- Diagrams with limited extractability: [Z pages]

**Recommendation:**
[If flagged pages > 5% of deck]: Recommend manual review of flagged pages for complete analysis
[If flagged pages < 5% of deck]: Automated extraction captured materially complete picture

---

OUTPUT FORMAT:
- Page-by-page analysis for EVERY slide
- Consistent structure for each page
- Executive summary synthesizing all pages
- Clear flagging of extraction limitations
- Honest assessment of what was captured vs what requires manual review
- Markdown formatting with clear headers
- Length scales to deck: 10-page deck = ~2,000 words; 40-page deck = ~6,000 words

Generate the complete page-by-page deck analysis now.
"""

# ==============================================================================
# PDF/TEXT EXTRACTION
# ==============================================================================

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from PDF file using PyPDF2.
    Returns full text concatenated from all pages.
    """
    LOG.info(f"Extracting text from PDF: {pdf_path}")

    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)

            LOG.info(f"PDF has {total_pages} pages")

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += page_text

                if (i + 1) % 50 == 0:
                    LOG.info(f"Extracted {i + 1}/{total_pages} pages")

            LOG.info(f"✅ Extracted {len(text)} characters from {total_pages} pages")
            return text

    except Exception as e:
        LOG.error(f"Failed to extract PDF text: {e}")
        raise


def extract_text_file(txt_path: str) -> str:
    """Extract text from plain text file"""
    LOG.info(f"Reading text file: {txt_path}")

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        LOG.info(f"✅ Read {len(text)} characters from text file")
        return text

    except Exception as e:
        LOG.error(f"Failed to read text file: {e}")
        raise


def fetch_sec_html_text(url: str) -> str:
    """
    Fetch 10-K HTML from SEC.gov and extract plain text.

    Args:
        url: SEC.gov HTML URL (from FMP API)

    Returns:
        Plain text extracted from HTML

    Raises:
        Exception: If fetch or parsing fails
    """
    import requests
    from bs4 import BeautifulSoup

    LOG.info(f"Fetching 10-K HTML from SEC.gov: {url}")

    try:
        # SEC requires proper User-Agent to prevent blocking
        headers = {
            "User-Agent": "StockDigest/1.0 (stockdigest.research@gmail.com)"
        }

        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        LOG.info(f"✅ Fetched HTML ({len(response.text)} chars)")

        # Parse HTML and extract text
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text with newline separation
        text = soup.get_text(separator='\n', strip=True)

        # Clean up whitespace (collapse multiple spaces/tabs)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        LOG.info(f"✅ Extracted {len(text)} characters from HTML")

        return text

    except Exception as e:
        LOG.error(f"Failed to fetch SEC HTML: {e}")
        raise


# ==============================================================================
# GEMINI AI PROFILE GENERATION
# ==============================================================================

def generate_sec_filing_profile_with_gemini(
    ticker: str,
    content: str,
    config: Dict,
    filing_type: str,  # '10-K' or '10-Q'
    fiscal_year: int,
    fiscal_quarter: str = None,  # 'Q1', 'Q2', 'Q3', 'Q4' (required for 10-Q)
    filing_date: str = None,
    gemini_api_key: str = None
) -> Optional[Dict]:
    """
    Generate comprehensive SEC filing profile using Gemini 2.5 Flash.

    Supports both 10-K (annual) and 10-Q (quarterly) filings with specialized prompts.

    Args:
        ticker: Stock ticker (e.g., 'AAPL', 'TSLA')
        content: Full text of SEC filing
        config: Ticker configuration dict with company_name, etc.
        filing_type: '10-K' for annual or '10-Q' for quarterly
        fiscal_year: Fiscal year (e.g., 2024)
        fiscal_quarter: 'Q1', 'Q2', 'Q3', 'Q4' (required for 10-Q, None for 10-K)
        filing_date: Filing date string (optional)
        gemini_api_key: Gemini API key

    Returns:
        {
            'profile_markdown': str (5,000-8,000 words for 10-K, 3,000-5,000 for 10-Q),
            'metadata': {
                'model': str,
                'generation_time_seconds': int,
                'token_count_input': int,
                'token_count_output': int
            }
        }
    """
    if not gemini_api_key:
        LOG.error("Gemini API key not configured")
        return None

    if filing_type not in ['10-K', '10-Q']:
        LOG.error(f"Unsupported filing type: {filing_type}. Must be '10-K' or '10-Q'")
        return None

    if filing_type == '10-Q' and not fiscal_quarter:
        LOG.error("fiscal_quarter is required for 10-Q filings")
        return None

    try:
        genai.configure(api_key=gemini_api_key)

        company_name = config.get("company_name", ticker)

        # Select appropriate prompt based on filing type
        if filing_type == '10-K':
            prompt_template = GEMINI_10K_PROMPT
            filing_desc = f"10-K for FY{fiscal_year}"
            target_words = "5,000-8,000"
        else:  # 10-Q
            prompt_template = GEMINI_10Q_PROMPT
            filing_desc = f"10-Q for {fiscal_quarter} {fiscal_year}"
            target_words = "3,000-5,000"
            # Extract quarter number for prompt (Q3 -> 3)
            quarter_num = fiscal_quarter[1] if fiscal_quarter else "?"

        LOG.info(f"Generating {filing_type} profile for {ticker} ({filing_desc}) using Gemini 2.0 Flash Thinking")
        LOG.info(f"Content length: {len(content):,} chars (~{len(content)//4:,} tokens)")
        LOG.info(f"Target output: {target_words} words")

        # Gemini 2.0 Flash Thinking Experimental (latest stable experimental model)
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-1219')

        generation_config = {
            "temperature": 0.3,
            "max_output_tokens": 16000  # Increased from 8000 to support comprehensive extraction
        }

        start_time = datetime.now()

        # Build full prompt with content (different formatting for 10-K vs 10-Q)
        if filing_type == '10-K':
            full_prompt = prompt_template.format(
                company_name=company_name,
                ticker=ticker,
                full_10k_text=content[:200000]  # Limit to ~50k tokens to fit context window
            )
        else:  # 10-Q
            full_prompt = prompt_template.format(
                company_name=company_name,
                ticker=ticker,
                quarter=quarter_num,
                fiscal_year=fiscal_year,
                full_10q_text=content[:150000]  # 10-Qs are typically shorter
            )

        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )

        end_time = datetime.now()
        generation_time = int((end_time - start_time).total_seconds())

        profile_markdown = response.text

        if not profile_markdown or len(profile_markdown) < 1000:
            LOG.warning(f"Gemini returned suspiciously short profile for {ticker} {filing_type} ({len(profile_markdown)} chars)")
            return None

        # Extract metadata
        metadata = {
            'model': 'gemini-2.0-flash-thinking-exp-1219',
            'filing_type': filing_type,
            'generation_time_seconds': generation_time,
            'token_count_input': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
            'token_count_output': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
        }

        word_count = len(profile_markdown.split())
        LOG.info(f"✅ Generated {filing_type} profile for {ticker}")
        LOG.info(f"   Length: {len(profile_markdown):,} chars (~{word_count:,} words)")
        LOG.info(f"   Time: {generation_time}s")
        LOG.info(f"   Tokens: {metadata['token_count_input']:,} in, {metadata['token_count_output']:,} out")

        return {
            'profile_markdown': profile_markdown,
            'metadata': metadata
        }

    except Exception as e:
        LOG.error(f"Failed to generate {filing_type} profile for {ticker}: {e}")
        LOG.error(f"Stacktrace: {traceback.format_exc()}")
        return None


# Backward compatibility alias (deprecated)
def generate_company_profile_with_gemini(
    ticker: str,
    content: str,
    config: Dict,
    fiscal_year: int,
    filing_date: str,
    gemini_api_key: str,
    gemini_prompt: str = None  # Ignored (using GEMINI_10K_PROMPT instead)
) -> Optional[Dict]:
    """
    DEPRECATED: Use generate_sec_filing_profile_with_gemini() instead.

    This function maintains backward compatibility with existing code.
    """
    LOG.warning("generate_company_profile_with_gemini() is deprecated. Use generate_sec_filing_profile_with_gemini()")

    return generate_sec_filing_profile_with_gemini(
        ticker=ticker,
        content=content,
        config=config,
        filing_type='10-K',  # Default to 10-K for backward compatibility
        fiscal_year=fiscal_year,
        fiscal_quarter=None,
        filing_date=filing_date,
        gemini_api_key=gemini_api_key
    )


def generate_investor_presentation_analysis_with_gemini(
    ticker: str,
    pdf_path: str,
    config: Dict,
    presentation_date: str,
    deck_type: str,  # 'earnings', 'investor_day', 'analyst_day', 'conference'
    gemini_api_key: str
) -> Optional[Dict]:
    """
    Analyze investor presentation PDF using Gemini multimodal (vision + text).

    This function uploads a PDF to Gemini, which extracts text and visuals,
    then generates a comprehensive page-by-page analysis with executive summary.

    Args:
        ticker: Stock ticker (e.g., 'AAPL')
        pdf_path: Path to PDF file on disk
        config: Ticker configuration dict with company_name, industry
        presentation_date: Date of presentation (YYYY-MM-DD)
        deck_type: Type of presentation ('earnings', 'investor_day', 'analyst_day', 'conference')
        gemini_api_key: Gemini API key

    Returns:
        {
            'analysis_markdown': str (2,000-6,000 words depending on deck size),
            'metadata': {
                'model': str,
                'generation_time_seconds': int,
                'token_count_input': int,
                'token_count_output': int,
                'file_size_bytes': int
            }
        }
    """
    if not gemini_api_key:
        LOG.error("Gemini API key not configured")
        return None

    if not os.path.exists(pdf_path):
        LOG.error(f"PDF file not found: {pdf_path}")
        return None

    try:
        genai.configure(api_key=gemini_api_key)

        company_name = config.get("company_name", ticker)

        LOG.info(f"Analyzing investor presentation for {ticker} ({deck_type}) using Gemini multimodal")
        LOG.info(f"PDF: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

        # 1. Upload PDF to Gemini
        start_upload = datetime.now()
        uploaded_file = genai.upload_file(pdf_path)
        upload_time = (datetime.now() - start_upload).total_seconds()

        LOG.info(f"✅ Uploaded to Gemini: {uploaded_file.name} ({uploaded_file.size_bytes} bytes) in {upload_time:.1f}s")

        # 2. Wait for Gemini to process (extracts text + images internally)
        while uploaded_file.state.name == "PROCESSING":
            import time
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
            LOG.info(f"Gemini processing PDF visuals...")

        if uploaded_file.state.name == "FAILED":
            LOG.error("Gemini failed to process PDF")
            return None

        LOG.info(f"✅ Gemini processed PDF successfully")

        # 3. Build prompt
        prompt = GEMINI_INVESTOR_DECK_PROMPT.format(
            company_name=company_name,
            ticker=ticker,
            presentation_date=presentation_date,
            deck_type=deck_type,
            full_deck_text="[Gemini will extract from uploaded PDF]"
        )

        # 4. Generate analysis
        model = genai.GenerativeModel('gemini-2.5-pro')

        generation_config = {
            "temperature": 0.3,
            "max_output_tokens": 16000  # 10-page deck ~2k words, 40-page deck ~6k words
        }

        start_time = datetime.now()

        LOG.info(f"Generating comprehensive deck analysis...")

        response = model.generate_content(
            [uploaded_file, prompt],  # Pass file object + prompt for multimodal analysis
            generation_config=generation_config
        )

        end_time = datetime.now()
        generation_time = int((end_time - start_time).total_seconds())

        analysis_markdown = response.text

        if not analysis_markdown or len(analysis_markdown) < 1000:
            LOG.warning(f"Gemini returned suspiciously short analysis for {ticker} ({len(analysis_markdown)} chars)")
            return None

        # 5. Extract metadata
        word_count = len(analysis_markdown.split())
        metadata = {
            'model': 'gemini-2.5-pro',
            'generation_time_seconds': generation_time,
            'token_count_input': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
            'token_count_output': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
            'file_size_bytes': uploaded_file.size_bytes
        }

        LOG.info(f"✅ Generated deck analysis for {ticker}")
        LOG.info(f"   Length: {len(analysis_markdown):,} chars (~{word_count:,} words)")
        LOG.info(f"   Time: {generation_time}s")
        LOG.info(f"   Tokens: {metadata['token_count_input']:,} in, {metadata['token_count_output']:,} out")

        # 6. Cleanup: Delete uploaded file from Gemini
        try:
            genai.delete_file(uploaded_file.name)
            LOG.info(f"✅ Deleted temp file from Gemini: {uploaded_file.name}")
        except Exception as e:
            LOG.warning(f"Failed to delete Gemini file (non-critical): {e}")

        return {
            'analysis_markdown': analysis_markdown,
            'metadata': metadata
        }

    except Exception as e:
        LOG.error(f"Failed to analyze presentation for {ticker}: {e}")
        LOG.error(f"Stacktrace: {traceback.format_exc()}")
        return None


# ==============================================================================
# EMAIL GENERATION
# ==============================================================================

def generate_company_profile_email(
    ticker: str,
    company_name: str,
    industry: str,
    fiscal_year: Optional[int],  # Can be None for presentations
    filing_date: str,
    profile_markdown: str,
    stock_price: str = "$0.00",
    price_change_pct: str = None,
    price_change_color: str = "#4ade80"
) -> Dict[str, str]:
    """
    Generate company profile email HTML.

    Returns:
        {"html": Full email HTML, "subject": Email subject}
    """
    LOG.info(f"Generating company profile email for {ticker}")

    current_date = datetime.now(timezone.utc).astimezone(pytz.timezone('US/Eastern')).strftime("%b %d, %Y")

    # Handle fiscal_year for presentations (can be None)
    fiscal_year_display = f"FY{fiscal_year}" if fiscal_year else filing_date

    # Extract first ~2000 chars for email preview
    profile_preview = profile_markdown[:2000] + "..." if len(profile_markdown) > 2000 else profile_markdown

    # Convert markdown to simple HTML (basic conversion - just wrap in pre tag for monospace)
    # TODO: Could use a proper markdown library here for better rendering
    profile_html = f'<div style="font-family: monospace; font-size: 12px; line-height: 1.5; white-space: pre-wrap; color: #374151; background-color: #f9fafb; padding: 16px; border-radius: 4px; overflow-x: auto;">{profile_preview}</div>'

    # Build HTML (same structure as transcript email)
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} Company Profile</title>
    <style>
        @media only screen and (max-width: 600px) {{
            .content-padding {{ padding: 16px !important; }}
            .header-padding {{ padding: 16px 20px 25px 20px !important; }}
            .price-box {{ padding: 8px 10px !important; }}
            .company-name {{ font-size: 20px !important; }}
        }}
    </style>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f8f9fa; color: #212529;">

    <table role="presentation" style="width: 100%; border-collapse: collapse;">
        <tr>
            <td align="center" style="padding: 40px 20px;">

                <table role="presentation" style="max-width: 700px; width: 100%; background-color: #ffffff; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-collapse: collapse; border-radius: 8px;">

                    <!-- Header -->
                    <tr>
                        <td class="header-padding" style="padding: 18px 24px 30px 24px; background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); color: #ffffff; border-radius: 8px 8px 0 0;">
                            <table role="presentation" style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td style="width: 58%;">
                                        <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0px; opacity: 0.85; font-weight: 600; color: #ffffff;">COMPANY PROFILE</div>
                                    </td>
                                    <td align="right" style="width: 42%;">
                                        <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0px; opacity: 0.85; font-weight: 600; color: #ffffff;">Generated: {current_date} | {fiscal_year_display}</div>
                                    </td>
                                </tr>
                                <tr>
                                    <td style="width: 58%; vertical-align: top;">
                                        <h1 class="company-name" style="margin: 0; font-size: 28px; font-weight: 700; letter-spacing: -0.5px; line-height: 1; color: #ffffff;">{company_name}</h1>
                                        <div style="font-size: 13px; margin-top: 2px; opacity: 0.9; color: #ffffff;">{ticker} | {industry}</div>
                                        <div style="font-size: 11px; margin-top: 4px; opacity: 0.8; color: #ffffff;">Form 10-K Filed: {filing_date}</div>
                                    </td>
                                    <td align="right" style="width: 42%; vertical-align: top;">
                                        <div style="font-size: 28px; font-weight: 700; letter-spacing: -0.5px; line-height: 1; color: #ffffff; margin-bottom: 2px;">{stock_price}</div>
                                        {f'<div style="font-size: 13px; font-weight: 600; color: {price_change_color};">{price_change_pct}</div>' if price_change_pct else ''}
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Content -->
                    <tr>
                        <td class="content-padding" style="padding: 24px;">

                            <!-- Profile Preview -->
                            <div style="margin-bottom: 20px;">
                                <h2 style="font-size: 16px; font-weight: 700; color: #1e40af; margin: 0 0 12px 0;">Profile Preview (First 2,000 Characters)</h2>
                                {profile_html}
                            </div>

                            <!-- Full Profile Info Box -->
                            <div style="margin: 32px 0 20px 0; padding: 12px 16px; background-color: #eff6ff; border-left: 4px solid #1e40af; border-radius: 4px;">
                                <p style="margin: 0; font-size: 12px; color: #1e40af; font-weight: 600; line-height: 1.4;">
                                    Full company profile ({len(profile_markdown):,} characters) saved to database.
                                    Access via Admin Panel at https://stockdigest.app/admin
                                </p>
                            </div>

                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); padding: 16px 24px; color: rgba(255,255,255,0.9); border-radius: 0 0 8px 8px;">
                            <table role="presentation" style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td>
                                        <div style="font-size: 14px; font-weight: 600; color: #ffffff; margin-bottom: 4px;">StockDigest Research Tools</div>
                                        <div style="font-size: 12px; opacity: 0.8; margin-bottom: 8px; color: #ffffff;">Company Profile Analysis</div>

                                        <!-- Legal Disclaimer -->
                                        <div style="font-size: 10px; opacity: 0.7; line-height: 1.4; margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.2); color: #ffffff;">
                                            For informational and educational purposes only. Not investment advice. See Terms of Service for full disclaimer.
                                        </div>

                                        <!-- Links -->
                                        <div style="font-size: 11px; margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.2);">
                                            <a href="https://stockdigest.app/terms-of-service" style="color: #ffffff; text-decoration: none; opacity: 0.9; margin-right: 12px;">Terms of Service</a>
                                            <span style="color: rgba(255,255,255,0.5); margin-right: 12px;">|</span>
                                            <a href="https://stockdigest.app/privacy-policy" style="color: #ffffff; text-decoration: none; opacity: 0.9; margin-right: 12px;">Privacy Policy</a>
                                            <span style="color: rgba(255,255,255,0.5); margin-right: 12px;">|</span>
                                            <a href="mailto:stockdigest.research@gmail.com" style="color: #ffffff; text-decoration: none; opacity: 0.9;">Contact</a>
                                        </div>

                                        <!-- Copyright -->
                                        <div style="font-size: 10px; opacity: 0.6; margin-top: 12px; color: #ffffff;">
                                            © 2025 StockDigest. All rights reserved.
                                        </div>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                </table>

            </td>
        </tr>
    </table>

</body>
</html>'''

    subject = f"📋 Company Profile: {company_name} ({ticker}) {fiscal_year_display}"

    return {"html": html, "subject": subject}


# ==============================================================================
# DATABASE OPERATIONS
# ==============================================================================

def save_company_profile_to_database(
    ticker: str,
    profile_markdown: str,
    config: Dict,
    metadata: Dict,
    db_connection
) -> None:
    """Save company profile to unified sec_filings table"""
    LOG.info(f"Saving company profile for {ticker} to sec_filings table")

    try:
        cur = db_connection.cursor()

        # Determine source type
        source_file = config.get('source_file', '')
        source_type = 'fmp_sec' if 'SEC.gov' in source_file else 'file_upload'

        cur.execute("""
            INSERT INTO sec_filings (
                ticker, filing_type, fiscal_year, fiscal_quarter,
                company_name, industry, filing_date,
                profile_markdown, source_file, source_type, sec_html_url,
                ai_provider, ai_model,
                generation_time_seconds, token_count_input, token_count_output,
                status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, filing_type, fiscal_year, fiscal_quarter) DO UPDATE SET
                company_name = EXCLUDED.company_name,
                industry = EXCLUDED.industry,
                filing_date = EXCLUDED.filing_date,
                profile_markdown = EXCLUDED.profile_markdown,
                source_file = EXCLUDED.source_file,
                source_type = EXCLUDED.source_type,
                sec_html_url = EXCLUDED.sec_html_url,
                ai_provider = EXCLUDED.ai_provider,
                ai_model = EXCLUDED.ai_model,
                generation_time_seconds = EXCLUDED.generation_time_seconds,
                token_count_input = EXCLUDED.token_count_input,
                token_count_output = EXCLUDED.token_count_output,
                generated_at = NOW(),
                status = 'active',
                error_message = NULL
        """, (
            ticker,
            '10-K',                              # filing_type (always 10-K for now)
            config.get('fiscal_year'),           # fiscal_year
            None,                                # fiscal_quarter (NULL for 10-K)
            config.get('company_name'),
            config.get('industry'),
            config.get('filing_date'),
            profile_markdown,
            source_file,
            source_type,
            config.get('sec_html_url'),         # SEC.gov HTML URL (if available)
            'gemini',
            metadata.get('model'),               # ai_model (e.g., 'gemini-2.5-flash')
            metadata.get('generation_time_seconds'),
            metadata.get('token_count_input'),
            metadata.get('token_count_output'),
            'active'
        ))

        db_connection.commit()
        cur.close()

        LOG.info(f"✅ Saved 10-K profile for {ticker} (FY{config.get('fiscal_year')}) to sec_filings table")

    except Exception as e:
        LOG.error(f"Failed to save company profile for {ticker}: {e}")
        raise
