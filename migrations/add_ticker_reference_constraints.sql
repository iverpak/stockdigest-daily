-- ============================================================================
-- Ticker Normalization & Consistency Migration
-- ============================================================================
--
-- PURPOSE: Enforce ticker normalization and consistency across all tables
--
-- WHAT THIS DOES:
-- 1. Adds foreign key constraints to ensure all tickers exist in ticker_reference
-- 2. Normalizes any existing non-standard ticker values
-- 3. Prevents future insertions of invalid/un-normalized tickers
--
-- HOW TO RUN:
--   psql $DATABASE_URL < migrations/add_ticker_reference_constraints.sql
--
-- SAFETY:
--   - All operations are idempotent (safe to run multiple times)
--   - Uses ON CONFLICT to skip if constraints already exist
--   - Normalization happens BEFORE adding constraints
--
-- ============================================================================

BEGIN;

-- ============================================================================
-- STEP 1: Normalize existing tickers in all tables
-- ============================================================================

-- NOTE: This uses PostgreSQL's UPPER() function as a simplified normalization
-- Real normalization happens in application code via normalize_ticker_format()
-- This is a safety net to catch obvious issues (lowercase, whitespace)

-- Normalize company_releases
UPDATE company_releases
SET ticker = UPPER(TRIM(ticker))
WHERE ticker != UPPER(TRIM(ticker));

-- Normalize sec_filings
UPDATE sec_filings
SET ticker = UPPER(TRIM(ticker))
WHERE ticker != UPPER(TRIM(ticker));

-- Normalize transcript_summaries
UPDATE transcript_summaries
SET ticker = UPPER(TRIM(ticker))
WHERE ticker != UPPER(TRIM(ticker));

-- Normalize beta_users (all 3 ticker columns)
UPDATE beta_users
SET
    ticker1 = UPPER(TRIM(ticker1)),
    ticker2 = UPPER(TRIM(ticker2)),
    ticker3 = UPPER(TRIM(ticker3))
WHERE
    ticker1 != UPPER(TRIM(ticker1)) OR
    ticker2 != UPPER(TRIM(ticker2)) OR
    ticker3 != UPPER(TRIM(ticker3));

-- ============================================================================
-- STEP 2: Add foreign key constraints (with error handling)
-- ============================================================================

-- company_releases → ticker_reference
DO $$
BEGIN
    -- First check if there are any tickers that would violate the constraint
    IF EXISTS (
        SELECT 1 FROM company_releases cr
        LEFT JOIN ticker_reference tr ON cr.ticker = tr.ticker
        WHERE tr.ticker IS NULL
    ) THEN
        -- Log violating tickers
        RAISE NOTICE 'WARNING: Found tickers in company_releases not in ticker_reference:';
        RAISE NOTICE '%', (
            SELECT STRING_AGG(DISTINCT ticker, ', ')
            FROM company_releases cr
            LEFT JOIN ticker_reference tr ON cr.ticker = tr.ticker
            WHERE tr.ticker IS NULL
        );
        RAISE NOTICE 'These rows will need to be fixed manually before constraint can be added.';
        RAISE NOTICE 'Options: 1) Add missing tickers to ticker_reference, or 2) Delete invalid releases';
    ELSE
        -- Safe to add constraint
        ALTER TABLE company_releases
        ADD CONSTRAINT fk_company_releases_ticker
        FOREIGN KEY (ticker) REFERENCES ticker_reference(ticker)
        ON DELETE RESTRICT
        ON UPDATE CASCADE;

        RAISE NOTICE 'Successfully added FK constraint: company_releases.ticker → ticker_reference.ticker';
    END IF;
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'FK constraint company_releases.ticker already exists - skipping';
END $$;

-- sec_filings → ticker_reference
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM sec_filings sf
        LEFT JOIN ticker_reference tr ON sf.ticker = tr.ticker
        WHERE tr.ticker IS NULL
    ) THEN
        RAISE NOTICE 'WARNING: Found tickers in sec_filings not in ticker_reference:';
        RAISE NOTICE '%', (
            SELECT STRING_AGG(DISTINCT ticker, ', ')
            FROM sec_filings sf
            LEFT JOIN ticker_reference tr ON sf.ticker = tr.ticker
            WHERE tr.ticker IS NULL
        );
    ELSE
        ALTER TABLE sec_filings
        ADD CONSTRAINT fk_sec_filings_ticker
        FOREIGN KEY (ticker) REFERENCES ticker_reference(ticker)
        ON DELETE RESTRICT
        ON UPDATE CASCADE;

        RAISE NOTICE 'Successfully added FK constraint: sec_filings.ticker → ticker_reference.ticker';
    END IF;
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'FK constraint sec_filings.ticker already exists - skipping';
END $$;

-- transcript_summaries → ticker_reference
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM transcript_summaries ts
        LEFT JOIN ticker_reference tr ON ts.ticker = tr.ticker
        WHERE tr.ticker IS NULL
    ) THEN
        RAISE NOTICE 'WARNING: Found tickers in transcript_summaries not in ticker_reference:';
        RAISE NOTICE '%', (
            SELECT STRING_AGG(DISTINCT ticker, ', ')
            FROM transcript_summaries ts
            LEFT JOIN ticker_reference tr ON ts.ticker = tr.ticker
            WHERE tr.ticker IS NULL
        );
    ELSE
        ALTER TABLE transcript_summaries
        ADD CONSTRAINT fk_transcript_summaries_ticker
        FOREIGN KEY (ticker) REFERENCES ticker_reference(ticker)
        ON DELETE RESTRICT
        ON UPDATE CASCADE;

        RAISE NOTICE 'Successfully added FK constraint: transcript_summaries.ticker → ticker_reference.ticker';
    END IF;
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'FK constraint transcript_summaries.ticker already exists - skipping';
END $$;

-- ============================================================================
-- STEP 3: Create indexes on ticker columns (improves FK performance)
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_company_releases_ticker_date
ON company_releases(ticker, filing_date DESC);

CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker
ON sec_filings(ticker);

CREATE INDEX IF NOT EXISTS idx_transcript_summaries_ticker
ON transcript_summaries(ticker);

-- ============================================================================
-- STEP 4: Validation queries
-- ============================================================================

-- Show summary of what was done
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'MIGRATION COMPLETE';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Company releases: % rows', (SELECT COUNT(*) FROM company_releases);
    RAISE NOTICE 'SEC filings: % rows', (SELECT COUNT(*) FROM sec_filings);
    RAISE NOTICE 'Transcript summaries: % rows', (SELECT COUNT(*) FROM transcript_summaries);
    RAISE NOTICE 'Beta users: % rows', (SELECT COUNT(*) FROM beta_users);
    RAISE NOTICE '========================================';
END $$;

COMMIT;
