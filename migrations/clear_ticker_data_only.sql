-- Clear ONLY Ticker-Related Data (Preserves User Data)
-- Date: October 31, 2025
-- Safe to run: Does NOT delete beta_users, unsubscribe_tokens, SEC filings, transcripts, press releases

-- ============================================================================
-- WHAT THIS DELETES:
-- ============================================================================
-- ✅ ticker_reference (ticker metadata)
-- ✅ ticker_articles (article-ticker links)
-- ✅ ticker_feeds (feed-ticker links)
-- ✅ feeds (RSS feeds)
-- ✅ articles (article content)
-- ✅ executive_summaries (daily AI summaries)
-- ✅ domain_names (domain metadata)
-- ✅ competitor_metadata (competitor relationships)
-- ✅ ticker_processing_jobs (job queue)
-- ✅ ticker_processing_batches (job batches)
-- ✅ email_queue (daily workflow email queue)

-- ============================================================================
-- WHAT THIS PRESERVES:
-- ============================================================================
-- ❌ beta_users (user signups)
-- ❌ unsubscribe_tokens (user preferences)
-- ❌ sec_filings (10-K/10-Q company profiles)
-- ❌ transcript_summaries (earnings transcripts)
-- ❌ press_releases (press release summaries)

-- ============================================================================
-- EXECUTION
-- ============================================================================

-- Step 1: Show counts BEFORE deletion (for verification)
SELECT 'BEFORE DELETION' as status;
SELECT 'ticker_reference' as table_name, COUNT(*) as row_count FROM ticker_reference
UNION ALL
SELECT 'articles', COUNT(*) FROM articles
UNION ALL
SELECT 'ticker_articles', COUNT(*) FROM ticker_articles
UNION ALL
SELECT 'feeds', COUNT(*) FROM feeds
UNION ALL
SELECT 'ticker_feeds', COUNT(*) FROM ticker_feeds
UNION ALL
SELECT 'executive_summaries', COUNT(*) FROM executive_summaries
UNION ALL
SELECT 'domain_names', COUNT(*) FROM domain_names
UNION ALL
SELECT 'competitor_metadata', COUNT(*) FROM competitor_metadata
UNION ALL
SELECT 'ticker_processing_jobs', COUNT(*) FROM ticker_processing_jobs
UNION ALL
SELECT 'ticker_processing_batches', COUNT(*) FROM ticker_processing_batches
UNION ALL
SELECT 'email_queue', COUNT(*) FROM email_queue;

-- Step 2: Delete data (in correct order to respect foreign keys)
DELETE FROM email_queue;
DELETE FROM ticker_processing_jobs;
DELETE FROM ticker_processing_batches;
DELETE FROM executive_summaries;
DELETE FROM ticker_articles;
DELETE FROM articles;
DELETE FROM ticker_feeds;
DELETE FROM feeds;
DELETE FROM domain_names;
DELETE FROM competitor_metadata;
DELETE FROM ticker_reference;

-- Step 3: Reset auto-increment sequences
ALTER SEQUENCE IF EXISTS articles_id_seq RESTART WITH 1;
ALTER SEQUENCE IF EXISTS feeds_id_seq RESTART WITH 1;
ALTER SEQUENCE IF EXISTS domain_names_id_seq RESTART WITH 1;
ALTER SEQUENCE IF EXISTS competitor_metadata_id_seq RESTART WITH 1;

-- Step 4: Show counts AFTER deletion (should all be 0)
SELECT 'AFTER DELETION' as status;
SELECT 'ticker_reference' as table_name, COUNT(*) as row_count FROM ticker_reference
UNION ALL
SELECT 'articles', COUNT(*) FROM articles
UNION ALL
SELECT 'ticker_articles', COUNT(*) FROM ticker_articles
UNION ALL
SELECT 'feeds', COUNT(*) FROM feeds
UNION ALL
SELECT 'ticker_feeds', COUNT(*) FROM ticker_feeds
UNION ALL
SELECT 'executive_summaries', COUNT(*) FROM executive_summaries
UNION ALL
SELECT 'domain_names', COUNT(*) FROM domain_names
UNION ALL
SELECT 'competitor_metadata', COUNT(*) FROM competitor_metadata
UNION ALL
SELECT 'ticker_processing_jobs', COUNT(*) FROM ticker_processing_jobs
UNION ALL
SELECT 'ticker_processing_batches', COUNT(*) FROM ticker_processing_batches
UNION ALL
SELECT 'email_queue', COUNT(*) FROM email_queue;

-- Step 5: Verify PRESERVED data (should be unchanged)
SELECT 'PRESERVED DATA (should be unchanged)' as status;
SELECT 'beta_users' as table_name, COUNT(*) as row_count FROM beta_users
UNION ALL
SELECT 'unsubscribe_tokens', COUNT(*) FROM unsubscribe_tokens
UNION ALL
SELECT 'sec_filings', COUNT(*) FROM sec_filings
UNION ALL
SELECT 'transcript_summaries', COUNT(*) FROM transcript_summaries
UNION ALL
SELECT 'press_releases', COUNT(*) FROM press_releases;

-- ============================================================================
-- EXPECTED OUTPUT:
-- ============================================================================
-- BEFORE DELETION: Shows current row counts
-- AFTER DELETION: All ticker tables = 0 rows
-- PRESERVED DATA: User data unchanged (beta_users, tokens, filings, etc.)

-- ============================================================================
-- NEXT STEPS AFTER RUNNING THIS:
-- ============================================================================
-- 1. Verify "PRESERVED DATA" counts are unchanged
-- 2. Restart your app to ensure schema is up to date
-- 3. Run /admin/init for your test tickers (e.g., TSLA, INTC, NFLX)
-- 4. Verify 19 feeds created per ticker
-- 5. Verify value chain metadata populated in ticker_reference
