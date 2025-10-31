-- Clear Database for Fresh Start
-- Date: October 31, 2025
-- WARNING: This deletes ALL data. Use with caution!

-- Option 1: DELETE all data but keep table structure (RECOMMENDED)
-- This is faster and preserves indexes/constraints

DELETE FROM ticker_processing_jobs;
DELETE FROM ticker_processing_batches;
DELETE FROM email_queue;
DELETE FROM unsubscribe_tokens;
DELETE FROM press_releases;
DELETE FROM transcript_summaries;
DELETE FROM sec_filings;
DELETE FROM executive_summaries;
DELETE FROM ticker_articles;
DELETE FROM articles;
DELETE FROM ticker_feeds;
DELETE FROM feeds;
DELETE FROM domain_names;
DELETE FROM competitor_metadata;
DELETE FROM ticker_reference;
DELETE FROM beta_users;

-- Reset sequences (auto-increment IDs)
ALTER SEQUENCE IF EXISTS articles_id_seq RESTART WITH 1;
ALTER SEQUENCE IF EXISTS feeds_id_seq RESTART WITH 1;

-- Verify tables are empty
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
SELECT 'executive_summaries', COUNT(*) FROM executive_summaries;

-- Option 2: DROP and recreate tables (NUCLEAR OPTION - use only if Option 1 fails)
-- Uncomment below if you want to completely recreate tables:

/*
DROP TABLE IF EXISTS ticker_processing_jobs CASCADE;
DROP TABLE IF EXISTS ticker_processing_batches CASCADE;
DROP TABLE IF EXISTS email_queue CASCADE;
DROP TABLE IF EXISTS unsubscribe_tokens CASCADE;
DROP TABLE IF EXISTS press_releases CASCADE;
DROP TABLE IF EXISTS transcript_summaries CASCADE;
DROP TABLE IF EXISTS sec_filings CASCADE;
DROP TABLE IF EXISTS executive_summaries CASCADE;
DROP TABLE IF EXISTS ticker_articles CASCADE;
DROP TABLE IF EXISTS articles CASCADE;
DROP TABLE IF EXISTS ticker_feeds CASCADE;
DROP TABLE IF EXISTS feeds CASCADE;
DROP TABLE IF EXISTS domain_names CASCADE;
DROP TABLE IF EXISTS competitor_metadata CASCADE;
DROP TABLE IF EXISTS ticker_reference CASCADE;

-- Then restart your app to recreate tables via ensure_schema()
*/
