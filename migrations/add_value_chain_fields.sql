-- Migration: Add Value Chain Fields to StockDigest Database
-- Date: October 31, 2025
-- Description: Adds 8 value chain columns + 2 value_chain_type columns

-- 1. Add 8 value chain columns to ticker_reference table
ALTER TABLE ticker_reference
ADD COLUMN IF NOT EXISTS upstream_1_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS upstream_1_ticker VARCHAR(20),
ADD COLUMN IF NOT EXISTS upstream_2_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS upstream_2_ticker VARCHAR(20),
ADD COLUMN IF NOT EXISTS downstream_1_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS downstream_1_ticker VARCHAR(20),
ADD COLUMN IF NOT EXISTS downstream_2_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS downstream_2_ticker VARCHAR(20);

-- 2. Add value_chain_type column to ticker_articles table
ALTER TABLE ticker_articles
ADD COLUMN IF NOT EXISTS value_chain_type VARCHAR(10)
CHECK (value_chain_type IN ('upstream', 'downstream'));

-- 3. Add value_chain_type column to feeds table
ALTER TABLE feeds
ADD COLUMN IF NOT EXISTS value_chain_type VARCHAR(10)
CHECK (value_chain_type IN ('upstream', 'downstream'));

-- Verify changes
SELECT column_name, data_type, character_maximum_length
FROM information_schema.columns
WHERE table_name = 'ticker_reference'
AND column_name LIKE '%stream%'
ORDER BY ordinal_position;

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'ticker_articles'
AND column_name = 'value_chain_type';

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'feeds'
AND column_name = 'value_chain_type';
