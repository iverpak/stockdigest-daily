-- Migration: Add entity_logs table for storing entity extractions
-- Date: 2025-01-15
-- Purpose: Replace ephemeral JSON files with database storage

CREATE TABLE IF NOT EXISTS entity_logs (
    ticker VARCHAR(20) PRIMARY KEY,
    company_name VARCHAR(255),

    -- JSONB array containing all extractions chronologically
    -- Structure: {"ticker": "AAPL", "company_name": "...", "extractions": [...]}
    entity_data JSONB NOT NULL DEFAULT '{"extractions": []}'::jsonb,

    -- Tracking
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_entity_logs_ticker ON entity_logs(ticker);
CREATE INDEX IF NOT EXISTS idx_entity_logs_updated ON entity_logs(last_updated DESC);

-- GIN index for JSONB querying (find tickers mentioning specific entities)
CREATE INDEX IF NOT EXISTS idx_entity_data ON entity_logs USING GIN (entity_data);

COMMENT ON TABLE entity_logs IS 'Stores all entity extractions (competitors, suppliers, customers) from SEC filings and transcripts for each ticker';
COMMENT ON COLUMN entity_logs.entity_data IS 'JSONB array of extractions: [{"filing_type": "10-K", "fiscal_year": 2024, "competitors": [...], "suppliers": [...], "customers": [...]}]';
