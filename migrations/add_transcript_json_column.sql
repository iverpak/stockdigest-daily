-- Migration: Add JSON storage for transcript summaries v2
-- Date: 2025-01-15
-- Purpose: Enable structured JSON storage alongside text for backward compatibility

-- Add columns for JSON storage and version tracking
ALTER TABLE transcript_summaries
ADD COLUMN IF NOT EXISTS summary_json JSONB,
ADD COLUMN IF NOT EXISTS prompt_version VARCHAR(10) DEFAULT 'v1';

-- Add index for querying by prompt version
CREATE INDEX IF NOT EXISTS idx_transcript_summaries_prompt_version
ON transcript_summaries(prompt_version);

-- Add index for JSON queries (GIN index for JSONB)
CREATE INDEX IF NOT EXISTS idx_transcript_summaries_json
ON transcript_summaries USING GIN (summary_json);

-- Verify migration
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'transcript_summaries'
AND column_name IN ('summary_json', 'prompt_version')
ORDER BY column_name;
