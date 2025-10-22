-- Migration: Add account_id to support multiple accounts per email
-- Date: 2025-10-22
-- Purpose: Allow users to have multiple ticker combinations with same email

-- Step 1: Add account_id column (will become PRIMARY KEY)
-- Use SERIAL to auto-generate unique IDs
ALTER TABLE beta_users ADD COLUMN IF NOT EXISTS account_id SERIAL;

-- Step 2: Remove the email UNIQUE constraint
-- First, find the constraint name (it varies by installation)
DO $$
DECLARE
    constraint_name_var text;
BEGIN
    SELECT conname INTO constraint_name_var
    FROM pg_constraint
    WHERE conrelid = 'beta_users'::regclass
    AND contype = 'u'
    AND conkey = ARRAY[(SELECT attnum FROM pg_attribute WHERE attrelid = 'beta_users'::regclass AND attname = 'email')];

    IF constraint_name_var IS NOT NULL THEN
        EXECUTE format('ALTER TABLE beta_users DROP CONSTRAINT %I', constraint_name_var);
        RAISE NOTICE 'Dropped UNIQUE constraint: %', constraint_name_var;
    ELSE
        RAISE NOTICE 'No UNIQUE constraint found on email column';
    END IF;
END $$;

-- Step 3: Make account_id the PRIMARY KEY
-- First drop existing primary key if it exists
ALTER TABLE beta_users DROP CONSTRAINT IF EXISTS beta_users_pkey CASCADE;

-- Add new primary key on account_id
ALTER TABLE beta_users ADD PRIMARY KEY (account_id);

-- Step 4: Add UNIQUE constraint on (email, ticker1, ticker2, ticker3)
-- This prevents exact duplicate ticker combinations for same email
CREATE UNIQUE INDEX IF NOT EXISTS beta_users_email_tickers_unique
ON beta_users(email, ticker1, ticker2, ticker3);

-- Step 5: Update unsubscribe_tokens table to reference account_id
-- Add new column
ALTER TABLE unsubscribe_tokens ADD COLUMN IF NOT EXISTS account_id INTEGER;

-- Migrate existing tokens to use account_id
-- This assumes current tokens map to the first account with that email
UPDATE unsubscribe_tokens ut
SET account_id = (
    SELECT account_id
    FROM beta_users bu
    WHERE bu.email = ut.user_email
    LIMIT 1
)
WHERE account_id IS NULL;

-- Add foreign key constraint
ALTER TABLE unsubscribe_tokens
ADD CONSTRAINT fk_unsubscribe_account
FOREIGN KEY (account_id)
REFERENCES beta_users(account_id)
ON DELETE CASCADE;

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_unsubscribe_tokens_account_id
ON unsubscribe_tokens(account_id);

-- Step 6: Add helpful indexes for multi-account queries
CREATE INDEX IF NOT EXISTS idx_beta_users_email ON beta_users(email);
CREATE INDEX IF NOT EXISTS idx_beta_users_status ON beta_users(status);
CREATE INDEX IF NOT EXISTS idx_beta_users_email_status ON beta_users(email, status);

-- Verification queries (run these after migration)
-- SELECT COUNT(*), email FROM beta_users GROUP BY email HAVING COUNT(*) > 1;
-- SELECT * FROM beta_users WHERE email = 'stockdigest.research@gmail.com';
-- SELECT COUNT(*) FROM unsubscribe_tokens WHERE account_id IS NULL;
