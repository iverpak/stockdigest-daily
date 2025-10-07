# Beta Users SQL Reference

Quick SQL queries for managing StockDigest beta users.

## View All Beta Users

```sql
-- All beta users with basic info
SELECT
    id,
    name,
    email,
    ticker1,
    ticker2,
    ticker3,
    status,
    created_at
FROM beta_users
ORDER BY created_at DESC;
```

## Active Users Only

```sql
-- Only users who will receive daily emails
SELECT
    name,
    email,
    ticker1,
    ticker2,
    ticker3,
    created_at
FROM beta_users
WHERE status = 'active'
ORDER BY created_at DESC;
```

## User Count Stats

```sql
-- Total users by status
SELECT
    status,
    COUNT(*) as user_count
FROM beta_users
GROUP BY status;
```

## Recent Signups

```sql
-- Users who signed up in last 7 days
SELECT
    name,
    email,
    ticker1,
    ticker2,
    ticker3,
    created_at
FROM beta_users
WHERE created_at >= NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;
```

## Search by Email

```sql
-- Find specific user by email
SELECT *
FROM beta_users
WHERE email = 'john@example.com';
```

## Search by Ticker

```sql
-- Find all users tracking a specific ticker
SELECT
    name,
    email,
    created_at
FROM beta_users
WHERE ticker1 = 'AAPL'
   OR ticker2 = 'AAPL'
   OR ticker3 = 'AAPL'
ORDER BY created_at DESC;
```

## Ticker Popularity

```sql
-- Most popular tickers among beta users
SELECT ticker, COUNT(*) as user_count
FROM (
    SELECT ticker1 as ticker FROM beta_users WHERE status = 'active'
    UNION ALL
    SELECT ticker2 FROM beta_users WHERE status = 'active'
    UNION ALL
    SELECT ticker3 FROM beta_users WHERE status = 'active'
) as all_tickers
GROUP BY ticker
ORDER BY user_count DESC
LIMIT 20;
```

## Update User Status

```sql
-- Pause a user (stop sending emails)
UPDATE beta_users
SET status = 'paused'
WHERE email = 'john@example.com';

-- Reactivate a user
UPDATE beta_users
SET status = 'active'
WHERE email = 'john@example.com';
```

## Delete User (Careful!)

```sql
-- Remove user completely
DELETE FROM beta_users
WHERE email = 'john@example.com';
```

## Export Preview (CSV Format)

```sql
-- Preview what will be exported to CSV
SELECT
    name,
    email,
    ticker1,
    ticker2,
    ticker3
FROM beta_users
WHERE status = 'active'
ORDER BY email;
```

## Duplicate Email Check

```sql
-- Check for duplicate emails (should be none due to UNIQUE constraint)
SELECT
    email,
    COUNT(*) as count
FROM beta_users
GROUP BY email
HAVING COUNT(*) > 1;
```

## User Engagement Timeline

```sql
-- Signups per day (last 30 days)
SELECT
    DATE(created_at) as signup_date,
    COUNT(*) as signups
FROM beta_users
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY signup_date DESC;
```

## Useful Filters

```sql
-- Users tracking Canadian stocks (.TO)
SELECT name, email, ticker1, ticker2, ticker3
FROM beta_users
WHERE ticker1 LIKE '%.TO'
   OR ticker2 LIKE '%.TO'
   OR ticker3 LIKE '%.TO';

-- Users tracking only US stocks
SELECT name, email, ticker1, ticker2, ticker3
FROM beta_users
WHERE ticker1 NOT LIKE '%.TO'
  AND ticker2 NOT LIKE '%.TO'
  AND ticker3 NOT LIKE '%.TO';
```

## Quick Stats Dashboard

```sql
-- Complete overview
SELECT
    (SELECT COUNT(*) FROM beta_users) as total_users,
    (SELECT COUNT(*) FROM beta_users WHERE status = 'active') as active_users,
    (SELECT COUNT(*) FROM beta_users WHERE status = 'paused') as paused_users,
    (SELECT COUNT(*) FROM beta_users WHERE created_at >= NOW() - INTERVAL '7 days') as signups_last_7_days,
    (SELECT COUNT(*) FROM beta_users WHERE created_at >= NOW() - INTERVAL '1 day') as signups_today;
```
