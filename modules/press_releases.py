"""
Press Release Management Module

Handles storage, retrieval, and validation of AI-generated press release summaries.
Press releases are uniquely identified by (ticker, report_date, pr_title).
"""

import logging
from typing import Dict, List, Optional

LOG = logging.getLogger(__name__)


def save_press_release_to_database(
    ticker: str,
    company_name: str,
    report_date: str,           # YYYY-MM-DD HH:MM:SS (full datetime)
    pr_title: str,
    summary_text: str,
    ai_provider: str,
    ai_model: str,
    processing_duration_seconds: int,
    job_id: str,
    db_connection
) -> None:
    """
    Save press release summary to database.

    UNIQUE constraint: (ticker, report_date, pr_title)
    ON CONFLICT: Update existing (allows regeneration)

    Args:
        ticker: Stock ticker
        company_name: Company name
        report_date: Full datetime string (YYYY-MM-DD HH:MM:SS) from FMP API
        pr_title: Press release title (max 200 chars)
        summary_text: AI-generated summary
        ai_provider: 'claude' or 'gemini'
        ai_model: Model name (e.g., 'claude-sonnet-4-5-20250929')
        processing_duration_seconds: Processing time
        job_id: Job ID for tracking
        db_connection: Database connection object
    """
    LOG.info(f"Saving press release for {ticker} ({ai_provider}) to database: {pr_title[:50]}...")

    try:
        cur = db_connection.cursor()

        cur.execute("""
            INSERT INTO press_releases (
                ticker, company_name, report_date, pr_title,
                summary_text, ai_provider, ai_model,
                processing_duration_seconds, job_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, report_date, pr_title)
            DO UPDATE SET
                summary_text = EXCLUDED.summary_text,
                ai_provider = EXCLUDED.ai_provider,
                ai_model = EXCLUDED.ai_model,
                processing_duration_seconds = EXCLUDED.processing_duration_seconds,
                job_id = EXCLUDED.job_id,
                generated_at = NOW()
        """, (
            ticker, company_name, report_date, pr_title,
            summary_text, ai_provider, ai_model,
            processing_duration_seconds, job_id
        ))

        db_connection.commit()
        cur.close()

        LOG.info(f"✅ Saved press release for {ticker} to database")

    except Exception as e:
        LOG.error(f"Failed to save press release for {ticker}: {e}")
        raise


def db_check_press_release_exists(ticker: str, report_date: str, pr_title: str) -> bool:
    """
    Check if exact press release exists (by ticker + datetime + title).

    Used by cron to avoid duplicate processing.

    Args:
        ticker: Stock ticker
        report_date: Full datetime string (YYYY-MM-DD HH:MM:SS)
        pr_title: Press release title

    Returns:
        True if exists, False otherwise
    """
    try:
        # Import here to avoid circular dependency
        from app import db

        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as count FROM press_releases
                WHERE ticker = %s
                  AND report_date = %s
                  AND pr_title = %s
            """, (ticker, report_date, pr_title))
            result = cur.fetchone()
            return result['count'] > 0 if result else False
    except Exception as e:
        LOG.error(f"Error checking press release for {ticker}: {e}")
        return False


def db_has_any_press_releases_for_ticker(ticker: str) -> bool:
    """
    Check if ticker has ANY press releases (for first-check detection).

    Used by cron for silent initialization logic.

    Args:
        ticker: Stock ticker

    Returns:
        True if ticker has at least one press release, False otherwise
    """
    try:
        # Import here to avoid circular dependency
        from app import db

        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as count FROM press_releases
                WHERE ticker = %s
            """, (ticker,))
            result = cur.fetchone()
            return result['count'] > 0 if result else False
    except Exception as e:
        LOG.error(f"Error checking press releases for {ticker}: {e}")
        return False


def db_get_latest_press_release_datetime(ticker: str) -> Optional[str]:
    """
    Get the datetime of the most recent press release for a ticker.

    Used by cron for datetime-aware filtering (only process PRs newer than this).

    Args:
        ticker: Stock ticker

    Returns:
        Full datetime string (YYYY-MM-DD HH:MM:SS) or None if no PRs exist
    """
    try:
        # Import here to avoid circular dependency
        from app import db

        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT report_date
                FROM press_releases
                WHERE ticker = %s
                ORDER BY report_date DESC
                LIMIT 1
            """, (ticker,))
            result = cur.fetchone()

            if result and result['report_date']:
                # Convert datetime object to string format
                return result['report_date'].strftime('%Y-%m-%d %H:%M:%S')
            return None
    except Exception as e:
        LOG.error(f"Error getting latest press release datetime for {ticker}: {e}")
        return None


def db_get_all_press_releases_for_ticker(ticker: str) -> List[Dict]:
    """
    Get all press releases for a ticker (for admin UI display).

    Args:
        ticker: Stock ticker

    Returns:
        List of press release dicts (sorted by datetime DESC)
    """
    try:
        # Import here to avoid circular dependency
        from app import db

        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT
                    ticker,
                    report_date,
                    pr_title,
                    summary_text,
                    ai_provider,
                    ai_model,
                    processing_duration_seconds,
                    generated_at
                FROM press_releases
                WHERE ticker = %s
                ORDER BY report_date DESC, generated_at DESC
            """, (ticker,))

            return cur.fetchall()
    except Exception as e:
        LOG.error(f"Error fetching press releases for {ticker}: {e}")
        return []
