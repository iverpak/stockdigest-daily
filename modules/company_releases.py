"""
Helper functions for querying and managing company_releases table.

This module provides database helper functions for:
- FMP press releases (source_type='fmp_press_release')
- 8-K SEC filings (source_type='8k_exhibit')

Replaces legacy modules/press_releases.py functions.
"""

import logging
from typing import Optional, List, Dict

LOG = logging.getLogger(__name__)


def db_has_any_fmp_releases_for_ticker(ticker: str) -> bool:
    """
    Check if ticker has ANY FMP press releases in company_releases table.

    Used by cron for silent initialization detection.

    Args:
        ticker: Stock ticker

    Returns:
        True if ticker has any FMP releases, False otherwise
    """
    try:
        from app import db

        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as count FROM company_releases
                WHERE ticker = %s
                  AND source_type = 'fmp_press_release'
            """, (ticker,))
            result = cur.fetchone()
            return result['count'] > 0 if result else False
    except Exception as e:
        LOG.error(f"Error checking FMP releases for {ticker}: {e}")
        return False


def db_has_any_8k_for_ticker(ticker: str) -> bool:
    """
    Check if ticker has ANY 8-K filings in company_releases table.

    Used by cron for silent initialization detection.

    Args:
        ticker: Stock ticker

    Returns:
        True if ticker has any 8-K filings, False otherwise
    """
    try:
        from app import db

        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as count FROM company_releases
                WHERE ticker = %s
                  AND source_type = '8k_exhibit'
            """, (ticker,))
            result = cur.fetchone()
            return result['count'] > 0 if result else False
    except Exception as e:
        LOG.error(f"Error checking 8-K filings for {ticker}: {e}")
        return False


def db_check_fmp_release_exists(ticker: str, filing_date: str, title: str) -> bool:
    """
    Check if specific FMP press release exists in company_releases.

    Used by cron to avoid duplicate processing.

    Args:
        ticker: Stock ticker
        filing_date: Date string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        title: Press release title

    Returns:
        True if exists, False otherwise
    """
    try:
        from app import db

        # Extract date part only (first 10 chars)
        date_only = filing_date[:10] if len(filing_date) >= 10 else filing_date

        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as count FROM company_releases
                WHERE ticker = %s
                  AND filing_date::text = %s
                  AND report_title = %s
                  AND source_type = 'fmp_press_release'
            """, (ticker, date_only, title))
            result = cur.fetchone()
            return result['count'] > 0 if result else False
    except Exception as e:
        LOG.error(f"Error checking FMP release for {ticker}: {e}")
        return False


def db_check_8k_filing_exists(ticker: str, filing_date: str, accession_number: str) -> bool:
    """
    Check if 8-K filing has been processed (checks if ANY exhibit exists for this 8-K).

    Used by cron to avoid reprocessing same 8-K.

    Args:
        ticker: Stock ticker
        filing_date: Date string (YYYY-MM-DD)
        accession_number: SEC accession number

    Returns:
        True if at least one exhibit from this 8-K exists, False otherwise
    """
    try:
        from app import db

        with db() as conn, conn.cursor() as cur:
            # Check if ANY exhibit from this 8-K exists
            # We join with sec_8k_filings to match by accession_number
            cur.execute("""
                SELECT COUNT(*) as count
                FROM company_releases cr
                JOIN sec_8k_filings s8k ON cr.source_id = s8k.id
                WHERE cr.ticker = %s
                  AND s8k.accession_number = %s
                  AND cr.source_type = '8k_exhibit'
            """, (ticker, accession_number))
            result = cur.fetchone()
            return result['count'] > 0 if result else False
    except Exception as e:
        LOG.error(f"Error checking 8-K filing for {ticker}: {e}")
        return False


def db_get_latest_fmp_release_datetime(ticker: str) -> Optional[str]:
    """
    Get most recent FMP press release datetime for ticker.

    Returns full datetime string for accurate comparison.

    Args:
        ticker: Stock ticker

    Returns:
        Datetime string 'YYYY-MM-DD HH:MM:SS' or None if no releases found
    """
    try:
        from app import db

        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT filing_date
                FROM company_releases
                WHERE ticker = %s
                  AND source_type = 'fmp_press_release'
                ORDER BY filing_date DESC, generated_at DESC
                LIMIT 1
            """, (ticker,))
            result = cur.fetchone()

            if result:
                # filing_date is DATE type, but we need to return datetime string for comparison
                filing_date = result['filing_date'] if isinstance(result, dict) else result[0]
                # Convert date to string and append time component for datetime comparison
                # Database stores DATE only (YYYY-MM-DD), append " 00:00:00" for consistency
                return str(filing_date) + " 00:00:00"
            return None
    except Exception as e:
        LOG.error(f"Error getting latest FMP release for {ticker}: {e}")
        return None


def db_get_latest_8k_filing_date(ticker: str) -> Optional[str]:
    """
    Get most recent 8-K filing date for ticker.

    Args:
        ticker: Stock ticker

    Returns:
        Date string 'YYYY-MM-DD' or None if no 8-Ks found
    """
    try:
        from app import db

        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT filing_date
                FROM company_releases
                WHERE ticker = %s
                  AND source_type = '8k_exhibit'
                ORDER BY filing_date DESC, generated_at DESC
                LIMIT 1
            """, (ticker,))
            result = cur.fetchone()

            if result:
                filing_date = result['filing_date'] if isinstance(result, dict) else result[0]
                return str(filing_date)
            return None
    except Exception as e:
        LOG.error(f"Error getting latest 8-K filing for {ticker}: {e}")
        return None
