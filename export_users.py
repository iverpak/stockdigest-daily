#!/usr/bin/env python3
"""
Standalone script to export beta users to CSV.
Lightweight version for Render cron job - no heavy dependencies.
"""
import os
import csv
import logging
import traceback
import psycopg
from psycopg.rows import dict_row

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

def export_beta_users_to_csv() -> int:
    """
    Export active beta users to CSV for daily processing.
    Returns: Number of users exported
    """
    output_path = "data/user_tickers.csv"

    try:
        # Connect to database
        with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT name, email, ticker1, ticker2, ticker3
                    FROM beta_users
                    WHERE status = 'active'
                    ORDER BY email
                """)
                users = cur.fetchall()

        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)

        # Write CSV with header
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'email', 'ticker1', 'ticker2', 'ticker3'])
            for user in users:
                writer.writerow([user['name'], user['email'], user['ticker1'], user['ticker2'], user['ticker3']])

        LOG.info(f"✅ Exported {len(users)} active beta users to {output_path}")
        return len(users)

    except Exception as e:
        LOG.error(f"❌ CSV export failed: {e}")
        LOG.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        count = export_beta_users_to_csv()
        print(f"Successfully exported {count} users")
        exit(0)
    except Exception as e:
        print(f"Export failed: {e}")
        exit(1)
