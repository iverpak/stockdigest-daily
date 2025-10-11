#!/usr/bin/env python3
"""Query domain_names table to see all stored mappings"""
import sys
sys.path.insert(0, '/workspaces/quantbrief-daily')
from app import db

with db() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                domain,
                formal_name,
                CASE
                    WHEN ai_generated = TRUE THEN 'AI'
                    ELSE 'Learned'
                END as source,
                created_at::date as first_seen
            FROM domain_names
            ORDER BY formal_name
        """)

        results = cur.fetchall()

        print(f"\n{'='*80}")
        print(f"DOMAIN MAPPINGS - Total: {len(results)}")
        print(f"{'='*80}\n")

        print(f"{'Publication Name':<40} {'Domain':<35} {'Source':<10} {'First Seen'}")
        print(f"{'-'*40} {'-'*35} {'-'*10} {'-'*10}")

        for row in results:
            print(f"{row['formal_name']:<40} {row['domain']:<35} {row['source']:<10} {row['first_seen']}")

        print(f"\n{'='*80}\n")
