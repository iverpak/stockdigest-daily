#!/usr/bin/env python3
"""Quick script to check value_chain articles in database"""
import sys
sys.path.insert(0, '/workspaces/quantbrief-daily')

# Import db connection from app
from app import db

with db() as conn:
    with conn.cursor() as cur:
        # Check value_chain articles by type
        print("=" * 60)
        print("VALUE_CHAIN ARTICLES BY TYPE:")
        print("=" * 60)
        cur.execute("""
            SELECT category, value_chain_type, COUNT(*) as count
            FROM ticker_articles
            WHERE category = 'value_chain'
            GROUP BY category, value_chain_type
            ORDER BY count DESC
        """)
        rows = cur.fetchall()
        if rows:
            for row in rows:
                print(f"Category: {row['category']:<15} | Type: {row['value_chain_type']:<12} | Count: {row['count']}")
        else:
            print("No value_chain articles found")

        print("\n" + "=" * 60)
        print("ALL CATEGORIES:")
        print("=" * 60)
        cur.execute("""
            SELECT category, COUNT(*) as count
            FROM ticker_articles
            GROUP BY category
            ORDER BY count DESC
        """)
        for row in cur.fetchall():
            print(f"{row['category']:<20} | Count: {row['count']}")

        print("\n" + "=" * 60)
        print("SAMPLE VALUE_CHAIN ARTICLES (first 5):")
        print("=" * 60)
        cur.execute("""
            SELECT ta.ticker, ta.category, ta.value_chain_type, a.title, ta.search_keyword
            FROM ticker_articles ta
            JOIN articles a ON ta.article_id = a.id
            WHERE ta.category = 'value_chain'
            ORDER BY ta.found_at DESC
            LIMIT 5
        """)
        rows = cur.fetchall()
        if rows:
            for i, row in enumerate(rows, 1):
                print(f"\n{i}. Ticker: {row['ticker']}")
                print(f"   Category: {row['category']} | Type: {row['value_chain_type']}")
                print(f"   Keyword: {row['search_keyword']}")
                print(f"   Title: {row['title'][:80]}...")
        else:
            print("No value_chain articles found")
