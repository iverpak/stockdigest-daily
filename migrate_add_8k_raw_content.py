#!/usr/bin/env python3
"""
Migration: Add raw_content column to sec_8k_filings table

This migration adds the missing raw_content TEXT column that was added
to the schema but not migrated to production database.

Run on Render:
    python migrate_add_8k_raw_content.py
"""

import os
import sys
import psycopg

def main():
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)

    print("🔄 Connecting to database...")

    try:
        conn = psycopg.connect(database_url)
        cur = conn.cursor()

        # Check if column already exists
        print("🔍 Checking if raw_content column exists...")
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'sec_8k_filings'
            AND column_name = 'raw_content';
        """)

        exists = cur.fetchone()

        if exists:
            print("✅ Column raw_content already exists - no migration needed")
            cur.close()
            conn.close()
            return

        # Add the missing column
        print("📝 Adding raw_content column to sec_8k_filings...")
        cur.execute("""
            ALTER TABLE sec_8k_filings
            ADD COLUMN raw_content TEXT;
        """)

        conn.commit()
        print("✅ Successfully added raw_content column")

        # Verify the migration
        print("\n🔍 Verifying migration...")
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'sec_8k_filings'
            ORDER BY ordinal_position;
        """)

        print("\n📋 Current schema for sec_8k_filings:")
        print("-" * 60)
        for row in cur.fetchall():
            nullable = "NULL" if row[2] == 'YES' else "NOT NULL"
            print(f"  {row[0]:30s} {row[1]:20s} {nullable}")
        print("-" * 60)

        cur.close()
        conn.close()

        print("\n✅ Migration completed successfully!")
        print("\n💡 You can now retry generating the 8-K summary for PLD")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
