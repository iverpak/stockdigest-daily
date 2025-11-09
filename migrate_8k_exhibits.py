#!/usr/bin/env python3
"""
Migration: Convert sec_8k_filings to exhibit-level granularity

This migration adds columns for exhibit-level tracking and changes the
UNIQUE constraint to allow multiple exhibits per filing.

Run on Render:
    python migrate_8k_exhibits.py
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

        # Check if columns already exist
        print("🔍 Checking current schema...")
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'sec_8k_filings'
            AND column_name IN ('exhibit_number', 'exhibit_description', 'exhibit_type', 'char_count');
        """)

        existing_columns = [row[0] for row in cur.fetchall()]

        if len(existing_columns) == 4:
            print("✅ Exhibit columns already exist - no migration needed")
            cur.close()
            conn.close()
            return

        print("📝 Adding exhibit-level columns to sec_8k_filings...")

        # Add new columns
        if 'exhibit_number' not in existing_columns:
            cur.execute("ALTER TABLE sec_8k_filings ADD COLUMN exhibit_number VARCHAR(10);")
            print("  ✅ Added exhibit_number column")

        if 'exhibit_description' not in existing_columns:
            cur.execute("ALTER TABLE sec_8k_filings ADD COLUMN exhibit_description VARCHAR(200);")
            print("  ✅ Added exhibit_description column")

        if 'exhibit_type' not in existing_columns:
            cur.execute("ALTER TABLE sec_8k_filings ADD COLUMN exhibit_type VARCHAR(50);")
            print("  ✅ Added exhibit_type column")

        if 'char_count' not in existing_columns:
            cur.execute("ALTER TABLE sec_8k_filings ADD COLUMN char_count INTEGER;")
            print("  ✅ Added char_count column")

        # Check if old unique constraint exists
        print("\n🔍 Checking unique constraints...")
        cur.execute("""
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = 'sec_8k_filings'
            AND constraint_type = 'UNIQUE'
            AND constraint_name = 'sec_8k_unique';
        """)

        old_constraint = cur.fetchone()

        if old_constraint:
            print("📝 Dropping old UNIQUE constraint (ticker, accession_number)...")
            cur.execute("ALTER TABLE sec_8k_filings DROP CONSTRAINT sec_8k_unique;")
            print("  ✅ Dropped old constraint")

        # Check if new constraint exists
        cur.execute("""
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = 'sec_8k_filings'
            AND constraint_type = 'UNIQUE'
            AND constraint_name = 'sec_8k_exhibit_unique';
        """)

        new_constraint = cur.fetchone()

        if not new_constraint:
            print("📝 Adding new UNIQUE constraint (ticker, accession_number, exhibit_number)...")
            cur.execute("""
                ALTER TABLE sec_8k_filings
                ADD CONSTRAINT sec_8k_exhibit_unique
                UNIQUE(ticker, accession_number, exhibit_number);
            """)
            print("  ✅ Added new constraint")

        conn.commit()
        print("\n✅ Successfully added exhibit-level columns and updated constraint")

        # Verify the migration
        print("\n🔍 Verifying migration...")
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'sec_8k_filings'
            ORDER BY ordinal_position;
        """)

        print("\n📋 Current schema for sec_8k_filings:")
        print("-" * 70)
        for row in cur.fetchall():
            nullable = "NULL" if row[2] == 'YES' else "NOT NULL"
            print(f"  {row[0]:30s} {row[1]:20s} {nullable}")
        print("-" * 70)

        cur.close()
        conn.close()

        print("\n✅ Migration completed successfully!")
        print("\n💡 You can now generate 8-K summaries with exhibit-level tracking")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
