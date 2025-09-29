# NEW FEED ARCHITECTURE
# This file contains the new many-to-many feed architecture functions

def ensure_new_feed_architecture():
    """Create the new many-to-many feed architecture with per-relationship categories"""
    from app import db, LOG

    with db() as conn, conn.cursor() as cur:
        # NEW ARCHITECTURE: Feeds table (category-neutral, shareable)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feeds (
                id SERIAL PRIMARY KEY,
                url VARCHAR(2048) UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                search_keyword VARCHAR(255),
                competitor_ticker VARCHAR(10),
                retain_days INTEGER DEFAULT 90,
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)

        # NEW ARCHITECTURE: Ticker-to-Feed mapping with CATEGORY per relationship
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ticker_feeds (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                feed_id INTEGER NOT NULL REFERENCES feeds(id) ON DELETE CASCADE,
                category VARCHAR(20) NOT NULL DEFAULT 'company',
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, feed_id)
            );
        """)

        # Create indexes for new architecture
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feeds_url ON feeds(url);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feeds_active ON feeds(active);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ticker_feeds_ticker ON ticker_feeds(ticker);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ticker_feeds_feed_id ON ticker_feeds(feed_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ticker_feeds_category ON ticker_feeds(category);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ticker_feeds_active ON ticker_feeds(active);")

        LOG.info("✅ New feed architecture (feeds + ticker_feeds with per-relationship categories) created successfully")

def fix_found_url_foreign_key():
    """Fix the found_url table foreign key to point to feeds instead of source_feed"""
    from app import db, LOG

    with db() as conn, conn.cursor() as cur:
        # Drop the old foreign key constraint if it exists
        cur.execute("""
            ALTER TABLE found_url
            DROP CONSTRAINT IF EXISTS found_url_feed_id_fkey;
        """)

        # Add new foreign key constraint pointing to feeds table
        cur.execute("""
            ALTER TABLE found_url
            ADD CONSTRAINT found_url_feed_id_fkey
            FOREIGN KEY (feed_id) REFERENCES feeds(id) ON DELETE SET NULL;
        """)

        LOG.info("✅ Fixed found_url foreign key constraint to point to feeds table")

def migrate_to_category_per_relationship():
    """
    CRITICAL MIGRATION: Move category from feeds table to ticker_feeds table
    This fixes the fundamental architecture flaw where feeds had single categories
    """
    from app import db, LOG

    with db() as conn, conn.cursor() as cur:
        LOG.info("🚀 Starting migration: category per relationship")

        # Step 1: Check if migration is needed
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'feeds' AND column_name = 'category'
        """)
        has_category_in_feeds = cur.fetchone() is not None

        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'ticker_feeds' AND column_name = 'category'
        """)
        has_category_in_ticker_feeds = cur.fetchone() is not None

        if not has_category_in_feeds:
            LOG.info("✅ Migration already complete - no category column in feeds table")
            return

        if has_category_in_ticker_feeds:
            LOG.info("✅ Migration already complete - category column exists in ticker_feeds table")
            return

        # Step 2: Add category column to ticker_feeds
        LOG.info("📝 Adding category column to ticker_feeds table...")
        cur.execute("""
            ALTER TABLE ticker_feeds
            ADD COLUMN IF NOT EXISTS category VARCHAR(20) NOT NULL DEFAULT 'company',
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW()
        """)

        # Step 3: Migrate existing category data
        LOG.info("📦 Migrating category data from feeds to ticker_feeds...")
        cur.execute("""
            UPDATE ticker_feeds tf
            SET category = f.category
            FROM feeds f
            WHERE tf.feed_id = f.id AND f.category IS NOT NULL
        """)

        # Step 4: Add index for category
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ticker_feeds_category ON ticker_feeds(category)")

        # Step 5: Remove category column from feeds table
        LOG.info("🗑️ Removing category column from feeds table...")
        cur.execute("ALTER TABLE feeds DROP COLUMN IF EXISTS category")

        # Step 6: Get migration stats
        cur.execute("SELECT COUNT(*) FROM ticker_feeds")
        total_associations = cur.fetchone()[0]

        cur.execute("SELECT category, COUNT(*) FROM ticker_feeds GROUP BY category ORDER BY category")
        category_stats = cur.fetchall()

        LOG.info(f"✅ Migration complete!")
        LOG.info(f"   Total associations: {total_associations}")
        for stat in category_stats:
            LOG.info(f"   {stat['category']}: {stat['count']} associations")

def upsert_feed_new_architecture(url: str, name: str, search_keyword: str = None,
                                competitor_ticker: str = None, retain_days: int = 90) -> int:
    """Insert/update feed in new architecture - NO CATEGORY (category is per-relationship)"""
    from app import db, LOG

    with db() as conn, conn.cursor() as cur:
        try:
            # Insert or get existing feed - NEVER overwrite existing feeds
            cur.execute("""
                INSERT INTO feeds (url, name, search_keyword, competitor_ticker, retain_days)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET
                    active = TRUE,
                    updated_at = NOW()
                RETURNING id;
            """, (url, name, search_keyword, competitor_ticker, retain_days))

            result = cur.fetchone()
            if result:
                feed_id = result['id']
                LOG.info(f"✅ Feed upserted: {name} (ID: {feed_id})")
                return feed_id
            else:
                raise Exception(f"Failed to upsert feed: {name}")

        except Exception as e:
            # Handle race condition: if feed was created by another process, try to get it
            if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
                LOG.warning(f"⚠️ Concurrent feed creation detected for {url}, attempting to retrieve existing feed")

                # CRITICAL FIX: Use separate transaction for recovery since current one is aborted
                try:
                    conn.rollback()  # End the aborted transaction
                    cur.execute("SELECT id FROM feeds WHERE url = %s", (url,))
                    result = cur.fetchone()
                    if result:
                        feed_id = result['id']
                        LOG.info(f"✅ Retrieved existing feed: {name} (ID: {feed_id})")
                        return feed_id
                except Exception as recovery_error:
                    LOG.error(f"❌ Recovery attempt failed: {recovery_error}")

            # Re-raise if not a concurrency issue or if we couldn't recover
            raise e

def associate_ticker_with_feed(ticker: str, feed_id: int, category: str) -> bool:
    """Associate a ticker with a feed with SPECIFIC CATEGORY for this relationship"""
    from app import db, LOG

    with db() as conn, conn.cursor() as cur:
        try:
            cur.execute("""
                INSERT INTO ticker_feeds (ticker, feed_id, category)
                VALUES (%s, %s, %s)
                ON CONFLICT (ticker, feed_id) DO UPDATE SET
                    category = EXCLUDED.category,
                    active = TRUE,
                    updated_at = NOW()
            """, (ticker, feed_id, category))

            LOG.info(f"✅ Associated ticker {ticker} with feed {feed_id} as category '{category}'")
            return True
        except Exception as e:
            LOG.error(f"❌ Failed to associate ticker {ticker} with feed {feed_id}: {e}")
            return False

def get_feeds_for_ticker(ticker: str) -> list:
    """Get all active feeds for a ticker with their per-relationship categories"""
    from app import db

    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                f.id, f.url, f.name, f.search_keyword, f.competitor_ticker,
                tf.category, tf.active as association_active, tf.created_at as associated_at
            FROM feeds f
            JOIN ticker_feeds tf ON f.id = tf.feed_id
            WHERE tf.ticker = %s AND f.active = TRUE AND tf.active = TRUE
            ORDER BY tf.category, f.name
        """, (ticker,))

        return cur.fetchall()

def create_feeds_for_ticker_new_architecture(ticker: str, metadata: dict) -> list:
    """Create feeds using new many-to-many architecture with per-relationship categories"""
    from app import LOG

    feeds_created = []
    company_name = metadata.get("company_name", ticker)

    LOG.info(f"🔄 Creating feeds for {ticker} using NEW ARCHITECTURE (category-per-relationship)")

    # 1. Company feeds (2 feeds) - will be associated with category="company"
    company_feeds = [
        {
            "url": f"https://news.google.com/rss/search?q=\"{company_name.replace(' ', '%20')}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
            "name": f"Google News: {company_name}",
            "search_keyword": company_name
        },
        {
            "url": f"https://finance.yahoo.com/rss/headline?s={ticker}",
            "name": f"Yahoo Finance: {ticker}",
            "search_keyword": ticker
        }
    ]

    for feed_config in company_feeds:
        try:
            feed_id = upsert_feed_new_architecture(
                url=feed_config["url"],
                name=feed_config["name"],
                search_keyword=feed_config["search_keyword"]
            )

            # Associate this feed with this ticker as "company" category
            if associate_ticker_with_feed(ticker, feed_id, "company"):
                feeds_created.append({
                    "feed_id": feed_id,
                    "config": {"category": "company", "name": feed_config["name"]}
                })

        except Exception as e:
            LOG.error(f"❌ Failed to create company feed for {ticker}: {e}")

    # 2. Industry feeds (up to 3) - will be associated with category="industry"
    industry_keywords = metadata.get("industry_keywords", [])[:3]
    for keyword in industry_keywords:
        try:
            feed_id = upsert_feed_new_architecture(
                url=f"https://news.google.com/rss/search?q=\"{keyword.replace(' ', '%20')}\"+when:7d&hl=en-US&gl=US&ceid=US:en",
                name=f"Industry: {keyword}",
                search_keyword=keyword
            )

            # Associate this feed with this ticker as "industry" category
            if associate_ticker_with_feed(ticker, feed_id, "industry"):
                feeds_created.append({
                    "feed_id": feed_id,
                    "config": {"category": "industry", "keyword": keyword}
                })

        except Exception as e:
            LOG.error(f"❌ Failed to create industry feed for {ticker}: {e}")

    # 3. Competitor feeds (up to 3) - will be associated with category="competitor"
    competitors = metadata.get("competitors", [])[:3]
    for comp in competitors:
        if isinstance(comp, dict) and comp.get('name') and comp.get('ticker'):
            comp_name = comp['name']
            comp_ticker = comp['ticker']

            try:
                # Google News competitor feed - neutral name, shareable
                feed_id = upsert_feed_new_architecture(
                    url=f"https://news.google.com/rss/search?q=\"{comp_name.replace(' ', '%20')}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
                    name=f"Google News: {comp_name}",  # Neutral name (no "Competitor:" prefix)
                    search_keyword=comp_name,
                    competitor_ticker=comp_ticker
                )

                # Associate this feed with this ticker as "competitor" category
                if associate_ticker_with_feed(ticker, feed_id, "competitor"):
                    feeds_created.append({
                        "feed_id": feed_id,
                        "config": {"category": "competitor", "name": comp_name}
                    })

                # Yahoo Finance competitor feed - neutral name, shareable
                feed_id = upsert_feed_new_architecture(
                    url=f"https://finance.yahoo.com/rss/headline?s={comp_ticker}",
                    name=f"Yahoo Finance: {comp_ticker}",  # Neutral name (no "Competitor:" prefix)
                    search_keyword=comp_name,
                    competitor_ticker=comp_ticker
                )

                # Associate this feed with this ticker as "competitor" category
                if associate_ticker_with_feed(ticker, feed_id, "competitor"):
                    feeds_created.append({
                        "feed_id": feed_id,
                        "config": {"category": "competitor", "name": comp_name}
                    })

            except Exception as e:
                LOG.error(f"❌ Failed to create competitor feeds for {ticker}: {e}")

    LOG.info(f"✅ Created {len(feeds_created)} feed associations for {ticker} using NEW ARCHITECTURE (category-per-relationship)")
    return feeds_created