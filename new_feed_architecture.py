# NEW FEED ARCHITECTURE
# This file contains the new many-to-many feed architecture functions

def ensure_new_feed_architecture():
    """Create the new many-to-many feed architecture"""
    from app import db, LOG

    with db() as conn, conn.cursor() as cur:
        # NEW ARCHITECTURE: Feeds table (unique feeds)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feeds (
                id SERIAL PRIMARY KEY,
                url VARCHAR(2048) UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                category VARCHAR(50) DEFAULT 'company',
                search_keyword VARCHAR(255),
                competitor_ticker VARCHAR(10),
                retain_days INTEGER DEFAULT 90,
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)

        # NEW ARCHITECTURE: Ticker-to-Feed mapping (many-to-many)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ticker_feeds (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                feed_id INTEGER NOT NULL REFERENCES feeds(id) ON DELETE CASCADE,
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, feed_id)
            );
        """)

        # Create indexes for new architecture
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feeds_url ON feeds(url);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feeds_category ON feeds(category);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feeds_active ON feeds(active);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ticker_feeds_ticker ON ticker_feeds(ticker);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ticker_feeds_feed_id ON ticker_feeds(feed_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ticker_feeds_active ON ticker_feeds(active);")

        LOG.info("✅ New feed architecture (feeds + ticker_feeds) created successfully")

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

def upsert_feed_new_architecture(url: str, name: str, category: str = "company",
                                search_keyword: str = None, competitor_ticker: str = None,
                                retain_days: int = 90) -> int:
    """Insert/update feed in new architecture and return feed ID"""
    from app import db, LOG

    with db() as conn, conn.cursor() as cur:
        # Insert or get existing feed
        cur.execute("""
            INSERT INTO feeds (url, name, category, search_keyword, competitor_ticker, retain_days)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET
                name = EXCLUDED.name,
                category = EXCLUDED.category,
                search_keyword = EXCLUDED.search_keyword,
                competitor_ticker = EXCLUDED.competitor_ticker,
                active = TRUE,
                updated_at = NOW()
            RETURNING id;
        """, (url, name, category, search_keyword, competitor_ticker, retain_days))

        result = cur.fetchone()
        if result:
            feed_id = result['id']
            LOG.info(f"✅ Feed upserted: {name} (ID: {feed_id})")
            return feed_id
        else:
            raise Exception(f"Failed to upsert feed: {name}")

def associate_ticker_with_feed(ticker: str, feed_id: int) -> bool:
    """Associate a ticker with a feed (many-to-many relationship)"""
    from app import db, LOG

    with db() as conn, conn.cursor() as cur:
        try:
            cur.execute("""
                INSERT INTO ticker_feeds (ticker, feed_id)
                VALUES (%s, %s)
                ON CONFLICT (ticker, feed_id) DO UPDATE SET
                    active = TRUE
            """, (ticker, feed_id))

            LOG.info(f"✅ Associated ticker {ticker} with feed {feed_id}")
            return True
        except Exception as e:
            LOG.error(f"❌ Failed to associate ticker {ticker} with feed {feed_id}: {e}")
            return False

def get_feeds_for_ticker(ticker: str) -> list:
    """Get all active feeds for a ticker using new architecture"""
    from app import db

    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT f.id, f.url, f.name, f.category, f.search_keyword, f.competitor_ticker
            FROM feeds f
            JOIN ticker_feeds tf ON f.id = tf.feed_id
            WHERE tf.ticker = %s AND f.active = TRUE AND tf.active = TRUE
            ORDER BY f.category, f.name
        """, (ticker,))

        return cur.fetchall()

def create_feeds_for_ticker_new_architecture(ticker: str, metadata: dict) -> list:
    """Create feeds using new many-to-many architecture"""
    from app import LOG

    feeds_created = []
    company_name = metadata.get("company_name", ticker)

    LOG.info(f"🔄 Creating feeds for {ticker} using NEW ARCHITECTURE")

    # 1. Company feeds (2 feeds)
    company_feeds = [
        {
            "url": f"https://news.google.com/rss/search?q=\"{company_name.replace(' ', '%20')}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
            "name": f"Google News: {company_name}",
            "category": "company",
            "search_keyword": company_name
        },
        {
            "url": f"https://finance.yahoo.com/rss/headline?s={ticker}",
            "name": f"Yahoo Finance: {ticker}",
            "category": "company",
            "search_keyword": ticker
        }
    ]

    for feed_config in company_feeds:
        try:
            feed_id = upsert_feed_new_architecture(
                url=feed_config["url"],
                name=feed_config["name"],
                category=feed_config["category"],
                search_keyword=feed_config["search_keyword"]
            )

            if associate_ticker_with_feed(ticker, feed_id):
                feeds_created.append({"feed_id": feed_id, "config": feed_config})

        except Exception as e:
            LOG.error(f"❌ Failed to create company feed for {ticker}: {e}")

    # 2. Industry feeds (up to 3)
    industry_keywords = metadata.get("industry_keywords", [])[:3]
    for keyword in industry_keywords:
        try:
            feed_id = upsert_feed_new_architecture(
                url=f"https://news.google.com/rss/search?q=\"{keyword.replace(' ', '%20')}\"+when:7d&hl=en-US&gl=US&ceid=US:en",
                name=f"Industry: {keyword}",
                category="industry",
                search_keyword=keyword
            )

            if associate_ticker_with_feed(ticker, feed_id):
                feeds_created.append({"feed_id": feed_id, "config": {"category": "industry", "keyword": keyword}})

        except Exception as e:
            LOG.error(f"❌ Failed to create industry feed for {ticker}: {e}")

    # 3. Competitor feeds (up to 3)
    competitors = metadata.get("competitors", [])[:3]
    for comp in competitors:
        if isinstance(comp, dict) and comp.get('name') and comp.get('ticker'):
            comp_name = comp['name']
            comp_ticker = comp['ticker']

            try:
                # Google News competitor feed
                feed_id = upsert_feed_new_architecture(
                    url=f"https://news.google.com/rss/search?q=\"{comp_name.replace(' ', '%20')}\"+stock+when:7d&hl=en-US&gl=US&ceid=US:en",
                    name=f"Competitor: {comp_name}",
                    category="competitor",
                    search_keyword=comp_name,
                    competitor_ticker=comp_ticker
                )

                if associate_ticker_with_feed(ticker, feed_id):
                    feeds_created.append({"feed_id": feed_id, "config": {"category": "competitor", "name": comp_name}})

                # Yahoo Finance competitor feed
                feed_id = upsert_feed_new_architecture(
                    url=f"https://finance.yahoo.com/rss/headline?s={comp_ticker}",
                    name=f"Yahoo Competitor: {comp_name} ({comp_ticker})",
                    category="competitor",
                    search_keyword=comp_name,
                    competitor_ticker=comp_ticker
                )

                if associate_ticker_with_feed(ticker, feed_id):
                    feeds_created.append({"feed_id": feed_id, "config": {"category": "competitor", "name": comp_name}})

            except Exception as e:
                LOG.error(f"❌ Failed to create competitor feeds for {ticker}: {e}")

    LOG.info(f"✅ Created {len(feeds_created)} feeds for {ticker} using NEW ARCHITECTURE")
    return feeds_created