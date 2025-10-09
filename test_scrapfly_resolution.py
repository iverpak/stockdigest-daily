#!/usr/bin/env python3
"""
Test script to verify ScrapFly Google News URL resolution

Usage:
    python test_scrapfly_resolution.py
"""
import asyncio
import os
import sys

# Import the resolution function from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import resolve_google_news_url_with_scrapfly

# Get API key from environment
SCRAPFLY_API_KEY = os.getenv("SCRAPFLY_API_KEY")

async def test_resolution():
    """Test ScrapFly resolution with a sample Google News URL"""

    # Your sample Google News URL
    test_url = "https://news.google.com/rss/articles/CBMivwFBVV95cUxQSlUxY1lTYy1TWGVFUXlJeERNbElENVc0SWJVbG1lU283YTVMdFpnd29OZ2g3NHhHRHVNS05PN0cxWWtyMXR1dWxFYmxvNkwyVUNjb3BOcC1IT0FNM3Q1SDhoMzJOVXRYMVFGUHpjTXdwYWwzY1VBZXVYajQxbUpEb0lDMkFMWU9DR2lsU3ZKeDJKbUtYcUpuUXgxZXJlVTlPVWNYeDI2RlB1b283WS1NX1FXeTFfZnZxZ3hiVEpnUQ?oc=5"

    ticker = "TEST"

    print("=" * 80)
    print("TESTING SCRAPFLY GOOGLE NEWS URL RESOLUTION")
    print("=" * 80)
    print()

    # Check if API key is configured
    if not SCRAPFLY_API_KEY:
        print("❌ ERROR: SCRAPFLY_API_KEY not configured in environment")
        print("   Please add it to your .env file or environment variables")
        return

    print(f"✅ ScrapFly API key found: {SCRAPFLY_API_KEY[:20]}...")
    print()
    print(f"🔗 Testing URL:")
    print(f"   {test_url[:120]}...")
    print()

    # Run the resolution
    print("🔄 Attempting resolution...")
    print()

    try:
        resolved_url = await resolve_google_news_url_with_scrapfly(test_url, ticker)

        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)

        if resolved_url:
            print(f"✅ SUCCESS! Resolved to:")
            print(f"   {resolved_url}")
            print()
            print("   This URL can now be scraped with ScrapFly's content extraction!")
        else:
            print("❌ FAILED: Could not resolve URL")
            print("   Check the logs above for details")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_resolution())
