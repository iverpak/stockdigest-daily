#!/usr/bin/env python3
"""Quick test script to debug Scrapfly scraping on a single URL"""
import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from main app
from app import scrape_with_scrapfly_async, SCRAPFLY_API_KEY

async def test_url(url: str, api_key: str = None):
    """Test Scrapfly scraping on a single URL"""
    print(f"🧪 Testing Scrapfly on: {url}\n")

    # Extract domain
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.replace("www.", "")
    print(f"📍 Domain: {domain}\n")

    # Check if API key is provided or in env
    import json
    test_key = api_key or SCRAPFLY_API_KEY

    if not test_key:
        print("❌ ERROR: SCRAPFLY_API_KEY not set!")
        print("\n💡 To test locally, you can:")
        print("   1. Set SCRAPFLY_API_KEY environment variable")
        print("   2. Or view the Scrapfly request parameters below:\n")

        # Show what parameters would be sent
        params = {
            "key": "YOUR_API_KEY",
            "url": url,
            "extract": json.dumps({
                "article": {
                    "selector": "auto",
                    "output": {
                        "title": True,
                        "text": True,
                        "date": True,
                        "author": True
                    }
                }
            }),
            "country": "us",
        }

        print("📤 Request Parameters:")
        print(json.dumps(params, indent=2))
        print(f"\n🔗 API Endpoint: https://api.scrapfly.io/scrape")
        print(f"\n💡 Test this URL manually at: https://scrapfly.io/dashboard/playground")
        return

    print(f"🔑 API Key: {test_key[:10]}...{test_key[-4:]}\n")
    print("🚀 Calling Scrapfly...\n")
    print("=" * 80)

    # Temporarily set the key
    import app
    original_key = app.SCRAPFLY_API_KEY
    app.SCRAPFLY_API_KEY = test_key

    try:
        # Call the scraping function
        content, error = await scrape_with_scrapfly_async(url, domain)

        print("=" * 80)
        print("\n📊 RESULTS:\n")

        if content:
            print(f"✅ SUCCESS!")
            print(f"   Content length: {len(content)} characters")
            print(f"   Preview (first 500 chars):")
            print(f"   {'-' * 80}")
            print(f"   {content[:500]}")
            print(f"   {'-' * 80}")
        else:
            print(f"❌ FAILED!")
            print(f"   Error: {error}")
    finally:
        app.SCRAPFLY_API_KEY = original_key

if __name__ == "__main__":
    # Test URL
    test_url_str = "https://apnews.com/article/peru-mercury-illegal-gold-mining-trade-bloc-ruling-indigenous-1459cd455bc15d0ab228bc75476752b7"

    # Run async test
    asyncio.run(test_url(test_url_str))
