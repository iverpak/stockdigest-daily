#!/bin/bash
# Test Scrapfly extraction on AP News article

echo "🧪 Testing Scrapfly Extraction API"
echo "=================================="
echo ""

# Check if SCRAPFLY_API_KEY is set
if [ -z "$SCRAPFLY_API_KEY" ]; then
    echo "❌ SCRAPFLY_API_KEY not set!"
    echo ""
    echo "💡 To run this test:"
    echo "   export SCRAPFLY_API_KEY='your_key_here'"
    echo "   ./test_scrapfly.sh"
    echo ""
    echo "Or copy this curl command and replace YOUR_KEY:"
    echo ""
fi

# Build the extraction JSON
EXTRACT_JSON='{"article":{"selector":"auto","output":{"title":true,"text":true,"date":true,"author":true}}}'

# Test URL
TEST_URL="https://apnews.com/article/peru-mercury-illegal-gold-mining-trade-bloc-ruling-indigenous-1459cd455bc15d0ab228bc75476752b7"

echo "📍 URL: $TEST_URL"
echo "📍 Domain: apnews.com"
echo ""
echo "🚀 Making request to Scrapfly..."
echo "=================================="
echo ""

# Make the request
curl -s "https://api.scrapfly.io/scrape" \
  -G \
  --data-urlencode "key=${SCRAPFLY_API_KEY:-YOUR_KEY_HERE}" \
  --data-urlencode "url=$TEST_URL" \
  --data-urlencode "extract=$EXTRACT_JSON" \
  --data-urlencode "country=us" \
  | jq '.' || cat

echo ""
echo ""
echo "=================================="
echo "✅ Response received (see above)"
