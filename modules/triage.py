"""
Triage Module

Handles article triage using Claude Sonnet 4.5 (primary) with Gemini Flash 2.5 (fallback).
Extracted from app.py for better modularity and easier prompt management.

Architecture:
- Claude Sonnet 4.5: Primary provider (with prompt caching)
- Gemini Flash 2.5: Fallback provider (cheaper, faster)
- Parallel execution: Both run simultaneously, merge results
- Smart fallback: Use Gemini if Claude fails
- Retry logic: 3 retries with exponential backoff

Categories:
1. Company articles (ticker-specific news)
2. Industry articles (fundamental drivers)
3. Competitor articles (competitive intelligence)
4. Upstream articles (supplier intelligence)
5. Downstream articles (customer intelligence)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai

LOG = logging.getLogger(__name__)


# ============================================================================
# PROMPT LOADERS
# ============================================================================

def load_prompt(prompt_file: str) -> str:
    """Load prompt from modules/ directory using relative path"""
    try:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(module_dir, prompt_file)
        with open(prompt_path, "r") as f:
            return f.read()
    except Exception as e:
        LOG.error(f"Failed to load prompt {prompt_file}: {e}")
        raise


# Load all triage prompts at module initialization
COMPANY_PROMPT = load_prompt("_triage_company_prompt")
INDUSTRY_PROMPT = load_prompt("_triage_industry_prompt")
COMPETITOR_PROMPT = load_prompt("_triage_competitor_prompt")
UPSTREAM_PROMPT = load_prompt("_triage_upstream_prompt")
DOWNSTREAM_PROMPT = load_prompt("_triage_downstream_prompt")


# ============================================================================
# RETRY LOGIC
# ============================================================================

def should_retry(exception: Exception, status_code: Optional[int] = None) -> bool:
    """Determine if we should retry based on exception or status code"""
    # Retry on HTTP 429 (rate limit), 500, 503
    if status_code in [429, 500, 503]:
        return True

    # Retry on timeout errors
    if "timeout" in str(exception).lower():
        return True

    # Retry on network errors
    if "connection" in str(exception).lower():
        return True

    return False


# ============================================================================
# CLAUDE TRIAGE FUNCTIONS (PRIMARY)
# ============================================================================

async def triage_company_articles_claude_async(
    articles: List[Dict],
    ticker: str,
    company_name: str,
    anthropic_api_key: str
) -> Tuple[Optional[List[Dict]], str, Optional[dict]]:
    """Claude-based company article triage

    Returns:
        Tuple[Optional[List[Dict]], str, Optional[dict]]: (results, provider, usage) where:
            - results: List of selected articles with scrape_priority and why
            - provider: "Claude" or "failed"
            - usage: {"input_tokens": X, "output_tokens": Y, "cache_creation_input_tokens": Z, "cache_read_input_tokens": W} or None
    """
    if not anthropic_api_key or not articles:
        return None, "failed", None

    # Prepare items for triage
    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(20, len(articles))
    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            # Import here to avoid circular dependency
            from anthropic import Anthropic

            client = Anthropic(api_key=anthropic_api_key)

            # User message with ticker-specific context
            user_content = f"""**TARGET COMPANY:** {company_name} ({ticker})
**ARTICLE COUNT:** {len(articles)}
**TARGET CAP:** {target_cap}

**YOUR TASK:**
Select the {target_cap} most important articles about {company_name} from the {len(articles)} candidates below.

**ARTICLES:**
{json.dumps(items, indent=2)}"""

            # Call Claude with prompt caching
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                temperature=0.0,
                system=[{
                    "type": "text",
                    "text": COMPANY_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }],
                messages=[{
                    "role": "user",
                    "content": user_content
                }]
            )

            # Extract JSON from response
            response_text = response.content[0].text

            # Try to parse JSON (handle both direct JSON and markdown-wrapped JSON)
            json_match = None
            if response_text.strip().startswith('['):
                json_match = response_text.strip()
            else:
                import re
                json_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
                match = re.search(json_pattern, response_text, re.DOTALL)
                if match:
                    json_match = match.group(1)

            if json_match:
                results = json.loads(json_match)

                # Extract usage metadata
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0)
                }

                LOG.info(f"✅ Claude company triage: {ticker} selected {len(results)}/{len(articles)} articles")
                return results, "Claude", usage
            else:
                LOG.error(f"Claude returned non-JSON response for {ticker} company triage")
                return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Claude company triage attempt {attempt + 1} failed for {ticker}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                LOG.error(f"Claude company triage failed for {ticker} after {attempt + 1} attempts: {e}")
                return None, "failed", None

    return None, "failed", None


async def triage_industry_articles_claude_async(
    articles: List[Dict],
    ticker: str,
    company_name: str,
    sector: str,
    peers: List[str],
    industry_keywords: List[str],
    geographic_markets: str,
    anthropic_api_key: str
) -> Tuple[Optional[List[Dict]], str, Optional[dict]]:
    """Claude-based industry article triage (fundamental drivers)"""
    if not anthropic_api_key or not articles:
        return None, "failed", None

    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(8, len(articles))
    driver_keywords_display = ', '.join(industry_keywords) if industry_keywords else 'Not configured'
    peers_display = ', '.join(peers[:5]) if peers else 'None'
    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=anthropic_api_key)

            user_content = f"""**TARGET COMPANY:** {company_name} ({ticker})
**FUNDAMENTAL DRIVER KEYWORDS:** {driver_keywords_display}
**SECTOR:** {sector}
**KNOWN PEERS:** {peers_display}
**GEOGRAPHIC MARKETS:** {geographic_markets if geographic_markets else 'Unknown'}
**ARTICLE COUNT:** {len(articles)}
**TARGET CAP:** {target_cap}

**YOUR TASK:**
Select up to {target_cap} articles most likely to contain quantifiable fundamental driver data from the {len(articles)} candidates below.

**ARTICLES:**
{json.dumps(items, indent=2)}"""

            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                temperature=0.0,
                system=[{
                    "type": "text",
                    "text": INDUSTRY_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }],
                messages=[{
                    "role": "user",
                    "content": user_content
                }]
            )

            response_text = response.content[0].text

            json_match = None
            if response_text.strip().startswith('['):
                json_match = response_text.strip()
            else:
                import re
                json_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
                match = re.search(json_pattern, response_text, re.DOTALL)
                if match:
                    json_match = match.group(1)

            if json_match:
                results = json.loads(json_match)

                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0)
                }

                LOG.info(f"✅ Claude industry triage: {ticker} selected {len(results)}/{len(articles)} articles")
                return results, "Claude", usage
            else:
                LOG.error(f"Claude returned non-JSON response for {ticker} industry triage")
                return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Claude industry triage attempt {attempt + 1} failed for {ticker}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                LOG.error(f"Claude industry triage failed for {ticker} after {attempt + 1} attempts: {e}")
                return None, "failed", None

    return None, "failed", None


async def triage_competitor_articles_claude_async(
    articles: List[Dict],
    ticker: str,
    competitor_name: str,
    competitor_ticker: str,
    anthropic_api_key: str
) -> Tuple[Optional[List[Dict]], str, Optional[dict]]:
    """Claude-based competitor article triage"""
    if not anthropic_api_key or not articles:
        return None, "failed", None

    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(5, len(articles))
    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=anthropic_api_key)

            user_content = f"""**TARGET COMPANY TICKER:** {ticker}
**COMPETITOR:** {competitor_name} ({competitor_ticker})
**ARTICLE COUNT:** {len(articles)}
**TARGET CAP:** {target_cap}

**YOUR TASK:**
Select the {target_cap} most important articles about {competitor_name} from the {len(articles)} candidates below.

**ARTICLES:**
{json.dumps(items, indent=2)}"""

            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                temperature=0.0,
                system=[{
                    "type": "text",
                    "text": COMPETITOR_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }],
                messages=[{
                    "role": "user",
                    "content": user_content
                }]
            )

            response_text = response.content[0].text

            json_match = None
            if response_text.strip().startswith('['):
                json_match = response_text.strip()
            else:
                import re
                json_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
                match = re.search(json_pattern, response_text, re.DOTALL)
                if match:
                    json_match = match.group(1)

            if json_match:
                results = json.loads(json_match)

                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0)
                }

                LOG.info(f"✅ Claude competitor triage: {ticker}/{competitor_ticker} selected {len(results)}/{len(articles)} articles")
                return results, "Claude", usage
            else:
                LOG.error(f"Claude returned non-JSON response for {ticker} competitor triage")
                return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Claude competitor triage attempt {attempt + 1} failed for {ticker}/{competitor_ticker}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                LOG.error(f"Claude competitor triage failed for {ticker}/{competitor_ticker} after {attempt + 1} attempts: {e}")
                return None, "failed", None

    return None, "failed", None


async def triage_upstream_articles_claude_async(
    articles: List[Dict],
    ticker: str,
    company_name: str,
    upstream_company_name: str,
    upstream_ticker: str,
    anthropic_api_key: str
) -> Tuple[Optional[List[Dict]], str, Optional[dict]]:
    """Claude-based upstream supplier article triage"""
    if not anthropic_api_key or not articles:
        return None, "failed", None

    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(5, len(articles))
    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=anthropic_api_key)

            user_content = f"""**TARGET COMPANY:** {company_name} ({ticker})
**UPSTREAM SUPPLIER:** {upstream_company_name} ({upstream_ticker})
**ARTICLE COUNT:** {len(articles)}
**TARGET CAP:** {target_cap}

**YOUR TASK:**
Select the {target_cap} highest-quality articles about {upstream_company_name} from the {len(articles)} candidates below.

**ARTICLES:**
{json.dumps(items, indent=2)}"""

            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                temperature=0.0,
                system=[{
                    "type": "text",
                    "text": UPSTREAM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }],
                messages=[{
                    "role": "user",
                    "content": user_content
                }]
            )

            response_text = response.content[0].text

            json_match = None
            if response_text.strip().startswith('['):
                json_match = response_text.strip()
            else:
                import re
                json_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
                match = re.search(json_pattern, response_text, re.DOTALL)
                if match:
                    json_match = match.group(1)

            if json_match:
                results = json.loads(json_match)

                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0)
                }

                LOG.info(f"✅ Claude upstream triage: {ticker}/{upstream_ticker} selected {len(results)}/{len(articles)} articles")
                return results, "Claude", usage
            else:
                LOG.error(f"Claude returned non-JSON response for {ticker} upstream triage")
                return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Claude upstream triage attempt {attempt + 1} failed for {ticker}/{upstream_ticker}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                LOG.error(f"Claude upstream triage failed for {ticker}/{upstream_ticker} after {attempt + 1} attempts: {e}")
                return None, "failed", None

    return None, "failed", None


async def triage_downstream_articles_claude_async(
    articles: List[Dict],
    ticker: str,
    company_name: str,
    downstream_company_name: str,
    downstream_ticker: str,
    anthropic_api_key: str
) -> Tuple[Optional[List[Dict]], str, Optional[dict]]:
    """Claude-based downstream customer article triage"""
    if not anthropic_api_key or not articles:
        return None, "failed", None

    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(5, len(articles))
    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=anthropic_api_key)

            user_content = f"""**TARGET COMPANY:** {company_name} ({ticker})
**DOWNSTREAM CUSTOMER:** {downstream_company_name} ({downstream_ticker})
**ARTICLE COUNT:** {len(articles)}
**TARGET CAP:** {target_cap}

**YOUR TASK:**
Select the {target_cap} highest-quality articles about {downstream_company_name} from the {len(articles)} candidates below.

**ARTICLES:**
{json.dumps(items, indent=2)}"""

            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                temperature=0.0,
                system=[{
                    "type": "text",
                    "text": DOWNSTREAM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }],
                messages=[{
                    "role": "user",
                    "content": user_content
                }]
            )

            response_text = response.content[0].text

            json_match = None
            if response_text.strip().startswith('['):
                json_match = response_text.strip()
            else:
                import re
                json_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
                match = re.search(json_pattern, response_text, re.DOTALL)
                if match:
                    json_match = match.group(1)

            if json_match:
                results = json.loads(json_match)

                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0)
                }

                LOG.info(f"✅ Claude downstream triage: {ticker}/{downstream_ticker} selected {len(results)}/{len(articles)} articles")
                return results, "Claude", usage
            else:
                LOG.error(f"Claude returned non-JSON response for {ticker} downstream triage")
                return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Claude downstream triage attempt {attempt + 1} failed for {ticker}/{downstream_ticker}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                LOG.error(f"Claude downstream triage failed for {ticker}/{downstream_ticker} after {attempt + 1} attempts: {e}")
                return None, "failed", None

    return None, "failed", None


# ============================================================================
# GEMINI TRIAGE FUNCTIONS (FALLBACK)
# ============================================================================

async def triage_company_articles_gemini_async(
    articles: List[Dict],
    ticker: str,
    company_name: str,
    gemini_api_key: str
) -> Tuple[Optional[List[Dict]], str, Optional[dict]]:
    """Gemini-based company article triage

    Returns:
        Tuple[Optional[List[Dict]], str, Optional[dict]]: (results, provider, usage)
    """
    if not gemini_api_key or not articles:
        return None, "failed", None

    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(20, len(articles))
    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=gemini_api_key)

            user_content = f"""**TARGET COMPANY:** {company_name} ({ticker})
**ARTICLE COUNT:** {len(articles)}
**TARGET CAP:** {target_cap}

**YOUR TASK:**
Select the {target_cap} most important articles about {company_name} from the {len(articles)} candidates below.

**ARTICLES:**
{json.dumps(items, indent=2)}

**CRITICAL:** Return ONLY a JSON array. No markdown, no explanations."""

            model = genai.GenerativeModel('gemini-2.5-flash')

            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 16384,
                "response_mime_type": "application/json"
            }

            full_prompt = COMPANY_PROMPT + "\n\n" + user_content

            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            # Parse JSON response
            results = json.loads(response.text)

            # Extract usage metadata
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }

            LOG.info(f"✅ Gemini company triage: {ticker} selected {len(results)}/{len(articles)} articles")
            return results, "Gemini", usage

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Gemini company triage attempt {attempt + 1} failed for {ticker}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                LOG.error(f"Gemini company triage failed for {ticker} after {attempt + 1} attempts: {e}")
                return None, "failed", None

    return None, "failed", None


async def triage_industry_articles_gemini_async(
    articles: List[Dict],
    ticker: str,
    company_name: str,
    sector: str,
    peers: List[str],
    industry_keywords: List[str],
    geographic_markets: str,
    gemini_api_key: str
) -> Tuple[Optional[List[Dict]], str, Optional[dict]]:
    """Gemini-based industry article triage (fundamental drivers)"""
    if not gemini_api_key or not articles:
        return None, "failed", None

    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(8, len(articles))
    driver_keywords_display = ', '.join(industry_keywords) if industry_keywords else 'Not configured'
    peers_display = ', '.join(peers[:5]) if peers else 'None'
    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=gemini_api_key)

            user_content = f"""**TARGET COMPANY:** {company_name} ({ticker})
**FUNDAMENTAL DRIVER KEYWORDS:** {driver_keywords_display}
**SECTOR:** {sector}
**KNOWN PEERS:** {peers_display}
**GEOGRAPHIC MARKETS:** {geographic_markets if geographic_markets else 'Unknown'}
**ARTICLE COUNT:** {len(articles)}
**TARGET CAP:** {target_cap}

**YOUR TASK:**
Select up to {target_cap} articles most likely to contain quantifiable fundamental driver data from the {len(articles)} candidates below.

**ARTICLES:**
{json.dumps(items, indent=2)}

**CRITICAL:** Return ONLY a JSON array. No markdown, no explanations."""

            model = genai.GenerativeModel('gemini-2.5-flash')

            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 16384,
                "response_mime_type": "application/json"
            }

            full_prompt = INDUSTRY_PROMPT + "\n\n" + user_content

            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            results = json.loads(response.text)

            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }

            LOG.info(f"✅ Gemini industry triage: {ticker} selected {len(results)}/{len(articles)} articles")
            return results, "Gemini", usage

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Gemini industry triage attempt {attempt + 1} failed for {ticker}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                LOG.error(f"Gemini industry triage failed for {ticker} after {attempt + 1} attempts: {e}")
                return None, "failed", None

    return None, "failed", None


async def triage_competitor_articles_gemini_async(
    articles: List[Dict],
    ticker: str,
    competitor_name: str,
    competitor_ticker: str,
    gemini_api_key: str
) -> Tuple[Optional[List[Dict]], str, Optional[dict]]:
    """Gemini-based competitor article triage"""
    if not gemini_api_key or not articles:
        return None, "failed", None

    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(5, len(articles))
    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=gemini_api_key)

            user_content = f"""**TARGET COMPANY TICKER:** {ticker}
**COMPETITOR:** {competitor_name} ({competitor_ticker})
**ARTICLE COUNT:** {len(articles)}
**TARGET CAP:** {target_cap}

**YOUR TASK:**
Select the {target_cap} most important articles about {competitor_name} from the {len(articles)} candidates below.

**ARTICLES:**
{json.dumps(items, indent=2)}

**CRITICAL:** Return ONLY a JSON array. No markdown, no explanations."""

            model = genai.GenerativeModel('gemini-2.5-flash')

            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 16384,
                "response_mime_type": "application/json"
            }

            full_prompt = COMPETITOR_PROMPT + "\n\n" + user_content

            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            results = json.loads(response.text)

            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }

            LOG.info(f"✅ Gemini competitor triage: {ticker}/{competitor_ticker} selected {len(results)}/{len(articles)} articles")
            return results, "Gemini", usage

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Gemini competitor triage attempt {attempt + 1} failed for {ticker}/{competitor_ticker}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                LOG.error(f"Gemini competitor triage failed for {ticker}/{competitor_ticker} after {attempt + 1} attempts: {e}")
                return None, "failed", None

    return None, "failed", None


async def triage_upstream_articles_gemini_async(
    articles: List[Dict],
    ticker: str,
    company_name: str,
    upstream_company_name: str,
    upstream_ticker: str,
    gemini_api_key: str
) -> Tuple[Optional[List[Dict]], str, Optional[dict]]:
    """Gemini-based upstream supplier article triage"""
    if not gemini_api_key or not articles:
        return None, "failed", None

    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(5, len(articles))
    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=gemini_api_key)

            user_content = f"""**TARGET COMPANY:** {company_name} ({ticker})
**UPSTREAM SUPPLIER:** {upstream_company_name} ({upstream_ticker})
**ARTICLE COUNT:** {len(articles)}
**TARGET CAP:** {target_cap}

**YOUR TASK:**
Select the {target_cap} highest-quality articles about {upstream_company_name} from the {len(articles)} candidates below.

**ARTICLES:**
{json.dumps(items, indent=2)}

**CRITICAL:** Return ONLY a JSON array. No markdown, no explanations."""

            model = genai.GenerativeModel('gemini-2.5-flash')

            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 16384,
                "response_mime_type": "application/json"
            }

            full_prompt = UPSTREAM_PROMPT + "\n\n" + user_content

            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            results = json.loads(response.text)

            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }

            LOG.info(f"✅ Gemini upstream triage: {ticker}/{upstream_ticker} selected {len(results)}/{len(articles)} articles")
            return results, "Gemini", usage

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Gemini upstream triage attempt {attempt + 1} failed for {ticker}/{upstream_ticker}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                LOG.error(f"Gemini upstream triage failed for {ticker}/{upstream_ticker} after {attempt + 1} attempts: {e}")
                return None, "failed", None

    return None, "failed", None


async def triage_downstream_articles_gemini_async(
    articles: List[Dict],
    ticker: str,
    company_name: str,
    downstream_company_name: str,
    downstream_ticker: str,
    gemini_api_key: str
) -> Tuple[Optional[List[Dict]], str, Optional[dict]]:
    """Gemini-based downstream customer article triage"""
    if not gemini_api_key or not articles:
        return None, "failed", None

    items = []
    for i, article in enumerate(articles):
        item = {
            "id": i,
            "title": article.get('title', ''),
            "published": article.get('published', '')
        }
        if article.get('description'):
            item["description"] = article.get('description')
        items.append(item)

    target_cap = min(5, len(articles))
    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=gemini_api_key)

            user_content = f"""**TARGET COMPANY:** {company_name} ({ticker})
**DOWNSTREAM CUSTOMER:** {downstream_company_name} ({downstream_ticker})
**ARTICLE COUNT:** {len(articles)}
**TARGET CAP:** {target_cap}

**YOUR TASK:**
Select the {target_cap} highest-quality articles about {downstream_company_name} from the {len(articles)} candidates below.

**ARTICLES:**
{json.dumps(items, indent=2)}

**CRITICAL:** Return ONLY a JSON array. No markdown, no explanations."""

            model = genai.GenerativeModel('gemini-2.5-flash')

            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 16384,
                "response_mime_type": "application/json"
            }

            full_prompt = DOWNSTREAM_PROMPT + "\n\n" + user_content

            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            results = json.loads(response.text)

            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }

            LOG.info(f"✅ Gemini downstream triage: {ticker}/{downstream_ticker} selected {len(results)}/{len(articles)} articles")
            return results, "Gemini", usage

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Gemini downstream triage attempt {attempt + 1} failed for {ticker}/{downstream_ticker}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                LOG.error(f"Gemini downstream triage failed for {ticker}/{downstream_ticker} after {attempt + 1} attempts: {e}")
                return None, "failed", None

    return None, "failed", None


# ============================================================================
# MERGE FUNCTION (reused from app.py with Gemini labels)
# ============================================================================

async def merge_triage_scores(
    claude_results: List[Dict],
    gemini_results: List[Dict],
    articles: List[Dict],
    target_cap: int,
    category_type: str,
    category_key: str,
    problematic_domains: set
) -> List[Dict]:
    """
    Merge Claude and Gemini triage results by combined score.
    Returns top N articles by combined score (claude_score + gemini_score).
    Handles fallback if one API fails (use single API's selections).
    """
    # Build URL-based lookup for matching articles across APIs
    url_scores = {}  # url -> {"claude": score, "gemini": score, "article": article_obj}

    # Process Claude results
    for result in claude_results:
        article_id = result["id"]
        if article_id < len(articles):
            article = articles[article_id]
            url = article.get("url", "")
            if url:
                if url not in url_scores:
                    url_scores[url] = {"claude": 0, "gemini": 0, "article": article, "id": article_id}
                # Score: 1=high, 2=medium, 3=low -> convert to reverse (high=3, low=1)
                priority = result.get("scrape_priority", 2)
                url_scores[url]["claude"] = 4 - priority  # 3, 2, 1
                url_scores[url]["why_claude"] = result.get("why", "")

    # Process Gemini results
    for result in gemini_results:
        article_id = result["id"]
        if article_id < len(articles):
            article = articles[article_id]
            url = article.get("url", "")
            if url:
                if url not in url_scores:
                    url_scores[url] = {"claude": 0, "gemini": 0, "article": article, "id": article_id}
                priority = result.get("scrape_priority", 2)
                url_scores[url]["gemini"] = 4 - priority
                url_scores[url]["why_gemini"] = result.get("why", "")

    # Track stats for logging
    total_unique_before_filter = len(url_scores)

    # Filter out problematic domains BEFORE ranking
    filtered_scores = {}
    blocked_count = 0
    for url, data in url_scores.items():
        article = data["article"]
        domain = article.get("domain", "")
        if domain in problematic_domains:
            blocked_count += 1
            LOG.debug(f"Blocked {domain} from triage selection (problematic domain)")
            continue
        filtered_scores[url] = data

    # Calculate combined scores and sort
    scored_articles = []
    for url, data in filtered_scores.items():
        combined_score = data["claude"] + data["gemini"]
        if combined_score > 0:  # At least one API selected it
            scored_articles.append({
                "url": url,
                "article": data["article"],
                "id": data["id"],
                "claude_score": data["claude"],
                "gemini_score": data["gemini"],
                "combined_score": combined_score,
                "why_claude": data.get("why_claude", ""),
                "why_gemini": data.get("why_gemini", "")
            })

    # Sort by combined score (descending), then by timestamp (recent first)
    scored_articles.sort(key=lambda x: (
        -x["combined_score"],
        -x["article"].get("published_at", datetime.min).timestamp() if x["article"].get("published_at") else 0
    ))

    # Take top N
    top_articles = scored_articles[:target_cap]

    # Enhanced logging with domain filter transparency
    if blocked_count > 0:
        LOG.info(f"  Dual scoring {category_type}/{category_key}: {len(claude_results)} Claude + {len(gemini_results)} Gemini = {total_unique_before_filter} unique ({blocked_count} blocked by domain filter) → {len(scored_articles)} remaining → top {len(top_articles)}")
    else:
        LOG.info(f"  Dual scoring {category_type}/{category_key}: {len(claude_results)} Claude + {len(gemini_results)} Gemini = {len(scored_articles)} unique → top {len(top_articles)}")

    # Return in format expected by downstream code
    result = []
    for item in top_articles:
        result.append({
            "id": item["id"],
            "scrape_priority": 1 if item["combined_score"] >= 5 else (2 if item["combined_score"] >= 3 else 3),
            "why": f"Claude: {item['why_claude'][:50]}... Gemini: {item['why_gemini'][:50]}..." if item['why_claude'] and item['why_gemini'] else (item['why_claude'] or item['why_gemini']),
            "confidence": 0.9 if (item["claude_score"] > 0 and item["gemini_score"] > 0) else 0.7,
            "likely_repeat": False,
            "repeat_key": "",
            "claude_score": item["claude_score"],
            "gemini_score": item["gemini_score"],
            "combined_score": item["combined_score"]
        })

    return result


# ============================================================================
# ORCHESTRATOR FUNCTIONS (PARALLEL WITH SMART FALLBACK)
# ============================================================================

async def triage_company_articles_dual(
    articles: List[Dict],
    ticker: str,
    company_name: str,
    anthropic_api_key: str,
    gemini_api_key: str,
    problematic_domains: set
) -> Tuple[List[Dict], str, dict]:
    """Orchestrator for company triage with parallel execution and smart fallback

    Returns:
        Tuple[List[Dict], str, dict]: (results, provider_name, usage_metadata) where:
            - results: Merged triage results
            - provider_name: 'dual', 'claude_only', or 'gemini_fallback'
            - usage_metadata: {'claude': {...}, 'gemini': {...}}
    """
    target_cap = min(20, len(articles))

    # Run both in parallel
    claude_task = triage_company_articles_claude_async(articles, ticker, company_name, anthropic_api_key)
    gemini_task = triage_company_articles_gemini_async(articles, ticker, company_name, gemini_api_key)

    claude_result, gemini_result = await asyncio.gather(claude_task, gemini_task, return_exceptions=True)

    # Unpack results
    claude_data, claude_provider, claude_usage = claude_result if not isinstance(claude_result, Exception) else (None, "failed", None)
    gemini_data, gemini_provider, gemini_usage = gemini_result if not isinstance(gemini_result, Exception) else (None, "failed", None)

    # Smart fallback logic
    if claude_data is None and gemini_data is not None:
        # Claude failed, Gemini succeeded → use Gemini
        LOG.info(f"⚠️ Claude failed, using Gemini fallback for {ticker} company triage")
        return gemini_data, 'gemini_fallback', {'gemini': gemini_usage}

    elif claude_data is not None and gemini_data is not None:
        # Both succeeded → merge and return both scores
        merged = await merge_triage_scores(
            claude_results=claude_data,
            gemini_results=gemini_data,
            articles=articles,
            target_cap=target_cap,
            category_type="company",
            category_key=ticker,
            problematic_domains=problematic_domains
        )
        LOG.info(f"✅ Dual scoring succeeded for {ticker} company triage")
        return merged, 'dual', {'claude': claude_usage, 'gemini': gemini_usage}

    elif claude_data is not None and gemini_data is None:
        # Only Claude succeeded
        LOG.info(f"✅ Claude only for {ticker} company triage (Gemini failed)")
        return claude_data, 'claude_only', {'claude': claude_usage}

    else:
        # Both failed
        LOG.error(f"❌ Both Claude and Gemini failed for {ticker} company triage")
        raise Exception(f"Both Claude and Gemini triage failed for {ticker}")


async def triage_industry_articles_dual(
    articles: List[Dict],
    ticker: str,
    company_name: str,
    sector: str,
    peers: List[str],
    industry_keywords: List[str],
    geographic_markets: str,
    anthropic_api_key: str,
    gemini_api_key: str,
    problematic_domains: set
) -> Tuple[List[Dict], str, dict]:
    """Orchestrator for industry triage with parallel execution and smart fallback"""
    target_cap = min(8, len(articles))

    claude_task = triage_industry_articles_claude_async(
        articles, ticker, company_name, sector, peers, industry_keywords, geographic_markets, anthropic_api_key
    )
    gemini_task = triage_industry_articles_gemini_async(
        articles, ticker, company_name, sector, peers, industry_keywords, geographic_markets, gemini_api_key
    )

    claude_result, gemini_result = await asyncio.gather(claude_task, gemini_task, return_exceptions=True)

    claude_data, claude_provider, claude_usage = claude_result if not isinstance(claude_result, Exception) else (None, "failed", None)
    gemini_data, gemini_provider, gemini_usage = gemini_result if not isinstance(gemini_result, Exception) else (None, "failed", None)

    if claude_data is None and gemini_data is not None:
        LOG.info(f"⚠️ Claude failed, using Gemini fallback for {ticker} industry triage")
        return gemini_data, 'gemini_fallback', {'gemini': gemini_usage}

    elif claude_data is not None and gemini_data is not None:
        merged = await merge_triage_scores(
            claude_results=claude_data,
            gemini_results=gemini_data,
            articles=articles,
            target_cap=target_cap,
            category_type="industry",
            category_key=ticker,
            problematic_domains=problematic_domains
        )
        LOG.info(f"✅ Dual scoring succeeded for {ticker} industry triage")
        return merged, 'dual', {'claude': claude_usage, 'gemini': gemini_usage}

    elif claude_data is not None and gemini_data is None:
        LOG.info(f"✅ Claude only for {ticker} industry triage (Gemini failed)")
        return claude_data, 'claude_only', {'claude': claude_usage}

    else:
        LOG.error(f"❌ Both Claude and Gemini failed for {ticker} industry triage")
        raise Exception(f"Both Claude and Gemini triage failed for {ticker}")


async def triage_competitor_articles_dual(
    articles: List[Dict],
    ticker: str,
    competitor_name: str,
    competitor_ticker: str,
    anthropic_api_key: str,
    gemini_api_key: str,
    problematic_domains: set
) -> Tuple[List[Dict], str, dict]:
    """Orchestrator for competitor triage with parallel execution and smart fallback"""
    target_cap = min(5, len(articles))

    claude_task = triage_competitor_articles_claude_async(
        articles, ticker, competitor_name, competitor_ticker, anthropic_api_key
    )
    gemini_task = triage_competitor_articles_gemini_async(
        articles, ticker, competitor_name, competitor_ticker, gemini_api_key
    )

    claude_result, gemini_result = await asyncio.gather(claude_task, gemini_task, return_exceptions=True)

    claude_data, claude_provider, claude_usage = claude_result if not isinstance(claude_result, Exception) else (None, "failed", None)
    gemini_data, gemini_provider, gemini_usage = gemini_result if not isinstance(gemini_result, Exception) else (None, "failed", None)

    if claude_data is None and gemini_data is not None:
        LOG.info(f"⚠️ Claude failed, using Gemini fallback for {ticker}/{competitor_ticker} competitor triage")
        return gemini_data, 'gemini_fallback', {'gemini': gemini_usage}

    elif claude_data is not None and gemini_data is not None:
        merged = await merge_triage_scores(
            claude_results=claude_data,
            gemini_results=gemini_data,
            articles=articles,
            target_cap=target_cap,
            category_type="competitor",
            category_key=competitor_ticker,
            problematic_domains=problematic_domains
        )
        LOG.info(f"✅ Dual scoring succeeded for {ticker}/{competitor_ticker} competitor triage")
        return merged, 'dual', {'claude': claude_usage, 'gemini': gemini_usage}

    elif claude_data is not None and gemini_data is None:
        LOG.info(f"✅ Claude only for {ticker}/{competitor_ticker} competitor triage (Gemini failed)")
        return claude_data, 'claude_only', {'claude': claude_usage}

    else:
        LOG.error(f"❌ Both Claude and Gemini failed for {ticker}/{competitor_ticker} competitor triage")
        raise Exception(f"Both Claude and Gemini triage failed for {ticker}/{competitor_ticker}")


async def triage_upstream_articles_dual(
    articles: List[Dict],
    ticker: str,
    company_name: str,
    upstream_company_name: str,
    upstream_ticker: str,
    anthropic_api_key: str,
    gemini_api_key: str,
    problematic_domains: set
) -> Tuple[List[Dict], str, dict]:
    """Orchestrator for upstream triage with parallel execution and smart fallback"""
    target_cap = min(5, len(articles))

    claude_task = triage_upstream_articles_claude_async(
        articles, ticker, company_name, upstream_company_name, upstream_ticker, anthropic_api_key
    )
    gemini_task = triage_upstream_articles_gemini_async(
        articles, ticker, company_name, upstream_company_name, upstream_ticker, gemini_api_key
    )

    claude_result, gemini_result = await asyncio.gather(claude_task, gemini_task, return_exceptions=True)

    claude_data, claude_provider, claude_usage = claude_result if not isinstance(claude_result, Exception) else (None, "failed", None)
    gemini_data, gemini_provider, gemini_usage = gemini_result if not isinstance(gemini_result, Exception) else (None, "failed", None)

    if claude_data is None and gemini_data is not None:
        LOG.info(f"⚠️ Claude failed, using Gemini fallback for {ticker}/{upstream_ticker} upstream triage")
        return gemini_data, 'gemini_fallback', {'gemini': gemini_usage}

    elif claude_data is not None and gemini_data is not None:
        merged = await merge_triage_scores(
            claude_results=claude_data,
            gemini_results=gemini_data,
            articles=articles,
            target_cap=target_cap,
            category_type="upstream",
            category_key=upstream_ticker,
            problematic_domains=problematic_domains
        )
        LOG.info(f"✅ Dual scoring succeeded for {ticker}/{upstream_ticker} upstream triage")
        return merged, 'dual', {'claude': claude_usage, 'gemini': gemini_usage}

    elif claude_data is not None and gemini_data is None:
        LOG.info(f"✅ Claude only for {ticker}/{upstream_ticker} upstream triage (Gemini failed)")
        return claude_data, 'claude_only', {'claude': claude_usage}

    else:
        LOG.error(f"❌ Both Claude and Gemini failed for {ticker}/{upstream_ticker} upstream triage")
        raise Exception(f"Both Claude and Gemini triage failed for {ticker}/{upstream_ticker}")


async def triage_downstream_articles_dual(
    articles: List[Dict],
    ticker: str,
    company_name: str,
    downstream_company_name: str,
    downstream_ticker: str,
    anthropic_api_key: str,
    gemini_api_key: str,
    problematic_domains: set
) -> Tuple[List[Dict], str, dict]:
    """Orchestrator for downstream triage with parallel execution and smart fallback"""
    target_cap = min(5, len(articles))

    claude_task = triage_downstream_articles_claude_async(
        articles, ticker, company_name, downstream_company_name, downstream_ticker, anthropic_api_key
    )
    gemini_task = triage_downstream_articles_gemini_async(
        articles, ticker, company_name, downstream_company_name, downstream_ticker, gemini_api_key
    )

    claude_result, gemini_result = await asyncio.gather(claude_task, gemini_task, return_exceptions=True)

    claude_data, claude_provider, claude_usage = claude_result if not isinstance(claude_result, Exception) else (None, "failed", None)
    gemini_data, gemini_provider, gemini_usage = gemini_result if not isinstance(gemini_result, Exception) else (None, "failed", None)

    if claude_data is None and gemini_data is not None:
        LOG.info(f"⚠️ Claude failed, using Gemini fallback for {ticker}/{downstream_ticker} downstream triage")
        return gemini_data, 'gemini_fallback', {'gemini': gemini_usage}

    elif claude_data is not None and gemini_data is not None:
        merged = await merge_triage_scores(
            claude_results=claude_data,
            gemini_results=gemini_data,
            articles=articles,
            target_cap=target_cap,
            category_type="downstream",
            category_key=downstream_ticker,
            problematic_domains=problematic_domains
        )
        LOG.info(f"✅ Dual scoring succeeded for {ticker}/{downstream_ticker} downstream triage")
        return merged, 'dual', {'claude': claude_usage, 'gemini': gemini_usage}

    elif claude_data is not None and gemini_data is None:
        LOG.info(f"✅ Claude only for {ticker}/{downstream_ticker} downstream triage (Gemini failed)")
        return claude_data, 'claude_only', {'claude': claude_usage}

    else:
        LOG.error(f"❌ Both Claude and Gemini failed for {ticker}/{downstream_ticker} downstream triage")
        raise Exception(f"Both Claude and Gemini triage failed for {ticker}/{downstream_ticker}")
