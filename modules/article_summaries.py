"""
Article Summaries Module

Handles article summarization using Gemini Flash 2.5 (primary) with Claude Sonnet 4.5 (fallback).
Extracted from app.py for better modularity and easier prompt management.

Architecture:
- Gemini Flash 2.5: Primary provider (cheaper, returns full JSON)
- Claude Sonnet 4.5: Fallback provider (with prompt caching)
- Retry logic: 3 retries with exponential backoff (copied from Phase 1 executive summary)
- Quality scores: Both providers return {"quality": X.X} in their output
"""

import json
import logging
import os
import time
from typing import Dict, Optional, Tuple
import google.generativeai as genai
import aiohttp

LOG = logging.getLogger(__name__)

# Content char limit for API calls
CONTENT_CHAR_LIMIT = 50000


# ============================================================================
# PROMPT LOADERS
# ============================================================================

def load_prompt(prompt_file: str) -> str:
    """Load prompt from modules/ directory using relative path"""
    try:
        # Get the directory where this module file is located
        module_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(module_dir, prompt_file)
        with open(prompt_path, "r") as f:
            return f.read()
    except Exception as e:
        LOG.error(f"Failed to load prompt {prompt_file}: {e}")
        raise


# Load all prompts at module initialization
COMPANY_PROMPT = load_prompt("_article_summary_company_prompt")
COMPETITOR_PROMPT = load_prompt("_article_summary_competitor_prompt")
UPSTREAM_PROMPT = load_prompt("_article_summary_upstream_prompt")
DOWNSTREAM_PROMPT = load_prompt("_article_summary_downstream_prompt")
INDUSTRY_PROMPT = load_prompt("_article_summary_industry_prompt")
RELEVANCE_GATE_PROMPT = load_prompt("_relevance_gate_industry_prompt")


# ============================================================================
# RETRY LOGIC (copied from Phase 1 executive summary)
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
# GEMINI ARTICLE SUMMARY FUNCTIONS (PRIMARY)
# ============================================================================

async def generate_gemini_article_summary_company(
    company_name: str,
    ticker: str,
    title: str,
    scraped_content: str,
    gemini_api_key: str
) -> Tuple[Optional[str], str, Optional[dict]]:
    """Generate Gemini summary for company article

    Returns:
        Tuple[Optional[str], str, Optional[dict]]: (summary, provider, usage) where:
            - summary: Summary text with quality JSON
            - provider: "Gemini" or "failed"
            - usage: {"input_tokens": X, "output_tokens": Y} or None
    """
    if not gemini_api_key or not scraped_content or len(scraped_content.strip()) < 200:
        return None, "failed", None

    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=gemini_api_key)

            # User content
            user_content = f"""**TARGET COMPANY:** {company_name} ({ticker})

**ARTICLE TITLE:**
{title}

**ARTICLE CONTENT:**
{scraped_content[:CONTENT_CHAR_LIMIT]}

🚨 CRITICAL: You MUST return JSON with this exact structure:
{{"summary": "Your 2-6 paragraph summary here...", "quality": X.X}}

The quality score (0-10) is MANDATORY."""

            # Gemini Flash 2.5
            model = genai.GenerativeModel('gemini-2.5-flash')

            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json"  # Force JSON output
            }

            full_prompt = COMPANY_PROMPT + "\n\n" + user_content

            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            # Parse JSON response
            result = json.loads(response.text)
            summary = result.get("summary", "").strip()
            quality = result.get("quality")

            if summary and quality is not None:
                summary_with_quality = f"{summary}\n{{\"quality\": {quality}}}"

                # Extract usage metadata
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count
                }

                LOG.info(f"Gemini company summary: {ticker} ({len(summary)} chars, quality: {quality})")
                return summary_with_quality, "Gemini", usage
            else:
                LOG.error(f"Gemini returned incomplete JSON for {ticker}")
                return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                LOG.warning(f"Gemini company summary attempt {attempt + 1} failed for {ticker}, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                LOG.error(f"Gemini company summary failed for {ticker} after {attempt + 1} attempts: {e}")
                return None, "failed", None

    return None, "failed", None


async def generate_gemini_article_summary_competitor(
    competitor_name: str,
    competitor_ticker: str,
    target_company: str,
    target_ticker: str,
    title: str,
    scraped_content: str,
    gemini_api_key: str
) -> Tuple[Optional[str], str, Optional[dict]]:
    """Generate Gemini summary for competitor article"""
    if not gemini_api_key or not scraped_content or len(scraped_content.strip()) < 200:
        return None, "failed", None

    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=gemini_api_key)

            user_content = f"""**TARGET COMPANY:** {target_company} ({target_ticker})
**COMPETITOR:** {competitor_name} ({competitor_ticker})

**ARTICLE TITLE:**
{title}

**ARTICLE CONTENT:**
{scraped_content[:CONTENT_CHAR_LIMIT]}

**YOUR TASK:**
Extract facts about {competitor_name}'s actions and performance. Do not speculate on impact to {target_company}.

🚨 CRITICAL: You MUST return JSON with this exact structure:
{{"summary": "Your 2-6 paragraph summary here...", "quality": X.X}}"""

            model = genai.GenerativeModel('gemini-2.5-flash')
            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json"
            }

            full_prompt = COMPETITOR_PROMPT + "\n\n" + user_content
            response = model.generate_content(full_prompt, generation_config=generation_config)

            result = json.loads(response.text)
            summary = result.get("summary", "").strip()
            quality = result.get("quality")

            if summary and quality is not None:
                # IMPORTANT: Return summary WITH quality JSON on last line (for parse_quality_score)
                summary_with_quality = f"{summary}\n{{\"quality\": {quality}}}"
                LOG.info(f"Gemini competitor summary: {target_ticker} vs {competitor_ticker} ({len(summary)} chars)")

                # Extract usage metadata
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count
                }
                return summary_with_quality, "Gemini", usage
            else:
                return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Gemini competitor summary retry {attempt + 1} for {target_ticker}: {e}")
                time.sleep(wait_time)
            else:
                LOG.error(f"Gemini competitor summary failed for {target_ticker}: {e}")
                return None, "failed", None

    return None, "failed", None


async def generate_gemini_article_summary_upstream(
    value_chain_company: str,
    value_chain_ticker: str,
    target_company: str,
    target_ticker: str,
    title: str,
    scraped_content: str,
    gemini_api_key: str
) -> Tuple[Optional[str], str, Optional[dict]]:
    """Generate Gemini summary for upstream supplier article"""
    if not gemini_api_key or not scraped_content or len(scraped_content.strip()) < 200:
        return None, "failed", None

    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=gemini_api_key)

            user_content = f"""**TARGET COMPANY:** {target_company} ({target_ticker})
**UPSTREAM SUPPLIER:** {value_chain_company} ({value_chain_ticker})

**ARTICLE TITLE:**
{title}

**ARTICLE CONTENT:**
{scraped_content[:CONTENT_CHAR_LIMIT]}

**YOUR TASK:**
Extract facts about {value_chain_company}'s supply capacity, costs, financial health, and operational performance. Focus on signals affecting supply security and input costs. Do not speculate on impact to {target_company}.

🚨 CRITICAL: You MUST return JSON with this exact structure:
{{"summary": "Your 2-6 paragraph summary here...", "quality": X.X}}"""

            model = genai.GenerativeModel('gemini-2.5-flash')
            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json"
            }

            full_prompt = UPSTREAM_PROMPT + "\n\n" + user_content
            response = model.generate_content(full_prompt, generation_config=generation_config)

            result = json.loads(response.text)
            summary = result.get("summary", "").strip()
            quality = result.get("quality")

            if summary and quality is not None:
                # IMPORTANT: Return summary WITH quality JSON on last line (for parse_quality_score)
                summary_with_quality = f"{summary}\n{{\"quality\": {quality}}}"
                LOG.info(f"Gemini upstream summary: {target_ticker} <- {value_chain_ticker} ({len(summary)} chars)")

                # Extract usage metadata
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count
                }
                return summary_with_quality, "Gemini", usage
            else:
                return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Gemini upstream summary retry {attempt + 1} for {target_ticker}: {e}")
                time.sleep(wait_time)
            else:
                LOG.error(f"Gemini upstream summary failed for {target_ticker}: {e}")
                return None, "failed", None

    return None, "failed", None


async def generate_gemini_article_summary_downstream(
    value_chain_company: str,
    value_chain_ticker: str,
    target_company: str,
    target_ticker: str,
    title: str,
    scraped_content: str,
    gemini_api_key: str
) -> Tuple[Optional[str], str, Optional[dict]]:
    """Generate Gemini summary for downstream customer article"""
    if not gemini_api_key or not scraped_content or len(scraped_content.strip()) < 200:
        return None, "failed", None

    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=gemini_api_key)

            user_content = f"""**TARGET COMPANY:** {target_company} ({target_ticker})
**DOWNSTREAM CUSTOMER:** {value_chain_company} ({value_chain_ticker})

**ARTICLE TITLE:**
{title}

**ARTICLE CONTENT:**
{scraped_content[:CONTENT_CHAR_LIMIT]}

**YOUR TASK:**
Extract facts about {value_chain_company}'s order trends, demand signals, financial health, and expansion plans. Focus on signals affecting demand visibility and revenue outlook. Do not speculate on impact to {target_company}.

🚨 CRITICAL: You MUST return JSON with this exact structure:
{{"summary": "Your 2-6 paragraph summary here...", "quality": X.X}}"""

            model = genai.GenerativeModel('gemini-2.5-flash')
            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json"
            }

            full_prompt = DOWNSTREAM_PROMPT + "\n\n" + user_content
            response = model.generate_content(full_prompt, generation_config=generation_config)

            result = json.loads(response.text)
            summary = result.get("summary", "").strip()
            quality = result.get("quality")

            if summary and quality is not None:
                # IMPORTANT: Return summary WITH quality JSON on last line (for parse_quality_score)
                summary_with_quality = f"{summary}\n{{\"quality\": {quality}}}"
                LOG.info(f"Gemini downstream summary: {target_ticker} -> {value_chain_ticker} ({len(summary)} chars)")

                # Extract usage metadata
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count
                }
                return summary_with_quality, "Gemini", usage
            else:
                return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Gemini downstream summary retry {attempt + 1} for {target_ticker}: {e}")
                time.sleep(wait_time)
            else:
                LOG.error(f"Gemini downstream summary failed for {target_ticker}: {e}")
                return None, "failed", None

    return None, "failed", None


async def generate_gemini_article_summary_industry(
    industry_keyword: str,
    target_company: str,
    target_ticker: str,
    title: str,
    scraped_content: str,
    gemini_api_key: str,
    geographic_markets: str = ""
) -> Tuple[Optional[str], str, Optional[dict]]:
    """Generate Gemini summary for industry/fundamental driver article"""
    if not gemini_api_key or not scraped_content or len(scraped_content.strip()) < 200:
        return None, "failed", None

    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=gemini_api_key)

            geographic_context = f"\n**GEOGRAPHIC MARKETS:** {geographic_markets}" if geographic_markets else ""

            user_content = f"""**TARGET COMPANY:** {target_company} ({target_ticker})
**FUNDAMENTAL DRIVER KEYWORD:** {industry_keyword}{geographic_context}

**ARTICLE TITLE:**
{title}

**ARTICLE CONTENT:**
{scraped_content[:CONTENT_CHAR_LIMIT]}

**YOUR TASK:**
Extract facts about EXTERNAL market forces (commodity prices, demand indicators, input costs, policy changes, supply/demand dynamics) that relate to the fundamental driver keyword: {industry_keyword}

🚨 CRITICAL: You MUST return JSON with this exact structure:
{{"summary": "Your 2-6 paragraph summary here...", "quality": X.X}}"""

            model = genai.GenerativeModel('gemini-2.5-flash')
            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json"
            }

            full_prompt = INDUSTRY_PROMPT + "\n\n" + user_content
            response = model.generate_content(full_prompt, generation_config=generation_config)

            result = json.loads(response.text)
            summary = result.get("summary", "").strip()
            quality = result.get("quality")

            if summary and quality is not None:
                # IMPORTANT: Return summary WITH quality JSON on last line (for parse_quality_score)
                summary_with_quality = f"{summary}\n{{\"quality\": {quality}}}"
                LOG.info(f"Gemini industry summary: {target_ticker} - {industry_keyword} ({len(summary)} chars)")

                # Extract usage metadata
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count
                }
                return summary_with_quality, "Gemini", usage
            else:
                return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Gemini industry summary retry {attempt + 1} for {target_ticker}: {e}")
                time.sleep(wait_time)
            else:
                LOG.error(f"Gemini industry summary failed for {target_ticker}: {e}")
                return None, "failed", None

    return None, "failed", None


# ============================================================================
# CLAUDE ARTICLE SUMMARY FUNCTIONS (FALLBACK with prompt caching)
# ============================================================================

async def generate_claude_article_summary_company(
    company_name: str,
    ticker: str,
    title: str,
    scraped_content: str,
    anthropic_api_key: str,
    anthropic_model: str,
    anthropic_api_url: str,
    http_session: aiohttp.ClientSession
) -> Tuple[Optional[str], str, Optional[dict]]:
    """Generate Claude summary for company article (fallback)

    Returns:
        Tuple[Optional[str], str]: (summary, status) where status is:
            - "summary: Summary text with quality JSON successfully
            - "provider: "Gemini" or "failed"
    """
    if not anthropic_api_key or not scraped_content or len(scraped_content.strip()) < 200:
        return None, "failed", None

    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            # User content
            user_content = f"""**TARGET COMPANY:** {company_name} ({ticker})

**ARTICLE TITLE:**
{title}

**ARTICLE CONTENT:**
{scraped_content[:CONTENT_CHAR_LIMIT]}

🚨 CRITICAL REMINDER: You MUST end your response with quality score JSON on the absolute final line:
{{"quality": X.X}}
Omitting this will cause processing failure. This is MANDATORY for every article."""

            headers = {
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",  # Prompt caching support
                "content-type": "application/json"
            }

            data = {
                "model": anthropic_model,
                "max_tokens": 8192,
                "temperature": 0.0,
                "system": [
                    {
                        "type": "text",
                        "text": COMPANY_PROMPT,
                        "cache_control": {"type": "ephemeral"}  # Cache the prompt
                    }
                ],
                "messages": [{"role": "user", "content": user_content}]
            }

            async with http_session.post(anthropic_api_url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=180)) as response:
                if response.status == 200:
                    result = await response.json()
                    summary = result.get("content", [{}])[0].get("text", "").strip()

                    if summary and len(summary) > 10:
                        # Extract usage metadata
                        usage_data = result.get("usage", {})
                        usage = {
                            "input_tokens": usage_data.get("input_tokens", 0),
                            "output_tokens": usage_data.get("output_tokens", 0),
                            "cache_creation_input_tokens": usage_data.get("cache_creation_input_tokens", 0),
                            "cache_read_input_tokens": usage_data.get("cache_read_input_tokens", 0)
                        }

                        LOG.info(f"Claude company summary: {ticker} ({len(summary)} chars)")
                        return summary, "Sonnet", usage
                else:
                    LOG.error(f"Claude company API error {response.status}")
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        LOG.warning(f"Retrying Claude in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Claude company summary retry {attempt + 1} for {ticker}: {e}")
                time.sleep(wait_time)
            else:
                LOG.error(f"Claude company summary failed for {ticker}: {e}")
                return None, "failed", None

    return None, "failed", None


async def generate_claude_article_summary_competitor(
    competitor_name: str,
    competitor_ticker: str,
    target_company: str,
    target_ticker: str,
    title: str,
    scraped_content: str,
    anthropic_api_key: str,
    anthropic_model: str,
    anthropic_api_url: str,
    http_session: aiohttp.ClientSession
) -> Tuple[Optional[str], str, Optional[dict]]:
    """Generate Claude summary for competitor article (fallback)"""
    if not anthropic_api_key or not scraped_content or len(scraped_content.strip()) < 200:
        return None, "failed", None

    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            user_content = f"""**TARGET COMPANY:** {target_company} ({target_ticker})
**COMPETITOR:** {competitor_name} ({competitor_ticker})

**ARTICLE TITLE:**
{title}

**ARTICLE CONTENT:**
{scraped_content[:CONTENT_CHAR_LIMIT]}

**YOUR TASK:**
Extract facts about {competitor_name}'s actions and performance. Do not speculate on impact to {target_company}.

🚨 CRITICAL REMINDER: You MUST end your response with quality score JSON on the absolute final line:
{{"quality": X.X}}
Omitting this will cause processing failure. This is MANDATORY for every article."""

            headers = {
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            data = {
                "model": anthropic_model,
                "max_tokens": 8192,
                "temperature": 0.0,
                "system": [
                    {
                        "type": "text",
                        "text": COMPETITOR_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "messages": [{"role": "user", "content": user_content}]
            }

            async with http_session.post(anthropic_api_url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=90)) as response:
                if response.status == 200:
                    result = await response.json()
                    summary = result.get("content", [{}])[0].get("text", "").strip()

                    if summary and len(summary) > 10:
                        LOG.info(f"Claude competitor summary: {target_ticker} vs {competitor_ticker} ({len(summary)} chars)")
                        # Extract usage metadata
                        usage_data = result.get("usage", {})
                        usage = {
                            "input_tokens": usage_data.get("input_tokens", 0),
                            "output_tokens": usage_data.get("output_tokens", 0),
                            "cache_creation_input_tokens": usage_data.get("cache_creation_input_tokens", 0),
                            "cache_read_input_tokens": usage_data.get("cache_read_input_tokens", 0)
                        }

                        return summary, "Sonnet", usage
                else:
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                LOG.error(f"Claude competitor summary failed: {e}")
                return None, "failed", None

    return None, "failed", None


async def generate_claude_article_summary_upstream(
    value_chain_company: str,
    value_chain_ticker: str,
    target_company: str,
    target_ticker: str,
    title: str,
    scraped_content: str,
    anthropic_api_key: str,
    anthropic_model: str,
    anthropic_api_url: str,
    http_session: aiohttp.ClientSession
) -> Tuple[Optional[str], str, Optional[dict]]:
    """Generate Claude summary for upstream supplier article (fallback)"""
    if not anthropic_api_key or not scraped_content or len(scraped_content.strip()) < 200:
        return None, "failed", None

    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            user_content = f"""**TARGET COMPANY:** {target_company} ({target_ticker})
**UPSTREAM SUPPLIER:** {value_chain_company} ({value_chain_ticker})

**ARTICLE TITLE:**
{title}

**ARTICLE CONTENT:**
{scraped_content[:CONTENT_CHAR_LIMIT]}

**YOUR TASK:**
Extract facts about {value_chain_company}'s supply capacity, costs, financial health, and operational performance. Focus on signals affecting supply security and input costs. Do not speculate on impact to {target_company}.

🚨 CRITICAL REMINDER: You MUST end your response with quality score JSON on the absolute final line:
{{"quality": X.X}}
Omitting this will cause processing failure. This is MANDATORY for every article."""

            headers = {
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            data = {
                "model": anthropic_model,
                "max_tokens": 8192,
                "temperature": 0.0,
                "system": [
                    {
                        "type": "text",
                        "text": UPSTREAM_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "messages": [{"role": "user", "content": user_content}]
            }

            async with http_session.post(anthropic_api_url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=90)) as response:
                if response.status == 200:
                    result = await response.json()
                    summary = result.get("content", [{}])[0].get("text", "").strip()

                    if summary and len(summary) > 10:
                        LOG.info(f"Claude upstream summary: {target_ticker} <- {value_chain_ticker} ({len(summary)} chars)")
                        # Extract usage metadata
                        usage_data = result.get("usage", {})
                        usage = {
                            "input_tokens": usage_data.get("input_tokens", 0),
                            "output_tokens": usage_data.get("output_tokens", 0),
                            "cache_creation_input_tokens": usage_data.get("cache_creation_input_tokens", 0),
                            "cache_read_input_tokens": usage_data.get("cache_read_input_tokens", 0)
                        }

                        return summary, "Sonnet", usage
                else:
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                LOG.error(f"Claude upstream summary failed: {e}")
                return None, "failed", None

    return None, "failed", None


async def generate_claude_article_summary_downstream(
    value_chain_company: str,
    value_chain_ticker: str,
    target_company: str,
    target_ticker: str,
    title: str,
    scraped_content: str,
    anthropic_api_key: str,
    anthropic_model: str,
    anthropic_api_url: str,
    http_session: aiohttp.ClientSession
) -> Tuple[Optional[str], str, Optional[dict]]:
    """Generate Claude summary for downstream customer article (fallback)"""
    if not anthropic_api_key or not scraped_content or len(scraped_content.strip()) < 200:
        return None, "failed", None

    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            user_content = f"""**TARGET COMPANY:** {target_company} ({target_ticker})
**DOWNSTREAM CUSTOMER:** {value_chain_company} ({value_chain_ticker})

**ARTICLE TITLE:**
{title}

**ARTICLE CONTENT:**
{scraped_content[:CONTENT_CHAR_LIMIT]}

**YOUR TASK:**
Extract facts about {value_chain_company}'s order trends, demand signals, financial health, and expansion plans. Focus on signals affecting demand visibility and revenue outlook. Do not speculate on impact to {target_company}.

🚨 CRITICAL REMINDER: You MUST end your response with quality score JSON on the absolute final line:
{{"quality": X.X}}
Omitting this will cause processing failure. This is MANDATORY for every article."""

            headers = {
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            data = {
                "model": anthropic_model,
                "max_tokens": 8192,
                "temperature": 0.0,
                "system": [
                    {
                        "type": "text",
                        "text": DOWNSTREAM_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "messages": [{"role": "user", "content": user_content}]
            }

            async with http_session.post(anthropic_api_url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=90)) as response:
                if response.status == 200:
                    result = await response.json()
                    summary = result.get("content", [{}])[0].get("text", "").strip()

                    if summary and len(summary) > 10:
                        LOG.info(f"Claude downstream summary: {target_ticker} -> {value_chain_ticker} ({len(summary)} chars)")
                        # Extract usage metadata
                        usage_data = result.get("usage", {})
                        usage = {
                            "input_tokens": usage_data.get("input_tokens", 0),
                            "output_tokens": usage_data.get("output_tokens", 0),
                            "cache_creation_input_tokens": usage_data.get("cache_creation_input_tokens", 0),
                            "cache_read_input_tokens": usage_data.get("cache_read_input_tokens", 0)
                        }

                        return summary, "Sonnet", usage
                else:
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                LOG.error(f"Claude downstream summary failed: {e}")
                return None, "failed", None

    return None, "failed", None


async def generate_claude_article_summary_industry(
    industry_keyword: str,
    target_company: str,
    target_ticker: str,
    title: str,
    scraped_content: str,
    anthropic_api_key: str,
    anthropic_model: str,
    anthropic_api_url: str,
    http_session: aiohttp.ClientSession,
    geographic_markets: str = ""
) -> Tuple[Optional[str], str, Optional[dict]]:
    """Generate Claude summary for industry/fundamental driver article (fallback)"""
    if not anthropic_api_key or not scraped_content or len(scraped_content.strip()) < 200:
        return None, "failed", None

    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            geographic_context = f"\n**GEOGRAPHIC MARKETS:** {geographic_markets}" if geographic_markets else ""

            user_content = f"""**TARGET COMPANY:** {target_company} ({target_ticker})
**FUNDAMENTAL DRIVER KEYWORD:** {industry_keyword}{geographic_context}

**ARTICLE TITLE:**
{title}

**ARTICLE CONTENT:**
{scraped_content[:CONTENT_CHAR_LIMIT]}

**YOUR TASK:**
Extract facts about EXTERNAL market forces (commodity prices, demand indicators, input costs, policy changes, supply/demand dynamics) that relate to the fundamental driver keyword: {industry_keyword}

🚨 CRITICAL REMINDER: You MUST end your response with quality score JSON on the absolute final line:
{{"quality": X.X}}
Omitting this will cause processing failure. This is MANDATORY for every article."""

            headers = {
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            data = {
                "model": anthropic_model,
                "max_tokens": 8192,
                "temperature": 0.0,
                "system": [
                    {
                        "type": "text",
                        "text": INDUSTRY_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "messages": [{"role": "user", "content": user_content}]
            }

            async with http_session.post(anthropic_api_url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=90)) as response:
                if response.status == 200:
                    result = await response.json()
                    summary = result.get("content", [{}])[0].get("text", "").strip()

                    if summary and len(summary) > 10:
                        LOG.info(f"Claude industry summary: {target_ticker} - {industry_keyword} ({len(summary)} chars)")
                        # Extract usage metadata
                        usage_data = result.get("usage", {})
                        usage = {
                            "input_tokens": usage_data.get("input_tokens", 0),
                            "output_tokens": usage_data.get("output_tokens", 0),
                            "cache_creation_input_tokens": usage_data.get("cache_creation_input_tokens", 0),
                            "cache_read_input_tokens": usage_data.get("cache_read_input_tokens", 0)
                        }

                        return summary, "Sonnet", usage
                else:
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    return None, "failed", None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                LOG.error(f"Claude industry summary failed: {e}")
                return None, "failed", None

    return None, "failed", None


# ============================================================================
# RELEVANCE GATE FUNCTIONS
# ============================================================================

async def score_industry_relevance_gemini(
    ticker: str,
    company_name: str,
    industry_keyword: str,
    title: str,
    scraped_content: str,
    gemini_api_key: str,
    geographic_markets: str = ""
) -> Optional[Dict]:
    """Score industry article relevance using Gemini (primary)

    Returns: {"score": float, "reason": str, "provider": "Gemini"} or None
    """
    if not gemini_api_key or not scraped_content or len(scraped_content.strip()) < 100:
        return None

    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=gemini_api_key)

            # User content - matches Oct 14 working configuration
            user_content = f"""**TARGET COMPANY:** {company_name} ({ticker})
**FUNDAMENTAL DRIVER KEYWORD:** {industry_keyword}
**GEOGRAPHIC MARKETS:** {geographic_markets if geographic_markets else 'Unknown'}

**ARTICLE TITLE:**
{title}

**ARTICLE CONTENT:**
{scraped_content[:8000]}

**YOUR TASK:**
Rate this article's relevance to {company_name} ({ticker}) fundamental drivers on a 0-10 scale. Focus on whether article contains quantifiable or qualitative intelligence about external market forces driving financial performance. Return JSON only."""

            model = genai.GenerativeModel('gemini-2.5-flash')
            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 512,
                "response_mime_type": "application/json"
            }

            full_prompt = RELEVANCE_GATE_PROMPT + "\n\n" + user_content
            response = model.generate_content(full_prompt, generation_config=generation_config)

            result = json.loads(response.text)
            score = result.get("score")
            reason = result.get("reason", "")

            if score is not None:
                return {
                    "score": float(score),
                    "reason": reason,
                    "provider": "Gemini"
                }
            else:
                return None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Gemini relevance gate retry {attempt + 1} for {ticker}: {e}")
                time.sleep(wait_time)
            else:
                LOG.error(f"Gemini relevance gate failed for {ticker}: {e}")
                return None

    return None


async def score_industry_relevance_claude(
    ticker: str,
    company_name: str,
    industry_keyword: str,
    title: str,
    scraped_content: str,
    anthropic_api_key: str,
    anthropic_model: str,
    anthropic_api_url: str,
    http_session: aiohttp.ClientSession,
    geographic_markets: str = ""
) -> Optional[Dict]:
    """Score industry article relevance using Claude (fallback)

    Returns: {"score": float, "reason": str, "provider": "Sonnet"} or None
    """
    if not anthropic_api_key or not scraped_content or len(scraped_content.strip()) < 100:
        return None

    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            # User content - matches Oct 14 working configuration
            user_content = f"""**TARGET COMPANY:** {company_name} ({ticker})
**FUNDAMENTAL DRIVER KEYWORD:** {industry_keyword}
**GEOGRAPHIC MARKETS:** {geographic_markets if geographic_markets else 'Unknown'}

**ARTICLE TITLE:**
{title}

**ARTICLE CONTENT:**
{scraped_content[:8000]}

**YOUR TASK:**
Rate this article's relevance to {company_name} ({ticker}) fundamental drivers on a 0-10 scale. Focus on whether article contains quantifiable or qualitative intelligence about external market forces driving financial performance. Return JSON only."""

            headers = {
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            data = {
                "model": anthropic_model,
                "max_tokens": 512,
                "temperature": 0.0,
                "system": [
                    {
                        "type": "text",
                        "text": RELEVANCE_GATE_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "messages": [{"role": "user", "content": user_content}]
            }

            async with http_session.post(anthropic_api_url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    result_json = await response.json()
                    response_text = result_json.get("content", [{}])[0].get("text", "").strip()

                    # VALIDATION: Check if response is valid before parsing
                    if not response_text or len(response_text) < 5:
                        LOG.error(f"Claude returned empty/invalid response for {ticker}")
                        LOG.error(f"Full Claude response: {result_json}")  # Changed to ERROR level
                        if attempt < max_retries:
                            wait_time = 2 ** attempt
                            LOG.warning(f"Retrying Claude relevance gate in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        return None

                    # Parse JSON from response (only if validation passed)
                    # Wrap in try/except to catch JSON errors and retry properly
                    try:
                        LOG.info(f"[{ticker}] Attempting to parse Claude response ({len(response_text)} chars): {response_text[:200]}")
                        result = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        LOG.error(f"Claude returned non-JSON response for {ticker}: {e}")
                        LOG.error(f"Response text (first 500 chars): {response_text[:500]}")
                        LOG.error(f"Full Claude API response: {result_json}")
                        if attempt < max_retries:
                            wait_time = 2 ** attempt
                            LOG.warning(f"Retrying Claude relevance gate in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        return None

                    score = result.get("score")
                    reason = result.get("reason", "")

                    if score is not None:
                        return {
                            "score": float(score),
                            "reason": reason,
                            "provider": "Sonnet"
                        }
                    else:
                        LOG.warning(f"Claude response missing 'score' field for {ticker}")
                        LOG.error(f"Parsed JSON (missing score): {result}")
                        if attempt < max_retries:
                            wait_time = 2 ** attempt
                            time.sleep(wait_time)
                            continue
                        return None
                else:
                    LOG.error(f"Claude API returned status {response.status} for {ticker}")
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    return None

        except Exception as e:
            if attempt < max_retries and should_retry(e):
                wait_time = 2 ** attempt
                LOG.warning(f"Claude relevance gate retry {attempt + 1} for {ticker}: {e}")
                time.sleep(wait_time)
            else:
                LOG.error(f"Claude relevance gate failed for {ticker} after {attempt + 1} attempts: {e}")
                return None

    return None
