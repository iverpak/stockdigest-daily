"""
Executive Summary Phase 1 - Article Theme Extraction

This module generates structured JSON executive summaries from news articles ONLY.
NO filing data is used in Phase 1 - that comes in Phase 2.

Key functions:
- generate_executive_summary_phase1(): Main entry point
- validate_phase1_json(): Schema validator
- convert_phase1_to_sections_dict(): Simple bullets for Email #3
- convert_phase1_to_enhanced_sections(): Full structure for Email #2
"""

import json
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import requests

LOG = logging.getLogger(__name__)


# Phase 1 System Prompt (embedded from modules/_build_executive_summary_prompt_phase1)
def get_phase1_system_prompt(ticker: str) -> str:
    """
    Get Phase 1 system prompt (static, no ticker substitution for caching).

    The prompt is now ticker-agnostic for optimal prompt caching.
    Ticker context is provided in user_content instead.

    Args:
        ticker: Stock ticker (unused, kept for API compatibility)

    Returns:
        Static system prompt string
    """
    # Read the prompt file (NO ticker substitution - enables prompt caching)
    try:
        import os
        prompt_path = os.path.join(os.path.dirname(__file__), '_build_executive_summary_prompt_phase1')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        return prompt_template  # Return as-is (no {TICKER} replacement)
    except Exception as e:
        LOG.error(f"Failed to load Phase 1 prompt: {e}")
        raise


def _build_phase1_user_content(
    ticker: str,
    categories: Dict[str, List[Dict]],
    config: Dict
) -> str:
    """
    Build user_content for Phase 1 (articles only, NO filings).

    Ports logic from _build_executive_summary_prompt() lines 14145-14250:
    - Collect flagged articles from company/industry/competitor/upstream/downstream
    - Add category tags [COMPANY], [INDUSTRY - keyword], [COMPETITOR], [UPSTREAM], [DOWNSTREAM]
    - Sort by published_at DESC (newest first)
    - Build unified timeline (up to 50 articles)

    Args:
        ticker: Stock ticker
        categories: Dict with keys: company, industry, competitor, upstream, downstream
        config: Ticker config dict

    Returns:
        Formatted article timeline string
    """
    company_name = config.get("name", ticker)

    # Collect ALL flagged articles across all categories
    all_flagged_articles = []

    # Company articles
    for article in categories.get("company", []):
        if article.get("ai_summary"):
            article['_category'] = 'COMPANY'
            article['_category_tag'] = '[COMPANY]'
            all_flagged_articles.append(article)

    # Industry articles
    for article in categories.get("industry", []):
        if article.get("ai_summary"):
            keyword = article.get("search_keyword", "Industry")
            article['_category'] = 'INDUSTRY'
            article['_category_tag'] = f'[INDUSTRY - {keyword}]'
            all_flagged_articles.append(article)

    # Competitor articles
    for article in categories.get("competitor", []):
        if article.get("ai_summary"):
            article['_category'] = 'COMPETITOR'
            article['_category_tag'] = '[COMPETITOR]'
            all_flagged_articles.append(article)

    # Upstream articles (value chain)
    for article in categories.get("upstream", []):
        if article.get("ai_summary"):
            article['_category'] = 'VALUE_CHAIN'
            article['_category_tag'] = '[UPSTREAM]'
            all_flagged_articles.append(article)

    # Downstream articles (value chain)
    for article in categories.get("downstream", []):
        if article.get("ai_summary"):
            article['_category'] = 'VALUE_CHAIN'
            article['_category_tag'] = '[DOWNSTREAM]'
            all_flagged_articles.append(article)

    # Build unified timeline with category tags (if articles exist)
    unified_timeline = []
    if all_flagged_articles:
        # Sort all articles globally by published_at DESC (newest first)
        all_flagged_articles.sort(
            key=lambda x: x.get("published_at") or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True
        )

        # Build timeline with sequential indices - only count articles with ai_summary
        # No limit - process all flagged articles from triage (max ~79 based on triage caps)
        timeline_idx = 0
        for article in all_flagged_articles:
            title = article.get("title", "")
            ai_summary = article.get("ai_summary", "")
            domain = article.get("domain", "")
            published_at = article.get("published_at")

            # Skip articles without ai_summary (safety check)
            if not ai_summary:
                continue

            # Format date with year for clarity in staleness checks
            if published_at:
                # Use format like "Oct 22, 2025" (includes year for temporal comparisons)
                date_str = published_at.strftime("%b %d, %Y")
            else:
                date_str = "Unknown date"

            category_tag = article.get("_category_tag", "[UNKNOWN]")

            # Get domain formal name (simplified - just use domain if lookup not available)
            source_name = domain if domain else "Unknown Source"
            # Use 0-indexed article numbers [0], [1], [2] for source tracking
            unified_timeline.append(f"[{timeline_idx}] {category_tag} {title} [{source_name}] {date_str}: {ai_summary}")
            timeline_idx += 1

    # Calculate report context (current date, day of week)
    current_date = datetime.now().strftime("%B %d, %Y")
    day_of_week = datetime.now().strftime("%A")

    # Build user_content
    # CRITICAL: Add ticker context here (not in system prompt) for prompt caching optimization
    ticker_header = f"TARGET COMPANY: {ticker} ({company_name})\n\n"

    # Add explicit current date for temporal staleness checks (matches Phase 2)
    current_date_header = f"CURRENT DATE: {current_date}\n\n"

    if not all_flagged_articles:
        user_content = (
            ticker_header +
            current_date_header +
            f"REPORT CONTEXT:\n"
            f"Report type: {day_of_week}\n\n"
            f"---\n\n"
            f"FLAGGED ARTICLE COUNT: 0\n\n"
            f"NO FLAGGED ARTICLES - Generate quiet day summary per template."
        )
    else:
        article_count = len(all_flagged_articles)
        user_content = (
            ticker_header +
            current_date_header +
            f"REPORT CONTEXT:\n"
            f"Report type: {day_of_week}\n\n"
            f"---\n\n"
            f"FLAGGED ARTICLE COUNT: {article_count}\n\n"
            f"UNIFIED ARTICLE TIMELINE (newest to oldest):\n"
            + "\n".join(unified_timeline)
        )

    return user_content


def _generate_phase1_gemini(
    ticker: str,
    categories: Dict[str, List[Dict]],
    config: Dict,
    gemini_api_key: str
) -> Optional[Dict]:
    """
    Generate Phase 1 executive summary using Gemini 2.5 Pro (primary).

    Args:
        ticker: Stock ticker
        categories: Dict with keys: company, industry, competitor
        config: Ticker configuration dict
        gemini_api_key: Google Gemini API key

    Returns:
        dict with:
            json_output: Full Phase 1 JSON structure
            model_used: "gemini-2.5-pro"
            prompt_tokens: int
            completion_tokens: int
            generation_time_ms: int
        Or None if failed
    """
    import google.generativeai as genai

    try:
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)

        # Build system prompt (static, cacheable)
        system_prompt = get_phase1_system_prompt(ticker)

        # Build user content from articles
        user_content = _build_phase1_user_content(ticker, categories, config)

        # Estimate token counts for logging
        system_tokens_est = len(system_prompt) // 4
        user_tokens_est = len(user_content) // 4
        total_tokens_est = system_tokens_est + user_tokens_est
        LOG.info(f"[{ticker}] Phase 1 Gemini prompt size: system={len(system_prompt)} chars (~{system_tokens_est} tokens), user={len(user_content)} chars (~{user_tokens_est} tokens), total=~{total_tokens_est} tokens")

        # Create Gemini model with system instruction
        model = genai.GenerativeModel(
            'gemini-2.5-pro',
            system_instruction=system_prompt
        )

        LOG.info(f"[{ticker}] Calling Gemini 2.5 Pro for Phase 1 executive summary")

        # Retry logic for transient errors
        max_retries = 2
        response = None
        generation_time_ms = 0

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = model.generate_content(
                    user_content,
                    generation_config={
                        'temperature': 0.0,
                        'max_output_tokens': 20000
                    }
                )
                generation_time_ms = int((time.time() - start_time) * 1000)

                # Success - break retry loop
                break

            except Exception as e:
                error_str = str(e)

                # Check for retryable errors (quota, rate limit, service unavailable)
                is_retryable = (
                    'ResourceExhausted' in error_str or
                    'quota' in error_str.lower() or
                    '429' in error_str or
                    'ServiceUnavailable' in error_str or
                    '503' in error_str or
                    'DeadlineExceeded' in error_str or
                    'timeout' in error_str.lower()
                )

                if is_retryable and attempt < max_retries:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    LOG.warning(f"[{ticker}] ⚠️ Gemini error (attempt {attempt + 1}/{max_retries + 1}): {error_str[:200]}")
                    LOG.warning(f"[{ticker}] 🔄 Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error or max retries reached
                    LOG.error(f"[{ticker}] ❌ Gemini 2.5 Pro Phase 1 failed after {attempt + 1} attempts: {error_str}")
                    return None

        # Check if we got a response
        if response is None:
            LOG.error(f"[{ticker}] ❌ No response from Gemini after {max_retries + 1} attempts")
            return None

        # Extract text from response
        response_text = response.text

        if not response_text or len(response_text.strip()) < 10:
            LOG.error(f"[{ticker}] ❌ Gemini returned empty Phase 1 response")
            return None

        # Parse JSON from response using unified parser
        from modules.json_utils import extract_json_from_claude_response
        json_output = extract_json_from_claude_response(response_text, ticker)

        if not json_output:
            LOG.error(f"[{ticker}] ❌ Failed to extract Phase 1 JSON from Gemini response")
            return None

        # Extract token usage from Gemini response
        prompt_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
        completion_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0

        LOG.info(f"✅ [{ticker}] Phase 1 Gemini generated JSON ({len(response_text)} chars, {prompt_tokens} prompt tokens, {completion_tokens} completion tokens, {generation_time_ms}ms)")

        return {
            "json_output": json_output,
            "model_used": "gemini-2.5-pro",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generation_time_ms": generation_time_ms
        }

    except Exception as e:
        LOG.error(f"❌ [{ticker}] Exception in Phase 1 Gemini generation: {e}", exc_info=True)
        return None


def _generate_phase1_claude(
    ticker: str,
    categories: Dict[str, List[Dict]],
    config: Dict,
    anthropic_api_key: str
) -> Optional[Dict]:
    """
    Generate Phase 1 executive summary using Claude Sonnet 4.5 (fallback).

    Args:
        ticker: Stock ticker
        categories: Dict with keys: company, industry, competitor
        config: Ticker configuration dict
        anthropic_api_key: Anthropic API key

    Returns:
        dict with:
            json_output: Full Phase 1 JSON structure
            model_used: "claude-sonnet-4-5-20250929"
            prompt_tokens: int
            completion_tokens: int
            generation_time_ms: int
        Or None if failed
    """
    try:
        # Build system prompt
        system_prompt = get_phase1_system_prompt(ticker)

        # Build user content from articles
        user_content = _build_phase1_user_content(ticker, categories, config)

        # Log prompt sizes
        system_tokens_est = len(system_prompt) // 4
        user_tokens_est = len(user_content) // 4
        total_tokens_est = system_tokens_est + user_tokens_est
        LOG.info(f"[{ticker}] Phase 1 prompt size: system={len(system_prompt)} chars (~{system_tokens_est} tokens), user={len(user_content)} chars (~{user_tokens_est} tokens), total=~{total_tokens_est} tokens")

        # Call Claude API
        headers = {
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01",  # Prompt caching support
            "content-type": "application/json"
        }

        data = {
            "model": "claude-sonnet-4-5-20250929",  # Sonnet 4.5
            "max_tokens": 20000,  # Generous limit for comprehensive output
            "temperature": 0.0,   # Deterministic
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}  # Enable prompt caching
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }

        LOG.info(f"[{ticker}] Calling Claude API for Phase 1 executive summary")

        # Retry logic for transient errors (503, 429, 500)
        max_retries = 2
        response = None
        generation_time_ms = 0

        for attempt in range(max_retries + 1):
            try:
                api_start_time = time.time()
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=180  # 3 minutes
                )
                generation_time_ms = int((time.time() - api_start_time) * 1000)

                # Success - break retry loop
                if response.status_code == 200:
                    break

                # Transient errors - retry with exponential backoff
                if response.status_code in [429, 500, 503] and attempt < max_retries:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    error_preview = response.text[:200] if response.text else "No details"
                    LOG.warning(f"[{ticker}] ⚠️ API error {response.status_code} (attempt {attempt + 1}/{max_retries + 1}): {error_preview}")
                    LOG.warning(f"[{ticker}] 🔄 Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Non-retryable error or max retries reached - break
                break

            except requests.exceptions.Timeout as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] ⏱️ Request timeout (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] ❌ Request timeout after {max_retries + 1} attempts")
                    return None

            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] 🔌 Network error (attempt {attempt + 1}/{max_retries + 1}): {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] ❌ Network error after {max_retries + 1} attempts: {e}")
                    return None

        # Check if we got a response
        if response is None:
            LOG.error(f"[{ticker}] ❌ No response received after {max_retries + 1} attempts")
            return None

        if response.status_code == 200:
            result = response.json()

            # Extract JSON from response using unified parser (4-tier fallback strategy)
            content = result.get("content", [{}])[0].get("text", "")

            # Use shared JSON extraction utility (handles all response formats)
            from modules.json_utils import extract_json_from_claude_response
            json_output = extract_json_from_claude_response(content, ticker)

            if not json_output:
                LOG.error(f"[{ticker}] Failed to extract Phase 1 JSON from response")
                return None

            # Track usage
            usage = result.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)

            # Log cache performance
            cache_creation = usage.get("cache_creation_input_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            if cache_creation > 0:
                LOG.info(f"[{ticker}] 💾 CACHE CREATED: {cache_creation} tokens (Phase 1)")
            elif cache_read > 0:
                LOG.info(f"[{ticker}] ⚡ CACHE HIT: {cache_read} tokens (Phase 1) - 90% savings!")

            LOG.info(f"✅ [{ticker}] Phase 1 generated JSON ({len(content)} chars, {prompt_tokens} prompt tokens, {completion_tokens} completion tokens, {generation_time_ms}ms)")

            return {
                "json_output": json_output,
                "model_used": "claude",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "generation_time_ms": generation_time_ms
            }
        else:
            error_text = response.text[:500] if response.text else "No error details"
            LOG.error(f"❌ [{ticker}] Claude API error {response.status_code}: {error_text}")
            return None

    except Exception as e:
        LOG.error(f"❌ [{ticker}] Exception calling Claude for Phase 1: {e}", exc_info=True)
        return None


def generate_executive_summary_phase1(
    ticker: str,
    categories: Dict[str, List[Dict]],
    config: Dict,
    anthropic_api_key: str,
    gemini_api_key: str = None
) -> Optional[Dict]:
    """
    Generate Phase 1 executive summary with Gemini 2.5 Pro (primary) and Claude fallback.

    This is the main entry point for Phase 1 generation. It attempts Gemini first
    for cost savings (60% cheaper), then falls back to Claude if Gemini fails.

    Args:
        ticker: Stock ticker (e.g., "AAPL", "RY.TO")
        categories: Dict with keys: company, industry, competitor
                   Each contains list of article dicts
        config: Ticker config dict (contains company_name, etc.)
        anthropic_api_key: Anthropic API key for Claude fallback
        gemini_api_key: Google Gemini API key (optional)

    Returns:
        {
            "json_output": {...},  # Full Phase 1 JSON structure
            "model_used": "gemini-2.5-pro" or "claude-sonnet-4-5-20250929",
            "prompt_tokens": 28500,
            "completion_tokens": 3500,
            "generation_time_ms": 45000
        }
        Or None if both providers failed
    """
    # Try Gemini 2.5 Pro first (primary)
    if gemini_api_key:
        LOG.info(f"[{ticker}] Phase 1: Attempting Gemini 2.5 Pro (primary)")
        gemini_result = _generate_phase1_gemini(
            ticker=ticker,
            categories=categories,
            config=config,
            gemini_api_key=gemini_api_key
        )

        if gemini_result and gemini_result.get("json_output"):
            LOG.info(f"[{ticker}] ✅ Phase 1: Gemini 2.5 Pro succeeded")
            return gemini_result
        else:
            LOG.warning(f"[{ticker}] ⚠️ Phase 1: Gemini 2.5 Pro failed, falling back to Claude Sonnet")
    else:
        LOG.warning(f"[{ticker}] ⚠️ No Gemini API key provided, using Claude Sonnet only")

    # Fall back to Claude Sonnet 4.5
    if anthropic_api_key:
        LOG.info(f"[{ticker}] Phase 1: Using Claude Sonnet 4.5 (fallback)")
        claude_result = _generate_phase1_claude(
            ticker=ticker,
            categories=categories,
            config=config,
            anthropic_api_key=anthropic_api_key
        )

        if claude_result and claude_result.get("json_output"):
            LOG.info(f"[{ticker}] ✅ Phase 1: Claude Sonnet succeeded (fallback)")
            return claude_result
        else:
            LOG.error(f"[{ticker}] ❌ Phase 1: Claude Sonnet also failed")
    else:
        LOG.error(f"[{ticker}] ❌ No Anthropic API key provided for fallback")

    # Both failed
    LOG.error(f"[{ticker}] ❌ Phase 1: Both Gemini and Claude failed - cannot generate executive summary")
    return None


def _validate_source_articles(source_articles, context: str) -> Tuple[bool, str]:
    """
    Validate source_articles field is array of non-negative integers.

    Args:
        source_articles: The source_articles value to validate
        context: Description for error messages (e.g., "major_developments[0]")

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(source_articles, list):
        return False, f"{context} source_articles must be array"

    for idx, val in enumerate(source_articles):
        if not isinstance(val, int) or val < 0:
            return False, f"{context} source_articles[{idx}] must be non-negative integer"

    return True, ""


def validate_phase1_json(json_output: Dict) -> Tuple[bool, str]:
    """
    Validate Phase 1 JSON matches expected schema.

    Checks:
    - "sections" key exists
    - All 10 required sections present
    - Bullet sections have correct structure (bullet_id, topic_label, content, source_articles, filing_hints)
    - Bottom line has content + source_articles + word_count
    - Scenarios have content + source_articles
    - source_articles arrays contain valid non-negative integers

    Returns:
        (is_valid, error_message)
    """
    try:
        if "sections" not in json_output:
            return False, "Missing 'sections' key"

        sections = json_output["sections"]

        # Check all 10 required sections present
        required_sections = [
            "bottom_line", "major_developments", "financial_performance",
            "risk_factors", "wall_street_sentiment", "competitive_industry_dynamics",
            "upcoming_catalysts", "upside_scenario", "downside_scenario", "key_variables"
        ]

        for section_name in required_sections:
            if section_name not in sections:
                return False, f"Missing required section: {section_name}"

        # Validate bottom_line structure
        bottom_line = sections["bottom_line"]
        if not isinstance(bottom_line, dict):
            return False, "bottom_line must be object"
        if "content" not in bottom_line:
            return False, "bottom_line missing 'content'"
        if "word_count" not in bottom_line:
            return False, "bottom_line missing 'word_count'"
        # Validate source_articles if present (optional for backward compatibility)
        if "source_articles" in bottom_line:
            is_valid, err = _validate_source_articles(bottom_line["source_articles"], "bottom_line")
            if not is_valid:
                return False, err

        # Validate bullet sections
        bullet_sections = [
            "major_developments", "financial_performance", "risk_factors",
            "wall_street_sentiment", "competitive_industry_dynamics", "upcoming_catalysts"
        ]

        for section_name in bullet_sections:
            section_content = sections[section_name]
            if not isinstance(section_content, list):
                return False, f"{section_name} must be array"

            for i, bullet in enumerate(section_content):
                if not isinstance(bullet, dict):
                    return False, f"{section_name}[{i}] must be object"

                required_fields = ["bullet_id", "topic_label", "content", "filing_hints"]
                for field in required_fields:
                    if field not in bullet:
                        return False, f"{section_name}[{i}] missing '{field}'"

                # Validate filing_hints structure
                filing_hints = bullet["filing_hints"]
                if not isinstance(filing_hints, dict):
                    return False, f"{section_name}[{i}] filing_hints must be object"

                # Check all 3 required keys exist and are arrays
                for filing_type in ["10-K", "10-Q", "Transcript"]:
                    if filing_type not in filing_hints:
                        return False, f"{section_name}[{i}] filing_hints missing '{filing_type}'"
                    if not isinstance(filing_hints[filing_type], list):
                        return False, f"{section_name}[{i}] filing_hints['{filing_type}'] must be array"

                # Empty arrays are valid (means no filing context needed)

                # Validate source_articles if present (optional for backward compatibility)
                if "source_articles" in bullet:
                    is_valid, err = _validate_source_articles(bullet["source_articles"], f"{section_name}[{i}]")
                    if not is_valid:
                        return False, err

        # Validate key_variables (no filing_hints required)
        key_variables = sections["key_variables"]
        if not isinstance(key_variables, list):
            return False, "key_variables must be array"

        for i, var in enumerate(key_variables):
            if not isinstance(var, dict):
                return False, f"key_variables[{i}] must be object"
            required_fields = ["bullet_id", "topic_label", "content"]
            for field in required_fields:
                if field not in var:
                    return False, f"key_variables[{i}] missing '{field}'"
            # Validate source_articles if present (optional for backward compatibility)
            if "source_articles" in var:
                is_valid, err = _validate_source_articles(var["source_articles"], f"key_variables[{i}]")
                if not is_valid:
                    return False, err

        # Validate scenarios
        for scenario_name in ["upside_scenario", "downside_scenario"]:
            scenario = sections[scenario_name]
            if not isinstance(scenario, dict):
                return False, f"{scenario_name} must be object"
            if "content" not in scenario:
                return False, f"{scenario_name} missing 'content'"
            # Validate source_articles if present (optional for backward compatibility)
            if "source_articles" in scenario:
                is_valid, err = _validate_source_articles(scenario["source_articles"], scenario_name)
                if not is_valid:
                    return False, err

        return True, ""

    except Exception as e:
        return False, f"Validation exception: {str(e)}"


def _should_include_bullet_in_email3(bullet: Dict, section_name: str) -> bool:
    """
    Determine if a bullet should be included in Email #3 (user-facing).

    Filtering rules:
    - Remove bullets with relevance = "none"
    - Remove bullets with relevance = "indirect" AND impact = "low impact"
    - Don't filter if fields are missing (safety)
    - Don't filter paragraphs (bottom_line, upside, downside)

    Applies to ALL bullet sections: major_developments, financial_performance,
    risk_factors, wall_street_sentiment, competitive_industry_dynamics, upcoming_catalysts

    Args:
        bullet: Bullet dict with optional relevance/impact fields
        section_name: Name of the section (e.g., "major_developments")

    Returns:
        True if bullet should be included, False if filtered out
    """
    # Don't filter paragraphs (they're not enriched with relevance/impact)
    if section_name in ["bottom_line", "upside_scenario", "downside_scenario"]:
        return True

    # Get enrichment fields
    relevance = bullet.get('relevance', '').lower()
    impact = bullet.get('impact', '').lower()

    # Safety: Don't filter if fields are missing
    if not relevance or not impact:
        return True

    # Filter rule 1: relevance is "none"
    if relevance == 'none':
        return False

    # Filter rule 2: relevance is "indirect" AND impact is "low impact"
    if relevance == 'indirect' and impact == 'low impact':
        return False

    # Keep bullet
    return True


def get_used_article_indices(phase_json: Dict) -> set:
    """
    Collect article indices from bullets/paragraphs that pass Email #3 filtering.

    This function applies the same filtering logic as convert_phase1_to_sections_dict()
    to determine which bullets survive filtering, then collects all source_articles
    from those surviving bullets.

    Use this to filter the Source Articles section in Email #3 to only show
    articles that actually contributed to the final report.

    Args:
        phase_json: Phase 1+2 (or Phase 2+3) merged JSON with source_articles fields

    Returns:
        Set of 0-indexed article numbers that contributed to the final report.
        Empty set if no source_articles tracking is present.
    """
    used_indices = set()
    sections = phase_json.get("sections", {})

    # Bullet sections that get filtered
    bullet_sections = [
        "major_developments", "financial_performance", "risk_factors",
        "wall_street_sentiment", "competitive_industry_dynamics",
        "upcoming_catalysts", "key_variables"
    ]

    for section_name in bullet_sections:
        section_content = sections.get(section_name, [])
        if not isinstance(section_content, list):
            continue

        for bullet in section_content:
            if not isinstance(bullet, dict):
                continue

            # Apply same filter as Email #3
            if _should_include_bullet_in_email3(bullet, section_name):
                # Collect source articles from this bullet
                source_articles = bullet.get('source_articles', [])
                if isinstance(source_articles, list):
                    used_indices.update(source_articles)

    # Paragraph sections (always included, no filtering)
    for para_section in ["bottom_line", "upside_scenario", "downside_scenario"]:
        para = sections.get(para_section, {})
        if isinstance(para, dict):
            source_articles = para.get('source_articles', [])
            if isinstance(source_articles, list):
                used_indices.update(source_articles)

    return used_indices


def convert_phase1_to_sections_dict(phase1_json: Dict) -> Dict[str, List[Dict]]:
    """
    Convert Phase 1+2 JSON to Email #3 user-facing format with bullet_id matching.

    NEW FORMAT (same as Email #4):
    **[Entity] Topic • Sentiment (reason)**
    Content paragraph (Nov 04)

    Args:
        phase1_json: Phase 1+2 merged JSON (or Phase 2+3 merged JSON)

    Returns:
        sections dict: {section_name: [{'bullet_id': '...', 'formatted': '...'}, ...]}
    """
    from modules.executive_summary_utils import format_bullet_header, add_dates_to_email_sections

    sections = {
        "bottom_line": [],
        "major_developments": [],
        "financial_operational": [],
        "risk_factors": [],
        "wall_street": [],
        "competitive_industry": [],
        "upcoming_catalysts": [],
        "upside_scenario": [],
        "downside_scenario": [],
        "key_variables": []
    }

    json_sections = phase1_json.get("sections", {})

    # Bottom Line (simple list, no bullet_id)
    if "bottom_line" in json_sections:
        # Use integrated content if available (Phase 3), fall back to Phase 1 + Phase 2 separately
        if json_sections["bottom_line"].get("content_integrated"):
            content = json_sections["bottom_line"]["content_integrated"]
        else:
            content = json_sections["bottom_line"].get("content", "")
            context = json_sections["bottom_line"].get("context", "")
            if context:
                content += f" Context: {context}"
        sections["bottom_line"] = [content]

    # Helper function to format bullets (simple, no metadata)
    def format_bullet_simple(bullet: Dict, section_name: str) -> Dict:
        """Format bullet with header and content. Returns {'bullet_id': '...', 'formatted': '...'}"""
        # Use shared utility for header (hide reason in Email #3 user-facing emails)
        # Pass section_name to control sentiment display (hidden for some sections)
        header = format_bullet_header(bullet, show_reason=False, section_name=section_name)

        # Use integrated content if available (Phase 3), fall back to Phase 1 + Phase 2 separately
        if bullet.get('content_integrated'):
            # Phase 3 has run - use integrated content
            content = bullet['content_integrated']
        else:
            # Phase 3 has not run - combine Phase 1 content with Phase 2 context
            content = bullet['content']
            context = bullet.get('context', '')
            if context:
                content += f" Context: {context}"

        return {
            'bullet_id': bullet['bullet_id'],
            'formatted': f"{header}\n{content}"
        }

    # All bullet sections
    section_mapping = {
        "major_developments": "major_developments",
        "financial_performance": "financial_operational",
        "risk_factors": "risk_factors",
        "wall_street_sentiment": "wall_street",
        "competitive_industry_dynamics": "competitive_industry",
        "upcoming_catalysts": "upcoming_catalysts",
        "key_variables": "key_variables"
    }

    for json_key, sections_key in section_mapping.items():
        if json_key in json_sections:
            # Apply filter to ALL bullet sections (uses module-level function)
            filtered_bullets = [
                b for b in json_sections[json_key]
                if _should_include_bullet_in_email3(b, json_key)
            ]

            sections[sections_key] = [
                format_bullet_simple(b, json_key)
                for b in filtered_bullets
            ]

    # Scenarios (simple lists, no bullet_id)
    for json_key, sections_key in [
        ("upside_scenario", "upside_scenario"),
        ("downside_scenario", "downside_scenario")
    ]:
        if json_key in json_sections:
            # Use integrated content if available (Phase 3), fall back to Phase 1 + Phase 2 separately
            if json_sections[json_key].get("content_integrated"):
                content = json_sections[json_key]["content_integrated"]
            else:
                content = json_sections[json_key].get("content", "")
                context = json_sections[json_key].get("context", "")
                if context:
                    content += f" Context: {context}"
            sections[sections_key] = [content]

    # Add dates to all sections using bullet_id matching
    sections = add_dates_to_email_sections(sections, phase1_json)

    return sections


def convert_phase1_to_enhanced_sections(phase1_json: Dict) -> Dict[str, List[Dict]]:
    """
    Convert Phase 1+2 JSON to Email #2 QA format with bullet_id matching.

    NEW FORMAT:
    **[Entity] Topic • Sentiment (reason)**
    Content paragraph (Nov 04)
      Filing hints: 10-K (Section A, B)
      Filing keywords: ["keyword1"]
      ID: bullet_id

    Args:
        phase1_json: Phase 1+2 merged JSON (with impact, sentiment, context, etc.)

    Returns:
        sections dict: {section_name: [{'bullet_id': '...', 'formatted': '...'}, ...]}
    """
    from modules.executive_summary_utils import format_bullet_header, add_dates_to_email_sections

    sections = {
        "bottom_line": [],
        "major_developments": [],
        "financial_operational": [],
        "risk_factors": [],
        "wall_street": [],
        "competitive_industry": [],
        "upcoming_catalysts": [],
        "upside_scenario": [],
        "downside_scenario": [],
        "key_variables": []
    }

    json_sections = phase1_json.get("sections", {})

    # Bottom Line (simple list, no bullet_id)
    if "bottom_line" in json_sections:
        content = json_sections["bottom_line"].get("content", "")
        context = json_sections["bottom_line"].get("context", "")
        if context:
            content += f" Context: {context}"
        sections["bottom_line"] = [content]

    # Helper function to format bullets with metadata
    def format_bullet_with_metadata(bullet: Dict) -> Dict:
        """Format bullet with header, content, and Email #2 metadata. Returns {'bullet_id': '...', 'formatted': '...'}"""
        # Use shared utility for header
        header = format_bullet_header(bullet)

        # Content (Phase 1 content)
        content = bullet['content']

        result = f"{header}\n{content}"

        # Add Phase 2 context if present (regex will bold "Context:" label)
        context = bullet.get('context', '')
        if context:
            result += f" Context: {context}"

        # Add Email #2 metadata
        # Filing hints
        hints = bullet.get("filing_hints", {})
        hint_parts = []
        for filing_type, sections_list in hints.items():
            if sections_list:
                hint_parts.append(f"{filing_type} ({', '.join(sections_list)})")
        if hint_parts:
            result += f"<br>  Filing hints: {'; '.join(hint_parts)}"

        # Filing keywords
        keywords = bullet.get("filing_keywords", [])
        if keywords:
            import json
            result += f"<br>  Filing keywords: {json.dumps(keywords)}"

        # Phase 2 Enrichment Metadata (QA purposes)
        # Only show if Phase 2 fields are present (indicates bullet was enriched)
        if any(key in bullet for key in ['impact', 'sentiment', 'reason', 'relevance']):
            metadata_parts = []
            # Entity (only present for competitive_industry_dynamics)
            entity_val = bullet.get('entity', 'N/A')
            metadata_parts.append(f"Entity: {entity_val}")
            # Relevance
            relevance_val = bullet.get('relevance', 'N/A')
            metadata_parts.append(f"Relevance: {relevance_val}")
            # Impact
            impact_val = bullet.get('impact', 'N/A')
            metadata_parts.append(f"Impact: {impact_val}")
            # Sentiment
            sentiment_val = bullet.get('sentiment', 'N/A')
            metadata_parts.append(f"Sentiment: {sentiment_val}")
            # Reason
            reason_val = bullet.get('reason', 'N/A')
            metadata_parts.append(f"Reason: {reason_val}")

            result += f"<br>  Metadata: {' | '.join(metadata_parts)}"

        # Source articles (Phase 1 tracking)
        source_articles = bullet.get('source_articles', [])
        if source_articles:
            result += f"<br>  Source Articles: {source_articles}"

        # Bullet ID
        result += f"<br>  ID: {bullet['bullet_id']}"

        return {
            'bullet_id': bullet['bullet_id'],
            'formatted': result
        }

    # All bullet sections
    section_mapping = {
        "major_developments": "major_developments",
        "financial_performance": "financial_operational",
        "risk_factors": "risk_factors",
        "wall_street_sentiment": "wall_street",
        "competitive_industry_dynamics": "competitive_industry",
        "upcoming_catalysts": "upcoming_catalysts",
        "key_variables": "key_variables"
    }

    for json_key, sections_key in section_mapping.items():
        if json_key in json_sections:
            sections[sections_key] = [
                format_bullet_with_metadata(b)
                for b in json_sections[json_key]
            ]

    # Scenarios (simple lists, no bullet_id)
    for json_key, sections_key in [
        ("upside_scenario", "upside_scenario"),
        ("downside_scenario", "downside_scenario")
    ]:
        if json_key in json_sections:
            content = json_sections[json_key].get("content", "")
            context = json_sections[json_key].get("context", "")
            if context:
                content += f" Context: {context}"
            sections[sections_key] = [content]

    # Add dates to all sections using bullet_id matching
    sections = add_dates_to_email_sections(sections, phase1_json)

    return sections


def convert_phase3_to_email2_sections(phase3_json: Dict) -> Dict[str, List[Dict]]:
    """
    Convert Phase 3 merged JSON to Email #2 QA format with deduplication display.

    NEW FORMAT (Nov 2025):
    ────────────────────────────────────────────────────────────────────────────────
    [bullet_id] Topic Label • Sentiment (reason)
    ────────────────────────────────────────────────────────────────────────────────
    Phase 1+2 (Original):
    Content from Phase 1 articles with Phase 2 context appended.

    Phase 3 (Integrated):
    Content with Phase 2 context woven inline.

    Metadata:
      Entity: N/A | Relevance: direct | Impact: high | Sentiment: bullish | Reason: xxx
      Filing hints: 10-K (Section A, B)
      Source Articles: [0, 3, 5]

    Deduplication:
      status: unique ✅
    ────────────────────────────────────────────────────────────────────────────────

    Args:
        phase3_json: Phase 3 merged JSON (Phase 1+2 metadata + Phase 3 integrated + deduplication)

    Returns:
        sections dict: {section_name: [{'bullet_id': '...', 'formatted': '...'}, ...]}
    """
    from modules.executive_summary_utils import format_bullet_header, add_dates_to_email_sections

    sections = {
        "bottom_line": [],
        "major_developments": [],
        "financial_operational": [],
        "risk_factors": [],
        "wall_street": [],
        "competitive_industry": [],
        "upcoming_catalysts": [],
        "upside_scenario": [],
        "downside_scenario": [],
        "key_variables": []
    }

    json_sections = phase3_json.get("sections", {})

    # Bottom Line (paragraph section - no deduplication)
    if "bottom_line" in json_sections:
        bl = json_sections["bottom_line"]
        content_p12 = bl.get("content", "")
        context = bl.get("context", "")
        content_p3 = bl.get("content_integrated", "")

        # Simplified: show content with optional context and Phase 3
        result = content_p12
        if context:
            result += f" Context: {context}"
        if content_p3 and content_p3 != content_p12:
            result += f"<br><strong>Phase 3:</strong> {content_p3}"

        sections["bottom_line"] = [result]

    # Helper function to format bullets with full QA display
    def format_bullet_with_dedup(bullet: Dict) -> Dict:
        """Format bullet showing Phase 1+2, Phase 3, Metadata, and Deduplication.

        Simplified format (Nov 2025) - matches Phase 1+2 style:
        - No heavy div wrappers with padding/margins
        - No hr separators
        - Clean header + content + metadata flow
        """
        # Header line (includes topic_label, sentiment, reason)
        header = format_bullet_header(bullet)

        # Start with header
        result = f"{header}\n"

        # Phase 1+2 content
        content_p12 = bullet.get('content', '')
        result += content_p12

        # Add Phase 2 context if present
        context = bullet.get('context', '')
        if context:
            result += f" Context: {context}"

        # Phase 3 integrated content (if different from Phase 1+2)
        content_p3 = bullet.get('content_integrated', '')
        if content_p3 and content_p3 != content_p12:
            result += f"<br><strong>Phase 3:</strong> {content_p3}"

        # Metadata line (compact)
        metadata_parts = []
        entity_val = bullet.get('entity')
        if entity_val and entity_val != 'N/A':
            metadata_parts.append(f"Entity: {entity_val}")
        relevance_val = bullet.get('relevance', 'N/A')
        metadata_parts.append(f"Relevance: {relevance_val}")
        impact_val = bullet.get('impact', 'N/A')
        metadata_parts.append(f"Impact: {impact_val}")
        sentiment_val = bullet.get('sentiment', 'N/A')
        metadata_parts.append(f"Sentiment: {sentiment_val}")
        if metadata_parts:
            result += f"<br>  Metadata: {' | '.join(metadata_parts)}"

        # Filing hints (compact)
        hints = bullet.get("filing_hints", {})
        hint_parts = []
        for filing_type, sections_list in hints.items():
            if sections_list:
                hint_parts.append(f"{filing_type} ({', '.join(sections_list)})")
        if hint_parts:
            result += f"<br>  Filing hints: {'; '.join(hint_parts)}"

        # Source articles
        source_articles = bullet.get('source_articles', [])
        if source_articles:
            result += f"<br>  Source Articles: {source_articles}"

        # Deduplication status (compact - only show if not unique)
        dedup = bullet.get('deduplication', {'status': 'unique'})
        status = dedup.get('status', 'unique')

        if status == 'unique':
            result += " <span style='color: #28a745;'>✅</span>"
        elif status == 'primary':
            absorbs = dedup.get('absorbs', [])
            result += f"<br>  Dedup: primary (absorbs {absorbs}) <span style='color: #007bff;'>🔗</span>"
            proposed = dedup.get('proposed_edit', '')
            if proposed:
                result += f"<br>  Proposed edit: {proposed}"
        elif status == 'duplicate':
            absorbed_by = dedup.get('absorbed_by', '')
            result += f"<br>  Dedup: duplicate (absorbed by {absorbed_by}) <span style='color: #dc3545;'>❌</span>"

        # Bullet ID (at end for reference)
        result += f"<br>  ID: {bullet.get('bullet_id', 'N/A')}"

        return {
            'bullet_id': bullet.get('bullet_id', ''),
            'formatted': result
        }

    # All bullet sections
    section_mapping = {
        "major_developments": "major_developments",
        "financial_performance": "financial_operational",
        "risk_factors": "risk_factors",
        "wall_street_sentiment": "wall_street",
        "competitive_industry_dynamics": "competitive_industry",
        "upcoming_catalysts": "upcoming_catalysts",
        "key_variables": "key_variables"
    }

    for json_key, sections_key in section_mapping.items():
        if json_key in json_sections:
            sections[sections_key] = [
                format_bullet_with_dedup(b)
                for b in json_sections[json_key]
            ]

    # Scenarios (paragraph sections - no deduplication)
    for json_key, sections_key in [
        ("upside_scenario", "upside_scenario"),
        ("downside_scenario", "downside_scenario")
    ]:
        if json_key in json_sections:
            scenario = json_sections[json_key]
            content_p12 = scenario.get("content", "")
            context = scenario.get("context", "")
            content_p3 = scenario.get("content_integrated", "")

            # Simplified: show content with optional context and Phase 3
            result = content_p12
            if context:
                result += f" Context: {context}"
            if content_p3 and content_p3 != content_p12:
                result += f"<br><strong>Phase 3:</strong> {content_p3}"

            sections[sections_key] = [result]

    # Add dates to all sections using bullet_id matching
    sections = add_dates_to_email_sections(sections, phase3_json)

    return sections
