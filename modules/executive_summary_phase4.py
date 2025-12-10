"""
Executive Summary Phase 4 - Paragraph Generation from Surviving Bullets

NEW (Dec 2025): Phase 4 generates paragraphs (bottom_line, upside, downside) from ONLY
surviving bullets after all filtering stages. This solves the problem where Phase 1
paragraphs reference content that gets filtered out later.

A/B Testing Mode:
- Phase 1 paragraphs are preserved (original baseline)
- Phase 4 paragraphs are generated separately (new approach)
- Email #2 shows both for comparison
- Email #3 continues to use Phase 1 paragraphs until A/B testing proves Phase 4 is better

Key functions:
- generate_executive_summary_phase4(): Main entry point - returns Phase 4 paragraphs
- _filter_surviving_bullets(): Extract only bullets that passed all filters
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import requests

LOG = logging.getLogger(__name__)

# Bullet sections that can be processed (not paragraphs)
BULLET_SECTIONS = [
    "major_developments",
    "financial_performance",
    "risk_factors",
    "wall_street_sentiment",
    "competitive_industry_dynamics",
    "upcoming_catalysts"
]


def _filter_surviving_bullets(phase3_json: Dict) -> Dict:
    """
    Extract only bullets that passed all filtering stages.

    Filtering criteria:
    - filter_status != "filtered_out"
    - deduplication.status = "primary" OR "unique" (not "duplicate")

    Args:
        phase3_json: Complete Phase 3 merged JSON

    Returns:
        Dict with same structure but only surviving bullets
    """
    surviving = {"sections": {}}
    sections = phase3_json.get("sections", {})

    total_input = 0
    total_surviving = 0

    for section_name in BULLET_SECTIONS:
        section_bullets = sections.get(section_name, [])
        surviving_bullets = []

        for bullet in section_bullets:
            total_input += 1

            # Check filter_status (set by relevance/impact filter)
            filter_status = bullet.get('filter_status', '').lower()
            if filter_status == 'filtered_out':
                continue

            # Check deduplication status
            dedup = bullet.get('deduplication', {})
            dedup_status = dedup.get('status', 'unique').lower()  # Default to unique if not present
            if dedup_status == 'duplicate':
                continue

            # Bullet survives - include it
            surviving_bullets.append(bullet)
            total_surviving += 1

        surviving["sections"][section_name] = surviving_bullets

    LOG.info(f"Phase 4 filter: {total_surviving}/{total_input} bullets survived")
    return surviving


def _build_phase4_user_content(ticker: str, phase3_json: Dict) -> str:
    """
    Build user content for Phase 4 prompt containing only surviving bullets.

    Args:
        ticker: Stock ticker
        phase3_json: Complete Phase 3 merged JSON

    Returns:
        JSON string with surviving bullets formatted for Phase 4 prompt
    """
    surviving = _filter_surviving_bullets(phase3_json)

    # Count sentiment distribution for logging
    bullish_count = 0
    bearish_count = 0

    for section_name in BULLET_SECTIONS:
        for bullet in surviving["sections"].get(section_name, []):
            sentiment = bullet.get('sentiment', '').lower()
            if sentiment == 'bullish':
                bullish_count += 1
            elif sentiment == 'bearish':
                bearish_count += 1

    LOG.info(f"[{ticker}] Phase 4 sentiment distribution: {bullish_count} bullish, {bearish_count} bearish")

    return json.dumps(surviving, indent=2)


def _generate_phase4_claude(
    ticker: str,
    phase3_json: Dict,
    anthropic_api_key: str
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Generate Phase 4 paragraphs using Claude Sonnet 4.5 (primary).

    Args:
        ticker: Stock ticker
        phase3_json: Complete Phase 3 merged JSON
        anthropic_api_key: Anthropic API key

    Returns:
        Tuple of (phase4_paragraphs, usage_dict) where:
            - phase4_paragraphs: Dict with phase4_bottom_line, phase4_upside, phase4_downside
            - usage_dict: {"input_tokens": X, "output_tokens": Y} or None
    """
    try:
        # 1. Load Phase 4 prompt from file
        prompt_path = os.path.join(os.path.dirname(__file__), '_build_executive_summary_prompt_phase4')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()

        # 2. Build user content with only surviving bullets
        user_content = _build_phase4_user_content(ticker, phase3_json)

        # 3. Call Claude API with prompt caching
        headers = {
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        data = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 8000,  # Paragraphs are shorter than full executive summary
            "temperature": 0.0,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}  # Prompt caching
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }

        # Retry logic for transient errors
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
                    timeout=120  # 2 minutes (shorter than Phase 3)
                )
                generation_time_ms = int((time.time() - api_start_time) * 1000)

                # Success - break retry loop
                if response.status_code == 200:
                    break

                # Transient errors - retry
                if response.status_code in [429, 500, 503, 529] and attempt < max_retries:
                    wait_time = 2 ** attempt
                    error_preview = response.text[:200] if response.text else "No details"
                    LOG.warning(f"[{ticker}] ⚠️ Phase 4 API error {response.status_code} (attempt {attempt + 1}/{max_retries + 1}): {error_preview}")
                    LOG.warning(f"[{ticker}] 🔄 Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Non-retryable error - break
                break

            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] ⏱️ Phase 4 timeout (attempt {attempt + 1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] ❌ Phase 4 timeout after {max_retries + 1} attempts")
                    return None, None

            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] 🔌 Phase 4 network error (attempt {attempt + 1}): {e}, retrying...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] ❌ Phase 4 network error after {max_retries + 1} attempts: {e}")
                    return None, None

        # Check response
        if response is None:
            LOG.error(f"[{ticker}] ❌ Phase 4: No response after {max_retries + 1} attempts")
            return None, None

        # Parse response
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("content", [{}])[0].get("text", "")

            # Parse JSON response
            phase4_json = _parse_phase4_json_response(response_text, ticker)
            if not phase4_json:
                LOG.error(f"[{ticker}] Failed to parse Phase 4 JSON response")
                return None, None

            usage_data = result.get("usage", {})
            usage = {
                "input_tokens": usage_data.get("input_tokens", 0),
                "output_tokens": usage_data.get("output_tokens", 0),
                "model": "claude-sonnet-4-5-20250929"
            }

            LOG.info(f"[{ticker}] ✅ Phase 4 Claude generated ({len(response_text)} chars, "
                    f"{usage['input_tokens']} prompt tokens, {usage['output_tokens']} completion tokens, {generation_time_ms}ms)")

            return phase4_json, usage

        else:
            error_text = response.text[:500] if response.text else "No error details"
            LOG.error(f"[{ticker}] Phase 4 API error {response.status_code}: {error_text}")
            return None, None

    except Exception as e:
        LOG.error(f"[{ticker}] Phase 4 Claude generation failed: {e}", exc_info=True)
        return None, None


def _generate_phase4_gemini(
    ticker: str,
    phase3_json: Dict,
    gemini_api_key: str
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Generate Phase 4 paragraphs using Gemini 2.5 Pro (fallback).

    Args:
        ticker: Stock ticker
        phase3_json: Complete Phase 3 merged JSON
        gemini_api_key: Google Gemini API key

    Returns:
        Tuple of (phase4_paragraphs, usage_dict) or (None, None) if failed
    """
    import google.generativeai as genai

    try:
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)

        # Load Phase 4 prompt from file
        prompt_path = os.path.join(os.path.dirname(__file__), '_build_executive_summary_prompt_phase4')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()

        # Build user content with only surviving bullets
        user_content = _build_phase4_user_content(ticker, phase3_json)

        # Create Gemini model with system instruction
        model = genai.GenerativeModel(
            'gemini-2.5-pro',
            system_instruction=system_prompt
        )

        LOG.info(f"[{ticker}] Calling Gemini 2.5 Pro for Phase 4 paragraph generation")

        # Retry logic
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
                        'max_output_tokens': 8000
                    }
                )
                generation_time_ms = int((time.time() - start_time) * 1000)
                break  # Success

            except Exception as e:
                error_str = str(e)
                is_retryable = (
                    'ResourceExhausted' in error_str or
                    'quota' in error_str.lower() or
                    '429' in error_str or
                    'ServiceUnavailable' in error_str or
                    '503' in error_str
                )

                if is_retryable and attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] ⚠️ Gemini Phase 4 error (attempt {attempt + 1}): {error_str[:200]}")
                    LOG.warning(f"[{ticker}] 🔄 Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] ❌ Gemini Phase 4 failed after {attempt + 1} attempts: {error_str}")
                    return None, None

        if response is None:
            LOG.error(f"[{ticker}] ❌ No response from Gemini Phase 4")
            return None, None

        # Extract text
        response_text = response.text
        if not response_text or len(response_text.strip()) < 10:
            LOG.error(f"[{ticker}] ❌ Gemini returned empty Phase 4 response")
            return None, None

        # Parse JSON response
        phase4_json = _parse_phase4_json_response(response_text, ticker)
        if not phase4_json:
            LOG.error(f"[{ticker}] Failed to parse Phase 4 JSON from Gemini response")
            return None, None

        # Extract token usage
        prompt_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
        completion_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0

        usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "model": "gemini-2.5-pro"
        }

        LOG.info(f"[{ticker}] ✅ Phase 4 Gemini generated ({len(response_text)} chars, "
                f"{prompt_tokens} prompt tokens, {completion_tokens} completion tokens, {generation_time_ms}ms)")

        return phase4_json, usage

    except Exception as e:
        LOG.error(f"[{ticker}] Exception in Phase 4 Gemini generation: {e}", exc_info=True)
        return None, None


def _parse_phase4_json_response(response_text: str, ticker: str) -> Optional[Dict]:
    """
    Parse Phase 4 JSON response.

    Uses unified JSON extraction utility with 4-tier fallback strategy.

    Args:
        response_text: Raw response text from AI
        ticker: Stock ticker (for logging)

    Returns:
        Parsed JSON dict or None if failed
    """
    from modules.json_utils import extract_json_from_claude_response

    parsed = extract_json_from_claude_response(response_text, ticker)

    if not parsed:
        return None

    # Validate required fields
    required_fields = ["phase4_bottom_line", "phase4_upside_scenario", "phase4_downside_scenario"]
    for field in required_fields:
        if field not in parsed:
            LOG.warning(f"[{ticker}] Phase 4 missing required field: {field}")
            # Add empty placeholder
            parsed[field] = {
                "content": "No content generated.",
                "context": "No context available.",
                "source_articles": [],
                "date_range": ""
            }

    return parsed


def generate_executive_summary_phase4(
    ticker: str,
    phase3_json: Dict,
    anthropic_api_key: str,
    gemini_api_key: str = None,
    primary_model: str = 'claude'
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Generate Phase 4 paragraphs from surviving bullets.

    This is the main entry point for Phase 4 generation. Phase 4 generates
    bottom_line, upside_scenario, and downside_scenario paragraphs using ONLY
    bullets that survived all filtering stages.

    A/B Testing Mode:
    - Phase 4 paragraphs are stored separately from Phase 1 paragraphs
    - Email #2 shows both for comparison
    - No modification to existing Phase 1/2/3 workflow

    Args:
        ticker: Stock ticker
        phase3_json: Complete merged JSON from Phase 1+2+3
        anthropic_api_key: Anthropic API key
        gemini_api_key: Google Gemini API key (optional)
        primary_model: Primary AI model ('claude' or 'gemini', defaults to 'claude')

    Returns:
        Tuple of (phase4_paragraphs, usage_dict) where:
            - phase4_paragraphs: Dict with phase4_bottom_line, phase4_upside_scenario, phase4_downside_scenario
            - usage_dict: {"input_tokens": X, "output_tokens": Y, "model": "..."} or None
    """
    LOG.info(f"[{ticker}] 📝 Starting Phase 4: Paragraph generation from surviving bullets")

    # Check if we have any surviving bullets
    surviving = _filter_surviving_bullets(phase3_json)
    total_surviving = sum(len(surviving["sections"].get(s, [])) for s in BULLET_SECTIONS)

    if total_surviving == 0:
        LOG.warning(f"[{ticker}] ⚠️ Phase 4: No surviving bullets - returning empty paragraphs")
        empty_result = {
            "phase4_bottom_line": {
                "content": "No material developments in recent articles.",
                "context": "No relevant filing context available.",
                "source_articles": [],
                "date_range": ""
            },
            "phase4_upside_scenario": {
                "content": "No material upside catalysts in recent articles.",
                "context": "No relevant filing context available.",
                "source_articles": [],
                "date_range": ""
            },
            "phase4_downside_scenario": {
                "content": "No material downside risks in recent articles.",
                "context": "No relevant filing context available.",
                "source_articles": [],
                "date_range": ""
            }
        }
        return empty_result, {"input_tokens": 0, "output_tokens": 0, "model": "none"}

    # Choose provider order based on primary_model setting
    if primary_model == 'gemini':
        # Try Gemini first
        if gemini_api_key:
            LOG.info(f"[{ticker}] Phase 4: Attempting Gemini 2.5 Pro (primary)")
            phase4_json, usage = _generate_phase4_gemini(ticker, phase3_json, gemini_api_key)

            if phase4_json and usage:
                LOG.info(f"[{ticker}] ✅ Phase 4: Gemini 2.5 Pro succeeded")
                return phase4_json, usage
            else:
                LOG.warning(f"[{ticker}] ⚠️ Phase 4: Gemini failed, falling back to Claude")

        # Fall back to Claude
        if anthropic_api_key:
            LOG.info(f"[{ticker}] Phase 4: Using Claude Sonnet 4.5 (fallback)")
            phase4_json, usage = _generate_phase4_claude(ticker, phase3_json, anthropic_api_key)

            if phase4_json and usage:
                LOG.info(f"[{ticker}] ✅ Phase 4: Claude Sonnet 4.5 succeeded (fallback)")
                return phase4_json, usage

    else:  # primary_model == 'claude' (default)
        # Try Claude first
        if anthropic_api_key:
            LOG.info(f"[{ticker}] Phase 4: Attempting Claude Sonnet 4.5 (primary)")
            phase4_json, usage = _generate_phase4_claude(ticker, phase3_json, anthropic_api_key)

            if phase4_json and usage:
                LOG.info(f"[{ticker}] ✅ Phase 4: Claude Sonnet 4.5 succeeded")
                return phase4_json, usage
            else:
                LOG.warning(f"[{ticker}] ⚠️ Phase 4: Claude failed, falling back to Gemini")

        # Fall back to Gemini
        if gemini_api_key:
            LOG.info(f"[{ticker}] Phase 4: Using Gemini 2.5 Pro (fallback)")
            phase4_json, usage = _generate_phase4_gemini(ticker, phase3_json, gemini_api_key)

            if phase4_json and usage:
                LOG.info(f"[{ticker}] ✅ Phase 4: Gemini 2.5 Pro succeeded (fallback)")
                return phase4_json, usage

    # Both failed
    LOG.error(f"[{ticker}] ❌ Phase 4: Both providers failed - returning empty paragraphs")
    empty_result = {
        "phase4_bottom_line": {
            "content": "Phase 4 generation failed.",
            "context": "No context available.",
            "source_articles": [],
            "date_range": ""
        },
        "phase4_upside_scenario": {
            "content": "Phase 4 generation failed.",
            "context": "No context available.",
            "source_articles": [],
            "date_range": ""
        },
        "phase4_downside_scenario": {
            "content": "Phase 4 generation failed.",
            "context": "No context available.",
            "source_articles": [],
            "date_range": ""
        }
    }
    return empty_result, None
