"""
Executive Summary Phase 3 - Deduplication Only

NEW (Dec 2025): Phase 3 is dedup-only. It passes through content/context unchanged.
Only merges the deduplication field (status, absorbs/absorbed_by, shared_theme).

Key functions:
- generate_executive_summary_phase3(): Main entry point - returns merged JSON with dedup metadata
- merge_phase3_with_phase2(): Merges Phase 3 dedup metadata with Phase 2 JSON using bullet_id
"""

import json
import logging
import os
import re
import time
from datetime import date
from typing import Dict, List, Optional, Tuple
import requests

LOG = logging.getLogger(__name__)


def _generate_phase3_gemini(
    ticker: str,
    phase2_merged_json: Dict,
    gemini_api_key: str
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Generate Phase 3 integrated content using Gemini 2.5 Pro (fallback).

    Args:
        ticker: Stock ticker
        phase2_merged_json: Complete merged JSON from Phase 1+2
        gemini_api_key: Google Gemini API key

    Returns:
        Tuple of (final_merged_json, usage_dict) where:
            - final_merged_json: Phase 2 metadata + Phase 3 integrated content (or None if failed)
            - usage_dict: {"prompt_tokens": X, "completion_tokens": Y} or None
    """
    import google.generativeai as genai

    try:
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)

        # Load Phase 3 prompt from file
        prompt_path = os.path.join(os.path.dirname(__file__), '_build_executive_summary_prompt_phase3')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()

        # Build user content (Phase 2 merged JSON as formatted string)
        user_content = json.dumps(phase2_merged_json, indent=2)

        # Create Gemini model with system instruction
        model = genai.GenerativeModel(
            'gemini-2.5-pro',
            system_instruction=system_prompt
        )

        LOG.info(f"[{ticker}] Calling Gemini 2.5 Pro for Phase 3 context integration")

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
                        'max_output_tokens': 16000
                    }
                )
                generation_time_ms = int((time.time() - start_time) * 1000)

                # Success - break retry loop
                break

            except Exception as e:
                error_str = str(e)

                # Check for retryable errors
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
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] ⚠️ Gemini Phase 3 error (attempt {attempt + 1}/{max_retries + 1}): {error_str[:200]}")
                    LOG.warning(f"[{ticker}] 🔄 Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] ❌ Gemini 2.5 Pro Phase 3 failed after {attempt + 1} attempts: {error_str}")
                    return None, None

        # Check if we got a response
        if response is None:
            LOG.error(f"[{ticker}] ❌ No response from Gemini Phase 3 after {max_retries + 1} attempts")
            return None, None

        # Extract text from response
        response_text = response.text

        if not response_text or len(response_text.strip()) < 10:
            LOG.error(f"[{ticker}] ❌ Gemini returned empty Phase 3 response")
            return None, None

        # Parse JSON response
        phase3_json = _parse_phase3_json_response(response_text, ticker)
        if not phase3_json:
            LOG.error(f"[{ticker}] Failed to parse Phase 3 JSON from Gemini response")
            return None, None

        # Extract token usage
        prompt_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
        completion_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model": "gemini-2.5-pro"  # Track which model was used
        }

        LOG.info(f"[{ticker}] ✅ Phase 3 Gemini JSON generated ({len(response_text)} chars, "
                f"{prompt_tokens} prompt tokens, {completion_tokens} completion tokens, {generation_time_ms}ms)")

        # Merge Phase 3 integrated content with Phase 2 metadata
        from modules.executive_summary_phase2 import merge_phase3_with_phase2
        final_merged = merge_phase3_with_phase2(phase2_merged_json, phase3_json)

        LOG.info(f"[{ticker}] ✅ Phase 3 Gemini merged with Phase 2 using bullet_id matching")

        return final_merged, usage

    except Exception as e:
        LOG.error(f"[{ticker}] Exception in Phase 3 Gemini generation: {e}", exc_info=True)
        return None, None


def _generate_phase3_claude(
    ticker: str,
    phase2_merged_json: Dict,
    anthropic_api_key: str
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Generate Phase 3 integrated content using Claude Sonnet 4.5 (primary).

    Args:
        ticker: Stock ticker
        phase2_merged_json: Complete merged JSON from Phase 1+2
        anthropic_api_key: Anthropic API key

    Returns:
        Tuple of (final_merged_json, usage_dict) where:
            - final_merged_json: Phase 2 metadata + Phase 3 integrated content (or None if failed)
            - usage_dict: {"input_tokens": X, "output_tokens": Y} or None
    """
    try:
        # 1. Load Phase 3 prompt from file (simplified prompt: context integration only)
        prompt_path = os.path.join(os.path.dirname(__file__), '_build_executive_summary_prompt_phase3')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()

        # 2. Build user content (Phase 2 merged JSON as formatted string)
        user_content = json.dumps(phase2_merged_json, indent=2)

        # 3. Call Claude API with prompt caching
        headers = {
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        data = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 16000,
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

        # Retry logic for transient errors (529, 503, 429, 500)
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
                if response.status_code in [429, 500, 503, 529] and attempt < max_retries:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    error_preview = response.text[:200] if response.text else "No details"
                    LOG.warning(f"[{ticker}] ⚠️ Phase 3 API error {response.status_code} (attempt {attempt + 1}/{max_retries + 1}): {error_preview}")
                    LOG.warning(f"[{ticker}] 🔄 Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Non-retryable error or max retries reached - break
                break

            except requests.exceptions.Timeout as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] ⏱️ Phase 3 timeout (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] ❌ Phase 3 timeout after {max_retries + 1} attempts")
                    return None

            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] 🔌 Phase 3 network error (attempt {attempt + 1}/{max_retries + 1}): {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] ❌ Phase 3 network error after {max_retries + 1} attempts: {e}")
                    return None

        # Check if we got a response
        if response is None:
            LOG.error(f"[{ticker}] ❌ Phase 3: No response after {max_retries + 1} attempts")
            return None

        # Parse response
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("content", [{}])[0].get("text", "")

            # Parse JSON response
            phase3_json = _parse_phase3_json_response(response_text, ticker)
            if not phase3_json:
                LOG.error(f"[{ticker}] Failed to parse Phase 3 JSON response")
                return None, None

            usage_data = result.get("usage", {})
            prompt_tokens = usage_data.get("input_tokens", 0)
            completion_tokens = usage_data.get("output_tokens", 0)

            # Create usage dict for cost tracking
            usage = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "model": "claude-sonnet-4-5-20250929"  # Track which model was used
            }

            LOG.info(f"[{ticker}] ✅ Phase 3 JSON generated ({len(response_text)} chars, "
                    f"{prompt_tokens} prompt tokens, {completion_tokens} completion tokens, {generation_time_ms}ms)")

            # 4. Merge Phase 3 integrated content with Phase 2 metadata using bullet_id
            from modules.executive_summary_phase2 import merge_phase3_with_phase2

            final_merged = merge_phase3_with_phase2(phase2_merged_json, phase3_json)

            LOG.info(f"[{ticker}] ✅ Phase 3 merged with Phase 2 using bullet_id matching")

            return final_merged, usage

        else:
            error_text = response.text[:500] if response.text else "No error details"
            LOG.error(f"[{ticker}] Phase 3 API error {response.status_code} after {max_retries + 1} attempts: {error_text}")
            return None, None

    except Exception as e:
        LOG.error(f"[{ticker}] Phase 3 generation failed: {e}", exc_info=True)
        return None, None


def generate_executive_summary_phase3(
    ticker: str,
    phase2_merged_json: Dict,
    anthropic_api_key: str,
    gemini_api_key: str = None,
    primary_model: str = 'claude'
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Generate Phase 3 integrated content with configurable primary model and fallback.

    This is the main entry point for Phase 3 generation. Primary model is determined
    by the primary_model parameter ('claude' or 'gemini').

    NEW: Phase 3 returns JSON (not markdown) with only integrated content.
    Result is merged with Phase 2 using bullet_id matching.

    Args:
        ticker: Stock ticker
        phase2_merged_json: Complete merged JSON from Phase 1+2
        anthropic_api_key: Anthropic API key
        gemini_api_key: Google Gemini API key (optional)
        primary_model: Primary AI model ('claude' or 'gemini', defaults to 'claude')

    Returns:
        Tuple of (final_merged_json, usage_dict) where:
            - final_merged_json: Phase 2 metadata + Phase 3 integrated content (or None if failed)
            - usage_dict: {"input_tokens": X, "output_tokens": Y} or {"prompt_tokens": X, "completion_tokens": Y} or None
    """
    # Choose provider order based on primary_model setting
    if primary_model == 'gemini':
        # Try Gemini 2.5 Pro first (primary)
        if gemini_api_key:
            LOG.info(f"[{ticker}] Phase 3: Attempting Gemini 2.5 Pro (primary)")
            gemini_result = _generate_phase3_gemini(
                ticker=ticker,
                phase2_merged_json=phase2_merged_json,
                gemini_api_key=gemini_api_key
            )

            final_merged, usage = gemini_result
            if final_merged and usage:
                LOG.info(f"[{ticker}] ✅ Phase 3: Gemini 2.5 Pro succeeded")
                # Convert Gemini usage format to match Claude format for compatibility
                if "prompt_tokens" in usage:
                    usage = {
                        "input_tokens": usage["prompt_tokens"],
                        "output_tokens": usage["completion_tokens"],
                        "model": usage.get("model", "gemini-2.5-pro")  # Preserve model info
                    }
                return final_merged, usage
            else:
                LOG.warning(f"[{ticker}] ⚠️ Phase 3: Gemini 2.5 Pro failed, falling back to Claude Sonnet 4.5")
        else:
            LOG.warning(f"[{ticker}] ⚠️ No Gemini API key provided, using Claude Sonnet 4.5 only")

        # Fall back to Claude Sonnet 4.5
        if anthropic_api_key:
            LOG.info(f"[{ticker}] Phase 3: Using Claude Sonnet 4.5 (fallback)")
            claude_result = _generate_phase3_claude(
                ticker=ticker,
                phase2_merged_json=phase2_merged_json,
                anthropic_api_key=anthropic_api_key
            )

            final_merged, usage = claude_result
            if final_merged and usage:
                LOG.info(f"[{ticker}] ✅ Phase 3: Claude Sonnet 4.5 succeeded (fallback)")
                return final_merged, usage
            else:
                LOG.error(f"[{ticker}] ❌ Phase 3: Claude Sonnet 4.5 also failed")
        else:
            LOG.error(f"[{ticker}] ❌ No Anthropic API key provided for fallback")

    else:  # primary_model == 'claude' (default)
        # Try Claude Sonnet 4.5 first (primary) with one retry
        if anthropic_api_key:
            max_attempts = 2  # 1 retry = 2 total attempts

            for attempt in range(1, max_attempts + 1):
                if attempt == 1:
                    LOG.info(f"[{ticker}] Phase 3: Attempting Claude Sonnet 4.5 (primary)")
                else:
                    LOG.info(f"[{ticker}] 🔄 Phase 3: Retrying Claude Sonnet 4.5 (attempt {attempt}/{max_attempts})")

                claude_result = _generate_phase3_claude(
                    ticker=ticker,
                    phase2_merged_json=phase2_merged_json,
                    anthropic_api_key=anthropic_api_key
                )

                final_merged, usage = claude_result
                if final_merged and usage:
                    LOG.info(f"[{ticker}] ✅ Phase 3: Claude Sonnet 4.5 succeeded on attempt {attempt}")
                    return final_merged, usage
                else:
                    # Failed - decide whether to retry or fall back
                    if attempt < max_attempts:
                        LOG.warning(f"[{ticker}] ⚠️ Phase 3: Claude attempt {attempt} failed (JSON/validation), retrying...")
                    else:
                        LOG.warning(f"[{ticker}] ⚠️ Phase 3: Claude failed {max_attempts} times, falling back to Gemini 2.5 Pro")
        else:
            LOG.warning(f"[{ticker}] ⚠️ No Anthropic API key provided, using Gemini 2.5 Pro only")

        # Fall back to Gemini 2.5 Pro
        if gemini_api_key:
            LOG.info(f"[{ticker}] Phase 3: Using Gemini 2.5 Pro (fallback)")
            gemini_result = _generate_phase3_gemini(
                ticker=ticker,
                phase2_merged_json=phase2_merged_json,
                gemini_api_key=gemini_api_key
            )

            final_merged, usage = gemini_result
            if final_merged and usage:
                LOG.info(f"[{ticker}] ✅ Phase 3: Gemini 2.5 Pro succeeded (fallback)")
                # Convert Gemini usage format to match Claude format for compatibility
                if "prompt_tokens" in usage:
                    usage = {
                        "input_tokens": usage["prompt_tokens"],
                        "output_tokens": usage["completion_tokens"],
                        "model": usage.get("model", "gemini-2.5-pro")  # Preserve model info
                    }
                return final_merged, usage
            else:
                LOG.error(f"[{ticker}] ❌ Phase 3: Gemini 2.5 Pro also failed")
        else:
            LOG.error(f"[{ticker}] ❌ No Gemini API key provided for fallback")

    # Both failed
    LOG.error(f"[{ticker}] ❌ Phase 3: Both providers failed for Phase 3 - cannot integrate context")
    return None, None


def _parse_phase3_json_response(response_text: str, ticker: str) -> Optional[Dict]:
    """
    Parse Phase 3 JSON response from Claude.

    Uses unified JSON extraction utility with 4-tier fallback strategy:
    1. Plain JSON
    2. Markdown wrapped with 'json' tag
    3. Markdown wrapped without tag
    4. Brace counting (handles deeply nested JSON)

    Args:
        response_text: Raw response text from Claude
        ticker: Stock ticker (for logging)

    Returns:
        Parsed JSON dict or None if failed
    """
    from modules.json_utils import extract_json_from_claude_response
    return extract_json_from_claude_response(response_text, ticker)
