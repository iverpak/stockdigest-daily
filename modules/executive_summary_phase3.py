"""
Executive Summary Phase 3 - Context Integration + Length Enforcement

NEW (Nov 2025): Phase 3 now returns JSON (not markdown) with integrated content only.
Phase 3 is purely mechanical: weaves Phase 2 context into Phase 1 content and enforces length limits.

Key functions:
- generate_executive_summary_phase3(): Main entry point - returns merged JSON with integrated content
- merge_phase3_with_phase2(): Merges Phase 3 integrated content with Phase 2 metadata using bullet_id

DEPRECATED functions (markdown-based, no longer used):
- add_date_ranges_to_phase3_markdown()
- parse_phase3_markdown_to_sections()
- save_editorial_summary()
"""

import json
import logging
import os
import re
import time
from datetime import date
from typing import Dict, List, Optional
import requests

LOG = logging.getLogger(__name__)


def generate_executive_summary_phase3(
    ticker: str,
    phase2_merged_json: Dict,
    anthropic_api_key: str
) -> Optional[Dict]:
    """
    Generate Phase 3 integrated content and merge with Phase 2 metadata.

    NEW: Phase 3 returns JSON (not markdown) with only integrated content.
    Result is merged with Phase 2 using bullet_id matching.

    Args:
        ticker: Stock ticker
        phase2_merged_json: Complete merged JSON from Phase 1+2
        anthropic_api_key: Claude API key

    Returns:
        Final merged JSON with Phase 2 metadata + Phase 3 integrated content
        Or None if failed
    """
    LOG.info(f"[{ticker}] Generating Phase 3 context-integrated JSON...")

    try:
        # 1. Load Phase 3 prompt from file (NEW simplified prompt)
        prompt_path = os.path.join(os.path.dirname(__file__), '_build_executive_summary_prompt_phase3_new')
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

        start_time = time.time()

        # Make request with timeout
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=180
        )

        generation_time_ms = int((time.time() - start_time) * 1000)

        # Parse response
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("content", [{}])[0].get("text", "")

            # Parse JSON response
            phase3_json = _parse_phase3_json_response(response_text, ticker)
            if not phase3_json:
                LOG.error(f"[{ticker}] Failed to parse Phase 3 JSON response")
                return None

            usage = result.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)

            LOG.info(f"[{ticker}] ✅ Phase 3 JSON generated ({len(response_text)} chars, "
                    f"{prompt_tokens} prompt tokens, {completion_tokens} completion tokens, {generation_time_ms}ms)")

            # 4. Merge Phase 3 integrated content with Phase 2 metadata using bullet_id
            from modules.executive_summary_phase2 import merge_phase3_with_phase2

            final_merged = merge_phase3_with_phase2(phase2_merged_json, phase3_json)

            LOG.info(f"[{ticker}] ✅ Phase 3 merged with Phase 2 using bullet_id matching")

            return final_merged

        else:
            error_text = response.text[:500]
            LOG.error(f"[{ticker}] Phase 3 API error {response.status_code}: {error_text}")
            return None

    except Exception as e:
        LOG.error(f"[{ticker}] Phase 3 generation failed: {e}", exc_info=True)
        return None


def _parse_phase3_json_response(response_text: str, ticker: str) -> Optional[Dict]:
    """
    Parse Phase 3 JSON response from Claude.

    Handles Claude's common response formats:
    1. Plain JSON
    2. JSON wrapped in markdown code blocks
    3. Text before/after JSON

    Args:
        response_text: Raw response text from Claude
        ticker: Stock ticker (for logging)

    Returns:
        Parsed JSON dict or None if failed
    """
    try:
        # Try direct JSON parse first
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    json_pattern = r'```(?:json)?\s*(\{.+?\})\s*```'
    match = re.search(json_pattern, response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in text
    json_pattern = r'\{[\s\S]*"sections"[\s\S]*\}'
    match = re.search(json_pattern, response_text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    LOG.error(f"[{ticker}] Could not parse Phase 3 JSON from response (length: {len(response_text)})")
    LOG.debug(f"[{ticker}] Response preview: {response_text[:500]}")
    return None
