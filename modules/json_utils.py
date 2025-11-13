"""
JSON Utilities Module

Provides robust JSON extraction from Claude API responses.
Handles all possible response formats with multiple fallback strategies.
"""

import json
import logging
from typing import Dict, Optional

LOG = logging.getLogger(__name__)


def extract_json_from_claude_response(response_text: str, ticker: str = "") -> Optional[Dict]:
    """
    Bulletproof JSON extraction from Claude API responses.

    Handles all possible response formats:
    1. Plain JSON: {"sections": {...}}
    2. Markdown wrapped: ```json\n{...}\n```
    3. Markdown without language tag: ```\n{...}\n```
    4. Text + JSON: "Here's the result:\n```json\n{...}\n```"
    5. Multiple JSON objects (takes first valid one)
    6. Deeply nested structures with multiple closing braces

    Uses 4-tier fallback strategy:
    - Strategy 1: Direct JSON parse (fastest)
    - Strategy 2: Markdown extraction with 'json' tag (string slicing - ROBUST)
    - Strategy 3: Markdown extraction without tag (handles alternatives)
    - Strategy 4: Brace counting (nuclear option - handles ANY valid JSON)

    Args:
        response_text: Raw text from Claude API content field
        ticker: Stock ticker for logging (optional)

    Returns:
        Parsed JSON dict or None if all extraction methods fail

    Example:
        >>> text = '```json\\n{"sections": {"bottom_line": {...}}}\\n```'
        >>> result = extract_json_from_claude_response(text, "AAPL")
        >>> print(result['sections']['bottom_line'])
    """
    if not response_text or len(response_text.strip()) < 2:
        if ticker:
            LOG.error(f"[{ticker}] Empty or too-short response text")
        return None

    # Strategy 1: Try direct JSON parse (fastest path for well-behaved responses)
    try:
        parsed = json.loads(response_text.strip())
        if ticker:
            LOG.debug(f"[{ticker}] Strategy 1 success: Direct JSON parse")
        return parsed
    except json.JSONDecodeError:
        pass  # Expected for markdown-wrapped responses

    # Strategy 2: Extract from markdown code block with 'json' tag (most common)
    # Uses string slicing (Phase 2's proven approach) - handles nested structures perfectly
    if "```json" in response_text:
        json_start = response_text.find("```json") + 7  # After ```json
        json_end = response_text.find("```", json_start)  # Find NEXT closing ```
        if json_end > json_start:
            json_str = response_text[json_start:json_end].strip()
            try:
                parsed = json.loads(json_str)
                if ticker:
                    LOG.debug(f"[{ticker}] Strategy 2 success: Markdown with 'json' tag")
                return parsed
            except json.JSONDecodeError as e:
                # Log but continue to next strategy
                if ticker:
                    LOG.warning(f"[{ticker}] Strategy 2 failed (invalid JSON in markdown block): {e}")

    # Strategy 3: Extract from plain markdown block (no 'json' tag)
    # Handles ``` without language tag
    if "```" in response_text:
        json_start = response_text.find("```") + 3
        json_end = response_text.find("```", json_start)
        if json_end > json_start:
            json_str = response_text[json_start:json_end].strip()
            try:
                parsed = json.loads(json_str)
                if ticker:
                    LOG.debug(f"[{ticker}] Strategy 3 success: Plain markdown block")
                return parsed
            except json.JSONDecodeError:
                pass  # Continue to Strategy 4

    # Strategy 4: Brace counting - nuclear option for malformed wrappers
    # Manually finds matching closing brace for deeply nested JSON
    # Guaranteed to find ANY valid JSON object in the text
    start_idx = response_text.find('{')
    if start_idx >= 0:
        depth = 0
        for i in range(start_idx, len(response_text)):
            char = response_text[i]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    # Found matching closing brace
                    json_str = response_text[start_idx:i+1]
                    try:
                        parsed = json.loads(json_str)
                        if ticker:
                            LOG.debug(f"[{ticker}] Strategy 4 success: Brace counting")
                        return parsed
                    except json.JSONDecodeError:
                        # This { wasn't the right one, try to find next one
                        next_start = response_text.find('{', i+1)
                        if next_start < 0:
                            break  # No more { characters
                        start_idx = next_start
                        depth = 0
                        continue

    # All 4 strategies failed - log detailed error
    if ticker:
        LOG.error(f"[{ticker}] ❌ All 4 JSON extraction strategies failed")
        LOG.error(f"[{ticker}] Response length: {len(response_text)} chars")
        LOG.error(f"[{ticker}] Response preview (first 500 chars):\n{response_text[:500]}")
        LOG.error(f"[{ticker}] Response ending (last 200 chars):\n{response_text[-200:]}")

    return None
