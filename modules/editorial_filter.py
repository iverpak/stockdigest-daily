"""
Editorial Filter (Phase 2.5)

Final editorial gate for institutional-grade equity research. Reviews Phase 1+2
enriched bullets and makes binary KEEP/REMOVE decisions based on whether content
belongs in a professional investment report.

This is NOT a fact-checker or staleness filter (that's Phase 1.5). This catches:
- Misplaced content (Company X news in Company Y's report)
- Misclassified relationships (vendor labeled as customer)
- Tangential analyst coverage (unrelated sectors)
- Content that damages report credibility
- Anything that makes readers question "why is this here?"

Architecture:
- Primary: Claude Sonnet 4.5 (3 retries)
- Fallback: Gemini 2.5 Pro (2 retries)
- Pass-through: If both fail, continue pipeline with all bullets

Output: Simple KEEP/REMOVE per bullet_id, parsed via regex.
Removal Cap: 20% of input bullets (if exceeded, keep all - treat as systemic issue).

STATUS: TEST MODE - Runs and emails results, but does not filter bullets from Phase 3.
"""

import logging
import re
import time
from datetime import datetime
from math import ceil
from typing import Dict, List, Optional, Tuple

import requests

LOG = logging.getLogger(__name__)


# =============================================================================
# PROMPT
# =============================================================================

EDITORIAL_FILTER_PROMPT = """You are the final editorial gate for an institutional-grade equity research product. Readers are portfolio managers and investment-savvy retail investors.

Review each bullet (content + context) and decide: KEEP or REMOVE.

REMOVE if a professional reader would pause and think "why is this in my report?"

Examples of what to remove:
- Content about Company X placed in Company Y's report due to name overlap
- Misclassified relationships (e.g., vendor labeled as customer, as indicated by context contradicting content)
- Analyst coverage of unrelated sectors (just because analyst name or bank name appears)
- News requiring 3+ logical steps to connect to this company
- Content that contradicts its own context
- Geographic/market data for regions where company has minimal exposure
- Routine corporate actions with no strategic signal
- Anything that damages report credibility

KEEP material developments, competitive intelligence, risk factors, and investment-relevant content.

BIAS: One bad bullet damages credibility more than one missing bullet. When uncertain, REMOVE.

REMOVAL CAP: Remove at most {max_removals} bullets (20% of {total_bullets} total).
If you want to remove more than this cap, still output your full judgment - we will handle the cap on our end.

TICKER: {ticker}
COMPANY: {company_name}

BULLETS TO REVIEW:
{bullets_formatted}

OUTPUT FORMAT: One line per bullet. Format exactly as shown:
BULLET_ID: KEEP
or
BULLET_ID: REMOVE

Output ONLY the bullet decisions, no explanations or other text."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_bullets_for_prompt(phase2_json: Dict) -> Tuple[str, List[str]]:
    """
    Format Phase 2 merged JSON bullets for the editorial filter prompt.

    Only includes bullet_id, content, and context (if present).
    Excludes paragraph sections (bottom_line, upside_scenario, downside_scenario).

    Args:
        phase2_json: Phase 1+2 merged JSON with sections

    Returns:
        Tuple of (formatted string for prompt, list of bullet_ids included)
    """
    sections = phase2_json.get('sections', {})

    # Bullet sections to review (exclude paragraphs)
    bullet_section_names = [
        'major_developments',
        'financial_performance',
        'risk_factors',
        'wall_street_sentiment',
        'competitive_industry_dynamics',
        'upcoming_catalysts',
        'key_variables'
    ]

    formatted_parts = []
    bullet_ids = []

    for section_name in bullet_section_names:
        section_data = sections.get(section_name, [])
        if not isinstance(section_data, list):
            continue

        for bullet in section_data:
            if not isinstance(bullet, dict):
                continue

            bullet_id = bullet.get('bullet_id', '')
            if not bullet_id:
                continue

            # Get content - prefer integrated, fall back to original
            content = bullet.get('content_integrated') or bullet.get('content', '')
            if not content:
                continue

            # Get context if present
            context = bullet.get('context', '')

            bullet_ids.append(bullet_id)

            # Format bullet
            part = f"BULLET_ID: {bullet_id}\n"
            part += f"CONTENT: {content}\n"
            if context:
                part += f"CONTEXT: {context}\n"
            part += "---"

            formatted_parts.append(part)

    formatted_str = "\n".join(formatted_parts)

    return formatted_str, bullet_ids


def _parse_editorial_response(response_text: str, expected_bullet_ids: List[str]) -> Dict[str, str]:
    """
    Parse LLM response into bullet_id -> KEEP/REMOVE mapping.

    Uses regex to extract decisions. Handles various formatting quirks.

    Args:
        response_text: Raw LLM response
        expected_bullet_ids: List of bullet IDs we sent to the LLM

    Returns:
        Dict mapping bullet_id to 'KEEP' or 'REMOVE'
    """
    decisions = {}

    # Pattern: BULLET_ID: KEEP or BULLET_ID: REMOVE
    # Allows for whitespace variations, supports lowercase bullet IDs
    # e.g., "q4_delivery_guidance: KEEP" or "DEV_001: REMOVE"
    pattern = r'^([a-zA-Z_0-9]+)\s*:\s*(KEEP|REMOVE)\s*$'

    # Build case-insensitive lookup for expected bullet IDs
    expected_lower = {bid.lower(): bid for bid in expected_bullet_ids}

    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            bullet_id_raw = match.group(1)
            decision = match.group(2).upper()

            # Match case-insensitively, but return the ORIGINAL bullet_id format
            bullet_id_lower = bullet_id_raw.lower()
            if bullet_id_lower in expected_lower:
                original_bullet_id = expected_lower[bullet_id_lower]
                decisions[original_bullet_id] = decision

    return decisions


def _apply_removal_cap(
    decisions: Dict[str, str],
    max_removals: int,
    ticker: str
) -> Tuple[Dict[str, str], bool]:
    """
    Apply removal cap to decisions.

    If removals exceed cap, keep ALL bullets (treat as systemic issue).

    Args:
        decisions: Dict mapping bullet_id to 'KEEP'/'REMOVE'
        max_removals: Maximum allowed removals (20% of total)
        ticker: For logging

    Returns:
        Tuple of (final decisions dict, cap_exceeded bool)
    """
    remove_count = sum(1 for d in decisions.values() if d == 'REMOVE')

    if remove_count > max_removals:
        LOG.warning(
            f"[{ticker}] Phase 2.5: Removal cap exceeded! "
            f"LLM wants to remove {remove_count}, cap is {max_removals}. "
            f"Keeping ALL bullets (treating as systemic issue)."
        )
        # Keep all bullets
        return {bid: 'KEEP' for bid in decisions.keys()}, True

    return decisions, False


# =============================================================================
# CLAUDE IMPLEMENTATION (PRIMARY)
# =============================================================================

def _call_claude_editorial_filter(
    ticker: str,
    company_name: str,
    bullets_formatted: str,
    bullet_ids: List[str],
    total_bullets: int,
    max_removals: int,
    anthropic_api_key: str
) -> Optional[Dict]:
    """
    Call Claude Sonnet 4.5 for editorial filtering.

    Args:
        ticker: Stock ticker
        company_name: Company name for context
        bullets_formatted: Formatted bullets string
        bullet_ids: List of bullet IDs to expect in response
        total_bullets: Total count for prompt
        max_removals: Max removals for prompt
        anthropic_api_key: API key

    Returns:
        Dict with decisions and metadata, or None if failed
    """
    try:
        # Build prompt
        prompt = EDITORIAL_FILTER_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            bullets_formatted=bullets_formatted,
            total_bullets=total_bullets,
            max_removals=max_removals
        )

        # Log sizes
        prompt_tokens_est = len(prompt) // 4
        LOG.info(f"[{ticker}] Phase 2.5 Claude prompt: ~{prompt_tokens_est} tokens, {total_bullets} bullets")

        headers = {
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        data = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 4000,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        LOG.info(f"[{ticker}] Phase 2.5: Calling Claude Sonnet 4.5 (primary)")

        # Retry logic
        max_retries = 2
        response = None
        generation_time_ms = 0

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                generation_time_ms = int((time.time() - start_time) * 1000)

                if response.status_code == 200:
                    break

                if response.status_code in [429, 500, 503] and attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] Phase 2.5 Claude error {response.status_code} (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    continue

                break

            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] Phase 2.5 Claude timeout (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] Phase 2.5 Claude timeout after {max_retries + 1} attempts")
                    return None

            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    LOG.warning(f"[{ticker}] Phase 2.5 Claude network error (attempt {attempt + 1}): {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] Phase 2.5 Claude network error: {e}")
                    return None

        if response is None:
            LOG.error(f"[{ticker}] Phase 2.5: No response from Claude")
            return None

        if response.status_code != 200:
            LOG.error(f"[{ticker}] Phase 2.5 Claude error {response.status_code}: {response.text[:500]}")
            return None

        # Parse response
        result = response.json()
        content = result.get("content", [{}])[0].get("text", "")

        if not content or len(content.strip()) < 5:
            LOG.error(f"[{ticker}] Phase 2.5: Claude returned empty response")
            return None

        # Parse decisions
        decisions = _parse_editorial_response(content, bullet_ids)

        if not decisions:
            LOG.error(f"[{ticker}] Phase 2.5: Failed to parse any decisions from Claude response")
            LOG.debug(f"[{ticker}] Response was: {content[:500]}")
            return None

        # Check coverage
        missing = set(bullet_ids) - set(decisions.keys())
        if missing:
            LOG.warning(f"[{ticker}] Phase 2.5: Claude missed {len(missing)} bullets: {list(missing)[:5]}")
            # Default missing bullets to KEEP
            for bid in missing:
                decisions[bid] = 'KEEP'

        # Get token counts
        usage = result.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        LOG.info(f"[{ticker}] Phase 2.5 Claude success: {prompt_tokens} prompt, {completion_tokens} completion, {generation_time_ms}ms")

        return {
            "decisions": decisions,
            "model_used": "claude-sonnet-4-5-20250929",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generation_time_ms": generation_time_ms,
            "raw_response": content
        }

    except Exception as e:
        LOG.error(f"[{ticker}] Phase 2.5 Claude exception: {e}", exc_info=True)
        return None


# =============================================================================
# GEMINI FALLBACK IMPLEMENTATION
# =============================================================================

def _call_gemini_editorial_filter(
    ticker: str,
    company_name: str,
    bullets_formatted: str,
    bullet_ids: List[str],
    total_bullets: int,
    max_removals: int,
    gemini_api_key: str
) -> Optional[Dict]:
    """
    Call Gemini 2.5 Pro for editorial filtering (fallback).

    Args:
        ticker: Stock ticker
        company_name: Company name for context
        bullets_formatted: Formatted bullets string
        bullet_ids: List of bullet IDs to expect in response
        total_bullets: Total count for prompt
        max_removals: Max removals for prompt
        gemini_api_key: API key

    Returns:
        Dict with decisions and metadata, or None if failed
    """
    try:
        import google.generativeai as genai

        genai.configure(api_key=gemini_api_key)

        # Build prompt
        prompt = EDITORIAL_FILTER_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            bullets_formatted=bullets_formatted,
            total_bullets=total_bullets,
            max_removals=max_removals
        )

        # Log sizes
        prompt_tokens_est = len(prompt) // 4
        LOG.info(f"[{ticker}] Phase 2.5 Gemini prompt: ~{prompt_tokens_est} tokens, {total_bullets} bullets")

        model = genai.GenerativeModel('gemini-2.5-pro')

        LOG.info(f"[{ticker}] Phase 2.5: Calling Gemini 2.5 Pro (fallback)")

        # Retry logic
        max_retries = 1
        response = None
        generation_time_ms = 0

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.0,
                        'max_output_tokens': 4000
                    }
                )
                generation_time_ms = int((time.time() - start_time) * 1000)
                break

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
                    LOG.warning(f"[{ticker}] Phase 2.5 Gemini error (attempt {attempt + 1}): {error_str[:200]}")
                    time.sleep(wait_time)
                    continue
                else:
                    LOG.error(f"[{ticker}] Phase 2.5 Gemini failed: {error_str}")
                    return None

        if response is None:
            LOG.error(f"[{ticker}] Phase 2.5: No response from Gemini")
            return None

        # Extract text
        response_text = response.text
        if not response_text or len(response_text.strip()) < 5:
            LOG.error(f"[{ticker}] Phase 2.5: Gemini returned empty response")
            return None

        # Parse decisions
        decisions = _parse_editorial_response(response_text, bullet_ids)

        if not decisions:
            LOG.error(f"[{ticker}] Phase 2.5: Failed to parse any decisions from Gemini response")
            LOG.debug(f"[{ticker}] Response was: {response_text[:500]}")
            return None

        # Check coverage
        missing = set(bullet_ids) - set(decisions.keys())
        if missing:
            LOG.warning(f"[{ticker}] Phase 2.5: Gemini missed {len(missing)} bullets: {list(missing)[:5]}")
            # Default missing bullets to KEEP
            for bid in missing:
                decisions[bid] = 'KEEP'

        # Get token counts
        prompt_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
        completion_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0

        LOG.info(f"[{ticker}] Phase 2.5 Gemini success: {prompt_tokens} prompt, {completion_tokens} completion, {generation_time_ms}ms")

        return {
            "decisions": decisions,
            "model_used": "gemini-2.5-pro",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generation_time_ms": generation_time_ms,
            "raw_response": response_text
        }

    except Exception as e:
        LOG.error(f"[{ticker}] Phase 2.5 Gemini exception: {e}", exc_info=True)
        return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_editorial_filter(
    phase2_json: Dict,
    ticker: str,
    company_name: str,
    anthropic_api_key: str,
    gemini_api_key: str
) -> Dict:
    """
    Run Phase 2.5 editorial filter on enriched bullets.

    Args:
        phase2_json: Phase 1+2 merged JSON with sections
        ticker: Stock ticker
        company_name: Company name for context
        anthropic_api_key: Anthropic API key
        gemini_api_key: Gemini API key

    Returns:
        Dict with:
            - success: bool
            - bypassed: bool (True if all retries failed)
            - decisions: Dict[bullet_id, 'KEEP'/'REMOVE']
            - cap_exceeded: bool
            - removal_stats: {total, keep_count, remove_count, removal_rate}
            - model_used: str
            - generation_time_ms: int
    """
    start_time = time.time()

    # Format bullets for prompt
    bullets_formatted, bullet_ids = _format_bullets_for_prompt(phase2_json)

    if not bullet_ids:
        LOG.warning(f"[{ticker}] Phase 2.5: No bullets to review")
        return {
            "success": True,
            "bypassed": False,
            "decisions": {},
            "cap_exceeded": False,
            "removal_stats": {"total": 0, "keep_count": 0, "remove_count": 0, "removal_rate": 0.0},
            "model_used": "none",
            "generation_time_ms": 0
        }

    total_bullets = len(bullet_ids)
    max_removals = ceil(total_bullets * 0.20)  # 20% cap

    LOG.info(f"[{ticker}] Phase 2.5: Reviewing {total_bullets} bullets (max {max_removals} removals)")

    # Try Claude first
    result = None
    if anthropic_api_key:
        result = _call_claude_editorial_filter(
            ticker=ticker,
            company_name=company_name,
            bullets_formatted=bullets_formatted,
            bullet_ids=bullet_ids,
            total_bullets=total_bullets,
            max_removals=max_removals,
            anthropic_api_key=anthropic_api_key
        )

    # Fallback to Gemini
    if result is None and gemini_api_key:
        LOG.info(f"[{ticker}] Phase 2.5: Claude failed, trying Gemini 2.5 Pro fallback")
        result = _call_gemini_editorial_filter(
            ticker=ticker,
            company_name=company_name,
            bullets_formatted=bullets_formatted,
            bullet_ids=bullet_ids,
            total_bullets=total_bullets,
            max_removals=max_removals,
            gemini_api_key=gemini_api_key
        )

    # Both failed - bypass
    if result is None:
        LOG.error(f"[{ticker}] Phase 2.5: All models failed, bypassing filter")
        total_time = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "bypassed": True,
            "decisions": {bid: 'KEEP' for bid in bullet_ids},
            "cap_exceeded": False,
            "removal_stats": {"total": total_bullets, "keep_count": total_bullets, "remove_count": 0, "removal_rate": 0.0},
            "model_used": "none",
            "generation_time_ms": total_time
        }

    # Apply removal cap
    decisions, cap_exceeded = _apply_removal_cap(result["decisions"], max_removals, ticker)

    # Calculate stats
    keep_count = sum(1 for d in decisions.values() if d == 'KEEP')
    remove_count = sum(1 for d in decisions.values() if d == 'REMOVE')
    removal_rate = remove_count / total_bullets if total_bullets > 0 else 0.0

    LOG.info(
        f"[{ticker}] Phase 2.5 Complete: {keep_count} KEEP, {remove_count} REMOVE "
        f"({removal_rate:.1%} removal rate), cap_exceeded={cap_exceeded}"
    )

    return {
        "success": True,
        "bypassed": False,
        "decisions": decisions,
        "cap_exceeded": cap_exceeded,
        "removal_stats": {
            "total": total_bullets,
            "keep_count": keep_count,
            "remove_count": remove_count,
            "removal_rate": removal_rate
        },
        "model_used": result.get("model_used", "unknown"),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "generation_time_ms": result.get("generation_time_ms", 0),
        "raw_response": result.get("raw_response", "")
    }


# =============================================================================
# EMAIL HTML GENERATOR
# =============================================================================

def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def generate_editorial_filter_email(
    ticker: str,
    company_name: str,
    phase2_json: Dict,
    filter_result: Dict
) -> str:
    """
    Generate HTML email showing editorial filter results.

    Args:
        ticker: Stock ticker
        company_name: Company name
        phase2_json: Phase 1+2 merged JSON (for bullet content)
        filter_result: Output from run_editorial_filter()

    Returns:
        HTML string for email body
    """
    decisions = filter_result.get("decisions", {})
    stats = filter_result.get("removal_stats", {})
    model = filter_result.get("model_used", "unknown")
    gen_time = filter_result.get("generation_time_ms", 0)
    bypassed = filter_result.get("bypassed", False)
    cap_exceeded = filter_result.get("cap_exceeded", False)

    total = stats.get("total", 0)
    keep_count = stats.get("keep_count", 0)
    remove_count = stats.get("remove_count", 0)
    removal_rate = stats.get("removal_rate", 0.0)

    # Get bullet content for display
    sections = phase2_json.get('sections', {})
    bullet_content = {}  # bullet_id -> {content, context, section}

    bullet_section_names = [
        'major_developments', 'financial_performance', 'risk_factors',
        'wall_street_sentiment', 'competitive_industry_dynamics',
        'upcoming_catalysts', 'key_variables'
    ]

    for section_name in bullet_section_names:
        section_data = sections.get(section_name, [])
        if not isinstance(section_data, list):
            continue
        for bullet in section_data:
            if not isinstance(bullet, dict):
                continue
            bullet_id = bullet.get('bullet_id', '')
            if bullet_id:
                content = bullet.get('content_integrated') or bullet.get('content', '')
                context = bullet.get('context', '')
                bullet_content[bullet_id] = {
                    'content': content,
                    'context': context,
                    'section': section_name
                }

    # Build HTML
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Warning banner if needed
    warning_html = ""
    if bypassed:
        warning_html = """
<div style="background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
    <strong>⚠️ Filter Bypassed</strong><br>
    All LLM attempts failed. Bullets passed through unfiltered. Manual review recommended.
</div>
"""
    elif cap_exceeded:
        warning_html = """
<div style="background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
    <strong>⚠️ Removal Cap Exceeded</strong><br>
    LLM wanted to remove more than 20% of bullets. All bullets kept (treating as systemic issue).
</div>
"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f9f9f9; }}
.header {{ background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
.header h2 {{ margin: 0 0 10px 0; }}
.header p {{ margin: 5px 0; opacity: 0.9; font-size: 14px; }}
.summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.stats-grid {{ display: flex; gap: 15px; margin-top: 15px; }}
.stat-box {{ flex: 1; text-align: center; padding: 15px; background: #f8f9fa; border-radius: 6px; }}
.stat-value {{ font-size: 28px; font-weight: bold; }}
.stat-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
.stat-keep {{ color: #28a745; }}
.stat-remove {{ color: #dc3545; }}
.bullet {{ background: white; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #ddd; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.bullet-keep {{ border-left-color: #28a745; }}
.bullet-remove {{ border-left-color: #dc3545; }}
.bullet-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
.bullet-id {{ font-family: monospace; font-size: 13px; color: #666; }}
.section-tag {{ font-size: 11px; color: #888; background: #f0f0f0; padding: 2px 8px; border-radius: 4px; }}
.badge {{ padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
.badge-keep {{ background: #d4edda; color: #155724; }}
.badge-remove {{ background: #f8d7da; color: #721c24; }}
.content {{ font-size: 14px; line-height: 1.5; color: #333; }}
.context {{ font-size: 13px; color: #666; margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 4px; border-left: 3px solid #6c757d; }}
.section-divider {{ background: #e9ecef; padding: 10px 15px; margin: 25px 0 15px 0; border-radius: 6px; font-weight: bold; color: #495057; }}
</style>
</head>
<body>

<div class="header">
    <h2>Phase 2.5: Editorial Filter - {ticker}</h2>
    <p><strong>Company:</strong> {_escape_html(company_name)}</p>
    <p><strong>Model:</strong> {model} ({gen_time}ms)</p>
    <p style="font-size: 12px; opacity: 0.7;">{timestamp}</p>
</div>

{warning_html}

<div class="summary">
    <strong>Filter Summary</strong>
    <div class="stats-grid">
        <div class="stat-box">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Total Bullets</div>
        </div>
        <div class="stat-box">
            <div class="stat-value stat-keep">{keep_count}</div>
            <div class="stat-label">Keep</div>
        </div>
        <div class="stat-box">
            <div class="stat-value stat-remove">{remove_count}</div>
            <div class="stat-label">Remove</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" style="color: {'#dc3545' if removal_rate > 0.15 else '#28a745'};">{removal_rate:.0%}</div>
            <div class="stat-label">Removal Rate</div>
        </div>
    </div>
</div>
"""

    # Group bullets by section
    bullets_by_section = {}
    for bullet_id, decision in decisions.items():
        info = bullet_content.get(bullet_id, {})
        section = info.get('section', 'unknown')
        if section not in bullets_by_section:
            bullets_by_section[section] = []
        bullets_by_section[section].append({
            'bullet_id': bullet_id,
            'decision': decision,
            'content': info.get('content', ''),
            'context': info.get('context', '')
        })

    # Section display names
    section_names = {
        'major_developments': '🔴 Major Developments',
        'financial_performance': '📊 Financial Performance',
        'risk_factors': '⚠️ Risk Factors',
        'wall_street_sentiment': '📈 Wall Street Sentiment',
        'competitive_industry_dynamics': '⚡ Competitive/Industry Dynamics',
        'upcoming_catalysts': '📅 Upcoming Catalysts',
        'key_variables': '🔍 Key Variables'
    }

    # Render bullets by section
    for section_key in bullet_section_names:
        bullets = bullets_by_section.get(section_key, [])
        if not bullets:
            continue

        section_display = section_names.get(section_key, section_key)
        html += f'<div class="section-divider">{section_display}</div>\n'

        for b in bullets:
            decision = b['decision']
            css_class = 'bullet-keep' if decision == 'KEEP' else 'bullet-remove'
            badge_class = 'badge-keep' if decision == 'KEEP' else 'badge-remove'

            content_preview = b['content'][:300] + '...' if len(b['content']) > 300 else b['content']

            html += f"""
<div class="bullet {css_class}">
    <div class="bullet-header">
        <span class="bullet-id">{b['bullet_id']}</span>
        <span class="badge {badge_class}">{decision}</span>
    </div>
    <div class="content">{_escape_html(content_preview)}</div>
"""
            if b['context']:
                context_preview = b['context'][:200] + '...' if len(b['context']) > 200 else b['context']
                html += f'    <div class="context"><strong>Context:</strong> {_escape_html(context_preview)}</div>\n'

            html += '</div>\n'

    html += """
</body>
</html>
"""

    return html


def apply_editorial_filter(phase2_json: Dict, filter_result: Dict) -> Dict:
    """
    Apply editorial filter decisions to Phase 2 JSON.

    Removes bullets marked as REMOVE from the sections.

    Args:
        phase2_json: Phase 1+2 merged JSON
        filter_result: Output from run_editorial_filter()

    Returns:
        Modified Phase 2 JSON with filtered bullets removed (deep copy)
    """
    import copy

    decisions = filter_result.get("decisions", {})

    # Deep copy to avoid modifying original
    result = copy.deepcopy(phase2_json)
    sections = result.get('sections', {})

    bullet_section_names = [
        'major_developments', 'financial_performance', 'risk_factors',
        'wall_street_sentiment', 'competitive_industry_dynamics',
        'upcoming_catalysts', 'key_variables'
    ]

    removed_count = 0
    kept_count = 0

    for section_name in bullet_section_names:
        section_data = sections.get(section_name, [])
        if not isinstance(section_data, list):
            continue

        # Filter bullets
        new_bullets = []
        for bullet in section_data:
            if not isinstance(bullet, dict):
                new_bullets.append(bullet)
                continue

            bullet_id = bullet.get('bullet_id', '')
            decision = decisions.get(bullet_id, 'KEEP')

            if decision == 'REMOVE':
                removed_count += 1
                continue  # Skip this bullet
            else:
                new_bullets.append(bullet)
                kept_count += 1

        sections[section_name] = new_bullets

    LOG.info(f"Editorial filter applied: {kept_count} kept, {removed_count} removed")

    return result
