"""
Phase 2 Executive Summary Enrichment Module

This module enriches Phase 1 executive summaries with filing context from 10-K, 10-Q, and Transcripts.

Phase 2 receives:
- Phase 1 JSON output (complete structure)
- Latest 10-K, 10-Q, Transcript from database (1-3 available)

Phase 2 returns:
- Enrichments dict keyed by bullet_id with: impact, sentiment, reason, context

The enrichments are then merged into Phase 1 JSON for final output.
"""

import json
import logging
import os
import copy
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Initialize logger
LOG = logging.getLogger(__name__)

# Constants
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


def get_phase2_system_prompt() -> str:
    """
    Load Phase 2 system prompt from file.

    The prompt is static (no ticker substitution) for optimal prompt caching.
    Ticker context is provided in user_content instead.

    Returns:
        str: Phase 2 system prompt
    """
    try:
        prompt_path = os.path.join(os.path.dirname(__file__), '_build_executive_summary_prompt_phase2')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        return prompt_template
    except Exception as e:
        LOG.error(f"Failed to load Phase 2 prompt: {e}")
        raise


def _fetch_available_filings(ticker: str, db_func) -> Dict[str, Dict]:
    """
    Fetch latest 10-K, 10-Q, and Transcript from database.

    Args:
        ticker: Stock ticker symbol
        db_func: Database connection function

    Returns:
        dict with keys '10k', '10q', 'transcript' (only keys present if data available)
        Each value is dict with: text, fiscal_year/quarter, filing_date/report_date, company_name
    """
    filings = {}

    try:
        with db_func() as conn, conn.cursor() as cur:
            # 1. Latest Transcript (prefer Claude if multiple exist for same period)
            cur.execute("""
                SELECT summary_text, quarter, year, report_date, company_name, ai_provider
                FROM transcript_summaries
                WHERE ticker = %s AND report_type = 'transcript'
                ORDER BY year DESC, quarter DESC,
                         CASE WHEN ai_provider = 'claude' THEN 0 ELSE 1 END
                LIMIT 1
            """, (ticker,))

            row = cur.fetchone()
            if row and row['summary_text']:
                filings['transcript'] = {
                    'text': row['summary_text'],
                    'quarter': row['quarter'],
                    'year': row['year'],
                    'date': row['report_date'],
                    'company_name': row['company_name']
                }
                LOG.debug(f"[{ticker}] Found Transcript: {row['quarter']} {row['year']}")

            # 2. Latest 10-Q
            cur.execute("""
                SELECT profile_markdown, fiscal_year, fiscal_quarter, filing_date, company_name
                FROM sec_filings
                WHERE ticker = %s AND filing_type = '10-Q'
                ORDER BY fiscal_year DESC, fiscal_quarter DESC
                LIMIT 1
            """, (ticker,))

            row = cur.fetchone()
            if row and row['profile_markdown']:
                filings['10q'] = {
                    'text': row['profile_markdown'],
                    'fiscal_year': row['fiscal_year'],
                    'fiscal_quarter': row['fiscal_quarter'],
                    'filing_date': row['filing_date'],
                    'company_name': row['company_name']
                }
                LOG.debug(f"[{ticker}] Found 10-Q: {row['fiscal_quarter']} {row['fiscal_year']}")

            # 3. Latest 10-K
            cur.execute("""
                SELECT profile_markdown, fiscal_year, filing_date, company_name
                FROM sec_filings
                WHERE ticker = %s AND filing_type = '10-K'
                ORDER BY fiscal_year DESC
                LIMIT 1
            """, (ticker,))

            row = cur.fetchone()
            if row and row['profile_markdown']:
                filings['10k'] = {
                    'text': row['profile_markdown'],
                    'fiscal_year': row['fiscal_year'],
                    'filing_date': row['filing_date'],
                    'company_name': row['company_name']
                }
                LOG.debug(f"[{ticker}] Found 10-K: FY{row['fiscal_year']}")

    except Exception as e:
        LOG.error(f"[{ticker}] Failed to fetch filings: {e}")
        return {}

    return filings


def _convert_emoji_headers_to_markdown(text: str) -> str:
    """
    Convert emoji section headers to markdown headers for Phase 2 compatibility.

    Phase 2 expects markdown format (# SECTION NAME), but transcript summaries
    use emoji format (📌 BOTTOM LINE).

    Args:
        text: Transcript summary text with emoji headers

    Returns:
        Text with markdown headers (emojis stripped, # prefix added)
    """
    import re

    # Define emoji-to-markdown mappings
    emoji_headers = [
        "📌 BOTTOM LINE",
        "💰 FINANCIAL RESULTS",
        "📊 OPERATIONAL METRICS",
        "🏢 MAJOR DEVELOPMENTS",
        "📈 GUIDANCE",
        "🎯 STRATEGIC INITIATIVES",
        "💼 MANAGEMENT SENTIMENT",
        "⚠️ RISK FACTORS",
        "🏭 INDUSTRY",
        "💡 CAPITAL ALLOCATION",
        "💬 Q&A HIGHLIGHTS",
        "📈 UPSIDE SCENARIO",
        "📉 DOWNSIDE SCENARIO",
        "🔍 KEY VARIABLES"
    ]

    # Replace each emoji header with markdown format
    for emoji_header in emoji_headers:
        # Extract text without emoji
        text_only = re.sub(r'^[^\w\s]+\s*', '', emoji_header)
        # Replace in text (must be at start of line)
        pattern = r'^' + re.escape(emoji_header)
        replacement = f'# {text_only}'
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

    return text


def _build_phase2_user_content(ticker: str, phase1_json: Dict, filings: Dict) -> str:
    """
    Build Phase 2 user content combining Phase 1 JSON and filing sources.

    Format matches old _build_executive_summary_prompt() structure:
    - Phase 1 JSON first
    - Then filing sources with proper headers (Transcript, 10-Q, 10-K)

    Args:
        ticker: Stock ticker symbol
        phase1_json: Complete Phase 1 JSON output
        filings: Dict with keys '10k', '10q', 'transcript'

    Returns:
        str: Formatted user content for Phase 2 prompt
    """
    # Start with Phase 1 JSON
    phase1_str = json.dumps(phase1_json, indent=2)
    content = f"PHASE 1 ANALYSIS (TO BE ENRICHED):\n\n{phase1_str}\n\n"
    content += "---\n\n"
    content += "AVAILABLE FILING SOURCES FOR CONTEXT:\n\n"

    # Add Transcript if available (matches old format)
    if 'transcript' in filings:
        t = filings['transcript']
        quarter = t['quarter']
        year = t['year']
        company = t['company_name'] or ticker
        date = t['date'].strftime('%b %d, %Y') if t['date'] else 'Unknown Date'

        # Convert emoji headers to markdown for Phase 2 compatibility
        transcript_text = _convert_emoji_headers_to_markdown(t['text'])

        content += f"LATEST EARNINGS CALL (TRANSCRIPT):\n\n"
        content += f"[{ticker} ({company}) {quarter} {year} Earnings Call ({date})]\n\n"
        content += f"{transcript_text}\n\n\n"

    # Add 10-Q if available (matches old format)
    if '10q' in filings:
        q = filings['10q']
        quarter = q['fiscal_quarter']
        year = q['fiscal_year']
        company = q['company_name'] or ticker
        date = q['filing_date'].strftime('%b %d, %Y') if q['filing_date'] else 'Unknown Date'

        content += f"LATEST QUARTERLY REPORT (10-Q):\n\n"
        content += f"[{ticker} ({company}) {quarter} {year} 10-Q Filing, Filed: {date}]\n\n"
        content += f"{q['text']}\n\n\n"

    # Add 10-K if available (matches old format)
    if '10k' in filings:
        k = filings['10k']
        year = k['fiscal_year']
        company = k['company_name'] or ticker
        date = k['filing_date'].strftime('%b %d, %Y') if k['filing_date'] else 'Unknown Date'

        content += f"COMPANY 10-K PROFILE:\n\n"
        content += f"[{ticker} ({company}) 10-K FILING FOR FISCAL YEAR {year}, Filed: {date}]\n\n"
        content += f"{k['text']}\n\n\n"

    return content


def generate_executive_summary_phase2(
    ticker: str,
    phase1_json: Dict,
    filings: Dict,
    config: Dict,
    anthropic_api_key: str,
    db_func
) -> Optional[Dict]:
    """
    Generate Phase 2 enrichments using Claude API.

    Takes Phase 1 JSON output and filing sources, returns enrichments dict.

    Args:
        ticker: Stock ticker symbol
        phase1_json: Complete Phase 1 JSON output
        filings: Dict with keys '10k', '10q', 'transcript' (from _fetch_available_filings)
        config: Ticker configuration dict
        anthropic_api_key: Anthropic API key
        db_func: Database connection function

    Returns:
        dict with:
            enrichments: dict keyed by bullet_id with impact, sentiment, reason, context
            model_used: "claude"
            prompt_tokens: int
            completion_tokens: int
            generation_time_ms: int
        Or None if failed
    """
    import time

    try:
        # Load system prompt (static, cached)
        system_prompt = get_phase2_system_prompt()

        # Build user content (Phase 1 JSON + filings)
        user_content = _build_phase2_user_content(ticker, phase1_json, filings)

        # Estimate token counts for logging
        system_tokens_est = len(system_prompt) // 4
        user_tokens_est = len(user_content) // 4
        total_tokens_est = system_tokens_est + user_tokens_est
        LOG.info(f"[{ticker}] Phase 2 prompt size: system={len(system_prompt)} chars (~{system_tokens_est} tokens), user={len(user_content)} chars (~{user_tokens_est} tokens), total=~{total_tokens_est} tokens")

        # Prepare API call
        headers = {
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01",  # Prompt caching support
            "content-type": "application/json"
        }

        data = {
            "model": "claude-sonnet-4-5-20250929",  # Sonnet 4.5
            "max_tokens": 20000,  # Generous limit for enrichments
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

        LOG.info(f"[{ticker}] Calling Claude API for Phase 2 enrichment")

        # Retry logic for transient errors (503, 429, 500)
        max_retries = 2
        response = None
        generation_time_ms = 0

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = requests.post(ANTHROPIC_API_URL, headers=headers, json=data, timeout=180)
                generation_time_ms = int((time.time() - start_time) * 1000)

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

            # Extract usage stats
            usage = result.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)

            # Log cache performance
            cache_creation = usage.get("cache_creation_input_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            if cache_creation > 0:
                LOG.info(f"[{ticker}] 💾 CACHE CREATED: {cache_creation} tokens cached (Phase 2)")
            elif cache_read > 0:
                LOG.info(f"[{ticker}] ⚡ CACHE HIT: {cache_read} tokens read from cache (Phase 2) - 90% savings!")

            # Parse JSON response
            response_text = result.get("content", [{}])[0].get("text", "")

            if not response_text or len(response_text.strip()) < 10:
                LOG.error(f"❌ [{ticker}] Claude returned empty Phase 2 response")
                return None

            # Parse JSON from response
            try:
                # Extract JSON if wrapped in markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_str = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    json_str = response_text[json_start:json_end].strip()
                else:
                    json_str = response_text.strip()

                parsed_json = json.loads(json_str)

                # Handle different possible structures from Claude
                # Claude might return:
                # 1. {"sections": {"major_developments": [...]}} - full structure
                # 2. {"enrichments": {"FIN_001": {...}}} - wrapped enrichments
                # 3. {"FIN_001": {...}, "bottom_line_context": "..."} - direct enrichments + scenario contexts (NEW)

                enrichments = {}
                scenario_contexts = {}

                if "enrichments" in parsed_json and isinstance(parsed_json["enrichments"], dict):
                    # Case 2: Wrapped in "enrichments" key
                    enrichments = parsed_json["enrichments"]
                    # Extract scenario contexts if present at root level
                    for key in ["bottom_line_context", "upside_scenario_context", "downside_scenario_context"]:
                        if key in parsed_json:
                            scenario_contexts[key] = parsed_json[key]
                elif "sections" in parsed_json and isinstance(parsed_json["sections"], dict):
                    # Case 1: Full section structure - need to flatten to bullet_id dict
                    for section_name, bullets in parsed_json["sections"].items():
                        if isinstance(bullets, list):
                            for bullet in bullets:
                                if isinstance(bullet, dict) and "bullet_id" in bullet:
                                    bid = bullet["bullet_id"]
                                    enrichments[bid] = {
                                        "impact": bullet.get("impact"),
                                        "sentiment": bullet.get("sentiment"),
                                        "reason": bullet.get("reason"),
                                        "relevance": bullet.get("relevance"),
                                        "context": bullet.get("context")
                                    }
                    # Extract scenario contexts if present at root level
                    for key in ["bottom_line_context", "upside_scenario_context", "downside_scenario_context"]:
                        if key in parsed_json:
                            scenario_contexts[key] = parsed_json[key]
                else:
                    # Case 3: Direct enrichments dict + scenario contexts
                    # Separate bullet enrichments from scenario contexts
                    for key, value in parsed_json.items():
                        if key in ["bottom_line_context", "upside_scenario_context", "downside_scenario_context"]:
                            scenario_contexts[key] = value
                        else:
                            # Assume it's a bullet enrichment (dict with impact, sentiment, etc.)
                            enrichments[key] = value

                # Debug logging: Show sample of what Claude actually returned (before validation)
                if enrichments:
                    sample_size = min(3, len(enrichments))
                    sample_bullets = list(enrichments.items())[:sample_size]
                    LOG.info(f"[{ticker}] 📋 Phase 2 raw output sample ({sample_size}/{len(enrichments)} bullets, BEFORE validation):")
                    for bullet_id, data in sample_bullets:
                        if isinstance(data, dict):
                            present_fields = [f for f in data.keys() if data.get(f)]
                            empty_fields = [f for f in data.keys() if not data.get(f)]
                            LOG.info(f"  • {bullet_id}:")
                            LOG.info(f"      ✓ Present: {', '.join(present_fields) if present_fields else 'NONE'}")
                            if empty_fields:
                                LOG.info(f"      ✗ Empty/None: {', '.join(empty_fields)}")
                            # Show first 60 chars of each field value
                            for field in ["impact", "sentiment", "reason", "relevance", "context"]:
                                if field in data:
                                    value = data[field]
                                    if value:
                                        preview = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)
                                        LOG.info(f"      → {field}: {preview}")
                                    else:
                                        LOG.info(f"      → {field}: (empty)")
                        else:
                            LOG.info(f"  • {bullet_id}: ⚠️ NOT A DICT (type={type(data).__name__})")

                # Debug logging: Show scenario contexts if present
                if scenario_contexts:
                    LOG.info(f"[{ticker}] 📄 Phase 2 scenario contexts found: {', '.join(scenario_contexts.keys())}")
                    for key, value in scenario_contexts.items():
                        if value:
                            preview = value[:80] + "..." if len(value) > 80 else value
                            LOG.info(f"  • {key}: {preview}")
                        else:
                            LOG.info(f"  • {key}: (empty)")
                else:
                    LOG.info(f"[{ticker}] 📄 No scenario contexts in Phase 2 output")

                LOG.info(f"✅ [{ticker}] Phase 2 enrichment generated ({len(json_str)} chars, {len(enrichments)} bullets enriched, {prompt_tokens} prompt tokens, {completion_tokens} completion tokens, {generation_time_ms}ms)")

                return {
                    "enrichments": enrichments,
                    "scenario_contexts": scenario_contexts,
                    "model_used": "claude",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "generation_time_ms": generation_time_ms
                }

            except json.JSONDecodeError as e:
                LOG.error(f"❌ [{ticker}] Failed to parse Phase 2 JSON: {e}")
                LOG.error(f"Response preview: {response_text[:500]}")
                return None

        else:
            error_text = response.text[:500] if response.text else "No error details"
            LOG.error(f"❌ [{ticker}] Claude API error {response.status_code} after {max_retries + 1} attempts: {error_text}")
            return None

    except Exception as e:
        LOG.error(f"❌ [{ticker}] Exception in Phase 2 generation: {e}")
        return None


def validate_phase2_json(enrichments: Dict) -> Tuple[bool, str, Dict]:
    """
    Validate Phase 2 enrichments structure with partial acceptance.

    This function filters out incomplete/invalid bullets but accepts the rest,
    preventing one bad bullet from destroying all Phase 2 enrichment work.

    Expected structure:
    {
        "bullet_id_1": {
            "impact": "high impact|medium impact|low impact",
            "sentiment": "bullish|bearish|neutral",
            "reason": "brief reason string",
            "relevance": "direct|indirect|none",
            "context": "prose paragraph combining filing excerpts"
        },
        "bullet_id_2": { ... }
    }

    Args:
        enrichments: Dict keyed by bullet_id

    Returns:
        Tuple of (is_valid: bool, error_message: str, valid_enrichments: Dict)
        - is_valid: True if ANY bullets are valid
        - error_message: Description of what was accepted/rejected
        - valid_enrichments: Dict containing only complete, valid bullets
    """
    if not isinstance(enrichments, dict):
        return False, "Enrichments must be object/dict", {}

    if not enrichments:
        return False, "Enrichments dict is empty", {}

    required_fields = ["impact", "sentiment", "reason", "relevance", "context"]
    valid_enrichments = {}
    invalid_bullets = []

    for bullet_id, data in enrichments.items():
        # Check if data is a dict
        if not isinstance(data, dict):
            invalid_bullets.append(f"{bullet_id} (not a dict)")
            continue

        # Check for missing fields
        missing_fields = [f for f in required_fields if f not in data or not data.get(f)]
        if missing_fields:
            # Enhanced error message: show what IS present too
            present_fields = [f for f in required_fields if f in data and data.get(f)]
            error_detail = f"missing: {', '.join(missing_fields)}"
            if present_fields:
                error_detail += f" | present: {', '.join(present_fields)}"
            invalid_bullets.append(f"{bullet_id} ({error_detail})")
            continue

        # Validate impact values
        if data["impact"] not in ["high impact", "medium impact", "low impact"]:
            # Show what valid fields exist despite bad impact
            other_valid = [f for f in ["sentiment", "reason", "relevance", "context"] if f in data and data.get(f)]
            error_detail = f"invalid impact: '{data['impact']}'"
            if other_valid:
                error_detail += f" | valid: {', '.join(other_valid)}"
            invalid_bullets.append(f"{bullet_id} ({error_detail})")
            continue

        # Validate sentiment values
        if data["sentiment"] not in ["bullish", "bearish", "neutral", "mixed"]:
            other_valid = [f for f in ["impact", "reason", "relevance", "context"] if f in data and data.get(f)]
            error_detail = f"invalid sentiment: '{data['sentiment']}'"
            if other_valid:
                error_detail += f" | valid: {', '.join(other_valid)}"
            invalid_bullets.append(f"{bullet_id} ({error_detail})")
            continue

        # Validate relevance values
        if data["relevance"] not in ["direct", "indirect", "none"]:
            other_valid = [f for f in ["impact", "sentiment", "reason", "context"] if f in data and data.get(f)]
            error_detail = f"invalid relevance: '{data['relevance']}'"
            if other_valid:
                error_detail += f" | valid: {', '.join(other_valid)}"
            invalid_bullets.append(f"{bullet_id} ({error_detail})")
            continue

        # Bullet passed all validation checks!
        valid_enrichments[bullet_id] = data

    # If no valid enrichments, Phase 2 completely failed
    if not valid_enrichments:
        if invalid_bullets:
            # Show ALL invalid bullets when complete failure (not just first 3)
            error_detail = '; '.join(invalid_bullets[:5])
            if len(invalid_bullets) > 5:
                error_detail += f" (+{len(invalid_bullets) - 5} more with similar issues)"
        else:
            error_detail = "No enrichments provided"
        return False, f"No valid enrichments found ({len(invalid_bullets)} bullets failed). Issues: {error_detail}", {}

    # At least some enrichments are valid - accept them
    if invalid_bullets:
        # Partial success - some bullets filtered out
        error_msg = f"Accepted {len(valid_enrichments)}/{len(enrichments)} bullets. Filtered out: {'; '.join(invalid_bullets[:3])}"
        if len(invalid_bullets) > 3:
            error_msg += f" (+{len(invalid_bullets) - 3} more)"
    else:
        # Complete success - all bullets valid
        error_msg = f"All {len(valid_enrichments)} bullets validated successfully"

    return True, error_msg, valid_enrichments


def strip_escape_hatch_context(phase2_result: Dict) -> Dict:
    """
    Replace escape hatch text with empty string for cleaner display.

    When Phase 2 finds no relevant filing context, it outputs:
    "No relevant filing context found for this development"

    This function replaces that text with "" so templates can simply check
    if context exists, without seeing "not found" messages in the UI.

    Args:
        phase2_result: Phase 2 result dict with 'enrichments' and 'scenario_contexts' keys

    Returns:
        Modified phase2_result with escape hatch text replaced
    """
    ESCAPE_HATCH = "No relevant filing context found for this development"

    # Strip escape hatch from bullet enrichments
    enrichments = phase2_result.get("enrichments", {})
    for bullet_id, enrichment in enrichments.items():
        if enrichment.get("context") == ESCAPE_HATCH:
            enrichment["context"] = ""

    # Strip escape hatch from scenario contexts
    scenario_contexts = phase2_result.get("scenario_contexts", {})
    for key in ["bottom_line_context", "upside_scenario_context", "downside_scenario_context"]:
        if scenario_contexts.get(key) == ESCAPE_HATCH:
            scenario_contexts[key] = ""

    return phase2_result


def merge_phase1_phase2(phase1_json: Dict, phase2_result: Dict) -> Dict:
    """
    Merge Phase 2 enrichments and scenario contexts into Phase 1 JSON.

    Takes Phase 1 JSON structure and:
    1. Adds impact, sentiment, reason, relevance, context fields to each bullet
    2. Adds context field to paragraph sections (bottom_line, upside_scenario, downside_scenario)

    Args:
        phase1_json: Complete Phase 1 JSON output
        phase2_result: Phase 2 result dict with 'enrichments' and 'scenario_contexts' keys

    Returns:
        Merged JSON with Phase 2 fields added to bullets and scenarios
    """
    merged = copy.deepcopy(phase1_json)
    enrichments = phase2_result.get("enrichments", {})
    scenario_contexts = phase2_result.get("scenario_contexts", {})

    # Merge bullet enrichments
    if enrichments:
        # List of sections with bullets (not paragraphs)
        bullet_sections = [
            "major_developments",
            "financial_performance",
            "risk_factors",
            "wall_street_sentiment",
            "competitive_industry_dynamics",
            "upcoming_catalysts",
            "key_variables"
        ]

        # Iterate through all sections
        for section_name in bullet_sections:
            if section_name not in merged.get("sections", {}):
                continue

            section_content = merged["sections"][section_name]

            if not isinstance(section_content, list):
                continue

            # Enrich each bullet
            for bullet in section_content:
                if not isinstance(bullet, dict):
                    continue

                bullet_id = bullet.get("bullet_id")
                if not bullet_id or bullet_id not in enrichments:
                    continue

                # Add Phase 2 enrichment fields
                enrichment = enrichments[bullet_id]
                bullet["impact"] = enrichment.get("impact")
                bullet["sentiment"] = enrichment.get("sentiment")
                bullet["reason"] = enrichment.get("reason")
                bullet["relevance"] = enrichment.get("relevance")
                bullet["context"] = enrichment.get("context")

    # Merge scenario contexts into paragraph sections
    if scenario_contexts:
        # Add context to bottom_line
        if "bottom_line_context" in scenario_contexts and scenario_contexts["bottom_line_context"]:
            if "bottom_line" in merged.get("sections", {}):
                merged["sections"]["bottom_line"]["context"] = scenario_contexts["bottom_line_context"]

        # Add context to upside_scenario
        if "upside_scenario_context" in scenario_contexts and scenario_contexts["upside_scenario_context"]:
            if "upside_scenario" in merged.get("sections", {}):
                merged["sections"]["upside_scenario"]["context"] = scenario_contexts["upside_scenario_context"]

        # Add context to downside_scenario
        if "downside_scenario_context" in scenario_contexts and scenario_contexts["downside_scenario_context"]:
            if "downside_scenario" in merged.get("sections", {}):
                merged["sections"]["downside_scenario"]["context"] = scenario_contexts["downside_scenario_context"]

    return merged
