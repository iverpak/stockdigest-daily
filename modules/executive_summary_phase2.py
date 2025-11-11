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


def format_entity_references(config: Dict) -> Dict[str, str]:
    """
    Extract and format competitor/upstream/downstream entities from ticker config.

    Formats entities as: "Company Name (TICKER)" or "Company Name" if no ticker.
    Used to populate entity reference section in Phase 2 prompt.

    Args:
        config: Ticker configuration dict from ticker_reference

    Returns:
        dict with keys: 'competitors', 'upstream', 'downstream'
        Each value is comma-separated string or 'None' if empty.

    Example output:
        {
            'competitors': 'SolarEdge Technologies (SEDG), Generac Holdings (GNRC)',
            'upstream': 'ON Semiconductor (ON), Texas Instruments (TXN)',
            'downstream': 'Sunrun Inc. (RUN), Tesla, Inc. (TSLA)'
        }
    """
    if not config:
        return {
            'competitors': 'None',
            'upstream': 'None',
            'downstream': 'None'
        }

    competitors = []
    upstream = []
    downstream = []

    # Extract competitors
    for comp in config.get("competitors", []):
        if isinstance(comp, dict):
            name = comp.get("name", "")
            ticker_sym = comp.get("ticker", "")
            if name and ticker_sym:
                competitors.append(f"{name} ({ticker_sym})")
            elif name:
                competitors.append(name)

    # Extract upstream
    value_chain = config.get("value_chain", {})
    for comp in value_chain.get("upstream", []):
        if isinstance(comp, dict):
            name = comp.get("name", "")
            ticker_sym = comp.get("ticker", "")
            if name and ticker_sym:
                upstream.append(f"{name} ({ticker_sym})")
            elif name:
                upstream.append(name)

    # Extract downstream
    for comp in value_chain.get("downstream", []):
        if isinstance(comp, dict):
            name = comp.get("name", "")
            ticker_sym = comp.get("ticker", "")
            if name and ticker_sym:
                downstream.append(f"{name} ({ticker_sym})")
            elif name:
                downstream.append(name)

    return {
        'competitors': ', '.join(competitors) if competitors else 'None',
        'upstream': ', '.join(upstream) if upstream else 'None',
        'downstream': ', '.join(downstream) if downstream else 'None'
    }


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

        # Format entity references from config
        entity_refs = format_entity_references(config)

        # Inject entity references into prompt template
        # Note: Using .replace() instead of .format() to avoid conflicts with JSON examples in prompt
        system_prompt = system_prompt.replace('{competitor_list}', entity_refs['competitors'])
        system_prompt = system_prompt.replace('{upstream_list}', entity_refs['upstream'])
        system_prompt = system_prompt.replace('{downstream_list}', entity_refs['downstream'])

        # Log entity references for debugging
        LOG.debug(f"[{ticker}] Entity references: Competitors={entity_refs['competitors']}, "
                  f"Upstream={entity_refs['upstream']}, Downstream={entity_refs['downstream']}")

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


def validate_phase2_json(enrichments: Dict, phase1_json: Dict = None, ticker: str = "") -> Tuple[bool, str, Dict]:
    """
    Validate Phase 2 enrichments structure with partial acceptance.

    This function filters out incomplete/invalid bullets but accepts the rest,
    preventing one bad bullet from destroying all Phase 2 enrichment work.

    Expected structure:
    {
        "bullet_id_1": {
            "context": "prose paragraph combining filing excerpts",
            "impact": "high impact|medium impact|low impact",
            "sentiment": "bullish|bearish|neutral",
            "reason": "brief reason string",
            "relevance": "direct|indirect|none",
            "entity": "Competitor|Market|Upstream|Downstream" (ONLY for competitive_industry_dynamics)
        },
        "bullet_id_2": { ... }
    }

    Args:
        enrichments: Dict keyed by bullet_id
        phase1_json: Optional Phase 1 JSON to determine bullet sections
        ticker: Optional ticker for logging

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

    # Build bullet_id to section mapping
    bullet_sections = {}
    if phase1_json and "sections" in phase1_json:
        for section_name, section_data in phase1_json["sections"].items():
            if isinstance(section_data, list):
                for bullet in section_data:
                    if isinstance(bullet, dict) and "bullet_id" in bullet:
                        bullet_sections[bullet["bullet_id"]] = section_name

    valid_enrichments = {}
    invalid_bullets = []

    for bullet_id, data in enrichments.items():
        # Check if data is a dict
        if not isinstance(data, dict):
            invalid_bullets.append(f"{bullet_id} (not a dict)")
            continue

        # Determine which section this bullet belongs to
        section_name = bullet_sections.get(bullet_id, "unknown")
        is_competitive_industry = (section_name == "competitive_industry_dynamics")

        # Set required fields based on section
        if is_competitive_industry:
            required_fields = ["context", "impact", "sentiment", "reason", "relevance", "entity"]
        else:
            required_fields = ["context", "impact", "sentiment", "reason", "relevance"]

        # Fill in missing fields with empty string (accept partial data)
        missing_fields = [f for f in required_fields if f not in data or not data.get(f)]
        for field in required_fields:
            if field not in data or not data.get(field):
                data[field] = ""  # Leave blank, don't reject

        # Validate entity values if present and not empty (for competitive_industry_dynamics only)
        if is_competitive_industry and data.get("entity"):
            valid_entities = ["Competitor", "Market", "Upstream", "Downstream"]
            if data["entity"] not in valid_entities:
                LOG.warning(f"[{ticker}] Phase 2: {bullet_id} has invalid entity '{data['entity']}', setting to empty")
                data["entity"] = ""

        # Log what was missing for debugging
        if missing_fields:
            LOG.info(f"[{ticker}] Phase 2: {bullet_id} accepted with missing fields: {', '.join(missing_fields)}")
            LOG.debug(f"[{ticker}] Phase 2: {bullet_id} full data: {json.dumps(data, indent=2)}")

        # Accept bullet with partial data (no value validation, only check if empty)
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


def sort_bullets_by_impact(bullets: List[Dict]) -> List[Dict]:
    """
    Sort bullets by impact level: high → medium → low → missing.

    Uses stable sort to preserve original order within same impact level.
    Un-enriched bullets (missing impact field) sink to bottom.

    Args:
        bullets: List of bullet dicts

    Returns:
        Sorted list of bullets
    """
    impact_order = {
        'high impact': 0,
        'medium impact': 1,
        'low impact': 2
    }

    def get_sort_key(bullet):
        impact = bullet.get('impact')
        if impact is None:
            return 999  # Un-enriched bullets go to bottom
        return impact_order.get(impact, 999)  # Unknown impact values go to bottom

    # Stable sort preserves original order for ties
    return sorted(bullets, key=get_sort_key)


def merge_phase1_phase2(phase1_json: Dict, phase2_result: Dict) -> Dict:
    """
    Merge Phase 2 enrichments and scenario contexts into Phase 1 JSON.

    Takes Phase 1 JSON structure and:
    1. Adds impact, sentiment, reason, relevance, context fields to each bullet
    2. Adds context field to paragraph sections (bottom_line, upside_scenario, downside_scenario)
    3. Sorts enriched bullet sections by impact (high → medium → low → missing)

    Args:
        phase1_json: Complete Phase 1 JSON output
        phase2_result: Phase 2 result dict with 'enrichments' and 'scenario_contexts' keys

    Returns:
        Merged JSON with Phase 2 fields added to bullets and scenarios, sorted by impact
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

                enrichment = enrichments[bullet_id]

                # HARD FILTER: Never merge context for wall_street_sentiment
                # Analyst opinions ARE the context - they already synthesize filing data
                if section_name == "wall_street_sentiment":
                    # Keep metadata (impact, sentiment, reason, relevance are valid)
                    bullet["impact"] = enrichment.get("impact")
                    bullet["sentiment"] = enrichment.get("sentiment")
                    bullet["reason"] = enrichment.get("reason")
                    bullet["relevance"] = enrichment.get("relevance")
                    # Force context to empty string (strip any generated context)
                    bullet["context"] = ""
                    if enrichment.get("context"):
                        LOG.warning(f"Stripped filing context from Wall Street bullet {bullet_id} "
                                   f"(context length: {len(enrichment.get('context', ''))} chars)")
                    continue  # Skip standard enrichment path

                # Standard enrichment for all other sections
                bullet["impact"] = enrichment.get("impact")
                bullet["sentiment"] = enrichment.get("sentiment")
                bullet["reason"] = enrichment.get("reason")
                bullet["relevance"] = enrichment.get("relevance")
                bullet["context"] = enrichment.get("context")
                bullet["entity"] = enrichment.get("entity")

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

    # Sort enriched bullet sections by impact (high → medium → low → missing)
    # Only sort sections that receive Phase 2 enrichments with impact field
    enriched_sections = [
        "major_developments",
        "financial_performance",
        "risk_factors",
        "wall_street_sentiment",
        "competitive_industry_dynamics"
    ]

    for section_name in enriched_sections:
        if section_name in merged.get("sections", {}):
            section_content = merged["sections"][section_name]
            if isinstance(section_content, list) and len(section_content) > 0:
                merged["sections"][section_name] = sort_bullets_by_impact(section_content)

    return merged


def merge_phase3_with_phase2(phase2_json: Dict, phase3_json: Dict) -> Dict:
    """
    Merge Phase 3 integrated content back into Phase 2 JSON using bullet_id matching.

    Phase 2 has: All metadata (impact, sentiment, reason, entity, date_range, filing_hints, context (original))
    Phase 3 has: Only bullet_id, topic_label, content (integrated)

    Result: Phase 2 metadata + Phase 3 integrated content

    Args:
        phase2_json: Phase 1+2 merged JSON with all metadata
        phase3_json: Phase 3 output with integrated content only

    Returns:
        Final merged JSON with Phase 2 metadata + Phase 3 integrated content
    """
    import copy

    # Deep copy Phase 2 to preserve all metadata
    merged = copy.deepcopy(phase2_json)

    # Bullet sections to merge
    bullet_sections = [
        "major_developments",
        "financial_performance",
        "risk_factors",
        "wall_street_sentiment",
        "competitive_industry_dynamics",
        "upcoming_catalysts",
        "key_variables"
    ]

    # Overlay Phase 3 integrated content onto Phase 2 bullets using bullet_id
    for section_name in bullet_sections:
        if section_name not in merged.get("sections", {}):
            continue

        # Build lookup by bullet_id from Phase 3
        phase3_bullets = phase3_json.get("sections", {}).get(section_name, [])
        phase3_map = {b['bullet_id']: b for b in phase3_bullets}

        # Overlay integrated content onto Phase 2 bullets
        phase2_bullets = merged["sections"][section_name]
        for bullet in phase2_bullets:
            bullet_id = bullet['bullet_id']
            if bullet_id in phase3_map:
                # Add integrated content as new field (preserve Phase 1 content and Phase 2 context for Quality Review)
                bullet['content_integrated'] = phase3_map[bullet_id]['content']
                # Don't delete content or context - needed for Quality Review verification

    # Scenarios (bottom_line, upside_scenario, downside_scenario)
    for section_name in ["bottom_line", "upside_scenario", "downside_scenario"]:
        if section_name in merged.get("sections", {}):
            phase3_section = phase3_json.get("sections", {}).get(section_name, {})
            if phase3_section and phase3_section.get("content"):
                # Add integrated content as new field (preserve Phase 1 content and Phase 2 context for Quality Review)
                merged["sections"][section_name]["content_integrated"] = phase3_section["content"]
                # Don't delete content or context - needed for Quality Review verification

    return merged
