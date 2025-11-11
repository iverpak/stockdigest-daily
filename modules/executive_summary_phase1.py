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

        for article in all_flagged_articles[:50]:  # Limit to 50 most recent
            title = article.get("title", "")
            ai_summary = article.get("ai_summary", "")
            domain = article.get("domain", "")
            published_at = article.get("published_at")

            # Format date
            if published_at:
                # Use simple format like "Oct 22"
                date_str = published_at.strftime("%b %d")
            else:
                date_str = "Unknown date"

            category_tag = article.get("_category_tag", "[UNKNOWN]")

            if ai_summary:
                # Get domain formal name (simplified - just use domain if lookup not available)
                source_name = domain if domain else "Unknown Source"
                unified_timeline.append(f"• {category_tag} {title} [{source_name}] {date_str}: {ai_summary}")

    # Calculate report context (start date, end date, day of week)
    end_date = datetime.now().strftime("%B %d, %Y")
    day_of_week = datetime.now().strftime("%A")

    # Calculate start date from oldest flagged article (or default to 7 days ago)
    if all_flagged_articles:
        oldest_article = min(all_flagged_articles, key=lambda x: x.get("published_at") or datetime.max.replace(tzinfo=timezone.utc))
        start_date = oldest_article.get("published_at").strftime("%B %d, %Y") if oldest_article.get("published_at") else end_date
    else:
        start_date = (datetime.now() - timedelta(days=7)).strftime("%B %d, %Y")

    # Build user_content
    # CRITICAL: Add ticker context here (not in system prompt) for prompt caching optimization
    ticker_header = f"TARGET COMPANY: {ticker} ({company_name})\n\n"

    if not all_flagged_articles:
        user_content = (
            ticker_header +
            f"REPORT CONTEXT:\n"
            f"Report type: {day_of_week}\n"
            f"Coverage period: {start_date} to {end_date}\n\n"
            f"---\n\n"
            f"FLAGGED ARTICLE COUNT: 0\n\n"
            f"NO FLAGGED ARTICLES - Generate quiet day summary per template."
        )
    else:
        article_count = len(all_flagged_articles)
        user_content = (
            ticker_header +
            f"REPORT CONTEXT:\n"
            f"Report type: {day_of_week}\n"
            f"Coverage period: {start_date} to {end_date}\n\n"
            f"---\n\n"
            f"FLAGGED ARTICLE COUNT: {article_count}\n\n"
            f"UNIFIED ARTICLE TIMELINE (newest to oldest):\n"
            + "\n".join(unified_timeline)
        )

    return user_content


def generate_executive_summary_phase1(
    ticker: str,
    categories: Dict[str, List[Dict]],
    config: Dict,
    anthropic_api_key: str
) -> Optional[Dict]:
    """
    Generate Phase 1 executive summary (articles only, NO filings).

    Args:
        ticker: Stock ticker (e.g., "AAPL", "RY.TO")
        categories: Dict with keys: company, industry, competitor
                   Each contains list of article dicts
        config: Ticker config dict (contains company_name, etc.)
        anthropic_api_key: Claude API key

    Returns:
        {
            "json_output": {...},  # Full Phase 1 JSON structure
            "model_used": "claude",
            "prompt_tokens": 28500,
            "completion_tokens": 3500,
            "generation_time_ms": 45000
        }
        Or None if generation fails
    """
    company_name = config.get("name", ticker)
    start_time = time.time()

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

            # Extract JSON from response
            content = result.get("content", [{}])[0].get("text", "")

            # Claude may wrap JSON in ```json ... ``` - extract it
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content

            # Parse JSON
            try:
                json_output = json.loads(json_str)
            except json.JSONDecodeError as e:
                LOG.error(f"[{ticker}] Failed to parse Phase 1 JSON: {e}")
                LOG.error(f"[{ticker}] Raw response (first 1000 chars): {content[:1000]}")
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

            LOG.info(f"✅ [{ticker}] Phase 1 generated JSON ({len(json_str)} chars, {prompt_tokens} prompt tokens, {completion_tokens} completion tokens, {generation_time_ms}ms)")

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


def validate_phase1_json(json_output: Dict) -> Tuple[bool, str]:
    """
    Validate Phase 1 JSON matches expected schema.

    Checks:
    - "sections" key exists
    - All 10 required sections present
    - Bullet sections have correct structure (bullet_id, topic_label, content, filing_hints)
    - Bottom line has content + word_count
    - Scenarios have content

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

        # Validate scenarios
        for scenario_name in ["upside_scenario", "downside_scenario"]:
            scenario = sections[scenario_name]
            if not isinstance(scenario, dict):
                return False, f"{scenario_name} must be object"
            if "content" not in scenario:
                return False, f"{scenario_name} missing 'content'"

        return True, ""

    except Exception as e:
        return False, f"Validation exception: {str(e)}"


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
        content = json_sections["bottom_line"].get("content", "")
        context = json_sections["bottom_line"].get("context", "")
        if context:
            content += f" Context: {context}"
        sections["bottom_line"] = [content]

    # Helper function to filter bullets for Email #3
    def should_include_in_email3(bullet: Dict, section_name: str) -> bool:
        """
        Email #3 filtering:
        - competitive_industry_dynamics: Remove bullets with relevance = "none"
        - All other sections: Keep all bullets
        """
        if section_name == "competitive_industry_dynamics":
            return bullet.get('relevance') != 'none'
        return True

    # Helper function to format bullets (simple, no metadata)
    def format_bullet_simple(bullet: Dict) -> Dict:
        """Format bullet with header and content. Returns {'bullet_id': '...', 'formatted': '...'}"""
        # Use shared utility for header
        header = format_bullet_header(bullet)

        # Content (Phase 1 content)
        content = bullet['content']

        # Add Phase 2 context if present (regex will bold "Context:" label)
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
            # Apply filter for competitive_industry_dynamics
            if json_key == "competitive_industry_dynamics":
                filtered_bullets = [
                    b for b in json_sections[json_key]
                    if should_include_in_email3(b, json_key)
                ]
            else:
                filtered_bullets = json_sections[json_key]

            sections[sections_key] = [
                format_bullet_simple(b)
                for b in filtered_bullets
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
