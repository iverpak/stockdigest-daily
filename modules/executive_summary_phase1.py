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


def convert_phase1_to_sections_dict(phase1_json: Dict) -> Dict[str, List[str]]:
    """
    Convert Phase 1 JSON to format expected by build_executive_summary_html().
    Output IDENTICAL to what parse_executive_summary_sections() returns.

    For Email #3 (user-facing format with Phase 2 enrichments if present).

    Format:
    - topic (impact, sentiment, reason): content
    - Context: prose paragraph

    Args:
        phase1_json: Phase 1 JSON output (may include Phase 2 enrichments)

    Returns:
        sections dict matching current template format
    """
    # Helper function to filter bullets for Email #3
    def should_include_in_email3(bullet: Dict, section_name: str) -> bool:
        """
        Email #3 filtering:
        - competitive_industry_dynamics: Remove bullets with relevance = "none"
        - All other sections: Keep all bullets
        """
        if section_name == "competitive_industry_dynamics":
            return bullet.get('relevance') != 'none'
        else:
            return True

    # Helper function to format bullets for Email #3 (user-facing)
    def format_bullet_for_email3(bullet: Dict) -> str:
        """Format bullet with (impact, sentiment, reason) and Context for user-facing email"""
        topic = bullet['topic_label']

        # Add (impact, sentiment, reason) if Phase 2 enriched
        if bullet.get('impact'):
            topic += f" ({bullet['impact']}, {bullet['sentiment']}, {bullet['reason']})"

        main_line = f"{topic}: {bullet['content']}"

        # Add Context as prose paragraph on separate line
        if bullet.get('context'):
            main_line += f"\nContext: {bullet['context']}"

        return main_line

    sections = {
        "bottom_line": [],
        "major_developments": [],
        "financial_operational": [],  # Note: different key name than JSON
        "risk_factors": [],
        "wall_street": [],  # Note: different key name than JSON
        "competitive_industry": [],  # Note: different key name than JSON
        "upcoming_catalysts": [],
        "upside_scenario": [],
        "downside_scenario": [],
        "key_variables": []
    }

    json_sections = phase1_json.get("sections", {})

    # Bottom Line (paragraph with Phase 2 context)
    if "bottom_line" in json_sections:
        content = json_sections["bottom_line"].get("content", "")
        context = json_sections["bottom_line"].get("context", "")

        if context:
            sections["bottom_line"] = [f"{content}\nContext: {context}"]
        else:
            sections["bottom_line"] = [content]

    # Major Developments (bullets)
    if "major_developments" in json_sections:
        sections["major_developments"] = [
            format_bullet_for_email3(b) for b in json_sections["major_developments"]
        ]

    # Financial Performance → financial_operational (template expects this key)
    if "financial_performance" in json_sections:
        sections["financial_operational"] = [
            format_bullet_for_email3(b) for b in json_sections["financial_performance"]
        ]

    # Risk Factors
    if "risk_factors" in json_sections:
        sections["risk_factors"] = [
            format_bullet_for_email3(b) for b in json_sections["risk_factors"]
        ]

    # Wall Street Sentiment → wall_street (template key)
    if "wall_street_sentiment" in json_sections:
        sections["wall_street"] = [
            format_bullet_for_email3(b) for b in json_sections["wall_street_sentiment"]
        ]

    # Competitive/Industry → competitive_industry (template key) - WITH FILTER
    if "competitive_industry_dynamics" in json_sections:
        sections["competitive_industry"] = [
            format_bullet_for_email3(b)
            for b in json_sections["competitive_industry_dynamics"]
            if should_include_in_email3(b, "competitive_industry_dynamics")
        ]

    # Upcoming Catalysts
    if "upcoming_catalysts" in json_sections:
        sections["upcoming_catalysts"] = [
            format_bullet_for_email3(b) for b in json_sections["upcoming_catalysts"]
        ]

    # Upside Scenario (paragraph with Phase 2 context)
    if "upside_scenario" in json_sections:
        content = json_sections["upside_scenario"].get("content", "")
        context = json_sections["upside_scenario"].get("context", "")

        if context:
            sections["upside_scenario"] = [f"{content}\nContext: {context}"]
        else:
            sections["upside_scenario"] = [content]

    # Downside Scenario (paragraph with Phase 2 context)
    if "downside_scenario" in json_sections:
        content = json_sections["downside_scenario"].get("content", "")
        context = json_sections["downside_scenario"].get("context", "")

        if context:
            sections["downside_scenario"] = [f"{content}\nContext: {context}"]
        else:
            sections["downside_scenario"] = [content]

    # Key Variables
    if "key_variables" in json_sections:
        sections["key_variables"] = [
            format_bullet_for_email3(b) for b in json_sections["key_variables"]
        ]

    return sections


def convert_phase1_to_enhanced_sections(phase1_json: Dict) -> Dict[str, List[str]]:
    """
    Convert Phase 1 JSON with FULL structure for Email #2 QA.

    Shows:
    - Topic Label: Content
    - 📁 Filing hints: 10-K (Section A, Section B), 10-Q (Section C)
    - 🔖 ID: {bullet_id}

    Args:
        phase1_json: Phase 1 JSON output

    Returns:
        sections dict with enhanced formatting for Email #2
    """
    # Use same section structure as simple converter
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

    # Bottom Line (with Phase 2 context if present)
    if "bottom_line" in json_sections:
        content = json_sections["bottom_line"].get("content", "")
        context = json_sections["bottom_line"].get("context", "")

        if context:
            sections["bottom_line"] = [f"{content}<br><br>Context: {context}"]
        else:
            sections["bottom_line"] = [content]

    # Helper function to format bullets with filing hints and Phase 2 metadata
    def format_bullet_with_hints(bullet: Dict) -> str:
        """Format bullet with topic label, content, Phase 2 metadata, filing hints, and ID"""
        main_text = f"{bullet['topic_label']}: {bullet['content']}"

        # Phase 2 enrichments (if present)
        if bullet.get('impact'):
            main_text += f"<br>  Impact: {bullet['impact']} | Sentiment: {bullet['sentiment']} | Reason: {bullet['reason']} | Relevance: {bullet.get('relevance', 'direct')}"

        if bullet.get('context'):
            main_text += f"<br>  Context: {bullet['context']}"

        # Phase 1 filing hints
        hints = bullet.get("filing_hints", {})
        hint_parts = []
        for filing_type, sections_list in hints.items():
            if sections_list:
                hint_parts.append(f"{filing_type} ({', '.join(sections_list)})")

        if hint_parts:
            hints_text = "; ".join(hint_parts)
            # Use <br> tags for HTML rendering (newlines collapse in HTML)
            main_text += f"<br>  Filing hints: {hints_text}"

        # Phase 1 filing keywords (if present)
        keywords = bullet.get("filing_keywords", [])
        if keywords:
            # Format as JSON array for readability
            import json
            keywords_text = json.dumps(keywords)
            main_text += f"<br>  Filing keywords: {keywords_text}"

        # Bullet ID (always show)
        main_text += f"<br>  ID: {bullet['bullet_id']}"

        return main_text

    # Major Developments
    if "major_developments" in json_sections:
        sections["major_developments"] = [
            format_bullet_with_hints(b) for b in json_sections["major_developments"]
        ]

    # Financial Performance → financial_operational
    if "financial_performance" in json_sections:
        sections["financial_operational"] = [
            format_bullet_with_hints(b) for b in json_sections["financial_performance"]
        ]

    # Risk Factors
    if "risk_factors" in json_sections:
        sections["risk_factors"] = [
            format_bullet_with_hints(b) for b in json_sections["risk_factors"]
        ]

    # Wall Street Sentiment → wall_street
    if "wall_street_sentiment" in json_sections:
        sections["wall_street"] = [
            format_bullet_with_hints(b) for b in json_sections["wall_street_sentiment"]
        ]

    # Competitive/Industry → competitive_industry
    if "competitive_industry_dynamics" in json_sections:
        sections["competitive_industry"] = [
            format_bullet_with_hints(b) for b in json_sections["competitive_industry_dynamics"]
        ]

    # Upcoming Catalysts
    if "upcoming_catalysts" in json_sections:
        sections["upcoming_catalysts"] = [
            format_bullet_with_hints(b) for b in json_sections["upcoming_catalysts"]
        ]

    # Upside/Downside Scenarios (with Phase 2 context if present)
    if "upside_scenario" in json_sections:
        content = json_sections["upside_scenario"].get("content", "")
        context = json_sections["upside_scenario"].get("context", "")

        if context:
            sections["upside_scenario"] = [f"{content}<br><br>Context: {context}"]
        else:
            sections["upside_scenario"] = [content]

    if "downside_scenario" in json_sections:
        content = json_sections["downside_scenario"].get("content", "")
        context = json_sections["downside_scenario"].get("context", "")

        if context:
            sections["downside_scenario"] = [f"{content}<br><br>Context: {context}"]
        else:
            sections["downside_scenario"] = [content]

    # Key Variables (no filing hints, but show ID)
    if "key_variables" in json_sections:
        sections["key_variables"] = [
            f"{b['topic_label']}: {b['content']}<br>  🔖 ID: {b['bullet_id']}"
            for b in json_sections["key_variables"]
        ]

    return sections
