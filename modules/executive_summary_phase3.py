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


def add_date_ranges_to_phase3_markdown(
    markdown: str,
    merged_json: Dict
) -> str:
    """
    Add date ranges from JSON to Phase 3 markdown.

    DOES NOT strip inline dates - just adds date_range from JSON.
    This creates temporary duplication for validation purposes.

    Args:
        markdown: Phase 3 markdown (has inline dates)
        merged_json: Phase 1+2 merged JSON (has date_range fields)

    Returns:
        Markdown with date ranges added at specified locations
    """

    # Step 1: Add date ranges for regular bullet sections
    for section_name in ['major_developments', 'financial_performance', 'risk_factors',
                          'wall_street_sentiment', 'competitive_industry_dynamics',
                          'upcoming_catalysts']:

        bullets = merged_json['sections'].get(section_name, [])

        for bullet in bullets:
            topic_label = bullet.get('topic_label', '')
            date_range = bullet.get('date_range', '')

            if not topic_label or not date_range:
                continue

            # Find this bullet's paragraph and add date at end
            # Pattern: Match from bullet header to end of integrated paragraph
            # Paragraph ends at: next bullet header, section header, or end of string

            # Escape special regex characters in topic_label
            escaped_topic = re.escape(topic_label)

            # Pattern matches:
            # - Topic Label • Sentiment (reason)
            # - [Entity] Topic Label • Sentiment (reason) (for competitive section)
            # Followed by blank line and paragraph content
            pattern = rf'(?:\[.+?\] )?{escaped_topic} • .+?\n\n(.+?)(?=\n\n(?:[A-Z]|\[|##|▸)|$)'

            def add_date(match):
                paragraph = match.group(1)
                # Add date at end of paragraph (before next section/bullet)
                return match.group(0)[:-len(paragraph)] + paragraph + f' ({date_range})'

            markdown = re.sub(pattern, add_date, markdown, flags=re.DOTALL, count=1)

    # Step 2: Add date range after "Key Developments:"
    bottom_line_date = merged_json['sections'].get('bottom_line', {}).get('date_range', '')
    if bottom_line_date:
        markdown = re.sub(
            r'(Key Developments:)(\n)',
            rf'\1 ({bottom_line_date})\2',
            markdown
        )

    # Step 3: Add date range after "Primary Drivers:"
    upside_date = merged_json['sections'].get('upside_scenario', {}).get('date_range', '')
    if upside_date:
        markdown = re.sub(
            r'(Primary Drivers:)(\n)',
            rf'\1 ({upside_date})\2',
            markdown
        )

    # Step 4: Add date range after "Primary Risks:"
    downside_date = merged_json['sections'].get('downside_scenario', {}).get('date_range', '')
    if downside_date:
        markdown = re.sub(
            r'(Primary Risks:)(\n)',
            rf'\1 ({downside_date})\2',
            markdown
        )

    return markdown


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


def parse_phase3_markdown_to_sections(markdown: str) -> Dict[str, List[str]]:
    """
    Parse Phase 3 markdown back to sections dict format.

    Output format matches parse_executive_summary_sections() from app.py.

    Converts:
    ## BOTTOM LINE
    [Thesis]
    Key Developments:
    - **Theme 1**: [content]

    To:
    {
      "bottom_line": ["[Thesis]\n\nKey Developments:\n- **Theme 1**: [content]"],
      "major_developments": ["**Topic** • Bullish (reason)\n\n[integrated paragraph]"],
      ...
    }

    Args:
        markdown: Phase 3 markdown output

    Returns:
        sections dict compatible with Email #3 HTML builder
    """
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

    # Section name mapping (markdown headers → dict keys)
    section_map = {
        "BOTTOM LINE": "bottom_line",
        "MAJOR DEVELOPMENTS": "major_developments",
        "FINANCIAL/OPERATIONAL PERFORMANCE": "financial_operational",
        "RISK FACTORS": "risk_factors",
        "WALL STREET SENTIMENT": "wall_street",
        "COMPETITIVE/INDUSTRY DYNAMICS": "competitive_industry",
        "UPCOMING CATALYSTS": "upcoming_catalysts",
        "UPSIDE SCENARIO": "upside_scenario",
        "DOWNSIDE SCENARIO": "downside_scenario",
        "KEY VARIABLES TO MONITOR": "key_variables"
    }

    # Split by section headers (## SECTION NAME)
    section_pattern = r'^## (.+)$'
    lines = markdown.split('\n')

    current_section = None
    current_content = []

    for line in lines:
        match = re.match(section_pattern, line)
        if match:
            # Save previous section
            if current_section and current_content:
                _save_section_content(sections, section_map, current_section, current_content)

            # Start new section
            current_section = match.group(1).strip()
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_section and current_content:
        _save_section_content(sections, section_map, current_section, current_content)

    return sections


def _save_section_content(sections: Dict, section_map: Dict, section_name: str, content_lines: List[str]):
    """Helper to save parsed section content"""
    content = '\n'.join(content_lines).strip()

    if not content:
        return

    section_key = section_map.get(section_name)
    if not section_key:
        LOG.warning(f"Unknown section in Phase 3 markdown: {section_name}")
        return

    # Bottom Line and Scenarios: Store as single string
    if section_key in ["bottom_line", "upside_scenario", "downside_scenario"]:
        sections[section_key] = [content]

    # Key Variables: Parse by ▸ marker
    elif section_key == "key_variables":
        sections[section_key] = _parse_variable_section(content)

    # Bullet sections: Parse by **Topic** • Sentiment pattern
    else:
        sections[section_key] = _parse_bullet_section(content)


def _parse_bullet_section(content: str) -> List[str]:
    """Parse bullet section into list of bullet strings"""
    # Pattern: **Topic • Sentiment (reason)** OR **[Entity] Topic • Sentiment (reason)**
    # Matches entire bolded header (not just topic)
    bullet_pattern = r'^\*\*(\[.+?\] )?(.+?) • (.+?)\*\*$'

    bullets = []
    current_bullet = []

    for line in content.split('\n'):
        # Check if line starts a new bullet
        if re.match(bullet_pattern, line):
            # Save previous bullet
            if current_bullet:
                bullets.append('\n'.join(current_bullet))  # Don't strip - preserve formatting
            current_bullet = [line]
        else:
            # Continuation of current bullet
            current_bullet.append(line)  # Keep ALL lines including empty ones (preserves \n\n paragraph breaks)

    # Save last bullet
    if current_bullet:
        bullets.append('\n'.join(current_bullet))  # Don't strip - preserve formatting

    return bullets


def _parse_variable_section(content: str) -> List[str]:
    """Parse Key Variables section by ▸ marker"""
    variables = []
    for line in content.split('\n'):
        if line.strip().startswith('▸'):
            # Remove ▸ marker and trim
            variables.append(line.strip()[2:].strip())
    return variables


def save_editorial_summary(
    ticker: str,
    summary_date: date,
    editorial_markdown: str,
    metadata: Dict
) -> bool:
    """
    Save Phase 3 editorial markdown to database.

    Updates executive_summaries table with editorial_markdown column.

    Args:
        ticker: Stock ticker
        summary_date: Date of summary
        editorial_markdown: Phase 3 markdown output
        metadata: Generation metadata (tokens, time, etc.)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Import db from app.py
        from app import db

        with db() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE executive_summaries
                SET editorial_markdown = %s
                WHERE ticker = %s AND summary_date = %s
            """, (editorial_markdown, ticker, summary_date))

            conn.commit()

            LOG.info(f"[{ticker}] ✅ Editorial markdown saved to database "
                    f"({metadata.get('prompt_tokens', 0)} prompt tokens, "
                    f"{metadata.get('completion_tokens', 0)} completion tokens)")

            return True

    except Exception as e:
        LOG.error(f"[{ticker}] Failed to save editorial summary: {e}", exc_info=True)
        return False
