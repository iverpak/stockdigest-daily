"""
Executive Summary Phase 3 - Editorial Markdown Formatting

This module transforms Phase 1+2 merged JSON into scannable, professional editorial format.
Phase 3 is a FORMATTER, not an ANALYZER - just rearranges existing content.

Key functions:
- generate_executive_summary_phase3(): Main entry point (Claude API call)
- parse_phase3_markdown_to_sections(): Converts markdown to sections dict for Email #4
- save_editorial_summary(): Saves markdown to database
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
    merged_json: Dict,
    anthropic_api_key: str
) -> Optional[Dict]:
    """
    Generate Phase 3 editorial markdown from merged Phase 1+2 JSON.

    Args:
        ticker: Stock ticker
        merged_json: Complete merged JSON from Phase 2
        anthropic_api_key: Claude API key

    Returns:
        {
            "markdown": "## BOTTOM LINE\n\n...",
            "model_used": "claude-sonnet-4-5-20250929",
            "prompt_tokens": 15000,
            "completion_tokens": 3500,
            "generation_time_ms": 35000
        }
        Or None if failed
    """
    LOG.info(f"[{ticker}] Generating Phase 3 editorial markdown...")

    try:
        # 1. Load Phase 3 prompt from file
        prompt_path = os.path.join(os.path.dirname(__file__), '_build_executive_summary_prompt_phase3')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()

        # 2. Build user content (merged JSON as formatted string)
        user_content = json.dumps(merged_json, indent=2)

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
            markdown = result.get("content", [{}])[0].get("text", "")

            usage = result.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)

            LOG.info(f"[{ticker}] ✅ Phase 3 markdown generated ({len(markdown)} chars, "
                    f"{prompt_tokens} prompt tokens, {completion_tokens} completion tokens, {generation_time_ms}ms)")

            return {
                "markdown": markdown,
                "model_used": "claude-sonnet-4-5-20250929",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "generation_time_ms": generation_time_ms
            }
        else:
            error_text = response.text[:500]
            LOG.error(f"[{ticker}] Phase 3 API error {response.status_code}: {error_text}")
            return None

    except Exception as e:
        LOG.error(f"[{ticker}] Phase 3 generation failed: {e}", exc_info=True)
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
