"""
Shared utilities for executive summary email generation.

Functions:
- format_bullet_header(): Universal bullet header formatter
- add_dates_to_email_sections(): Add dates using bullet_id matching
"""

import re
from typing import Dict, List, Optional


def format_bullet_header(bullet: Dict) -> str:
    """
    Universal bullet formatter - adapts based on available fields.

    Formats:
    - With entity + sentiment: **[Market] Topic • Bullish (supply constraint)**
    - With sentiment only: **Topic • Bullish (supply constraint)**
    - Without sentiment: **Topic**

    Args:
        bullet: Bullet dict with topic_label, and optionally entity/sentiment/reason

    Returns:
        Formatted header string (bolded with markdown **)

    Used by: All bullet sections in Email #2, #3, #4
    """
    header = bullet['topic_label']

    # Add entity if present (competitive_industry_dynamics only)
    if bullet.get('entity'):
        header = f"[{bullet['entity']}] {header}"

    # Add sentiment/reason if present (all sections except Key Variables/Catalysts)
    if bullet.get('sentiment') and bullet.get('reason'):
        sentiment_cap = bullet['sentiment'].title()  # Bullish, Bearish, Neutral, Mixed
        header = f"{header} • {sentiment_cap} ({bullet['reason']})"

    return f"**{header}**"


def _insert_date_before_metadata(text: str, date: str) -> str:
    """
    Insert date (Nov 04) before metadata lines in formatted bullet.

    Metadata lines start with <br>  Filing hints:, <br>  ID:, etc.

    Args:
        text: Formatted bullet string
        date: Date string like "Nov 04" or "Nov 03-08"

    Returns:
        Text with date inserted before metadata
    """
    # Find where metadata starts
    metadata_patterns = [
        r'(<br>\s*Filing hints:)',
        r'(<br>\s*Filing keywords:)',
        r'(<br>\s*ID:)',
        r'(<br>\s*Impact:)'
    ]

    for pattern in metadata_patterns:
        match = re.search(pattern, text)
        if match:
            # Insert date before metadata
            pos = match.start()
            return f"{text[:pos]} ({date}){text[pos:]}"

    # No metadata found, append date at end
    return f"{text} ({date})"


def add_dates_to_email_sections(
    sections: Dict[str, List[Dict]],
    merged_json: Dict
) -> Dict[str, List[Dict]]:
    """
    Add (date) at end of each bullet/paragraph using bullet_id matching.

    Works for Email #2, #3, #4 sections dict format where each item is:
    {'bullet_id': '...', 'formatted': '...'}

    Args:
        sections: Dict like {"major_developments": [{'bullet_id': '...', 'formatted': '...'}, ...]}
        merged_json: Phase 1+2 (or Phase 2+3) JSON with date_range fields

    Returns:
        sections dict with dates appended to 'formatted' field
    """
    # Map section keys (sections dict uses different names than JSON)
    section_mapping = {
        "major_developments": "major_developments",
        "financial_operational": "financial_performance",
        "risk_factors": "risk_factors",
        "wall_street": "wall_street_sentiment",
        "competitive_industry": "competitive_industry_dynamics",
        "upcoming_catalysts": "upcoming_catalysts",
        "key_variables": "key_variables"
    }

    for section_key, json_key in section_mapping.items():
        if section_key not in sections:
            continue

        # Build lookup by bullet_id
        bullets_json = merged_json['sections'].get(json_key, [])
        bullet_map = {b['bullet_id']: b for b in bullets_json}

        # Match and add dates
        formatted_list = sections[section_key]
        for formatted_item in formatted_list:
            bullet_id = formatted_item['bullet_id']
            source = bullet_map.get(bullet_id)

            if source and source.get('date_range'):
                date = source['date_range']
                # Insert date before metadata (if Email #2) or at end (if Email #3)
                formatted_item['formatted'] = _insert_date_before_metadata(
                    formatted_item['formatted'],
                    date
                )

    # Handle Bottom Line, Upside, Downside (paragraphs, not bullets - no bullet_id)
    for section_key, json_key in [
        ("bottom_line", "bottom_line"),
        ("upside_scenario", "upside_scenario"),
        ("downside_scenario", "downside_scenario")
    ]:
        if section_key in sections and sections[section_key]:
            date_range = merged_json['sections'].get(json_key, {}).get('date_range', '')
            if date_range:
                # These are simple lists, not dicts with bullet_id
                sections[section_key][0] = f"{sections[section_key][0]} ({date_range})"

    return sections
