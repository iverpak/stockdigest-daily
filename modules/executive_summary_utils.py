"""
Shared utilities for executive summary email generation.

Functions:
- format_bullet_header(): Universal bullet header formatter
- add_dates_to_email_sections(): Add dates using bullet_id matching
- filter_bullets_for_email3(): Apply Email #3 filter to remove low-quality bullets
"""

import re
from typing import Dict, List, Optional


def format_bullet_header(bullet: Dict, show_reason: bool = True) -> str:
    """
    Universal bullet formatter - adapts based on available fields.

    Formats:
    - With entity + sentiment + reason: **[Market] Topic • Bullish (supply constraint)**
    - With entity + sentiment (no reason): **[Market] Topic • Bullish**
    - With sentiment only: **Topic • Bullish (supply constraint)**
    - Without sentiment: **Topic**

    Args:
        bullet: Bullet dict with topic_label, and optionally entity/sentiment/reason
        show_reason: If True, include reason in parentheses after sentiment.
                     If False, show only sentiment without reason.
                     Default True for backward compatibility (Email #2 shows all metadata).
                     Email #3 passes False to hide reason from user-facing emails.

    Returns:
        Formatted header string (bolded with markdown **)

    Used by: All bullet sections in Email #2, #3, #4
    """
    header = bullet['topic_label']

    # Add entity if present (competitive_industry_dynamics only)
    if bullet.get('entity'):
        header = f"[{bullet['entity']}] {header}"

    # Add sentiment/reason if present (all sections except Key Variables/Catalysts)
    if bullet.get('sentiment'):
        sentiment_cap = bullet['sentiment'].title()  # Bullish, Bearish, Neutral, Mixed
        if show_reason and bullet.get('reason'):
            header = f"{header} • {sentiment_cap} ({bullet['reason']})"
        else:
            header = f"{header} • {sentiment_cap}"

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


def filter_bullets_for_email3(phase2_json: Dict) -> Dict:
    """
    Apply Email #3 filter to Phase 2 JSON: Remove low-quality bullets.

    Filtering rules:
    - Remove bullets with relevance = 'none'
    - Remove bullets with relevance = 'indirect' AND impact = 'low impact'
    - Scenarios (bottom_line, upside_scenario, downside_scenario) pass through unchanged

    This function applies the same logic as should_include_in_email3() from
    executive_summary_phase1.py, but operates on full JSON structure.

    Args:
        phase2_json: Phase 1+2 merged JSON with enrichment metadata

    Returns:
        New JSON dict with filtered bullets (deep copy, original unchanged)

    Used by:
    - Phase 3 generation (filter before context integration)
    - Email #3 display (filter before showing to users)
    - Email #4 display (filter before showing to users)
    """
    import copy

    # Deep copy to avoid mutating original
    filtered_json = copy.deepcopy(phase2_json)

    def should_include_bullet(bullet: Dict) -> bool:
        """
        Returns True if bullet should be included in Email #3.

        Same logic as executive_summary_phase1.py:should_include_in_email3()
        """
        # Get enrichment fields
        relevance = bullet.get('relevance', '').lower()
        impact = bullet.get('impact', '').lower()

        # Safety: Don't filter if fields are missing
        if not relevance or not impact:
            return True

        # Filter rule 1: relevance is "none"
        if relevance == 'none':
            return False

        # Filter rule 2: relevance is "indirect" AND impact is "low impact"
        if relevance == 'indirect' and impact == 'low impact':
            return False

        # Keep bullet
        return True

    # Filter all bullet sections
    bullet_sections = [
        "major_developments",
        "financial_performance",
        "risk_factors",
        "wall_street_sentiment",
        "competitive_industry_dynamics",
        "upcoming_catalysts",
        "key_variables"
    ]

    sections = filtered_json.get('sections', {})

    for section_name in bullet_sections:
        if section_name in sections:
            # Filter bullets in place
            sections[section_name] = [
                b for b in sections[section_name]
                if should_include_bullet(b)
            ]

    # Scenarios pass through unchanged (no filtering)
    # bottom_line, upside_scenario, downside_scenario are dicts, not arrays

    return filtered_json
