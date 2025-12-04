"""
Shared utilities for executive summary email generation.

Functions:
- format_bullet_header(): Universal bullet header formatter
- add_dates_to_email_sections(): Add dates using bullet_id matching
- filter_bullets_for_email3(): Apply Email #3 filter to remove low-quality bullets
"""

import re
from typing import Dict, List, Optional


# Sections where sentiment tags are hidden in Email #3 (user-facing)
# These sections have self-evident sentiment or re-labeling adds confusion/risk
HIDE_SENTIMENT_SECTIONS = {
    'risk_factors',           # Risks are inherently negative; bullish/bearish is confusing
    'wall_street_sentiment',  # Analyst opinions ARE sentiment; redundant to re-label
    'upcoming_catalysts',     # Future events; sentiment doesn't apply cleanly
    'financial_performance',  # Results speak for themselves; "beat" = bullish is obvious
}


def format_bullet_header(bullet: Dict, show_reason: bool = True, section_name: str = None) -> str:
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
        section_name: If provided, used to determine whether to show sentiment.
                      Sections in HIDE_SENTIMENT_SECTIONS will not display sentiment tags.
                      Default None shows sentiment (backward compatible for Email #2).

    Returns:
        Formatted header string (bolded with markdown **)

    Used by: All bullet sections in Email #2, #3, #4
    """
    header = bullet['topic_label']

    # Add entity if present (competitive_industry_dynamics only)
    if bullet.get('entity'):
        header = f"[{bullet['entity']}] {header}"

    # Determine if sentiment should be shown for this section
    show_sentiment = section_name not in HIDE_SENTIMENT_SECTIONS if section_name else True

    # Add sentiment/reason if present and allowed for this section
    if show_sentiment and bullet.get('sentiment'):
        sentiment_cap = bullet['sentiment'].title()  # Bullish, Bearish, Neutral, Mixed
        if show_reason and bullet.get('reason'):
            header = f"{header} • {sentiment_cap} ({bullet['reason']})"
        else:
            header = f"{header} • {sentiment_cap}"

    return f"**{header}**"


def _insert_date_before_metadata(text: str, date: str) -> str:
    """
    Insert date (Nov 04) after deduplication block but before ID line.

    New format (Nov 2025):
    - Date goes AFTER: Deduplication block (✅ UNIQUE, 🔗 PRIMARY, ❌ DUPLICATE)
    - Date goes BEFORE: ID: bullet_id

    Args:
        text: Formatted bullet string
        date: Date string like "Nov 04" or "Nov 03-08"

    Returns:
        Text with date inserted in correct position
    """
    # Look for ID line at the end - insert date before it
    # Pattern: <br><br>ID: (with two line breaks before ID)
    id_pattern = r'(<br><br>ID:)'
    match = re.search(id_pattern, text)
    if match:
        pos = match.start()
        # Insert date between dedup and ID
        return f"{text[:pos]}<br>({date}){text[pos:]}"

    # Fallback: Look for single <br>ID: pattern
    id_pattern_single = r'(<br>ID:)'
    match = re.search(id_pattern_single, text)
    if match:
        pos = match.start()
        return f"{text[:pos]}<br>({date}){text[pos:]}"

    # No ID found, append date at end
    return f"{text} ({date})"


def add_dates_to_email_sections(
    sections: Dict[str, List[Dict]],
    merged_json: Dict
) -> Dict[str, List[Dict]]:
    """
    Add (date) at end of each bullet/paragraph using bullet_id matching.

    For Email #3, also appends context_suffix after date with em dash:
    <content> (Dec 04) — <em><context></em>

    Works for Email #2, #3, #4 sections dict format where each item is:
    {'bullet_id': '...', 'formatted': '...', 'context_suffix': '...' (optional)}

    Args:
        sections: Dict like {"major_developments": [{'bullet_id': '...', 'formatted': '...', 'context_suffix': '...'}, ...]}
        merged_json: Phase 1+2 (or Phase 2+3) JSON with date_range fields

    Returns:
        sections dict with dates and context_suffix appended to 'formatted' field
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

            # Append context_suffix after date (Email #3 format)
            # Format: <content> (date) — <em><context></em>
            context_suffix = formatted_item.pop('context_suffix', '')
            if context_suffix:
                formatted_item['formatted'] += f" — <em>{context_suffix}</em>"

    # Handle Bottom Line, Upside, Downside (paragraphs, not bullets - no bullet_id)
    for section_key, json_key in [
        ("bottom_line", "bottom_line"),
        ("upside_scenario", "upside_scenario"),
        ("downside_scenario", "downside_scenario")
    ]:
        if section_key in sections and sections[section_key]:
            item = sections[section_key][0]
            date_range = merged_json['sections'].get(json_key, {}).get('date_range', '')

            # Handle new dict format with context_suffix
            if isinstance(item, dict):
                formatted = item['formatted']
                if date_range:
                    formatted += f" ({date_range})"
                context_suffix = item.get('context_suffix', '')
                if context_suffix:
                    formatted += f" — <em>{context_suffix}</em>"
                sections[section_key][0] = formatted  # Flatten to string
            else:
                # Fallback for old string format (Email #2 or legacy)
                if date_range:
                    sections[section_key][0] = f"{item} ({date_range})"

    return sections


def _get_filter_status(bullet: Dict) -> tuple:
    """
    Determine filter status for a bullet.

    Returns:
        Tuple of (should_include: bool, filter_reason: str or None)
        - (True, None) = bullet should be included
        - (False, "relevance=none") = filtered due to no relevance
        - (False, "indirect + low impact") = filtered due to indirect relevance with low impact
    """
    # Get enrichment fields
    relevance = bullet.get('relevance', '').lower()
    impact = bullet.get('impact', '').lower()

    # Safety: Don't filter if fields are missing
    if not relevance or not impact:
        return (True, None)

    # Filter rule 1: relevance is "none"
    if relevance == 'none':
        return (False, "relevance=none")

    # Filter rule 2: relevance is "indirect" AND impact is "low impact"
    if relevance == 'indirect' and impact == 'low impact':
        return (False, "indirect + low impact")

    # Keep bullet
    return (True, None)


def mark_filtered_bullets(phase2_json: Dict) -> Dict:
    """
    Mark bullets with filter_status without removing them.

    Adds to each bullet:
    - filter_status: 'included' or 'filtered_out'
    - filter_reason: None or reason string (only if filtered_out)

    Args:
        phase2_json: Phase 1+2 merged JSON with enrichment metadata

    Returns:
        New JSON dict with all bullets marked (deep copy, original unchanged)

    Used by:
    - Email #2 display (shows all bullets with filter status indicator)
    """
    import copy

    # Deep copy to avoid mutating original
    marked_json = copy.deepcopy(phase2_json)

    # All bullet sections
    bullet_sections = [
        "major_developments",
        "financial_performance",
        "risk_factors",
        "wall_street_sentiment",
        "competitive_industry_dynamics",
        "upcoming_catalysts",
        "key_variables"
    ]

    sections = marked_json.get('sections', {})

    for section_name in bullet_sections:
        if section_name in sections:
            for bullet in sections[section_name]:
                should_include, filter_reason = _get_filter_status(bullet)
                if should_include:
                    bullet['filter_status'] = 'included'
                    bullet['filter_reason'] = None
                else:
                    bullet['filter_status'] = 'filtered_out'
                    bullet['filter_reason'] = filter_reason

    # Scenarios pass through unchanged (no filtering applied to paragraphs)
    # bottom_line, upside_scenario, downside_scenario are dicts, not arrays

    return marked_json


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
        New JSON dict with filtered bullets REMOVED (deep copy, original unchanged)

    Used by:
    - Phase 3 generation (filter before context integration)
    - Email #3 display (filter before showing to users)
    """
    import copy

    # Deep copy to avoid mutating original
    filtered_json = copy.deepcopy(phase2_json)

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
            # Filter bullets - keep only those that pass filter
            sections[section_name] = [
                b for b in sections[section_name]
                if _get_filter_status(b)[0]  # [0] is should_include
            ]

    # Scenarios pass through unchanged (no filtering)
    # bottom_line, upside_scenario, downside_scenario are dicts, not arrays

    return filtered_json


def merge_filtered_bullets_back(phase3_json: Dict, marked_phase2_json: Dict) -> Dict:
    """
    Merge filtered-out bullets back into Phase 3 JSON for Email #2 display.

    Phase 3 only processes included bullets, so filtered bullets are missing.
    This function adds them back with their filter_status preserved.

    Args:
        phase3_json: Phase 3 merged JSON (only has included bullets with content_integrated)
        marked_phase2_json: Phase 2 JSON with all bullets marked via mark_filtered_bullets()

    Returns:
        New JSON with all bullets - included ones have content_integrated,
        filtered ones have filter_status='filtered_out' and no content_integrated

    Used by:
    - generate_executive_summary_all_phases() after Phase 3 completes
    - Regenerate endpoint after Phase 3 completes
    """
    import copy

    # Deep copy Phase 3 to avoid mutating
    merged_json = copy.deepcopy(phase3_json)

    # All bullet sections
    bullet_sections = [
        "major_developments",
        "financial_performance",
        "risk_factors",
        "wall_street_sentiment",
        "competitive_industry_dynamics",
        "upcoming_catalysts",
        "key_variables"
    ]

    # Ensure 'sections' key exists in merged_json
    if 'sections' not in merged_json:
        merged_json['sections'] = {}

    merged_sections = merged_json['sections']
    marked_sections = marked_phase2_json.get('sections', {})

    for section_name in bullet_sections:
        if section_name not in marked_sections:
            continue

        # Build lookup of Phase 3 bullets by bullet_id
        phase3_bullets_by_id = {}
        if section_name in merged_sections:
            for b in merged_sections[section_name]:
                bullet_id = b.get('bullet_id')
                if bullet_id:
                    phase3_bullets_by_id[bullet_id] = b

        # Rebuild section with ALL bullets from marked_phase2_json
        # Use Phase 3 version if available (has content_integrated)
        # Otherwise use marked Phase 2 version (filtered_out, no content_integrated)
        new_bullets = []
        for marked_bullet in marked_sections[section_name]:
            bullet_id = marked_bullet.get('bullet_id')

            if bullet_id and bullet_id in phase3_bullets_by_id:
                # Use Phase 3 version (has content_integrated)
                phase3_bullet = phase3_bullets_by_id[bullet_id]
                # Preserve filter_status from marked version
                phase3_bullet['filter_status'] = marked_bullet.get('filter_status', 'included')
                phase3_bullet['filter_reason'] = marked_bullet.get('filter_reason')
                new_bullets.append(phase3_bullet)
            else:
                # Filtered bullet - use marked Phase 2 version
                # It already has filter_status='filtered_out' and filter_reason
                new_bullets.append(copy.deepcopy(marked_bullet))

        merged_sections[section_name] = new_bullets

    return merged_json
