# Phase 3 Deduplication Discussion

**Date:** November 29, 2025
**Status:** In Progress - Awaiting User Input on Open Questions

---

## Problem Statement

Bullets/themes are duplicated across sections in executive summaries, sometimes as much as 80% overlap. Common duplication patterns:
- Major Developments ↔ Risk Factors
- Wall Street Sentiment ↔ Major Developments
- Risk Factors ↔ Competitive/Industry Dynamics

The prompt has a hard time truly de-duplicating in Phase 1 despite extensive optimization. The report "screams AI generated" when 2-3 bullets cover the exact same theme with barely any difference in presentation.

---

## Why Duplication Persists in Phase 1

### Core Problem: Overlapping Section Definitions

Sections have genuine semantic overlap by design:

| Theme | Could Legitimately Go To... |
|-------|---------------------------|
| "Morgan Stanley downgraded citing margin pressure" | Wall Street Sentiment (analyst action) **OR** Risk Factors (forward-looking concern) |
| "FDA investigation opened" | Major Developments (company event) **OR** Risk Factors (regulatory risk) |
| "Competitor launched cheaper EV" | Competitive/Industry Dynamics (competitor action) **OR** Risk Factors (competitive threat) |
| "CFO flagged tariff headwinds in call" | Financial Performance (management commentary) **OR** Risk Factors (author-flagged concern) |

### Why Phase 1 Optimization Hit a Wall

Phase 1 prompt already has:
- Explicit "NO DUPLICATION" rule (line 1569)
- "Each theme appears in EXACTLY ONE bullet section" (line 1569)
- "First match wins" hierarchy (line 1575)
- "Default: pick one section and move on" (line 1597)

But the AI interprets **"theme"** differently:
- User means: "Intel supplier transition"
- AI thinks: "analyst opinion about Intel" ≠ "company strategic move with Intel"

These feel like **different themes** to the AI because they come from different article categories and use different framing.

---

## Proposed Solution: Phase 3 Deduplication with Consolidation

### Why Phase 3 is the Right Place

1. **Smaller input scope** - Phase 3 sees condensed output, not 40+ articles
2. **Theme-level visibility** - Themes are already synthesized into bullets, making semantic comparison feasible
3. **Editorial fit** - "Is this duplicated?" is an editorial question, fits Phase 3's role
4. **Doesn't pollute Phase 3 scope significantly** - Similar to how Phase 2 does integrated context, tags, etc.

### Design Philosophy

**Goal:** 1 bullet = 1 theme, matched to the closest section

**Two-step process:**
1. **Detect duplicates** - Flag bullets that share the same core theme across sections
2. **Rescue unique details** - Before discarding the duplicate, extract any unique information and propose a revised version of the surviving bullet that incorporates it

This is **consolidation**, not just deduplication. The 25% unique content doesn't get lost.

---

## Proposed Schema

### For the Surviving Bullet (Primary)

```json
{
  "bullet_id": "wall_street_2",
  "topic_label": "Intel supplier analysis",
  "content_integrated": "...(Phase 3 integrated content)...",
  "duplication": {
    "status": "primary",
    "absorbs": ["major_dev_3"],
    "shared_theme": "Intel supply chain transition",
    "proposed_revision": "TF Securities analyst Ming-Chi Kuo reported Apple may source modem chips from Intel starting 2026, representing a potential shift from Qualcomm's 90% supply share; management noted ongoing supplier diversification strategy in Q3 call (Nov 04)"
  }
}
```

### For the Duplicate Bullet (Removed)

```json
{
  "bullet_id": "major_dev_3",
  "topic_label": "Intel supplier rumors",
  "content_integrated": "...",
  "duplication": {
    "status": "duplicate",
    "absorbed_by": "wall_street_2",
    "shared_theme": "Intel supply chain transition",
    "unique_content_contributed": "management noted ongoing supplier diversification strategy in Q3 call"
  }
}
```

### For Unique Bullets (No Duplication)

```json
{
  "bullet_id": "financial_1",
  "topic_label": "Q3 revenue beat",
  "content_integrated": "...",
  "duplication": {
    "status": "unique"
  }
}
```

---

## Post-Processing Logic

```python
def consolidate_duplicates(phase3_json):
    for section_name, bullets in phase3_json['sections'].items():
        consolidated = []
        for bullet in bullets:
            dup = bullet.get('duplication', {})

            if dup.get('status') == 'duplicate':
                # Skip - this bullet is absorbed elsewhere
                continue

            if dup.get('status') == 'primary' and dup.get('proposed_revision'):
                # Use the revised content that includes absorbed info
                bullet['content_integrated'] = dup['proposed_revision']

            consolidated.append(bullet)

        phase3_json['sections'][section_name] = consolidated

    return phase3_json
```

---

## Hierarchy for Deciding Which Bullet Survives

### User's Rationale for Hierarchy

It's easy to distinguish whether something is analyst commentary or not. An analyst flagging a major development is still Wall Street Sentiment - they're the ones talking about it, they're the ones that flagged it. It's meaningful to show that it's not our interpretation that this is a major event, but the professionals are actually flagging it. 95% of the time you don't see something breaking coming from analysts first - they do their work from articles/filings/PRs.

The hierarchy follows a list of item criteria that is easy to distinguish, from easiest to hardest.

### Proposed Decision Rule (Core Logic from Phase 1)

```
When the same theme appears in multiple sections, keep ONE bullet using this priority:

1. WHO is the primary source?
   - Named analyst/firm → Wall Street Sentiment
   - Company management → Major Developments or Financial Performance
   - Article author inference → Risk Factors or Competitive Dynamics

2. WHAT is the primary content?
   - Specific event/action → Major Developments
   - Quantified metrics → Financial Performance
   - Future dated event → Upcoming Catalysts
   - External entity focus → Competitive/Industry Dynamics
   - Forward-looking concern → Risk Factors

The bullet with the strongest "WHO" or "WHAT" match wins.
The duplicate's unique details get absorbed into the survivor.
```

---

## Email #2 Display (QA Purposes)

### Surviving Bullet
```
**[TF Securities] Intel Supplier Analysis • Bearish (supply chain risk)**
TF Securities analyst Ming-Chi Kuo reported Apple may source modem chips
from Intel starting 2026, representing a potential shift from Qualcomm's
90% supply share; management noted ongoing supplier diversification
strategy in Q3 call (Nov 04)
  Filing hints: 10-K (Material Dependencies)
  ID: wall_street_2
  Duplication: ✅ Primary | Absorbed: major_dev_3 | Theme: Intel supply chain transition
```

### Removed Bullet (Shown in Email #2 for QA, Hidden in Email #3)
```
**Intel Supplier Rumors**
Ming-Chi Kuo reported Apple may source...
  ID: major_dev_3
  Duplication: ❌ Removed | Absorbed by: wall_street_2
  Unique content contributed: "management noted ongoing supplier diversification strategy"
```

---

## Open Questions (Awaiting User Input)

### 1. Proposed Revision Scope

Should Phase 3 rewrite the ENTIRE bullet with all context integrated, or just append the unique content?

| Option | Pros | Cons |
|--------|------|------|
| Full rewrite | Cleaner, more natural flow | More AI variance, potential drift |
| Append unique content | Safer, predictable | Potentially awkward phrasing |

### 2. Overlap Threshold

At what point is it "the same theme"?

- Two bullets about "Intel" but one is supply chain, one is Intel's earnings → NOT duplicates
- Two bullets both about "Apple sourcing from Intel" → duplicates

Should we leave this to AI judgment, or try to define it more precisely?

### 3. Cross-Section Only, or Same-Section Too?

Can two bullets in the SAME section (e.g., two Risk Factors bullets) be duplicates of each other?

Or is duplication only a problem when it crosses sections (Wall Street vs Major Dev vs Risk)?

### 4. What If There Are 3+ Duplicates?

If A in Major Dev, B in Wall Street, C in Risk all cover same theme:
- Does the primary absorb both B and C?
- Or do we pick primary first, then check if others are still duplicates of each other?

Proposed: Primary absorbs ALL duplicates of the same theme. `absorbs: ["major_dev_3", "risk_2"]`

---

## Next Steps

1. User to answer open questions above
2. Draft Phase 3 prompt addition
3. Update `executive_summary_phase3.py` with new schema
4. Update post-processing in `consolidate_duplicates()`
5. Update Email #2 display to show duplication metadata
6. Test with real reports that have known duplication issues

---

## Files to Modify

- `modules/_build_executive_summary_prompt_phase3_new` - Add deduplication instructions
- `modules/executive_summary_phase3.py` - Update JSON parsing for new schema
- `modules/executive_summary_phase1.py` - Update `convert_phase1_to_enhanced_sections()` for Email #2 display
- `app.py` - Add `consolidate_duplicates()` post-processing step after Phase 3
